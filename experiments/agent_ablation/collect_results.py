"""Сводные таблицы кампании абляции: качает CSV траекторий с HF и агрегирует.

Что считается. Строка таблицы для статьи — это (уравнение, режим абляции).
Главная метрика абляции — доля успешных траекторий (success rate), рядом —
итоговая L2RE. Обе усредняются по НЕЗАВИСИМО ОБУЧЕННЫМ агентам (разные сиды):
одиночный прогон по такой метрике шумит, и в rebuttal это прямо названо
ограничением, поэтому агрегатор всегда считает и разброс между агентами
(95% доверительный интервал + отношение худшего к лучшему).

Выход:
  <out>                 — по (уравнение, режим): суммарно и между агентами;
  <out>.by_agent.csv    — по (уравнение, режим, сид): по одному агенту в строке.

Источник: HF-датасет результатов, файлы
  <prefix>/<pde>/<mode>/<run_tag>/results/trajectory_metrics.csv
Прогоны одной ячейки (цепочка резюмов) склеиваются по run_tag; строки с
smoke_test=True выбрасываются. success rate считается от ЗАВЕРШЁННЫХ
траекторий (done=1 плюс done=-1): траектория, прерванная дедлайном сессии,
попадает в CSV с done=0 и в знаменатель не идёт — иначе режим выглядел бы тем
хуже, чем неудачнее лёг конец сессии.

Запуск (токен не нужен — датасет открытый):
    python experiments/agent_ablation/collect_results.py --prefix runs_kaggle_v6 \
        --out ablation_v6_final.csv
    python experiments/agent_ablation/collect_results.py --prefix runs_kaggle_v5 \
        --out ablation_v5_recomputed.csv --upload
"""
import argparse
import csv
import json
import math
import os
import statistics
import sys
import collections
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from experiments.agent_ablation.pde_registry import ABLATION_MODES, PDE_SPECS

AGG_FIELDS = [
    "pde_name", "title", "ablation",
    "n_agents", "n_trajectories", "n_success", "n_fail", "n_interrupted", "success_rate",
    "success_rate_mean_over_agents", "success_rate_ci95",
    "l2re_op_median", "l2re_op_best",
    "l2re_median", "l2re_best", "l2re_min_median", "l2re_min_median_success",
    "l2re_min_best", "l2re_last_success",
    "l2re_min_best_mean_over_agents", "l2re_min_best_ci95",
    "l2re_min_best_worst_to_best_ratio",
    "steps_median", "elapsed_s_median", "elapsed_s_per_step_median",
    "est_10_trajectories_h", "seeds", "run_tags",
]

AGENT_FIELDS = [
    "pde_name", "ablation", "seed", "n_runs", "n_trajectories", "n_success", "n_fail",
    "n_interrupted",
    "success_rate", "l2re_op_median", "l2re_op_best",
    "l2re_median", "l2re_best", "l2re_min_median",
    "l2re_min_median_success", "l2re_min_best", "l2re_last_success",
    "steps_median", "elapsed_s_median", "run_tags",
]


def as_float(row, key):
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan
    return value


def as_int(row, key):
    try:
        return int(float(row.get(key, "") or 0))
    except (TypeError, ValueError):
        return 0


def finite(values):
    return [v for v in values if isinstance(v, float) and math.isfinite(v)]


def median_or_nan(values):
    values = finite(values)
    return statistics.median(values) if values else math.nan


def mean_ci95(values):
    """Среднее и полуширина 95% интервала (нормальное приближение по SE)."""
    values = finite(values)
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], math.nan
    mean = statistics.mean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half


def fmt(value, digits=6):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return value


def run_params(repo, run_base):
    """params.json прогона: по нему отличаем прогоны с разным критерием успеха.

    Под одним префиксом могут лежать прогоны разных ревизий протокола — так
    вышло с runs_kaggle_v6, где первая попытка считалась по train loss, а
    вторая по L2RE против эталона (критерий статьи). Смешивать их в одну
    строку таблицы нельзя.
    """
    from huggingface_hub import hf_hub_download

    try:
        local = hf_hub_download(repo, repo_type="dataset",
                                filename=f"{run_base}/results/params.json")
        with open(local, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def read_rows(repo, path):
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo, repo_type="dataset", filename=path)
    with open(local, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def agent_stats(rows):
    """Метрики одного независимо обученного агента (одна ячейка, один сид).

    Семантика колонок — как в таблице кампании v5, чтобы новые числа ложились
    в тот же ряд: l2re_* считаются ТОЛЬКО по успешным траекториям (у неуспешной
    цепочка оборвалась, её L2RE не сопоставима), а l2re_min_* — по всем,
    включая провальные: это лучшая точка, до которой траектория вообще дошла.
    """
    l2re_min = [as_float(r, "l2re_min") for r in rows]
    steps = [as_float(r, "steps") for r in rows]
    elapsed = [as_float(r, "elapsed_s") for r in rows]
    success = [r for r in rows if as_int(r, "done") == 1]
    fails = [r for r in rows if as_int(r, "done") == -1]
    # done=0 — траектория, прерванная дедлайном сессии или сигналом: ни успех,
    # ни провал, в знаменатель success rate не идёт.
    interrupted = [r for r in rows if as_int(r, "done") == 0]
    terminal = len(success) + len(fails)
    success_l2re = finite([as_float(r, "l2re") for r in success])
    # Величина статьи и ответа ревьюерам — ошибка решения против эталона, то
    # есть tester.l2re; в CSV она лежит в колонке l2re_op. Колонка l2re — это
    # sqrt(l2re_op^2 + l2re_bnd^2), она включает ошибку на границе и в таблице 1
    # ей ничего не соответствует. По ней же считается критерий остановки,
    # поэтому и в отчёте должна стоять она, иначе success rate и L2RE в одной
    # строке измеряют разные вещи (на poissoninv это давало «успех» с медианой
    # 0.0309 при пороге 0.0306).
    success_l2re_op = finite([as_float(r, "l2re_op") for r in success])

    return {
        "n_trajectories": len(rows),
        "n_success": len(success),
        "n_fail": len(fails),
        "n_interrupted": len(interrupted),
        "success_rate": len(success) / terminal if terminal else math.nan,
        "l2re_op_median": median_or_nan(success_l2re_op),
        "l2re_op_best": min(success_l2re_op, default=math.nan),
        "l2re_median": median_or_nan(success_l2re),
        "l2re_best": min(success_l2re, default=math.nan),
        "l2re_min_median": median_or_nan(l2re_min),
        # То же по одним успешным: такой вариант считала таблица v5, когда
        # успехи в ячейке были. Держим обе колонки, чтобы числа кампаний
        # сравнивались однозначно.
        "l2re_min_median_success": median_or_nan(
            [as_float(r, "l2re_min") for r in success]),
        "l2re_min_best": min(finite(l2re_min), default=math.nan),
        "l2re_last_success": success_l2re[-1] if success_l2re else math.nan,
        "steps_median": median_or_nan(steps),
        "elapsed_s_median": median_or_nan(elapsed),
        "elapsed_s_mean": statistics.mean(finite(elapsed)) if finite(elapsed) else math.nan,
    }


# Заголовки колонок — как в ответе ревьюерам (комментарий авторов от 03.08.2026,
# вопрос 4 рецензента DV8H), чтобы расширенная таблица читалась как продолжение
# уже опубликованной, а не как другая таблица.
REBUTTAL_COLUMNS = [("none", "full"), ("no_per", "no PER"),
                    ("no_soft_watkins", "no soft-Watkins"),
                    ("no_trust_region", "no trust region")]


def render_rebuttal_table(agg_rows):
    """Таблица в форме ответа ревьюерам: на уравнение две строки — success rate
    (доля цепочек, достигших критерия остановки, с сырым счётом в скобках) и
    медианная L2RE. Медиана считается по успешным цепочкам, поэтому в ячейке
    без единого успеха стоит прочерк — ровно как в опубликованной таблице,
    где такие клетки оставлены пустыми.
    """
    by_cell = {(r["pde_name"], r["ablation"]): r for r in agg_rows}
    pdes = []
    for r in agg_rows:
        if r["pde_name"] not in pdes:
            pdes.append(r["pde_name"])

    header = ["PDE", "metric"] + [label for _, label in REBUTTAL_COLUMNS]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for pde in pdes:
        title = next((r["title"] or pde for r in agg_rows if r["pde_name"] == pde), pde)
        sr_cells, l2_cells = [], []
        for mode, _ in REBUTTAL_COLUMNS:
            row = by_cell.get((pde, mode))
            if row is None:
                sr_cells.append("—")
                l2_cells.append("—")
                continue
            terminal = int(row["n_success"]) + int(row["n_fail"])
            rate = float(row["success_rate"]) if row["success_rate"] != "" else math.nan
            sr_cells.append(f"{rate:.2f} ({row['n_success']}/{terminal})"
                            if terminal else "— (0/0)")
            l2_cells.append(row["l2re_op_median"] if row["l2re_op_median"] != ""
                            else "нет успешных цепочек")
        lines.append("| " + " | ".join([title, "success rate"] + sr_cells) + " |")
        lines.append("| " + " | ".join(["", "L2RE median"] + l2_cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="danil-e/rlpinn-ablation-runs")
    parser.add_argument("--prefix", default="runs_kaggle_v6")
    parser.add_argument("--out", default="ablation_final_metrics.csv")
    parser.add_argument("--pde", action="append", default=None,
                        help="Ограничить уравнениями (можно повторять).")
    parser.add_argument("--success-metric", default=None,
                        choices=["l2re", "rmse", "loss"],
                        help="Брать только прогоны с этим критерием успеха (читается "
                             "из results/params.json). Под одним префиксом могут лежать "
                             "прогоны разных ревизий протокола.")
    parser.add_argument("--keep-smoke", action="store_true",
                        help="Не выбрасывать строки smoke_test=True.")
    parser.add_argument("--upload", action="store_true",
                        help="Положить обе таблицы в тот же датасет (нужен HF_TOKEN).")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(args.hf_repo, repo_type="dataset")
    csv_paths = [f for f in files
                 if f.startswith(args.prefix + "/") and f.endswith("results/trajectory_metrics.csv")]
    print(f"Найдено {len(csv_paths)} trajectory_metrics.csv под {args.prefix}/")
    if not csv_paths:
        sys.exit("Данных нет — кампания ещё не отработала?")

    # (pde, mode, seed) -> строки; сид берём из строки CSV, а не из run_tag
    by_agent = defaultdict(list)
    run_tags = defaultdict(set)
    skipped_metric = collections.Counter()
    for path in sorted(csv_paths):
        parts = path.split("/")  # prefix / pde / mode / run_tag / results / file
        if len(parts) < 6:
            continue
        pde, mode, run_tag = parts[1], parts[2], parts[3]
        if args.pde and pde not in args.pde:
            continue
        if args.success_metric:
            got = run_params(args.hf_repo, "/".join(parts[:4])).get("success_metric", "loss")
            if got != args.success_metric:
                skipped_metric[got] += 1
                continue
        try:
            rows = read_rows(args.hf_repo, path)
        except Exception as exc:
            print(f"⚠️  не прочитан {path}: {exc}")
            continue
        for row in rows:
            if not args.keep_smoke and str(row.get("smoke_test", "")).strip().lower() in ("true", "1"):
                continue
            seed = row.get("seed", "") or "?"
            by_agent[(pde, mode, seed)].append(row)
            run_tags[(pde, mode, seed)].add(run_tag)

    if skipped_metric:
        print("пропущено прогонов с другим критерием успеха: "
              + ", ".join(f"{k}={v}" for k, v in sorted(skipped_metric.items())))

    agent_rows, agg_rows = [], []
    pdes = sorted({k[0] for k in by_agent}, key=lambda p: list(PDE_SPECS).index(p) if p in PDE_SPECS else 99)
    for pde in pdes:
        for mode in ABLATION_MODES:
            seeds = sorted({k[2] for k in by_agent if k[0] == pde and k[1] == mode},
                           key=lambda s: (0, int(s)) if s.lstrip("-").isdigit() else (1, 0))
            if not seeds:
                continue
            per_agent = []
            all_rows = []
            for seed in seeds:
                rows = by_agent[(pde, mode, seed)]
                stats = agent_stats(rows)
                per_agent.append(stats)
                all_rows.extend(rows)
                agent_rows.append({
                    "pde_name": pde, "ablation": mode, "seed": seed,
                    "n_runs": len(run_tags[(pde, mode, seed)]),
                    "run_tags": " ".join(sorted(run_tags[(pde, mode, seed)])),
                    **{k: fmt(v) for k, v in stats.items() if k in AGENT_FIELDS},
                })

            pooled = agent_stats(all_rows)
            sr_mean, sr_ci = mean_ci95([a["success_rate"] for a in per_agent])
            l2re_mean, l2re_ci = mean_ci95([a["l2re_min_best"] for a in per_agent])
            bests = finite([a["l2re_min_best"] for a in per_agent])
            ratio = (max(bests) / min(bests)) if len(bests) > 1 and min(bests) > 0 else math.nan
            per_step = pooled["elapsed_s_median"] / pooled["steps_median"] \
                if pooled["steps_median"] and math.isfinite(pooled["steps_median"]) and pooled["steps_median"] > 0 \
                else math.nan

            agg_rows.append({
                "pde_name": pde,
                "title": PDE_SPECS[pde].title if pde in PDE_SPECS else "",
                "ablation": mode,
                "n_agents": len(seeds),
                "n_trajectories": pooled["n_trajectories"],
                "n_success": pooled["n_success"],
                "n_fail": pooled["n_fail"],
                "n_interrupted": pooled["n_interrupted"],
                "success_rate": fmt(pooled["success_rate"], 4),
                "success_rate_mean_over_agents": fmt(sr_mean, 4),
                "success_rate_ci95": fmt(sr_ci, 3),
                "l2re_op_median": fmt(pooled["l2re_op_median"]),
                "l2re_op_best": fmt(pooled["l2re_op_best"]),
                "l2re_median": fmt(pooled["l2re_median"]),
                "l2re_best": fmt(pooled["l2re_best"]),
                "l2re_min_median": fmt(pooled["l2re_min_median"]),
                "l2re_min_median_success": fmt(pooled["l2re_min_median_success"]),
                "l2re_min_best": fmt(pooled["l2re_min_best"]),
                "l2re_last_success": fmt(pooled["l2re_last_success"]),
                "l2re_min_best_mean_over_agents": fmt(l2re_mean),
                "l2re_min_best_ci95": fmt(l2re_ci, 3),
                "l2re_min_best_worst_to_best_ratio": fmt(ratio, 4),
                "steps_median": fmt(pooled["steps_median"], 4),
                "elapsed_s_median": fmt(pooled["elapsed_s_median"], 6),
                "elapsed_s_per_step_median": fmt(per_step, 6),
                "est_10_trajectories_h": fmt(pooled["elapsed_s_median"] * 10 / 3600, 4),
                "seeds": " ".join(seeds),
                "run_tags": " ".join(sorted(set().union(*(run_tags[(pde, mode, s)] for s in seeds)))),
            })
            print(f"[{pde:26s} {mode:16s}] агентов={len(seeds)} траекторий={pooled['n_trajectories']:3d} "
                  f"(прервано {pooled['n_interrupted']:2d}) успех={pooled['success_rate']:.2f} "
                  f"l2re_op_med={fmt(pooled['l2re_op_median'], 4)}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        writer.writeheader()
        writer.writerows(agg_rows)
    by_agent_out = os.path.splitext(args.out)[0] + ".by_agent.csv"
    with open(by_agent_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AGENT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(agent_rows)

    table_out = os.path.splitext(args.out)[0] + ".rebuttal.md"
    table = render_rebuttal_table(agg_rows)
    with open(table_out, "w", encoding="utf-8") as f:
        f.write("# Абляция компонентов DQN\n\n"
                "Форма таблицы — как в ответе ревьюерам (вопрос 4 рецензента DV8H).\n"
                "Один сид на конфигурацию, одинаковый бюджет PINN. success rate —\n"
                "доля построенных цепочек, достигших критерия остановки; число\n"
                "траекторий по ячейкам разное, поэтому сравнима именно доля.\n\n")
        f.write(table + "\n")
    print("\n" + table)
    print(f"\n✅ {args.out} ({len(agg_rows)} строк), {by_agent_out} ({len(agent_rows)} строк), "
          f"{table_out}")

    if args.upload:
        token = os.getenv("HF_TOKEN")
        if not token:
            sys.exit("--upload требует HF_TOKEN")
        api = HfApi(token=token)
        for path in (args.out, by_agent_out):
            api.upload_file(path_or_fileobj=path,
                            path_in_repo=f"{args.prefix}/{os.path.basename(path)}",
                            repo_id=args.hf_repo, repo_type="dataset")
            print(f"⬆️  {args.hf_repo}/{args.prefix}/{os.path.basename(path)}")


if __name__ == "__main__":
    main()
