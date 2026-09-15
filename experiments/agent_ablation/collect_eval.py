"""Сводка оценки обученных агентов как в статье (таблица 1): mean (std) L2RE.

Прогон оценки — замороженный агент (сид обучения agent_seed) строит цепочку
оптимизаторов, пока не исчерпан бюджет эпох PINN (в статье 7000, приложение E);
порог eps и Kmax цепочку не останавливают. seed в строке CSV — сид прогона:
точки коллокации, инициализация PINN, автоэнкодер. Сиды прогонов одинаковы у
всех агентов, поэтому режимы сравниваются на одних и тех же инициализациях.

Одна строка trajectory_metrics.csv под <prefix>/<pde>/<mode>/<tag>/results/ —
один прогон. Засчитываются только прогоны с results/eval_done.json: status.json
= finished бывает и у прогона, оборванного дедлайном сессии посреди цепочки.
Если сид оценки агента досчитан дважды, берётся более поздний тег.

Отчётная величина — L2RE в конце бюджета, l2re=sqrt(l2re_op^2+l2re_bnd^2).
Рядом лучшая точка цепочки (l2re_min, минимум той же величины по валидациям):
статья отмечает, что лучшая точка бывает не на последней стадии.

Файлы:
  <out>.csv          — по (уравнение, режим): mean/std/median, число агентов и прогонов
  <out>.by_agent.csv — по агенту: среднее по его сидам оценки (для парных сравнений)
  <out>.by_run.csv   — все засчитанные прогоны
  <out>.md           — таблица в форме таблицы 1 статьи, рядом PELINE из статьи

Запуск:
    python experiments/agent_ablation/collect_eval.py --prefix evals_kaggle_v10 --out eval_v10.csv
"""
import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from experiments.agent_ablation.pde_registry import PDE_SPECS  # noqa: E402

MODES = [("none", "full"), ("no_per", "no PER"),
         ("no_soft_watkins", "no soft-Watkins"), ("no_trust_region", "no trust region")]
L2RE_DEFINITION = "l2re=sqrt(l2re_op^2+l2re_bnd^2)"
# PELINE из таблицы 1 статьи. В реестре у poisson_boltzmann_2d база порога другая
# (медиана из ответа ревьюерам), а для сравнения с таблицей нужно её число.
TABLE1_OVERRIDE = {"poisson_boltzmann_2d": 4.11e-3}


def as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def finite(values):
    return [v for v in values if isinstance(v, float) and math.isfinite(v)]


def stats(values):
    vals = finite(values)
    if not vals:
        return math.nan, math.nan, math.nan
    std = statistics.stdev(vals) if len(vals) > 1 else math.nan
    return statistics.mean(vals), std, statistics.median(vals)


def sci(value):
    return "—" if not math.isfinite(value) else f"{value:.2E}"


def read_rows(repo, path):
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo, repo_type="dataset", filename=path)
    with open(local, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="danil-e/rlpinn-ablation-runs")
    parser.add_argument("--prefix", required=True, help="Префикс результатов оценки (например evals_kaggle_v10).")
    parser.add_argument("--out", default="eval_summary.csv")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(args.hf_repo, repo_type="dataset")
    fset = set(files)
    csv_paths = sorted(f for f in files
                       if f.startswith(args.prefix + "/") and f.endswith("results/trajectory_metrics.csv"))
    print(f"Найдено {len(csv_paths)} trajectory_metrics.csv под {args.prefix}/")

    latest, unfinished = {}, 0
    for path in csv_paths:
        base = path[: -len("results/trajectory_metrics.csv")]
        if base + "results/eval_done.json" not in fset:
            unfinished += 1
            continue
        _prefix, pde, mode, tag = path.split("/")[:4]
        for row in read_rows(args.hf_repo, path):
            if str(row.get("eval_budget_epochs", "")).strip() in ("", "0"):
                continue  # строка не оценки
            run = {
                "pde_name": pde, "ablation": mode, "run_tag": tag,
                "agent_seed": row.get("agent_seed", ""), "eval_seed": row.get("seed", ""),
                "l2re": as_float(row.get("l2re")), "l2re_op": as_float(row.get("l2re_op")),
                "l2re_bnd": as_float(row.get("l2re_bnd")), "l2re_min": as_float(row.get("l2re_min")),
                "epochs_used": row.get("epochs_used", ""),
                "steps": row.get("steps", ""), "done": row.get("done", ""),
                "elapsed_s": as_float(row.get("elapsed_s")), "chain_key": row.get("chain_key", ""),
            }
            key = (pde, mode, run["agent_seed"], run["eval_seed"])
            if key not in latest or tag > latest[key]["run_tag"]:
                latest[key] = run
    runs = sorted(latest.values(), key=lambda r: (r["pde_name"], r["ablation"], r["agent_seed"], r["eval_seed"]))
    print(f"Засчитано прогонов: {len(runs)}; без eval_done.json (не досчитаны): {unfinished}")
    if not runs:
        sys.exit("Досчитанных прогонов оценки нет.")

    by_agent = defaultdict(list)
    for run in runs:
        by_agent[(run["pde_name"], run["ablation"], run["agent_seed"])].append(run)
    agent_rows = []
    for (pde, mode, agent_seed), items in sorted(by_agent.items()):
        mean, std, median = stats([r["l2re"] for r in items])
        agent_rows.append({"pde_name": pde, "ablation": mode, "agent_seed": agent_seed,
                           "n_runs": len(items), "l2re_mean": mean, "l2re_std": std,
                           "l2re_min_mean": stats([r["l2re_min"] for r in items])[0],
                           "eval_seeds": " ".join(sorted(r["eval_seed"] for r in items))})

    cells = defaultdict(list)
    for run in runs:
        cells[(run["pde_name"], run["ablation"])].append(run)
    summary = []
    for (pde, mode), items in sorted(cells.items()):
        mean, std, median = stats([r["l2re"] for r in items])
        best_mean, best_std, _ = stats([r["l2re_min"] for r in items])
        spec = PDE_SPECS.get(pde)
        table1 = TABLE1_OVERRIDE.get(pde, spec.peline_l2re if spec else math.nan)
        summary.append({
            "pde_name": pde, "title": spec.title if spec else pde, "ablation": mode,
            "n_agents": len({r["agent_seed"] for r in items}), "n_runs": len(items),
            "l2re_mean": mean, "l2re_std": std, "l2re_median": median,
            "l2re_min_mean": best_mean, "l2re_min_std": best_std,
            "table1_peline_l2re": table1,
            "mean_over_table1": mean / table1 if table1 and math.isfinite(mean) else math.nan,
            "metric": f"L2RE в конце бюджета ({L2RE_DEFINITION})",
        })

    stem = os.path.splitext(args.out)[0]
    for path, rows in ((args.out, summary), (stem + ".by_agent.csv", agent_rows), (stem + ".by_run.csv", runs)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    by_cell = {(r["pde_name"], r["ablation"]): r for r in summary}
    pdes = sorted({r["pde_name"] for r in summary})
    header = ["PDE", "метрика"] + [label for _, label in MODES] + ["PELINE (табл. 1)"]
    lines = [
        "# Оценка обученных агентов как в статье (таблица 1)",
        "",
        "Замороженный агент строит цепочку, пока не исчерпан бюджет эпох PINN; "
        "mean (std) по всем прогонам: агенты (сиды обучения) × сиды оценки.",
        f"L2RE везде — {L2RE_DEFINITION}; «конец бюджета» — после последней эпохи, "
        "«лучшая точка» — минимум по валидациям цепочки.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for pde in pdes:
        title = next(r["title"] for r in summary if r["pde_name"] == pde)
        end_cells, best_cells, n_cells = [], [], []
        for mode, _ in MODES:
            row = by_cell.get((pde, mode))
            if row is None:
                end_cells.append("—"); best_cells.append("—"); n_cells.append("—")
                continue
            end_cells.append(f"{sci(row['l2re_mean'])} ({sci(row['l2re_std'])})")
            best_cells.append(f"{sci(row['l2re_min_mean'])} ({sci(row['l2re_min_std'])})")
            n_cells.append(f"{row['n_agents']} × {round(row['n_runs'] / max(row['n_agents'], 1), 1):g}")
        table1 = next(r["table1_peline_l2re"] for r in summary if r["pde_name"] == pde)
        lines.append("| " + " | ".join([title, "L2RE конец бюджета"] + end_cells + [sci(table1)]) + " |")
        lines.append("| " + " | ".join(["", "L2RE лучшая точка"] + best_cells + [""]) + " |")
        lines.append("| " + " | ".join(["", "агентов × сидов оценки"] + n_cells + [""]) + " |")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n✅ {args.out}, {stem}.by_agent.csv, {stem}.by_run.csv, {stem}.md")


if __name__ == "__main__":
    main()
