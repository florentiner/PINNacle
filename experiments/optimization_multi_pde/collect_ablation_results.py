"""Сводный CSV финальных метрик кампании абляции.

Качает с HF все trajectory_metrics.csv из <prefix>/<pde>/<mode>/<run_tag>/,
и для каждой пары (pde, ablation) выбирает итоговую траекторию:

  * последняя УСПЕШНАЯ (done == 1);
  * если успешных нет — последняя завершённая (с пометкой в final_source).

Строки исходного формата сохраняются целиком (run_timestamp..chain_json +
ошибка агента), добавляются сводные колонки: n_trajectories, n_success,
n_fail, best_l2re (минимальный l2re среди успешных), final_source, run_tag.

Запуск (токен не нужен — датасет открытый):
    python experiments/optimization_multi_pde/collect_ablation_results.py \\
        --out ablation_final_metrics.csv
Опционально: --prefix runs_kaggle --hf-repo danil-e/rlpinn-ablation-runs
             --upload  (положить итоговый CSV в тот же датасет, нужен HF_TOKEN)
"""
import argparse
import csv
import io
import math
import os
import sys
from collections import defaultdict

from huggingface_hub import HfApi, hf_hub_download

PDES = ["poisson_boltzmann_2d", "poisson3d_complexgeometry", "ns2d_liddriven"]
MODES = ["none", "no_per", "no_soft_watkins", "no_trust_region"]

SUMMARY_FIELDS = [
    "n_trajectories", "n_success", "n_fail",
    "best_l2re", "final_source", "run_tag",
]


def read_rows(repo, path):
    local = hf_hub_download(repo, repo_type="dataset", filename=path)
    with open(local, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_final(rows):
    """Последняя успешная траектория, иначе последняя завершённая."""
    def done_of(r):
        try:
            return int(float(r.get("done", "") or 0))
        except ValueError:
            return 0

    success = [r for r in rows if done_of(r) == 1]
    if success:
        return success[-1], "last_success"
    if rows:
        return rows[-1], "last_any_no_success"
    return None, "no_trajectories"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="danil-e/rlpinn-ablation-runs")
    parser.add_argument("--prefix", default="runs_kaggle")
    parser.add_argument("--out", default="ablation_final_metrics.csv")
    parser.add_argument("--upload", action="store_true",
                        help="Загрузить итоговый CSV в датасет (нужен HF_TOKEN).")
    args = parser.parse_args()

    api = HfApi()
    files = api.list_repo_files(args.hf_repo, repo_type="dataset")
    csv_paths = [
        f for f in files
        if f.startswith(args.prefix + "/") and f.endswith("results/trajectory_metrics.csv")
    ]
    print(f"Найдено {len(csv_paths)} trajectory_metrics.csv под {args.prefix}/")

    # (pde, mode) -> список (run_tag, rows); объединяем все запуски пары
    groups = defaultdict(list)
    for path in sorted(csv_paths):
        parts = path.split("/")  # prefix / pde / mode / run_tag / results / file
        if len(parts) < 6:
            continue
        _, pde, mode, run_tag = parts[0], parts[1], parts[2], parts[3]
        try:
            rows = read_rows(args.hf_repo, path)
        except Exception as exc:
            print(f"⚠️  не прочитан {path}: {exc}")
            continue
        groups[(pde, mode)].append((run_tag, rows))

    out_rows = []
    base_fields = None
    for pde in PDES:
        for mode in MODES:
            runs = groups.get((pde, mode), [])
            all_rows = []
            for run_tag, rows in sorted(runs):
                for r in rows:
                    # проверочные запуски (smoke_test=True) не участвуют в итогах
                    if str(r.get("smoke_test", "")).strip().lower() in ("true", "1"):
                        continue
                    r["_run_tag"] = run_tag
                    all_rows.append(r)
            # без сортировки между запусками: внутри запуска порядок хронологический,
            # запуски отсортированы по run_tag (таймстемп в начале тега)
            final, source = pick_final(all_rows)
            if final is None:
                print(f"[{pde:26s} {mode:16s}] траекторий нет")
                continue
            if base_fields is None:
                base_fields = [k for k in final.keys() if k != "_run_tag"]

            def l2re_of(r):
                try:
                    return float(r.get("l2re", "nan"))
                except ValueError:
                    return math.nan

            succ = [r for r in all_rows if r.get("done") not in ("", None) and float(r["done"]) == 1]
            best = min((l2re_of(r) for r in succ), default=math.nan)

            row = {k: final.get(k, "") for k in base_fields}
            row["n_trajectories"] = len(all_rows)
            row["n_success"] = len(succ)
            row["n_fail"] = sum(1 for r in all_rows if r.get("done") not in ("", None) and float(r["done"]) == -1)
            row["best_l2re"] = best
            row["final_source"] = source
            row["run_tag"] = final.get("_run_tag", "")
            out_rows.append(row)
            print(f"[{pde:26s} {mode:16s}] траекторий={row['n_trajectories']:3d} "
                  f"успешных={row['n_success']:3d} | итоговая l2re={row.get('l2re')} ({source})")

    if not out_rows:
        sys.exit("Нет данных — кампания ещё не отработала?")

    fields = base_fields + SUMMARY_FIELDS
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\n✅ Итог: {args.out} ({len(out_rows)} строк)")

    if args.upload:
        token = os.getenv("HF_TOKEN")
        if not token:
            sys.exit("--upload требует HF_TOKEN")
        HfApi(token=token).upload_file(
            path_or_fileobj=args.out,
            path_in_repo=f"{args.prefix}/ablation_final_metrics.csv",
            repo_id=args.hf_repo,
            repo_type="dataset",
        )
        print(f"⬆️  Загружен в {args.hf_repo}/{args.prefix}/ablation_final_metrics.csv")


if __name__ == "__main__":
    main()
