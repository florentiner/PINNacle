"""Экспорт буферов транзишенов для абляции: Comet -> HF, пачкой по реестру.

Абляции на новом уравнении нужен буфер: из него агент вычитывает политику на
оффлайн-претрене (500 шагов) и им же инициализируется онлайн-обучение. Буфер
лежит в Comet-проекте уравнения (workspace saitama32), а на Kaggle Comet не
ходим — все запуски читают открытый HF-датасет. Этот скрипт делает разовый
перенос: Comet -> локальная папка -> упаковка по эксперименту -> HF.

Нужен COMET_API_KEY с доступом к workspace saitama32 (чтение) и HF_TOKEN с
правом записи в датасет буферов.

Примеры:
    # одно уравнение
    COMET_API_KEY=... HF_TOKEN=... python experiments/agent_ablation/export_buffers.py \
        --pde burgers1d

    # всё, что осталось посчитать (solvable, буфера ещё нет на HF)
    COMET_API_KEY=... HF_TOKEN=... python experiments/agent_ablation/export_buffers.py \
        --tier solvable --skip-existing

    # что вообще нужно экспортировать (без ключей, ничего не качает)
    python experiments/agent_ablation/export_buffers.py --tier solvable --dry-run
"""
import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from experiments.agent_ablation.pde_registry import PDE_SPECS, get_spec, select

DEFAULT_HF_REPO = "danil-e/rlpinn-ablation-buffers"


def hf_subdirs_present(hf_repo):
    """Какие уравнения уже лежат на HF (по подпапкам с .pt-файлами)."""
    from huggingface_hub import HfApi

    try:
        files = HfApi().list_repo_files(hf_repo, repo_type="dataset")
    except Exception as exc:
        print(f"⚠️  не удалось прочитать {hf_repo}: {exc}")
        return set()
    return {f.split("/")[0] for f in files if f.endswith(".pt") and "/" in f}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", action="append", default=None,
                        help="Ключ уравнения из реестра; можно повторять.")
    parser.add_argument("--tier", action="append", default=None,
                        choices=["solvable", "borderline", "unsolvable"],
                        help="Взять все уравнения этого класса; можно повторять.")
    parser.add_argument("--out-root", type=str, default="buffer_export",
                        help="Куда складывать сырой экспорт (файл на транзишен).")
    parser.add_argument("--pack-root", type=str, default="buffer_export_packed",
                        help="Куда складывать упакованный экспорт (файл на эксперимент) — "
                             "именно он заливается на HF.")
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_HF_REPO,
                        help="HF-датасет буферов; none — только локальный экспорт.")
    parser.add_argument("--n-exps", type=int, default=200,
                        help="Сколько последних экспериментов проекта брать.")
    parser.add_argument("--min-duration-hours", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Пропускать уравнения, чья папка уже есть на HF.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Только показать план: что откуда куда поедет.")
    args = parser.parse_args()

    keys = list(args.pde or [])
    for tier in args.tier or []:
        keys += [k for k in select(tiers=(tier,)) if k not in keys]
    if not keys:
        parser.error("нужен хотя бы один --pde или --tier")
    for key in keys:
        get_spec(key)  # ранняя проверка ключей

    use_hf = args.hf_repo and args.hf_repo.lower() != "none"
    present = hf_subdirs_present(args.hf_repo) if (use_hf and (args.skip_existing or args.dry_run)) else set()

    plan = []
    for key in keys:
        spec = PDE_SPECS[key]
        if args.skip_existing and key in present:
            print(f"[{key:26s}] уже на HF — пропуск")
            continue
        plan.append(spec)

    print(f"\nК экспорту: {len(plan)} уравнений")
    for spec in plan:
        mark = " (уже есть на HF)" if spec.key in present else ""
        tol = "порог не откалиброван" if spec.tolerance is None else f"tolerance={spec.tolerance:.10g}"
        print(f"  {spec.key:26s} <- comet:{spec.comet_project:45s} {tol}{mark}")
    if args.dry_run:
        print("\n--dry-run: ничего не выгружено.")
        return

    if not os.getenv("COMET_API_KEY"):
        raise SystemExit(
            "COMET_API_KEY не задан. Нужен ключ с доступом на чтение workspace "
            "saitama32 — буферы уравнений живут только там."
        )
    if use_hf and not os.getenv("HF_TOKEN"):
        raise SystemExit("HF_TOKEN не задан (нужен с правом записи в датасет буферов).")

    from experiments.optimization_multi_pde.export_buffer_transitions import (
        export_project, pack_export, upload_to_hf,
    )

    failures = []
    for spec in plan:
        print(f"\n{'=' * 70}\n=== {spec.key} <- {spec.comet_project}\n{'=' * 70}")
        raw_dir = os.path.join(args.out_root, spec.key)
        packed_dir = os.path.join(args.pack_root, spec.key)
        try:
            export_project(
                proj_name=spec.comet_project,
                out_dir=raw_dir,
                n_exps=args.n_exps,
                min_duration_hours=args.min_duration_hours,
                strip_solver_models=True,
            )
            pack_export(raw_dir, packed_dir)
            if use_hf:
                upload_to_hf(packed_dir, args.hf_repo, spec.key)
        except Exception as exc:
            print(f"❌ {spec.key}: {type(exc).__name__}: {exc}")
            failures.append((spec.key, f"{type(exc).__name__}: {exc}"))

    print(f"\n{'=' * 70}")
    print(f"Готово: {len(plan) - len(failures)}/{len(plan)} уравнений выгружено.")
    for key, err in failures:
        print(f"  ❌ {key}: {err}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
