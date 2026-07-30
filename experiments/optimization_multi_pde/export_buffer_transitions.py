"""Экспорт буфера транзишенов из Comet в локальную папку (и опционально на HF).

Запускается ЕДИНОЖДЫ владельцем данных (нужен COMET_API_KEY с доступом к
workspace-источнику, по умолчанию saitama32). Дальше все обучения читают буфер
локально/из HF без Comet.

Отбор экспериментов тот же, что в collect_all_comet_transitions:
новые первыми, длительность >= 1 ч, первые --n-exps штук.

Структура вывода (её ждёт collect_all_local_transitions):
    out_dir/
        manifest.json
        001_<exp_name>/entry_step_00000.pt
        001_<exp_name>/entry_step_00001.pt
        002_<exp_name>/...

Тяжёлое поле solver_models (state_dict'ы PINN-моделей) в буфере не
используется и по умолчанию вырезается (--keep-solver-models чтобы оставить).

Заливка на HF: задать --hf-repo (и HF_TOKEN в окружении с правом записи).

Примеры:
    COMET_API_KEY=<ключ_с_доступом> python experiments/optimization_multi_pde/export_buffer_transitions.py \
        --proj rlpinn-poisson-boltzmann2d-tolerance --out buffer_export/poisson_boltzmann_2d

    HF_TOKEN=<токен> COMET_API_KEY=<ключ> python experiments/optimization_multi_pde/export_buffer_transitions.py \
        --proj rlpinn-poisson-boltzmann2d-tolerance --out buffer_export/poisson_boltzmann_2d \
        --hf-repo danil-e/rlpinn-ablation-buffers --hf-subdir poisson_boltzmann_2d
"""
import os
import sys
import io
import json
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

import torch

from RL.rl_utils.load_buffer.load_exps_from_comet import (
    get_comet_api,
    get_end_time,
    get_duration_hours,
    get_asset_step,
    WORKSPACE,
)


def export_project(proj_name, out_dir, n_exps, min_duration_hours, strip_solver_models=True):
    api = get_comet_api()
    print(f"🔍 Получаем эксперименты {WORKSPACE}/{proj_name}...")
    experiments = list(api.get_experiments(workspace=WORKSPACE, project_name=proj_name))
    experiments = sorted(experiments, key=get_end_time, reverse=True)
    experiments = [e for e in experiments if get_duration_hours(e) >= min_duration_hours]
    experiments = experiments[:n_exps]
    print(f"✅ Отобрано {len(experiments)} экспериментов (новые первыми).")

    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "workspace": WORKSPACE,
        "project": proj_name,
        "n_exps": len(experiments),
        "min_duration_hours": min_duration_hours,
        "solver_models_stripped": strip_solver_models,
        "experiments": [],
    }

    for index, exp in enumerate(experiments, 1):
        meta = exp.get_metadata()
        exp_name = meta.get("experimentName") or meta.get("experimentKey")
        exp_dir = os.path.join(out_dir, f"{index:03d}_{exp_name}")
        os.makedirs(exp_dir, exist_ok=True)

        assets = exp.get_asset_list()
        pt_assets = [
            a for a in assets
            if a["fileName"].endswith(".pt") and "entry_step" in a["fileName"]
        ]
        pt_assets = sorted(pt_assets, key=get_asset_step)

        saved = 0
        for asset in pt_assets:
            step = get_asset_step(asset)
            try:
                file_bytes = exp.get_asset(asset["assetId"], return_type="binary")
                data = torch.load(io.BytesIO(file_bytes), map_location="cpu")
                if strip_solver_models and isinstance(data, dict):
                    data.pop("solver_models", None)
                torch.save(data, os.path.join(exp_dir, f"entry_step_{step:05d}.pt"))
                saved += 1
            except Exception as exc:
                print(f"   skipped {asset['fileName']}: {exc}")

        manifest["experiments"].append({
            "index": index,
            "key": meta.get("experimentKey"),
            "name": exp_name,
            "n_transitions": saved,
            "duration_hours": round(get_duration_hours(exp), 3),
        })
        print(f"[{index:3d}/{len(experiments)}] {exp_name}: {saved} транзишенов")

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total = sum(e["n_transitions"] for e in manifest["experiments"])
    print(f"\n🚀 Экспортировано {total} транзишенов в {out_dir}")
    return manifest


def pack_export(src_dir, dst_dir):
    """Упаковывает сырой экспорт (файл на транзишен) в файл на эксперимент.

    src_dir/001_<exp>/entry_step_*.pt  ->  dst_dir/001_<exp>.pt (список транзишенов)

    Порядок транзишенов внутри эксперимента сохраняется (по номеру entry_step) —
    он важен: цепочки эпизодов и chain-rewards восстанавливаются по нему.
    Такой формат на два порядка дружелюбнее к HF (80 файлов вместо ~6000).
    """
    from RL.rl_utils.load_buffer.load_exps_from_comet import _entry_step_from_filename

    os.makedirs(dst_dir, exist_ok=True)
    exp_dirs = sorted(
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d)) and not d.startswith(".")
    )

    total = 0
    for exp_dir in exp_dirs:
        dir_path = os.path.join(src_dir, exp_dir)
        pt_files = sorted(
            (f for f in os.listdir(dir_path) if f.endswith(".pt") and "entry_step" in f),
            key=_entry_step_from_filename,
        )
        transitions = [
            torch.load(os.path.join(dir_path, f), map_location="cpu") for f in pt_files
        ]
        torch.save(transitions, os.path.join(dst_dir, f"{exp_dir}.pt"))
        total += len(transitions)
        print(f"[pack] {exp_dir}: {len(transitions)} транзишенов")

    manifest_src = os.path.join(src_dir, "manifest.json")
    if os.path.exists(manifest_src):
        with open(manifest_src, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["layout"] = "packed_per_experiment"
        with open(os.path.join(dst_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n📦 Упаковано {total} транзишенов из {len(exp_dirs)} экспериментов в {dst_dir}")
    return dst_dir


def upload_to_hf(out_dir, hf_repo, hf_subdir):
    from huggingface_hub import HfApi

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("Для заливки на HF нужен HF_TOKEN в окружении.")
    api = HfApi(token=token)
    api.create_repo(hf_repo, repo_type="dataset", private=False, exist_ok=True)
    print(f"⬆️ Заливаем {out_dir} -> {hf_repo}/{hf_subdir} ...")
    api.upload_folder(
        folder_path=out_dir,
        repo_id=hf_repo,
        repo_type="dataset",
        path_in_repo=hf_subdir,
    )
    print(f"✅ Готово: https://huggingface.co/datasets/{hf_repo}/tree/main/{hf_subdir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", type=str, default="rlpinn-poisson-boltzmann2d-tolerance")
    parser.add_argument("--out", type=str, default="buffer_export/poisson_boltzmann_2d")
    parser.add_argument("--n-exps", type=int, default=200)
    parser.add_argument("--min-duration-hours", type=float, default=1.0)
    parser.add_argument("--keep-solver-models", action="store_true",
                        help="Не вырезать solver_models (файлы будут сильно тяжелее).")
    parser.add_argument("--pack-to", type=str, default=None,
                        help="Куда сложить упакованную версию (файл на эксперимент); "
                             "именно она заливается на HF.")
    parser.add_argument("--pack-only", action="store_true",
                        help="Не ходить в Comet: только упаковать уже выгруженный --out.")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="Например danil-e/rlpinn-ablation-buffers; без него — только локальный экспорт.")
    parser.add_argument("--hf-subdir", type=str, default="poisson_boltzmann_2d")
    args = parser.parse_args()

    if not args.pack_only:
        export_project(
            proj_name=args.proj,
            out_dir=args.out,
            n_exps=args.n_exps,
            min_duration_hours=args.min_duration_hours,
            strip_solver_models=not args.keep_solver_models,
        )

    upload_dir = args.out
    if args.pack_to:
        upload_dir = pack_export(args.out, args.pack_to)

    if args.hf_repo:
        upload_to_hf(upload_dir, args.hf_repo, args.hf_subdir)


if __name__ == "__main__":
    main()
