"""Приёмка нового уравнения в кампанию: буфер -> порог -> вердикт.

Буферы приезжают порциями, и на каждое новое уравнение нужно проделать одно и
то же: убедиться, что буфер целый, подобрать порог успеха, проверить, что с
этим порогом агент реально обучается. Скрипт делает всё три шага подряд и
печатает готовую строку для реестра.

Что проверяется:
  1. структура — все .pt читаются, нет NaN в наградах и состояниях, формы
     состояний одинаковые, формат действий понятен загрузчику;
  2. порог — распределение лучшего лосса по цепочкам (той величиной, которую
     сравнивает загрузчик: `_transition_loss_value`, а НЕ сырое поле reward);
  3. вердикт — сколько успешных терминалов останется в буфере после загрузки
     с этим порогом. Ноль означает, что агенту не на чем учиться успеху,
     и запускать кампанию бессмысленно.

Запуск:
    python experiments/agent_ablation/prepare_pde.py --pde wave1d
    python experiments/agent_ablation/prepare_pde.py --pde wave1d --target-success-frac 0.75
"""
import argparse
import json
import math
import os
import statistics
import sys

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from experiments.agent_ablation.pde_registry import PDE_SPECS, get_spec  # noqa: E402

DONE_KEYS = ("abs_done", "done")
# Ниже этого числа успешных терминалов после загрузки считаем, что порог
# непригоден: success replay почти пуст и success rate ничего не измерит.
MIN_SUCCESS_TERMINALS = 30


def _done(tr):
    for key in DONE_KEYS:
        if key in tr:
            try:
                return int(tr[key])
            except (TypeError, ValueError):
                return 0
    return 0


def manifest_check(buffer_dir):
    """Сверяет список экспериментов манифеста с файлами на диске.

    Экспортёр пишет manifest.json заранее и помечает complete=true, когда
    считает работу законченной, но выгрузка части файлов может не доехать:
    у heat2d_multiscale манифест объявил 139 экспериментов при 64 файлах, то
    есть 53% переходов отсутствовало. Ни загрузчик, ни проверка структуры
    такого не заметят — они читают то, что лежит. Поэтому сверяем явно.
    """
    manifest_path = os.path.join(buffer_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return {"summary": "manifest.json нет — сверить полноту не с чем "
                           "(старый буфер, выгружен до появления манифестов)",
                "fatal": ""}
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    exps = manifest.get("experiments", [])
    files = [f for f in os.listdir(buffer_dir) if f.endswith(".pt")]
    missing, present_tr, missing_tr = [], 0, 0
    for e in exps:
        name = e.get("name", "")
        n_tr = int(e.get("n_transitions", 0) or 0)
        if name and any(name in f for f in files):
            present_tr += n_tr
        else:
            missing.append(name or "(без имени)")
            missing_tr += n_tr
    orphans = [f for f in files
               if not any(e.get("name") and e["name"] in f for e in exps)]
    total_tr = present_tr + missing_tr
    summary = (f"манифест: экспериментов {len(exps)}, complete={manifest.get('complete')}; "
               f"файлов {len(files)}; без файла {len(missing)}; "
               f"файлов вне манифеста {len(orphans)}")
    fatal = ""
    if missing:
        доля = missing_tr / total_tr if total_tr else 1.0
        fatal = (f"у {len(missing)} экспериментов из {len(exps)} нет файла: "
                 f"потеряно {missing_tr} переходов из {total_tr} ({доля:.0%}). "
                 f"Примеры: {', '.join(missing[:3])}")
    elif orphans:
        summary += ("\n   ВНИМАНИЕ: файлы вне манифеста — вероятно, остатки "
                    f"прошлой попытки выгрузки: {', '.join(orphans[:3])}")
    return {"summary": summary, "fatal": fatal}


def structural_check(buffer_dir):
    """Читает весь буфер и собирает всё, что может сломать загрузку."""
    import torch
    from RL.rl_utils.load_buffer.load_exps_from_comet import _transition_loss_value

    files = sorted(f for f in os.listdir(buffer_dir) if f.endswith(".pt"))
    if not files:
        raise SystemExit(f"В {buffer_dir} нет .pt-файлов — буфер не выгружен?")

    rep = {"files": len(files), "transitions": 0, "chains": 0, "done1": 0, "done_1": 0,
           "bad_files": [], "missing_keys": 0, "nan_state": 0, "no_loss": 0,
           "action_bad": 0, "shapes": set(), "chain_best": [], "chain_lens": []}
    for fname in files:
        try:
            payload = torch.load(os.path.join(buffer_dir, fname), map_location="cpu",
                                 weights_only=False)
        except Exception as exc:
            rep["bad_files"].append(f"{fname}: {type(exc).__name__}: {exc}")
            continue
        transitions = payload if isinstance(payload, list) else [payload]
        current = []
        for tr in transitions:
            if not isinstance(tr, dict):
                rep["missing_keys"] += 1
                continue
            keys = set(tr)
            if not ({"state", "next_state"} <= keys
                    and ({"reward", "env_raw_reward"} & keys)
                    and ({"done", "abs_done"} & keys)):
                rep["missing_keys"] += 1
                continue
            rep["transitions"] += 1

            state = tr.get("state")
            if isinstance(state, dict):
                for key in ("loss_total", "loss_oper", "loss_bnd"):
                    value = state.get(key)
                    if torch.is_tensor(value):
                        rep["shapes"].add(tuple(value.shape))
                        if not torch.isfinite(value).all():
                            rep["nan_state"] += 1
                            break
            action = tr.get("action", tr.get("action_raw"))
            if not (isinstance(action, (tuple, list)) and len(action) >= 2):
                rep["action_bad"] += 1

            loss = _transition_loss_value(tr, loss_key="loss_total", state_loss_is_log=False)
            if loss is None or not math.isfinite(loss):
                rep["no_loss"] += 1
                continue
            current.append(abs(float(loss)))
            done = _done(tr)
            if done != 0:
                rep["chains"] += 1
                rep["chain_lens"].append(len(current))
                rep["chain_best"].append(min(current))
                rep["done1" if done == 1 else "done_1"] += 1
                current = []
        if current:
            rep["chains"] += 1
            rep["chain_lens"].append(len(current))
            rep["chain_best"].append(min(current))
    return rep


def quantile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def loaded_success_count(spec, tolerance, buffer_dir):
    """Сколько успешных терминалов останется после НАСТОЯЩЕЙ загрузки буфера."""
    from RL.rl_utils.load_buffer.load_exps_from_comet import collect_all_local_transitions
    from RL.rl_utils.per_buffer import PrioritizedReplayBuffer

    buf = collect_all_local_transitions(
        PrioritizedReplayBuffer(20000),
        buffer_dir=buffer_dir,
        max_exps_last=200,
        tolerance=tolerance,
        prev_tol=0.0,
        new_tol=True,
        use_log_state=False,
        proj_name=spec.key,
        recompute_chain_rewards=True,
        set_reward_from_next_loss=True,
    )
    dones = [t.done for t in buf.memory]
    return len(buf), dones.count(1), dones.count(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", required=True, help="Ключ уравнения из реестра.")
    parser.add_argument("--hf-repo", default="danil-e/rlpinn-ablation-buffers")
    parser.add_argument("--target-success-frac", type=float, default=0.75,
                        help="Доля успешных цепочек, под которую подбирается порог, "
                             "если в реестре его нет или он непригоден.")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="Проверить конкретный порог вместо реестрового.")
    args = parser.parse_args()

    spec = get_spec(args.pde)
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id=args.hf_repo, repo_type="dataset",
                             allow_patterns=[f"{spec.key}/*"])
    buffer_dir = os.path.join(root, spec.key)
    if not os.path.isdir(buffer_dir):
        raise SystemExit(f"В {args.hf_repo} нет папки {spec.key} — буфер ещё не выгружен.")

    print(f"=== {spec.key} ({spec.title}) ===\n{buffer_dir}\n")

    print("--- 0. полнота выгрузки ---")
    manifest_report = manifest_check(buffer_dir)
    print(manifest_report["summary"])
    if manifest_report["fatal"]:
        print("\n" + "=" * 70)
        print(f"НЕ ГОДИТСЯ: {spec.key}")
        print(f"  - {manifest_report['fatal']}")
        print("  Буфер надо выгрузить заново: запускать кампанию на неполном "
              "буфере нельзя, недостача ничем себя не проявит.")
        sys.exit(1)

    print("\n--- 1. структура буфера ---")
    rep = structural_check(buffer_dir)
    print(f"файлов {rep['files']}, транзишенов {rep['transitions']}, цепочек {rep['chains']} "
          f"(done=1 {rep['done1']}, done=-1 {rep['done_1']})")
    print(f"битых файлов {len(rep['bad_files'])}, без обязательных ключей {rep['missing_keys']}, "
          f"кривых action {rep['action_bad']}, NaN в состояниях {rep['nan_state']}, "
          f"без извлекаемого лосса {rep['no_loss']}")
    print(f"формы состояний: {sorted(rep['shapes'])}, длина цепочки: медиана "
          f"{statistics.median(rep['chain_lens']):.0f}, макс {max(rep['chain_lens'])}")
    for bad in rep["bad_files"][:5]:
        print(f"  БИТЫЙ ФАЙЛ: {bad}")

    fatal = []
    if rep["bad_files"]:
        fatal.append(f"{len(rep['bad_files'])} файлов не читаются")
    if rep["nan_state"]:
        fatal.append(f"NaN/inf в состояниях у {rep['nan_state']} переходов")
    if rep["action_bad"]:
        fatal.append(f"{rep['action_bad']} переходов с непонятным action")
    if len(rep["shapes"]) > 1:
        fatal.append(f"разные формы состояний: {sorted(rep['shapes'])}")
    if rep["transitions"] < 500:
        fatal.append(f"мало переходов: {rep['transitions']}")

    print("\n--- 2. порог ---")
    bests = rep["chain_best"]
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"   p{int(q * 100):02d}: {quantile(bests, q):.6g}")
    proposed = quantile(bests, args.target_success_frac)
    tolerance = args.tolerance if args.tolerance is not None else spec.tolerance
    if tolerance is not None:
        frac = sum(1 for b in bests if b <= tolerance) / max(len(bests), 1)
        print(f"порог из реестра {tolerance:.10g}: {frac:.1%} цепочек достигают его")
        if not (0.05 <= frac <= 0.95):
            print(f"   не годится (нужно 5–95%), берём калиброванный "
                  f"{proposed:.10g} под {args.target_success_frac:.0%}")
            tolerance = proposed
    else:
        tolerance = proposed
        print(f"порога в реестре нет — калиброванный {tolerance:.10g} "
              f"под {args.target_success_frac:.0%}")

    print("\n--- 3. что останется в буфере после загрузки с этим порогом ---")
    total, done1, done_1 = loaded_success_count(spec, tolerance, buffer_dir)
    print(f"переходов {total}, успешных терминалов {done1}, провальных {done_1}")
    if done1 < MIN_SUCCESS_TERMINALS:
        fatal.append(f"успешных терминалов всего {done1} (< {MIN_SUCCESS_TERMINALS}) — "
                     "агенту не на чем учиться успеху")

    print("\n" + "=" * 70)
    if fatal:
        print(f"НЕ ГОДИТСЯ: {spec.key}")
        for f in fatal:
            print(f"  - {f}")
        sys.exit(1)
    print(f"ГОДИТСЯ: {spec.key}, tolerance={tolerance:.10g}, "
          f"успешных терминалов {done1}")
    if spec.tolerance is None or abs((spec.tolerance or 0) - tolerance) > 1e-12:
        print(f"\nв реестр:  tolerance={tolerance:.10g},")
    print("дальше:    smoke_test_hf_buffer.py --pde " + spec.key +
          "  (обучение агента во всех 4 режимах)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
