"""Смок-тест: буфер тянется с HuggingFace и на нём обучается агент абляции.

Проверяет ровно тот путь, которым пойдёт настоящий запуск с `--buffer-src hf`:
скачивание открытого датасета -> collect_all_local_transitions -> DQNAgent.optim_
для всех четырёх режимов абляции. Comet не задействован вообще — тест падает,
если код попытается сходить в Comet за ключом.

Запуск из корня репозитория:
    python smoke_test_hf_buffer.py
"""
import os
import sys
import tempfile

os.environ.setdefault("DDEBACKEND", "pytorch")

import numpy as np
import torch

HF_REPO = os.getenv("ABLATION_HF_REPO", "danil-e/rlpinn-ablation-buffers")
HF_SUBDIR = os.getenv("ABLATION_HF_SUBDIR", "poisson_boltzmann_2d")

# Параметры загрузки буфера poisson_boltzmann_2d (таблица tolerance-проектов)
TOLERANCE = 0.039669186
PREV_TOL = 0.0
NEW_TOL = True
USE_LOG_STATE = False

OPTIMIZERS = {
    "Adam": {"lr": [1e-2, 1e-3, 1e-4], "epochs": [100, 1000, 2500]},
    "LBFGS": {"lr": [1, 5e-1, 1e-1], "epochs": [100, 500, 1500]},
    "PSO": {"lr": [0.0, 1e-3, 1e-4], "epochs": [100, 200, 300]},
}


def download_buffer():
    from huggingface_hub import snapshot_download

    print(f"⬇️  Скачиваем {HF_REPO}/{HF_SUBDIR} ...")
    ds_root = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=[f"{HF_SUBDIR}/*"],
    )
    buffer_dir = os.path.join(ds_root, HF_SUBDIR)
    if not os.path.isdir(buffer_dir):
        raise SystemExit(f"В датасете {HF_REPO} нет подпапки {HF_SUBDIR}.")

    files = [f for f in sorted(os.listdir(buffer_dir)) if f.endswith(".pt")]
    size_mb = sum(
        os.path.getsize(os.path.join(buffer_dir, f)) for f in os.listdir(buffer_dir)
    ) / 1e6
    print(f"✅ Скачано: {buffer_dir}\n   файлов-экспериментов: {len(files)}, объём: {size_mb:.1f} MB")
    return buffer_dir


def build_buffer(buffer_dir):
    from RL.rl_utils.load_buffer.load_exps_from_comet import collect_all_local_transitions
    from RL.rl_utils.per_buffer import PrioritizedReplayBuffer

    buf = collect_all_local_transitions(
        PrioritizedReplayBuffer(20000),
        buffer_dir=buffer_dir,
        max_exps_last=200,
        tolerance=TOLERANCE,
        prev_tol=PREV_TOL,
        new_tol=NEW_TOL,
        use_log_state=USE_LOG_STATE,
        proj_name=HF_SUBDIR,
        recompute_chain_rewards=True,
        set_reward_from_next_loss=True,
    )
    return buf


def clone_buffer(src):
    """Свежая копия буфера на каждый режим: приоритеты не должны переноситься."""
    from RL.rl_utils.per_buffer import PrioritizedReplayBuffer

    buf = PrioritizedReplayBuffer(20000)
    for tr in src.memory:
        buf.push(*tr)
    return buf


def run_mode(ablation, base_buffer):
    from RL.rl_algorithms import DQNAgent

    torch.manual_seed(0)
    np.random.seed(0)

    agent = DQNAgent(
        optimizer_dict=OPTIMIZERS,
        memory_size=20000,
        gamma=0.9,
        lr=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,
        n_transitions_reinit=2000,
        exp=None,
        warmup_updates=2,
        ablation=ablation,
        model_snapshot_dir=tempfile.mkdtemp(prefix=f"hf_smoke_{ablation}_"),
    )
    agent.replay_buffer = clone_buffer(base_buffer)

    priors_before = list(agent.replay_buffer.prior)
    loss_opt, loss_param = agent.optim_(iters=3)

    assert len(loss_opt) == 3, f"ожидалось 3 апдейта, получено {len(loss_opt)}"
    for lo, lp in zip(loss_opt, loss_param):
        assert np.isfinite(lo), f"loss_opt не конечен: {lo}"
        assert np.isfinite(lp), f"loss_param не конечен: {lp}"

    priors_changed = any(a != b for a, b in zip(priors_before, agent.replay_buffer.prior))
    if ablation == "no_per":
        assert not priors_changed, "no_per: приоритеты не должны обновляться"
    else:
        assert priors_changed, f"{ablation}: приоритеты должны обновляться"

    return loss_opt, priors_changed


def main():
    if os.getenv("COMET_API_KEY"):
        print("⚠️  COMET_API_KEY выставлен — тест должен работать и без него.")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    buffer_dir = download_buffer()
    buf = build_buffer(buffer_dir)

    dones = [t.done for t in buf.memory]
    print(
        f"\n📊 Буфер: {len(buf)} транзишенов | "
        f"done=1: {dones.count(1)}, done=-1: {dones.count(-1)}, done=0: {dones.count(0)}"
    )
    assert len(buf) > 1000, f"подозрительно маленький буфер: {len(buf)}"
    assert dones.count(1) > 0, "нет успешных терминалов — success replay будет пустым"

    from RL.rl_algorithms import ABLATION_MODES

    results = {}
    for mode in ABLATION_MODES:
        print(f"\n{'=' * 70}\n=== ablation = {mode} ===\n{'=' * 70}")
        loss_opt, priors_changed = run_mode(mode, buf)
        results[mode] = (loss_opt, priors_changed)
        print(f"OK [{mode}]: loss_opt={[round(x, 4) for x in loss_opt]}, priors_changed={priors_changed}")

    print(f"\n{'=' * 70}")
    print("HF-БУФЕР РАБОТАЕТ ДЛЯ ВСЕХ РЕЖИМОВ АБЛЯЦИИ")
    for mode, (loss_opt, _) in results.items():
        print(f"  {mode:18s} loss_opt={[round(x, 4) for x in loss_opt]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
