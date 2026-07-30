"""Смок-тест абляции DQN-стека (без Comet и без обучения PINN).

Проверяет для каждого режима абляции:
  - агент строится и оптимизируется на синтетическом буфере;
  - лоссы конечны, веса сетей обновляются;
  - no_per: приоритеты буфера НЕ обновляются, warmup выключен;
  - none/no_soft_watkins/no_trust_region: приоритеты обновляются;
  - select_action возвращает корректный action.

Запуск из корня репозитория:
    python smoke_test_ablation.py
"""
import os
import random
import tempfile

os.environ.setdefault("DDEBACKEND", "pytorch")

import numpy as np
import torch

from RL.rl_algorithms import DQNAgent, ABLATION_MODES

SNAPSHOT_TMP = tempfile.mkdtemp(prefix="rl_smoke_snapshots_")

STATE_HW = 26
OPTIMIZERS = {
    "Adam": {"lr": [1e-2, 1e-3, 1e-4], "epochs": [100, 1000, 2500]},
    "LBFGS": {"lr": [1, 5e-1, 1e-1], "epochs": [100, 500, 1500]},
    "PSO": {"lr": [0.0, 1e-3, 1e-4], "epochs": [100, 200, 300]},
}


def make_state(device):
    return {
        "loss_total": torch.rand(STATE_HW, STATE_HW, device=device),
        "loss_oper": torch.rand(STATE_HW, STATE_HW, device=device),
        "loss_bnd": torch.rand(STATE_HW, STATE_HW, device=device),
        "delta": torch.rand(STATE_HW, STATE_HW, device=device) * 2 - 1,
    }


def make_action():
    optim_class = random.randint(0, len(OPTIMIZERS) - 1)
    # формат как в rl_trainer: (optim_class, {'lr': i, 'epochs': j})
    return (optim_class, {"lr": random.randint(0, 2), "epochs": random.randint(0, 2)})


def fill_buffer(agent, device, n_chains=30, chain_len=6):
    """Синтетические цепочки: done=0...0, в конце done=1 (успех) или -1 (fail)."""
    for chain_i in range(n_chains):
        success = chain_i % 2 == 0
        state = make_state(device)
        for step in range(chain_len):
            next_state = make_state(device)
            is_last = step == chain_len - 1
            done = (1 if success else -1) if is_last else 0
            reward = float(np.random.rand() * 0.1)          # ~ loss value
            model_reward = float(np.random.randn())          # chain reward
            if is_last and success:
                model_reward = abs(model_reward) + 1.0       # > success_threshold
            agent.push_memory((
                state,
                next_state,
                make_action(),
                reward,
                done,
                model_reward,
                random.choice([-1, 0, 1]),
            ))
            state = next_state


def snapshot_params(agent):
    return torch.cat([p.detach().flatten().clone() for p in agent.model_optim.parameters()])


def run_mode(ablation, device):
    print(f"\n{'=' * 70}\n=== ablation = {ablation} ===\n{'=' * 70}")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    agent = DQNAgent(
        n_observation=None,
        n_action=None,
        optimizer_dict=OPTIMIZERS,
        memory_size=10000,
        gamma=0.9,
        lr=1e-3,
        device=device,
        batch_size=16,
        n_transitions_reinit=2000,
        exp=None,
        warmup_updates=1,   # короткий warmup, чтобы смок прошёл offline-recalc веткой
        ablation=ablation,
        model_snapshot_dir=os.path.join(SNAPSHOT_TMP, ablation),
    )

    fill_buffer(agent, device)
    assert len(agent.replay_buffer) >= agent.batch_size

    if ablation == "no_per":
        assert not agent.warmup_active, "no_per должен отключать warmup"

    priors_before = list(agent.replay_buffer.prior)
    params_before = snapshot_params(agent)

    loss_opt_arr, loss_param_arr = agent.optim_(iters=3)

    assert len(loss_opt_arr) == 3, f"ожидалось 3 апдейта, получено {len(loss_opt_arr)}"
    for lo, lp in zip(loss_opt_arr, loss_param_arr):
        assert np.isfinite(lo), f"loss_opt не конечен: {lo}"
        assert np.isfinite(lp), f"loss_param не конечен: {lp}"

    params_after = snapshot_params(agent)
    assert not torch.equal(params_before, params_after), "веса model_optim не обновились"

    priors_changed = any(a != b for a, b in zip(priors_before, agent.replay_buffer.prior))
    if ablation == "no_per":
        assert not priors_changed, "no_per: приоритеты не должны обновляться"
    else:
        assert priors_changed, f"{ablation}: приоритеты должны обновляться"

    action_dict, action_raw, is_model = agent.select_action(make_state(device))
    assert action_dict["type"] in OPTIMIZERS
    assert action_dict["epochs"] in OPTIMIZERS[action_dict["type"]]["epochs"]
    assert action_dict["params"]["lr"] in OPTIMIZERS[action_dict["type"]]["lr"]

    print(f"\nOK [{ablation}]: losses opt={[round(x, 4) for x in loss_opt_arr]}, "
          f"param={[round(x, 4) for x in loss_param_arr]}, priors_changed={priors_changed}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    for mode in ABLATION_MODES:
        run_mode(mode, device)
    print(f"\n{'=' * 70}\nВСЕ РЕЖИМЫ АБЛЯЦИИ ПРОШЛИ СМОК-ТЕСТ: {ABLATION_MODES}\n{'=' * 70}")


if __name__ == "__main__":
    main()
