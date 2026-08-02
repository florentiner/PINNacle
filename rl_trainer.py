import os
import sys
import time
import json
import dill
import random
import itertools
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import datetime

dill.settings["recurse"] = True
import torch
import deepxde as dde
from RL.rl_environment import EnvRLOptimizer
from RL.rl_algorithms import DQNAgent
from src.utils.callbacks import ModelSaverCallback  
from deepxde.optimizers.config import set_LBFGS_options, set_PSO_options, LBFGS_options, PSO_options
from typing import Any, Dict
from RL.rl_utils.load_buffer.load_exps_from_comet import (
    collect_all_comet_transitions,
    collect_all_local_transitions,
)

# Enforce single-precision defaults before any model/layer creation.
dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

output_dir = os.path.join('.', 'transitions')

os.makedirs(output_dir, exist_ok=True)

def _greedy_action_from_state(rl_agent, state):
    """Чисто жадное действие агента из состояния (без ε и без инкремента steps_done)."""
    with torch.no_grad():
        x = rl_agent._stack_state(state).unsqueeze(0)
        flat, q_opt = rl_agent.model_optim(x)
        optim_class = int(torch.argmax(q_opt).item())
        optim_name = rl_agent.i2opt[optim_class]
        param_dict = rl_agent.model_params(flat, [optim_name])[0]

        epochs_class = 0
        param_class = {}
        for key in param_dict:
            if key == "epochs":
                epochs_class = int(torch.argmax(param_dict[key]).item())
            else:
                param_class[key] = int(torch.argmax(param_dict[key]).item())
    return rl_agent.post_proc_model(optim_class, epochs_class, param_class)


def _print_offline_greedy_chain_diagnostic(rl_agent, max_len=12):
    """Диагностика после оффлайн-претрена: что жадная политика выберет
    из нулевого состояния и вдоль последнего успешного эпизода буфера."""
    print("\n=== Диагностика жадной политики после оффлайн-претрена ===")

    memory = rl_agent.replay_buffer.memory
    if not memory:
        print("Буфер пуст — диагностика пропущена.")
        return

    template = memory[0].state["loss_total"]
    zero_state = {
        key: torch.zeros_like(template)
        for key in ("loss_total", "loss_oper", "loss_bnd")
    }
    first_action = _greedy_action_from_state(rl_agent, zero_state)
    print(f"Из нулевого состояния: {first_action['type']}"
          f"(lr={first_action['params'].get('lr')}, epochs={first_action['epochs']})")

    # последний успешный эпизод в буфере
    success_chain, current = [], []
    for tr in memory:
        current.append(tr)
        if tr.done != 0:
            if tr.done == 1:
                success_chain = current
            current = []

    if not success_chain:
        print("Успешных эпизодов в буфере нет — цепочная диагностика пропущена.")
        return

    print(f"Жадные действия вдоль успешного эпизода (длина {len(success_chain)}):")
    for i, tr in enumerate(success_chain[:max_len]):
        greedy = _greedy_action_from_state(rl_agent, tr.state)
        taken_name = rl_agent.i2opt[int(tr.action[0])]
        print(f"  s{i}: greedy={greedy['type']}"
              f"(lr={greedy['params'].get('lr')}, epochs={greedy['epochs']})"
              f" | в буфере был {taken_name}")


# --- утилита: реинициализация torch модулей (для "новой траектории") ---
def reinit_torch_weights(module):
    import torch

    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

def get_state_shape(loss_surface_params):
    min_x, max_x, xnum = loss_surface_params["x_range"]
    min_y, max_y = min_x, max_x
    step_size = (max_x - min_x) / xnum

    x_coords = torch.arange(min_x, max_x + step_size, step_size)
    y_coords = torch.arange(min_y, max_y + step_size, step_size)

    return tuple(torch.meshgrid(x_coords, y_coords)[0].shape)


def _serialize_solver_models(solver_models):
    if solver_models is None:
        return None

    serialized_models = []
    for solver_model in solver_models:
        if solver_model is None:
            serialized_models.append(None)
            continue
        serialized_models.append({
            "class_name": type(solver_model).__name__,
            "state_dict": {
                key: value.detach().to("cpu").clone()
                for key, value in solver_model.state_dict().items()
            },
        })
    return serialized_models


def _build_torch_optimizer(opt_name: str, params, action: Dict[str, Any]):

    name = (opt_name or "").lower()
    opt_params = action.get("params", {})
    if name == "adam":
        lr = float(opt_params.get("lr", 1e-3))
        return torch.optim.Adam(
            params, lr=lr,
        )

    if name in ["lbfgs", "l-bfgs", "l_bfgs", "LBFGS"]:
        # torch LBFGS (для pytorch backend DeepXDE норм)
        opt = torch.optim.LBFGS(
            params,
            lr=action["params"]["lr"],
            line_search_fn="strong_wolfe",
            max_iter = 10
        )

        return opt 

    if name in ["pso", "PSO", "Pso"]:
        # Передаём гиперпараметры через глобальные PSO_options
        set_PSO_options(
            lr=float(opt_params.get("lr", 1e-3)),
        )
        return "PSO"  # deepxde/optimizers/pytorch/pso.PSO

    raise ValueError(f"Unknown optimizer type: {opt_name}. Expected Adam / LBFGS / PSO.")


def _extract_weighted_train_loss(model) -> float:
    loss_train = getattr(getattr(model, "train_state", None), "loss_train", None)
    if loss_train is None:
        return float("nan")

    loss_train = np.asarray(loss_train, dtype=np.float64)
    loss_value = float(np.sum(loss_train))
    if not np.isfinite(loss_value) or loss_value < 0.0:
        return float("nan")
    return loss_value


def run_deepxde_rl_training(
    model,
    loss_weights,
    train_args: Dict[str, Any],
    rl_agent_params,
    optimizers_dict: Dict[str, Any],
    AE_model_params=None,
    AE_train_params=None,
    loss_surface_params=None,
    save_path: str = ".",

):
    """
    model: deepxde.Model (уже созданный get_model())
    train_args: то, что раньше шло в model.train(**train_args)
    env_ctor: класс/фабрика EnvRLOptimizer
    agent_ctor: класс/фабрика DQNAgent
    """

    # callbacks базовые (Tester/Loss/Plot и т.п.)
    base_callbacks = train_args.get("callbacks", [])
    equation_params = train_args.get("equation_params", [])
    display_every = int(train_args.get("display_every", 100))
    trajectory_logger = rl_agent_params.get("trajectory_logger")
    run_control = rl_agent_params.get("run_control")

    # Режим оценки: жадная политика, фиксированный бюджет шагов, агент не обучается
    eval_only = bool(rl_agent_params.get("eval_only", False))
    fixed_steps = int(rl_agent_params.get("fixed_steps", 0))

    # Доводка PINN после достижения порога (l2re глубже, RL-семантика не меняется)
    refine_steps = int(rl_agent_params.get("refine_steps", 0))
    refine_optimizer = rl_agent_params.get("refine_optimizer", "LBFGS")
    refine_lr = float(rl_agent_params.get("refine_lr", 0.5))
    refine_epochs = int(rl_agent_params.get("refine_epochs", 1500))

    # создаём env/agent (как раньше внутри model.py, только теперь снаружи)
    env = EnvRLOptimizer(optimizers=optimizers_dict,
                         equation_params=equation_params,
                         callbacks=None,
                         AE_model_params=AE_model_params,
                         AE_train_params=AE_train_params,
                         loss_surface_params=loss_surface_params,
                         n_save_models=rl_agent_params['n_save_models'],
                         tolerance=rl_agent_params["tolerance"])
    env.configure_chain_reward(
        alpha=rl_agent_params.get("chain_reward_alpha", 0.2),
        dense_clip=rl_agent_params.get("chain_reward_dense_clip", 5.0),
        success_bonus=rl_agent_params.get("chain_success_bonus", 10.0),
        fail_penalty=rl_agent_params.get("chain_fail_penalty", -5.0),
    )

    # These objects must be created after the first optimizer is started
    n_observation = env.observation_space
    # state_dim = np.prod(env.observation_space.shape)
    n_action = env.action_space

    rl_agent = DQNAgent(n_observation,
                        n_action,
                        optimizer_dict=optimizers_dict,
                        memory_size=rl_agent_params["rl_buffer_size"],
                        gamma=rl_agent_params["gamma"],
                        lr=rl_agent_params["lr"],
                        device=device,
                        batch_size=rl_agent_params["rl_batch_size"],
                        n_transitions_reinit = rl_agent_params["n_transitions_reinit"],
                        exp = rl_agent_params["exp"],
                        model_snapshot_dir=f"{save_path}/rl_model_snapshots",
                        ablation=rl_agent_params.get("ablation", "none"))

    # init state (как у тебя в model.py: нулевые карты)
    state_shape = get_state_shape(loss_surface_params)
    def zero_state():
        z = torch.zeros(state_shape, device=device)
        return {"loss_total": z.clone(), "loss_oper": z.clone(), "loss_bnd": z.clone()}
    
    if eval_only:
        # Буфер не нужен: агент не обучается, только исполняет жадную политику
        print("🎯 eval-only: буфер не загружается, агент не обучается, ε-исследование выключено.")
        rl_agent.greedy_only = True
        if not rl_agent_params.get("resume_checkpoint"):
            raise RuntimeError("eval-only требует чекпоинт агента (--resume-from auto/путь).")
    elif rl_agent_params.get("buffer_dir"):
        # Локальный буфер (экспортированный из Comet заранее) — COMET_API_KEY не нужен
        rl_agent.replay_buffer = collect_all_local_transitions(rl_agent.replay_buffer, buffer_dir=rl_agent_params["buffer_dir"],
                                                               max_exps_last=rl_agent_params.get("n_exps", 500), tolerance = rl_agent_params["tolerance"],
                                                               prev_tol= rl_agent_params["prev_tol"], new_tol = rl_agent_params["new_tol"],
                                                               use_log_state=rl_agent_params["log_key"],
                                                               proj_name=rl_agent_params["proj_name"],
                                                               reset_success_done_to_failure=rl_agent_params.get("reset_success_done_to_failure", False),
                                                               recompute_chain_rewards=rl_agent_params.get("recompute_chain_rewards", True),
                                                            set_reward_from_next_loss=rl_agent_params.get("set_reward_from_next_loss", True))
    else:
        rl_agent.replay_buffer = collect_all_comet_transitions(rl_agent.replay_buffer, max_exps_last=rl_agent_params.get("n_exps", 500), tolerance = rl_agent_params["tolerance"],
                                                           prev_tol= rl_agent_params["prev_tol"], use_tol = rl_agent_params["use_tol"], new_tol = rl_agent_params["new_tol"],
                                                           use_log_state=rl_agent_params["log_key"],
                                                           proj_name=rl_agent_params["proj_name"],
                                                           reset_success_done_to_failure=rl_agent_params.get("reset_success_done_to_failure", False),
                                                           recompute_chain_rewards=rl_agent_params.get("recompute_chain_rewards", True),
                                                        set_reward_from_next_loss=rl_agent_params.get("set_reward_from_next_loss", True))
    # if backup_params is not None:
    #     optim_state, params_state = load_rl_agent_from_comet(backup_params["experiment_key"], map_location=device_type())
    #     rl_agent.model_optim.load_state_dict(optim_state)
    #     rl_agent.model_params.load_state_dict(params_state)

    # --- Продолжение с чекпоинта прошлой сессии (Kaggle-цепочка) ---
    resume_checkpoint = rl_agent_params.get("resume_checkpoint")
    resumed = False
    if resume_checkpoint:
        if resume_checkpoint["kind"] == "final":
            rl_agent.load_final_model(resume_checkpoint["path"])
        else:
            rl_agent.load_head_snapshots(
                resume_checkpoint["optim"],
                resume_checkpoint["params"],
                steps_done=resume_checkpoint.get("steps_done"),
            )
        resumed = True
        print(f"⏯  Продолжаем обучение с запуска {resume_checkpoint.get('tag')} — "
              "оффлайн-претрен пропущен.")

        # Приоритеты PER под загруженные веса: если чекпоинт их не принёс,
        # пересчитываем оффлайн (иначе буфер остаётся с плоскими дефолтами)
        if (not eval_only and getattr(rl_agent, "needs_priority_recalc", False)
                and len(rl_agent.replay_buffer) > 0):
            from RL.rl_utils.per_offline import recalc_all_priorities_batched
            print("⏯  Оффлайн-пересчёт приоритетов PER под загруженные веса...")
            recalc_all_priorities_batched(rl_agent, batch_size=rl_agent.recalc_batch_size)
            rl_agent.needs_priority_recalc = False

    # --- Оффлайн-претрен агента чисто на буфере, до онлайн-траекторий ---
    # Компромисс при малом бюджете онлайн-шагов: агент сначала выучивается на
    # оффлайн-транзишенах, онлайн-часть стартует с осмысленной политикой.
    # При продолжении с чекпоинта не нужен: агент уже обучен прошлой сессией.
    offline_pretrain_steps = 0 if resumed else int(rl_agent_params.get("offline_pretrain_steps", 0))
    offline_pretrain_iters = int(rl_agent_params.get("offline_pretrain_iters", 5))
    if offline_pretrain_steps > 0:
        if len(rl_agent.replay_buffer) < rl_agent.batch_size:
            raise RuntimeError(
                "Not enough replay transitions for offline pretraining: "
                f"{len(rl_agent.replay_buffer)} < batch_size({rl_agent.batch_size})"
            )

        print(
            "\nStarting offline RL pretraining: "
            f"steps={offline_pretrain_steps}, iters_per_step={offline_pretrain_iters}."
        )
        offline_start_time = time.time()
        for step in range(1, offline_pretrain_steps + 1):
            if run_control is not None and run_control.should_stop():
                print(f"⏹  Оффлайн-претрен прерван на шаге {step}: {run_control.stop_reason}")
                break
            loss_optim, loss_param = rl_agent.optim_(iters=offline_pretrain_iters)
            rl_agent.steps_done += 1
            if not loss_optim or not loss_param:
                raise RuntimeError(
                    f"Offline pretraining stopped at step {step}: no updates were made."
                )
            print(
                f"[offline {step}/{offline_pretrain_steps}] "
                f"optim_loss_mean={np.mean(loss_optim):.6f}, "
                f"param_loss_mean={np.mean(loss_param):.6f}"
            )

        print(f"Offline pretraining took {time.time() - offline_start_time:.1f}s.")
        _print_offline_greedy_chain_diagnostic(rl_agent)
        rl_agent.reinit_target()
        rl_agent.transition_counter = 0

    idx_traj = 0

    for traj in range(train_args["n_trajectories"]):
        # Одна траектория идёт 1–2 часа, поэтому проверяем бюджет времени и
        # запрос на остановку ДО начала новой — иначе запуск не закончится сам.
        if run_control is not None and run_control.should_stop():
            print(f"\n⏹  Останавливаем набор траекторий: {run_control.stop_reason}. "
                  f"Завершено траекторий: {traj}.")
            break

        # реинициализация сети на новую траекторию
        if hasattr(model.net, "apply"):
            model.net.apply(reinit_torch_weights)

        # сброс трекинга лучшей точки траектории (l2re_min в CSV)
        if base_callbacks and hasattr(base_callbacks[0], "reset_trajectory_tracking"):
            base_callbacks[0].reset_trajectory_tracking()

        # сброс локальных переменных траектории
        state = zero_state()
        prev_reward = -1.0
        last_opt = None
        same_opt_streak = 0
        optimizers_history = []
        rl_penalty = 0
        total_reward = 0.0
        trajectory_transitions = []
        trajectory_losses = []
        final_done = 0
        trajectory_actions = []
        trajectory_start_time = time.time()

        print('\n############################################################################' +
        f'\nStarting trajectory {idx_traj + 1}/{rl_agent_params["n_trajectories"]} ' +
        'with a new initial point.')


        for t in itertools.count():

            # Не начинаем новый чанк оптимизатора после запроса на остановку:
            # один чанк LBFGS может идти больше получаса.
            if run_control is not None and run_control.stop_requested and t > 0:
                print(f"\n⏹  Обрываем траекторию на шаге {t}: {run_control.stop_reason}.")
                break

            # --- agent action ---
            action, action_raw, is_model = rl_agent.select_action(state)
            agent_step = rl_agent.steps_done
            action_raw[2]['epochs'] = action_raw[1]
            action_raw = (action_raw[0], action_raw[2])

            # штраф за повтор оптимизатора (как у тебя было)
            if last_opt == action["type"]:
                same_opt_streak += 1
            else:
                same_opt_streak = 0
            last_opt = action["type"]

            if is_model:
                print("Action by model")
            else:
                print("Action by epsilon-greedy")
            print(f"\naction = {action}")

            trajectory_actions.append(action)

            # --- compile optimizer for this chunk ---
            chunk_iters = int(action["epochs"])
            torch_opt = _build_torch_optimizer(action["type"], model.net.parameters(), action)


            model.compile(torch_opt, loss_weights=loss_weights)
            model.optimizer = torch_opt
            saver = ModelSaverCallback(total_iterations=chunk_iters, n_save_models=train_args['n_save_models'])
            callbacks = list(base_callbacks) + [saver]

            print('\n===========================================================================\n' +
                    f'\nRL agent training: step {t + 1}.'
                    f'\nTime: {datetime.datetime.now()}.'
                    f'\nUsing optimizer: {action["type"]} for {action["epochs"]} epochs.'
                    f'\nTotal Reward = {total_reward}.\n')

            model.train(
                iterations=chunk_iters,
                display_every=display_every,
                callbacks=callbacks,
                model_save_path=save_path,
                save_model=False,
            )

            solver_models = saver.saved_models
            tester_callback = callbacks[0]
            rmse = tester_callback.rmse
            b_rmse = tester_callback.brmse
            train_loss = _extract_weighted_train_loss(model)
            transition_ready = False

            if np.isfinite(train_loss):
                print(f"Operator RMSE: {rmse}, Boundary RMSE: {b_rmse}")
                print(f"Weighted train loss: {train_loss}")

                env.solver_models = solver_models
                env.reward_params = {
                    "loss": train_loss,
                }
                env.rl_penalty = rl_penalty

                optimizers_history.append(action["type"])
                print(f'\nPassed optimizer {action["type"]}.')


                env.set_step_context(
                    prev_state=state,
                    step_i=t,
                    same_opt_streak=same_opt_streak,
                    is_model=is_model,
                    rl_opt_step=rl_agent.opt_step,
                    prev_reward_scalar=None if prev_reward == -1 else prev_reward,
                )

                next_state, reward_shaped, done, info = env.step()
                final_done = done
                transition_ready = True

                # prev_reward — теперь просто хранит reward_scalar из info
                prev_reward = info["reward_scalar"]

                trajectory_transitions.append({
                    "state": state,
                    "next_state": next_state,
                    "solver_models": _serialize_solver_models(solver_models),
                    "action_raw": action_raw,
                    "agent_step": agent_step,
                    "done": done,
                    "opt_model_i": info["opt_model_i"],
                    "reward_scalar": float(info["reward_scalar"]),
                    "old_reward_model": float(reward_shaped.item()),
                    "current_loss": float(train_loss),
                })
                trajectory_losses.append(float(train_loss))

                total_reward += float(reward_shaped.item())
            else:
                done = -1
                final_done = -1
                reward_shaped = torch.tensor(-10.0, device=device)
                info = {
                    "reward_scalar": 0.0,
                    "opt_model_i": -1,
                }
                print(f"Operator RMSE: {rmse}, Boundary RMSE: {b_rmse}. Stopping trajectory with done = -1.")
                print(f"Weighted train loss: {train_loss}. Stopping trajectory with done = -1.")

            print(f'\nCurrent reward after {action["type"]} optimizer: {info["reward_scalar"]}.\n'
                    f'Reward after taking prev reward and penalty: {reward_shaped}\n'
                    f'Total reward after using {", ".join(optimizers_history)} '
                    f'{"optimizers" if len(optimizers_history) > 1 else "optimizer"}: {total_reward}.\n'
                    f'\ndone = {done}')
            
            if not eval_only and len(rl_agent.replay_buffer) >= rl_agent_params["agent_min_buffer"]:
                rl_agent.optim_(iters=rl_agent_params["agent_update_iters"])

            # callbacks.callbacks[1].save_every = self.t
            # env.render()
            if transition_ready:
                state = next_state
            # Фиксированный бюджет шагов (режим оценки): игнорируем done=1,
            # каждая траектория получает одинаковое число решений агента
            if fixed_steps > 0 and (t + 1) >= fixed_steps:
                print(f"\n⏹  Достигнут фиксированный бюджет {fixed_steps} шагов — конец траектории.")
                break
            if done == 1:
                break
            elif done == 0:
                if t == 10:
                    rl_penalty = -1
            elif done == -1:
                rl_penalty = 0
                break

        if len(trajectory_transitions) > 0:
            if final_done == -1:
                trajectory_transitions[-1]["done"] = -1

            trajectory_rewards = env.compute_chain_rewards(
                losses=trajectory_losses,
                done=final_done,
            )

            assert len(trajectory_rewards) == len(trajectory_transitions), (
                f"len(trajectory_rewards)={len(trajectory_rewards)} != "
                f"len(trajectory_transitions)={len(trajectory_transitions)}"
            )

            chain_total_reward = 0.0

            for tr, chain_reward in zip(trajectory_transitions, trajectory_rewards):
                chain_reward = float(chain_reward)
                chain_total_reward += chain_reward

                rl_agent.push_memory((
                    tr["state"],
                    tr["next_state"],
                    tr["action_raw"],
                    chain_reward,
                    tr["done"],
                    chain_reward,
                    tr["opt_model_i"],
                ))

                step_done = tr["agent_step"]

                try:
                    file_path = os.path.join(output_dir, f'transitions_{step_done}.pt')

                    entry = {
                        'state': tr["state"],
                        'next_state': tr["next_state"],
                        'solver_models': tr["solver_models"],
                        'action': tr["action_raw"],
                        'reward': tr["reward_scalar"],
                        'current_loss': tr["current_loss"],
                        'done': tr["done"],
                        'reward_model_raw': chain_reward,
                        'reward_model': chain_reward,
                        'reward_scheme': "env_chain_reward",
                        'old_reward_model': tr["old_reward_model"],
                        'opt_model_i': tr["opt_model_i"],
                    }
                    torch.save(entry, file_path)

                    rl_agent_params['exp'].log_asset(
                        file_path,
                        file_name=f"entry_step_{step_done}.pt",
                        step=step_done,
                        overwrite=True
                    )

                except Exception as e:
                    print(e)

            print(
                f"\nPushed trajectory with env chain rewards. "
                f"steps={len(trajectory_transitions)}, "
                f"final_loss={trajectory_losses[-1]}, "
                f"final_done={final_done}, "
                f"chain_total_reward={chain_total_reward}\n"
            )

            if len(rl_agent.replay_buffer) >= rl_agent_params["agent_min_buffer"]:
                rl_agent.optim_(iters=rl_agent_params["agent_update_iters"])

        # Снимок метрик "на пороге" — ДО доводки: основные колонки CSV должны
        # отражать момент остановки траектории, иначе сравнение сломается.
        tester = base_callbacks[0] if base_callbacks else None
        at_stop = {
            "mse_op": getattr(tester, "mse", float("nan")),
            "mse_bnd": getattr(tester, "bc_mse", float("nan")),
            "l2re_op": getattr(tester, "l2re", float("nan")),
            "l2re_bnd": getattr(tester, "bc_l2re", float("nan")),
        }

        # --- Доводка PINN после достижения порога (вариант A) ---
        # RL-семантика не меняется: транзишены доводки в буфер не пишутся,
        # награды уже посчитаны; глубже обучается только сеть PINN.
        refined = None
        if refine_steps > 0 and final_done == 1 and trajectory_actions:
            print(f"\n🔧 Доводка после порога: {refine_steps} x {refine_optimizer}"
                  f"(lr={refine_lr}, epochs={refine_epochs})")
            for _ in range(refine_steps):
                if run_control is not None and run_control.should_stop():
                    break
                refine_opt = _build_torch_optimizer(
                    refine_optimizer, model.net.parameters(),
                    {"params": {"lr": refine_lr}})
                model.compile(refine_opt, loss_weights=loss_weights)
                model.optimizer = refine_opt
                model.train(iterations=refine_epochs, display_every=display_every,
                            callbacks=list(base_callbacks), model_save_path=save_path,
                            save_model=False)
            tester = base_callbacks[0] if base_callbacks else None
            if tester is not None:
                refined = {
                    "l2re_op": getattr(tester, "l2re", float("nan")),
                    "l2re_bnd": getattr(tester, "bc_l2re", float("nan")),
                }
                refined["l2re"] = float(np.hypot(refined["l2re_op"], refined["l2re_bnd"])) \
                    if np.isfinite(refined["l2re_bnd"]) else float(refined["l2re_op"])
                print(f"🔧 l2re после доводки: {refined['l2re']:.6g}")

        # --- строка метрик по завершённой траектории ---
        if trajectory_logger is not None and trajectory_actions:
            traj_l2re_min = getattr(tester, "traj_l2re_min", float("inf"))
            trajectory_logger.log_trajectory(
                mse_op=at_stop["mse_op"],
                mse_bnd=at_stop["mse_bnd"],
                l2re_op=at_stop["l2re_op"],
                l2re_bnd=at_stop["l2re_bnd"],
                elapsed_s=time.time() - trajectory_start_time,
                actions=trajectory_actions,
                trajectory_index=traj,
                extra={
                    "ablation": rl_agent_params.get("ablation", "none"),
                    "done": final_done,
                    "steps": len(trajectory_actions),
                    "final_loss": trajectory_losses[-1] if trajectory_losses else float("nan"),
                    "total_reward": total_reward,
                    # Ошибка обучения самого агента — её и сравниваем между режимами
                    "agent_loss_optim": rl_agent.last_optim_loss_mean,
                    "agent_loss_param": rl_agent.last_param_loss_mean,
                    "agent_td_abs": getattr(rl_agent, "last_td_abs_mean", float("nan")),
                    "agent_q_abs": getattr(rl_agent, "last_q_abs_mean", float("nan")),
                    "agent_tr_drop_frac": getattr(rl_agent, "last_tr_drop_frac", float("nan")),
                    # лучшая точка траектории (включая доводку, если была)
                    "l2re_min": traj_l2re_min if traj_l2re_min != float("inf") else float("nan"),
                    # l2re после пост-доводки (вариант A); пусто без неё
                    "l2re_refined": refined["l2re"] if refined else "",
                },
            )

        if done == 1:
            idx_traj += 1

    # --- финальная модель агента: её забирает HF-логгер при последней синхронизации ---
    final_model_dir = os.path.join(save_path, "model")
    rl_agent.save_final_model(
        final_model_dir,
        metadata={
            "ablation": rl_agent_params.get("ablation", "none"),
            "n_trajectories": train_args["n_trajectories"],
            "successful_trajectories": idx_traj,
            "buffer_size": len(rl_agent.replay_buffer),
            "steps_done": rl_agent.steps_done,
        },
    )


def train_process_rl(data, save_path, device, seed, rl_agent_params):
    """
    drop-in replacement for train_process(...)
    """
    # hooked = HookedStdout(f"{save_path}/log.txt")
    # sys.stdout = hooked
    # sys.stderr = HookedStdout(f"{save_path}/logerr.txt", sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dde.config.set_default_float("float32")
    # dde.config.set_random_seed(seed)

    payload = dill.loads(data)

    # совместимость: если раньше data был (get_model, train_args)
    # теперь можно передать (get_model, train_args, rl_payload)
    if len(payload) == 2:
        get_model, train_args = payload
        model = get_model()
        model.train(**train_args, model_save_path=save_path)
        return

    get_model, train_args, optimizers, AE_model_params, AE_train_params,  loss_surface_params = payload
    model, loss_weights = get_model()
    # rl_payload структура:
    # {
    #   "train_args": {...},
    #   "optimizers_dict": {...},
    #   "equation_params": ...,
    #   "AE_model_params": ...,
    #   "AE_train_params": ...,
    #   "loss_surface_params": ...
    # }

    run_deepxde_rl_training(
        model=model,
        loss_weights=loss_weights,
        train_args=train_args,
        rl_agent_params=rl_agent_params,
        optimizers_dict=optimizers,
        AE_model_params=AE_model_params,
        AE_train_params=AE_train_params,
        loss_surface_params=loss_surface_params,
        save_path=save_path,
    )
