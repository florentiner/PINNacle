#!/usr/bin/env python
"""
ПОЛНЫЙ онлайн-цикл PINN-PELINE с обучением агента (как в rl_trainer.py авторов),
но с возможностью подменить архитектуру Q-сети (CNN / ConvNeXt, DQN / CQL /
дуэлинговая голова).

Отличие от online_eval_env.py: там политика заморожена и меряется одна цепочка;
здесь агент УЧИТСЯ по ходу — переходы копятся в ЛОКАЛЬНОМ буфере, после каждой
цепочки веса PINN переинициализируются, и так до исчерпания лимита времени.

Метрика: l2re последней ЗАВЕРШЁННОЙ цепочки. Оборванная по времени цепочка не
учитывается — её частичный результат хранится отдельно (last_partial), поэтому
срез сессии Kaggle никогда не портит итог.

Состояние строится их же кодом (автоэнкодер над траекторией весов -> сетка 26x26
в латенте -> лоссы -> sign*log1p -> канал delta), см. online_eval_env.

  python experiments/rl_arch/online_train_env.py --variant convnext_dqn \
      --pde poissonboltzmann2d --hours 11 --seed 42
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)
os.environ.setdefault("DDEBACKEND", "pytorch")

import torch  # noqa: E402
import deepxde as dde  # noqa: E402

from online_eval_env import (ACTION_TABLE, AE_MODEL_PARAMS, GRID_RANGE, LOSS_TYPES,  # noqa: E402
                             OUT_REPO, build_optimizer, build_state)
from offline_rl import QNet  # noqa: E402

EPS_START, EPS_END, EPS_DECAY = 0.5, 0.05, 50   # значения авторов (rl_algorithms.py)
GAMMA = 0.9                                      # их rl_agent_params


class LocalBuffer:
    """Локальный буфер переходов. На HF ничего не уходит — только итоги цепочек."""

    def __init__(self, capacity=10000):
        self.cap = capacity
        self.s, self.a, self.r, self.s2, self.d = [], [], [], [], []

    def push(self, s, a, r, s2, d):
        if len(self.s) >= self.cap:
            for arr in (self.s, self.a, self.r, self.s2, self.d):
                arr.pop(0)
        self.s.append(s); self.a.append(a); self.r.append(r)
        self.s2.append(s2); self.d.append(d)

    def __len__(self):
        return len(self.s)

    def sample(self, n, device):
        idx = np.random.randint(0, len(self.s), size=min(n, len(self.s)))
        t = lambda arr, dt: torch.as_tensor(np.array([arr[i] for i in idx]), dtype=dt, device=device)
        return (t(self.s, torch.float32), t(self.a, torch.long), t(self.r, torch.float32),
                t(self.s2, torch.float32), t(self.d, torch.float32))


def agent_update(net, buf, opt, batch_size, iters, variant, cql_alpha=1.0):
    import torch.nn.functional as F
    dev = next(net.model.parameters()).device
    for _ in range(iters):
        s, a, r, s2, d = buf.sample(batch_size, dev)
        q = net.q_scalar(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            a2 = net.q_scalar(s2).argmax(1)
            q2 = (net.q_target(s2).mean(-1) if variant in ("cnn_qrdqn", "cnx_cql_qr")
                  else net.q_target(s2)).gather(1, a2[:, None]).squeeze(1)
            tgt = r + GAMMA * (1 - d) * q2
        loss = F.mse_loss(q, tgt)
        if "cql" in variant:
            qs = net.q_scalar(s)
            loss = loss + cql_alpha * (torch.logsumexp(qs, 1) - qs.gather(1, a[:, None]).squeeze(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        net.soft_update()
    return float(loss.item())


def agent_update_mixed(net, off_buf, on_buf, opt, half, variant, cql_alpha=1.0,
                       max_bellman=False):
    """RLPD: симметричная выборка — половина батча офлайн, половина онлайн."""
    import torch.nn.functional as F
    dev = next(net.model.parameters()).device
    parts = [b.sample(half, dev) for b in (off_buf, on_buf)]
    s, a, r, s2, d = [torch.cat([p[i] for p in parts], 0) for i in range(5)]
    q = net.q_scalar(s).gather(1, a[:, None]).squeeze(1)
    with torch.no_grad():
        a2 = net.q_scalar(s2).argmax(1)
        q2 = (net.q_target(s2).mean(-1) if variant in ("cnn_qrdqn", "cnx_cql_qr")
              else net.q_target(s2)).gather(1, a2[:, None]).squeeze(1)
        tgt = (torch.maximum(r, GAMMA * (1 - d) * q2) if max_bellman
               else r + GAMMA * (1 - d) * q2)
    loss = F.mse_loss(q, tgt)
    if "cql" in variant:
        qs = net.q_scalar(s)
        loss = loss + cql_alpha * (torch.logsumexp(qs, 1) - qs.gather(1, a[:, None]).squeeze(1)).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    net.soft_update()
    return float(loss.item())


def behaviour_update(behaviour, opt, buf, batch_size, iters, device):
    """Модель поведения для маски BCQ: что вообще встречалось в данных."""
    import torch.nn.functional as F
    loss = 0.0
    for _ in range(iters):
        s, a, _, _, _ = buf.sample(batch_size, device)
        loss = F.cross_entropy(behaviour(s.flatten(1)), a)
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())


def upload(row, name):
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        return
    import io
    from huggingface_hub import upload_file
    for attempt in range(3):
        try:
            upload_file(path_or_fileobj=io.BytesIO(json.dumps(row, indent=1).encode()),
                        path_in_repo=f"rl_arch/online_train/{name}.json",
                        repo_id=OUT_REPO, repo_type="dataset", token=tok,
                        commit_message=f"rl_arch online_train {name}")
            return
        except Exception as e:
            print(f"upload retry {attempt}: {e}", flush=True)
            time.sleep(8 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="convnext_dqn")
    ap.add_argument("--pde", default="poissonboltzmann2d")
    ap.add_argument("--hidden-layers", default="100*5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hours", type=float, default=11.0, help="лимит по времени (сессия Kaggle)")
    ap.add_argument("--max-chain-steps", type=int, default=12, help="K_max из статьи")
    ap.add_argument("--tolerance", type=float, default=0.0, help="досрочный конец цепочки по ошибке")
    ap.add_argument("--n-save-models", type=int, default=10)
    ap.add_argument("--ae-epochs", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-buffer", type=int, default=32)
    ap.add_argument("--update-iters", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warm-start", default=None, help="чекпоинт офлайн-агента (опционально)")
    ap.add_argument("--rlpd", action="store_true",
                    help="RLPD (arXiv 2302.02948): симметричная выборка 50/50 из офлайнового "
                         "и онлайнового буферов + высокое число обновлений на шаг среды")
    ap.add_argument("--rlpd-subdir", default="poisson3d_complexgeometry",
                    help="какой офлайн-буфер подмешивать")
    ap.add_argument("--rlpd-utd", type=int, default=8,
                    help="обновлений на шаг среды (update-to-data ratio)")
    ap.add_argument("--rlpd-full", action="store_true",
                    help="RLPD целиком: к симметричной выборке добавить LayerNorm в "
                         "критике и ансамбль из N критиков со случайным подмножеством "
                         "размера M для таргета (в статье N=10, M=2)")
    ap.add_argument("--rlpd-ensemble", type=int, default=10)
    ap.add_argument("--rlpd-subset", type=int, default=2)
    ap.add_argument("--qwm", action="store_true",
                    help="QWM (arXiv 2608.17163): мировая модель поверх Q-обучения — "
                         "одношаговый поиск по всем 27 действиям при выборе действия. "
                         "Модель предобучается на офлайновом буфере (--rlpd-subdir) и "
                         "дообучается онлайн; критик учится только на реальных переходах")
    ap.add_argument("--qwm-pretrain", type=int, default=4000,
                    help="шагов предобучения мировой модели на офлайновом буфере")
    ap.add_argument("--qwm-alpha", type=float, default=0.5,
                    help="вес критика в комбинированной оценке (Eq. 9: 0.5)")
    ap.add_argument("--boot-heads", type=int, default=0,
                    help="bootstrapped DQN: K голов со случайными prior-сетями, одна "
                         "голова на цепочку вместо eps-дрожания (Osband 2016/2018)")
    ap.add_argument("--risk-tau", type=float, default=0.0,
                    help="risk-seeking: действовать по среднему квантилей выше tau "
                         "(нужен вариант cnx_qrdqn); 0 = выкл")
    ap.add_argument("--max-bellman", action="store_true",
                    help="max-reward Bellman (To the Max, ICML 2024): таргет "
                         "max(r, gamma*maxQ') вместо суммы")
    ap.add_argument("--sil", action="store_true",
                    help="self-imitation: доп. лосс к возвратам топ-5 цепочек")
    ap.add_argument("--go-explore", action="store_true",
                    help="Go-Explore: архив лучших срезов цепочек (веса PINN + карта), "
                         "продолжение с лучшего среза вместо старта с нуля")
    ap.add_argument("--preload", default=None,
                    help="схема авторов: предзаполнить ОБЩИЙ буфер прошлым опытом "
                         "(rl_trainer.collect_all_comet_transitions) и дальше "
                         "сэмплировать равномерно, а не половиной батча как RLPD")
    ap.add_argument("--preload-max", type=int, default=500,
                    help="сколько прошлых переходов подгружать (у авторов max_exps_last=500)")
    ap.add_argument("--bcq", action="store_true",
                    help="discrete BCQ: модель поведения по собранным переходам + "
                         "маска поддержки при выборе действия")
    ap.add_argument("--bcq-threshold", type=float, default=0.3,
                    help="порог маски: оставляем действия с p >= threshold * max p")
    ap.add_argument("--display-every", type=int, default=100)
    ap.add_argument("--save-dir", default="runs_rl_train")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.rlpd_full:
        args.rlpd = True        # полному RLPD нужен тот же офлайновый буфер
    if args.smoke:  # только как дефолты — явные флаги не перезаписываем
        given = set(x.split("=")[0] for x in sys.argv[1:] if x.startswith("--"))
        if "--hours" not in given: args.hours = 0.05
        if "--ae-epochs" not in given: args.ae_epochs = 40
        if "--n-save-models" not in given: args.n_save_models = 3
        if "--max-chain-steps" not in given: args.max_chain_steps = 2

    from experiments.chain_eval.pde_registry import build_get_model
    from src.utils.callbacks import TesterCallback, ModelSaverCallback
    from landscape_visualization._aux.visualization_model import VisualizationModel
    from landscape_visualization._aux.plot_loss_surface import PlotLossSurface
    from landscape_visualization._aux.early_stopping_plot import EarlyStopping

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dde.config.set_default_float("float32")
    torch.set_default_dtype(torch.float32)
    dde.config.set_random_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    get_model = build_get_model(args.pde, args.hidden_layers)
    get_model_rec = build_get_model(args.pde, args.hidden_layers)

    if args.boot_heads:
        from advanced_agents import BootQNet
        net = BootQNet(dev, n_heads=args.boot_heads)
        print(f"bootstrapped DQN: {args.boot_heads} голов с prior-сетями, "
              f"eps-дрожание отключено", flush=True)
    elif args.rlpd_full:
        from advanced_agents import RlpdCritic
        net = RlpdCritic(dev, n_critics=args.rlpd_ensemble, subset=args.rlpd_subset)
        print(f"RLPD целиком: {args.rlpd_ensemble} критиков с LayerNorm, "
              f"таргет по минимуму из {args.rlpd_subset} случайных, "
              f"{net.n_params()/1e6:.2f} млн параметров", flush=True)
    else:
        net = QNet(args.variant, dev)
    mean = std = None
    if args.warm_start and os.path.exists(args.warm_start):
        ck = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        net.model.load_state_dict(ck["state_dict"])
        mean, std = ck["mean"], ck["std"]
        print(f"тёплый старт из {args.warm_start}", flush=True)
    q_opt = torch.optim.Adam(net.params(), lr=args.lr)

    # discrete BCQ: модель поведения по собранным переходам + маска поддержки на
    # выборе действия. Без неё вариант cnx_bcq в онлайне неотличим от обычного
    # DQN — вся суть метода именно в маске
    behaviour = beh_opt = None
    if args.bcq:
        import torch.nn as nn
        behaviour = nn.Sequential(nn.Linear(4 * 26 * 26, 256), nn.LayerNorm(256),
                                  nn.GELU(), nn.Linear(256, 27)).to(dev)
        beh_opt = torch.optim.Adam(behaviour.parameters(), lr=1e-3)

    vm = VisualizationModel(device=str(dev), path_to_plot_model=None,
                            path_to_trajectories=None, **AE_MODEL_PARAMS)
    buf = LocalBuffer()
    off_buf = None
    if args.rlpd or args.qwm:
        # офлайновая половина батча: тот же буфер, на котором учатся офлайн-агенты
        from offline_rl import load_episodes, episodes_to_arrays, split_by_episode
        od = episodes_to_arrays(load_episodes(None, args.rlpd_subdir),
                                fix_next_state=(args.rlpd_subdir == "poisson_boltzmann_2d"))
        otr, _ = split_by_episode(od)
        oidx = np.where(otr)[0]
        om = od["S"][oidx].mean(axis=(0, 2, 3), keepdims=True)
        os_ = od["S"][oidx].std(axis=(0, 2, 3), keepdims=True) + 1e-6
        off_buf = LocalBuffer(capacity=len(oidx) + 10)
        for i in oidx:
            off_buf.push(((od["S"][i][None] - om) / os_)[0], int(od["A"][i]), float(od["R"][i]),
                         ((od["S2"][i][None] - om) / os_)[0], float(od["D"][i]))
        if mean is None:
            mean, std = om, os_
        print(f"RLPD: офлайновый буфер {len(off_buf)} переходов из {args.rlpd_subdir}, "
              f"UTD={args.rlpd_utd}", flush=True)
    if args.preload:
        # схема авторов: прошлые переходы кладутся в ТОТ ЖЕ буфер, из которого потом
        # идёт равномерная выборка — отличие от RLPD, где офлайн держится отдельно
        # и всегда занимает ровно половину батча
        from offline_rl import load_episodes, episodes_to_arrays, split_by_episode
        od = episodes_to_arrays(load_episodes(None, args.preload),
                                fix_next_state=(args.preload == "poisson_boltzmann_2d"))
        otr, _ = split_by_episode(od)
        oidx = np.where(otr)[0][-args.preload_max:]
        om = od["S"][oidx].mean(axis=(0, 2, 3), keepdims=True)
        os_ = od["S"][oidx].std(axis=(0, 2, 3), keepdims=True) + 1e-6
        for i in oidx:
            buf.push(((od["S"][i][None] - om) / os_)[0], int(od["A"][i]), float(od["R"][i]),
                     ((od["S2"][i][None] - om) / os_)[0], float(od["D"][i]))
        if mean is None:
            mean, std = om, os_
        print(f"предзагрузка по схеме авторов: {len(buf)} переходов из {args.preload}", flush=True)

    wm = wm_opt = None
    if args.qwm:
        from advanced_agents import QwmWorldModel, wm_update, qwm_select
        wm = QwmWorldModel().to(dev)
        wm_opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        wl = 0.0
        for i in range(args.qwm_pretrain):
            wl = wm_update(wm, wm_opt, [off_buf], 128, dev)
        print(f"QWM: мировая модель предобучена ({args.qwm_pretrain} шагов, "
              f"итоговый лосс {wl:.4f}), поиск по Eq.9 с D=1", flush=True)

    rng = np.random.default_rng(args.seed)
    tag = args.tag or f"{args.variant}_{args.pde}_seed{args.seed}"
    save_dir = os.path.join(args.save_dir, tag)
    os.makedirs(save_dir, exist_ok=True)

    sil_buf = None
    if args.sil:
        from advanced_agents import SilBuffer
        sil_buf = SilBuffer(top_k=5, gamma=GAMMA)
    archive = []          # Go-Explore: [(dict E, err, w, state, raw, spent, steps)]

    t_start = time.time()
    deadline = t_start + args.hours * 3600
    steps_done = 0
    chains = []              # завершённые цепочки: (l2re, шагов, эпох)
    last_partial = None      # оборванная цепочка — отдельно, в итог не идёт

    traj = 0
    while time.time() < deadline:
        traj += 1
        model, loss_weights = get_model()

        def reinit(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        model.net.apply(reinit)

        state = np.zeros((4, 26, 26), dtype=np.float32)
        prev_raw = None
        spent, chain, done = 0, [], 0
        truncated = False   # True только если ПРЕРВАЛИ цепочку посередине по лимиту
        l2re = float("inf")
        prev_err = None
        start_step, resumed_from = 0, None
        chain_trans = []    # (s_норм, a, r) текущей цепочки — для self-imitation

        if args.boot_heads:
            net.active = int(rng.integers(0, net.K))   # одна голова на всю цепочку

        if args.go_explore and archive and len(chains) >= 2 and rng.random() < 0.6:
            # ранг-взвешенный выбор: лучшие срезы чаще (Go-Explore 'return')
            w = np.array([1.0 / (i + 1) for i in range(len(archive))]); w /= w.sum()
            e = archive[int(rng.choice(len(archive), p=w))]
            model.net.load_state_dict({k: v.to(dev) for k, v in e["w"].items()})
            state = e["state"].copy()
            prev_raw = ({k: v.clone() for k, v in e["raw"].items()}
                        if e["raw"] is not None else None)
            spent, start_step, prev_err = e["spent"], e["steps"], e["err"]
            l2re, resumed_from = e["E"], e["E"]
            print(f"[траектория {traj}] Go-Explore: продолжаю срез l2re={e['E']:.4f} "
                  f"(шаг {e['steps']}, {e['spent']} эпох)", flush=True)

        for step in range(start_step, args.max_chain_steps):
            if time.time() >= deadline:
                truncated = True
                last_partial = dict(l2re=l2re, steps=len(chain), epochs=spent, trajectory=traj)
                print(f"[траектория {traj}] лимит времени — цепочка оборвана на шаге {len(chain)}", flush=True)
                break

            # eps-greedy как у авторов
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
            steps_done += 1
            if rng.random() < eps and not args.boot_heads:
                a = int(rng.integers(0, 27))
                how = "случайно"
            else:
                x = torch.as_tensor(((state[None] - mean) / std) if mean is not None else state[None],
                                    device=dev).float()
                if wm is not None:
                    # QWM: одношаговый поиск вместо argmax Q (Eq. 9, D=1)
                    a = qwm_select(net, wm, x, gamma=GAMMA, alpha=args.qwm_alpha)
                    how = "поиск QWM"
                elif args.boot_heads:
                    with torch.no_grad():
                        a = int(net.q_head(x, net.active).argmax(1).item())
                    how = f"голова {net.active}"
                elif args.risk_tau > 0:
                    from advanced_agents import q_upper
                    with torch.no_grad():
                        a = int(q_upper(net, x, args.risk_tau).argmax(1).item())
                    how = f"квантиль>{args.risk_tau}"
                else:
                  with torch.no_grad():
                    q = net.q_scalar(x)
                    if behaviour is not None:
                        # BCQ: выбираем только среди действий, которые модель
                        # поведения считает правдоподобными для этих данных
                        p = behaviour(x.flatten(1)).softmax(-1)
                        keep = p >= args.bcq_threshold * p.max(-1, keepdim=True).values
                        q = q.masked_fill(~keep, -1e9)
                    a = int(q.argmax(1).item())
                  how = "по модели"
            opt_name, lr, epochs = ACTION_TABLE[a]

            optimizer = build_optimizer(opt_name, lr, model.net)
            model.compile(optimizer, loss_weights=loss_weights)
            tester = TesterCallback(log_every=args.display_every)
            saver = ModelSaverCallback(total_iterations=epochs, n_save_models=args.n_save_models)
            model.train(iterations=epochs, display_every=args.display_every,
                        callbacks=[tester, saver], model_save_path=save_dir, save_model=False)
            spent += epochs
            chain.append([opt_name, lr, epochs])

            rmse = float(getattr(tester, "rmse", float("inf")))
            brmse = float(getattr(tester, "brmse", float("inf")))
            l2re = math.hypot(float(getattr(tester, "l2re", float("inf"))),
                              float(getattr(tester, "bc_l2re", float("inf"))))
            if not (np.isfinite(rmse) or np.isfinite(brmse)):
                done = -1
                print(f"[траектория {traj}] метрики разошлись — обрыв", flush=True)
                break

            # награда авторов: absolute, E = rmse + brmse; r = E_t - E_{t+1}
            err = (rmse if np.isfinite(rmse) else 0.0) + (brmse if np.isfinite(brmse) else 0.0)
            reward = 0.0 if prev_err is None else (prev_err - err)
            prev_err = err
            done = 1 if (args.tolerance > 0 and err < args.tolerance) else 0

            # следующее состояние — их пайплайном
            ae = vm.train(5e-4, 1200, args.ae_epochs, 100, args.batch_size, True,
                          finetune_AE_model=False, callbacks=[EarlyStopping(patience=4000)],
                          solver_models=saver.saved_models)
            pls = PlotLossSurface(solver_models=saver.saved_models, AE_model=ae,
                                  dde_pde_model=get_model_rec, x_range=GRID_RANGE,
                                  batch_size=args.batch_size, loss_types=LOSS_TYPES,
                                  loss_name="loss_total", path_to_plot_model=None,
                                  path_to_trajectories=None, img_dir="")
            raw = pls.save_equation_loss_surface(log_key=True)
            next_state = build_state(raw, prev_raw)
            prev_raw = raw
            del pls, ae
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if mean is None:      # нормировка по первому состоянию (агент с нуля)
                mean = next_state.mean(axis=(1, 2), keepdims=True)[None]
                std = next_state.std(axis=(1, 2), keepdims=True)[None] + 1e-6
            s_norm = ((state[None] - mean) / std)[0]
            buf.push(s_norm, a, reward,
                     ((next_state[None] - mean) / std)[0], float(done == 1))
            if args.sil:
                chain_trans.append((s_norm.copy(), a, float(reward)))
            state = next_state

            if args.go_explore and len(chain) + start_step < args.max_chain_steps:
                # срез: веса PINN + карта + сырые лоссы — чтобы можно было вернуться
                entry = dict(E=l2re, err=prev_err, spent=spent,
                             steps=start_step + len(chain), state=next_state.copy(),
                             raw={k: v.detach().cpu().clone() for k, v in prev_raw.items()},
                             w={k: v.detach().cpu().clone()
                                for k, v in model.net.state_dict().items()})
                if len(archive) < 8 or l2re < archive[-1]["E"]:
                    archive.append(entry)
                    archive.sort(key=lambda x: x["E"])
                    del archive[8:]

            if behaviour is not None and len(buf) >= 8:
                behaviour_update(behaviour, beh_opt, buf, args.batch_size, 2, dev)

            if wm is not None:
                # дообучение мировой модели на свежих переходах (критик как учился
                # на реальных данных, так и учится — модель мира его не подменяет)
                wbufs = [off_buf] + ([buf] if len(buf) >= 8 else [])
                for _ in range(4):
                    wm_update(wm, wm_opt, wbufs, 64, dev)

            if args.rlpd and off_buf is not None:
                # RLPD: половина батча из офлайна, половина из онлайна; много обновлений
                iters = args.rlpd_utd
                half = max(1, args.batch_size // 2)
                ql = 0.0
                for _ in range(iters):
                    src = buf if len(buf) >= 8 else off_buf
                    if args.boot_heads:
                        from advanced_agents import boot_rlpd_update
                        ql = boot_rlpd_update(net, off_buf, src, q_opt, half, GAMMA)
                    elif args.risk_tau > 0:
                        from advanced_agents import rlpd_qr_update
                        ql = rlpd_qr_update(net, off_buf, src, q_opt, half, GAMMA)
                    elif args.rlpd_full:
                        from advanced_agents import rlpd_update
                        ql = rlpd_update(net, off_buf, src, q_opt, half, GAMMA)
                    elif len(buf) >= 8:
                        ql = agent_update_mixed(net, off_buf, buf, q_opt, half, args.variant,
                                                max_bellman=args.max_bellman)
                    else:
                        ql = agent_update(net, off_buf, q_opt, args.batch_size, 1, args.variant)
                if args.sil and sil_buf is not None and sil_buf.chains:
                    from advanced_agents import sil_update
                    sil_update(net, sil_buf, q_opt, args.batch_size, dev)
                print(f"[траектория {traj}] шаг {len(chain)}: {opt_name} lr={lr} ep={epochs} ({how}, eps={eps:.2f}) "
                      f"l2re={l2re:.4e} reward={reward:+.4f} буфер={len(buf)}+{len(off_buf)} q-loss={ql:.4f}", flush=True)
            elif len(buf) >= args.min_buffer:
                ql = agent_update(net, buf, q_opt, args.batch_size, args.update_iters, args.variant)
                print(f"[траектория {traj}] шаг {len(chain)}: {opt_name} lr={lr} ep={epochs} ({how}, eps={eps:.2f}) "
                      f"l2re={l2re:.4e} reward={reward:+.4f} буфер={len(buf)} q-loss={ql:.4f}", flush=True)
            else:
                print(f"[траектория {traj}] шаг {len(chain)}: {opt_name} lr={lr} ep={epochs} ({how}) "
                      f"l2re={l2re:.4e} reward={reward:+.4f} буфер={len(buf)}", flush=True)
            if done == 1:
                break
        else:
            done = done or 0

        # цепочка засчитывается, если дошла до конца (K_max / допуск / расхождение),
        # даже если лимит времени истёк во время последнего шага; не засчитывается
        # только та, которую прервали ПОСЕРЕДИНЕ
        if chain and not truncated:
            if args.sil and sil_buf is not None and np.isfinite(l2re):
                sil_buf.add_chain(l2re, chain_trans)
            rec = dict(l2re=l2re, steps=len(chain), epochs=spent, done=done, chain=chain)
            if resumed_from is not None:
                rec["resumed_from"] = resumed_from
            chains.append(rec)
            row = dict(variant=args.variant, pde=args.pde, seed=args.seed,
                       trajectories=len(chains), buffer=len(buf),
                       l2re_last_complete=chains[-1]["l2re"],
                       l2re_best=min(c["l2re"] for c in chains),
                       l2re_first=chains[0]["l2re"],
                       chains=[{k: v for k, v in c.items() if k != "chain"} for c in chains],
                       last_chain=chains[-1]["chain"], last_partial=last_partial,
                       elapsed_h=round((time.time() - t_start) / 3600, 2))
            print(json.dumps({k: v for k, v in row.items() if k not in ("chains", "last_chain")}), flush=True)
            if not args.smoke:
                upload(row, tag)

    print(f"\nИТОГ: завершённых цепочек {len(chains)}, "
          f"l2re последней завершённой = {chains[-1]['l2re']:.4e}" if chains else "\nИТОГ: ни одной завершённой цепочки",
          flush=True)


if __name__ == "__main__":
    main()
