#!/usr/bin/env python
"""
Online L2RE evaluation of an offline-trained agent using the AUTHORS' state
construction (autoencoder latent loss surface), not a reconstruction.

State pipeline (verified against branch rlpinn_pde_tolerance):
  1. every chunk saves n_save_models copies of the solver net (ModelSaverCallback)
  2. VisualizationModel trains an autoencoder over the flat concat of all
     state_dict tensors of those copies (input_dim = 40801 for FNN 100*5;
     layers_AE[0]/[2] are dead code, only the 125-wide hidden matters)
  3. PlotLossSurface decodes a 26x26 grid of the 2D latent (x_range
     [-1.25, 1.25, 25] -> step 0.1 -> 26 points) back to weights and evaluates
     PDE/BC losses at each point with a SECOND model factory
  4. save_equation_loss_surface(log_key=True) applies sign(x)*log1p(|x|)
  5. delta channel (env logic, replicated here):
         d = total_now - total_prev
         delta = sign(d)*log1p(|d|); delta /= max|delta|; clamp(-1, 1)
     channel order: loss_total, loss_oper, loss_bnd, delta

Action space (verified in RL/rl_algorithms.py: i2opt = dict key order):
    0 Adam   lr [1e-2, 1e-3, 1e-4]  epochs [100, 1000, 2500]
    1 LBFGS  lr [1, 5e-1, 1e-1]     epochs [100, 500, 1000]   (max_iter=10!)
    2 PSO    lr [0, 1e-3, 1e-4]     epochs [100, 200, 300]
    index = opt*9 + lr*3 + epochs

Comet/gym/dill are NOT imported: only landscape_visualization._aux is used and
the env's delta/reward logic is replicated locally.
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

OUT_REPO = "danil-e/pinnacle-optuna-db"
GRID_RANGE = [-1.25, 1.25, 25]

ACTION_TABLE = []
for _opt, _lrs, _eps in [
    ("Adam", [1e-2, 1e-3, 1e-4], [100, 1000, 2500]),
    ("LBFGS", [1.0, 5e-1, 1e-1], [100, 500, 1000]),
    ("PSO", [0.0, 1e-3, 1e-4], [100, 200, 300]),
]:
    for _lr in _lrs:
        for _ep in _eps:
            ACTION_TABLE.append((_opt, _lr, _ep))
assert len(ACTION_TABLE) == 27

AE_MODEL_PARAMS = dict(
    mode="NN", num_of_layers=3, layers_AE=[991, 125, 15], num_models=None,
    from_last=False, prefix="model-", every_nth=1, grid_step=0.1, d_max_latent=2,
    anchor_mode="circle", rec_weight=10000.0, anchor_weight=0.0, lastzero_weight=0.0,
    polars_weight=0.0, wellspacedtrajectory_weight=0.0, gridscaling_weight=0.0,
)
LOSS_TYPES = ["loss_total", "loss_oper", "loss_bnd"]


def build_optimizer(opt_name, lr, net):
    """Mirrors rl_trainer._build_torch_optimizer (note LBFGS max_iter=10)."""
    from deepxde.optimizers.config import set_PSO_options

    if opt_name == "Adam":
        return torch.optim.Adam(net.parameters(), lr=lr)
    if opt_name == "LBFGS":
        return torch.optim.LBFGS(net.parameters(), lr=lr,
                                 line_search_fn="strong_wolfe", max_iter=10)
    if opt_name == "PSO":
        set_PSO_options(lr=lr)
        return "PSO"
    raise ValueError(opt_name)


def load_agent(model_file):
    from offline_rl import QNet

    if model_file and os.path.exists(model_file):
        path = model_file
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(OUT_REPO, model_file, repo_type="dataset")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["variant"] == "cnx_smdp":       # другой класс сети (advanced_agents)
        from advanced_agents import SmdpAgent
        net = SmdpAgent(torch.device("cpu"))
        net.net.load_state_dict(ckpt["state_dict"])
        net.net.eval()
    else:
        net = QNet(ckpt["variant"], torch.device("cpu"))
        net.model.load_state_dict(ckpt["state_dict"])
        net.model.eval()
    return net, ckpt["mean"], ckpt["std"], ckpt["variant"]


def pick_action(agent, state, mean, std, variant):
    # deepxde на GPU ставит default device = cuda, поэтому вход надо создавать
    # на том же устройстве, где лежат веса агента
    dev = next(agent.model.parameters()).device
    x = torch.as_tensor((state[None] - mean) / std, device=dev).float()
    with torch.no_grad():
        q = (agent.q_cvar(x) if variant in ("cnn_qrdqn", "cnx_cql_qr")
             else agent.q_scalar(x))
    return int(q.argmax(1).item())


def landscape_stall_prob(state, trig):
    """Ландшафтный триггер коллеги в сильнейшей форме: логистическая модель,
    обученная на здоровом буфере (landscape_trigger.json, AUC 0.657)."""
    m = state[0].astype(np.float64)
    g = np.gradient(m)
    f = np.array([m.std(),
                  np.mean(np.abs(g[0])) + np.mean(np.abs(g[1])),
                  np.abs(np.gradient(g[0])[0] + np.gradient(g[1])[1]).mean(),
                  m[13, 13] - m.min(),
                  np.percentile(m, 95) - np.percentile(m, 5),
                  m[13, 13], m.mean()])
    z = (f - np.array(trig["mean"])) / np.array(trig["std"])
    return float(1.0 / (1.0 + np.exp(-(z @ np.array(trig["coef"]) + trig["intercept"]))))


class BoostedNet(torch.nn.Module):
    """u = base + eps*boost, база заморожена (arXiv 2307.08934)."""

    def __init__(self, base, boost, eps):
        super().__init__()
        self.base, self.boost, self.eps = base, boost, float(eps)
        self.regularizer = None
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.eps * self.boost(x)


CKPT_DIR = "rl_arch/online_env_ckpt"


def save_ckpt(name, payload):
    """Состояние прогона (веса PINN + цепочка + карты) — чтобы следующий кернел
    продолжил с того же места: сессия Kaggle живёт 12 ч, а бюджет 31000 эпох при
    мелких действиях агента требует больше."""
    local = f"ckpt_{name}.pt"
    torch.save(payload, local)
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        return
    from huggingface_hub import upload_file
    for attempt in range(3):
        try:
            upload_file(path_or_fileobj=local, path_in_repo=f"{CKPT_DIR}/{name}.pt",
                        repo_id=OUT_REPO, repo_type="dataset", token=tok,
                        commit_message=f"ckpt {name}")
            return
        except Exception as e:
            print(f"ckpt upload retry {attempt}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))


def load_ckpt(name):
    local = f"ckpt_{name}.pt"
    if os.path.exists(local):
        return torch.load(local, map_location="cpu", weights_only=False)
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(OUT_REPO, f"{CKPT_DIR}/{name}.pt", repo_type="dataset")
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"чекпоинта нет ({type(e).__name__}) — старт с нуля", flush=True)
        return None


def build_state(raw, prev_raw):
    """Replicates EnvRLOptimizer.step delta logic + channel order."""
    tot = raw["loss_total"].detach().float().cpu()
    op = raw["loss_oper"].detach().float().cpu()
    bn = raw["loss_bnd"].detach().float().cpu()
    if prev_raw is None:
        delta = torch.zeros_like(tot)
    else:
        d = tot - prev_raw["loss_total"].detach().float().cpu()
        delta = torch.sign(d) * torch.log1p(torch.abs(d))
        delta = delta / (delta.abs().max() + 1e-6)
        delta = delta.clamp(-1, 1)
    return torch.stack([tot, op, bn, delta]).numpy().astype(np.float32)


def run_seed(seed, args, progress_cb=None):
    from experiments.chain_eval.pde_registry import build_get_model
    from src.utils.callbacks import TesterCallback, ModelSaverCallback
    from landscape_visualization._aux.visualization_model import VisualizationModel
    from landscape_visualization._aux.plot_loss_surface import PlotLossSurface
    from landscape_visualization._aux.early_stopping_plot import EarlyStopping

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dde.config.set_default_float("float32")
    torch.set_default_dtype(torch.float32)
    dde.config.set_random_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # two independent factories: one trains, one is overwritten 676x per state
    get_model = build_get_model(args.pde, args.hidden_layers)
    get_model_rec = build_get_model(args.pde, args.hidden_layers)

    model, loss_weights = get_model()

    def reinit(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    model.net.apply(reinit)

    agent = mean = std = variant = None
    if args.policy == "agent":
        agent, mean, std, variant = load_agent(args.model_file)

    vm = VisualizationModel(device=dev, path_to_plot_model=None,
                            path_to_trajectories=None, **AE_MODEL_PARAMS)

    rng = np.random.default_rng(seed * 613 + 7)
    save_dir = os.path.join(args.save_dir, f"{args.pde}_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    # initial state: zero maps (rl_trainer.zero_state)
    state = np.zeros((4, 26, 26), dtype=np.float32)
    prev_raw = None
    trig = (json.load(open(os.path.join(SCRIPT_DIR, "landscape_trigger.json")))
            if args.boost_trigger.startswith("landscape") else None)
    boosted, loss_hist, p_hist = False, [], []
    boost_layers, boost_eps = None, None
    spent, chain, t0 = 0, [], time.time()
    rmse = brmse = l2re_op = l2re_bnd = float("inf")

    ckpt_name = getattr(args, "_ckpt_name", None)
    ck = load_ckpt(ckpt_name) if (args.resume and ckpt_name) else None
    if ck is not None:
        if ck.get("boosted"):
            boost = dde.nn.FNN(ck["boost_layers"], "tanh", "Glorot normal").float()
            pde_ref = getattr(model, "pde", None)
            model = dde.Model(model.data, BoostedNet(model.net, boost, ck["boost_eps"]))
            model.pde = pde_ref
            boosted, boost_layers, boost_eps = True, ck["boost_layers"], ck["boost_eps"]
        model.net.load_state_dict(ck["net"])
        spent, chain, state = ck["spent"], ck["chain"], ck["state"]
        prev_raw = ck["prev_raw"]
        loss_hist, p_hist = ck["loss_hist"], ck["p_hist"]
        rmse, brmse, l2re_op, l2re_bnd = ck["metrics"]
        print(f"[seed {seed}] докатка: spent={spent}/{args.budget}, шагов уже {len(chain)}",
              flush=True)

    def dump_ckpt():
        if not ckpt_name:
            return
        save_ckpt(ckpt_name, dict(
            net={k: v.detach().cpu() for k, v in model.net.state_dict().items()},
            spent=spent, chain=chain, state=state,
            prev_raw=({k: v.detach().cpu() for k, v in prev_raw.items()}
                      if prev_raw is not None else None),
            loss_hist=loss_hist, p_hist=p_hist, boosted=boosted,
            boost_layers=boost_layers, boost_eps=boost_eps,
            metrics=(rmse, brmse, l2re_op, l2re_bnd)))

    while spent < args.budget:
        if args.policy == "agent":
            a = pick_action(agent, state, mean, std, variant)
        elif args.policy == "fixed":
            a = args.fixed_action
        else:
            a = int(rng.integers(0, 27))
        opt_name, lr, epochs = ACTION_TABLE[a]
        epochs = min(epochs, args.budget - spent)
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
        l2re_op = float(getattr(tester, "l2re", float("inf")))
        l2re_bnd = float(getattr(tester, "bc_l2re", float("inf")))
        print(f"[seed {seed}] step {len(chain)}: {opt_name} lr={lr} ep={epochs} "
              f"spent={spent}/{args.budget} l2re={math.hypot(l2re_op, l2re_bnd):.4e}", flush=True)
        if progress_cb is not None:
            progress_cb(dict(seed=seed, policy=args.policy, pde=args.pde, partial=True,
                             l2re=math.hypot(l2re_op, l2re_bnd), l2re_op=l2re_op,
                             l2re_bnd=l2re_bnd, rmse=rmse, brmse=brmse, spent=spent,
                             budget=args.budget, n_steps=len(chain), chain=chain,
                             elapsed_s=round(time.time() - t0, 1)))

        if not (np.isfinite(rmse) or np.isfinite(brmse)):
            print(f"[seed {seed}] non-finite metrics — stopping (done=-1)", flush=True)
            break
        if spent >= args.budget:
            break

        # ---- решение о бустинге: проверка идеи статьи с ландшафтным триггером ----
        loss_hist.append(float(np.asarray(model.train_state.loss_train, dtype=float).sum()))
        if args.boost_trigger != "none" and not boosted and spent < args.budget:
            fire = False
            if trig is not None and len(chain) > 1:
                p_stall = landscape_stall_prob(state, trig)
                p_hist.append(p_stall)
                if args.boost_trigger == "landscape":
                    fire = p_stall > args.boost_threshold
                else:  # landscape_peak: момент, который ландшафт считает самым застойным
                    fire = (len(p_hist) > args.boost_warmup
                            and p_stall >= max(p_hist[:-1]))
                print(f"[seed {seed}] ландшафтный триггер: P(застой)={p_stall:.3f} "
                      f"(max={max(p_hist):.3f})", flush=True)
            elif args.boost_trigger == "midpoint":
                fire = spent >= args.budget // 2
            elif args.boost_trigger == "plateau" and len(loss_hist) >= 4:
                r4 = loss_hist[-4:]
                fire = (max(r4) - min(r4)) < 1e-3 * abs(r4[-1] + 1e-12)
            if fire:
                eps = float(np.sqrt(max(float(np.asarray(model.train_state.loss_train,
                                                         dtype=float)[0]), 1e-30)))
                layers = [model.net.linears[0].in_features, 64, 64, 64,
                          model.net.linears[-1].out_features]
                boost = dde.nn.FNN(layers, "tanh", "Glorot normal").float()
                pde_ref = getattr(model, "pde", None)
                model = dde.Model(model.data, BoostedNet(model.net, boost, eps))
                model.pde = pde_ref
                boosted, boost_layers, boost_eps = True, layers, eps
                chain.append(["BOOST", eps, 0])
                print(f"[seed {seed}] БУСТИНГ подключён (eps={eps:.3e}, триггер "
                      f"{args.boost_trigger})", flush=True)

        # ---- state: AE over the saved trajectory, then latent loss surface ----
        # армам none/plateau/midpoint карты не нужны: политика fixed, а триггер
        # смотрит на историю лосса или на счётчик эпох. Обучение PINN от этого не
        # зависит (проверено: l2re совпадает до последнего знака), а построение
        # карт — это ~90% времени прогона
        if not args.no_state:
            t_ae = time.time()
            ae = vm.train(args.ae_lr, args.ae_cosine_patience, args.ae_epochs, 100,
                          args.ae_batch, True, finetune_AE_model=False,
                          callbacks=[EarlyStopping(patience=args.ae_es_patience)],
                          solver_models=saver.saved_models)
            t_ae = time.time() - t_ae
            t_srf = time.time()
            pls = PlotLossSurface(solver_models=saver.saved_models, AE_model=ae,
                                  dde_pde_model=get_model_rec, x_range=GRID_RANGE,
                                  batch_size=args.ae_batch, loss_types=LOSS_TYPES,
                                  loss_name="loss_total", path_to_plot_model=None,
                                  path_to_trajectories=None, img_dir="")
            raw = pls.save_equation_loss_surface(log_key=True)
            t_srf = time.time() - t_srf
            print(f"[seed {seed}] state built: AE {t_ae:.1f}s (epochs={args.ae_epochs}), "
                  f"surface {t_srf:.1f}s", flush=True)
            state = build_state(raw, prev_raw)
            prev_raw = raw
            del pls, ae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # чекпоинт ставим ПОСЛЕ построения состояния: веса PINN и карты должны
        # соответствовать друг другу, иначе докатка стартует с рассогласования
        if ckpt_name and args.ckpt_every and len(chain) % args.ckpt_every == 0:
            dump_ckpt()
        if args.hours and (time.time() - t0) / 3600.0 >= args.hours:
            dump_ckpt()
            print(f"[seed {seed}] лимит времени ({args.hours} ч): чекпоинт на "
                  f"spent={spent}/{args.budget}, продолжит следующий кернел", flush=True)
            return dict(seed=seed, policy=args.policy, pde=args.pde, unfinished=True,
                        l2re=math.hypot(l2re_op, l2re_bnd), spent=spent,
                        budget=args.budget, n_steps=len(chain),
                        elapsed_s=round(time.time() - t0, 1))

    l2re = math.hypot(l2re_op, l2re_bnd)
    return dict(seed=seed, policy=args.policy, pde=args.pde, l2re=l2re,
                boost_trigger=args.boost_trigger, boosted=boosted, rmse=rmse,
                brmse=brmse, l2re_op=l2re_op, l2re_bnd=l2re_bnd, budget=args.budget,
                n_steps=len(chain), chain=chain, ae_epochs=args.ae_epochs,
                elapsed_s=round(time.time() - t0, 1))


def upload(row, name):
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        print("no HF token — result printed only", flush=True)
        return
    import io
    from huggingface_hub import upload_file
    for attempt in range(3):
        try:
            upload_file(path_or_fileobj=io.BytesIO(json.dumps(row, indent=1).encode()),
                        path_in_repo=f"rl_arch/online_env/{name}.json",
                        repo_id=OUT_REPO, repo_type="dataset", token=tok,
                        commit_message=f"rl_arch online_env {name}")
            print(f"uploaded rl_arch/online_env/{name}.json", flush=True)
            return
        except Exception as e:
            print(f"upload retry {attempt}: {e}", flush=True)
            time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["agent", "random", "fixed"])
    ap.add_argument("--fixed-action", type=int, default=4,
                    help="Индекс повторяемого действия для --policy fixed "
                         "(4 = Adam lr 1e-3 x 1000 эпох)")
    ap.add_argument("--model-file", default=None)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--pde", default="poissonboltzmann2d")
    ap.add_argument("--hidden-layers", default="100*5")
    ap.add_argument("--budget", type=int, default=31000)
    ap.add_argument("--n-save-models", type=int, default=10)
    ap.add_argument("--display-every", type=int, default=100)
    ap.add_argument("--ae-epochs", type=int, default=10000)
    ap.add_argument("--ae-lr", type=float, default=5e-4)
    ap.add_argument("--ae-batch", type=int, default=32)
    ap.add_argument("--ae-cosine-patience", type=int, default=1200)
    ap.add_argument("--ae-es-patience", type=int, default=4000)
    ap.add_argument("--save-dir", default="runs_rl_online")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--boost-trigger", default="none",
                    choices=["none", "landscape", "landscape_peak", "plateau", "midpoint"])
    ap.add_argument("--boost-warmup", type=int, default=8,
                    help="Сколько шагов копить историю до срабатывания peak-триггера")
    ap.add_argument("--boost-threshold", type=float, default=0.5)
    ap.add_argument("--hours", type=float, default=0.0,
                    help="мягкий лимит по времени: сохранить чекпоинт и выйти "
                         "(0 = без лимита). Сессия Kaggle живёт 12 ч")
    ap.add_argument("--resume", action="store_true",
                    help="продолжить прогон с чекпоинта (локального или с HF)")
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="как часто страховочно сохранять чекпоинт, в шагах цепочки")
    ap.add_argument("--no-state", action="store_true",
                    help="не строить карты ландшафта (AE + поверхность лоссов). "
                         "Допустимо только для policy=fixed/random с триггерами "
                         "none/plateau/midpoint — там карты не используются")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.no_state and (args.policy == "agent"
                          or args.boost_trigger.startswith("landscape")):
        sys.exit("--no-state несовместим с policy=agent и ландшафтными триггерами: "
                 "им нужны карты")
    if args.smoke:  # только как дефолты — явные флаги не перезаписываем
        given = set(a.split("=")[0] for a in sys.argv[1:] if a.startswith("--"))
        if "--budget" not in given: args.budget = 300
        if "--ae-epochs" not in given: args.ae_epochs = 50
        if "--n-save-models" not in given: args.n_save_models = 3

    tag = args.tag or (os.path.basename(args.model_file).replace(".pt", "")
                       if args.model_file else "random")
    for seed in [int(s) for s in args.seeds.split(",")]:
        name = f"{args.pde}_{tag}_seed{seed}"
        args._ckpt_name = name
        cb = None if args.smoke else (lambda r, n=name: upload(r, n))
        row = run_seed(seed, args, progress_cb=cb)
        row["smoke"] = args.smoke
        print(json.dumps({k: v for k, v in row.items() if k != "chain"}), flush=True)
        if row.get("unfinished"):
            # итог не заливаем: строка выглядела бы завершённой, а бюджет не выбран
            print(f"[seed {seed}] прогон не закончен — нужен ещё один кернел с --resume",
                  flush=True)
            continue
        if not args.smoke:
            upload(row, f"{args.pde}_{tag}_seed{seed}")


if __name__ == "__main__":
    main()
