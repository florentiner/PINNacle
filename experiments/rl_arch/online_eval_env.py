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
    net = QNet(ckpt["variant"], torch.device("cpu"))
    net.model.load_state_dict(ckpt["state_dict"])
    net.model.eval()
    return net, ckpt["mean"], ckpt["std"], ckpt["variant"]


def pick_action(agent, state, mean, std, variant):
    x = torch.as_tensor((state[None] - mean) / std).float()
    with torch.no_grad():
        q = (agent.q_cvar(x) if variant in ("cnn_qrdqn", "cnx_cql_qr")
             else agent.q_scalar(x))
    return int(q.argmax(1).item())


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
    spent, chain, t0 = 0, [], time.time()
    rmse = brmse = l2re_op = l2re_bnd = float("inf")

    while spent < args.budget:
        a = (pick_action(agent, state, mean, std, variant) if args.policy == "agent"
             else int(rng.integers(0, 27)))
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

        # ---- state: AE over the saved trajectory, then latent loss surface ----
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

    l2re = math.hypot(l2re_op, l2re_bnd)
    return dict(seed=seed, policy=args.policy, pde=args.pde, l2re=l2re, rmse=rmse,
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
    ap.add_argument("--policy", required=True, choices=["agent", "random"])
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
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:  # только как дефолты — явные флаги не перезаписываем
        given = set(a.split("=")[0] for a in sys.argv[1:] if a.startswith("--"))
        if "--budget" not in given: args.budget = 300
        if "--ae-epochs" not in given: args.ae_epochs = 50
        if "--n-save-models" not in given: args.n_save_models = 3

    tag = args.tag or (os.path.basename(args.model_file).replace(".pt", "")
                       if args.model_file else "random")
    for seed in [int(s) for s in args.seeds.split(",")]:
        name = f"{args.pde}_{tag}_seed{seed}"
        cb = None if args.smoke else (lambda r, n=name: upload(r, n))
        row = run_seed(seed, args, progress_cb=cb)
        row["smoke"] = args.smoke
        print(json.dumps({k: v for k, v in row.items() if k != "chain"}), flush=True)
        if not args.smoke:
            upload(row, f"{args.pde}_{tag}_seed{seed}")


if __name__ == "__main__":
    main()
