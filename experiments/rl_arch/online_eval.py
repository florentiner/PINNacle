#!/usr/bin/env python
"""
Online L2RE evaluation of an offline-trained agent policy on poissonboltzmann2d.

Reconstructed rlpinn environment inside chain_eval infrastructure:
  state   4x26x26 maps over a fixed pair of filter-normalized random directions:
          log1p of (weighted total / operator / boundary) loss on a 26x26 grid,
          plus delta = clip(map_now - map_prev, -1, 1) of the total channel
          (calibrated against the buffer: channels 0.004..12.3 = log1p(loss),
          delta in [-1,1] with saturated tails).
  actions 27 discrete, decoding supplied by the authors:
          Adam   lr {1e-2, 1e-3, 1e-4} x epochs {100, 1000, 2500}
          L-BFGS lr {1.0, 0.5, 0.1}   x epochs {100, 500, 1000}
          PSO    lr {0, 1e-3, 1e-4}   x epochs {100, 200, 300}
  budget  31000 epochs total (chain_eval convention); final metric = l2re
          = hypot(l2re_op, l2re_bnd) via TesterCallback, as everywhere else.

Policies: --policy agent (checkpoint from HF rl_arch/models/, greedy over Q;
CVaR@0.25 for quantile variants) or --policy random (behavioral baseline).

Caveat (stated in the report): the state computation is a calibrated
reconstruction, not the authors' exact code — direction sampling, grid radius
and loss subsampling may differ; agent inputs are normalized with the SAME
buffer statistics it was trained on.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("DDEBACKEND", "pytorch")

OUT_REPO = "danil-e/pinnacle-optuna-db"
BUDGET = 31_000
GRID = 26
RADIUS = 0.5  # relative filter-normalized radius

ACTION_TABLE = []
for oi, (opt, lrs, eps) in enumerate([
    ("Adam", [1e-2, 1e-3, 1e-4], [100, 1000, 2500]),
    ("LBFGS", [1.0, 0.5, 0.1], [100, 500, 1000]),
    ("PSO", [0.0, 1e-3, 1e-4], [100, 200, 300]),
]):
    for li, lr in enumerate(lrs):
        for ei, ep in enumerate(eps):
            ACTION_TABLE.append((opt, lr, ep))
assert len(ACTION_TABLE) == 27


def loss_components(model, torch):
    """(total, oper, bnd) weighted train losses at current weights.
    NOTE: must run grad-ENABLED — PDE residuals need input derivatives."""
    losses = model.outputs_losses_train(
        model.train_state.X_train, model.train_state.y_train)[1]
    losses = torch.stack([l.detach() for l in losses])
    w = getattr(model, "_le_weights", None)
    if w is not None:
        losses = losses * w
    types = getattr(model, "_le_types", ["pde"] * len(losses))
    total = float(losses.sum())
    oper = float(sum(l for l, t in zip(losses, types) if t == "pde"))
    bnd = float(sum(l for l, t in zip(losses, types) if t != "pde"))
    return total, oper, bnd


class LandscapeProbe:
    """Fixed pair of filter-normalized directions; 26x26 log1p loss maps."""

    def __init__(self, model, torch, seed):
        self.torch = torch
        self.model = model
        g = torch.Generator(device="cpu").manual_seed(seed * 7919 + 13)
        self.dirs = []
        for _ in range(2):
            d = []
            for p in model.net.parameters():
                v = torch.randn(p.shape, generator=g, device="cpu").to(p.device)
                v = v / (v.norm() + 1e-12) * (p.detach().norm() + 1e-12)
                d.append(v)
            self.dirs.append(d)
        self.prev_total = None
        self.alphas = np.linspace(-RADIUS, RADIUS, GRID)

    def maps(self):
        torch = self.torch
        params = [p for p in self.model.net.parameters()]
        backup = [p.detach().clone() for p in params]
        m_tot = np.zeros((GRID, GRID), dtype=np.float32)
        m_op = np.zeros_like(m_tot)
        m_bn = np.zeros_like(m_tot)
        for i, a in enumerate(self.alphas):
            for j, b in enumerate(self.alphas):
                with torch.no_grad():
                    for p, p0, d1, d2 in zip(params, backup, *self.dirs):
                        p.copy_(p0 + a * d1 + b * d2)
                t, o, bn = loss_components(self.model, torch)
                m_tot[i, j] = math.log1p(max(t, 0.0))
                m_op[i, j] = math.log1p(max(o, 0.0))
                m_bn[i, j] = math.log1p(max(bn, 0.0))
        with torch.no_grad():
            for p, p0 in zip(params, backup):
                p.copy_(p0)
        delta = (np.clip(m_tot - self.prev_total, -1, 1)
                 if self.prev_total is not None else np.zeros_like(m_tot))
        self.prev_total = m_tot
        return np.stack([m_tot, m_op, m_bn, delta])


def load_agent(ckpt_path, torch):
    sys.path.insert(0, SCRIPT_DIR)
    from offline_rl import QNet
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    variant = ckpt["variant"]
    net = QNet(variant, torch.device("cpu"))
    net.model.load_state_dict(ckpt["state_dict"])
    net.model.eval()
    return net, ckpt["mean"], ckpt["std"], variant


def pick_action(net, state, mean, std, variant, torch):
    dev = next(net.model.parameters()).device
    x = torch.as_tensor((state[None] - mean) / std, device=dev).float()
    with torch.no_grad():
        q = net.q_cvar(x) if variant in ("cnn_qrdqn", "cnx_cql_qr") else net.q_scalar(x)
    return int(q.argmax(1).item())


def run_episode(seed, policy, ckpt_path, args):
    budget = args.budget
    import torch
    import deepxde as dde
    from experiments.chain_eval.pde_registry import build_get_model
    from experiments.chain_eval.run_chain_pde import build_stage_optimizer
    from src.utils.callbacks import TesterCallback

    dde.config.set_default_float("float32")
    torch.set_default_dtype(torch.float32)
    dde.config.set_random_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, loss_weights = build_get_model("poissonboltzmann2d", "100*5")()

    def reinit(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    model.net.apply(reinit)

    # annotate loss structure for the probe
    pde_obj = model.pde
    model._le_weights = torch.as_tensor(loss_weights, dtype=torch.float32)
    model._le_types = [c.get("type", "pde") if isinstance(c, dict) else "pde"
                      for c in getattr(pde_obj, "loss_config", [])]
    if len(model._le_types) != len(loss_weights):
        model._le_types = ["pde"] * len(loss_weights)

    agent = None
    mean = std = variant = None
    if policy == "agent":
        agent, mean, std, variant = load_agent(ckpt_path, torch)

    rng = np.random.default_rng(seed * 613 + 7)
    # compile once so train_state is populated for the probe
    model.compile(torch.optim.Adam(model.net.parameters(), lr=1e-3),
                  loss_weights=loss_weights)
    model.train(iterations=1, display_every=10**9)

    probe = LandscapeProbe(model, torch, seed)

    spent = 0
    chain_log = []
    while spent < budget:
        state = probe.maps()
        if policy == "agent":
            a = pick_action(agent, state, mean, std, variant, torch)
        else:
            a = int(rng.integers(0, 27))
        opt_name, lr, epochs = ACTION_TABLE[a]
        epochs = min(epochs, budget - spent)
        stage = {"optimizer": opt_name, "lr": lr, "epochs": epochs}
        if opt_name == "PSO" and lr == 0.0:
            chain_log.append([opt_name, lr, 0])   # explicit no-op action
            spent += epochs
            continue
        optimizer = build_stage_optimizer(stage, model.net, lbfgs_max_iter=1)
        model.compile(optimizer, loss_weights=loss_weights)
        model.train(iterations=epochs, display_every=10**9)
        spent += epochs
        chain_log.append([opt_name, lr, epochs])

    # final metric pass: 1 step with lr=0 so the tester logs exactly once
    tester = TesterCallback(log_every=1)
    save_dir = os.path.join("/tmp", f"online_eval_seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    model.compile(torch.optim.Adam(model.net.parameters(), lr=0.0),
                  loss_weights=loss_weights)
    model.train(iterations=1, display_every=1, callbacks=[tester],
                model_save_path=save_dir, save_model=False)
    metrics = dict(rmse=float(tester.rmse), brmse=float(tester.brmse),
                   l2re_op=float(tester.l2re), l2re_bnd=float(tester.bc_l2re))
    l2re = math.hypot(metrics["l2re_op"], metrics["l2re_bnd"])
    return dict(seed=seed, policy=policy, l2re=l2re, **metrics,
                n_stages=len(chain_log), chain=chain_log)


def upload_result(row, name):
    tok = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not tok:
        return
    import io
    from huggingface_hub import upload_file
    for attempt in range(3):
        try:
            upload_file(path_or_fileobj=io.BytesIO(json.dumps(row, indent=1).encode()),
                        path_in_repo=f"rl_arch/online/{name}.json",
                        repo_id=OUT_REPO, repo_type="dataset", token=tok,
                        commit_message=f"rl_arch online {name}")
            print(f"uploaded rl_arch/online/{name}.json", flush=True)
            return
        except Exception as e:
            print(f"upload retry {attempt}: {e}", flush=True)
            time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["agent", "random"])
    ap.add_argument("--model-file", default=None,
                    help="HF path rl_arch/models/<name>.pt (required for agent)")
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.budget = 300

    ckpt_path = None
    if args.policy == "agent":
        if args.model_file and os.path.exists(args.model_file):
            ckpt_path = args.model_file
        else:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(OUT_REPO, args.model_file, repo_type="dataset")

    for seed in [int(s) for s in args.seeds.split(",")]:
        t0 = time.time()
        row = run_episode(seed, args.policy, ckpt_path, args)
        row["elapsed_s"] = round(time.time() - t0, 1)
        row["smoke"] = args.smoke
        tag = (os.path.basename(args.model_file).replace(".pt", "")
               if args.model_file else "random")
        print(json.dumps(row), flush=True)
        if not args.smoke:
            upload_result(row, f"{tag}_{args.policy}_seed{seed}")


if __name__ == "__main__":
    main()
