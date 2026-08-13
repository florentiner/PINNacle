#!/usr/bin/env python
"""
Positive control for the multi-stage boosting idea (arXiv 2307.08934) in the
paper's own regime: smooth 1D Poisson, float64, stage-1 trained to its floor,
stage-2 = fresh net on the residual with a FRESH budget.

    -u''(x) = f(x) on [0,1],  u(0)=u(1)=0,
    u*(x) = sin(2*pi*x) + 0.5*sin(4*pi*x)

Arms (same total budget):
    single : SOAP 2*N steps, one net
    boost  : SOAP N steps -> freeze, add eps*boost (plain FNN)   -> SOAP N steps
    boostf : SOAP N steps -> freeze, add eps*boost (Fourier FNN) -> SOAP N steps

If the paper's mechanism works at all with our BoostedNet machinery, it must
show up here (their demos gain orders of magnitude exactly in this setting).
Run:  DDEBACKEND=pytorch python experiments/chain_eval/paper_control_1d.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("DDEBACKEND", "pytorch")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import deepxde as dde
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vendor.soap import SOAP

DTYPE = os.environ.get("PC_DTYPE", "float64")
dde.config.set_default_float(DTYPE)

N_STEPS = 31_000
SEEDS = [42, 43, 44]
_suf = "" if DTYPE == "float64" else f"_{DTYPE}"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"paper_control_results{_suf}.json")
ARMS_ENV = [a for a in os.environ.get("PC_ARMS", "").split(",") if a]


def u_star(x):
    return np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x)


def f_rhs(x):
    return (2 * np.pi) ** 2 * np.sin(2 * np.pi * x) + 0.5 * (4 * np.pi) ** 2 * np.sin(4 * np.pi * x)


def make_data():
    geom = dde.geometry.Interval(0, 1)

    def pde(x, u):
        du_xx = dde.grad.hessian(u, x)
        f = (2 * np.pi) ** 2 * torch.sin(2 * np.pi * x) \
            + 0.5 * (4 * np.pi) ** 2 * torch.sin(4 * np.pi * x)
        return -du_xx - f

    bc = dde.icbc.DirichletBC(geom, lambda x: 0.0, lambda x, on_b: on_b)
    return dde.data.PDE(geom, pde, bc, num_domain=256, num_boundary=2, num_test=512)


class BoostedNet(torch.nn.Module):
    def __init__(self, base, boost, eps):
        super().__init__()
        self.base, self.boost, self.eps = base, boost, float(eps)
        self.regularizer = None
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.eps * self.boost(x)


class FourierFNN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden, sigmas=(1, 10), n_feats=64):
        super().__init__()
        per = max(1, n_feats // len(sigmas))
        self.register_buffer(
            "B", torch.cat([torch.randn(in_dim, per) * s for s in sigmas], dim=1))
        self.fnn = dde.nn.FNN([2 * self.B.shape[1]] + hidden + [out_dim], "tanh", "Glorot normal")
        self.regularizer = None

    def forward(self, x):
        z = 2 * np.pi * (x @ self.B)
        return self.fnn(torch.cat([torch.sin(z), torch.cos(z)], dim=1))


def l2re(model):
    x = np.linspace(0, 1, 2001)[:, None]
    pred = model.predict(x)
    ref = u_star(x)
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def train_soap(model, steps):
    opt = SOAP(
        [p for p in model.net.parameters() if p.requires_grad],
        lr=3e-3, betas=(0.95, 0.95), precondition_frequency=2,
    )
    model.compile(opt)
    model.train(iterations=steps, display_every=5000)
    return model


def train_lbfgs(model, maxiter=20_000):
    """Full-batch L-BFGS polish to the stage's floor (paper trains every stage
    to convergence; SOAP alone stalls at ~2e-2 L2RE on this problem)."""
    dde.optimizers.config.set_LBFGS_options(maxiter=maxiter, ftol=0, gtol=1e-14)
    model.compile("L-BFGS")
    model.train(display_every=5000)
    return model


def residual_mse(model):
    losses = np.asarray(model.train_state.loss_train, dtype=float)
    return float(losses[0])


def run_arm(arm, seed):
    dde.config.set_default_float(DTYPE)  # arms may switch dtype globally (f64sw) — reset per arm
    dde.config.set_random_seed(seed)
    data = make_data()
    # *_small arms: under-parameterized stage 1 -> high floor with a SMOOTH,
    # learnable residual (the regime where the paper's stage 2 must shine).
    stage1_sizes = [1, 16, 16, 1] if arm.endswith("_small") else [1, 64, 64, 64, 1]
    net = dde.nn.FNN(stage1_sizes, "tanh", "Glorot normal")
    model = dde.Model(data, net)
    arm_base = arm[:-len("_small")] if arm.endswith("_small") else arm

    if arm_base == "single":     # SOAP-only, 2N steps (matches the Kaggle arms)
        train_soap(model, 2 * N_STEPS)
        stage1 = None
    elif arm_base in ("boost", "boostf"):   # SOAP-only boosting, budget-matched
        train_soap(model, N_STEPS)
        stage1 = l2re(model)
        model = _add_boost(model, fourier=(arm_base == "boostf"))
        train_soap(model, N_STEPS)
    elif arm_base == "single_lb":  # paper-faithful: train the stage to ITS floor
        train_soap(model, N_STEPS)
        train_lbfgs(model)
        stage1 = None
    elif arm_base in ("boost_lb", "boostf_lb"):
        train_soap(model, N_STEPS)
        train_lbfgs(model)
        stage1 = l2re(model)     # the exhausted single-net floor
        model = _add_boost(model, fourier=arm_base.startswith("boostf"))
        train_soap(model, N_STEPS)
        train_lbfgs(model)
    elif arm_base == "single_f64sw":   # precision-escalation ACTION (run under PC_DTYPE=float32)
        train_soap(model, N_STEPS)
        train_lbfgs(model)
        stage1 = l2re(model)     # float32 floor
        dde.config.set_default_float("float64")  # sets dde real + torch default dtype
        model.net.double()
        data64 = make_data()     # regenerate collocation/BC tensors in float64
        model = dde.Model(data64, model.net)
        train_soap(model, N_STEPS)
        train_lbfgs(model)
    elif arm_base == "single_restart":  # optimizer-state reset, same net, same lr
        train_soap(model, N_STEPS)
        stage1 = l2re(model)
        train_soap(model, N_STEPS)   # fresh SOAP instance = state reset
    elif arm_base == "single_lrdrop":   # classic lr-decay restart
        train_soap(model, N_STEPS)
        stage1 = l2re(model)
        opt = SOAP([q for q in model.net.parameters() if q.requires_grad],
                   lr=3e-4, betas=(0.95, 0.95), precondition_frequency=2)
        model.compile(opt)
        model.train(iterations=N_STEPS, display_every=5000)
    else:
        raise ValueError(arm)
    return {"arm": arm, "seed": seed, "l2re": l2re(model),
            "l2re_stage1": stage1}


def _add_boost(model, fourier):
    eps = float(np.sqrt(max(residual_mse(model), 1e-30)))
    if fourier:
        boost = FourierFNN(1, 1, [64, 64, 64])
    else:
        boost = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
    print(f"  boost added: eps={eps:.3e} fourier={fourier}", flush=True)
    return dde.Model(model.data, BoostedNet(model.net, boost, eps))


def main():
    results = []
    if os.path.exists(OUT):
        results = json.load(open(OUT))
    done = {(r["arm"], r["seed"]) for r in results}
    all_arms = ARMS_ENV or ["single_lb", "boost_lb", "boostf_lb", "single", "boost", "boostf",
                            "single_lb_small", "boost_lb_small", "boostf_lb_small"]
    for arm in all_arms:
        for seed in SEEDS:
            if (arm, seed) in done:
                continue
            t0 = time.time()
            r = run_arm(arm, seed)
            r["elapsed_s"] = round(time.time() - t0, 1)
            results.append(r)
            json.dump(results, open(OUT, "w"), indent=1)
            print(f"### {arm} seed={seed}: L2RE={r['l2re']:.3e} "
                  f"(stage1={r['l2re_stage1']}) [{r['elapsed_s']}s]", flush=True)
    print("\nSummary:")
    for arm in ("single_lb", "boost_lb", "boostf_lb", "single", "boost", "boostf",
                "single_lb_small", "boost_lb_small", "boostf_lb_small"):
        vals = [r["l2re"] for r in results if r["arm"] == arm]
        if vals:
            print(f"  {arm:10s} mean={np.mean(vals):.3e} min={np.min(vals):.3e}")


if __name__ == "__main__":
    main()
