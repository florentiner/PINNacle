"""StarSSE-style basin-escape experiment for the trivial-attractor study.

Adaptation of arXiv:2303.03374 (SSE/StarSSE, basin analysis via linear connectivity)
to the trivial-solution problem: the vanilla-collapsed checkpoint plays the role of
the pre-trained model sitting in a (bad) basin. Children are trained independently
from that checkpoint with one cosine cycle each (star topology); the max-LR "kick"
multiplier controls stay-vs-leave; barriers along linear interpolation to the
checkpoint diagnose basin membership; same-kick pairs are averaged into soups.
Inverted-sign hypothesis vs the paper: since our basin is the WRONG minimum, leaving
it is the goal, not the failure mode.

No SOTA machinery: same architecture, same vanilla loss, only the LR schedule and
weight-space moves.
"""
import argparse
import json
import os
import time

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde

from src.utils import forensic
from src.utils.trivial_guard import TrivialGuardCallback


def build_pde(case):
    if case == "heatlt":
        from src.pde.heat import Heat2D_LongTime
        return Heat2D_LongTime()
    elif case == "ks":
        from src.pde.chaotic import KuramotoSivashinskyEquation
        return KuramotoSivashinskyEquation()
    raise ValueError(case)


class CosineLR(dde.callbacks.Callback):
    def __init__(self, lr_max, lr_min, total):
        super().__init__()
        self.lr_max, self.lr_min, self.total = lr_max, lr_min, total
        self.i = 0

    def on_epoch_begin(self):
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + np.cos(np.pi * min(self.i / self.total, 1.0)))
        for g in self.model.opt.param_groups:
            g["lr"] = lr
        self.i += 1


class GuardWatch(TrivialGuardCallback):
    """Signals-only variant: log C_enrich/A_late, never intervene."""

    def __init__(self, period=500, log_dir=None):
        super().__init__(mode="off", period=period, log_dir=log_dir)

    def on_epoch_end(self):
        self._since += 1
        if self._since < self.period:
            return
        self._since = 0
        step = self.model.train_state.epoch
        pts = self._pool()
        u = self.model.predict(pts)
        r = self._residual(pts)
        r2 = (r ** 2).sum(axis=1)
        c_enrich, A, flag = self._signals(pts, u, r2)
        self._rows.append({"step": step, "C_enrich": c_enrich, "A_late": A,
                           "flag": int(flag), "acted": 0, "mode": "watch"})
        self._flush()


def flat_params(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def load_flat(net, vec):
    i = 0
    for p in net.parameters():
        n = p.numel()
        p.data.copy_(vec[i:i + n].reshape(p.shape))
        i += n


def unweighted_loss(model, pde, pool, ic_pts, ic_vals, bc_pts):
    """Vanilla-objective proxy on FIXED points: mean resid^2 + IC MSE + BC MSE."""
    def op(x, y):
        res = pde.pde(x, y)
        if not isinstance(res, (list, tuple)):
            res = [res]
        res = [r if r.dim() == 2 else r.unsqueeze(-1) for r in res]
        return torch.cat(res, dim=1)

    rs = []
    for i in range(0, len(pool), 4096):
        rs.append(model.predict(pool[i:i + 4096], operator=op))
    l_res = float((np.concatenate(rs) ** 2).mean())
    u_ic = model.predict(ic_pts)
    l_ic = float(((u_ic - ic_vals) ** 2).mean())
    l_bc = 0.0
    if bc_pts is not None:
        u_bc = model.predict(bc_pts)
        l_bc = float((u_bc ** 2).mean())
    return l_res + l_ic + l_bc, {"res": l_res, "ic": l_ic, "bc": l_bc}


def eval_child(model, pde, mapper):
    pred = model.predict(mapper.points)
    pred_grid = mapper.to_grid(pred)
    ref_grid = mapper.ref_grid()
    l2re = float(np.linalg.norm(pred_grid - ref_grid) / np.linalg.norm(ref_grid))
    norm_ratio = float(np.linalg.norm(pred_grid) / np.linalg.norm(ref_grid))
    return pred_grid, l2re, norm_ratio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["heatlt", "ks"], required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--kicks", type=str, default="1,2,4,8,32")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--cycle-iters", type=int, default=2500)
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="0")
    args = p.parse_args()

    if args.device != "cpu":
        torch.cuda.set_device(int(args.device))
        dde.config.default_device = f"cuda:{args.device}"
    os.makedirs(args.outdir, exist_ok=True)

    pde = build_pde(args.case)
    mapper = forensic.GridMapper(pde.ref_data, pde.input_dim, pde.output_dim)

    def make_model():
        net = dde.nn.FNN([pde.input_dim] + [100] * 5 + [pde.output_dim],
                         "tanh", "Glorot normal").float()
        model = pde.create_model(net)
        model.compile(torch.optim.Adam(net.parameters(), args.base_lr),
                      loss_weights=np.ones(pde.num_loss))
        model.pde = pde
        return model

    model = make_model()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model.net.load_state_dict(state)
    theta_triv = flat_params(model.net).clone()

    # fixed evaluation sets for the barrier/loss proxy
    xall = model.data.train_x_all
    lo, hi = xall.min(0), xall.max(0)
    rng = np.random.default_rng(0)
    pool = rng.uniform(lo, hi, size=(16384, len(lo))).astype(np.float32)
    ic_pts = xall[np.abs(xall[:, -1] - lo[-1]) < 1e-9]
    if len(ic_pts) < 512:
        ic_pts = pool.copy(); ic_pts[:, -1] = lo[-1]
    if args.case == "heatlt":
        ic_vals = (np.sin(4 * np.pi * ic_pts[:, 0:1])
                   * np.sin(3 * np.pi * ic_pts[:, 1:2])).astype(np.float32)
        nb = 2048
        bc = rng.uniform(lo, hi, size=(nb, 3)).astype(np.float32)
        side = rng.integers(0, 4, nb); r = rng.random(nb)
        bc[:, 0] = np.where(side == 0, lo[0], np.where(side == 1, hi[0], bc[:, 0]))
        bc[:, 1] = np.where(side == 2, lo[1], np.where(side == 3, hi[1], bc[:, 1]))
        bc_pts = bc
    else:  # ks: IC only (x in [0, 2pi]); u0 = cos(x)(1+sin(x))
        ic_vals = (np.cos(ic_pts[:, 0:1]) * (1 + np.sin(ic_pts[:, 0:1]))).astype(np.float32)
        bc_pts = None

    l_triv, comp_triv = unweighted_loss(model, pde, pool, ic_pts, ic_vals, bc_pts)
    pred0, l2re0, nr0 = eval_child(model, pde, mapper)
    print(f"[start] trivial ckpt: loss {l_triv:.3e} {comp_triv} l2re {l2re0:.4f} "
          f"norm_ratio {nr0:.3f}", flush=True)

    kicks = [float(k) for k in args.kicks.split(",")]
    results = {"trivial": {"loss": l_triv, "components": comp_triv,
                           "l2re": l2re0, "norm_ratio": nr0},
               "children": [], "soups": [], "config": vars(args)}
    children_by_kick = {}

    for kick in kicks:
        for seed in range(args.seeds):
            tag = f"k{kick:g}_s{seed}"
            dde.config.set_random_seed(1234 + seed)
            load_flat(model.net, theta_triv)
            model.compile(torch.optim.Adam(model.net.parameters(), args.base_lr),
                          loss_weights=np.ones(pde.num_loss))
            cdir = os.path.join(args.outdir, f"child_{tag}")
            os.makedirs(cdir, exist_ok=True)
            cbs = [CosineLR(args.base_lr * kick, 1e-5, args.cycle_iters),
                   GuardWatch(period=500, log_dir=cdir)]
            t0 = time.time()
            model.train(iterations=args.cycle_iters, display_every=1000,
                        callbacks=cbs, model_save_path=None)
            theta_c = flat_params(model.net).clone()
            pred, l2re, nr = eval_child(model, pde, mapper)
            l_c, comp_c = unweighted_loss(model, pde, pool, ic_pts, ic_vals, bc_pts)
            # barrier along linear path to the trivial checkpoint
            path = []
            for a in np.linspace(0, 1, 11):
                load_flat(model.net, (1 - a) * theta_triv + a * theta_c)
                l_a, _ = unweighted_loss(model, pde, pool, ic_pts, ic_vals, bc_pts)
                path.append(l_a)
            barrier = float(max(path) - max(path[0], path[-1]))
            load_flat(model.net, theta_c)
            np.save(os.path.join(cdir, "pred_final.npy"), pred.astype(np.float32))
            torch.save({"model_state_dict": model.net.state_dict()},
                       os.path.join(cdir, "child_final.pt"))
            row = {"tag": tag, "kick": kick, "seed": seed, "l2re": l2re,
                   "norm_ratio": nr, "loss": l_c, "components": comp_c,
                   "barrier_to_trivial": barrier, "path_losses": path,
                   "dist_rel": float(torch.norm(theta_c - theta_triv)
                                     / torch.norm(theta_triv)),
                   "walltime_s": time.time() - t0}
            results["children"].append(row)
            children_by_kick.setdefault(kick, []).append(theta_c)
            print(f"[child {tag}] l2re {l2re:.4f} norm_ratio {nr:.3f} "
                  f"loss {l_c:.3e} barrier {barrier:.3e} "
                  f"dist {row['dist_rel']:.3f}", flush=True)
            with open(os.path.join(args.outdir, "sse_results.json"), "w") as f:
                json.dump(results, f, indent=1)

    # soups per kick (paper: works iff same basin)
    for kick, thetas in children_by_kick.items():
        if len(thetas) < 2:
            continue
        soup = torch.stack(thetas).mean(0)
        load_flat(model.net, soup)
        pred, l2re, nr = eval_child(model, pde, mapper)
        l_s, comp_s = unweighted_loss(model, pde, pool, ic_pts, ic_vals, bc_pts)
        results["soups"].append({"kick": kick, "l2re": l2re, "norm_ratio": nr,
                                 "loss": l_s, "components": comp_s})
        print(f"[soup k{kick:g}] l2re {l2re:.4f} loss {l_s:.3e}", flush=True)

    with open(os.path.join(args.outdir, "sse_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("[DONE] star-SSE experiment complete.", flush=True)


if __name__ == "__main__":
    main()
