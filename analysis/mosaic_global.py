"""Approximate error space for the GLOBAL causal map — a mosaic of TRUE local slices.

A single surface through all windows is fiction (the joint plane holds only ~36-40% of
the run's parameter variance, so a projected window solution evaluates ~1e8x above its
true loss). What IS legitimate: for each window k, slice ITS objective through ITS OWN
solution along the shared global directions,

    Z_k(a, b) = loss_k( theta_k + (a - a_k) d1 + (b - b_k) d2 ),

which is a real 2D cut of a real objective, anchored at a real trained network, and
oriented like the global map. Painting each grid cell with the slice of its NEAREST
window gives an approximate global error space made only of true evaluations; the seams
between regions are region boundaries, not features of any landscape.

Writes analysis/out/mosaic_{case}.npz (Z, owner, A, B + per-window stats).
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "analysis")
os.environ.setdefault("DDEBACKEND", "pytorch")
from landscape_full_training import CFG, flat, unflat, window_ic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["ks", "gs"], required=True)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--npts", type=int, default=400)
    args = ap.parse_args()
    cfg = CFG[args.case]
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(0)

    from causalpinn.cases import get_case
    from causalpinn.train import CausalConfig
    from causalpinn.jax_bridge import jax_npz_to_state_dict

    ccfg = CausalConfig(case=args.case, device="cpu", windows=cfg["n_win"])
    case = get_case(args.case, ccfg)
    net = case.build_net(cfg["encoding"], ccfg.seed, torch.device("cpu"))
    ref_sd = {k: v.clone() for k, v in net.state_dict().items()}

    joint = np.load(f"analysis/out/joint_causal_{args.case}.npz", allow_pickle=True)
    proj = {int(k[1:]): joint[k] for k in joint.files if re.fullmatch(r"w\d+", k)}
    A, B = joint["A"], joint["B"]

    # rebuild the exact joint plane (same construction as landscape_joint_causal)
    from landscape_full_training import gather_snaps
    vecs, tags = [], []
    for k in range(cfg["n_win"]):
        for sf in gather_snaps(cfg, k):
            net.load_state_dict(jax_npz_to_state_dict(np.load(sf)), strict=False)
            vecs.append(flat(net.state_dict()).clone()); tags.append((k, "snap"))
        fin = os.path.join(cfg["final_dir"], "trajectory", f"w{k}_final_params.npz")
        if os.path.exists(fin):
            net.load_state_dict(jax_npz_to_state_dict(np.load(fin)), strict=False)
            vecs.append(flat(net.state_dict()).clone()); tags.append((k, "final"))
    T = torch.stack(vecs)
    center = T.mean(0)
    X = T - center
    _, S, V = torch.pca_lowrank(X, q=8)
    d1 = V[:, 0] / V[:, 0].norm()
    d2 = V[:, 1] - (V[:, 1] @ d1) * d1
    d2 = d2 / d2.norm()
    ev2 = float((S[:2] ** 2).sum() / (X ** 2).sum())
    print(f"[{args.case}] joint plane holds {ev2*100:.1f}% of variance", flush=True)

    theta = {}
    for i, (k, t) in enumerate(tags):
        if t == "final":
            theta[k] = T[i]
    anchors = {k: (float((theta[k] - center) @ d1), float((theta[k] - center) @ d2))
               for k in theta}

    # per-window objective (residual + that window's own IC)
    def make_loss(k):
        ic_coords, ic_vals = window_ic(case, cfg, k)
        ic_c = torch.tensor(np.asarray(ic_coords), dtype=torch.float32)
        ic_v = torch.tensor(np.asarray(ic_vals), dtype=torch.float32)
        t0 = torch.zeros(len(ic_c), 1)
        if args.case == "ks":
            tw = torch.tensor(rng.uniform(0, case.T_w * 1.01, size=(args.npts, 1)),
                              dtype=torch.float32)
            xw = torch.tensor(rng.uniform(0, 2 * np.pi, size=(args.npts, 1)),
                              dtype=torch.float32)

            def f(sd):
                net.load_state_dict(sd)
                with torch.enable_grad():
                    r = case.residual(net, tw, xw, case.T_w)
                return float((r ** 2).mean() + ((net(t0, ic_c) - ic_v) ** 2).mean())
        else:
            tw = torch.tensor(rng.uniform(0, 1.01, size=(args.npts, 1)), dtype=torch.float32)
            xw = torch.tensor(rng.uniform(-1, 1, size=(args.npts, 2)), dtype=torch.float32)

            def f(sd):
                net.load_state_dict(sd)
                with torch.enable_grad():
                    r = case.residual(net, tw, xw, case.T_w)
                p = net(t0, ic_c[:, 0:1], ic_c[:, 1:2])
                return float((r ** 2).mean() + ((p - ic_v) ** 2).mean())
        return f

    losses = {k: make_loss(k) for k in theta}

    allp = np.concatenate([proj[k] for k in sorted(proj)])
    pa = 0.10 * (allp[:, 0].max() - allp[:, 0].min())
    pb = 0.10 * (allp[:, 1].max() - allp[:, 1].min())
    A = np.linspace(allp[:, 0].min() - pa, allp[:, 0].max() + pa, args.grid)
    B = np.linspace(allp[:, 1].min() - pb, allp[:, 1].max() + pb, args.grid)
    ks = sorted(theta)
    ax_ = np.array([anchors[k][0] for k in ks])
    ay_ = np.array([anchors[k][1] for k in ks])

    Z = np.zeros((len(B), len(A)))
    own = np.zeros((len(B), len(A)), dtype=int)
    for i, b in enumerate(B):
        for j, a in enumerate(A):
            kk = ks[int(np.argmin((ax_ - a) ** 2 + (ay_ - b) ** 2))]
            th = theta[kk] + (a - anchors[kk][0]) * d1 + (b - anchors[kk][1]) * d2
            Z[i, j] = losses[kk](unflat(th, ref_sd))
            own[i, j] = kk
        print(f"  row {i+1}/{len(B)}", flush=True)

    hist = np.load(os.path.join(cfg["final_dir"], "causal", "history_jax.npz"),
                   allow_pickle=True)
    stats = {k: dict(loss=float(hist["loss"][hist["window"] == k][-1]),
                     l2=float(hist["l2_window"][hist["window"] == k][-1]),
                     wmin=float(hist["w_min"][hist["window"] == k].max()))
             for k in ks if (hist["window"] == k).any()}
    np.savez(f"analysis/out/mosaic_{args.case}.npz", Z=Z, own=own, A=A, B=B,
             ev2=ev2, anchors=np.array([[anchors[k][0], anchors[k][1]] for k in ks]),
             ks=np.array(ks),
             wloss=np.array([stats[k]["loss"] for k in ks if k in stats]),
             wl2=np.array([stats[k]["l2"] for k in ks if k in stats]))
    print(f"saved mosaic_{args.case}.npz  (loss range {Z.min():.2e} … {Z.max():.2e})")


if __name__ == "__main__":
    main()
