"""Vanilla full training vs the WHOLE causal run on ONE shared error space.

Left  : vanilla's complete run on its own landscape (cached).
Right : ALL causal windows in ONE joint plane over ONE terrain, with the full
        training path — white dots = snapshots (steps), red dots = window start/end,
        dashed grey = the fresh-net re-initialization between windows.
Bottom: one window as an example (its own objective, from the fig27 cache).

Why a single terrain is legitimate here: KS and GS are autonomous, and every window
is trained with window-local time, so the PDE-residual term of the objective is the
SAME functional for every window — it is the shared part of all N objectives and is
what the terrain shows. What differs per window is the IC anchor (the handoff), which
is exactly what the red window-boundary markers denote.

Usage: python analysis/landscape_joint_causal.py --case ks|gs [--grid 21]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.getcwd())
os.environ.setdefault("DDEBACKEND", "pytorch")

OUT = "analysis/report_figs"
from landscape_full_training import CFG, flat, unflat, surface, gather_snaps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["ks", "gs"], required=True)
    ap.add_argument("--grid", type=int, default=21)
    ap.add_argument("--npts", type=int, default=500)
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

    # ---------- collect the whole run: every snapshot of every window + finals ----------
    paths = {}          # window -> list of (iter, flat_vec); final appended with iter=inf
    for k in range(cfg["n_win"]):
        pts = []
        for sf in gather_snaps(cfg, k):
            it = int(re.search(r"snap_(\d+)", sf).group(1))
            net.load_state_dict(jax_npz_to_state_dict(np.load(sf)), strict=False)
            pts.append((it, flat(net.state_dict()).clone()))
        fin = os.path.join(cfg["final_dir"], "trajectory", f"w{k}_final_params.npz")
        if os.path.exists(fin):
            net.load_state_dict(jax_npz_to_state_dict(np.load(fin)), strict=False)
            pts.append((10 ** 9, flat(net.state_dict()).clone()))
        if pts:
            pts.sort(key=lambda z: z[0])
            paths[k] = pts
    allvec = torch.stack([v for k in sorted(paths) for _, v in paths[k]])
    print(f"[{args.case}] {len(paths)} windows, {len(allvec)} points total", flush=True)

    center = allvec.mean(0)
    X = allvec - center
    _, _, V = torch.pca_lowrank(X, q=4)
    d1 = V[:, 0] / V[:, 0].norm()
    d2 = V[:, 1] - (V[:, 1] @ d1) * d1
    d2 = d2 / d2.norm()
    proj = {k: np.stack([[(v - center) @ d1, (v - center) @ d2] for _, v in pts])
            for k, pts in paths.items()}

    # ---------- shared terrain: the PDE-residual term (identical functional per window)
    ref_sd = {kk: v.clone() for kk, v in net.state_dict().items()}
    if args.case == "ks":
        tw = torch.tensor(rng.uniform(0, case.T_w * 1.01, size=(args.npts, 1)),
                          dtype=torch.float32)
        xw = torch.tensor(rng.uniform(0, 2 * np.pi, size=(args.npts, 1)),
                          dtype=torch.float32)
    else:
        tw = torch.tensor(rng.uniform(0, 1.01, size=(args.npts, 1)), dtype=torch.float32)
        xw = torch.tensor(rng.uniform(-1, 1, size=(args.npts, 2)), dtype=torch.float32)

    def rloss(sd):
        net.load_state_dict(sd)
        with torch.enable_grad():
            r = case.residual(net, tw, xw, case.T_w)
        return float((r ** 2).mean())

    xy = np.concatenate([proj[k] for k in sorted(proj)])
    pa = 0.12 * (xy[:, 0].max() - xy[:, 0].min())
    pb = 0.12 * (xy[:, 1].max() - xy[:, 1].min())
    print("computing joint terrain ...", flush=True)
    A, B, Z = surface(rloss, unflat(center, ref_sd), d1, d2,
                      (xy[:, 0].min() - pa, xy[:, 0].max() + pa),
                      (xy[:, 1].min() - pb, xy[:, 1].max() + pb), n=args.grid)
    np.savez(f"analysis/out/joint_causal_{args.case}.npz", A=A, B=B, Z=Z,
             **{f"w{k}": proj[k] for k in proj})

    # ---------- figure ----------
    van = np.load(cfg["vanilla_cache"], allow_pickle=True)
    cache = np.load(f"analysis/out/full_training_{args.case}.npz", allow_pickle=True)
    fig = plt.figure(figsize=(15.5, 10.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1], hspace=0.26, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    cs = ax.contourf(van["A"], van["B"], np.log10(van["Z"]), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 loss")
    ax.plot(van["a_traj"], van["b_traj"], "-", color="w", lw=1.6, zorder=4)
    ax.scatter(van["a_traj"], van["b_traj"], s=26, color="w", edgecolors="k",
               linewidths=0.4, zorder=5)
    ax.scatter([van["a_traj"][0], van["a_traj"][-1]], [van["b_traj"][0], van["b_traj"][-1]],
               s=130, color="red", edgecolors="k", linewidths=0.6, zorder=6)
    st = van["steps"]
    ax.annotate(f"start (it {st[0]})", (van["a_traj"][0], van["b_traj"][0]), color="w",
                fontsize=9, xytext=(6, 6), textcoords="offset points")
    ax.annotate(f"end (it {st[-1]})", (van["a_traj"][-1], van["b_traj"][-1]), color="w",
                fontsize=9, xytext=(6, -14), textcoords="offset points")
    ax.set_title("VANILLA — the complete training run in error space\n"
                 "(one network, one objective: a single descent)", fontsize=11.5)
    ax.set_xlabel("PCA dir 1"); ax.set_ylabel("PCA dir 2")

    ax = fig.add_subplot(gs[0, 1])
    cs = ax.contourf(A, B, np.log10(Z + 1e-14), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 PDE-residual loss (shared by all windows)")
    ks_sorted = sorted(proj)
    for i, k in enumerate(ks_sorted):
        p = proj[k]
        if len(p) > 1:
            ax.plot(p[:, 0], p[:, 1], "-", color="w", lw=1.3, zorder=4)
            ax.scatter(p[1:-1, 0], p[1:-1, 1], s=13, color="w", edgecolors="none",
                       zorder=5)
        ax.scatter(p[[0, -1], 0], p[[0, -1], 1], s=70, color="red", edgecolors="k",
                   linewidths=0.5, zorder=6)
        if i + 1 < len(ks_sorted):
            q = proj[ks_sorted[i + 1]]
            ax.plot([p[-1, 0], q[0, 0]], [p[-1, 1], q[0, 1]], "--", color="0.75",
                    lw=1.0, zorder=3)
        lab = p[-1]
        ax.annotate(f"w{k}", lab, color="w", fontsize=8.5, xytext=(5, 4),
                    textcoords="offset points", zorder=7)
    ax.set_title(f"CAUSAL (SOTA) — the complete run: all {cfg['n_win']} windows on ONE shared\n"
                 "error space; white = training steps, red = window start/end, dashed = re-init",
                 fontsize=11.5)
    ax.set_xlabel("joint PCA dir 1"); ax.set_ylabel("joint PCA dir 2")

    ax = fig.add_subplot(gs[1, :])
    k0 = int(cache["p0_k"])
    cs = ax.contourf(cache["p0_A"], cache["p0_B"], np.log10(cache["p0_Z"] + 1e-14),
                     levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 loss (window objective: residual + its IC)")
    a, b = cache["p0_a"], cache["p0_b"]
    ax.plot(a, b, "-", color="w", lw=1.5, zorder=4)
    ax.scatter(a[:-1], b[:-1], s=20, color="w", edgecolors="none", zorder=5)
    ax.scatter([a[0], 0], [b[0], 0], s=150, color="red", edgecolors="k",
               linewidths=0.6, zorder=6)
    ax.annotate("window start", (a[0], b[0]), color="w", fontsize=9.5,
                xytext=(7, 6), textcoords="offset points")
    ax.annotate(f"window solution (L2 = {float(cache['p0_l2']):.1e})", (0, 0), color="w",
                fontsize=9.5, xytext=(8, -14), textcoords="offset points")
    ax.set_title(f"EXAMPLE — a single causal window (w{k0}) in its own error space: "
                 "the descent into the funnel, one link of the chain above", fontsize=11.5)
    ax.set_xlabel("PCA dir 1 (window plane)"); ax.set_ylabel("PCA dir 2")

    fig.suptitle(f"{cfg['title']}: complete training in error space — vanilla (left) vs the entire causal run "
                 f"({cfg['n_win']} windows, right)", fontsize=13.5, y=0.955)
    fig.savefig(f"{OUT}/fig28_joint_causal_{args.case}.png", dpi=150, bbox_inches="tight")
    print(f"saved fig28_joint_causal_{args.case}.png")


if __name__ == "__main__":
    main()
