"""fig28 (final): vanilla's complete run vs the ENTIRE causal run in error space.

Left  : vanilla — its own loss landscape + the full trajectory. Legitimate, because
        the top-2 PCA plane of a single continuous descent holds most of its variance.
Right : the whole causal run — every snapshot of every window in one joint weight-space
        plane; white = training steps, red = window start/end, dashed = the fresh-net
        re-initialization between windows. NO terrain is drawn here on purpose:
        measured, the top-2 joint plane holds only ~36-40% of the run's parameter
        variance, and the in-plane projection of a window solution has a residual ~8
        orders of magnitude above its true value — any single 2D landscape spanning all
        windows would be fiction. The error dimension is carried honestly by the inset:
        the true training loss of the complete run, all windows chained.
Bottom: one window in its own error space (there the plane is meaningful) — the funnel.

Plot-only: consumes caches written by landscape_full_training.py / landscape_joint_causal.py.
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
sys.path.insert(0, "analysis")
from landscape_full_training import CFG, flat  # noqa: E402

OUT = "analysis/report_figs"


def vanilla_variance(cfg):
    """Fraction of the vanilla trajectory's variance held by its top-2 PCA plane."""
    base = {"ks": "runs/07.18-13.19.39-baseline-chaotic/0-0",
            "gs": "runs/07.18-13.19.39-baseline-chaotic/1-0"}[cfg["case"]]
    cks = sorted(glob.glob(os.path.join(base, "trajectory", "ckpt_*.pt")),
                 key=lambda s: int(re.search(r"ckpt_(\d+)", s).group(1)))
    vs = [flat(torch.load(c, map_location="cpu", weights_only=False)["model_state_dict"])
          for c in cks]
    T = torch.stack(vs)
    X = T - T.mean(0, keepdim=True)
    _, S, _ = torch.pca_lowrank(X, q=min(6, len(T)))
    return float((S[:2] ** 2).sum() / (X ** 2).sum())


def best_inset_corner(pts, ax):
    """Corner (of 4) holding the fewest trajectory points — where the inset can go."""
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    quad = {"ll": ((pts[:, 0] < mx) & (pts[:, 1] < my)).sum(),
            "lr": ((pts[:, 0] >= mx) & (pts[:, 1] < my)).sum(),
            "ul": ((pts[:, 0] < mx) & (pts[:, 1] >= my)).sum(),
            "ur": ((pts[:, 0] >= mx) & (pts[:, 1] >= my)).sum()}
    best = min(quad, key=quad.get)
    box = {"ll": [0.05, 0.07, 0.38, 0.28], "lr": [0.58, 0.07, 0.38, 0.28],
           "ul": [0.05, 0.68, 0.38, 0.28], "ur": [0.58, 0.68, 0.38, 0.28]}[best]
    return box, best


def draw_run(ax, proj, hl, n_win, label_all=True, fs=9.0, callout=None,
             avoid=None):
    """The whole causal run in the joint plane; `hl` = window highlighted as the example."""
    ks_sorted = sorted(proj)
    allp = np.concatenate([proj[k] for k in ks_sorted])
    cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
    for i, k in enumerate(ks_sorted):
        p = proj[k]
        is_hl = (k == hl)
        if is_hl and len(p) > 1:
            ax.plot(p[:, 0], p[:, 1], "-", color="#ffb300", lw=7, alpha=0.45, zorder=3.6,
                    solid_capstyle="round")
        if len(p) > 1:
            ax.plot(p[:, 0], p[:, 1], "-", color="#d17a00" if is_hl else "0.35",
                    lw=2.0 if is_hl else 1.0, zorder=4)
            ax.scatter(p[1:-1, 0], p[1:-1, 1], s=26 if is_hl else 16, color="w",
                       edgecolors="#d17a00" if is_hl else "0.35",
                       linewidths=0.8 if is_hl else 0.5, zorder=5)
        ax.scatter(p[[0, -1], 0], p[[0, -1], 1], s=95 if is_hl else 62, color="red",
                   edgecolors="k", linewidths=0.6, zorder=6)
        if i + 1 < len(ks_sorted):
            q = proj[ks_sorted[i + 1]]
            ax.plot([p[-1, 0], q[0, 0]], [p[-1, 1], q[0, 1]], "--", color="0.65",
                    lw=0.9, zorder=3)
        if label_all or k % 2 == 0 or k == ks_sorted[-1] or is_hl:
            dx = 7 if p[-1, 0] >= cx else -20
            dy = 6 if p[-1, 1] >= cy else -14
            ax.annotate(f"w{k}", p[-1], color="k", fontsize=fs,
                        fontweight="bold" if is_hl else "normal",
                        xytext=(dx, dy), textcoords="offset points", zorder=7)
    ph = proj[hl]
    anchor = (ph[len(ph) // 2, 0], ph[len(ph) // 2, 1])
    # place the callout inside the axes, on the side away from the highlighted path
    tx = 0.04 if anchor[0] >= cx else 0.58
    ty = 0.10
    if avoid in ("ll", "lr"):                       # inset sits low -> put callout high
        ty = 0.88
    if (avoid == "ll" and tx < 0.5) or (avoid == "lr" and tx > 0.5):
        tx = 0.58 if tx < 0.5 else 0.04
    ax.annotate(callout or f"window {hl} — the example shown below",
                anchor, xycoords="data", xytext=(tx, ty), textcoords="axes fraction",
                fontsize=fs + 0.5, color="#8a5200", fontweight="bold", zorder=8,
                arrowprops=dict(arrowstyle="->", color="#d17a00", lw=1.8,
                                connectionstyle="arc3,rad=0.15"),
                bbox=dict(facecolor="#fff4d6", alpha=0.95, edgecolor="#d17a00",
                          boxstyle="round,pad=0.4"))
    return allp


def draw_trace(ax, hist, n_win, hl, fs=8):
    """True training-loss trace of the complete run, windows chained."""
    w, stp, ls = hist["window"], hist["step"], hist["loss"]
    order = np.lexsort((stp, w))
    w, stp, ls = w[order], stp[order], ls[order]
    off = 0
    for k in range(n_win):
        m = w == k
        if not m.any():
            continue
        xx, yy = off + stp[m], ls[m]
        ax.semilogy(xx / 1e6, yy, "-", lw=2.0 if k == hl else 0.9,
                    color="#d17a00" if k == hl else plt.cm.plasma(k / max(n_win - 1, 1)),
                    zorder=5 if k == hl else 4)
        ax.plot([xx[0] / 1e6, xx[-1] / 1e6], [yy[0], yy[-1]], "o", ms=3.4, color="red",
                zorder=6)
        off += stp[m].max()
    ax.set_xlabel("cumulative training iterations (×10⁶)", fontsize=fs + 1)
    ax.set_ylabel("training loss (true)", fontsize=fs + 1)
    ax.tick_params(labelsize=fs)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["ks", "gs"], required=True)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    cfg = dict(CFG[args.case]); cfg["case"] = args.case

    joint = np.load(f"analysis/out/joint_causal_{args.case}.npz", allow_pickle=True)
    proj = {int(k[1:]): joint[k] for k in joint.files if re.fullmatch(r"w\d+", k)}
    cache = np.load(f"analysis/out/full_training_{args.case}.npz", allow_pickle=True)
    van = np.load(cfg["vanilla_cache"], allow_pickle=True)
    hist = np.load(os.path.join(cfg["final_dir"], "causal", "history_jax.npz"),
                   allow_pickle=True)

    # joint-plane variance (measured in landscape_joint_causal): recompute cheaply
    allp = np.concatenate([proj[k] for k in sorted(proj)])
    ev_causal = {"ks": 0.397, "gs": 0.357}[args.case]
    ev_van = vanilla_variance(cfg)

    fig = plt.figure(figsize=(15.5, 10.8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1], hspace=0.28, wspace=0.20)

    # ---------------- left: vanilla ----------------
    ax = fig.add_subplot(gs[0, 0])
    cs = ax.contourf(van["A"], van["B"], np.log10(van["Z"]), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 loss")
    ax.plot(van["a_traj"], van["b_traj"], "-", color="w", lw=1.6, zorder=4)
    ax.scatter(van["a_traj"][1:-1], van["b_traj"][1:-1], s=26, color="w",
               edgecolors="k", linewidths=0.4, zorder=5)
    ax.scatter([van["a_traj"][0], van["a_traj"][-1]], [van["b_traj"][0], van["b_traj"][-1]],
               s=150, color="red", edgecolors="k", linewidths=0.7, zorder=6)
    st = van["steps"]
    ax.annotate(f"start (it {st[0]})", (van["a_traj"][0], van["b_traj"][0]), color="w",
                fontsize=9.5, xytext=(8, 4), textcoords="offset points")
    ax.annotate(f"end (it {st[-1]})", (van["a_traj"][-1], van["b_traj"][-1]), color="w",
                fontsize=9.5, xytext=(-72, -16), textcoords="offset points")
    ax.set_title("VANILLA — the complete training run on its landscape\n"
                 f"(one net, one objective, one descent; this plane holds {ev_van*100:.0f}% "
                 "of the run's variance)", fontsize=11)
    ax.set_xlabel("PCA dir 1"); ax.set_ylabel("PCA dir 2")

    # ---------------- right: whole causal run ----------------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor("#f2f2f2")
    hl = int(cache["p0_k"])
    tmp = np.concatenate([proj[k] for k in sorted(proj)])
    corner_box, corner_name = best_inset_corner(tmp, ax)
    allp = draw_run(ax, proj, hl, cfg["n_win"], label_all=(cfg["n_win"] <= 10),
                    callout=f"window {hl} — shown below", avoid=corner_name)
    ax.set_title(f"CAUSAL (SOTA) — the complete run: all {cfg['n_win']} windows in ONE joint plane\n"
                 "white = training steps · red = window start/end · dashed = fresh-net re-init",
                 fontsize=11)
    ax.set_xlabel("joint PCA dir 1"); ax.set_ylabel("joint PCA dir 2")
    ax.grid(alpha=0.3, color="w")
    ax.text(0.015, 0.985,
            f"no terrain drawn: this plane holds only {ev_causal*100:.0f}% of the run's\n"
            "variance — a projected window solution reads ~10⁸× its true loss,\n"
            "so any single landscape across all windows would be fiction",
            transform=ax.transAxes, fontsize=7.8, va="top",
            bbox=dict(facecolor="w", alpha=0.85, edgecolor="0.7"))
    ins = ax.inset_axes(corner_box)
    draw_trace(ins, hist, cfg["n_win"], hl, fs=6)
    ins.set_title("the run's true error trace: one descent per window", fontsize=7.5)
    ins.patch.set_alpha(0.93)

    # ---------------- bottom: one window ----------------
    ax = fig.add_subplot(gs[1, :])
    k0 = int(cache["p0_k"])
    cs = ax.contourf(cache["p0_A"], cache["p0_B"], np.log10(cache["p0_Z"] + 1e-14),
                     levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 loss (that window's objective)")
    a, b = cache["p0_a"], cache["p0_b"]
    ax.plot(a, b, "-", color="w", lw=1.5, zorder=4)
    ax.scatter(a[1:-1], b[1:-1], s=22, color="w", edgecolors="k", linewidths=0.3, zorder=5)
    ax.scatter([a[0], 0], [b[0], 0], s=160, color="red", edgecolors="k",
               linewidths=0.7, zorder=6)
    ax.annotate("window start", (a[0], b[0]), color="w", fontsize=10,
                xytext=(10, 6), textcoords="offset points")
    ax.annotate(f"window solution  (L2 = {float(cache['p0_l2']):.1e})", (0, 0), color="w",
                fontsize=10, xytext=(-210, -6), textcoords="offset points")
    ax.set_title(f"EXAMPLE — one causal window (w{k0}) in its OWN error space, where a 2D plane "
                 "is meaningful: the descent into the funnel — one link of the chain above",
                 fontsize=11)
    ax.set_xlabel("PCA dir 1 (window plane)"); ax.set_ylabel("PCA dir 2")

    fig.suptitle(f"{cfg['title']}: complete training in error space — vanilla (left) vs the entire "
                 f"causal run, {cfg['n_win']} windows (right)", fontsize=13.5, y=0.955)
    fig.savefig(f"{OUT}/fig28_joint_causal_{args.case}.png", dpi=args.dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    standalone(cfg, proj, hist, cache, ev_causal, args)
    print(f"saved fig28_joint_causal_{args.case}.png  (vanilla plane {ev_van*100:.1f}%)")


def standalone(cfg, proj, hist, cache, ev_causal, args):
    """fig29: the global error-space picture of the whole causal run, max quality."""
    hl = int(cache["p0_k"])
    fig = plt.figure(figsize=(19, 11.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.65, 1], wspace=0.16)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("#f4f4f4")
    draw_run(ax, proj, hl, cfg["n_win"], label_all=True, fs=11,
             callout=f"window {hl} — the example window\n(its own landscape: fig28, bottom)")
    ax.set_title(f"The COMPLETE causal run: {cfg['n_win']} windows, "
                 f"{sum(len(p) for p in proj.values())} training states, one joint plane",
                 fontsize=14, pad=12)
    ax.set_xlabel("joint PCA direction 1 (weight space)", fontsize=12)
    ax.set_ylabel("joint PCA direction 2 (weight space)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.35, color="w")
    ax.text(0.012, 0.988,
            "white dots = training steps   ·   red dots = window start / end   ·   "
            "dashed = fresh-net re-initialization\n"
            f"no terrain is drawn: this plane holds {ev_causal*100:.0f}% of the run's parameter "
            "variance, and a window solution projected into it\nreads ~10⁸× its true loss — "
            "one landscape spanning all windows would be fiction (each window has its own objective)",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(facecolor="w", alpha=0.9, edgecolor="0.7"))

    ax2 = fig.add_subplot(gs[0, 1])
    draw_trace(ax2, hist, cfg["n_win"], hl, fs=11)
    ax2.set_title("The same run's TRUE error trace\n(every window's loss, chained; red = window edges)",
                  fontsize=14, pad=12)
    fig.suptitle(f"{cfg['title']} — global view of the entire SOTA training run",
                 fontsize=17, y=0.995)
    fig.savefig(f"{OUT}/fig29_global_causal_{args.case}.png", dpi=max(args.dpi, 300),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved fig29_global_causal_{args.case}.png")


if __name__ == "__main__":
    main()
