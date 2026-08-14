"""fig30: the REAL global error space of the causal run (stitched-solution objective)."""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "analysis")
from landscape_full_training import CFG  # noqa: E402

OUT = "analysis/report_figs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["ks", "gs"], required=True)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    cfg = CFG[args.case]
    N = cfg["n_win"]
    d = np.load(f"analysis/out/global_stitched_{args.case}.npz", allow_pickle=True)
    A, B, Z, AZ, BZ, ZZ = d["A"], d["B"], d["Z"], d["AZ"], d["BZ"], d["ZZ"]
    a_tr, b_tr, lab, lt = d["a_tr"], d["b_tr"], d["labels"], d["loss_traj"]
    res0, ic0, cont0 = d["parts"]
    ev2 = float(d["ev2"])
    hist = np.load(os.path.join(cfg["final_dir"], "causal", "history_jax.npz"),
                   allow_pickle=True)
    wloss = np.array([hist["loss"][hist["window"] == k][-1] for k in range(N)])
    wl2 = np.array([hist["l2_window"][hist["window"] == k][-1] for k in range(N)])

    wins = lab[:, 0]
    ends = np.array([np.where(wins == k)[0][-1] for k in range(N)])
    starts = np.array([np.where(wins == k)[0][0] for k in range(N)])

    fig = plt.figure(figsize=(19, 11.5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.45, 1, 1], height_ratios=[1.1, 1],
                           hspace=0.32, wspace=0.34)

    # ---- A: the global landscape, wide ----
    ax = fig.add_subplot(gs[:, 0])
    cs = ax.contourf(A, B, np.log10(Z), levels=40, cmap="viridis")
    fig.colorbar(cs, ax=ax, fraction=0.040, pad=0.02, label="log10 L_global")
    ax.plot(a_tr, b_tr, "-", color="w", lw=1.1, zorder=4)
    ax.scatter(a_tr, b_tr, s=13, color="w", edgecolors="none", zorder=5)
    ax.scatter(a_tr[starts], b_tr[starts], s=52, color="red", edgecolors="k",
               linewidths=0.5, zorder=6)
    ax.scatter(a_tr[ends], b_tr[ends], s=52, color="red", edgecolors="k",
               linewidths=0.5, zorder=6)
    for k in range(N):
        if N <= 10 or k % 2 == 0:
            off = (7, 5) if k % 2 == 0 else (-22, -13)
            ax.annotate(f"w{k}", (a_tr[ends[k]], b_tr[ends[k]]), color="w", fontsize=9,
                        xytext=off, textcoords="offset points", zorder=7)
    ax.scatter([0], [0], marker="*", s=460, color="gold", edgecolors="k",
               linewidths=0.8, zorder=8)
    ax.annotate("Θ* — the full stitched solution\n"
                f"L_global = {res0+ic0+cont0:.2e}", (0, 0), color="gold", fontsize=10,
                fontweight="bold", xytext=(-250, -46), textcoords="offset points", zorder=9,
                arrowprops=dict(arrowstyle="->", color="gold", lw=1.5),
                bbox=dict(facecolor="#00000099", edgecolor="gold"))
    ax.set_title("A REAL global error space: the stitched-solution objective on the product\n"
                 f"space of all {N} window networks — the run descends into a true global funnel",
                 fontsize=12.5)
    ax.set_xlabel("global PCA dir 1 (all windows' parameters)")
    ax.set_ylabel("global PCA dir 2")
    ax.text(0.015, 0.982,
            "white = training steps · red = window start/end · ★ = trained full solution\n"
            f"surface = true L_global evaluations; plane through Θ* holds {ev2*100:.0f}% of the\n"
            "global trajectory's variance",
            transform=ax.transAxes, fontsize=8.2, va="top", color="w",
            bbox=dict(facecolor="#00000077", edgecolor="none"))

    # ---- B: zoom on the funnel ----
    ax = fig.add_subplot(gs[0, 1])
    cs = ax.contourf(AZ, BZ, np.log10(ZZ), levels=36, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 L_global")
    m = (a_tr >= AZ.min()) & (a_tr <= AZ.max()) & (b_tr >= BZ.min()) & (b_tr <= BZ.max())
    ax.plot(a_tr[m], b_tr[m], "w.-", lw=1.0, ms=4, zorder=4)
    ax.scatter([0], [0], marker="*", s=420, color="gold", edgecolors="k",
               linewidths=0.8, zorder=6)
    ax.set_title(f"Zoom on Θ*: the global optimum is a genuine funnel\n"
                 f"({np.log10(ZZ.max()/ZZ.min()):.1f} orders of magnitude across this patch)",
                 fontsize=11)
    ax.set_xlabel("global PCA dir 1"); ax.set_ylabel("global PCA dir 2")

    # ---- C: global loss as the front advances ----
    ax = fig.add_subplot(gs[0, 2])
    ok = ~np.isnan(lt)
    ax.semilogy(np.arange(len(lt))[ok], lt[ok], "-", color="0.4", lw=1.2, zorder=3)
    ax.scatter(ends, lt[ends], s=46, color="red", edgecolors="k", linewidths=0.5,
               zorder=5, label="window completed")
    ax.axhline(res0 + ic0 + cont0, color="gold", ls="--", lw=1.4,
               label=f"final Θ* = {res0+ic0+cont0:.1e}")
    ax.set_xlabel("training state along the whole run")
    ax.set_ylabel("L_global (true)")
    ax.set_title("Global error while the causal front advances:\nevery completed window lowers the whole solution's loss",
                 fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # ---- D: per-window loss, separately ----
    ax = fig.add_subplot(gs[1, 1:])
    x = np.arange(N)
    ax.bar(x - 0.21, wloss, width=0.42, color="tab:blue", label="final training loss of that window")
    ax.bar(x + 0.21, wl2, width=0.42, color="tab:orange", label="that window's L2 error")
    ax.set_yscale("log")
    for i in range(N):
        ax.text(i - 0.21, wloss[i] * 1.25, f"{wloss[i]:.0e}", ha="center", fontsize=6.6,
                rotation=90, color="tab:blue")
        ax.text(i + 0.21, wl2[i] * 1.25, f"{wl2[i]:.0e}", ha="center", fontsize=6.6,
                rotation=90, color="chocolate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"w{k}" for k in range(N)], fontsize=8.5)
    ax.set_ylabel("value (log)")
    ax.set_title("The loss of every window, separately — the training loss stays flat "
                 "(each window is solved to the same standard)\nwhile the L2 error grows "
                 "monotonically: that growth is inherited handoff error amplified by the dynamics",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"{cfg['title']} — the causal run in its TRUE global error space "
                 f"(one objective over all {N} windows: physics + initial condition + interface continuity)",
                 fontsize=14.5, y=0.965)
    fig.savefig(f"{OUT}/fig30_global_stitched_{args.case}.png", dpi=args.dpi,
                bbox_inches="tight", facecolor="white")
    print(f"saved fig30_global_stitched_{args.case}.png  "
          f"(Θ* {res0+ic0+cont0:.3e}; wide max {Z.max():.3e}; plane {ev2*100:.1f}%)")


if __name__ == "__main__":
    main()
