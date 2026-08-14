"""fig31: vanilla's full run (left) vs the causal run on its TRUE global landscape
(right), with the example window highlighted, and that window's own space below.

Left   — vanilla: its complete trajectory on its own loss landscape.
Right  — causal: the whole run on the REAL global objective (stitched solution over the
         product space of all window networks: physics + IC + interface continuity);
         the segment belonging to the example window is highlighted in gold.
Bottom — that example window alone, in its own error space.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "analysis")
from landscape_full_training import CFG  # noqa: E402

OUT = "analysis/report_figs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["ks", "gs"], required=True)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    cfg = dict(CFG[args.case]); cfg["case"] = args.case
    N = cfg["n_win"]

    van = np.load(cfg["vanilla_cache"], allow_pickle=True)
    g = np.load(f"analysis/out/global_stitched_{args.case}.npz", allow_pickle=True)
    cache = np.load(f"analysis/out/full_training_{args.case}.npz", allow_pickle=True)
    A, B, Z = g["A"], g["B"], g["Z"]
    a_tr, b_tr, lab, lt = g["a_tr"], g["b_tr"], g["labels"], g["loss_traj"]
    res0, ic0, cont0 = g["parts"]
    ev2 = float(g["ev2"])
    hl = int(cache["p0_k"])
    wins = lab[:, 0]

    itp = RegularGridInterpolator((B, A), Z, bounds_error=False, fill_value=None)
    ok = ~np.isnan(lt)
    ratio = float(np.median(itp(np.stack([b_tr[ok], a_tr[ok]], axis=1)) / lt[ok]))

    fig = plt.figure(figsize=(17, 11.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.32, 1], hspace=0.30, wspace=0.28)

    # ---------------- left: vanilla, complete run ----------------
    ax = fig.add_subplot(gs[0, 0])
    cs = ax.contourf(van["A"], van["B"], np.log10(van["Z"]), levels=34, cmap="viridis")
    fig.colorbar(cs, ax=ax, fraction=0.043, pad=0.02, label="log10 loss")
    ax.plot(van["a_traj"], van["b_traj"], "-", color="w", lw=1.7, zorder=4)
    ax.scatter(van["a_traj"][1:-1], van["b_traj"][1:-1], s=30, color="w",
               edgecolors="k", linewidths=0.4, zorder=5)
    ax.scatter([van["a_traj"][0], van["a_traj"][-1]],
               [van["b_traj"][0], van["b_traj"][-1]], s=165, color="red",
               edgecolors="k", linewidths=0.7, zorder=6)
    st = van["steps"]
    ax.annotate(f"start (it {st[0]})", (van["a_traj"][0], van["b_traj"][0]), color="w",
                fontsize=10, xytext=(9, 5), textcoords="offset points")
    ax.annotate(f"end (it {st[-1]})", (van["a_traj"][-1], van["b_traj"][-1]), color="w",
                fontsize=10, xytext=(-78, -18), textcoords="offset points")
    ax.set_title("VANILLA", fontsize=13)
    ax.set_xlabel("PCA dir 1"); ax.set_ylabel("PCA dir 2")

    # ---------------- right: causal on the TRUE global landscape ----------------
    ax = fig.add_subplot(gs[0, 1])
    cs = ax.contourf(A, B, np.log10(Z), levels=40, cmap="viridis")
    fig.colorbar(cs, ax=ax, fraction=0.043, pad=0.02, label="log10 L_global")
    for k in range(N):
        m = wins == k
        if not m.any():
            continue
        if k == hl:
            ax.plot(a_tr[m], b_tr[m], "-", color="#ffb300", lw=7, alpha=0.5, zorder=3.6,
                    solid_capstyle="round")
            ax.plot(a_tr[m], b_tr[m], "-", color="#ff8f00", lw=2.2, zorder=4.5)
            ax.scatter(a_tr[m], b_tr[m], s=26, color="#fff3cd", edgecolors="#ff8f00",
                       linewidths=0.7, zorder=5.5)
        else:
            ax.plot(a_tr[m], b_tr[m], "-", color="w", lw=1.1, zorder=4)
            ax.scatter(a_tr[m], b_tr[m], s=13, color="w", edgecolors="none", zorder=5)
        idx = np.where(m)[0]
        ax.scatter(a_tr[idx[[0, -1]]], b_tr[idx[[0, -1]]], s=58, color="red",
                   edgecolors="k", linewidths=0.5, zorder=6)
        if N <= 10 or k % 3 == 0 or k == N - 1:
            off = (8, 6) if k % 2 == 0 else (-26, -16)
            ax.annotate(f"w{k}", (a_tr[idx[-1]], b_tr[idx[-1]]), color="w", fontsize=9,
                        fontweight="bold" if k == hl else "normal",
                        xytext=off, textcoords="offset points", zorder=7)
    ax.scatter([0], [0], marker="*", s=430, color="gold", edgecolors="k",
               linewidths=0.8, zorder=8)
    mh = wins == hl
    ax.annotate(f"window {hl} — the segment shown below",
                (a_tr[mh][len(a_tr[mh]) // 2], b_tr[mh][len(b_tr[mh]) // 2]),
                xycoords="data", xytext=(0.03, 0.09), textcoords="axes fraction",
                fontsize=10, color="#8a5200", fontweight="bold", zorder=9,
                arrowprops=dict(arrowstyle="->", color="#ff8f00", lw=1.8,
                                connectionstyle="arc3,rad=0.2"),
                bbox=dict(facecolor="#fff4d6", alpha=0.95, edgecolor="#d17a00",
                          boxstyle="round,pad=0.4"))
    ax.set_title("CAUSAL (SOTA)", fontsize=13)
    ax.set_xlabel("global PCA dir 1 (all windows' parameters)")
    ax.set_ylabel("global PCA dir 2")

    # ---------------- bottom: the example window alone ----------------
    ax = fig.add_subplot(gs[1, :])
    cs = ax.contourf(cache["p0_A"], cache["p0_B"], np.log10(cache["p0_Z"] + 1e-14),
                     levels=34, cmap="viridis")
    fig.colorbar(cs, ax=ax, fraction=0.020, pad=0.015,
                 label="log10 loss (that window's own objective)")
    a, b = cache["p0_a"], cache["p0_b"]
    ax.plot(a, b, "-", color="#ff8f00", lw=2.0, zorder=4)
    ax.scatter(a[1:-1], b[1:-1], s=26, color="#fff3cd", edgecolors="#ff8f00",
               linewidths=0.6, zorder=5)
    ax.scatter([a[0], 0], [b[0], 0], s=170, color="red", edgecolors="k",
               linewidths=0.7, zorder=6)
    ax.annotate("window start", (a[0], b[0]), color="w", fontsize=10.5,
                xytext=(11, 6), textcoords="offset points")
    ax.annotate(f"window solution  (L2 = {float(cache['p0_l2']):.1e})", (0, 0), color="w",
                fontsize=10.5, xytext=(-235, -8), textcoords="offset points")
    ax.set_title(f"EXAMPLE — window {hl}", fontsize=13)
    ax.set_xlabel("PCA dir 1 (window plane)"); ax.set_ylabel("PCA dir 2")

    fig.savefig(f"{OUT}/fig31_vanilla_vs_global_{args.case}.png", dpi=args.dpi,
                bbox_inches="tight", facecolor="white")
    print(f"saved fig31_vanilla_vs_global_{args.case}.png "
          f"(plane {ev2*100:.1f}%, surface/true median {ratio:.1f}x)")


if __name__ == "__main__":
    main()
