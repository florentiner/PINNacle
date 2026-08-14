"""fig32: the training-loss trace of both methods, side by side.

Left  — vanilla: its single loss curve over the whole run.
Right — causal: the true error trace of the complete run, every window chained,
        red = window edges.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "analysis")
from landscape_full_training import CFG  # noqa: E402

OUT = "analysis/report_figs"
BASE = {"ks": "runs/07.18-13.19.39-baseline-chaotic/0-0",
        "gs": "runs/07.18-13.19.39-baseline-chaotic/1-0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["ks", "gs"], required=True)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    cfg = CFG[args.case]
    N = cfg["n_win"]

    m = pd.read_csv(os.path.join(BASE[args.case], "metrics.csv"))
    hist = np.load(os.path.join(cfg["final_dir"], "causal", "history_jax.npz"),
                   allow_pickle=True)

    st_v = m["step"].values / 1e3
    L = m["loss_train_total"].values
    i_sp = int(np.argmax(L))
    spike = L[i_sp] > 5 * np.median(L)
    rest = L[L < 3 * np.median(L)]

    fig = plt.figure(figsize=(15, 5.8))
    outer = gridspec.GridSpec(1, 2, wspace=0.22)
    if spike:
        # broken y-axis: a narrow band for the spike, a stretched band for the real range
        left = outer[0].subgridspec(2, 1, height_ratios=[1, 3.4], hspace=0.08)
        ax_t = fig.add_subplot(left[0])
        ax = fig.add_subplot(left[1], sharex=ax_t)
        ax_t.semilogy(st_v, L, "-", color="tab:red", lw=1.6)
        ax_t.set_ylim(rest.max() * 2.2, L[i_sp] * 1.5)
        ax_t.spines["bottom"].set_visible(False)
        ax_t.tick_params(labelbottom=False, labelsize=8)
        ax_t.grid(alpha=0.3, which="both")
        ax_t.set_title("VANILLA", fontsize=13)
        ax_t.annotate(f"transient spike {L[i_sp]:.2e} at it {int(m['step'].values[i_sp]):,}",
                      (st_v[i_sp], L[i_sp]), fontsize=9, color="darkred",
                      xytext=(18, -8), textcoords="offset points",
                      arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
                      bbox=dict(facecolor="w", alpha=0.9, edgecolor="darkred"))
        # the curve without the spike, so the real dynamics are visible
        Lm = L.copy()
        Lm[L > 3 * np.median(L)] = np.nan
        ax.semilogy(st_v, Lm, "-", color="tab:red", lw=1.8)
        ax.axvline(st_v[i_sp], color="darkred", ls=":", lw=1.2)
        ax.annotate("spike removed here\n(see band above)", (st_v[i_sp], rest.min() * 1.02),
                    fontsize=8, color="darkred", xytext=(7, 4),
                    textcoords="offset points")
        ax.set_ylim(rest.min() * 0.97, rest.max() * 1.06)
        ax.spines["top"].set_visible(False)
        kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=9, linestyle="none",
                  color="k", mec="k", mew=1, clip_on=False)
        ax_t.plot([0, 1], [0, 0], transform=ax_t.transAxes, **kw)
        ax.plot([0, 1], [1, 1], transform=ax.transAxes, **kw)
    else:
        ax = fig.add_subplot(outer[0])
        ax.semilogy(st_v, L, "-", color="tab:red", lw=1.8)
        ax.set_title("VANILLA", fontsize=13)
    ax.scatter([st_v[0], st_v[-1]], [L[0], L[-1]], s=90, color="red", edgecolors="k",
               linewidths=0.6, zorder=5)
    ax.set_xlabel("training iterations (×10³)")
    ax.set_ylabel("training loss (log scale)")
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(outer[1])
    w, stp, ls = hist["window"], hist["step"], hist["loss"]
    order = np.lexsort((stp, w))
    w, stp, ls = w[order], stp[order], ls[order]
    off = 0
    for k in range(N):
        sel = w == k
        if not sel.any():
            continue
        xx, yy = off + stp[sel], ls[sel]
        ax.semilogy(xx / 1e6, yy, "-", lw=1.0,
                    color=plt.cm.plasma(k / max(N - 1, 1)))
        ax.plot([xx[0] / 1e6, xx[-1] / 1e6], [yy[0], yy[-1]], "o", ms=5, color="red",
                markeredgecolor="k", markeredgewidth=0.4, zorder=5)
        if N <= 10 or k % 2 == 0:
            ax.annotate(f"w{k}", (xx[-1] / 1e6, yy[-1]), fontsize=8,
                        xytext=(2, -11), textcoords="offset points")
        off += stp[sel].max()
    ax.set_xlabel("cumulative training iterations (×10⁶)")
    ax.set_ylabel("training loss (log scale)")
    ax.set_title("CAUSAL (SOTA)", fontsize=13)
    ax.grid(alpha=0.3, which="both")

    fig.savefig(f"{OUT}/fig32_loss_traces_{args.case}.png", dpi=args.dpi,
                bbox_inches="tight", facecolor="white")
    print(f"saved fig32_loss_traces_{args.case}.png  "
          f"(vanilla {m['loss_train_total'].iloc[0]:.2e} -> {m['loss_train_total'].iloc[-1]:.2e}; "
          f"causal per-window drops ~{np.log10(ls[w == 0][0] / ls[w == 0][-1]):.1f} orders)")


if __name__ == "__main__":
    main()
