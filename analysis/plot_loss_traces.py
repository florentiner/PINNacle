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

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    ax = axes[0]
    ax.semilogy(m["step"] / 1e3, m["loss_train_total"], "-", color="tab:red", lw=1.8)
    ax.scatter([m["step"].iloc[0] / 1e3, m["step"].iloc[-1] / 1e3],
               [m["loss_train_total"].iloc[0], m["loss_train_total"].iloc[-1]],
               s=90, color="red", edgecolors="k", linewidths=0.6, zorder=5)
    ax.set_xlabel("training iterations (×10³)")
    ax.set_ylabel("training loss")
    ax.set_title("VANILLA", fontsize=13)
    ax.grid(alpha=0.3)

    ax = axes[1]
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
    ax.set_ylabel("training loss")
    ax.set_title("CAUSAL (SOTA)", fontsize=13)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig32_loss_traces_{args.case}.png", dpi=args.dpi,
                bbox_inches="tight", facecolor="white")
    print(f"saved fig32_loss_traces_{args.case}.png  "
          f"(vanilla {m['loss_train_total'].iloc[0]:.2e} -> {m['loss_train_total'].iloc[-1]:.2e}; "
          f"causal per-window drops ~{np.log10(ls[w == 0][0] / ls[w == 0][-1]):.1f} orders)")


if __name__ == "__main__":
    main()
