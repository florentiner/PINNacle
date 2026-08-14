"""fig32: training-loss traces — Vanilla on top, and below it the SOTA run seen two
ways: per time window (left) and as the single global loss of the stitched solution
as the causal front advances (right).

The vanilla panel uses a broken y-axis when one transient spike would otherwise flatten
the whole curve: the narrow top band holds the spike, the stretched bottom band shows
the real dynamics before and after it.
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
DEFAULT_CUT = {"gs": 1e-2}      # split the GS broken axis at 1e-2 (reproduces the figure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["ks", "gs"], required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--cut", type=float, default=None,
                    help="split value for the broken y-axis (default 1.15x median)")
    args = ap.parse_args()
    cfg = CFG[args.case]
    N = cfg["n_win"]

    m = pd.read_csv(os.path.join(BASE[args.case], "metrics.csv"))
    hist = np.load(os.path.join(cfg["final_dir"], "causal", "history_jax.npz"),
                   allow_pickle=True)
    gl = np.load(f"analysis/out/global_stitched_{args.case}.npz", allow_pickle=True)

    st_v = m["step"].values / 1e3
    L = m["loss_train_total"].values
    med = float(np.median(L))
    i_sp = int(np.argmax(L))
    spike = L[i_sp] > 5 * med
    cut = args.cut or DEFAULT_CUT.get(args.case) or 1.15 * med   # broken-axis split
    core = L[L <= cut]                            # the band the curve really lives in

    fig = plt.figure(figsize=(14.5, 9.4))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.30)

    # ---------------- top: Vanilla ----------------
    if spike:
        top = outer[0].subgridspec(2, 1, height_ratios=[1, 3.6], hspace=0.07)
        ax_t = fig.add_subplot(top[0])
        ax = fig.add_subplot(top[1], sharex=ax_t)
        ax_t.semilogy(st_v, L, "-", color="tab:red", lw=1.5)
        ax_t.set_ylim(cut, L[i_sp] * 1.6)     # bands meet at `cut`: nothing is hidden
        ax_t.spines["bottom"].set_visible(False)
        ax_t.tick_params(labelbottom=False, labelsize=8)
        ax_t.grid(alpha=0.3, which="both")
        ax_t.set_title("Vanilla", fontsize=13)
        Lm = L.copy()
        Lm[L > cut] = np.nan                      # shown in the band above instead
        ax.semilogy(st_v, Lm, "-", color="tab:red", lw=1.8)
        ax.axvline(st_v[i_sp], color="darkred", ls=":", lw=1.2)
        ax.set_ylim(core.min() * 0.995, cut)
        ax.spines["top"].set_visible(False)
        kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=9, linestyle="none",
                  color="k", mec="k", mew=1, clip_on=False)
        ax_t.plot([0, 1], [0, 0], transform=ax_t.transAxes, **kw)
        ax.plot([0, 1], [1, 1], transform=ax.transAxes, **kw)
    else:
        ax = fig.add_subplot(outer[0])
        ax.semilogy(st_v, L, "-", color="tab:red", lw=1.8)
        ax.set_title("Vanilla", fontsize=13)
    ax.scatter([st_v[0], st_v[-1]], [L[0], L[-1]], s=90, color="red", edgecolors="k",
               linewidths=0.6, zorder=5)
    ax.set_xlabel("training iterations (×10³)")
    ax.set_ylabel("training loss (log scale)")
    ax.grid(alpha=0.3, which="both")

    # ---------------- bottom-left: SOTA per window ----------------
    bot = outer[1].subgridspec(1, 2, wspace=0.24)
    ax = fig.add_subplot(bot[0])
    w, stp, ls = hist["window"], hist["step"], hist["loss"]
    order = np.lexsort((stp, w))
    w, stp, ls = w[order], stp[order], ls[order]
    off, offsets, wmax = 0, {}, {}
    for k in range(N):
        sel = w == k
        if not sel.any():
            continue
        offsets[k] = off
        wmax[k] = float(stp[sel].max())
        xx, yy = off + stp[sel], ls[sel]
        ax.semilogy(xx / 1e6, yy, "-", lw=1.0, color=plt.cm.plasma(k / max(N - 1, 1)))
        ax.plot([xx[0] / 1e6, xx[-1] / 1e6], [yy[0], yy[-1]], "o", ms=5, color="red",
                markeredgecolor="k", markeredgewidth=0.4, zorder=5)
        if N <= 10 or k % 2 == 0:
            ax.annotate(f"w{k}", (xx[-1] / 1e6, yy[-1]), fontsize=8,
                        xytext=(2, -11), textcoords="offset points")
        off += wmax[k]
    ax.set_xlabel("cumulative training iterations (×10⁶)")
    ax.set_ylabel("training loss (log scale)")
    ax.set_title("SOTA windows", fontsize=13)
    ax.grid(alpha=0.3, which="both")

    # ---------------- bottom-right: SOTA global ----------------
    ax = fig.add_subplot(bot[1])
    lt, lab = gl["loss_traj"], gl["labels"]
    xs = np.array([offsets.get(int(k), 0) + (st if st >= 0 else wmax.get(int(k), 0))
                   for k, st in lab]) / 1e6
    ok = ~np.isnan(lt)
    ax.semilogy(xs[ok], lt[ok], "-", color="0.35", lw=1.4, zorder=3)
    ends = np.array([np.where(lab[:, 0] == k)[0][-1] for k in range(N)
                     if (lab[:, 0] == k).any()])
    ax.scatter(xs[ends], lt[ends], s=52, color="red", edgecolors="k", linewidths=0.5,
               zorder=5)
    fin = float(gl["parts"].sum())
    ax.axhline(fin, color="gold", ls="--", lw=1.5)
    ax.annotate(f"final Θ* = {fin:.1e}", (xs.max(), fin), fontsize=9,
                color="darkgoldenrod", xytext=(-120, 9), textcoords="offset points")
    ax.set_xlabel("cumulative training iterations (×10⁶)")
    ax.set_ylabel("global loss of the stitched solution (log)")
    ax.set_title("SOTA global", fontsize=13)
    ax.grid(alpha=0.3, which="both")

    fig.savefig(f"{OUT}/fig32_loss_traces_{args.case}.png", dpi=args.dpi,
                bbox_inches="tight", facecolor="white")
    print(f"saved fig32_loss_traces_{args.case}.png (vanilla core band "
          f"{core.min():.3e}-{core.max():.3e}; global {lt[ok][0]:.2e} -> {fin:.2e})")


if __name__ == "__main__":
    main()
