"""Generate the figures embedded in REPORT.md from a runs_landscape_compare tree.

Produces small PNGs under experiments/landscape_compare/figures/ (committed with the repo,
so REPORT.md renders on GitHub without the results tree):

  per_seed_ablation.png   final rel-L2 per method per seed, both PDEs (origin/all highlighted)
  ks_horizon.png          per-time-band error curves + amplitude decay (the horizon wall)
  ks_decomposition.png    the A -> +C -> +W -> +G chain with per-seed dots
  dissociation.png        PCA of final weights (clusters by seed) vs trajectories (by method)
  gs_insurance.png        GS error-vs-progress: origin's seed-1236 overshoot vs the stack

Usage:
    python experiments/landscape_compare/make_report_figures.py --runs runs_landscape_compare
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import similarity_analysis as sa  # noqa: E402

# validated categorical palette (dataviz skill, light surface): blue, aqua, orange, violet
BLUE, AQUA, ORANGE, VIOLET = "#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7"
GRAY, INK, MUTED = "#b9b8b2", "#0b0b0b", "#52514e"
SEED_COLORS = {1234: BLUE, 1235: AQUA, 1236: VIOLET}
SEEDS = [1234, 1235, 1236]
LETTERS = ["C", "W", "A", "G"]
COMBOS = [tuple(c) for r in range(5) for c in itertools.combinations(LETTERS, r)]


def mname(c):
    return "ablation_none" if not c else ("ablation_all" if len(c) == 4 else "ablation_" + "".join(c))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(MUTED)


def rel(runs, pde, m, s):
    return json.load(open(f"{runs}/seed_{s}/{pde}/{m}/metrics.json"))["relative_l2"]


def fig_per_seed(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor="white")
    for ax, pde, title in [(axes[0], "kuramoto_sivashinsky", "Kuramoto–Sivashinsky"),
                           (axes[1], "grayscott", "Gray–Scott")]:
        vals = {mname(c): [rel(runs, pde, mname(c), s) for s in SEEDS] for c in COMBOS}
        order = sorted(vals, key=lambda m: np.mean(vals[m]))
        for i, m in enumerate(order):
            col = BLUE if m == "ablation_none" else (ORANGE if m == "ablation_all" else GRAY)
            z = 5 if col != GRAY else 3
            ax.scatter(vals[m], [i] * 3, s=26 if col != GRAY else 16, color=col,
                       edgecolors="white", linewidths=0.5, zorder=z)
            ax.plot([np.mean(vals[m])] * 2, [i - 0.28, i + 0.28], color=col, lw=1.6, zorder=z)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([m.replace("ablation_", "") for m in order], fontsize=8,
                           color=MUTED)
        for i, m in enumerate(order):
            if m == "ablation_none":
                ax.annotate("origin", (np.mean(vals[m]), i), xytext=(6, 8),
                            textcoords="offset points", fontsize=8, color=INK, fontweight="bold")
            if m == "ablation_all":
                ax.annotate("best_practice", (np.mean(vals[m]), i), xytext=(6, 8),
                            textcoords="offset points", fontsize=8, color=INK, fontweight="bold")
        if pde == "grayscott":
            bad = vals["ablation_none"][SEEDS.index(1236)]
            ax.annotate("origin @ seed 1236\n(overshoot attractor)", (bad, order.index("ablation_none")),
                        xytext=(-10, -26), textcoords="offset points", fontsize=7.5, color=MUTED,
                        ha="right", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("final relative-L2 (dots = seeds, tick = mean)", fontsize=8.5, color=MUTED)
        style(ax)
    fig.suptitle("Ablation sweep, per seed: 16 ingredient combos × 3 seeds", fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def band_errors(runs, pde, m, s, n_bands=10):
    f = np.load(f"{runs}/seed_{s}/{pde}/{m}/solution/fields.npz")
    coords, pred, ref = f["coords"], f["pred"].astype(float), f["ref"].astype(float)
    t = coords[:, -1]
    edges = np.linspace(t.min(), t.max(), n_bands + 1)
    centers, errs, amp = [], [], []
    for i in range(n_bands):
        msk = (t >= edges[i]) & (t <= edges[i + 1])
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        errs.append(np.sqrt(((pred[msk] - ref[msk]) ** 2).mean()) / max(np.sqrt((ref[msk] ** 2).mean()), 1e-12))
        amp.append(np.sqrt((pred[msk] ** 2).mean()) / max(np.sqrt((ref[msk] ** 2).mean()), 1e-12))
    return np.array(centers), np.array(errs), np.array(amp)


def fig_ks_horizon(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")
    series = [("ablation_none", "origin", BLUE), ("ablation_A", "A (modified MLP)", AQUA),
              ("ablation_all", "best_practice", ORANGE), ("ablation_CWA", "CWA (no G rescue)", VIOLET)]
    ax = axes[0]
    for m, label, col in series:
        E = np.mean([band_errors(runs, "kuramoto_sivashinsky", m, s)[1] for s in SEEDS], axis=0)
        c = band_errors(runs, "kuramoto_sivashinsky", m, SEEDS[0])[0]
        ax.plot(c, E, "-o", ms=3.5, lw=1.8, color=col, label=label)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls="--")
    ax.annotate("failure threshold", (0.02, 0.5), xytext=(0, 5), textcoords="offset points",
                fontsize=7.5, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("time t", fontsize=8.5, color=MUTED)
    ax.set_ylabel("relative-L2 in time band", fontsize=8.5, color=MUTED)
    ax.set_title("The horizon wall: every method collapses at t ≈ 0.2–0.3", fontsize=10, color=INK)
    style(ax)

    ax = axes[1]
    for m, label, col in [("ablation_none", "origin", BLUE), ("ablation_all", "best_practice", ORANGE)]:
        A = np.mean([band_errors(runs, "kuramoto_sivashinsky", m, s)[2] for s in SEEDS], axis=0)
        c = band_errors(runs, "kuramoto_sivashinsky", m, SEEDS[0])[0]
        ax.plot(c, A, "-o", ms=3.5, lw=1.8, color=col)
        ax.annotate(label, (c[-1], A[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=col, va="center")
    ax.axhline(1.0, color=MUTED, lw=0.9, ls="--")
    ax.annotate("reference amplitude", (0.02, 1.0), xytext=(0, 4), textcoords="offset points",
                fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 1.42)
    ax.set_xlabel("time t", fontsize=8.5, color=MUTED)
    ax.set_ylabel("rms(prediction) / rms(reference)", fontsize=8.5, color=MUTED)
    ax.set_title("Track-then-decay toward the trivial branch (u → 0)", fontsize=10, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_ks_decomposition(runs, out):
    chain = [("ablation_A", "A"), ("ablation_CA", "+C\n(CA)"), ("ablation_CWA", "+W\n(CWA)"),
             ("ablation_all", "+G\n(all)")]
    vals = [[rel(runs, "kuramoto_sivashinsky", m, s) for s in SEEDS] for m, _ in chain]
    means = [np.mean(v) for v in vals]
    origin = np.mean([rel(runs, "kuramoto_sivashinsky", "ablation_none", s) for s in SEEDS])

    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor="white")
    x = np.arange(len(chain))
    cols = [AQUA, GRAY, VIOLET, ORANGE]
    ax.bar(x, means, 0.55, color=cols, edgecolor="white", linewidth=1)
    for i, v in enumerate(vals):
        ax.scatter([i] * 3, v, s=16, color=INK, alpha=0.55, zorder=5)
    for i in range(1, len(means)):
        d = means[i] - means[i - 1]
        ax.annotate(f"{d:+.3f}", ((x[i] + x[i - 1]) / 2, max(means[i], means[i - 1]) + 0.03),
                    ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.axhline(origin, color=BLUE, lw=1.4, ls="--")
    ax.annotate(f"origin = {origin:.3f}", (len(chain) - 0.55, origin), xytext=(0, 5),
                textcoords="offset points", fontsize=8.5, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in chain], fontsize=9, color=MUTED)
    ax.set_ylabel("final relative-L2 (KS)", fontsize=8.5, color=MUTED)
    ax.set_ylim(0.85, 1.55)
    ax.set_title("Why A beats the full stack: the C-tax, the W-wound, the G-bandage",
                 fontsize=10, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_dissociation(runs, out):
    from sklearn.decomposition import PCA
    all_runs = sa.collect_runs(runs)
    pruns = [(m, s, rd) for p, m, s, rd in all_runs if p == "kuramoto_sivashinsky"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="white")
    for ax, set_name, fn, title, note in [
        (axes[0], "weights", sa.weight_features,
         "Final weights (plain-FNN methods): clusters by SEED",
         "η²(seed)=0.63   η²(method)=0.12"),
        (axes[1], "trajectory", sa.trajectory_features,
         "Training trajectories (same colors): no seed clusters",
         "η²(seed)=0.01   η²(method)=0.94"),
    ]:
        feats, seeds = [], []
        for m, s, rd in pruns:
            f = fn(rd)
            if f is not None:
                feats.append(f)
                seeds.append(s)
        X = sa.standardize(np.vstack(feats))
        P = PCA(n_components=2).fit_transform(X)
        for s in SEEDS:
            idx = [i for i, ss in enumerate(seeds) if ss == s]
            ax.scatter(P[idx, 0], P[idx, 1], s=34, color=SEED_COLORS[s],
                       edgecolors="white", linewidths=0.6, label=f"seed {s}")
        ax.set_title(title, fontsize=10, color=INK)
        ax.annotate(note, (0.02, 0.03), xycoords="axes fraction", fontsize=8.5, color=MUTED)
        ax.set_xlabel("PC1", fontsize=8.5, color=MUTED)
        ax.set_ylabel("PC2", fontsize=8.5, color=MUTED)
        style(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Method changes the path, seed locks the destination (KS)", fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_gs_insurance(runs, out):
    """origin's full per-epoch error curves (valid: single window, full-domain net) vs the
    stack's FINAL stitched errors (its per-checkpoint curve would be misleading: with
    time-marching each checkpoint is one window's net evaluated outside its window)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor="white")
    ls = {1234: "-", 1235: "--", 1236: ":"}
    for s in SEEDS:
        h = np.loadtxt(f"{runs}/seed_{s}/grayscott/ablation_none/metrics_history.csv",
                       delimiter=",", skiprows=1)
        ax.plot(h[:, 0] / h[-1, 0], h[:, 5], ls[s], color=BLUE,
                lw=1.9 if s == 1236 else 1.2, alpha=1.0 if s == 1236 else 0.6)
    for s in SEEDS:
        v = rel(runs, "grayscott", "ablation_all", s)
        ax.plot(1.02, v, "*", ms=13, color=ORANGE, markeredgecolor="white", zorder=6)
    ax.set_ylim(0, 0.52)  # clip the early transient so the attractor split is readable
    ax.set_xlim(-0.02, 1.12)
    ax.annotate("origin @ seed 1236: overshoot attractor —\nerror RISES from ~5k iterations on",
                (0.42, 0.38), fontsize=8, color=BLUE)
    ax.annotate("origin @ seeds 1234/1235: 0.094", (0.45, 0.055), fontsize=8, color=BLUE)
    ax.annotate("best_practice,\nfinal (stitched):\nall seeds 0.09–0.11", (1.035, 0.16),
                fontsize=8, color=ORANGE, ha="left")
    handles = [plt.Line2D([], [], color=BLUE, lw=1.6, label="origin, error vs epoch (3 seeds)"),
               plt.Line2D([], [], color=MUTED, lw=1.4, ls=":", label="seed 1236"),
               plt.Line2D([], [], ls="", marker="*", ms=11, color=ORANGE,
                          label="best_practice, final error")]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")
    ax.set_xlabel("training progress (normalized; early transient clipped)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("relative-L2 (Gray–Scott)", fontsize=8.5, color=MUTED)
    ax.set_title("The stack's GS 'win' is insurance: it saves the one seed origin loses",
                 fontsize=10, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=str, default="runs_landscape_compare")
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_per_seed(args.runs, os.path.join(args.out, "per_seed_ablation.png"))
    fig_ks_horizon(args.runs, os.path.join(args.out, "ks_horizon.png"))
    fig_ks_decomposition(args.runs, os.path.join(args.out, "ks_decomposition.png"))
    fig_dissociation(args.runs, os.path.join(args.out, "dissociation.png"))
    fig_gs_insurance(args.runs, os.path.join(args.out, "gs_insurance.png"))
    for f in sorted(os.listdir(args.out)):
        print(f"[fig] {os.path.join(args.out, f)} ({os.path.getsize(os.path.join(args.out, f))//1024} KB)")


if __name__ == "__main__":
    main()
