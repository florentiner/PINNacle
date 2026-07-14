"""Similarity & variance analysis: do method (ablation type) or seed dominate the differences
between runs, in error-landscape space, trajectory space, solution space and parameter space?

Motivation (HYPOTHESIS.md H8-H10): if the 16 ablation variants produce landscapes/trajectories
that differ from each other LESS than reruns of the same variant with a different seed do, the
ingredient choice is not reshaping the optimization problem at all -- strong evidence that the
chaotic failure is a property of the problem (predictability horizon, trivial attractor), not
of the method. Conversely, methods that cluster apart (e.g. diverging time-marching combos)
are genuinely different optimization processes.

Feature sets per run (pde, method, seed), each analyzed separately:
  landscape  : embedding-invariant descriptors of the 2D loss landscape -- deciles of the
               log10 loss grid + roughness (TV & high-freq), init->end barrier, basin
               fraction, #local minima, corr(log loss_oper, log loss_bnd).
  trajectory : the TRUE-error curve along training (trajectory_error.csv) and the log10
               training-loss curve (trajectory_losses.npz), each resampled to 24 points on
               normalized progress [0, 1] -> 48 dims, absolutely comparable across runs.
  solution   : the predicted field on a fixed 512-point subsample of the common reference
               grid (comparable across ALL methods, both architectures).
  weights    : the final checkpoint's flattened parameters (plain-FNN runs only -- the
               modified-MLP lives in a different parameter space), where the shared-init
               property makes "clusters by seed vs by method" directly meaningful.

For every (pde x feature set):
  * PCA and t-SNE 2D embeddings, points colored by method (origin/ablation_none and
    ablation_all/best_practice highlighted), marker shape = seed.
  * Pooled eta^2 variance decomposition: fraction of total feature variance explained by
    grouping on method vs grouping on seed (one-way, computed per factor).
  * Silhouette score of the standardized features under method-labels vs seed-labels
    (higher = that factor forms tighter, better-separated clusters).

Outputs under <runs>/similarity/:  variance_decomposition.csv, <pde>_<set>_embedding.pdf
CPU-only; needs numpy/scipy/matplotlib/scikit-learn + torch (only to deserialize checkpoints).

Usage:
    python experiments/landscape_compare/similarity_analysis.py --runs runs_landscape_compare
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_landscapes as cl  # noqa: E402  (roughness/barrier/grid helpers, no heavy deps)

MARKERS = ["o", "s", "^", "D", "v", "P"]  # per seed
N_TRAJ_POINTS = 24
N_FIELD_POINTS = 512


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def landscape_features(run_dir):
    g2 = os.path.join(run_dir, "landscape", "grid_2d.npz")
    gall = os.path.join(run_dir, "landscape", "grid_losses_all.npz")
    tj = os.path.join(run_dir, "landscape", "trajectory_2d.npy")
    if not os.path.exists(g2):
        return None
    d = np.load(g2)
    x_axis, y_axis, L = cl._regular_grid(d["grid_xx"], d["grid_yy"], cl._log_surface(d["grid_losses"]))
    deciles = np.percentile(L, np.arange(5, 100, 10))  # embedding-orientation invariant
    tv, hf = cl.roughness(L)
    traj = np.load(tj) if os.path.exists(tj) else None
    end = traj[-1] if traj is not None and len(traj) else [x_axis.mean(), y_axis.mean()]
    basin, curv = cl.basin_and_sharpness(x_axis, y_axis, L, end)
    bar = cl.barrier(x_axis, y_axis, L, traj)
    nmin = cl.n_local_minima(L)
    corr_ob = np.nan
    if os.path.exists(gall):
        g = np.load(gall)
        lo = np.log10(np.clip(g["loss_oper"], 1e-30, None)).ravel()
        lb = np.log10(np.clip(g["loss_bnd"], 1e-30, None)).ravel()
        if np.std(lo) > 0 and np.std(lb) > 0:
            corr_ob = float(np.corrcoef(lo, lb)[0, 1])
    feats = np.concatenate([deciles, [tv, hf, bar, curv, basin, float(nmin), corr_ob]])
    return np.nan_to_num(feats, nan=0.0)


def _resample(y, n):
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(y) < 2:
        return None
    x = np.linspace(0, 1, len(y))
    return np.interp(np.linspace(0, 1, n), x, y)


def trajectory_features(run_dir):
    te = os.path.join(run_dir, "trajectory_error.csv")
    tl = os.path.join(run_dir, "landscape", "trajectory_losses.npz")
    if not (os.path.exists(te) and os.path.exists(tl)):
        return None
    err = np.loadtxt(te, delimiter=",", skiprows=1)
    err = err[:, 1] if err.ndim == 2 else err.reshape(-1)
    loss = np.log10(np.clip(np.asarray(np.load(tl)["loss_total"]).reshape(-1), 1e-30, None))
    e, l = _resample(err, N_TRAJ_POINTS), _resample(loss, N_TRAJ_POINTS)
    if e is None or l is None:
        return None
    return np.concatenate([e, l])


def solution_features(run_dir, sub_idx):
    fp = os.path.join(run_dir, "solution", "fields.npz")
    if not os.path.exists(fp):
        return None
    pred = np.load(fp)["pred"].astype(float)
    return pred.reshape(pred.shape[0], -1)[sub_idx].reshape(-1)


def weight_features(run_dir):
    """Final checkpoint's flat parameter vector -- plain FNN only (arch-comparable)."""
    import torch
    cfg_p = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_p) and json.load(open(cfg_p)).get("arch", "fnn") != "fnn":
        return None
    ck = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ck):
        return None
    files = sorted(f for f in os.listdir(ck) if f.endswith(".pt"))
    if not files:
        return None
    sd = torch.load(os.path.join(ck, files[-1]), map_location="cpu")
    return np.concatenate([v.numpy().astype(float).ravel() for v in sd.values()])


# --------------------------------------------------------------------------- #
# Variance decomposition + embedding
# --------------------------------------------------------------------------- #
def eta_squared(X, labels):
    """Pooled one-way eta^2: sum-over-dims between-group SS / total SS for this grouping."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    ss_total = ((X - mu) ** 2).sum()
    ss_between = 0.0
    for g in set(labels):
        Xg = X[[i for i, l in enumerate(labels) if l == g]]
        ss_between += len(Xg) * ((Xg.mean(axis=0) - mu) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


def standardize(X):
    X = np.asarray(X, dtype=float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def embed_and_plot(X, methods, seeds, pde, set_name, out_dir):
    Xs = standardize(X)
    n = len(Xs)
    pca = PCA(n_components=2).fit(Xs)
    P = pca.transform(Xs)
    perp = max(2, min(10, (n - 1) // 3))
    T = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(Xs)

    uniq_m = sorted(set(methods))
    cmap = plt.get_cmap("tab20")
    mcolor = {m: cmap(i % 20) for i, m in enumerate(uniq_m)}
    mcolor["ablation_none"] = "red"
    mcolor["ablation_all"] = "cyan"
    uniq_s = sorted(set(seeds))
    smark = {s: MARKERS[i % len(MARKERS)] for i, s in enumerate(uniq_s)}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, E, title in [(axes[0], P, f"PCA ({pca.explained_variance_ratio_[:2].sum():.0%} var)"),
                         (axes[1], T, f"t-SNE (perplexity {perp})")]:
        for i in range(n):
            hl = methods[i] in ("ablation_none", "ablation_all")
            ax.scatter(E[i, 0], E[i, 1], c=[mcolor[methods[i]]], marker=smark[seeds[i]],
                       s=90 if hl else 55, edgecolors="k", linewidths=1.2 if hl else 0.4,
                       zorder=5 if hl else 3)
        ax.set_title(f"{pde} / {set_name}: {title}")
        ax.grid(alpha=0.25)
    handles = ([plt.Line2D([], [], ls="", marker="o", color=mcolor[m],
                           markeredgecolor="k", label=m) for m in uniq_m]
               + [plt.Line2D([], [], ls="", marker=smark[s], color="grey", label=f"seed {s}")
                  for s in uniq_s])
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{pde}_{set_name}_embedding.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")


# --------------------------------------------------------------------------- #
def collect_runs(runs_root):
    """[(pde, method, seed, run_dir)] over the seed_*/ layout (or flat)."""
    seeds = [(int(n[5:]), os.path.join(runs_root, n)) for n in sorted(os.listdir(runs_root))
             if n.startswith("seed_") and os.path.isdir(os.path.join(runs_root, n))]
    if not seeds:
        seeds = [(None, runs_root)]
    out = []
    for seed, root in seeds:
        for pde in sorted(os.listdir(root)):
            pde_dir = os.path.join(root, pde)
            if not os.path.isdir(pde_dir):
                continue
            for method in sorted(os.listdir(pde_dir)):
                rd = os.path.join(pde_dir, method)
                if os.path.isdir(os.path.join(rd, "checkpoints")):  # gradient runs only
                    out.append((pde, method, seed, rd))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=str, default="runs_landscape_compare")
    args = parser.parse_args()

    runs = collect_runs(args.runs)
    pdes = sorted({p for p, _, _, _ in runs})
    out_dir = os.path.join(args.runs, "similarity")
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(runs)} gradient runs across {pdes}")

    extractors = {"landscape": landscape_features, "trajectory": trajectory_features,
                  "solution": None, "weights": weight_features}
    rows = []
    for pde in pdes:
        pruns = [(m, s, rd) for p, m, s, rd in runs if p == pde]
        # fixed subsample of the reference grid, shared by every run of this pde
        first_fields = np.load(os.path.join(pruns[0][2], "solution", "fields.npz"))
        n_pts = first_fields["pred"].shape[0]
        sub_idx = np.random.default_rng(0).choice(n_pts, size=min(N_FIELD_POINTS, n_pts), replace=False)

        for set_name, fn in extractors.items():
            feats, methods, seeds = [], [], []
            for m, s, rd in pruns:
                f = solution_features(rd, sub_idx) if set_name == "solution" else fn(rd)
                if f is not None:
                    feats.append(f)
                    methods.append(m)
                    seeds.append(s)
            if len(set(methods)) < 2 or len(set(seeds)) < 2:
                print(f"[skip] {pde}/{set_name}: not enough groups")
                continue
            X = np.vstack(feats)
            Xs = standardize(X)
            eta_m, eta_s = eta_squared(Xs, methods), eta_squared(Xs, seeds)
            try:
                sil_m = silhouette_score(Xs, methods) if len(set(methods)) < len(methods) else float("nan")
            except Exception:
                sil_m = float("nan")
            try:
                sil_s = silhouette_score(Xs, seeds)
            except Exception:
                sil_s = float("nan")
            rows.append({"pde": pde, "feature_set": set_name, "n_runs": len(feats),
                         "n_dims": X.shape[1], "eta2_method": round(eta_m, 4),
                         "eta2_seed": round(eta_s, 4),
                         "silhouette_method": round(sil_m, 4), "silhouette_seed": round(sil_s, 4),
                         "dominant_factor": "method" if eta_m > eta_s else "seed"})
            embed_and_plot(X, methods, seeds, pde, set_name, out_dir)

    with open(os.path.join(out_dir, "variance_decomposition.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {os.path.join(out_dir, 'variance_decomposition.csv')}")

    print("\n========== VARIANCE DECOMPOSITION (eta^2: fraction of feature variance) ==========")
    print(f"{'pde/feature_set':<40}{'n':<5}{'eta2_method':<13}{'eta2_seed':<11}{'sil_method':<12}{'sil_seed':<10}{'dominant'}")
    for r in rows:
        print(f"{r['pde'] + '/' + r['feature_set']:<40}{r['n_runs']:<5}{r['eta2_method']:<13}"
              f"{r['eta2_seed']:<11}{r['silhouette_method']:<12}{r['silhouette_seed']:<10}{r['dominant_factor']}")
    print("\nReading guide: eta2_method >> eta2_seed  -> the ablation ingredients genuinely change"
          "\nthat aspect of the run; eta2_seed >= eta2_method -> reruns differ as much as method"
          "\nswaps, i.e. the ingredient choice is NOT reshaping that space (H8).")


if __name__ == "__main__":
    main()
