"""Offline analysis of the chaotic-PDE landscape comparison (CPU-only, no torch/deepxde).

Consumes the runs_landscape_compare/ tree produced by run_all.py / run_experiment.py and
turns it into the quantitative evidence for the hypotheses in HYPOTHESIS.md:

  Solution tier   : relative-L2, Fourier low/mid/high band error, IC error.
  Landscape tier  : from each gradient run's 2D loss grid --
                      * roughness   (total-variation + high-frequency FFT energy fraction)
                      * sharpness   (curvature at the trajectory endpoint; basin width)
                      * barrier     (max loss on the init->final segment minus the endpoint)
                      * n_local_minima
                    and, from the trajectory, the loss<->error alignment
                    (corr of PDE loss vs true relative-L2 across checkpoints -- a weak/negative
                     correlation is a *deceptive* landscape: low loss but high error).
  Frozen tier     : feature-matrix condition number + singular-value decay slope
                    (the well-conditioned/convex contrast to the SGD landscapes).

Outputs:  <runs>/compare_summary.csv  and  <runs>/comparison_figures/*.pdf

Run after the experiments (works on a laptop with just numpy/scipy/matplotlib):
    python experiments/landscape_compare/compare_landscapes.py --runs runs_landscape_compare
"""
import argparse
import csv
import json
import os

import numpy as np

try:
    from scipy.interpolate import griddata
    from scipy.ndimage import minimum_filter
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _log_surface(loss_grid):
    """Stable signed-log of a loss grid (losses can span many orders of magnitude)."""
    g = np.asarray(loss_grid, dtype=float)
    g = np.nan_to_num(g, nan=np.nanmax(g[np.isfinite(g)]) if np.isfinite(g).any() else 0.0)
    return np.log10(np.clip(g, 1e-30, None))


def _regular_grid(gx, gy, gz):
    """Resample a (possibly transposed) meshgrid onto a regular (x_axis, y_axis) grid L[i,j]."""
    x_axis = np.unique(gx)
    y_axis = np.unique(gy)
    if _HAVE_SCIPY:
        XX, YY = np.meshgrid(x_axis, y_axis, indexing="ij")
        L = griddata((gx.ravel(), gy.ravel()), gz.ravel(), (XX, YY), method="linear")
        L = np.nan_to_num(L, nan=np.nanmean(gz))
    else:  # assume gx varies along axis 0 already
        L = gz
    return x_axis, y_axis, L


def roughness(L):
    """Total-variation roughness + high-frequency FFT energy fraction of a log-loss surface."""
    rng = float(np.nanmax(L) - np.nanmin(L)) or 1.0
    gy, gx = np.gradient(L)
    tv = float(np.nanmean(np.sqrt(gx ** 2 + gy ** 2)) / rng)
    F = np.fft.fftshift(np.abs(np.fft.fft2(L - np.nanmean(L))) ** 2)
    ny, nx = L.shape
    cy, cx = ny // 2, nx // 2
    ky = (np.arange(ny) - cy) / max(cy, 1)
    kx = (np.arange(nx) - cx) / max(cx, 1)
    KR = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    total = float(F.sum()) or 1.0
    hf = float(F[KR > 0.5].sum() / total)
    return tv, hf


def n_local_minima(L):
    if not _HAVE_SCIPY:
        return float("nan")
    mn = minimum_filter(L, size=3, mode="nearest")
    return int(np.sum((L <= mn + 1e-9) & (L < np.nanmax(L))))


def basin_and_sharpness(x_axis, y_axis, L, end_xy):
    """low-loss basin fraction (flat=big) and endpoint curvature (sharp=big)."""
    lo, hi = np.nanmin(L), np.nanmax(L)
    thr = lo + 0.1 * (hi - lo)
    basin_frac = float(np.mean(L <= thr))
    # nearest grid cell to the trajectory endpoint
    i = int(np.argmin(np.abs(x_axis - end_xy[0])))
    j = int(np.argmin(np.abs(y_axis - end_xy[1])))
    i = min(max(i, 1), L.shape[0] - 2)
    j = min(max(j, 1), L.shape[1] - 2)
    lap = abs(L[i + 1, j] + L[i - 1, j] + L[i, j + 1] + L[i, j - 1] - 4 * L[i, j])
    return basin_frac, float(lap)


def barrier(x_axis, y_axis, L, traj_xy):
    """Max log-loss on the straight segment from the first to the last trajectory point,
    minus the endpoint log-loss (a positive barrier => a hump to climb before descending)."""
    if traj_xy is None or len(traj_xy) < 2:
        return float("nan")
    p0, p1 = np.asarray(traj_xy[0]), np.asarray(traj_xy[-1])
    ts = np.linspace(0, 1, 50)
    seg = p0[None, :] * (1 - ts[:, None]) + p1[None, :] * ts[:, None]
    if _HAVE_SCIPY:
        vals = griddata(np.stack(np.meshgrid(x_axis, y_axis, indexing="ij"), -1).reshape(-1, 2),
                        L.ravel(), seg, method="linear")
        vals = np.nan_to_num(vals, nan=np.nanmax(L))
    else:
        vals = np.array([L[int(np.argmin(np.abs(x_axis - p[0]))), int(np.argmin(np.abs(y_axis - p[1])))] for p in seg])
    return float(np.nanmax(vals) - vals[-1])


def loss_error_alignment(run_dir):
    """Pearson corr between PDE loss and true relative-L2 across checkpoints (deceptiveness)."""
    tl = os.path.join(run_dir, "landscape", "trajectory_losses.npz")
    te = os.path.join(run_dir, "trajectory_error.csv")
    if not (os.path.exists(tl) and os.path.exists(te)):
        return float("nan")
    try:
        losses = np.asarray(np.load(tl)["loss_total"]).reshape(-1)
        err = np.loadtxt(te, delimiter=",", skiprows=1)
        err = err[:, 1] if err.ndim == 2 else err.reshape(-1)
        n = min(len(losses), len(err))
        if n < 3:
            return float("nan")
        a, b = np.log10(np.clip(losses[:n], EPS, None)), np.log10(np.clip(err[:n], EPS, None))
        if np.std(a) < EPS or np.std(b) < EPS:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return float("nan")


def frozen_conditioning(run_dir):
    sp = os.path.join(run_dir, "frozen", "feature_spectrum.npz")
    if not os.path.exists(sp):
        return {}
    try:
        sv = np.asarray(np.load(sp)["singular_values"], dtype=float)
        sv = sv[sv > 0]
        cond = float(sv[0] / sv[-1]) if len(sv) else float("nan")
        # log-log decay slope of the singular spectrum
        if len(sv) >= 3:
            r = np.log(np.arange(1, len(sv) + 1))
            s = np.log(sv)
            slope = float(np.polyfit(r, s, 1)[0])
        else:
            slope = float("nan")
        return {"sv_condition_number": cond, "sv_decay_slope": slope}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# per-run collection
# --------------------------------------------------------------------------- #
def analyze_run(run_dir):
    row = {}
    mpath = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(mpath):
        return None
    with open(mpath) as f:
        m = json.load(f)
    for k in ["relative_l2", "mse", "mae", "ic_relative_l2", "boundary_relative_l2",
              "fourier_low", "fourier_mid", "fourier_high", "wall_clock_sec",
              "condition_number", "num_features", "integrator_success"]:
        if k in m:
            row[k] = m[k]

    grid_path = os.path.join(run_dir, "landscape", "grid_2d.npz")
    traj_path = os.path.join(run_dir, "landscape", "trajectory_2d.npy")
    if os.path.exists(grid_path):
        try:
            d = np.load(grid_path)
            x_axis, y_axis, L = _regular_grid(d["grid_xx"], d["grid_yy"], _log_surface(d["grid_losses"]))
            tv, hf = roughness(L)
            row["roughness_tv"] = tv
            row["roughness_hf"] = hf
            row["n_local_minima"] = n_local_minima(L)
            traj_xy = np.load(traj_path) if os.path.exists(traj_path) else None
            end_xy = traj_xy[-1] if traj_xy is not None and len(traj_xy) else [x_axis.mean(), y_axis.mean()]
            row["basin_fraction"], row["end_curvature"] = basin_and_sharpness(x_axis, y_axis, L, end_xy)
            row["barrier"] = barrier(x_axis, y_axis, L, traj_xy)
        except Exception as e:
            row["landscape_error"] = str(e)
    row["loss_error_corr"] = loss_error_alignment(run_dir)
    row.update(frozen_conditioning(run_dir))
    return row


def _seed_dirs(runs_root):
    """Sorted (seed:int, path) pairs if runs_root uses the run_all.py --n-repeats/--seeds
    nesting (<runs_root>/seed_<N>/<pde>/<method>/...), else []."""
    out = []
    if not os.path.isdir(runs_root):
        return out
    for name in sorted(os.listdir(runs_root)):
        if name.startswith("seed_") and os.path.isdir(os.path.join(runs_root, name)):
            try:
                out.append((int(name[len("seed_"):]), os.path.join(runs_root, name)))
            except ValueError:
                pass
    return out


def _collect_flat(root, seed, rows):
    for pde in sorted(os.listdir(root)):
        pde_dir = os.path.join(root, pde)
        if not os.path.isdir(pde_dir):
            continue
        for method in sorted(os.listdir(pde_dir)):
            run_dir = os.path.join(pde_dir, method)
            if not os.path.isdir(run_dir):
                continue
            r = analyze_run(run_dir)
            if r is not None:
                rows[(pde, method, seed)] = r


def collect(runs_root):
    """Returns {(pde, method, seed): row}. seed is None for a plain (non-repeated) run."""
    rows = {}
    seed_dirs = _seed_dirs(runs_root)
    if seed_dirs:
        for seed, path in seed_dirs:
            _collect_flat(path, seed, rows)
    else:
        _collect_flat(runs_root, None, rows)
    return rows


NUMERIC_COLUMNS = ["relative_l2", "ic_relative_l2", "boundary_relative_l2",
                   "fourier_low", "fourier_mid", "fourier_high",
                   "roughness_tv", "roughness_hf", "barrier", "end_curvature", "basin_fraction",
                   "n_local_minima", "loss_error_corr", "condition_number", "sv_condition_number",
                   "sv_decay_slope", "wall_clock_sec"]


def aggregate(rows):
    """Group per-seed rows by (pde, method) -> {n_seeds, <col>_mean, <col>_std}.

    This is what answers "to be sure": std small & consistent sign = robust; std large or
    sign-flipping across seeds = the single-seed number was noise, not signal.
    """
    groups = {}
    for (pde, method, _seed), r in rows.items():
        groups.setdefault((pde, method), []).append(r)
    agg = {}
    for key, rs in groups.items():
        out = {"n_seeds": len(rs)}
        for col in NUMERIC_COLUMNS:
            vals = [r[col] for r in rs if isinstance(r.get(col), (int, float)) and np.isfinite(r.get(col))]
            if vals:
                out[f"{col}_mean"] = float(np.mean(vals))
                out[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            else:
                out[f"{col}_mean"] = float("nan")
                out[f"{col}_std"] = float("nan")
        agg[key] = out
    return agg


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
COLUMNS = ["pde", "method", "seed"] + NUMERIC_COLUMNS
AGG_COLUMNS = ["pde", "method", "n_seeds"]
for _c in NUMERIC_COLUMNS:
    AGG_COLUMNS += [f"{_c}_mean", f"{_c}_std"]


def write_csv(rows, path):
    """One row per (pde, method, seed) -- the raw, ungrouped per-run numbers."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for (pde, method, seed), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] if kv[0][2] is not None else -1)):
            w.writerow({"pde": pde, "method": method, "seed": seed if seed is not None else "", **r})
    print(f"[csv] {path}")


def write_agg_csv(agg, path):
    """One row per (pde, method): mean +/- std across repeats -- the 'to be sure' view."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGG_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for (pde, method), r in sorted(agg.items()):
            w.writerow({"pde": pde, "method": method, **r})
    print(f"[csv] {path}")


def _grouped_bar(agg, metric, title, path, logy=False):
    """agg: {(pde, method): {..., f'{metric}_mean', f'{metric}_std'}}. Error bars show the
    across-seed std (zero/absent when n_seeds==1, so a single-seed run renders identically
    to before)."""
    pdes = sorted({p for p, _ in agg})
    methods = sorted({m for _, m in agg})
    x = np.arange(len(pdes))
    w = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(pdes) + 3, 4))
    for k, method in enumerate(methods):
        means = [agg.get((p, method), {}).get(f"{metric}_mean", np.nan) for p in pdes]
        stds = [agg.get((p, method), {}).get(f"{metric}_std", 0.0) for p in pdes]
        means = [v if isinstance(v, (int, float)) else np.nan for v in means]
        stds = [v if isinstance(v, (int, float)) and np.isfinite(v) else 0.0 for v in stds]
        yv = np.abs(means) if logy else means
        ax.bar(x + k * w, yv, w, yerr=stds, capsize=3, label=method)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(pdes, rotation=15)
    ax.set_title(title)
    ax.set_ylabel(metric)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


def _fourier_bands(agg, path):
    keys = sorted(agg)
    labels = [f"{p}\n{m}" for p, m in keys]
    low = [agg[k].get("fourier_low_mean", np.nan) for k in keys]
    mid = [agg[k].get("fourier_mid_mean", np.nan) for k in keys]
    high = [agg[k].get("fourier_high_mean", np.nan) for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(1.1 * len(keys) + 3, 4))
    ax.bar(x, low, 0.6, label="low-k")
    ax.bar(x, mid, 0.6, bottom=np.nan_to_num(low), label="mid-k")
    ax.bar(x, high, 0.6, bottom=np.nan_to_num(low) + np.nan_to_num(mid), label="high-k")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_ylabel("Fourier-band error energy (rel., mean across seeds)")
    ax.set_title("Spectral error by band (chaotic structure lives in mid/high-k)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


def _loss_error_scatter(agg, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    plotted = False
    for (pde, method), r in sorted(agg.items()):
        rl2 = r.get("relative_l2_mean")
        corr = r.get("loss_error_corr_mean")
        if isinstance(rl2, (int, float)) and isinstance(corr, (int, float)) and np.isfinite(corr) and np.isfinite(rl2):
            ax.scatter(corr, rl2, s=60)
            ax.annotate(f"{pde[:4]}/{method}", (corr, rl2), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
            plotted = True
    ax.set_xlabel("loss<->error correlation along trajectory  (low = deceptive landscape)")
    ax.set_ylabel("final relative-L2 (mean across seeds)")
    ax.set_yscale("log")
    ax.axvline(0.5, color="grey", ls="--", alpha=0.5)
    ax.set_title("Deceptive-landscape view (H2)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if plotted:
        fig.savefig(path)
        print(f"[fig] {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=str, default="runs_landscape_compare")
    args = parser.parse_args()

    rows = collect(args.runs)
    if not rows:
        print(f"No completed runs found under {args.runs}. Run run_all.py first.")
        return

    write_csv(rows, os.path.join(args.runs, "compare_summary.csv"))
    agg = aggregate(rows)
    write_agg_csv(agg, os.path.join(args.runs, "compare_summary_agg.csv"))

    fig_dir = os.path.join(args.runs, "comparison_figures")
    os.makedirs(fig_dir, exist_ok=True)

    _grouped_bar(agg, "relative_l2", "Solution accuracy (relative-L2, lower=better; error bars = std across seeds)",
                 os.path.join(fig_dir, "relative_l2.pdf"), logy=True)
    _grouped_bar(agg, "roughness_tv", "Landscape roughness (total variation of log-loss)",
                 os.path.join(fig_dir, "roughness.pdf"))
    _grouped_bar(agg, "barrier", "Init->final barrier on the loss landscape",
                 os.path.join(fig_dir, "barrier.pdf"))
    _fourier_bands(agg, os.path.join(fig_dir, "fourier_bands.pdf"))
    _loss_error_scatter(agg, os.path.join(fig_dir, "deceptive_landscape.pdf"))

    # console summary (mean +/- std across seeds; std is 0 for a plain single-seed run)
    print("\n==================== COMPARE SUMMARY (mean +/- std across seeds) ====================")
    print(f"{'cell':<34}{'n':<4}{'rel-L2':<22}{'rough_tv':<10}{'barrier':<10}{'loss~err':<10}{'cond':<12}")
    for (pde, method), r in sorted(agg.items()):
        def g(k, fmt="{:.3e}"):
            v = r.get(k)
            return fmt.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "-"
        rl2_mean, rl2_std, n = r.get("relative_l2_mean"), r.get("relative_l2_std"), r.get("n_seeds", 1)
        if isinstance(rl2_mean, (int, float)) and np.isfinite(rl2_mean):
            rl2_s = f"{rl2_mean:.3e}+/-{rl2_std:.1e}" if n > 1 else f"{rl2_mean:.3e}"
        else:
            rl2_s = "-"
        cond = r.get("condition_number_mean")
        if not (isinstance(cond, (int, float)) and np.isfinite(cond)):
            cond = r.get("sv_condition_number_mean")
        cond_s = "{:.2e}".format(cond) if isinstance(cond, (int, float)) and np.isfinite(cond) else "-"
        print(f"{pde+'/'+method:<34}{n:<4}{rl2_s:<22}{g('roughness_tv_mean','{:.3f}'):<10}"
              f"{g('barrier_mean','{:.3f}'):<10}{g('loss_error_corr_mean','{:.2f}'):<10}{cond_s:<12}")
    print(f"\nPer-seed rows:     {os.path.join(args.runs, 'compare_summary.csv')}")
    print(f"Mean +/- std (agg): {os.path.join(args.runs, 'compare_summary_agg.csv')}")
    print(f"Figures:           {fig_dir}")


if __name__ == "__main__":
    main()
