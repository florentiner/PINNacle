"""Compare vanilla DeepXDE PINN vs Causal PINN on a chaotic case using ONLY the
saved forensic arrays (no retraining, no model loading).

Usage:
  python analysis/compare_chaotic.py --case ks \
      --baseline runs/<...>-baseline/0-0 --causal runs/<...>-causal-ks/0-0 \
      [--ablation runs/<...>/0-0] [--out analysis/out/ks]

Reads per run dir:
  arrays/ref.npy, arrays/pred_*.npy, arrays/err_*.npy, arrays/resid_*.npy,
  arrays/grid_meta.json, metrics.csv, causal/history.npz (causal runs)
Writes side-by-side landscapes, residual evolution, error-growth-in-time curves,
the causal-weight front, metric curves, and summary.json.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- loading ----------------
class Run:
    def __init__(self, path, label):
        self.path = path.rstrip("/")
        self.label = label
        self.meta = json.load(open(os.path.join(path, "arrays/grid_meta.json")))
        self.ref = np.load(os.path.join(path, "arrays/ref.npy"))
        self.axes = [np.asarray(a) for a in self.meta["axes"]]

    def _steps(self, pattern):
        out = {}
        for f in glob.glob(os.path.join(self.path, "arrays", pattern)):
            m = re.search(r"_(\d+)\.npy$", f)
            if m:
                out[int(m.group(1))] = f
        return dict(sorted(out.items()))

    def snapshots(self, kind):  # kind in {pred, err, resid}
        """Baseline-style step-indexed snapshots {step: array}."""
        return {k: np.load(v) for k, v in self._steps(f"{kind}_*.npy").items()}

    def final_err(self):
        for cand in ["err_stitched_final.npy"]:
            p = os.path.join(self.path, "arrays", cand)
            if os.path.exists(p):
                return np.load(p)
        snaps = self._steps("err_*.npy")
        if snaps:
            return np.load(list(snaps.values())[-1])
        raise FileNotFoundError(f"no final error array in {self.path}")

    def window_series(self, kind):
        """Causal-style per-window finals: {window: array (n_t, n_pts, n_comp)}."""
        out = {}
        for f in glob.glob(os.path.join(self.path, "arrays", f"{kind}_w*_final.npy")):
            m = re.search(r"_w(\d+)_final\.npy$", f)
            out[int(m.group(1))] = np.load(f)
        return dict(sorted(out.items()))

    def metrics(self):
        p = os.path.join(self.path, "metrics.csv")
        if not os.path.exists(p):
            return None
        import csv
        rows = list(csv.DictReader(open(p)))
        return rows

    def history(self):
        p = os.path.join(self.path, "causal/history.npz")
        return np.load(p) if os.path.exists(p) else None


def as_xt(arr, case):
    """-> (nx_like, nt, n_comp) with time as axis 1 for plotting e(t) / landscapes.
    KS grids are (512, 251, c); GS grids (100, 100, 21, c) -> flatten space."""
    if case == "ks":
        return arr
    return arr.reshape(-1, arr.shape[2], arr.shape[3])


def stitch_windows(run, case, kind, upto=None):
    """Rebuild a full-domain grid from per-window finals (NaN where not covered)."""
    grid = np.full_like(run.ref, np.nan, dtype=np.float32)
    wins = run.window_series(kind)
    n_win = run.meta.get("n_windows", len(wins))
    spw = run.meta.get("steps_per_win", 1)
    for k, arr in wins.items():
        if upto is not None and k > upto:
            continue
        cols = np.arange(k * spw, (k + 1) * spw + 1)
        if case == "ks":
            grid[:, cols, :] = np.moveaxis(arr, 0, 1)
        else:
            g = arr.reshape(len(cols), grid.shape[0], grid.shape[1], grid.shape[3])
            grid[:, :, cols, :] = np.moveaxis(g, 0, 2)
    return grid


# ---------------- figures ----------------
def fig_landscapes(runs, case, out, kind="err", logabs=True):
    cols = len(runs)
    n_comp = runs[0].ref.shape[-1]
    fig, axs = plt.subplots(n_comp, cols, figsize=(5.2 * cols, 4 * n_comp),
                            squeeze=False)
    # shared color scale across all runs -> panels directly comparable
    imgs = {}
    vmin, vmax = np.inf, -np.inf
    for j, r in enumerate(runs):
        if kind == "err":
            arr = r.final_err()
        else:
            arr = stitch_windows(r, case, "resid") if r.window_series("resid") \
                else list(r.snapshots("resid").values())[-1]
        A = as_xt(arr, case)
        for i in range(n_comp):
            v = np.abs(A[:, :, i]) if logabs else A[:, :, i]
            with np.errstate(all="ignore"):
                img = np.log10(v + 1e-10) if logabs else v
            imgs[(i, j)] = img
            if np.isfinite(img).any():
                vmin = min(vmin, np.nanmin(img[np.isfinite(img)]))
                vmax = max(vmax, np.nanmax(img[np.isfinite(img)]))
    for j, r in enumerate(runs):
        t = r.axes[-1]
        for i in range(n_comp):
            img = imgs[(i, j)]
            im = axs[i][j].imshow(img[::-1], aspect="auto",
                                  extent=[t.min(), t.max(), 0, 1],
                                  cmap="viridis", vmin=vmin, vmax=vmax)
            axs[i][j].set_title(f"{r.label} | comp {i} "
                                + ("log10|err|" if kind == "err" else "log10|resid|"))
            axs[i][j].set_xlabel("t")
            axs[i][j].set_ylabel("space (flattened)" if case == "gs" else "x/2pi")
            fig.colorbar(im, ax=axs[i][j])
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"landscape_{kind}.png"), dpi=140)
    plt.close(fig)


def fig_error_growth(runs, case, out):
    """spatial L2RE per time slice — the causality fingerprint."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in runs:
        err = as_xt(r.final_err(), case)
        ref = as_xt(r.ref, case)
        t = r.axes[-1]
        num = np.sqrt(np.nanmean(err ** 2, axis=(0, 2)))
        den = np.sqrt(np.nanmean(ref ** 2, axis=(0, 2)))
        ax.semilogy(t, num / den, label=r.label)
    ax.set_xlabel("t")
    ax.set_ylabel("spatial L2RE(t)")
    ax.set_title("Error growth over time (final models)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "error_growth.png"), dpi=140)
    plt.close(fig)


def fig_causal_front(run, out):
    h = run.history()
    if h is None or len(h["step"]) == 0:
        return
    W, t_r = h["W"], h["t_r"]
    step, window = h["step"], h["window"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    # left: W vs t within batches, colored by training progress (first window)
    w0 = window == window.min()
    idx = np.where(w0)[0]
    sel = idx[:: max(1, len(idx) // 40)]
    cmap = plt.cm.plasma(np.linspace(0, 1, len(sel)))
    for c, i in zip(cmap, sel):
        order = np.argsort(t_r[i])
        axs[0].plot(t_r[i][order], W[i][order], color=c, alpha=0.7, lw=1)
    axs[0].set_xlabel("t (window-local)")
    axs[0].set_ylabel("causal weight W")
    axs[0].set_title(f"{run.label}: causal front propagation (window {int(window.min())})")
    # right: W_min trajectory across all logged steps, colored by window
    sc = axs[1].scatter(np.arange(len(step)), h["w_min"], c=window, cmap="tab10", s=8)
    axs[1].set_yscale("symlog", linthresh=1e-3)
    axs[1].set_xlabel("log entry")
    axs[1].set_ylabel("min W")
    axs[1].set_title("W_min across training (color = window)")
    fig.colorbar(sc, ax=axs[1], label="window")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "causal_front.png"), dpi=140)
    plt.close(fig)


def fig_metric_curves(runs, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in runs:
        rows = r.metrics()
        if not rows:
            continue
        xs, ys = [], []
        for row in rows:
            l2 = row.get("l2re") or row.get("l2re_global") or row.get("l2re_window")
            try:
                l2 = float(l2)
                if np.isfinite(l2):
                    xs.append(float(row.get("walltime_s", np.nan)))
                    ys.append(l2)
            except (TypeError, ValueError):
                continue
        if ys:
            ax.semilogy(xs, ys, label=r.label, marker=".", ms=3, lw=1)
    ax.set_xlabel("walltime (s)")
    ax.set_ylabel("L2RE")
    ax.set_title("L2RE vs training walltime")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "l2re_vs_walltime.png"), dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["ks", "gs"], required=True)
    p.add_argument("--baseline", type=str, required=True)
    p.add_argument("--causal", type=str, required=True)
    p.add_argument("--ablation", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    out = args.out or f"analysis/out/{args.case}"
    os.makedirs(out, exist_ok=True)

    runs = [Run(args.baseline, "vanilla PINN (DeepXDE)"),
            Run(args.causal, "Causal PINN (SOTA)")]
    if args.ablation:
        runs.append(Run(args.ablation, "arch-only ablation (no causal W)"))

    fig_landscapes(runs, args.case, out, kind="err")
    try:
        fig_landscapes(runs, args.case, out, kind="resid")
    except Exception as e:
        print("resid landscape skipped:", e)
    fig_error_growth(runs, args.case, out)
    fig_causal_front(runs[1], out)
    fig_metric_curves(runs, out)

    summary = {}
    for r in runs:
        err, ref = r.final_err(), r.ref
        m = ~np.isnan(err)
        summary[r.label] = {
            "l2re_final": float(np.sqrt(np.nanmean(err[m] ** 2) / np.mean(ref[m] ** 2))),
            "coverage": float(np.mean(m)),
            "path": r.path,
        }
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("figures ->", out)


if __name__ == "__main__":
    main()
