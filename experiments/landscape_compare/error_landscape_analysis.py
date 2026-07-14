"""Error-landscape analysis: WHY do the fix methods differ from `origin` on chaotic PDEs?

compare_landscapes.py ranks methods by final error and by generic loss-landscape geometry.
This script digs into the *error* landscape -- how the TRUE solution error is laid out over
the region of parameter space that training actually traversed -- to explain the mechanism
behind each method's behaviour. It uses only saved artifacts (checkpoints, loss grids,
reference fields); CPU-only, no deepxde/GPU needed (networks are re-run with a tiny manual
tanh-MLP forward pass straight from the checkpoint state dicts).

Per (pde, method, seed) it computes:
  * loss & true rel-L2 at every checkpoint (the "honesty" of the loss signal),
  * a true-error section along the training path in FULL parameter space: rel-L2 evaluated
    at interpolations between consecutive checkpoints (no autoencoder distortion),
  * trivial-attraction: rms(prediction)/rms(reference) per checkpoint -- distance to the
    trivial exact solution (KS: u=0 -> ratio 0; Gray-Scott: v=0 pattern erased),
  * deceptive-area fraction of the 2D loss grid: cells where the PDE-residual loss
    (loss_oper) is in its lowest decile while the IC/boundary loss (loss_bnd) is above its
    median -- the size of the "residual-cheap but wrong" region the trivial attractor
    carves into the landscape, plus corr(log loss_oper, log loss_bnd) over the grid.

Outputs (under <runs>/error_landscape/):
  error_landscape_summary.csv    one row per (pde, method, seed) + printed mean table
  <pde>_loss_vs_error.pdf        loss & error vs training progress, all methods
  <pde>_error_section.pdf        rel-L2 along the full-parameter-space training path
  <pde>_trivial_attraction.pdf   rms(pred)/rms(ref) per checkpoint
  <pde>_<method>_trajmap.pdf     loss contours + trajectory colored by TRUE error (seed 1234)

Usage:
    python experiments/landscape_compare/error_landscape_analysis.py --runs <results_dir>
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12
_FIXED_COLORS = {"origin": "tab:red", "causal": "tab:blue", "soap": "tab:orange",
                 "soap_causal": "tab:green", "best_practice": "tab:cyan",
                 "adam_baseline": "tab:brown", "lbfgs_baseline": "tab:purple",
                 "ablation_none": "tab:red", "ablation_C": "tab:blue",
                 "ablation_all": "tab:cyan"}
_FALLBACK_COLORS = list(plt.get_cmap("tab20").colors)


def method_color(method):
    """Stable color per method; fixed for the core methods, hashed for everything else
    (e.g. the 16 ablation_* combos), so new METHOD_SPEC entries never need edits here."""
    if method in _FIXED_COLORS:
        return _FIXED_COLORS[method]
    return _FALLBACK_COLORS[hash(method) % len(_FALLBACK_COLORS)]


# --------------------------------------------------------------------------- #
# Tiny forward pass straight from an FNN checkpoint (no torch/deepxde needed)
# --------------------------------------------------------------------------- #
# Exact-periodicity Fourier embedding used by the harness (run_experiment.PDE_REGISTRY):
# checkpoints trained with fourier_modes > 0 expect embedded inputs, so the manual forward
# must apply the same transform. (spatial_dims, base wavenumber) per PDE:
PERIODIC_SPEC = {"kuramoto_sivashinsky": (1, 1.0), "grayscott": (2, np.pi), "burgers1d": (1, None)}


def load_state(path):
    """Returns ("fnn", layers) or ("modified_mlp", (enc_u, enc_v, layers)) from a checkpoint;
    layers/encoders are (W, b) numpy pairs. Architecture is detected from the state-dict keys
    (ModifiedMLP additionally has encoder_u/encoder_v)."""
    import torch  # local import: only used to deserialize; math below is numpy
    sd = torch.load(path, map_location="cpu")
    layers = []
    i = 0
    while f"linears.{i}.weight" in sd:
        layers.append((sd[f"linears.{i}.weight"].numpy().astype(np.float64),
                       sd[f"linears.{i}.bias"].numpy().astype(np.float64)))
        i += 1
    if not layers:
        raise ValueError(f"No 'linears.*' keys in {path} -- not a recognized checkpoint")
    if "encoder_u.weight" in sd:
        enc_u = (sd["encoder_u.weight"].numpy().astype(np.float64),
                 sd["encoder_u.bias"].numpy().astype(np.float64))
        enc_v = (sd["encoder_v.weight"].numpy().astype(np.float64),
                 sd["encoder_v.bias"].numpy().astype(np.float64))
        return ("modified_mlp", (enc_u, enc_v, layers))
    return ("fnn", layers)


def embed_inputs(x, pde, fourier_modes):
    """Numpy replica of run_experiment.make_periodic_transform (identity if modes == 0)."""
    if not fourier_modes:
        return x
    spatial_dims, base_k = PERIODIC_SPEC[pde]
    ks = np.arange(1, fourier_modes + 1, dtype=np.float64) * base_k
    feats = []
    for d in range(spatial_dims):
        ang = x[:, d:d + 1] * ks
        feats.append(np.cos(ang))
        feats.append(np.sin(ang))
    feats.append(x[:, spatial_dims:])
    return np.concatenate(feats, axis=1)


def forward(net, x, pde=None, fourier_modes=0):
    """net: the (arch, params) pair from load_state. x: (N, in_dim) float64 raw coords ->
    (N, out_dim). Applies the periodic embedding when the run was trained with one, then the
    architecture's forward pass (tanh FNN, or the modified MLP's encoder-gated layers)."""
    arch, params = net
    h = embed_inputs(x, pde, fourier_modes) if fourier_modes else x
    if arch == "fnn":
        layers = params
        if h.shape[1] != layers[0][0].shape[1]:
            raise ValueError(f"input dim {h.shape[1]} != first-layer dim {layers[0][0].shape[1]} "
                             f"(fourier_modes={fourier_modes} mismatch with checkpoint?)")
        for k, (W, b) in enumerate(layers):
            h = h @ W.T + b
            if k < len(layers) - 1:
                h = np.tanh(h)
        return h
    # modified MLP (Wang et al. 2021): encoder-gated hidden chain -- mirror of
    # run_experiment.ModifiedMLP.forward
    (Wu, bu), (Wv, bv), layers = params
    U = np.tanh(h @ Wu.T + bu)
    V = np.tanh(h @ Wv.T + bv)
    H = np.tanh(h @ layers[0][0].T + layers[0][1])
    for (W, b) in layers[1:-1]:
        Z = np.tanh(H @ W.T + b)
        H = (1 - Z) * U + Z * V
    return H @ layers[-1][0].T + layers[-1][1]


def _lerp_pairs(a, b, lam):
    return [(Wa * (1 - lam) + Wb * lam, ba * (1 - lam) + bb * lam)
            for (Wa, ba), (Wb, bb) in zip(a, b)]


def lerp_nets(a, b, lam):
    """Interpolate two load_state results of the same architecture."""
    arch, pa = a
    _, pb = b
    if arch == "fnn":
        return (arch, _lerp_pairs(pa, pb, lam))
    (ua, va, la), (ub, vb, lb) = pa, pb
    return (arch, (_lerp_pairs([ua], [ub], lam)[0], _lerp_pairs([va], [vb], lam)[0],
                   _lerp_pairs(la, lb, lam)))


def rel_l2(pred, ref):
    d = np.sqrt(np.mean(ref ** 2))
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / d) if d > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Per-run analysis
# --------------------------------------------------------------------------- #
def analyze_run(run_dir, n_ref=4000, n_lambda=5, rng=None):
    """Returns dict with per-checkpoint arrays + scalar metrics, or None if incomplete."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    fields_p = os.path.join(run_dir, "solution", "fields.npz")
    tl_p = os.path.join(run_dir, "landscape", "trajectory_losses.npz")
    if not (os.path.isdir(ckpt_dir) and os.path.exists(fields_p)):
        return None

    f = np.load(fields_p)
    coords, ref = f["coords"].astype(np.float64), f["ref"].astype(np.float64)
    rng = rng or np.random.default_rng(0)
    sub = rng.choice(coords.shape[0], size=min(n_ref, coords.shape[0]), replace=False)
    X, Y = coords[sub], ref[sub]
    ref_rms = np.sqrt((Y ** 2).mean())

    # run config: pde name + Fourier-embedding modes (checkpoints expect embedded inputs)
    cfg_p = os.path.join(run_dir, "config.json")
    cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
    pde_name = cfg.get("pde", os.path.basename(os.path.dirname(run_dir)))
    fmodes = int(cfg.get("fourier_modes", 0) or 0)

    ckpts = sorted(p for p in os.listdir(ckpt_dir) if p.endswith(".pt"))
    nets = [load_state(os.path.join(ckpt_dir, p)) for p in ckpts]

    # per-checkpoint error + prediction rms (trivial attraction)
    errs, pred_rms = [], []
    for net in nets:
        P = forward(net, X, pde_name, fmodes).reshape(Y.shape)
        errs.append(rel_l2(P, Y))
        pred_rms.append(float(np.sqrt((P ** 2).mean()) / max(ref_rms, EPS)))
    errs, pred_rms = np.array(errs), np.array(pred_rms)

    losses = None
    if os.path.exists(tl_p):
        losses = np.asarray(np.load(tl_p)["loss_total"]).reshape(-1)[: len(errs)]

    # true-error section along the training path (full parameter space, no AE)
    lam_grid = np.linspace(0, 1, n_lambda + 1)[1:-1] if n_lambda > 1 else []
    path_x, path_err = [0.0], [errs[0]]
    for i in range(len(nets) - 1):
        for lam in lam_grid:
            P = forward(lerp_nets(nets[i], nets[i + 1], lam), X, pde_name, fmodes).reshape(Y.shape)
            path_x.append(i + lam)
            path_err.append(rel_l2(P, Y))
        path_x.append(float(i + 1))
        path_err.append(errs[i + 1])

    # deceptive-area fraction of the 2D loss grid (oper cheap, bnd wrong)
    dec_frac = corr_ob = float("nan")
    gall = os.path.join(run_dir, "landscape", "grid_losses_all.npz")
    if os.path.exists(gall):
        g = np.load(gall)
        lo = np.log10(np.clip(g["loss_oper"], 1e-30, None)).ravel()
        lb = np.log10(np.clip(g["loss_bnd"], 1e-30, None)).ravel()
        dec_frac = float(np.mean((lo <= np.quantile(lo, 0.10)) & (lb >= np.median(lb))))
        corr_ob = float(np.corrcoef(lo, lb)[0, 1])

    half = max(1, len(errs) // 2)
    out = {
        "errs": errs, "pred_rms": pred_rms, "losses": losses,
        "path_x": np.array(path_x), "path_err": np.array(path_err),
        "err_first": float(errs[0]), "err_last": float(errs[-1]),
        "err_slope_late": float(errs[-1] - errs[-half]),
        "trivial_ratio_last": float(pred_rms[-1]),
        "deceptive_area_frac": dec_frac, "grid_corr_oper_bnd": corr_ob,
    }
    if losses is not None and len(losses) >= 3 and np.std(np.log10(np.clip(losses, EPS, None))) > EPS:
        out["loss_err_corr"] = float(np.corrcoef(
            np.log10(np.clip(losses, EPS, None)), np.log10(np.clip(errs, EPS, None)))[0, 1])
    else:
        out["loss_err_corr"] = float("nan")
    return out


def collect(runs_root):
    """{(pde, method, seed): analysis} over seed_*/pde/method run dirs (flat tree = seed None)."""
    seeds = [(int(n[5:]), os.path.join(runs_root, n)) for n in sorted(os.listdir(runs_root))
             if n.startswith("seed_") and os.path.isdir(os.path.join(runs_root, n))]
    if not seeds:
        seeds = [(None, runs_root)]
    out = {}
    for seed, root in seeds:
        for pde in sorted(os.listdir(root)):
            pde_dir = os.path.join(root, pde)
            if not os.path.isdir(pde_dir):
                continue
            for method in sorted(os.listdir(pde_dir)):
                if not os.path.isdir(os.path.join(pde_dir, method, "checkpoints")):
                    continue  # not a gradient run (frozen has no checkpoints)
                r = analyze_run(os.path.join(pde_dir, method))
                if r is not None:
                    out[(pde, method, seed)] = r
                    print(f"[run] {pde}/{method}@{seed}: err {r['err_first']:.3f}->{r['err_last']:.3f} "
                          f"corr={r['loss_err_corr']:.2f} trivial={r['trivial_ratio_last']:.2f} "
                          f"dec_area={r['deceptive_area_frac']:.3f}")
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_loss_vs_error(rows, pde, path, seed):
    sel = {m: r for (p, m, s), r in rows.items() if p == pde and s == seed and r["losses"] is not None}
    if not sel:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for m, r in sorted(sel.items()):
        c = method_color(m)
        x = np.arange(len(r["errs"]))
        ax1.plot(x[: len(r["losses"])], r["losses"], "-o", color=c, ms=3, label=m)
        ax2.plot(x, r["errs"], "-o", color=c, ms=3, label=m)
    ax1.set_yscale("log"); ax1.set_title(f"{pde}: training loss along checkpoints (seed {seed})")
    ax1.set_xlabel("checkpoint"); ax1.set_ylabel("loss_total"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_title("TRUE relative-L2 at the same checkpoints")
    ax2.set_xlabel("checkpoint"); ax2.set_ylabel("relative-L2"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"[fig] {path}")


def fig_error_section(rows, pde, path, seed):
    sel = {m: r for (p, m, s), r in rows.items() if p == pde and s == seed}
    if not sel:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, r in sorted(sel.items()):
        ax.plot(r["path_x"] / r["path_x"].max(), r["path_err"], "-", color=method_color(m), label=m)
        ck = np.isin(r["path_x"], np.arange(len(r["errs"])))
        ax.plot(r["path_x"][ck] / r["path_x"].max(), r["path_err"][ck], "o", color=method_color(m), ms=4)
    ax.set_xlabel("training progress (path through full parameter space, checkpoints = dots)")
    ax.set_ylabel("TRUE relative-L2")
    ax.set_title(f"{pde}: error landscape section along each method's training path (seed {seed})")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"[fig] {path}")


def fig_trivial(rows, pde, path, seed):
    sel = {m: r for (p, m, s), r in rows.items() if p == pde and s == seed}
    if not sel:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, r in sorted(sel.items()):
        ax.plot(np.arange(len(r["pred_rms"])), r["pred_rms"], "-o", ms=3, color=method_color(m), label=m)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="reference amplitude")
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("checkpoint"); ax.set_ylabel("rms(prediction) / rms(reference)")
    ax.set_title(f"{pde}: attraction to the trivial solution (ratio -> 0 = collapse), seed {seed}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"[fig] {path}")


def fig_trajmap(rows, runs_root, pde, method, path, seed):
    key = (pde, method, seed)
    if key not in rows:
        return
    seed_dir = os.path.join(runs_root, f"seed_{seed}") if seed is not None else runs_root
    land = os.path.join(seed_dir, pde, method, "landscape")
    g2, t2 = os.path.join(land, "grid_2d.npz"), os.path.join(land, "trajectory_2d.npy")
    if not (os.path.exists(g2) and os.path.exists(t2)):
        return
    g = np.load(g2)
    traj = np.load(t2)
    errs = rows[key]["errs"][: len(traj)]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    L = np.log10(np.clip(g["grid_losses"], 1e-30, None))
    cs = ax.contourf(g["grid_xx"], g["grid_yy"], L, levels=25, cmap="Greys")
    fig.colorbar(cs, ax=ax, label="log10 loss_total (landscape)")
    ax.plot(traj[:, 0], traj[:, 1], "-", color="tab:red", lw=1, alpha=0.7)
    sc = ax.scatter(traj[:, 0], traj[:, 1], c=errs, cmap="RdYlGn_r", s=70,
                    edgecolors="k", linewidths=0.6, zorder=5,
                    vmin=float(np.nanmin(errs)), vmax=float(np.nanmax(errs)))
    fig.colorbar(sc, ax=ax, label="TRUE relative-L2 at checkpoint")
    ax.annotate("start", traj[0], fontsize=8), ax.annotate("end", traj[-1], fontsize=8)
    ax.set_title(f"{pde}/{method} (seed {seed}): loss landscape vs TRUE error along trajectory")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"[fig] {path}")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=str, default="runs_landscape_compare")
    parser.add_argument("--seed-for-figures", type=int, default=1234)
    args = parser.parse_args()

    rows = collect(args.runs)
    if not rows:
        print("No analyzable gradient runs found (need checkpoints/ + solution/fields.npz).")
        return
    out_dir = os.path.join(args.runs, "error_landscape")
    os.makedirs(out_dir, exist_ok=True)

    cols = ["pde", "method", "seed", "err_first", "err_last", "err_slope_late", "loss_err_corr",
            "trivial_ratio_last", "deceptive_area_frac", "grid_corr_oper_bnd"]
    with open(os.path.join(out_dir, "error_landscape_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (pde, method, seed), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
            w.writerow({"pde": pde, "method": method, "seed": seed,
                        **{k: r[k] for k in cols[3:]}})
    print(f"[csv] {os.path.join(out_dir, 'error_landscape_summary.csv')}")

    pdes = sorted({p for p, _, _ in rows})
    seed = args.seed_for_figures if any(s == args.seed_for_figures for _, _, s in rows) else sorted(
        {s for _, _, s in rows}, key=lambda x: (x is None, x))[0]
    for pde in pdes:
        fig_loss_vs_error(rows, pde, os.path.join(out_dir, f"{pde}_loss_vs_error.pdf"), seed)
        fig_error_section(rows, pde, os.path.join(out_dir, f"{pde}_error_section.pdf"), seed)
        fig_trivial(rows, pde, os.path.join(out_dir, f"{pde}_trivial_attraction.pdf"), seed)
        for method in sorted({m for p, m, _ in rows if p == pde}):
            fig_trajmap(rows, args.runs, pde, method,
                        os.path.join(out_dir, f"{pde}_{method}_trajmap.pdf"), seed)

    # mean table across seeds
    print("\n================ ERROR-LANDSCAPE SUMMARY (mean across seeds) ================")
    print(f"{'cell':<32}{'err first->last':<20}{'late slope':<12}{'loss~err':<10}"
          f"{'trivial':<9}{'dec.area':<10}{'oper~bnd':<9}")
    groups = {}
    for (pde, method, _), r in rows.items():
        groups.setdefault((pde, method), []).append(r)
    for (pde, method), rs in sorted(groups.items()):
        def mean(k):
            vals = [x[k] for x in rs if np.isfinite(x[k])]
            return np.mean(vals) if vals else float("nan")
        print(f"{pde + '/' + method:<32}"
              f"{mean('err_first'):.3f} -> {mean('err_last'):.3f}      "
              f"{mean('err_slope_late'):+.3f}     "
              f"{mean('loss_err_corr'):+.2f}     "
              f"{mean('trivial_ratio_last'):.2f}     "
              f"{mean('deceptive_area_frac'):.3f}     "
              f"{mean('grid_corr_oper_bnd'):+.2f}")
    print(f"\nOutputs under: {out_dir}")


if __name__ == "__main__":
    main()
