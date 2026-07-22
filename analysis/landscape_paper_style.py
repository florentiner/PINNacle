"""Render loss landscapes in the Li et al. (2018) paper style to match the
user's existing figures (runs_landscape_compare/.../landscape/map_loss_*.pdf):
  - line contours (not filled) with inline value labels
  - log-scale "loss value" colorbar
  - black optimizer trajectory overlaid, axes normalized to ~[-1,1]
  - weighted TOTAL loss (mean residual^2 + w_ic * mean IC^2), wide dynamic range

Reuses the model replicas / residuals from analysis/loss_landscape.py.

Usage:
  python analysis/landscape_paper_style.py --case ks --which causal \
      --traj runs/kaggle-ks-w0-trajectory/w0_traj/0-0 --window 0
  python analysis/landscape_paper_style.py --case ks --which vanilla \
      --traj runs/07.18-13.19.39-baseline-chaotic/0-0
  python analysis/landscape_paper_style.py --case gs --which causal \
      --traj runs/kaggle-causal-gs-session6/causal_gs_jax/0-0 --window 5
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker

sys.path.insert(0, os.getcwd())
os.environ.setdefault("DDEBACKEND", "pytorch")
from analysis.loss_landscape import (FNNReplica, ks_residual, flat, unflat)

W_IC = 1e4


def flat_np_to_sd(vec, ref_sd):
    return unflat(torch.tensor(vec, dtype=torch.float32), ref_sd)


# ---------------- loss builders (weighted total loss) ----------------
def ks_vanilla_lossfn(traj_dir):
    col = np.load(os.path.join(traj_dir, "collocation.npz"))
    rng = np.random.default_rng(0)
    sub = col["train_x_all"][rng.choice(len(col["train_x_all"]), 3000, replace=False)]
    x_r = torch.tensor(sub[:, 0:1], dtype=torch.float32)
    t_r = torch.tensor(sub[:, 1:2], dtype=torch.float32)
    x_ic = torch.tensor(col["train_x_bc"][:, 0:1], dtype=torch.float32)
    u_ic = torch.cos(x_ic) * (1 + torch.sin(x_ic))
    t_ic = torch.zeros_like(x_ic)
    net = FNNReplica()

    def loss(sd):
        net.load_state_dict(sd)
        with torch.enable_grad():
            r = ks_residual(net, t_r, x_r)
        ic = net(t_ic, x_ic) - u_ic
        return float((r ** 2).mean() + W_IC * (ic ** 2).mean())
    # torch-checkpoint trajectory
    cks = sorted(glob.glob(os.path.join(traj_dir, "trajectory", "ckpt_*.pt")),
                 key=lambda s: int(re.search(r"ckpt_(\d+)", s).group(1)))
    sds = [torch.load(c, map_location="cpu", weights_only=False)["model_state_dict"]
           for c in cks]
    return loss, net.state_dict(), sds


def ks_causal_lossfn(traj_dir, window):
    from causalpinn.cases import get_case
    from causalpinn.train import CausalConfig
    from causalpinn.jax_bridge import jax_npz_to_state_dict
    cfg = CausalConfig(case="ks", device="cpu", windows=10)
    case = get_case("ks", cfg)
    net = case.build_net("fourier", cfg.seed, torch.device("cpu"))
    init_sd = {k: v.clone() for k, v in net.state_dict().items()}  # true iter-0
    rng = np.random.default_rng(0)
    xs = torch.tensor(case.x_star[:, None], dtype=torch.float32)
    u0 = torch.tensor(case.ref[:, 0, :], dtype=torch.float32)
    t0 = torch.zeros_like(xs)
    tw = torch.tensor(rng.uniform(0, 0.101, size=(1500, 1)), dtype=torch.float32)
    xw = torch.tensor(rng.uniform(0, 2 * np.pi, size=(1500, 1)), dtype=torch.float32)

    def loss(sd):
        net.load_state_dict(sd)
        with torch.enable_grad():
            r = case.residual(net, tw, xw, case.T_w)
        ic = net(t0, xs) - u0
        return float((r ** 2).mean() + W_IC * (ic ** 2).mean())
    snaps = sorted(glob.glob(os.path.join(traj_dir, "trajectory",
                                          f"w{window}_snap_*.npz")),
                   key=lambda s: int(re.search(r"snap_(\d+)", s).group(1)))
    sds = []   # trajectory-PCA plane from the training snapshots (no far init)
    for sf in snaps:
        net.load_state_dict(jax_npz_to_state_dict(np.load(sf)), strict=False)
        sds.append({k: v.clone() for k, v in net.state_dict().items()})
    fin = os.path.join(traj_dir, "trajectory", f"w{window}_final_params.npz")
    if os.path.exists(fin):
        net.load_state_dict(jax_npz_to_state_dict(np.load(fin)), strict=False)
        sds.append({k: v.clone() for k, v in net.state_dict().items()})
    return loss, {k: v.clone() for k, v in net.state_dict().items()}, sds


def gs_causal_lossfn(traj_dir, window):
    from causalpinn.cases import get_case
    from causalpinn.train import CausalConfig
    from causalpinn.jax_bridge import jax_npz_to_state_dict
    cfg = CausalConfig(case="gs", device="cpu", windows=20)
    case = get_case("gs", cfg)
    net = case.build_net("plain", cfg.seed, torch.device("cpu"))
    rng = np.random.default_rng(0)
    cols = case.window_ref_cols(window)
    ic_coords, _ = case.ic_arrays()
    ic_vals = case.ref[:, :, cols[0], :].reshape(-1, 2)
    ic_c = torch.tensor(ic_coords, dtype=torch.float32)
    ic_v = torch.tensor(ic_vals, dtype=torch.float32)
    t0 = torch.zeros(len(ic_coords), 1)
    tw = torch.tensor(rng.uniform(0, 1.01, size=(1000, 1)), dtype=torch.float32)
    xw = torch.tensor(rng.uniform(-1, 1, size=(1000, 2)), dtype=torch.float32)

    def loss(sd):
        net.load_state_dict(sd)
        with torch.enable_grad():
            r = case.residual(net, tw, xw, case.T_w)
        pic = net(t0, ic_c[:, 0:1], ic_c[:, 1:2])
        return float((r ** 2).mean() + W_IC * ((pic - ic_v) ** 2).mean())
    snaps = sorted(glob.glob(os.path.join(traj_dir, "trajectory",
                                          f"w{window}_snap_*.npz")),
                   key=lambda s: int(re.search(r"snap_(\d+)", s).group(1)))
    sds = []
    for sf in snaps:
        net.load_state_dict(jax_npz_to_state_dict(np.load(sf)), strict=False)
        sds.append({k: v.clone() for k, v in net.state_dict().items()})
    fin = os.path.join(traj_dir, "trajectory", f"w{window}_final_params.npz")
    net.load_state_dict(jax_npz_to_state_dict(np.load(fin)), strict=False)
    sds.append({k: v.clone() for k, v in net.state_dict().items()})
    return loss, {k: v.clone() for k, v in net.state_dict().items()}, sds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["ks", "gs"], required=True)
    p.add_argument("--which", choices=["causal", "vanilla"], default="causal")
    p.add_argument("--traj", required=True)
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--grid", type=int, default=45)
    p.add_argument("--margin", type=float, default=0.25)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    torch.set_grad_enabled(False)

    if args.case == "ks" and args.which == "vanilla":
        loss, ref_sd, sds = ks_vanilla_lossfn(args.traj)
        tag = "ks_vanilla"
    elif args.case == "ks":
        loss, ref_sd, sds = ks_causal_lossfn(args.traj, args.window)
        tag = f"ks_causal_w{args.window}"
    else:
        loss, ref_sd, sds = gs_causal_lossfn(args.traj, args.window)
        tag = f"gs_causal_w{args.window}"
    out = args.out or f"analysis/out/paper_style/{tag}"
    os.makedirs(out, exist_ok=True)

    # 2 trajectory-PCA directions (unit), then normalize axes to trajectory extent
    theta_f = flat(sds[-1])
    diffs = torch.stack([flat(sd) - theta_f for sd in sds[:-1]])
    _, _, V = torch.pca_lowrank(diffs, q=min(4, len(diffs)))
    d1 = V[:, 0] / V[:, 0].norm()
    d2 = V[:, 1] - (V[:, 1] @ d1) * d1
    d2 = d2 / d2.norm()
    a = np.append((diffs @ d1).numpy(), 0.0)
    b = np.append((diffs @ d2).numpy(), 0.0)
    # normalize so trajectory spans ~[-1,1]
    ac, ar = (a.max() + a.min()) / 2, max(a.max() - a.min(), 1e-9) / 2
    bc, br = (b.max() + b.min()) / 2, max(b.max() - b.min(), 1e-9) / 2
    a_n, b_n = (a - ac) / ar, (b - bc) / br

    m = args.margin
    G = np.linspace(-1 - m, 1 + m, args.grid)
    Z = np.zeros((args.grid, args.grid))
    t0 = flat(ref_sd)
    for i, av in enumerate(G):
        for j, bv in enumerate(G):
            vec = t0 + (av * ar + ac) * d1 + (bv * br + bc) * d2
            Z[j, i] = loss(unflat(vec, ref_sd))
    Z = np.clip(Z, 1e0, None)

    # ---- render: Li et al. style ----
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    levels = np.logspace(np.log10(max(Z.min(), 1e0)), np.log10(Z.max()), 30)
    cs = ax.contour(G, G, Z, levels=levels, norm=LogNorm(),
                    cmap="viridis", linewidths=0.8)
    ax.clabel(cs, cs.levels[::3], inline=True, fontsize=6,
              fmt=lambda x: f"{x:.2e}")
    ax.plot(a_n, b_n, "k.-", lw=1.3, ms=5)
    sm = plt.cm.ScalarMappable(norm=LogNorm(vmin=Z.min(), vmax=Z.max()),
                               cmap="viridis")
    fig.colorbar(sm, ax=ax, label="loss value")
    ax.set_xlim(-1 - m, 1 + m)
    ax.set_ylim(-1 - m, 1 + m)
    fig.tight_layout()
    path = os.path.join(out, "map_total_loss.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=150)
    np.savez_compressed(os.path.join(out, "landscape.npz"),
                        G=G, Z=Z, a=a_n, b=b_n)
    print("saved", path, "| loss range", f"{Z.min():.2e}..{Z.max():.2e}")


if __name__ == "__main__":
    main()
