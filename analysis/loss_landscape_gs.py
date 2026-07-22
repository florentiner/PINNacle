"""Gray-Scott loss landscape + training trajectories (Li et al. trajectory-PCA),
mirroring analysis/loss_landscape.py for KS. Both panels: loss surface in the
top-2 PCA plane of that model's own checkpoints, real trajectory overlaid.

Left : vanilla DeepXDE GS baseline (11 checkpoints, single-shot full domain).
Right: causal GS at a chosen window (param snapshots from the JAX runner).
Loss = unweighted mean-sq PDE residual + mean-sq IC error on fixed points.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
os.environ.setdefault("DDEBACKEND", "pytorch")
from causalpinn.cases import get_case
from causalpinn.train import CausalConfig
from causalpinn.jax_bridge import jax_npz_to_state_dict
from analysis.loss_landscape import flat, unflat, surface

B, D, EPS1, EPS2 = 0.04, 0.1, 1e-5, 5e-6


class GSBaselineFNN(nn.Module):
    def __init__(self, sizes=(3, 100, 100, 100, 100, 100, 2)):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)])

    def forward(self, xyt):
        h = xyt
        for lin in self.linears[:-1]:
            h = torch.tanh(lin(h))
        return self.linears[-1](h)


def gs_baseline_residual(net, xyt):
    xyt = xyt.clone().requires_grad_(True)
    out = net(xyt)
    u, v = out[:, 0:1], out[:, 1:2]
    ones = torch.ones_like(u)

    def d(f, i):
        return torch.autograd.grad(f, xyt, ones, create_graph=True)[0][:, i:i + 1]

    def d2(f, i):
        g = torch.autograd.grad(f, xyt, ones, create_graph=True)[0][:, i:i + 1]
        return torch.autograd.grad(g, xyt, torch.ones_like(g), create_graph=True)[0][:, i:i + 1]

    u_t, v_t = d(u, 2), d(v, 2)
    u_xx, u_yy = d2(u, 0), d2(u, 1)
    v_xx, v_yy = d2(v, 0), d2(v, 1)
    r_u = u_t - (EPS1 * (u_xx + u_yy) + B * (1 - u) - u * v ** 2)
    r_v = v_t - (EPS2 * (v_xx + v_yy) - D * v + u * v ** 2)
    return torch.cat([r_u, r_v], dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="runs/07.18-13.19.39-baseline-chaotic/1-0")
    p.add_argument("--causal", default="runs/kaggle-causal-gs-session6/causal_gs_jax/0-0")
    p.add_argument("--window", type=int, default=5, help="causal window to profile")
    p.add_argument("--out", default="analysis/out/gs-losslandscape")
    p.add_argument("--grid", type=int, default=25)
    p.add_argument("--npts", type=int, default=2000)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    # ---------------- Panel A: baseline GS with trajectory ----------------
    col = np.load(os.path.join(args.baseline, "collocation.npz"))
    rng = np.random.default_rng(0)
    pts = col["train_x_all"]
    sub = pts[rng.choice(len(pts), min(args.npts, len(pts)), replace=False)]
    xyt_r = torch.tensor(sub, dtype=torch.float32)
    ic = col["train_x_bc"]
    ic = ic[np.isclose(ic[:, 2], 0.0)][:1500]
    xyt_ic = torch.tensor(ic, dtype=torch.float32)
    # IC targets from GS formulas (src/pde/chaotic.py)
    xi, yi = ic[:, 0], ic[:, 1]
    u_ic = torch.tensor((1 - np.exp(-80 * ((xi + 0.05) ** 2 + (yi + 0.02) ** 2)))[:, None],
                        dtype=torch.float32)
    v_ic = torch.tensor(np.exp(-80 * ((xi - 0.05) ** 2 + (yi - 0.02) ** 2))[:, None],
                        dtype=torch.float32)

    bnet = GSBaselineFNN()

    def base_loss(sd):
        bnet.load_state_dict(sd)
        with torch.enable_grad():
            r = gs_baseline_residual(bnet, xyt_r)
        out_ic = bnet(xyt_ic)
        ic_err = (out_ic[:, 0:1] - u_ic) ** 2 + (out_ic[:, 1:2] - v_ic) ** 2
        return float((r ** 2).mean() + ic_err.mean())

    cks = sorted(glob.glob(os.path.join(args.baseline, "trajectory", "ckpt_*.pt")),
                 key=lambda s: int(re.search(r"ckpt_(\d+)\.pt", s).group(1)))
    sds = [torch.load(c, map_location="cpu", weights_only=False)["model_state_dict"]
           for c in cks]
    steps = [int(re.search(r"ckpt_(\d+)\.pt", c).group(1)) for c in cks]
    theta_f = flat(sds[-1])
    diffs = torch.stack([flat(sd) - theta_f for sd in sds[:-1]])
    _, _, V = torch.pca_lowrank(diffs, q=min(4, len(diffs)))
    d1 = V[:, 0] / V[:, 0].norm()
    d2 = V[:, 1] - (V[:, 1] @ d1) * d1
    d2 = d2 / d2.norm()
    a_traj = np.append((diffs @ d1).numpy(), 0.0)
    b_traj = np.append((diffs @ d2).numpy(), 0.0)
    pa = 0.25 * (a_traj.max() - a_traj.min() + 1e-9)
    pb = 0.25 * (b_traj.max() - b_traj.min() + 1e-9)
    print("[A] GS baseline surface ...", flush=True)
    A, Bs, Z = surface(base_loss, sds[-1], d1, d2,
                       (a_traj.min() - pa, a_traj.max() + pa),
                       (b_traj.min() - pb, b_traj.max() + pb), n=args.grid)

    # ---------------- Panel B: causal GS window with trajectory ----------------
    cfg = CausalConfig(case="gs", device="cpu", windows=20)
    case = get_case("gs", cfg)
    k = args.window
    snaps = sorted(glob.glob(os.path.join(args.causal, "trajectory",
                                          f"w{k}_snap_*.npz")),
                   key=lambda s: int(re.search(r"snap_(\d+)", s).group(1)))
    cnet = case.build_net("plain", cfg.seed, torch.device("cpu"))
    csnap_sds, csnap_steps = [], []
    for sf in snaps:
        cnet.load_state_dict(jax_npz_to_state_dict(np.load(sf)), strict=False)
        csnap_sds.append({kk: v.clone() for kk, v in cnet.state_dict().items()})
        csnap_steps.append(int(re.search(r"snap_(\d+)", sf).group(1)))
    fin = os.path.join(args.causal, "trajectory", f"w{k}_final_params.npz")
    cnet.load_state_dict(jax_npz_to_state_dict(np.load(fin)), strict=False)
    csd = {kk: v.clone() for kk, v in cnet.state_dict().items()}

    # window collocation: normalized local tau in [0,1], xy in [-1,1]^2; IC = ref at window start
    cols = case.window_ref_cols(k)
    ic_coords, _ = case.ic_arrays()
    ic_vals = case.ref[:, :, cols[0], :].reshape(-1, 2)
    ic_c = torch.tensor(ic_coords, dtype=torch.float32)
    ic_v = torch.tensor(ic_vals, dtype=torch.float32)
    t0 = torch.zeros(len(ic_coords), 1)
    tw = torch.tensor(rng.uniform(0, 1.01, size=(args.npts // 2, 1)), dtype=torch.float32)
    xw = torch.tensor(rng.uniform(-1, 1, size=(args.npts // 2, 2)), dtype=torch.float32)

    def causal_loss(sd):
        cnet.load_state_dict(sd)
        with torch.enable_grad():
            r = case.residual(cnet, tw, xw, case.T_w)
        pred_ic = cnet(t0, ic_c[:, 0:1], ic_c[:, 1:2])
        return float((r ** 2).mean() + ((pred_ic - ic_v) ** 2).mean())

    ctf = flat(csd)
    cdiffs = torch.stack([flat(sd) - ctf for sd in csnap_sds])
    _, _, CV = torch.pca_lowrank(cdiffs, q=min(4, len(cdiffs)))
    cd1 = CV[:, 0] / CV[:, 0].norm()
    cd2 = CV[:, 1] - (CV[:, 1] @ cd1) * cd1
    cd2 = cd2 / cd2.norm()
    ca = np.append((cdiffs @ cd1).numpy(), 0.0)
    cb = np.append((cdiffs @ cd2).numpy(), 0.0)
    pa2 = 0.25 * (ca.max() - ca.min() + 1e-9)
    pb2 = 0.25 * (cb.max() - cb.min() + 1e-9)
    print("[B] causal GS window surface ...", flush=True)
    CA, CB, CZ = surface(causal_loss, csd, cd1, cd2,
                         (ca.min() - pa2, ca.max() + pa2),
                         (cb.min() - pb2, cb.max() + pb2), n=args.grid)

    # ---------------- figure ----------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    cs = axs[0].contourf(A, Bs, np.log10(Z), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=axs[0], label="log10 loss")
    axs[0].plot(a_traj, b_traj, "w.-", lw=1.5, ms=6)
    for i in [0, len(a_traj) // 2, len(a_traj) - 1]:
        axs[0].annotate(f"{steps[i] if i < len(steps) else steps[-1]}",
                        (a_traj[i], b_traj[i]), color="w", fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    axs[0].set_title("vanilla GS PINN: loss landscape + trajectory\n"
                     "(top-2 PCA plane of checkpoints; labels = step)")
    axs[0].set_xlabel("PCA dir 1"); axs[0].set_ylabel("PCA dir 2")

    cs2 = axs[1].contourf(CA, CB, np.log10(CZ), levels=30, cmap="viridis")
    fig.colorbar(cs2, ax=axs[1], label="log10 loss")
    axs[1].plot(ca, cb, "w.-", lw=1.5, ms=6)
    for i in [0, len(ca) // 2, len(ca) - 1]:
        lbl = csnap_steps[i] if i < len(csnap_steps) else "final"
        axs[1].annotate(f"{lbl}", (ca[i], cb[i]), color="w", fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    axs[1].plot([0], [0], "r*", ms=14, label=f"trained w{k} solution")
    axs[1].legend(loc="upper right")
    axs[1].set_title(f"causal GS PINN (window {k}): loss landscape + trajectory\n"
                     "(top-2 PCA plane of snapshots; labels = window iter)")
    axs[1].set_xlabel("PCA dir 1"); axs[1].set_ylabel("PCA dir 2")
    fig.tight_layout()
    out = os.path.join(args.out, "loss_landscape_trajectory.png")
    fig.savefig(out, dpi=140)
    np.savez_compressed(os.path.join(args.out, "loss_landscape_data.npz"),
                        A=A, B=Bs, Z=Z, a_traj=a_traj, b_traj=b_traj,
                        steps=np.array(steps), CA=CA, CB=CB, CZ=CZ,
                        ca_traj=ca, cb_traj=cb, csnap_steps=np.array(csnap_steps),
                        window=k)
    print("saved", out)


if __name__ == "__main__":
    main()
