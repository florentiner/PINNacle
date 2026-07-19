"""Optimization loss landscape with the training trajectory overlaid (Li et al.
2018, trajectory-PCA variant) — computed from saved checkpoints only.

Panel A: vanilla baseline — loss surface in the top-2 PCA plane of its own
checkpoint trajectory, with the projected trajectory drawn on it.
Panel B: causal window-0 solution — surface in filter-normalized random
directions around the trained minimum (JAX engine stores window finals, not
intra-window snapshots, so no path is drawn — basin shape only).

Loss in both panels = unweighted mean-sq PDE residual + mean-sq IC error on
fixed saved collocation points (the training objective without weights).
"""
import argparse
import glob
import json
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

ALPHA, BETA, GAMMA = 100 / 16, 100 / 256, 100 / 16 ** 4


class FNNReplica(nn.Module):
    """Matches vendored dde.nn.FNN state_dict (linears.{i}.weight/bias, tanh)."""

    def __init__(self, sizes=(2, 100, 100, 100, 100, 100, 1)):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)])

    def forward(self, t, x):          # causal-adapter signature; dde order is (x,t)
        h = torch.cat([x, t], dim=1)
        for lin in self.linears[:-1]:
            h = torch.tanh(lin(h))
        return self.linears[-1](h)


def ks_residual(net, t, x):
    from torch.func import jvp
    v = torch.ones_like(x)

    def u_of_x(x_):
        return net(t, x_)

    def d1(x_):
        return jvp(u_of_x, (x_,), (v,))[1]

    def d2(x_):
        return jvp(d1, (x_,), (v,))[1]

    def d3(x_):
        return jvp(d2, (x_,), (v,))[1]

    u, u_x = jvp(u_of_x, (x,), (v,))
    u_xx = jvp(d1, (x,), (v,))[1]
    u_xxxx = jvp(d3, (x,), (v,))[1]
    u_t = jvp(lambda t_: net(t_, x), (t,), (torch.ones_like(t),))[1]
    return u_t + ALPHA * u * u_x + BETA * u_xx + GAMMA * u_xxxx


def flat(sd):
    return torch.cat([v.reshape(-1) for v in sd.values()])


def unflat(vec, ref_sd):
    out, i = {}, 0
    for k, v in ref_sd.items():
        n = v.numel()
        out[k] = vec[i:i + n].reshape(v.shape)
        i += n
    return out


def surface(loss_fn, theta0_sd, d1, d2, arange, brange, n=31):
    ref = theta0_sd
    t0 = flat(ref)
    A = np.linspace(*arange, n)
    B = np.linspace(*brange, n)
    Z = np.zeros((n, n))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            vec = t0 + float(a) * d1 + float(b) * d2
            Z[j, i] = loss_fn(unflat(vec, ref))
    return A, B, Z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="runs/07.18-13.19.39-baseline-chaotic/0-0")
    p.add_argument("--causal", default="runs/kaggle-causal-ks-session1/causal_ks/0-0")
    p.add_argument("--out", default="analysis/out/ks-losslandscape")
    p.add_argument("--grid", type=int, default=31)
    p.add_argument("--npts", type=int, default=3000)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    # ---------------- Panel A: baseline with trajectory ----------------
    col = np.load(os.path.join(args.baseline, "collocation.npz"))
    pts = col["train_x_all"]
    rng = np.random.default_rng(0)
    sub = pts[rng.choice(len(pts), min(args.npts, len(pts)), replace=False)]
    x_r = torch.tensor(sub[:, 0:1], dtype=torch.float32)
    t_r = torch.tensor(sub[:, 1:2], dtype=torch.float32)
    x_ic = torch.tensor(col["train_x_bc"][:, 0:1], dtype=torch.float32)
    t_ic = torch.zeros_like(x_ic)
    u_ic = torch.cos(x_ic) * (1 + torch.sin(x_ic))

    net = FNNReplica()

    def base_loss(sd):
        net.load_state_dict(sd)
        with torch.enable_grad():
            r = ks_residual(net, t_r, x_r)
        ic = net(t_ic, x_ic) - u_ic
        return float((r ** 2).mean() + (ic ** 2).mean())

    cks = sorted(glob.glob(os.path.join(args.baseline, "trajectory", "ckpt_*.pt")),
                 key=lambda s: int(re.search(r"ckpt_(\d+)\.pt", s).group(1)))
    sds = [torch.load(c, map_location="cpu", weights_only=False)["model_state_dict"]
           for c in cks]
    steps = [int(re.search(r"ckpt_(\d+)\.pt", c).group(1)) for c in cks]
    theta_f = flat(sds[-1])
    diffs = torch.stack([flat(sd) - theta_f for sd in sds[:-1]])
    U, S, V = torch.pca_lowrank(diffs, q=min(4, len(diffs)))
    d1 = V[:, 0] / V[:, 0].norm()
    d2 = V[:, 1] - (V[:, 1] @ d1) * d1
    d2 = d2 / d2.norm()
    a_traj = (diffs @ d1).numpy()
    b_traj = (diffs @ d2).numpy()
    a_traj = np.append(a_traj, 0.0)
    b_traj = np.append(b_traj, 0.0)
    pad_a = 0.25 * (a_traj.max() - a_traj.min() + 1e-9)
    pad_b = 0.25 * (b_traj.max() - b_traj.min() + 1e-9)
    arange = (a_traj.min() - pad_a, a_traj.max() + pad_a)
    brange = (b_traj.min() - pad_b, b_traj.max() + pad_b)
    print("[A] computing baseline surface ...", flush=True)
    A, B, Z = surface(base_loss, sds[-1], d1, d2, arange, brange, n=args.grid)

    # ---------------- Panel B: causal w0 ----------------
    from causalpinn.cases import get_case
    from causalpinn.train import CausalConfig
    from causalpinn.jax_bridge import jax_npz_to_state_dict
    cfg = CausalConfig(case="ks", device="cpu", windows=10)
    case = get_case("ks", cfg)
    cnet = case.build_net("fourier", cfg.seed, torch.device("cpu"))

    # trajectory snapshots (w0_snap_*.npz from --param-snap-every) if available
    snap_files = sorted(glob.glob(os.path.join(args.causal, "trajectory",
                                               "w0_snap_*.npz")),
                        key=lambda s: int(re.search(r"snap_(\d+)\.npz", s).group(1)))
    csnap_sds = []
    csnap_steps = []
    for sf in snap_files:
        sd = jax_npz_to_state_dict(np.load(sf))
        cnet.load_state_dict(sd, strict=False)
        csnap_sds.append({k: v.clone() for k, v in cnet.state_dict().items()})
        csnap_steps.append(int(re.search(r"snap_(\d+)\.npz", sf).group(1)))

    w0_pt = os.path.join(args.causal, "trajectory", "w0_final.pt")
    if os.path.exists(w0_pt):
        w0 = torch.load(w0_pt, map_location="cpu", weights_only=False)["model_state_dict"]
        cnet.load_state_dict(w0, strict=False)
    elif os.path.exists(os.path.join(args.causal, "trajectory", "w0_final_params.npz")):
        cnet.load_state_dict(jax_npz_to_state_dict(
            np.load(os.path.join(args.causal, "trajectory", "w0_final_params.npz"))),
            strict=False)
    csd = {k: v.clone() for k, v in cnet.state_dict().items()}

    xs = torch.tensor(case.x_star[:, None], dtype=torch.float32)
    u0 = torch.tensor(case.ref[:, 0, :], dtype=torch.float32)
    tw = torch.tensor(rng.uniform(0, 0.101, size=(args.npts // 4, 1)),
                      dtype=torch.float32)
    xw = torch.tensor(rng.uniform(0, 2 * np.pi, size=(args.npts // 4, 1)),
                      dtype=torch.float32)
    t0_ic = torch.zeros_like(xs)

    def causal_loss(sd):
        cnet.load_state_dict(sd)
        with torch.enable_grad():
            r = case.residual(cnet, tw, xw, case.T_w)
        ic = cnet(t0_ic, xs) - u0
        return float((r ** 2).mean() + (ic ** 2).mean())

    ca_traj = cb_traj = None
    if csnap_sds:
        # trajectory-PCA plane, same convention as the vanilla panel
        ctheta_f = flat(csd)
        cdiffs = torch.stack([flat(sd) - ctheta_f for sd in csnap_sds])
        _, _, CV = torch.pca_lowrank(cdiffs, q=min(4, len(cdiffs)))
        cd1 = CV[:, 0] / CV[:, 0].norm()
        cd2 = CV[:, 1] - (CV[:, 1] @ cd1) * cd1
        cd2 = cd2 / cd2.norm()
        ca_traj = np.append((cdiffs @ cd1).numpy(), 0.0)
        cb_traj = np.append((cdiffs @ cd2).numpy(), 0.0)
        pad_a = 0.25 * (ca_traj.max() - ca_traj.min() + 1e-9)
        pad_b = 0.25 * (cb_traj.max() - cb_traj.min() + 1e-9)
        crange_a = (ca_traj.min() - pad_a, ca_traj.max() + pad_a)
        crange_b = (cb_traj.min() - pad_b, cb_traj.max() + pad_b)
    else:
        # no snapshots: filter-normalized random directions (Li et al.)
        gen = torch.Generator().manual_seed(7)

        def filt_dir():
            parts = []
            for k, v in csd.items():
                r = torch.randn(v.shape, generator=gen)
                if v.dim() >= 2:
                    r = r * (v.norm(dim=tuple(range(1, v.dim())), keepdim=True)
                             / (r.norm(dim=tuple(range(1, v.dim())), keepdim=True) + 1e-12))
                else:
                    r = r * (v.norm() / (r.norm() + 1e-12))
                parts.append(r.reshape(-1))
            return torch.cat(parts)

        cd1, cd2 = filt_dir(), filt_dir()
        cd2 = cd2 - (cd2 @ cd1) / (cd1 @ cd1) * cd1
        crange_a = crange_b = (-0.4, 0.4)
    print("[B] computing causal w0 surface ...", flush=True)
    CA, CB, CZ = surface(causal_loss, csd, cd1, cd2, crange_a, crange_b,
                         n=args.grid)

    # ---------------- figure ----------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    cs = axs[0].contourf(A, B, np.log10(Z), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=axs[0], label="log10 loss")
    axs[0].plot(a_traj, b_traj, "w.-", lw=1.5, ms=6)
    for i in [0, len(a_traj) // 2, len(a_traj) - 1]:
        axs[0].annotate(f"{steps[i] if i < len(steps) else steps[-1]}",
                        (a_traj[i], b_traj[i]), color="w", fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    axs[0].set_title("vanilla PINN: loss landscape + training trajectory\n"
                     "(top-2 PCA plane of its checkpoints; labels = step)")
    axs[0].set_xlabel("PCA dir 1")
    axs[0].set_ylabel("PCA dir 2")

    cs2 = axs[1].contourf(CA, CB, np.log10(CZ), levels=30, cmap="viridis")
    fig.colorbar(cs2, ax=axs[1], label="log10 loss")
    if ca_traj is not None:
        axs[1].plot(ca_traj, cb_traj, "w.-", lw=1.5, ms=6)
        for i in [0, len(ca_traj) // 2, len(ca_traj) - 1]:
            lbl = csnap_steps[i] if i < len(csnap_steps) else "final"
            axs[1].annotate(f"{lbl}", (ca_traj[i], cb_traj[i]), color="w",
                            fontsize=8, xytext=(4, 4), textcoords="offset points")
        axs[1].plot([0], [0], "r*", ms=14, label="trained w0 solution")
        axs[1].legend(loc="upper right")
        axs[1].set_title("causal PINN (window 0): loss landscape + training "
                         "trajectory\n(top-2 PCA plane of its snapshots; "
                         "labels = window iter)")
        axs[1].set_xlabel("PCA dir 1")
        axs[1].set_ylabel("PCA dir 2")
    else:
        axs[1].plot([0], [0], "r*", ms=14, label="trained w0 solution")
        axs[1].legend(loc="upper right")
        axs[1].set_title("causal PINN (window 0): basin around solution\n"
                         "(filter-normalized random plane; no stored path)")
        axs[1].set_xlabel("dir 1 (filter-normalized)")
        axs[1].set_ylabel("dir 2")
    fig.tight_layout()
    out = os.path.join(args.out, "loss_landscape_trajectory.png")
    fig.savefig(out, dpi=140)
    extra = {}
    if ca_traj is not None:
        extra = {"ca_traj": ca_traj, "cb_traj": cb_traj,
                 "csnap_steps": np.array(csnap_steps)}
    np.savez_compressed(os.path.join(args.out, "loss_landscape_data.npz"),
                        A=A, B=B, Z=Z, a_traj=a_traj, b_traj=b_traj,
                        steps=np.array(steps), CA=CA, CB=CB, CZ=CZ, **extra)
    print("saved", out)


if __name__ == "__main__":
    main()
