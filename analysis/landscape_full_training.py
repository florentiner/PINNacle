"""Full-training error-space view: vanilla (ONE descent) vs causal (N sequential descents).

The existing §1.3 figures compare vanilla's complete 20k-iteration trajectory against
only window 0 of the causal run. This script removes that asymmetry: it draws the
WHOLE causal run — the chain of all N window solutions in weight space plus the
landscape+trajectory of individual windows spread across the time domain.

Why the causal run cannot have a single landscape: each window trains a FRESH network
(same architecture and seed) against its OWN objective — residual on its time slice +
its handoff initial condition. So "full training in error space" for the causal method
is by construction a SEQUENCE of landscapes, which is exactly what the bottom row shows.

Loss (both methods, same convention as loss_landscape.py): unweighted mean-square PDE
residual + mean-square IC error on fixed points.
Usage: python analysis/landscape_full_training.py --case ks|gs [--grid 15]
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
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.getcwd())
os.environ.setdefault("DDEBACKEND", "pytorch")

OUT = "analysis/report_figs"

CFG = {
    "ks": {
        "windows": [0, 5, 7, 9],
        "n_win": 10,
        "final_dir": "runs/kaggle-causal-ks-session9/causal_ks/0-0",
        "snap_roots": ["runs/kaggle-causal-ks-session*/causal_ks/0-0",
                       "runs/kaggle-ks-w0-trajectory/*/0-0"],
        "vanilla_cache": "analysis/out/ks-losslandscape-both/loss_landscape_data.npz",
        "encoding": "fourier",
        "title": "Kuramoto–Sivashinsky",
    },
    "gs": {
        "windows": [1, 6, 12, 19],
        "n_win": 20,
        "final_dir": "runs/kaggle-causal-gs-session7/causal_gs_jax/0-0",
        "snap_roots": ["runs/kaggle-causal-gs-session*/causal_gs_jax/0-0"],
        "vanilla_cache": "analysis/out/gs-losslandscape/loss_landscape_data.npz",
        "encoding": "plain",
        "title": "Gray–Scott",
    },
}


def flat(sd):
    return torch.cat([v.reshape(-1) for v in sd.values()])


def unflat(vec, ref_sd):
    out, i = {}, 0
    for k, v in ref_sd.items():
        n = v.numel()
        out[k] = vec[i:i + n].reshape(v.shape).clone()
        i += n
    return out


def surface(loss_fn, sd0, d1, d2, arange, brange, n):
    th0 = flat(sd0)
    A = np.linspace(*arange, n)
    B = np.linspace(*brange, n)
    Z = np.zeros((n, n))
    for i, b in enumerate(B):
        for j, a in enumerate(A):
            Z[i, j] = loss_fn(unflat(th0 + a * d1 + b * d2, sd0))
        print(f"    row {i + 1}/{n}", flush=True)
    return A, B, Z


def gather_snaps(cfg, k):
    """All snapshots of window k across sessions, deduped by iteration."""
    found = {}
    for root in cfg["snap_roots"]:
        for p in glob.glob(os.path.join(root, "trajectory", f"w{k}_snap_*.npz")):
            it = int(re.search(r"snap_(\d+)", p).group(1))
            found[it] = p                      # later session wins; same params anyway
    return [found[i] for i in sorted(found)]


def window_ic(case, cfg, k):
    """The IC window k was actually trained against: handoff out of window k-1."""
    if k == 0:
        coords, vals = case.ic_arrays() if hasattr(case, "ic_arrays") else (None, None)
        return coords, vals
    p = os.path.join(cfg["final_dir"], "causal", f"window_{k - 1}_handoff_ic.npy")
    vals = np.load(p)
    coords = case.ic_arrays()[0]
    return coords, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["ks", "gs"], required=True)
    ap.add_argument("--grid", type=int, default=15)
    ap.add_argument("--npts", type=int, default=600)
    args = ap.parse_args()
    cfg = CFG[args.case]
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(0)

    from causalpinn.cases import get_case
    from causalpinn.train import CausalConfig
    from causalpinn.jax_bridge import jax_npz_to_state_dict

    ccfg = CausalConfig(case=args.case, device="cpu", windows=cfg["n_win"])
    case = get_case(args.case, ccfg)
    net = case.build_net(cfg["encoding"], ccfg.seed, torch.device("cpu"))

    # ---- per-window landscapes with their own trajectories ----
    hist = np.load(os.path.join(cfg["final_dir"], "causal", "history_jax.npz"),
                   allow_pickle=True)
    panels = []
    for k in cfg["windows"]:
        snaps = gather_snaps(cfg, k)
        fin = os.path.join(cfg["final_dir"], "trajectory", f"w{k}_final_params.npz")
        print(f"[{args.case} w{k}] {len(snaps)} snapshots", flush=True)
        sds, steps = [], []
        for sf in snaps:
            net.load_state_dict(jax_npz_to_state_dict(np.load(sf)), strict=False)
            sds.append({kk: v.clone() for kk, v in net.state_dict().items()})
            steps.append(int(re.search(r"snap_(\d+)", sf).group(1)))
        net.load_state_dict(jax_npz_to_state_dict(np.load(fin)), strict=False)
        fsd = {kk: v.clone() for kk, v in net.state_dict().items()}

        ic_coords, ic_vals = window_ic(case, cfg, k)
        ic_c = torch.tensor(np.asarray(ic_coords), dtype=torch.float32)
        ic_v = torch.tensor(np.asarray(ic_vals), dtype=torch.float32)
        t0 = torch.zeros(len(ic_c), 1)
        if args.case == "ks":
            tw = torch.tensor(rng.uniform(0, case.T_w * 1.01, size=(args.npts, 1)),
                              dtype=torch.float32)
            xw = torch.tensor(rng.uniform(0, 2 * np.pi, size=(args.npts, 1)),
                              dtype=torch.float32)

            def closs(sd):
                net.load_state_dict(sd)
                with torch.enable_grad():
                    r = case.residual(net, tw, xw, case.T_w)
                ic = net(t0, ic_c) - ic_v
                return float((r ** 2).mean() + (ic ** 2).mean())
        else:
            tw = torch.tensor(rng.uniform(0, 1.01, size=(args.npts, 1)),
                              dtype=torch.float32)
            xw = torch.tensor(rng.uniform(-1, 1, size=(args.npts, 2)),
                              dtype=torch.float32)

            def closs(sd):
                net.load_state_dict(sd)
                with torch.enable_grad():
                    r = case.residual(net, tw, xw, case.T_w)
                ic = net(t0, ic_c[:, 0:1], ic_c[:, 1:2]) - ic_v
                return float((r ** 2).mean() + (ic ** 2).mean())

        tf = flat(fsd)
        diffs = torch.stack([flat(sd) - tf for sd in sds])
        _, _, V = torch.pca_lowrank(diffs, q=min(4, len(diffs)))
        d1 = V[:, 0] / V[:, 0].norm()
        d2 = V[:, 1] - (V[:, 1] @ d1) * d1
        d2 = d2 / d2.norm()
        a = np.append((diffs @ d1).numpy(), 0.0)
        b = np.append((diffs @ d2).numpy(), 0.0)
        pa = 0.25 * (a.max() - a.min() + 1e-9)
        pb = 0.25 * (b.max() - b.min() + 1e-9)
        print(f"  surface w{k} ...", flush=True)
        A, B, Z = surface(closs, fsd, d1, d2, (a.min() - pa, a.max() + pa),
                          (b.min() - pb, b.max() + pb), n=args.grid)
        m = hist["window"] == k
        l2 = float(hist["l2_window"][m][-1]) if m.any() else float("nan")
        panels.append(dict(k=k, A=A, B=B, Z=Z, a=a, b=b, steps=steps, l2=l2))

    # ---- the chain: all window solutions in one joint plane ----
    finals = []
    for k in range(cfg["n_win"]):
        p = os.path.join(cfg["final_dir"], "trajectory", f"w{k}_final_params.npz")
        if not os.path.exists(p):
            continue
        net.load_state_dict(jax_npz_to_state_dict(np.load(p)), strict=False)
        finals.append(flat(net.state_dict()).clone())
    T = torch.stack(finals)
    Tc = T - T.mean(0, keepdim=True)
    _, _, VV = torch.pca_lowrank(Tc, q=min(4, len(Tc)))
    chain = torch.stack([Tc @ VV[:, 0], Tc @ VV[:, 1]], dim=1).numpy()

    np.savez(f"analysis/out/full_training_{args.case}.npz",
             chain=chain, **{f"p{i}_{kk}": np.asarray(v)
                             for i, pn in enumerate(panels)
                             for kk, v in pn.items()})

    # ---- figure ----
    van = np.load(cfg["vanilla_cache"], allow_pickle=True)
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.15, 1], hspace=0.34, wspace=0.30)

    ax = fig.add_subplot(gs[0, :2])
    cs = ax.contourf(van["A"], van["B"], np.log10(van["Z"]), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 loss")
    ax.plot(van["a_traj"], van["b_traj"], "w.-", lw=1.8, ms=7)
    st = van["steps"]
    for i in [0, len(van["a_traj"]) // 2, len(van["a_traj"]) - 1]:
        lbl = st[i] if i < len(st) else st[-1]
        ax.annotate(f"it {lbl}", (van["a_traj"][i], van["b_traj"][i]), color="w",
                    fontsize=9, xytext=(5, 5), textcoords="offset points")
    ax.set_title("VANILLA: the complete training run (one network, one objective,\n"
                 "one descent) — and it ends on a plateau", fontsize=11)
    ax.set_xlabel("PCA dir 1"); ax.set_ylabel("PCA dir 2")

    ax = fig.add_subplot(gs[0, 2:])
    cmap = plt.cm.plasma(np.linspace(0.05, 0.95, len(chain)))
    ax.plot(chain[:, 0], chain[:, 1], "-", color="0.6", lw=1.4, zorder=3)
    ax.scatter(chain[:, 0], chain[:, 1], c=cmap, s=110, edgecolors="k",
               linewidths=0.6, zorder=5)
    for i in range(len(chain)):
        if i % max(1, len(chain) // 8) == 0 or i == len(chain) - 1:
            ax.annotate(f"w{i}", chain[i], fontsize=8.5, xytext=(6, 5),
                        textcoords="offset points")
    ax.set_title(f"CAUSAL: the complete run = {len(chain)} window solutions in weight space\n"
                 "(each window a FRESH net trained on its own objective — the curriculum chain)",
                 fontsize=11)
    ax.set_xlabel("joint PCA dir 1"); ax.set_ylabel("joint PCA dir 2")
    ax.grid(alpha=0.3)

    for i, pn in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        cs = ax.contourf(pn["A"], pn["B"], np.log10(pn["Z"] + 1e-14), levels=26,
                         cmap="viridis")
        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.03)
        ax.plot(pn["a"], pn["b"], "w.-", lw=1.4, ms=4)
        ax.scatter([0], [0], marker="*", s=170, color="red", zorder=6)
        ax.set_title(f"causal window {pn['k']}  (L2 = {pn['l2']:.1e})\n"
                     f"{len(pn['steps'])} snapshots, its own objective", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{cfg['title']}: the FULL training of both methods in error space — "
                 f"vanilla is one descent into a plateau; the causal run is {cfg['n_win']} "
                 "sequential descents, each into its own funnel", fontsize=13, y=0.975)
    fig.savefig(f"{OUT}/fig27_full_training_{args.case}.png", dpi=150,
                bbox_inches="tight")
    print(f"saved fig27_full_training_{args.case}.png")


if __name__ == "__main__":
    main()
