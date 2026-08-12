"""Figure 24: the paper's (arXiv:2303.03374) machinery shown IN error space —
the StarSSE star on the vanilla loss landscape.

Plane per PDE: PCA over {trivial checkpoint, children x1..x32}; terrain = unweighted
vanilla loss; rays trivial->child are the linear paths whose max-height-above-ends
is the paper's barrier (values annotated). Small kicks stay in the basin around the
trivial star; large kicks land across walls in other (equally wrong) basins.
"""
import glob
import json
import sys

import numpy as onp
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from benchmark_sse import build_pde, unweighted_loss, flat_params, load_flat  # noqa: E402

import deepxde as dde  # noqa: E402

OUT = "analysis/report_figs"


def panel_data(case, sse_glob, trivial_ckpt, grid_n, pool_n):
    pde = build_pde(case)
    net = dde.nn.FNN([pde.input_dim] + [100] * 5 + [pde.output_dim],
                     "tanh", "Glorot normal").float()
    model = pde.create_model(net)
    model.compile(torch.optim.Adam(net.parameters(), 1e-3),
                  loss_weights=onp.ones(pde.num_loss))
    model.pde = pde

    ck = torch.load(trivial_ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model_state_dict"])
    theta0 = flat_params(net).numpy()
    cdirs = sorted(glob.glob(sse_glob), key=lambda p: float(p.split("_k")[-1].split("_")[0]))
    kids, kicks = [], []
    for cd in cdirs:
        ckc = torch.load(cd + "/child_final.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(ckc["model_state_dict"])
        kids.append(flat_params(net).numpy())
        kicks.append(float(cd.split("_k")[-1].split("_")[0]))
    T = onp.vstack([theta0[None], onp.stack(kids)])
    center = theta0
    X = T - center
    _, _, Vt = onp.linalg.svd(X, full_matrices=False)
    d1, d2 = Vt[0], Vt[1]
    xy = X @ onp.stack([d1, d2]).T

    xall = model.data.train_x_all
    lo, hi = xall.min(0), xall.max(0)
    rng = onp.random.default_rng(0)
    pool = rng.uniform(lo, hi, size=(pool_n, len(lo))).astype(onp.float32)
    ic_pts = pool.copy(); ic_pts[:, -1] = lo[-1]
    if case == "heatlt":
        ic_vals = (onp.sin(4 * onp.pi * ic_pts[:, 0:1]) * onp.sin(3 * onp.pi * ic_pts[:, 1:2])).astype(onp.float32)
        nb = 512
        bc = rng.uniform(lo, hi, size=(nb, 3)).astype(onp.float32)
        side = rng.integers(0, 4, nb)
        bc[:, 0] = onp.where(side == 0, lo[0], onp.where(side == 1, hi[0], bc[:, 0]))
        bc[:, 1] = onp.where(side == 2, lo[1], onp.where(side == 3, hi[1], bc[:, 1]))
        bc_pts = bc
    else:
        ic_vals = (onp.cos(ic_pts[:, 0:1]) * (1 + onp.sin(ic_pts[:, 0:1]))).astype(onp.float32)
        bc_pts = None

    r1 = xy[:, 0].max() - xy[:, 0].min()
    r2 = max(xy[:, 1].max() - xy[:, 1].min(), 0.25 * r1)
    g1 = onp.linspace(xy[:, 0].min() - 0.15 * r1, xy[:, 0].max() + 0.15 * r1, grid_n)
    g2 = onp.linspace(xy[:, 1].min() - 0.25 * r2, xy[:, 1].max() + 0.25 * r2, grid_n)
    Z = onp.zeros((len(g2), len(g1)))
    for i, b in enumerate(g2):
        for j, a in enumerate(g1):
            load_flat(net, torch.tensor(center + a * d1 + b * d2, dtype=torch.float32))
            l, _ = unweighted_loss(model, pde, pool, ic_pts, ic_vals, bc_pts)
            Z[i, j] = l
        print(f"[{case}] row {i+1}/{len(g2)}", flush=True)
    return Z, g1, g2, xy, kicks


def main():
    res = json.load(open(glob.glob("runs/kaggle-trivial-heat-sse/**/sse_results.json", recursive=True)[0]))
    barH = {c["kick"]: c["barrier_to_trivial"] for c in res["children"]}
    resK = json.load(open(glob.glob("runs/kaggle-trivial-ks-sse/**/sse_results.json", recursive=True)[0]))
    barK = {c["kick"]: c["barrier_to_trivial"] for c in resK["children"]}

    ZH, gh1, gh2, xyH, kicksH = panel_data(
        "heatlt", "runs/kaggle-trivial-heat-sse/**/child_k*_s0",
        "runs/kaggle-trivial-vanilla/08.10-15.20.22-trivial-heatlt-vanilla/0-0/trajectory/ckpt_20000.pt",
        21, 2048)
    ZK, gk1, gk2, xyK, kicksK = panel_data(
        "ks", "runs/kaggle-trivial-ks-sse/**/child_k*_s0",
        "runs/07.18-13.19.39-baseline-chaotic/0-0/trajectory/ckpt_20000.pt",
        17, 1024)
    onp.savez("analysis/out/fig24_cache.npz", ZH=ZH, gh1=gh1, gh2=gh2, xyH=xyH,
              ZK=ZK, gk1=gk1, gk2=gk2, xyK=xyK)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), constrained_layout=True)
    for ax, Z, g1, g2, xy, kicks, bars, name in [
            (axes[0], ZH, gh1, gh2, xyH, kicksH, barH, "Heat-LT"),
            (axes[1], ZK, gk1, gk2, xyK, kicksK, barK, "KS")]:
        cs = ax.contourf(g1, g2, onp.log10(Z + 1e-12), levels=30, cmap="viridis")
        fig.colorbar(cs, ax=ax, label="log10 vanilla loss")
        colors = plt.cm.autumn(onp.linspace(0, 0.85, len(kicks)))
        for k in range(len(kicks)):
            p = xy[k + 1]
            ax.plot([0, p[0]], [0, p[1]], "-", color="w", lw=1.0, alpha=0.8, zorder=4)
            ax.scatter([p[0]], [p[1]], s=90, color=colors[k], edgecolors="k",
                       linewidths=0.6, zorder=6)
            ax.annotate(f"×{kicks[k]:g}\nbarrier {bars[kicks[k]]:.2g}",
                        p, textcoords="offset points", xytext=(7, 5), fontsize=8,
                        color="w")
        ax.scatter([0], [0], marker="*", s=320, color="red", zorder=7)
        ax.annotate("trivial (collapsed)\ncheckpoint — star center", (0, 0),
                    textcoords="offset points", xytext=(8, -30), color="red", fontsize=9)
        ax.set_title(f"{name}: the StarSSE star on the vanilla loss landscape —\n"
                     "small kicks stay in the basin, large kicks cross walls into other GHOST basins")
        ax.set_xlabel("PCA dir 1 (star plane)")
        ax.set_ylabel("PCA dir 2")
    fig.suptitle("arXiv:2303.03374 in error space: kick-controlled escape is real terrain — "
                 "but every basin the rays reach is a wrong minimum (all L2RE ≈ 1)", fontsize=12)
    fig.savefig(f"{OUT}/fig24_sse_star_landscape.png", dpi=150, bbox_inches="tight")
    print("saved fig24")


if __name__ == "__main__":
    main()
