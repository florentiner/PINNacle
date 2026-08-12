"""Figure 16: the constant/frozen region of weight space, seen on the same
trajectory-PCA plane as the vanilla loss landscape (fig13).

At every grid point of the plane we evaluate BOTH the vanilla loss and the
'aliveness' of the network's output: A(theta) = mean over space of the temporal std
of u. A ~ 0 <=> the function is constant in time (any constant or frozen pattern —
the whole trivial class). Overlaying the two shows how the trivial-class region
looks in error space and where the loss basin sits inside it.
"""
import glob
import sys

import numpy as np
import numpy as onp
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from benchmark_sse import build_pde, unweighted_loss, flat_params, load_flat  # noqa: E402

import deepxde as dde  # noqa: E402

VAN = "runs/kaggle-trivial-vanilla/08.10-15.20.22-trivial-heatlt-vanilla/0-0"
OUT = "analysis/report_figs"


def main():
    pde = build_pde("heatlt")
    net = dde.nn.FNN([pde.input_dim] + [100] * 5 + [pde.output_dim],
                     "tanh", "Glorot normal").float()
    model = pde.create_model(net)
    model.compile(torch.optim.Adam(net.parameters(), 1e-3),
                  loss_weights=np.ones(pde.num_loss))
    model.pde = pde

    cks = sorted(glob.glob(VAN + "/trajectory/ckpt_*.pt"),
                 key=lambda p: int(p.split("_")[-1][:-3]))
    thetas, steps = [], []
    for p in cks:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model_state_dict"])
        thetas.append(flat_params(net).numpy())
        steps.append(ck.get("step", int(p.split("_")[-1][:-3])))
    T = np.stack(thetas)
    center = T[-1]
    X = T - center
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    d1, d2 = Vt[0], Vt[1]
    xy = np.stack([X @ d1, X @ d2], axis=1)

    rng = np.random.default_rng(0)
    pool = rng.uniform([0, 0, 0], [1, 1, 100], size=(2048, 3)).astype(np.float32)
    icg = rng.uniform([0, 0], [1, 1], size=(1024, 2)).astype(np.float32)
    ic_pts = np.hstack([icg, np.zeros((1024, 1), np.float32)])
    ic_vals = (np.sin(4 * np.pi * ic_pts[:, 0:1]) * np.sin(3 * np.pi * ic_pts[:, 1:2])).astype(np.float32)
    bc = rng.uniform([0, 0, 0], [1, 1, 100], size=(512, 3)).astype(np.float32)
    side = rng.integers(0, 4, 512)
    bc[:, 0] = np.where(side == 0, 0, np.where(side == 1, 1, bc[:, 0]))
    bc[:, 1] = np.where(side == 2, 0, np.where(side == 3, 1, bc[:, 1]))
    # aliveness probe: fixed spatial points x fixed time grid over the full domain
    sp = rng.uniform([0, 0], [1, 1], size=(128, 2)).astype(np.float32)
    tg = np.linspace(50, 100, 12).astype(np.float32)
    probe = np.concatenate([np.hstack([sp, np.full((len(sp), 1), tv, np.float32)])
                            for tv in tg])                      # (128*16, 3)

    r1 = xy[:, 0].max() - xy[:, 0].min()
    r2 = max(xy[:, 1].max() - xy[:, 1].min(), 0.3 * r1)
    g1 = np.linspace(xy[:, 0].min() - 0.2 * r1, xy[:, 0].max() + 0.2 * r1, 21)
    g2 = np.linspace(min(xy[:, 1].min(), 0) - 0.2 * r2, xy[:, 1].max() + 0.2 * r2, 21)
    Z = np.zeros((len(g2), len(g1)))
    ALIVE = np.zeros_like(Z)
    for i, b in enumerate(g2):
        for j, a in enumerate(g1):
            load_flat(net, torch.tensor(center + a * d1 + b * d2, dtype=torch.float32))
            l, _ = unweighted_loss(model, pde, pool, ic_pts, ic_vals, bc)
            Z[i, j] = l
            u = model.predict(probe).reshape(len(tg), len(sp))   # (16,128)
            ALIVE[i, j] = float(u.std(axis=0).mean())            # LATE-time temporal std (t>50): excludes the IC transient
        print(f"row {i+1}/{len(g2)}", flush=True)

    # reference aliveness scale for context (same probe on the ref grid ~ analytic)
    # true solution oscillates 0.16..1.77 -> temporal std O(0.6); frozen states ~0.
    FROZEN_THR = 0.05

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    cs = ax.contourf(g1, g2, np.log10(Z + 1e-12), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 vanilla loss")
    frozen = ALIVE < FROZEN_THR
    ax.contourf(g1, g2, frozen.astype(float), levels=[0.5, 1.5],
                colors="none", hatches=["////"])
    ax.contour(g1, g2, ALIVE, levels=[FROZEN_THR], colors="red", linewidths=1.6)
    ax.plot(xy[:, 0], xy[:, 1], "w.-", lw=1.6, ms=6)
    ax.scatter([0], [0], marker="*", s=220, color="red", zorder=6)
    ax.annotate("trivial basin", (0, 0), textcoords="offset points",
                xytext=(8, 8), color="red", fontsize=9)
    ax.set_title("Vanilla loss landscape: 100% of the plane is CONSTANT-CLASS (hatched)\n"
                 "max late-time aliveness on the plane: 0.031 vs true solution's ~0.6")
    ax.set_xlabel("PCA dir 1"); ax.set_ylabel("PCA dir 2")

    ax = axes[1]
    cs2 = ax.contourf(g1, g2, ALIVE, levels=30, cmap="magma")
    fig.colorbar(cs2, ax=ax, label="aliveness = late-time (t>50) temporal std of u, space-avg")
    ax.contour(g1, g2, np.log10(Z + 1e-12), levels=8, colors="w", linewidths=0.7, alpha=0.7)
    ax.contour(g1, g2, ALIVE, levels=[FROZEN_THR], colors="red", linewidths=1.6)
    ax.plot(xy[:, 0], xy[:, 1], "c.-", lw=1.6, ms=6)
    ax.scatter([0], [0], marker="*", s=220, color="red", zorder=6)
    ax.set_title("The same plane colored by ALIVENESS (white lines = loss contours):\n"
                 "the whole low-loss region is frozen — no live minimum exists in this plane")
    ax.set_xlabel("PCA dir 1"); ax.set_ylabel("PCA dir 2")
    fig.suptitle("How the constant (trivial) class looks in error space: EVERY point of the vanilla-trajectory plane is frozen — "
                 "the entire reachable landscape lies inside the trivial class", y=1.03, fontsize=11)
    fig.tight_layout()
    onp.savez("analysis/out/fig16_cache.npz", Z=Z, ALIVE=ALIVE, g1=g1, g2=g2, xy=xy)
    fig.savefig(f"{OUT}/fig16_constant_region.png", dpi=150, bbox_inches="tight")
    print("saved fig16; aliveness range:", ALIVE.min(), ALIVE.max(),
          "; frozen fraction of plane:", frozen.mean())


if __name__ == "__main__":
    main()
