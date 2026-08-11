"""Figure 13: heat vanilla loss landscape (trajectory-PCA plane) descending into the
trivial basin + the detectable pattern (residual-layer evolution) highlighted.
Reads only archived checkpoints/arrays; loss = unweighted vanilla proxy on fixed sets.
"""
import glob
import sys

import numpy as np
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
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    d1, d2 = Vt[0], Vt[1]
    xy = np.stack([X @ d1, X @ d2], axis=1)

    # fixed eval sets
    rng = np.random.default_rng(0)
    pool = rng.uniform([0, 0, 0], [1, 1, 100], size=(2048, 3)).astype(np.float32)
    icg = rng.uniform([0, 0], [1, 1], size=(1024, 2)).astype(np.float32)
    ic_pts = np.hstack([icg, np.zeros((1024, 1), np.float32)])
    ic_vals = (np.sin(4 * np.pi * ic_pts[:, 0:1]) * np.sin(3 * np.pi * ic_pts[:, 1:2])).astype(np.float32)
    bc = rng.uniform([0, 0, 0], [1, 1, 100], size=(512, 3)).astype(np.float32)
    side = rng.integers(0, 4, 512)
    bc[:, 0] = np.where(side == 0, 0, np.where(side == 1, 1, bc[:, 0]))
    bc[:, 1] = np.where(side == 2, 0, np.where(side == 3, 1, bc[:, 1]))

    r1 = xy[:, 0].max() - xy[:, 0].min()
    r2 = max(xy[:, 1].max() - xy[:, 1].min(), 0.3 * r1)
    g1 = np.linspace(xy[:, 0].min() - 0.2 * r1, xy[:, 0].max() + 0.2 * r1, 21)
    g2 = np.linspace(min(xy[:, 1].min(), 0) - 0.2 * r2, xy[:, 1].max() + 0.2 * r2, 21)
    Z = np.zeros((len(g2), len(g1)))
    for i, b in enumerate(g2):
        for j, a in enumerate(g1):
            load_flat(net, torch.tensor(center + a * d1 + b * d2, dtype=torch.float32))
            l, _ = unweighted_loss(model, pde, pool, ic_pts, ic_vals, bc)
            Z[i, j] = l
        print(f"row {i+1}/{len(g2)}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    ax = axes[0]
    cs = ax.contourf(g1, g2, np.log10(Z + 1e-12), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 vanilla loss")
    ax.plot(xy[:, 0], xy[:, 1], "w.-", lw=1.6, ms=6)
    for k in [0, len(xy) // 2, len(xy) - 1]:
        ax.annotate(f"it {steps[k]}", (xy[k, 0], xy[k, 1]), color="w", fontsize=8,
                    textcoords="offset points", xytext=(6, 5))
    ax.scatter([0], [0], marker="*", s=220, color="red", zorder=6)
    ax.annotate("trivial basin\n(u≈0: PDE residual ≡ 0,\nloss = tiny IC/BC leftovers;\nL2RE = 0.999)",
                (0, 0), textcoords="offset points", xytext=(-150, -58), color="red", fontsize=9)
    ax.set_xlabel("PCA dir 1 (collapse direction)")
    ax.set_ylabel("PCA dir 2")
    ax.set_title("Vanilla loss landscape (trajectory-PCA plane):\nthe descent into the trivial basin — a certified wrong minimum")

    # panel B: THE PATTERN — residual layer shrinking toward t=0 across training
    resids = sorted(glob.glob(VAN + "/arrays/resid_*.npy"),
                    key=lambda p: int(p.split("_")[-1][:-4]))
    rows, labels = [], []
    for p in resids:
        r = np.load(p)[..., 0]
        rows.append(np.sqrt((r ** 2).mean(axis=(0, 1))))
        labels.append(int(p.split("_")[-1][:-4]))
    Rt = np.stack(rows)
    ax = axes[1]
    im = ax.imshow(np.log10(Rt + 1e-9), aspect="auto", origin="lower", cmap="magma",
                   extent=[0, 100, 0, len(rows) - 1])
    fig.colorbar(im, ax=ax, label="log10 residual RMS")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("t")
    ax.set_ylabel("training iteration (snapshot)")
    ax.axvspan(0, 5, color="cyan", alpha=0.18)
    ax.text(6, len(rows) - 3.2, "THE PATTERN: all residual mass\nretreats into a thin layer at t≈0\n(C_enrich 20×; reference-free)",
            color="w", fontsize=9)
    ax.set_title("The detectable pattern in error space:\nresidual concentrates into the causality-violating layer")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig13_trivial_landscape_pattern.png", dpi=150, bbox_inches="tight")
    print("saved fig13")


if __name__ == "__main__":
    main()
