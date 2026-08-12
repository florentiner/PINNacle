"""Figure 17: the SAME trivial state in two error spaces, with real trajectories.

Left:  vanilla loss landscape (trajectory-PCA of the vanilla run) — the trivial
       state is the BOTTOM of the basin the trajectory falls into.
Right: causal window objective (unweighted window residual + 1e4*IC), PCA plane of
       the causal-from-trivial-init run (v3, sine encoding) — the same trivial state
       is HIGH ON A WALL (IC term 1e4*0.25 ~ 2.5e3) and the trajectory walks down to
       the true-branch solution.

Architectures differ (torch FNN vs JAX modified-MLP), so each panel uses its own
weight-space plane; the shared object is the trivial STATE (u=0) and its role.
"""
import glob
import sys

import numpy as onp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

OUT = "analysis/report_figs"
VAN = "runs/kaggle-trivial-vanilla/08.10-15.20.22-trivial-heatlt-vanilla/0-0"
V3 = "runs/kaggle-trivial-causal-triv-v3/out_heat_trivial_v3"


# ---------------- left panel: vanilla (torch) ----------------
def vanilla_panel_data():
    import torch
    from benchmark_sse import build_pde, unweighted_loss, flat_params, load_flat
    import deepxde as dde

    pde = build_pde("heatlt")
    net = dde.nn.FNN([pde.input_dim] + [100] * 5 + [pde.output_dim],
                     "tanh", "Glorot normal").float()
    model = pde.create_model(net)
    model.compile(torch.optim.Adam(net.parameters(), 1e-3),
                  loss_weights=onp.ones(pde.num_loss))
    model.pde = pde
    cks = sorted(glob.glob(VAN + "/trajectory/ckpt_*.pt"),
                 key=lambda p: int(p.split("_")[-1][:-3]))
    thetas, steps = [], []
    for p in cks:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model_state_dict"])
        thetas.append(flat_params(net).numpy())
        steps.append(int(p.split("_")[-1][:-3]))
    T = onp.stack(thetas)
    center = T[-1]
    X = T - center
    _, _, Vt = onp.linalg.svd(X, full_matrices=False)
    d1, d2 = Vt[0], Vt[1]
    xy = onp.stack([X @ d1, X @ d2], axis=1)
    rng = onp.random.default_rng(0)
    pool = rng.uniform([0, 0, 0], [1, 1, 100], size=(2048, 3)).astype(onp.float32)
    icg = rng.uniform([0, 0], [1, 1], size=(1024, 2)).astype(onp.float32)
    ic_pts = onp.hstack([icg, onp.zeros((1024, 1), onp.float32)])
    ic_vals = (onp.sin(4 * onp.pi * ic_pts[:, 0:1]) * onp.sin(3 * onp.pi * ic_pts[:, 1:2])).astype(onp.float32)
    bc = rng.uniform([0, 0, 0], [1, 1, 100], size=(512, 3)).astype(onp.float32)
    side = rng.integers(0, 4, 512)
    bc[:, 0] = onp.where(side == 0, 0, onp.where(side == 1, 1, bc[:, 0]))
    bc[:, 1] = onp.where(side == 2, 0, onp.where(side == 3, 1, bc[:, 1]))
    r1 = xy[:, 0].max() - xy[:, 0].min()
    r2 = max(xy[:, 1].max() - xy[:, 1].min(), 0.3 * r1)
    g1 = onp.linspace(xy[:, 0].min() - 0.2 * r1, xy[:, 0].max() + 0.2 * r1, 21)
    g2 = onp.linspace(min(xy[:, 1].min(), 0) - 0.2 * r2, xy[:, 1].max() + 0.2 * r2, 21)
    import torch as th
    Z = onp.zeros((len(g2), len(g1)))
    for i, b in enumerate(g2):
        for j, a in enumerate(g1):
            load_flat(net, th.tensor(center + a * d1 + b * d2, dtype=th.float32))
            l, _ = unweighted_loss(model, pde, pool, ic_pts, ic_vals, bc)
            Z[i, j] = l
        print(f"[vanilla] row {i+1}/{len(g2)}", flush=True)
    onp.savez("analysis/out/fig17_vanilla_cache.npz", Z=Z, g1=g1, g2=g2, xy=xy, steps=steps)
    return Z, g1, g2, xy, steps


# ---------------- right panel: causal v3 (jax) ----------------
def causal_panel_data():
    import jax
    import jax.numpy as np
    from jax import jit, vmap
    from causalpinn.jax_runner_heat import HeatCausalJax, load_ref_wide

    class A:  # args stub matching the v3 run
        encoding = "sine"; n_t = 16; n_s = 256; ic_grid = 32
        windows = 50; w_ic = 1e4

    ref_grid, xs, ys, t_star = load_ref_wide("ref/heat_longtime.dat")
    model = HeatCausalJax(A, ref_grid, xs, ys, t_star)

    KEYS = None

    def params_from_npz(path):
        d = onp.load(path)
        n = int(d["n_layers"])
        params = [(np.asarray(d[f"W{i}"]), np.asarray(d[f"bb{i}"])) for i in range(n)]
        return (params, np.asarray(d["U1"]), np.asarray(d["b1"]),
                np.asarray(d["U2"]), np.asarray(d["b2"]))

    def flat(params):
        p, U1, b1, U2, b2 = params
        arrs = [U1, b1, U2, b2] + [a for Wb in p for a in Wb]
        return onp.concatenate([onp.asarray(a).ravel() for a in arrs]), \
               [onp.asarray(a).shape for a in arrs]

    def unflat(vec, shapes, n_layers):
        arrs, i = [], 0
        for s in shapes:
            n = int(onp.prod(s))
            arrs.append(np.asarray(vec[i:i + n].reshape(s))); i += n
        U1, b1, U2, b2 = arrs[:4]
        rest = arrs[4:]
        params = [(rest[2 * k], rest[2 * k + 1]) for k in range(n_layers)]
        return (params, U1, b1, U2, b2)

    snaps = sorted(glob.glob(V3 + "/trajectory/w0_snap_*.npz"),
                   key=lambda p: int(p.split("_")[-1][:-4]))
    paths = [V3 + "/trajectory/trivial_init_params.npz"] + snaps + \
            [V3 + "/trajectory/w0_final_params.npz"]
    vecs, shapes, nl = [], None, None
    for p in paths:
        prm = params_from_npz(p)
        v, sh = flat(prm)
        vecs.append(v)
        shapes = sh
        nl = len(prm[0])
    T = onp.stack(vecs)
    center = T[-1]
    X = T - center
    _, _, Vt = onp.linalg.svd(X, full_matrices=False)
    d1, d2 = Vt[0], Vt[1]
    xy = onp.stack([X @ d1, X @ d2], axis=1)

    # fixed window-0 objective: unweighted residual + 1e4 * IC
    rng = onp.random.default_rng(0)
    t_r = np.asarray(onp.sort(rng.uniform(0, 1.01, A.n_t)))
    x_r = np.asarray(rng.uniform(0, 1, A.n_s))
    y_r = np.asarray(rng.uniform(0, 1, A.n_s))
    icx, icy = onp.asarray(model.ic_x), onp.asarray(model.ic_y)
    ic0 = np.asarray((onp.sin(4 * onp.pi * icx) * onp.sin(3 * onp.pi * icy))[:, None])
    r_batch = vmap(vmap(model.residual, (None, None, 0, 0, None)), (None, 0, None, None, None))
    u_ic = vmap(model.u_fn, (None, None, 0, 0))

    @jit
    def loss(params):
        r = r_batch(params, t_r, x_r, y_r, 0.0)
        lic = np.mean((u_ic(params, 0.0, model.ic_x, model.ic_y) - ic0) ** 2)
        return np.mean(r ** 2) + A.w_ic * lic

    r1 = xy[:, 0].max() - xy[:, 0].min()
    r2 = max(xy[:, 1].max() - xy[:, 1].min(), 0.3 * r1)
    g1 = onp.linspace(xy[:, 0].min() - 0.2 * r1, xy[:, 0].max() + 0.2 * r1, 21)
    g2 = onp.linspace(xy[:, 1].min() - 0.2 * r2, xy[:, 1].max() + 0.2 * r2, 21)
    Z = onp.zeros((len(g2), len(g1)))
    for i, b in enumerate(g2):
        for j, a in enumerate(g1):
            prm = unflat(center + a * d1 + b * d2, shapes, nl)
            Z[i, j] = float(loss(prm))
        print(f"[causal] row {i+1}/{len(g2)}", flush=True)
    onp.savez("analysis/out/fig17_causal_cache.npz", Z=Z, g1=g1, g2=g2, xy=xy)
    return Z, g1, g2, xy


def main():
    Zv, gv1, gv2, xyv, steps = vanilla_panel_data()
    Zc, gc1, gc2, xyc = causal_panel_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    cs = ax.contourf(gv1, gv2, onp.log10(Zv + 1e-12), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 vanilla loss")
    ax.plot(xyv[:, 0], xyv[:, 1], "w.-", lw=1.7, ms=6)
    ax.annotate("it 0 (random init)", xyv[0], color="w", fontsize=8,
                textcoords="offset points", xytext=(6, 6))
    ax.scatter([0], [0], marker="*", s=260, color="red", zorder=6)
    ax.annotate("TRIVIAL STATE u≈0:\nbottom of the basin (loss 4e-3)\nL2RE 0.999", (0, 0),
                textcoords="offset points", xytext=(-160, -50), color="red", fontsize=9)
    ax.set_title("VANILLA objective: the trajectory falls INTO the trivial state\n(it is the minimum)")
    ax.set_xlabel("PCA dir 1 (vanilla run)"); ax.set_ylabel("PCA dir 2")

    ax = axes[1]
    cs2 = ax.contourf(gc1, gc2, onp.log10(Zc + 1e-12), levels=30, cmap="viridis")
    fig.colorbar(cs2, ax=ax, label="log10 causal window objective")
    ax.plot(xyc[:, 0], xyc[:, 1], "w.-", lw=1.4, ms=4)
    ax.scatter([xyc[0, 0]], [xyc[0, 1]], marker="*", s=260, color="red", zorder=6)
    ax.annotate("the SAME trivial state:\nhigh on the IC-gate wall\n(objective ≈ 2.5e3)", xyc[0],
                textcoords="offset points", xytext=(10, 18), color="red", fontsize=9)
    ax.scatter([0], [0], marker="*", s=260, color="lime", zorder=6)
    ax.annotate("true-branch solution\n(window L2 0.162, corr 0.987)", (0, 0),
                textcoords="offset points", xytext=(8, -28), color="lime", fontsize=9)
    ax.set_title("CAUSAL objective (same trivial state as init): now it is a WALL —\nthe trajectory walks down to the true branch")
    ax.set_xlabel("PCA dir 1 (causal run)"); ax.set_ylabel("PCA dir 2")
    fig.suptitle("The same point in state space, two error spaces: minimum of the vanilla objective — wall of the causal one",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig17_two_objectives_trajectories.png", dpi=150, bbox_inches="tight")
    print("saved fig17")


if __name__ == "__main__":
    main()
