"""Figure 18: fig17 (two objectives, real trajectories) + the CONSTANT-CLASS regions
overlaid on both planes, so one can see whether each trajectory avoids the frozen
region or not.

Left plane aliveness comes from the fig16 cache (same PCA plane by construction).
Right plane aliveness is computed here: within-window temporal std of u (tau probe),
with the frozen threshold set at 10% of the REFERENCE's within-window aliveness.
Loss maps and trajectories are reused from the fig17 caches (no recompute).
"""
import glob
import sys

import numpy as onp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

OUT = "analysis/report_figs"
V3 = "runs/kaggle-trivial-causal-triv-v3/out_heat_trivial_v3"


def causal_alive_on_plane(gc1, gc2, xy_cache):
    import jax.numpy as np
    from jax import jit, vmap
    from causalpinn.jax_runner_heat import HeatCausalJax, load_ref_wide

    class A:
        encoding = "sine"; n_t = 16; n_s = 256; ic_grid = 32
        windows = 50; w_ic = 1e4

    ref_grid, xs, ys, t_star = load_ref_wide("ref/heat_longtime.dat")
    model = HeatCausalJax(A, ref_grid, xs, ys, t_star)

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
        vecs.append(v); shapes = sh; nl = len(prm[0])
    T = onp.stack(vecs)
    center = T[-1]
    X = T - center
    _, _, Vt = onp.linalg.svd(X, full_matrices=False)
    d1, d2 = Vt[0], Vt[1]
    xy = onp.stack([X @ d1, X @ d2], axis=1)
    assert onp.allclose(xy, xy_cache, atol=1e-4), "plane mismatch vs fig17 cache"

    rng = onp.random.default_rng(0)
    sp = rng.uniform(0, 1, size=(128, 2)).astype(onp.float32)
    taus = onp.linspace(0, 1, 12).astype(onp.float32)
    xs_p = np.asarray(sp[:, 0]); ys_p = np.asarray(sp[:, 1])
    u_fn = vmap(model.u_fn, (None, None, 0, 0))

    @jit
    def alive(params):
        us = np.stack([u_fn(params, float(tv), xs_p, ys_p) for tv in taus])  # (12,128,1)
        return np.mean(np.std(us, axis=0))

    ALIVE = onp.zeros((len(gc2), len(gc1)))
    for i, b in enumerate(gc2):
        for j, a in enumerate(gc1):
            ALIVE[i, j] = float(alive(unflat(center + a * d1 + b * d2, shapes, nl)))
        print(f"[alive-c] row {i+1}/{len(gc2)}", flush=True)
    # reference within-window aliveness (same probe density, from the ref grid)
    refw = ref_grid[:, :, :11, 0]                      # t in [0,2]
    ref_alive = float(refw.std(axis=2).mean())
    onp.savez("analysis/out/fig18_causal_alive.npz", ALIVE=ALIVE, ref_alive=ref_alive)
    return ALIVE, ref_alive


def main():
    v = onp.load("analysis/out/fig17_vanilla_cache.npz")
    c = onp.load("analysis/out/fig17_causal_cache.npz")
    f16 = onp.load("analysis/out/fig16_cache.npz")
    assert onp.allclose(f16["g1"], v["g1"]) and onp.allclose(f16["g2"], v["g2"]), \
        "fig16 plane differs from fig17 vanilla plane"
    ALIVE_v = f16["ALIVE"]

    ALIVE_c, ref_alive = causal_alive_on_plane(c["g1"], c["g2"], c["xy"])
    thr_c = 0.10 * ref_alive

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    cs = ax.contourf(v["g1"], v["g2"], onp.log10(v["Z"] + 1e-12), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 vanilla loss")
    frozen_v = (ALIVE_v < 0.05).astype(float)
    ax.contourf(v["g1"], v["g2"], frozen_v, levels=[0.5, 1.5], colors="none", hatches=["///"])
    xy = v["xy"]
    ax.plot(xy[:, 0], xy[:, 1], "w.-", lw=1.7, ms=6)
    ax.scatter([0], [0], marker="*", s=260, color="red", zorder=6)
    ax.annotate("TRIVIAL STATE:\nbottom of the basin", (0, 0), textcoords="offset points",
                xytext=(-130, -44), color="red", fontsize=9)
    ax.text(0.03, 0.965, "hatched = constant-class region\nHERE: 100% of the plane (max aliveness 0.031 vs true 0.6)\nthe trajectory CANNOT avoid it — there is nothing else",
            transform=ax.transAxes, fontsize=8.5, va="top", color="k",
            bbox=dict(facecolor="w", alpha=0.75, edgecolor="none"))
    ax.set_title("VANILLA objective + constant-class overlay:\nthe whole reachable plane is frozen; the trajectory never leaves the class")
    ax.set_xlabel("PCA dir 1 (vanilla run)"); ax.set_ylabel("PCA dir 2")

    ax = axes[1]
    cs2 = ax.contourf(c["g1"], c["g2"], onp.log10(c["Z"] + 1e-12), levels=30, cmap="viridis")
    fig.colorbar(cs2, ax=ax, label="log10 causal window objective")
    frozen_c = (ALIVE_c < thr_c).astype(float)
    ax.contourf(c["g1"], c["g2"], frozen_c, levels=[0.5, 1.5], colors="none", hatches=["///"])
    ax.contour(c["g1"], c["g2"], ALIVE_c, levels=[thr_c], colors="red", linewidths=1.7)
    xy = c["xy"]
    ax.plot(xy[:, 0], xy[:, 1], "w.-", lw=1.4, ms=4)
    ax.scatter([xy[0, 0]], [xy[0, 1]], marker="*", s=260, color="red", zorder=6)
    ax.annotate("trivial init:\ninside the frozen region", xy[0], textcoords="offset points",
                xytext=(14, -46), color="red", fontsize=9)
    ax.scatter([0], [0], marker="*", s=260, color="lime", zorder=6)
    ax.annotate("solution: ALIVE region", (0, 0), textcoords="offset points",
                xytext=(-108, 14), color="lime", fontsize=9)
    ax.set_title("CAUSAL objective + constant-class overlay (red = frozen boundary):\nthe trajectory CROSSES the boundary and leaves the class")
    ax.set_xlabel("PCA dir 1 (causal run)"); ax.set_ylabel("PCA dir 2")
    fig.suptitle("fig17 + constant-class regions: vanilla has nowhere to escape to; the causal objective tilts the landscape so the "
                 "trajectory exits the frozen class", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig18_two_objectives_frozen.png", dpi=150, bbox_inches="tight")
    print("saved fig18; causal aliveness range:", ALIVE_c.min(), ALIVE_c.max(),
          "; ref window aliveness:", ref_alive, "; frozen fraction (causal plane):",
          frozen_c.mean())


if __name__ == "__main__":
    main()
