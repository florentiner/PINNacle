"""Additional evidence figures for REPORT_TRIVIAL.md (each proves one specific claim).

fig18 (redraw): two objectives + constant-class overlays, trajectories COLORED by
                their own aliveness -> shows exactly where each path leaves the class.
fig19: the ladder as weight-space REGIONS (joint vanilla+march plane, same torch
       arch): level bands 0 / sqrt(pi) / sqrt(2pi) + both trajectories.
fig20: policy timelines from logs — heat veto sawtooth (6 kicks, no false accepts)
       and KS march honest stall.
fig22: early warning — detector signals vs training iteration on vanilla runs
       (the pattern is detectable long before convergence).
fig12 (upgrade): gate escape, plain vs v3 panels.
"""
import csv
import glob
import sys

import numpy as onp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
sys.path.insert(0, "analysis")

OUT = "analysis/report_figs"
VAN = "runs/kaggle-trivial-vanilla/08.10-15.20.22-trivial-heatlt-vanilla/0-0"
V3 = "runs/kaggle-trivial-causal-triv-v3/out_heat_trivial_v3"
SQ = [0.0, onp.sqrt(onp.pi), onp.sqrt(2 * onp.pi), onp.sqrt(3 * onp.pi)]


# ---------------- shared torch helpers ----------------
def torch_model():
    import torch
    from benchmark_sse import build_pde
    import deepxde as dde
    pde = build_pde("heatlt")
    net = dde.nn.FNN([pde.input_dim] + [100] * 5 + [pde.output_dim],
                     "tanh", "Glorot normal").float()
    model = pde.create_model(net)
    model.compile(torch.optim.Adam(net.parameters(), 1e-3),
                  loss_weights=onp.ones(pde.num_loss))
    model.pde = pde
    return model, net


def load_ckpt_vec(net, path):
    import torch
    from benchmark_sse import flat_params
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model_state_dict"])
    return flat_params(net).numpy()


# ---------------- fig18 redraw ----------------
def fig18():
    import torch
    from benchmark_sse import load_flat
    import jax.numpy as np
    from jax import jit, vmap
    from causalpinn.jax_runner_heat import HeatCausalJax, load_ref_wide

    v = onp.load("analysis/out/fig17_vanilla_cache.npz")
    c = onp.load("analysis/out/fig17_causal_cache.npz")
    f16 = onp.load("analysis/out/fig16_cache.npz")
    a18 = onp.load("analysis/out/fig18_causal_alive.npz")
    ALIVE_c, ref_alive = a18["ALIVE"], float(a18["ref_alive"])

    # aliveness AT the vanilla trajectory points (late-time probe, torch)
    model, net = torch_model()
    rng = onp.random.default_rng(0)
    sp = rng.uniform([0, 0], [1, 1], size=(128, 2)).astype(onp.float32)
    tg = onp.linspace(50, 100, 12).astype(onp.float32)
    probe = onp.concatenate([onp.hstack([sp, onp.full((len(sp), 1), tv, onp.float32)])
                             for tv in tg])
    cks = sorted(glob.glob(VAN + "/trajectory/ckpt_*.pt"),
                 key=lambda p: int(p.split("_")[-1][:-3]))
    al_v = []
    for p in cks:
        load_ckpt_vec(net, p)
        u = model.predict(probe).reshape(len(tg), len(sp))
        al_v.append(float(u.std(axis=0).mean()))

    # aliveness AT the causal trajectory points (window probe, jax)
    class A:
        encoding = "sine"; n_t = 16; n_s = 256; ic_grid = 32
        windows = 50; w_ic = 1e4
    ref_grid, xs, ys, t_star = load_ref_wide("ref/heat_longtime.dat")
    cm = HeatCausalJax(A, ref_grid, xs, ys, t_star)

    def params_from_npz(path):
        d = onp.load(path)
        n = int(d["n_layers"])
        params = [(np.asarray(d[f"W{i}"]), np.asarray(d[f"bb{i}"])) for i in range(n)]
        return (params, np.asarray(d["U1"]), np.asarray(d["b1"]),
                np.asarray(d["U2"]), np.asarray(d["b2"]))
    snaps = sorted(glob.glob(V3 + "/trajectory/w0_snap_*.npz"),
                   key=lambda p: int(p.split("_")[-1][:-4]))
    paths = [V3 + "/trajectory/trivial_init_params.npz"] + snaps + \
            [V3 + "/trajectory/w0_final_params.npz"]
    xs_p = np.asarray(sp[:, 0]); ys_p = np.asarray(sp[:, 1])
    taus = onp.linspace(0, 1, 12)
    u_fn = vmap(cm.u_fn, (None, None, 0, 0))

    @jit
    def alive_j(params):
        us = np.stack([u_fn(params, float(tv), xs_p, ys_p) for tv in taus])
        return np.mean(np.std(us, axis=0))
    al_c = [float(alive_j(params_from_npz(p))) for p in paths]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    ax = axes[0]
    cs = ax.contourf(v["g1"], v["g2"], onp.log10(v["Z"] + 1e-12), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 vanilla loss")
    ax.contourf(v["g1"], v["g2"], (f16["ALIVE"] < 0.05).astype(float),
                levels=[0.5, 1.5], colors="none", hatches=["///"])
    xy = v["xy"]
    ax.plot(xy[:, 0], xy[:, 1], "-", color="w", lw=1.2, zorder=4)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=al_v, cmap="spring", vmin=0, vmax=0.6,
                    s=45, edgecolors="k", linewidths=0.4, zorder=5)
    ax.scatter([0], [0], marker="*", s=240, color="red", zorder=6)
    ax.text(0.03, 0.97, "hatched = constant class (100% of the plane)\ntrajectory colored by ITS OWN aliveness:\nstays dark violet = never leaves the class",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(facecolor="w", alpha=0.8, edgecolor="none"))
    ax.set_title("VANILLA: the trajectory never exits the constant class")
    ax.set_xlabel("PCA dir 1 (vanilla run)"); ax.set_ylabel("PCA dir 2")

    ax = axes[1]
    cs2 = ax.contourf(c["g1"], c["g2"], onp.log10(c["Z"] + 1e-12), levels=30, cmap="viridis")
    fig.colorbar(cs2, ax=ax, label="log10 causal window objective")
    lowlife = (ALIVE_c < 0.25 * ref_alive).astype(float)
    ax.contourf(c["g1"], c["g2"], lowlife, levels=[0.5, 1.5], colors="none", hatches=["\\\\\\"])
    xyc = c["xy"]
    ax.plot(xyc[:, 0], xyc[:, 1], "-", color="w", lw=1.0, zorder=4)
    ax.scatter(xyc[:, 0], xyc[:, 1], c=al_c, cmap="spring", vmin=0, vmax=0.6,
               s=40, edgecolors="k", linewidths=0.3, zorder=5)
    ax.scatter([xyc[0, 0]], [xyc[0, 1]], marker="*", s=240, color="red", zorder=6)
    ax.scatter([0], [0], marker="*", s=240, color="lime", zorder=6)
    ax.annotate(f"trivial init: aliveness {al_c[0]:.3f}\n(frozen; the frozen set is a sliver\nnarrower than one grid cell)",
                xyc[0], textcoords="offset points", xytext=(12, -54), color="red", fontsize=8.5)
    ax.annotate(f"solution: aliveness {al_c[-1]:.2f}\n(reference: {ref_alive:.2f})", (0, 0),
                textcoords="offset points", xytext=(-128, 14), color="k", fontsize=8.5,
                bbox=dict(facecolor="w", alpha=0.7, edgecolor="none"))
    fig.colorbar(sc, ax=axes.ravel().tolist(), location="bottom", shrink=0.45,
                 pad=0.04, label="trajectory aliveness (0 = constant class; reference 0.41)")
    ax.set_title("CAUSAL: the trajectory exits the class within the first snapshots\n(hatched = low-life zone < 25% of reference)")
    ax.set_xlabel("PCA dir 1 (causal run)"); ax.set_ylabel("PCA dir 2")
    fig.suptitle("Same as fig17 + constant-class overlays; trajectories colored by their own aliveness\n(note: even the vanilla RANDOM INIT is already near-frozen at late times — aliveness 0.023)", fontsize=11)
    fig.savefig(f"{OUT}/fig18_two_objectives_frozen.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig18 redrawn; vanilla traj aliveness:", [round(a, 3) for a in al_v[:3]], "...",
          "| causal traj:", [round(a, 3) for a in al_c[:4]], "...", round(al_c[-1], 3))


# ---------------- fig19: ladder regions ----------------
def fig19():
    import torch
    from benchmark_sse import load_flat
    model, net = torch_model()
    van_cks = sorted(glob.glob(VAN + "/trajectory/ckpt_*.pt"),
                     key=lambda p: int(p.split("_")[-1][:-3]))
    mar_dir = glob.glob("runs/kaggle-trivial-heat-march/*march-heatlt*/0-0/trajectory")[0]
    mar_cks = sorted(glob.glob(mar_dir + "/ckpt_*.pt"),
                     key=lambda p: int(p.split("_")[-1][:-3]))
    Tv = onp.stack([load_ckpt_vec(net, p) for p in van_cks])
    Tm = onp.stack([load_ckpt_vec(net, p) for p in mar_cks])
    T = onp.vstack([Tv, Tm])
    center = Tv[-1]                     # vanilla final = rung-0 anchor
    X = T - center
    _, _, Vt = onp.linalg.svd(X, full_matrices=False)
    d1, d2 = Vt[0], Vt[1]
    xyv = (Tv - center) @ onp.stack([d1, d2]).T
    xym = (Tm - center) @ onp.stack([d1, d2]).T

    rng = onp.random.default_rng(0)
    sp = rng.uniform([0, 0], [1, 1], size=(160, 2)).astype(onp.float32)
    tg = onp.linspace(5, 20, 8).astype(onp.float32)      # plateau probe, past IC transient
    probe = onp.concatenate([onp.hstack([sp, onp.full((len(sp), 1), tv, onp.float32)])
                             for tv in tg])
    allxy = onp.vstack([xyv, xym])
    r1 = allxy[:, 0].max() - allxy[:, 0].min()
    r2 = max(allxy[:, 1].max() - allxy[:, 1].min(), 0.3 * r1)
    g1 = onp.linspace(allxy[:, 0].min() - 0.15 * r1, allxy[:, 0].max() + 0.15 * r1, 23)
    g2 = onp.linspace(allxy[:, 1].min() - 0.15 * r2, allxy[:, 1].max() + 0.15 * r2, 23)
    LVL = onp.zeros((len(g2), len(g1)))
    import torch as th
    for i, b in enumerate(g2):
        for j, a in enumerate(g1):
            load_flat(net, th.tensor(center + a * d1 + b * d2, dtype=th.float32))
            u = model.predict(probe)
            LVL[i, j] = float(onp.percentile(onp.abs(u), 95))
        print(f"[fig19] row {i+1}/{len(g2)}", flush=True)
    onp.savez("analysis/out/fig19_cache.npz", LVL=LVL, g1=g1, g2=g2, xyv=xyv, xym=xym)

    bounds = [0, SQ[1] / 2, (SQ[1] + SQ[2]) / 2, (SQ[2] + SQ[3]) / 2, 4.0]
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(["#2b2d42", "#1b9e77", "#d95f02", "#7570b3"])
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    pm = ax.pcolormesh(g1, g2, LVL, cmap=cmap, norm=norm, shading="auto", alpha=0.85)
    cb = fig.colorbar(pm, ax=ax, ticks=[SQ[1] / 4, SQ[1], SQ[2], SQ[3] * 0.95])
    cb.ax.set_yticklabels(["rung 0\n(trivial)", "rung $\\sqrt{\\pi}$\n(TRUE level)",
                           "rung $\\sqrt{2\\pi}$", "rung $\\sqrt{3\\pi}$+"])
    cb.set_label("plateau level of the network output (p95 |u|, t∈[5,20])")
    ax.plot(xyv[:, 0], xyv[:, 1], "o-", color="w", lw=2, ms=5, label="vanilla trajectory (stays on rung 0)")
    ax.plot(xym[:, 0], xym[:, 1], "s--", color="yellow", lw=2, ms=5,
            label="march trajectory (hops rung 0 → rung $\\sqrt{2\\pi}$)")
    ax.annotate("vanilla final\n(rung 0)", xyv[-1], textcoords="offset points",
                xytext=(8, -26), color="w", fontsize=9)
    ax.annotate("march final\n(rung $\\sqrt{2\\pi}$ — wrong level)", xym[-1],
                textcoords="offset points", xytext=(8, 10), color="yellow", fontsize=9)
    ax.set_xlabel("PCA dir 1 (joint plane of both runs)")
    ax.set_ylabel("PCA dir 2")
    ax.set_title("The ladder of constants as REGIONS of weight space (same architecture, joint PCA plane):\n"
                 "each colored band is a self-gating level; escape policies hop between bands, "
                 "but the TRUE band ($\\sqrt{\\pi}$) is not where they land")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(f"{OUT}/fig19_ladder_regions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig19 saved; level range:", LVL.min(), LVL.max())


# ---------------- fig20: policy timelines ----------------
def fig20():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    g = glob.glob("runs/kaggle-trivial-heat-veto/*veto-heatlt*/0-0/trivialguard_log.csv")[0]
    rows = list(csv.DictReader(open(g)))
    st = onp.array([int(r["step"]) for r in rows])
    ts = onp.array([float(r["tstar_frac"]) for r in rows])
    ax = axes[0]
    ax.plot(st / 1e3, ts * 100, "-", color="tab:blue", lw=1.8)
    resets = onp.where(onp.diff(ts) < -0.05)[0]
    for k, i in enumerate(resets):
        ax.axvline(st[i + 1] / 1e3, color="tab:red", ls="--", lw=1)
        if k == 0:
            ax.text(st[i + 1] / 1e3 + 0.4, 12.5, "veto → kick → re-march", color="tab:red",
                    fontsize=8, rotation=90)
    ax.set_xlabel("iteration (×10³)"); ax.set_ylabel("covered horizon t* (time units)")
    ax.set_title("Heat rung-veto timeline: 6 cycles march→frozen-tail veto→kick;\n"
                 "every dead rung refused, none accepted (0 false accepts)")
    ax.grid(alpha=0.3)
    g2 = glob.glob("runs/kaggle-trivial-ks-march/*/0-0/trivialguard_log.csv")[0]
    rows2 = list(csv.DictReader(open(g2)))
    st2 = onp.array([int(r["step"]) for r in rows2])
    ts2 = onp.array([float(r["tstar_frac"]) for r in rows2])
    r2c = onp.array([float(r["r2_cov"]) for r in rows2])
    ax = axes[1]
    ax.plot(st2 / 1e3, ts2, "-", color="tab:blue", lw=1.8, label="covered horizon t*")
    ax.set_xlabel("iteration (×10³)"); ax.set_ylabel("t* (fraction of domain)", color="tab:blue")
    ax2 = ax.twinx()
    ax2.semilogy(st2 / 1e3, r2c, color="tab:orange", lw=1.2, label="covered residual")
    ax2.axhline(3e-3, color="k", ls=":", lw=1)
    ax2.text(25, 3.6e-3, "advance threshold", fontsize=8)
    ax2.set_ylabel("mean covered residual²", color="tab:orange")
    ax.set_title("KS march timeline: honest stall — the residual never clears the\n"
                 "threshold again, so the policy refuses to advance (no goodharting)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig20_policy_timelines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig20 saved")


# ---------------- fig22: early warning ----------------
def fig22():
    from trivial_detector import signals
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for name, arrays, color in [("heat vanilla", VAN + "/arrays", "tab:red"),
                                ("KS vanilla", "runs/07.18-13.19.39-baseline-chaotic/0-0/arrays", "tab:orange")]:
        preds = sorted(glob.glob(arrays + "/pred_*.npy"), key=lambda p: int(p.split("_")[-1][:-4]))
        its, ce = [], []
        for p in preds:
            it = int(p.split("_")[-1][:-4])
            r = arrays + f"/resid_{it}.npy"
            s = signals(onp.load(p), onp.load(r))
            its.append(it); ce.append(s["C_enrich"])
        ax.plot(its, ce, "o-", color=color, lw=1.8, label=f"{name}: C_enrich")
    ax.axhline(3, color="k", ls="--", lw=1)
    ax.text(200, 3.3, "detection threshold (C_enrich = 3)", fontsize=8.5)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("C_enrich (early-residual concentration)")
    ax.set_title("Early warning: the trivial-branch pattern is detectable from ~iteration 2000 —\n"
                 "an agent can intervene long before the 20k-iteration budget is spent")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/fig22_early_warning.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig22 saved")


# ---------------- fig12 upgrade ----------------
def fig12():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=False)
    for ax, path, title in [
        (axes[0], "runs/kaggle-trivial-causal-triv/out_heat_trivial/causal/history_jax.npz",
         "plain encoding, Δt=5 window: escapes (L2 1.00→0.85)\nbut representation-limited"),
        (axes[1], V3 + "/causal/history_jax.npz",
         "v3: sine + Δt=2 window: escapes to the TRUE branch\n(L2 1.00→0.162, corr 0.987)")]:
        h = onp.load(path, allow_pickle=True)
        m = h["window"] == 0
        st, lic, l2 = h["step"][m], h["loss_ic"][m], h["l2_window"][m]
        ax.semilogy(st / 1e3, lic, color="tab:blue", lw=1.6, label="IC loss (gate)")
        ax.set_xlabel("iteration (×10³)")
        ax.set_ylabel("IC loss (log)", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(st / 1e3, l2, color="tab:green", lw=1.8, label="window L2 error")
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel("window L2", color="tab:green")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Escape from inside the trivial attractor, both causal configs: the 10⁴·L_IC gate forces the exit; "
                 "matched geometry then reaches the true branch", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig12_gate_escape.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig12 upgraded")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "20"): fig20()
    if which in ("all", "22"): fig22()
    if which in ("all", "12"): fig12()
    if which in ("all", "19"): fig19()
    if which in ("all", "18"): fig18()
    print("done")
