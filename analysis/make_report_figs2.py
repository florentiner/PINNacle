"""Report figures, batch 2: solution heatmaps, per-window cost, P1 traces, certification cliff.

Reads only archived arrays under runs/ — no retraining.
"""
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "analysis/report_figs"

KS_C = "runs/kaggle-causal-ks-session9/causal_ks/0-0"
GS_C = "runs/kaggle-causal-gs-session7/causal_gs_jax/0-0"
KS_B = "runs/07.18-13.19.39-baseline-chaotic/0-0"
GS_B = "runs/07.18-13.19.39-baseline-chaotic/1-0"
AD8 = "runs/kaggle-adaptive-w8/causal_ks_adaptive/0-0"


def fig0_ks():
    ref = np.load(f"{KS_C}/arrays/ref.npy")[..., 0]
    van = np.load(f"{KS_B}/arrays/pred_20000.npy")[..., 0]
    cau = np.load(f"{KS_C}/arrays/pred_stitched_final.npy")[..., 0]
    vmax = np.abs(ref).max()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    titles = ["reference (ETDRK4)", "original PINN — L2RE 1.007", "causal PINN — L2RE 3.56e-2"]
    for ax, arr, ti in zip(axes, [ref, van, cau], titles):
        im = ax.imshow(arr, extent=[0, 1, 0, 2 * np.pi], origin="lower", aspect="auto",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(ti)
        ax.set_xlabel("t")
    axes[0].set_ylabel("x")
    fig.colorbar(im, ax=axes, shrink=0.9, label="u(x,t)")
    fig.suptitle("Kuramoto–Sivashinsky: what each method actually produces", y=1.02)
    fig.savefig(f"{OUT}/fig0_ks_solutions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig0_gs():
    ref = np.load(f"{GS_C}/arrays/ref.npy")
    van = np.load(f"{GS_B}/arrays/pred_20000.npy")
    cau = np.load(f"{GS_C}/arrays/pred_stitched_final.npy")
    tidx, comp = 20, 1  # t = 200, v component (the spot pattern)
    fields = [ref[:, :, tidx, comp], van[:, :, tidx, comp], cau[:, :, tidx, comp]]
    vmin, vmax = fields[0].min(), fields[0].max()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    titles = ["reference", "original PINN — L2RE 0.094", "causal PINN — L2RE 1.42e-2"]
    for ax, arr, ti in zip(axes, fields, titles):
        im = ax.imshow(arr.T, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_title(ti)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes, shrink=0.9, label="v(x,y, t=200)")
    fig.suptitle("Gray–Scott, final time t=200 (v component): spot pattern fidelity", y=1.02)
    fig.savefig(f"{OUT}/fig0_gs_solutions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig5_cost():
    hk = np.load(f"{KS_C}/causal/history_jax.npz", allow_pickle=True)
    hg = np.load(f"{GS_C}/causal/history_jax.npz", allow_pickle=True)
    ks_it = [int(hk["step"][hk["window"] == w].max()) for w in range(10)]
    gs_it = [int(hg["step"][hg["window"] == w].max()) for w in range(20)]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xk = (np.arange(10) + 0.5) / 10
    xg = (np.arange(20) + 0.5) / 20
    ax.plot(xk, np.array(ks_it) / 1e3, "o-", color="tab:red", label="KS (4th order, stiff)")
    ax.plot(xg, np.array(gs_it) / 1e3, "s-", color="tab:blue", label="GS (2nd order)")
    ax.axhline(735, color="tab:red", ls=":", lw=1)
    ax.text(0.01, 745, "KS window budget ceiling", color="tab:red", fontsize=9)
    ax.set_xlabel("position in time domain (window midpoint, normalized)")
    ax.set_ylabel("Adam iterations per window  (×10³)")
    ax.set_title("Training cost per window: landscape geometry predicts it\n"
                 "KS escalates 237k → ceiling as chaos deepens; GS stays flat (~205k) in its wide basin")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/fig5_cost_per_window.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig6_p1():
    hk = np.load(f"{KS_C}/causal/history_jax.npz", allow_pickle=True)
    ha = np.load(f"{AD8}/causal/history_jax.npz", allow_pickle=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True)
    for ax, d, name, color in [
        (axes[0], hk, "fixed hand-tuned schedule — 735k iters", "tab:blue"),
        (axes[1], ha, "adaptive stage controller (RL policy class) — 615k iters (−16%)", "tab:orange"),
    ]:
        m = d["window"] == 8
        st, wm, sg, tol = d["step"][m], d["w_min"][m], d["stage"][m], d["tol"][m]
        ax.plot(st / 1e3, wm, color=color, lw=0.9)
        ax.axhline(0.99, color="k", ls="--", lw=0.8)
        ax.text(3, 1.005, "certificate  W$_{min}$ = 0.99", fontsize=8)
        for s in np.unique(sg):
            i0 = st[sg == s].min()
            ax.axvline(i0 / 1e3, color="gray", lw=0.6, alpha=0.6)
            ax.text(i0 / 1e3 + 4, 0.06, f"tol={tol[sg == s][0]:g}", rotation=90,
                    fontsize=7.5, color="gray")
        ax.set_ylabel("W$_{min}$")
        ax.set_title(name, fontsize=10, loc="left")
        ax.grid(alpha=0.25)
    fs = (hk["window"] == 8) & (hk["stage"] == 5)
    axes[0].annotate("final stage plateaus at W$_{min}$≈0.74 —\ncertificate unreachable; error already at\nthe inherited floor (L2RE 4.401e-2)",
                     xy=(680, 0.70), xytext=(430, 0.35), fontsize=9,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[1].annotate("controller stops at stage cap instead of\nextending a stalled stage; identical L2RE 4.401e-2",
                     xy=(612, 0.72), xytext=(330, 0.30), fontsize=9,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[1].set_xlabel("iteration within window 8  (×10³)")
    fig.suptitle("P1 experiment — KS window 8 replayed from the identical handoff state", y=0.985)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_p1_adaptive_vs_fixed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig7_cliff():
    hk = np.load(f"{KS_C}/causal/history_jax.npz", allow_pickle=True)
    hg = np.load(f"{GS_C}/causal/history_jax.npz", allow_pickle=True)

    def cliff(d, n):
        out = []
        for w in range(n):
            m = d["window"] == w
            fs = d["stage"][m].max()
            out.append(d["w_min"][m & (d["stage"] == fs)].max())
        return np.array(out)

    ck, cg = cliff(hk, 10), cliff(hg, 20)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot((np.arange(10) + 0.5) / 10, ck, "o-", color="tab:red", label="KS")
    ax.plot((np.arange(20) + 0.5) / 20, cg, "s-", color="tab:blue", label="GS")
    ax.axhline(0.99, color="k", ls="--", lw=0.9)
    ax.text(0.62, 0.995, "certificate  W$_{min}$ = 0.99", fontsize=9)
    ax.axvspan(0.4, 1.0, color="tab:red", alpha=0.06)
    ax.text(0.62, 0.62, "KS certification cliff:\nw4 onward the final-stage certificate\nis unreachable — the same windows\nthat sit on the inherited-error floor",
            fontsize=9, color="tab:red")
    ax.set_xlabel("position in time domain (window midpoint, normalized)")
    ax.set_ylabel("best W$_{min}$ in final stage (tol = 100)")
    ax.set_ylim(0.55, 1.02)
    ax.set_title("Certification reachability per window — an oracle-free chaos-depth meter")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/fig7_certification_cliff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig0_ks()
    fig0_gs()
    fig5_cost()
    fig6_p1()
    fig7_cliff()
    print("done")
