"""Figures for REPORT_TRIVIAL.md — built from archived runs only."""
import csv
import glob
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "analysis")
from trivial_detector import signals  # noqa: E402

OUT = "analysis/report_figs"
VAN = "runs/kaggle-trivial-vanilla/08.10-15.20.22-trivial-heatlt-vanilla/0-0"


def last(path_glob):
    return sorted(glob.glob(path_glob), key=lambda p: int(p.split("_")[-1][:-4]))[-1]


def fig9_collapse_vs_escape():
    ref = np.load(VAN + "/arrays/ref.npy")[..., 0]
    van = np.load(last(VAN + "/arrays/pred_*.npy"))[..., 0]
    ts = np.linspace(0, 100, ref.shape[2])
    # causal plain (C-triv): stitch w0..w1 preds
    ct = "runs/kaggle-trivial-causal-triv/out_heat_trivial/arrays"
    cw0 = np.load(ct + "/pred_w0_final.npy")[..., 0]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(ts, np.abs(ref).max(axis=(0, 1)), color="k", lw=1.8, label="reference (forced, alive forever)")
    ax.plot(ts, np.abs(van).max(axis=(0, 1)), color="tab:red", lw=1.8,
            label="vanilla PINN — collapses to the trivial branch")
    ax.plot(ts[:cw0.shape[2]], np.abs(cw0).max(axis=(0, 1)), color="tab:green", lw=2.2,
            label="causal PINN started AT the trivial solution (window 0)")
    ax.axhline(np.sqrt(np.pi), color="gray", ls="--", lw=1)
    ax.text(70, np.sqrt(np.pi) + 0.05, r"$\sqrt{\pi}$ — second self-gating level of $\sin(u^2)$",
            fontsize=9, color="gray")
    ax.axhline(0, color="gray", lw=0.6)
    ax.text(70, 0.05, "trivial level $u\\equiv 0$", fontsize=9, color="gray")
    ax.set_xlabel("t")
    ax.set_ylabel("max |u|")
    ax.set_xlim(0, 100)
    ax.set_title("Heat2D-LT: the two self-gating levels — vanilla falls to level 0; "
                 "the causal gate climbs out even when STARTED at the trivial solution")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/fig9_collapse_vs_escape.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig10_cheap_sliver():
    resid = np.load(last(VAN + "/arrays/resid_*.npy"))[..., 0]
    ts = np.linspace(0, 100, resid.shape[2])
    rt = np.sqrt((resid ** 2).mean(axis=(0, 1)))
    import pandas as pd
    m = pd.read_csv(VAN + "/metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    axes[0].semilogy(ts, rt + 1e-12, color="tab:red", lw=1.6)
    axes[0].axvspan(0, 1, color="tab:red", alpha=0.15)
    axes[0].text(2, rt.max() * 0.5, "91.6% of total squared residual\nlives in t<=1 (1% of the domain)",
                 fontsize=9)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("residual RMS")
    axes[0].set_title("The cheap sliver: vanilla pays its residual\nonly in a thin causality-violating layer")
    axes[0].grid(alpha=0.3)
    ax = axes[1]
    ax.semilogy(m["step"], m["loss_train_0"] + 1e-12, label="PDE residual loss", lw=1.5)
    ax.semilogy(m["step"], m["loss_train_1"] + 1e-12, label="IC loss", lw=1.5)
    ax.semilogy(m["step"], m["loss_train_2"] + 1e-12, label="BC loss", lw=1.5)
    ax2 = ax.twinx()
    ax2.plot(m["step"], m["l2re"], color="k", lw=2, label="true error (L2RE)")
    ax2.set_ylim(0, 1.1); ax2.set_ylabel("L2RE (true error)")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss components")
    ax.set_title("A ghost with a certificate of convergence:\nevery loss component -> ~0 while the true error stays 1.0")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig10_cheap_sliver.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig11_detector_map():
    pts = []

    def add(label, arrays_dir, color, marker):
        pred = np.load(last(arrays_dir + "/pred_*.npy"))
        resid = np.load(last(arrays_dir + "/resid_*.npy"))
        s = signals(pred, resid)
        pts.append((label, s["C_enrich"], s["A_late_amp_ratio"], color, marker))

    add("heat vanilla", VAN + "/arrays", "tab:red", "o")
    add("heat rar", glob.glob("runs/kaggle-trivial-heat-rar/*/0-0/arrays")[0], "tab:red", "s")
    add("heat uni", glob.glob("runs/kaggle-trivial-heat-uni/*/0-0/arrays")[0], "tab:red", "^")
    add("KS vanilla", "runs/07.18-13.19.39-baseline-chaotic/0-0/arrays", "tab:orange", "o")
    add("KS rar (goodharted)", glob.glob("runs/kaggle-trivial-ks-rar/*/0-0/arrays")[0], "tab:orange", "s")
    add("KS uni", glob.glob("runs/kaggle-trivial-ks-uni/*/0-0/arrays")[0], "tab:orange", "^")
    add("GS vanilla", "runs/07.18-13.19.39-baseline-chaotic/1-0/arrays", "tab:purple", "o")
    # KS causal control (stitched, from the July final)
    A = "runs/kaggle-causal-ks-session9/causal_ks/0-0/arrays"
    rs = [np.load(A + f"/resid_w{k}_final.npy").transpose(1, 0, 2) for k in range(10)]
    resid = np.concatenate([r[:, :-1] if k < 9 else r for k, r in enumerate(rs)], axis=1)
    s = signals(np.load(A + "/pred_stitched_final.npy"), resid)
    pts.append(("KS causal (healthy)", s["C_enrich"], s["A_late_amp_ratio"], "tab:green", "*"))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.axvspan(3, 100, color="tab:red", alpha=0.06)
    ax.axhspan(0.001, 0.1, color="tab:blue", alpha=0.08)
    ax.text(3.3, 0.25, "front-collapse zone\n(C_enrich > 3)", fontsize=8, color="tab:red")
    ax.text(0.013, 0.012, "dead-dynamics zone (A < 0.1)", fontsize=8, color="tab:blue")
    offsets = {"heat vanilla": (7, 14), "heat rar": (7, 2), "heat uni": (7, -10),
               "KS vanilla": (-8, 12), "KS uni": (7, -12)}
    for label, c, a, color, marker in pts:
        ax.scatter([max(c, 1e-2)], [max(a, 1e-3)], s=140 if marker == "*" else 60,
                   color=color, marker=marker, zorder=5)
        ax.annotate(label, (max(c, 1e-2), max(a, 1e-3)), textcoords="offset points",
                    xytext=offsets.get(label, (7, 4)), fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("C_enrich — early-time residual concentration (1 = uniform)")
    ax.set_ylabel("A_late — late-time dynamics alive (variance ratio)")
    ax.set_title("Reference-free trivial-branch detector map\n"
                 "(note KS-rar: the intervention moved the run OUT of the flagged zones without fixing it)")
    ax.grid(alpha=0.3, which="both")
    fig.savefig(f"{OUT}/fig11_detector_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig12_gate_escape():
    h = np.load("runs/kaggle-trivial-causal-triv/out_heat_trivial/causal/history_jax.npz",
                allow_pickle=True)
    m = h["window"] == 0
    st, lic, l2 = h["step"][m], h["loss_ic"][m], h["l2_window"][m]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.semilogy(st / 1e3, lic, color="tab:blue", lw=1.8, label="IC loss (the gate): 0.243 → 1.8e-7")
    ax.set_xlabel("iteration (×10³), window 0, STARTED at the trivial solution")
    ax.set_ylabel("IC loss (log)")
    ax2 = ax.twinx()
    ax2.plot(st / 1e3, l2, color="tab:green", lw=1.8, label="window L2 error: 1.00 → 0.85")
    ax2.set_ylabel("window L2 rel. error")
    ax2.set_ylim(0, 1.1)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2l = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2l, loc="center right", fontsize=9)
    ax.set_title("Escape from inside the trivial attractor (causal objective, plain encoding):\n"
                 "the 10⁴·L_IC gate forces the state off u≡0 — identical to the random-init run")
    ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/fig12_gate_escape.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig9_collapse_vs_escape()
    fig10_cheap_sliver()
    fig11_detector_map()
    fig12_gate_escape()
    print("done")
