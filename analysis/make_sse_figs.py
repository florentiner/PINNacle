"""fig23: the complete experimental record of the arXiv:2303.03374 adaptation."""
import glob
import json

import numpy as onp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = json.load(open(glob.glob("runs/kaggle-trivial-heat-sse/**/sse_results.json", recursive=True)[0]))
K = json.load(open(glob.glob("runs/kaggle-trivial-ks-sse/**/sse_results.json", recursive=True)[0]))
AM = json.load(open("runs/kaggle-trivial-heat-march/sse_after_march/sse_results.json"))


def per_kick(res, key):
    out = {}
    for c in res["children"]:
        out.setdefault(c["kick"], []).append(c[key])
    ks = sorted(out)
    return ks, [out[k][0] for k in ks]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    ax = axes[0, 0]
    for res, name, color in [(H, "Heat-LT", "tab:red"), (K, "KS", "tab:orange")]:
        ks, b = per_kick(res, "barrier_to_trivial")
        ax.plot(ks, [max(x, 1e-4) for x in b], "o-", color=color, lw=1.8, label=name)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.axhline(1e-1, color="gray", ls=":", lw=1)
    ax.text(1.05, 0.13, "basin boundary scale", fontsize=8, color="gray")
    ax.set_xlabel("kick multiplier (× base LR)")
    ax.set_ylabel("barrier to trivial ckpt (loss units)")
    ax.set_title("(a) Escape is kick-controlled: barriers grow monotonically\n(the paper's stay-vs-leave picture, quantitative)")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    ks, l = per_kick(H, "loss")
    ax.plot(ks, l, "o-", color="tab:red", lw=1.8, label="child final loss (heat)")
    ax.axhline(H["trivial"]["loss"], color="k", ls="--", lw=1.2,
               label=f'trivial ckpt loss ({H["trivial"]["loss"]:.1e})')
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.annotate("kick ×8: LEFT the basin into a\nSTRICTLY DEEPER minimum (5.9e-3)\n— with identical error 0.999",
                (8, 5.88e-3), textcoords="offset points", xytext=(-175, 38), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("kick multiplier"); ax.set_ylabel("vanilla loss")
    ax.set_title("(b) Where children land (heat): equivalent or DEEPER\nwrong minima — the field of ghosts, in numbers")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    for res, name, color, van in [(H, "Heat-LT", "tab:red", 0.9993), (K, "KS", "tab:orange", 1.0068)]:
        ks, e = per_kick(res, "l2re")
        ax.plot(ks, e, "o-", color=color, lw=1.8, label=f"{name} children")
        ax.axhline(van, color=color, ls=":", lw=1)
    mk, me = per_kick(AM, "l2re")
    ax.plot(mk, me, "s--", color="tab:green", lw=1.6, label="children around the MARCH ckpt")
    ax.axhline(AM["trivial"]["l2re"], color="tab:green", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("kick multiplier"); ax.set_ylabel("child L2RE (true error)")
    ax.set_title("(c) The error never recovers: escaped children stay wrong\n(dotted lines = their parent checkpoints)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    VAN = "runs/kaggle-trivial-vanilla/08.10-15.20.22-trivial-heatlt-vanilla/0-0/arrays"
    ref = onp.load(VAN + "/ref.npy")[..., 0]
    ts = onp.linspace(0, 100, ref.shape[2])
    ax.plot(ts, onp.abs(ref).max(axis=(0, 1)), "k-", lw=1.6, label="reference")
    cdirs = sorted(glob.glob("runs/kaggle-trivial-heat-sse/**/child_k*_s0", recursive=True))
    colors = plt.cm.viridis(onp.linspace(0.15, 0.95, len(cdirs)))
    for cd, col in zip(cdirs, colors):
        p = onp.load(cd + "/pred_final.npy")[..., 0]
        kick = cd.split("_k")[-1].split("_")[0]
        ax.plot(ts, onp.abs(p).max(axis=(0, 1)), color=col, lw=1.3, label=f"child ×{kick}")
    for lvl in [onp.sqrt(onp.pi), onp.sqrt(2 * onp.pi)]:
        ax.axhline(lvl, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("t"); ax.set_ylabel("max |u|")
    ax.set_title("(d) What the children's fields actually look like (heat):\nevery kick lands in the same dead class — none wakes the dynamics")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.suptitle("arXiv:2303.03374 machinery on the trivial problem — the complete experimental record "
                 "(10 children × 2 PDEs + 4 around the march checkpoint)", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig("analysis/report_figs/fig23_sse_experiment_data.png", dpi=150, bbox_inches="tight")
    print("fig23 saved")


if __name__ == "__main__":
    main()
