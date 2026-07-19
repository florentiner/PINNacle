"""Verify a causal run directory contains the complete forensic artifact set."""
import json
import os
import sys

import numpy as np


def check(outdir, case):
    outdir = outdir.rstrip("/") + "/"
    ok = True

    def need(path, kind="file"):
        nonlocal ok
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            ok = False
            return False
        return True

    for f in ["run_config.json", "metrics.csv", "collocation.npz",
              "arrays/ref.npy", "arrays/grid_meta.json", "causal/history.npz"]:
        need(outdir + f)

    if need(outdir + "arrays/grid_meta.json"):
        meta = json.load(open(outdir + "arrays/grid_meta.json"))
        shape = tuple(meta["grid_array_shape"])
        n_win = meta["n_windows"]
        if need(outdir + "arrays/ref.npy"):
            ref = np.load(outdir + "arrays/ref.npy")
            if ref.shape != shape:
                print(f"  BAD SHAPE: ref.npy {ref.shape} != {shape}")
                ok = False
        done = []
        for k in range(n_win):
            if os.path.exists(outdir + f"arrays/pred_w{k}_final.npy"):
                done.append(k)
                for f in [f"arrays/err_w{k}_final.npy", f"arrays/resid_w{k}_final.npy",
                          f"trajectory/w{k}_final.pt",
                          f"causal/window_{k}_handoff_ic.npy"]:
                    need(outdir + f)
        print(f"  windows completed: {len(done)}/{n_win} {done}")
        if len(done) == n_win:
            for f in ["arrays/pred_stitched_final.npy", "arrays/err_stitched_final.npy",
                      "errors.txt", "model_output.txt"]:
                need(outdir + f)
            if os.path.exists(outdir + "arrays/err_stitched_final.npy"):
                err = np.load(outdir + "arrays/err_stitched_final.npy")
                nan_frac = float(np.isnan(err).mean())
                if nan_frac > 0:
                    print(f"  WARNING: stitched err has {nan_frac:.1%} NaNs")
                    ok = False

    if need(outdir + "causal/history.npz"):
        h = np.load(outdir + "causal/history.npz")
        n = len(h["step"]) if "step" in h.files else 0
        print(f"  history rows: {n}; keys: {list(h.files)}")
        if n == 0:
            ok = False

    if need(outdir + "metrics.csv"):
        n_rows = sum(1 for _ in open(outdir + "metrics.csv")) - 1
        print(f"  metrics.csv rows: {n_rows}")
        if n_rows <= 0:
            ok = False
    return ok


if __name__ == "__main__":
    outdir = sys.argv[1]
    case = sys.argv[2] if len(sys.argv) > 2 else "ks"
    sys.exit(0 if check(outdir, case) else 1)
