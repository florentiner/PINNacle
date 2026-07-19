#!/usr/bin/env python3
"""Standalone server runner: causal KS (JAX engine) + artifacts. Not Kaggle-dependent.

Runs the full 10-window chaotic-KS causal training with checkpoint/resume in
OUTDIR, then bridges params into the torch pipeline to produce the complete
error-landscape artifact set, consolidates trajectories, and verifies artifacts.

Examples:
  python scripts/server_run_ks.py --outdir runs/server-ks-seed2024 --seed 2024
  python scripts/server_run_ks.py --outdir runs/server-ks-seed2024 --resume   # continue
Requires: jax[cuda] (training), torch + matplotlib + scipy (bridge/plots).
Run from the repo root. GPU selection: CUDA_VISIBLE_DEVICES=<idx>.
"""
import argparse
import os
import subprocess
import sys


def sh(cmd, env=None):
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=env)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True, help="e.g. runs/server-ks-seed2024")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--windows", type=int, default=10)
    p.add_argument("--iter-cap", type=int, default=200000)
    p.add_argument("--max-hours", type=float, default=1e9,
                   help="per-invocation budget; rerun with --resume to continue")
    p.add_argument("--param-snap-every", type=int, default=10000)
    p.add_argument("--resume", action="store_true",
                   help="informational; resume is automatic if outdir has jax_ckpt.pkl")
    p.add_argument("--skip-train", action="store_true", help="bridge/artifacts only")
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    out = os.path.join(args.outdir, "0-0") \
        if os.path.basename(args.outdir) != "0-0" else args.outdir
    os.makedirs(out, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    env.setdefault("DDEBACKEND", "pytorch")

    if not args.skip_train:
        rc = sh([sys.executable, "-m", "causalpinn.jax_runner",
                 "--outdir", out,
                 "--windows", str(args.windows),
                 "--iter-cap", str(args.iter_cap),
                 "--log-every", "1000",
                 "--seed", str(args.seed),
                 "--param-snap-every", str(args.param_snap_every),
                 "--max-hours", str(args.max_hours)], env=env)
        if rc != 0:
            print(f"[server-ks] training exited rc={rc} "
                  f"(0 after time-guard = resumable; rerun same command)")
            if rc != 0:
                sys.exit(rc)

    import glob
    done = len(glob.glob(os.path.join(out, "trajectory", "w*_final_params.npz")))
    print(f"[server-ks] windows complete: {done}/{args.windows}")
    if done == 0:
        print("[server-ks] nothing trained yet — rerun to continue; skipping artifacts")
        return
    sh([sys.executable, "-m", "causalpinn.jax_bridge",
        "--outdir", out, "--case", "ks", "--windows", str(args.windows),
        "--device", "cpu"], env=env)
    sh([sys.executable, "analysis/consolidate_trajectory.py", out], env=env)
    sh([sys.executable, "-m", "causalpinn.check_artifacts", out, "ks"], env=env)
    print(f"[server-ks] DONE — artifacts in {out} "
          f"(error landscapes: u_err.png, arrays/err_stitched_final.npy)")


if __name__ == "__main__":
    main()
