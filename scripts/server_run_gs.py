#!/usr/bin/env python3
"""Standalone server runner: causal Gray-Scott (JAX engine, plain encoding) +
artifacts. Not Kaggle-dependent. See server_run_ks.py for the pattern.

Examples:
  python scripts/server_run_gs.py --outdir runs/server-gs-seed2024 --seed 2024
  python scripts/server_run_gs.py --outdir runs/server-gs-seed2024   # auto-resumes
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
    p.add_argument("--outdir", required=True, help="e.g. runs/server-gs-seed2024")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--windows", type=int, default=20)
    p.add_argument("--iter-cap", type=int, default=100000)
    p.add_argument("--max-hours", type=float, default=1e9)
    p.add_argument("--param-snap-every", type=int, default=10000)
    p.add_argument("--skip-train", action="store_true")
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
        rc = sh([sys.executable, "-m", "causalpinn.jax_runner_gs",
                 "--outdir", out,
                 "--windows", str(args.windows),
                 "--iter-cap", str(args.iter_cap),
                 "--log-every", "1000",
                 "--seed", str(args.seed),
                 "--param-snap-every", str(args.param_snap_every),
                 "--max-hours", str(args.max_hours)], env=env)
        if rc != 0:
            sys.exit(rc)

    import glob
    done = len(glob.glob(os.path.join(out, "trajectory", "w*_final_params.npz")))
    print(f"[server-gs] windows complete: {done}/{args.windows}")
    if done == 0:
        print("[server-gs] nothing trained yet — rerun to continue; skipping artifacts")
        return
    sh([sys.executable, "-m", "causalpinn.jax_bridge",
        "--outdir", out, "--case", "gs", "--windows", str(args.windows),
        "--device", "cpu"], env=env)
    sh([sys.executable, "analysis/consolidate_trajectory.py", out], env=env)
    sh([sys.executable, "-m", "causalpinn.check_artifacts", out, "gs"], env=env)
    print(f"[server-gs] DONE — artifacts in {out}")


if __name__ == "__main__":
    main()
