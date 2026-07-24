#!/usr/bin/env python
"""
Run the chain evaluation over a list of PDEs (each: N seeds + HF upload).

Thin sequential loop around run_chain_pde.py — one PDE at a time, seeds
parallelized over GPUs inside each PDE run. A failing PDE does not stop the
loop. Seeds already recorded on HF are skipped, so re-running after a killed
session resumes where it stopped.

    python experiments/chain_eval/run_all.py --pdes all
    python experiments/chain_eval/run_all.py --pdes burgers_1d,wave1d --test-epochs 3 --n-seeds 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RUNNER = os.path.join(SCRIPT_DIR, "run_chain_pde.py")

PASSTHROUGH = [
    "chain_json", "chain_key", "n_seeds", "seed_base", "devices", "n_parallel",
    "display_every", "hidden_layers", "save_dir", "test_epochs",
    "hf_repo", "hf_dir", "hf_token_write", "hf_token_read",
]


def main():
    sys.path.insert(0, REPO_ROOT)
    from experiments.chain_eval.pde_names import ALL_PDE_NAMES

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdes", default="all", help="'all' or comma-separated PDE names")
    parser.add_argument("--chain-json", default=None)
    parser.add_argument("--chain-key", default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--n-parallel", type=int, default=None)
    parser.add_argument("--display-every", type=int, default=None)
    parser.add_argument("--hidden-layers", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--test-epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--hf-dir", default=None)
    parser.add_argument("--hf-token-write", default=None)
    parser.add_argument("--hf-token-read", default=None)
    args = parser.parse_args()

    if args.pdes.strip().lower() == "all":
        pdes = list(ALL_PDE_NAMES)
    else:
        pdes = [p.strip() for p in args.pdes.split(",") if p.strip()]
        unknown = [p for p in pdes if p not in ALL_PDE_NAMES]
        if unknown:
            parser.error(f"Unknown PDEs {unknown}. Available: {', '.join(ALL_PDE_NAMES)}")

    statuses = {}
    for i, pde in enumerate(pdes, 1):
        cmd = [sys.executable, RUNNER, "--pde-name", pde]
        for name in PASSTHROUGH:
            val = getattr(args, name)
            if val is not None:
                cmd += [f"--{name.replace('_', '-')}", str(val)]
        if args.force:
            cmd.append("--force")
        if args.no_upload:
            cmd.append("--no-upload")

        print(f"\n{'#' * 70}\n# [{i}/{len(pdes)}] {pde}  ({time.strftime('%Y-%m-%d %H:%M:%S')})\n{'#' * 70}", flush=True)
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        statuses[pde] = rc
        if rc != 0:
            print(f"WARNING: {pde} exited with code {rc}; continuing.", flush=True)

    print(f"\n{'=' * 70}\nAll done. Status per PDE:")
    for pde, rc in statuses.items():
        print(f"  {'OK  ' if rc == 0 else 'FAIL'}  {pde}")
    sys.exit(0 if all(rc == 0 for rc in statuses.values()) else 1)


if __name__ == "__main__":
    main()
