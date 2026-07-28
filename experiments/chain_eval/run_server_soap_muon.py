#!/usr/bin/env python
"""
Server orchestrator for the SOAP/Muon baseline campaign (single-GPU V100 box).

One family per server:
    server 1:  python experiments/chain_eval/run_server_soap_muon.py --family muon
    server 2:  python experiments/chain_eval/run_server_soap_muon.py --family soap

For its family the server runs BOTH the standalone chain (X) and the Adam->X
chain on every PDE in SERVER_PDES, in two priority phases:

    phase 1: seeds 42-44  (3 seeds for every pde x config as fast as possible)
    phase 2: seeds 45-51  (top everything up to 10)

Already-recorded seeds are skipped via the HF resume check, so the script is
safe to rerun/restart at any point, and PDEs that already hold seeds 42-46
(burgers_1d, heat2d_longtime, ns2d_longtime) just get the missing ones.

Results: csv_{family}/{pde}.csv and csv_adam_{family}/{pde}.csv (schema with
the extra l2re = hypot(l2re_op, l2re_bnd) column), uploaded after EVERY seed.

Parallel workers per PDE are sized for one V100-32GB (no session cap on the
server, so we optimize throughput, not per-seed latency).

Requires: HF_TOKEN_WRITE in the environment (write token for the dataset).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# PDEs owned by the servers. The complementary set (KAGGLE_PDES) runs on the
# Kaggle fleet — keep the two sets disjoint so work is never computed twice.
KAGGLE_PDES = [
    "grayscott", "kuramoto_sivashinsky", "poissonnd", "heatnd",
    "poisson3d_complexgeometry", "ns2d_backstep",
]
SERVER_PDES = [
    # light first: quick wins & early signal
    "burgers_1d", "poisson2d_classic", "wave1d", "poissonboltzmann2d",
    "poisson2d_manyarea", "poissoninv", "heat2d_varyingcoef",
    "heat2d_multiscale", "heat2d_complexgeometry",
    # medium
    "wave2d_heterogeneous", "wave2d_longtime", "ns2d_classic",
    "burgers_2d", "heatinv", "heat2d_longtime",
    # heavy last
    "ns2d_longtime",
]

# V100-32GB parallel workers per PDE (memory is never the binding constraint
# for these nets; this trades per-seed speed for throughput).
N_PARALLEL = {
    "burgers_1d": 4, "poisson2d_classic": 4, "wave1d": 4,
    "poissonboltzmann2d": 4, "poisson2d_manyarea": 4, "poissoninv": 4,
    "heat2d_varyingcoef": 4, "heat2d_multiscale": 3, "heat2d_complexgeometry": 3,
    "wave2d_heterogeneous": 3, "wave2d_longtime": 3, "ns2d_classic": 3,
    "burgers_2d": 2, "heatinv": 3, "heat2d_longtime": 2,
    "ns2d_longtime": 2,
    # Kaggle set (in case a server is asked to cover for it)
    "grayscott": 2, "kuramoto_sivashinsky": 3, "poissonnd": 2, "heatnd": 2,
    "poisson3d_complexgeometry": 2, "ns2d_backstep": 3,
}

PHASE_SEEDS = {1: "42,43,44", 2: "45,46,47,48,49,50,51"}


def run_job(pde: str, family: str, chained: bool, seeds: str, display_every: int) -> int:
    chain_json = os.path.join(
        SCRIPT_DIR, f"chain_adam_{family}.json" if chained else f"chain_{family}.json"
    )
    hf_dir = f"csv_adam_{family}" if chained else f"csv_{family}"
    chain_key = f"adam_{family}" if chained else family
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "run_chain_pde.py"),
        "--pde-name", pde,
        "--chain-json", chain_json,
        "--seeds", seeds,
        "--n-parallel", str(N_PARALLEL.get(pde, 2)),
        "--display-every", str(display_every),
        "--save-dir", os.path.join(REPO_ROOT, "runs_chain_eval", hf_dir, pde),
        "--hf-dir", hf_dir,
        "--csv-name", pde,
        "--chain-key", chain_key,
        "--value-type", family,
        "--schema", "chain_l2re",
    ]
    print(f"\n{'#' * 72}\n# {pde} | {chain_key} | seeds {seeds} | "
          f"n_parallel={N_PARALLEL.get(pde, 2)}\n{'#' * 72}", flush=True)
    env = dict(os.environ, DDEBACKEND="pytorch")
    return subprocess.run(cmd, env=env, cwd=REPO_ROOT).returncode


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=("soap", "muon"))
    p.add_argument("--phase", default="all", choices=("1", "2", "all"))
    p.add_argument("--pdes", default=None,
                   help="Comma list overriding SERVER_PDES (e.g. to cover the Kaggle set)")
    p.add_argument("--display-every", type=int, default=100)
    args = p.parse_args()

    if not (os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")):
        sys.exit("Set HF_TOKEN_WRITE (results must reach HF after every seed).")

    pdes = [s for s in (args.pdes.split(",") if args.pdes else SERVER_PDES) if s]
    phases = [1, 2] if args.phase == "all" else [int(args.phase)]

    statuses = {}
    for phase in phases:
        seeds = PHASE_SEEDS[phase]
        print(f"\n{'=' * 72}\n= PHASE {phase}: seeds {seeds}\n{'=' * 72}", flush=True)
        for pde in pdes:
            for chained in (False, True):  # standalone first, then Adam->X
                key = f"p{phase}/{pde}/{'adam_' if chained else ''}{args.family}"
                statuses[key] = run_job(pde, args.family, chained, seeds, args.display_every)

    print("\n" + "=" * 72)
    for k, rc in statuses.items():
        print(f"{'OK  ' if rc == 0 else 'FAIL'}  {k}")
    sys.exit(0 if all(rc == 0 for rc in statuses.values()) else 1)


if __name__ == "__main__":
    main()
