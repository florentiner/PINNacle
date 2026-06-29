#!/usr/bin/env python3
"""
Evaluate best optimizer chains for wave1d (fixed + continuous).

Downloads chains from HF Hub, runs N_SEEDS evaluations in parallel using
chain_eval_worker.py subprocesses, then appends results to per-type CSVs
on HF Hub.

Usage:
    python experiments/optuna_multi_pde/run_wave1d.py

Set HF_TOKEN_WRITE env var (or --hf-token) to upload results to HF Hub.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime

os.environ["DDEBACKEND"] = "pytorch"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_PINNACLE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# ── CONFIG ────────────────────────────────────────────────────────────────────
PDE_NAME      = "wave1d"
VALUE_TYPES   = ["fixed"]   # continuous already evaluated; pass --value-types continuous to rerun it

N_PROCESSES   = 2    # parallel workers
N_SEEDS       = 10   # seeds per value_type
SEED_BASE     = 42   # seeds = SEED_BASE .. SEED_BASE+N_SEEDS-1
HIDDEN_LAYERS = "100*5"
DISPLAY_EVERY = 100

HF_REPO_ID  = "danil-e/pinnacle-optuna-db"
HF_CSV_DIR  = "csv_seed"

# None = use chain epochs as-is; int = cap all stages (for fast testing)
TEST_EPOCHS: int | None = None
# ─────────────────────────────────────────────────────────────────────────────

WORKER_SCRIPT = os.path.join(_SCRIPT_DIR, "chain_eval_worker.py")
PYTHON_BIN    = sys.executable


def _download_hf(repo_id: str, filename: str, local_dir: str, token=None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset",
        token=token, local_dir=local_dir, force_download=True,
    )


def _load_chain(chains_path: str, key: str) -> list[dict]:
    with open(chains_path) as f:
        all_chains = json.load(f)
    if key not in all_chains:
        avail = sorted(all_chains.keys())
        raise KeyError(f"Key '{key}' not in chains JSON.\nAvailable: {avail}")
    return all_chains[key]


def _run_seeds(chain: list[dict], value_type: str, results_dir: str, hf_write_token) -> list[dict]:
    os.makedirs(results_dir, exist_ok=True)

    chain_file = os.path.join(results_dir, "chain.json")
    with open(chain_file, "w") as f:
        json.dump(chain, f)

    seeds = list(range(SEED_BASE, SEED_BASE + N_SEEDS))

    try:
        import torch
        compatible_gpus = [
            i for i in range(torch.cuda.device_count())
            if (lambda p: p.major * 10 + p.minor >= 70)(torch.cuda.get_device_properties(i))
        ]
    except Exception:
        compatible_gpus = []

    print(f"\n{'='*60}")
    print(f"{PDE_NAME}_{value_type} — {len(seeds)} seeds, {N_PROCESSES} parallel")
    if compatible_gpus:
        print(f"GPUs: {compatible_gpus}")
    else:
        print("No compatible GPU — using CPU")
    print(f"{'='*60}")

    active: list[tuple[subprocess.Popen, int]] = []

    def _launch(slot: int, seed: int) -> subprocess.Popen:
        rjson = os.path.join(results_dir, f"result_seed_{seed}.json")
        env = os.environ.copy()
        env["DDEBACKEND"] = "pytorch"
        if compatible_gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(compatible_gpus[slot % len(compatible_gpus)])
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [
            PYTHON_BIN, WORKER_SCRIPT,
            "--pde-name",      PDE_NAME,
            "--chain-json",    chain_file,
            "--seed",          str(seed),
            "--result-json",   rjson,
            "--display-every", str(DISPLAY_EVERY),
            "--hidden-layers", HIDDEN_LAYERS,
        ]
        p = subprocess.Popen(cmd, env=env)
        print(f"  [seed {seed}] started (PID {p.pid})")
        return p

    slot = 0
    for seed in seeds:
        while len(active) >= N_PROCESSES:
            still = []
            for p, s in active:
                if p.poll() is not None:
                    print(f"  [seed {s}] finished (exit {p.returncode})")
                    slot += 1
                else:
                    still.append((p, s))
            active[:] = still
            if len(active) >= N_PROCESSES:
                time.sleep(5)
        active.append((_launch(slot, seed), seed))

    # drain remaining
    for p, s in active:
        p.wait()
        print(f"  [seed {s}] finished (exit {p.returncode})")

    # collect results
    rows = []
    key = f"{PDE_NAME}_{value_type}"
    for seed in seeds:
        rpath = os.path.join(results_dir, f"result_seed_{seed}.json")
        if not os.path.exists(rpath):
            print(f"WARNING: missing result for seed {seed}")
            continue
        with open(rpath) as f:
            r = json.load(f)

        mse_op  = r.get("mse", float("nan"))
        brmse   = r.get("brmse", float("nan"))
        mse_bnd = brmse ** 2 if math.isfinite(brmse) else float("nan")
        mse_tot = mse_op + mse_bnd if (math.isfinite(mse_op) and math.isfinite(mse_bnd)) else float("nan")
        l2re_op  = r.get("l2re", float("nan"))
        l2re_bnd = r.get("bc_l2re", float("nan"))
        l2re_tot = l2re_op + l2re_bnd if (math.isfinite(l2re_op) and math.isfinite(l2re_bnd)) else float("nan")

        rows.append({
            "run_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "pde_name":   PDE_NAME,
            "value_type": value_type,
            "smoke_test": False,
            "chain_key":  key,
            "seed":       seed,
            "mse_op":    mse_op,
            "mse_bnd":   mse_bnd,
            "mse_total": mse_tot,
            "l2re_op":    l2re_op,
            "l2re_bnd":   l2re_bnd,
            "l2re_total": l2re_tot,
            "elapsed_s":  r.get("elapsed_s"),
            "chain_json": json.dumps(chain),
        })

    return rows


def _append_and_upload(rows: list[dict], value_type: str, hf_write_token):
    import pandas as pd

    hf_csv_file = f"{HF_CSV_DIR}/{PDE_NAME}_{value_type}.csv"
    local_csv   = os.path.join("hf_cache", hf_csv_file)
    os.makedirs(os.path.dirname(local_csv), exist_ok=True)

    existing_df = None
    try:
        csv_local = _download_hf(HF_REPO_ID, hf_csv_file, "hf_cache")
        existing_df = pd.read_csv(csv_local)
        print(f"Existing CSV: {len(existing_df)} rows")
    except Exception as e:
        if "404" in str(e) or "EntryNotFound" in type(e).__name__:
            print("No existing CSV — will create new file.")
        else:
            print(f"WARNING: could not download CSV: {e}")

    new_df      = pd.DataFrame(rows)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None else new_df
    combined_df.to_csv(local_csv, index=False)
    print(f"\n{len(new_df)} new rows → {len(combined_df)} total rows in {hf_csv_file}")

    # summary
    for col in ["mse_op", "mse_bnd", "mse_total", "l2re_op", "l2re_bnd", "l2re_total"]:
        v = new_df[col].replace([float("inf"), float("-inf")], float("nan")).dropna()
        if len(v):
            print(f"  {col:12s}: mean={v.mean():.4e}  std={v.std():.4e}  "
                  f"min={v.min():.4e}  max={v.max():.4e}")

    if hf_write_token:
        from huggingface_hub import upload_file
        upload_file(
            path_or_fileobj=local_csv,
            path_in_repo=hf_csv_file,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=hf_write_token,
            commit_message=f"add {len(new_df)} {PDE_NAME}_{value_type} results",
        )
        print(f"Uploaded to https://huggingface.co/datasets/{HF_REPO_ID}")
    else:
        print(f"Saved locally: {local_csv}  (set HF_TOKEN_WRITE to upload)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-token",    default=None, help="HF write token (or set HF_TOKEN_WRITE env)")
    parser.add_argument("--n-seeds",     type=int, default=N_SEEDS)
    parser.add_argument("--n-processes", type=int, default=N_PROCESSES)
    parser.add_argument("--seed-base",   type=int, default=SEED_BASE)
    parser.add_argument("--test-epochs", type=int, default=None,
                        help="Cap all stage epochs (for fast testing)")
    parser.add_argument("--value-types", nargs="+", default=VALUE_TYPES,
                        choices=["fixed", "continuous"],
                        help="Which value types to run (default: both)")
    args = parser.parse_args()

    # apply CLI overrides to module-level config
    global N_SEEDS, N_PROCESSES, SEED_BASE, TEST_EPOCHS
    N_SEEDS      = args.n_seeds
    N_PROCESSES  = args.n_processes
    SEED_BASE    = args.seed_base
    TEST_EPOCHS  = args.test_epochs

    hf_write_token = args.hf_token or os.environ.get("HF_TOKEN_WRITE")

    sys.path.insert(0, _PINNACLE_ROOT)
    os.chdir(_PINNACLE_ROOT)

    os.makedirs("hf_cache", exist_ok=True)
    chains_local = _download_hf(HF_REPO_ID, "best_optimizer_chains.json", "hf_cache")
    print(f"Downloaded chains JSON from HF Hub")

    for value_type in args.value_types:
        key   = f"{PDE_NAME}_{value_type}"
        chain = _load_chain(chains_local, key)

        if TEST_EPOCHS is not None:
            chain = [dict(s, epochs=TEST_EPOCHS) for s in chain]
            print(f"TEST_EPOCHS={TEST_EPOCHS}: all stage epochs capped")

        print(f"\nChain for '{key}' ({len(chain)} stages):")
        for i, s in enumerate(chain):
            print(f"  Stage {i}: {s['optimizer']:5s}  lr={s['lr']:.4g}  epochs={s['epochs']}")

        results_dir = os.path.join("eval_results", key)
        rows = _run_seeds(chain, value_type, results_dir, hf_write_token)
        _append_and_upload(rows, value_type, hf_write_token)

    print("\nDone.")


if __name__ == "__main__":
    main()
