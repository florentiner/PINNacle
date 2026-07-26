#!/usr/bin/env python
"""
Evaluate one optimizer chain on one PINNacle PDE over N seeds.

Orchestrator (default mode): runs one worker subprocess per seed (parallel
across usable GPUs), and after EVERY finished seed merges the results into
{hf_dir}/{pde_name}.csv on the Hugging Face dataset repo, so partial progress
survives killed Kaggle sessions. Seeds already present in the remote CSV are
skipped unless --force is given.

Examples:
    # full run, 10 seeds, default Adam->LBFGS chain, upload to HF
    python experiments/chain_eval/run_chain_pde.py --pde-name burgers_1d

    # quick smoke test (2 seeds, 3 epochs per stage)
    python experiments/chain_eval/run_chain_pde.py --pde-name burgers_1d \
        --n-seeds 2 --test-epochs 3

The LBFGS stage uses torch.optim.LBFGS with max_iter=1 per training step, so
"epochs" in the chain config counts true L-BFGS iterations. Optional stage
keys: history_size (default 100), max_iter (default 1).
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

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DEFAULT_CHAIN_JSON = os.path.join(SCRIPT_DIR, "chain_adam_lbfgs.json")
DEFAULT_HF_REPO = "danil-e/pinnacle-optuna-db"
DEFAULT_HF_DIR = "csv_chain"


def _enter_repo_root():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    os.chdir(REPO_ROOT)


def load_chain(chain_json: str, test_epochs: int | None):
    with open(chain_json) as f:
        chain = json.load(f)
    if test_epochs is not None:
        chain = [dict(stage, epochs=int(test_epochs)) for stage in chain]
    return chain


def load_chains_by_seed(chains_json: str, seeds: list[int], test_epochs: int | None):
    """{seed: chain} from a per-seed chain file (keys are seed numbers as strings)."""
    with open(chains_json) as f:
        raw = json.load(f)
    out = {}
    for seed in seeds:
        if str(seed) not in raw:
            sys.exit(f"{chains_json}: no chain for seed {seed} (has {sorted(raw)})")
        chain = raw[str(seed)]
        if test_epochs is not None:
            chain = [dict(stage, epochs=int(test_epochs)) for stage in chain]
        out[seed] = chain
    return out


# ---------------------------------------------------------------------------
# Worker: one seed, executed in a subprocess
# ---------------------------------------------------------------------------

def build_stage_optimizer(stage: dict, params, lbfgs_max_iter: int = 1):
    import torch

    name = (stage.get("optimizer") or stage.get("type") or "").lower()
    lr = float(stage["lr"])
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name in ("lbfgs", "l-bfgs", "l_bfgs"):
        return torch.optim.LBFGS(
            params,
            lr=lr,
            max_iter=int(stage.get("max_iter", lbfgs_max_iter)),
            history_size=int(stage.get("history_size", 100)),
            line_search_fn="strong_wolfe",
        )
    if name == "pso":
        from deepxde.optimizers.config import set_PSO_options

        set_PSO_options(lr=lr)
        return "PSO"
    raise ValueError(f"Unknown optimizer '{stage}'. Expected Adam / LBFGS / PSO.")


def run_worker(args) -> int:
    _enter_repo_root()

    import torch

    if (
        not torch.cuda.is_available()
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        and hasattr(torch, "set_default_device")
    ):
        torch.set_default_device("mps")

    import numpy as np
    import deepxde as dde

    dde.config.set_default_float("float32")
    torch.set_default_dtype(torch.float32)

    from experiments.chain_eval.pde_registry import build_get_model
    from src.utils.callbacks import TesterCallback

    chain = load_chain(args.chain_json, args.test_epochs)
    seed = int(args.seed)

    dde.config.set_random_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, loss_weights = build_get_model(args.pde_name, args.hidden_layers)()

    def reinit(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    model.net.apply(reinit)

    save_path = os.path.join(args.save_dir, f"{args.pde_name}_seed_{seed}")
    os.makedirs(save_path, exist_ok=True)

    rmse = brmse = l2re = bc_l2re = float("inf")
    stages = []
    t0 = time.time()

    for stage_idx, stage in enumerate(chain):
        opt_name = stage.get("optimizer") or stage.get("type")
        epochs = int(stage["epochs"])
        print(f"\n{'=' * 70}")
        print(f"[seed {seed}] Stage {stage_idx}: {opt_name} | lr={stage['lr']} | epochs={epochs}")
        print(f"{'=' * 70}\n", flush=True)

        opt = build_stage_optimizer(stage, model.net.parameters(), args.lbfgs_max_iter)
        model.compile(opt, loss_weights=loss_weights)
        model.optimizer = opt

        tester = TesterCallback(log_every=args.display_every)
        model.train(
            iterations=epochs,
            display_every=args.display_every,
            callbacks=[tester],
            model_save_path=save_path,
            save_model=False,
        )

        rmse = float(getattr(tester, "rmse", float("inf")))
        brmse = float(getattr(tester, "brmse", float("inf")))
        l2re = float(getattr(tester, "l2re", float("inf")))
        bc_l2re = float(getattr(tester, "bc_l2re", float("inf")))
        stages.append(
            {"stage": stage_idx, "optimizer": opt_name, "epochs": epochs,
             "rmse": rmse, "brmse": brmse, "l2re": l2re, "bc_l2re": bc_l2re}
        )
        print(f"After stage {stage_idx} ({opt_name}): RMSE={rmse:.4e} BRMSE={brmse:.4e} "
              f"L2RE={l2re:.4e} BC_L2RE={bc_l2re:.4e}", flush=True)

        if not np.isfinite(rmse):
            print("NaN/Inf detected — stopping chain early.")
            break

    elapsed = time.time() - t0
    mse = rmse ** 2 if math.isfinite(rmse) else float("inf")

    result = {
        "pde_name": args.pde_name,
        "seed": seed,
        "chain": chain,
        "mse": mse,
        "rmse": rmse,
        "brmse": brmse,
        "l2re": l2re,
        "bc_l2re": bc_l2re,
        "stages": stages,
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.result_json)), exist_ok=True)
    with open(args.result_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[seed {seed}] done: MSE={mse:.4e} L2RE={l2re:.4e} ({elapsed:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# Orchestrator: N seeds, parallel over devices, HF upload after each seed
# ---------------------------------------------------------------------------

def probe_devices(spec: str) -> list[str]:
    """Return worker device assignments: CUDA indices (as strings), 'mps' or 'cpu'."""
    import torch

    if spec == "cpu":
        return ["cpu"]
    if spec != "auto":
        return [d.strip() for d in spec.split(",") if d.strip() != ""]

    usable = []
    for i in range(torch.cuda.device_count()):
        try:
            (torch.zeros(4, device=f"cuda:{i}") + 1).sum().item()
            usable.append(str(i))
        except Exception as e:
            name = torch.cuda.get_device_name(i)
            print(f"Skipping GPU {i} ({name}): {e}")
    if usable:
        return usable
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return ["mps"]
    print("No usable GPU — running on CPU.")
    return ["cpu"]


def result_to_row(r: dict, chain: list, chain_key: str, smoke: bool, value_type: str = "chain",
                  schema: str = "chain", pde_name_out: str | None = None) -> dict:
    mse_op = r.get("mse", float("nan"))
    brmse = r.get("brmse", float("nan"))
    l2re_op = r.get("l2re", float("nan"))
    l2re_bnd = r.get("bc_l2re", float("nan"))
    mse_bnd = brmse ** 2 if isinstance(brmse, (int, float)) and math.isfinite(brmse) else float("nan")
    mse_tot = (
        mse_op + mse_bnd
        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (mse_op, mse_bnd))
        else float("nan")
    )
    l2re_tot = (
        l2re_op + l2re_bnd
        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (l2re_op, l2re_bnd))
        else float("nan")
    )
    if schema == "random":
        # csv_random layout: no run metadata, plus l2re = hypot(l2re_op, l2re_bnd).
        return {
            "pde_name": pde_name_out or r["pde_name"],
            "seed": int(r["seed"]),
            "mse_op": mse_op,
            "mse_bnd": mse_bnd,
            "mse_total": mse_tot,
            "l2re_op": l2re_op,
            "l2re_bnd": l2re_bnd,
            "l2re": (
                math.hypot(l2re_op, l2re_bnd)
                if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (l2re_op, l2re_bnd))
                else float("nan")
            ),
            "l2re_total": l2re_tot,
            "chain_json": json.dumps(chain),
        }
    return {
        "run_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "pde_name": r["pde_name"],
        "value_type": value_type,
        "smoke_test": smoke,
        "chain_key": chain_key,
        "seed": int(r["seed"]),
        "mse_op": mse_op,
        "mse_bnd": mse_bnd,
        "mse_total": mse_tot,
        "l2re_op": l2re_op,
        "l2re_bnd": l2re_bnd,
        "l2re_total": l2re_tot,
        "elapsed_s": r.get("elapsed_s"),
        "chain_json": json.dumps(chain),
    }


def run_orchestrator(args) -> int:
    _enter_repo_root()
    from experiments.chain_eval import hf_results

    chain = load_chain(args.chain_json, args.test_epochs) if not args.chains_json else None
    smoke = args.test_epochs is not None
    chain_key = args.chain_key or os.path.splitext(
        os.path.basename(args.chains_json or args.chain_json)
    )[0]
    csv_name = args.csv_name or args.pde_name
    # csv_random keys rows by the dataset's own PDE spelling (heat_2d_longtime).
    pde_name_out = csv_name if args.schema == "random" else args.pde_name
    write_token = args.hf_token_write or os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    read_token = args.hf_token_read or os.environ.get("HF_TOKEN_READ") or write_token

    if args.seeds:
        seeds = [int(s) for s in args.seeds.replace(" ", "").split(",") if s != ""]
    else:
        seeds = [args.seed_base + i for i in range(args.n_seeds)]

    if args.upload and not args.force:
        remote = hf_results.download_csv(
            args.hf_repo, hf_results.csv_path_in_repo(args.hf_dir, csv_name), read_token
        )
        done = hf_results.existing_seeds(remote, pde_name_out, chain_key, smoke, args.schema)
        skipped = [s for s in seeds if s in done]
        seeds = [s for s in seeds if s not in done]
        if skipped:
            print(f"Skipping seeds already on HF for {pde_name_out}/{chain_key}: {skipped}")
    if not seeds:
        print("Nothing to do — all seeds already recorded.")
        return 0

    # Per-seed chains (csv_random) or one shared chain for every seed.
    if args.chains_json:
        chains_by_seed = load_chains_by_seed(args.chains_json, seeds, args.test_epochs)
        chain_dir = os.path.join(args.save_dir, "chains")
        os.makedirs(chain_dir, exist_ok=True)
        chain_paths = {}
        for seed, seed_chain in chains_by_seed.items():
            p = os.path.join(chain_dir, f"seed_{seed}.json")
            with open(p, "w") as f:
                json.dump(seed_chain, f)
            chain_paths[seed] = p
    else:
        chains_by_seed = {s: chain for s in seeds}
        chain_paths = {s: args.chain_json for s in seeds}

    devices = probe_devices(args.devices)
    if args.n_parallel:
        n_parallel = args.n_parallel
    elif devices[0] in ("cpu", "mps"):
        n_parallel = 1
    else:
        n_parallel = len(devices) * max(1, args.workers_per_gpu)
    print(f"PDE={args.pde_name} chain_key={chain_key} smoke={smoke}")
    print(f"Seeds to run: {seeds}")
    print(f"Devices: {devices} | parallel workers: {n_parallel}", flush=True)

    results_dir = os.path.join(args.save_dir, "results_json")
    os.makedirs(results_dir, exist_ok=True)

    def launch(slot: int, seed: int):
        result_json = os.path.join(results_dir, f"{args.pde_name}_seed_{seed}.json")
        env = os.environ.copy()
        env["DDEBACKEND"] = "pytorch"
        dev = devices[slot % len(devices)]
        if dev == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif dev == "mps":
            env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        else:
            env["CUDA_VISIBLE_DEVICES"] = dev
        cmd = [
            sys.executable, os.path.abspath(__file__), "--worker",
            "--pde-name", args.pde_name,
            "--seed", str(seed),
            "--chain-json", chain_paths[seed],
            "--result-json", result_json,
            "--display-every", str(args.display_every),
            "--hidden-layers", args.hidden_layers,
            "--save-dir", args.save_dir,
            "--lbfgs-max-iter", str(args.lbfgs_max_iter),
        ]
        if args.test_epochs is not None:
            cmd += ["--test-epochs", str(args.test_epochs)]
        p = subprocess.Popen(cmd, env=env)
        print(f"  started seed {seed} on device '{dev}' (pid {p.pid})", flush=True)
        return p, seed, result_json

    completed_rows: list[dict] = []
    failed_seeds: list[int] = []

    def collect(seed: int, result_json: str, returncode: int):
        if returncode != 0 or not os.path.exists(result_json):
            print(f"WARNING: seed {seed} failed (exit {returncode}); no result recorded.")
            failed_seeds.append(seed)
            return
        with open(result_json) as f:
            r = json.load(f)
        completed_rows.append(result_to_row(
            r, chains_by_seed[seed], chain_key, smoke, args.value_type, args.schema, pde_name_out
        ))
        if args.upload:
            hf_results.upload_rows(
                args.hf_repo, args.hf_dir, csv_name, completed_rows,
                write_token, read_token, local_dir=os.path.join(args.save_dir, "csv"),
                schema=args.schema,
            )
        else:
            import pandas as pd

            local_dir = os.path.join(args.save_dir, "csv")
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, f"{csv_name}.csv")
            columns, _ = hf_results.schema_columns(args.schema)
            pd.DataFrame(completed_rows, columns=columns).to_csv(local_path, index=False)

    active: list[tuple] = []  # (proc, seed, result_json, slot)
    slots = list(range(n_parallel))
    queue = list(seeds)
    while queue or active:
        while queue and slots:
            slot = slots.pop(0)
            p, seed, rj = launch(slot, queue.pop(0))
            active.append((p, seed, rj, slot))
        still = []
        for p, seed, rj, slot in active:
            if p.poll() is None:
                still.append((p, seed, rj, slot))
            else:
                print(f"  seed {seed} finished (exit {p.returncode})", flush=True)
                collect(seed, rj, p.returncode)
                slots.append(slot)
        active = still
        if active:
            time.sleep(5)

    # final summary
    print(f"\n{'=' * 60}\nSummary: {args.pde_name} / {chain_key} ({len(completed_rows)} seeds ok, "
          f"{len(failed_seeds)} failed{': ' + str(failed_seeds) if failed_seeds else ''})")
    if completed_rows:
        import pandas as pd

        df = pd.DataFrame(completed_rows)
        for col in ["mse_op", "mse_bnd", "mse_total", "l2re_op", "l2re_bnd", "l2re_total"]:
            v = pd.to_numeric(df[col], errors="coerce").replace([float("inf"), float("-inf")], float("nan")).dropna()
            if len(v):
                print(f"  {col:12s}: mean={v.mean():.4e}  std={v.std():.4e}  min={v.min():.4e}  max={v.max():.4e}")
    return 0 if not failed_seeds else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pde-name", required=True)
    parser.add_argument("--chain-json", default=DEFAULT_CHAIN_JSON)
    parser.add_argument("--chains-json", default=None,
                        help="JSON {seed: chain} — a separate chain per seed (csv_random baseline)")
    parser.add_argument("--chain-key", default=None, help="Label for CSV rows (default: chain file stem)")
    parser.add_argument("--lbfgs-max-iter", type=int, default=1,
                        help="max_iter for LBFGS stages that don't set it (optuna_trainer used 10)")
    parser.add_argument("--schema", default="chain", choices=sorted(("chain", "random")),
                        help="CSV layout: 'chain' (csv_chain/csv_seed) or 'random' (csv_random)")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--seeds", default=None,
                        help="Explicit seed list like '3,6' (overrides --seed-base/--n-seeds)")
    parser.add_argument("--devices", default="auto", help="'auto', 'cpu', or CUDA ids like '0,1'")
    parser.add_argument("--n-parallel", type=int, default=None,
                        help="Total parallel workers (overrides --workers-per-gpu)")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Seed workers per GPU (PINNacle nets are small; 2 can raise T4 throughput)")
    parser.add_argument("--display-every", type=int, default=100)
    parser.add_argument("--hidden-layers", default="100*5")
    parser.add_argument("--save-dir", default="runs_chain_eval")
    parser.add_argument("--test-epochs", type=int, default=None,
                        help="Cap every stage to N epochs (marks rows smoke_test=True)")
    parser.add_argument("--force", action="store_true", help="Re-run seeds already present on HF")
    # HF
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-dir", default=DEFAULT_HF_DIR)
    parser.add_argument("--value-type", default="chain",
                        help="value_type column in CSV rows (e.g. chain / continuous / fixed)")
    parser.add_argument("--csv-name", default=None,
                        help="CSV file stem in hf_dir (default: pde name; e.g. burgers_1d_continuous)")
    parser.add_argument("--hf-token-write", default=None)
    parser.add_argument("--hf-token-read", default=None)
    parser.add_argument("--no-upload", dest="upload", action="store_false")
    # worker mode (internal)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if args.seed is None or args.result_json is None:
            parser.error("--worker requires --seed and --result-json")
        sys.exit(run_worker(args))
    sys.exit(run_orchestrator(args))


if __name__ == "__main__":
    main()
