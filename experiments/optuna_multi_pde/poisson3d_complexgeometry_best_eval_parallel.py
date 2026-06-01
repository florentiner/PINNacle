#!/usr/bin/env python3
"""
Load the best Optuna trial from poisson3d_complexgeometry.db (latest study by default),
run N evaluation trajectories with distinct RNG seeds, at most ``--parallel`` processes,
and write ``poisson3d_complexgeometry.csv`` in the same format as other PDE result CSVs.

Requires CUDA (GPU). Uses ``spawn`` multiprocessing for PyTorch safety.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["DDEBACKEND"] = "pytorch"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PINNACLE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))


def _resolve_latest_study_name(db_path: str) -> str:
    import optuna

    storage = optuna.storages.RDBStorage(url=f"sqlite:///{db_path}")
    summaries = optuna.study.get_all_study_summaries(storage)
    if not summaries:
        raise RuntimeError(f"No studies in {db_path}")
    latest = max(summaries, key=lambda s: s.datetime_start)
    return latest.study_name


def _build_get_model(hidden_layers: str, datapath: str):
    import argparse as ap

    import numpy as np
    import deepxde as dde

    from src.pde.poisson import Poisson3D_ComplexGeometry
    from src.utils.args import parse_hidden_layers

    def get_model():
        pde = Poisson3D_ComplexGeometry(datapath=datapath)
        layers = (
            [pde.input_dim]
            + parse_hidden_layers(ap.Namespace(hidden_layers=hidden_layers))
            + [pde.output_dim]
        )
        net = dde.nn.FNN(layers, "tanh", "Glorot normal")
        net = net.float()

        loss_weights = np.ones(pde.num_loss, dtype=np.float32)
        for i, c in enumerate(pde.loss_config):
            t = c.get("type", "")
            if t in ("boundary", "initial", "ic"):
                loss_weights[i] = 100.0
            elif t == "pde":
                loss_weights[i] = 1.0
            else:
                loss_weights[i] = 1.0

        model = pde.create_model(net)
        return model, loss_weights

    return get_model


def _eval_worker(payload):
    """Run one eval trajectory; must stay top-level for multiprocessing spawn."""
    (
        run_id,
        seed,
        chain_config,
        save_root,
        display_every,
        hidden_layers,
        datapath,
        trial_number,
    ) = payload
    sys.path.insert(0, _PINNACLE_ROOT)
    os.chdir(_PINNACLE_ROOT)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in worker process; refusing to run on CPU.")

    from optuna_trainer import train_chain

    get_model = _build_get_model(hidden_layers, datapath)
    save_path = os.path.join(save_root, f"eval_run_{run_id}")
    mse, rmse, brmse = train_chain(
        get_model,
        chain_config,
        display_every,
        save_path,
        None,
        trial_number,
        seed=seed,
    )
    return run_id, float(mse), float(rmse), float(brmse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-path",
        type=str,
        default="optuna_studies/poisson3d_complexgeometry.db",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name. If omitted, uses the study with the latest datetime_start.",
    )
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--datapath", type=str, default="ref/poisson_3d.dat")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out", type=str, default="runs_optuna")
    parser.add_argument("--name", type=str, default="poisson3d_complexgeometry_best_eval")
    parser.add_argument("--n-eval-runs", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=1234, help="Seeds are seed_base + run_id.")
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="Maximum concurrent worker processes (each uses the GPU).",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="poisson3d_complexgeometry.csv",
        help="Basename only; written under PINNacle repo root.",
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA_VISIBLE_DEVICES value.")
    args = parser.parse_args()

    sys.path.insert(0, _PINNACLE_ROOT)
    os.chdir(_PINNACLE_ROOT)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. This script is intended to run on GPU (set CUDA_VISIBLE_DEVICES / drivers)."
        )

    import optuna

    from optuna_trainer import (
        PINNACLE_ROOT,
        params_to_chain_config,
        save_best_params,
        write_results_csv,
    )

    db_path = args.db_path
    if not os.path.isabs(db_path):
        db_path = os.path.join(PINNACLE_ROOT, db_path)

    study_name = args.study_name or _resolve_latest_study_name(db_path)
    print(f"Using study: {study_name} (db: {db_path})")

    storage = optuna.storages.RDBStorage(url=f"sqlite:///{db_path}")
    study = optuna.load_study(study_name=study_name, storage=storage)

    optimizers = {
        "Adam": {"lr": [1e-2, 1e-3, 1e-4], "epochs": [100, 1000, 2500]},
        "LBFGS": {"lr": [1, 5e-1, 1e-1], "epochs": [100, 500, 1500]},
        "PSO": {"lr": [0.0, 1e-3, 1e-4], "epochs": [100, 200, 300]},
    }

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_base_path = os.path.join(PINNACLE_ROOT, args.out, f"{date_str}-{args.name}")
    os.makedirs(save_base_path, exist_ok=True)

    save_best_params(study, optimizers, save_base_path)
    best_trial = study.best_trial
    best_chain = params_to_chain_config(best_trial.params, optimizers)
    traj_json = json.dumps(best_chain, indent=None)

    best_mse = best_trial.user_attrs.get("last_mse")
    best_rmse = best_trial.user_attrs.get("last_rmse")
    best_brmse = best_trial.user_attrs.get("last_brmse")

    eval_save = os.path.join(save_base_path, "best_chain_eval")
    os.makedirs(eval_save, exist_ok=True)

    n_runs = args.n_eval_runs
    payloads = [
        (
            run_id,
            int(args.seed_base + run_id),
            best_chain,
            eval_save,
            args.log_every,
            args.hidden_layers,
            args.datapath,
            10_000 + run_id,
        )
        for run_id in range(n_runs)
    ]

    ctx = mp.get_context("spawn")
    eval_results_map: dict[int, tuple] = {}
    max_workers = max(1, min(args.parallel, n_runs))

    print(f"Starting {n_runs} eval runs with max_workers={max_workers} (GPU, CUDA_VISIBLE_DEVICES={args.device})")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_eval_worker, p): p[0] for p in payloads}
        for fut in as_completed(futures):
            run_id, mse, rmse, brmse = fut.result()
            eval_results_map[run_id] = (mse, rmse, brmse)
            print(f"Finished eval run_id={run_id} MSE={mse} RMSE={rmse} BRMSE={brmse}")

    eval_results = [eval_results_map[i] for i in range(n_runs)]

    rows = []
    rows.append(
        {
            "phase": "optuna_best",
            "run_id": best_trial.number,
            "mse": best_mse if best_mse is not None else "",
            "rmse": best_rmse if best_rmse is not None else "",
            "brmse": best_brmse if best_brmse is not None else "",
            "trajectory_json": traj_json,
        }
    )

    for k, (mse, rmse, brmse) in enumerate(eval_results):
        rows.append(
            {
                "phase": "eval",
                "run_id": k,
                "mse": mse,
                "rmse": rmse,
                "brmse": brmse,
                "trajectory_json": traj_json,
            }
        )

    if eval_results:
        mses = [r[0] for r in eval_results if np.isfinite(r[0])]
        brmses = [r[2] for r in eval_results if np.isfinite(r[2])]
        rows.append(
            {
                "phase": "eval_summary",
                "run_id": "mean",
                "mse": float(np.mean(mses)) if mses else "",
                "rmse": float(np.mean([r[1] for r in eval_results if np.isfinite(r[1])]))
                if eval_results
                else "",
                "brmse": float(np.mean(brmses)) if brmses else "",
                "trajectory_json": traj_json,
            }
        )

    out_csv = os.path.join(save_base_path, "results.csv")
    write_results_csv(out_csv, rows)

    project_csv = None
    if args.results_csv:
        if os.path.basename(args.results_csv) != args.results_csv:
            raise ValueError("results_csv must be a basename only")
        project_csv = os.path.join(PINNACLE_ROOT, args.results_csv)
        write_results_csv(project_csv, rows)

    print(f"Done. Wrote {out_csv}" + (f" and {project_csv}" if project_csv else ""))


if __name__ == "__main__":
    main()
