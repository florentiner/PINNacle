#!/usr/bin/env python3
"""
Load a completed continuous-chain Optuna study, save its best decoded chain, run
fixed-chain evaluations in parallel, and write a CSV matching the project result format.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["DDEBACKEND"] = "pytorch"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PINNACLE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))


OPTIMIZERS = {
    "Adam": {
        "lr": (1e-4, 1e-2),
        "lr_log": False,
        "epochs": (100, 2500),
    },
    "LBFGS": {
        "lr": (1e-1, 1.0),
        "lr_log": False,
        "epochs": (100, 1500),
    },
    "PSO": {
        "lr": (0.0, 1e-3),
        "lr_log": False,
        "epochs": (100, 300),
    },
}


PDE_DEFAULTS = {
    "burgers_1d": {
        "db_path": "optuna_studies/burgers_1d.db",
        "study_name": "burgers_1d_continues_24",
        "results_csv": "burgers_1d_continuous.csv",
        "name": "burgers_1d_continuous_best_eval",
        "hidden_layers": "100*5",
        "datapath": None,
    },
    "heatinv": {
        "db_path": "optuna_studies/heatinv.db",
        "study_name": "heatinv_chain_continues",
        "results_csv": "heatinv_continuous.csv",
        "name": "heatinv_continuous_best_eval",
        "hidden_layers": None,
        "datapath": None,
    },
    "ns2d_longtime": {
        "db_path": "optuna_studies/ns2d_longtime.db",
        "study_name": "ns2d_longtime_chain_continues_1",
        "results_csv": "ns2d_longtime_continuous.csv",
        "name": "ns2d_longtime_continuous_best_eval",
        "hidden_layers": "100*5",
        "datapath": "ref/ns_long.dat",
    },
    "kuramoto_sivashinsky": {
        "db_path": "optuna_studies/kuramoto_sivashinsky.db",
        "study_name": "kuramoto_sivashinsky_chain_continues",
        "results_csv": "kuramoto_sivashinsky_continuous.csv",
        "name": "kuramoto_sivashinsky_continuous_best_eval",
        "hidden_layers": "100*5",
        "datapath": None,
    },
}


def _detect_gpu_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def _build_get_model(pde_name: str, hidden_layers: str | None, datapath: str | None):
    import argparse as ap

    import deepxde as dde
    import numpy as np

    from src.utils.args import parse_hidden_layers

    if pde_name == "burgers_1d":
        from src.pde.burgers import Burgers1D

        def get_model():
            pde = Burgers1D()
            layers = (
                [pde.input_dim]
                + parse_hidden_layers(ap.Namespace(hidden_layers=hidden_layers or "100*5"))
                + [pde.output_dim]
            )
            net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()
            return pde.create_model(net), _loss_weights(pde, np)

        return get_model

    if pde_name == "heatinv":
        from src.pde.inverse import HeatInv

        def get_model():
            pde = HeatInv()
            net = pde.recommend_net.float()
            return pde.create_model(net), _loss_weights(pde, np)

        return get_model

    if pde_name == "ns2d_longtime":
        from src.pde.ns import NS2D_LongTime

        def get_model():
            pde = NS2D_LongTime(datapath=datapath or "ref/ns_long.dat")
            layers = (
                [pde.input_dim]
                + parse_hidden_layers(ap.Namespace(hidden_layers=hidden_layers or "100*5"))
                + [pde.output_dim]
            )
            net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()
            return pde.create_model(net), _loss_weights(pde, np)

        return get_model

    if pde_name == "kuramoto_sivashinsky":
        from src.pde.chaotic import KuramotoSivashinskyEquation

        def get_model():
            pde = KuramotoSivashinskyEquation()
            layers = (
                [pde.input_dim]
                + parse_hidden_layers(ap.Namespace(hidden_layers=hidden_layers or "100*5"))
                + [pde.output_dim]
            )
            net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()
            return pde.create_model(net), _loss_weights(pde, np)

        return get_model

    raise ValueError(f"Unsupported PDE: {pde_name}")


def _loss_weights(pde, np_module):
    weights = np_module.ones(pde.num_loss, dtype=np.float32)
    for i, config in enumerate(pde.loss_config):
        loss_type = config.get("type", "")
        weights[i] = 100.0 if loss_type in ("boundary", "initial", "ic") else 1.0
    return weights


def _eval_worker(payload):
    (
        run_id,
        seed,
        chain_config,
        save_root,
        display_every,
        pde_name,
        hidden_layers,
        datapath,
        trial_number,
        gpu_id,
    ) = payload

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    sys.path.insert(0, _PINNACLE_ROOT)
    os.chdir(_PINNACLE_ROOT)

    from optuna_trainer import train_chain

    get_model = _build_get_model(pde_name, hidden_layers, datapath)
    save_path = os.path.join(save_root, f"eval_run_{run_id}")
    mse, rmse, brmse = train_chain(
        get_model=get_model,
        chain_config=chain_config,
        display_every=display_every,
        save_path=save_path,
        experiment=None,
        trial_number=trial_number,
        seed=seed,
    )
    return run_id, float(mse), float(rmse), float(brmse)


def _resolve(path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.join(_PINNACLE_ROOT, path)


def _write_best_conf(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"Best configuration written to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", choices=sorted(PDE_DEFAULTS), required=True)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--hidden-layers", type=str, default=None)
    parser.add_argument("--datapath", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out", type=str, default="runs_optuna")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--n-eval-runs", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--parallel", type=int, required=True)
    parser.add_argument("--results-csv", type=str, default=None)
    parser.add_argument("--best-conf", type=str, default=None)
    args = parser.parse_args()

    defaults = PDE_DEFAULTS[args.pde]
    db_path = _resolve(args.db_path or defaults["db_path"])
    study_name = args.study_name or defaults["study_name"]
    hidden_layers = args.hidden_layers if args.hidden_layers is not None else defaults["hidden_layers"]
    datapath = args.datapath if args.datapath is not None else defaults["datapath"]
    run_name = args.name or defaults["name"]
    results_csv = args.results_csv if args.results_csv is not None else defaults["results_csv"]
    best_conf = _resolve(args.best_conf or f"{args.pde}_best_conf")

    sys.path.insert(0, _PINNACLE_ROOT)
    os.chdir(_PINNACLE_ROOT)

    import numpy as np
    import optuna

    from optuna_trainer import create_optuna_rdb_storage, params_to_chain_config, write_results_csv

    storage = create_optuna_rdb_storage(db_path)
    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = study.best_trial
    best_chain = params_to_chain_config(
        best_trial.params,
        OPTIMIZERS,
        use_continuous_params=True,
    )
    traj_json = json.dumps(best_chain, indent=None)

    best_mse = best_trial.user_attrs.get("last_mse")
    best_rmse = best_trial.user_attrs.get("last_rmse")
    best_brmse = best_trial.user_attrs.get("last_brmse")

    best_data = {
        "pde": args.pde,
        "db_path": db_path,
        "study_name": study_name,
        "trial_number": best_trial.number,
        "objective_rmse_plus_brmse": best_trial.value,
        "params": best_trial.params,
        "chain_config": best_chain,
        "n_processes": args.parallel,
    }
    _write_best_conf(best_conf, best_data)

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_base_path = os.path.join(_PINNACLE_ROOT, args.out, f"{date_str}-{run_name}")
    eval_save = os.path.join(save_base_path, "best_chain_eval")
    os.makedirs(eval_save, exist_ok=True)

    gpu_count = _detect_gpu_count()
    max_workers = max(1, min(args.parallel, args.n_eval_runs))
    print(
        f"Using study={study_name}, best_trial={best_trial.number}, "
        f"parallel={max_workers}, detected_gpus={gpu_count}"
    )

    payloads = []
    for run_id in range(args.n_eval_runs):
        gpu_id = run_id % gpu_count if gpu_count > 0 else None
        payloads.append(
            (
                run_id,
                int(args.seed_base + run_id),
                best_chain,
                eval_save,
                args.log_every,
                args.pde,
                hidden_layers,
                datapath,
                10_000 + run_id,
                gpu_id,
            )
        )

    ctx = mp.get_context("spawn")
    eval_results_map: dict[int, tuple[float, float, float]] = {}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_eval_worker, payload): payload[0] for payload in payloads}
        for future in as_completed(futures):
            run_id, mse, rmse, brmse = future.result()
            eval_results_map[run_id] = (mse, rmse, brmse)
            print(f"Finished eval run_id={run_id} MSE={mse} RMSE={rmse} BRMSE={brmse}")

    eval_results = [eval_results_map[i] for i in range(args.n_eval_runs)]
    rows = [
        {
            "phase": "optuna_best",
            "run_id": best_trial.number,
            "mse": best_mse if best_mse is not None else "",
            "rmse": best_rmse if best_rmse is not None else "",
            "brmse": best_brmse if best_brmse is not None else "",
            "trajectory_json": traj_json,
        }
    ]
    for run_id, (mse, rmse, brmse) in enumerate(eval_results):
        rows.append(
            {
                "phase": "eval",
                "run_id": run_id,
                "mse": mse,
                "rmse": rmse,
                "brmse": brmse,
                "trajectory_json": traj_json,
            }
        )

    mses = [item[0] for item in eval_results if np.isfinite(item[0])]
    rmses = [item[1] for item in eval_results if np.isfinite(item[1])]
    brmses = [item[2] for item in eval_results if np.isfinite(item[2])]
    rows.append(
        {
            "phase": "eval_summary",
            "run_id": "mean",
            "mse": float(np.mean(mses)) if mses else "",
            "rmse": float(np.mean(rmses)) if rmses else "",
            "brmse": float(np.mean(brmses)) if brmses else "",
            "trajectory_json": traj_json,
        }
    )

    out_csv = os.path.join(save_base_path, "results.csv")
    write_results_csv(out_csv, rows)

    if results_csv:
        if os.path.basename(results_csv) != results_csv:
            raise ValueError("results_csv must be a basename only")
        project_csv = os.path.join(_PINNACLE_ROOT, results_csv)
        write_results_csv(project_csv, rows)
        print(f"Done. Wrote {out_csv} and {project_csv}")
    else:
        print(f"Done. Wrote {out_csv}")


if __name__ == "__main__":
    main()
