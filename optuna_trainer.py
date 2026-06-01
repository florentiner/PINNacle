import os
import csv
import json
import datetime
import numpy as np
import torch
import deepxde as dde
import optuna
from optuna.samplers import TPESampler

from src.utils.callbacks import TesterCallback, ModelSaverCallback
from deepxde.optimizers.config import set_PSO_options

dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)

# Repository root (directory containing this file); portable across machines.
PINNACLE_ROOT = os.path.dirname(os.path.abspath(__file__))
CHAIN_STEPS = 5


def reinit_torch_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def _stage_optimizer_name(stage):
    return stage.get("optimizer") or stage.get("type")


def _build_torch_optimizer(opt_name, params, lr):
    name = (opt_name or "").lower()

    if name == "adam":
        return torch.optim.Adam(params, lr=float(lr))

    if name in ("lbfgs", "l-bfgs", "l_bfgs"):
        return torch.optim.LBFGS(
            params,
            lr=float(lr),
            line_search_fn="strong_wolfe",
            max_iter=10,
        )

    if name == "pso":
        set_PSO_options(lr=float(lr))
        return "PSO"

    raise ValueError(f"Unknown optimizer type: {opt_name}. Expected Adam / LBFGS / PSO.")


def train_chain(get_model, chain_config, display_every, save_path, experiment, trial_number):
    """
    Train a PINN with the given optimizer chain. Each stage is a dict with
    keys optimizer (or legacy type), lr, epochs.
    Returns (mse, rmse, brmse) after the last stage.
    """
    os.makedirs(save_path, exist_ok=True)

    model, loss_weights = get_model()
    model.net.apply(reinit_torch_weights)

    rmse = float("inf")
    brmse = float("inf")
    mse = float("inf")

    for stage_idx, stage in enumerate(chain_config):
        opt_type = _stage_optimizer_name(stage)
        lr = stage["lr"]
        epochs = int(stage["epochs"])

        print(f"\n{'='*70}")
        print(
            f"Trial {trial_number} | Stage {stage_idx}: {opt_type} | lr={lr} | epochs={epochs}"
        )
        print(f"Time: {datetime.datetime.now()}")
        print(f"{'='*70}\n")

        torch_opt = _build_torch_optimizer(opt_type, model.net.parameters(), lr)
        model.compile(torch_opt, loss_weights=loss_weights)
        model.optimizer = torch_opt

        tester = TesterCallback(log_every=display_every)
        saver = ModelSaverCallback(total_iterations=epochs, n_save_models=1)
        callbacks = [tester, saver]

        model.train(
            iterations=epochs,
            display_every=display_every,
            callbacks=callbacks,
            model_save_path=save_path,
            save_model=False,
        )

        rmse = tester.rmse
        brmse = tester.brmse
        mse = float(rmse) ** 2 if np.isfinite(rmse) else float("inf")

        print(f"After {opt_type} (stage {stage_idx}): MSE={mse}, RMSE={rmse}, BRMSE={brmse}")

        if experiment is not None:
            tag = f"stage_{stage_idx}_{opt_type.lower()}"
            experiment.log_metric(f"rmse_after_{tag}", rmse, step=trial_number)
            experiment.log_metric(f"brmse_after_{tag}", brmse, step=trial_number)

        if not np.isfinite(rmse) or not np.isfinite(brmse):
            print(f"NaN/Inf detected after {opt_type}. Stopping chain early.")
            break

    return mse, rmse, brmse


def suggest_chain_config(trial, optimizers, chain_steps=CHAIN_STEPS):
    """Optuna: flat index-based parameter space for full correlation learning."""
    opt_types = list(optimizers.keys())
    n_lr = max(len(v["lr"]) for v in optimizers.values())
    n_ep = max(len(v["epochs"]) for v in optimizers.values())

    chain = []
    for i in range(chain_steps):
        opt_type = trial.suggest_categorical(f"step_{i}_type", opt_types)
        lr_idx = trial.suggest_categorical(f"step_{i}_lr_idx", list(range(n_lr)))
        ep_idx = trial.suggest_categorical(f"step_{i}_epochs_idx", list(range(n_ep)))
        lr = optimizers[opt_type]["lr"][lr_idx]
        epochs = int(optimizers[opt_type]["epochs"][ep_idx])

        trial.set_user_attr(f"step_{i}_lr", lr)
        trial.set_user_attr(f"step_{i}_epochs", epochs)

        chain.append({"optimizer": opt_type, "lr": lr, "epochs": epochs})
    return chain


def params_to_chain_config(params, optimizers, chain_steps=CHAIN_STEPS):
    """Rebuild chain from Optuna trial.params (flat index-based names)."""
    chain = []
    for i in range(chain_steps):
        opt_type = params[f"step_{i}_type"]
        lr_idx = params[f"step_{i}_lr_idx"]
        ep_idx = params[f"step_{i}_epochs_idx"]
        lr = optimizers[opt_type]["lr"][lr_idx]
        epochs = int(optimizers[opt_type]["epochs"][ep_idx])
        chain.append({"optimizer": opt_type, "lr": lr, "epochs": epochs})
    return chain


def create_optuna_objective(
    get_model, optimizers, display_every, save_base_path, experiment, chain_steps=CHAIN_STEPS
):
    def objective(trial):
        chain_config = suggest_chain_config(trial, optimizers, chain_steps=chain_steps)

        trial_save_path = os.path.join(save_base_path, f"trial_{trial.number}")

        print(f"\n{'#'*70}")
        print(f"Starting Optuna Trial {trial.number}")
        print(f"Chain config: {chain_config}")
        print(f"{'#'*70}\n")

        mse, rmse, brmse = train_chain(
            get_model=get_model,
            chain_config=chain_config,
            display_every=display_every,
            save_path=trial_save_path,
            experiment=experiment,
            trial_number=trial.number,
        )

        trial.set_user_attr("last_mse", mse if np.isfinite(mse) else None)
        trial.set_user_attr("last_rmse", rmse if np.isfinite(rmse) else None)
        trial.set_user_attr("last_brmse", brmse if np.isfinite(brmse) else None)

        obj_value = rmse + brmse if (np.isfinite(rmse) and np.isfinite(brmse)) else float("inf")

        if experiment is not None:
            experiment.log_metrics(
                {
                    "final_mse": mse if np.isfinite(mse) else -1,
                    "final_rmse": rmse if np.isfinite(rmse) else -1,
                    "final_brmse": brmse if np.isfinite(brmse) else -1,
                    "final_objective": obj_value if np.isfinite(obj_value) else -1,
                },
                step=trial.number,
            )

        return obj_value

    return objective


def save_best_params(study, optimizers, save_dir, chain_steps=CHAIN_STEPS):
    best = study.best_trial
    chain_config = params_to_chain_config(best.params, optimizers, chain_steps=chain_steps)
    best_data = {
        "trial_number": best.number,
        "objective_rmse_plus_brmse": best.value,
        "params": best.params,
        "chain_config": chain_config,
    }
    path = os.path.join(save_dir, "best_optuna_params.json")
    with open(path, "w") as f:
        json.dump(best_data, f, indent=2)
    print(f"Best params saved to {path}")
    return best_data


def _trajectory_json(chain_config):
    return json.dumps(chain_config, indent=None)


def write_results_csv(
    path,
    rows,
):
    """rows: list of dicts with keys phase, run_id, mse, rmse, brmse, trajectory_json"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["phase", "run_id", "mse", "rmse", "brmse", "trajectory_json"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Results written to {path}")


def run_eval_trajectories(
    get_model,
    chain_config,
    display_every,
    save_dir,
    experiment,
    n_runs,
    eval_offset=10_000,
):
    """Run several trajectories with fixed chain; return list of (mse, rmse, brmse)."""
    results = []
    for k in range(n_runs):
        sub = os.path.join(save_dir, f"eval_run_{k}")
        print(f"\n{'#'*70}")
        print(f"Evaluation trajectory {k + 1}/{n_runs}")
        print(f"Chain config: {chain_config}")
        print(f"{'#'*70}\n")

        mse, rmse, brmse = train_chain(
            get_model=get_model,
            chain_config=chain_config,
            display_every=display_every,
            save_path=sub,
            experiment=experiment,
            trial_number=eval_offset + k,
        )
        results.append((mse, rmse, brmse))
        print(f"Eval {k}: MSE={mse}, RMSE={rmse}, BRMSE={brmse}")

    if experiment is not None and results:
        ms = [r[0] for r in results if np.isfinite(r[0])]
        brs = [r[2] for r in results if np.isfinite(r[2])]
        if ms:
            experiment.log_metric("eval_mean_mse", float(np.mean(ms)))
        if brs:
            experiment.log_metric("eval_mean_brmse", float(np.mean(brs)))

    return results


def run_optuna_study(
    study_name,
    db_path,
    n_trials,
    get_model,
    optimizers,
    display_every,
    save_base_path,
    experiment,
    chain_steps=CHAIN_STEPS,
    n_eval_runs=10,
    results_csv_basename=None,
):
    """
    Optuna search over CHAIN_STEPS-length chains (Adam / LBFGS / PSO per step).
    After n_trials: save best params, run n_eval_runs with best chain, write CSV.

    results_csv_basename: if set (e.g. 'burgers_1d.csv'), write under PINNACLE_ROOT.
    """
    if results_csv_basename and os.path.basename(results_csv_basename) != results_csv_basename:
        raise ValueError("results_csv_basename must be a basename only, e.g. 'burgers_1d.csv'")

    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    storage = optuna.storages.RDBStorage(url=f"sqlite:///{db_path}")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=TPESampler(
            multivariate=True,
            group=True,           # supports conditional/dynamic search spaces
            n_startup_trials=20,
            n_ei_candidates=48,
            constant_liar=True,
        ),
    )

    objective = create_optuna_objective(
        get_model=get_model,
        optimizers=optimizers,
        display_every=display_every,
        save_base_path=save_base_path,
        experiment=experiment,
        chain_steps=chain_steps,
    )

    study.optimize(objective, n_trials=n_trials)

    print(f"\n{'='*70}")
    print(f"Optuna study '{study_name}' completed.")
    print(f"Total trials in study: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best objective (rmse + brmse): {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")
    print(f"{'='*70}\n")

    if experiment is not None:
        experiment.log_parameters(
            {
                "best_trial_number": study.best_trial.number,
                "best_objective_rmse_plus_brmse": study.best_trial.value,
                **{f"best_{k}": v for k, v in study.best_trial.params.items()},
            }
        )

    best_data = save_best_params(study, optimizers, save_base_path, chain_steps=chain_steps)
    best_chain = best_data["chain_config"]
    traj_json = _trajectory_json(best_chain)

    # Best trial metrics: re-use last trial values from study if available
    best_trial = study.best_trial
    best_mse = best_trial.user_attrs.get("last_mse")
    best_rmse = best_trial.user_attrs.get("last_rmse")
    best_brmse = best_trial.user_attrs.get("last_brmse")

    eval_save = os.path.join(save_base_path, "best_chain_eval")
    eval_results = run_eval_trajectories(
        get_model=get_model,
        chain_config=best_chain,
        display_every=display_every,
        save_dir=eval_save,
        experiment=experiment,
        n_runs=n_eval_runs,
    )

    rows = []
    # Optuna-best row: objective from study; mse/rmse/brmse from best trial attrs or NaN
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

    if results_csv_basename:
        project_csv = os.path.join(PINNACLE_ROOT, results_csv_basename)
        write_results_csv(project_csv, rows)

    return study
