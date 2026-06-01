from __future__ import annotations

import os
import csv
import json
import datetime
import math
import numpy as np
import torch
import deepxde as dde
import optuna
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

from src.utils.callbacks import TesterCallback, ModelSaverCallback
from deepxde.optimizers.config import set_PSO_options

dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)

# Repository root (directory containing this file); portable across machines.
PINNACLE_ROOT = os.path.dirname(os.path.abspath(__file__))
CHAIN_STEPS = 5


def create_optuna_rdb_storage(db_path: str) -> optuna.storages.RDBStorage:
    """
    Build RDBStorage for SQLite with a canonical URL and long lock timeout.

    Relative ``db_path`` is resolved to an absolute path so workers, the dashboard,
    and one-off scripts all target the same file regardless of process cwd. Parallel
    workers then use ``connect_args["timeout"]`` well above SQLite's default (5s) to
    reduce lock-related failures when many processes share one DB.
    """
    abs_path = os.path.abspath(os.path.expanduser(db_path))
    if os.name == "nt":
        norm = os.path.normpath(abs_path).replace("\\", "/")
        url = f"sqlite:///{norm}"
    else:
        norm = os.path.normpath(abs_path).replace("\\", "/")
        if norm.startswith("/"):
            url = f"sqlite:////{norm[1:]}"
        else:
            url = f"sqlite:///{norm}"

    engine_kwargs = {
        "connect_args": {
            "timeout": 300.0,
            "check_same_thread": False,
        }
    }
    return optuna.storages.RDBStorage(url=url, engine_kwargs=engine_kwargs)

# TPE: expensive training runs → more quasi-random exploration before model-heavy acquisition.
# CMA-ES: population scale ~ 4+3*log(n); startup trials ≈ 2*pop to seed the CMA state.
TPE_N_STARTUP_TRIALS = 32
TPE_N_EI_CANDIDATES = 64


def _cmaes_n_startup_trials(n_params: int) -> int:
    if n_params < 1:
        return 20
    pop = 4 + int(3 * math.log(n_params + 1))
    return max(20, min(80, 2 * pop + 4))


def build_study_sampler(
    name: str,
    *,
    use_conditional_chain: bool,
    seed: int | None = None,
    cmaes_param_count: int | None = None,
):
    """
    Construct an Optuna sampler for this project.

    * **``"tpe"`` (default)** — :class:`TPESampler` tuned for expensive PINN-style trials:
      high startup exploration, many EI candidates, endpoint-aware priors, ``constant_liar``
      for parallel processes. With a **flat** chain search space (fixed param names per step),
      ``multivariate`` and ``group`` are True. If you pass ``use_conditional_chain=True``,
      both are False (legacy define-by-run branches).

    * **``"cmaes"``** — :class:`CmaEsSampler` (requires the ``cmaes`` package). Optuna’s
      CMA-ES does **not** support categoricals, so the objective must use a **static**
      encoding: ``suggest_int`` for the optimizer index plus **all** per-optimizer
      lr/epoch dimensions (``cmaes_static_space`` in :func:`suggest_chain_config`). CMA-ES
      is non-smooth when only one branch is active; TPE is often a safer default for this
      chain structure. Parallel CMA-ES is weak in Optuna; prefer TPE for many workers.

    * **``"random"``** — uniform baseline.

    * ``cmaes_param_count`` — number of CMA-ES dimensions; required when name is
      ``"cmaes"`` (used to set ``n_startup_trials``).
    """
    key = (name or "tpe").strip().lower()
    if key == "tpe":
        # Optuna requires ``group=True`` only when ``multivariate`` is True.
        _mvt = not use_conditional_chain
        return TPESampler(
            multivariate=_mvt,
            n_startup_trials=TPE_N_STARTUP_TRIALS,
            n_ei_candidates=TPE_N_EI_CANDIDATES,
            consider_endpoints=True,
            group=_mvt,
            constant_liar=True,
            seed=seed,
        )
    if key == "cmaes":
        if cmaes_param_count is None or cmaes_param_count < 1:
            raise ValueError("cmaes_param_count (positive int) is required for sampler 'cmaes'")
        try:
            import cmaes  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "CmaEsSampler needs the 'cmaes' package. Install: pip install cmaes"
            ) from e
        return CmaEsSampler(
            n_startup_trials=_cmaes_n_startup_trials(cmaes_param_count),
            with_margin=True,
            seed=seed,
        )
    if key == "random":
        return RandomSampler(seed=seed)
    raise ValueError(
        f"Unknown sampler {name!r}. Supported: 'tpe' (default), 'cmaes' (static encoding), 'random'."
    )


def _comet_log_metric(experiment, name, value, step=None):
    """Comet: no-op when experiment is None (offline / no tracking)."""
    if experiment is None:
        return
    experiment.log_metric(name, value, step=step)


def _comet_log_metrics(experiment, metrics, step=None):
    if experiment is None:
        return
    experiment.log_metrics(metrics, step=step)


def _comet_log_parameters(experiment, params):
    if experiment is None:
        return
    experiment.log_parameters(params)


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


def _global_epochs_bounds(optimizers: dict) -> tuple[int, int]:
    """Min/max epoch limits across all optimizer specs (for a single global ``suggest_int``)."""
    lows, highs = [], []
    for v in optimizers.values():
        e_lo, e_hi = int(v["epochs"][0]), int(v["epochs"][1])
        if e_hi < e_lo:
            e_lo, e_hi = e_hi, e_lo
        lows.append(e_lo)
        highs.append(e_hi)
    return min(lows), max(highs)


def _decode_flat_lr_epochs(u_lr: float, epochs_guess: int, spec: dict) -> tuple[float, int]:
    """
    Map dashboard-friendly flat params to (lr, epochs) for one stage.

    * ``u_lr`` — uniform in [0, 1], mapped linearly or log-uniformly into ``spec['lr']``.
    * ``epochs_guess`` — int from global range, clipped/snapped into this optimizer's epoch bounds.
    """
    u_lr = float(min(1.0, max(0.0, u_lr)))
    lr_lo, lr_hi = (float(spec["lr"][0]), float(spec["lr"][1]))
    e_lo, e_hi = (int(spec["epochs"][0]), int(spec["epochs"][1]))
    if e_hi < e_lo:
        e_lo, e_hi = e_hi, e_lo
    log_scale = bool(spec.get("lr_log", False))
    e_step = int(spec.get("epochs_step", 1) or 1)
    if e_step < 1:
        e_step = 1

    if log_scale and lr_lo > 0.0 and lr_hi > 0.0:
        lr = math.exp(math.log(lr_lo) + u_lr * (math.log(lr_hi) - math.log(lr_lo)))
    else:
        lr = lr_lo + u_lr * (lr_hi - lr_lo)

    epochs = int(max(e_lo, min(e_hi, int(epochs_guess))))
    if e_step > 1:
        k = round((epochs - e_lo) / e_step)
        epochs = e_lo + int(k) * e_step
        epochs = max(e_lo, min(e_hi, epochs))
    return float(lr), int(epochs)


def _suggest_flat_lr_epochs(
    trial, step_idx: int, spec: dict, e_glob_lo: int, e_glob_hi: int
) -> tuple[float, int]:
    """Fixed param names per step: ``step_{i}_lr`` ∈ [0,1], ``step_{i}_epochs`` int (global bounds)."""
    u_lr = trial.suggest_float(f"step_{step_idx}_lr", 0.0, 1.0)
    epochs_guess = trial.suggest_int(f"step_{step_idx}_epochs", e_glob_lo, e_glob_hi)
    return _decode_flat_lr_epochs(u_lr, epochs_guess, spec)


def _suggest_lr_epochs_for_opt(trial, step_idx: int, opt_name: str, spec: dict) -> tuple:
    """
    Suggest lr and epochs for a single (step, optimizer) branch. Call only for the
    optimizer chosen for this step (conditional / define-by-run); names stay
    ``step_{i}__{opt_name}__*`` so each branch has its own float/int distributions.

    spec keys: lr (min, max), epochs (e_min, e_max), optional lr_log, optional epochs_step.
    Learning rate is a float; epochs are an integer in [e_min, e_max] (``suggest_int``, with
    ``step=epochs_step`` when that is greater than 1).
    """
    lr_lo, lr_hi = (float(spec["lr"][0]), float(spec["lr"][1]))
    e_lo, e_hi = (int(spec["epochs"][0]), int(spec["epochs"][1]))
    if e_hi < e_lo:
        e_lo, e_hi = e_hi, e_lo
    log_scale = bool(spec.get("lr_log", False))
    e_step = int(spec.get("epochs_step", 1) or 1)
    if e_step < 1:
        e_step = 1

    prefix = f"step_{step_idx}__{opt_name}"
    if log_scale and lr_lo > 0.0 and lr_hi > 0.0:
        lr = trial.suggest_float(f"{prefix}__lr", lr_lo, lr_hi, log=True)
    else:
        lr = trial.suggest_float(f"{prefix}__lr", lr_lo, lr_hi, log=False)

    if e_step == 1:
        epochs = trial.suggest_int(f"{prefix}__epochs", e_lo, e_hi)
    else:
        epochs = trial.suggest_int(f"{prefix}__epochs", e_lo, e_hi, step=e_step)
    return float(lr), int(epochs)


def train_chain(get_model, chain_config, display_every, save_path, experiment, trial_number, seed=None):
    """
    Train a PINN with the given optimizer chain. Each stage is a dict with
    keys optimizer (or legacy type), lr, epochs.
    Returns (mse, rmse, brmse) after the last stage.

    If ``seed`` is set, DeepXDE / NumPy / torch seeds are applied before building the model.
    """
    os.makedirs(save_path, exist_ok=True)

    if seed is not None:
        dde.config.set_random_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

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
        l2re = getattr(tester, "l2re", float("inf"))
        bc_l2re = getattr(tester, "bc_l2re", float("inf"))

        print(f"After {opt_type} (stage {stage_idx}): MSE={mse}, RMSE={rmse}, BRMSE={brmse}, L2RE={l2re}, BC_L2RE={bc_l2re}")

        tag = f"stage_{stage_idx}_{opt_type.lower()}"
        _comet_log_metric(experiment, f"rmse_after_{tag}", rmse, step=trial_number)
        _comet_log_metric(experiment, f"brmse_after_{tag}", brmse, step=trial_number)
        _comet_log_metric(experiment, f"l2re_after_{tag}", l2re, step=trial_number)

        if not np.isfinite(rmse) or not np.isfinite(brmse):
            print(f"NaN/Inf detected after {opt_type}. Stopping chain early.")
            break

    return mse, rmse, brmse, l2re, bc_l2re


def suggest_chain_config(
    trial,
    optimizers,
    chain_steps=CHAIN_STEPS,
    use_continuous_params=False,
    cmaes_static_space: bool = False,
):
    """Optuna: discrete index grids, or TPE flat (type + lr unit + epochs int), or CMA-ES static multibranch."""
    opt_types = list(optimizers.keys())
    n_opt = len(opt_types)
    if use_continuous_params:
        chain = []
        for i in range(chain_steps):
            if cmaes_static_space:
                k = trial.suggest_int(f"step_{i}_opt", 0, n_opt - 1)
                opt_type = opt_types[k]
                branch = {}
                for oname in opt_types:
                    branch[oname] = _suggest_lr_epochs_for_opt(
                        trial, i, oname, optimizers[oname]
                    )
                lr, epochs = branch[opt_type]
            else:
                opt_type = trial.suggest_categorical(f"step_{i}_type", opt_types)
                e_lo_g, e_hi_g = _global_epochs_bounds(optimizers)
                lr, epochs = _suggest_flat_lr_epochs(
                    trial, i, optimizers[opt_type], e_lo_g, e_hi_g
                )

            trial.set_user_attr(f"step_{i}_lr", lr)
            trial.set_user_attr(f"step_{i}_epochs", epochs)
            trial.set_user_attr(f"step_{i}_type", opt_type)

            chain.append({"optimizer": opt_type, "lr": lr, "epochs": epochs})
        return chain

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
        trial.set_user_attr(f"step_{i}_type", opt_type)

        chain.append({"optimizer": opt_type, "lr": lr, "epochs": epochs})
    return chain


def params_to_chain_config(
    params,
    optimizers,
    chain_steps=CHAIN_STEPS,
    use_continuous_params=False,
    cmaes_static_space: bool = False,
):
    """Rebuild chain from Optuna trial.params (index-based, TPE continuous, or CMA-ES static)."""
    opt_types = list(optimizers.keys())
    chain = []
    for i in range(chain_steps):
        if use_continuous_params:
            if cmaes_static_space:
                k = int(params[f"step_{i}_opt"])
                if k < 0 or k >= len(opt_types):
                    raise ValueError(f"Invalid step_{i}_opt: {k}")
                opt_type = opt_types[k]
            else:
                opt_type = params[f"step_{i}_type"]
            # Flat space (same keys every trial; dashboard shows lr + epochs)
            if f"step_{i}_lr" in params:
                u_lr = float(params[f"step_{i}_lr"])
                epochs_guess = int(round(float(params[f"step_{i}_epochs"])))
                lr, epochs = _decode_flat_lr_epochs(
                    u_lr, epochs_guess, optimizers[opt_type]
                )
            else:
                # Legacy: conditional branch names ``step_{i}__{Optimizer}__*``
                lr = float(params[f"step_{i}__{opt_type}__lr"])
                epochs = int(round(float(params[f"step_{i}__{opt_type}__epochs"])))
        else:
            opt_type = params[f"step_{i}_type"]
            lr_idx = params[f"step_{i}_lr_idx"]
            ep_idx = params[f"step_{i}_epochs_idx"]
            lr = optimizers[opt_type]["lr"][lr_idx]
            epochs = int(optimizers[opt_type]["epochs"][ep_idx])
        chain.append({"optimizer": opt_type, "lr": lr, "epochs": epochs})
    return chain


def create_optuna_objective(
    get_model,
    optimizers,
    display_every,
    save_base_path,
    experiment,
    chain_steps=CHAIN_STEPS,
    use_continuous_params=False,
    cmaes_static_space: bool = False,
):
    def objective(trial):
        chain_config = suggest_chain_config(
            trial,
            optimizers,
            chain_steps=chain_steps,
            use_continuous_params=use_continuous_params,
            cmaes_static_space=cmaes_static_space,
        )

        trial_save_path = os.path.join(save_base_path, f"trial_{trial.number}")

        print(f"\n{'#'*70}")
        print(f"Starting Optuna Trial {trial.number}")
        print(f"Chain config: {chain_config}")
        print(f"{'#'*70}\n")

        mse, rmse, brmse, l2re, bc_l2re = train_chain(
            get_model=get_model,
            chain_config=chain_config,
            display_every=display_every,
            save_path=trial_save_path,
            experiment=experiment,
            trial_number=trial.number,
        )

        trial.set_user_attr("last_mse", float(mse) if np.isfinite(mse) else None)
        trial.set_user_attr("last_rmse", float(rmse) if np.isfinite(rmse) else None)
        trial.set_user_attr("last_brmse", float(brmse) if np.isfinite(brmse) else None)
        trial.set_user_attr("last_l2re", float(l2re) if np.isfinite(l2re) else None)
        trial.set_user_attr("last_bc_l2re", float(bc_l2re) if np.isfinite(bc_l2re) else None)

        obj_value = rmse + brmse if (np.isfinite(rmse) and np.isfinite(brmse)) else float("inf")

        _comet_log_metrics(
            experiment,
            {
                "final_mse": mse if np.isfinite(mse) else -1,
                "final_rmse": rmse if np.isfinite(rmse) else -1,
                "final_brmse": brmse if np.isfinite(brmse) else -1,
                "final_l2re": l2re if np.isfinite(l2re) else -1,
                "final_bc_l2re": bc_l2re if np.isfinite(bc_l2re) else -1,
                "final_objective": obj_value if np.isfinite(obj_value) else -1,
            },
            step=trial.number,
        )

        return obj_value

    return objective


def save_best_params(
    study,
    optimizers,
    save_dir,
    chain_steps=CHAIN_STEPS,
    use_continuous_params=False,
    cmaes_static_space: bool = False,
):
    best = study.best_trial
    chain_config = params_to_chain_config(
        best.params,
        optimizers,
        chain_steps=chain_steps,
        use_continuous_params=use_continuous_params,
        cmaes_static_space=cmaes_static_space,
    )
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
    """rows: list of dicts with keys phase, run_id, mse, rmse, brmse, l2re, bc_l2re, trajectory_json"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["phase", "run_id", "mse", "rmse", "brmse", "l2re", "bc_l2re", "trajectory_json"]
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
    seeds=None,
):
    """Run several trajectories with fixed chain; return list of (mse, rmse, brmse).

    If ``seeds`` is a list of length ``n_runs``, run *k* uses ``seeds[k]`` (distinct RNG seeds).
    """
    results = []
    for k in range(n_runs):
        sub = os.path.join(save_dir, f"eval_run_{k}")
        print(f"\n{'#'*70}")
        print(f"Evaluation trajectory {k + 1}/{n_runs}")
        print(f"Chain config: {chain_config}")
        print(f"{'#'*70}\n")

        seed_k = None
        if seeds is not None:
            seed_k = int(seeds[k])

        mse, rmse, brmse, l2re, bc_l2re = train_chain(
            get_model=get_model,
            chain_config=chain_config,
            display_every=display_every,
            save_path=sub,
            experiment=experiment,
            trial_number=eval_offset + k,
            seed=seed_k,
        )
        results.append((mse, rmse, brmse, l2re, bc_l2re))
        print(f"Eval {k}: MSE={mse}, RMSE={rmse}, BRMSE={brmse}, L2RE={l2re}, BC_L2RE={bc_l2re}")

    if results:
        ms = [r[0] for r in results if np.isfinite(r[0])]
        brs = [r[2] for r in results if np.isfinite(r[2])]
        l2res = [r[3] for r in results if np.isfinite(r[3])]
        bc_l2res = [r[4] for r in results if np.isfinite(r[4])]
        if ms:
            _comet_log_metric(experiment, "eval_mean_mse", float(np.mean(ms)))
        if brs:
            _comet_log_metric(experiment, "eval_mean_brmse", float(np.mean(brs)))
        if l2res:
            _comet_log_metric(experiment, "eval_mean_l2re", float(np.mean(l2res)))
        if bc_l2res:
            _comet_log_metric(experiment, "eval_mean_bc_l2re", float(np.mean(bc_l2res)))

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
    use_continuous_chain_params=False,
    timeout_seconds=None,
    sampler_name: str = "tpe",
    sampler_seed: int | None = None,
):
    """
    Optuna search over CHAIN_STEPS-length chains (Adam / LBFGS / PSO per step).
    After n_trials: save best params, run n_eval_runs with best chain, write CSV.

    results_csv_basename: if set (e.g. 'burgers_1d.csv'), write under PINNACLE_ROOT.
    use_continuous_chain_params: if True, each step samples ``step_i_type`` (categorical),
        ``step_i_lr`` (float in [0, 1], mapped to that optimizer’s lr range), and
        ``step_i_epochs`` (int in the global min/max across optimizers, then clipped to the
        chosen type’s bounds). Same parameter names on every trial so dashboards list lr/epochs.
    timeout_seconds: optional wall-clock cap for study.optimize (None = no cap).
    sampler_name / sampler_seed: see :func:`build_study_sampler` (default ``tpe``).
        Use ``cmaes`` only with ``use_continuous_chain_params=True`` (static multibranch encoding).
    """
    if results_csv_basename and os.path.basename(results_csv_basename) != results_csv_basename:
        raise ValueError("results_csv_basename must be a basename only, e.g. 'burgers_1d.csv'")

    _sampler_key = (sampler_name or "tpe").strip().lower()
    cmaes_mode = _sampler_key == "cmaes"
    if cmaes_mode and not use_continuous_chain_params:
        raise ValueError("sampler_name='cmaes' requires use_continuous_chain_params=True")

    n_opt = len(optimizers)
    cmaes_param_count = chain_steps * (1 + 2 * n_opt) if cmaes_mode else None
    # Flat chain params share names across trials → multivariate TPE is allowed and helps.
    _conditional = False

    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    storage = create_optuna_rdb_storage(db_path)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=build_study_sampler(
            sampler_name,
            use_conditional_chain=_conditional,
            seed=sampler_seed,
            cmaes_param_count=cmaes_param_count,
        ),
    )

    objective = create_optuna_objective(
        get_model=get_model,
        optimizers=optimizers,
        display_every=display_every,
        save_base_path=save_base_path,
        experiment=experiment,
        chain_steps=chain_steps,
        use_continuous_params=use_continuous_chain_params,
        cmaes_static_space=cmaes_mode,
    )

    opt_kwargs = {"n_trials": n_trials}
    if timeout_seconds is not None and timeout_seconds > 0:
        opt_kwargs["timeout"] = float(timeout_seconds)
    study.optimize(objective, **opt_kwargs)

    print(f"\n{'='*70}")
    print(f"Optuna study '{study_name}' completed.")
    print(f"Total trials in study: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best objective (rmse + brmse): {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")
    print(f"{'='*70}\n")

    _comet_log_parameters(
        experiment,
        {
            "best_trial_number": study.best_trial.number,
            "best_objective_rmse_plus_brmse": study.best_trial.value,
            **{f"best_{k}": v for k, v in study.best_trial.params.items()},
        },
    )

    best_data = save_best_params(
        study,
        optimizers,
        save_base_path,
        chain_steps=chain_steps,
        use_continuous_params=use_continuous_chain_params,
        cmaes_static_space=cmaes_mode,
    )
    best_chain = best_data["chain_config"]
    traj_json = _trajectory_json(best_chain)

    # Best trial metrics: re-use last trial values from study if available
    best_trial = study.best_trial
    best_mse = best_trial.user_attrs.get("last_mse")
    best_rmse = best_trial.user_attrs.get("last_rmse")
    best_brmse = best_trial.user_attrs.get("last_brmse")
    best_l2re = best_trial.user_attrs.get("last_l2re")
    best_bc_l2re = best_trial.user_attrs.get("last_bc_l2re")

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
    # Optuna-best row: objective from study; metrics from best trial attrs
    rows.append(
        {
            "phase": "optuna_best",
            "run_id": best_trial.number,
            "mse": best_mse if best_mse is not None else "",
            "rmse": best_rmse if best_rmse is not None else "",
            "brmse": best_brmse if best_brmse is not None else "",
            "l2re": best_l2re if best_l2re is not None else "",
            "bc_l2re": best_bc_l2re if best_bc_l2re is not None else "",
            "trajectory_json": traj_json,
        }
    )

    for k, (mse, rmse, brmse, l2re, bc_l2re) in enumerate(eval_results):
        rows.append(
            {
                "phase": "eval",
                "run_id": k,
                "mse": mse,
                "rmse": rmse,
                "brmse": brmse,
                "l2re": l2re,
                "bc_l2re": bc_l2re,
                "trajectory_json": traj_json,
            }
        )

    if eval_results:
        mses = [r[0] for r in eval_results if np.isfinite(r[0])]
        brmses = [r[2] for r in eval_results if np.isfinite(r[2])]
        l2res = [r[3] for r in eval_results if np.isfinite(r[3])]
        bc_l2res = [r[4] for r in eval_results if np.isfinite(r[4])]
        rows.append(
            {
                "phase": "eval_summary",
                "run_id": "mean",
                "mse": float(np.mean(mses)) if mses else "",
                "rmse": float(np.mean([r[1] for r in eval_results if np.isfinite(r[1])]))
                if eval_results
                else "",
                "brmse": float(np.mean(brmses)) if brmses else "",
                "l2re": float(np.mean(l2res)) if l2res else "",
                "bc_l2re": float(np.mean(bc_l2res)) if bc_l2res else "",
                "trajectory_json": traj_json,
            }
        )

    out_csv = os.path.join(save_base_path, "results.csv")
    write_results_csv(out_csv, rows)

    if results_csv_basename:
        project_csv = os.path.join(PINNACLE_ROOT, results_csv_basename)
        write_results_csv(project_csv, rows)

    return study
