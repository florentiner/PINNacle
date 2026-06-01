import os, sys
os.environ["DDEBACKEND"] = "pytorch"
from comet_ml import start

proj_name = "rlpinn-poisson-boltzmann-2d-optuna"
experiment = start(
    api_key="NM8cfXp7qp88rpeXKKvf9ZIdd",
    project_name=proj_name,
    workspace="florentiner"
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
import time
import argparse
import numpy as np
import torch
import deepxde as dde

from src.pde.poisson import PoissonBoltzmann2D
from src.utils.args import parse_hidden_layers
from optuna_trainer import run_optuna_study


def build_get_model(hidden_layers: str):
    def get_model():
        pde = PoissonBoltzmann2D()
        layers = [pde.input_dim] + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)) + [pde.output_dim]
        net = dde.nn.FNN(layers, "tanh", "Glorot normal")
        net = net.float()

        loss_weights = np.ones(pde.num_loss, dtype=float)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="poisson_boltzmann_2d_optuna")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=2000)
    parser.add_argument("--out", type=str, default="runs_optuna")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-eval-runs", type=int, default=10)
    parser.add_argument(
        "--results-csv",
        type=str,
        default="poisson_boltzmann_2d.csv",
        help="Written under PINNacle repo root (basename only).",
    )
    parser.add_argument("--study-name", type=str, default="poisson_boltzmann_2d_chain")
    parser.add_argument("--db-path", type=str, default="optuna_studies/poisson_boltzmann_2d.db")
    args = parser.parse_args()

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{date_str}-{args.name}")
    os.makedirs(save_path, exist_ok=True)

    get_model = build_get_model(args.hidden_layers)

    optimizers = {
        'Adam': {
            'lr': [1e-2, 1e-3, 1e-4],
            'epochs': [100, 1000, 2500],
        },
        'LBFGS': {
            'lr': [1, 5e-1, 1e-1],
            'epochs': [100, 500, 1500],
        },
        'PSO': {
            'lr': [0.0, 1e-3, 1e-4],
            'epochs': [100, 200, 300],
        },
    }

    experiment.log_parameters({
        "optimizers": optimizers,
        "n_trials": args.n_trials,
        "n_eval_runs": args.n_eval_runs,
        "results_csv": args.results_csv,
        "study_name": args.study_name,
        "hidden_layers": args.hidden_layers,
    })

    run_optuna_study(
        study_name=args.study_name,
        db_path=args.db_path,
        n_trials=args.n_trials,
        get_model=get_model,
        optimizers=optimizers,
        display_every=args.log_every,
        save_base_path=save_path,
        experiment=experiment,
        n_eval_runs=args.n_eval_runs,
        results_csv_basename=args.results_csv,
    )


if __name__ == "__main__":
    main()
