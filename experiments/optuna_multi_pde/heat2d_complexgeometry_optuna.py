import os, sys
os.environ["DDEBACKEND"] = "pytorch"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
import time
import argparse
import numpy as np
import torch
import deepxde as dde

from src.pde.heat import Heat2D_ComplexGeometry
from src.utils.args import parse_hidden_layers
from optuna_trainer import run_optuna_study


def build_get_model(hidden_layers: str, datapath: str):
    def get_model():
        pde = Heat2D_ComplexGeometry(datapath=datapath)
        layers = [pde.input_dim] + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)) + [pde.output_dim]
        net = dde.nn.FNN(layers, "tanh", "Glorot normal")
        net = net.float()

        loss_weights = np.ones(pde.num_loss, dtype=np.float32)
        for i, c in enumerate(pde.loss_config):
            t = c.get("type", "")
            if t in ("boundary", "initial", "ic"):
                loss_weights[i] = 100.0
            else:
                loss_weights[i] = 1.0

        model = pde.create_model(net)
        return model, loss_weights
    return get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="heat2d_complexgeometry_optuna")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out", type=str, default="runs_optuna")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-eval-runs", type=int, default=10)
    parser.add_argument("--results-csv", type=str, default="heat2d_complexgeometry.csv")
    parser.add_argument("--study-name", type=str, default="heat2d_complexgeometry_chain")
    parser.add_argument("--db-path", type=str, default="optuna_studies/heat2d_complexgeometry.db")
    parser.add_argument("--datapath", type=str, default="ref/heat_complex.dat")
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    parser.add_argument("--sampler", type=str, default="tpe", choices=("tpe", "random"))
    parser.add_argument("--sampler-seed", type=int, default=None)
    parser.add_argument("--value-type", type=str, default="continuous",
                        choices=("continuous", "fixed"),
                        help="'continuous': ranges; 'fixed': discrete lists.")
    args = parser.parse_args()

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{date_str}-{args.name}")
    os.makedirs(save_path, exist_ok=True)

    get_model = build_get_model(args.hidden_layers, args.datapath)

    _optimizers_continuous = {
        "Adam":  {"lr": (1e-4, 1e-2),  "lr_log": False, "epochs": (100, 2500)},
        "LBFGS": {"lr": (1e-1, 1.0),   "lr_log": False, "epochs": (100, 1500)},
        "PSO":   {"lr": (0.0, 1e-3),   "lr_log": False, "epochs": (100, 300)},
    }
    _optimizers_fixed = {
        "Adam":  {"lr": [1e-2, 1e-3, 1e-4], "epochs": [100, 1000, 2500]},
        "LBFGS": {"lr": [1.0, 5e-1, 1e-1],  "epochs": [100, 500, 1500]},
        "PSO":   {"lr": [0.0, 1e-3, 1e-4],  "epochs": [100, 200, 300]},
    }
    use_continuous = (args.value_type == "continuous")
    optimizers = _optimizers_continuous if use_continuous else _optimizers_fixed

    timeout_seconds = None if args.timeout_hours <= 0 else float(args.timeout_hours) * 3600.0

    run_optuna_study(
        study_name=args.study_name,
        db_path=args.db_path,
        n_trials=args.n_trials,
        get_model=get_model,
        optimizers=optimizers,
        display_every=args.log_every,
        save_base_path=save_path,
        experiment=None,
        n_eval_runs=args.n_eval_runs,
        results_csv_basename=args.results_csv,
        use_continuous_chain_params=use_continuous,
        timeout_seconds=timeout_seconds,
        sampler_name=args.sampler,
        sampler_seed=args.sampler_seed,
    )


if __name__ == "__main__":
    main()
