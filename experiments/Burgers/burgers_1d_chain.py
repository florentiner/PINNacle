import os
import sys
os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from comet_ml import start
from dotenv import load_dotenv
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
api_key = os.getenv("COMET_API_KEY")

experiment = start(
    api_key=api_key,
    project_name="rlpinn_burgers1d_tolerance_corrected",
    workspace="saitama32",
)


sys.path.insert(0, PROJECT_ROOT)  # insert (not append) so local landscape_visualization/src shadow any same-named packages in site-packages
import time
import argparse
import dill
import numpy as np
import torch
import deepxde as dde


from src.pde.burgers import Burgers1D
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback
from src.frozen_pinn import solve_burgers1d_frozen
from rl_trainer import train_process_rl


experiment.log_parameters({
    "param": "v_1",
    "reward_function": "v_2",
    "description": "tolerance_burgers1d_rl_optimizer",
})


def build_get_model_burgers1d(hidden_layers: str, **pde_kwargs):
    def get_model():
        pde = Burgers1D(**pde_kwargs)

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


def run_frozen_pinn(args, save_path):
    """Gradient-free solve via Frozen-PINN, bypassing the RL/optimizer training pipeline."""
    pde = Burgers1D(datapath=args.datapath, nu=args.nu)

    t0 = time.time()
    sol, features, predict = solve_burgers1d_frozen(
        geom=(pde.geom.l, pde.geom.r),
        time=(pde.geomtime.timedomain.t0, pde.geomtime.timedomain.t1),
        nu=args.nu,
        num_features=args.frozen_num_features,
        num_collocation=args.frozen_num_collocation,
        eta=args.frozen_eta,
        seed=args.seed,
        num_time_eval=args.frozen_num_time_eval,
    )
    elapsed = time.time() - t0

    u_pred = predict(pde.ref_data[:, 0], pde.ref_data[:, 1])
    u_true = pde.ref_data[:, 2]
    rmse = float(np.sqrt(np.mean((u_pred - u_true) ** 2)))

    print(f"\nFrozen-PINN solve finished in {elapsed:.3f}s ({sol.nfev} RHS evaluations).")
    print(f"RMSE vs reference data: {rmse:.6f}")

    np.savez(
        os.path.join(save_path, "frozen_pinn_solution.npz"),
        x_ref=pde.ref_data[:, 0],
        t_ref=pde.ref_data[:, 1],
        u_true=u_true,
        u_pred=u_pred,
        rmse=rmse,
        solve_time=elapsed,
    )

    experiment.log_parameters({
        "pinn_mode": "frozen",
        "frozen_num_features": args.frozen_num_features,
        "frozen_num_collocation": args.frozen_num_collocation,
        "frozen_eta": args.frozen_eta,
        "frozen_num_time_eval": args.frozen_num_time_eval,
    })
    experiment.log_metrics({"frozen_rmse": rmse, "frozen_solve_time": elapsed})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="burgers1d_rl")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=2000)
    parser.add_argument("--n-trajectories", type=int, default=1000)
    parser.add_argument("--n-save-models", type=int, default=10)
    parser.add_argument("--out", type=str, default="runs_single")

    parser.add_argument("--datapath", type=str, default="ref/burgers1d.dat", help="Reference data path")
    parser.add_argument("--nu", type=float, default=float(0.01 / np.pi), help="Viscosity")

    parser.add_argument("--loss-type", type=str, choices=["origin", "causal"], default="origin",
                         help="'origin' for the plain mean PDE loss, 'causal' for causal-training weighting (Wang et al. 2022)")
    parser.add_argument("--causal-eps", type=float, default=1.0, help="Causal weighting steepness (only used if --loss-type=causal)")
    parser.add_argument("--num-causal-buckets", type=int, default=32, help="Number of time buckets for causal weighting (only used if --loss-type=causal)")
    parser.add_argument("--optimizer-type", type=str, choices=["origin", "second-order"], default="origin",
                         help="'origin' for the current RL-driven Adam/LBFGS/PSO optimizer schedule, 'second-order' to use the SOAP optimizer")

    parser.add_argument("--pinn-mode", type=str, choices=["origin", "frozen"], default="origin",
                         help="'origin' for the current RL-driven gradient-descent PINN pipeline, "
                              "'frozen' to solve with Frozen-PINN (frozen random features + gradient-free ODE-in-time solve, arXiv:2405.20836) instead")
    parser.add_argument("--frozen-num-features", type=int, default=2000, help="Number of frozen random features (only used if --pinn-mode=frozen)")
    parser.add_argument("--frozen-num-collocation", type=int, default=4000, help="Number of spatial collocation points (only used if --pinn-mode=frozen)")
    parser.add_argument("--frozen-eta", type=float, default=2.0, help="Random-feature bias range [-eta, eta] (only used if --pinn-mode=frozen)")
    parser.add_argument("--frozen-num-time-eval", type=int, default=201, help="Number of time points to evaluate the ODE solution at (only used if --pinn-mode=frozen)")

    args = parser.parse_args()

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{date_str}-{args.name}")
    os.makedirs(save_path, exist_ok=True)

    if args.pinn_mode == "frozen":
        run_frozen_pinn(args, save_path)
        return

    pde_kwargs = dict(
        datapath=args.datapath,
        nu=args.nu,
        loss_type=args.loss_type,
        causal_eps=args.causal_eps,
        num_causal_buckets=args.num_causal_buckets,
    )

    get_model = build_get_model_burgers1d(args.hidden_layers, **pde_kwargs)
    get_model_rec = build_get_model_burgers1d(args.hidden_layers, **pde_kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_args = {
        "iterations": 1,
        "display_every": args.log_every,
        "callbacks": [
            TesterCallback(log_every=args.log_every),
            PlotCallback(log_every=args.plot_every, fast=True),
            LossCallback(verbose=True),
        ],
        "n_trajectories": args.n_trajectories,
        "n_save_models": args.n_save_models,
        "operator_coeff": 1,
        "bnd_coeff": 1,
    }

    optimizers_origin = {
        "Adam": {"lr": [1e-2, 1e-3, 1e-4], "epochs": [100, 1000, 2500]},
        "LBFGS": {"lr": [1, 5e-1, 1e-1], "epochs": [100, 500, 1500]},
        "PSO": {"lr": [0.0, 1e-3, 1e-4], "epochs": [100, 200, 300]},
    }
    optimizers_second_order = {
        "Adam": {"lr": [1e-2, 1e-3], "epochs": [100, 1000]},
        "SOAP": {"lr": [3e-3, 1e-3], "epochs": [500, 2000]},
    }
    optimizers = optimizers_second_order if args.optimizer_type == "second-order" else optimizers_origin

    AE_model_params = {
        "mode": "NN",
        "num_of_layers": 3,
        "layers_AE": [991, 125, 15],
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "every_nth": 1,
        "grid_step": 0.1,
        "d_max_latent": 2,
        "anchor_mode": "circle",
        "rec_weight": 10000.0,
        "anchor_weight": 0.0,
        "lastzero_weight": 0.0,
        "polars_weight": 0.0,
        "wellspacedtrajectory_weight": 0.0,
        "gridscaling_weight": 0.0,
        "device": device,
    }

    AE_train_params = {
        "first_RL_epoch_AE_params": {
            "epochs": 10000,
            "patience_scheduler": 4000,
            "cosine_scheduler_patience": 1200,
        },
        "other_RL_epoch_AE_params": {
            "epochs": 20000,
            "patience_scheduler": 4000,
            "cosine_scheduler_patience": 1200,
        },
        "batch_size": 32,
        "every_epoch": 100,
        "learning_rate": 5e-4,
        "resume": True,
        "finetune_AE_model": False,
        "log_key": True,
    }

    loss_surface_params = {
        "loss_types": ["loss_total", "loss_oper", "loss_bnd"],
        "every_nth": 1,
        "num_of_layers": 3,
        "layers_AE": [991, 125, 15],
        "batch_size": 32,
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "loss_name": "loss_total",
        "x_range": [-1.25, 1.25, 25],
        "vmax": -1.0,
        "vmin": -1.0,
        "vlevel": 30.0,
        "key_models": None,
        "key_modelnames": None,
        "density_type": "CKA",
        "density_p": 2,
        "density_vmax": -1,
        "density_vmin": -1,
        "colorFromGridOnly": True,
        "img_dir": "",
        "dde_pde_model": get_model_rec,
    }

    rl_agent_params = {
        "n_save_models": args.n_save_models,
        "n_trajectories": args.n_trajectories,
        "tolerance": 0.0,
        "stuck_threshold": 10,
        "min_loss_change": 1e-7,
        "min_grad_norm": 1e-5,
        "rl_buffer_size": 10000,
        "rl_batch_size": 32,
        "n_transitions_reinit": 2000,
        "gamma": 0.9,
        "rl_reward_method": "absolute",
        "reward_operator_coeff": 1,
        "reward_boundary_coeff": 1,
        "agent_min_buffer": 32,
        "agent_update_iters": 5,
        "lr": 1e-3,
        "exp": experiment,
    }

    experiment.log_parameters(rl_agent_params)
    experiment.log_parameters({
        "loss_type": args.loss_type,
        "causal_eps": args.causal_eps,
        "num_causal_buckets": args.num_causal_buckets,
        "optimizer_type": args.optimizer_type,
    })

    data = dill.dumps((get_model, train_args, optimizers, AE_model_params, AE_train_params, loss_surface_params))
    train_process_rl(data=data, save_path=save_path, device=args.device, seed=args.seed, rl_agent_params=rl_agent_params)


if __name__ == "__main__":
    main()
