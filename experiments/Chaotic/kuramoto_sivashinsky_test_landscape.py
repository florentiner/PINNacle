"""Quick, non-RL sanity test: train KS PINN with plain Adam for 1000 iterations,
save a handful of checkpoints along the way, and render a single loss-landscape
visualization (loss_total) from that trajectory.

This bypasses the RL optimizer-selection loop in kuramoto_sivashinsky_chain.py
and instead reuses the same model/AE/landscape building blocks directly.
"""
import os
os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# insert (not append): a stale copy of `landscape_visualization` in global
# site-packages otherwise shadows this repo's local package.
sys.path.insert(0, PROJECT_ROOT)  # insert (not append) so local landscape_visualization/src shadow any same-named packages in site-packages
import time
import argparse
import numpy as np
import torch
import deepxde as dde

from src.pde.chaotic import KuramotoSivashinskyEquation
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import ModelSaverCallback
from landscape_visualization._aux.visualization_model import VisualizationModel
from landscape_visualization._aux.early_stopping_plot import EarlyStopping
from landscape_visualization._aux.plot_loss_surface import PlotLossSurface

dde.config.set_default_float("float32")


def build_get_model_kuramoto_sivashinsky(hidden_layers: str, **pde_kwargs):
    def get_model():
        pde = KuramotoSivashinskyEquation(**pde_kwargs)

        layers = [pde.input_dim] + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)) + [pde.output_dim]
        net = dde.nn.FNN(layers, "tanh", "Glorot normal")
        net = net.float()

        loss_weights = np.ones(pde.num_loss, dtype=float)
        for i, c in enumerate(pde.loss_config):
            t = c.get("type", "")
            loss_weights[i] = 100.0 if t in ("boundary", "initial", "ic") else 1.0

        model = pde.create_model(net)
        return model, loss_weights

    return get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--display-every", type=int, default=100)
    parser.add_argument("--n-save-models", type=int, default=10)
    parser.add_argument("--datapath", type=str, default=os.path.join(PROJECT_ROOT, "ref/Kuramoto_Sivashinsky.dat"))
    parser.add_argument("--alpha", type=float, default=float(100 / 16))
    parser.add_argument("--beta", type=float, default=float(100 / (16 * 16)))
    parser.add_argument("--gamma", type=float, default=float(100 / (16**4)))
    parser.add_argument("--out", type=str, default=os.path.join(PROJECT_ROOT, "runs_landscape_test", "kuramoto_sivashinsky_adam"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    pde_kwargs = dict(datapath=args.datapath, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    get_model = build_get_model_kuramoto_sivashinsky(args.hidden_layers, **pde_kwargs)
    get_model_rec = build_get_model_kuramoto_sivashinsky(args.hidden_layers, **pde_kwargs)

    # ---- 1) Plain Adam training, saving checkpoints along the trajectory ----
    model, loss_weights = get_model()
    model.compile("adam", lr=args.lr, loss_weights=loss_weights)

    saver = ModelSaverCallback(total_iterations=args.iterations, n_save_models=args.n_save_models)

    start = time.time()
    model.train(iterations=args.iterations, display_every=args.display_every, callbacks=[saver])
    print(f"Adam training done in {time.time() - start:.1f}s")

    solver_models = saver.saved_models
    print(f"Collected {len(solver_models)} checkpoints for the landscape trajectory.")

    # ---- 2) Train the trajectory autoencoder (same defaults as the RL chain) ----
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
        "device": "cpu",
    }
    vis_model = VisualizationModel(**AE_model_params)

    cb_es = EarlyStopping(patience=4000)
    ae_model = vis_model.train(
        lr=5e-4,
        cosine_scheduler_patience=1200,
        epochs=10000,
        every_epoch=100,
        batch_size=min(32, len(solver_models)),
        resume=True,
        callbacks=[cb_es],
        solver_models=solver_models,
    )

    # ---- 3) Build a single loss-landscape visualization (loss_total only) ----
    plotter = PlotLossSurface(
        loss_types=["loss_total"],
        every_nth=1,
        num_of_layers=3,
        layers_AE=[991, 125, 15],
        batch_size=min(32, len(solver_models)),
        num_models=None,
        from_last=False,
        prefix="model-",
        loss_name="loss_total",
        x_range=[-1.25, 1.25, 25],
        vmax=-1.0,
        vmin=-1.0,
        vlevel=30.0,
        key_models=None,
        key_modelnames=None,
        density_type="CKA",
        density_p=2,
        density_vmax=-1,
        density_vmin=-1,
        colorFromGridOnly=True,
        img_dir=args.out,
        solver_models=solver_models,
        AE_model=ae_model,
        dde_pde_model=get_model_rec,
    )

    trajectory_losses, original_trajectory_losses, trajectory_coordinates = plotter.get_coordinates_and_losses_of_trajectories()
    grid_losses, grid_xx, grid_yy, rec_grid_models = plotter.get_coordinates_and_losses_of_surface()

    for loss_type in plotter.loss_types:
        plotter.loss_type = loss_type
        plotter.plotting(
            trajectory_losses[loss_type], original_trajectory_losses[loss_type], trajectory_coordinates,
            grid_losses[loss_type], grid_xx, grid_yy, rec_grid_models,
        )

    # ---- 4) Save the raw matrices behind the plot: 2D latent trajectory/grid
    #         data, and the original (un-reduced) nD checkpoint parameters ----
    data_dir = os.path.join(args.out, "data")
    os.makedirs(data_dir, exist_ok=True)

    np.save(os.path.join(data_dir, "trajectory_2d.npy"), trajectory_coordinates.numpy())
    np.save(os.path.join(data_dir, "trajectory_original_nd.npy"), plotter.trajectory_original_nd.numpy())
    np.save(os.path.join(data_dir, "trajectory_reconstructed_nd.npy"), plotter.trajectory_reconstructed_nd.numpy())

    loss_type = plotter.loss_types[0]
    np.savez(
        os.path.join(data_dir, "grid_2d.npz"),
        grid_xx=grid_xx.detach().cpu().numpy(),
        grid_yy=grid_yy.detach().cpu().numpy(),
        grid_losses=grid_losses[loss_type].detach().cpu().numpy(),
    )

    print(f"Saved trajectory/grid matrices (2D latent + original/reconstructed nD) to {data_dir}")
    print(f"Landscape visualization saved to {args.out}")


if __name__ == "__main__":
    main()
