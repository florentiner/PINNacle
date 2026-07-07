"""SHARED loss & TRUE-ERROR landscape: all methods overlaid in ONE 2D space per (pde, seed).

The per-run landscapes built by run_experiment.py each train their own autoencoder, so two
methods' maps are different embeddings and cannot be compared point-for-point. This script
builds the comparison the experiment is really after:

  For each (pde, seed): take the checkpoints of EVERY gradient method (they share the same
  architecture and the same seed-derived initial weights -- checkpoint 0 is the common start),
  train ONE autoencoder on the union, evaluate ONE loss grid (plain/origin total loss as the
  neutral yardstick) AND one TRUE-ERROR grid (each decoded grid net's relative-L2 against the
  reference), then overlay every method's trajectory on those two shared maps.

This is the artifact that answers "how do the methods behave differently on the SAME error
landscape, starting from the SAME initial weights".

Needs torch/deepxde (run on the results machine, or CPU with --ae-epochs lowered). Outputs to
<runs>/shared_landscape/<pde>_seed<seed>/:
    shared_grid.npz        grid_xx, grid_yy, loss_grid, error_grid
    trajectories.npz       per-method 2D coords + per-checkpoint true error
    loss_map.pdf           shared log-loss contours + all trajectories
    error_map.pdf          shared TRUE-ERROR contours + all trajectories
    endpoints.json         final loss/error per method on the shared maps

Usage (after run_all.py):
    python experiments/landscape_compare/shared_landscape.py --runs runs_landscape_compare
"""
import os
import sys

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_experiment as rex
from landscape_visualization._aux.visualization_model import VisualizationModel
from landscape_visualization._aux.early_stopping_plot import EarlyStopping
from landscape_visualization._aux.plot_loss_surface import PlotLossSurface

METHOD_COLORS = {"origin": "tab:red", "causal": "tab:blue", "soap": "tab:orange",
                 "soap_causal": "tab:green", "best_practice": "tab:cyan",
                 "adam_baseline": "tab:brown", "lbfgs_baseline": "tab:purple"}
GRADIENT_METHODS = list(METHOD_COLORS)


def _rel_l2(pred, ref):
    d = np.sqrt(np.mean(ref ** 2))
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / d) if d > 0 else float("nan")


def load_run_checkpoints(run_dir, template_net):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    nets = []
    for p in sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt")):
        sd = torch.load(os.path.join(ckpt_dir, p), map_location="cpu")
        net = copy.deepcopy(template_net)
        net.load_state_dict(sd)
        nets.append(net)
    return nets


def build_shared(pde, seed, seed_dir, methods, args, out_root):
    # -- consistency: all methods must share architecture + embedding + hidden layers --
    cfgs = {}
    for m in methods:
        cp = os.path.join(seed_dir, pde, m, "config.json")
        if os.path.exists(cp):
            cfgs[m] = json.load(open(cp))
    if len(cfgs) < 2:
        print(f"[skip] {pde}@seed{seed}: fewer than 2 gradient runs with configs")
        return
    # Methods with a different parameter space (e.g. best_practice's modified MLP) cannot live
    # in the same autoencoder embedding as the plain-FNN methods: drop them with a note instead
    # of failing the whole build, then require the remaining runs to match exactly.
    archs = {m: c.get("arch", "fnn") for m, c in cfgs.items()}
    majority_arch = max(set(archs.values()), key=list(archs.values()).count)
    dropped = [m for m, a in archs.items() if a != majority_arch]
    if dropped:
        print(f"[note] {pde}@seed{seed}: excluding {dropped} (arch != '{majority_arch}'); a "
              f"shared landscape requires one parameter space")
        for m in dropped:
            cfgs.pop(m)
    if len(cfgs) < 2:
        print(f"[skip] {pde}@seed{seed}: fewer than 2 same-architecture gradient runs")
        return
    fmodes = {c.get("fourier_modes", 0) for c in cfgs.values()}
    hiddens = {c.get("hidden_layers") for c in cfgs.values()}
    if len(fmodes) > 1 or len(hiddens) > 1:
        print(f"[skip] {pde}@seed{seed}: inconsistent architectures across methods "
              f"(fourier_modes={fmodes}, hidden={hiddens}) -- rerun with matching settings")
        return
    fourier_modes, hidden = fmodes.pop(), hiddens.pop()

    # -- template model (ORIGIN loss = neutral shared yardstick for the loss grid) --
    get_model = rex.build_get_model(pde, hidden, "origin", 1.0, 32, fourier_modes=fourier_modes,
                                    arch=majority_arch)

    # -- union of checkpoints, remembering per-method slices --
    tmpl_model, _ = get_model()
    union, slices = [], {}
    for m in sorted(cfgs):
        nets = load_run_checkpoints(os.path.join(seed_dir, pde, m), tmpl_model.net)
        if len(nets) >= 2:
            slices[m] = (len(union), len(union) + len(nets))
            union += nets
    if len(slices) < 2:
        print(f"[skip] {pde}@seed{seed}: fewer than 2 methods with checkpoints")
        return
    print(f"[shared] {pde}@seed{seed}: union of {len(union)} checkpoints from {list(slices)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(out_root, f"{pde}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    # -- ONE autoencoder on the union --
    layers_AE = [991, 125, 15]
    batch = min(32, max(2, len(union)))
    vis = VisualizationModel(mode="NN", num_of_layers=3, layers_AE=layers_AE, num_models=None,
                             from_last=False, prefix="model-", every_nth=1, grid_step=0.1,
                             d_max_latent=2, anchor_mode="circle", rec_weight=10000.0,
                             anchor_weight=0.0, lastzero_weight=0.0, polars_weight=0.0,
                             wellspacedtrajectory_weight=0.0, gridscaling_weight=0.0, device=device)
    ae = vis.train(lr=5e-4, cosine_scheduler_patience=1200, epochs=args.ae_epochs, every_epoch=100,
                   batch_size=batch, resume=True, callbacks=[EarlyStopping(patience=4000)],
                   solver_models=union)

    # -- ONE loss grid + decoded grid nets, trajectories of all methods in the same space --
    plotter = PlotLossSurface(
        loss_types=["loss_total"], every_nth=1, num_of_layers=3, layers_AE=layers_AE,
        batch_size=batch, num_models=None, from_last=False, prefix="model-",
        loss_name="loss_total", x_range=[-1.25, 1.25, args.grid_xnum], vmax=-1.0, vmin=-1.0,
        vlevel=30.0, key_models=None, key_modelnames=None, density_type="CKA", density_p=2,
        density_vmax=-1, density_vmin=-1, colorFromGridOnly=True, img_dir=out_dir,
        solver_models=union, AE_model=ae, dde_pde_model=get_model,
    )
    traj_losses, orig_traj_losses, traj_coords = plotter.get_coordinates_and_losses_of_trajectories()
    grid_losses, grid_xx, grid_yy, rec_grid_models = plotter.get_coordinates_and_losses_of_surface()
    loss_grid = grid_losses["loss_total"].detach().cpu().numpy().reshape(grid_xx.shape)
    gx, gy = grid_xx.cpu().numpy(), grid_yy.cpu().numpy()

    # -- reference subsample for TRUE error --
    ref = tmpl_model.pde.ref_data
    ref = ref[~np.isnan(ref).any(axis=1)]
    rng = np.random.default_rng(0)
    sub = rng.choice(ref.shape[0], size=min(args.n_ref, ref.shape[0]), replace=False)
    in_dim = tmpl_model.pde.input_dim
    X = ref[sub, :in_dim].astype(np.float32)
    Y = ref[sub, in_dim:].astype(np.float64)

    eval_model = plotter.dde_pde_model  # compiled; net architecture matches the union

    def error_of_flat(vec):
        from landscape_visualization._aux.utils import repopulate_model
        net = repopulate_model(vec, copy.deepcopy(eval_model.net))
        eval_model.net.load_state_dict(net.state_dict())
        pred = eval_model.predict(X).reshape(Y.shape).astype(np.float64)
        return _rel_l2(pred, Y)

    # -- TRUE-ERROR grid (each decoded grid point -> a network -> rel-L2 vs reference) --
    n_grid = rec_grid_models.shape[0]
    err_grid = np.empty(n_grid)
    for i in range(n_grid):
        err_grid[i] = error_of_flat(rec_grid_models[i].detach())
    err_grid = err_grid.reshape(gx.shape)

    # -- per-method trajectory errors (original, un-reconstructed checkpoints) --
    original_nd = plotter.trajectory_original_nd
    coords2d = traj_coords.numpy()
    traj_out = {}
    endpoints = {}
    for m, (a, b) in slices.items():
        errs = np.array([error_of_flat(original_nd[i]) for i in range(a, b)])
        traj_out[f"{m}_coords"] = coords2d[a:b]
        traj_out[f"{m}_errors"] = errs
        losses_m = traj_losses["loss_total"][a:b]
        losses_m = losses_m.detach().cpu().numpy() if isinstance(losses_m, torch.Tensor) else np.asarray(losses_m)
        endpoints[m] = {"final_loss": float(losses_m[-1]), "final_error": float(errs[-1]),
                        "start_error": float(errs[0]), "n_checkpoints": int(b - a)}

    np.savez(os.path.join(out_dir, "shared_grid.npz"),
             grid_xx=gx, grid_yy=gy, loss_grid=loss_grid, error_grid=err_grid)
    np.savez(os.path.join(out_dir, "trajectories.npz"), **traj_out)
    with open(os.path.join(out_dir, "endpoints.json"), "w") as f:
        json.dump(endpoints, f, indent=2)

    # -- the two overlay maps --
    for name, Z, label in [("loss_map", np.log10(np.clip(loss_grid, 1e-30, None)), "log10 total loss (origin yardstick)"),
                           ("error_map", err_grid, "TRUE relative-L2 of decoded network")]:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        cs = ax.contourf(gx, gy, Z, levels=30, cmap="viridis")
        fig.colorbar(cs, ax=ax, label=label)
        for m, (a, b) in slices.items():
            c2 = coords2d[a:b]
            ax.plot(c2[:, 0], c2[:, 1], "-o", ms=3.5, lw=1.4, color=METHOD_COLORS.get(m), label=m)
            ax.plot(c2[-1, 0], c2[-1, 1], "*", ms=14, color=METHOD_COLORS.get(m),
                    markeredgecolor="k", zorder=6)
        first = next(iter(slices.values()))
        ax.plot(coords2d[first[0], 0], coords2d[first[0], 1], "ks", ms=9, zorder=7,
                label="shared init")
        ax.set_title(f"{pde} seed {seed}: SHARED {'loss' if name == 'loss_map' else 'TRUE-ERROR'} "
                     f"landscape, all methods from the same init")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
        plt.close(fig)
        print(f"[fig] {os.path.join(out_dir, name + '.pdf')}")

    print(f"[endpoints] {json.dumps(endpoints, indent=2)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=str, default=os.path.join(PROJECT_ROOT, "runs_landscape_compare"))
    parser.add_argument("--pdes", nargs="+", default=["kuramoto_sivashinsky", "grayscott"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="default: all seed_* dirs")
    parser.add_argument("--methods", nargs="+", default=["origin", "causal", "soap", "soap_causal"])
    parser.add_argument("--ae-epochs", type=int, default=10000)
    parser.add_argument("--grid-xnum", type=int, default=25)
    parser.add_argument("--n-ref", type=int, default=2000)
    args = parser.parse_args()

    if args.seeds is not None:
        seed_dirs = [(s, os.path.join(args.runs, f"seed_{s}")) for s in args.seeds]
    else:
        seed_dirs = [(int(n[5:]), os.path.join(args.runs, n)) for n in sorted(os.listdir(args.runs))
                     if n.startswith("seed_") and os.path.isdir(os.path.join(args.runs, n))]
        if not seed_dirs:
            seed_dirs = [(None, args.runs)]  # flat single-seed layout

    out_root = os.path.join(args.runs, "shared_landscape")
    os.makedirs(out_root, exist_ok=True)
    for pde in args.pdes:
        for seed, sdir in seed_dirs:
            if os.path.isdir(os.path.join(sdir, pde)):
                build_shared(pde, seed, sdir, args.methods, args, out_root)
    print(f"\nShared landscapes under: {out_root}")


if __name__ == "__main__":
    main()
