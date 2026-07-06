"""Single controlled (PDE, method) run for the chaotic-PDE landscape comparison.

Generalizes experiments/Chaotic/kuramoto_sivashinsky_test_landscape.py into a clean,
deterministic, Comet-free / RL-free harness. One process = one (pde, method) cell of
the comparison matrix. Everything needed for the offline comparison is written to
    <out>/<pde>/<method>/
using the layout documented in experiments/landscape_compare/README.md.

Two tiers of data are produced (see HYPOTHESIS.md):
  * Solution-accuracy tier  (ALL methods, incl. gradient-free Frozen-PINN):
        prediction / reference / error fields on the reference grid + metrics.json
        (relative-L2, MSE, MAE, boundary/IC error, Fourier low/mid/high band error).
  * Loss-landscape tier      (gradient methods only):
        the 2D-embedded loss landscape (trajectory + loss grid + per-loss grids)
        via the existing autoencoder + PlotLossSurface machinery.
    For Frozen-PINN (which solves a *linear* system, so has no weight-space SGD
    landscape) we instead save the frozen feature/projection matrix's singular
    spectrum as the "conditioning / convexity" contrast.

Usage:
    python experiments/landscape_compare/run_experiment.py \
        --pde kuramoto_sivashinsky --method adam_baseline --out runs_landscape_compare
    python experiments/landscape_compare/run_experiment.py \
        --pde grayscott --method frozen --quick
"""
import os
import sys

os.environ.setdefault("DDEBACKEND", "pytorch")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# insert (not append) so the local landscape_visualization/src packages shadow any
# same-named packages that might be installed in site-packages.
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import deepxde as dde

from src.pde.chaotic import KuramotoSivashinskyEquation, GrayScottEquation
from src.pde.burgers import Burgers1D
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import ModelSaverCallback
from src import frozen_pinn
from deepxde.optimizers.config import set_SOAP_options
from landscape_visualization._aux.visualization_model import VisualizationModel
from landscape_visualization._aux.early_stopping_plot import EarlyStopping
from landscape_visualization._aux.plot_loss_surface import PlotLossSurface

dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)


# =========================================================================== #
# PDE registry
# =========================================================================== #
def _make_ks(loss_type, causal_eps, num_causal_buckets):
    pde = KuramotoSivashinskyEquation()
    pde.set_loss_type(loss_type, causal_eps=causal_eps, num_causal_buckets=num_causal_buckets)
    return pde


def _make_gs(loss_type, causal_eps, num_causal_buckets):
    pde = GrayScottEquation()
    pde.set_loss_type(loss_type, causal_eps=causal_eps, num_causal_buckets=num_causal_buckets)
    return pde


def _make_burgers(loss_type, causal_eps, num_causal_buckets):
    return Burgers1D(loss_type=loss_type, causal_eps=causal_eps, num_causal_buckets=num_causal_buckets)


PDE_REGISTRY = {
    "kuramoto_sivashinsky": {
        "factory": _make_ks,
        "spatial_dims": 1,          # inputs: (x, t)
        "output_names": ["u"],
        "hidden_default": "100*5",
    },
    "grayscott": {
        "factory": _make_gs,
        "spatial_dims": 2,          # inputs: (x, y, t)
        "output_names": ["u", "v"],
        "hidden_default": "100*5",
    },
    "burgers1d": {
        "factory": _make_burgers,
        "spatial_dims": 1,          # inputs: (x, t)
        "output_names": ["u"],
        "hidden_default": "100*5",
    },
}

# method -> (loss_type, optimizer schedule). schedule = list of (opt_name, lr, frac)
# where frac is the fraction of --iterations spent in that phase. frozen has no schedule.
METHOD_SPEC = {
    "adam_baseline":  {"loss_type": "origin", "schedule": [("adam", 1e-3, 1.0)]},
    "lbfgs_baseline": {"loss_type": "origin", "schedule": [("adam", 1e-3, 0.5), ("lbfgs", 1.0, 0.5)]},
    "causal":         {"loss_type": "causal", "schedule": [("adam", 1e-3, 1.0)]},
    "soap":           {"loss_type": "origin", "schedule": [("soap", 3e-3, 1.0)]},
    "soap_causal":    {"loss_type": "causal", "schedule": [("soap", 3e-3, 1.0)]},
    "frozen":         {"loss_type": "origin", "schedule": None},
}


# =========================================================================== #
# Model builder (closure, as PlotLossSurface / VisualizationModel expect)
# =========================================================================== #
def build_get_model(pde_name, hidden_layers, loss_type, causal_eps, num_causal_buckets):
    entry = PDE_REGISTRY[pde_name]

    def get_model():
        pde = entry["factory"](loss_type, causal_eps, num_causal_buckets)
        layers = [pde.input_dim] + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)) + [pde.output_dim]
        net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()

        loss_weights = np.ones(pde.num_loss, dtype=float)
        for i, c in enumerate(pde.loss_config):
            if c.get("type", "") in ("boundary", "initial", "ic"):
                loss_weights[i] = 100.0
        model = pde.create_model(net)
        return model, loss_weights

    return get_model


# =========================================================================== #
# Optimizer compilation (mirrors rl_trainer._build_torch_optimizer)
# =========================================================================== #
def compile_optimizer(model, opt_name, lr, loss_weights):
    if opt_name == "adam":
        model.compile(torch.optim.Adam(model.net.parameters(), lr=lr), loss_weights=loss_weights)
    elif opt_name == "lbfgs":
        opt = torch.optim.LBFGS(model.net.parameters(), lr=lr, line_search_fn="strong_wolfe", max_iter=10)
        model.compile(opt, loss_weights=loss_weights)
    elif opt_name == "soap":
        set_SOAP_options(lr=lr)
        model.compile("SOAP", loss_weights=loss_weights)
    else:
        raise ValueError(f"Unknown optimizer '{opt_name}'")


# =========================================================================== #
# Solution-accuracy tier (works for gradient models AND Frozen-PINN)
# =========================================================================== #
def _get_ref(pde):
    """Return (coords, values) from pde.ref_data with NaN rows dropped."""
    data = pde.ref_data
    mask = ~np.isnan(data).any(axis=1)
    coords = data[mask, : pde.input_dim].astype(float)
    values = data[mask, pde.input_dim :].astype(float)
    return coords, values


def _relative_l2(pred, ref):
    denom = np.sqrt(np.mean(ref ** 2))
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / denom) if denom > 0 else float("nan")


def _reconstruct_tensor_grid(coords, spatial_dims):
    """Try to view a point list as a tensor grid: return (axes, shape, order_index) or None.

    axes: list of sorted unique values per input dim (spatial dims first, time last).
    order_index: indices that sort `coords` into C-order over those axes.
    """
    ndim = coords.shape[1]
    axes = [np.unique(coords[:, d]) for d in range(ndim)]
    if int(np.prod([len(a) for a in axes])) != coords.shape[0]:
        return None  # not a clean tensor grid
    # lexsort with the LAST axis varying fastest -> C-order for shape (len(axis0), ..)
    keys = tuple(coords[:, d] for d in reversed(range(ndim)))
    order = np.lexsort(keys)
    shape = tuple(len(a) for a in axes)
    return axes, shape, order


def _spectral_band_error(coords, pred, ref, spatial_dims, n_bands=3):
    """Fourier low/mid/high band relative error over the spatial axes, summed over time.

    Returns dict {band: rel_error} per output component or NaNs if the reference points
    do not form a clean tensor grid. Chaotic fine structure lives in the mid/high bands.
    """
    out = {f"band{b}": float("nan") for b in range(n_bands)}
    recon = _reconstruct_tensor_grid(coords, spatial_dims)
    if recon is None:
        return out
    axes, shape, order = recon
    err = (pred - ref)[order]
    n_out = ref.shape[1]
    # spatial axes are the first `spatial_dims`, time (if any) is the last input dim.
    spatial_axes = tuple(range(spatial_dims))

    band_energy = np.zeros(n_bands)
    for c in range(n_out):
        e_grid = err[:, c].reshape(shape)
        E = np.fft.fftn(e_grid, axes=spatial_axes)         # FFT over spatial axes only
        freqs = [np.fft.fftfreq(shape[a]) * shape[a] for a in spatial_axes]
        mesh = np.meshgrid(*freqs, indexing="ij")
        kr = np.sqrt(np.sum([m ** 2 for m in mesh], axis=0))  # radial wavenumber (spatial shape)
        kmax = kr.max() if kr.max() > 0 else 1.0
        edges = np.linspace(0, kmax + 1e-9, n_bands + 1)
        power = np.abs(E) ** 2
        non_spatial = tuple(a for a in range(power.ndim) if a not in spatial_axes)
        power_sp = power.sum(axis=non_spatial) if non_spatial else power  # collapse time
        for b in range(n_bands):
            band_energy[b] += float(power_sp[(kr >= edges[b]) & (kr < edges[b + 1])].sum())
    # report the FRACTION of the total error energy in each band (sums to 1) -- directly
    # answers "where in scale does the error live" (H4). low band = coarse, high = fine.
    total = float(band_energy.sum()) or 1e-30
    for b in range(n_bands):
        out[f"band{b}"] = float(band_energy[b] / total)
    return out


def evaluate_solution(pde, predict_fn, spatial_dims, output_names):
    """Predict on the reference grid and compute the solution-accuracy metrics + fields."""
    coords, values = _get_ref(pde)
    pred = np.asarray(predict_fn(coords), dtype=float).reshape(values.shape)

    metrics = {}
    metrics["relative_l2"] = _relative_l2(pred, values)
    metrics["mse"] = float(np.mean((pred - values) ** 2))
    metrics["mae"] = float(np.mean(np.abs(pred - values)))
    metrics["max_error"] = float(np.max(np.abs(pred - values)))
    for c, name in enumerate(output_names):
        metrics[f"relative_l2_{name}"] = _relative_l2(pred[:, c : c + 1], values[:, c : c + 1])

    # IC error (t == t0): time is the last input dim for these time PDEs.
    has_time = pde.input_dim == spatial_dims + 1
    if has_time:
        t = coords[:, -1]
        ic = np.isclose(t, t.min(), atol=1e-9)
        if ic.any():
            metrics["ic_relative_l2"] = _relative_l2(pred[ic], values[ic])

    # spatial-boundary error via bbox extremes (excluding the IC slice)
    bbox = np.asarray(pde.bbox, dtype=float)
    bmask = np.zeros(coords.shape[0], dtype=bool)
    for d in range(spatial_dims):
        bmask |= np.isclose(coords[:, d], bbox[2 * d], atol=1e-9) | np.isclose(coords[:, d], bbox[2 * d + 1], atol=1e-9)
    if has_time:
        bmask &= ~np.isclose(coords[:, -1], coords[:, -1].min(), atol=1e-9)
    if bmask.any():
        metrics["boundary_relative_l2"] = _relative_l2(pred[bmask], values[bmask])

    bands = _spectral_band_error(coords, pred, values, spatial_dims)
    metrics["fourier_low"] = bands["band0"]
    metrics["fourier_mid"] = bands["band1"]
    metrics["fourier_high"] = bands["band2"]

    fields = {"coords": coords.astype(np.float32),
              "pred": pred.astype(np.float32),
              "ref": values.astype(np.float32),
              "abs_error": np.abs(pred - values).astype(np.float32)}
    return metrics, fields


# =========================================================================== #
# Frozen-PINN dispatch
# =========================================================================== #
def run_frozen(pde_name, quick, seed=0):
    """Run the appropriate frozen solver; return (predict_fn(coords)->(N,outdim), extra).

    `seed` drives the frozen random features for grayscott/burgers1d (so repeated runs with
    different --seed values give a genuine spread). Kuramoto-Sivashinsky uses a deterministic
    Fourier basis with no randomness, so it has no seed and repeats are bit-identical -- that
    is expected, not a bug.
    """
    if pde_name == "kuramoto_sivashinsky":
        # NOTE: keep num_collocation >> 2*num_modes (de-aliasing of u u_x) and rtol loose in
        # quick mode -- a tight rtol on this stiff rescaled KS makes the implicit solver crawl.
        kw = dict(num_modes=24, num_collocation=256, num_time_eval=26, rtol=1e-4) if quick \
            else dict(num_modes=64, num_collocation=512, num_time_eval=251)
        sol, feats, predict, diag = frozen_pinn.solve_kuramoto_sivashinsky_frozen(**kw)

        def predict_fn(coords):
            return predict(coords[:, 0], coords[:, 1]).reshape(-1, 1)

    elif pde_name == "grayscott":
        kw = dict(num_features=60, num_collocation_per_dim=12, rtol=1e-4) if quick \
            else dict(num_features=300, num_collocation_per_dim=32)
        sol, feats, predict, diag = frozen_pinn.solve_grayscott_frozen(seed=seed, **kw)

        def predict_fn(coords):
            return predict(coords[:, 0], coords[:, 1], coords[:, 2])

    elif pde_name == "burgers1d":
        kw = dict(num_features=200, num_collocation=400, num_time_eval=51) if quick \
            else dict(num_features=2000, num_collocation=4000, num_time_eval=201)
        res = frozen_pinn.solve_burgers1d_frozen(seed=seed, **kw)
        sol, feats, predict = res[0], res[1], res[2]
        diag = res[3] if len(res) > 3 else {
            "singular_values": np.array([np.nan]), "condition_number": float("nan"),
            "num_features": kw["num_features"], "basis": "tanh_random"}

        def predict_fn(coords):
            return predict(coords[:, 0], coords[:, 1]).reshape(-1, 1)

    else:
        raise ValueError(f"No frozen solver registered for '{pde_name}'")

    coeffs = None
    if sol is not None and getattr(sol, "t", None) is not None:
        coeffs = {"t": np.asarray(sol.t, dtype=np.float32), "coefficients": np.asarray(sol.y, dtype=np.float32)}
    extra = {"diagnostics": diag, "coeffs": coeffs}
    return predict_fn, extra


# =========================================================================== #
# Landscape tier (gradient methods) -- generalizes test_landscape's block
# =========================================================================== #
def build_landscape(solver_models, get_model_rec, landscape_dir, ae_epochs, grid_xnum, device):
    os.makedirs(landscape_dir, exist_ok=True)
    n = len(solver_models)
    layers_AE = [991, 125, 15]
    batch_size = min(32, max(2, n))

    AE_model_params = dict(
        mode="NN", num_of_layers=3, layers_AE=layers_AE, num_models=None, from_last=False,
        prefix="model-", every_nth=1, grid_step=0.1, d_max_latent=2, anchor_mode="circle",
        rec_weight=10000.0, anchor_weight=0.0, lastzero_weight=0.0, polars_weight=0.0,
        wellspacedtrajectory_weight=0.0, gridscaling_weight=0.0, device=device,
    )
    vis_model = VisualizationModel(**AE_model_params)
    ae_model = vis_model.train(
        lr=5e-4, cosine_scheduler_patience=1200, epochs=ae_epochs, every_epoch=100,
        batch_size=batch_size, resume=True, callbacks=[EarlyStopping(patience=4000)],
        solver_models=solver_models,
    )

    loss_types = ["loss_total", "loss_oper", "loss_bnd"]
    plotter = PlotLossSurface(
        loss_types=loss_types, every_nth=1, num_of_layers=3, layers_AE=layers_AE,
        batch_size=batch_size, num_models=None, from_last=False, prefix="model-",
        loss_name="loss_total", x_range=[-1.25, 1.25, grid_xnum], vmax=-1.0, vmin=-1.0,
        vlevel=30.0, key_models=None, key_modelnames=None, density_type="CKA", density_p=2,
        density_vmax=-1, density_vmin=-1, colorFromGridOnly=True, img_dir=landscape_dir,
        solver_models=solver_models, AE_model=ae_model, dde_pde_model=get_model_rec,
    )

    traj_losses, orig_traj_losses, traj_coords = plotter.get_coordinates_and_losses_of_trajectories()
    grid_losses, grid_xx, grid_yy, rec_grid_models = plotter.get_coordinates_and_losses_of_surface()

    for lt in loss_types:
        plotter.loss_type = lt
        plotter.plotting(traj_losses[lt], orig_traj_losses[lt], traj_coords,
                         grid_losses[lt], grid_xx, grid_yy, rec_grid_models)

    np.save(os.path.join(landscape_dir, "trajectory_2d.npy"), traj_coords.numpy())
    np.save(os.path.join(landscape_dir, "trajectory_original_nd.npy"), plotter.trajectory_original_nd.numpy())
    np.save(os.path.join(landscape_dir, "trajectory_reconstructed_nd.npy"), plotter.trajectory_reconstructed_nd.numpy())

    gx = grid_xx.detach().cpu().numpy()
    gy = grid_yy.detach().cpu().numpy()
    # main grid (loss_total) for the comparison script + all loss grids together
    np.savez(os.path.join(landscape_dir, "grid_2d.npz"),
             grid_xx=gx, grid_yy=gy, grid_losses=grid_losses["loss_total"].detach().cpu().numpy())
    np.savez(os.path.join(landscape_dir, "grid_losses_all.npz"), grid_xx=gx, grid_yy=gy,
             **{lt: grid_losses[lt].detach().cpu().numpy() for lt in loss_types})

    # per-trajectory-point losses (for loss-vs-error-along-trajectory analysis)
    def _to_np(v):
        return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    np.savez(os.path.join(landscape_dir, "trajectory_losses.npz"),
             **{lt: _to_np(traj_losses[lt]) for lt in loss_types})
    return {"n_checkpoints": n, "grid_shape": list(gx.shape)}


# =========================================================================== #
# Main
# =========================================================================== #
def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pde", required=True, choices=list(PDE_REGISTRY.keys()))
    parser.add_argument("--method", required=True, choices=list(METHOD_SPEC.keys()))
    parser.add_argument("--out", type=str, default=os.path.join(PROJECT_ROOT, "runs_landscape_compare"))
    parser.add_argument("--hidden-layers", type=str, default=None, help="e.g. 100*5 (default: per-PDE)")
    parser.add_argument("--iterations", type=int, default=5000, help="total gradient iterations")
    parser.add_argument("--n-save-models", type=int, default=10, help="checkpoints along the trajectory")
    parser.add_argument("--display-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--causal-eps", type=float, default=1.0)
    parser.add_argument("--num-causal-buckets", type=int, default=32)
    parser.add_argument("--ae-epochs", type=int, default=10000, help="autoencoder training epochs")
    parser.add_argument("--grid-xnum", type=int, default=25, help="landscape grid resolution")
    parser.add_argument("--no-landscape", action="store_true", help="skip the landscape tier (gradient methods)")
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test settings")
    args = parser.parse_args()

    if args.quick:
        args.iterations = min(args.iterations, 200)
        args.n_save_models = min(args.n_save_models, 5)
        args.ae_epochs = min(args.ae_epochs, 400)
        args.grid_xnum = min(args.grid_xnum, 9)

    entry = PDE_REGISTRY[args.pde]
    spec = METHOD_SPEC[args.method]
    hidden_layers = args.hidden_layers or entry["hidden_default"]
    spatial_dims = entry["spatial_dims"]
    output_names = entry["output_names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = os.path.join(args.out, args.pde, args.method)
    os.makedirs(run_dir, exist_ok=True)
    solution_dir = os.path.join(run_dir, "solution")
    os.makedirs(solution_dir, exist_ok=True)

    config = {
        "pde": args.pde, "method": args.method, "loss_type": spec["loss_type"],
        "hidden_layers": hidden_layers, "iterations": args.iterations,
        "n_save_models": args.n_save_models, "seed": args.seed, "quick": args.quick,
        "causal_eps": args.causal_eps, "num_causal_buckets": args.num_causal_buckets,
        "ae_epochs": args.ae_epochs, "grid_xnum": args.grid_xnum, "device": device,
        "git_commit": git_commit(), "started_at": datetime.now().isoformat(),
    }
    print(f"\n=== {args.pde} / {args.method} ===\n{json.dumps(config, indent=2)}\n")
    t_start = time.time()

    metrics = {}
    landscape_info = {}

    if args.method == "frozen":
        # ---- gradient-free path ----
        predict_fn, extra = run_frozen(args.pde, args.quick, seed=args.seed)
        pde_eval = entry["factory"](spec["loss_type"], args.causal_eps, args.num_causal_buckets)
        sol_metrics, fields = evaluate_solution(pde_eval, predict_fn, spatial_dims, output_names)
        metrics.update(sol_metrics)

        frozen_dir = os.path.join(run_dir, "frozen")
        os.makedirs(frozen_dir, exist_ok=True)
        diag = extra["diagnostics"]
        np.savez(os.path.join(frozen_dir, "feature_spectrum.npz"),
                 singular_values=np.asarray(diag["singular_values"], dtype=np.float64))
        if extra["coeffs"] is not None:
            np.savez(os.path.join(frozen_dir, "coefficients.npz"), **extra["coeffs"])
        metrics["condition_number"] = float(diag.get("condition_number", float("nan")))
        metrics["num_features"] = int(diag.get("num_features", 0))
        metrics["frozen_basis"] = diag.get("basis", "")
        metrics["integrator_success"] = bool(diag.get("integrator_success", True))
    else:
        # ---- gradient path ----
        get_model = build_get_model(args.pde, hidden_layers, spec["loss_type"],
                                    args.causal_eps, args.num_causal_buckets)
        get_model_rec = build_get_model(args.pde, hidden_layers, spec["loss_type"],
                                        args.causal_eps, args.num_causal_buckets)
        model, loss_weights = get_model()

        n_phases = len(spec["schedule"])
        saves_per_phase = max(1, args.n_save_models // n_phases)
        solver_models = []
        for (opt_name, lr, frac) in spec["schedule"]:
            phase_iters = max(1, int(round(args.iterations * frac)))
            compile_optimizer(model, opt_name, lr, loss_weights)
            saver = ModelSaverCallback(total_iterations=phase_iters, n_save_models=saves_per_phase)
            print(f"--- phase: {opt_name} lr={lr} iters={phase_iters} ---")
            model.train(iterations=phase_iters, display_every=args.display_every,
                        callbacks=[saver], model_save_path=run_dir, save_model=False)
            solver_models.extend(saver.saved_models)

        # final-model solution metrics
        def predict_fn(coords):
            return model.predict(coords.astype(np.float32))
        pde_eval = model.pde
        sol_metrics, fields = evaluate_solution(pde_eval, predict_fn, spatial_dims, output_names)
        metrics.update(sol_metrics)

        # loss history (per display step, per component) from the accumulated losshistory
        try:
            lh = model.losshistory
            steps = np.asarray(lh.steps).reshape(-1, 1)
            loss_train = np.asarray(lh.loss_train)
            np.savetxt(os.path.join(run_dir, "loss_history.csv"),
                       np.hstack([steps, loss_train]), delimiter=",",
                       header="step," + ",".join(f"loss_{i}" for i in range(loss_train.shape[1])), comments="")
        except Exception as e:
            print(f"[warn] could not save loss history: {e}")

        # per-checkpoint trajectory error (rel-L2 at each landscape trajectory point)
        try:
            coords, values = _get_ref(pde_eval)
            sub = np.random.choice(coords.shape[0], size=min(4000, coords.shape[0]), replace=False)
            rows = []
            base_net_state = {k: v.clone() for k, v in model.net.state_dict().items()}
            for i, net in enumerate(solver_models):
                model.net.load_state_dict(net.state_dict())
                p = model.predict(coords[sub].astype(np.float32)).reshape(values[sub].shape)
                rows.append([i, _relative_l2(p, values[sub])])
            model.net.load_state_dict(base_net_state)
            np.savetxt(os.path.join(run_dir, "trajectory_error.csv"), np.asarray(rows),
                       delimiter=",", header="checkpoint_idx,relative_l2", comments="")
        except Exception as e:
            print(f"[warn] could not save trajectory error: {e}")

        # landscape tier
        if not args.no_landscape and len(solver_models) >= 2:
            checkpoints_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            for i, net in enumerate(solver_models):
                torch.save(net.state_dict(), os.path.join(checkpoints_dir, f"model-{i:03d}.pt"))
            try:
                landscape_info = build_landscape(
                    solver_models, get_model_rec, os.path.join(run_dir, "landscape"),
                    args.ae_epochs, args.grid_xnum, device)
            except Exception as e:
                print(f"[warn] landscape build failed: {e}")
                landscape_info = {"error": str(e)}

    # ---- save fields + metrics + config ----
    np.savez(os.path.join(solution_dir, "fields.npz"), **fields)
    metrics["wall_clock_sec"] = round(time.time() - t_start, 2)
    config["finished_at"] = datetime.now().isoformat()
    config.update({"landscape": landscape_info})
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[done] {args.pde}/{args.method}  relative_l2={metrics.get('relative_l2'):.4e}  "
          f"({metrics['wall_clock_sec']}s)\n  -> {run_dir}")
    return metrics


if __name__ == "__main__":
    main()
