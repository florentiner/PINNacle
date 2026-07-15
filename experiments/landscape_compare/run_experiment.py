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
import copy
import itertools
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
from deepxde.nn.pytorch.nn import NN as DDENNBase
from landscape_visualization._aux.visualization_model import VisualizationModel
from landscape_visualization._aux.early_stopping_plot import EarlyStopping
from landscape_visualization._aux.plot_loss_surface import PlotLossSurface

REAL = np.float32  # numpy dtype for every predict/save cast; switched by --float64


def set_precision(float64):
    """Set the global training precision. Must run before any model/net construction."""
    global REAL
    if float64:
        REAL = np.float64
        dde.config.set_default_float("float64")
        torch.set_default_dtype(torch.float64)
    else:
        REAL = np.float32
        dde.config.set_default_float("float32")
        torch.set_default_dtype(torch.float32)


set_precision(False)


# =========================================================================== #
# PDE registry
# =========================================================================== #
def _make_ks(loss_type, causal_eps, num_causal_buckets, time_range=None, ic_func=None):
    bbox = [0, 2 * np.pi] + (list(time_range) if time_range else [0, 1])
    pde = KuramotoSivashinskyEquation(bbox=bbox, ic_func=ic_func)
    pde.set_loss_type(loss_type, causal_eps=causal_eps, num_causal_buckets=num_causal_buckets)
    return pde


def _make_gs(loss_type, causal_eps, num_causal_buckets, time_range=None, ic_func=None):
    bbox = [-1, 1, -1, 1] + (list(time_range) if time_range else [0, 200])
    pde = GrayScottEquation(bbox=bbox, ic_func=ic_func)
    pde.set_loss_type(loss_type, causal_eps=causal_eps, num_causal_buckets=num_causal_buckets)
    return pde


def _make_burgers(loss_type, causal_eps, num_causal_buckets, time_range=None, ic_func=None):
    if time_range is not None or ic_func is not None:
        raise ValueError("time-marching is only supported for the chaotic PDEs (KS, Gray-Scott)")
    return Burgers1D(loss_type=loss_type, causal_eps=causal_eps, num_causal_buckets=num_causal_buckets)


PDE_REGISTRY = {
    "kuramoto_sivashinsky": {
        "factory": _make_ks,
        "spatial_dims": 1,          # inputs: (x, t)
        "output_names": ["u"],
        "hidden_default": "100*5",
        "time_bbox": (0.0, 1.0),
        # Reference solution is exactly periodic on [0, 2pi] (verified: edge rms diff = 0), but
        # the PINNacle PDE definition imposes ONLY the IC -- no spatial BC at all, so the problem
        # the network sees is ill-posed (a 4th-order PDE with no BC has non-periodic solutions
        # that legitimately zero the residual while diverging from the periodic reference).
        # Fix, following the causal paper (arXiv:2203.07404): enforce periodicity EXACTLY with a
        # Fourier feature embedding of x (period 2pi -> base wavenumber 1).
        "periodic": {"base_k": 1.0, "modes_default": 10},
    },
    "grayscott": {
        "factory": _make_gs,
        "spatial_dims": 2,          # inputs: (x, y, t)
        "output_names": ["u", "v"],
        "hidden_default": "100*5",
        "time_bbox": (0.0, 200.0),
        # Same missing-BC issue as KS; reference is periodic on [-1,1]^2 (period 2 per axis ->
        # base wavenumber pi).
        "periodic": {"base_k": float(np.pi), "modes_default": 5},
    },
    "burgers1d": {
        "factory": _make_burgers,
        "spatial_dims": 1,          # inputs: (x, t)
        "output_names": ["u"],
        "hidden_default": "100*5",
        "time_bbox": (0.0, 1.0),
        "periodic": None,           # Dirichlet BCs are already in the PDE definition
    },
}


def make_periodic_transform(spatial_dims, base_k, n_modes):
    """Input transform (x_1..x_s, t) -> (cos/sin(k x_1).., cos/sin(k x_s).., t), k = base_k*1..K.

    All spatial dependence goes through period-(2pi/base_k) features, so the network output is
    exactly periodic in every spatial dimension, to all derivative orders -- the hard-constraint
    equivalent of periodic BCs used by Wang et al. 2022 for Kuramoto-Sivashinsky.
    """
    def transform(x):
        ks = torch.arange(1, n_modes + 1, device=x.device, dtype=x.dtype) * base_k
        feats = []
        for d in range(spatial_dims):
            ang = x[:, d:d + 1] * ks
            feats.append(torch.cos(ang))
            feats.append(torch.sin(ang))
        feats.append(x[:, spatial_dims:])
        return torch.cat(feats, dim=1)
    return transform


class RWFLinear(torch.nn.Module):
    """Random Weight Factorization (Expert's Guide arXiv:2308.08468, App. their rec.: mu=1.0,
    sigma=0.1): W = diag(exp(s)) @ V with s, V trained jointly; at init exp(s)*V equals the
    standard init, so RWF changes the *parameterization* (and thus the loss geometry), not the
    initial function. seed_init_network knows how to (re)initialize this deterministically."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.s = torch.nn.Parameter(torch.empty(out_features))
        self.V = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self, generator=None):
        w = torch.empty(self.out_features, self.in_features)
        torch.nn.init.xavier_normal_(w) if generator is None else w.copy_(
            torch.randn(w.shape, generator=generator) * (2.0 / (self.in_features + self.out_features)) ** 0.5)
        s = torch.ones(self.out_features) if generator is None else (
            1.0 + 0.1 * torch.randn(self.out_features, generator=generator))
        with torch.no_grad():
            self.s.copy_(s.to(self.s.dtype))
            self.V.copy_((w / torch.exp(s).unsqueeze(1)).to(self.V.dtype))
            self.bias.zero_()

    def forward(self, x):
        return x @ (torch.exp(self.s).unsqueeze(1) * self.V).T + self.bias


class ModifiedMLP(DDENNBase):
    """Modified MLP of Wang, Teng & Perdikaris 2021 (arXiv:2001.04536), the architecture the
    Expert's Guide (arXiv:2308.08468) and the causal paper use for their chaotic-PDE results.

    Two encoders U = act(W_u x + b_u), V = act(W_v x + b_v) gate every hidden layer:
        H_1 = act(L_0 x);   H_{l+1} = (1 - Z_l) * U + Z_l * V,  Z_l = act(L_l H_l)
    with a plain linear output layer. The hidden chain lives in `self.linears` (like DeepXDE's
    FNN) so the landscape machinery's generic flatten/repopulate and the fallback in
    extract_layers_from_dde_fnn both keep working; the encoders are separate submodules.
    Supports apply_feature_transform (needed for the exact-periodicity Fourier embedding).
    rwf=True swaps every Linear for RWFLinear (same init function, factorized parameterization).
    """

    def __init__(self, layer_sizes, activation="tanh", rwf=False):
        super().__init__()
        self.rwf = rwf
        if len(layer_sizes) < 3:
            raise ValueError("ModifiedMLP needs at least [in, hidden, out]")
        widths = set(layer_sizes[1:-1])
        if len(widths) != 1:
            raise ValueError("ModifiedMLP requires equal hidden widths")
        in_dim, width, out_dim = layer_sizes[0], layer_sizes[1], layer_sizes[-1]
        self.activation = torch.tanh if activation == "tanh" else getattr(torch, activation)
        Lin = RWFLinear if rwf else torch.nn.Linear
        self.encoder_u = Lin(in_dim, width)
        self.encoder_v = Lin(in_dim, width)
        self.linears = torch.nn.ModuleList(
            [Lin(in_dim, width)]
            + [Lin(width, width) for _ in range(len(layer_sizes) - 3)]
            + [Lin(width, out_dim)]
        )
        if not rwf:
            for m in [self.encoder_u, self.encoder_v, *self.linears]:
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        U = self.activation(self.encoder_u(x))
        V = self.activation(self.encoder_v(x))
        H = self.activation(self.linears[0](x))
        for linear in self.linears[1:-1]:
            Z = self.activation(linear(H))
            H = (1 - Z) * U + Z * V
        out = self.linears[-1](H)
        if self._output_transform is not None:
            out = self._output_transform(inputs, out)
        return out

# method -> (loss_type, optimizer schedule). schedule = list of (opt_name, lr, frac)
# where frac is the fraction of --iterations spent in that phase. frozen has no schedule.
# Paper-backed "best" gradient pipeline for chaotic PINNs, applied to *every* gradient method
# so the loss/approach is the only variable. Both the causal paper (Wang, Sankaran & Perdikaris,
# "Respecting causality...", 2022, arXiv:2203.07404) and the "Expert's Guide to Training PINNs"
# (Wang et al. 2023, arXiv:2308.08468) use **Adam only -- NOT L-BFGS** for stiff/chaotic
# time-dependent PDEs (the Expert's Guide recommends "Adam exclusively"), with lr 1e-3 and an
# exponential (step) learning-rate decay of x0.9 every 2000 iterations. Causal loss uses a fixed,
# moderately large eps=1.0 (Expert's Guide default, so all temporal weights converge to 1; the
# causal paper's alternative is to anneal eps through [1e-2, 1e-1, 1, 10, 100]).
# Second pipeline: SOAP (Shampoo-preconditioned Adam), per "Gradient Alignment in Physics-
# informed Neural Networks: A Second-Order Optimization Perspective" (arXiv:2502.00604), the
# paper that actually benchmarks SOAP on Kuramoto-Sivashinsky AND Grey-Scott (30.6x lower error
# than Adam on Grey-Scott). Their recipe: lr warmup 0->1e-3 over 5000 steps then x0.9 decay,
# precondition_frequency=2 (not the SOAP default of 10), high momentum beta1=0.99. Their own
# baseline pipeline runs causal training by default, so the paper's "SOAP" result is really
# SOAP+causal -- that combination is what `soap_causal` below is meant to reproduce.
SOAP_P = ("soap", 1e-3, ("warmup_step", 5000, 2000, 0.9))

# Each schedule phase is (opt_name, lr, decay, frac_of_total_iterations).
ADAM_P = ("adam", 1e-3, ("step", 2000, 0.9))   # the shared best pipeline P*

METHOD_SPEC = {
    # -- recommended comparison: identical best pipeline P*, differing ONLY in the loss --
    "origin":         {"loss_type": "origin", "schedule": [(*ADAM_P, 1.0)]},
    "causal":         {"loss_type": "causal", "schedule": [(*ADAM_P, 1.0)]},
    # -- SOAP/second-order: the paper-tuned config above, on both losses --
    "soap":           {"loss_type": "origin", "schedule": [(*SOAP_P, 1.0)]},
    "soap_causal":    {"loss_type": "causal", "schedule": [(*SOAP_P, 1.0)]},
    # -- optimizer-variant methods kept for the "which optimizer" question; the causal-paper /
    #    Expert's-Guide literature says plain L-BFGS is worse on chaotic, so lbfgs_baseline
    #    mainly exists to confirm that (see also arXiv:2501.16371, which finds *self-scaled*
    #    quasi-Newton variants like SSBroyden far outperform plain L-BFGS on Kuramoto-Sivashinsky
    #    -- not implemented here, but worth knowing plain L-BFGS is not the strongest QN baseline) --
    "adam_baseline":  {"loss_type": "origin", "schedule": [(*ADAM_P, 1.0)]},   # alias of `origin`
    "lbfgs_baseline": {"loss_type": "origin", "schedule": [(*ADAM_P, 0.5), ("lbfgs", 1.0, None, 0.5)]},
    # -- best_practice: the FULL literature stack for chaotic PINNs, everything the papers'
    #    successful chaotic runs combine (not any single fix):
    #      * causal loss + eps annealing            (arXiv:2203.07404, wired like `causal`)
    #      * time-marching over 10 windows          (the causal paper's chaotic-KS setting;
    #                                                 forced by the method, no flag needed)
    #      * modified MLP architecture              (arXiv:2001.04536; used by both papers)
    #      * grad-norm loss balancing               (Expert's Guide arXiv:2308.08468: update
    #                                                 every 1000 steps, moving average 0.9,
    #                                                 weights start at 1)
    #      * exact-periodicity Fourier embedding    (already default for KS/GS)
    #      * Adam 1e-3 + x0.9/2000 step decay       (the shared pipeline P*)
    #    NOTE: the modified-MLP architecture has a different parameter space than the plain
    #    FNN, so best_practice cannot share the same-init/shared-landscape overlay with the
    #    other methods (shared_landscape.py skips mismatched architectures gracefully); it is
    #    compared on the solution-accuracy tier + its own landscape.
    "best_practice":  {"loss_type": "causal", "schedule": [(*ADAM_P, 1.0)],
                       "arch": "modified_mlp", "grad_norm_weights": True, "time_windows": 10},
    "frozen":         {"loss_type": "origin", "schedule": None},
}


# --------------------------------------------------------------------------- #
# Ablation sweep: which single best_practice ingredient (or pair, or triple) actually drives
# the improvement over `origin`? Full power set of the 4 binary ingredients:
#   C = causal loss + eps annealing        W = time-marching (10 windows)
#   A = modified-MLP Architecture          G = Grad-norm loss balancing
# `ablation_none` == `origin`, `ablation_C` == `causal`, `ablation_all` == `best_practice`
# (all three kept under both names on purpose: the plain names for readability elsewhere in
# this file/README, the systematic names for scripting a full sweep with `--ablation`).
# --------------------------------------------------------------------------- #
INGREDIENT_LETTERS = ["C", "W", "A", "G"]


def _ablation_spec(letters):
    spec = {"loss_type": "causal" if "C" in letters else "origin", "schedule": [(*ADAM_P, 1.0)]}
    if "A" in letters:
        spec["arch"] = "modified_mlp"
    if "G" in letters:
        spec["grad_norm_weights"] = True
    if "W" in letters:
        spec["time_windows"] = 10
    return spec


def _ablation_name(combo):
    if not combo:
        return "ablation_none"
    if len(combo) == len(INGREDIENT_LETTERS):
        return "ablation_all"
    return "ablation_" + "".join(combo)


ABLATION_METHODS = []
for _r in range(len(INGREDIENT_LETTERS) + 1):
    for _combo in itertools.combinations(INGREDIENT_LETTERS, _r):
        _name = _ablation_name(_combo)
        METHOD_SPEC[_name] = _ablation_spec(set(_combo))
        ABLATION_METHODS.append(_name)


# =========================================================================== #
# Model builder (closure, as PlotLossSurface / VisualizationModel expect)
# =========================================================================== #
def build_get_model(pde_name, hidden_layers, loss_type, causal_eps, num_causal_buckets,
                    fourier_modes=0, time_range=None, ic_func=None, arch="fnn", rwf=False):
    """fourier_modes > 0 applies the exact-periodicity Fourier embedding (see PDE_REGISTRY);
    time_range/ic_func restrict the PDE to one time-marching window with a handed-off IC;
    arch selects the network: "fnn" (DeepXDE FNN) or "modified_mlp" (Wang et al. 2021);
    rwf applies Random Weight Factorization to the modified MLP (Expert's Guide rec.)."""
    entry = PDE_REGISTRY[pde_name]

    def get_model():
        pde = entry["factory"](loss_type, causal_eps, num_causal_buckets,
                               time_range=time_range, ic_func=ic_func)
        sd = entry["spatial_dims"]
        in_dim = 2 * fourier_modes * sd + (pde.input_dim - sd) if fourier_modes > 0 else pde.input_dim
        layers = [in_dim] + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)) + [pde.output_dim]
        if arch == "modified_mlp":
            net = ModifiedMLP(layers, "tanh", rwf=rwf)
        elif arch == "fnn":
            net = dde.nn.FNN(layers, "tanh", "Glorot normal")
        else:
            raise ValueError(f"Unknown arch '{arch}'")
        if fourier_modes > 0:
            net.apply_feature_transform(make_periodic_transform(sd, entry["periodic"]["base_k"], fourier_modes))

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
def compile_optimizer(model, opt_name, lr, loss_weights, decay=None):
    """decay: a DeepXDE decay spec e.g. ("step", step_size, gamma) for StepLR, or None.
    Applied to Adam/SOAP (the paper recipe's x0.9-every-2000 schedule); L-BFGS ignores it."""
    if opt_name == "adam":
        model.compile(torch.optim.Adam(model.net.parameters(), lr=lr), decay=decay, loss_weights=loss_weights)
    elif opt_name == "lbfgs":
        opt = torch.optim.LBFGS(model.net.parameters(), lr=lr, line_search_fn="strong_wolfe", max_iter=10)
        model.compile(opt, loss_weights=loss_weights)
    elif opt_name == "soap":
        # precondition_frequency=2 and betas=(0.99, 0.999) per arXiv:2502.00604's ablation
        # (their default SOAP_options in deepxde/optimizers/config.py use frequency=10, beta1=0.95
        # -- untuned for PINNs); shampoo_beta left at its class default (not specified by the paper).
        set_SOAP_options(lr=lr, betas=(0.99, 0.999), precondition_frequency=2)
        model.compile("SOAP", decay=decay, loss_weights=loss_weights)
    else:
        raise ValueError(f"Unknown optimizer '{opt_name}'")


def seed_init_network(net, seed):
    """(Re)initialize an FNN's weights deterministically from `seed` ALONE.

    Guarantees the controlled-experiment property the comparison needs: every method at a given
    seed starts from the *same* weights (so the method is the only variable), while different
    seeds start from *different* weights (so repeats give a genuine spread). Reproduces DeepXDE's
    "Glorot normal" (Xavier-normal, gain=1) with zero biases, using a dedicated CPU Generator so
    the result depends only on the seed -- not on the device, the loss type, or how much RNG the
    rest of model construction happened to consume first.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.Linear):
                fan_out, fan_in = m.weight.shape[0], m.weight.shape[1]
                std = (2.0 / (fan_in + fan_out)) ** 0.5  # Xavier/Glorot normal, gain=1
                w = torch.randn(m.weight.shape, generator=g) * std
                m.weight.copy_(w.to(m.weight.device, m.weight.dtype))
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, RWFLinear):
                # deterministic RWF re-init from the same generator (factorized Xavier + s~N(1,0.1))
                m.reset_parameters(generator=g)


# =========================================================================== #
# Training (optimizer schedule x causal-eps annealing x time-marching)
# =========================================================================== #
# Causal-eps annealing schedule of Wang et al. 2022 (arXiv:2203.07404): train with a small eps
# first (gentle weighting while residuals are large), advance to the next eps once every causal
# weight has converged to ~1 (min_i w_i > delta, delta = 0.99 recommended). A fixed moderate eps
# (the Expert's-Guide simplification we used previously) under-weights late times for the whole
# run when residuals start large -- one of the reasons causal training underperformed.
CAUSAL_EPS_SCHEDULE = [1e-2, 1e-1, 1.0, 10.0, 100.0]


class CausalEpsAdvance(dde.callbacks.Callback):
    """Stop the current training phase early once min_i causal weight > delta (paper's rule)."""

    def __init__(self, delta=0.99, check_every=100):
        super().__init__()
        self.delta = delta
        self.check_every = check_every
        self.n = 0

    def on_epoch_end(self):
        self.n += 1
        if self.n % self.check_every:
            return
        w = getattr(self.model.data, "last_causal_min_weight", None)
        if w is not None and w > self.delta:
            print(f"[causal] min bucket weight {w:.4f} > {self.delta} -> advancing eps phase")
            self.model.stop_training = True


class GradNormLossWeights(dde.callbacks.Callback):
    """Gradient-norm loss balancing of the Expert's Guide (arXiv:2308.08468, Algorithm 1):
    every `update_every` steps, set lambda_hat_i = (sum_j ||grad L_j||) / ||grad L_i|| and move
    the weights by lambda_i <- alpha*lambda_i + (1-alpha)*lambda_hat_i (their defaults: update
    every 1000 iterations, alpha = 0.9, weights initialized to 1).

    `weights` must be the SAME torch tensor that was passed to model.compile(loss_weights=...):
    DeepXDE's compiled loss closure does `losses *= torch.as_tensor(loss_weights)`, and
    torch.as_tensor returns the tensor itself (no copy) when given a tensor of matching
    dtype/device, so in-place updates here take effect on the very next training step.
    """

    def __init__(self, weights, update_every=1000, alpha=0.9):
        super().__init__()
        self.weights = weights
        self.update_every = update_every
        self.alpha = alpha
        self.n = 0

    def on_epoch_end(self):
        self.n += 1
        if self.n % self.update_every:
            return
        m = self.model
        _, losses = m.outputs_losses_train(m.train_state.X_train, m.train_state.y_train)
        params = [p for p in m.net.parameters() if p.requires_grad]
        norms = []
        for i in range(len(losses)):
            grads = torch.autograd.grad(losses[i], params,
                                        retain_graph=(i < len(losses) - 1), allow_unused=True)
            sq = sum((g ** 2).sum() for g in grads if g is not None)
            # losses come back already weighted; ||grad(w_i L_i)|| = w_i ||grad L_i||
            norms.append(torch.sqrt(sq) / self.weights[i].clamp(min=1e-12))
        norms = torch.stack(norms).detach()
        lam_hat = norms.sum() / norms.clamp(min=1e-12)
        with torch.no_grad():
            self.weights.mul_(self.alpha).add_((1 - self.alpha) * lam_hat)
        print(f"[grad-norm] step {self.n}: loss weights -> "
              + np.array2string(self.weights.detach().cpu().numpy(), precision=2))


METRIC_COLUMNS = ["mse", "mae", "mxe", "l1re", "l2re", "crmse", "ic_mse", "bc_mse", "bc_rmse", "bc_l2re"]


class PerEpochMetricsCallback(dde.callbacks.Callback):
    """Records mse/mae/mxe/l1re/l2re/crmse/ic_mse/bc_mse/bc_rmse/bc_l2re every `log_every` epochs
    against a fixed reference sub-sample, WITHOUT ever clearing its own history.

    src.utils.callbacks.TesterCallback computes the same metrics but is designed to be reused
    across many short training chunks in the RL pipeline: its on_train_end wipes every
    accumulator list back to empty (and resets its epoch counter to 0) once it has read off the
    chunk-final .rmse/.brmse it needs -- confirmed by direct tracing (valid_epoch correctly
    reaches N during training, then is reset to 0 inside on_train_end). That makes it unusable
    for "train, then read the history back" the way this harness wants: attaching a fresh
    TesterCallback per optimizer phase and reading its lists after model.train() returns always
    yields empty lists, silently. This callback exists purely to avoid that trap: build ONE
    instance per run, pass the SAME instance into every model.train() call across every phase/
    sub-phase, and its own epoch counter -- independent of DeepXDE's train_state.step, which
    ModelSaverCallback resets to 0 after every phase -- gives one continuous, non-resetting
    step axis for free.
    """

    def __init__(self, pde, spatial_dims, log_every=100, n_ref=4000, seed=0):
        super().__init__()
        coords, values = _get_ref(pde)
        rng = np.random.default_rng(seed)
        sub = rng.choice(coords.shape[0], size=min(n_ref, coords.shape[0]), replace=False)
        self.coords, self.values = coords[sub], values[sub]
        self.ic_mask, self.bc_mask = _ic_bc_masks(pde, self.coords, spatial_dims)
        self.solution_l1 = float(np.abs(self.values).mean())
        self.solution_l2 = float(np.sqrt((self.values ** 2).mean()))
        self.log_every = max(1, log_every)
        self._since_last = 0
        self.epoch = 0
        self.rows = []

    def on_epoch_end(self):
        self.epoch += 1
        self._since_last += 1
        if self._since_last < self.log_every:
            return
        self._since_last = 0

        with torch.no_grad():
            y = self.model.predict(self.coords)
        err = y - self.values
        mse = float((err ** 2).mean())
        row = {
            "epoch": self.epoch, "mse": mse, "mae": float(np.abs(err).mean()),
            "mxe": float(np.abs(err).max()),
            "l1re": float(np.abs(err).mean() / self.solution_l1) if self.solution_l1 > 0 else float("nan"),
            "l2re": float(np.sqrt(mse) / self.solution_l2) if self.solution_l2 > 0 else float("nan"),
            "crmse": float(np.abs(err.mean())),
        }
        if self.ic_mask.any():
            row["ic_mse"] = float((err[self.ic_mask] ** 2).mean())
        else:
            row["ic_mse"] = float("nan")
        if self.bc_mask.any():
            bc_mse = float((err[self.bc_mask] ** 2).mean())
            row["bc_mse"] = bc_mse
            row["bc_rmse"] = float(np.sqrt(bc_mse))
            row["bc_l2re"] = row["bc_rmse"] / max(self.solution_l2, 1e-12)
        else:
            row["bc_mse"] = row["bc_rmse"] = row["bc_l2re"] = float("nan")
        self.rows.append(row)


def train_one_model(model, loss_weights, spec, iterations, n_saves, display_every, run_dir,
                    causal_eps_schedule=None, causal_delta=0.99, step_offset=0.0, metrics_cb=None,
                    extra_callbacks=None):
    """Run the method's optimizer schedule on `model` (already seed-initialized/warm-started).

    For causal-loss methods with an eps schedule, each optimizer phase is split into annealing
    sub-phases: model.data.causal_eps is raised through the schedule, advancing early when all
    causal weights exceed `causal_delta`.

    Every model.train() call resets DeepXDE's internal step counter to 0 (ModelSaverCallback's
    on_train_end does this), so a naive concatenation of model.losshistory across phases would
    have overlapping/repeating step numbers for any multi-phase method (lbfgs_baseline's Adam+
    L-BFGS tail, and every causal method's 5 eps sub-phases). `step_offset` fixes that: each
    phase's harvested loss-history rows get `+= step_offset` before `step_offset` is advanced by
    that phase's iteration count, giving one monotonically increasing step axis for the whole
    run (across windows too, when the caller threads step_offset through time-marching).

    metrics_cb: an already-constructed PerEpochMetricsCallback (or None to skip per-epoch
    validation metrics entirely -- used for time-marching, where the PDE's ref_data spans the
    FULL time domain regardless of the current window's bbox, so scoring a mid-window model
    against it would mix in massive extrapolation error from times it never trained on). The
    SAME instance must be passed to every phase in a window (it accumulates its own epoch
    counter across model.train() calls, unlike TesterCallback -- see its docstring for why a
    fresh TesterCallback per phase silently loses everything).

    Returns (solver_models, loss_rows, step_offset):
      loss_rows : list of 1D arrays [step, loss_0, loss_1, ...] (DeepXDE's raw component losses)
    """
    solver_models, loss_rows = [], []

    def run_phase(opt_name, lr, decay, phase_iters, n_save_models, causal_eps=None):
        nonlocal step_offset
        compile_optimizer(model, opt_name, lr, loss_weights, decay=decay)
        if causal_eps is not None:
            model.data.causal_eps = causal_eps
        prev_len = len(model.losshistory.steps)
        saver = ModelSaverCallback(total_iterations=phase_iters, n_save_models=n_save_models)
        callbacks = [saver]
        if causal_eps is not None:
            callbacks.append(CausalEpsAdvance(delta=causal_delta))
        if metrics_cb is not None:
            callbacks.append(metrics_cb)
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        tag = f"causal_eps={causal_eps} " if causal_eps is not None else ""
        print(f"--- phase: {opt_name} lr={lr} decay={decay} {tag}iters<={phase_iters} "
              f"(steps {step_offset:.0f}+) ---")
        model.train(iterations=phase_iters, display_every=display_every, callbacks=callbacks,
                    model_save_path=run_dir, save_model=False)

        steps = np.asarray(model.losshistory.steps[prev_len:], dtype=float)
        if len(steps):
            lt = np.asarray(model.losshistory.loss_train[prev_len:], dtype=float)
            loss_rows.append(np.hstack([(steps + step_offset)[:, None], lt]))
        step_offset += phase_iters
        solver_models.extend(saver.saved_models)

    n_phases = len(spec["schedule"])
    for (opt_name, lr, decay, frac) in spec["schedule"]:
        phase_iters = max(1, int(round(iterations * frac)))
        # A fresh compile happens per phase (and per time-marching window); if the SOAP lr
        # warmup would eat more than half the phase, shrink it proportionally so the phase
        # actually reaches the target lr.
        if decay is not None and decay[0] == "warmup_step" and decay[1] > phase_iters // 2:
            decay = ("warmup_step", max(1, phase_iters // 5), decay[2], decay[3])
        if spec["loss_type"] == "causal" and causal_eps_schedule:
            sub_iters = max(1, phase_iters // len(causal_eps_schedule))
            sub_saves = max(1, n_saves // (n_phases * len(causal_eps_schedule)))
            for eps in causal_eps_schedule:
                run_phase(opt_name, lr, decay, sub_iters, sub_saves, causal_eps=eps)
        else:
            run_phase(opt_name, lr, decay, phase_iters, max(1, n_saves // n_phases))
    return solver_models, loss_rows, step_offset


def make_ic_handoff(prev_model, output_dim):
    """IC function for time-marching window k>0: the previous window's prediction at the
    interface time (the IC collocation points already carry t = t_interface)."""
    def ic_scalar(x):
        return prev_model.predict(np.asarray(x, dtype=REAL)).astype(np.float64)

    def ic_multi(x, component):
        pred = prev_model.predict(np.asarray(x, dtype=REAL)).astype(np.float64)
        return pred[:, component]

    return ic_multi if output_dim > 1 else ic_scalar


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


def _ic_bc_masks(pde, coords, spatial_dims):
    """IC mask (t == t0) and spatial-boundary mask (bbox extremes, excluding the IC slice)."""
    has_time = pde.input_dim == spatial_dims + 1
    ic = np.isclose(coords[:, -1], coords[:, -1].min(), atol=1e-9) if has_time else np.zeros(len(coords), dtype=bool)
    bbox = np.asarray(pde.bbox, dtype=float)
    bmask = np.zeros(coords.shape[0], dtype=bool)
    for d in range(spatial_dims):
        bmask |= np.isclose(coords[:, d], bbox[2 * d], atol=1e-9) | np.isclose(coords[:, d], bbox[2 * d + 1], atol=1e-9)
    if has_time:
        bmask &= ~ic
    return ic, bmask


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

    ic, bmask = _ic_bc_masks(pde, coords, spatial_dims)
    if ic.any():
        metrics["ic_relative_l2"] = _relative_l2(pred[ic], values[ic])
    if bmask.any():
        metrics["boundary_relative_l2"] = _relative_l2(pred[bmask], values[bmask])

    bands = _spectral_band_error(coords, pred, values, spatial_dims)
    metrics["fourier_low"] = bands["band0"]
    metrics["fourier_mid"] = bands["band1"]
    metrics["fourier_high"] = bands["band2"]

    fields = {"coords": coords.astype(REAL),
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
    parser.add_argument("--causal-eps", type=float, default=None,
                        help="fixed causal eps; default None = the paper's annealing schedule "
                             f"{CAUSAL_EPS_SCHEDULE} advancing when min weight > --causal-delta")
    parser.add_argument("--causal-delta", type=float, default=0.99,
                        help="min-causal-weight threshold to advance the eps annealing phase")
    parser.add_argument("--num-causal-buckets", type=int, default=32)
    parser.add_argument("--fourier-modes", type=int, default=None,
                        help="modes for the exact-periodicity Fourier embedding (default: per-PDE; "
                             "KS 10, Gray-Scott 5, Burgers off). 0 disables.")
    parser.add_argument("--time-windows", type=int, default=1,
                        help="time-marching: split the time domain into this many sequentially "
                             "trained windows, IC handed off from the previous window (chaotic "
                             "PDEs only; the causal paper's setting for chaotic KS)")
    parser.add_argument("--ae-epochs", type=int, default=10000, help="autoencoder training epochs")
    parser.add_argument("--grid-xnum", type=int, default=25, help="landscape grid resolution")
    parser.add_argument("--no-landscape", action="store_true", help="skip the landscape tier (gradient methods)")
    parser.add_argument("--float64", action="store_true",
                        help="train in float64 (tests the precision-floor hypothesis H12: the "
                             "chaos horizon t* ~ ln(1/eps)/lambda extends only if the achievable "
                             "field error eps actually drops)")
    parser.add_argument("--rwf", action="store_true",
                        help="Random Weight Factorization on the modified MLP (Expert's Guide "
                             "arXiv:2308.08468: W = diag(exp(s))V, s~N(1,0.1); same init "
                             "function, better-conditioned parameterization). Only affects "
                             "arch=modified_mlp methods")
    parser.add_argument("--warmup", type=int, default=0,
                        help="linear lr warmup steps for Adam phases (Expert's Guide uses 5000 "
                             "for their large-scale configs; 0 = off, the benchmark default)")
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test settings")
    args = parser.parse_args()

    if args.quick:
        args.iterations = min(args.iterations, 200)
        args.n_save_models = min(args.n_save_models, 5)
        args.ae_epochs = min(args.ae_epochs, 400)
        args.grid_xnum = min(args.grid_xnum, 9)

    set_precision(args.float64)  # BEFORE any net/model construction

    entry = PDE_REGISTRY[args.pde]
    spec = METHOD_SPEC[args.method]
    hidden_layers = args.hidden_layers or entry["hidden_default"]
    spatial_dims = entry["spatial_dims"]
    output_names = entry["output_names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.warmup > 0:
        # Expert's-Guide-style linear lr warmup for the Adam phases (their large-scale configs
        # warm up over 5000 steps). Rewrites ("step", a, b) decay -> ("warmup_step", W, a, b).
        spec = dict(spec)
        spec["schedule"] = [
            (opt, lr, ("warmup_step", args.warmup, decay[1], decay[2]) if (opt == "adam" and decay and decay[0] == "step") else decay, frac)
            for (opt, lr, decay, frac) in spec["schedule"]] if spec["schedule"] else None

    # periodic embedding: per-PDE default unless overridden; 0 disables
    if args.fourier_modes is None:
        fourier_modes = entry["periodic"]["modes_default"] if entry.get("periodic") else 0
    else:
        fourier_modes = args.fourier_modes if entry.get("periodic") else 0

    # causal-eps: fixed value if given, else the paper's annealing schedule
    causal_eps_schedule = [args.causal_eps] if args.causal_eps is not None else list(CAUSAL_EPS_SCHEDULE)
    causal_eps0 = causal_eps_schedule[0]

    # method-level architecture / loss weighting (best_practice) -- see METHOD_SPEC
    arch = spec.get("arch", "fnn")
    grad_norm_weights = bool(spec.get("grad_norm_weights", False))

    # time-marching windows: explicit CLI flag wins; otherwise the method's own default
    # (best_practice forces 10 windows, everything else defaults to 1)
    time_windows = args.time_windows if args.time_windows > 1 else int(spec.get("time_windows", 1))
    if time_windows > 1 and (args.pde == "burgers1d" or args.method == "frozen"):
        # marching applies only to gradient methods on the chaotic PDEs; downgrade quietly so
        # run_all.py can pass --time-windows to the whole matrix
        print(f"[note] time-windows ignored for {args.pde}/{args.method}")
        time_windows = 1
    if args.quick and time_windows > 2:
        time_windows = 2  # keep smoke tests fast; real runs use the full window count

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
        "causal_eps_schedule": causal_eps_schedule, "causal_delta": args.causal_delta,
        "num_causal_buckets": args.num_causal_buckets,
        "fourier_modes": fourier_modes, "time_windows": time_windows,
        "arch": arch, "grad_norm_weights": grad_norm_weights,
        "float64": args.float64, "rwf": args.rwf, "warmup": args.warmup,
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
        pde_eval = entry["factory"](spec["loss_type"], causal_eps0, args.num_causal_buckets)
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
        # ---- gradient path (optionally time-marched over sequential windows) ----
        W = max(1, time_windows)
        t_lo, t_hi = entry["time_bbox"]
        edges = np.linspace(t_lo, t_hi, W + 1)
        iters_per_window = max(1, args.iterations // W)

        # landscape/reconstruction closure: FULL time domain, analytic IC (grid losses are then
        # "how does the full-domain loss see each visited point", comparable across methods)
        get_model_rec = build_get_model(args.pde, hidden_layers, spec["loss_type"],
                                        causal_eps0, args.num_causal_buckets,
                                        fourier_modes=fourier_modes, arch=arch, rwf=args.rwf)

        # A validation metric evaluated against the PDE's ref_data only means what it looks like
        # when the model being scored was trained over ref_data's whole time range: with time-
        # marching (W>1), each window's model only "knows" its own sub-interval, so scoring it
        # against the full-domain reference every log_every steps would silently mix in massive
        # extrapolation error from times it never trained on. Keep the standard per-epoch metric
        # curve (mse/l2re/etc.) for the common W==1 case; for W>1 rely on the (already-correct,
        # because predict_fn stitches windows) final metrics.json + trajectory_error.csv instead.
        attach_tester = (W == 1)

        solver_models = []
        window_models = []
        all_loss_rows = []
        metrics_cb = None
        step_offset = 0.0
        prev_model = None
        for w in range(W):
            time_range = (float(edges[w]), float(edges[w + 1])) if W > 1 else None
            ic_func = make_ic_handoff(prev_model, len(output_names)) if w > 0 else None
            gm = build_get_model(args.pde, hidden_layers, spec["loss_type"],
                                 causal_eps0, args.num_causal_buckets,
                                 fourier_modes=fourier_modes, time_range=time_range, ic_func=ic_func,
                                 arch=arch, rwf=args.rwf)
            model, loss_weights = gm()
            extra_cbs = None
            if grad_norm_weights:
                # Expert's Guide grad-norm balancing: weights start at 1 and adapt; must be a
                # torch tensor so the in-place callback updates reach the compiled loss closure
                # (see GradNormLossWeights). Fresh per window -- the balance re-adapts to each
                # window's problem, and a window is its own training run under time-marching.
                loss_weights = torch.ones(len(loss_weights))
                extra_cbs = [GradNormLossWeights(loss_weights, update_every=1000, alpha=0.9)]
            if w == 0:
                # Shared, method-independent starting point: identical weights for every method
                # at this seed, different weights across seeds (see seed_init_network).
                seed_init_network(model.net, args.seed)
                # Save the shared init as checkpoint 0 so every method's trajectory starts at
                # the SAME visible point in the (shared) landscape and per-checkpoint error
                # curves begin at the true common origin.
                solver_models.append(copy.deepcopy(model.net))
                if attach_tester:
                    # ONE instance, reused across every phase/sub-phase of this (single) window --
                    # see PerEpochMetricsCallback's docstring for why a fresh one per phase silently
                    # loses everything.
                    metrics_cb = PerEpochMetricsCallback(model.pde, spatial_dims,
                                                         log_every=args.display_every, seed=args.seed)
            else:
                # warm start from the previous window's solution (standard time-marching)
                model.net.load_state_dict(prev_model.net.state_dict())
            if W > 1:
                print(f"\n==== time window {w + 1}/{W}: t in [{edges[w]:.4g}, {edges[w + 1]:.4g}] ====")
            new_models, loss_rows, step_offset = train_one_model(
                model, loss_weights, spec, iters_per_window, max(1, args.n_save_models // W),
                args.display_every, run_dir, causal_eps_schedule=causal_eps_schedule,
                causal_delta=args.causal_delta, step_offset=step_offset, metrics_cb=metrics_cb,
                extra_callbacks=extra_cbs)
            solver_models += new_models
            all_loss_rows += loss_rows
            window_models.append(model)
            prev_model = model

        # final-model solution metrics (stitched across windows when time-marching)
        if W == 1:
            def predict_fn(coords):
                return model.predict(coords.astype(REAL))
        else:
            def predict_fn(coords):
                coords = np.asarray(coords)
                idx = np.clip(np.searchsorted(edges[1:-1], coords[:, -1], side="right"), 0, W - 1)
                out = np.empty((coords.shape[0], len(output_names)), dtype=np.float64)
                for k in range(W):
                    m = idx == k
                    if m.any():
                        out[m] = window_models[k].predict(coords[m].astype(REAL)).reshape(-1, len(output_names))
                return out
        pde_eval = model.pde
        sol_metrics, fields = evaluate_solution(pde_eval, predict_fn, spatial_dims, output_names)
        metrics.update(sol_metrics)

        # loss history (per display step, per component): step_offset already makes this
        # monotonic across every phase/sub-phase/window (see train_one_model's docstring).
        try:
            loss_hist = np.vstack(all_loss_rows)
            np.savetxt(os.path.join(run_dir, "loss_history.csv"),
                       loss_hist, delimiter=",",
                       header="step," + ",".join(f"loss_{i}" for i in range(loss_hist.shape[1] - 1)), comments="")
        except Exception as e:
            print(f"[warn] could not save loss history: {e}")

        # standard per-epoch validation metrics (mse/mae/mxe/l1re/l2re/crmse/ic_mse/bc_mse/...) --
        # previously not saved at all (only the single final-checkpoint snapshot in metrics.json
        # existed). Empty/absent for time-marching runs (see the attach_tester note above).
        if metrics_cb is not None and metrics_cb.rows:
            try:
                cols = ["epoch"] + METRIC_COLUMNS
                arr = np.array([[r[c] for c in cols] for r in metrics_cb.rows])
                np.savetxt(os.path.join(run_dir, "metrics_history.csv"), arr, delimiter=",",
                           header=",".join(cols), comments="")
            except Exception as e:
                print(f"[warn] could not save metrics history: {e}")
        elif not attach_tester:
            print("[note] metrics_history.csv skipped: time-marching (--time-windows > 1) makes "
                  "a per-epoch validation metric against the full-domain reference misleading "
                  "for a single window's model; see metrics.json for the correct stitched result.")

        # per-checkpoint trajectory error (rel-L2 at each landscape trajectory point)
        try:
            coords, values = _get_ref(pde_eval)
            sub = np.random.choice(coords.shape[0], size=min(4000, coords.shape[0]), replace=False)
            rows = []
            base_net_state = {k: v.clone() for k, v in model.net.state_dict().items()}
            for i, net in enumerate(solver_models):
                model.net.load_state_dict(net.state_dict())
                p = model.predict(coords[sub].astype(REAL)).reshape(values[sub].shape)
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
