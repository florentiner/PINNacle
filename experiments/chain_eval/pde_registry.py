"""
Registry of the 22 PINNacle PDEs for chain evaluation.

Every entry builds (model, loss_weights) the same way the optuna scripts in
experiments/optuna_multi_pde/ did: default PDE constructor, FNN "100*5" tanh
Glorot-normal net (recommend_net for the inverse problems), loss weight 100 on
boundary/initial losses and 1 on PDE losses.
"""
from __future__ import annotations

import argparse

import numpy as np
import deepxde as dde

from src.pde.burgers import Burgers1D, Burgers2D
from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.pde.heat import (
    Heat2D_VaryingCoef,
    Heat2D_Multiscale,
    Heat2D_ComplexGeometry,
    Heat2D_LongTime,
    HeatND,
)
from src.pde.inverse import PoissonInv, HeatInv
from src.pde.ns import NS2D_Classic, NS2D_BackStep, NS2D_LongTime
from src.pde.poisson import (
    Poisson2D_Classic,
    PoissonBoltzmann2D,
    Poisson3D_ComplexGeometry,
    Poisson2D_ManyArea,
    PoissonND,
)
from src.pde.wave import Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime
from src.utils.args import parse_hidden_layers

# name -> (class, constructor kwargs). Names match the optuna experiment names
# (and therefore the historical CSV names in the HF dataset).
FORWARD_PDES = {
    # Burgers 1d-C / 2d-C
    "burgers_1d": (Burgers1D, {}),
    "burgers_2d": (Burgers2D, {}),
    # Poisson 2d-C / 2d-CG / 3d-CG / 2d-MS
    "poisson2d_classic": (Poisson2D_Classic, {}),
    "poissonboltzmann2d": (PoissonBoltzmann2D, {}),
    "poisson3d_complexgeometry": (Poisson3D_ComplexGeometry, {}),
    "poisson2d_manyarea": (Poisson2D_ManyArea, {}),
    # Heat 2d-VC / 2d-MS / 2d-CG / 2d-LT
    "heat2d_varyingcoef": (Heat2D_VaryingCoef, {}),
    "heat2d_multiscale": (Heat2D_Multiscale, {}),
    "heat2d_complexgeometry": (Heat2D_ComplexGeometry, {}),
    "heat2d_longtime": (Heat2D_LongTime, {}),
    # NS 2d-C / 2d-CG / 2d-LT
    "ns2d_classic": (NS2D_Classic, {}),
    "ns2d_backstep": (NS2D_BackStep, {}),
    "ns2d_longtime": (NS2D_LongTime, {}),
    # Wave 1d-C / 2d-CG / 2d-MS
    "wave1d": (Wave1D, {}),
    "wave2d_heterogeneous": (Wave2D_Heterogeneous, {}),
    "wave2d_longtime": (Wave2D_LongTime, {}),
    # Chaotic GS / KS
    "grayscott": (GrayScottEquation, {}),
    "kuramoto_sivashinsky": (KuramotoSivashinskyEquation, {}),
    # High-dim PNd / HNd
    "poissonnd": (PoissonND, {"dim": 5}),
    "heatnd": (HeatND, {"dim": 5}),
}

INVERSE_PDES = {
    "poissoninv": (PoissonInv, {}),
    "heatinv": (HeatInv, {}),
}

ALL_PDE_NAMES = list(FORWARD_PDES) + list(INVERSE_PDES)


def _loss_weights(pde):
    weights = np.ones(pde.num_loss, dtype=np.float32)
    for i, c in enumerate(pde.loss_config):
        if c.get("type", "") in ("boundary", "initial", "ic"):
            weights[i] = 100.0
    return weights


def build_get_model(pde_name: str, hidden_layers: str = "100*5"):
    """Return a get_model() -> (model, loss_weights) callable for the PDE."""
    if pde_name in FORWARD_PDES:
        cls, kwargs = FORWARD_PDES[pde_name]
        inverse = False
    elif pde_name in INVERSE_PDES:
        cls, kwargs = INVERSE_PDES[pde_name]
        inverse = True
    else:
        raise KeyError(
            f"Unknown PDE '{pde_name}'. Available: {', '.join(ALL_PDE_NAMES)}"
        )

    def get_model():
        pde = cls(**kwargs)
        if inverse:
            net = pde.recommend_net
        else:
            layers = (
                [pde.input_dim]
                + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers))
                + [pde.output_dim]
            )
            net = dde.nn.FNN(layers, "tanh", "Glorot normal")
        net = net.float()
        model = pde.create_model(net)
        return model, _loss_weights(pde)

    return get_model
