"""
Model builders for the 22 PINNacle PDEs (see pde_names.py for the list).

Every entry builds (model, loss_weights) the same way the optuna scripts in
experiments/optuna_multi_pde/ did: default PDE constructor, FNN "100*5" tanh
Glorot-normal net (recommend_net for the inverse problems), loss weight 100 on
boundary/initial losses and 1 on PDE losses.

Importing this module requires torch/deepxde (DDEBACKEND=pytorch); use
pde_names.py when only the name list is needed.
"""
from __future__ import annotations

import argparse
import importlib

import numpy as np
import deepxde as dde

from experiments.chain_eval.pde_names import (
    ALL_PDE_NAMES,
    INVERSE_PDE_NAMES,
    PDE_SPECS,
)
from src.utils.args import parse_hidden_layers

__all__ = ["ALL_PDE_NAMES", "build_get_model"]


def _loss_weights(pde):
    weights = np.ones(pde.num_loss, dtype=np.float32)
    for i, c in enumerate(pde.loss_config):
        if c.get("type", "") in ("boundary", "initial", "ic"):
            weights[i] = 100.0
    return weights


def build_get_model(pde_name: str, hidden_layers: str = "100*5"):
    """Return a get_model() -> (model, loss_weights) callable for the PDE."""
    if pde_name not in PDE_SPECS:
        raise KeyError(
            f"Unknown PDE '{pde_name}'. Available: {', '.join(ALL_PDE_NAMES)}"
        )
    module_name, class_name, kwargs = PDE_SPECS[pde_name]
    cls = getattr(importlib.import_module(module_name), class_name)
    inverse = pde_name in INVERSE_PDE_NAMES

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
