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


class FourierFNN(dde.nn.pytorch.NN):
    """FNN on fixed random Fourier features x -> [sin(2*pi*xB), cos(2*pi*xB)].
    Base-architecture variant of the boost-net spectral fix: counters spectral
    bias from step 0 instead of mid-training. B is drawn under the run's seed
    (set_random_seed happens before model build), half the features per sigma."""

    def __init__(self, in_dim, out_dim, hidden_sizes, sigmas=(1, 10), n_feats=128):
        import torch
        super().__init__()
        per = max(1, n_feats // len(sigmas))
        cols = [torch.randn(in_dim, per) * s for s in sigmas]
        self.register_buffer("B", torch.cat(cols, dim=1))
        self.fnn = dde.nn.FNN([2 * self.B.shape[1]] + hidden_sizes + [out_dim],
                              "tanh", "Glorot normal")

    def forward(self, x):
        import torch
        if self._input_transform is not None:
            x = self._input_transform(x)
        z = 2 * np.pi * (x @ self.B)
        y = self.fnn(torch.cat([torch.sin(z), torch.cos(z)], dim=1))
        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y


def _loss_weights(pde):
    weights = np.ones(pde.num_loss, dtype=np.float32)
    for i, c in enumerate(pde.loss_config):
        if c.get("type", "") in ("boundary", "initial", "ic"):
            weights[i] = 100.0
    return weights


def build_get_model(pde_name: str, hidden_layers: str = "100*5", net_type: str = "fnn",
                    inverse_plain_fnn: bool = False):
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
        if inverse and not inverse_plain_fnn:
            net = pde.recommend_net
        else:
            layers = (
                [pde.input_dim]
                + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers))
                + [pde.output_dim]
            )
            if net_type == "fourier":
                net = FourierFNN(pde.input_dim, pde.output_dim, layers[1:-1])
            else:
                net = dde.nn.FNN(layers, "tanh", "Glorot normal")
        net = net.float()
        model = pde.create_model(net)
        return model, _loss_weights(pde)

    return get_model
