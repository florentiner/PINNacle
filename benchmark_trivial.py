import argparse
import time
import os
from trainer import Trainer

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde
from src.pde.heat import Heat2D_LongTime
from src.pde.wave import Wave1D
from src.utils.args import parse_hidden_layers, parse_loss_weight
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback

# Trivial-attractor study: vanilla baselines on PDEs with exact trivial solutions
# (see analysis/TRIVIAL_HYPOTHESIS.md). Same harness pattern as benchmark_chaotic.py.
pde_list = [Heat2D_LongTime]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINNBench trainer (trivial-attractor cases)')
    parser.add_argument('--name', type=str, default="benchmark-trivial")
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--hidden-layers', type=str, default="100*5")
    parser.add_argument('--loss-weight', type=str, default="")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--iter', type=int, default=20000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--plot-every', type=int, default=2000)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--method', type=str, default="adam")
    parser.add_argument('--case', type=str, default="heatlt", choices=["heatlt", "wave1d", "all"])
    parser.add_argument('--hyp-data', action='store_true', help="save forensic data (checkpoints, fields, metrics.csv)")

    command_args = parser.parse_args()

    seed = command_args.seed
    if seed is not None:
        dde.config.set_random_seed(seed)
    date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())
    trainer = Trainer(f"{date_str}-{command_args.name}", command_args.device)

    if command_args.case == "heatlt":
        pde_list = [Heat2D_LongTime]
    elif command_args.case == "wave1d":
        pde_list = [Wave1D]
    else:
        pde_list = [Heat2D_LongTime, Wave1D]

    for pde_config in pde_list:

        def get_model_dde():
            if isinstance(pde_config, tuple):
                pde = pde_config[0](**pde_config[1])
            else:
                pde = pde_config()

            net = dde.nn.FNN([pde.input_dim] + parse_hidden_layers(command_args) + [pde.output_dim], "tanh", "Glorot normal")
            net = net.float()

            loss_weights = parse_loss_weight(command_args)
            if loss_weights is None:
                loss_weights = np.ones(pde.num_loss)
            else:
                loss_weights = np.array(loss_weights)

            opt = torch.optim.Adam(net.parameters(), command_args.lr)

            model = pde.create_model(net)
            model.compile(opt, loss_weights=loss_weights)
            return model

        callbacks = [
            TesterCallback(log_every=command_args.log_every),
            PlotCallback(log_every=command_args.plot_every, fast=True),
            LossCallback(verbose=True),
        ]
        if command_args.hyp_data:
            from src.utils.hypothesis_callback import HypothesisDataCallback
            callbacks.append(HypothesisDataCallback(log_every=command_args.log_every, ckpt_every=command_args.plot_every))

        trainer.add_task(get_model_dde, {
            "iterations": command_args.iter,
            "display_every": command_args.log_every,
            "callbacks": callbacks,
        })

    trainer.setup(__file__, seed)
    trainer.set_repeat(command_args.repeat)
    trainer.train_all()
    trainer.summary()
