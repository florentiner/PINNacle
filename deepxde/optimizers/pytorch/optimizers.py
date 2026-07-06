__all__ = ["get", "is_external_optimizer"]

import torch

# from .nncg import NNCG
from .pso import PSO
from .soap import SOAP
from ..config import LBFGS_options, NNCG_options, PSO_options, SOAP_options


# NOTE: edited
def is_external_optimizer(optimizer):
    return optimizer in ["L-BFGS", "L-BFGS-B", "NNCG", "PSO"]


def get(params, optimizer, learning_rate=None, decay=None, weight_decay=0):
    """Retrieves an Optimizer instance."""
    # Custom Optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optim = optimizer
    elif optimizer in ["L-BFGS", "L-BFGS-B"]:
        if weight_decay > 0:
            raise ValueError("L-BFGS optimizer doesn't support weight_decay > 0")
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        optim = torch.optim.LBFGS(
            params,
            lr= LBFGS_options["lr"] if LBFGS_options["lr"]  is not None else 1,
            max_iter=LBFGS_options["iter_per_step"],
            max_eval=LBFGS_options["fun_per_step"],
            tolerance_grad=LBFGS_options["gtol"],
            tolerance_change=LBFGS_options["ftol"],
            history_size=LBFGS_options["maxcor"],
            line_search_fn=("strong_wolfe" if LBFGS_options["maxls"] > 0 else None),
        )
    elif optimizer == "NNCG":
        if weight_decay > 0:
            raise ValueError("NNCG optimizer doesn't support weight_decay > 0")
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        optim = NNCG(
            params,
            lr=NNCG_options["lr"],
            rank=NNCG_options["rank"],
            mu=NNCG_options["mu"],
            update_freq=NNCG_options["updatefreq"],
            chunk_size=NNCG_options["chunksz"],
            cg_tol=NNCG_options["cgtol"],
            cg_max_iters=NNCG_options["cgmaxiter"],
            line_search_fn=NNCG_options["lsfun"],
            verbose=NNCG_options["verbose"],
        )
    elif optimizer == "SOAP":
        if weight_decay > 0:
            raise ValueError(
                "weight_decay is ignored for SOAP; set it via dde.optimizers.set_SOAP_options(weight_decay=...)"
            )
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}; set it via dde.optimizers.set_SOAP_options(lr=...)".format(optimizer))
        optim = SOAP(
            params,
            lr=SOAP_options["lr"],
            betas=SOAP_options["betas"],
            shampoo_beta=SOAP_options["shampoo_beta"],
            eps=SOAP_options["eps"],
            weight_decay=SOAP_options["weight_decay"],
            precondition_frequency=SOAP_options["precondition_frequency"],
            max_precond_dim=SOAP_options["max_precond_dim"],
            precondition_1d=SOAP_options["precondition_1d"],
            correct_bias=SOAP_options["correct_bias"],
        )
    elif optimizer == "PSO":
        if weight_decay > 0:
            raise ValueError("PSO optimizer doesn't support weight_decay > 0")
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        optim = PSO(
            params,
            pop_size=PSO_options["pop_size"],
            b=PSO_options["b"],
            c1=PSO_options["c1"],
            c2=PSO_options["c2"],
            lr=PSO_options["lr"],
            betas=PSO_options["betas"],
            c_decrease=PSO_options["c_decrease"],
            variance=PSO_options["variance"],
            epsilon=PSO_options["epsilon"],
            n_iter=PSO_options["n_iter"],
        )
    else:
        if learning_rate is None:
            raise ValueError("No learning rate for {}.".format(optimizer))
        if optimizer == "sgd":
            optim = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "rmsprop":
            optim = torch.optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adam":
            optim = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adamw":
            if weight_decay == 0:
                raise ValueError("AdamW optimizer requires non-zero weight decay")
            optim = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"{optimizer} to be implemented for backend pytorch.")
    lr_scheduler = _get_learningrate_scheduler(optim, decay)
    return optim, lr_scheduler


def _get_learningrate_scheduler(optim, decay):
    if decay is None:
        return None

    # NOTE: edited
    if isinstance(decay, torch.optim.lr_scheduler._LRScheduler) or decay.__class__.__name__ == "ReduceLROnPlateau":
        return decay

    if decay[0] == "step":
        return torch.optim.lr_scheduler.StepLR(optim, step_size=decay[1], gamma=decay[2])

    if decay[0] == "warmup_step":
        # Linear warmup from ~0 to the base lr over `warmup_steps`, then StepLR(step_size, gamma)
        # decay thereafter. Matches the SOAP-for-PINNs recipe in "Gradient Alignment in PINNs: A
        # Second-Order Optimization Perspective" (arXiv:2502.00604): warmup 0->1e-3 over 5000
        # steps, then exponential decay.
        _, warmup_steps, step_size, gamma = decay
        warmup = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
        )
        after_warmup = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
        return torch.optim.lr_scheduler.SequentialLR(
            optim, schedulers=[warmup, after_warmup], milestones=[warmup_steps]
        )

    # TODO: More learning rate scheduler
    raise NotImplementedError(f"{decay[0]} learning rate scheduler to be implemented for backend pytorch.")
