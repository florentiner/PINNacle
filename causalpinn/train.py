"""Causal PINN training loop: tol-annealed causal weighting + time marching.

Faithful port of CausalPINNs/KS/chaotic_KS.py structure:
  per window: fresh net (identical seed each window, as in the reference),
  fresh Adam + exp-decay lr; anneal tol through tol_list, each stage capped at
  iter_cap iterations with early stop when min(W) > wmin_threshold;
  causal weights W_i = stopgrad(exp(-tol * (sum_{j<i} L_j + w_ic * L_ic))).
Ablation (--no-causal): W = 1 identically, same everything else.
"""
import os
import sys
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from causalpinn import checkpoint as ckpt_mod
from causalpinn.hypothesis_log import RunLogger


@dataclass
class CausalConfig:
    case: str = "ks"
    device: str = "cpu"
    seed: int = 1234
    encoding: str = "fourier"          # fourier | plain
    causal: bool = True                # False => ablation run
    windows: int = 10
    n_t: int = 32
    n_s: int = 256                     # spatial points per batch
    tol_list: tuple = (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2)
    iter_cap: int = 200000             # per tol stage
    check_every: int = 1000
    log_every: int = 1000
    ckpt_every: int = 10000
    snapshot_every: int = 25000        # forensic field snapshots (per window)
    wmin_threshold: float = 0.99
    w_ic: float = 1e4
    width: int = 128
    depth: int = 8
    M_t_ks: int = 6
    M_x_ks: int = 5
    M_t_gs: int = 2
    M_x_gs: int = 5
    lr: float = 1e-3
    decay_rate: float = 0.9
    decay_steps: int = 5000
    max_hours: float = 1e9
    outdir: str = "runs/causal-dev/0-0"
    resume_dir: str = ""
    compile: bool = False              # torch.compile the net forward (GPU: big win)


class TimeGuard:
    def __init__(self, max_hours, already_used=0.0, margin_s=180.0):
        self.t0 = time.time()
        self.budget = max_hours * 3600.0 - margin_s
        self.already = already_used

    def elapsed_total(self):
        return self.already + (time.time() - self.t0)

    def over(self):
        return (time.time() - self.t0) > max(60.0, self.budget - self.already)


def unwrap(net):
    """Underlying module of a torch.compile wrapper (state_dict portability)."""
    return getattr(net, "_orig_mod", net)


def _torch_pts(a, device, grad=False):
    t = torch.tensor(np.asarray(a), dtype=torch.float32, device=device)
    if grad:
        t.requires_grad_(True)
    return t


def make_batch(case, rng, device):
    """Sorted local times x spatial points, expanded to the full tensor product."""
    cfg = case.cfg
    t1_local = case.T_w * case.t_scale
    t_r = np.sort(rng.uniform(0.0, 1.01 * t1_local, size=cfg.n_t))
    s_r = case.sample_spatial(rng, cfg.n_s)                       # (n_s, d)
    t_full = np.repeat(t_r, cfg.n_s)[:, None]                     # (n_t*n_s, 1)
    s_full = np.tile(s_r, (cfg.n_t, 1))                           # (n_t*n_s, d)
    # jvp-based residuals: inputs must NOT require grad (see cases.py)
    return (_torch_pts(t_full, device),
            _torch_pts(s_full, device), t_r)


@torch.no_grad()
def predict_grid(net, t_loc, coords, device, chunk=16384, n_comp=1):
    """Prediction on (len(t_loc) x len(coords)) grid -> (n_t, n_pts, n_comp)."""
    outs = []
    coords = np.asarray(coords)
    for tv in np.asarray(t_loc):
        t_full = np.full((len(coords), 1), tv, dtype=np.float32)
        row = []
        for i in range(0, len(coords), chunk):
            tt = _torch_pts(t_full[i:i + chunk], device)
            cc = _torch_pts(coords[i:i + chunk], device)
            row.append(net(tt, *[cc[:, j:j + 1] for j in range(cc.shape[1])]).cpu().numpy())
        outs.append(np.concatenate(row, axis=0))
    return np.stack(outs, axis=0)


def residual_grid(case, net, t_loc, coords, device, chunk=4096):
    """PDE residual field on a (t x pts) grid via autograd -> (n_t, n_pts, n_comp)."""
    outs = []
    coords = np.asarray(coords)
    for tv in np.asarray(t_loc):
        row = []
        for i in range(0, len(coords), chunk):
            n = len(coords[i:i + chunk])
            tt = _torch_pts(np.full((n, 1), tv, dtype=np.float32), device)
            cc = _torch_pts(coords[i:i + chunk], device)
            r = case.residual(net, tt, cc, case.T_w)
            row.append(r.detach().cpu().numpy())
        outs.append(np.concatenate(row, axis=0))
    return np.stack(outs, axis=0)


def run(case, cfg: CausalConfig):
    device = torch.device(cfg.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    state = ckpt_mod.try_resume(cfg)
    logger = RunLogger(case, cfg, resume=state is not None)

    if state is None:
        rng = np.random.default_rng(cfg.seed)
        stitched = np.full_like(case.ref, np.nan, dtype=np.float32)
        ic_coords, ic_vals = case.ic_arrays()
        state = dict(window=0, stage=0, it=0, win_step=0,
                     net_sd=None, opt_sd=None,
                     rng_state=rng.bit_generator.state,
                     stitched=stitched, ic_vals=ic_vals.astype(np.float32),
                     walltime_used=0.0)
        # column 0 (t=0) of the stitched field = exact IC handled by window 0 preds
    rng = np.random.default_rng(cfg.seed)
    rng.bit_generator.state = state["rng_state"]
    guard = TimeGuard(cfg.max_hours, already_used=state["walltime_used"])

    ic_coords, _ = case.ic_arrays()
    n_ic = len(ic_coords)
    # strictly-lower-triangular causal matrix (reference: triu(ones,k=1).T)
    M = torch.tril(torch.ones(cfg.n_t, cfg.n_t, device=device), diagonal=-1)

    for k in range(state["window"], case.n_windows):
        # --- window init (reference quirk: identical init every window) ---
        net = case.build_net(cfg.encoding, cfg.seed, device)
        if cfg.compile:
            try:
                net = torch.compile(net)
            except Exception as e:  # pragma: no cover
                print(f"[WARN] torch.compile failed ({e}); continuing eager")
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: cfg.decay_rate ** (s / cfg.decay_steps))
        if k == state["window"] and state["net_sd"] is not None:
            unwrap(net).load_state_dict(state["net_sd"])
            opt.load_state_dict(state["opt_sd"])   # restores decayed lr exactly
            sched.last_epoch = state["win_step"]   # future steps continue the decay
        ic_c = _torch_pts(ic_coords, device)
        ic_v = _torch_pts(state["ic_vals"], device)
        ic_args = [ic_c[:, j:j + 1] for j in range(ic_c.shape[1])]
        t_ic = torch.zeros(n_ic, 1, device=device)

        stage0 = state["stage"] if k == state["window"] else 0
        for stage in range(stage0, len(cfg.tol_list)):
            tol = cfg.tol_list[stage]
            it0 = state["it"] if (k == state["window"] and stage == state["stage"]) else 0
            for it in range(it0, cfg.iter_cap):
                t_b, s_b, t_r = make_batch(case, rng, device)
                r = case.residual(net, t_b, s_b, case.T_w)          # (n_t*n_s, n_comp)
                L_t = (r ** 2).reshape(cfg.n_t, cfg.n_s, case.n_comp).mean(dim=(1, 2))
                L_ic = ((net(t_ic, *ic_args) - ic_v) ** 2).mean()
                L0 = cfg.w_ic * L_ic
                if cfg.causal:
                    W = torch.exp(-tol * (M @ L_t + L0)).detach()
                else:
                    W = torch.ones_like(L_t)
                loss = (W * L_t + L0).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                state["win_step"] += 1
                w_min = float(W.min())

                if (it + 1) % cfg.log_every == 0 or it == 0:
                    logger.log_step(k, stage, tol, it + 1, state["win_step"],
                                    loss.item(), L_ic.item(), L_t.mean().item(),
                                    w_min, W.detach().cpu().numpy(),
                                    L_t.detach().cpu().numpy(), t_r,
                                    sched.get_last_lr()[0], guard.elapsed_total(),
                                    net, predict_grid)
                if (it + 1) % cfg.snapshot_every == 0:
                    logger.field_snapshot(k, state["win_step"], net,
                                          predict_grid, residual_grid)
                need_ckpt = (it + 1) % cfg.ckpt_every == 0
                if need_ckpt or guard.over():
                    state.update(stage=stage, it=it + 1, window=k,
                                 net_sd=unwrap(net).state_dict(), opt_sd=opt.state_dict(),
                                 rng_state=rng.bit_generator.state,
                                 walltime_used=guard.elapsed_total())
                    ckpt_mod.save(cfg, state)
                    if guard.over():
                        logger.flush()
                        print(f"[TIME GUARD] saved resumable checkpoint at "
                              f"window {k} stage {stage} it {it + 1}; exiting cleanly.")
                        sys.exit(0)
                if cfg.causal and (it + 1) % cfg.check_every == 0 \
                        and w_min > cfg.wmin_threshold:
                    break
            state["it"] = 0  # stage finished cleanly
        state["stage"] = 0

        # --- window done: stitch predictions on ref grid, hand off IC ---
        t_loc, coords = case.eval_points_local(k)
        pred = predict_grid(net, t_loc, coords, device, n_comp=case.n_comp)
        logger.window_done(k, state["win_step"], net, opt, pred, state["stitched"],
                           predict_grid, residual_grid)
        t1_local = case.T_w * case.t_scale
        ic_pred = predict_grid(net, [t1_local], ic_coords, device, n_comp=case.n_comp)[0]
        state["ic_vals"] = ic_pred.astype(np.float32)
        logger.save_handoff(k, state["ic_vals"])
        state.update(window=k + 1, stage=0, it=0, win_step=0,
                     net_sd=None, opt_sd=None,
                     rng_state=rng.bit_generator.state,
                     walltime_used=guard.elapsed_total())
        state["win_step"] = 0
        ckpt_mod.save(cfg, state)

    logger.finalize(state["stitched"])
    return state
