"""Case definitions for the causal PINN runs.

PDE coefficients are read from the PINNacle classes in src/pde/chaotic.py via
inspect (single source of truth); ref data / grids come through the same
GridMapper the baseline forensic callback uses — guaranteeing an identical
evaluation protocol.
"""
import inspect

import numpy as np
import torch

from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.utils.forensic import GridMapper

from causalpinn.model import GS2DEncoding, KSEncoding, ModifiedMLP, PlainEncoding


def _defaults(cls):
    return {k: v.default for k, v in inspect.signature(cls.__init__).parameters.items()
            if v.default is not inspect.Parameter.empty}


class KSCase:
    """Chaotic KS: u_t + a u u_x + b u_xx + g u_xxxx = 0, x in [0,2pi], t in [0,1].
    10 time-marching windows x 25 ref steps (dt_ref=0.004) -> window length 0.1.
    Net input uses RAW window-local t (reference behavior)."""

    name = "ks"
    n_comp = 1

    def __init__(self, cfg):
        self.cfg = cfg
        p = _defaults(KuramotoSivashinskyEquation)
        self.alpha, self.beta, self.gamma = float(p["alpha"]), float(p["beta"]), float(p["gamma"])
        self.bbox = list(p["bbox"])  # [0, 2pi, 0, 1]
        pde = KuramotoSivashinskyEquation()
        self.mapper = GridMapper(pde.ref_data, pde.input_dim, pde.output_dim)  # axes: (x, t)
        self.x_star = self.mapper.axes[0]            # (512,) includes both endpoints
        self.t_star = self.mapper.axes[1]            # (251,)
        self.ref = self.mapper.ref_grid()            # (512, 251, 1)
        self.n_windows = cfg.windows                 # default 10
        self.steps_per_win = (len(self.t_star) - 1) // self.n_windows  # 25
        self.T_w = self.t_star[self.steps_per_win] - self.t_star[0]    # 0.1
        self.t_scale = 1.0                           # raw local time (reference)
        self.spatial_dim = 1

    def build_net(self, encoding_kind, seed, device):
        if encoding_kind == "fourier":
            enc = KSEncoding(M_t=self.cfg.M_t_ks, M_x=self.cfg.M_x_ks, L=2 * np.pi)
        else:
            enc = PlainEncoding(self.cfg.M_t_ks, 1, self.bbox)
        net = ModifiedMLP(enc, [self.cfg.width] * self.cfg.depth, self.n_comp, seed=seed)
        return net.to(device)

    def ic_arrays(self):
        """window-0 IC on the spatial grid."""
        return self.x_star[:, None], self.ref[:, 0, :]  # (512,1) coords, (512,1) values

    def sample_spatial(self, rng, n):
        return rng.uniform(0.0, 2 * np.pi, size=(n, 1))

    def residual(self, net, t, x, T_w):
        """Forward-mode (jvp) derivative chains — mathematically identical to
        nested reverse-mode but ~an order of magnitude cheaper for u_xxxx
        (the JAX reference uses Taylor-mode `jet` for the same reason)."""
        from torch.func import jvp
        v = torch.ones_like(x)

        def u_of_x(x_):
            return net(t, x_)

        def d1(x_):
            return jvp(u_of_x, (x_,), (v,))[1]

        def d2(x_):
            return jvp(d1, (x_,), (v,))[1]

        def d3(x_):
            return jvp(d2, (x_,), (v,))[1]

        u, u_x = jvp(u_of_x, (x,), (v,))
        u_xx = jvp(d1, (x,), (v,))[1]
        u_xxxx = jvp(d3, (x,), (v,))[1]
        u_t = jvp(lambda t_: net(t_, x), (t,), (torch.ones_like(t),))[1]
        return u_t + self.alpha * u * u_x + self.beta * u_xx + self.gamma * u_xxxx  # (N,1)

    def residual_reference(self, net, t, x, T_w):
        """Nested reverse-mode version (slow) kept for numerical cross-checks."""
        u = net(t, x)
        ones = torch.ones_like(u)
        u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
        u_xxxx = torch.autograd.grad(u_xxx, x, torch.ones_like(u_xxx), create_graph=True)[0]
        return u_t + self.alpha * u * u_x + self.beta * u_xx + self.gamma * u_xxxx  # (N,1)

    # --- evaluation helpers ---
    def window_ref_cols(self, k):
        """global t-column indices covered by window k (incl. both ends)."""
        s = self.steps_per_win
        return np.arange(k * s, (k + 1) * s + 1)

    def eval_points_local(self, k):
        """(t_local, coords) meshes for the window's ref columns."""
        cols = self.window_ref_cols(k)
        t_loc = self.t_star[cols] - self.t_star[cols[0]]
        return t_loc, self.x_star[:, None]


class GSCase:
    """Gray-Scott on [-1,1]^2, t in [0,200], 20 windows x 10 time units.
    Net input uses NORMALIZED window-local time tau = t_local / T_w in [0,1];
    residual rescaled by T_w (adaptation - the causal paper never did GS)."""

    name = "gs"
    n_comp = 2

    def __init__(self, cfg):
        self.cfg = cfg
        p = _defaults(GrayScottEquation)
        self.b, self.d = float(p["b"]), float(p["d"])
        self.eps = tuple(float(e) for e in p["epsilon"])
        self.bbox = list(p["bbox"])  # [-1,1,-1,1,0,200]
        pde = GrayScottEquation()
        self.mapper = GridMapper(pde.ref_data, pde.input_dim, pde.output_dim)  # axes: (x, y, t)
        self.xs, self.ys, self.t_star = self.mapper.axes
        self.ref = self.mapper.ref_grid()            # (100, 100, 21, 2)
        self.n_windows = cfg.windows                 # default 20
        self.steps_per_win = (len(self.t_star) - 1) // self.n_windows  # 1 (ref dt=10)
        self.T_w = self.t_star[self.steps_per_win] - self.t_star[0]    # 10.0
        self.t_scale = 1.0 / self.T_w
        self.spatial_dim = 2
        xx, yy = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.ic_grid = np.stack([xx.ravel(), yy.ravel()], axis=1)      # (10000, 2)

    def build_net(self, encoding_kind, seed, device):
        if encoding_kind == "fourier":
            enc = GS2DEncoding(M_t=self.cfg.M_t_gs, M_x=self.cfg.M_x_gs,
                               M_y=self.cfg.M_x_gs, L_x=2.0, L_y=2.0)
        else:
            enc = PlainEncoding(self.cfg.M_t_gs, 2, self.bbox)
        net = ModifiedMLP(enc, [self.cfg.width] * self.cfg.depth, self.n_comp, seed=seed)
        return net.to(device)

    def ic_arrays(self):
        return self.ic_grid, self.ref[:, :, 0, :].reshape(-1, 2)

    def sample_spatial(self, rng, n):
        return rng.uniform(-1.0, 1.0, size=(n, 2))

    def residual(self, net, t, xy, T_w):
        """Forward-mode (jvp) derivatives. t: (N,1) normalized local tau;
        xy: (N,2), inputs must NOT require grad (parameter grads flow through
        the net; input-grad edges both waste memory and break repeated backward
        when mixed with torch.func transforms)."""
        from torch.func import jvp
        x, y = xy[:, 0:1], xy[:, 1:2]
        vx, vy, vt = torch.ones_like(x), torch.ones_like(y), torch.ones_like(t)

        out, out_t = jvp(lambda t_: net(t_, x, y), (t,), (vt,))

        def dx1(x_):
            return jvp(lambda x2: net(t, x2, y), (x_,), (vx,))[1]

        def dy1(y_):
            return jvp(lambda y2: net(t, x, y2), (y_,), (vy,))[1]

        out_xx = jvp(dx1, (x,), (vx,))[1]
        out_yy = jvp(dy1, (y,), (vy,))[1]

        u, v = out[:, 0:1], out[:, 1:2]
        u_t, v_t = out_t[:, 0:1], out_t[:, 1:2]
        u_xx, v_xx = out_xx[:, 0:1], out_xx[:, 1:2]
        u_yy, v_yy = out_yy[:, 0:1], out_yy[:, 1:2]
        r_u = u_t - T_w * (self.eps[0] * (u_xx + u_yy) + self.b * (1 - u) - u * v ** 2)
        r_v = v_t - T_w * (self.eps[1] * (v_xx + v_yy) - self.d * v + u * v ** 2)
        return torch.cat([r_u, r_v], dim=1)  # (N,2)

    def window_ref_cols(self, k):
        s = self.steps_per_win
        return np.arange(k * s, (k + 1) * s + 1)

    def eval_points_local(self, k):
        cols = self.window_ref_cols(k)
        t_loc = (self.t_star[cols] - self.t_star[cols[0]]) * self.t_scale
        return t_loc, self.ic_grid


def get_case(name, cfg):
    return {"ks": KSCase, "gs": GSCase}[name](cfg)
