"""Reference-free anti-trivial intervention for vanilla PINN training.

No SOTA machinery (no causal weights, no time windows, no architecture change):
every `period` steps the callback computes the trivial-branch detector signals on a
random candidate pool (see analysis/trivial_detector.py, validated on Heat-LT/KS/GS):

  C_enrich - squared PDE residual share in the earliest 5% of time / 5%
             (>1 = residual concentrated in an early causality-violating layer)
  A_late   - late-time solution variability / initial-slice variability
             (~0 = dynamics switched off, frozen state)

If the trivial flag fires (C_enrich > 3 or A_late < 0.1), mode "rar" replaces a
fraction of the PDE collocation points with pool points sampled with probability
proportional to squared residual — concentrating the loss where the collapse front
lives, so the "cheap thin layer" trade that makes the trivial branch profitable
(measured: 91.6% of squared residual in 1% of the domain at zero extra cost)
stops being cheap. Mode "uniform" resamples uniformly on the same schedule
(control: separates "any resampling" from "pattern-driven resampling").
"""
import csv
import os

import numpy as np
import torch

from deepxde.callbacks import Callback


class TrivialGuardCallback(Callback):

    def __init__(self, mode="rar", period=1000, pool_size=20000, keep_frac=0.5,
                 resid_chunk=4096, log_dir=None, t_frac=0.05,
                 enrich_thr=3.0, dead_thr=0.10,
                 march_t0_frac=0.02, march_dt_frac=0.02, march_eps=3e-3,
                 veto=False, force_period_frac=0.08, veto_freeze_thr=0.2,
                 kick_mult=6.0, kick_iters=1500, max_vetoes=6):
        super().__init__()
        assert mode in ("rar", "uniform", "march", "off")
        self.mode = mode
        self.period = period
        self.pool_size = pool_size
        self.keep_frac = keep_frac
        self.resid_chunk = resid_chunk
        self.log_dir = log_dir
        self.t_frac = t_frac
        self.enrich_thr = enrich_thr
        self.dead_thr = dead_thr
        self.march_t0_frac = march_t0_frac
        self.march_dt_frac = march_dt_frac
        self.march_eps = march_eps
        self._tstar_frac = march_t0_frac
        self.veto = veto
        self.force_period_frac = force_period_frac
        self.veto_freeze_thr = veto_freeze_thr
        self.kick_mult = kick_mult
        self.kick_iters = kick_iters
        self.max_vetoes = max_vetoes
        self._vetoes = 0
        self._kick_left = 0
        self._base_lr = None
        self._since = 0
        self._rows = []
        self._bbox = None

    # ---------- helpers ----------
    def _pool(self):
        lo, hi = self._bbox
        return np.random.uniform(lo, hi, size=(self.pool_size, len(lo))).astype(np.float32)

    def _residual(self, pts):
        pde = self.model.pde

        def op(x, y):
            res = pde.pde(x, y)
            if not isinstance(res, (list, tuple)):
                res = [res]
            res = [r if r.dim() == 2 else r.unsqueeze(-1) for r in res]
            return torch.cat(res, dim=1)

        out = []
        for i in range(0, len(pts), self.resid_chunk):
            out.append(self.model.predict(pts[i:i + self.resid_chunk], operator=op))
        return np.concatenate(out, axis=0)

    def _signals(self, pts, u, r2):
        t = pts[:, -1]
        t0, t1 = float(t.min()), float(t.max())
        cut = t0 + self.t_frac * (t1 - t0)
        total = float(r2.sum()) + 1e-30
        C = float(r2[t < cut].sum()) / total
        c_enrich = C / self.t_frac
        med = 0.5 * (t0 + t1)
        u_late = u[t > med]
        u_ic = u[t < cut]
        a_late = float(np.sqrt(((u_late - u_late.mean(0)) ** 2).mean()))
        a_ic = float(np.sqrt(((u_ic - u_ic.mean(0)) ** 2).mean())) + 1e-12
        A = a_late / a_ic
        flag = (c_enrich > self.enrich_thr) or (A < self.dead_thr)
        return c_enrich, A, flag

    # ---------- callback API ----------
    def on_train_begin(self):
        xall = self.model.data.train_x_all
        self._bbox = (xall.min(axis=0), xall.max(axis=0))
        self._n_pde = len(xall)

    def on_epoch_end(self):
        self._since += 1
        if self._since < self.period:
            return
        self._since = 0
        step = self.model.train_state.epoch
        pts = self._pool()
        u = self.model.predict(pts)
        r = self._residual(pts)
        r2 = (r ** 2).sum(axis=1)
        c_enrich, A, flag = self._signals(pts, u, r2)
        acted = 0
        if self.mode == "march" and self._kick_left > 0:
            # kick phase: hold high LR, then restart the march from scratch
            self._kick_left -= self.period
            if self._kick_left <= 0:
                for gpar in self.model.opt.param_groups:
                    gpar["lr"] = self._base_lr
                self._tstar_frac = self.march_t0_frac
                print(f"[guard veto] kick done -> re-march from t*={self._tstar_frac}",
                      flush=True)
            return
        if self.mode == "march":
            # RL-policy emulation, vanilla loss untouched: the agent only controls the
            # SAMPLING distribution — collocation points live in t in [t_lo, t_lo + tstar]
            # and the horizon expands when the covered residual is clean. A discrete,
            # sampling-only version of causal marching.
            lo, hi = self._bbox
            span = hi[-1] - lo[-1]
            t_hi = lo[-1] + self._tstar_frac * span
            m_cov = pts[:, -1] <= t_hi
            r2_cov = float(r2[m_cov].mean()) if m_cov.any() else float("inf")
            if r2_cov < self.march_eps and self._tstar_frac < 1.0:
                self._tstar_frac = min(1.0, self._tstar_frac + self.march_dt_frac)
                t_hi = lo[-1] + self._tstar_frac * span
                acted = 1
            # rung veto: once the horizon covers >1.5 forcing periods, the trailing
            # period must show live dynamics (|u_t| at the tail); frozen tail =>
            # self-gated wrong rung => kick out and re-march
            if (self.veto and self._vetoes < self.max_vetoes
                    and self._tstar_frac > 1.5 * self.force_period_frac):
                tail_lo = t_hi - self.force_period_frac * span
                m_tail = (pts[:, -1] > tail_lo) & (pts[:, -1] <= t_hi)
                if m_tail.sum() > 200:
                    pts_t = pts[m_tail].copy()
                    eps_t = 0.01 * span
                    pts_p = pts_t.copy(); pts_p[:, -1] += eps_t
                    u0 = self.model.predict(pts_t); u1 = self.model.predict(pts_p)
                    ut = np.abs((u1 - u0) / eps_t).mean()
                    omega = 2 * np.pi / (self.force_period_frac * span)
                    f_freeze = float(ut / (0.5 * omega))
                    if f_freeze < self.veto_freeze_thr:
                        self._vetoes += 1
                        self._base_lr = self.model.opt.param_groups[0]["lr"]
                        for gpar in self.model.opt.param_groups:
                            gpar["lr"] = self._base_lr * self.kick_mult
                        self._kick_left = self.kick_iters
                        print(f"[guard veto] FROZEN TAIL (F_freeze {f_freeze:.3f} < "
                              f"{self.veto_freeze_thr}) at t*={self._tstar_frac:.2f} "
                              f"-> kick x{self.kick_mult} ({self._vetoes}/"
                              f"{self.max_vetoes})", flush=True)
                    else:
                        print(f"[guard veto-check] tail alive: F_freeze {f_freeze:.3f}",
                              flush=True)
            new = self._pool()[: self._n_pde]
            new[:, -1] = lo[-1] + np.random.rand(len(new)) * (t_hi - lo[-1])
            self.model.data.replace_with_anchors(new.astype(np.float32))
            self._rows.append({"step": step, "C_enrich": c_enrich, "A_late": A,
                               "flag": int(flag), "acted": acted, "mode": self.mode,
                               "tstar_frac": self._tstar_frac, "r2_cov": r2_cov})
            print(f"[guard march] step {step}: t*={self._tstar_frac:.2f} "
                  f"r2_cov {r2_cov:.2e} C_enrich {c_enrich:.2f} A {A:.3f}", flush=True)
            self._flush()
            return
        if self.mode == "uniform":
            self.model.data.resample_train_points(True, False)
            acted = 1
        elif self.mode == "rar" and flag:
            n_new = int(self._n_pde * (1 - self.keep_frac))
            n_keep = self._n_pde - n_new
            cur = self.model.data.train_x_all
            keep_idx = np.random.choice(len(cur), size=min(n_keep, len(cur)),
                                        replace=False)
            p = r2 / r2.sum()
            new_idx = np.random.choice(len(pts), size=n_new, replace=True, p=p)
            anchors = np.vstack([cur[keep_idx], pts[new_idx]])
            self.model.data.replace_with_anchors(anchors)
            acted = 1
        self._rows.append({"step": step, "C_enrich": c_enrich, "A_late": A,
                           "flag": int(flag), "acted": acted, "mode": self.mode})
        print(f"[guard {self.mode}] step {step}: C_enrich {c_enrich:.2f} "
              f"A_late {A:.3f} flag {int(flag)} acted {acted}", flush=True)
        self._flush()

    def _flush(self):
        if not self.log_dir:
            self.log_dir = self.model.model_save_path or "."
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "trivialguard_log.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            w.writeheader()
            w.writerows(self._rows)

    def on_train_end(self):
        if self._rows:
            self._flush()
