"""HypothesisDataCallback: forensic data logging for baseline DeepXDE runs.

Saves everything needed to analyse WHY a run succeeded/failed without re-running:
  arrays/ref.npy, arrays/pred_{step}.npy, arrays/err_{step}.npy, arrays/resid_{step}.npy
  arrays/grid_meta.json, trajectory/ckpt_{step}.pt (+final), metrics.csv,
  collocation.npz, run_config.json
Formats match the causalpinn package (shared helpers in src/utils/forensic.py).
"""
import os
import time

import numpy as np
import torch
from deepxde.callbacks import Callback

from src.utils import forensic


class HypothesisDataCallback(Callback):

    def __init__(self, log_every=100, ckpt_every=2000, resid_chunk=4096):
        super().__init__()
        self.log_every = log_every
        self.ckpt_every = ckpt_every
        self.resid_chunk = resid_chunk
        self.valid_epoch = 0
        self.rows = []
        self.t0 = None

    # ---------- setup ----------
    def on_train_begin(self):
        self.t0 = time.time()
        pde = self.model.pde
        base = self.model.model_save_path + "/"
        self.arrays_dir = base + "arrays/"
        self.traj_dir = base + "trajectory/"
        os.makedirs(self.arrays_dir, exist_ok=True)
        os.makedirs(self.traj_dir, exist_ok=True)
        self.base = base

        assert pde.ref_data is not None, "HypothesisDataCallback needs pde.ref_data"
        self.mapper = forensic.GridMapper(pde.ref_data, pde.input_dim, pde.output_dim)
        np.save(self.arrays_dir + "ref.npy", self.mapper.ref_grid())
        forensic.write_json(self.arrays_dir + "grid_meta.json", self.mapper.meta({
            "pde": type(pde).__name__,
            "input_dim": pde.input_dim,
            "output_dim": pde.output_dim,
            "bbox": list(pde.bbox),
            "loss_config": pde.loss_config,
        }))

        # exact collocation points used for training/testing
        data = self.model.data
        np.savez_compressed(
            self.base + "collocation.npz",
            train_x_all=np.asarray(data.train_x_all),
            train_x=np.asarray(data.train_x),
            train_x_bc=np.asarray(data.train_x_bc) if data.train_x_bc is not None else np.empty(0),
            num_bcs=np.asarray(data.num_bcs),
            test_x=np.asarray(self.mapper.points),
        )

        forensic.write_json(self.base + "run_config.json", {
            "method": "vanilla-deepxde",
            "pde": type(pde).__name__,
            "net": str(self.model.net),
            "optimizer": str(self.model.opt),
            "env": forensic.env_info(),
            "log_every": self.log_every,
            "ckpt_every": self.ckpt_every,
        })

        self._find_tester()
        self._snapshot(0)

    def _find_tester(self):
        self.tester = None
        cbl = getattr(self.model, "callbacks", None)
        for cb in getattr(cbl, "callbacks", []):
            if type(cb).__name__ == "TesterCallback":
                self.tester = cb

    # ---------- per-epoch ----------
    def on_epoch_end(self):
        self.valid_epoch += 1
        if self.valid_epoch % self.log_every == 0:
            self._record_metrics_row()
        if self.valid_epoch % self.ckpt_every == 0:
            self._snapshot(self.valid_epoch)

    def _record_metrics_row(self):
        row = {"step": self.valid_epoch, "walltime_s": time.time() - self.t0}
        try:
            row["lr"] = self.model.opt.param_groups[0]["lr"]
        except Exception:
            row["lr"] = np.nan
        # mirror TesterCallback's freshly appended values (it runs before us)
        if self.tester is not None and len(self.tester.indexes) and \
                self.tester.indexes[-1] == self.valid_epoch:
            t = self.tester
            row.update({"mse": t.mses[-1], "mae": t.maes[-1], "mxe": t.mxes[-1],
                        "l1re": t.l1res[-1], "l2re": t.l2res[-1], "crmse": t.crmses[-1],
                        "frmse_low": t.frmses[-1][0], "frmse_mid": t.frmses[-1][1],
                        "frmse_high": t.frmses[-1][2]})
        lh = self.model.losshistory
        if len(lh.steps):
            for i, v in enumerate(np.asarray(lh.loss_train[-1]).ravel()):
                row[f"loss_train_{i}"] = v
            for i, v in enumerate(np.asarray(lh.loss_test[-1]).ravel()):
                row[f"loss_test_{i}"] = v
            row["loss_train_total"] = float(np.sum(lh.loss_train[-1]))
        self.rows.append(row)

    # ---------- snapshots ----------
    def _snapshot(self, step):
        pred = self.model.predict(self.mapper.points)
        pred_grid = self.mapper.to_grid(pred)
        np.save(self.arrays_dir + f"pred_{step}.npy", pred_grid.astype(np.float32))
        np.save(self.arrays_dir + f"err_{step}.npy",
                (pred_grid - self.mapper.ref_grid()).astype(np.float32))
        np.save(self.arrays_dir + f"resid_{step}.npy", self._residual_field().astype(np.float32))
        forensic.save_checkpoint(self.traj_dir + f"ckpt_{step}.pt", self.model.net,
                                 optimizer=self.model.opt, step=step)

    def _residual_field(self):
        pde = self.model.pde

        def op(x, y):
            res = pde.pde(x, y)
            if not isinstance(res, (list, tuple)):
                res = [res]
            res = [r if r.dim() == 2 else r.unsqueeze(-1) for r in res]
            return torch.cat(res, dim=1)

        pts = self.mapper.points
        out = []
        for i in range(0, len(pts), self.resid_chunk):
            out.append(self.model.predict(pts[i:i + self.resid_chunk], operator=op))
        res = np.concatenate(out, axis=0)
        return self.mapper.to_grid(res)

    # ---------- teardown ----------
    def on_train_end(self):
        step = self.model.train_state.epoch
        self._snapshot(step)
        forensic.save_checkpoint(self.traj_dir + "final.pt", self.model.net,
                                 optimizer=self.model.opt, step=step)
        keys = ["step", "walltime_s", "lr"] + forensic.METRIC_KEYS + \
               ["frmse_low", "frmse_mid", "frmse_high"]
        extra = sorted({k for r in self.rows for k in r} - set(keys))
        forensic.write_metrics_csv(self.base + "metrics.csv", self.rows, keys + extra)
