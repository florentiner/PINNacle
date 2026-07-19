"""RunLogger: forensic data + PINNacle-parity artifacts for causal runs.

Produces the same file formats as the baseline HypothesisDataCallback
(arrays/, trajectory/, metrics.csv, errors.txt, *_pred.png, *_err.png) plus
causal-specific records (causal/history.npz, window IC handoffs).
"""
import os
import time

import numpy as np
import torch

from src.utils import forensic, plot
from src.utils.forensic import METRIC_KEYS


class RunLogger:

    def __init__(self, case, cfg, resume=False):
        self.case = case
        self.cfg = cfg
        base = cfg.outdir.rstrip("/") + "/"
        self.base = base
        self.arrays = base + "arrays/"
        self.traj = base + "trajectory/"
        self.causal_dir = base + "causal/"
        self.windows_dir = base + "windows/"
        for d in (base, self.arrays, self.traj, self.causal_dir, self.windows_dir):
            os.makedirs(d, exist_ok=True)

        self.metrics_path = base + "metrics.csv"
        self.metric_cols = ["step", "window", "stage", "tol", "it_in_stage", "walltime_s",
                            "lr", "loss", "loss_ic", "loss_res", "w_min",
                            "l2re_window", "l2re_global"] + METRIC_KEYS
        if not resume or not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w") as f:
                f.write(",".join(self.metric_cols) + "\n")

        self.hist_path = self.causal_dir + "history.npz"
        if resume and os.path.exists(self.hist_path):
            h = np.load(self.hist_path, allow_pickle=False)
            self.hist = {k: list(h[k]) for k in h.files}
        else:
            self.hist = {"step": [], "window": [], "stage": [], "tol": [],
                         "W": [], "L_t": [], "t_r": [], "w_min": [], "loss_ic": []}

        np.save(self.arrays + "ref.npy", case.ref.astype(np.float32))
        forensic.write_json(self.arrays + "grid_meta.json", case.mapper.meta({
            "pde": type(case).__name__,
            "case": case.name,
            "n_windows": case.n_windows,
            "steps_per_win": int(case.steps_per_win),
            "window_T": float(case.T_w),
            "t_scale": float(case.t_scale),
        }))
        forensic.write_json(self.base + "run_config.json", {
            "method": ("causal-pinn" if cfg.causal else "causal-arch-ablation")
                      + f"-{cfg.encoding}",
            "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()},
            "env": forensic.env_info(),
        })
        ic_coords, ic_vals = case.ic_arrays()
        np.savez_compressed(self.base + "collocation.npz",
                            ic_coords=ic_coords, ic_vals=ic_vals,
                            batch_spec=np.array([cfg.n_t, cfg.n_s]),
                            seed=np.array([cfg.seed]))
        self._global_cols_done = 0

    # ---------- streaming logs ----------
    def log_step(self, window, stage, tol, it, win_step, loss, loss_ic, loss_res,
                 w_min, W, L_t, t_r, lr, walltime, net, predict_grid):
        # window-slice L2RE (current net) + global-so-far L2RE (stitched cols)
        t_loc, coords = self.case.eval_points_local(window)
        pred = predict_grid(net, t_loc, coords, torch.device(self.cfg.device),
                            n_comp=self.case.n_comp)
        ref_w = self._window_ref(window)
        l2_win = float(np.sqrt(((pred - ref_w) ** 2).mean() / (ref_w ** 2).mean()))
        row = {"step": win_step, "window": window, "stage": stage, "tol": tol,
               "it_in_stage": it, "walltime_s": walltime, "lr": lr, "loss": loss,
               "loss_ic": loss_ic, "loss_res": loss_res, "w_min": w_min,
               "l2re_window": l2_win, "l2re_global": np.nan}
        row.update({k: np.nan for k in METRIC_KEYS})
        with open(self.metrics_path, "a") as f:
            f.write(",".join(str(row[c]) for c in self.metric_cols) + "\n")
        h = self.hist
        h["step"].append(win_step); h["window"].append(window)
        h["stage"].append(stage); h["tol"].append(tol)
        h["W"].append(W.astype(np.float32)); h["L_t"].append(L_t.astype(np.float32))
        h["t_r"].append(np.asarray(t_r, dtype=np.float32))
        h["w_min"].append(w_min); h["loss_ic"].append(loss_ic)

    def flush(self):
        np.savez_compressed(self.hist_path,
                            **{k: np.asarray(v) for k, v in self.hist.items()})

    # ---------- field snapshots ----------
    def field_snapshot(self, window, win_step, net, predict_grid, residual_grid):
        dev = torch.device(self.cfg.device)
        t_loc, coords = self.case.eval_points_local(window)
        pred = predict_grid(net, t_loc, coords, dev, n_comp=self.case.n_comp)
        res = residual_grid(self.case, net, t_loc, coords, dev)
        tag = f"w{window}_{win_step}"
        np.save(self.arrays + f"pred_{tag}.npy", self._to_window_grid(pred))
        np.save(self.arrays + f"err_{tag}.npy",
                self._to_window_grid(pred) - self._window_ref_grid(window))
        np.save(self.arrays + f"resid_{tag}.npy", self._to_window_grid(res))
        self.flush()

    # ---------- window lifecycle ----------
    def window_done(self, window, win_step, net, opt, pred, stitched,
                    predict_grid, residual_grid):
        cols = self.case.window_ref_cols(window)
        self._stitch(stitched, cols, pred)
        self._global_cols_done = cols[-1] + 1

        dev = torch.device(self.cfg.device)
        t_loc, coords = self.case.eval_points_local(window)
        res = residual_grid(self.case, net, t_loc, coords, dev)
        wg = self._to_window_grid(pred)
        np.save(self.arrays + f"pred_w{window}_final.npy", wg)
        np.save(self.arrays + f"err_w{window}_final.npy",
                wg - self._window_ref_grid(window))
        np.save(self.arrays + f"resid_w{window}_final.npy", self._to_window_grid(res))
        forensic.save_checkpoint(self.traj + f"w{window}_final.pt",
                                 getattr(net, "_orig_mod", net), optimizer=opt,
                                 step=win_step, extra={"window": window})

        # parity plots for this window + running stitched artifacts
        wdir = self.windows_dir + f"{window}/"
        os.makedirs(wdir, exist_ok=True)
        self._plots(wg, self._window_ref_grid(window), self._window_points(window), wdir)
        self._stitched_artifacts(stitched, partial=True)
        # global L2RE so far into metrics
        mask = ~np.isnan(stitched)
        l2g = float(np.sqrt(((stitched[mask] - self.case.ref[mask]) ** 2).mean()
                            / (self.case.ref[mask] ** 2).mean()))
        with open(self.metrics_path, "a") as f:
            row = {c: np.nan for c in self.metric_cols}
            row.update({"step": win_step, "window": window, "stage": -1, "tol": -1,
                        "it_in_stage": -1, "walltime_s": time.time(),
                        "l2re_global": l2g})
            f.write(",".join(str(row[c]) for c in self.metric_cols) + "\n")
        self.flush()
        print(f"[WINDOW {window}] done at win_step {win_step}; global L2RE so far = {l2g:.4e}")

    def save_handoff(self, window, ic_vals):
        np.save(self.causal_dir + f"window_{window}_handoff_ic.npy", ic_vals)

    # ---------- finalize ----------
    def finalize(self, stitched):
        self._stitched_artifacts(stitched, partial=False)
        ref, pts = self.case.ref, self.case.mapper.points
        mask = ~np.isnan(stitched)  # partial runs: metrics over covered region only
        m = forensic.metric_row(stitched[mask].reshape(-1), ref[mask].reshape(-1))
        m["coverage"] = float(mask.mean())
        with open(self.base + "errors.txt", "w") as f:
            f.write("# final stitched metrics vs ref (TesterCallback formulas)\n")
            for k, v in m.items():
                f.write(f"{k} {v:.10e}\n")
        # model_output.txt parity (points + prediction values in ref-row order)
        vals_rows = stitched.reshape(-1, self.case.n_comp)[self.case.mapper.flat_idx]
        out = np.concatenate([pts, vals_rows], axis=1)
        np.savetxt(self.base + "model_output.txt", out,
                   header=f"pde: {type(self.case).__name__} (causal stitched)")
        self.flush()
        print(f"[FINAL] stitched L2RE = {m['l2re']:.4e}  (errors.txt written)")

    # ---------- helpers ----------
    def _window_ref(self, window):
        """ref values on the window eval grid, shape (n_t, n_pts, n_comp)."""
        return self._window_ref_grid(window)

    def _window_ref_grid(self, window):
        cols = self.case.window_ref_cols(window)
        if self.case.name == "ks":     # ref (512, 251, 1) -> (n_t, 512, 1)
            return np.moveaxis(self.case.ref[:, cols, :], 1, 0)
        else:                          # ref (100,100,21,2) -> (n_t, 10000, 2)
            g = np.moveaxis(self.case.ref[:, :, cols, :], 2, 0)
            return g.reshape(g.shape[0], -1, self.case.n_comp)

    def _to_window_grid(self, pred):
        return np.asarray(pred, dtype=np.float32)

    def _window_points(self, window):
        t_loc, coords = self.case.eval_points_local(window)
        cols = self.case.window_ref_cols(window)
        t_glob = self.case.t_star[cols]
        n_pts = len(coords)
        tt = np.repeat(t_glob, n_pts)[:, None]
        cc = np.tile(coords, (len(t_glob), 1))
        return np.concatenate([cc, tt], axis=1)  # (x[,y], t) column order like ref

    def _stitch(self, stitched, cols, pred):
        """pred: (n_t, n_pts, n_comp) -> write into stitched ref-shaped grid."""
        if self.case.name == "ks":
            stitched[:, cols, :] = np.moveaxis(pred, 0, 1)
        else:
            g = pred.reshape(len(cols), len(self.case.xs), len(self.case.ys),
                             self.case.n_comp)
            stitched[:, :, cols, :] = np.moveaxis(g, 0, 2)

    def _plots(self, pred, ref, pts, outdir):
        names = ["u"] if self.case.n_comp == 1 else ["u", "v"]
        p2 = pred.reshape(-1, self.case.n_comp)
        r2 = ref.reshape(-1, self.case.n_comp)
        for i, nm in enumerate(names):
            if self.case.name == "ks":
                plot.plot_heatmap(pts[:, 0], pts[:, 1], p2[:, i],
                                  outdir + f"{nm}_pred.png",
                                  title=f"Prediction for {nm}", xlabel="x", ylabel="t")
                plot.plot_heatmap(pts[:, 0], pts[:, 1], p2[:, i] - r2[:, i],
                                  outdir + f"{nm}_err.png",
                                  title=f"Error for {nm}", xlabel="x", ylabel="t")
            else:
                plot.plot_3dheatmap(pts[:, 0], pts[:, 1], pts[:, 2], p2[:, i],
                                    outdir + f"{nm}_pred.png", title=f"Prediction for {nm}")
                plot.plot_3dheatmap(pts[:, 0], pts[:, 1], pts[:, 2], p2[:, i] - r2[:, i],
                                    outdir + f"{nm}_err.png", title=f"Error for {nm}")

    def _stitched_artifacts(self, stitched, partial):
        step_tag = "partial" if partial else "final"
        np.save(self.arrays + f"pred_stitched_{step_tag}.npy", stitched)
        np.save(self.arrays + f"err_stitched_{step_tag}.npy",
                stitched - self.case.ref.astype(np.float32))
        # root-level parity heatmaps over the full domain (NaN cols left blank)
        mask_col = ~np.isnan(stitched).reshape(-1, self.case.n_comp).any(axis=1)
        pts = self.case.mapper.points
        vals = np.nan_to_num(stitched.reshape(-1, self.case.n_comp))
        # grid order: mapper.points rows correspond to flat_idx positions
        vals_rows = stitched.reshape(-1, self.case.n_comp)[self.case.mapper.flat_idx]
        ref_rows = self.case.ref.reshape(-1, self.case.n_comp)[self.case.mapper.flat_idx]
        ok = ~np.isnan(vals_rows).any(axis=1)
        names = ["u"] if self.case.n_comp == 1 else ["u", "v"]
        for i, nm in enumerate(names):
            if self.case.name == "ks":
                plot.plot_heatmap(pts[ok, 0], pts[ok, 1], vals_rows[ok, i],
                                  self.base + f"{nm}_pred.png",
                                  title=f"Prediction for {nm}", xlabel="x", ylabel="t")
                plot.plot_heatmap(pts[ok, 0], pts[ok, 1], vals_rows[ok, i] - ref_rows[ok, i],
                                  self.base + f"{nm}_err.png",
                                  title=f"Error for {nm}", xlabel="x", ylabel="t")
            else:
                plot.plot_3dheatmap(pts[ok, 0], pts[ok, 1], pts[ok, 2], vals_rows[ok, i],
                                    self.base + f"{nm}_pred.png", title=f"Prediction for {nm}")
                plot.plot_3dheatmap(pts[ok, 0], pts[ok, 1], pts[ok, 2],
                                    vals_rows[ok, i] - ref_rows[ok, i],
                                    self.base + f"{nm}_err.png", title=f"Error for {nm}")
