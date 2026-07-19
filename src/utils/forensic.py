"""Shared helpers for the forensic data spec (used by both the baseline
HypothesisDataCallback and the causalpinn package): canonical grid mapping,
error metrics identical to TesterCallback, checkpoint and metadata saving.
"""
import json
import os
import subprocess
import time

import numpy as np
import torch


class GridMapper:
    """Maps between ref_data point rows (arbitrary but consistent ordering) and a
    canonical dense grid of shape (*axis_sizes, output_dim).

    Canonical axis order = ref_data input-column order (KS: (x, t); GS: (x, y, t)),
    so KS grids are (512, 251, 1) and GS grids are (100, 100, 21, 2).
    """

    def __init__(self, ref_data, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        nan_mask = np.isnan(ref_data).any(axis=1)
        self.points = ref_data[~nan_mask, :input_dim].astype(np.float64)
        self.values = ref_data[~nan_mask, input_dim:input_dim + output_dim]
        self.axes = [np.unique(self.points[:, i]) for i in range(input_dim)]
        self.shape = tuple(len(a) for a in self.axes)
        assert np.prod(self.shape) == len(self.points), \
            f"ref data is not a dense grid: {self.shape} vs {len(self.points)} rows"
        # flat canonical index of every ref row
        idx = np.zeros(len(self.points), dtype=np.int64)
        for i, ax in enumerate(self.axes):
            j = np.searchsorted(ax, self.points[:, i])
            # guard against float mismatch
            assert np.allclose(ax[j], self.points[:, i]), f"axis {i} lookup mismatch"
            idx = idx * len(ax) + j
        assert len(np.unique(idx)) == len(idx), "duplicate grid points in ref data"
        self.flat_idx = idx

    def to_grid(self, point_values):
        """(N, output_dim) values in ref-row order -> (*shape, output_dim) grid."""
        point_values = np.asarray(point_values)
        if point_values.ndim == 1:
            point_values = point_values[:, None]
        out = np.full((int(np.prod(self.shape)), point_values.shape[1]), np.nan,
                      dtype=point_values.dtype)
        out[self.flat_idx] = point_values
        return out.reshape(*self.shape, point_values.shape[1])

    def ref_grid(self):
        return self.to_grid(self.values)

    def meta(self, extra=None):
        m = {
            "axes": [a.tolist() for a in self.axes],
            "axis_sizes": list(self.shape),
            "grid_array_shape": list(self.shape) + [self.output_dim],
            "note": "arrays/*.npy have shape grid_array_shape; axes[i] gives the "
                    "coordinate values along dim i (input-column order of ref data)",
        }
        if extra:
            m.update(extra)
        return m


# --- metrics: formulas identical to src/utils/callbacks.py TesterCallback ---
def metric_row(y_pred, y_ref):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)
    mse = ((y_pred - y_ref) ** 2).mean()
    mae = np.abs(y_pred - y_ref).mean()
    mxe = np.max(np.abs(y_pred - y_ref))
    l1re = mae / np.abs(y_ref).mean()
    l2re = np.sqrt(mse) / np.sqrt((y_ref ** 2).mean())
    crmse = np.abs((y_pred - y_ref).mean())
    return {"mse": mse, "mae": mae, "mxe": mxe, "l1re": l1re, "l2re": l2re, "crmse": crmse}


METRIC_KEYS = ["mse", "mae", "mxe", "l1re", "l2re", "crmse"]


def save_checkpoint(path, net, optimizer=None, step=None, extra=None):
    payload = {
        "step": step,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_rng_states(payload):
    torch.set_rng_state(payload["torch_rng_state"].cpu()
                        if isinstance(payload["torch_rng_state"], torch.Tensor)
                        else payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    if "cuda_rng_state" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu())


def env_info():
    info = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        info["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        info["git_hash"] = None
    return info


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def write_metrics_csv(path, rows, keys):
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r.get(k, np.nan)}" for k in keys) + "\n")
