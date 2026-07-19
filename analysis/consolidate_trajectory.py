"""Consolidate per-snapshot param files into single easy-to-work-with arrays.

For each window with snapshots under <run>/trajectory/:
  <run>/trajectory/w{k}_trajectory_flat.npy   (n_snaps+1, n_params) float32
                                              (rows ordered by iter; last = final)
  <run>/trajectory/w{k}_trajectory_steps.npy  (n_snaps+1,) int64
Baseline torch runs (ckpt_*.pt) get the same treatment:
  <run>/trajectory/trajectory_flat.npy / trajectory_steps.npy

Usage: python analysis/consolidate_trajectory.py <run_dir> [<run_dir> ...]
"""
import glob
import os
import re
import sys

import numpy as np


def flat_from_npz(f):
    z = np.load(f)
    n = int(z["n_layers"])
    parts = [z["U1"], z["b1"], z["U2"], z["b2"]]
    for i in range(n):
        parts += [z[f"W{i}"], z[f"bb{i}"]]
    return np.concatenate([p.ravel() for p in parts]).astype(np.float32)


def flat_from_pt(f):
    import torch
    sd = torch.load(f, map_location="cpu", weights_only=False)["model_state_dict"]
    return np.concatenate([v.numpy().ravel() for v in sd.values()]).astype(np.float32)


def consolidate(run):
    traj = os.path.join(run, "trajectory")
    if not os.path.isdir(traj):
        print(f"[skip] no trajectory dir in {run}")
        return
    # JAX-style per-window snapshots
    wins = sorted({int(m.group(1)) for f in glob.glob(traj + "/w*_snap_*.npz")
                   for m in [re.search(r"w(\d+)_snap", f)] if m})
    for k in wins:
        snaps = sorted(glob.glob(traj + f"/w{k}_snap_*.npz"),
                       key=lambda s: int(re.search(r"snap_(\d+)", s).group(1)))
        steps = [int(re.search(r"snap_(\d+)", s).group(1)) for s in snaps]
        rows = [flat_from_npz(s) for s in snaps]
        fin = traj + f"/w{k}_final_params.npz"
        if os.path.exists(fin):
            rows.append(flat_from_npz(fin))
            steps.append(steps[-1] + 1 if steps else 0)
        np.save(traj + f"/w{k}_trajectory_flat.npy", np.stack(rows))
        np.save(traj + f"/w{k}_trajectory_steps.npy", np.array(steps))
        print(f"[ok] {run} w{k}: {len(rows)} snapshots -> "
              f"w{k}_trajectory_flat.npy {rows[0].shape[0]} params")
    # torch-style checkpoints (baseline)
    cks = sorted(glob.glob(traj + "/ckpt_*.pt"),
                 key=lambda s: int(re.search(r"ckpt_(\d+)", s).group(1)))
    if cks:
        steps = [int(re.search(r"ckpt_(\d+)", s).group(1)) for s in cks]
        rows = [flat_from_pt(c) for c in cks]
        np.save(traj + "/trajectory_flat.npy", np.stack(rows))
        np.save(traj + "/trajectory_steps.npy", np.array(steps))
        print(f"[ok] {run}: {len(rows)} torch ckpts -> trajectory_flat.npy")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        consolidate(r)
