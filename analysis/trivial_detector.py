"""Reference-free trivial-solution detector, built from the error-space patterns
measured in the trivial-attractor study (analysis/TRIVIAL_HYPOTHESIS.md).

Signals (all computable WITHOUT the reference solution):
  C  - residual time-concentration: share of total squared PDE residual in the
       earliest 5% of the time axis. Trivial collapse pays its residual in a thin
       causality-violating layer at t~0 (measured: 91.6% in 1% of t on Heat-LT).
  A  - late-amplitude ratio: RMS(u) over the last half of the time axis divided by
       RMS(u) at the initial slice. Trivial branch decays to a constant state ->
       A ~ 0 for zero-trivial; more generally late-time variance dies.
  G  - Gini-style concentration of squared residual over time bins (0=uniform,
       ->1=all in one bin). Robust version of C without the 5% cutoff choice.
score = C * (1 - min(A, 1))  in [0,1]: high = trivial-branch signature.

Works on the forensic arrays: resid_{step}.npy / pred_{step}.npy on the canonical
grid where the LAST axis before components is time (KS: (x,t,1); heat: (x,y,t,1);
GS: (x,y,t,2)).
"""
import glob
import json
import os
import re

import numpy as np


def _step_of(p):
    m = re.search(r"_(\d+)\.npy$", p)
    return int(m.group(1)) if m else -1


def signals(pred, resid, t_frac=0.05):
    """pred, resid: (..., nt, ncomp) arrays on the canonical grid."""
    sq = (resid ** 2).mean(axis=tuple(range(resid.ndim - 2)))   # (nt, ncomp)
    sq = sq.sum(axis=-1)                                        # (nt,)
    nt = len(sq)
    k = max(1, int(round(t_frac * nt)))
    total = sq.sum() + 1e-30
    C = float(sq[:k].sum() / total)
    # amplitude: deviation from the final-time state (constant trivial state has
    # zero late variance even if the constant is nonzero, e.g. GS background)
    u = pred
    late = u[..., nt // 2:, :]
    a_late = float(np.sqrt(((late - late.mean(axis=tuple(range(late.ndim - 2)),
                                              keepdims=True)) ** 2).mean()))
    a_ic = float(np.sqrt(((u[..., :1, :] - u[..., :1, :].mean()) ** 2).mean()))
    A = a_late / (a_ic + 1e-12)
    # Gini of sq over time
    s = np.sort(sq) / total
    G = float(1 - 2 * np.sum(np.cumsum(s) / nt) + 1 / nt)
    score = C * (1 - min(A, 1.0))
    C_enrich = C / t_frac                       # 1.0 = uniform-in-time residual
    flag_front = C_enrich > 3.0                 # causality-violating early layer
    flag_dead = A < 0.10                        # late-time dynamics switched off
    return {"C_early_resid_share": C, "C_enrich": C_enrich,
            "A_late_amp_ratio": A, "G_resid_gini": G,
            "flag_front_collapse": bool(flag_front), "flag_dead_dynamics": bool(flag_dead),
            "trivial_flag": bool(flag_front or flag_dead),
            "trivial_score": score}


def analyze_run(arrays_dir, label, t_frac=0.05):
    preds = sorted(glob.glob(os.path.join(arrays_dir, "pred_*.npy")), key=_step_of)
    resids = sorted(glob.glob(os.path.join(arrays_dir, "resid_*.npy")), key=_step_of)
    rows = []
    for p, r in zip(preds, resids):
        sp, sr = _step_of(p), _step_of(r)
        if sp != sr:
            continue
        sig = signals(np.load(p), np.load(r), t_frac=t_frac)
        sig["step"] = sp
        sig["run"] = label
        rows.append(sig)
    return rows


if __name__ == "__main__":
    import sys
    runs = [
        ("heat-lt vanilla", "runs/kaggle-trivial-vanilla/08.10-15.20.22-trivial-heatlt-vanilla/0-0/arrays"),
        ("KS vanilla", "runs/07.18-13.19.39-baseline-chaotic/0-0/arrays"),
        ("GS vanilla", "runs/07.18-13.19.39-baseline-chaotic/1-0/arrays"),
    ]
    if len(sys.argv) > 2:
        runs = [(sys.argv[1], sys.argv[2])]
    out = []
    for label, d in runs:
        rows = analyze_run(d, label)
        out.extend(rows)
        if rows:
            first, last = rows[0], rows[-1]
            print(f"{label:18s} step {first['step']:>6}: score {first['trivial_score']:.3f} "
                  f"(C {first['C_early_resid_share']:.3f} A {first['A_late_amp_ratio']:.3f}) "
                  f"-> step {last['step']:>6}: score {last['trivial_score']:.3f} "
                  f"(C {last['C_early_resid_share']:.3f} A {last['A_late_amp_ratio']:.3f} "
                  f"G {last['G_resid_gini']:.3f})")
    with open("analysis/out/trivial_detector_rows.json", "w") as f:
        json.dump(out, f, indent=1)
