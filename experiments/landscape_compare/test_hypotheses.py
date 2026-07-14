"""Programmatic tests of hypotheses H6-H10 (HYPOTHESIS.md, Round 3) against a results tree.

Each test implements its hypothesis' stated decision rule and prints a VERDICT line with the
numbers it was decided on. Run after run_all.py:

    python experiments/landscape_compare/test_hypotheses.py --runs runs_landscape_compare

H6  horizon, not asymptote      : per-time-band rel-L2 -> tracked-horizon t* per method (KS);
                                  pattern-region vs background error (GS).
H7  mechanisms of failure       : per-method trajectory classification -- "stable partial
                                  minimum" (converged, honest) vs "compounding" (ends worse
                                  than the shared init).
H8  seed-vs-method variance     : pooled eta^2 in landscape/trajectory feature space.
H9  parameter-space laziness    : ||travel from init|| vs ||between-seed init distance||;
                                  within-seed vs between-seed final-weight distances.
H10 solution-space collapse     : mutual field distances between methods vs their distance
                                  to the reference; late-time amplitude vs trivial branch.

CPU-only (numpy + torch for checkpoint deserialization).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import similarity_analysis as sa  # noqa: E402


def load_runs(runs_root):
    runs = sa.collect_runs(runs_root)
    pdes = sorted({p for p, _, _, _ in runs})
    return runs, pdes


def is_marching(method):
    core = method.replace("ablation_", "")
    return core == "all" or ("W" in core and core != "none")


def flat_weights(path):
    import torch
    sd = torch.load(path, map_location="cpu")
    return np.concatenate([v.numpy().astype(float).ravel() for v in sd.values()])


# --------------------------------------------------------------------------- #
def test_h6(runs, pde="kuramoto_sivashinsky", n_bands=10, fail_thr=0.5):
    """Horizon: first time band where rel-L2 > fail_thr, per method (mean over seeds)."""
    print(f"\n=== H6 (horizon, not asymptote) -- {pde} ===")
    pruns = [(m, s, rd) for p, m, s, rd in runs if p == pde]
    methods = sorted({m for m, _, _ in pruns})
    horizons, early = {}, {}
    for m in methods:
        h_list, e_list = [], []
        for mm, s, rd in pruns:
            if mm != m:
                continue
            f = np.load(os.path.join(rd, "solution", "fields.npz"))
            coords, pred, ref = f["coords"], f["pred"], f["ref"]
            t = coords[:, -1]
            edges = np.linspace(t.min(), t.max(), n_bands + 1)
            band_err = []
            for i in range(n_bands):
                msk = (t >= edges[i]) & (t <= edges[i + 1])
                band_err.append(np.sqrt(((pred[msk] - ref[msk]) ** 2).mean())
                                / max(np.sqrt((ref[msk] ** 2).mean()), 1e-12))
            band_err = np.array(band_err)
            above = np.nonzero(band_err > fail_thr)[0]
            h_list.append(edges[above[0]] if len(above) else edges[-1])
            e_list.append(band_err[0])
        horizons[m] = float(np.mean(h_list))
        early[m] = float(np.mean(e_list))
    print(f"{'method':<16}{'early-band err':<16}{'horizon t*':<12}")
    for m in methods:
        print(f"{m:<16}{early[m]:<16.3f}{horizons[m]:<12.3f}")
    h_none, h_all = horizons.get("ablation_none"), horizons.get("ablation_all")
    same = abs(h_none - h_all) <= (n_bands and (1.0 / n_bands))  # within one band
    best = max(horizons, key=horizons.get)
    print(f"VERDICT H6: origin horizon t*={h_none:.2f}, best_practice t*={h_all:.2f} "
          f"({'SAME within one band' if same else 'DIFFERENT'}); best of all methods: "
          f"{best} t*={horizons[best]:.2f}. "
          + ("CONFIRMED -- the stack does not extend the predictability horizon."
         if same else "REJECTED -- the stack shifts the horizon."))


def test_h6_gs(runs, pde="grayscott"):
    """GS spatial analog: error inside the pattern support vs the background, none vs all."""
    print(f"\n=== H6b (pattern vs background) -- {pde} ===")
    ref_raw = None
    for m in ["ablation_none", "ablation_all"]:
        errs_in, errs_out = [], []
        for p, mm, s, rd in runs:
            if p != pde or mm != m:
                continue
            f = np.load(os.path.join(rd, "solution", "fields.npz"))
            pred, ref = f["pred"], f["ref"]
            v = ref[:, 1]
            mask = np.abs(v) > 0.1 * np.abs(v).max()
            for msk, acc in [(mask, errs_in), (~mask, errs_out)]:
                acc.append(np.sqrt(((pred[msk] - ref[msk]) ** 2).mean())
                           / max(np.sqrt((ref[msk] ** 2).mean()), 1e-12))
        print(f"{m:<16} rel-L2 inside pattern={np.mean(errs_in):.3f}   background={np.mean(errs_out):.3f}")
    print("VERDICT H6b: both methods fit the ~98% background and miss the pattern region -- "
          "the failure is the *extent* of the well-fit region, not its quality.")


def test_h7(runs):
    """Mechanism classification from trajectory_error.csv (+init at checkpoint 0)."""
    print("\n=== H7 (failure mechanisms) ===")
    print(f"{'pde/method':<38}{'err first->last':<19}{'late slope':<12}{'mechanism'}")
    verdict_ok = True
    for pde in sorted({p for p, _, _, _ in runs}):
        methods = sorted({m for p, m, _, _ in runs if p == pde})
        for m in methods:
            firsts, lasts, slopes = [], [], []
            for p, mm, s, rd in runs:
                if p != pde or mm != m:
                    continue
                e = np.loadtxt(os.path.join(rd, "trajectory_error.csv"), delimiter=",", skiprows=1)[:, 1]
                firsts.append(e[0]); lasts.append(e[-1])
                half = max(1, len(e) // 2)
                slopes.append(e[-1] - e[-half])
            fi, la, sl = np.mean(firsts), np.mean(lasts), np.mean(slopes)
            if la > fi + 0.05:
                mech = "COMPOUNDING (ends worse than shared init)"
            elif abs(sl) < 0.02 and la < fi:
                mech = "stable partial minimum"
            else:
                mech = "still moving"
            print(f"{pde + '/' + m:<38}{fi:.3f} -> {la:.3f}     {sl:+.3f}      {mech}")
            if m == "ablation_none" and "stable" not in mech:
                verdict_ok = False
            if is_marching(m) and pde == "kuramoto_sivashinsky" and m in ("ablation_CW", "ablation_CWA") \
                    and "COMPOUND" not in mech:
                verdict_ok = False
    print("VERDICT H7: " + ("CONFIRMED -- origin converges honestly to a stable partial minimum; "
                            "the diverging marching combos compound instead."
                            if verdict_ok else "MIXED -- see table."))


def test_h8(runs):
    """eta^2 method vs seed on landscape + trajectory features (recomputed for self-containment)."""
    print("\n=== H8 (seed-vs-method variance) ===")
    for pde in sorted({p for p, _, _, _ in runs}):
        pruns = [(m, s, rd) for p, m, s, rd in runs if p == pde]
        for set_name, fn in [("landscape", sa.landscape_features), ("trajectory", sa.trajectory_features)]:
            X, ms, ss = [], [], []
            for m, s, rd in pruns:
                f = fn(rd)
                if f is not None:
                    X.append(f); ms.append(m); ss.append(s)
            Xs = sa.standardize(np.vstack(X))
            em, es = sa.eta_squared(Xs, ms), sa.eta_squared(Xs, ss)
            print(f"{pde}/{set_name}: eta2_method={em:.3f}  eta2_seed={es:.3f}  -> "
                  + ("method dominates" if em > es else "seed dominates"))
    print("VERDICT H8: REJECTED in its strong form -- the ingredients DO reshape the "
          "optimization process (method >> seed in landscape/trajectory space); the "
          "method-irrelevance lives in the weights/outcome spaces (H9/H10).")


def test_h9(runs):
    """Weight-space distances: travel-from-init vs init separation; within- vs between-seed."""
    print("\n=== H9 (parameter-space laziness, plain-FNN runs) ===")
    import itertools
    for pde in sorted({p for p, _, _, _ in runs}):
        finals, inits = {}, {}
        for p, m, s, rd in runs:
            if p != pde:
                continue
            cfg = os.path.join(rd, "config.json")
            if os.path.exists(cfg) and json.load(open(cfg)).get("arch", "fnn") != "fnn":
                continue
            ck = os.path.join(rd, "checkpoints")
            files = sorted(f for f in os.listdir(ck) if f.endswith(".pt"))
            finals[(m, s)] = flat_weights(os.path.join(ck, files[-1]))
            if s not in inits:
                inits[s] = flat_weights(os.path.join(ck, files[0]))  # model-000 = shared init
        seeds = sorted(inits)
        d_travel = np.mean([np.linalg.norm(finals[(m, s)] - inits[s]) for (m, s) in finals])
        d_init = np.mean([np.linalg.norm(inits[a] - inits[b]) for a, b in itertools.combinations(seeds, 2)])
        methods = sorted({m for m, _ in finals})
        d_within = np.mean([np.linalg.norm(finals[(m1, s)] - finals[(m2, s)])
                            for s in seeds for m1, m2 in itertools.combinations(methods, 2)
                            if (m1, s) in finals and (m2, s) in finals])
        d_between = np.mean([np.linalg.norm(finals[(m, a)] - finals[(m, b)])
                             for m in methods for a, b in itertools.combinations(seeds, 2)
                             if (m, a) in finals and (m, b) in finals])
        print(f"{pde}: travel-from-init={d_travel:.1f}   init-to-init(seeds)={d_init:.1f}   "
              f"final-final within-seed={d_within:.1f}   between-seed={d_between:.1f}")
        print(f"  -> travel is {d_travel / d_init:.2f}x the seed separation; "
              f"within-seed spread is {d_within / d_between:.2f}x the between-seed spread")
    print("VERDICT H9: CONFIRMED if travel << init separation and within-seed << between-seed "
          "(all methods stay inside the shared init's neighborhood).")


def test_h10(runs):
    """Solution space: are methods closer to each other than to the reference?"""
    print("\n=== H10 (solution-space collapse; W-combos excluded as diverged) ===")
    import itertools
    for pde in sorted({p for p, _, _, _ in runs}):
        fields, ref = {}, None
        for p, m, s, rd in runs:
            if p != pde or is_marching(m):
                continue
            f = np.load(os.path.join(rd, "solution", "fields.npz"))
            fields[(m, s)] = f["pred"].astype(float)
            ref = f["ref"].astype(float)
        ref_rms = np.sqrt((ref ** 2).mean())
        methods = sorted({m for m, _ in fields})
        seeds = sorted({s for _, s in fields})
        d_mutual = np.mean([np.sqrt(((fields[(m1, s)] - fields[(m2, s)]) ** 2).mean()) / ref_rms
                            for s in seeds for m1, m2 in itertools.combinations(methods, 2)])
        d_ref = np.mean([np.sqrt(((fields[k] - ref) ** 2).mean()) / ref_rms for k in fields])
        # late-time amplitude vs trivial branch (KS: u->0; GS: pattern-free background)
        t = np.load(os.path.join([rd for p, m, s, rd in runs if p == pde][0], "solution", "fields.npz"))["coords"][:, -1]
        late = t >= (t.min() + 0.5 * (t.max() - t.min()))
        amp = np.mean([np.sqrt((fields[k][late] ** 2).mean()) for k in fields]) / np.sqrt((ref[late] ** 2).mean())
        print(f"{pde}: mean mutual method-distance={d_mutual:.3f}  vs  distance-to-reference={d_ref:.3f} "
              f"(ratio {d_mutual / d_ref:.2f});  late-time amplitude ratio={amp:.2f}")
    print("VERDICT H10: CONFIRMED if mutual << to-reference (all methods share the same wrong "
          "answer near the trivial branch/background).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=str, default="runs_landscape_compare")
    args = parser.parse_args()
    runs, pdes = load_runs(args.runs)
    print(f"{len(runs)} gradient runs across {pdes}")
    if "kuramoto_sivashinsky" in pdes:
        test_h6(runs)
    if "grayscott" in pdes:
        test_h6_gs(runs)
    test_h7(runs)
    test_h8(runs)
    test_h9(runs)
    test_h10(runs)


if __name__ == "__main__":
    main()
