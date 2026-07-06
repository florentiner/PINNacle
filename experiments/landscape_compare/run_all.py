"""Matrix driver for the chaotic-PDE landscape comparison.

Runs the full {pde} x {method} matrix by launching run_experiment.py once per cell in
an isolated subprocess (so a failure or memory blow-up in one cell cannot take down the
others, and DeepXDE / torch global state never leaks between runs). Results are collated
into <out>/MANIFEST.json. The driver is resumable: a cell whose metrics.json already
exists is skipped unless --force is given.

This is the single entry point to run on the other machine:

    # smoke-test the whole default matrix first (tiny, ~minutes):
    python experiments/landscape_compare/run_all.py --quick

    # then the real thing:
    python experiments/landscape_compare/run_all.py

    # subset / options:
    python experiments/landscape_compare/run_all.py \
        --pdes kuramoto_sivashinsky grayscott \
        --methods adam_baseline lbfgs_baseline causal soap frozen \
        --iterations 15000

    # repeat each cell 3x with different seeds, to check the result is robust and not a fluke:
    python experiments/landscape_compare/run_all.py \
        --pdes kuramoto_sivashinsky grayscott --n-repeats 3

Repeats are nested under <out>/seed_<seed>/<pde>/<method>/ (only when more than one seed is
requested -- a plain single-seed run keeps the original flat <out>/<pde>/<method>/ layout).
compare_landscapes.py detects the nesting automatically and reports mean +/- std across seeds.
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_EXPERIMENT = os.path.join(os.path.dirname(__file__), "run_experiment.py")

ALL_PDES = ["kuramoto_sivashinsky", "grayscott", "burgers1d"]
# Default comparison, four methods covering the three candidate fixes from the literature plus
# the failure baseline -- see METHOD_SPEC in run_experiment.py for the exact paper-backed configs:
#   origin       - the failure case: Adam + lr-decay, plain MSE loss.
#   causal       - same Adam pipeline, causal time-weighted loss (Wang et al. 2022, 2203.07404).
#   soap_causal  - SOAP (Shampoo-preconditioned Adam) + causal loss, tuned per the paper that
#                  benchmarks SOAP on KS/Grey-Scott directly (arXiv:2502.00604); this IS "the
#                  SOAP/second-order option" -- not the same thing as `lbfgs_baseline` (Adam->
#                  L-BFGS), which is a different, weaker-on-chaotic optimizer axis kept only to
#                  confirm that plain L-BFGS underperforms.
#   frozen       - gradient-free control (Frozen-PINN).
DEFAULT_METHODS = ["origin", "causal", "soap_causal", "frozen"]
ALL_METHODS = ["origin", "causal", "adam_baseline", "lbfgs_baseline", "soap", "soap_causal", "frozen"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdes", nargs="+", default=ALL_PDES, choices=ALL_PDES)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=ALL_METHODS)
    parser.add_argument("--out", type=str, default=os.path.join(PROJECT_ROOT, "runs_landscape_compare"))
    parser.add_argument("--force", action="store_true", help="rerun cells even if metrics.json already exists")
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test settings (passed through)")
    # passthrough knobs
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--hidden-layers", type=str, default=None)
    parser.add_argument("--n-save-models", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="base seed (also used as-is for a single run)")
    parser.add_argument("--n-repeats", type=int, default=1,
                        help="repeat every cell this many times with seeds base, base+1, .. (base = --seed or 1234)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="explicit seed list, overrides --n-repeats; implies the nested seed_<N>/ layout")
    parser.add_argument("--ae-epochs", type=int, default=None)
    parser.add_argument("--grid-xnum", type=int, default=None)
    parser.add_argument("--causal-eps", type=float, default=None)
    parser.add_argument("--num-causal-buckets", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "MANIFEST.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    if args.seeds:
        seeds = args.seeds
    elif args.n_repeats > 1:
        base = args.seed if args.seed is not None else 1234
        seeds = [base + i for i in range(args.n_repeats)]
    else:
        seeds = [args.seed]  # single run; None lets run_experiment.py use its own default seed
    multi_seed = len(seeds) > 1

    passthrough = []
    for flag, val in [("--iterations", args.iterations), ("--hidden-layers", args.hidden_layers),
                      ("--n-save-models", args.n_save_models),
                      ("--ae-epochs", args.ae_epochs), ("--grid-xnum", args.grid_xnum),
                      ("--causal-eps", args.causal_eps), ("--num-causal-buckets", args.num_causal_buckets)]:
        if val is not None:
            passthrough += [flag, str(val)]
    if args.quick:
        passthrough.append("--quick")

    cells = [(p, m, s) for p in args.pdes for m in args.methods for s in seeds]
    print(f"Running {len(cells)} cells: pdes={args.pdes} methods={args.methods} seeds={seeds}\n")

    def cell_key(pde, method, seed):
        return f"{pde}/{method}/seed_{seed}" if multi_seed else f"{pde}/{method}"

    for idx, (pde, method, seed) in enumerate(cells, 1):
        seed_out = os.path.join(args.out, f"seed_{seed}") if multi_seed else args.out
        key = cell_key(pde, method, seed)
        run_dir = os.path.join(seed_out, pde, method)
        metrics_path = os.path.join(run_dir, "metrics.json")

        if os.path.exists(metrics_path) and not args.force:
            with open(metrics_path) as f:
                m = json.load(f)
            manifest[key] = {"status": "skipped (exists)", "relative_l2": m.get("relative_l2"),
                             "wall_clock_sec": m.get("wall_clock_sec")}
            print(f"[{idx}/{len(cells)}] {key}: SKIP (metrics.json exists)")
            continue

        cell_args = list(passthrough)
        if seed is not None:
            cell_args += ["--seed", str(seed)]
        cmd = [sys.executable, RUN_EXPERIMENT, "--pde", pde, "--method", method, "--out", seed_out] + cell_args
        print(f"[{idx}/{len(cells)}] {key}: RUN  ({' '.join(cmd[2:])})")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        dt = round(time.time() - t0, 1)

        if proc.returncode == 0 and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            manifest[key] = {
                "status": "ok", "seed": seed, "relative_l2": m.get("relative_l2"),
                "wall_clock_sec": m.get("wall_clock_sec"),
                "condition_number": m.get("condition_number"),
                "fourier_low": m.get("fourier_low"), "fourier_mid": m.get("fourier_mid"),
                "fourier_high": m.get("fourier_high"),
            }
            print(f"        -> ok  relative_l2={m.get('relative_l2')}  ({dt}s)")
        else:
            manifest[key] = {"status": f"failed (code {proc.returncode})", "driver_sec": dt}
            print(f"        -> FAILED (return code {proc.returncode})")

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # summary table
    print("\n==================== SUMMARY ====================")
    print(f"{'cell':<48} {'status':<20} {'rel-L2':<12}")
    for (pde, method, seed) in cells:
        key = cell_key(pde, method, seed)
        info = manifest.get(key, {})
        rl2 = info.get("relative_l2")
        rl2s = f"{rl2:.3e}" if isinstance(rl2, (int, float)) else "-"
        print(f"{key:<48} {info.get('status', '-'):<20} {rl2s:<12}")
    print(f"\nManifest: {manifest_path}")
    print("Next: python experiments/landscape_compare/compare_landscapes.py --runs " + args.out)


if __name__ == "__main__":
    main()
