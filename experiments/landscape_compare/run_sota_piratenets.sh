#!/usr/bin/env bash
# ============================================================================
# SOTA -- PirateNets recipe for the GPU machine.
#
# Implements the reference architecture BEST_PRACTICES.md pins as state-of-the-art for
# stiff/chaotic PINNs: Wang, Li & Perdikaris, "PirateNets: Physics-informed Deep Learning
# with Residual Adaptive Networks", arXiv:2402.00326 (JMLR 2025). That paper is the "proof"
# asked for -- it is the published result that beats prior PINN best practice on stiff PDEs
# (including the Kuramoto-Sivashinsky / Gray-Scott family) by combining, in ONE model:
#   * physics-informed adaptive residual blocks, alpha-gate initialized to 0
#     (the network starts as a shallow linear map and deepens as the alphas grow)   [arch]
#   * physics-informed least-squares init of the output layer to the IC              [method `sota`]
#   * random Fourier feature embedding (also gives exact periodicity for KS/GS)      [--fourier-modes]
#   * random weight factorization                                                    [--rwf]
#   * causal training + grad-norm loss weighting + Adam warmup/decay                 [method `sota`]
# NO time-marching (the paper trains the whole spatiotemporal domain at once). Frozen-PINN NOT used.
#
# The `sota` method (METHOD_SPEC in run_experiment.py) already encodes the EXACT recipe:
#   loss_type = "causal"   -> causal training with eps annealing is used (not plain MSE)
#   grad_norm_weights=True -> grad-norm loss balancing is used
#   arch = "piratenet"     -> PirateNet with alpha-gates + LSQ output init is used
#   time_windows           -> NOT set, so NO time-marching (matches the paper)
# This script does not override any of that; it only supplies scale (width/modes/iters/warmup).
#
# HEAD-TO-HEAD: `sota` vs the deepxde `origin` baseline, at the paper's scale (width 256,
# depth L=3 blocks, warmup 5000). For EACH PDE the 3 seeds run IN PARALLEL (all three at once),
# and PDEs run one after another: KS's 3 seeds together, then -- if RUN_GS=1 -- Gray-Scott's 3
# seeds together. Within a PDE the `origin` baseline runs after the `sota` arm (also 3-in-parallel).
#
# Knobs (env overrides):
#   ITER   total Adam iterations       (default 200000; the paper uses up to ~3e5)
#   MODES  Fourier modes (KS)          (default 32; KS 32-mode representation ceiling = 0.020)
#   SEEDS  the 3 seeds (run parallel)  (default "1234 1235 1236")
#   WIDTH  hidden width                (default 256)
#   RUN_GS 1 = also run Gray-Scott     (default 0)
#
# Outputs: runs_sota/<method>/seed_<S>/<pde>/<method>/metrics.json  -- resumable (done cells skip).
# Transfer back:  zip -r sota_results.zip runs_sota
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export DDEBACKEND=pytorch KMP_DUPLICATE_LIB_OK=TRUE

RE=experiments/landscape_compare/run_experiment.py
ITER="${ITER:-200000}"
MODES="${MODES:-32}"
SEEDS="${SEEDS:-1234 1235 1236}"
WIDTH="${WIDTH:-256}"
RUN_GS="${RUN_GS:-0}"

PDES="kuramoto_sivashinsky"
[ "$RUN_GS" = "1" ] && PDES="kuramoto_sivashinsky grayscott"

# GS is 2D+t: a Fourier product embedding over (x,y) blows up the input dim, so cap its modes.
gs_modes() { [ "$1" = "grayscott" ] && echo 8 || echo "$MODES"; }

# Launch the 3 seeds of one (pde, method) arm together and block until all three finish.
# `wait` returns non-zero if any child failed; `set -e` then aborts the script (fail-fast).
run_arm_parallel() {  # $1=pde $2=method  (remaining args passed through to run_experiment)
  local pde="$1" method="$2"; shift 2
  local pids=() S OUT rc=0
  for S in $SEEDS; do
    OUT="runs_sota/${method}/seed_${S}"
    if [ -f "${OUT}/${pde}/${method}/metrics.json" ]; then
      echo "skip ${pde}/${method}/seed_${S} (done)"; continue
    fi
    echo "launch ${pde}/${method}/seed_${S}"
    python "$RE" --pde "$pde" --method "$method" \
      --iterations "$ITER" --no-landscape --seed "$S" --out "$OUT" "$@" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

for PDE in $PDES; do
  M=$(gs_modes "$PDE")
  echo "================ ${PDE}: SOTA arm (3 seeds in parallel) ================"
  # sota arm: PirateNet + Fourier + RWF + warmup. Loss/grad-norm/LSQ come from the method spec.
  run_arm_parallel "$PDE" sota \
    --hidden-layers "${WIDTH}*4" --fourier-modes "$M" --rwf --warmup 5000

  echo "================ ${PDE}: origin baseline (3 seeds in parallel) ================"
  # origin must stay the shipped deepxde baseline it has to beat: plain FNN, default modes,
  # origin (plain-MSE) loss, no RWF. Only matched in width + warmup for a fair-scale comparison.
  run_arm_parallel "$PDE" origin \
    --hidden-layers "${WIDTH}*4" --warmup 5000
done

echo "================ done ================"
echo "Transfer back:  zip -r sota_results.zip runs_sota"
echo "Compare:        grep -rh rel_l2 runs_sota/*/*/*/*/metrics.json"
