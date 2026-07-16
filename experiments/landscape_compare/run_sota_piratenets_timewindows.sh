#!/usr/bin/env bash
# ============================================================================
# SOTA + TIME-MARCHING -- the full PirateNets recipe, PLUS time-windows and Fourier features.
#
# Same reference architecture as run_sota_piratenets.sh (PirateNets, Wang, Li & Perdikaris,
# arXiv:2402.00326, JMLR 2025): everything from the paper --
#   * PirateNet arch: alpha-gated adaptive residual blocks initialized to 0            [arch]
#   * physics-informed least-squares output init to the IC                            [method `sota`]
#   * random Fourier feature embedding (also exact periodicity for KS/GS)             [--fourier-modes]
#   * random weight factorization                                                     [--rwf]
#   * causal training + grad-norm loss weighting + Adam warmup/decay                  [method `sota`]
# ...and ADDITIONALLY layers time-marching on top:
#   * the spatiotemporal domain is split into WINDOWS sequential time windows; each window
#     is trained in turn and warm-starts the next (the causal paper's chaotic-KS setting,
#     arXiv:2203.07404). The base `sota` recipe does NOT march (the PirateNets paper trains
#     the whole domain at once); this script adds it via --time-windows to test whether
#     marching helps the SOTA model on the chaotic horizon.
#
# The LSQ output init fires on the FIRST window (w=0) from the global IC; later windows warm-
# start from the previous window's solution, and grad-norm re-adapts per window (as designed).
# Fourier features are ON in both scripts -- this one just adds the marching.
#
# For EACH PDE the 3 seeds run IN PARALLEL (KS's 3 together, then Gray-Scott's 3 together);
# PDEs run one after another. Within a PDE the `origin` baseline runs after the `sota` arm.
# origin here also gets time-marching + matched width/warmup, so the ONLY difference between
# the two arms is the PirateNets stack -- a fair same-setting head-to-head.
#
# Knobs (env overrides):
#   ITER    total Adam iterations (across ALL windows)  (default 200000)
#   MODES   Fourier modes (KS)                          (default 32)
#   WINDOWS time-marching windows                       (default 10; causal paper's KS setting)
#   SEEDS   the 3 seeds (run parallel)                  (default "1234 1235 1236")
#   WIDTH   hidden width                                (default 256)
#   RUN_GS  1 = also run Gray-Scott                     (default 0)
#
# Outputs: runs_sota_tw/<method>/seed_<S>/<pde>/<method>/metrics.json -- resumable.
# Transfer back:  zip -r sota_tw_results.zip runs_sota_tw
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export DDEBACKEND=pytorch KMP_DUPLICATE_LIB_OK=TRUE

RE=experiments/landscape_compare/run_experiment.py
ITER="${ITER:-200000}"
MODES="${MODES:-32}"
WINDOWS="${WINDOWS:-10}"
SEEDS="${SEEDS:-1234 1235 1236}"
WIDTH="${WIDTH:-256}"
RUN_GS="${RUN_GS:-0}"

PDES="kuramoto_sivashinsky"
[ "$RUN_GS" = "1" ] && PDES="kuramoto_sivashinsky grayscott"

# GS is 2D+t: a Fourier product embedding over (x,y) blows up the input dim, so cap its modes.
gs_modes() { [ "$1" = "grayscott" ] && echo 8 || echo "$MODES"; }

# Launch the 3 seeds of one (pde, method) arm together and block until all three finish.
run_arm_parallel() {  # $1=pde $2=method  (remaining args passed through to run_experiment)
  local pde="$1" method="$2"; shift 2
  local pids=() S OUT rc=0
  for S in $SEEDS; do
    OUT="runs_sota_tw/${method}/seed_${S}"
    if [ -f "${OUT}/${pde}/${method}/metrics.json" ]; then
      echo "skip ${pde}/${method}/seed_${S} (done)"; continue
    fi
    echo "launch ${pde}/${method}/seed_${S}  (windows=${WINDOWS})"
    python "$RE" --pde "$pde" --method "$method" --time-windows "$WINDOWS" \
      --iterations "$ITER" --no-landscape --seed "$S" --out "$OUT" "$@" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

for PDE in $PDES; do
  M=$(gs_modes "$PDE")
  echo "================ ${PDE}: SOTA + time-marching (3 seeds in parallel) ================"
  # full PirateNets stack + Fourier + RWF + warmup + time-marching. Loss/grad-norm/LSQ come
  # from the `sota` method spec; --time-windows adds the marching on top.
  run_arm_parallel "$PDE" sota \
    --hidden-layers "${WIDTH}*4" --fourier-modes "$M" --rwf --warmup 5000

  echo "================ ${PDE}: origin baseline + time-marching (3 seeds in parallel) ================"
  # same time-marching + width + warmup, plain FNN / default modes / origin loss: isolates the
  # PirateNets stack as the only difference from the sota arm.
  run_arm_parallel "$PDE" origin \
    --hidden-layers "${WIDTH}*4" --warmup 5000
done

echo "================ done ================"
echo "Transfer back:  zip -r sota_tw_results.zip runs_sota_tw"
echo "Compare:        grep -rh rel_l2 runs_sota_tw/*/*/*/*/metrics.json"
