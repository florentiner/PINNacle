#!/bin/bash
# Sequential 10-seed eval with best hyperparams from latest "continu*" Optuna studies.
# Kuramoto: continuous chain params. Poisson: discrete index grid (continues study name).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$SCRIPT_DIR/optuna_rl/bin/python"
LOG_DIR="$SCRIPT_DIR/gpu_logs/continuous_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "Logs: $LOG_DIR"
echo "Starting kuramoto_sivashinsky (parallel=1, 10 seeds)..."

"$PYTHON" experiments/optuna_multi_pde/continuous_best_eval_parallel.py \
    --pde kuramoto_sivashinsky \
    --parallel 1 \
    --n-eval-runs 10 \
    2>&1 | tee "$LOG_DIR/kuramoto_sivashinsky_continuous_eval.log"

echo "Starting poisson3d_complexgeometry (parallel=1, 10 seeds)..."

"$PYTHON" experiments/optuna_multi_pde/poisson3d_complexgeometry_best_eval_parallel.py \
    --study-name poisson3d_complexgeometry_continues \
    --parallel 1 \
    --n-eval-runs 10 \
    --results-csv poisson3d_complexgeometry_continuous.csv \
    --name poisson3d_complexgeometry_continuous_best_eval \
    2>&1 | tee "$LOG_DIR/poisson3d_complexgeometry_continuous_eval.log"

echo "Done. CSV outputs:"
echo "  $SCRIPT_DIR/kuramoto_sivashinsky_continuous.csv"
echo "  $SCRIPT_DIR/poisson3d_complexgeometry_continuous.csv"
