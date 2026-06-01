#!/bin/bash
# Parallel Optuna workers on Apple Silicon with PyTorch MPS.
# All workers share the single Metal GPU (unified memory); do not treat this like multi-GPU CUDA.
#
# Usage:
#   ./poisson3d_complexgeometry_parallel_run_mps.sh [N_TRIALS_PER_WORKER] [N_WORKERS]
#   N_TRIALS_PER_WORKER defaults to 500. N_WORKERS defaults to auto (see below).
#   Override worker count: MPS_N_WORKERS=4 ./poisson3d_complexgeometry_parallel_run_mps.sh
#
# How many processes max?
#   There is no fixed "max" — limited by unified RAM and MPS contention. A practical cap is
#   min(4, max(2, hw.physicalcpu / 2)): e.g. 8 cores -> 4 workers, 4 cores -> 2 workers.
#   If you see OOM or GPU timeouts, reduce N_WORKERS to 2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
else
    PYTHON="python3"
fi

SCRIPT="experiments/optuna_multi_pde/poisson3d_complexgeometry_optuna.py"
DB_PATH="optuna_studies/poisson3d_complexgeometry.db"
STUDY_NAME="poisson3d_complexgeometry_chain"
N_TRIALS="${1:-500}"

# Default parallel worker count for MPS (single GPU): physical CPUs / 2, clamped [2, 4]
if [ -n "${MPS_N_WORKERS:-}" ]; then
    N_WORKERS="$MPS_N_WORKERS"
elif [ -n "${2:-}" ]; then
    N_WORKERS="$2"
else
    PHYS="$(sysctl -n hw.physicalcpu 2>/dev/null || echo 8)"
    N_WORKERS=$(( PHYS / 2 ))
    [ "$N_WORKERS" -lt 2 ] && N_WORKERS=2
    [ "$N_WORKERS" -gt 4 ] && N_WORKERS=4
fi

mkdir -p optuna_studies

if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: This script is intended for Apple Silicon (Darwin/arm64). Continuing anyway."
fi

if ! "$PYTHON" -c "import torch; assert torch.backends.mps.is_available(), 'MPS not available'" 2>/dev/null; then
    echo "Error: PyTorch MPS is not available. Use a Mac with Apple GPU and a PyTorch build with MPS."
    exit 1
fi

# Optional: reduce transient OOM on long runs (PyTorch may grow MPS pool aggressively otherwise)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

echo "PyTorch MPS: OK"
echo "Trials per worker: $N_TRIALS"
echo "Parallel workers: $N_WORKERS (single shared MPS device)"
echo "Total trials (approx): $(( N_TRIALS * N_WORKERS ))"

for ((i = 1; i <= N_WORKERS; i++)); do
    echo "Starting worker $i/$N_WORKERS..."
    "$PYTHON" "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials "$N_TRIALS" &
    # Stagger launches to avoid SQLite alembic_version race condition on DB init
    sleep 5
done

wait
echo "All workers finished."
