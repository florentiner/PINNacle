#!/bin/bash
# Sequential Optuna run: ns2d_longtime → kuramoto_sivashinsky → heatinv
# Each PDE: 8 parallel workers, --n-trials per worker defaulting to 8 (≈64 total per PDE).
#
# Usage:
#   ./run_ns2d_kuramoto_heatinv_sequential.sh [N_TRIALS_PER_WORKER] [N_WORKERS]
#   Defaults: N_TRIALS_PER_WORKER=8, N_WORKERS=8
#   Override workers: MPS_N_WORKERS=4 ./run_ns2d_kuramoto_heatinv_sequential.sh
#
# Device selection:
#   - CUDA: uses available GPUs (one or two, else CPU/MPS fallback)
#   - MPS:  auto-detected on Apple Silicon; set PYTORCH_MPS_HIGH_WATERMARK_RATIO to 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
else
    PYTHON="python3"
fi

# ── Parameters ────────────────────────────────────────────────────────────────
N_TRIALS="${1:-8}"   # trials per worker; 8 workers × 8 = 64 ≈ 60 total per PDE

if [ -n "${MPS_N_WORKERS:-}" ]; then
    N_WORKERS="$MPS_N_WORKERS"
elif [ -n "${2:-}" ]; then
    N_WORKERS="$2"
else
    N_WORKERS=8
fi

mkdir -p optuna_studies

# ── Device detection ──────────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)

USE_CUDA=false
USE_MPS=false

if [ "$NUM_GPUS" -ge 1 ] 2>/dev/null; then
    USE_CUDA=true
elif "$PYTHON" -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    USE_MPS=true
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"
fi

echo "================================================================"
echo "Sequential Optuna run: ns2d_longtime → kuramoto → heatinv"
echo "Workers per PDE : $N_WORKERS"
echo "Trials per worker: $N_TRIALS  (~$(( N_TRIALS * N_WORKERS )) total per PDE)"
if $USE_CUDA; then
    echo "Device          : CUDA ($NUM_GPUS GPU(s))"
elif $USE_MPS; then
    echo "Device          : MPS (Apple Silicon)"
else
    echo "Device          : CPU"
fi
echo "================================================================"

# ── Helper: launch N_WORKERS parallel Optuna workers for a given script/DB ───
run_pde() {
    local SCRIPT="$1"
    local DB_PATH="$2"
    local STUDY_NAME="$3"
    local PDE_LABEL="$4"

    echo ""
    echo "----------------------------------------------------------------"
    echo "Starting PDE: $PDE_LABEL"
    echo "Script : $SCRIPT"
    echo "DB     : $DB_PATH"
    echo "Study  : $STUDY_NAME"
    echo "----------------------------------------------------------------"

    local GPU_IDX=0
    for ((i = 1; i <= N_WORKERS; i++)); do
        echo "  Launching worker $i/$N_WORKERS..."
        if $USE_CUDA; then
            # Distribute workers round-robin across available GPUs
            GPU_IDX=$(( (i - 1) % NUM_GPUS ))
            CUDA_VISIBLE_DEVICES=$GPU_IDX "$PYTHON" "$SCRIPT" \
                --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials "$N_TRIALS" &
        else
            # MPS or CPU: all workers share the same device
            "$PYTHON" "$SCRIPT" \
                --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials "$N_TRIALS" &
        fi
        # Stagger to avoid SQLite alembic_version race condition on first-time DB init
        sleep 5
    done

    echo "  Waiting for all $N_WORKERS workers to finish ($PDE_LABEL)..."
    wait
    echo "  Done: $PDE_LABEL"
}

# ── Run each PDE sequentially ─────────────────────────────────────────────────

run_pde \
    "experiments/optuna_multi_pde/ns2d_longtime_optuna.py" \
    "optuna_studies/ns2d_longtime.db" \
    "ns2d_longtime_chain" \
    "NS2D LongTime"

run_pde \
    "experiments/optuna_multi_pde/kuramoto_sivashinsky_optuna.py" \
    "optuna_studies/kuramoto_sivashinsky.db" \
    "kuramoto_sivashinsky_chain" \
    "Kuramoto-Sivashinsky"

run_pde \
    "experiments/optuna_multi_pde/heatinv_optuna.py" \
    "optuna_studies/heatinv.db" \
    "heatinv_chain" \
    "HeatInv"

echo ""
echo "================================================================"
echo "All three PDEs finished."
echo "================================================================"
