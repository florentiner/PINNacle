#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -f "$SCRIPT_DIR/optuna_rl/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/optuna_rl/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

SCRIPT="experiments/optuna_multi_pde/burgers_2d_optuna.py"
DB_PATH="optuna_studies/burgers_2d.db"
STUDY_NAME="burgers_2d_chain"
N_TRIALS="${1:-500}"
N_PROCESSES="${2:-12}"
TIMEOUT_HOURS="${3:-48}"
VALUE_TYPE="${4:-fixed}"

mkdir -p optuna_studies

NUM_GPUS=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')
fi

echo "PDE: burgers_2d | Workers: $N_PROCESSES | value-type: $VALUE_TYPE | trials/worker: $N_TRIALS | timeout: ${TIMEOUT_HOURS}h"

if [ "${NUM_GPUS:-0}" -gt 0 ]; then
    echo "Launching $N_PROCESSES workers across $NUM_GPUS GPU(s)..."
    for i in $(seq 0 $((N_PROCESSES - 1))); do
        gpu=$((i % NUM_GPUS))
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$SCRIPT" \
            --db-path "$DB_PATH" --study-name "$STUDY_NAME" \
            --sampler tpe --n-trials "$N_TRIALS" \
            --timeout-hours "$TIMEOUT_HOURS" --value-type "$VALUE_TYPE" &
    done
else
    echo "No CUDA GPUs. Launching $N_PROCESSES CPU workers..."
    for _ in $(seq 1 "$N_PROCESSES"); do
        $PYTHON "$SCRIPT" \
            --db-path "$DB_PATH" --study-name "$STUDY_NAME" \
            --sampler tpe --n-trials "$N_TRIALS" \
            --timeout-hours "$TIMEOUT_HOURS" --value-type "$VALUE_TYPE" &
    done
fi

wait
echo "All processes finished."
