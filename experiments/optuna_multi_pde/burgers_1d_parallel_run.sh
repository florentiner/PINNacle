#!/bin/bash
# Automatic launch of burgers_1d_optuna.py with GPU/MPS/CPU distribution

# Use venv python if available, otherwise system python
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
else
    PYTHON="python"
fi

SCRIPT="experiments/optuna_multi_pde/burgers_1d_optuna.py"
DB_PATH="optuna_studies/burgers_1d.db"
STUDY_NAME="burgers_1d_chain"
N_TRIALS="${1:-500}"  # per process; 2 processes => 1000 total

mkdir -p optuna_studies

# Detect available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

echo "Detected GPUs: $NUM_GPUS"
echo "Trials per process: $N_TRIALS"

if [ "$NUM_GPUS" -ge 2 ]; then
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials $N_TRIALS &
    CUDA_VISIBLE_DEVICES=1 $PYTHON "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials $N_TRIALS &
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on a single GPU..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials $N_TRIALS &
    CUDA_VISIBLE_DEVICES=0 $PYTHON "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials $N_TRIALS &
else
    echo "No CUDA GPUs found. Using MPS (Apple) or CPU..."
    $PYTHON "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials $N_TRIALS &
    $PYTHON "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials $N_TRIALS &
fi

wait
echo "All processes finished."
