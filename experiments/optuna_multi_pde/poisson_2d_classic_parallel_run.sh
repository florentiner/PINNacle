#!/bin/bash
# Automatic launch of poisson_2d_classic_optuna.py with GPU/MPS/CPU distribution

SCRIPT="experiments/optuna_multi_pde/poisson_2d_classic_optuna.py"
DB_PATH="optuna_studies/poisson_2d_classic.db"
STUDY_NAME="poisson_2d_classic_chain"
N_TRIALS="${1:-500}"

mkdir -p optuna_studies

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

echo "Detected GPUs: $NUM_GPUS"
echo "Trials per process: $N_TRIALS"

if [ "$NUM_GPUS" -ge 2 ]; then
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials $N_TRIALS &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials $N_TRIALS &
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on a single GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials $N_TRIALS &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials $N_TRIALS &
else
    echo "No CUDA GPUs found. Using MPS (Apple) or CPU..."
    python "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials $N_TRIALS &
    python "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials $N_TRIALS &
fi

wait
echo "All processes finished."
