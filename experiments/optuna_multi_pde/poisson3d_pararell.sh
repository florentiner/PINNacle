#!/bin/bash
# Automatic launch of burgers_1d_optuna.py with GPU distribution via nvidia-smi
# Python: optuna_rl venv. Code/deepxde: local tree under $SCRIPT_DIR (see PYTHONPATH).
# Install PyTorch matching the host driver (e.g. cu121 wheels for CUDA 12.1 + driver 530+).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -f "$SCRIPT_DIR/optuna_rl/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/optuna_rl/bin/python"
else
    echo "ERROR: Expected venv at $SCRIPT_DIR/optuna_rl/bin/python" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

SCRIPT="experiments/optuna_multi_pde/poisson3d_complexgeometry_optuna.py"
DB_PATH="optuna_studies/poisson3d_complexgeometry.db"
# New name recommended if the DB was used with a different search space (e.g. CMA-ES multibranch or int epochs).
STUDY_NAME="poisson3d_complexgeometry_continues"
N_TRIALS="${1:-8}"  # per process; 8 processes => 8 * 8 = 64 total trials
N_PROCESSES="${2:-8}"  # Default to 8 processes, can be overridden

mkdir -p optuna_studies

# Detect NVIDIA GPUs using nvidia-smi (same tool the driver exposes)
NUM_GPUS=0
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -L >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
        NUM_GPUS=$(echo "$NUM_GPUS" | tr -d '[:space:]')
    fi
fi

echo "Parallel workers: $N_PROCESSES | trials per worker: $N_TRIALS"
if command -v nvidia-smi >/dev/null 2>&1 && [ "${NUM_GPUS:-0}" -gt 0 ] 2>/dev/null; then
    echo "nvidia-smi GPU summary:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo "Detected NVIDIA GPU count (nvidia-smi -L): $NUM_GPUS"
else
    echo "nvidia-smi not available or no NVIDIA GPUs detected (count=$NUM_GPUS)."
fi

if [ "${NUM_GPUS:-0}" -gt 0 ]; then
    if ! "$PYTHON" -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        echo "ERROR: nvidia-smi reports $NUM_GPUS GPU(s) but torch.cuda.is_available() is false." >&2
        echo "Reinstall a PyTorch build that matches this machine's CUDA driver, e.g.:" >&2
        echo "  $PYTHON -m pip install 'torch' 'torchvision' --index-url https://download.pytorch.org/whl/cu121" >&2
        exit 1
    fi
    echo "PyTorch sees CUDA: $($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || true)"
    echo "Launching $N_PROCESSES Optuna processes (round-robin across $NUM_GPUS GPU(s))..."
    for i in $(seq 0 $((N_PROCESSES - 1))); do
        gpu=$((i % NUM_GPUS))
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials "$N_TRIALS" &
    done
else
    echo "No CUDA GPUs via nvidia-smi. Falling back to CPU ($N_PROCESSES processes)..."
    for _ in $(seq 1 "$N_PROCESSES"); do
        $PYTHON "$SCRIPT" --db-path "$DB_PATH" --study-name "$STUDY_NAME" --n-trials "$N_TRIALS" &
    done
fi

wait
echo "All processes finished."
