#!/bin/bash
# Automatic launch of heatinv_chain.py with GPU distribution

SCRIPT="experiments/optimization_multi_pde/heatinv_chain.py"
log_enable="true"

# Detect available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

echo "Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on a single GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$log_enable" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$log_enable" &
else
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$log_enable" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key "$log_enable" &
fi

wait
echo "All processes finished."
