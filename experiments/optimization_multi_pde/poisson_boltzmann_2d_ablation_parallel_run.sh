#!/bin/bash
# Запуск абляции DQN-стека на poisson_boltzmann_2d: 4 режима с раскладкой по GPU.
# Использование: bash experiments/optimization_multi_pde/poisson_boltzmann_2d_ablation_parallel_run.sh

SCRIPT="experiments/optimization_multi_pde/poisson_boltzmann_2d_ablation_chain.py"
MODES=("none" "no_per" "no_soft_watkins" "no_trust_region")

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

i=0
for mode in "${MODES[@]}"; do
    gpu=$((i % NUM_GPUS))
    echo "Launching ablation=$mode on GPU $gpu..."
    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" --ablation "$mode" &
    i=$((i + 1))
done

wait
echo "All ablation processes finished."
