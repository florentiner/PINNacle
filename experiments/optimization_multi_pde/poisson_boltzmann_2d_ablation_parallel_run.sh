#!/bin/bash
# Абляция DQN-стека на poisson_boltzmann_2d: 4 режима, раскладка по GPU.
# Буфер тянется с HuggingFace, логи и результаты уезжают туда же. Comet не нужен.
#
# Использование:
#   export HF_TOKEN=<токен с правом записи в датасет результатов>
#   bash experiments/optimization_multi_pde/poisson_boltzmann_2d_ablation_parallel_run.sh
#
# Переменные окружения (все необязательные):
#   HF_RESULTS   — датасет для логов/результатов (по умолчанию danil-e/rlpinn-ablation-runs)
#   HF_BUFFER    — датасет с буфером (по умолчанию danil-e/rlpinn-ablation-buffers)
#   SEED         — сид запуска (по умолчанию 1234)
#   MODES        — список режимов через пробел (по умолчанию все четыре)

set -u

SCRIPT="experiments/optimization_multi_pde/poisson_boltzmann_2d_ablation_chain.py"
HF_RESULTS="${HF_RESULTS:-danil-e/rlpinn-ablation-runs}"
HF_BUFFER="${HF_BUFFER:-danil-e/rlpinn-ablation-buffers}"
SEED="${SEED:-1234}"
MODES="${MODES:-none no_per no_soft_watkins no_trust_region}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN не задан — логи и результаты не смогут уехать на HF."
    echo "export HF_TOKEN=<токен с правом записи в $HF_RESULTS>"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "Detected GPUs: $NUM_GPUS"
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

mkdir -p logs
i=0
for mode in $MODES; do
    gpu=$((i % NUM_GPUS))
    tag="$(date +%Y-%m-%d_%H-%M-%S)_$(hostname)_seed${SEED}"
    echo "Launching ablation=$mode on GPU $gpu (run_tag=$tag)"
    CUDA_VISIBLE_DEVICES=$gpu nohup python "$SCRIPT" \
        --ablation "$mode" \
        --seed "$SEED" \
        --buffer-src hf --hf-repo "$HF_BUFFER" \
        --hf-results "$HF_RESULTS" \
        --run-tag "$tag" \
        > "logs/ablation_${mode}_seed${SEED}.log" 2>&1 &
    i=$((i + 1))
done

echo
echo "Запущено процессов: $i. Локальные логи: logs/ablation_<режим>_seed${SEED}.log"
echo "Результаты: https://huggingface.co/datasets/$HF_RESULTS/tree/main/runs/poisson_boltzmann_2d"
wait
echo "All ablation processes finished."
