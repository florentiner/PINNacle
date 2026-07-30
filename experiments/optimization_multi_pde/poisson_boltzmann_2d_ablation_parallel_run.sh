#!/bin/bash
# Абляция DQN-стека на poisson_boltzmann_2d: все 4 режима параллельно.
# Буфер тянется с HuggingFace, логи/результаты/модель уезжают в отдельный
# HF-датасет. Comet не нужен.
#
# Запуск одной командой:
#   export HF_TOKEN=<токен с правом записи> && bash experiments/optimization_multi_pde/poisson_boltzmann_2d_ablation_parallel_run.sh
#
# Сеть агента маленькая, поэтому все четыре режима спокойно живут на одной GPU;
# при нескольких GPU процессы раскладываются по ним по очереди.
#
# Переменные окружения (все необязательные):
#   HF_RESULTS — датасет для логов/результатов (по умолчанию danil-e/rlpinn-ablation-runs)
#   HF_BUFFER  — датасет с буфером (по умолчанию danil-e/rlpinn-ablation-buffers)
#   SEED       — сид запуска (по умолчанию 1234)
#   MODES      — режимы через пробел (по умолчанию все четыре)
#   PER_GPU    — сколько процессов на одну GPU (по умолчанию все на первую, если GPU одна)

set -u

SCRIPT="experiments/optimization_multi_pde/poisson_boltzmann_2d_ablation_chain.py"
HF_RESULTS="${HF_RESULTS:-danil-e/rlpinn-ablation-runs}"
HF_BUFFER="${HF_BUFFER:-danil-e/rlpinn-ablation-buffers}"
SEED="${SEED:-1234}"
MODES="${MODES:-none no_per no_soft_watkins no_trust_region}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ВНИМАНИЕ: HF_TOKEN не задан — результаты останутся только локально в runs_single/."
    echo "Для выгрузки на HF: export HF_TOKEN=<токен с правом записи в $HF_RESULTS>"
    echo
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
[ -z "$NUM_GPUS" ] && NUM_GPUS=0
echo "Detected GPUs: $NUM_GPUS"
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "CUDA-устройств не найдено — запускаемся на CPU (будет медленно)."
fi

# Буфер скачиваем один раз до старта: иначе четыре процесса полезут в один
# и тот же кеш HF одновременно.
if [ -n "${HF_BUFFER}" ]; then
    echo "Прогреваем кеш буфера ($HF_BUFFER)..."
    python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$HF_BUFFER', repo_type='dataset', allow_patterns=['poisson_boltzmann_2d/*'])
print('буфер готов:', p)
" || { echo "Не удалось скачать буфер"; exit 1; }
fi

mkdir -p logs
STAMP="$(date +%Y-%m-%d_%H-%M-%S)_$(hostname)_seed${SEED}"
i=0
for mode in $MODES; do
    if [ "$NUM_GPUS" -gt 0 ]; then
        gpu=$((i % NUM_GPUS))
        export CUDA_VISIBLE_DEVICES=$gpu
        echo "Launching ablation=$mode on GPU $gpu"
    else
        echo "Launching ablation=$mode on CPU"
    fi
    nohup python "$SCRIPT" \
        --ablation "$mode" \
        --seed "$SEED" \
        --buffer-src hf --hf-repo "$HF_BUFFER" \
        --hf-results "$HF_RESULTS" \
        --run-tag "$STAMP" \
        > "logs/ablation_${mode}_seed${SEED}.log" 2>&1 &
    i=$((i + 1))
done

echo
echo "Запущено процессов: $i (run_tag=$STAMP)"
echo "Локальные логи:  tail -f logs/ablation_*.log"
echo "Результаты на HF: https://huggingface.co/datasets/$HF_RESULTS/tree/main/runs/poisson_boltzmann_2d"
echo "Метрики по траекториям: results/trajectory_metrics.csv внутри каждого запуска"
wait
echo "All ablation processes finished."
