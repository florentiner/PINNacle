#!/bin/bash
# Абляция DQN-стека на poisson_boltzmann_2d: все 4 режима параллельно.
# Буфер тянется с HuggingFace, логи/результаты/модель уезжают в отдельный
# HF-датасет. Comet не нужен.
#
# Запуск одной командой:
#   export HF_TOKEN=<токен с правом записи> && bash experiments/optimization_multi_pde/poisson_boltzmann_2d_ablation_parallel_run.sh
#
# Одна траектория идёт 1–2 часа, поэтому у запуска есть бюджет времени
# (MAX_HOURS): по его исчерпании процессы сами сохраняют модель агента,
# выгружают результаты и выходят штатно.
#
# Переменные окружения (все необязательные):
#   MAX_HOURS  — бюджет времени на запуск (по умолчанию 20)
#   HF_RESULTS — датасет для логов/результатов (по умолчанию danil-e/rlpinn-ablation-runs)
#   HF_BUFFER  — датасет с буфером (по умолчанию danil-e/rlpinn-ablation-buffers)
#   SEED       — сид запуска (по умолчанию 1234)
#   MODES      — режимы через пробел (по умолчанию все четыре)
#   PYTHON     — интерпретатор (по умолчанию python3, затем python)

set -u

SCRIPT="experiments/optimization_multi_pde/poisson_boltzmann_2d_ablation_chain.py"
HF_RESULTS="${HF_RESULTS:-danil-e/rlpinn-ablation-runs}"
HF_BUFFER="${HF_BUFFER:-danil-e/rlpinn-ablation-buffers}"
SEED="${SEED:-1234}"
MAX_HOURS="${MAX_HOURS:-20}"
MODES="${MODES:-none no_per no_soft_watkins no_trust_region}"

# python3 — основной интерпретатор; на многих серверах голого `python` нет.
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif command -v python >/dev/null 2>&1; then
    PY="python"
else
    echo "Не найден python3/python. Задайте PYTHON=<путь к интерпретатору>."
    exit 1
fi
echo "Интерпретатор: $PY ($($PY --version 2>&1))"

if [ ! -f "$SCRIPT" ]; then
    echo "Не найден $SCRIPT — запускайте из корня репозитория PINNacle."
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ВНИМАНИЕ: HF_TOKEN не задан — результаты останутся только локально в runs_single/."
    echo "Для выгрузки на HF: export HF_TOKEN=<токен с правом записи в $HF_RESULTS>"
    echo
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
[ -z "$NUM_GPUS" ] && NUM_GPUS=0
echo "Detected GPUs: $NUM_GPUS"
[ "$NUM_GPUS" -eq 0 ] && echo "CUDA-устройств не найдено — запускаемся на CPU (будет медленно)."

# Буфер скачиваем ОДИН раз до старта: иначе четыре процесса лезут в один и тот
# же кеш HF одновременно.
echo "Прогреваем кеш буфера ($HF_BUFFER)..."
$PY -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$HF_BUFFER', repo_type='dataset', allow_patterns=['poisson_boltzmann_2d/*'])
print('буфер готов:', p)
" || { echo "Не удалось скачать буфер"; exit 1; }

mkdir -p logs
STAMP="$(date +%Y-%m-%d_%H-%M-%S)_$(hostname)_seed${SEED}"
declare -a PIDS=()
declare -a NAMES=()

i=0
for mode in $MODES; do
    if [ "$NUM_GPUS" -gt 0 ]; then
        gpu=$((i % NUM_GPUS))
        export CUDA_VISIBLE_DEVICES=$gpu
        echo "Launching ablation=$mode on GPU $gpu"
    else
        echo "Launching ablation=$mode on CPU"
    fi
    # setsid + nohup: процесс уходит в свою сессию и переживает закрытие SSH.
    setsid nohup "$PY" -u "$SCRIPT" \
        --ablation "$mode" \
        --seed "$SEED" \
        --max-hours "$MAX_HOURS" \
        --buffer-src hf --hf-repo "$HF_BUFFER" \
        --hf-results "$HF_RESULTS" \
        --run-tag "$STAMP" \
        > "logs/ablation_${mode}_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
    NAMES+=("$mode")
    i=$((i + 1))
done

echo
echo "Запущено процессов: $i (run_tag=$STAMP, бюджет ${MAX_HOURS} ч)"
for idx in "${!PIDS[@]}"; do
    echo "  ${NAMES[$idx]}: PID ${PIDS[$idx]}"
done
echo "Локальные логи:   tail -f logs/ablation_*.log"
echo "Результаты на HF: https://huggingface.co/datasets/$HF_RESULTS/tree/main/runs/poisson_boltzmann_2d"
echo
echo "Процессы отвязаны от сессии (setsid) — SSH можно закрывать."
echo "Остановить досрочно и сохранить результаты: kill ${PIDS[*]}"
echo

# Ждём каждый процесс отдельно, чтобы знать КТО и КАК завершился.
FAILED=0
for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    mode="${NAMES[$idx]}"
    if wait "$pid"; then
        echo "✅ $mode (PID $pid): завершился штатно"
    else
        code=$?
        if [ "$code" -gt 128 ]; then
            sig=$((code - 128))
            signame=$(kill -l "$sig" 2>/dev/null || echo "signal $sig")
            echo "❌ $mode (PID $pid): убит сигналом $signame (код $code)"
            echo "   Внешнее убийство: смотрите OOM (dmesg -T | grep -i 'killed process'),"
            echo "   лимиты планировщика или остановку контейнера."
        else
            echo "❌ $mode (PID $pid): вышел с кодом $code — смотрите logs/ablation_${mode}_seed${SEED}.log"
        fi
        FAILED=$((FAILED + 1))
    fi
done

echo
if [ "$FAILED" -eq 0 ]; then
    echo "Все процессы завершились штатно."
else
    echo "Завершено с ошибками: $FAILED из ${#PIDS[@]}. Причина каждого — в results/status.json"
    echo "внутри соответствующего запуска (локально в runs_single/ и на HF)."
fi
