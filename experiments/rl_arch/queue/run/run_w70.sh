#!/usr/bin/env bash
# Сгенерировано experiments/rl_arch/queue/export_wave.py — шаги задач очереди, как в кернеле Kaggle.
set -uo pipefail
cd "$(dirname "$0")"/. 2>/dev/null || true
cd "$(git rev-parse --show-toplevel)"
export DDEBACKEND=pytorch
if [ -z "${HF_TOKEN_WRITE:-}${HF_TOKEN:-}" ]; then echo "нужен HF_TOKEN_WRITE (результаты идут в HF)"; exit 2; fi
PY="${PYTHON:-python}"

job_pb2d_w70_orig_g_b() {  # агент статьи только с откатом, сиды 47,48,49,50,51
  echo "=== pb2d-w70-orig-g-b (~4.05 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/agents_online/their_dqn_pb2d_verify_seed42.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --keep-opt --keep-opt-mode safe --pde poissonboltzmann2d --budget 7000 --hours 2.0 --resume --ckpt-every 10 --seeds 47,48,49,50,51 --tag t3m_orig_g || return 1
}
job_pb2d_w70_orig_ss_a() {  # агент статьи с откатом и мягкой обрезкой (как итоговый метод), сиды 42,43,44,45,46
  echo "=== pb2d-w70-orig-ss-a (~4.05 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/agents_online/their_dqn_pb2d_verify_seed42.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde poissonboltzmann2d --budget 7000 --hours 2.0 --resume --ckpt-every 10 --seeds 42,43,44,45,46 --tag t3m_orig_ss || return 1
}
job_pb2d_w70_orig_ss_b() {  # агент статьи с откатом и мягкой обрезкой (как итоговый метод), сиды 47,48,49,50,51
  echo "=== pb2d-w70-orig-ss-b (~4.05 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/agents_online/their_dqn_pb2d_verify_seed42.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde poissonboltzmann2d --budget 7000 --hours 2.0 --resume --ckpt-every 10 --seeds 47,48,49,50,51 --tag t3m_orig_ss || return 1
}
job_pb2d_w70_our_m1_a() {  # итоговый метод, обучение 1, мягкая обрезка, сиды 47,48,49,50,51
  echo "=== pb2d-w70-our-m1-a (~4.05 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/models/convnext_dqn_cfix_rdelta_vb_w67proc3n20two_seed1.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde poissonboltzmann2d --budget 7000 --hours 2.0 --resume --ckpt-every 10 --seeds 47,48,49,50,51 --tag t3m_w67proc3n20two_m1_ss || return 1
}
job_pb2d_w70_our_m2_a() {  # итоговый метод, обучение 2, мягкая обрезка, сиды 47,48,49,50,51
  echo "=== pb2d-w70-our-m2-a (~4.05 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/models/convnext_dqn_cfix_rdelta_vb_w67proc3n20two_seed2.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde poissonboltzmann2d --budget 7000 --hours 2.0 --resume --ckpt-every 10 --seeds 47,48,49,50,51 --tag t3m_w67proc3n20two_m2_ss || return 1
}
job_w70_orig_g_a() {  # агент статьи только с откатом, сиды 42,43,44
  echo "=== w70-orig-g-a (~4.9 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/agents_online/their_dqn_ns2d_full_seed42.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 42,43,44 --tag t3m_orig_g || return 1
}
job_w70_orig_g_b() {  # агент статьи только с откатом, сиды 45,46,47
  echo "=== w70-orig-g-b (~4.9 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/agents_online/their_dqn_ns2d_full_seed42.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 45,46,47 --tag t3m_orig_g || return 1
}
job_w70_orig_g_c() {  # агент статьи только с откатом, сиды 48,49
  echo "=== w70-orig-g-c (~3.4 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/agents_online/their_dqn_ns2d_full_seed42.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 48,49 --tag t3m_orig_g || return 1
}
job_w70_orig_g_d() {  # агент статьи только с откатом, сиды 50,51
  echo "=== w70-orig-g-d (~3.4 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/agents_online/their_dqn_ns2d_full_seed42.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 50,51 --tag t3m_orig_g || return 1
}
job_w70_orig_ss_a() {  # агент статьи с откатом и мягкой обрезкой (как итоговый метод), сиды 42,43,44
  echo "=== w70-orig-ss-a (~4.9 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/agents_online/their_dqn_ns2d_full_seed42.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 42,43,44 --tag t3m_orig_ss || return 1
}
job_w70_orig_ss_b() {  # агент статьи с откатом и мягкой обрезкой (как итоговый метод), сиды 45,46,47
  echo "=== w70-orig-ss-b (~4.9 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/agents_online/their_dqn_ns2d_full_seed42.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 45,46,47 --tag t3m_orig_ss || return 1
}
job_w70_orig_ss_c() {  # агент статьи с откатом и мягкой обрезкой (как итоговый метод), сиды 48,49
  echo "=== w70-orig-ss-c (~3.4 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/agents_online/their_dqn_ns2d_full_seed42.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 48,49 --tag t3m_orig_ss || return 1
}
job_w70_orig_ss_d() {  # агент статьи с откатом и мягкой обрезкой (как итоговый метод), сиды 50,51
  echo "=== w70-orig-ss-d (~3.4 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/agents_online/their_dqn_ns2d_full_seed42.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 50,51 --tag t3m_orig_ss || return 1
}
job_w70_our_m1_a() {  # итоговый метод, обучение 1, мягкая обрезка, сиды 47,48,49
  echo "=== w70-our-m1-a (~4.9 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/models/convnext_dqn_cfix_rdelta_vb_ns2d_w65src3n20two_seed1.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 47,48,49 --tag t3m_w65src3n20two_m1_ss || return 1
}
job_w70_our_m1_b() {  # итоговый метод, обучение 1, мягкая обрезка, сиды 50,51
  echo "=== w70-our-m1-b (~3.4 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/models/convnext_dqn_cfix_rdelta_vb_ns2d_w65src3n20two_seed1.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 50,51 --tag t3m_w65src3n20two_m1_ss || return 1
}
job_w70_our_m2_a() {  # итоговый метод, обучение 2, мягкая обрезка, сиды 47,48,49
  echo "=== w70-our-m2-a (~4.9 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/models/convnext_dqn_cfix_rdelta_vb_ns2d_w65src3n20two_seed2.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 47,48,49 --tag t3m_w65src3n20two_m2_ss || return 1
}
job_w70_our_m2_b() {  # итоговый метод, обучение 2, мягкая обрезка, сиды 50,51
  echo "=== w70-our-m2-b (~3.4 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/models/convnext_dqn_cfix_rdelta_vb_ns2d_w65src3n20two_seed2.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 50,51 --tag t3m_w65src3n20two_m2_ss || return 1
}
job_w70_our_m2_c() {  # итоговый метод, обучение 2, мягкая обрезка, сиды 43,44
  echo "=== w70-our-m2-c (~3.4 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/models/convnext_dqn_cfix_rdelta_vb_ns2d_w65src3n20two_seed2.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 43,44 --tag t3m_w65src3n20two_m2_ss || return 1
}

WANT="${1:-}"
fail=0
if [ -z "$WANT" ] || [[ ",$WANT," == *",pb2d-w70-orig-g-b,"* ]]; then job_pb2d_w70_orig_g_b || { echo "ОШИБКА pb2d-w70-orig-g-b"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",pb2d-w70-orig-ss-a,"* ]]; then job_pb2d_w70_orig_ss_a || { echo "ОШИБКА pb2d-w70-orig-ss-a"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",pb2d-w70-orig-ss-b,"* ]]; then job_pb2d_w70_orig_ss_b || { echo "ОШИБКА pb2d-w70-orig-ss-b"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",pb2d-w70-our-m1-a,"* ]]; then job_pb2d_w70_our_m1_a || { echo "ОШИБКА pb2d-w70-our-m1-a"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",pb2d-w70-our-m2-a,"* ]]; then job_pb2d_w70_our_m2_a || { echo "ОШИБКА pb2d-w70-our-m2-a"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w70-orig-g-a,"* ]]; then job_w70_orig_g_a || { echo "ОШИБКА w70-orig-g-a"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w70-orig-g-b,"* ]]; then job_w70_orig_g_b || { echo "ОШИБКА w70-orig-g-b"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w70-orig-g-c,"* ]]; then job_w70_orig_g_c || { echo "ОШИБКА w70-orig-g-c"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w70-orig-g-d,"* ]]; then job_w70_orig_g_d || { echo "ОШИБКА w70-orig-g-d"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w70-orig-ss-a,"* ]]; then job_w70_orig_ss_a || { echo "ОШИБКА w70-orig-ss-a"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w70-orig-ss-b,"* ]]; then job_w70_orig_ss_b || { echo "ОШИБКА w70-orig-ss-b"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w70-orig-ss-c,"* ]]; then job_w70_orig_ss_c || { echo "ОШИБКА w70-orig-ss-c"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w70-orig-ss-d,"* ]]; then job_w70_orig_ss_d || { echo "ОШИБКА w70-orig-ss-d"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w70-our-m1-a,"* ]]; then job_w70_our_m1_a || { echo "ОШИБКА w70-our-m1-a"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w70-our-m1-b,"* ]]; then job_w70_our_m1_b || { echo "ОШИБКА w70-our-m1-b"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w70-our-m2-a,"* ]]; then job_w70_our_m2_a || { echo "ОШИБКА w70-our-m2-a"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w70-our-m2-b,"* ]]; then job_w70_our_m2_b || { echo "ОШИБКА w70-our-m2-b"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w70-our-m2-c,"* ]]; then job_w70_our_m2_c || { echo "ОШИБКА w70-our-m2-c"; fail=1; }; fi
exit $fail
