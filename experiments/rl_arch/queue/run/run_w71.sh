#!/usr/bin/env bash
# Сгенерировано experiments/rl_arch/queue/export_wave.py — шаги задач очереди, как в кернеле Kaggle.
set -uo pipefail
cd "$(dirname "$0")"/. 2>/dev/null || true
cd "$(git rev-parse --show-toplevel)"
export DDEBACKEND=pytorch
if [ -z "${HF_TOKEN_WRITE:-}${HF_TOKEN:-}" ]; then echo "нужен HF_TOKEN_WRITE (результаты идут в HF)"; exit 2; fi
PY="${PYTHON:-python}"

job_pb2d_w71_cnxd4_m1() {  # ConvNeXt без предобучения + D4, журналы 3 x 20, сид обучения 1, мягкая обрезка, сиды 42,43,44,45,46
  echo "=== pb2d-w71-cnxd4-m1 (~5.05 GPU-ч)"
  "$PY" experiments/rl_arch/offline_rl.py --variant convnext_dqn --subdir poisson_boltzmann_2d --chain-fix --reward-form delta --init-err 0.811 --value-bound --gamma 0.99 --episode-budget 7000 --ctx-kmax 70 --save-model --scalar-ctx --encoder convnext --mb-aux 0.5 --mb-h 3 --mb-lam 0.5 --mb-subdirs burgers1d,poisson2d_classic,ns2d_backstep --mb-src-chains 20 --mb-src-seed 0 --target-buffer rl_arch/buffers_new/tbuf_pb2d_w0 --teacher-from rl_arch/agents_online/qpb2d_teach_mbpo_w65src3n20.pt --teacher-w 0.5 --two-nets --seeds 1 --model-tag w71cnxd4two --aug d4 || return 1
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file /tmp/agent_convnext_dqn_cfix_rdelta_vb_w71cnxd4two_seed1.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde poissonboltzmann2d --budget 7000 --hours 2.0 --resume --ckpt-every 10 --seeds 42,43,44,45,46 --tag t3m_w71cnxd4two_m1_ss || return 1
}
job_pb2d_w71_cnxd4_m2() {  # ConvNeXt без предобучения + D4, журналы 3 x 20, сид обучения 2, мягкая обрезка, сиды 42,43,44,45,46
  echo "=== pb2d-w71-cnxd4-m2 (~5.05 GPU-ч)"
  "$PY" experiments/rl_arch/offline_rl.py --variant convnext_dqn --subdir poisson_boltzmann_2d --chain-fix --reward-form delta --init-err 0.811 --value-bound --gamma 0.99 --episode-budget 7000 --ctx-kmax 70 --save-model --scalar-ctx --encoder convnext --mb-aux 0.5 --mb-h 3 --mb-lam 0.5 --mb-subdirs burgers1d,poisson2d_classic,ns2d_backstep --mb-src-chains 20 --mb-src-seed 0 --target-buffer rl_arch/buffers_new/tbuf_pb2d_w0 --teacher-from rl_arch/agents_online/qpb2d_teach_mbpo_w65src3n20.pt --teacher-w 0.5 --two-nets --seeds 2 --model-tag w71cnxd4two --aug d4 || return 1
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file /tmp/agent_convnext_dqn_cfix_rdelta_vb_w71cnxd4two_seed2.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde poissonboltzmann2d --budget 7000 --hours 2.0 --resume --ckpt-every 10 --seeds 42,43,44,45,46 --tag t3m_w71cnxd4two_m2_ss || return 1
}
job_pb2d_w71_pre_cnx() {  # MAE на 5000 процедурных цепочках, энкодер ConvNeXt, 1400 шагов (ноль реальных данных)
  echo "=== pb2d-w71-pre-cnx (~0.6 GPU-ч)"
  "$PY" experiments/rl_arch/cv_probe.py --tag w71proccnx --procedural 5000 --configs ssl-mae-convnext --pretrain-only --ssl-steps 1400 --save-encoder --out-dir /kaggle/working || return 1
}
job_w71_cnx_m1() {  # ConvNeXt без предобучения, без аугментации, журналы 3 x 20, сид обучения 1, мягкая обрезка, сиды 42,43,44
  echo "=== w71-cnx-m1 (~5.9 GPU-ч)"
  "$PY" experiments/rl_arch/offline_rl.py --variant convnext_dqn --subdir ns2d_liddriven --chain-fix --reward-form delta --init-err 0.49 --value-bound --gamma 0.99 --episode-budget 7000 --ctx-kmax 70 --save-model --scalar-ctx --encoder convnext --mb-aux 0.5 --mb-h 3 --mb-lam 0.5 --mb-subdirs burgers1d,poisson2d_classic,ns2d_backstep --mb-src-chains 20 --mb-src-seed 0 --target-buffer rl_arch/buffers_new/tbuf_ns2d_w0,rl_arch/buffers_new/tbuf_ns2d_w1 --teacher-from rl_arch/agents_online/q_teach_mbpo_w65src3n20.pt --teacher-w 0.5 --two-nets --seeds 1 --model-tag w71cnxtwo || return 1
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file /tmp/agent_convnext_dqn_cfix_rdelta_vb_ns2d_w71cnxtwo_seed1.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 42,43,44 --tag t3m_w71cnxtwo_m1_ss || return 1
}
job_w71_cnx_m2() {  # ConvNeXt без предобучения, без аугментации, журналы 3 x 20, сид обучения 2, мягкая обрезка, сиды 42,43,44
  echo "=== w71-cnx-m2 (~5.9 GPU-ч)"
  "$PY" experiments/rl_arch/offline_rl.py --variant convnext_dqn --subdir ns2d_liddriven --chain-fix --reward-form delta --init-err 0.49 --value-bound --gamma 0.99 --episode-budget 7000 --ctx-kmax 70 --save-model --scalar-ctx --encoder convnext --mb-aux 0.5 --mb-h 3 --mb-lam 0.5 --mb-subdirs burgers1d,poisson2d_classic,ns2d_backstep --mb-src-chains 20 --mb-src-seed 0 --target-buffer rl_arch/buffers_new/tbuf_ns2d_w0,rl_arch/buffers_new/tbuf_ns2d_w1 --teacher-from rl_arch/agents_online/q_teach_mbpo_w65src3n20.pt --teacher-w 0.5 --two-nets --seeds 2 --model-tag w71cnxtwo || return 1
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file /tmp/agent_convnext_dqn_cfix_rdelta_vb_ns2d_w71cnxtwo_seed2.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 42,43,44 --tag t3m_w71cnxtwo_m2_ss || return 1
}
job_w71_cnxd4_m1() {  # ConvNeXt без предобучения + D4, журналы 3 x 20, сид обучения 1, мягкая обрезка, сиды 42,43,44
  echo "=== w71-cnxd4-m1 (~5.9 GPU-ч)"
  "$PY" experiments/rl_arch/offline_rl.py --variant convnext_dqn --subdir ns2d_liddriven --chain-fix --reward-form delta --init-err 0.49 --value-bound --gamma 0.99 --episode-budget 7000 --ctx-kmax 70 --save-model --scalar-ctx --encoder convnext --mb-aux 0.5 --mb-h 3 --mb-lam 0.5 --mb-subdirs burgers1d,poisson2d_classic,ns2d_backstep --mb-src-chains 20 --mb-src-seed 0 --target-buffer rl_arch/buffers_new/tbuf_ns2d_w0,rl_arch/buffers_new/tbuf_ns2d_w1 --teacher-from rl_arch/agents_online/q_teach_mbpo_w65src3n20.pt --teacher-w 0.5 --two-nets --seeds 1 --model-tag w71cnxd4two --aug d4 || return 1
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file /tmp/agent_convnext_dqn_cfix_rdelta_vb_ns2d_w71cnxd4two_seed1.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 42,43,44 --tag t3m_w71cnxd4two_m1_ss || return 1
}
job_w71_cnxd4_m2() {  # ConvNeXt без предобучения + D4, журналы 3 x 20, сид обучения 2, мягкая обрезка, сиды 42,43,44
  echo "=== w71-cnxd4-m2 (~5.9 GPU-ч)"
  "$PY" experiments/rl_arch/offline_rl.py --variant convnext_dqn --subdir ns2d_liddriven --chain-fix --reward-form delta --init-err 0.49 --value-bound --gamma 0.99 --episode-budget 7000 --ctx-kmax 70 --save-model --scalar-ctx --encoder convnext --mb-aux 0.5 --mb-h 3 --mb-lam 0.5 --mb-subdirs burgers1d,poisson2d_classic,ns2d_backstep --mb-src-chains 20 --mb-src-seed 0 --target-buffer rl_arch/buffers_new/tbuf_ns2d_w0,rl_arch/buffers_new/tbuf_ns2d_w1 --teacher-from rl_arch/agents_online/q_teach_mbpo_w65src3n20.pt --teacher-w 0.5 --two-nets --seeds 2 --model-tag w71cnxd4two --aug d4 || return 1
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file /tmp/agent_convnext_dqn_cfix_rdelta_vb_ns2d_w71cnxd4two_seed2.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 42,43,44 --tag t3m_w71cnxd4two_m2_ss || return 1
}
job_pb2d_w71_pcnxd4_m1() {  # ConvNeXt + процедурный MAE + D4, журналы 3 x 20, сид обучения 1, мягкая обрезка, сиды 42,43,44,45,46
  echo "=== pb2d-w71-pcnxd4-m1 (~5.05 GPU-ч)"
  "$PY" experiments/rl_arch/offline_rl.py --variant convnext_dqn --subdir poisson_boltzmann_2d --chain-fix --reward-form delta --init-err 0.811 --value-bound --gamma 0.99 --episode-budget 7000 --ctx-kmax 70 --save-model --scalar-ctx --encoder convnext --encoder-init rl_arch/encoders/cvprobe_w71proccnx_enc_mae_convnext.pt --mb-aux 0.5 --mb-h 3 --mb-lam 0.5 --mb-subdirs burgers1d,poisson2d_classic,ns2d_backstep --mb-src-chains 20 --mb-src-seed 0 --target-buffer rl_arch/buffers_new/tbuf_pb2d_w0 --teacher-from rl_arch/agents_online/qpb2d_teach_mbpo_w65src3n20.pt --teacher-w 0.5 --two-nets --seeds 1 --model-tag w71pcnxd4two --aug d4 || return 1
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file /tmp/agent_convnext_dqn_cfix_rdelta_vb_w71pcnxd4two_seed1.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde poissonboltzmann2d --budget 7000 --hours 2.0 --resume --ckpt-every 10 --seeds 42,43,44,45,46 --tag t3m_w71pcnxd4two_m1_ss || return 1
}
job_pb2d_w71_pcnxd4_m2() {  # ConvNeXt + процедурный MAE + D4, журналы 3 x 20, сид обучения 2, мягкая обрезка, сиды 42,43,44,45,46
  echo "=== pb2d-w71-pcnxd4-m2 (~5.05 GPU-ч)"
  "$PY" experiments/rl_arch/offline_rl.py --variant convnext_dqn --subdir poisson_boltzmann_2d --chain-fix --reward-form delta --init-err 0.811 --value-bound --gamma 0.99 --episode-budget 7000 --ctx-kmax 70 --save-model --scalar-ctx --encoder convnext --encoder-init rl_arch/encoders/cvprobe_w71proccnx_enc_mae_convnext.pt --mb-aux 0.5 --mb-h 3 --mb-lam 0.5 --mb-subdirs burgers1d,poisson2d_classic,ns2d_backstep --mb-src-chains 20 --mb-src-seed 0 --target-buffer rl_arch/buffers_new/tbuf_pb2d_w0 --teacher-from rl_arch/agents_online/qpb2d_teach_mbpo_w65src3n20.pt --teacher-w 0.5 --two-nets --seeds 2 --model-tag w71pcnxd4two --aug d4 || return 1
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file /tmp/agent_convnext_dqn_cfix_rdelta_vb_w71pcnxd4two_seed2.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde poissonboltzmann2d --budget 7000 --hours 2.0 --resume --ckpt-every 10 --seeds 42,43,44,45,46 --tag t3m_w71pcnxd4two_m2_ss || return 1
}
job_w71_cnx_m1_b() {  # ConvNeXt без предобучения, без аугментации, журналы 3 x 20, сид обучения 1, мягкая обрезка, сиды 45,46
  echo "=== w71-cnx-m1-b (~3.4 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/models/convnext_dqn_cfix_rdelta_vb_ns2d_w71cnxtwo_seed1.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 45,46 --tag t3m_w71cnxtwo_m1_ss || return 1
}
job_w71_cnx_m2_b() {  # ConvNeXt без предобучения, без аугментации, журналы 3 x 20, сид обучения 2, мягкая обрезка, сиды 45,46
  echo "=== w71-cnx-m2-b (~3.4 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/models/convnext_dqn_cfix_rdelta_vb_ns2d_w71cnxtwo_seed2.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 45,46 --tag t3m_w71cnxtwo_m2_ss || return 1
}
job_w71_cnxd4_m1_b() {  # ConvNeXt без предобучения + D4, журналы 3 x 20, сид обучения 1, мягкая обрезка, сиды 45,46
  echo "=== w71-cnxd4-m1-b (~3.4 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/models/convnext_dqn_cfix_rdelta_vb_ns2d_w71cnxd4two_seed1.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 45,46 --tag t3m_w71cnxd4two_m1_ss || return 1
}
job_w71_cnxd4_m2_b() {  # ConvNeXt без предобучения + D4, журналы 3 x 20, сид обучения 2, мягкая обрезка, сиды 45,46
  echo "=== w71-cnxd4-m2-b (~3.4 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/models/convnext_dqn_cfix_rdelta_vb_ns2d_w71cnxd4two_seed2.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 45,46 --tag t3m_w71cnxd4two_m2_ss || return 1
}
job_w71_pcnxd4_m1() {  # ConvNeXt + процедурный MAE + D4, журналы 3 x 20, сид обучения 1, мягкая обрезка, сиды 42,43,44
  echo "=== w71-pcnxd4-m1 (~5.9 GPU-ч)"
  "$PY" experiments/rl_arch/offline_rl.py --variant convnext_dqn --subdir ns2d_liddriven --chain-fix --reward-form delta --init-err 0.49 --value-bound --gamma 0.99 --episode-budget 7000 --ctx-kmax 70 --save-model --scalar-ctx --encoder convnext --encoder-init rl_arch/encoders/cvprobe_w71proccnx_enc_mae_convnext.pt --mb-aux 0.5 --mb-h 3 --mb-lam 0.5 --mb-subdirs burgers1d,poisson2d_classic,ns2d_backstep --mb-src-chains 20 --mb-src-seed 0 --target-buffer rl_arch/buffers_new/tbuf_ns2d_w0,rl_arch/buffers_new/tbuf_ns2d_w1 --teacher-from rl_arch/agents_online/q_teach_mbpo_w65src3n20.pt --teacher-w 0.5 --two-nets --seeds 1 --model-tag w71pcnxd4two --aug d4 || return 1
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file /tmp/agent_convnext_dqn_cfix_rdelta_vb_ns2d_w71pcnxd4two_seed1.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 42,43,44 --tag t3m_w71pcnxd4two_m1_ss || return 1
}
job_w71_pcnxd4_m2() {  # ConvNeXt + процедурный MAE + D4, журналы 3 x 20, сид обучения 2, мягкая обрезка, сиды 42,43,44
  echo "=== w71-pcnxd4-m2 (~5.9 GPU-ч)"
  "$PY" experiments/rl_arch/offline_rl.py --variant convnext_dqn --subdir ns2d_liddriven --chain-fix --reward-form delta --init-err 0.49 --value-bound --gamma 0.99 --episode-budget 7000 --ctx-kmax 70 --save-model --scalar-ctx --encoder convnext --encoder-init rl_arch/encoders/cvprobe_w71proccnx_enc_mae_convnext.pt --mb-aux 0.5 --mb-h 3 --mb-lam 0.5 --mb-subdirs burgers1d,poisson2d_classic,ns2d_backstep --mb-src-chains 20 --mb-src-seed 0 --target-buffer rl_arch/buffers_new/tbuf_ns2d_w0,rl_arch/buffers_new/tbuf_ns2d_w1 --teacher-from rl_arch/agents_online/q_teach_mbpo_w65src3n20.pt --teacher-w 0.5 --two-nets --seeds 2 --model-tag w71pcnxd4two --aug d4 || return 1
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file /tmp/agent_convnext_dqn_cfix_rdelta_vb_ns2d_w71pcnxd4two_seed2.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 42,43,44 --tag t3m_w71pcnxd4two_m2_ss || return 1
}
job_w71_pcnxd4_m1_b() {  # ConvNeXt + процедурный MAE + D4, журналы 3 x 20, сид обучения 1, мягкая обрезка, сиды 45,46
  echo "=== w71-pcnxd4-m1-b (~3.4 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/models/convnext_dqn_cfix_rdelta_vb_ns2d_w71pcnxd4two_seed1.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 45,46 --tag t3m_w71pcnxd4two_m1_ss || return 1
}
job_w71_pcnxd4_m2_b() {  # ConvNeXt + процедурный MAE + D4, журналы 3 x 20, сид обучения 2, мягкая обрезка, сиды 45,46
  echo "=== w71-pcnxd4-m2-b (~3.4 GPU-ч)"
  "$PY" experiments/rl_arch/online_eval_env.py --policy agent --model-file rl_arch/models/convnext_dqn_cfix_rdelta_vb_ns2d_w71pcnxd4two_seed2.pt --guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO --keep-opt --keep-opt-mode safe --pde ns2d_liddriven --budget 7000 --hours 3.5 --resume --ckpt-every 10 --seeds 45,46 --tag t3m_w71pcnxd4two_m2_ss || return 1
}

WANT="${1:-}"
fail=0
if [ -z "$WANT" ] || [[ ",$WANT," == *",pb2d-w71-cnxd4-m1,"* ]]; then job_pb2d_w71_cnxd4_m1 || { echo "ОШИБКА pb2d-w71-cnxd4-m1"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",pb2d-w71-cnxd4-m2,"* ]]; then job_pb2d_w71_cnxd4_m2 || { echo "ОШИБКА pb2d-w71-cnxd4-m2"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",pb2d-w71-pre-cnx,"* ]]; then job_pb2d_w71_pre_cnx || { echo "ОШИБКА pb2d-w71-pre-cnx"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w71-cnx-m1,"* ]]; then job_w71_cnx_m1 || { echo "ОШИБКА w71-cnx-m1"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w71-cnx-m2,"* ]]; then job_w71_cnx_m2 || { echo "ОШИБКА w71-cnx-m2"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w71-cnxd4-m1,"* ]]; then job_w71_cnxd4_m1 || { echo "ОШИБКА w71-cnxd4-m1"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w71-cnxd4-m2,"* ]]; then job_w71_cnxd4_m2 || { echo "ОШИБКА w71-cnxd4-m2"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",pb2d-w71-pcnxd4-m1,"* ]]; then job_pb2d_w71_pcnxd4_m1 || { echo "ОШИБКА pb2d-w71-pcnxd4-m1"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",pb2d-w71-pcnxd4-m2,"* ]]; then job_pb2d_w71_pcnxd4_m2 || { echo "ОШИБКА pb2d-w71-pcnxd4-m2"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w71-cnx-m1-b,"* ]]; then job_w71_cnx_m1_b || { echo "ОШИБКА w71-cnx-m1-b"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w71-cnx-m2-b,"* ]]; then job_w71_cnx_m2_b || { echo "ОШИБКА w71-cnx-m2-b"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w71-cnxd4-m1-b,"* ]]; then job_w71_cnxd4_m1_b || { echo "ОШИБКА w71-cnxd4-m1-b"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w71-cnxd4-m2-b,"* ]]; then job_w71_cnxd4_m2_b || { echo "ОШИБКА w71-cnxd4-m2-b"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w71-pcnxd4-m1,"* ]]; then job_w71_pcnxd4_m1 || { echo "ОШИБКА w71-pcnxd4-m1"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w71-pcnxd4-m2,"* ]]; then job_w71_pcnxd4_m2 || { echo "ОШИБКА w71-pcnxd4-m2"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w71-pcnxd4-m1-b,"* ]]; then job_w71_pcnxd4_m1_b || { echo "ОШИБКА w71-pcnxd4-m1-b"; fail=1; }; fi
if [ -z "$WANT" ] || [[ ",$WANT," == *",w71-pcnxd4-m2-b,"* ]]; then job_w71_pcnxd4_m2_b || { echo "ОШИБКА w71-pcnxd4-m2-b"; fail=1; }; fi
exit $fail
