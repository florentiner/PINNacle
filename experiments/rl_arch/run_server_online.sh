#!/usr/bin/env bash
# Онлайн-оценка политик в СРЕДЕ АВТОРОВ (состояние = поверхность лосса в латенте
# автоэнкодера). Один сид ~ 12 часов (столько же, сколько их собственные прогоны
# в манифесте буфера), поэтому запускать на серверах без лимита сессии.
#
#   сервер 1:  bash experiments/rl_arch/run_server_online.sh 1
#   сервер 2:  bash experiments/rl_arch/run_server_online.sh 2
#
# Перед запуском:  export HF_TOKEN_WRITE=<токен на запись>
set -u
SERVER="${1:?укажите номер сервера: 1 или 2}"
SEEDS="${SEEDS:-42,43,44}"
BUDGET="${BUDGET:-31000}"
PDE="${PDE:-poissonboltzmann2d}"
cd "$(dirname "$0")/../.."

if [ -z "${HF_TOKEN_WRITE:-}${HF_TOKEN:-}" ]; then
  echo "Нужен HF_TOKEN_WRITE — результаты пишутся на HF после каждого сида." >&2
  exit 1
fi
export DDEBACKEND=pytorch

run() {  # policy, model-file (или -), tag
  local pol="$1" mf="$2" tag="$3"
  echo -e "\n######## $tag | policy=$pol | seeds=$SEEDS | pde=$PDE ########"
  local args=(--policy "$pol" --seeds "$SEEDS" --budget "$BUDGET" --pde "$PDE" --tag "$tag")
  [ "$mf" != "-" ] && args+=(--model-file "$mf")
  python3 experiments/rl_arch/online_eval_env.py "${args[@]}"
}

if [ "$SERVER" = "1" ]; then
  # их бэйзлайн + случайная политика (контроль в той же среде)
  run agent  rl_arch/models/their_dqn_poisson3d_seed1.pt  their_dqn
  run random -                                            random
else
  # предлагаемые улучшения
  run agent  rl_arch/models/cnx_cql_poisson3d_seed1.pt    cnx_cql
  run agent  rl_arch/models/cnx_cql_fixns_seed1.pt        cnx_cql_fixns
fi
echo -e "\nГотово. Результаты: rl_arch/online_env/*.json в датасете danil-e/pinnacle-optuna-db"
