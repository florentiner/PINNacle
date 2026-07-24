#!/usr/bin/env bash
# Run the full chain evaluation (all 22 PDEs x 10 seeds) on a GPU server,
# uploading per-seed results to the HF dataset exactly like the Kaggle kernels.
#
# Usage:
#   export HF_TOKEN_WRITE=hf_...          # required for uploads
#   export HF_TOKEN_READ=hf_...           # optional (public repo)
#   ./experiments/chain_eval/run_server.sh                     # all PDEs, GPUs auto
#   ./experiments/chain_eval/run_server.sh --pdes burgers_1d,wave1d --devices 0,1
#   nohup ./experiments/chain_eval/run_server.sh > chain_eval.log 2>&1 &
#
# Interrupted runs are resumable: seeds already present in the HF CSVs are
# skipped on relaunch.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export DDEBACKEND=pytorch

if [[ -z "${HF_TOKEN_WRITE:-}" ]]; then
    echo "WARNING: HF_TOKEN_WRITE is not set — results will only be saved locally." >&2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" experiments/chain_eval/run_all.py --pdes all "$@"
