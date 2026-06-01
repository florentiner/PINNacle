#!/bin/bash
# Parallel workers on Apple Silicon with PyTorch MPS.
#
# Usage:
#   ./heatinv_parallel_run_mps.sh [N_WORKERS]
#   N_WORKERS defaults to auto: min(4, max(2, hw.physicalcpu / 2))
#   Override: MPS_N_WORKERS=2 ./heatinv_parallel_run_mps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
else
    PYTHON="python3"
fi

SCRIPT="experiments/optimization_multi_pde/heatinv_chain.py"
log_enable="true"

if [ -n "${MPS_N_WORKERS:-}" ]; then
    N_WORKERS="$MPS_N_WORKERS"
elif [ -n "${1:-}" ]; then
    N_WORKERS="$1"
else
    PHYS="$(sysctl -n hw.physicalcpu 2>/dev/null || echo 8)"
    N_WORKERS=$(( PHYS / 2 ))
    [ "$N_WORKERS" -lt 2 ] && N_WORKERS=2
    [ "$N_WORKERS" -gt 4 ] && N_WORKERS=4
fi

if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: This script is intended for Apple Silicon (Darwin/arm64). Continuing anyway."
fi

if ! "$PYTHON" -c "import torch; assert torch.backends.mps.is_available(), 'MPS not available'" 2>/dev/null; then
    echo "Error: PyTorch MPS is not available."
    exit 1
fi

export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

echo "PyTorch MPS: OK"
echo "Parallel workers: $N_WORKERS (single shared MPS device)"

for ((i = 1; i <= N_WORKERS; i++)); do
    echo "Starting worker $i/$N_WORKERS..."
    "$PYTHON" "$SCRIPT" --log_key "$log_enable" &
done

wait
echo "All processes finished."
