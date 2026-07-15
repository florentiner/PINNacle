#!/usr/bin/env bash
# ROUND 7 -- the Fourier-resolution recipe sweep to beat deepxde origin on KS.
# Measures the reach-factor curve: {origin(FNN), ablation_A(modified MLP)} x {10,16,24,32 modes}
# x 3 seeds. origin@10 is the deepxde baseline (0.914). Prediction: >=24 modes beats it by
# 7-37x IF optimization reaches the lifted representation ceiling (10m=0.72, 24m=0.095, 32m=0.02).
# Best on GPU (32-mode FNN ~2.5h/run on CPU; modified-MLP ~3x). Resumable.
#   Transfer back:  zip -r round7_recipe.zip runs_recipe
set -euo pipefail
cd "$(dirname "$0")/../.."
export DDEBACKEND=pytorch KMP_DUPLICATE_LIB_OK=TRUE
RA=experiments/landscape_compare/run_all.py
for MODES in 10 16 24 32; do
  python $RA --pdes kuramoto_sivashinsky --methods ablation_none ablation_A \
    --iterations 30000 --no-landscape --fourier-modes $MODES \
    --n-repeats 3 --parallel 3 \
    --out "runs_recipe/modes_${MODES}"
done
echo "done -> zip -r round7_recipe.zip runs_recipe"
