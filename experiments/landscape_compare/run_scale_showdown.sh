#!/usr/bin/env bash
# ============================================================================
# SCRIPT 1 -- Round-5 main experiment for the GPU machine (two steps).
#
# STEP 1: H-KS-2 window dose-response (the last unconfirmed Round-4 hypothesis).
#   ablation_all on KS with 2 and 5 time windows at the SAME 30k budget/seeds as the
#   existing sweep (W=10 and origin already exist in runs_landscape_compare).
#   Prediction: rel-L2 improves monotonically as windows decrease
#   (W10 0.960 -> W5 -> W2 -> ~ablation_CAG 0.939 at the W=1 equivalent),
#   approaching but not beating origin (0.918) -- confirming that at fixed budget the
#   stack's KS deficit is window starvation, not the ingredients themselves.
#
# STEP 2: the scale showdown (H11). The papers' successful chaotic runs use width
#   128-512 modified MLPs and 10-100x more iterations per window than the benchmark
#   sweep; this step gives the stack that scale and asks whether it NOW beats origin:
#     * origin        (plain FNN 256*4, origin loss)   -- theoretically worst
#     * ablation_A    (modified MLP 256*4, origin loss) -- best architecture, no stack
#     * ablation_all  (full stack, modified MLP 256*4, 10 windows x 15k iterations)
#   150k total iterations each, Expert's-Guide warmup 5000, 3 seeds, KS.
#   Set SCALE_GS=1 to also run the same trio on Gray-Scott.
#
# Outputs: runs_dose/ and runs_scale/ -- zip both and transfer back for analysis:
#     zip -r round5_results.zip runs_dose runs_scale
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export DDEBACKEND=pytorch KMP_DUPLICATE_LIB_OK=TRUE

RE=experiments/landscape_compare/run_experiment.py
RA=experiments/landscape_compare/run_all.py
SEEDS="1234 1235 1236"

echo "================ STEP 1: H-KS-2 window dose-response ================"
# separate --out per window count (same pde/method name would collide in one tree)
for W in 2 5; do
  for S in $SEEDS; do
    python $RE --pde kuramoto_sivashinsky --method ablation_all \
      --iterations 30000 --time-windows $W --no-landscape \
      --seed $S --out "runs_dose/w${W}/seed_${S}" &
  done
  wait   # 3 seeds of one W in parallel; W-counts sequential to bound VRAM
done

echo "================ STEP 2: scale showdown (H11) ================"
# 3 methods x 3 seeds at width 256*4, 150k iterations, warmup 5000. --parallel 3 puts the
# three seeds of the whole batch through together; landscape tier kept ON (it is the point
# of the study) but grid reduced to bound the cost of 256-width loss evaluations.
python $RA --pdes kuramoto_sivashinsky \
    --methods origin ablation_A ablation_all \
    --hidden-layers "256*4" --iterations 150000 --warmup 5000 \
    --n-repeats 3 --parallel 3 --grid-xnum 15 \
    --out runs_scale

if [ "${SCALE_GS:-0}" = "1" ]; then
  python $RA --pdes grayscott \
      --methods origin ablation_A ablation_all \
      --hidden-layers "256*4" --iterations 150000 --warmup 5000 \
      --n-repeats 3 --parallel 3 --grid-xnum 15 \
      --out runs_scale
fi

echo "================ done ================"
echo "Transfer back:  zip -r round5_results.zip runs_dose runs_scale"
