#!/usr/bin/env bash
# ============================================================================
# SCRIPT 2 -- Round-5 additional experiments for the GPU machine.
# Each block tests one open hypothesis (see HYPOTHESIS.md, Round 5); run all, or comment
# out blocks you want to skip. Everything is resumable (completed cells are skipped).
#
#   (a) H13 -- A-vs-origin significance: 15 extra seeds (1237-1251) of ablation_A and
#       ablation_none at the benchmark config, giving 18 paired seeds total with the
#       existing 1234-1236 -- the sample size the power analysis says is needed to decide
#       A's ~0.5% advantage at p<0.05 (paired d_z=0.66).
#   (b) H12 -- precision floor: the horizon argument (t* ~ ln(1/eps)/lambda) says the
#       tracked horizon extends only if the achievable field error drops. float64 removes
#       the float32 arithmetic floor: if the horizon moves, precision was binding; if not,
#       optimization (not arithmetic) sets eps. origin / ablation_A / ablation_all on KS,
#       3 seeds, --float64.
#   (c) H15 -- Random Weight Factorization (Expert's Guide rec., mu=1 sigma=0.1):
#       prediction err(A + RWF) <= err(A) at the same budget. 3 seeds, KS.
#
# Outputs: runs_seeds18/, runs_float64/, runs_rwf/ -- zip and transfer back:
#     zip -r round5_additional.zip runs_seeds18 runs_float64 runs_rwf
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export DDEBACKEND=pytorch KMP_DUPLICATE_LIB_OK=TRUE

RA=experiments/landscape_compare/run_all.py

echo "================ (a) H13: 18-seed A-vs-origin significance ================"
python $RA --pdes kuramoto_sivashinsky \
    --methods ablation_none ablation_A \
    --iterations 30000 --no-landscape \
    --seeds 1237 1238 1239 1240 1241 1242 1243 1244 1245 1246 1247 1248 1249 1250 1251 \
    --parallel 3 \
    --out runs_seeds18

echo "================ (c) H15: Random Weight Factorization ================"
python $RA --pdes kuramoto_sivashinsky \
    --methods ablation_A ablation_all \
    --iterations 30000 --rwf \
    --n-repeats 3 --parallel 3 --grid-xnum 15 \
    --out runs_rwf

echo "================ (b, runs LAST) H12: float64 precision-floor test ================"
# WARNING: on consumer GPUs (GeForce/RTX) fp64 throughput is 1/32..1/64 of fp32 -- this block
# may be VERY slow there (fine on A100/V100/H100). It runs last so you can stop after (a)+(c)
# and still have complete results for those hypotheses.
python $RA --pdes kuramoto_sivashinsky \
    --methods origin ablation_A ablation_all \
    --iterations 30000 --float64 \
    --n-repeats 3 --parallel 3 --grid-xnum 15 \
    --out runs_float64

echo "================ done ================"
echo "Transfer back:  zip -r round5_additional.zip runs_seeds18 runs_float64 runs_rwf"
