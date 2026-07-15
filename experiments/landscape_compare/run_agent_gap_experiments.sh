#!/usr/bin/env bash
# ============================================================================
# ROUND 6 -- PINN-PELINE agent-gap oracle experiments (GPU machine).
# Tests H16-H19 (HYPOTHESIS.md Round 6) WITHOUT integrating the RL agent: every
# single-episode policy the DQN can express is a fixed optimizer chain, so a grid over
# the paper's action space upper-bounds the trained agent at equal budget.
#
#   E1 (H16): 12-chain oracle grid from the PELINE action space (Adam/L-BFGS/PSO stages),
#             origin(+fixes) config, 30k budget, 3 seeds, KS + GS.
#   E2 (H17): the switch-timing subset of E1 (same components, split at 25/50/75/90%).
#   E3 (H18): 9 extra GS origin seeds (1237-1245) for best-of-k restart-insurance stats.
#   E4 (H19): offline -- analyzed from E1/E3 loss_history.csv (per-component) +
#             metrics_history.csv after transfer; no extra runs.
#
# All runs are --no-landscape (histories carry the needed signals), so cells are cheap.
# Everything resumable. Transfer back:
#     zip -r round6_results.zip runs_chains runs_gs_seeds
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export DDEBACKEND=pytorch KMP_DUPLICATE_LIB_OK=TRUE

RE=experiments/landscape_compare/run_experiment.py
RA=experiments/landscape_compare/run_all.py
SEEDS="1234 1235 1236"
ITER=30000

# --- E1/E2: the chain grid ---------------------------------------------------
# name|chain  (names become directory keys; fractions of the 30k budget)
CHAINS=(
  "origin_ctl|adam:1e-3:1.0"                                   # control, identical config to origin
  "adam_hi|adam:1e-2:1.0"
  "adam_lo|adam:1e-4:1.0"
  "lr_ladder|adam:1e-2:0.33,adam:1e-3:0.33,adam:1e-4:0.34"
  "alb_25|adam:1e-3:0.25,lbfgs:1.0:0.75"                       # E2: switch-timing family
  "alb_50|adam:1e-3:0.5,lbfgs:1.0:0.5"
  "alb_75|adam:1e-3:0.75,lbfgs:1.0:0.25"
  "alb_90|adam:1e-3:0.9,lbfgs:1.0:0.1"
  "ahilb|adam:1e-2:0.5,lbfgs:1.0:0.5"
  "pso_start|pso:1e-3:0.01,adam:1e-3:0.99"                     # swarm exploration first (300 PSO iters ~ paper stage length; PSO costs ~30 evals/iter)
  "pso_mid|adam:1e-3:0.495,pso:1e-3:0.01,adam:1e-3:0.495"      # basin hop mid-training (300 PSO iters)
  "alternate|adam:1e-3:0.3,lbfgs:1.0:0.2,adam:1e-4:0.3,lbfgs:0.5:0.2"
)

echo "================ E1/E2: chain oracle grid (${#CHAINS[@]} chains x 2 PDEs x 3 seeds) ================"
for PDE in kuramoto_sivashinsky grayscott; do
  for entry in "${CHAINS[@]}"; do
    NAME="${entry%%|*}"; CHAIN="${entry#*|}"
    for S in $SEEDS; do
      # resume guard: skip cells that already finished (safe to rerun the script)
      if [ -f "runs_chains/${NAME}/seed_${S}/${PDE}/chain/metrics.json" ]; then
        echo "skip ${PDE}/${NAME}/seed_${S} (done)"; continue
      fi
      python $RE --pde $PDE --method chain --chain "$CHAIN" \
        --iterations $ITER --no-landscape --seed $S \
        --out "runs_chains/${NAME}/seed_${S}" &
    done
    wait   # 3 seeds of one chain in parallel
  done
done

# --- E3: GS restart-insurance seeds -------------------------------------------
echo "================ E3: GS origin, 9 extra seeds for best-of-k stats ================"
python $RA --pdes grayscott --methods ablation_none \
    --iterations $ITER --no-landscape \
    --seeds 1237 1238 1239 1240 1241 1242 1243 1244 1245 \
    --parallel 3 \
    --out runs_gs_seeds

echo "================ done ================"
echo "Transfer back:  zip -r round6_results.zip runs_chains runs_gs_seeds"
