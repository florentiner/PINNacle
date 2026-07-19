# Server runners (non-Kaggle backup + seed-robustness runs)

Standalone GPU runs of the causal-PINN comparison on your own server.
Purpose: (a) backup path independent of Kaggle, (b) repeat with a different
seed to check the Kaggle results aren't outliers.

## Setup

```bash
git clone <this repo> && cd PINNacle    # branch: causal-pinn-comparison
pip install "jax[cuda12]"               # training engine (GPU)
pip install torch numpy scipy matplotlib dill pandas scikit-learn
# ref data required: ref/Kuramoto_Sivashinsky.dat, ref/grayscott.dat
```

## Run (from repo root)

```bash
# KS, new seed (paper/Kaggle runs used seed 1234)
CUDA_VISIBLE_DEVICES=0 python scripts/server_run_ks.py \
    --outdir runs/server-ks-seed2024 --seed 2024

# GS, new seed (plain encoding — the pass-1 winner)
CUDA_VISIBLE_DEVICES=1 python scripts/server_run_gs.py \
    --outdir runs/server-gs-seed2024 --seed 2024
```

- **Resume is automatic**: rerun the same command; it picks up `jax_ckpt.pkl`.
  Use `--max-hours H` to bound a single invocation (clean exit + checkpoint).
- Expected cost (P100-class GPU): KS ≈ 25–35 h total; GS: measure first
  windows and extrapolate. Both are fully checkpointed.
- Outputs per run dir (`.../0-0/`): error landscapes (`u_err.png`,
  `arrays/err_stitched_final.npy`), per-window fields, causal history
  (`causal/history_jax.npz` — full W/L_t vectors), parameter trajectories
  (`trajectory/w{k}_trajectory_flat.npy` after consolidation), metrics.

## Compare against the vanilla baseline

```bash
python analysis/compare_chaotic.py --case ks \
    --baseline runs/07.18-13.19.39-baseline-chaotic/0-0 \
    --causal runs/server-ks-seed2024/0-0 --out analysis/out/ks-server-seed2024
python analysis/loss_landscape.py --causal runs/server-ks-seed2024/0-0 \
    --out analysis/out/ks-losslandscape-seed2024
```

(The baseline itself can be re-run with another seed too:
`python benchmark_chaotic.py --device 0 --seed 2024 --iter 20000 --hyp-data`.)
