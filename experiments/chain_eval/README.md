# Chain evaluation: fixed Adam → L-BFGS chain, 22 PDEs × 10 seeds

Evaluates one fixed optimizer chain (no optuna) on all 22 PINNacle PDEs and
streams per-seed metrics into per-PDE CSVs on the Hugging Face dataset
[`danil-e/pinnacle-optuna-db`](https://huggingface.co/datasets/danil-e/pinnacle-optuna-db)
under `csv_chain/{pde_name}.csv`.

The default chain ([chain_adam_lbfgs.json](chain_adam_lbfgs.json)):

```json
[
    {"optimizer": "Adam",  "epochs": 1000,  "lr": 0.001},
    {"optimizer": "LBFGS", "epochs": 30000, "lr": 1.0, "history_size": 100}
]
```

L-BFGS runs with `max_iter=1` per training step, so `epochs` counts true
L-BFGS iterations (strong-Wolfe line search, `history_size` as given).
Optional stage keys: `history_size`, `max_iter`. `PSO` stages are also
supported for backward compatibility with the optuna chains.

Model setup matches the optuna experiments: default PDE constructors,
FNN `100*5` tanh / Glorot normal (`recommend_net` for the two inverse
problems), loss weight 100 on boundary/initial losses.

## CSV output

Same column layout as the old `csv_seed/` files
(`run_timestamp, pde_name, value_type, smoke_test, chain_key, seed, mse_op,
mse_bnd, mse_total, l2re_op, l2re_bnd, l2re_total, elapsed_s, chain_json`),
with `value_type="chain"`. The CSV is re-uploaded after **every finished
seed**, so a killed Kaggle session loses at most the in-flight seed. On
relaunch, seeds already recorded for `(pde_name, chain_key, smoke_test)` are
skipped automatically (`--force` re-runs them).

## Run one PDE locally

```bash
export HF_TOKEN_WRITE=hf_...   # omit to keep results local-only
python experiments/chain_eval/run_chain_pde.py --pde-name burgers_1d
# quick end-to-end check: 2 seeds, 3 epochs per stage, rows marked smoke_test=True
python experiments/chain_eval/run_chain_pde.py --pde-name burgers_1d --n-seeds 2 --test-epochs 3
```

PDE names: `burgers_1d, burgers_2d, poisson2d_classic, poissonboltzmann2d,
poisson3d_complexgeometry, poisson2d_manyarea, heat2d_varyingcoef,
heat2d_multiscale, heat2d_complexgeometry, heat2d_longtime, ns2d_classic,
ns2d_backstep, ns2d_longtime, wave1d, wave2d_heterogeneous, wave2d_longtime,
grayscott, kuramoto_sivashinsky, poissonnd, heatnd, poissoninv, heatinv`

## Run on a GPU server

```bash
export HF_TOKEN_WRITE=hf_...
nohup ./experiments/chain_eval/run_server.sh > chain_eval.log 2>&1 &
# subset / explicit GPUs:
./experiments/chain_eval/run_server.sh --pdes grayscott,heatnd --devices 0,1
```

Seeds are parallelized over the listed GPUs inside each PDE; PDEs run
sequentially. `--workers-per-gpu 2` packs two seed workers on each GPU —
the PINNacle nets are small, so this raises throughput on underutilized
GPUs (drop back to 1 if a heavy PDE hits GPU OOM; failed seeds are retried
on the next launch anyway).

## Run on Kaggle (GPU)

1. `cp experiments/chain_eval/accounts.example.json experiments/chain_eval/accounts.json`
   and fill in the HF tokens and one entry per Kaggle account
   (`kaggle_token` = a `KGAT_...` access token from kaggle.com → Settings →
   API). `accounts.json` is gitignored — never commit tokens.
2. Optionally pin PDEs per account with `"pdes": [...]`; anything unassigned
   is split round-robin across accounts. Kernels request the T4 x2 machine
   (`"machine_shape": "NvidiaTeslaT4"`, the default) — do not switch to
   `NvidiaTeslaP100`: its sm_60 is unsupported by Kaggle's PyTorch build and
   the run silently falls back to CPU.
3. Push (requires `pip install kaggle>=1.7`, internet-enabled private GPU
   kernels are created):

```bash
# tiny end-to-end test first (1 PDE, 2 seeds, 3 epochs):
python experiments/chain_eval/launch_kaggle.py launch --smoke
python experiments/chain_eval/launch_kaggle.py status
# full run:
python experiments/chain_eval/launch_kaggle.py launch
# fetch kernel logs:
python experiments/chain_eval/launch_kaggle.py output
```

Kaggle GPU sessions are capped (~9–12 h) — kernels that die mid-run can
simply be pushed again: already-recorded seeds are skipped, so the run
continues where it stopped.

Note: `landscape_visualization/` (error-landscape plotting) was removed on
this branch; `RL/` still references it and is not usable here (RL work lives
on the `rlpinn_*` branches).
