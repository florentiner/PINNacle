# Chaotic-PDE method comparison via loss-landscape visualization

Controlled experiments to understand **why vanilla PINNs fail on the chaotic PDEs**
(Kuramoto–Sivashinsky, Gray–Scott) and which fixes recover a good solution — using the
repo's loss-landscape visualization to explain the difference. See
[`HYPOTHESIS.md`](HYPOTHESIS.md) for the scientific design; this file is the run guide.

Everything here is **self-contained**: no Comet, no RL, no `.env`, deterministic seed. Each
(pde, method) cell runs in its own process and writes all of its data to disk, so you can
run the matrix on one machine and analyze it anywhere.

## What runs

Default matrix = `{kuramoto_sivashinsky, grayscott, burgers1d} × {origin, causal, frozen}`.
`burgers1d` is the **control** (a PDE PINNs solve well, and where Frozen-PINN is
known-accurate). `origin` = plain MSE loss, `causal` = causal time-weighted loss — **both use
the same paper-backed optimizer pipeline** (below), so the loss is the only variable. `frozen`
is the gradient-free control. Optional one-flag methods: `adam_baseline` (alias of `origin`),
`lbfgs_baseline` (Adam→L-BFGS), `soap`, `soap_causal`.

### Optimizer pipeline (used by every gradient method)

Chosen from the literature, not guessed: both the causal paper (Wang, Sankaran & Perdikaris,
"Respecting causality is all you need…", 2022, arXiv:2203.07404) and the "Expert's Guide to
Training PINNs" (Wang et al. 2023, arXiv:2308.08468) use **Adam only — *not* L-BFGS** for
stiff/chaotic time-dependent PDEs (the Expert's Guide recommends *"Adam exclusively"*), with
**lr 1e-3 and exponential (step) decay ×0.9 every 2000 iterations**, default betas
(0.9, 0.999). Causal loss uses a fixed **ε = 1.0** (Expert's-Guide default, so all temporal
weights converge to 1; the causal paper's alternative anneals ε through [1e-2, 1e-1, 1, 10,
100]) with `num_causal_buckets = 32`. `lbfgs_baseline` keeps an L-BFGS tail on purpose — the
literature says it's *worse* on chaotic, so it's there to confirm that, not as a recommendation.
The papers use ≥1e4 iterations, so pass `--iterations 15000` (or more) for real runs.

### Fixes after the first full run (why causal wasn't beating origin)

The first 30k-iteration run showed every gradient method collapsing to the trivial exact
solution (see `ANALYSIS.md`). Root-causing that against the causal paper's setup found three
gaps, now fixed:

1. **Missing periodicity (a genuine setup bug).** The PINNacle KS/Gray–Scott definitions
   impose *only the IC* — no spatial BC at all — while the reference solutions are exactly
   periodic (edge mismatch = 0, verified). An unconstrained 4th-order PDE is ill-posed: zero
   residual does not pin the network to the periodic reference. Fix: **exact periodicity via a
   Fourier feature embedding** of the spatial inputs (`--fourier-modes`, default ON: KS 10
   modes, GS 5; `0` disables), the hard-constraint approach of the causal paper.
2. **Causal ε was fixed.** Now uses the paper's **annealing schedule** ε ∈ {1e-2, 1e-1, 1, 10,
   100}, advancing when every causal weight exceeds `--causal-delta` (0.99). A fixed moderate ε
   under-weights late times all run when residuals start large — and at the trivial attractor
   (residuals ≈ 0) *any* fixed-ε causal loss degenerates to the origin loss.
   `--causal-eps X` still forces a fixed value.
3. **No time-marching.** `--time-windows N` trains the chaotic PDEs over N sequential time
   windows, warm-starting each window from the previous one and handing the IC across the
   interface — the setting the causal paper actually used for chaotic KS (their Δt = 0.1 ⇒
   `--time-windows 10`). Gradient methods on chaotic PDEs only; ignored elsewhere.

Recommended chaotic rerun + analysis:
```bash
python experiments/landscape_compare/run_all.py \
    --pdes kuramoto_sivashinsky grayscott --n-repeats 3 --iterations 30000 --time-windows 10
python experiments/landscape_compare/compare_landscapes.py --runs runs_landscape_compare
python experiments/landscape_compare/error_landscape_analysis.py --runs runs_landscape_compare
python experiments/landscape_compare/shared_landscape.py --runs runs_landscape_compare
```

### Shared error landscape (the core comparison artifact)

Because every gradient method at a given seed starts from the **same initial weights** (saved
as checkpoint `model-000`), their trajectories can be embedded into **one** shared 2D space:
`shared_landscape.py` trains a single autoencoder on the union of all methods' checkpoints,
evaluates one loss grid (plain/origin loss as the neutral yardstick) and one **TRUE-ERROR
grid** (each decoded grid network's relative-L2 vs the reference), and overlays every method's
path on both maps — same landscape, same start, methods differ only in where they go. Outputs
under `<runs>/shared_landscape/<pde>_seed<k>/` (`loss_map.pdf`, `error_map.pdf`,
`shared_grid.npz`, `trajectories.npz`, `endpoints.json`). Runs on the results machine
(needs torch/deepxde; GPU used if available).

### Controlled weight initialization

Every gradient method at a given seed starts from the **same** network weights (a deterministic
seed-derived Glorot-normal init, independent of loss type), so any difference between `origin`
and `causal` is due to the *method*, not the starting point. Different seeds start from
*different* weights, so `--n-repeats` gives a genuine spread. (Collocation points are likewise
shared across methods within a seed.) `frozen` has no network; its Gray-Scott/Burgers random
features are seeded instead, and KS-frozen is deterministic by construction.

## Requirements

Same environment as PINNacle (PyTorch backend). The analysis step needs only
numpy/scipy/matplotlib.

```bash
export DDEBACKEND=pytorch
export KMP_DUPLICATE_LIB_OK=TRUE      # the scripts also set these defaults themselves
# GPU is used automatically if torch.cuda.is_available(); otherwise CPU.
```

## Run it

**1. Smoke-test the whole pipeline first (tiny settings, minutes):**
```bash
python experiments/landscape_compare/run_all.py --quick
python experiments/landscape_compare/compare_landscapes.py --runs runs_landscape_compare
```
This confirms every cell writes its full layout and the analysis produces a CSV + figures.
Numbers from `--quick` are meaningless — it exists only to validate plumbing.

**2. The real run:**
```bash
# full default matrix {origin, causal, frozen} (gradient methods train for --iterations)
python experiments/landscape_compare/run_all.py --iterations 15000

# add optimizer-variant methods if you also want the "which optimizer" comparison:
python experiments/landscape_compare/run_all.py \
    --methods origin causal lbfgs_baseline soap soap_causal frozen \
    --iterations 15000

# analyze:
python experiments/landscape_compare/compare_landscapes.py --runs runs_landscape_compare
```

`run_all.py` is **resumable** — a cell whose `metrics.json` exists is skipped (use `--force`
to redo). You can also run a single cell directly:
```bash
python experiments/landscape_compare/run_experiment.py --pde grayscott --method frozen
```

**3. Repeat trials (recommended — checks a result is robust, not a fluke):**
```bash
python experiments/landscape_compare/run_all.py --pdes kuramoto_sivashinsky grayscott --n-repeats 3
python experiments/landscape_compare/compare_landscapes.py --runs runs_landscape_compare
```
`--n-repeats 3` runs every requested `(pde, method)` cell 3x with seeds `1234, 1235, 1236`
(or `base, base+1, base+2` if you also pass `--seed base`; use `--seeds 7 42 99` for full
control). Repeats nest under `<out>/seed_<N>/<pde>/<method>/`; `compare_landscapes.py`
detects this automatically and reports **mean ± std across seeds** — a small std means the
result is robust, a large one (or a sign flip) means the single-seed number was noise.
Note: KS-frozen and the fully-deterministic parts of a run are bit-identical across seeds by
construction (no randomness to vary) — that's expected, not a bug; only the network-init/
collocation-sampling (gradient methods) and the random-feature draw (Gray–Scott/Burgers
frozen) actually change with the seed.

### Useful flags (passed through `run_all.py`)
`--pdes`, `--methods`, `--iterations`, `--hidden-layers 100*5`, `--n-save-models 10`,
`--grid-xnum 25` (landscape grid resolution), `--ae-epochs 10000`, `--seed 1234`,
`--n-repeats`, `--seeds`, `--causal-eps`, `--num-causal-buckets`, `--quick`, `--force`.

## Outputs (per cell: `runs_landscape_compare/<pde>/<method>/`)

```
config.json           # every hyperparameter, seed, git commit, timestamps
metrics.json          # relative-L2, MSE/MAE, IC/boundary error, Fourier low/mid/high, wall-clock
solution/fields.npz   # coords, pred, ref, abs_error on the full reference grid  (ALL methods)
loss_history.csv      # per-display-step loss components, one monotonic step axis (gradient)
                      #   across every phase/sub-phase/window (see train_one_model)
metrics_history.csv   # epoch,mse,mae,mxe,l1re,l2re,crmse,ic_mse,bc_mse,bc_rmse,bc_l2re every
                      #   --display-every epochs (PerEpochMetricsCallback); absent for
                      #   --time-windows > 1 runs, where a per-window number against the
                      #   full-domain reference would be misleading -- see metrics.json instead
trajectory_error.csv  # relative-L2 at each landscape checkpoint                  (gradient)
checkpoints/model-*.pt# the saved trajectory of network weights                   (gradient)
landscape/            # 2D-embedded loss landscape                                (gradient)
  trajectory_2d.npy, trajectory_original_nd.npy, trajectory_reconstructed_nd.npy
  grid_2d.npz               # grid_xx, grid_yy, grid_losses (loss_total)
  grid_losses_all.npz       # loss_total, loss_oper, loss_bnd grids
  trajectory_losses.npz     # per-checkpoint losses
  map_*.pdf                 # rendered landscape / error maps + CKA density
frozen/               # coefficients.npz (C(t)); feature_spectrum.npz (conditioning)  (frozen)
```
With `--n-repeats`/`--seeds` > 1, everything above nests one level deeper under
`seed_<N>/<pde>/<method>/...`. Top level: `MANIFEST.json` (status + key metric per cell,
written by `run_all.py`), `compare_summary.csv` (one row per run), `compare_summary_agg.csv`
(one row per `(pde, method)`, mean ± std across seeds) and `comparison_figures/*.pdf`
(bars show std as error bars) — all written by `compare_landscapes.py`.

## Runtime notes / knobs

- **KS Frozen** is a spectral solve — very accurate (relative-L2 ≈ 3e-5) and fast (~7 s; an
  analytic Jacobian tames the stiff BDF integration). `--quick` shrinks modes/tolerances.
- **Gray–Scott Frozen** is *best-effort* (2D tanh features, no exact BC, horizon T=200): it
  runs in ~30 s and captures `u` reasonably but `v` poorly — an expected limitation, and a
  data point in its own right (see H5 in `HYPOTHESIS.md`).
- **Gradient landscapes** are the heavy part: the loss grid is `(grid_xnum+1)²` full PINN
  loss evaluations, and the trajectory autoencoder trains for `--ae-epochs`. Lower
  `--grid-xnum` and `--ae-epochs` if you are CPU-bound; Gray–Scott (2D, ~32k points) is the
  slowest.
- Because each cell is a subprocess, a failure in one cell (e.g. an integrator blow-up) is
  recorded in `MANIFEST.json` and does **not** stop the rest of the matrix.
