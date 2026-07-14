# Why do PINNs fail on chaotic PDEs? — hypotheses and how to compare methods

This experiment asks a single question:

> **Vanilla PINNs fail on the two chaotic PDEs (Kuramoto–Sivashinsky and Gray–Scott).
> Which candidate fixes actually recover a good solution, and *why* — what is it about
> the loss landscape that explains the difference?**

We answer it with a **controlled** comparison: identical network, identical training
schedule and identical reference data, changing only the *method*. Each run saves two
tiers of data (see `README.md` for the exact files):

1. **Solution-accuracy tier** — did the method find the right answer? (relative-L2 vs the
   reference, plus a Fourier-band breakdown of the error).
2. **Loss-landscape tier** — what does the optimization surface look like? (the 2D-embedded
   loss landscape for gradient methods; the frozen feature-matrix conditioning for
   Frozen-PINN, which has no gradient-descent landscape).

`compare_landscapes.py` turns these into the quantitative descriptors used below.

---

## The methods being compared

| method            | family                | what it changes                                   |
|-------------------|-----------------------|---------------------------------------------------|
| `adam_baseline`   | gradient PINN         | the failure case — plain Adam, plain MSE loss     |
| `lbfgs_baseline`  | gradient PINN         | Adam warm-up → L-BFGS second-order refinement     |
| `frozen`          | **gradient-free**     | frozen random/Fourier features + ODE integration  |
| `causal`*         | gradient PINN         | Adam + causal time-weighted loss (Wang et al. 2022)|
| `soap`*           | gradient PINN         | SOAP second-order optimizer                       |
| `soap_causal`*    | gradient PINN         | SOAP + causal loss                                |

\* optional (one flag) — off in the default matrix.

The three families attack the problem in three different places, which is what makes the
comparison informative:

- **`causal`** changes the *objective* — it stops the network from fitting late-time data
  before it has fit early-time data, i.e. it reshapes *which* minimum the same landscape
  pushes you toward.
- **`soap` / `lbfgs`** change the *optimizer* — same landscape, but a better-preconditioned
  path through it.
- **`frozen`** changes the *hypothesis class* — it removes the non-convex landscape
  altogether by making the problem linear in the unknowns. This is the crucial control: if
  `frozen` succeeds where the gradient methods fail, the pathology is the *landscape*, not
  the network's ability to represent the solution.

---

## Landscape descriptors (computed by `compare_landscapes.py`)

All are computed on the log-loss surface `L = log10(loss)` over the 2D autoencoder
embedding of the training trajectory.

- **`roughness_tv`** — mean gradient magnitude of `L`, range-normalized. High = corrugated.
- **`roughness_hf`** — fraction of 2D-FFT energy above half the max wavenumber. High =
  fine-scale ruggedness (many nearby ripples), the kind of surface SGD stalls on.
- **`barrier`** — the maximum log-loss along the straight segment from the *initial* to the
  *final* trajectory point, minus the endpoint value. Positive = you must climb over a hump
  to get from start to solution (a non-convex trap between them).
- **`end_curvature`** — discrete Laplacian of `L` at the trajectory endpoint. High = a sharp,
  narrow minimum (typically poor generalization / hard to reach).
- **`basin_fraction`** — fraction of the grid within 10% of the global min. Small = a narrow
  low-loss region; large = a wide flat basin.
- **`n_local_minima`** — number of local minima in the grid (multi-modality).
- **`loss_error_corr`** — correlation between PDE loss and *true* relative-L2 across the
  training checkpoints. **This is the key one.** ~1 means loss is a faithful proxy for error;
  ≤0 means the landscape is *deceptive* — driving the loss down does not drive the error
  down.
- **`condition_number` / `sv_decay_slope`** (Frozen only) — condition number and singular
  decay of the frozen feature matrix. Near 1 / flat decay = a well-posed convex linear
  problem; huge / steep = ill-conditioned.

---

## Hypotheses (falsifiable)

**H1 — the failure is real and specific.**
`adam_baseline` (and `lbfgs_baseline`) reach a *low PDE loss* but a *high relative-L2* on KS
and Gray–Scott, whereas on the Burgers control they reach low relative-L2.
→ Evidence: `relative_l2` column; compare chaotic rows vs the `burgers1d` rows.

**H2 — the chaotic landscape is *deceptive* (the mechanism).**
For KS/Gray–Scott, `loss_error_corr` is weak or negative: the optimizer keeps reducing the
residual while the solution error plateaus or worsens. This is *the* reason gradient descent
is misled — the thing it minimizes stops tracking the thing we care about.
→ Evidence: `loss_error_corr` and the `deceptive_landscape.pdf` figure. Falsified if the
correlation stays strongly positive despite a bad `relative_l2`.

**H3 — the chaotic landscape is geometrically harder.**
KS/Gray–Scott gradient landscapes show higher `roughness_tv` / `roughness_hf`, higher
`barrier`, and/or more `n_local_minima` than the Burgers control.
→ Evidence: `roughness.pdf`, `barrier.pdf`, and those columns.

**H4 — the error is spectral.**
Failed methods carry most of their error in the **mid/high Fourier bands** — they capture the
coarse trend but miss the fine chaotic structure. A fix that works should shift error energy
back toward the low band.
→ Evidence: `fourier_low/mid/high` and `fourier_bands.pdf`.

**H5 — Frozen-PINN succeeds by *avoiding* the landscape.**
Frozen-PINN replaces the non-convex optimization with a well-conditioned linear problem
(`condition_number` small — for KS the Fourier basis gives ~1) and, where the basis suits the
PDE, attains a low `relative_l2` (KS ≈ 1e-4). This isolates the cause: the network *can*
represent the solution; ordinary training just can't *find* it.
→ Evidence: KS `frozen` `relative_l2` and `condition_number` vs the gradient rows.
→ **Boundary of H5:** on Gray–Scott the tanh feature matrix is *ill-conditioned* (cond ≫ 1)
and the `v` field is poorly captured — a real, expected limitation (2D, no exact BC, long
horizon). This is itself a finding: a good frozen basis must match the PDE (Fourier for
periodic KS works; generic tanh features for 2D reaction–diffusion do not).

---

## How to read the comparison to "find the reason"

1. **Establish the failure (H1).** In `compare_summary.csv`, confirm `relative_l2` is large
   for the gradient methods on KS/Gray–Scott and small on Burgers. If a fix helps, its
   `relative_l2` drops.
2. **Find the mechanism (H2).** Look at `loss_error_corr`. A low/negative value on the
   failing cells is the smoking gun: the loss landscape is *deceptive*. This is the headline
   explanation.
3. **Corroborate with geometry (H3).** Cross-check that the failing cells are also the
   rough / high-barrier / multi-modal ones. Rugged + deceptive ⇒ gradient descent settles in
   a low-loss-but-wrong region.
4. **Confirm the symptom (H4).** The Fourier bands show *where* in scale the error lives;
   chaotic cases should be mid/high-heavy.
5. **Attribute the cause (H5).** Frozen-PINN is the control that changes the hypothesis
   class. If it solves KS with condition number ≈ 1 while Adam/L-BFGS fail on the same PDE,
   the problem was the *landscape*, not the model capacity — and the fixes that help
   (`causal` reshaping the objective, `soap`/`lbfgs` preconditioning the path) help precisely
   to the extent they de-deceive / smooth that landscape.

The narrative the data should support: **chaotic PDEs give PINNs a rugged, deceptive loss
landscape where residual and true error decouple; methods win by either reshaping the
objective so low loss again means low error (causal), preconditioning the descent
(SOAP/L-BFGS), or discarding the landscape for a well-conditioned linear solve (Frozen).**

---

## Round 3 — origin vs best_practice: hypotheses testable from `runs_landscape_compare`

The 96-cell ablation sweep (16 ingredient combos × 3 seeds × 2 PDEs, shared init per seed)
contains enough per-run data — error/loss trajectories, per-epoch metrics, 2D loss grids,
checkpoints, solution fields — to ask *why* origin underperforms (or fails identically to)
the best-practice stack, at four different levels of description. Each hypothesis names its
data artifact and its decision rule; `similarity_analysis.py` implements H8–H10.

**H6 — horizon, not asymptote.** If origin's deficit vs best_practice is a *predictability
horizon* effect, the two should differ in the time (KS) / space (GS) *extent* of the
well-fit region, not in the fit quality inside it.
→ Artifact: `solution/fields.npz` (rel-L2 by time band), `metrics_history.csv`.
→ Rule: compare per-band errors; equal early-band + equal collapse point ⇒ same horizon ⇒
   the stack does not extend predictability at this scale (matches Round-2: t≈0.2 for both).

**H7 — different failure *mechanisms*, same failure.** origin should fail by converging
honestly to a stable partial-tracking minimum (flat error curve, corr(loss, err) ≈ +1),
while marching-based stacks should fail by *compounding* (error growing along the window
chain, ending above the shared-init error).
→ Artifact: `trajectory_error.csv` + `landscape/trajectory_losses.npz`.
→ Rule: origin's late error slope ≈ 0 with high corr; W-combos' final error > initial error.
   (Round-2 confirms: origin 1.04→0.914 flat; CW 1.04→1.81, CWA →1.99.)

**H8 — seed-vs-method variance dominance.** If, over the 16 combos, the *between-seed*
variance of landscape/trajectory features is comparable to or larger than the
*between-method* variance, the ingredients are not reshaping the optimization problem at
all — the strongest possible statement that the wall is a property of the PDE, not the
method.
→ Artifact: all runs' landscape descriptors + resampled trajectories.
→ Rule: pooled one-way η²(method) vs η²(seed) + silhouette scores under each labeling
   (`similarity/variance_decomposition.csv`); η²_seed ≥ η²_method ⇒ method-irrelevance.

**H9 — parameter-space laziness.** Within a seed, every plain-FNN method starts at the SAME
weights. If final weights cluster by *seed* rather than by *method* (PCA/t-SNE of final
checkpoints), the optimizer never leaves the init's basin regardless of ingredient — the
methods choose *where in the shared basin* to settle, not *which basin*.
→ Artifact: `checkpoints/model-*.pt` (last), FNN runs only.
→ Rule: silhouette(seed) > silhouette(method) in `*_weights_embedding.pdf`.

**H10 — solution-space collapse.** If all methods' *predicted fields* form essentially one
cluster per PDE (near the trivial branch/point), then the 16 different optimization
processes are converging to the *same wrong answer* — the trivial attractor is the unique
reachable optimum, and origin is not "worse than best practice" so much as "identical to
it in outcome".
→ Artifact: `solution/fields.npz` predictions on the common reference grid.
→ Rule: t-SNE/PCA of predicted fields shows no method separation beyond the diverged
   W-combos; within-cluster spread ≥ between-method spread.

### Round-3 verdicts (96-cell sweep, `similarity/variance_decomposition.csv`)

| space | η²(method) | η²(seed) | silhouette(method / seed) | verdict |
|---|---|---|---|---|
| KS landscape   | 0.87 | 0.008 | +0.13 / −0.05 | method reshapes the landscape |
| KS trajectory  | 0.94 | 0.013 | +0.28 / −0.06 | method reshapes the path |
| KS solution    | 0.56 | 0.025 | −0.01 / −0.06 | graded field differences, no clusters |
| KS weights     | 0.12 | **0.63** | −0.21 / **+0.45** | **clusters by seed = init basin** |
| GS landscape   | 0.82 | 0.011 | −0.08 / −0.05 | method reshapes the landscape |
| GS trajectory  | 0.82 | 0.021 | −0.03 / −0.05 | method reshapes the path |
| GS solution    | 0.45 | 0.038 | −0.22 / −0.08 | graded, no clusters |
| GS weights     | 0.07 | **0.80** | −0.26 / **+0.63** | **clusters by seed = init basin** |

- **H9 confirmed strongly**: final weights cluster by *seed*, not method — no ingredient
  ever escapes the shared init's basin; methods only choose where *within* it to settle.
- **H8 rejected in its strong form** for trajectory/landscape space: the ingredients DO
  produce genuinely different optimization processes (~90% of feature variance is
  method-driven)…
- **H10 (the synthesis)**: …but the reshaping never changes the outcome class. Solution
  differences are graded (η² ≈ 0.5 even excluding the diverged W-combos) with *negative*
  method-silhouettes and equal final errors (KS 0.91–0.96, GS trivial). **The ingredients
  change how the run travels, not where it can arrive** — parameter space is init-locked,
  outcome space is horizon/attractor-locked, and that is precisely why `origin` and the
  full best-practice stack are indistinguishable on chaotic PDEs at this scale.
