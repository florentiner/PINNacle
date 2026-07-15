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

### H6/H7/H10 measured verdicts (`test_hypotheses.py` on the 96-cell sweep)

- **H6 CONFIRMED (horizon, not asymptote).** KS tracked horizon t\* (first time band with
  rel-L2 > 0.5): origin **0.30**, best_practice 0.23 — same within one band, and origin is
  *tied for the best horizon of all 16 combos* (best: ablation_A, t\*=0.30). Several combos
  are *worse* (G, WG, WAG, CWA: t\*=0.10). No ingredient extends predictability; some damage
  the early fit. GS analog (H6b): origin fits the ~98% background to 0.128 and misses the
  pattern region at 1.13; best_practice: background 0.039, pattern 1.24 — same shape of
  failure, only the *extent* of the well-fit region matters.
- **H7 CONFIRMED (mechanisms).** origin = "stable partial minimum" on both PDEs (KS
  1.04→0.909, late slope −0.007); the diverging marching combos = "compounding" (CW
  1.04→1.84, slope +0.79; CWA →2.02) — ends *worse than the shared init*.
- **H9 CONFIRMED quantitatively.** Weight-space travel from init is only 0.19× (GS) / 0.42×
  (KS) of the between-seed init separation; within-seed final spread is 0.26× / 0.47× of the
  between-seed spread. Every method stays in the shared init's neighborhood.
- **H10 CONFIRMED for KS, PARTIAL for GS.** KS: mutual method-distance 0.24 vs
  distance-to-reference 0.94 (ratio 0.25 — all methods share the same wrong answer; late-time
  amplitude 0.19 of reference = the decayed trivial branch). GS: ratio 0.79 — but both
  distances are small (0.10 vs 0.13) because ~98% of the reference *is* the trivial
  background all methods reproduce (late-time amplitude 0.96); the entire failure is
  concentrated in the 1.7% pattern region (H6b), where every method scores rel-L2 ≥ 1.1.

---

## Round 5 — when/why SHOULD best practice beat origin? (scale, precision, parameterization)

Round 2–4 established that at benchmark scale (100×5 net, 30k iterations, float32) no
ingredient beats origin on KS, because origin already sits at the predictability-horizon
wall and the stack pays fixed-budget overhead (window starvation, grad-norm cold start).
The papers' successful chaotic runs differ from our controlled sweep in exactly three
resources: **network scale**, **per-window compute**, and (implicitly, through the
achievable residual) **arithmetic precision**. Round 5 turns each into a falsifiable
hypothesis; `run_scale_showdown.sh` and `run_additional_experiments.sh` run them.

**H11 — scale unlock.** The stack's ingredients are *complements at scale, liabilities at
starvation*: with a width-256 modified MLP and 150k iterations (15k per window — the
papers' regime), the ordering flips to `ablation_all < ablation_A < origin` (lower = better)
on KS, because marching stops starving (each window gets 5× the whole benchmark budget) and
grad-norm's cold start amortizes.
→ Test: script 1 step 2 (origin / ablation_A / ablation_all @ 256\*4, 150k, warmup 5000,
3 seeds). Falsified if origin still ties or wins at that scale.
→ Corollary (H-KS-2, script 1 step 1): at the OLD 30k budget, error rises monotonically
with window count (W2 < W5 < W10), pinning the Round-4 starvation chain causally.

**H12 — precision floor.** The horizon law t\* ≈ ln(1/ε)/λ says the tracked horizon extends
only if the achievable field error ε drops. If float32 arithmetic (not optimization) is
what floors ε, float64 extends t\* measurably at the same budget; if the floor is
optimization-set (loss stalls at 6e-3 far above float32's ~1e-7 capability), float64
changes nothing — which would prove the wall is optimization-hard, not arithmetic-hard.
→ Test: script 2 block (b) — origin / A / all on KS, 3 seeds, `--float64`.
→ Read-out: compare per-time-band error and t\* between float32 (existing) and float64 runs.

**H13 — A's advantage is real.** The modified MLP's consistent ~0.5% edge over origin
(3/3 seeds, 100% of late epochs, but p=0.125 at n=3) survives 18 paired seeds at p<0.05
(power analysis: d_z=0.66 ⇒ n≈18 for 80% power).
→ Test: script 2 block (a) — seeds 1237–1251 of ablation_none + ablation_A; pool with the
existing 1234–1236; paired t + Wilcoxon.

**H14 — budget-dependent synergy sign-flip.** The G×W interaction measured at 30k
(G hurts −W: +0.024; G rescues +W: −0.150) flips toward genuine complementarity at 150k:
in script-1-step-2 data, `all` beats `A` (G and W each *add* value on top of the
architecture) — i.e., the same interaction term changes sign with per-window budget.
→ Test: recompute the stratified G/W effects on the scale runs vs the benchmark runs.

**H15 — RWF (Random Weight Factorization).** The Expert's Guide's remaining untested
recommendation (W = diag(exp(s))·V, s∼N(1, 0.1); identical initial function, re-conditioned
parameterization) improves the modified MLP at fixed budget: err(A+RWF) ≤ err(A).
→ Test: script 2 block (c) — ablation_A / ablation_all with `--rwf`, 3 seeds.

**Interpretation guide.** H11+H14 confirmed ⇒ "best practice beats origin *given the
resources it was designed for*; the benchmark-scale loss was an artifact of budget parity."
H12 confirmed ⇒ precision is a first-class ingredient the literature under-reports.
All falsified ⇒ the horizon wall binds even at paper scale for this KS parameterization,
and the gradient-free route (Frozen-PINN, 3.2e-5) is not just convenient but necessary.

---

## Round 6 — can the PINN-PELINE agent close the origin→best-practice gap? (oracle experiments, no agent integration)

Context: the PINN-PELINE paper (this repo's RL pipeline) trains a DQN to build optimizer
chains — actions = (optimizer ∈ {Adam, L-BFGS, PSO}, lr, stage length) — from 4-channel
autoencoder loss-landscape states, reward = per-stage reduction in RMSE vs the reference
(with residual/loss-based rewards named as the next step). Its own PINNacle results hit
exactly the walls this study identified: **GS 9.35e-2 = the trivial-attractor value; KS
9.46e-1 = the horizon wall** ("on long-time and chaotic problems, both methods yield
L2RE ≈ 1"). The question: operating on origin(+the well-posedness fixes), which parts of
the best-practice gap can an *adaptive scheduler* recover — given that Round 4 proved the
stack's losses are precisely *scheduling artifacts* (window starvation, grad-norm cold
start, fixed ε-phases), i.e., the kind of thing an adaptive policy exists to avoid?

**Key device: the chain oracle.** Any single-episode policy the DQN can express IS a fixed
optimizer chain, so the best fixed chain from the paper's action space, found by grid,
**upper-bounds the trained agent** at equal budget — no integration needed. (`--method
chain --chain "adam:1e-3:0.5,lbfgs:1.0:0.5"` runs one; `run_agent_gap_experiments.sh`
runs the grid.)

**H16 — KS: no headroom for the agent in final error, real headroom in cost.** No chain in
the action space beats origin's 0.918 wall (the oracle-best chain lands within noise of
it), because the wall is set by the PDE, not the schedule. The agent's realizable value on
KS is *efficiency*: some chains reach wall-level error in far fewer iterations than others.
→ Test: E1 chain grid (12 chains × 3 seeds, KS); compare best-chain final rel-L2 vs origin,
and iterations-to-0.93 across chains (from `metrics_history.csv`).
→ Falsified if some chain lands materially below 0.91 — which would ALSO falsify part of
the Round-2 horizon story, so this doubles as a robustness check.

**H17 — switch-timing carries real signal (the agent's core premise).** Chains with the
same components but different switch points (Adam→L-BFGS at 25/50/75/90%) differ in final
error and in stability by more than seed noise — i.e., WHEN to switch matters, which is
exactly the decision a state-conditioned policy can make and a fixed recipe cannot.
→ Test: E2 = the switch-fraction subset of E1; variance across split points vs across seeds.
→ If timing variance ≈ seed noise, the agent's per-state switching adds nothing over a
tuned fixed chain (the paper's gains would then come only from per-problem chain selection).

**H18 — GS: the agent's reinit action recovers the stack's insurance for free.** The entire
GS best-practice gap is seed-insurance against the overshoot attractor (Round 4). The
PELINE episode loop already contains the needed action: reinit/abandon a failing
trajectory. Best-of-k restarts of plain origin should recover stack-level reliability
(0.177 → ≈0.094) at ~k× cost with zero stack ingredients — and the wrong-attractor capture
is *visible early*: origin@1236's error curve was already stuck at 0.29 by epoch 5k, so a
policy could cut losses at ~1/6 of the budget, making the insurance ≈1.3×, not 3×.
→ Test: E3 = 9 extra GS origin seeds → bootstrap best-of-k (k = 1..3) statistics; early-
detectability check: does the epoch-5k error separate eventual-good from eventual-bad seeds
across all 12 seeds?

**H19 — loss-as-reward is viable on chaotic ONLY with the fixes + the multichannel state.**
The paper's proposed residual-based reward works where loss↔error correlation is high.
Post-periodicity-fix that holds on KS (corr +0.9), so Δloss reward ranks chain stages like
ΔRMSE does; but on GS the *scalar* total loss cannot distinguish the trivial attractor
(every method plateaus at 0.9628) — while the state's separate L_oper/L_bnd channels can
(the overshoot branch has a distinct component signature). Prediction: per-stage
corr(Δloss_total, ΔRMSE) is high on KS and degenerate on GS, but adding the loss
*components* (the state channels) restores discriminability.
→ Test: E4 = offline analysis of E1/E3 runs' `loss_history.csv` (per-component) +
`metrics_history.csv`; no new runs needed beyond E1/E3.

**Interpretation guide.** H16+H17+H18 confirmed ⇒ the agent can deliver best practice's
*actual* benefits over origin (reliability, efficiency, per-problem scheduling) without its
fixed-schedule costs — "agent + origin(+fixes) ≈ best practice, cheaper"; the final-error
wall stays where physics puts it. H19 confirmed ⇒ the paper's ground-truth-free reward is
workable on chaotic PDEs *because of* the well-posedness fixes and the multichannel state —
a concrete, evidence-backed design recommendation for the agent's next version.
