# Why causal training beats the original PINN on chaotic PDEs — and what an RL agent can (and cannot) add

**Comparative study on PINNacle's chaotic cases: Kuramoto–Sivashinsky (KS) and Gray–Scott (GS).**
Methods: *original* = vanilla DeepXDE PINN (PINNacle defaults); *SOTA* = the causal
training recipe (Wang, Sankaran & Perdikaris, CMAME 2024): **causal loss weighting +
time-marching windows + Fourier/multi-scale input features + gated "modified MLP"**.
All numbers below are from our own runs (seed 1234 main line; ~150 GPU-h across
baseline, full causal KS 10/10 windows, full causal GS 20/20 windows, a complete
10-window no-causal ablation, 36 optimizer-chain runs, and one adaptive-controller
experiment). Every figure has its underlying arrays archived under `runs/` and
`analysis/out/`.

---

## Part 1 — Why SOTA beats the original

### 1.1 The headline result

| | Original (vanilla) | SOTA (causal) | Factor |
|---|---|---|---|
| **KS**, full t∈[0,1] | L2RE = **1.007** | **3.56e-2** | **28×** |
| **GS**, full t∈[0,200] | L2RE = **0.094** | **1.42e-2** | **6.6×** |

The KS number matches the causal paper's reported 2.46e-2 (same order; ours is a
faithful port verified to reproduce the reference code numerically — bridged forward
agreement ~4e-7, window-0 L2 identical to 7 digits across reruns). The GS result is,
to our knowledge, the first causal-PINN solution of Gray–Scott.

On KS the word "beats" understates it: **the original does not produce a solution at
all.** Its per-time-slice relative error is O(1) from t≈0 onward — it fits the initial
condition hyperplane and never tracks the dynamics
(`analysis/out/ks-FINAL/error_growth.png`). The causal solution's error instead starts
at 6e-6 and grows along a clean exponential — the signature of a *correct* trajectory
whose error is amplified by chaos (Lyapunov growth), not of optimization failure.

![error landscapes](out/ks-FINAL/landscape_err.png)
*Error landscapes (log10|pred−ref|, shared scale). Vanilla: uniform O(1) failure.
Causal: 10⁻²…10⁻⁴, structure follows the physics. GS versions:
`out/gs-FINAL/landscape_err.png`.*

### 1.2 The failure mechanism: ghost minima

The decisive diagnostic is the **residual landscape**
(`out/ks-FINAL/landscape_resid.png`, `out/gs-FINAL/landscape_resid.png`): the vanilla
model achieves a *smooth, moderately low PDE residual everywhere* (10⁻²–10⁰ on KS,
~10⁻³ on GS) while being 100% wrong. Training minimized exactly what it was asked to
minimize — and converged to a **spurious "ghost" solution**: low residual, wrong
dynamics. Low residual is necessary but nowhere near sufficient; *the order in which
residual is minimized decides which solution you get.*

Quantified: across 36 independent single-shot chain runs, the training loss falls by
**2×10⁴–6.5×10⁵×** while true error falls by **1.19×** (§2.3, Fig. 4). The objective
is not merely unhelpful — it is actively deceptive.

### 1.3 The optimization geometry: plateau vs funnel

We computed loss landscapes with the **real training trajectories** overlaid
(Li-et-al-style; both PCA-plane and contour renderings):

- **Vanilla KS** (`out/ks-losslandscape-FINAL/loss_landscape_trajectory.png`, left
  panel; paper-style: `out/paper_style/ks_vanilla/`): the entire reachable
  landscape is a **shallow high plateau** — loss 6.7e3–1.5e4, barely a 2× range.
  The trajectory drifts around a gentle ridge and settles in a broad dent: converged,
  in the optimizer's sense, to the wrong attractor. There is no gradient path to the
  true solution from generic initialization.
- **Causal KS window** (same figure, right panel; `out/paper_style/ks_causal_w0/`):
  the true solution sits in a **needle-shaped funnel ~8 orders of magnitude deep**.
  The 48-snapshot trajectory shows the mechanism: huge early strides while the
  causally-relaxed objective is easy, then contraction into the funnel as the
  tolerance anneals. The funnel *exists only in the causal objective* — causal
  training does not search the vanilla landscape, it replaces it with a sequence of
  landscapes, each with a findable minimum adjacent to the previous one.
- **GS** (`out/paper_style/gs_causal_w5_wide/`): a wide flat basin (loss varies <10%
  across the trajectory plane) — GS is well-conditioned (2nd-order), which predicts
  and explains its training cost profile (§1.6).

### 1.4 Ingredient attribution (complete ablation, 10/10 windows)

![ingredients](report_figs/fig1_ingredient_decomposition.png)

| KS configuration | full-domain L2RE |
|---|---|
| vanilla (single-shot, plain FNN) | 1.007 |
| + time windows + Fourier features + modified MLP, **W≡1** | 8.61e-2 |
| + causal weighting (full SOTA) | 3.56e-2 |

- **Time-marching + architecture are the oxygen** (11.7× of the 28×): with Δt=0.1
  windows, hard IC anchoring and the Fourier/gated net, even uniform-weight training
  solves chaotic KS windows. Short windows are themselves a coarse causality
  mechanism.
- **Causal weighting is the sharpener** (further 2.4×): per-window it costs the
  ablation a stable ~1.8× penalty through mid-chaos that **widens to 2.0× (w8) and
  2.5× (w9) exactly where the dynamics are hardest** — and it supplies the
  convergence certificate (min W → 0.99) that uniform weighting cannot express.
- Encoding note (GS): the reference's 2D tensor-product Fourier encoding *lost* to a
  plain encoding at matched budget (9.6e-3 vs 3.0e-3 window-0) — GS's localized spots
  don't match global Fourier structure. Fourier features are load-bearing for KS
  (exact periodicity), not universal.

![per-window](report_figs/fig2_perwindow_floor.png)

### 1.5 The mechanism, observed directly

The causal weights W(t) form a **trust front** that sweeps each window from t=0 to its
end as training progresses (`out/ks-FINAL/causal_front.png`): six annealing cycles per
window, W_min collapsing at each tolerance increase and recovering to ~1 as that
stage's causality is satisfied. Combined with §1.3: the curriculum walks the optimizer
down the funnel one wall at a time — which is precisely why its loss signal stays
honest (§2.3) while the vanilla loss lies.

### 1.6 Conditioning corollary

Landscape geometry predicted the observed training cost before we could measure it:
KS (4th-order, stiff → funnel + plateau) escalated from 237k iterations (window 0) to
~1M (windows 7–9); GS (2nd-order, flat basin) held ~200k for all 20 windows. Same
recipe, different physics, cost curve follows the geometry.

### 1.7 Reproducibility notes (Part-1 provenance)

1. PINNacle's KS reference data is bit-identical to the causal authors' `ks_chaotic.mat`;
   an independent ETDRK4 integration reproduces it to 1e-12 — formulations verified.
2. The published reference's "multi-scale" time encoding is integer-power JAX
   arithmetic: negative exponents silently evaluate to 0, so trained networks actually
   see k_t = [0,0,0,1,10,100]. Our port replicates this exactly (window-0 L2
   2.3042e-5 ≡ 2.302e-5).
3. Full determinism: re-running window 0 reproduced its error to 7 digits.

---

## Part 2 — How the RL agent closes the gap (and which gap it can close)

Context: RL-PINN-OC (ICLR 2026) trains a DQN over autoencoder-compressed loss-landscape
states to build optimizer chains; reward = stepwise error decrease, with a loss-based
variant when no reference exists. On chaotic KS the published agent lands at L2RE ≈
1.02 — inside the failure band. The data collected here shows exactly why, and exactly
where the agent's real leverage is.

### 2.1 What the agent cannot do: cross the wall from inside the optimizer action space

![wall](report_figs/fig3_optimizer_wall.png)

Twelve hand-designed chains (Adam→L-BFGS at 25/50/75/90%, PSO injections, LR ladders,
alternating; 3 seeds each): **every viable chain lands at KS L2RE = 0.913–0.915 with
cross-seed std ≈ 0.000** — the identical ghost attractor; 4 of 12 chains actively
diverge or break. Moreover the wall is reached at 3.1–6.9k epochs — *before the
earliest switch point* — so switch-timing carries zero signal on chaotic KS (final
error AND epochs-to-wall are identical across all splits; H17 refuted). The landscape
(§1.3) says why: every optimizer explores the same deceptive plateau. **No policy over
{optimizer, hyperparameters, switch time} can cross a 26× gap that is formulation-bound.**

The corollary is the agent's first and largest lever:

> **L1 — adopt the curriculum.** The gap-closing action is a change of *what is
> scheduled*: the causal curriculum's controls (tolerance ladder, stage budgets,
> window advancement), not the optimizer chain. The agent's own framework already
> carries these knobs in its config (`causal_eps_schedule`, `causal_delta`,
> `num_causal_buckets`, `time_windows`) — this is an action-space change, not an
> infrastructure change. Formulation switch alone: 1.007 → 8.6e-2; with causal
> weighting: → 3.56e-2.

### 2.2 What the agent measurably adds inside the curriculum: efficiency, reliability, certification

We ran the agent's policy class directly (P1 experiment): an adaptive stage controller
— advance tolerance on sustained W_min>0.9, abort stalled stages, spend savings on
final-stage certification — replaying KS window 8 from the identical handoff state at
the identical budget:

| KS window 8 | iterations | best W_min | final L2RE |
|---|---|---|---|
| fixed hand-tuned schedule | 735,000 | 0.648 (under-annealed) | 4.401e-2 |
| **adaptive controller (RL policy class)** | **615,000 (−16%)** | **0.997 (certified)** | 4.401e-2 (identical) |

Two findings in one experiment:

- **Efficiency + certification are real**: same result, 16% cheaper, *with* the
  convergence certificate the fixed schedule failed to reach. Across a 10-window run
  the same policy recycles ~10–20% of total compute, and in single-shot settings the
  observable wall-detection saves 75–90% of budget (all post-wall iterations are
  waste). Add reliability: an agent that merely avoids the 4/12 harmful chains is
  already valuable insurance.
- **Accuracy headroom inside a window is ZERO** — the honest refutation of our own
  initial prediction. Certifying w8 did not move its error a digit, because the
  late-window error is **inherited**: accumulated w0→w7 handoff error amplified by
  chaos sets a floor that within-window scheduling cannot break (Fig. 2: w8/w9 sit on
  the Lyapunov growth line). This unifies everything under one law: *on chaotic
  problems the accuracy wall is physics* — ghost attractor in single-shot, inherited
  handoff error in the curriculum — *and the agent's real currencies are efficiency,
  reliability and certification.*

The one remaining accuracy lever the data points to is **window geometry** (shorter or
overlapping windows near late times reduce per-handoff amplification) — an action
(Δt / window count) that no fixed recipe explores and that fits the agent's stage-wise
MDP naturally. Untested; the only proposed mechanism that could lower the late-window
floor.

### 2.3 Reward design: when the loss can be trusted (the paper's loss-in-reward, made precise)

![reward calibration](report_figs/fig4_reward_calibration.png)

- **Single-shot (red)**: loss falls 4–6 orders of magnitude; true error falls 1.19×.
  Fine-grained corr(Δlog loss, Δlog error) = 0.06–0.14 — and per-component channels
  (operator/IC) do not rescue it. A loss-based reward *pays the agent for descending
  into the ghost*; an oracle-RMSE reward is equally unlearnable (flat at ~0.91).
- **Inside the causal curriculum (green)**: loss and error fall together (≈7 orders ↔
  ≈5 orders per window; calibration ratio ≈0.6 vs ≈0.03 single-shot). The curriculum
  is not only the accuracy fix — **it is what makes the ground-truth-free reward
  truthful.**
- Practical reward: dense Δlog-loss within stages + an **event bonus at W_min ≥ 0.99**
  (the oracle-free convergence certificate; note W_min is a sawtooth by design — an
  event signal, not a dense one) − λ·iterations (budget economy; P1 shows the policy
  this induces is exactly the profitable one).

### 2.4 State: the landscape representation transfers, and a free pretraining corpus exists

The paper's AE-compressed landscape state is the right observable here too: the
funnel-descending vs plateau-stuck regimes are separable at a glance (§1.3 figures),
which is precisely the distinction the agent needs for advance/hold decisions. Our
archives contain 150+ full parameter snapshots across causal, ablation and vanilla
runs on identical architectures (`trajectory/*_trajectory_flat.npy`) — a ready offline
corpus to pretrain the encoder and warm-start the replay buffer with *real* curriculum
transitions instead of the paper's 10k random ones.

### 2.5 Concrete integration (summary)

| Component | From the paper | Change |
|---|---|---|
| MDP structure | stage-wise decisions | keep |
| State | AE loss-landscape | keep; pretrain on our trajectory corpus |
| Action | optimizer × hyperparams | **curriculum controls**: tol move, stage budget, window advance, window Δt (+ optimizer as secondary, e.g. L-BFGS polish after certification) |
| Reward | ΔRMSE (oracle) / Δloss | **Δlog-loss (valid in-curriculum) + W_min≥0.99 event − λ·iters** |
| Termination | error ≤ ε, K_max, divergence | window certification; divergence guard unchanged |

Predictions status: P1 (budget reallocation improves accuracy) — **refuted, measured**;
P1′ (it buys efficiency + certification at equal accuracy) — **confirmed, −16% iters**;
P2 (loss-reward ≡ oracle-reward in-curriculum) — **confirmed at macro scale** (Fig. 4);
P3 (landscape state separates regimes) — supported by §1.3 geometry; window-geometry
accuracy lever — open, one-session testable.

### 2.6 Honest limits

An RL agent — any scheduler — cannot beat the physics: single-shot chaotic PINNs end
in ghosts regardless of policy, and inside the curriculum the late-window floor is set
by accumulated handoff error, addressable (if at all) only through window geometry.
What the agent delivers is everything *around* that limit: it discovers/schedules the
formulation that closes 26×, runs it 10–20% cheaper, certifies convergence without a
reference solution, and removes the manual tuning the fixed recipe demands — which is
the paper's thesis, now with the chaotic regime's boundary conditions mapped.

---

## Data & figure index

- Final runs: `runs/kaggle-causal-ks-session9/` (KS 10/10), `runs/kaggle-causal-gs-session7/`
  (GS 20/20), `runs/kaggle-causal-ks-ablation-s5/` (ablation 10/10),
  `runs/07.18-13.19.39-baseline-chaotic/` (vanilla both), `runs/kaggle-adaptive-w8/` (P1),
  `runs/kaggle-ks-w0-trajectory/` (48-snapshot trajectory line).
- Chain study: `runs_chains.zip` (12 chains × 3 seeds × {KS, GS}, 30k epochs).
- Comparison figures: `analysis/out/{ks,gs}-FINAL/`, loss landscapes
  `analysis/out/ks-losslandscape-*/`, `analysis/out/gs-losslandscape/`, paper-style
  contours `analysis/out/paper_style/`, report syntheses `analysis/report_figs/`.
- Documents: `analysis/FINDINGS.md` (study), `analysis/RL_AGENT_HYPOTHESIS.md`
  (hypothesis + H16–H19 verdicts + P1), `analysis/DR_FIGURE_IDEAS.md`.
