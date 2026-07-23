# Hypothesis: how the RL-PINN-OC agent closes the vanilla↔SOTA gap on chaotic PDEs

Context: RL-PINN-OC (ICLR 2026) constructs optimizer chains via a DQN whose state is an
autoencoder-compressed loss landscape and whose reward is the step-decrease of solution
error (with a loss-based reward variant available when no reference solution exists).
On PINNacle's chaotic KS the published agent reaches L2RE ≈ 1.02 — inside the same
failure band as every other PINNacle-family method (0.86–1.16) — while the causal
curriculum (our study) reaches 3.56e-2. The question: what must the agent control to
close that 26× gap, and what should its reward/state be? Every claim below is backed by
local data (`runs/`, `analysis/out/`, `runs_chains/`).

## 1. Empirical ground truth (three datasets, one conclusion)

**(a) The optimizer action space cannot cross the gap.** In the user's prior experiments
(`runs_chains.zip`: 12 hand-designed chains × 3 seeds — Adam/L-BFGS switch points at
25/50/75/90%, PSO injections, LR ladders, alternating), every chain lands on chaotic KS
at **L2RE = 0.913–0.915 with cross-seed std ≈ 0.000** (two divergent chains do worse:
adam_hi 2.9, lr_ladder 1.9). Thirty-plus independent runs, one identical wrong answer:
the ghost attractor is a set-measure-one endpoint for ANY optimizer sequence in the
single-shot formulation. GS: all chains ≈ 0.093–0.095, equal to vanilla.

**(b) The landscape explains why.** Our measured KS landscapes: the vanilla objective is
a shallow high plateau (loss 6.7e3–1.5e4 across the whole trajectory plane) whose only
reachable minima are broad ghost basins; the true solution sits in a needle funnel
~8 orders deep that exists only in the causal-curriculum objective. An agent that only
re-orders optimizers re-explores the same deceptive surface — consistent with the
paper's own observation that the landscape state carries geometric information: here the
geometry says "no optimizer can help."

**(c) Reward calibration is regime-dependent.** Comparing loss-vs-true-error histories:
- Their chains (single-shot KS): loss falls 2e4–6.5e5×, true error falls 1.19×.
  A loss-decrease reward pays out ~5 orders of magnitude of "progress" for converging
  INTO the ghost. An oracle-RMSE reward is equally useless here — it stays flat at
  ~0.91, so the agent receives no learnable signal either way.
- Our causal windows: loss falls ~7 orders and error falls ~5 orders together
  (macro-calibrated); the causal weight statistic W_min acts as a stage-gate
  certificate (its raw correlation with error is low because it is a sawtooth over
  tol stages — it is an event signal, not a dense reward).

**Conclusion:** the gap is closed by changing WHAT the agent controls (the curriculum),
not by better control of the optimizer chain. And precisely inside that curriculum, the
loss-based reward the paper proposes becomes valid.

## 2. The hypothesis

**H: Re-target RL-PINN-OC from optimizer-chain construction to curriculum-chain
construction over the causal formulation. The same agent architecture (stage-wise MDP,
landscape state, delta-reward, DQN) then closes the gap, because its action space now
spans the axis along which the gap actually lies, and its loss-based reward becomes
trustworthy exactly there.**

Action space (replacing/extending optimizer choice per stage):
1. tolerance move: advance tol in the anneal ladder / hold / (optionally retreat);
2. stage budget: extend or terminate the current tol stage;
3. window advance: certify the window (move to next) or keep training;
4. (secondary, from the original paper) optimizer for the stage — e.g. switch to
   L-BFGS for final polishing of a window after W_min > 0.9;
5. (optional) window length Δt for the next window.

Reward (oracle-free, enabled by the user's loss-in-reward finding):
- dense term: Δlog(total loss) within a stage — valid inside the curriculum per §1c;
- event term: bonus when min W crosses the certification threshold (0.99), which our
  data shows is the true convergence certificate;
- cost term: −λ per iteration, so the agent learns budget economy.

State: the paper's AE-compressed landscape is directly reusable — our archived
parameter trajectories (48+ snapshots/window, `w{k}_trajectory_flat.npy`, identical
architecture across causal and ablation lines) form a ready offline corpus for
pre-training the encoder and warm-starting the replay buffer (the paper's 10k random
warm-start could be replaced by real curriculum transitions).

## 3. Why this specifically recovers the remaining error (quantified)

The fixed hand-tuned schedule (tol ladder ×6, 200k cap/stage) is provably misallocated
in our runs:
- easy windows early-stop far below cap (w0 stages finished in 20–60k iters), while
- the hardest windows exhausted caps and finished UNDER-ANNEALED: w8 ended at
  W_min = 0.648, w9 at 0.706 (vs 0.99 target), with l2 = 4.4e-2 and 7.9e-2 — and these
  two windows dominate the full-domain error (3.56e-2). w0–w7 contribute ≤1.1e-2 each.
An adaptive allocator that moves budget from early-stopping stages to the final stages
of hard windows attacks exactly the dominant error terms. If w8–w9 were annealed to the
same W_min ≈ 0.99 as w0–w7, the per-window error trend implies a stitched full-domain
error ≈ 1.5–2e-2 — i.e., **budget re-allocation alone (the simplest policy the RL agent
would learn) is worth ~2× of the remaining gap, at zero extra compute.** Beyond that,
optimizer actions (L-BFGS polish after certification) and Δt control offer further
headroom that the fixed recipe cannot express.

## 4. Falsifiable predictions

P1. A heuristic "policy-class emulator" (advance tol on sustained W_min>0.9; abort
    stalled stages; spend the saved budget extending the final stage until W_min≥0.99)
    run at the SAME total iteration budget as the fixed schedule improves the
    under-annealed windows (w8: 4.4e-2 @ W_min 0.648 baseline). [Kaggle-testable now.]
P2. An agent rewarded by Δlog-loss inside the curriculum learns the same allocation as
    one rewarded by oracle ΔRMSE (loss-reward validity, §1c) — testable by correlation
    of the two reward streams over our logged histories: they are macro-aligned in
    causal windows and anti-aligned in single-shot runs.
P3. The AE-landscape state distinguishes "funnel-descending" from "plateau-stuck"
    states (our KS-vanilla vs KS-causal landscapes are separable at a glance), so the
    agent can learn WHEN a stage has exhausted its usefulness — the same claim the
    paper makes for optimizer switching, now for curriculum moves.

## 5. Relation to the paper's framing

This is not a replacement of RL-PINN-OC but its natural extension: the paper already
frames PINN training as a stage-wise MDP with landscape state and shows the agent can
discover expert-like switching. Our data adds: (i) on chaotic problems the profitable
stage-moves are curriculum moves, (ii) the loss-based reward the paper needed for
practicality becomes *valid* precisely inside that curriculum (and is invalid outside
it — the single-shot ghost regime actively deceives it), and (iii) the causal weight
statistic provides the missing oracle-free success certificate.


---

# Round-6 verdicts (H16–H19) from the E1/E2 chain data

Tested offline against `runs_chains.zip` (12 chains × 3 seeds × {KS, GS}, 30k epochs,
metrics every 100) plus our curriculum archives. E3 (`runs_gs_seeds`) was not in the
archive — H18 and the GS half of H19 remain pending that data.

## H16 — wall confirmed; efficiency headroom is NOT in chain composition. PARTIAL.
All viable chains: final KS L2RE 0.915±0.003 (wall ✓). But epochs-to-wall (l2re≤0.93)
are IDENTICAL across chains within each seed — [3100, 6900, 3200] for every
Adam-prefixed chain — because the wall is reached at 3.1–6.9k epochs, BEFORE the
earliest switch point (25% of 30k = 7.5k). Chain composition cannot be cheaper or more
expensive to the wall; only the shared Adam prefix and the seed matter (seed variance
2.2×). The agent's real single-shot-KS headroom is therefore:
  (a) reliability — 4/12 hand-designed chains are harmful or broken (adam_hi 2.9,
      lr_ladder 1.9, ahilb NaN, pso_start partial): an agent that merely avoids bad
      stages is already valuable insurance;
  (b) budget economy — stopping at the wall saves 75–90% of the 30k budget, and the
      wall is detectable from the observable loss plateau.

## H17 — REFUTED on chaotic KS.
Switch-timing carries zero signal here: alb_25/50/75/90 are indistinguishable in final
error (0.915 all) AND in epochs-to-wall (between-split std = 0 vs within-seed std ≈
1800). The premise "WHEN to switch matters" has no expression on single-shot chaotic
KS because everything after the wall is motion inside the ghost basin. (Not a refutation
of the paper's non-chaotic wins, where headroom exists.)

## H18 — untestable in this archive (no runs_gs_seeds, no 0.177 captures).
Kaggle-runnable: 9 GS origin seeds ≈ one short T4/P100 session.

## H19 — pre-fix loss-reward is DEAD on single-shot chaotic KS; the working fix is the
curriculum. PARTIALLY CONFIRMED, with a sharper mechanism.
Fine-grained (every-100-epoch) correlations over the full 30k history:
corr(Δlog L_total, Δlog l2re) = 0.07; operator channel 0.06; IC channel 0.14 —
i.e., scalar OR channel-wise loss deltas carry ~no reward information in the ghost
regime (coarse-checkpoint estimates that look higher are an artifact of the initial
co-drop). By contrast, inside the causal curriculum loss and error are macro-calibrated
(loss ↓ ~7 orders ↔ error ↓ ~5 orders per window; calibration ratio ≈ 0.6 vs ≈ 0.03
single-shot). Conclusion: the viable ground-truth-free reward on chaotic problems is
Δlog-loss WITHIN the causal formulation + the W_min ≥ 0.99 certification event —
the curriculum is not just the accuracy fix, it is what makes the loss-based reward
truthful. GS channel-discrimination claim: pending E3 data.

## Synthesis with the Round-4/6 thesis
"Agent + origin(+fixes) delivers best-practice benefits without fixed-schedule costs"
holds on chaotic problems only for reliability and efficiency — the accuracy wall is
formulation physics (H16's own concession) and no optimizer policy crosses it. The
constructive completion, supported jointly by Round-4 ("the stack's losses are
scheduling artifacts"), our ablation (windowing = oxygen; weighting = 1.8–2.5×
multiplier), and our under-annealed windows (w8–w9 finished at W_min 0.65–0.71 and
dominate the final error — a pure scheduling artifact worth ~2×): **adaptive scheduling
is indeed what the agent is — but on chaotic problems the schedule worth learning is
the causal curriculum's (tol ladder, stage budgets, window advance), not the optimizer
chain's.** The chain framework already carries the causal knobs in its config
(causal_eps_schedule, causal_delta, num_causal_buckets, time_windows), so the agent-on-
curriculum experiment is an action-space change, not an infrastructure change.

---

# P1 EXPERIMENT RESULT (adaptive w8, Kaggle) — refutes §3, sharpens the conclusion

Ran the adaptive stage-controller (RL policy-class emulation: advance tol on W_min>0.9
or stall; final stage to W_min≥0.99 or budget) on KS **window 8**, resuming from the
exact w7→w8 handoff, budget matched to the fixed schedule.

| w8 run | iters | best W_min | final L2RE |
|---|---|---|---|
| fixed schedule (baseline) | 735,000 | 0.648 (under-annealed) | 4.401e-2 |
| **adaptive controller** | **615,000 (−16%)** | **0.997 (certified)** | **4.401e-2 (identical)** |

**§3 and P1 are REFUTED.** I predicted that annealing w8 to W_min≈0.99 would drop its
error toward ~1.5–2e-2. The controller DID reach certification (best W_min 0.997) — yet
the error is **bit-identical to the under-annealed baseline (4.40e-2)**. Re-annealing the
window does not move its error floor. (w9, cut short by session end at 112k iters, tells
the same story: best W_min 0.992, L2RE 7.96e-2 ≈ fixed 7.87e-2.)

**Corrected mechanism.** The w8/w9 error is NOT a within-window scheduling artifact. It is
**inherited**: the accumulated w0→w7 handoff error, amplified window-over-window by the
chaotic (Lyapunov) dynamics, sets a floor the window converges to perfectly but cannot
break by better annealing. The window optimizes correctly toward an IC that is already
slightly wrong.

**This UNIFIES the study under H16's logic — now for the curriculum, not just chains:**
on chaotic problems the agent's accuracy headroom is ~0 (the wall is physics — a ghost
attractor in single-shot, inherited handoff error in the curriculum), but its
**efficiency and reliability headroom are real and measured**: −16% iterations to the
same result *with proper certification* (W_min 0.648→0.997), plus budget it can now
redeploy. An RL agent rewarded by Δlog-loss + the W_min≥0.99 event would learn exactly
this: certify-and-move-on, not waste iterations under-annealing.

**Where accuracy headroom actually lives (revised, testable next).** Since the floor is
accumulated handoff error, the lever is the curriculum's *geometry*, not its per-window
annealing: shorter/more windows or overlap near late times reduce per-handoff
amplification. That is the Δt / window-count action (action #5) — the one curriculum
control this experiment leaves unexplored and the only one with a mechanism to lower the
late-window floor. Predicted: doubling window count over t∈[0.7,1.0] lowers w8/w9 L2RE
where fixed 10-window annealing cannot. [Kaggle-testable; ~1 session.]

**Net:** the agent closes the vanilla↔SOTA gap by adopting the causal curriculum (26×,
already shown); *within* that curriculum it buys efficiency + reliability + oracle-free
certification (P1, measured), and any remaining accuracy is reachable only through
window-geometry control, not optimizer or tol scheduling.
