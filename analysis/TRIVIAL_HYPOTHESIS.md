# Trivial-attractor hypothesis: design and PDE selection

**Hypothesis (user's addition to the study):** the vanilla failure mode is convergence
to a *trivial solution* — an exact, zero-residual state of the PDE that ignores the
IC — and the SOTA (causal) machinery works because it makes that state unreachable.
Test plan: (1) pick a repo PDE with an exact trivial attractor, (2) drive vanilla into
it, (3) initialize the SOTA model *at* the trivial solution and causal-train — does it
escape? (4) landscape analysis of both, including whether the trivial pattern is
detectable in the vanilla objective *without* SOTA machinery.

## Prior evidence already in our data

- **GS vanilla (measured)** converged to exactly the trivial homogeneous steady state
  u≡1, v≡0 (an exact solution: all derivatives zero, reaction terms cancel at
  background) — the fig0_gs vanilla panel is literally the blank background, L2RE
  0.094 = "spots missing".
- **KS vanilla (measured)** collapsed to a near-flat field ≈ 0 after t≈0.35; u≡0 is an
  exact KS solution (every term carries u or its derivatives).

Both chaotic collapses are consistent with the hypothesis; the experiment below makes
it controlled and non-chaotic (shows the mechanism is general, not chaos-specific).

## Candidate table (time-dependent repo PDEs)

| PDE | exact trivial state | what trivial violates | vanilla difficulty | verdict |
|---|---|---|---|---|
| **Heat2D_LongTime** | **u≡0: residual ≤ 3e-16 (verified via repo class)** — the source `5·sin(k·u²)·f(x,y,t)` **self-gates**: sin(0)=0 kills the forcing | only the IC (BC=0 satisfied) | t∈[0,100], PINNacle-class long-time failure; ref alive forever (RMS 1.07, amplitude 1.8↔0.16, period-8 forcing) | **PRIMARY** |
| Wave1D | u≡0 (homogeneous eq) | only Dirichlet IC (Neumann IC u_t=0 *satisfied*) | partial failure at T=1 | backup (cheapest; would need horizon extension for full collapse) |
| KS / GS (chaotic) | u≡0 / (1,0) | IC | measured collapse | already done (main study) |
| Burgers1D/2D | u≡0 | IC | 1D works fine vanilla | weak candidate |
| NS2D_LongTime | zero flow | inlet BC forcing | — | forcing enters via BC, not self-gated; unclear trivial |
| Wave2D_LongTime | — | — | — | **EXCLUDED: repo inconsistency found** (below) |

### Repo finding: Wave2D_LongTime ref/PDE mismatch

`ref_sol` of `Wave2D_LongTime` does **not** solve the coded PDE `u_tt = u_xx + 2·u_yy`:
its residual is exactly `−2π²·u` (verified analytically and by finite differences;
both sinh-mode terms give p² = m² − a²n² − 2 mismatch). The reference actually solves
the Klein–Gordon-type equation `u_tt = u_xx + 2u_yy − 2π²u`. Any benchmark on this
case measures distance to a non-solution; excluded from our experiment.

## Why Heat2D_LongTime is the ideal testbed

1. **The trivial state is exact and mechanistically interesting**: the nonlinearity
   `sin(k·u²)` lets the network *switch the physics off* — at u≡0 the forcing
   disappears and the PDE residual is identically zero everywhere. The only loss
   component resisting collapse is the IC (`u(x,y,0)=sin(4πx)sin(3πy)`, verified to
   7e-4 against the ref grid).
2. **The true solution never decays** — periodically forced (period 8), amplitude
   oscillates 0.16↔1.8 through t=100 — so "trivial" is wrong at (almost) all times:
   L2RE of the trivial state is ≈1.0, cleanly separated from any partial solution.
3. **Stock repo configuration** (no domain edits), data reference shipped in
   `ref/heat_longtime.dat` (16×12 spatial grid × 501 time steps, verified).
4. **Cheapest causal port**: first-order in time, 2D space — the existing GS JAX
   engine (jet second derivatives, 20-window marching, checkpoint/resume) transfers
   with only the residual and forcing changed; window handoff is u-only.
5. Non-chaotic → separates the trivial-attractor mechanism from chaos.

## Experiment design

| run | init | engine | budget | question |
|---|---|---|---|---|
| **V** vanilla | random | DeepXDE torch, T4, 20k iters + forensic callback | ~1–2 h | does it collapse to u≈0? (step 2) |
| **C-rand** causal | random | JAX P100, windows Δt=5, causal W, ≤4 windows | ~1 session | control: SOTA behavior from generic init |
| **C-triv** causal | **distilled to u≡0** | same as C-rand | ~1 session | does causal training *escape* the trivial attractor it was placed in? (step 3) |

Prediction from the mechanism (§1.4 of REPORT): at u≡0 the causal IC gate sees
L_IC ≈ 0.25 (IC RMS² of sin·sin/4) → 10⁴·L_IC ≈ 2.5e3 → all W≈0 → the *only* gradient
is the IC term → the optimizer is forced off the trivial point along the IC direction,
then the front sweeps the window. In the vanilla objective the same point is a
near-stationary ghost (PDE residual exactly 0 on all collocation points; only the
IC/BC terms pull, diluted by loss averaging). If C-triv escapes and matches C-rand's
window error, the "SOTA removes the trivial minimum" claim is demonstrated *causally*,
not just correlationally.

Landscape deliverables (step 4): vanilla loss landscape around the collapsed state
(trivial basin geometry) vs causal landscape around the same point (gate wall);
trajectory overlays for V, C-rand, C-triv; component-space signature of the trivial
state in the *vanilla* objective (L_PDE ≈ 0 while L_IC ≫ 0 — a detectable pattern that
needs no reference solution) and a test of a no-SOTA avoidance heuristic based on it.

---

## RESULT — step 2 (vanilla collapse): CONFIRMED (2026-08-10, kaggle-trivial-vanilla)

20k iters, T4, stock config. Final **L2RE = 0.9988**; **||pred||/||ref|| = 0.031** (33x
closer to the trivial zero than to the solution). Amplitude: 0.83 at t=0 (IC patch,
corr 0.95 with the IC mode) -> 0.36 at t=0.2 -> 0.09 at t=5 -> machine zero beyond
t~25. The mechanism, measured:

- **All loss components converge**: PDE 2e-4, IC 2e-4, BC 1e-5 — the vanilla
  objective reports "solved" at L2RE 0.999. A ghost with a certificate of convergence.
- **91.6% of the total squared residual is concentrated in t<=1 — 1% of the domain.**
  Outside the thin layer the self-gated source is off and the residual is exactly 0.
  The uniform-in-time loss happily pays an O(1) residual on a measure-1% sliver to
  buy zero residual on the other 99% — the trivial branch wins by construction.
- This is precisely the trade causal weighting forbids: with W_i gated by cumulative
  early-time residual, the boundary layer is the ONLY thing that matters until it is
  resolved (prediction for step 3).

---

## Step 4 pre-registration: no-SOTA avoidance via error-space patterns (2026-08-10)

**Detector (reference-free, validated offline)**: C_enrich (early-time residual
enrichment; >3 = causality-violating layer) + A_late (late dynamics alive; <0.1 =
frozen). Separation on saved runs: heat-vanilla (20.0, 0.000: both flags), KS-vanilla
(4.3, 0.70: front flag), GS-vanilla (1.9, 0.005: dead flag), KS-causal control
(0.012, 1.65: clean).

**Intervention (vanilla toolbox only — no causal weights, no windows, no arch
change)**: TrivialGuardCallback, mode "rar" — when flagged, replace 50% of PDE
collocation points with pool points sampled proportional to squared residual (the
measured mechanism is a *cheap* thin layer: 91.6% of squared residual in 1% of the
domain; concentrating points there makes the trivial trade expensive). Control mode
"uniform" resamples uniformly on the same schedule.

Experiment matrix (T4, 20k iters, seed 1234): {heatlt, ks} x {rar, uniform}.

Pre-registered hypotheses:
- H-T1 (heat, rar): the guard prevents full collapse — L2RE substantially below the
  vanilla 0.9988 and/or the collapse front (amplitude decay point) is pushed to
  later t; possible self-organized marching (front moves -> residual moves -> samples
  follow).
- H-T2 (heat, uniform control): no material improvement over vanilla (the trade
  stays cheap under uniform points).
- H-T3 (ks, rar): escapes the trivial *flatness* (||pred|| rises toward O(||ref||),
  mid-time structure appears, front flag clears) but final L2RE likely remains poor
  (>0.5) — pattern-driven sampling defeats the trivial basin, not chaos itself.
- H-T4: detector separates all vanilla-trivial runs from causal runs (incl. the
  incoming C-rand / C-triv heat data).

---

## Separate experiment: StarSSE basin escape (arXiv:2303.03374 adaptation, 2026-08-11)

The paper (Sadrtdinov et al., NeurIPS 2023) studies ensembling from a pre-trained
checkpoint: cyclic-LR "kicks" of controlled size either keep children inside the
pre-train loss basin (linearly connected, soup-able) or push them out; leaving the
basin costs transfer quality. **Inverted-sign mapping to our problem**: the
vanilla-collapsed checkpoint sits in the *trivial* basin — a BAD basin — so their
escape machinery becomes a candidate no-SOTA cure: same loss, same architecture,
only LR-schedule moves in weight space.

Protocol (benchmark_sse.py; T4; per case {heat-LT, KS}): star of 10 children from the
collapsed ckpt (kick multipliers ×{1,2,4,8,32} of base LR 1e-3, 2 seeds each), one
cosine cycle 2500 iters per child; per child: reference-free detector signals,
oracle L2RE, ||pred||/||ref||, relative weight distance, and the paper's
linear-connectivity barrier to the trivial point (11-point path of the unweighted
vanilla loss); per kick: uniform soup of the 2 children.

Pre-registered hypotheses:
- H-S1 (basin picture transfers): escape is kick-controlled — small kicks give zero
  barrier to the trivial point (same basin, score stays ~1); only large kicks
  produce barriers.
- H-S2 (payoff): escaped children improve over the trivial state (norm_ratio rises
  from 0.03, front pushed later, L2RE < 0.999) but do NOT reach the true solution —
  random kicks find *neighboring* basins, and the nearest basins of the vanilla loss
  are other near-trivial states (front slightly later). Escape alone ≠ solution.
- H-S3 (soups): same-kick children in the trivial basin soup to a trivial state
  (flat basin); children that escaped to different basins produce broken soups
  (barrier between them) — soup quality is itself a basin diagnostic.
- H-S4 (selection): the reference-free trivial-score ranks children consistently
  with oracle L2RE — enabling escape-and-select without a reference solution.

---

## RESULT — step 4, KS guard pair (2026-08-11)

| run | L2RE | norm_ratio | final C_enrich | flags | guard actions |
|---|---|---|---|---|---|
| vanilla (July) | 1.007 | 0.357 | 4.3 | front | — |
| ks-rar (pattern-triggered) | 1.013 | 0.304 | **0.59** | **none** | **1** (step 3000) |
| ks-uni (uniform control) | 1.012 | 0.373 | 5.37 | front | 20/20 |

**H-T3 verdict: the optimistic half is REFUTED, and a detector blind spot is exposed.**
One pattern-triggered resample permanently removed the front-collapse signature
(C_enrich 3.3 -> 0.4-0.9 for the rest of training) — but accuracy did not move
(L2RE 1.013; zero time slices track the reference). On the chaotic plateau the
optimizer, taxed on the thin layer, does not propagate the solution — it finds
*another ghost* with uniformly spread residual, invisible to the front-collapse
signal (A_late also reads "alive" at 1.11 from the wash variance). The uniform
control shows the signature only stays visible when nobody optimizes against it.

Consequences: (1) on chaotic PDEs, error-space patterns are *gameable* — an
intervention (or an RL agent rewarded on these signals) can optimize the signature
away without touching the truth; any no-SOTA scheme needs signals that cannot be
redistributed (the causal certificate W_min is exactly such a signal — it is
anchored at the IC, not at a spatial pattern); (2) the trivial-signature detector
remains valid as a *diagnostic* (it correctly never reported recovery of accuracy —
no "solved" claim), but not as a sole optimization target.

---

## RESULT — StarSSE basin escape (2026-08-11)

Implementation note (honesty): children within a kick are exact clones — full-batch
deterministic training erased seed diversity (effective n=1 per kick; soup == child;
H-S3 untested). The kick axis, which carries the main question, is unaffected.

| kick | heat: barrier / loss / L2RE | KS: barrier / loss / L2RE |
|---|---|---|
| trivial ckpt | — / 7.72e-3 / 0.9993 | — / 3.06e-1 / 1.0068 |
| x1 | 0 / 7.74e-3 / 0.9994 | 0 / 2.74e-1 / 1.0141 |
| x2 | 0 / 1.06e-2 / 0.9990 | 4.4e-2 / 2.79e-1 / 1.0093 |
| x4 | 3.0e-2 / 7.76e-3 / 0.9986 | 5.7e-1 / 3.14e-1 / 0.9784 |
| x8 | **5.1e-1 / 5.88e-3 / 0.9987** | 6.3 / 3.24e-1 / 0.9765 |
| x32 | 16.2 / 2.55e-2 / 0.9981 | 38.9 / 3.78e-1 / 0.9751 |

- **H-S1 CONFIRMED**: escape is kick-controlled, monotone barrier growth — the
  paper's basin picture transfers to PINN weight space exactly.
- **H-S2: pessimistic half CONFIRMED**: escape != solution. The sharpest data point:
  heat kick x8 LEFT the trivial basin (barrier 0.51, weight distance 0.48) and landed
  in a *strictly deeper* minimum of the vanilla objective (loss 5.88e-3 < 7.72e-3) —
  with identical error (L2RE 0.9987). **The vanilla loss landscape is
  ghost-dominated: neighboring basins are equivalent trivial-class solutions, and
  kick-based escape is a random walk between ghosts.** On KS, big kicks even worsen
  the objective (0.27 -> 0.38) while cosmetically nudging L2RE (norm shrinks toward
  zero-field).
- **H-S4 (selection)**: moot at this resolution — all children score as ghosts by
  both the detector and the oracle; nothing to select.

Combined verdict of the two no-SOTA interventions (guard + SSE): error-space
patterns give reference-free DETECTION and basin DIAGNOSIS (barriers, escape
certification) — but no CURE: taxing the signature produces signature-free ghosts
(KS guard), and escaping the basin produces neighboring ghosts (SSE). The cure
requires changing what the objective rewards along time (the causal gate) — the
pending C-triv run tests exactly that from the same starting point.

---

## RESULT — step 4, heat guard pair (2026-08-11)

| run | L2RE | C_enrich | flags | collapse front (t-index of last alive slice /500) | guard actions |
|---|---|---|---|---|---|
| vanilla | 0.9993 | 20.0 | front+dead | 4 | — |
| heat-rar | 0.9984 | 19.85 | front+dead | **12 (3x push)** | 20/20 |
| heat-uni | 0.9992 | 20.0 | front+dead | 4 | 20/20 |

- **H-T1: materially REFUTED.** Pattern-driven resampling produces a real but
  microscopic propagation effect — the collapse front moves 3x later (t 0.8 -> 2.4
  of 100) — and then stalls; self-organized marching does not bootstrap. The
  optimizer prefers paying the concentrated-residual tax to propagating the
  solution. L2RE unchanged (0.9984 vs 0.9993).
- **H-T2: CONFIRMED** (uniform resampling: zero effect).
- **Detector honesty contrast (vs the KS pair):** on Heat-LT the signature CANNOT be
  gamed away — the trivial branch is exact, the residual has nowhere to
  redistribute, and the flags stay correctly on (C_enrich ~20 throughout). On the
  chaotic plateau the signature was removable without progress. Reference-free
  signals are honest exactly when the trivial state is an *isolated exact* branch,
  and gameable when ghosts form a continuum.

---

## RESULT — step 3, causal from trivial init, PLAIN encoding (2026-08-11, 11h P100 x2)

**The core escape question is answered: YES — and initialization does not matter.**
Both runs (C-rand from random init, C-triv distilled to u==0, window-0 l2 of init =
1.0000) behaved identically: the IC gate closed (loss_IC 0.243 -> 1.8e-7), the
prediction's amplitude is alive through the whole window (max|u| 0.85 -> 1.85 vs ref
1.77) — the trivial state is simply not a reachable attractor of the causal
objective. C-rand w0 l2 = 0.832, C-triv w0 l2 = 0.851 (same ceiling, same behavior).

**But the window did not certify** (W_min per stage [0.999, 0.955, 0.036, 0, 0, 0]):
the plain-encoded network holds the correct dominant modes (kx=2, ky~1.5 = the forced
and IC modes) at ~1/3 amplitude — spectral bias of the plain [t,x,y] encoding against
the high-frequency sin(4pi x)sin(3pi y) content. A fresh, measured row for the
bottleneck table of REPORT §1.4: *causality respected + representation mismatched =
escape without accuracy* (l2 0.83 vs trivial 1.0 vs target ~1e-2).

Fix per the synergy analysis: modal sine features (Dirichlet eigenbasis, k=1..6 per
axis). Relaunched both inits with `--encoding sine` (ptc-heat-rand-sine on usekag1aa,
ptc-heat-triv-sine on usekag2a).

---

## RESULT — march-by-sampling, KS (2026-08-11)

Policy (RL emulation, vanilla loss untouched, sampling-only): expand collocation
horizon [0, t*] when covered residual < eps. Outcome: t* marched 0.02 -> 0.18 (9x)
and stalled honestly — covered residual plateaued at 5.6e-3..1.4e-2 > eps, the
policy refused to advance (no goodharting, unlike KS-rar). The covered part is ALIVE
(norm_ratio 0.995 vs trivial 0.36-flatness; first ~6% of slices track), the uncovered
region is unsampled garbage (global L2RE 1.31 is meaningless there). The stall wall
is the known KS representation ceiling of the vanilla FNN — sampling policy is not a
representation fix. Heat-march (where the vanilla arch CAN represent the local
solution) is the decisive test; running.

---

## RESULT — march-by-sampling, heat + SSE chain (2026-08-11): the ghost ladder

Heat march: **escapes the trivial attractor** (norm_ratio 0.76 vs vanilla 0.031;
field alive) — the strongest no-loss-change result — **but lands on the WRONG rung**:
the field plateaus at 2.512 = sqrt(2*pi) to 3 digits (the SECOND zero of sin(u^2)),
overshooting the true level sqrt(pi). March stalls honestly at t*=14/100. Mechanism:
sin(u^2) self-gates at u in {0, sqrt(pi), sqrt(2pi), ...} — a LADDER of
near-zero-residual branches; the vanilla loss cannot distinguish rungs, so any
escape policy picks an arbitrary one. Branch selection = time-ordering from the IC =
exactly what the causal gate encodes (the plain-causal run from the same trivial
init climbed to the correct sqrt(pi) regime). SSE around the march checkpoint
(paper's stay-in-basin kicks x1/x2): local polish only (1.255 -> 1.199), no bridge
between rungs. Final division of labor: pattern+sampling agent = detection, escape,
economy; causal ordering = branch selection.

---

## RESULT — rung-veto experiment (constants generalization; 2026-08-12)

User extension: the trivial class includes CONSTANTS and frozen states generally, not
just u=0. Unifying property (measured): every member self-gates its dynamics — and
Heat-LT's forcing is explicitly time-dependent, so the true solution MUST respond.
New reference-free signals: F_freeze = |u_t| (abs-normalized) and F_drive = spectral
fraction at the known forcing frequency. Validation: ref 0.95/0.76; vanilla rung-0
0.015/0.043; march rung-sqrt(2pi) 0.15/**0.006** (dead to the forcing — caught even
where C_enrich/A_late were weak); GS constants caught by A_late (autonomous PDE: no
drive signal exists — honest scope limit).

Veto policy (march + frozen-tail veto -> kick -> re-march, 6 retries, loss untouched):
**6/6 dead branches correctly refused, zero false accepts** — unlike plain march
(which settled on sqrt(2pi)) the veto never certified a dead rung. But blind kicks
never FOUND the live branch (its weight-space measure is negligible among dead
basins); budget expired back at rung 0 (L2RE 0.999).

**Verdict on "avoid converging to constants via error space": YES as rejection —
the class-level frozen-dynamics signal is un-gameable (you cannot fake being alive)
and the policy never accepts a dead state; NO as synthesis — finding the live branch
by blind exploration fails, branch construction still requires causal ordering.**
