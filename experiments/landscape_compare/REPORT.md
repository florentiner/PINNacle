# Final report: original DeepXDE vs the best-practice stack on chaotic PDEs

**Data**: 96-run ablation sweep in `runs_landscape_compare` — {Kuramoto–Sivashinsky (KS),
Gray–Scott (GS)} × all 16 combinations of the 4 best-practice ingredients × 3 seeds, 30k
iterations, shared per-seed initialization, exact-periodicity embedding on everywhere.
Ingredients: **C** = causal loss + ε-annealing (arXiv:2203.07404), **W** = 10-window
time-marching, **A** = modified MLP (arXiv:2001.04536), **G** = grad-norm loss balancing
(arXiv:2308.08468). `origin` = ablation_none, `best_practice` = ablation_all.
Sources: `ANALYSIS.md` (rounds 1–2), `similarity/variance_decomposition.csv`,
`error_landscape/`, `test_hypotheses.py` verdicts (H6–H10 in `HYPOTHESIS.md`).

---

## 1. Executive summary

**The honest headline first: at benchmark scale (100×5 net, 30k iterations), the full
best-practice stack is *not* better than origin on KS (+4.5% worse final error) and is
better on GS (−43.6%) only by making convergence to the trivial background *reliable* —
neither recovers the chaotic solution.** Both fail at the same wall: a chaos predictability
horizon (KS: λ≈25/unit ⇒ any ~4%-accurate field decouples from the truth after t≈0.2, which
is exactly where every method collapses) and a volume-dominant trivial attractor (GS: the
pattern occupies 1.7% of the reference spacetime; the pattern-free background zeroes the
residual on the other 98%).

What the ingredients *do* accomplish is real but lives one level down: they reshape the
**training process** (81–94% of trajectory/landscape variance is ingredient-driven), while
the **destination** stays locked twice over — final weights cluster by seed (η² 0.63–0.80:
no ingredient leaves the shared init's basin) and final solutions land on the same
horizon-limited answer.

## 2. Which part makes what impact — in percent

Factorial main effect of each ingredient on the **final relative-L2** (mean over all 8
with/without pairs; negative = improves over not having it):

| ingredient | KS impact | GS impact |
|---|---|---|
| C — causal loss | +5.5% (worse) | **−15.6%** |
| W — time-marching | +10.3% (worse) | −1.8% |
| A — modified MLP | +6.8% (worse)\* | +0.3% |
| G — grad-norm balancing | **−6.9%** | **−11.9%** |
| **full stack vs origin** | **+4.5% (worse)** | **−43.6%** |

\* interaction caveat: **A alone is the best single KS cell** (0.914 vs origin 0.918); its
negative *average* is dragged by interactions with budget-starved marching (WA/CWA diverge
to 1.17–1.41). Main effects hide interactions — the full 16-cell table is in
`compare_summary_agg.csv`.

The GS −43.6% decomposes almost entirely into **reliability, not pattern recovery**: origin
lands at 0.177 ± 0.118 across seeds (bad seeds stall above the trivial optimum), while C- or
W-containing combos pin every seed to the trivial-background solution (0.094–0.095 ±
**0.0007–0.0009** — a ~150× variance reduction) and fit the background 3× better (0.128 →
0.039). The pattern region itself stays ≥1.1 rel-L2 for *every* method, origin and stack
alike (H6b).

## 3. How each part influences the training process (and in what sense it "beats" origin)

Main effects on process metrics, KS (origin baseline: early-band error 0.053, tracked
horizon t\*=0.30, init→final barrier 3.67, loss↔error corr +0.91, final amplitude ratio 0.45):

| ingredient | early fit | horizon | barrier | loss↔err honesty | mechanism |
|---|---|---|---|---|---|
| C | **−0.08 (better)** | +0.01 | **−1.24 (smoother)** | −0.22 | ε-annealing spends the budget on early time buckets: better early fit and a much smoother descent path (barrier 3.67→~0.5 for C alone) — but the horizon does not move, so the better-fit region is the same-size region. |
| W | +0.19 (worse) | −0.04 | **−1.99 (smoothest)** | +0.04 | warm-started windows keep the trajectory in-basin (barrier ≈ 0) and make GS convergence deterministic; on KS at fixed budget, 10×3k iterations *starves* each window and the IC-handoff error **compounds** (CW ends at 1.84 — worse than the shared init, H7). |
| A | −0.02 (best) | **+0.02 (only positive)** | +0.98 | **+0.19 (most honest)** | the encoder-gated architecture represents the early dynamics best (t<0.2 error 0.038–0.050 across its healthy combos) and gives the most faithful loss signal — the right *representation* ingredient, but it buys ~one time-band of horizon at most. |
| G | +0.16 (worse) | −0.06 | +0.15 | −0.23 | auto-balancing rediscovers ~the IC×100 weighting then keeps shifting weight toward the residual: it *sacrifices* early-fit quality (and amplitude, −0.48) to settle faster into the least-bad average — the only KS ingredient that lowers the aggregate error, and it does so without extending tracking at all. |

**The one-line mechanism summary:** C makes the descent smoother, A makes it more honest
and better-fitted early, W makes it deterministic, G makes it settle lower on average — and
none of them moves the predictability horizon (t\* stays 0.10–0.30 across all 16 combos,
with origin *tied for best*), because the horizon is set by the PDE's error-amplification
rate, not by the optimization process.

## 4. Why origin fails — and why the stack can't fix it at this scale

From `ANALYSIS.md` rounds 1–2 and the H6–H10 tests:

1. **(Fixed) The benchmark was ill-posed** — PINNacle's KS/GS ship with no spatial BCs while
   the references are periodic; before the periodicity embedding, origin's landscape was
   actively deceptive (loss↔error corr −0.72). With the fix, origin improves 0.97→0.918 and
   the landscape becomes honest (+0.91) — the remaining failure is *not* a landscape trick.
2. **The predictability horizon** — origin solves t<0.2 to 4% error, then its prediction
   decays to the trivial branch while the truth grows; it converges (error flat from epoch
   10k to 30k) at total loss 6e-3. λ≈25/unit means holding t=1 would need ~1e-11 field
   error. "Track-then-decay" is the best minimum that exists for a soft residual loss.
3. **The stack changes the path, not the destination** — trajectory/landscape features are
   81–94% ingredient-determined (the parts genuinely work as advertised *on the process*),
   but final weights are seed-locked (travel from init = 0.19–0.42× the between-seed init
   distance) and final solutions are attractor-locked (KS: all methods 4× closer to each
   other than to the reference; GS: every method reproduces the 98% background and misses
   the 1.7% pattern).

## 5. Bottom line

- **"Original DeepXDE is not good at chaotic PDEs"** is true, for three stacked reasons:
  an under-determined benchmark definition (fixable, fixed here), a chaos predictability
  horizon that caps *any* soft-residual gradient method at this scale, and a
  volume-dominant trivial attractor.
- **"Best practice is better than origin"** is true only in specific, quantifiable senses:
  GS mean error −43.6% (≈150× seed-variance reduction + 3× better background fit), KS
  grad-norm −6.9% on the aggregate; plus genuinely better process properties (smoother,
  more honest, better early fit — Section 3). It is **not** better where it matters most:
  neither extends the KS horizon (origin is tied for the best t\*) nor recovers the GS
  pattern.
- **What would actually help**, per this data + the literature deltas: (a) scale — width
  ≥256 modified MLPs with 10–100× more iterations *per window* (so W stops starving and
  compounding); (b) for KS-like periodic problems, skip the fight entirely — the
  gradient-free Frozen-PINN solves the same equation to 3.2e-5 in 4 seconds because it
  replaces the horizon-limited optimization with a well-conditioned linear solve.

*Figures*: `similarity/*_embedding.pdf` (process-vs-destination dissociation),
`error_landscape/kuramoto_sivashinsky_trivial_attraction.pdf` (amplitude decay to the
trivial branch, all methods), `comparison_figures/relative_l2.pdf` (final errors ± seed
spread), `error_landscape/*_trajmap.pdf` (per-method trajectories over their landscapes).

---

## Appendix: double-check of the "origin beats best_practice on KS" result

Because the result is counterintuitive, it was re-verified end to end. **It is real, and its
cause is fully identified — three fixed-budget schedule artifacts, not a code bug.**

**Pipeline checks (all pass):**
- Configs correct (`ablation_all`: causal + modified_mlp + grad_norm + 10 windows; `none`:
  plain FNN, origin loss, 1 window; both fourier=10, 30k iterations).
- Sign-consistent in **all 3 seeds** (none 0.914/0.926/0.915 vs all 0.950/0.989/0.941).
- Independent recomputation from raw `fields.npz` reproduces `metrics.json` exactly.
- Window stitching clean: prediction jumps across all 9 window boundaries are *smaller*
  than the reference's own step-to-step change (no evaluation/handoff bug).
- Both runs reach comparable, tiny final losses (6.0e-3 vs 7.6e-3) — the difference is in
  *which function* that loss buys, not in failed optimization.

**Root cause chain (each link verified from the data):**
1. **Window starvation (W):** window 1 of the stack — 3000 iterations exclusively on
   t∈[0, 0.1], *analytic* IC, no handoff involved — fits its own window at rel-L2 **0.254**
   vs origin's **0.041** on the same region. `ablation_W` alone confirms: early-band fit
   2.3× worse than origin.
2. **Grad-norm cold start (G):** weights start each window at [1, 1] (paper-faithful) and
   the log shows them reaching only ~[1, 9] by iteration 2000 of 3000 — vs origin's fixed
   IC×100. The IC is under-enforced precisely when it matters; `ablation_G` alone: early
   fit 2.6× worse (0.108). In later windows grad-norm settles at ≈[1.1, 1.1] (handed-off
   ICs are easy, so it never re-weights them up).
3. **Compounding:** the IC handoff then propagates window-1's 0.25 error through 9 more
   windows (H7 mechanism).
Secondary: causal methods recompile per ε sub-phase, so the lr scheduler restarts each time
— stacked runs train at constant lr 1e-3 while origin decays to 0.21×1e-3 (final polish).
No loss blow-ups at the restarts (checked), so this is a minor contributor.

**Exculpated ingredients:** the modified MLP is innocent — `ablation_A` alone *beats*
origin's early fit (0.027 vs 0.041) and ties its horizon; causal alone is mild (0.063).

**Interpretation:** this is a *fixed-total-budget* comparison (30k iterations for everyone —
the fair controlled design). The stack's ingredients carry per-window/warm-up overhead that,
under budget parity, costs more than they return, while origin already sits at the
predictability-horizon wall (t\*=0.30, tied best of all 16 combos) — so there was no headroom
for the stack to win on KS in the first place. At the papers' budgets (10–100× more
iterations per window) artifacts 1–2 dissolve; the claim "origin > best_practice" holds at
benchmark scale, and says the stack's overhead is real, not that the papers are wrong.
