# Round 7 — finding the recipe that beats deepxde origin on KS

Goal: not "which single ingredient", but the **configuration** that actually beats origin.
Two levers, found by auditing how the current best-practice stack *uses* the techniques.

## Audit: the existing best practices were used improperly

| # | improper usage | evidence | fix |
|---|---|---|---|
| **1** | **Fourier features under-resolved** — KS embeds only **10 modes** | a 10-mode basis has a hard **0.72 rel-L2 representation ceiling** (even a perfect net); Frozen needed 64 modes for 3e-5 | **`--fourier-modes 32`** (ceiling 0.020) or 48 (0.0009) |
| **2** | **Time-marching double-starvation** — causal ε-annealing (5 phases) runs *inside* each of 10 windows | 30000 / 10 windows / 5 ε = **600 iters per (window,ε)** — two budget-splitters compounding | on KS, **drop marching** (it can't beat no-marching at fixed budget, Round 4); if kept, use ≤2 windows and fixed ε |
| **3** | **Grad-norm cold start** — loss weights init [1,1] | takes ~1000–2000 iters to re-discover the IC×100 weighting, under-enforcing the IC exactly when it matters (Round 4) | init grad-norm weights at [1,100], or use physics-informed init |
| **4** | **No physics-informed init** — random start | cold start wastes early budget on all methods | PirateNets LSQ init of the last layer (`[GAP]`) |

Fix #1 is primary and quantifiable; the rest are secondary (they matter once #1 is fixed).

## The reach-factor prediction (why fixing #1 should win big)

origin@10 reached rel-L2 **0.914** against its **0.72** ceiling → optimization got within
**1.27×** of the best its basis allows. If that same reach-factor holds at higher resolution
(optimistic — the chaos horizon may prevent it):

| modes | representation ceiling | predicted achievable (1.27×) | vs origin@10 = 0.914 |
|---|---|---|---|
| 10 (shipped) | 0.719 | 0.914 (observed) | baseline |
| 16 | 0.347 | ~0.44 | **2× better** |
| 24 | 0.095 | ~0.12 | **7× better** |
| **32** | **0.020** | **~0.025** | **~37× better** |
| 48 | 0.0009 | ~0.001 | ~800× better |

The running CPU experiment (`origin@24`, `origin@32`, 20k iters, seed 1234) tests whether the
reach-factor holds or whether chaos stalls optimization far short of the lifted ceiling. The
**pessimistic** alternative: high modes are unreachable and error stalls ~0.7 regardless —
which would mean representation is necessary but the horizon still binds.

## The winning recipe (runnable with current flags)

```bash
# KS: the configuration predicted to beat origin -- modified MLP + adequate Fourier resolution,
# NO time-marching / causal / grad-norm (all shown to hurt KS at fixed budget in Round 4).
python experiments/landscape_compare/run_experiment.py --pde kuramoto_sivashinsky \
    --method ablation_A --fourier-modes 32 --iterations 30000 --seed 1234 --out runs_recipe

# the deepxde-origin baseline it must beat (as shipped, 10 modes):
python experiments/landscape_compare/run_experiment.py --pde kuramoto_sivashinsky \
    --method ablation_none --iterations 30000 --seed 1234 --out runs_recipe_origin
```

`run_recipe.sh` runs the full modes sweep {origin, ablation_A} × {10, 16, 24, 32} × 3 seeds
so the reach-factor curve is measured, not assumed. Best on a GPU (each 32-mode run is
~2.5 h on CPU; the modified-MLP arm ~3×).

## Additional best practices to try if #1 alone doesn't fully win (priority order)

1. **More collocation points** — Nyquist for 32 modes needs ≥64 spatial points; raise
   `num_domain` so the residual is not aliased at the new resolution.
2. **Physics-informed LSQ init** (audit #4) — removes the cold start; cheap.
3. **Higher / scheduled learning rate** — a richer 32-mode function may need a larger lr or
   longer warmup to converge near its ceiling.
4. **SSBroyden optimizer** — the strongest quasi-Newton (beats L-BFGS ~1000× on KS loss in
   arXiv:2501.16371); the optimization lever for closing the reach-factor gap.
5. **RBA / SA-PINN per-point weighting** — focus late-time residuals where the horizon bites.

## Honest bottom line (updated as results land)

Confirmed now: **the 10-mode Fourier embedding is a hard 0.72 ceiling — the single most
important fixable bottleneck, and it was hiding in "we already use Fourier features".**
Raising modes is the concrete recipe change predicted to beat origin by 1–2 orders of
magnitude *if* optimization can follow. Whether it does is being measured; this file is
updated with the verdict (do not cite a win until the numbers appear here).
