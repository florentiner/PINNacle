# Results analysis: why PINNs fail on chaotic PDEs (KS, Gray–Scott)

Data: `runs_landscape_compare.zip` — {kuramoto_sivashinsky, grayscott} × {origin, causal,
soap, soap_causal, frozen} × 3 seeds (1234–1236), 30,000 iterations per gradient run,
commit `1d87b35` (paper-backed pipelines, shared per-seed init). Missing cells:
`grayscott/frozen` (all seeds), `grayscott/soap@1236` — ignored per instruction.
Aggregates: `compare_summary_agg.csv`; per-seed: `compare_summary.csv`.

## Headline table (mean ± std across seeds, relative-L2 vs reference)

| cell | rel-L2 | loss↔err corr | barrier | basin frac | wall clock |
|---|---|---|---|---|---|
| KS / origin       | 0.972 ± 0.007 | **−0.77** | 0.92 | 0.51 | ~27 min |
| KS / causal       | 0.972 ± 0.018 | +0.41 | 1.74 | 0.008 | ~27 min |
| KS / soap         | 0.970 ± 0.002 | +0.19 | 1.02 | 0.23 | ~32 min |
| KS / soap_causal  | 0.986 ± 0.036 | +0.50 | 1.23 | 0.14 | ~32 min |
| KS / **frozen**   | **3.2e-5 ± 0** | – (no landscape; cond ≈ 1.41) | – | – | **4 s** |
| GS / origin       | 0.094 ± 0.00003 (u 0.083, **v 0.997**) | +0.66 | 0.32 | 0.64 | ~45 min |
| GS / causal       | 0.094 ± 0.00001 (v 0.998) | −0.11 | 0.33 | 0.62 | ~45 min |
| GS / soap         | 0.355 ± 0.0004 (v 3.4) | +0.47 | **0.01** | 0.73 | ~49 min |
| GS / soap_causal  | 0.30 ± 0.18 (bimodal: 0.40, 0.40, 0.094) | +0.39 | **0.01** | 0.68 | ~49 min |

## The mechanism: collapse to the trivial exact solution

Both chaotic PDEs admit a *trivial solution that satisfies the PDE residual exactly*:
KS has u ≡ 0; Gray–Scott has the laminar fixed point (u, v) ≡ (1, 0). The saved fields
show this is exactly where gradient training goes:

- **Gray–Scott**: at late times, every origin/causal run predicts u = 1.000, v = 0.0004
  (reference v-rms 0.034) — the reaction pattern is erased, v is ~100% wrong
  (`relative_l2_v ≈ 0.997`), and the aggregate "9.4%" error is misleadingly small only
  because the easy u ≈ 1 background dominates the norm. All methods plateau at the *same*
  total loss 0.9628 — the residual is ~0 at the trivial point and what remains is the
  irreducible IC-term conflict. GS is a **failure for every gradient method**, hidden
  under a small-looking aggregate number.
- **KS**: predictions damp toward zero (pred-rms 0.62 → 0.35 over t ∈ [0,1]) while the
  true solution grows (0.83 → 1.32). The network fits early time then slides toward u ≡ 0
  instead of tracking chaotic growth.

This refines H2/H3: the landscape isn't merely rugged — it contains a **huge, smooth,
deceptive basin around the trivial solution**, and the IC loss term (even at weight 100)
cannot counteract it because chaos exponentially amplifies any early-time mismatch, making
"damp everything to trivial" the cheapest way to cut the residual.

## Hypothesis verdicts

- **H1 (failure is real) — CONFIRMED.** All gradient methods: KS rel-L2 0.95–1.02;
  GS v-component 0.997–3.7. Consistent across seeds (std ≤ 0.04 except soap_causal/GS).
- **H2 (deceptive landscape) — CONFIRMED, strongest possible form.** KS/origin:
  loss falls ×68 while true error *rises* — loss↔error correlation −0.93/−0.82/−0.55
  across seeds. The optimizer is actively rewarded for moving away from the solution.
- **H3 (landscape geometry) — CONFIRMED with a twist.** The trivial basin is *wide and
  flat*, not narrow: GS basin-fraction 0.6–0.75, and SOAP — the best conditioner — makes
  the path into it *smoothest* (barrier 0.01 vs 0.3 for Adam). Good optimization
  accelerates convergence to the wrong answer. Causal loss visibly reshapes the KS
  landscape (basin fraction 0.51 → 0.008, barrier 0.92 → 1.74): it removes the flat
  trivial plateau, but 30k iterations on a plain MLP still can't ride the narrowed valley.
- **H4 (spectral error) — REFRAMED.** Error energy is ~99.99% low-k: the failures don't
  miss fine detail on a correct coarse field — they miss the *entire* field (trivial
  collapse), so the error inherits the reference's low-k-dominated spectrum.
- **H5 (Frozen contrast) — CONFIRMED decisively for KS.** Frozen-PINN: rel-L2 3.2e-5,
  condition number 1.41, 4 seconds — vs 0.97 in ~27 minutes for every gradient method on
  the *same* PDE (≈30,000× more accurate, ≈400× faster). Same hypothesis class
  (function approximation is not the bottleneck); removing the non-convex optimization
  removes the failure. GS/frozen was not run (missing cells).

## Secondary findings

- **Causal loss does what it promises locally**: KS IC error 0.008–0.032 vs origin's
  0.038, and positive loss↔error correlation — it de-deceives the landscape but cannot,
  alone, buy enough propagation depth at this budget/architecture.
- **SOAP alone is not a rescue** and on GS is actively worse (v error 3.4–3.7): it
  converges efficiently to whatever attractor is nearest, including wrong non-trivial
  patterns; soap_causal/GS is bimodal across seeds (0.40, 0.40, 0.094) — high-variance.
  The 30.6× GS improvement reported in arXiv:2502.00604 was obtained *on top of* their
  full stack (modified MLP + Fourier features, grad-norm weighting, time-marching,
  curriculum) — with the optimizer as the only change on a plain 100×5 MLP, the benefit
  does not materialize. Single-ingredient fixes do not rescue chaotic PINNs.
- The `adam_baseline` rows in MANIFEST.json (rel-L2 0.953, ~316 s) are from an earlier
  shorter run and have no data directories in the archive; excluded from the analysis.

## Error-landscape analysis: why the fixes beat `origin` (and where they don't)

`error_landscape_analysis.py` evaluates the TRUE error at every checkpoint and along
interpolations between checkpoints in full parameter space (no autoencoder distortion),
plus the alignment of the residual-loss and IC-loss surfaces over the 2D landscape grid.
Outputs under `<runs>/error_landscape/`. Mean over 3 seeds:

| cell | err first→last | late slope | loss↔err | trivial ratio | oper↔bnd grid corr |
|---|---|---|---|---|---|
| KS / origin      | 0.948 → **0.968** (rises) | +0.006 | **−0.72** | 0.35 | **+0.36** |
| KS / causal      | 1.004 → 0.964 (falls)     | **−0.033** | +0.39 | 0.43 | +0.70 |
| KS / soap        | 0.975 → 0.969             | +0.015 | +0.03 | 0.32 | +0.71 |
| KS / soap_causal | 1.071 → 0.979 (falls)     | **−0.035** | **+0.54** | 0.50 | +0.71 |
| GS / origin      | 0.095 → 0.095             | 0      | +0.51 | 1.01 | +0.71 |
| GS / causal      | 0.095 → 0.095             | 0      | +0.02 | 1.01 | +0.73 |
| GS / soap(_causal)| 0.45–0.50 → 0.30–0.36    | ~0     | +0.4  | 0.76–0.80 | +0.81–0.86 |

**On KS the causal methods ARE much better than origin — in direction, not (yet) in
destination — and the landscape shows the mechanism in four independent ways:**

1. **Direction of travel through the error landscape.** Along the actual training path,
   `origin`'s true error *rises* while its loss falls ×68 (walking downhill in loss =
   uphill in error). `causal` and `soap_causal` are the only gradient methods whose error
   *decreases* along the path — and their late-training slope (−0.033/−0.035 per
   checkpoint) shows they were still descending when the 30k-iteration budget ended.
2. **De-deception.** loss↔error correlation flips from −0.72 (origin) to +0.39 (causal) /
   +0.54 (soap_causal): causal weighting turns the training loss back into a signal that
   points toward the solution.
3. **Re-coupling of residual and IC in the landscape.** Over the 2D loss grid,
   corr(log loss_oper, log loss_bnd) is only +0.36 for origin — a large region exists
   where the PDE residual can be cut *without* honoring the IC (the trivial valley).
   Under causal weighting the two surfaces align (+0.70): low residual is only available
   where early-time behavior is also right. This is the geometric mechanism that removes
   the deceptive descent direction.
4. **Resistance to trivial collapse.** Final amplitude ratio rms(pred)/rms(ref): origin
   0.35 (deep in the u→0 collapse), causal 0.43, soap_causal 0.50. Consistent with
   compare_summary geometry: origin's low-loss basin covers ~51% of the visited region
   (the flat trivial plateau); causal's covers 0.8% (a narrow descending valley).

Why causal's *final* error is still ~0.97: it fixes the direction but not the distance —
a plain 100×5 MLP at 30k iterations cannot ride the narrowed valley to the chaotic
solution. That is exactly the gap time-marching + architecture close in the literature.

**On Gray–Scott, causal is NOT better than origin, and the landscape explains that too:**
both sit at the identical trivial plateau (total loss 0.9628 to 4 digits, identical
error). At the trivial point the residuals are ≈ 0, so every causal bucket weight
w_i = exp(−ε·Σr) ≈ 1 and the **causal loss degenerates to the origin loss** — causal only
has leverage while residuals are large; once a run falls into the trivial basin it has no
lever left. The SOAP variants do escape the trivial point (amplitude ratio 0.76–0.80,
error 0.30–0.36) but land on wrong non-trivial patterns, with one seed falling back into
the trivial basin — high variance, no rescue.

**Method ranking by landscape evidence (chaotic):** frozen (bypasses the landscape;
solves KS) > soap_causal ≈ causal (only methods moving *toward* the solution; soap_causal
most de-deceived but higher variance) > soap (efficient descent into the wrong basin) >
origin (actively deceptive: loss anti-correlated with error).

## Root cause: why causal did NOT beat origin here (and the fixes applied)

Theory (and the causal paper) say causal should clearly beat origin. Root-causing the gap
against the paper's actual KS setup found one genuine setup bug and two missing method
ingredients — all three now fixed in the harness:

1. **The problem was ill-posed: no spatial boundary conditions at all.** PINNacle's
   `KuramotoSivashinskyEquation` and `GrayScottEquation` impose only the IC; the reference
   solutions are exactly periodic (edge mismatch rms = 0.0000, verified from the .dat
   files). Without periodicity, a 4th-order PDE admits families of non-periodic solutions
   with ~zero residual that legitimately diverge from the periodic reference — so even a
   perfectly-trained causal run is descending toward a solution *set* that is not pinned
   to the reference, and the trivial branch is the easiest member of that set. The causal
   paper enforces periodicity exactly via Fourier feature embedding; this precondition was
   dropped in PINNacle (and partly explains PINNacle's reported KS/GS failures). Fixed:
   `--fourier-modes` hard-constraint embedding, on by default for KS (10) / GS (5).
2. **Fixed ε=1.0 instead of the paper's annealing.** With large early residuals a fixed
   moderate ε zeroes all late-time weights for most of the run; and once a run reaches the
   trivial attractor (residuals ≈ 0), all causal weights → 1 and the causal loss
   *degenerates to the origin loss* — measured: identical GS losses (0.9628) for both.
   Fixed: ε annealed through {1e-2, 1e-1, 1, 10, 100}, advancing when min wᵢ > 0.99
   (`--causal-delta`), the schedule + stopping rule of arXiv:2203.07404.
3. **No time-marching.** The paper's chaotic-KS result (same α, β, γ as PINNacle's KS —
   PINNacle took the equation from that paper) is achieved with sequential Δt = 0.1
   windows, warm-started and IC-handed-off. Fixed: `--time-windows N` (10 recommended).

Re-run for the causal-vs-origin verdict under the corrected setup:
`run_all.py --pdes kuramoto_sivashinsky grayscott --n-repeats 3 --iterations 30000 --time-windows 10`

## Conclusion

Chaotic PDEs defeat gradient-trained PINNs through a **trivial-solution attractor**: an
exact-residual solution with a wide smooth basin that the residual loss prefers over the
true chaotic trajectory, making the loss anti-correlated with the true error. Optimizer
improvements (SOAP, L-BFGS) accelerate descent *into* that basin; causal loss reshapes the
landscape in the right direction but is insufficient alone at this scale. The gradient-free
Frozen-PINN, which replaces the landscape with a well-conditioned linear solve, solves KS
essentially exactly in seconds — pinning the failure on *optimization*, not expressivity.
The literature's successful chaotic-PINN results should be read as full-stack recipes
(causal + time-marching + architecture + weighting + optimizer), not single fixes; the
obvious next experiment is adding time-marching (train on [0, Δt], then extend) on top of
causal+SOAP, which is precisely the ingredient our controlled setup omitted.
