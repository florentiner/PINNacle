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
