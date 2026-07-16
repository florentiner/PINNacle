# The full catalog of PINN best practices for chaotic / stiff PDEs

A literature survey of *every* technique that matters for hard time-dependent PDEs
(Kuramoto–Sivashinsky, Gray–Scott, Allen–Cahn, KdV, Navier–Stokes), grouped by the axis it
acts on, with: what it does, why, the paper, whether **this repo** already has it, and how it
maps onto the two failure walls this study identified — the **predictability horizon** (KS)
and the **trivial attractor** (GS).

Legend for status: **[HAVE]** implemented in `experiments/landscape_compare/`; **[FLAG]**
implemented behind a CLI flag; **[GAP]** not implemented (candidate next work).

The single most important meta-finding of this project: the papers' "best practice" is a
**full stack**, and the state-of-the-art single method (**PirateNets**, JMLR 2025) is
literally the union of ~8 of the techniques below. No single ingredient is the fix.

---

## 0. The reference recipe: PirateNets = the whole stack in one architecture

PirateNets (Wang, Sifan et al., *Physics-informed Deep Learning with Residual Adaptive
Networks*, arXiv:2402.00326, JMLR 25) is the current SOTA and is exactly a checklist of this
document. It combines, all at once:
random Fourier feature embedding · modified-MLP (two-encoder) gating · **adaptive residual
blocks with an α-gate initialized to 0** (net starts shallow/linear, deepens as training
proceeds — the piece unique to PirateNets) · **physics-informed least-squares initialization**
of the last layer · Random Weight Factorization · tanh · exact periodic embedding ·
grad-norm/NTK loss balancing · causal training · Adam with 5000-step warmup + ×0.9/2000 decay.
Result: it is the *only* architecture that keeps improving as depth grows to 18 layers
(Allen–Cahn 2.2e-5, KdV 4.3e-4). **Every item in this list is one line of that recipe.**

---

## 1. Architecture

| technique | what / why | paper | status |
|---|---|---|---|
| **Modified MLP** (two encoders U,V gate every layer) | mitigates gradient pathologies; the JAX-PI default | Wang, Teng & Perdikaris 2021 (arXiv:2001.04536) | **[HAVE]** `ModifiedMLP` (method `A`/`best_practice`) |
| **PirateNets adaptive residual block** | `x_{l+1}=α·h_l+(1−α)·x_l`, α init 0 ⇒ deep net starts as identity/shallow and deepens; fixes derivative-trainability collapse at depth | arXiv:2402.00326 | **[GAP]** — the highest-value missing architecture piece for going deep |
| **Random Weight Factorization (RWF)** | `W=diag(exp(s))V`, s∼N(1,0.1); same init function, better-conditioned geometry | Expert's Guide arXiv:2308.08468 | **[FLAG]** `--rwf` |
| **Adaptive / learnable activations** (e.g. per-layer scaled tanh, or KANs) | reduce spectral bias; steeper features | Jagtap et al. 2020; KAN-PINN 2024 | **[GAP]** |
| **Hard-constraint output layers** (BC/IC baked into the ansatz) | removes a whole loss term; our Frozen-PINN uses this idea | many; PINNacle uses it for some cases | **[HAVE]** for Frozen-PINN; partial for gradient methods (periodicity) |

---

## 2. Input embedding / spectral bias

| technique | what / why | paper | status |
|---|---|---|---|
| **Random Fourier Features** `Φ(x)=[cos(Bx),sin(Bx)]`, B∼N(0,σ²) / more modes | canonical spectral-bias fix; **necessary** (10-mode KS embedding has a hard 0.72 rel-L2 ceiling, Round 7) but **NOT sufficient**: raising modes alone at fixed budget made KS *worse* (0.914->0.947@24->1.047@32) because the richer basis is a harder optimization problem (IC error 0.0016->0.247). Must be paired with stronger optimization/more budget. | Tancik 2020; Wang 2021 (arXiv:2012.10047); Round 7 | **[TESTED: necessary, not sufficient]** |
| **Exact-periodicity Fourier embedding** (our fix) | hard-encodes the true spatial period ⇒ well-posed; the specific RFF instance for periodic domains | causal paper arXiv:2203.07404 | **[HAVE]** `--fourier-modes` (KS 10 / GS 5, on by default) |
| **Multi-scale / multi-frequency Fourier features** | separate σ bands for multi-scale problems | Wang 2021 multi-scale | **[GAP]** |

*Our finding*: on KS the exact-periodicity embedding was **the** well-posedness fix (turned
loss↔error correlation from −0.72 to +0.91). A general tunable-σ RFF is the obvious next
lever for representing finer scales — but note it does **not** move the predictability
horizon (that's optimization/precision-bound, §8/§9), only what's representable.

---

## 3. Initialization

| technique | what / why | paper | status |
|---|---|---|---|
| **Physics-informed least-squares init** | solve `min_W‖WΦ−Y‖` for the last layer against the IC/BC/linearized PDE ⇒ "optimal initial guess", removes the cold start | PirateNets arXiv:2402.00326 | **[GAP]** — cheap, high-value, directly attacks the grad-norm/causal cold-start artifact Round 4 found |
| **Controlled seed init** (shared across methods, differs across seeds) | fair controlled comparison; not an accuracy trick | this project | **[HAVE]** `seed_init_network` |

---

## 4. Loss weighting (the λ_r, λ_b, λ_0 balance)

| technique | what / why | paper | status |
|---|---|---|---|
| **Fixed weights** (e.g. IC×100) | baseline | — | **[HAVE]** origin default |
| **Grad-norm balancing** | λ_i ∝ (Σ‖∇L_j‖)/‖∇L_i‖, moving-avg | Expert's Guide arXiv:2308.08468 | **[HAVE]** `GradNormLossWeights` (method `G`) |
| **NTK weighting** | balance by neural-tangent-kernel eigenvalues | Wang 2022 (arXiv:2007.14527) | **[GAP]** |
| **Self-adaptive weights (SA-PINN)** | per-point trainable weights, max–min game; attention to hard points | McClenny & Braga-Neto 2020 (arXiv:2009.04544) | **[GAP]** |
| **Residual-based attention (RBA)** | gradient-free per-point weights from residual magnitude; ~free, strong | Anagnostopoulos 2024 (arXiv:2307.00379) | **[GAP]** — cheapest high-value weighting upgrade |
| **BRDR** (balanced residual decay rate) | weights by per-residual convergence rate; beats RBA/NTK in recent benchmarks | arXiv:2407.01613 | **[GAP]** |
| **Causal (time-bucket) weighting** | forbid fitting late time before early; the chaos-specific one | causal paper arXiv:2203.07404 | **[HAVE]** `C` + ε-annealing |

*Our finding*: grad-norm helped only as a rescue of starved marching, and its cold start
*hurt* the early fit at fixed budget (Round 4). RBA/SA-PINN (per-point, gradient-free) are
the untested members and are the most likely to help on GS's sparse-pattern region (§ where
1.7% of points carry the signal).

---

## 5. Training curriculum (time / stiffness)

| technique | what / why | paper | status |
|---|---|---|---|
| **Causal ε-annealing** | sweep ε through [1e-2…1e2], advance at min-weight>0.99 | arXiv:2203.07404 | **[HAVE]** |
| **Time-marching / seq-to-seq** (train [0,Δt], extend) | the standard chaotic-KS device; each window a subproblem | arXiv:2203.07404 | **[HAVE]** `--time-windows` |
| **Curriculum on stiffness / physical parameter** | ramp the hard coefficient (viscosity, reaction rate) from easy→true | curriculum-adaptive works 2024–25 | **[GAP]** |

*Our finding*: time-marching is a double-edged sword — it makes GS deterministic but at fixed
budget **starves** each KS window (the dominant cause of `best_practice` losing to origin,
Round 4). Its value is entirely budget-dependent (H11/H14, pending on the GPU machine).

---

## 6. Adaptive collocation sampling

| technique | what / why | paper | status |
|---|---|---|---|
| **RAR** (residual adaptive refinement) | add top-k highest-residual points iteratively | Lu 2021 (DeepXDE) | **[GAP]** |
| **RAD / RAR-D** (residual adaptive distribution) | resample the whole set ∝ residual density | Wu 2023 (arXiv:2207.10289) | **[GAP]** |
| **RL / adversarial adaptive sampling** | learn where to sample | RL-PINNs arXiv:2504.12949; RAMS 2025 | **[GAP]** (the repo's *other* RL pipeline targets this axis) |

*Our finding*: **this is the most promising untested family for GS.** The pattern occupies
1.7% of spacetime; uniform collocation lets the trivial-background residual dominate. RAR/RAD
would concentrate points on the pattern region — directly attacking the trivial-attractor
volume argument, which no ingredient in our current stack does.

---

## 7. Optimization (optimizer + schedule)

| technique | what / why | paper | status |
|---|---|---|---|
| **Adam + exp-decay + warmup** | the universal first-order base (Adam-only for chaotic) | Expert's Guide arXiv:2308.08468 | **[HAVE]** (`--warmup`, step-decay) |
| **L-BFGS refinement** | 2nd-order polish; but *weak on chaotic* | classic | **[FLAG]** `lbfgs_baseline` (kept to confirm it underperforms) |
| **SOAP** (Shampoo+Adam) | quasi-2nd-order; 30× on GS in its paper (with the full stack) | arXiv:2502.00604 | **[HAVE]** `soap`/`soap_causal`, paper-tuned |
| **Self-scaled Broyden (SSBroyden / SSBFGS)** | far outperforms L-BFGS on KS (up to 1000× on loss) | arXiv:2501.16371, arXiv:2507.16008 | **[GAP]** — the strongest quasi-Newton, untested here |
| **NysNewton / Gauss-Newton / natural-gradient** | curvature-aware; strong on stiff | 2024–25 | **[GAP]** |
| **RL-chosen optimizer chains** | learn the switch schedule | PINN-PELINE (this repo) | **[HAVE (oracle)]** `--method chain` (Round 6) |

---

## 8. Domain decomposition

| technique | what / why | paper | status |
|---|---|---|---|
| **FBPINN** (finite-basis, overlapping subdomains) | per-subdomain normalization kills spectral bias; divide-and-conquer | Moseley 2021 (arXiv:2107.07871) | **[GAP]** |
| **XPINN / cPINN** (space-time / spatial decomposition) | many small nets, parallel; XPINN also splits time | Jagtap 2020/2021 | **[GAP]** (our time-marching is the 1-D-in-time special case) |
| **Multilevel DD + coarse-space correction** | scalability | arXiv 2024 | **[GAP]** |

---

## 9. Precision

| technique | what / why | paper | status |
|---|---|---|---|
| **float64 training** | the horizon law t*≈ln(1/ε)/λ makes the achievable field error ε a first-class knob; float32 floors ε | folklore + our H12 | **[FLAG]** `--float64` (H12 test pending) |

---

## How each family maps onto *our two walls*

- **KS predictability horizon** (any soft-residual method decouples from truth after
  t≈ln(1/ε)/λ): only things that lower the achievable error ε at late time can move it —
  **richer representation** (RFF-σ, PirateNets depth, adaptive activations), **better
  optimization** (SSBroyden, natural-gradient), and **precision** (float64). Weighting and
  sampling reshape the *path*, not the wall. This is why *no* ingredient in our benchmark
  sweep beat origin on KS.
- **GS trivial attractor** (pattern = 1.7% of volume; laminar point zeroes the residual on
  the other 98%): the direct attacks are **adaptive sampling** (RAR/RAD — concentrate on the
  pattern), **per-point weighting** (RBA/SA-PINN — up-weight the pattern residuals), and
  **hard IC constraints**. Our current stack only buys *reliability* (steering every seed to
  the less-bad trivial branch), never pattern recovery — consistent with it lacking every
  member of exactly those two families.

## Priority of the untested items (highest expected value first)

1. **RAR/RAD adaptive sampling** — the one family that directly attacks GS's trivial
   attractor; cheap; the repo's RL sampler already lives on this axis.
2. **RBA / SA-PINN per-point weighting** — ~free, complements sampling, untested.
3. **PirateNets residual block + physics-informed LSQ init** — removes the cold-start
   artifact and unlocks depth; the SOTA architecture we're one component short of.
4. **General tunable-σ random Fourier features** — finer representable scales on KS.
5. **SSBroyden optimizer** — the strongest quasi-Newton; the "which optimizer" paper's winner.
6. **float64** — already flagged; the cheapest test of whether KS's wall is precision-bound.

Items 1–3 are the concrete recommendations for a "Round 7" if the goal shifts from *why
origin fails* (answered) to *actually solving* these chaotic PDEs with a gradient PINN.
