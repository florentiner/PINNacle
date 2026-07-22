# Dimensionality-reduction figures from the collected data

How the saved trajectories / solution fields / causal histories can drive
meaningful low-dimensional figures. Two correctness rules first, because they
decide whether a figure is informative or misleading.

## Rules

1. **Parameter-space comparison is valid ONLY within the same architecture.**
   Causal and ablation both use the modified-MLP (identical param dim, identical
   seed/init, per window) → comparable in weight space. Vanilla is a plain FNN
   (different architecture and dim) → its weights are NOT comparable to causal.
   Cross-architecture comparison must be done in **solution/function space**
   (predictions live on the same (x,t) grid regardless of the network).

2. **PCA ≠ t-SNE/UMAP — different questions.**
   - PCA: linear, preserves global geometry, axes are meaningful → use for
     *trajectory shape, divergence, on-manifold projection*.
   - t-SNE/UMAP: nonlinear, distort global distance, axes meaningless → use ONLY
     for *clustering/neighborhood* over MANY points. A t-SNE of a single
     trajectory is a meaningless squiggle. Never read distances/axes off it.

## Figures (ranked by scientific value)

### A. Causal vs ablation trajectory divergence — parameter-space PCA  ★★★
- Data: `runs/.../trajectory/w{k}_trajectory_flat.npy` for causal (w0–9) and
  ablation (w0–8). Same arch, same init, same window ⇒ comparable.
- Method: stack both paths for a window, joint PCA → 2–3D, draw both trajectories
  in the shared plane; mark the shared init and each final.
- Shows: the fork point where causal weighting steers off the uniform-weight
  path, and whether they reach the same basin or split (finals differ ~1.8× ⇒
  expect a visible fork). Visually explains the ablation number.
- Report: variance explained by PC1/PC2.

### B. Solution-space evolution — UMAP/t-SNE across ALL methods  ★★★
- Data: prediction snapshots (`arrays/pred_*.npy`, per-window preds) from causal,
  ablation, and vanilla + the **reference solution as an anchor point**.
- Method: each snapshot = flattened u(x,t); embed all with UMAP (or t-SNE);
  color by L2RE; mark the ref anchor and the IC.
- Shows: causal/ablation predictions trace a path approaching the true-solution
  anchor; vanilla clusters at a "ghost" attractor far from it. Legitimate t-SNE
  use (many points, meaningful L2 input, anchored, colored by a real metric).
- One-figure summary of the whole study.

### C. POD / on-manifold test — PCA of the REFERENCE solution  ★★★
- Data: `arrays/ref.npy` (reference snapshots) → PCA gives the true POD modes of
  the chaotic attractor.
- Method: project causal, ablation, vanilla predictions onto the top ref POD
  modes; plot trajectories in mode-coefficient space, ref as the target curve.
- Shows: causal stays ON the physical manifold; vanilla falls OFF it.
  Quantitative, physically interpretable (reduced-order-model view), no t-SNE.

### D. Seed-robustness cluster — t-SNE of final solutions  ★★ (needs seed-2024)
- Data: final solutions across seeds × methods (Kaggle seed 1234 + server
  seed-2024 runs).
- Shows: causal finals form a tight cluster (seed-robust — the outlier check);
  vanilla scatters, all far from truth. Directly answers "is 1234 an outlier?".

### E. Causal-front self-similarity — PCA of W(t) trajectories  ★★
- Data: `causal/history.npz` W-vectors per window (per log step).
- Method: treat each window's W-over-training as a path; PCA all windows together.
- Shows: the tol-annealing W-front is the same shape across all 10 windows
  (paths collapse onto one curve) ⇒ the mechanism is window-invariant.

## Pitfalls to avoid
- t-SNE of a single trajectory (meaningless).
- Any vanilla-vs-causal comparison in weight space (architecture mismatch).
- Reading quantitative distance/axes off a t-SNE/UMAP plot.
- Omitting the reference anchor in B/D (removes the only interpretable landmark).

## Runnable now vs pending
- A, C, E: runnable immediately from local data (pure numpy/torch + PCA).
- B: needs `umap-learn` (or sklearn t-SNE) install.
- D: needs the server seed-2024 runs first.
