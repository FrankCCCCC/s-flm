# H-FLM Curvature × Adaptive-Schedule — GenPPL Table

**Metric:** GenPPL (gpt2-large retokenized generative perplexity; **lower = better**). This is a
TinyStories language-modeling task, so there is no "accuracy" — GenPPL is the performance metric.

**Setup:** Self-conditioning **ON** throughout. Each cell is averaged over **init × prior_cov**
(3 × 3 = 9 runs). Learning rate is **not** swept (fixed 3e-4). Cells with entropy < 3.0 (degenerate
collapse) are **excluded** from mean/median (their low GenPPL is deceptive) and counted as `Nc`.

- **w/o ada** = naive `log-linear` noise schedule (from `geo_curv_sc_tinystories_256` Part B)
- **w/ ada**  = `log-linear-adaptive` noise schedule (from `hflm_ada_sc_tinystories_256`, SC-on cells)

| K (Gaussian curvature) | w/o ada — mean / median (done/9, collapses) | w/ ada — mean / median (done/9, collapses) |
|---|---|---|
| -0.01 | 19.2 / 17.7 (9/9, 0c) | 21.9 / 15.1 (9/9, 0c) |
| -0.1 | 48.1 / 28.3 (9/9, 2c) | 190.3 / 58.1 (9/9, 3c) |
| -0.25 | 33.6 / 33.4 (9/9, 0c) | 22.4 / 19.8 (9/9, 0c) |
| -0.5 | 34.9 / 32.2 (9/9, 0c) | 58.4 / 60.6 (9/9, 1c) |
| -0.75 | 39.0 / 38.6 (9/9, 0c) | — (9/9, 9c) |

## Reads

1. **Flat curvature (K=−0.01) wins for both schedules.** At K=−0.01, the adaptive median (15.1) beats
   naive (17.7), and the single best cell overall is adaptive (`k−0.01_c0.01_pc1.0_scon` = **10.70**).
2. **Adaptive is higher-variance.** Its mean exceeds its median at K=−0.01 (great best-case, some poor
   cells drag the average up), whereas naive's mean ≈ median. Averaging *hides* the adaptive win.
3. **K=−0.1 is the fragile zone** for both (collapses; adaptive worst — mean inflated by a near-degenerate cell).
4. **Deep curvature (K=−0.5, −0.75) is poor** for both; adaptive is worse / more collapse-prone there.

## Caveats

- **Complete:** all 90 cells done (2026-07-24).
- Single-run GenPPL has large run-to-run variance (seed×3 error-bars pending on the best cells).
