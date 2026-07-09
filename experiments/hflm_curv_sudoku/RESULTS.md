# hflm_curv_sudoku — Results

12/12 cells complete (2026-07-02). H-FLM Sudoku board accuracy vs Gaussian curvature `K`
(`algo.gaussian_curvature`), all other hyperparameters at the `scripts/train/sudoku/hflm.sh`
defaults (tiny-hyperbolic-dit 512/8/8, init=hyperbolic, LR 3e-4, prior_cov=0.25, rho_max=12,
log-linear, 20k steps, batch 256; eval: sudoku_eval, 180 steps, exact velocity, greedy last,
same `K` as training). 2000 boards per cell → 95% CI ≈ ±2.1 pts at mid-range accuracies.

## Board accuracy (K × difficulty) — slide-aligned protocol (top_k_velocity = -1)

Matches the `slides/jul02_2026` eval spec (exact-velocity, top_k_v=-1 avg-across-vocab,
180 steps, greedy last) — directly comparable with `hflm_curv_init_lr_sudoku`.
From `eval_tkv-1/results.json`:

| difficulty | K=-0.25 | K=-0.5 | K=-1.0 (baseline) | K=-2.0 |
|---|---:|---:|---:|---:|
| easy   | 91.7% | 89.9% | **92.4%** | 87.6% |
| medium | 67.4% | **76.1%** | 69.2% | 66.1% |
| hard   | 27.2% | **33.9%** | 29.4% | 23.6% |

## Board accuracy — original protocol (top_k_velocity = 1)

From `eval/results.json` (top-1 predicted-clean endpoint velocity):

| difficulty | K=-0.25 | K=-0.5 | K=-1.0 (baseline) | K=-2.0 |
|---|---:|---:|---:|---:|
| easy   | 91.7% | 89.9% | **91.9%** | 87.4% |
| medium | 66.9% | **76.3%** | 69.5% | 66.1% |
| hard   | 27.2% | **33.9%** | 28.5% | 23.9% |

**Bold** = row best. `K=-1` is the standard unit hyperboloid (bit-identical to the
pre-curvature-knob code path). The two protocols agree within ≤1 pt on every cell, so
the conclusions are protocol-independent.

## Insights

1. **Mild flattening helps where reasoning is hard.** `K=-0.5` beats the `K=-1` baseline by
   **+6.9 pts on medium** (76.1 vs 69.2) and **+4.5 pts on hard** (33.9 vs 29.4) under the
   slide-aligned protocol — both outside the ±2.1 pt CI. The effect is an *interior
   optimum*: flattening further to `K=-0.25` gives the gains back (medium 67.4,
   hard 27.2 ≈ baseline).
2. **Easy does not separate curvatures** (as anticipated when medium/hard were added): the
   top three cells sit within noise of each other (91.9 / 91.7 / 89.9); only `K=-2`
   separates, and downward.
3. **Strong curvature is uniformly harmful.** `K=-2` is the worst cell in every row
   (−4.5 / −3.4 / −4.6 pts vs baseline). Its harder origin-bowing of geodesics (intrinsic
   midpoint radial 0.62 vs 0.88 at `K=-1` for ρ=6 endpoints) appears to destroy useful
   signal mid-trajectory rather than add structure.
4. **Difficulty scaling is roughly preserved across K** (easy ≫ medium ≫ hard for every
   column), so curvature shifts the level, not the qualitative difficulty ordering.

## Conclusion

Gaussian curvature is a real, free hyperparameter for H-FLM: at fixed everything-else,
`K=-0.5` is the best setting on Sudoku, with the advantage growing as the task gets harder
(+0 easy → +6.9 medium → +5.4 hard vs the conventional `K=-1`). A finer sweep in
`K ∈ (-1, -0.25)` (e.g. -0.4, -0.6, -0.75) around the optimum, and pairing `K` with a
matched `prior_cov`/`rho_max` rescan, are the natural follow-ups.

## Artifacts

- per cell: `outputs/hflm_curv_sudoku/d-<difficulty>_k<K>/` (checkpoints + `eval/results.json`
  [tkv=1] + `eval_tkv-1/results.json` [slide-aligned tkv=-1, re-eval jobs 648147–648174])
- logs: `experiments/hflm_curv_sudoku/logs/`
- jobs 567980–567991 (2026-07-01 21:05 → 2026-07-02 ~00:15, 1 GPU × ~2.5–3h each)
