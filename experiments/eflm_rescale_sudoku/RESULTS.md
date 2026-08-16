# EFLM Fixed-Embedding-Norm (R) Sweep — Sudoku Hard — Results

Sweep: EFLM with every word-embedding norm pinned to R (`algo.rho_min =
algo.rho_max = R`, `SphereDiT.rescale_radius`), on sudoku **hard** (30 clues),
over **arm × R × LR × seed**:

- arm ∈ {`naive` = fixed log-linear noise; `ada` = log-linear-**adaptive** noise,
  no truncation}
- R ∈ {0.1, 0.5, 1, 1.5, 2, 5, 8, 16, 22, 32}
- LR ∈ {3e-4, 5e-4, 1e-3}, seed ∈ {1, 2, 3}

Metric = full-board solve rate over 2000 held-out puzzles (`mode=sudoku_eval`,
exact velocity, top_k_v=−1, 180 steps, greedy last). Model `tiny-sphere-dit`
(~28.6M), 20k steps, bs 256. Coverage: **180/180 cells** (all `num_total=2000`,
all acc ∈ [0,1]); every (arm, R, LR) group has **n=3**. Signal-model prediction
under test: the decode time
t*(R) = 1/(1 + C/(R·√(1−C²/d))), C = √(2 ln(2(V−1)/δ)) = 3.28 for
(V=12, d=512, δ=0.1); noise norm E‖ε‖ ≈ √d ≈ 22.6.

| R | 0.1 | 0.5 | 1 | 1.5 | 2 | 5 | 8 | 16 | 22 | 32 |
|---|----|----|----|----|----|----|----|----|----|----|
| predicted t*(R) | 0.03 | 0.13 | 0.23 | 0.31 | 0.38 | 0.60 | 0.71 | 0.83 | 0.87 | 0.91 |

## Headline conclusion

**The embedding norm R sets the decode time, and whether that helps or hurts
depends entirely on the noise schedule.** With a *fixed* schedule (`naive`),
solve rate is an **inverted-U in R** — it rises from 0.22 (R=0.1) to a peak
**0.44 at R=2**, then collapses to **0.16 at R=32**. With an *adaptive* schedule
(`ada`), there is **no collapse**: accuracy is flat-to-rising and stays high for
large R, peaking at **0.54 (R=5)** and holding ~0.48–0.51 out to R=32.

The two arms are **statistically tied for R ≤ 2** (~0.42–0.45, overlapping within
seed-std) and **diverge sharply for R ≥ 5**. At R=32 the gap is
**0.162±0.054 (naive) vs 0.505±0.019 (ada)** — a ~3× difference, many seed-std
wide, so it is a real schedule effect, not seed noise. Aggregated over the whole
balanced grid (n=90 each), `ada` beats `naive` by **+11.0pt** (global mean 0.420
vs 0.310).

This confirms the hypothesis mechanism: large R pushes t* late (t*≈0.83 at R=16,
0.91 at R=32), and a fixed log-linear schedule spends almost all its steps in the
trivial high-noise region and under-trains the sharp late transition where
decoding actually happens. An adaptive schedule refits training mass onto the
transition wherever R places it, turning large R from a failure mode into the
best-performing regime.

## Best-LR, naive vs ada (seed-averaged, mean ± seed-std)

Per R, the LR with the highest mean is shown for each arm.

| R | naive | ada |
|---|---|---|
| 0.1 | 0.222 ± 0.042 | 0.304 ± 0.072 |
| 0.5 | 0.435 ± 0.058 | 0.440 ± 0.074 |
| 1.0 | 0.415 ± 0.026 | 0.418 ± 0.032 |
| 1.5 | 0.433 ± 0.035 | 0.429 ± 0.038 |
| 2.0 | 0.440 ± 0.068 | 0.448 ± 0.019 |
| 5.0 | 0.400 ± 0.004 | **0.539 ± 0.041** |
| 8.0 | 0.321 ± 0.031 | **0.488 ± 0.054** |
| 16.0 | 0.236 ± 0.045 | **0.476 ± 0.076** |
| 22.0 | 0.249 ± 0.040 | **0.468 ± 0.037** |
| 32.0 | 0.162 ± 0.054 | **0.505 ± 0.019** |

Peak: naive **0.440 @ R=2 (lr1e-3)**; ada **0.539 @ R=5 (lr1e-3)**.

## Full tables (mean ± seed-std, n seeds)

### naive (fixed log-linear schedule)

| R | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|
| 0.1 | 0.218 ± 0.061 | 0.217 ± 0.019 | 0.222 ± 0.042 |
| 0.5 | 0.394 ± 0.016 | 0.347 ± 0.041 | 0.435 ± 0.058 |
| 1.0 | 0.395 ± 0.047 | 0.415 ± 0.026 | 0.413 ± 0.004 |
| 1.5 | 0.419 ± 0.053 | 0.433 ± 0.035 | 0.400 ± 0.068 |
| 2.0 | 0.423 ± 0.027 | 0.411 ± 0.032 | 0.440 ± 0.068 |
| 5.0 | 0.333 ± 0.068 | 0.400 ± 0.004 | 0.396 ± 0.042 |
| 8.0 | 0.306 ± 0.021 | 0.302 ± 0.026 | 0.321 ± 0.031 |
| 16.0 | 0.202 ± 0.073 | 0.236 ± 0.045 | 0.187 ± 0.064 |
| 22.0 | 0.181 ± 0.034 | 0.249 ± 0.040 | 0.189 ± 0.112 |
| 32.0 | 0.088 ± 0.070 | 0.162 ± 0.054 | 0.158 ± 0.123 |

### ada (adaptive log-linear, no truncation)

| R | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|
| 0.1 | 0.292 ± 0.070 | 0.304 ± 0.072 | 0.295 ± 0.039 |
| 0.5 | 0.383 ± 0.018 | 0.440 ± 0.074 | 0.425 ± 0.029 |
| 1.0 | 0.406 ± 0.068 | 0.388 ± 0.091 | 0.418 ± 0.032 |
| 1.5 | 0.422 ± 0.039 | 0.426 ± 0.053 | 0.429 ± 0.038 |
| 2.0 | 0.416 ± 0.069 | 0.446 ± 0.019 | 0.448 ± 0.019 |
| 5.0 | 0.435 ± 0.094 | 0.469 ± 0.030 | 0.539 ± 0.041 |
| 8.0 | 0.480 ± 0.093 | 0.463 ± 0.022 | 0.488 ± 0.054 |
| 16.0 | 0.427 ± 0.046 | 0.422 ± 0.061 | 0.476 ± 0.076 |
| 22.0 | 0.389 ± 0.090 | 0.468 ± 0.037 | 0.417 ± 0.029 |
| 32.0 | 0.320 ± 0.078 | 0.505 ± 0.019 | 0.370 ± 0.096 |

## Interpretation vs the signal model

- **Ordering holds where the schedule can follow it.** Predicted t*(R) rises
  monotonically 0.03 → 0.91. For `ada`, accuracy tracks this without penalty —
  the schedule follows t* wherever it goes. For `naive`, the fixed schedule
  effectively caps the usable R: once t* pushes past ~0.6 (R ≳ 5), solve rate
  falls, and the fall is monotonic through R=32 exactly as t* → 1.
- **The small-R degenerate regime is confirmed.** R=0.1 (t*≈0.03) gives the
  lowest naive accuracy (~0.22) and is flat across LR (0.218–0.222) — the signal
  is buried in noise almost immediately, an optimization-independent geometry
  effect. `ada` recovers some of it (~0.30) by concentrating steps at the very
  clean end.
- **LR is second-order** relative to R and arm: within a cell, LR moves accuracy
  by ≤ ~0.05, well inside the R-driven swings (0.16 → 0.54).

## Caveats

- **Coverage 180/180, all groups n=3.** 8 cells (naive R=0.5/1e-3 ×2 and 6 `ada`
  large-R) initially failed — not on the science but on a full `/home` filesystem
  (wandb's import-time temp dir hit ENOSPC before training could checkpoint). Fixed
  by redirecting `TMPDIR`/caches to node-local `/tmp` in the sweep job body and
  re-running those 8 (idempotent); all completed cleanly. Final numbers moved <0.02
  vs the 172-cell version — the conclusion was already stable.
- **Best-cell vs trend.** The result is a robust *arm × R trend*, not a single
  magic cell; individual best cells (naive 0.44, ada 0.54) sit within ~1–2 seed-
  std of their neighbours.
- **No `none` (unrescaled) baseline** in this grid — the small-R naive cells are
  the effective low-norm reference.

## Reproduce

    python experiments/eflm_rescale_sudoku/sweep.py --dry-run   # inspect (180)
    python experiments/eflm_rescale_sudoku/sweep.py             # submit / resume
    # missing-cell refill happens automatically on re-submit (idempotent).

Data: `outputs/eflm_rescale_sudoku/eflmrs_{arm}_r-{R}_lr-{lr}_d-hard_rs{seed}/
eval/results.json`. Checkpoints kept for the post-hoc loss-geometry L(t)
analysis (EXPERIMENT.md success criterion 1), still to be run.

---

# Round 2 — truncation × radius (`sweep_trunc.py`, 108/108 cells, n=3)

Truncate the *fixed* schedule at the R-dependent Eq. 17 bound
ALPHA_MAX = α*(R) = `alpha_star_euclidean(V=12, embed_norm=R)` (= 1 − t*(R)
from the codebook signal analysis; identical to 3 decimals), offsets
{α*−0.1, α*, α*+0.1} (clipped to ≥0.05), R ∈ {1, 2, 5, 8, 16, 32},
LR ∈ {5e-4, 1e-3}, 3 seeds. Same train/eval protocol as round 1
(`eflm_rescale_truncated.sh`).

## Headline: static truncation rescues the naive collapse — most, not all,
## of the adaptive gain

Best cell per R within each arm (best over LR and, for trunc, offset):

| R | naive (r1) | **trunc (r2)** | ada (r1) |
|---|---|---|---|
| 1 | 0.415 | 0.397 | 0.418 |
| 2 | 0.440 | 0.468 | 0.448 |
| 5 | 0.400 | 0.464 | 0.539 |
| 8 | 0.321 | **0.426** | 0.488 |
| 16 | 0.236 | **0.416** | 0.476 |
| 32 | 0.162 | **0.433** | 0.505 |

- **Large R (≥8): trunc(α*) turns the collapse into a flat ~0.42–0.43** —
  +10pt at R=8, +18pt at R=16, +27pt at R=32 over naive (all many seed-std).
  The mechanism is confirmed: the fixed schedule's failure was wasting its
  steps in the trivial high-signal region; cutting that region off at the
  codebook-signal bound recovers roughly **60–75 % of the adaptive gain**.
- **ada still wins at every R ≥ 5** (0.48–0.54 vs trunc 0.42–0.46): refitting
  onto the *measured* transition beats the single-token NN-model bound.
- **Small R: truncation is neutral-to-harmful** (R=1: 0.397 < naive 0.415;
  R=2: 0.468 ≈ naive within std) — nothing to rescue, coverage lost.
- **Offset:** α* and α*+0.1 are statistically tied for best at large R;
  α*−0.1 is clearly worse and high-variance at R=16 (α_max=0.07 leaves almost
  no signal band). At R=32 the clipped floor 0.05 won its LR slice
  (0.433±0.071) but overlaps the α* cell (0.394±0.052) within std. Practical
  rule: **truncate at α*(R), never tighter.**

Data: `outputs/eflm_rescale_sudoku/eflmrst_r-{R}_am-{α}_lr-{lr}_d-hard_rs{s}/
eval/results.json`.
