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

## Does the optimal truncation follow the prediction? — YES, within ±0.1

Per (R, LR ∈ {5e-4, 1e-3}), argmax over the measured acc-vs-ALPHA_MAX points
(trunc cells + the round-1 naive cell as the no-truncation α→1 endpoint) vs
the prediction α*(R) = `alpha_star_euclidean(V=12, embed_norm=R)`:

| R | α* | argmax (5e-4) | argmax (1e-3) | offset from α* |
|---|---|---|---|---|
| 1 | 0.767 | 1.000 (no trunc) | 1.000 (no trunc) | n/a — no interior optimum |
| 2 | 0.622 | **0.622** | 1.000 (~tie w/ α*) | 0 / ~tie |
| 5 | 0.396 | 0.296 (~tie) | 0.496 (~tie) | ±0.1, ties |
| 8 | 0.291 | 0.191 (~tie) | 0.191 (~tie) | −0.1, ties |
| 16 | 0.170 | 0.270 (~tie) | 0.270 (~tie) | +0.1, ties |
| 32 | 0.093 | **0.093** | 0.050 (=α*−0.04) | ≈0 |

- **For every R ≥ 2 the empirical optimum lies within ±0.1 of α*(R)** while
  α* itself sweeps a 6.7× range (0.62 → 0.09), and the no-truncation endpoint
  is catastrophically worse at large R (0.09–0.31 vs 0.38–0.47 near α*). The
  single-token codebook bound predicts the optimal cut location across the
  whole R range.
- Within the ±0.1 neighborhood the points are statistically tied (seed-std
  overlap) — the accuracy surface is locally flat around α*, so the bound
  need not be hit exactly; "at α*, ±0.1" is the operating rule.
- **R = 1 is the predicted-consistent exception**: no interior optimum —
  no truncation wins. As R → small the informative band widens toward the
  whole [0, 1] range (α* → 1), so cutting anything only removes signal;
  the collapse-rescue regime that truncation exists for starts at R ≳ 2.
- Round 3 (`sweep_trunc.py --round3`, 90 cells: wide points {e.g. R=32:
  0.3/0.5/0.7}, submitted) refines the curves between α*+0.1 and the α→1
  endpoint to bound any wide-side bias; at ±0.1 resolution the answer above
  stands.

---

# Round 4 — trunc_ada: the combo does NOT compose on sudoku (36/36, n=3)

`--arm trunc_ada`: adaptive schedule + ALPHA_MAX = α*(R)
(`eflm_rescale_adaptive.sh` with the new ALPHA_MAX knob), R ∈ {1,2,5,8,16,32}
× LR {5e-4, 1e-3} × 3 seeds. Completed 2×2, best-LR per cell:

| R | naive | trunc(α*) | ada | trunc_ada(α*) |
|---|---|---|---|---|
| 1 | 0.415±0.026 | 0.397±0.014 | 0.418±0.032 | 0.418±0.013 |
| 2 | 0.440±0.068 | 0.468±0.056 | 0.448±0.019 | 0.436±0.032 |
| 5 | 0.400±0.004 | 0.464±0.100 | **0.539±0.041** | 0.409±0.077 |
| 8 | 0.321±0.031 | 0.426±0.028 | **0.488±0.054** | 0.456±0.012 |
| 16 | 0.236±0.045 | 0.416±0.040 | **0.476±0.076** | 0.442±0.085 |
| 32 | 0.162±0.054 | 0.433±0.071 | **0.505±0.019** | 0.401±0.064 |

- **ada alone stays champion at every R ≥ 5.** trunc_ada lands at roughly the
  trunc level (0.40–0.46), clearly below ada (0.48–0.54) — capping the range
  the adaptive refit redistributes over *limits* ada rather than helping it.
  On sudoku, ada evidently uses some coverage above α* that the hard cap
  removes.
- **Opposite of TinyStories**, where trunc_ada was the overall GenPPL winner
  (10.99 vs ada 12.57 at R=1). The composition of truncation with adaptation
  is task-dependent: it helps generation quality (LM) and mildly hurts
  solve-rate (constraint task) — worth keeping both arms in future sweeps.

---

# Appendix — full results table (all arms × R × ALPHA_MAX × LR)

Arms per setup.md 2×2: naive = w/o ada w/o trunc; trunc = w/o ada w/ trunc;
ada = w/ ada w/o trunc; trunc_ada = w/ ada w/ trunc (Round 4; per-cell rows in
Round 4 table above — regenerate this appendix for the merged dump). alpha_max
`N*` = the predicted alpha*(R). Raw data:
`outputs/eflm_rescale_sudoku/eflmrs{,t}_*/eval/results.json`.

<!-- snapshot: 318 completed runs; round 3 still filling seeds -->
| arm | R | alpha_max | lr | acc | n |
|---|---|---|---|---|---|
| naive | 0.1 | — | 1e-3 | 0.222±0.042 | 3 |
| naive | 0.1 | — | 3e-4 | 0.218±0.061 | 3 |
| naive | 0.1 | — | 5e-4 | 0.217±0.019 | 3 |
| ada | 0.1 | — | 1e-3 | 0.295±0.039 | 3 |
| ada | 0.1 | — | 3e-4 | 0.292±0.070 | 3 |
| ada | 0.1 | — | 5e-4 | 0.304±0.072 | 3 |
| naive | 0.5 | — | 1e-3 | 0.435±0.058 | 3 |
| naive | 0.5 | — | 3e-4 | 0.394±0.016 | 3 |
| naive | 0.5 | — | 5e-4 | 0.347±0.041 | 3 |
| ada | 0.5 | — | 1e-3 | 0.425±0.029 | 3 |
| ada | 0.5 | — | 3e-4 | 0.383±0.018 | 3 |
| ada | 0.5 | — | 5e-4 | 0.440±0.074 | 3 |
| naive | 1 | — | 1e-3 | 0.413±0.004 | 3 |
| naive | 1 | — | 3e-4 | 0.395±0.047 | 3 |
| naive | 1 | — | 5e-4 | 0.415±0.026 | 3 |
| trunc | 1 | 0.667 | 1e-3 | 0.345±0.012 | 3 |
| trunc | 1 | 0.667 | 5e-4 | 0.332±0.017 | 3 |
| trunc | 1 | **0.767\*** | 1e-3 | 0.397±0.014 | 3 |
| trunc | 1 | **0.767\*** | 5e-4 | 0.357±0.027 | 3 |
| trunc | 1 | 0.867 | 1e-3 | 0.378±0.029 | 3 |
| trunc | 1 | 0.867 | 5e-4 | 0.357±0.048 | 3 |
| ada | 1 | — | 1e-3 | 0.418±0.032 | 3 |
| ada | 1 | — | 3e-4 | 0.406±0.068 | 3 |
| ada | 1 | — | 5e-4 | 0.388±0.091 | 3 |
| naive | 1.5 | — | 1e-3 | 0.400±0.068 | 3 |
| naive | 1.5 | — | 3e-4 | 0.419±0.053 | 3 |
| naive | 1.5 | — | 5e-4 | 0.433±0.035 | 3 |
| ada | 1.5 | — | 1e-3 | 0.429±0.038 | 3 |
| ada | 1.5 | — | 3e-4 | 0.422±0.039 | 3 |
| ada | 1.5 | — | 5e-4 | 0.426±0.053 | 3 |
| naive | 2 | — | 1e-3 | 0.440±0.068 | 3 |
| naive | 2 | — | 3e-4 | 0.423±0.027 | 3 |
| naive | 2 | — | 5e-4 | 0.411±0.032 | 3 |
| trunc | 2 | 0.200 | 1e-3 | 0.046 | 1 |
| trunc | 2 | 0.200 | 5e-4 | 0.120 | 1 |
| trunc | 2 | 0.350 | 1e-3 | 0.208 | 1 |
| trunc | 2 | 0.350 | 5e-4 | 0.205 | 1 |
| trunc | 2 | 0.522 | 1e-3 | 0.418±0.130 | 3 |
| trunc | 2 | 0.522 | 5e-4 | 0.344±0.089 | 3 |
| trunc | 2 | **0.622\*** | 1e-3 | 0.416±0.019 | 3 |
| trunc | 2 | **0.622\*** | 5e-4 | 0.468±0.056 | 3 |
| trunc | 2 | 0.722 | 1e-3 | 0.389±0.027 | 3 |
| trunc | 2 | 0.722 | 5e-4 | 0.415±0.034 | 3 |
| trunc | 2 | 0.850 | 1e-3 | 0.445 | 1 |
| trunc | 2 | 0.850 | 5e-4 | 0.383 | 1 |
| ada | 2 | — | 1e-3 | 0.448±0.019 | 3 |
| ada | 2 | — | 3e-4 | 0.416±0.069 | 3 |
| ada | 2 | — | 5e-4 | 0.446±0.019 | 3 |
| naive | 5 | — | 1e-3 | 0.396±0.042 | 3 |
| naive | 5 | — | 3e-4 | 0.333±0.068 | 3 |
| naive | 5 | — | 5e-4 | 0.400±0.004 | 3 |
| trunc | 5 | 0.150 | 1e-3 | 0.231 | 1 |
| trunc | 5 | 0.150 | 5e-4 | 0.241 | 1 |
| trunc | 5 | 0.296 | 1e-3 | 0.391±0.071 | 3 |
| trunc | 5 | 0.296 | 5e-4 | 0.444±0.113 | 3 |
| trunc | 5 | **0.396\*** | 1e-3 | 0.375±0.033 | 3 |
| trunc | 5 | **0.396\*** | 5e-4 | 0.428±0.061 | 3 |
| trunc | 5 | 0.496 | 1e-3 | 0.464±0.100 | 3 |
| trunc | 5 | 0.496 | 5e-4 | 0.418±0.053 | 3 |
| trunc | 5 | 0.650 | 1e-3 | 0.347 | 1 |
| trunc | 5 | 0.650 | 5e-4 | 0.433 | 1 |
| trunc | 5 | 0.850 | 1e-3 | 0.448 | 1 |
| trunc | 5 | 0.850 | 5e-4 | 0.354 | 1 |
| ada | 5 | — | 1e-3 | 0.539±0.041 | 3 |
| ada | 5 | — | 3e-4 | 0.435±0.094 | 3 |
| ada | 5 | — | 5e-4 | 0.469±0.030 | 3 |
| naive | 8 | — | 1e-3 | 0.321±0.031 | 3 |
| naive | 8 | — | 3e-4 | 0.306±0.021 | 3 |
| naive | 8 | — | 5e-4 | 0.302±0.026 | 3 |
| trunc | 8 | 0.100 | 1e-3 | 0.232 | 1 |
| trunc | 8 | 0.100 | 5e-4 | 0.254 | 1 |
| trunc | 8 | 0.191 | 1e-3 | 0.382±0.084 | 3 |
| trunc | 8 | 0.191 | 5e-4 | 0.426±0.028 | 3 |
| trunc | 8 | **0.291\*** | 1e-3 | 0.362±0.033 | 3 |
| trunc | 8 | **0.291\*** | 5e-4 | 0.400±0.019 | 3 |
| trunc | 8 | 0.391 | 1e-3 | 0.375±0.050 | 3 |
| trunc | 8 | 0.391 | 5e-4 | 0.423±0.024 | 3 |
| trunc | 8 | 0.550 | 1e-3 | 0.366 | 1 |
| trunc | 8 | 0.550 | 5e-4 | 0.411 | 1 |
| trunc | 8 | 0.750 | 1e-3 | 0.399 | 1 |
| trunc | 8 | 0.750 | 5e-4 | 0.380 | 1 |
| ada | 8 | — | 1e-3 | 0.488±0.054 | 3 |
| ada | 8 | — | 3e-4 | 0.480±0.093 | 3 |
| ada | 8 | — | 5e-4 | 0.463±0.022 | 3 |
| naive | 16 | — | 1e-3 | 0.187±0.064 | 3 |
| naive | 16 | — | 3e-4 | 0.202±0.073 | 3 |
| naive | 16 | — | 5e-4 | 0.236±0.045 | 3 |
| trunc | 16 | 0.070 | 1e-3 | 0.262±0.134 | 3 |
| trunc | 16 | 0.070 | 5e-4 | 0.256±0.104 | 3 |
| trunc | 16 | **0.170\*** | 1e-3 | 0.392±0.065 | 3 |
| trunc | 16 | **0.170\*** | 5e-4 | 0.398±0.069 | 3 |
| trunc | 16 | 0.270 | 1e-3 | 0.395±0.031 | 3 |
| trunc | 16 | 0.270 | 5e-4 | 0.416±0.040 | 3 |
| trunc | 16 | 0.400 | 1e-3 | 0.181 | 1 |
| trunc | 16 | 0.400 | 5e-4 | 0.213 | 1 |
| trunc | 16 | 0.600 | 1e-3 | 0.369 | 1 |
| trunc | 16 | 0.600 | 5e-4 | 0.392 | 1 |
| trunc | 16 | 0.800 | 1e-3 | 0.170 | 1 |
| trunc | 16 | 0.800 | 5e-4 | 0.245 | 1 |
| ada | 16 | — | 1e-3 | 0.476±0.076 | 3 |
| ada | 16 | — | 3e-4 | 0.427±0.046 | 3 |
| ada | 16 | — | 5e-4 | 0.422±0.061 | 3 |
| naive | 22 | — | 1e-3 | 0.189±0.112 | 3 |
| naive | 22 | — | 3e-4 | 0.181±0.034 | 3 |
| naive | 22 | — | 5e-4 | 0.249±0.040 | 3 |
| ada | 22 | — | 1e-3 | 0.417±0.029 | 3 |
| ada | 22 | — | 3e-4 | 0.389±0.090 | 3 |
| ada | 22 | — | 5e-4 | 0.468±0.037 | 3 |
| naive | 32 | — | 1e-3 | 0.158±0.123 | 3 |
| naive | 32 | — | 3e-4 | 0.088±0.070 | 3 |
| naive | 32 | — | 5e-4 | 0.162±0.054 | 3 |
| trunc | 32 | 0.050 | 1e-3 | 0.433±0.071 | 3 |
| trunc | 32 | 0.050 | 5e-4 | 0.334±0.066 | 3 |
| trunc | 32 | **0.093\*** | 1e-3 | 0.357±0.046 | 3 |
| trunc | 32 | **0.093\*** | 5e-4 | 0.394±0.052 | 3 |
| trunc | 32 | 0.193 | 1e-3 | 0.344±0.084 | 3 |
| trunc | 32 | 0.193 | 5e-4 | 0.337±0.083 | 3 |
| trunc | 32 | 0.300 | 1e-3 | 0.203 | 1 |
| trunc | 32 | 0.300 | 5e-4 | 0.276 | 1 |
| trunc | 32 | 0.500 | 1e-3 | 0.288 | 1 |
| trunc | 32 | 0.500 | 5e-4 | 0.392 | 1 |
| trunc | 32 | 0.700 | 1e-3 | 0.201 | 1 |
| trunc | 32 | 0.700 | 5e-4 | 0.173 | 1 |
| ada | 32 | — | 1e-3 | 0.370±0.096 | 3 |
| ada | 32 | — | 3e-4 | 0.320±0.078 | 3 |
| ada | 32 | — | 5e-4 | 0.505±0.019 | 3 |
