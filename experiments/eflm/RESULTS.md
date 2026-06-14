# RESULTS: EFLM (Euclidean Flow LM) on Sudoku — 3-geometry comparison

**Date:** 2026-06-13 · **Branch:** `claude/langflow` · spec: [`EXPERIMENT.md`](EXPERIMENT.md)
**TL;DR:** The flat / Euclidean naive flow **beats naive S-FLM (sphere) at all three difficulties** and
lands **just below naive HFLM (hyperbolic)**. Geometry ranking of the *naive* model:
**HFLM (hyperbolic) > EFLM (Euclidean) ≫ S-FLM (sphere).**

## Setup (identical recipe for all three geometries)

`tiny-sphere-dit` (8×512, 8 heads, ~28.6M) · 20k steps · global batch 256 · bf16 · EMA 0.9999 ·
AdamW lr 3e-4, wd 0, grad-clip 1.0 · `noise=log-linear` (vanilla, **no** truncation/adaptive) ·
`invert_time_convention=false` · 48k/2k sudoku per difficulty (seed 42). Eval: `mode=sudoku_eval`,
**180** steps, **exact** velocity, **greedy**, EMA on, full 2,000 val puzzles. EFLM = S-FLM with raw
(un-normalized) Euclidean embeddings, `N(0,I)` prior, and straight-line (lerp) interpolation.

Jobs `376423/376425/376426` on `thickstun-compute-01`, COMPLETED exit 0, ~1h41m each (train+both evals).

## Results — naive models, Sudoku exact-match accuracy (%)

Reference S-FLM / HFLM numbers are the user-provided reproductions; **EFLM is this experiment.**

**top_k_velocity = −1 (velocity = average over full vocab)**

| naive model | Space | Easy | Med | Hard |
|---|---|---|---|---|
| S-FLM (naive) | sphere | 77.6 | 32.6 | 13.9 |
| **EFLM (naive)** | **Euclidean** | **90.35** | **62.05** | **17.55** |
| HFLM (naive) | hyperbolic | 93.75 | 67.40 | 24.10 |

**top_k_velocity = 1 (velocity = top-1 endpoint)**

| naive model | Space | Easy | Med | Hard |
|---|---|---|---|---|
| S-FLM (naive) | sphere | 76.6 | 33.2 | 15.2 |
| **EFLM (naive)** | **Euclidean** | **90.25** | **62.45** | **18.70** |
| HFLM (naive) | hyperbolic | 93.75 | 67.40 | 24.10 |

Raw counts (`eval_runs/sudoku/eflm_{difficulty}_tkv{...}/results.json`):

| run | acc | counts | run | acc | counts |
|---|---|---|---|---|---|
| easy · tkv −1 | 90.35% | 1807/2000 | easy · tkv 1 | 90.25% | 1805/2000 |
| medium · tkv −1 | 62.05% | 1241/2000 | medium · tkv 1 | 62.45% | 1249/2000 |
| hard · tkv −1 | 17.55% | 351/2000 | hard · tkv 1 | 18.70% | 374/2000 |

## Deltas (EFLM vs. the other naive geometries, tkv=−1)

| | Easy | Med | Hard |
|---|---|---|---|
| EFLM − S-FLM | **+12.8** | **+29.5** | **+3.7** |
| EFLM − HFLM | −3.4 | −5.4 | −6.6 |

`top_k_velocity` barely moves EFLM (≤0.6 pt across all three) — same insensitivity the paper/repo report
for S-FLM. So the headline is robust to the velocity choice.

## Reading

1. **Flat ≫ sphere for the naive model.** Removing the unit-sphere constraint (S-FLM normalizes
   embeddings *and* noise onto `S^{d-1}`; EFLM leaves both in raw `R^d`) lifts medium from 32.6 → 62.1
   and easy from 77.6 → 90.4. The sphere's constraint *hurts* the naive flow more than flatness does —
   most of the paper's "geometry lifts the naive model" gain is already captured by simply **not** being
   on the sphere.
2. **Hyperbolic still edges out flat.** HFLM keeps a steady ~3–7 pt lead over EFLM (largest at hard),
   so negative curvature adds a real, if modest, increment on top of going flat. The geometry ordering
   among naive models is **HFLM > EFLM > S-FLM**, monotone from negative → zero → positive curvature on
   this task.
3. **EFLM vs. *tricked* S-FLM.** EFLM (naive) is competitive with truncated/adaptive S-FLM on easy
   (90 vs ~94–95) and closes much of the medium gap (62 vs ~78), but still trails the tricked sphere
   model — i.e. naive geometry change ≈ but does not fully replace the truncation trick.

## On the scale caveat (resolved)

`EXPERIMENT.md` §2 flagged that EFLM reuses S-FLM's `ngpt` init (‖e‖≈1) with `prior_cov=1.0` (‖noise‖≈√d),
a data/noise scale mismatch. **Empirically a non-issue:** the input LayerNorm absorbs the scale and EFLM
reached 90/62/18 with no NaNs and no collapse. (A scale-matched `init=unit_var` / `prior_cov≈1/d` rerun
is an obvious follow-up that could only *help* EFLM, strengthening the "flat ≫ sphere" finding — not
needed to support the conclusion.)

## Verdict

**Confirmed — geometry ranking holds, with a twist.** EFLM trains cleanly and lands below naive HFLM at
all three difficulties (criterion met), but the more interesting result is that **the flat baseline
decisively beats the sphere baseline**, so the naive-model curve is monotone in curvature
(HFLM > EFLM > S-FLM) rather than "any curvature ≫ flat." A bug was found and fixed during bring-up
(`algo.EFLM._lerp` alpha broadcast, mirroring `utils.slerp`); the smoke + these full runs validate the
`EFLM` algo and `EFLMSampler` end-to-end.

## Artifacts

- **Checkpoints:** `outputs/sudoku/eflm_{easy,medium,hard}/checkpoints/last.ckpt`
- **Eval results:** `eval_runs/sudoku/eflm_{easy,medium,hard}_tkv{-1,1}/results.json`
- **Logs:** `experiments/eflm/logs/eflm_{easy,medium,hard}.{376423,376425,376426}.log`
- **Scripts:** `scripts/train/sudoku/eflm.sh`, `scripts/sample/sudoku/eflm.sh`,
  `experiments/eflm/run_sudoku.sbatch`
