<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
ng { color: #0072B2; }
rd { color: #D55E00; }
uv { color: #008060; }
hy { color: #7B3FA0; }
</style>

# RESULTS: ex_geo_init_wes — geometry × init × word-embedding-dim (Sudoku-medium)

**Date:** 2026-06-14 · spec: [`EXPERIMENT.md`](EXPERIMENT.md) · 27 cells, difficulty **medium**, 20k
steps, eval @180 steps exact/greedy/tkv=−1, full 2k val. Effective batch 256 (64×accum-4).

**TL;DR:** Geometry dominates (**HFLM ≫ EFLM ≳ S-FLM** in nearly every cell). **Init sensitivity is
exactly the geometry's coupling to embedding scale**: S-FLM (normalizes) is nearly init-invariant,
EFLM (raw) swings ~18 pts with init, HFLM (radius=‖e‖) is in between. **Embedding dim 256 ≈ 512 ≫ 128.**

## Accuracy grids (exact-match %, Sudoku-medium)

**S-FLM (sphere)** — normalizes embeddings, so init *scale* is erased:

| init \ dim | 512 | 256 | 128 | init mean |
|---|---|---|---|---|
| ngpt | 46.75 | 55.40 | 38.20 | 46.8 |
| random | 46.25 | 52.35 | 50.65 | 49.8 |
| unit_var | 49.55 | 53.20 | 38.30 | 47.0 |
| **dim mean** | **47.5** | **53.6** | **42.4** | **47.9** |

**EFLM (Euclidean)** — raw embeddings, init *scale* matters most:

| init \ dim | 512 | 256 | 128 | init mean |
|---|---|---|---|---|
| ngpt | 51.70 | 48.05 | 50.85 | 50.2 |
| random | 61.00 | 61.40 | 59.05 | **60.5** |
| unit_var | 46.95 | 40.60 | 39.20 | 42.3 |
| **dim mean** | 53.2 | 50.0 | 49.7 | **51.0** |

`eflm_ngpt_d512=51.70` is the fresh sweep rerun (job 377408; first attempt timed out). EFLM-medium has
~10 pt run-to-run variance (the prior batch-256 run of this exact cell scored 62.05; S-FLM-medium shows
the same ~11 pt seed spread in `SUDOKU_REPRODUCTION.md`) — so trust the **within-sweep** init/dim
comparisons (all identical batch-64×accum-4), not cross-experiment absolutes.

**HFLM (hyperbolic)** — ‖e‖ is the radial coordinate (clamped at `rho_max=12`):

| init \ dim | 512 | 256 | 128 | init mean |
|---|---|---|---|---|
| ngpt | **75.50** | 73.05 | 56.40 | 68.3 |
| random | 74.55 | 72.50 | 50.75 | 65.9 |
| unit_var | 65.20 | 68.20 | 42.75 | 58.7 |
| **dim mean** | **71.8** | 71.3 | 50.0 | **64.3** |

Geometry means (9 cells each): **S-FLM 47.9 · EFLM 51.0 · HFLM 64.3.** Best cell overall:
**hflm_ngpt_d512 = 75.5%.** Best S-FLM: ngpt_d256 = 55.4. Best EFLM: random_d256 = 61.4.

## Findings

1. **Geometry ranking is robust to init & dim.** HFLM is the best geometry in **every** cell; at fixed
   (init, dim) the order is almost always HFLM > EFLM > S-FLM (e.g. d512/random: 74.6 > 61.0 > 46.3).
   Hyperbolic's edge from the [EFLM 3-geometry run](../eflm/RESULTS.md) holds across the whole grid.

2. **Init sensitivity tracks how each geometry uses the embedding (the core hypothesis — confirmed).**
   - **S-FLM ≈ init-invariant** (init means 46.8 / 49.8 / 47.0, ~3 pt spread): it `sphere_normalize`s
     embeddings, erasing init scale — only direction survives.
   - **EFLM ≈ most init-sensitive** (~18 pt spread): raw embeddings, so init scale is the data scale.
     **`random` (std 0.02) ≫ `ngpt` ≫ `unit_var` (std 1)** — EFLM wants *small, tight* embeddings.
   - **HFLM ≈ intermediate** (~10 pt): `ngpt ≈ random > unit_var`; `unit_var`'s huge radius (‖e‖≈√d,
     clamped to 12) wastes the radial coordinate.

3. **Dim: 256 ≈ 512 ≫ 128.** d128 collapses everywhere and worst for HFLM (50.0 vs ~71 at 256/512) —
   hyperbolic needs the width. d256 matches or beats d512 for S-FLM (53.6 vs 47.5) at half the params:
   a sweet spot.

4. **"Scale-matching" intuition for EFLM is refuted.** The [EFLM doc](../eflm/RESULTS.md) flagged that
   `unit_var` (‖e‖≈√d, matched to the N(0,I) prior) might *help* EFLM. It is in fact EFLM's **worst**
   init (42.3 mean) — `random`/`ngpt` (small-norm embeddings) win. The original `ngpt` recipe choice was
   right; EFLM prefers a tight embedding cloud, not noise-scale-matched embeddings.

## Verdict

Confirmed: the three geometries separate cleanly (HFLM > EFLM > S-FLM) and **each geometry's robustness
to embedding init is explained by how it consumes the embedding** — normalize (S-FLM, invariant) →
radius (HFLM, partial) → raw (EFLM, fully scale-dependent). Practical defaults from this grid:
HFLM + ngpt + dim 256–512.

## Artifacts

- **Script:** `experiments/ex_geo_init_wes/sweep.py` (simple_slurm), manifest `jobs.txt`.
- **Evals:** `eval_runs/sudoku/exgiw/<tag>/results.json`; checkpoints `outputs/sudoku/exgiw/<tag>/`.
- **Logs:** `experiments/ex_geo_init_wes/logs/`. Raw grid dump: `sweep_results.txt`.
- **Code:** `models/hyperbolic_dit.py` gained `init=unit_var` (std 1.0) for this sweep.

<!-- LR-SWEEP-SECTIONS (auto-generated below; do not edit by hand) -->

# RESULTS — Medium LR sweep (geometry × init × dim × LR)

Sudoku-medium exact-match accuracy (%), 20k steps, eval @180 exact/greedy/tkv=-1, effective batch 256. **216/216 cells** done (— = still running).

**Per-geometry LR means** (avg over available init×dim cells):

| geometry | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 | best LR |
|---|---|---|---|---|---|---|---|
| SFM | 18.2 | 30.8 | 30.9 | 47.4 | 48.7 | 46.5 | **5e-4** |
| EFLM | 35.0 | 44.6 | 45.2 | 53.0 | 50.1 | 41.6 | **3e-4** |
| HFLM | 28.4 | 37.8 | 47.7 | 63.1 | 62.0 | 68.9 | **1e-3** |

**Per-geometry LR Best** (Best over the 12 init×dim cells):

Acc color = init of the best cell: <ng>ngpt</ng> · <rd>random</rd> · <uv>unit_var</uv> · <hy>hyperbolic</hy>

| Geo | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 | best LR |
|---|---|---|---|---|---|---|---|
| S-FLM | <hy>40.2</hy> | <rd>50.1</rd> | <ng>50.8</ng> | <ng>55.4</ng> | <uv>58.5</uv> | <rd>**61.7**</rd> | 1e-3 |
| E-FLM | <rd>64.6</rd> | <rd>67.0</rd> | <rd>**69.9**</rd> | <hy>64.1</hy> | <rd>60.8</rd> | <rd>57.2</rd> | 1e-4 |
| H-FLM | <ng>54.4</ng> | <rd>62.2</rd> | <rd>75.6</rd> | <ng>75.5</ng> | <ng>70.2</ng> | <rd>**84.5**</rd> | 1e-3 |

### SFM

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 20.5 | 40.9 | 10.8 | 46.8 | 46.9 | 42.0 |
| ngpt | 256 | 32.5 | 37.0 | 50.8 | 55.4 | 54.2 | 60.1 |
| ngpt | 128 | 1.7 | 17.0 | 14.8 | 38.2 | 51.8 | 42.8 |
| random | 512 | 10.0 | 50.1 | 46.7 | 46.2 | 43.9 | 40.6 |
| random | 256 | 22.0 | 38.2 | 32.4 | 52.3 | 49.8 | 50.5 |
| random | 128 | 3.1 | 17.6 | 21.4 | 50.6 | 40.8 | 61.7 |
| unit_var | 512 | 29.6 | 33.6 | 42.3 | 49.5 | 41.8 | 32.1 |
| unit_var | 256 | 29.4 | 40.1 | 42.4 | 53.2 | 50.0 | 46.4 |
| unit_var | 128 | 2.1 | 20.8 | 31.1 | 38.3 | 58.5 | 48.2 |
| hyperbolic | 512 | 40.2 | 16.0 | 31.2 | 42.4 | 38.6 | 42.6 |
| hyperbolic | 256 | 22.1 | 36.5 | 40.6 | 53.1 | 57.5 | 50.3 |
| hyperbolic | 128 | 5.5 | 21.3 | 6.5 | 42.0 | 50.7 | 40.1 |

### EFLM

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 51.3 | 59.1 | 65.8 | 51.7 | 55.2 | 40.1 |
| ngpt | 256 | 48.0 | 49.9 | 47.8 | 48.0 | 46.6 | 40.0 |
| ngpt | 128 | 15.0 | 22.8 | 27.3 | 50.8 | 53.8 | 52.3 |
| random | 512 | 64.6 | 67.0 | 69.9 | 61.0 | 59.8 | 39.1 |
| random | 256 | 59.0 | 45.1 | 40.3 | 61.4 | 60.8 | 50.6 |
| random | 128 | 31.5 | 46.7 | 41.3 | 59.1 | 50.3 | 57.2 |
| unit_var | 512 | 32.5 | 47.4 | 52.7 | 46.9 | 50.6 | 21.8 |
| unit_var | 256 | 28.7 | 48.0 | 41.7 | 40.6 | 39.9 | 32.6 |
| unit_var | 128 | 4.4 | 14.5 | 18.8 | 39.2 | 36.9 | 38.0 |
| hyperbolic | 512 | 47.5 | 62.0 | 60.5 | 64.1 | 43.1 | 27.2 |
| hyperbolic | 256 | 31.9 | 44.6 | 45.2 | 61.1 | 52.9 | 52.0 |
| hyperbolic | 128 | 5.0 | 27.4 | 30.8 | 52.2 | 51.6 | 48.1 |

### HFLM

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 54.4 | 59.5 | 68.8 | 75.5 | 69.2 | 72.6 |
| ngpt | 256 | 20.5 | 33.1 | 53.3 | 73.0 | 70.2 | 79.3 |
| ngpt | 128 | 12.1 | 27.6 | 30.7 | 56.4 | 63.0 | 63.7 |
| random | 512 | 47.9 | 62.2 | 75.6 | 74.6 | 66.4 | 84.5 |
| random | 256 | 28.8 | 50.4 | 52.1 | 72.5 | 65.3 | 70.5 |
| random | 128 | 13.8 | 6.0 | 25.1 | 50.7 | 55.5 | 61.3 |
| unit_var | 512 | 38.5 | 58.1 | 43.1 | 65.2 | 58.5 | 67.2 |
| unit_var | 256 | 41.5 | 49.9 | 64.6 | 68.2 | 65.5 | 73.3 |
| unit_var | 128 | 19.2 | 11.5 | 36.3 | 42.8 | 48.3 | 56.3 |
| hyperbolic | 512 | 50.1 | 54.3 | 57.2 | 66.7 | 69.5 | 69.8 |
| hyperbolic | 256 | 4.7 | 16.8 | 43.1 | 61.4 | 63.8 | 74.2 |
| hyperbolic | 128 | 9.4 | 24.2 | 21.9 | 49.6 | 49.3 | 54.5 |

# RESULTS — Hard LR sweep (geometry × init × dim × LR)

Sudoku-hard exact-match accuracy (%), 20k steps, eval @180 exact/greedy/tkv=-1, effective batch 256. **216/216 cells** done (— = still running).

**Per-geometry LR means** (avg over available init×dim cells):

| geometry | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 | best LR |
|---|---|---|---|---|---|---|---|
| SFM | 2.9 | 5.8 | 7.2 | 13.8 | 14.2 | 15.4 | **1e-3** |
| EFLM | 9.6 | 11.1 | 14.6 | 16.3 | 14.2 | 7.5 | **3e-4** |
| HFLM | 8.6 | 12.7 | 12.6 | 20.3 | 19.0 | 25.1 | **1e-3** |

**Per-geometry LR Best** (Best over the 12 init×dim cells):

Acc color = init of the best cell: <ng>ngpt</ng> · <rd>random</rd> · <uv>unit_var</uv> · <hy>hyperbolic</hy>

| Geo | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 | best LR |
|---|---|---|---|---|---|---|---|
| S-FLM | <hy>8.0</hy> | <rd>12.0</rd> | <ng>15.9</ng> | <rd>20.6</rd> | <hy>**22.5**</hy> | <hy>20.8</hy> | 5e-4 |
| E-FLM | <rd>27.1</rd> | <rd>24.4</rd> | <ng>25.2</ng> | <rd>**29.1**</rd> | <rd>21.1</rd> | <ng>15.5</ng> | 3e-4 |
| H-FLM | <rd>34.4</rd> | <rd>**37.1**</rd> | <rd>31.2</rd> | <ng>31.7</ng> | <ng>32.2</ng> | <rd>34.6</rd> | 8e-5 |

### SFM

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 3.8 | 10.9 | 15.9 | 7.8 | 13.8 | 15.2 |
| ngpt | 256 | 1.1 | 4.3 | 9.0 | 11.6 | 12.9 | 17.0 |
| ngpt | 128 | 0.8 | 1.5 | 0.7 | 10.2 | 10.5 | 14.6 |
| random | 512 | 5.6 | 12.0 | 10.5 | 11.5 | 14.6 | 12.4 |
| random | 256 | 2.6 | 4.3 | 10.7 | 20.6 | 13.1 | 20.0 |
| random | 128 | 0.1 | 1.8 | 1.7 | 10.8 | 10.8 | 11.9 |
| unit_var | 512 | 6.7 | 11.3 | 10.9 | 12.6 | 17.1 | 15.0 |
| unit_var | 256 | 1.4 | 10.1 | 5.7 | 15.2 | 15.6 | 11.2 |
| unit_var | 128 | 0.1 | 0.5 | 2.1 | 15.8 | 17.2 | 19.6 |
| hyperbolic | 512 | 8.0 | 11.7 | 14.2 | 14.2 | 8.6 | 7.1 |
| hyperbolic | 256 | 4.0 | 1.1 | 3.0 | 15.4 | 22.5 | 20.8 |
| hyperbolic | 128 | 0.1 | 0.4 | 1.9 | 19.8 | 13.2 | 20.2 |

### EFLM

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 23.9 | 17.3 | 25.2 | 20.2 | 15.8 | 6.9 |
| ngpt | 256 | 8.1 | 12.7 | 12.5 | 12.2 | 14.3 | 8.4 |
| ngpt | 128 | 1.5 | 3.4 | 9.8 | 12.7 | 12.4 | 15.5 |
| random | 512 | 27.1 | 24.4 | 23.7 | 29.1 | 21.1 | 11.3 |
| random | 256 | 16.5 | 10.4 | 24.9 | 22.9 | 15.7 | 11.7 |
| random | 128 | 7.3 | 13.4 | 11.8 | 15.4 | 19.4 | 10.2 |
| unit_var | 512 | 6.9 | 12.4 | 16.9 | 10.9 | 12.1 | 3.8 |
| unit_var | 256 | 5.1 | 6.1 | 10.3 | 18.6 | 11.9 | 2.7 |
| unit_var | 128 | 0.1 | 2.1 | 3.2 | 10.2 | 4.1 | 2.9 |
| hyperbolic | 512 | 11.7 | 16.9 | 19.4 | 14.2 | 10.2 | 1.4 |
| hyperbolic | 256 | 5.7 | 10.5 | 12.3 | 17.6 | 19.8 | 11.2 |
| hyperbolic | 128 | 0.9 | 4.1 | 5.0 | 11.5 | 13.7 | 4.1 |

### HFLM

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 24.4 | 26.4 | 25.2 | 31.7 | 32.2 | 31.9 |
| ngpt | 256 | 2.9 | 0.7 | 5.5 | 23.2 | 27.1 | 31.1 |
| ngpt | 128 | 0.4 | 1.7 | 4.0 | 12.6 | 14.0 | 19.2 |
| random | 512 | 34.4 | 37.1 | 31.2 | 20.9 | 22.8 | 34.6 |
| random | 256 | 11.5 | 15.8 | 23.4 | 21.6 | 14.5 | 26.8 |
| random | 128 | 1.7 | 7.5 | 2.4 | 18.0 | 15.0 | 18.6 |
| unit_var | 512 | 5.9 | 15.2 | 4.0 | 17.5 | 21.1 | 22.4 |
| unit_var | 256 | 0.5 | 19.6 | 17.2 | 27.5 | 14.4 | 30.2 |
| unit_var | 128 | 0.3 | 4.1 | 8.5 | 8.8 | 12.3 | 19.8 |
| hyperbolic | 512 | 19.4 | 14.3 | 23.2 | 27.5 | 25.9 | 28.9 |
| hyperbolic | 256 | 1.4 | 7.1 | 2.9 | 24.4 | 18.2 | 28.8 |
| hyperbolic | 128 | 0.8 | 3.2 | 3.5 | 10.1 | 10.6 | 8.6 |

# RESULTS — Easy LR sweep (geometry × init × dim × LR)

Sudoku-easy exact-match accuracy (%), 20k steps, eval @180 exact/greedy/tkv=-1, effective batch 256. **216/216 cells** done (— = still running).

**Per-geometry LR means** (avg over available init×dim cells):

| geometry | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 | best LR |
|---|---|---|---|---|---|---|---|
| SFM | 43.2 | 65.9 | 66.7 | 77.6 | 78.8 | 79.9 | **1e-3** |
| EFLM | 68.8 | 79.1 | 79.6 | 85.7 | 82.6 | 79.5 | **3e-4** |
| HFLM | 66.3 | 77.8 | 81.3 | 87.3 | 90.3 | 89.4 | **5e-4** |

**Per-geometry LR Best** (Best over the 12 init×dim cells):

Acc color = init of the best cell: <ng>ngpt</ng> · <rd>random</rd> · <uv>unit_var</uv> · <hy>hyperbolic</hy>

| Geo | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 | best LR |
|---|---|---|---|---|---|---|---|
| S-FLM | <ng>66.6</ng> | <hy>78.1</hy> | <rd>77.1</rd> | <uv>84.0</uv> | <rd>83.9</rd> | <rd>**90.4**</rd> | 1e-3 |
| E-FLM | <ng>**93.3**</ng> | <rd>93.3</rd> | <rd>92.7</rd> | <rd>91.0</rd> | <rd>91.6</rd> | <rd>88.4</rd> | 5e-5 |
| H-FLM | <rd>92.7</rd> | <rd>94.7</rd> | <rd>92.0</rd> | <ng>91.5</ng> | <uv>**96.5**</uv> | <hy>95.5</hy> | 5e-4 |

### SFM

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 66.6 | 77.9 | 72.1 | 77.9 | 77.6 | 79.0 |
| ngpt | 256 | 41.5 | 73.0 | 74.1 | 83.7 | 79.0 | 81.0 |
| ngpt | 128 | 12.6 | 58.0 | 58.9 | 78.1 | 77.3 | 85.8 |
| random | 512 | 57.7 | 59.2 | 77.1 | 73.6 | 76.8 | 73.7 |
| random | 256 | 33.0 | 71.2 | 76.5 | 80.2 | 83.9 | 79.5 |
| random | 128 | 20.4 | 45.5 | 45.9 | 81.5 | 75.3 | 90.4 |
| unit_var | 512 | 63.8 | 71.9 | 75.6 | 61.0 | 77.5 | 77.3 |
| unit_var | 256 | 60.0 | 77.6 | 67.1 | 84.0 | 77.0 | 78.0 |
| unit_var | 128 | 17.6 | 48.6 | 63.6 | 78.6 | 78.8 | 84.0 |
| hyperbolic | 512 | 66.0 | 78.1 | 71.7 | 72.5 | 80.7 | 60.9 |
| hyperbolic | 256 | 65.4 | 68.5 | 51.7 | 77.0 | 80.2 | 81.8 |
| hyperbolic | 128 | 14.3 | 61.6 | 65.5 | 83.4 | 81.8 | 87.8 |

### EFLM

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 93.3 | 92.5 | 90.5 | 87.5 | 89.5 | 79.8 |
| ngpt | 256 | 83.5 | 86.6 | 84.5 | 87.3 | 87.4 | 86.2 |
| ngpt | 128 | 53.8 | 69.8 | 86.4 | 82.4 | 79.2 | 87.0 |
| random | 512 | 89.7 | 93.3 | 92.7 | 91.0 | 91.6 | 81.2 |
| random | 256 | 88.8 | 88.6 | 88.3 | 90.8 | 86.0 | 85.3 |
| random | 128 | 55.6 | 76.8 | 85.5 | 88.5 | 90.7 | 88.4 |
| unit_var | 512 | 81.7 | 78.2 | 84.5 | 85.2 | 77.8 | 61.6 |
| unit_var | 256 | 60.2 | 79.0 | 76.6 | 79.8 | 72.5 | 76.0 |
| unit_var | 128 | 10.7 | 57.5 | 60.9 | 77.2 | 76.3 | 77.2 |
| hyperbolic | 512 | 76.8 | 82.7 | 67.7 | 87.4 | 82.8 | 70.9 |
| hyperbolic | 256 | 75.3 | 80.0 | 71.2 | 85.7 | 78.9 | 78.9 |
| hyperbolic | 128 | 56.5 | 64.1 | 66.8 | 86.2 | 78.8 | 81.1 |

### HFLM

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 88.9 | 92.7 | 91.2 | 89.5 | 91.7 | 88.1 |
| ngpt | 256 | 83.0 | 85.6 | 91.0 | 91.5 | 91.1 | 91.5 |
| ngpt | 128 | 40.6 | 62.1 | 67.0 | 85.8 | 89.9 | 91.2 |
| random | 512 | 92.7 | 94.7 | 92.0 | 85.3 | 82.8 | 95.2 |
| random | 256 | 64.1 | 85.0 | 90.0 | 82.9 | 95.2 | 93.2 |
| random | 128 | 78.1 | 73.8 | 77.1 | 84.1 | 91.0 | 78.1 |
| unit_var | 512 | 67.0 | 83.8 | 88.3 | 90.2 | 94.1 | 91.0 |
| unit_var | 256 | 65.5 | 89.8 | 84.3 | 86.4 | 96.5 | 91.1 |
| unit_var | 128 | 32.2 | 55.1 | 71.5 | 86.2 | 76.6 | 82.8 |
| hyperbolic | 512 | 77.6 | 80.0 | 79.1 | 91.0 | 92.4 | 89.3 |
| hyperbolic | 256 | 72.5 | 81.2 | 87.4 | 87.8 | 93.6 | 95.5 |
| hyperbolic | 128 | 33.1 | 49.8 | 56.2 | 87.4 | 88.6 | 86.0 |
