# eflm_rescale_auto_trunc_cov_reparam_sudoku — Results (INTERIM)

**Status 2026-10-09 00:05 EDT:** 71 / 81 cells evaluated (all `auto` / `autoada` seeds except one, `llada` seeds 1-2), 8 training, 2 queued.
Design: `EXPERIMENT.md`. Table: `python report.py`. Solve rate (%) on 2000 hard boards, gaussian readout everywhere.

| noise | R | LR | seed 1 | seed 2 | seed 3 | mean ± sd |
|---|---|---|---|---|---|---|
| auto | 5.0 | 3e-4 | 41.7 | 37.7 | 28.5 | 36.0 ± 6.7 (n=3) |
| auto | 5.0 | 5e-4 | 42.7 | 33.4 | 29.6 | 35.2 ± 6.7 (n=3) |
| auto | 5.0 | 1e-3 | 38.0 | 19.1 | 24.1 | 27.1 ± 9.7 (n=3) |
| auto | 8 | 3e-4 | 41.5 | 47.5 | 25.6 | 38.2 ± 11.3 (n=3) |
| auto | 8 | 5e-4 | 27.4 | 44.5 | 41.9 | 37.9 ± 9.2 (n=3) |
| auto | 8 | 1e-3 | 31.2 | 42.2 | 31.7 | 35.0 ± 6.2 (n=3) |
| auto | 16 | 3e-4 | 49.9 | 34.8 | 35.0 | 39.9 ± 8.6 (n=3) |
| auto | 16 | 5e-4 | 34.8 | 31.9 | 39.0 | 35.2 ± 3.5 (n=3) |
| auto | 16 | 1e-3 | 26.8 | 41.0 | 29.9 | 32.6 ± 7.5 (n=3) |
| autoada | 5.0 | 3e-4 | 35.9 | 37.0 | 39.2 | 37.4 ± 1.7 (n=3) |
| autoada | 5.0 | 5e-4 | 33.9 | 35.6 | 43.5 | 37.7 ± 5.1 (n=3) |
| autoada | 5.0 | 1e-3 | 23.0 | 38.9 | 37.2 | 33.0 ± 8.7 (n=3) |
| autoada | 8 | 3e-4 | 40.5 | 40.5 | 29.8 | 36.9 ± 6.2 (n=3) |
| autoada | 8 | 5e-4 | 49.6 | 40.6 | 21.9 | 37.4 ± 14.1 (n=3) |
| autoada | 8 | 1e-3 | 37.4 | 21.0 | 40.6 | 33.0 ± 10.5 (n=3) |
| autoada | 16 | 3e-4 | 43.4 | 34.8 | 30.0 | 36.0 ± 6.8 (n=3) |
| autoada | 16 | 5e-4 | 36.5 | 36.6 | 31.6 | 34.9 ± 2.9 (n=3) |
| autoada | 16 | 1e-3 | 46.6 | 39.1 | — | 42.9 ± 5.2 (n=2) |
| llada | 5.0 | 3e-4 | 22.4 | 28.4 | — | 25.4 ± 4.3 (n=2) |
| llada | 5.0 | 5e-4 | 32.6 | 18.4 | — | 25.5 ± 10.0 (n=2) |
| llada | 5.0 | 1e-3 | 25.9 | 42.2 | — | 34.1 ± 11.6 (n=2) |
| llada | 8 | 3e-4 | 33.6 | 36.8 | — | 35.2 ± 2.3 (n=2) |
| llada | 8 | 5e-4 | 32.6 | 14.9 | — | 23.8 ± 12.5 (n=2) |
| llada | 8 | 1e-3 | 42.4 | 30.9 | — | 36.7 ± 8.1 (n=2) |
| llada | 16 | 3e-4 | 28.3 | 34.0 | — | 31.1 ± 4.0 (n=2) |
| llada | 16 | 5e-4 | 46.1 | 31.2 | — | 38.6 ± 10.5 (n=2) |
| llada | 16 | 1e-3 | 35.9 | 34.9 | — | 35.4 ± 0.7 (n=2) |

## Reads so far

- 3-seed means sit at **27-40 %** for every (noise, R, LR); the best is `autoada` R=16 lr 1e-3 42.9 ± 5.2 (2 seeds),
  then `auto` R=16 lr 3e-4 39.9 ± 8.6. No arm or R separates from the others beyond seed noise (per-seed sd 2-14 pt).
- Naive-readout references (Ampere/Ada harness, 256 x 1, different tau): auto + adaptive R=16 **52.0 ± 1.4**,
  log-linear + adaptive R=5 **53.9 ± 4.1**, auto naive R=8 45.1 ± 1.3, log-linear + trunc + adaptive R=8 45.6 ± 1.2.
  The gaussian readout is 10-15 pt below them. The harness offset measured for HBFM (4.6 pt between Turing/128x2 and
  Ampere/256x1) is smaller than that gap, but a matched naive arm on this harness would be needed to make it clean.
- Seed variance is very large (e.g. `autoada` R=8 lr 5e-4: 49.6 / 40.6 / 21.9), much larger than in the naive sweeps.
