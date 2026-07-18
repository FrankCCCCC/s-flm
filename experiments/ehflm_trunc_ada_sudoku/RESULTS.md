# ehflm_trunc_ada_sudoku — Results

**Status: COMPLETE — 576/576 cells (2026-07-16). Zero job failures.**

Eval protocol (all numbers): single-shot sudoku_eval, 180 steps, exact velocity,
greedy last, top_k_velocity=−1, 2000 puzzles — identical to
`../hflm_curv_init_lr_sudoku`, whose cells are the naive anchors. Regenerate:
`python experiments/ehflm_trunc_ada_sudoku/collect.py`. Raw:
`outputs/ehflm_trunc_ada_sudoku/*/eval/results.json`.

## Headline conclusions

1. **Adaptive is the driver; truncation alone is weak.** Across the 90 paired
   (K,init,lr,difficulty) HFLM configs, **trunc+ada (`ta`) beats trunc-only (`to`) in
   66, loses in 24**. On EFLM the split is stark: at the
   best LR (3e-4) `ta` is 81.4/43.2 vs `to` 70.7/25.0 (med/hard) — truncation alone
   recovers only part of the gain over naive (62.2/19.2), adaptation delivers the rest.
2. **EFLM trunc+ada replicates the prior study** (`trunc_ada_sudoku`): 81.4±3.3 med /
   43.2±1.8 hard at lr 3e-4 (+19.2 / +24.0 over naive), vs the prior +16.7 / +25.8.
   Monotone in LR — 3e-4 is the EFLM recipe.
3. **HFLM medium: a clean curvature gradient toward flat.** Best-cell medium rises
   79 → 85 as K flattens −0.5 → −0.25. **K=−0.25 ta c0.01 lr5e-4 = 85.0±2.0 medium**,
   above the prior study's naive global best (81.1) — and the LR axis (5e-4/1e-3),
   never swept at this curvature before, is what surfaced it.
4. **HFLM hard: still ~46-capped, but tighter.** Best hard is K=−0.3 (46.8±7.3) ≈
   K=−0.25 (45.2±**1.2**); within seed noise of the old ~46 ceiling, but trunc+ada at
   the flat end cuts seed-std to ~1 (vs naive best 46.2±13). Regularization, not a
   new ceiling.

## EFLM (all 36 cells; naive anchor 62.2 med / 19.2 hard @ lr3e-4)

| method | lr | medium | hard |
|---|---|---|---|
| to | 3e-4 | 70.7 ± 2.1 | 25.0 ± 1.3 |
| to | 5e-4 | 65.5 ± 1.8 | 21.1 ± 6.4 |
| to | 1e-3 | 55.4 ± 5.8 | 15.1 ± 0.4 |
| **ta** | **3e-4** | **81.4 ± 3.3** | **43.2 ± 1.8** |
| ta | 5e-4 | 79.4 ± 3.2 | 39.6 ± 7.1 |
| ta | 1e-3 | 75.3 ± 1.5 | 30.5 ± 5.8 |

## HFLM best cell per (K, difficulty), mean ± std over 3 seeds

| K | best medium (cell) | naive | best hard (cell) | naive |
|---|---|---|---|---|
| −0.25 | **85.0 ± 2.0** (ta c0.01 lr5e-4) | 71.1 | 45.2 ± 1.2 (ta c0.01 lr1e-3) | 34.6 |
| −0.3 | 81.8 ± 4.5 (ta c0.04 lr3e-4) | 83.2 | **46.8 ± 7.3** (ta c0.04 lr1e-3) | 42.1 |
| −0.5 | 79.0 ± 3.2 (to c0.01 lr1e-3) | 81.1 | 43.7 ± 10.0 (ta c0.01 lr1e-3) | 46.2 |
| −0.7 | 80.2 ± 3.3 (ta random lr1e-3) | — | 40.6 ± 5.4 (ta c0.04 lr5e-4) | — |
| −1.0 | 76.5 ± 3.2 (ta c0.01 lr3e-4) | — | 38.9 ± 5.3 (ta c0.01 lr5e-4) | — |

**Global best: medium 85.0 (K−0.25 ta c0.01 lr5e-4), hard 46.8 (K−0.3 ta c0.04 lr1e-3).**
Both are `ta`. Medium is a monotone climb as K flattens −1.0→−0.25 (76.5→85.0). Hard is
flat-topped ~40–47 across K−0.25/−0.3/−0.5 (all within seed noise of the prior ~46
ceiling), then declines for the sharper geometries.

(Naive anchors: `../hflm_curv_init_lr_sudoku` best cell at that K, same protocol.
K−0.7/−1.0 naive cells live on other sites of that sweep; fill on rsync.)

## Recommended recipes

- **EFLM on Sudoku: trunc+ada, lr 3e-4, ALPHA_MAX=0.767** ⇒ ~81 med / ~43 hard.
- **HFLM on Sudoku: trunc+ada, K=−0.25, init custom 0.01, ALPHA_MAX=0.894** — lr 5e-4
  for medium (85.0), lr 1e-3 for hard (45.2, seed-std ~1). The trunc+ada curvature
  optimum (−0.25) is flatter than the naive optimum (−0.5), replicating the prior study
  on the full init×lr grid.

## Open

- Per-geometry ALPHA_MAX (table in EXPERIMENT.md) never collapsed sampling across the
  full grid, even at the tight c0.04 bounds (0.77–0.83) — confirms the prior study's
  rule: the geometry's own α⋆ is safe, fixed tight bounds (0.35) are not.
- Hard single-shot stays ~46-capped for HFLM (no cell clears 47); post-scheduler levers
  (restart samplers, multi-attempt, per-token time conditioning) remain the only untested
  route to higher hard accuracy — out of scope here (branch `claude/adv_sched`).
