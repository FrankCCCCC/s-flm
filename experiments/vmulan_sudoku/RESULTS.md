# vmulan_sudoku — RESULTS

## ROUND 2 (corrected implementation, 2026-07-26) — 36/36 cells

After the external audit (see EXPERIMENT.md header) fixed the oracle context
(→ `var_context=prompt`, MuLAN D.1), the scalar time conditioning (→
per-token adaLN + z-conditioning), and the GammaNet dropout, plus the exact
u-interval weighting and a 2k-step warm-up gate. Same protocol as round 1;
degree 5 re-baselines the fixed implementation, degree 9 is the higher
polynomial order.

### Exact-match accuracy (mean ± std over 3 seeds, %)

| R | baseline (fixed) | deg5 global | deg5 positional | deg9 global | deg9 positional |
|---|---|---|---|---|---|
| 5  | 33.3 ± 6.8 | **38.6 ± 3.2** | 26.5 ± 5.9 | 29.8 ± 4.3 | 23.6 ± 10.5 |
| 8  | **30.6 ± 2.1** | 25.3 ± 7.0 | 17.3 ± 4.8 | 25.7 ± 9.5 | 19.9 ± 7.1 |
| 16 | 20.2 ± 7.3 | **23.9 ± 11.0** | 21.4 ± 1.8 | 15.2 ± 11.5 | 12.1 ± 3.5 |

Per-seed: deg5 global r5 {40.9, 39.9, 34.9} — every seed above the baseline
mean, and the lowest variance of any r5 arm; r8 {25.7, 18.2, 32.1};
r16 {11.3, 30.7, 29.9}. deg9 global r5 {24.8, 32.5, 32.1}, r8 {18.7, 36.5,
21.9}, r16 {12.9, 5.0, 27.7}. Positional cells in the table.

### Findings

1. **The corrected learned schedule can beat the fixed baseline** — the
   first arm to do so: deg5 *global* wins at R = 5 (+5.3 pts, all three
   seeds ≥ 34.9) and R = 16 (+3.7 pts), loses at R = 8. The round-1
   implementation lost all 6 comparisons; the audit fixes are what changed
   the outcome.
2. **Higher polynomial order hurts**: deg9 ≤ deg5 in 5 of 6 cells (r5
   global −8.8, r16 global −8.7). Consistent with the audit's conditioning
   warning (t^8 monomials, correlated columns) and MuLAN's own Occam
   argument — degree 5 is the *simplest* polynomial with the desired
   properties, not a lower bound to scale past.
3. **A learnable decoding order still does not pay on sudoku**: positional ≤
   global in every matched cell, and training telemetry shows why — with the
   honest (prompt) context the positional warp barely separates positions
   (`alpha_spread` ≈ 2e-4 vs 0.3 under round 1's solution leak). The
   objective's per-position coupling is evidently too weak a signal here;
   round 1's dramatic "order" was the oracle artifact, not a usable
   mechanism.
4. Seed variance remains large (σ up to 11 pts at R = 16); the R = 5 deg5
   global win is the only comparison that clears one pooled σ.

### Verdict

MuLAN-style learned noise schedules help EFLM on sudoku hard **when the
context is inference-available (prompt/clues), the scope is global, and the
polynomial stays at degree 5** — the configuration closest to MuLAN's own.
Per-position schedules (the learnable-decoding-order hypothesis) and higher
polynomial degree both degrade it.

---

# ROUND 1 (superseded — oracle-context implementation)

Learned (MuLAN-style) noise schedule for EFLM on **sudoku hard** (30 clues),
20k steps, bs 256, lr 3e-4, `tiny-sphere-dit`, `rho_min = rho_max = R`,
`mode=sudoku_eval` @ 180 steps / exact velocity / greedy last.
Baseline = `eflm_rescale_sudoku` `naive` arm (fixed log-linear schedule),
identical protocol. 3 seeds per cell.

Status: **complete — 18/18 cells.**

## Exact-match accuracy (mean ± std over seeds, %)

| R | baseline (fixed log-linear) | vmulan global | vmulan positional |
|---|---|---|---|
| 5  | **33.3 ± 6.8** | 24.1 ± 7.0 | 25.6 ± 7.9 |
| 8  | **30.6 ± 2.1** | 24.1 ± 9.6 | 15.2 ± 8.0 |
| 16 | **20.2 ± 7.3** | 10.2 ± 5.1 | 14.4 ± 3.4 |

Per-seed: positional r5 {27.1, 32.6, 17.1}, r8 {24.2, 8.9, 12.4},
r16 {13.5, 18.1, 11.5}; global r5 {23.7, 17.4, 31.4}, r8 {19.3, 17.8, 35.1},
r16 {15.4, 5.3, 9.9}; baseline r5 {33.3, 40.1, 26.6}, r8 {30.2, 32.9, 28.7},
r16 {12.5, 27.0, 21.3}.

## Hypothesis verdicts

- **H1 (order is learnable): CONFIRMED.** Every `positional` run reaches
  `sched/alpha_spread` ≈ 0.28–0.30 (std of α across the 89 solution positions)
  within ~200 steps and holds it; every `global` control logs exactly 0. The
  positional arms train with strong reweighting (`sched/weight_max` ≈ 8.7)
  and markedly lower weighted train loss (~0.005 vs ~0.03 global). The
  mechanism works: the schedule genuinely differentiates positions.

- **H2 (order helps accuracy): REFUTED at n=3.** The learned schedule
  *underperforms* the fixed log-linear baseline at every R (both scopes,
  all three R values — 6/6 cells below baseline). `positional` < `global`
  at R = 8, comparable elsewhere. Seed variance is large (σ ≈ 5–10 pts), so
  single cells are noisy, but the direction is uniform.

- **H3 (learned warp compensates for R): NOT SUPPORTED.** Accuracy spread
  across R: baseline 13.1 pts, positional 11.2 pts — within noise.

## Interpretation: why a real learned order didn't pay off

The gap between "the schedule separated" (H1) and "accuracy dropped" (H2)
points at train/sample mismatches specific to how the order is *used*:

1. **Context distribution shift.** At training, c comes from a base-schedule
   corruption of the TRUE solution; at sampling, c comes from the evolving
   integration state — early in sampling that state is mostly noise plus
   partial *predictions*, which the gamma net never saw. A per-position
   schedule amplifies this: wrong early orders steer the integrator into
   joint states that training never produced.
2. **Reweighting starves early-decode positions.** `weight_max` ≈ 8.7 means
   the CE gradient concentrates on late-decode (high `-dα/dt`) positions;
   easy/early positions are down-weighted by up to ~9x, yet they are exactly
   what the sampler's early steps rely on.
3. **Heterogeneous per-position dt.** The Euler step is exact per position,
   but positions now integrate at different speeds, so the joint state visits
   mixed-α configurations that the training distribution (single shared t per
   sequence... per-position α but one t) only sparsely covers.

## Follow-ups worth running (not launched)

- **Warm-up the warp**: anneal a gate on the head output (γ-scale 0 → 1 over
  the first 5k steps) so the DLM first learns the base task, then the order.
- **Sample-time context from x̂**: feed the gamma net the predicted-clean
  embedding `p @ E` instead of the raw state, closing gap (1) (mirrors
  self-conditioning's carry).
- **Weight clipping/tempering**: `w ← w^κ` with κ ≈ 0.5, or clip at ~3, to
  soften gap (2) while keeping the invariance approximately.
- **Order-only ablation**: freeze the trained positional schedule, retrain
  the DLM under it with UNWEIGHTED CE — separates "the order is bad" from
  "the reweighting starves the model".

## Artifacts

- Runs + checkpoints: `outputs/vmulan_sudoku/vmulan_{scope}_r-{R}_lr-3e-4_d-hard_rs{seed}/`
  (checkpoints kept; the learned schedule lives in `noise.gamma_net.*`).
- W&B: project `debug`, e.g. run `oag6r9us` (positional r-5 rs1) —
  `sched/alpha_spread`, `sched/weight_max` traces back H1.
- Baseline numbers: `s-flm-dev1/.../outputs/eflm_rescale_sudoku/eflmrs_naive_*`.
