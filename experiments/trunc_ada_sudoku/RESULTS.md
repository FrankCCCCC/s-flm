# trunc_ada_sudoku — Results (9 rounds, ~100 runs)

Eval protocol for all headline numbers: **single-shot** sudoku_eval, 180 steps, exact
velocity, greedy last, top_k_velocity=-1, 2000 puzzles (identical to the
hflm_curv_init_lr_sudoku sweep). Raw: `outputs/trunc_ada_sudoku/tas_*/eval/`;
regenerate with `python experiments/trunc_ada_sudoku/collect.py`.
(Multi-attempt / restart-sampler protocols live on branch `claude/adv_sched`.)

## Headline conclusions

1. **The trunc+ada implementation is correct**, and delivers the theory-predicted
   gains in flat space: E-FLM trunc+ada **91.6 / 78.9 / 45.0** easy/med/hard vs naive
   88.2 / 62.2 / 19.2 (+3.4 / +16.7 / +25.8), beating S-FLM trunc+ada on medium+hard.
2. **On H^d, single-shot accuracy is capped at ~46±6 (hard)** across every schedule
   variant tested (9 rounds, table below). The scheduler's one robust effect is
   **regularization**: it rescues degraded geometries (pc=0.01: 15.6 → 36.9) and cuts
   seed-variance (±10 → ±2 at K=−0.3) — it pulls runs toward their geometry's mean,
   never above it.
3. **Best single-shot HFLM configs** (statistical tie): trunc+ada K=−0.25/c0.01/3e-4/
   α≤0.894 = **80.1±5.5 med / 44.6±6.1 hard** (+9/+10 vs same-K naive, z≈2) vs the
   naive K=−0.5 best cell 81.1 / 46.2±13. The trunc+ada curvature optimum (−0.25) is
   flatter than the naive optimum (−0.5).
4. **The 60%-single-shot target was NOT reached** — an exhaustively-supported
   negative result (every refuted lever documented below). Levers beyond the
   trunc+ada schedule family (restart samplers, multi-attempt selection) live
   on branch `claude/adv_sched`.

## Round 1: implementation check (35 cells, easy/med/hard)

All runs: `refit_count` exact, adapted alpha range exactly [eps, ALPHA_MAX], no NaNs.
E-FLM wins big (headline 1). HFLM: no collapse, but round-1 configs at/below naive —
which triggered the tuning rounds.

## Rounds 2–5: decomposition and curvature (K=−0.5 anchors: 81.1 med / 46.2 hard)

| arm | medium | hard | reading |
|---|---|---|---|
| ada-only (no trunc) | 79.75 | 41.40 | adaptive alone ≈ neutral |
| trunc-only α≤0.907 | 74.40 | 35.80 | even loose truncation costs ~6–10 |
| trunc+ada α≤0.907 (3 seeds med) | 73.1 ± 4.5 | 39.10 | tracks the truncation cost |
| trunc+ada α≤0.907, umix=0.3 | 78.25 | 36.35 | mild medium recovery only |
| trunc-only α≤0.35 | **14.65** | **0.00** | collapse — WITHOUT adaptive |
| trunc+ada α≤0.35 (3 seeds) | 12.9 ± 6.7 | 0.9 ± 1.1 | same collapse |
| trunc+ada α≤0.20 | 4.45 | 0.00 | monotone worse |
| trunc+ada K=−1, init 0.3, α≤0.624 | 73.95 | 19.85 | (init confounds hard) |
| **trunc+ada K=−0.25, α≤0.894 (3 seeds)** | **80.1 ± 5.5** | **44.6 ± 6.1** | the win: +9/+10 vs same-K naive |
| trunc+ada K=−0.3 lr3/lr5 (3 seeds) | 77.7±1.8 / 78.9±3.1 | 43.0±2.2 / 37.4±2.3 | no transfer to the higher base; std crushed |
| trunc+ada K=−0.1 (1 seed) | 79.90 | 19.95 | flat-ward trend breaks |
| K=−0.25 + umix 0.3 (3 seeds) | 77.6 ± 6.8 | 34.1 ± 7.4 | umix hurts at −0.25 |

## The refuted hypothesis (important negative result)

The measured loss geometry (loss_geometry_vis sudoku_hard K0.5: L(t)≈0 for α>0.34)
suggested "most of the schedule is uninformative → truncate to the band". This is
**experimentally false on H^d**: cutting to α≤0.35 collapses generation to ~0–17%
even with truncation alone, and the harm is monotone in tightness
(α_max 1.0 → 81.1, 0.907 → 74.4, 0.35 → 14.7, 0.20 → 4.5 on medium). Teacher-forced
zero-loss does NOT mean a region is dispensable: the loss is measured on true
geodesic interpolants, while the sampler visits self-generated states with
accumulated error; the near-clean range must be trained for trajectories to snap
back onto the data manifold. On the sphere the α⋆ bound doubles as a Voronoi-collapse
certificate (decoding at α⋆ is safe), so truncation and sampling stay consistent —
that equivalence simply does not hold in hyperbolic space. This also reinterprets
the historical α_max=0.093 HFLM collapse: it was not "the bound was mis-set"; ANY
tight truncation kills hyperbolic sampling.

Curvature works where truncation fails because the transition-band *width* scales
like 1/(c·D): flattening K widens the informative band (EFLM, the K→0 limit, gains
+26 on hard) — but only down to a point (K=−0.1 hard drops to ~20 on 1 seed; the
trunc+ada optimum sits near K=−0.25, between the naive optimum −0.5 and flat).

## Rounds 6–7: every remaining schedule lever (hard, 3 seeds unless noted)

| lever | result | verdict |
|---|---|---|
| cosine² base (no trunc/ada) | 24.4 ± 2.6 | refuted — base shape hurts |
| cos²+ada K=−0.5 / K=−0.25 | 32.5±13.6 / 43.1±4.2 | refuted — ada re-warps the base away |
| late adaptation (warmup 10k) | 40.8 ± 4.0 | refuted |
| eval-schedule A/B (same ckpt) | 37.25 / 37.15 / 37.55 | sampler insensitive to the warp |
| prior_cov 0.05 / 0.01 (naive) | 24.0±4.6 / 15.6±1.9 | refuted — coverage loss dominates |
| prior_cov 0.05 / 0.01 (+ada) | 33.6±4.5 / 36.9±2.8 | ada rescues, still < baseline |
| log-domain ada importance | 32.7 ± 6.4 | refuted |
| 40k steps (ta_k25) | 36.6 ± 6.9 | plateaued |

Also established: "same-seed" retrains differ by up to 12pt — GPU nondeterminism
spans the seed-variance band, so the spread is training-dynamics noise, not seed RNG.

## Recommended single-shot recipe & open items

- **HFLM on Sudoku: K=−0.25, init custom 0.01, lr 3e-4, ALPHA_MAX=0.894
  (alpha_star_numeric), adaptive on (defaults).** Expect ≈80 med / ≈45 hard,
  seed-std ≈ half of naive.
- Do NOT truncate below ~0.9 at K≤−0.5; never use loss-geometry-derived tight bounds
  for HFLM.
- Untested follow-ups: K∈{−0.35}, more seeds; post-scheduler levers (per-token time
  conditioning, self-conditioning, init/basin engineering) are where single-shot
  headroom would have to come from.
- Caveats: K=−0.1 and several decomposition arms are single-seed; hflm_ta_k1 hard
  lacks a same-init naive anchor; sweep checkpoints were deleted after eval (two
  retained: `outputs/trunc_ada_sudoku/{abtest_k25_hard_rs1,recipe_k25_hard_rs2}`).

---

Multi-attempt / restart-sampler / verified best-of-k results and code live on
branch `claude/adv_sched` (protocol-changing evaluation; kept out of headline
comparisons per the presentation rule).
