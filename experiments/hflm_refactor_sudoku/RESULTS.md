# hflm_refactor_sudoku — Results (66/66 cells, completed 2026-07-11 08:28)

Sweep: refactored HFLM (VFM expected-velocity sampler) on Sudoku easy/medium/hard,
K ∈ {−0.3, −0.5, −0.7, −1.0 (+−1.5 anchor)} × LR {3e-4, 5e-4, 1e-3} × init
{c0.01, c0.02, c0.04, random}, seed 1, 20k steps. Every checkpoint evaluated twice:
`topk1` (≡ old argmax sampler) and `topkall` (corrected expected-velocity sampler,
2000 puzzles each). Full tables in `analyze.py` output / `all_results.csv`.
Mid-sweep float32-lift bug and fix: see EXPERIMENT.md incident note; all numbers
below are post-fix.

## Headline conclusions

1. **No regression — the refactor reproduces the old sampler exactly.** Over 44
   cells matched against the old sweep (same K/init/LR, seed 1), new `topk1` vs
   old: **mean −0.42pt, median +0.00pt**. Cells whose checkpoints came from
   unperturbed training match old seed-1 numbers to ≤0.1pt (many exactly 0.00);
   the larger residuals (−17.8…+17.1) are confined to retrained cells on the
   noisy medium/hard slices and are symmetric around zero (old across-seed std
   reaches 8–14pt there).

2. **The corrected expected-velocity sampler beats argmax, consistently and
   more on harder tasks** (topkall − topk1, same checkpoints):
   easy **+0.51pt** (22 cells, max +1.5), medium **+1.77pt** (max +3.1),
   hard **+2.29pt** (max +5.3); virtually no cell got worse (min −0.7).
   Since `topk1` ≡ old behavior, the refactored version **surpasses** the old
   one whenever the true VFM marginal field (`top_k_velocity=-1`) is used.

3. **Curvature (single seed — treat ordering as indicative, not significant):**
   - **medium**: clean monotone preference for mild curvature at 3e-4/5e-4:
     K=−0.3 (86.0/84.6) > −0.5 > −0.7 > −1.0 (78.7/62.5); K=−1.5 in between
     (73.2). Replicates the old sweep's aggregate "mild curvature helps".
   - **easy** (never swept before): everything ≥89%; best K=−0.5
     (96.9–97.3 across *all three* LRs — strikingly LR-insensitive);
     K=−1.5 worst (89.4). Curvature effect exists but is small (~3–7pt).
   - **hard**: LR dominates K in this seed — the 5e-4 column (41–48%) crushes
     3e-4 (21–33%) at every K; within 5e-4 the K ordering is flat/noisy
     (−1.0 topkall 48.1 is the single best hard cell, beating the old sweep's
     best seed-averaged cell 46.2). Old sweep's caution stands: hard best-cells
     are seed noise; only aggregates separate curvatures.

4. **Init (at per-difficulty reference LR):** hard strongly prefers larger
   inits — c0.02/random reach 38–46% vs c0.01's 22–31% at 3e-4; medium prefers
   c0.01 at mild K but c0.02/random at K=−1.0; easy is insensitive (93–97%
   everywhere). Interaction with K is real: at K=−1.0 the larger inits
   consistently win on medium/hard.

## Best cells achieved (topkall)

| difficulty | best cell | acc | old-sweep context |
|---|---|---|---|
| easy | K=−0.5, c0.01 @ 1e-3 | **97.30%** | not previously swept (S-FLM paper ~81.5%) |
| medium | K=−0.3, c0.01 @ 3e-4 | **86.95%** | old best cell 83.2±5.5 (seed avg) |
| hard | K=−1.0, c0.01 @ 5e-4 | **48.05%** | old best cell 46.2±13.1 (seed avg) |

Single-seed caveat applies to all "best cell" claims (old sweep 2·SE ≈ 9–11pt);
the defensible claims are the paired, same-checkpoint sampler deltas (#2) and
the exact no-regression match (#1).

## Interpretation

- The VFM refactor is validated end-to-end: training untouched (bit-identical
  losses), old sampler behavior exactly recoverable (`top_k_velocity=1`), and
  the theoretically-correct marginal-field sampler is a free, uniform
  improvement (+0.5 to +2.3pt) at identical NFE.
- The sampler improvement grows with task difficulty, consistent with the VFM
  view: harder tasks keep more posterior mass off the argmax token deep into
  the trajectory, so integrating the expectation instead of the mode matters more.
- Curvature helps on easy/medium (mild K ≈ −0.3…−0.5 over K=−1), replicating
  the old aggregate finding on a new slice (easy); on hard, LR/init dwarf K at
  a single seed.

## Follow-ups

- 3-seed repeat of the {medium, hard} × K {−0.3, −0.5, −1.0} × best-init/LR
  subgrid with topkall to make the curvature + sampler-delta claims
  significance-grade.
- `top_k_velocity` intermediate values (e.g. 4) as a cheap knob between the two
  regimes; and port the factored-velocity sampler to `DFLMSampler` on
  `claude/dflm` before merging that branch.
