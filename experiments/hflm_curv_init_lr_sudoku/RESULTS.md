# H-FLM Curvature × Init × LR Sudoku Sweep — Final Results

Sweep: H-FLM, K ∈ {−0.25, −0.3, −0.5, −0.7, −1.0, −1.5} × init ∈ {ngpt, random, c0.01, c0.02, c0.04, c0.06, c0.08} × LR ∈ {1e-4, 3e-4, 5e-4, 1e-3} × difficulty ∈ {medium, hard} × seed ∈ {1, 2, 3}. Metric = full-board solve rate over 2000 held-out puzzles (exact velocity, top_k_v=−1, 180 steps, greedy last). K=−1.0 is the standard unit hyperboloid (baseline to beat). Coverage: 1008/1008 cells complete; all 336 (diff, K, init, LR) groups have n=3. All accuracies num_total=2000, all in [0,1]; no duplicate (cell, seed) rows.

## Headline conclusion (survives adversarial verification)

The prior single-seed claim that **K=−0.5 beats K=−1.0 by ~+20pt on medium does NOT survive seed-averaging** and is refuted: that gap was an artifact of one unlucky K=−1.0 baseline seed (64.4%), and seed-averaging shrinks the exact random@1e-3 delta from ~+20pt to +10.25pt, then to +1.80pt once each K uses its own best init/LR — within noise. What *does* survive is a smaller, robust **aggregate** curvature effect: averaged over the full balanced init×LR×seed grid (n≈84 runs/K), mild negative curvature beats the unit hyperboloid on both slices. On medium the per-K global mean traces an inverted-U peaking at K=−0.3 (76.7%) with K=−0.3 vs −1.0 = +4.9pt (Welch p=1.7e-7); on hard it plateaus over K∈[−0.7,−0.3] peaking at K=−0.5 (34.2%) with K=−0.5 vs −1.0 = +6.4pt (p=2.8e-7), confirmed by a paired t-test over 28 matched (init, LR) cells (p<1e-4) and holding within every individual LR (so it is not an LR confound). Strong curvature K=−1.5 reliably hurts on both slices (>2σ). Crucially, **no single best cell** clears the seed-noise bar (pooled across-seed 2·SE ≈ 9pt medium / 11pt hard), so the effect is real only as a seed/init/LR-averaged trend, not as a magic configuration.

## Medium — per-K best cell (seed-averaged, n=3)

| K | best init@lr | acc % ± seed-std | n |
|---|---|---|---|
| −0.25 | c0.01 @ 1e-3 | 80.78 ± 2.82 | 3 |
| −0.30 | c0.01 @ 5e-4 | 83.23 ± 5.46 | 3 |
| −0.50 | random @ 1e-3 | 82.68 ± 1.75 | 3 |
| −0.70 | c0.02 @ 5e-4 | 81.20 ± 1.08 | 3 |
| **−1.00 (baseline)** | c0.01 @ 5e-4 | 80.88 ± 0.63 | 3 |
| −1.50 | c0.02 @ 1e-3 | 75.87 ± 1.90 | 3 |

Pooled within-cell across-seed std (medium) = 5.5pt. Best-cell delta K=−0.5 − K=−1.0 = +1.80pt (n.s.). These "best cells" are reported for completeness only; per-cell winners do not beat baseline once seed noise is accounted for (see significance section).

## Hard — per-K best cell (seed-averaged, n=3)

| K | best init@lr | acc % ± seed-std | n |
|---|---|---|---|
| −0.25 | c0.01 @ 5e-4 | 39.87 ± 1.18 | 3 |
| −0.30 | c0.01 @ 3e-4 | 42.07 ± 9.92 | 3 |
| −0.50 | c0.01 @ 3e-4 | 46.22 ± 13.13 | 3 |
| −0.70 | c0.04 @ 1e-3 | 40.43 ± 4.93 | 3 |
| **−1.00 (baseline)** | c0.01 @ 3e-4 | 40.37 ± 5.28 | 3 |
| −1.50 | c0.01 @ 3e-4 | 34.98 ± 9.02 | 3 |

Pooled within-cell across-seed std (hard) = 6.6pt. The 46.22% at K=−0.5 is inflated by one lucky 58.2% seed (seeds 32.2 / 48.4 / 58.2); its edge over same-config K=−1.0 (35.7 / 39.3 / 46.1) is +5.85pt but not significant (p=0.51). Best-cell ordering on hard is unreliable — only the n≈84 global means separate curvatures.

## Medium — per-K × per-LR mean (over 7 inits, of seed-means; acc %)

| K | 1e-4 | 3e-4 | 5e-4 | 1e-3 | best LR |
|---|---|---|---|---|---|
| −0.25 | 72.5 | **75.6** | 75.4 | 73.2 | 3e-4 |
| −0.30 | 73.4 | 77.2 | **78.8** | 77.4 | 5e-4 |
| −0.50 | 68.3 | **78.1** | 76.7 | 78.0 | 3e-4 / 1e-3 (tie) |
| −0.70 | 68.2 | 76.3 | **77.1** | 76.5 | 5e-4 |
| −1.00 | 66.1 | 72.6 | **75.4** | 73.1 | 5e-4 |
| −1.50 | 58.5 | 68.0 | 69.5 | **73.7** | 1e-3 |

## Hard — per-K × per-LR mean (over 7 inits, of seed-means; acc %)

| K | 1e-4 | 3e-4 | 5e-4 | 1e-3 | best LR |
|---|---|---|---|---|---|
| −0.25 | 30.0 | **33.8** | 32.5 | 25.3 | 3e-4 |
| −0.30 | 29.1 | **36.5** | 35.4 | 28.9 | 3e-4 |
| −0.50 | 29.1 | **39.6** | 36.2 | 33.0 | 3e-4 |
| −0.70 | 26.7 | 34.4 | **35.7** | 34.9 | 5e-4 |
| −1.00 | 21.6 | **30.6** | 30.5 | 28.7 | 3e-4 |
| −1.50 | 19.0 | 26.1 | **28.2** | 22.9 | 5e-4 |

LR pattern (both slices): 1e-4 is worst at every curvature; 3e-4/5e-4 win for moderate curvature; 1e-3 is preferred only at K=−1.5 (medium) and degrades at flat/deep curvatures on hard. Within every fixed LR, K=−0.5 > K=−1.0 on hard (+4.3..+8.9pt, significant at all 4 LRs), confirming the curvature effect is not an LR confound.

## What the seeds changed (reconciliation vs the prior single-seed claim)

The prior result claimed K=−0.5 beat K=−1.0 by ~+20pt on medium (random @ 1e-3). Averaging that exact cell over 3 seeds:

| K | seed1 | seed2 | seed3 | mean | std | range |
|---|---|---|---|---|---|---|
| −0.5 | 84.25 | 80.80 | 83.00 | 82.68 | 1.75 | 3.45 |
| −1.0 | 64.40 | 78.30 | 74.60 | 72.43 | 7.20 | 13.90 |

- The inflation lived in the **baseline, not the treatment**: the K=−1.0 cell had one bad seed (64.4%) driving a 7.2pt std, while the K=−0.5 cell was stable (1.75pt std).
- Seed-averaging the same cell collapses the delta from ~+20pt to **+10.25pt** (sem 4.28, z=2.40).
- Letting each K pick its own best init/LR collapses it further to **+1.80pt** (sem 1.07, z=1.68) — not significant.
- On hard, the analogous best-cell delta (+5.85pt) is likewise noise: driven by a single 58.2% seed, p=0.51.

Conclusion: the headline "curvature helps by ~20pt" was single-seed noise. The genuine surviving effect is ~3–5pt and only visible in the seed/init/LR-aggregate.

## Significance statement (what exceeds seed noise, what doesn't)

**Exceeds noise (aggregate-level, trustworthy):**
- Medium global mean (n≈84/K): inverted-U, K=−0.3 (76.7%) vs K=−1.0 (71.8%) = +4.9pt, Welch p=1.7e-7. K=−0.7 also significantly > baseline.
- Hard global mean (n≈84/K): plateau over K∈[−0.7,−0.3], peak K=−0.5 (34.2%); K=−0.5 vs K=−1.0 = +6.4pt, p=2.8e-7; paired t over 28 matched (init, LR) cells p<1e-4. K=−0.7 vs −1.0 = +5.1pt, p=1.3e-5.
- Strong curvature K=−1.5 is significantly worse than baseline on both slices (medium all-cell 67.4%, hard 24.1%; >2σ).

**Within noise (do NOT over-claim):**
- All per-K **best-cell** deltas vs K=−1.0 fail the pooled 2·SE bar: medium K=−0.3 +2.35pt and K=−0.5 +1.80pt vs 2·SE≈9.0pt (|d|/2SE = 0.26, 0.20); hard K=−0.5 +5.85pt vs 2·SE≈10.8pt (0.54). No single configuration provably beats baseline.
- Within the top plateau, K=−0.5 vs K=−0.7 on hard = +1.4pt, p=0.20 (tie); on medium K∈{−0.3, −0.5, −0.7} best cells are mutually within ~1σ.
- Init effects: c0.01 is nominally the top init (hard 32.9% mean over K/LR), but its lead over ngpt/c0.04/c0.06 is ~2pt, within cross-cell spread (~6pt) — not significant. Large-scale inits c0.06/c0.08 are consistently weakest.

## Caveats and limitations

- **The effect is aggregate-only.** The per-K best-cell tables above name single-config winners (e.g. hard K=−0.5 = 46.22%, medium K=−0.3 = 83.23%), but these carry large seed std (13.1pt, 5.5pt) and do NOT beat baseline at 2·SE. The defensible curvature signal exists only at the seed/init/LR-averaged level. Do not report individual "winning" cells as beating the baseline.
- **Seed noise is large**, especially on hard (pooled per-run std 6.6pt; a 3-seed cell mean carries SE ≈ 3.8pt). 72/336 groups have across-seed range >15pt, concentrated in c0.01/c0.02 inits and the hard slice; max range 0.280 (hard, K=−1.0, c0.02, 3e-4). Three seeds is the floor for stable cell-level ranking here; more seeds would be needed to resolve within-plateau differences.
- **Sweep complete (1008/1008).** The last cell (hard, K=−0.5, c0.02, 5e-4) finished with seeds 21.2/42.4/46.6% (25pt across-seed range — itself an example of the hard-slice noise); adding it moved the hard K=−0.5 global mean by −0.2pt (34.4→34.2%, n=84), changing no conclusion.
- **The medium and hard optima differ slightly** (medium peaks at K=−0.3, hard at K=−0.5), consistent with a broad moderate-curvature optimum rather than a single sharp K*; treat "moderate K in −0.3..−0.7" as the operating region rather than any exact value.
- **ngpt does not collapse here.** Contrary to the TinyStories H-FLM lesson (where ngpt init killed the radial coordinate), on sudoku ngpt is mid-pack and healthy on both slices — the radial-init mismatch does not manifest for this task/metric.
- Numbers were independently re-derived from all_results.csv with pandas (accuracy×100) and reproduce the reference aggregator to <0.1pt across all reported cells, global means, and significance stats.