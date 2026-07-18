# H-FLM Curvature × Init × LR Sudoku Sweep — Final Results

Sweep: H-FLM, K ∈ {−0.25, −0.3, −0.5, −0.7, −1.0, −1.5} × init ∈ {ngpt, random, c0.01, c0.02, c0.04, c0.06, c0.08} × LR ∈ {1e-4, 3e-4, 5e-4, 1e-3} × difficulty ∈ {medium, hard} × seed ∈ {1, 2, 3}. Metric = full-board solve rate over 2000 held-out puzzles (exact velocity, top_k_v=−1, 180 steps, greedy last). K=−1.0 is the standard unit hyperboloid (baseline to beat). Coverage: 1008/1008 cells complete; all 336 (diff, K, init, LR) groups have n=3. All accuracies num_total=2000, all in [0,1]; no duplicate (cell, seed) rows.

## Headline conclusion (survives adversarial verification)

The prior single-seed claim that **K=−0.5 beats K=−1.0 by ~+20pt on medium does NOT survive seed-averaging** and is refuted: that gap was an artifact of one unlucky K=−1.0 baseline seed (64.4%), and seed-averaging shrinks the exact random@1e-3 delta from ~+20pt to +10.25pt, then to +1.80pt once each K uses its own best init/LR — within noise. What *does* survive is a smaller, robust **aggregate** curvature effect: averaged over the full balanced init×LR×seed grid (n≈84 runs/K), mild negative curvature beats the unit hyperboloid on both slices. On medium the per-K global mean traces an inverted-U peaking at K=−0.3 (76.7%) with K=−0.3 vs −1.0 = +4.9pt (Welch p=1.7e-7); on hard it plateaus over K∈[−0.7,−0.3] peaking at K=−0.5 (34.2%) with K=−0.5 vs −1.0 = +6.4pt (p=2.8e-7), confirmed by a paired t-test over 28 matched (init, LR) cells (p<1e-4) and holding within every individual LR (so it is not an LR confound). Strong curvature K=−1.5 reliably hurts on both slices (>2σ). Crucially, **no single best cell** clears the seed-noise bar (pooled across-seed 2·SE ≈ 9pt medium / 11pt hard), so the effect is real only as a seed/init/LR-averaged trend, not as a magic configuration.

## Best Over {Init * LR} per Curvature

### Medium — per-K best cell (seed-averaged, n=3)

| K | best init@lr | acc % ± seed-std | n |
|---|---|---|---|
| −0.25 | c0.01 @ 1e-3 | 80.78 ± 2.82 | 3 |
| −0.30 | c0.01 @ 5e-4 | 83.23 ± 5.46 | 3 |
| −0.50 | random @ 1e-3 | 82.68 ± 1.75 | 3 |
| −0.70 | c0.02 @ 5e-4 | 81.20 ± 1.08 | 3 |
| **−1.00 (baseline)** | c0.01 @ 5e-4 | 80.88 ± 0.63 | 3 |
| −1.50 | c0.02 @ 1e-3 | 75.87 ± 1.90 | 3 |

Pooled within-cell across-seed std (medium) = 5.5pt. Best-cell delta K=−0.5 − K=−1.0 = +1.80pt (n.s.). These "best cells" are reported for completeness only; per-cell winners do not beat baseline once seed noise is accounted for (see significance section).

### Hard — per-K best cell (seed-averaged, n=3)

| K | best init@lr | acc % ± seed-std | n |
|---|---|---|---|
| −0.25 | c0.01 @ 5e-4 | 39.87 ± 1.18 | 3 |
| −0.30 | c0.01 @ 3e-4 | 42.07 ± 9.92 | 3 |
| −0.50 | c0.01 @ 3e-4 | 46.22 ± 13.13 | 3 |
| −0.70 | c0.04 @ 1e-3 | 40.43 ± 4.93 | 3 |
| **−1.00 (baseline)** | c0.01 @ 3e-4 | 40.37 ± 5.28 | 3 |
| −1.50 | c0.01 @ 3e-4 | 34.98 ± 9.02 | 3 |

Pooled within-cell across-seed std (hard) = 6.6pt. The 46.22% at K=−0.5 is inflated by one lucky 58.2% seed (seeds 32.2 / 48.4 / 58.2); its edge over same-config K=−1.0 (35.7 / 39.3 / 46.1) is +5.85pt but not significant (p=0.51). Best-cell ordering on hard is unreliable — only the n≈84 global means separate curvatures.

## Average Over {Init * LR} per Curvature

Each curvature's solve rate averaged over the **full init × LR × seed grid** (n=84 runs/K) — the trustworthy aggregate the headline rests on. `± std` is the across-run dispersion (spans init/LR/seed, hence large); the standard error of the per-K mean is std/√84 ≈ 0.5–0.9pt, so the curvature ordering is well-resolved even though no single cell beats baseline (see the significance section). Δ is vs the K=−1.00 unit-hyperboloid baseline.

### Medium — per-K global mean (over init × LR × seed, n=84)

| K | acc % ± std | Δ vs −1.00 |
|---|---|---|
| −0.25 | 74.2 ± 5.8 | +2.4 |
| −0.30 | **76.7 ± 4.6** | +4.9 |
| −0.50 | 75.3 ± 6.6 | +3.5 |
| −0.70 | 74.5 ± 6.1 | +2.7 |
| **−1.00 (baseline)** | 71.8 ± 6.8 | — (baseline) |
| −1.50 | 67.4 ± 8.5 | −4.4 |

### Hard — per-K global mean (over init × LR × seed, n=84)

| K | acc % ± std | Δ vs −1.00 |
|---|---|---|
| −0.25 | 30.4 ± 7.1 | +2.6 |
| −0.30 | 32.5 ± 7.7 | +4.7 |
| −0.50 | **34.2 ± 7.6** | +6.4 |
| −0.70 | 32.9 ± 6.8 | +5.1 |
| **−1.00 (baseline)** | 27.8 ± 7.8 | — (baseline) |
| −1.50 | 24.1 ± 7.7 | −3.7 |

Inverted-U on medium (peak K=−0.30) and a plateau over K∈[−0.7,−0.3] on hard (peak K=−0.5); K=−1.5 hurts on both. Deltas match the significance section (medium K=−0.3 +4.9, Welch p=1.7e-7; hard K=−0.5 +6.4, p=2.8e-7).

## Average Over {Init} per Curvature * LR

Each (K, LR) cell averaged over the **7 inits** (of seed-means; 21 runs/cell). Bold = best LR in that K row; the `best LR` column names it. `± std` is the across-run dispersion (spans init × seed); the SE on the cell mean is std/√21 ≈ 0.9–1.9pt.

### Medium — per-K × per-LR mean ± std (over 7 inits × 3 seeds = 21 runs; acc %)

| K | 1e-4 | 3e-4 | 5e-4 | 1e-3 | best LR |
|---|---|---|---|---|---|
| −0.25 | 72.5 ± 6.4 | **75.6 ± 5.0** | 75.4 ± 4.3 | 73.2 ± 6.8 | 3e-4 |
| −0.30 | 73.4 ± 4.3 | 77.2 ± 4.4 | **78.8 ± 4.1** | 77.4 ± 3.9 | 5e-4 |
| −0.50 | 68.3 ± 6.8 | **78.1 ± 5.4** | 76.7 ± 4.4 | 78.0 ± 4.0 | 3e-4 / 1e-3 (tie) |
| −0.70 | 68.2 ± 5.8 | 76.3 ± 4.3 | **77.1 ± 4.4** | 76.5 ± 5.3 | 5e-4 |
| −1.00 | 66.1 ± 6.0 | 72.6 ± 5.7 | **75.4 ± 5.2** | 73.1 ± 6.6 | 5e-4 |
| −1.50 | 58.5 ± 8.7 | 68.0 ± 5.7 | 69.5 ± 6.5 | **73.7 ± 4.6** | 1e-3 |

### Hard — per-K × per-LR mean ± std (over 7 inits × 3 seeds = 21 runs; acc %)

| K | 1e-4 | 3e-4 | 5e-4 | 1e-3 | best LR |
|---|---|---|---|---|---|
| −0.25 | 30.0 ± 6.6 | **33.8 ± 5.7** | 32.5 ± 6.4 | 25.3 ± 7.1 | 3e-4 |
| −0.30 | 29.1 ± 6.6 | **36.5 ± 6.6** | 35.4 ± 6.5 | 28.9 ± 7.9 | 3e-4 |
| −0.50 | 29.1 ± 7.6 | **39.6 ± 6.1** | 35.1 ± 6.6 | 33.0 ± 6.3 | 3e-4 |
| −0.70 | 26.7 ± 7.3 | 34.4 ± 5.5 | **35.7 ± 5.2** | 34.9 ± 5.3 | 5e-4 |
| −1.00 | 21.6 ± 6.9 | **30.6 ± 8.0** | 30.5 ± 5.5 | 28.7 ± 7.4 | 3e-4 |
| −1.50 | 19.0 ± 6.7 | 26.1 ± 8.4 | **28.2 ± 6.5** | 22.9 ± 6.1 | 5e-4 |

LR pattern (both slices): 1e-4 is worst at every curvature; 3e-4/5e-4 win for moderate curvature; 1e-3 is preferred only at K=−1.5 (medium) and degrades at flat/deep curvatures on hard. Within every fixed LR, K=−0.5 > K=−1.0 on hard (+4.3..+8.9pt, significant at all 4 LRs), confirming the curvature effect is not an LR confound.

## Average Over {LR} per Curvature * Init

Each (K, init) cell averaged over the **4 LRs** (of seed-means; 12 runs/cell). Bold = best init in that K row; the `best init` column names it. `± std` is the across-run dispersion (spans LR × seed); the SE on the cell mean is std/√12 ≈ 0.8–3.4pt.

### Medium — per-K × per-init mean ± std (over 4 LRs × 3 seeds = 12 runs; acc %)

| K \ init | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 | best init |
|---|---|---|---|---|---|---|---|---|
| −0.25 | 74.2 ± 6.3 | **76.0 ± 5.0** | 75.6 ± 6.4 | 75.0 ± 5.6 | 73.2 ± 4.1 | 71.2 ± 7.1 | 73.9 ± 5.9 | **random** |
| −0.30 | 75.0 ± 3.9 | 78.3 ± 3.8 | **80.0 ± 5.3** | 79.2 ± 4.0 | 74.6 ± 3.9 | 73.9 ± 3.9 | 75.7 ± 3.8 | **c0.01** |
| −0.50 | 74.7 ± 4.5 | **79.0 ± 5.6** | 73.3 ± 9.8 | 77.2 ± 6.8 | 76.5 ± 3.8 | 72.8 ± 6.4 | 73.6 ± 6.5 | **random** |
| −0.70 | 75.3 ± 6.0 | 73.8 ± 7.8 | 73.3 ± 7.4 | **76.9 ± 6.0** | 75.4 ± 4.8 | 73.0 ± 6.1 | 73.8 ± 4.6 | **c0.02** |
| −1.00 | **75.6 ± 6.7** | 70.1 ± 6.8 | 70.7 ± 8.7 | 73.3 ± 5.1 | 72.1 ± 4.4 | 71.4 ± 6.3 | 69.3 ± 8.0 | **ngpt** |
| −1.50 | 67.1 ± 8.7 | 66.1 ± 9.6 | 67.5 ± 8.9 | **70.3 ± 6.9** | 68.4 ± 7.1 | 64.6 ± 9.4 | 68.0 ± 9.6 | **c0.02** |

### Hard — per-K × per-init mean ± std (over 4 LRs × 3 seeds = 12 runs; acc %)

| K \ init | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 | best init |
|---|---|---|---|---|---|---|---|---|
| −0.25 | 29.3 ± 6.5 | 26.2 ± 6.0 | **35.6 ± 7.1** | 27.4 ± 6.3 | 32.0 ± 8.0 | 32.4 ± 4.4 | 29.8 ± 8.0 | **c0.01** |
| −0.30 | 34.0 ± 8.5 | 33.2 ± 8.6 | **36.8 ± 8.5** | 30.2 ± 7.7 | 31.1 ± 5.0 | 31.0 ± 6.3 | 31.0 ± 8.1 | **c0.01** |
| −0.50 | 34.9 ± 6.2 | 32.3 ± 8.0 | 35.2 ± 10.0 | 34.4 ± 11.7 | **35.9 ± 2.6** | 34.0 ± 5.7 | 32.6 ± 6.5 | **c0.04** |
| −0.70 | 34.4 ± 4.7 | 30.4 ± 8.5 | 33.0 ± 10.4 | 32.5 ± 8.1 | **36.1 ± 4.8** | 32.1 ± 4.0 | 32.1 ± 4.6 | **c0.04** |
| −1.00 | 28.8 ± 7.2 | 24.0 ± 9.0 | **31.2 ± 7.5** | 25.0 ± 10.6 | 26.8 ± 8.1 | 29.3 ± 4.9 | 29.9 ± 4.8 | **c0.01** |
| −1.50 | 23.2 ± 9.5 | 25.4 ± 8.6 | **25.8 ± 9.2** | 22.9 ± 7.2 | 23.1 ± 5.4 | 24.0 ± 7.9 | 24.0 ± 6.8 | **c0.01** |

Init pattern: **c0.01** is the modal best init — top on hard at 4/6 curvatures and best overall on hard (mean over K = 32.9%, ~2pt above ngpt/c0.04 at 30.8%) — but its lead sits within the ~6pt cross-cell spread, so it is not significant. On medium the best init is curvature-dependent (random/c0.01/c0.02/ngpt) with no single winner. Large-scale inits **c0.06/c0.08** are consistently the weakest on both slices.

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

# Appendix

## Sweep Config

**Grid (1008 cells = 336 (diff, K, init, LR) groups × 3 seeds).** Spec: `slides/jul02_2026`; orchestrated by `sweep.py`, aggregated by `analyze.py`. Numbers in this report were re-derived from `all_results.csv` with pandas (accuracy ×100, sample std ddof=1).

| axis | values | n | knob → config |
|---|---|--:|---|
| curvature K | −0.25, −0.30, −0.50, −0.70, −1.00, −1.50 | 6 | `GAUSS_CURV` → `algo.gaussian_curvature` |
| init | ngpt (N(0, 1/d)), random (N(0, 4e-4)), custom std {0.01, 0.02, 0.04, 0.06, 0.08} | 7 | `INIT` / `INIT_STD` → `model.init` / `model.init_std` |
| LR | 1e-4, 3e-4, 5e-4, 1e-3 | 4 | `LR` → `optim.lr` |
| difficulty | medium (35 clues), hard (30 clues) | 2 | `DIFFICULTY` |
| seed | 1, 2, 3 (reported average) | 3 | `SEED` |

`random` ≡ custom std 0.02, and ngpt ≈ std 0.0442 at d=512 — the named baselines double as consistency checks on the custom-std scan. K=−1.0 is the standard unit hyperboloid (baseline to beat). All K satisfy the float64 Lorentz bound ρ/R ≤ 20 (worst case K=−1.5: 12·√1.5 ≈ 14.7).

**Fixed (all cells).** Model: `tiny-hyperbolic-dit`, 512 wide / 8 deep / 8 heads (~28.6M params), seq 180, `prior_cov` 0.25, `rho_max` 12, noise = log-linear. Training: 20k steps, batch 256, bf16, EMA 0.9999, AdamW (wd 0, betas 0.9/0.999, eps 1e-8), grad clip 1.0. Data: 48k train / 2k val puzzles, data seed 42. Each cell runs `scripts/train/sudoku/hflm.sh` → `scripts/sample/sudoku/hflm.sh` (train → eval; idempotent + resumable, skips cells whose `eval/results.json` exists).

**Eval.** `sudoku_eval`, 180 sampling steps, exact velocity, greedy last step, `top_k_velocity=−1` (velocity averaged across the full vocab). Metric = full-board (81-cell exact) solve rate over the 2000-puzzle held-out val set. `top_k_v=−1` differs from `hflm_curv_sudoku`'s top-1, so accuracies are **not** comparable across the two experiments.

**Compute (3 sites, disjoint K split; 1 GPU/cell, 6 h wall limit, `--nice=0 --requeue`).**

| site (user) | K | queues | CPU / mem | W&B |
|---|---|---|---|---|
| unicorn (sc3379) | −0.25, −0.50 | `thickstun,desa` (excl. `desa-compute-01`) | 4 / 16G | online |
| tc (shengyenc) | −0.30, −0.70 | h200/a100 preemptable + normal (`swan_research_dlm`) | 4 / 32G | offline |
| falcon (shengyenc) | −1.00, −1.50 | a30/l40s preemptable + normal (`swan_research_dlm`) | 4 / 32G | offline |

tc and falcon share the ARC `/home` filesystem, so their K sets are kept disjoint; results-based skipping makes the sweep idempotent across all three sites. Per-cell checkpoints (~1.8 G) are deleted once `eval/results.json` exists.

## Detailed Results

Per-cell full-board solve rate, **mean ± across-seed std** (%, sample std ddof=1, n=3 seeds each). One matrix per difficulty × LR; rows = curvature K, columns = init. Bold = best init in that (K, LR) row. These per-cell values are the source the body's per-K best-cell, per-K×LR-mean, and global-mean tables aggregate from.

### Medium

**LR: 1e-4**

| K \ init | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 |
|---|---|---|---|---|---|---|---|
| −0.25 | 69.68 ± 3.71 | **79.82 ± 2.86** | 74.28 ± 3.69 | 78.98 ± 3.11 | 70.77 ± 3.80 | 65.38 ± 4.26 | 68.40 ± 8.50 |
| −0.30 | 71.93 ± 5.46 | **78.23 ± 3.89** | 73.90 ± 5.12 | 76.18 ± 2.28 | 70.03 ± 1.38 | 70.43 ± 4.25 | 72.88 ± 3.05 |
| −0.50 | 70.28 ± 5.86 | 71.57 ± 5.99 | 60.72 ± 8.91 | 68.50 ± 8.51 | **71.97 ± 3.62** | 67.92 ± 4.63 | 67.13 ± 8.75 |
| −0.70 | 69.68 ± 6.23 | 64.88 ± 3.26 | 67.32 ± 10.43 | **70.43 ± 4.99** | 68.95 ± 3.70 | 65.97 ± 6.31 | 69.98 ± 7.45 |
| −1.00 | **69.25 ± 6.11** | 63.53 ± 6.17 | 66.67 ± 2.86 | 69.25 ± 1.48 | 67.10 ± 5.30 | 65.97 ± 9.70 | 60.82 ± 8.13 |
| −1.50 | 60.10 ± 13.04 | 54.65 ± 5.43 | 54.38 ± 2.34 | **63.17 ± 8.21** | 61.22 ± 7.86 | 56.18 ± 10.51 | 60.08 ± 14.06 |

**LR: 3e-4**

| K \ init | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 |
|---|---|---|---|---|---|---|---|
| −0.25 | 76.83 ± 5.76 | 75.83 ± 5.73 | 71.08 ± 9.37 | 75.83 ± 5.73 | 73.92 ± 2.02 | **77.97 ± 2.88** | 77.70 ± 1.78 |
| −0.30 | 74.85 ± 3.33 | 76.93 ± 4.24 | 80.78 ± 2.25 | **81.67 ± 4.72** | 73.53 ± 2.59 | 75.27 ± 4.76 | 77.38 ± 5.06 |
| −0.50 | 78.58 ± 3.26 | **82.52 ± 2.99** | 81.07 ± 6.13 | 77.33 ± 1.11 | 77.33 ± 3.15 | 74.32 ± 10.71 | 75.73 ± 5.46 |
| −0.70 | **79.13 ± 1.93** | 74.90 ± 5.73 | 75.43 ± 5.87 | 75.80 ± 6.78 | 77.23 ± 1.34 | 74.70 ± 6.30 | 76.73 ± 2.58 |
| −1.00 | **76.87 ± 5.80** | 69.20 ± 5.94 | 72.87 ± 7.88 | 75.05 ± 6.66 | 72.20 ± 3.60 | 72.77 ± 2.63 | 69.38 ± 8.06 |
| −1.50 | 66.32 ± 7.74 | 63.17 ± 5.63 | 70.33 ± 5.46 | 69.58 ± 3.74 | 68.78 ± 7.01 | 66.90 ± 7.81 | **70.83 ± 4.23** |

**LR: 5e-4**

| K \ init | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 |
|---|---|---|---|---|---|---|---|
| −0.25 | **78.97 ± 2.91** | 73.20 ± 4.00 | 76.30 ± 6.41 | 75.50 ± 1.56 | 75.85 ± 4.34 | 72.62 ± 5.45 | 75.47 ± 5.62 |
| −0.30 | 77.90 ± 1.96 | 78.37 ± 3.80 | **83.23 ± 5.46** | 80.25 ± 5.51 | 79.03 ± 2.39 | 74.65 ± 3.36 | 77.88 ± 3.07 |
| −0.50 | 73.90 ± 0.13 | 79.07 ± 1.63 | 75.72 ± 7.32 | **81.15 ± 3.03** | 77.65 ± 3.03 | 74.02 ± 5.96 | 75.72 ± 3.90 |
| −0.70 | 77.02 ± 2.83 | 78.62 ± 3.39 | 77.18 ± 8.22 | **81.20 ± 1.08** | 78.37 ± 3.73 | 74.47 ± 4.29 | 73.02 ± 1.52 |
| −1.00 | 78.12 ± 7.24 | 75.20 ± 3.66 | **80.88 ± 0.63** | 76.52 ± 7.06 | 74.15 ± 2.01 | 71.35 ± 6.01 | 71.43 ± 2.40 |
| −1.50 | 67.58 ± 2.68 | 72.02 ± 7.56 | 71.22 ± 6.19 | **72.58 ± 6.67** | 68.28 ± 4.04 | 64.02 ± 7.23 | 70.98 ± 11.35 |

**LR: 1e-3**

| K \ init | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 |
|---|---|---|---|---|---|---|---|
| −0.25 | 71.48 ± 8.71 | 75.07 ± 6.61 | **80.78 ± 2.82** | 69.87 ± 8.05 | 72.35 ± 5.70 | 69.02 ± 9.46 | 74.18 ± 3.33 |
| −0.30 | 75.22 ± 3.30 | 79.75 ± 4.90 | **82.25 ± 3.84** | 78.78 ± 2.61 | 75.98 ± 1.66 | 75.28 ± 2.36 | 74.58 ± 3.20 |
| −0.50 | 75.87 ± 3.32 | **82.68 ± 1.75** | 75.88 ± 3.02 | 81.95 ± 2.08 | 79.20 ± 2.08 | 75.08 ± 1.64 | 75.70 ± 5.38 |
| −0.70 | 75.47 ± 8.76 | 76.82 ± 10.58 | 73.18 ± 2.57 | **80.35 ± 3.72** | 76.98 ± 3.91 | 77.05 ± 0.96 | 75.32 ± 3.30 |
| −1.00 | **78.08 ± 6.31** | 72.43 ± 7.20 | 62.37 ± 7.65 | 72.47 ± 1.94 | 74.98 ± 1.74 | 75.70 ± 2.95 | 75.55 ± 6.22 |
| −1.50 | 74.43 ± 4.50 | 74.58 ± 4.72 | 74.05 ± 3.43 | **75.87 ± 1.90** | 75.30 ± 1.11 | 71.30 ± 8.75 | 70.20 ± 5.98 |

### Hard

**LR: 1e-4**

| K \ init | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 |
|---|---|---|---|---|---|---|---|
| −0.25 | 27.07 ± 9.43 | 29.58 ± 6.66 | **32.90 ± 10.30** | 28.78 ± 7.80 | 31.22 ± 7.24 | 30.88 ± 5.75 | 29.62 ± 3.71 |
| −0.30 | 27.65 ± 6.33 | 27.68 ± 11.01 | 32.30 ± 12.60 | **32.38 ± 6.48** | 29.35 ± 3.06 | 26.77 ± 4.05 | 27.18 ± 1.59 |
| −0.50 | 29.62 ± 2.35 | 26.40 ± 6.33 | 32.45 ± 9.86 | 21.53 ± 13.63 | **35.60 ± 3.00** | 27.40 ± 0.30 | 30.78 ± 7.10 |
| −0.70 | **33.83 ± 8.23** | 21.23 ± 3.96 | 19.37 ± 3.74 | 22.68 ± 9.25 | 32.27 ± 5.40 | 28.73 ± 3.65 | 28.92 ± 4.84 |
| −1.00 | 22.02 ± 5.18 | 14.93 ± 6.89 | **27.42 ± 2.78** | 16.13 ± 5.28 | 20.00 ± 10.87 | 26.18 ± 1.63 | 24.58 ± 5.41 |
| −1.50 | 17.03 ± 10.62 | 18.17 ± 9.22 | 19.22 ± 6.07 | 17.13 ± 3.74 | **22.12 ± 2.52** | 21.37 ± 11.21 | 17.75 ± 5.88 |

**LR: 3e-4**

| K \ init | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 |
|---|---|---|---|---|---|---|---|
| −0.25 | 35.38 ± 3.23 | 27.78 ± 5.09 | 34.55 ± 10.47 | 30.00 ± 5.31 | **37.10 ± 3.87** | 35.72 ± 2.39 | 35.98 ± 4.06 |
| −0.30 | 40.75 ± 3.26 | 38.35 ± 1.83 | **42.07 ± 9.92** | 33.95 ± 5.09 | 32.38 ± 4.03 | 34.72 ± 4.21 | 33.07 ± 11.53 |
| −0.50 | 39.27 ± 4.64 | 40.27 ± 4.02 | **46.22 ± 13.13** | 39.32 ± 3.88 | 37.87 ± 0.88 | 36.47 ± 6.15 | 37.47 ± 3.73 |
| −0.70 | 34.05 ± 3.13 | 35.68 ± 9.86 | **39.08 ± 8.57** | 33.25 ± 5.63 | 35.13 ± 2.91 | 33.52 ± 2.52 | 30.37 ± 3.34 |
| −1.00 | 30.12 ± 8.50 | 31.32 ± 3.78 | **40.37 ± 5.28** | 24.67 ± 14.41 | 27.77 ± 8.55 | 30.22 ± 4.45 | 29.93 ± 3.45 |
| −1.50 | 22.60 ± 8.98 | 29.30 ± 9.09 | **34.98 ± 9.02** | 26.38 ± 9.22 | 22.55 ± 7.05 | 19.83 ± 8.11 | 27.35 ± 5.69 |

**LR: 5e-4**

| K \ init | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 |
|---|---|---|---|---|---|---|---|
| −0.25 | 31.07 ± 1.50 | 22.97 ± 4.48 | **39.87 ± 1.18** | 29.30 ± 6.57 | 35.98 ± 8.62 | 34.78 ± 2.39 | 33.53 ± 1.70 |
| −0.30 | 38.52 ± 7.15 | 35.98 ± 9.83 | 35.73 ± 6.96 | 31.63 ± 6.72 | 33.58 ± 8.45 | 33.13 ± 3.96 | **38.97 ± 4.51** |
| −0.50 | **38.13 ± 3.93** | 30.87 ± 9.99 | 31.12 ± 4.38 | 36.72 ± 13.64 | 36.27 ± 0.32 | 36.10 ± 6.50 | 36.55 ± 0.91 |
| −0.70 | 37.22 ± 4.86 | 32.70 ± 8.64 | 33.72 ± 7.60 | **38.87 ± 1.64** | 36.47 ± 3.58 | 33.25 ± 6.02 | 37.32 ± 2.80 |
| −1.00 | 31.97 ± 8.97 | 27.20 ± 5.68 | 28.25 ± 9.19 | 30.67 ± 3.22 | 31.38 ± 6.68 | **33.35 ± 1.51** | 30.43 ± 3.02 |
| −1.50 | 31.47 ± 5.49 | **33.45 ± 3.22** | 26.37 ± 10.23 | 25.75 ± 9.44 | 25.78 ± 6.15 | 29.25 ± 1.98 | 25.63 ± 7.08 |

**LR: 1e-3**

| K \ init | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 |
|---|---|---|---|---|---|---|---|
| −0.25 | 23.52 ± 3.25 | 24.57 ± 8.28 | **35.22 ± 4.77** | 21.68 ± 4.37 | 23.58 ± 6.49 | 28.32 ± 2.60 | 20.00 ± 9.93 |
| −0.30 | 29.05 ± 9.84 | 30.97 ± 8.86 | **37.02 ± 4.35** | 22.67 ± 9.81 | 28.90 ± 4.06 | 29.22 ± 10.50 | 24.68 ± 4.79 |
| −0.50 | 32.73 ± 8.55 | 31.77 ± 7.03 | 31.00 ± 4.04 | **40.02 ± 4.54** | 33.98 ± 3.95 | 35.92 ± 3.06 | 25.78 ± 6.04 |
| −0.70 | 32.68 ± 0.93 | 32.03 ± 5.75 | 39.80 ± 6.88 | 35.02 ± 4.97 | **40.43 ± 4.93** | 32.80 ± 3.11 | 31.63 ± 3.82 |
| −1.00 | 31.03 ± 2.95 | 22.68 ± 11.30 | 28.78 ± 4.01 | 28.33 ± 13.83 | 27.98 ± 4.91 | 27.47 ± 7.82 | **34.55 ± 0.75** |
| −1.50 | 21.60 ± 10.49 | 20.63 ± 0.49 | 22.73 ± 5.66 | 22.35 ± 4.11 | 21.90 ± 7.25 | **25.55 ± 8.19** | 25.40 ± 7.52 |