# Word-Embedding Spectra — naive_ar_tinystories_s256

Singular/eigenvalue spectrum of the word-embedding matrix `E` (`[V, d]` =
`[50258, 768]`) for each method at **LR = 1e-3, seed = 1, 30K steps**, using the
EMA weights as `trainer.validate()` sees them (`last.ckpt`).

`E` is rectangular, so it has no eigenvalues; the two well-defined quantities are
the `d` singular values `sigma_i(E)` and the Gram eigenvalues
`lambda_i(E^T E) = sigma_i^2`. **Stable rank** is `||E||_F^2 / sigma_max^2` — an
effective dimension: `d` for a perfectly isotropic codebook, `1` for a rank-1 one.

Two centering variants, both applied row-wise over the vocabulary:

- **Normalized** — every row projected onto the unit sphere,
  `normalize(E)`.
- **Normalized-Mean-Shift** — normalize *first*, then subtract the mean
  direction, `normalize(E) - mean_rows(normalize(E))`.

## Normalized spectrum (lr=1e-3, seed=1, 30K steps, EMA)

| method | σ_median | σ_max | σ_min | λ_median | λ_max | stable rank |
|---|---|---|---|---|---|---|
| AR | 6.360 | 30.69 | 5.010 | 40.45 | 941.9 | 53.36 |
| MDLM | 6.435 | 31.62 | 5.256 | 41.41 | 1000.0 | 50.26 |
| DUO | 6.820 | 44.79 | 1.919 | 46.51 | 2006 | 25.05 |
| FLM | 6.139 | 21.88 | 3.116 | 37.69 | 478.9 | 104.94 |
| S-FLM+Ada+Trunc | 5.552 | 82.23 | 4.581 | 30.82 | 6762 | 7.43 |

## Normalized-Mean-Shift spectrum (lr=1e-3, seed=1, 30K steps, EMA)

| method | σ_median | σ_max | σ_min | λ_median | λ_max | stable rank |
|---|---|---|---|---|---|---|
| AR | 6.360 | 27.37 | 5.010 | 40.45 | 749.3 | 66.61 |
| MDLM | 6.435 | 28.48 | 5.256 | 41.41 | 811.2 | 61.64 |
| DUO | 6.820 | 28.50 | 1.919 | 46.51 | 812.5 | 60.27 |
| FLM | 6.139 | 21.88 | 3.116 | 37.69 | 478.9 | 104.94 |
| S-FLM+Ada+Trunc | 5.552 | 60.49 | 4.581 | 30.82 | 3659 | 12.88 |

## Observations

**Mean-centering is a rank-1 edit, and it shows.** Across both tables the median,
the min, and (not shown) the whole bulk are unchanged to 5-6 significant figures
— e.g. AR `σ_median` 6.3603 → 6.36013, DUO 6.82017 → 6.81982. Subtracting the
mean direction removes one rank-1 component `1·μ^T`; it moves `σ_max` and nothing
else. Every stable-rank change between the two tables is therefore attributable
to the top singular value alone.

**The bulk is method-independent; only the tail separates the methods.**
`σ_median` spans just 5.55–6.82 (1.23×) across all five methods, while `σ_max`
spans 21.9–82.2 (3.8×) and `λ_max` 479–6762 (14×). The 14× spread in stable rank
is entirely a tail phenomenon.

**S-FLM+Ada+Trunc has by far the most anisotropic codebook.** Stable rank 7.43
normalized — 3.4× below DUO and 14× below FLM — and it stays lowest (12.88) after
centering. It is simultaneously the extreme at both ends: the *lowest* median
(5.552, ~13% below DUO) and the *highest* max (82.23). Its figures show a narrow
lognormal bulk at σ≈4–6 with a hard right edge, then ~35 fully detached outliers,
with nothing occupying σ≈8–15. The anisotropy is multi-directional, not one rogue
mean direction: centering roughly doubles its stable rank but leaves it 5-8×
below the discrete baselines.

**The size of the mean direction predicts how much centering helps.**
Measured `||mean(normalize(E))||`: S-FLM+Ada+Trunc 0.250, DUO 0.160, AR 0.083,
MDLM 0.071, FLM 0.0065. DUO gains the most from centering (stable rank
25.05 → 60.27, 2.4×) and FLM the least — its mean is ~0, so the two tables are
identical for it to 4 s.f. (they differ only in the 5th: 104.938 vs 104.936).

**DUO reaches its low normalized stable rank differently from S-FLM+Ada+Trunc.**
DUO has the *highest* median of the five and the smallest `σ_min` (1.919) — a
broad continuous bulk with no spectral gap, plus two isolated outliers. Once the
mean is removed it sits with AR and MDLM (60.27 vs 66.61 / 61.64). Its
normalized-table position is driven almost entirely by the mean direction.

**AR and MDLM are near-interchangeable** on every statistic in both tables.

**FLM is the most isotropic** (stable rank 104.94, ~2× AR/MDLM) and the only one
with a bimodal spectrum — a main lobe at σ≈4–7 plus a second cluster at σ≈12–18.

## Figures

`experiments/naive_ar_tinystories_s256/figures/codebook_eigen_dist_{run}_{variant}[_logx].png`
for `run` in `m-{ar,mdlm,duo,flm,sfmta}_lr-1e-3_sd-1` and `variant` in
`{normalized, normalized-mean-shift, mean-shift-normalized}`. Read the `_logx`
pass — the spectrum is heavy-tailed, so linear bins put ~99% of the values in the
leftmost few.

Raw numbers backing both tables: `spectral_stats.json` (same directory).

## Reproduce

CPU-only, but run it on a compute node:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python visualization/codebook_eigen_dist.py \
  --ckpt outputs/naive_ar_tinystories_s256/m-sfmta_lr-1e-3_sd-1/checkpoints/last.ckpt \
  --variants normalized,normalized-mean-shift \
  --out experiments/naive_ar_tinystories_s256/figures/codebook_eigen_dist_m-sfmta_lr-1e-3_sd-1
```
