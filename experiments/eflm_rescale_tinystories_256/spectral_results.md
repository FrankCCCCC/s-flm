# Word-Embedding Spectra — EFLM Rescale + Naive Baselines

Singular/eigenvalue spectrum of the word-embedding matrix `E` (`[V, d]` =
`[50258, 768]`), using the EMA weights as `trainer.validate()` sees them
(`last.ckpt`). All runs: **seed 1, 30K steps, seq 256, global batch 512**.

`E` is rectangular, so it has no eigenvalues; the two well-defined quantities are
the `d` singular values `sigma_i(E)` and the Gram eigenvalues
`lambda_i(E^T E) = sigma_i^2`. **Stable rank** is `||E||_F^2 / sigma_max^2` — an
effective dimension: `d` for a perfectly isotropic codebook, `1` for a rank-1 one.

Two centering variants, both applied row-wise over the vocabulary:

- **Normalized** — every row projected onto the unit sphere, `normalize(E)`.
- **Normalized-Mean-Shift** — normalize *first*, then subtract the mean
  direction, `normalize(E) - mean_rows(normalize(E))`.

**LR is not held constant across the two source projects.** The five baselines
come from `naive_ar_tinystories_s256` at LR 1e-3; `eflm_rescale_tinystories_256`
fixes LR at 3e-4 for every cell (`sweep.py`), so no LR-matched EFLM run exists.
Each row carries its own LR below — the EFLM comparison is confounded by it and
should be read as indicative, not controlled.

| method | run | project | lr |
|---|---|---|---|
| AR / MDLM / DUO / FLM / S-FLM+Ada+Trunc | `m-{ar,mdlm,duo,flm,sfmta}_lr-1e-3_sd-1` | `naive_ar_tinystories_s256` | 1e-3 |
| EFLM+Ada+Trunc (ρ=1) | `eflmrs256_trunc_ada_r-1_rs1` | `eflm_rescale_tinystories_256` | 3e-4 |

## Normalized spectrum (seed=1, 30K steps, EMA)

| method | lr | σ_median | σ_max | σ_min | λ_median | λ_max | stable rank |
|---|---|---|---|---|---|---|---|
| AR | 1e-3 | 6.360 | 30.69 | 5.010 | 40.45 | 941.9 | 53.36 |
| MDLM | 1e-3 | 6.435 | 31.62 | 5.256 | 41.41 | 1000.0 | 50.26 |
| DUO | 1e-3 | 6.820 | 44.79 | 1.919 | 46.51 | 2006 | 25.05 |
| FLM | 1e-3 | 6.139 | 21.88 | 3.116 | 37.69 | 478.9 | 104.94 |
| S-FLM+Ada+Trunc | 1e-3 | 5.552 | 82.23 | 4.581 | 30.82 | 6762 | 7.43 |
| **EFLM+Ada+Trunc (ρ=1)** | 3e-4 | 5.871 | 106.7 | 4.910 | 34.47 | 11374 | **4.42** |

## Normalized-Mean-Shift spectrum (seed=1, 30K steps, EMA)

| method | lr | σ_median | σ_max | σ_min | λ_median | λ_max | stable rank |
|---|---|---|---|---|---|---|---|
| AR | 1e-3 | 6.360 | 27.37 | 5.010 | 40.45 | 749.3 | 66.61 |
| MDLM | 1e-3 | 6.435 | 28.48 | 5.256 | 41.41 | 811.2 | 61.64 |
| DUO | 1e-3 | 6.820 | 28.50 | 1.919 | 46.51 | 812.5 | 60.27 |
| FLM | 1e-3 | 6.139 | 21.88 | 3.116 | 37.69 | 478.9 | 104.94 |
| S-FLM+Ada+Trunc | 1e-3 | 5.552 | 60.49 | 4.581 | 30.82 | 3659 | 12.88 |
| **EFLM+Ada+Trunc (ρ=1)** | 3e-4 | 5.871 | 76.74 | 4.910 | 34.46 | 5888 | **7.60** |

## Observations

**EFLM+Ada+Trunc has the most anisotropic codebook of the six.** Stable rank
4.42 normalized — 1.7× below S-FLM+Ada+Trunc, 5.7× below DUO, 24× below FLM — and
it stays lowest after centering (7.60). Its `λ_max` of 11374 is 24× FLM's. The
two flow methods with Ada+Trunc occupy the bottom two slots in both tables, well
separated from the three discrete baselines.

**For this run the `normalized` row is not a diagnostic transform — it is the
operational codebook.** `eflmrs256_trunc_ada_r-1_rs1` has
`rho_min = rho_max = 1`, so `rescale_radius` pins every row to unit norm and the
flow embeds into exactly `normalize(E)` (cf. `EFLM.q_xt` / `_sc_embed_table`).
For AR/MDLM/DUO/FLM, normalization is purely analytic and the model uses the raw
`E`. That asymmetry is worth keeping in mind when comparing the normalized table
across the two families.

**Mean-centering is a rank-1 edit, and it shows.** Across both tables the median,
the min, and (not shown) the whole bulk are unchanged to 5-6 significant figures
— AR `σ_median` 6.3603 → 6.36013, DUO 6.82017 → 6.81982, EFLM 5.87091 → 5.87068.
Subtracting the mean direction removes one rank-1 component `1·μ^T`; it moves
`σ_max` and nothing else. Every stable-rank change between the two tables is
therefore attributable to the top singular value alone.

**The bulk is method-independent; only the tail separates the methods.**
`σ_median` spans just 5.55–6.82 (1.23×) across all six methods — EFLM's 5.871
sits mid-pack — while `σ_max` spans 21.9–106.7 (4.9×), `λ_max` 479–11374 (24×),
and stable rank 4.42–104.94 (24×). The entire spread is a tail phenomenon; no
method differs meaningfully in where the typical singular value sits.

**The size of the mean direction predicts how much centering helps.**
Measured `||mean(normalize(E))||`: EFLM+Ada+Trunc 0.331, S-FLM+Ada+Trunc 0.250,
DUO 0.160, AR 0.083, MDLM 0.071, FLM 0.0065. EFLM has the largest mean direction
of the six, and centering gives it the largest absolute `λ_max` drop
(11374 → 5888). DUO gains the most in *relative* stable rank (25.05 → 60.27,
2.4×); FLM gains nothing — its mean is ~0, so its two rows are identical to 4
s.f. (they differ only in the 5th: 104.938 vs 104.936).

**Centering does not rescue either Ada+Trunc method.** EFLM 4.42 → 7.60 and
S-FLM 7.43 → 12.88 — both roughly double, yet both stay 5-9× below the discrete
baselines. Their anisotropy is genuinely multi-directional, not one rogue mean
direction. The figures show the signature: a narrow lognormal bulk with a hard
right edge, then dozens of fully detached outliers and an empty gap between.

**DUO reaches its low normalized stable rank differently.** DUO has the *highest*
median of the six and the smallest `σ_min` (1.919) — a broad continuous bulk with
no spectral gap, plus two isolated outliers. Once the mean is removed it sits
with AR and MDLM (60.27 vs 66.61 / 61.64). Its normalized-table position is
driven almost entirely by the mean direction, unlike the Ada+Trunc methods.

**AR and MDLM are near-interchangeable** on every statistic in both tables.

**FLM is the most isotropic** (stable rank 104.94, ~2× AR/MDLM) and the only one
with a bimodal spectrum — a main lobe at σ≈4–7 plus a second cluster at σ≈12–18.

## Figures

- EFLM: `experiments/eflm_rescale_tinystories_256/imgs/codebook_eigen_dist_eflmrs256_trunc_ada_r-1_rs1_{variant}[_logx].png`
- Baselines: `experiments/naive_ar_tinystories_s256/figures/codebook_eigen_dist_{run}_{variant}[_logx].png`

`variant` in `{raw, rescaled, normalized, normalized-mean-shift,
mean-shift-normalized}` (EFLM) / `{normalized, normalized-mean-shift,
mean-shift-normalized}` (baselines). Read the `_logx` pass — the spectrum is
heavy-tailed, so linear bins put ~99% of the values in the leftmost few.

Raw numbers: `spectral_stats.json` here (EFLM) and in
`experiments/naive_ar_tinystories_s256/` (baselines).

## Reproduce

CPU-only, but run it on a compute node:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python visualization/codebook_eigen_dist.py \
  --ckpt outputs/eflm_rescale_tinystories_256/eflmrs256_trunc_ada_r-1_rs1/checkpoints/last.ckpt \
  --variants normalized,normalized-mean-shift \
  --out experiments/eflm_rescale_tinystories_256/imgs/codebook_eigen_dist_eflmrs256_trunc_ada_r-1_rs1
```
