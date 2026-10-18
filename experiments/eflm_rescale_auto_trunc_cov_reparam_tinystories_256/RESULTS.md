# eflm_rescale_auto_trunc_cov_reparam_tinystories_256 — Results

**Status 2026-10-07: COMPLETE.** 33 / 36 cells evaluated; 3 lr-5e-3 cells diverged to NaN during training (no eval). Design: `EXPERIMENT.md`.
Table regenerated with `python report.py` (entropy < 3.0 = degenerate samples; `val/ppl` is not comparable across arms).

| noise | R | LR | valid PPL (bound) | GenPPL | entropy | note |
|---|---|---|---|---|---|---|
| auto | 0.1 | 1e-3 | 120.14 | 58.05 | 1.45 | collapse? |
| auto | 0.1 | 3e-4 | 172.70 | 45447.18 | 0.38 | collapse? |
| auto | 0.1 | 5e-3 | nan | 1.09 | 0.00 | collapse? |
| auto | 0.5 | 1e-3 | 41.59 | 65.23 | 3.66 |  |
| auto | 0.5 | 3e-4 | 27.89 | 15.48 | 3.84 |  |
| auto | 0.5 | 5e-3 | 72.81 | 58.62 | 1.43 | collapse? |
| auto | 1.0 | 1e-3 | 24.43 | 32.76 | 3.72 |  |
| auto | 1.0 | 3e-4 | 19.00 | 15.40 | 3.89 |  |
| auto | 1.0 | 5e-3 | 55.65 | 86.76 | 1.48 | collapse? |
| auto | 8.0 | 1e-3 | 27.37 | 1006.83 | 1.48 | collapse? |
| auto | 8.0 | 3e-4 | 8.35 | 19.20 | 3.91 |  |
| auto | 8.0 | 5e-3 | nan | 1.09 | 0.00 | collapse? |
| autoada | 0.1 | 1e-3 | 111.39 | 57053.41 | 0.15 | collapse? |
| autoada | 0.1 | 3e-4 | 43.19 | 56.89 | 3.25 |  |
| autoada | 0.5 | 1e-3 | 34.96 | 79.29 | 3.08 |  |
| autoada | 0.5 | 3e-4 | 22.93 | 16.27 | 3.85 |  |
| autoada | 1.0 | 1e-3 | 35.11 | 84.70 | 1.55 | collapse? |
| autoada | 1.0 | 3e-4 | 19.88 | 17.73 | 3.77 |  |
| autoada | 1.0 | 5e-3 | 172.93 | 167867.55 | 0.47 | collapse? |
| autoada | 8.0 | 1e-3 | 11.49 | 14.45 | 3.90 |  |
| autoada | 8.0 | 3e-4 | 11.87 | 16.36 | 3.90 |  |
| autoada | 8.0 | 5e-3 | 23.60 | 83.97 | 1.53 | collapse? |
| llada | 0.1 | 1e-3 | 184.78 | 25.68 | 1.84 | collapse? |
| llada | 0.1 | 3e-4 | 127.90 | 23.04 | 3.69 |  |
| llada | 0.5 | 1e-3 | 82.51 | 66.17 | 1.45 | collapse? |
| llada | 0.5 | 3e-4 | 52.14 | 15.10 | 3.85 |  |
| llada | 0.5 | 5e-3 | 93.86 | 28425.18 | 0.77 | collapse? |
| llada | 1.0 | 1e-3 | 41.13 | 55.63 | 3.87 |  |
| llada | 1.0 | 3e-4 | 35.17 | 15.29 | 3.80 |  |
| llada | 1.0 | 5e-3 | 132.92 | 8825.67 | 0.23 | collapse? |
| llada | 8.0 | 1e-3 | 13.72 | 14.79 | 3.92 |  |
| llada | 8.0 | 3e-4 | 13.97 | 15.29 | 3.89 |  |
| llada | 8.0 | 5e-3 | inf | 1.15 | 0.00 | collapse? |

## Headline

**No: the gaussian readout does not improve the TinyStories GenPPL-entropy frontier.** Pareto front over
non-degenerate cells (entropy >= 3.0; lower GenPPL, higher entropy):

| readout | front (GenPPL / entropy) | non-degenerate cells |
|---|---|---|
| gaussian (this sweep) | autoada R=8 lr 1e-3 **14.45 / 3.90**; llada R=8 lr 1e-3 14.79 / 3.92 | 17 / 33 |
| naive (earlier sweeps, same truncation) | 10.36 / 3.58, 10.74 / 3.71, 10.86 / 3.85, 11.72 / 3.86, 13.96 / 4.01 | 50 / 57 |

Every gaussian frontier point is dominated by a naive one (13.96 / 4.01 beats 14.45 / 3.90 on both axes). The
gaussian readout only reaches parity at R = 8, where its geometry term is weak; at R <= 1 it is 20-65 % worse at
lr 3e-4 and collapses at lr >= 1e-3. Every lr 5e-3 cell is degenerate or diverged.

## Early reads (1 seed)

Naive-readout reference at the same truncation tau*(R), lr 1e-3, plain CE, no cov weight
(`eflm_rescale_auto_trunc_tinystories_256`, stage 1): R=0.1 **12.25** (3.76), R=0.5 **11.53** (3.81), R=1 **11.72** (3.86).

| cell | this sweep (gaussian) | naive reference | read |
|---|---|---|---|
| auto R=0.1 | 58.05 (1.45) | 12.25 (3.76) | collapsed: punctuation / stop words |
| auto R=0.5 | 65.23 (3.66) | 11.53 (3.81) | story structure, many dropped tokens |
| auto R=1.0 | 32.76 (3.72) | 11.72 (3.86) | ~2.8x worse |
| auto R=8.0 | 1006.8 (1.48) | 15.39 (3.88) | collapsed |
| autoada R=0.1 | 57053 (0.15) | — | collapsed: every token is `<|endoftext|>` |
| autoada R=0.5 | 79.29 (3.08) | 10.74 (parent sweep, 1 seed) | ~7x worse, entropy near the bar |
| autoada R=1.0 | 84.70 (1.55) | — | collapsed |
| llada R=0.1 | 25.68 (1.84) | — | collapsed |
| llada R=0.5 | 66.17 (1.45) | — | collapsed |
| llada R=1.0 | 55.63 (3.87) | 10.36 (naive llada, R=1, project incumbent) | ~5.4x worse |
| llada R=8.0 | **14.79 (3.92)** | 15.39 (3.88) (naive *auto*) | healthy, like autoada R=8 |
| autoada R=8.0 | **14.45 (3.90)** | 15.39 (3.88) (naive *auto*, no adaptive) | first healthy cell; on par with / slightly better than naive |

- **Exception: R=8** (`autoada` 14.45, entropy 3.90; `llada` 14.79, entropy 3.92). At R=8 tau* = 0.505, so b_t >= 0.6 and beta_t = a_t/b_t^2 <= 1.1:
  the geometry term is weak and the model is close to the naive one. The damage tracks beta_t (small R, long horizon).
  But `auto` R=8 (same tau*, plus the cov weight) collapsed (entropy 1.48), so at R=8 the cov weight, not the readout,
  is the likely culprit. At lr 3e-4, however, `auto` R=8 is healthy (19.20, entropy 3.91; naive at lr 3e-4: 18.29),
  so the R=8 + cov-weight collapse is an LR 1e-3 instability rather than a hard failure.
- Every other gaussian-readout cell so far is far worse than the naive readout at matched R and LR, and the gap grows as
  R shrinks (larger tau*, larger beta_t = a_t/b_t^2, up to ~2800 at R=0.1).
- `autoada` has no cov weight, so the cov weight alone does not explain the collapse at R=0.1; the gaussian readout
  is implicated. Whether this is the method or an implementation issue is not yet diagnosed.

## Diagnostic: gaussian vs naive denoiser (2026-10-03, job 921628)

Script: session scratchpad `diag/readout_diag.py`. 64 validation sequences, EMA weights, R=0.5, lr 1e-3, tau_max 2.4436.
Gaussian = `ns-auto_r-0.5_lr-1e-3` (this sweep, cov weight on); naive = `eflmratr_lr-1e-3_r-0.5_m-1.0`
(`eflm_rescale_auto_trunc_tinystories_256`, no cov weight). Top-1 accuracy of the denoiser's posterior per training time t
(alpha = signal level); for the gaussian model also the residual logits alone and the geometry term alone (uniform prior).

| t | alpha | gaussian full | gaussian residual only | geometry only | naive full |
|---|---|---|---|---|---|
| 0.02 | 0.909 | 0.938 | 0.446 | 0.540 | 0.967 |
| 0.10 | 0.889 | 0.846 | 0.425 | 0.263 | 0.934 |
| 0.20 | 0.859 | 0.678 | 0.381 | 0.065 | 0.863 |
| 0.35 | 0.796 | 0.423 | 0.284 | 0.006 | 0.644 |
| 0.50 | 0.706 | 0.222 | 0.164 | 0.001 | 0.298 |
| 0.90 | 0.218 | 0.076 | 0.076 | 0.000 | 0.077 |

Embedding table (word embeddings, norm pinned to 0.5):

| | mean pairwise cos | nearest-neighbour cos (median / p90) | share of words with a NN at cos > 0.9 |
|---|---|---|---|
| gaussian | 0.001 | 0.16 / 0.29 | 0.2% |
| naive | 0.113 | 0.81 / 0.97 | 43% |

Reads:
- The readout itself behaves as designed: the full posterior is better than either part, and the geometry term alone
  is the exact uniform-prior posterior (54% at the clean end of this truncation, falling fast with noise).
- The gaussian model is a **worse denoiser at every t**, worst in the mid-noise range that drives generation
  (t = 0.2-0.35: 0.68 / 0.42 vs 0.86 / 0.64). That is consistent with the GenPPL gap; it is not a crash or a wiring bug.
- The embeddings stay near-random (NN cos 0.16), whereas the naive model packs 43% of words next to a near-duplicate.
  So the readout does prevent the embedding "collapse" it was built for, but on TinyStories that clustering seems to
  be what makes the naive denoiser strong at mid noise: with spread embeddings the noised state carries less shared
  semantic signal, and the residual network has to recover the token from context alone.
- Caveat: this gaussian checkpoint also has the cov weight; the `autoada` cells (no cov weight) are just as bad on
  GenPPL, so the cov weight is not the main factor.

## LR dependence (added 2026-10-04 02:30)

At lr 3e-4 the gaussian cells recover much of the gap seen at lr 1e-3:

| cell | lr 1e-3 | lr 3e-4 | naive, lr 3e-4 |
|---|---|---|---|
| auto R=0.5 | 65.23 (3.66) | **15.48 (3.84)** | 11.81 (3.82) |
| auto R=1.0 | 32.76 (3.72) | **15.40 (3.89)** | 12.56 (3.83) |
| auto R=8.0 | 1006.8 (1.48) | **19.20 (3.91)** | 18.29 (3.89) |
| autoada R=0.1 | 57053 (0.15) | 56.89 (3.25) | — |
| autoada R=1.0 | 84.70 (1.55) | **17.73 (3.77)** | — |
| autoada R=8.0 | **14.45 (3.90)** | 16.36 (3.90) | — |
| llada R=0.1 | 25.68 (1.84) | 23.04 (3.69) | best R=0.1 cell so far; still ~2x the naive R=0.1 (12.1) |
| llada R=1.0 | 55.63 (3.87) | **15.29 (3.80)** | 10.36 (naive llada R=1, its own LR) |
| autoada R=0.5 | 79.29 (3.08) | **16.27 (3.85)** | — (naive autoada R=0.5: 10.74 at its own LR) |
| auto R=0.1 | 58.05 (1.45) | 45447 (0.38) | 12.12 (3.61) |

The gaussian readout is much more LR-sensitive than the naive one (which is flat between 3e-4 and 1e-3): its
logits carry the beta_t-scaled geometry term, so updates to the shared embedding table move every logit at once.
lr 1e-3 (the naive optimum) is too high for it; R=0.1 fails at both LRs.

## lr 5e-3

- `auto` R=0.1, lr 5e-3: training **diverged to NaN** (model_output NaN logged in epoch 4, ~step 17k); the eval of the
  NaN checkpoint gives GenPPL 1.09 / entropy 0.00 (degenerate). The naive readout also collapsed at lr 5e-3 for R <= 1
  (`eflm_rescale_auto_trunc_tinystories_256` stage 1), so this column is expected to be unusable at small R.
- Every lr 5e-3 cell evaluated so far is degenerate (entropy <= 1.53), including R=8.
- `autoada` R=0.1 / R=0.5 and `llada` R=0.1 at lr 5e-3 **diverged to NaN during training** and the job died
  (the adaptive refit cannot run on NaN losses), so they have no eval. Not resubmitted: resuming would restart
  from the NaN checkpoint. Counted as diverged.


## Gaussian vs naive readout, matched noise / R / LR (2026-10-04 15:50)

Naive = earlier runs, same truncation tau*(R) / alpha*(R) and protocol: `auto` from `eflm_rescale_auto_tinystories_256` (lr 3e-4) and
`eflm_rescale_auto_trunc_tinystories_256` (lr 1e-3 / 5e-3) — **without** the cov weight; `autoada` / `llada` from
`eflm_rescale_auto_tinystories_256` (lr 3e-4 only). All single seed. ⚠ = entropy < 3.0 (degenerate).

| noise | R | LR | gaussian: GenPPL / entropy | naive: GenPPL / entropy |
|---|---|---|---|---|
| auto | 0.1 | 3e-4 | 45447.18 / 0.38 ⚠ | 12.12 / 3.61 |
| auto | 0.1 | 1e-3 | 58.05 / 1.45 ⚠ | 12.25 / 3.76 |
| auto | 0.1 | 5e-3 | 1.09 / 0.00 ⚠ | 1.82 / 0.35 ⚠ |
| auto | 0.5 | 3e-4 | 15.48 / 3.84 | 11.81 / 3.82 |
| auto | 0.5 | 1e-3 | 65.23 / 3.66 | 11.53 / 3.81 |
| auto | 0.5 | 5e-3 | — | 1.11 / 0.00 ⚠ |
| auto | 1.0 | 3e-4 | 15.40 / 3.89 | 12.56 / 3.83 |
| auto | 1.0 | 1e-3 | 32.76 / 3.72 | 11.72 / 3.86 |
| auto | 1.0 | 5e-3 | — | 4.30 / 0.63 ⚠ |
| auto | 8.0 | 3e-4 | 19.20 / 3.91 | 18.29 / 3.89 |
| auto | 8.0 | 1e-3 | 1006.83 / 1.48 ⚠ | 15.39 / 3.88 |
| auto | 8.0 | 5e-3 | — | 13.96 / 4.01 |
| autoada | 0.1 | 3e-4 | 56.89 / 3.25 | — |
| autoada | 0.1 | 1e-3 | 57053.41 / 0.15 ⚠ | — |
| autoada | 0.5 | 3e-4 | 16.27 / 3.85 | 10.74 / 3.71 |
| autoada | 0.5 | 1e-3 | 79.29 / 3.08 | — |
| autoada | 1.0 | 3e-4 | 17.73 / 3.77 | 11.52 / 3.78 |
| autoada | 1.0 | 1e-3 | 84.70 / 1.55 ⚠ | — |
| autoada | 8.0 | 3e-4 | 16.36 / 3.90 | 16.09 / 3.90 |
| autoada | 8.0 | 1e-3 | 14.45 / 3.90 | — |
| llada | 0.1 | 3e-4 | 23.04 / 3.69 | — |
| llada | 0.1 | 1e-3 | 25.68 / 1.84 ⚠ | — |
| llada | 0.5 | 3e-4 | — | 12.02 / 3.59 |
| llada | 0.5 | 1e-3 | 66.17 / 1.45 ⚠ | — |
| llada | 1.0 | 3e-4 | 15.29 / 3.80 | 10.36 / 3.58 |
| llada | 1.0 | 1e-3 | 55.63 / 3.87 | — |
| llada | 8.0 | 1e-3 | 14.79 / 3.92 | — |
