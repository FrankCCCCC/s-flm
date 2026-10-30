# eflm_cov_reparam_fisher_cond — Results

**Status 2026-10-19: complete** (6/6 runs, v2 script; every number re-checked against the JSONs by an independent review,
workflow `wf_f03fef5b-344`). Design and definitions: `EXPERIMENT.md`. Tables at 9 t values: `report_out.md`
(`python report.py`). Full 51-point curves (t step 0.02) for all 6 models: `fisher_cond_state_t.png` (column 1 = cond),
`fisher_cond_state_snr.png` (same curves vs SNR), `fisher_cond_bar.png` (state-averaged), and
`results/<label>.json` → `rows[i].{uni,flat,ctx}`.
Addendum 2026-10-19 (v3 rerun, `results_v3/`): Shannon erank, stable rank and three-timestep tables, see
"Three timesteps" below; "effective rank" was renamed PR throughout.
Addendum 2026-10-19 (setup.md): the slides' angular-Fisher Sigma_t on the tangent space, see "Angular-Fisher Sigma_t"
below (`visualization/angular_fisher_sigma_t.py`, `figures/angular_fisher_sigma_t_{t,snr}.png`).
Addendum 2026-10-22 (GOAL2, complete 54/54): the six models at embedding dim 16 / 48 / 768 x 3 seeds, with
GenPPL, train / valid NELBO, and per-dim angular-Fisher and codebook figures. See "GOAL2" at the end.

TinyStories EFLM, `noise=autonomous` truncated at tau*(R), lr 3e-4, R in {0.1, 0.5, 1.0}. Naive readout (plain CE)
vs Gaussian readout (|alpha'|-weighted CE). Sigma_t = Cov_{q(v|x_t)}[e_v] is the word-posterior Fisher up to
beta_t^2, so cond(F_t) = cond(Sigma_t). d = 768, V = 50258. t = 0 is the truncated clean end, the slides' "far from
the origin". t = 1 is pure noise, the slides' near-origin limit, where Sigma_t = Sigma_data. Two posteriors:
- `uni` is the slides' object: the exact posterior under the train-unigram prior q_data.
- `ctx` is a contextual generalisation: the model's p_theta(x^i | x_t) on validation text.

PR = (sum lambda_i)^2 / sum lambda_i^2 is the participation ratio, exp of the Renyi-2 entropy of p_i = lambda_i / tr.
Earlier versions of this file called it "effective rank". It is not Roy & Vetterli's erank = exp(Shannon entropy of
p), which weights the tail more: 1 ≤ tr/lambda_1 ≤ PR ≤ erank ≤ rank ≤ d. PR is the measure behind the KL-variance
argument: for an isotropic error, Var(KL) / E[KL]^2 = 2 / PR.

gauss R=0.1 is a collapsed model (GenPPL 45447, entropy 0.38). Its codebook is indistinguishable from a random
isotropic one on every control, so it is reported separately from the five non-collapsed runs.

## Answer: yes, the KL is strongly skewed into a few directions at every t

- **Non-collapsed runs (5):** at every t, in both `uni` and `ctx`, 90% of the local KL energy lies in at most 12 of
  the 768 directions (median per-state k90 2-12).
- **All six runs:** the top eigenvalue is 58-661x the mean eigenvalue.
- **Exception:** the collapsed gauss R=0.1 at t ≳ 0.3 has one '.' spike on a nearly isotropic bulk (k90 250-360), the
  same as a random codebook with the real frequencies.

Which directions depends on where you are along the trajectory:

1. **Clean end: the axis between the true word and its competitors, which changes from state to state.**
   - *At t = 0:* posterior perplexity is 1.0-4.1 (`uni`) and 1.0 (`ctx`). Sigma_t ≈ sum q_i q_j (e_i - e_j)(e_i -
     e_j)^T, with PR 1.3-3. Its top eigenvector points from the posterior mean to the true word: median
     |cos(u1, e_true - mu_t)| = 0.94-1.00 in both modes.
   - *How long it lasts:* in `ctx` (non-collapsed) until t ≈ 0.2-0.3. In `uni` the posterior spreads by t = 0.1-0.2:
     PR 3.5-5.1, and |cos| falls to 0.59-0.90 at t = 0.2.
   - *What cond means here:* cond is 10^6.9 to ≥10^12. It is set by directions holding under 1% of the energy, i.e.
     by how deep the posterior tail is, so k90, k99 and PR are the meaningful measures.
2. **Noise end (t ≳ 0.5 for `uni`, t ≳ 0.7-0.8 for `ctx`): a shared subspace of about 10 frequent-word directions.**
   Sigma_t → Sigma_data, with k90 7-12 (non-collapsed). Its top eigenvector is a direction shared by the most frequent
   words. '.' (7.25% of train tokens) contributes 31-46% of lambda_1, the top 10 words 50-64%, and the top 100 69-84%
   (the collapsed R=0.1: 88 / 98 / 99%).
3. **The readout changes the tail, not the head.**
   - *Head (same for both):* at R ≥ 0.5, both readouts put 90% of the noise-end energy in the same number of
     directions: k90 is 10 / 12 for both, and PR is 10.4 / 12.1 (naive) vs 10.3 / 12.1 (gauss). The Shannon erank,
     which weights the tail more, is 10.8 / 12.6 vs 14.9 / 17.7: under that measure gaussian Sigma_data is about 1.4x
     higher-rank, and that difference also comes from the tail.
   - *Tail (different):* naive leaves under 1% of the trace outside 11-13 directions. Gaussian spreads its last 7-15%
     over about 470 directions (k99 11-13 vs 465-487). That is why cond(Sigma_data) is 10^6.6-10^6.7 for naive vs
     10^3.8-10^3.9 for gaussian, a gap of 2.7-2.9 decades (about 500-750x).
   - *Along the trajectory:* in `ctx` for t ≤ 0.3, k90 (2-5) and PR (1.3-2.7) match across readouts, but
     cond still differs by 1.5-2.4 decades.
   - *Confounds:* the readout comes with a different CE weighting and code version, and there is one seed (see
     Caveats).
4. **Word frequency is a major cause, but mostly through where the frequent words sit in the learned codebook.**
   - *Overall:* a uniform prior on the same codebook lowers cond(Sigma_data) by 3.8-4.0 decades (naive) and 2.3-2.6
     (gaussian).
   - *Gaussian R ≥ 0.5:* frequency and word placement together create the head.
     - Uniform prior: lambda_1/mean 20-22, PR 225-260, k90 about 650.
     - Permuted pi (same frequency spectrum, words shuffled): 56, 58-59, about 300.
     - Actual: 111-130, 10-12, 10-12.
   - *Naive:* the geometry alone already gives a strong head. Under a uniform prior, lambda_1/mean is 58-119, the top
     10 directions hold 58-64% of the trace, and PR is 16-25. The real frequencies then concentrate the
     rest (k90 396-464 → 7-12) and remove the low-energy directions. That is where its extra decades of cond come
     from: +2.5 over permuted pi.
   - *The slides' mechanism:* one frequent word far from the mean gives the slides' bound of 10^1.8 for every
     codebook, which is about the lambda_1/mean level. It explains neither the 3.8-decade spread between runs nor
     lambda_min.

## Slide claims checked against this data

| slide claim (sep25_2026) | here (Euclidean analog) |
|---|---|
| cond(Sigma) lower bound from the top word | **Holds, but loose.** A scale-invariant form, (d-1) L/(tr - L) with L = max_v pi_v \|e_v - mu\|^2/(1 - pi_v), gives 10^1.8 for every codebook, against a measured 10^2.9-10^6.7 (1.1 decades short for gauss R=0.1, 2.0-2.1 for gauss R ≥ 0.5, 4.8-4.9 for naive). The slides' literal form (no tr - L, unit norm) is weaker still. |
| posterior bound on cond(Sigma_t) | **Holds at every t and in every mode** (0 violations of the median). It is largest at the clean end (median 10^2.7-10^3.6 at t = 0, vs 10^1.8 at t = 1) but sits 4-9 decades below cond there. It captures the lambda_1 side, not the tail-set lambda_min. |
| long-tailed q_data → ill-conditioned Sigma | **Confirmed.** The uniform-prior control removes 2.3-4.0 decades. The mechanism is mainly the placement of the frequent words, not the single top word (point 4). |
| "prior frequencies enter only near the origin" | **Not confirmed.** At t = 0, `uni` vs the uniform-prior control: cond 10^6.9-10^11.5 vs 10^5.7-10^5.9, k99 6-40 vs 98-164. This is partly because the two modes sample different words (v ~ pi vs v ~ uniform). |
| "far from the origin the posterior collapses, F → 0" | **Partly.** In `ctx` (perplexity 1.0 at t = 0), tr F_t falls 1.4-2.9 decades from its peak (t ≈ 0.26-0.44) to t = 0. It stays finite because of the truncation. The skew does not vanish: cond and the bound are both largest at t = 0. In `uni` tr F barely falls, except in the collapsed gauss R=0.1. |
| global whitening A* = c Sigma^{-1/2} | **Not tested** (cond(Sigma_data^{-1/2} Sigma_bar_t Sigma_data^{-1/2}) was not computed). See §3 for what the data do and do not support. |

The Euclidean F_t covers all d directions; it is the exact Fisher of EFLM's Gaussian forward process, not the
slides' angular Fisher. The two coincide only near the origin (t = 1 here), where the slides' F ∝ P_w Sigma P_w with
the same Sigma = Sigma_data; there, by Cauchy interlacing, the cond here is an upper bound on the tangent-space cond.
At other t the slides' Sigma_t uses a Poisson-kernel likelihood and 1/D_kappa-weighted tangent vectors, so no bound
follows.

## 1. Noise end: Sigma_data = Cov_pi[e_v] and the frequency-vs-geometry controls

| run | log10 cond | lambda_1/(tr/d) | PR | k90 | k99 | uniform prior | permuted pi | isotropic codebook, real pi |
|---|---|---|---|---|---|---|---|---|
| naive R=0.1 | **6.63** | 160 | 6.7 | 7 | 7 | 2.84 | 4.16 | 2.90 |
| naive R=0.5 | **6.64** | 110 | 10.4 | 10 | 11 | 2.71 | 4.09 | 2.90 |
| naive R=1.0 | **6.68** | 100 | 12.1 | 12 | 13 | 2.67 | 4.10 | 2.90 |
| gauss R=0.1 (collapsed) | 2.91 | 60 | 64.2 | 362 | 688 | 0.30 | 2.91 | 2.90 |
| gauss R=0.5 | 3.92 | 130 | 10.3 | 10 | 465 | 1.57 | 3.01 | 2.90 |
| gauss R=1.0 | 3.81 | 111 | 12.1 | 12 | 487 | 1.54 | 3.00 | 2.90 |

(The last three columns are log10 cond.)

- **The head is about 10-dimensional, not one direction.** For R ≥ 0.5 both readouts have lambda_1 at 100-130x the
  mean, but only 1.3-1.5x lambda_2 and 13-17% of the trace. The top 10 directions hold 84-94%.
- **After the head the readouts differ.** In naive almost nothing remains: lambda_1/lambda_k99 is only 10^0.35-0.42, so
  the remaining ~755 eigenvalues are tiny. Gaussian spreads its last 7-15% over a bulk of about 470 directions.
- **The collapsed gauss R=0.1 codebook matches a random isotropic codebook** with the same pi: cond 2.91 vs 2.90,
  lambda_1/mean 60 vs 53, PR 64 vs 69, k99 688 vs 682. With a uniform prior its cond is 10^0.30, close to
  the Marchenko-Pastur value for a random codebook (about 10^0.28).
- **The uniform prior is not a pure reweighting.** It moves the mass from about 387 effective words (exp H(pi)) to the
  29,250 seen words, most of them rare.

**Which direction.** u1 is the top eigenvector of Sigma_data. The null comes from 32 random permutations of pi on the
same codebook (median [quartiles]).

| run | \|cos(u1, e_'.' - mu_pi)\| | null | lambda_1 share of the top 1 / 10 / 100 words | \|Spearman(log freq, <e_v, u1>)\|, seen words |
|---|---|---|---|---|
| naive R=0.1 | 0.946 | 0.72 [0.30, 0.88] | 0.31 / 0.50 / 0.69 | 0.17 |
| naive R=0.5 | 0.935 | 0.76 [0.53, 0.91] | 0.44 / 0.59 / 0.79 | 0.03 |
| naive R=1.0 | 0.915 | 0.77 [0.43, 0.92] | 0.46 / 0.64 / 0.84 | 0.08 |
| gauss R=0.1 (collapsed) | 0.995 | 0.99 [0.99, 1.00] | 0.88 / 0.98 / 0.99 | 0.04 |
| gauss R=0.5 | 0.908 | 0.99 [0.98, 0.99] | 0.36 / 0.54 / 0.77 | 0.37 |
| gauss R=1.0 | 0.885 | 0.99 [0.98, 0.99] | 0.39 / 0.59 / 0.80 | 0.37 |

- **Naive:** the alignment with '.' sits near the null's upper quartile (above it for R = 0.1 / 0.5, just below it for
  R = 1.0). There is no evidence that '.' is more aligned with u1 than any word given the same mass would be. Across
  the vocabulary, the projection on u1 is essentially unrelated to frequency (|Spearman| 0.03-0.17).
- **Gaussian R = 0.5 / 1.0:**
  - '.' is *less* aligned with u1 than the null, because u1 is shared by several frequent words.
  - Their actual placement concentrates the energy far beyond what the spectrum alone gives: k90 10-12 vs 297-301,
    and cond 10^3.8-3.9 vs 10^3.0 under permuted pi.
  - There is a moderate global frequency trend (|Spearman| 0.37 over 29,250 words).
- **Only the collapsed R=0.1** is "the frequency spectrum acting on a structureless codebook": actual, permuted and
  isotropic cond are all 2.90-2.91.

## 2. Per diffusion time: per-state Sigma_t, median over 128 states

Exact unigram-prior posterior (`uni`, the slides' object). `≥12` = at least half of the states are at the fp64 floor.
Full 51-point curves are in `fisher_cond_state_t.png`.

| run | log10 cond, t = 0 / 0.1 / 0.2 / 0.3 / 0.5 / 1 | k99, same t | PR, same t |
|---|---|---|---|
| naive R=0.1 | 11.5 / 8.3 / 7.3 / 6.9 / 6.6 / 6.6 | 6 / 7 / 7 / 7 / 7 / 7 | 3.0 / 4.1 / 5.1 / 5.9 / 6.5 / 6.7 |
| naive R=0.5 | 11.4 / 9.1 / 8.0 / 7.3 / 6.8 / 6.6 | 9 / 10 / 11 / 11 / 11 / 11 | 2.5 / 4.1 / 5.0 / 6.6 / 9.1 / 10.4 |
| naive R=1.0 | 11.4 / 9.6 / 8.3 / 7.6 / 6.9 / 6.7 | 9 / 12 / 12 / 13 / 13 / 13 | 2.4 / 3.6 / 4.1 / 6.3 / 9.6 / 12.1 |
| gauss R=0.1 (collapsed) | 6.9 / 5.2 / 4.1 / 3.4 / 3.0 / 2.9 | 40 / 270 / 531 / 636 / 681 / 688 | 2.1 / 3.5 / 12.3 / 31.2 / 55.0 / 64.2 |
| gauss R=0.5 | 9.8 / 6.9 / 5.5 / 4.6 / 4.1 / 3.9 | 10 / 28 / 106 / 278 / 438 / 465 | 2.3 / 3.5 / 4.3 / 6.5 / 8.9 / 10.3 |
| gauss R=1.0 | 9.7 / 6.8 / 5.7 / 4.9 / 4.2 / 3.8 | 9 / 35 / 90 / 192 / 422 / 487 | 2.2 / 4.2 / 4.9 / 6.3 / 9.3 / 12.1 |

The model's own posterior p_theta on validation text (`ctx`):

| run | log10 cond, t = 0 / 0.1 / 0.2 / 0.3 / 0.5 / 1 | k99, same t | PR, same t |
|---|---|---|---|
| naive R=0.1 | 11.4 / 9.7 / 8.2 / 7.4 / 6.7 / 7.0 | 5 / 6 / 7 / 7 / 7 / 7 | 1.7 / 2.2 / 2.7 / 3.5 / 6.3 / 6.7 |
| naive R=0.5 | ≥12 / 10.9 / 9.9 / 8.7 / 7.4 / 6.6 | 5 / 7 / 8 / 10 / 11 / 11 | 1.6 / 2.0 / 2.2 / 2.7 / 6.5 / 10.4 |
| naive R=1.0 | ≥12 / 11.3 / 10.2 / 9.4 / 7.8 / 6.7 | 5 / 6 / 9 / 10 / 13 / 13 | 1.5 / 1.7 / 1.8 / 2.2 / 5.1 / 12.0 |
| gauss R=0.1 (collapsed) | 7.1 / 5.4 / 4.3 / 3.9 / 3.7 / 3.0 | 35 / 220 / 485 / 591 / 644 / 679 | 2.2 / 3.6 / 9.7 / 11.0 / 10.4 / 50.7 |
| gauss R=0.5 | ≥12 / 9.4 / 7.6 / 6.5 / 4.6 / 3.9 | 5 / 9 / 13 / 22 / 288 / 475 | 1.6 / 1.9 / 2.3 / 2.7 / 6.8 / 10.4 |
| gauss R=1.0 | ≥12 / 9.8 / 8.2 / 6.9 / 5.2 / 3.8 | 4 / 7 / 10 / 15 / 152 / 500 | 1.3 / 1.9 / 2.1 / 2.2 / 4.6 / 12.3 |

- **Context sharpens the posterior.** For R = 0.5 / 1.0, `ctx` stays at PR 2-3 until t ≈ 0.3 (naive
  R=0.1: 3.5 at t = 0.3), while `uni` is at about 6.
  - Gaussian R = 0.5 / 1.0: `ctx` k99 is 15-22 at t = 0.3 and passes 100 only from t = 0.38 / 0.44. Under `uni` the
    same runs are already at 192-278.
  - With context, the KL stays on a few dozen directions up to t ≈ 0.35-0.4 in both readouts.
  - Under the slides' unigram posterior, that holds only for naive (k99 ≤ 13 at every t). Gaussian `uni` reaches
    k99 90-278 by t = 0.2-0.3, although its PR stays ≤ 12.
- **The collapsed gauss R=0.1 is degenerate in `ctx`.** Accuracy is ≤ 0.02 for t ≥ 0.4, and posterior perplexity is
  46-73 for t in [0.4, 0.72], rising to 313 at t = 1 (other runs: 384-429).
- **The per-state bound (slide table) holds at every t.** At t = 0 its median is 10^2.7-10^3.2 (`uni`) and
  10^3.1-10^3.6 (`ctx`).
- **The uniform-prior control over t** (`flat`: uniform prior over the 29,250 seen words, and states drawn with
  v ~ uniform):
  - For t ≥ 0.3 the flat posterior is already close to its prior (perplexity 1,100-29,250), so these cells mostly
    repeat the Sigma_data control above. cond falls to 10^0.3-10^3.4 and k99 rises to 620-758.
  - The naive KL still keeps a dominant head (lambda_1 58-119x the mean, PR 16-27), against 20-47x and
    PR 97-273 for gaussian R ≥ 0.5.
- **Compare across R at matched SNR** (`fisher_cond_state_snr.png`; its x-axis runs from the clean end on the left to
  noise on the right).
  - tau*(R) equalises the SNR only at t = 0. At t = 0.5, R = 0.1 has 3.6-5.7x lower SNR than R = 0.5 / 1.
  - The three naive runs fall nearly on one curve against SNR, so their differences at the same t are an SNR effect.
  - Gauss R=0.1 stays less skewed than gauss R = 0.5 / 1 even at matched SNR (and at t = 0, where the SNR is equal),
    because its collapsed codebook is near-random.

## 3. State-averaged Sigma_bar_t = E[Sigma_t] (the relevant object for a global reparametrisation such as A*)

Only `uni` is shown; `ctx` Sigma_bar is not converged at small t (see the last bullet).

| run | log10 cond(Sigma_bar_t), t = 0 / 0.2 / 0.4 / 1 | PR, all t | energy along e_'.' - mu_pi (rq), t = 0 / 1 |
|---|---|---|---|
| naive R=0.1 | 6.06 / 6.23 / 6.54 / 6.63 | 6.7-7.0 | 107 / 153 |
| naive R=0.5 | 6.02 / 6.12 / 6.46 / 6.64 | 10.4-11.0 | 57 / 104 |
| naive R=1.0 | 6.04 / 6.09 / 6.41 / 6.68 | 12.1-12.9 | 44 / 94 |
| gauss R=0.1 (collapsed) | 1.88 / 2.58 / 2.87 / 2.91 | 64-368 | 9 / 60 |
| gauss R=0.5 | 3.22 / 3.33 / 3.72 / 3.92 | 10.3-16.0 | 53 / 117 |
| gauss R=1.0 | 3.19 / 3.20 / 3.54 / 3.81 | 12.1-18.6 | 36 / 99 |

rq = r^T Sigma_bar r / (tr / d); 1 is the average direction.

- **The averaged conditioning is nearly time-invariant.** It varies by 0.7-0.8 decades over t (1.0 for gauss R=0.1),
  against 4.0-5.9 decades for the per-state median cond.
- **The frequent-word directions stay stiff at every t.** Along e_'.' - mu_pi the KL is 36-153x stiffer than average
  (collapsed R=0.1: 9-60x). Under the uniform prior this falls to 0.8-53x.
- **The leading direction is not fixed.** The stiffness along pc1(Sigma_data) changes 1.5-2.8x from t = 1 to t = 0
  (about 7x for gauss R=0.1). Near t = 0 the top eigenvector of Sigma_bar_t is nearly orthogonal to pc1(Sigma_data):
  |cos| 0.04-0.22 for the non-collapsed runs, inside a near-degenerate top pair (lambda_1/lambda_2 ≤ 1.12).
- **What this means for A\*.**
  - A* = c Sigma_data^{-1/2} is exact only at t = 1. Whether it whitens Sigma_bar_t at t < 1 was not computed.
  - No global A can remove the per-state clean-end skew, because that skew lies along e_true - mu_t and changes from
    state to state.
  - For naive codebooks (k99 7-13), A* would have to amplify about 755 directions that hold under 1% of the energy,
    so it needs the slides' support-whitening or regularisation.
- **Convergence of Sigma_bar.** The full and half-sample estimates agree within 0.1 decade for `uni` and `flat`
  (except gauss R=0.1 at t ≤ 0.04, gap up to 0.22). For `ctx` they agree only for t ≥ 0.5 (except gauss R=0.1, which
  agrees at every t). Below that the gap is up to 0.7 decades (1.5 at one cell), with trace-ESS 440-6,900.

## Three timesteps: share of lambda_1, stable rank, k90 / k99, PR, cond (v3 rerun)

`report_timesteps.py` → `report_timesteps.md`, from `results_v3/`. Per-state medians over 128 states. These tables
use the **reverse** time convention: timestep 0 = pure noise (code t = 1), timestep 1 = clean end (code t = 0).
lambda_1^2 / sum lambda_i^2 = 1 / stable rank. `censored` = share of states with cond ≥ 10^12.

### timestep 0.0 (code t = 1.0)

| posterior | readout | R | λ₁/Σλᵢ | λ₁²/Σλᵢ² | k90 | k99 | PR | log10 cond | censored |
|---|---|---|---|---|---|---|---|---|---|
| uni | naive | 0.1 | 0.208 | 0.289 | 7 | 7 | 6.7 | 6.63 | 0.00 |
| uni | naive | 0.5 | 0.143 | 0.212 | 10 | 11 | 10.4 | 6.64 | 0.00 |
| uni | naive | 1.0 | 0.131 | 0.206 | 12 | 13 | 12.1 | 6.68 | 0.00 |
| uni | gauss | 0.1 | 0.078 | 0.395 | 362 | 688 | 64.2 | 2.91 | 0.00 |
| uni | gauss | 0.5 | 0.169 | 0.295 | 10 | 465 | 10.3 | 3.92 | 0.00 |
| uni | gauss | 1.0 | 0.145 | 0.253 | 12 | 487 | 12.1 | 3.81 | 0.00 |
| ctx | naive | 0.1 | 0.204 | 0.277 | 7 | 7 | 6.7 | 7.03 | 0.00 |
| ctx | naive | 0.5 | 0.144 | 0.215 | 10 | 11 | 10.4 | 6.61 | 0.00 |
| ctx | naive | 1.0 | 0.132 | 0.211 | 12 | 13 | 12.1 | 6.70 | 0.00 |
| ctx | gauss | 0.1 | 0.088 | 0.395 | 346 | 679 | 50.7 | 3.02 | 0.00 |
| ctx | gauss | 0.5 | 0.168 | 0.293 | 10 | 475 | 10.4 | 3.90 | 0.00 |
| ctx | gauss | 1.0 | 0.142 | 0.249 | 12 | 500 | 12.3 | 3.78 | 0.00 |

### timestep 0.5 (code t = 0.5)

| posterior | readout | R | λ₁/Σλᵢ | λ₁²/Σλᵢ² | k90 | k99 | PR | log10 cond | censored |
|---|---|---|---|---|---|---|---|---|---|
| uni | naive | 0.1 | 0.217 | 0.310 | 6 | 7 | 6.6 | 6.65 | 0.00 |
| uni | naive | 0.5 | 0.194 | 0.338 | 9 | 11 | 9.0 | 6.80 | 0.00 |
| uni | naive | 1.0 | 0.194 | 0.361 | 11 | 13 | 9.9 | 6.91 | 0.00 |
| uni | gauss | 0.1 | 0.085 | 0.403 | 343 | 681 | 55.0 | 2.99 | 0.00 |
| uni | gauss | 0.5 | 0.206 | 0.380 | 10 | 437.5 | 8.9 | 4.11 | 0.00 |
| uni | gauss | 1.0 | 0.210 | 0.432 | 11 | 438.5 | 9.3 | 4.13 | 0.00 |
| ctx | naive | 0.1 | 0.237 | 0.353 | 6 | 7 | 6.3 | 6.67 | 0.00 |
| ctx | naive | 0.5 | 0.288 | 0.540 | 9 | 11 | 6.7 | 7.38 | 0.00 |
| ctx | naive | 1.0 | 0.397 | 0.722 | 8 | 13 | 4.5 | 7.82 | 0.00 |
| ctx | gauss | 0.1 | 0.297 | 0.918 | 270 | 644.5 | 10.4 | 3.74 | 0.00 |
| ctx | gauss | 0.5 | 0.287 | 0.569 | 9 | 288 | 6.8 | 4.60 | 0.00 |
| ctx | gauss | 1.0 | 0.346 | 0.634 | 9 | 172.5 | 5.3 | 5.14 | 0.00 |

### timestep 1.0 (code t = 0.0)

| posterior | readout | R | λ₁/Σλᵢ | λ₁²/Σλᵢ² | k90 | k99 | PR | log10 cond | censored |
|---|---|---|---|---|---|---|---|---|---|
| uni | naive | 0.1 | 0.560 | 0.844 | 4 | 6 | 2.7 | 11.81 | 0.48 |
| uni | naive | 0.5 | 0.593 | 0.860 | 4 | 8 | 2.4 | 11.73 | 0.46 |
| uni | naive | 1.0 | 0.586 | 0.872 | 4 | 9 | 2.5 | 11.47 | 0.40 |
| uni | gauss | 0.1 | 0.682 | 0.952 | 5 | 39.5 | 2.1 | 6.91 | 0.00 |
| uni | gauss | 0.5 | 0.632 | 0.912 | 4 | 10 | 2.3 | 9.84 | 0.12 |
| uni | gauss | 1.0 | 0.639 | 0.913 | 3 | 9 | 2.2 | 9.76 | 0.11 |
| ctx | naive | 0.1 | 0.692 | 0.924 | 3 | 5 | 2.0 | 11.45 | 0.34 |
| ctx | naive | 0.5 | 0.782 | 0.970 | 2 | 6 | 1.6 | ≥12 | 0.57 |
| ctx | naive | 1.0 | 0.752 | 0.968 | 2 | 6 | 1.7 | ≥12 | 0.62 |
| ctx | gauss | 0.1 | 0.650 | 0.946 | 5 | 35 | 2.2 | 7.08 | 0.00 |
| ctx | gauss | 0.5 | 0.765 | 0.957 | 2 | 5 | 1.6 | 11.97 | 0.50 |
| ctx | gauss | 1.0 | 0.803 | 0.980 | 2 | 5 | 1.5 | ≥12 | 0.52 |

- **Head (lambda_1 share, k90, PR):** the same for both readouts at all three timesteps (R ≥ 0.5). At timesteps 0 and
  0.5 gaussian's lambda_1^2 / sum lambda_i^2 is slightly *higher* (more lambda_1-dominated), not lower, in 7 of the 8
  (R, posterior, timestep) cells, by 0.03-0.08; the exception is `ctx` R = 1.0 at timestep 0.5 (0.63 vs 0.72).
- **Tail (k99, cond):** gaussian differs at timesteps 0 and 0.5 (k99 172-500 vs 11-13, cond 10^3.8-10^5.1 vs
  10^6.6-10^7.8).
- **Clean end (timestep 1):** both readouts are nearly rank-1-2 (PR 1.5-2.5, lambda_1 holds 59-80% of the trace), and
  k99 is also the same (5-10 both). Only `uni` cond still differs (10^9.8 vs 10^11.5-10^11.7); in `ctx` about half
  the states are censored at 10^12 for both readouts.
- **Shannon erank (median, `uni` / `ctx`, R = 0.5, 1.0), timesteps 0 / 0.5 / 1:** naive 10.8 / 10.0 / 3.5 and
  12.6 / 11.4 / 3.8 (`uni`), 10.8 / 8.5 / 2.1 and 12.6 / 7.1 / 2.3 (`ctx`); gaussian 14.9 / 14.0 / 3.4 and
  17.7 / 16.1 / 3.2 (`uni`), 15.1 / 12.0 / 2.2 and 18.2 / 10.0 / 2.0 (`ctx`).
- **Precision:** 4 of the 6 v3 jobs ran on a different GPU type than v2 and drew different states (gauss R=0.1 / 0.5
  reproduced bit-for-bit; Sigma_data is deterministic and identical). Per-state medians moved by up to 0.7 in log10
  cond, 1-2 in k90, about 50 in k99 when k99 is in the hundreds, about 1 in PR when PR < 30 (up to 15% when it is
  in the hundreds, `flat`) and 0.1 in lambda_1/tr.
  Differences smaller than that are sampling noise.

## Angular-Fisher Sigma_t (setup.md): `visualization/angular_fisher_sigma_t.py`

Figures `figures/angular_fisher_sigma_t_t.png` (x = code t) and `figures/angular_fisher_sigma_t_snr.png` (x = log10
SNR_eff), styled as `figures/fisher_cond_state_{t,snr}.png`: median over 128 states and their 25-75% quartile band,
dashed = naive, solid = gaussian, one color per R, star = Sigma. Data `data/<label>.json`; design and the EFLM mapping in
`EXPERIMENT.md` ("Angular Fisher Sigma_t"). Sigma_t = Cov_q[P_w v] on the 767-dim tangent space at w = x_t / |x_t|,
v = e_v / R. EFLM's likelihood is a vMF kernel in w, i.e. the slides' Poisson kernel to first order in kappa rho_t, so
D = 1. `check_angular_fisher.py` confirms that |x_t|^2 beta_t^2 R^2 Sigma_t is the Fisher of the KL under rotations of
x_t (finite-difference KL / prediction within 3.3e-3 of 1 at eps = 1e-4, at t = 0.1 / 0.3 / 0.5 / 0.8).

Answers (R ≥ 0.5 unless stated; gauss R=0.1 is the collapsed model):

1. **Projecting out the radial direction changes little, except slightly at the clean end.** Comparison with the
   Euclidean `results_v3/` (independent draws of 128 states):
   - *Noise end:* `data_freq` and `uniform` agree to 3 digits (|Δ log10 cond| ≤ 0.002): w is nearly a random direction
     and the posterior equals the prior. `model_pred` differs by up to 0.08 in log10 cond, because this script keeps
     the model's full-vocabulary posterior while `ctx` in `fisher_cond.py` drops unseen words (mass ~5e-6).
   - *Clean end (t = 0):* the angular `data_freq` cond is lower in all 6 runs (by 0.17-0.70 decades for R ≥ 0.5,
     0.78 for naive R=0.1) and lambda_1 / sum is lower in all 6 (by up to 0.07), so this is a small systematic effect,
     plausibly because w (cos ≈ 0.19 with the true word) carries part of the true-word-vs-competitors axis there.
   - *In between:* differences alternate in sign across t (largest at t = 0.1-0.3, e.g. gauss R=1.0 `data_freq` PR
     5.3 vs 3.7 at t = 0.2), consistent with sampling noise. The conclusions of the Euclidean analysis carry over.
2. **Sigma: the frequency prior is what makes it ill-conditioned (confirms the slides).** Same codebook, q_data vs
   uniform: log10 cond 6.64-6.68 vs 2.67-2.71 for naive (+3.9-4.0 decades) and 3.81-3.92 vs 1.54-1.57 for gauss
   (+2.3-2.4); k90 10-12 vs 396-416 (naive). Under q_data the energy sits in about 10 directions for both readouts
   (PR 10.3-12.1).
3. **Readout: same head, different tail.**
   - *Head:* over all 51 t, lambda_1 / sum, k90 and PR differ between readouts by at most 0.06, 2 and 1.3 (`data_freq`)
     and 0.10, 2 and 0.85 (`model_pred`).
   - *Tail, t ≥ 0.5:* `data_freq` k99 is 422-489 (gauss) vs 11-13 (naive) and log10 cond 3.8-4.2 vs 6.6-7.0;
     `model_pred` k99 182-503 vs 11-13, cond 3.8-5.0 vs 6.5-7.6. Toward the clean end the tail gap shrinks.
   - *Uniform prior:* for t ≥ 0.5 the gauss codebook is far more isotropic, PR 216-272 vs 22-26, i.e. its rare words
     are spread, not clustered.
4. **Clean end: the posterior covariance has 1-3 dominant directions with the real prior or the model.** At code t = 0,
   PR is 2.3-3.0 with k90 4-5 and lambda_1 holding 52-63% of the trace (`data_freq`), and PR 1.4-1.8 with k90 2 and
   lambda_1 72-83% (`model_pred`). With the uniform prior it is less concentrated (PR 3.0 gauss, 8.0-8.8 naive; k99
   118-134). log10 cond is largest here: ≥ 9.4 (`data_freq`) and at the 12 ceiling (median censored) for `model_pred`,
   both readouts.
5. **Slide claim "prior frequencies enter only near the origin": not tested cleanly here.** At the clean end
   `data_freq` and `uniform` still differ (k99 9-11 vs 118-134, log10 cond 9.4-11.2 vs 5.7-5.9), but switching the
   prior also switches which words are drawn as the true word (frequent vs uniform), so the gap mixes both. A clean test
   would hold v* fixed and swap only the prior. What the data do show: far from the origin the skew does not vanish;
   cond is largest at the clean end.

**noise end (code t = 1, your timestep 0)**

| prior | readout | R | lambda_1 / sum | k90 | k99 | PR | log10 cond |
|---|---|---|---|---|---|---|---|
| data_freq | naive | 0.1 | 0.208 | 7 | 7 | 6.7 | 6.63 |
| data_freq | naive | 0.5 | 0.143 | 10 | 11 | 10.4 | 6.64 |
| data_freq | naive | 1.0 | 0.131 | 12 | 13 | 12.1 | 6.68 |
| data_freq | gauss | 0.1 | 0.078 | 362 | 687 | 64.2 | 2.91 |
| data_freq | gauss | 0.5 | 0.169 | 10 | 464 | 10.3 | 3.92 |
| data_freq | gauss | 1.0 | 0.145 | 12 | 487 | 12.1 | 3.81 |
| uniform | naive | 0.1 | 0.155 | 464 | 727 | 15.9 | 2.84 |
| uniform | naive | 0.5 | 0.090 | 416 | 717 | 22.5 | 2.71 |
| uniform | naive | 1.0 | 0.075 | 396 | 714 | 24.9 | 2.67 |
| uniform | gauss | 0.1 | 0.002 | 667 | 757 | 747.6 | 0.30 |
| uniform | gauss | 0.5 | 0.028 | 647 | 755 | 224.7 | 1.57 |
| uniform | gauss | 1.0 | 0.027 | 649 | 755 | 259.3 | 1.54 |
| model_pred | naive | 0.1 | 0.204 | 7 | 7 | 6.7 | 7.03 |
| model_pred | naive | 0.5 | 0.144 | 10 | 11 | 10.4 | 6.53 |
| model_pred | naive | 1.0 | 0.132 | 12 | 13 | 12.1 | 6.62 |
| model_pred | gauss | 0.1 | 0.084 | 345.5 | 678 | 51.8 | 3.00 |
| model_pred | gauss | 0.5 | 0.168 | 10 | 474 | 10.4 | 3.90 |
| model_pred | gauss | 1.0 | 0.142 | 12 | 500 | 12.3 | 3.78 |

**code t = 0.5**

| prior | readout | R | lambda_1 / sum | k90 | k99 | PR | log10 cond |
|---|---|---|---|---|---|---|---|
| data_freq | naive | 0.1 | 0.218 | 6 | 7 | 6.5 | 6.69 |
| data_freq | naive | 0.5 | 0.194 | 9 | 11 | 8.9 | 6.82 |
| data_freq | naive | 1.0 | 0.208 | 11 | 13 | 9.2 | 6.99 |
| data_freq | gauss | 0.1 | 0.090 | 341 | 680 | 52.7 | 3.02 |
| data_freq | gauss | 0.5 | 0.215 | 10 | 429 | 8.7 | 4.13 |
| data_freq | gauss | 1.0 | 0.218 | 11 | 421.5 | 9.1 | 4.19 |
| uniform | naive | 0.1 | 0.154 | 453 | 725 | 16.1 | 2.86 |
| uniform | naive | 0.5 | 0.091 | 381 | 706 | 23.3 | 2.83 |
| uniform | naive | 1.0 | 0.084 | 329 | 692 | 25.5 | 2.91 |
| uniform | gauss | 0.1 | 0.002 | 661 | 756 | 738.5 | 0.36 |
| uniform | gauss | 0.5 | 0.033 | 622.5 | 751 | 222.9 | 1.76 |
| uniform | gauss | 1.0 | 0.032 | 605 | 749 | 245.0 | 1.80 |
| model_pred | naive | 0.1 | 0.240 | 6 | 7 | 6.2 | 6.67 |
| model_pred | naive | 0.5 | 0.288 | 9 | 11 | 6.5 | 7.17 |
| model_pred | naive | 1.0 | 0.345 | 9 | 13 | 5.7 | 7.56 |
| model_pred | gauss | 0.1 | 0.293 | 268 | 643.5 | 10.6 | 3.73 |
| model_pred | gauss | 0.5 | 0.319 | 9 | 286 | 6.3 | 4.66 |
| model_pred | gauss | 1.0 | 0.355 | 10 | 192.5 | 5.3 | 5.01 |

**clean end (code t = 0, your timestep 1)**

| prior | readout | R | lambda_1 / sum | k90 | k99 | PR | log10 cond |
|---|---|---|---|---|---|---|---|
| data_freq | naive | 0.1 | 0.487 | 4 | 6 | 3.1 | 11.04 |
| data_freq | naive | 0.5 | 0.519 | 5 | 9 | 3.0 | 11.04 |
| data_freq | naive | 1.0 | 0.570 | 4 | 9 | 2.6 | 11.16 |
| data_freq | gauss | 0.1 | 0.637 | 5 | 49.5 | 2.3 | 6.81 |
| data_freq | gauss | 0.5 | 0.572 | 4 | 11 | 2.4 | 9.67 |
| data_freq | gauss | 1.0 | 0.630 | 4 | 11 | 2.3 | 9.43 |
| uniform | naive | 0.1 | 0.212 | 17.5 | 218 | 7.8 | 5.37 |
| uniform | naive | 0.5 | 0.256 | 13 | 117.5 | 8.0 | 5.73 |
| uniform | naive | 1.0 | 0.249 | 14 | 126 | 8.8 | 5.82 |
| uniform | gauss | 0.1 | 0.572 | 11 | 124.5 | 2.9 | 5.84 |
| uniform | gauss | 0.5 | 0.560 | 12.5 | 121.5 | 3.0 | 5.86 |
| uniform | gauss | 1.0 | 0.556 | 12 | 134 | 3.0 | 5.88 |
| model_pred | naive | 0.1 | 0.729 | 2 | 5 | 1.7 | 10.74 |
| model_pred | naive | 0.5 | 0.725 | 2 | 6 | 1.8 | ≥12 |
| model_pred | naive | 1.0 | 0.785 | 2 | 5 | 1.6 | ≥12 |
| model_pred | gauss | 0.1 | 0.633 | 5 | 42 | 2.3 | 6.93 |
| model_pred | gauss | 0.5 | 0.827 | 2 | 5 | 1.4 | ≥12 |
| model_pred | gauss | 1.0 | 0.828 | 2 | 4 | 1.4 | ≥12 |

**Sigma (all 768 dims, no projection)**

| q | readout | R | lambda_1 / sum | k90 | k99 | PR | log10 cond |
|---|---|---|---|---|---|---|---|
| q_data | naive | 0.1 | 0.208 | 7 | 7 | 6.7 | 6.63 |
| q_data | naive | 0.5 | 0.143 | 10 | 11 | 10.4 | 6.64 |
| q_data | naive | 1.0 | 0.131 | 12 | 13 | 12.1 | 6.68 |
| q_data | gauss | 0.1 | 0.078 | 362 | 688 | 64.2 | 2.91 |
| q_data | gauss | 0.5 | 0.169 | 10 | 465 | 10.3 | 3.92 |
| q_data | gauss | 1.0 | 0.145 | 12 | 487 | 12.1 | 3.81 |
| uniform | naive | 0.1 | 0.155 | 464 | 728 | 15.9 | 2.84 |
| uniform | naive | 0.5 | 0.090 | 416 | 718 | 22.5 | 2.71 |
| uniform | naive | 1.0 | 0.075 | 396 | 715 | 25.0 | 2.67 |
| uniform | gauss | 0.1 | 0.002 | 668 | 758 | 748.6 | 0.30 |
| uniform | gauss | 0.5 | 0.028 | 648 | 756 | 224.9 | 1.57 |
| uniform | gauss | 1.0 | 0.027 | 650 | 756 | 259.5 | 1.54 |

Caveats specific to this section: log10 cond is censored at 12 (eigvalsh of the formed Sigma_t cannot resolve
lambda_min < 1e-12 lambda_1; a median of 12.00 means at least half the states are censored). Medians over 128 states
move between independent draws by up to 0.42 in log10 cond and 34 in k99 (cancelled 256-state round vs these runs,
t ≤ 0.5), and by up to 0.64 / 58 between the v2 and v3 Euclidean reruns (different GPU types, different draws). `data_freq` gives
unused words prior 0, so they never enter (setup.md's "including word not used" has no effect). `model_pred` uses
p_theta(x^i | x_t) directly as the posterior (a plug-in; a second Bayes step would count x_t^i twice). Stable rank
enters only as lambda_1 / sum lambda_i = 1 / stable rank of A (Sigma_t = A^T A, i.e. tr / lambda_1), as setup.md's five
metrics list; the v3 section's srank(Sigma) = sum lambda^2 / lambda_1^2 is a different quantity and is not computed
here. The readout is confounded with the CE
weight and code version, one seed, as above. A first 256-state round was cancelled when setup.md fixed 128 states.

## Caveats

- **Readout is confounded.** The configs differ only in `forward_type` and `noise_sched_cov_weight`, but:
  - The gaussian runs down-weight the clean end by exp(-tau*) (0.019 / 0.087 / 0.16 for R = 0.1 / 0.5 / 1.0). This
    acts through both the codebook E and p_theta.
  - The naive runs were trained about two months earlier (Jul-Aug 2026), before these options existed, so the code
    version also differs.
  - A clean comparison needs naive + |alpha'| (or gaussian + plain CE) at lr 3e-4.
- **Limited runs.** One seed per config, EMA weights, lr 3e-4 only. gauss R=0.1 is collapsed.
- **When cond means tail depth.** cond is set by directions holding under 1% of the trace whenever log10 cond is much
  larger than log10 head cond (lambda_1/lambda_k99). That covers the clean end in every run, and every t for naive
  (head cond 10^0.3-10^1.9 vs cond 10^6.6-10^11.5). In those cells, read k90 / k99 / PR /
  lambda_1/(tr/d) instead.
- **Censoring.** cond ≥ 10^12 is reported as `≥12`. The q ~ e^-27 tail estimate comes from SNR_eff = 27.7 at t = 0; it
  was not measured.
- **What `ctx` is.** It replaces the slides' unigram prior with p_theta(x^i | x_t), as a plug-in for the true
  contextual posterior; its Fisher with respect to x_t^i has the same form. It is not the network's input sensitivity
  J^T (diag p - p p^T) J.
- **Superseded numbers.** v1 results (`results_v1/`) had no censoring, Sigma_bar from 128 states, and no controls.

## HBFM angular-Fisher Sigma_t: `visualization/angular_fisher_sigma_t_hbfm.py` (2026-10-20)

Setup in EXPERIMENT.md. Medians over 128 states; Sigma_t has 32 tangent dims, Sigma 48 dims.

| run | t | heat | data_freq top1 / k90 / k99 / PR / log10 cond | uniform | model_pred |
| --- | --- | --- | --- | --- | --- |
| naive_ada_ub-1 | 0 (clean) | 0.25 | 0.351 / 14 / 29 / 5.5 / 2.21 | 0.103 / 26 / 32 / 23.8 / 0.83 | 0.613 / 4 / 15 / 2.4 / 3.51 |
| naive_fix_ub-0.5 | 0 (clean) | 0.5 | 0.586 / 7 / 24 / 2.7 / 2.90 | 0.369 / 18 / 30 / 6.2 / 1.87 | 0.723 / 3 / 8 / 1.8 / 4.23 |
| horo_fix_ub-0.5 | 0 (clean) | 0.5 | 0.593 / 7 / 23 / 2.6 / 2.99 | 0.319 / 19 / 30 / 6.9 / 1.81 | 0.733 / 3 / 7 / 1.8 / 4.61 |
| naive_ada_ub-1 | 0.5 | 0.054 | 0.167 / 22 / 31 / 14.7 / 1.38 | 0.042 / 29 / 32 / 31.5 / 0.22 | 0.329 / 17 / 30 / 6.8 / 1.95 |
| naive_fix_ub-0.5 | 0.5 | 0.164 | 0.306 / 20 / 31 / 8.0 / 1.76 | 0.062 / 28 / 32 / 29.1 / 0.50 | 0.587 / 7 / 24 / 2.6 / 2.84 |
| horo_fix_ub-0.5 | 0.5 | 0.164 | 0.290 / 19 / 30 / 8.6 / 1.80 | 0.065 / 28 / 32 / 28.9 / 0.52 | 0.592 / 6 / 23 / 2.6 / 3.01 |
| naive_ada_ub-1 | 1 (origin) | 7.9e-5 | 0.104 / 25 / 31 / 21.8 / 1.04 | 0.040 / 29 / 32 / 31.7 / 0.18 | 0.105 / 25 / 31 / 21.8 / 1.04 |
| naive_fix_ub-0.5 | 1 (origin) | 2.6e-4 | 0.099 / 26 / 32 / 23.4 / 0.86 | 0.038 / 29 / 32 / 31.8 / 0.16 | 0.101 / 26 / 32 / 23.4 / 0.86 |
| horo_fix_ub-0.5 | 1 (origin) | 2.6e-4 | 0.100 / 25 / 32 / 22.8 / 0.94 | 0.039 / 29 / 32 / 31.8 / 0.17 | 0.101 / 25 / 32 / 22.7 / 0.96 |

Sigma = Cov_{q_data}[v] (48 dims), data_freq: log10 cond 1.29 (ada), 1.06 (fix naive), 1.18 (fix horo), PR 28-31;
uniform: log10 cond 0.22-0.24, PR 47.5.

1. **The HBFM codebook is well conditioned.** log10 cond(Sigma) is 1.1-1.3 under the frequency prior, against
   6.6-6.7 for the naive EFLM codebooks (768 dims vs 48, so not like-for-like); the frequency prior still costs about one decade over uniform
   (0.22-0.24). The worst per-state value, at the clean end with model_pred, is 10^4.2-10^4.6, never near the
   10^12 censoring that EFLM reaches.
2. **Horosphere readout = naive readout here.** horo_fix and naive_fix agree within the quartile band on every
   metric and prior, including model_pred (clean-end log10 cond 4.61 vs 4.23, PR 1.8 vs 1.8). Under the
   data_freq / uniform priors this is largely the shared seed-1 codebook (see caveats); only model_pred involves
   the readout.
3. **Adaptive vs fixed curvature is a clock change.** Plotted against effective heat time the three runs fall on one
   curve; on the t axis the adaptive run looks better conditioned only because it stops at heat time 0.25
   (median kappa rho 0.56 at its clean end vs 0.79).
4. **Skew grows toward the clean end**, as in EFLM: model_pred concentrates 61-73% of tr Sigma_t in one direction
   (PR 1.8-2.4) at t = 0, and data_freq 35-59%.

Caveats: one seed per config; Sigma_t's spectrum depends on the per-factor metric (raw angles would reweight factor m
by sinh^2(kappa rho_m)); model_pred is the same plug-in as for EFLM; adaptive + horosphere was not available.

**The three seed-1 codebooks are nearly the same matrix (checked 2026-10-21).** The checkpoints are the right,
distinct runs (forward_type naive / horosphere / naive + curvature_bound 4 in their configs; different files), but
all three start from the same seed-1 ngpt init and the 48-dim codebook moves little in 30k steps. EMA
`sphere_embed.weight`, mean per-row cosine: naive_fix vs horo_fix 0.959 (0.994 at step 5k), vs naive_ada 0.965,
horo_fix vs naive_ada 0.955; naive_fix seed 1 vs seed 2 (same config) 0.001, |A-B|/|A| = 1.41 = sqrt 2, i.e.
independent. Row norms stay ~1.02 (init 1.0), and the normalized-mean-shift singular values (30.4-35.6) sit near
the random-Gaussian Marchenko-Pastur band (31.4-33.4). So the codebook spectra in `figures/codebook_hbfm/` and the
data_freq / uniform Sigma_t (functions of the codebook and the geometry alone) are expected to look alike across
the three runs. They say little about forward_type or curvature; a seed-2/3 comparison would show the
seed-to-seed spread instead.

## GOAL2: the six models at embedding dim 16 / 48 / 768 x 3 seeds (2026-10-22)

**Status: complete, 54/54 cells.** Each cell has an eval (val PPL, GenPPL and entropy on 64 samples) and train /
valid NELBO. Setup and run commands are in EXPERIMENT.md (GOAL2). Regenerate the table with
`python experiments/eflm_cov_reparam_fisher_cond/report.py`.

Columns:
- *val PPL:* exp of the validation training objective. naive uses CE averaged over t; gauss uses |alpha_t'| x CE.
  This objective is not a likelihood bound and depends on R and on the arm, so compare val PPL only within one
  arm x R.
- *NELBO:* the codebase's EFLM ELBO in nats per token (`algo.loss_type=elbo`, EMA weights; `nelbo.py`). It has
  three terms: the VDM diffusion term, the reconstruction CE at t = sampling_eps and the prior KL at t = 1. It is
  measured on 2048 sequences of the shuffled train loader or the valid loader, with one (t, noise) draw per
  sequence. NELBO is comparable across all cells.
  - Every cell gets the same sequences and the same draws, so differences between cells are paired. The
    per-sequence standard error of one cell is ~0.03 (0.026-0.17 over all 108 values), but the seed std of one
    config is 0.003-0.03.

Values are mean ± std over seeds 1-3.

| arm | R | dim | val PPL | GenPPL | entropy | train NELBO | valid NELBO |
|---|---|---|---|---|---|---|---|
| naive | 0.1 | 16 | 55.4 ± 0.2 | 15.42 ± 0.53 | 3.86 ± 0.01 | 1.992 ± 0.008 | 2.004 ± 0.016 |
| naive | 0.1 | 48 | 55.2 ± 0.3 | 15.46 ± 0.41 | 3.86 ± 0.03 | 2.018 ± 0.029 | 2.036 ± 0.031 |
| naive | 0.1 | 768 | 57.3 ± 0.3 | 12.64 ± 0.59 | 3.64 ± 0.04 | 1.944 ± 0.005 | 1.963 ± 0.004 |
| naive | 0.5 | 16 | 25.9 ± 0.1 | 15.12 ± 0.78 | 3.86 ± 0.02 | 1.915 ± 0.007 | 1.922 ± 0.005 |
| naive | 0.5 | 48 | 25.9 ± 0.2 | 14.57 ± 0.35 | 3.88 ± 0.03 | 1.967 ± 0.016 | 1.984 ± 0.017 |
| naive | 0.5 | 768 | 26.5 ± 0.2 | 12.39 ± 0.64 | 3.79 ± 0.02 | 1.904 ± 0.005 | 1.919 ± 0.004 |
| naive | 1.0 | 16 | 18.0 ± 0.1 | 15.39 ± 0.18 | 3.87 ± 0.02 | 1.909 ± 0.003 | 1.923 ± 0.006 |
| naive | 1.0 | 48 | 17.9 ± 0.1 | 15.29 ± 0.33 | 3.90 ± 0.03 | 1.953 ± 0.009 | 1.965 ± 0.007 |
| naive | 1.0 | 768 | 18.0 ± 0.1 | 13.82 ± 1.09 | 3.87 ± 0.04 | 1.911 ± 0.018 | 1.928 ± 0.032 |
| gauss | 0.5 | 16 | 26.8 ± 0.1 | 14.39 ± 0.45 | 3.82 ± 0.03 | 1.980 ± 0.005 | 1.991 ± 0.004 |
| gauss | 0.5 | 48 | 27.9 ± 0.5 | 17.55 ± 2.53 | 3.74 ± 0.08 | 2.081 ± 0.027 | 2.097 ± 0.026 |
| gauss | 0.5 | 768 | 27.8 ± 0.1 | 15.48 ± 0.07 | 3.83 ± 0.02 | 2.096 ± 0.022 | 2.113 ± 0.018 |
| gauss | 1.0 | 16 | 18.4 ± 0.0 | 14.62 ± 0.56 | 3.83 ± 0.02 | 1.949 ± 0.020 | 1.966 ± 0.013 |
| gauss | 1.0 | 48 | 18.4 ± 0.0 | 14.41 ± 0.37 | 3.82 ± 0.02 | 1.978 ± 0.006 | 1.997 ± 0.004 |
| gauss | 1.0 | 768 | 19.0 ± 0.1 | 15.90 ± 0.58 | 3.87 ± 0.02 | 2.069 ± 0.012 | 2.081 ± 0.013 |

gauss R = 0.1 is listed per seed, because some of its seeds collapse:

| dim | seed | val PPL | GenPPL | entropy | train NELBO | valid NELBO | state |
|---|---|---|---|---|---|---|---|
| 16 | 1 | 58.8 | 18.61 | 3.66 | 2.136 | 2.147 | degraded |
| 16 | 2 | 65.9 | 26.01 | 3.41 | 2.408 | 2.435 | degraded |
| 16 | 3 | 130.7 | 52.39 | 1.34 | 5.968 | 5.954 | collapsed |
| 48 | 1 | 119.9 | 60.13 | 1.44 | 5.910 | 5.938 | collapsed |
| 48 | 2 | 119.9 | 55.11 | 1.41 | 5.927 | 5.928 | collapsed |
| 48 | 3 | 119.7 | 54.67 | 1.46 | 5.933 | 5.952 | collapsed |
| 768 | 1 (original) | 172.7 | 45447 | 0.38 | 6.554 | 6.602 | collapsed |
| 768 | 2 | 78.6 | 88.68 | 2.65 | 3.292 | 3.304 | degraded |
| 768 | 3 | 81.2 | 74.24 | 2.61 | 3.423 | 3.466 | degraded |

Rough scale reference: s-flm-dev2's `elbo_tune_tinystories_256` loss table uses its own ELBO code on the same
2048-sequence protocol:
- MDLM: 1.571 train / 1.589 valid.
- EFLM CE R 0.5 at lr 1e-3: 1.878 / 1.913.
- Here, naive R 0.5 dim 768 at lr 3e-4 gives 1.904 / 1.919.

Findings (valid NELBO; per run, train is 0.00-0.03 lower):

1. **Naive arm: dim 16 matches dim 768 on NELBO but not on GenPPL. Dim 48 is worse than both.**
   - *NELBO:* at R ≥ 0.5, dim 16 is within 0.01 of dim 768 (1.922 vs 1.919 at R 0.5, 1.923 vs 1.928 at R 1.0).
     Dim 48 is 0.04-0.07 worse than dim 768 and 0.04-0.06 worse than dim 16 (seed std ≤ 0.03). The dimension
     effect is not monotone.
   - *G3 seeds 2 / 3 only* (same site at every dim):
     - dim 48 − dim 16 = +0.06 at R 0.5 and +0.04 at R 1.0.
     - dim 768 − dim 16 = +0.001 at R 0.5 and +0.025 at R 1.0.
     - So dim 48 is the worst dim on G3 as well.
   - *R 0.1:* dim 768 is best (1.963 vs 2.004 at dim 16 and 2.036 at dim 48).
   - *GenPPL:* dim 768 samples are better in all three R (12.4-13.8 vs 14.6-15.5).
     - At R 0.1 and 0.5, part of that gap comes with lower entropy (3.64 / 3.79 vs 3.86-3.88).
     - At R 1.0 the entropy is equal (3.87), and dim 768 is still 1.5 GenPPL lower (13.82 vs 15.29-15.39).
       On G3 seeds only, this R 1.0 gap shrinks to ~0.9 (14.46 vs 15.30 / 15.46).
   - *val PPL:* dimension barely changes it (within 0.6 at R ≥ 0.5, within 2.1 at R 0.1). So the trained
     objective is matched while NELBO and GenPPL differ.
2. **The gauss arm has a worse NELBO than the naive arm, and the gap grows with dim.**
   - gauss − naive valid NELBO:

     | R | dim 16 | dim 48 | dim 768 |
     |---|---|---|---|
     | 0.5 | +0.069 | +0.113 | +0.194 |
     | 1.0 | +0.043 | +0.032 | +0.153 |

     On G3 seeds only, dim 768 gives +0.19 at R 0.5 and +0.13 at R 1.0.

   - GenPPL does not follow the same pattern. At dim 16, gauss samples have lower GenPPL than naive (14.39 vs
     15.12 at R 0.5, 14.62 vs 15.39 at R 1.0), with slightly lower entropy (3.82-3.83 vs 3.86-3.87). At dim 768,
     gauss is clearly worse (15.48 vs 12.39, 15.90 vs 13.82).
   - So GenPPL and NELBO rank the two arms differently at small dim.
   - The arms also differ in loss weight (|alpha'| x CE vs CE), so this is not a pure readout effect. See the
     readout caveat above.
3. **gauss R 0.1 is unstable at every dim; the collapse of the original was not a one-off.**
   - 5 of 9 runs collapse: all three dim-48 seeds, dim-16 seed 3 and the dim-768 original.
   - The other 4 are degraded: NELBO 2.15-3.47 vs 1.96-2.07 per run for naive R 0.1.
   - The collapsed dim-16 and dim-48 runs reach the same state. NELBO is 5.91-5.97 (train and valid) for every one
     of them, val PPL ~120 at dim 48, and entropy 1.3-1.5.
   - naive R 0.1 never collapses (worst seed NELBO 2.065).
   - gauss down-weights the clean end by exp(-tau*) = 0.019 at R 0.1, so the clean end gets almost no training
     signal. That is a plausible cause, but it was not tested.
4. **Seed spread is small for non-collapsed cells.**
   - NELBO seed std is 0.003-0.03.
   - GenPPL seed std is 0.07-1.1, except gauss R 0.5 dim 48 at 2.5, where seed 1 has GenPPL 20.4 and entropy 3.65.
   - So the gauss − naive gaps in finding 2 and the dim-48 penalty at R 0.5 are well outside seed noise. The
     dim-48 penalty at R 1.0 is 1-3 seed stds.
   - The 64-sample GenPPL resampling noise is ~0.3 (eval check, EXPERIMENT.md).
5. **No overfitting.** Per run, valid − train NELBO is −0.003 to +0.030 for every run with NELBO < 3; the two
   degraded dim-768 gauss R 0.1 runs give +0.012 and +0.043. MDLM in dev2 has +0.018.

### Angular-Fisher Sigma_t and the codebook spectrum per dim (seed 1)

Figures:
- `figures/angular_fisher_sigma_t_ed-{16,48,768}_{t,snr}.png`: one figure per dim, data in `data/ed-<dim>/`. The
  ed-768 data are the originals' `data/*.json`.
- `figures/codebook_eflm/ed-<dim>/`: the eigenvalue distribution (normalized, mean-shift-normalized and
  normalized-mean-shift, each also with a log-x version) and the eigen-projection scatter.

All are seed-1 checkpoints. The tangent space has d − 1 dims. Medians over 128 states.

Sigma, the noise-end codebook covariance under the frequency prior (log10 cond / PR, and PR as a fraction of d):

| | dim 16 | dim 48 | dim 768 |
|---|---|---|---|
| naive | 0.52-0.61 / 13.5-14.1 (0.84-0.88 d) | 2.05-3.77 / 18.3-22.2 (0.38-0.46 d) | 6.63-6.68 / 6.7-12.1 (≤ 0.016 d) |
| gauss (non-collapsed seed 1) | 0.69-0.94 (R 0.1: 2.06) | 2.30-2.35 | 3.81-3.92 |
| uniform prior, naive | 0.16-0.18 | 0.47-0.68 | 2.67-2.84 |

1. **The frequency-induced ill-conditioning of Sigma is mostly a large-dim effect.**
   - At dim 16 the frequency prior costs ~0.4 decades over uniform, and Sigma is nearly isotropic.
   - At dim 48 it costs 1.6-3.1 decades. At dim 768 it costs 3.8-4.0 decades.
   - The naive-vs-gauss tail gap at dim 768 (6.6 vs 3.9) is gone at dim 48 (naive 2.05-2.40 vs gauss 2.30-2.35 at
     R ≥ 0.5). At dim 16 it is slightly reversed (0.52-0.55 vs 0.69-0.94).
2. **The clean-end skew does not depend on dim; the tail depth does.**
   - At t = 0, the head of Sigma_t has the same shape at all three dims (non-collapsed runs):
     - `data_freq`: lambda_1 / tr 0.49-0.67, k90 3-5, PR 2.0-3.1.
     - `model_pred`: lambda_1 / tr 0.72-0.83, k90 2, PR 1.4-1.8.
   - `model_pred` log10 cond grows with dim: 3.8-4.6 at dim 16, 6.2-7.1 at dim 48, 10.7-≥12 at dim 768.
   - So 1-3 directions carry the clean-end KL at every dim, and the deep tail is a property of the larger
     embedding.
3. **Collapsed codebooks stay at their init.**
   - gauss R 0.1 dim 48 has Sigma PR 36.6 / 48 (`data_freq`) and 47.9 / 48 (uniform), with log10 cond 0.76.
   - Its normalized-mean-shift singular values (30.3-33.4) sit in the random-matrix band sqrt(V/d)(1 ± sqrt(d/V))
     = 31.4-33.4. The collapsed dim-768 original looks the same.
   - In contrast, naive R 0.5 dim 48 has singular values 28.4-43.4 and one dominant direction.
   - gauss R 0.1 dim 16 seed 1 is degraded but not collapsed. It has the most anisotropic dim-16 Sigma: log10 cond
     2.06, PR 8.8.
4. **At dims 16 / 48 the same-seed codebooks share their init, as in HBFM.**
   - Mean row cosine of the EMA `sphere_embed`:
     - naive vs gauss R 0.5: 0.962 at dim 16 and 0.877 at dim 48.
     - naive R 0.5 vs naive R 1.0: 0.995 at dim 16 and 0.980 at dim 48.
     - seed 1 vs seed 2 of one config: -0.003 at dim 16 and -0.001 at dim 48, i.e. independent.
     - The seed-2 pairs give the same picture (naive vs gauss 0.975 / 0.901).
   - So within one dim, the `data_freq` and uniform Sigma of different arms and R are a shared init plus a training
     change.
   - That change is still large where the table shows it, e.g. at dim 48 gauss R 0.5 Sigma has PR 8.0 vs 20.3 for
     naive.
   - The figures cover seed 1 only. They show the effect of arm and R, not the seed spread.

Caveats (GOAL2):
- **Training site and code version.** 35 cells trained on G3 (1 B300 x 128 x accumulation 4) and 13 on Unicorn
  (4 GPUs x 32 x accumulation 4). The global batch is the same.
  - The six dim-768 seed-1 cells are the original runs. The naive originals were trained with the older `s-flm`
    checkout.
  - On G3, rotary embeddings use flash-attn's torch path, which matches the triton kernel bit for bit on an Ada GPU.
  - Every cell was evaluated on Unicorn with the same code.
  - Site is confounded with seed. At dims 16 / 48, seed 1 ran on Unicorn and seeds 2 / 3 on G3; the one exception
    is naive R 0.1 dim 16 seed 2, which also ran on Unicorn.
    - Over the 10 non-collapsed dim-16/48 cells, seed 1 has the highest GenPPL of the three seeds in 7 and the
      highest valid NELBO in 5.
    - The gaps are small: NELBO +0.008 to +0.016 over the G3 mean (+0.042 for gauss R 0.5 dim 48), and GenPPL
      +0.3 to +1.0 (+4.3 there).
    - The same-seed codebooks share their init (finding 4 above), so this could be a seed-1 effect or a site
      effect; the two cannot be separated.
  - At dim 768, seed 1 is always the original and seeds 2 / 3 are always G3. Here the seed-1 original is the best
    seed: it has the lowest naive GenPPL in all three R (11.81-12.56 vs 12.27-14.50), and at R 1.0 also the lowest
    NELBO (1.892 vs 1.937 / 1.954).
  - Findings 1-2 give the G3-only (seeds 2 / 3) numbers next to the means. The dim-48 penalty and the gauss − naive
    gap hold on G3 alone.
- **G3 reruns.** Three G3 cells (naive R 0.5 / 1.0 and gauss R 0.5, dim 768 seed 3) failed to save a checkpoint
  ("No space left on device" on /share/thickstun) and were retrained from scratch.
- **NELBO code fix.** The merged `EFLM.reconstruction_loss` passed `context=None`, which crashes the Gaussian
  readout. It now passes `TrainingContext(alpha_t=alpha_t)`. naive values do not depend on this fix.
