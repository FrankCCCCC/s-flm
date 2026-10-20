# eflm_cov_reparam_fisher_cond — is the EFLM word-posterior KL skewed in certain directions?

Question (from `slides/sep25_2026/slides.md`, "Angular Fisher" / "cond(Sigma)" slides): at each diffusion time t,
how ill-conditioned is the Fisher of the word-posterior KL, for EFLM `noise=autonomous` (truncated at tau*(R)),
R in {0.1, 0.5, 1.0}, lr 3e-4, naive vs Gaussian (Bayes) readout? If cond is large, the KL cost of a latent
error is concentrated in a few directions; we also ask *which* directions, and whether word frequency causes it.

## Euclidean analog of the slides' Sigma_t

EFLM forward (`algo.EFLM.q_xt`, `invert_time_convention=false`): x_t = a_t e_v + b_t eps, a_t = alpha_t,
b_t = 1 - alpha_t, eps ~ N(0, I_d), ||e_v|| = R (`rho_min = rho_max = R`); autonomous clock
b_t = (1 - eps) exp(-tau*(1 - t)), so t = 0 is the truncated clean end and t = 1 pure noise.

For any posterior of the form q(v | x_t) ∝ prior(v) N(x_t; a_t e_v, b_t^2 I):

    grad_{x_t} log q(v | x_t) = beta_t (e_v - mu_t),   beta_t = a_t / b_t^2,   mu_t = E_q[e_v]
    F_t = E_q[score score^T] = beta_t^2 Sigma_t,       Sigma_t = Cov_{q(v | x_t)}[e_v]
    KL(q(. | x_t) || q(. | x_t + delta)) = delta^T F_t delta / 2 + o(|delta|^2)

so cond(F_t) = cond(Sigma_t). Flat space: no tangent projection, the bound uses d-1 (not d-2), and the
near-origin limit is t = 1 (beta -> 0), where Sigma_t -> Sigma_data = Cov_pi[e_v], the slides' Sigma.
The posterior depends on (R, a_t, b_t) only through SNR_eff = (a_t R / b_t)^2 and the normalized codebook;
tau*(R) makes SNR_eff(t=0) = 27.7 for every R, but at t = 0.5 it is 0.40 / 1.44 / 2.26 for R = 0.1 / 0.5 / 1.0.
So compare readouts at the same (R, t), and R's at matched SNR_eff (both axes are plotted).

Posteriors (modes):

| mode | posterior | states x_t |
|---|---|---|
| `uni` | exact, prior = train unigram frequencies pi (the slides' q_data) | v ~ pi, fresh eps |
| `flat` | control: exact, uniform prior over the seen words (same codebook, no frequency skew) | v ~ uniform(seen) |
| `ctx` | the model's p_theta(x^i \| x_t) restricted to seen words: a plug-in for the contextual posterior q(x^i=v \| x_t) ∝ q(v \| x_t^{-i}) N(x_t^i; a e_v, b^2 I), whose Fisher in x_t^i (the (i, i) block) has the same form | inner positions of 32 validation sequences |

`uni` is the slides' object (posterior with the q_data unigram prior). `ctx` is its contextual generalisation: a
plug-in for the true contextual posterior's Fisher, not the network's own input sensitivity J^T (diag p - p p^T) J. Codebook E = the EMA, radius-rescaled table (`_sc_embed_table`) of each checkpoint, so the
readouts differ in E for every mode, and additionally in p_theta for `ctx`.

## Metrics

Per (run, mode, t), 128 states, median and quartiles, all V words, float64, covariance centred at the posterior mode:

- `log10_cond` = log10(lambda_max / lambda_min), **censored at 12** (lambda_min < 1e-12 lambda_max is below the fp64
  eigensolver floor); `singular` = censored fraction (a median with `*` means >= half censored). Near t = 0 the
  posterior sits on a few words and lambda_min is set by tail mass (q_v ~ e^{-27}): cond then measures tail depth, so
- tail-insensitive measures lead: `k90`, `k99` = number of eigen-directions holding 90% / 99% of tr Sigma_t;
  `eff_rank` = (tr)^2 / tr(Sigma^2); `aniso` = lambda_1 / (tr / d) (1 isotropic, d rank-1); `top1`, `top10`;
  `log10_head_cond` = lambda_1 / lambda_k99.
- `ppl_q` = exp(posterior entropy); `acc`, `ce` of the posterior on the true word; `log10_bound` = (d-1) L / (tr - L),
  L = max_v q_v ||e_v - mu||^2 / (1 - q_v) (Euclidean version of the slides' bound; sanity: <= `log10_cond`).

State-averaged Sigma_bar_t = E[Sigma_t] (the Fisher of a state-independent error such as a global reparametrisation,
cf. the slides' A*; an isotropic error's expected KL depends only on tr): closed form E^T diag(qbar) E - mean_j mu_j
mu_j^T over 8192 states (uni / flat) or all 8128 inner positions (ctx), with a half-sample value (convergence) and the
trace effective sample size. Directions: `rq_r` = r^T Sigma_bar r / (tr / d) for r in {mu_pi, e_top - mu_pi,
pc1(Sigma_data)} (gap-independent; 1 = average direction), |cos(u1, r)| and the eigengap lambda_1 / lambda_2.

Sigma_data controls: uniform prior on the same codebook (geometry only), permuted pi (same frequency spectrum, words
shuffled), isotropic random codebook of norm R with pi (frequency only); direction nulls from 32 permuted priors (not the
isotropic 0.03, since every compared vector lives in the codebook's low-rank span); lambda_1 share of the top 1 / 10 /
100 frequent words, sum_{top k} pi_v <e_v - mu_pi, u1>^2 / lambda_1.

## Runs (lr 3e-4, autonomous, tau_max = tau*(R), 30k steps, seed 1)

| label | checkpoint | readout | CE weight |
|---|---|---|---|
| `gauss_r-{0.1,0.5,1.0}` | `outputs/eflm_rescale_auto_trunc_cov_reparam_tinystories_256/ns-auto_r-{R}_lr-3e-4` | gaussian | \|alpha'\| (cov) |
| `naive_r-{0.1,0.5,1}` | `outputs/eflm_rescale_auto_trunc_tinystories_256/eflmratr_lr-3e-4_r-{R}_m-1.0` | naive | plain (dt) |

Caveats: the readout is confounded with the CE weight (the configs differ only in `forward_type` and
`noise_sched_cov_weight`; |alpha'| down-weights the clean end by exp(-tau*) = 0.019 / 0.087 / 0.16 for R = 0.1 / 0.5
/ 1.0 relative to the noise end), which acts through both E and p_theta. One seed per config. The unigram prior
counts the wrapped blocks' BOS/EOS tokens.

v1 (`results_v1/`, superseded): per-state cond uncensored (negative lambda_min -> sentinel ~299), Sigma_bar from only
128 states, no frequency controls. Review: workflow `wf_3dd34990-fe8` (derivation re-checked from the code; bound proven;
defects adjudicated).

## Run

    sbatch -J fc2_<label> -o logs/v2_<label>_%j.log run.sbatch <ckpt> <label>    # 1 GPU, ~35 min Ada, ~70 min A5000
    python experiments/eflm_cov_reparam_fisher_cond/report.py                   # -> tables + figures for RESULTS.md

## Angular Fisher Sigma_t (setup.md, 2026-10-19)

Question (`setup.md`): cond and stable rank of the slides' angular-Fisher covariance Sigma_t = Cov_{q(v|z_t)}[v~_t],
v~_t = P_w v / D_kappa(v, rho_t, w), along t, and of Sigma = Cov_{q_data}[v], for the same 6 checkpoints.

**Mapping to EFLM.** Every EFLM codeword has norm R, so at fixed |x_t| the Gaussian likelihood in w = x_t / |x_t| is a
vMF kernel exp(kappa_eff w^T v), kappa_eff = a_t |x_t| R / b_t^2, v = e_v / R in S^{d-1}. That is the slides' Poisson
kernel D^{-(d-1)} to first order in kappa rho_t ((d-1) sinh(kappa rho_t) <-> kappa_eff, D <-> 1; not kappa = 0, which
would make the posterior equal the prior at every t). So v~_t = P_w v: the slides' 1/D reweighting has no EFLM
counterpart (the matched kappa rho_t = asinh(kappa_eff / (d - 1)) is at most about 0.19, at t = 0; there
D = cosh - sinh w^T v would lie in [0.83, 1.21] over |w^T v| ≤ 1, and within about ±4% for the |w^T v| ≲ 0.2 that
occur, since |x_t| is mostly noise). This is EFLM's exact angular Fisher: rotating x_t at fixed norm,
grad_w log q(v | x_t) = |x_t| beta_t P_w (e_v - mu_t), so F_ang = |x_t|^2 beta_t^2 R^2 Cov_q[P_w v] (the slides'
(d-1)^2 sinh^2 prefactor is replaced by |x_t|^2 beta_t^2 R^2; all metrics are scale-free). Spectra are taken on the
tangent space w^perp (d - 1 = 767 dims). Unlike the Euclidean Sigma_t above, the direction w (mostly noise: cos(x_t,
e_true) ≈ 0.19 even at SNR_eff = 27.7) is projected out.

**Posteriors** q(v | z_t) ∝ N(x_t; a_t e_v, b_t^2 I) q(v), true word v* ~ q(v):

| label | q(v) | v* drawn | answers |
|---|---|---|---|
| `data_freq` | train unigram frequency over the whole embedding matrix (unused words get 0) | by frequency | the slides' q_data |
| `uniform` | uniform over the 29250 words seen in training | uniformly | control: codebook geometry only |
| `model_pred` | p_theta(x^i \| x_t) on validation text (context), full vocabulary, bf16 forward | data | plug-in for the contextual posterior q(v \| x_t^{-i}) N(x_t^i; a e_v, b^2 I); no second Bayes step (that would count x_t^i twice) |

**Metrics** per state (lambda_i descending, on the tangent space): lambda_1 / sum lambda_i (= 1 / stable rank of
A = diag(sqrt q)(V - mu), Sigma_t = A^T A), k90, k99, PR = (sum lambda)^2 / sum lambda^2, log10 cond (censored at 12).
Median and 25-75% quartile band over N_STATE = 128 states per (prior, t), at `--nt` (default 51) t values
np.linspace(0, 1, nt). Sigma (all d dims) is the near-origin limit, drawn as a star at the noise end (Cov_uniform[v]
for the `uniform` row).

**Correctness check** (`check_angular_fisher.py`): finite-difference KL under rotations of x_t vs eps^2/2 u^T F u, and the
Householder tangent eigenvalues vs eig(P_w Sigma P_w). Passed: KL/prediction within 3e-3 of 1 at eps = 1e-4 (error
falls 10x per 10x eps), eigenvalues to 1e-15.

**Run.** Script `visualization/angular_fisher_sigma_t.py`:
`sbatch --partition=thickstun -J af128_<label> -o logs/af128_<label>_%j.log run_angular.sbatch
visualization/angular_fisher_sigma_t.py --ckpt <ckpt> --label <label>` (1 GPU each, 6 jobs, ~1.5 min per t on an
RTX 6000 Ada with 256 states, so ~40 min at 128). Output: `data/<label>.json`; figures, styled as
`figures/fisher_cond_state_{t,snr}.png`: `figures/angular_fisher_sigma_t_t.png` (x = code t, 0 = clean end at tau*,
1 = noise) and `figures/angular_fisher_sigma_t_snr.png` (x = log10 SNR_eff, inverted so the clean end is on the left).
`--plot` redraws both from `data/*.json`. A first round at 256 states (jobs 420302-420307) was cancelled at about
t = 0.5 when setup.md fixed 128.
