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

## HBFM angular Fisher Sigma_t and codebook spectra (2026-10-20)

Same metrics, priors, 128 states and `--nt 51` grid as the EFLM run above, for seed 1 of the two HBFM configs in
`s-flm-dev2/experiments/elbo_tune_tinystories_256/RESULTS.md` ("HBFM CE-trained, (H^3)^16, adaptive curv., ub 1"
and "... fixed curv., ub 0.5"; all CE-trained, 30k steps, embed 48 = 16 factors of H^3 at K = -0.5, trunc_exp rate 3):

| label | checkpoint (under outputs/) |
| --- | --- |
| naive_ada_ub-1 | `hbfm_rlf_ada_curv_tinystories_256/cb-4_mode-texp_rate-3_ub-1_tm-1_lr-3e-4_seed-1` |
| naive_fix_ub-0.5 | `claude_test_hbfm_tinystories_256/mode-trunc_exp_rate-3_k-0.5_ub-0.5_seed-1` |
| horo_fix_ub-0.5 | `claude_test_hbfm_tinystories_256/mode-trunc_exp_rate-3_k-0.5_ub-0.5_fwd-horo_seed-1` |

Both named configs are `forward_type=naive`. The horosphere twin exists only for the fixed config; adaptive + horo
(CE, embed 48) was never trained and is left out.

- **Posterior.** HBFM's bridge direction law is the Poisson kernel, so q(v | z_t) ∝ q(v) prod_m D_m(v)^{-(d-1)}
  exactly, D_m = cosh s_m - sinh s_m w_m^T u_{v,m}, s_m = kappa rho_m (numeric/horosphere.py `compute_horosphere`).
- **Sigma_t** = Cov_q[(P_{w_m} u_{v,m} / D_m)_m] on the 32-dim tangent space of (S^2)^16. In the manifold's metric
  this is the angular Fisher divided by (d-1)^2 kappa^2; per raw angle dw_m it would weight factor m by sinh^2(s_m).
  Finite-difference KL under per-factor rotations matches it: worst |KL/pred - 1| = 2.7e-4 (fixed) and < 1e-2 (adaptive)
  at eps = 1e-4, except one adaptive state with a point-mass posterior whose predicted KL (1.6e-15) is at fp64 round-off.
- **Adaptive curvature** rescales position i's clock to lambda_i tau with the base-K readout. The model reads
  lambda = 0.2500001 at every position (the lower bound 1 / curvature_bound), so it is the fixed-K bridge on a 4x
  slower clock (effective heat time <= 0.25 vs 0.5).
- **Sigma** = Cov_q[v] over all 48 dims (stars in the figure); Sigma_t lives in 32 tangent dims.
- **Code.** dev1's HBFM has no `algo.curvature_bound`, so models load from `s-flm-dev2` via `SFLM_REPO`
  (`visualization/loss_geometry.py`), which also gained the `hbfm` algo entry and a `curvature_bound = 1` default.

Run (compute node, 1 GPU, ~5 min per checkpoint):

    SFLM_REPO=/share/desa/nfs02/sc3379/workspace/research/s-flm-dev2/s-flm TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
      python visualization/angular_fisher_sigma_t_hbfm.py --ckpt <ckpt> --label <label>

Output: `data/hbfm/<label>.json`, `figures/angular_fisher_sigma_t_hbfm_t.png` (x = code t) and
`figures/angular_fisher_sigma_t_hbfm_heat.png` (x = effective heat time lambda tau(t), log, inverted).
Codebook spectra: `visualization/codebook_eigen_dist.py --variants normalized,normalized-mean-shift,mean-shift-normalized`
and `codebook_eigen_proj_scatter.py --mode <each>`, same SFLM_REPO, into `figures/codebook_hbfm/`.

## GOAL2: the six angular-Fisher checkpoints at embedding dim 16 / 48 / 768 x 3 seeds (2026-10-21)

Question: do the six EFLM models above (naive / gauss x R in {0.1, 0.5, 1.0}) keep their GenPPL, entropy and
Sigma_t shape when the embedding (state) dimension shrinks from 768 to the HBFM sizes 48 and 16, and how large is
seed-to-seed variance?

- **Grid:** arm {naive, gauss} x R {0.1, 0.5, 1.0} x `model.embed_dim` {16, 48, 768} x seed {1, 2, 3} = 54 cells.
  The six (768, seed 1) cells are the original runs, symlinked into `outputs/eflm_cov_reparam_fisher_cond/`; 48 are
  trained. Run name `fwd-{arm}_r-{R}_ed-{dim}_seed-{seed}`, job name `eflm_cov_reparam_fisher_cond_<run>`.
- **Same config as each original run:** small-sphere-dit, ngpt init, lr 3e-4, seq 256, 30k steps, global batch 512,
  autonomous clock with TAU_MAX = tau*(R) (3.9811 / 2.4436 / 1.8338; tau* does not depend on the dimension).
  naive: FORWARD_TYPE=naive, COV_WEIGHT=false. gauss: FORWARD_TYPE=gaussian, COV_WEIGHT=true (CE x |alpha_t'|).
  So naive vs gauss also differs in the loss weight, as in the originals.
- **Code change:** `SphereDiT` reads `model.embed_dim` (null = hidden_size, the old behaviour; checkpoints without the
  key load unchanged). A smaller embed_dim keeps the 768-wide trunk and lifts the state with a linear `in_proj`
  (as in HyperbolicDiT); the readout is the same V-way head. `EMBED_DIM` knob in
  `scripts/{train,sample}/tinystories/eflm_rescale_auto_truncation.sh`. Smoke test (dim 16, 20 steps, train + both
  evals) passed: `sphere_embed` (50258, 16), `in_proj` 16 -> 768.
- **Caveat:** the naive originals were trained from the older `s-flm` checkout, the new cells from s-flm-dev1.
- **GPUs / wall clock:** an original run took ~10.3 h on 4x A6000 (+ ~4 min eval): ~41 GPU-h per cell, ~2,000 GPU-h
  for 48. `sweep.py --cluster unicorn` keeps the original layout (4 GPUs x 32 x accumulation 4, thickstun,desa);
  `--cluster g3` uses 1 B300 x 128 x accumulation 4 (same global batch, qos g3, checkpoint every 1k steps).
  G3 is not usable as is: a probe job failed at once without writing its log, i.e. `/share/desa/nfs02` (repo,
  outputs) is not mounted there. All 48 cells were submitted to Unicorn (jobs 448768-448799, 449152-449185).
  Observed: ~4 h per cell on thickstun (RTX 6000 Ada), ~9.5 h on kuleshov-compute-03 (A5000).
- **G3 (2026-10-21, to finish sooner):** the 35 seed-2/3 cells not yet started on Unicorn were cancelled there
  (with approval) and resubmitted to G3, jobs 74511-74545 (`--cluster g3`, qos g3, 1 B300 x 128 x accumulation 4,
  ~5.7 it/s = ~6 h per cell). G3 runs from the checkout `/share/thickstun/sychou/workspace_g3/research/s-flm-dev1/s-flm`
  (G3 has no /share/desa) with the same code (the changed files copied over) and the bs256 TinyStories cache
  copied into its `data_cache/`; outputs and logs live under that checkout. Two B300 (sm_103) issues with the sfm
  env (torch 2.7 / CUDA 12.8 / triton 3.3):
  1. triton 3.3 cannot compile for sm_103, which breaks flash-attn's rotary kernel. `models/dit.py
     apply_rotary_pos_emb` now uses the torch path on capability >= 10.3, in fp32 like the kernel; on an RTX 6000
     Ada it matches the triton kernel bit for bit (max |diff| 0). flash-attn's CUDA attention runs on B300 (max
     |diff| vs SDPA 1e-3, bf16).
  2. NVRTC 12.8 cannot JIT for sm_103 (`torch.special.entr` in the eval). So G3 only trains (`TRAIN_DONE` marker),
     and `sweep.py --eval-g3` evaluates each G3-trained cell on Unicorn (1 GPU, ~4 min) into
     `outputs/eflm_cov_reparam_fisher_cond/<run>/eval`, with `<run>/checkpoints` linked to the G3 run. Every cell
     is therefore evaluated with the same code on the same kind of GPU. Checked end to end on a 300-step G3 smoke
     run.
- **Eval check (2026-10-21):** the six (768, seed 1) originals were re-evaluated with the current code (same
  sampler settings) into `outputs/eflm_cov_reparam_fisher_cond/reeval/`. Val ppl reproduces to <0.2 %, GenPPL to
  within ~0.4 (naive 12.40 / 12.00 / 12.56 vs 12.12 / 11.81 / 12.56 original), so the older-checkout eval of the
  naive originals is comparable with the new cells; run-to-run GenPPL noise at 64 samples is ~0.3.
- **G3 reruns (2026-10-22):** 74541, 74542 and 74544 (naive R 0.5 / 1.0 and gauss R 0.5, dim 768 seed 3) failed
  to save a checkpoint ("No space left on device" on /share/thickstun, g3-compute-08). They were resubmitted as
  76965-76967 and trained from scratch.
- **NELBO (`nelbo.py`, Unicorn, 1 GPU per dim):** the codebase's EFLM ELBO (`algo.loss_type=elbo`, EMA weights of
  last.ckpt), nats per token.
  - Protocol of dev2's `elbo_tune_tinystories_256` loss table: the first 2048 sequences of the shuffled train
    loader (seed 1) or of the valid loader, one (t, noise) draw per sequence (seed 1). Train and valid get the
    same draws, and so do all cells.
  - Output: `data/nelbo/<run>_{train,valid}.json`.
  - Fix needed in `algo.py`: the merged `EFLM.reconstruction_loss` passed `context=None`, which crashes the
    Gaussian readout. It now passes `trainer_base.TrainingContext(alpha_t=alpha_t)`.
- **Per-dim figures (seed 1):**
  - `visualization/angular_fisher_sigma_t.py --group ed-<dim>` gives `figures/angular_fisher_sigma_t_ed-<dim>_{t,snr}.png`,
    with data in `data/ed-<dim>/`. ed-768 reuses the originals' `data/*.json`.
  - `visualization/codebook_eigen_dist.py` and `codebook_eigen_proj_scatter.py` write to
    `figures/codebook_eflm/ed-<dim>/`.
- **Table:** `report.py` prints mean ± std over seeds per arm x R x dim. Results are in RESULTS.md (GOAL2).

Run:

    python experiments/eflm_cov_reparam_fisher_cond/sweep.py --cluster unicorn --dims 16 48 --seeds 1 2
    python experiments/eflm_cov_reparam_fisher_cond/sweep.py [--cluster g3]     # the rest; skips done / queued cells
    python experiments/eflm_cov_reparam_fisher_cond/sweep.py --cluster g3 --seeds 2 3   # what ran on G3
    python experiments/eflm_cov_reparam_fisher_cond/sweep.py --eval-g3          # Unicorn evals of G3-trained cells
    python experiments/eflm_cov_reparam_fisher_cond/nelbo.py --runs <run> ...   # GPU node; skips existing JSONs
    python experiments/eflm_cov_reparam_fisher_cond/report.py                   # GOAL2 table
