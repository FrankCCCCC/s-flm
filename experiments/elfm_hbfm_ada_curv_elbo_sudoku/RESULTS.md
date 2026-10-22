# elfm_hbfm_ada_curv_elbo_sudoku — Results

ELBO vs CE training of EFLM and HBFM (adaptive curvature) on hard Sudoku, against MDLM (design:
`EXPERIMENT.md`, math: `math.md`). Regenerate the tables with
`python experiments/elfm_hbfm_ada_curv_elbo_sudoku/report.py`. Every `±` in the per-configuration
table is the sample std over seeds (ddof=1); in the per-cell table it is the s.e. over the
likelihood passes.

**Status: in progress — stage 1 running in a 6-slot pilot (2026-10-10 01:40 EDT).**

<!-- report:begin -->
23 cells with evals.

### Every cell

| arm | trained on | LR | seed | board acc | NELBO (nats / solution token) | CE objective |
|---|---|---|---|---:|---:|---:|
| EFLM (R=8, tau 0.5) | ce | 5e-4 | 1 | 46.8% | 0.0196 ± 0.0007 (n=4) | 0.0246 ± 0.0008 (n=4) |
| EFLM (R=8, tau 0.5) | ce | 5e-4 | 2 | 45.1% | 0.0214 ± 0.0004 (n=4) | 0.0248 ± 0.0007 (n=4) |
| EFLM (R=8, tau 0.5) | ce | 5e-4 | 3 | 43.5% | 0.0219 ± 0.0006 (n=4) | 0.0271 ± 0.0004 (n=4) |
| EFLM (R=8, tau 0.5) | elbo | 3e-4 | 1 | 48.8% | 0.0173 ± 0.0003 (n=4) | 0.0275 ± 0.0003 (n=4) |
| EFLM (R=8, tau 0.5) | elbo | 3e-4 | 2 | 26.2% | 0.0294 ± 0.0013 (n=4) | 0.0490 ± 0.0004 (n=4) |
| EFLM (R=8, tau 0.5) | elbo | 3e-4 | 3 | 41.3% | 0.0227 ± 0.0004 (n=4) | 0.0342 ± 0.0004 (n=4) |
| EFLM (R=8, tau 0.5) | elbo | 5e-4 | 1 | 36.6% | 0.0203 ± 0.0001 (n=4) | 0.0483 ± 0.0009 (n=4) |
| EFLM (R=8, tau 0.5) | elbo | 1e-3 | 1 | 31.4% | 0.0228 ± 0.0003 (n=4) | 0.0486 ± 0.0010 (n=4) |
| HBFM (ada curv, b=4) | ce | 3e-4 | 1 | 66.5% | 0.0172 ± 0.0003 (n=4) | 0.0095 ± 0.0001 (n=4) |
| HBFM (ada curv, b=4) | ce | 3e-4 | 2 | 51.5% | 0.0245 ± 0.0002 (n=4) | 0.0132 ± 0.0001 (n=4) |
| HBFM (ada curv, b=4) | ce | 3e-4 | 3 | 59.4% | 0.0216 ± 0.0006 (n=4) | 0.0125 ± 0.0002 (n=4) |
| HBFM (ada curv, b=4) | ce | 5e-4 | 1 | 66.2% | 0.0175 ± 0.0002 (n=4) | 0.0093 ± 0.0000 (n=4) |
| HBFM (ada curv, b=4) | ce | 1e-3 | 1 | 47.8% | 0.0232 ± 0.0001 (n=4) | 0.0123 ± 0.0002 (n=4) |
| HBFM (ada curv, b=4) | elbo | 3e-4 | 1 | 54.6% | 0.0229 ± 0.0002 (n=4) | 0.0142 ± 0.0002 (n=4) |
| HBFM (ada curv, b=4) | elbo | 5e-4 | 1 | 49.7% | 0.0225 ± 0.0003 (n=4) | 0.0143 ± 0.0002 (n=4) |
| HBFM (ada curv, b=4) | elbo | 1e-3 | 1 | 56.7% | 0.0169 ± 0.0003 (n=4) | 0.0120 ± 0.0002 (n=4) |
| HBFM (ada curv, b=4) | elbo | 1e-3 | 2 | 50.0% | 0.0184 ± 0.0004 (n=4) | 0.0277 ± 0.0001 (n=4) |
| HBFM (ada curv, b=4) | elbo | 1e-3 | 3 | 62.7% | 0.0162 ± 0.0003 (n=4) | 0.0111 ± 0.0002 (n=4) |
| MDLM | elbo | 3e-4 | 1 | 30.6% | 0.0183 ± 0.0001 (n=4) | — |
| MDLM | elbo | 5e-4 | 1 | 34.0% | 0.0171 ± 0.0002 (n=4) | — |
| MDLM | elbo | 1e-3 | 1 | 39.3% | 0.0142 ± 0.0001 (n=4) | — |
| MDLM | elbo | 1e-3 | 2 | 23.5% | 0.0222 ± 0.0002 (n=4) | — |
| MDLM | elbo | 1e-3 | 3 | 43.5% | 0.0134 ± 0.0001 (n=4) | — |

### Per configuration (mean ± std over seeds)

| arm | trained on | LR | seeds | board acc | NELBO (nats / solution token) |
|---|---|---|---|---:|---:|
| EFLM (R=8, tau 0.5) | ce | 5e-4 | 1,2,3 | **45.1 ± 1.6** (n=3) | **0.0209 ± 0.0012** (n=3) |
| EFLM (R=8, tau 0.5) | elbo | 3e-4 | 1,2,3 | **38.7 ± 11.5** (n=3) | **0.0231 ± 0.0061** (n=3) |
| EFLM (R=8, tau 0.5) | elbo | 5e-4 | 1 | 36.6 (n=1) | 0.0203 (n=1) |
| EFLM (R=8, tau 0.5) | elbo | 1e-3 | 1 | 31.4 (n=1) | 0.0228 (n=1) |
| HBFM (ada curv, b=4) | ce | 3e-4 | 1,2,3 | **59.1 ± 7.5** (n=3) | **0.0211 ± 0.0037** (n=3) |
| HBFM (ada curv, b=4) | ce | 5e-4 | 1 | 66.2 (n=1) | 0.0175 (n=1) |
| HBFM (ada curv, b=4) | ce | 1e-3 | 1 | 47.8 (n=1) | 0.0232 (n=1) |
| HBFM (ada curv, b=4) | elbo | 3e-4 | 1 | 54.6 (n=1) | 0.0229 (n=1) |
| HBFM (ada curv, b=4) | elbo | 5e-4 | 1 | 49.7 (n=1) | 0.0225 (n=1) |
| HBFM (ada curv, b=4) | elbo | 1e-3 | 1,2,3 | **56.5 ± 6.4** (n=3) | **0.0172 ± 0.0011** (n=3) |
| MDLM | elbo | 3e-4 | 1 | 30.6 (n=1) | 0.0183 (n=1) |
| MDLM | elbo | 5e-4 | 1 | 34.0 (n=1) | 0.0171 (n=1) |
| MDLM | elbo | 1e-3 | 1,2,3 | **35.4 ± 10.5** (n=3) | **0.0166 ± 0.0049** (n=3) |

<!-- report:end -->

## 1. Verification of the losses (no training involved)

`tests/test_elbo_loss.py`: **8 / 8 pass** (RTX 6000, job 182598, 2026-10-10 20:21, on the loss code
as restructured that evening into named pieces — `ce_loss`, `diffusion_loss`, `prior_loss`,
`reconstruction_loss`, `endpoint_losses`, `elbo_loss` — with the dispatch inside `nll_per_token`).
In the first run (2080 Ti, job 156271, 01:17) the eighth test failed only through its own setup:
it probed a block parameter that the adaLN-Zero initialization leaves without gradient under any
loss; it now probes the adaLN modulation bias, as the repo's other tests do.

| claim (`math.md`) | check | result |
|---|---|---|
| §0 tightness, EFLM | unigram, closed-form Gaussian posterior through `forward`; NELBO through `nll` (prior + diffusion + reconstruction) vs $\mathsf H(p)$, 16k sequences | pass (within 4 s.e. + 1 %) |
| §0 a bound | tempered posterior (EFLM) / biased residual (HBFM) | pass: the value rises by > 5 % |
| §0 tightness, HBFM | horosphere readout, zero residual = context-free Bayes posterior, uniform words: NELBO vs $\log 12$ | pass (within 4 s.e. + 1 %) |
| §2 adaptive curvature | mixed $\lambda_i\in[1/4,4]$ per position: same mean as $\lambda = 1$, $= \log 12$ | pass |
| CE path untouched | `loss_type=ce` of both models equals the (weighted) CE of before; unknown types rejected | pass |
| trainable | real DiTs under bf16 autocast, gradients at embeddings / trunk / curvature readout | pass |

The related suites (`test_hbfm.py`, `test_hbfm_ada_curv.py`, `test_hbfm_sampler.py`,
`test_eflm_sde.py`): 70 pass, 6 fail — the five `test_hbfm.py` failures the previous experiment
documented (the yaml defaults `forward_type: naive` / float32 vs the tests' horosphere / float64
assumptions) plus `test_curvature_is_read_without_dropout`, which asserts `model.dropout > 0` while
the shared test overrides set it to 0. None involves the code changed here.

Sampler: the `em` step of the HBFM sampler (the draft's `step_type` plumbing made runnable) reproduces
the recorded board accuracy of the cb-4 anchor checkpoint, seed 1: **66.55 %** (1331 / 2000) vs 66.5 %
in `hbfm_rlf_ada_curv/RESULTS.md`.

## 2. Smoke (300 steps + the evals, 2080 Ti)

Pilot 156441, four cells sharing the node (speeds in `EXPERIMENT.md`, Compute). The likelihood pass
(`mode=ppl_eval` on the 2000 validation boards, one (t, noise) draw per board) works for all three
methods; on the reused HBFM-CE checkpoint (cb-4, LR 3e-4, seed 1) it gives

| objective | val/nll (nats / solution token), one pass |
|---|---|
| NELBO (`loss_type=elbo`): weighted path-KL + reconstruction CE at the horizon | 0.0176 |
| CE objective (`loss_type=ce`): the weighted denoising CE it was trained on | 0.0096 |

The CE number matches that study's held-out weighted loss for this cell (0.0102 ± 0.0007, 256 boards);
the bound is 1.8× it.

The three 300-step training cells (ELBO loss; EMA weights, which have barely left their
initialization at 300 steps, as in the earlier smoke): every cell trained, saved `last.ckpt`, and ran
both likelihood passes and the board eval.

| cell (300 steps) | NELBO | CE objective | boards |
|---|---|---|---|
| HBFM-ELBO, b = 4, LR 3e-4 | 2.337 | 1.679 | 0 / 2000 |
| EFLM-ELBO, LR 3e-4 | 2.801 | 1.187 | 0 / 2000 |
| MDLM, LR 3e-4 | 2.145 | — | 0 / 2000 |

(log 12 = 2.485 is the entropy of a uniform guess over the 12 tokens.) The first pilot's cells had
crashed before training: the per-cell TMPDIR name made the dataloader workers' Unix-socket path exceed
108 bytes (`OSError: AF_UNIX path too long`); the path is now per (job, GPU slot). The
`Device or resource busy` tracebacks at the end of every cell are the workers failing to remove their
temp dir on NFS at exit (known; harmless).

## 3. Stage 1 — LR per loss (seed 1)

Pilot job 156442 (6 slots, 2026-10-10 01:54 → ~2026-10-11 07:00). The table at the top of this file
(`report.py`) holds the numbers; what follows is the reading as the cells came in.

**HBFM-CE, LR.** 3e-4: 66.5 / 51.5 / 59.4 % (the reused anchor, 3 seeds; NELBO 0.0211 ± 0.0037);
5e-4: 66.2 %, NELBO 0.0175 (seed 1); 1e-3: 47.8 %, NELBO 0.0232. By the pre-registered rule (seed-1
accuracy, NELBO breaks ties) the control stays at LR 3e-4, i.e. the existing three seeds.

**HBFM-ELBO: LR 5e-4 (the first ELBO cell) 49.7 % boards, NELBO 0.0225 ± 0.0003, CE objective
0.0143; LR 3e-4 54.6 %, NELBO 0.0229 ± 0.0002, CE objective 0.0142; LR 1e-3 56.7 %, NELBO
0.0169 ± 0.0003, CE objective 0.0120.** The pick is **LR 1e-3** (best accuracy, and the best bound
of any HBFM cell: 0.0169 vs the CE-trained 0.0172–0.0175), the opposite of the CE arm, whose LR 1e-3
cell is its worst (47.8 %). Yet the picked cell stays 10 points below the CE arm's seed-1 accuracy
(66.2–66.5 %): for this pair the bound and the greedy accuracy part ways. With three seeds (§4) the
arms overlap — HBFM-ELBO 56.5 ± 6.4 % vs HBFM-CE 59.1 ± 7.5 % — so the seed-1 gap is mostly the
seed-1 CE model being the best of its three (66.5 / 51.5 / 59.4 %). The 5e-4 / 3e-4
cells, worse than the CE twins on every number (66.2 % / 0.0175 / 0.0093 at 5e-4; 66.5 % / 0.0172 /
0.0095 at 3e-4), were diagnosed first; §3b below adds the 1e-3 pair. Diagnosis so far
(`hbfm_rlf_ada_curv/curvature_stats.py`, `loss_stats.py` on the two LR 5e-4 checkpoints, 256
validation boards):

| checkpoint (LR 5e-4, seed 1) | λ of cells to solve | λ of given cells | λ of separators | prior accuracy at t = 0 | own objective at learned λ | at λ = 1 |
|---|---|---|---|---|---|---|
| ELBO-trained | 0.361 ± 0.050 | 0.516 ± 0.087 | 0.875 ± 0.035 | 0.829 | NELBO 0.0215 ± 0.0004 (var 0.0012) | 0.0221 (var 0.0032) |
| CE-trained | 0.336 ± 0.073 | 0.494 ± 0.126 | 1.040 ± 0.150 | 0.860 | weighted CE 0.0099 ± 0.0003 (var 0.0006) | 0.0101 (var 0.0015) |
| ELBO-trained, LR 1e-3 (the pick; 512 boards) | 0.369 ± 0.020 | 0.425 ± 0.041 | 1.162 ± 0.145 | 0.851 | — | — |
| CE-trained, LR 1e-3 (512 boards) | 0.309 ± 0.020 | 0.443 ± 0.129 | 0.517 ± 0.033 | 0.833 | — | — |

- The adaptive curvature is **not** the cause: both models learned the same clocks (cells to solve
  at λ ≈ 0.35, Spearman(λ, entropy) −0.78 / −0.83), so the sampler takes the same per-position steps
  and the training mass sits at the same heat times; the variance mechanism works for the ELBO
  estimator as well (2.7× lower variance at the learned λ).
- The ELBO-trained model is a worse denoiser at the same budget under **both** functionals, and its
  training NELBO (0.030 over the last 2k steps, 0.023 at 20k on the EMA weights) sits above the
  value the CE-trained model attains under the same bound (0.0175). So this is an optimization
  effect of the objective, not a looser bound.
- **Where the optimization fails** (`grad_snr.py`, job 183161 on an RTX 6000 Ada: the gradient of
  each objective at the *online* weights of `last.ckpt` over 31 micro-batches of 64 validation
  boards, signal ‖E g‖², noise E‖g − E g‖²; and the per-t pieces of the objective on the solution
  cells of 256 boards at the EMA weights):

  | weights (seed 1) | objective | SNR per micro-batch of 64 | micro-batches to SNR 1 | gradient noise |
  |---|---|---|---|---|
  | HBFM ELBO-trained, LR 5e-4 | elbo | 0.13 | 7.6 | 0.0084 |
  | HBFM ELBO-trained, LR 5e-4 | ce | 0.15 | 6.6 | 0.0059 |
  | HBFM CE-trained, LR 5e-4 | elbo | **0.021** | **49** | **0.072** |
  | HBFM CE-trained, LR 5e-4 | ce | 0.11 | 8.9 | 0.0075 |
  | HBFM ELBO-trained, LR 3e-4 | elbo | 0.050 | 20 | 0.026 |
  | HBFM ELBO-trained, LR 3e-4 | ce | 0.093 | 11 | 0.011 |
  | HBFM CE-trained, LR 3e-4 | elbo | 0.045 | 22 | 0.036 |
  | HBFM CE-trained, LR 3e-4 | ce | 0.071 | 14 | 0.016 |
  | HBFM ELBO-trained, LR 1e-3 (the pick) | elbo | 0.076 | 13 | 0.012 |
  | HBFM ELBO-trained, LR 1e-3 (the pick) | ce | 0.026 | 38 | 0.012 |
  | HBFM CE-trained, LR 1e-3 | elbo | 0.124 | 8.0 | 0.011 |
  | HBFM CE-trained, LR 1e-3 | ce | 0.162 | 6.2 | 0.0034 |
  | EFLM ELBO-trained, LR 3e-4 | elbo | 0.023 | 45 | 0.026 |
  | EFLM ELBO-trained, LR 3e-4 | ce | 0.053 | 19 | 0.041 |
  | EFLM CE-trained, LR 5e-4 | elbo | 0.057 | 18 | 0.022 |
  | EFLM CE-trained, LR 5e-4 | ce | 0.032 | 32 | 0.023 |

  At the CE-trained HBFM the ELBO's gradient noise is 8.6× what it is at the ELBO-trained model
  (and 10× the CE's noise at the same weights) while its signal is the same size: the HBFM ELBO
  is a heavy-tailed estimator — importance weight 1/(3t) times the hyperbolic rate of a confident
  mistake — that is noise-dominated precisely at confident solutions (at the training batch of
  256 = 4 micro-batches its SNR at the CE solution is 0.08). SGD on it cannot hold such a solution
  and settles where the estimator is quiet. The profile shows what that costs: the ELBO-trained
  posterior has twice the entropy of the CE-trained one at every t (0.20 vs 0.10 at the origin,
  0.055 vs 0.024 at t = 0.5), a lower argmax accuracy (0.902 vs 0.920 at the origin, 0.977 vs
  0.986 at t = 0.5) and a worse rate in the mid-range (t = 0.15: 0.0133 vs 0.0035; t = 0.5: 0.101
  vs 0.072); only at the origin end (t ≥ 0.9) are the rates equal while CE and argmax still differ
  (the rate, 9 drift coordinates per position, does not pin a 12-word posterior — a secondary
  effect). The LR 3e-4 pair shows the same picture, weaker (noise 0.036 vs 0.026). For EFLM the
  ELBO's noise is the same at both solutions (0.026 vs 0.022: a confident Euclidean mistake costs
  at most (2R)²) and there ELBO training *helps* (below).
- Not run (outside the pre-registered protocol): variance reduction of the HBFM estimator (a
  proposal matched to the 1/(3t) weight, e.g. `trunc_exp` with a finite horizon, or a larger
  batch) or an ELBO fine-tune from the CE checkpoint, which would test directly whether the ELBO
  can hold the CE solution.

**§3b. The LR 1e-3 pair: a better model that solves fewer boards.** The picked HBFM-ELBO cell is,
on forward-bridge states, at least as good a denoiser as the best CE-trained model (`grad_snr.py`
profile, solution cells of 256 validation boards, EMA weights):

| t (α) | path-KL rate: ELBO 1e-3 / CE 5e-4 / CE 1e-3 | denoising CE | argmax accuracy | entropy of p_θ |
|---|---|---|---|---|
| 0.999 (origin) | **0.329** / 0.404 / 0.449 | **0.206** / 0.228 / 0.228 | 0.918 / 0.920 / 0.903 | 0.187 / 0.101 / 0.205 |
| 0.5 | 0.064 / 0.072 / 0.103 | 0.046 / 0.039 / 0.057 | 0.984 / 0.986 / 0.981 | 0.043 / 0.024 / 0.050 |
| 0.2 | 0.0079 / 0.0113 / 0.0186 | 0.0069 / 0.0081 / 0.0101 | 0.998 / 0.998 / 0.996 | 0.006 / 0.002 / 0.007 |

(lower rate and CE, equal argmax accuracy, a calibrated rather than over-confident posterior:
the CE-trained model has half the entropy at the same CE). Its gradients show no pathology
(table above: ELBO SNR 0.076 at its own weights; at the CE-1e-3 weights 0.124, noise 0.011), its
curvature is unremarkable (cells to solve λ 0.369 ± 0.020 vs the CE models' 0.31–0.34). Yet it
solves 56.7 % of the boards against 66.2 %. The 2000 sampled boards (protocol decoding: 180 EM
steps, exact velocity over the whole vocabulary, greedy end) and the same checkpoints under other
decodings (`logs/steps_diag.sbatch`, `logs/temp_diag.sbatch`; `outputs/.../diag/decode/`):

| seed-1 checkpoint | boards solved | wrong cells per board | cell error rate | failed boards with ≤ 2 / 3–8 / > 8 wrong cells |
|---|---|---|---|---|
| HBFM-ELBO 1e-3 | 56.7 % | 6.75 | 8.33 % | 1 / 136 / 729 |
| HBFM-ELBO 3e-4 | 54.6 % | 7.11 | 8.77 % | 7 / 146 / 754 |
| HBFM-ELBO 5e-4 | 49.7 % | 7.77 | 9.59 % | 1 / 175 / 830 |
| HBFM-CE 5e-4 | 66.2 % | 5.31 | 6.56 % | 5 / 92 / 578 |
| HBFM-CE 3e-4 | 66.5 % | 5.44 | 6.72 % | 1 / 79 / 589 |
| HBFM-CE 1e-3 | 47.8 % | 8.18 | 10.1 % | 9 / 165 / 871 |
| EFLM-ELBO 3e-4 | 48.7 % | 8.15 | 10.1 % | 1 / 168 / 856 |
| EFLM-CE 5e-4 | 46.8 % | 8.20 | 10.1 % | 5 / 202 / 858 |

| decoding (seed 1) | HBFM-ELBO 1e-3 | HBFM-CE 5e-4 |
|---|---|---|
| protocol: 180 steps, exact velocity, T = 1 | 56.7 % | 66.2 % |
| 60 steps | 57.4 % | 65.2 % |
| 500 steps | 55.7 % | 65.2 % |
| velocity committed to the argmax word (top-1) | 57.9 % | 66.0 % |
| posterior temperature 0.5 / 0.25 | 56.9 % / 57.7 % | 66.4 % (T = 0.5) |
| posterior temperature 2.0 | — | 66.1 % |

Reading. (i) No decoding knob moves the ELBO model by more than 1.2 points, and the CE model is
insensitive to all of them: the gap is in the trajectories the model's own posterior produces,
not in the step size, the commitment of the velocity or the sharpness of the posterior. (ii) The
sampler improves the CE model on its single-shot prediction from the origin (cell error 8.0 % at
t = 0.999 teacher-forced → 6.6 % sampled) and does nothing for the ELBO model (8.2 % → 8.3 %):
iterative refinement is worth 1.5 points per cell to the CE-trained posterior and nothing to the
ELBO-trained one. (iii) Every model fails by cascades of the same size (≈ 15.6 wrong cells per
failed board for both); the ELBO model fails more boards (43 % vs 34 %). So this ELBO-trained
posterior, better on forward-bridge states, transfers worse to the sampler's own partially
committed states — the one kind of state neither objective ever scores in training. Why the CE
objective shapes a posterior that extrapolates better there is open; the facts above rule out the
bound, the curvature, the step size, the velocity's commitment, the temperature, and the
posterior's accuracy on forward-bridge states. Two caveats from §4: this is a seed-1 pair, in which
the CE model is the best of its three seeds (66.5 / 51.5 / 59.4 %) and the ELBO model the middle
of its (56.7 / 50.0 / 62.7 %), so the 10-point gap overstates the arms' difference (2.7 points on
the three-seed means, within the seed spread); and the mechanism it exposes — a model can be the
better per-token denoiser and still solve fewer boards — is what makes the board accuracy at this
budget so seed-sensitive for both objectives.

**EFLM-ELBO, LR 3e-4 (first EFLM cell): 48.8 % boards, NELBO 0.0173, CE objective 0.0275** vs the
CE arm's 45.1 ± 1.6 % and 0.0209 ± 0.0012: the ELBO loss ties or beats the CE loss on accuracy and
tightens the bound by 17 %.

**MDLM (NELBO-trained, the reference): LR 3e-4 30.6 % boards, NELBO 0.0183 ± 0.0001; LR 5e-4
34.0 %, 0.0171 ± 0.0002; LR 1e-3 39.3 %, 0.0142 ± 0.0001.** The pick is **LR 1e-3**, best on both
counts and the best bound of any cell of the study (0.0142 < HBFM-ELBO 0.0169 < EFLM-ELBO 0.0173),
while its board accuracy is far below both continuous arms (39 % vs 57 % and 49 %). Both numbers
improve monotonically with the LR over the pre-registered grid, so MDLM may not be at its best LR;
the grid is the same for every arm and was fixed in advance, and the ranking of the arms is not
affected (even the CE controls' worst cells beat MDLM's best on accuracy).

**EFLM-ELBO, LR: 3e-4 48.8 % boards, NELBO 0.0173 ± 0.0003, CE objective 0.0275; 5e-4 36.6 %,
0.0203 ± 0.0001, 0.0483; 1e-3 31.4 %, 0.0228 ± 0.0006, 0.0486.** The pick is **LR 3e-4**, best on both
counts; EFLM-ELBO degrades steeply with the LR (the CE arm had been tuned to 5e-4 in its own study,
so the ELBO prefers a smaller step than the CE here, the opposite of HBFM).

**LR picks (seed-1 accuracy, NELBO breaks ties):** HBFM-ELBO 1e-3, EFLM-ELBO 3e-4, MDLM 1e-3; the
CE controls keep their studies' LRs (HBFM 3e-4, EFLM 5e-4). Stage 1 ended 2026-10-11 06:19 (pilot job
156442, 28.4 h). Stage 2 went in as three 2-slot pilots as each pick became known: HBFM-ELBO seeds
2–3 = job 183756 (2026-10-10 21:00), MDLM seeds 2–3 = job 193573 (2026-10-11 06:23), EFLM-ELBO
seeds 2–3 = job 195710 (2026-10-11 06:23).

## 4. Stage 2 — three seeds at the picked LR

All cells done 2026-10-11 22:20 (pilot jobs 183756, 193573, 195710). Hard Sudoku, 2000 validation
boards; NELBO in nats per solution token, mean of 4 evaluation passes per cell; `figures/loss_curves.png`
has the training curves of every cell.

| arm | trained on | LR | boards solved, seeds 1 / 2 / 3 | mean ± std | NELBO, seeds 1 / 2 / 3 | mean ± std |
|---|---|---|---|---:|---|---:|
| HBFM, adaptive curvature | **ELBO** | 1e-3 | 56.7 / 50.0 / 62.7 % | **56.5 ± 6.4 %** | 0.0169 / 0.0184 / 0.0162 | **0.0172 ± 0.0011** |
| HBFM, adaptive curvature | CE (control) | 3e-4 | 66.5 / 51.5 / 59.4 % | 59.1 ± 7.5 % | 0.0172 / 0.0245 / 0.0216 | 0.0211 ± 0.0037 |
| EFLM (R = 8, τ 0.5) | **ELBO** | 3e-4 | 48.8 / 26.2 / 41.3 % | **38.7 ± 11.5 %** | 0.0173 / 0.0294 / 0.0227 | **0.0231 ± 0.0061** |
| EFLM (R = 8, τ 0.5) | CE (control) | 5e-4 | 46.8 / 45.1 / 43.5 % | 45.1 ± 1.6 % | 0.0196 / 0.0214 / 0.0219 | 0.0209 ± 0.0012 |
| MDLM (reference) | its NELBO | 1e-3 | 39.3 / 23.5 / 43.5 % | 35.4 ± 10.5 % | 0.0142 / 0.0222 / 0.0134 | 0.0166 ± 0.0049 |

Differences of three-seed means (Welch's t, 3 vs 3 seeds; |t| > 3 is roughly the 5 % level here):

| comparison | boards solved | NELBO |
|---|---|---|
| HBFM-ELBO − HBFM-CE | −2.7 points (t = −0.5) | −0.0039 (t = −1.8) |
| EFLM-ELBO − EFLM-CE | −6.4 points (t = −0.9) | +0.0022 (t = +0.6) |
| HBFM-ELBO − MDLM | +21.0 points (t = +3.0) | +0.0006 (t = +0.2) |
| EFLM-ELBO − MDLM | +3.3 points (t = +0.4) | +0.0065 (t = +1.4) |

Reading:

- **HBFM.** The ELBO arm ties the CE control on accuracy within the seed spread (every ELBO seed lies
  inside the CE seeds' range 51.5–66.5 %), and its bound is tighter: all three ELBO seeds
  (0.0162–0.0184) sit at or below the CE control's best seed (0.0172), the mean is 0.0039 lower
  (t = −1.8). The seed-1 pair of §3b (56.7 vs 66.5 %) was the widest gap of the nine possible pairs.
- **EFLM.** The ELBO arm is 6.4 points below the CE control on the mean and no better on the bound,
  both driven by seed 2 (26.2 %, NELBO 0.0294), which trained worse from ~8k steps on (median
  training loss 0.171 at 8–10k vs 0.140 / 0.145 for seeds 1 and 3) without diverging. Seeds 1 and 3
  bracket the CE mean (48.8 / 41.3 vs 45.1 %), and seed 1 holds the tightest EFLM bound of the
  study (0.0173 vs the control's 0.0196–0.0219). Neither difference is significant at three seeds.
- **Seed variance.** ELBO training spreads the seeds far more than CE training for EFLM (std 11.5 vs
  1.6 points) and about as much for HBFM (6.4 vs 7.5); MDLM spreads as much as the ELBO arms (10.5).
  At 20k steps on this task, run-to-run variation is the first-order effect and three seeds do not
  resolve differences of a few points.
- **Against MDLM.** The bounds of HBFM-ELBO and MDLM are equal within noise (0.0172 ± 0.0011 vs
  0.0166 ± 0.0049; MDLM's best seeds, 0.0134 and 0.0142, are the tightest cells of the study, its
  worst, 0.0222, is looser than every HBFM-ELBO seed); EFLM-ELBO's is looser (0.0231 ± 0.0061).
  On boards solved HBFM-ELBO beats MDLM by 21 points (t = +3.0, the one significant difference);
  EFLM-ELBO by 3 points (not significant). Accuracy does not follow the bound: at the same bound
  HBFM solves 57 % of the boards and MDLM 35 %.

## 5. Conclusion

1. **Implementation.** `algo.loss_type=elbo` gives EFLM and HBFM a trainable negative ELBO
   (`math.md`: EFLM = VDM diffusion term with Plaid's categorical reparameterization + reconstruction
   CE + prior KL; HBFM = importance-weighted path-KL rate + reconstruction CE at the horizon, with the
   adaptive curvature's clock and weight), verified tight at the Bayes posterior and loosening away
   from it (`tests/test_elbo_loss.py`, 8 tests), with the CE path untouched. It costs ~2× the CE
   loss per sample (a second forward pass).
2. **H1 (accuracy ties CE): supported for HBFM, not for EFLM.** HBFM-ELBO 56.5 ± 6.4 % vs HBFM-CE
   59.1 ± 7.5 % (−2.7 points, t = −0.5); EFLM-ELBO 38.7 ± 11.5 % vs EFLM-CE 45.1 ± 1.6 % (−6.4
   points, t = −0.9, one of three seeds at 26 %). Under the criterion "at least tie with the CE
   version", HBFM passes within the seed noise and EFLM does not on the three-seed mean.
3. **H2 (tighter bound than CE): supported for HBFM, not for EFLM.** HBFM 0.0172 ± 0.0011 vs
   0.0211 ± 0.0037; EFLM 0.0231 ± 0.0061 vs 0.0209 ± 0.0012, where only the best ELBO seed (0.0173)
   improves on the control.
4. **H3 (geometries vs MDLM): refuted in its strong form.** HBFM's bound is not far below MDLM's — the
   two are equal within noise on Sudoku, and on TinyStories MDLM is far ahead (1.58 vs 2.10 nats per
   token, §6). EFLM's bound sits above HBFM's on both datasets, as predicted. What the geometry buys
   is accuracy, not likelihood: HBFM solves 21 points more boards than MDLM at the same bound.
5. **Why a tighter bound need not solve more boards (§3, §3b).** The seed-1 HBFM-ELBO model is the
   better per-token denoiser on forward-bridge states (lower path-KL rate and CE, equal argmax
   accuracy, calibrated rather than over-confident) yet gains nothing from the sampler's 180-step
   refinement, whereas the CE model gains 1.5 points per cell from it; no decoding change (step
   count, committed velocity, temperature) closes that gap. The adaptive curvature, gradient noise
   and the LR were ruled out as causes. Across seeds the effect is within the run-to-run spread.
6. **Fairness.** Same data, tokenizer, DiT (512 / 8 / 8), batch 256, 20k steps, EMA, AdamW, LR grid
   {3e-4, 5e-4, 1e-3} picked by seed-1 accuracy, 3 seeds, one evaluation protocol (180 sampler
   steps, exact velocity, greedy end; 4 NELBO passes). The CE controls are the earlier studies'
   runs on the same code (re-scored here, §1). Two LR caveats: EFLM-ELBO and MDLM were best at an
   edge of the grid (3e-4 and 1e-3 respectively), so both may improve with a wider grid.
7. **Follow-ups (not run).** More seeds, or an ELBO fine-tune from the CE checkpoints, to get HBFM's
   tighter bound without the seed risk; a lower LR for EFLM-ELBO and a higher one for MDLM; variance
   reduction of the HBFM estimator (a proposal matched to its 1/(3t) weight); and, for the
   open mechanism of item 5, scoring the posterior on sampler-generated states.

## 6. TinyStories: bounds of the best pre-trained models

`tinystories_elbo.py` (2026-10-10 15:50–17:40, jobs 172591–172597): one `ppl_eval` pass per
checkpoint over the full validation set (1172 batches × 16 sequences of 256 tokens; one (t, noise)
draw per sequence), nats per token. "CE objective" is the weighted denoising CE the model was trained
on (the `val/nll` of the earlier studies), which is not a bound; both CE numbers and MDLM's NELBO
reproduce the recorded values of those studies to the fourth digit.

| model (best cell of its study) | seeds | NELBO | as PPL | CE objective |
|---|---|---:|---:|---:|
| MDLM, LR 1e-3 (`naive_ar_tinystories_s256`) | 1–3 | **1.582 ± 0.007** | 4.87 | — |
| EFLM, R = 0.5, tau* = 2.44, LR 1e-3 (`eflm_rescale_auto_trunc_tinystories_256`) | 1–3 | **1.893 ± 0.004** | 6.64 | 3.280 |
| HBFM, ada curv b = 4, trunc-exp ub = 1, LR 3e-4 (`hbfm_rlf_ada_curv_tinystories_256`) | 1 | **2.104** | 8.20 | 0.494 |

- MDLM's bound is the tightest; EFLM's sits 0.31 nats/token above it, HBFM's 0.52. For HBFM the
  path-KL bound is 4.3× the CE integral it was trained on (on Sudoku the ratio is ~1.8: §2–3), for
  EFLM the bound is *below* its CE integral (the CE on the autonomous clock integrates the full
  noise-fraction band, the bound weights the quadratic by −SNR′/2 ≤ 0.6).
- Seeds 2–3 of the HBFM cell are on the A100 node's scratch and were not pulled; the EFLM/MDLM seed
  spreads (0.004–0.007) suggest the single HBFM read is representative to ~0.01.
- Decomposition (`ts_decompose.py`, jobs 179080–179081, same validation set, EMA weights; the
  endpoint terms scored alone, the diffusion term = NELBO − endpoints):

  | model | endpoints (reconstruction CE at the horizon, + prior KL for EFLM) | diffusion term |
  |---|---:|---:|
  | HBFM, trunc-exp ub = 1 | 0.431 | 1.673 |
  | EFLM, tau* = 2.44 | 0.088 | 1.805 |

  HBFM's trained heat-time range stops at t = 1, where 16 H³ factors at K = −0.5 have not resolved
  the token yet: the readout's CE there is 0.43 nats/token (its importance-weighted CE integral over
  [0, 1] is 0.49, so most of that integral sits near the horizon), and the sampler decodes greedily
  at that point. The remaining 1.67 is the path-KL of the CE-trained posterior — 3.4× the CE
  integral. EFLM's horizon is the decode point tau*(R) where the token is nearly identified (0.09),
  so its bound is almost entirely the diffusion term.
- Where the EFLM and MDLM bounds are paid (`ts_profile.py`, jobs 180714: the per-token integrand of
  each bound at fixed schedule times on the same 1024 validation sequences; the trapezoid integrals,
  1.80 and 1.60, reproduce the measured diffusion terms):

  | t | EFLM: noise fraction b | weight −SNR′/2 | MSE ‖e − x̂‖² (‖e‖² = 0.25) | integrand | denoising CE | MDLM: masked fraction 1 − α | integrand | CE of masked tokens |
  |---|---|---|---|---|---|---|---|---|
  | 0.001 | 0.087 | 295 | 0.0026 | 0.75 | 0.09 | 0.001 | 0.21 | 0.28 |
  | 0.1 | 0.111 | 177 | 0.0073 | 1.30 | 0.21 | 0.10 | 0.38 | 0.38 |
  | 0.2 | 0.141 | 105 | 0.019 | 2.00 | 0.49 | 0.20 | 0.45 | 0.45 |
  | 0.3 | 0.181 | 61 | 0.047 | 2.91 | 1.09 | 0.30 | 0.57 | 0.57 |
  | 0.4 | 0.231 | 35 | 0.103 | 3.64 | 2.26 | 0.40 | 0.75 | 0.75 |
  | 0.5 | 0.294 | 20 | 0.170 | 3.38 | 3.76 | 0.50 | 0.96 | 0.96 |
  | 0.6 | 0.376 | 11 | 0.215 | 2.32 | 4.91 | 0.60 | 1.32 | 1.32 |
  | 0.7 | 0.480 | 5.5 | 0.236 | 1.30 | 5.52 | 0.70 | 1.85 | 1.86 |
  | 0.8 | 0.613 | 2.5 | 0.244 | 0.62 | 5.79 | 0.80 | 2.69 | 2.69 |
  | 0.9 | 0.782 | 0.87 | 0.247 | 0.21 | 5.90 | 0.90 | 3.98 | 3.99 |
  | 0.999 | 0.997 | 0.008 | 0.248 | 0.00 | 5.93 | 1.00 | 5.90 | 5.91 |

  EFLM pays two thirds of its bound in the band b ∈ [0.14, 0.5] (t ∈ [0.2, 0.7]), where every
  position is partially corrupted at once and the posterior is diffuse (denoising CE 0.5–5 nats,
  MSE 0.02–0.24 of a possible 0.25) while the weight is still 5–100; its integrand is still 0.75 at
  the decode point. MDLM's integrand is the CE of the masked tokens given exact context, times
  α′/(1 − α): small until most of the sequence is masked, and at full masking it is the unigram
  entropy (5.9) that both models reach. The 0.3-nat gap is the price of inferring tokens from noisy
  neighbours in a 768-dimensional Gaussian channel with CE-trained, norm-pinned embeddings — not
  the architecture or the compute, which are identical to MDLM's here (see the discussion in the
  session notes: Plaid trains the bound itself with learned 16-dim embeddings and schedule
  endpoints; LangFlow adds self-conditioning, the output prior and an ODE-based bound, and both
  train 30× longer).

## 7. Incidents

- 2026-10-10: all A6000 / A5000 / RTX-6000 GPUs were held by another session's `hbfm_adats_*` sweep
  (27 running, 114 pending, the pending ones excluding the 2080 Ti node). The scheduler's
  `bf_max_job_user=100` keeps a user's jobs beyond its 100 highest-priority pending ones out of the
  backfill window, so this project's jobs (lower priority: fewer CPUs / less memory) sat on
  `(Priority)` with seven 2080 Ti GPUs idle until that queue drained below ~95 pending. Lowering the
  other jobs' `nice` was not done (denied by the session's permission policy); nothing of the other
  sweep was touched.
