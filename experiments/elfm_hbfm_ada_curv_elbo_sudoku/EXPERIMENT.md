# elfm_hbfm_ada_curv_elbo_sudoku — ELBO training of EFLM and HBFM on hard Sudoku

Spec: `setup.md`. Math: `math.md`. Sweep: `sweep.py` (orchestration only; calls
`scripts/train/sudoku/{hbfm,eflm_rescale_auto,mdlm}.sh` + `scripts/sample/sudoku/{...}.sh`). Table:
`report.py` → `RESULTS.md`. Analysis (compute node, nothing trained): `loss_curves.py` (training
curves from the offline W&B logs), `grad_snr.py` (gradient signal-to-noise of the two objectives at
a trained cell + the per-t profile of its posterior), `tinystories_elbo.py` / `ts_decompose.py` /
`ts_profile.py` (the TinyStories add-on).

## Question

Does training on the ELBO (a bound on the NLL) instead of the denoising CE cost board accuracy, for
the Euclidean flow (EFLM) and the hyperbolic bridge with adaptive curvature (HBFM)? And where do the
two geometries' bounds sit against MDLM's on the same task? `setup.md` asks for the ELBO and the
accuracy of each, in comparison with MDLM, with the ELBO arm at least tying its CE arm.

## Method (what was added; `math.md`)

One knob, `algo.loss_type` ∈ {`ce`, `elbo`} on EFLM and HBFM (`ce` = the loss of every earlier run,
unchanged). With `elbo` the per-token loss is the NELBO estimator with one noise level per sequence:

| | diffusion term (×1/q(t)) | reconstruction | prior |
|---|---|---|---|
| EFLM | $-\tfrac12\mathrm{SNR}'(t)\,\|\mathbf e_x-f_\theta^\top\mathbf E\|^2$ (VDM Eq. 17 with Plaid's categorical reparameterization) | CE of the readout at the clean end of the trained range ($b=e^{-\tau_{\max}}$: the decode point) | closed-form Gaussian KL at the noise floor |
| HBFM | path-KL rate $\ell_{\mathrm{rad}}+\ell_{\mathrm{ang}}$ of the bridge (`Loss.bridge_loss_elbo_refactor`), the slides' Girsanov terms | CE at the horizon of the trained heat-time range | 0 (both start at the origin) |

Both are tight at the Bayes posterior (verified: `tests/test_elbo_loss.py`, 8 tests), and the
reconstruction term costs one more forward pass per step. The adaptive curvature keeps its weight
$\lambda_i/q(t)$ and variance gradient; its mean stays $\mathsf H$ at the Bayes posterior.

## Design

Pinned by `setup.md`: hard Sudoku (30 clues), 48k train / 2k val (data seed 42); tiny DiT 512/8/8;
20k steps, batch 256, bf16, EMA 0.9999; AdamW (wd 0, betas 0.9/0.999, eps 1e-8), clip 1.0; LR from
{3e-4, 5e-4, 1e-3}; 3 seeds; eval = exact velocity, `top_k_velocity=-1`, 180 steps, greedy last
step, 2000 boards.

Arms:

| arm | fixed config (from the earlier study) | source of the CE control |
|---|---|---|
| HBFM, ada curv | (H³)³ at K=−0.5, exp proposal rate 3 (unit rate 6), horosphere readout, curvature bound 4, log-linear schedule, HyLa off, sampler t_max 6 | `hbfm_rlf_ada_curv` cb-4 anchor, LR 3e-4, seeds 1–3 (59.1 ± 7.5); same training code (the commits since touched only the sampler draft and this loss), same GPU type |
| EFLM | R = 8 (rho_min = rho_max), autonomous clock tau_max 0.5, no self-conditioning | `eflm_rescale_auto_sudoku` naive arm, CE, LR 5e-4, seeds 1–3 (45.1 ± 1.3; its own 3-LR sweep); same training code since 2026-07-27 |
| MDLM | tiny DiT, log-linear, its own NELBO, 180 ancestral steps | trained here |

The reused controls are linked into `outputs/elfm_hbfm_ada_curv_elbo_sudoku/<cell>/` (checkpoints
as symlinks, recorded `results.json` copied) and get the same likelihood passes as every new cell; the
smoke stage re-runs the board eval of one of them with the current sampler as a check.

Stages:

- **smoke** — 300 steps + evals for HBFM-ELBO, EFLM-ELBO, MDLM, plus the sampler check: speed, memory,
  the `ppl_eval` pass on Sudoku.
- **1 (seed 1)** — LR ∈ {3e-4, 5e-4, 1e-3} for HBFM-ELBO, EFLM-ELBO and MDLM; LR ∈ {5e-4, 1e-3} for
  HBFM-CE (3e-4 is the reused anchor), so that both losses of the HBFM arm see the same LR grid; the
  six reused controls (eval only). 17 jobs.
- **2** — seeds 2 and 3 at the LR picked per (arm, loss) from stage 1 (accuracy; NELBO breaks ties).

Likelihood: `mode=ppl_eval` on the 2000 validation boards, four passes with different seeds (one
(t, noise) draw per board and pass), reported as the mean over passes in nats per solution token
(89 tokens: 81 cells + 8 separators), masked like the training loss. Every HBFM/EFLM checkpoint is
scored under both objectives (`elbo` = the bound; `ce` = the weighted CE integral it trained on, not a
bound); MDLM under its own NELBO. All three bounds are on −log p(solution | clues) per solution token
and are directly comparable; each carries the same `eps = 1e-3` truncation of the schedule at both
ends (HBFM: at most 4·10⁻³ nats at the origin end, `math.md` §2).

Fairness: same data, seeds, model width/depth, steps, batch, optimizer, EMA, gradient clip, eval
protocol and GPU type for all cells; the LR is tuned per loss on the same grid; CE controls are
same-code checkpoints. Differences that remain: the parameter counts differ by the embedding /
input-projection sizes (HBFM's 9-dim state, EFLM's 512-dim embeddings, MDLM's 13-token vocabulary),
and the ELBO arms use 64 × 4 micro-batches instead of 128 × 2 (same optimizer batch) to fit the second
forward pass into 11 GB.

## Hypotheses

- **H1 (accuracy).** The ELBO arm ties its CE arm within the seed spread (~5 points on this task) for
  both geometries. Both losses share the Bayes-optimal minimizer (posterior matching); they differ in
  how the posterior error is weighted — in embedding/path-KL geometry rather than in log-probability.
- **H2 (bound).** The ELBO-trained models reach a lower NELBO than the CE-trained ones scored under the
  same bound: the CE arms were never trained on the terms the bound weights most (the reconstruction
  posterior at the decode point for EFLM; the path-KL weighting for HBFM).
- **H3 (geometries vs MDLM).** HBFM's bound is far below MDLM's (its states carry the word's identity
  continuously and the trained model's held-out weighted CE was ~0.01 nats/token in
  `hbfm_rlf_ada_curv`); EFLM's sits above HBFM's because its decode point truncates the flow at a
  noise fraction of 0.61 where the posterior is still uncertain. Accuracy need not follow the bound
  (an exact match of 81 cells by a 180-step sampler).

Outcome (`RESULTS.md` §4–5, three seeds): H1 holds for HBFM (56.5 ± 6.4 % vs 59.1 ± 7.5 %) and not
for EFLM (38.7 ± 11.5 % vs 45.1 ± 1.6 %, one seed at 26 %); H2 holds for HBFM (NELBO 0.0172 ± 0.0011
vs 0.0211 ± 0.0037) and not for EFLM (0.0231 ± 0.0061 vs 0.0209 ± 0.0012); H3 is refuted in its
strong form — HBFM's bound equals MDLM's within noise (0.0172 vs 0.0166 ± 0.0049) while HBFM solves
21 points more boards (t = 3.0), and EFLM's bound sits above HBFM's as predicted. Seed variance is the
first-order effect at this budget; none of the ELBO-vs-CE differences is significant at three seeds.

## Add-on: TinyStories bounds of the best pre-trained models

Asked for on 2026-10-10: the NELBO of the best pre-trained EFLM, adaptive-curvature HBFM and MDLM
on TinyStories (seq 256), evaluated with the same `ppl_eval` pass on the full validation set
(`tinystories_elbo.py`; nothing is trained). "Best" = the headline cell of each earlier study:

| arm | checkpoint(s) | why |
|---|---|---|
| HBFM, ada curv b = 4 | `hbfm_rlf_ada_curv_tinystories_256`: trunc_exp, rate 3, ub = t_max = 1, LR 3e-4, seed 1 | the study's best GenPPL (11.9, 3 seeds); seed 1 is the only checkpoint on this file system (seeds 2-3 sit on the A100 node's scratch) |
| EFLM | `eflm_rescale_auto_trunc_tinystories_256`: R = 0.5, LR 1e-3, tau_max = tau*(0.5) = 2.44, seeds 1-3 | the study's recommendation (GenPPL 11.44 +- 0.54, stable) |
| MDLM | `naive_ar_tinystories_s256`: LR 1e-3, seeds 1-3 | its best bound (val/ppl 4.87) |

HBFM and EFLM are also scored under the CE objective they were trained on (the `val/ppl` the earlier
studies report), which is not a bound. The HBFM path-KL at a 50k vocabulary builds (B, L, V, 3)
float64 tensors per factor (16 factors), hence an A6000 (48 GB) and batch 16.

## Compute

`desa`, 2080 Ti only (`desa-compute-01`, 8 GPUs, 11 GB; 7 free, the other nodes being fully
allocated by another sweep of the same account). The cells run inside **pilot jobs**
(`sweep.py --pilot N`, `pilot.py`): one SLURM job of N GPUs whose slots pull the cells one after
another. A single-GPU job of this account was not scheduled at all while that other sweep held
~100 pending jobs (its priority sits below the scheduler's attempt depth; `RESULTS.md` §6), whereas
a job asking for several GPUs gets the TRES priority to start at once on the idle cards. SLURM's
books show only 56 GB of the node's memory as free (another allocation holds 195 GB, 160 GB are
actually available), so a slot is budgeted at 9 GB (measured peak 7.5 GB per cell) and a pilot has
at most 6 slots.

Measured in the smoke stage (300 steps, 4 cells sharing the node):

| cell | micro-batch | speed | per optimizer step (256) | GPU memory | 20k steps |
|---|---|---|---|---|---|
| HBFM-ELBO, b = 4 | 64 × 4 | 1.11 it/s | 3.6 s | 9.1 GB | ~20 h |
| HBFM-CE, b = 4 (earlier study) | 128 × 2 | 1.02 it/s | 1.96 s | 9.2 GB | ~11 h |
| EFLM-ELBO | 64 × 4 | 1.32 it/s | 3.0 s | 9.1 GB | ~17 h |
| MDLM | 128 × 2 | 1.49 it/s | 1.34 s | 9.1 GB (cached) | ~7.5 h |

The ELBO arms cost ~2× their CE arm per sample: the reconstruction term is a second forward pass
with gradients, and the halved micro-batch is less efficient on this card. Stage 1 (11 training
cells + 6 evaluations, ≈ 170 GPU-h) runs on a 6-slot pilot in ≈ 29 h; stage 2 adds ≈ 24 h.

As run: stage 1 = pilot job 156442 (6 slots, 2026-10-10 01:54 → 2026-10-11 morning); a cell took,
evaluations included, 18.4 h (HBFM-ELBO), 16.2 h (EFLM-ELBO), ≈ 8 h (MDLM) — `sweep.py`'s `HOURS`
budgets were raised to 36 / 22 / 12 h accordingly. Stage 2 went in three 2-slot pilots as soon as each
arm's LR was known: job 183756 (HBFM-ELBO seeds 2–3 + the test suite, started 2026-10-10 21:00 on
the two cards another user had just released), job 193573 (MDLM seeds 2–3) and job 195710
(EFLM-ELBO seeds 2–3), both started 2026-10-11 06:23 within five minutes of the stage-1 pilot's end.
The diagnostics of `RESULTS.md` §3 (`grad_snr.py`, `hbfm_rlf_ada_curv/curvature_stats.py`, and
the re-decodings of two seed-1 checkpoints under other sampler settings — `logs/steps_diag.sbatch`,
`logs/temp_diag.sbatch`, using the `TOPK_VELOCITY` / `TEMPERATURE` knobs added to
`scripts/sample/sudoku/hbfm.sh`, whose defaults are the protocol's) ran as five single-GPU jobs of
2–15 min each on an idle RTX 6000 Ada of `thickstun`; nothing was trained there.
