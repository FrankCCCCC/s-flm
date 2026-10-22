# elbo_tune_tinystories_256 — can EFLM and HBFM reach MDLM's likelihood bound on TinyStories?

Request (user, 2026-10-12): tune the hyperparameters of HBFM and EFLM to reach MDLM's ELBO on
TinyStories under a fair protocol; the question is whether the two algorithms are worse than MDLM.
Sweep: `sweep.py` (orchestration only, three sites); table: `report.py` → `RESULTS.md`. Losses:
`algo.loss_type=elbo` of `experiments/elfm_hbfm_ada_curv_elbo_sudoku/math.md`.

## Starting point

The bounds of the best pre-trained TinyStories models of each earlier study
(`elfm_hbfm_ada_curv_elbo_sudoku/RESULTS.md` §6; nats per token, full validation set):

| model | trained on | NELBO | of which endpoints | diffusion term |
|---|---|---:|---:|---:|
| MDLM, LR 1e-3, 30k steps (`naive_ar_tinystories_s256`) | its NELBO | **1.582 ± 0.007** (3 seeds) | — | ≈ 1.60 |
| EFLM, R 0.5, τ* 2.44, LR 1e-3 (`eflm_rescale_auto_trunc_tinystories_256`) | CE | 1.893 ± 0.004 | 0.088 | 1.805 |
| HBFM, cb 4, trunc_exp ub 1, LR 3e-4 (`hbfm_rlf_ada_curv_tinystories_256`) | CE | 2.104 (seed 1) | 0.431 | 1.673 |

Those models were trained on the denoising CE and tuned for GenPPL, not for the bound. MDLM trains
its own NELBO. So the first fairness gap is the objective, and the gaps have identifiable sources:
HBFM pays 0.43 nats in the reconstruction at its horizon (heat-time range stopped at ub = 1 where
16 H³ factors have not resolved the token); EFLM pays two thirds of its diffusion term in the band
of noise fractions 0.14–0.5 and 0.75 nats per unit t at the clean end, where its CE-trained posterior
is 1 % off in squared error under a weight of 295. Plaid (`papers/Likelihood-Based Diffusion
Language Models.pdf`), the likelihood-trained continuous diffusion LM this EFLM resembles, gets its
likelihoods from training the VDM bound itself with a learned (variance-minimising) noise schedule,
learned embeddings of dimension 16 ("sensitive to the embedding dimension, small values best"),
tuned schedule endpoints and self-conditioning. Of these the code has the bound, the learned
schedule (`noise=*-adaptive`) and the endpoints (R, τ_max); it has no low-dimensional embedding for
the sphere DiT (the embedding is the DiT width, 768), and self-conditioning triples the forward
passes — both are outside this sweep and noted as the next levers.

## Protocol (fair to MDLM)

Everything MDLM's reference run had, for every cell: TinyStories (GPT-2 tokenizer, V = 50257), seq
256, small DiT 768 / 12 / 12 (`small`, `small-sphere-dit`, `small-hyperbolic-dit`), ngpt init,
global batch 512, bf16, EMA 0.9999, AdamW wd 0, clip 1.0, the same data order per seed, and the
full 30k steps for the final comparison. Each arm trains its own NELBO (`loss_type=elbo`) — MDLM
its masked-diffusion NELBO, EFLM the VDM bound, HBFM the path-KL bound — and is scored by the same
`mode=ppl_eval` pass over the full validation set (1172 × 16 sequences, one (t, noise) draw per
sequence per pass). The hyperparameter grids are method-specific but of the same size class as
MDLM's LR grid in its study ({3e-4, 1e-3, 5e-3}): one axis at a time around each arm's incumbent,
seed 1, ranked at a third of the budget (10k steps), then the pick trained at the full budget with
3 seeds. Differences in compute per step are reported, not equalised: the ELBO arms run a second
forward pass (the reconstruction term) per sample, MDLM one.

## Stage 1 — ranking at 10k steps (seed 1)

| id | arm | cell (one change from the incumbent) | why |
|---|---|---|---|
| E0 | EFLM | R 0.5, τ_max = τ*(0.5) = 2.44, autonomous clock, LR 1e-3, **ELBO** | incumbent, now trained on the bound |
| E1 | EFLM | LR 3e-4 | the Sudoku ELBO arm preferred the smallest LR |
| E2 | EFLM | R 2, τ_max = τ*(2) = 1.29 | more signal per noise; the horizon follows τ*(R) |
| E3 | EFLM | R 8, τ_max = τ*(8) = 0.51 | the Sudoku setting (where ELBO training helped) |
| E4 | EFLM | τ_max 4 | a cleaner clean end (noise-fraction floor e⁻⁴ instead of e⁻²·⁴⁴): the reconstruction and the clean-end integrand |
| E5 | EFLM | noise log-linear (α_t = 1 − t), no truncation | MDLM's clock; the untruncated bound |
| E6 | EFLM | noise autonomous-adaptive | Plaid / VDM's learned variance-minimising schedule |
| E7 | EFLM | CE-trained | the control: what ELBO training buys at equal budget |
| H0 | HBFM | cb 4, trunc_exp rate 3 ub 1, (H³)¹⁶ at K −0.5, naive readout, LR 3e-4, **ELBO** | incumbent, now trained on the bound |
| H1 | HBFM | LR 1e-3 | the Sudoku ELBO arm preferred the largest LR |
| H2 | HBFM | ub 2 (sampler horizon 2) | halve the reconstruction at the horizon |
| H3 | HBFM | horosphere readout | the Bayes-structured posterior (logits = residual on the Busemann log-densities) |
| H4 | HBFM | cb 1 (fixed curvature) | on unconditional text the adaptive λ is one global clock riding its guard |
| H5 | HBFM | embed_dim 96 = (H³)³² | twice the evidence per unit heat time |
| H6 | HBFM | CE-trained | the control |
| M0 | MDLM | LR 1e-3 | the reference at the same budget |

Decision rule: per arm, the ELBO cell with the lowest stage-1 NELBO; ties (< 0.01) broken toward
the incumbent. A pick that beats the incumbent on one axis is not combined with another axis
unless a second stage-1 round is run (noted in RESULTS.md).

## Stage 2 — the picks at MDLM's budget

The picked EFLM and HBFM cells for 30k steps, seeds 1–3, two NELBO passes per seed; MDLM's three
30k seeds are the existing reference (re-scored under this code: 1.582 ± 0.007). The answer is the
three-seed NELBO of each arm against 1.582, with the stage-1 CE controls and the earlier studies'
CE-trained models as context.

## Hypotheses

- **H1.** ELBO training alone closes a large part of each gap: HBFM's reconstruction term (0.43) and
  EFLM's clean-end integrand are what the CE never trained.
- **H2.** HBFM's remaining gap is the horizon: ub 2 or (H³)³² lowers the bound; the horosphere readout
  helps at the horizon.
- **H3.** EFLM does not reach MDLM without Plaid's low-dimensional embedding: the band
  b ∈ [0.14, 0.5] is where a 768-dim Gaussian channel with norm-pinned embeddings loses to masking.
- **H4.** MDLM's bound at 10k steps is ≈ 1.65–1.70; a continuous arm within 0.05 of it at 10k is
  within reach at 30k.

## Compute

Per cell, 10k steps (stage 1): ELBO cells ≈ 2× a CE cell. Measured CE costs (30k steps):
A100 1 GPU × 64: 16 h; 4 × A6000/Ada × 16: 4–5 h; 8 × 2080 Ti × 16: 28 h. So an ELBO stage-1 cell
≈ 11 A100-h, ≈ 3 h on four 48 GB cards, ≈ 19 h on the whole 2080 Ti node; plus the NELBO pass over
the validation set (0.3–1 h). Stage 1 (16 cells) ≈ 150 A100-equivalent GPU-h; stage 2 (6 cells at
30k) ≈ 200. Sites: `a100` (ch2263, `nlplarge`, preemptible with `--requeue`, 1 GPU per cell, code
copy `/home/ch2263/syc_workspace/s-flm-dev2-elbo`), `a6000` (sc3379, `desa,thickstun` without the
2080 Ti node, 4 GPUs per cell), `2080ti` (sc3379, `desa-compute-01`, 8 GPUs per cell, micro-batch
8 for the ELBO arms). At submission (2026-10-12 00:30) every A6000 / A5000 / Ada / A100 was in use
and the 2080 Ti node idle, so the wall clock is set by the queues. No `nice` is set by the sweep.

Smoke findings (2026-10-12 00:35–01:20, `outputs/.../smoke/`, 10–30 steps + the full NELBO pass):
EFLM-ELBO trains at micro-batch 8 on the 11 GB cards (1.4 it/s on 8 of them, ≈ 5 s per optimizer
step → 10k steps ≈ 15 h) and MDLM at 8; the NELBO pass over the validation set takes 6–13 min on
a 2080 Ti. HBFM-ELBO is another matter: the path-KL rate forms per-word, per-factor tensors, and
at V = 50257 the explicit form of `Loss.bridge_loss_elbo_refactor` needed 1.2 GB per factor per
4 sequences (16 factors: out of memory at micro-batch 4). It was rewritten into the algebraically
identical form `θ Σ_v μ_v (radial_v − ⟨θ, φ_v⟩)/D_v + (μ/D) @ φ` with per-factor recomputation
in the backward pass (`tests/test_elbo_loss.py`: value and gradient equal to the explicit form to
1e-10); that fits, but the float64 (B, L, V) work of 16 factors runs at 0.2 it/s at micro-batch
4 on a 2080 Ti (1/32-rate float64) and still exceeded 11 GB after 65 iterations. HBFM-ELBO cells
therefore run on the A100s only (full-rate float64, 80 GB); its grid is fixed after the A100 smoke
measures the step time.

A100 measurement (06:12–07:00, smoke job 216223 and `bench_rate.py` inside it): the exact rate is
GPU-bound — ~1,700 float64 elementwise kernels over (B, L, V) per micro-batch, 2.5 s forward +
backward at B = 16 (20 GB), 0.35 s forward-only at B = 4 → a 512-sequence optimizer step costs
≈ 80 s and 10k steps ≈ 9.5 A100-days per cell, 40× the CE arm; the exact bound over the
validation set costs ≈ 30 min (fine for evaluation). `torch.compile` fuses the chain 1.5× but
undoes the per-factor recomputation (25 GB at B = 4) — not adopted. The cost is linear in
the number of factors, so the user's decision (2026-10-12 11:30) is to run HBFM on TinyStories on
the **(H³)³ product manifold** (embed_dim 9, the geometry of the Sudoku ELBO study) with the exact
rate: 3 factors instead of 16 cut the rate to ≈ 0.5 s per micro-batch of 16 on an A100, i.e.
≈ 15 s per optimizer step on one A100 or ≈ 8 s on four Ada GPUs (10k steps ≈ 1–2 days per
cell). A Monte-Carlo estimate of the rate over sampled words had been implemented in the
meantime; the user rejected it as a method and it was removed from the code. The (H³)¹⁶ cells
H0–H6 that had started on it are discarded (the queued ones were held; the running ones are the
user's to cancel). The memory-light exact form stays (`tests/test_elbo_loss.py` checks it against
the explicit form in value and gradient to 1e-10).

(H³)³ grid, stage 1 (seed 1, 10k steps, full validation set): G0 = cb 4, exp proposal of rate 3,
sampler / reconstruction horizon T_MAX 6 (`configs/algo/hbfm.yaml`: (H³)³ resolves a word by unit
time ≈ 3/|K| = 6 at K = −0.5; the Sudoku ELBO arm's setting), horosphere readout in float64, LR
1e-3 (the Sudoku ELBO pick), ELBO-trained; G1 = LR 3e-4; G2 = naive readout; G3 = trunc_exp
truncated at the horizon (ub 6); G4 = fixed curvature; G5 = CE control on the same geometry.

Submission state (2026-10-12 06:50): E0 on the 2080 Ti node, E5 and E6 on four A5000s, E7 done
(CE control: NELBO 2.089); remote (`a100`): E1 and M0 running, E2 / E3 / E4 queued behind the
account's CPU cap. The (H³)³ HBFM cells follow their smoke run.

(H³)³ smoke (2026-10-12 11:40–12:15, 10 steps + the exact bound on the full validation set): the
horosphere cell with the float64 readout trains at 0.63 it/s on four A5000s at micro-batch 8
(≈ 25 s per optimizer step), the naive-readout cell at 1.5 it/s on four Ada GPUs (≈ 11 s per
step); the exact evaluation takes 19 min on an A5000. A 10k-step cell is therefore 1.5–3 days on
four 48 GB cards, and 30k steps three times that. The six (H³)³ cells G0–G5 were submitted on the
`a6000` site at 12:14 (jobs 223190–223195); the (H³)¹⁶ cells 220013–220019 were cancelled with
the user's approval at 12:05. MDLM's 10k-step reference landed at NELBO 1.760 (its 30k value is
1.582), the yardstick for the stage-1 ranking.

Radii actually visited (`radius_probe.py`, 4096 bridges per heat time, (H³)³ at K = −0.5, R = √2,
the code's float64 state; user question of 2026-10-12): the exp proposal of rate 3 with ε = 10⁻³
draws heat times in [0, 2.30] per position (× λ ∈ [1/4, 4] under the adaptive curvature); the
sampler runs to T_MAX = 6.

| heat time | where it occurs | ρ p50 / p99 / max | u = κρ max | min 1 − ‖z‖²/R² |
|---|---|---|---|---|
| 2.30 | end of the training range (and the reconstruction horizon) | 2.8 / 5.9 / 7.9 | 5.6 | 1.5e-2 |
| 6.0 | sampler horizon | 5.5 / 10.7 / 14.1 | 10.0 | 1.9e-4 |
| 9.21 | training range × λ = 4 | 7.9 / 14.5 / 19.5 | 13.8 | 4.2e-6 |
| 24.0 | sampler horizon × λ = 4 | 18.5 / 29.5 / 38.3 | 27.1 | 7.0e-12 |

None of these reaches the Cartesian → polar cap (‖z‖/R ≤ 1 − 8ε₆₄, i.e. u ≤ 34.7); float32 would
saturate a third of the states at heat time 24 (its cap is u ≈ 14.6), which is why the state and
the rate are float64. The bound itself is not computed from Cartesian quantities: the state is
converted to (ρ, θ) once and every term of the rate and of the horosphere readout is a bounded
ratio of the e^{±u} split with no 1 − ‖z‖² denominator. The one numerical difference of the
memory-light rate is sin²(angle/2) = (1 − ⟨θ, φ_v⟩)/2 instead of ‖θ − φ_v‖²/4; for the target
word (the only one that gets aligned with θ) that form is accurate to 1.4e-10 at heat time 2.30,
2.2e-7 at 6.0 and 9.4e-6 at 9.21, and fails only at 24 (sin² ≈ 1e-22, below float64 cancellation),
a regime training and evaluation never enter and the sampler's readout, which keeps the
difference form, handles.

## Closed EFLM axes (stage 1)

- E5 (MDLM's log-linear clock, untruncated): training loss 3.6 but validation NELBO 136. The VDM
  weight −SNR′/2 ∝ 1/b³ reaches 10⁹ at the clean end b = ε = 10⁻³ and a posterior mean that is not
  exactly the token's embedding there makes the integrand heavy-tailed: the Euclidean bound needs a
  truncated (or learned) clean end — Plaid's endpoints — which the autonomous clock's τ_max gives.
- E4 (τ_max 4 at R 0.5, a cleaner clean end, floor b = 0.018): NELBO 2.240 vs 1.914 at τ*(0.5):
  extending the integral into the high-weight clean end costs more than the reconstruction it saves.
- E6 (the repo's adaptive clock, `criterion=slope`): logged training loss 0.66, NELBO 2.58 under
  its own clock and 2.55 under the plain one (job 222506). Probe of the learned clock
  (`adaptive_clock_probe.py`, 2026-10-13): it sent 85 % of the training samples to the clean end
  b < 0.14 (plain clock: 20 %) and 9 % to the band b ∈ [0.14, 0.5] (plain: 52 %) where EFLM's
  integrand peaks, with |db/dt| up to 108 (plain ≤ 2.4); its last loss buffer has median 0.70 but
  mean 2.91, with samples up to 459. The estimator stayed unbiased (the buffer mean matches the scored
  bound; the logged per-micro-batch loss is the median of a heavy-tailed draw), but its variance
  exploded and the model trained worse than on the plain clock. Cause: importance max(dL/dt, 0) is
  built for a denoising CE that grows with the noise, while the ELBO integrand decreases over the
  noisy half, which therefore got almost no mass; the profile measured under the current clock is
  also mapped onto the base clock (the alternation the config warns about at EMA 0).

## Added arm: the VDM clock (2026-10-13) — cancelled and rolled back

At the user's suggestion (Plaid / RePlaid: the remaining difference between EFLM and Plaid is VDM's
learned schedule) a variance-minimising criterion was added to `noise_schedules.AdaptiveSchedule`
(noise level drawn with density ∝ the RMS of the bound's integrand; endpoints, hence the bound,
unchanged) and run on the EFLM pick (R 2, τ*(2), LR 1e-3; jobs 246612 and 247218 died on the 2080 Ti
node before the clock activated, 247716 trained to step ~6k on four A6000s). The clock it learned by
step 2000 was healthy and close to the plain autonomous clock (the integrand varies only ~3× over the
truncated range). On 2026-10-13 16:10 the user asked to cancel the VDM jobs and roll back the VDM
code: job 247716 was cancelled before evaluation (no result), and the criterion, its config keys
and tests, the EFLM scripts' `NOISE_ARGS` knob and the sweep's `vdm` stage were removed;
`noise_schedules.py` is back to its original form (`tests/test_elbo_loss.py`: 10 pass). Its partial
outputs stay under `outputs/elbo_tune_tinystories_256/eflm_..._noise-autonomous-vdm_..._seed-1/`.
