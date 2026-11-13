# hbfm_ce_tinystories_256 — experiment design

HBFM (16 H³ factors, `EMBED_DIM=48`, small DiT 768/12/12, naive float32 readout) on TinyStories (seq 256), trained on the
denoising CE (`LOSS_TYPE=ce`) and ranked by the exact validation NELBO. Spec: `setup.md`. Target: **NELBO < 1.70**.
Reference: `hbfm_elbo_tinystories_256` trained on the NELBO itself and reached 1.683.

## Questions (setup.md GOAL)

1. Which hyperparameters give a CE-trained model the lowest validation NELBO, and is it below 1.70?
2. Do `sampling_eps` and `T_MAX` change the expectation and the variance of the NELBO?
3. Which gives the lower final NELBO, an infinite or a finite horizon, and how long is the best horizon?

## How the knobs act under the CE

Read from the code (`algo.py`, `trainer_base.py`, `noise_schedules.py`); the runs below measure the size of the effects.

- `T_MAX` is `sampler.t_max`. The NELBO never reads it, so it cannot move the NELBO's mean or variance. It moves the
  samples (GenPPL). `trunc_exp` / `unif` require `T_MAX` ≤ ub. Cells use `T_MAX` = their horizon, as the ELBO sweep did.
- `sampling_eps` draws t ~ U[eps, 1], so under `exp` it sets the horizon T = ln(1/eps)/rate: 3.84 / 2.30 / 1.54 at
  1e-5 / 1e-3 / 1e-2, rate 3. Under `trunc_exp` the horizon is ub, and eps only trims the last eps of the draw.
- `exp` at eps is the same proposal as `trunc_exp` at ub = ln(1/eps)/rate, so one ub axis and three `exp` cells cover the
  horizon. "Infinite horizon" here means `exp` with a small eps; no finite-precision run reaches T = ∞.
- **Bias at large eps.** t is drawn on [eps, 1], but the weight omits the factor (1 − eps). The integral, and so the
  path-KL term, is overstated by 1/(1 − eps): 0.1% at 1e-3 (every number in `hbfm_elbo_tinystories_256`), 1% at 1e-2,
  11% at 0.1. Left unchanged so NELBOs stay comparable with that sweep. `horizon.py` reports the corrected value.
- `PROPOSAL_RATE` changes the variance of the importance-weighted estimate, not the integral the CE trains on.
- `ELBO_NUM_WORDS` is **not a knob here**: it only applies to `loss_type=elbo`. Evaluation always uses the exact word sum.

## Cells

Base = the best recipe of the ELBO sweep: `trunc_exp`, ub 0.6, rate 3, K −0.5, fix (`CURV_BOUND=1`), GPT-2 XL, dropout 0,
peak LR 2e-3 held to 14k then cosine to 0 at 30k, eps 1e-3. Each tuning cell changes one knob from it.

| axis | cells |
|---|---|
| horizon | `ub0.4`, `base` (0.6), `ub0.8`, `ub1`, `ub1.5`, `exp_eps1e-2` (T 1.54), `exp_eps1e-3` (T 2.30), `exp_eps1e-5` (T 3.84) |
| proposal | `rate1.5`, `rate6`, `unif` |
| curvature | `K0.25` (rate 1.5, ub 1.2), `K1` (rate 6, ub 0.3): the same unit proposal, only the state scale moves; `ada` (`CURV_BOUND=4`) |
| optimizer, regularizer, embedding | `lr1e-3`, `lr3e-3`, `do0.1`, `bert` |
| setup.md's grid, at its literal recipe (LR 3e-4, constant, dropout 0.1, `exp`, eps 1e-3) | `grid_r3_K0.5_fix`, `grid_r3_K0.5_ada`, `grid_r1.5_K0.5_ada`, `grid_r5_K0.5_ada`, `grid_r6_K1_ada` |

Wave 2, added 2026-11-07 03:20 from the first 6k scores (ub 0.4 2.150, 0.6 2.185, 0.8 2.185):

| axis | cells |
|---|---|
| shorter horizons | `ub0.3`, `ub0.2` |
| CE without its importance weight (`CE_TIME_WEIGHTED=false`) | `now` (rate 3, ub 0.6), `now_r6` (rate 6, ub 0.6) |

The importance weight makes the CE a uniform integral over t, while the NELBO's path-KL rate falls about 5.6× over [0, 0.5].
Without the weight, the CE is averaged under the proposal itself, so training concentrates on early t and the proposal rate
becomes a real training knob. This objective is new to this sweep (`algo.ce_time_weighted`, default true, read from the env
through `configs/algo/hbfm.yaml` so `hbfm.sh` need not change under running jobs). Evaluation is unchanged.

Follow-ups on `ub0.4`'s lead: `ub0.4_lr3e-3`, `ub0.4_wsd20k` (peak held to 20k), `ub0.4_bert`, and, after `ub0.4_bert` diverged at
peak LR 2e-3, `ub0.4_bert_lr1e-3`.

Wave 5, added 2026-11-07 18:10 at the user's request: is "the short horizon is better" robust? At 30k, `exp` T 2.30 scored
1.936 against ub 0.4's 1.845, but `exp` trained on base's recipe, with importance weights up to 333 at peak LR 2e-3, and
spiked at 6k.

| question | cells (all `exp` unless noted) |
|---|---|
| Does a lower LR rescue `exp`? (T 2.30) | `exp_lr1e-3`, `exp_lr6e-4`, `exp_lr3e-4` |
| Does removing the importance weights rescue it? | `exp_now` (T 2.30, LR 2e-3) |
| Even longer horizons, without the weights | `exp_eps1e-5_now` (T 3.84), `exp_eps1e-7_now` (T 5.37); the weighted `exp_eps1e-5` already runs |
| Finite counterparts with the same recipe change | `ub0.4_now`, `ub0.4_lr1e-3` |

- 2026-11-08 01:40: cancelled `exp_lr1e-3` after 10k (2.167). It was 2.4% behind `exp_now` (2.116), and 5.1% behind the
  finite horizon at the same LR (`lr1e-3`, 2.061), so the LR-1e-3 comparison is already settled. Its slot went to
  `ub0.4_bert_lr1e-3`.
- 2026-11-08 04:00: cancelled `exp_lr3e-4` after 6k (2.368, the worst `exp` variant: +6% vs `exp_now`). The weighted `exp`
  gets monotonically worse as the LR drops (1e-3 / 6e-4 / 3e-4 → 2.310 / 2.326 / 2.368), and LR 3e-4 also finished behind the
  higher LRs at 30k in the ELBO sweep.
- 2026-11-08 07:00: cancelled `exp_eps1e-5_now` after 10k. Unweighted `exp` at T 2.30 / 3.84 / 5.37 scores 2.116 / 2.108 / 2.110,
  a plateau, so the middle point is redundant. `exp_now` (paired with `ub0.4_now`) and `exp_eps1e-7_now` (the longest) go on to 30k.
- 2026-11-08 04:45: cancelled `exp_lr6e-4` after 10k (2.176: +2.8% vs `exp_now`, level with `exp_lr1e-3` at 2.167).

All run to 30k with no gate (`--finish`), because an `exp` cell can spike at 6k and recover (`exp_eps1e-3`: 4.015 at 6k,
2.193 at 10k). From their 10k scores on, variants that clearly lose to a sibling are cancelled. At eps 1e-7 the sampler's
step (T/180 = 0.03) is coarser than the guidance (0.02 at K −0.5), so that cell's GenPPL is biased. Its NELBO is not affected.

LR schedule shape is held out of the first wave: every first-wave cell at peak LR 2e-3 shares one LR trajectory, and
schedule effects only show at 30k (`hbfm_elbo` waves 30–44).

## Early stopping

One job per cell, which stops itself (`gate.py`; no `scancel`): train to 6k, score, gate; train to 10k, score, gate; then
30k and the full evals. A stage score is the exact NELBO on 10% of validation (the same sequences and t draws for every
cell). The bars live in `bars.json`, which jobs read at each gate.

- Initial bars: setup.md's 3.0 at 6k and 2.6 at 10k.
- Tightened 2026-11-07 03:25, from the first 6k scores, to **2.6 at 6k and 2.1 at 10k**, because setup.md's bars alone would
  stop almost nothing. 6k stays loose: at peak LR 2e-3, transient loss spikes close to the checkpoint make single 6k scores
  noisy (`ub1` 2.622 next to `ub0.8` 2.185). A cell at 2.6 would still be near 2.2 at 30k on the ELBO sweep's curve.
  The 10k bar is about 3.5% above where the leaders should land (6k × 0.945, the ELBO sweep's ratio).
- 2026-11-07 10:00, the user: test every queued cell at 6k and 10k rather than cancel the ones the main effects seem to
  answer, because the interactions between the knobs are uncertain. So every queued cell runs through both gates, and goes on
  to 30k only if it beats `base` at 10k.
- 2026-11-07 10:20, the user allowed cancelling this sweep's own jobs to speed up the search. Cancelled `now` and `ub0.3`
  after 10k (2.067 / 2.062, both at `base`'s level; the 2.06 bar would have stopped them). Moved `now_r6` and `ub0.2` from the
  A100 queue to the 2 GPUs per desa node they freed (`desa2` layout). Also cancelled `ub1` at about 21.1k (10k: 2.090, level
  with `ub0.8`, which stays as the long-horizon point at 30k) and `rate1.5` at about 23.4k (10k: 2.062; under the weighted CE the
  rate only changes the variance), so `lr3e-3` (A100) and `bert` (desa) start about 5.5 h earlier. `lr1e-3` keeps running:
  LR effects reversed between 10k and 30k in the ELBO sweep.
- **2026-11-07 11:12, the A100 site was lost.** The ch2263 account itself cancelled all 12 remaining hbfm_ce jobs, running and
  queued, and started interactive GPU jobs. Our launch at 23:33 had preempted three of its interactive sessions. With the user's
  approval:
  - One CPU-only job copied the four interrupted 30k runs' `last.ckpt` (step 23000: `ub0.4`, `base`, `ub0.8`, `exp_eps1e-3`)
    off the node's `/scratch`, rewriting the ModelCheckpoint paths to the sc3379 run dirs so the resume is a same-dir resume.
  - `grid_r6_K1_ada` was cancelled after its 6k score (2.583, +18% vs `base`): it could not reach the 10k bar, and the slot went
    to two resumes.
  - `ub0.2` was cancelled after its 6k score (2.528, +16% vs `base`; the horizon optimum is near 0.4).
  - The four runs resume on desa (`desa2`, `--finish`: they had passed the gates under the earlier bars).
  - The eight queued A100 cells were resubmitted on desa at `nice` 50. No more GPU jobs go to ch2263.
- Tightened 2026-11-07 09:45 to **2.06 at 10k**, `base`'s own 10k score. Every 10k score so far is between 2.041 and 2.093 except
  the diverged cells, so 2.1 no longer separated anything, and the `ub0.4` follow-ups need the slots.
- `exp_eps1e-3` skips the gate: question 3 needs a 30k `exp` final.
- Training loss is never used to stop a cell (it is the last micro-batch, at fixed time bands).
- A bar compares cells only when they share an LR trajectory up to that step: higher-LR schedules trailed mid-run and won
  at 30k in the ELBO sweep. Cells with a different peak LR are judged against each other only at 30k or by trend.

## Evaluation (the same protocol as `hbfm_elbo_tinystories_256`)

`LOSS_TYPE=elbo`, words summed exactly (`elbo_num_words` null), `SEED=1`, `EVAL_BS=16`, each cell at its own horizon and
proposal, EMA weights, full validation set for the final score. Plus val/ppl and 64 samples at 180 steps, `TOPK_VELOCITY=1`,
greedy last step, `T_MAX` = horizon. Question 2 uses `horizon.py` on one trained checkpoint: 256 sequences × 4 draws per
setting, with the same seeds in every setting.

## Compute

Measured on the running cells (8 micro-batches per step on both layouts):

| site | layout | speed | to 6k | to 30k |
|---|---|---:|---:|---:|
| ch2263, `nlplarge` (A100 80 GB) | 1 GPU × 64 | 4.8 it/s ≈ 1.7 s/step | ≈ 2.8 h | ≈ 14 h |
| sc3379, `desa` (A6000 / A5000) | 4 GPUs × 16 | 5.3 it/s ≈ 1.5 s/step (`ada` 4.6) | ≈ 2.5 h | ≈ 13 h |

Add about 10 min per stage score (10% of validation) and up to 1 h for the final evals. Capacity at submission: 5 A100 cells
and 4 desa cells at once, with 14 queued in order of promise. Both partitions can preempt lower-tier jobs and can be
preempted in turn, so jobs use `--requeue` and resume from `last.ckpt`.

## W&B

Project `hbfm_ce_tinystories_256`, entity `syctw`, one run per cell (run id = tag with `.` → `p`, so stages and requeues
stay in one run), with `val/nelbo` plotted against `trainer/global_step`.

- desa cells log live. Since 2026-11-07 11:12 every cell runs on desa; the four A100 runs resumed there continue the same W&B runs.
- A100 cells log offline and are replayed from sc3379 by `wandb_sync.sbatch` every 10 minutes (`sweep.py --pull`, then
  `--wandb-sync`). This keeps the API key off the ch2263 account, which other people use.
