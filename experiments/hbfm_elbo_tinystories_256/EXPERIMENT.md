# hbfm_elbo_tinystories_256 — experiment design

HBFM (16 H³ factors, `EMBED_DIM=48`, small DiT 768/12/12, naive float32 readout) on TinyStories
(seq 256), trained on the NELBO (`LOSS_TYPE=elbo`) with the path-KL rate's sum over words estimated by
Monte Carlo (`ELBO_NUM_WORDS`). Spec and grid rationale: `setup.md`. Global batch 512, LR 3e-4, 30k
steps, bf16, EMA 0.9999, seed 1. Ranked by the exact NELBO (nats/token) on the full validation set.

## Questions (setup.md GOAL)

1. Which hyperparameters give the lowest validation NELBO: curvature K, `CURV_BOUND`, `PROPOSAL_RATE`,
   `TIME_CONVERSION_MODE` / `TIME_RANGE_UPPER_BOUND`, `ELBO_NUM_WORDS`, and (added 2026-10-30) the
   word-embedding init (GPT-2 XL vs BERT-large-uncased)?
2. Does `TIME_RANGE_UPPER_BOUND` (ub) change the expectation and the variance of the NELBO, and which
   is better for the final validation NELBO: `trunc_exp` (horizon ub) or `exp` (horizon
   T = ln(1/ε)/rate = 2.30 at rate 3)?

## Hypotheses

- H1 (horizon, expectation). The bound is tight at the Bayes posterior for every horizon, but the
  model's posterior is factorized over positions, and the reconstruction CE at the horizon pays for
  that. A shorter horizon (`trunc_exp`, ub < T) gives a higher NELBO for the same model
  (`elbo_tune_tinystories_256`: +0.43 nats at ub 1, λ = 1/4, CE-trained). The adaptive clock can
  stretch the horizon to λ·ub ≤ 4·ub, which may close part of the gap.
- H2 (horizon, variance). The importance weight of `trunc_exp` is at most (1 − e^{−r·ub}) e^{r·ub}/r:
  1.5 at ub 0.5 and 6.4 at ub 1, vs 1/(rε) = 333 for `exp` at rate 3. A shorter horizon gives a
  lower-variance estimate and gradient; whether that buys a lower final NELBO than `exp`'s tighter
  bound is the open question.
- H3 (words). The S-word MC rate is unbiased, so S changes only variance. S = 4 adds noise; S = 64
  removes most of the MC part. If the t-sampling variance dominates, S does not matter.
- H4 (embedding). BERT-large-uncased is mapped onto GPT-2's vocabulary (a word's row = the mean of
  its WordPiece embeddings, `utils.load_pretrained_word_embedding`). The mapping merges case and the
  leading space: 27,148 of 50,257 GPT-2 words share their row with another word (" the" = "The" =
  "the"), and 394 whitespace words (incl. "\n") get [UNK]'s row. The embedding is trainable, so
  merged rows can separate. Expect a slower start than GPT-2 XL.

## Cells

Wave 1 (running since 2026-10-29/30, run to 30k):

| cell | rate | K | arm | site |
|---|---:|---:|---|---|
| r3_K0.5 (anchor) | 3 | −0.5 | ada, fix | ada: 1× A100 (ch2263); fix: 4× A5000 (desa) |
| r1.5_K0.5 | 1.5 | −0.5 | ada | 1× A100 |
| r5_K0.5 | 5 | −0.5 | ada | 1× A100 |
| r6_K1 | 6 | −1 | ada | 4× A5000 |

Wave 2 (ada, `CURV_BOUND=4`): each cell changes one knob of the anchor (rate 3, K −0.5, `exp`, S = 16,
GPT-2 XL). All run on desa, 4 GPUs × micro-batch 16.

| cell | change | horizon (λ = 1) | question |
|---|---|---:|---|
| r3_K0.5_ub0.5 | `trunc_exp`, ub 0.5, `T_MAX` 0.5 | 0.5 | 2 |
| r3_K0.5_ub1 | `trunc_exp`, ub 1, `T_MAX` 1 | 1.0 | 2 |
| r3_K0.5_S4 | `ELBO_NUM_WORDS=4` | 2.30 | 1 |
| r3_K0.5_S64 | `ELBO_NUM_WORDS=64` | 2.30 | 1 |
| r3_K0.5_bert | `PRETRAINED_EMBED=google-bert/bert-large-uncased` | 2.30 | 1 |

The anchor itself (`exp`, ub = T) is the third point of the ub axis.

Changes made on 2026-10-30 at 20:40, from the first 10%-validation scores at 10k:
- The fix arm led both ada cells by 4.2%: 2.457 vs 2.561 (r3_K0.5 ada, A100) and 2.562 (r6_K1 ada,
  desa). The two ada cells have the same unit proposal and different hardware, and they agree to
  0.001, so the gap is the curvature, not the site. Every wave-2 cell above is ada, so the cell
  **r3_K0.5_ub1 / fix** (`trunc_exp` ub 1, λ = 1) was added. It answers the trunc_exp-vs-exp
  question on the arm that leads.
- `nelbo_variance.py` on the fix model at 20k: the S-word MC adds about 7% to the estimator
  variance at S = 16 (std 0.314 vs 0.303 exact) and about 14% at S = 4, and leaves the mean unchanged.
  The S4 and S64 cells were given `nice=500`, so they run only after bert and the fix ub1 cell.

## Early stopping (set before any wave-2 result)

Wave-2 cells train in stages: `sweep.py --steps 6000` (1/5), then `--steps 10000` (1/3), then to
30000. Each stage resumes from `last.ckpt`; the LR schedule (constant after 2.5k warmup) does not
depend on `max_steps`. After a stage the job scores the exact NELBO on 10% of the validation set
(`LIMIT_VAL_BATCHES=0.1`, `SEED=1`, `EVAL_BS=16`: the validation loader is shuffled with the seed,
so every cell sees the same random 10% and the same t draws) → `eval/nelbo_{steps}.json`. A cell goes on only if it clears the user's bar
(setup.md, added 2026-10-30 19:37) and is close to the best cell (`gate.py`):

- at 6k: NELBO < 3.0, and within 2% of the best wave-2 cell at 6k;
- at 10k: NELBO < 2.6, and within 1% of the best cell at 10k, wave 1 included. Wave-1 cells are
  scored on their `keep-10000.ckpt` with the same protocol.

The bars alone would continue almost every cell. Both wave-1 desa cells are below 2.6 at 10k, yet
r6_K1 is already 4.3% behind the fix arm. The relative margin is what saves compute; it is safe only
if rankings settle by 1/5, which the wave-1 trajectories (5k / 10k / 15k / 20k scored the same way)
check. Scores arrive one cell at a time, so the gate is applied to the scores already in: a cell
that passes against the best so far is continued, and a better score that arrives later tightens
the next gate.

The training loss is not used for the decision. `trainer/loss` is the last micro-batch, a fixed
t stratum, and the per-step `trainer/loss_std` is a spread, not a mean.

### Changes made on 2026-10-31 at 01:00 (user: final NELBO 1.7, a stricter rule, cancel what fails)

- **Target.** The fix anchor's 10%-validation scores fit 1.736 + 1.150 (step/5000)^−0.673 (5k, 10k
  and 20k fitted; 15k predicted 2.285, measured 2.286). That curve gives 2.08 at 30k and levels off at
  1.74. The listed knobs have moved the score by at most 5%, so 1.7 needs a cell well below the anchor
  curve, not one near it.
- **Stricter gate** (`gate.py`): at 6k, NELBO < 2.77 (the anchor curve's 2.753 + 0.5%) and within 1%
  of the best staged cell; at 10k, NELBO < 2.46 (the anchor's own 2.457) and within 0.5% of the best.
  At their 10k scores every ada cell fails it: r3 2.561, r6_K1 2.562, r5 2.570, r1.5 2.728.
- **Cancelled** (they fail the 1/3 rule, about 2/3 trained, on preempted A100s): r1.5_K0.5 ada (795774,
  stopped near 19k) and r5_K0.5 ada (795775, near 20k). Their 10k scores stand as their result.
  r3_K0.5 ada and r6_K1 ada also fail but were left to finish (over 85% done): their full-validation
  NELBO and GenPPL at 30k are the ada arm's answer.
- **Wave 3**, fix arm (CURV_BOUND 1, the leading arm), staged under the stricter gate:
  `r3_K0.5_lr6e-4` and `r3_K0.5_lr1e-3` (A100; LR is outside setup.md's knob list, but the target is
  out of reach of the listed knobs), `r5_K0.5`, `r3_K0.5_S64`, `r3_K0.5_bert` (desa). With the fix ub1
  cell already queued, the fix arm now covers rate, ub, S, embedding and LR. The ada bert cell
  went to `nice=500`, behind them (S4 and S64 ada were already there).
- **Not a measurement error.** `nelbo_variance.py` gave 1.93 on the fix model at 20k vs 2.19 from
  the stage score. Job 853558: the backbone in fp32 and under bf16 autocast give the same NELBO
  (1.912 / 1.930 for two draws), and the official eval gives 2.160 on its first 256 sequences.
  `nelbo_variance.py` takes the first 256 validation sequences, a contiguous and easier block, while
  the official loader shuffles. Its paired Δ's stand; its absolute level is that block's.

### 2026-10-31 01:45: the first 1/5 scores, and wave 4

| cell | arm | NELBO @ 6k | val/loss_std | gate |
|---|---|---:|---:|---|
| r3_K0.5_ub0.5 (trunc_exp, horizon 0.5) | ada | **2.1255** | 0.83 | continued to 10k (853909) |
| r3_K0.5_ub1 (trunc_exp, horizon 1) | ada | 2.1809 | — | stopped (+2.6%) |
| r3_K0.5 (exp, horizon 2.30), fitted curve | fix | 2.753 | ~1.1 | reference |

The finite horizon trains far faster. It is a bound on the process that stops at ub, so this is
not an evaluation artifact. On the exp-trained fix model, cutting the horizon to 0.5 at
evaluation raised its bound by 0.077 (nelbo_variance), so the gain comes from training at ub 0.5.
The exp cells of wave 3 now need ≤ 2.147 at 6k (1% of the best) and will most likely stop there.
Wave 4 explores the ub axis around 0.5: ub 0.25 ada (desa), 0.35 and 0.7 ada (A100), and ub 0.5 on
the fix arm (desa); the fix ub1 cell is already running. The exp cells of wave 3 still queued on
desa (S64, bert) went to nice 400, behind wave 4.

### 2026-10-31 03:50: wave 5, the proposal rate under trunc_exp

Under `trunc_exp` the rate no longer sets the horizon. It only shapes the t-proposal on [0, ub], so
it acts on the variance alone. On the fix model at 20k, the path-KL accumulated by ub 0.25 / 0.5 / 1
is 1.456 / 1.715 / 1.859 (nelbo_variance). That is a rate of 5.8 per unit time on [0, 0.25] and 1.04
on [0.25, 0.5], a fall of about 5.6× that a rate-6 to 7 proposal follows better than rate 3 does.
Cell **r6_K0.5_ub0.5** (ada, rate 6, ub 0.5) went to the A100 queue (856940). It starts when a wave-3
LR cell stops at 6k.

04:50: the fix anchor finished at **2.064** (full validation, 30k) vs 2.175 for the ada anchor. With
fix ahead under `exp` at every step, ub0.5 fix (853911) was moved ahead of ub0.25 ada (853910, nice 50)
for the next desa slot. `nelbo_variance.py` on the fix model's keep-30000 is queued (857397).

07:10: r6_K1 ada finished at 2.162. The 1/5 gate stopped fix ub1 (2.187, +2.9%) and r5 fix (exp,
2.756, +29.7%), both of which ended at 6k as staged jobs do. At ub 1, fix and ada tie, unlike under
`exp`. The freed desa slots went to ub0.25 ada and ub0.5 fix. The exp cells still queued (S64 / bert
on fix and ada, S4 ada) would most likely fail at 6k as r5 fix did. Wave 6 tests the setup.md knobs
on the best base instead: **r3_K0.5_ub0.5_bert** (ada, 859788, nice 0, ahead of them).
`sweep.py --pull` now writes into this account's outputs (the repo's `outputs/` link is not writable
by sc3379) and skips ch2263's stale stubs of the two desa wave-1 cells.

07:55: LR 6e-4 fix scored 2.735 at 6k, 0.6% under the anchor's curve (2.753): LR is not the lever.
It stopped at 6k, and r6_K0.5_ub0.5 took its A100. **r3_K0.5_ub0.5_S64** (ada, 860090) is queued
on the A100s for the next free one (the LR 1e-3 cell's, ~08:15).
08:15: LR 1e-3 fix scored 2.722 at 6k (−1.1% vs the anchor curve) and stopped; S64 ub0.5 took its A100.

08:40: **ub0.5 ada scored 2.008 at 10k** (6k: 2.126) and passed the 1/3 gate (bar 2.46; the next
best at 10k is the fix anchor, 2.457, +22%). It went to 30k as 860505, which is first in the desa queue.
Its slot had already gone to the queued exp cell S64 fix (853620) when the stage ended at 08:39,
because a stage's continuation is only submitted after its score. Through 6k and 10k with the
anchor's exponent (0.673), ub0.5 ada projects to 1.86 at 30k and an asymptote of 1.72.

09:10: with the user's approval, the five exp-mode cells were **cancelled**: S64 fix (853620, 27 min in),
plus bert fix, bert / S64 / S4 ada, still queued (853621, 849445/6/7). Every exp cell scored
2.72–2.76 at 6k against 2.126 for ub0.5, so they could not pass the 1/5 gate. Their knobs (S, BERT) are
tested on the ub 0.5 base instead (r3_K0.5_ub0.5_S64, r3_K0.5_ub0.5_bert). The ub0.5 ada run to 30k
(860505) started at once. The user also said cancelling this experiment's jobs needs no approval
from now on.

09:15: ub0.35 ada scored 2.152 at 6k (+1.25%) and stopped. The horizon has an optimum near 0.5 (0.35:
2.152, 0.5: 2.126, 1: 2.181; 0.7 and 0.25 are training). For the freed A100: **r3_K0.5_unif0.5**
(ada, 860761), `TIME_CONVERSION_MODE=unif` on [0, 0.5]. It is the setup.md mode not yet tried, the
flat (rate → 0) end of the proposal family that rate 6 brackets from the steep side. `sweep.py`
cells now take an optional `mode` (default trunc_exp when ub is set).

10:40: ub0.7 ada scored 2.132 at 6k (+0.31%) and went to 10k (861620, A100). With ub 0.35 / 0.5 / 0.7 / 1
at 2.152 / 2.126 / 2.132 / 2.181, the optimum is broad, between 0.5 and 0.7. Wave 7: **r6_K1_ub0.25**
(ada, 861621, desa, pending), K −1 with rate 6 and ub 0.25. The unit rate (6) and unit horizon (0.25)
are ub 0.5's, so only the state scale R changes (1.41 → 1), as r6_K1 did under `exp`.
`curvature_ts.py` now skips the `keep-N-v1` re-saves of resumed stages, which crashed it.

12:35: **ub0.5 fix scored 2.081 at 6k**, 2.2% under ub0.5 ada (2.126). Fix beats ada at ub 0.5 as it did
under `exp` (at ub 1 they tied). The pending K −1 ada cell (861621) had just taken the freed desa slot.
It was cancelled 2 minutes in, and ub0.5 fix went to 10k there (864309, ~16:10). Its 30k stage is
queued behind it with `--dependency=afterok` (864328, new `sweep.py --after`), so the slot can't be lost
between the stage end and the gate. If the 10k score fails the 1/3 gate, cancel 864328. Wave 7 moves
to fix: **r6_K1_ub0.25 fix** (864330) and **r3_K0.5_ub0.35 fix** (864331), pending for the slots ub0.25
and bert ada free at ~13:00. The ada knob cells already running (bert, S64, rate 6, unif) go on to 6k.
Their effect against the ada base (2.126) carries over to fix, but they won't pass the gate against
the fix leader.

13:00: ub0.25 ada scored 2.164 at 6k (+4.0%) and stopped. Ada's ub curve at 6k: 0.25 / 0.35 / 0.5 / 0.7 / 1 →
2.164 / 2.152 / 2.126 / 2.132 / 2.181. r6_K1_ub0.25 fix (864330) took its slot.
13:02: ub0.5_bert ada scored 2.123 at 6k, against 2.126 for GPT-2 XL on the same base (−0.1%, a tie).
It stopped (+2.1% vs the fix leader). r3_K0.5_ub0.35 fix (864331) took its slot.
15:10: r6_K0.5_ub0.5 ada scored 2.135 at 6k (+0.45% vs rate 3 on the same base) and stopped: the rate is
not a lever under trunc_exp. **r3_K0.5_ub0.7 fix** (867046, A100, running since 15:12) took its A100. It
brackets the fix arm's horizon from above, with ub 0.35 fix (864331) below.
15:40: **ub0.7 ada scored 2.007 at 10k**, a tie with ub0.5 ada (2.008). It was 0.31% behind at 6k, so the
longer horizon catches up with training. It passed the 1/3 gate and went to 30k (867339, A100). ub0.5
fix's 10k score arrives at ~16:10. If fix leads it by more than the 0.5% margin, 867339 is cancelled and
its A100 goes to a fix cell.
16:12: **ub0.5 fix scored 1.959 at 10k** (6k: 2.081), 2.5% under both ada cells. Its 30k stage 864328
started in the freed slot at once (`afterok`). With the anchor's exponent through 6k and 10k:
1.80 at 30k, asymptote 1.66. ub0.7 ada (867339) now fails the 0.5% margin. It keeps its A100 until the
`unif` / S64 ada scores (~16:30 / ~17:00) decide which fix cell replaces it.
16:28: r3_K0.5_unif0.5 ada scored 2.207 at 6k (+3.8% vs trunc_exp rate 3 on the same base) and stopped.
Inside [0, ub] the proposal shape now reads unif 2.207, rate 3 2.126, rate 6 2.135, so rate 3 stays.
Wave 8 on its A100: **r3_K0.5_ub0.5_lr1e-3 fix** (868664). Under exp, LR 1e-3 gained 1.1% at 6k. The lower
gradient variance of trunc_exp might tolerate a higher LR better.
16:50: r3_K0.5_ub0.5_S64 ada scored 2.117 at 6k (−0.39% vs S = 16) but trains 13% slower (1.60 vs ~1.84 it/s on
the same A100 node); per GPU-hour S = 16 is ahead (it gains ~0.024 in 13% more steps here). S = 16 stays.
S64 stopped. Its A100 went to **r3_K0.5_ub0.5_S4 ada** (869152): S = 4 / 16 / 64 on one base, for the MC word
count setup.md asks to choose. **ub0.7 ada (867339) is kept to 30k after all.** It is the exact ub pair of
ub0.5 ada (860505), both ada and both at 30k. At 10k the two tie and ub 0.7 is still catching up, so this
pair answers whether the best horizon grows with training. The fix arm's ub comparison (ub0.7 fix,
867046) gets to 30k a day later, if at all.
18:26: r3_K0.5_ub0.35 fix scored 2.109 at 6k (+1.36%) and stopped; on ada ub 0.35 was +1.2% too.
**r6_K1_ub0.25 fix scored 2.097** (+0.77%), inside the 1% margin, so it passed the 1/5 gate and went to 10k
(870629, desa, ~22:30). Under `exp`, K −1 tied K −0.5 at 10k and finished 0.6% ahead at 30k, so the
gap may close. Wave 9 in the second freed desa slot: **r1.5_K0.25_ub1 fix** (870630). It is K −0.25
with ub 0.5's unit proposal (unit rate 1.5/0.25 = 6, unit horizon 0.25 × 1 = 0.25), so the state
scale R grows to 2 on the other side of the anchor's 1.41.
22:05: r3_K0.5_ub0.7 fix scored 2.110 at 6k (+1.43%) and stopped. On fix, ub 0.35 / 0.5 / 0.7 →
2.109 / 2.081 / 2.110, a sharper optimum at 0.5 than ada's (where 0.7 was only +0.31%). LR 1e-3 on
ub 0.5 fix (868664) was cancelled at step 4.6k on its logged train loss (flat at ~4.28 from step 1000,
against 2.64 for the LR 3e-4 cell on desa). **That cancel was a mistake (23:35).** Wave 10 on the two free
A100s: **r3_K0.5_ub0.5_lr6e-4 fix** (874030), the middle LR, and **r3_K0.5_ub0.5 on a new arm `ada2`**
(874031, CURV_BOUND 2, λ ∈ [1/2, 2]), the middle of the bound axis (fix 1, ada 4).
23:20: **r6_K1_ub0.25 fix scored 1.966 at 10k**, +0.36% vs ub 0.5 fix (1.959), down from +0.77% at 6k. It
passed the 1/3 gate (0.5% margin) and went to 30k (874737, desa; ~Nov 1 17:30 plus evals). Its stage
ended at 22:05, two minutes before the watcher was re-armed, so the event was missed and the slot sat
idle for 70 minutes. The watcher now keeps its last state in ~/tmp/hbfm_elbo_event.snap and starts from it.
23:35: **The A100 site's logged train loss is not usable to judge a run, even as a trend.** The logged
value is the last micro-batch only, at fixed antithetic time bands. The A100 runs 1 GPU × micro-batch
64 and desa 4 GPUs × 16, so the two log different time bands. On the A100, every fix cell's logged
loss is flat or rising from step ~1000: ub 0.7 fix 4.24 → 4.39 over steps 2000–6000 (val 2.110 at 6k),
and under `exp` LR 1e-3 fix 4.46 → 4.63 (val 2.721, the best `exp` cell at 6k). LR 1e-3 on ub 0.5 fix
looked the same (4.28 flat), so it was cancelled on an artifact. It was resubmitted (874787, pending
for the next free A100) and resumes from last.ckpt (step 4500, on the node's /scratch). The S = 4 "stall"
of 22:05 was the same artifact: on the A100, S = 4 (3.35 at steps 5200–6000) sits between S = 64 (3.22)
and ub 0.35 ada (3.49). No train-loss kill rule: runs are judged only on the stage NELBOs.
23:55: **r1.5_K0.25_ub1 fix scored 2.077 at 6k**, the best 6k score (−0.16% vs ub 0.5 fix at K −0.5). At the
same unit proposal K −1 / −0.5 / −0.25 → 2.097 / 2.081 / 2.077, so a larger state scale R helps a little at
6k. The K −1 gap halved by 10k, so the ordering may not hold to 30k. It went to 10k (875137, desa, ~03:40).
00:10: r3_K0.5_ub0.5_S4 ada scored 2.193 at 6k (+3.2% vs S = 16) and stopped. On one base S = 4 / 16 / 64 →
2.193 / 2.126 / 2.117, with S = 4 no faster than S = 16 on the A100. **S = 16 is the MC word count.** LR 1e-3 fix
(874787) took its A100 and restored last.ckpt (step 4500); its 6k score is due ~02:15.
01:27: **LR 1e-3 on ub 0.5 fix scored 2.002 at 6k, −3.8% vs LR 3e-4 (2.081)**, by far the largest gain
since trunc_exp itself (under `exp` the same LR step gained only 1.1%). The schedule is constant after a
2500-step warmup, so a larger LR is not cut back later. It goes to 10k on the A100 (876308). K −1 fix to 30k
(874737, ~13k) was cancelled: it tied ub 0.5 fix at 10k (+0.36%), and at LR 3e-4 its 30k score cannot
change the choice. Its slot runs LR 2e-3 on ub 0.5 fix (wave 11, 876316), to see whether the LR gain
continues before LR 1e-3 is carried to 30k. If the 3.8% ratio held to 30k, LR 1e-3 would reach ~1.73.
03:10: K −0.25 fix scored 1.955 at 10k: K −1 / −0.5 / −0.25 → 1.966 / 1.959 / 1.955, the same order as at 6k.
It passes the 1/3 gate, but at LR 3e-4 it would lose to the LR 1e-3 cells by ~3%, so its 30k run is not the
best use of a slot. Instead K −0.25 is retrained at LR 1e-3 (wave 12, 877057, desa) to see whether the two
gains add up. Its LR 3e-4 checkpoints are kept, so it can still go to 30k if a slot is idle.
03:51: LR 6e-4 on ub 0.5 fix scored 2.026 at 6k, so LR 3e-4 / 6e-4 / 1e-3 → 2.081 / 2.026 / 2.002. The gain
shrinks per LR step; LR 2e-3 (07:00) shows whether 1e-3 is near the top. LR 6e-4 stopped (+1.2%). Wave 13: ub
0.7 at LR 1e-3 (877382, A100). At LR 3e-4, ub 0.7 closed its 6k gap to ub 0.5 by 10k on ada. A higher LR is
like more training, so the best horizon may move up.
04:21: **ub 0.5 ada finished: NELBO 1.828 at 30k on the full validation set** (GenPPL 10.45, entropy 3.99). That
is −16% vs `exp` ada (2.175) and below its two-point projection (1.86). Through 6k / 10k / 30k its curve is
1.534 + 0.640 (step/5000)^−0.435. That exponent is flatter than the `exp` anchor's 0.673, so late training
gains more. With it, ub 0.5 fix (LR 3e-4) projects to ≈1.77 and LR 1e-3 (at its 6k ratio) to ≈1.70. The
cosine LR schedule (configs/lr_scheduler/cosine_decay_warmup.yaml) could add more. It would need a
schedule knob in scripts/train/tinystories/hbfm.sh, and that script must not be edited while jobs are
running, so it is left as a follow-up.
04:24: **CURV_BOUND 2 (ada2) scored 2.129 at 6k**, level with bound 4 (2.126) and 2.3% behind fix (2.081). ada's λ
at 6k is 0.66, inside [1/2, 2], so bound 2 does not bind. What costs is learning λ, not the range. ub 0.5 ada's
λ: 0.87 / 0.66 / 0.61 / 0.555 at 2k / 6k / 10k / 24k. Wave 14 uses the freed slots. ada at LR 1e-3 (A100) checks
fix > ada at the chosen LR. S = 64 at LR 1e-3 (desa) asks whether S = 64's per-step gain (−0.39% at 6k, LR 3e-4)
stacks: the target is the NELBO at 30k steps, not per GPU-hour.
06:04: **LR 1e-3 scored 1.907 at 10k, −2.65% vs LR 3e-4** (−3.8% at 6k). With the ub 0.5 ada exponent (0.435),
the two-point fits give 1.762 (LR 1e-3) vs 1.772 (LR 3e-4) at 30k, with asymptotes 1.525 vs 1.468. A constant
higher LR learns faster early but plateaus higher: the usual sign that it should decay. It goes to 30k
(878861) as the constant-LR reference. A cosine schedule (2500-step warmup as now, then cosine to 1e-6 at 30k)
keeps the early gain and removes the late noise floor. It is the most likely way to 1.7. Implementation:
`LR_SCHED` / `LR_DECAY_STEPS` in the launcher (opt-in, default unchanged), `sched` in sweep.py's cells. Its
peak LR is chosen at 07:00 from LR 2e-3's 6k score. At 6k a cosine run is still at 96% of its peak LR, so
its stage scores track the constant run's. The schedule pays off late, so it trains to 30k without being gated
on its 6k score.
07:20: **LR 2e-3 scored 1.994 at 6k**, −0.36% vs LR 1e-3 (2.002). The constant-LR gain nearly stops at 2e-3, so
2e-3 is the cosine peak. The cosine run (r3_K0.5_ub0.5_lr2e-3_cos, 879486) starts from LR 2e-3's 6k checkpoint
instead of step 0. The schedule is then warmup 2500, constant 2e-3 to 6k, and cosine to 1e-6 at 30k. At 6k a
from-scratch cosine run would be at 96% of its peak LR, so the two schedules differ little, and the resume
saves ~5 h on desa. Its 10k score (from keep-10000.ckpt) is the 1/3 check against LR 1e-3's 1.907.
09:12: K −0.25 at LR 1e-3 scored 2.008 at 6k, +0.31% vs K −0.5 (at LR 3e-4 it was −0.16%), so K is a ±0.3% knob
and is closed at K −0.5. Not continued, although inside the 1% gate: it differs from a better cell only in K.
Its desa slot goes to wave 16, a twin of the cosine run with S = 64 (same checkpoint, same schedule). At LR 3e-4,
S 64 gained 0.39% at 6k. Estimator variance should cost most late in training, while the LR decays, so the
twin pairs S with the schedule where it should count most. It runs ~13% slower per step.
10:17: **ub 0.5 fix (LR 3e-4) finished at 1.759** (full validation), below its projection (1.772) and 3.7% below
ub 0.5 ada (1.828). Its three points fit c = 0.349 (ada: 0.435). With it, constant LR 1e-3 projects to 1.752 at
30k, so a constant LR leaves ~3% to the target. The remaining levers are the LR decay (the two cosine runs) and
the horizon: at LR 3e-4, ub 0.7 ada closed its 6k gap to ub 0.5 by 10k (2.007 vs 2.008), so the best ub may
grow with training. ub 0.7 ada's 30k final (~17:00) and ub 0.7 fix's 6k at LR 1e-3 (~10:50) decide that.
10:30: ub 0.7 at LR 1e-3 is +1.2% at 6k (fix, LR 3e-4: +1.4%), so the horizon optimum does not move with the
LR, and ub 0.7 with cosine is dropped. S = 64 at LR 1e-3 is −0.58% at 6k (LR 3e-4: −0.39%). The remaining
levers are the schedule, the peak LR and S, so the free slots go to cosine runs that differ in the peak:
1e-3 (from S64 LR 1e-3's 6k checkpoint), 2e-3 (waves 15–16) and 3e-3 (from step 0). The A100 slot runs a
seed-2 copy of ub 0.5 LR 1e-3 to 6k. Several conclusions rest on 0.3–0.6% gaps between single seeds
(K, rate, S), and seed noise had not been measured.
12:21: **the cosine run (peak 2e-3) scored 1.893 at 10k**, −0.74% vs constant LR 1e-3 (1.907). Through 10k its LR
has fallen only 7%, so this is mostly the higher peak: at 6k, constant 2e-3 led 1e-3 by 0.36%. With c = 0.349,
its two points give 1.727 at 30k as a constant-LR curve, and the decay acts after that. Constant LR 1e-3 (878861)
fails the 1/3 margin against it (+0.75%) but stays as the constant reference (status.md).
13:13: **its S = 64 twin scored 1.885 at 10k**, −0.42% vs the S = 16 run (same 6k checkpoint, same schedule).
That is a clean S comparison over 4k steps at high LR, and the gain is at least as large as at constant LR
(−0.39% / −0.58% at 6k). S = 64 is the MC rate; both cosine runs pass the 1/3 gate.
14:32: the S = 64 cosine run from peak 1e-3 scored 1.893 at 10k: +0.45% vs peak 2e-3 at S = 64, inside the
0.5% margin, so it continues. Peak 2e-3 beats 1e-3 under the same S and schedule; peak 3e-3 (882038) decides
whether the optimum is higher.
15:03: ub 0.7 ada finished at 1.815 on the full validation set, −0.71% vs ub 0.5 ada (1.828), after +0.3% at 6k
and a tie at 10k. Its λ fell to 0.447 (ub 0.5: 0.545), so the mean clock horizon λ·ub is 0.31 vs 0.27, both
below fix's 0.5. The ada swing is consistent with a fixed optimum near 0.5 that ada approaches from below. Fix's
6k deficit at ub 0.7 (1.2–1.4%) exceeds ada's 1.0% swing, so ub 0.5 stays for the 30k runs. Wave 19, ub 0.6 fix
at LR 1e-3 to 6k (A100), maps the fix side between 0.5 and 0.7, with ub 0.5's seeds 1–3 as the noise yardstick.
16:22: peak 3e-3 (S = 64, cosine from step 0) scored 1.995 at 6k, +0.24% vs S = 64 at 1e-3 (1.990). At 6k it is
past the optimum; it passes the 1/5 gate, and its 10k score against peak 2e-3 (1.885) decides the peak.
17:21: seed 2 of ub 0.5 fix at LR 1e-3 scored 2.004 at 6k vs 2.002 for seed 1 (+0.10%; eval seed and val subset
fixed, so this is training-seed noise). Wave 20 hedges the horizon at 30k: the best recipe (S 64, cosine from 2e-3)
at ub 0.6 on the A100, gated at 6k and by wave 19.
18:27: seed 3 scored 2.010: seeds 1–3 give 2.005 ± 0.0045 (0.22%), so two single-seed runs differ by ~0.32% (1σ)
from the seed alone. The variance measurement is repeated on the trunc_exp-trained ub 0.5 fix 30k model (ub 0.25 /
0.35 / 0.5 and S 4 / 16 / 64 at ub 0.5), the setting the final recipe uses.
20:07: peak 3e-3 scored 1.9005 at 10k, +0.84% vs peak 2e-3 (1.885); cosine peaks 1e-3 / 2e-3 / 3e-3 at S 64 give
1.893 / 1.885 / 1.900, so 2e-3 is the peak, and 3e-3 was cancelled. Its slot (wave 21) trains the chosen recipe from
step 0 with S 64 throughout, the clean version of wave 16 (whose first 6k ran at S 16).
21:44: ub 0.6 fix at LR 1e-3 scored 2.009 at 6k (ub 0.5: 2.005 ± 0.0045 over three seeds; ub 0.7: 2.025). The fix
horizon curve is flat from 0.5 to 0.6, so the ub 0.6 hedge at 30k stays, and ub 0.6 continues to 10k to see whether
the longer horizon gains with training on fix as it did on ada.

### 2026-11-06: closed

On 2026-11-05 the user stopped new waves after wave 40 and let the running ones finish (waves 39, 41–45); the last
finished on 2026-11-06 at 16:08. Waves later than the entries above are logged in `status.md`, and the outcome is in
`RESULTS.md`.

## Variance measurement (question 2, H2, H3)

`nelbo_variance.py` scores 256 validation sequences × 4 independent draws on a fixed trained
checkpoint, with the same seeds in every setting. Settings: `exp`; `trunc_exp` at ub 0.25 / 0.5 / 1 /
2; and the S = 4 / 16 / 64 MC rate. For each it reports the mean NELBO split into the reconstruction
and path-KL terms, the paired shift from `exp`, the estimator std (within a sequence, across draws)
and the between-sequence std. Run it on the fix arm (λ = 1) and on the ada cells. The training side
is `trainer/loss_std` in the wave-2 runs. It was added to `trainer_base.py` on 2026-10-30, so wave-1
runs don't log it.

## Compute and wall clock

- desa (sc3379): 4× A5000/A6000 per cell, ~3.3 s/step. To 6k: ~5.5 h; 6k → 10k: ~3.7 h;
  10k → 30k: ~18 h. A stage-end NELBO on 10% of validation takes ~10 min on one GPU.
- Wave-1 eval per cell after 30k (val PPL, 64 samples, full exact NELBO, λ): ~0.5–1 h on an A100
  and 1–2 h on desa (estimate).
- Slots: wave-1's desa jobs free 8 A5000s around Oct 31 07:00–09:00. Wave 2 queues behind them.
  Stage 1 (5 cells × ~22 GPU-h) then takes roughly a day on two slots. The 10k and 30k stages need
  ~15 GPU-h per cell and ~73 GPU-h per cell.
- Logging: wandb offline, synced afterwards to project `hbfm_elbo_tinystories_256` (entity
  `syctw`), one wandb run per cell. Later offline dirs (stages, preemption restarts) are appended
  with `wandb sync --id <cell> --append`.
