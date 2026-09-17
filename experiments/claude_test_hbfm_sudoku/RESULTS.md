# claude_test_hbfm_sudoku — Results

HBFM board accuracy on hard Sudoku, LR × noise schedule × seed (see `EXPERIMENT.md`). Regenerate the table
with `python experiments/claude_test_hbfm_sudoku/report.py`. Every `±` is the sample std over seeds (ddof=1).

<!-- report:begin -->
48/48 cells evaluated (2000 boards each).

| rate | K | LR | noise | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---|---|---:|---:|---:|---:|
| 0.01 | -0.5 | 3e-4 | log-linear | 0.1% | 0.8% | 0.1% | **0.4 ± 0.4** (n=3) |
| 0.01 | -0.5 | 3e-4 | log-linear-adaptive | 28.7% | 22.1% | 22.4% | **24.4 ± 3.7** (n=3) |
| 0.01 | -0.5 | 5e-4 | log-linear | 0.0% | 0.2% | 0.0% | **0.1 ± 0.1** (n=3) |
| 0.01 | -0.5 | 5e-4 | log-linear-adaptive | 24.3% | 22.9% | 25.4% | **24.2 ± 1.2** (n=3) |
| 0.01 | -0.5 | 1e-3 | log-linear | 0.0% | 0.0% | 0.0% | **0.0 ± 0.0** (n=3) |
| 0.01 | -0.5 | 1e-3 | log-linear-adaptive | 0.1% | 18.5% | 20.0% | **12.8 ± 11.1** (n=3) |
| 0.01 | -1.0 | 3e-4 | log-linear | 0.0% | 0.0% | 0.0% | **0.0 ± 0.0** (n=3) |
| 0.01 | -1.0 | 3e-4 | log-linear-adaptive | 10.6% | 8.3% | 17.4% | **12.1 ± 4.8** (n=3) |
| 0.01 | -1.0 | 5e-4 | log-linear | 0.0% | 0.0% | 0.0% | **0.0 ± 0.0** (n=3) |
| 0.01 | -1.0 | 5e-4 | log-linear-adaptive | 0.0% | 15.8% | 7.4% | **7.8 ± 7.9** (n=3) |
| 0.01 | -1.0 | 1e-3 | log-linear | 0.0% | 0.0% | 0.0% | **0.0 ± 0.0** (n=3) |
| 0.01 | -1.0 | 1e-3 | log-linear-adaptive | 0.0% | 8.8% | 0.0% | **2.9 ± 5.1** (n=3) |
| 3 | -0.5 | 3e-4 | log-linear | 60.4% | 59.8% | 63.2% | **61.1 ± 1.8** (n=3) |
| 3 | -0.5 | 3e-4 | log-linear-adaptive | 46.8% | 38.6% | 45.5% | **43.6 ± 4.4** (n=3) |
| 3 | -1.0 | 3e-4 | log-linear | 49.6% | 50.7% | 53.9% | **51.4 ± 2.2** (n=3) |
| 3 | -1.0 | 3e-4 | log-linear-adaptive | 41.6% | 41.0% | 46.3% | **43.0 ± 2.9** (n=3) |
<!-- report:end -->

## K = −1 half (18 cells, finished 2026-09-15 ~22:15) — analysis

Trained BEFORE the two fixes below (weight `1/(0.01 u)` without `|α′_t|`; adaptive yaml
defaults `refit_every=500, ema=0`). 2000 hard boards each; ±2.1 pt 95% CI at mid-range accuracy.
Diagnostics: a CPU workflow over `eval/results.json`, the checkpoints (raw + adaptive buffers)
and the wandb logs; a completeness critic reviewed the draft. "Measured" = a number in those
artifacts; "hypothesis" = an inference still to be tested.

### Measured

1. **Decode regimes.** Cell accuracy on the 81 digit cells (the 8 separators are reproduced at
   ~100% and excluded) separates three regimes:
   A "solving" 79.8–83.5% (all non-zero adaptive cells; 7–17% exact boards, the bulk of the
   wrong boards have 11–40 wrong cells, rows/boxes 86–94% valid, columns 74–84%: consistent
   alternative grids, not near-misses); B "copy the givens" 45.8–60.6% with the givens reproduced
   at 99.96–100% and non-given accuracy 14–37% (chance 11%) — all log-linear cells at 3e-4/5e-4
   (except 5e-4 s2) and the three 0% adaptive cells; C "collapsed" 11–36% with the givens
   themselves lost — log-linear at 1e-3 (all seeds) and 5e-4 s2. No NaN/inf anywhere; weight norms
   of the failed cells sit inside the good cells' range.
2. **Heat-time coverage of the rate-0.01 proposal.** With `u ~ U[1e-3, 0.999]` and
   `t = −ln u / 0.01`: P(t ≤ 3) = 2.9% (≈7 of 256 sequences per batch), P(t < 2) = 1.9%, 37.9% of
   every batch is clamped at t = 97 (state already identifies the word; CE = 0), median t = 69, and
   training never sees t < 0.10 (the `noise.eps=1e-3` floor). On the context-free Bayes profile of
   `(H^3)^3` (CE 1.96 at t=0.1, 0.20 at 1, 0.018 at 2, 0.002 at 3) 99% of E[w·CE] comes from the
   1.9% of draws at t < 2. The sampler's grid spends 174 of its 180 steps in [0.1, 3].
3. **What the adaptive schedule did.** All adaptive `last.ckpt`s have P(clamped) = 0 and the six
   solving cells P(t ≤ 3) = 0.61–0.88 (median t 1.2–2.3); the three 0% adaptive cells P(t ≤ 3) =
   0.44–0.51 — so the remap alone is not sufficient. `AdaptiveSchedule._refit` rebuilds the map
   from the BASE schedule using a loss profile recorded under the CURRENT map; with `ema=0` this
   is a 2-cycle (simulated 74–90% vs 11–17% of mass at t ≤ 3 on alternate refits; the logged loss
   alternates in 500-step blocks). Because the HBFM weight lacked `|α′_t|`, the batch loss scale
   moved with the map: ~2–9× between alternate refit periods early on, and a ~30–50× jump at the
   first refit (step 1500, during LR warmup) in all nine adaptive cells; the three that end at 0%
   are the three still at 15–23 at step ~2000 (others 1–12), cleanly separated by step 2600–3000,
   and they never fit: unweighted CE at t < 0.5 stays 0.21–0.45 at 20k vs 0.005–0.008 for the
   siblings. Failures are LR-ordered: 0/3 at 3e-4, 1/3 at 5e-4, 2/3 at 1e-3.
4. **Geometry is not the bottleneck** on the three checkpoints probed: horosphere-only Bayes
   decoding with the trained boundary tables is 95.1–95.5% at t=1 and 100% at t=3, close to the
   93.4% a random ngpt-scale table gives at t=1 (`coverage.log`). Pairwise separability of all
   18 tables was not measured.
5. **Seed clustering.** Same-seed cells' solved-board sets overlap 4.3–5.7× the independent
   expectation, cross-seed cells only 1.2–1.9×;
   `sudoku_eval` seeds the Brownian stream with the training seed, so this confounds init and
   eval noise. The union of the six solving models covers 39.7% of the boards (best single 17.4%).
6. Minor: `readout_precision=float32` underflows the per-token CE to exactly 0 for 30–63% of the
   adaptive buffer (harmless for the mean); `_sample_t`'s sorted antithetic strata give the
   epoch's 128-sample tail batch only t_u ≤ 0.5 (heat ≥ 69) — harmless.

### Hypotheses and how they are being tested

- **H1 — the rate-0.01 proposal is mismatched to the `(H^3)^3` resolution scale (t ≈ 0.1–3).**
  Predicts log-linear cells learn essentially no prompt → cell mapping (regime B) and that a
  matched rate (1 or 3) beats rate 0.01 at equal steps. Test: 5k-step log-linear runs at rate 1
  and 3 vs the 5k-step rate-0.01 checkpoints — see "H1 confirmed" below. Caveat (critic): the B-vs-C split inside
  the log-linear arm is an LR/stability effect that coverage does not explain.
- **H2 — the uncorrected importance weight + undamped refits destabilise the adaptive arm.**
  Fixes applied after this half: F2 `weight *= |dα_t|` in `HyperbolicBoundaryFM.nll` (MDLM's
  `dα/(1−α)` over the rate; makes the loss invariant to the remap) and F1 the repo's adaptive
  recipe (`ADA_REFIT_EVERY=50, ADA_EMA=0.9`, buffer 50×batch, as every `*_truncated_adaptive.sh`
  does). Test: 5k-step adaptive runs on the worst (1e-3 s1) and best (3e-4 s3) cells, F2+yaml
  defaults vs F2+recipe, against their old 5k checkpoints — see "H2 measured" below.
- **H3 — sampler horizon / step size.** t_max = 3 was calibrated on the Bayes posterior and
  dt = 0.017 is inside the unbiased regime; predicts ablations (t_max 1/3/10, 600 steps,
  velocity=sample) change little. Test: GPU ablations on finished checkpoints — see "H3 supported" below.
- **H4 — sampler implementation with a real model.** Predicts a cheating one-hot posterior fed
  through the real `sudoku_eval` path reproduces ≈100% boards. Test: see "H4 refuted" below.
- **H5 — the t < 0.1 gap** (never trained, but where the sampler makes its first 6 steps) is a shared
  handicap, not what separates the arms (the solving cells solve anyway). Untested; would need a
  smaller `noise.eps` (t_min = eps/rate).

### Verification results so far (GPU, K = −1 checkpoints, 256 validation boards)

- **H4 refuted — the sampler machinery is correct.** Through the real `generate_samples` /
  prefix / greedy-decode path, a cheating one-hot posterior gives 100% boards, so does
  0.5·truth + 0.5·uniform, and the trained model with the truth injected at 50% of the solution
  cells; the model itself reproduces its official score (cell 0.865, board 0.184 on these 256 boards
  at the checkpoint's own sampler config, t_max = 30 / 200 steps, vs 17.4% on the 2000-board
  eval at t_max = 3 / 180 steps — different boards, so not a horizon comparison; H3 is tested
  within one run below).
- **H5/H6 measured — where the residual is accurate, and what generation actually uses.**
  Residual-only cell accuracy on solution cells (bridge state from the truth, prompt pinned):

  | cell | ckpt | t=0 (origin) | t=0.3 | t=1 | generation (cell / board) |
  |---|---|---:|---:|---:|---:|
  | 3e-4 ada-1 s3 (17.4%) | 5k / 10k / 15k / 20k | 0.79 / 0.84 / 0.85 / 0.86 | 1.00 | 1.00 | 0.78/0.004 → 0.87/0.18 |
  | 1e-3 ada-1 s1 (0%) | 5k / 10k / 15k / 20k | 0.36 / 0.50 / 0.51 / 0.58 | 0.33 → 0.60 (full 0.78 → 0.88) | 0.30 → 0.64 (full 0.98 → 0.99) | 0.28/0 → 0.53/0 |
  | 3e-4 ada-0 s1 (0%) | 5k / 10k / 15k / 20k | 0.51 / 0.52 / 0.54 / 0.54 | 0.51 → 0.59 (full 0.84 → 0.88) | 0.51 → 0.65 (full ≥ 0.98) | 0.49/0 → 0.52/0 |

  Three facts follow. (i) The good model's residual already reads the answer off the state by
  t = 0.3 (100%; 97% at t = 0.05, 99% at t = 0.1), and its context-only prior at the origin is
  86%. (ii) Generation's cell accuracy equals the origin prior's (0.865 vs 0.857) but its board
  accuracy does not: one-shot argmax of the origin prior solves **0/256** boards while the SDE
  solves 18% — the sequential commitment (each step's posterior reads the cells already
  committed through the state) is what turns 86% independent cell predictions into consistent
  grids; the cell-level ceiling, however, is set by the origin prior. (iii) The 0% adaptive cell
  and the log-linear cell never build a strong prior (58% / 54% at the origin after 20k steps;
  the log-linear one gains 3 points in 15k steps), which is the coverage failure (H1) and the
  refit/weight instability (H2) seen from the model's side; the log-linear residual cannot even
  read the answer off a resolved state (72% at t ≥ 3 residual-only; the full posterior is 100%
  there from the horosphere), i.e. it received almost no gradient anywhere. None of the three
  runs is degenerate (no NaN, adaptive buffers finite, refits 7/17/27/37 at the four checkpoints).
- **H3 supported — sampler settings are not the limiter.** On the 17.4% cell (256 boards, cell /
  board), all from the checkpoint's own sampler config (t_max = 30, 200 steps): default 0.865 /
  0.184; t_max=1 0.838 / 0.180; t_max=10 0.848 / 0.160; 600 steps 0.848 / 0.160; velocity=sample
  0.864 / 0.180; ancestral last step 0.865 / 0.184. (The official t_max = 3 / 180-step setting was
  not re-run on these 256 boards; it gives 17.4% on the 2000-board eval.) The 0% cells stay at 0
  under every setting (cell 0.52–0.60 = their origin prior).
- Consequence for the K = −0.5 half: evidence accumulates at unit rate |K|·t, so at K = −0.5 the
  bridge locks in half as fast and the prior has more steps to be revised before the state
  saturates — consistent with the K = −0.5 adaptive cells (22–29% at LR 3e-4 / 5e-4) roughly
  doubling the K = −1 ones.

- **H1 confirmed — the proposal rate is the dominant hyperparameter.** Plain log-linear schedule,
  K = −1, LR 3e-4, seed 1, **5k steps**, everything else as in the sweep:

  | `time_exp_rate` | P(t ≤ 3) in training | boards @5k | boards @20k |
  |---|---:|---:|---:|
  | 0.01 (spec) | 2.9% | 0.0% | 0.0% |
  | 1 | 95% | **27.3%** | — |
  | 3 | 100% | **30.4%** | — |

  At a rate matched to the `(H^3)^3` resolution scale HBFM reaches the H-FLM reference (29.4% at
  20k steps) in a quarter of the steps, with no adaptive schedule. The 12 follow-up cells at
  rate 3 (both K, both schedules, 3 seeds, 20k steps; tags `_rate-3`) are reported in "Rate-3
  follow-up" below.
- **H2 measured — the `|α′_t|` weight (F2) helps the unstable arm; the EMA refit recipe (F1) does
  not.** 5k-step re-runs of the failing and the best adaptive cell (K = −1, rate 0.01), scored on
  the residual prior at the origin / at t = 0.3 (the quantity generation is bounded by):

  | cell @5k | old weight, yaml refit | **F2**, yaml refit | F2 + recipe (refit 50, ema 0.9) |
  |---|---|---|---|
  | 1e-3 s1 (ends 0%) | 0.36 / 0.33 (P(t≤3) 0.65) | **0.57 / 0.71** (0.83) | 0.36 / 0.36 (0.41) |
  | 3e-4 s3 (ends 17.4%) | 0.79 / 1.00 (0.67) | 0.80 / 1.00 (0.66) | 0.76 / 0.99 (0.69) |

  F2 lifts the failing cell's prior at 5k to what the old run only reached at 20k (0.575) and is
  neutral on the good cell; the damped recipe slows the coverage remap at this rate (P(t≤3) 0.41
  vs 0.83) and helps neither. Board accuracy is 0–1% for all six at 5k, so it cannot separate
  them. Kept: F2 (`weight *= |dα_t|` in `HyperbolicBoundaryFM.nll`); the `ADA_REFIT_EVERY` /
  `ADA_EMA` script knobs default to the yaml values. For reference, the rate-1 / rate-3 log-linear
  models have origin priors of 0.883 / 0.886 after the same 5k steps.

## K = −0.5 half (the spec; 18 cells, finished 2026-09-16 ~01:10)

Same pre-fix settings as the K = −1 half (`algo.time_exp_rate=0.01`, adaptive yaml defaults);
eval horizon t_max = 3/|K| = 6 (180 steps).

| LR | log-linear | log-linear-adaptive |
|---|---:|---:|
| 3e-4 | 0.10 / 0.80 / 0.15 → **0.4 ± 0.4** | 28.7 / 22.1 / 22.4 → **24.4 ± 3.7** |
| 5e-4 | 0.0 / 0.2 / 0.0 → **0.1 ± 0.1** | 24.3 / 22.9 / 25.4 → **24.2 ± 1.2** |
| 1e-3 | 0.0 / 0.0 / 0.0 → **0.0 ± 0.0** | 0.1 / 18.5 / 20.0 → **12.8 ± 11.1** |

- The log-linear arm is again a coverage failure: at K = −0.5 the physical rate 0.01 is unit
  rate 0.02, so P(t_unit ≤ 3) rises from 2.9% to 5.7% — enough for a handful of boards (up to
  0.8%) but not for solving.
- The adaptive arm doubles the K = −1 numbers (24.4 ± 3.7 and 24.2 ± 1.2 at LR 3e-4 / 5e-4 vs
  12.1 ± 4.8 / 7.8 ± 7.9), with much smaller seed spread at 5e-4, and one LR-1e-3 seed still fails
  (1 board of 2000) as at K = −1. Two effects compound: the coverage doubling above, and the slower
  evidence accumulation (unit time |K|·t) that gives the prior more SDE steps before the
  horosphere locks the state in (see H5/H6).
- Best spec-conforming cell: LR 3e-4 seed 1, **28.7%**, vs 29.4% for H-FLM at K = −1 and
  33.9% for H-FLM at K = −0.5 (`hflm_curv_sudoku`, `eval_tkv-1` = `top_k_velocity=-1`, the
  protocol this eval uses).

## Rate-3 follow-up (12 cells, 20k steps, finished 2026-09-16 ~03:00)

LR 3e-4, `algo.time_exp_rate = 3` (unit rate 3/|K|), the `|α′_t|` weight, everything else as in
the main sweep; the adaptive cells use the refit-50 / ema-0.9 recipe (submitted before the A/B
showed it does not help). Eval unchanged (t_max = 3/|K|, 180 steps, greedy).

| K | noise | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---:|---:|---:|---:|
| −0.5 | log-linear | 60.4 | 59.8 | 63.2 | **61.1 ± 1.8** |
| −0.5 | log-linear-adaptive (recipe) | 46.8 | 38.6 | 45.5 | 43.6 ± 4.4 |
| −1.0 | log-linear | 49.6 | 50.7 | 53.9 | **51.4 ± 2.2** |
| −1.0 | log-linear-adaptive (recipe) | 41.6 | 41.0 | 46.3 | 43.0 ± 2.9 |

- Matching the proposal to the geometry's resolution scale is worth a factor of ~2.5 over the
  best spec cell (61.1 vs 24.4 at K = −0.5) and ~1.8× the H-FLM reference at the same curvature
  (33.9%), with a seed spread of ±2 instead of ±4–11.
- With a matched rate the adaptive schedule is no longer needed and, with the refit-50 / ema-0.9
  recipe, is 17.5 points worse at K = −0.5 and 8.4 at K = −1 than the plain schedule: its remap keeps chasing the steepest part
  of the (now well-covered) loss profile and over-concentrates the proposal.
- K = −0.5 > K = −1 at every 20k setting (61.1 vs 51.4 here; 24.4 vs 12.1 in the spec sweep;
  53.4 vs 51.4 at matched unit rate 3, below), though the 5k probes rank them the other way; the
  curvature × rate grid (below) locates the optimum.

## `forward_type=naive` probes (5k steps, LR 3e-4, seed 1)

Plain logits instead of "residual + Busemann log-densities"; the sampler consumes the model's
own softmax. Scorer = full-posterior cell accuracy on solution cells along unit heat time
(bridge from the truth) and generation on 256 boards; boards @5k on the 2000-board eval.

| setting | readout | prior at unit t = 0 / 0.1 / 0.3 | gen cell | boards (256) | boards (2000) |
|---|---|---|---:|---:|---:|
| K=−1, rate 3, log-linear | horosphere | 0.886 / 0.997 / 1.000 | 0.876 | 30.5% | 30.4% |
| | **naive** | 0.879 / 0.995 / 1.000 | 0.870 | 25.8% | 25.4% |
| K=−0.5, rate 0.01, adaptive (spec) | horosphere | 0.804 / 0.978 / 1.000 | 0.783 | 3.1% | 2.1% |
| | **naive** | 0.826 / 0.983 / 1.000 | 0.813 | 6.6% | 5.0% |
| K=−0.5, rate 0.01, log-linear (spec) | horosphere | 0.647 / 0.857 / 0.979 | 0.600 | 0.0% | 0.0% |
| | **naive** | 0.747 / 0.932 / 0.995 | 0.690 | 0.0% | 0.0% |

- The model learns the Busemann geometry on its own: by unit t = 0.3 the naive posterior is as
  resolved as the analytic one in every setting.
- At a matched rate the analytic readout is worth ~5 points @5k (30.5% vs 25.8% of 256 boards,
  ~12 boards) — a modest advantage at one seed; at the spec's rate 0.01 naive is slightly
  better, plausibly because it also gets gradient from the samples the horosphere already
  resolves (CE ≈ 0 there under `horosphere`; 14% of draws are clamped at K = −0.5) — two cells
  at one seed, mechanism not measured. Neither readout rescues the spec rate. Conclusion: keep `horosphere` (the
  default) and fix the rate; `naive` is a viable fallback, not a lever.

## Curvature × proposal-rate tuning (`sweep_tune.py`, `report_tune.py`)

Probe grid, 5k steps, LR 3e-4, seed 1, log-linear, `|α′_t|` weight; unit rate =
`time_exp_rate/|K|` so the heat-time coverage is comparable across K; eval t_max = 3/|K|:

| K \ unit rate | 1 | 3 | 10 |
|---|---:|---:|---:|
| −0.25 | 14.8 | 21.6 | 22.0 |
| −0.5 | 24.3 | 22.9 | 22.9 |
| −1.0 | 27.3 | **30.4** | 21.4 |
| −2.0 | 25.8 | 18.6 | 29.3 |

- Every cell of this grid beats every rate-0.01 cell at the same step count (0–2%): the rate
  fix is robust across K; within the grid the rate dependence is weak and non-monotone (single
  seed, ±2 pt CI, 5k steps), except that unit rate 10 hurts at K = −1.
- **5k probes rank curvature the wrong way round.** At 5k K = −1 leads K = −0.5 (30.4 vs 22.9),
  but at 20k K = −0.5 wins by 10 points (61.1 vs 51.4; the `_rate-3` cells, physical rate 3 =
  unit rate 6 at K = −0.5). Flatter curvature accumulates evidence more slowly (unit time
  |K|·t), which makes early training harder but leaves the prior more room to be revised
  before the state locks in. Curvature therefore has to be chosen at full length.
- 20k-step confirmation (3 seeds each, log-linear, LR 3e-4) was run for the open cells:
  K = −0.25 @ unit 3, K = −0.5 @ unit {1, 3, 10}, K = −2 @ unit 10 (tags `tune20k_*`); with the
  existing K = −0.5 @ 6 (61.1 ± 1.8) and K = −1 @ 3 (51.4 ± 2.2) this spans K ∈ [−2, −0.25] and
  unit rate ∈ [1, 10] at full length — next section. `python report_tune.py` prints both tables.

## 20k-step confirmation of the tuning (15 + 6 cells, finished 2026-09-16 ~05:25)

3 seeds each, LR 3e-4, log-linear, `|α′_t|` weight, eval t_max = 3/|K|; the two `_rate-3`
rows of the earlier follow-up are included (physical rate 3 = unit rate 3/|K|).

| K | unit rate (`time_exp_rate/|K|`) | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---:|---:|---:|---:|
| −0.25 | 3 | 56.6 | 55.5 | 56.4 | 56.1 ± 0.6 |
| −0.5 | 1 | 35.2 | 29.8 | 28.8 | 31.3 ± 3.5 |
| −0.5 | 3 | 54.8 | 52.8 | 52.6 | 53.4 ± 1.2 |
| −0.5 | 6 (physical 3) | 60.4 | 59.8 | 63.2 | **61.1 ± 1.8** |
| −0.5 | 10 | 62.2 | 56.8 | 54.9 | 57.9 ± 3.8 |
| −1.0 | 3 (physical 3) | 49.6 | 50.7 | 53.9 | 51.4 ± 2.2 |
| −2.0 | 10 | 54.3 | 59.9 | 54.4 | 56.2 ± 3.2 |

- **Rate.** At K = −0.5 the unit rate has a broad optimum around 6–10 (61.1 / 57.9) with a
  cliff below it (53.4 at 3, 31.3 at 1); the failure at the spec's 0.01 (unit 0.02) is the far
  end of that cliff. A unit rate r means the proposal's mean heat time is 1/r in unit time —
  the best settings put it at 0.1–0.17, i.e. inside the window where the context-free posterior
  is still uncertain (CE 1.96 at t=0.1, 0.71 at 0.5; `coverage.log`) and the model must use the prompt.
- **Curvature.** Every K ∈ [−2, −0.25] lands in 51–61% once the rate is in its good range, but
  each K except −0.5 was confirmed at a single unit rate, so curvature and rate are partly
  confounded: at the matched unit rate 3, K = −0.5 leads K = −1 by only 2.0 pts (53.4 ± 1.2 vs
  51.4 ± 2.2), and K = −1 was never run at 20k at unit 6 or 10. The 5k probe ranking (K = −1
  first) is not predictive of 20k.
- **Best configuration found:** `prod_factor_gaussian_curvature = [−0.5]*3`,
  `algo.time_exp_rate = 3` (unit 6), log-linear schedule, LR 3e-4 → **61.1 ± 1.8%** hard-board
  accuracy (H-FLM at K = −0.5: 33.9%; the best spec-conforming HBFM cell: 24.4 ± 3.7%). Unit
  rate 10 at the same curvature gives 57.9 ± 3.8 (~2 SE away): the rate optimum is broad and
  61.1 is the argmax of 7 cells, i.e. an optimistic point estimate.

## Fixed-rate curvature test — is (rate, K) → (c·rate, c·K) a symmetry? (9 cells, finished 2026-09-16 ~09:05)

Hypothesis (user): a large `time_exp_rate` and a small |K| both skew the heat-time allocation
towards early unit time, so holding the spec's rate 0.01 and shrinking |K| should reproduce the
tuned cells. Since τ = |K|·t and u = exp(−rate·t) = exp(−(rate/|K|)·τ), the pair enters the
*problem* only through the unit rate r = rate/|K|. Cells (`sweep_tune.py --fixed-rate`, 20k
steps, LR 3e-4, log-linear, 3 seeds, physical rate 0.01, eval t_max = 3/|K|):

| K | R = 1/√\|K\| | unit rate | matched tuned cell (K = −0.5) |
|---|---|---|---|
| −0.001 | 31.6 | 10 | 57.9 ± 3.8 |
| −0.0016667 | 24.5 | 6 | **61.1 ± 1.8** |
| −0.0033333 | 17.3 | 3 | 53.4 ± 1.2 |

**Code audit (3 lenses + synthesis, `logs/equiv/`), run while the cells train.** Everything on
the *problem* side is symmetric to float64 rounding, verified numerically at K = −0.5 / rate 3
vs K = −0.0016667 / rate 0.0100002: `time_conversion` (unit τ identical to 1e−16),
`max_heat_time` (96.96 unit both), `poincare_bridge_prod` (matched-seed unit-ball states equal to
5e−15), the polar readback and `horosphere_geometry` (log-softmax equal to ≤ 4e−12, target CE
identical), `_clean_state` (finite readback at ρ = 350R, one-hot margin 397 nats both), the DiT's
sigma (from u, and `time_conditioning: False` zeroes it anyway), the sampler grid and
`euler_maruyama_step` (179-step rollout with a Bayes-oracle posterior, 512 paths, matched seeds:
end states equal to 4e−14, decoded tokens 100 % identical), and the hydra plumbing (resolved
rate 0.0100002, curvature [−0.0016667]×3, t_max 1799.96, 180 steps). A trained K = −0.5 checkpoint
evaluated at K = −0.0016667 with `in_proj.weight` scaled by 1/17.32 reproduces its log-posterior
(argmax 100 %, max |Δ log p| 3e−5) — the two cells span the same function class.

The symmetry breaks in exactly two *optimizer*-side places:

1. **Loss scale × gradient clipping.** The weight 1/(rate·u)·|α′| is not renormalized, so the loss
   and every gradient are 300× larger at rate 0.01. With `trainer.gradient_clip_val = 1.0` the
   fixed-rate cells are norm-clipped at 100 % of steps (global grad norm 700–1000 at init, 8–14 at
   the 20k-equivalent point; clip factors 0.001–0.09), while the rate-3 cell is unclipped from
   somewhere before step 5000 on (norm 0.08 at 5k, 0.04 at 20k). Adam removes a constant factor
   but not per-batch normalization: in the fixed-rate cells the heavy 1/u tail of the importance
   weights is flattened batch by batch for ~19k of 20k steps.
2. **DiT input scale.** The trunk consumes z = R·w, so `in_proj` sees a 17.3× larger input at
   R = 24.5 with an R-independent init and Adam step: the residual stream starts 14–16× larger,
   the first LayerNorm sees a different function (cos 0.63 on bridge positions at identical w),
   and `in_proj` has a 17.3× higher function-space learning rate (its gradient is 5196× vs 300× for
   all other parameters at the function-equivalent point). Adam can absorb the rescale in ~1k
   steps (A never rescaled `in_proj` at all: Frobenius norm 13.05 → 13.26 over 20k).

**Paired loss traces (same seed ⇒ same data order, t draws and bridge noise; wandb offline
logs):** B/A = 300 for the first ~400 steps, a transient lag of 3.6× / 1.4× / 3.7× per seed in
unit terms peaking at steps 1400–1600, then parity from step ~5000 (B/(300·A) = 1.07 / 0.96 /
1.00 over steps 5000–6100). The audit's prediction: 58–62 %, ≈65 % chance the 3-seed mean lands
inside 61.1 ± 3.6; a result outside ~55–65 % would implicate the clipping (break 1), whose
isolating controls are `trainer.gradient_clip_val=300` at K = −0.0016667 and 1/300 at K = −0.5.
Logged losses of the fixed-rate cells must be divided by the loss-scale ratio (500 / 300 / 150 at unit
rate 10 / 6 / 3; R = 31.6 / 24.5 / 17.3) before comparing to the K = −0.5 cells.

**Results** (physical rate 0.01 for all rows; paired by seed with the K = −0.5 cells — same init,
data order, schedule-time draws and bridge noise):

| unit rate | K (fixed rate 0.01) | seed 1 | seed 2 | seed 3 | mean ± std | K = −0.5 reference | paired Δ (B − A) |
|---|---|---:|---:|---:|---:|---:|---:|
| 10 | −0.001 | 65.8 | 59.7 | 66.0 | **63.8 ± 3.6** | 62.2 / 56.8 / 54.9 → 57.9 ± 3.8 | +3.6 / +2.9 / +11.1 |
| 6 | −0.0016667 | 61.2 | 47.1 | 68.0 | **58.8 ± 10.7** | 60.4 / 59.8 / 63.2 → 61.1 ± 1.8 | +0.8 / −12.7 / +4.8 |
| 3 | −0.0033333 | 54.4 | 45.1 | 62.1 | **53.9 ± 8.5** | 54.8 / 52.8 / 52.6 → 53.4 ± 1.2 | −0.4 / −7.7 / +9.5 |

- **In the mean the equivalence holds.** Pooled over the 9 paired cells the fixed-rate set scores
  58.8 vs 57.5 for the K = −0.5 set; the paired difference is +1.3 ± 2.5 (SE over 9 pairs; std of
  the differences 7.6), i.e. no detectable shift. Row by row: +5.9 / −2.3 / +0.5 points, each inside
  its seed spread. So yes — at the spec's `time_exp_rate = 0.01`, curvature alone (K ≈ −0.001 …
  −0.0033) reproduces the tuned-rate accuracies: 63.8 ± 3.6 at K = −0.001 is the best mean of the
  whole study, and K = −0.0016667 averages 58.8 against the 61.1 reference.
- **In the variance it does not.** The within-row seed std is 3.6 / 10.7 / 8.5 (fixed rate) vs
  3.8 / 1.8 / 1.2 (K = −0.5) — a ~10× variance ratio on 6 + 6 df (F ≈ 10, p ≈ 0.01) — and the seeds
  are ordered identically in all three fixed-rate rows (seed 2 lowest: 59.7 / 47.1 / 45.1; seed 3
  highest: 66.0 / 68.0 / 62.1), which is not what independent eval noise (±2.2 pt CI) looks like.
  The seed enters both training (init, data order, noise) and eval (the Brownian stream). Which of
  the two optimizer-side asymmetries (300× loss scale under `gradient_clip_val = 1`, 17× input
  scale) or plain seed sensitivity drives it is diagnosed below; the isolating control
  (`sweep_tune.py --clip-control`: K = −0.0016667, unit 6, `trainer.gradient_clip_val = 300`,
  3 seeds, tags `..._clip300`) was submitted 2026-09-16 ~09:20.

**Diagnosis of the seed spread (CPU workflow over the 18 paired cells: wandb loss traces +
board lenses; `logs/fixedrate/`).** The two anomalous cells (unit 6 and unit 3, seed 2) are an
*optimization* deficit, not a decoding one; the unit-10 seed-2 cell is ordinary seed variance.

- Loss predicts accuracy across the 18 cells: Spearman(mean loss over 5k–20k, boards) = −0.94.
  On identical batches the seed-2 fixed-rate cells train to a 1.36× (unit 6) / 1.17× (unit 3)
  higher late loss than their paired K = −0.5 runs (paired t over 49 windows, p < 1e−12) and
  1.76× / 1.40× higher than their own seed siblings; within the K = −0.5 set seed 2 is normal
  (1.01–1.08×). The divergence is a smooth late plateau, not instability: the unit-6 seed-2 loss
  ratio to its pair is 0.97 at 6–8k and then rises monotonically 1.10 → 1.19 → 1.22 → 1.29 →
  1.33 → 1.42 (18–20k); no spikes (max/median over the last 5k = 1.25, the lowest of the 18).
- The decode side is clean: givens 99.8 %, 0 malformed boards, near-misses 0.1–0.3 % of the
  wrong boards, row/col/box validity inside the 18-cell range. What the seed-2 cells lose is
  the *consensus-easy* boards — 23.4 % (unit 6) / 19.9 % (unit 3) of the boards all five sibling
  cells solve, vs 6.9 % / 5.9 % for the paired K = −0.5 cells — and the cell-accuracy deficit is
  entirely on non-given cells (84.5 vs 88.5–89.9 %): a weaker posterior, i.e. under-training.
- Paired McNemar on shared boards: unit-6 seed 2 −12.7 pt (z = −11), unit-3 seed 2 −7.7 pt
  (z = −8) vs the ±2.2 pt eval CI; the seed main effect is F = 11.2 (p = 0.02) in the fixed-rate
  set and 0.93 (n.s.) in the K = −0.5 set with identical init / data / noise — a seed × regime
  interaction, not luck.
- Not all in one direction: the fixed-rate cells reach a *lower* late loss than their pairs in
  7 of 9 (geo-ratio 0.77–0.97) and beat them on boards in 5 of 9 (all three unit-10 pairs, by
  +3.6 / +2.9 / +11.1 with McNemar z ≥ 2.6); only the two seed-2 cells lose. So the fixed-rate
  regime is a different optimizer, better on average and worse in the tail.
- Which asymmetry: the late onset (8–10k, where the K = −0.5 runs are unclipped and the
  fixed-rate runs remain clipped) points at the 300× loss scale × `gradient_clip_val = 1`
  (the clip factor moves from 0.001 to ~0.1 over training, so Adam does *not* see a constant
  rescale). The 17× input scale is not excluded but drives only the early transient: partial
  traces of the clip-300 control show a *larger* early lag without the clip (1.6–5.8× at
  1.5–3k vs 1.2–2.7× with it), so the transient is input-scale/Adam-eps driven and gone by
  4–5k in every pair. Caveats: the fixed-rate runs' own gradient norms were never logged (the
  clip state is inferred from the ×300 loss scale on the K = −0.5 checkpoints), the eval seed
  equals the training seed (so same-seed solved-set overlaps are confounded by the shared
  Brownian stream), and the control does not isolate an Adam-eps channel.
- Decision rule for the clip-300 control (K = −0.0016667, 3 seeds): seed 2 ≥ 55 % and seed
  std ≤ 5 → clipping is the cause; seed 2 ≤ 50 %, std ≥ 8, same seed order → input scale;
  std ≥ 8 with a different outlier seed → chaotic sensitivity to the regime; in between → mixed.

**Clip control (finished 2026-09-16 ~13:40; `--clip-control`, tags `..._clip300`).** Same cell
(K = −0.0016667, rate 0.01, unit 6, log-linear, LR 3e-4) with `trainer.gradient_clip_val = 300`,
i.e. the clip threshold scaled with the 300× loss so the optimizer sees what the K = −0.5 / rate-3
run sees:

| cell | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---:|---:|---:|---:|
| K = −0.0016667, rate 0.01, clip 1.0 | 61.2 | 47.1 | 68.0 | 58.8 ± 10.7 |
| K = −0.0016667, rate 0.01, **clip 300** | 62.2 | 57.6 | 55.8 | **58.5 ± 3.3** |
| K = −0.5, rate 3, clip 1.0 (reference) | 60.4 | 59.8 | 63.2 | 61.1 ± 1.8 |

- The decision rule's "clipping" branch fires: seed 2 recovers from 47.1 to 57.6, the seed std
  drops from 10.7 to 3.3, the seed ordering of the clip-1 rows disappears (seed 3 goes from best,
  68.0, to worst, 55.8), and the mean is unchanged (58.5 vs 58.8). Paired against the K = −0.5
  reference: +1.8 / −2.2 / −7.4 → −2.6 ± 2.8 (SE), i.e. equivalent within noise in both mean and
  spread once the clip threshold is scaled with the loss. The remaining −7.4 on seed 3 is either
  the 17× `in_proj` input-scale asymmetry or seed noise; n = 3 cannot separate them.
- So the (rate, K) → (c·rate, c·K) symmetry is exact for the problem and holds for training too
  *provided every scale-dependent optimizer setting is transformed with it* — here
  `gradient_clip_val` (and, untested, Adam's ε). Left at 1.0, the 300× larger loss turns Adam into
  a normalized-gradient optimizer for the whole run: same mean, ~3× the seed std, and a higher
  ceiling (the study's three best cells — 68.0, 66.0, 65.8 — are all clip-1 fixed-rate cells).
  That regime may be worth exploiting deliberately (e.g. a small clip value at K = −0.5 / rate 3),
  but that is a new experiment, not a confound.

## Conclusion

1. **The spec's `time_exp_rate = 0.01` is the defect, not the implementation.** Under it only
   ~6% of training heat times fall where the model must use the prompt (unit t ≤ 3 at K = −0.5;
   ~3% at K = −1), 14% sit at the radial clamp with zero gradient (38% at K = −1), and unit
   t < 0.05 is never seen; the plain
   log-linear arm is a coverage failure by construction (0–0.8% boards) and the adaptive arm
   only recovers because its refit moves mass into that window (best 28.7%). Sampler, geometry
   and readout were each verified independently (cheating-posterior test = 100%, SDE marginals
   = exact bridge, trained boundary tables separable, t_max/step-size ablations flat).
2. **With a matched rate HBFM is a strong Sudoku solver:** 61.1 ± 1.8% at K = −0.5 — 1.8× the
   H-FLM reference at that curvature, 2.5× the best spec cell (the argmax of 7 tuned cells; unit
   rate 10 is ~2 SE behind) — and the plain schedule beats the adaptive one at a matched rate by
   17.5 points (K = −0.5) / 8.4 (K = −1). Across K the 20k gaps are within ~2 std except vs
   K = −1, which was only run at unit rate 3.
3. **Two genuine code-level fixes came out of the diagnosis** (both verified): the importance
   weight now carries `|dα_t|` (MDLM's `dα/(1−α)` over the rate; lifts the one failing adaptive cell
   tested, 1e-3 s1, from 0.36 to 0.57 at 5k, is neutral on the good cell, and makes the loss
   invariant to refits), and
   `poincare_cartesian_to_hyperbolic_polar` clamps a few ulps inside the ball so every capped
   state reads back as one finite radius. The repo's refit-50/ema-0.9 recipe and the `naive`
   readout were tested and not adopted (no gain / −5 boards at a matched rate).
4. **Fixed-rate equivalence (user's hypothesis, verified):** holding the spec's rate 0.01 and
   shrinking |K| so that rate/|K| matches reproduces the tuned accuracies — K = −0.001 gives
   63.8 ± 3.6, the best mean of the study; K = −0.0016667 gives 58.8 vs 61.1 for the K = −0.5 /
   rate-3 reference — because (rate, K) enter the problem only through rate/|K| and |K|·t
   (verified numerically end to end). The only asymmetry is optimizer-side: the 300× larger loss
   under `gradient_clip_val = 1` makes Adam gradient-normalized for the whole run, which triples
   the seed spread (±10.7 vs ±1.8); scaling the clip with the loss (`gradient_clip_val = 300`)
   restores it (58.5 ± 3.3). Rule: when trading rate for curvature, transform the clip threshold
   (and ε) with the loss scale, or renormalize the importance weights.
5. **Open items:** (i) the sampler commits to the model's prior at the origin — the SDE adds
   board consistency but the cell-level ceiling is the one-shot prior (86% for the 17% cell);
   a small `noise.eps` (so training covers t → 0) or best-of-n decoding are the natural next
   levers; (ii) eval seeds equal training seeds, which correlates same-seed cells; (iii) the
   confirmation grid is 3 seeds per cell — the K = −0.5 vs −0.25/−2 gap (61 vs 56) is ~2 std.

## Artifacts

- Tables: `python report.py` (spec sweep + `_rate-3`), `python report_tune.py` (probe grid +
  20k confirmation). Cells: `outputs/claude_test_hbfm_sudoku/<tag>/{checkpoints,eval/results.json}`.
- Diagnostic scripts, logs and tables: `experiments/claude_test_hbfm_sudoku/logs/diag/`
  (coverage arithmetic, board lenses, adaptive-refit simulation, wandb curves, GPU
  denoising-vs-heat-time / cheating-posterior / checkpoint-progression logs, the F2 A/B scorer),
  `logs/hyp/` (rate-1/3 and F1/F2 5k probe job scripts and logs), `logs/naive/` (readout probes
  and `score.out`). Copied from the session scratchpad; the 5k probe checkpoints themselves were
  not kept.
