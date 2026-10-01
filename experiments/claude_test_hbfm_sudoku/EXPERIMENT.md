# claude_test_hbfm_sudoku — HBFM on hard Sudoku: LR × seed

Spec: `setup.md` in this folder. Sweep: `sweep.py` (orchestration only; calls
`scripts/train/sudoku/hbfm.sh` + `scripts/sample/sudoku/hbfm.sh`). Report: `report.py`.

## Hypothesis

The hyperbolic bridge DLM (`algo.HyperbolicBoundaryFM`: Brownian bridge on the product
manifold `(H^3_{-0.5})^3` toward each word's boundary point, denoising cross-entropy of the
horosphere-corrected posterior, heat time from the noise schedule via
`t = -log(1 - alpha_t) / 0.01`, clamped at the radial sampler's ceiling ~97) trains
stably on hard Sudoku and, sampled with its bridge
SDE (`samplers.HBFMSampler`, exact posterior-mean drift, 180 Euler-Maruyama steps over
`t in [0, 0.5]`, greedy last step), reaches a board accuracy comparable to the sibling
H-FLM runs (`hflm_curv_sudoku`: 29.4% hard at K=-1). We measure the learning-rate
sensitivity, the seed variance, and whether the `AdaptiveSchedule` (which refits the
heat-time proposal from the loss profile, `noise=log-linear-adaptive`) helps on top of
the very wide rate-0.01 exp proposal.

## Design

| | |
|---|---|
| data | Sudoku hard (30 clues), 48k train / 2k val, data seed 42 |
| model | tiny-hyperbolic-dit 512/8/8 (~28.6M), `model.embed_dim=9` (lifted by `in_proj`), init ngpt |
| geometry | `prod_factor_dim=[3,3,3]`, `prod_factor_gaussian_curvature=[K,K,K]` → 3 factors of H^3; K = −0.5 (setup.md) and K = −1 (an earlier submission made before the spec changed, kept as a curvature comparison) |
| objective | denoising CE (`forward_type=horosphere`), `algo.time_exp_rate=0.01` (physical; the scripts take `UNIT_PROPOSAL_RATE = 0.01/|K|`), `readout_precision=float32` |
| training | 20k steps, batch 256 (1 GPU × 256), bf16, EMA 0.9999, AdamW (wd 0, betas 0.9/0.999, eps 1e-8), grad-clip 1.0 |
| noise | `log-linear` (ada-0) vs `log-linear-adaptive` (ada-1: buffer 12800, refit every 500 steps after 1000 warmup, 1000-point grid) |
| grid | K ∈ {−0.5, −1} × LR ∈ {3e-4, 5e-4, 1e-3} × noise ∈ {ada-0, ada-1} × seed ∈ {1, 2, 3} = 36 cells |
| eval | `mode=sudoku_eval` on the 2000 validation boards: exact velocity, `top_k_velocity=-1`, 180 steps, `t_max = 3.0/|K|` physical (unit time 3, calibrated below), greedy last step; metric = exact-match board accuracy |
| outputs | `outputs/claude_test_hbfm_sudoku/lr-{lr}_ada-{0,1}_k{K}_seed-{seed}/{checkpoints,eval/results.json}` (the `lr-*_seed-*` dirs are a first, superseded submission at rate 20 / 32 factors without the adaptive axis) |

Success criterion: all cells finish with finite loss; the LR × noise (× K) ranking is reported
with mean ± std over the 3 seeds (2000 boards → ±2.1 pt 95% CI per cell at mid-range accuracy).

## Amendments after the first 36 cells (2026-09-16)

- **Objective weight.** `HyperbolicBoundaryFM.nll` now multiplies the heat-time importance weight
  `1/(rate·u)` by `|dα_t|`, the change of variables from the schedule's uniform time (MDLM's
  `dα/(1−α)` over the rate). Without it the loss rescaled at every adaptive refit; the A/B in
  RESULTS.md (H2) shows it lifts the failing adaptive cell's prior at 5k steps from 0.36 to 0.57.
  All 36 rate-0.01 cells were trained before this change; the rate-3 cells after it.
- **Rate follow-up.** `sweep.py` `EXTRA_RATES = ['3']` adds the LR-3e-4 column at
  `algo.time_exp_rate = 3` (tags `_rate-3`, both K, both schedules, 3 seeds, 20k steps) after
  5k-step probes at rate 1 / 3 reached 27.3% / 30.4% boards vs 0% at the spec's 0.01 (H1).
- **Adaptive knobs.** `scripts/{train,sample}/sudoku/hbfm.sh` expose `ADA_REFIT_EVERY` /
  `ADA_EMA` (defaults = the yaml's 500 / 0.0; the repo's 50 / 0.9 recipe did not help at rate 0.01,
  RESULTS.md H2).
- **Curvature × rate tuning.** `sweep_tune.py` runs a 5k-step probe grid (K ∈ {−0.25, −0.5, −1, −2}
  × unit rate `time_exp_rate/|K|` ∈ {1, 3, 10}, seed 1, tags `tune_k{K}_ur{r}_seed-1`) and, with
  `--confirm`, 3 seeds × 20k steps at the cells the grid left open (tags `tune20k_*`);
  `report_tune.py` prints both tables. Result (RESULTS.md): K = −0.5 at physical rate 3 (unit 6),
  log-linear, LR 3e-4 → 61.1 ± 1.8%; the 5k probe ranking does not predict the 20k ranking.

## Sampling horizon

`t_max` is not in the spec. Calibrated with the Bayes-optimal posterior (zero residual,
uniform prior, V=12, ngpt-scale random boundary table) on bridge states drawn by
`poincare_bridge_prod`: on `(H^3)^3` the posterior's CE / argmax accuracy is
1.96 / 0.32 at t=0.1, 0.69 / 0.76 at t=0.5, 0.22 / 0.93 at t=1, 0.027 / 0.99 at t=2 and
0.000 / 1.00 at t=3 (the 32-factor manifold of the first submission was resolved by
t=0.25, hence the scripts' old 0.5 default). `UNIT_T_MAX=3.0` therefore ends the SDE where
even the context-free posterior is one-hot; 180 uniform steps give dt = 0.017.

## Compute

18 jobs per curvature × 1 GPU (`thickstun,desa`, excluding `desa-compute-01`), 8 CPUs, 64 GB, 12 h limit.
Measured 2.1–3.4 it/s → ~2–2.6 h per cell (20k steps at batch 256 on an A5000 / RTX 6000
Ada) + ~3 min of sudoku_eval; cells run in parallel as GPUs free up.

---

# Round 3 (2026-09-17) — the heat-time PROPOSAL: range × shape

`setup.md` was revised: the axes are now `forward_type` ∈ {naive, horosphere}, `time_exp_rate`,
the global curvature, `time_conversion_mode` ∈ {exp, trunc_exp, unif} and `time_range_upper_bound`
(LR only from {3e-4, 5e-4, 1e-3}); everything else is pinned. Sweep: `sweep_mode.py`, report:
`report_mode.py` (tags `mode-*`, marker block `report3` in RESULTS.md).

## What the two new modes actually change

`HyperbolicBoundaryFM.time_conversion` (algo.py) turns the schedule's noise fraction
`u = 1 − α_t` into the bridge heat time `t` and the importance weight `1/q(t)`; `nll` multiplies
that weight by `|α′_t|`. With `noise=log-linear` the schedule draws `t′ ~ U[1e-3, 1]`
(`trainer_base._sample_t`, `training.sampling_eps=1e-3`, antithetic) and `u = (1−ε)t′`, so:

| mode | trained heat-time range | density on it | weight |
|---|---|---|---|
| `exp` | `[0, ln(1/ε)/rate] = [0, 6.9/rate]` — bounded by the ε floor, not by the mode | `rate·e^{−rate·t}` | `1/(rate·u)`, spans ~3 decades |
| `trunc_exp` | `[0, ub]` exactly | exp(rate) renormalized on `[0, ub]` | `Z/(rate·e^{−rate t})`, `Z = 1−e^{−rate·ub}` |
| `unif` | `[0, ub]` exactly | flat | `ub` (constant) |

So the grid is really **range × shape**: `exp` at rate *r* is `trunc_exp(r, 6.9/r)` up to the
normalizer `Z ≈ 1`; the anchor (rate 3) trains on `t ∈ [0, 2.3]`. Two things can therefore be
tested separately for the first time: the *range* of heat times the model sees, and the *shape*
of the density on it — plus, as a side effect, the weight's dynamic range, which matters because
`trainer.gradient_clip_val` is pinned at 1.0 and round 2 showed the seed spread tracks the loss
scale (RESULTS.md, "Clip control": ±10.7 at clip 1.0 → ±3.3 at clip 300 for the same cell).

Anchor for every comparison (finished in round 2, not re-run): `exp`, physical rate 3
(unit rate 6), K = −0.5, log-linear, LR 3e-4, `forward_type=horosphere`, sampler `t_max` 6.0,
180 steps → **61.1 ± 1.8 %** boards (seeds 1/2/3 = 60.4 / 59.8 / 63.2).

## Hypotheses

- **H1 (range).** Accuracy is set by how much of the decision zone the proposal covers. The
  Bayes-optimal posterior on `(H^3_{−0.5})^3` has CE 1.96 / 0.22 / 0.027 / 0.000 at physical
  t = 0.1 / 1 / 2 / 3 ("Sampling horizon" above), so the zone is `t ∈ [0.1, 2]`. Predict:
  ub = 2.3 ≥ ub = 6 > ub = 1 (ub = 1 clips the resolved end, ub = 6 spends 2/3 of the draws
  on states that are already one-hot).
- **H2 (shape).** At a matched range, a flat proposal beats an exponential one because its
  weight is constant, so the loss scale is stable under `gradient_clip_val=1`: predict
  `unif(2.3) ≥ trunc_exp(3, 2.3) ≈ exp(3)`.
- **H3 (the spec's rate).** `time_exp_rate = 0.01` fails as `exp` (0.4 % boards — the proposal's
  mass sits at t ≈ 100) but is harmless once truncated: `trunc_exp(0.01, 2.3)` is uniform on
  `[0, 2.3]` to within 1 %, so it should land on `unif(2.3)`, not on 0.4 %.
- **H4 (curvature).** Only unit time `|K|·t` is physical, so at fixed unit rate 6 the K axis
  measures what is left: the DiT input scale `R ∝ 1/√|K|`. Predict a weak, non-monotone effect.
- **H5 (forward_type).** `naive` drops the Busemann log-density prior from the readout. At 5k
  steps it lost 25.4 vs 30.4 (K = −1, rate 3); predict it still loses at 20k, but this is the
  cell that decides whether the repo's new `forward_type: naive` default is right for sudoku.
- **H6 (LR).** 3e-4 won at rate 0.01, where the loss was 300× larger. At rate 3 the loss is O(1),
  so 5e-4 / 1e-3 may now be trainable.

## Design (stage 1: 13 cells, seed 1)

| block | cells |
|---|---|
| range × shape | `unif` ub ∈ {1, 2.3, 6}; `trunc_exp` rate 3, ub ∈ {1, 2.3, 6}; `exp` rate 6.9 (range 1) |
| spec rate | `trunc_exp` rate 0.01, ub 2.3 |
| forward | `naive` at the anchor |
| LR | 5e-4, 1e-3 at the anchor |
| curvature | K = −0.25 (rate 1.5) and K = −1 (rate 6), both at unit rate 6 |

`sampler.t_max` = ub for the bounded modes (`_validate_configuration` rejects `t_max > ub`) and
6 for `exp` at K = −0.5 — i.e. unit time 3 throughout, so the two K cells use `t_max = 3/|K|`
(12 and 3) and every cell takes the same 180 steps of the same unit `dt`. That makes "shorter horizon" a confound of every bounded cell, so it is measured
separately and for free: `sweep_mode.py --anchor-tmax` re-evaluates the three finished anchor
checkpoints at `t_max` ∈ {1.0, 2.3} (eval only, ~10 min each).

## Amendment after the design review (2026-09-18 01:10)

A 10-agent review (recon on round 2 + `eflm_rescale_auto_sudoku`, a numerical calibration of every
proposal, a code audit, three independent grid proposals and three judges) confirmed the grid and
corrected two labels, then added three cells:

- **`trunc_exp(rate 3, ub 6)` is numerically identical to the anchor.** `rate·ub = 18 ≫ 6.9`, so the
  `noise.eps` floor truncates before `ub` does (verified against `time_conversion` itself at the real
  draw `u ~ U[1e-3, 0.999]`: max |Δt| = 5.1e-6, max relative Δw = 1.5e-5). That cell is therefore the
  **anchor-reproduction control** — it re-measures 60.4 % under the round-3 harness (Turing, batch
  128 × 2 accumulation) — not a range point.
- **`trunc_exp(rate 0.01, ub 2.3)` is a duplicate of `unif(2.3)`** (`rate·ub = 0.023`, max relative
  Δt ≈ 1.1e-2). Kept: it answers the spec's literal `time_exp_rate = 0.01` and doubles as an internal
  consistency check on the two code paths.
- **`time_exp_rate` is unused when `mode=unif`** (`algo.py` `time_conversion`), so the rate in those
  cells' tags is inert.
- **Only the `ub = 6` family is free of the horizon confound** (its `t_max` = 6 = the anchor's).
  The headline mode claim is made there, so the shape ladder is run at fixed range and fixed horizon:
  mean trained `t` = 0.33 (`trunc_exp` 3 ≡ anchor) → 0.95 (`trunc_exp` 1) → 1.6 (`trunc_exp` 0.5) →
  3.0 (`unif` 6). Added: `trunc_exp(1, 6)` and `trunc_exp(0.5, 6)`.
- Added `exp` rate 4 (unit rate 8), the only unmeasured point between unit 6 (61.1 %) and unit 10
  (57.9 %).
- Added the long arm of the horizon control, `--anchor-tmax 4,9,12` (eval only): the anchor is
  trained only to `t ≤ 2.3` but sampled to 6, so whether a longer horizon helps is free to measure.

### What `eflm_rescale_auto_sudoku` (the study `setup.md` points at) predicts

Its "auto" arm is `noise=autonomous`: `tau` uniform on `[0, tau_max]`, constant loss weight — which
is **structurally identical to HBFM's `unif`**, while plain log-linear is the twin of `exp`. Its
results, on this same hard-Sudoku task:

- The horizon bought **+11.1 / +13.0 / +20.7** points over the log-linear control at R = 5 / 8 / 16,
  and the optimum tracked the closed-form context-free decode time `tau*(R)` within one grid step
  while `tau*` swept 2.7x. Overshooting is the expensive direction (R=16: `tau` 1 → 24.3 %,
  2 → 5.9 %, 4 → 3.5 %, 7 → 0.4 %).
- **Weight dynamic range is the stability variable**: ~360x → best cells, ~1e3 → mixed / one seed
  diverged, ~5e4 → collapse. HBFM's `exp` has dynamic range **exactly 1000 at every rate**
  (it is `1/sampling_eps`) — the "mixed" rung. `unif` has dynamic range **1**.
- Schedule invariance fails at a fixed step budget: the SNR-matched twin of log-linear measured
  4.2 vs 30.4. "At 20k steps the proposal is a training-mass allocator, not a variance-reduction
  device" — which is why the ESS-optimal proposal is not the accurate one.
- HBFM's own rate sweep has unknowingly re-run that `tau_max` sweep: horizon/`tau*` of
  8.3 / 2.8 / 1.38 / 0.83 gave 31.3 / 53.4 / **61.1** / 57.9 %, the same curve shape and the same
  peak at ratio ≈ 1.

Measured for this manifold (`logs/diag/coverage.json`): context-free Bayes accuracy is 0.50 at unit
`tau` 0.212, **0.90 at `tau* = 0.833`**, 0.99 at 1.78; 90 % of `∫CE dtau` sits below `tau` 0.922.

So the two hypotheses make *incompatible* predictions for the best `unif` horizon, and both are now
cells: `unif(ub = 1.67)` (= unit `tau*` 0.833, EFLM's rule) versus `unif(ub = 0.44)` (= unit 0.22,
the horizon that allocates training mass like the incumbent `exp` rate 3). They cannot both win.

### Counter-evidence, stated before the results (pre-registered)

`experiments/trunc_ada_sudoku` truncated H-FLM's **clean** end — the same direction as `ub` here —
and collapsed monotonically: hard Sudoku 46.2 % (no truncation) → 35.8 → 0.9 → 0.0. Its conclusion:
"teacher-forced zero-loss does NOT mean a region is dispensable: the loss is measured on true
geodesic interpolants, while the sampler visits self-generated states with accumulated error; the
near-clean range must be trained for trajectories to snap back onto the data manifold", with the
recommendation "do NOT truncate below ~0.9 at K <= -0.5". HBFM is a bridge rather than a geodesic
interpolant, so the argument need not carry — but if the small-`ub` cells collapse rather than merely
underperform, that is the explanation, and the `ub` ladder (0.44 / 1 / 1.67 / 2.3 / 6) is exactly the
measurement that separates the two stories.

Stage 1 is therefore **18 cells**: two waves of the node's 8 GPUs plus a short third.
If a `unif` cell wins, stage 2 must re-tune LR on it (its constant weight changes the loss scale by
~100x versus `exp`, and EFLM saw `gradient_clip_val` stop binding and a seed diverge at 1e-3).

Stage 2 (`CONFIRM` + `--confirm`): seeds 2 and 3 for the ≤ 3 cells whose seed-1 accuracy beats the
anchor's seed-1 (60.4 %), so the winner is reported as mean ± std over 3 seeds like round 2.

## Compute

2080 Ti only (`setup.md`), i.e. the 8 GPUs of `desa-compute-01` — `thickstun-compute-01` is RTX
6000 Ada, so `--partition=desa,thickstun --gres=gpu:nvidia_geforce_rtx_2080_ti:1` resolves to
`desa` alone; the rest of the cluster is running the TinyStories HBFM sweep. Measured on a
2080 Ti (11 GB): batch 256 in one micro-batch **OOMs**, so a cell runs `PER_GPU_BS=128` ×
2 accumulation steps (identical optimizer batch of 256; `_sample_t` strata are split across the
two micro-batches and re-joined by accumulation). Measured 1.53 s per optimizer step with
`forward_type=naive` and 1.55 s with `horosphere` (the V=12 readout costs nothing here) →
**≈ 8.6 h per cell** + ~15 min of `sudoku_eval`, 4 CPUs / 28 GB each so all 8 GPUs of the node are
usable at once. 13 cells ≈ 2 waves ≈ 18 h; stage 2 (≤ 6 cells) ≈ 9 h.

Success criterion: every cell finishes with finite loss and an `eval/results.json`; the range ×
shape block is read as a 2-factor table; a new best is only claimed if its 3-seed mean beats
61.1 ± 1.8 % by more than the seed spread.
