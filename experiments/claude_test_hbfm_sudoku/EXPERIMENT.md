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
