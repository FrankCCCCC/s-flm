# eflm_rescale_auto_sudoku — autonomous clock + VDM SNR-weighted CE (EFLM, sudoku hard)

Two changes to rescaled E-FLM, both from `slides/jul09_2026` (+ `papers/
Variational Diffusion Models.pdf`), evaluated on the grid of
`experiments/eflm_rescale_sudoku` so the log-linear runs there are the
control:

1. **Autonomous flow** (`noise=autonomous`, `noise.tau_max`). A Euclidean
   bridge conditioned on `X_T = y` has the singular drift `(y - X_t)/(T - t)`.
   Under the clock `tau = -log((T - t)/T)` it becomes **time-invariant**,
   `v(X) = y - X`, at the price of a noise level that only vanishes as
   `tau -> infinity`. For E-FLM's interpolant `x_t = alpha_t e + (1-alpha_t) z`
   that clock is exactly an exponentially decaying noise fraction:

       b_t := 1 - alpha_t = exp(-tau),   tau = tau_max * (1 - t)

   so `d x / d tau = y - x` holds identically, a uniform `t` grid is uniform in
   `tau`, and the sampler's Euler step `(b_t - b_s)/b_t = 1 - exp(-d_tau)` is
   the **same at every step** (`tests/test_autonomous_flow.py`). No sampler
   change is needed: the existing EFLM step is already the exact bridge move,
   the schedule is the whole mechanism. `tau_max` is the truncation the "target
   reached only as tau -> infinity" price forces on us.

2. **SNR-weighted CE** (`algo.snr_weighted_ce`). E-FLM trains on an unweighted
   CE, which is *not* a likelihood bound for a Gaussian interpolant. VDM Eq. 16
   gives the continuous-time diffusion loss as
   `-1/2 E_{t~U(0,1)}[SNR'(t) ||x - x_hat||^2]`; substituting the categorical
   reconstruction error (the CE) for `||x - x_hat||^2` gives the weighted CE

       w(t) * CE,   w(t) = -SNR'(t)/2 = (1 - b_t)|b_t'| / b_t^3,
       SNR(t) = (1 - b_t)^2 / b_t^2.

   (`algo.snr_weight`, checked against autograd through `SNR(t)`.)

The two interact: in `tau` time the weight is `w = tau_max (1 - b)/b^2`, one
power of `b` tamer than the log-linear clock's `(1 - b)/b^3` — the autonomous
clock is the variance-reducing change of variables for this weight.

## Hypotheses

**H1 (clock changes the implicit weighting).** With plain CE the schedule shape
*is* the objective: `t ~ U(0,1)` under the autonomous clock makes `b`
log-uniform, moving training mass toward the near-clean end. Random-codebook
analysis (`experiments/eflm_rescale_sudoku`, `alpha_star_euclidean`, V=12,
d=512, delta=0.1 -> C=3.28) puts the decode point at `b*(R) = 1/(1 + C/R)`:

| R    | b*(R) | tau*(R) = -log b* | steps before decode, log-linear | ... = tau*/tau_max |
|------|-------|-------------------|--------------------------------|--------------------|
| 5    | 0.604 | 0.505             | 0.396                          | 0.505 / tau_max    |
| 8    | 0.709 | 0.344             | 0.291                          | 0.344 / tau_max    |
| 16   | 0.830 | 0.187             | 0.170                          | 0.187 / tau_max    |

so the autonomous clock matches log-linear's step budget around the decision at
**tau_max ~ 1.1-1.3** and starves it (3-7x fewer pre-decode steps) at
tau_max >= 4. Prediction: with plain CE, accuracy peaks at small tau_max and
decays with tau_max, and the effect is strongest at R=16 (earliest decode,
where the log-linear baseline is worst: 18.7%).

**H2 (weighting restores the bound, and makes the clock ~irrelevant).** VDM
§5.1: the continuous-time bound depends on the schedule only through
`SNR_min/SNR_max`. `tau_max` sets `SNR_max = ((1-e^-tau_max)/e^-tau_max)^2`:

| tau_max | 0.5  | 1.0  | 2.0  | 3.0 | 4.0  | 7.0    | log-linear (eps=1e-3) |
|---------|------|------|------|-----|------|--------|-----------------------|
| SNR_max | 0.42 | 2.95 | 40.8 | 364 | 2873 | 1.2e6  | 1.0e6                 |

**tau_max = 7 is the SNR-matched counterpart of log-linear.** So under the
weighted CE, `auto tau=7 + snr` and `ll + snr` optimize the *same* objective
and should land within seed noise of each other, differing only in MC-estimator
variance; under plain CE they should not. This is the sharpest test in the
sweep and needs no new baseline (the `ll` arm is run here too).

**H3 (does the bound help at all?).** Weighting up-weights high-SNR samples by
`1/b^2`, where the CE is near 0 for a rescaled codebook — the product peaks at
the decode transition, so the SNR weight is a *learned-free* focusing mechanism
on the timesteps that matter. Prediction: `snr` >= `ce` at the tau_max values
that starve the transition (>= 2), and roughly neutral at tau_max ~ 1 where the
clock already allocates well. Measured at init: weighted loss 408 vs CE 2.32
(step 99, R=8, tau=3) collapsing to 0.18 by step 299 — stable, no clipping
pathology.

## Design

- Data: sudoku **hard** (30 clues), 48k train / 2k val, seed 42.
- Model: `tiny-sphere-dit` (512 wide, 8 blocks, 8 heads, ~28.6M), `init=ngpt`
  (N(0, 1/sqrt(d)), std 0.0442), seq len 180.
- Training: 20k steps, bs 256, bf16, EMA 0.9999, AdamW (wd 0, betas
  (0.9,0.999), eps 1e-8, clip 1.0), `invert_time_convention=false`.
- Eval: `mode=sudoku_eval`, 180 sampling steps, exact velocity,
  `top_k_velocity=-1`, greedy last step, 2000 val puzzles.
- Axes (`sweep.py`): `clock x arm x R x tau_max x weight x LR x seed`
  - clock: `auto` (noise=autonomous) / `ll` (noise=log-linear, control)
  - arm: `naive` (fixed schedule) / `ada` (adaptive noise schedule)
  - R = `algo.rho_min = rho_max` in {5.0, 8, 16}
  - tau_max in {0.5, 1, 2, 4, 7} (pilot) -> single value (main)
  - weight: `ce` / `snr` (`algo.snr_weighted_ce`)
  - LR in {3e-4, 5e-4, 1e-3}; seed in {1, 2, 3}

**Stage 1 — tau_max pilot (24 cells, submitted 2026-07-26).**
`--clocks auto ll --arms naive --taus 0.5 1 2 4 7 --weights ce snr
 --rhos 5.0 16 --lrs 1e-3 --seeds 1`
Picks tau_max (H1), tests the SNR weight in isolation on both clocks (H2/H3),
and re-runs the log-linear control in this checkout.

**tau_max selection rule (pre-registered, written before the pilot finished).**
Take the tau_max maximising the mean over the four naive pilot cells
{R=5, R=16} x {ce, snr} — robust across R and weight rather than the single
best cell; ties (< 2 points, i.e. inside seed noise) break toward the *smaller*
tau_max, which has the tamer weight and the cheaper SNR range. The four `ada`
pilot cells (R=16, tau in {1, 7}) are a cross-check: if the `ada` arm's
preference contradicts the naive arm's, run the main grid's `ada` half at the
tau_max the `ada` cells prefer and say so in RESULTS.md.

**tau_max chosen: 0.5** (submitted 2026-07-26 21:30). It wins the pre-registered
rule on every completed pilot cell — 4-cell mean 19.9 vs 18.2 (tau=1), 12.0
(tau=2), 7.4 (tau=4), 5.0 (tau=7) — and wins decisively on the `ce` cells
(41.3 / 35.6 at R=5 / R=16 vs 29.2 / 24.3 at tau=1). *Deviation:* the
tau in {0.1, 0.25} cells were preempted and requeued, so the main grid was
launched before they finished rather than idling ~15 GPUs for another ~2h;
theory says they cannot win, since they stop the flow at b_min = 0.78 / 0.90,
*before* R=5's decode point b* = 0.604 (R=16's b* = 0.83 is the one case they
could clear). Their numbers go into RESULTS.md when they land, and the sweep is
idempotent if a re-run at another tau is warranted.

**Stage 2 — main grid (108 cells).** At tau_max = 0.5:
`arm{2} x weight{2} x R{3} x LR{3} x seed{3}`, the `setup.md` grid with the
loss-weighting arm added. 3-seed means are the headline numbers.

## Baseline (control, `experiments/eflm_rescale_sudoku`, mean of 3 seeds, %)

| arm   | R=5  | R=8  | R=16 |
|-------|------|------|------|
| naive (log-linear) | **39.6** (lr 1e-3) / 40.0 (5e-4) | 32.1 | 23.6 |
| ada                | **53.9** (lr 1e-3) | 48.8 | 47.6 |

(best LR per cell; seed spread is +-5-8 points, so a real effect needs to be
bigger than ~5 points on the 3-seed mean.)

## Success criteria

1. **H1**: monotone accuracy-vs-tau_max ordering in the pilot, peaking at
   tau_max <= 2, with a larger gap at R=16 than at R=5.
2. **H2**: `auto tau=7 + snr` within seed noise of `ll + snr`, while
   `auto tau=7 + ce` is clearly worse than `ll + ce` (schedule invariance holds
   only for the weighted bound).
3. **H3**: at the chosen tau_max, `snr` >= `ce` on the 3-seed mean of the main
   grid; and best autonomous cell >= the 39.6 / 53.9 log-linear baselines
   (naive / ada).
4. **Sanity**: no divergence/NaN from the `1/b^3` weight; `ll + snr` (the
   untamed weight, SNR_max ~ 1e6) is the stress case.

## Compute

- 1-GPU SLURM jobs, `thickstun,desa` (exclude `desa-compute-01`), 2 cpus, 16G,
  6h walltime. ~2.5-3h/cell (20k steps at ~2.5 it/s + 2000-puzzle eval).
- Nodes are shared with other partitions: ~10-14 of my jobs run concurrently,
  so the pilot is ~2 waves (~6h) and the main grid ~9 waves (~20-24h).
- Outputs: `outputs/eflm_rescale_auto_sudoku/{tag}/` with `eval/results.json`,
  `tag = eflmra_{clock}-{arm}_r-{R}_tau-{tau}_w-{weight}_lr-{lr}_d-hard_rs{seed}`.
  Checkpoints kept (loss-geometry L(t) analysis is the follow-up).

## Not tested here (noted)

- The model still conditions on `sigma = -log(alpha_t)`; under the autonomous
  clock most of the schedule sits at `alpha ~ 1`, where the sinusoidal timestep
  embedding barely resolves changes. That is *consistent* with autonomy (the
  drift needs no clock) but a `time_conditioning=false` ablation would test the
  strong form of the claim.
- Substituting CE for `||x - x_hat||^2` in VDM Eq. 16 is a parameterization
  choice, not an identity; the alternative (weight `gamma'(t)/2`, the
  epsilon-space form of Eq. 17) is untested.

## Run

    python experiments/eflm_rescale_auto_sudoku/sweep.py --dry-run
    # stage 1
    python experiments/eflm_rescale_auto_sudoku/sweep.py --clocks auto ll \
        --arms naive --taus 0.5 1.0 2.0 4.0 7.0 --rhos 5.0 16 \
        --lrs 1e-3 --seeds 1
    # stage 2 (after picking tau_max; TAUS default in sweep.py)
    python experiments/eflm_rescale_auto_sudoku/sweep.py
    # collect
    python experiments/eflm_rescale_auto_sudoku/analyze.py
