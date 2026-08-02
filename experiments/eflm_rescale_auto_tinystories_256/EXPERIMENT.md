# eflm_rescale_auto_tinystories_256 — autonomous clock on TinyStories (seq 256)

Carries the sudoku result (`experiments/eflm_rescale_auto_sudoku`) to language:
rescaled E-FLM on the **autonomous clock**, crossed with {w/, w/o Ada} x
{w/, w/o Trunc} over six pinned embedding norms R.

## The clock, and why tau_max *is* the truncation

The autonomous re-parameterisation (slides/jul09_2026) turns the singular
bridge drift `(y - X_t)/(T - t)` into the time-invariant `v(X) = y - X` by
running the clock `tau = -log((T-t)/T)`. For E-FLM's interpolant that is
exactly an exponentially decaying noise fraction,

    b_t := 1 - alpha_t = exp(-tau),   tau = tau_max * (1 - t),

so a uniform `t` grid is uniform in `tau` and every Euler step advances the
flow by the same `d_tau` (`noise_schedules.Autonomous`, tests in
`tests/test_autonomous_flow.py`). The flow reaches the target only as
`tau -> infinity`, so **tau_max is the truncation on this clock**, and it maps
onto the Eq.-17 bound the log-linear arms used:

    tau*(R) = -log b*(R),   b*(R) = 1 - alpha_star_euclidean(V=50257, R)
                                  = 1 / (1 + C/R),   C = 5.2575

| R | alpha*(R) | b*(R) = flow-time t* | **tau*(R)** |
|---|---|---|---|
| 0.5 | 0.9132 | 0.0868 | **2.4436** |
| 1   | 0.8402 | 0.1598 | **1.8338** |
| 5   | 0.5126 | 0.4874 | **0.7186** |
| 8   | 0.3966 | 0.6034 | **0.5051** |
| 16  | 0.2473 | 0.7527 | **0.2841** |
| 28  | 0.1581 | 0.8419 | **0.1721** |

**Verified against the schedule code**: `Autonomous(tau_max=tau*(R))` and
`TruncatedScheduleWrapper(LogLinear, alpha_max=alpha*(R))` have the same
`b_min`/`b_max` at all six R (max discrepancy 8e-4, the `1-eps` factor). So the
w/ Trunc arms are a **matched-endpoint** comparison against the log-linear
`trunc` / `trunc_ada` arms of `experiments/eflm_rescale_tinystories_256`: same
SNR range, different traversal. That is exactly the VDM 5.1 invariance
question, which sudoku answered negatively at a fixed step budget.

w/o Trunc uses `tau_max = -log(1e-3) = 6.9078`, the log-linear schedule's own
training floor (`training.sampling_eps`), so it spans the same range as the
untruncated log-linear arm.

## Hypotheses

**H1 (clock helps the fixed-schedule arm).** On sudoku the clock was worth
+11 to +21 points on the naive arm and its benefit grew with R (large R decodes
earliest, so the log-linear clock wastes the most steps there). Here R=28 has
b* = 0.84, i.e. log-linear spends ~84 % of its steps after the decision.
Prediction: `auto_trunc` > log-linear `trunc` on GenPPL, with the gap growing
in R.

**H2 (clock and adaptive scheduler are substitutes).** Sudoku: combining them
gained nothing (-3.8 / +1.8 / +4.4). Prediction: `auto_trunc_ada` ~
`auto_trunc`, and the prior arm ordering (trunc_ada < ada ~ trunc) compresses.

**H3 (truncation still matters, and more than on sudoku).** Untruncated
tau_max = 6.9 was catastrophic on sudoku (0.4 % at R=16). Prediction: `auto` /
`auto_ada` (w/o Trunc) lose badly to their truncated counterparts, most at
large R — the clock is only useful when truncated near the decode point.

**H4 (small R still wins on language).** The prior TinyStories sweep found
GenPPL monotone in R (R=1 < 8 < 28), the opposite of sudoku accuracy, which it
attributed to generation quality favouring a smooth, early transition with many
usable denoising steps. The autonomous clock changes *how* those steps are
allocated, so this is the interesting interaction: if the clock's step
reallocation is what matters, large R should improve most and the monotonicity
should flatten.

## Design

- Data: TinyStories, 475M train / 5M val (seed 42), seq **256**.
- Model: `small-sphere-dit` (768 wide, 12 blocks, 12 heads), `init=ngpt`
  (N(0, 1/d)), norms pinned via `algo.rho_min = rho_max = R`.
- Training: 30k steps, global batch 512 (4 GPUs x 32 x accum 4), bf16,
  EMA 0.9999, AdamW lr 3e-4, wd 0, betas (0.9, 0.999), eps 1e-8, clip 1.0,
  plain CE (`algo.snr_weighted_ce=false`; see below), 1 seed.
- Eval: `ppl_eval` (valid PPL, the flow bound) + `sample_eval` (GenPPL via
  gpt2-large retokenisation, entropy, samples), **exact velocity,
  top_k_velocity = 1, 180 steps, greedy last**.
- Grid: `arm{4} x R{6}` = **24 cells**
  - `auto` = autonomous, tau_max 6.9078 (w/o Trunc, w/o Ada)
  - `auto_ada` = + adaptive noise schedule
  - `auto_trunc` = tau_max = tau*(R)
  - `auto_trunc_ada` = tau_max = tau*(R) + adaptive

**Loss weighting: the base grid runs plain CE**, plus one added arm.
Sudoku showed the VDM Eq.-16 SNR-weighted CE only helps when the weight's
dynamic range stays <= ~1e3, which holds only at short horizons; at V=50257
tau*(1) = 1.83 already gives ~3e4 and the untruncated horizon ~1e9.

- `auto_snr` (added 2026-07-29 on request): untruncated + SNR-weighted CE,
  6 cells, submitted at low priority. **Pre-registered prediction: WORSE than
  `auto`, not better.** On the untruncated horizon the weight
  `w = tau_max (1-b)/b^2` is ~1e6x larger at the clean end (b = 1e-3, w = 6.9e6)
  than at the decode point (b*(8) = 0.603, w = 7.5), i.e. it up-weights exactly
  the post-decode region that already explains the untruncated arm's failure
  (its GenPPL follows `9.2 * (pre-decode step share)^-0.637`, R^2 = 0.996). So
  it should compound the failure. Sudoku agrees: at tau_max = 7 the snr arm
  scored 0.0-4.2 % vs 0.4-15.2 % for CE.
- Untested, and the more promising rescue: VDM's epsilon-parameterisation
  (Eq. 17) weight `gamma'(t)/2 = tau_max/(1-b)`, which runs the other way
  (69 at the pure-noise end, 17 at b*, 7 deep post-decode) and so concentrates
  training where an untruncated flow is starving. Needs a mode flag on
  `algo.snr_weight`.

## Reference numbers (log-linear, `experiments/eflm_rescale_tinystories_256`)

| arm | R=1 | R=8 | R=28 |
|---|---|---|---|
| ada | 12.57 | 17.87 | 20.35 |
| trunc | 12.78 | 16.72 | 18.39 |
| trunc_ada | **10.99** | 15.11 | 15.86 |

(GenPPL, lower better; entropy 3.69-3.93 everywhere. Raw-norm naive E-FLM
baseline: GenPPL 34.58 / entropy 3.67.) Valid PPL is **not** comparable across
arms — truncation changes the bound's integration range.

## Success criteria

1. `auto_trunc` beats log-linear `trunc` at matched R and matched endpoints
   (H1); the gap grows with R.
2. `auto_trunc_ada` ~ `auto_trunc` within noise (H2 — substitutes).
3. w/o Trunc arms clearly worse than w/ Trunc (H3).
4. Best cell of this sweep vs the 10.99 incumbent (H4).
5. Entropy >= 3.0 on every reported cell (degeneracy bar) — GenPPL is only
   meaningful read together with entropy.

## Compute

- 24 cells x 4 GPUs, `thickstun,desa` (exclude `desa-compute-01`), 8 cpus,
  64G, 2-day walltime. ~4-5 h/cell (30k steps at ~266k tok/s over 4 GPUs) plus
  ~1 h eval; with ~28 shared GPUs that is ~6-7 concurrent cells, so ~4 waves,
  roughly 20-24 h wall clock.
- Outputs: `outputs/eflm_rescale_auto_tinystories_256/eflmrat_{arm}_r-{R}/`,
  eval in `eval/{ppl.json, samples_genppl.json}`. Checkpoints KEPT.

## Run

    python experiments/eflm_rescale_auto_tinystories_256/sweep.py --dry-run
    python experiments/eflm_rescale_auto_tinystories_256/sweep.py
    # subsets:
    python .../sweep.py --arms auto_trunc auto_trunc_ada --rhos 1 8 28
