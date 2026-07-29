# eflm_rescale_auto_sudoku — results

**COMPLETE: 194 / 194 cells, zero failures** (2026-07-27). Sudoku **hard** (30 clues), 2000 val puzzles,
full-board solve rate (%), tiny sphere-DiT, 20k steps, bs 256, 180 sampling
steps, exact velocity, greedy last step, 3 seeds per cell unless noted.

Regenerate: `analyze.py --tau <tau>` (tables), `plot_tau_curve.py`
(`tau_curve.png`).

## Headline

**The autonomous clock, truncated near the decode point, gives the naive
(fixed-schedule) E-FLM most of what the adaptive noise scheduler gives it —
without a scheduler.**

Best cell per arm and embedding norm R, 3-seed mean +- sd, vs the log-linear
control (`experiments/eflm_rescale_sudoku`, best LR per cell):

| arm | R | autonomous best | config | log-linear best | delta |
|---|---|---|---|---|---|
| naive | 5  | **51.1 +- 3.5** | tau=0.25, snr, lr 1e-3 | 40.0 | **+11.1** |
| naive | 8  | **45.1 +- 1.3** | tau=0.5, ce, lr 5e-4   | 32.1 | **+13.0** |
| naive | 16 | **44.3 +- 1.8** | tau=0.1, ce, lr 1e-3   | 23.6 | **+20.7** |
| ada   | 5  | 50.1 +- 2.7 | tau=0.25, snr, lr 1e-3 | 53.9 | -3.8 |
| ada   | 8  | 50.6 +- 2.9 | tau=0.5, ce, lr 5e-4   | 48.8 | +1.8 |
| ada   | 16 | 52.0 +- 1.4 | tau=0.5, ce, lr 1e-3   | 47.6 | +4.4 |

The gain is entirely in the **naive** arm and grows with R; on the **ada** arm
the two mechanisms are redundant (they solve the same problem — where to spend
sampling/training time) and do not stack: the `ada` arm lands at 50-52 % under
either clock.

## 1. tau_max is the dominant knob, and its optimum is *predicted*

Accuracy vs clock horizon (naive, lr 1e-3, 3 seeds where available):

| R | tau=0.1 | tau=0.25 | tau=0.5 | tau=1 | tau=2 | tau=4 | tau=7 | best tau | tau*(R) predicted |
|---|---|---|---|---|---|---|---|---|---|
| 5  | 7.2 | 36.9+-3.3 | **39.8+-1.4** | 29.2 | 29.1 | 24.0 | 15.2 | 0.5 | 0.505 |
| 8  | 27.2+-4.1 | **41.9+-6.7** | 38.1+-1.4 | · | · | · | · | 0.25 | 0.344 |
| 16 | **44.3+-1.8** | 29.3+-0.5 | 37.4+-1.3 | 24.3 | 5.9 | 3.5 | 0.4 | 0.1 | 0.187 |

- A single schedule parameter moves solve rate from 0.4 % to 44 %. The
  autonomous flow is **not** automatically good; it is good only when truncated
  near the decode point.
- **The optimum tracks tau*(R) = -log b*(R), b*(R) = 1/(1 + C/R), C = 3.284**
  (random-codebook analysis, V=12, delta=0.1) — the best horizon shrinks
  monotonically with R and lands within one grid step of the closed-form
  prediction at all three norms. The truncation does not have to be tuned
  blindly; it can be read off the codebook geometry.
- Mechanistically this is the same statement as the `eflm_rescale_sudoku`
  finding (large R decodes late/sharply): the log-linear clock spends
  (1 - b*) = 17 % of its steps before the decision at R=16 and 40 % at R=5, so
  the reallocation pays most at large R — which is exactly the delta ordering
  above (+11.1 / +13.0 / +20.7).

## 2. The autonomous clock removes the R-dependence

At a single fixed horizon (tau=0.5, naive, ce, lr 1e-3) across R = 5 / 8 / 16:

| clock | R=5 | R=8 | R=16 | spread |
|---|---|---|---|---|
| log-linear | 39.6 +- 3.4 | 32.1 +- 2.5 | 18.7 +- 5.2 | **20.9** |
| autonomous (tau=0.5) | 39.8 +- 1.4 | 38.1 +- 1.4 | 37.4 +- 1.3 | **2.4** |

Solve rate stops depending on the embedding norm — the thing the rescale study
set out to control. Seed variance also drops (sd ~1.4 vs 2.5-5.2).

## 3. The SNR-weighted CE (VDM Eq. 16): helps at short horizons, fatal at long ones

| naive, lr 1e-3 | tau=0.1 | tau=0.25 | tau=0.5 | tau=1 | tau=2 | tau=7 |
|---|---|---|---|---|---|---|
| R=5, snr  | 0.0 | **51.1+-3.5** | 26.0+-18.4 | 18.4 | 12.9 | 4.2 |
| R=16, snr | 43.5+-7.7 | 29.9+-4.1 | 5.3+-3.5 | 0.9 | 0.0 | 0.0 |

- The best `snr` cell (51.1 at R=5, tau=0.25) is the best *naive-arm* cell in
  the experiment and beats its `ce` counterpart by +14 points.
- At tau >= 1 `snr` collapses to ~0 on both clocks (log-linear + snr: 30.4 vs
  34.4 at R=5, 7.8 vs 25.0 at R=16).

One mechanism explains both regimes — the weight `w = tau_max (1-b)/b^2` over
the band `b in [e^-tau, 1]`:

| tau_max | b range | w dynamic range | outcome |
|---|---|---|---|
| 0.25 | [0.78, 1] | ~360x | best cells |
| 0.5  | [0.61, 1] | ~1e3 | mixed / unstable |
| 2.0  | [0.14, 1] | ~5e4 | collapse |
| 7.0  | [1e-3, 1] | ~1e9 | dead |
| log-linear | [1e-3, 1] | ~1e12 | -17 pts vs its own CE |

On a short horizon the weight is a mild tilt toward the clean edge of a narrow
band — i.e. toward where the decision is actually made. On a long horizon the
1/b^2 (1/b^3 for log-linear) growth dominates, and because a rescaled codebook
makes CE ~ 0 for b << b*, the objective is dominated by samples the model has
already solved: the weighted training loss falls to ~2e-3 while plain CE sits
at 2.7e-2, and accuracy follows it down.

**This is not a defect of the bound but of the parameterisation.** VDM Eq. 16
weights `||x - x_hat||^2`, whose MMSE decays *polynomially* in SNR; a CE over a
discrete codebook decays *exponentially*, so the same weight over-integrates the
solved region. The same bound in the epsilon-parameterisation (Eq. 17) carries
`gamma'(t)/2 = tau_max/(1-b)`, which is nearly constant over a short horizon —
so **plain CE on a short autonomous clock already is that variational bound up
to a constant**, and the x-space weight is a useful correction only where its
dynamic range stays small.

## 4. VDM schedule invariance (§5.1): not supported at a fixed step budget

tau_max = 7 is the SNR-matched twin of log-linear (SNR_max 1.2e6 vs 1.0e6), so
under the weighted bound the two should differ only in estimator variance.
Measured (`snr`): 4.2 vs 30.4 (R=5), 0.0 vs 7.8 (R=16). The premise (same
objective, same SNR endpoints) holds; the prediction fails because at 20k steps
the two clocks are very different Monte-Carlo estimators of that integral, and
the autonomous one draws nearly all samples where the integrand is flat.

## 5. Stability / LR interaction — resolved

`snr` at tau=0.5 / R=5 / lr 1e-3 diverged on one seed (0.0 vs 37.8 / 40.1;
loss 0.006 at step 2.1k -> 0.18 at 5.1k). At lr 3e-4 / 5e-4 the same cell is
stable: 38.0 +- 4.6 / 39.3 +- 5.9. So the divergence is an optimization
artifact — the SNR weight shrinks the loss scale ~1000x, gradient clipping
(1.0) stops binding and the effective step size rises — not a property of the
bound. LR mattered elsewhere too: naive/ce/R=8 is 45.0-45.1 at 3e-4/5e-4 vs
38.1 at 1e-3, so the pilot's lr 1e-3 was not the right LR at every R.

## 6. Full LR axis at tau = 0.5 (3-seed means)

| arm | w | R | lr 3e-4 | lr 5e-4 | lr 1e-3 |
|---|---|---|---|---|---|
| naive | ce  | 5  | 40.7+-3.1 | 41.2+-5.5 | 39.8+-1.4 |
| naive | ce  | 8  | 45.0+-3.2 | **45.1+-1.3** | 38.1+-1.4 |
| naive | ce  | 16 | 35.3+-6.7 | 34.8+-6.1 | 37.4+-1.3 |
| naive | snr | 5  | 38.0+-4.6 | 39.3+-5.9 | 26.0+-18.4 |
| naive | snr | 8  | 30.6+-3.2 | 24.0+-7.5 | 37.5+-3.1 |
| naive | snr | 16 | 28.8+-2.0 | 24.4+-11.5 | 5.3+-3.5 |
| ada   | ce  | 5  | 40.6+-3.9 | 47.0+-5.8 | 38.8+-6.0 |
| ada   | ce  | 8  | 44.1+-4.0 | **50.6+-2.9** | 46.4+-10.8 |
| ada   | ce  | 16 | 49.3+-5.0 | 49.4+-5.7 | **52.0+-1.4** |
| ada   | snr | 5  | 25.8+-7.9 | 21.0+-2.3 | 22.8+-4.5 |
| ada   | snr | 8  | 28.8+-3.5 | 26.8+-4.0 | 18.4+-17.5 |
| ada   | snr | 16 | 17.0+-7.5 | 15.0+-5.0 | 7.5+-6.3 |

## 7. Conclusions

1. **Ship the autonomous clock for the naive arm.** `noise=autonomous` with
   `tau_max ~ tau*(R)` is a one-line schedule change worth +11 to +21 points on
   sudoku hard and it flattens the R-dependence (spread 20.9 -> 2.4).
2. **Set tau_max from the codebook, not by search**: tau*(R) = log(1 + C/R)
   with C = sqrt(2 log(2(V-1)/delta)) predicted the empirical optimum at all
   three norms.
3. **The autonomous clock and the adaptive noise scheduler are substitutes,
   not complements.** Combining them gains nothing (-3.8 / +1.8 / +4.4); the
   clock is the cheaper of the two (closed form, no spline fitting, no extra
   buffers).
4. **The VDM SNR-weighted CE is only usable on a short horizon.** Where its
   dynamic range stays <= ~1e3 it is the best objective tested (51.1 at R=5);
   beyond that it destroys training. If the weighted bound is wanted in
   general, use the epsilon-parameterisation (Eq. 17) instead — under the
   autonomous clock that is nearly plain CE, which is what already works.

### Follow-ups

- `time_conditioning=false` ablation: if the flow is genuinely autonomous the
  drift needs no clock, and under a short horizon the model barely resolves
  `sigma = -log(alpha_t)` anyway.
- Loss-geometry L(t) curves from the retained checkpoints, to confirm that the
  best tau_max coincides with the measured (not just predicted) transition.
- Carry the short autonomous clock to TinyStories, where the codebook is 50k
  and tau*(R) is correspondingly different.
