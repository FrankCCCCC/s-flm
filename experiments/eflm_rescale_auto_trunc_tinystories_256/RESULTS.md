# eflm_rescale_auto_trunc_tinystories_256 — Results

Task (`setup.md`): *vary R, LR and the truncation timestep to find the optimal
GenPPL **without collapsing***, for truncated autonomous-clock E-FLM on
TinyStories seq 256, plain CE (`{w/o SNR}`).

Protocol: `small-sphere-dit` (768x12x12), ngpt init, 30k steps, global batch 512
(4 GPU x 32 x accum 4), bf16, EMA 0.9999, AdamW wd 0, clip 1.0, 1 seed.
Eval: exact velocity, `top_k_velocity=1`, 180 steps, greedy last;
GenPPL = gpt2-large retokenised generative perplexity.

**STATUS: COMPLETE. LR x R (26 cells) + truncation (6-point curve) +
3-seed replication of the two finalists (4 cells). 35 cells total.**

---

## 1. Headline

**Recommended: `R = 0.5, lr 1e-3, TAU_MAX = tau*(0.5) = 2.4436`
-> GenPPL 11.44 +- 0.54 (3 seeds), entropy 3.82.**

This is **not** the best single cell in the sweep. That was
`R = 0.05, lr 3e-4` at GenPPL 10.556 — which seed replication showed to be the
lucky draw of a config whose 3-seed mean is **13.66 +- 3.55**:

| config | seed 1 | seed 2 | seed 3 | mean | sd | range |
|---|---|---|---|---|---|---|
| R=0.05, lr 3e-4 | **10.56** | 12.88 | 17.53 | 13.66 | **3.55** | **6.98** |
| **R=0.5, lr 1e-3** | 11.53 | 10.86 | 11.93 | **11.44** | **0.54** | **1.07** |

**43.6x variance ratio.** The single-seed grid optimum sits 3.10 GenPPL (0.9 sd)
*below its own config's mean*. Selecting on it would have shipped a
configuration that is better on average **only 1 seed in 3**, and worse than the
recommendation by 6 points on a bad draw.

Why R=0.05 is unstable is not mysterious: it holds the sweep's lowest admissible
entropy (3.38-3.71), every neighbour of it on the truncation axis collapses
(m <= 0.7), and §4.1 establishes that variance explodes near that boundary.
R=0.5's sd of 0.54 matches the independent `seed_errbar` estimate (0.47-0.85)
almost exactly, confirming that figure holds *in stable regimes* and that
R=0.05's 3.55 is a real property of the edge, not a bad estimate.

**The difference of means (2.22) is itself only t = 1.07** — the two configs are
not separated on mean GenPPL. The recommendation rests on **variance and margin
from the collapse boundary**, which is what "without collapsing" asks for.

| reference (same protocol) | GenPPL | n seeds |
|---|---|---|
| **this sweep, recommended (R=0.5, lr 1e-3)** | **11.44 +- 0.54** | **3** |
| autonomous + trunc, R=0.5, lr 3e-4 (**same arm**, parent) | 11.81 | 1 |
| log-linear + trunc + **adaptive**, R=1 (project incumbent) | 10.36 | 1 |
| autonomous + trunc + **adaptive**, R=0.5 (parent best) | 10.74 | 1 |
| autonomous, untruncated, R=0.5 | 18.90 | 1 |
| raw-norm naive E-FLM baseline | 34.58 | 1 |

**Comparison caveat:** every reference above is a *single seed*. Comparing this
sweep's 3-seed mean against another run's best single draw is not like-for-like
— and this sweep has just demonstrated that a single draw can sit 3 points below
its own mean. The defensible statement is that the tuned scheduler-free arm is
**within seed noise of the adaptive-scheduler incumbent** (11.44 +- 0.54 vs a
single 10.36); neither a tie nor a win is established without replicating the
incumbent. H4 is therefore **not settled**, contrary to what a single-seed
reading would have suggested.

What *is* established: LR and truncation tuning did not move this arm
meaningfully off its inherited setting (11.81 at lr 3e-4 vs 11.44 +- 0.54 at
lr 1e-3 — a 0.37 difference against sd 0.54). **The parent sweep's
configuration was already near-optimal.** The value of this sweep is in the
mechanism (§4.5) and the negative results, not in a GenPPL gain.

**All three knobs have interior optima**, and the third one lands exactly on the
closed form:

| knob | optimum | penalty for moving one grid step | margin |
|---|---|---|---|
| R | **0.05** | +3.19 (R=0.02) / +1.56 (R=0.1) | 2-6 sigma |
| LR | **3e-4** at that R | +1.34 (1e-3) / collapse (5e-3) | 2 sigma / total |
| **TAU_MAX** | **tau*(R) = 4.665** exactly | +4.79 (m=0.85) / +5.82 (m=1.25) | **6-11 sigma** |

The truncation is the **sharpest** of the three axes, not the flattest, and its
optimum coincides with `alpha_star_euclidean` to within the grid resolution.

---

## 2. The anti-collapse gate is load-bearing, not a formality

`setup.md` asks for the optimal GenPPL *without collapsing*. That qualifier
decides the experiment, because **degenerate models occupy the TOP of the GenPPL
ranking, not the bottom**:

| cell | GenPPL | entropy | unique tokens generated | output |
|---|---|---|---|---|
| lr 5e-3, R=0.05 | **1.093** | 0.000 | **1** / 16384 | `'!'` x 100.00% |
| lr 5e-3, R=0.5 | **1.114** | 0.000 | 1 / 16384 | single token |
| lr 5e-3, R=0.1 | 1.824 | 0.351 | few | near-degenerate |
| lr 5e-3, R=1 | 4.305 | 0.631 | few | near-degenerate |
| **best real model** | **10.556** | 3.382 | 643 / 16384 | fluent stories |

A model emitting one `!` for all 16 384 tokens scores **10.5x "better"** than the
genuine optimum, because gpt2-large finds a constant string maximally
predictable. Ranking on GenPPL alone does not merely tolerate collapse — it
*prefers* it.

Practical rule for this codebase: **any code that ranks cells by GenPPL must
filter `entropy >= 3.0` first.** `analyze.py` initially had this bug and marked
the `'!'` cell as best; fixed (`ENT_MIN`). Valid PPL is a useful corroborating
detector — collapsed cells report `nan` / `inf` / ~1e134.

---

## 3. R: a clean U over three orders of magnitude, optimum at 0.05

At each R's own best LR (marginalising over the wrong LR misranks R by up to
4 points, so the joint grid `setup.md` specified was necessary):

| R | 0.01 | 0.02 | **0.05** | 0.1 | 0.5 | 1 | 5 | 8 | 16 | 28 |
|---|---|---|---|---|---|---|---|---|---|---|
| GenPPL | 15.68 | 13.75 | **10.56** | 12.12 | 11.53 | 11.72 | 15.14 | 13.96 | 16.33 | 16.09 |
| best lr | 3e-4 | 3e-4 | 3e-4 | 3e-4 | 1e-3 | 1e-3 | 1e-3 | 5e-3 | 5e-3 | 5e-3 |
| tau*(R) | 6.27 | 5.58 | **4.67** | 3.98 | 2.44 | 1.83 | 0.72 | 0.51 | 0.28 | 0.17 |
| % of untruncated horizon | 91% | 81% | **68%** | 58% | 35% | 27% | 10% | 7% | 4% | 2% |

**Both arms of the U are explained by one variable: the fraction of the flow the
closed-form bound leaves untruncated.**

- **R too small** (>= 81% of the horizon): approaches the untruncated regime,
  which is independently known to fail on this arm (18.90 at R=0.5 untruncated).
- **R too large** (<= 10%): the flow is truncated so aggressively there are too
  few effective denoising steps to resolve tokens.
- **Optimum at ~2/3 of the horizon** (R=0.05, 68%).

R=0.02 and R=0.01 were run *below* `setup.md`'s grid specifically to test whether
R=0.05 was a real optimum or a grid-edge artifact. It is real: 10.56 -> 13.75 ->
15.68 as R falls. The R=0.01 degradation was **predicted before the cell ran**
from the horizon-fraction reading, and landed in the right direction and
magnitude.

The axis has a natural floor: below R ~ 0.005, `tau*(R) > 6.908` exceeds the
schedule's own untruncated horizon, so "truncation at the decode point" ceases
to truncate at all. R=0.01 is already at 91%, so the axis is closed, not merely
unexplored.

---

## 4. LR: no global optimum — it must be tuned jointly with R

Full grid, GenPPL (entropy), at `m = 1.0`:

| lr \ R | 0.05 | 0.1 | 0.5 | 1 | 5 | 8 | 16 | 28 |
|---|---|---|---|---|---|---|---|---|
| **3e-4** | **10.56** (3.38) | **12.12** (3.61) | 11.81 (3.82) | 12.56 (3.83) | 17.25 (3.85) | 18.29 (3.89) | 19.76 (3.93) | 18.19 (3.90) |
| **1e-3** | 11.89 (3.47) | 12.25 (3.76) | **11.53** (3.81) | **11.72** (3.86) | **15.14** (3.90) | 15.39 (3.88) | 16.95 (3.93) | 17.24 (3.99) |
| **5e-3** | 1.09 (0.00) ⚠ | 1.82 (0.35) ⚠ | 1.11 (0.00) ⚠ | 4.30 (0.63) ⚠ | 16.42 (3.93) | **13.96** (4.01) | **16.33** (3.97) | **16.09** (3.96) |

**The optimal LR climbs monotonically with R** — 3e-4 at R <= 0.1, 1e-3 at
R in [0.5, 5], 5e-3 at R >= 8 — and the penalty for using the wrong one reaches
4.3 points (R=8: 18.29 at 3e-4 vs 13.96 at 5e-3).

Mechanism: small R means a long horizon (`tau*` up to 4.7) and therefore many
more effective denoising steps per unit of progress, so smaller optimizer steps
are appropriate; large R compresses the whole flow into `tau* <= 0.7` and
tolerates — indeed needs — a larger step.

**The 5e-3 failure region is a contiguous block at R <= 1**, healthy at R >= 5.

### 4.1 The stability boundary is bimodal, not gradual

The cell `(R=1, lr 5e-3)` was run twice under schedules differing by **0.25%**
in endpoint (`b_min` 0.1596 vs 0.1600), same seed:

| run | b_min | GenPPL | entropy |
|---|---|---|---|
| `alpha_max=null` | 0.1596 | **4.305** | 0.631 ⚠ collapsed |
| `alpha_max=0.84` | 0.1600 | **14.651** | 4.027 ✅ healthy |

Outcomes are 10 points apart with nothing between them. **Near the boundary,
training either survives or falls into single-token collapse, and an
imperceptible perturbation — or plain run-to-run nondeterminism — decides
which.** A +-0.5 error bar is meaningless against a bimodal 10-point gap.

Consequence for the recommendation: **prefer robustness over the best single
number.** 5e-3 wins at R=8 and R=16 but annihilates R <= 1; the recommended
configuration (R=0.05, lr 3e-4) is far from any observed instability.

---

## 4.5 Truncation: `tau*(R)` is exactly optimal, and the reason is an invariant

Six-point curve at the winner (R=0.05, lr 3e-4), `TAU_MAX = m * tau*(0.05)`:

| m | 0.4 | 0.5 | 0.7 | 0.85 | **1.0** | 1.25 |
|---|---|---|---|---|---|---|
| tau_max | 1.87 | 2.33 | 3.27 | 3.97 | **4.665** | 5.83 |
| % of untruncated horizon | 27% | 34% | 47% | 57% | **68%** | 84% |
| b_min | 0.155 | 0.097 | 0.038 | 0.019 | **0.0094** | 0.0029 |
| endpoint noise:signal | 102 | 60 | 22 | 10.7 | **5.27** | 1.6 |
| GenPPL | 41.5 ⚠ | 69.8 ⚠ | 43.7 ⚠ | 15.34 | **10.56** | 16.38 |
| entropy | 0.449 | 0.941 | 2.615 | 3.514 | **3.382** | 3.740 |

A clean interior optimum at **m = 1.0**: +4.79 GenPPL one step short, +5.82 one
step long, both 6-11 sigma. Shortening the horizon past m=0.7 **collapses** the
model outright.

### The invariant that explains it

The operative quantity is the **noise-to-signal ratio at the clean endpoint**,
`b_min * sqrt(d) / ((1 - b_min) * R)`. At `m = 1` the closed-form bound makes it
**independent of R**: since `b*/(1 - b*) = R/C`,

    ratio = sqrt(d) / C = sqrt( d / (2 ln(2(V-1)/delta)) ) = sqrt(768/27.64) = 5.27

| cell (m=1) | b_min | signal | noise | ratio |
|---|---|---|---|---|
| R=0.05 | 0.0094 | 0.0495 | 0.261 | **5.27** |
| R=0.5 | 0.0868 | 0.457 | 2.405 | **5.27** |
| R=28 | 0.8411 | 4.45 | 23.31 | **5.27** |

**This is what `alpha_star_euclidean` actually does — it holds endpoint
decodability constant across R.** It explains why the whole `m = 1.0` row is
collapse-free at 3e-4 and 1e-3 over three orders of magnitude in R, while the
U-shape in R (§3) is a second-order effect of horizon length and step count.

Deviating from `m = 1` breaks the invariance, and the ratio predicts the outcome:
**admissibility requires ratio <~ 10**, crossed between m=0.7 (22:1, entropy
2.615) and m=0.85 (10.7:1, entropy 3.514). Entropy is monotone in the ratio
across the collapse regime (0.449 / 0.941 / 2.615 / 3.382) while GenPPL is not
(41.5 / 69.8 / 43.7) — once output is degenerate GenPPL measures *which token got
stuck*, not quality.

Three predictions were made from this closed form before the cells finished
(m=0.7 collapse, m=0.85 borderline-admissible, m=1.25 admissible-but-worse) and
all three held.

### R and the horizon are not interchangeable

A control fell out of the design: two cells with near-identical horizons but 10x
different embedding norm.

| cell | tau_max | R | b_min | ratio | GenPPL | entropy |
|---|---|---|---|---|---|---|
| R=0.5, m=1.0 | 2.444 | 0.5 | 0.0868 | 5.27 | **11.81** | 3.82 ✅ |
| R=0.05, m=0.5 | 2.332 | 0.05 | 0.0971 | 60 | **69.77** | 0.94 ⚠ |

Same horizon, opposite outcome. **R does not act only through `tau*(R)`** — the
norm sets the signal scale, so the endpoint ratio (not `tau_max`) is the variable
that matters. This is the sharpest evidence in the sweep for what the "rescale"
in E-FLM-rescale buys.

---

## 5. Metric hygiene: valid PPL must not be used to select

| lr \ R | 0.05 | 0.1 | 0.5 | 1 | 5 | 8 | 16 | 28 |
|---|---|---|---|---|---|---|---|---|
| 3e-4 | 78.07 | 57.51 | 26.38 | 18.11 | 9.01 | 7.89 | 6.86 | 6.32 |
| 1e-3 | 77.09 | 56.58 | 26.69 | 18.14 | 8.85 | 7.77 | 6.75 | 6.17 |
| 5e-3 | nan | 2.6e119 | 3.6e134 | inf | 8.96 | 7.79 | 6.74 | 6.23 |
| tau_max | 4.665 | 3.981 | 2.444 | 1.834 | 0.719 | 0.505 | 0.284 | 0.172 |

Three independent reasons:

1. **Columns are not comparable.** Each R has its own horizon, so the bound
   integrates over a different range. PPL falls 78 -> 6.3 across R purely because
   `tau_max` shrinks 27x. The R=28 cells report "PPL 6.2" for a flow that spans
   `b in [0.84, 1]` and barely denoises.
2. **It cannot distinguish the learning rates.** At 3e-4 vs 1e-3 the bound is
   near-identical at every R (78.07/77.09, 26.38/26.69, 18.11/18.14, ...) while
   GenPPL differs by up to 2.9 points.
3. **It diverges rather than degrades on failure** (`nan`, `inf`, 1e134).

It *is* a useful collapse detector, and is retained as a diagnostic column.

---

## 6. Secondary result: schedule shape matters at fixed endpoints

An earlier run of this sweep (archived in
`outputs/eflm_rescale_auto_trunc_tinystories_256_amax084/`, 28 cells) was
executed with `noise.alpha_max=0.84` active, which pins `b_min = 0.16` for
*every* cell regardless of R and `m` (see §7). That makes it useless as a
truncation sweep — but it is a valid **fixed-endpoint, varying-path** experiment,
i.e. exactly the VDM 5.1 schedule-invariance question:

| m (R=0.5, lr 1e-3) | 0.5 | 0.7 | 0.85 | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|
| GenPPL | 11.96 | 11.73 | 12.94 | 11.75 | 11.32 | 10.90 | **10.50** |
| entropy | 3.52 | 3.59 | 3.44 | 3.62 | 3.66 | 3.77 | 3.77 |

Same endpoints, same 180-step budget, different traversal -> a **2.4-point
monotone improvement** from m=1.0 to m=2.0 (~3-5 sigma), entropy healthy
throughout. **Schedule invariance does not hold here**: how the flow traverses
between fixed endpoints materially changes generation quality. (Discovered
accidentally; would need a purpose-built replication to publish.)

---

## 7. Methodological incident: `noise.alpha_max` silently overrode the axis

Recorded because it invalidated ~28 cells and several intermediate conclusions.

`scripts/train/tinystories/eflm_rescale_auto_truncation.sh` passes
`noise.alpha_max=${ALPHA_MAX}` with default `0.840`. `TruncatedScheduleWrapper`
rescales alpha so `alpha(t=0) = alpha_max` **always**, hence `b_min = 0.16`
identically for every cell:

| R | m | tau_max | b_min *intended* | b_min *with wrapper* |
|---|---|---|---|---|
| 0.5 | 0.5 | 1.222 | 0.2944 | 0.1600 |
| 0.5 | 2.0 | 4.887 | 0.0075 | 0.1600 |
| 1 | 1.0 | 1.834 | 0.1596 | 0.1600 |
| 28 | 1.0 | 0.172 | 0.8411 | 0.1600 |

Effects: (a) the truncation axis measured nothing — `m` changed only path
curvature; (b) the R-dependence of the endpoint was removed; (c) the grid was
*split*, since the 6 inherited `lr 3e-4` cells came from the parent sweep with
`alpha_max=null`, so every LR comparison was confounded by a schedule
difference. The inversion is stark: at R=5 the contaminated data said 1e-3 was
**worse** by 2.58; the clean data says it is **better** by 2.11.

It also caused four spurious "collapses" at small R: with `b_min` pinned at 0.16,
R=0.05 has signal `0.84 x 0.05 = 0.042` against noise `0.16 x sqrt(768) = 4.4`, a
**105:1** noise-to-signal ratio at the supposedly-clean endpoint — the flow was
amputated 17x too early. Unwrapped, that same cell is the **best in the sweep**.

Resolution: sweep now passes `ALPHA_MAX=null` explicitly (verified: schedule
resolves to plain `Autonomous`, `b_min = b*(R)`). On the autonomous clock
`TAU_MAX` *is* the truncation; a second truncation on top also destroys the
pure exponential `b_t` that makes the clock autonomous.

**Lesson:** a defaulted env var in a shared script became load-bearing. Cell
provenance should be verified from each run's resolved `config_tree.txt`, not
from the sweep's intent — that check is what caught this.

---

## 8. The resolution floor

`experiments/seed_errbar_tinystories_256`, same protocol, 3 seeds:
sfm sd **0.470**, hflm sd **0.845**. Working figure **sigma ~ 0.5-0.85**, so
**2 sigma ~ 1.0-1.7 GenPPL**. Independently corroborated here: the same cell
`(R=1, lr 1e-3)` run twice at the same seed on different nodes gave 11.455 vs
11.718, a **0.26** spread from nondeterminism alone.

Every comparison in this report is labelled against that floor. In particular
the headline 10.56 vs the incumbent 10.36 is a **tie**, not a win.

---

## 9. Seed variance is a function of position, not a global constant

The two finalists differ 43.6x in seed variance (§1). This is the single most
actionable methodological finding here, because **the entire 26-cell grid is
1 seed per cell** and its noise is therefore *not* uniform:

- Cells in the interior of the admissible region (R in [0.5, 1]) have
  sd ~ 0.5, matching `seed_errbar`.
- Cells adjacent to the collapse boundary (R <= 0.1, or any lr 5e-3 cell at
  R >= 5) can have sd ~ 3.5, and near the bimodal boundary (§4.1) the spread is
  ~10 points and not even unimodal.

**Consequence: the grid's apparent optimum is biased toward high-variance
cells**, because a max over noisy cells preferentially selects the ones with the
fattest upper tail. R=0.05 won stage 1 by exactly this mechanism. Any future
sweep in this codebase that ranks single-seed cells should either replicate the
top-k or prefer cells with high entropy margin, which is the cheap proxy for
distance from the boundary.

---

## What is not yet established

1. **The `<= 10` admissibility threshold on the endpoint ratio** is bracketed by
   two cells (22:1 collapse, 10.7:1 admissible), not localised. It is a rule of
   thumb from this sweep, not a calibrated constant.
2. **The endpoint-ratio invariant is verified at m=1 across R, and its failure
   mode is verified along m at one R (0.05).** It has not been tested by varying
   m at large R, where `tau*` is short — the prediction there would be that
   *lengthening* the horizon (m > 1) is what breaks decodability.
3. **The truncation optimum was localised at R=0.05, not at R=0.5.** m=1 is
   justified at the recommended R by the invariant (§4.5) plus the fact that the
   whole m=1 row is collapse-free, not by a direct m-sweep at R=0.5. A 3-point
   m-sweep at R=0.5 would close this.
4. **H4 (scheduler-free arm vs adaptive incumbent) needs the incumbent
   replicated** — comparing 3 seeds against 1 is not a comparison.
5. **§6's schedule-shape result** came from a run whose configuration was
   accidental. It needs a purpose-built replication before it is more than
   suggestive.

## Reproduce

    python experiments/eflm_rescale_auto_trunc_tinystories_256/sweep.py --dry-run
    python experiments/eflm_rescale_auto_trunc_tinystories_256/sweep.py
    python .../sweep.py --lrs 3e-4 --rhos 0.05 --mults 0.4 0.5 0.7 0.85 1.25
    python experiments/eflm_rescale_auto_trunc_tinystories_256/analyze.py
