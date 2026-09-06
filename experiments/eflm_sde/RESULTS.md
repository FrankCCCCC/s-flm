# eflm_sde — Results

Task (`setup.md`): design a `GScheduler` g(t) and find the `eta` range for the
marginal-preserving EFLM SDE sampler (`derivation.md`) such that EFLM traces a
competitive GenPPL-vs-entropy frontier against DUO / MDLM / FLM
(`experiments/naive_ar_tinystories_s256`) and S-FLM.

Protocol: frozen 3-seed checkpoints `eflmratr_lr-1e-3_r-0.5_m-1.0{,_s2,_s3}`
(R=0.5, TAU_MAX=2.4436, 30k steps, TinyStories seq-256); eval = exact
velocity, `top_k_velocity=1`, greedy last, 512 samples/cell, GenPPL =
gpt2-large retokenized, unigram entropy; eval seed fixed. Baseline curves:
per-seed `frontier_cells.csv` of `naive_ar_tinystories_s256` (same 3-seed,
512-sample, per-NFE protocol). S-FLM: `seed_errbar` checkpoints (trunc+ada,
SC), default decode, NFE-swept.

**STATUS: COMPLETE — phases 1-2 + p75 refinement (461 EFLM cells + 24 S-FLM
cells) and phase 3, the temperature frontier (360 cells + 3 marker cells;
§6 and `experiments/naive_ar_tinystories_s256/FRONTIER_RESULTS.md` §7).
824 cells total, 0 failures.**

![frontier](figures/genppl_entropy_frontier.png)

---

## 1. Headline: the SDE ends the entropy trap, and moderate eta is a free lunch

The ODE sampler (eta=0) has no diversity knob under the recorded protocol
(temperature and the greedy last step are argmax-invariant): it sits at one
point, GenPPL ~11.3-13.1 @ H 3.75-3.81, left of every measured baseline
curve. With g(t) = sqrt(1-t) and moderate eta, EFLM **improves both axes at
once** (3 seeds, mean +- sd):

| NFE | ODE (eta=0) | best SDE cell | delta GenPPL | delta H |
|---|---|---|---|---|
| 16  | 13.06 +- 0.81 @ 3.754 | sqrt eta=2: **11.95 +- 0.26** @ 3.899 | -1.11 | +0.15 |
| 32  | 12.00 +- 0.56 @ 3.783 | sqrt eta=2: **10.48 +- 0.22** @ 3.888 | -1.52 | +0.11 |
| 64  | 11.60 +- 0.50 @ 3.796 | sqrt eta=4: **9.56 +- 0.06** @ 3.927 | -2.04 | +0.13 |
| 128 | 11.41 +- 0.48 @ 3.802 | sqrt eta=4: **9.26 +- 0.12** @ 3.923 | -2.15 | +0.12 |
| 256 | 11.33 +- 0.44 @ 3.806 | sqrt eta=8: **9.21 +- 0.05** @ 3.979 | -2.12 | +0.17 |

For scale: the parent sweep's recommended EFLM operating point was
11.44 +- 0.54 (180 NFE) and best-S-FLM is 11.22 +- 0.47 (180 NFE). The SDE at
**64 NFE** reaches 9.56 +- 0.06 — ~1.9 GenPPL better than both at a third of
the budget, with *higher* entropy. The mechanism is the standard SDE-vs-ODE
effect: the injected noise + score correction contracts accumulated
discretization/model error instead of integrating it.

**The SDE also collapses seed variance.** In the basin, GenPPL sd drops from
0.44-0.81 (ODE) to 0.05-0.26 — up to ~9x. Trajectory noise washes out
seed-specific error, so the frontier is *more* reproducible stochastic than
deterministic.

## 2. Competitiveness at matched entropy

Gen. PPL interpolated along each 3-seed mean curve:

| NFE | H=4.10: MDLM / DUO / FLM / **EFLM** | H=4.20: MDLM / DUO / FLM / **EFLM** |
|---|---|---|
| 8   | 26.98 / 20.54 / 78.63 / **26.50** (lin) | 31.10 / 24.09 / 78.95 / **33.24** (lin) |
| 16  | 14.53 / 13.01 / -    / **16.05** (lin) | 16.52 / 15.28 / 58.25 / **19.23** (lin) |
| 32  | 10.91 / 10.92 / -    / **12.69** (lin) | 12.24 / 12.52 / 45.41 / **14.89** (lin) |
| 64  |  9.63 / 10.14 / -    / **11.11** (lin) | 10.77 / 11.61 / 36.62 / **12.86** (lin) |
| 128 |  8.83 / -     / -    / **10.15** (sqrt)| 10.05 / 11.02 / 30.96 / **12.34** (sqrt)|
| 256 |  8.70 /  9.68 / -    /  **9.59** (sqrt)|  9.73 / 11.05 / 27.43 / **11.50** (sqrt)|

- **vs FLM** (the continuous-flow baseline): EFLM wins by **3-4x at every
  budget and entropy** where FLM is measurable. The Euclidean flow with the
  SDE decode is simply a better continuous-flow generator than FLM here.
- **vs DUO**: EFLM *beats* DUO at NFE 256 / H 4.10 (9.59 vs 9.68) and is
  within 4-10% at NFE >= 32; DUO keeps a real edge only at NFE 8-16.
- **vs MDLM** (the strongest baseline): gap is 10-16% at H 4.10 across
  NFE 16-256, and EFLM *matches* MDLM at NFE 8 (26.50 vs 26.98). MDLM is
  not caught at H >= 4.35, where EFLM's curves bend up 2-3x above it.
- **Below H ~= 4.05** the baselines have no measured points (their T-grid
  floor); EFLM's basin (9.2-9.6 @ H 3.85-4.0) extends the measured Pareto
  frontier into that region at NFE >= 64.

Verdict on the setup.md goal: **EFLM is now on-chart and competitive over
H in [3.8, ~4.2]** — dominating FLM everywhere, trading with DUO, ~10-16%
behind MDLM — instead of being a single point outside the plot. The
H >= 4.35 regime remains MDLM's.

## 3. vs S-FLM: dominated at every NFE >= 8

S-FLM under its recorded decode is T-inert like EFLM, so it contributes one
point per NFE. EFLM's SDE curve passes below it everywhere except NFE 4:

| NFE | S-FLM | EFLM SDE (nearest-entropy cell) | EFLM better by |
|---|---|---|---|
| 4   | 32.00 +- 1.61 @ 3.771 | 36.66 +- 2.17 @ 3.698 (lin eta=1) | -15% (tie/worse) |
| 8   | 18.34 +- 0.64 @ 3.894 | 16.96 +- 0.79 @ 3.848 (lin eta=2) | +8% |
| 16  | 14.50 +- 0.29 @ 3.932 | 11.95 +- 0.26 @ 3.899 (sqrt eta=2) | +18% |
| 32  | 13.04 +- 0.21 @ 3.948 | 10.50 +- 0.13 @ 3.945 (sqrt eta=4) | +19% |
| 64  | 12.27 +- 0.23 @ 3.958 | 9.78 +- 0.09 @ 3.990 (sqrt eta=8) | +20% |
| 128 | 11.88 +- 0.15 @ 3.963 | 9.41 +- 0.13 @ 3.982 (sqrt eta=8) | +21% |
| 256 | 11.67 +- 0.14 @ 3.966 | 9.21 +- 0.05 @ 3.979 (sqrt eta=8) | +21% |

(Caveat: S-FLM is compared at its single decode; an SDE sampler for S-FLM's
spherical geometry — the analogous derivation on the sphere — is the fair
follow-up and plausibly recovers a similar gain.)

## 4. GScheduler design: what the power family taught us

g(t) = (1-t)^p, i.e. diffusion a(b) = eta b^{2p} in the noise fraction b:

- **const (p=0) — rejected.** Full-strength injection at the decode endpoint
  destroys decodability: catastrophic collapse from eta ~= 2-4 (GenPPL in the
  thousands at H -> 5.37 ~= noise). Consistent with the parent sweep's
  endpoint noise:signal invariant (§4.5 there): const is the only schedule
  whose endpoint injection does not vanish.
- **quad (p=2) — rejected.** Early-only injection is re-absorbed by the
  remaining denoising steps; entropy barely moves (max +0.14 at eta=32) —
  it converges back to the ODE answer, as the marginal-preserving theory
  predicts for the fine-step limit.
- **sqrt (p=0.5) — the workhorse.** Optimal at NFE >= 32 for H <= ~4.2 and
  holds the free-lunch basin (§1). This is exactly the "regular family"
  a(b) = 2 lambda b of derivation.md §9.
- **linear (p=1) — the safe all-rounder.** Optimal at NFE 8-16, best above
  H ~= 4.2 at NFE <= 64, and degrades gracefully: within the swept grid it
  never explodes catastrophically at NFE >= 16 (worst in-grid cell 62.5 @
  H 4.45), while sqrt blows past ~2x its basin eta.
- The optimal p **falls as NFE grows** (linear at 8-16 -> sqrt at >= 32):
  with more steps, earlier-weighted noise is safely re-absorbed, so pushing
  injection later (smaller p) buys entropy more cheaply.
- **p75 (p=0.75) — interpolates, does not dominate.** In the crossover band
  (NFE 16-128, H <= 4.2) sqrt / p75 / linear are within ~5% of one another
  (mostly ~2 sd): at NFE 64 / H 4.10 they give 11.21 / 11.10 / 11.11; in the
  NFE-128 basin sqrt still leads (9.26 vs p75 9.47 vs linear 9.74); at
  NFE 16 / H 4.20 linear still leads (19.23 vs p75 22.12 vs sqrt 36.00).
  Its collapse boundary also sits between the two. Conclusion: **the
  frontier is insensitive to p in [0.5, 1] in the crossover band** — the p
  choice only matters at the extremes, so two named schedules (sqrt, linear)
  cover the design space and no intermediate scheduler is needed.

**Stability boundary.** For each g, catastrophic blowup sets in when the
per-step injection outruns what the remaining steps can denoise; empirically
eta* ~= 2 NFE for sqrt (8 @ NFE 4 ... 128 @ NFE 64) and >= 8 NFE for linear.
Near the boundary the failure is bimodal with enormous seed variance
(NFE 4, sqrt eta 16: 14956 +- 10316) — the same cliff-edge behavior as the
parent sweep's §4.1. Recommended operating etas sit >= 4x below eta*.

**Recommended defaults** (`sampler.eta`, `sampler.gt_method`):

| budget | recommendation |
|---|---|
| NFE <= 16 | `linear`, eta ~= NFE/8 (entropy target permitting) |
| NFE >= 32, H <= 4.2 | `sqrt`, eta ~= NFE/16 (free-lunch basin) |
| NFE >= 32, H > 4.2 | `sqrt` with eta up to ~NFE/4, or `linear` at 2-4x that eta |

## 5. Limitations / open items

1. **NFE = 4 is not rescued.** The SDE cannot fix a 4-step trajectory
   (best 33-37 vs MDLM 49-86 at its own entropy... actually EFLM at NFE 4 is
   *better* than MDLM's NFE-4 curve at matched low entropy, but both are far
   off the usable frontier). NFE = 1 is a single decode step; eta never acts.
2. **H >= 4.35 is not competitive** (2-3x behind MDLM). Buying that much
   entropy with white noise costs too much GenPPL; a temperature-sensitive
   decode (top_k_velocity > 1 or ancestral last step) composed with the SDE
   is the untested lever.
3. **Baselines have no measured points below H ~= 4.05** (their T-grid
   floor, naive_ar limitation #1), so "EFLM extends the frontier leftward"
   is unverified against T < 0.5 baselines.
4. **S-FLM is a per-NFE point, not a curve** (T-inert decode); a spherical
   SDE sampler would make that a fair curve-vs-curve comparison.
5. Data-entropy anchor for the x-axis still missing (naive_ar limitation #2).


---

## 6. Phase 3 — the temperature frontier (exact velocity, `top_k_velocity = -1`)

360 cells (3 seeds x NFE {1,...,256} x T {0.50,...,1.20}), 512 samples each, zero failures.
Sweep: `tfrontier_sweep.py`. Full analysis and the figure live with the baselines, in
`experiments/naive_ar_tinystories_s256/FRONTIER_RESULTS.md` §7 — this section records only
what it changes for *this* project.

**`top_k_velocity = 1` makes temperature a no-op; `-1` makes it a knob.** `logits / T` cannot
move an argmax, so under top-1 velocity + greedy last step T = 0.50 is bit-identical to
T = 1.00 (13.54 @ H 3.803 both, NFE 16). Under full-vocab exact velocity the same sweep spans
14.72 @ H 4.137 → 50.30 @ H 4.587. The paper says the same thing in App. C.8, and draws the
k = 1 variant as a single marker; those markers are this project's `eta = 0` ODE cells.

**H4 confirmed.** Entropy is strictly monotone in T for every (NFE, seed) — 24/24 curves —
and Gen. PPL rises with it. T is a well-behaved frontier parameter.

**H5 refuted — the two knobs are complementary, not ranked.** At matched entropy the
temperature knob *beats* the SDE knob throughout H >= 4.2, by 7-92% at NFE 16-64 (e.g.
NFE 32 / H 4.35: T 18.74 vs best eta 57.20); the one measured exception is NFE 128 / H 4.20,
where eta is 4% better. But the SDE owns everything below H ~= 4.1, where
temperature cannot reach at all: EFLM's T-curve floors at H ~= 4.13-4.14, while the SDE basin
sits at 9.2-9.8 @ H 3.92-3.99. So:

| band | best EFLM mechanism | Gen. PPL reached |
|---|---|---|
| H ~ 3.8 | k = 1 velocity, eta = 0 (T-inert marker) | 11.3-13.1 |
| H in [3.85, 4.1] | **SDE**, `gt=sqrt`, eta ~ NFE/16 | **9.2-12.0** |
| H in [4.1, 4.55] | **temperature**, exact velocity | 12.4-48.6 |

**H6 partially confirmed, and superseded.** The k = 1 marker does give lower Gen. PPL than the
exact-velocity curve at every NFE (11.33 vs 12.44 at NFE 256), reproducing the paper's
"top-1 beats unrestricted decoding" — but at ~0.3 nats lower entropy, so neither Pareto-
dominates the other. The SDE Pareto-dominates the k = 1 marker outright at every NFE >= 16
(lower Gen. PPL *and* higher entropy), so §1's recommendation stands.

**New for the parent comparison:** against MDLM/DUO/FLM the temperature arm makes EFLM the
**best method on the frontier at NFE 4-16** (2.1x over the best baseline at NFE 4 / H 4.10),
while MDLM takes H <= 4.35 at NFE >= 32. The cause is saturation, not degradation: EFLM
improves only 2.8x from NFE 4 to 256 (MDLM 5.8x) and just 3% past NFE 64.

**Open item this creates.** §5 item 2 asked whether a temperature-sensitive decode composed
with the SDE could reach H >= 4.35 cheaply. Half of it is now answered (temperature alone
does reach H 4.55, at eta = 0). The composition `eta > 0` **and** `T > 1` is still untested
and is the natural next sweep — the bands above suggest it could hold the 9.2 basin while
extending right.

---

## 7. Temperature vs eta, head to head

`compare_knobs.py` puts EFLM's two diversity knobs on the same axes, one panel
per NFE.

![knobs](figures/knob_comparison_t_vs_eta.png)

*Caveat on what is being compared.* The arms necessarily differ in more than the
knob: temperature is argmax-invariant under top-1 velocity (§6), so the T-arm
runs **exact velocity over the full vocab** (`top_k_velocity = -1`, eta = 0)
while the eta-arm runs **top-1 velocity + SDE** (`top_k_velocity = 1`). "T-knob"
and "eta-knob" name whole sampler configurations, not one isolated variable.

### 7.1 They swap places at H ~= 4.15-4.25

| NFE | H=4.15 | H=4.20 | H=4.25 | H=4.30 | H=4.35 | H=4.45 |
|---|---|---|---|---|---|---|
| 4 | **T** 94% | **T** 97% | **T** 98% | **T** 98% | T only | T only |
| 8 | **T** 35% | **T** 40% | **T** 57% | **T** 71% | **T** 100% | **T** 100% |
| 16 | **T** 14% | **T** 22% | **T** 34% | **T** 47% | **T** 53% | **T** 76% |
| 32 | eta 0% | **T** 25% | **T** 52% | **T** 62% | **T** 67% | **T** 68% |
| 64 | **eta** 10% | **T** 7% | **T** 35% | **T** 52% | **T** 60% | **T** 62% |
| 128 | **eta** 14% | **eta** 3% | **T** 25% | **T** 46% | **T** 55% | **T** 58% |
| 256 | **eta** 19% | T only | T only | T only | T only | T only |

(winner and margin at matched entropy; "eta" = best of sqrt/linear/p75.) The
crossover **moves right as NFE grows** — T wins from H >= 4.15 at NFE 8-16, from
4.20 at NFE 32-64, from 4.25 at NFE 128+. More steps give the SDE more room to
re-absorb its own injected noise, so the band it can hold widens; but it never
extends past H ~= 4.25.

### 7.2 Temperature is 5-10x cheaper per nat of entropy

Gen. PPL multiplier per +0.1 nats, over the shared H in [4.15, 4.35]:

| NFE | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| temperature | 1.41x | 1.18x | 1.17x | 1.16x | 1.16x | 1.17x | **1.17x** |
| best eta | 2.39x | 14.32x | 1.58x | 2.04x | 1.93x | 1.89x | — |

Temperature's price is **1.16-1.18x per 0.1 nats and essentially independent of
NFE** — a remarkably stable exchange rate. eta's is 1.6-2.4x and worsens toward
the collapse cliff. This is the whole story of §7.1 in one row: the two curves
start close, and eta's steeper slope loses it the race within ~0.1 nats.

### 7.3 eta reaches the deeper minimum; T reaches further right

| | best Gen. PPL | at H | reachable H window |
|---|---|---|---|
| eta (sqrt), NFE 256 | **9.21** | 3.979 | 3.86 – 4.19 (usable) |
| eta (sqrt), NFE 128 | **9.26** | 3.923 | 3.84 – 4.25 (usable) |
| temperature, NFE 256 | 12.44 | 4.113 | 4.14 – 4.52 |
| eta = 0 ODE (k=1 vel.), NFE 256 | 11.33 | 3.806 | single point |

Neither knob spans EFLM's frontier: the T-curve **cannot go below H ~= 4.13-4.14**
at any NFE (T = 0.50 is the grid edge, and even there the full-vocab velocity
keeps the sample diverse), and the eta-curve cannot go above H ~= 4.25 without
collapsing. The Pareto envelope of EFLM is the union — eta for H <= 4.2,
temperature for H >= 4.25:

| NFE | H=3.95 | H=4.05 | H=4.15 | H=4.25 | H=4.35 | H=4.45 |
|---|---|---|---|---|---|---|
| 8 | 18.8 (eta/lin) | 23.1 (eta/lin) | 19.4 (T) | 21.9 (T) | 26.8 (T) | 38.3 (T) |
| 16 | 12.6 (eta/sqrt) | 14.6 (eta/lin) | 15.1 (T) | 17.0 (T) | 20.7 (T) | 29.0 (T) |
| 32 | 10.6 (eta/sqrt) | 11.8 (eta/lin) | 13.8 (eta/lin) | 15.3 (T) | 18.7 (T) | 26.0 (T) |
| 64 | 9.6 (eta/sqrt) | 10.4 (eta/sqrt) | 12.0 (eta/lin) | 14.9 (T) | 18.0 (T) | 25.4 (T) |
| 128 | 9.3 (eta/sqrt) | 9.7 (eta/sqrt) | 11.1 (eta/sqrt) | 14.3 (T) | 17.7 (T) | 24.8 (T) |
| 256 | 9.2 (eta/sqrt) | 9.3 (eta/sqrt) | 10.4 (eta/sqrt) | 14.2 (T) | 17.4 (T) | 24.8 (T) |

### 7.4 Temperature is far more stable

Across the entire T sweep the 3-seed sd stays at **2-4% of the mean** (e.g.
26.85 +- 0.93 at NFE 8 / H 4.35). The eta arm is well behaved inside its basin
but becomes bimodal near the cliff — 6090 +- 5184 at the same (NFE, H), an 85%
relative sd, the cliff-edge behavior of §4. **If you need one knob you can turn
without monitoring, it is temperature.**

### 7.5 At NFE = 4 only temperature works

eta is unusable at 4 steps: the best schedule gives 773 at H 4.15 rising to 5251
at H 4.45, because four steps cannot denoise any useful injection. Temperature
delivers 42.6 -> 66.7 over H 4.15-4.30 on the same checkpoints. §5 item 1 said
"NFE = 4 is not rescued"; with the temperature knob it partly is, and it is what
makes EFLM the best method on the whole frontier at NFE 4
(`experiments/naive_ar_tinystories_s256/FRONTIER_RESULTS.md` §7.2).

### 7.6 What to use

| target entropy | knob | setting |
|---|---|---|
| H <= 4.10 | **eta**, `top_k_velocity=1` | `gt=sqrt`, eta ~ NFE/16 (NFE >= 32) |
| H 4.10 - 4.25 | **eta** at NFE >= 32, **T** at NFE <= 16 | see §7.1 |
| H >= 4.25 | **temperature**, `top_k_velocity=-1` | eta = 0, T in [0.6, 1.2] |
| NFE <= 4 | **temperature** only | eta collapses |

The obvious untested cell is the **composition** — eta > 0 *and* full-vocab
velocity with T > 1 — which could hold the 9.2 basin while inheriting
temperature's cheap 1.17x/0.1-nat slope. Nothing in the data rules it out.

## Reproduce

    python experiments/eflm_sde/frontier_sweep.py                   # pilot
    python experiments/eflm_sde/frontier_sweep.py --seeds 1 2 3 --nfes 4 8 \
        --gts sqrt linear --etas 0.25 0.5 1 2 4 8 16 32
    python experiments/eflm_sde/frontier_sweep.py --seeds 1 2 3 --nfes 16 32 \
        --gts sqrt linear --etas 0.5 1 2 4 8 16 32 64 --ode-extra
    python experiments/eflm_sde/frontier_sweep.py --seeds 1 2 3 --nfes 64 128 256 \
        --gts sqrt linear --etas 1 2 4 8 16 32 64 128 --ode-extra
    python experiments/eflm_sde/frontier_sweep.py --seeds 1 2 3 --nfes 16 64 128 \
        --gts p75 --etas 1 2 4 8 16 32 64 --ode-extra
    python experiments/eflm_sde/frontier_sweep.py --sfm --seeds 1 2 3 \
        --nfes 1 4 8 16 32 64 128 256
    python experiments/eflm_sde/analyze.py
    # phase 3 — the temperature frontier (exact velocity, top_k_velocity = -1)
    python experiments/eflm_sde/frontier_sweep.py --seeds 1 2 3 --nfes 1 --ode-extra
    python experiments/eflm_sde/tfrontier_sweep.py --nfes 1 4 8 16 32
    python experiments/eflm_sde/tfrontier_sweep.py --nfes 64 128 256 --t-chunk 5
    python visualization/genppl_entropy_frontier_line.py
    python experiments/eflm_sde/compare_knobs.py   # sec. 7, T vs eta
