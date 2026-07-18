# Loss-Geometry Band Metrics — TV_lin and the Quality Panel

**Question.** The loss-geometry curve L(t) (token-mean denoising CE at pinned flow
time t) differs sharply across geometries and noise-schedule tricks: Euclidean
flows are a "cliff" (all loss appears in a sliver at t→1), hyperbolic flows and
truncation/Gumbel schedules spread it out. We want a scalar metric of this
"spread" that (a) is principled, (b) predicts generation quality (GenPPL), and
(c) flags pathologies. This note defines the metric panel and validates it on
all cached TinyStories loss-geometry runs.

**Provenance.** All curves come from `visualization/loss_geometry.py` (33-point
t-grid in [0.001, 1], 8 val batches × 16 seqs, EMA weights, final checkpoint of
each run — 30K steps, TinyStories seq-256). Curve caches live under
`experiments/loss_geometry_vis/tinystories/`. GenPPL values are the
corresponding entries from the experiment RESULTS tables (gpt2-large judge).

## Definition: TV_lin

Normalize the curve by its own ceiling (the pure-noise loss, which equals the
unigram entropy ≈ 5.94 nats on TinyStories):

    g(t) = L(t) / L(1),        g: [0,1] → [0,1]

If the model releases information at a perfectly constant rate over flow time
(the "information-uniform" ideal that LangFlow's trainable Gumbel schedule
optimizes for), g(t) is a straight line and g''(t) = 0 everywhere. **TV_lin
measures the total deviation from constant slope:**

    TV_lin = ∫₀¹ |g''(t)| dt  =  total variation of g'(t)
           ≈ Σᵢ | g'(tᵢ₊₁) − g'(tᵢ) |        (finite differences on the t-grid)

Interpretation and anchors (33-point grid, Δt ≈ 0.031):

| shape of L(t)                      | TV_lin |
|---|---|
| perfectly linear ramp (uniform information rate) | 0 |
| single-step cliff (all loss appears in one grid step) | ≈ 2/Δt ≈ 64 |
| non-monotone overshoot above the ceiling (degenerate model) | ≫ 64 |

TV_lin is scale-invariant (ceiling-normalized) and offset-blind (second
derivatives kill constants) — the latter is a feature for shape comparison and
a blind spot for pathology detection, hence the companion axes below.

## Companion axes

- **mid-mass g(0.5) = L(0.5)/L(1):** how much of the ceiling is already
  unresolved at mid-schedule. Captures the low-decade (log-scale) structure
  that linear-curve curvature cannot see.
- **area = mean g(t) = (1/L(1))·∫L dt:** the schedule-averaged denoising CE
  normalized by the ceiling — i.e., the validation bound (val/ppl) in
  disguise, made cross-geometry comparable by the ceiling normalization.
- **floor = L(0.001):** loss on nearly-clean inputs. Must stay ≈ 0; a lifted
  floor means the model cannot even finish denoising (the LangFlow failure
  mode). Invisible to TV_lin by construction.
- **t10 / t50 / t90:** the first t at which g(t) crosses 0.1 / 0.5 / 0.9 of the
  ceiling (linearly interpolated on the grid) — the position of the
  transition. **bandwidth = t90 − t10** is the width of the top transition;
  descriptive companions to TV_lin (a cliff has bandwidth ≈ 0 regardless of
  where it sits; TV_lin and bandwidth are near-monotone inverses of each other
  on healthy curves, but bandwidth is easier to read as "how much of the flow
  time the transition occupies").

## Results (TinyStories, final checkpoints)

| run | TV_lin | mid g(0.5) | area | floor (nats) | t10 | t50 | t90 | bandwidth | GenPPL |
|---|---|---|---|---|---|---|---|---|---|
| LangFlow + ada (Gumbel) | **2.7** | 0.494 | 0.533 | **0.567** | 0.01 | 0.51 | 0.81 | **0.80** | 20.7 |
| S-FLM + ada + trunc | 3.1 | 0.346 | 0.429 | 0.047 | 0.25 | 0.60 | 0.86 | 0.61 | **11.0** |
| S-FLM + trunc | 6.1 | 0.078 | 0.293 | 0.014 | 0.53 | 0.73 | 0.88 | 0.35 | 12.9 |
| S-FLM + ada | 13.9 | 0.001 | 0.078 | 0.000 | 0.87 | 0.95 | 0.99 | 0.12 | 20.2 |
| H-FLM prior_cov 0.3 | 15.7 | 0.019 | 0.327 | 0.001 | 0.58 | 0.69 | 0.75 | 0.17 | 29.2 |
| H-FLM prior_cov 1.0 | 15.8 | 0.052 | 0.389 | 0.001 | 0.54 | 0.63 | 0.68 | 0.15 | 17.7 |
| E-FLM naive | 29.4 | 0.000 | 0.034 | 0.001 | 0.97 | 0.98 | 1.00 | 0.03 | 34.6 |
| H-FLM prior_cov 0.001 | 32.0 | 0.000 | 0.030 | 0.000 | 0.97 | 0.98 | 1.00 | 0.03 | 102.1 |
| H-FLM prior_cov 0.04 (collapsed) | **117.4** | — | — | — | 0.67 | 0.72 | 0.73 | 0.07 | degenerate (entropy 0) |

Rank correlation with GenPPL over the 8 healthy labeled runs (Spearman):

| metric | ρ vs GenPPL | p |
|---|---|---|
| mid-mass g(0.5) (higher = better) | **+0.714** | 0.047 |
| TV_lin (lower = better) | +0.667 | 0.071 |
| area (higher = better) | +0.595 | 0.120 |

## Conclusions

1. **TV_lin works as a shape metric.** It cleanly orders cliff geometries
   (E-FLM 29.4, H-FLM at tiny prior_cov 32.0) below schedule-shaped runs
   (trunc 6.1, ada+trunc 3.1, Gumbel 2.7), operationalizing "how far is the
   information-release rate from constant" — the design target behind both
   the truncation trick and the trainable Gumbel schedule.
2. **Free pathology detector.** The collapsed H-FLM cell (prior_cov 0.04,
   sample entropy 0.0) scores TV_lin = 117 — far above the one-step-cliff
   anchor of 64 — because its L(t) overshoots the unigram ceiling
   (confidently-wrong predictions at mid-noise). TV_lin > ~64 can be used to
   auto-flag degenerate runs from the loss curve alone, without sampling.
3. **Blind spot #1 — the low decades.** Computed on the linear normalized
   curve, |g''| is dominated by the top transition; structure below
   g ≈ 0.05 contributes almost nothing. Measured consequences: S-FLM+ada
   (mid-mass 0.001) scores *better* than H-FLM pc1.0 (13.9 vs 15.8) yet has
   *worse* GenPPL (20.2 vs 17.7); H-FLM pc0.3 and pc1.0 are indistinguishable
   by TV_lin (15.7 vs 15.8) yet differ 29.2 vs 17.7 in GenPPL. The
   exponential-ramp advantage of hyperbolic geometry lives in the low decades
   and is captured by mid-mass/area, not by linear-curve curvature.
4. **Blind spot #2 — the floor.** Second derivatives are offset-blind:
   LangFlow+Gumbel has the *best* TV_lin (2.7) while being broken on
   nearly-clean inputs (floor 0.567 nats), which caps its GenPPL at 20.7.
5. **Recommended panel (three axes).** Report together:
   (i) **TV_lin** — shape uniformity + collapse flag;
   (ii) **mid-mass g(0.5)** (or area) — low-decade mass; the best single
   GenPPL predictor here, and interpretable as the ceiling-normalized
   validation bound;
   (iii) **floor** — clean-end integrity.
   A run is "good" iff TV_lin is low, mid-mass is high, and floor ≈ 0. No
   single scalar suffices: each axis has a measured counterexample.

## Caveats

- n = 8 labeled runs; the correlations are directionally consistent but not
  individually decisive, and the three metrics are collinear.
- The t-axis is *nominal*: truncation/adaptive/Gumbel schedules reparametrize
  t → physical noise, so part of their "spreading" is an axis remap, while
  H-FLM's spreading is physical (same log-linear schedule). The operational
  consequences (training-sample allocation, per-step decision density) are
  the same, which is what the metrics measure.
- Second differences on a 33-point grid amplify noise; use only
  well-averaged curves (these are means over 32k tokens).
- A log-domain variant (total variation of d log L/dt, "is the exponential
  rate constant") rewards the hyperbolic ramp shape directly, but needs a
  noise-floor threshold and fails on curves with short support (e.g.,
  prior_cov 0.001); treat it as an optional fourth axis.

## Pointers

- Curves and caches: `experiments/loss_geometry_vis/tinystories/*/`,
  pc-sweep overlay: `experiments/loss_geometry_vis/tinystories/hflm_pc_sweep/`
- Metric computation: this table was produced from the cached `.json` curves;
  the definition above is self-contained (normalize, finite-difference twice,
  sum absolute values).
