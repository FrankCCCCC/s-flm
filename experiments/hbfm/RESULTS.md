# HBFM — Results & Verdict

Analysis of the HyperbolicBoundaryFM (HBFM) parity experiment against
`experiments/hbfm/EXPERIMENT.md`. Sudoku-easy, greedy decode, 180 sampler steps,
full 2,000-puzzle valid set, seed 1.

---

## Verdict

**H1 is REFUTED at the tested capacity (d ≤ 64), but the experiment is
INCONCLUSIVE about the hyperbolic approach itself.**

- **Refuted vs threshold:** HBFM(d=64) scored **0.0% exact-match** (0/2000),
  which is **1.5 pt below** the d=64 S-FLM re-baseline (1.5%). The spec's refute
  condition is "≥5 pt below S-FLM(d=64)." The raw gap (1.5 pt) does **not** by
  itself cross that 5-pt bar — but HBFM is **at the absolute floor (literally
  zero correct grids)**, which fails the success criterion ("≥0 pt of S-FLM and
  ≥2 pt absolute, or statistically indistinguishable while training stably") on
  every clause. Against the primary hypothesis as written — "HBFM matches or
  beats S-FLM at the same d, target ≥0.70-class parity" — H1 is **not supported**.

- **Inconclusive about the geometry:** the comparison is **confounded by a
  capacity collapse that hits the baseline too.** The d=64 S-FLM re-baseline
  itself collapsed to **1.5%**, down from **77.6%** at d=512. At d=64 *both*
  methods sit on the floor (0% and 1.5%, a margin inside floor noise), so this
  experiment measures a degenerate-capacity regime — **not** the hyperbolic
  vs spherical forward process. The geometry question is **untested**.

**One-sentence reason:** HBFM produced zero correct grids, but the d=64 S-FLM
control also collapsed (77.6% → 1.5%), so the result reflects a forced
low-capacity regime (a `sample_radial` tooling cap), not the hyperbolic bridge.

---

## Results table

| run | exact-match | final train CE (20k) | wall-clock | W&B run |
|---|---|---|---|---|
| HBFM d=64 | **0.0%** (0/2000) | 0.0156 | 4h55m | `syctw/debug/xlyt114s` |
| S-FLM d=64 (re-baseline) | **1.5%** (30/2000) | 0.0193 | 26m | `syctw/debug/44qxrkf4` |
| HBFM d=2 (smoke/ablation) | 0.0% (0/2000) | 2.302 = log(10), uniform | 22m | `syctw/debug/kavt641d` |
| S-FLM d=512 (reference, prior) | 77.6% (1552/2000) | — | — | `sfm_easy_eval.log` / `eval_runs/sudoku/sfm_easy/results.json` |

Eval JSONs confirmed on disk: `eval_runs/sudoku/sfm_d64_easy/results.json`
(`accuracy=0.015`, `num_correct=30`) and `eval_runs/sudoku/sfm_easy/results.json`
(`accuracy=0.776`, `num_correct=1552`).

---

## Central finding: capacity collapse, not geometry

The headline of this experiment is the **S-FLM 77.6% → 1.5% drop when d goes
512 → 64.** That single number reframes everything:

- The spec's primary comparand is **S-FLM re-baselined at d=64** (correctly
  insisting on a matched-d control rather than the d=512 paper row). That control
  came back at **1.5%** — i.e. the d=64 backbone is essentially incapable of this
  task regardless of forward process.
- HBFM(d=64) at 0% is therefore being compared to a baseline that is itself
  broken. The 0% vs 1.5% gap is **within floor noise**; it does **not** license
  the claim "hyperbolic is worse than spherical." It only says both are at the
  floor at d=64.
- The fair test of the hyperbolic hypothesis requires the capacity where S-FLM
  actually works (d ≥ 256, ideally d=512 where it hits 77.6%). That regime was
  **never run**, because of the tooling cap below.

### Why d=64 was forced — verified root cause (tooling, not method)

`geo_bridge.HyperbolicHeatKernel.sample_radial` builds the marginal
`π(ρ) ∝ sinh^{d-1}(ρ) p_H(ρ; t)` in **linear** float64 space, which overflows as
`d` (and heat time `t`) grow. I reproduced the overflow directly:

| heat time t | d=64 | d=80 | d=128 | d=256 | d=512 |
|---|---|---|---|---|---|
| 0.05 (= configured `t_max`) | finite (ρ̄≈1.8) | finite (ρ̄≈2.1) | **NaN** | **NaN** | **NaN** |
| 0.10 | finite (ρ̄≈3.3) | **NaN** | **NaN** | **NaN** | **NaN** |
| 0.30 | **NaN** | **NaN** | **NaN** | **NaN** | **NaN** |

So at the experiment's configured `t_max = 0.05`, **d ≥ 128 already NaNs**, and
even d=64 NaNs once `t ≳ 0.3`. The spec's intuition (`E[ρ] ~ (d-1)t/2` pushes the
`sinh^{d-1}` integrand past the `exp(709)` float64 ceiling) is correct, and the
practical implication is that **d=512 — the only capacity where S-FLM is known to
solve Sudoku-easy — is unreachable** with the current sampler. This is a sampler
implementation limit, not a property of negative-curvature geometry.

---

## Secondary gates and diagnostics

**H2 (mechanism) gate — passed for d=64, vacuous for d=2:**
- `hbfm/rho_saturated_frac = 0` → no `ρ > _LORENTZ_RHO_MAX=20` clamp pressure, no
  boundary blow-up (clears the §2 gate of `< 0.01`).
- `hbfm/emb_norm_mean ≈ 1.01`, `mean_rho ≈ 1.03` → embedding left free (not
  renormalized, as designed in §4), radii finite and well-behaved.
- The §11 failure mode "radial overflow / saturation" did **not** fire during the
  d=64 run. The geometry wiring is healthy at d=64; the limitation is purely that
  d=64 is too small for the task.

**Per-token CE ≠ sequence correctness (consistent with the paper's caveat):**
both d=64 models drive train CE to near-zero (HBFM 0.0156, S-FLM 0.0193) yet
score ~0% exact-match. Spot-checking the HBFM/SFM generations shows grids that are
"Sudoku-ish" but contain repeated digits within a row (e.g. an `8 3 6 7 5 9 5 2 4`
row in the d=64 S-FLM output). Low marginal token CE is fully compatible with
violating the joint all-81-cells constraint — exact-match is the right primary
metric and it is unambiguous here.

**d=2 is genuinely degenerate, as designed:** CE never left `log(10) = 2.302`
(uniform); the model learned nothing. The d=2 boundary is a 1-D circle — this is
the smoke/ablation point (A1 / §6 Phase 0), not a real model, and its 0% should
**not** be read as evidence about the method. `emb_norm ≈ 0.87`, finite radii.

**Cost (OQ-1):** HBFM trained **~11× slower** than S-FLM at the same d (4h55m vs
26m). This is consistent with the flagged `sample_radial` per-step
inverse-CDF/grid rebuild on every `q_xt` call. A real scaling study at d≥256 is
not viable until this is addressed.

---

## Caveats / threats to validity

- **The HBFM vs S-FLM gap (0% vs 1.5%) is not interpretable.** Both are at the
  floor; the 1.5-pt margin is noise. Additionally, HBFM's geodesic-sampler `dest`
  construction is **unverified "new geometry"** (ARCH §6) — so even the small gap
  could be capacity, an immature sampler, or both. Do **not** conclude HBFM < SFM
  from this run.
- **Single seed.** Only seed 1 was run; the spec's success/refute conditions are
  defined "over 3 seeds, mean." Seeds 2–3 were not run (and would be wasted effort
  at d=64 floor capacity — defer until d≥256 works).
- **`d`-mismatch contamination avoided correctly.** The report's central
  comparison is HBFM(d=64) vs the matched S-FLM(d=64) re-baseline, never the
  d=512 paper row (which is reference-only, per §11). Good — but it also means the
  comparison lands in the dead zone where neither method functions.
- **Eval protocol:** greedy decode, 180 steps, EMA on, full 2,000 valid set,
  matches §10 for like-for-like comparison with the S-FLM reproduction. No
  protocol drift detected; train/valid disjoint by the seed-42 split.

---

## Recommended next steps (ranked)

1. **Fix `sample_radial` to a log-space marginal → unblock d ≥ 256 → rerun the
   real test.** Replace the linear `sinh^{d-1}(ρ) p_H(ρ; t)` construction with a
   `logsumexp`/log-space integrand so the marginal is finite at d=256/512 and the
   configured heat times. Then run **HBFM vs S-FLM at d=256 and d=512** — the
   capacity where S-FLM reaches 77.6%. This is the only configuration that
   actually tests H1; everything below it is a degenerate-capacity artifact.
   (This is the deferred geo_bridge change called out in EXPERIMENT.md §4.)

2. **Verify / harden the HBFM geodesic sampler `dest` construction** (ARCH §6).
   It is unvalidated new geometry and is a live confound for any HBFM↔SFM gap.
   Add a sanity check that the d=2 closed-form and general-d bridge agree on a
   shared case (the §11 `TBD`), and confirm the sampler state is active at
   checkpoint load (the §11 "eval protocol drift" audit).

3. **Address the ~11× per-step cost (OQ-1).** Cache / vectorize the radial grid
   so it is not rebuilt every `q_xt` call. Required before any d≥256 run is
   affordable. Re-probe with `hbfm/qxt_time` after the fix.

4. **Only then add seeds 2–3** at d≥256 to satisfy the 3-seed success/refute
   criteria. Running more seeds at d=64 is not informative.

---

## Bottom line

At the only capacity this experiment could reach (d ≤ 64), **HBFM does not match
S-FLM and produces zero correct grids — H1 is not supported as stated.** But the
d=64 S-FLM control collapses to 1.5% (from 77.6% at d=512), so this run measures a
broken-capacity regime forced by a `sample_radial` float64 overflow, **not** the
hyperbolic-vs-spherical geometry. The hyperbolic hypothesis remains **untested**;
the immediate blocker is the linear-space radial marginal, and fixing it to unlock
d≥256 is the single highest-value next step.
