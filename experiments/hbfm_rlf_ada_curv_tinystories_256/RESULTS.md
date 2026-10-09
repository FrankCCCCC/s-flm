# hbfm_rlf_ada_curv_tinystories_256 — Results

HBFM with the adaptive curvature (bound 4 around K = −0.5) on TinyStories seq 256; design and
hypotheses in `EXPERIMENT.md`. On this unconditional task λ is one learned global clock. Regenerate the
tables with `python sweep.py --pull && python report.py` (λ from `curvature_ts.py`, run on the A100 node
where the checkpoints live). GenPPL = gpt2-large retokenised generative perplexity of 64 samples (180
steps, exact velocity, top-k 1, greedy last step); read it with the entropy and the distinct-sample
count. `val/ppl` is the importance-weighted CE bound, not a likelihood.

**Status (2026-10-09 10:50): complete — training (stages 1–3, 18 runs), frontier k = 1 markers
(3 seeds each) and exact-velocity frontier curves (6 jobs: fix and ada seeds 1–3, 120 cells each,
~40 A100-h per job; the last, ada seed 1, finished 2026-10-09 09:20). No jobs running.**

<!-- report:begin -->
25 evaluated (cell, step, horizon) reads.

### Per read

| mode | rate | ub | LR | t_max | step | seed | λ | λ·t_max | val/ppl | GenPPL | entropy | uniq |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| exp | 3 | na | 3e-4 | 0.5 | 30000 | 1 | 0.316 | 0.158 | 1.786 | 14.88 | 3.863 | 64/64 |
| exp | 3 | na | 3e-4 | 0.5 | 30000 | 2 | 0.319 | 0.159 | 1.794 | 15.85 | 3.883 | 64/64 |
| exp | 3 | na | 3e-4 | 0.5 | 30000 | 3 | n/a | n/a | 1.778 | 15.40 | 3.920 | 64/64 |
| exp | 3 | na | 3e-4 | 0.8 | 30000 | 1 | 0.316 | 0.253 | 1.786 | 14.34 | 3.863 | 64/64 |
| exp | 3 | na | 3e-4 | 1.6 | 30000 | 1 | 0.316 | 0.505 | 1.786 | 13.80 | 3.881 | 64/64 |
| exp | 3 | na | 3e-4 | 3 | 30000 | 1 | 0.316 | 0.948 | 1.786 | 14.27 | 3.889 | 64/64 |
| exp | 3 | na | 3e-4 | 3 | 30000 | 2 | 0.319 | 0.956 | 1.794 | 14.80 | 3.868 | 64/64 |
| exp | 3 | na | 3e-4 | 3 | 30000 | 3 | n/a | n/a | 1.778 | 14.53 | 3.896 | 64/64 |
| exp | 3 | na | 1e-3 | 0.5 | 30000 | 1 | 0.314 | 0.157 | 1.774 | 15.71 | 3.886 | 64/64 |
| exp | 3 | na | 1e-3 | 0.8 | 30000 | 1 | 0.314 | 0.251 | 1.774 | 15.25 | 3.903 | 64/64 |
| exp | 3 | na | 1e-3 | 1.6 | 30000 | 1 | 0.314 | 0.502 | 1.774 | 13.54 | 3.893 | 64/64 |
| exp | 3 | na | 1e-3 | 3 | 30000 | 1 | 0.314 | 0.942 | 1.774 | 14.57 | 3.888 | 64/64 |
| exp | 20 | na | 3e-4 | 0.35 | 30000 | 1 | 0.250 | 0.088 | 1.425 | 27.71 | 3.671 | 64/64 |
| exp | 20 | na | 3e-4 | 0.5 | 30000 | 1 | 0.250 | 0.125 | 1.425 | 24.34 | 3.748 | 64/64 |
| exp | 20 | na | 3e-4 | 3 | 30000 | 1 | 0.250 | 0.750 | 1.425 | 209.54 | 3.942 | 64/64 |
| texp | 3 | 0.5 | 3e-4 | 0.5 | 30000 | 1 | 0.250 | 0.125 | 1.483 | 12.56 | 3.910 | 64/64 |
| texp | 3 | 0.5 | 3e-4 | 0.5 | 30000 | 2 | 0.250 | 0.125 | 1.481 | 12.75 | 3.927 | 64/64 |
| texp | 3 | 0.5 | 3e-4 | 0.5 | 30000 | 3 | 0.250 | 0.125 | 1.483 | 12.69 | 3.966 | 64/64 |
| texp | 3 | 0.75 | 3e-4 | 0.75 | 30000 | 1 | 0.250 | 0.188 | 1.579 | 12.06 | 3.946 | 64/64 |
| texp | 3 | 1 | 3e-4 | 1 | 30000 | 1 | 0.250 | 0.250 | 1.639 | 11.91 | 3.930 | 64/64 |
| texp | 3 | 1 | 3e-4 | 1 | 30000 | 2 | 0.250 | 0.250 | 1.643 | 12.35 | 3.957 | 64/64 |
| texp | 3 | 1 | 3e-4 | 1 | 30000 | 3 | 0.250 | 0.250 | 1.641 | 12.37 | 3.959 | 64/64 |
| texp | 3 | 1.5 | 3e-4 | 1.5 | 30000 | 1 | 0.250 | 0.375 | 1.708 | 12.46 | 3.922 | 64/64 |
| texp | 3 | 2 | 3e-4 | 2 | 30000 | 1 | 0.285 | 0.570 | 1.758 | 14.02 | 3.893 | 64/64 |
| unif | – | 2 | 3e-4 | 2 | 30000 | 1 | 0.250 | 0.500 | 1.711 | 12.94 | 3.894 | 64/64 |

### Mean ± std over seeds

| mode | rate | ub | LR | t_max | step | n | λ | GenPPL | entropy |
|---|---|---|---|---|---|---:|---:|---:|---:|
| exp | 3 | na | 3e-4 | 0.5 | 30000 | 3 | 0.317 ± 0.002 | 15.37 ± 0.49 | 3.889 ± 0.029 |
| exp | 3 | na | 3e-4 | 0.8 | 30000 | 1 | 0.316 | 14.34 | 3.863 |
| exp | 3 | na | 3e-4 | 1.6 | 30000 | 1 | 0.316 | 13.80 | 3.881 |
| exp | 3 | na | 3e-4 | 3 | 30000 | 3 | 0.317 ± 0.002 | 14.53 ± 0.27 | 3.884 ± 0.015 |
| exp | 3 | na | 1e-3 | 0.5 | 30000 | 1 | 0.314 | 15.71 | 3.886 |
| exp | 3 | na | 1e-3 | 0.8 | 30000 | 1 | 0.314 | 15.25 | 3.903 |
| exp | 3 | na | 1e-3 | 1.6 | 30000 | 1 | 0.314 | 13.54 | 3.893 |
| exp | 3 | na | 1e-3 | 3 | 30000 | 1 | 0.314 | 14.57 | 3.888 |
| exp | 20 | na | 3e-4 | 0.35 | 30000 | 1 | 0.250 | 27.71 | 3.671 |
| exp | 20 | na | 3e-4 | 0.5 | 30000 | 1 | 0.250 | 24.34 | 3.748 |
| exp | 20 | na | 3e-4 | 3 | 30000 | 1 | 0.250 | 209.54 | 3.942 |
| texp | 3 | 0.5 | 3e-4 | 0.5 | 30000 | 3 | 0.250 ± 0.000 | 12.67 ± 0.10 | 3.934 ± 0.028 |
| texp | 3 | 0.75 | 3e-4 | 0.75 | 30000 | 1 | 0.250 | 12.06 | 3.946 |
| texp | 3 | 1 | 3e-4 | 1 | 30000 | 3 | 0.250 ± 0.000 | 12.21 ± 0.26 | 3.949 ± 0.016 |
| texp | 3 | 1.5 | 3e-4 | 1.5 | 30000 | 1 | 0.250 | 12.46 | 3.922 |
| texp | 3 | 2 | 3e-4 | 2 | 30000 | 1 | 0.285 | 14.02 | 3.893 |
| unif | – | 2 | 3e-4 | 2 | 30000 | 1 | 0.250 | 12.94 | 3.894 |
<!-- report:end -->

## Conclusion (training complete 2026-10-07; frontier complete 2026-10-09)

**Best configuration with adaptive curvature on:** `trunc_exp`, `time_exp_rate` 3,
`time_range_upper_bound` 1, LR 3e-4 (bound 4 around K = −0.5), sampler t_max = ub = 1 →
**GenPPL 12.21 ± 0.26 at entropy 3.949 ± 0.016** (3 seeds; 180 steps, 64 samples), and
12.31 ± 0.11 @ 3.958 / 12.44 ± 0.26 @ 3.956 at NFE 256 / 128 on the frontier protocol (512 samples).

| | GenPPL ↓ | entropy ↑ | protocol |
|---|---:|---:|---|
| **HBFM adaptive (trunc_exp ub 1)** | **12.21 ± 0.26** | **3.949** | 180 steps, 64 samples, 3 seeds |
| HBFM fixed curvature, best (trunc_exp ub 0.5) | 12.64 ± 0.43 | 3.934 | same |
| S-FLM + trunc + ada (LR 1e-3) | 12.95 ± 0.65 | 3.945 | same |
| E-FLM, best (R 0.5, LR 1e-3) | 11.44 ± 0.54 | 3.82 | same, lower entropy |

1. **Adaptive vs fixed curvature, matched entropy:** on the frontier protocol the adaptive model is
   5.7–9.6 % lower in GenPPL at every NFE from 8 to 256 (3 seeds, sd 0.1–0.3; NFE 64: t ≈ 9), and
   1–11 % lower along the exact-velocity temperature curves (every NFE 8–256, H 4.20–4.40; 3–11 %
   except at NFE 256, H 4.40: 1 %). The 64-sample evals point the same way
   (−0.43) but are too noisy to resolve it (p ≈ 0.2).
2. **Against S-FLM + trunc + ada it depends on the sampler, the NFE and the entropy band.** With the
   spec's top-1 sampler (entropy ≈ 3.95) HBFM adaptive is 0.74 lower (identical sampler settings,
   7 of 9 seed pairs, p ≈ 0.17). On the temperature curves of the frontier (exact velocity,
   T 0.50–1.20, matched entropy) HBFM adaptive is above S-FLM and E-FLM at every NFE ≥ 8: by
   7–14 % at NFE 128–256 in the band 4.20–4.30, by 18–33 % at NFE 16–32, by 26–47 % at the top
   of its range (H ≈ 4.40) and by 37–64 % at NFE 8. MDLM, the lowest curve at NFE ≥ 32, is 10–30 %
   below S-FLM in that band; at NFE 8 HBFM adaptive is below MDLM across the band.
3. **What the curvature does here.** Without a prompt λ is one global number (min = max at every
   checkpoint), and in 6 of 9 configurations it settles at the lower guard 1/4 (the other three are
   still falling at 30k: 0.285 for `trunc_exp` ub 2, ≈ 0.315 for `exp` rate 3): every proposal is
   bounded (`exp` too, by u ≥ 1e-3), so the loss mean grows with λ and its second-moment gradient
   (variance + mean²) shrinks the trained clock range. The adaptive curvature thus acts as a
   learned re-parametrisation of the proposal (rate × 4, range ÷ 4); the best cell trains
   `trunc_exp` at rate 12 on the clock range [0, 0.25].
4. **The axes:** `trunc_exp` ub is a smooth U with its floor at 0.75–1 (12.06 / 11.91 vs 12.56 at
   0.5, 12.46 at 1.5, 14.02 at 2); `exp` is worse (spec 14.53, best clock 13.80); a too-flat
   proposal is repaired (`unif` ub 2: 17.18 → 12.94) and a too-fast one broken (`exp` rate 20: λ at
   the guard leaves a trained range of [0, 0.086], GenPPL 24–210); LR 1e-3 loses to 3e-4.
5. **Open:** (a) a fixed-curvature run at the best cell's effective proposal (`trunc_exp` rate 12,
   ub 0.25) — separates the curvature's contribution from the proposal's; (b) training λ on the
   variance alone (centre the hook's factor), so it adapts rather than riding the guard;
   (c) per-position curvature for unconditional text needs a per-sequence latent (MuLAN-style).

## Stage 1 (seed 1, 30k steps)

GenPPL at the cell's own horizon (64 samples; 64/64 distinct in every read):

| id | cell | t_max | λ (30k) | clock horizon λ·t_max | GenPPL | entropy | fixed-curvature reference (own / t_max 0.5) |
|---|---|---:|---:|---:|---:|---:|---|
| F | trunc_exp, rate 3, ub 1 | 1 | 0.250 | 0.25 | **11.91** | 3.930 | — |
| B | trunc_exp, rate 3, ub 0.5 | 0.5 | 0.250 | 0.125 | 12.56 | 3.910 | 12.64 ± 0.43 (the fixed optimum) |
| G | trunc_exp, rate 3, ub 2 | 2 | 0.285 | 0.57 | 14.02 | 3.893 | — |
| A | exp, rate 3 (spec) | 3 / 0.5 | 0.316 | 0.95 / 0.16 | 14.27 / 14.88 | 3.889 / 3.863 | 15.58 ± 0.67 / 13.25 ± 0.56 |
| L | exp, rate 3, LR 1e-3 | 3 / 0.5 | 0.314 | 0.94 / 0.16 | 14.57 / 15.71 | 3.888 / 3.886 | — |
| C | unif, ub 2 | 2 | 0.250 | 0.5 | 12.94 | 3.894 | 17.18 / 15.80 |
| D | exp, rate 20 | 3 / 0.5 | 0.250 | 0.75 / 0.125 | 209.5 / 24.34 | 3.942 / 3.748 | 19.01 / 16.36 |

A seed 2 (2080 Ti): 14.80 / 15.85 @ 3.868 / 3.883, λ 0.319 — A over two seeds 14.53 ± 0.38 at t_max 3
(fixed: 15.58 ± 0.67) and 15.36 ± 0.69 at t_max 0.5 (fixed: 13.25 ± 0.56).

- **Best read: F, 11.91 @ 3.93** — 0.73 below the fixed-curvature optimum (12.64 ± 0.43, 3 seeds) at a
  higher entropy, and above E-FLM's 11.44 ± 0.54 @ 3.82 at a higher entropy. One seed: the fixed study's
  single-seed gaps below ~8 % were not resolved; F's seeds 2–3 are in Stage 2.
- **The flat proposal is repaired** (`EXPERIMENT.md` H2): `unif` ub 2 goes from 17.18 (fixed) to 12.94,
  where the fixed `unif` ub 0.5 sits (13.19 ± 0.36) — its clock is pinned at 1/4, so it trains U[0, 0.5].
- **The fast exp proposal is broken, not repaired** (H1 refuted): D's clock went to 1/4, not up; it
  trains the clock range [0, 0.086] and its sampler decodes at clock 0.75 / 0.125 — word salad at
  t_max 3 (GenPPL 209.5). The read at the end of its trained range (t_max 0.35) is below.
- **At a matched clock horizon the exp cells stay behind the fixed spec model.** A (λ 0.316) read at
  t_max 0.5 / 0.8 / 1.6 / 3 = clock 0.16 / 0.25 / 0.51 / 0.95 gives 14.88 / 14.34 / **13.80** / 14.27
  (L, LR 1e-3: 15.71 / 15.25 / **13.54** / 14.57). The fixed spec model's best horizon is 0.5
  (12.90 seed 1, 13.25 ± 0.56 over three) — its clock is the physical time, so clock 0.5 there too.
  At its own best clock the adaptive spec cell is 0.5–0.9 worse.
- **D read at the end of its trained range** (t_max 0.35, clock 0.088): 27.71 @ 3.671 — worse than at
  t_max 0.5. Its trained clock range ends before the posterior resolves the words (the fixed study's
  context-free table puts the resolution scale near clock 1.6); the cell is broken at every horizon.
- **LR 1e-3 loses to 3e-4** at the spec point at both horizons (14.57 vs 14.27, 15.71 vs 14.88): no LR
  axis in stage 2 (the rule fixed in `EXPERIMENT.md`).
- `val/ppl` (1.48–1.79) integrates the CE over the trained clock range [0, λ·ub], which differs per cell;
  it is not comparable across cells.

### The learned clock goes to the guard

λ per checkpoint (EMA weights, `curvature_ts.py`; min = max at every checkpoint: one global value):

| cell | 5k | 10k | 15k | 20k | 25k | 30k |
|---|---:|---:|---:|---:|---:|---:|
| A exp, rate 3 | 0.432 | 0.376 | 0.351 | 0.337 | 0.325 | 0.316 |
| L exp, rate 3, LR 1e-3 | 0.408 | 0.368 | 0.345 | 0.332 | 0.322 | 0.314 |
| B trunc_exp ub 0.5 | 0.270 | 0.251 | 0.250 | 0.250 | 0.250 | 0.250 |
| F trunc_exp ub 1 | 0.284 | 0.252 | 0.250 | 0.250 | 0.250 | 0.250 |
| G trunc_exp ub 2 | 0.395 | 0.339 | 0.316 | 0.302 | 0.292 | 0.285 |
| C unif ub 2 | 0.259 | 0.251 | 0.250 | 0.250 | 0.250 | 0.250 |
| D exp, rate 20 | 0.260 | 0.251 | 0.250 | 0.250 | 0.250 | 0.250 |

The trajectory is reproducible across seeds: A seed 2 0.440 / 0.379 / 0.356 / 0.340 / 0.327 / 0.319,
F seed 2 0.291 / 0.252 / 0.250 (5k–15k).

Every cell's λ falls; five of seven sit at the lower guard 1/4 from ~15k on. That includes the exp
cell at rate 20, for which `EXPERIMENT.md` H1 predicted λ ≈ 20/7 (refuted). The cause is that **no
proposal here is unbounded**: HBFM draws the diffusion time in [1e-3, 1] (`training.sampling_eps`) and
the log-linear schedule gives u = 1 − α_t ≥ 1e-3, so `exp` is itself truncated at the heat time
−log(1e-3)/rate = 6.9/rate (2.3 at rate 3, 0.35 at rate 20). On a bounded range the mean of the loss,
∫₀^{λ·ub} F, grows with λ (`math.md` §3 needs T → ∞), and λ is trained on E[L̂²] = variance + mean²
(§6): the mean² term pulls λ down until the guard stops it. Only exp at rate 3, whose range 2.3 covers
almost all of the CE mass, keeps a slow drift (0.43 → 0.32) instead of a collapse.

So in this experiment the adaptive curvature acts as a re-parametrisation of the proposal pinned at the
guard: F trains `trunc_exp` at rate 3/λ = 12 on the clock range [0, 0.25] and decodes at clock 0.25; B
the same on [0, 0.125]; C U[0, 0.5]; D Exp(80) on [0, 0.086] while its sampler walks to clock 0.75 / 0.125
— past the trained range, hence its word salad. A fixed-curvature cell with the same effective proposal as F (`trunc_exp`, rate 12,
ub 0.25) would separate "the adaptive curvature helps" from "this proposal helps"; it is not in the
spec's search space (adaptive curvature on in every run) and is not run.

A variance-only gradient for λ would centre the hook's factor: `2 (L̂_b − mean_b L̂_b)` instead of
`2 L̂_b` (`algo.HyperbolicBoundaryFM.nll`), the batch estimate of ∇Var = ∇E[L̂²] − ∇(E L̂)². Not applied:
it changes the method.

## Stage 3 — the trunc_exp range around F (seed 1, 4× A6000)

| ub | 0.5 (B) | 0.75 | 1 (F) | 1.5 | 2 (G) |
|---|---:|---:|---:|---:|---:|
| GenPPL | 12.56 | 12.06 | **11.91** | 12.46 | 14.02 |
| entropy | 3.910 | 3.946 | 3.930 | 3.922 | 3.893 |

A smooth U over the range with its floor at ub 0.75–1: 0.75 and 1 are within 1.3 % of each other
(single seeds; seed spread ~3–4 %), and 0.75 has the highest entropy of the sweep. Both are on the
Gen. PPL / entropy front of the HBFM cells. λ reaches the 1/4 guard here too — from 15k at ub 0.75
(0.288 / 0.252 / 0.250) and from 25k at ub 1.5 (0.339 / 0.274 / 0.254 / 0.250) — so at the guard
the five cells train `trunc_exp` at effective rate 12 on the clock ranges [0, 0.125], [0, 0.19],
[0, 0.25], [0, 0.375] (G: λ 0.285, rate 10.5, [0, 0.57]); the best range is 0.19–0.25.

## Stage 2 — seeds

180 steps, 64 samples, Gen. PPL @ entropy:

| cell | seed 1 | seed 2 | seed 3 | mean ± sd |
|---|---|---|---|---|
| **F** adaptive, trunc_exp ub 1 | 11.91 @ 3.930 | 12.35 @ 3.957 | 12.37 @ 3.959 | **12.21 ± 0.26 @ 3.949 ± 0.016** |
| fixed best, trunc_exp ub 0.5 (`claude_test_hbfm_tinystories_256`) | 12.59 @ 3.918 | 12.25 @ 3.922 | 13.10 @ 3.963 | 12.64 ± 0.43 @ 3.934 ± 0.025 |
| S-FLM + trunc + ada, LR 1e-3 (`naive_ar_tinystories_s256`) | 12.20 @ 3.933 | 13.39 @ 3.956 | 13.24 @ 3.945 | 12.95 ± 0.65 @ 3.945 ± 0.011 |
| B adaptive, trunc_exp ub 0.5 | 12.56 @ 3.910 | 12.75 @ 3.927 | 12.69 @ 3.966 | 12.67 ± 0.10 @ 3.934 ± 0.028 |
| A adaptive, exp rate 3, t_max 3 | 14.27 @ 3.889 | 14.80 @ 3.868 | 14.53 @ 3.896 | 14.53 ± 0.27 @ 3.884 ± 0.015 |
| A adaptive, exp rate 3, t_max 0.5 | 14.88 @ 3.863 | 15.85 @ 3.883 | 15.40 @ 3.920 | 15.37 ± 0.49 @ 3.889 ± 0.029 |
| fixed spec, exp rate 3, t_max 3 / 0.5 (`claude_test_hbfm_tinystories_256`) | | | | 15.58 ± 0.67 / 13.25 ± 0.56 |

The sampler settings are identical across the four HBFM / S-FLM rows (exact velocity, top-k 1,
greedy last step, T 1, float64 state, EMA weights, gpt2-large); only the method's sampler and its own
truncation differ (HBFM t_max = ub; S-FLM α_max 0.121).

- **F, the best HBFM cell, after three seeds:** −0.43 Gen. PPL against the fixed-curvature optimum and
  −0.74 against S-FLM + trunc + ada, at slightly higher entropy (+0.015 / +0.004). Consistent in
  direction (F wins 7 of the 9 seed pairs against each) but not significant at three seeds (Welch
  t ≈ 1.5, p ≈ 0.2; t ≈ 1.8, p ≈ 0.17). Seed 1 (11.91, the stage-1 pick) was the luckiest draw.
- **At the spec point (exp rate 3) the adaptive curvature loses at the best horizon of each model:**
  14.53 ± 0.27 (t_max 3; its best single read was 13.80 at t_max 1.6) vs the fixed 13.25 ± 0.56
  (t_max 0.5). It wins only at the shared default horizon t_max 3 (14.53 vs 15.58), which on the
  adaptive model's clock (λ ≈ 0.32) is near the good range.
- **At the fixed optimum's own proposal the adaptive curvature changes the mean by nothing** (B
  12.67 ± 0.10 vs 12.64 ± 0.43, entropy 3.934 both) **and the seed spread by a factor of four.**
- **Attribution:** with λ at the 1/4 guard, F trains the proposal `trunc_exp` rate 12 on the clock
  range [0, 0.25] — a cell the fixed-curvature study never trained. Whether the gain belongs to the
  adaptive curvature or to that proposal needs the fixed-curvature control at it (not run: outside
  the spec's "adaptive curvature in every run").

## Frontier — HBFM on the Gen. PPL / entropy frontier

Protocol of `experiments/naive_ar_tinystories_s256/FRONTIER_RESULTS.md` (NFE 1–256, 512 samples per
cell, 3 training seeds); design in `EXPERIMENT.md` → Frontier evaluation. Figure:
`figures/genppl_entropy_frontier_line.png` (k = 1 markers and exact-velocity curves: 3 seeds; redrawn
2026-10-09 10:50); cells in `frontier_cells.csv`.

**Truncation pick** (seed 1, `top_k_velocity = 1`, Gen. PPL @ entropy):

| model | t_max | NFE 4 | NFE 8 | NFE 16 | NFE 32 | NFE 64 | NFE 128 | NFE 256 | Σ log PPL (8–256) |
|---|---:|---|---|---|---|---|---|---|---:|
| ada | 0.25 | 43.01 @ 3.354 | 29.57 @ 3.667 | 23.53 | 21.07 | 19.91 | 20.03 | 19.16 @ 3.852 | 18.534 |
| ada | 0.5 | 34.06 @ 3.654 | 19.98 @ 3.835 | 16.28 | 14.50 | 14.02 | 13.78 | 13.37 @ 3.935 | 16.316 |
| ada | 0.75 | 37.94 @ 3.619 | 18.60 @ 3.865 | 14.76 | 13.50 | 12.78 | 12.73 | 12.43 @ 3.950 | 15.829 |
| ada | **1** | 41.35 @ 3.109 | 19.91 @ 3.866 | 14.90 | 13.32 | 12.65 | 12.16 | 12.20 @ 3.958 | **15.819** |
| fix | **0.25** | 43.59 @ 3.664 | 21.48 @ 3.859 | 16.61 | 14.73 | 13.63 | 13.28 | 12.88 @ 3.938 | **16.321** |
| fix | 0.35 | 61.26 @ 2.917 | 24.99 @ 3.855 | 17.14 | 14.71 | 13.57 | 13.14 | 12.87 @ 3.948 | 16.487 |
| fix | 0.5 | 58.50 @ 2.314 | 35.91 @ 3.762 | 19.23 | 15.58 | 14.03 | 13.23 | 12.87 @ 3.944 | 17.063 |

NFE 1 decodes the origin: one distinct sample, entropy 0, in every cell (as for the baselines).

- **Picked: ada t_max = 1, fix t_max = 0.25** — both decode at clock 0.25. For ada 0.75 and 1 tie
  (15.829 vs 15.819); for fix the short horizon matters at low NFE only (NFE 8: 21.5 vs 35.9 at
  t_max 0.5; NFE 128–256: all three within 0.4).
- **Seed 1, matched truncation:** the adaptive model's marker is below the fixed one's at every NFE
  from 8 to 256 (−5 % to −10 % Gen. PPL) at equal or higher entropy.
- **Three seeds, 512 samples per cell (k = 1 markers, Gen. PPL ± sd @ entropy):**

  | NFE | ada (t_max 1) | fix (t_max 0.25) | Δ |
  |---:|---|---|---:|
  | 8 | 19.39 ± 0.75 @ 3.853 | 21.21 ± 0.76 @ 3.861 | −8.6 % |
  | 16 | 14.92 ± 0.09 @ 3.891 | 16.51 ± 0.32 @ 3.897 | −9.6 % |
  | 32 | 13.46 ± 0.13 @ 3.924 | 14.68 ± 0.27 @ 3.926 | −8.3 % |
  | 64 | 12.74 ± 0.13 @ 3.945 | 13.74 ± 0.14 @ 3.945 | −7.3 % |
  | 128 | 12.44 ± 0.26 @ 3.956 | 13.26 ± 0.19 @ 3.950 | −6.2 % |
  | 256 | 12.31 ± 0.11 @ 3.958 | 13.05 ± 0.21 @ 3.958 | −5.7 % |

  **At equal entropy the adaptive model is better at every NFE from 8 to 256** (NFE 64: t ≈ 9). The
  512-sample reads have a seed spread of 0.1–0.3, a third of the 64-sample reads' (0.26–0.43), which
  is why the stage-2 comparison above could not resolve what this one does. Still to separate: the
  adaptive curvature from the proposal it pins (λ = 1/4: trunc_exp rate 12 on [0, 0.25]).
- **Exact-velocity temperature curves (T 0.50–1.20), matched entropy.** Complete for 3 seeds (the
  last job, ada seed 1, finished 2026-10-09 09:20). Gen. PPL at H*, the highest entropy every method
  reaches (≈ 4.41; the plotted mean curve over the seeds):

  | NFE | H* | MDLM | DUO | S-FLM + trunc + ada | E-FLM | **HBFM ada** | HBFM fix |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 8 | 4.369 | 44.76 | 71.27 | **26.20** | 28.32 | 40.94 | 45.00 |
  | 16 | 4.406 | 28.77 | 50.34 | **24.26** | 24.56 | 36.17 | 38.26 |
  | 32 | 4.415 | **22.78** | 38.06 | 23.14 | 22.88 | 33.16 | 34.37 |
  | 64 | 4.411 | **20.22** | 33.46 | 22.09 | 21.67 | 30.11 | 31.24 |
  | 128 | 4.408 | **19.39** | 31.53 | 21.55 | 21.23 | 27.68 | 28.85 |
  | 256 | 4.409 | **19.10** | 30.65 | 21.44 | 21.01 | 28.16 | 28.51 |

  H* is near the top of HBFM's range (at NFE 64–256 its curves end at 4.48–4.51, T 1.20), where they are
  steepest: ada seed 2 at NFE 256 rises 165 Gen. PPL per nat between T 1.00 and 1.05, 39 between
  T 0.70 and 0.75. Across the band every curve covers — from ≈ 4.19, the highest T = 0.50 start
  (S-FLM's) — each seed's curve interpolated at H, then the mean over seeds (seed sd ≤ 1.0 in every
  cell but DUO at NFE 8, H 4.30: 1.4):

  | NFE | H | MDLM | DUO | S-FLM + trunc + ada | E-FLM | **HBFM ada** | HBFM fix | ada vs S-FLM | ada vs fix |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 8 | 4.20 | 31.08 | 24.19 | **19.28** | 20.44 | 26.39 | 29.49 | +37 % | −11 % |
  | 8 | 4.30 | 38.00 | 34.63 | **22.23** | 24.01 | 32.28 | 36.06 | +45 % | −10 % |
  | 16 | 4.20 | 16.52 | **15.28** | 16.10 | 15.92 | 20.22 | 21.96 | +26 % | −8 % |
  | 16 | 4.30 | 20.45 | 21.09 | **18.44** | 18.51 | 24.45 | 26.51 | +33 % | −8 % |
  | 32 | 4.20 | **12.23** | 12.53 | 14.89 | 14.50 | 17.50 | 18.67 | +18 % | −6 % |
  | 32 | 4.30 | **15.39** | 16.64 | 17.08 | 16.75 | 21.07 | 22.40 | +23 % | −6 % |
  | 64 | 4.20 | **10.77** | 11.59 | 14.39 | 13.96 | 16.02 | 17.10 | +11 % | −6 % |
  | 64 | 4.30 | **13.41** | 15.66 | 16.52 | 16.03 | 19.52 | 20.28 | +18 % | −4 % |
  | 128 | 4.20 | **10.05** | 11.02 | 14.09 | 13.52 | 15.28 | 16.13 | +8 % | −5 % |
  | 128 | 4.30 | **12.79** | 14.73 | 16.13 | 15.74 | 18.44 | 19.19 | +14 % | −4 % |
  | 256 | 4.20 | **9.74** | 11.05 | 13.99 | 13.40 | 14.95 | 15.75 | +7 % | −5 % |
  | 256 | 4.30 | **12.44** | 15.13 | 16.00 | 15.59 | 18.21 | 18.79 | +14 % | −3 % |

  - **HBFM's gap depends on the NFE and the entropy.** Against S-FLM + trunc + ada, HBFM adaptive
    is +7–14 % at NFE 128–256 in the 4.20–4.30 band and widens to +28–31 % at H*; at NFE 8–32 it
    is +18–64 % across the band. MDLM is lowest at H ≤ 4.30 for NFE ≥ 32, with DUO below HBFM
    there too; at NFE 8 HBFM adaptive is below MDLM across the band and below DUO from 4.30, and at
    H ≈ 4.40, where DUO's curve turns up, it is at or below DUO at every NFE. The earlier "~40–50 %
    worse" holds at H* for NFE 8–64 only, and not against MDLM at NFE 8.
  - **The adaptive curvature is below fixed at every (NFE, H)**: by 3–11 % in the band 4.20–4.35,
    by 5–8 % at H 4.40 for NFE 8–128 and by 1 % at NFE 256 (seed sd 0.8–0.9 there) —
    the k = 1 markers' result, now along the whole temperature curve.
  - **NFE 128 → 256:** MDLM, S-FLM and E-FLM gain 0.1–0.3 at H*; for HBFM the change there is
    within the seed spread (per seed: ada 28.23 → 28.30, 26.70 → 28.91 and 27.78 → 27.18, mean
    27.57 → 28.13; fix 28.38 → 29.09, 28.58 → 28.60 and 29.47 → 27.46, mean 28.81 → 28.38). In the
    band HBFM gains 0.2–0.4 (MDLM, S-FLM and E-FLM 0.1–0.35; DUO none). At a fixed T the extra
    steps lower HBFM's Gen. PPL for every seed and T (90 of 90 pairs) and move its entropy by at
    most ±0.02 (lower in 69 of 90).
  - The lowest point of every curve is at T 0.50: HBFM's curves start at entropy 4.14–4.16 (S-FLM's
    at 4.18, MDLM's at 4.04), so HBFM's low-entropy regime (≈ 3.95) is reached only by the k = 1
    sampler.
- **Against the other models:** HBFM's k = 1 markers sit at entropy 3.86–3.96, left of where the
  temperature curves of MDLM / DUO / S-FLM start (≈ 4.0+) and right of E-FLM's k = 1 marker
  (3.68–3.81, Gen. PPL 11.3–16.5). The matched-entropy comparison is therefore made on HBFM's own
  temperature curves above (the exact-velocity arm: ~40 A100-h per model and seed, 120 cells, an
  NFE 256 cell takes 80 min).

## Incidents

- **2026-10-04 10:21, A seed 2 on the 2080 Ti node (job 951300): `CUDA error: an illegal memory
  access` on rank 1 at step 855**, before the first checkpoint; no warning before it. Not a second
  process: device cgroups are enforced on desa-compute-01 (a job sees only its own GPU) and the node's
  only other job was CPU-only. The same code ran ~150 GPU-h of Sudoku on this node without it; the
  TinyStories model is larger (10.1 of 11 GB per GPU). Resubmitted once (job 951523); a second crash
  moves the cell to the A100 queue.
- **2026-10-04/05, F and G preempted on `nlplarge`** (other users' higher-tier jobs) and requeued by
  Slurm; both resumed from `checkpoints/last.ckpt` (≤ 1k steps lost) and finished at 05:51 / 09:22.
  The requeue overwrote the first part of their logs.
