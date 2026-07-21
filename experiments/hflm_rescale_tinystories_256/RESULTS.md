# hflm_rescale_tinystories_256 — Results

**Verdict: the √d radial rescale is sampler-fatal on TinyStories at EVERY init_std
tested (0.04 → 1.0). Negative result.** No cell generates coherent text; the
un-rescaled baseline (same knobs) does. Larger init_std does NOT fix it.

## Full init_std sweep (pc 1.0, K=−1, lr 3e-4, seq 256, 30k steps)

| init_std | Valid PPL (denoising bound) | GenPPL | entropy | sample text |
|---|---|---|---|---|
| 0.04 | 1.14 | 1.04 | 0.001 | `"sssss…"` (pure single token) |
| 0.30 | 1.14 | 1.15 | 0.043 | `"sssss…"` |
| 0.50 | 1.13 | 1.56 | 0.153 | `"sssss…"` |
| 1.00 | 1.12 | 23.69 | 1.348 | `"sssss blue superst sss cigarettes… diseases…"` |
| **baseline (no rescale)** | **9.9** | **17.7** | **4.03** | coherent stories |

**Do not read std 1.0's GenPPL 23.7 / entropy 1.35 as recovery.** The sample is still
"sssss" spam with random tokens injected — the higher entropy is *fake diversity* (a
long tail of stray tokens over an 's'-dominated distribution), not coherent text.
gpt2-large just finds token-salad moderately perplexing. Every cell is degenerate.

## The tell: denoising bound trivialized independent of std

`val/ppl ≈ 1.12–1.14` across ALL rescaled cells, vs **9.9** un-rescaled. The rescale
makes the denoising objective near-trivial regardless of init_std, and generation
collapses. That the objective is trivial *and constant* across std is the signature of
a broken geometry, not a mis-tuned knob.

## Likely mechanism (for the follow-up diagnosis, not more std sweeps)

The clamp compresses BOTH manifolds toward each other: noise radius
`rho_max·tanh(√(pc·d)/(rho_max·√d)) = 12·tanh(1/12) ≈ 1.0`, clean radius ≈ init_std.
The noise manifold is pinned at ~1.0 no matter what (it depends on √pc, not std), so the
flow has almost no radial transport to learn between clean and noise → the model
"denoises" by barely moving (val/ppl ~1.1) and the sampler, integrating a near-zero
velocity field from pure noise, collapses to a fixed point. The `tanh(1/12)` factor is
the smoking gun: dividing the *already-√d-scaled* prior radius by √d again crushes it
into the clamp's deep-linear regime where it barely varies.

## Recommendation: DROP the rescale (revert `_rho_clamp`)

Grounded in the rescaled-radius table (`clean=std`, `noise=12·tanh(√pc/12)`):

| geometry | clean_eff | noise_eff | ratio |
|---|---|---|---|
| baseline, NO rescale (works, GenPPL 17.7) | 1.11 | **11.77** | 10.6 |
| rescale std0.04 | 0.04 | 1.00 | 24.9 |
| rescale std0.5 | 0.50 | 1.00 | 2.0 |
| rescale std1.0 | 1.00 | 1.00 | **1.0** |

- **The rescale flattens the noise manifold from radius 11.8 → 1.0**, and the rescaled
  noise radius `= 12·tanh(√pc/12)` depends only on `prior_cov`, NOT on init_std or d.
  So no init_std can restore it; at std 1.0 clean = noise (ratio 1.0) ⇒ zero radial
  transport. HFLM works *because* noise sits deep in H^d (r≈11.8, cosh≈6.6e4); the
  rescale moves it to r≈1.0 (cosh 1.5, near-flat) and kills the flow.
- **"Move the √d divide outside the clamp" is a non-fix** — `12·tanh(ρ/12/√d)` and
  `12·tanh((ρ/√d)/12)` are algebraically identical (verified). (Corrects an earlier
  draft of this file.)
- Cranking `prior_cov` to ~205 would nominally restore noise r≈10, but that just undoes
  the √d divide with an absurd knob — pointless.
- **If the real goal is dimension-independence across model widths, do it as a
  knob-scaling convention, not a geometry change:** hold absolute radii fixed with
  `std ∝ 1/√d`, `prior_cov ∝ 1/d`. Verified to keep (clean 1.10, noise 11.77) across
  d ∈ {384, 768, 1536}; at d=768 that IS the working baseline (std 0.04, pc 1.0). No
  `_rho_clamp` edit needed.

Raw: `outputs/hflm_rescale_tinystories_256/std*/eval/{ppl.json,samples_genppl.json}`.
Jobs 24252 (std0.04), 72626/72627/72628 (std0.3/0.5/1.0).
