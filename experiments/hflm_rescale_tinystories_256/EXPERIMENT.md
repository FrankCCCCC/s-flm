# hflm_rescale_tinystories_256 — rescaled H-FLM init × prior_cov sweep

Branch `var_scale`. The √d radial rescale in `HFLM._rho_clamp`
(`rho_eff = rho_max·tanh(‖·‖ / rho_max / √d)`, d=768) de-saturates the embedding
radial coordinate off the Poincaré boundary (setup.md numerical example). This sweep
tests whether the rescaled H-FLM can be made to generate by tuning **prior_cov** —
the knob that sets the rescaled noise radius.

## Background: why prior_cov (not init_std)

Round-1 std-sweep (RESULTS.md) collapsed at every init_std 0.04→1.0 (pc fixed 1.0).
Root cause: the rescaled **noise** radius `= 12·tanh(√pc/12)` depends only on
prior_cov, so pc 1.0 pins it at ≈1.0 and no init_std restores clean↔noise transport.
This sweep raises prior_cov to push the noise radius back out.

## Grid (15 cells; setup.md)

| axis | values | n |
|---|---|---|
| init | random (N(0,4e-4)=std 0.02), custom std 0.01, custom std 0.04 | 3 |
| prior_cov | 2.0, 4.0, 8.0, 10.0, 16.0 | 5 |

Fixed: small-hyperbolic-dit (768/12/12), **K=−1.0**, rho_max 12, self-cond off,
noise log-linear, seq **256**, 30k steps, global batch 512 (1 GPU × PER_GPU_BS 32,
accum 16), bf16, EMA 0.9999, AdamW lr 3e-4 wd 0 betas (0.9,0.999) eps 1e-8 clip 1.0.
Eval: exact velocity, **top_k_velocity 1**, 180 steps, greedy last (ppl + GenPPL).

## Rescaled geometry across the grid (γ = ball radius, boundary = 1)

| prior_cov | noise ρ | noise γ | clean γ (std 0.04) | noise/clean ratio |
|---|---|---|---|---|
| (1.0 — round 1, collapsed) | 1.00 | 0.46 | 0.020 | 23 |
| 2.0 | 1.41 | 0.61 | 0.020 | 30 |
| 4.0 | 1.98 | 0.76 | 0.020 | 38 |
| 8.0 | 2.78 | 0.88 | 0.020 | 44 |
| 10.0 | 3.09 | 0.91 | 0.020 | 46 |
| 16.0 | 3.86 | 0.96 | 0.020 | 48 |

## Hypothesis

If the collapse was clean↔noise overlap (noise γ 0.46 ≈ clean band), then pushing
noise γ toward the boundary (0.61→0.96 as pc 2→16) should restore transport and
recover non-degenerate generation (entropy back toward ~4, coherent text). Success
bar: any cell with entropy ≳ 2 and coherent samples; stretch: match/beat the
un-rescaled baseline (GenPPL 17.7, entropy 4.03).

## GPU & wall clock

15 × 1 GPU on unicorn `thickstun,desa` (excl. desa-compute-01), 30k steps ≈ 15–16 h
each, `--requeue` (auto-resume). Prioritized with `nice`. Deliverables:
`outputs/hflm_rescale_tinystories_256/{init}_pc{pc}_K-1.0/{checkpoints,eval/{ppl,samples_genppl}.json}`.
Round-1 std-sweep results retained in RESULTS.md.
