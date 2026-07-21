# hflm_rescale_tinystories_256 — √d-rescaled radial clamp for H-FLM (TinyStories, seq 256)

Branch `var_scale`. Single training run testing the **√d radial rescale** in
`algo.py` `HFLM._rho_clamp` at a large-noise geometry (`prior_cov=1.0`).

## The code change (algo.py, `HFLM._rho_clamp`)

Original soft clamp: `rho_eff = rho_max · tanh(rho / rho_max)`.
New (`norm_by="sqrt_dim"`): `rho_eff = rho_max · tanh(rho / rho_max / √hidden_size)`.

- `hidden_size` = `backbone.embed_dim` (hyperbolic_dit.py:28), i.e. the true
  dimension of the radial norm — so the divide exactly cancels the √d growth of
  `rho`. Applied consistently to clean embeddings, the wrapped-normal prior, and
  the self-cond embedding table (algo.py:680, 681, 799).
- Still `tanh ≤ 1 ⇒ rho_eff < rho_max = 12 < _LORENTZ_RHO_MAX = 20`, so the
  overflow guard is preserved. The change only rescales the *operating* radius,
  it does not relax the cap.

**Why it's needed here.** `rho ~ √(prior_cov · d)` for the wrapped-normal prior
(cov = per-dim variance). At `prior_cov=1.0`, `d=768` ⇒ prior radius ≈ √768 ≈
27.7, far above `rho_max=12`: the original `tanh(27.7/12) ≈ 0.98` pins *every*
noise sample at ≈ `rho_max`, collapsing the radial distribution. The √d rescale
puts the clamp back in its near-linear (information-preserving) regime, giving
**dimension-independent radii**: clean ≈ `init_std`, noise ≈ `√prior_cov`.

Resulting geometry for this run (K=−1): clean embeddings at radius ≈ 0.04 (near
the flat origin), prior noise at radius ≈ 1.0 (where curvature is O(1)); geodesics
cross the curved band. Ratio prior/clean ≈ 25, same as un-rescaled (uniform √d
contraction preserves relative geometry).

## Run config (single cell)

| knob | value |
|---|---|
| algo | hflm (log-linear noise, no trunc/ada) |
| init | custom, **std 0.04** |
| prior_cov | **1.0** |
| gaussian_curvature | **−1.0** |
| rho_max | 12 |
| lr (AdamW) | **3e-4** |
| model | small-hyperbolic-dit (768 / 12 / 12), seq **256** |
| steps / batch | 30k / global 512 (1 GPU × PER_GPU_BS 32, accum 16), bf16, EMA 0.9999 |

Eval mirrors training geometry (`PRIOR_COV=1.0 RHO_MAX=12 GAUSS_CURV=-1.0`,
SEQ_LEN=256): valid PPL (`ppl_eval`) + GenPPL/entropy (`sample_eval`, exact
velocity, 180 steps, top_k_velocity=1, greedy last).

## Hypothesis

With the √d rescale, `prior_cov=1.0` H-FLM trains stably (no radial collapse) and
produces a usable generator (finite GenPPL, entropy in a healthy range, not the
low-GenPPL+low-entropy degenerate corner). This is a **correctness/plumbing
check** of the rescale, not a tuned baseline.

## GPU & wall clock

1 GPU on `thickstun,desa` (excl. desa-compute-01), 30k steps ≈ 6–12 h depending
on node; `--requeue`, auto-resumes from `last.ckpt`. Deliverables land in
`outputs/hflm_rescale_tinystories_256/<run>/` (train logs + `checkpoints/`) and
`.../eval/` (`ppl.json`, `samples_genppl.json`). RESULTS.md written after eval.

**Runs from this checkout** (`s-flm-dev/s-flm`, branch `var_scale`) — the rescale
lives here, not in the shared `/share/.../s-flm` tree the other sweeps use.
