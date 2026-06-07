# HFLM — Results (Sudoku-easy)

**Verdict: CONFIRMED.** A Hyperbolic Flow Language Model (HFLM) — wrapped-normal prior +
constant-speed hyperbolic geodesic posterior, embedding-length-as-radial, trained with the
*same* cross-entropy objective as S-FLM — trains stably at `d=512` and **outperforms** the
hyperspherical baseline on Sudoku-easy.

## Headline numbers (validation set, 2000 puzzles, 180 sampling steps)

| Model | Geometry | Decoding | Exact-match acc |
|---|---|---|---|
| **HFLM** | Hyperbolic `H^d` (Poincaré I/O) | top-1 geodesic step | **93.75%** (1875/2000) |
| S-FLM (baseline) | Sphere `S^{d-1}` | top-1 velocity | 79.05% (1581/2000) |
| S-FLM (baseline) | Sphere `S^{d-1}` | exact velocity (full vocab) | 78.45% (1569/2000) |

HFLM beats S-FLM by **+14.7 pts** under apples-to-apples top-1 decoding (and +15.3 vs exact
velocity), so the gap is **not** a decoding-asymmetry artifact.

## Success criteria (from EXPERIMENT.md)
- `A_HFLM ≥ A_SFLM − 5pts` → 93.75% ≥ 74.05% ✅
- `A_HFLM ≥ 60%` → 93.75% ✅
- No NaNs / divergence / ρ-bound crashes during training or sampling ✅
→ **CONFIRMED** (in fact HFLM *exceeds* S-FLM rather than merely matching).

## Setup
- **Data**: Sudoku-easy (40/81 givens), 48k train / 2k val, seed 42.
- **Backbone**: `tiny-hyperbolic-dit` vs `tiny-sphere-dit` — identical DiT body (8 layers,
  d=512); the only HFLM deltas are the essential manifold I/O (unnormalized flexible-length
  embeddings, no embedding renormalization, plain-logit output).
- **Training**: 20k steps, batch 256, `noise=log-linear`, `invert_time_convention=false`,
  single GPU (RTX 6000 Ada), ~100 min/run.
- **HFLM knobs**: `prior_cov=0.25`, `rho_max=12` (soft clamp `ρ_eff=rho_max·tanh(ρ/rho_max)`).
  Observed: no ρ>20 events at d=512 — the cov-scaling + clamp keep geodesic inputs in-bounds.
- **Sampling**: 180 steps; HFLM = top-1 predicted-clean geodesic step; S-FLM = paper velocity.
- Artifacts: `eval_runs/sudoku/{hflm_easy, sfm_easy, sfm_easy_top1}/results.json`;
  checkpoints under `outputs/sudoku/{hflm_easy,sfm_easy}/checkpoints/`; W&B (offline) under `./wandb/`.

## Caveats / honest scope
1. **Single seed, single budget.** One 20k-step run per model on the *tiny* config. The gap is
   large and consistent across decoders, but a 2–3 seed repeat would firm up the variance.
2. **HFLM "exact" = top-1.** The shipped HFLM sampler integrates by stepping the geodesic toward
   the top-1 (or sampled) predicted-clean token; the fully marginalized hyperbolic velocity
   (sum of `log_{z_t}(e_v)` over all V, RFM eq. 15) is **not** implemented (would need a
   vectorized hyperbolic log-map). S-FLM's "exact velocity" *is* the full marginalization — and
   it still loses to HFLM top-1, and to S-FLM top-1, consistent with the paper's finding that
   top-1 ≥ exact for this family.
3. **Why HFLM may help here**: the radial degree of freedom (embedding length) gives the flow an
   extra, semantically-meaningful axis the sphere lacks; Sudoku's hard constraints seem to
   benefit. This is a hypothesis, not established — worth probing (e.g. ablate `prior_cov`,
   inspect learned ρ distribution by token).

## Suggested next steps
- Repeat HFLM & S-FLM at 2–3 seeds; report mean±std.
- Sudoku medium/hard (35/30 givens) to see if the hyperbolic advantage widens or narrows.
- Implement the true marginalized hyperbolic velocity and compare to top-1.
- Small `prior_cov`/`rho_max` sweep; log per-token learned radius to understand what the radial axis encodes.
