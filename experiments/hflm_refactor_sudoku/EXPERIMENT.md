# hflm_refactor_sudoku — validate the VFM expected-velocity sampler refactor

> **Incident (2026-07-11 ~02:40, resolved ~03:10).** The first 9 completed cells
> showed K=−1.0 at ~0% solve rate (all-blank boards) with K=−0.3/−0.5 only mildly
> degraded. Root cause: `HFLMSampler._lorentz_vocab_embeddings` lifted the vocab
> table polar→Lorentz in float32 and cast to float64 *after* — at trained
> embedding norms (raw ρ up to ~18, clamped ~11) the f32 lift puts endpoints
> O(100) off the hyperboloid (⟨e,e⟩_L ≠ −R²), corrupting every log/exp map;
> severity scales with cosh²(ρ/R), hence K=−1 dead / mild-K degraded. Fixed by
> lifting in the slerp dtype AND switching to a factored, cancellation-free
> velocity/step (`hflm_compute_velocity` returns (w, s) with v = w − s·x;
> `hflm_exp_step` uses ‖v‖² = ⟨w,w⟩_L + R²s² exactly). Regression test:
> `tests/test_hflm_velocity.py::test_trained_norm_f32_table_step_matches_geodesic`.
> All eval outputs produced before the fix were quarantined as
> `eval_*_bad0711/` and the affected cells resubmitted; training was never
> affected (losses 0.002–0.03).

Branch `hflm_refactor` replaced `HFLMSampler.step`'s argmax-endpoint geodesic step
with the true VFM marginal velocity `v = Σ_v p_v·log_x(e_v)` (Lorentz log map,
posterior expectation, `exp_x(dt·v)`), and made `top_k_velocity` functional
(truncate + renormalize; it was a dead knob — every old eval ran argmax regardless).
**Training is untouched by the refactor.** `velocity='exact', top_k_velocity=1`
reproduces the old argmax step exactly (pinned by `tests/test_hflm_velocity.py`).

## Hypotheses

1. **No regression**: retrained with the refactored code at the old sweep's best
   hyperparameters, the `top_k_velocity=1` eval (sampler-identical to the old code)
   matches the old sweep's numbers within seed noise (pooled across-seed 2·SE ≈ 9pt
   medium / 11pt hard; per-cell seed std 5.5 / 6.6pt).
2. **Sampler effect**: the corrected expected-velocity eval (`top_k_velocity=-1`)
   on the *same checkpoint* is ≥ or ≈ the argmax eval (for Sudoku's near-deterministic
   posteriors the two should converge; a large drop would indicate a bug).
3. **Curvature**: mild curvature (K=−0.3/−0.5) ≥ K=−1.0, replicating the old sweep's
   aggregate finding, now on easy as well.

## Run matrix

**Expanded 2026-07-11 ~00:40 (user request: as many hyperparameters as possible
overnight): 66 cells total, seed 1, 20k steps, dual eval each.**

1. **Curvature × LR core** (init c0.01): K {−0.3, −0.5, −0.7, −1.0} × LR {3e-4,
   5e-4, 1e-3} × {easy, medium, hard} = 36 cells (includes the initial 9).
2. **K=−1.5 anchor** @ 3e-4, c0.01 × 3 difficulties = 3 cells (replicates
   "strong curvature hurts").
3. **Init axis** at per-difficulty best LR (easy 3e-4 / medium 5e-4 / hard 3e-4):
   init {c0.02, c0.04, random} × K {−0.3, −0.5, −1.0} × 3 difficulties = 27 cells.

Submission is priority-ordered (medium/hard curvature core → easy → anchors →
init axis) so the highest-value comparisons finish first if GPUs run out before
10 AM; the sweep is idempotent, stragglers just finish later.

### Initial 9 cells (old-sweep best cells; the no-regression comparison set)

| difficulty | K | LR | old seed-1 acc | old cell mean ± std (n=3) |
|---|---|---|---|---|
| easy   | −0.3 | 3e-4 | — (not swept) | — |
| easy   | −0.5 | 3e-4 | — | — |
| easy   | −1.0 | 3e-4 | — | — |
| medium | −0.3 | 5e-4 | 87.20% | 83.23 ± 5.46 |
| medium | −0.5 | 3e-4 | 82.50% | 81.07 ± 6.10 |
| medium | −1.0 | 5e-4 | 80.20% | 80.88 ± 0.63 |
| hard   | −0.3 | 3e-4 | 31.00% | 42.07 ± 9.92 |
| hard   | −0.5 | 3e-4 | 32.15% | 46.22 ± 13.13 |
| hard   | −1.0 | 3e-4 | 35.70% | 40.37 ± 5.28 |

LR/init picked per (difficulty, K) from `experiments/hflm_curv_init_lr_sudoku/`
(c0.01 best cells; easy reuses hard's 3e-4 since easy was not swept; S-FLM paper
easy ≈ 81.5% as loose context). Old numbers from `all_results.csv`.

**Fixed recipe** (mirrors the old sweep): tiny-hyperbolic-dit (512/8/8), batch 256,
seq 180, `noise=log-linear`, `invert_time_convention=false`, `prior_cov=0.25`,
`rho_max=12`, 20k steps, ckpt every 5k, data_cache seed-42 Sudoku 48k/2k.

**Eval** (per checkpoint, 2000 held-out puzzles, 180 steps, greedy final decode):
- `eval_topk1/`  — `VELOCITY=exact TOPK_VELOCITY=1`  (≡ old argmax sampler → hypothesis 1)
- `eval_topkall/` — `VELOCITY=exact TOPK_VELOCITY=-1` (true expected velocity → hypothesis 2)

## Success criteria

1. Each medium/hard `eval_topk1` accuracy within 2·seed-std of its old cell mean
   (equivalently ≥ old seed-1 − ~2pt run-to-run noise). Failing this on multiple
   cells ⇒ investigate the refactor (or training env drift) before any conclusion.
2. `eval_topkall` within a few points of `eval_topk1` (expected ≈ on Sudoku); report
   the signed delta per cell — this is the headline number for the sampler fix.
3. Curvature ordering reported per difficulty; single-seed, so only deltas beyond
   ~2·seed-std are called signal (mirror the old sweep's caution: aggregate-only effect).

## Compute

Unicorn `sc3379`, partition `thickstun,desa` (exclude desa-compute-01), 1 GPU/cell,
4 CPU, 16G, 6h limit. Train ≈ 2.7 h + 2 evals ≈ 40 min + torch.compile warm-up
⇒ ≈ 3.5 h/cell; 9 cells ≈ 32 GPU-h, ~4 h wall if all schedule at once.
Outputs: `outputs/hflm_refactor_sudoku/d-{diff}_k{K}_i-c0.01_lr{lr}_rs1/`
(checkpoints removed after both evals complete). Logs: `experiments/hflm_refactor_sudoku/logs/`.
Results table → `RESULTS.md` (analysis phase).
