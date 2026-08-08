# naive_ar_tinystories_s256 — Experiment Design

**Slides:** `slides/jun25_2026/slides.md` — "Naive AR Baseline", **Max Seq Len = 256**.
**Spec:** `setup.md` (methods / grid / eval protocol).

## Hypothesis
Standard non-geometry small-DiT baselines trained on TinyStories at seq-256 are the
reference points against which the geometry flows (S/E/H-FLM) and advanced tricks are
compared. Four methods, each swept over LR and seed:
- **`ar`**   — causal autoregressive. Valid PPL is a *true* AR PPL.
- **`mdlm`** — masked (absorbing) discrete diffusion. Valid PPL is a denoising-ELBO bound.
- **`duo`**  — uniform-state discrete diffusion. Valid PPL is a denoising-ELBO bound.
- **`flm`**  — base flow language model. Valid PPL is an unweighted denoising CE.

## Design
- **36 cells** = 4 methods × LR {3e-4, 1e-3, 5e-3} × seed {1, 2, 3}.
- Identical small DiT (width 768, depth 12, heads 12): `model=small` for ar/mdlm/duo,
  `model=small-flm` for flm (same 768/12/12).
- 30k steps, global batch 512, **seq 256**, bf16, EMA 0.9999, AdamW (wd 0,
  betas (0.9,0.999), eps 1e-8, grad-clip 1.0), constant schedule w/ 2500-step warmup.
- **Checkpoints every 5k steps, all retained** (`SAVE_TOPK=-1`).
- Eval: 180 sampling steps for mdlm/duo/flm (`setup.md`); greedy last step where the
  sampler exposes it (duo `noise_removal=greedy`; flm's Euler sampler always argmaxes the
  last step). AR ignores `sampler.steps` and does one forward per token.
- Only the **training** seed is swept — the sample scripts expose no SEED knob, so eval
  noise is common across seeds and the error bars isolate training-seed variance
  (same convention as `experiments/seed_errbar_tinystories_256`).

## Known comparability caveats
- **Valid PPL mixes three estimands** (exact NLL / ELBO bound / denoising CE) and must not
  be ranked in one column. Only `mdlm` ↔ `duo` are directly comparable.
- **AR is decoded greedily** (`scripts/sample/tinystories/ar.sh` defaults `GREEDY=true`), so
  its 64 samples collapse to one string and its GenPPL is a mode decode. Re-run that cell
  with `GREEDY=false` for a number rankable against the stochastic rows.
- **AR carries `algo.adaLN: False`** (162.2M params vs 169.6M for the other three), so
  "parameter-matched" is approximate.

## GPU allocation
- 1 job per cell, `gpu:4` on `thickstun,desa` (exclude desa-compute-01). `PER_GPU_BS=32`
  (accum = 512/(4×32) = 4). Train→eval in one SLURM job; idempotent/resumable.

## Expected wall-clock
- ~16.4 GPU-hr per cell on RTX 6000 Ada at bs=32 → ≈ 4–5 hr train + ~0.5 hr eval per cell
  on 4 GPUs. 36 cells ≈ **720 GPU-hr**; at ~7 concurrent 4-GPU jobs on a free cluster that
  is ~6 waves ≈ **1.5–2 days**, longer under contention.

## Outputs
`outputs/naive_ar_tinystories_s256/m-{method}_lr-{lr}_sd-{seed}/` → checkpoints/,
eval/ppl.json, eval/samples_genppl.json.
Report: `experiments/naive_ar_tinystories_s256/RESULTS.md` (via `report.py`).

**Legacy dirs:** `ar/` and `mdlm/` predate this naming and are the lr-3e-4 / seed-1 cells.
Rename them to `m-ar_lr-3e-4_sd-1` / `m-mdlm_lr-3e-4_sd-1` to reuse their checkpoints;
otherwise those two cells retrain from scratch. Their *eval* outputs are stale regardless
(mdlm was evaluated at 256 sampling steps, now 180).
