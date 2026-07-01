# hflm_sweep_tinystories_s256 — Experiment Design

**Slides:** `slides/jun25_2026/slides.md` — "H-FLM Sweep", **Max Seq Len = 256**.

## Hypothesis
H-FLM's quality hinges on matching the word-embedding scale to the hyperbolic prior
radius. Sweep embedding-init × diffusion init-noise (prior_cov) to find the regime where
H-FLM works at seq-256 (cf. [[hflm-ngpt-init-mismatch]]: ngpt collapses the radial coord;
a prior-matched init is needed).

## Design
- 132 cells = 12 inits × 11 prior_covs:
  - init: `ngpt` + custom std {0.001,0.01,0.02,0.04,0.1,0.3,0.5,0.8,1.0,1.5,2.0} (12)
  - prior_cov: {0.001,0.01,0.02,0.04,0.1,0.3,0.5,0.8,1.0,1.5,2.0} (11)
  - rho_max = 12.
- `small-hyperbolic-dit` 768/12/12, `algo=hflm`, noise=log-linear, 30k steps,
  global batch 512, **seq 256**, bf16, EMA 0.9999, AdamW lr 3e-4, cross-entropy.
- Eval: exact-velocity, top_k_v=1, 180 steps, greedy last.
- **Checkpoints every 5k steps, all retained** (`SAVE_TOPK=-1`).

## GPU allocation
- 132 jobs, `gpu:1` each, `cpu=8, mem=32G` on `thickstun,desa` (exclude desa-compute-01).
  `PER_GPU_BS=32` (accum = 16). Train→eval per job; idempotent/resumable. This is the
  dominant-cost experiment; submitted last so headline baselines land first.

## Expected wall-clock
- ~16.4 hr (RTX 6000 Ada) / ~30 hr (A5000) train + ~0.5–1 hr eval per run.
- ~2,640 GPU-hr total → the long tail: **~7–11 days** at 10–16 sustained concurrent GPUs.

## Outputs
`outputs/hflm_sweep_tinystories_s256/{ngpt|std{std}}_pc{pc}/`.
Report: `experiments/hflm_sweep_tinystories_s256/RESULTS.md`.
