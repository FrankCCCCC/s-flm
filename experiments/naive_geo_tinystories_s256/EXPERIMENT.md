# naive_geo_tinystories_s256 — Experiment Design

**Slides:** `slides/jun25_2026/slides.md` — "Naive Geometry Baseline", **Max Seq Len = 256**.

## Hypothesis
With *no* schedule/sampling tricks (init=ngpt, noise=log-linear), how do the three
geometric flows compare on TinyStories at seq-256? S-FLM (sphere), E-FLM (Euclidean),
H-FLM (hyperbolic). H-FLM is run twice: init=ngpt (slide spec) and init=hyperbolic
(prior-matched radius — the config that beats S-FLM on Sudoku; ngpt collapses the
radial coord, see [[hflm-ngpt-init-mismatch]]).

## Design
- 4 runs: `sfm`, `eflm`, `hflm` (ngpt), `hflm_hyperbolic`. Small DiT 768/12/12 on the
  sphere/hyperbolic backbone. 30k steps, global batch 512, **seq 256**, bf16, EMA 0.9999,
  AdamW lr 3e-4, cross-entropy loss.
- Eval: exact-velocity, top_k_v=1, 180 sampling steps, greedy last step.
- **Checkpoints every 5k steps, all retained** (`SAVE_TOPK=-1`).

## GPU allocation
- 4 jobs, `gpu:4` each on `thickstun,desa` (exclude desa-compute-01). `PER_GPU_BS=32`
  (accum = 512/(4×32) = 4). Train→eval per job; idempotent/resumable.

## Expected wall-clock
- ~4–5 hr train + ~0.5–1 hr eval per run on 4× RTX 6000 Ada. With ~18 GPUs the 4 runs
  overlap → **< 1 day**.

## Outputs
`outputs/naive_geo_tinystories_s256/{sfm,eflm,hflm,hflm_hyperbolic}/`.
Report: `experiments/naive_geo_tinystories_s256/RESULTS.md`.
