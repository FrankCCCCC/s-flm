# adv_geo_tinystories_s256 — Experiment Design

**Slides:** `slides/jun25_2026/slides.md` — "Adv Geometry Baseline", **Max Seq Len = 256**.

## Hypothesis
Do the "advanced" geometry recipes improve over the naive flows at seq-256? Two families,
each across an LR sweep:
- **S-FLM**: adaptive schedule, truncation, adaptive+truncation (init=ngpt).
- **LangFlow**: adaptive schedule, adaptive+self-conditioning (init=unit_var — N(0,I) VP
  prior is scale-matched to unit_var, not ngpt).

## Design
- 25 cells = 5 variants × 5 LRs {5e-5, 1e-4, 3e-4, 1e-3, 5e-3}.
  - `sfm_ada` (ALPHA_MAX=null), `sfm_trunc` (0.121), `sfm_ada_trunc` (0.121),
    `lf_ada` (self_cond off), `lf_ada_sc` (self_cond on).
- Small DiT 768/12/12, 30k steps, global batch 512, **seq 256**, bf16, EMA 0.9999,
  AdamW wd 0 / grad-clip 1.0, cross-entropy. Eval: exact-velocity, top_k_v=1, 180 steps.
- **Checkpoints every 5k steps, all retained** (`SAVE_TOPK=-1`).

## GPU allocation
- 25 jobs, `gpu:1` each, `cpu=8, mem=32G` on `thickstun,desa` (exclude desa-compute-01).
  `PER_GPU_BS=32` (accum = 16). Train→eval per job; idempotent/resumable.

## Expected wall-clock
- ~16.4 hr (RTX 6000 Ada) / ~30 hr (A5000) train + ~0.5–1 hr eval per run. With ~18
  GPUs the 25 cells finish in **~2–3 days**.

## Outputs
`outputs/adv_geo_tinystories_s256/{variant}_lr{lr}/`.
Report: `experiments/adv_geo_tinystories_s256/RESULTS.md`.
