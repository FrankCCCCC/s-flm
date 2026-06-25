# naive_ar_tinystories_s256 — Experiment Design

**Slides:** `slides/jun25_2026/slides.md` — "Naive AR Baseline", **Max Seq Len = 256**.

## Hypothesis
A standard causal AR small-DiT trained on TinyStories at seq-256 is the reference
point against which the geometry flows (S/E/H-FLM) and advanced tricks are compared
(valid PPL is a true AR PPL here; for the flow models it is a denoising-CE bound).

## Design
- 1 run. Causal small DiT (width 768, depth 12, heads 12), `model=small`, `algo=ar`.
- 30k steps, global batch 512, **seq 256**, bf16, EMA 0.9999, AdamW (lr 3e-4, wd 0,
  betas (0.9,0.999), eps 1e-8, grad-clip 1.0). Greedy decoding eval.
- **Checkpoints every 5k steps, all retained** (`SAVE_TOPK=-1`).

## GPU allocation
- 1 job, `gpu:4` on `thickstun,desa` (exclude desa-compute-01). `PER_GPU_BS=32`
  (accum = 512/(4×32) = 4). Train→eval in one SLURM job; idempotent/resumable.

## Expected wall-clock
- ~16.4 GPU-hr of compute on RTX 6000 Ada at bs=32; with 4 GPUs ≈ **4–5 hr** train
  + ~0.5 hr eval. Delivered in **< 1 day**.

## Outputs
`outputs/naive_ar_tinystories_s256/ar/` → checkpoints/, eval/ppl.json, eval/samples_genppl.json.
Report: `experiments/naive_ar_tinystories_s256/RESULTS.md` (via `report.py`).
