# TinyStories 3-way pre-training (geometry vs engineering tricks)

Fair comparison of **S-FLM (Naive)**, **S-FLM (truncation + adaptive noise)**, and **HFLM**
on TinyStories, with OWT-matched hyperparameters. Launched 2026-06-09.

## Models (only the geometry / noise differs; everything else held fixed)

| Run | script | model | algo | noise | sampler (eval) |
|-----|--------|-------|------|-------|--------|
| S-FLM Naive | `scripts/train/tinystories/sfm.sh` | small-sphere-dit (init ngpt) | sfm | log-linear | sfm |
| S-FLM Trunc+Adaptive | `scripts/train/tinystories/sfm_truncated_adaptive.sh` | small-sphere-dit (init ngpt) | sfm | log-linear-adaptive, α_max=0.121, refit50, ema0.9, buf25600, umix1e-3 | sfm |
| HFLM | `scripts/train/tinystories/hlfm.sh` | small-hyperbolic-dit | hflm (prior_cov=0.25, rho_max=12) | log-linear | hflm |

**Held fixed (OWT-matched):** model size 768/12/12 (~169M), `global_batch_size=512`,
`batch_size=32`, AdamW lr 3e-4, EMA 0.9999, `algo.invert_time_convention=false`,
`renormalize_weights=False`, gpt2 tokenizer, block_size 1024.

**Training budget:** `max_steps=30_000` optimizer steps. TinyStories train ≈ 471.6M gpt2
tokens → ~900 opt-steps/epoch at bs512 ⇒ **~33 epochs** (well above the AR-typical 4–10;
~3 EMA horizons; ~33h at ~1 micro-batch/s × accum4 on 4 GPUs, fits the 48h walltime).
Checkpoints every 2_500 steps; validation OFF during training (see recipe).

## Launch recipe (this cluster — hard-won; see memory `multi-gpu-ddp-recipe`)

sbatch wrappers in `experiments/tinystories/{sfm,sfm_truncated_adaptive,hflm}.sub`:
- `export SLURM_JOB_NAME=bash` → Lightning native multi-GPU launcher (else SLURMEnvironment hangs).
- `export NCCL_P2P_DISABLE=1`, `export NCCL_IB_DISABLE=1`.
- `trainer.devices=4` — **devices=2 DDP hangs** at the first NCCL collective; 4 works.
- 48GB GPU (a6000 / 6000ada) — the model uses ~44GB/GPU; **a5000 (24GB) OOMs**.
- `+wandb.offline=true` — online wandb is unreliable here; sync afterward.
- `--cpus-per-task=16` so two 4-GPU jobs share one node.
- validation OFF (`limit_val_batches=0`, `num_sanity_val_steps=0`) — eval is done separately.

## W&B (project `syctw/tinystories-flm`, runs offline → synced)

| run | W&B run id | node |
|-----|-----------|------|
| tinystories_sfm_naive | `rpaukpol` | kuleshov-compute-02 (a6000) |
| tinystories_sfm_trunc_adaptive | `lduq3rsg` | kuleshov-compute-02 (a6000) |
| tinystories_hflm | `c7uhmd4o` | thickstun-compute-01 (6000ada) |

(Earlier ids in the project are cancelled debug attempts — ignore.)
Sync helper: `bash experiments/tinystories/sync_wandb.sh` (run periodically + once after training).

## Eval — two parts: held-out val/ppl + an identical-decode GenPPL sweep

`scripts/sample/tinystories/eval.sh` (single-GPU, forward-only) is parametrized by env vars:
`STEPS, NOISE_REMOVAL, TOPKV (top_k_velocity), NUM_SAMPLE_BATCHES, DO_PPL, DO_SAMPLE`.
With no overrides it runs `mode=ppl_eval` (→ `val/ppl` in W&B + `ppl.json`) then
`mode=sample_eval` (→ `gen_ppl_first_chunk_retok` + samples in `samples_genppl.json`).

**(A) Held-out val/ppl** (NFE-independent) — one `ppl_eval` per model at 30k:
```
sbatch --constraint=gpu-high --export=ALL,MODEL_TYPE=sfm,CKPT=<abs>/...-30000.ckpt,STEP_TAG=30k,DO_PPL=1,DO_SAMPLE=0 experiments/tinystories/eval.sub
```

**(B) Decode-policy sweep** (the fair GenPPL comparison) — `experiments/tinystories/sweep_array.sub`
(60-cell SLURM array, `%8`), manifest from `sweep_gen_manifest.sh`, collected by `sweep_collect.py`.
Within each cell ALL decode knobs are **identical across models** (steps, noise_removal,
top_k_velocity, + pinned velocity=exact/temperature=1.0/p_nucleus=1.0/top_k=-1); only the
geometry/predictor differs. Grid (30k checkpoint):

| axis | values |
|------|--------|
| steps (NFE) | 32, 64, 128, 256, 512, 1024 |
| noise_removal | ancestral, greedy |
| top_k_velocity | **1 for all 3** (aligned) + **−1 for sfm/sfm_adaptive** (sphere-native ref) |

**Key sampler facts (verified in `samplers.py`):**
- `top_k_velocity` is a **no-op for HFLM** (`HFLMSampler.step` always geodesic-steps toward the
  single `argmax` predicted-clean token). So `=1` is the *only* value alignable across both
  geometries; with `=1` the per-step update is identical up to geometry (sphere exp-map vs
  hyperbolic geodesic, same target token). The `=-1` sphere-native column guards against the
  aligned gap being misread as geometry when it's a sampler handicap.
- `noise_removal` only changes the **final** step (greedy argmax vs ancestral sample); with
  velocity=exact the trajectory is otherwise deterministic, so the greedy/ancestral delta at
  high steps may be within the 64-sample variance — read with care.

**Cluster gotchas (hard-won this run):**
- Sampling **OOMs a 24GB a5000** (float64 full-vocab `log_p` at bs16) → sweep pins
  `--constraint=gpu-high` (48GB a6000/6000ada). val/ppl is also routed to gpu-high.
- Running **many `ppl_eval` jobs at once on one a5000 froze them** (node thrash; `main.log`
  mtime stalled) → keep `ppl_eval` low-concurrency / on gpu-high.
- Manifest resolves explicit `*-30000.ckpt` (not rolling `last.ckpt`) so the `30k` label is
  honest; `sweep_gen_manifest.sh` aborts (atomic) if any 30k ckpt is missing.

After training, `final_eval_monitor.sh` auto-submits (A) then generates the manifest + submits (B).

## Eval results — GenPPL is the comparison; `val/ppl` is a diagnostic only

Final 3-way table + GenPPL-vs-NFE curves (with entropy beside each GenPPL as a degeneracy
guard) are produced by `sweep_collect.py` → `sweep_results.csv` / `sweep_genppl_vs_nfe.png`
once the array finishes.

⚠️ **`val/ppl` is NOT a perplexity for these models and CANNOT rank them.** SFM/HFLM
`nll_per_token` return the *unweighted* denoising cross-entropy (algo.py:261,394 `del dalpha_t`;
contrast MDLM algo.py:128 which keeps the `dalpha_t/(1-alpha_t)` ELBO weight that makes a true
token-NLL bound), averaged over each model's OWN noise schedule (trainer_base.py:551). Hence
sfm's val/ppl≈1.24 (0.22 nats/tok, below text entropy). It is comparable ONLY across
checkpoints of the *same* model (a convergence/overfitting signal) — never across geometries,
nor across the two sphere noise schedules (naive log-linear vs adaptive α_max=0.121). Report it
as "held-out denoising CE", and rank models by **GenPPL** (external gpt2-large judge on decoded
tokens — the only cross-geometry-fair metric).

**30k held-out denoising CE (`exp`, diagnostic only — do NOT rank with these):**

| model (30k) | val/ppl (denoising CE) |
|---|---|
| S-FLM naive | 1.241 |
| S-FLM trunc+adaptive | 8.888 |
| HFLM | 6.018 |

The 1.24 / 8.89 / 6.02 spread is itself the proof of non-comparability: the adaptive run is
scored under its OWN concentrated (α_max=0.121) schedule, so its CE is on a different basis —
not "9× worse." GenPPL (below) is the metric that actually ranks them.

### GenPPL — matched decode (top_k_v=1), gpt2-large, lower=better  [3-way, complete]

GenPPL (entropy) at noise_removal=ancestral (greedy is within ~1 pt, same shape):

| NFE | S-FLM naive | S-FLM **trunc+adaptive** | HFLM |
|----:|------------:|-------------------------:|-----:|
| 32   | 104.1 (3.4) | **9.26** (4.2) | 47.7 (4.8) |
| 64   | 81.5 (3.8)  | **8.91** (4.3) | 47.8 (4.9) |
| 128  | 61.9 (4.0)  | **8.71** (4.3) | 47.5 (4.9) |
| 256  | 52.4 (4.1)  | **8.63** (4.3) | 48.9 (4.9) |
| 512  | 47.4 (4.2)  | **8.54** (4.3) | 48.7 (4.9) |
| 1024 | 44.8 (4.2)  | **8.55** (4.3) | 49.7 (4.9) |

**Headline — on TinyStories generation, the noise-schedule TRICKS dominate the GEOMETRY:**
- **S-FLM + truncation + adaptive noise ≈ GenPPL 8.5** (flat across NFE) — ~5–6× better than both
  naive S-FLM (45–104) and HFLM (~48), i.e. near-AR fluency. Verified NOT degenerate: samples are
  coherent dialogue with consistent characters (entropy ~4.3, not collapsed); naive-S-FLM / HFLM
  samples are visibly disfluent ("the and and", topic drift).
- **Geometry alone (HFLM) buys sample-EFFICIENCY, not the quality ceiling:** HFLM is flat ~48 at
  *every* NFE (good at 32 steps); naive S-FLM needs many steps (104→45) and overtakes HFLM only
  ≈512 NFE. So HFLM wins low-NFE, naive-S-FLM wins high-NFE — but **both are far above the
  trick-equipped S-FLM at all NFE.**
- ⇒ For this task, engineering tricks ≫ geometry change; the natural next experiment is **HFLM +
  truncation/adaptive** (geometry *and* tricks).

Sphere-native (top_k_v=−1) reference: sfm/adaptive at −1 are more diverse (entropy ~5.1–5.5) but
need more steps; **6 of the heaviest −1 cells (sfm@1024, adaptive@512/1024) TIMED OUT** at the 4h
walltime (full-vocab velocity × 1024 steps) — re-run with longer walltime if the full −1 curve is
needed. The matched top_k_v=1 comparison above is complete for all 3 models.

Plot: `sweep_genppl_vs_nfe.png` · table: `sweep_results.csv`.

> NOTE: the earlier HFLM GenPPL numbers (10k 43.6 / 20k 41.1 / 30k 49.4) were at the
> **180-step greedy/top-1 native** sampler and are **NOT comparable** to S-FLM — superseded by
> the identical-decode sweep above. Smoke checks at steps=64 already gave hflm@ancestral 47.2
> and sfm@greedy+kv1 81.0 (`avg_nfe==steps` confirmed).

## Live training sanity (2026-06-09, ~step 400–500)
trainer/loss dropping from ~9.9 (≈ uniform-init ln 50257) — sfm 9.95→1.35, hflm 9.94→3.48,
adaptive 9.82→5.97 (adaptive is higher only because its noise spline is in its ~1000-step
warmup; expected to catch up). No NaN. All three healthy.
