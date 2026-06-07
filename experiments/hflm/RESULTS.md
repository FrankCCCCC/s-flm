# HFLM — Results (Sudoku: easy / medium / hard)

**Verdict: CONFIRMED.** A Hyperbolic Flow Language Model (HFLM) — origin wrapped-normal prior +
constant-speed hyperbolic geodesic posterior, embedding-length-as-radial, trained with the *same*
cross-entropy objective as S-FLM — trains stably at `d=512` and, as a **drop-in geometry swap with
no schedule tricks, beats the equivalently-naive S-FLM at every Sudoku difficulty.**

All numbers: tiny DiT backbone (8 layers, d=512), 48k/2k train/val, 20k steps, batch 256,
180 sampling steps, single A6000/RTX-6000-Ada GPU. HFLM sampler = top-1 predicted-clean geodesic step.

## Headline: naive HFLM vs naive S-FLM across difficulty

| Sudoku | naive S-FLM (paper Table 1) | naive S-FLM (ours, easy only) | **naive HFLM (ours)** | Δ vs naive S-FLM |
|---|---|---|---|---|
| Easy   | 81.5 | 78.45 (exact) / 79.05 (top-1) | **93.75%** (1875/2000) | **+12.3** |
| Medium | 50.6 | — | **67.40%** (1348/2000) | **+16.8** |
| Hard   | 14.0 | — | **24.10%** ( 482/2000) | **+10.1** |

Naive HFLM wins at all three difficulties by 10–17 pts. (Our naive S-FLM easy reproduces the
paper's 81.5 to within ~2–3 pts, validating the harness.)

## Full context vs all baselines (paper Table 1; both DiT, 20k steps, 180-step sampling)

| Model | Easy | Med | Hard |
|---|---|---|---|
| AR (greedy) | 14.6 | 5.1 | 1.0 |
| MDLM | 92.0 | 77.1 | 30.2 |
| Duo | 96.3 | 84.7 | 58.4 |
| CANDI | 79.3 | 45.9 | 16.7 |
| FLM (one-hot) | 94.2 | 82.7 | 44.5 |
| S-FLM (naive) | 81.5 | 50.6 | 14.0 |
| S-FLM + α⋆(0.1) trunc | 94.0 | 77.6 | 43.2 |
| S-FLM + α⋆(0.1) + adaptive | 94.8 | 85.2 | 45.0 |
| **HFLM (naive, ours)** | **93.75** | **67.40** | **24.10** |

Naive HFLM (no tricks) is competitive with **tuned** S-FLM on easy and clearly above naive S-FLM
everywhere, but trails tuned S-FLM / FLM on medium & hard — where the noise-schedule tricks matter
most and HFLM doesn't yet have hyperbolic-appropriate versions of them (see below).

## Fairness study (Sudoku-easy, in-house, matched 20k/256 budget)

Prompted by the question "is naive-vs-naive a fair comparison?" — the paper's headline S-FLM (94.8%)
uses truncation + adaptive schedule. So we ran best-vs-best:

| Config (easy) | Acc |
|---|---|
| naive S-FLM (exact / top-1) | 78.45 / 79.05 |
| **naive HFLM** (top-1) | **93.75** |
| **S-FLM + trunc + adaptive** | **95.20** (reproduces paper's 94.8 ✓) |
| HFLM + trunc + adaptive (sphere α⋆=0.093), truncated sampling | **12.25** (collapse) |
| HFLM + trunc + adaptive checkpoint, full-range sampling | **0.00** (collapse) |

**Key finding — sphere schedule tricks do NOT transfer to hyperbolic.** Naively applying the
sphere-derived truncation bound α⋆=0.093 (paper eq. 17, a function of |V|, d for the **sphere**)
collapses HFLM (93.75 → 12.25). The full-range-sampling diagnostic gives 0.00%, proving it's the
truncated **training** (the model only ever saw the tiny window [0, 0.093] — appropriate for the
sphere, wrong for H^d) that breaks it, not just a sampling-config mismatch.

Honest best-vs-best read: **tuned S-FLM (95.2) slightly edges naive HFLM (93.75) on easy**, but the
hyperbolic geometry delivers, with *zero* tuning, ~what the sphere needs two tricks to reach. HFLM's
own truncation/adaptive analysis remains to be derived.

## Success criteria (EXPERIMENT.md): CONFIRMED
- `A_HFLM ≥ A_SFLM(naive) − 5pts` and `≥ 60%` on easy → 93.75% ✅ (exceeds naive S-FLM).
- Trains/samples with no NaNs, no ρ-bound crashes at d=512 (cov-scaling + tanh clamp hold) ✅.

## Setup notes
- HFLM: `prior_cov=0.25`, `rho_max=12`, soft clamp `ρ_eff=rho_max·tanh(ρ/rho_max)`; Poincaré-ball
  cartesian network I/O; unnormalized flexible-length embeddings; CE loss; gradient through the
  geodesic clean endpoint.
- Artifacts: `eval_runs/sudoku/{hflm_easy,hflm_medium,hflm_hard, sfm_easy, sfm_easy_top1,
  hflm_ta_easy, sfm_ta_easy, hflm_ta_easy_fullsample}/results.json`; checkpoints under
  `outputs/sudoku/hflm_{easy,medium,hard}/checkpoints/`.
- Infra: runs use `strategy=single-device` + `SLURM_JOB_NAME=bash` + `NCCL_P2P_DISABLE=1` to avoid
  DDP/NCCL on single GPU; pin to modern GPUs (RTX 6000 Ada / A6000) — TITAN X (sm_52) nodes are
  incompatible with this PyTorch build, and the 2080 Ti (11 GB) OOMs at batch 256.

## Hyperparameters (identical across easy / medium / hard — only `data.difficulty` differs)

| Group | Hyperparameter | Value |
|---|---|---|
| Model | backbone | `tiny-hyperbolic-dit` (`HyperbolicDiT`, standard DiT body) |
| | hidden size `d` | 512 |
| | blocks / heads | 8 / 8 |
| | dropout | 0.1 |
| | adaLN | true |
| | seq length | 180 |
| | embedding init | `hyperbolic` (std 0.3, **unnormalized**, flexible length) |
| | model eps | 1e-6 |
| Algo (HFLM) | loss | cross-entropy |
| | prior_cov | 0.25 |
| | rho_max (soft radial clamp) | 12 |
| | renormalize_weights | false |
| | invert_time_convention | false |
| Noise | schedule | log-linear (eps 1e-3) |
| | truncation (alpha_max) | none |
| | adaptive | false |
| Optimizer | type | AdamW |
| | learning rate | 3e-4 |
| | weight_decay | 0 |
| | betas | (0.9, 0.999) |
| | eps | 1e-8 |
| | gradient clip | 1.0 |
| | LR schedule | constant + warmup (`get_constant_schedule_with_warmup`) |
| | warmup steps | 2500 |
| | max steps | 20000 |
| | EMA decay | 0.9999 |
| Batch | global / per-GPU | 256 / 256 |
| | grad accumulation | 1 |
| | devices / nodes | 1 / 1 (effective batch = 256) |
| Precision | training | bf16 |
| | geodesic / slerp | float64 |
| Data | dataset | Sudoku, 48k train / 2k val, seed 42 |
| | **difficulty** | **easy (40) / medium (35) / hard (30) givens — THE ONLY VARIABLE** |
| Sampling | steps | 180 |
| | sampler / velocity | `hflm`, exact → top-1 geodesic step (`top_k_velocity=1`) |
| | noise_removal / temperature | greedy / 1.0 |

## Compute / environment

| Item | Value |
|---|---|
| Scheduler | SLURM (Cornell cluster); partitions with user priority: `thickstun`, `desa` |
| Per training job | 1× GPU (`--gres=gpu:1`), 8 CPU cores (`--cpus-per-task=8`), 96 GB host RAM (`--mem=96G`), 1 node |
| Per eval job | 1× GPU, 8 CPU cores, 64 GB host RAM |
| GPU — easy | NVIDIA RTX 6000 Ada (48 GB) — `thickstun-compute-01` |
| GPU — medium | NVIDIA RTX A6000 (48 GB) — `bala-compute-02` |
| GPU — hard | NVIDIA RTX A6000 (48 GB) — `kuleshov-compute-02` |
| GPU VRAM needed | >11 GB at batch 256 (OOMs on 11 GB 2080 Ti); fits ≥24 GB; run on 48 GB cards |
| Software | Python 3.12.13, PyTorch 2.7.0+cu128 (CUDA 12.8), Lightning 2.5.1, NumPy 1.26.4 |
| HW compatibility | cu128 build requires compute capability ≥ sm_75 (GTX TITAN X sm_52 is incompatible) |
| Wall-clock (20k steps, 1 GPU) | easy ≈100 min (≈3.3 it/s, 6000 Ada); medium ≈135 min (≈2.5 it/s, A6000); hard ≈155 min (≈2.2 it/s, A6000) |
| One-time Sudoku data-gen | easy ≈2 min, medium ≈5 min, hard ≈25–30 min (32 workers); eval ≈5–15 min each |

Because all three ran `devices=1` with `batch_size=256` and `accumulate_grad_batches=1`, the effective
optimization batch is **256 in every run** — the differing GPU *model* (6000 Ada vs A6000) does not
change any hyperparameter; it only affects throughput and low-level bf16 kernel rounding (sub-noise;
the geodesic path is float64).

## Reproduction (exact commands)

All three difficulties use **identical** hyperparameters (batch 256, lr 3e-4, wd 0, Adam (0.9,0.999),
EMA 0.9999, 2500 warmup, 20k steps, `tiny-hyperbolic-dit`, `prior_cov=0.25`, `rho_max=12`,
`noise=log-linear`, CE loss); only `data.difficulty` changes. The non-default flags
(`strategy=single-device`, `--job-name=bash`, `NCCL_P2P_DISABLE`) are cluster infra to avoid a
DDP/NCCL crash on a single GPU — pin to a modern GPU (RTX 6000 Ada / A6000; avoid TITAN X sm_52 and
11 GB 2080 Ti). The model/algo config also lives in `scripts/{train,sample}/sudoku/hflm.sh`.

```bash
cd /share/thickstun/sychou/workspace/research/s-flm-dev/s-flm
REPO=$PWD

# ── Train naive HFLM on easy / medium / hard (identical HPs; only difficulty differs) ──
for DIFF in easy medium hard; do
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  srun --partition=thickstun --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=08:00:00 --job-name=bash \
    python -u -m main \
      data=sudoku data.cache_dir="$REPO/data_cache" data.difficulty=$DIFF data.sudoku_num_workers=8 \
      model=tiny-hyperbolic-dit algo=hflm algo.invert_time_convention=false \
      algo.prior_cov=0.25 algo.rho_max=12 sampler=hflm noise=log-linear strategy=single-device \
      loader.global_batch_size=256 loader.batch_size=256 loader.eval_batch_size=256 loader.num_workers=8 \
      eval.generate_samples=False trainer.num_nodes=1 trainer.devices=1 \
      trainer.val_check_interval=20000 trainer.limit_val_batches=0 trainer.max_steps=20000 \
      callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
      wandb.project=hflm-sudoku +wandb.offline=true \
      hydra.run.dir="$REPO/outputs/sudoku/hflm_$DIFF"
done

# ── Eval (180-step top-1 geodesic sampling) → prints "Sudoku accuracy: N/2000 (X%)" ──
#    and writes eval_runs/sudoku/hflm_<DIFF>/results.json
for DIFF in easy medium hard; do
  CKPT_PATH="$REPO/outputs/sudoku/hflm_$DIFF/checkpoints/last.ckpt" \
  CACHE_DIR="$REPO/data_cache" DIFFICULTY=$DIFF \
  OUTPUT_DIR="$REPO/eval_runs/sudoku/hflm_$DIFF" \
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  srun --partition=thickstun --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --job-name=bash \
    bash scripts/sample/sudoku/hflm.sh
done
```

Notes:
- As actually run, easy was on `thickstun-compute-01` (RTX 6000 Ada) and medium/hard on A6000
  (`bala-compute-02`, `kuleshov-compute-02`); hard used a separate `data.cache_dir=$REPO/data_cache_hard`
  only to avoid a write race with a concurrent (doomed) job — `data_cache` works for a clean serial run.
- `srun … python -m main` directly (not `sbatch`) plus `--job-name=bash` keeps Lightning from
  initialising NCCL for a single device.

## Caveats / honest scope
1. **Single seed per cell, tiny config.** Gaps are large/consistent but a 2–3 seed repeat would firm variance.
2. **HFLM sampler is top-1 only** (no full marginalized hyperbolic velocity yet).
3. **Naive vs naive is the clean geometry comparison;** naive HFLM is *not* claimed to beat tuned S-FLM
   (it doesn't, on medium/hard). The fair best-vs-best needs hyperbolic-appropriate truncation — open work.

## Suggested next steps
- Derive a **hyperbolic** truncation bound (the H^d analogue of eq. 17) and re-run HFLM + trunc(+adaptive)
  at all difficulties — the most likely path to close the medium/hard gap to tuned S-FLM / FLM.
- 2–3 seeds; implement the true marginalized hyperbolic velocity; `prior_cov`/`rho_max` sweep.
