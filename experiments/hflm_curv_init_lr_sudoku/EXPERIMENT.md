# hflm_curv_init_lr_sudoku — Curvature × Init × LR search (Sudoku medium)

**Question:** Which (Gaussian curvature K, embedding init, learning rate) combination is
best for H-FLM on Sudoku, and how do the three knobs interact? Follow-up to
`hflm_curv_sudoku`, which found an interior curvature optimum (K=-0.5 ≫ K=-1 on
medium/hard) at fixed init=hyperbolic, lr=3e-4 — this refines the K grid around -0.5
and crosses it with init and LR (spec: `slides/jul02_2026/slides.md`).

## Grid (168 cells)

| axis | values | knob |
|---|---|---|
| curvature K | -0.25, -0.3, -0.5, -0.7, -1.0, -1.5 | `GAUSS_CURV` → `algo.gaussian_curvature` |
| init | `ngpt` (N(0,1/d)), `random` (N(0,4e-4)), `custom` std ∈ {0.01, 0.02, 0.04, 0.06, 0.08} | `INIT`/`INIT_STD` → `model.init`/`model.init_std` |
| LR | 1e-4, 3e-4, 5e-4, 1e-3 | `LR` → `optim.lr` |

Note `random` ≡ `custom std=0.02` and `ngpt` ≈ std 0.0442 at d=512 — the named baselines
double as consistency checks on the custom scan.

**Fixed:** difficulty=**medium** (48k/2k, seed 42, 35 clues — chosen because
`hflm_curv_sudoku` showed easy saturates ~90% and hard is low/noisy, while medium
separates best), tiny-hyperbolic-dit (512 wide / 8 deep / 8 heads, ~28.6M), seq 180,
20k steps, batch 256, bf16, EMA 0.9999, AdamW (wd 0, betas 0.9/0.999, eps 1e-8),
grad clip 1.0, prior_cov 0.25, rho_max 12, noise=log-linear.

**Eval:** `sudoku_eval`, 180 steps, exact velocity, greedy last step,
`top_k_velocity=-1` (velocity averaged across the full vocab — differs from
`hflm_curv_sudoku`'s top-1, so accuracies are not directly comparable across the
two experiments).

All K in the grid satisfy the float64 Lorentz bound ρ/R ≤ 20 (worst case K=-1.5:
12·√1.5 ≈ 14.7).

## Priority

`init=random × lr ∈ {3e-4, 1e-3} × all 6 K` (12 cells) run first: submitted ahead of the
rest and with `sbatch --nice=0` vs `--nice=100` for the remainder. Re-tier a queued job
anytime with `scontrol update jobid=<id> nice=<n>`.

## GPU allocation (2 sites, static split on the K axis)

| site | K values | cells | resources |
|---|---|---|---|
| unicorn (sc3379) | -0.25, -0.5, -1.0 | 84 | `thickstun,desa` (excl. desa-compute-01), 1 GPU, 8 CPU, 32G, 6h |
| TinkerCliffs (shengyenc) | -0.3, -0.7, -1.5 | 84 | `a100_normal_q`, 1 GPU, 8 CPU, 64G, 6h |

~2.5–3h train + ~10 min eval per cell (1 GPU) ⇒ ~450 GPU·h total; wall clock depends on
queue share (e.g. ~10 concurrent GPUs/site ⇒ ~1–1.5 days).

## Launch

```bash
python experiments/hflm_curv_init_lr_sudoku/sweep.py --site unicorn   # on sc3379@unicorn
python experiments/hflm_curv_init_lr_sudoku/sweep.py --site tc       # on shengyenc@tinkercliffs
```

Orchestration-only: each cell calls `scripts/train/sudoku/hflm.sh` then
`scripts/sample/sudoku/hflm.sh`. Idempotent/resumable — skips cells with
`eval/results.json` or an in-queue job name (`hcil_k<K>_i-<init>_lr<lr>`); resubmitting
auto-resumes from `last.ckpt`.

## Outputs

- per cell: `outputs/hflm_curv_init_lr_sudoku/k<K>_i-<init>_lr<lr>/` (checkpoints + `eval/results.json`)
- logs: `experiments/hflm_curv_init_lr_sudoku/logs/<tag>_<jobid>.log`
- TinkerCliffs cells live on TC storage and are rsynced back for RESULTS.md.
