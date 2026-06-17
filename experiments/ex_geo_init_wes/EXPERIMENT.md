# EXPERIMENT: ex_geo_init_wes — geometry × init × word-embedding-dim

**Owner:** sc3379@cornell.edu · **Date:** 2026-06-14 · **Branch:** `claude/langflow`
**One-line:** Sweep the three flow geometries against embedding **init** and **word-embedding dimension**
to see how each geometry's Sudoku accuracy depends on the embedding parameterization.

## Grid (36 baseline + 180 LR = 216 cells)

| Axis | Values |
|---|---|
| geometry | `sfm` (sphere), `eflm` (Euclidean), `hflm` (hyperbolic) |
| `model.init` | `ngpt` (std 1/√d), `random` (std 0.02), `unit_var` (std 1), `hyperbolic` (std 0.3 — hflm's native init) |
| word-embedding dim (`model.hidden_size`) | 512, 256, 128 |
| `optim.lr` | **3e-4** (baseline = the 36-cell grid, un-lr-tagged) **+ {1e-4, 5e-4, 1e-3, 8e-5, 5e-5}** (180 lr-tagged cells) |

The LR axis cells are tagged `..._lr{lr}_...`; lr=3e-4 is the config default and is exactly the
geo×init×dim baseline grid (no `_lr` tag), so the two don't collide and `sweep.py` stays idempotent.

Held fixed: difficulty (default **medium** — most discriminative; flip `DIFFICULTIES` in `sweep.py`
to `['easy','medium','hard']` for the full 81-run grid). Each cell = train + sudoku_eval.

## Identical recipe (paper Table 1)

`tiny` DiT — **depth 8, heads 8** (width = swept `hidden_size`) · **20k** steps · effective batch
**256** (per-device 64 × grad-accum 4, so it fits desa's 11GB 2080ti; identical to batch 256 — no
batchnorm) ·
seq-len 180 · bf16 · EMA **0.9999** · AdamW lr **3e-4**, wd 0, betas (0.9, 0.999), eps 1e-8,
grad-clip **1.0** · cross-entropy loss · `noise=log-linear`, `invert_time_convention=false` ·
Sudoku 48k/2k (seed 42). Eval: `mode=sudoku_eval`, **180** steps, exact velocity, greedy,
`top_k_velocity=-1`, EMA on, full 2k puzzles. Per-geometry: `hflm` adds `prior_cov=0.25, rho_max=12`.

## Why these axes

- **sfm** sphere-normalizes embeddings → init's *scale* is erased; only direction survives. Expect
  init to matter least.
- **eflm** uses raw embeddings → init scale directly sets the data/noise SNR. Expect strong init
  dependence.
- **hflm** uses ‖e_v‖ as the radial (hyperbolic) coordinate → init sets the radius (clamped at
  `rho_max=12`). Note the grid does **not** include hflm's native `hyperbolic` init (std 0.3), so this
  also probes hflm's robustness to off-design inits.
- **dim** {512,256,128}: does a smaller embedding/model close or widen the geometry gaps?

## Run

```bash
python experiments/ex_geo_init_wes/sweep.py --dry-run   # inspect grid + one command
python experiments/ex_geo_init_wes/sweep.py             # submit 27 jobs (thickstun,desa)
```
Job→tag manifest: `experiments/ex_geo_init_wes/jobs.txt`. Logs: `experiments/ex_geo_init_wes/logs/`.
Checkpoints: `outputs/sudoku/exgiw/<tag>/`. Evals: `eval_runs/sudoku/exgiw/<tag>/results.json`
(`<tag> = {geo}_{init}_d{dim}_{difficulty}`).

## Code note

`models/hyperbolic_dit.py` gained an `init=unit_var` branch (`std=1.0`, mirroring `sphere_dit`) so the
3 `hflm × unit_var` cells run (it previously supported only `{random, ngpt, hyperbolic, pretrained}`).

## Results

Filled into `RESULTS.md` once evals land — a 3×3 accuracy grid per geometry, plus the
"which init/dim each geometry prefers" reading.
