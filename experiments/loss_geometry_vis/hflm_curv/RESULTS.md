# HFLM curvature loss-geometry — checkpoints & hyperparameters

Documents the exact checkpoint path and hyperparameters behind each figure in
this directory. For the analysis/findings see [`../RESULT.md`](../RESULT.md)
(§ "HFLM — loss geometry per Gaussian curvature").

## Common to all 6 figures

- **Task/data:** `data=sudoku`, `difficulty=hard`, gpt-style sudoku tokenizer,
  `num_valid=2000`, `data_seed=42`. Conditional task (`[BOS] puzzle(89) [BOS]
  solution(89)`; loss on solution cells only).
- **Algo/model:** `algo=hflm`, `model=tiny-hyperbolic-dit`, `noise=log-linear`,
  `invert_time_convention=false`.
- **Fixed HFLM knobs:** `prior_cov=0.25`, `rho_max=12`. Only `gaussian_curvature`
  (and the best init/lr) vary across curvatures.
- **Training:** `seed=1`, `max_steps=20000`, `optim` per row.
- **Selection:** best `(init, lr)` per curvature by sudoku accuracy
  (`eval/results.json`), at **seed rs1** (the accessible seed).
- **L(t) eval:** 33-point t-grid in [0.001, 1.0]; 8 val batches × 16 = 128 seqs;
  EMA weights. Drawn with the **non-dev1 `claude/curv`** tool
  `/share/thickstun/sychou/workspace/research/s-flm/visualization/loss_geometry_curv.py`
  (HFLM needs the `gaussian_curvature` knob, absent on `main`).
- **Checkpoint base dir:**
  `/share/thickstun/sychou/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku/`
- **Checkpoint files** (in each run's `checkpoints/`): `26-5000.ckpt`,
  `53-10000.ckpt`, `79-15000.ckpt`, `106-20000.ckpt` (+ `last.ckpt`).

## Per-figure

| Figure (`K<k>{,_log}.png`) | gaussian_curvature | init | init_std | lr | steps drawn | run dir (under base) |
|---|---|---|---|---|---|---|
| `K0.25` | −0.25 | custom | 0.04 | 5e-4 | 5/10/15/20K | `d-hard_k-0.25_i-c0.04_lr5e-4_rs1` |
| `K0.3`  | −0.3  | custom | 0.01 | 3e-4 | 5/10/15/20K | `d-hard_k-0.3_i-c0.01_lr3e-4_rs1` |
| `K0.5`  | −0.5  | custom | 0.01 | 3e-4 | 5/10/15/20K | `d-hard_k-0.5_i-c0.01_lr3e-4_rs1` |
| `K0.7`  | −0.7  | custom | 0.01 | 3e-4 | 5/10/15/20K | `d-hard_k-0.7_i-c0.01_lr3e-4_rs1` |
| `K1.0`  | −1.0  | custom | 0.01 | 3e-4 | 5/10/15/20K | `d-hard_k-1.0_i-c0.01_lr3e-4_rs1` |
| `K1.5`  | −1.5  | custom | 0.01 | 3e-4 | 5/10/15/20K | `d-hard_k-1.5_i-c0.01_lr3e-4_rs1` |

All six figures now have the full four curves (5/10/15/20K). K0.5 was re-trained
from scratch (2026-07-09) to regenerate its 5K/10K, which its original run lacked.

`overlay{,_log}.png` = one curve per curvature at its final step (all 20K).

## Provenance caveat

The best-*scoring* config for K=0.3/0.5/0.7/1.0/1.5 was seed **rs2/rs3** (trained
under the `ch2263` account, checkpoints on unreachable `/scratch/ch2263`). These
figures use the same best `(init, lr)` config at **seed rs1** — the best
hyperparameters, not the highest-scoring seed. Only `K0.25` (rs1) is also its
best seed.
