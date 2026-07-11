# Sudoku-hard baselines loss-geometry — checkpoints & hyperparameters

Documents the exact checkpoint path and hyperparameters behind each figure. Each
run's figures + `.json` live in their own `<run>/` subfolder here (e.g.
`langflow_ada/langflow_ada.png`); the HFLM-curvature runs are `hflm_K*/` (see
`hflm_curv_RESULTS.md`). For the analysis/findings see [`../RESULT.md`](../RESULT.md)
(§ "Sudoku (hard, seed=1)").

## Common to all 6 figures

- **Task/data:** `data=sudoku`, `difficulty=hard`, `num_valid=2000`,
  `data_seed=42`. Conditional task (`[BOS] puzzle(89) [BOS] solution(89)`; loss
  on solution cells only) → no unigram ceiling; L(1) = solve-from-clues loss.
- **Model:** `model=tiny-sphere-dit`, `invert_time_convention=false`.
- **Training:** `seed=1`, `lr=3e-4`, `max_steps=20000`.
- **L(t) eval:** 33-point t-grid in [0.001, 1.0]; 8 val batches × 16 = 128 seqs;
  EMA weights. Drawn with `visualization/loss_geometry.py` (dev1 `main`; SFM/EFLM/
  LangFlow classes are identical across trees, so loading is exact).
- **Checkpoint base dir:**
  `/share/thickstun/sychou/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku/`
- **Steps drawn:** 5K/10K/15K/20K for all — files `26-5000.ckpt`,
  `53-10000.ckpt`, `79-15000.ckpt`, `106-20000.ckpt` (+ `last.ckpt`) in each run's
  `checkpoints/`.

## Per-figure

| Figure (`<name>{,_log}.png`) | algo | noise schedule | init | key params | run dir (under base) |
|---|---|---|---|---|---|
| `sfm` | sfm | log-linear | ngpt | — | `bl_d-hard_a-sfm_rs1` |
| `sfm_trunc` | sfm | log-linear, **truncated** | ngpt | `alpha_max=0.093` | `bl_d-hard_a-sfm_trunc_rs1` |
| `sfm_trunc_ada` | sfm | log-linear, **truncated + adaptive** | ngpt | `alpha_max=0.093`, `adaptive=true` | `bl_d-hard_a-sfm_trunc_ada_rs1` |
| `eflm` | eflm | log-linear | ngpt | `prior_cov=1.0` | `bl_d-hard_a-eflm_rs1` |
| `langflow_ada` | langflow | **gumbel** (trainable) | unit_var | `self_conditioning=false`, `logit_bias=true` (warmup 5000), `p_self_cond=0.25` | `bl_d-hard_a-langflow_ada_rs1` |
| `langflow_full` | langflow | **gumbel** (trainable) | unit_var | `self_conditioning=true`, `logit_bias=true` (warmup 5000), `p_self_cond=0.25` | `bl_d-hard_a-langflow_full_rs1` |

Notes:
- `sfm_trunc_ada` = s-flm + adaptive + truncation; `sfm_trunc` = s-flm + truncation.
- `langflow_ada` = LangFlow + trainable-Gumbel schedule; `langflow_full` adds
  self-conditioning (the `_full` variant).
- The requested `s-flm + ada` (pure adaptive, no truncation) has no such baseline
  run at hard/seed=1, so it is not included.
