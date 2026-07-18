# TinyStories loss-geometry — checkpoints & hyperparameters

Documents the exact checkpoint path and hyperparameters behind each figure. Each
run's figures + `.json` live in their own `<run>/` subfolder here (e.g.
`eflm_naive_geo/eflm_naive_geo.png`). For the analysis/findings see
[`../RESULT.md`](../RESULT.md) (§ "TinyStories runs").

## Common to all 7 figures

- **Task/data:** `data=tinystories`, `gpt2` tokenizer, `wrap=true`, seq length 256.
  Unconditional LM → pure-noise ceiling L(1) ≈ TinyStories unigram entropy
  (5.940 nats).
- **Training:** `seed=1`, `max_steps=30000`, `invert_time_convention=false`.
- **Steps drawn:** **5K / 20K / 30K** (files `1-5000.ckpt`, `5-20000.ckpt`,
  `8-30000.ckpt`). Each run's `checkpoints/` also holds 10K/15K/25K, not used.
- **L(t) eval:** 33-point t-grid in [0.001, 1.0]; 8 val batches × 16 = 128 seqs;
  EMA weights. Drawn with `visualization/loss_geometry.py` (dev1 `main`).
- **Checkpoint base dir:**
  `/share/thickstun/sychou/workspace/research/s-flm/outputs/`

## Per-figure

Model is `small-*` for all; `lr` and schedule vary.

| Figure (`<name>{,_log}.png`) | algo | model | noise schedule | lr | init | key params | run dir (under base) |
|---|---|---|---|---|---|---|---|
| `eflm_naive_geo` | eflm | small-sphere-dit | log-linear | 3e-4 | ngpt | `prior_cov=1.0` | `naive_geo_tinystories_s256/eflm` |
| `hflm_std0.04_pc1.0` | hflm | small-hyperbolic-dit | log-linear | 3e-4 | custom (`init_std=0.04`) | `prior_cov=1.0`, `rho_max=12` | `hflm_sweep_tinystories_s256/std0.04_pc1.0` |
| `sfm_ada_lr1e-3` | sfm | small-sphere-dit | log-linear, **adaptive** | 1e-3 | ngpt | `adaptive=true` | `adv_geo_tinystories_s256/sfm_ada_lr1e-3` |
| `sfm_ada_trunc_lr1e-3` | sfm | small-sphere-dit | log-linear, **truncated + adaptive** | 1e-3 | ngpt | `alpha_max=0.121`, `adaptive=true` | `adv_geo_tinystories_s256/sfm_ada_trunc_lr1e-3` |
| `sfm_trunc_lr1e-3` | sfm | small-sphere-dit | log-linear, **truncated** | 1e-3 | ngpt | `alpha_max=0.121` | `adv_geo_tinystories_s256/sfm_trunc_lr1e-3` |
| `lf_ada_lr1e-3` | langflow | small-sphere-dit | **gumbel** (trainable) | 1e-3 | unit_var | `self_conditioning=false`, `logit_bias=true` (warmup 5000), `p_self_cond=0.25` | `adv_geo_tinystories_s256/lf_ada_lr1e-3` |
| `lf_ada_sc_lr1e-3` | langflow | small-sphere-dit | **gumbel** (trainable) | 1e-3 | unit_var | `self_conditioning=true`, `logit_bias=true` (warmup 5000), `p_self_cond=0.25` | `adv_geo_tinystories_s256/lf_ada_sc_lr1e-3` |

Notes:
- `eflm_naive_geo` and `hflm_std0.04_pc1.0` are the two original reference figures
  (lr 3e-4); the five `adv_geo` runs are all lr 1e-3.
- `lf_ada_sc` = `lf_ada` + self-conditioning; both use the trainable Gumbel
  (log-NSR) schedule, so t is pinned via the Gumbel quantile γ(t), not `_sample_t`.
