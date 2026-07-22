# hflm_ada_sc_tinystories_256 — Experiment Design

H-FLM with the **adaptive noise schedule**, swept over the geo_curv Part B "H-FLM Sweep"
grid **× self-conditioning {on, off}**. Isolates two axes the naive Part B fixes:
does the adaptive schedule and/or self-cond change H-FLM's curvature/init behaviour
(including whether it mitigates the small-init collapse)?

## Grid — 90 cells
- init: random(std 0.02), c0.01(std 0.01), c0.04(std 0.04)  (3)
- prior_cov: 0.5, 0.8, 1.0  (3)
- K (gaussian_curvature): −0.01, −0.1, −0.25, −0.5, −0.75  (5)
- self_cond: on, off  (2)   → 3×3×5×2 = **90**

Fixed: small-hyperbolic-dit 768/12/12, **adaptive** noise (`log-linear-adaptive`, no
truncation), rho_max=12, 30k steps, global batch 512, seq 256, bf16, EMA 0.9999,
AdamW lr 3e-4 wd 0 clip 1.0. Eval: exact velocity, top_k_v=1, 180 steps, greedy last →
`eval/ppl.json` + `eval/samples_genppl.json`. Scripts: `hlfm_adaptive.sh` (train, +SELF_COND)
/ `hflm_adaptive.sh` (eval, matching adaptive schedule).

> **Collapse caveat (real data):** in the prior 132-cell *naive* H-FLM sweep, of these 9
> init×prior_cov combos only **c0.01 × prior_cov=1.0** collapsed (entropy 0.00, PPL 459.8);
> the other 8 were healthy (entropy 3.36–4.14). Collapse elsewhere in that sweep was
> sporadic (training instability). Adaptive+SC may shift this — it's part of what we measure.

## GPU allocation (nice=0, prioritized — ASAP)
| Site | Hardware | K subset | Cells | per-gpu bs |
|---|---|---|---|---|
| **ch2263** | nlplarge-compute-01, 8× A100-80GB | −0.5, −0.75 | 36 | 64 |
| **ARC Falcon** | L40S-48G / A30-24G | −0.1, −0.25 | 36 | 32 |
| **ARC TC** | A100-80G / H200 (saturated) | −0.01 | 18 | 64 |

1 GPU/cell, run concurrently (throughput-optimal). ch2263 is its own cluster; tc/falcon
share ARC `/home` so their K sets are disjoint. `--curvatures` override allows live
rebalancing if a site starves. Idempotent (skip on `eval/samples_genppl.json`), `--requeue`
+ ckpt-5k auto-resume; checkpoints deleted after eval. rsync ARC→unicorn before `report.py`.

## Deliverable
`experiments/report.py hflm_ada_sc_tinystories_256` → `RESULTS.md` (val/ppl, GenPPL,
entropy per cell; flag entropy<3.0 collapse; compare adaptive×SC vs naive Part B).
