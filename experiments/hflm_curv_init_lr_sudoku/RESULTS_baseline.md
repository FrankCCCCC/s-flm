# Sudoku Baselines — Seed-Averaged Results (jul09_2026 spec)

Full-board solve rate (%), mean ± seed-std over seeds {1,2,3}. Eval: exact velocity, top_k_v=-1, 180 steps, greedy last (LangFlow top_k=1 canonical).

coverage: 21/21 (algo,diff) groups | 63/63 seed-runs

| Model | easy | medium | hard |
|---|---|---|---|
| AR | 14.7 ± 3.5 (n=3) | 3.4 ± 0.3 (n=3) | 0.5 ± 0.3 (n=3) |
| S-FLM (naive) | 78.8 ± 1.1 (n=3) | 43.8 ± 3.2 (n=3) | 11.1 ± 1.7 (n=3) |
| S-FLM + trunc | 94.4 ± 0.4 (n=3) | 79.8 ± 1.7 (n=3) | 42.4 ± 3.4 (n=3) |
| S-FLM + trunc + adaptive | 95.0 ± 0.8 (n=3) | 76.7 ± 7.3 (n=3) | 42.2 ± 2.8 (n=3) |
| E-FLM (naive) | 88.2 ± 1.2 (n=3) | 62.2 ± 2.3 (n=3) | 19.2 ± 3.3 (n=3) |
| LangFlow + ada sched | 81.2 ± 0.9 (n=3) | 52.4 ± 2.7 (n=3) | 18.2 ± 2.1 (n=3) |
| LangFlow + ada sched + SC | 97.0 ± 0.5 (n=3) | 87.2 ± 1.9 (n=3) | 50.4 ± 4.6 (n=3) |

# Appendix

## Sweep Config

**Grid (63 cells = 21 (algo, diff) groups × 3 seeds).** Spec: `slides/jul09_2026`; orchestrated by `sweep_baseline.py`, aggregated by `analyze_baseline.py` (seed mean ± std over seeds {1,2,3}). Companion to the H-FLM curvature sweep (`sweep.py`) run under the **same eval protocol** so the comparison is apples-to-apples. Not a hypothesis test — a measurement of trustworthy seed-averaged baseline solve rates.

| axis | values | n |
|---|---|--:|
| algo | ar, sfm (S-FLM naive), sfm_trunc (+truncation), sfm_trunc_ada (+truncation+adaptive), eflm (E-FLM naive), langflow_ada (+adaptive schedule), langflow_full (+ada+self-cond) | 7 |
| difficulty | easy (40 clues), medium (35), hard (30) | 3 |
| seed | 1, 2, 3 (reported average) | 3 |

LR is **not** swept — fixed at the config default **3e-4** (the slide fixes LR = {3e-4}).

**Fixed (all algos, identical to the curvature sweep).** tiny DiT 512/8/8 (~28.6M) · 20k steps · batch 256 · seq 180 · bf16 · EMA 0.9999 · AdamW lr 3e-4, wd 0, betas 0.9/0.999, eps 1e-8, clip 1.0 · cross-entropy loss. Each algo maps to its single-run scripts `scripts/{train,sample}/sudoku/{ar,sfm,sfm_truncated,sfm_truncated_adaptive,eflm,langflow}.sh`; a `SEED` knob was added to the six baseline train scripts on 2026-07-07 (mirroring `hflm.sh`) — the only code change this experiment needed. LangFlow's two variants are its `VARIANT` knob: `ada_sched`, `full` (= ada_sched + self-conditioning).

**Eval (identical to the curvature runs).** `sudoku_eval`, 180 sampling steps, greedy last step, 2000-puzzle val set, exact 81-cell match. FLM family (`sfm*` / `eflm`): shared defaults `velocity=exact`, `top_k_velocity=−1` (avg across vocab). `ar`: autoregressive greedy decode (no velocity/top-k). `langflow`: analog knob `sampler.top_k`, left at the canonical fair-comparison value **top_k=1** (the value behind the deck's "Recall the Former Results" LangFlow row; switchable via `--langflow-topk -1`).

**Compute.** All 63 cells: 1 GPU/cell, 6 h wall limit, `--nice=0 --requeue`, idempotent + resumable (skip if `eval/results.json` exists or job queued; resume from `last.ckpt`, ckpt every 5k). Planned 3-site disjoint algo split (unicorn {ar, sfm}; tc {eflm, langflow_ada}; falcon {sfm_trunc, sfm_trunc_ada, langflow_full}); the actual re-run (2026-07-08, **checkpoints retained**) rebalanced live to **32 cells on Falcon + 31 on unicorn** when Falcon got priority-walled mid-run. Results gathered to unicorn `/share` via `gather_baseline.sh` (`--ckpts` also pulls the retained ~2.2 G/cell checkpoints before `/scratch` is purged).