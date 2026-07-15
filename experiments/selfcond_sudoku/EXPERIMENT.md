# selfcond_sudoku — LangFlow-style self-conditioning for the geometric flow LMs

**Date:** 2026-07-12 (phase 1: 9-cell hard pilot) / 2026-07-13 (phase 2: full
jul02 grid) · **Branch:** `claude/sehflm_sc`

## Hypothesis

Self-conditioning is LangFlow's single biggest win on sudoku-hard
(`18.2 → 50.4` solve-rate, jul09 baseline sweep). The mechanism — feed the
model's own predicted-clean embedding back as an auxiliary input through
zero-init `W_in`/`W_sc` — is geometry-agnostic, so the same trick
(`algo.SelfConditioning`, `SELF_COND=true`) should lift S-FLM / E-FLM / H-FLM
as well, with the largest headroom for the weaker E-FLM.

## Design

**Phase 1 — pilot (done 2026-07-13, see RESULTS.md):** 9 cells,
`algo ∈ {sfm_trunc_ada, eflm, hflm}` × `seed ∈ {1,2,3}`, hard only, SC on,
lr 3e-4 (script default), HFLM at script defaults (K=-1, init=hyperbolic).
Result: +27 / +26.5 / +31.5 pt — hypothesis confirmed, so phase 2 runs the
full controlled grid from slides/jul02_2026 §"SFLM, EFLM, HFLM + Self Cond".

**Phase 2 — full grid (108 cells):**
`algo ∈ {sfm_trunc_ada, eflm, hflm}` × `difficulty ∈ {medium, hard}` ×
`lr ∈ {3e-4, 5e-4, 1e-3}` × `self-cond ∈ {on, off}` × `seed ∈ {1,2,3}`.
`p_self_cond=0.25` (paper value). Recipe otherwise the fair-comparison one:
tiny DiT (512/8/8, seq 180), 20k steps, global batch 256, AdamW wd=0,
bf16, EMA 0.9999. Init: sfm/eflm `ngpt` (script default); **hflm
`custom std=0.01`, K=-0.5** (the curvature sweep's best hard config, per the
slide), ρ_max=12, prior_cov=0.25.

Eval: `sudoku_eval`, 180 steps, velocity=exact, `top_k_velocity=-1`, greedy
last step — the jul09 baseline/curvature-sweep protocol. Caveat: this branch's
HFLM sampler treats `exact` as the argmax geodesic endpoint regardless of
`top_k_velocity` (the expected-velocity variant lives on the unmerged
`hflm_refactor` branch), so hflm numbers here are argmax-eval — the same code
path as the reused curvature numbers, keeping sc-on/off comparable. At
sampling, self-conditioning is on iff trained with it (carry `z_sc` = soft
predicted-clean, zeros at step 0), mirroring LangFlow Alg. 2.

**Reuse (36/108 cells, symlinked into `outputs/selfcond_sudoku` by sweep.py —
same recipe + eval protocol, verified 2026-07-13):**

| cells | count | source |
|---|---|---|
| sc-on · hard · lr3e-4 · sfm_trunc_ada/eflm | 6 | phase-1 runs `sc_d-hard_a-{algo}_rs{s}` |
| sc-off · lr3e-4 · sfm_trunc_ada/eflm | 12 | jul09 baselines `bl_d-{diff}_a-{algo}_rs{s}` |
| sc-off · hflm · all lrs | 18 | curvature sweep `d-{diff}_k-0.5_i-c0.01_lr{lr}_rs{s}` |

The remaining **72 cells** are new train+eval jobs. Phase-1's 3 hflm cells
(K=-1, init=hyperbolic) are not part of the phase-2 grid and stay as extra
data points.

## Reference numbers (sudoku-hard, 3-seed unless noted)

| Model | hard acc (no SC) | source |
|---|---|---|
| S-FLM + trunc + adaptive | 42.2 ± 2.8 | hflm_curv_init_lr_sudoku/RESULTS_baseline.md |
| E-FLM | 19.2 ± 3.3 | ibid. |
| H-FLM (script defaults, n=1) | 24.1 | experiments/hflm/RESULTS.md |
| H-FLM (K=-1, lr=3e-4, best init c0.01) | 40.4 ± 5.3 | hflm_curv_init_lr_sudoku |
| LangFlow + ada sched | 18.2 ± 2.1 → **50.4 ± 4.6 with SC** | RESULTS_baseline.md |

## Compute

Phase 2: 72 jobs × 1 GPU (unicorn `thickstun,desa`, exclude desa-compute-01,
2 CPU, 16 GB, 6 h limit). Train ≈ 2 h + eval ≈ 15 min per cell →
≈ 145 GPU-hours; ~8–12 h wall clock at 15–25 concurrent GPUs (competing
pending sweeps deprioritized via `scontrol update nice`, per Agent.md rule 1).
Self-cond adds ~12.5% train FLOPs (extra no-grad pass on 25% of batches) and
~0 at eval. Outputs → `outputs/selfcond_sudoku/{run}`; logs →
`experiments/selfcond_sudoku/logs/`.

## Launch

```
python experiments/selfcond_sudoku/sweep.py            # symlink reuse + submit rest
python experiments/selfcond_sudoku/sweep.py --dry-run  # inspect
```
Idempotent: skips cells with `eval/results.json` (incl. reuse symlinks) or a
queued job of the same name; resubmission auto-resumes from
`checkpoints/last.ckpt`.
