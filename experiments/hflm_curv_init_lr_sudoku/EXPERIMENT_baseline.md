# Sudoku Baselines — 3-Seed Fair Comparison (spec: slides/jul09_2026)

Companion to the H-FLM curvature sweep (`sweep.py` / `EXPERIMENT.md`). Produces
faithful, **seed-averaged** baseline numbers to replace the deck's single-run
"Recall the Former Results" table, under the **same eval protocol** as the H-FLM
curvature runs so the comparison is apples-to-apples.

## Hypothesis / purpose
Not a hypothesis test — a measurement. Establish trustworthy (3-seed mean ± std)
Sudoku full-board solve rates for every non-hyperbolic baseline, so the H-FLM
curvature result is compared against a fair baseline rather than a lucky single run.

## Grid (63 cells)
- **algo (7):** `ar`, `sfm` (S-FLM naive), `sfm_trunc` (S-FLM+truncation),
  `sfm_trunc_ada` (S-FLM+truncation+adaptive), `eflm` (E-FLM naive),
  `langflow_ada` (LangFlow + adaptive schedule), `langflow_full` (LangFlow + ada + self-cond)
- **difficulty (3):** easy (40 clues) / medium (35) / hard (30)
- **seed (3):** 1, 2, 3  → report the average

## Fixed config (from the slide, identical to the curvature sweep)
tiny DiT 512/8/8 (~28.6M) · 20k steps · batch 256 · seq 180 · bf16 · EMA 0.9999 ·
AdamW **lr 3e-4** (config default; slide fixes LR={3e-4}) wd 0 betas (0.9,0.999)
eps 1e-8 clip 1.0 · cross-entropy loss.

## Eval protocol (identical to the curvature runs)
`sudoku_eval`, 180 sampling steps, greedy last step, 2000-puzzle val set, exact
81-cell match. FLM family (`sfm*`/`eflm`) use the shared default `velocity=exact`,
`top_k_velocity=-1` (avg across vocab). `ar` is autoregressive (no velocity/top-k).
`langflow` uses its analog knob `sampler.top_k`, left at the canonical
fair-comparison value **top_k=1** (the value behind the deck's LangFlow row);
switchable via `--langflow-topk -1` (the literal top_k_v=-1 analog) if desired.

## Method scripts (single-run, env-var driven)
`scripts/train|sample/sudoku/{ar,sfm,sfm_truncated,sfm_truncated_adaptive,eflm,langflow}.sh`.
A `SEED` knob was added to the six baseline **train** scripts on 2026-07-07
(mirroring `hflm.sh`) to enable seeding — the only code change this experiment needed.
LangFlow's two variants are its `VARIANT` knob (`ada_sched`, `full`).

## GPU allocation (re-run 2026-07-08; checkpoints RETAINED)
Re-run of all 63 cells with **checkpoints saved** (the 2026-07-07 run deleted them).
Live check 2026-07-08: unicorn GPU-saturated by another experiment; TinkerCliffs A100
priority-walled (idle CPUs but 8/8 GPUs allocated to higher-priority users); **Falcon
wide open** (~20 idle a30/l40s GPUs). So the whole grid runs on **Falcon**, one site,
one filesystem — simplest gather.

Started all 63 on Falcon; mid-run Falcon got priority-walled by other users (31 cells
stuck `Reason=Priority` behind them). Meanwhile **unicorn freed up** (~28 idle GPUs), so
the 32 already-done stayed on Falcon and the **31 remaining were moved to unicorn** (faster
a6000/a5000/6000ada, ~1.1 h/cell, and checkpoints land straight on `/share`). Actual split:

| site | algos/cells | GPUs | storage |
|---|---|---|---|
| **falcon** | 32 cells (first waves, all algos) | a30/l40s normal | ckpts on `/scratch/shengyenc/...`; results gathered to `/share` |
| **unicorn** | 31 cells (the priority-stuck remainder) | thickstun+desa (a6000/a5000/6000ada) | results **and** ckpts on `/share` directly |

Submitted `--nice=0 --requeue`; idempotent + resumable (skip if `eval/results.json` exists
or job queued; resume from `last.ckpt`, ckpt every 5k). **Checkpoints are RETAINED**
(`save_last` + 5k/10k/15k/20k periodic, ~2.2 G/cell) as a deliverable of this re-run:
31 cells on unicorn `/share`, 32 on Falcon `/scratch` (pull with `gather_baseline.sh --ckpts`
before scratch is purged).

**Storage.** Runs (incl. ~1.8 G/cell checkpoints) live on `/scratch/shengyenc/sfm_output/
hflm_curv_init_lr_sudoku` — scratch is per-cluster, login-visible, huge (713 T free), and
avoids the ARC `/home` per-user quota (which would blow up the login node with 113 G of
checkpoints). `/scratch` is periodically purged, so `gather_baseline.sh` persists the
deliverable back to unicorn `/share` (results.json always; `--ckpts` for the checkpoints).

## Expected wall-clock
~20k steps: ~1.6–1.9 h/run (A5000/A6000/Ada/L40S/A30 @ ~2.9–3.7 it/s) + ~10 min eval.
Falcon (wide open, high parallelism) ≈ 2–4 h; unicorn's 18 drip over ~6–10 h given
CPU contention; tc trickles between OWT jobs. Whole sweep expected complete same day.

## Outputs
On Falcon: `/scratch/shengyenc/sfm_output/hflm_curv_init_lr_sudoku/bl_d-{difficulty}_a-{algo}_rs{seed}/`
with `eval/results.json` + retained `checkpoints/`. `gather_baseline.sh` pulls these to
unicorn `outputs/hflm_curv_init_lr_sudoku/bl_.../eval/results.json` (the `bl_` prefix keeps
them distinct from the curvature runs and out of `analyze.py`'s K-regex). Aggregate with
`analyze_baseline.py` → seed-mean ± std per (algo, difficulty).
