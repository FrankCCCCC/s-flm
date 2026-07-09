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

## GPU allocation (live check 2026-07-07; sites disjoint on the ALGO axis)
tc + falcon SHARE the ARC /home filesystem, so their algo sets must not overlap;
unicorn is a separate filesystem.

| site | algos | cells | queues | note |
|---|---|---|---|---|
| **unicorn** | ar, sfm | 18 | thickstun,desa (excl. desa-compute-01) | CPU-saturated but reliable home FS |
| **tc** (TinkerCliffs) | eflm, langflow_ada | 18 | a100/h200 normal+preemptable | busy with the OWT experiment → light share |
| **falcon** | sfm_trunc, sfm_trunc_ada, langflow_full | 27 | l40s/a30 normal+preemptable | **wide open** → most work |

Submitted identical priority (`sbatch --nice=0 --requeue`); idempotent + resumable
(skip if `eval/results.json` exists or job queued; resubmit resumes from `last.ckpt`,
ckpt every 5k). Checkpoints auto-deleted after eval writes `results.json` (quota).

## Expected wall-clock
~20k steps: ~1.6–1.9 h/run (A5000/A6000/Ada/L40S/A30 @ ~2.9–3.7 it/s) + ~10 min eval.
Falcon (wide open, high parallelism) ≈ 2–4 h; unicorn's 18 drip over ~6–10 h given
CPU contention; tc trickles between OWT jobs. Whole sweep expected complete same day.

## Outputs
`outputs/hflm_curv_init_lr_sudoku/bl_d-{difficulty}_a-{algo}_rs{seed}/eval/results.json`
(the `bl_` prefix keeps them distinct from the curvature runs and out of `analyze.py`'s
K-regex). Aggregate with `analyze_baseline.py` → seed-mean ± std per (algo, difficulty).
