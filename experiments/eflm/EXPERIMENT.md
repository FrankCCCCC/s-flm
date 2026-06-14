# EXPERIMENT: EFLM (Euclidean Flow Language Model) on Sudoku

**Owner:** sc3379@cornell.edu · **Date:** 2026-06-13 · **Branch:** `claude/langflow`
**One-line:** Add the **flat / Euclidean** corner to the geometry comparison — naive flow-matching in
raw embedding space — and measure it against naive S-FLM (sphere) and naive HFLM (hyperbolic) under one
identical recipe.

This is a *comparison run*, not a new method. EFLM is deliberately the **naive** model: uniform `t`,
**straight-line** interpolation as the "geodesic", a Gaussian prior, and the **same cross-entropy loss**
as S-FLM/HFLM — *no* truncation, *no* adaptive schedule, *no* normalization. It is "S-FLM with the
sphere flattened."

---

## 1. Hypothesis

Three geometries, one denoiser objective. S-FLM lives on the sphere `S^{d-1}`, HFLM on `H^d`, **EFLM on
flat `R^d`**. If geometry is what lifts the *naive* model (the paper's story: naive HFLM ≫ naive S-FLM),
then the flat-space point should fall **below** both curved geometries — quantifying how much curvature
(positive or negative) buys over doing nothing.

Falsifiable form: with the **identical** backbone (`tiny-sphere-dit`, 8×512, 8 heads, ~28.6M), data,
batch, step budget, and 180-step exact-velocity greedy sampler, EFLM trains without NaNs and yields a
Sudoku exact-match accuracy that we slot into the naive-model table below.

EFLM = S-FLM with two lines changed: (i) `get_sphere_embeddings` → `get_raw_embeddings` (no unit-sphere
projection), (ii) slerp → lerp (straight line); prior is `N(0, prior_cov·I)` instead of uniform-on-sphere.

---

## 2. Identical recipe (same for S-FLM / HFLM / EFLM)

| Item | Value |
|---|---|
| Model | `tiny-sphere-dit` — 8 blocks, hidden **512**, 8 heads, seq-len **180**, **~28.6M** params |
| Training | **20,000** steps, global batch **256**, bf16, EMA **0.9999**, grad-clip **1.0** |
| Optimizer | AdamW, lr **3e-4**, wd **0**, betas (0.9, 0.999), eps 1e-8 |
| Data | Sudoku, **48k train / 2k val** per difficulty, seed 42 · easy 40 / med 35 / hard 30 clues |
| Noise | `log-linear` (vanilla — naive, **no** truncation/adaptive), `invert_time_convention=false` |
| Sampler @ eval | `mode=sudoku_eval`, **180** steps, **exact** velocity, **greedy**, EMA on, full 2k puzzles |
| top_k_velocity | **−1** (average over full vocab) **and** **1** (top-1 endpoint) — both reported |

**EFLM-specific (the only geometry knobs):** raw (un-normalized) Euclidean embeddings `E`, Gaussian
prior `N(0, prior_cov·I)` with `prior_cov=1.0`, straight-line velocity `v = Σ_k p_k(e_k − x) = p@E − x`,
Euler update `x ← x + dt·v` (same `sfm_step_size` dt as S-FLM).

> **Scale note (first knob if EFLM underperforms).** The recipe reuses S-FLM's `ngpt` init
> (per-coord std `1/√d`, ‖e_v‖≈1) but the default `prior_cov=1.0` gives a prior of norm ≈ `√d`. Unlike
> S-FLM (data and noise both unit-norm on the sphere) the data/noise scales here differ; the input
> LayerNorm absorbs the scale so training is stable, but the effective SNR schedule is shifted toward
> noise. If EFLM lands far below the naive curves, the fix is scale-matching — `model.init=unit_var`
> (à la LangFlow) or `prior_cov≈1/d` — and is itself a finding ("naive Euclidean needs the scale trick
> that the sphere gets for free from normalization").

---

## 3. Scripts

| Phase | Script | Notes |
|---|---|---|
| train | `scripts/train/sudoku/eflm.sh` | mirrors `sfm.sh`, `algo=eflm` |
| sample | `scripts/sample/sudoku/eflm.sh` | mirrors `sfm.sh` sample, `sampler=eflm`, `TOPK_VELOCITY` env |
| runner | `experiments/eflm/run_sudoku.sbatch` | train → eval(tkv=−1) → eval(tkv=1); 1 job/difficulty |

Submit (one GPU each, `thickstun`/`desa`):
```bash
for d in easy medium hard; do
  sbatch --job-name=eflm_$d --export=ALL,DIFFICULTY=$d experiments/eflm/run_sudoku.sbatch
done
```

---

## 4. Comparison table — naive models, Sudoku exact-match accuracy (%)

Reference numbers are the user-provided reproductions (S-FLM, HFLM); **EFLM rows are this experiment.**

**top_k_velocity = −1 (velocity = average over full vocab)**

| naive model | Space | Easy | Med | Hard |
|---|---|---|---|---|
| S-FLM (naive) | sphere | 77.6 | 32.6 | 13.9 |
| **EFLM (naive)** | **Euclidean** | **90.35** | **62.05** | **17.55** |
| HFLM (naive) | hyperbolic | 93.75 | 67.40 | 24.10 |

**top_k_velocity = 1 (velocity = top-1 endpoint)**

| naive model | Space | Easy | Med | Hard |
|---|---|---|---|---|
| S-FLM (naive) | sphere | 76.6 | 33.2 | 15.2 |
| **EFLM (naive)** | **Euclidean** | **90.25** | **62.45** | **18.70** |
| HFLM (naive) | hyperbolic | 93.75 | 67.40 | 24.10 |

> **Result (2026-06-13):** EFLM ≫ S-FLM at all three, just below HFLM → ranking **HFLM > EFLM > S-FLM**.
> Full write-up + reading + verdict in [`RESULTS.md`](RESULTS.md).

Tricked S-FLM anchors (for context, not naive): +α⋆(0.093) trunc ≈ 95/78/48; +adaptive ≈ 94/79/46.

---

## 5. Success criteria

Primary metric: Sudoku **exact-match accuracy (%)** over the 2k val puzzles at 180 steps (paper's
metric, `main.py` `(generated == gt).all(dim=1)`). 1 seed per cell (mirrors the paper / the existing
S-FLM/HFLM runs). AR-greedy ≈ 14.6/5.1/1.0 is the "learned global structure" floor.

| Verdict | Condition |
|---|---|
| **Confirmed (geometry ranking holds)** | EFLM trains cleanly and lands **below** naive HFLM at all three difficulties; EFLM ≥ floor on easy (>30%). The flat point quantifies the curvature gain. |
| **Surprise** | EFLM ≈ or > naive HFLM/S-FLM somewhere — flat space competitive; report and inspect (likely the scale knob matters more than geometry). |
| **Inconclusive** | EFLM trains (CE ↓, no NaN) but collapses to the floor (≤ ~15% easy) — most likely the §2 scale mismatch; rerun scale-matched before drawing a geometry conclusion. |
| **Refuted (impl bug)** | NaNs in CE / `log_x_theta`, divergence, or sampler produces non-finite trajectories. |

---

## 6. Status / artifacts

- **Code:** `algo.EFLM` (q_xt raw-embedding lerp corruption + CE), `samplers.EFLMSampler`
  (straight-line Euler, `gaussian` prior, `top_k_velocity`/`velocity` knobs), `main.py` dispatch,
  `configs/algo/eflm.yaml`, `configs/sampler/eflm.yaml`. Smoke-tested end-to-end (job 376127).
  Bug fixed during bring-up: `_lerp` alpha broadcast (`algo.py`, mirrors `utils.slerp`).
- **Checkpoints:** `outputs/sudoku/eflm_{easy,medium,hard}/checkpoints/last.ckpt`.
- **Eval results:** `eval_runs/sudoku/eflm_{difficulty}_tkv{-1,1}/results.json`.
- **Logs:** `experiments/eflm/logs/eflm_{difficulty}.<jobid>.log`.

_Status: jobs submitted (easy/medium/hard). Table filled in §4 / a RESULTS.md once evals land._
