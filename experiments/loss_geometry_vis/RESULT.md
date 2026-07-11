# Loss Geometry

Loss-vs-flow-time curves in the style of LangFlow **Fig. 2 (Left)**: one curve per
checkpoint, plotting the algorithm's own training denoising loss `L(t)` against
flow time `t`, produced by `visualization/loss_geometry.py`.

## Method

For each checkpoint we pin the model's flow time to a fixed grid instead of
sampling it, then evaluate the algo's own `_loss` (EMA weights, TinyStories val
split) at each `t`:

- **t-grid:** 33 points, `t ∈ [0.001, 1.0]` linear.
- **Data:** 8 val batches × 16 seqs = 128 sequences/point, length 256.
- **Weights:** EMA (as `trainer.validate` uses), same as eval.
- **Checkpoints:** training steps 5K / 20K / 30K per run.
- **Metric:** token-mean cross-entropy (nats).

**Time convention** (`invert_time_convention=false`, MDLM/diffusion): `t=0` is
clean, `t=1` is pure noise, so every curve reads **left (clean) → right (noise)**,
matching the paper. Sampling starts at `t=1` because generation begins from the
prior.

**Pure-noise ceiling:** at `t=1` the input carries no information, so `L(1)`
should equal the corpus unigram entropy. TinyStories unigram entropy ≈ **5.940
nats** — every curve below tops out at 5.92–5.97, confirming the pipeline is
faithful.

> ⚠️ **Nominal `t` is not the same physical noise across schedules or across
> checkpoints of a learned schedule.** See [Caveat](#caveat-what-t-means) — read
> this before comparing runs point-by-point.

Figures are organized as `{dataset}/{run}/` — each run folder holds its linear-Y
(`.png`), log-Y (`_log.png`) and `‖x_t‖`-axis (`_xtnorm.png`, `_xtnorm_log.png`)
figures plus a cached `.json`. Datasets: `tinystories/` and `sudoku_hard/` (the
HFLM-curvature runs are `sudoku_hard/hflm_K*/`).

## TinyStories runs

All `outputs/adv_geo_tinystories_s256/*` unless noted, `lr=1e-3`,
`invert_time_convention=false`.

| Figure | Algo | Noise schedule | Notes |
|---|---|---|---|
| `eflm_naive_geo` | EFLM | log-linear | from `naive_geo_tinystories_s256/eflm` |
| `hflm_std0.04_pc1.0` | HFLM | log-linear | `hflm_sweep_…`, init_std=0.04, prior_cov=1.0 |
| `sfm_ada_lr1e-3` | SFM | log-linear **adaptive** (spline) | not truncated |
| `sfm_ada_trunc_lr1e-3` | SFM | log-linear adaptive + **truncated** | α rescaled to ≤ 0.121 |
| `sfm_trunc_lr1e-3` | SFM | log-linear **truncated** | α rescaled to ≤ 0.121, no adaptive |
| `lf_ada_lr1e-3` | LangFlow | **Gumbel** (trainable) | self-cond off, logit-bias warmup 5000 |
| `lf_ada_sc_lr1e-3` | LangFlow | Gumbel (trainable) | **self-cond on** (p=0.25) |

### Per-run geometry (nats; t=0.001 / 0.50 / 1.0 at step 30K)

- **EFLM — naive geometry.** `0.001 / 0.002 / 5.917`. Loss is ≈0 across almost
  the entire schedule; *all* denoising signal is crammed into a thin `t→1`
  sliver, and it sharpens over training. This is the pathological "naive"
  geometry — the model is only doing work at the noisiest step.
- **HFLM (std0.04, pc1.0).** `0.001 / 0.31 / 5.97`. Signal concentrated at
  low-to-mid `t`; the curve saturates to the ceiling already by `t≈0.75`. Learns
  the low-noise regime well, does little in the top half.
- **SFM + adaptive.** `0.000 / 0.004 / 5.926`. Best-behaved: the low-loss region
  **expands monotonically** with training (5K rises at `t≈0.5`; 30K stays ≈0
  until `t≈0.75`). The adaptive spline concentrates learning cleanly.
- **SFM + adaptive + truncated.** `0.047 / 2.05 / 5.927`. Whole curve shifted up
  (truncation caps signal at α=0.121; see caveat). **Non-monotonic** over
  training — 30K mid-range (2.05) is *worse* than 20K (1.21), a sign of
  instability from combining the adaptive spline with truncation.
- **SFM + truncated.** `0.014 / 0.47 / 5.927`. Truncated but non-adaptive:
  improves monotonically 5K→30K, steep rise after `t≈0.5`. Low floor (~0.014).
- **LangFlow + adaptive (Gumbel).** 5K: `0.000 / 0.006 / 5.94`; 30K:
  `0.570 / 2.93 / 5.931`. **The low-`t` loss *rises* over training** — 5K is
  near-zero up to `t≈0.7`, but by 20K/30K the floor jumps to ~0.55. Partly a
  schedule-remap artifact (the trainable Gumbel params shift, so nominal
  `t=0.001` maps to a noisier γ at 30K than at 5K — see caveat), but the shift
  is large enough to be a genuine training-dynamics signal worth investigating.
- **LangFlow + adaptive + self-cond.** 5K: `0.000 / 0.006 / 5.936`; 30K:
  `0.561 / 3.07 / 5.930`. Essentially **identical** to `lf_ada` — self-conditioning
  barely moves the loss geometry on this task.

### Cross-run observations

1. **Adaptive SFM has the cleanest geometry** — monotonic expansion of the
   well-modeled region, no instability.
2. **Truncation** lifts the whole curve and steepens the mid-range (its nominal
   `t` axis covers only the noisy band α ≤ 0.121); combined with the adaptive
   spline it also becomes non-monotonic across checkpoints.
3. **LangFlow-Gumbel degrades at low `t` over training** and is **insensitive to
   self-conditioning** here.
4. **EFLM's naive geometry** (signal only at `t→1`) contrasts sharply with the
   SFM/HFLM curves that spread signal across low-to-mid `t`.

## Caveat: what `t` means

The x-axis is **nominal flow time**, not a fixed physical noise level. How `t`
maps to physical corruption depends on each run's schedule:

- **SFM/EFLM/HFLM (log-linear):** `t` maps directly, `α_t = 1−t`. Comparable
  across these runs.
- **Truncated (`sfm_trunc`, `sfm_ada_trunc`):** the base α∈[≈0,1] is rescaled to
  `[α_min, 0.121]`. So nominal `t=0` corresponds to physical `α=0.121` (only ~12%
  signal, never clean) and `t=1` to `α≈0`. The entire curve lives in the noisy
  band — **do not compare these to untruncated runs at the same nominal `t`.**
- **LangFlow (Gumbel) & adaptive SFM (spline):** `t` is mapped through a
  **learned** schedule stored in the checkpoint. Because that schedule changes
  during training, the *same* nominal `t` maps to *different* physical noise at
  5K vs 30K. Each curve is internally in-distribution for its checkpoint, but
  cross-checkpoint low-`t` comparisons conflate schedule drift with model quality.

Each curve is valid *in-distribution* for its own model; the point-by-point
overlay is only apples-to-apples within the log-linear group.

## Sudoku (hard, seed=1)

Same tool applied to the re-trained hard/seed=1 baselines in
`s-flm/outputs/hflm_curv_init_lr_sudoku/bl_d-hard_a-*_rs1` (non-dev1 tree,
`claude/curv` branch). Figures in [`sudoku_hard/`](sudoku_hard/); checkpoints at
5K/10K/15K/20K (`max_steps=20000`).

**6 of the 7 requested** — `s-flm + ada` (pure adaptive, no truncation) was not
re-trained, so it's omitted. Runs drawn: `sfm`, `sfm_trunc`, `sfm_trunc_ada`
(= s-flm+ada+trunc), `eflm`, `langflow_ada`, `langflow_full` (= langflow+ada+sc).

> Code provenance: the checkpoints were trained on `claude/curv`; the only
> `algo.py` diff vs `main` is inside **HFLM** (the `gaussian_curvature` knob).
> SFM/EFLM/LangFlow classes are byte-identical, so loading with the dev1 tool is
> exact for all 6 runs (none is HFLM).

### The Sudoku task is *conditional* — read L(t) differently

Each example is `[BOS] puzzle(89) [BOS] solution(89)` (180 tokens); **loss is
computed only on the solution cells, with the puzzle given as a fixed prefix**
(`dataloader.py:285,328`). Consequences for the geometry:

- There is **no unigram-entropy ceiling** here. At `t=1` the solution cells are
  pure noise, but the puzzle is still visible, so `L(1)` = the model's residual
  **solve-from-clues error** per solution token — not the marginal entropy.
- `L(1)` should *decrease with training* as the solver improves — and it does
  (e.g. sfm 0.395→0.287 over 5K→20K). That, not a fixed ceiling, is the validity
  check for these curves.
- With any signal left on the solution cells (`t < ~0.75`) completion is trivial,
  so loss sits at ≈0; **all the structure is in the `t→1` region.** The curves are
  much flatter/lower than TinyStories — the log-Y figures are the readable ones.

### Per-run geometry (nats; L(1) = solve loss at pure noise)

| Run | L(1): 5K → 20K | Behavior |
|---|---|---|
| `sfm` | 0.395 → 0.287 | clean, monotonic improvement; action only at `t→1` |
| `sfm_trunc` | 0.297 → 0.238 | small rise from `t≈0.75` (truncation caps solution-cell signal); monotonic |
| `sfm_trunc_ada` | 0.236 → **0.424** | **non-monotonic / unstable** — degrades after 10K, worst at 20K; rise from `t≈0.5` |
| `eflm` | 0.330 → 0.265 | clean, monotonic; signal only at `t→1` |
| `langflow_ada` | 0.490 → 0.294 | improves overall, but a **mid-t bump at `t≈0.75` emerges by 20K** (0.147) |
| `langflow_full` | 0.478 → 0.266 | same as langflow_ada; self-conditioning barely changes it |

### Cross-run observations (Sudoku)

1. **`sfm`, `eflm`, `sfm_trunc` are stable** — L(1) falls monotonically; the model
   steadily gets better at solving from clues.
2. **`sfm_trunc_ada` is unstable** — L(1) rises after 10K and the loss spreads to
   lower t. Same adaptive+truncation instability seen on TinyStories.
3. **LangFlow develops a late `t≈0.75` bump** (both ada and full) that isn't
   present early in training — the trainable Gumbel schedule reallocates mass as
   it learns (nominal-t remap caveat applies). **Self-conditioning (`_full` vs
   `_ada`) makes essentially no difference**, echoing the TinyStories result.
4. Every Sudoku curve is flat-≈0 until `t→1`, mirroring EFLM's "naive geometry"
   on TinyStories — but here it's intrinsic to the conditional task, not a
   pathology.

## HFLM — loss geometry per Gaussian curvature

One curve per Gaussian curvature `K`, using the **best `(init, lr)`** config of
each curvature from the Sudoku-hard sweep `outputs/hflm_curv_init_lr_sudoku`
(`prior_cov` is **fixed at 0.25**; only `gaussian_curvature = −K` varies).
"Best" = highest `eval/results.json` sudoku accuracy (num_correct / 2000).
Figures in [`sudoku_hard/hflm_K*/`](sudoku_hard/). **HFLM must be drawn with the non-dev1
`claude/curv` code** (the `gaussian_curvature` knob is absent on `main`; dev1
would silently use K=−1) — done via `s-flm/visualization/loss_geometry_curv.py`.

| K | best config (init/lr) | best acc (seed) | drawn | final step | L(1) |
|---|---|---|---|---|---|
| 0.25 | c0.04 / 5e-4 | 0.459 (rs1) | ✅ rs1 (=best) | 20K | 0.250 |
| 0.3  | c0.01 / 3e-4 | 0.502 (rs2) | ⚠️ rs1 (0.310) | 20K | 0.246 |
| 0.5  | c0.01 / 3e-4 | **0.582** (rs2) | ⚠️ rs1 (fresh run) | 20K | 0.244 |
| 0.7  | c0.01 / 3e-4 | 0.467 (rs3) | ✅ rs1 (0.361) | 20K | 0.251 |
| 1.0  | c0.01 / 3e-4 | 0.461 (rs2) | ⚠️ rs1 (0.357) | 20K | 0.264 |
| 1.5  | c0.01 / 3e-4 | 0.451 (rs2) | ✅ rs1 (0.278) | 20K | **0.436** |

**Accuracy vs curvature** peaks at **K=0.5 (0.58)** and falls toward both extremes
— a real curvature effect (the interesting scientific result of the sweep).

**Loss geometry vs curvature (final, seed rs1, all at 20K):** L(1) is tight for
K=0.25–1.0 (**0.244–0.264**) but jumps to **0.436 at K=1.5** — and K=1.5's L(1)
*rose* over training (0.306@5K → 0.436@20K), i.e. the highest curvature is
unstable / degrades. So the curvature effect that shows up as an accuracy drop at
the extremes is mirrored in the loss geometry at K=1.5. The overlay is now a fair
same-step (20K) comparison across all six curvatures.

**Status / caveats:**
- **All 6 curvatures drawn; task complete** (seed rs1). The best-*scoring* config
  for 5/6 was seed **rs2/rs3** (trained under `ch2263`, checkpoints on unreachable
  `/scratch/ch2263`); the user re-ran the **rs1** sweep (accessible), so figures
  use the best `(init,lr)` config at **seed rs1** — the best hyperparameters, not
  the highest-scoring seed. Only K=0.25's rs1 run is also its best seed.
- **Final steps: all 6 at 20K** (K=0.3's rs1 run has since finished to 20K), so
  the overlay is a fair same-step comparison.
- The **2-hour poll is now retired** (cron `81ebad32` deleted) — all 6 best-config
  rs1 runs are finished, so no further checkpoints are coming. To upgrade to the
  higher-scoring **rs2/rs3** seeds would require retrieving them from `ch2263`
  (needs your authorization); recipe in `sudoku_hard/hflm_curv_POLL_STATE.md`.

## Reproduce

```bash
# TinyStories: one run, steps mode (per-checkpoint curves); linear + log-Y + .json
sbatch visualization/loss_geometry.sbatch \
  --mode steps \
  --project outputs/adv_geo_tinystories_s256 \
  --run sfm_ada_lr1e-3 \
  --steps 5000 20000 30000 \
  --out experiments/loss_geometry_vis/tinystories/sfm_ada_lr1e-3

# Sudoku: checkpoints live in the non-dev1 tree; --project is absolute, steps 5-20K
sbatch visualization/loss_geometry.sbatch \
  --mode steps \
  --project /share/thickstun/sychou/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku \
  --run bl_d-hard_a-sfm_rs1 \
  --steps 5000 10000 15000 20000 \
  --out experiments/loss_geometry_vis/sudoku_hard/sfm
```

GPU forward passes → always run on a compute node via
`visualization/loss_geometry.sbatch` (never the login node). If desa/thickstun are
saturated, move a pending job to the shared partition without cancelling:
`scontrol update jobid=<id> Partition=gpu Features=gpu-high` (gpu-high = a6000/6000ada
class, 48 GB, sm_86+ — safe for the cu128 build).
