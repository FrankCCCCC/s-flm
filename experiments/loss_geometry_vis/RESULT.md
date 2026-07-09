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

Figures live in [`tinystories/`](tinystories/); each run has a linear-Y (`.png`)
and log-Y (`_log.png`) figure plus a cached `.json` of the raw curves.

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

## Sudoku (hard, seed=1) — pending

Same tool, applied to `outputs/hflm_curv_init_lr_sudoku` (hard, seed=1) for
s-flm, s-flm+ada, s-flm+ada+trunc, s-fm+trunc, eflm, langflow+ada,
langflow+ada+sc. **Blocked on checkpoints** — the original runs retained no
checkpoints, so these are being **re-trained** (`bl_d-hard_a-*_rs1` jobs). Figures
will go in [`sudoku_hard/`](sudoku_hard/) once checkpoints at 5K/10K/15K/20K exist.
The tool is data-agnostic (reads each run's `.hydra/config.yaml`), so no code
change is needed.

## Reproduce

```bash
# one run, steps mode (per-checkpoint curves); writes linear + log-Y + .json
sbatch visualization/loss_geometry.sbatch \
  --mode steps \
  --project outputs/adv_geo_tinystories_s256 \
  --run sfm_ada_lr1e-3 \
  --steps 5000 20000 30000 \
  --out experiments/loss_geometry_vis/tinystories/sfm_ada_lr1e-3
```

GPU forward passes → always run on a compute node via
`visualization/loss_geometry.sbatch` (never the login node).
