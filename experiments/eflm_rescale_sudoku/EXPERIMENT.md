# eflm_rescale_sudoku — embedding norm R vs decoding time (EFLM, sudoku hard)

## Hypothesis

From the random-codebook signal analysis (slides/jul09_2026, jul24_2026;
`visualization/codebook_signal_vs_lossgeo.py`): **the word-embedding norm
decides the decoding time.** For EFLM's interpolant `x_t = (1-t) e + t ε` the
true-word cosine stays above the distractor ceiling `τ = C/√d` until

    t*(R) = 1 / (1 + C / (R √(1 − C²/d))),     C = √(2 ln(2(V−1)/δ))

so **longer embeddings decode later and more abruptly** (loss ≈ 0 until a late,
sharp take-off), while **shorter embeddings decode earlier over more
timesteps** (earlier take-off, smoother L(t) slope). By pinning every embedding
norm to a fixed R (`algo.rho_min = algo.rho_max = R`, `_radius_rescale`), the
transition should also become uniform across tokens — no rare/frequent split.

Sudoku numbers (V = 12, d = 512, δ = 0.1): C = 3.28, τ = 0.145. Predicted
transitions over the sweep grid:

| R     | 0.1  | 0.5  | 1    | 1.5  | 2    | 5    | 8    | 16   | 22   | 32   |
|-------|------|------|------|------|------|------|------|------|------|------|
| t*(R) | 0.03 | 0.13 | 0.23 | 0.31 | 0.38 | 0.60 | 0.71 | 0.83 | 0.87 | 0.91 |

(t convention: t = 0 clean, t = 1 noise; noise norm E‖ε‖ ≈ √d ≈ 22.6, so the
grid brackets it — small R decodes early/smoothly, R ≳ √d decodes late/sharply.)

## Design

- Data: sudoku **hard** (30 clues), 48k train / 2k val.
- Model: `tiny-sphere-dit` (hidden 512, 8 blocks, 8 heads, ~28.6M).
- Algo: `eflm`, `invert_time_convention=false`, bs 256, 20k steps — identical
  to `scripts/train/sudoku/eflm.sh` except `algo.rho_min = algo.rho_max = R`.
- Grid (`sweep.py`): **R × arm × LR × seed** on sudoku hard →
  10 × 2 × 3 × 3 = **180 jobs**.
  - R ∈ {0.1, 0.5, 1, 1.5, 2.0, 5.0, 8, 16, 22, 32} (`algo.rho_min = rho_max`).
  - arm ∈ {`naive` = log-linear noise (`eflm_rescale.sh`), `ada` = log-linear-
    **adaptive** noise, no truncation (`eflm_rescale_adaptive.sh`)}.
  - LR ∈ {3e-4, 5e-4, 1e-3}; seed ∈ {1, 2, 3} (averaged).
- Eval: `mode=sudoku_eval`, 180 steps, exact velocity, greedy last,
  top_k_velocity=-1 (matching `hflm_curv_init_lr_sudoku` protocol), passing
  the same R (the sampler rebuilds the model from CLI overrides; the rescale
  also enters the sampler's velocity table and clue clamping). The `ada` arm's
  eval mirrors the adaptive noise config so the fitted schedule loads.

## Success criteria

1. **Ordering** (primary): measured per-R loss-geometry curves L(t)
   (`visualization/loss_geometry.py` on `checkpoints/last.ckpt`) show the
   transition time increasing monotonically with R and tracking t*(R); slope
   at the transition sharpens with R.
2. **Smoothness**: small-R runs show a visibly smoother/earlier L(t) ramp
   (the hypothesis' "more timesteps to decode" signature).
3. **Sanity**: solve accuracy (`eval/results.json`) stays in a reasonable band
   vs the unrescaled baseline for mid-range R; degradation at extreme R is
   itself informative (signal too weak / prior mismatch).

## Compute

- 180 × 1-GPU SLURM jobs, partition `thickstun,desa` (exclude
  `desa-compute-01`), 2 cpus, 16G, walltime 6h (train ~2-4h + eval).
- Outputs: `outputs/eflm_rescale_sudoku/{tag}/` with `eval/` inside,
  `tag = eflmrs_{arm}_r-{R}_lr-{lr}_d-hard_rs{seed}`.
  **Checkpoints are kept** (~1-2G/cell) — the loss-geometry analysis needs
  them; clean up after RESULTS.md is written.

## Run

    python experiments/eflm_rescale_sudoku/sweep.py --dry-run   # inspect (180)
    python experiments/eflm_rescale_sudoku/sweep.py             # submit all
    # subsets, e.g. the naive lr3e-4 pilot first:
    python experiments/eflm_rescale_sudoku/sweep.py --adas naive --lrs 3e-4 --seeds 1

Post-hoc analysis (compute node): loss-geometry L(t) per R on
`outputs/eflm_rescale_sudoku/*/checkpoints/last.ckpt`, then RESULTS.md with
the measured t50/slope vs the t*(R) table above.
