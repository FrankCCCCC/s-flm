# eflm_rescale_tinystories_256 — rescaled EFLM {ada, trunc, trunc+ada} on TinyStories

## Hypothesis / question

Transfer of the sudoku result (experiments/eflm_rescale_sudoku rounds 1-2) to
language modeling: with every word-embedding norm pinned to R, the decode
transition sits at t*(R); a fixed schedule wastes steps outside the transition,
adaptive refits onto it, and static truncation at alpha*(R) recovers most of
the adaptive gain. Does the same schedule ordering (trunc_ada ≥ ada ≥ trunc)
hold on TinyStories seq 256, and does pinning R + matching the schedule beat
the raw-norm naive EFLM baseline?

alpha*(R) = alpha_star_euclidean(V=50257, embed_norm=R)
(= 1 − t*(R), visualization/codebook_signal_vs_lossgeo.py):

| R | 1 | 8 | 28 (≈√d) |
|---|---|---|---|
| alpha*(R) | 0.840 | 0.397 | 0.158 |

Raw-norm reference points (30K ckpt, codebook fig): type-median ‖e‖=6.4,
token-median 138.9, noise norm √d≈27.7 — R∈{1, 8, 28} spans well-below-noise
to at-noise-scale.

## Design

- Grid: **arm × R × 1 seed** = {ada, trunc, trunc_ada} × {1, 8, 28} × {1} →
  **9 jobs** (`sweep.py`).
  - `ada`: `eflm_rescale_truncated_adaptive.sh`, ALPHA_MAX=null (adaptive only)
  - `trunc`: `eflm_rescale_truncated.sh`, ALPHA_MAX=alpha*(R)
  - `trunc_ada`: `eflm_rescale_truncated_adaptive.sh`, ALPHA_MAX=alpha*(R)
- Fixed (mirrors `naive_geo_tinystories_s256`): `small-sphere-dit`, ngpt init,
  30k steps, global batch 512 (DEVICES=4 × PER_GPU_BS=32 × accum 4), seq 256,
  lr 3e-4, adaptive knobs refit 50 / buffer 25600 / ema 0.9 / umix 1e-3.
- Eval: `ppl_eval` (valid PPL flow bound) + `sample_eval` (GenPPL gpt2-large
  retokenized + entropy), exact velocity, top_k_v=1, 180 steps.
- Baseline: `naive_geo_tinystories_s256` `eflm` cell — raw norms, no
  trunc/ada: valid PPL 1.1014, GenPPL 34.58, entropy 3.67. (No rescale-naive
  arm here; add later if the schedule arms warrant it.)

## Success criteria

1. GenPPL (with entropy ≥ 3.0 sanity) vs the naive baseline 34.58 per (arm, R);
   the sudoku prediction is trunc_ada / ada > trunc, with large-R arms only
   viable because the schedule follows/concentrates at the late transition.
2. Valid-PPL bound sane (no hflm-style collapse row).
3. Follow-up (checkpoints kept): loss-geometry L(t) per R — transition should
   sit at t*(R) and be uniform across rare/frequent words (no type/token-median
   split, unlike the raw-norm EFLM codebook figure).

## Compute

- 9 × 4-GPU SLURM jobs, partition `thickstun,desa` (exclude desa-compute-01),
  16 cpus, 64G, walltime 4d (naive_geo template; ~1-2d expected each).
- Outputs: `outputs/eflm_rescale_tinystories_256/eflmrs256_{arm}_r-{R}_rs1/`
  (+ `eval/ppl.json`, `eval/samples_genppl.json`). Checkpoints kept.

## Run

    python experiments/eflm_rescale_tinystories_256/sweep.py --dry-run
    python experiments/eflm_rescale_tinystories_256/sweep.py
