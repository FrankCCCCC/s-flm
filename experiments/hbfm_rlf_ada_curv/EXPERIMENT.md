# hbfm_rlf_ada_curv — adaptive per-position curvature for HBFM on hard Sudoku

Spec: `setup.md`. Math: `math.md`. Sweep: `sweep.py` (orchestration only; calls
`scripts/train/sudoku/hbfm.sh` + `scripts/sample/sudoku/hbfm.sh`). Table: `report.py` →
`RESULTS.md`.

## Method (what was added)

One knob, `algo.curvature_bound` = b (1 = the fixed-curvature HBFM, unchanged):

- The hyperbolic DiT gets a second zero-init readout, one curvature output per position. Read at
  the t = 0 state (every free position at the origin, the prompt pinned), it gives
  λ_i = b^tanh(h_i) ∈ [1/b, b]; position i runs at curvature λ_i·K.
- Curvature = clock: position i's bridge state is the base bridge at heat time λ_i·t.
- The CE of position i is weighted by λ_i, the importance weight of that clock. The mean of the
  loss does not depend on λ; its variance does.
- λ is trained on that variance (VDM App. I.2): the gradient reaching λ is multiplied by twice the
  sequence's loss. The readout sees detached trunk features, so the denoiser is trained on the
  loss only; it is read on the scale of one hidden unit (÷ √D). Both choices were forced by a
  pre-flight measurement (`RESULTS.md` §2).
- The sampler reads λ once at its initial state and steps position i by λ_i·dt.

## Verification (before any training run)

`tests/test_hbfm_ada_curv.py` (15 tests, all pass) and `verify_invariance.py`; numbers in
`RESULTS.md` §1. Five tests of `tests/test_hbfm.py` fail with and without this change (the yaml
default is `forward_type: naive` / float32 since 03bb642; the tests assume horosphere / float64).

| claim (`math.md`) | check |
|---|---|
| §1 curvature = clock | `q_xt` at curvature λ matches the fixed-curvature bridge at heat time λ·t (KS on radius and angle) |
| §3 mean invariant | weighted loss at λ ∈ {1/4, 1/2, 1, 2, 4, mixed} equals the fixed-curvature model's within Monte-Carlo error |
| §4 variance not | the variance moves 12× over the same λ, U-shaped: cliff toward flat, linear toward sharp |
| §6 λ's gradient is the variance gradient | pathwise gradient through the hook = score form E[loss²(1 − r·t)] |
| §6 closed-form H³ radial sampler | KS against the closed-form CDF and against the grid sampler; differentiable in t |
| §7 sampler | SDE with per-position steps reproduces the exact bridge at λ·t; Bayes oracle decodes the prior at a slow and a fast position |
| trunk still trained | after the no-grad curvature pass the loss gradient reaches the trunk (autocast cache cleared) |

## Hypotheses

- **H1 (best cell).** At the best fixed-curvature cell of `claude_test_hbfm_sudoku` (exp proposal,
  rate 3, K = −0.5, horosphere, LR 3e-4; 63.8 ± 3.9 % on the 2080 Ti harness) the adaptive
  curvature changes board accuracy by less than the seed spread. That study's central finding was
  that the heat-time proposal acts as a training-mass allocator and that variance-optimal
  proposals did not win, so a variance-trained curvature is not expected to beat a tuned cell.
- **H2 (mis-set proposal).** The curvature can re-scale every position's clock by up to b, so it
  should repair proposals the fixed-curvature study found mis-set: a too-slow rate (rate 0.5,
  unit rate 1: 31.3 ± 3.5 %) and a flat proposal (`unif`, ub 2.3: 35.6 ± 6.3 %). These gaps are
  25–30 points, far above the ~6-point resolution of a 3-seed mean, so one seed is informative.
- **H3 (what λ learns).** λ starts as one global scale and acquires per-position structure slowly.
  Reported from the checkpoints: its mean, spread over cells, and distance from the bounds.

## Verdicts (2026-10-03; numbers in `RESULTS.md` §4–5)

- **H1 — confirmed.** Fixed 62.7 ± 3.4 vs adaptive 59.1 ± 7.5 (b = 4) and 60.3 ± 3.8 (b = 2),
  three seeds each: −3.6 and −2.4 against s.e. of the difference 4.7 and 2.9 (pooled adaptive
  59.7 ± 5.4, n = 6: −3.0 at 1.0 s.e.). No gain; every adaptive point estimate is below the control.
  The b = 4 arm's seed spread is twice the fixed arm's; its weakest seed is also the worse denoiser
  at λ = 1.
- **H2 — confirmed for the slow and the flat proposal, with a smaller margin than the old
  references suggested.** Rate 0.5: 44.0 → 55.5 against the same-code control (+11.5, single
  seeds). `unif`: 60.1 against an older-code 35.6 ± 6.3 (+24). The too-fast proposal (rate 6) is
  not repaired (53.6 vs an older 62.6): its variance optimum is a faster clock still.
- **H3 — as predicted.** One global scale first, per-position structure from ~400 steps, drifting
  slower for the cells to solve as the model learns the prompt (anchor 1.09 at 5k → 0.37 at 20k)
  and faster for the separators; λ ∝ proposal rate; Spearman(λ, cell entropy) −0.66 to −0.84 in
  every cell; the optimum sits inside [1/4, 4] and at the lower bound of [1/2, 2].
- **Variance (the mechanism `setup.md` asked for):** held-out loss mean unchanged (≤ 1.3 s.e.) and
  variance 1.6–4.4× lower with the learned curvature than with the same weights at λ = 1, in ten
  of eleven adaptive runs; none for rate 6.

## Design

Pinned by `setup.md`: hard Sudoku (30 clues), 48k train / 2k val (data seed 42); tiny DiT 512/8/8;
`prod_factor_dim=[3,3,3]`, base curvature [−0.5]×3; 20k steps, batch 256, bf16, EMA 0.9999;
AdamW (wd 0, betas 0.9/0.999, eps 1e-8), clip 1.0; log-linear schedule; HyLa off; eval = exact
velocity, `top_k_velocity=-1`, 180 steps, greedy last step, 2000 boards.

Stage 1 (seed 1, 8 cells):

| block | cells | fixed-curvature reference |
|---|---|---|
| A anchor | exp rate 3, b ∈ {1, 2, 4} | b = 1 is the control, run here |
| B slow rate | exp rate 0.5 (unit rate 1), b ∈ {1, 4} | b = 1 is the control, run here (31.3 ± 3.5 on A-series GPUs) |
| C flat proposal | `unif`, ub 2.3 (t_max 2.3), b = 4 | 35.6 ± 6.3 (round 3 of `claude_test_hbfm_sudoku`, same GPU type and batching) |
| D readout | `forward_type=naive`, b = 4 | 55.9 (n = 1, same source) |
| E fast rate | exp rate 6 (unit rate 12), b = 4 | 62.6 at rate 6.9 (n = 1, same source) |

Stage 2 (submitted with stage 1): seeds 2 and 3 of block A — H1 needs three seeds per arm. Seeds
2 and 3 of a B–E cell only if it moves by more than the noise floor.

The references of C–E are from an earlier code version; they are quoted as context, not as
controls. Only A and B have same-code controls.

Sampler horizon: `t_max` 6 for `exp` (unit time 3), ub for the bounded mode (the HBFM validator
rejects `t_max > ub`). Position i reaches base heat time λ_i·t_max.

## Compute

`desa,thickstun`, 2080 Ti only (`desa-compute-01`, 8 GPUs, 11 GB), 1 CPU / 14 GB per cell — the
node has 8 free cores. Both arms use 128 sequences × 2 accumulation = the optimizer batch of 256.

| arm | measured | per cell (20k steps + eval) |
|---|---|---|
| fixed curvature | 1.36 micro-batches/s → 1.47 s per optimizer step | 8.2 h + 12 min |
| adaptive (one extra no-grad trunk pass) | 1.02 micro-batches/s → 1.96 s per optimizer step | 10.9 h + 12 min |

GPU memory 9.1 GB (fixed) / 9.2 GB (adaptive). 14 cells ≈ 135 GPU-h; with all 8 GPUs, ~20 h
wall-clock. A–E and the stage-2 seeds were submitted together on 2026-10-02 13:25 EDT.
