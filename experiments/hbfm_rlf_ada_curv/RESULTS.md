# hbfm_rlf_ada_curv — Results

Adaptive per-position curvature for HBFM on hard Sudoku (design: `EXPERIMENT.md`, math:
`math.md`). Regenerate the table with `python experiments/hbfm_rlf_ada_curv/report.py`.

**Status: complete — all 14 cells evaluated (2026-10-03 12:40 EDT). Conclusion in §5.**

<!-- report:begin -->
14 cells evaluated (2000 boards each).

| curvature bound | mode | rate | K | unit rate | ub | t_max | fwd | LR | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|
| fixed (1) | exp | 0.5 | -0.5 | 1 | na | 6 | horo | 3e-4 | 44.0% | pending | pending | 44.0 (n=1) |
| 4 | exp | 0.5 | -0.5 | 1 | na | 6 | horo | 3e-4 | 55.5% | pending | pending | 55.5 (n=1) |
| fixed (1) | exp | 3 | -0.5 | 6 | na | 6 | horo | 3e-4 | 63.6% | 65.5% | 58.9% | **62.7 ± 3.4** (n=3) |
| 2 | exp | 3 | -0.5 | 6 | na | 6 | horo | 3e-4 | 62.5% | 55.9% | 62.5% | **60.3 ± 3.8** (n=3) |
| 4 | exp | 3 | -0.5 | 6 | na | 6 | horo | 3e-4 | 66.5% | 51.5% | 59.4% | **59.1 ± 7.5** (n=3) |
| 4 | exp | 3 | -0.5 | 6 | na | 6 | naive | 3e-4 | 62.6% | pending | pending | 62.6 (n=1) |
| 4 | exp | 6 | -0.5 | 12 | na | 6 | horo | 3e-4 | 53.6% | pending | pending | 53.6 (n=1) |
| 4 | unif | 3 | -0.5 | 6 | 2.3 | 2.3 | horo | 3e-4 | 60.1% | pending | pending | 60.1 (n=1) |

<!-- report:end -->

## 1. Verification of the math (no training involved)

`tests/test_hbfm_ada_curv.py`: 15 / 15 pass. Numbers below from `verify_invariance.py`: the real
HBFM model at init with the horosphere readout (zero residual = the context-free Bayes posterior,
for which the invariance is exact), (H³)⁴ at K = −1, exp proposal of rate 1, 65,536 sequences per
row.

**The mean does not depend on the curvature; the variance does** (`math.md` §3–4):

| curvature scale λ | mean ± s.e. | variance |
|---|---|---|
| fixed-curvature model (old code path) | 0.7269 ± 0.0033 | 0.721 |
| 1 (adaptive code path) | 0.7279 ± 0.0033 | 0.726 |
| 1/4 | 0.7310 ± 0.0030 | 0.595 |
| 1/2 | 0.7296 ± 0.0022 | 0.305 |
| 2 | 0.7225 ± 0.0051 | 1.699 |
| 4 | 0.7225 ± 0.0075 | 3.706 |
| mixed per position [1/4, 1/2, 1, 2, 4, 1/2, 2, 1] | 0.7231 ± 0.0035 | 0.795 |

- Every mean is within 1.3 s.e. of the fixed-curvature model's. The variance moves 12×.
- The variance is U-shaped, as §4 predicts: minimum near λ = 1/2, linear growth on the sharp side
  (0.73 → 1.70 → 3.71), and a rise again at λ = 1/4, the edge of the variance cliff.
- The residual downward trend of the mean (0.731 → 0.7225) is the proposal's floor at the origin:
  heat times below 0.001 are never drawn, which removes λ·0.001·log V from every position — 1.4 %
  at λ = 4. It is a finite-horizon effect (§3), not a defect of the weight.

**The curvature's gradient is the variance gradient** (§6): pathwise gradient through the hook of
`algo.py` against the score form E[loss²(1 − r·t)], global scale, 16 batches of 4096:

| λ | pathwise (hook) | score form | s.e. of the difference |
|---|---|---|---|
| 1/2 | 0.232 | 0.179 | 0.095 |
| 1 | 0.972 | 0.964 | 0.012 |
| 2 | 2.010 | 2.023 | 0.019 |

At λ = 1/4 the two estimates disagree (0.02 vs −1.89 ± 0.71): that point sits on the variance
cliff, where the squared loss has no finite second moment and neither sample mean is reliable.

## 2. Training dynamics of the curvature (pre-flight, before the sweep)

`diag_grad.py` runs the anchor cell with b = 4 outside Lightning (batch 64, 100-step warmup, same
optimizer and clip) and prints gradient norms and λ. Four designs of the curvature output were
measured; logs in `logs/grad_*.log`.

| design | gradient norm of the rest of the model | clip factor | λ over training |
|---|---|---|---|
| extra logit channel, trunk trained through the curvature pass | 35–100 (steps 150–750) | 0.01–0.03 | 1 → 2.2 in 40 steps, then ±20 % jitter per step |
| same, readout ÷ D | 35–100 | 0.01–0.03 | 1 → 2.3 in 200 steps |
| separate readout on detached trunk features, ÷ D | 0.5–3 (the model's own) | 0.3–1.0 | inert: 1.02 ± 0.02 for 1200 steps |
| **separate readout on detached trunk features, ÷ √D, gradient × 1/√D (kept)** | 0.3–3 | 0.3–1.0 | 1 → 1.63 (step 200) → 1.23 (step 1500) |

- **Trunk in the graph.** The variance gradient then dominates the gradient of the whole model and
  the global clip (1.0) scales the denoising gradient down ~50×. Hence the detached readout: the
  trunk and the logits are trained on the loss only.
- **Readout scale.** A sum over D = 512 hidden units moves log λ by ~0.1 per Adam step whatever the
  gradient's size, so λ followed the sign of each batch's noisy variance gradient; ÷ D makes it
  inert. ÷ √D tracks: λ rises while the model is untrained (the context-free posterior resolves
  slowly, so a sharper curvature lowers the variance) and comes back down as the model learns to
  use the prompt.
- **Per-position structure appears early.** Kept design, spread of λ over positions: 0.00 at step
  100, 0.44 at step 600 (range 0.61–1.95), 0.21 at step 1500 (range 0.70–1.52).
- **Autocast.** The no-grad curvature pass poisons autocast's weight cache (the trunk silently
  drops out of the graph in the loss forward that follows); `HyperbolicDiT.curvature` clears it.
  Negative control: without the clear the trunk's gradient is `None`, with it 0.36.

Kept design, every 300 steps (`logs/grad_sqrtD_893101.log`; `|g_curv|` is the curvature readout):

| step | loss | \|g_curv\| | \|g_rest\| | λ mean | λ std over positions | λ min / max |
|---|---|---|---|---|---|---|
| 0 | 1.746 | 0.280 | 2.45 | 1.000 | 0.000 | 1.00 / 1.00 |
| 300 | 1.145 | 0.003 | 1.80 | 1.537 | 0.277 | 0.76 / 1.74 |
| 600 | 0.771 | 0.003 | 1.30 | 1.423 | 0.436 | 0.61 / 1.95 |
| 900 | 0.348 | 0.003 | 0.76 | 1.325 | 0.293 | 0.66 / 1.76 |
| 1200 | 0.166 | 0.001 | 0.46 | 1.244 | 0.226 | 0.68 / 1.58 |
| 1500 | 0.095 | 0.001 | 0.27 | 1.225 | 0.205 | 0.70 / 1.52 |

## 3. Smoke (300 steps + the 2000-board eval, 2080 Ti)

| arm | micro-batch | speed | per optimizer step | GPU memory | train → checkpoint → eval |
|---|---|---|---|---|---|
| fixed curvature (b = 1) | 128 × 2 | 1.36 it/s | 1.47 s | 9.1 GB | runs; 0 / 2000 boards |
| adaptive (b = 4) | 128 × 2 | 1.02 it/s | 1.96 s | 9.2 GB | runs; see below |

- 0 % is expected at 300 steps: the eval uses the EMA weights (decay 0.9999), which have barely
  left their initialization.
- The adaptive arm costs 1.33× the fixed one (one extra trunk pass without gradients), i.e.
  10.9 h instead of 8.2 h per 20k-step cell.
- The tracebacks in every log (`OSError: [Errno 16] Device or resource busy ... pymp-*`) are the
  dataloader workers failing to remove their temp dir on NFS at exit. Both arms print six of them;
  they do not affect training or the eval.

## 4. Board accuracy at 20k steps, 2000 hard boards (stage 1: seed 1 of every cell; stage 2: seeds 2–3 of the anchor block)

| cell | fixed curvature | adaptive, bound 4 | Δ |
|---|---|---|---|
| A anchor: exp, rate 3, horosphere | **62.7 ± 3.4** (63.6, 65.5, 58.9; this sweep, 3 seeds) | **59.1 ± 7.5** (66.5, 51.5, 59.4; bound 4); **60.3 ± 3.8** (62.5, 55.9, 62.5; bound 2); 3 seeds each | **−3.6** (bound 4), **−2.4** (bound 2) — both inside the noise, both negative |
| B slow proposal: exp, rate 0.5 (unit rate 1) | **44.0** (this sweep, seed 1); 31.3 ± 3.5 in `claude_test_hbfm_sudoku` (A-series GPUs, 3 seeds, older code) | **55.5** | **+11.5** vs the same-code control (+24 vs the earlier reference) |
| C flat proposal: `unif`, ub 2.3, t_max 2.3 | 35.6 ± 6.3 (round 3 of the same study, same GPU type, 3 seeds) | **60.1** | **+24** |
| D plain logits: `naive`, rate 3 | 55.9 (round 3, n = 1) | **62.6** | +6.7 |
| E fast proposal: exp, rate 6 (unit rate 12) | 62.6 at rate 6.9 (round 3, n = 1); 57.6 at rate 4 | **53.6** | **−9** |

Noise floor: the earlier study measured a per-seed std of ~5.4 points on this task (3-seed means
resolve gaps ≳ 6 points).

- **H2 holds (mis-set proposals), with a smaller margin than the old references suggested.** The
  same-code rate-0.5 control scores 44.0 on this harness, 13 points above the earlier study's 31.3
  (different GPU type, batching and code), so the clean repair at B is **+11.5** on single seeds —
  1.5× the ~7.6-point std of a difference of two single seeds — and the +24 at C is measured
  against the earlier study's number (same GPU type and batching, older code) and may shrink the
  same way. Both are the cells whose checkpoints show the curvature of the cells to solve at the
  lower bound and the loss variance cut by 4×.
- **H1 holds: no accuracy gain at the tuned anchor.** Three seeds per arm: fixed 62.7 ± 3.4,
  adaptive 59.1 ± 7.5 (bound 4) and 60.3 ± 3.8 (bound 2). The differences of means (−3.6, −2.4)
  are below the s.e. of those differences (4.7, 2.9); pooling the two adaptive arms, 59.7 ± 5.4
  (n = 6) vs 62.7 ± 3.4 (n = 3) is −3.0 at 1.0 s.e. So the arms are not distinguishable in
  accuracy, with every adaptive point estimate on the negative side. The bound-4 arm's seeds span
  15 points (51.5–66.5) against 6.6 for the fixed arm, and its weak seed is also the worse denoiser
  (held-out loss at λ = 1 0.0140 vs 0.0112 for seed 1, 2.5 s.e.; table below): the variance
  objective did not stabilise training, it added a seed-dependent degree of freedom.
- **D** (+6.7, single seeds on both sides) is suggestive only.
- **E — the too-fast proposal is not repaired, and may be hurt (−9, single seeds).** A proposal
  that is too fast puts its mass before the cells are decidable; the variance-optimal reaction is a
  faster clock, which in the sampler commits the cells earlier — the opposite of what accuracy wants
  (the earlier study found accuracy prefers the slower side of the variance optimum). Its checkpoint
  confirms it: cells to solve at λ = 0.63, the fastest of all cells, and no variance reduction
  (table below).
- The fixed-curvature anchor reproduces the earlier study's 63.8 ± 3.9 on this harness (63.6).

### Interim: what the curvature has learned by step 5,000 (of 20,000)

`curvature_stats.py` and `loss_stats.py` on the 5k-step checkpoints (EMA weights, 256 validation
boards, run on CPU — `common.py` swaps the CUDA-only flash-attention kernels for their PyTorch
equivalents; on the smoke checkpoint the CPU path gives λ = 1.167, the GPU path 1.167). Seed 1.

λ by kind of solution position (mean ± std over positions), bound b = 4:

| cell | empty cells (to solve) | given cells | row separators | Spearman(λ, givens constraining the cell) | Spearman(λ, posterior entropy at t = 0) |
|---|---|---|---|---|---|
| anchor: exp, rate 3 | 1.09 ± 0.06 | 1.04 ± 0.06 | 0.88 ± 0.01 | −0.09 | +0.11 |
| slow proposal: exp, rate 0.5 | **0.51 ± 0.05** | 0.76 ± 0.05 | 1.43 ± 0.01 | +0.57 | −0.71 |
| flat proposal: `unif`, ub 2.3 | **0.50 ± 0.04** | 0.83 ± 0.06 | 1.62 ± 0.01 | +0.58 | −0.64 |
| plain logits: exp, rate 3, `naive` | 0.97 ± 0.01 | 0.98 ± 0.01 | 1.01 ± 0.00 | +0.33 | −0.61 |

Held-out weighted loss, 256 boards × 8 draws of (t, noise):

| cell | curvature | mean ± s.e. | variance |
|---|---|---|---|
| anchor, b = 4 | learned λ | 0.0152 ± 0.0008 | 0.0014 |
| | λ = 1, same weights | 0.0154 ± 0.0008 | 0.0013 |
| anchor, fixed-curvature control | — | 0.0186 ± 0.0008 | 0.0015 |
| slow proposal, b = 4 | learned λ | 0.0154 ± 0.0013 | **0.0034** |
| | λ = 1, same weights | 0.0152 ± 0.0019 | 0.0075 |
| flat proposal, b = 4 | learned λ | 0.0198 ± 0.0016 | **0.0053** |
| | λ = 1, same weights | 0.0197 ± 0.0023 | 0.0111 |
| plain logits (`naive`), b = 4 | learned λ | 0.0146 ± 0.0008 | 0.0013 |
| | λ = 1, same weights | 0.0145 ± 0.0008 | 0.0013 |

- **The mean does not move, on trained models too.** Switching the learned curvature off changes
  the held-out loss by at most 0.0002 in all four cells (≤ 0.3 s.e.).
- **At the tuned proposal the curvature has nothing to do.** With rate 3 (both readouts) λ stays
  within 0.9–1.2 and the variance is the same with and without it.
- **At a mis-set proposal it moves the way §4 of `math.md` predicts and halves the variance.** A
  proposal that is too slow (rate 0.5) or flat (`unif`) puts too little mass where the loss is, and
  the curvature of the cells to solve drops to half the base value (not to the bound, 0.25), which
  stretches their loss profile under the proposal. Variance 0.0034 vs 0.0075 and 0.0053 vs 0.0111
  at the same mean.
- **It is per-cell, and ordered.** In both mis-set cells the more uncertain a cell is at t = 0, the
  flatter (slower) its curvature (Spearman −0.64 / −0.71); cells with more givens in their row,
  column and box run faster (+0.57 / +0.58). In the sampler this is an easy-cells-first order.
- Whether any of this changes board accuracy is what the 20k evals answer; none of it is a result
  about accuracy.

### What the curvature learned by 20k steps (final checkpoints, EMA weights, 256 boards)

λ by kind of solution position, bound b = 4 (`curvature_stats.py`):

| cell | empty cells (to solve) | given cells | row separators | at a bound | Spearman(λ, givens constraining) | Spearman(λ, entropy at t = 0) |
|---|---|---|---|---|---|---|
| anchor: exp, rate 3 | **0.37 ± 0.08** | 0.50 ± 0.06 | 2.49 ± 0.09 | 0 % | +0.42 | −0.78 |
| anchor, seed 2 (the 51.5 % seed) | **0.39 ± 0.07** | 0.54 ± 0.08 | 2.24 ± 0.05 | 0 % | +0.47 | −0.72 |
| anchor, seed 3 | **0.36 ± 0.07** | 0.56 ± 0.09 | 1.65 ± 0.06 | 0 % | +0.44 | −0.83 |
| slow proposal: exp, rate 0.5 | **0.26 ± 0.03** | 0.26 ± 0.02 | 3.71 ± 0.06 | 96 % / 69 % / 6 % | +0.51 | −0.78 |
| flat proposal: `unif`, ub 2.3 | **0.26 ± 0.03** | 0.26 ± 0.01 | **3.85 ± 0.03** | 95 % / 64 % / 97 % | +0.50 | −0.77 |
| plain logits: exp, rate 3, `naive` | **0.33 ± 0.04** | 0.38 ± 0.04 | 1.94 ± 0.11 | 0 % | +0.38 | −0.84 |
| fast proposal: exp, rate 6 | **0.63 ± 0.08** | 0.89 ± 0.10 | 1.03 ± 0.02 | 0 % | +0.36 | −0.66 |
| anchor, **bound 2** | 0.53 ± 0.05 | 0.63 ± 0.06 | 1.58 ± 0.04 | 69 % / 0 % / 0 % | +0.44 | −0.79 |
| anchor, bound 2, seed 2 | 0.53 ± 0.04 | 0.61 ± 0.04 | 1.59 ± 0.02 | 68 % / 0 % / 0 % | +0.42 | −0.81 |
| anchor, bound 2, seed 3 | 0.54 ± 0.05 | 0.67 ± 0.08 | 1.12 ± 0.08 | 64 % / 0 % / 0 % | +0.40 | −0.81 |

Held-out weighted loss, 256 boards × 8 draws (`loss_stats.py`):

| cell | curvature | mean ± s.e. | variance |
|---|---|---|---|
| anchor, b = 4 | learned λ | 0.0102 ± 0.0007 | **0.0009** |
| | λ = 1, same weights | 0.0112 ± 0.0008 | 0.0015 |
| anchor, fixed-curvature control | — | 0.0083 ± 0.0006 | 0.0008 |
| anchor, b = 4, **seed 2** | learned λ | 0.0135 ± 0.0005 | **0.0005** |
| | λ = 1, same weights | 0.0140 ± 0.0008 | 0.0013 |
| anchor, fixed-curvature control, seed 2 | — | 0.0083 ± 0.0007 | 0.0009 |
| anchor, b = 4, **seed 3** | learned λ | 0.0122 ± 0.0005 | **0.0006** |
| | λ = 1, same weights | 0.0131 ± 0.0009 | 0.0017 |
| anchor, fixed-curvature control, seed 3 | — | 0.0094 ± 0.0006 | 0.0008 |
| slow proposal, b = 4 | learned λ | 0.0088 ± 0.0007 | **0.0011** |
| | λ = 1, same weights | 0.0095 ± 0.0015 | 0.0048 |
| slow proposal, fixed-curvature control | — | 0.0103 ± 0.0015 | 0.0046 |
| flat proposal, b = 4 | learned λ | 0.0083 ± 0.0007 | **0.0011** |
| | λ = 1, same weights | 0.0082 ± 0.0014 | 0.0042 |
| plain logits, b = 4 | learned λ | 0.0107 ± 0.0005 | **0.0005** |
| | λ = 1, same weights | 0.0108 ± 0.0008 | 0.0013 |
| fast proposal (rate 6), b = 4 | learned λ | 0.0167 ± 0.0008 | 0.0012 |
| | λ = 1, same weights | 0.0158 ± 0.0007 | 0.0011 |
| anchor, b = 2 | learned λ | 0.0106 ± 0.0006 | **0.0007** |
| | λ = 1, same weights | 0.0104 ± 0.0008 | 0.0012 |
| anchor, b = 2, seed 2 | learned λ | 0.0108 ± 0.0006 | **0.0007** |
| | λ = 1, same weights | 0.0108 ± 0.0007 | 0.0011 |
| anchor, b = 2, seed 3 | learned λ | 0.0101 ± 0.0005 | **0.0006** |
| | λ = 1, same weights | 0.0099 ± 0.0007 | 0.0011 |

- Between 5k and 20k the curvature of the cells to solve kept falling in every cell: anchor
  1.09 → 0.37, `naive` 0.97 → 0.33, rate 0.5 0.51 → 0.26 and `unif` 0.50 → 0.26 (the last two at
  the lower bound for 95 % of the cells). This is the predicted drift: as the model learns to use
  the prompt its loss profile decays faster, and the variance-optimal clock gets slower
  (`math.md` §4, λ* = r/a). The anchor does not reach the bound; the mis-set proposals do.
- The separators go the other way (2.5–3.9, the upper bound under `unif`): a position whose word
  is certain carries no loss, so its clock has no variance to lose by running fast.
- The mean is unchanged in every run (≤ 1.3 s.e. apart); the variance is 1.7× / 2.6× / 2.8×
  (anchor, seeds 1 / 2 / 3), 4.4× (rate 0.5), 3.8× (`unif`), 2.6× (`naive`) and 1.7× / 1.6× / 1.8×
  (bound 2, seeds 1 / 2 / 3) lower with the learned curvature than with the same weights at λ = 1 —
  ten of eleven adaptive runs; the exception is the too-fast proposal (below). The fixed-curvature controls confirm the mechanism
  from the other side: the rate-0.5 control's loss variance (0.0046) is 4× the anchor control's
  (0.0008–0.0009) and matches what the adaptive model sees at λ = 1 (0.0048).
- **A lower loss variance did not translate into accuracy at the anchor.** Seed 2 of the adaptive
  anchor has the lowest variance of all anchor runs (0.0005) and the lowest board accuracy
  (51.5 %); its λ profile is the same as seeds 1 and 3 (cells 0.39 vs 0.37 / 0.36, Spearman −0.72
  vs −0.78 / −0.83), so the curvature did what it was trained to do and the denoiser behind it is
  simply worse (0.0140 vs 0.0112 / 0.0131 at λ = 1). The three fixed-curvature controls are
  tighter (0.0083 / 0.0083 / 0.0094) than the three adaptive denoisers read at λ = 1
  (0.0112 / 0.0140 / 0.0131) — the same ordering as the board accuracies.
- The ordering is strong at 20k in every cell: λ vs the model's own uncertainty about a cell,
  Spearman −0.66 to −0.84 — the least certain cells get the slowest clocks.
- **The learned clock scales with the proposal rate**, as λ* = r/a predicts for a fixed loss
  profile: cells to solve at λ = 0.26 (bound) / 0.37 / 0.63 for rate 0.5 / 3 / 6, i.e. r/λ ≈ 8–10
  per unit of physical heat time at the two unbounded rates. The curvature makes the proposal seen
  by a cell's own clock the same whatever the physical rate — which is why it repairs a too-slow
  proposal, and why it has little left to do when the rate is already high (rate 6: variance
  0.0012 vs 0.0011, no reduction; mean 0.0167 vs 0.0158, 1 s.e.).
- With bound 2 the anchor's cells sit at the lower bound (64–69 % in all three seeds): the
  variance optimum (0.36–0.39 at bound 4) is outside the allowed range.
- **The adaptive denoisers reach a higher path-KL than the fixed controls at the same step.** Read
  on the base ray (λ = 1, the same functional for both models — `math.md` §3b): 0.0112 / 0.0140 /
  0.0131 for the adaptive seeds 1–3 at b = 4 against 0.0083 / 0.0083 / 0.0094 for the controls,
  25–70 % higher (b = 2: 0.0104 / 0.0108 / 0.0099, 5–30 % higher — the smaller shift of the smaller
  bound), while the board accuracies are 66.5 / 51.5 / 59.4 vs 63.6 / 65.5 / 58.9. The loss does not
  order the board accuracy seed by seed at this resolution (an exact match of 51 cells decoded by a
  180-step sampler), but the adaptive arm is worse on both on average. The mechanism is the
  training-mass shift that the variance asks for: the cells are trained at base heat times 0.37×
  the physical ones, an effective proposal rate ≈ 8 instead of 3, i.e. at noisier states than the
  tuned proposal chose — the variance optimum is faster than the accuracy optimum, as the
  fixed-curvature study found.

### Incidents (2026-10-02)

- **One cell lost and resubmitted (my error).** At 15:15 I attached a small analysis process
  (`srun --jobid 893446 --overlap python curvature_stats.py`, 64 boards, no gradients) to the GPU of
  the running rate-0.5 fixed-curvature control. The training process died 30 s later with
  `CUDA error: an illegal memory access` (the 11 GB card was at 9.1 GB), at step 4617 and before its
  first checkpoint; the batch script went on to an eval with no checkpoint and Slurm recorded
  `COMPLETED 0:0`. The cell was resubmitted from scratch (job 897023). No other cell was touched.
  Consequence: intermediate checkpoints are not analysed while training runs; the curvature and
  loss statistics are run as their own jobs afterwards.
- **Eight cells queued behind another sweep for three hours.** Six cells started at submission.
  The other eight (rate 6; bound 2; seeds 2–3 of the anchor block) and the resubmitted control were
  pending behind a sweep that another session queued minutes later at higher priority (~118 jobs);
  two to three 2080 Ti GPUs sat idle from 13:26 to 16:31 with no job of the account being started
  on them. At 16:31 that session's single-GPU jobs left the queue and the cells started as GPUs
  freed up; by 00:20 on 10-03 all but the resubmitted control were running.

## 5. Conclusion

Question asked (`setup.md`): does a learned per-position curvature improve HBFM on hard Sudoku,
and does it lower the variance of the loss?

**Variance: yes, at an unchanged mean.** In ten of the eleven adaptive runs the held-out weighted
loss has the same mean as the same weights at λ = 1 (≤ 1.3 s.e.) and 1.6–4.4× lower variance —
exactly the pattern `math.md` §3–4 predicts (mean invariant to the clock, variance not) and §1 of
this file verified on a toy. The gain is largest where the physical proposal is furthest from the
variance optimum (rate 0.5: 4.4×; `unif`: 3.8×) and absent for the too-fast proposal (rate 6:
0.0012 vs 0.0011), because there the learned clock has nowhere to go: λ* = r/a is already ≈ 1.

**Accuracy: no gain where the proposal is tuned; a repair where it is mis-set; no repair where it is
too fast.**

| situation | fixed curvature | adaptive | verdict |
|---|---|---|---|
| tuned anchor (exp, rate 3), 3 seeds each | 62.7 ± 3.4 | 59.1 ± 7.5 (b = 4); 60.3 ± 3.8 (b = 2) | **no gain** (−3.6 / −2.4, inside the noise); the seed spread doubles at b = 4 |
| too-slow proposal (rate 0.5), 1 seed each | 44.0 | 55.5 | **+11.5** (1.5× the noise of a single-seed difference) |
| flat proposal (`unif`, ub 2.3), 1 seed vs older 3-seed reference | 35.6 ± 6.3 | 60.1 | **+24** against an older-code reference |
| plain logits (`naive`), 1 seed vs older reference | 55.9 | 62.6 | +6.7, suggestive |
| too-fast proposal (rate 6), 1 seed vs older reference | 62.6 | 53.6 | **−9**, not repaired |

Why the two answers differ: the curvature is trained on the variance of the loss, and a lower
variance is a better *estimate* of the same objective, not a better objective. What it changes in
practice is where the training mass goes. The learned clock of a cell to solve is λ ≈ 0.26 / 0.37 /
0.63 at physical rate 0.5 / 3 / 6, so the proposal a cell sees has effective rate r/λ ≈ 1.9 (held
back by the bound) / 8 / 9.5 whatever the physical rate: the curvature is a self-tuning proposal that
drives every cell toward one noise regime — the variance optimum — and runs the certain positions
(separators, λ = 2–4) fast because they have no loss to lose. Read against the fixed-curvature
accuracy curve (rate 0.5 → 44, rate 3 → 63, rate 7 → 63), this explains all four outcomes: a slow
or flat proposal is pulled toward the good regime (+11.5, +24); the tuned anchor is pushed past it
(effective rate 8: −3.6 at b = 4 and −2.4 at b = 2, inside the noise, with a 25–70 % higher
path-KL at λ = 1; at b = 4 the seed spread also doubles, and the weakest adaptive seed is the one
with the lowest loss variance — the extra learned degree of freedom cost stability where it had the
most room); a too-fast proposal is pushed faster still (−9, n = 1). The variance optimum sits on the
fast side of the accuracy optimum, as the earlier study found.

What λ learns (H3) is robust across seeds and cells: a global rise to 1.6 in the first 200 steps,
per-position structure from ~400 steps on (std over positions 0.3 while the mean falls back to 1.2
by 2.5k; §2), then a slow sharpening to 20k; the cells to solve drift slower as the model learns
the prompt (anchor 1.09 at 5k → 0.37 at 20k), the separators faster; the learned clock scales with
the physical rate (λ ∝ r); with bound 2 the anchor's cells sit at the lower bound, so b = 4 is the
smallest bound that leaves the optimum inside the range.

Practical reading: the adaptive curvature is a safe default in the sense that it never changed the
loss mean and it makes the result insensitive to a badly chosen heat-time proposal, at the price
of one extra no-grad trunk pass per step (+33 % step time) and, at b = 4, a wider seed spread at a
tuned proposal. It is not a way to beat a tuned fixed-curvature cell on this task.
