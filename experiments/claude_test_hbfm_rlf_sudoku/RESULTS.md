# claude_test_hbfm_rlf_sudoku — Results

HBFM board accuracy on hard Sudoku with random Laplacian features (HyLa) of the input state vs the plain coordinates (see `EXPERIMENT.md`). Regenerate with `python report.py`. Every `±` is the sample std over seeds (ddof=1).

<!-- report:begin -->
117 cells evaluated (2000 boards each).

| arm | fwd | LR | HyLa scale / sphere degree | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---|---|---:|---:|---:|---:|
| none | horo | 3e-4 | 1 | 61.3% | 62.0% | 61.8% | **61.7 ± 0.4** (n=3) |
| none | horo | 5e-4 | 1 | 66.8% | 57.5% | 56.1% | **60.1 ± 5.8** (n=3) |
| pure | horo | 3e-4 | 1 | 0.0% | pending | pending | 0.0 (n=1) |
| cap5 | horo | 3e-4 | 0.005 | 47.9% | 59.5% | pending | **53.7 ± 8.1** (n=2) |
| cap5 | horo | 3e-4 | 0.005 (unnormalized) | 59.0% | 57.9% | pending | **58.4 ± 0.8** (n=2) |
| cap5 | horo | 3e-4 | 0.05 | 49.5% | 62.4% | pending | **56.0 ± 9.1** (n=2) |
| cap5 | horo | 3e-4 | 0.05 (unnormalized) | 56.4% | 59.3% | pending | **57.8 ± 2.1** (n=2) |
| cap5 | horo | 3e-4 | 0.25 | 68.0% | 59.7% | pending | **63.8 ± 5.8** (n=2) |
| cap5 | horo | 3e-4 | 0.25 (unnormalized) | 63.9% | 61.6% | pending | **62.7 ± 1.7** (n=2) |
| cap5 | horo | 3e-4 | 0.5 | 46.2% | pending | pending | 46.2 (n=1) |
| cap5 | horo | 3e-4 | 1 | 55.0% | 59.9% | 41.2% | **52.0 ± 9.6** (n=3) |
| cap5 | horo | 3e-4 | 2 | 52.8% | pending | pending | 52.8 (n=1) |
| cap5 | horo | 5e-4 | 1 | 63.8% | 0.0% | 67.3% | **43.7 ± 37.9** (n=3) |
| cap5 | horo | 1e-3 | 1 | 58.8% | pending | pending | 58.8 (n=1) |
| cap5f | horo | 3e-4 | 0.25 | 66.6% | 54.4% | 54.7% | **58.6 ± 7.0** (n=3) |
| cap5f | horo | 3e-4 | 0.25 (unnormalized) | 60.8% | 33.0% | 63.5% | **52.4 ± 16.9** (n=3) |
| cap5f | horo | 3e-4 | 1 | 59.5% | 49.9% | 56.1% | **55.1 ± 4.9** (n=3) |
| cap5f | horo | 5e-4 | 1 | 50.7% | 50.8% | 43.0% | **48.2 ± 4.5** (n=3) |
| cat | horo | 3e-4 | 1 | 63.6% | 49.4% | 61.5% | **58.1 ± 7.7** (n=3) |
| cat | horo | 5e-4 | 1 | 55.0% | pending | pending | 55.0 (n=1) |
| cat | horo | 1e-3 | 1 | 60.2% | pending | pending | 60.2 (n=1) |
| sphere | horo | 3e-4 | L=2 | 67.3% | 57.3% | 68.3% | **64.3 ± 6.1** (n=3) |
| sphere | horo | 3e-4 | L=4 | 68.0% | 57.6% | 62.0% | **62.5 ± 5.2** (n=3) |
| sphere | horo | 3e-4 | L=4 c=[2] | 57.7% | 55.6% | 59.8% | **57.7 ± 2.1** (n=3) |
| sphere | horo | 3e-4 | L=6 | 67.2% | 56.3% | 55.6% | **59.7 ± 6.5** (n=3) |
| sphere | horo | 3e-4 | L=8 | 54.5% | 55.2% | 65.3% | **58.4 ± 6.1** (n=3) |
| sphere | horo | 3e-4 | L=12 | 49.0% | 56.9% | 59.6% | **55.2 ± 5.5** (n=3) |
| sphere | horo | 5e-4 | L=2 | 62.7% | 70.0% | 51.2% | **61.3 ± 9.5** (n=3) |
| sphere | horo | 5e-4 | L=4 | 66.6% | 54.9% | 64.5% | **62.0 ± 6.3** (n=3) |
| sphere | horo | 1e-3 | L=4 | 69.9% | 61.0% | 61.9% | **64.2 ± 4.9** (n=3) |
| none | naive | 3e-4 | 1 | 62.1% | 57.4% | 60.9% | **60.1 ± 2.4** (n=3) |
| none | naive | 5e-4 | 1 | 56.8% | 61.8% | 60.4% | **59.6 ± 2.6** (n=3) |
| none | naive | 1e-3 | 1 | 53.9% | 55.0% | 49.5% | **52.8 ± 2.9** (n=3) |
| cap5 | naive | 3e-4 | 1 | 56.9% | 44.5% | 49.0% | **50.1 ± 6.2** (n=3) |
| cap5f | naive | 3e-4 | 0.25 (unnormalized) | 62.1% | 60.7% | 56.9% | **59.9 ± 2.7** (n=3) |
| cat | naive | 3e-4 | 1 | 46.6% | 61.2% | 58.1% | **55.3 ± 7.7** (n=3) |
| sphere | naive | 3e-4 | L=2 | 58.7% | 65.2% | 54.9% | **59.6 ± 5.2** (n=3) |
| sphere | naive | 3e-4 | L=4 | 64.5% | 61.0% | 56.0% | **60.5 ± 4.2** (n=3) |
| sphere | naive | 3e-4 | L=6 | 54.1% | 62.1% | 61.1% | **59.1 ± 4.4** (n=3) |
| sphere | naive | 3e-4 | L=8 | 57.6% | 62.6% | 64.4% | **61.6 ± 3.5** (n=3) |
| sphere | naive | 5e-4 | L=2 | 62.5% | 62.1% | 64.3% | **63.0 ± 1.2** (n=3) |
| sphere | naive | 5e-4 | L=4 | 62.5% | 66.6% | 64.6% | **64.6 ± 2.1** (n=3) |
| sphere | naive | 5e-4 | L=4 MC M=16 | 62.5% | 68.0% | 65.5% | **65.3 ± 2.8** (n=3) |
| sphere | naive | 5e-4 | L=4 MC M=4 | 53.3% | 65.5% | 58.4% | **59.1 ± 6.2** (n=3) |
| sphere | naive | 1e-3 | L=4 | 55.4% | 60.2% | 53.5% | **56.4 ± 3.4** (n=3) |

### RLF vs none (LR 3e-4, scale 1, 3-seed means)

| fwd | none | cap5 | cat | cap5 − none | cat − none |
|---|---:|---:|---:|---:|---:|
| horo | 61.7 | 52.0 | 58.1 | -9.7 | -3.5 |
| naive | 60.1 | 50.1 | 55.3 | -10.0 | -4.8 |

### sphere vs none (LR 3e-4, 3-seed means; stage 4, the harmonic basis)

| fwd | none | L=2 | L=4 | L=6 | L=2 − none | L=4 − none | L=6 − none |
|---|---:|---:|---:|---:|---:|---:|---:|
| horo | 61.7 | 64.3 | 62.5 | 59.7 | +2.6 | +0.9 | -2.0 |
| naive | 60.1 | 59.6 | 60.5 | 59.1 | -0.5 | +0.4 | -1.0 |

<!-- report:end -->

## Analysis (2026-10-01 05:15 EDT; all 25 planned cells done, 2 follow-up seeds of the LR-5e-4 screen running)

Numbers = exact-match board accuracy on the 2000 validation boards (`eval/results.json`); `±` = std over 3 seeds.

### Main grid (LR 3e-4, HyLa scale 1, 768 features = 256 per H³ factor)

| arm | readout | seeds 1 / 2 / 3 | mean ± std | vs none | seeds below none's worst |
|---|---|---|---:|---:|---|
| none (coordinates → `in_proj`) | horosphere | 61.3 / 62.0 / 61.8 | **61.7 ± 0.4** | — | — |
| cap5 (RLF replaces the coordinates, pin radius 5) | horosphere | 55.0 / 59.9 / 41.2 | 52.0 ± 9.7 | −9.7 | 3 / 3 |
| cat (RLF next to the coordinates) | horosphere | 63.6 / 49.4 / 61.5 | 58.2 ± 7.6 | −3.5 | 1 / 3 |
| pure (paper-verbatim RLF, no cap; control) | horosphere | 0.0 / – / – | 0.0 (n=1) | −61.7 | 1 / 1 |
| none | naive | 62.1 / 57.4 / 60.9 | **60.1 ± 2.4** | — | — |
| cap5 | naive | 56.9 / 44.5 / 49.0 | 50.1 ± 6.3 | −10.0 | 3 / 3 |
| cat | naive | 46.6 / 61.2 / 58.1 | 55.3 ± 7.7 | −4.8 | 1 / 3 |

### Screens (seed 1, horosphere)

| knob | cap5 | cat |
|---|---:|---:|
| LR 3e-4 (main grid) | 55.0 | 63.6 |
| LR 5e-4 | **63.8** (seeds 2-3 running) | 55.0 |
| LR 1e-3 | 58.8 | 60.2 |
| scale 0.5 (LR 3e-4) | 46.2 | – |
| scale 2 (LR 3e-4) | 52.8 | – |

### Reading

1. **The harness reproduces the incumbent.** `none`/horosphere = 61.7 ± 0.4 against 61.1 ± 1.8 (round 2) and
   63.8 ± 3.9 (round 3) of `claude_test_hbfm_sudoku`, with an unusually tight seed spread. `naive` is 1.6 pt behind on
   3 seeds; the earlier "naive costs 9 pt" rested on one seed (55.9).
2. **The paper-verbatim map cannot do Sudoku (0.0 %).** The prompt digits are pinned at the boundary (dimensionless
   radial ≈ 35 after the polar readback cap), where every HyLa feature is ≈ 0 (the e^{(d−1)P/2} envelope is a
   square-root Poisson kernel), so the trunk never sees the clues. Measured before the runs: feature norm 0.70 for
   s ≤ 3, 0.02 at s = 8, 0 at the pin. The `cap5` and `cat` arms are the two ways around this.
3. **RLF replacing the coordinates loses in every seed** under both readouts (cap5: −9.7 / −10.0 pt on average, worst
   seed −20.6), and **RLF next to the coordinates is a tie at best** (cat: 63.6 / 61.5 vs 61.7 under horosphere; one
   seed at 49.4; under naive 1 of 3 seeds above the baseline). Neither arm meets the pre-registered bar (+7 pt in all
   three seeds); the sign is the opposite in 10 of 12 RLF cells of the main grid.
4. **RLF makes training seed-fragile.** Every RLF cell group has one seed collapsing to 41–49 %, giving seed stds of
   6–10 pt against 0.4 / 2.4 for the baseline. Two plausible sources, not separated here: the random basis itself
   changes with the seed (a 256-feature estimate of the kernel per factor has "basis luck"), and the heavy-tailed
   features (a few large values per token wherever a boundary point falls near the state's direction) make the input
   scale non-stationary across heat times, which the fixed-LR AdamW recipe tuned for the coordinate input does not
   absorb. The LR / scale screens do not rescue it: scale 0.5 / 2 bracket scale 1 (46.2 / 52.8 vs 55.0), LR 1e-3
   gives 58.8 / 60.2, and the one read above the baseline (cap5 at LR 5e-4, 63.8) is a single seed of an arm whose
   seed spread is 10 pt — its seeds 2-3 are running and are reported below when done.
5. **Why no gain is the expected outcome here.** The DiT already reads the 9 Poincaré coordinates exactly through a
   linear layer, and the horosphere readout adds the exact Busemann geometry at the output; HyLa replaces an exact
   input with a random kernel expansion of it, trading nothing the model lacks for estimation noise. The paper's gains
   come from a *linear* graph model on *learned* embeddings held near the origin, where the kernel view supplies the
   nonlinearity; a 28M-parameter transformer on bridge states that reach the boundary is the opposite regime.

### Follow-up: cap5 / horosphere at LR 5e-4, seeds 2-3 (2026-10-01 05:45 EDT)

| seed 1 | seed 2 | seed 3 | mean ± std |
|---:|---:|---:|---:|
| 63.8 | **0.0** | 67.3 | 43.7 ± 37.9 |

Two of three seeds beat the baseline (+2.1 / +5.6 pt over 61.7, the two best RLF cells of the study) and the third
collapses to 0 / 2000 boards. The collapsed cell is not a numerical failure: no NaN/inf in its log, all checkpoint
tensors finite, `in_proj` weights of ordinary scale (mean |w| 0.057), and its boards are plausible-looking but wrong
digits — an optimization failure to a poor solution. So the higher LR sharpens both tails of the same seed fragility
seen at LR 3e-4 (41.2 / 44.5 / 49.4 collapses); it does not make RLF "consistently better" (1 of 3 seeds below the
baseline's worst by 61 pt).

**Answer to the task:** random Laplacian features do not improve HBFM on hard Sudoku consistently. Replacing the
coordinates costs ~10 points on average at the incumbent LR and makes training seed-fragile under every LR / scale
tried (one collapsed seed per cell group, including a 0 % seed at LR 5e-4 whose sibling seeds are the best cells of the
study at 63.8 / 67.3); adding the features next to the coordinates is a noisy tie. The incumbent (coordinates,
horosphere, LR 3e-4) stands at 61.7 ± 0.4 on this harness. If RLF were to be pursued, the lever is the instability
(e.g. a per-token normalization of the heavy-tailed features or a lower LR for `in_proj`), not the LR or the kernel
scale.

## Stage 2 — HyLa hyperparameters per the reference code (2026-10-02 00:05 EDT, 8 / 12 cells)

The reference implementation (github.com/ydtydr/HyLa, `hype/models.py`) halves λ in the angle and tunes `lambda_scale`
to 0.01 / 0.1 / 0.5 (our `hyla_scale` 0.005 / 0.05 / 0.25; stage 1 used 1.0), and its features carry no 1/√D
(`hyla_normalize=false`). Verified equal to our map by `test_hyla_matches_the_reference_implementation`. Cells: cap5,
horosphere, LR 3e-4, 2 seeds each.

| hyla_scale | normalized (paper φ) | unnormalized (reference code) |
|---|---|---|
| 0.005 | 47.9 / 59.5 → 53.7 | 59.0 / 57.9 → **58.5** |
| 0.05 | 49.5 / 62.4 → 56.0 | 56.4 / 59.3 → 57.9 |
| 0.25 | 68.0 / 59.7 → **63.9 ± 5.9** | 63.9 / 61.6 → **62.8 ± 1.6** |
| 1 (stage 1, 3 seeds) | 55.0 / 59.9 / 41.2 → 52.0 | – |
| baseline (none) | 61.3 / 62.0 / 61.8 → **61.7** | |

- The reference-code regime helps: the low-frequency, unnormalized features train stably (seed spread ≤ 3 pt instead
  of 10–19) and reach 58–59 %, but still 3 pt below the coordinate baseline; the normalized variants keep one weak seed
  each (47.9, 49.5). Scale 0.005 vs 0.05 is a tie.
- **Scale 0.25 is the best region** (stage 2 complete, 03:20 EDT): unnormalized 63.9 / 61.6 = 62.8 ± 1.6, both seeds
  above the baseline mean (61.7 ± 0.4) — the first HyLa configuration to match or beat the coordinates on every seed —
  and normalized 68.0 / 59.7 = 63.9 ± 5.9 (the best single cell of the study, with the pre-fix seed spread). The gain
  (+1.1 pt) is inside the pre-registered tie band (< 7 pt); the post-fix `cap5f` re-runs at scale 0.25 (normalized and
  unnormalized, 3 seeds each, plus the naive readout) decide whether it is real.

## Stage 3 — the instability is a gradient blow-up through the pinned prompts (2026-10-02 02:00 EDT)

Why does `cap5`, scale 1, LR 5e-4 give 63.8 / **0.0** / 67.3? Diagnosed on the collapsed checkpoint (seed 2) against its
healthy sibling (seed 1), on one random sudoku-shaped batch (B = 64, 35 % prompt pins), gradient norms per module
(`logs/` of the scratch diagnostic; scripts `diag_collapse*.py`):

| embedding-table gradient norm | collapsed seed 2 | healthy seed 1 |
|---|---:|---:|
| full batch (prompt pins at the cap radius 5) | **1638** | 19.3 |
| same batch without prompt pins (all positions on the bridge) | 22.9 | 3.1 |
| pins, cap lowered to 2 | 56 | 7.3 |
| pins, HyLa path detached (Busemann readout only) | 1.05 | 0.65 |

- No NaN/inf anywhere; the trunk's gradients are ordinary (blocks 29, in_proj 34). 99.9 % of the embedding gradient
  comes from the HyLa features of the **pinned prompt digits**, through the feature map. The pinned state's direction
  is the digit's own embedding direction, and the Poisson envelope's sensitivity to that direction grows like
  e^{2P}·sinh(s) ≈ 74·e^{2P} at the cap, so once Adam drifts a digit direction toward one of the random boundary
  points the feature spikes (per-token max feature 0.79 vs 0.23 in the healthy run), the embedding gradient explodes,
  the global grad-clip (1.0) scales every other gradient down by ~1/1600 and the trunk stops learning while the
  embeddings keep moving: a self-reinforcing collapse. LR 5e-4 only makes the first drift more likely; the same
  mechanism is behind the single collapsed seed in every scale-1 / normalized cell group of stages 1–2.
- **Fix (2 lines, `models/hyperbolic_dit.py::HyLaFeatures.forward`):** past the cap the map is evaluated at a frozen pin
  — `torch.where(pinned, us.detach(), us)` — so a pinned prompt stays visible to the trunk but no longer backpropagates
  into the embedding through the envelope; bridge states inside the cap keep their full gradient (test
  `test_hyla_radius_cap_stops_the_gradient_at_the_pin`). The baseline (coordinates) never had this path: its prompt
  input is linear in the embedding direction.
- Re-runs after the fix, arm `cap5f` (same knobs as `cap5`, new code): the unstable config (scale 1, LR 5e-4), the
  stage-1 main config (scale 1, LR 3e-4) and the stage-2 best (scale 0.25, normalized: 68.0 on seed 1), 3 seeds each.

### Post-fix results (`cap5f`; complete, 2026-10-02 06:45 EDT, 15 cells)

| config | readout | pre-fix seeds 1 / 2 / 3 → mean | post-fix seeds 1 / 2 / 3 → mean | baseline (none) |
|---|---|---|---|---:|
| scale 1, LR 3e-4 | horosphere | 55.0 / 59.9 / 41.2 → 52.0 ± 9.7 | 59.5 / 49.9 / 56.1 → 55.2 ± 4.8 | 61.7 ± 0.4 |
| scale 1, LR 5e-4 | horosphere | 63.8 / 0.0 / 67.3 → 43.7 ± 37.9 | 50.7 / 50.8 / 43.0 → 48.2 ± 4.5 | 61.7 ± 0.4 |
| scale 0.25, normalized | horosphere | 68.0 / 59.7 → 63.9 ± 5.9 | 66.6 / 54.4 / 54.7 → 58.6 ± 7.0 | 61.7 ± 0.4 |
| scale 0.25, unnormalized | horosphere | 63.9 / 61.6 → 62.8 ± 1.6 | 60.8 / **33.0** / 63.5 → 52.4 ± 16.9 | 61.7 ± 0.4 |
| scale 0.25, unnormalized | naive | – | 62.1 / 60.7 / 56.9 → **59.9 ± 2.7** | 60.1 ± 2.4 |

- The stop-gradient removes the embedding-gradient blow-up (post-fix embedding gradients are 1–4 on the same batch in
  every cell, including the 33.0 one — `diag_collapse4.py`) and with it the 0 % and 41 % seeds of the scale-1 groups,
  but it also removes the signal that made the pre-fix "lucky" cells (63.8 / 67.3 / 68.0): the prompt-pin gradient
  shaped the digit embeddings so that their prompt features were readable, and without it the scale-1 configs settle
  6–13 pt below the baseline. Seed-level failures still occur post-fix (33.0 at scale 0.25 unnormalized, horosphere)
  with small gradients everywhere: an ordinary optimization failure of the trunk on the large unnormalized inputs
  (per-token max feature ≈ 12 at the pins), not the gradient mechanism — the remaining fragility is intrinsic to
  training a transformer on heavy-tailed random features rather than a bug.
- The one clean result is **scale 0.25 unnormalized with the naive readout: 59.9 ± 2.7 vs the naive baseline's
  60.1 ± 2.4**, three seeds within the baseline's spread and no collapse — a tie, the first HyLa configuration that
  neither loses nor destabilizes. No configuration beats the coordinate baseline under the pre-registered rule, and the
  two pre-fix cells above it (68.0 normalized, 63.9 / 61.6 unnormalized) did not replicate post-fix.

## Final answer for Sudoku (2026-10-02 06:45 EDT; 54 cells)

Random Laplacian features do not improve HBFM on hard Sudoku. Paper-verbatim HyLa scores 0 % (the prompt pin zeroes
the features); the paper's scale (λ ~ N(0, 1)) is unstable and 10 pt worse; the reference code's regime (λ scale
0.005–0.25, unnormalized) is stable but ties the baseline at best (59.9 ± 2.7 vs 60.1 ± 2.4 naive; 58–63 vs 61.7
horosphere); concatenating the features to the coordinates is a noisy tie; the one gradient bug found (prompt-pin
gradient through the Poisson envelope) explains the catastrophic seeds and is fixed, but fixing it does not turn HyLa
into a gain. The incumbent (coordinates → `in_proj`, horosphere, LR 3e-4) stands at 61.7 ± 0.4.

## Stage 4 — the `sphere` map (hyperbolic-harmonic basis): COMPLETE, 18 / 18 cells (2026-10-03 06:15 EDT)

`model.sphere_degree = L` (EXPERIMENT.md "Stage 4"): per H³ factor the (L+1)² bounded harmonics f_l(s) Y_lm(u) plus 6
radial channels tanh(s/c), replacing the 9 coordinates (L = 1 ≙ the coordinates) — no HyLa, no cap. Board accuracy,
LR 3e-4, 20k steps, 2000 boards:

| fwd | L | seed 1 | seed 2 | seed 3 | mean ± std | `none` (3 seeds) | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| horo | 2 | 67.3 | 57.3 | 68.3 | 64.3 ± 6.1 | 61.7 ± 0.4 (61.3 / 62.0 / 61.8) | +2.6 |
| horo | 4 | 68.0 | 57.6 | 62.0 | 62.5 ± 5.2 | | +0.9 |
| horo | 6 | 67.2 | 56.3 | 55.6 | 59.7 ± 6.5 | | −2.0 |
| naive | 2 | 58.7 | 65.2 | 54.9 | 59.6 ± 5.2 | 60.1 ± 2.4 (62.1 / 57.4 / 60.9) | −0.5 |
| naive | 4 | 64.5 | 61.0 | 56.0 | 60.5 ± 4.2 | | +0.4 |
| naive | 6 | 54.1 | 62.1 | 61.1 | 59.1 ± 4.4 | | −1.0 |

- **It recovers the baseline, where uncapped HyLa scored 0.0 %** — the design note's sudoku check passes: the pinned
  prompt is seen through its direction (features = Y_lm(u) at the pin; bounded sensitivity 23 into the embedding, see
  EXPERIMENT.md), every cell lands inside the normal range (min 54.1, max 68.3), no collapse (the HyLa study had 0.0 /
  33.0 seeds), no NaN, no cap or stop-gradient needed.
- **No improvement on Sudoku**: every 3-seed mean is within ±2.6 pt of `none`, i.e. inside the sphere arm's own
  seed spread (σ 4–6). The seed effect is strongly shared across L: under horosphere seeds 1 / 2 give 67–68 / 56–58 at
  all three L, so the seed-shared stream (data order, sampled heat times, bridge noise) moves the result by 10 pt while L
  moves it by < 3. The coordinate trunk is far less seed-sensitive (σ 0.4 horosphere); the sphere trunk's sensitivity is
  benign (no failures) but real, and it would take ≳ 10 seeds to resolve a few-point effect.
- **L**: horosphere means fall 64.3 → 62.5 → 59.7 from L = 2 to 6 (−4.6, about 1σ of a 3-seed mean); naive is flat.
  Nothing favours higher degree on 12 words, where 3 factors separate the vocabulary at L = 1 already; the design
  note's question (how large L must be to resolve a large vocabulary on 16–32 factors) is a TinyStories question.

**Verdict for Sudoku:** the harmonic basis is a *safe* replacement of the coordinates (bounded, pin-proof, no collapse)
but not a *better* one here — a tie at every L under both readouts. Gate for TinyStories (≥ baseline − 2 pt, no collapsed
seed): passed at L = 2 and 4 under both readouts; `sphere-L4` × naive × 3 seeds is training in
`claude_test_hbfm_rlf_tinystories_256` (jobs 921544-921546, started 04:38–05:14 EDT).

**TinyStories follow-up (2026-10-04):** `sphere-L4` × naive × 3 seeds = 16.09 ± 1.09 / 13.53 ± 0.34 GenPPL vs the
coordinates 15.58 ± 0.67 / 13.25 ± 0.56 — a tie there too (and 16 % / 5 % better than HyLa's best); see
`claude_test_hbfm_rlf_tinystories_256/RESULTS.md` "Verdict for TinyStories".

## Stage 5 — tuning the sphere map on Sudoku: COMPLETE (2026-10-05 11:00 EDT, 18 cells, 3 seeds each)

| fwd | config | seeds 1 / 2 / 3 | mean ± std | Δ vs `none` |
|---|---|---|---:|---:|
| horo | L = 2 (stage 4) | 67.3 / 57.3 / 68.3 | 64.3 ± 6.1 | +2.6 |
| horo | L = 4 (stage 4) | 68.0 / 57.6 / 62.0 | 62.5 ± 5.2 | +0.8 |
| horo | L = 6 (stage 4) | 67.2 / 56.3 / 55.6 | 59.7 ± 6.5 | −2.0 |
| horo | **L = 8** | 54.5 / 55.2 / 65.3 | 58.4 ± 6.1 | −3.4 |
| horo | **L = 12** | 49.0 / 56.9 / 59.6 | 55.2 ± 5.5 | −6.5 |
| horo | **L = 4, radial channel c = [2] only** | 57.7 / 55.6 / 59.8 | 57.7 ± 2.1 | −4.0 |
| horo | **L = 4, LR 5e-4** | 66.6 / 54.9 / 64.5 | 62.0 ± 6.3 | +0.3 |
| naive | L = 2 / 4 / 6 (stage 4) | | 59.6 / 60.5 / 59.1 | −0.5 / +0.4 / −1.0 |
| naive | **L = 8** | 57.6 / 62.6 / 64.4 | 61.6 ± 3.5 | +1.5 |
| naive | **L = 4, LR 5e-4** | 62.5 / 66.6 / 64.6 | **64.6 ± 2.1** | **+4.5** |

(`none`: horosphere 61.7 ± 0.4, naive 60.1 ± 2.4, both at LR 3e-4.)

- **Degree:** under horosphere the mean falls monotonically with L, 64.3 → 62.5 → 59.7 → 58.4 → 55.2 from L = 2 to 12
  (−9 pt, about 3σ of a 3-seed mean): on 12 words the extra harmonics are pure nuisance inputs that the trunk must learn
  to ignore. Under naive the degree is flat (59–62).
- **Radial channels:** a single tanh(s/2) channel (the coordinates' own radius) instead of the 6-channel ladder loses 5 pt
  (57.7 vs 62.5) — the ladder's resolution of larger radii (the end of the bridge, the pin) is used.
- **LR:** the one knob that moved. At L = 4 and LR 5e-4 the naive readout scores 64.6 ± 2.1, every seed above `none`'s
  mean and above its seed-paired baseline (+0.4 / +9.2 / +3.7); horosphere ties (62.0 ± 6.3). Whether this is a
  *feature* gain or an *LR* gain the coordinates would share is exactly what stage 6 asks: `none` was never LR-screened
  under the naive readout in this harness (horosphere LR 5e-4 / 1e-3 lost in `claude_test_hbfm_sudoku`).

Stage 6 (submitted 11:15 EDT, 21 cells): `none` at LR 5e-4 (both readouts) and 1e-3 (naive); sphere L = 4 at LR 1e-3
(both); sphere L = 2 at LR 5e-4 (both). Tags `hyla-none_…_lr-5e-4_…`, `…_lr-1e-3_…`, `sphere-L2_…_lr-5e-4_…`.

### Stage 6, first control (2026-10-05 13:00 EDT): the LR gain is specific to the sphere map

| naive readout | LR 3e-4 | LR 5e-4 |
|---|---:|---:|
| `none` (coordinates) | 60.1 ± 2.4 (62.1 / 57.4 / 60.9) | 59.6 ± 2.6 (56.8 / 61.8 / 60.4) |
| sphere L = 4 | 60.5 ± 4.2 | **64.6 ± 2.1** (62.5 / 66.6 / 64.6) |

Seed-paired at LR 5e-4, sphere beats the coordinates in all three seeds (+5.7 / +4.8 / +4.2; mean +5.0, Welch t ≈ 2.6
on 3 + 3 seeds). The coordinate trunk is indifferent to LR 3e-4 → 5e-4; the sphere trunk (10× more inputs) needs it.
This is the first configuration in the whole RLF / sphere study that beats the best coordinate run. Pending: `none` at
LR 1e-3 (naive), sphere at LR 1e-3, sphere L = 2 at 5e-4, `none` horosphere at 5e-4, and the TinyStories control.
- 2026-10-05 14:50 EDT: `none` horosphere at LR 5e-4 = 66.8 / 57.5 / 56.1 = 60.1 ± 5.8 (vs 61.7 ± 0.4 at 3e-4): the higher LR
  makes the coordinate trunk seed-noisy without a gain; sphere L = 4 horosphere at 5e-4 = 62.0 ± 6.3 → tie under horosphere.
  The sphere win stays naive-readout-only so far.
- 2026-10-05 17:05 EDT: LR 1e-3, naive: `none` 53.9 / 55.0 / 49.5 = 52.8 ± 2.9; sphere L = 4 55.4 / 60.2 / 53.5 = 56.4 ± 3.5. Both
  degrade past 5e-4 (sphere still +3.6 ahead, seed-paired +1.5 / +5.2 / +4.0) → LR 5e-4 is the sphere optimum (64.6). Sphere
  horosphere at 1e-3 seed 1 = 69.9 (seeds 2-3 running).
- 2026-10-05 19:00 EDT: sphere L = 2 at LR 5e-4, naive = 62.5 / 62.1 / 64.3 = 63.0 ± 1.2 (seed-paired vs `none` @ 5e-4: +5.7 / +0.3 /
  +3.9) — the naive-readout gain at 5e-4 reproduces at a second degree. Sphere L = 4 horosphere at LR 1e-3 = 69.9 / 61.0 / 61.9
  = 64.3 ± 4.9 vs `none` horosphere 61.7 ± 0.4 (3e-4) / 60.1 ± 5.8 (5e-4): +2.6, within noise. L = 2 horosphere @ 5e-4 seed 3 pending
  (62.7 / 70.0 so far).
- 2026-10-05 20:45 EDT: sphere L = 2 horosphere at LR 5e-4 = 62.7 / 70.0 / 51.2 = 61.3 ± 9.4 (tie, very noisy). Stage 6 sudoku
  COMPLETE: the only significant sphere win is the naive readout at LR 5e-4 (L 4: 64.6 ± 2.1, L 2: 63.0 ± 1.2 vs ≤ 60.1).

## Stage 7 — Monte-Carlo (Poisson-kernel IS) sphere features: COMPLETE (2026-10-18, 6 cells, 3 seeds each)

Anchor: sphere L = 4, naive readout, LR 5e-4 with the closed-form features (stage 5). Same config, plus
`model.sphere_mc_samples = M` in training. `eval` = sampled with the same M (headline); `eval_exact` = the same
checkpoint sampled with the closed form. Seed-paired Δ is against the anchor seed.

| features | eval (train = test) | seed-paired Δ | eval_exact (closed form at sampling) | seed-paired Δ |
|---|---:|---:|---:|---:|
| closed form (anchor) | **64.6 ± 2.1** (62.5 / 66.6 / 64.6) | — | — | — |
| MC, M = 16 | **65.3 ± 2.8** (62.5 / 68.0 / 65.5) | 0.0 / +1.4 / +0.9 (mean +0.8) | 64.9 ± 2.0 (62.6 / 66.0 / 66.1) | +0.2 / −0.7 / +1.5 (+0.3) |
| MC, M = 4 | 59.1 ± 6.2 (53.3 / 65.5 / 58.4) | −9.2 / −1.1 / −6.3 (mean −5.5) | 57.1 ± 5.8 (51.7 / 63.2 / 56.3) | −10.8 / −3.5 / −8.4 (−7.5) |
| coordinates (`none`), same LR | 59.6 ± 2.6 | | | |

Verdict under the stage-7 rule:
- **M = 16: tie** (+0.8, below the +3 bar). The noise neither helps nor hurts; the trained network barely cares which
  features it is sampled with (eval vs eval_exact within ±2 per seed, mean +0.5 for MC).
- **M = 4: hurts** (−5.5, every seed below its anchor; the seed spread triples to 6.2). With 4 angles the per-feature
  noise is ≈ 0.55 at s ≲ 2, where most training states are, and the trunk learns a worse map.
- With M = 4 the checkpoint scores +1.6 / +2.4 / +2.1 *higher* with the noisy features at sampling than with the exact
  ones: a network trained on noisy inputs prefers the same noise at test (train/test match), not a sign that MC helps.

So MC over the boundary angle with the Poisson kernel as the proposal is safe at M = 16 and harmful at M = 4, and at
best recovers the closed form. Expected: the estimator is unbiased for the closed form, which was already bounded and
exact at the pin, so the MC mode can only add noise. It does not beat the closed form on Sudoku.
