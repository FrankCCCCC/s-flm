# claude_test_hbfm_rlf_sudoku — HBFM + random Laplacian features (RLF / HyLa) on hard Sudoku

Spec: `setup.md` ("copy `claude_test_hbfm_sudoku/setup.md`, pick a good config, re-train with RLF and see whether RLF
performs consistently better than without"). Sweep: `sweep.py` (orchestration only; calls
`scripts/train/sudoku/hbfm.sh` + `scripts/sample/sudoku/hbfm.sh`). Report: `report.py` → `RESULTS.md`.

## The feature map under test

`models/hyperbolic_dit.py::HyLaFeatures` (Yu & De Sa, *Random Laplacian Features for Learning with Hyperbolic Space*,
ICLR 2023), enabled by `model.hyla_dim`. Per product factor H^3 it draws `hyla_dim / 3` fixed boundary points
ω ~ Unif(S²), eigenvalue parameters λ ~ N(0, `hyla_scale`²) and phases b ~ U[0, 2π), and maps the factor's
dimensionless polar state (s = ρ/R, u) to `exp(P) cos(λP + b) / sqrt(hyla_dim)` with `P = −log(cosh s − sinh s ⟨u, ω⟩)`
the signed distance from the origin to the horocycle through the state at ω (the paper's eq. 1). Each feature is a
Laplace–Beltrami eigenfunction of its factor and the concatenation's inner product estimates the mean of the factors'
isometry-invariant kernels (verified by `tests/test_hbfm.py::test_hyla_*`: Poincaré-formula match, Monte-Carlo kernel
= ½·sin(λd)/(λ sinh d) on H³). The DiT then consumes the features through `in_proj` instead of the 9 Poincaré
coordinates.

**Boundary caveat that shapes the design.** The eigenfunctions carry the envelope e^{P}, a square-root Poisson kernel:
sparse near the boundary. Measured with 16 features per factor the feature norm is 0.70 (= the kernel's k(x,x) = ½) up
to s ≈ 3, 0.55 at s = 4, 0.18 at s = 6, 0.02 at s = 8 and exactly 0 at the sudoku prompt pin (`_clean_state` →
s ≈ 34.7 after the polar readback cap). With the map as published the trunk therefore cannot see the clue digits at
all, so the paper-verbatim arm is expected to fail on Sudoku. Two remedies are compared:

| arm | knobs | meaning |
|---|---|---|
| `none` | `HYLA_DIM=null` | the incumbent: Poincaré coordinates → `in_proj` (9 → 512) |
| `pure` | `HYLA_DIM=768` | the paper's map verbatim, 256 features per factor, scale 1 — control, one seed |
| `cap5` | `HYLA_DIM=768 HYLA_RADIUS_CAP=5` | the map evaluated at min(s, 5): a finite pin radius (`model.hyla_radius_cap`). Never binds on the trained states (rate 3 → s < ~2); at s = 5 the direction still identifies the word (Busemann margin ≈ 13 nats between the target and the nearest of 12 words) |
| `cat` | `HYLA_DIM=768 HYLA_CONCAT_STATE=true` | the map (no cap) concatenated to the coordinates: `in_proj` sees 9 + 768 channels (`model.hyla_concat_state`) — "does adding RLF help" rather than "can RLF replace the coordinates" |

## Hypothesis

H1 (paper transfer): the Euclidean DiT with a kernel feature map of the hyperbolic state learns the Sudoku posterior at
least as well as from raw ball coordinates, i.e. `cap5` ≥ `none` for both readouts; the horosphere readout already adds
the exact geometry at the output, so any RLF gain should be larger under `naive` (where the trunk must learn the
Bayes posterior from the input alone; `naive` costs ~8 pt vs horosphere in the previous study).
H2: `pure` fails (≪ `none`) because the prompt is invisible — the control that motivates the cap.
H0 (null, likely given the 5.4-pt per-seed spread measured before): no arm separates from `none` by more than the seed
noise; then "consistently better" is not supported.

## Design — the anchor is the previous study's incumbent (`claude_test_hbfm_sudoku`, round 3)

| | |
|---|---|
| data | Sudoku hard (30 clues), 48k train / 2k val, data seed 42 |
| model | tiny-hyperbolic-dit 512/8/8 (~28.6M), `model.embed_dim = 9` (lifted by `in_proj`), init ngpt |
| geometry | `prod_factor_dim = [3,3,3]`, `prod_factor_gaussian_curvature = [−0.5]*3` → (H³_{−0.5})³ |
| objective | denoising CE, `forward_type ∈ {horosphere, naive}` (setup.md lists both), `readout_precision = float32` |
| proposal | `time_conversion_mode = exp`, physical `time_exp_rate = 3` (unit rate 6; setup.md's 0.01 gave 0–1 % boards in the previous study, so the "good config" is its tuned rate), `time_range_upper_bound = 1.0` (unused by exp), noise `log-linear` |
| training | 20k steps, batch 256 (1 GPU × 256), bf16, EMA 0.9999, AdamW (LR 3e-4 main grid; 5e-4 / 1e-3 screened), wd 0, betas 0.9/0.999, eps 1e-8, grad-clip 1.0 |
| RLF | `hyla_dim 768` (256 / factor), `hyla_scale 1` (screened 0.5 / 2), arms as above; the random draws are seeded by `seed` (buffers in the checkpoint) |
| eval | `mode=sudoku_eval` on the 2000 validation boards: exact velocity, `top_k_velocity = −1`, 180 steps, physical `t_max = 6` (unit 3, the incumbent's trained horizon), greedy last step; metric = exact-match board accuracy |
| outputs | `outputs/claude_test_hbfm_rlf_sudoku/hyla-{arm}_fwd-{horo,naive}_lr-{lr}_sc-{scale}_seed-{s}/{checkpoints, eval/results.json}` |

Cells (25): main grid {none, cap5, cat} × {horo, naive} × seeds {1, 2, 3} = 18, plus controls at seed 1 / horosphere:
`pure` (1), LR 5e-4 and 1e-3 for `cap5` and `cat` (4), scale 0.5 and 2 for `cap5` (2). The `none` cells are re-trained
in this harness (same recipe as the previous study's `lr-3e-4_ada-0_k-0.5_seed-*_rate-3` = 61.1 ± 1.8 and its round-3
reproduction 63.8 ± 3.9; naive seed 1 = 55.9) so that the comparison is paired on GPU type and code version.

## GPU allocation and expected wall clock

One GPU per cell on `thickstun,desa` minus the 2080 Ti node (`--exclude=desa-compute-01`; the 11 GB card OOMs at
batch 256). Measured before: 1 h 50 min training + 3 min eval on an A5000 (RTX 6000 Ada / A6000 faster). 25 cells ≈
50 GPU-hours; the cluster is currently saturated by the user's own `init_test_*` jobs, so the wall clock is set by
the queue, ~1–2 days. Submission order = priority: `none`/horo, `cap5`/horo, `cat`/horo, the naive arms, then the controls.

## Decision rules (fixed before the runs)

- Per-seed σ ≈ 4–5 pt (previous study), so a 3-seed mean has ±3 pt standard error. "RLF performs better consistently"
  = the arm beats `none` in all three seeds AND by more than ~2 SE of the difference (≈ 7 pt) for a given readout.
  Anything smaller is reported as a tie with the numbers.
- LR / scale screens are single-seed: a screen cell is followed up with seeds 2-3 only if it beats its 3e-4 / scale-1
  counterpart by more than 8 pt (≈ 1.5 σ); otherwise the main-grid LR stands.
- `pure` is one seed: it is a mechanism check, not a competitor.

## Stage 4 — the `sphere` map: hyperbolic-harmonic spherical-harmonic features (2026-10-02 19:30 EDT)

The user's design note (after the advisor's verdict that HyLa blows up / vanishes at the boundary and cannot be used on
unbounded states, normalization or not, so the basis has to change): replace the point-sampled Poisson-kernel powers by
the *Poisson integrals* of the spherical harmonics. `models/hyperbolic_dit.py::HarmonicFeatures`, enabled by
`model.sphere_degree = L` (exclusive with `hyla_dim`; knobs `SPHERE_DEGREE` / `SPHERE_RADIAL_SCALES` in the 7 HBFM
scripts). Per H³ factor, from the dimensionless polar state (s = ρ/R, u ∈ S²):

- h_lm(z) = f_l(s) · Y_lm(u), l = 0…L, m = −l…l, (L+1)² features, with f_l(s) = r^l ₂F₁(l, −½; l+3/2; r²) / ₂F₁(l, −½; l+3/2; 1),
  r = tanh(s/2): the radial solution of the H³ Laplace equation regular at the origin, normalized to 1 at the boundary.
  Harmonic (Δh = 0), bounded (|h_lm| ≤ √(2l+1)), and h_lm → Y_lm(u) as s → ∞, so the pinned prompt (s = 34.66) is seen
  through its direction with no cap; f_l rises from 0 at the origin to 1 past s ≈ ln(2l) (f₁(1) = 0.59, f₄(2) = 0.52,
  f₈(3) = 0.67): a low-pass filter on the direction whose cutoff degree grows with the radius. The plain coordinates
  z = R tanh(s/2) u are the L = 1 case. f_l is tabulated at construction (scipy ₂F₁, step 1/256 up to s = 36) and linearly
  interpolated; Y_lm are the real harmonics, orthonormal under the uniform probability measure (Y₀₀ = 1, Y_1m = √3 u).
- plus the radial channels tanh(s/c), c ∈ {1, 2, 4, 8, 16, 32} (f₀ = 1 and every f_l saturates by s ≈ 3, while the trunk
  has no time conditioning): L = 4 → 3 × (25 + 6) = 93 inputs to `in_proj` (sudoku), vs 9 coordinates / 768 HyLa.

Tests (`tests/test_hbfm.py`, sphere block): f_l solves the radial Laplace equation on the table (central differences,
residual < 1e-4) and matches the l = 1 closed form and the design note's table; Y_lm orthonormal by quadrature; features
bounded and finite with finite gradients on s ∈ [0, 34.66] incl. the exact pin and the origin; h_lm → Y_lm(u); GPU
end-to-end with the pinned clean state. Nothing random → no new checkpoint keys.

Cells (18, `sweep.py` arm `sphere`, tags `sphere-L{L}_fwd-{horo,naive}_lr-3e-4_seed-{s}`): L ∈ {2, 4, 6} × both readouts ×
seeds 1-3 at the anchor recipe (LR 3e-4, 20k steps). Baseline = the stage-1 `none` cells (61.7 ± 0.4 horo / 60.1 ± 2.4
naive); the no-cap HyLa control = the stage-1 `pure` cell (0.0). Decision rule as before: a 3-seed mean ≥ baseline − 2 pt
with no collapsed seed = "works" → scale to TinyStories (user, 2026-10-02: "test it on sudoku first, if works well, then
scale to tinystories"); > +7 pt = consistently better.

## Status

- 2026-09-30 16:35 EDT: knobs (`hyla_radius_cap`, `hyla_concat_state`), sweep, report written; the `cap5` and `cat`
  job bodies smoke-tested on an A5000 (20 steps + the full 2000-board eval, `logs/` of the scratch run: both wrote
  `eval/results.json`); `tests/test_hbfm.py -k hyla` 5/5.
- 2026-09-30 17:00 EDT: 25 cells submitted, SLURM jobs 718945-718969 in the priority order above (1 GPU each,
  `thickstun,desa` minus the 2080 Ti). The cluster is saturated by the user's `init_test_*` jobs, so most cells wait
  on `Priority`. The sweep runs the shared scripts from a text snapshot (`bash -c "$(cat script)" script`): bash
  reads a script file incrementally, and a live edit of `scripts/train/*/hbfm.sh` during a cell's python call was
  observed to corrupt the running script (smoke test, exit 127).
- 2026-10-01 02:15 EDT: 14 / 25 cells done; interim analysis in RESULTS.md — `cap5` loses in every seed (52.0 ± 9.7 vs
  61.7 ± 0.4 horosphere; 50.7 vs 60.1 naive so far), `cat` ties at best (58.2 ± 7.6) with one collapsed seed each.
- 2026-10-01 05:15 EDT: all 25 planned cells done; analysis in RESULTS.md. Outcome: H0/H2 — RLF replacing the
  coordinates loses in 6/6 seeds (−10 pt), RLF next to the coordinates is a noisy tie, the paper-verbatim map scores
  0 %. Follow-up seeds 2-3 of the LR-5e-4 screen (jobs 765072-765073) running under the screen rule.
- 2026-10-01 05:45 EDT: LR-5e-4 follow-up done (63.8 / 0.0 / 67.3): two best cells of the study and one optimization
  collapse (no NaNs). Sudoku experiment COMPLETE (27 cells); final analysis + answer in RESULTS.md.
- 2026-10-01 21:50 EDT, **stage 2 — HyLa hyperparameters per the reference code.** The reference implementation
  (github.com/ydtydr/HyLa, `hype/models.py`) computes `cos(Lambdas/2 · log PsK + b) · sqrt(PsK)^(d−1)` with tuned
  `lambda_scale` 0.01 / 0.1 / 0.5 and no 1/sqrt(D): in our units `hyla_scale` 0.005 / 0.05 / 0.25 (stage 1 used 1.0,
  i.e. 20–200× higher cosine frequency) and `hyla_normalize=false`. Verified by
  `tests/test_hbfm.py::test_hyla_matches_the_reference_implementation` (our map = the reference formula with
  2× lambda and no 1/sqrt(D)). Screen: cap5 × scale {0.005, 0.05, 0.25} × normalize {1, 0} × horosphere × LR 3e-4 ×
  seeds {1, 2} = 12 cells (jobs 842236–842247, tags `..._sc-{scale}[_nm-0]_seed-{s}`); the best config then gets
  seed 3, the naive readout and LR 5e-4 under the stage-1 rules.
- 2026-10-02 02:00 EDT, **stage 3 — pin-gradient fix.** The seed collapses are a gradient blow-up through the HyLa
  features of the pinned prompts (RESULTS.md "Stage 3"); fixed by a stop-gradient on the direction of states past
  the cap. Re-runs as arm `cap5f` (same knobs, new code): scale 1 at LR 5e-4 and 3e-4, scale 0.25 normalized, 3 seeds.
- 2026-10-02 06:45 EDT: **COMPLETE** (54 cells). Final answer in RESULTS.md: no HyLa configuration beats the
  baseline; best stable = scale 0.25 unnormalized + naive readout, a tie (59.9 ± 2.7 vs 60.1 ± 2.4); the pin-gradient
  bug is fixed but post-fix seed failures (33.0) persist for horosphere + unnormalized features.
- 2026-10-02 19:30 EDT, **stage 4 — `sphere`**: all pending HyLa jobs cancelled on the user's instruction (ch2263
  886614/886616/886618/892100; local paired reads 901776/901777; the two running TinyStories `best` seeds 2-3 cells
  866622/866623 left running). `HarmonicFeatures` implemented + 5 tests; GPU test job 909308 (tests, feature profile,
  20-step sudoku smoke of the `sphere-L4` job body), then the 18 stage-4 cells.
- 2026-10-02 22:35 EDT: test job 909810 (A5000): 22/22 sphere + HyLa + validation tests pass; the `sphere-L4` job body
  smoke-tested (20 steps + full 2000-board eval, `results.json` written). Feature profile ((H³)³, L = 4): per-factor
  harmonic norm rises 1.0 → 5.0 between s = 0 and 6 (max |h| 2.13 = √(2·2+1)·…, bounded), the sensitivity of a linear
  readout to the word's DIRECTION saturates at 23 for every s including the pin (HyLa's grew like e^{2P} sinh s),
  while the sensitivity to the Cartesian state's radius is steep past s ≈ 8 (the tanh(s/c) channels resolving s near
  the boundary) — irrelevant to training, because the bridge's radius is drawn independently of the embedding and the
  pinned prompt's radius is clamped by the readback. 18 stage-4 cells submitted 22:33 EDT, jobs 910214–910232
  (seed-major order; 3 started at once, 15 wait on `Priority`).
- 2026-10-03 02:30 EDT: seed 1 of all six (L, readout) cells done. Horosphere: L = 2 / 4 / 6 → 67.3 / 68.0 / 67.2 (the
  stage-1 `none` seed 1 = 61.3; 3-seed 61.7 ± 0.4) — +6 pt at every L. Naive: 58.7 / 64.5 / 54.1 (`none` 62.1 / 57.4 /
  60.9 = 60.1 ± 2.4) — mixed, inside the naive readout's seed spread. No collapse, no NaN. Seeds 2 running (5 cells,
  ~04:20 EDT), seed 3 queued. TinyStories arm prepared (not submitted) pending the 3-seed means.
- 2026-10-03 04:30 EDT: seed 2 of all six cells done (12 / 18). Horosphere seed 2: L = 2 / 4 / 6 → 57.3 / 57.6 / 56.3
  (`none` seed 2 = 62.0) — the whole seed moved by −10 pt against seed 1's 67.3 / 68.0 / 67.2, all three L together.
  Naive seed 2: 65.2 / 61.0 / 62.1 (`none` 57.4). Two-seed means: horosphere 62.3 / 62.8 / 61.8 vs 61.7, naive 62.0 / 62.8 /
  58.1 vs 60.1 — a tie, no collapse (min 54.1). Reading: on a 12-word vocabulary the sphere map recovers the baseline
  (the design note's sudoku check: "acc 應恢復正常" — it does, where uncapped HyLa scored 0), with no sign of the
  HyLa seed failures; the angular-resolution payoff, if any, needs a large vocabulary. Decision: TinyStories arm
  `sphere-L4` submitted under the naive readout first (3 seeds, jobs 921544-921546, mirroring the stage-3 `best` arm);
  horosphere there only if naive shows a gain. Seed 3 (6 cells running, ~06:00 EDT) completes the sudoku table.
- 2026-10-03 06:15 EDT: **stage 4 COMPLETE** (18 / 18; 72 cells in the project). 3-seed means: horosphere L = 2 / 4 / 6 =
  64.3 ± 6.1 / 62.5 ± 5.2 / 59.7 ± 6.5 vs 61.7 ± 0.4; naive 59.6 ± 5.2 / 60.5 ± 4.2 / 59.1 ± 4.4 vs 60.1 ± 2.4 — a tie at
  every L, no collapse. Final analysis in RESULTS.md "Stage 4". TinyStories `sphere-L4` naive cells running.

## Stage 5 — tuning the sphere map (2026-10-04 07:40 EDT; user: "hyperparameter tuning for the sphere features, try to
make it perform better than the raw Poincaré coordinates")

The map's own knobs are the degree L (`model.sphere_degree`) and the radial channels (`model.sphere_radial_scales`,
tanh(s/c) per c); the knobs that interact with it are the trunk LR (tuned for 48 coordinates, now 10-60× more inputs)
and the geometry (fixed: H³ factors, K −0.5). Sudoku cannot judge angular resolution (12 words; L = 2/4/6 tied) but can
judge optimization stability and the radial channels, so the cells here are the 3-seed safety net for the TinyStories
screens: L = 8 × both readouts, L = 12 × horosphere, LR 5e-4 at L = 4 × both readouts, and a single radial channel
c = [2] (= the coordinates' own |z|/R) at L = 4 × horosphere — 18 cells, jobs 951482-951499, tags
`sphere-L{8,12}_…`, `…_lr-5e-4_…`, `…_rc-2_…`. Decision rule unchanged (3-seed mean vs `none`, no collapsed seed).
- 2026-10-05 11:15 EDT: stage 5 COMPLETE (RESULTS.md "Stage 5"): degree hurts under horosphere (L 2 → 12: 64.3 → 55.2),
  a single radial channel hurts (−5), LR 5e-4 at L = 4 is the only gain (naive 64.6 ± 2.1 vs 60.1 ± 2.4). Stage 6 submitted
  (21 cells): the coordinates at LR 5e-4 / 1e-3 as the control, sphere L = 4 at 1e-3 and L = 2 at 5e-4.

## Stage 7 — Monte-Carlo (Poisson-kernel IS) sphere features (2026-10-18; user: "Train the IS spherical harmonic on
sudoku hard")

`model.sphere_mc_samples = M` (models/hyperbolic_dit.py `HarmonicFeatures`): the harmonics' value is the mean of
Y_lm(w) over M fresh boundary angles w ~ P(z, ·) = (cosh s − sinh s ⟨u, w⟩)^−2 per H³ factor — the Poisson kernel as its
own importance proposal, so every weight is 1 and the estimate is unbiased for the closed form f_l(s) Y_lm(u), bounded
by √(2l+1), and exact at the pin; the gradient is the closed form's (straight-through); fresh draws every forward in
training and sampling. Cost of the change is noise, largest at s ≈ 0.5-2 (≈ 0.27 per feature at M = 16, ∝ 1/√M).
Anchor = the stage-5/6 winner, sphere L = 4, naive readout, LR 5e-4 (64.6 ± 2.1; 62.5 / 66.6 / 64.6), everything else
unchanged. Cells: M ∈ {4, 16} × seeds 1-3 = 6 cells, tags `sphere-L4_fwd-naive_lr-5e-4_mc-{4,16}_seed-{1,2,3}`. Each cell
is sampled twice: `eval/` with the same M (the headline, train = test) and `eval_exact/` with the closed form
(SPHERE_MC_SAMPLES=null), which separates the training effect from the sampling noise.
Decision rule (fixed before the runs; per-seed σ ≈ 2-5 pt): MC **helps** if its 3-seed mean ≥ anchor + 3 pt with ≥ 2 / 3
seed-paired gains; **hurts** if ≤ anchor − 3 pt; otherwise a tie.
- 2026-10-18 10:15 EDT: **stage 7 COMPLETE** (6 / 6; jobs 368713-368718): M = 16 = 65.3 ± 2.8 (tie, +0.8 seed-paired),
  M = 4 = 59.1 ± 6.2 (hurts, −5.5, all seeds below). Analysis in RESULTS.md "Stage 7".
