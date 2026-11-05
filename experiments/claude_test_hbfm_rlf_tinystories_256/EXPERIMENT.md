# claude_test_hbfm_rlf_tinystories_256 — HBFM + random Laplacian features (RLF / HyLa) on TinyStories (seq 256)

Spec: `setup.md` ("copy the settings, pick a good config, re-train with RLF and see whether RLF performs consistently
better than without"; both `desa` and `thickstun`, no 2080 Ti). Sweep: `sweep.py` (orchestration only; calls
`scripts/train/tinystories/hbfm.sh` + `scripts/sample/tinystories/hbfm.sh`). Report: `report.py` → `RESULTS.md`.

## The feature map under test

`models/hyperbolic_dit.py::HyLaFeatures` — see `experiments/claude_test_hbfm_rlf_sudoku/EXPERIMENT.md` for the map, the
kernel tests and the boundary caveat. Here the manifold is (H³_{−0.5})¹⁶ (`model.embed_dim 48`), so the RLF arm uses
`hyla_dim 1536` = 96 features per factor, `hyla_scale 1`, and the radius cap 5 (`model.hyla_radius_cap`): the cap
never binds on the trained states (exp rate 3 = unit rate 6 → dimensionless radial s < ~2) but keeps the features
readable for tokens that drift past s ≈ 6 during sampling at the default horizon (unit 1.5 → s ≈ 3 ± 1.7), where the
uncapped map would return ~0 and, under `naive`, the posterior of an already-resolved token would collapse to the
context-only guess. The concat variant (`hyla_concat_state`) is NOT run here (cost); it is decided on Sudoku first.

| arm | knobs | cells |
|---|---|---|
| `none` | `HYLA_DIM=null` | naive seeds 1-3 and horosphere seed 1 REUSED from `claude_test_hbfm_tinystories_256` (spec cell `mode-exp_rate-3_k-0.5_ub-1.0[_fwd-horo]_seed-N`, same recipe, trained 2026-09-18..21); horosphere seeds 2-3 trained here |
| `cap5` | `HYLA_DIM=1536 HYLA_SCALE=1 HYLA_RADIUS_CAP=5` | naive and horosphere, seeds 1-3 |

## Hypothesis

H1: the kernel feature map gives the Euclidean DiT a better-conditioned input than the raw ball coordinates and lowers
GenPPL at equal entropy, more so under `naive` (the trunk must learn the Bayes posterior from the input) than under
`horosphere` (which adds the exact geometry at the output and tied `naive` at matched horizon in the previous study).
H0: no difference beyond the seed spread (previous study: ±0.56 GenPPL over seeds at the matched horizon 0.5, ±0.67 at
the default horizon) → "consistently better" is not supported.

## Design — the anchor is the previous study's spec cell

| | |
|---|---|
| data | TinyStories, GPT-2 tokenizer, wrapped seq 256, 475M train / 5M val tokens |
| model | small-hyperbolic-dit 768/12/12, `model.embed_dim = 48` lifted by `in_proj` (48 → 768 for `none`; 1536 → 768 for `cap5`), init ngpt |
| geometry | `prod_factor_dim = 3`, `prod_factor_gaussian_curvature = −0.5` → 16 factors of H³ |
| objective | denoising CE, `forward_type ∈ {naive, horosphere}`, `readout_precision = float32`, chunked horosphere readout |
| proposal | `time_conversion_mode = exp`, physical `time_exp_rate = 3` (unit 6), `time_range_upper_bound = 1.0`, noise `log-linear` |
| training | 30k steps, global batch 512 (4 GPUs × micro-batch 16 × accum 8 on 48 GB; 4 × 8 × 16 on the 24 GB A5000), bf16, EMA 0.9999, AdamW LR 3e-4 / wd 0 / betas 0.9, 0.999 / eps 1e-8, grad-clip 1.0; checkpoints every 1k (rolling) + `keep-{5k,10k,…}.ckpt` |
| eval | `ppl_eval` (importance-weighted denoising-CE bound) + `sample_eval`: 64 samples, 180 steps, exact velocity, `top_k_velocity = 1`, greedy last step, gpt2-large retokenized GenPPL + entropy; at the default horizon (physical `t_max 3.0` = unit 1.5 → `eval/`) AND at the matched horizon that won the previous study (`t_max 0.5` → `eval_tmax0.5/`) |
| paired read | `sweep.py --paired-eval --t-max {0.5, 3.0}` with `EVAL_NODELIST` re-reads every final checkpoint of both arms on ONE node (`paired/<tag>/`), because GenPPL re-reads differ by up to 0.46 across GPU types — the headline table uses these |
| outputs | `outputs/claude_test_hbfm_rlf_tinystories_256/hyla-{arm}_fwd-{naive,horo}_seed-{s}/{checkpoints, eval/, eval_tmax0.5/, eval_step<N>[_tmax<T>]/}` and `paired/` |

## GPU allocation and expected wall clock

4 GPUs per training cell on `thickstun,desa` minus the 2080 Ti node. Measured before: naive ≈ 7.3 h on 4 RTX 6000 Ada
(~11 h on A6000, ~15 h on A5000); horosphere ≈ 2.5× (16.5 / 27.5 / 38.5 h). 8 cells ≈ 3 × 11 + 5 × 27 ≈ 170 job-hours
× 4 GPUs. The cluster is saturated by the user's own `init_test_*` jobs, so expect several days of wall clock; the
naive RLF cells are submitted first, then horosphere RLF, then the two horosphere baselines. Early reads
(`--eval-step 5000/10000/…`) compare with the previous study's per-step table B while the cells run.

## Decision rules (fixed before the runs)

- Headline = 3-seed mean GenPPL at entropy ≥ 3.8 and 64/64 distinct samples, at BOTH horizons, from the paired reads.
  "Consistently better" = lower GenPPL in all three seeds and a mean gain beyond ~2 SE of the difference (≈ 1 GenPPL
  at the matched horizon; the previous seed std was 0.56); a smaller gain is reported as a tie.
- If `cap5` helps under `naive` but not under `horosphere`, the reading is "RLF supplies what the horosphere readout
  already supplies"; if neither, H0.
- The concat arm is added here only if it clearly beats `cap5` on Sudoku (≥ 7 pt).

## Stage 4 — the `sphere` map (2026-10-03)

HyLa is replaced by the bounded hyperbolic-harmonic basis of the user's design note (`models/hyperbolic_dit.py::
HarmonicFeatures`, `model.sphere_degree = 4`; see `claude_test_hbfm_rlf_sudoku/EXPERIMENT.md` "Stage 4" for the map,
its tests and profile). Per H³ factor: 25 features f_l(s) Y_lm(u), l ≤ 4, + 6 radial channels tanh(s/c), c = 1…32 →
16 × 31 = 496 inputs to `in_proj` (the `best` HyLa arm fed 1536, the baseline 48). Gate: Sudoku stage 4 recovered the
baseline with no collapsed seed (2-seed means within ±1 pt of `none` at every L; uncapped HyLa scored 0 there), so the
sudoku check of the design note passes while the angular-resolution question is this experiment's. Arm `sphere-L4`
(sweep.py; tags `sphere-L4_fwd-{naive,horo}_seed-{s}`), same recipe and evals as the other arms (30k steps, 4 GPUs,
`eval/` at t_max 3 and `eval_tmax0.5/`). Naive readout first (3 seeds), horosphere only if naive shows a gain — the
stage-3 precedent. Decision rule as in "Decision rules": vs the `none` naive baseline 15.58 ± 0.67 (t_max 3) /
13.25 ± 0.56 (t_max 0.5) on own-node reads, a gain is a 3-seed mean below the baseline's in every seed; a paired
single-node re-read settles anything within the GPU-type spread (0.46).

## Status

- 2026-09-30 16:35 EDT: sweep + report written. Restored the persistent `keep-{step}.ckpt` callback in
  `scripts/train/tinystories/hbfm.sh` (its `KEEP_EVERY` knob had been declared but not wired since commit d7a747b, so
  `--eval-step` reads would have found nothing). Smoke test of the `cap5`/naive body on an A5000 (4 steps, 1 GPU,
  0.2 % of the validation set, 16 samples): `keep-4.ckpt`, `eval/` and `eval_tmax0.5/` all written.
- 2026-09-30 17:00 EDT: 8 cells submitted, SLURM jobs 718970-718977 (4 GPUs each) in the order naive RLF seeds 1-3,
  horosphere RLF seeds 1-3, horosphere baseline seeds 2-3; all pending on `Priority` behind the user's `init_test_*`
  jobs. Same script-snapshot invocation as the sudoku sweep (see its EXPERIMENT.md).
- 2026-10-01 02:15 EDT: all 6 RLF cells training (5k–13k of 30k); 5k / 10k reads in RESULTS.md — RLF 20–30 % worse
  GenPPL under naive at matched steps, ~2× worse under horosphere at 5k (1.8× at 10k), at higher entropy.
- 2026-10-01 05:45 EDT: 15k / 20k reads added to RESULTS.md (RLF 15–30 % worse under naive, 1.6× under horosphere at
  15k); first 30k finals due from ~07:30 EDT; horosphere RLF seed 3 and the two horosphere baselines still queued.
- 2026-10-01 21:55 EDT, **stage 2 — HyLa hyperparameter screen (`sweep.py --screen`).** Same reference-code finding
  as the sudoku project (λ halved there and `lambda_scale` tuned to 0.01–0.5 → our `hyla_scale` 0.005 / 0.05 / 0.25;
  no 1/sqrt(D) → `hyla_normalize=false`). Six 5k-step cells (naive, seed 1, 4 GPUs, tags
  `screen5k_sc-{scale}_nm-{1,0}_seed-1`), read at t_max 3.0 and 0.5 and compared with the baseline's 5k reads
  (21.1 ± 1.5 at 0.5 / 25.0 ± 3.1 at 3.0) and the scale-1 arm's (26.0–26.9 / 32.5–34.8). The best config is then
  trained to 30k on 3 seeds. The stage-1 cells (scale 1) continue unchanged.
- 2026-10-02 03:30 EDT, **stage 3 — the sudoku winner at 30k.** Sudoku stage 2 found scale 0.25 unnormalized (reference-
  code features) at 62.8 ± 1.6 vs the baseline 61.7 ± 0.4; the 5k screens here cannot rank the small-scale configs
  (30.3–32.5 / 23.6–24.6, all within the baseline's 5k seed spread), so that config goes to 30k on 3 seeds as arm
  `best` (naive readout; horosphere only if naive shows a gain — it collapsed in 2/3 seeds at scale 1).
- 2026-10-02 07:30 EDT: the `best` cells train with the intended HyLa config (hyla_dim 1536, scale 0.25, unnormalized —
  verified in their config trees) but without kept checkpoints: commit f3c4013 (the user's) rewrote
  `scripts/train/tinystories/hbfm.sh` without the `+callbacks.checkpoint_keep.*` block, so `--eval-step` reads are not
  possible for this arm; only the 30k finals (`eval/`, `eval_tmax0.5/`) will be read. Left as committed.
- 2026-10-02 11:25 EDT, **migration to ch2263 (user request).** `sweep_ch2263.py` runs the same cells on the A100 node
  (partitions `nlplarge-claire-highpri,nlplarge`, 2 GPUs x micro-batch 32 x accumulation 8, env `sfm` — the documented
  `hodlr` env has a broken flash_attn build): `best` x horosphere seeds 1-3 (new) and A100 replicates of `best` x naive
  seeds 2-3 (pulled into `eval_a100*/`). Code, data_cache and the GPT-2 caches were rsynced to the ch2263 copy; the repo
  copy's `data_cache`/`outputs` are the user's symlinks into /scratch (compute nodes only), so jobs stage data
  themselves and copy evals back to /home/ch2263/syc_workspace/s-flm_results. 1-GPU smoke (4 steps + tiny eval) passed.
- 2026-10-02 19:30 EDT: **closed.** The user cancelled all pending HyLa jobs (ch2263 886614/886616/886618/892100, local
  paired reads 901776/901777 of `best` seed 1); `best` seeds 2-3 (866622/866623) were running and finish on their own
  (~04:30 EDT Oct 3). HyLa is superseded by the harmonic basis (`sphere`), see the sudoku project's stage 4; TinyStories
  cells for it will be added here only if sudoku passes.
- 2026-10-03 04:30 EDT: **stage 4** — `sphere-L4` × naive × seeds 1-3 submitted (jobs 921544-921546, 4 GPUs each; they
  queue behind the sudoku seed-3 cells and the `best` seeds 2-3 cells, which are at their 30k finals).
- 2026-10-03 06:00 EDT: HyLa `best` seeds 2-3 finals in (19.37 / 13.93 and 19.52 / 14.25): arm complete, 19.18 ± 0.46 /
  14.30 ± 0.40 vs baseline 15.58 ± 0.67 / 13.25 ± 0.56 → the HyLa study on TinyStories is CLOSED with no improving
  configuration (RESULTS.md "Stage 3"). The three `sphere-L4` naive cells started 04:38–05:14 EDT (A6000 / A5000s,
  ~24 h on the A5000s).
- 2026-10-03 09:35 EDT: sphere-L4 naive seed 1 @ 10k: 21.72 / 18.10 (bound 1.845) vs `none` @ 10k
  20.03 ± 1.72 / 16.53 ± 0.80 (bound 1.840): +8 % / +9 % behind at matched
  steps (HyLa cap5 was +36 % / +24 %). The 10k checkpoint was copied to keep-10000.ckpt by hand (the train script no
  longer keeps periodic checkpoints; the Lightning `2-10000.ckpt` is overwritten at 20k) so `sweep.py --eval-step`
  could read it. Finals pending.
- 2026-10-03 15:30 EDT: 10k reads of all three sphere-L4 naive seeds: 20.86 ± 1.49 / 17.42 ± 0.93 vs `none` @ 10k 20.03 ± 1.72 / 16.53 ± 0.80: behind in every seed by 2–11 % (RESULTS.md "Stage 4"). Finals pending.
- 2026-10-03 18:40 EDT: sphere-L4 naive seed 1 final: 16.86 @ 3.92 / 13.47 @ 3.93, bound 1.717 vs `none` seed 1 15.56 /
  12.90, bound 1.715 → +8 % / +4 % GenPPL at equal entropy and bound; 9–13 % better than every HyLa final. Paired
  re-reads of this cell on thickstun-compute-01 submitted. Seeds 2-3 at epoch 4.3 of 8.2 (~08:00 EDT Oct 4).
- 2026-10-04 06:50 EDT: **stage 4 COMPLETE** — sphere-L4 naive 3 seeds = 16.09 ± 1.09 / 13.53 ± 0.34,
  bound 1.719 vs `none` 15.58 ± 0.67 / 13.25 ± 0.56, bound 1.715: a tie (+3 % / +2 %, inside seed and
  GPU-read spread), equal entropy; 16 % / 5 % better than HyLa's best. No horosphere arm (naive gain rule not met).
  Experiment CLOSED; final answer in RESULTS.md.

## Stage 5 — tuning the sphere map (2026-10-04 07:40 EDT)

User: "hyperparameter tuning for the sphere features, try to make it perform better than the raw Poincaré coordinates".
Rationale for the grid: the harmonic basis is a hyperbolic positional encoding — degree l resolves angular detail
~1/l on each factor's S², gated on past s ≈ ln(2l) — so on a 50k-word vocabulary the degree is the knob that can add
information the coordinates (L = 1) lack, and L = 4 (the only TinyStories value so far) may simply be too low; the
trunk LR is the knob that interacts with a 10× wider input (the L = 4 10k reads lagged, then caught up). Seed-1
screens at 30k (naive readout): `sphere-L8` (1392 inputs, job 951500), `sphere-L12` (2800 inputs, job 951501),
`sphere-L4-lr5e-4` (job 951503; `ARM_LR` in sweep.py). Baseline to beat on seed 1: `none` 15.56 / 12.90; sphere-L4
16.86 / 13.47. A screen that beats `none` by > 1 GenPPL at either horizon gets seeds 2-3; the sudoku stage-5 cells
(same knobs, 3 seeds) guard against instabilities.
- 2026-10-05 11:15 EDT: stage 5 screens COMPLETE (RESULTS.md "Stage 5"): L 8 / 12 = L 4 (16.41 / 16.07 vs 16.86 at t_max 3),
  bound worsens with L; LR 5e-4 at L 4 = 15.70 / 13.30, bound 1.710 (< none 1.715). Stage 6 submitted: none at LR 5e-4
  (control, job 990767), sphere-L4 at 1e-3 (990768), sphere-L2 at 5e-4 (990769).
