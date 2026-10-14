# hbfm_rlf_ada_curv_tinystories_256 — adaptive curvature for HBFM on TinyStories (seq 256)

Spec: `setup.md`. Method and math: `experiments/hbfm_rlf_ada_curv/math.md` (implemented and verified in
the Sudoku study `experiments/hbfm_rlf_ada_curv`, whose `EXPERIMENT.md` / `RESULTS.md` hold the tests
and the pre-flight). Sweep: `sweep.py` (orchestration only; calls `scripts/train/tinystories/hbfm.sh`
+ `scripts/sample/tinystories/hbfm.sh`). Checkpoint storage on ch2263: `experiment.py`.

## Task

Hold the spec fixed — TinyStories seq 256, small hyperbolic DiT 768/12/12, `(H^3)^16` (embed_dim 48)
at base curvature K = −0.5, `naive` readout, plain CE, log-linear noise, 30k steps, global batch 512,
bf16, EMA 0.9999, AdamW (wd 0, betas 0.9/0.999, eps 1e-8), clip 1.0 — with adaptive curvature on
(bound b = `algo.curvature_bound` = 4: λ ∈ [1/4, 4] around K), and vary only `time_conversion_mode`,
`time_exp_rate`, `time_range_upper_bound` and LR ∈ {3e-4, 5e-4, 1e-3} to find the lowest GenPPL at a
high entropy. Eval: exact velocity, `top_k_velocity = 1`, 180 steps, greedy last step, 64 samples.

## What the adaptive curvature is on TinyStories: one learned global clock

λ is read at the t = 0 state, the only state the forward bridge and the sampler share along the whole
path (`math.md` §5). At t = 0 every free position is at the origin; only a prompt (pinned clean
positions) distinguishes positions. TinyStories is unconditional and wrapped (no prompt, no padding),
and the DiT has rotary attention only (no absolute positions): identical inputs give identical
outputs. So the head outputs one λ for all 256 positions and all sequences — a learned global clock,
trained to minimise E[L̂²] (`math.md` §6). On Sudoku the pinned givens made λ per-position; here they
cannot. (Agreed with the user 2026-10-04 before launch: run as built; per-word curvature would need a
MuLAN-style latent.)

With a global λ the cell (mode, r, ub) trains the clock s = λt with the proposal

| mode | proposal on the clock s = λt | sampler horizon on the clock |
|---|---|---|
| `exp`, rate r | Exp(r/λ) | λ · t_max |
| `trunc_exp`, rate r, ub | Exp(r/λ) truncated to [0, λ·ub] | λ · t_max ≤ λ · ub |
| `unif`, ub | U[0, λ·ub] | λ · t_max ≤ λ · ub |

so the curvature re-parametrises the three swept knobs rather than adding a new one.

## Hypotheses

- **H1 — exp: the rate axis flattens.** The mean of L̂ is λ-invariant only for an unbounded
  proposal (`math.md` §3); there λ is set by the variance alone and tracks the rate (λ* = r/a, Sudoku:
  λ ∝ r). Prediction: λ(rate 20) / λ(rate 3) ≈ 20/3 unless clipped at 4; the trained models of rates 3
  and 20 are closer than the fixed ones (fixed: 13.25 vs 16.36 at t_max 0.5), most visibly at a
  matched clock horizon λ·t_max.
- **H2 — bounded modes: the clock shrinks the range.** With a bounded proposal E[L̂] = ∫₀^{λ·ub} F,
  which grows with λ, so the second-moment gradient (variance + mean²) pulls λ down. For `unif`
  E[L̂²] = λ·ub ∫₀^{λ·ub} G is increasing in λ: λ → 1/4 (Sudoku `unif`: 95 % of cells at the lower
  bound). Predictions: `unif` ub 2 trains U[0, 0.5] on the clock and lands near the fixed `unif` ub
  0.5 (13.19) instead of the fixed ub 2 (17.18); `trunc_exp` ub 0.5 shrinks below the fixed optimum's
  range (no gain over 12.64, possibly a loss); `trunc_exp` ub 1 and 2 shrink back toward the good
  range.
- **H3 — no gain at the optimum.** No adaptive cell beats the fixed-curvature optimum (`trunc_exp`
  rate 3 ub 0.5: 12.64 ± 0.43 @ entropy 3.93) by more than seed noise; the adaptive curvature repairs
  mis-set proposals (C, D) — the Sudoku pattern.
- **LR.** LR 1e-3 at the spec point (E-FLM's best LR, 11.44 ± 0.54 @ 3.82 on the same protocol) is
  untested for HBFM; no prediction.

## Design

Stage 1 (seed 1, submitted 2026-10-04 09:33; all cells b = 4):

| id | mode | rate | ub | LR | own horizon t_max | fixed-curvature reference (own / t_max 0.5) |
|---|---|---:|---:|---|---:|---|
| A | exp (spec) | 3 | – | 3e-4 | 3 | 15.58 ± 0.67 / 13.25 ± 0.56 |
| L | exp | 3 | – | 1e-3 | 3 | — |
| B | trunc_exp | 3 | 0.5 | 3e-4 | 0.5 | 12.64 ± 0.43 (the fixed optimum) |
| F | trunc_exp | 3 | 1.0 | 3e-4 | 1.0 | — (spec read at t_max 1: 13.34) |
| G | trunc_exp | 3 | 2.0 | 3e-4 | 2.0 | — |
| C | unif | – | 2.0 | 3e-4 | 2.0 | 17.18 / 15.80 |
| D | exp | 20 | – | 3e-4 | 3 | 19.01 / 16.36 |

Plus seeds 2–3 of A (the spec point, `setup.md` seeds {1, 2, 3}): seed 2 started on the idle 2080 Ti
node with stage 1. Stage 2: seeds 2–3 of the best one or two stage-1 cells, and LR 5e-4 / 1e-3 on the
best cell if L beats A.

References are the fixed-curvature cells of `claude_test_hbfm_tinystories_256` (older code, Ada /
A6000 / A5000); cross-node reads differ by 2–3 % there, seeds by 3–4 % at 30k.

Reads per cell (`eval*/`): the own horizon (exp: t_max 3 = unit 1.5; bounded: t_max = ub, the HBFM
validator rejects t_max > ub — `setup.md`'s "T_MAX ≥ ub" and the validator's "T_MAX ≤ ub" both hold
at T_MAX = ub; exp ignores ub) and, for exp, t_max 0.5. After training: λ per checkpoint
(`curvature_ts.py`, EMA weights) and extra reads at a matched clock horizon λ·t_max where they decide a
ranking.

Metrics: GenPPL (gpt2-large, retokenised, 64 samples) read with the unigram entropy and the number of
distinct samples; λ; `val/ppl` is the importance-weighted CE bound — logged, not used to rank.

## Compute and wall clock

| site | cell layout | measured speed | per cell (30k + eval) |
|---|---|---|---|
| ch2263, `nlplarge-claire-highpri,nlplarge` (A100 80 GB) | 1 GPU × micro-batch 64 × accum 8 | 4.1–4.2 it/s → 1.9 s/step | ~16 h + ~0.5 h |
| sc3379, desa-compute-01 (8× 2080 Ti) | 8 GPUs × 16 × accum 4 | 1.18 it/s → 3.4 s/step | ~28 h |
| sc3379, desa / thickstun 4× A6000 / A5000 | 4 GPUs × 16 × accum 8 | — | not used: Slurm's earliest start was 2026-10-06 18:36 (sc3379 has ~290 queued jobs of other sweeps) |

At submission A and L started on A100s (the highpri QOS caps the account at 2 GPUs, one held by an
interactive job; `nlplarge` jobs start only when GPUs free up) and A seed 2 on the 2080 Ti node; the
other five stage-1 cells wait for A100s. A100 jobs carry `--requeue` (preemptible); a requeued job
resumes from `checkpoints/last.ckpt` (≤ 1k steps lost). Stage 1 ≈ 7 × 16.5 A100-h + 28 h of the
2080 Ti node; wall clock ≥ 17 h, set by when the pending cells get GPUs.

Checkpoints on ch2263 live on the node's `/scratch/ch2263/syc_workspace/sfm_output/<project>/<run>/`
(linked from the run directory by `experiment.py`); every 1k steps (rolling) + `keep-{step}.ckpt` every
5k. Results come back with `python sweep.py --pull`.

## Frontier evaluation (requested 2026-10-05)

HBFM joins the Gen. PPL / entropy frontier of `experiments/naive_ar_tinystories_s256`
(`FRONTIER_RESULTS.md`: MDLM, DUO, FLM, S-FLM + trunc + ada, E-FLM) under its protocol: NFE
{1, 4, 8, 16, 32, 64, 128, 256}, 512 samples per cell, training seeds {1, 2, 3}, greedy last step.
Driver: `frontier_sweep.py`; figure: `visualization/genppl_entropy_frontier_line.py --hbfm-dir`.

- **Models.** `ada` = this study's best cell (trunc_exp ub 1, adaptive curvature, bound 4);
  `fix` = the fixed-curvature best (trunc_exp ub 0.5, `claude_test_hbfm_tinystories_256`).
- **Two velocity arms.** `k1` (`top_k_velocity = 1`, the spec sampler) is temperature-inert —
  the velocity and the last step are argmaxes — so it is one marker per NFE, like E-FLM's k = 1.
  `exact` (`top_k_velocity = -1`, the full-vocabulary velocity) lets T 0.50–1.20 trace a curve,
  the same deviation the frontier made for S-FLM and E-FLM. Agreed order: markers first, the
  curve afterwards (~50 A100-h per model: dense [B, 256, 50257] float64 per factor per step).
- **Truncation.** The sampler horizon t_max (≤ ub, the HBFM validator) is picked per model on
  the k1 ladder of seed 1: ada t_max ∈ {0.25, 0.5, 0.75, 1}, fix ∈ {0.25, 0.35, 0.5}. Rule: the
  lowest Gen. PPL summed in log over NFE 8–256, unless its entropy is lower than another
  candidate's by more than seed noise (~0.03), in which case the frontier — lower Gen. PPL at
  no lower entropy — decides.
- **Compute.** ch2263 A100s (the fixed checkpoints copied there with `--copy-fixed`). k1:
  ~30 min per (model, seed, t_max) job.
- **Picked (2026-10-06, seed 1 k1 ladder, 512 samples):** ada **t_max = 1** (Σ log PPL over NFE
  8–256 = 15.819 vs 15.829 at 0.75 — a tie; mean entropy 3.923 vs 3.922), fix **t_max = 0.25**
  (16.321 vs 16.487 at 0.35 and 17.063 at 0.5; entropy 3.913, not lower). Both decode at clock 0.25.

## Frontier: the ODE and SDE step types of the adaptive model (requested 2026-10-09)

`samplers.HBFMSampler` gained the marginal-preserving family of the bridge SDE (`sampler.step_type`):
`em` (γ = 1, the Euler–Maruyama step used so far), `ode` (γ = 0, the probability-flow ODE) and
`sde` (γ = g(t / t_max), `sampler.gt_method` ∈ {sqrt, linear, quad} = (1 − s)^{0.5, 1, 2};
`const` is `em`). Every step type takes one Euler–Maruyama step from the origin (a point mass).
Task: put the `ada` model's ODE and best-g(t) SDE on the frontier figure next to its `em` arm.

**Fair comparison — the same selection protocol as the `em` arm, per sampler.**
1. Seed-1 k1 ladder (512 samples, NFE 1–256) over t_max ∈ {0.25, 0.5, 0.75, 1} for `ode` and for
   each `sde` g(t); pick per sampler by the rule above (lowest Σ log Gen. PPL over NFE 8–256,
   unless its entropy is lower than another candidate's by more than ~0.03 — then the frontier
   decides); for `sde` the pick spans (g, t_max) jointly, and only the best g is presented.
2. At the pick: k1 seeds 2–3, then the exact-velocity curve (T 0.50–1.20, seeds 1–3).
3. Same checkpoints (EMA), 512 samples per cell, greedy last step, gpt2-large Gen. PPL; the sample
   batch is 16 (8 on the 24 GB A5000s — the batch only changes the random stream).
Nothing else differs between the three step types: they share the marginals in the continuum,
so a gap is a discretization / noise effect at that NFE.

**Compute (ASAP, both accounts).** `frontier_sweep.py --site a6000` runs on sc3379's desa /
thickstun GPUs (the ada checkpoints were exported from ch2263's /scratch to
`outputs/<NAME>/<cell>/checkpoints/last.ckpt`); `--site a100` on ch2263's
`nlplarge-claire-highpri,nlplarge`. Jobs are split per NFE (`--split nfe`, NFE ≤ 32) or per
(NFE, T) cell (`--split cell`, NFE ≥ 64) so any freed GPU takes one. Submitted 2026-10-09 ~14:00:
the four seed-1 ladders (16 k1 jobs) on sc3379. Next check-in: pull, pick (t_max, g), submit
step 2 (k1 seeds 2–3; exact arm per cell on both sites), then draw with
`visualization/genppl_entropy_frontier_line.py --hbfm-tmax ada:1 fix:0.25 ada-ode:<t> ada-sde-<g>:<t>`
and `--methods mdlm,duo,flm,sfmta,eflm,hbfm-ada,hbfm-fix,hbfm-ada-ode,hbfm-ada-sde-<g>`.

**Picked (2026-10-09, seed-1 k1 ladders, `frontier_pick.py`):** ODE **t_max = 0.75** (Σ log PPL
over NFE 8–256 = 17.125 vs 17.223 at 1, 17.378 at 0.5; mean entropy 3.872 vs 3.865 — better on
both). t_max 1 is better at NFE ≥ 64 (14.80 vs 15.12 at 256) and 0.75 at NFE ≤ 32 (NFE 8: 24.1
vs 26.9); the rule sums the log over the ladder. The ODE exact arm first launched at t_max 1 on
ch2263 (before the ladder finished) is kept at the back of that queue (nice 100000) as an extra
same-horizon read, not the presented arm. SDE: pick recorded below when its ladder completes.
The sc3379 exact-arm jobs sample with batch 8 (24–48 GB GPUs), the A100 jobs with 16.
SDE **g = sqrt ((1 − s)^0.5), t_max = 1** (Σ log PPL 16.015 @ mean H 3.910 — lowest and
highest of the 12 (g, t_max) cells; sqrt@0.75 16.061 @ 3.909, linear@1 16.187 @ 3.901,
quad@1 16.448 @ 3.886). Stage 2 submitted 2026-10-09 ~15:00: k1 seeds 2–3 (sc3379) and the
exact arm — seed 1 on ch2263, seeds 2–3 on sc3379 — for ODE@0.75 and SDE-sqrt@1 (≈ 300 jobs).
**Complete (2026-10-10 17:00):** 3-seed k = 1 markers and exact-velocity curves at both picks are in
`RESULTS.md` (last section) and `figures/genppl_entropy_frontier_line.png`; cells in
`frontier_cells.csv`, all figure tables in `figures/frontier_tables.md`. 46 seed-1 cells were run on
both sites (the `_x` jobs) when ch2263 fell to 3 GPUs; the 44 ODE@t_max-1 extras are left to drain
at the back of ch2263's queue and are not presented.
