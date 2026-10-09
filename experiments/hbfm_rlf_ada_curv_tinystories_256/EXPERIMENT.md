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
