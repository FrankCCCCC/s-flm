# claude_test_hbfm_tinystories_256 — HBFM on TinyStories (seq 256): heat-time proposal × curvature

Spec: `setup.md` (2026-09-17 revision: `forward_type = naive`, `(H^3)^16`, `time_conversion_mode`, `time_range_upper_bound`).
Sweep: `sweep.py` (orchestration only; calls `scripts/train/tinystories/hbfm.sh` + `scripts/sample/tinystories/hbfm.sh`).
Report: `report.py` → `RESULTS.md`. Geometry calibration: `logs/calib/` (`calib_ts16.py` → `calib16.log`, `calib_ts16.json`;
the previous round's 32-factor run is `calib_ts.py` / `calib.log`). Script smoke tests and the test-suite runs: `logs/tests/`.

## Task

Hold the spec fixed (TinyStories seq 256, small hyperbolic DiT 768/12/12, `(H^3_K)^16` = `model.embed_dim 48`, denoising
CE on plain logits (`naive`), log-linear noise, 30k steps, global batch 512, bf16, EMA 0.9999, AdamW LR 3e-4 / wd 0 /
clip 1.0) and vary ONLY `time_exp_rate`, the global curvature `K`, `time_conversion_mode` and `time_range_upper_bound`
to find the best GenPPL at a healthy entropy. Seeds 1–3 for the cells that matter.

## Hypothesis

The heat-time proposal is the dominant hyperparameter (sudoku: rate 0.01 → 0–1% boards, matched rate → 61%; the previous
horosphere-forward round here on 32 factors: unit rate 0.02 → GenPPL 316 word salad, unit 5–20 → 33–46 at 5–10k steps).
On `(H^3)^16` at K = −0.5 the spec's `time_exp_rate = 3` is unit rate 6, which the calibration below puts exactly at the
ESS optimum of the exp proposal; sudoku's optimum sat at a mean heat time of ~5% of the resolution scale, which here maps
to unit rate ~25 (physical ~12). We therefore expect a broad optimum between physical rate 3 and 10–20 and a cliff on
both sides. Curvature at a fixed physical rate changes the unit rate (`rate/|K|`) and the DiT input scale `R = 1/sqrt|K|`
(sudoku: `(rate, K) → (c·rate, c·K)` is a symmetry of the problem broken only by optimizer scales), so K −0.25 / −1 at
rate 3 mostly re-samples the unit-rate axis at 12 / 3. `trunc_exp` drops the late tail of the exp proposal and `unif`
spreads the samples evenly over [0, ub]; both also bound the sampler horizon at ub. Whether either beats `exp` — and
whether a short bridge (decode at unit tau 0.25–0.5, below the context-free resolution scale) works for a trained,
contextual posterior — is the open question.

Reference points (`experiments/naive_ar_tinystories_s256/RESULTS.md`, same DiT size / data / steps, mean of 3 seeds at
the best LR): MDLM 18.8 @ entropy 4.39, DUO 17.4 @ 4.34, FLM 49.2 @ 4.55, S-FLM+trunc+ada 12.95 @ 3.95; AR 7.2 @ 4.26
is a greedy mode decode (1 distinct sample).

## Geometry calibration — `(H^3)^16`, K = −0.5 (`logs/calib/calib16.log`, CPU, Bayes posterior with a random ngpt-scale table)

Unit time `tau = |K| t`. With V = 50257 the context-free posterior's CE / argmax accuracy is 10.8 / 0.00 at tau = 0.001,
7.4 / 0.04 at 0.1, 4.4 / 0.27 at 0.2, 2.1 / 0.59 at 0.3, 0.9 / 0.79 at 0.4, 0.44 / 0.91 at 0.5, 0.08 / 0.98 at 0.6,
0.003 / 1.00 at 0.8 and 0 from tau ≈ 1.25 on — the resolution scale is tau ≈ 0.8 (0.4 for the 32-factor manifold: the
factors' evidence adds). The CE integral over heat time sits below tau 0.5 (23% in [0, 0.05], 20% in [0.05, 0.1], 38% in
[0.1, 0.25], 17% in [0.25, 0.5], 1.7% in [0.5, 1]). The trained model has context and resolves earlier than this table.

Coverage of the stage-1 proposals (unit time; `u ~ U[1e-3, 0.999]`; ESS = effective sample size of the importance
weights against the context-free CE profile, 1.0 = perfect):

| cell | proposal on tau | mean tau | P(tau ≤ 0.1) | P(tau ≤ 0.5) | ESS | draws with CE < 0.01 | horizon (unit / physical) |
|---|---|---:|---:|---:|---:|---:|---|
| exp rate 3 (spec), K −0.5 | exp(6) | 0.167 | 45% | 95% | **0.96** | 0.8% | 1.5 / 3.0 |
| exp rate 5 | exp(10) | 0.10 | 63% | 99.4% | 0.79 | 0% | 1.5 / 3.0 |
| exp rate 10 | exp(20) | 0.05 | 87% | 100% | 0.21 | 0% | 1.5 / 3.0 |
| exp rate 20 | exp(40) | 0.025 | 98% | 100% | 0.08 | 0% | 1.5 / 3.0 |
| exp rate 3, K −0.25 | exp(12) | 0.083 | 70% | 99.8% | ~0.7 | 0% | 1.5 / 6.0 |
| exp rate 3, K −1 | exp(3) | 0.33 | 26% | 78% | 0.71 | 9% | 1.5 / 1.5 |
| trunc_exp rate 3, ub 0.5 | exp(6) on [0, 0.25] | 0.10 | 58% | 100% | — (misses 19% of the CE mass) | 0% | 0.25 / 0.5 |
| unif ub 0.5 | U[0, 0.25] | 0.125 | 40% | 100% | 0.91 (misses 19%) | 0% | 0.25 / 0.5 |
| unif ub 1.0 (spec ub) | U[0, 0.5] | 0.25 | 20% | 100% | 0.63 (misses 1.7%) | 0% | 0.5 / 1.0 |
| unif ub 2.0 | U[0, 1.0] | 0.50 | 10% | 50% | 0.33 | 22% | 1.0 / 2.0 |

(unit rate 40 / 60 / 100 → ESS 0.08 / 0.07 / 0.05; the spec-0.01 proposal of the previous round would be ESS 0.006 here.)

`time_range_upper_bound` and `sampler.t_max` are PHYSICAL heat times. For `exp` the horizon is `UNIT_T_MAX = 1.5`
(≈ 2× the tau at which the context-free posterior is one-hot; 180 steps → dtau 0.0083, inside the dt ≲ 0.01/|K| guidance
of `configs/sampler/hbfm.yaml`; 60 of the 180 forward passes fall in the decision window tau < 0.5). For `trunc_exp` /
`unif` the SDE may not walk past the trained range (`algo._validate_configuration` rejects `t_max > ub` — at model
construction, so the sweep passes `T_MAX` to training too), so the horizon is `min(ub, 1.5/|K|)`: the ub-0.5 cells
decode at unit tau 0.25, where the context-free posterior is only 40% accurate, and the ub-1.0 cells at 0.5 (91%).
Because the exp cells decode at unit 1.5, a mode difference is confounded with the horizon; `sweep.py --eval-step N
--t-max T` re-samples the kept checkpoints of the exp cells at the bounded cells' horizons (0.5 / 1.0 / 2.0 physical)
for ~1 GPU-hour each, which separates the two.

## Design

| | |
|---|---|
| data | TinyStories, GPT-2 tokenizer, wrapped seq 256 (`data_cache/tinystories_*_bs256_wrapped.dat`), 475M train / 5M val tokens |
| model | small-hyperbolic-dit 768/12/12, `model.embed_dim = 48` lifted by `in_proj`, init ngpt |
| geometry | `prod_factor_dim = 3`, `prod_factor_gaussian_curvature = K` → 16 factors of H^3_K; spec K = −0.5 |
| objective | denoising CE of the model's plain logits (`forward_type = naive`), `readout_precision = float32`, importance weight `1/q(t) · |alpha'_t|` of the chosen proposal |
| proposal | `time_conversion_mode` ∈ {exp, trunc_exp, unif} with `time_exp_rate` (physical) and `time_range_upper_bound` (physical); noise `log-linear` (eps 1e-3, untruncated) |
| training | 30k steps, global batch 512 (4 GPUs × micro-batch 16 × accum 8 on 48 GB; 4 × 8 × 16 on the 24 GB A5000), bf16, EMA 0.9999, AdamW (LR 3e-4, wd 0, betas 0.9/0.999, eps 1e-8), grad-clip 1.0, constant LR + 2500 warmup |
| checkpoints | every 1k (rolling, `save_top_k = 1` + `last.ckpt`) and every 5k kept (`keep-{step}.ckpt`, a second `ModelCheckpoint` via `KEEP_EVERY`) |
| eval | `ppl_eval` on the full validation set (importance-weighted denoising-CE bound, NOT a likelihood) + `sample_eval`: 64 samples, 180 steps, exact velocity, `top_k_velocity = 1`, greedy last step, horizon as above, gpt2-large retokenized GenPPL + unigram entropy |
| outputs | `outputs/claude_test_hbfm_tinystories_256/mode-{m}_rate-{r}_k{K}_ub-{ub}_seed-{s}/{checkpoints, eval/, eval_step{N}/, eval_step{N}_tmax{T}/}` (`rate` is a no-op for unif, `ub` for exp; both stay at the yaml defaults 3 / 1.0 in the tag) |

### Stage 1 (seed 1, one axis at a time around the spec point; 10 cells, submitted 2026-09-17 evening)

| axis | cells |
|---|---|
| rate (exp, K −0.5) | rate ∈ {**3**, 5, 10, 20} — unit rate 6 / 10 / 20 / 40 |
| curvature (exp, rate 3) | K ∈ {−0.25, −1.0} (+ −0.5 from the rate ladder) — unit rate 12 / 3, R = 2 / 1 (1.41 at −0.5) |
| trunc_exp (rate 3, K −0.5) | ub = 0.5 (ub 1.0 = a 5%-of-draws truncation of the spec cell; deferred to stage 2, see `CONFIRM`) |
| unif (K −0.5) | ub ∈ {0.5, 1.0, 2.0} |

Priority (SLURM `nice`, set after submission, not by the sweep): spec point first, then the rate ladder (5, 10, 20), the
curvature pair, unif 1.0, trunc_exp 0.5, unif 0.5 / 2.0. Early reads: `sweep.py --eval-step 5000` (then 10000, 20000)
evaluates the kept checkpoints of every cell on one GPU so the axes can be ranked before the 30k runs finish (the previous
round's 5k rate ranking held at 10k); `--eval-step N --t-max {0.5, 1.0, 2.0}` on the exp cells gives the horizon control.

### Stage 2 (after the 5k/10k reads; `CONFIRM` in `sweep.py`)

Seeds 2–3 of the spec point and of the best cell of each winning axis; one extra point on any axis whose optimum is at the
edge of the ladder (rate 40 if 20 wins, unif ub 0.25 if 0.5 wins, trunc_exp ub 1.0 if trunc_exp 0.5 wins). Cells are
added to `CONFIRM` and submitted with `python sweep.py`; the sweep is idempotent (skips finished / queued cells) and
every cell resumes from `checkpoints/last.ckpt`.

Stage 2, first additions (2026-09-18 01:15, after the complete 5k rate ladder 25.1 / 29.4 / 31.7 / 38.3 for rate 3 / 5 /
10 / 20 — monotone, spec best): `('exp', '1.5', '-0.5', '1.0', '1')` (unit 3, one point below the spec; the same unit rate
as the K −1 cell, so the pair is the rate/|K| equivalence test at unit 3) and seeds 2–3 of the spec cell, queued behind
all stage-1 cells. Rate 2 / rate 1 follow only if 1.5 beats 3.

Stage 2, second addition (2026-09-18 17:30, after trunc_exp ub 0.5 finished at 12.59 @ 3.92 vs the spec's 12.90 at the
same horizon, a 2–4% edge at every checkpoint from 15k): `('trunc_exp', '3', '-0.5', '0.5', '2')` and `('…', '3')`, queued
behind the spec's seeds 2–3, so the winner is decided on 3-seed means. Rate 5 (tied with the spec at the default
horizon on and off, 4–9% behind at the short horizon) and unif ub 1.0 (5–10% behind exp at matched horizon) get no seeds.

Success criterion: every cell finishes with finite loss; the ranking is on GenPPL at comparable entropy (≥ 3.8, 64/64
distinct samples), with the spec point (exp, 3, −0.5) as the reference; the answer is the best (mode, rate, K, ub) with
its 3-seed mean ± std.

What actually ran (stage 2 closed 2026-09-19 18:55): `CONFIRM` = rate 1.5 seed 1, spec seeds 2–3, trunc_exp ub 0.5
seeds 2–3 — 15 cells in total, all finished at 30k with finite loss and `eval/samples_genppl.json`. Not run: the planned
trunc_exp ub 1.0 edge cell (the spec cell sampled at `t_max` 1.0 is its proxy, 13.34), rate 40 and unif ub 0.25 (their
axes did not win — note that unif ub 0.5 is the best *and lowest* point of the unif ladder, so that axis is open on the low
side like trunc_exp's; unif ub 0.25 was still not run), and seeds for unif ub 0.5 (13.18, single seed, 4% behind
trunc_exp). Rate 5 got no seeds because it lost at matched horizon (10k 17.85 vs 16.33, 30k 14.06 vs 13.25) despite tying
the spec at the default horizon at 10k–25k. The checkpoint reads at
5k/10k/15k/20k/25k and the matched-horizon (`t_max` 0.5) rows at 10k/20k/30k exist for every cell (140 distinct (cell, step, horizon) reads from 152 eval files; 12 are duplicate re-reads, and 7 are the 2026-09-20 shorter-horizon probe).

### Stage 4 — `forward_type = horosphere` (added to `setup.md` by the user 2026-09-20)

The spec now reads `forward_type: {naive, horosphere}` and its task line adds `forward_type` to the arguments to vary.
Everything in stages 1–3 is the `naive` half of that grid, and **the repo contains no head-to-head horosphere-vs-naive
run**: round 1 below was horosphere, but on 32 factors, at 5k–11k of 30k steps, sampled at physical `t_max` 2.0 and with
no `time_conversion_mode` knob, so not one of its numbers transfers.

What horosphere changes (`algo.py:941-961`): the DiT logits are treated as a *residual* on the geometry — the per-word
Busemann log-densities computed from the bridge state are added before the `log_softmax`, so the readout is the exact
Bayes posterior of the bridge direction and the boundary codebook gets gradient through the geometry. Bridge, weights,
loss and trunk input are byte-identical to `naive`. There is no config assert tying it to anything; it needs the chunked
readout (`algo.horosphere_forward_chunked: True`, the default — the dense form is a (B,L,V,m,d) tensor that OOMs), and it
must be passed to the SAMPLER too, because `samplers.py` calls `model.forward` and a horosphere checkpoint evaluated
without it is silently scored with the naive readout instead of erroring.

Cost: predicted ~2.5x naive (~1.95 s/step Ada / ~3.3 A6000 / ~4.6 A5000), and **the prediction held**: measured
1.89 s/step on the Ada (15.7 h per 30k; cross-checked against the checkpoint timeline, 4000 steps in 2 h 06 m) and
2.99–3.42 s/step on the A6000 (24.9–28.5 h, two cells sharing the node), against naive 0.79 / 1.52 — i.e. 2.4x.
Memory is a non-issue, because the chunked readout is O(B·L·V) in the factor count.

**Accumulation factor, because it has now caused two wrong throughput claims:** the progress bar counts MICRO-batches,
and micro-batch size depends on GPU memory, so steps = micro-batches / accum with accum = 512 / (4 x micro_batch):
the 48 GB nodes (RTX 6000 Ada, A6000) run micro-batch 16 -> **accum 8**, 29142 micro-batches per epoch; the 24 GB A5000
runs micro-batch 8 -> **accum 16**, 58283 per epoch. Read it off `config_tree.txt` (`loader.batch_size`) rather than
assuming.
So the probe is 3 cells at seed 1, not a re-run of stage 1: the spec point (head-to-head with naive 15.58 ± 0.67 own
horizon / 13.25 ± 0.56 at `t_max` 0.5), the naive winner's config (`trunc_exp` ub 0.5, naive 12.64 ± 0.43), and rate 10
(naive falls off to 17.82 there; if carrying the geometry explicitly makes the model less sensitive to the proposal, the
rate optimum should flatten or move up, as round 1 at 32 factors preferred unit 5–20). Everything beyond these three is
decided from their 5k/10k reads, as in stages 1–2.

**Stage-5 decision (2026-09-20 21:50, all three cells at 5k+10k):** no new cells. Horosphere starts far behind and
closes monotonically — matched-horizon gap 17.4% → 2.9% → 0.6% (spec, 5k/10k/15k) and 11.7% → 2.4% (trunc_exp, matched
by construction) — so it neither wins by the ~4% seed spread nor clearly loses; it converges to parity at ~0.03 nat
higher entropy. The 30k reads already being paid for are the cheapest tie-breaker, so the rule is re-applied there.
The early-read heuristic that decided stages 1–2 does NOT transfer to a change of objective: at 5k horosphere looked
12–38% worse in all three cells, which would have been a wrong call.

Tooling change this required: cells are now `(mode, rate, k, ub, seed, forward_type)`, and `tag_of()` appends `_fwd-horo`
**only** for non-naive cells, so all 17 existing directories, their eval JSONs and their rows keep their names. Without
it a horosphere cell would reuse the naive cell's directory: 15 of them would be silently skipped as already-done, and
the two in-flight ones would resume naive weights under a horosphere readout and overwrite the published results.
`report.py` / `final_tables.py` carry an optional `_fwd-` group and a `fwd` column; the regression check (the report
regenerated over the existing data differs from the published block only by that column and by the new horizon rows) is
in the scratchpad `verify_fwd.sh`.

## Previous round (horosphere forward, 32 factors; paused 2026-09-17 11:20, kept for reference)

Nine cells trained with `forward_type = horosphere` (the backbone's logits as a residual on the Busemann log-densities)
on `(H^3)^32` (`embed_dim 96`, a deviation from the spec's 16 factors), tags `rate-*_k-0.5_ada-*_lr-*_seed-*`, stopped at
5k–11k steps when the user refactored the code. Their 5k/10k reads are the static table in `RESULTS.md` ("Previous
round"). They are a different objective and manifold and are not resumed into this grid; the checkpoints stay in
`outputs/` under their old tags.

## Code changes made for this experiment (uncommitted; commits need the user)

- `scripts/train/tinystories/hbfm.sh`: knobs `FORWARD_TYPE` (default naive), `TIME_CONVERSION_MODE`, `TIME_RANGE_UPPER_BOUND`,
  `KEEP_EVERY` (second `ModelCheckpoint` writing `keep-{step}.ckpt`, `save_top_k = -1`; 0 = off; the `'keep-{step}'` quotes
  must reach hydra), `PROPOSAL_RATE` (physical rate; default still `UNIT_PROPOSAL_RATE * |K|`), `UNIT_T_MAX` / `T_MAX`
  (`sampler.t_max`, needed at construction for the bounded modes).
- `scripts/sample/tinystories/hbfm.sh`: `FORWARD_TYPE`, `TIME_CONVERSION_MODE`, `TIME_RANGE_UPPER_BOUND`, `PROPOSAL_RATE`,
  `T_MAX` (physical horizon; default still `UNIT_T_MAX / |K|`).
- Verified on GPU nodes before submission (`logs/tests/`): the three modes train → resume (both checkpoint callbacks
  restore; a resume that lands exactly on a keep step writes a duplicate `keep-N-v1.ckpt`, 2 GB, harmless) → ppl_eval →
  sample_eval; the bounded modes are rejected when `T_MAX > ub` (negative test); the sweep body's exact knobs with
  `EMBED_DIM 48`.
- Test suite on HEAD (`tests/test_hbfm.py` + `test_hbfm_sampler.py`, GPU): 38 pass, 5 fail, none from this experiment's
  changes — `test_horosphere_readout_is_bayes_posterior[×2]` and `test_bayes_limits_at_init` build the model with the yaml
  default `forward_type = naive` (commit 03bb642) and assert the horosphere posterior; `test_naive_readout_is_plain_log_softmax`
  compares a float32 readout (`readout_precision` default since 03bb642) against a float64 reference (`Float did not
  match Double`); `test_nll_end_to_end_is_finite_and_differentiable` asserts a non-zero embedding gradient at init, which
  under `naive` is exactly zero because `DDiTFinalLayer` is zero-initialised (nothing upstream of it gets gradient at
  init). `logs/tests/emb_grad_path.py` confirms the embedding table does receive gradient under `naive` once the output
  layer is non-zero (through the bridge state → `in_proj` path; 4.7 vs 6.6 for horosphere on the same batch), so the
  boundary codebook is trained, not frozen. The tests need `algo.forward_type=horosphere` / a dtype cast / a non-zero
  output layer respectively; not touched here.

## Compute

One 4-GPU job per cell (`thickstun,desa` minus `desa-compute-01`, per the spec), 16 CPUs, 128 GB, 7-day limit; the job
picks the micro-batch from the GPU memory. Previous-round throughput with the horosphere readout (rank-0 micro-batches/s,
4 GPUs, global batch 512): RTX 6000 Ada 2.47 it/s at micro-batch 16 → 3.2 s/step → ~27 h per 30k cell; A6000 1.46 it/s →
5.5 s/step → ~46 h; A5000 2.31 it/s at micro-batch 8 → 6.9 s/step → ~58 h. The naive forward skips the horosphere readout
(108 ms per micro-batch) and the manifold is half the size, so expect less. At submission the A6000 node was fully
allocated (an 8-GPU 2-day job), the Ada node had 1 free GPU and the A5000 node 9: stage 1 starts with 2 cells and widens
as GPUs free up; the 10 seed-1 cells need 40 GPU-slots × 1–2.5 days. Measured throughput goes into `RESULTS.md`.

Measured (naive forward, `(H^3)^16`, 4 GPUs, global batch 512): RTX 6000 Ada ≈ 0.8 s/step → 6.3 h per 30k cell including
the final eval; A6000 ≈ 1.5 s/step → 12.3 h; A5000 ≈ 2.8–3.1 s/step → 22.8 h. A 1-GPU checkpoint read (64 samples ×
180 steps + gpt2-large scoring) takes 2 min on the Ada, 3 min on the A5000. From 2026-09-19 ~01:00 the partitions were
saturated by another session's GPU-job flood and CPU-only jobs stopped being scheduled at all, so checkpoint reads were
submitted through a 20-second `--gres=gpu:1` tick (`RESULTS.md` Status, 02:50 entry).

## Outcome — stage 4, `forward_type` (2026-09-21; detail in `RESULTS.md` → Status)

All three horosphere probe cells finished 30k. **At matched sampler horizon horosphere ties naive; at the default long
horizon it loses 5–7%; it costs 2.4x the compute.** Matched-horizon, horosphere seed 1 vs the naive mean at the same
config: spec point 12.81 @ 3.958 vs 13.25 ± 0.56 (−3.3%); `trunc_exp` ub 0.5 12.75 @ 3.948 vs 12.64 ± 0.43 (+0.8%);
rate 10 14.87 vs 15.01 (−0.9%). Every gap is inside the naive seed spread. Stage 5 was therefore NOT run: the rule was
to expand only on a win larger than the ~4% seed spread, and seeds 2-3 of the best horosphere cell would have been ~57 h
of A6000 time to resolve a tie. The recommended config is unchanged by the forward_type axis.

Two methodological results worth more than the tie itself:
1. **Early reads do not transfer across a change of objective.** At 5k horosphere looked 12–38% worse in all three
   cells; the matched-horizon gap then closed monotonically (spec: +14.8 → +2.9 → +1.2 → +1.0 → +0.1 → −3.3%). Its
   logits are a residual on a strong geometric prior, so it starts behind and catches up. The 5k/10k screening that
   decided stages 1–2 (hyperparameters within one objective) would have made the wrong call here.
2. **Compare at a matched sampler horizon or not at all.** Horosphere gains ~22% from shortening the horizon vs naive's
   ~15%, so own-horizon reads systematically flatter naive. Two claims in the Status log had to be retracted for this
   reason (rate 10 "not converging"; horosphere "much more rate-sensitive" — measured properly it is 1.16x vs 1.13x).

## Outcome (2026-09-19; full analysis in `RESULTS.md` → Analysis)

Best cell: `trunc_exp`, rate 3, K −0.5, ub 0.5 → GenPPL 12.64 ± 0.43 @ entropy 3.93 ± 0.02 (3 seeds, 64/64 distinct).
The spec point (exp, 3, −0.5) gives 15.58 ± 0.67 at its default horizon and 13.25 ± 0.56 when sampled at `t_max` 0.5;
the sampler horizon (−15%) is a larger effect than any training-time knob, and among training knobs the spec's rate 3 /
K −0.5 (unit rate 6, the calibration's ESS optimum) is the best point on both the rate and the curvature axis. Wider
proposals (rate ≥ 10, unif ub ≥ 1.0) and other curvatures lose 10–25%. The best cell ties S-FLM + trunc + ada (12.95 @
3.95) and beats MDLM / DUO on GenPPL at ~0.45 nat lower unigram entropy.

