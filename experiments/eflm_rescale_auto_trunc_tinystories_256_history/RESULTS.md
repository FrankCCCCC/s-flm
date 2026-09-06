# eflm_rescale_auto_trunc_tinystories_256_history — Results

Task (`setup.md`): *vary R, LR and the truncation timestep to find the optimal
GenPPL **without collapsing***, for truncated autonomous-clock E-FLM on
TinyStories seq 256, plain CE (`{w/o SNR}`), **retaining the full checkpoint
history** — every 1k steps, `save_top_k=-1`.

Protocol: `small-sphere-dit` (768x12x12), ngpt init, 30k steps, global batch 512
(4 GPU x 32 x accum 4), seq 256, bf16, EMA 0.9999, AdamW wd 0, clip 1.0,
`ALPHA_MAX=null`. Eval: exact velocity, `top_k_velocity=1`, 180 sampling steps,
greedy last; GenPPL = gpt2-large retokenised generative perplexity.
Design and hypotheses: `EXPERIMENT.md`. Live tables: `RESULTS_tables.md`.

**STATUS: COMPLETE. 15 cells — 8 (LR x R) + 4 (truncation) + 3 extra seeds.
465 checkpoints, 1.2 TB. Every cell admissible; no collapses anywhere.**

---

## 1. Headline

**Recommended: `R = 0.5, lr 1e-3, TAU_MAX = tau*(0.5) = 2.4436`
-> GenPPL 11.04 +- 0.85 (3 seeds), entropy 3.74-3.80.**

All three knobs have interior optima, and the truncation lands exactly on the
closed form:

| knob | optimum | penalty for one grid step | evidence |
|---|---|---|---|
| LR | **1e-3** | +1.6 to +2.1 (3e-4) at every R | 8-cell grid |
| R | **0.5** | +1.06 (R=1), +0.85 (R=0.05) | 3 seeds at R=0.5 and R=1 |
| **TAU_MAX** | **tau*(R)** exactly | +0.72 short, +3.20 / +3.53 long | 4-point curve |

The single best *cell* was `R = 0.5` at seed 1, **10.24** — but that is the best
of three draws from a config whose mean is 11.04. The recommendation is stated
as the mean, not the draw.

| reference (same protocol, same arm) | GenPPL | seeds |
|---|---|---|
| **this sweep, recommended (R=0.5, lr 1e-3)** | **11.04 +- 0.85** | **3** |
| runner-up (R=1, lr 1e-3) | 12.10 +- 0.31 | 3 |
| parent sweep, same configuration (mixed hardware) | 11.44 +- 0.54 | 3 |
| project incumbent: log-linear + trunc + **adaptive**, R=1 | 10.36 | 1 |

Agreement with the parent's independent 3-seed estimate of the same
configuration (11.04 vs 11.44, well inside either sd) is the strongest evidence
here that the configuration is right. Against the adaptive-scheduler incumbent
the scheduler-free arm remains **within noise of, not better than**, a single
10.36 draw — unchanged from the parent's reading, and still not settled without
replicating the incumbent.

## 2. The deliverable: a complete 1k-step checkpoint history

Every one of the 15 cells carries **30 periodic checkpoints + `last.ckpt`**
(2.72 GB each, ~84 GB per cell, **465 checkpoints / 1.2 TB** total), verified by
`analyze.py`'s `ckpts` column reading 31 for all 15. Resolved config on every
run: `every_n_train_steps: 1000`, `save_top_k: -1`, `save_last: true`.

The parent project trained the same arm with `CKPT_EVERY=5000, save_top_k=1`, so
exactly one periodic checkpoint survives per run and no trajectory is
recoverable. That is why nothing was inherited and all 15 cells were trained
fresh. Measured cost on `kuleshov-compute-03` (A5000): ~59 optimizer steps/min,
~17 min per checkpoint, ~8.5 h per cell — matching the parent's 8.2-8.6 h.

## 3. Stage 1 — LR x R at the closed-form truncation (m = 1.0, seed 1)

GenPPL (entropy):

| lr \ R | 0.05 | 0.1 | 0.5 | 1 |
|---|---|---|---|---|
| **3e-4** | 13.50 (3.70) | 14.33 (3.67) | 12.60 (3.74) | 12.56 (3.83) |
| **1e-3** | 11.89 (3.47) | 12.25 (3.76) | **10.24** (3.74) | 11.80 (3.85) |

| valid PPL (diagnostic only) | 0.05 | 0.1 | 0.5 | 1 |
|---|---|---|---|---|
| 3e-4 | 72.63 | 56.64 | 26.90 | 18.11 |
| 1e-3 | 77.09 | 56.58 | 26.92 | 18.28 |
| tau_max | 4.665 | 3.981 | 2.444 | 1.834 |

**`lr 1e-3` wins at every R, by 1.6 to 2.1 points.** R = 0.5 is an interior
optimum on that row. No cell collapsed.

### 3.1 This contradicts the parent's LR conclusion — and the contradiction is hardware

The parent reported that *"the optimal LR climbs monotonically with R"* — 3e-4
at R <= 0.1, 1e-3 at R in [0.5, 5], 5e-3 at R >= 8 — with a mechanism attached
(small R means a long horizon and many effective denoising steps, so smaller
optimizer steps suit it). On fixed hardware **no such climb exists over
R in [0.05, 1]**: 1e-3 is uniformly better.

The parent's claim rests on its `lr 3e-4, R = 0.05` cell scoring 10.56. That
cell ran on an RTX 6000 Ada. Re-run on an A5000 it scores **13.50**, and 1e-3
(11.89, an A5000 cell in both projects, bit-identical) wins comfortably. §6
shows the parent's LR rows are confounded with node assignment, so at least the
low-R end of its LR ladder is an artifact of which GPU each cell landed on.
**H1 is falsified as stated**, for a reason that has nothing to do with the
model.

## 4. Stage 2 — the truncation axis at the winner (lr 1e-3, R = 0.5)

`TAU_MAX = m * tau*(0.5)`, `tau*(0.5) = 2.4436`:

| m | 0.85 | **1.0** | 1.15 | 1.25 |
|---|---|---|---|---|
| tau_max | 2.0771 | **2.4436** | 2.8102 | 3.0546 |
| b_min = exp(-tau_max) | 0.1253 | **0.0868** | 0.0602 | 0.0471 |
| endpoint noise:signal | 7.94 | **5.27** | 3.55 | 2.74 |
| **GenPPL** | 10.97 | **10.24** | 13.44 | 13.78 |
| entropy | 3.79 | 3.74 | 3.86 | 3.95 |
| valid PPL | 47.27 | 26.92 | 17.06 | 13.70 |

**`m = 1.0` is an interior optimum: the closed-form decode point `tau*(R)` is
optimal on this axis.** +0.72 one step short, +3.20 and +3.53 long. **H2 holds.**
Nothing collapsed — entropy is 3.74-3.95 and rises monotonically with m, so the
optimum is not a low-entropy artifact.

### 4.1 The endpoint invariant predicts admissibility across R; m does not

The operative quantity is the endpoint noise-to-signal ratio
`b_min sqrt(d) / ((1 - b_min) R)`, which at `m = 1` equals `sqrt(d)/C = 5.27`
for **every** R — that is what `alpha_star_euclidean` holds fixed. The parent
set an admissibility threshold of ratio <~ 10 from its R = 0.05 curve, where
`m = 0.85` gives ratio **10.73** and was borderline (entropy 3.514, every
shorter m collapsing outright).

At R = 0.5 the same `m = 0.85` gives ratio **7.94** and is comfortably
admissible (entropy 3.79, GenPPL 10.97). **The same multiplier is safe at one R
and marginal at another; the ratio is what separates them.** The parent could
not make this comparison — it swept truncation at a single R.

### 4.2 Valid PPL is anti-correlated with quality along this axis

| m | 0.85 | 1.0 | 1.15 | 1.25 |
|---|---|---|---|---|
| valid PPL (lower "better") | 47.27 | 26.92 | 17.06 | **13.70** |
| GenPPL (lower better) | 10.97 | **10.24** | 13.44 | 13.78 |

The bound improves **3.4x** across the axis while GenPPL gets 34% *worse*. At
`m = 1.25` valid PPL reports the best value anywhere in this project for the
model that generates the worst text of the four. A longer `tau_max` integrates
the bound over a range where the model is nearly denoised, so the number shrinks
regardless of sample quality. **Ranking these cells by valid PPL gives exactly
the reverse of the truth on the long side.**

## 5. Stage 3 — seed replication, and what it does and does not settle

`lr 1e-3, m = 1.0`, three seeds each, all on `kuleshov-compute-03`:

| cell | seed 1 | seed 2 | seed 3 | mean | sd | entropy |
|---|---|---|---|---|---|---|
| **R = 0.5** | 10.24 | 10.94 | 11.93 | **11.04** | 0.85 | 3.74 / 3.76 / 3.80 |
| R = 1 | 11.80 | 12.07 | 12.42 | **12.10** | 0.31 | 3.85 / 3.88 / 3.82 |

R = 0.5 is better by **1.06** and wins **8 of 9** pairwise seed comparisons, but
with n = 3 that is **not a significant separation**: Welch t = -2.04,
**p = 0.15**; Mann-Whitney U = 1, **p = 0.20** (0.10 is the smallest attainable
p at n = 3 vs 3). The recommendation of R = 0.5 rests on a consistently lower
mean and a verified truncation optimum, not on a demonstrated difference.

**H3 holds, and it bit this sweep too.** Stage 1's headline cell scored 10.24;
its config's mean is 11.04. Selecting on the single draw would have overstated
the result by 0.79 — 0.9 sd. R = 1 is the more *reproducible* configuration
(sd 0.31 vs 0.85); a use case that values predictability over mean quality
should prefer it.

## 6. Training is bit-deterministic given the node; GPU model moves GenPPL as much as R does

All 15 cells here ran on `kuleshov-compute-03` (A5000). Against the parent's
runs of the same configurations, reproduction splits **perfectly** on which GPU
the parent's run happened to land on:

| cell | parent node / GPU | parent | here (A5000) | outcome |
|---|---|---|---|---|
| lr 3e-4, R = 1 | kuleshov-compute-03, **A5000** | 12.56 | 12.56 | **bit-identical** |
| lr 1e-3, R = 0.05 | kuleshov-compute-03, **A5000** | 11.89 | 11.89 | **bit-identical** |
| lr 1e-3, R = 0.1 | kuleshov-compute-03, **A5000** | 12.25 | 12.25 | **bit-identical** |
| lr 1e-3, R = 0.5, seed 3 | kuleshov-compute-03, **A5000** | 11.9278 | 11.9278 | **bit-identical** |
| lr 3e-4, R = 0.05 | thickstun-compute-01, Ada | 10.56 | 13.50 | +2.94 |
| lr 3e-4, R = 0.1 | kuleshov-compute-02, A6000 | 12.12 | 14.33 | +2.21 |
| lr 3e-4, R = 0.5 | thickstun-compute-01, Ada | 11.81 | 12.60 | +0.79 |
| lr 1e-3, R = 0.5 | kuleshov-compute-02, A6000 | 11.53 | 10.24 | -1.28 |
| lr 1e-3, R = 1 | thickstun-compute-01, Ada | 11.72 | 11.80 | +0.09 |
| lr 1e-3, R = 0.5, seed 2 | kuleshov-compute-02, A6000 | 10.86 | 10.94 | +0.08 |

**Ten for ten.** Same GPU model => identical to the last bit of GenPPL, entropy
*and* `val/nll` (e.g. lr 3e-4 R = 1: `val/nll` 2.896428286072464, GenPPL
12.5607328414917, entropy 3.834523916244507, in files written a month apart).
Different GPU model => 0.08 to 2.94 apart, in both directions.

1. **The pipeline has no hidden nondeterminism.** Given (seed, node) a run
   reproduces exactly, `cudnn_benchmark` and TF32 notwithstanding.
2. **A GPU swap perturbs GenPPL by up to 2.94** — the size of the R and LR
   effects being measured, and comparable to the 3-seed sd (0.85 at R = 0.5).
   Perturbation size tracks seed sensitivity: largest at R = 0.05, smallest at
   R = 0.5 / 1. Floating point and seed excite the same instability.

**Consequences.** Both of the parent's LR rows mix Ada / A6000 / A5000, so
neither is a clean R axis (§3.1 shows one conclusion that does not survive).
Every cell of this project is pinned to one node (`sweep.py`, `NODE`).
For future sweeps: **pin the node, or treat hardware as a swept factor.**

### 6.1 A pre-registered prediction, half wrong — a log-globbing error, not the rule

Before the last two stage-1 cells finished, this file recorded that
`lr 1e-3, R = 0.5` should reproduce **11.53 exactly** and `lr 1e-3, R = 1`
should **not** reproduce 11.72.

- R = 1: **held** — 11.7178 -> 11.8048.
- R = 0.5: **failed** — predicted bit-identical, measured 11.5288 -> 10.2447.

The node attribution behind it was wrong, not the rule. The glob
`eflmratr_lr-1e-3_r-0.5_m-1.0_*.log` also matches the **seed-2 and seed-3** logs
(`..._m-1.0_s2_*.log`, `..._s3_*.log`), and `tail -1` selected the seed-3 log,
which ran on `kuleshov-compute-03`. The seed-1 run that produced the stored
result ran on `kuleshov-compute-02`. Re-derived from seed-1 logs only, the cell
is an A6000 cell, a difference is predicted, and a difference occurred — which
is why the table above is 10/10.

Recorded because the failure mode generalises: a wildcard over run logs silently
pulled in a different seed's run, and the resulting claim was confidently stated
and wrong. **Cross-check node attribution against the job id that wrote the
eval JSON, never a glob.**

## 7. Anti-collapse gate and metric hygiene

`setup.md` asks for the optimal GenPPL *without collapsing*, and degenerate
models occupy the **top** of the GenPPL ranking: in the parent sweep a model
emitting a single `!` for all 16384 tokens scored **1.09** against 10.56 for the
genuine optimum, because gpt2-large finds a constant string maximally
predictable. Every ranking in `analyze.py` filters `entropy >= 3.0` first, and
`sweep.py --auto` applies the same gate before choosing the cell to build the
next stage around.

**No cell in this project collapsed** — the entropy range across all 15 is
3.47-3.95. Dropping `lr 5e-3` (which annihilated R <= 1 in the parent) and
staying at `m >= 0.85` kept the whole grid inside the admissible region.

Valid PPL must not select, for three independent reasons, all visible here:
columns are not comparable (each R integrates the bound over its own `tau_max`,
so PPL falls 77 -> 18 across R for mechanical reasons); it barely distinguishes
the learning rates (26.90 vs 26.92 at R = 0.5, where GenPPL differs by 2.36);
and along the truncation axis it is **anti-correlated** with quality (§4.2). It
is retained only as a collapse detector — collapsed cells report `nan` / `inf` /
~1e134.

## 8. Hypothesis scorecard

| | verdict |
|---|---|
| **H1** — reproduces the parent's LR x R surface within seed noise | **falsified**, but by hardware confounding in the parent, not by the model (§3.1, §6) |
| **H2** — `m = 1.0` is a sharp interior optimum; ratio <~ 10 governs admissibility | **confirmed** (§4, §4.1) |
| **H3** — single-seed GenPPL cannot rank the finalists | **confirmed**; cost 0.79 GenPPL here (§5) |
| **H4** — the 1k-step histories separate cells indistinguishable at step 30k | **not tested** — the histories exist (465 checkpoints) but no per-step evaluation was run |

**H4 is the open item.** The artifact is complete and the analysis is cheap:
`scripts/sample/tinystories/eflm_rescale_auto_truncation.sh` with
`RUN_PPL_EVAL=false` against each `{epoch}-{step}.ckpt` is ~3 min per point, so
a full 30-point GenPPL/entropy trajectory is ~1.5 GPU-h per cell.

## 9. Operational record

| item | value |
|---|---|
| jobs | 191729-36 (stage 1), 252857-9 (stage 2), 272061-4 (stage 3), 201004 (babysitter) |
| node | `kuleshov-compute-03` (A5000) for all 15 cells — pinned |
| concurrency | 2 cells at a time; the rest of `thickstun,desa` was held by an unrelated `init_refactor_16d` sweep, left untouched |
| wall clock | Sep 2 09:08 -> Sep 5 07:00, ~2.9 days |
| disk | 465 ckpts, **1.2 TB**; `/share/desa/nfs02` went 6.3 -> 5.2 TB free over the run |
| staging | `sweep.py --auto` opened stages 2 and 3 unattended, gated on `entropy >= 3.0` and 800 GB free |
