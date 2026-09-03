# simpflm_auto_trunc_tinystories_256 — SimpFLM on the truncated autonomous clock

Source spec: `experiments/simpflm_auto_trunc_tinystories_256/setup.md`.
Arms: `scripts/{train,sample}/tinystories/simpflm_{auto_truncation,truncated}{,_adaptive}.sh`.
Sweep: `sweep.py`.

## What SimpFLM is

**E-FLM with the word-embedding matrix replaced by a diagonal.** Every
"embedding" is a one-hot simplex vertex, `E = R * I_V`, so the Gaussian
Euclidean flow

    x_t = (1 - b_t) e + b_t z,    z ~ N(0, I),    b_t = 1 - alpha_t

runs in the one-hot (logit) space R^V instead of a learned d-dim space.
`algo.SimpFLM` subclasses `algo.EFLM` and overrides exactly the places the
embedding table enters:

| override | what it becomes |
|---|---|
| `_clean_embeddings(x)` | rows of the diagonal, `R * one_hot(x)` (E-FLM: `get_rescaled_embeddings`) |
| `_sc_embed_table()` | the diagonal `R * I_V` itself |
| `_validate_configuration()` | rejects the E-FLM knobs that no longer mean anything |

Everything else — the interpolant, the CE loss, the noise schedules, the Euler /
Euler–Maruyama sampler — is inherited. The `small-flm` (`flm-dit`) backbone
projects the `[B, L, V]` blend down to the model width through
`FLMEmbeddingLayer` (a linear map, no softmax), the same input path the existing
FLM baseline uses.

`R` is the **radius of the simplex sphere**: E-FLM's radial rescale with
`algo.rho_min == algo.rho_max == R` (enforced — a *range* has nothing to clamp
when all V rows already share the norm 1) pins every vertex to norm R, so the
diagonal is exactly `R * I_V`. It is the only geometric knob.

The sampler is `samplers.SimpFLMSampler`, a subclass of `EFLMSampler` that
overrides the one method touching `E` (`_velocity`) so that `p @ E` collapses to
`R * p` — the V x V diagonal is never materialized (10 GB at V = 50257).
`tests/test_simpflm.py` checks that substitution against E-FLM's *own* code
path evaluated on the materialized diagonal at small V, in both places
(interpolant endpoint and sampler drift), and pins that the sampling loop never
materializes it.

**One deliberate difference from the E-FLM arms:** they run the interpolant and
the sampler in float64 (`algo.slerp_precision=float64`, `sampler.use_float64`),
SimpFLM runs both in float32. At V = 50257 a float64 `[B, L, V]` tensor is
~3.3 GB where float32 is 1.6 GB, and the flow is a plain lerp with no
catastrophic cancellation (E-FLM's float64 is inherited from S-FLM's `acos`/
`sin` slerp, which does have one). So "same protocol" covers the sampler
schedule, step count, velocity mode and decode rule — not the arithmetic
precision.

Two consequences distinguish it from E-FLM:

1. **Nothing about the geometry is learned.** The vertices are orthogonal with
   a fixed norm, so there is no embedding drift, no `renormalize_weights`, and
   no self-conditioning path (flm-dit has no `W_in`/`W_sc`). All three are
   rejected at construction.
2. **The Eq.-17 decode point applies verbatim.** For orthogonal vertices the
   impostor score `<x_t, e_v> = R b_t z_v` is *literally* Gaussian, so the union
   bound behind `noise_schedules.alpha_star_euclidean` applies directly instead
   of through E-FLM's sub-Gaussian estimate of random directions in d dimensions
   (`test_decode_point_holds_the_impostor_union_bound`).

## The truncation, and why it is TAU_MAX

With R = 1 (setup.md) and V = 50257, delta = 0.1:

    C          = sqrt(2 log(2 (V - 1) / delta))                 = 5.2575
    alpha*(1)  = C / (1 + C)                                    = 0.8402
    t* = b*(1) = 1 - alpha*                                     = 0.1598
    tau*(1)    = -log b*  = log(1 + C)                          = 1.8338

On the autonomous clock (`noise=autonomous`) the noise fraction decays
exponentially, `b_t = (1 - eps) exp(-tau)` with `tau = TAU_MAX (1 - t)`, which
is what makes the bridge drift the time-invariant `v(X) = y - X` and every Euler
step advance the same `d_tau`. The flow reaches the data only as
`tau -> infinity`, so **TAU_MAX *is* the truncation** — the noise-fraction floor
is `exp(-TAU_MAX)`. The swept knob is therefore the **relative** multiplier `m`:

| m | 0.7 | 0.85 | **1.0** | 1.25 | 1.5 |
|---|---|---|---|---|---|
| `TAU_MAX = m tau*(1)` | 1.2836 | 1.5587 | **1.8338** | 2.2922 | 2.7507 |
| noise floor `exp(-TAU_MAX)` | 0.2770 | 0.2104 | **0.1598** | 0.1010 | 0.0639 |
| signal level `alpha_t(0)` | 0.7233 | 0.7898 | **0.8404** | 0.8991 | 0.9362 |

`m = 1` stops exactly at the decode point; `m > 1` runs past it toward the
untruncated horizon `-log(1e-3) = 6.9078` (`m = 3.77`).

**Why `m` and not `noise.alpha_max`.** `TruncatedScheduleWrapper` rescales alpha
*affinely*, which turns `b_t` into `const + scale * exp(-tau)` — no longer a pure
exponential, so the drift stops being time-invariant and the clock stops being
autonomous. Scaling TAU_MAX keeps `b_t` exponential and moves only the endpoint.
(Verified: the fitted adaptive schedule of a `TAU_MAX=1.834` run tops out at
`alpha = 0.8404`, exactly `alpha_t(0)` above.) On the *log-linear* arms
(`simpflm_truncated{,_adaptive}.sh`) there is no autonomy to preserve, so there
`ALPHA_MAX = alpha*(R) = 0.840` is the right knob.

## The three schedulers, and how they compose

| piece | config | role |
|---|---|---|
| **auto** | `noise=autonomous`, `noise.tau_max` | exponential `b_t` -> time-invariant drift, constant Euler step |
| **trunc** | `noise.tau_max = m tau*(R)` (auto) / `noise.alpha_max = alpha*(R)` (log-linear) | stop the flow at the decode point |
| **ada** | `noise=*-adaptive` | spline remap of `t` onto where `|dL/dt|` is largest |

`ada` sits *on top of* the truncated schedule (`AdaptiveSchedule(base_schedule=…)`),
so it moves only the density of visited noise levels — the decode point is
unchanged. It requires the MDLM time convention
(`algo.invert_time_convention=false`), which is SimpFLM's default and is
asserted in `EFLM._validate_configuration`.

Note for eval: the fitted remap ships in the checkpoint (`noise.alpha_vals` /
`noise.has_schedule`), so an `_adaptive` run must be evaluated with the
`_adaptive` sample script — evaluating it under the plain schedule silently uses
the wrong `alpha_t`.

## Design

- Data: TinyStories, 475M train / 5M val (seed 42), seq **256**.
- Model: `small-flm` (`flm-dit`, 768 wide, 12 blocks, 12 heads) — **169.6M**
  params (larger than the E-FLM arms' `small-sphere-dit`: the `[V, d]` input
  projection is a separate parameter from the `[d, V]` output head).
- Training: 30k steps, global batch 512 (4 GPUs x 32 x accum 4), bf16,
  EMA 0.9999, AdamW wd 0, betas (0.9, 0.999), eps 1e-8, clip 1.0, **plain CE**.
- Eval: `ppl_eval` (flow-bound valid PPL) + `sample_eval` (GenPPL via gpt2-large
  retokenization, entropy, samples); exact velocity, `top_k_velocity = 1`,
  180 steps, greedy last.
- **Staged coordinate search**, not the full grid:
  - **Stage 1 — LR at m = 1.0**, arm `trunc`: 3 cells, setup.md's ladder
    {3e-4, 1e-3, 5e-3}.
  - **Stage 2 — truncation**: m in {0.7, 0.85, 1.25, 1.5} at the winning LR,
    4 cells. Denser below 1 (see H1).
  - **Stage 3 — adaptive + replication**: `trunc_ada` at the winning (LR, m),
    plus seeds 2 and 3 on the top two configs. ~6 cells.

## Hypotheses

**H1 (the closed-form decode point is close to optimal, and the response is
asymmetric).** On sudoku (`eflm_rescale_auto_sudoku`) and TinyStories
(`eflm_rescale_auto_trunc_tinystories_256`) the best horizon landed within one
grid step of `tau*` and, more often, slightly *below* it. Prediction: best m in
[0.7, 1.0]; m > 1 degrades faster than m < 1, because that direction runs toward
the untruncated failure mode.

**H2 (SimpFLM's decode point should be *more* reliable than E-FLM's).** E-FLM's
`tau*` assumes embeddings uniform on a sphere and is only as good as that
approximation — and E-FLM's embeddings are trained, so the assumption decays.
SimpFLM's vertices are orthogonal by construction and never move. Prediction:
the m-response curve is *sharper* around m = 1 than the corresponding E-FLM
curve, i.e. the closed form predicts the empirical optimum more tightly.

**H3 (larger LR helps at 30k steps).** Same reasoning as the E-FLM sweep: a
12-layer 768-wide DiT at global batch 512 for 30k steps is under-trained.
Prediction: 1e-3 beats 3e-4; 5e-3 is at or past the stability edge.

**H4 (ada pays less here than for E-FLM).** The adaptive remap's job is to fix a
mismatched loss profile. The autonomous clock already equalizes the Euler step,
and SimpFLM's fixed geometry removes the embedding-norm drift that makes the
E-FLM loss profile move during training. Prediction: `trunc_ada` is within noise
of `trunc` at the same (LR, m) — itself a result, since for E-FLM the adaptive
arm was worth ~1 GenPPL point.

## The resolution floor: 1 seed cannot see a ~1-point GenPPL difference

`experiments/seed_errbar_tinystories_256` measures seed-to-seed GenPPL spread at
**this exact protocol** (seq 256, global batch 512, 30k steps, 180 sampling
steps, same eval): sfm 11.218 +/- 0.470, hflm 11.252 +/- 0.845 over 3 seeds.

**Working figure: sigma ~ 0.5-0.85, so 2 sigma ~ 1.0-1.7 GenPPL.** Every
stage-1/2 cell is 1 seed, so:

- A gap below ~1.7 GenPPL is **not a result** — it is reported as a tie.
- What *is* resolvable: a mis-set horizon (the untruncated E-FLM arms missed by
  6-90 points) and any entropy collapse.
- The final winner must be **seed-replicated** (stage 3), and any "best cell"
  claim is stated with its margin in sigma.

## Success criteria

1. SimpFLM trains without collapse and its GenPPL is reported next to the E-FLM
   arms on the identical protocol.
2. Stage 1 identifies an LR; stage 2 either finds `m != 1.0` beating `m = 1.0`
   at matched LR, or confirms the closed-form `tau*` is already optimal —
   either is a result (H1/H2).
3. Stage 3 resolves whether the adaptive remap pays on top of auto+trunc (H4).
4. **Entropy >= 3.0 is a hard admissibility gate** on every reported cell: low
   GenPPL reached with low entropy is degenerate/repetitive collapse, not
   quality. GenPPL is read only together with entropy.

## Compute

Measured on one RTX 6000 Ada (48 GB), seq 256, per-GPU batch 32:

| quantity | measured |
|---|---|
| step time (micro-batch 32 x 256) | 180 ms |
| throughput | 45.5k tok/s/GPU |
| peak memory (fwd+bwd+AdamW+EMA) | 13.4 GiB at bs 32 (8.1 at 16, 5.4 at 8) |

**Measured in situ (4-GPU DDP, run 387972 on kuleshov-compute-02):** 2.95
micro-batches/s = 24.2k tok/s/GPU, i.e. **1.9x slower per GPU than the
single-GPU probe above**, reaching optimizer step 5000 in 2 h 24 min. So the
real figure is **node-dependent**: 5.7 it/s on thickstun-compute-01 (RTX 6000
Ada) = **~5.8 h/cell**, but 2.95 it/s on kuleshov-compute-03 (A5000) =
**~11.3 h/cell** -- a 2x spread, so schedule estimates should quote the slower
number. Stage 1 (3 cells) ~ 100-170 GPU-h; stages 1-3 (~13 cells) ~ 450-730 GPU-h.

The gap is DDP communication, not the algorithm: 169.6M params means a 678 MB
gradient all-reduce per optimizer step, and the SLURM boilerplate this repo
prescribes sets `NCCL_P2P_DISABLE=1` (to avoid NCCL hangs), which routes that
all-reduce through host memory instead of P2P. Worth an A/B before scaling the
sweep up — the boilerplate was written for the smaller sphere-dit arms, whose
[V, d] input projection is tied to the output head.

The `[B, L, V]` activations are what make SimpFLM heavier per token than E-FLM
(the flow lives in R^50257, not R^768); 13.4 GiB at batch 32 still fits the
24 GB A5000 nodes, so the whole pool is usable.

## GenPPL / entropy frontier evaluation

`setup.md`'s second section ("Frontier Line Evaluation") asks for the S-FLM
paper Fig. 10 protocol: for each method in {MDLM, DUO, FLM, **Auto + Trunc +
SimpFLM**} and each pretrained seed in {1, 2, 3}, sweep

* NFE in {1, 4, 8, 16, 32, 64, 128, 256}
* T in {0.50, 0.55, ..., 1.20} (15 values)

at 512 samples per cell, exact velocity, `top_k_velocity = 1`, greedy last
step; then draw Gen. PPL (y) vs per-sample unigram entropy (x), one panel per
NFE, as the mean over seeds with a +/- 1 sd band.

**This is run inside the existing `naive_ar_tinystories_s256` project**, per
`setup.md` ("output in `experiments/naive_ar_tinystories_s256`"), not here:
that project already owns the frontier orchestration
(`frontier_sweep.py`, 1080 completed cells for MDLM / DUO / FLM) and the
plotting (`visualization/genppl_entropy_frontier_line.py`). Adding a fourth
method to a finished 3-method grid is the whole task; nothing was rebuilt.

Changes made to that shared machinery:

| change | why |
|---|---|
| `METHODS` += `simpflm`; `EVAL_BS['simpflm'] = 16` | SimpFLM carries a dense [B, L, V] sampler state, like FLM |
| `SCRIPT` map | SimpFLM's arm is `simpflm_auto_truncation.sh`, not `simpflm.sh` |
| `EXTRA_ENV['simpflm'] = 'RHO=1.0 TAU_MAX=1.8338'` | the R = 1 decode point `setup.md` names |
| `TMPDIR` export + `trap` | the ENOSPC trap documented in RESULTS.md |
| `--methods` / `--seeds` filters + checkpoint guard | stage seeds as their pretrained models land, instead of burning allocations on missing checkpoints |

**Configuration choice.** The frontier fixes LR = 1e-3 for every method
(`frontier_sweep.py`, from that project's RESULTS.md Sec. 3.4), so SimpFLM is
evaluated at **LR 1e-3, R = 1.0** — not the R = 2.0 cell that scored marginally
best in the grid above. Matched LR across methods is what makes the frontier
comparable, R = 1 is `setup.md`'s reference value, and the R=1-vs-R=2 gap was
inside the noise floor (F2) anyway.

**Seeds.** The grid above trained seed 1 only; `setup.md` requires {1, 2, 3},
so seeds 2 and 3 are trained at (1e-3, R=1.0, m=1.0) by the project sweep
(`--seeds 2 3`) and symlinked into the frontier project's naming
(`m-simpflm_lr-1e-3_sd-{1,2,3}`) so the plot script's path regex matches.

**Allocation and wall clock.** 3 seeds x 8 NFE = 24 SLURM jobs, 1 GPU each,
every job walking its 15 temperatures sequentially = 360 cells. Cost per job
scales as NFE x 512 samples, so the NFE-256 job dominates (~1.5-2 h) and NFE<=16
is minutes; ~3-4 GPU-h per seed, ~10 GPU-h total for the frontier, plus 2 x
~6-11 h on 4 GPUs for the two extra pretrained seeds.

**Expected shape of the result (pre-registered).** F6 found `top_k_velocity`,
not temperature, is SimpFLM's dominant entropy knob, and the frontier protocol
pins `top_k_velocity = 1`. So SimpFLM's curve is expected to be SHORT and
concentrated at low entropy (~3.7-3.9) rather than spanning DUO's range
(~4.3-4.4), and may not overlap the other methods across the full x-axis. If
that holds, the honest reading is that SimpFLM cannot reach the high-entropy
region under this protocol at all — which is a property of the one-hot velocity
target, not a plotting artifact.

## Results

`RESULTS.md` is produced by `experiments/report.py simpflm_auto_trunc_tinystories_256`
once cells complete. **Caveat:** `experiments/report.py` hardcodes
`REPO = '/share/thickstun/sychou/workspace/research/s-flm'` (a different
checkout), so it must be pointed at this one before it can see these cells —
pre-existing for every experiment here, not specific to this project. **No runs have been submitted yet** — the arm is implemented
and smoke-verified (train -> checkpoint -> ppl_eval -> sample_eval, plain and
adaptive schedulers), but the sweep is awaiting a go-ahead.
