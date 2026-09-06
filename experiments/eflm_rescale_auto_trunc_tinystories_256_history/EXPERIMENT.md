# eflm_rescale_auto_trunc_tinystories_256_history — LR x R x truncation, with checkpoint history

Re-run of the truncated autonomous-clock E-FLM hyperparameter search on
TinyStories seq 256, narrowed by `setup.md` to the admissible region and
executed with the **full training-checkpoint history retained** — every 1k
steps, `save_top_k=-1`, 30 checkpoints + `last.ckpt` per run.

Arm: `scripts/{train,sample}/tinystories/eflm_rescale_auto_truncation.sh`
(rescaled E-FLM, autonomous clock, truncated, no adaptive scheduler, plain CE).
Parent project (single final checkpoint only):
`experiments/eflm_rescale_auto_trunc_tinystories_256`.

## Why re-run rather than reuse the parent

The parent trained with `CKPT_EVERY=5000, save_top_k=1`, so exactly one periodic
checkpoint survives per run. Nothing about the training *trajectory* can be
recovered from it. This project's deliverable is the trajectory: a 30-point
checkpoint history per cell, at 1k-step resolution, over the LR x R x truncation
grid — the input any later analysis of training dynamics needs (embedding
geometry over training, e.g. `visualization/codebook_eigen_dist.py`; GenPPL /
entropy vs. step; when and where a cell collapses).

## Task (`setup.md`)

*Vary R, LR and the truncation timestep to find the optimal GenPPL **without
collapsing***, plain CE (`{w/o SNR}`).

## Grid

| knob | values | hydra target |
|---|---|---|
| `LR` | 3e-4, 1e-3 | `optim.lr` |
| `R` | 0.05, 0.1, 0.5, 1.0 | `algo.rho_min = algo.rho_max` |
| `m` | `TAU_MAX = m * tau*(R)` | `noise.tau_max` |
| seed | 1, 2, 3 | `seed` |

On the autonomous clock the noise fraction is `b_t = 1 - alpha_t = exp(-tau)`,
`tau = TAU_MAX (1 - t)` (`noise_schedules.Autonomous`), which is what makes the
bridge drift the time-invariant `v(X) = y - X`. The flow reaches the data only
as `tau -> infinity`, so **TAU_MAX *is* the truncation** — the noise-fraction
floor `exp(-TAU_MAX)`. The closed-form decode point on this clock is

    tau*(R) = -log b*(R) = log(1 + C/R),   C = sqrt(2 log(2(V-1)/delta)) = 5.2575
            = alpha_star_euclidean(V=50257, embed_norm=R, auto_clock=True)

| R | 0.05 | 0.1 | 0.5 | 1.0 |
|---|---|---|---|---|
| **tau*(R)** | **4.6649** | **3.9811** | **2.4436** | **1.8338** |
| b*(R) | 0.0094 | 0.0187 | 0.0868 | 0.1598 |
| % of the untruncated horizon (6.9078) | 68% | 58% | 35% | 27% |

matching the values `setup.md` lists. `m = 1` stops exactly at the decode point;
`m < 1` short of it, `m > 1` past it.

`ALPHA_MAX=null` everywhere. `TruncatedScheduleWrapper` rescales alpha
*affinely*, turning `b_t` into `const + scale*exp(-tau)` — no longer a pure
exponential, so the drift stops being time-invariant and the clock stops being
autonomous. It also pins `b_min = alpha_max` independently of R and m, which
silently voided ~28 cells of the parent sweep (its RESULTS.md §7).

Fixed: `small-sphere-dit` (768 x 12 x 12), `model.init=ngpt`, 30k steps, global
batch 512 (4 GPU x 32 x accum 4), seq 256, bf16, EMA 0.9999, AdamW wd 0,
betas (0.9, 0.999), eps 1e-8, clip 1.0, `algo.snr_weighted_ce=false`.
Eval: `ppl_eval` + `sample_eval`, exact velocity, `top_k_velocity=1`, 180
sampling steps, greedy last step.

## Staging

Fully crossing 4 R x 2 LR x 3 seeds plus a truncation axis is ~28 runs = ~2.4 TB
of checkpoints, so the grid is walked in stages, each conditioned on the last.
`sweep.py` is idempotent, so a stage is just another invocation.

| stage | cells | what it decides |
|---|---|---|
| 1 | 8 — LR x R at `m = 1.0`, seed 1 | the (LR, R) region to spend seeds on |
| 2 | 3 — `m` in {0.85, 1.15, 1.25} at the stage-1 winner | whether `tau*` is the interior optimum here too |
| 3 | 4 — seeds 2, 3 at the two finalist cells | whether the stage-1 ranking survives seed noise |

## Hypotheses

- **H1 (replication).** With the identical protocol modulo checkpoint frequency,
  stage 1 reproduces the parent's LR x R surface within seed noise (sd ~0.5
  GenPPL in stable regimes): 3e-4 best at R <= 0.1, 1e-3 best at R in [0.5, 1],
  every cell admissible (entropy >= 3.0). Falsified if any m = 1.0 cell
  collapses or the LR ranking flips at a fixed R by more than ~1 point.
- **H2 (truncation).** `m = 1.0` is an interior optimum, sharply: the endpoint
  noise-to-signal ratio `b_min sqrt(d) / ((1 - b_min) R)` equals `sqrt(d)/C =
  5.27` for **every** R at `m = 1` (that is what `alpha_star_euclidean` holds
  fixed), and admissibility requires ratio <~ 10. So `m = 0.85` (ratio 10.7)
  should be borderline and `m = 1.25` (ratio 1.6) admissible but worse.
- **H3 (selection).** Single-seed GenPPL cannot rank the finalists: the parent
  measured a 43.6x variance ratio between R = 0.05 (sd 3.55) and R = 0.5
  (sd 0.54). The recommendation must come from the 3-seed means plus the margin
  from the collapse boundary, not from the best single cell.
- **H4 (history, the deliverable).** The 1k-step histories separate cells that
  are indistinguishable at step 30k: collapsed cells should show entropy
  falling monotonically rather than an abrupt failure, and the stable/unstable
  boundary should be visible in the trajectory well before the final step.

## Anti-collapse gate

`setup.md` asks for the optimal GenPPL *without collapsing*, and degenerate
models occupy the **top** of the GenPPL ranking — a model emitting one `!` for
all 16384 tokens scored 1.09 in the parent sweep vs 10.56 for the genuine
optimum, because gpt2-large finds a constant string maximally predictable.
**Any ranking must filter `entropy >= 3.0` first.** Valid PPL corroborates
(collapsed cells report `nan` / `inf` / ~1e134) but must not be used to select:
each R has its own horizon, so the bound integrates over a different range and
falls monotonically with R for reasons unrelated to quality.

## Resources

- **Per cell:** 4 GPUs (RTX 6000 Ada / A6000 / A5000), 8 CPU, 64 GB RAM.
  ~8.5 h train + ~3 min eval, measured on the parent's identical cells.
- **Partitions:** `thickstun,desa`, `--exclude=desa-compute-01` (11 GB 2080 Ti
  OOMs at this sequence length).
- **Disk:** 2.72 GB per checkpoint x 31 = **~84 GB per cell**; ~675 GB for
  stage 1, ~1.3 TB for all three stages. `/share/desa/nfs02` had 6.3 TB free at
  submission — worth re-checking before stage 3.
- **Wall clock:** with ~8 free GPUs (2 concurrent 4-GPU jobs) stage 1 is ~4 x
  8.5 h ≈ 1.5 days; all three stages ≈ 3 days, longer while the concurrent
  `init_refactor_16d` jobs hold GPUs.

## Deliverables

- `outputs/eflm_rescale_auto_trunc_tinystories_256_history/{run}/checkpoints/` —
  30 x `{epoch}-{step}.ckpt` at 1k-step spacing, plus `last.ckpt`.
- `outputs/.../{run}/eval/{ppl.json, samples_genppl.json}`.
- `analyze.py` → the LR x R and truncation tables.
- `RESULTS.md` → numerical tables, the collapse audit, and the recommendation.
