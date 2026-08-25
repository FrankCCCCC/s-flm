# eflm_rescale_auto_trunc_tinystories_256 — LR x R x truncation for the truncated autonomous clock

Hyperparameter search for the one arm that won the parent sweep's
fixed-schedule comparison: **rescaled E-FLM on the autonomous clock, truncated**
(`scripts/{train,sample}/tinystories/eflm_rescale_auto_truncation.sh`,
no adaptive scheduler, plain CE). Source spec:
`experiments/eflm_rescale_auto_tinystories_256/setup.md`, section
*"Rescale + Autonomous EFLM * {w/ Trunc}"*.

The parent sweep (`eflm_rescale_auto_tinystories_256`) held **lr = 3e-4** and
pinned the truncation at the closed-form decode point tau*(R). This sweep opens
all three knobs.

## The three knobs

On the autonomous clock the noise fraction decays exponentially,

    b_t := 1 - alpha_t = (1 - eps) exp(-tau),    tau = TAU_MAX (1 - t)

(`noise_schedules.Autonomous`), which is what makes the bridge drift the
time-invariant `v(X) = y - X` and every Euler step advance the same `d_tau`.
The flow reaches the data only as `tau -> infinity`, so **TAU_MAX *is* the
truncation** — it is the noise-fraction floor `exp(-TAU_MAX)`. Hence:

| knob | values | meaning |
|---|---|---|
| `LR` | 3e-4, 1e-3, 5e-3 | AdamW lr (`optim.lr`) |
| `R` | 0.05, 0.1, 0.5, 1, 5, 8, 16, 28 | pinned embedding norm (`algo.rho_min = rho_max = R`) |
| `m` | `TAU_MAX = m * tau*(R)` | truncation, **relative to the closed-form decode point** |

with the Eq.-17 decode point on this clock

    tau*(R) = -log b*(R),  b*(R) = 1 - alpha_star_euclidean(V=50257, R)
            = log(1 + C/R),  C = sqrt(2 log(2(V-1)/delta)) = 5.2575

| R | 0.05 | 0.1 | 0.5 | 1 | 5 | 8 | 16 | 28 |
|---|---|---|---|---|---|---|---|---|
| b*(R) | 0.0094 | 0.0187 | 0.0868 | 0.1598 | 0.4874 | 0.6034 | 0.7527 | 0.8419 |
| **tau*(R)** | **4.6649** | **3.9811** | **2.4436** | **1.8338** | **0.7186** | **0.5051** | **0.2841** | **0.1721** |

`m = 1` stops exactly at the decode point; `m < 1` stops before it; `m > 1`
runs past it toward the untruncated horizon `-log(1e-3) = 6.9078`. R = 0.05 and
0.1 are new to this sweep and are where `tau*` starts to approach that
untruncated horizon — i.e. where truncation should stop paying.

**Why `m` and not `noise.alpha_max`.** `TruncatedScheduleWrapper` rescales
alpha *affinely*, which turns `b_t` into `const + scale * exp(-tau)` — no
longer a pure exponential, so the drift stops being time-invariant and the
clock stops being autonomous. Scaling TAU_MAX keeps `b_t` exponential and moves
only the endpoint. (`ALPHA_MAX` is declared but unused in
`scripts/train/tinystories/eflm_rescale_auto_truncation.sh`.)

## What is already known (inherited, not re-run)

The parent sweep's `auto_trunc` arm **is** this grid's `(lr 3e-4, m = 1.0)`
slice — same script modulo the output-dir default, same protocol. Those six
cells are symlinked into `outputs/eflm_rescale_auto_trunc_tinystories_256/`
by `sweep.py`, so the results table is complete without recompute:

| R | 0.5 | 1 | 5 | 8 | 16 | 28 |
|---|---|---|---|---|---|---|
| GenPPL | **11.81** | 12.56 | 17.25 | 18.29 | 19.76 | 18.19 |
| entropy | 3.82 | 3.84 | 3.85 | 3.89 | 3.93 | 3.90 |

Reference points, same protocol (30k steps, batch 512, seq 256, 180 sampling
steps, exact velocity, top_k_v = 1, greedy last):

| run | GenPPL |
|---|---|
| autonomous + trunc + adaptive, R=0.5 (parent best) | 10.74 |
| log-linear + trunc + adaptive, R=1 (`eflm_rescale_tinystories_256`) | **10.36** |
| autonomous, untruncated, R=0.5 | 18.90 |
| raw-norm naive E-FLM baseline | 34.58 |

**Loss weighting is fixed to plain CE.** `setup.md` for this project specifies
`{w/o SNR}`, so `algo.snr_weighted_ce=false` everywhere and the swept axes are
LR, R and the truncation. The parent sweep's Eq.-16 cells (lr 3e-4, m = 1.0,
all six R) are the reference for that choice — GenPPL 18.17 / 18.61 / 21.98 /
21.74 / 20.54 / 22.13, worse than plain CE at every R:

| R | 0.5 | 1 | 5 | 8 | 16 | 28 |
|---|---|---|---|---|---|---|
| CE | **11.81** | 12.56 | 17.25 | 18.29 | 19.76 | 18.19 |
| SNR (Eq. 16) | 18.17 | 18.61 | 21.98 | 21.74 | 20.54 | 22.13 |

Mechanism, for the record: on this clock `|alpha'| = TAU_MAX * b`, so the
Eq.-16 weight collapses to `w(b) = TAU_MAX (1 - b) / b^2` and its dynamic range
over the horizon is fixed by the truncation —

| R | 0.05 | 0.1 | 0.5 | 1 | 5 | 8 | 16 | 28 |
|---|---|---|---|---|---|---|---|---|
| range at m=1 | 1.1e7 | 2.8e6 | 1.2e5 | 3.3e4 | 2.2e3 | 1.1e3 | 438 | 224 |

Sudoku found the weighting helps only below ~1e3, which at m = 1 holds just at
R = 16 and 28 — exactly where the CE-vs-SNR gap above is smallest (0.8 and
3.9 points, vs 6.4 at R = 0.5). Consistent, but it never wins, so CE it is.
(`sweep.py --weights snr` can pull those cells in for a side-by-side.)

## Hypotheses

**H1 (R has an interior optimum below 0.5).** GenPPL is monotone decreasing in
R over 28 -> 0.5, but as R -> 0 the decode point tau*(R) -> infinity, and the
untruncated horizon is catastrophic (18.90 at R=0.5, 100.8 at R=28). So the R
curve must turn. Prediction: R = 0.1 or 0.05 is no better than R = 0.5, i.e.
the optimum is R ~ 0.5 and the curve is U-shaped with a flat left arm.

**H2 (larger LR helps at 30k steps).** The parent's 3e-4 was inherited from the
log-linear sweep, not tuned. A 12-layer 768-wide DiT at global batch 512 for
only 30k steps is under-trained; the concurrent AR/MDLM/DUO sweep at the same
seq/batch searches the same {3e-4, 1e-3, 5e-3} ladder. Prediction: 1e-3 beats
3e-4; 5e-3 is at or past the stability edge (watch for NaN / entropy collapse).

**H3 (the optimum truncation sits at or slightly below tau*(R)).** On sudoku
(`eflm_rescale_auto_sudoku`) the best horizon landed within one grid step of
tau*(R) and, at two of three norms, *below* it (R=8: 0.25 vs 0.344 predicted;
R=16: 0.1 vs 0.187). Prediction: best m in [0.5, 1.0], and the response is
sharply asymmetric — m > 1 degrades fast (that direction is the untruncated
failure mode), m < 1 degrades slowly.

**H4 (a tuned no-scheduler arm reaches the adaptive-scheduler incumbent).**
If H2 and H3 both pay, the best cell should close the 11.81 -> 10.36 gap to
the log-linear + adaptive incumbent *without* any adaptive scheduler.

## Design

- Data: TinyStories, 475M train / 5M val (seed 42), seq **256**.
- Model: `small-sphere-dit` (768 wide, 12 blocks, 12 heads), `init=ngpt`.
- Training: 30k steps, global batch 512 (4 GPUs x 32 x accum 4), bf16,
  EMA 0.9999, AdamW wd 0, betas (0.9, 0.999), eps 1e-8, clip 1.0, plain CE,
  1 seed.
- Eval: `ppl_eval` (flow-bound valid PPL) + `sample_eval` (GenPPL via
  gpt2-large retokenisation, entropy, samples); exact velocity,
  top_k_velocity = 1, 180 steps, greedy last.
- **Staged coordinate search**, not the full 3-way grid (72 cells is ~430
  GPU-days of a shared 28-GPU pool):

  - **Stage 1 — LR x R at m = 1.0**: 3 x 8 = 24 cells, 6 inherited,
    **18 submitted**. Establishes the LR ladder over the whole R grid and adds
    the two new small norms.
  - **Stage 2 — truncation**: m in {0.5, 0.7, 0.85, 1.25} at the winning LR and
    the top two R, **8 cells**. Grid is denser below 1 per H3.
  - **Stage 3 — confirm**: if either stage lands on a grid edge, extend it;
    otherwise refine (R, m) jointly around the winner. ~4-6 cells.

  Coordinate descent is sound here because LR is near-orthogonal to the
  schedule knobs, while R and the truncation are coupled through tau*(R) —
  which is exactly why the truncation is swept as the **relative** multiplier
  m, holding the R-dependence of the endpoint fixed.

## The resolution floor: 1 seed cannot see a ~1-point GenPPL difference

`experiments/seed_errbar_tinystories_256` measures seed-to-seed GenPPL spread at
**this exact protocol** (seq 256, global batch 512, 30k steps, 180 sampling
steps, same eval): 3 seeds per config,

| config | mean | sd | range |
|---|---|---|---|
| sfm  | 11.218 | **0.470** | 10.676 - 11.500 |
| hflm | 11.252 | **0.845** | 10.705 - 12.225 |

(Different algorithm arms, but same model scale, data, schedule length and
eval — so this is the right order of magnitude for GenPPL seed noise here.)

**Working figure: sigma ~ 0.5-0.85, so 2 sigma ~ 1.0-1.7 GenPPL.** Consequences
this sweep must respect, since every cell is 1 seed:

- A gap below ~1.7 is **not a result**. It gets reported as a tie.
- That covers more comparisons than is comfortable: lr 3e-4 vs 1e-3 at R=0.5
  (11.81 vs 11.75, 0.06 = 0.1 sigma) is a dead tie; R=0.5 vs R=1 (11.81 vs
  12.56, 0.9 sigma) is unresolved; and even this arm vs the log-linear+adaptive
  incumbent (11.81 vs 10.36, 1.7-2.9 sigma) is only marginal.
- What *is* resolvable: R=0.5 vs R>=5 (5.4+ points, >6 sigma), and every
  collapse (entropy < 3.0 is unambiguous, and those cells miss by 50-100 points).
- Therefore **the final winner must be seed-replicated**, not read off a
  single-seed grid maximum. Stage 3 budget is reserved for 2 extra seeds on the
  top configs rather than for more single-seed grid points, and any claim of a
  "best" cell is stated with its margin in sigma.

## Success criteria

Task, per `setup.md`: *vary R, LR and the truncation timestep to find the
optimal GenPPL **without collapsing***. "Collapsing" is the degenerate/repetitive
failure GenPPL alone cannot see — a low GenPPL reached by emitting
low-diversity text. It is caught by sample entropy, so **entropy >= 3.0 is a
hard admissibility gate**: a cell below it is not eligible to be the optimum
regardless of its GenPPL. `analyze.py` flags such cells with ⚠. (For scale,
every cell of the parent sweep scored 3.25-3.94; the untruncated R=28 arm was
the lowest at 3.56, so this gate bites exactly where the horizon is mis-set.)

1. Stage 1 identifies a best LR that is stable across R (if the LR optimum
   flips with R, stage 2 must be run at both).
2. The R curve turns: some R < 0.5 is no better than R = 0.5 (H1), so the
   reported optimum is interior to the grid, not at its edge.
3. Stage 2 finds `m != 1.0` beating `m = 1.0` at matched (LR, R), or confirms
   the closed-form tau*(R) is already optimal — either is a result.
4. Best cell vs the 11.81 same-arm baseline and the 10.36 incumbent (H4).
5. Entropy >= 3.0 on every reported cell (degeneracy bar): GenPPL is read only
   together with entropy.

## Compute

- 4 GPUs/cell, `thickstun,desa` (exclude `desa-compute-01`), 8 cpus, 64G,
  2-day walltime. Measured cost from the parent sweep: **4.3 h/cell** on
  thickstun-compute-01 (RTX 6000 Ada), **8.6 h/cell** on kuleshov-compute-03
  (A5000); eval adds ~2-10 min.
- ~28 shared GPUs = ~7 concurrent cells => stage 1 (18 cells) ~3 waves,
  ~18-24 h; stage 2 ~8-12 h. Total ~1.5-2 days wall clock.
- Outputs: `outputs/eflm_rescale_auto_trunc_tinystories_256/eflmratr_lr-{lr}_r-{R}_m-{m}/`,
  eval in `eval/{ppl.json, samples_genppl.json}`. Checkpoints KEPT.

## Run

    python experiments/eflm_rescale_auto_trunc_tinystories_256/sweep.py --dry-run
    python experiments/eflm_rescale_auto_trunc_tinystories_256/sweep.py            # stage 1
    python .../sweep.py --lrs 1e-3 --rhos 0.5 1 --mults 0.5 0.7 0.85 1.25          # stage 2
    python experiments/report.py eflm_rescale_auto_trunc_tinystories_256
