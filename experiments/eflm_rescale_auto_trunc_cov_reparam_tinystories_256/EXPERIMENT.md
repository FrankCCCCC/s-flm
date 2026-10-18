# eflm_rescale_auto_trunc_cov_reparam_tinystories_256 — experiment design

Spec: `setup.md` (this folder). Method notes: `../eflm_rescale_auto_trunc_cov_reparam_sudoku/REPARAM.md`
(Gaussian readout), `docs/truncation.md`, `adaptivesched_refit_bug.md` (known AdaptiveSchedule bugs, not fixed).

## Question

With the Bayes-reparametrised readout (`algo.forward_type=gaussian`), which noise schedule (autonomous + |alpha'|
change-of-variables weight, autonomous + adaptive, log-linear + adaptive + truncation), embedding norm R and LR
give the best TinyStories GenPPL without collapse (entropy >= 3.0), with every schedule truncated at the
closed-form decode point of its R?

## Grid (36 cells = 3 noise x 4 R x 3 LR, 1 seed)

| noise (tag `ns-`) | schedule | truncation | CE weight |
|---|---|---|---|
| `auto` | `noise=autonomous` (`alpha_max` null) | `tau_max = tau*(R)` | `noise_sched_cov_weight=true` |
| `autoada` | `noise=autonomous-adaptive` | `tau_max = tau*(R)` | plain CE |
| `llada` | `noise=log-linear-adaptive` | `alpha_max = alpha*(R)` | plain CE |

| R | 0.1 | 0.5 | 1.0 | 8.0 |
|---|---|---|---|---|
| `tau*` = `alpha_star_euclidean(50257, R, auto_clock=True)` | 3.9811 | 2.4436 | 1.8338 | 0.5051 |
| `alpha*` = `alpha_star_euclidean(50257, R)` | 0.9813 | 0.9132 | 0.8402 | 0.3966 |

LR in {3e-4, 1e-3, 5e-3}. Fixed: `forward_type=gaussian`, small-sphere-dit 768/12/12, ngpt init, seq 256,
30k steps, global batch 512 = **4 GPUs x 32 x accumulation 4** for every arm (required for the adaptive arms,
see the refit note), bf16, EMA 0.9999, AdamW (wd 0, betas 0.9/0.999, eps 1e-8), clip 1.0, seed 1.
Eval (1 GPU): `ppl_eval` -> `eval/ppl.json` (denoising-CE bound, not an AR PPL) and `sample_eval` -> 
`eval/samples_genppl.json` (64 samples, exact velocity, `top_k_velocity=1`, 180 steps, greedy last; GenPPL =
gpt2-large retokenised, read together with entropy).

Run dirs: `outputs/eflm_rescale_auto_trunc_cov_reparam_tinystories_256/ns-{noise}_r-{R}_lr-{lr}/`;
SLURM job name = `eflm_rescale_auto_trunc_cov_reparam_tinystories_256_<tag>`.

## Reference numbers (naive readout, same protocol)

| config | GenPPL | source |
|---|---|---|
| autonomous + trunc tau*, R=0.5, lr 1e-3, plain CE | 11.44 ± 0.54 (3 seeds), entropy 3.82 | `eflm_rescale_auto_trunc_tinystories_256` |
| autonomous + trunc + adaptive, R=0.5 | 10.74 (1 seed) | same, parent sweep |
| log-linear + trunc + adaptive, R=1 | 10.36 (1 seed) | project incumbent |

Single-seed GenPPL spread in stable regimes is ~0.5-0.9 (and ~3.5 near the collapse boundary), so
differences under ~1 GenPPL between cells of this 1-seed grid are not resolved.

## Caveats (pre-launch audit, minor, pre-existing, accepted)

- The 4 DDP ranks share one CUDA RNG stream, so they draw the same Gaussian noise and dropout masks (as in all earlier
  TS sweeps, so comparisons with them still hold).
- `ppl.json` val/ppl is averaged over each arm's own (adapted / truncated) alpha measure: it is not a bound and is not
  comparable across noise arms or R. GenPPL + entropy are the deliverable.

## Hypotheses

- **H1.** The Gaussian readout matches or beats the naive-readout references at matched schedule and R.
- **H2.** The cov weight moves training mass toward the noisy end; on TinyStories flatter time weights have won on
  sample quality before, so `auto` (with cov) may trail the adaptive arms.
- **H3.** R = 0.1 (tau* 3.98, beta_t up to ~2800) is the most collapse-prone end of the grid.

## Compute

- 4 GPUs per cell, `thickstun,desa` excluding `desa-compute-01` (2080 Ti), 16 CPUs, 128 GB, 3-day limit, `--requeue`.
- Smoke job 892235 (RTX 6000 Ada, 1 GPU x 32): peak 18 GB with the Gaussian readout (16 GB naive), so the 24 GB
  A5000 fits; the readout costs ~+20% step time. Previous naive cost: 4.3 h/cell on the Ada, 8.6 h on the A5000,
  so ~5-10 h per cell here + ~10 min eval. 36 cells ≈ 1100 GPU-h; with ~28 shared GPUs (busy with other jobs)
  that is ~2-4 days of wall clock. Submission order: LR 1e-3 first, then 3e-4, then 5e-3.

## Run

    python experiments/eflm_rescale_auto_trunc_cov_reparam_tinystories_256/sweep.py --dry-run
    python experiments/eflm_rescale_auto_trunc_cov_reparam_tinystories_256/sweep.py      # submitted 2026-10-02
    python experiments/eflm_rescale_auto_trunc_cov_reparam_tinystories_256/report.py     # -> RESULTS.md tables
