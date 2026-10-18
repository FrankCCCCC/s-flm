# eflm_rescale_auto_trunc_cov_reparam_sudoku — experiment design

Spec: `setup.md` (this folder). Method notes: `REPARAM.md` (Gaussian readout derivation), `docs/truncation.md`
(truncation values), `../eflm_rescale_auto_trunc_cov_reparam_tinystories_256/adaptivesched_refit_bug.md`
(known AdaptiveSchedule bugs, deliberately not fixed).

## Question

With the Bayes-reparametrised readout (`algo.forward_type=gaussian`: the network's logits are a residual on
log N(x_t; a_t e_v, b_t^2 I)), which noise schedule (autonomous + |alpha'| change-of-variables weight,
autonomous + adaptive, log-linear + adaptive + truncation), embedding norm R and LR give the best hard-Sudoku
solve rate when every schedule is truncated at the closed-form decode point of its R?

## Grid (81 cells = 3 noise x 3 R x 3 LR x 3 seeds)

| noise (tag `ns-`) | schedule | truncation | CE weight |
|---|---|---|---|
| `auto` | `noise=autonomous` | `tau_max = tau*(R)` | `noise_sched_cov_weight=true` (CE x \|alpha_t'\|) |
| `autoada` | `noise=autonomous-adaptive` | `tau_max = tau*(R)` | plain CE |
| `llada` | `noise=log-linear-adaptive` | `alpha_max = alpha*(R)` | plain CE |

| R | 5.0 | 8 | 16 |
|---|---|---|---|
| `tau*` = `alpha_star_euclidean(12, R, auto_clock=True)` | 0.5049 | 0.3440 | 0.1867 |
| `alpha*` = `alpha_star_euclidean(12, R)` | 0.3965 | 0.2911 | 0.1703 |

LR in {3e-4, 5e-4, 1e-3}; seeds {1, 2, 3}. Fixed: `forward_type=gaussian`, hard (30 clues), 48k/2k data (seed 42),
tiny-sphere-dit 512/8/8, ngpt init, 20k steps, global batch 256, bf16, EMA 0.9999, AdamW (wd 0, betas
0.9/0.999, eps 1e-8), clip 1.0. Adaptive arms: refit every 50 steps, buffer 50 x 256, EMA 0.9, uniform mix 1e-3
(script defaults). Eval: `sudoku_eval` on the 2000 validation boards, exact velocity, `top_k_velocity=-1`,
180 steps, greedy last step; metric = exact-board solve rate, reported as mean ± sd over the 3 seeds.

Tags / run dirs: `outputs/eflm_rescale_auto_trunc_cov_reparam_sudoku/ns-{noise}_r-{R}_lr-{lr}_s-{seed}/`,
result in `eval/results.json`; SLURM job name = `eflm_rescale_auto_trunc_cov_reparam_sudoku_<tag>`.

## Hypotheses

- **H1 (schedule).** At matched (R, LR) the adaptive arms beat `auto`: earlier naive-readout runs had
  autonomous + adaptive at 52.0 ± 1.4 vs autonomous naive 45.1 ± 1.3 (`eflm_rescale_auto_sudoku`).
- **H2 (R-invariance).** Under the exact readout the decode-point margin is independent of R (`REPARAM.md` §2.4),
  so the solve rate should vary less across R than in the naive-readout sweeps.
- **H3 (truncation + adaptive).** `llada` (log-linear + trunc + adaptive) was 45.6 ± 1.2 with the naive readout,
  i.e. truncation and adaptivity did not compose; the Gaussian readout may change that.

No naive-readout arm is in the spec, so readout effects are only read against earlier projects, and those ran
on a different harness (see Caveats).

## Compute

- 1 GPU per cell, **RTX 2080 Ti only** (`--partition=desa,thickstun --gres=gpu:nvidia_geforce_rtx_2080_ti:1`,
  i.e. the 8 GPUs of `desa-compute-01`, shared with other jobs), 4 CPUs, 28 GB, 1-day limit, `--requeue`.
- Batch 256 in one micro-batch OOMs on the 11 GB card (smoke job 892231: 9.8 GiB allocated, then OOM), so
  cells run `PER_GPU_BS=128` x 2 accumulation (same optimizer batch; peak 9.1 GB). Adaptive arms therefore refit
  twice per refit step (Bug 2 in the refit note; A = 2 behaves like A = 1 there).
- Measured ~1.5 s per optimizer step -> ~8.5 h per cell + ~12 min eval. 81 cells ≈ 700 GPU-h; with the node's
  8 GPUs that is ~10-11 waves, **~4-5 days**; more if other jobs hold 2080 Tis. Cells are submitted
  seed-major, so the 27 seed-1 cells (a full single-seed picture) finish first, ~1.5 days in.

## Caveats

- Harness: Turing with emulated bf16 and 128 x 2 accumulation. The HBFM sudoku project measured a +4.6 pt offset
  between this harness and Ampere/native-bf16 at 256 x 1, so the reference numbers above (Ampere/Ada) are not
  directly comparable. Comparisons within this project are.
- Adaptive arms use `AdaptiveSchedule` as implemented (the half-adapted schedule of Bug 1).
- Pre-launch audit (minor, pre-existing, accepted): from epoch 1 on, the data order is the same for every seed
  (`FaultTolerantDistributedSampler` uses its default seed 0), and the eval scripts ignore `SEED`, so all seeds are
  decoded from the same prior noise. The 3-seed sd therefore covers init, t-sampling and forward-noise variance only.
  `adaptivesched_refit_bug.md` says sudoku runs A = 1; this sweep runs A = 2.

## Run

    python experiments/eflm_rescale_auto_trunc_cov_reparam_sudoku/sweep.py --dry-run
    python experiments/eflm_rescale_auto_trunc_cov_reparam_sudoku/sweep.py      # submitted 2026-10-02
    python experiments/eflm_rescale_auto_trunc_cov_reparam_sudoku/report.py     # -> RESULTS.md tables
