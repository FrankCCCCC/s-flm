# sudoku_reproduce — Experiment design

Spec: `setup.md`. Goal: retrain every Sudoku model from scratch with 3 seeds, report mean ± std board
accuracy per difficulty against the paper (Deschenaux & Gulcehre, arXiv:2605.11125, Table 1), and draw
the stage figures and the accuracy-along-the-sampler curves for every checkpoint.

## Grid (63 cells)

7 models × {easy (40 clues), medium (35), hard (30)} × seeds {1, 2, 3}. Every cell runs its method's
training script with its defaults (20k steps, global batch 256, AdamW, EMA 0.9999, bf16), then its
sampling script's `sudoku_eval` (2000 validation boards, 180 steps, greedy last step; flows: exact
velocity over the whole vocabulary).

| model tag | scripts | settings beyond the script defaults | paper (easy / med / hard) |
|---|---|---|---|
| `sfm` | `sfm.sh` | log-linear noise | 81.5 / 50.6 / 14.0 |
| `sfm_trunc` | `sfm_truncated.sh` | log-linear, `alpha_max=0.093` | 94.0 / 77.6 / 43.2 |
| `sfm_trunc_ada` | `sfm_truncated_adaptive.sh` | + adaptive schedule, refit every 50 steps | 94.8 / 85.2 / 45.0 |
| `mdlm` | `mdlm.sh` | LR 3e-4, `model=tiny` | 92.0 / 77.1 / 30.2 |
| `duo` | `duo.sh` | `algo=duo-base`, `model=tiny` | 96.3 / 84.7 / 58.4 |
| `eflm_auto_trunc` | `eflm_rescale_auto.sh` | R = 8, autonomous clock truncated at `tau_max=0.5`, CE, naive readout, LR 5e-4 | — |
| `eflm_auto_trunc_sc` | `eflm_rescale_auto.sh` | the same + `SELF_COND=true` | — |

The E-FLM setting is the best plain-CE cell of `experiments/eflm_rescale_auto_sudoku` (hard:
45.1 ± 1.3%). That study tuned it on hard only; easy and medium reuse it.

## Hypotheses

- S-FLM truncated and truncated+adaptive match the paper within seed noise at every difficulty; base
  S-FLM is unstable (earlier single-seed run: medium 32.6–43.3% vs the paper's 50.6%).
- MDLM and DUO land near the paper's numbers.
- Self-conditioning lifts E-FLM by 20–30 points, as it did for plain E-FLM in `experiments/selfcond_sudoku`.

## Compute

- One GPU per cell, 4 CPUs, 24 GB RAM, 6 h limit, partitions `desa,gpu,default_partition` (the last two
  are preemptible; `--requeue` resumes a preempted run from `checkpoints/last.ckpt`).
- GPUs: Ampere, Ada or Hopper only (`--constraint=ampere|ada|hopper`), all with at least 24 GB, so every
  cell trains as one batch of 256 with no gradient accumulation. Excluded nodes: `bhattacharjee-compute-02`
  (mixes in Blackwell GPUs, which the env's kernels do not support), `unicorn-compute-01` (mixes in 2080 Ti
  GPUs) and `rocky` (slow L4 GPUs).
- The 63 cells in `results.md` trained on an A6000 (41), A5000 (10), RTX 6000 Ada (5), RTX 3090 (5),
  A100 (1) or A5500 (1).
- Measured speed: 3.1 steps/s on an A6000 or A5000 (memory-bandwidth bound), about 1.9 h per cell.
- Operational history (2026-11-04):
  - The first submission (Ampere/Ada/Hopper only) trained some cells; the rest (jobs 947709–947776)
    stayed pending, because the free Ampere GPUs were "reserved for other job" by other users' hidden
    pending jobs.
  - Any-GPU replacements for those (957005–957066) trained 38 cells on Turing GPUs: 35 on a 2080 Ti as
    2 × 128 gradient accumulation (batch 256 runs out of its 11 GB), 3 on a Titan RTX or Quadro RTX 6000
    at batch 256. The pending first-submission jobs were cancelled with the user's approval.
  - At the user's request those 38 cells were retrained on Ampere+ GPUs at batch 256 (jobs
    971813–971847, plus 3 check runs). The Turing results are kept in `outputs/sudoku_reproduce/turing/`
    and `figures/turing/` and compared in `results.md`.
  - A last check (2026-11-05, jobs 973487–973495) trained MDLM easy and S-FLM + trunc + adaptive medium,
    seeds 1–3, as 2 × 128 on Ampere or Ada GPUs (`outputs/sudoku_reproduce/check/acc128_*`) to separate
    gradient accumulation from the GPU type.
  - At the user's request (2026-11-05, jobs 974645–974647), MDLM easy seeds 1–3 were trained on A6000 GPUs with
    the 2080 Ti's attention (`sdpa_math=True allow_tf32=False`, submitted by `sdpa_check.py`;
    `probe_kernels.py` shows the kernels each GPU runs). `sdpa_math` is a new flag in `configs/config.yaml`
    (default False), and `scripts/train/sudoku/mdlm.sh` now passes extra Hydra overrides through.
  - At the user's request, seeds 4–6 were added to both A6000 settings (fp32 attention: jobs 978767–978769;
    FlashAttention: 978773–978775). Three FlashAttention reruns of seeds 1–3 (978770–978772) were submitted by
    mistake and cancelled with the user's approval.
  - At the user's request every easy and medium job got `nice=1000` (set with `scontrol` after
    submission, not in `sweep.py`).

## Outputs

- Checkpoints and eval: `outputs/sudoku_reproduce/sudoku_reproduce_m-<model>_d-<difficulty>_s-<seed>/`
  (`checkpoints/last.ckpt`, `eval/results.json`).
- Report: `results.md` (via `report.py`).
- Figures: `figures/boards_stages_<cell>_confidence.png` (`visualization/sudoku_boards_stages.py`) and
  `figures/acc_diffusion_time_<cell>.png` (`visualization/sudoku_acc_diffusion_time.py`).

## Scripts

- `sweep.py`: submits the cells, skips done and queued ones. `python experiments/sudoku_reproduce/sweep.py`.
- `report.py`: collects `eval/results.json` into `results.md`.
- `viz.py`: submits one job per evaluated cell that renders its figures, skips done and queued ones.
  `cd experiments/sudoku_reproduce && python viz.py`.
