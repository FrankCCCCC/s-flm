# Sudoku reproduction: 7 models × 3 difficulties × 3 seeds

Purpose: compare our retrained Sudoku models with Table 1 of Deschenaux & Gulcehre
([arXiv:2605.11125](https://arxiv.org/abs/2605.11125)), and show how each sampler reaches its answer.
Reader: the project team. Date: 2026-11-05. Spec: [setup.md](setup.md); design and compute: [EXPERIMENT.md](EXPERIMENT.md).

Every number is exact-match board accuracy (%) on the 2000 validation boards, mean ± std over seeds 1, 2, 3.
All 63 cells were retrained from scratch for this report, each as one batch of 256 on an Ampere or Ada GPU.
Earlier single-seed numbers are compared in [Compared with the earlier reproduction](#compared-with-the-earlier-reproduction).

## Contents

1. [Overview](#overview)
2. [Board accuracy against the paper](#board-accuracy-against-the-paper)
3. [Per-seed accuracy](#per-seed-accuracy)
4. [How each sampler reaches its answer](#how-each-sampler-reaches-its-answer)
5. [Retraining the Turing cells moved some models up and MDLM easy down](#retraining-the-turing-cells-moved-some-models-up-and-mdlm-easy-down)
6. [Compared with the earlier reproduction](#compared-with-the-earlier-reproduction)
7. [Deviations from the paper's setup](#deviations-from-the-papers-setup)
8. [Figures](#figures)
9. [Reproduce](#reproduce)

## Overview

- **12 of the 15 paper cells match** within two standard deviations of our seeds. The 3 that fall short are
  MDLM easy (88.2 vs 92.0), S-FLM medium (45.8 vs 50.6) and S-FLM + trunc + adaptive hard (41.6 vs 45.0).
  S-FLM is below the paper at every difficulty (−8.0, −4.8 and −3.4 points); easy and hard pass only because
  their seed spread is wide.
- **Every cell now trains the same way**: one batch of 256, no gradient accumulation, on an Ampere or Ada GPU.
  The 38 cells first trained on Turing GPUs (35 of them on a 2080 Ti as 2 × 128) were retrained.
- **Retraining helped S-FLM + trunc and + adaptive but hurt MDLM easy.** The S-FLM + trunc and + adaptive cells
  rose on 16 of 17 (+3.2 points on average), which closed most of their gap to the paper. MDLM easy fell 4.3–5.2
  points on all 3 seeds, from 92.8 to 88.2. Each checkpoint scores the same on either GPU type, so the
  difference comes from training. Training MDLM easy as 2 × 128 on Ampere GPUs gives 87.1, so gradient
  accumulation is not the cause; the drop goes with the GPU type. With 6 seeds each on A6000 GPUs, the 2080 Ti's
  fp32 attention gives 90.3 and FlashAttention 89.3 (p = 0.38), so the attention kernel is not the cause either.
  MDLM easy on A6000 GPUs stays short of the paper, and the cause of the 2080 Ti's higher score is still open
  ([details](#retraining-the-turing-cells-moved-some-models-up-and-mdlm-easy-down)).
- **E-FLM + auto + trunc + self-cond is the best model at every difficulty**: 98.9 / 94.3 / 76.4. On hard it beats
  the paper's best model (DUO, 58.4) by 18 points. Self-conditioning adds 5.9, 19.3 and 28.9 points to E-FLM.
- **Truncation is what fixes S-FLM** (+20.8 / +33.9 / +30.9 points). The adaptive schedule adds almost nothing
  on top in our runs (+0.8 / +0.3 / +0.2), where the paper reports +0.8 / +7.6 / +1.8.
- **Seed spread is wide on medium and hard**: 8 of the 14 medium and hard cells have a std of 3.5–10.8 points
  (DUO hard runs from 42.9 to 64.3). A single-seed number there is not reliable.
- **Along the sampler, no model un-solves boards** (the decoded board is within 0.25 points of the best step).
  MDLM's board accuracy is final by about 40% of the steps, and it cannot undo a wrong commit. DUO and E-FLM keep
  fixing cells until 81–99% of the steps. E-FLM + self-cond is done by 4–24% of the steps.

## Board accuracy against the paper

Mean ± std over 3 seeds. "Match" means |ours − paper| ≤ 2 × our std.

| model | easy (ours) | easy (paper) | medium (ours) | medium (paper) | hard (ours) | hard (paper) |
|---|---:|---:|---:|---:|---:|---:|
| MDLM | 88.2 ± 1.0 | 92.0 (short) | 73.8 ± 5.5 | 77.1 (match) | 36.4 ± 5.7 | 30.2 (match, ours higher) |
| DUO | 95.6 ± 2.0 | 96.3 (match) | 87.5 ± 4.5 | 84.7 (match) | 54.6 ± 10.8 | 58.4 (match) |
| S-FLM | 73.5 ± 4.8 | 81.5 (match) | 45.8 ± 1.0 | 50.6 (short) | 10.6 ± 3.3 | 14.0 (match) |
| S-FLM + trunc | 94.3 ± 1.1 | 94.0 (match) | 79.7 ± 1.4 | 77.6 (match) | 41.4 ± 1.5 | 43.2 (match) |
| S-FLM + trunc + adaptive | 95.1 ± 0.6 | 94.8 (match) | 80.0 ± 4.1 | 85.2 (match) | 41.6 ± 1.0 | 45.0 (short) |
| E-FLM + auto + trunc | 93.1 ± 2.3 | not in paper | 75.0 ± 4.0 | not in paper | 47.5 ± 3.5 | not in paper |
| E-FLM + auto + trunc + self-cond | 98.9 ± 0.4 | not in paper | 94.3 ± 0.8 | not in paper | 76.4 ± 4.2 | not in paper |

With 3 seeds the std itself is uncertain, so "match" is a loose test.

## Per-seed accuracy

Every cell trained as one batch of 256 on an A6000 (41 cells), A5000 (10), RTX 6000 Ada (5), RTX 3090 (5),
A100 (1) or A5500 (1).

| model | difficulty | seed 1 | seed 2 | seed 3 |
|---|---|---:|---:|---:|
| MDLM | easy | 87.20 | 89.15 | 88.35 |
| MDLM | medium | 74.25 | 79.00 | 68.05 |
| MDLM | hard | 34.00 | 42.90 | 32.40 |
| DUO | easy | 97.15 | 96.20 | 93.35 |
| DUO | medium | 83.65 | 92.40 | 86.40 |
| DUO | hard | 56.60 | 64.30 | 42.90 |
| S-FLM | easy | 78.20 | 73.60 | 68.55 |
| S-FLM | medium | 46.10 | 46.60 | 44.60 |
| S-FLM | hard | 13.40 | 6.90 | 11.40 |
| S-FLM + trunc | easy | 94.50 | 93.05 | 95.25 |
| S-FLM + trunc | medium | 81.25 | 79.15 | 78.60 |
| S-FLM + trunc | hard | 41.55 | 42.90 | 39.85 |
| S-FLM + trunc + adaptive | easy | 94.85 | 94.65 | 95.80 |
| S-FLM + trunc + adaptive | medium | 81.55 | 75.30 | 83.15 |
| S-FLM + trunc + adaptive | hard | 41.05 | 42.80 | 40.95 |
| E-FLM + auto + trunc | easy | 94.40 | 94.45 | 90.40 |
| E-FLM + auto + trunc | medium | 78.10 | 76.35 | 70.50 |
| E-FLM + auto + trunc | hard | 49.40 | 49.55 | 43.50 |
| E-FLM + auto + trunc + self-cond | easy | 99.25 | 99.05 | 98.50 |
| E-FLM + auto + trunc + self-cond | medium | 94.95 | 93.40 | 94.50 |
| E-FLM + auto + trunc + self-cond | hard | 78.80 | 78.90 | 71.55 |

## How each sampler reaches its answer

Each accuracy figure decodes the argmax of every cell after each of the 180 sampler steps, on all 2000 boards.
x is the fraction of steps taken (0 = pure noise, 1 = the decoded board). The table averages the 3 seeds;
each accuracy entry is empty-cell accuracy / board accuracy. The last two columns give the first x at which
board accuracy comes within 0.5 points of its decoded value, and the first x at which it reaches that value.

| model | difficulty | x = 0 | x = 0.5 | x = 1 | within 0.5 points at x ≈ | reaches decoded value at x ≈ |
|---|---|---:|---:|---:|---:|---:|
| MDLM | easy | 97.3 / 63.4 | 97.6 / 89.0 | 97.6 / 89.0 | 0.30 | 0.40 |
| MDLM | medium | 93.5 / 35.4 | 93.5 / 74.7 | 93.4 / 74.7 | 0.32 | 0.42 |
| MDLM | hard | 83.2 / 3.9 | 80.7 / 38.0 | 80.5 / 38.0 | 0.33 | 0.42 |
| DUO | easy | 97.4 / 68.2 | 98.6 / 91.7 | 98.9 / 95.4 | 0.82 | 0.98 |
| DUO | medium | 93.5 / 41.3 | 95.7 / 79.3 | 96.2 / 86.7 | 0.93 | 0.99 |
| DUO | hard | 82.0 / 5.8 | 84.0 / 40.4 | 85.1 / 54.2 | 0.96 | 0.99 |
| E-FLM + auto + trunc | easy | 97.6 / 69.8 | 98.5 / 92.0 | 98.5 / 92.5 | 0.50 | 0.81 |
| E-FLM + auto + trunc | medium | 92.8 / 33.7 | 94.1 / 73.8 | 94.2 / 74.9 | 0.57 | 0.97 |
| E-FLM + auto + trunc | hard | 84.6 / 7.9 | 84.6 / 46.1 | 84.6 / 47.6 | 0.67 | 0.93 |
| E-FLM + auto + trunc + self-cond | easy | 97.9 / 72.4 | 99.7 / 99.0 | 99.7 / 99.0 | 0.02 | 0.04 |
| E-FLM + auto + trunc + self-cond | medium | 93.0 / 34.3 | 98.3 / 94.4 | 98.2 / 94.4 | 0.03 | 0.09 |
| E-FLM + auto + trunc + self-cond | hard | 82.2 / 4.5 | 92.0 / 76.0 | 91.9 / 76.0 | 0.06 | 0.24 |

What this shows:

- **The first guess is about the same for every model.** From pure noise, the argmax already gets 82–98% of the
  empty cells right, but only 63–72% (easy), 34–41% (medium) and 4–8% (hard) of whole boards. Everything
  after that is refinement.
- **No model un-solves boards.** Across all 36 runs, board accuracy dips by at most 0.55 points between two
  steps (DUO), and the decoded board is within 0.25 points of the best step.
- **MDLM cannot undo a wrong commit.** Its board accuracy is final by about 40% of the steps, once nearly every
  cell is unmasked. On hard its empty-cell accuracy ends below its first guess (83.2 → 80.5): an early wrong
  digit makes later digits wrong too. The [MDLM hard seed 1 stage figure](figures/boards_stages_sudoku_reproduce_m-mdlm_d-hard_s-1_confidence.png)
  shows board 2 going from 11 wrong cells at step 0 to 24 at the end.
- **DUO and E-FLM do fix wrong cells late.** DUO's uniform-noise tokens can still change, and its board
  accuracy is still more than 0.5 points short of its final value until 82–96% of the steps. E-FLM is within
  0.5 points by 50–67% of the steps and keeps gaining small amounts until 81–97%.
- **Self-conditioning makes E-FLM fix cells early and then hold.** It is within 0.5 points of its final
  accuracy by 2–6% of the steps and reaches it by 4% (easy), 9% (medium) and 24% (hard). It also fixes far more
  cells: on hard, empty-cell accuracy goes from 82.2 to 91.9, against 84.6 to 84.6 without self-conditioning.

The figure runs use sampler seed 3, so the decoded accuracy of the stochastic samplers (MDLM, DUO) differs
slightly from the eval (MDLM hard: 38.0 here, 36.4 in the eval).

## Retraining the Turing cells moved some models up and MDLM easy down

In the first round, 38 cells trained on Turing GPUs: 35 on an 11 GB 2080 Ti as 2 × 128 gradient accumulation,
and 3 on a 24 GB Titan RTX or Quadro RTX 6000 at batch 256. All 38 were retrained with the same seeds at batch
256 on Ampere or Ada GPUs; the tables above use only the retrained cells. The first-round results are kept in
`outputs/sudoku_reproduce/turing/`.

| model | difficulty | cells retrained | first round, mean ± std | now, mean ± std |
|---|---|---:|---:|---:|
| MDLM | easy | 3 | 92.8 ± 0.6 | 88.2 ± 1.0 |
| MDLM | medium | 3 | 72.9 ± 8.8 | 73.8 ± 5.5 |
| MDLM | hard | 1 | 39.6 ± 6.3 | 36.4 ± 5.7 |
| DUO | easy | 1 | 95.8 ± 2.1 | 95.6 ± 2.0 |
| DUO | medium | 1 | 85.9 ± 5.7 | 87.5 ± 4.5 |
| DUO | hard | 3 | 52.1 ± 0.8 | 54.6 ± 10.8 |
| S-FLM | medium | 1 | 43.2 ± 4.3 | 45.8 ± 1.0 |
| S-FLM | hard | 2 | 11.8 ± 5.1 | 10.6 ± 3.3 |
| S-FLM + trunc | easy | 3 | 93.2 ± 1.4 | 94.3 ± 1.1 |
| S-FLM + trunc | medium | 2 | 75.9 ± 5.1 | 79.7 ± 1.4 |
| S-FLM + trunc | hard | 3 | 39.6 ± 1.6 | 41.4 ± 1.5 |
| S-FLM + trunc + adaptive | easy | 3 | 94.3 ± 0.6 | 95.1 ± 0.6 |
| S-FLM + trunc + adaptive | medium | 3 | 72.5 ± 5.5 | 80.0 ± 4.1 |
| S-FLM + trunc + adaptive | hard | 3 | 38.4 ± 1.8 | 41.6 ± 1.0 |
| E-FLM + auto + trunc | easy | 3 | 93.8 ± 0.6 | 93.1 ± 2.3 |
| E-FLM + auto + trunc + self-cond | medium | 1 | 93.9 ± 0.6 | 94.3 ± 0.8 |
| E-FLM + auto + trunc + self-cond | hard | 2 | 77.0 ± 3.6 | 76.4 ± 4.2 |

Rows cover the model and difficulty pairs with at least one retrained cell; their means include the cells
that were not retrained.

Change per retrained cell, by model:

| model | cells | higher now | mean change | range |
|---|---:|---:|---:|---:|
| S-FLM + trunc and S-FLM + trunc + adaptive | 17 | 16 | +3.2 | −0.3 to +12.8 |
| DUO | 5 | 3 | +2.3 | −8.6 to +11.2 |
| S-FLM | 3 | 1 | +1.3 | −3.7 to +7.8 |
| E-FLM, with and without self-cond | 6 | 3 | −0.4 | −2.8 to +1.1 |
| MDLM | 7 | 2 | −3.0 | −9.6 to +3.1 |

The MDLM easy checkpoints were then evaluated on the other GPU type (outputs
`outputs/sudoku_reproduce/check/xeval_mdlm_easy_*`):

| seed | Turing-trained, scored on a 2080 Ti | Turing-trained, scored on Ampere | Ampere-trained, scored on Ampere | Ampere-trained, scored on a 2080 Ti |
|---|---:|---:|---:|---:|
| 1 | 92.35 | 92.40 (A5000) | 87.20 | 87.30 |
| 2 | 93.50 | 93.55 (A6000) | 89.15 | 89.20 |
| 3 | 92.60 | 92.70 (RTX 3090) | 88.35 | 88.45 |

What this shows:

- **The S-FLM + trunc and + adaptive cells were low on Turing.** They rose on 16 of 17 cells, by up to 12.8
  points (S-FLM + trunc + adaptive medium seed 3: 70.40 → 83.15).
- **MDLM easy was high on Turing.** All 3 seeds fell 4.3–5.2 points, a gap about 5 times the larger seed std
  of the two groups (1.0). MDLM easy now falls short of the paper (88.2 vs 92.0), where the Turing cells matched it.
- **The eval is not the cause.** Every checkpoint scores within 0.1 points on either GPU type, so the
  difference is in the trained weights.
- **The Turing runs differ in two ways**: gradient accumulation (2 × 128) and
  the GPU itself (no native bf16, no FlashAttention kernels in PyTorch's attention). The code treats 2 × 128 as
  one step of 256 in expectation: the stratified training times are drawn for the whole batch of 256 and split
  between the two micro-batches, Lightning averages the two losses, and EMA updates once per optimizer step.
  The three Turing cells trained at batch 256 moved both ways (MDLM hard seed 1: −9.6, MDLM medium seed 3:
  +3.1, E-FLM easy seed 3: −2.8), too few to decide.

To separate the two, MDLM easy and S-FLM + trunc + adaptive medium were trained again as 2 × 128 on Ampere and
Ada GPUs (A5000, RTX 3090, RTX 6000 Ada), seeds 1–3 (`outputs/sudoku_reproduce/check/acc128_*`). All Turing
runs in this table were on a 2080 Ti.

| model and difficulty | training | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---:|---:|---:|---:|
| MDLM easy | 2 × 128 on a 2080 Ti (first round) | 92.35 | 93.50 | 92.60 | 92.8 ± 0.6 |
| MDLM easy | 2 × 128 on Ampere or Ada (check) | 89.55 | 82.60 | 89.25 | 87.1 ± 3.9 |
| MDLM easy | 256 on Ampere or Ada (now) | 87.20 | 89.15 | 88.35 | 88.2 ± 1.0 |
| S-FLM + trunc + adaptive medium | 2 × 128 on a 2080 Ti (first round) | 78.80 | 68.40 | 70.40 | 72.5 ± 5.5 |
| S-FLM + trunc + adaptive medium | 2 × 128 on Ampere or Ada (check) | 75.85 | 76.20 | 75.05 | 75.7 ± 0.6 |
| S-FLM + trunc + adaptive medium | 256 on Ampere or Ada (now) | 81.55 | 75.30 | 83.15 | 80.0 ± 4.1 |

- **For MDLM easy, the GPU type matters and gradient accumulation does not.** As 2 × 128 on Ampere or Ada,
  MDLM easy scores 87.1, like batch 256 there (88.2) and unlike 2 × 128 on a 2080 Ti (92.8). All 6 Ampere and
  Ada runs (82.6–89.6) are below all 3 Turing runs (92.4–93.5). Which part of the Turing GPU matters, its bf16
  handling or its attention kernel, is not tested.
- **For S-FLM + trunc + adaptive medium, the check does not decide.** As 2 × 128 on Ampere or Ada it scores
  75.7, between the 2080 Ti runs (72.5) and batch 256 (80.0). With 3 seeds and stds of 4–6 points on the
  other two rows, neither difference is clear.
- **The same seed does not give the same model across batch layouts.** MDLM easy seed 2 scores 82.60 as
  2 × 128 and 89.15 as 256, both on Ampere GPUs. Compare the means, not seed-by-seed pairs.

A training step differs between the two GPUs in one place (`probe_kernels.py`, one forward and backward pass
profiled on each GPU). A 2080 Ti cannot run PyTorch's FlashAttention in bf16, so attention falls back to
PyTorch's math kernel, which computes in fp32. An A6000 runs FlashAttention in bf16. The other matrix
multiplications take bf16 inputs and accumulate in fp32 on both. The code, the data cache and the environment
were the same in both rounds.

MDLM easy was therefore trained again on A6000 GPUs, the GPU of its three batch-256 cells, with the 2080 Ti's
attention: `sdpa_math=True allow_tf32=False`. TF32 has to be off, or the A6000 runs the fp32 attention matrix
multiplications in TF32. Seeds 4–6 were added to both A6000 settings, so each has 6 seeds (`sdpa_check.py`,
`outputs/sudoku_reproduce/check/{sdpa_math,flash}_*`; FlashAttention seeds 1–3 are the batch-256 cells). Batch
256, steps and eval match the batch-256 cells, and each job's log lists the kernels its GPU ran. FlashAttention
seed 5 was preempted and resumed from its step-5000 checkpoint on another A6000.

| MDLM easy trained as | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | mean ± std |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 × 128 on a 2080 Ti, math attention in fp32 (first round) | 92.35 | 93.50 | 92.60 | – | – | – | 92.82 ± 0.60 |
| 256 on an A6000, FlashAttention in bf16 | 87.20 | 89.15 | 88.35 | 90.00 | 90.10 | 90.70 | 89.25 ± 1.30 |
| 256 on an A6000, math attention in fp32 | 93.90 | 87.70 | 89.15 | 87.70 | 91.40 | 92.20 | 90.34 ± 2.55 |

- **The attention kernel is not the cause.** On the same GPU model with 6 seeds each, fp32 attention scores 1.1
  points above FlashAttention, and the difference is not significant (Welch t-test p = 0.38).
- **The 2080 Ti runs are higher than both A6000 settings**: 92.8 against 89.3 with FlashAttention (p = 0.001) and
  90.3 with fp32 attention (p = 0.065). What causes this is still open. The other differences we know of are in how
  each GPU runs the same bf16 linear layers: tensor cores on the A6000, an fp32 fallback kernel on the 2080 Ti,
  both accumulating in fp32.
- **With 6 seeds, MDLM easy on A6000 GPUs averages 89.3 ± 1.3**, just short of the paper's 92.0 (two-std range
  86.7–91.8). Seeds 4–6 (90.00–90.70) scored higher than seeds 1–3 (87.20–89.15). The main table keeps seeds 1–3,
  like every other cell.

## Compared with the earlier reproduction

The earlier reproduction trained one seed per cell (two or three on some medium cells) on an A6000 at batch 256.

| model | difficulty | earlier | now (mean ± std, 3 seeds) |
|---|---|---:|---:|
| S-FLM | easy | 77.60 | 73.5 ± 4.8 |
| S-FLM | medium | 39.0 (3 seeds: 32.55, 43.25, 41.25) | 45.8 ± 1.0 |
| S-FLM | hard | 13.85 | 10.6 ± 3.3 |
| S-FLM + trunc | easy | 95.05 | 94.3 ± 1.1 |
| S-FLM + trunc | medium | 77.90 | 79.7 ± 1.4 |
| S-FLM + trunc | hard | 48.30 | 41.4 ± 1.5 |
| S-FLM + trunc + adaptive | easy | 94.15 | 95.1 ± 0.6 |
| S-FLM + trunc + adaptive | medium | 78.2 (2 seeds: 79.20, 77.20) | 80.0 ± 4.1 |
| S-FLM + trunc + adaptive | hard | 45.95 | 41.6 ± 1.0 |

Seven of nine cells agree. The two hard cells are lower now: S-FLM + trunc by 6.9 points and S-FLM + trunc +
adaptive by 4.4, both above every one of today's seeds.

- **The eval is not the cause.** The earlier S-FLM + trunc hard checkpoint scores 49.4% with today's eval code
  and data (48.3% in the earlier report; `outputs/sudoku_reproduce/check/oldckpt_sfm_trunc_hard/`).
- **No difference was found in the training setup.** The earlier run (commit 18d586f, 2026-05-29, seed 1,
  A6000, batch 256) and today's runs share the S-FLM training code (today's adds self-conditioning, off by
  default), the noise schedule, the model and the data generator (deterministic, seed 42). Their environments
  have the same torch and Lightning versions.
- **One lucky seed is an unlikely explanation**, because both earlier hard cells sit 4–5 of today's seed
  stds above today's mean. The cause is not found.

The earlier report also found and fixed a bug: the adaptive schedule was inactive when a checkpoint was loaded
for eval (`noise_schedules.py`, `_maybe_rebuild_spline`). This retrain uses the fixed code.

## Deviations from the paper's setup

- **GPU types are mixed** (A6000, A5000, RTX 6000 Ada, RTX 3090, A100, A5500), so cells are not bitwise
  comparable. Retraining a cell on another GPU family can move it by several points
  ([retraining the Turing cells](#retraining-the-turing-cells-moved-some-models-up-and-mdlm-easy-down)).
- **E-FLM settings were tuned on hard only.** R = 8, autonomous clock truncated at τ_max = 0.5, cross-entropy
  loss, LR 5e-4: the best plain cross-entropy cell of `experiments/eflm_rescale_auto_sudoku`. Easy and medium reuse it.
- **The data cache was regenerated** with the repository's generator (seed 42), because the earlier cache
  location is no longer readable.
- **Figure runs use sampler seed 3**, as noted above.

## Figures

Stage figures show boards 0–5 at steps 0, 10%, 25%, 50%, 75% and the decoded board; each cell is coloured by
the model's confidence, and red digits differ from the solution. Accuracy figures plot empty-cell and board
accuracy of the argmax along the sampler (requested for MDLM, DUO and the two E-FLM models only). Figures of the
first-round, Turing-trained cells are kept in `figures/turing/`.

| model | difficulty | stage figures (seed 1, 2, 3) | accuracy along the sampler (seed 1, 2, 3) |
|---|---|---|---|
| MDLM | easy | [MDLM easy seed 1 stages](figures/boards_stages_sudoku_reproduce_m-mdlm_d-easy_s-1_confidence.png), [MDLM easy seed 2 stages](figures/boards_stages_sudoku_reproduce_m-mdlm_d-easy_s-2_confidence.png), [MDLM easy seed 3 stages](figures/boards_stages_sudoku_reproduce_m-mdlm_d-easy_s-3_confidence.png) | [MDLM easy seed 1 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-mdlm_d-easy_s-1.png), [MDLM easy seed 2 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-mdlm_d-easy_s-2.png), [MDLM easy seed 3 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-mdlm_d-easy_s-3.png) |
| MDLM | medium | [MDLM medium seed 1 stages](figures/boards_stages_sudoku_reproduce_m-mdlm_d-medium_s-1_confidence.png), [MDLM medium seed 2 stages](figures/boards_stages_sudoku_reproduce_m-mdlm_d-medium_s-2_confidence.png), [MDLM medium seed 3 stages](figures/boards_stages_sudoku_reproduce_m-mdlm_d-medium_s-3_confidence.png) | [MDLM medium seed 1 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-mdlm_d-medium_s-1.png), [MDLM medium seed 2 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-mdlm_d-medium_s-2.png), [MDLM medium seed 3 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-mdlm_d-medium_s-3.png) |
| MDLM | hard | [MDLM hard seed 1 stages](figures/boards_stages_sudoku_reproduce_m-mdlm_d-hard_s-1_confidence.png), [MDLM hard seed 2 stages](figures/boards_stages_sudoku_reproduce_m-mdlm_d-hard_s-2_confidence.png), [MDLM hard seed 3 stages](figures/boards_stages_sudoku_reproduce_m-mdlm_d-hard_s-3_confidence.png) | [MDLM hard seed 1 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-mdlm_d-hard_s-1.png), [MDLM hard seed 2 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-mdlm_d-hard_s-2.png), [MDLM hard seed 3 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-mdlm_d-hard_s-3.png) |
| DUO | easy | [DUO easy seed 1 stages](figures/boards_stages_sudoku_reproduce_m-duo_d-easy_s-1_confidence.png), [DUO easy seed 2 stages](figures/boards_stages_sudoku_reproduce_m-duo_d-easy_s-2_confidence.png), [DUO easy seed 3 stages](figures/boards_stages_sudoku_reproduce_m-duo_d-easy_s-3_confidence.png) | [DUO easy seed 1 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-duo_d-easy_s-1.png), [DUO easy seed 2 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-duo_d-easy_s-2.png), [DUO easy seed 3 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-duo_d-easy_s-3.png) |
| DUO | medium | [DUO medium seed 1 stages](figures/boards_stages_sudoku_reproduce_m-duo_d-medium_s-1_confidence.png), [DUO medium seed 2 stages](figures/boards_stages_sudoku_reproduce_m-duo_d-medium_s-2_confidence.png), [DUO medium seed 3 stages](figures/boards_stages_sudoku_reproduce_m-duo_d-medium_s-3_confidence.png) | [DUO medium seed 1 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-duo_d-medium_s-1.png), [DUO medium seed 2 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-duo_d-medium_s-2.png), [DUO medium seed 3 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-duo_d-medium_s-3.png) |
| DUO | hard | [DUO hard seed 1 stages](figures/boards_stages_sudoku_reproduce_m-duo_d-hard_s-1_confidence.png), [DUO hard seed 2 stages](figures/boards_stages_sudoku_reproduce_m-duo_d-hard_s-2_confidence.png), [DUO hard seed 3 stages](figures/boards_stages_sudoku_reproduce_m-duo_d-hard_s-3_confidence.png) | [DUO hard seed 1 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-duo_d-hard_s-1.png), [DUO hard seed 2 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-duo_d-hard_s-2.png), [DUO hard seed 3 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-duo_d-hard_s-3.png) |
| S-FLM | easy | [S-FLM easy seed 1 stages](figures/boards_stages_sudoku_reproduce_m-sfm_d-easy_s-1_confidence.png), [S-FLM easy seed 2 stages](figures/boards_stages_sudoku_reproduce_m-sfm_d-easy_s-2_confidence.png), [S-FLM easy seed 3 stages](figures/boards_stages_sudoku_reproduce_m-sfm_d-easy_s-3_confidence.png) | not requested |
| S-FLM | medium | [S-FLM medium seed 1 stages](figures/boards_stages_sudoku_reproduce_m-sfm_d-medium_s-1_confidence.png), [S-FLM medium seed 2 stages](figures/boards_stages_sudoku_reproduce_m-sfm_d-medium_s-2_confidence.png), [S-FLM medium seed 3 stages](figures/boards_stages_sudoku_reproduce_m-sfm_d-medium_s-3_confidence.png) | not requested |
| S-FLM | hard | [S-FLM hard seed 1 stages](figures/boards_stages_sudoku_reproduce_m-sfm_d-hard_s-1_confidence.png), [S-FLM hard seed 2 stages](figures/boards_stages_sudoku_reproduce_m-sfm_d-hard_s-2_confidence.png), [S-FLM hard seed 3 stages](figures/boards_stages_sudoku_reproduce_m-sfm_d-hard_s-3_confidence.png) | not requested |
| S-FLM + trunc | easy | [S-FLM + trunc easy seed 1 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_d-easy_s-1_confidence.png), [S-FLM + trunc easy seed 2 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_d-easy_s-2_confidence.png), [S-FLM + trunc easy seed 3 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_d-easy_s-3_confidence.png) | not requested |
| S-FLM + trunc | medium | [S-FLM + trunc medium seed 1 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_d-medium_s-1_confidence.png), [S-FLM + trunc medium seed 2 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_d-medium_s-2_confidence.png), [S-FLM + trunc medium seed 3 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_d-medium_s-3_confidence.png) | not requested |
| S-FLM + trunc | hard | [S-FLM + trunc hard seed 1 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_d-hard_s-1_confidence.png), [S-FLM + trunc hard seed 2 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_d-hard_s-2_confidence.png), [S-FLM + trunc hard seed 3 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_d-hard_s-3_confidence.png) | not requested |
| S-FLM + trunc + adaptive | easy | [S-FLM + trunc + adaptive easy seed 1 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_ada_d-easy_s-1_confidence.png), [S-FLM + trunc + adaptive easy seed 2 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_ada_d-easy_s-2_confidence.png), [S-FLM + trunc + adaptive easy seed 3 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_ada_d-easy_s-3_confidence.png) | not requested |
| S-FLM + trunc + adaptive | medium | [S-FLM + trunc + adaptive medium seed 1 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_ada_d-medium_s-1_confidence.png), [S-FLM + trunc + adaptive medium seed 2 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_ada_d-medium_s-2_confidence.png), [S-FLM + trunc + adaptive medium seed 3 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_ada_d-medium_s-3_confidence.png) | not requested |
| S-FLM + trunc + adaptive | hard | [S-FLM + trunc + adaptive hard seed 1 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_ada_d-hard_s-1_confidence.png), [S-FLM + trunc + adaptive hard seed 2 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_ada_d-hard_s-2_confidence.png), [S-FLM + trunc + adaptive hard seed 3 stages](figures/boards_stages_sudoku_reproduce_m-sfm_trunc_ada_d-hard_s-3_confidence.png) | not requested |
| E-FLM + auto + trunc | easy | [E-FLM + auto + trunc easy seed 1 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_d-easy_s-1_confidence.png), [E-FLM + auto + trunc easy seed 2 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_d-easy_s-2_confidence.png), [E-FLM + auto + trunc easy seed 3 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_d-easy_s-3_confidence.png) | [E-FLM + auto + trunc easy seed 1 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_d-easy_s-1.png), [E-FLM + auto + trunc easy seed 2 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_d-easy_s-2.png), [E-FLM + auto + trunc easy seed 3 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_d-easy_s-3.png) |
| E-FLM + auto + trunc | medium | [E-FLM + auto + trunc medium seed 1 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_d-medium_s-1_confidence.png), [E-FLM + auto + trunc medium seed 2 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_d-medium_s-2_confidence.png), [E-FLM + auto + trunc medium seed 3 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_d-medium_s-3_confidence.png) | [E-FLM + auto + trunc medium seed 1 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_d-medium_s-1.png), [E-FLM + auto + trunc medium seed 2 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_d-medium_s-2.png), [E-FLM + auto + trunc medium seed 3 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_d-medium_s-3.png) |
| E-FLM + auto + trunc | hard | [E-FLM + auto + trunc hard seed 1 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_d-hard_s-1_confidence.png), [E-FLM + auto + trunc hard seed 2 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_d-hard_s-2_confidence.png), [E-FLM + auto + trunc hard seed 3 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_d-hard_s-3_confidence.png) | [E-FLM + auto + trunc hard seed 1 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_d-hard_s-1.png), [E-FLM + auto + trunc hard seed 2 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_d-hard_s-2.png), [E-FLM + auto + trunc hard seed 3 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_d-hard_s-3.png) |
| E-FLM + auto + trunc + self-cond | easy | [E-FLM + auto + trunc + self-cond easy seed 1 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_sc_d-easy_s-1_confidence.png), [E-FLM + auto + trunc + self-cond easy seed 2 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_sc_d-easy_s-2_confidence.png), [E-FLM + auto + trunc + self-cond easy seed 3 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_sc_d-easy_s-3_confidence.png) | [E-FLM + auto + trunc + self-cond easy seed 1 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_sc_d-easy_s-1.png), [E-FLM + auto + trunc + self-cond easy seed 2 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_sc_d-easy_s-2.png), [E-FLM + auto + trunc + self-cond easy seed 3 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_sc_d-easy_s-3.png) |
| E-FLM + auto + trunc + self-cond | medium | [E-FLM + auto + trunc + self-cond medium seed 1 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_sc_d-medium_s-1_confidence.png), [E-FLM + auto + trunc + self-cond medium seed 2 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_sc_d-medium_s-2_confidence.png), [E-FLM + auto + trunc + self-cond medium seed 3 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_sc_d-medium_s-3_confidence.png) | [E-FLM + auto + trunc + self-cond medium seed 1 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_sc_d-medium_s-1.png), [E-FLM + auto + trunc + self-cond medium seed 2 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_sc_d-medium_s-2.png), [E-FLM + auto + trunc + self-cond medium seed 3 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_sc_d-medium_s-3.png) |
| E-FLM + auto + trunc + self-cond | hard | [E-FLM + auto + trunc + self-cond hard seed 1 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_sc_d-hard_s-1_confidence.png), [E-FLM + auto + trunc + self-cond hard seed 2 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_sc_d-hard_s-2_confidence.png), [E-FLM + auto + trunc + self-cond hard seed 3 stages](figures/boards_stages_sudoku_reproduce_m-eflm_auto_trunc_sc_d-hard_s-3_confidence.png) | [E-FLM + auto + trunc + self-cond hard seed 1 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_sc_d-hard_s-1.png), [E-FLM + auto + trunc + self-cond hard seed 2 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_sc_d-hard_s-2.png), [E-FLM + auto + trunc + self-cond hard seed 3 accuracy](figures/acc_diffusion_time_sudoku_reproduce_m-eflm_auto_trunc_sc_d-hard_s-3.png) |

## Reproduce

1. Train and evaluate every cell: `python experiments/sudoku_reproduce/sweep.py` (skips finished and queued cells).
2. Draw the figures: `cd experiments/sudoku_reproduce && python viz.py`.
3. Print the tables: `python experiments/sudoku_reproduce/report.py`.

Checkpoints and eval results: `outputs/sudoku_reproduce/sudoku_reproduce_m-<model>_d-<difficulty>_s-<seed>/`
(`checkpoints/last.ckpt`, `eval/results.json`).
