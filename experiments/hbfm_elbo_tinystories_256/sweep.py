#!/usr/bin/env python
"""hbfm_elbo_tinystories_256 -- HBFM on TinyStories (seq 256) trained on the NELBO, with the path-KL
rate estimated by Monte Carlo over sampled words (`algo.elbo_num_words`, `Loss.bridge_loss_elbo_mc`).

Spec: `setup.md`. Shared by all cells: small DiT 768/12/12, naive readout in float32, (H^3)^16 =
embed_dim 48, GPT-2 XL word embedding PCA-reduced to 48 (training init), `exp` proposal, log-linear
noise, LR 3e-4, 30k steps, global batch 512, bf16, EMA 0.9999, loss_type elbo with S = NUM_WORDS
sampled words per token.

Grid (setup.md): cells (PROPOSAL_RATE, GAUSS_CURV) with T_MAX = T = ln(1/eps)/rate, eps = 1e-3.
  ada   curvature_bound 4 (lambda = 4^tanh(h)) on every cell, + lambda per keep-{step}.ckpt
  fix   curvature_bound 1 on the anchor r3_K0.5 only
Wave 2: one knob off the anchor per cell -- trunc_exp ub 0.5 / 1 (T_MAX = ub), ELBO_NUM_WORDS 4 /
64, BERT-large word embedding (ada), and trunc_exp ub 1 on the fix arm. Trained in stages (`--steps 6000`, then 10000, then 30000; each resumes
from last.ckpt) with the exact NELBO on STAGE_VAL of the validation set after a stage
(eval/nelbo_{steps}.json); only the cells that pass the early-stopping rule (EXPERIMENT.md) are
submitted for the next stage.

Eval per cell (setup.md's commented protocol): validation PPL (the CE objective) + 64 samples at
180 steps, TOPK_VELOCITY=1, greedy last step; then the exact NELBO (elbo_num_words unset) over the
full validation set, LOSS_TYPE=elbo SEED=1 EVAL_BS=16 -> eval/nelbo.json.

Sites (each reached over ssh at `host`; code copy with `--sync-code`, results back with `--pull`):
  a100   ch2263@unicorn, nlplarge (A100 80GB): 1 GPU x micro-batch 64 per cell (41.6 GB measured);
         checkpoints on the node's /scratch (experiment.py).
  a6000  sc3379@unicorn, desa without the 2080 Ti node (A6000 48 GB / A5000 24 GB): 4 GPUs x
         micro-batch 16 per cell. The env, data and HF cache are copies in SFM_HOME, since the
         /share/thickstun ones behind ~/anaconda3 and ~/.cache are not readable by sc3379.
Micro-batch 32 used 23 GB on the A100 smoke test (job 795047).

ORCHESTRATION ONLY. Idempotent: a cell with eval/nelbo.json (eval/nelbo_{steps}.json for a stage;
local, after `--pull`) or a queued job of the same name is skipped; a resubmitted cell resumes from
checkpoints/last.ckpt. A site whose user is the current one is driven locally instead of over ssh.

Usage:  python sweep.py --site {a100,a6000} --sync-code
        python sweep.py --site {a100,a6000} [--arms ada,fix] [--cells c1,c2] [--steps N] [--seeds 1] [--dry-run]
        python sweep.py --site {a100,a6000} --pull
        python sweep.py --site a6000 --wandb-sync [--cells c1,c2]   # from sc3379; copy the a100 cells' wandb/ first
"""
import argparse
import getpass
import glob
import json
import math
import os
import subprocess
import sys
import textwrap

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))   # this checkout
NAME = 'hbfm_elbo_tinystories_256'
EXP = f'{REPO}/experiments/{NAME}'
OUT = f'{REPO}/outputs/{NAME}'
JOB = 'hbfm_elbo_ts'
KEY = os.environ.get('SWEEP_KEY', '/home/sc3379/.ssh/unicorn_internal')   # laptop: ~/.ssh/cornell
SFM_HOME = '/share/desa/nfs02/sc3379/sfm_home'   # sc3379 env, data and HF cache (/share/thickstun is not readable)
CODE = ['*.py', 'configs', 'models', 'numeric', 'callbacks', 'scripts', 'data', 'tests', 'experiments/hbfm_rlf_ada_curv',
        'experiments/hbfm_rlf_ada_curv_tinystories_256']   # curvature_ts.py and its load_cell

NUM_WORDS = 16   # words drawn from p_theta per token for the path-KL rate
PRETRAINED_EMBED = 'gpt2-xl'   # word embedding init (1600-d), PCA-reduced to EMBED_DIM
EPS = 1e-3       # sampling_eps: the horizon T = ln(1/eps)/rate of the exp proposal

SITES = {
  'a100': dict(host='ch2263@unicorn-login-02.coecis.cornell.edu', repo='/home/ch2263/syc_workspace/s-flm-dev/s-flm',
               out='/home/ch2263/syc_workspace/s-flm-dev/s-flm/outputs',
               envbin='/home/ch2263/miniconda3/envs/sfm/bin', env='',
               cache='/home/ch2263/syc_workspace/s-flm_data_home/data_cache', tmp='/home/ch2263/tmp',
               devices=1, per_gpu_bs=64, workers=4,
               sbatch='--partition=nlplarge-claire-highpri,nlplarge --gres=gpu:1 --cpus-per-task=4 --mem=64G',
               sbatch1='--partition=nlplarge-claire-highpri,nlplarge --gres=gpu:1 --cpus-per-task=4 --mem=64G'),
  'a6000': dict(host='sc3379@unicorn-login-05.coecis.cornell.edu',
                repo='/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm',
                out=f'{SFM_HOME}/outputs',   # the repo's outputs/ links into /share/thickstun too
                envbin=f'{SFM_HOME}/envs/sfm/bin', env=f'export HF_HOME={SFM_HOME}/hf_cache XDG_CACHE_HOME={SFM_HOME}/cache '
                    f'MPLCONFIGDIR={SFM_HOME}/cache/matplotlib WANDB_CACHE_DIR={SFM_HOME}/cache/wandb '
                    f'WANDB_CONFIG_DIR={SFM_HOME}/cache/wandb',   # ~/.cache is not readable either
                cache=f'{SFM_HOME}/data_cache', tmp='/home/sc3379/tmp',
                devices=4, per_gpu_bs=16, workers=4,   # A6000 48 GB / A5000 24 GB nodes
                sbatch='--partition=desa --exclude=desa-compute-01 --gres=gpu:4 --cpus-per-task=16 --mem=128G',
                sbatch1='--partition=desa,gpu --exclude=desa-compute-01 --constraint="ampere|ada" --gres=gpu:1 '
                        '--cpus-per-task=8 --mem=64G'),   # scores also backfill the shared gpu partition (Ampere/Ada only)
}

ANCHOR = dict(rate=3, k=-0.5, ub=None, words=NUM_WORDS, embed=PRETRAINED_EMBED, lr=3e-4,
              sched='constant', sched_from=0, hold=3500, dropout=0.1)   # ub None = exp
CELLS = {   # name: what differs from the anchor (PROPOSAL_RATE, GAUSS_CURV, trunc_exp ub, ELBO_NUM_WORDS, embedding, LR,
            # LR_SCHED, LR_SCHED_FROM, LR_HOLD_STEPS, DROPOUT)
  'r3_K0.5': {},     # anchor
  'r1.5_K0.5': dict(rate=1.5),
  'r5_K0.5': dict(rate=5),
  'r6_K1': dict(rate=6, k=-1.0),
  # wave 2 (ada only, staged and early-stopped at 1/5 and 1/3 of training, setup.md)
  'r3_K0.5_ub0.5': dict(ub=0.5),
  'r3_K0.5_ub1': dict(ub=1.0),
  'r3_K0.5_S4': dict(words=4),
  'r3_K0.5_S64': dict(words=64),
  'r3_K0.5_bert': dict(embed='google-bert/bert-large-uncased'),
  # wave 3 (fix only, staged; the user's target: final NELBO 1.7, 2026-10-31)
  'r3_K0.5_lr6e-4': dict(lr=6e-4),
  'r3_K0.5_lr1e-3': dict(lr=1e-3),
  # wave 4 (staged): the ub axis around ub 0.5, which scored 2.126 at 6k (exp cells: ~2.75)
  'r3_K0.5_ub0.25': dict(ub=0.25),
  'r3_K0.5_ub0.35': dict(ub=0.35),
  'r3_K0.5_ub0.7': dict(ub=0.7),
  # wave 5 (staged): the proposal rate under trunc_exp, where it no longer sets the horizon; the
  # path-KL rate of the fix model falls ~5.6x over [0, 0.5] (nelbo_variance), steeper than rate 3
  'r6_K0.5_ub0.5': dict(rate=6, ub=0.5),
  # wave 6 (staged): the setup.md knobs on the best base (ub 0.5) instead of exp
  'r3_K0.5_ub0.5_bert': dict(ub=0.5, embed='google-bert/bert-large-uncased'),
  'r3_K0.5_ub0.5_S64': dict(ub=0.5, words=64),
  'r3_K0.5_unif0.5': dict(ub=0.5, mode='unif'),   # time_conversion_mode unif: the rate -> 0 end of trunc_exp
  # wave 7 (staged): the curvature under trunc_exp. K -1 with the rate and ub that keep ub 0.5's unit
  # proposal (unit rate 6, unit horizon 0.25), so only the state scale R changes (1.41 -> 1), as r6_K1 under exp
  'r6_K1_ub0.25': dict(rate=6, k=-1.0, ub=0.25),
  # wave 8 (fix, staged): LR on the trunc_exp leader (ub 0.5 fix: 1.959 at 10k); under exp, 1e-3 gained 1.1% at 6k
  'r3_K0.5_ub0.5_lr1e-3': dict(ub=0.5, lr=1e-3),
  'r3_K0.5_ub0.5_S4': dict(ub=0.5, words=4),   # ada: S = 4 / 16 / 64 on one base (the MC word count, setup.md)
  # wave 9 (fix, staged): K -0.25 with ub 0.5's unit proposal (unit rate 6, unit horizon 0.25), R 1.41 -> 2;
  # K -1 (R 1) scored +0.77% at 6k
  'r1.5_K0.25_ub1': dict(rate=1.5, k=-0.25, ub=1.0),
  # wave 10 (staged): LR 6e-4 between 3e-4 and 1e-3 on ub 0.5 fix
  'r3_K0.5_ub0.5_lr6e-4': dict(ub=0.5, lr=6e-4),
  # wave 11 (fix, staged): LR 1e-3 scored 2.002 at 6k on ub 0.5 fix, -3.8% vs 3e-4 (2.081)
  'r3_K0.5_ub0.5_lr2e-3': dict(ub=0.5, lr=2e-3),
  # wave 12 (fix, staged): K -0.25 scored 1.955 at 10k (-0.20% vs K -0.5); combined with LR 1e-3
  'r1.5_K0.25_ub1_lr1e-3': dict(rate=1.5, k=-0.25, ub=1.0, lr=1e-3),
  # wave 13 (fix, staged): LR 3e-4 / 6e-4 / 1e-3 -> 2.081 / 2.026 / 2.002 at 6k on ub 0.5. The horizon at
  # LR 1e-3: at LR 3e-4 ub 0.7 closed its gap to ub 0.5 with training (ada: +0.3% at 6k, -0.05% at 10k)
  'r3_K0.5_ub0.7_lr1e-3': dict(ub=0.7, lr=1e-3),
  # wave 14 (staged): CURV_BOUND 2 scored 2.129 at 6k (bound 4: 2.126, fix: 2.081), so any learned lambda costs;
  # ada again at LR 1e-3 (ARMS['ada']), and S = 64 at LR 1e-3: -0.39% per step vs S = 16 at LR 3e-4
  'r3_K0.5_ub0.5_S64_lr1e-3': dict(ub=0.5, words=64, lr=1e-3),
  # wave 15 (fix, to 30k): LR 2e-3 scored 1.994 at 6k (1e-3: 2.002), but the constant-LR lead shrinks with
  # training (1e-3 vs 3e-4: -3.8% at 6k, -2.65% at 10k). LR 2e-3's 6k checkpoint (checkpoints/last.ckpt copied
  # in), then cosine decay to 1e-6 at 30k
  'r3_K0.5_ub0.5_lr2e-3_cos': dict(ub=0.5, lr=2e-3, sched='cosine', sched_from=6000),
  # wave 16 (fix, to 30k): the same 6k checkpoint and schedule with S = 64 words. A lower-variance gradient
  # should matter most while the LR decays (S 64 vs 16: -0.39% at 6k, ada, LR 3e-4)
  'r3_K0.5_ub0.5_lr2e-3_cos_S64': dict(ub=0.5, lr=2e-3, sched='cosine', sched_from=6000, words=64),
  # wave 17 (fix, to 30k): S = 64 at LR 1e-3 scored 1.990 at 6k (-0.58% vs S 16). Its 6k checkpoint
  # (keep-6000.ckpt copied in as checkpoints/last.ckpt), then cosine from 1e-3: the peak LR under cosine, vs wave 16
  'r3_K0.5_ub0.5_S64_lr1e-3_cos': dict(ub=0.5, lr=1e-3, sched='cosine', sched_from=6000, words=64),
  # wave 18 (fix, to 30k, from step 0): the peak above the constant-LR optimum (1e-3 ~ 2e-3 at 6k), which a
  # decay can afford; warmup 2500, then cosine to 1e-6 at 30k
  'r3_K0.5_ub0.5_S64_lr3e-3_cos': dict(ub=0.5, lr=3e-3, sched='cosine', words=64),
  # wave 19 (fix, A100, to 6k): ub 0.7 ada beat ub 0.5 ada at 30k (1.815 vs 1.828) after trailing at 6k, but its
  # lambda (0.45) keeps the clock horizon at 0.31. Fix ub 0.6 at LR 1e-3, against ub 0.5 (seeds 1-3) and 0.7 at 6k
  'r3_K0.5_ub0.6_lr1e-3': dict(ub=0.6, lr=1e-3),
  # wave 20 (fix, A100, 6k then 30k): the best recipe so far (S 64, cosine from 2e-3) at ub 0.6, from step 0. A hedge
  # on the horizon at 30k: a 30k run started now on the A100 ends about when one started on the first free desa slot
  'r3_K0.5_ub0.6_S64_lr2e-3_cos': dict(ub=0.6, lr=2e-3, sched='cosine', words=64),
  # wave 21 (fix, desa, to 30k from step 0): peak 3e-3 failed the 1/3 gate (1.9005 at 10k, +0.84% vs peak 2e-3), so
  # its slot runs the chosen recipe from step 0: S = 64 throughout (wave 16 trained its first 6k at S 16), cosine from 2e-3
  'r3_K0.5_ub0.5_S64_lr2e-3_cos': dict(ub=0.5, lr=2e-3, sched='cosine', words=64),
  # wave 22 (fix, A100, 6k then 30k): at LR 1e-3, ub 0.6's gap to ub 0.5 closed from +0.35% at 6k to +0.05% at 10k,
  # as ub 0.7 ada's did before it won at 30k. The best recipe at ub 0.7 (+1.0-1.2% at 6k on fix at LR 1e-3)
  'r3_K0.5_ub0.7_S64_lr2e-3_cos': dict(ub=0.7, lr=2e-3, sched='cosine', words=64),
  # wave 23 (fix, desa, to 30k from step 0, after 879486): the word-count axis past 64. S 4 -> 16 gained 3.2%,
  # 16 -> 64 0.4-0.7%, growing with steps and LR. The best recipe at S = 256
  'r3_K0.5_ub0.5_S256_lr2e-3_cos': dict(ub=0.5, lr=2e-3, sched='cosine', words=256),
  # wave 24 (fix, desa, to 30k from step 0, after 880403): with the best recipe ub 0.6 went from +0.42% at 6k to
  # -0.37% at 10k vs ub 0.5. The horizon curve past 0.7 (ub 1 was +5% at 6k at LR 3e-4)
  'r3_K0.5_ub0.8_S64_lr2e-3_cos': dict(ub=0.8, lr=2e-3, sched='cosine', words=64),
  # wave 25 (fix, desa, 10k to 30k): S 64 vs 16 from the same 6k checkpoint gained 0.42% at 10k and 0.53% at 30k
  # (1.712, the best final). Wave 20's (ub 0.6, the best 10k) keep-10000.ckpt copied in, the same cosine (its
  # scheduler state resumes) at S = 256
  'r3_K0.5_ub0.6_S64_lr2e-3_cos_S256': dict(ub=0.6, lr=2e-3, sched='cosine', words=256),
  # wave 26 (fix, desa, to 30k from step 0, in wave 23's slot: S 256 tied S 64 at 6k, 1.9780 vs 1.9772, at -22% speed).
  # The decay shape: wave 20's recipe (ub 0.6, S 64, peak 2e-3) with a linear decay to 0 at 30k. Its LR is below
  # cosine's until 16250 (both 1e-3 there) and above it after, so compare against wave 20 at 16k
  'r3_K0.5_ub0.6_S64_lr2e-3_lin': dict(ub=0.6, lr=2e-3, sched='linear', words=64),
  # wave 27 (fix, desa, to 30k from step 0, in wave 24's slot: ub 0.8 was +1.91% at 6k). The proposal rate on wave 20's
  # recipe: under trunc_exp it only shapes the importance-weighted time draw, and unif / 3 / 6 gave 2.207 / 2.126 /
  # 2.135 at 6k (ada, ub 0.5, LR 3e-4), whose parabola has its minimum near 4.2
  'r4_K0.5_ub0.6_S64_lr2e-3_cos': dict(rate=4, ub=0.6, lr=2e-3, sched='cosine', words=64),
  # wave 28 (fix, desa, to 30k from step 0, in wave 27's slot: rate 4 was +0.26% at 10k). Dropout 0 on wave 20's recipe:
  # 30k steps x 512 x 256 is ~8 epochs of TinyStories, and the bridge noise already regularizes
  'r3_K0.5_ub0.6_S64_lr2e-3_cos_do0': dict(ub=0.6, lr=2e-3, sched='cosine', words=64, dropout=0),
  # wave 29 (fix, desa, in wave 21's slot): wave 21 (cosine from 2e-3 at step 0, S 64, ub 0.5) finished at 1.7201 vs 880403's
  # 1.7121 (constant 2e-3 to 6k, then cosine over 24k): the peak held to 6k keeps the LR higher to the end. That schedule at
  # ub 0.6 (-0.4% vs ub 0.5 at 10k-16k): constant 2e-3 at S 64 to 6k (a stage), then its keep-6000.ckpt copied into the
  # cosine cell (LR_SCHED_FROM 6000, as 880403)
  'r3_K0.5_ub0.6_S64_lr2e-3': dict(ub=0.6, lr=2e-3, words=64),
  'r3_K0.5_ub0.6_S64_lr2e-3_const6k_cos': dict(ub=0.6, lr=2e-3, sched='cosine', sched_from=6000, words=64),
  # wave 30 (fix, A100, to 30k from step 0, in 896271's slot): wave 29's schedule in one job (LR_SCHED=wsd: warmup 2500,
  # 2e-3 to 6k, cosine to 0 at 30k; transformers.get_wsd_schedule) with dropout 0 (wave 28). Kept only if wave 28's
  # dropout 0 passes its 6k
  'r3_K0.5_ub0.6_S64_lr2e-3_wsd6k_do0': dict(ub=0.6, lr=2e-3, sched='wsd', words=64, dropout=0),
  # wave 31 (fix, A100, to 30k from step 0, in 888087's slot; 888087 finished at 1.7081): one of two, picked on wave 28's
  # 10k. Dropout 0 holds -> the peak held to 14k (LR_HOLD_STEPS 11500, cosine over the last 16k); dropout 0 overfits ->
  # dropout 0.05 on wave 30's schedule
  'r3_K0.5_ub0.6_S64_lr2e-3_wsd14k_do0': dict(ub=0.6, lr=2e-3, sched='wsd', hold=11500, words=64, dropout=0),
  'r3_K0.5_ub0.6_S64_lr2e-3_wsd6k_do0.05': dict(ub=0.6, lr=2e-3, sched='wsd', words=64, dropout=0.05),
  # wave 32 (fix, desa, to 30k from step 0, in wave 29's slot; wave 29 failed its 10k): the horizon under dropout 0. ub 0.6
  # lost at 6k (+0.42%) and won at 30k (-0.70%) vs ub 0.5; a better-fit model may move the optimum up. Wave 28 at ub 0.7
  'r3_K0.5_ub0.7_S64_lr2e-3_cos_do0': dict(ub=0.7, lr=2e-3, sched='cosine', words=64, dropout=0),
  # wave 33 (fix, desa, to 30k from step 0, in wave 25's slot): wave 26's linear decay (-0.56 / -0.29 / -0.11% vs cosine at
  # 6k / 10k / 16k) with dropout 0. Cancelled if wave 26's 30k final ties or loses to cosine (888087)
  'r3_K0.5_ub0.6_S64_lr2e-3_lin_do0': dict(ub=0.6, lr=2e-3, sched='linear', words=64, dropout=0),
  # wave 34 (fix, desa, to 30k from step 0, in wave 26's slot; wave 26 finished at 1.7053): the peak LR under dropout 0 and the
  # linear decay. At dropout 0.1 / cosine, 3e-3 trailed 2e-3 by +0.9 / +0.53% at 6k / 10k and was stopped while the gap shrank
  'r3_K0.5_ub0.6_S64_lr3e-3_lin_do0': dict(ub=0.6, lr=3e-3, sched='linear', words=64, dropout=0),
  # wave 35 (fix, desa, to 30k from step 0, in wave 33's slot; wave 33 failed its 16k: +0.42% vs cosine, same seed): the
  # peak LR on the cosine decay under dropout 0. Wave 34 (3e-3, linear) was level with 2e-3 at 6k (-0.11%)
  'r3_K0.5_ub0.6_S64_lr3e-3_cos_do0': dict(ub=0.6, lr=3e-3, sched='cosine', words=64, dropout=0),
  # wave 36 (fix, desa, to 30k from step 0, in wave 28's slot after 915962 ends): the LR axis past 3e-3 under dropout 0,
  # cosine. With wave 28 (2e-3) and wave 35 (3e-3) a three-point LR curve at the final recipe
  'r3_K0.5_ub0.6_S64_lr4e-3_cos_do0': dict(ub=0.6, lr=4e-3, sched='cosine', words=64, dropout=0),
  # wave 37 (fix, desa, to 30k from step 0, in wave 35's slot; wave 35 failed its 10k: +0.62% vs wave 28, same seed): the
  # LR axis below 2e-3 under cosine and dropout 0 (3e-3 / 4e-3 trail 2e-3 by +0.42 / +0.90% at 6k)
  'r3_K0.5_ub0.6_S64_lr1.5e-3_cos_do0': dict(ub=0.6, lr=1.5e-3, sched='cosine', words=64, dropout=0),
  # wave 38 (fix, desa, to 30k from step 0, in wave 36's slot; wave 36 stopped at 6k, behind wave 35, which failed its
  # 10k): wave 28 at dropout 0.05. Dropout 0's lead over 0.1 halved from 16k to 30k (-1.34% -> -0.74%)
  'r3_K0.5_ub0.6_S64_lr2e-3_cos_do0.05': dict(ub=0.6, lr=2e-3, sched='cosine', words=64, dropout=0.05),
  # wave 40 (fix, desa, to 30k from step 0, in wave 38's slot; wave 38 failed its 10k: +0.59% vs wave 28, same seed): the
  # hold length of wave 30's held peak (1.6901, -0.32% vs cosine). 2e-3 held to 10k (LR_HOLD_STEPS 7500), cosine over the
  # last 20k; between wave 30 (6k) and wave 31 (14k). Its 6k equals wave 30's schedule; no cosine gate after 6k
  'r3_K0.5_ub0.6_S64_lr2e-3_wsd10k_do0': dict(ub=0.6, lr=2e-3, sched='wsd', hold=7500, words=64, dropout=0),
  # wave 41 (fix, desa, to 30k from step 0, in wave 34's slot; wave 34 finished at 1.6876, the best): the peak LR above
  # 3e-3 under the linear decay and dropout 0 (linear 2e-3 trailed cosine at 16k and was stopped; 3e-3 won at 30k)
  'r3_K0.5_ub0.6_S64_lr4e-3_lin_do0': dict(ub=0.6, lr=4e-3, sched='linear', words=64, dropout=0),
  # waves 42 / 43 (fix, desa, to 30k from step 0, in the slots of waves 40 / 37, both stopped as dominated): wave 31 (2e-3
  # held to 14k, 1.6849) beat wave 30 (held to 6k, 1.6901), and wave 34 (3e-3, linear, 1.6876) beat 2e-3 cosine. More
  # steps at a high LR win at 30k, so: the hold to 20k (cosine over the last 10k), and wave 31's hold at peak LR 3e-3.
  # Their LR stays high after 6k, so only the 6k gate applies; they run to 30k
  'r3_K0.5_ub0.6_S64_lr2e-3_wsd20k_do0': dict(ub=0.6, lr=2e-3, sched='wsd', hold=17500, words=64, dropout=0),
  'r3_K0.5_ub0.6_S64_lr3e-3_wsd14k_do0': dict(ub=0.6, lr=3e-3, sched='wsd', hold=11500, words=64, dropout=0),
  # wave 44 (fix, desa, to 30k from step 0, in wave 37's GPUs): the corner of the 2 x 2 (peak 2e-3 / 3e-3 x hold 14k /
  # 20k) around wave 31, with waves 42 and 43
  'r3_K0.5_ub0.6_S64_lr3e-3_wsd20k_do0': dict(ub=0.6, lr=3e-3, sched='wsd', hold=17500, words=64, dropout=0),
}
ARMS = {   # arm: (curvature_bound, cells)
  'ada': (4, ['r3_K0.5', 'r1.5_K0.5', 'r5_K0.5', 'r6_K1',                                  # wave 1
              'r3_K0.5_ub0.5', 'r3_K0.5_ub1', 'r3_K0.5_S4', 'r3_K0.5_S64', 'r3_K0.5_bert',   # wave 2
              'r3_K0.5_ub0.25', 'r3_K0.5_ub0.35', 'r3_K0.5_ub0.7',                         # wave 4
              'r6_K0.5_ub0.5',                                                             # wave 5
              'r3_K0.5_ub0.5_bert', 'r3_K0.5_ub0.5_S64', 'r3_K0.5_unif0.5',                # wave 6
              'r6_K1_ub0.25',                                                              # wave 7
              'r3_K0.5_ub0.5_S4',                                                          # wave 8
              'r3_K0.5_ub0.5_lr1e-3']),                                                    # wave 14
  'fix': (1, ['r3_K0.5', 'r3_K0.5_ub1',   # ub1: added when fix led ada by 4% at 10k
              'r3_K0.5_lr6e-4', 'r3_K0.5_lr1e-3', 'r5_K0.5', 'r3_K0.5_S64', 'r3_K0.5_bert',   # wave 3
              'r3_K0.5_ub0.5',                                                             # wave 4
              'r6_K1_ub0.25', 'r3_K0.5_ub0.35', 'r3_K0.5_ub0.7',   # wave 7: fix led ada by 2.2% at ub 0.5, 6k
              'r3_K0.5_ub0.5_lr1e-3',                                                      # wave 8
              'r1.5_K0.25_ub1',                                                            # wave 9
              'r3_K0.5_ub0.5_lr6e-4',                                                      # wave 10
              'r3_K0.5_ub0.5_lr2e-3',                                                      # wave 11
              'r1.5_K0.25_ub1_lr1e-3',                                                     # wave 12
              'r3_K0.5_ub0.7_lr1e-3',                                                      # wave 13
              'r3_K0.5_ub0.5_S64_lr1e-3',                                                  # wave 14
              'r3_K0.5_ub0.5_lr2e-3_cos',                                                  # wave 15
              'r3_K0.5_ub0.5_lr2e-3_cos_S64',                                              # wave 16
              'r3_K0.5_ub0.5_S64_lr1e-3_cos', 'r3_K0.5_ub0.5_S64_lr3e-3_cos',              # waves 17, 18
              'r3_K0.5_ub0.6_lr1e-3', 'r3_K0.5_ub0.6_S64_lr2e-3_cos',                      # waves 19, 20
              'r3_K0.5_ub0.5_S64_lr2e-3_cos',                                              # wave 21
              'r3_K0.5_ub0.7_S64_lr2e-3_cos',                                              # wave 22
              'r3_K0.5_ub0.5_S256_lr2e-3_cos',                                             # wave 23
              'r3_K0.5_ub0.8_S64_lr2e-3_cos',                                              # wave 24
              'r3_K0.5_ub0.6_S64_lr2e-3_cos_S256',                                         # wave 25
              'r3_K0.5_ub0.6_S64_lr2e-3_lin',                                              # wave 26
              'r4_K0.5_ub0.6_S64_lr2e-3_cos',                                              # wave 27
              'r3_K0.5_ub0.6_S64_lr2e-3_cos_do0',                                          # wave 28
              'r3_K0.5_ub0.6_S64_lr2e-3', 'r3_K0.5_ub0.6_S64_lr2e-3_const6k_cos',          # wave 29
              'r3_K0.5_ub0.6_S64_lr2e-3_wsd6k_do0',                                        # wave 30
              'r3_K0.5_ub0.6_S64_lr2e-3_wsd14k_do0', 'r3_K0.5_ub0.6_S64_lr2e-3_wsd6k_do0.05',    # wave 31
              'r3_K0.5_ub0.7_S64_lr2e-3_cos_do0',                                          # wave 32
              'r3_K0.5_ub0.6_S64_lr2e-3_lin_do0',                                        # wave 33
       'r3_K0.5_ub0.6_S64_lr3e-3_lin_do0',                                        # wave 34
       'r3_K0.5_ub0.6_S64_lr3e-3_cos_do0',                                        # wave 35
       'r3_K0.5_ub0.6_S64_lr4e-3_cos_do0',                                        # wave 36
       'r3_K0.5_ub0.6_S64_lr1.5e-3_cos_do0',                                      # wave 37
       'r3_K0.5_ub0.6_S64_lr2e-3_cos_do0.05',                                     # wave 38
       'r3_K0.5_ub0.6_S64_lr2e-3_wsd10k_do0',                                     # wave 40
       'r3_K0.5_ub0.6_S64_lr4e-3_lin_do0',                                        # wave 41
       'r3_K0.5_ub0.6_S64_lr2e-3_wsd20k_do0',                                     # wave 42
       'r3_K0.5_ub0.6_S64_lr3e-3_wsd14k_do0',                                     # wave 43
       'r3_K0.5_ub0.6_S64_lr3e-3_wsd20k_do0']),                                   # wave 44
  'ada2': (2, ['r3_K0.5_ub0.5']),   # wave 10: CURV_BOUND 2 (lambda in [1/2, 2]) between fix (1) and ada (4)
}
WAVE1 = [(c, 'ada') for c in ARMS['ada'][1][:4]] + [('r3_K0.5', 'fix')]   # trained to 30k in one go
STAGED = [(c, a) for a in ARMS for c in ARMS[a][1] if (c, a) not in WAVE1]   # (cell, arm), early-stopped
MAX_STEPS = 30000
STAGE_VAL = 0.1   # fraction of the validation set behind a stage's NELBO (eval/nelbo_{steps}.json)


def cfg(cell):
  return {**ANCHOR, **CELLS[cell]}


def tag_of(cell, arm, seed):
  return f'{cell}_curv-{arm}_seed-{seed}'


def knobs(cell, arm):
  c = cfg(cell)
  if c['ub'] is None:   # exp: the horizon T = ln(1/eps)/rate
    time = f'TIME_CONVERSION_MODE=exp T_MAX={math.log(1 / EPS) / c["rate"]:.4f}'
  else:                 # trunc_exp (or unif): the proposal and the horizon end at ub
    time = (f'TIME_CONVERSION_MODE={c.get("mode", "trunc_exp")} TIME_RANGE_UPPER_BOUND={c["ub"]:g} '
            f'T_MAX={c["ub"]:g}')
  return (f'FORWARD_TYPE=naive READOUT_PRECISION=float32 EMBED_DIM=48 FACTOR_DIM=3 GAUSS_CURV={c["k"]:g} '
          f'CURV_BOUND={ARMS[arm][0]} PROPOSAL_RATE={c["rate"]:g} {time} GLOBAL_BATCH=512 SEQ_LEN=256')


def header(s):
  """The environment every job of the site starts with."""
  return textwrap.dedent(f'''\
    #!/bin/bash
    set -e
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 SLURM_JOB_NAME=bash NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
    export TMPDIR={s['tmp']}/{JOB}_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR; trap 'rm -rf "$TMPDIR"' EXIT
    export PATH={s['envbin']}:$PATH
    {s['env']}
    cd {s['repo']}
    hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    ''')


def stage_score(s, cell, arm, seed, steps, ckpt='last.ckpt'):
  """The early-stopping score: the exact NELBO of <tag>/checkpoints/<ckpt> on STAGE_VAL of the
  validation set (the same sequences and t draws for every cell) -> eval/nelbo_{steps}.json."""
  tag = tag_of(cell, arm, seed)
  tdir = f'{s["out"]}/{NAME}/{tag}'
  return textwrap.dedent(f'''\
    echo "[$(date)] NELBO {tag} step {steps}"
    {knobs(cell, arm)} LOSS_TYPE=elbo SEED=1 EVAL_BS=16 RUN_SAMPLE_EVAL=false LIMIT_VAL_BATCHES={STAGE_VAL} \\
      PPL_JSON={tdir}/eval/nelbo_{steps}.json CKPT_PATH={tdir}/checkpoints/{ckpt} DEVICES=1 \\
      CACHE_DIR={s["cache"]} OUTPUT_DIR={tdir}/eval/step{steps} bash scripts/sample/tinystories/hbfm.sh
    echo "[$(date)] DONE {tag} step {steps}"
    ''')


def body(s, cell, arm, seed, steps=MAX_STEPS):
  """Train to `steps` (resuming from last.ckpt), then the full evals at MAX_STEPS, or at a stage
  (steps < MAX_STEPS) only the exact NELBO on STAGE_VAL of the validation set."""
  tag = tag_of(cell, arm, seed)
  tdir = f'{s["out"]}/{NAME}/{tag}'
  c = cfg(cell)
  ckpt = f'CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval DEVICES=1 CACHE_DIR={s["cache"]}'
  train = header(s) + textwrap.dedent(f'''\
    echo "[$(date)] TRAIN {tag} to step {steps}"
    python experiment.py {tdir} {NAME} {tag}
    LR={c['lr']:g} LR_SCHED={c['sched']} LR_DECAY_STEPS={MAX_STEPS} LR_SCHED_FROM={c['sched_from']} LR_HOLD_STEPS={c['hold']} DROPOUT={c['dropout']:g} SEED={seed} {knobs(cell, arm)} LOSS_TYPE=elbo ELBO_NUM_WORDS={c['words']} \\
      PRETRAINED_EMBED={c['embed']} CACHE_DIR={s['cache']} \\
      OUTPUT_DIR={tdir} RUN_NAME={tag} WANDB_GROUP={NAME} DEVICES={s['devices']} PER_GPU_BS={s['per_gpu_bs']} \\
      NUM_WORKERS={s['workers']} MAX_STEPS={steps} CKPT_EVERY=1000 KEEP_EVERY=2000 bash scripts/train/tinystories/hbfm.sh
    ''')
  if steps < MAX_STEPS:   # a stage: the early-stopping NELBO only
    return train + stage_score(s, cell, arm, seed, steps)
  text = train + textwrap.dedent(f'''\
    echo "[$(date)] EVAL {tag}"
    {knobs(cell, arm)} TOPK_VELOCITY=1 STEPS=180 {ckpt} bash scripts/sample/tinystories/hbfm.sh
    echo "[$(date)] NELBO {tag}"
    {knobs(cell, arm)} LOSS_TYPE=elbo SEED=1 EVAL_BS=16 RUN_SAMPLE_EVAL=false PPL_JSON={tdir}/eval/nelbo.json {ckpt} \\
      bash scripts/sample/tinystories/hbfm.sh
    ''')
  if ARMS[arm][0] > 1:   # lambda per keep-{step}.ckpt -> <cell>/curvature.json
    text += f'python experiments/hbfm_rlf_ada_curv_tinystories_256/curvature_ts.py {tdir}\n'
  return text + f'echo "[$(date)] DONE {tag}"\n'


def local(s):
  return s['host'].split('@')[0] == getpass.getuser()


def run(s, cmd, stdin=None):
  """A shell command at the site's login node (over ssh unless it is this account; a login shell for slurm)."""
  cmd = ['bash', '-lc', cmd] if local(s) else [
    'ssh', '-o', 'BatchMode=yes', '-i', KEY, s['host'], "bash -lc '" + cmd.replace("'", "'\\''") + "'"]
  return subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=120)


def offline_rows(path):
  """The config and the history rows of an offline wandb run file (out/<tag>/wandb/offline-run-*/*.wandb)."""
  from wandb.proto import wandb_internal_pb2
  from wandb.sdk.internal import datastore
  config, rows = {}, []
  if os.path.getsize(path) == 0:   # a restart preempted before its first write
    return config, rows
  ds = datastore.DataStore()
  ds.open_for_scan(path)
  while True:
    try:
      data = ds.scan_data()
    except AssertionError:   # the tail of a file still being written
      break
    if data is None:
      break
    r = wandb_internal_pb2.Record()
    r.ParseFromString(data)
    if r.HasField('config'):
      config.update((i.key, json.loads(i.value_json)) for i in r.config.update if not i.key.startswith('_'))
    elif r.HasField('history'):
      row = ((i.key or '/'.join(i.nested_key), json.loads(i.value_json)) for i in r.history.item)
      rows.append({k: v for k, v in row if not k.startswith('_')})
  return config, rows


def wandb_sync(out, cells=None):
  """Each cell under out/ as one W&B run of project NAME (id: the tag with . -> p): the history rows
  of all its offline runs (stages and preemption restarts, in launch order), plus the stage NELBOs
  and the final evals in the summary. Only the rows the W&B run does not have yet are uploaded, so a
  cell still training can be synced again later. (`wandb sync --append` of a restart overwrites the
  earlier run's rows: both count _step from 0.) The API key comes from <REPO>/.wandb.env."""
  import wandb
  if 'WANDB_API_KEY' not in os.environ:
    os.environ.update(line.strip().split('=', 1) for line in open(f'{REPO}/.wandb.env') if '=' in line)
  os.environ.setdefault('WANDB_CONFIG_DIR', f'{SFM_HOME}/cache/wandb')
  os.environ.setdefault('WANDB_CACHE_DIR', f'{SFM_HOME}/cache/wandb')
  for d in sorted(glob.glob(f'{out}/*/wandb')):
    tdir = os.path.dirname(d)
    tag = os.path.basename(tdir)
    cell, arm = tag.split('_curv-')[0], tag.split('_curv-')[1].split('_seed-')[0]
    if cell not in CELLS or (cells and cell not in cells):
      continue
    config, rows = {}, []
    for f in sorted(glob.glob(f'{d}/offline-run-*/*.wandb')):
      c, r = offline_rows(f)
      config, rows = config or c, rows + r
    run = wandb.init(project=NAME, id=tag.replace('.', 'p'), name=tag, group=NAME, resume='allow',
                     config=dict(config, cell=cell, arm=arm), dir=f'{SFM_HOME}/cache/wandb',
                     settings=wandb.Settings(silent=True, console='off'))
    run.define_metric('trainer/global_step')
    run.define_metric('*', step_metric='trainer/global_step', step_sync=True)
    done = run.summary.get('synced_rows', 0)
    for row in rows[done:]:
      run.log(row)
    summary = {'synced_rows': len(rows)}
    for f in glob.glob(f'{tdir}/eval/nelbo_*.json'):   # the stage scores (10% of the validation set)
      v = json.load(open(f))
      k = f.split('nelbo_')[1][:-5]
      summary.update({f'stage/nelbo@{k}': v['val/nll'], f'stage/nelbo_std@{k}': v.get('val/loss_std')})
    for name, key, f in [('nelbo', 'val/nll', 'nelbo.json'), ('val_ppl', 'val/ppl', 'ppl.json'),
                         ('gen_ppl', 'gen_ppl_first_chunk_retok', 'samples_genppl.json'),
                         ('entropy', 'entropy', 'samples_genppl.json')]:
      if os.path.exists(f'{tdir}/eval/{f}'):
        summary[f'final/{name}'] = json.load(open(f'{tdir}/eval/{f}')).get(key)
    if os.path.exists(f'{tdir}/curvature.json'):
      summary['final/lambda'] = json.load(open(f'{tdir}/curvature.json')).get(str(MAX_STEPS), {}).get('mean')
    run.summary.update(summary)
    run.finish()
    print(f'  {tag}: {len(rows) - done} new rows ({len(rows)} in all)')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--site', choices=sorted(SITES))
  ap.add_argument('--arms', default='ada,fix')
  ap.add_argument('--cells', default=None, help='comma list; default every cell of the arm')
  ap.add_argument('--steps', type=int, default=MAX_STEPS, help='train to this step (< MAX_STEPS = a stage)')
  ap.add_argument('--seeds', default='1')
  ap.add_argument('--sync-code', action='store_true')
  ap.add_argument('--pull', action='store_true')
  ap.add_argument('--wandb-sync', action='store_true', help='upload the cells under the local outputs (again: only what is new)')
  ap.add_argument('--score-keep', action='store_true',
                  help='1-GPU job per cell: the stage score of its keep-{steps}.ckpt (no training)')
  ap.add_argument('--after', default=None,
                  help='job id: queue the next stage behind its running stage (afterok), so a freed slot is not lost')
  ap.add_argument('--dry-run', action='store_true')
  args = ap.parse_args()
  ssh = f'ssh -o BatchMode=yes -i {KEY}'

  s = SITES[args.site]
  remote = f"{s['host']}:{s['repo']}"
  here = next((f'{x["out"]}/{NAME}' for x in SITES.values() if local(x)), OUT)   # this account's outputs
  if args.sync_code:   # the code only; data, outputs and checkpoints stay where they are
    inc = ' '.join(f"--include='/{p}'" if '*' in p else f"--include='/{p}/***'" for p in CODE)
    os.system(f"rsync -a --exclude='__pycache__/' --include='/experiments/' {inc} --exclude='/*' "
              f"--exclude='/experiments/*' -e '{ssh}' {REPO}/ {remote}/")
    return
  if args.pull:        # results (eval JSONs, hydra configs, logs; no checkpoints) -> local
    # ch2263's stale stubs of the two wave-1 cells trained on desa must not overwrite the real ones
    stale = ' '.join(f"--exclude='/{t}/'" for t in ('r6_K1_curv-ada_seed-1', 'r3_K0.5_curv-fix_seed-1'))
    os.system(f"rsync -a --exclude='checkpoints' --exclude='wandb/' {stale * (args.site == 'a100')} "
              f"-e '{ssh}' {s['host']}:{s['out']}/{NAME}/ {here}/")
    os.system(f"rsync -a -e '{ssh}' {remote}/experiments/{NAME}/logs/ {EXP}/logs/{args.site}/")
    return

  queued = set()   # a cell queued at either site counts
  if not args.dry_run:
    for site in SITES.values():
      queued |= set(run(site, f'squeue -h -u {site["host"].split("@")[0]} -o %j').stdout.split())
  logs = f'{s["repo"]}/experiments/{NAME}/logs'
  out = f'{s["out"]}/{NAME}' if local(s) else here
  if args.wandb_sync:
    wandb_sync(out, args.cells and args.cells.split(','))
    return
  result = 'nelbo.json' if args.steps >= MAX_STEPS else f'nelbo_{args.steps}.json'
  for arm in args.arms.split(','):
    for cell in ARMS[arm][1]:
     if args.cells and cell not in args.cells.split(','):
       continue
     for seed in args.seeds.split(','):
      tag = tag_of(cell, arm, seed)
      if args.score_keep:
        if local(s) and not os.path.exists(f'{out}/{tag}/checkpoints/keep-{args.steps}.ckpt'):
          print(f'  skip {tag} (no keep-{args.steps}.ckpt here)')
          continue
        jobname, sbatch = f'{JOB}_score_{tag}_{args.steps}', s['sbatch1']
        text = header(s) + stage_score(s, cell, arm, seed, args.steps, f'keep-{args.steps}.ckpt')
      else:
        jobname, sbatch, text = f'{JOB}_{tag}', s['sbatch'], body(s, cell, arm, seed, args.steps)
      if os.path.exists(f'{out}/{tag}/eval/{result}') or (jobname in queued and not args.after):
        print(f'  skip {tag}')
        continue
      if args.dry_run:
        print(f'--- {jobname} ({sbatch})\n{text}')
        continue
      limit = '06:00:00' if args.score_keep else '7-00:00:00'   # short scoring jobs can backfill
      after = f'--dependency=afterok:{args.after} ' if args.after else ''
      r = run(s, f'mkdir -p {logs} && sbatch --job-name={jobname} {after}{sbatch} --time={limit} --requeue '
                 f'--output={logs}/{jobname}_%j.log', stdin=text)
      print(f'  {jobname}: {(r.stdout + r.stderr).strip()}')


if __name__ == '__main__':
  main()
