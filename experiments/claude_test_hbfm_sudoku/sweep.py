#!/usr/bin/env python
"""claude_test_hbfm_sudoku — HBFM (hyperbolic bridge DLM) LR x seed grid on hard Sudoku.

Spec: experiments/claude_test_hbfm_sudoku/setup.md.
Grid (36 cells):  K in {-0.5 (setup.md), -1.0 (earlier submission, kept for comparison)}
                  x  LR in {3e-4, 5e-4, 1e-3}  x  noise in {log-linear, log-linear-adaptive}
                  x  seed in {1, 2, 3}
Recipe: scripts/train/sudoku/hbfm.sh — tiny-hyperbolic-dit 512/8/8, hard difficulty,
denoising CE, 20k steps, batch 256, bf16, EMA 0.9999, AdamW wd 0 / grad-clip 1.0 — with
the spec's manifold (H^3_K)^3 (prod_factor_dim [3,3,3], curvature [K,K,K], i.e.
model.embed_dim 9 lifted into the DiT by in_proj) and algo.time_exp_rate 0.01 (the
scripts take the rate in unit time, so UNIT_PROPOSAL_RATE = 0.01 / |K|). K, optim.lr
(LR), the noise schedule
(NOISE: plain log-linear vs the AdaptiveSchedule wrapper, which refits the heat-time
proposal from the loss profile) and seed (SEED) vary.
Eval: scripts/sample/sudoku/hbfm.sh (sudoku_eval, 180 steps, exact velocity,
top_k_velocity=-1, greedy last step) with the same knobs and t_max = UNIT_T_MAX / |K|
(physical), UNIT_T_MAX = 3 chosen so the Bayes posterior on (H^3)^3 is resolved at the
horizon (unit time is |K| t, so the physical horizon scales with 1/|K|).

ORCHESTRATION ONLY — each cell calls the single-run shared scripts:
  scripts/train/sudoku/hbfm.sh   (LR / NOISE / SEED / UNIT_PROPOSAL_RATE / EMBED_DIM /
                                  PROD_FACTOR_DIM / PROD_FACTOR_CURV env knobs)
  scripts/sample/sudoku/hbfm.sh  (the same + UNIT_T_MAX / CKPT_PATH)
Idempotent + resumable (skips done/queued cells; resubmits auto-resume from last.ckpt).

Usage:  python sweep.py [--dry-run]      then  python report.py  ->  RESULTS.md
"""
import argparse
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/claude_test_hbfm_sudoku'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/claude_test_hbfm_sudoku'

CURVS = ['-0.5', '-1.0']      # setup.md says -0.5; -1.0 is the earlier submission, kept for comparison
LRS = ['3e-4', '5e-4', '1e-3']
NOISES = {'0': 'log-linear', '1': 'log-linear-adaptive'}   # ada-0 / ada-1 in the cell tag
SEEDS = ['1', '2', '3']
TIME_EXP_RATE = 0.01          # setup.md: algo.time_exp_rate (physical); the scripts take UNIT_PROPOSAL_RATE = rate / |K|
UNIT_T_MAX = '3.0'            # sampling horizon (unit time): the Bayes posterior on (H^3)^3 is resolved by t ~ 3 (EXPERIMENT.md)
MAX_STEPS = 20000
CKPT_EVERY = 5000


# Follow-up after the diagnosis (RESULTS.md, H1): at the spec's rate 0.01 only ~3% of training
# heat times fall where the model must use the prompt, and a 5k-step run at rate 3 already
# reaches 30.4% boards (vs 0% at rate 0.01). These cells re-run the LR-3e-4 column at
# algo.time_exp_rate = 3 (both curvatures, both schedules; the adaptive ones with the repo's
# refit recipe and the |alpha'_t| weight), tagged `_rate-3`.
EXTRA_RATES = ['3']


def geometry(k, rate=TIME_EXP_RATE):
    kf = float(k)
    return (f"EMBED_DIM=9 PROD_FACTOR_DIM='[3,3,3]' PROD_FACTOR_CURV='[{kf},{kf},{kf}]' "
            f"GAUSS_CURV={kf} UNIT_PROPOSAL_RATE={float(rate) / abs(kf):g}")


def tag_of(k, lr, ada, seed, rate=None):
    return f'lr-{lr}_ada-{ada}_k{k}_seed-{seed}' + (f'_rate-{rate}' if rate else '')


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(k, lr, ada, seed, rate=None):
    tag = tag_of(k, lr, ada, seed, rate)
    noise = NOISES[ada]
    geom = geometry(k, rate or TIME_EXP_RATE)
    tdir = f'{OUT}/{tag}'
    edir = f'{tdir}/eval'
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export WANDB_MODE=offline
        export TMPDIR=/home/sc3379/tmp/hbfm_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR
        trap 'rm -rf "$TMPDIR"' EXIT
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        LR={lr} NOISE={noise} SEED={seed} {geom} \\
            OUTPUT_DIR={tdir} DEVICES=1 PER_GPU_BS=256 GLOBAL_BATCH=256 \\
            MAX_STEPS={MAX_STEPS} CKPT_EVERY={CKPT_EVERY} \\
            bash scripts/train/sudoku/hbfm.sh
        echo "[$(date)] EVAL {tag}"
        NOISE={noise} SEED={seed} {geom} UNIT_T_MAX={UNIT_T_MAX} GLOBAL_BATCH=256 \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 \\
            bash scripts/sample/sudoku/hbfm.sh
        echo "[$(date)] DONE {tag}"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    cells = [(k, lr, ada, seed, None) for k, lr, ada, seed in
             itertools.product(CURVS, LRS, NOISES.keys(), SEEDS)]
    cells += [(k, '3e-4', ada, seed, rate) for k, ada, seed, rate in
              itertools.product(CURVS, NOISES.keys(), SEEDS, EXTRA_RATES)]
    print(f'claude_test_hbfm_sudoku: {len(cells)} cells ({len(CURVS)} K x {len(LRS)} lr x '
          f'{len(NOISES)} noise x {len(SEEDS)} seeds at rate {TIME_EXP_RATE}, plus the LR-3e-4 '
          f'column at rate {EXTRA_RATES})')
    if args.dry_run:
        for k, lr, ada, seed, rate in cells:
            print('  hbfm_sudoku_' + tag_of(k, lr, ada, seed, rate))
        print('\n--- example body ---\n' + job_body(*cells[-1]))
        return
    active = active_jobnames()
    n_sub = n_skip = 0
    for k, lr, ada, seed, rate in cells:
        tag = tag_of(k, lr, ada, seed, rate)
        jobname = f'hbfm_sudoku_{tag}'
        if os.path.exists(f'{OUT}/{tag}/eval/results.json') or jobname in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=jobname, partition='thickstun,desa', gres='gpu:1',
                      ntasks=1, cpus_per_task=8, mem='64G', time='12:00:00',
                      exclude='desa-compute-01', output=f'{LOGS}/{tag}_%j.log')
        jid = slurm.sbatch(job_body(k, lr, ada, seed, rate))
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
