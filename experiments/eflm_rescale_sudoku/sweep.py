#!/usr/bin/env python
"""sweep.py — EFLM fixed-embedding-norm (R) sweep on sudoku hard.

Hypothesis (see EXPERIMENT.md): the embedding norm decides the decoding time
t*(R); larger R -> later/sharper transition, smaller R -> earlier/smoother.
Pins every word-embedding norm to R via algo.rho_min = algo.rho_max = R and
sweeps R x {naive, adaptive noise} x LR x seed on sudoku hard:

  naive : scripts/train|sample/sudoku/eflm_rescale.sh           (log-linear)
  ada   : scripts/train|sample/sudoku/eflm_rescale_adaptive.sh  (log-linear-
          adaptive, no truncation)

Usage:  python experiments/eflm_rescale_sudoku/sweep.py
            [--rhos ...] [--adas naive ada] [--lrs 3e-4 5e-4 1e-3]
            [--difficulties hard] [--seeds 1 2 3] [--dry-run]
Idempotent: skips a cell whose eval/results.json exists or whose job name is
already in squeue; resubmitting auto-resumes from last.ckpt.  Checkpoints are
KEPT for the post-hoc loss-geometry analysis.
"""
import argparse
import getpass
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev1/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'

RHOS = ['0.1', '0.5', '1', '1.5', '2.0', '5.0', '8', '16', '22', '32']
LRS = ['3e-4', '5e-4', '1e-3']
SEEDS = ['1', '2', '3']
# adaptive arm -> train/sample script stem (same stem for both passes)
STEMS = {'naive': 'eflm_rescale', 'ada': 'eflm_rescale_adaptive'}


def tag_of(ada, rho, lr, difficulty, seed):
    return f'eflmrs_{ada}_r-{rho}_lr-{lr}_d-{difficulty}_rs{seed}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(ada, rho, lr, tdir, difficulty, seed):
    stem = STEMS[ada]
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export TORCHDYNAMO_DISABLE=1
        export PATH={ENVBIN}:$PATH
        # /home/sc3379 is full (100%); redirect temp + caches to node-local
        # /tmp so wandb/matplotlib import-time temp dirs don't ENOSPC.
        export TMPDIR=/tmp
        export MPLCONFIGDIR=/tmp/mpl-$SLURM_JOB_ID
        export WANDB_CACHE_DIR=/tmp/wandb-cache-$SLURM_JOB_ID
        mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$WANDB_CACHE_DIR"
        cd {REPO}
        if [ -f {tdir}/eval/results.json ]; then
            echo "[$(date)] cell already completed elsewhere -> no-op"; exit 0
        fi
        echo "[$(date)] TRAIN {ada} r={rho} lr={lr} on $(hostname)"
        RHO={rho} LR={lr} DIFFICULTY={difficulty} SEED={seed} OUTPUT_DIR={tdir} DEVICES=1 \\
            bash scripts/train/sudoku/{stem}.sh
        echo "[$(date)] EVAL"
        RHO={rho} DIFFICULTY={difficulty} SEED={seed} CKPT_PATH={tdir}/checkpoints/last.ckpt \\
            OUTPUT_DIR={tdir}/eval DEVICES=1 \\
            bash scripts/sample/sudoku/{stem}.sh
        # checkpoints are kept: the loss-geometry analysis (EXPERIMENT.md
        # success criterion 1) reads checkpoints/last.ckpt post-hoc.
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--rhos', nargs='+', default=RHOS, choices=RHOS)
    ap.add_argument('--adas', nargs='+', default=list(STEMS), choices=list(STEMS))
    ap.add_argument('--lrs', nargs='+', default=LRS, choices=LRS)
    ap.add_argument('--difficulties', nargs='+', default=['hard'],
                    choices=['easy', 'medium', 'hard'])
    ap.add_argument('--seeds', nargs='+', default=SEEDS)
    args = ap.parse_args()

    exp = f'{REPO}/experiments/eflm_rescale_sudoku'
    logs = f'{exp}/logs'
    out = f'{REPO}/outputs/eflm_rescale_sudoku'
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)

    cells = list(itertools.product(args.adas, args.rhos, args.lrs,
                                   args.difficulties, args.seeds))
    print(f'eflm_rescale_sudoku: {len(cells)} cells '
          f'({len(args.adas)} ada x {len(args.rhos)} R x {len(args.lrs)} lr x '
          f'{len(args.difficulties)} difficulty x {len(args.seeds)} seed)')
    if args.dry_run:
        for ada, rho, lr, diff, seed in cells:
            print(f'  {tag_of(ada, rho, lr, diff, seed)}')
        ada, rho, lr, diff, seed = cells[0]
        print('\n--- example body (first cell) ---\n'
              + job_body(ada, rho, lr, f'{out}/{tag_of(ada, rho, lr, diff, seed)}',
                         diff, seed))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for ada, rho, lr, diff, seed in cells:
        tag = tag_of(ada, rho, lr, diff, seed)
        if os.path.exists(f'{out}/{tag}/eval/results.json') or tag in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=tag, output=f'{logs}/{tag}_%j.log',
                      partition='thickstun,desa', exclude='desa-compute-01',
                      gres='gpu:1', ntasks=1, cpus_per_task=2, mem='16G',
                      time='06:00:00')
        jid = slurm.sbatch(job_body(ada, rho, lr, f'{out}/{tag}', diff, seed),
                           sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
