#!/usr/bin/env python
"""sweep_trunc.py — round 2: truncation x radius on the rescaled EFLM.

Round 1 showed the fixed (naive) schedule collapses at large R because the
decode transition t*(R) moves late and the schedule under-samples it, while
the adaptive schedule rescues large R. Round 2 tests the *static* fix:
truncate the schedule at the R-dependent Eq. 17 bound

    alpha*(R) = alpha_star_euclidean(V=12, embed_norm=R)   (noise_schedules.py)
              = 1 - t*(R) from visualization/codebook_signal_vs_lossgeo.py
                (both derivations agree to 3 decimals at d=512)

so training never samples the trivial high-signal region alpha > alpha*.
Per R we try ALPHA_MAX in {alpha*-0.1, alpha*, alpha*+0.1} (clipped to
[0.05, 0.95]) to probe whether the theory point, a tighter, or a wider band
is best. Baselines for "does it improve ACC": the round-1 naive (no-trunc)
and ada cells at the same R/LR (outputs/eflm_rescale_sudoku/eflmrs_*).

Usage:  python experiments/eflm_rescale_sudoku/sweep_trunc.py
            [--rhos 1 2 5 8 16 32] [--offsets -0.1 0 0.1]
            [--lrs 5e-4 1e-3] [--seeds 1 2 3] [--dry-run]
Idempotent: skips cells whose eval/results.json exists or whose job name is
in squeue; resubmit auto-resumes from last.ckpt. Checkpoints are KEPT.
"""
import argparse
import getpass
import itertools
import os
import subprocess
import sys
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev1/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
sys.path.insert(0, REPO)
from noise_schedules import alpha_star_euclidean  # noqa: E402

V = 12
RHOS = ['1', '2', '5', '8', '16', '32']
OFFSETS = ['-0.1', '0', '0.1']
LRS = ['5e-4', '1e-3']
SEEDS = ['1', '2', '3']
STEM = 'eflm_rescale_truncated'


def alpha_max_of(rho, offset):
    a = alpha_star_euclidean(V, embed_norm=float(rho)) + float(offset)
    return f'{min(0.95, max(0.05, a)):.3f}'


def tag_of(rho, am, lr, seed):
    return f'eflmrst_r-{rho}_am-{am}_lr-{lr}_d-hard_rs{seed}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(rho, am, lr, tdir, seed):
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export TORCHDYNAMO_DISABLE=1
        export PATH={ENVBIN}:$PATH
        # temp + caches on node-local /tmp (shared /home has run full before)
        export TMPDIR=/tmp
        export MPLCONFIGDIR=/tmp/mpl-$SLURM_JOB_ID
        export WANDB_CACHE_DIR=/tmp/wandb-cache-$SLURM_JOB_ID
        mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$WANDB_CACHE_DIR"
        cd {REPO}
        if [ -f {tdir}/eval/results.json ]; then
            echo "[$(date)] cell already completed elsewhere -> no-op"; exit 0
        fi
        echo "[$(date)] TRAIN r={rho} alpha_max={am} lr={lr} on $(hostname)"
        RHO={rho} ALPHA_MAX={am} LR={lr} DIFFICULTY=hard SEED={seed} \\
            OUTPUT_DIR={tdir} DEVICES=1 \\
            bash scripts/train/sudoku/{STEM}.sh
        echo "[$(date)] EVAL"
        RHO={rho} ALPHA_MAX={am} DIFFICULTY=hard SEED={seed} \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt \\
            OUTPUT_DIR={tdir}/eval DEVICES=1 \\
            bash scripts/sample/sudoku/{STEM}.sh
        # checkpoints kept for the loss-geometry analysis.
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--rhos', nargs='+', default=RHOS, choices=RHOS)
    ap.add_argument('--offsets', nargs='+', default=OFFSETS, choices=OFFSETS)
    ap.add_argument('--lrs', nargs='+', default=LRS, choices=LRS)
    ap.add_argument('--seeds', nargs='+', default=SEEDS)
    args = ap.parse_args()

    exp = f'{REPO}/experiments/eflm_rescale_sudoku'
    logs = f'{exp}/logs'
    out = f'{REPO}/outputs/eflm_rescale_sudoku'
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)

    cells = [(rho, alpha_max_of(rho, off), lr, seed)
             for rho, off, lr, seed in itertools.product(
                 args.rhos, args.offsets, args.lrs, args.seeds)]
    print(f'eflm_rescale_sudoku trunc: {len(cells)} cells '
          f'({len(args.rhos)} R x {len(args.offsets)} offset x '
          f'{len(args.lrs)} lr x {len(args.seeds)} seed)')
    if args.dry_run:
        for rho, am, lr, seed in cells:
            print(f'  {tag_of(rho, am, lr, seed)}')
        rho, am, lr, seed = cells[0]
        print('\n--- example body (first cell) ---\n'
              + job_body(rho, am, lr, f'{out}/{tag_of(rho, am, lr, seed)}', seed))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for rho, am, lr, seed in cells:
        tag = tag_of(rho, am, lr, seed)
        if os.path.exists(f'{out}/{tag}/eval/results.json') or tag in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=tag, output=f'{logs}/{tag}_%j.log',
                      partition='thickstun,desa', exclude='desa-compute-01',
                      gres='gpu:1', ntasks=1, cpus_per_task=2, mem='16G',
                      time='06:00:00')
        jid = slurm.sbatch(job_body(rho, am, lr, f'{out}/{tag}', seed),
                           sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
