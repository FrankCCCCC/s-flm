#!/usr/bin/env python
"""sweep.py — learned (MuLAN-style) EFLM noise schedule on sudoku hard.

See EXPERIMENT.md.  Sweeps the fixed embedding norm R x schedule scope x seed:

  positional : gamma_phi(c, t) per position -> a learned decoding order
  global     : gamma_phi(c, t) per sequence -> control arm (H2)

Both use scripts/{train,sample}/sudoku/eflm_rescale_variational.sh; the scope is passed
through VAR_SCOPE.  The fixed-schedule baseline at the same R lives in
experiments/eflm_rescale_sudoku (`naive` arm).

Usage:  python experiments/vmulan_sudoku/sweep.py
            [--rhos 5.0 8.0 16.0] [--scopes positional global]
            [--lrs 3e-4] [--difficulties hard] [--seeds 1 2 3] [--dry-run]
Idempotent: skips a cell whose eval/results.json exists or whose job name is
already in squeue; resubmitting auto-resumes from last.ckpt.  Checkpoints are
KEPT — the learned schedule is the object of study.
"""
import argparse
import getpass
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'

RHOS = ['5.0', '8.0', '16.0']
SCOPES = ['positional', 'global']
LRS = ['3e-4']
SEEDS = ['1', '2', '3']
CONTEXTS = ['prompt']    # prompt (MuLAN D.1, clue-adaptive) / dlm-state (D.2 ablation)
DEGREES = ['5', '9']     # 5 = fixed-implementation reference, 9 = higher order
STEM = 'eflm_rescale_variational'


def tag_of(ctx, degree, scope, rho, lr, difficulty, seed):
    # Round-1 (dlm-state, degree 5) tags were vmulan_{scope}_...; the
    # context/degree prefix keeps round-2 cells from colliding with them.
    return f'vmulan_{ctx}_deg{degree}_{scope}_r-{rho}_lr-{lr}_d-{difficulty}_rs{seed}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(ctx, degree, scope, rho, lr, tdir, difficulty, seed):
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export TORCHDYNAMO_DISABLE=1
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        if [ -f {tdir}/eval/results.json ]; then
            echo "[$(date)] cell already completed elsewhere -> no-op"; exit 0
        fi
        echo "[$(date)] TRAIN ctx={ctx} deg={degree} scope={scope} r={rho} lr={lr} on $(hostname)"
        VAR_CONTEXT={ctx} VAR_DEGREE={degree} VAR_SCOPE={scope} \\
            RHO={rho} LR={lr} DIFFICULTY={difficulty} SEED={seed} \\
            OUTPUT_DIR={tdir} DEVICES=1 \\
            bash scripts/train/sudoku/{STEM}.sh
        echo "[$(date)] EVAL"
        VAR_CONTEXT={ctx} VAR_DEGREE={degree} VAR_SCOPE={scope} \\
            RHO={rho} DIFFICULTY={difficulty} SEED={seed} \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt \\
            OUTPUT_DIR={tdir}/eval DEVICES=1 \\
            bash scripts/sample/sudoku/{STEM}.sh
        # checkpoints are kept: the learned schedule (noise.gamma_net.*) is
        # read post-hoc to plot the decoding order.
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--rhos', nargs='+', default=RHOS)
    ap.add_argument('--contexts', nargs='+', default=CONTEXTS,
                    choices=['prompt', 'dlm-state'])
    ap.add_argument('--degrees', nargs='+', default=DEGREES)
    ap.add_argument('--scopes', nargs='+', default=SCOPES, choices=SCOPES)
    ap.add_argument('--lrs', nargs='+', default=LRS)
    ap.add_argument('--difficulties', nargs='+', default=['hard'],
                    choices=['easy', 'medium', 'hard'])
    ap.add_argument('--seeds', nargs='+', default=SEEDS)
    args = ap.parse_args()

    exp = f'{REPO}/experiments/vmulan_sudoku'
    logs = f'{exp}/logs'
    out = f'{REPO}/outputs/vmulan_sudoku'
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)

    cells = list(itertools.product(args.contexts, args.degrees, args.scopes,
                                   args.rhos, args.lrs, args.difficulties,
                                   args.seeds))
    print(f'vmulan_sudoku: {len(cells)} cells '
          f'({len(args.contexts)} ctx x {len(args.degrees)} deg x '
          f'{len(args.scopes)} scope x {len(args.rhos)} R x {len(args.lrs)} lr x '
          f'{len(args.difficulties)} difficulty x {len(args.seeds)} seed)')
    if args.dry_run:
        for cell in cells:
            print(f'  {tag_of(*cell)}')
        cell = cells[0]
        print('\n--- example body (first cell) ---\n'
              + job_body(*cell[:5], f'{out}/{tag_of(*cell)}', *cell[5:]))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for cell in cells:
        ctx, degree, scope, rho, lr, diff, seed = cell
        tag = tag_of(*cell)
        if os.path.exists(f'{out}/{tag}/eval/results.json') or tag in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=tag, output=f'{logs}/{tag}_%j.log',
                      partition='thickstun,desa', exclude='desa-compute-01',
                      gres='gpu:1', ntasks=1, cpus_per_task=2, mem='16G',
                      time='08:00:00')
        jid = slurm.sbatch(job_body(ctx, degree, scope, rho, lr,
                                    f'{out}/{tag}', diff, seed),
                           sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
