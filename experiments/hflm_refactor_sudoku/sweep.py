#!/usr/bin/env python
"""hflm_refactor_sudoku — validate the VFM expected-velocity sampler refactor.

9 fixed cells (see EXPERIMENT.md): difficulty x K in {-0.3, -0.5, -1.0},
init=custom std 0.01, per-cell best LR from hflm_curv_init_lr_sudoku, seed 1,
20k steps. Each job trains once, then evals the SAME checkpoint twice:
  eval_topk1/   TOPK_VELOCITY=1   (== the old argmax-endpoint sampler)
  eval_topkall/ TOPK_VELOCITY=-1  (the corrected expected-velocity sampler)

ORCHESTRATION ONLY — calls scripts/train/sudoku/hflm.sh and
scripts/sample/sudoku/hflm.sh. Idempotent: skips a cell whose two results.json
exist or whose job name is queued; resubmits auto-resume from last.ckpt.

Usage:  python experiments/hflm_refactor_sudoku/sweep.py [--dry-run]
"""
import argparse
import getpass
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/thickstun/sychou/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'

# init tag -> (INIT, INIT_STD) knobs for scripts/train/sudoku/hflm.sh
INITS = {
    'c0.01': ('custom', '0.01'),
    'c0.02': ('custom', '0.02'),
    'c0.04': ('custom', '0.04'),
    'random': ('random', 'null'),
}
# per-difficulty best LR from the hflm_curv_init_lr_sudoku sweep (easy unswept
# -> hard's 3e-4)
BEST_LR = {'easy': '3e-4', 'medium': '5e-4', 'hard': '3e-4'}


def build_cells():
    """(difficulty, K, init_tag, lr), ordered by priority (submitted FIFO).

    1) curvature x LR core at init c0.01 (medium/hard before easy):
       K {-0.3,-0.5,-0.7,-1.0} x LR {3e-4,5e-4,1e-3}          36 cells
       (the 9 initial-launch cells are a subset -> skipped as queued/done)
    2) K=-1.5 anchor at 3e-4, c0.01 (replicate 'strong curvature hurts')
                                                               3 cells
    3) init axis at per-difficulty best LR:
       init {c0.02,c0.04,random} x K {-0.3,-0.5,-1.0}         27 cells
    """
    cells = []
    for diff in ['medium', 'hard', 'easy']:
        for k in ['-0.3', '-0.5', '-0.7', '-1.0']:
            for lr in ['3e-4', '5e-4', '1e-3']:
                cells.append((diff, k, 'c0.01', lr))
    for diff in ['medium', 'hard', 'easy']:
        cells.append((diff, '-1.5', 'c0.01', '3e-4'))
    for diff in ['medium', 'hard', 'easy']:
        for itag in ['c0.02', 'c0.04', 'random']:
            for k in ['-0.3', '-0.5', '-1.0']:
                cells.append((diff, k, itag, BEST_LR[diff]))
    return cells


def tag_of(diff, k, itag, lr):
    return f'd-{diff}_k{k}_i-{itag}_lr{lr}_rs1'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(diff, k, init, istd, lr, tdir):
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        if [ -f {tdir}/eval_topk1/results.json ] && [ -f {tdir}/eval_topkall/results.json ]; then
            echo "[$(date)] cell already completed -> no-op"; exit 0
        fi
        echo "[$(date)] TRAIN on $(hostname)"
        DIFFICULTY={diff} GAUSS_CURV={k} INIT={init} INIT_STD={istd} LR={lr} \\
            SEED=1 OUTPUT_DIR={tdir} DEVICES=1 \\
            bash scripts/train/sudoku/hflm.sh
        echo "[$(date)] EVAL top_k_velocity=1 (old argmax-equivalent)"
        DIFFICULTY={diff} GAUSS_CURV={k} INIT={init} INIT_STD={istd} \\
            SEED=1 CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval_topk1 \\
            DEVICES=1 VELOCITY=exact TOPK_VELOCITY=1 \\
            bash scripts/sample/sudoku/hflm.sh
        echo "[$(date)] EVAL top_k_velocity=-1 (expected velocity)"
        DIFFICULTY={diff} GAUSS_CURV={k} INIT={init} INIT_STD={istd} \\
            SEED=1 CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval_topkall \\
            DEVICES=1 VELOCITY=exact TOPK_VELOCITY=-1 \\
            bash scripts/sample/sudoku/hflm.sh
        # checkpoints are transient bulk (~1.8G/cell): once both eval deliverables
        # exist, drop them to keep /share within quota
        if [ -f {tdir}/eval_topk1/results.json ] && [ -f {tdir}/eval_topkall/results.json ]; then
            rm -rf {tdir}/checkpoints && echo "[$(date)] checkpoints cleaned"
        fi
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    exp = f'{REPO}/experiments/hflm_refactor_sudoku'
    logs = f'{exp}/logs'
    out = f'{REPO}/outputs/hflm_refactor_sudoku'
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)

    cells = build_cells()
    print(f'hflm_refactor_sudoku: {len(cells)} cells')
    if args.dry_run:
        for diff, k, itag, lr in cells:
            print(f'  hrs_{tag_of(diff, k, itag, lr)}')
        diff, k, itag, lr = cells[0]
        init, istd = INITS[itag]
        print('\n--- example body (first cell) ---\n'
              + job_body(diff, k, init, istd, lr,
                         f'{out}/{tag_of(diff, k, itag, lr)}'))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for diff, k, itag, lr in cells:
        tag = tag_of(diff, k, itag, lr)
        jobname = f'hrs_{tag}'
        tdir = f'{out}/{tag}'
        init, istd = INITS[itag]
        if (os.path.exists(f'{tdir}/eval_topk1/results.json')
                and os.path.exists(f'{tdir}/eval_topkall/results.json')) \
                or jobname in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=jobname, output=f'{logs}/{tag}_%j.log',
                      partition='thickstun,desa', exclude='desa-compute-01',
                      gres='gpu:1', ntasks=1, cpus_per_task=4, mem='16G',
                      time='06:00:00')
        jid = slurm.sbatch(job_body(diff, k, init, istd, lr, tdir))
        print(f'  submitted {jobname}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
