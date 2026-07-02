#!/usr/bin/env python
"""hflm_curv_sudoku — H-FLM Gaussian-curvature x Sudoku-difficulty grid.

Grid (12 cells):
  gaussian_curvature : {-0.25, -0.5, -1.0, -2.0}   (4; -1.0 = unit hyperboloid baseline)
  difficulty         : {easy, medium, hard}         (3)
Recipe: the exact scripts/train/sudoku/hflm.sh defaults — tiny-hyperbolic-dit,
algo=hflm, prior_cov=0.25, rho_max=12, noise=log-linear, 20k steps, batch 256,
1 GPU. ONLY algo.gaussian_curvature (GAUSS_CURV) and data.difficulty vary.
Eval: scripts/sample/sudoku/hflm.sh defaults (sudoku_eval, 180 steps, exact
velocity, greedy, top_k_velocity=1) with the SAME GAUSS_CURV as training.
K = -4 is excluded: the float64 Lorentz bound is rho/R <= 20 (R = 1/sqrt(-K));
rho_max=12 at K=-4 gives ~24 -> raises.

ORCHESTRATION ONLY — each cell calls the single-run shared scripts:
  scripts/train/sudoku/hflm.sh   (DIFFICULTY / GAUSS_CURV env knobs)
  scripts/sample/sudoku/hflm.sh  (DIFFICULTY / GAUSS_CURV / CKPT_PATH env knobs)
Idempotent + resumable (skips done/queued cells; resubmits auto-resume from last.ckpt).

Usage:  python sweep.py [--dry-run]
"""
import argparse
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/thickstun/sychou/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/hflm_curv_sudoku'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/hflm_curv_sudoku'

CURVATURES = ['-0.25', '-0.5', '-1.0', '-2.0']   # 4
DIFFICULTIES = ['easy', 'medium', 'hard']        # 3


def tag_of(diff, k):
    return f'd-{diff}_k{k}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(diff, k):
    tag = tag_of(diff, k)
    tdir = f'{OUT}/{tag}'
    edir = f'{tdir}/eval'
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        DIFFICULTY={diff} GAUSS_CURV={k} OUTPUT_DIR={tdir} DEVICES=1 \\
            bash scripts/train/sudoku/hflm.sh
        echo "[$(date)] EVAL {tag}"
        DIFFICULTY={diff} GAUSS_CURV={k} CKPT_PATH={tdir}/checkpoints/last.ckpt \\
            OUTPUT_DIR={edir} DEVICES=1 \\
            bash scripts/sample/sudoku/hflm.sh
        echo "[$(date)] DONE {tag}"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    cells = list(itertools.product(DIFFICULTIES, CURVATURES))
    print(f'hflm_curv_sudoku: {len(cells)} cells '
          f'({len(DIFFICULTIES)} difficulty x {len(CURVATURES)} curvature)')
    if args.dry_run:
        for diff, k in cells:
            print('  hcurv_' + tag_of(diff, k))
        print('\n--- example body ---\n' + job_body(*cells[0]))
        return
    active = active_jobnames()
    n_sub = n_skip = 0
    for diff, k in cells:
        tag = tag_of(diff, k)
        jobname = f'hcurv_{tag}'
        if os.path.exists(f'{OUT}/{tag}/eval/results.json') or jobname in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=jobname, partition='thickstun,desa', gres='gpu:1',
                      ntasks=1, cpus_per_task=8, mem='64G', time='06:00:00',
                      exclude='desa-compute-01', output=f'{LOGS}/{tag}_%j.log')
        jid = slurm.sbatch(job_body(diff, k))
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
