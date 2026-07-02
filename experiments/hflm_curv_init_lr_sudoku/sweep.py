#!/usr/bin/env python
"""hflm_curv_init_lr_sudoku — H-FLM curvature x embedding-init x LR grid on Sudoku (medium).

Grid (168 cells):
  gaussian_curvature : {-0.25, -0.3, -0.5, -0.7, -1.0, -1.5}                (6)
  init               : ngpt, random, custom std {0.01,0.02,0.04,0.06,0.08}  (7)
  lr                 : {1e-4, 3e-4, 5e-4, 1e-3}                             (4)
Fixed: difficulty=medium (easy saturates / hard is noisy per hflm_curv_sudoku),
tiny-hyperbolic-dit (512/8/8), 20k steps, batch 256, seq 180, bf16, EMA 0.9999,
AdamW wd=0 betas=(0.9,0.999) eps=1e-8 clip=1.0, prior_cov=0.25, rho_max=12,
noise=log-linear.  Eval: sudoku_eval, 180 steps, exact velocity, greedy,
top_k_velocity=-1 (avg across vocab; NOTE: differs from hflm_curv_sudoku's top-1).

PRIORITY cells (submitted first, sbatch --nice=0; the rest get --nice=100):
  init=random x lr in {3e-4, 1e-3} x all curvatures  (12 cells).
Re-tier later with:  scontrol update jobid=<id> nice=<n>

Two sites, static split on the curvature axis (6 priority cells each):
  --site unicorn : K in {-0.25, -0.5, -1.0}  (84 cells; partition thickstun,desa)
  --site tc      : K in {-0.3, -0.7, -1.5}   (84 cells; TinkerCliffs a100_normal_q)
Run this script ON the submitting cluster (sc3379@unicorn / shengyenc@tinkercliffs).

ORCHESTRATION ONLY — each cell calls the single-run shared scripts:
  scripts/train/sudoku/hflm.sh   (DIFFICULTY/GAUSS_CURV/INIT/INIT_STD/LR knobs)
  scripts/sample/sudoku/hflm.sh  (same + CKPT_PATH/TOPK_VELOCITY knobs)
Idempotent + resumable (skips done/queued cells; resubmits auto-resume from last.ckpt).

Usage:  python sweep.py --site {unicorn,tc} [--dry-run]
"""
import argparse
import getpass
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

SITES = {
    'unicorn': dict(
        repo='/share/thickstun/sychou/workspace/research/s-flm',
        envbin='/home/sc3379/anaconda3/envs/sfm/bin',
        # widened to all six K on 2026-07-02: TC's a100 queue starved (19/84 done),
        # so unicorn backfills TC's remainder — idempotency skips synced-done cells
        curvatures=['-0.25', '-0.3', '-0.5', '-0.7', '-1.0', '-1.5'],
        slurm=dict(partition='thickstun,desa', exclude='desa-compute-01',
                   gres='gpu:1', ntasks=1, cpus_per_task=8, mem='32G',
                   time='06:00:00'),
        wandb_offline=False,
    ),
    'tc': dict(
        repo='/home/shengyenc/workspace/research/s-flm',
        envbin='/home/shengyenc/anaconda3/envs/sfm/bin',
        curvatures=['-0.3', '-0.7', '-1.5'],
        slurm=dict(partition='a100_normal_q', account='swan_research_dlm',
                   gres='gpu:1', ntasks=1, cpus_per_task=8, mem='64G',
                   time='06:00:00'),
        wandb_offline=True,
    ),
}

# (tag, INIT, INIT_STD) — INIT_STD='null' for non-custom modes
INITS = [('ngpt', 'ngpt', 'null'), ('random', 'random', 'null')] + [
    (f'c{s}', 'custom', s) for s in ['0.01', '0.02', '0.04', '0.06', '0.08']]
LRS = ['1e-4', '3e-4', '5e-4', '1e-3']
PRIORITY_INITS = {'random'}
PRIORITY_LRS = {'3e-4', '1e-3'}


def tag_of(k, init_tag, lr, difficulty='medium'):
    # medium keeps the original unprefixed tags (back-compat with existing outputs)
    prefix = '' if difficulty == 'medium' else f'd-{difficulty}_'
    return f'{prefix}k{k}_i-{init_tag}_lr{lr}'


def is_priority(init_tag, lr):
    return init_tag in PRIORITY_INITS and lr in PRIORITY_LRS


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(site, k, init, init_std, lr, tdir, difficulty='medium'):
    offline = 'export WANDB_MODE=offline\n' if site['wandb_offline'] else ''
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PATH={site['envbin']}:$PATH
        {offline}cd {site['repo']}
        echo "[$(date)] TRAIN on $(hostname)"
        DIFFICULTY={difficulty} GAUSS_CURV={k} INIT={init} INIT_STD={init_std} LR={lr} \\
            OUTPUT_DIR={tdir} DEVICES=1 \\
            bash scripts/train/sudoku/hflm.sh
        echo "[$(date)] EVAL"
        DIFFICULTY={difficulty} GAUSS_CURV={k} INIT={init} INIT_STD={init_std} \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval \\
            DEVICES=1 TOPK_VELOCITY=-1 \\
            bash scripts/sample/sudoku/hflm.sh
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True, choices=SITES)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--difficulty', default='medium', choices=['easy', 'medium', 'hard'])
    ap.add_argument('--priority-only', action='store_true',
                    help='submit only the priority slice (random x {3e-4, 1e-3})')
    args = ap.parse_args()
    site = SITES[args.site]
    exp = f"{site['repo']}/experiments/hflm_curv_init_lr_sudoku"
    logs = f'{exp}/logs'
    out = f"{site['repo']}/outputs/hflm_curv_init_lr_sudoku"
    os.makedirs(logs, exist_ok=True)

    # (k, init_tag, INIT, INIT_STD, lr), priority cells first
    cells = [(k, itag, init, istd, lr)
             for k, (itag, init, istd), lr
             in itertools.product(site['curvatures'], INITS, LRS)]
    if args.priority_only:
        cells = [c for c in cells if is_priority(c[1], c[4])]
    cells.sort(key=lambda c: (not is_priority(c[1], c[4]),))

    print(f'hflm_curv_init_lr_sudoku [{args.site}, {args.difficulty}]: {len(cells)} cells '
          f'({len(site["curvatures"])} K x {len(set(c[1] for c in cells))} init x '
          f'{len(set(c[4] for c in cells))} lr), '
          f'{sum(is_priority(c[1], c[4]) for c in cells)} priority')
    if args.dry_run:
        for k, itag, init, istd, lr in cells:
            nice = 0 if is_priority(itag, lr) else 100
            print(f'  hcil_{tag_of(k, itag, lr, args.difficulty)}  nice={nice}')
        k, itag, init, istd, lr = cells[0]
        print('\n--- example body (first priority cell) ---\n'
              + job_body(site, k, init, istd, lr,
                         f'{out}/{tag_of(k, itag, lr, args.difficulty)}',
                         args.difficulty))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for k, itag, init, istd, lr in cells:
        tag = tag_of(k, itag, lr, args.difficulty)
        jobname = f'hcil_{tag}'
        if os.path.exists(f'{out}/{tag}/eval/results.json') or jobname in active:
            n_skip += 1
            continue
        # --nice on the command line: simple_slurm's '#SBATCH --nice N' directive
        # is rejected by sbatch (--nice takes an optional arg, needs '=' form)
        nice = 0 if is_priority(itag, lr) else 100
        slurm = Slurm(job_name=jobname, output=f'{logs}/{tag}_%j.log',
                      **site['slurm'])
        jid = slurm.sbatch(job_body(site, k, init, istd, lr, f'{out}/{tag}',
                                    args.difficulty),
                           sbatch_cmd=f'sbatch --nice={nice}', verbose=False)
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
