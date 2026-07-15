#!/usr/bin/env python
"""sweep.py — SFLM/EFLM/HFLM x self-conditioning {on,off} on sudoku {medium,hard}
(slides/jul02_2026 "SFLM, EFLM, HFLM + Self Cond Exp: Setting Up").

Grid (108 cells): algo {sfm_trunc_ada, eflm, hflm} x difficulty {medium, hard}
x lr {3e-4, 5e-4, 1e-3} x self-cond {on, off} x seed {1, 2, 3}.
Fixed (from the slide): tiny DiT (512/8/8), 20k steps, batch 256, bf16,
EMA 0.9999, AdamW wd=0 betas=(0.9,0.999) eps=1e-8 clip=1.0, CE loss.
Init: sfm/eflm ngpt (script default); hflm custom std=0.01 with K=-0.5 (the
curvature sweep's best hard config). Eval: sudoku_eval, 180 steps,
velocity=exact, top_k_velocity=-1, greedy last step — same protocol as the
jul09 baseline + curvature sweeps. (NOTE: on this branch the HFLM sampler's
'exact' is the argmax endpoint regardless of top_k_velocity — identical to the
code path that produced the reused curvature numbers, so the hflm sc-on/off
arms are eval-consistent.)

36/108 cells REUSE completed runs with the exact same train recipe + eval
protocol, materialized as symlinks under outputs/selfcond_sudoku (created
here):
  - sc-on hard lr3e-4 sfm_trunc_ada/eflm (6): this project's jul12 runs
    sc_d-hard_a-{algo}_rs{seed} (lr3e-4 was the script default they used).
  - sc-off lr3e-4 sfm_trunc_ada/eflm (12): jul09 baseline re-run
    ../hflm_curv_init_lr_sudoku/bl_d-{diff}_a-{algo}_rs{seed}.
  - sc-off hflm all lrs (18): curvature sweep
    ../hflm_curv_init_lr_sudoku/d-{diff}_k-0.5_i-c0.01_lr{lr}_rs{seed}.

One SLURM job per remaining cell (72): train then sudoku_eval, both with the
cell's SELF_COND so the eval model rebuilds the checkpoint's arch. Idempotent
+ resumable: skips a cell whose eval/results.json exists (incl. via symlink)
or whose job name is already queued; resubmitting the same OUTPUT_DIR
auto-resumes from checkpoints/last.ckpt. No nice here (Agent.md):
prioritization is handled outside the sweep.

Usage: python experiments/selfcond_sudoku/sweep.py [--dry-run]
         [--algos ...] [--difficulties ...] [--lrs ...] [--sc on off]
         [--seeds 1 2 3]
"""
import argparse
import getpass
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/thickstun/sychou/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
OUT = f'{REPO}/outputs/selfcond_sudoku'
LOGS = f'{REPO}/experiments/selfcond_sudoku/logs'
CURV = '../hflm_curv_init_lr_sudoku'  # reuse-symlink targets, relative to OUT

# algo tag -> (script stem under scripts/{train,sample}/sudoku/,
#              extra env for train, extra env for eval)
ALGOS = {
    'sfm_trunc_ada': ('sfm_truncated_adaptive', '', ''),
    'eflm': ('eflm', '', ''),
    'hflm': ('hflm',
             'GAUSS_CURV=-0.5 INIT=custom INIT_STD=0.01 ',
             'GAUSS_CURV=-0.5 INIT=custom INIT_STD=0.01 TOPK_VELOCITY=-1 '),
}
DIFFICULTIES = ['medium', 'hard']
LRS = ['3e-4', '5e-4', '1e-3']
SCS = ['on', 'off']
SEEDS = ['1', '2', '3']

# (algo, difficulty, lr, sc) -> completed-run dir, relative to OUT; {s} = seed
REUSE = {}
for _a in ('sfm_trunc_ada', 'eflm'):
    REUSE[(_a, 'hard', '3e-4', 'on')] = f'sc_d-hard_a-{_a}_rs{{s}}'
    for _d in DIFFICULTIES:
        REUSE[(_a, _d, '3e-4', 'off')] = f'{CURV}/bl_d-{_d}_a-{_a}_rs{{s}}'
for _d in DIFFICULTIES:
    for _lr in LRS:
        REUSE[('hflm', _d, _lr, 'off')] = \
            f'{CURV}/d-{_d}_k-0.5_i-c0.01_lr{_lr}_rs{{s}}'

SLURM_KW = dict(partition='thickstun,desa', exclude='desa-compute-01',
                gres='gpu:1', ntasks=1, cpus_per_task=2, mem='16G',
                time='06:00:00')


def tag_of(algo, diff, lr, sc, seed):
    return f'sc_d-{diff}_a-{algo}_lr{lr}_sc-{sc}_rs{seed}'


def link_reuse(cell):
    """Symlink a reusable completed run to this cell's dir; True if linked."""
    algo, diff, lr, sc, seed = cell
    target = REUSE.get((algo, diff, lr, sc), '')
    if not target:
        return False
    target = target.format(s=seed)
    link = f'{OUT}/{tag_of(*cell)}'
    if not os.path.exists(f'{OUT}/{target}/eval/results.json'):
        print(f'  WARN reuse target missing results: {target} -> will run')
        return False
    if not os.path.lexists(link):
        os.symlink(target, link)
    return True


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(cell):
    algo, diff, lr, sc, seed = cell
    stem, tenv, eenv = ALGOS[algo]
    sc_flag = 'true' if sc == 'on' else 'false'
    tdir = f'{OUT}/{tag_of(*cell)}'
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
        echo "[$(date)] TRAIN {algo} (SELF_COND={sc_flag}) on $(hostname)"
        {tenv}SELF_COND={sc_flag} DIFFICULTY={diff} LR={lr} SEED={seed} \\
            OUTPUT_DIR={tdir} DEVICES=1 \\
            bash scripts/train/sudoku/{stem}.sh
        echo "[$(date)] EVAL"
        {eenv}SELF_COND={sc_flag} DIFFICULTY={diff} \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt \\
            OUTPUT_DIR={tdir}/eval DEVICES=1 \\
            bash scripts/sample/sudoku/{stem}.sh
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--algos', nargs='+', default=list(ALGOS), choices=list(ALGOS))
    ap.add_argument('--difficulties', nargs='+', default=DIFFICULTIES,
                    choices=DIFFICULTIES)
    ap.add_argument('--lrs', nargs='+', default=LRS)
    ap.add_argument('--sc', nargs='+', default=SCS, choices=SCS)
    ap.add_argument('--seeds', nargs='+', default=SEEDS)
    args = ap.parse_args()

    cells = list(itertools.product(args.algos, args.difficulties, args.lrs,
                                   args.sc, args.seeds))
    print(f'selfcond_sudoku full grid: {len(cells)} cells '
          f'({len(args.algos)} algo x {len(args.difficulties)} diff x '
          f'{len(args.lrs)} lr x {len(args.sc)} sc x {len(args.seeds)} seed)')
    if args.dry_run:
        for cell in cells:
            kind = 'reuse' if cell[:4] in REUSE else 'run'
            print(f'  [{kind}] {tag_of(*cell)}')
        print('\n--- example body (first cell) ---\n' + job_body(cells[0]))
        return

    os.makedirs(LOGS, exist_ok=True)
    active = active_jobnames()
    n_sub = n_skip = n_link = 0
    for cell in cells:
        tag = tag_of(*cell)
        if link_reuse(cell):
            n_link += 1
            continue
        if os.path.exists(f'{OUT}/{tag}/eval/results.json') or tag in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=tag, output=f'{LOGS}/{tag}_%j.log', **SLURM_KW)
        jid = slurm.sbatch(job_body(cell), sbatch_cmd='sbatch --requeue',
                           verbose=False)
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, reused {n_link}, skipped {n_skip}')


if __name__ == '__main__':
    main()
