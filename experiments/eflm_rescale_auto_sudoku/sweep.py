#!/usr/bin/env python
"""sweep.py — autonomous-clock EFLM (+ VDM SNR-weighted CE) on sudoku hard.

Grid: clock x arm x R x tau_max x loss weight x LR x seed (see EXPERIMENT.md).

  clock  : auto = noise=autonomous (1 - alpha_t = exp(-tau), tau = tau_max(1-t))
           ll   = noise=log-linear (the eflm_rescale_sudoku baseline clock)
  arm    : naive = fixed schedule, ada = + adaptive noise schedule
  weight : ce  = plain CE, snr = CE weighted by -SNR'(t)/2 (VDM Eq. 16)

  auto/naive -> scripts/{train,sample}/sudoku/eflm_rescale_auto.sh
  auto/ada   -> scripts/{train,sample}/sudoku/eflm_rescale_auto_adaptive.sh
  ll/naive   -> scripts/{train,sample}/sudoku/eflm_rescale.sh
  ll/ada     -> scripts/{train,sample}/sudoku/eflm_rescale_adaptive.sh

Usage:  python experiments/eflm_rescale_auto_sudoku/sweep.py [--dry-run]
            [--clocks auto ll] [--arms naive ada] [--weights ce snr]
            [--rhos 5.0 8 16] [--taus 3.0] [--lrs ...] [--seeds ...]
        # tau_max pilot (naive clock, one LR/seed):
            --taus 0.5 1.0 2.0 4.0 7.0 --arms naive --lrs 1e-3 --seeds 1
Idempotent: skips a cell whose eval/results.json exists or whose job name is
already in squeue; resubmitting auto-resumes from last.ckpt. Checkpoints are
KEPT for post-hoc loss-geometry analysis.
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

RHOS = ['5.0', '8', '16']
TAUS = ['0.5']   # main grid: pilot winner (flow ends just past the decode point)
LRS = ['3e-4', '5e-4', '1e-3']
SEEDS = ['1', '2', '3']
WEIGHTS = {'ce': 'false', 'snr': 'true'}  # algo.snr_weighted_ce
# (clock, arm) -> train/sample script stem (same stem for both passes)
STEMS = {
    ('auto', 'naive'): 'eflm_rescale_auto',
    ('auto', 'ada'): 'eflm_rescale_auto_adaptive',
    ('ll', 'naive'): 'eflm_rescale',
    ('ll', 'ada'): 'eflm_rescale_adaptive',
}


def tag_of(clock, arm, rho, tau, weight, lr, difficulty, seed):
    # tau_max only exists on the autonomous clock
    tau = tau if clock == 'auto' else 'na'
    return (f'eflmra_{clock}-{arm}_r-{rho}_tau-{tau}_w-{weight}'
            f'_lr-{lr}_d-{difficulty}_rs{seed}')


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(clock, arm, rho, tau, weight, lr, tdir, difficulty, seed):
    stem = STEMS[(clock, arm)]
    knobs = (f'RHO={rho} TAU_MAX={tau} SNR_CE={WEIGHTS[weight]} '
             f'LR={lr} DIFFICULTY={difficulty} SEED={seed}')
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
        echo "[$(date)] TRAIN {clock}/{arm} r={rho} tau={tau} w={weight} lr={lr} on $(hostname)"
        {knobs} OUTPUT_DIR={tdir} DEVICES=1 \\
            bash scripts/train/sudoku/{stem}.sh
        echo "[$(date)] EVAL"
        {knobs} CKPT_PATH={tdir}/checkpoints/last.ckpt \\
            OUTPUT_DIR={tdir}/eval DEVICES=1 \\
            bash scripts/sample/sudoku/{stem}.sh
        # checkpoints are kept for the post-hoc loss-geometry analysis.
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--clocks', nargs='+', default=['auto'], choices=['auto', 'll'])
    ap.add_argument('--arms', nargs='+', default=['naive', 'ada'],
                    choices=['naive', 'ada'])
    ap.add_argument('--weights', nargs='+', default=list(WEIGHTS),
                    choices=list(WEIGHTS))
    ap.add_argument('--rhos', nargs='+', default=RHOS)
    ap.add_argument('--taus', nargs='+', default=TAUS)
    ap.add_argument('--lrs', nargs='+', default=LRS)
    ap.add_argument('--difficulties', nargs='+', default=['hard'],
                    choices=['easy', 'medium', 'hard'])
    ap.add_argument('--seeds', nargs='+', default=SEEDS)
    args = ap.parse_args()

    exp = f'{REPO}/experiments/eflm_rescale_auto_sudoku'
    logs = f'{exp}/logs'
    out = f'{REPO}/outputs/eflm_rescale_auto_sudoku'
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)

    cells = list(itertools.product(args.clocks, args.arms, args.rhos, args.taus,
                                   args.weights, args.lrs, args.difficulties,
                                   args.seeds))
    # the log-linear clock ignores tau_max: keep one cell per (arm, R, ...)
    cells = list(dict.fromkeys(
        (c, a, r, t if c == 'auto' else TAUS[0], w, lr, d, s)
        for c, a, r, t, w, lr, d, s in cells))
    print(f'eflm_rescale_auto_sudoku: {len(cells)} cells '
          f'({len(args.clocks)} clock x {len(args.arms)} arm x '
          f'{len(args.rhos)} R x {len(args.taus)} tau x {len(args.weights)} w x '
          f'{len(args.lrs)} lr x {len(args.difficulties)} difficulty x '
          f'{len(args.seeds)} seed)')
    if args.dry_run:
        for cell in cells:
            print(f'  {tag_of(*cell)}')
        print('\n--- example body (first cell) ---\n'
              + job_body(*cells[0][:5], cells[0][5],
                         f'{out}/{tag_of(*cells[0])}', *cells[0][6:]))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for cell in cells:
        tag = tag_of(*cell)
        if os.path.exists(f'{out}/{tag}/eval/results.json') or tag in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=tag, output=f'{logs}/{tag}_%j.log',
                      partition='thickstun,desa', exclude='desa-compute-01',
                      gres='gpu:1', ntasks=1, cpus_per_task=2, mem='16G',
                      time='06:00:00')
        jid = slurm.sbatch(job_body(*cell[:5], cell[5], f'{out}/{tag}',
                                    *cell[6:]),
                           sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
