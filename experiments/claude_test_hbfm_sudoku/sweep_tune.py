#!/usr/bin/env python
"""claude_test_hbfm_sudoku / tune — curvature x proposal-rate probe grid (5k steps).

After RESULTS.md H1 (rate 0.01 covers the decision phase with ~3% of samples; rate 3 reaches
30% boards in 5k steps), tune the two geometry knobs jointly with short runs:
  K          in {-0.25, -0.5, -1.0, -2.0}    (prod_factor_gaussian_curvature = [K, K, K])
  unit rate  in {1, 3, 10}                    (algo.time_exp_rate = unit_rate * |K|, so the
                                               heat-time coverage in UNIT time is comparable)
Fixed: (H^3_K)^3 (embed_dim 9), LR 3e-4, seed 1, noise=log-linear, 5000 steps, batch 256,
eval = sudoku_eval with unit t_max 3 (physical 3/|K|), 180 steps, exact velocity, greedy.
Outputs: outputs/claude_test_hbfm_sudoku/tune_k{K}_ur{rate}_seed-1/{checkpoints,eval/results.json}.

`--confirm`: the 20k-step, 3-seed confirmation of the open (K, unit rate) cells (CONFIRM below;
tags tune20k_k{K}_ur{rate}_seed-{s}). The 5k probes rank curvature unreliably: at 5k K=-1 beats
K=-0.5 (30.4 vs 22.9) while at 20k K=-0.5 wins (61.1 vs 51.4, the `_rate-3` cells of sweep.py,
which are PHYSICAL rate 3 = unit rate 6 at K=-0.5).

ORCHESTRATION ONLY (scripts/train/sudoku/hbfm.sh + scripts/sample/sudoku/hbfm.sh);
idempotent (skips cells with results.json / queued jobs).
`--fixed-rate`: the spec's physical rate 0.01 held fixed and the curvature tuned instead
(K = -0.01/unit_rate, so unit rate 10 / 6 / 3 <-> K = -0.001 / -0.0016667 / -0.0033333;
tests the (rate, K) -> (c*rate, c*K) equivalence at 20k steps, 3 seeds; same tune20k_* tags).

`--clip-control`: the fixed-rate K=-0.0016667 / unit-6 cell with trainer.gradient_clip_val=300 (= clip 1.0 at
the rate-3 loss scale; isolates the loss-scale x clipping asymmetry found by the audit); tags `..._clip300`.

Usage: python sweep_tune.py [--dry-run] [--confirm | --fixed-rate | --clip-control]
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

CURVS = ['-0.25', '-0.5', '-1.0', '-2.0']
UNIT_RATES = ['1', '3', '10']
LR = '3e-4'
UNIT_T_MAX = '3.0'
CONFIRM = [('-0.25', '3'), ('-0.5', '1'), ('-0.5', '3'), ('-0.5', '10'), ('-2.0', '10')]
CONFIRM_SEEDS = ['1', '2', '3']
FIXED_RATE = [('-0.001', '10'), ('-0.0016667', '6'), ('-0.0033333', '3')]  # unit_rate * |K| = 0.01
CLIP_CONTROL = [('-0.0016667', '6')]


def tag_of(k, ur, seed='1', steps=5000, suffix=''):
    return f'{"tune20k" if steps == 20000 else "tune"}_k{k}_ur{ur}_seed-{seed}{suffix}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(k, ur, seed='1', steps=5000, suffix=''):
    tag = tag_of(k, ur, seed, steps, suffix)
    extra = "EXTRA_ARGS='trainer.gradient_clip_val=300'" if suffix == '_clip300' else ''
    tdir = f'{OUT}/{tag}'
    kf = float(k)
    geom = (f"EMBED_DIM=9 PROD_FACTOR_DIM='[3,3,3]' PROD_FACTOR_CURV='[{kf},{kf},{kf}]' "
            f"GAUSS_CURV={kf} UNIT_PROPOSAL_RATE={ur}")
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
        LR={LR} NOISE=log-linear SEED={seed} {geom} {extra} \\
            OUTPUT_DIR={tdir} DEVICES=1 PER_GPU_BS=256 GLOBAL_BATCH=256 \\
            MAX_STEPS={steps} CKPT_EVERY=5000 \\
            bash scripts/train/sudoku/hbfm.sh
        echo "[$(date)] EVAL {tag}"
        NOISE=log-linear SEED={seed} {geom} UNIT_T_MAX={UNIT_T_MAX} GLOBAL_BATCH=256 \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval DEVICES=1 \\
            bash scripts/sample/sudoku/hbfm.sh
        echo "[$(date)] DONE {tag}"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--confirm', action='store_true', help='20k-step, 3-seed confirmation cells')
    ap.add_argument('--fixed-rate', action='store_true', help='physical rate 0.01, curvature tuned (20k, 3 seeds)')
    ap.add_argument('--clip-control', action='store_true', help='fixed-rate K=-0.0016667 cell with gradient_clip_val=300 (20k, 3 seeds)')
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    suffix = '_clip300' if args.clip_control else ''
    if args.confirm or args.fixed_rate or args.clip_control:
        steps = 20000
        grid = CLIP_CONTROL if args.clip_control else FIXED_RATE if args.fixed_rate else CONFIRM
        cells = [(k, ur, seed) for (k, ur), seed in itertools.product(grid, CONFIRM_SEEDS)]
        mode = '--clip-control' if args.clip_control else '--fixed-rate' if args.fixed_rate else '--confirm'
        print(f'tune {mode}: {len(cells)} cells ({len(grid)} (K, unit rate) x {len(CONFIRM_SEEDS)} seeds), 20000 steps')
    else:
        steps = 5000
        cells = [(k, ur, '1') for k, ur in itertools.product(CURVS, UNIT_RATES)]
        print(f'tune: {len(cells)} cells ({len(CURVS)} K x {len(UNIT_RATES)} unit rates), 5000 steps each')
    if args.dry_run:
        for k, ur, seed in cells:
            print('  hbfm_tune_' + tag_of(k, ur, seed, steps, suffix))
        print('\n--- example body ---\n' + job_body(*cells[0], steps, suffix))
        return
    active = active_jobnames()
    n_sub = n_skip = 0
    for k, ur, seed in cells:
        tag = tag_of(k, ur, seed, steps, suffix)
        jobname = f'hbfm_tune_{tag}'
        if os.path.exists(f'{OUT}/{tag}/eval/results.json') or jobname in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=jobname, partition='thickstun,desa', gres='gpu:1',
                      ntasks=1, cpus_per_task=8, mem='64G', time='12:00:00' if steps == 20000 else '03:00:00',
                      exclude='desa-compute-01', output=f'{LOGS}/{tag}_%j.log')
        jid = slurm.sbatch(job_body(k, ur, seed, steps, suffix))
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
