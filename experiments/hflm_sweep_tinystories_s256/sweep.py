#!/usr/bin/env python
"""hflm_sweep_tinystories_s256 (slides jun25_2026) — H-FLM embedding-init x init-noise grid, SEQ LEN 256.

Grid (132 cells):
  init       : ngpt  +  custom std {0.001,0.01,0.02,0.04,0.1,0.3,0.5,0.8,1.0,1.5,2.0}  (12)
  prior_cov  : {0.001,0.01,0.02,0.04,0.1,0.3,0.5,0.8,1.0,1.5,2.0}                        (11)
  rho_max    : 12
Recipe: small-hyperbolic-dit 768/12/12, 30k steps, effective batch 512 (1 GPU x
PER_GPU_BS=8, accum auto = 64), seq 256, bf16, EMA 0.9999, AdamW lr 3e-4, noise=log-linear.
Checkpoints every 5k steps, all retained (SAVE_TOPK=-1).
Eval: exact-velocity, top_k_v=1, 180 steps, greedy last.

ORCHESTRATION ONLY — each cell calls the single-run shared scripts (SEQ_LEN=256 knob):
  scripts/train/tinystories/hlfm.sh   (INIT / INIT_STD / PRIOR_COV / RHO_MAX env knobs)
  scripts/sample/tinystories/hflm.sh  (PRIOR_COV / RHO_MAX env knobs)
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
EXP = f'{REPO}/experiments/hflm_sweep_tinystories_s256'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/hflm_sweep_tinystories_s256'

SEQ_LEN = 256
CKPT_EVERY = 5000
SAVE_TOPK = -1

STDS = ['0.001', '0.01', '0.02', '0.04', '0.1', '0.3', '0.5', '0.8', '1.0', '1.5', '2.0']
INITS = [('ngpt', None)] + [('custom', s) for s in STDS]                                  # 12
PRIOR_COVS = ['0.001', '0.01', '0.02', '0.04', '0.1', '0.3', '0.5', '0.8', '1.0', '1.5', '2.0']  # 11
RHO_MAX = '12'


def tag_of(init, std, pc):
    iname = 'ngpt' if init == 'ngpt' else f'std{std}'
    return f'{iname}_pc{pc}'


def init_env(init, std):
    return 'INIT=ngpt' if init == 'ngpt' else f'INIT=custom INIT_STD={std}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(init, std, pc):
    tag = tag_of(init, std, pc)
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
        {init_env(init, std)} PRIOR_COV={pc} RHO_MAX={RHO_MAX} OUTPUT_DIR={tdir} RUN_NAME=hflm_{tag}_s256 DEVICES=1 PER_GPU_BS=32 \\
            SEQ_LEN={SEQ_LEN} CKPT_EVERY={CKPT_EVERY} SAVE_TOPK={SAVE_TOPK} \\
            bash scripts/train/tinystories/hlfm.sh
        echo "[$(date)] EVAL {tag}"
        PRIOR_COV={pc} RHO_MAX={RHO_MAX} CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 SEQ_LEN={SEQ_LEN} \\
            bash scripts/sample/tinystories/hflm.sh
        echo "[$(date)] DONE {tag}"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    cells = [(i, s, pc) for (i, s), pc in itertools.product(INITS, PRIOR_COVS)]
    print(f'hflm_sweep_tinystories_s256: {len(cells)} cells '
          f'({len(INITS)} init x {len(PRIOR_COVS)} prior_cov)')
    if args.dry_run:
        for i, s, pc in cells:
            print('  hflm256_' + tag_of(i, s, pc))
        print('\n--- example body ---\n' + job_body(*cells[0]))
        return
    active = active_jobnames()
    n_sub = n_skip = 0
    for init, std, pc in cells:
        tag = tag_of(init, std, pc)
        jobname = f'hflm256_{tag}'
        if os.path.exists(f'{OUT}/{tag}/eval/ppl.json') or jobname in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=jobname, partition='thickstun,desa', gres='gpu:1',
                      ntasks=1, cpus_per_task=8, mem='32G', time='10-00:00:00',
                      exclude='desa-compute-01', output=f'{LOGS}/{tag}_%j.log')
        jid = slurm.sbatch(job_body(init, std, pc))
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
