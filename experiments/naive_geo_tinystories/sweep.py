#!/usr/bin/env python
"""naive_geo_tinystories (slides jun25_2026) — Naive Geometry baseline.

3 runs (S-FLM, E-FLM, H-FLM), identical small DiT (768/12/12), init=ngpt,
noise=log-linear (NO tricks), 30k steps, batch 512, seq 1024, bf16, EMA 0.9999,
AdamW (config defaults). Eval = exact-velocity, top_k_v=1, 180 steps, greedy last.

ORCHESTRATION ONLY — calls the single-run shared scripts:
  scripts/train/tinystories/{sfm,eflm,hlfm}.sh
  scripts/sample/tinystories/{sfm,eflm,hflm}.sh
Idempotent + resumable (skips done/queued cells; resubmits auto-resume).

Usage:  python sweep.py [--dry-run]
"""
import argparse
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/thickstun/sychou/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/naive_geo_tinystories'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/naive_geo_tinystories'
DEVICES = 4
PER_GPU_BS = 8

# tag -> (train_script, sample_script, train_env)   [eval needs no extra env: prior_cov/rho_max default]
CELLS = {
    'sfm':  ('sfm.sh',  'sfm.sh',  ''),
    'eflm': ('eflm.sh', 'eflm.sh', ''),
    'hflm': ('hlfm.sh', 'hflm.sh', 'INIT=ngpt'),   # slide: naive_geo H-FLM uses init=ngpt
    # Fair geometry swap: H-FLM needs init matched to the prior radius. ngpt (||e||~1)
    # collapses the radial coord; hyperbolic (std 0.3, ||e||~8.3) matches E[rho_prior]~9.8
    # and is the config that beats S-FLM on Sudoku (s-flm-dev/experiments/hflm).
    'hflm_hyperbolic': ('hlfm.sh', 'hflm.sh', 'INIT=hyperbolic'),
}


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(tag, train_script, sample_script, train_env):
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
        OUTPUT_DIR={tdir} RUN_NAME=naive_geo_{tag} DEVICES={DEVICES} PER_GPU_BS={PER_GPU_BS} {train_env} \\
            bash scripts/train/tinystories/{train_script}
        echo "[$(date)] EVAL {tag}"
        CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 \\
            bash scripts/sample/tinystories/{sample_script}
        echo "[$(date)] DONE {tag}"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    if args.dry_run:
        for tag, (tr, sa, env) in CELLS.items():
            print(f'  ngeo_{tag}: train={tr} sample={sa} env="{env}"')
        ex = next(iter(CELLS))
        print('\n--- example body ---\n' + job_body(ex, *CELLS[ex]))
        return
    active = active_jobnames()
    n_sub = n_skip = 0
    for tag, (tr, sa, env) in CELLS.items():
        jobname = f'ngeo_{tag}'
        if os.path.exists(f'{OUT}/{tag}/eval/ppl.json') or jobname in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=jobname, partition='thickstun,desa', gres=f'gpu:{DEVICES}',
                      ntasks=1, cpus_per_task=16, mem='128G', time='10-00:00:00',
                      exclude='desa-compute-01', output=f'{LOGS}/{tag}_%j.log')
        jid = slurm.sbatch(job_body(tag, tr, sa, env))
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
