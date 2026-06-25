#!/usr/bin/env python
"""naive_ar_tinystories_s256 (slides jun25_2026) — Naive AR baseline, SEQ LEN 256.

Single run: causal small DiT (768/12/12), 30k steps, batch 512, seq 256, bf16,
EMA 0.9999, AdamW (config defaults already match the slide). Eval = greedy decoding.
Checkpoints every 5k steps, all retained (SAVE_TOPK=-1).

ORCHESTRATION ONLY — calls the single-run shared scripts (SEQ_LEN=256 knob):
  scripts/train/tinystories/ar.sh   (train)
  scripts/sample/tinystories/ar.sh  (valid PPL + GenPPL)
Idempotent + resumable: skips cells whose eval/ppl.json exists or that are already
queued; a resubmitted cell auto-resumes from checkpoints/last.ckpt (same OUTPUT_DIR).

Usage:  python sweep.py [--dry-run]
"""
import argparse
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/thickstun/sychou/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/naive_ar_tinystories_s256'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/naive_ar_tinystories_s256'
DEVICES = 4
PER_GPU_BS = 32
SEQ_LEN = 256
CKPT_EVERY = 5000
SAVE_TOPK = -1


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(tag):
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
        OUTPUT_DIR={tdir} RUN_NAME={tag} DEVICES={DEVICES} PER_GPU_BS={PER_GPU_BS} \\
            SEQ_LEN={SEQ_LEN} CKPT_EVERY={CKPT_EVERY} SAVE_TOPK={SAVE_TOPK} \\
            bash scripts/train/tinystories/ar.sh
        echo "[$(date)] EVAL {tag}"
        CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 SEQ_LEN={SEQ_LEN} \\
            bash scripts/sample/tinystories/ar.sh
        echo "[$(date)] DONE {tag}"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    tag = 'ar'
    jobname = f'nar256_{tag}'
    if args.dry_run:
        print(f'1 cell: {jobname}\n\n--- body ---\n{job_body(tag)}')
        return
    if os.path.exists(f'{OUT}/{tag}/eval/ppl.json'):
        print(f'skip {tag}: already evaluated')
        return
    if jobname in active_jobnames():
        print(f'skip {tag}: already queued')
        return
    slurm = Slurm(job_name=jobname, partition='thickstun,desa', gres=f'gpu:{DEVICES}',
                  ntasks=1, cpus_per_task=16, mem='64G', time='10-00:00:00',
                  exclude='desa-compute-01', output=f'{LOGS}/{tag}_%j.log')
    jid = slurm.sbatch(job_body(tag))
    print(f'submitted {tag}: job {jid}')


if __name__ == '__main__':
    main()
