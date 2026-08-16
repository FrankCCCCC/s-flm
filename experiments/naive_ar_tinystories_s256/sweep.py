#!/usr/bin/env python
"""naive_ar_tinystories_s256 (slides jun25_2026) — Naive baselines, SEQ LEN 256.

Reference points for the geometry flows (S/E/H-FLM). Four methods on an identical
small DiT (width 768 / depth 12 / heads 12), 30k steps, batch 512, seq 256, bf16,
EMA 0.9999, AdamW (wd 0, betas (0.9,0.999), eps 1e-8, grad-clip 1.0). Grid per
`setup.md`: methods x LR {3e-4,1e-3,5e-3} x seed {1,2,3} = 36 cells.

  ar    causal autoregressive        — valid PPL is a *true* AR PPL
  mdlm  masked (absorbing) diffusion — valid PPL is a denoising-ELBO bound
  duo   uniform-state diffusion      — valid PPL is a denoising-ELBO bound
  flm   base flow language model     — valid PPL is an unweighted denoising CE

Valid PPL mixes three estimands across these rows; compare cells on GenPPL + entropy.
AR is decoded greedily (`scripts/sample/tinystories/ar.sh` defaults GREEDY=true), so its
64 samples are one string repeated — its GenPPL is a mode decode, not rankable against the
stochastic rows. Re-run that cell with GREEDY=false for the comparable number.

Sampling is matched at 180 steps for mdlm/duo/flm (setup.md); AR ignores sampler.steps.
Only the TRAIN seed is swept — the sample scripts expose no SEED knob, so eval noise is
common across seeds and the error bars isolate training-seed variance (same convention as
experiments/seed_errbar_tinystories_256).

ORCHESTRATION ONLY — calls the single-run shared scripts (SEQ_LEN/LR/SEED env knobs):
  scripts/train/tinystories/{ar,mdlm,duo,flm}.sh   (train)
  scripts/sample/tinystories/{ar,mdlm,duo,flm}.sh  (valid PPL + GenPPL)
Idempotent + resumable: skips cells whose eval/ppl.json exists or that are already
queued; a resubmitted cell auto-resumes from checkpoints/last.ckpt (same OUTPUT_DIR).

NOTE: the two pre-existing run dirs `ar/` and `mdlm/` predate this grid's naming. They are
the lr-3e-4 / seed-1 cells; rename them to `m-ar_lr-3e-4_sd-1` / `m-mdlm_lr-3e-4_sd-1` to
have their checkpoints reused, otherwise those two cells retrain from scratch.

Usage:  python sweep.py [--dry-run]
"""
import argparse
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/naive_ar_tinystories_s256'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/naive_ar_tinystories_s256'
DEVICES = 4
PER_GPU_BS = 32
SEQ_LEN = 256
CKPT_EVERY = 5000
SAVE_TOPK = -1

# Searched axes (setup.md). Script basename is shared by the train + sample script.
METHODS = ['ar', 'mdlm', 'duo', 'flm']
LRS = ['3e-4', '1e-3', '5e-3']
SEEDS = [1, 2, 3]


def cells():
    return [(f'm-{m}_lr-{lr}_sd-{sd}', m, lr, sd)
            for m in METHODS for lr in LRS for sd in SEEDS]


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(tag, method, lr, seed):
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
            LR={lr} SEED={seed} \\
            bash scripts/train/tinystories/{method}.sh
        echo "[$(date)] EVAL {tag}"
        CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 SEQ_LEN={SEQ_LEN} \\
            bash scripts/sample/tinystories/{method}.sh
        echo "[$(date)] DONE {tag}"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    grid = cells()
    print(f'naive_ar_tinystories_s256: {len(grid)} cells '
          f'({len(METHODS)} methods x {len(LRS)} lr x {len(SEEDS)} seeds), '
          f'{DEVICES} GPUs each')
    if args.dry_run:
        for tag, *_ in grid:
            print(f'  nar256_{tag}')
        print('\n--- example body ---\n' + job_body(*grid[0]))
        return
    os.makedirs(LOGS, exist_ok=True)
    active = active_jobnames()
    n_sub = n_skip = 0
    for tag, method, lr, seed in grid:
        jobname = f'nar256_{tag}'
        if os.path.exists(f'{OUT}/{tag}/eval/ppl.json') or jobname in active:
            print(f'skip {tag}: already evaluated or queued')
            n_skip += 1
            continue
        slurm = Slurm(job_name=jobname, partition='thickstun,desa', gres=f'gpu:{DEVICES}',
                      ntasks=1, cpus_per_task=16, mem='64G', time='10-00:00:00',
                      exclude='desa-compute-01', output=f'{LOGS}/{tag}_%j.log')
        jid = slurm.sbatch(job_body(tag, method, lr, seed))
        print(f'submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
