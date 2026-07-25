#!/usr/bin/env python
"""hflm_rescale_tinystories_256 — rescaled H-FLM init x prior_cov sweep (setup.md).

Branch `var_scale` (HFLM._rho_clamp divides rho by √hidden_size). The earlier
std-sweep (std 0.04→1.0 at pc 1.0) collapsed at every std because the rescaled
NOISE radius = 12·tanh(√pc/12) depends only on prior_cov, and pc 1.0 pins it at
~1.0 (RESULTS.md). This sweep raises prior_cov to restore the noise radius while
keeping the small embedding inits — the lever the std-sweep didn't touch.

Grid (15 cells; setup.md):
  init       : random (std 0.02) + custom std {0.01, 0.04}     (3)
  prior_cov  : {2.0, 4.0, 8.0, 10.0, 16.0}                     (5)
Fixed: small-hyperbolic-dit 768/12/12, K=-1.0, rho_max 12, seq 256, self-cond off,
noise log-linear, 30k steps, global batch 512 (1 GPU x PER_GPU_BS 32, accum 16),
bf16, EMA 0.9999, AdamW lr 3e-4.  Eval: exact velocity, top_k_velocity 1, 180
steps, greedy last.

Rescaled geometry (clean_gamma ~ std, noise_gamma = tanh(12·tanh(√pc/12)/2)):
  noise_gamma: pc2 0.61, pc4 0.76, pc8 0.88, pc10 0.91, pc16 0.96  (clean std0.04 = 0.02)

RUNS FROM THIS CHECKOUT (branch var_scale) — REPO is s-flm-dev/s-flm, NOT the shared
/share/.../s-flm tree.

ORCHESTRATION ONLY — calls the single-run shared scripts:
  scripts/train/tinystories/hlfm.sh   (INIT/INIT_STD/PRIOR_COV/RHO_MAX/GAUSS_CURV/LR)
  scripts/sample/tinystories/hflm.sh  (PRIOR_COV/RHO_MAX/GAUSS_CURV/TOPK_VELOCITY)
Idempotent + resumable: skip if eval/ppl.json exists or job queued; auto-resume.

Usage:  python experiments/hflm_rescale_tinystories_256/sweep.py [--dry-run]
"""
import argparse
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/thickstun/sychou/workspace/research/s-flm-dev/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/hflm_rescale_tinystories_256'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/hflm_rescale_tinystories_256'

# (init_tag, INIT, INIT_STD) — random == N(0,4e-4) == std 0.02 (setup.md)
INITS = [('random', 'random', None),
         ('c0.01', 'custom', '0.01'),
         ('c0.04', 'custom', '0.04')]
PRIOR_COVS = ['2.0', '4.0', '8.0', '10.0', '16.0']
FIXED = dict(gauss_curv='-1.0', rho_max='12', lr='3e-4')
SEQ_LEN = 256
PER_GPU_BS = 32          # global 512 -> accum 16
MAX_STEPS = 30000
CKPT_EVERY = 5000


def tag_of(init_tag, pc):
    return f'{init_tag}_pc{pc}_K-1.0'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(init_tag, init, std, pc):
    tag = tag_of(init_tag, pc)
    tdir = f'{OUT}/{tag}'
    edir = f'{tdir}/eval'
    f = FIXED
    init_env = (f'INIT={init}' if init != 'custom'
                else f'INIT=custom INIT_STD={std}')
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        if [ -f {edir}/ppl.json ]; then
            echo "[$(date)] cell already done -> no-op"; exit 0
        fi
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        {init_env} PRIOR_COV={pc} RHO_MAX={f['rho_max']} GAUSS_CURV={f['gauss_curv']} \\
            LR={f['lr']} OUTPUT_DIR={tdir} RUN_NAME=hflm_rescale_{tag} \\
            WANDB_GROUP=hflm_rescale_pc DEVICES=1 PER_GPU_BS={PER_GPU_BS} \\
            SEQ_LEN={SEQ_LEN} MAX_STEPS={MAX_STEPS} CKPT_EVERY={CKPT_EVERY} \\
            bash scripts/train/tinystories/hlfm.sh
        echo "[$(date)] EVAL {tag}"
        PRIOR_COV={pc} RHO_MAX={f['rho_max']} GAUSS_CURV={f['gauss_curv']} \\
            TOPK_VELOCITY=1 CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} \\
            DEVICES=1 SEQ_LEN={SEQ_LEN} \\
            bash scripts/sample/tinystories/hflm.sh
        echo "[$(date)] DONE {tag}"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    cells = list(itertools.product(INITS, PRIOR_COVS))
    if args.dry_run:
        print(f'hflm_rescale_tinystories_256: {len(cells)} cells '
              f'({len(INITS)} init x {len(PRIOR_COVS)} prior_cov)')
        for (itag, init, std), pc in cells:
            print('  hfresc_' + tag_of(itag, pc))
        (itag, init, std), pc = cells[0]
        print('\n--- example job body ---\n' + job_body(itag, init, std, pc))
        return
    os.makedirs(LOGS, exist_ok=True)
    active = active_jobnames()
    n_sub = n_skip = 0
    for (itag, init, std), pc in cells:
        tag = tag_of(itag, pc)
        jobname = f'hfresc_{tag}'
        if os.path.exists(f'{OUT}/{tag}/eval/ppl.json') or jobname in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=jobname, partition='thickstun,desa', gres='gpu:1',
                      ntasks=1, cpus_per_task=8, mem='32G', time='2-00:00:00',
                      exclude='desa-compute-01', output=f'{LOGS}/{tag}_%j.log')
        jid = slurm.sbatch(job_body(itag, init, std, pc),
                           sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
