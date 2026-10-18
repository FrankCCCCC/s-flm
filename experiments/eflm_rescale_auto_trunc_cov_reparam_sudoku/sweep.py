#!/usr/bin/env python
"""eflm_rescale_auto_trunc_cov_reparam_sudoku — rescaled E-FLM with the Gaussian (Bayes) readout on hard
Sudoku: noise schedule x R x LR x seed (setup.md; design in EXPERIMENT.md).

  noise  auto    = noise=autonomous,           TAU_MAX   = tau*(R),   CE x |alpha_t'| (noise_sched_cov_weight)
         autoada = noise=autonomous-adaptive,  TAU_MAX   = tau*(R),   plain CE
         llada   = noise=log-linear-adaptive,  ALPHA_MAX = alpha*(R), plain CE
  R      {5.0, 8, 16}  (rho_min = rho_max = R)
  LR     {3e-4, 5e-4, 1e-3}
  seed   {1, 2, 3}
tau*(R) / alpha*(R) = alpha_star_euclidean(V=12, embed_norm=R, auto_clock=True / False) (docs/truncation.md).

Fixed: algo.forward_type=gaussian, hard difficulty, tiny-sphere-dit, 20k steps, global batch 256 as 128 x 2
accumulation (one 256 micro-batch OOMs on the 11 GB 2080 Ti), bf16, EMA 0.9999.
Eval: sudoku_eval, exact velocity, top_k_velocity -1, 180 steps, greedy last step -> eval/results.json.
GPUs: the RTX 2080 Ti of partitions desa/thickstun only (setup: "desa and thickstun 2080ti").

ORCHESTRATION ONLY: one SLURM job per cell runs scripts/train/sudoku/<stem>.sh then scripts/sample/sudoku/<stem>.sh.
Idempotent + resumable: skips a cell whose eval/results.json exists or whose job is queued; resubmitting
resumes from checkpoints/last.ckpt. Cells are submitted seed-major, so all seed-1 cells run first.

Usage:  python experiments/eflm_rescale_auto_trunc_cov_reparam_sudoku/sweep.py [--dry-run]
            [--noises auto autoada llada] [--rhos 5.0 8 16] [--lrs 3e-4 5e-4 1e-3] [--seeds 1 2 3]
then    python experiments/eflm_rescale_auto_trunc_cov_reparam_sudoku/report.py
"""
import argparse
import itertools
import os
import subprocess
import sys
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev1/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
sys.path.insert(0, REPO)
from noise_schedules import alpha_star_euclidean  # noqa: E402

PROJECT = 'eflm_rescale_auto_trunc_cov_reparam_sudoku'
LOGS = f'{REPO}/experiments/{PROJECT}/logs'
OUT = f'{REPO}/outputs/{PROJECT}'
V = 12
# noise -> (train/sample script stem, truncation knob, noise_sched_cov_weight)
NOISES = {
  'auto': ('eflm_rescale_auto', 'TAU_MAX', 'true'),
  'autoada': ('eflm_rescale_auto_adaptive', 'TAU_MAX', 'false'),
  'llada': ('eflm_rescale_truncated_adaptive', 'ALPHA_MAX', 'false'),
}
RHOS = ['5.0', '8', '16']
LRS = ['3e-4', '5e-4', '1e-3']
SEEDS = ['1', '2', '3']
PER_GPU_BS = 128   # x 2 accumulation = global batch 256
MAX_STEPS = 20000
CKPT_EVERY = 5000


def trunc(noise, rho):
  """tau*(R) on the autonomous clock, alpha*(R) for the log-linear one."""
  auto_clock = NOISES[noise][1] == 'TAU_MAX'
  return f'{alpha_star_euclidean(V, embed_norm=float(rho), auto_clock=auto_clock):.4f}'


def tag_of(noise, rho, lr, seed):
  return f'ns-{noise}_r-{rho}_lr-{lr}_s-{seed}'


def active_jobnames():
  try:
    out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                         capture_output=True, text=True).stdout
    return set(out.split())
  except Exception:
    return set()


def job_body(noise, rho, lr, seed):
  stem, knob, cov = NOISES[noise]
  tdir = f'{OUT}/{tag_of(noise, rho, lr, seed)}'
  model = f'RHO={rho} {knob}={trunc(noise, rho)} FORWARD_TYPE=gaussian DIFFICULTY=hard'   # must match train <-> eval
  return textwrap.dedent(f'''\
    set -e
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export SLURM_JOB_NAME=bash
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TORCHDYNAMO_DISABLE=1
    export WANDB_MODE=offline
    export TMPDIR=/tmp/{PROJECT}_$SLURM_JOB_ID MPLCONFIGDIR=/tmp/{PROJECT}_$SLURM_JOB_ID/mpl WANDB_CACHE_DIR=/tmp/{PROJECT}_$SLURM_JOB_ID/wandb
    mkdir -p $TMPDIR $MPLCONFIGDIR $WANDB_CACHE_DIR; trap 'rm -rf "$TMPDIR"' EXIT
    export PATH={ENVBIN}:$PATH
    cd {REPO}
    if [ -f {tdir}/eval/results.json ]; then
        echo "[$(date)] cell already completed -> no-op"; exit 0
    fi
    echo "[$(date)] TRAIN {noise} R={rho} lr={lr} seed={seed} on $(hostname)"
    {model} COV_WEIGHT={cov} LR={lr} SEED={seed} \\
        OUTPUT_DIR={tdir} DEVICES=1 PER_GPU_BS={PER_GPU_BS} MAX_STEPS={MAX_STEPS} CKPT_EVERY={CKPT_EVERY} \\
        bash scripts/train/sudoku/{stem}.sh
    echo "[$(date)] EVAL"
    {model} \\
        CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval DEVICES=1 \\
        bash scripts/sample/sudoku/{stem}.sh
    echo "[$(date)] DONE"
    ''')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dry-run', action='store_true')
  ap.add_argument('--noises', nargs='+', default=list(NOISES), choices=list(NOISES))
  ap.add_argument('--rhos', nargs='+', default=RHOS)
  ap.add_argument('--lrs', nargs='+', default=LRS)
  ap.add_argument('--seeds', nargs='+', default=SEEDS)
  args = ap.parse_args()

  cells = [(n, r, lr, s) for s, n, r, lr in
           itertools.product(args.seeds, args.noises, args.rhos, args.lrs)]
  print(f'{PROJECT}: {len(cells)} cells ({len(args.noises)} noise x {len(args.rhos)} R x '
        f'{len(args.lrs)} lr x {len(args.seeds)} seed), 1 x 2080 Ti each')
  if args.dry_run:
    for c in cells:
      print(f'  {PROJECT}_{tag_of(*c):34} {NOISES[c[0]][1]}={trunc(c[0], c[1])} COV_WEIGHT={NOISES[c[0]][2]}')
    print('\n--- example body (first cell) ---\n' + job_body(*cells[0]))
    return

  os.makedirs(LOGS, exist_ok=True)
  active = active_jobnames()
  n_sub = n_skip = 0
  for c in cells:
    tag = tag_of(*c)
    jobname = f'{PROJECT}_{tag}'
    if os.path.exists(f'{OUT}/{tag}/eval/results.json') or jobname in active:
      n_skip += 1
      continue
    slurm = Slurm(job_name=jobname, output=f'{LOGS}/{tag}_%j.log',
                  partition='desa,thickstun', gres='gpu:nvidia_geforce_rtx_2080_ti:1',
                  ntasks=1, cpus_per_task=4, mem='28G', time='1-00:00:00')
    jid = slurm.sbatch(job_body(*c), sbatch_cmd='sbatch --requeue', verbose=False)
    print(f'  submitted {tag}: job {jid}')
    n_sub += 1
  print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
  main()
