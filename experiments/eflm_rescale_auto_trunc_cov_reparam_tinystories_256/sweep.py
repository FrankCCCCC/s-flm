#!/usr/bin/env python
"""eflm_rescale_auto_trunc_cov_reparam_tinystories_256 — rescaled E-FLM with the Gaussian (Bayes) readout on
TinyStories seq 256: noise schedule x R x LR (setup.md; design in EXPERIMENT.md).

  noise  auto    = noise=autonomous,           TAU_MAX   = tau*(R),   CE x |alpha_t'| (noise_sched_cov_weight)
         autoada = noise=autonomous-adaptive,  TAU_MAX   = tau*(R),   plain CE
         llada   = noise=log-linear-adaptive,  ALPHA_MAX = alpha*(R), plain CE
  R      {0.1, 0.5, 1.0, 8.0}  (rho_min = rho_max = R)
  LR     {3e-4, 1e-3, 5e-3}
tau*(R) / alpha*(R) = alpha_star_euclidean(V=50257, embed_norm=R, auto_clock=True / False) (docs/truncation.md).

Fixed: algo.forward_type=gaussian, small-sphere-dit, ngpt init, 30k steps, global batch 512 = 4 GPUs x 32 x
accumulation 4 (setup.md: the adaptive refit fires once per micro-batch, see adaptivesched_refit_bug.md; the
autonomous arm uses the same layout), bf16, EMA 0.9999, 1 seed.
Eval (1 GPU): ppl_eval -> eval/ppl.json, sample_eval (GenPPL + entropy) -> eval/samples_genppl.json;
exact velocity, top_k_velocity 1, 180 steps, greedy last step.

ORCHESTRATION ONLY: one SLURM job per cell runs scripts/train/tinystories/<stem>.sh then
scripts/sample/tinystories/<stem>.sh. Idempotent + resumable: skips a cell whose eval/samples_genppl.json exists
or whose job is queued; resubmitting resumes from checkpoints/last.ckpt.

Usage:  python experiments/eflm_rescale_auto_trunc_cov_reparam_tinystories_256/sweep.py [--dry-run]
            [--noises auto autoada llada] [--rhos 0.1 0.5 1.0 8.0] [--lrs 3e-4 1e-3 5e-3]
then    python experiments/eflm_rescale_auto_trunc_cov_reparam_tinystories_256/report.py
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

PROJECT = 'eflm_rescale_auto_trunc_cov_reparam_tinystories_256'
LOGS = f'{REPO}/experiments/{PROJECT}/logs'
OUT = f'{REPO}/outputs/{PROJECT}'
V = 50257
# noise -> (train/sample script stem, truncation knob, noise_sched_cov_weight)
NOISES = {
  'auto': ('eflm_rescale_auto_truncation', 'TAU_MAX', 'true'),
  'autoada': ('eflm_rescale_auto_adaptive', 'TAU_MAX', 'false'),
  'llada': ('eflm_rescale_truncated_adaptive', 'ALPHA_MAX', 'false'),
}
RHOS = ['0.1', '0.5', '1.0', '8.0']
LRS = ['1e-3', '3e-4', '5e-3']   # submission order: the previous sweep's best LR first
SEQ_LEN = 256
DEVICES = 4
PER_GPU_BS = 32
MAX_STEPS = 30000
CKPT_EVERY = 5000
EXCLUDE = 'desa-compute-01'   # 2080 Ti (11 GB)


def trunc(noise, rho):
  """tau*(R) on the autonomous clock, alpha*(R) for the log-linear one."""
  auto_clock = NOISES[noise][1] == 'TAU_MAX'
  return f'{alpha_star_euclidean(V, embed_norm=float(rho), auto_clock=auto_clock):.4f}'


def tag_of(noise, rho, lr):
  return f'ns-{noise}_r-{rho}_lr-{lr}'


def active_jobnames():
  try:
    out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                         capture_output=True, text=True).stdout
    return set(out.split())
  except Exception:
    return set()


def job_body(noise, rho, lr):
  stem, knob, cov = NOISES[noise]
  tag = tag_of(noise, rho, lr)
  tdir = f'{OUT}/{tag}'
  # must match train <-> eval; the autonomous clock is truncated by TAU_MAX only (noise.alpha_max must stay null)
  model = (f'RHO={rho} {knob}={trunc(noise, rho)} FORWARD_TYPE=gaussian SEQ_LEN={SEQ_LEN}'
           + (' ALPHA_MAX=null' if noise == 'auto' else ''))
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
    if [ -f {tdir}/eval/samples_genppl.json ]; then
        echo "[$(date)] cell already completed -> no-op"; exit 0
    fi
    echo "[$(date)] TRAIN {noise} R={rho} lr={lr} on $(hostname)"
    {model} COV_WEIGHT={cov} LR={lr} \\
        OUTPUT_DIR={tdir} RUN_NAME={tag} WANDB_GROUP={PROJECT} \\
        DEVICES={DEVICES} PER_GPU_BS={PER_GPU_BS} MAX_STEPS={MAX_STEPS} CKPT_EVERY={CKPT_EVERY} SAVE_TOPK=1 \\
        bash scripts/train/tinystories/{stem}.sh
    echo "[$(date)] EVAL"
    {model} \\
        CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval DEVICES=1 \\
        bash scripts/sample/tinystories/{stem}.sh
    echo "[$(date)] DONE"
    ''')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dry-run', action='store_true')
  ap.add_argument('--noises', nargs='+', default=list(NOISES), choices=list(NOISES))
  ap.add_argument('--rhos', nargs='+', default=RHOS)
  ap.add_argument('--lrs', nargs='+', default=LRS)
  args = ap.parse_args()

  cells = [(n, r, lr) for lr, n, r in itertools.product(args.lrs, args.noises, args.rhos)]
  print(f'{PROJECT}: {len(cells)} cells ({len(args.noises)} noise x {len(args.rhos)} R x '
        f'{len(args.lrs)} lr), {DEVICES} GPUs each')
  if args.dry_run:
    for c in cells:
      print(f'  {PROJECT}_{tag_of(*c):28} {NOISES[c[0]][1]}={trunc(c[0], c[1])} COV_WEIGHT={NOISES[c[0]][2]}')
    print('\n--- example body (first cell) ---\n' + job_body(*cells[0]))
    return

  os.makedirs(LOGS, exist_ok=True)
  active = active_jobnames()
  n_sub = n_skip = 0
  for c in cells:
    tag = tag_of(*c)
    jobname = f'{PROJECT}_{tag}'
    if os.path.exists(f'{OUT}/{tag}/eval/samples_genppl.json') or jobname in active:
      n_skip += 1
      continue
    slurm = Slurm(job_name=jobname, output=f'{LOGS}/{tag}_%j.log',
                  partition='thickstun,desa', exclude=EXCLUDE, gres=f'gpu:{DEVICES}',
                  ntasks=1, cpus_per_task=16, mem='128G', time='3-00:00:00')
    jid = slurm.sbatch(job_body(*c), sbatch_cmd='sbatch --requeue', verbose=False)
    print(f'  submitted {tag}: job {jid}')
    n_sub += 1
  print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
  main()
