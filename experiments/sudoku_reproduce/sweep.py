#!/usr/bin/env python
"""sweep.py -- Sudoku reproduction (setup.md): 7 models x 3 difficulties x 3 seeds = 63 cells.

Models: S-FLM base / truncated / truncated+adaptive (paper Table 1 rows), MDLM, DUO,
E-FLM + autonomous clock truncated at tau_max (R = 8, tau_max = 0.5, LR 5e-4: the best
plain-CE cell of experiments/eflm_rescale_auto_sudoku), and the same E-FLM + self-conditioning.
Every cell uses its method's script defaults otherwise (20k steps, global batch 256, bf16,
EMA 0.9999) and the scripts' sudoku_eval (2000 validation boards, 180 steps).

One SLURM job per cell: train, then sudoku_eval -> outputs/sudoku_reproduce/<tag>/eval/results.json.
Idempotent + resumable: a cell whose results.json exists or whose job name is queued is skipped;
a requeued (preempted) job resumes from checkpoints/last.ckpt. No nice here (Agent.md).

Usage: python experiments/sudoku_reproduce/sweep.py [--dry-run] [--models ...] [--difficulties ...] [--seeds ...]
"""
import argparse
import getpass
import itertools
import os
import subprocess

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev1/s-flm'
SFM_HOME = '/share/desa/nfs02/sc3379/sfm_home'
OUT = f'{REPO}/outputs/sudoku_reproduce'
LOGS = f'{REPO}/experiments/sudoku_reproduce/logs'

EFLM_AUTO = 'RHO=8 TAU_MAX=0.5'
# model tag -> (script stem under scripts/{train,sample}/sudoku/, env for train, env for eval)
MODELS = {
    'sfm': ('sfm', '', ''),
    'sfm_trunc': ('sfm_truncated', '', ''),
    'sfm_trunc_ada': ('sfm_truncated_adaptive', '', ''),
    'mdlm': ('mdlm', '', ''),
    'duo': ('duo', '', ''),
    'eflm_auto_trunc': ('eflm_rescale_auto', f'{EFLM_AUTO} LR=5e-4', EFLM_AUTO),
    'eflm_auto_trunc_sc': ('eflm_rescale_auto', f'{EFLM_AUTO} LR=5e-4 SELF_COND=true',
                           f'{EFLM_AUTO} SELF_COND=true'),
}
DIFFICULTIES = ['easy', 'medium', 'hard']
SEEDS = ['1', '2', '3']

# Ampere / Ada / Hopper GPUs (all >= 24 GB, so batch 256 without accumulation) on desa + the shared
# (preemptible) gpu / default partitions. Turing GPUs are left out: cells trained on a 2080 Ti as 2 x 128
# scored differently from batch 256 on Ampere (S-FLM trunc lower, MDLM easy higher; results.md). Excluded nodes: Blackwell or
# 2080 Ti mixed in (bhattacharjee-compute-02, unicorn-compute-01), slow L4s (rocky). ~2 h per cell on an A6000.
SLURM_KW = dict(partition='desa,gpu,default_partition', constraint='ampere|ada|hopper',
                exclude='bhattacharjee-compute-02,unicorn-compute-01,rocky', gres='gpu:1', ntasks=1,
                cpus_per_task=4, mem='24G', time='06:00:00')


def tag_of(model, diff, seed):
  return f'sudoku_reproduce_m-{model}_d-{diff}_s-{seed}'


def queued():
  out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                       capture_output=True, text=True).stdout
  return set(out.split())


def job_body(model, diff, seed):
  stem, tenv, eenv = MODELS[model]
  out = f'{OUT}/{tag_of(model, diff, seed)}'
  return f'''\
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export SLURM_JOB_NAME=bash
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export HF_HOME={SFM_HOME}/hf_cache XDG_CACHE_HOME={SFM_HOME}/cache
export PATH={SFM_HOME}/envs/sfm/bin:$PATH
cd {REPO}
# global batch 256 everywhere; a GPU under 20 GB would run it as 2 x 128 (none is in SLURM_KW's constraint)
PER_GPU_BS=$([ $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) -ge 20000 ] && echo 256 || echo 128)
echo "[$(date)] TRAIN {model} {diff} seed {seed} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader), PER_GPU_BS=$PER_GPU_BS"
{tenv} DIFFICULTY={diff} SEED={seed} OUTPUT_DIR={out} DEVICES=1 PER_GPU_BS=$PER_GPU_BS bash scripts/train/sudoku/{stem}.sh
echo "[$(date)] EVAL"
{eenv} DIFFICULTY={diff} CKPT_PATH={out}/checkpoints/last.ckpt OUTPUT_DIR={out}/eval DEVICES=1 \\
    bash scripts/sample/sudoku/{stem}.sh
echo "[$(date)] DONE"
'''


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dry-run', action='store_true')
  ap.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
  ap.add_argument('--difficulties', nargs='+', default=DIFFICULTIES, choices=DIFFICULTIES)
  ap.add_argument('--seeds', nargs='+', default=SEEDS)
  args = ap.parse_args()
  cells = list(itertools.product(args.models, args.difficulties, args.seeds))
  if args.dry_run:
    print(job_body(*cells[0]))
  os.makedirs(LOGS, exist_ok=True)
  active = queued()
  for cell in cells:
    tag = tag_of(*cell)
    if os.path.exists(f'{OUT}/{tag}/eval/results.json'):
      print(f'done    {tag}')
      continue
    if tag in active:
      print(f'queued  {tag}')
      continue
    if args.dry_run:
      print(f'submit  {tag} (dry run)')
      continue
    slurm = Slurm(job_name=tag, output=f'{LOGS}/{tag}_%j.log', **SLURM_KW)
    jid = slurm.sbatch(job_body(*cell), sbatch_cmd='sbatch --requeue', verbose=False)
    print(f'submit  {tag} -> job {jid}')


if __name__ == '__main__':
  main()
