#!/usr/bin/env python
"""viz.py -- figures of every evaluated cell of sweep.py (setup.md), one SLURM job per cell.

  boards_stages_<tag>_confidence.png   visualization/sudoku_boards_stages.py: boards 0-5 at steps 0, 10%,
                                       25%, 50%, 75% and the decoded board (every model)
  acc_diffusion_time_<tag>.png         visualization/sudoku_acc_diffusion_time.py: empty-cell and board
                                       accuracy of the argmax along the sampler, 2000 boards (ACC_MODELS)
-> experiments/sudoku_reproduce/figures/. Skips a cell whose figures exist or whose job is queued, and a cell
not evaluated yet (rerun later). Any GPU works (analysis only), so the 2080 Ti nodes are allowed too.

Usage: python experiments/sudoku_reproduce/viz.py [--dry-run]
"""
import argparse
import itertools
import os

from simple_slurm import Slurm

from sweep import DIFFICULTIES, LOGS, MODELS, OUT, REPO, SEEDS, SFM_HOME, queued, tag_of

FIG = f'{REPO}/experiments/sudoku_reproduce/figures'
ACC_MODELS = ['mdlm', 'duo', 'eflm_auto_trunc', 'eflm_auto_trunc_sc']
SLURM_KW = dict(partition='desa,gpu,default_partition', constraint='2080ti|ampere|ada|hopper',
                exclude='bhattacharjee-compute-02', gres='gpu:1', ntasks=1, cpus_per_task=4,
                mem='24G', time='04:00:00')


def figures(model, tag):
  out = [f'{FIG}/boards_stages_{tag}_confidence.png']
  return out + ([f'{FIG}/acc_diffusion_time_{tag}.png'] if model in ACC_MODELS else [])


def job_body(model, tag):
  cell = f'{OUT}/{tag}'
  acc = (f'python visualization/sudoku_acc_diffusion_time.py {cell} --out {FIG}/acc_diffusion_time_{tag}.png\n'
         if model in ACC_MODELS else '')
  return f'''\
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline
export HF_HOME={SFM_HOME}/hf_cache XDG_CACHE_HOME={SFM_HOME}/cache
export PATH={SFM_HOME}/envs/sfm/bin:$PATH
cd {REPO}
echo "[$(date)] VIZ {tag} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python visualization/sudoku_boards_stages.py {cell} --out {FIG}/boards_stages_{tag}_confidence.png
{acc}echo "[$(date)] DONE"
'''


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dry-run', action='store_true')
  args = ap.parse_args()
  os.makedirs(FIG, exist_ok=True)
  active = queued()
  for model, diff, seed in itertools.product(MODELS, DIFFICULTIES, SEEDS):
    tag = tag_of(model, diff, seed)
    name = tag.replace('sudoku_reproduce_', 'sudoku_reproduce_viz_')
    if not os.path.exists(f'{OUT}/{tag}/eval/results.json'):
      continue
    if all(os.path.exists(f) for f in figures(model, tag)):
      print(f'done    {name}')
      continue
    if name in active:
      print(f'queued  {name}')
      continue
    if args.dry_run:
      print(f'submit  {name} (dry run)\n{job_body(model, tag)}')
      continue
    slurm = Slurm(job_name=name, output=f'{LOGS}/{name}_%j.log', **SLURM_KW)
    print(f'submit  {name} -> job {slurm.sbatch(job_body(model, tag), verbose=False)}')


if __name__ == '__main__':
  main()
