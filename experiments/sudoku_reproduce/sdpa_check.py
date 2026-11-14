#!/usr/bin/env python
"""sdpa_check.py -- does the attention kernel explain why MDLM easy scored 92.8 on 2080 Ti GPUs and 88.2 on Ampere?

probe_kernels.py finds one numeric difference in a training step: a 2080 Ti cannot run FlashAttention in bf16, so
PyTorch uses its math attention kernel, which computes in fp32 (fp32 matmuls and softmax). The linear layers take
bf16 and accumulate in fp32 on both GPUs. This trains MDLM easy, seeds 1-3, on an A6000 (the GPU of all three
batch-256 MDLM easy cells) with `sdpa_math=True allow_tf32=False`: the 2080 Ti's attention (arm `sdpa_math`).
Arm `flash` is the same job with the default kernel, for extra seeds of the batch-256 cells on the same GPU. The
rest is sweep.py's job unchanged: same script, batch 256, 20k steps, eval (the eval keeps the default kernel;
results.md shows a checkpoint scores the same on either GPU type). Each job first logs the kernels its GPU runs.
A run whose results.json exists or whose job is queued is skipped.

Output: outputs/sudoku_reproduce/check/<arm>_<tag>/eval/results.json
Usage: python experiments/sudoku_reproduce/sdpa_check.py [--arms sdpa_math flash] [--seeds 1 2 3]
"""
import argparse
import os

from simple_slurm import Slurm

import sweep

ARMS = {  # arm -> (training overrides, probe_kernels.py flags)
    'sdpa_math': ('sdpa_math=True allow_tf32=False', '--sdpa-math --no-tf32'),
    'flash': ('', ''),
}

ap = argparse.ArgumentParser()
ap.add_argument('--arms', nargs='+', default=['sdpa_math'], choices=list(ARMS))
ap.add_argument('--seeds', nargs='+', default=sweep.SEEDS)
args = ap.parse_args()
active = sweep.queued()
for arm in args.arms:
  overrides, probe_flags = ARMS[arm]
  for seed in args.seeds:
    tag = sweep.tag_of('mdlm', 'easy', seed)
    name = f'sudoku_reproduce_check_{arm}_mdlm_easy_s{seed}'
    new_out = f'{sweep.OUT}/check/{arm}_{tag}'
    if os.path.exists(f'{new_out}/eval/results.json') or name in active:
      print(f'skip    {name}')
      continue
    body = sweep.job_body('mdlm', 'easy', seed)
    old_out = f'{sweep.OUT}/{tag}'
    train, train_echo = 'bash scripts/train/sudoku/mdlm.sh\n', 'echo "[$(date)] TRAIN'
    assert body.count(old_out) == 3 and body.count(train) == 1 and body.count(train_echo) == 1
    body = (body.replace(old_out, new_out)
            .replace(train, f'bash scripts/train/sudoku/mdlm.sh {overrides}\n')
            .replace(train_echo, f'python experiments/sudoku_reproduce/probe_kernels.py {probe_flags}\n' + train_echo))
    slurm = Slurm(job_name=name, output=f'{sweep.LOGS}/{name}_%j.log', **{**sweep.SLURM_KW, 'constraint': 'a6000'})
    print(f'submit  {name} ->', slurm.sbatch(body, sbatch_cmd='sbatch --requeue', verbose=False))
