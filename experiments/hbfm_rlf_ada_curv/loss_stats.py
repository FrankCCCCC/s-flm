#!/usr/bin/env python
"""Mean and variance of the training loss on held-out puzzles (analysis only).

For one trained cell, draws fresh (t, bridge noise) on validation puzzles and reports the mean and
the variance of the per-sequence weighted loss -- the two quantities math.md is about. For an
adaptive cell the same checkpoint is also evaluated with its curvature switched off (lambda = 1):
the mean should not move, the variance is what the learned curvature bought.

Usage (compute node, GPU or CPU):
  python experiments/hbfm_rlf_ada_curv/loss_stats.py <outputs/hbfm_rlf_ada_curv/cell> \
      [--ckpt last.ckpt] [--boards 512] [--repeats 32] [--no-ema]
"""
import argparse
import sys

import torch

sys.path.insert(0, 'experiments/hbfm_rlf_ada_curv')
from common import DEVICE, load_cell  # noqa: E402
import dataloader  # noqa: E402
import trainer_base  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('cell')
ap.add_argument('--ckpt', default='last.ckpt')
ap.add_argument('--boards', type=int, default=512)
ap.add_argument('--repeats', type=int, default=32)
ap.add_argument('--no-ema', action='store_true')
args = ap.parse_args()

cfg, tokenizer, model = load_cell(args.cell, args.ckpt, ema=not args.no_ema)

data = dataloader.get_dataset(cfg, tokenizer, mode='valid')[:args.boards]
x = torch.as_tensor(data['input_ids']).to(DEVICE)
valid = torch.as_tensor(data['attention_mask']).to(DEVICE)


def sequence_losses():
  out = []
  torch.manual_seed(0)
  with torch.no_grad():
    for _ in range(args.repeats):
      loss, _ = model.nll(x, None, trainer_base.TrainingContext(), valid_tokens=valid)
      out.append((loss * valid).sum(-1) / valid.sum(-1))
  return torch.cat(out)


def row(name, v):
  print(f'| {name} | {v.mean():.4f} ± {v.std() / len(v) ** 0.5:.4f} | {v.var():.4f} '
        f'| {v.std() / v.mean():.2f} | {v.max():.2f} |', flush=True)


print(f'{args.cell.rstrip("/").split("/")[-1]}  ({args.ckpt}, {"online" if args.no_ema else "EMA"} '
      f'weights, {len(x)} boards x {args.repeats} draws)')
print('| curvature | mean ± s.e. | variance | std / mean | max |')
print('|---|---|---|---|---|')
if cfg.algo.curvature_bound > 1:
  row('learned lambda', sequence_losses())
  learned = model.curvature_scale
  model.curvature_scale = lambda xt: torch.ones_like(learned(xt))
  row('lambda = 1 (same weights)', sequence_losses())
else:
  row('fixed', sequence_losses())
