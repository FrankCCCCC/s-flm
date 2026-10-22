#!/usr/bin/env python
"""Does dropout move the curvature read during training? (analysis only)

`HyperbolicDiT.curvature` runs the trunk under the module's current mode, so during training the
clock lambda is read through dropout (p = model.dropout) while the sampler reads it in eval mode.
For one trained cell, compares lambda read in eval mode with `--draws` reads in train mode on the
same validation puzzles (EMA weights, the sampler's t = 0 state).

Usage (compute node): python experiments/hbfm_rlf_ada_curv/dropout_check.py <outputs/.../cell>
"""
import argparse
import sys

import torch

sys.path.insert(0, 'experiments/hbfm_rlf_ada_curv')
from common import DEVICE, load_cell  # noqa: E402
import dataloader  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('cell')
ap.add_argument('--ckpt', default='last.ckpt')
ap.add_argument('--boards', type=int, default=256)
ap.add_argument('--draws', type=int, default=8)
args = ap.parse_args()

cfg, tokenizer, model = load_cell(args.cell, args.ckpt)
data = dataloader.get_dataset(cfg, tokenizer, mode='valid')
x = torch.as_tensor(data[:args.boards]['input_ids']).to(DEVICE)
P = tokenizer.prompt_len
xt = torch.zeros(*x.shape, model.backbone.embed_dim, dtype=torch.float64, device=x.device)
xt[:, :P] = model._clean_state(x[:, :P])
with torch.no_grad():
  model.backbone.eval()
  lam_eval = model.curvature_scale(xt)[:, P:]
  model.backbone.train()
  lam_train = torch.stack([model.curvature_scale(xt)[:, P:] for _ in range(args.draws)])
  model.backbone.eval()
ratio = lam_train / lam_eval
print(f'{args.cell}  (dropout {cfg.model.dropout}, {args.boards} boards x {args.draws} train-mode draws)')
print(f'lambda eval mode: mean {lam_eval.mean():.4f}, std over positions {lam_eval.std():.4f}')
print(f'train / eval: mean {ratio.mean():.4f}, std {ratio.std():.4f}, '
      f'|log ratio| p50 {ratio.log().abs().median():.4f} p99 {ratio.log().abs().quantile(0.99):.4f}')
print(f'std of the train-mode draws / eval lambda, per position: mean {(lam_train.std(0) / lam_eval).mean():.4f}')
