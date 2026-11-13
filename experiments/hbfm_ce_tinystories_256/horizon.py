#!/usr/bin/env python
"""Expectation and variance of the NELBO estimate vs the horizon, on one trained cell (analysis only).

The same N validation sequences are scored under each setting, with R independent draws of (t, bridge noise) per
sequence and the same seeds in every setting:
  exp, eps        t ~ U[eps, 1]: the horizon is ln(1/eps)/rate
  trunc_exp, ub   the proposal cut at ub, the reconstruction CE taken at ub
Only horizons inside the cell's trained range are scored. Per setting (nats/token): the mean NELBO, split into the
reconstruction CE at the horizon and the path-KL integral, the estimator std (within a sequence, across draws), the
between-sequence std, and the total std (what val/loss_std logs).

`corrected`: t is drawn on [eps, 1], but algo.HyperbolicBoundaryFM's weight is 1/q(t) without its factor (1 - eps), so
the path-KL term is overstated by 1 / (1 - eps): 0.1% at eps 1e-3, 1% at 1e-2, 11% at 0.1. corrected = recon. + (1 - eps) path.
sampler.t_max is not read by the NELBO, so T_MAX moves neither its mean nor its variance; it only moves the samples.

Usage (compute node, 1 GPU):
  python experiments/hbfm_ce_tinystories_256/horizon.py <cell dir> [--ckpt last.ckpt] [--n 256] [--draws 4]
      [--eps 1e-5,1e-3,1e-2,0.05,0.1,0.2] [--ubs 0.3,0.4,0.5,0.6,0.8,1,1.5,2]
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, 'experiments/hbfm_rlf_ada_curv')
from common import DEVICE, load_cell  # noqa: E402
import dataloader  # noqa: E402
import trainer_base  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('cell')
ap.add_argument('--ckpt', default='last.ckpt')
ap.add_argument('--n', type=int, default=256)
ap.add_argument('--draws', type=int, default=4)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--eps', default='1e-5,1e-3,1e-2,0.05,0.1,0.2')
ap.add_argument('--ubs', default='0.3,0.4,0.5,0.6,0.8,1,1.5,2')
args = ap.parse_args()

cfg, tokenizer, model = load_cell(args.cell, args.ckpt)
algo = model.config.algo
rate, own_eps, own = algo.time_exp_rate, model.sampling_eps, algo.time_conversion_mode
trained = math.log(1 / own_eps) / rate if own == 'exp' else algo.time_range_upper_bound
model.loss_type = 'elbo'   # trained on the CE; scored on the exact NELBO
model.elbo_num_words = None
data = dataloader.get_dataset(cfg, tokenizer, mode='valid')[:args.n]
x = torch.as_tensor(data['input_ids']).to(DEVICE)
valid = torch.as_tensor(data['attention_mask']).to(DEVICE)

recon = {}
reconstruction_loss = model.reconstruction_loss
def stash(**kwargs):   # keep the reconstruction term of each elbo_loss call
  recon['loss'] = reconstruction_loss(**kwargs)
  return recon['loss']
model.reconstruction_loss = stash


def score(mode, eps, ub=None):
  """Per-sequence NELBO and reconstruction term, [draws, n] each."""
  algo.time_conversion_mode, model.sampling_eps = mode, eps
  if ub is not None:
    algo.time_range_upper_bound = ub
  total, rec = torch.zeros(2, args.draws, len(x), dtype=torch.float64)
  with torch.no_grad():
    for r in range(args.draws):
      torch.manual_seed(r)
      for i in range(0, len(x), args.batch):
        xb, vb = x[i:i + args.batch], valid[i:i + args.batch]
        loss, _ = model.nll(xb, None, trainer_base.TrainingContext(), valid_tokens=vb)
        total[r, i:i + args.batch] = ((loss * vb).sum(-1) / vb.sum(-1)).cpu()
        rec[r, i:i + args.batch] = ((recon['loss'] * vb).sum(-1) / vb.sum(-1)).cpu()
  return total, rec


settings = [(f'exp eps {e:g}', math.log(1 / e) / rate, dict(mode='exp', eps=e))
            for e in map(float, filter(None, args.eps.split(',')))]
settings += [(f'trunc_exp ub {ub:g}', ub, dict(mode='trunc_exp', eps=own_eps, ub=ub))
             for ub in map(float, filter(None, args.ubs.split(',')))]
name = os.path.basename(args.cell.rstrip('/'))
print(f'{name} ({args.ckpt}, EMA, {len(x)} sequences x {args.draws} draws; trained {own} to horizon {trained:.2f}, '
      f'rate {rate:g}, curvature_bound {algo.curvature_bound:g})', flush=True)
print('| setting | horizon | NELBO | recon. CE | path KL | corrected | estimator std | between-seq std | total std |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|', flush=True)
rows = {}
for label, T, kw in sorted(settings, key=lambda s: s[1]):
  if T > trained * 1.001:
    continue
  total, rec = score(**kw)
  path = (total - rec).mean().item()
  r = rows[label] = dict(
    horizon=T, nelbo=total.mean().item(), recon=rec.mean().item(), path=path,
    corrected=rec.mean().item() + (1 - kw['eps']) * path, estimator_std=total.var(0).mean().sqrt().item(),
    between_std=total.mean(0).std().item(), total_std=total.reshape(-1).std().item())
  print(f'| {label} | {T:.2f} | {r["nelbo"]:.4f} | {r["recon"]:.4f} | {path:.4f} | {r["corrected"]:.4f} | '
        f'{r["estimator_std"]:.4f} | {r["between_std"]:.4f} | {r["total_std"]:.4f} |', flush=True)
with open(f'{args.cell}/horizon_{os.path.splitext(args.ckpt)[0]}.json', 'w') as f:
  json.dump(dict(ckpt=args.ckpt, n=len(x), draws=args.draws, rows=rows), f, indent=1)
