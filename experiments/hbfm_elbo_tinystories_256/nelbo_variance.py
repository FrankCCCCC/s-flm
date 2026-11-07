#!/usr/bin/env python
"""Expectation and variance of the NELBO estimate vs the time proposal and the word MC (analysis only).

For one trained cell, the same N validation sequences are scored under each setting, with R
independent draws of (t, bridge noise, words) per sequence and the same seeds in every setting:
  exp             the cell's own proposal; the horizon is T = ln(1/eps)/rate
  trunc_exp ub    the proposal cut at ub, and the reconstruction CE taken at ub: the bound of the
                  process that stops at ub (ub <= T keeps t inside the trained range)
  exp, S words    the path-KL rate's sum over the V words replaced by its S-word MC estimate
                  (algo.elbo_num_words, unbiased); every other setting uses the exact rate
A trunc_exp-trained cell is scored at its own ub first, then at the shorter ubs, and with S words at its ub.
Per setting (nats/token): the mean NELBO, split into the reconstruction CE at the horizon and the
path-KL integral, its paired difference from exp (mean +- s.e. over sequences), the estimator std
(within a sequence, across draws), the between-sequence std of the per-sequence mean, and the
total std across (sequence, draw), which is what trainer/loss_std and val/loss_std log.
At the Bayes posterior every horizon gives the same bound; a trained model whose posterior is
factorized over positions pays more at a short horizon through the reconstruction term.

Usage (compute node, 1 GPU):
  python experiments/hbfm_elbo_tinystories_256/nelbo_variance.py <outputs/.../cell> \
      [--ckpt last.ckpt] [--n 256] [--draws 4] [--ubs 0.25,0.5,1,2] [--words 4,16,64]
"""
import argparse
import json
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
ap.add_argument('--ubs', default='0.25,0.5,1,2', help="'' = none")
ap.add_argument('--words', default='4,16,64', help="'' = none")
args = ap.parse_args()

cfg, tokenizer, model = load_cell(args.cell, args.ckpt)
algo = model.config.algo
own = algo.time_conversion_mode
assert own in ('exp', 'trunc_exp'), 'the settings are cut from an exp- or trunc_exp-trained cell'
data = dataloader.get_dataset(cfg, tokenizer, mode='valid')[:args.n]
x = torch.as_tensor(data['input_ids']).to(DEVICE)
valid = torch.as_tensor(data['attention_mask']).to(DEVICE)

recon = {}
reconstruction_loss = model.reconstruction_loss
def stash(**kwargs):   # keep the reconstruction term of each elbo_loss call
  recon['loss'] = reconstruction_loss(**kwargs)
  return recon['loss']
model.reconstruction_loss = stash


def score(mode, ub=None, words=None):
  """Per-sequence NELBO and reconstruction term, [draws, n] each."""
  algo.time_conversion_mode, model.elbo_num_words = mode, words
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


if own == 'exp':
  T = 6.907755 / algo.time_exp_rate   # ln(1 / sampling_eps) / rate, sampling_eps = 1e-3
  settings = [('exp', f'exp (T = {T:.2f})', dict(mode='exp'))]
else:   # trained on [0, ub]: only the shorter horizons stay inside the trained range
  T = algo.time_range_upper_bound
  settings = [('trunc_exp', f'trunc_exp ub {T:g} (trained)', dict(mode='trunc_exp', ub=T))]
settings += [('trunc_exp', f'trunc_exp ub {ub:g}', dict(mode='trunc_exp', ub=ub))
             for ub in map(float, filter(None, args.ubs.split(','))) if ub < T]
settings += [(own, f'{settings[0][1]}, S = {s}', dict(settings[0][2], words=s))
             for s in map(int, filter(None, args.words.split(',')))]

name = os.path.basename(args.cell.rstrip('/'))
print(f'{name}  ({args.ckpt}, EMA, {len(x)} sequences x {args.draws} draws; curvature_bound '
      f'{algo.curvature_bound:g}, rate {algo.time_exp_rate:g})', flush=True)
print('| setting | NELBO | recon. CE | path KL | Δ vs row 1 | estimator std | between-seq std | total std |')
print('|---|---:|---:|---:|---:|---:|---:|---:|', flush=True)
rows, base = {}, None
for _, label, kw in settings:
  total, rec = score(**kw)
  seq = total.mean(0)
  base = seq if base is None else base
  diff = seq - base
  rows[label] = dict(
    nelbo=total.mean().item(), recon=rec.mean().item(), path=(total - rec).mean().item(),
    diff=diff.mean().item(), diff_se=(diff.std() / len(x) ** 0.5).item(),
    estimator_std=total.var(0).mean().sqrt().item(), between_std=seq.std().item(),
    total_std=total.reshape(-1).std().item())
  r = rows[label]
  print(f'| {label} | {r["nelbo"]:.4f} | {r["recon"]:.4f} | {r["path"]:.4f} | {r["diff"]:+.4f} ± '
        f'{r["diff_se"]:.4f} | {r["estimator_std"]:.4f} | {r["between_std"]:.4f} | {r["total_std"]:.4f} |',
        flush=True)
with open(f'{args.cell}/nelbo_variance_{os.path.splitext(args.ckpt)[0]}.json', 'w') as f:
  json.dump(dict(ckpt=args.ckpt, n=len(x), draws=args.draws, rows=rows), f, indent=1)
