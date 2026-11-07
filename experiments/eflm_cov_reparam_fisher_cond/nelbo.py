#!/usr/bin/env python
"""Train / valid NELBO of the GOAL2 EFLM cells (analysis only, GPU), nats per token, EMA weights of last.ckpt.

The NELBO is the codebase's own (`algo.loss_type=elbo`: EFLM.nll -> elbo_loss = VDM diffusion term + reconstruction
CE at the clean end + prior KL), on the protocol of elbo_tune_tinystories_256's train / valid rows: the first
n sequences of the shuffled training loader (seed 1) or of the validation loader, one (t, noise) draw per
sequence (seed 1), per token = averaged over a sequence's tokens, then over sequences (mean +- standard error).

Usage:  python experiments/eflm_cov_reparam_fisher_cond/nelbo.py --runs <run> [<run> ...] [--n-seqs 2048]
        python experiments/eflm_cov_reparam_fisher_cond/nelbo.py --report
  <run> = a directory under outputs/eflm_cov_reparam_fisher_cond; writes data/nelbo/<run>_<split>.json,
  skipping existing ones.
"""
import argparse
import glob
import json
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, f'{REPO}/visualization']
OUT = f'{REPO}/outputs/eflm_cov_reparam_fisher_cond'
DATA = f'{HERE}/data/nelbo'


def load(run, bs):
  from loss_geometry import ALGO_BY_NAME, _load_config
  import dataloader, main as main_mod
  ckpt = os.path.realpath(f'{OUT}/{run}/checkpoints/last.ckpt')  # G3 runs: checkpoints/ links to the G3 checkout
  cfg = _load_config(os.path.dirname(os.path.dirname(ckpt)), ckpt,
                     types.SimpleNamespace(cache_dir=f'{REPO}/data_cache', batch_size=bs))
  cfg.algo.loss_type = 'elbo'
  tok = dataloader.get_tokenizer(cfg)
  m = main_mod._load_from_checkpoint(ALGO_BY_NAME['eflm'], cfg, tok).cuda().eval()
  m._eval_mode()  # EMA weights, as trainer.validate does
  return cfg, tok, m


def get_loader(cfg, tok, split):
  import dataloader
  torch.manual_seed(1)  # the training loader shuffles with the global RNG
  if split == 'valid':
    return dataloader.get_dataloaders(cfg, tok, skip_train=True, valid_seed=1)[1]
  return dataloader.get_dataloaders(cfg, tok, skip_valid=True)[0]


@torch.no_grad()
def run(m, dl, n_seqs):
  import trainer_base
  per_seq, n = [], 0
  it = iter(dl)
  torch.manual_seed(1)  # the (t, noise) draws, after the shuffling: the same draws for train and valid
  for batch in it:
    x, valid = batch['input_ids'].cuda(), batch['attention_mask'].cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
      loss, _ = m.nll(x, None, context=trainer_base.TrainingContext(), valid_tokens=valid)
    per_seq.append(((loss.double() * valid).sum(-1) / valid.sum(-1)).cpu())
    n += x.shape[0]
    if n >= n_seqs:
      break
  v = torch.cat(per_seq)[:n_seqs]
  return dict(nelbo=v.mean().item(), se=(v.std() / len(v) ** 0.5).item(), n_seqs=len(v))


def report():
  print('| run | split | n | NELBO |')
  print('|---|---|---:|---:|')
  for f in sorted(glob.glob(f'{DATA}/*.json')):
    r = json.load(open(f))
    print(f'| {r["run"]} | {r["split"]} | {r["n_seqs"]} | {r["nelbo"]:.3f} ± {r["se"]:.3f} |')


def main_():
  ap = argparse.ArgumentParser()
  ap.add_argument('--runs', nargs='*', default=[])
  ap.add_argument('--splits', nargs='+', default=['train', 'valid'])
  ap.add_argument('--n-seqs', type=int, default=2048)
  ap.add_argument('--bs', type=int, default=32)
  ap.add_argument('--report', action='store_true')
  args = ap.parse_args()
  if args.report:
    return report()
  os.makedirs(DATA, exist_ok=True)
  for r in args.runs:
    todo = [s for s in args.splits if not os.path.exists(f'{DATA}/{r}_{s}.json')]
    if not todo or not os.path.exists(f'{OUT}/{r}/checkpoints/last.ckpt'):
      continue
    cfg, tok, m = load(r, args.bs)
    for split in todo:
      res = run(m, get_loader(cfg, tok, split), args.n_seqs)
      res.update(run=r, split=split, ckpt=os.path.realpath(f'{OUT}/{r}/checkpoints/last.ckpt'))
      json.dump(res, open(f'{DATA}/{r}_{split}.json', 'w'), indent=1)
      print(f'{r} {split}: NELBO {res["nelbo"]:.4f} +- {res["se"]:.4f}', flush=True)
    del m
    torch.cuda.empty_cache()


if __name__ == '__main__':
  main_()
