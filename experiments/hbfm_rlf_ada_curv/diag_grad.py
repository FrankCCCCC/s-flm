#!/usr/bin/env python
"""Diagnostic (analysis only): gradient scale of the curvature channel vs the rest of the model.

Runs a few hundred optimizer steps of the anchor cell with adaptive curvature outside Lightning
(same model, data, bf16 autocast, AdamW, warmup and clip as `scripts/train/sudoku/hbfm.sh`) and
prints, every `--every` steps, the loss, the gradient norm of the curvature readout
(`backbone.curvature_layer`), the gradient norm of everything else, and the curvature scale
lambda of the batch.

`--hook raw` is the hook of `algo.py` (factor 2 * loss_b); `--hook rel` divides it by the batch
mean loss (the relative second moment), to compare their scale against `gradient_clip_val`.

Usage (compute node):  python experiments/hbfm_rlf_ada_curv/diag_grad.py --hook raw --steps 400
"""
import argparse
import sys

import hydra
import torch
import transformers

sys.path.insert(0, '.')
import algo  # noqa: E402
import dataloader  # noqa: E402
import main  # noqa: E402,F401  (registers the hydra resolvers)
import trainer_base  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--hook', default='raw', choices=['raw', 'rel'])
ap.add_argument('--bound', type=float, default=4.0)
ap.add_argument('--rate', type=float, default=3.0)
ap.add_argument('--steps', type=int, default=400)
ap.add_argument('--every', type=int, default=20)
ap.add_argument('--batch', type=int, default=128)
ap.add_argument('--warmup', type=int, default=2500)
args = ap.parse_args()


class _NoMetrics:
  def __init__(self, **kwargs):
    pass

  def to(self, *a, **k):
    return self


trainer_base.metrics.Metrics = _NoMetrics
with hydra.initialize(config_path='../../configs', version_base=None):
  cfg = hydra.compose(config_name='config', overrides=[
    'data=sudoku', 'data.cache_dir=data_cache', 'data.difficulty=hard',
    'model=tiny-hyperbolic-dit', 'model.embed_dim=9', 'model.init=ngpt',
    'algo=hbfm', 'algo.prod_factor_dim=[3,3,3]',
    'algo.prod_factor_gaussian_curvature=[-0.5,-0.5,-0.5]',
    f'algo.curvature_bound={args.bound}', f'algo.time_exp_rate={args.rate}',
    'algo.forward_type=horosphere', 'sampler=hbfm', 'sampler.t_max=6.0', 'noise=log-linear',
    f'loader.global_batch_size={args.batch}', f'loader.batch_size={args.batch}',
    'trainer.devices=1', 'trainer.num_nodes=1'])
torch.manual_seed(1)
tokenizer = dataloader.get_tokenizer(cfg)
model = algo.HyperbolicBoundaryFM(cfg, tokenizer=tokenizer).cuda().train()
data = dataloader.get_dataset(cfg, tokenizer, mode='train')
opt = torch.optim.AdamW(model.backbone.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                        weight_decay=0.0)
sched = transformers.get_constant_schedule_with_warmup(opt, num_warmup_steps=args.warmup)

scale = {'value': 1.0}
curv_log = {}
orig = model.curvature_scale


def curvature_scale(xt):
  curvature = orig(xt)
  curv_log['c'] = curvature.detach()
  if curvature.requires_grad:   # runs before algo.py's hook; the two factors multiply
    curvature.register_hook(lambda g: g * scale['value'])
  return curvature


model.curvature_scale = curvature_scale
gen = torch.Generator().manual_seed(0)
print('step  loss  |g_curv|  |g_rest|  clip  lambda mean/std/min/max  frac@bounds', flush=True)
for step in range(args.steps):
  idx = torch.randint(0, len(data), (args.batch,), generator=gen)
  batch = data[idx.tolist()]
  x = torch.as_tensor(batch['input_ids']).cuda()
  valid = torch.as_tensor(batch['attention_mask']).cuda()
  with torch.autocast('cuda', dtype=torch.bfloat16):
    loss, _ = model.nll(x, None, trainer_base.TrainingContext(), train_mode=True,
                        valid_tokens=valid)
    seq = (loss * valid).sum(-1) / valid.sum(-1)
    mean_loss = (loss * valid).sum() / valid.sum()
  scale['value'] = 1.0 if args.hook == 'raw' else 1.0 / seq.mean().item()
  opt.zero_grad()
  mean_loss.backward()
  norm = lambda params: torch.sqrt(sum(  # noqa: E731
    p.grad.square().sum() for p in params if p.grad is not None)).item()
  g_curv = norm(model.backbone.curvature_layer.parameters())
  g_all = norm(model.backbone.parameters())
  g_rest = max(g_all ** 2 - g_curv ** 2, 0.0) ** 0.5
  torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.0)
  opt.step()
  sched.step()
  if step % args.every == 0 or step == args.steps - 1:
    c = curv_log['c'][valid.bool()]
    at_bound = ((c > 0.95 * args.bound) | (c < 1 / (0.95 * args.bound))).float().mean().item()
    print(f'{step:4d}  {mean_loss.item():7.3f}  {g_curv:10.4f}  {g_rest:8.4f}  '
          f'{min(1.0, 1.0 / g_all):5.3f}  {c.mean().item():.3f}/{c.std().item():.3f}/'
          f'{c.min().item():.3f}/{c.max().item():.3f}  {at_bound:.3f}', flush=True)
