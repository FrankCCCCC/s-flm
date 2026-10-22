#!/usr/bin/env python
"""The learned clock of each cell (analysis only): lambda = curvature_bound^tanh(h) read at the
sampler's initial state -- the origin at every position -- with EMA weights, for every
keep-{step}.ckpt not yet in <cell>/curvature.json (which report.py reads). Without a prompt every
position sees the same input, so lambda is one number per model; its spread over positions is
printed as the check.

Usage (compute node, GPU or CPU):
  python experiments/hbfm_rlf_ada_curv_tinystories_256/curvature_ts.py <outputs/.../cell> ...
"""
import glob
import json
import os
import re
import sys

import torch

sys.path.insert(0, 'experiments/hbfm_rlf_ada_curv')
from common import DEVICE, load_cell  # noqa: E402


def step_of(path):
  return int(re.search(r'keep-(\d+)\.ckpt$', path).group(1))


for cell in sys.argv[1:]:
  path = f'{cell}/curvature.json'
  done = json.load(open(path)) if os.path.exists(path) else {}
  for ckpt in sorted(glob.glob(f'{cell}/checkpoints/keep-*.ckpt'), key=step_of):
    if str(step_of(ckpt)) in done:
      continue
    cfg, _, model = load_cell(cell, os.path.basename(ckpt))
    z = torch.zeros(1, cfg.model.length, model.backbone.embed_dim, dtype=torch.float64, device=DEVICE)
    with torch.no_grad():
      lam = model.curvature_scale(z)
    done[str(step_of(ckpt))] = dict(mean=lam.mean().item(), min=lam.min().item(), max=lam.max().item())
    print(f'{os.path.basename(cell.rstrip("/"))} step {step_of(ckpt)}: lambda {lam.mean():.4f} '
          f'(min {lam.min():.4f}, max {lam.max():.4f}; bound {cfg.algo.curvature_bound:g})', flush=True)
    with open(path, 'w') as f:
      json.dump(done, f, indent=1)
