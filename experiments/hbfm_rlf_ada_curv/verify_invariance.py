#!/usr/bin/env python
"""Verification (analysis only): the numbers behind math.md sections 3, 4 and 6.

On the real HBFM model at init with the horosphere readout (zero residual = the context-free
Bayes posterior, for which the invariance is exact) and the test geometry ((H^3)^4, K = -1,
exp proposal of rate 1), prints

  1. mean +- s.e. and variance of the per-sequence weighted loss for several curvature patterns
     (the mean must not move, the variance must);
  2. the curvature gradient: the pathwise estimate through the hook of `algo.py` against the
     score-form E[loss^2 (1 - r t)], for several global scales.

Usage (compute node):  python experiments/hbfm_rlf_ada_curv/verify_invariance.py
"""
import math
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, 'tests')
sys.path.insert(0, '.')
import trainer_base  # noqa: E402
from test_hbfm import DEVICE, _build_model, _tokens  # noqa: E402

LOSS = ['algo.forward_type=horosphere', 'algo.readout_precision=float64']
mp = pytest.MonkeyPatch()
model, cfg = _build_model(mp, LOSS + ['algo.curvature_bound=4.0'])
base, _ = _build_model(mp, LOSS)
rate = cfg.algo.time_exp_rate
torch.manual_seed(5)
x = _tokens(model, B=4096)


def fix(pattern):
  model.curvature_scale = lambda xt: torch.as_tensor(
    pattern, dtype=torch.float64, device=xt.device).expand(xt.shape[0], xt.shape[1])


def losses(m, repeats=16):
  with torch.no_grad():
    return torch.cat([m.nll(x, None, trainer_base.TrainingContext())[0].mean(-1)
                      for _ in range(repeats)])


print('1. per-sequence weighted loss, 65536 sequences per row')
print('| curvature scale lambda | mean ± s.e. | variance |')
print('|---|---|---|')
ref = losses(base)
print(f'| fixed-curvature model | {ref.mean():.4f} ± {ref.std() / math.sqrt(len(ref)):.4f} | {ref.var():.3f} |')
for name, pattern in [('1 (adaptive path)', 1.0), ('1/4', 0.25), ('1/2', 0.5), ('2', 2.0), ('4', 4.0),
                      ('mixed [1/4, 1/2, 1, 2, 4, 1/2, 2, 1]', [0.25, 0.5, 1.0, 2.0, 4.0, 0.5, 2.0, 1.0])]:
  fix(pattern)
  v = losses(model)
  print(f'| {name} | {v.mean():.4f} ± {v.std() / math.sqrt(len(v)):.4f} | {v.var():.3f} |', flush=True)

print('\n2. d E[loss^2] / d log(lambda), global scale, 16 batches of 4096')
print('| lambda | pathwise (hook) | score form E[loss^2 (1 - r t)] | s.e. of the difference |')
print('|---|---|---|---|')
for scale in [0.25, 0.5, 1.0, 2.0]:
  log_scale = torch.zeros((), dtype=torch.float64, device=DEVICE, requires_grad=True)
  model.curvature_scale = lambda xt: (scale * log_scale.exp()).expand(xt.shape[0], xt.shape[1])
  pathwise, score = [], []
  for _ in range(16):
    loss, t = model.nll(x, None, trainer_base.TrainingContext())
    seq = loss.mean(-1)
    (grad,) = torch.autograd.grad(seq.mean(), log_scale)
    heat = -torch.log(1 - model.noise(t)[1].double()) / rate
    pathwise.append(grad.item() * cfg.model.hidden_size ** 0.5)   # undo the hook's 1 / sqrt(D)
    score.append((seq.detach().square() * (1 - rate * heat)).mean().item())
  d = np.array(pathwise) - np.array(score)
  print(f'| {scale:g} | {np.mean(pathwise):.4f} | {np.mean(score):.4f} | {d.std(ddof=1) / 4:.4f} |', flush=True)
