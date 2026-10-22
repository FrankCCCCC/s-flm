#!/usr/bin/env python
"""Learned curvature vs the model's uncertainty, per empty Sudoku cell (adaptive-curvature HBFM,
experiments/hbfm_rlf_ada_curv).

For every empty cell of every validation board, at the sampler's initial state (origin everywhere,
the puzzle pinned at its clean boundary points):
  x = ln lambda, the log of the curvature scale the sampler uses (`curvature_scale`; position i runs
      at K_i = lambda_i K, so ln lambda = ln(K_i / K): < 0 flatter than the base curvature, > 0
      sharper; ln|K_i| is the same axis shifted by ln|K|),
  y = the entropy (nats) of the model's own posterior over the cell at that state.
The title carries the Spearman correlation (the number of
`experiments/hbfm_rlf_ada_curv/curvature_stats.py`) and the Pearson correlation of log lambda with
the entropy, over all plotted cells. Needs a trained cell with curvature_bound > 1.

Example (compute node, GPU or CPU):
  python visualization/curvature_vs_uncertainty.py \
    outputs/hbfm_rlf_ada_curv/cb-4_mode-exp_rate-3_k-0.5_ub-na_tm-6_fwd-horo_lr-3e-4_seed-1
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, 'experiments/hbfm_rlf_ada_curv')
from common import DEVICE, load_cell  # noqa: E402
import dataloader  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('cell')
ap.add_argument('--ckpt', default='last.ckpt')
ap.add_argument('--boards', type=int, default=512)
ap.add_argument('--out', default=None, help='default: experiments/hbfm_rlf_ada_curv/figures/'
                'curvature_vs_uncertainty_<cell>.png')
args = ap.parse_args()
name = args.cell.rstrip('/').split('/')[-1]
out = args.out or f'experiments/hbfm_rlf_ada_curv/figures/curvature_vs_uncertainty_{name}.png'

cfg, tokenizer, model = load_cell(args.cell, args.ckpt)
data = dataloader.get_dataset(cfg, tokenizer, mode='valid')
x = torch.as_tensor(data[:args.boards]['input_ids']).to(DEVICE)
P = tokenizer.prompt_len
with torch.no_grad():
  xt = torch.zeros(*x.shape, model.backbone.embed_dim, dtype=torch.float64, device=x.device)
  xt[:, :P] = model._clean_state(x[:, :P])
  curvature = model.curvature_scale(xt)[:, P:]
  log_p = model.forward(xt=xt, sigma=torch.zeros(len(x), 1, device=x.device))[:, P:]
entropy = -(log_p.exp() * log_p).sum(-1)
empty = x[:, 1:1 + tokenizer.seq_len] == tokenizer.empty_token_id     # aligned with the solution
lam, ent = curvature[empty].cpu().numpy(), entropy[empty].cpu().numpy()
rho, r = stats.spearmanr(lam, ent).statistic, stats.pearsonr(np.log(lam), ent).statistic

fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.scatter(np.log(lam), ent, s=5, alpha=0.25, linewidths=0, color='#2a78d6')
ax.axvline(0, color='0.5', lw=1)
ax.text(0, 1, ' base curvature K', transform=ax.get_xaxis_transform(), va='top', fontsize=8, color='0.35')
ax.set(xlabel=r'log curvature scale $\ln\lambda = \ln(K_i / K)$  (< 0: flatter than the base K)',
       ylabel='posterior entropy at t = 0 (nats)')
ax.set_title(f'{name}\n{len(lam)} empty cells of {len(x)} boards ({args.ckpt}, EMA): '
             f'Spearman $\\rho$ = {rho:+.3f}, Pearson r(ln $\\lambda$, entropy) = {r:+.3f}', fontsize=9)
ax.grid(alpha=0.3)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight')
np.savez(out[:-len('.png')] + '.npz', lam=lam, ent=ent)   # the plotted cells, for re-analysis without the model
print(f'wrote {out}: Spearman {rho:+.3f}, Pearson(ln lambda) {r:+.3f}, {len(lam)} cells; '
      f'ln lambda in [{np.log(lam).min():.2f}, {np.log(lam).max():.2f}]')
