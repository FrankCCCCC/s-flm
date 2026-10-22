#!/usr/bin/env python
"""Learned curvature drawn on Sudoku boards (adaptive-curvature HBFM, experiments/hbfm_rlf_ada_curv).

Validation boards of the cell's own data config (the hbfm_rlf_ada_curv cells train on 'hard',
30 clues). A given cell shows its digit; every empty cell is coloured by the curvature scale
lambda_i the sampler runs it at (`curvature_scale` at the sampler's initial state: origin
everywhere, the puzzle pinned at its clean boundary points) and annotated with the value.
lambda < 1 is flatter (a slower clock) than the base curvature K, lambda > 1 sharper; the colour
is ln lambda over the range of the drawn empty cells (the trained bound [1/b, b] is far wider
than what the model uses, so a fixed scale would hide the differences inside a board). Needs a
trained cell with curvature_bound > 1.

Example (compute node, GPU or CPU):
  python visualization/curvature_sudoku_boards.py \
    outputs/hbfm_rlf_ada_curv/cb-4_mode-exp_rate-3_k-0.5_ub-na_tm-6_fwd-horo_lr-3e-4_seed-1 \
    --boards 0 1 2 3 4 5
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, 'experiments/hbfm_rlf_ada_curv')
from common import DEVICE, load_cell  # noqa: E402
import dataloader  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('cell')
ap.add_argument('--ckpt', default='last.ckpt')
ap.add_argument('--boards', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5],
                help='indices into the validation split')
ap.add_argument('--out', default=None, help='default: experiments/hbfm_rlf_ada_curv/figures/'
                'curvature_sudoku_boards_<cell>.png')
args = ap.parse_args()
name = args.cell.rstrip('/').split('/')[-1]
out = args.out or f'experiments/hbfm_rlf_ada_curv/figures/curvature_sudoku_boards_{name}.png'

cfg, tokenizer, model = load_cell(args.cell, args.ckpt)
bound = cfg.algo.curvature_bound
assert bound > 1, 'fixed-curvature cell: every lambda is 1'
data = dataloader.get_dataset(cfg, tokenizer, mode='valid')
x = torch.as_tensor(data.select(args.boards)['input_ids']).to(DEVICE)
P = tokenizer.prompt_len
with torch.no_grad():
  xt = torch.zeros(*x.shape, model.backbone.embed_dim, dtype=torch.float64, device=x.device)
  xt[:, :P] = model._clean_state(x[:, :P])
  curvature = model.curvature_scale(xt)[:, P:]                        # [N, 89] solution positions
puzzle = x[:, 1:1 + tokenizer.seq_len]                                # aligned with the solution
cells = puzzle[0] != tokenizer.row_separator_id                       # the 81 cells among the 89
digits = puzzle[:, cells].view(-1, 9, 9).cpu().numpy()
lam = curvature[:, cells].view(-1, 9, 9).cpu().numpy()
empty = digits == tokenizer.empty_token_id

ncols = min(3, len(args.boards))
nrows = -(-len(args.boards) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.5 * nrows), squeeze=False)
fig.subplots_adjust(wspace=0.25, hspace=0.3)
cmap = plt.get_cmap('viridis').copy()
cmap.set_bad('white')
lo, hi = np.log(lam[empty].min()), np.log(lam[empty].max())
for ax, idx, d, l, e in zip(axes.ravel(), args.boards, digits, lam, empty):
  im = ax.imshow(np.ma.masked_where(~e, np.log(l)), cmap=cmap, vmin=lo, vmax=hi)
  for i in range(9):
    for j in range(9):
      if e[i, j]:
        dark = (np.log(l[i, j]) - lo) < 0.6 * (hi - lo)      # viridis: light ink on the dark end
        ax.text(j, i, f'{l[i, j]:.2f}', ha='center', va='center', fontsize=7,
                color='white' if dark else '0.15')
      else:
        ax.text(j, i, str(d[i, j]), ha='center', va='center', fontsize=12, weight='bold', color='0.1')
  for k in range(10):
    ax.axhline(k - 0.5, color='0.2', lw=2 if k % 3 == 0 else 0.5)
    ax.axvline(k - 0.5, color='0.2', lw=2 if k % 3 == 0 else 0.5)
  ax.set(xticks=[], yticks=[], xlim=(-0.5, 8.5), ylim=(8.5, -0.5))
  v = l[e]
  ax.set_title(f'board {idx}: {(~e).sum()} clues, $\\lambda$ {v.min():.2f}–{v.max():.2f} '
               f'(median {np.median(v):.2f})', fontsize=9)
for ax in axes.ravel()[len(args.boards):]:
  ax.axis('off')
ticks = [t for t in (0.25, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 4) if lo <= np.log(t) <= hi]
cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, ticks=np.log(ticks))
cb.ax.set_yticklabels([f'{t:g}' for t in ticks])
cb.set_label(r'curvature scale $\lambda_i = K_i / K$  (base K = 1; < 1: flatter, slower clock)'
             f'\ntrained bound [{1 / bound:g}, {bound:g}]')
fig.suptitle(f'{name} ({args.ckpt}, EMA): the curvature the sampler uses for each empty cell at t = 0',
             fontsize=10)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight')
v = lam[empty]
print(f'wrote {out}: {len(args.boards)} boards, {empty.sum()} empty cells, lambda in '
      f'[{v.min():.3f}, {v.max():.3f}], median {np.median(v):.3f}; given cells: lambda '
      f'[{lam[~empty].min():.3f}, {lam[~empty].max():.3f}]')
