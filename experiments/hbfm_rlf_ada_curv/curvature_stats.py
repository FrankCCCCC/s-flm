#!/usr/bin/env python
"""What did the curvature learn? (analysis only)

For one trained cell of the sweep, loads a checkpoint and evaluates the curvature scale lambda the
sampler would use on validation puzzles (`curvature_scale` at the t = 0 state: origin, prompt
pinned), then prints, as a markdown row per group of solution positions,

  * empty cells (the ones to solve), given cells (copied from the prompt), row separators:
    mean / std / min / max of lambda and the fraction within 5 % of a bound;
  * for the empty cells: the Spearman correlation of lambda with (a) the number of givens that
    constrain the cell (its row, column and box) and (b) the entropy of the model's own posterior
    for the cell at the t = 0 state -- does the curvature slow down what the model is unsure of?

Usage (compute node, GPU or CPU):
  python experiments/hbfm_rlf_ada_curv/curvature_stats.py <outputs/hbfm_rlf_ada_curv/cell> \
      [--ckpt last.ckpt] [--boards 512] [--no-ema]
"""
import argparse
import sys

import torch
from scipy import stats

sys.path.insert(0, 'experiments/hbfm_rlf_ada_curv')
from common import DEVICE, load_cell  # noqa: E402
import dataloader  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('cell')
ap.add_argument('--ckpt', default='last.ckpt')
ap.add_argument('--boards', type=int, default=512)
ap.add_argument('--no-ema', action='store_true')
args = ap.parse_args()

cfg, tokenizer, model = load_cell(args.cell, args.ckpt, ema=not args.no_ema)
bound = cfg.algo.curvature_bound

data = dataloader.get_dataset(cfg, tokenizer, mode='valid')
x = torch.as_tensor(data[:args.boards]['input_ids']).to(DEVICE)
P = tokenizer.prompt_len
with torch.no_grad():
  # The sampler's initial state: origin everywhere, the prompt pinned at the clean end.
  xt = torch.zeros(*x.shape, model.backbone.embed_dim, dtype=torch.float64, device=x.device)
  xt[:, :P] = model._clean_state(x[:, :P])
  curvature = model.curvature_scale(xt)[:, P:]                         # [N, 89] solution positions
  log_p = model.forward(xt=xt, sigma=torch.zeros(len(x), 1, device=x.device))[:, P:]
entropy = -(log_p.exp() * log_p).sum(-1)
correct = log_p.argmax(-1) == x[:, P:]

puzzle = x[:, 1:1 + tokenizer.seq_len]                                 # aligned with the solution
separator = puzzle == tokenizer.row_separator_id
empty = puzzle == tokenizer.empty_token_id
given = ~separator & ~empty
# Number of givens in the row, column and box of every cell.
grid = given[:, ~separator[0]].view(-1, 9, 9).float()
box = grid.view(-1, 3, 3, 3, 3).sum((2, 4)).repeat_interleave(3, 1).repeat_interleave(3, 2)
constraints = torch.zeros_like(given, dtype=torch.float)
constraints[:, ~separator[0]] = (grid.sum(2, keepdim=True) + grid.sum(1, keepdim=True)
                                 + box - 3 * grid).flatten(1)

print(f'{args.cell.rstrip("/").split("/")[-1]}  ({args.ckpt}, {"online" if args.no_ema else "EMA"} '
      f'weights, {len(x)} boards, bound {bound:g})')
print('| positions | mean | std | min | max | at a bound |')
print('|---|---|---|---|---|---|')
for name, mask in [('empty cells', empty), ('given cells', given), ('separators', separator)]:
  c = curvature[mask]
  at_bound = ((c > 0.95 * bound) | (c < 1 / (0.95 * bound))).float().mean().item()
  print(f'| {name} | {c.mean():.3f} | {c.std():.3f} | {c.min():.3f} | {c.max():.3f} | {at_bound:.1%} |')
c = curvature[empty].cpu().numpy()
print(f'\nempty cells: prior accuracy at t = 0: {correct[empty].float().mean():.3f}; '
      f'spread of lambda inside a board (mean std): '
      f'{torch.stack([curvature[i][empty[i]].std() for i in range(len(x))]).mean():.4f}')
for name, v in [('givens constraining the cell', constraints[empty]),
                ('posterior entropy of the cell at t = 0', entropy[empty])]:
  r = stats.spearmanr(c, v.cpu().numpy())
  print(f'Spearman(lambda, {name}) = {r.statistic:+.3f}  (p = {r.pvalue:.1e})')
