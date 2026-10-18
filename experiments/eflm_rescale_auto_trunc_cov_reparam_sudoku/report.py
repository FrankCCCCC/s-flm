#!/usr/bin/env python
"""Solve-rate table for eflm_rescale_auto_trunc_cov_reparam_sudoku: one row per (noise, R, LR), seeds 1-3,
mean +- sd. Reads outputs/<project>/<tag>/eval/results.json; prints markdown for RESULTS.md."""
import json
import os
import statistics

import sweep

rows = ['| noise | R | LR | seed 1 | seed 2 | seed 3 | mean ± sd |', '|---|---|---|---|---|---|---|']
for noise in sweep.NOISES:
  for rho in sweep.RHOS:
    for lr in sweep.LRS:
      accs = []
      for seed in sweep.SEEDS:
        p = f'{sweep.OUT}/{sweep.tag_of(noise, rho, lr, seed)}/eval/results.json'
        accs.append(100 * json.load(open(p))['accuracy'] if os.path.exists(p) else None)
      done = [a for a in accs if a is not None]
      if len(done) > 1:
        summary = f'{statistics.mean(done):.1f} ± {statistics.stdev(done):.1f} (n={len(done)})'
      else:
        summary = f'{done[0]:.1f} (n=1)' if done else '—'
      cells = ' | '.join('—' if a is None else f'{a:.1f}' for a in accs)
      rows.append(f'| {noise} | {rho} | {lr} | {cells} | {summary} |')
print('\n'.join(rows))
