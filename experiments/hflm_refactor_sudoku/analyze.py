#!/usr/bin/env python
"""Collect hflm_refactor_sudoku results and compare against the old sweep.

Scans outputs/hflm_refactor_sudoku/*/eval_{topk1,topkall}/results.json, writes
all_results.csv, and prints markdown tables:
  1. no-regression: new topk1 (== old argmax sampler) vs old sweep seed-1 /
     mean±std for every (diff, K, init, lr) the old sweep covered (medium/hard)
  2. curvature x LR grids per difficulty (topk1 | topkall)
  3. init x K per difficulty
  4. topkall - topk1 (expected-velocity vs argmax) delta summary

Safe to run on partial results (cells without both evals are skipped).
Usage: python experiments/hflm_refactor_sudoku/analyze.py
"""
import csv
import json
import os
import re
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'outputs/hflm_refactor_sudoku')
EXP = os.path.join(REPO, 'experiments/hflm_refactor_sudoku')
OLD = os.path.join(REPO, 'experiments/hflm_curv_init_lr_sudoku/all_results.csv')
TAG = re.compile(r'd-(\w+)_k(-[\d.]+)_i-([\w.]+)_lr([\de.-]+)_rs1')


def load_new():
  rows = []
  for name in sorted(os.listdir(OUT)) if os.path.isdir(OUT) else []:
    m = TAG.fullmatch(name)
    if not m:
      continue
    accs = {}
    for ev in ('topk1', 'topkall'):
      p = os.path.join(OUT, name, f'eval_{ev}', 'results.json')
      if os.path.exists(p):
        accs[ev] = json.load(open(p))['accuracy']
    if accs:
      rows.append(dict(diff=m[1], k=m[2], init=m[3], lr=m[4],
                       topk1=accs.get('topk1'), topkall=accs.get('topkall')))
  return rows


def load_old():
  ref = {}
  for r in csv.DictReader(open(OLD)):
    ref.setdefault((r['diff'], r['k'], r['init'], r['lr']), {})[r['seed']] = \
      float(r['accuracy'])
  return ref


def fmt(x, pct=True):
  return '—' if x is None else f'{100 * x:.2f}' if pct else f'{x:.4f}'


def main():
  rows = load_new()
  ref = load_old()
  n_full = sum(1 for r in rows if r['topk1'] is not None and r['topkall'] is not None)
  print(f'{len(rows)} cells with >=1 eval, {n_full} with both evals\n')

  with open(os.path.join(EXP, 'all_results.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, ['diff', 'k', 'init', 'lr', 'topk1', 'topkall'])
    w.writeheader()
    w.writerows(rows)

  # 1. no-regression vs old sweep (medium/hard only; old sweep had no easy)
  print('## No-regression: new topk1 (argmax-equivalent) vs old sweep\n')
  print('| diff | K | init | lr | new topk1 | old seed1 | old mean±std | new−seed1 |')
  print('|---|---|---|---|---|---|---|---|')
  deltas = []
  for r in rows:
    key = (r['diff'], r['k'], r['init'], r['lr'])
    if key not in ref or r['topk1'] is None:
      continue
    seeds = ref[key]
    mean = statistics.mean(seeds.values())
    std = statistics.stdev(seeds.values()) if len(seeds) > 1 else 0.0
    d = r['topk1'] - seeds['1'] if '1' in seeds else None
    if d is not None:
      deltas.append(d)
    print(f"| {r['diff']} | {r['k']} | {r['init']} | {r['lr']} | {fmt(r['topk1'])} "
          f"| {fmt(seeds.get('1'))} | {100 * mean:.2f}±{100 * std:.2f} | "
          f"{'—' if d is None else f'{100 * d:+.2f}'} |")
  if deltas:
    print(f'\nnew−old(seed1) over {len(deltas)} matched cells: '
          f'mean {100 * statistics.mean(deltas):+.2f}pt, '
          f'median {100 * statistics.median(deltas):+.2f}pt, '
          f'min {100 * min(deltas):+.2f}, max {100 * max(deltas):+.2f}\n')

  # 2. curvature x LR grid (init c0.01)
  for diff in ('easy', 'medium', 'hard'):
    sel = {(r['k'], r['lr']): r for r in rows
           if r['diff'] == diff and r['init'] == 'c0.01'}
    if not sel:
      continue
    ks = sorted({k for k, _ in sel}, key=float, reverse=True)
    lrs = ['3e-4', '5e-4', '1e-3']
    print(f'## {diff} — curvature x LR (init c0.01; topk1 | topkall, acc %)\n')
    print('| K | ' + ' | '.join(lrs) + ' |')
    print('|---|' + '---|' * len(lrs))
    for k in ks:
      cells = [sel.get((k, lr)) for lr in lrs]
      print(f'| {k} | ' + ' | '.join(
        '—' if c is None else f"{fmt(c['topk1'])} \\| {fmt(c['topkall'])}"
        for c in cells) + ' |')
    print()

  # 3. init x K at per-difficulty best LR
  best_lr = {'easy': '3e-4', 'medium': '5e-4', 'hard': '3e-4'}
  for diff in ('easy', 'medium', 'hard'):
    sel = {(r['init'], r['k']): r for r in rows
           if r['diff'] == diff and r['lr'] == best_lr[diff]}
    inits = [i for i in ('c0.01', 'c0.02', 'c0.04', 'random')
             if any(i == a for a, _ in sel)]
    if not inits:
      continue
    ks = sorted({k for _, k in sel}, key=float, reverse=True)
    print(f'## {diff} — init x K @ lr {best_lr[diff]} (topk1 | topkall, acc %)\n')
    print('| init | ' + ' | '.join(ks) + ' |')
    print('|---|' + '---|' * len(ks))
    for i in inits:
      cells = [sel.get((i, k)) for k in ks]
      print(f'| {i} | ' + ' | '.join(
        '—' if c is None else f"{fmt(c['topk1'])} \\| {fmt(c['topkall'])}"
        for c in cells) + ' |')
    print()

  # 4. sampler delta
  ds = [(r['topkall'] - r['topk1'], r) for r in rows
        if r['topk1'] is not None and r['topkall'] is not None]
  if ds:
    print('## Expected-velocity vs argmax (topkall − topk1) per difficulty\n')
    for diff in ('easy', 'medium', 'hard'):
      sub = [d for d, r in ds if r['diff'] == diff]
      if sub:
        print(f'- {diff}: mean {100 * statistics.mean(sub):+.2f}pt over {len(sub)} '
              f'cells (min {100 * min(sub):+.2f}, max {100 * max(sub):+.2f})')


if __name__ == '__main__':
  main()
