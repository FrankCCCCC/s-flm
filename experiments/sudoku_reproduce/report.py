#!/usr/bin/env python
"""report.py -- collect outputs/sudoku_reproduce/*/eval/results.json into a table (printed; pasted into results.md).

Per model and difficulty: board accuracy of every seed and the mean +- std over seeds, next to the paper
(Deschenaux & Gulcehre, arXiv:2605.11125, Table 1). Run from the repo root.
"""
import json
import os
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f'{REPO}/outputs/sudoku_reproduce'
MODELS = {'mdlm': 'MDLM', 'duo': 'DUO', 'sfm': 'S-FLM', 'sfm_trunc': 'S-FLM + trunc',
          'sfm_trunc_ada': 'S-FLM + trunc + adaptive', 'eflm_auto_trunc': 'E-FLM + auto + trunc',
          'eflm_auto_trunc_sc': 'E-FLM + auto + trunc + self-cond'}
PAPER = {'mdlm': (92.0, 77.1, 30.2), 'duo': (96.3, 84.7, 58.4), 'sfm': (81.5, 50.6, 14.0),
         'sfm_trunc': (94.0, 77.6, 43.2), 'sfm_trunc_ada': (94.8, 85.2, 45.0)}
DIFFICULTIES = ['easy', 'medium', 'hard']
SEEDS = [1, 2, 3]


def acc(model, diff, seed):
  f = f'{OUT}/sudoku_reproduce_m-{model}_d-{diff}_s-{seed}/eval/results.json'
  return 100 * json.load(open(f))['accuracy'] if os.path.exists(f) else None


def fmt(vals):
  if not vals:
    return '—'
  if len(vals) == 1:
    return f'{vals[0]:.1f} (n=1)'
  return f'{statistics.mean(vals):.1f} ± {statistics.stdev(vals):.1f}' + ('' if len(vals) == 3 else f' (n={len(vals)})')


def main():
  lines = ['| model | ' + ' | '.join(f'{d} (ours)' + ' | ' + f'{d} (paper)' for d in DIFFICULTIES) + ' |',
           '|---|' + '---:|---:|' * len(DIFFICULTIES)]
  for m, name in MODELS.items():
    cells = []
    for i, d in enumerate(DIFFICULTIES):
      cells.append(fmt([a for a in (acc(m, d, s) for s in SEEDS) if a is not None]))
      cells.append(f'{PAPER[m][i]:.1f}' if m in PAPER else '—')
    lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
  lines += ['', '| model | difficulty | seed 1 | seed 2 | seed 3 |', '|---|---|---:|---:|---:|']
  for m, name in MODELS.items():
    for d in DIFFICULTIES:
      vals = [acc(m, d, s) for s in SEEDS]
      lines.append(f'| {name} | {d} | ' + ' | '.join('—' if a is None else f'{a:.2f}' for a in vals) + ' |')
  done = sum(acc(m, d, s) is not None for m in MODELS for d in DIFFICULTIES for s in SEEDS)
  print(f'{done}/{len(MODELS) * len(DIFFICULTIES) * len(SEEDS)} cells evaluated\n')
  print('\n'.join(lines))


if __name__ == '__main__':
  main()
