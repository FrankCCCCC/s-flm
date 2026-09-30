#!/usr/bin/env python
"""Aggregate claude_test_hbfm_sudoku: board accuracy per (LR, noise schedule, seed) and
mean ± std over seeds, from outputs/claude_test_hbfm_sudoku/<cell>/eval/results.json
(written by mode=sudoku_eval). Prints the table and rewrites the results section of
RESULTS.md between the markers.

Usage:  python report.py
"""
import glob
import itertools
import json
import os
import re
import statistics

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm'
OUT = f'{REPO}/outputs/claude_test_hbfm_sudoku'
RESULTS = f'{REPO}/experiments/claude_test_hbfm_sudoku/RESULTS.md'
CURVS = ['-0.5', '-1.0']
LRS = ['3e-4', '5e-4', '1e-3']
NOISES = {'0': 'log-linear', '1': 'log-linear-adaptive'}
SEEDS = ['1', '2', '3']


def main():
    acc = {}
    for path in glob.glob(f'{OUT}/*/eval/results.json'):
        tag = os.path.basename(os.path.dirname(os.path.dirname(path)))
        m = re.match(r'lr-(.+)_ada-([01])_k(-[\d.]+)_seed-(\d+)(?:_rate-([\d.]+))?$', tag)
        if not m:
            continue
        with open(path) as f:
            r = json.load(f)
        rate = m.group(5) or '0.01'
        acc[(rate, m.group(3), m.group(1), m.group(2), m.group(4))] = (100.0 * r['accuracy'], r['num_correct'], r['num_total'])

    rows = [('0.01', k, lr, ada) for k, lr, ada in itertools.product(CURVS, LRS, NOISES)]
    rows += [('3', k, '3e-4', ada) for k, ada in itertools.product(CURVS, NOISES)]
    lines = ['| rate | K | LR | noise | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | mean ± std |',
             '|---|---|---|---|' + '---:|' * (len(SEEDS) + 1)]
    for rate, k, lr, ada in rows:
        vals = [acc.get((rate, k, lr, ada, s)) for s in SEEDS]
        cells = [f'{v[0]:.1f}%' if v else 'pending' for v in vals]
        done = [v[0] for v in vals if v]
        if len(done) >= 2:
            summary = f'**{statistics.mean(done):.1f} ± {statistics.stdev(done):.1f}** (n={len(done)})'
        elif len(done) == 1:
            summary = f'{done[0]:.1f} (n=1)'
        else:
            summary = 'pending'
        lines.append(f'| {rate} | {k} | {lr} | {NOISES[ada]} | ' + ' | '.join(cells) + f' | {summary} |')
    n_total = next(iter(acc.values()))[2] if acc else 0
    n_cells = len(rows) * len(SEEDS)
    table = '\n'.join(lines)
    print(f'{len(acc)}/{n_cells} cells evaluated ({n_total} boards each)\n' + table)

    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            doc = f.read()
        block = f'<!-- report:begin -->\n{len(acc)}/{n_cells} cells evaluated ({n_total} boards each).\n\n{table}\n<!-- report:end -->'
        new = re.sub(r'<!-- report:begin -->.*?<!-- report:end -->', block, doc, flags=re.S)
        with open(RESULTS, 'w') as f:
            f.write(new)


if __name__ == '__main__':
    main()
