#!/usr/bin/env python
"""Seed-averaged analysis for the sudoku baseline sweep (sweep_baseline.py).

Reads outputs/hflm_curv_init_lr_sudoku/bl_d-{diff}_a-{algo}_rs{seed}/eval/results.json,
averages full-board solve rate over seeds {1,2,3}, and prints one row per algo with
easy/medium/hard columns (mean ± seed-std, %), matching the slide's "Recall the Former
Results" table so the numbers drop straight into slides/jul09_2026.

Usage:  python analyze_baseline.py [--md RESULTS_baseline.md]
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

OUT = '/share/thickstun/sychou/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku'
TAG = re.compile(r'bl_d-(?P<diff>\w+?)_a-(?P<algo>.+)_rs(?P<seed>\d+)')
ALGOS = ['ar', 'sfm', 'sfm_trunc', 'sfm_trunc_ada', 'eflm',
         'langflow_ada', 'langflow_full']
LABEL = {'ar': 'AR', 'sfm': 'S-FLM (naive)', 'sfm_trunc': 'S-FLM + trunc',
         'sfm_trunc_ada': 'S-FLM + trunc + adaptive', 'eflm': 'E-FLM (naive)',
         'langflow_ada': 'LangFlow + ada sched',
         'langflow_full': 'LangFlow + ada sched + SC'}
DIFFS = ['easy', 'medium', 'hard']


def load():
    acc = defaultdict(dict)  # acc[(algo,diff)] = {seed: accuracy}
    for f in glob.glob(f'{OUT}/bl_*/eval/results.json'):
        m = TAG.search(f)
        if not m:
            continue
        try:
            a = json.load(open(f))['accuracy']
        except Exception:
            continue
        d = m.groupdict()
        acc[(d['algo'], d['diff'])][int(d['seed'])] = a
    return acc


def cell(vals):
    vals = list(vals)
    if not vals:
        return '·'
    m = st.mean(vals) * 100
    s = (st.pstdev(vals) if len(vals) > 1 else 0.0) * 100
    return f'{m:.1f} ± {s:.1f} (n={len(vals)})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=None)
    args = ap.parse_args()
    acc = load()
    have = sum(len(v) for v in acc.values())
    lines = [f'coverage: {len(acc)}/{len(ALGOS)*len(DIFFS)} (algo,diff) groups | '
             f'{have}/{len(ALGOS)*len(DIFFS)*3} seed-runs', '',
             '| Model | ' + ' | '.join(DIFFS) + ' |',
             '|---|' + '---|' * len(DIFFS)]
    for algo in ALGOS:
        row = ' | '.join(cell(acc[(algo, d)].values()) for d in DIFFS)
        lines.append(f'| {LABEL[algo]} | {row} |')
    # flag incomplete cells
    inc = [(f'{a}/{d}', len(acc[(a, d)])) for a in ALGOS for d in DIFFS
           if len(acc[(a, d)]) < 3]
    if inc:
        lines += ['', f'incomplete (<3 seeds): {inc}']
    report = '\n'.join(lines)
    print(report)
    if args.md:
        with open(args.md, 'w') as f:
            f.write('# Sudoku Baselines — Seed-Averaged Results (jul09_2026 spec)\n\n'
                    'Full-board solve rate (%), mean ± seed-std over seeds {1,2,3}. '
                    'Eval: exact velocity, top_k_v=-1, 180 steps, greedy last '
                    '(LangFlow top_k=1 canonical).\n\n' + report + '\n')
        print(f'\nwrote {args.md}')


if __name__ == '__main__':
    main()
