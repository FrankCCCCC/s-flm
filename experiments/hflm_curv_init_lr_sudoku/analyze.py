#!/usr/bin/env python
"""Seed-averaged analysis for hflm_curv_init_lr_sudoku (jul02 slides spec).

Reads outputs/hflm_curv_init_lr_sudoku/d-{diff}_k{K}_i-{init}_lr{lr}_rs{seed}/eval/results.json,
aggregates board accuracy over seeds {1,2,3}, and emits per-(difficulty) tables:
  - per-geometry-curvature LR means  (avg over the 7 init cells, seed-averaged)
  - per-K best cell (seed-averaged) with its init/lr and seed std
  - curvature-optimum vs K=-1.0 baseline, with seed error bars

Usage:  python analyze.py [--md RESULTS.md]
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

OUT = '/share/thickstun/sychou/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku'
TAG = re.compile(r'd-(?P<diff>\w+)_k(?P<k>-[\d.]+)_i-(?P<init>[^_]+)_lr(?P<lr>[\de.-]+)_rs(?P<seed>\d+)')
KS = ['-0.25', '-0.3', '-0.5', '-0.7', '-1.0', '-1.5']
LRS = ['1e-4', '3e-4', '5e-4', '1e-3']
INITS = ['ngpt', 'random', 'c0.01', 'c0.02', 'c0.04', 'c0.06', 'c0.08']
DIFFS = ['medium', 'hard']


def load():
    # acc[(diff,k,init,lr)] = {seed: accuracy}
    acc = defaultdict(dict)
    for f in glob.glob(f'{OUT}/*/eval/results.json'):
        m = TAG.search(f)
        if not m:
            continue
        try:
            a = json.load(open(f))['accuracy']
        except Exception:
            continue
        d = m.groupdict()
        acc[(d['diff'], d['k'], d['init'], d['lr'])][int(d['seed'])] = a
    return acc


def mean_std(vals):
    vals = list(vals)
    if not vals:
        return None, None, 0
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0), len(vals)


def coverage(acc):
    have = sum(len(v) for v in acc.values())
    print(f'cells with >=1 seed: {len(acc)} / {len(DIFFS)*len(KS)*len(INITS)*len(LRS)} '
          f'| total seed-runs: {have} / {len(DIFFS)*len(KS)*len(INITS)*len(LRS)*3}')
    # flag incomplete (fewer than 3 seeds)
    incomplete = [(k, len(v)) for k, v in acc.items() if len(v) < 3]
    if incomplete:
        print(f'  cells with <3 seeds: {len(incomplete)} '
              f'(e.g. {incomplete[:3]})')


def lr_means_table(acc, diff):
    # per-K row: mean over the 7 init cells of the seed-mean accuracy, per LR
    lines = [f'### {diff} — per-curvature LR means (avg over 7 inits, seed-averaged)', '',
             '| K | ' + ' | '.join(LRS) + ' | best LR |', '|---|' + '---|' * (len(LRS) + 1)]
    for k in KS:
        cells = []
        for lr in LRS:
            per_init = [mean_std(acc[(diff, k, init, lr)].values())[0]
                        for init in INITS]
            per_init = [x for x in per_init if x is not None]
            cells.append(st.mean(per_init) if per_init else None)
        best_lr = LRS[max(range(len(LRS)), key=lambda i: cells[i] if cells[i] is not None else -1)]
        row = ' | '.join(f'{c*100:.1f}' if c is not None else '·' for c in cells)
        lines.append(f'| {k} | {row} | **{best_lr}** |')
    return '\n'.join(lines)


def best_cell_table(acc, diff):
    lines = [f'### {diff} — best cell per curvature (seed-averaged, ± seed std)', '',
             '| K | best cell | acc | seed std | n |', '|---|---|--:|--:|--:|']
    for k in KS:
        best = None
        for init in INITS:
            for lr in LRS:
                m, s, n = mean_std(acc[(diff, k, init, lr)].values())
                if m is None:
                    continue
                if best is None or m > best[0]:
                    best = (m, s, n, init, lr)
        if best:
            m, s, n, init, lr = best
            lines.append(f'| {k} | {init} @ {lr} | {m*100:.1f} | ±{s*100:.1f} | {n} |')
        else:
            lines.append(f'| {k} | — | · | · | 0 |')
    return '\n'.join(lines)


def optimum_vs_baseline(acc, diff):
    # best-cell curve vs K=-1.0's best cell, with seed error bars
    def best_of(k):
        b = None
        for init in INITS:
            for lr in LRS:
                m, s, n = mean_std(acc[(diff, k, init, lr)].values())
                if m is not None and (b is None or m > b[0]):
                    b = (m, s, n, init, lr)
        return b
    base = best_of('-1.0')
    lines = [f'**{diff}:** curvature best-cells vs K=-1.0 baseline '
             f'({base[3]}@{base[4]} = {base[0]*100:.1f}±{base[1]*100:.1f}%):']
    for k in KS:
        if k == '-1.0':
            continue
        b = best_of(k)
        if not b:
            continue
        delta = (b[0] - base[0]) * 100
        # pooled seed std as rough error bar on the delta
        err = (b[1]**2 + base[1]**2) ** 0.5 * 100
        sig = '**' if abs(delta) > 2 * err and err > 0 else ''
        lines.append(f'  - K={k}: {b[0]*100:.1f}±{b[1]*100:.1f}%  '
                     f'({sig}{delta:+.1f} vs baseline, ±{err:.1f}{sig})')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=None)
    args = ap.parse_args()
    acc = load()
    coverage(acc)
    blocks = []
    for diff in DIFFS:
        blocks += [lr_means_table(acc, diff), '', best_cell_table(acc, diff), '',
                   optimum_vs_baseline(acc, diff), '', '---', '']
    report = '\n'.join(blocks)
    print('\n' + report)
    if args.md:
        with open(args.md, 'w') as f:
            f.write('# hflm_curv_init_lr_sudoku — Seed-Averaged Results\n\n'
                    'H-FLM Gaussian-curvature × init × LR on Sudoku {medium, hard}, '
                    '3 seeds averaged. Eval: exact velocity, top_k_v=-1, 180 steps, '
                    'greedy last (slides/jul02_2026 spec).\n\n' + report)
        print(f'\nwrote {args.md}')


if __name__ == '__main__':
    main()
