#!/usr/bin/env python
"""analyze.py — aggregate the phase-2 grid into RESULTS.md.

Walks outputs/selfcond_sudoku/sc_d-{diff}_a-{algo}_lr{lr}_sc-{on|off}_rs{seed}/
eval/results.json (incl. the 36 reuse symlinks), averages solve-rate over
seeds, and rewrites everything in RESULTS.md below the AUTO marker: one table
per difficulty (rows algo x lr; columns SC off / SC on / delta), a
best-over-lr summary, and the raw per-cell values. Manual prose above the
marker is preserved. Safe to run on a partial grid (cells with n<3 seeds are
flagged; missing cells shown as '-').

Usage: python experiments/selfcond_sudoku/analyze.py [--stdout]
"""
import argparse
import itertools
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(f'{HERE}/../../outputs/selfcond_sudoku')
RESULTS_MD = f'{HERE}/RESULTS.md'
MARKER = '<!-- analyze.py: AUTO-GENERATED BELOW; edits below are overwritten -->'

ALGOS = ['sfm_trunc_ada', 'eflm', 'hflm']
DIFFICULTIES = ['medium', 'hard']
LRS = ['3e-4', '5e-4', '1e-3']
SEEDS = ['1', '2', '3']
LABEL = {'sfm_trunc_ada': 'S-FLM+trunc+ada', 'eflm': 'E-FLM',
         'hflm': 'H-FLM (K=-0.5, c0.01)'}


def load():
    """{(algo, diff, lr, sc): [acc% per found seed]}"""
    accs = {}
    for algo, diff, lr, sc, seed in itertools.product(
            ALGOS, DIFFICULTIES, LRS, ['on', 'off'], SEEDS):
        f = (f'{OUT}/sc_d-{diff}_a-{algo}_lr{lr}_sc-{sc}_rs{seed}'
             '/eval/results.json')
        if os.path.exists(f):
            with open(f) as fh:
                accs.setdefault((algo, diff, lr, sc), []).append(
                    json.load(fh)['accuracy'] * 100)
    return accs


def fmt(vals):
    if not vals:
        return '-'
    m = statistics.mean(vals)
    s = f'{m:.1f} ± {statistics.stdev(vals):.1f}' if len(vals) > 1 else f'{m:.1f}'
    return s if len(vals) == 3 else f'{s} (n={len(vals)})'


def render(accs):
    lines = [MARKER, '', '## Phase 2 — full grid (mean ± seed-std, solve rate %)', '']
    total = sum(len(v) for v in accs.values())
    lines += [f'_{total}/108 cells loaded from `outputs/selfcond_sudoku`._', '']
    for diff in DIFFICULTIES:
        lines += [f'### {diff.capitalize()}', '',
                  '| Model | LR | SC off | SC on | Δ (on−off) |',
                  '|---|---|---|---|---|']
        for algo in ALGOS:
            for lr in LRS:
                off = accs.get((algo, diff, lr, 'off'), [])
                on = accs.get((algo, diff, lr, 'on'), [])
                d = (f'{statistics.mean(on) - statistics.mean(off):+.1f}'
                     if off and on else '-')
                lines.append(f'| {LABEL[algo]} | {lr} | {fmt(off)} | {fmt(on)} | {d} |')
        lines.append('')
    lines += ['### Best over LR per model', '',
              '| Model | difficulty | SC off (best lr) | SC on (best lr) | Δ |',
              '|---|---|---|---|---|']
    for algo, diff in itertools.product(ALGOS, DIFFICULTIES):
        best = {}
        for sc in ('off', 'on'):
            cands = [(statistics.mean(a), lr) for lr in LRS
                     if (a := accs.get((algo, diff, lr, sc), []))]
            best[sc] = max(cands) if cands else None
        d = (f'{best["on"][0] - best["off"][0]:+.1f}'
             if best['on'] and best['off'] else '-')
        cell = lambda b: f'{b[0]:.1f} @ {b[1]}' if b else '-'
        lines.append(f'| {LABEL[algo]} | {diff} | {cell(best["off"])} '
                     f'| {cell(best["on"])} | {d} |')
    lines += ['', '<details><summary>Per-seed raw values</summary>', '']
    for key in sorted(accs):
        algo, diff, lr, sc = key
        vals = ', '.join(f'{v:.2f}' for v in accs[key])
        lines.append(f'- `{diff}/{algo}/lr{lr}/sc-{sc}`: {vals}')
    lines += ['', '</details>', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stdout', action='store_true',
                    help='print instead of rewriting RESULTS.md')
    args = ap.parse_args()
    body = render(load())
    if args.stdout:
        print(body)
        return
    with open(RESULTS_MD) as f:
        head = f.read().split(MARKER)[0].rstrip() + '\n\n'
    with open(RESULTS_MD, 'w') as f:
        f.write(head + body)
    print(f'wrote {RESULTS_MD}')


if __name__ == '__main__':
    main()
