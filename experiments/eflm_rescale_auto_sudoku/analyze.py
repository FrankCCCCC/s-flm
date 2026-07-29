#!/usr/bin/env python
"""Collector for the autonomous-clock E-FLM sweep.

Reads outputs/eflm_rescale_auto_sudoku/eflmra_*/eval/results.json and prints
(a) the tau_max pilot table (accuracy vs tau_max, per R, per loss weight) and
(b) the main-grid table (seed-averaged accuracy per arm x weight x R x LR),
alongside the log-linear control from experiments/eflm_rescale_sudoku.

Usage:  python analyze.py [--tau 0.5] [--md RESULTS_tables.md]
(RESULTS.md is hand-written prose + these tables; --md writes tables only, so
point it at a scratch file rather than clobbering the report.)
"""
import argparse
import glob
import json
import re
import statistics as st
from collections import defaultdict

REPO = '/share/thickstun/sychou/workspace/research/s-flm'
OUT = f'{REPO}/outputs/eflm_rescale_auto_sudoku'
BASE = ('/share/desa/nfs02/sc3379/workspace/research/s-flm-dev1/s-flm'
        '/outputs/eflm_rescale_sudoku')
TAG = re.compile(r'eflmra_(?P<clock>\w+)-(?P<arm>\w+)_r-(?P<R>[\d.]+)'
                 r'_tau-(?P<tau>[\d.na]+)_w-(?P<w>\w+)_lr-(?P<lr>[\w-]+)'
                 r'_d-(?P<diff>\w+)_rs(?P<seed>\d+)')
BASE_TAG = re.compile(r'eflmrs_(?P<arm>\w+)_r-(?P<R>[\d.]+)_lr-(?P<lr>[\w-]+)'
                      r'_d-(?P<diff>\w+)_rs(?P<seed>\d+)')
RHOS = ['5.0', '8', '16']
LRS = ['3e-4', '5e-4', '1e-3']


def load(pattern, tag_re, out_dir):
    runs = []
    for f in glob.glob(f'{out_dir}/{pattern}/eval/results.json'):
        m = tag_re.search(f)
        if not m:
            continue
        try:
            d = dict(m.groupdict(), acc=json.load(open(f))['accuracy'])
        except Exception:
            continue
        runs.append(d)
    return runs


def cell(vals):
    vals = [100 * v for v in vals]
    if not vals:
        return '·'
    if len(vals) == 1:
        return f'{vals[0]:.1f}'
    return f'{st.mean(vals):.1f}±{st.pstdev(vals):.1f}'


def group(runs, keyfn):
    g = defaultdict(list)
    for r in runs:
        g[keyfn(r)].append(r['acc'])
    return g


def pilot_table(runs):
    """accuracy vs tau_max, rows = arm x R x weight, cols = tau_max (+ log-linear)."""
    auto = [r for r in runs if r['clock'] == 'auto']
    ll = [r for r in runs if r['clock'] == 'll']
    taus = sorted({r['tau'] for r in auto}, key=float)
    g = group(auto, lambda r: (r['arm'], r['R'], r['w'], r['tau']))
    gl = group(ll, lambda r: (r['arm'], r['R'], r['w']))
    out = ['| arm | R | weight | ' + ' | '.join(f'tau={t}' for t in taus)
           + ' | log-linear |',
           '|---|---|---|' + '---|' * (len(taus) + 1)]
    for arm in ('naive', 'ada'):
        for R in RHOS:
            for w in ('ce', 'snr'):
                row = [cell(g.get((arm, R, w, t), [])) for t in taus]
                if not any(c != '\u00b7' for c in row) and (arm, R, w) not in gl:
                    continue
                out.append(f'| {arm} | {R} | {w} | ' + ' | '.join(row) + ' | '
                           + cell(gl.get((arm, R, w), [])) + ' |')
    return out


def main_table(runs, base, tau):
    """seed-averaged accuracy: rows = arm x weight x R, cols = LR.

    Only the autonomous cells at the main-grid tau_max; the log-linear cells of
    this checkout and the eflm_rescale_sudoku control get their own rows.
    """
    g = group([r for r in runs if r['clock'] == 'auto' and r['tau'] == tau],
              lambda r: (r['arm'], r['w'], r['R'], r['lr']))
    gll = group([r for r in runs if r['clock'] == 'll'],
                lambda r: (r['arm'], r['w'], r['R'], r['lr']))
    gb = group(base, lambda r: (r['arm'], r['R'], r['lr']))
    out = [f'auto cells at tau_max={tau}', '',
           '| arm | weight | R | ' + ' | '.join(f'lr={l}' for l in LRS)
           + ' | best |',
           '|---|---|---|' + '---|' * (len(LRS) + 1)]
    for arm in ('naive', 'ada'):
        for w in ('ce', 'snr', 'll+ce (here)', 'll+snr (here)',
                  'baseline(ll,ce)'):
            for R in RHOS:
                if w == 'baseline(ll,ce)':
                    vals = [gb.get((arm, R, l), []) for l in LRS]
                elif w.startswith('ll+'):
                    key = 'ce' if 'ce' in w.split('+')[1] else 'snr'
                    vals = [gll.get((arm, key, R, l), []) for l in LRS]
                else:
                    vals = [g.get((arm, w, R, l), []) for l in LRS]
                if not any(vals):
                    continue
                best = max((st.mean(v) for v in vals if v), default=None)
                out.append(f'| {arm} | {w} | {R} | '
                           + ' | '.join(cell(v) for v in vals)
                           + f' | {100 * best:.1f} |')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=None,
                    help='write the tables to this file (NOT RESULTS.md, '
                         'which carries the hand-written analysis)')
    ap.add_argument('--tau', default='1.0', help='tau_max of the main grid')
    args = ap.parse_args()

    runs = load('eflmra_*', TAG, OUT)
    base = [r for r in load('eflmrs_*', BASE_TAG, BASE) if r['R'] in RHOS]
    pilot = [r for r in runs if r['lr'] == '1e-3' and r['seed'] == '1']

    lines = [f'runs collected: {len(runs)} '
             f'(pilot rows {len(pilot)}), log-linear control: {len(base)}',
             '', '## tau_max pilot (lr 1e-3, seed 1, % solved)', '']
    lines += pilot_table(pilot)
    lines += ['', '## main grid (mean±std over seeds, % solved)', '']
    lines += main_table(runs, base, args.tau)
    text = '\n'.join(lines)
    print(text)
    if args.md:
        with open(args.md, 'w') as f:
            f.write('# eflm_rescale_auto_sudoku — results\n\n' + text + '\n')


if __name__ == '__main__':
    main()
