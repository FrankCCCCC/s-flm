#!/usr/bin/env python
"""Collector for the autonomous-clock E-FLM sweep.

Reads outputs/eflm_rescale_auto_sudoku/eflmra_*/eval/results.json and prints,
all seed-averaged (mean+-sd):
  1. tau_max sweep      — accuracy vs tau_max, per arm x R x loss weight
  2. LR axis per tau_max — arm x weight x R vs LR, one table per horizon
  3. at the predicted-optimal tau*(R) — the same LR table with each row placed
     at the horizon nearest tau*(R) = log(1 + C/R), plus the empirically best
     horizon for comparison
  4. controls — this checkout's log-linear cells + the eflm_rescale_sudoku
     baseline (both tau-independent)

Usage:  python analyze.py [--tau 0.5] [--md RESULTS_tables.md]
(RESULTS.md is hand-written prose + these tables; --md writes tables only, so
point it at a scratch file rather than clobbering the report.)
"""
import argparse
import glob
import json
import math
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
C_CODEBOOK = math.sqrt(2 * math.log(2 * (12 - 1) / 0.1))  # V=12, delta=0.1
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


def tau_table(runs):
    """accuracy vs tau_max, rows = arm x R x weight, cols = tau_max (+ log-linear).

    Seed-averaged: mean+-sd(n) over every seed present at that cell, so the
    3-seed numbers quoted in RESULTS.md (e.g. tau=0.25 / naive / snr / R=5 =
    51.1+-3.5) appear here and not only their seed-1 member.
    """
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


def tau_star(R):
    """Predicted decode horizon: tau*(R) = -log b*(R), b* = 1/(1 + C/R),
    C = sqrt(2 log(2(V-1)/delta)) (random-codebook bound, V=12, delta=0.1).
    The flow has to run past b*; everything beyond it re-integrates a decision
    the model already made."""
    return math.log(1 + C_CODEBOOK / float(R))


def nearest_tau(R, taus):
    return min(taus, key=lambda t: abs(float(t) - tau_star(R)))


def _lr_row(arm, w, R, get, prefix=''):
    vals = [get(arm, w, R, l) for l in LRS]
    if not any(vals):
        return None
    # scale before averaging, exactly as cell() does, so `best` can never
    # disagree with the column it was taken from at the rounding boundary
    best = max(st.mean([100 * x for x in v]) for v in vals if v)
    return (f'| {arm} | {w} | {R} | {prefix}'
            + ' | '.join(cell(v) for v in vals)
            + f' | {best:.1f} |')


def lr_table(runs, tau):
    """rows = arm x weight x R, cols = LR, at one tau_max (autonomous only)."""
    g = group([r for r in runs if r['clock'] == 'auto' and r['tau'] == tau],
              lambda r: (r['arm'], r['w'], r['R'], r['lr']))
    matched = [R for R in RHOS if nearest_tau(R, ['0.1', '0.25', '0.5']) == tau]
    note = (f'  (predicted optimum for R = {", ".join(matched)})'
            if matched else '')
    out = [f'### tau_max = {tau}{note}', '',
           '| arm | weight | R | ' + ' | '.join(f'lr={l}' for l in LRS)
           + ' | best |',
           '|---|---|---|' + '---|' * (len(LRS) + 1)]
    for arm in ('naive', 'ada'):
        for w in ('ce', 'snr'):
            for R in RHOS:
                row = _lr_row(arm, w, R,
                              lambda a, x, r, l: g.get((a, x, r, l), []))
                if row:
                    out.append(row)
    return out


def opt_tau_table(runs, taus):
    """One row per (arm, weight, R) at the tau_max nearest the predicted
    tau*(R) — the table you would read off to pick a configuration."""
    g = group([r for r in runs if r['clock'] == 'auto'],
              lambda r: (r['arm'], r['w'], r['R'], r['tau'], r['lr']))
    emp = {}  # empirical best tau per (arm, w, R), lr 1e-3
    for (a, w, R, t, lr), v in g.items():
        if lr == '1e-3' and st.mean(v) > emp.get((a, w, R), (0, None))[0]:
            emp[(a, w, R)] = (st.mean(v), t)
    out = ['| arm | weight | R | tau*(R) pred | tau used | '
           + ' | '.join(f'lr={l}' for l in LRS)
           + ' | best | tau emp. best |',
           '|---|---|---|---|---|' + '---|' * (len(LRS) + 2)]
    for arm in ('naive', 'ada'):
        for w in ('ce', 'snr'):
            for R in RHOS:
                t = nearest_tau(R, taus)
                row = _lr_row(arm, w, R,
                              lambda a, x, r, l: g.get((a, x, r, t, l), []),
                              prefix=f'{tau_star(R):.3f} | {t} | ')
                if row:
                    out.append(row + f' {emp.get((arm, w, R), (0, "·"))[1]} |')
    return out


def controls_table(runs, base):
    """tau-independent reference rows: this checkout's log-linear cells and the
    eflm_rescale_sudoku control."""
    gll = group([r for r in runs if r['clock'] == 'll'],
                lambda r: (r['arm'], r['w'], r['R'], r['lr']))
    gb = group(base, lambda r: (r['arm'], r['R'], r['lr']))
    out = ['| arm | weight | R | ' + ' | '.join(f'lr={l}' for l in LRS)
           + ' | best |',
           '|---|---|---|' + '---|' * (len(LRS) + 1)]
    for arm in ('naive', 'ada'):
        for label, get in (
                ('log-linear+ce (here)',
                 lambda a, x, r, l: gll.get((a, 'ce', r, l), [])),
                ('log-linear+snr (here)',
                 lambda a, x, r, l: gll.get((a, 'snr', r, l), [])),
                ('baseline(log-linear,ce)',
                 lambda a, x, r, l: gb.get((a, r, l), []))):
            for R in RHOS:
                row = _lr_row(arm, label, R, get)
                if row:
                    out.append(row)
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
    tau_runs = [r for r in runs if r['lr'] == '1e-3']

    taus = sorted({r['tau'] for r in runs if r['clock'] == 'auto'}, key=float)
    lr_taus = [t for t in taus if any(
        r['tau'] == t and r['lr'] != '1e-3' for r in runs)] or [args.tau]

    lines = [f'runs collected: {len(runs)} '
             f'(tau-sweep rows {len(tau_runs)}), log-linear control: {len(base)}',
             '', '## tau_max sweep (lr 1e-3, mean+-sd over seeds, % solved)', '']
    lines += tau_table(tau_runs)
    lines += ['', '## LR axis per tau_max (mean+-sd over seeds, % solved)', '',
              'The LR axis was only swept at tau_max = '
              + ', '.join(lr_taus) + '; the other horizons ran at lr 1e-3 only '
              '(shown as `·`).', '']
    for t in [t for t in taus if float(t) <= 0.5]:
        lines += lr_table(runs, t) + ['']
    lines += ['## at the predicted-optimal tau*(R)', '',
              'tau*(R) = log(1 + C/R), C = '
              f'{C_CODEBOOK:.3f} (V=12, delta=0.1); `tau used` is the nearest '
              'swept horizon, `tau emp. best` the empirically best one at '
              'lr 1e-3.', '']
    lines += opt_tau_table(runs, [t for t in taus if float(t) <= 0.5])
    lines += ['', '## controls (tau-independent)', '']
    lines += controls_table(runs, base)
    text = '\n'.join(lines)
    print(text)
    if args.md:
        with open(args.md, 'w') as f:
            f.write('# eflm_rescale_auto_sudoku — results\n\n' + text + '\n')


if __name__ == '__main__':
    main()
