#!/usr/bin/env python
"""Collector for the truncated autonomous-clock E-FLM hyperparameter sweep.

Reads outputs/eflm_rescale_auto_trunc_tinystories_256/eflmratr_*/eval/
{ppl.json, samples_genppl.json} and prints:
  1. GenPPL over LR x R at m = 1.0            (stage 1: the LR ladder)
  2. GenPPL over m x R at each LR present     (stage 2: the truncation axis)
  3. flat table (valid PPL, GenPPL, entropy), best cells first

GenPPL is gpt2-large retokenized generative perplexity, lower better, and is
only meaningful read together with entropy (< 3.0 => degenerate collapse).
Valid PPL is a flow bound and is NOT comparable across truncations.

Usage:  python analyze.py [--md tables.md]
"""
import argparse
import glob
import json
import math
import os
import re

REPO = '/share/thickstun/sychou/workspace/research/s-flm'
OUT = f'{REPO}/outputs/eflm_rescale_auto_trunc_tinystories_256'
TAG = re.compile(r'eflmratr_lr-(?P<lr>[\w.-]+)_r-(?P<R>[\d.]+)_m-(?P<m>[\d.]+)'
                 r'(?:_w-(?P<w>\w+))?$')   # no _w- suffix => plain CE
LRS = ['3e-4', '1e-3', '5e-3']
ENT_MIN = 3.0        # anti-collapse gate (setup.md: "without collapsing")
RHOS = ['0.05', '0.1', '0.5', '1', '5', '8', '16', '28']
C = math.sqrt(2 * math.log(2 * (50257 - 1) / 0.1))  # 5.2575


def tau_star(R):
    return math.log1p(C / float(R))


def load():
    runs = []
    for ed in sorted(glob.glob(f'{OUT}/*/eval')):
        m = TAG.search(os.path.basename(os.path.dirname(ed)))
        if not m:
            continue
        r = dict(m.groupdict())
        r['w'] = r['w'] or 'ce'
        try:
            g = json.load(open(f'{ed}/samples_genppl.json'))
        except Exception:
            continue
        r['gen'] = g.get('gen_ppl_first_chunk_retok')
        r['ent'] = g.get('entropy')
        try:
            r['ppl'] = json.load(open(f'{ed}/ppl.json')).get('val/ppl')
        except Exception:
            r['ppl'] = None
        r['tau'] = float(r['m']) * tau_star(r['R'])
        runs.append(r)
    return runs


def cell(runs, **kw):
    hit = [r for r in runs if all(r[k] == v for k, v in kw.items())]
    return hit[0] if hit else None


def pivot(runs, rows, row_key, cols, col_key, fixed, title, lines):
    """rows x cols table of GenPPL (entropy in parentheses)."""
    sub = [r for r in runs if all(r[k] == v for k, v in fixed.items())]
    if not sub:
        return
    cols = [c for c in cols if any(r[col_key] == c for r in sub)]
    rows = [c for c in rows if any(r[row_key] == c for r in sub)]
    lines += ['', f'### {title}', '',
              f'| {row_key} \\ {col_key} | ' + ' | '.join(cols) + ' |',
              '|---' * (len(cols) + 1) + '|']
    # "Best" must respect the anti-collapse gate: a degenerate cell has the
    # LOWEST GenPPL in the sweep (1.09, a single repeated token), so ranking
    # without the entropy filter marks a collapse as the winner.
    best = min((r['gen'] for r in sub if r['gen'] is not None
                and r['ent'] is not None and r['ent'] >= ENT_MIN), default=None)
    for rv in rows:
        out = []
        for cv in cols:
            c = cell(sub, **{row_key: rv, col_key: cv})
            if c is None or c['gen'] is None:
                out.append('·')
                continue
            s = f"{c['gen']:.2f}"
            if c['gen'] == best:
                s = f'**{s}**'
            flag = ' ⚠' if c['ent'] is not None and c['ent'] < ENT_MIN else ''
            out.append(f"{s} ({c['ent']:.2f}){flag}")
        lines.append(f'| {rv} | ' + ' | '.join(out) + ' |')


def ppl_pivot(runs, lines):
    """Valid PPL over LR x R — WITH the warning that makes it readable.

    The flow bound integrates over the horizon, and tau*(R) shrinks with R, so
    a column at large R covers a much smaller range than one at small R. Valid
    PPL therefore falls monotonically in R for mechanical reasons and must not
    be compared across columns. Even within a column it tracks GenPPL poorly.
    """
    sub = [r for r in runs if r['m'] == '1.0' and r['ppl'] is not None]
    if not sub:
        return
    cols = [c for c in RHOS if any(r['R'] == c for r in sub)]
    lines += ['', '### Valid PPL over LR x R (m = 1.0) — NOT a selection metric',
              '', '| lr \\ R | ' + ' | '.join(cols) + ' |',
              '|---' * (len(cols) + 1) + '|']
    for lr in LRS:
        out = []
        for R in cols:
            c = cell(sub, lr=lr, R=R)
            out.append('·' if c is None else f"{c['ppl']:.2f}")
        if any(o != '·' for o in out):
            lines.append(f'| {lr} | ' + ' | '.join(out) + ' |')
    lines += ['| **tau_max** | '
              + ' | '.join(f'{tau_star(R):.3f}' for R in cols) + ' |', '',
              'Each column is a different integration range (tau_max row), so '
              'PPL is comparable only *within* a column — and at large R it is '
              'anti-correlated with GenPPL. Select on GenPPL + entropy.']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md')
    args = ap.parse_args()

    runs = load()
    lines = [f'{len(runs)} cell(s) with GenPPL. '
             f'Entries: GenPPL (entropy); ⚠ = entropy < {ENT_MIN} (degenerate, '
             'inadmissible). **bold** = best ADMISSIBLE cell.']

    pivot(runs, LRS, 'lr', RHOS, 'R', {'m': '1.0'},
          'Stage 1 — GenPPL over LR x R at the closed-form truncation (m = 1.0)',
          lines)
    ppl_pivot(runs, lines)

    mults = sorted({r['m'] for r in runs}, key=float)
    if len(mults) > 1:
        for lr in LRS:
            if not any(r['lr'] == lr and r['m'] != '1.0' for r in runs):
                continue
            pivot(runs, mults, 'm', RHOS, 'R', {'lr': lr},
                  f'Stage 2 — GenPPL over m x R at lr = {lr} '
                  f'(TAU_MAX = m · tau*(R))', lines)

    lines += ['', '### All cells (best GenPPL first)', '',
              '| cell | tau_max | valid PPL | GenPPL | entropy |',
              '|---|---|---|---|---|']
    for r in sorted(runs, key=lambda r: (r['gen'] is None, r['gen'])):
        p = f"{r['ppl']:.3f}" if isinstance(r['ppl'], (int, float)) else 'n/a'
        g = f"{r['gen']:.3f}" if r['gen'] is not None else 'n/a'
        e = f"{r['ent']:.3f}" if r['ent'] is not None else 'n/a'
        w = '' if r['w'] == 'ce' else f", {r['w']}"
        lines.append(f"| lr {r['lr']}, R {r['R']}, m {r['m']}{w} | "
                     f"{r['tau']:.4f} | {p} | {g} | {e} |")

    txt = '\n'.join(lines)
    print(txt)
    if args.md:
        with open(args.md, 'w') as f:
            f.write(txt + '\n')


if __name__ == '__main__':
    main()
