#!/usr/bin/env python
"""Collector for the truncated autonomous-clock E-FLM sweep with checkpoint history.

Reads outputs/eflm_rescale_auto_trunc_tinystories_256_history/eflmrath_*/eval/
{ppl.json, samples_genppl.json} and prints:
  1. GenPPL over LR x R at m = 1.0             (stage 1: the LR ladder)
  2. valid PPL over LR x R                     (diagnostic only, NOT selection)
  3. GenPPL over m x R                         (stage 2: the truncation axis)
  4. seed aggregation, mean +- sd              (stage 3: what actually ranks)
  5. flat table incl. the checkpoint-history depth of each cell

GenPPL is gpt2-large retokenized generative perplexity, lower better, and is
only meaningful read together with entropy (< 3.0 => degenerate collapse: a
model emitting one repeated token scores ~1.1, i.e. it WINS an unfiltered
ranking). Valid PPL is a flow bound whose integration range is tau*(R), so it
is not comparable across R.

Usage:  python analyze.py [--md RESULTS_tables.md]
"""
import argparse
import glob
import json
import math
import os
import re
import statistics

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm'
EXP = 'eflm_rescale_auto_trunc_tinystories_256_history'
OUT = f'{REPO}/outputs/{EXP}'
TAG = re.compile(r'eflmrath_lr-(?P<lr>[\w.-]+)_r-(?P<R>[\d.]+)_m-(?P<m>[\d.]+)'
                 r'(?:_s(?P<seed>\d+))?$')
LRS = ['3e-4', '1e-3']
RHOS = ['0.05', '0.1', '0.5', '1']
ENT_MIN = 3.0        # anti-collapse gate (setup.md: "without collapsing")
C = math.sqrt(2 * math.log(2 * (50257 - 1) / 0.1))  # 5.2575


def tau_star(R):
    return math.log1p(C / float(R))


def load():
    runs = []
    for ed in sorted(glob.glob(f'{OUT}/*/eval')):
        cell_dir = os.path.dirname(ed)
        m = TAG.search(os.path.basename(cell_dir))
        if not m:
            continue
        r = dict(m.groupdict())
        r['seed'] = r['seed'] or '1'
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
        r['nckpt'] = len(glob.glob(f'{cell_dir}/checkpoints/*.ckpt'))
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
    # The anti-collapse gate is load-bearing: without it the degenerate cell,
    # which has the LOWEST GenPPL in the sweep, is marked as the winner.
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
    """Valid PPL over LR x R — with the caveat that makes it readable."""
    sub = [r for r in runs if r['m'] == '1.0' and r['seed'] == '1'
           and r['ppl'] is not None]
    if not sub:
        return
    cols = [c for c in RHOS if any(r['R'] == c for r in sub)]
    lines += ['', '### Valid PPL over LR x R (m = 1.0) — NOT a selection metric',
              '', '| lr \\ R | ' + ' | '.join(cols) + ' |',
              '|---' * (len(cols) + 1) + '|']
    for lr in LRS:
        out = [('·' if cell(sub, lr=lr, R=R) is None
                else f"{cell(sub, lr=lr, R=R)['ppl']:.2f}") for R in cols]
        if any(o != '·' for o in out):
            lines.append(f'| {lr} | ' + ' | '.join(out) + ' |')
    lines += ['| **tau_max** | '
              + ' | '.join(f'{tau_star(R):.3f}' for R in cols) + ' |', '',
              'Each column integrates the flow bound over its own horizon '
              '(tau_max row), so PPL falls with R for mechanical reasons and is '
              'comparable only *within* a column. Select on GenPPL + entropy; '
              'PPL is kept as a collapse detector (nan / inf / ~1e134).']


def seed_table(runs, lines):
    groups = {}
    for r in runs:
        if r['gen'] is not None:
            groups.setdefault((r['lr'], r['R'], r['m']), []).append(r)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if not multi:
        return
    lines += ['', '### Seed replication (mean +- sd over seeds)', '',
              '| cell | seeds | GenPPL per seed | mean | sd | entropy |',
              '|---|---|---|---|---|---|']
    for (lr, R, m), v in sorted(multi.items(),
                                key=lambda kv: statistics.mean(
                                    r['gen'] for r in kv[1])):
        v = sorted(v, key=lambda r: int(r['seed']))
        g = [r['gen'] for r in v]
        lines.append(
            f'| lr {lr}, R {R}, m {m} | {",".join(r["seed"] for r in v)} | '
            + ' / '.join(f'{x:.2f}' for x in g)
            + f' | **{statistics.mean(g):.2f}** | {statistics.stdev(g):.2f} | '
            + ' / '.join(f'{r["ent"]:.2f}' for r in v) + ' |')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md')
    args = ap.parse_args()

    runs = load()
    lines = [f'{len(runs)} cell(s) with GenPPL. '
             f'Entries: GenPPL (entropy); ⚠ = entropy < {ENT_MIN} (degenerate, '
             'inadmissible). **bold** = best ADMISSIBLE cell.']

    pivot(runs, LRS, 'lr', RHOS, 'R', {'m': '1.0', 'seed': '1'},
          'Stage 1 — GenPPL over LR x R at the closed-form truncation (m = 1.0)',
          lines)
    ppl_pivot(runs, lines)

    mults = sorted({r['m'] for r in runs}, key=float)
    if len(mults) > 1:
        for lr in LRS:
            if any(r['lr'] == lr and r['m'] != '1.0' for r in runs):
                pivot(runs, mults, 'm', RHOS, 'R', {'lr': lr, 'seed': '1'},
                      f'Stage 2 — GenPPL over m x R at lr = {lr} '
                      f'(TAU_MAX = m · tau*(R))', lines)
    seed_table(runs, lines)

    lines += ['', '### All cells (best GenPPL first)', '',
              '| cell | seed | tau_max | valid PPL | GenPPL | entropy | ckpts |',
              '|---|---|---|---|---|---|---|']
    for r in sorted(runs, key=lambda r: (r['gen'] is None, r['gen'])):
        p = f"{r['ppl']:.3f}" if isinstance(r['ppl'], (int, float)) else 'n/a'
        g = f"{r['gen']:.3f}" if r['gen'] is not None else 'n/a'
        e = f"{r['ent']:.3f}" if r['ent'] is not None else 'n/a'
        lines.append(f"| lr {r['lr']}, R {r['R']}, m {r['m']} | {r['seed']} | "
                     f"{r['tau']:.4f} | {p} | {g} | {e} | {r['nckpt']} |")

    txt = '\n'.join(lines)
    print(txt)
    if args.md:
        with open(args.md, 'w') as f:
            f.write(txt + '\n')


if __name__ == '__main__':
    main()
