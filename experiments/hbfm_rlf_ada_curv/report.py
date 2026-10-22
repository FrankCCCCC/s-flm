#!/usr/bin/env python
"""Table of the hbfm_rlf_ada_curv sweep (sweep.py): board accuracy per (curvature bound,
time_conversion_mode, physical rate, K, time_range_upper_bound, sampler t_max, forward_type, LR)
and mean +- std over seeds, from outputs/hbfm_rlf_ada_curv/<cell>/eval/results.json.

Rewrites the marker block in RESULTS.md. Usage: python report.py
"""
import glob
import json
import os
import re
import statistics

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm'
OUT = f'{REPO}/outputs/hbfm_rlf_ada_curv'
RESULTS = f'{REPO}/experiments/hbfm_rlf_ada_curv/RESULTS.md'
SEEDS = ['1', '2', '3']
TAG = re.compile(r'cb-([\d.]+)_mode-(exp|texp|unif)_rate-([\d.]+)_k(-[\d.]+)_ub-([\d.]+|na)'
                 r'_tm-([\d.]+)_fwd-(naive|horo)_lr-([\d.e-]+)_seed-(\d+)$')


def main():
    acc, n_boards = {}, 0
    for path in glob.glob(f'{OUT}/cb-*/eval/results.json'):
        m = TAG.match(os.path.basename(os.path.dirname(os.path.dirname(path))))
        if not m:
            continue
        with open(path) as f:
            r = json.load(f)
        acc[m.groups()] = 100.0 * r['accuracy']
        n_boards = r['num_total']

    keys = sorted({k[:-1] for k in acc},
                  key=lambda c: (c[1], float(c[2]), float(c[3]), c[4], float(c[5]), c[6], float(c[7]), float(c[0])))
    lines = ['| curvature bound | mode | rate | K | unit rate | ub | t_max | fwd | LR | '
             + ' | '.join(f'seed {s}' for s in SEEDS) + ' | mean ± std |',
             '|---|---|---|---|---|---|---|---|---|' + '---:|' * (len(SEEDS) + 1)]
    for cb, mode, rate, k, ub, tm, fwd, lr in keys:
        vals = [acc.get((cb, mode, rate, k, ub, tm, fwd, lr, s)) for s in SEEDS]
        done = [v for v in vals if v is not None]
        if len(done) >= 2:
            summary = f'**{statistics.mean(done):.1f} ± {statistics.stdev(done):.1f}** (n={len(done)})'
        elif done:
            summary = f'{done[0]:.1f} (n=1)'
        else:
            summary = 'pending'
        bound = 'fixed (1)' if float(cb) == 1 else cb
        lines.append(f'| {bound} | {mode} | {rate} | {k} | {float(rate) / abs(float(k)):g} | {ub} | {tm} '
                     f'| {fwd} | {lr} | '
                     + ' | '.join(f'{v:.1f}%' if v is not None else 'pending' for v in vals)
                     + f' | {summary} |')
    table = '\n'.join(lines) + '\n'
    print(f'{len(acc)} cells evaluated ({n_boards} boards each)\n' + table)

    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            doc = f.read()
        block = (f'<!-- report:begin -->\n{len(acc)} cells evaluated ({n_boards} boards each).\n\n'
                 f'{table}\n<!-- report:end -->')
        if '<!-- report:begin -->' in doc:
            doc = re.sub(r'<!-- report:begin -->.*?<!-- report:end -->', lambda _: block, doc, flags=re.S)
            with open(RESULTS, 'w') as f:
                f.write(doc)
        else:
            print('\n(RESULTS.md has no report marker block yet; printed only)')


if __name__ == '__main__':
    main()
