#!/usr/bin/env python
"""Table for the round-3 proposal-mode grid (sweep_mode.py): board accuracy per
(time_conversion_mode, physical rate, K, time_range_upper_bound, sampler t_max,
forward_type, LR) and mean +- std over seeds, from
outputs/claude_test_hbfm_sudoku/<cell>/eval/results.json.

Rewrites the marker block in RESULTS.md. Usage: python report_mode.py
"""
import glob
import json
import os
import re
import statistics

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm'
OUT = f'{REPO}/outputs/claude_test_hbfm_sudoku'
RESULTS = f'{REPO}/experiments/claude_test_hbfm_sudoku/RESULTS.md'
SEEDS = ['1', '2', '3']
TAG = re.compile(r'mode-(exp|texp|unif)_rate-([\d.]+)_k(-[\d.]+)_ub-([\d.]+|na)'
                 r'_tm-([\d.]+)_fwd-(naive|horo)_lr-([\d.e-]+)_seed-(\d+)$')


def horizon_table():
    """The eval-only sampler-horizon control (`sweep_mode.py --anchor-tmax`): the three finished
    round-2 anchor checkpoints (exp, rate 3, K -0.5, horosphere, LR 3e-4) re-sampled at several
    physical `sampler.t_max`. Their own horizon is 6.0, in `eval/`; the rest in `eval_tm-<t>/`.
    This is what de-confounds a bounded cell's shorter horizon from its different proposal."""
    curve = {}
    for seed in SEEDS:
        base = f'{OUT}/lr-3e-4_ada-0_k-0.5_seed-{seed}_rate-3'
        for path in glob.glob(f'{base}/eval/results.json') + glob.glob(f'{base}/eval_tm-*/results.json'):
            d = os.path.basename(os.path.dirname(path))
            tmax = 6.0 if d == 'eval' else float(d[len('eval_tm-'):])
            try:
                with open(path) as f:
                    curve.setdefault(tmax, {})[seed] = 100.0 * json.load(f)['accuracy']
            except (ValueError, KeyError):
                continue
    if not curve:
        return ''
    lines = ['', '### Sampler-horizon control (eval only, round-2 anchor checkpoints)', '',
             '| physical t_max | unit t_max | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | mean ± std |',
             '|---|---|' + '---:|' * (len(SEEDS) + 1)]
    for tmax in sorted(curve):
        vals = [curve[tmax].get(s) for s in SEEDS]
        done = [v for v in vals if v is not None]
        if len(done) >= 2:
            summary = f'**{statistics.mean(done):.1f} ± {statistics.stdev(done):.1f}**'
        elif done:
            summary = f'{done[0]:.1f} (n=1)'
        else:
            summary = 'pending'
        mark = ' (trained horizon)' if abs(tmax - 6.0) < 1e-9 else ''
        lines.append(f'| {tmax:g}{mark} | {0.5 * tmax:g} | '
                     + ' | '.join(f'{v:.1f}%' if v is not None else 'pending' for v in vals)
                     + f' | {summary} |')
    return '\n'.join(lines) + '\n'


def main():
    acc, n_boards = {}, 0
    for path in glob.glob(f'{OUT}/mode-*/eval/results.json'):
        m = TAG.match(os.path.basename(os.path.dirname(os.path.dirname(path))))
        if not m:
            continue
        with open(path) as f:
            r = json.load(f)
        mode, rate, k, ub, tm, fwd, lr, seed = m.groups()
        acc[(mode, rate, k, ub, tm, fwd, lr, seed)] = 100.0 * r['accuracy']
        n_boards = r['num_total']

    keys = sorted({k[:-1] for k in acc}, key=lambda c: (c[0], float(c[1]), float(c[2]), c[3], float(c[4]), c[5], float(c[6])))
    lines = ['| mode | rate | K | unit rate | ub | t_max | fwd | LR | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | mean ± std |',
             '|---|---|---|---|---|---|---|---|' + '---:|' * (len(SEEDS) + 1)]
    for mode, rate, k, ub, tm, fwd, lr in keys:
        vals = [acc.get((mode, rate, k, ub, tm, fwd, lr, s)) for s in SEEDS]
        done = [v for v in vals if v is not None]
        if len(done) >= 2:
            summary = f'**{statistics.mean(done):.1f} ± {statistics.stdev(done):.1f}** (n={len(done)})'
        elif done:
            summary = f'{done[0]:.1f} (n=1)'
        else:
            summary = 'pending'
        unit = f'{float(rate) / abs(float(k)):g}'
        lines.append(f'| {mode} | {rate} | {k} | {unit} | {ub} | {tm} | {fwd} | {lr} | '
                     + ' | '.join(f'{v:.1f}%' if v is not None else 'pending' for v in vals) + f' | {summary} |')
    table = '\n'.join(lines) + '\n' + horizon_table()
    print(f'{len(acc)} round-3 cells evaluated ({n_boards} boards each)\n' + table)

    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            doc = f.read()
        block = (f'<!-- report3:begin -->\n{len(acc)} round-3 cells evaluated '
                 f'({n_boards} boards each).\n\n{table}\n<!-- report3:end -->')
        if '<!-- report3:begin -->' in doc:
            doc = re.sub(r'<!-- report3:begin -->.*?<!-- report3:end -->', block, doc, flags=re.S)
            with open(RESULTS, 'w') as f:
                f.write(doc)
        else:
            print('\n(RESULTS.md has no report3 marker block yet; printed only)')


if __name__ == '__main__':
    main()
