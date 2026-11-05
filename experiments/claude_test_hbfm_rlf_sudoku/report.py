#!/usr/bin/env python
"""Aggregate claude_test_hbfm_rlf_sudoku: board accuracy per (HyLa arm, forward_type, LR, HyLa scale) and mean ± std
over seeds, from outputs/claude_test_hbfm_rlf_sudoku/<tag>/eval/results.json (mode=sudoku_eval), plus the headline
"RLF vs none" deltas per forward type. Rewrites the block between the markers in RESULTS.md (created if missing).

Usage:  python report.py
"""
import glob
import json
import os
import re
import statistics

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm'
OUT = f'{REPO}/outputs/claude_test_hbfm_rlf_sudoku'
RESULTS = f'{REPO}/experiments/claude_test_hbfm_rlf_sudoku/RESULTS.md'
SEEDS = ['1', '2', '3']
TAG = re.compile(r'hyla-(none|pure|cap5f|cap5|cat)_fwd-(horo|naive)_lr-([\d.e-]+)_sc-([\d.]+)(_nm-0)?(_d\d+)?_seed-(\d+)$')
SPHERE_TAG = re.compile(r'sphere-L(\d+)_fwd-(horo|naive)_lr-([\d.e-]+)(?:_rc-([\d.,]+))?(?:_mc-(\d+))?_seed-(\d+)$')   # stages 4-7: the harmonic basis (_mc-: Monte-Carlo features)
ARM_ORDER = {'none': 0, 'pure': 1, 'cap5': 2, 'cap5f': 3, 'cat': 4, 'sphere': 5}
FWD_ORDER = {'horo': 0, 'naive': 1}
HEADER = ('# claude_test_hbfm_rlf_sudoku — Results\n\nHBFM board accuracy on hard Sudoku with random Laplacian features '
          '(HyLa) of the input state vs the plain coordinates (see `EXPERIMENT.md`). Regenerate with `python report.py`. '
          'Every `±` is the sample std over seeds (ddof=1).\n\n<!-- report:begin -->\n<!-- report:end -->\n')


def summary(vals):
    done = [v for v in vals if v is not None]
    if len(done) >= 2:
        return f'**{statistics.mean(done):.1f} ± {statistics.stdev(done):.1f}** (n={len(done)})', statistics.mean(done)
    if done:
        return f'{done[0]:.1f} (n=1)', done[0]
    return 'pending', None


def main():
    acc, n_boards = {}, 0
    for path in glob.glob(f'{OUT}/hyla-*/eval/results.json') + glob.glob(f'{OUT}/sphere-*/eval/results.json'):
        tag = os.path.basename(os.path.dirname(os.path.dirname(path)))
        m, ms = TAG.match(tag), SPHERE_TAG.match(tag)
        if not m and not ms:
            continue
        try:
            with open(path) as f:
                r = json.load(f)
        except ValueError:
            continue
        if m:
            arm, fwd, lr, sc, nm, dim, seed = m.groups()
            key = (arm, fwd, lr, sc + (' (unnormalized)' if nm else '') + (f' D={dim[2:]}' if dim else ''), seed)
        else:
            L, fwd, lr, rc, mc, seed = ms.groups()
            key = ('sphere', fwd, lr, f'L={L}' + (f' c=[{rc}]' if rc else '') + (f' MC M={mc}' if mc else ''), seed)   # the "scale" column carries the degree (+ radial scales)
        acc[key] = 100.0 * r['accuracy']
        n_boards = r['num_total']

    keys = sorted({k[:-1] for k in acc},
                  key=lambda c: (FWD_ORDER[c[1]], ARM_ORDER[c[0]], float(c[2]), float(c[3].split()[0].lstrip('L=')), c[3]))
    lines = ['| arm | fwd | LR | HyLa scale / sphere degree | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | mean ± std |',
             '|---|---|---|---|' + '---:|' * (len(SEEDS) + 1)]
    means = {}
    for arm, fwd, lr, sc in keys:
        vals = [acc.get((arm, fwd, lr, sc, s)) for s in SEEDS]
        text, mean = summary(vals)
        means[(arm, fwd, lr, sc)] = mean
        lines.append(f'| {arm} | {fwd} | {lr} | {sc} | '
                     + ' | '.join(f'{v:.1f}%' if v is not None else 'pending' for v in vals) + f' | {text} |')
    head = ['', '### RLF vs none (LR 3e-4, scale 1, 3-seed means)', '',
            '| fwd | none | cap5 | cat | cap5 − none | cat − none |', '|---|---:|---:|---:|---:|---:|']
    for fwd in ('horo', 'naive'):
        m = {arm: means.get((arm, fwd, '3e-4', '1')) for arm in ('none', 'cap5', 'cat')}
        fmt = lambda x: f'{x:.1f}' if x is not None else 'pending'  # noqa: E731
        delta = lambda arm: (f'{m[arm] - m["none"]:+.1f}' if m[arm] is not None and m['none'] is not None  # noqa: E731
                             else 'pending')
        head.append(f'| {fwd} | {fmt(m["none"])} | {fmt(m["cap5"])} | {fmt(m["cat"])} | {delta("cap5")} | {delta("cat")} |')
    head += ['', '### sphere vs none (LR 3e-4, 3-seed means; stage 4, the harmonic basis)', '',
             '| fwd | none | L=2 | L=4 | L=6 | L=2 − none | L=4 − none | L=6 − none |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for fwd in ('horo', 'naive'):
        m = {'none': means.get(('none', fwd, '3e-4', '1'))}
        m.update({L: means.get(('sphere', fwd, '3e-4', L)) for L in ('L=2', 'L=4', 'L=6')})
        fmt = lambda x: f'{x:.1f}' if x is not None else 'pending'  # noqa: E731
        delta = lambda k: (f'{m[k] - m["none"]:+.1f}' if m[k] is not None and m['none'] is not None  # noqa: E731
                           else 'pending')
        head.append(f'| {fwd} | {fmt(m["none"])} | {fmt(m["L=2"])} | {fmt(m["L=4"])} | {fmt(m["L=6"])} | '
                    f'{delta("L=2")} | {delta("L=4")} | {delta("L=6")} |')
    table = '\n'.join(lines) + '\n' + '\n'.join(head) + '\n'
    print(f'{len(acc)} cells evaluated ({n_boards} boards each)\n' + table)

    if not os.path.exists(RESULTS):
        with open(RESULTS, 'w') as f:
            f.write(HEADER)
    with open(RESULTS) as f:
        doc = f.read()
    block = f'<!-- report:begin -->\n{len(acc)} cells evaluated ({n_boards} boards each).\n\n{table}\n<!-- report:end -->'
    doc = re.sub(r'<!-- report:begin -->.*?<!-- report:end -->', lambda _: block, doc, flags=re.S)
    with open(RESULTS, 'w') as f:
        f.write(doc)


if __name__ == '__main__':
    main()
