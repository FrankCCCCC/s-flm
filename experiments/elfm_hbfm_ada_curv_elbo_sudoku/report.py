#!/usr/bin/env python
"""Tables of the elfm_hbfm_ada_curv_elbo_sudoku sweep (sweep.py), from
outputs/elfm_hbfm_ada_curv_elbo_sudoku/<cell>/eval/{results.json, ppl_elbo_s*.json, ppl_ce_s*.json}:

  1. every cell: board accuracy, NELBO (mean +- s.e. over the ppl passes, nats per solution token)
     and the CE objective;
  2. per (arm, training loss, LR): mean +- std over seeds of the accuracy and of the NELBO.

Rewrites the marker block in RESULTS.md. Usage: python report.py
"""
import glob
import json
import math
import os
import re
import statistics

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev2/s-flm'
OUT = f'{REPO}/outputs/elfm_hbfm_ada_curv_elbo_sudoku'
RESULTS = f'{REPO}/experiments/elfm_hbfm_ada_curv_elbo_sudoku/RESULTS.md'
TAG = re.compile(r'(hbfm|eflm|mdlm)_loss-(ce|elbo)_lr-([\d.e-]+)_seed-(\d+)$')
ARM = {'hbfm': 'HBFM (ada curv, b=4)', 'eflm': 'EFLM (R=8, tau 0.5)', 'mdlm': 'MDLM'}


def load(path):
    with open(path) as f:
        return json.load(f)


def nll_passes(eval_dir, loss):
    vals = []
    for p in sorted(glob.glob(f'{eval_dir}/ppl_{loss}_s*.json')):
        d = load(p)
        key = next((k for k in ('val/nll', 'val_nll') if k in d), None)
        if key is not None:
            vals.append(float(d[key]))
    return vals


def mean_se(vals):
    if not vals:
        return None
    m = statistics.mean(vals)
    se = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else float('nan')
    return m, se, len(vals)


def fmt_ms(x, digits=4):
    if x is None:
        return 'pending'
    m, se, n = x
    return f'{m:.{digits}f} ± {se:.{digits}f} (n={n})' if n > 1 else f'{m:.{digits}f} (n=1)'


def main():
    rows = {}
    for d in sorted(glob.glob(f'{OUT}/*/eval')):
        m = TAG.match(os.path.basename(os.path.dirname(d)))
        if not m:
            continue
        arm, loss, lr, seed = m.groups()
        acc = 100.0 * load(f'{d}/results.json')['accuracy'] if os.path.exists(f'{d}/results.json') else None
        rows[(arm, loss, lr, int(seed))] = dict(acc=acc, nelbo=mean_se(nll_passes(d, 'elbo')),
                                                ce=mean_se(nll_passes(d, 'ce')))

    lines = ['### Every cell', '',
             '| arm | trained on | LR | seed | board acc | NELBO (nats / solution token) | CE objective |',
             '|---|---|---|---|---:|---:|---:|']
    for (arm, loss, lr, seed), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], float(kv[0][2]), kv[0][3])):
        acc = f'{r["acc"]:.1f}%' if r['acc'] is not None else 'pending'
        lines.append(f'| {ARM[arm]} | {loss} | {lr} | {seed} | {acc} | {fmt_ms(r["nelbo"])} | '
                     f'{fmt_ms(r["ce"]) if arm != "mdlm" else "—"} |')

    lines += ['', '### Per configuration (mean ± std over seeds)', '',
              '| arm | trained on | LR | seeds | board acc | NELBO (nats / solution token) |',
              '|---|---|---|---|---:|---:|']
    groups = sorted({k[:3] for k in rows}, key=lambda k: (k[0], k[1], float(k[2])))
    for arm, loss, lr in groups:
        seeds = sorted(s for (a, l, r, s) in rows if (a, l, r) == (arm, loss, lr))
        accs = [rows[(arm, loss, lr, s)]['acc'] for s in seeds if rows[(arm, loss, lr, s)]['acc'] is not None]
        nelbos = [rows[(arm, loss, lr, s)]['nelbo'][0] for s in seeds if rows[(arm, loss, lr, s)]['nelbo']]

        def agg(vals, digits):
            if len(vals) >= 2:
                return f'**{statistics.mean(vals):.{digits}f} ± {statistics.stdev(vals):.{digits}f}** (n={len(vals)})'
            return f'{vals[0]:.{digits}f} (n=1)' if vals else 'pending'
        lines.append(f'| {ARM[arm]} | {loss} | {lr} | {",".join(map(str, seeds))} | {agg(accs, 1)} | {agg(nelbos, 4)} |')
    table = '\n'.join(lines) + '\n'
    print(f'{len(rows)} cells\n' + table)

    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            doc = f.read()
        block = f'<!-- report:begin -->\n{len(rows)} cells with evals.\n\n{table}\n<!-- report:end -->'
        if '<!-- report:begin -->' in doc:
            doc = re.sub(r'<!-- report:begin -->.*?<!-- report:end -->', lambda _: block, doc, flags=re.S)
            with open(RESULTS, 'w') as f:
                f.write(doc)
        else:
            print('(RESULTS.md has no report marker block yet; printed only)')


if __name__ == '__main__':
    main()
