#!/usr/bin/env python
"""Gen. PPL / entropy frontier line, one panel per NFE.

Scrapes the sample_eval result jsons written by
`experiments/naive_ar_tinystories_s256/frontier_sweep.py` under
  <dir>/m-{method}_lr-{lr}_sd-{seed}/frontier/nfe-{nfe}_t-{temp}/samples_genppl.json
Each (method, seed, NFE, T) cell contributes one (entropy, Gen. PPL) point; the
temperature sweep traces a curve per method, drawn as the mean over training
seeds with a +/- 1 sd shaded band. Protocol: S-FLM paper Fig. 10 / App. C.8.

Writes the figure and a per-cell csv, and prints the markdown result tables.

Example:
  python visualization/genppl_entropy_frontier_line.py \
    --dir outputs/naive_ar_tinystories_s256 \
    --out experiments/naive_ar_tinystories_s256/figures/genppl_entropy_frontier_line.png
"""
import argparse, csv, glob, json, os, re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CELL_RE = re.compile(
  r'/m-(?P<method>[^_/]+)_lr-(?P<lr>[^_/]+)_sd-(?P<seed>\d+)'
  r'/frontier/nfe-(?P<nfe>\d+)_t-(?P<temp>[\d.]+)/samples_genppl\.json$')
STYLE = {'mdlm': ('MDLM', '#1f77b4'), 'duo': ('DUO', '#ff7f0e'),
         'flm': ('FLM', '#2ca02c'), 'ar': ('AR', '#d62728')}


def load(dir_, lr):
  """-> list of per-cell dicts scraped from every samples_genppl.json."""
  cells = []
  for path in sorted(glob.glob(
      os.path.join(dir_, 'm-*/frontier/nfe-*_t-*/samples_genppl.json'))):
    m = CELL_RE.search(path)
    if m is None or m['lr'] != lr:
      continue
    d = json.load(open(path))
    cells.append(dict(method=m['method'], seed=int(m['seed']),
                      nfe=int(m['nfe']), temp=float(m['temp']),
                      gen_ppl=d['gen_ppl_first_chunk_retok'],
                      entropy=d['entropy'],
                      num_unique=len(set(d['text'])),
                      num_samples=len(d['text'])))
  return cells


def curve(cells, method, nfe):
  """Mean/sd over seeds at each temperature, ordered by temperature."""
  sel = [c for c in cells if c['method'] == method and c['nfe'] == nfe]
  out = []
  for t in sorted({c['temp'] for c in sel}):
    at = [c for c in sel if c['temp'] == t]
    e = np.array([c['entropy'] for c in at])
    p = np.array([c['gen_ppl'] for c in at])
    out.append((t, e.mean(), e.std(ddof=1) if len(e) > 1 else 0.0,
                p.mean(), p.std(ddof=1) if len(p) > 1 else 0.0,
                np.mean([c['num_unique'] for c in at]), len(at)))
  return np.array(out).reshape(-1, 7)


def plot(cells, methods, nfes, out, title):
  ncol = min(4, len(nfes))
  nrow = int(np.ceil(len(nfes) / ncol))
  fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.8 * nrow),
                           squeeze=False)
  for i, nfe in enumerate(nfes):
    ax = axes[i // ncol][i % ncol]
    for m in methods:
      c = curve(cells, m, nfe)
      if not len(c):
        continue
      label, color = STYLE.get(m, (m, None))
      ax.fill_between(c[:, 1], c[:, 3] - c[:, 4], c[:, 3] + c[:, 4],
                      color=color, alpha=0.2, linewidth=0)
      ax.plot(c[:, 1], c[:, 3], '-o', ms=3.5, color=color, label=label)
    ax.set_yscale('log')
    ax.set(title=f'NFE = {nfe}',
           xlabel='per-sample unigram entropy (nats)',
           ylabel='Gen. PPL (gpt2-large)')
    ax.grid(alpha=0.3, which='both')
    if i == 0:
      ax.legend(fontsize=8)
  for j in range(len(nfes), nrow * ncol):
    axes[j // ncol][j % ncol].axis('off')
  fig.suptitle(title, y=1.002)
  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  fig.savefig(out, dpi=150, bbox_inches='tight')
  print(f'wrote {out}')


def write_csv(cells, path):
  keys = ['method', 'seed', 'nfe', 'temp', 'gen_ppl', 'entropy',
          'num_unique', 'num_samples']
  with open(path, 'w', newline='') as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    for c in sorted(cells, key=lambda c: (c['method'], c['nfe'], c['temp'],
                                          c['seed'])):
      w.writerow({k: c[k] for k in keys})
  print(f'wrote {path}')


def tables(cells, methods, nfes):
  print('\n### Best Gen. PPL per (method, NFE), minimised over T\n')
  print('| NFE | ' + ' | '.join(f'{STYLE.get(m, (m,))[0]}: PPL ±sd @ T (H)'
                                for m in methods) + ' |')
  print('|---' * (len(methods) + 1) + '|')
  for nfe in nfes:
    row = [str(nfe)]
    for m in methods:
      c = curve(cells, m, nfe)
      if not len(c):
        row.append('—'); continue
      k = int(np.argmin(c[:, 3]))
      row.append(f'{c[k,3]:.2f} ±{c[k,4]:.2f} @ {c[k,0]:.2f} ({c[k,1]:.3f})')
    print('| ' + ' | '.join(row) + ' |')

  print('\n### Matched-entropy Gen. PPL at H* = highest entropy reached by all '
        'methods\n')
  print('| NFE | H* | ' + ' | '.join(STYLE.get(m, (m,))[0] for m in methods)
        + ' |')
  print('|---' * (len(methods) + 2) + '|')
  for nfe in nfes:
    cs = {m: curve(cells, m, nfe) for m in methods}
    if any(not len(c) for c in cs.values()):
      continue
    h = min(c[:, 1].max() for c in cs.values())
    row = [str(nfe), f'{h:.3f}']
    for m in methods:
      c = cs[m]
      o = np.argsort(c[:, 1])
      row.append(f'{np.interp(h, c[o, 1], c[o, 3]):.2f}')
    print('| ' + ' | '.join(row) + ' |')

  print('\n### Every cell (mean ± sd over seeds)\n')
  print('| method | NFE | T | Gen. PPL | ±sd | entropy | ±sd | uniq | n seeds |')
  print('|---|---|---|---|---|---|---|---|---|')
  for m in methods:
    for nfe in nfes:
      for r in curve(cells, m, nfe):
        print(f'| {m} | {nfe} | {r[0]:.2f} | {r[3]:.2f} | {r[4]:.2f} '
              f'| {r[1]:.4f} | {r[2]:.4f} | {r[5]:.0f} | {r[6]:.0f} |')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--dir', default='outputs/naive_ar_tinystories_s256')
  p.add_argument('--lr', default='1e-3', help='LR of the pretrained cells')
  p.add_argument('--methods', default='mdlm,duo,flm')
  p.add_argument('--out', default='experiments/naive_ar_tinystories_s256/'
                 'figures/genppl_entropy_frontier_line.png')
  p.add_argument('--csv', default='experiments/naive_ar_tinystories_s256/'
                 'frontier_cells.csv')
  p.add_argument('--title', default='TinyStories seq-256 — Gen. PPL / entropy '
                 'frontier (mean ± sd over training seeds, 512 samples/cell)')
  args = p.parse_args()

  cells = load(args.dir, args.lr)
  assert cells, f'no frontier samples_genppl.json under {args.dir}'
  methods = [m for m in args.methods.split(',')
             if any(c['method'] == m for c in cells)]
  nfes = sorted({c['nfe'] for c in cells})
  seeds = sorted({c['seed'] for c in cells})
  temps = sorted({c['temp'] for c in cells})
  print(f'{len(cells)} cells: {len(methods)} methods x {len(seeds)} seeds x '
        f'{len(nfes)} NFE x {len(temps)} temperatures '
        f'(complete = {len(methods) * len(seeds) * len(nfes) * len(temps)})')

  plot(cells, methods, nfes, args.out, args.title)
  write_csv(cells, args.csv)
  tables(cells, methods, nfes)


if __name__ == '__main__':
  main()
