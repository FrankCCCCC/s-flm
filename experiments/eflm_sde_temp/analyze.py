#!/usr/bin/env python
"""eflm_sde_temp — pick the best `eta` for EFLM's temperature frontier line.

Scrapes the (NFE, eta, T) cells of `sweep.py` from
  outputs/eflm_sde_temp/sd-{seed}/nfe-{nfe}_eta-{eta}_t-{temp}/samples_genppl.json
and the eta = 0 reference column from
  outputs/eflm_sde/sd-{seed}/tfrontier/nfe-{nfe}_t-{temp}/samples_genppl.json
(already complete; `sweep.py` never re-runs it).

Each (NFE, eta) is one temperature curve in (entropy, Gen. PPL). The winner at
a budget is the eta whose curve is lowest at matched entropy, so the ranking is
done by interpolating Gen. PPL at fixed H **per seed** and then averaging --
seeds that do not bracket H are dropped, never extrapolated.

  python experiments/eflm_sde_temp/analyze.py [--h-targets 4.2 4.35 4.5]
"""
import argparse, csv, glob, json, os, re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CELL_RE = re.compile(
  r'/sd-(?P<seed>\d+)/nfe-(?P<nfe>\d+)_eta-(?P<eta>[\d.]+)_t-(?P<temp>[\d.]+)'
  r'/samples_genppl\.json$')
REF_RE = re.compile(
  r'/sd-(?P<seed>\d+)/tfrontier/nfe-(?P<nfe>\d+)_t-(?P<temp>[\d.]+)'
  r'/samples_genppl\.json$')
# Cliff-edge cells (eflm_sde/RESULTS.md §4): kept in the csv, excluded from fits.
COLLAPSE = dict(entropy=5.0, gen_ppl=200.0)


def _read(path, m, eta):
  d = json.load(open(path))
  sc = d['config']['sampler']            # guard: the cell ran what its name says
  assert (sc['eta'], sc['temperature'], sc['steps'], sc['top_k_velocity'],
          sc['gt_method']) == (eta, float(m['temp']), int(m['nfe']), -1,
                               'linear'), f'config/name mismatch: {path}'
  return dict(method='eflm', seed=int(m['seed']), nfe=int(m['nfe']), eta=eta,
              temp=float(m['temp']), gen_ppl=d['gen_ppl_first_chunk_retok'],
              entropy=d['entropy'], num_unique=len(set(d['text'])),
              num_samples=len(d['text']))


def load(dir_, ref_dir):
  cells = []
  for path in sorted(glob.glob(
      os.path.join(dir_, 'sd-*/nfe-*_eta-*_t-*/samples_genppl.json'))):
    m = CELL_RE.search(path)
    if m:
      cells.append(_read(path, m, float(m['eta'])))
  for path in sorted(glob.glob(
      os.path.join(ref_dir, 'sd-*/tfrontier/nfe-*_t-*/samples_genppl.json'))):
    m = REF_RE.search(path)
    if m:
      cells.append(_read(path, m, 0.0))
  return cells


def ok(c):
  return (c['gen_ppl'] is not None and np.isfinite(c['gen_ppl'])
          and c['gen_ppl'] <= COLLAPSE['gen_ppl']
          and c['entropy'] <= COLLAPSE['entropy'])


def curve(cells, nfe, eta, seed=None):
  """T-ordered [(T, H_mean, H_sd, PPL_mean, PPL_sd, n_seeds), ...]."""
  sel = [c for c in cells if c['nfe'] == nfe and c['eta'] == eta
         and (seed is None or c['seed'] == seed)]
  out = []
  for t in sorted({c['temp'] for c in sel}):
    at = [c for c in sel if c['temp'] == t]
    e = np.array([c['entropy'] for c in at])
    p = np.array([c['gen_ppl'] for c in at])
    out.append((t, e.mean(), e.std(ddof=1) if len(e) > 1 else 0.0,
                p.mean(), p.std(ddof=1) if len(p) > 1 else 0.0, len(at)))
  return np.array(out).reshape(-1, 6)


def at_entropy(cells, h, **match):
  """Per-seed interpolation at matched entropy -> (mean, sd, n_seeds).

  A seed contributes only if its own curve brackets h (no extrapolation).
  `match` selects the arm, e.g. nfe=64, eta=16.0 or method='mdlm'.
  """
  vals = []
  for sd in sorted({c['seed'] for c in cells}):
    sel = [c for c in cells
           if all(c.get(k) == v for k, v in match.items())
           and c['seed'] == sd and ok(c)]
    if len(sel) < 2:
      continue
    e = np.array([c['entropy'] for c in sel])
    p = np.array([c['gen_ppl'] for c in sel])
    if not (e.min() <= h <= e.max()):
      continue
    o = np.argsort(e)
    vals.append(float(np.interp(h, e[o], p[o])))
  if not vals:
    return None
  v = np.array(vals)
  return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def plot(cells, nfes, etas, out, baselines=None):
  ncol = min(4, len(nfes))
  nrow = int(np.ceil(len(nfes) / ncol))
  fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow),
                           squeeze=False)
  cmap = plt.get_cmap('viridis')
  pos = [e for e in etas if e > 0]
  for i, nfe in enumerate(nfes):
    ax = axes[i // ncol][i % ncol]
    if baselines is not None:
      for m, color in (('mdlm', '#1f77b4'), ('duo', '#ff7f0e'),
                       ('flm', '#2ca02c')):
        b = baselines.get((m, nfe))
        if b is not None:
          ax.plot(b[:, 0], b[:, 1], '--', lw=1.1, color=color, alpha=0.7,
                  label=f'{m.upper()} (baseline)')
    for eta in etas:
      c = curve(cells, nfe, eta)
      if not len(c):
        continue
      if eta == 0:
        color, lw, label = 'k', 2.0, 'eta = 0 (ODE ref)'
      else:
        j = pos.index(eta) / max(1, len(pos) - 1)
        color, lw, label = cmap(j), 1.4, f'eta = {eta:g}'
      ax.fill_between(c[:, 1], c[:, 3] - c[:, 4], c[:, 3] + c[:, 4],
                      color=color, alpha=0.15, linewidth=0)
      ax.plot(c[:, 1], c[:, 3], '-o', ms=3.2, lw=lw, color=color, label=label)
    ax.set_yscale('log')
    ax.set(title=f'NFE = {nfe}', xlabel='per-sample unigram entropy (nats)',
           ylabel='Gen. PPL (gpt2-large)')
    ax.grid(alpha=0.3, which='both')
  for j in range(len(nfes), nrow * ncol):
    axes[j // ncol][j % ncol].axis('off')
  by_label = {}
  for row in axes:
    for ax in row:
      by_label.update(dict(zip(*ax.get_legend_handles_labels()[::-1])))
  order = ([k for k in by_label if k.endswith('(baseline)')]
           + [k for k in by_label if k.startswith('eta = 0 ')]
           + sorted((k for k in by_label if k.startswith('eta = ')
                     and not k.startswith('eta = 0 ')),
                    key=lambda k: float(k.split('= ')[1])))
  axes[0][0].legend([by_label[k] for k in order], order, fontsize=7)
  fig.suptitle('EFLM SDE x temperature — Gen. PPL / entropy frontier '
               '(exact velocity, top_k_v = -1, GT = linear)', y=1.002)
  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  fig.savefig(out, dpi=150, bbox_inches='tight')
  print(f'wrote {out}')


def load_baselines(path):
  """{(method, nfe): [[H, PPL], ...]} seed-mean curves from the shared csv."""
  if not os.path.exists(path):
    return None
  raw = {}
  for r in csv.DictReader(open(path)):
    if r['method'] not in ('mdlm', 'duo', 'flm'):
      continue
    raw.setdefault((r['method'], int(r['nfe'])), {}).setdefault(
      float(r['temp']), []).append((float(r['entropy']), float(r['gen_ppl'])))
  return {k: np.array([np.mean(v[t], axis=0) for t in sorted(v)])
          for k, v in raw.items()}


def load_baseline_cells(path):
  """Per-seed {mdlm, duo, flm} cells from the shared frontier csv."""
  if not os.path.exists(path):
    return []
  out = []
  for r in csv.DictReader(open(path)):
    if r['method'] not in ('mdlm', 'duo', 'flm'):
      continue
    out.append(dict(method=r['method'], seed=int(r['seed']), nfe=int(r['nfe']),
                    temp=float(r['temp']), gen_ppl=float(r['gen_ppl']),
                    entropy=float(r['entropy']), num_unique=0, num_samples=512))
  return out


def competitiveness(cells, base, eta_of_nfe, h_targets):
  """setup.md's question: is EFLM competitive with MDLM / DUO / FLM now?"""
  print('\n### Competitiveness at matched entropy — EFLM eta* vs the baselines\n')
  print('| NFE | H | MDLM | DUO | FLM | EFLM eta=0 | EFLM eta* | eta* vs MDLM |')
  print('|---|---|---|---|---|---|---|---|')
  for nfe in sorted(eta_of_nfe):
    for h in h_targets:
      row, vals = [str(nfe), f'{h:.2f}'], {}
      for m in ('mdlm', 'duo', 'flm'):
        q = at_entropy(base, h, method=m, nfe=nfe)
        vals[m] = q
        row.append('—' if q is None else f'{q[0]:.2f} ±{q[1]:.2f}')
      for lbl, eta in (('ref', 0.0), ('sde', eta_of_nfe[nfe])):
        q = at_entropy(cells, h, nfe=nfe, eta=eta)
        vals[lbl] = q
        row.append('—' if q is None else f'{q[0]:.2f} ±{q[1]:.2f}')
      if vals['sde'] and vals['mdlm']:
        d = 100 * (vals['sde'][0] - vals['mdlm'][0]) / vals['mdlm'][0]
        row.append(f'**{d:+.1f}%**' + (' (EFLM wins)' if d < 0 else ''))
      else:
        row.append('—')
      print('| ' + ' | '.join(row) + ' |')


def write_csv(cells, path):
  keys = ['nfe', 'eta', 'temp', 'seed', 'gen_ppl', 'entropy', 'num_unique',
          'num_samples']
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  with open(path, 'w', newline='') as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    for c in sorted(cells, key=lambda c: (c['nfe'], c['eta'], c['temp'],
                                          c['seed'])):
      w.writerow({k: c[k] for k in keys})
  print(f'wrote {path}')


def tables(cells, nfes, etas, h_targets):
  print('\n### Matched-entropy Gen. PPL per (NFE, eta) — per-seed '
        'interpolation, mean ± sd (n seeds)\n')
  for h in h_targets:
    print(f'\n**H = {h}**\n')
    print('| NFE | ' + ' | '.join(f'eta={e:g}' for e in etas)
          + ' | best eta | gain vs eta=0 |')
    print('|---' * (len(etas) + 3) + '|')
    for nfe in nfes:
      row, best = [str(nfe)], None
      for e in etas:
        q = at_entropy(cells, h, nfe=nfe, eta=e)
        if q is None:
          row.append('—')
          continue
        row.append(f'{q[0]:.2f} ±{q[1]:.2f} ({q[2]})')
        if best is None or q[0] < best[1]:
          best = (e, q[0])
      ref = at_entropy(cells, h, nfe=nfe, eta=0.0)
      if best is None:
        row += ['—', '—']
      else:
        row.append(f'**{best[0]:g}**')
        row.append('—' if ref is None
                   else f'{100 * (ref[0] - best[1]) / ref[0]:+.1f}%')
      print('| ' + ' | '.join(row) + ' |')

  print('\n### Best cell per (NFE, eta), minimised over T\n')
  print('| NFE | eta | Gen. PPL ±sd | H | T | n seeds |')
  print('|---|---|---|---|---|---|')
  for nfe in nfes:
    for e in etas:
      c = curve(cells, nfe, e)
      if not len(c):
        continue
      keep = [r for r in c if r[3] <= COLLAPSE['gen_ppl']
              and r[1] <= COLLAPSE['entropy']]
      if not keep:
        continue
      r = min(keep, key=lambda r: r[3])
      print(f'| {nfe} | {e:g} | {r[3]:.2f} ±{r[4]:.2f} | {r[1]:.3f} '
            f'| {r[0]:.2f} | {r[5]:.0f} |')

  print('\n### Every cell (mean ± sd over seeds)\n')
  print('| NFE | eta | T | Gen. PPL | ±sd | H | ±sd | n seeds | flag |')
  print('|---|---|---|---|---|---|---|---|---|')
  for nfe in nfes:
    for e in etas:
      for r in curve(cells, nfe, e):
        flag = ('collapse' if (r[3] > COLLAPSE['gen_ppl']
                               or r[1] > COLLAPSE['entropy']) else '')
        print(f'| {nfe} | {e:g} | {r[0]:.2f} | {r[3]:.2f} | {r[4]:.2f} '
              f'| {r[1]:.4f} | {r[2]:.4f} | {r[5]:.0f} | {flag} |')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--dir', default='outputs/eflm_sde_temp')
  p.add_argument('--ref-dir', default='outputs/eflm_sde',
                 help='eta = 0 temperature column (experiments/eflm_sde)')
  p.add_argument('--baselines',
                 default='experiments/naive_ar_tinystories_s256/'
                         'frontier_cells.csv')
  p.add_argument('--h-targets', nargs='+', type=float,
                 default=[4.20, 4.35, 4.50])
  p.add_argument('--out', default='experiments/eflm_sde_temp/figures/'
                                  'eta_temp_frontier.png')
  p.add_argument('--csv', default='experiments/eflm_sde_temp/cells.csv')
  p.add_argument('--eta-star', nargs='+',
                 default=['32:8', '64:16', '128:32', '256:64'],
                 help='chosen eta per NFE for the competitiveness table')
  args = p.parse_args()

  cells = load(args.dir, args.ref_dir)
  assert cells, f'no samples_genppl.json under {args.dir}'
  nfes = sorted({c['nfe'] for c in cells if c['eta'] > 0})
  cells = [c for c in cells if c['nfe'] in nfes]
  etas = sorted({c['eta'] for c in cells})
  seeds = sorted({c['seed'] for c in cells})
  print(f'{len(cells)} cells: {len(nfes)} NFE x {len(etas)} eta x '
        f'{len(seeds)} seeds; {sum(1 for c in cells if not ok(c))} collapsed')

  plot(cells, nfes, etas, args.out, load_baselines(args.baselines))
  write_csv(cells, args.csv)
  tables(cells, nfes, etas, args.h_targets)
  eta_of_nfe = {int(k): float(v) for k, v in
                (x.split(':') for x in args.eta_star)}
  if eta_of_nfe:
    competitiveness(cells, load_baseline_cells(args.baselines), eta_of_nfe,
                    args.h_targets)


if __name__ == '__main__':
  main()
