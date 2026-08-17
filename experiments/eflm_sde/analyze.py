#!/usr/bin/env python
"""eflm_sde — analysis: EFLM SDE eta-curves vs the baseline frontier.

Scrapes the sample_eval jsons written by `frontier_sweep.py` under
  outputs/eflm_sde/sd-{seed}/frontier/nfe-{nfe}_gt-{gt}_eta-{eta}/
  outputs/eflm_sde/sfm_sd-{seed}/frontier/nfe-{nfe}/
and overlays them, one panel per NFE, on the DUO/MDLM/FLM temperature
frontier measured by `experiments/naive_ar_tinystories_s256`
(per-seed data: frontier_cells.csv). Writes figure + per-cell csv and prints
markdown tables (per-curve cells and matched-entropy comparisons).

Example:
  python experiments/eflm_sde/analyze.py
"""
import argparse, csv, glob, json, os, re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EFLM_RE = re.compile(
  r'/sd-(?P<seed>\d+)/frontier/nfe-(?P<nfe>\d+)_gt-(?P<gt>[a-z][a-z0-9]*)'
  r'_eta-(?P<eta>[\d.]+)/samples_genppl\.json$')
SFM_RE = re.compile(
  r'/sfm_sd-(?P<seed>\d+)/frontier/nfe-(?P<nfe>\d+)/samples_genppl\.json$')

ENT_MIN = 3.0          # below this a cell is collapse, never ranked on GenPPL
BASE_STYLE = {'mdlm': ('MDLM', '#1f77b4'), 'duo': ('DUO', '#ff7f0e'),
              'flm': ('FLM', '#2ca02c')}
GT_STYLE = {'const': '#d62728', 'sqrt': '#9467bd', 'p75': '#17becf',
            'linear': '#8c564b', 'quad': '#e377c2'}


def load_eflm(dir_):
  cells = []
  for path in sorted(glob.glob(
      os.path.join(dir_, 'sd-*/frontier/nfe-*_gt-*_eta-*/samples_genppl.json'))):
    m = EFLM_RE.search(path)
    if m is None:
      continue
    d = json.load(open(path))
    cells.append(dict(method='eflm', gt=m['gt'], eta=float(m['eta']),
                      seed=int(m['seed']), nfe=int(m['nfe']),
                      gen_ppl=d['gen_ppl_first_chunk_retok'],
                      entropy=d['entropy'],
                      num_unique=len(set(d['text'])),
                      num_samples=len(d['text'])))
  return cells


def load_sfm(dir_):
  cells = []
  for path in sorted(glob.glob(
      os.path.join(dir_, 'sfm_sd-*/frontier/nfe-*/samples_genppl.json'))):
    m = SFM_RE.search(path)
    if m is None:
      continue
    d = json.load(open(path))
    cells.append(dict(method='sfm', gt='', eta=float('nan'),
                      seed=int(m['seed']), nfe=int(m['nfe']),
                      gen_ppl=d['gen_ppl_first_chunk_retok'],
                      entropy=d['entropy'],
                      num_unique=len(set(d['text'])),
                      num_samples=len(d['text'])))
  return cells


def load_baselines(csv_path):
  cells = []
  with open(csv_path) as f:
    for r in csv.DictReader(f):
      cells.append(dict(method=r['method'], seed=int(r['seed']),
                        nfe=int(r['nfe']), temp=float(r['temp']),
                        gen_ppl=float(r['gen_ppl']),
                        entropy=float(r['entropy'])))
  return cells


def seed_mean(cells, key):
  """Group by `key` value, -> [(key_val, H_mean, H_sd, P_mean, P_sd, n)]."""
  out = []
  for v in sorted({c[key] for c in cells}):
    at = [c for c in cells if c[key] == v]
    e = np.array([c['entropy'] for c in at])
    p = np.array([c['gen_ppl'] for c in at])
    out.append((v, e.mean(), e.std(ddof=1) if len(e) > 1 else 0.0,
                p.mean(), p.std(ddof=1) if len(p) > 1 else 0.0, len(at)))
  return out


def interp_at(points, h):
  """GenPPL at entropy h, linear interp over the entropy-sorted curve."""
  pts = sorted((e, p) for _, e, _, p, _, _ in points)
  es = np.array([e for e, _ in pts])
  ps = np.array([p for _, p in pts])
  if h < es.min() or h > es.max():
    return None
  return float(np.interp(h, es, ps))


def plot(eflm, sfm, base, nfes, out, title):
  n = len(nfes)
  ncol = min(4, n)
  nrow = (n + ncol - 1) // ncol
  fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow),
                           squeeze=False)
  for i, nfe in enumerate(nfes):
    ax = axes[i // ncol][i % ncol]
    for meth, (label, color) in BASE_STYLE.items():
      pts = seed_mean([c for c in base if c['method'] == meth
                       and c['nfe'] == nfe], 'temp')
      if not pts:
        continue
      e = np.array([p[1] for p in pts]); s = np.array([p[2] for p in pts])
      g = np.array([p[3] for p in pts]); gs = np.array([p[4] for p in pts])
      ax.plot(e, g, '-', color=color, lw=1.4, label=label, alpha=0.85)
      ax.fill_between(e, g - gs, g + gs, color=color, alpha=0.15, lw=0)
    for gt, color in GT_STYLE.items():
      sel = [c for c in eflm if c['gt'] == gt and c['nfe'] == nfe]
      if not sel:
        continue
      pts = seed_mean(sel, 'eta')
      e = np.array([p[1] for p in pts]); g = np.array([p[3] for p in pts])
      gs = np.array([p[4] for p in pts])
      ax.plot(e, g, 'o-', color=color, lw=1.6, ms=3.5,
              label=f'EFLM {gt}')
      ax.fill_between(e, g - gs, g + gs, color=color, alpha=0.15, lw=0)
    ode = [c for c in eflm if c['gt'] == 'ode' and c['nfe'] == nfe]
    if ode:
      pts = seed_mean(ode, 'nfe')[0]
      ax.plot(pts[1], pts[3], '*', color='k', ms=13, label='EFLM ODE (eta=0)')
    sf = [c for c in sfm if c['nfe'] == nfe]
    if sf:
      pts = seed_mean(sf, 'nfe')[0]
      ax.plot(pts[1], pts[3], 'D', color='#7f7f7f', ms=7, label='S-FLM')
    ax.set_yscale('log')
    ax.set_title(f'NFE = {nfe}')
    ax.set_xlabel('entropy')
    ax.set_ylabel('Gen. PPL')
    ax.grid(alpha=0.25)
  for j in range(n, nrow * ncol):
    axes[j // ncol][j % ncol].axis('off')
  by_label = {}
  for row in axes:
    for ax in row:
      h, l = ax.get_legend_handles_labels()
      by_label.update(dict(zip(l, h)))
  fig.legend(by_label.values(), by_label.keys(), loc='upper left',
             fontsize=8, ncol=1, framealpha=0.9)
  fig.suptitle(title)
  fig.tight_layout(rect=(0, 0, 1, 0.96))
  os.makedirs(os.path.dirname(out), exist_ok=True)
  fig.savefig(out, dpi=160)
  print(f'wrote {out}')


def write_csv(cells, path):
  cols = ['method', 'gt', 'eta', 'seed', 'nfe', 'gen_ppl', 'entropy',
          'num_unique', 'num_samples']
  with open(path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for c in sorted(cells, key=lambda c: (c['method'], c['gt'], c['nfe'],
                                          c['eta'], c['seed'])):
      w.writerow({k: c[k] for k in cols})
  print(f'wrote {path}')


def tables(eflm, sfm, base, nfes):
  print('\n## EFLM SDE cells (mean over seeds; ⚠ = entropy < '
        f'{ENT_MIN}, collapse)\n')
  print('| NFE | gt | eta | Gen. PPL | ±sd | entropy | ±sd | n seeds |')
  print('|---|---|---|---|---|---|---|---|')
  for nfe in nfes:
    for gt in ['ode'] + list(GT_STYLE):
      sel = [c for c in eflm if c['gt'] == gt and c['nfe'] == nfe]
      for eta, e, es, p, ps, n in seed_mean(sel, 'eta'):
        flag = ' ⚠' if e < ENT_MIN else ''
        print(f'| {nfe} | {gt} | {eta:g} | {p:.2f} | {ps:.2f} '
              f'| {e:.3f}{flag} | {es:.3f} | {n} |')
  print('\n## Matched-entropy Gen. PPL (linear interp along each curve; '
        '"-" = H unreachable)\n')
  hs = [4.10, 4.20, 4.35]
  hdr = ' | '.join(f'H={h:.2f}' for h in hs)
  print(f'| NFE | curve | {hdr} |')
  print('|---|---|' + '---|' * len(hs))
  for nfe in nfes:
    rows = []
    for meth in BASE_STYLE:
      pts = seed_mean([c for c in base if c['method'] == meth
                       and c['nfe'] == nfe], 'temp')
      if pts:
        rows.append((meth, pts))
    for gt in GT_STYLE:
      sel = [c for c in eflm if c['gt'] == gt and c['nfe'] == nfe
             and c['entropy'] >= ENT_MIN]
      if len(sel) > 1:
        rows.append((f'eflm/{gt}', seed_mean(sel, 'eta')))
    for name, pts in rows:
      vals = ' | '.join(f'{v:.2f}' if (v := interp_at(pts, h)) is not None
                        else '-' for h in hs)
      print(f'| {nfe} | {name} | {vals} |')
  if sfm:
    print('\n## S-FLM reference points (T-inert; one per NFE)\n')
    print('| NFE | Gen. PPL | ±sd | entropy | ±sd | n seeds |')
    print('|---|---|---|---|---|---|')
    for nfe in sorted({c['nfe'] for c in sfm}):
      _, e, es, p, ps, n = seed_mean([c for c in sfm if c['nfe'] == nfe],
                                     'nfe')[0]
      print(f'| {nfe} | {p:.2f} | {ps:.2f} | {e:.3f} | {es:.3f} | {n} |')


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--dir', default=f'{REPO}/outputs/eflm_sde')
  ap.add_argument('--base-csv', default=f'{REPO}/experiments/'
                  'naive_ar_tinystories_s256/frontier_cells.csv')
  ap.add_argument('--out', default=f'{REPO}/experiments/eflm_sde/figures/'
                  'genppl_entropy_frontier.png')
  ap.add_argument('--csv', default=f'{REPO}/experiments/eflm_sde/'
                  'frontier_cells.csv')
  ap.add_argument('--title', default='TinyStories seq-256 — EFLM SDE '
                  '(eta x g(t)) vs baseline frontier')
  args = ap.parse_args()

  eflm = load_eflm(args.dir)
  sfm = load_sfm(args.dir)
  base = load_baselines(args.base_csv)
  if not eflm:
    print('no EFLM cells found; nothing to do')
    return
  nfes = sorted({c['nfe'] for c in eflm})
  write_csv(eflm + sfm, args.csv)
  plot(eflm, sfm, base, nfes, args.out, args.title)
  tables(eflm, sfm, base, nfes)


if __name__ == '__main__':
  main()
