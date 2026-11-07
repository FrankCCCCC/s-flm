#!/usr/bin/env python
"""GOAL2 table: per arm x R x embedding dim, mean +- std over seeds of val PPL, GenPPL, entropy (eval/ of each
run under outputs/eflm_cov_reparam_fisher_cond) and train / valid NELBO (data/nelbo/, nelbo.py). Cells with fewer
than 3 seeds show n; a cell with a collapsed seed (GenPPL > 100 or train NELBO > 3) is listed per seed instead.

Usage:  python experiments/eflm_cov_reparam_fisher_cond/report.py
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f'{os.path.dirname(os.path.dirname(HERE))}/outputs/eflm_cov_reparam_fisher_cond'
COLS = (('val PPL', 1), ('GenPPL', 2), ('entropy', 2), ('train NELBO', 3), ('valid NELBO', 3))


def seed_metrics(run):
  e = f'{OUT}/{run}/eval'
  if not os.path.exists(f'{e}/samples_genppl.json'):
    return None
  g = json.load(open(f'{e}/samples_genppl.json'))
  m = {'val PPL': json.load(open(f'{e}/ppl.json'))['val/ppl'], 'GenPPL': g['gen_ppl_first_chunk_retok'],
       'entropy': g['entropy']}
  for split in ('train', 'valid'):
    f = f'{HERE}/data/nelbo/{run}_{split}.json'
    if os.path.exists(f):
      m[f'{split} NELBO'] = json.load(open(f))['nelbo']
  return m


def fmt(vals, digits):
  if not vals:
    return '—'
  s = f'{st.mean(vals):.{digits}f}' + (f' ± {st.stdev(vals):.{digits}f}' if len(vals) > 1 else '')
  return s + ('' if len(vals) == 3 else f' (n={len(vals)})')


def main():
  collapsed = []
  print('| arm | R | dim | ' + ' | '.join(c for c, _ in COLS) + ' |')
  print('|---|---|---|' + '---|' * len(COLS))
  for arm in ('naive', 'gauss'):
    for rho in ('0.1', '0.5', '1.0'):
      for dim in ('16', '48', '768'):
        runs = [f'fwd-{arm}_r-{rho}_ed-{dim}_seed-{s}' for s in (1, 2, 3)]
        per = [(r, m) for r in runs if (m := seed_metrics(r)) is not None]
        if any(m['GenPPL'] > 100 or m.get('train NELBO', 0) > 3 for _, m in per):
          collapsed.append((arm, rho, dim, per))
          print(f'| {arm} | {rho} | {dim} | ' + ' | '.join('see below' for _ in COLS) + ' |')
          continue
        print(f'| {arm} | {rho} | {dim} | '
              + ' | '.join(fmt([m[c] for _, m in per if c in m], d) for c, d in COLS) + ' |')
  for arm, rho, dim, per in collapsed:
    print(f'\n{arm} R {rho} dim {dim}, per seed:')
    for r, m in per:
      print(f'  seed {r[-1]}: ' + ', '.join(f'{c} {m[c]:.{d}f}' for c, d in COLS if c in m))


if __name__ == '__main__':
  main()
