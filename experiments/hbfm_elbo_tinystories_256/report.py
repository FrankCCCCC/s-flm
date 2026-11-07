#!/usr/bin/env python
"""Table of hbfm_elbo_tinystories_256 (read after `python sweep.py --pull`), ranked by the exact NELBO.

Per cell x arm x seed: the exact NELBO (nats/token, `eval/nelbo.json`, full validation set, SEED=1),
the unsampled-slack bound 0.128 / r_unit at lambda = 1 (setup.md), val/ppl of the CE bound
(`eval/ppl.json`), GenPPL (gpt2-large, retokenised, 64 samples at T_MAX = T) with its unigram entropy
and distinct samples, and the learned clock lambda of the 30k checkpoint (`curvature.json`, ada only).

Usage:  python report.py
"""
import glob
import json
import math
import os
import re

from sweep import CELLS, NAME, REPO, cfg

OUT = f'{REPO}/outputs/{NAME}'
TAG = re.compile(r'(.+)_curv-(\w+)_seed-(\d+)$')


def load(path):
  try:
    with open(path) as f:
      return json.load(f)
  except (OSError, ValueError):
    return None


def fmt(x, p=3):
  return f'{x:.{p}f}' if isinstance(x, (int, float)) else 'n/a'


def main():
  rows = []
  for d in glob.glob(f'{OUT}/*'):
    m = TAG.match(os.path.basename(d))
    if not m or m.group(1) not in CELLS:
      continue
    cell, arm, seed = m.groups()
    c = cfg(cell)
    gen = load(f'{d}/eval/samples_genppl.json') or {}
    text = gen.get('text', [])
    rows.append(dict(
      cell=cell, arm=arm, seed=seed, rate=c['rate'], k=c['k'],
      nelbo=(load(f'{d}/eval/nelbo.json') or {}).get('val/nll'),
      slack=0.128 * abs(c['k']) / c['rate'] * (1 if c['ub'] is None else -math.expm1(-c['rate'] * c['ub'])),
      ppl=(load(f'{d}/eval/ppl.json') or {}).get('val/ppl'), gen=gen.get('gen_ppl_first_chunk_retok'),
      ent=gen.get('entropy'), uniq=f'{len(set(text))}/{len(text)}' if text else 'n/a',
      lam=(load(f'{d}/curvature.json') or {}).get('30000', {}).get('mean')))
  rows.sort(key=lambda r: (r['nelbo'] is None, r['nelbo'] or 0))
  print('| cell | arm | seed | rate | K | NELBO | slack | val/ppl (CE) | GenPPL | entropy | uniq | λ (30k) |')
  print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
  for r in rows:
    print(f'| {r["cell"]} | {r["arm"]} | {r["seed"]} | {r["rate"]:g} | {r["k"]:g} | {fmt(r["nelbo"])} | '
          f'{fmt(r["slack"])} | {fmt(r["ppl"])} | {fmt(r["gen"], 2)} | {fmt(r["ent"])} | {r["uniq"]} | '
          f'{fmt(r["lam"])} |')

  # The stage / early-stopping scores: exact NELBO on 10% of the validation set (eval/nelbo_{steps}.json).
  stages = {}
  for d in glob.glob(f'{OUT}/*'):
    m = TAG.match(os.path.basename(d))
    for f in glob.glob(f'{d}/eval/nelbo_*.json'):
      if m and m.group(1) in CELLS:
        stages.setdefault(m.group(1, 2), {})[int(re.search(r'nelbo_(\d+)', f).group(1))] = load(f) or {}
  steps = sorted({k for v in stages.values() for k in v})
  print(f'\nNELBO on 10% of the validation set at step (val/loss_std in brackets)\n')
  print('| cell | arm | ' + ' | '.join(f'{k // 1000}k' for k in steps) + ' |')
  print('|---|---|' + '---:|' * len(steps))
  for (cell, arm), v in sorted(stages.items(), key=lambda kv: (list(CELLS).index(kv[0][0]), kv[0][1])):
    cols = [f'{fmt(v[k].get("val/nll"))} ({fmt(v[k].get("val/loss_std"), 2)})' if k in v else '' for k in steps]
    print(f'| {cell} | {arm} | ' + ' | '.join(cols) + ' |')


if __name__ == '__main__':
  main()
