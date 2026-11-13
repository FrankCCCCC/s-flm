#!/usr/bin/env python
"""Tables of hbfm_ce_tinystories_256 (read after `python sweep.py --site a100 --pull`), ranked by the exact NELBO.

Final table: the exact NELBO (nats/token, eval/nelbo.json, full validation set, SEED=1), val/ppl of the CE objective
(eval/ppl.json), GenPPL (gpt2-large, retokenised, 64 samples, T_MAX = the horizon) with its entropy and distinct samples.
Stage table: the exact NELBO on 10% of the validation set at 6k / 10k (val/loss_std in brackets).

Usage:  python report.py
"""
import glob
import json
import os
import re

from gate import NAME, stage_scores
from sweep import SITES

OUT = f'{SITES["a6000"]["out"]}/{NAME}'
TAG = re.compile(r'(.+)_seed-(\d+)$')


def load(path):
  try:
    with open(path) as f:
      return json.load(f)
  except (OSError, ValueError):
    return {}


def fmt(x, p=3):
  return f'{x:.{p}f}' if isinstance(x, (int, float)) else 'n/a'


cells = sorted(d for d in glob.glob(f'{OUT}/*') if TAG.match(os.path.basename(d)))
rows = []
for d in cells:
  gen = load(f'{d}/eval/samples_genppl.json')
  text = gen.get('text', [])
  rows.append((os.path.basename(d), load(f'{d}/eval/nelbo.json').get('val/nll'), load(f'{d}/eval/nelbo.json').get('val/loss_std'),
               load(f'{d}/eval/ppl.json').get('val/ppl'), gen.get('gen_ppl_first_chunk_retok'), gen.get('entropy'),
               f'{len(set(text))}/{len(text)}' if text else 'n/a'))
rows.sort(key=lambda r: (r[1] is None, r[1] or 0))
print('| cell | NELBO | std | val/ppl (CE) | GenPPL | entropy | uniq |')
print('|---|---:|---:|---:|---:|---:|---:|')
for r in rows:
  if r[1] is not None:
    print(f'| {r[0]} | {fmt(r[1], 4)} | {fmt(r[2], 2)} | {fmt(r[3])} | {fmt(r[4], 2)} | {fmt(r[5])} | {r[6]} |')

stages = {os.path.basename(d): stage_scores(d) for d in cells}
steps = sorted({k for v in stages.values() for k in v})
print('\n| cell | ' + ' | '.join(f'{k // 1000}k' for k in steps) + ' |')
print('|---|' + '---:|' * len(steps))
first = steps[0] if steps else None
for tag, v in sorted(stages.items(), key=lambda kv: kv[1].get(first, {}).get('val/nll', 1e9)):
  cols = [f'{fmt(v[k]["val/nll"])} ({fmt(v[k].get("val/loss_std"), 2)})' if k in v else '' for k in steps]
  print(f'| {tag} | ' + ' | '.join(cols) + ' |')
