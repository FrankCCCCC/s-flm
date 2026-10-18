#!/usr/bin/env python
"""Valid PPL / GenPPL / entropy table for eflm_rescale_auto_trunc_cov_reparam_tinystories_256, one row per
(noise, R, LR). Reads outputs/<project>/<tag>/eval/{ppl.json, samples_genppl.json}; prints markdown for RESULTS.md.
Entropy < 3.0 flags degenerate (repetitive) samples, where a low GenPPL is not quality."""
import json
import os

import sweep


def load(path):
  return json.load(open(path)) if os.path.exists(path) else {}


def fmt(x, nd=2):
  return '—' if x is None else f'{x:.{nd}f}'


rows = ['| noise | R | LR | valid PPL (bound) | GenPPL | entropy | note |', '|---|---|---|---|---|---|---|']
for noise in sweep.NOISES:
  for rho in sweep.RHOS:
    for lr in sweep.LRS:
      edir = f'{sweep.OUT}/{sweep.tag_of(noise, rho, lr)}/eval'
      ppl, gen = load(f'{edir}/ppl.json'), load(f'{edir}/samples_genppl.json')
      ent = gen.get('entropy')
      note = 'collapse?' if ent is not None and ent < 3.0 else ''
      rows.append(f"| {noise} | {rho} | {lr} | {fmt(ppl.get('val/ppl'))} | "
                  f"{fmt(gen.get('gen_ppl_first_chunk_retok'))} | {fmt(ent)} | {note} |")
print('\n'.join(rows))
