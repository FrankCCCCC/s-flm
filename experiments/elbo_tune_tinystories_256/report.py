#!/usr/bin/env python
"""Tabulate the NELBO of every evaluated cell into RESULTS.md (between the report markers).

Reads outputs/elbo_tune_tinystories_256/<tag>/eval/nelbo_s*.json (and ce_s1.json of the CE-trained
controls); a cell's NELBO is the mean over its passes (one (t, noise) draw per validation sequence
per pass). Usage:  python sweep.py --pull && python report.py
"""
import glob
import json
import os
import re
import statistics as st

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev2/s-flm'
NAME = 'elbo_tune_tinystories_256'
OUT = f'{REPO}/outputs/{NAME}'
RESULTS = f'{REPO}/experiments/{NAME}/RESULTS.md'
MDLM_30K = 1.582   # naive_ar_tinystories_s256, LR 1e-3, 3 seeds (elfm_hbfm_ada_curv_elbo_sudoku/RESULTS.md §6)


def nll(path):
  return json.load(open(path))['val/nll']


def main():
  rows = []
  for d in sorted(glob.glob(f'{OUT}/*_seed-*')):
    tag = os.path.basename(d)
    passes = [nll(f) for f in sorted(glob.glob(f'{d}/eval/nelbo_s*.json'))]
    if not passes:
      continue
    steps = None
    m = re.search(r'max_steps=(\d+)', open(f'{d}/.hydra/overrides.yaml').read()) if os.path.exists(f'{d}/.hydra/overrides.yaml') else None
    if m:
      steps = int(m.group(1))
    ce = f'{d}/eval/ce_s1.json'
    rows.append(dict(tag=tag, config=tag.rsplit('_seed-', 1)[0], seed=int(tag.rsplit('_seed-', 1)[1]),
                     steps=steps, nelbo=st.mean(passes), n=len(passes),
                     ce=nll(ce) if os.path.exists(ce) else None))
  lines = [f'{len(rows)} evaluated cells; MDLM at 30k steps (3 seeds) = {MDLM_30K:.3f} nats/token.', '',
           '### Every cell', '',
           '| cell | steps | seed | NELBO (nats / token) | passes | CE objective | vs MDLM 30k |',
           '|---|---:|---:|---:|---:|---:|---:|']
  for r in rows:
    ce = '' if r['ce'] is None else f"{r['ce']:.4f}"
    lines.append(f"| {r['tag']} | {r['steps'] or '?'} | {r['seed']} | {r['nelbo']:.4f} | {r['n']} "
                 f"| {ce} | {r['nelbo'] - MDLM_30K:+.3f} |")
  lines += ['', '### Per configuration (mean ± std over seeds)', '',
            '| configuration | steps | seeds | NELBO (nats / token) |', '|---|---:|---|---:|']
  by = {}
  for r in rows:
    by.setdefault((r['config'], r['steps']), []).append(r)
  for (cfg, steps), rs in sorted(by.items()):
    v = [r['nelbo'] for r in rs]
    sd = f' ± {st.stdev(v):.4f}' if len(v) > 1 else ''
    lines.append(f"| {cfg} | {steps or '?'} | {','.join(str(r['seed']) for r in rs)} | **{st.mean(v):.4f}{sd}** (n={len(v)}) |")
  table = '\n'.join(lines)
  print(table)
  if os.path.exists(RESULTS):
    text = open(RESULTS).read()
    if '<!-- report:begin -->' in text:
      text = re.sub(r'<!-- report:begin -->.*?<!-- report:end -->',
                    '<!-- report:begin -->\n' + table + '\n<!-- report:end -->', text, flags=re.S)
      open(RESULTS, 'w').write(text)
      print(f'updated {RESULTS}')


if __name__ == '__main__':
  main()
