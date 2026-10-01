#!/usr/bin/env python
"""Aggregate claude_test_hbfm_tinystories_256: valid PPL (importance-weighted denoising-CE bound,
NOT a likelihood), GenPPL (gpt2-large retokenized), sample entropy and distinct samples per cell,
from outputs/claude_test_hbfm_tinystories_256/<tag>/eval*/{ppl.json,samples_genppl.json}
(`eval/` = the 30k last.ckpt; `eval_step<N>/` = the persistent keep-<N>.ckpt; `eval_step<N>_tmax<T>/`
= the same checkpoint sampled at a different physical horizon T). Prints the table and rewrites the
block between the markers in RESULTS.md. Read GenPPL WITH entropy and uniq/64: low GenPPL + low
entropy / few distinct samples = degenerate, not good.

Tags: mode-{exp|trunc_exp|unif}_rate-{physical time_exp_rate}_k{K}_ub-{time_range_upper_bound}_seed-{s}
(rate is a no-op for unif, ub for exp). Unit rate = rate / |K|; t_max is the physical sampler horizon.

Usage:  python report.py
"""
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import t_max as default_t_max  # noqa: E402

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm'
OUT = f'{REPO}/outputs/claude_test_hbfm_tinystories_256'
RESULTS = f'{REPO}/experiments/claude_test_hbfm_tinystories_256/RESULTS.md'
# The _fwd- component is optional so the naive cells of stages 1-3 keep their original directory names.
TAG = re.compile(r'mode-(\w+?)_rate-([\d.e+-]+)_k(-[\d.]+)_ub-([\d.]+)(?:_fwd-(\w+))?_seed-(\d+)$')
EVAL_DIR = re.compile(r'eval(?:_step(\d+))?(?:_tmax([\d.]+))?$')
MODE_ORDER = {'exp': 0, 'trunc_exp': 1, 'unif': 2}
FWD_ORDER = {'naive': 0, 'horo': 1}


def load(path):
  try:
    with open(path) as f:
      return json.load(f)
  except Exception:
    return None


def main():
  rows = {}
  for ed in sorted(glob.glob(f'{OUT}/*/eval*')):
    tag = os.path.basename(os.path.dirname(ed))
    m, me = TAG.match(tag), EVAL_DIR.match(os.path.basename(ed))
    if not m or not me:
      continue
    pj, gj = load(f'{ed}/ppl.json'), load(f'{ed}/samples_genppl.json')
    if pj is None and gj is None:
      continue
    mode, rate, k, ub, fwd, seed = m.groups()
    fwd = fwd or 'naive'
    step = int(me.group(1)) if me.group(1) else 30000
    tm = float(me.group(2)) if me.group(2) else float(default_t_max(mode, k, ub))
    text = gj.get('text', []) if gj else []
    rows[(fwd, mode, float(rate), float(k), float(ub), tm, step, int(seed))] = dict(
      ppl=pj.get('val/ppl') if pj else None,
      gen=gj.get('gen_ppl_first_chunk_retok') if gj else None,
      ent=gj.get('entropy') if gj else None, nfe=gj.get('avg_nfe') if gj else None,
      uniq=f'{len(set(text))}/{len(text)}' if text else 'n/a')

  def fmt(x, p=2):
    return f'{x:.{p}f}' if isinstance(x, (int, float)) else 'n/a'

  def order(key):
    fwd, mode, rate, k, ub, tm, step = key[:7]
    return (FWD_ORDER.get(fwd, 9), MODE_ORDER.get(mode, 9), rate, -k, ub, -step, tm) + tuple(key[7:])

  lines = ['| fwd | mode | rate | K | unit rate | ub | t_max | step | seed | val/ppl | GenPPL | entropy | uniq | NFE |',
           '|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|']
  groups = {}
  for key in sorted(rows, key=order):
    fwd, mode, rate, k, ub, tm, step, seed = key
    r = rows[key]
    lines.append(f'| {fwd} | {mode} | {rate:g} | {k:g} | {rate / abs(k):g} | {ub:g} | {tm:g} | {step} | {seed} | {fmt(r["ppl"], 3)} | '
                 f'{fmt(r["gen"])} | {fmt(r["ent"], 3)} | {r["uniq"]} | {fmt(r["nfe"], 0)} |')
    groups.setdefault(key[:7], []).append(r)
  agg = ['| fwd | mode | rate | K | unit rate | ub | t_max | step | n | val/ppl | GenPPL | entropy |',
         '|---|---|---|---|---|---|---|---|---|---:|---:|---:|']
  for key, rs in sorted(groups.items(), key=lambda kv: order(kv[0])):
    fwd, mode, rate, k, ub, tm, step = key

    def ms(vals, p):
      vals = [v for v in vals if isinstance(v, (int, float))]
      if not vals:
        return 'n/a'
      return f'{statistics.mean(vals):.{p}f} ± {statistics.stdev(vals):.{p}f}' if len(vals) > 1 else f'{vals[0]:.{p}f}'
    agg.append(f'| {fwd} | {mode} | {rate:g} | {k:g} | {rate / abs(k):g} | {ub:g} | {tm:g} | {step} | {len(rs)} | '
               f'{ms([r["ppl"] for r in rs], 3)} | {ms([r["gen"] for r in rs], 2)} | {ms([r["ent"] for r in rs], 3)} |')
  table = (f'{len(rows)} evaluated (cell, step, horizon) triples.\n\n### Per cell\n\n' + '\n'.join(lines)
           + '\n\n### Mean ± std over seeds\n\n' + '\n'.join(agg))
  print(table)
  if os.path.exists(RESULTS):
    with open(RESULTS) as f:
      doc = f.read()
    block = f'<!-- report:begin -->\n{table}\n<!-- report:end -->'
    new = re.sub(r'<!-- report:begin -->.*?<!-- report:end -->', lambda _: block, doc, flags=re.S)
    with open(RESULTS, 'w') as f:
      f.write(new)


if __name__ == '__main__':
  main()
