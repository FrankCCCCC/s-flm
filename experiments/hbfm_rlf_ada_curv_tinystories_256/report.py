#!/usr/bin/env python
"""Tables of hbfm_rlf_ada_curv_tinystories_256 (read after `python sweep.py --pull`).

Per (cell, step, sampler horizon t_max): the learned clock lambda (EMA weights, `curvature.json` from
curvature_ts.py; one number per model, see EXPERIMENT.md), the clock horizon lambda * t_max, val/ppl
(the importance-weighted CE bound, not a likelihood), GenPPL (gpt2-large, retokenised, 64 samples) with
its unigram entropy and distinct samples; then mean +- std over seeds. `eval/` is the 30k checkpoint at
the cell's own horizon, `eval_tmax<T>/` the same at horizon T, `eval_step<N>[_tmax<T>]/` keep-<N>.ckpt.
Prints the tables and rewrites the block between the markers in RESULTS.md.

Usage:  python report.py
"""
import glob
import json
import os
import re
import statistics

OUT = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm/outputs/hbfm_rlf_ada_curv_tinystories_256'
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RESULTS.md')
TAG = re.compile(r'cb-[\d.]+_mode-(\w+?)_rate-([\d.]+)_ub-(na|[\d.]+)_tm-([\d.]+)_lr-([\d.e-]+)(?:_fwd-(\w+))?_seed-(\d+)$')
EVAL = re.compile(r'eval(?:_step(\d+))?(?:_tmax([\d.]+))?$')
MODE = {'exp': 0, 'texp': 1, 'unif': 2}


def load(path):
  try:
    with open(path) as f:
      return json.load(f)
  except (OSError, ValueError):
    return None


def fmt(x, p=2):
  return f'{x:.{p}f}' if isinstance(x, (int, float)) else 'n/a'


def mean_std(vals, p):
  vals = [v for v in vals if isinstance(v, (int, float))]
  if len(vals) < 2:
    return fmt(vals[0], p) if vals else 'n/a'
  return f'{statistics.mean(vals):.{p}f} ± {statistics.stdev(vals):.{p}f}'


def order(key):
  mode, rate, ub, lr, tm, step = key[:6]
  return (MODE.get(mode, 9), rate, -1 if ub == 'na' else float(ub), float(lr), -step, tm) + key[6:]


def main():
  rows = {}
  for ed in glob.glob(f'{OUT}/*/eval*'):
    m, e = TAG.match(os.path.basename(os.path.dirname(ed))), EVAL.match(os.path.basename(ed))
    gen = load(f'{ed}/samples_genppl.json')
    if not (m and e and gen):
      continue
    mode, rate, ub, tm, lr, fwd, seed = m.groups()
    step, tm = int(e.group(1) or 30000), float(e.group(2) or tm)
    lam = (load(f'{os.path.dirname(ed)}/curvature.json') or {}).get(str(step), {}).get('mean')
    text = gen.get('text', [])
    rows[(mode, float(rate), ub, lr, tm, step, fwd or 'naive', int(seed))] = dict(
      lam=lam, clock=lam * tm if lam else None, ppl=(load(f'{ed}/ppl.json') or {}).get('val/ppl'),
      gen=gen.get('gen_ppl_first_chunk_retok'), ent=gen.get('entropy'), uniq=f'{len(set(text))}/{len(text)}')

  head = '| mode | rate | ub | LR | t_max | step | fwd |'
  per = [head + ' seed | λ | λ·t_max | val/ppl | GenPPL | entropy | uniq |',
         '|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|']
  groups = {}
  for key in sorted(rows, key=order):
    r = rows[key]
    mode, rate, ub, lr, tm, step, fwd, seed = key
    rate = '–' if mode == 'unif' else f'{rate:g}'
    per.append(f'| {mode} | {rate} | {ub} | {lr} | {tm:g} | {step} | {fwd} | {seed} | {fmt(r["lam"], 3)} | '
               f'{fmt(r["clock"], 3)} | {fmt(r["ppl"], 3)} | {fmt(r["gen"])} | {fmt(r["ent"], 3)} | {r["uniq"]} |')
    groups.setdefault(key[:7], []).append(r)
  agg = [head + ' n | λ | GenPPL | entropy |', '|---|---|---|---|---|---|---|---:|---:|---:|---:|']
  for key, rs in sorted(groups.items(), key=lambda kv: order(kv[0])):
    mode, rate, ub, lr, tm, step, fwd = key
    rate = '–' if mode == 'unif' else f'{rate:g}'
    agg.append(f'| {mode} | {rate} | {ub} | {lr} | {tm:g} | {step} | {fwd} | {len(rs)} | '
               f'{mean_std([r["lam"] for r in rs], 3)} | {mean_std([r["gen"] for r in rs], 2)} | '
               f'{mean_std([r["ent"] for r in rs], 3)} |')
  table = (f'{len(rows)} evaluated (cell, step, horizon) reads.\n\n### Per read\n\n' + '\n'.join(per)
           + '\n\n### Mean ± std over seeds\n\n' + '\n'.join(agg))
  print(table)
  if os.path.exists(RESULTS):
    with open(RESULTS) as f:
      doc = f.read()
    block = f'<!-- report:begin -->\n{table}\n<!-- report:end -->'
    with open(RESULTS, 'w') as f:
      f.write(re.sub(r'<!-- report:begin -->.*?<!-- report:end -->', lambda _: block, doc, flags=re.S))


if __name__ == '__main__':
  main()
