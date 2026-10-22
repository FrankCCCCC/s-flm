#!/usr/bin/env python
"""Rank the seed-1 k1 ladders of frontier_sweep.py (EXPERIMENT.md, "Frontier evaluation" /
"ODE and SDE step types"): per (model, step type, t_max) the Gen. PPL @ entropy at every NFE,
the sum of log Gen. PPL over NFE 8-256 (lower = better) and the mean entropy there (a candidate
whose entropy is lower by more than ~0.03 loses the tie-break to the frontier). Reads the local
tree (run `sweep.py --pull` first for cells computed on the A100 site).

Usage:  python frontier_pick.py [--model ada] [--step -ode|-sde-sqrt|...]
"""
import argparse, glob, json, math, os, re

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '../../outputs/hbfm_rlf_ada_curv_tinystories_256/frontier'))
RE = re.compile(r'/(?P<model>ada|fix)_sd-1/tmax-(?P<tmax>[\d.]+)_k1(?P<step>-ode|-sde-[a-z0-9]+)?'
                r'/nfe-(?P<nfe>\d+)_t-1\.00/samples_genppl\.json$')
NFES, RANKED = [4, 8, 16, 32, 64, 128, 256], [8, 16, 32, 64, 128, 256]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--model', default='')
  ap.add_argument('--step', default=None, help="'' = em, -ode, -sde-<gt>; default: every step type")
  args = ap.parse_args()
  ladders = {}
  for path in glob.glob(f'{ROOT}/*_sd-1/tmax-*_k1*/nfe-*_t-1.00/samples_genppl.json'):
    m = RE.search(path)
    if m is None or (args.model and m['model'] != args.model):
      continue
    step = m['step'] or ''
    if args.step is not None and step != args.step:
      continue
    d = json.load(open(path))
    ladders.setdefault((m['model'], step, float(m['tmax'])), {})[int(m['nfe'])] = (
      d['gen_ppl_first_chunk_retok'], d['entropy'])
  print('| model | step | t_max | ' + ' | '.join(f'NFE {n}' for n in NFES)
        + ' | Σ log PPL (8–256) | mean H (8–256) |')
  print('|---' * (len(NFES) + 5) + '|')
  for (model, step, tmax), cells in sorted(ladders.items()):
    row = [model, step.lstrip('-') or 'em', f'{tmax:g}']
    row += [f'{cells[n][0]:.2f} @ {cells[n][1]:.3f}' if n in cells else '—' for n in NFES]
    if all(n in cells for n in RANKED):
      row += [f'{sum(math.log(cells[n][0]) for n in RANKED):.3f}',
              f'{sum(cells[n][1] for n in RANKED) / len(RANKED):.3f}']
    else:
      row += ['incomplete', '']
    print('| ' + ' | '.join(row) + ' |')


if __name__ == '__main__':
  main()
