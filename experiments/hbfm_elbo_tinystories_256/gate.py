#!/usr/bin/env python
"""Early-stopping gate of the staged cells (EXPERIMENT.md): read the stage scores
eval/nelbo_{steps}.json (exact NELBO, 10% of the validation set), print them, and list the staged
cells below the bar `max` and within `margin` of the best score; `--submit` sends those to the next
stage via sweep.py, at the site holding their checkpoints. Pull the a100 cells' scores first
(`sweep.py --site a100 --pull`).

  1/5:  python gate.py --steps 6000 --next 10000 --max 2.77 --margin 0.01 [--submit]    (staged cells only)
  1/3:  python gate.py --steps 10000 --next 30000 --max 2.46 --margin 0.005 [--submit]  (wave 1 included, keep-10000)
The bars (2026-10-31, the user's target of a final NELBO of 1.7): the fix anchor's own curve, fitted
to its keep-5000..20000 scores (1.736 + 1.150 (step/5000)^-0.673), is 2.753 at 6k and 2.457 at 10k;
a cell continues only if it is within 0.5% of that at 6k and not worse at 10k.

Scores arrive one cell at a time, so the gate is applied to what is there: a cell that passes against
the best score so far is continued, and a better score that arrives later tightens the next gate.
"""
import argparse
import json
import os
import subprocess
import sys

from sweep import ARMS, MAX_STEPS, NAME, SITES, STAGED, tag_of

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, required=True)
ap.add_argument('--next', type=int, required=True)
ap.add_argument('--max', type=float, required=True)
ap.add_argument('--margin', type=float, required=True)
ap.add_argument('--submit', action='store_true')
args = ap.parse_args()

out = f'{SITES["a6000"]["out"]}/{NAME}'
pool = STAGED if args.steps < 10000 else [(c, a) for a in ARMS for c in ARMS[a][1]]
scores = {}
for cell, arm in pool:
  path = f'{out}/{tag_of(cell, arm, 1)}/eval/nelbo_{args.steps}.json'
  if os.path.exists(path):
    scores[(cell, arm)] = json.load(open(path))['val/nll']
if not scores:
  sys.exit(f'no nelbo_{args.steps}.json yet')
best = min(scores.values())
passed = []
print(f'| cell | arm | NELBO @ {args.steps} | vs best | continue |')
print('|---|---|---:|---:|---|')
for (cell, arm), v in sorted(scores.items(), key=lambda kv: kv[1]):
  staged = (cell, arm) in STAGED
  ok = staged and v < args.max and v <= best * (1 + args.margin)
  passed += [(cell, arm)] * ok
  print(f'| {cell} | {arm} | {v:.4f} | {v / best - 1:+.2%} | '
        f'{"yes" if ok else "no" if staged else "(wave 1)"} |')
missing = [f'{c} {a}' for c, a in STAGED if (c, a) not in scores]
print(f'best {best:.4f}; bar < {args.max:g}, margin {args.margin:.1%}; no score yet: {", ".join(missing) or "none"}')
for cell, arm in passed * args.submit:
  # the next stage runs where the checkpoints are: here (desa), else on ch2263's /scratch (a100)
  site = 'a6000' if os.path.exists(f'{out}/{tag_of(cell, arm, 1)}/checkpoints/last.ckpt') else 'a100'
  subprocess.run([sys.executable, 'sweep.py', '--site', site, '--arms', arm, '--cells', cell]
                 + (['--steps', str(args.next)] if args.next < MAX_STEPS else []),
                 cwd=os.path.dirname(os.path.abspath(__file__)))
