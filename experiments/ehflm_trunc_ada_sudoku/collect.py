#!/usr/bin/env python
"""collect.py — gather ehflm_trunc_ada_sudoku results.

Reads outputs/ehflm_trunc_ada_sudoku/*/eval/results.json, writes
all_results.csv (one row per finished cell) and prints a per-config
mean +- std table (aggregated over seeds) for RESULTS.md.
"""
import csv
import glob
import json
import os
import re
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f'{REPO}/outputs/ehflm_trunc_ada_sudoku'
EXP = os.path.dirname(os.path.abspath(__file__))
TAG = re.compile(r'^(eflm|hflm)-(to|ta)(?:_k(-[0-9.]+)_i-([a-z0-9.]+))?'
                 r'_lr([0-9e.-]+)_d-(easy|medium|hard)_rs(\d+)$')

rows = []
for d in sorted(glob.glob(f'{OUT}/*')):
    m = TAG.match(os.path.basename(d))
    if not m:
        continue
    model, method, k, init, lr, diff, seed = m.groups()
    res = f'{d}/eval/results.json'
    acc = (json.load(open(res))['accuracy'] * 100
           if os.path.exists(res) else None)
    rows.append({'model': model, 'method': method, 'k': k or '', 'init': init or '',
                 'lr': lr, 'difficulty': diff, 'seed': seed, 'acc': acc})

done = [r for r in rows if r['acc'] is not None]
with open(f'{EXP}/all_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
    w.writeheader()
    w.writerows(done)
print(f'{len(done)}/{len(rows)} started cells finished '
      f'(cells appear once their job creates the output dir); '
      f'csv -> {EXP}/all_results.csv\n')

configs = {}
for r in done:
    key = (r['model'], r['method'], r['k'], r['init'], r['lr'], r['difficulty'])
    configs.setdefault(key, []).append(r['acc'])

print('| model | method | K | init | lr | difficulty | acc mean±std (n seeds) |')
print('|---|---|---|---|---|---|---|')
for key in sorted(configs):
    accs = configs[key]
    sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print(f'| {key[0]} | {key[1]} | {key[2] or "—"} | {key[3] or "ngpt"} | '
          f'{key[4]} | {key[5]} | {statistics.mean(accs):.2f} ± {sd:.2f} '
          f'({len(accs)}) |')
