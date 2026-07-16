#!/usr/bin/env python
"""collect.py — gather trunc_ada_sudoku results into a markdown table.

Reads outputs/trunc_ada_sudoku/tas_*/eval/{results.json,noise_state.json}
and prints the accuracy table plus the adaptive-schedule correctness
checks (has_schedule / refit_count / adapted alpha range vs ALPHA_MAX).
"""
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f'{REPO}/outputs/trunc_ada_sudoku'
ALPHA_MAX = {'eflm_ta': 0.767, 'hflm_ta_best': 0.907, 'hflm_ta_k1': 0.624,
             'hflm_to_best': 0.907, 'hflm_ao_best': None,
             'hflm_ta_umix03': 0.907, 'hflm_ta_k25': 0.894,
             'hflm_ta_e35': 0.35, 'hflm_to_e35': 0.35, 'hflm_ta_e20': 0.20,
             'hflm_ta_k10': 0.878, 'hflm_ta_k30_lr3': 0.897,
             'hflm_ta_k30_lr5': 0.897, 'hflm_ta_k25_umix03': 0.894,
             'hflm_cos2_k5': None, 'hflm_cos2_ta_k5': 0.907,
             'hflm_cos2_ta_k25': 0.894, 'hflm_ta_k5_warm10k': 0.907,
             'hflm_pc05_naive': None, 'hflm_pc05_ada': None,
             'hflm_pc01_naive': None, 'hflm_pc01_ada': None,
             'hflm_logada_k5': None, 'hflm_ta_k25_40k': 0.894}
ANCHOR = {  # naive anchors, same eval protocol (see EXPERIMENT.md)
    'eflm_ta': {'easy': 88.2, 'medium': 62.2, 'hard': 19.2},
    'hflm_ta_best': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_ta_k1': {'easy': None, 'medium': 72.9, 'hard': 40.4},
    'hflm_to_best': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_ao_best': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_ta_umix03': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_ta_k25': {'easy': None, 'medium': 71.1, 'hard': 34.6},
    'hflm_ta_e35': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_to_e35': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_ta_e20': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_ta_k10': {'easy': None, 'medium': None, 'hard': None},
    'hflm_ta_k30_lr3': {'easy': None, 'medium': 80.8, 'hard': 42.1},
    'hflm_ta_k30_lr5': {'easy': None, 'medium': 83.2, 'hard': 35.7},
    'hflm_ta_k25_umix03': {'easy': None, 'medium': 71.1, 'hard': 34.6},
    'hflm_cos2_k5': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_cos2_ta_k5': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_cos2_ta_k25': {'easy': None, 'medium': 71.1, 'hard': 34.6},
    'hflm_ta_k5_warm10k': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_pc05_naive': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_pc05_ada': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_pc01_naive': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_pc01_ada': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_logada_k5': {'easy': None, 'medium': 81.1, 'hard': 46.2},
    'hflm_ta_k25_40k': {'easy': None, 'medium': 71.1, 'hard': 34.6},
}

rows = []
for d in sorted(glob.glob(f'{OUT}/tas_*')):
    m = re.match(r'tas_(.+)_d-(easy|medium|hard)_rs(\d+)', os.path.basename(d))
    if not m:
        continue
    arm, diff, seed = m.groups()
    res, noise = f'{d}/eval/results.json', f'{d}/eval/noise_state.json'
    acc = (json.load(open(res))['accuracy'] * 100
           if os.path.exists(res) else None)
    ns = json.load(open(noise)) if os.path.exists(noise) else {}
    rows.append((arm, diff, seed, acc, ns))

print('| arm | difficulty | seed | acc % | naive anchor % | refits | '
      'adapted alpha range | alpha_max |')
print('|---|---|---|---|---|---|---|---|')
order = {'easy': 0, 'medium': 1, 'hard': 2}
for arm, diff, seed, acc, ns in sorted(
        rows, key=lambda r: (r[0], order[r[1]], r[2])):
    anchor = ANCHOR.get(arm, {}).get(diff)
    a = f'{acc:.2f}' if acc is not None else 'pending'
    adaptive = bool(ns) and ns.get('has_schedule') is not None
    rng = (f"[{ns['alpha_vals_min']:.3f}, {ns['alpha_vals_max']:.3f}]"
           if adaptive else '—')
    rc = ns.get('refit_count') if adaptive else '—'
    print(f'| {arm} | {diff} | {seed} | {a} | '
          f'{anchor if anchor is not None else "n/a"} | {rc} | {rng} | '
          f'{ALPHA_MAX.get(arm)} |')

done = sum(1 for *_, acc, _ in rows if acc is not None)
print(f'\n{done}/{len(rows)} cells complete '
      f'(cells appear once their job has started writing output)')
bad = [(a, d) for a, d, s, acc, ns in rows
       if ns and ns.get('has_schedule') is not None  # adaptive runs only
       and (not ns.get('has_schedule') or (ns.get('refit_count') or 0) < 1
            or (ALPHA_MAX.get(a) is not None
                and ns.get('alpha_vals_max', 1) > ALPHA_MAX[a] + 1e-3))]
if bad:
    print('CORRECTNESS-BAR FAILURES (no refit or alpha_max exceeded):', bad)
elif done:
    print('adaptive-schedule correctness checks pass on all completed cells')
