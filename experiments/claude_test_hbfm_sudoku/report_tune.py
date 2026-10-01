#!/usr/bin/env python
"""Table for the curvature x unit-rate probe grid (sweep_tune.py): board accuracy at 5k steps."""
import glob, json, os, re
REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm'
OUT = f'{REPO}/outputs/claude_test_hbfm_sudoku'
CURVS = ['-0.25', '-0.5', '-1.0', '-2.0']; UNIT_RATES = ['1', '3', '10']
acc = {}
for path in glob.glob(f'{OUT}/tune_k*_ur*_seed-*/eval/results.json'):
    m = re.match(r'tune_k(-[\d.]+)_ur([\d.]+)_seed-(\d+)', os.path.basename(os.path.dirname(os.path.dirname(path))))
    if m:
        r = json.load(open(path)); acc[(m.group(1), m.group(2))] = 100.0 * r['accuracy']
print(f'{len(acc)}/{len(CURVS) * len(UNIT_RATES)} probe cells (5k steps, LR 3e-4, seed 1, log-linear, 2000 hard boards)')
print('| K \\ unit rate | ' + ' | '.join(UNIT_RATES) + ' |'); print('|---|' + '---:|' * len(UNIT_RATES))
for k in CURVS:
    print(f'| {k} | ' + ' | '.join(f'{acc[(k, u)]:.1f}%' if (k, u) in acc else 'pending' for u in UNIT_RATES) + ' |')

# 20k-step confirmation (tune20k_*), plus the sweep.py `_rate-3` cells as references (physical rate 3)
import statistics
conf = {}
for path in glob.glob(f'{OUT}/tune20k_k*_ur*_seed-*/eval/results.json'):
    m = re.match(r'tune20k_k(-[\d.]+)_ur([\d.]+)_seed-(\d+)(_clip300)?$', os.path.basename(os.path.dirname(os.path.dirname(path))))
    if m:
        conf.setdefault((m.group(1), m.group(2), m.group(4) or ''), {})[m.group(3)] = 100.0 * json.load(open(path))['accuracy']
for path in glob.glob(f'{OUT}/lr-3e-4_ada-0_k*_seed-*_rate-3/eval/results.json'):
    m = re.match(r'lr-3e-4_ada-0_k(-[\d.]+)_seed-(\d+)_rate-3', os.path.basename(os.path.dirname(os.path.dirname(path))))
    if m:
        k = m.group(1); ur = f'{3.0 / abs(float(k)):g}'
        conf.setdefault((k, ur, ''), {})[m.group(2)] = 100.0 * json.load(open(path))['accuracy']
if conf:
    print('\n20k-step confirmation (3 seeds, LR 3e-4, log-linear; rows marked * are the sweep.py `_rate-3` cells, physical rate 3;\n'
          '|K| < 0.01 rows are the fixed-rate cells, physical rate 0.01; `clip300` = trainer.gradient_clip_val=300 control):')
    print('| K | unit rate | seed 1 | seed 2 | seed 3 | mean ± std |'); print('|---|---|---:|---:|---:|---:|')
    for (k, ur, sfx), d in sorted(conf.items(), key=lambda kv: (float(kv[0][0]), float(kv[0][1]), kv[0][2])):
        vals = [d.get(s) for s in ['1', '2', '3']]; done = [v for v in vals if v is not None]
        star = '*' if abs(float(ur) * abs(float(k)) - 3.0) < 1e-9 and (k, ur) not in {(k2, u2) for k2, u2 in [('-0.25','3')]} and len(glob.glob(f'{OUT}/tune20k_k{k}_ur{ur}_seed-*')) == 0 else ''
        summ = f'**{statistics.mean(done):.1f} ± {statistics.stdev(done):.1f}**' if len(done) >= 2 else (f'{done[0]:.1f}' if done else 'pending')
        print(f'| {k}{star}{" " + sfx.strip("_") if sfx else ""} | {ur} | ' + ' | '.join(f'{v:.1f}%' if v is not None else 'pending' for v in vals) + f' | {summ} |')
