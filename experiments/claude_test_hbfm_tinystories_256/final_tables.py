"""Final-analysis tables for claude_test_hbfm_tinystories_256 (markdown to stdout).

Reads the same eval JSONs as report.py and prints: (A) the 30k ranking of every cell at its own sampling
horizon and at the matched physical horizon 0.5, (B) the 3-seed cells (spec / trunc_exp 0.5) as mean +- std
per checkpoint at both horizons, (C) the sampler-horizon control on the spec cell, (D) the 20k matched-horizon
seed comparison. Run on a compute node: `python final_tables.py > logs/final_tables.md`.
"""
import glob
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import EVAL_DIR, OUT, TAG, load  # noqa: E402
from sweep import t_max as default_t_max  # noqa: E402

REF = {'MDLM': (18.76, 4.39), 'DUO': (17.41, 4.34), 'S-FLM + trunc + ada': (12.95, 3.95)}


def rows():
  out = {}
  for ed in sorted(glob.glob(f'{OUT}/*/eval*')):
    tag = os.path.basename(os.path.dirname(ed))
    m, me = TAG.match(tag), EVAL_DIR.match(os.path.basename(ed))
    if not m or not me:
      continue
    gj = load(f'{ed}/samples_genppl.json')
    if gj is None:
      continue
    mode, rate, k, ub, fwd, seed = m.groups()
    fwd = fwd or 'naive'
    step = int(me.group(1)) if me.group(1) else 30000
    tm = float(me.group(2)) if me.group(2) else float(default_t_max(mode, k, ub))
    text = gj.get('text', [])
    pj = load(f'{ed}/ppl.json')
    key = (mode, float(rate), float(k), float(ub), fwd, int(seed), step, tm)
    rec = dict(gen=gj['gen_ppl_first_chunk_retok'], ent=gj['entropy'], uniq=len(set(text)), n=len(text),
               ppl=(pj or {}).get('val/ppl'), src=os.path.basename(ed))
    if key in out:  # `eval/` (final, last.ckpt) sorts before `eval_step30000_tmax0.5` (keep-30000 re-read): keep the final
      DUP.setdefault(key, []).append(rec)
      continue
    out[key] = rec
  return out


DUP = {}


def cell_of(key):
  return key[:5]          # (mode, rate, k, ub, forward_type)


def label(cell):
  mode, rate, k, ub, fwd = cell
  ubs = f'{ub:g}' if mode != 'exp' else '–'
  return f'{fwd} | {mode} | {rate:g} | {k:g} | {rate / abs(k):g} | {ubs}'


def ms(xs):
  if not xs:
    return 'n/a'
  if len(xs) == 1:
    return f'{xs[0]:.2f}'
  return f'{statistics.mean(xs):.2f} ± {statistics.stdev(xs):.2f} (n={len(xs)})'


def main():
  R = rows()
  cells = sorted({cell_of(k) for k in R},
                 key=lambda c: ({'naive': 0, 'horo': 1}.get(c[4], 9), {'exp': 0, 'trunc_exp': 1, 'unif': 2}[c[0]], c[1], -c[2], c[3]))

  def get(cell, seed, step, tm):
    return R.get(cell + (seed, step, tm))

  def seeds(cell):
    return sorted({k[5] for k in R if cell_of(k) == cell})

  own = {c: float(default_t_max(c[0], f'{c[2]:g}', f'{c[3]:g}')) for c in cells}

  print('### A. 30k ranking (final checkpoint; GenPPL @ entropy, distinct/64)\n')
  print('Sorted by GenPPL at the matched physical horizon 0.5 (own-horizon number alongside; for ub-0.5 cells both are the same sampler).\n')
  print('| fwd | mode | rate | K | unit rate | ub | own t_max | seeds | GenPPL @ own t_max | GenPPL @ t_max 0.5 | entropy @ 0.5 | distinct | val/ppl |')
  print('|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|')
  table = []
  for c in cells:
    sd = seeds(c)
    g_own = [get(c, s, 30000, own[c])['gen'] for s in sd if get(c, s, 30000, own[c])]
    r05 = [get(c, s, 30000, 0.5) or (get(c, s, 30000, own[c]) if own[c] == 0.5 else None) for s in sd]
    r05 = [r for r in r05 if r]
    g05 = [r['gen'] for r in r05]
    e05 = [r['ent'] for r in r05]
    un = min((r['uniq'] for r in r05), default=None)
    pp = [r['ppl'] for s in sd for r in [get(c, s, 30000, own[c])] if r and r['ppl'] is not None]
    sortkey = statistics.mean(g05) if g05 else 1e9
    table.append((sortkey, f'| {label(c)} | {own[c]:g} | {len(sd)} | {ms(g_own)} | {ms(g05)} | '
                           f'{ms(e05).replace("(n=%d)" % len(e05), "").strip() if e05 else "n/a"} | {un if un is not None else "n/a"}/64 | {ms(pp)} |'))
  for _, line in sorted(table):
    print(line)
  print('\nReference points (same DiT / data / steps, 3-seed means): ' +
        ', '.join(f'{k} {v[0]} @ {v[1]}' for k, v in REF.items()) + '.\n')
  if DUP:
    print('Duplicate reads of the same (cell, step, horizon) — the table uses the first (final `eval/` = last.ckpt):\n')
    for key, recs in sorted(DUP.items()):
      first = R[key]
      print(f'- {label(cell_of(key))} seed {key[5]} @ {key[6]} t_max {key[7]:g}: {first["gen"]:.2f} ({first["src"]}) vs ' +
            ', '.join(f'{r["gen"]:.2f} ({r["src"]})' for r in recs))
    print()

  print('### B. 3-seed cells per checkpoint (mean ± std over seeds)\n')
  print('| cell | horizon | 5k | 10k | 15k | 20k | 25k | 30k |')
  print('|---|---|---:|---:|---:|---:|---:|---:|')
  for c in cells:
    sd = seeds(c)
    if len(sd) < 2:
      continue
    for tm in sorted({own[c], 0.5}):
      vals = []
      for step in (5000, 10000, 15000, 20000, 25000, 30000):
        xs = [get(c, s, step, tm)['gen'] for s in sd if get(c, s, step, tm)]
        vals.append(ms(xs).replace(f' (n={len(xs)})', '') if xs else '–')
      print(f'| {label(c)} | {tm:g} | ' + ' | '.join(vals) + ' |')
  print()

  print('### C. Sampler-horizon control (spec cell, seed 1, 30k)\n')
  spec = ('exp', 3.0, -0.5, 1.0, 'naive')
  print('| t_max (physical) | unit t_max | GenPPL | entropy |')
  print('|---:|---:|---:|---:|')
  for k in sorted(k for k in R if cell_of(k) == spec and k[5] == 1 and k[6] == 30000):
    r = R[k]
    print(f'| {k[7]:g} | {k[7] * 0.5:g} | {r["gen"]:.2f} | {r["ent"]:.3f} |')
  print()

  print('### D. Matched-horizon (t_max 0.5) seed comparison, spec vs trunc_exp 0.5\n')
  print('| step | spec seeds 1/2/3 | spec mean ± std | trunc_exp seeds 1/2/3 | trunc_exp mean ± std |')
  print('|---:|---|---:|---|---:|')
  tr = ('trunc_exp', 3.0, -0.5, 0.5, 'naive')
  for step in (10000, 20000, 30000):
    a = [get(spec, s, step, 0.5) for s in (1, 2, 3)]
    b = [get(tr, s, step, 0.5) for s in (1, 2, 3)]
    fa = ' / '.join(f'{r["gen"]:.2f}' if r else '–' for r in a)
    fb = ' / '.join(f'{r["gen"]:.2f}' if r else '–' for r in b)
    print(f'| {step} | {fa} | {ms([r["gen"] for r in a if r])} | {fb} | {ms([r["gen"] for r in b if r])} |')


if __name__ == '__main__':
  main()
