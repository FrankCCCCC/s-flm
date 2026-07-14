#!/usr/bin/env python
"""Regenerate the band-metric table in RESULTS.md from the cached
loss-geometry curves. Metrics: TV_lin = total variation of the normalized
curve's slope (= integral of |g''|), mid-mass g(0.5), area = mean g, floor.
CPU-only; reads the .json caches under experiments/loss_geometry_vis."""
import json, os

import numpy as np
from scipy.stats import spearmanr

B = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', 'loss_geometry_vis', 'tinystories')
RUNS = [  # (name, cache json, GenPPL or None, curve label or None=last ckpt)
  ('LangFlow + ada',      'lf_ada_lr1e-3/lf_ada_lr1e-3.json',              20.7, None),
  ('S-FLM + ada + trunc', 'sfm_ada_trunc_lr1e-3/sfm_ada_trunc_lr1e-3.json', 11.0, None),
  ('S-FLM + trunc',       'sfm_trunc_lr1e-3/sfm_trunc_lr1e-3.json',        12.9, None),
  ('S-FLM + ada',         'sfm_ada_lr1e-3/sfm_ada_lr1e-3.json',            20.2, None),
  ('H-FLM pc0.3',   'hflm_pc_sweep/std0.04_pcsweep.json', 29.2, 'std0.04_pc0.3, 30K'),
  ('H-FLM pc1.0',   'hflm_pc_sweep/std0.04_pcsweep.json', 17.7, 'std0.04_pc1.0, 30K'),
  ('E-FLM naive',         'eflm_naive_geo/eflm_naive_geo.json',            34.6, None),
  ('H-FLM pc0.001', 'hflm_pc_sweep/std0.04_pcsweep.json', 102.1, 'std0.04_pc0.001, 30K'),
  ('H-FLM pc0.04 (collapsed)', 'hflm_pc_sweep/std0.04_pcsweep.json', None,
   'std0.04_pc0.04, 30K'),
]


def crossing(t, g, q):
  """First t where g >= q, linearly interpolated."""
  i = np.argmax(g >= q)
  if g[i] < q: return 1.0
  if i == 0: return t[0]
  return t[i-1] + (q - g[i-1]) / max(g[i] - g[i-1], 1e-9) * (t[i] - t[i-1])


def metrics(curve, t):
  g = curve / curve[-1]
  dt = t[1] - t[0]
  gp = np.diff(g) / dt
  t10, t50, t90 = (crossing(t, g, q) for q in (0.1, 0.5, 0.9))
  return dict(tv_lin=np.abs(np.diff(gp)).sum(), mid=g[16], area=g.mean(),
              floor=curve[0], t10=t10, t50=t50, t90=t90, width=t90 - t10)


def main():
  rows = []
  for name, rel, gen, label in RUNS:
    d = json.load(open(os.path.join(B, rel)))
    if label is None:
      label = list(d['curves'].keys())[-1]
    m = metrics(np.array(d['curves'][label]), np.array(d['t']))
    rows.append((name, m, gen))
  print(f'{"run":28} {"TV_lin":>7} {"mid g(0.5)":>10} {"area":>6} '
        f'{"floor":>7} {"t10":>5} {"t50":>5} {"t90":>5} {"width":>6} '
        f'{"GenPPL":>7}')
  for name, m, gen in rows:
    print(f'{name:28} {m["tv_lin"]:7.1f} {m["mid"]:10.3f} {m["area"]:6.3f} '
          f'{m["floor"]:7.3f} {m["t10"]:5.2f} {m["t50"]:5.2f} {m["t90"]:5.2f} '
          f'{m["width"]:6.2f} {gen if gen else "collapse":>7}')
  ok = [(m, g) for _, m, g in rows if g]
  for key, sign in (('tv_lin', 1), ('mid', -1), ('area', -1)):
    r = spearmanr([sign * m[key] for m, _ in ok], [g for _, g in ok])
    print(f'Spearman({key}, GenPPL) = {r.statistic:+.3f} (p={r.pvalue:.3f})')


if __name__ == '__main__':
  main()
