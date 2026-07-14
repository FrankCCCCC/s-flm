#!/usr/bin/env python
"""Grouped per-word loss-geometry curves: by embedding-length quintile and by
word count (frequency), for one HFLM checkpoint.

Reuses t10_tv_freq_len.compute_perword (same hook/seed/batches as
loss_geometry.py) and its <out>_perword_{step}.npz cache. Two figures:

  <out>_timing_by_length.png  per-word normalized curves f_v = L_v / L_v(1),
                              averaged within embedding-length quintiles
                              (words with count >= --min-count for stable
                              per-word curves); legend shows each quintile's
                              length range and median t50. Log y.
  <out>_timing_by_freq.png    token-weighted group curves L(t)/L(1) for count
                              bins {1, 2-4, 5-9, 10-49, 50-199, >=200} over
                              ALL occurring words. Log y.

Each figure is also written with a linear y axis (suffix _linear), same
settings otherwise.

Example:
  python visualization/perword_loss_geo_by_len_freq.py \
    --project outputs/hflm_sweep_tinystories_s256 --run std0.04_pc1.0 \
    --step 30000 --out experiments/curv_loss_geo/std0.04_pc1.0
"""
import argparse, os, sys

import matplotlib, numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loss_geometry import checkpoint_training_step2tag  # noqa: E402
from t10_tv_freq_len import compute_perword  # noqa: E402

COUNT_BINS = [(1, 1), (2, 4), (5, 9), (10, 49), (50, 199), (200, 10**9)]


def _crossing(t, fv, q):
  """First t where fv >= q, linearly interpolated."""
  i = np.argmax(fv >= q)
  if fv[i] < q: return 1.0
  if i == 0: return t[0]
  return t[i-1] + (q - fv[i-1]) / max(fv[i] - fv[i-1], 1e-9) * (t[i] - t[i-1])


def plot_by_length(t, sums, cnts, eucl, out, min_count, tag, run, logy=True):
  cnt = cnts[0]
  sel = np.where(cnt >= min_count)[0]
  L = sums[:, sel] / cnts[:, sel]
  ok = L[-1] > 0.5
  sel, L = sel[ok], L[:, ok]
  f = L / L[-1]  # normalized only for the t50 timing marker
  t50 = np.array([_crossing(t, f[:, j], 0.5) for j in range(f.shape[1])])
  length = eucl[sel]
  q = np.quantile(length, [0, .2, .4, .6, .8, 1.])
  fig, ax = plt.subplots(figsize=(6.5, 4.5))
  for i in range(5):
    m = (length >= q[i]) & (length <= q[i+1])
    ax.plot(t, L[:, m].mean(1), marker='o', ms=3,
            label=f'len {q[i]:.1f}-{q[i+1]:.1f} '
                  f'(n={int(m.sum())}, t50={np.median(t50[m]):.2f})')
  ax.set(xlabel='t', ylabel='Loss',
         title=f'HFLM {run} {tag}: per-word loss timing by length quintile')
  if logy:
    ax.set_yscale('log')
  ax.grid(alpha=0.3); ax.legend(fontsize=7)
  fig.tight_layout()
  path = f'{out}_timing_by_length{"" if logy else "_linear"}.png'
  fig.savefig(path, dpi=150); plt.close(fig)
  print(f'wrote {path}')


def plot_by_freq(t, sums, cnts, out, tag, run, logy=True):
  cnt = cnts[0]
  fig, ax = plt.subplots(figsize=(6.5, 4.5))
  for lo, hi in COUNT_BINS:
    m = (cnt >= lo) & (cnt <= hi)
    g = sums[:, m].sum(1) / np.maximum(cnts[:, m].sum(1), 1)
    if g[-1] <= 0: continue
    ax.plot(t, g, marker='o', ms=3,
            label=f'cnt {lo}-{hi if hi < 10**9 else "max"} (n={int(m.sum())})')
  ax.set(xlabel='t', ylabel='Loss',
         title=f'HFLM {run} {tag}: group curves by word count (frequency)')
  if logy:
    ax.set_yscale('log')
  ax.grid(alpha=0.3); ax.legend(fontsize=7)
  fig.tight_layout()
  path = f'{out}_timing_by_freq{"" if logy else "_linear"}.png'
  fig.savefig(path, dpi=150); plt.close(fig)
  print(f'wrote {path}')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--project', required=True)
  p.add_argument('--run', required=True)
  p.add_argument('--step', type=int, required=True)
  p.add_argument('--out', required=True, help='output prefix (no extension)')
  p.add_argument('--min-count', type=int, default=10,
                 help='min occurrences for stable per-word curves '
                      '(length-quintile figure only)')
  p.add_argument('--num-batches', type=int, default=8)
  p.add_argument('--batch-size', type=int, default=16)
  p.add_argument('--freq-batches', type=int, default=512)
  p.add_argument('--cache-dir',
                 default='/share/thickstun/sychou/workspace/research/s-flm/data_cache')
  args = p.parse_args()

  tag = checkpoint_training_step2tag(args.step)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  cache = f'{args.out}_perword_{tag}.npz'
  if os.path.exists(cache):
    d = np.load(cache)
    t, sums, cnts, eucl = d['t'], d['sums'], d['cnts'], d['eucl']
  else:
    t, sums, cnts, eucl, freq = compute_perword(
      os.path.join(args.project, args.run), args.step, args)
    np.savez_compressed(cache, t=t, sums=sums, cnts=cnts, eucl=eucl, freq=freq)
    print(f'wrote {cache}')

  plot_by_length(t, sums, cnts, eucl, args.out, args.min_count, tag, args.run)
  plot_by_length(t, sums, cnts, eucl, args.out, args.min_count, tag, args.run,
                 logy=False)
  plot_by_freq(t, sums, cnts, args.out, tag, args.run)
  plot_by_freq(t, sums, cnts, args.out, tag, args.run, logy=False)


if __name__ == '__main__':
  main()
