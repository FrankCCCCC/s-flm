#!/usr/bin/env python
"""GenPPL vs sampling-step budget (NFE) for an SC-on/SC-off checkpoint pair.

Reads the sample_eval result jsons named sc{on,off}_s{STEPS}.json under --dir
(produced by `python -m main mode=sample_eval ... sampler.steps=STEPS` on the
twin checkpoints that differ only in algo.self_conditioning) and plots the two
GenPPL-vs-NFE curves with the per-budget SC gain annotated.  Discriminates the
SC mechanism: gain concentrated at low NFE = consistency/shortcut-style step
compression; gain flat across NFE = parallel-decoding coordination channel.

Example:
  python visualization/sc_vs_nfe.py --dir experiments/curv_loss_geo/nfe_sweep \
    --title 'flat HFLM (K=−0.01, c0.04, pc0.5, 20K twins)'
"""
import argparse, glob, json, os, re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(dir_, sc):
  out = {}
  for f in glob.glob(os.path.join(dir_, f'sc{sc}_s*.json')):
    steps = int(re.search(r'_s(\d+)\.json$', f).group(1))
    out[steps] = json.load(open(f))['gen_ppl_first_chunk_retok']
  return dict(sorted(out.items()))


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--dir', default='experiments/curv_loss_geo/nfe_sweep')
  p.add_argument('--title', default='')
  p.add_argument('--out', default=None,
                 help='output png (default: <dir>/sc_vs_nfe.png)')
  args = p.parse_args()
  out = args.out or os.path.join(args.dir, 'sc_vs_nfe.png')

  off, on = load(args.dir, 'off'), load(args.dir, 'on')
  steps = sorted(set(off) & set(on))
  assert steps, f'no matched scon/scoff jsons under {args.dir}'

  fig, ax = plt.subplots(figsize=(7, 5))
  ax.plot(steps, [off[s] for s in steps], 'o-', color='tab:red', label='SC off')
  ax.plot(steps, [on[s] for s in steps], 'o-', color='tab:blue', label='SC on')
  for s in steps:
    a, b = off[s], on[s]
    ax.annotate(f'{a:.1f}', (s, a), textcoords='offset points', xytext=(0, 8),
                fontsize=8, color='tab:red')
    ax.annotate(f'{b:.1f}', (s, b), textcoords='offset points', xytext=(0, -14),
                fontsize=8, color='tab:blue')
    ax.annotate(f'−{100*(a-b)/a:.0f}%', (s, (a + b) / 2), fontsize=8,
                color='0.4', ha='center')
  ax.axhline(off[steps[-1]], color='tab:red', ls=':', lw=1, alpha=0.6)
  ax.set_xscale('log')
  ax.set(xlabel='sampling steps (NFE)',
         ylabel='GenPPL (gpt2-large, retok first chunk)',
         title=f'SC gain vs step budget{" — " + args.title if args.title else ""}')
  ax.grid(alpha=0.3, which='both'); ax.legend()
  fig.tight_layout()
  fig.savefig(out, dpi=150)
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
