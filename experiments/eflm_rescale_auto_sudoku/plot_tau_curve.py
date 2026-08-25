#!/usr/bin/env python
"""Sudoku-hard solve rate vs the autonomous-clock horizon tau_max.

One panel per embedding norm R; one line per loss weight (ce / snr), with the
log-linear control as a horizontal band. The dotted vertical is tau*(R) =
-log b*(R), the decode point of the random-codebook analysis: the flow has to
run past it, and everything beyond it is spent on already-decided tokens.

Usage:  python plot_tau_curve.py [--out tau_curve.png]
"""
import argparse
import math

import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze import BASE, BASE_TAG, OUT, RHOS, TAG, group, load

C = math.sqrt(2 * math.log(2 * (12 - 1) / 0.1))  # V=12, delta=0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tau_curve.png')
    args = ap.parse_args()

    runs = [r for r in load('eflmra_*', TAG, OUT)
            if r['lr'] == '1e-3' and r['arm'] == 'naive']
    auto = group([r for r in runs if r['clock'] == 'auto'],
                 lambda r: (r['R'], r['w'], float(r['tau'])))
    ll = group([r for r in runs if r['clock'] == 'll'],
               lambda r: (r['R'], r['w']))
    base = group([r for r in load('eflmrs_*', BASE_TAG, BASE)
                  if r['arm'] == 'naive'], lambda r: (r['R'], r['lr']))

    rhos = [R for R in RHOS if any(k[0] == R for k in auto)]
    fig, axes = plt.subplots(1, len(rhos), figsize=(5 * len(rhos), 4),
                             squeeze=False, sharey=True)
    for ax, R in zip(axes[0], rhos):
        for w, style in (('ce', 'o-'), ('snr', 's--')):
            pts = sorted((t, 100 * st.mean(v),
                          100 * (st.pstdev(v) if len(v) > 1 else 0))
                         for (r_, w_, t), v in auto.items()
                         if r_ == R and w_ == w)
            if pts:
                x, y, e = zip(*pts)
                ax.errorbar(x, y, yerr=e, fmt=style, capsize=3,
                            label=f'autonomous, {w}')
        for w, c in (('ce', 'k'), ('snr', 'gray')):
            if (R, w) in ll:
                v = 100 * st.mean(ll[(R, w)])
                ax.axhline(v, color=c, ls=':', lw=1.5,
                           label=f'log-linear, {w} (same seed)')
        b = base.get((R, '1e-3'), [])
        if b:
            m = 100 * st.mean(b)
            ax.axhspan(m - 5, m + 5, color='tab:blue', alpha=.08)
            ax.axhline(m, color='tab:blue', lw=1, alpha=.5,
                       label='log-linear 3-seed mean')
        tau_star = -math.log(1 / (1 + C / float(R)))
        ax.axvline(tau_star, color='tab:red', ls='-.', lw=1)
        ax.annotate(r'$\tau^*(R)$', (tau_star, 2), color='tab:red',
                    fontsize=9, rotation=90, va='bottom', ha='right')
        ax.set(xscale='log', xlabel=r'$\tau_{\max}$', title=f'R = {R}')
        ax.grid(alpha=.3)
    axes[0][0].set_ylabel('sudoku hard: solved (%)')
    axes[0][-1].legend(fontsize=8, loc='upper right')
    fig.suptitle('Autonomous-clock E-FLM on sudoku hard: solve rate vs clock '
                 r'horizon $\tau_{\max}$ (lr 1e-3, mean $\pm$ sd over up to '
                 '3 seeds)')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
