#!/usr/bin/env python
"""Directional cosine of the HFLM interpolant vs curvature (random codebook).

For a word at clamped intrinsic radius rho_k and origin-wrapped-normal noise at
rho_n = rho_max*tanh(sqrt(d)/rho_max), the constant-speed Lorentz geodesic's
spatial direction is a*u_k + b*v (v = noise direction, ~orthogonal in high d):

    a = sinh((1-t) D/R) sinh(rho_k/R),   b = sinh(t D/R) sinh(rho_n/R)
    cos<(z_t, u_k) = a / sqrt(a^2 + b^2)
    cosh(D/R) = cosh(rho_k/R) cosh(rho_n/R),   R = 1/sqrt(|K|)

The signal is lost once the cosine drops below the max-of-V distractor
alignment tau = sqrt(2 ln(2(V-1)/delta) / d).  rho values are the clamped
intrinsic radii (K-independent in this codebase; K enters only through rho/R).
Draws one panel per curvature (3 word depths + rho=0.1, 0.5) and a summary of
t* / rotation width vs |K|.  Pure math -- no checkpoint, CPU-only.

Example:
  python visualization/angular_cosine_vs_K.py \
    --out experiments/curv_loss_geo/angular_cosine_vs_K.png
"""
import argparse, os

import matplotlib, numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V, D, DELTA = 50258, 768, 0.1
WORDS = [(0.1, 'ρ=0.1'), (0.5, 'ρ=0.5'), (3.0, 'rare  ρ=3'),
         (9.14, '‖e‖=12 → ρ=9.14'), (11.99, '‖e‖=50 → ρ=11.99')]
KS = [-0.01, -0.1, -0.25, -0.5, -1.0]
COLORS = ['tab:purple', 'tab:red', 'tab:blue', 'tab:orange', 'tab:green']


def cos_angle(rho_k, rho_n, K, t):
  R = 1 / np.sqrt(abs(K))
  Dr = np.arccosh(np.cosh(rho_k / R) * np.cosh(rho_n / R))  # D/R
  a = np.sinh((1 - t) * Dr) * np.sinh(rho_k / R)
  b = np.sinh(t * Dr) * np.sinh(rho_n / R)
  return a / np.sqrt(a * a + b * b)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--out',
                 default='experiments/curv_loss_geo/angular_cosine_vs_K.png')
  args = p.parse_args()

  tau = np.sqrt(2 * np.log(2 * (V - 1) / DELTA) / D)
  rho_n = 12.0 * np.tanh(np.sqrt(D) / 12.0)
  t = np.linspace(1e-4, 1 - 1e-4, 2001)

  def t_at(c, level):
    i = np.argmax(c < level)
    return t[i] if c[i] < level else 1.0

  fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
  for p_i, K in enumerate(KS):
    a = ax.reshape(-1)[p_i]
    for (rho_k, lab), col in zip(WORDS, COLORS):
      c = cos_angle(rho_k, rho_n, K, t)
      ts = t_at(c, tau)
      a.plot(t, c, color=col, label=f'{lab}  t*={ts:.2f}')
      a.axvline(ts, color=col, ls=':', lw=1)
    a.axhline(tau, color='r', ls='--', lw=1)
    a.set(title=f'K={K}   (R={1/np.sqrt(abs(K)):.2f})', xlabel='t',
          ylabel='directional cosine', ylim=(-0.02, 1.02))
    a.legend(fontsize=8); a.grid(alpha=0.3)

  a = ax.reshape(-1)[5]
  for (rho_k, lab), col in zip(WORDS, COLORS):
    tss, wid = [], []
    for K in KS:
      c = cos_angle(rho_k, rho_n, K, t)
      tss.append(t_at(c, tau))
      wid.append(t_at(c, tau) - t_at(c, 0.9))  # cos 0.9 -> tau rotation window
    a.plot([abs(K) for K in KS], tss, 'o-', color=col, label=f'{lab}  t*')
    a.plot([abs(K) for K in KS], wid, 's--', color=col, alpha=0.6,
           label=f'{lab}  width')
  a.set_xscale('log')
  a.set(title='t* (solid) and rotation width cos:0.9→τ (dashed) vs |K|',
        xlabel='|K| (log)', ylabel='t')
  a.legend(fontsize=7); a.grid(alpha=0.3)

  fig.suptitle(f'Hyperbolic angular signal vs curvature   (ρ_noise={rho_n:.2f},'
               f' τ={tau:.3f}; ρ fixed = clamped values, K enters via ρ/R)',
               fontsize=12)
  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  fig.savefig(args.out, dpi=150)
  print(f'wrote {args.out}')


if __name__ == '__main__':
  main()
