#!/usr/bin/env python
"""Random-codebook signal analysis (paper C.2 of Hyperspherical Flows, extended
to Euclidean and hyperbolic geometry) vs the measured loss-geometry curves.

Conventions: t = flow time of this repo's loss geometry (t=0 clean, t=1 noise);
the paper's alpha = 1 - t.  All three panels share one y axis, the margin ratio

    y = cos<(x_t, true-word direction) / tau,      tau = C/sqrt(d),
    C = sqrt(2 ln(2(V-1)/delta))

i.e. "how many times does your alignment with the true word beat the luckiest
of the V-1 wrong words"; y < 1 = the word is lost in the crowd, loss takes off.

  Sphere (SFM):      cos = cos(pi t / 2)                      (w ~ pi/2)
  Euclidean (EFLM):  x_t=(1-t)e+t*eps => ratio = (1-t)s / (C sqrt((1-t)^2 s^2/d
                     + t^2)), s = ||e|| (type- and token-median of the measured
                     30K norms; the norm cancels into direction inertia)
  Hyperbolic (HFLM): cos<(z_t, u_k) from the Lorentz sinh blend (angular
                     signal; see visualization/angular_cosine_vs_K.py)

Panel 4 compares the hyperbolic predicted t*(rho) (angular + pessimistic
distance-NN variant) against measured per-word t10/t50 from the cached
per-word npz.  Measured aggregate curves are overlaid in grey.  CPU-only;
all inputs are cached artifacts:

  experiments/loss_geometry_vis/tinystories/{sfm_lr3e-4,eflm_naive_geo,
    hflm_std0.04_pc1.0}/*.json          (loss_geometry.py curve caches)
  experiments/curv_loss_geo/std0.04_pc1.0_perword_30K.npz  (t10_tv_freq_len.py)
  experiments/curv_loss_geo/eflm_norms_30K.npy   (EFLM 30K embedding norms)

Example:
  python visualization/codebook_signal_vs_lossgeo.py \
    --out experiments/curv_loss_geo/codebook_signal_vs_lossgeo.png
"""
import argparse, json, os

import matplotlib, numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V, D, DELTA = 50258, 768, 0.1
RHO_MAX = 12.0


def crossing(t, g, q):
  i = np.argmax(g >= q)
  if g[i] < q: return 1.0
  if i == 0: return t[0]
  return t[i-1] + (q - g[i-1]) / max(g[i] - g[i-1], 1e-9) * (t[i] - t[i-1])


def hyp_angular(rho_k, rho_n, t):
  """cos<(z_t direction, u_k) along the Lorentz sinh-blend geodesic (K=-1)."""
  Dg = np.arccosh(np.cosh(rho_k) * np.cosh(rho_n))
  a = np.sinh((1 - t) * Dg) * np.sinh(rho_k)
  b = np.sinh(t * Dg) * np.sinh(rho_n)
  return a / np.sqrt(a * a + b * b), Dg


def hyp_distance_margin(rho_k, rho_n, tau, t):
  """Pessimistic nearest-neighbour-by-distance margin min_j d_j(t) - t*D."""
  Dg = np.arccosh(np.cosh(rho_k) * np.cosh(rho_n))
  x0 = (np.sinh((1 - t) * Dg) * np.cosh(rho_k)
        + np.sinh(t * Dg) * np.cosh(rho_n)) / np.sinh(Dg)
  rho_t = np.arccosh(np.maximum(x0, 1.0))
  grid = np.linspace(0.05, RHO_MAX * 0.9999, 400)
  cd = (np.cosh(rho_t)[:, None] * np.cosh(grid)[None, :]
        - np.sinh(rho_t)[:, None] * np.sinh(grid)[None, :] * tau)
  return np.arccosh(np.maximum(cd, 1.0)).min(1) - t * Dg


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--curves-dir', default='experiments/loss_geometry_vis/tinystories')
  p.add_argument('--perword-npz',
                 default='experiments/curv_loss_geo/std0.04_pc1.0_perword_30K.npz')
  p.add_argument('--eflm-norms',
                 default='experiments/curv_loss_geo/eflm_norms_30K.npy')
  p.add_argument('--label', default='30K', help='checkpoint label in the curve caches')
  p.add_argument('--out',
                 default='experiments/curv_loss_geo/codebook_signal_vs_lossgeo.png')
  args = p.parse_args()

  C = np.sqrt(2 * np.log(2 * (V - 1) / DELTA))
  tau = C / np.sqrt(D)
  t = np.linspace(0, 1, 401)

  def curve(rel):
    j = json.load(open(os.path.join(args.curves_dir, rel)))
    return np.array(j['t']), np.array(j['curves'][args.label])

  tm_s, L_s = curve('sfm_lr3e-4/sfm_lr3e-4.json')
  tm_e, L_e = curve('eflm_naive_geo/eflm_naive_geo.json')
  tm_h, L_h = curve('hflm_std0.04_pc1.0/hflm_std0.04_pc1.0.json')

  # sphere
  ts_star = 1 - 2 / np.pi * np.arcsin(tau)

  # euclidean: type-median + token-median measured norms
  dz = np.load(args.perword_npz)
  freq = dz['freq']
  en = np.load(args.eflm_norms)
  n = min(len(en), len(freq)); en, fq = en[:n], freq[:n]
  m = fq > 0
  s_med = np.median(en[m])
  o = np.argsort(en[m]); cw = np.cumsum(fq[m][o]) / fq[m].sum()
  s_tok = en[m][o][np.searchsorted(cw, 0.5)]
  def t_star_euc(s): return 1 / (1 + C / (s * np.sqrt(max(1 - C**2 / D, 1e-9))))

  # hyperbolic: clamped depths + per-word measured t10/t50
  eucl, cnts, sums, tg = dz['eucl'], dz['cnts'], dz['sums'], dz['t']
  rho_n = RHO_MAX * np.tanh(np.sqrt(D) / RHO_MAX)
  sel = np.where(cnts[0] >= 10)[0]
  L = sums[:, sel] / cnts[:, sel]; ok = L[-1] > 0.5; sel, L = sel[ok], L[:, ok]
  g = L / L[-1]
  t10_meas = np.array([crossing(tg, g[:, j], .1) for j in range(g.shape[1])])
  t50_meas = np.array([crossing(tg, g[:, j], .5) for j in range(g.shape[1])])
  rho_meas = RHO_MAX * np.tanh(eucl[sel] / RHO_MAX)

  def t_star_hyp_ang(rho_k):
    c, _ = hyp_angular(rho_k, rho_n, t)
    i = np.argmax(c < tau)
    return t[i] if c[i] < tau else 1.0

  def t_star_hyp_nn(rho_k):
    mgn = hyp_distance_margin(rho_k, rho_n, tau, t)
    i = np.argmax(mgn < 0)
    return t[i] if mgn[i] < 0 else 1.0

  # ---------- figure ----------
  fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))
  YLIM = (0, 5.7)

  def overlay(a, tm, Lm):
    a2 = a.twinx()
    a2.fill_between(tm, Lm / Lm[-1], color='0.55', alpha=0.25, lw=0)
    a2.set_ylabel('measured  L(t)/L(1)', color='0.35'); a2.set_ylim(0, 1.05)

  def ratio_panel(a, title):
    a.axhline(1.0, color='r', ls='--', lw=1.2, label='collapse threshold (ratio=1)')
    a.set(title=title, xlabel='t', ylabel='signal / distractor ceiling', ylim=YLIM)

  a = ax[0, 0]
  ratio_panel(a, f'Sphere (SFM)   t*={ts_star:.2f}')
  a.plot(t, np.cos(np.pi / 2 * t) / tau, label=r'$\cos(\pi t/2)/\tau$')
  a.axvline(ts_star, color='k', ls=':')
  overlay(a, tm_s, L_s)
  a.legend(fontsize=8, loc='upper right'); a.grid(alpha=0.3)

  a = ax[0, 1]
  ratio_panel(a, 'Euclidean (EFLM)')
  for lab, s in [(f'type-median ‖e‖={s_med:.1f} (rare)', s_med),
                 (f'TOKEN-median ‖e‖={s_tok:.1f} (frequent)', s_tok)]:
    r = (1 - t) * s / (C * np.sqrt((1 - t)**2 * s**2 / D + t**2))
    ln, = a.plot(t, r, label=f'{lab}  t*={t_star_euc(s):.2f}')
    a.axvline(t_star_euc(s), color=ln.get_color(), ls=':')
  overlay(a, tm_e, L_e)
  a.legend(fontsize=8, loc='upper right'); a.grid(alpha=0.3)

  a = ax[1, 0]
  ratio_panel(a, r'Hyperbolic (HFLM): cos$\angle$(z$_t$,$\hat u_k$)/$\tau$')
  for rho_k, lab in [(3.0, 'rare word  ρ=3'), (9.14, '‖e‖=12 → ρ=9.14'),
                     (11.99, '‖e‖=50 → ρ=11.99')]:
    c, _ = hyp_angular(rho_k, rho_n, t)
    ln, = a.plot(t, c / tau, label=f'{lab}  t*={t_star_hyp_ang(rho_k):.2f}')
    a.axvline(t_star_hyp_ang(rho_k), color=ln.get_color(), ls=':')
  overlay(a, tm_h, L_h)
  a.legend(fontsize=8, loc='upper right'); a.grid(alpha=0.3)

  a = ax[1, 1]
  rg = np.linspace(0.3, 11.99, 60)
  a.plot(rg, [t_star_hyp_ang(r) for r in rg], 'b-', lw=2,
         label='predicted t* (angular signal)')
  a.plot(rg, [t_star_hyp_nn(r) for r in rg], 'b--', lw=1.2,
         label='predicted t* (distance-NN, pessimistic)')
  a.scatter(rho_meas, t10_meas, s=8, alpha=0.35, color='green',
            edgecolors='none', label='measured per-word t10 (onset)')
  a.scatter(rho_meas, t50_meas, s=8, alpha=0.35, color='orange',
            edgecolors='none', label='measured per-word t50')
  a.set(title='HFLM: predicted vs measured transition vs depth',
        xlabel=r'word depth $\rho_{eff}$', ylabel='t')
  a.legend(fontsize=8); a.grid(alpha=0.3)

  fig.suptitle(f'Random-codebook signal analysis vs measured loss geometry   '
               f'(V={V}, d={D}, δ={DELTA}, C={C:.2f}, τ={tau:.3f})', fontsize=12)
  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  fig.savefig(args.out, dpi=150)
  print(f'wrote {args.out}')


if __name__ == '__main__':
  main()
