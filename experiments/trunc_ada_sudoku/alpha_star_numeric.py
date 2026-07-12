#!/usr/bin/env python
"""Numeric hyperbolic truncation bound alpha*(delta) for general curvature.

Same tractable model as noise_schedules.alpha_star_hyperbolic (clean
embeddings at clamped radius rho1 with iid uniform directions, origin
wrapped-normal noise at clamped radius rho0, worst impostor at angle
cos(theta) = t by the union bound), but evaluated EXACTLY on the
hyperboloid instead of via the tree approximation, so it stays valid
when c*rho1 is small (tiny inits like custom std=0.01) and for any
Gaussian curvature K = -c^2. The analytic function in
noise_schedules.py assumes K=-1 and c*rho1 >> 1; at (K=-1,
init=hyperbolic std 0.3) the tree approximation overestimates by
~0.10 (0.624 vs 0.527 numeric; c*rho1 = 6.1 is not deeply
asymptotic). Both are heuristics of the same crude model — the
scripts keep the shipped 0.624 (truncating less is the safe
direction); treat [0.53, 0.62] as the model's uncertainty band.

Geometry (unit hyperboloid after rescaling radii by c = sqrt(-K);
alpha is a fraction of arc length, hence scale-invariant):
  u0 = c*clamp(rho0), u1 = c*clamp(rho1),
  clamp(r) = rho_max * tanh(r / rho_max)
  cosh(D) = cosh(u0) cosh(u1)                    (orthogonal directions)
  geodesic X(s) = [sinh(D-s) X0 + sinh(s) X1] / sinh(D)
  cosh(r(a)) = [sinh((1-a)D) cosh(u0) + sinh(aD) cosh(u1)] / sinh(D)
  d_target(a)   = (1-a) D
  d_impostor(a) = arccosh[cosh(r(a)) cosh(u1) - sinh(r(a)) sinh(u1) t]
  alpha* = min{a : d_target(a) <= d_impostor(a)},  found by bisection.
"""
import math


def alpha_star_hyperbolic_numeric(vocab_size, dim, delta=0.1,
                                  prior_cov=0.25, embed_std=0.3,
                                  rho_max=12.0, gaussian_curvature=-1.0):
  c = math.sqrt(-gaussian_curvature)

  def clamp(r):
    return rho_max * math.tanh(r / rho_max)

  u0 = c * clamp(math.sqrt(prior_cov * dim))
  u1 = c * clamp(embed_std * math.sqrt(dim))
  t = math.sqrt(2 * math.log(2 * (vocab_size - 1) / delta) / dim)
  D = math.acosh(math.cosh(u0) * math.cosh(u1))

  def target_wins(a):
    cosh_r = (math.sinh((1 - a) * D) * math.cosh(u0)
              + math.sinh(a * D) * math.cosh(u1)) / math.sinh(D)
    sinh_r = math.sqrt(max(cosh_r ** 2 - 1, 0.0))
    arg = cosh_r * math.cosh(u1) - sinh_r * math.sinh(u1) * t
    d_imp = math.acosh(max(arg, 1.0))
    return (1 - a) * D <= d_imp

  lo, hi = 0.0, 1.0  # target_wins(1) trivially True
  for _ in range(60):
    mid = (lo + hi) / 2
    if target_wins(mid):
      hi = mid
    else:
      lo = mid
  return hi


if __name__ == '__main__':
  import sys
  sys.path.insert(0, __file__.rsplit('/experiments/', 1)[0])
  from noise_schedules import alpha_star_hyperbolic

  # Consistency check vs the shipped tree-approx bound in its regime
  tree = alpha_star_hyperbolic(12, 512)          # K=-1, std 0.3
  num = alpha_star_hyperbolic_numeric(12, 512)
  print(f'K=-1.0 std=0.30 : tree={tree:.4f}  numeric={num:.4f}  '
        f'(agree to {abs(tree - num):.4f})')

  # The sweep-best HFLM config (hflm_curv_init_lr_sudoku)
  best = alpha_star_hyperbolic_numeric(
    12, 512, embed_std=0.01, gaussian_curvature=-0.5)
  print(f'K=-0.5 std=0.01 : numeric={best:.4f}  <- ALPHA_MAX for the '
        f'hflm_ta_best arm')
