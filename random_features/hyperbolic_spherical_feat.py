import math
from typing import List, Tuple, Union

import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gamma, hyp2f1

from numeric.geo_bridge import GeoUtils, HyperbolicHeatKernel
from .feature_base import FeatureBase

class SphericalFeature(FeatureBase):
  """Hyperbolic-harmonic spherical-harmonic features of the HBFM state (the
  `sphere` input map, `model.random_feature_type`).

  Per H^3 factor, the dimensionless polar state (s = rho / R, u in S^2) is
  mapped to the harmonic extensions of the real spherical harmonics,

      h_lm(z) = f_l(s) Y_lm(u),   l = 0..L, m = -l..l,   (L + 1)^2 features,

      f_l(s) = r^l 2F1(l, -1/2; l + 3/2; r^2) / 2F1(l, -1/2; l + 3/2; 1),
      r = tanh(s / 2),

  f_l being the radial solution of the H^3 Laplace equation regular at the
  origin, normalized to f_l -> 1 at the boundary. h_lm is the Poisson integral
  of Y_lm over the boundary, so it is harmonic (Delta h_lm = 0), bounded
  (|h_lm| <= max |Y_lm| = sqrt(2l + 1)) and tends to Y_lm(u) as s -> inf: a
  state pinned at the boundary (`HyperbolicBoundaryFM._clean_state`) is seen
  through its direction, where HyLa's features vanish. f_l rises from 0
  (l >= 1) at the origin to 1 past s ~ ln(2l), so the basis is a low-pass
  filter on the direction whose cutoff degree grows with the radius; the
  plain coordinates z = R tanh(s/2) u are the L = 1 case. Y_lm are the real
  harmonics, orthonormal under the uniform probability measure on S^2
  (Y_00 = 1, Y_1m = sqrt(3) u_m). f_l depends on s only: it is tabulated at
  construction (scipy's 2F1 on a grid of step `TABLE_STEP` up to `TABLE_MAX`,
  past which f_l = 1 to double precision) and linearly interpolated.

  f_0 = 1 and every f_l saturates at a small radius, so the trunk, which has
  no time conditioning, is also given the radial channels tanh(s / c) for c in
  `radial_scales`: `num_features` = `num_factors` * ((L + 1)^2 + len(radial_scales)).
  `mc_samples` = M (null = off) replaces the harmonics' value by their Monte-Carlo
  Poisson integral over M fresh boundary angles w ~ P(z, .) = (cosh s - sinh s <u, w>)^{-2},
  the Poisson kernel being its own importance proposal (every weight is 1): the mean of
  Y_lm(w) is unbiased for f_l(s) Y_lm(u), bounded by sqrt(2l + 1), and collapses onto
  Y_lm(u) at the boundary, where the kernel does; the gradient is the closed form's and
  the radial channels stay exact. The module adds no checkpoint keys.

  `radial_type` names the radial part of the map: `exp_decay` is the f_l above with the
  tanh channels; `radial` notes why a flat f_l = 1 was removed. `phase` keeps the harmonics
  f_l Y_lm and replaces the tanh channels by the boundary-phase family of the
  radial-eigenfunction design note (experiments/claude_test_hbfm_rlf_tinystories_256/
  HBFM Radial Eigenfunctions.html): per scale c, mu = 2 + i / c, the real and imaginary
  parts of Y_lm(u) r_{mu,l}(s) / C(mu, l), l = 0..L, r_{mu,l} being the radial solution
  of Delta f = mu (mu - 2) f regular at the origin (r ~ s^l) and
  C(mu, l) = (2l + 1)!! Gamma(mu - 1) / Gamma(mu + l) its boundary amplitude: r / C ->
  e^{i s / c} at the boundary (modulus 1, the pin stays visible), and mu = 2 (c -> inf)
  is f_l. Tabulated by the float64 ODE route (`phase_radial`); `num_features` =
  `num_factors` * (L + 1)^2 * (1 + 2 len(radial_scales)). Exclusive with `mc_samples`.
  """
  RADIAL_TYPES = ('exp_decay', 'phase')
  TABLE_STEP = 1.0 / 256
  TABLE_MAX = 36.0        # > 2 atanh(1 - 8 eps) = 34.66, the readback's largest s

  def __init__(
    self,
    num_factors: int,
    degree: int,
    radial_scales: List[float],
    radial_type: str="exp_decay",
    mc_samples=None
  ):
    super().__init__()
    if degree is None or int(degree) < 0 or not radial_scales:
        raise ValueError('model.sphere_degree must be >= 0 and '
                        'model.sphere_radial_scales non-empty.')
    if mc_samples is not None and int(mc_samples) < 1:
        raise ValueError(f'model.sphere_mc_samples={mc_samples} must be null or an int >= 1.')
    if radial_type not in self.RADIAL_TYPES:
        raise ValueError(f'model.sphere_radial_type={radial_type} must be one of {self.RADIAL_TYPES}.')
    if radial_type == 'phase' and mc_samples is not None:
        raise ValueError("model.sphere_radial_type='phase' has no Monte-Carlo form (sphere_mc_samples).")
    self.num_factors = int(num_factors)
    self.degree = degree = int(degree)
    self.mc_samples = None if mc_samples is None else int(mc_samples)
    self.radial_type = radial_type
    s = np.arange(0.0, self.TABLE_MAX + self.TABLE_STEP, self.TABLE_STEP)
    r = np.tanh(s / 2)
    table = np.stack([
        r ** l * hyp2f1(l, -0.5, l + 1.5, r * r)
        / (math.gamma(l + 1.5) / (math.gamma(1.5) * math.gamma(l + 2)))   # Gauss: 2F1(..; 1)
        for l in range(degree + 1)], axis=-1)
    table = np.concatenate([table, table[-1:]], axis=0)   # one spare row for the interpolation
    self.register_buffer('table', torch.tensor(table, dtype=torch.float64),
                         persistent=False)
    if radial_type == 'phase':
        phase = np.stack([np.stack([self.phase_radial(2 + 1j / c, l, s) for l in range(degree + 1)], -1)
                          for c in radial_scales], -2)             # (N, |c|, L + 1), complex
        phase = np.concatenate([phase, phase[-1:]], axis=0)
        self.register_buffer('phase', torch.tensor(np.stack([phase.real, phase.imag], -1)),
                             persistent=False)                    # (N + 1, |c|, L + 1, 2)
    self.register_buffer('ell_of_feature', torch.tensor(
        [l for l in range(degree + 1) for _ in range(2 * l + 1)]), persistent=False)
    self.register_buffer('radial_scales', torch.tensor(
        [float(c) for c in radial_scales], dtype=torch.float64), persistent=False)

  @property
  def num_features(self) -> int:
    """The number of features produced by this module, over all product factors."""
    num_radial = (2 * len(self.radial_scales) * (self.degree + 1) ** 2 if self.radial_type == 'phase'
                  else len(self.radial_scales))
    return self.num_factors * ((self.degree + 1) ** 2 + num_radial)

  def _lerp(self, table: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """A radial table linearly interpolated at radii s (...) -> (..., *table.shape[1:])."""
    x = (s / self.TABLE_STEP).clamp(0.0, table.shape[0] - 2)
    i = x.floor().long()
    w = (x - i).reshape(*x.shape, *[1] * (table.dim() - 1))
    return table[i] * (1 - w) + table[i + 1] * w

  def exp_decay_radial(self, s: torch.Tensor) -> torch.Tensor:
    r"""
    $\mu=2$, with eigenvalue $\eta = \mu (\mu - 2) = 0$

    f_l(s), l = 0..L, for dimensionless radii s (...) -> (..., L + 1).
    """
    return self._lerp(self.table, s)

  @staticmethod
  def phase_radial(mu, l, s, s0=0.01):
    r"""
    $\mu = 2 + i / c$ per radial scale $c$, with eigenvalue $\eta = \mu (\mu - 2) = -1 / c^2 + 2 i / c$

    r_{mu,l}(s) / C(mu, l) on the ascending numpy grid s, complex128: the design note's
    ODE route. g = sinh(s) r solves g'' = [(mu - 1)^2 + l (l + 1) / sinh^2 s] g, started at s0
    from the series r = s^l (1 + a s^2) and integrated by DOP853 (the regular solution
    dominates forward for Re mu >= 1, so this is stable); below s0 the series itself.
    """
    a = (mu * (mu - 2) - l * (l + 3) / 3) / (4 * l + 6)
    r0 = s0 ** l * (1 + a * s0 ** 2)
    dr0 = l * s0 ** (l - 1) * (1 + a * s0 ** 2) + 2 * a * s0 ** (l + 1) if l else 2 * a * s0
    sol = solve_ivp(lambda t, y: [y[1], ((mu - 1) ** 2 + l * (l + 1) / np.sinh(t) ** 2) * y[0]],
                    (s0, s[-1]), [np.sinh(s0) * r0 + 0j, np.cosh(s0) * r0 + np.sinh(s0) * dr0 + 0j],
                    method='DOP853', rtol=1e-12, atol=0, dense_output=True)
    t = np.maximum(s, s0)
    r = np.where(s < s0, s ** l * (1 + a * s * s), sol.sol(t)[0] / np.sinh(t))
    return r * gamma(mu + l) / (math.prod(range(1, 2 * l + 2, 2)) * gamma(mu - 1))

  def radial(self, s: torch.Tensor, radial_type: str="exp_decay") -> torch.Tensor:
    """The radial table `radial_type` at dimensionless radii s: `exp_decay` is f_l(s), l = 0..L,
    (...) -> (..., L + 1); `phase` is Re and Im of r_{2+i/c,l}(s) / C, (...) -> (..., |c|, L + 1, 2)."""
    if radial_type == "exp_decay":
      return self.exp_decay_radial(s=s)

      r"""
      Note: Why constant radial $f_l(s) = 1$ brings explosion in gradient?
      Let's say the feature is h_lm(z) = f_l(s) · Y_lm(u)
      and z = \kappa * tanh(s/2) · u, with \kappa = 1/sqrt(|K|)
      while taking gradient of h_lm(z) w.r.t. z with product rule, we have

      $$dh_lm(z) / dz = (df_l(s) / ds) * (ds / dz) * Y_lm(u) + f_l(s) * (dY_lm(u) / du) * (du / dz)$$

      Then, $\frac{ds}{dz}$ and $\frac{du}{dz}$ are given by the following equations:
      $$\frac{ds}{dz} = \frac{2/\kappa}{1 - \| z \|_2^2 / \kappa^2} u^\top$$
      $$\frac{du}{dz} = \frac{I - uu^\top}{\| z \|_2 }$$

      However, near the origin, $\| z \|_2 \approx \kappa * s / 2$, so $\| z \|_2$ close to 0, 
      causing $\frac{du}{dz}$ become very large, which leads to the explosion in gradient.

      However, if $f_l(s)$ can cancel out the explosion in $\frac{du}{dz}$, then the gradient will be stable.
      In the case of constant radial, $f_l(s) = 1$, so it cannot cancel out the explosion in $\frac{du}{dz}$,
      which leads to the explosion in gradient.
      exp_decay does cancel it: $f_l(s) \sim s^l$ near the origin, and $l = 0$ has no $du/dz$ term as $Y_{00}$ is constant.

      Measured $\| d(c \cdot h) / dz \|$ for a fixed random $c$, degree 4, K = -1:
          s           1     0.1    0.01    1e-3    1e-6    0
          exp_decay   4.7   3.4    3.5     3.5     3.5     0
          constant    31    289    2884    2.9e4   2.9e7   inf

      In training this did not diverge. Sudoku, (H^3)^32, K = -0.5, one seed, 2000 steps, same batches, 0.02% of
      the free positions at s < 0.01: no NaN in either arm; largest total gradient norm 13.1 (constant) vs 2.13
      (exp_decay); loss over the last 250 steps 0.0257 vs 0.0062. So the cost seen was spikes and slower learning.
      """
    elif radial_type == "phase":
      return self._lerp(self.phase, s)
    else:
      raise ValueError(f'Unknown radial_type {radial_type}.')

  def spherical_harmonics(self, u: torch.Tensor) -> torch.Tensor:
    """Real Y_lm(u) of unit vectors u (..., 3) -> (..., (L + 1)^2), l = 0..L,
    m = -l..l, orthonormal under the uniform probability measure on S^2:
    Y_lm = N_lm dP_l^m(z) Re (x + iy)^m, Y_l,-m = N_lm dP_l^m(z) Im (x + iy)^m
    (m > 0) and Y_l0 = N_l0 P_l(z), with dP_l^m = d^m P_l / dz^m (the
    associated Legendre function without its (1 - z^2)^{m/2}, which (x + iy)^m
    carries) by the standard recursions, no Condon-Shortley phase."""
    x, y, z = u.unbind(-1)
    cols = {}
    re, im = torch.ones_like(x), torch.zeros_like(x)     # (x + iy)^m
    for m in range(self.degree + 1):
        if m > 0:
            re, im = x * re - y * im, x * im + y * re
        p_prev, p = None, torch.full_like(x, float(math.prod(range(1, 2 * m, 2))))  # dP_m^m = (2m - 1)!!
        for l in range(m, self.degree + 1):
            if l == m + 1:
                p_prev, p = p, (2 * m + 1) * z * p
            elif l > m + 1:
                p_prev, p = p, ((2 * l - 1) * z * p - (l + m - 1) * p_prev) / (l - m)
            norm = math.sqrt((2 * l + 1) * math.factorial(l - m) / math.factorial(l + m))
            if m == 0:
                cols[(l, 0)] = norm * p
            else:
                cols[(l, m)] = math.sqrt(2) * norm * p * re
                cols[(l, -m)] = math.sqrt(2) * norm * p * im
    return torch.stack([cols[(l, m)] for l in range(self.degree + 1)
                        for m in range(-l, l + 1)], dim=-1)

  def forward(
    self,
    z: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    input_coord: str,
    prod_factor_dim: List[int],
    prod_factor_gaussian_curvature: List[float],
  ) -> torch.Tensor:
    """Map a state to its hyperbolic-harmonic features, concatenated over the product factors.

    Args:
      z (`torch.Tensor` or `Tuple[torch.Tensor, torch.Tensor]`): the state in the
        coordinates `input_coord`, see `FeatureBase.forward`; the Cartesian bridge
        state has shape `(B, L, sum(d_i))`.
      input_coord (`str`): the `Coordinate` tag of `z`.
      prod_factor_dim (`List[int]`): the dimension of each factor; all must be 3, and
        there must be `num_factors` of them.
      prod_factor_gaussian_curvature (`List[float]`): the curvature of each factor,
        which sets the dimensionless radius s = rho / R of this call.

    Returns:
      `torch.Tensor` of shape `(B, L, num_features)`, float64: per factor, h_lm(z)
      for l = 0..L, m = -l..l (their Monte-Carlo Poisson integrals if `mc_samples`
      is set), then tanh(s / c) per radial scale c, or (`phase`) Re and Im of
      Y_lm(u) r_{2+i/c,l}(s) / C per scale.
    """
    if any(d != 3 for d in prod_factor_dim):
      raise ValueError(
        f'the sphere map needs H^3 factors (harmonics on S^2); got {prod_factor_dim}.')
    if len(prod_factor_dim) != self.num_factors:
      raise ValueError(f'this map was built for {self.num_factors} factors; got {prod_factor_dim}.')
    rhos, us = super().forward(z, input_coord, prod_factor_dim, prod_factor_gaussian_curvature)
    us = us.unflatten(-1, (len(prod_factor_dim), 3))           # (B, L, m, 3)
    kappas = rhos.new_tensor(
      [1.0 / GeoUtils._curvature_scale(k) for k in prod_factor_gaussian_curvature])
    ss = rhos * kappas                                          # (B, L, m)
    Y = self.spherical_harmonics(us)                            # (B, L, m, (L+1)^2)
    harmonics = Y * self.radial(ss)[..., self.ell_of_feature]   # f_l for either radial_type
    if self.mc_samples is not None:
        # Monte Carlo over the boundary angle, w ~ P(z, .) exactly (unit importance
        # weights): no weight to spike near the boundary. The draws carry no graph;
        # the gradient is the closed form's (straight-through).
        w = HyperbolicHeatKernel._reflect_to_target(
            HyperbolicHeatKernel._angular_boost(
            ss.detach().unsqueeze(-1).expand(*ss.shape, self.mc_samples), 3),
            us.detach().unsqueeze(-2))                                 # (B, L, m, M, 3)
        w = w / w.norm(dim=-1, keepdim=True)   # the boost's c0 == -1 rounding at large s
        mc = self.spherical_harmonics(w).mean(-2)                    # (B, L, m, (L+1)^2)
        harmonics = harmonics + (mc - harmonics).detach()
    if self.radial_type == 'phase':                                  # (B, L, m, |c| (L+1)^2 2)
      channels = (Y[..., None, :, None]
                  * self.radial(ss, radial_type='phase')[..., self.ell_of_feature, :]).flatten(-3)
    else:
      channels = torch.tanh(ss.unsqueeze(-1) / self.radial_scales)   # (B, L, m, |c|)
    return torch.cat([harmonics, channels], dim=-1).flatten(-2)