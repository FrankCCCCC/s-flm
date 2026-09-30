"""Poincare-ball <-> polar conversions in geo_bridge.GeoUtils, single ball and
product manifold: the inverse pair `poincare_cartesian_to_hyperbolic_polar` /
`hyperbolic_polar_to_poincare_cartesian` and their `*_prod` forms.

CPU-only. The invariant: the inverse recovers `(rho, u)` of everything the
forward map emits to an absolute error of ~eps R e^{rho/R} / 2 (the ball
encodes rho through 1 - ||z||/R ~ 2 e^{-rho/R}), i.e. to float precision
for rho <~ 18 R in float64 / ~8 R in float32; radii past the forward map's
`1 - eps` tanh cap read back as the inverse's single cap value
`2R atanh(1 - 8 eps)` (~34.66 R in float64), never inf / NaN.
"""
import math

import pytest
import torch

from numeric.geo_bridge import Coordinate, GeoUtils, HyperbolicHeatKernel
from conftest import REPO_ROOT  # noqa: F401

torch.manual_seed(0)


def _unit(*shape, dtype=torch.float64):
  u = torch.randn(*shape, dtype=dtype)
  return u / u.norm(dim=-1, keepdim=True)


@pytest.mark.parametrize('K', [-0.25, -1.0, -4.0])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_single_ball_round_trip_and_cap(K, dtype):
  R = 1.0 / math.sqrt(-K)
  eps = torch.finfo(dtype).eps
  rho = torch.tensor([0.0, 1e-4, 1e-2, 1.0, 3.0 * R, 8.0 * R, 16.0 * R, 30.0 * R], dtype=dtype)
  u = _unit(8, 5, dtype=dtype)
  z = GeoUtils.hyperbolic_polar_to_poincare_cartesian(rho, u, gaussian_curvature=K)
  assert (z.norm(dim=-1) < R).all()
  rho_back, u_back = GeoUtils.poincare_cartesian_to_hyperbolic_polar(z, gaussian_curvature=K)
  assert rho_back.dtype == dtype and u_back.dtype == dtype
  # Absolute error bound of the ball representation: eps R e^{rho/R} / 2 (x4 slack).
  bound = 4 * eps * R * torch.exp(rho / R) / 2 + 4 * eps * rho
  assert ((rho_back - rho).abs() <= bound).all(), (rho_back - rho, bound)
  assert torch.allclose(u_back[1:], u[1:], atol=8 * eps)
  # The origin: rho = 0 and a zero direction (no NaN).
  assert rho_back[0] == 0 and torch.equal(u_back[0], torch.zeros(5, dtype=dtype))
  # Past the forward map's cap everything reads back as the inverse's single cap
  # value, whatever the direction (2000 of them), finite.
  cap = 2.0 * R * math.atanh(1.0 - 8 * eps)
  far = GeoUtils.hyperbolic_polar_to_poincare_cartesian(
    torch.full((2000,), 1e6, dtype=dtype), _unit(2000, 5, dtype=dtype), gaussian_curvature=K)
  rho_far, _ = GeoUtils.poincare_cartesian_to_hyperbolic_polar(far, gaussian_curvature=K)
  assert torch.allclose(rho_far, torch.full_like(rho_far, cap), rtol=1e-6)
  # A point exactly on / beyond the boundary hits the clamp: exactly the cap.
  rho_b, _ = GeoUtils.poincare_cartesian_to_hyperbolic_polar(
    torch.stack([R * u[0], 1.5 * R * u[1]]), gaussian_curvature=K)
  assert torch.allclose(rho_b, torch.full_like(rho_b, cap), rtol=1e-6)


def test_single_ball_matches_lorentz_route():
  K = -2.0
  rho = torch.rand(4, 7, dtype=torch.float64) * 6   # rho / R <= 8.5: both routes exact here
  z = GeoUtils.hyperbolic_polar_to_poincare_cartesian(rho, _unit(4, 7, 3), gaussian_curvature=K)
  rho_a, u_a = GeoUtils.poincare_cartesian_to_hyperbolic_polar(z, gaussian_curvature=K)
  rho_b, u_b = GeoUtils.lorentz_cartesian_to_hyperbolic_polar(
    GeoUtils.poincare_cartesian_to_lorentz_cartesian(z, gaussian_curvature=K), gaussian_curvature=K)
  assert torch.allclose(rho_a, rho_b, rtol=1e-8, atol=1e-8)
  assert torch.allclose(u_a, u_b, atol=1e-12)


def test_gradient_is_finite_through_the_inverse():
  z = GeoUtils.hyperbolic_polar_to_poincare_cartesian(
    torch.tensor([1e-3, 2.0, 100.0], dtype=torch.float64), _unit(3, 4)).requires_grad_(True)
  rho, u = GeoUtils.poincare_cartesian_to_hyperbolic_polar(z)
  (rho.sum() + u.sum()).backward()
  assert torch.isfinite(z.grad).all()


def test_prod_round_trip_unequal_factors():
  dims, curvs = [2, 3, 4], [-1.0, -4.0, -0.25]
  B, L = 3, 5
  rhos = torch.rand(B, L, 3, dtype=torch.float64) * 8
  thetas = torch.cat([_unit(B, L, d) for d in dims], dim=-1)
  z = GeoUtils.hyperbolic_polar_to_poincare_cartesian_prod(rhos, thetas, dims, curvs)
  assert z.shape == (B, L, 9)
  # per-factor single-ball conversion, concatenated
  expected = torch.cat([
    GeoUtils.hyperbolic_polar_to_poincare_cartesian(rhos[..., i], th, gaussian_curvature=K)
    for i, (th, K) in enumerate(zip(thetas.split(dims, dim=-1), curvs))], dim=-1)
  assert torch.equal(z, expected)
  rhos_back, thetas_back = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(z, dims, curvs)
  assert rhos_back.shape == (B, L, 3) and thetas_back.shape == (B, L, 9)
  assert torch.allclose(rhos_back, rhos, rtol=1e-9, atol=1e-9)
  assert torch.allclose(thetas_back, thetas, atol=1e-12)


def test_prod_defaults_to_the_single_ball():
  rhos = torch.rand(6, 1, dtype=torch.float64) * 3
  thetas = _unit(6, 8)
  z = GeoUtils.hyperbolic_polar_to_poincare_cartesian_prod(rhos, thetas)
  assert torch.equal(z, GeoUtils.hyperbolic_polar_to_poincare_cartesian(rhos[..., 0], thetas))
  rhos_back, thetas_back = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(z)
  assert rhos_back.shape == (6, 1)
  assert torch.allclose(rhos_back[..., 0], rhos[..., 0], rtol=1e-9, atol=1e-9)
  assert torch.allclose(thetas_back, thetas, atol=1e-12)


def test_prod_matches_poincare_bridge_prod_layout():
  """With the same seed, poincare_bridge_prod's CARTESIAN output is the forward
  prod conversion of its HYPERBOLIC_POLAR output, and the inverse recovers it."""
  dims, curvs = [2, 3, 3], [-1.0, -4.0, -0.25]
  V, E = 7, 8
  emb = torch.randn(V, E, dtype=torch.float64)
  x = torch.randint(0, V, (2, 4))
  ts = torch.tensor([0.5, 3.0], dtype=torch.float64)
  torch.manual_seed(123)
  rhos, us = HyperbolicHeatKernel.poincare_bridge_prod(
    ts, x, emb, output_coord=Coordinate.HYPERBOLIC_POLAR,
    prod_factor_dim=dims, prod_factor_gaussian_curvature=curvs)
  torch.manual_seed(123)
  z = HyperbolicHeatKernel.poincare_bridge_prod(
    ts, x, emb, output_coord=Coordinate.CARTESIAN,
    prod_factor_dim=dims, prod_factor_gaussian_curvature=curvs)
  assert torch.equal(z, GeoUtils.hyperbolic_polar_to_poincare_cartesian_prod(rhos, us, dims, curvs))
  rhos_back, us_back = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(z, dims, curvs)
  assert torch.allclose(rhos_back, rhos, rtol=1e-8, atol=1e-8)
  assert torch.allclose(us_back, us, atol=1e-12)


@pytest.mark.parametrize('dims,curvs,err', [
  ([2, 3], [-1.0], ValueError),          # length mismatch
  ([2, 2], [-1.0, -1.0], ValueError),    # does not sum to E = 5
  ([1, 4], [-1.0, -1.0], ValueError),    # factor dim < 2
  ([2, 3], [-1.0, 1.0], ValueError),     # K >= 0
  ([2, 3], [-1.0, float('nan')], ValueError),  # NaN curvature
  ((2, 3), (-1.0, -1.0), TypeError),     # not lists
  ([2, 3], None, TypeError),             # only one list given
])
def test_prod_validation(dims, curvs, err):
  z = torch.randn(4, 5, dtype=torch.float64) * 0.1
  with pytest.raises(err):
    GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(z, dims, curvs)
  with pytest.raises(err):
    GeoUtils.hyperbolic_polar_to_poincare_cartesian_prod(torch.rand(4, 2), _unit(4, 5), dims, curvs)


def test_forward_prod_rejects_wrong_radial_count():
  with pytest.raises(AssertionError):
    GeoUtils.hyperbolic_polar_to_poincare_cartesian_prod(
      torch.rand(4, 3), _unit(4, 6), [3, 3], [-1.0, -1.0])
