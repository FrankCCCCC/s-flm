"""Regression test for the d>=343 gamma-ratio overflow in `_euclid_mean`.

`_euclid_mean(d)` originally used the direct Gamma ratio
`sqrt(2) Gamma((d+1)/2)/Gamma(d/2)`, which raises `OverflowError` for d>=343
(Gamma(256.5) is out of float range), crashing `sample_radial` at the primary
d=512 before any grid is built. The log-space lgamma form is finite and
mathematically identical for valid d, removing that crash.

NOTE: `sample_radial` itself still NaNs at large d (>=~128) for an independent
reason documented in its own docstring (the `sinh^{d-1}(rho)` marginal is formed
in linear, not log, space and overflows once `rho ~ 709/(d-1)`). The B1 fix
targets only the `_euclid_mean` Gamma overflow; the marginal overflow is a
separate `sample_radial` issue (out of scope for the B1 edit, which is confined
to `_euclid_mean`). See the xfail below.
"""
import math

import pytest
import torch

import geo_bridge
from conftest import REPO_ROOT  # noqa: F401  (ensures repo root on sys.path)

PRIMARY_D = 512


def test_euclid_mean_finite_at_primary_d():
  """B1: the short-time radial scale no longer overflows at d=512."""
  val = geo_bridge.HyperbolicHeatKernel._euclid_mean(PRIMARY_D)
  assert math.isfinite(val)
  assert val > 0.0


def test_euclid_mean_matches_direct_gamma_for_valid_d():
  """The lgamma form equals the direct Gamma ratio wherever the latter is finite."""
  for d in (2, 8, 342):
    direct = math.sqrt(2.0) * math.gamma((d + 1) / 2.0) / math.gamma(d / 2.0)
    log_space = geo_bridge.HyperbolicHeatKernel._euclid_mean(d)
    assert math.isclose(direct, log_space, rel_tol=1e-12)


@pytest.mark.xfail(
  reason='sample_radial NaNs at d=512: the sinh^(d-1) marginal overflows in '
         'linear space (separate from the _euclid_mean fix; see docstring).',
  strict=True)
def test_sample_radial_finite_at_primary_d():
  """`sample_radial` at the primary d=512 yields finite, >=0 radii."""
  ts = torch.tensor([0.01, 0.05], dtype=torch.float64)
  rhos = geo_bridge.HyperbolicHeatKernel.sample_radial(ts, d=PRIMARY_D, seq_len=4)
  assert tuple(rhos.shape) == (2, 4)
  assert torch.isfinite(rhos).all().item()
  assert (rhos >= 0.0).all().item()
