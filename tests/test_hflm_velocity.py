"""Tests for the HFLM expected-velocity sampler step (VFM marginal field).

`HFLMSampler.step` must integrate the marginal vector field
`v = sum_k p_k * log_x(e_k)` in the tangent space, mirroring
`sfm_compute_velocity` on the sphere. Unlike the sphere, hyperbolic
ambient components grow like e^{d(x,y)}, so the velocity is kept in the
factored form `v = w - s*x` (`hflm_compute_velocity`) and integrated with
the cancellation-free `hflm_exp_step`. These tests pin:

  * factored (w, s) == a manual per-token log-map sum (moderate norms,
    where the ambient reference is well-conditioned).
  * one-hot posterior + `hflm_exp_step` reproduces the constant-speed
    geodesic (`HyperbolicHeatKernel.geodesic`) at fraction `dt`, for
    K=-1.0 and K=-0.25.
  * REGRESSION (2026-07-11 sweep bug): the same one-hot equivalence when
    `E` is built exactly the way the sampler builds it — through
    `_lorentz_vocab_embeddings` from FLOAT32 weights with TRAINED-scale
    norms (rho up to ~18, clamped ~11) at K=-1.0. Lifting the table in
    float32 and casting after puts endpoints O(100) off the hyperboloid
    and silently zeroes solve rates; the lift must happen in the slerp
    dtype.
  * `top_k_velocity=1` reproduces the old argmax-endpoint step exactly.
  * full-vocab sampler steps stay in the open ball at trained-scale
    embedding norms and decode; unknown mode raises ValueError.
"""
import math

import pytest
import torch

import samplers
from geo_bridge import GeoUtils, HyperbolicHeatKernel, Coordinate, Geometry
from conftest import REPO_ROOT  # noqa: F401

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Minimal model stub (mirrors tests/test_hflm_dispatch.py)
# ---------------------------------------------------------------------------

class _StubBackbone(torch.nn.Module):
  def __init__(self, d, vocab_size, embed_scale=0.3):
    super().__init__()
    self.embed_dim = d
    self.vocab_size = vocab_size
    self.sphere_embed = torch.nn.Embedding(vocab_size, d)
    torch.nn.init.normal_(self.sphere_embed.weight, std=embed_scale)

  def get_hyperbolic_polar_embeddings(self, token_ids):
    emb = self.sphere_embed(token_ids)
    rhos = emb.norm(p=2, dim=-1, keepdim=True)
    thetas = emb / rhos.clamp_min(torch.finfo(emb.dtype).tiny)
    return rhos, thetas


class _StubModel:
  def __init__(self, d=16, vocab_size=8, length=4, embed_scale=0.3):
    self.num_tokens = length
    self.device = torch.device('cpu')
    self.backbone = _StubBackbone(d, vocab_size, embed_scale)
    self.vocab_size = vocab_size

  def noise(self, t):
    t = torch.as_tensor(t)
    alpha_t = (1.0 - t).clamp(1e-4, 1.0)
    return -torch.ones_like(alpha_t), alpha_t

  def _sigma_from_alphat(self, alpha_t):
    return -torch.log(torch.as_tensor(alpha_t).clamp_min(1e-6))

  def forward(self, *, xt=None, sigma=None, context=None):
    B, L = xt.shape[0], xt.shape[1]
    # Deterministic non-uniform logits so argmax / velocity are well-defined.
    logits = torch.zeros(B, L, self.vocab_size, dtype=torch.float64)
    logits[..., 0] = 1.0
    return logits.log_softmax(-1)


def _build_sampler(velocity='exact', top_k_velocity=-1,
                   gaussian_curvature=-1.0):
  return samplers.HFLMSampler(
    noise_removal='greedy', velocity=velocity, use_float64=True,
    slerp_float64=True, eps=1e-6, temperature=1.0, p_nucleus=1.0,
    top_k=-1, top_k_velocity=top_k_velocity, invert_time_convention=False,
    prior_cov=0.25, rho_max=12.0, gaussian_curvature=gaussian_curvature)


def _random_lorentz(B, L, d, K, rho_lo=0.3, rho_hi=3.3, seed=1):
  g = torch.Generator().manual_seed(seed)
  rhos = torch.rand(B, L, generator=g, dtype=torch.float64) \
      * (rho_hi - rho_lo) + rho_lo
  thetas = torch.randn(B, L, d, generator=g, dtype=torch.float64)
  thetas = thetas / thetas.norm(dim=-1, keepdim=True)
  return GeoUtils.hyperbolic_polar_to_lorentz_cartesian(
    rhos=rhos, thetas=thetas, gaussian_curvature=K)


def _ref_log_map(x, y, K):
  """Independent reference: log_x(y) = d(x,y) * (y - b x) / ||y - b x||_L."""
  R = 1.0 / math.sqrt(-K)
  inner = -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(-1)
  b = (-inner / R**2).clamp_min(1.0)
  d = R * torch.acosh(b)
  w = y - b.unsqueeze(-1) * x
  w_norm = (-w[..., 0]**2 + (w[..., 1:]**2).sum(-1)).clamp_min(1e-30).sqrt()
  return d.unsqueeze(-1) * w / w_norm.unsqueeze(-1)


def _onehot_logp(B, L, V, j):
  logits = torch.full((B, L, V), -1e9, dtype=torch.float64)
  logits[..., j] = 0.0
  return logits.log_softmax(-1)


# ---------------------------------------------------------------------------
# Factored (w, s): exact == manual posterior expectation of log maps
# ---------------------------------------------------------------------------

def test_exact_velocity_matches_manual_expectation():
  B, L, d, V = 2, 3, 6, 5
  K = -1.0
  x = _random_lorentz(B, L, d, K)
  E = _random_lorentz(1, V, d, K, seed=2).squeeze(0)  # [V, d+1]
  logits = torch.randn(B, L, V, dtype=torch.float64,
                       generator=torch.Generator().manual_seed(3))
  log_p = logits.log_softmax(-1)

  w, s = samplers.hflm_compute_velocity(
    x, E, log_p, mode='exact', eps=1e-6, gaussian_curvature=K)
  vel = w - s * x  # safe to materialize at these moderate norms

  ref = torch.zeros_like(x)
  p = log_p.exp()
  for k in range(V):
    y = E[k].expand(B, L, d + 1)
    ref = ref + p[..., k:k + 1] * _ref_log_map(x, y, K)
  assert torch.allclose(vel, ref, atol=1e-9)


# ---------------------------------------------------------------------------
# One-hot posterior + hflm_exp_step == constant-speed geodesic at fraction dt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('K', [-1.0, -0.25])
def test_onehot_velocity_exp_step_matches_geodesic(K):
  B, L, d, V = 2, 3, 6, 5
  x = _random_lorentz(B, L, d, K)
  E = _random_lorentz(1, V, d, K, seed=2).squeeze(0)  # [V, d+1]
  log_p = _onehot_logp(B, L, V, 1)
  dt = torch.tensor(0.3, dtype=torch.float64)

  w, s = samplers.hflm_compute_velocity(
    x, E, log_p, mode='exact', eps=1e-6, gaussian_curvature=K)
  x_new = samplers.hflm_exp_step(x, w, s, dt, gaussian_curvature=K)

  # ||v||_L = sqrt(<w,w>_L + R^2 s^2) must equal the distance d(x, e_1).
  R = 1.0 / math.sqrt(-K)
  ww = -w[..., 0]**2 + (w[..., 1:]**2).sum(-1)
  n = (ww + (R * s.squeeze(-1))**2).clamp_min(0).sqrt()
  inner = -x[..., 0] * E[1][0] + (x[..., 1:] * E[1][1:]).sum(-1)
  dist = R * torch.acosh((-inner / R**2).clamp_min(1.0))
  assert torch.allclose(n, dist, atol=1e-9)

  rho1, theta1 = GeoUtils.lorentz_cartesian_to_hyperbolic_polar(
    E[1], gaussian_curvature=K)
  geo = HyperbolicHeatKernel.geodesic(
    t=float(dt), gaussian_curvature=K,
    src_cartesian=x, cartesian_model=Geometry.LORENTZ,
    dest_radial=rho1.expand(B, L), dest_angular=theta1.expand(B, L, d),
    output_coord=Coordinate.CARTESIAN)
  assert torch.allclose(x_new, geo, atol=1e-8)


# ---------------------------------------------------------------------------
# REGRESSION: trained-scale norms + float32 weight table (2026-07-11 bug)
# ---------------------------------------------------------------------------

def test_trained_norm_f32_table_step_matches_geodesic():
  """E built exactly like the sampler builds it (`_lorentz_vocab_embeddings`
  from float32 weights, lift in the slerp dtype) at trained-scale norms
  (rho up to ~18 raw -> ~11 clamped) and a far x (rho ~9.9), K=-1. The
  buggy float32 lift put E off the hyperboloid and drove solve rates to 0.
  """
  torch.manual_seed(0)
  K = -1.0
  B, L, d, V = 2, 3, 512, 12
  sampler = _build_sampler(velocity='exact', top_k_velocity=-1,
                           gaussian_curvature=K)
  model = _StubModel(d=d, vocab_size=V)
  with torch.no_grad():  # trained-like spread of embedding norms, f32
    W = model.backbone.sphere_embed.weight
    W.copy_(torch.randn(V, d) / math.sqrt(d)
            * torch.linspace(0.4, 18.0, V).unsqueeze(-1))

  E = sampler._lorentz_vocab_embeddings(model, torch.float64)  # [V, d+1]
  # Every endpoint must be on the hyperboloid: <e,e>_L = -R^2.
  inv = -E[:, 0]**2 + (E[:, 1:]**2).sum(-1)
  assert float((inv + 1.0).abs().max()) < 1e-6

  x = _random_lorentz(B, L, d, K, rho_lo=9.8, rho_hi=9.9, seed=4)
  j = V - 1  # the largest-norm (clamped) endpoint = worst case
  log_p = _onehot_logp(B, L, V, j)
  dt = torch.tensor(0.0056, dtype=torch.float64)

  w, s = samplers.hflm_compute_velocity(
    x, E, log_p, mode='exact', eps=1e-6, gaussian_curvature=K)
  x_new = samplers.hflm_exp_step(x, w, s, dt, gaussian_curvature=K)

  rho_j, theta_j = GeoUtils.lorentz_cartesian_to_hyperbolic_polar(
    E[j], gaussian_curvature=K)
  geo = HyperbolicHeatKernel.geodesic(
    t=float(dt), gaussian_curvature=K,
    src_cartesian=x, cartesian_model=Geometry.LORENTZ,
    dest_radial=rho_j.expand(B, L), dest_angular=theta_j.expand(B, L, d),
    output_coord=Coordinate.CARTESIAN)
  rel = ((x_new - geo).abs() / geo.abs().clamp_min(1.0)).max()
  assert float(rel) < 1e-8
  # Result stays exactly on the hyperboloid.
  inv_new = -x_new[..., 0]**2 + (x_new[..., 1:]**2).sum(-1)
  assert float((inv_new + 1.0).abs().max()) < 1e-6


# ---------------------------------------------------------------------------
# top_k_velocity=1 reproduces the old argmax-endpoint geodesic step
# ---------------------------------------------------------------------------

def test_topk1_step_equals_argmax_endpoint_step():
  torch.manual_seed(0)
  sampler = _build_sampler(velocity='exact', top_k_velocity=1)
  model = _StubModel(d=16, vocab_size=8)
  state = sampler.init_state(model, 2, num_steps=4)
  x0 = state.xt.clone()

  # Manual argmax-endpoint step: the stub's argmax token is 0 everywhere.
  rhos, thetas = model.backbone.get_hyperbolic_polar_embeddings(
    torch.zeros(2, model.num_tokens, dtype=torch.long))
  dest_rho = sampler._rho_clamp(rhos.squeeze(-1)).to(torch.float64)
  dest_theta = thetas.to(torch.float64)
  _, a_t = model.noise(state.t_schedule[0])
  _, a_s = model.noise(state.t_schedule[1])
  dt = ((a_s - a_t) / (1 - a_t).clamp(min=1e-6)).clamp(0.0, 1.0)
  expected = HyperbolicHeatKernel.geodesic(
    t=dt.reshape(-1, 1, 1).to(torch.float64),
    gaussian_curvature=-1.0,
    src_cartesian=x0.to(torch.float64), cartesian_model=Geometry.POINCARE,
    dest_radial=dest_rho, dest_angular=dest_theta,
    output_coord=Coordinate.CARTESIAN)

  state = sampler.step(model, state)
  assert torch.allclose(state.xt.double(), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Sampler integration at trained-scale norms: in-ball, moves, decodes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('velocity', ['exact', 'sample'])
def test_sampler_steps_stay_in_ball_and_decode(velocity):
  torch.manual_seed(0)
  sampler = _build_sampler(velocity=velocity, top_k_velocity=-1)
  # d=512 + std 0.35 -> embedding norms ~8, the regime real models train to.
  model = _StubModel(d=512, vocab_size=8, embed_scale=0.35)
  state = sampler.init_state(model, 2, num_steps=4)
  x_prev = state.xt.clone()

  for _ in range(3):  # non-last steps
    state = sampler.step(model, state)
    assert torch.isfinite(state.xt).all()
    assert float(state.xt.norm(p=2, dim=-1).max()) < 1.0
  assert not torch.allclose(state.xt, x_prev)

  state = sampler.step(model, state)  # last step decodes
  assert state.done
  assert state.xt.dtype == torch.int64
  assert tuple(state.xt.shape) == (2, model.num_tokens)
  assert int(state.xt.min()) >= 0 and int(state.xt.max()) < 8


def test_unknown_velocity_mode_raises():
  sampler = _build_sampler(velocity='argmax')
  model = _StubModel(d=16, vocab_size=8)
  state = sampler.init_state(model, 2, num_steps=4)
  with pytest.raises(ValueError):
    sampler.step(model, state)
