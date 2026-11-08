"""Contract tests for HyperbolicBoundaryFM (HBFM, `algo.py`).

Independent checks of the hyperbolic bridge DLM that the training scripts
(`scripts/train/*/hbfm.sh`) run:

  * the repo imports again (`loss.py` no longer imports unigram2's `model`);
  * `time_conversion` is the -log map from the noise schedule's alpha_t to the
    bridge heat time with the exp proposal's 1 / q(t) weight, unbiased for
    int CE(t) dt under the log-linear schedule, truncated to the alpha range
    by a `TruncatedScheduleWrapper`, and well-defined on a refit
    `AdaptiveSchedule`;
  * the Cartesian bridge state `q_xt` (product manifold, one heat time per
    sequence, sudoku-prompt positions pinned at the clean end) and the
    polar coordinates `_process_model_output` recovers from it;
  * the horosphere readout equals the product Poisson-kernel Bayes posterior,
    per-factor curvature included;
  * `nll` / `_loss` on the REAL model (hydra-composed configs, real
    HyperbolicDiT): finite, differentiable, log V at the origin and 0 at the
    boundary at init (the DiT's zero-init readout makes the residual 0);
  * `HyperbolicModelBase.forward_horosphere` accepts a Cartesian trunk state
    and falls back to the model's own factor spec (the hyper_model.py fixes);
  * sampler registration and config validation.

Built with the real configs (`hydra.compose`) and the sudoku tokenizer, all
offline: only `metrics.Metrics` (which downloads a GPT-2 tokenizer) is stubbed.
The tests that run the DiT need a GPU (its rotary / attention kernels are
flash_attn, CUDA only); the bridge, readout and schedule tests run anywhere.
"""
import math

import pytest
import torch

from conftest import REPO_ROOT  # noqa: F401  (repo root on sys.path)
from numeric.geo_bridge import Coordinate

torch.manual_seed(0)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
needs_gpu = pytest.mark.skipif(
  DEVICE != 'cuda', reason='HyperbolicDiT runs flash_attn kernels (CUDA only)')

BASE_OVERRIDES = [
  'algo=hbfm', 'model=tiny-hyperbolic-dit', 'sampler=hbfm', 'data=sudoku',
  'noise=log-linear',
  'model.embed_dim=12', 'model.hidden_size=64', 'model.n_heads=4',
  'model.cond_dim=32', 'model.n_blocks=2', 'model.length=8',
  'model.dropout=0.0',
  'algo.prod_factor_dim=3', 'algo.prod_factor_gaussian_curvature=-1.0',
  'algo.time_exp_rate=1.0',   # heat times in [1e-3, 6.9]: informative states, non-zero CE
  'loader.global_batch_size=4', 'loader.batch_size=4',
  'trainer.devices=1', 'trainer.num_nodes=1',
]


def _compose(overrides=()):
  import hydra
  import main  # noqa: F401  registers the cwd / device_count / eval / div_up resolvers
  with hydra.initialize(config_path='../configs', version_base=None):
    return hydra.compose(config_name='config',
                         overrides=BASE_OVERRIDES + list(overrides))


class _NoMetrics:
  def __init__(self, **kwargs):
    pass

  def to(self, *args, **kwargs):
    return self


def _build_model(monkeypatch, overrides=()):
  import algo
  import dataloader
  import trainer_base
  monkeypatch.setattr(trainer_base.metrics, 'Metrics', _NoMetrics)
  cfg = _compose(overrides)
  torch.manual_seed(0)
  model = algo.HyperbolicBoundaryFM(cfg, tokenizer=dataloader.get_tokenizer(cfg))
  return model.to(DEVICE), cfg


def _tokens(model, B=2, L=8):
  return torch.randint(0, model.vocab_size, (B, L), device=model.device)


def _alpha_at_heat_time(model, B, heat_time):
  """alpha_t (MDLM convention) whose time_conversion is `heat_time`."""
  u = math.exp(-model.config.algo.time_exp_rate * heat_time)
  return torch.full((B, 1), 1.0 - u, dtype=torch.float64, device=model.device)


# ---------------------------------------------------------------------------
# Imports / config resolution
# ---------------------------------------------------------------------------

def test_repo_imports():
  import numeric.loss  # noqa: F401  used to fail: `from model import ...`
  import numeric.horosphere  # noqa: F401
  import algo  # noqa: F401
  import main  # noqa: F401


def test_resolve_prod_factors():
  import omegaconf
  from algo import HyperbolicBoundaryFM as H
  cfg = lambda d, k: omegaconf.OmegaConf.create(  # noqa: E731
    {'prod_factor_dim': d, 'prod_factor_gaussian_curvature': k})
  assert H._resolve_prod_factors(cfg(3, -2.0), 12) == ([3] * 4, [-2.0] * 4)
  assert H._resolve_prod_factors(cfg(None, None), 12) == ([12], [-1.0])
  assert H._resolve_prod_factors(cfg([4, 8], [-1.0, -0.5]), 12) == (
    [4, 8], [-1.0, -0.5])
  with pytest.raises(ValueError):
    H._resolve_prod_factors(cfg(5, -1.0), 12)      # 12 % 5 != 0
  with pytest.raises(ValueError):
    H._resolve_prod_factors(cfg([4, 4], -1.0), 12)  # does not sum to 12


# ---------------------------------------------------------------------------
# time_conversion: the -log map and the exp proposal's weight
# ---------------------------------------------------------------------------

def test_time_conversion_is_exp_proposal():
  from algo import HyperbolicBoundaryFM as H
  from noise_schedules import LogLinear
  eps, rate = 1e-3, 5.0
  t = torch.linspace(1e-3, 1.0, 1001)
  _, alpha = LogLinear(eps)(t)
  s, w = H.time_conversion(alpha, False, rate)
  u = (1.0 - alpha).double()
  assert s.dtype == torch.float64 and w.dtype == torch.float64
  assert torch.allclose(s, -u.log() / rate)
  # 1 / q(t) of the exp proposal q(t) = rate e^{-rate t} (loss.Proposal).
  assert torch.allclose(w, torch.exp(rate * s) / rate)
  # Noisier t -> smaller heat time (u = 1 is the origin).
  assert (s[1:] < s[:-1]).all()
  # SFM convention: u = alpha_t.
  s2, _ = H.time_conversion(alpha, True, rate)
  assert torch.allclose(s2, -alpha.double().log() / rate)


def test_time_conversion_weight_is_unbiased_under_log_linear():
  """E_t[w(t) f(s(t))] == int f(s) ds / (1 - eps) over the covered heat times
  (u = (1 - eps) t is uniform): a quadrature identity on a fine t grid."""
  from algo import HyperbolicBoundaryFM as H
  from noise_schedules import LogLinear
  eps, rate = 1e-3, 3.0
  t = torch.linspace(1e-3, 1.0, 200001, dtype=torch.float64)
  _, alpha = LogLinear(eps)(t)
  s, w = H.time_conversion(alpha, False, rate)
  lhs = torch.trapezoid(w * torch.exp(-s), t)
  s_lo, s_hi = s[-1], s[0]           # s decreases in t
  rhs = (torch.exp(-s_lo) - torch.exp(-s_hi)) / (1 - eps)
  assert abs(lhs - rhs) < 1e-4 * rhs, (lhs, rhs)


def test_time_conversion_truncated_schedule_covers_the_alpha_range():
  from algo import HyperbolicBoundaryFM as H
  from noise_schedules import LogLinear, TruncatedScheduleWrapper
  rate, alpha_min, alpha_max = 2.0, 0.2, 0.9
  sched = TruncatedScheduleWrapper(LogLinear(1e-3), alpha_min, alpha_max, 1e-3)
  _, alpha = sched(torch.tensor([1e-3, 1.0]))
  s, w = H.time_conversion(alpha, False, rate)
  assert s[1] == pytest.approx(-math.log(1 - alpha_min) / rate, rel=1e-5)
  assert s[0] == pytest.approx(-math.log(1 - alpha_max) / rate, rel=2e-2)
  assert torch.isfinite(w).all()


def test_time_conversion_is_finite_at_the_clean_endpoint():
  """A refit AdaptiveSchedule can return alpha_t == 1 exactly (u = 0) for
  sampled t; the weight must stay finite (0 * inf would poison training)."""
  from algo import HyperbolicBoundaryFM as H
  alpha = torch.tensor([1.0, 1.0 - 1e-3], dtype=torch.float32)
  s, w = H.time_conversion(alpha, False, 20.0)
  assert torch.isfinite(s).all() and torch.isfinite(w).all()
  assert w[0] == pytest.approx(1e12 / 20.0, rel=1e-6)
  assert s[1] == pytest.approx(-math.log(1e-3) / 20.0, rel=1e-4)


def test_time_conversion_on_refit_adaptive_schedule():
  from algo import HyperbolicBoundaryFM as H
  from noise_schedules import AdaptiveSchedule, LogLinear
  torch.manual_seed(1)
  sched = AdaptiveSchedule(
    LogLinear(eps=1e-3), buffer_size=2048, refit_every=1, n_grid=200,
    n_knots=10, spline_degree=3, ridge_alpha=1e-3, uniform_mix=1e-3,
    max_steps=20, warmup_steps=0, ema=0.0)
  for step in range(1, 12):
    t = torch.rand(256)
    sched.record_time_loss_pair(t, t ** 3 + 0.05 * torch.rand(256), step)
  assert bool(sched.has_schedule)
  t = torch.linspace(1e-3, 1.0, 1001)
  _, alpha = sched(t)
  s, w = H.time_conversion(alpha, False, 2.0)
  assert torch.isfinite(s).all() and torch.isfinite(w).all()
  assert (s[1:] <= s[:-1]).all()     # the remapped alpha stays monotone
  _, alpha_base = LogLinear(1e-3)(t)
  s_base, _ = H.time_conversion(alpha_base, False, 2.0)
  assert not torch.allclose(s, s_base)  # the refit actually moved the proposal


def test_time_conversion_unif_is_the_unif_proposal():
  """mode='unif': heat time uniform on [0, range_upper_bound] under the
  log-linear schedule, decreasing in the noise fraction like the exp map
  (u = 1 is the origin), weighted by the interval (loss.Proposal's unif
  proposal's 1 / q(t)); float64 tensors of alpha_t's shape."""
  from algo import HyperbolicBoundaryFM as H
  from noise_schedules import LogLinear
  eps, ub = 1e-3, 2.5
  t = torch.linspace(1e-3, 1.0, 200001, dtype=torch.float64)
  _, alpha = LogLinear(eps)(t)
  s, w = H.time_conversion(alpha.float(), False, mode='unif', range_upper_bound=ub)
  assert s.dtype == torch.float64 and w.dtype == torch.float64
  assert s.shape == alpha.shape and w.shape == alpha.shape
  assert torch.allclose(s, alpha * ub, atol=1e-6)
  assert (s >= 0).all() and (s <= ub).all()
  assert (s[1:] < s[:-1]).all()          # noisier t -> smaller heat time
  assert torch.equal(w, torch.full_like(s, ub))
  # E_t[w f(s(t))] == int f(s) ds / (1 - eps): the exp map's quadrature identity.
  s, w = H.time_conversion(alpha, False, mode='unif', range_upper_bound=ub)
  lhs = torch.trapezoid(w * torch.exp(-s), t)
  rhs = (torch.exp(-s[-1]) - torch.exp(-s[0])) / (1 - eps)
  assert abs(lhs - rhs) < 1e-4 * rhs, (lhs, rhs)
  # SFM convention: u = alpha_t.
  s2, _ = H.time_conversion(alpha, True, mode='unif', range_upper_bound=ub)
  assert torch.allclose(s2, (1.0 - alpha) * ub)
  with pytest.raises(ValueError):
    H.time_conversion(alpha, False, mode='bogus')


def test_time_conversion_trunc_exp_is_the_truncated_exp_proposal():
  """mode='trunc_exp': loss.Proposal's truncated_exp proposal on
  [0, range_upper_bound] with uniform draw 1 - u and its 1 / q(t) weight,
  the exp map's quadrature identity, and the exp map itself as
  range_upper_bound -> inf."""
  from algo import HyperbolicBoundaryFM as H
  from noise_schedules import LogLinear
  eps, rate, ub = 1e-3, 3.0, 1.5
  t = torch.linspace(1e-3, 1.0, 200001, dtype=torch.float64)
  _, alpha = LogLinear(eps)(t)
  s, w = H.time_conversion(alpha, False, rate, mode='trunc_exp', range_upper_bound=ub)
  normalizer = 1.0 - math.exp(-rate * ub)
  s_ref = -torch.log1p(-alpha * normalizer) / rate      # the uniform draw is alpha_t
  w_ref = normalizer / (rate * torch.exp(-rate * s_ref))
  assert s.dtype == torch.float64 and w.dtype == torch.float64
  assert torch.allclose(s, s_ref) and torch.allclose(w, w_ref)
  assert (s >= 0).all() and (s <= ub).all()
  assert (s[1:] < s[:-1]).all()          # noisier t -> smaller heat time
  lhs = torch.trapezoid(w * torch.exp(-s), t)
  rhs = (torch.exp(-s[-1]) - torch.exp(-s[0])) / (1 - eps)
  assert abs(lhs - rhs) < 1e-4 * rhs, (lhs, rhs)
  # SFM convention: u = alpha_t.
  s2, _ = H.time_conversion(alpha, True, rate, mode='trunc_exp', range_upper_bound=ub)
  assert torch.allclose(s2, -torch.log1p(-(1.0 - alpha) * normalizer) / rate)
  # alpha_t == 1 exactly (u = 0) lands at the truncation point, finite weight.
  s1, w1 = H.time_conversion(torch.tensor([1.0]), False, rate, mode='trunc_exp',
                             range_upper_bound=ub)
  assert s1.item() == pytest.approx(ub, rel=1e-6) and torch.isfinite(w1).all()
  # range_upper_bound -> inf is the exp map.
  s_exp, w_exp = H.time_conversion(alpha, False, rate)
  s_inf, w_inf = H.time_conversion(alpha, False, rate, mode='trunc_exp', range_upper_bound=1e3)
  assert torch.allclose(s_inf, s_exp) and torch.allclose(w_inf, w_exp)


# ---------------------------------------------------------------------------
# q_xt: polar bridge state on the product manifold
# ---------------------------------------------------------------------------

def test_q_xt_cartesian_state_contract(monkeypatch):
  from numeric.geo_bridge import GeoUtils
  model, cfg = _build_model(monkeypatch)
  assert model.prod_factor_dim == [3, 3, 3, 3]
  B, L = 2, 8
  x = _tokens(model, B, L)
  alpha = torch.cat([_alpha_at_heat_time(model, 1, 0.3),
                     _alpha_at_heat_time(model, 1, 2.0)])
  xt, weight = model.q_xt(x, alpha, use_pure_noise=False)
  assert xt.shape == (B, L, 12) and xt.dtype == torch.float64
  ball = xt.unflatten(-1, (4, 3)).norm(dim=-1)
  assert (ball < 1.0).all()                       # inside each unit ball
  assert ball[1].mean() > ball[0].mean()          # larger heat time -> further out
  rate = cfg.algo.time_exp_rate
  assert weight.shape == (B, 1)
  assert torch.allclose(weight, 1.0 / (rate * (1.0 - alpha)))

  # Sudoku-style mask: prompt positions pinned at the clean end (boundary).
  valid = torch.ones(B, L, dtype=torch.long, device=model.device)
  valid[:, :3] = 0
  xt_m, _ = model.q_xt(x, alpha, use_pure_noise=False, valid_tokens=valid)
  clean_theta = model.word_embedding[x].double().unflatten(-1, (4, 3))
  clean_theta = (clean_theta / clean_theta.norm(dim=-1, keepdim=True)).flatten(-2)
  assert torch.allclose(xt_m[:, :3], clean_theta[:, :3], atol=1e-12)
  assert not torch.allclose(xt_m[:, 3:], clean_theta[:, 3:], atol=1e-3)
  rhos, thetas = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(
    xt_m, model.prod_factor_dim, model.prod_factor_gaussian_curvature)
  assert torch.isfinite(rhos).all()
  assert (rhos[:, :3] > 30).all()                  # the boundary reads as a large, finite rho
  assert torch.allclose(thetas[:, :3], clean_theta[:, :3], atol=1e-12)

  with pytest.raises(NotImplementedError):
    model.q_xt(x, alpha, use_pure_noise=True)


def test_poincare_cartesian_to_hyperbolic_polar_prod_inverts_the_bridge_coordinates(monkeypatch):
  from numeric.geo_bridge import Coordinate, GeoUtils, HyperbolicHeatKernel
  model, _ = _build_model(monkeypatch, [
    'algo.prod_factor_dim=[3,3,3,3]',
    'algo.prod_factor_gaussian_curvature=[-1.0,-4.0,-0.25,-2.0]'])
  B, L = 2, 8
  x = _tokens(model, B, L)
  ts = torch.full((B,), 2.0, dtype=torch.float64, device=model.device)
  rhos, thetas = HyperbolicHeatKernel.poincare_bridge_prod(
    ts=ts, targets=x, word_embedding=model.word_embedding,
    output_coord=Coordinate.HYPERBOLIC_POLAR,
    prod_factor_dim=model.prod_factor_dim,
    prod_factor_gaussian_curvature=model.prod_factor_gaussian_curvature)
  xt = GeoUtils.hyperbolic_polar_to_poincare_cartesian_prod(
    rhos, thetas, model.prod_factor_dim, model.prod_factor_gaussian_curvature)
  # The product helper is the per-factor single-ball conversion, concatenated.
  assert torch.equal(xt, torch.cat([
    GeoUtils.hyperbolic_polar_to_poincare_cartesian(rhos[..., m], th, gaussian_curvature=K)
    for m, (th, K) in enumerate(zip(thetas.split(3, dim=-1),
                                    model.prod_factor_gaussian_curvature))], dim=-1))
  rhos_back, thetas_back = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(
    xt, model.prod_factor_dim, model.prod_factor_gaussian_curvature)
  assert torch.allclose(rhos_back, rhos, rtol=1e-8, atol=1e-8)
  assert torch.allclose(thetas_back, thetas, atol=1e-12)


def test_q_xt_clamps_the_heat_time_at_the_radial_ceiling(monkeypatch):
  """alpha_t = 1 (u floored at 1e-12) is heat time -log(1e-12) / rate = 276
  at rate 0.1, past max_heat_time (~97 for H^3): clamped there, so the radial
  sampler never sees the overflow regime and the state stays finite."""
  from numeric.geo_bridge import GeoUtils
  model, _ = _build_model(monkeypatch, ['algo.time_exp_rate=0.1'])
  assert model.max_heat_time < 276
  x = _tokens(model)
  alpha = torch.ones(2, 1, dtype=torch.float64, device=model.device)
  xt, _ = model.q_xt(x, alpha, use_pure_noise=False)
  assert torch.isfinite(xt).all()
  rhos, _ = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(
    xt, model.prod_factor_dim, model.prod_factor_gaussian_curvature)
  assert rhos.mean() > 30  # E[rho] ~ (d-1) t / 2 ~ 97 at d = 3, capped at ~34.7 by the ball read-back


# ---------------------------------------------------------------------------
# Horosphere readout == product Poisson-kernel Bayes posterior
# ---------------------------------------------------------------------------

def _reference_log_posterior(model, rhos, thetas):
  """-sum_m (d_m - 1) log(cosh u_m - sinh u_m <phi_v,m, theta_m>), u = kappa rho,
  computed the textbook way (fine for the moderate u used here)."""
  dims, curvs = model.prod_factor_dim, model.prod_factor_gaussian_curvature
  phi = model.word_embedding.detach().double()
  log_q = 0.0
  off = 0
  for m, (d, K) in enumerate(zip(dims, curvs)):
    phi_m = phi[:, off:off + d]
    phi_m = phi_m / phi_m.norm(dim=-1, keepdim=True)
    theta_m = thetas[..., off:off + d]
    off += d
    u = (math.sqrt(-K) * rhos[..., m]).unsqueeze(-1)              # (B, L, 1)
    inner = theta_m @ phi_m.T                                     # (B, L, V)
    log_q = log_q - (d - 1) * torch.log(torch.cosh(u) - torch.sinh(u) * inner)
  return log_q.log_softmax(-1)


@pytest.mark.parametrize('precision,atol', [('float64', 1e-9), ('float32', 1e-4)])
def test_horosphere_readout_is_bayes_posterior(monkeypatch, precision, atol):
  model, _ = _build_model(monkeypatch, [
    'algo.forward_type=horosphere',
    'algo.prod_factor_dim=[3,3,3,3]',
    'algo.prod_factor_gaussian_curvature=[-1.0,-4.0,-0.25,-2.0]',
    f'algo.readout_precision={precision}'])
  assert model.prod_factor_gaussian_curvature == [-1.0, -4.0, -0.25, -2.0]
  from numeric.geo_bridge import Coordinate, GeoUtils, HyperbolicHeatKernel
  B, L, V = 2, 8, model.vocab_size
  x = _tokens(model, B, L)
  ts = torch.full((B,), 0.5, dtype=torch.float64, device=model.device)
  rhos, thetas = HyperbolicHeatKernel.poincare_bridge_prod(
    ts=ts, targets=x, word_embedding=model.word_embedding,
    output_coord=Coordinate.HYPERBOLIC_POLAR,
    prod_factor_dim=model.prod_factor_dim,
    prod_factor_gaussian_curvature=model.prod_factor_gaussian_curvature)
  xt = GeoUtils.hyperbolic_polar_to_poincare_cartesian_prod(
    rhos, thetas, model.prod_factor_dim, model.prod_factor_gaussian_curvature)
  with torch.no_grad():
    log_p = model._process_model_output(
      torch.zeros(B, L, V, device=model.device), xt, None, None)
  assert log_p.dtype == getattr(torch, precision)
  ref = _reference_log_posterior(model, rhos, thetas)
  assert torch.allclose(log_p.double(), ref, atol=atol)
  # The posterior is peaked on the target word already at t = 0.5.
  assert (log_p.argmax(-1) == x).float().mean() > 0.9
  # A sampler context's temperature tempers the geometry too.
  import samplers
  with torch.no_grad():
    log_p_T = model._process_model_output(
      torch.zeros(B, L, V, device=model.device), xt, None,
      samplers.SFMContext(temperature=2.0, z_sc=None))
  from numeric.horosphere import HorosphereGeometry
  horo = HorosphereGeometry.compute_horosphere(
    theta=thetas, radius=rhos, word_embedding=model.word_embedding,
    prod_factor_dim=model.prod_factor_dim,
    prod_factor_gaussian_curvature=model.prod_factor_gaussian_curvature,
    readout_dtype=model.readout_dtype)
  assert horo.dtype == getattr(torch, precision)
  assert torch.allclose(log_p_T.double(), (horo.double() / 2).log_softmax(-1), atol=atol)


@pytest.mark.parametrize('precision,atol', [('float64', 1e-10), ('float32', 1e-4)])
def test_chunked_horosphere_readout_matches_tensor_form(monkeypatch, precision, atol):
  """HorosphereGeometry.horosphere_geometry_chunk (vocabulary in blocks through
  _HorosphereChunk: exact forward, inner-product-form backward, in readout_dtype)
  equals the (B, L, V, m, d) broadcast horosphere_geometry_tensor in value and
  in the boundary-table gradient."""
  from numeric.horosphere import HorosphereGeometry
  model, _ = _build_model(monkeypatch, [
    'algo.prod_factor_dim=[3,3,3,3]',
    'algo.prod_factor_gaussian_curvature=[-1.0,-4.0,-0.25,-2.0]',
    f'algo.readout_precision={precision}'])
  monkeypatch.setattr(HorosphereGeometry, 'READOUT_CHUNK', 5)   # V = 12 -> chunks of 5, 5, 2
  B, L = 2, 8
  theta = torch.randn(B, L, 4, 3, dtype=torch.float64, device=model.device)
  theta = (theta / theta.norm(dim=-1, keepdim=True)).flatten(-2)
  radius = torch.rand(B, L, 4, dtype=torch.float64, device=model.device) * 3
  kw = dict(theta=theta, radius=radius, word_embedding=model.word_embedding,
            prod_factor_dim=model.prod_factor_dim,
            prod_factor_gaussian_curvature=model.prod_factor_gaussian_curvature)
  out = HorosphereGeometry.horosphere_geometry_chunk(**kw, readout_dtype=model.readout_dtype)
  ref = HorosphereGeometry.horosphere_geometry_tensor(**kw, readout_dtype=torch.float64)
  assert out.dtype == model.readout_dtype and ref.dtype == torch.float64
  assert out.shape == (B, L, model.vocab_size)
  assert torch.allclose(out.double(), ref, atol=atol)
  g_out = torch.autograd.grad(out.sum(), model.word_embedding)[0]
  g_ref = torch.autograd.grad(ref.sum(), model.word_embedding)[0]
  assert torch.allclose(g_out.double(), g_ref.double(), atol=atol, rtol=1e-4)
  # no_grad path (sampling)
  with torch.no_grad():
    assert torch.allclose(
      HorosphereGeometry.horosphere_geometry_chunk(**kw, readout_dtype=model.readout_dtype).double(), ref, atol=atol)
  # the model's forward_type=horosphere path goes through compute_horosphere with the config's chunk flag
  assert torch.allclose(HorosphereGeometry.compute_horosphere(
    **kw, readout_dtype=model.readout_dtype,
    forward_chunked=model.config.algo.horosphere_forward_chunked).double(), ref, atol=atol)


def test_naive_readout_is_plain_log_softmax(monkeypatch):
  model, _ = _build_model(monkeypatch, ['algo.forward_type=naive'])
  logits = torch.randn(2, 8, model.vocab_size)
  with torch.no_grad():
    log_p = model._process_model_output(logits, None, None, None)
  assert log_p.dtype == model.readout_dtype
  assert torch.allclose(log_p.double(), logits.double().log_softmax(-1), atol=1e-5)


def test_unknown_forward_type_raises_at_readout(monkeypatch):
  model, _ = _build_model(monkeypatch, ['algo.forward_type=bogus'])
  with pytest.raises(ValueError):
    model._process_model_output(torch.zeros(2, 8, model.vocab_size), None, None, None)


# ---------------------------------------------------------------------------
# End to end on the real model
# ---------------------------------------------------------------------------

@needs_gpu
def test_nll_end_to_end_is_finite_and_differentiable(monkeypatch):
  # Horosphere: under naive the zero-init readout gives the embedding no gradient at step 0.
  model, _ = _build_model(monkeypatch, ['algo.forward_type=horosphere'])
  B, L = 2, 8
  x = _tokens(model, B, L)
  loss, t = model.nll(x, None, None)
  assert loss.shape == (B, L) and t.shape == (B,)
  assert torch.isfinite(loss).all() and (loss >= 0).all()
  loss.sum().backward()
  emb_grad = model.backbone.sphere_embed.weight.grad
  assert emb_grad is not None and torch.isfinite(emb_grad).all()
  assert emb_grad.abs().sum() > 0
  blk = next(model.backbone.blocks[0].parameters())
  assert blk.grad is not None and torch.isfinite(blk.grad).all()


@needs_gpu
@pytest.mark.parametrize('mode', ['unif', 'trunc_exp'])
def test_nll_end_to_end_bounded_time_conversion(monkeypatch, mode):
  model, _ = _build_model(monkeypatch, [
    f'algo.time_conversion_mode={mode}', 'algo.time_range_upper_bound=3.0'])
  B, L = 2, 8
  x = _tokens(model, B, L)
  alpha = torch.tensor([[0.2], [0.9]], dtype=torch.float64, device=model.device)
  xt, weight = model.q_xt(x, alpha, use_pure_noise=False)
  assert xt.dtype == torch.float64 and weight.shape == (B, 1)
  assert (weight > 0).all()
  if mode == 'unif':
    assert torch.allclose(weight, torch.full_like(weight, 3.0))
  # alpha_t = 1 is clean: the higher alpha_t lands further out on the bridge.
  ball = xt.unflatten(-1, (4, 3)).norm(dim=-1)
  assert ball[1].mean() > ball[0].mean()
  loss, t = model.nll(x, None, None)
  assert loss.shape == (B, L) and torch.isfinite(loss).all()
  loss.sum().backward()
  assert torch.isfinite(model.backbone.sphere_embed.weight.grad).all()


@needs_gpu
def test_trainer_base_loss_with_sudoku_mask(monkeypatch):
  model, _ = _build_model(monkeypatch)
  B, L = 2, 8
  x = _tokens(model, B, L)
  mask = torch.ones(B, L, dtype=torch.long, device=model.device)
  mask[:, :3] = 0
  out = model._loss(x, mask)
  assert torch.isfinite(out.loss) and int(out.num_tokens) == B * (L - 3)


def _ce_at_heat_time(model, x, heat_time):
  xt, _ = model.q_xt(
    x, _alpha_at_heat_time(model, x.shape[0], heat_time), use_pure_noise=False)
  with torch.no_grad():
    log_p = model.forward(
      x0=x, xt=xt, sigma=torch.zeros(x.shape[0], 1, device=x.device),
      context=None)
  return -log_p.gather(-1, x.unsqueeze(-1)).squeeze(-1)


@needs_gpu
def test_bayes_limits_at_init(monkeypatch):
  """At init the DiT readout is zero (DDiTFinalLayer zero-init), so the
  log-posterior is the horosphere term alone: uniform (CE = log V) at the
  origin, one-hot (CE = 0) near the boundary."""
  model, _ = _build_model(monkeypatch, ['algo.forward_type=horosphere'])
  B, L = 2, 8
  x = _tokens(model, B, L)
  ce_origin = _ce_at_heat_time(model, x, 1e-6)
  assert torch.allclose(ce_origin, torch.full_like(ce_origin, math.log(model.vocab_size)),
                        atol=0.05)
  ce_boundary = _ce_at_heat_time(model, x, 30.0)
  assert (ce_boundary < 1e-6).all()
  # Beyond the radial sampler's float64 ceiling the heat time is clamped, not
  # NaN'd (alpha_t = 1 is heat time inf).
  ce_far = _ce_at_heat_time(model, x, float('inf'))
  assert torch.isfinite(ce_far).all() and (ce_far < 1e-6).all()
  # The Cartesian point lies inside each factor's ball of radius R = 1.
  xt, _ = model.q_xt(x, _alpha_at_heat_time(model, B, 0.5), False)
  assert (xt.unflatten(-1, (4, 3)).norm(dim=-1) < 1.0).all()


# ---------------------------------------------------------------------------
# hyper_model.HyperbolicModelBase: the two readout fixes
# ---------------------------------------------------------------------------

class _TinyHyperModel(__import__('numeric.hyper_model', fromlist=['HyperbolicModelBase']).HyperbolicModelBase):
  """Identity trunk over the boundary channels + a fixed linear head."""

  def __init__(self, V, dims, curvs):
    super().__init__()
    E = sum(dims)
    g = torch.Generator().manual_seed(0)
    self._emb = torch.randn(V, E, dtype=torch.float64, generator=g)
    self._head = 0.1 * torch.randn(E, V, dtype=torch.float64, generator=g)
    self.prod_factor_dim, self.prod_factor_gaussian_curvature = dims, curvs
    self.output_radial_dim = 0
    self.lm_head = lambda h: h @ self._head
    self.E = E

  @property
  def word_embedding(self):
    return self._emb

  def model_forward(self, z, t):
    return z.double()[..., :self.E]


def test_forward_horosphere_accepts_cartesian_trunk_state_and_own_factors():
  import numeric.hyper_model as hyper_model
  dims, curvs = [3, 3], [-1.0, -4.0]
  m = _TinyHyperModel(V=7, dims=dims, curvs=curvs)
  B, L = 2, 5
  theta = torch.randn(B, L, 2, 3, dtype=torch.float64)
  theta = (theta / theta.norm(dim=-1, keepdim=True)).flatten(-2)
  radius = torch.rand(B, L, 2, dtype=torch.float64) * 2
  z = theta * hyper_model.HyperbolicModelBase.radius_conversion(
    radius, dims, curvs).repeat_interleave(3, dim=-1)
  # Cartesian trunk state + polar readout (used to raise "not both").
  out = m.forward_horosphere(z=z, theta=theta, radius=radius)
  from numeric.horosphere import HorosphereGeometry
  ref = m.forward_naive(z=z, theta=None, radius=None) + HorosphereGeometry.compute_horosphere(
    theta=theta, radius=radius, word_embedding=m.word_embedding, prod_factor_dim=dims,
    prod_factor_gaussian_curvature=curvs)
  assert torch.allclose(out, ref)
  # Both factor lists None fall back to the model's OWN factors (two H^3_K
  # factors, not one H^6), for the polar trunk path too.
  out_polar = m.forward_horosphere(z=None, theta=theta, radius=radius)
  ref_polar = m.forward_naive(z=None, theta=theta, radius=radius) + (
    HorosphereGeometry.compute_horosphere(theta=theta, radius=radius, word_embedding=m.word_embedding,
                                          prod_factor_dim=dims,
                                          prod_factor_gaussian_curvature=curvs))
  assert torch.allclose(out_polar, ref_polar)
  assert out.shape == (B, L, 7)


# ---------------------------------------------------------------------------
# Wiring / validation
# ---------------------------------------------------------------------------

def test_get_sampler_registers_hbfm():
  import samplers
  cfg = _compose()
  assert isinstance(samplers.get_sampler(cfg), samplers.HBFMSampler)


@needs_gpu
def test_adaptive_schedule_composes(monkeypatch):
  import noise_schedules
  model, cfg = _build_model(monkeypatch, ['noise=log-linear-adaptive'])
  assert isinstance(model.noise, noise_schedules.AdaptiveSchedule)
  loss, _ = model.nll(_tokens(model), None, None)
  assert torch.isfinite(loss).all()


@pytest.mark.parametrize('overrides', [
  ['noise=log-linear-adaptive', 'algo.invert_time_convention=true'],
  ['algo.prod_factor_dim=5'],
  ['algo.time_conversion_mode=unif', 'algo.time_range_upper_bound=0'],
  ['algo.time_conversion_mode=unif', 'sampler.t_max=2.0'],  # > time_range_upper_bound=1.0
  ['algo.time_conversion_mode=trunc_exp', 'sampler.t_max=2.0'],
  ['algo.time_exp_rate=0'],
  ['model.sphere_mc_samples=16'],                        # MC without the sphere map
  ['model.random_feature_type=sphere', 'model.sphere_mc_samples=0'],
  ['model.random_feature_type=sphere', 'model.sphere_radial_type=bogus'],
  ['model.random_feature_type=sphere', 'model.sphere_radial_type=constant'],   # see SphericalFeature.radial
  ['model.random_feature_type=sphere', 'model.sphere_radial_type=phase', 'model.sphere_mc_samples=4'],   # no MC phase
  ['model.random_feature_type=sphere', 'model.sphere_degree=null'],
  ['model.random_feature_type=SPHERE'],                  # the value is lower case
  ['model.random_feature_type=hyla'],                    # a class, not a model option
])
def test_validate_configuration_rejects(monkeypatch, overrides):
  with pytest.raises(ValueError):
    _build_model(monkeypatch, overrides)


# ---------------------------------------------------------------------------
# HyLa: random Laplacian features of the state (random_features.hyla)
# ---------------------------------------------------------------------------

class _BoundFeatureMap:
  """A feature map with its product factors bound, as `HyperbolicDiT.forward` calls it."""

  def __init__(self, feature_map, dims, curvs):
    self.feature_map, self.dims, self.curvs = feature_map, dims, curvs

  def __call__(self, z, input_coord=Coordinate.CARTESIAN):
    return self.feature_map(z, input_coord, self.dims, self.curvs)

  def __getattr__(self, name):
    return getattr(self.feature_map, name)


def _hyla(dims, curvs, num_features, scale=1.0, seed=0, radius_cap=None, normalize=True):
  from random_features.hyla import HyLaFeatures
  hyla = _BoundFeatureMap(HyLaFeatures(num_features, len(dims), scale, seed, radius_cap, normalize), dims, curvs)
  hyla(torch.zeros(1, 1, sum(dims), dtype=torch.float64))   # the first forward draws the features
  return hyla


def test_hyla_features_match_the_poincare_formula():
  """Per factor the feature is exp((d-1)/2 P) cos(lambda P + b) / sqrt(D) with
  P = log((1 - |w|^2) / |w - omega|^2) on the unit-ball image w = z / R of the
  factor's state (paper eq. 1); the origin has P = 0, i.e. cos(b) / sqrt(D)."""
  dims, curvs = [3, 3, 3], [-1.0, -0.5, -2.0]
  hyla = _hyla(dims, curvs, 24)
  assert hyla.omegas.shape == (3, 8, 3)
  assert torch.allclose(hyla.omegas.norm(dim=-1), torch.ones(3, 8, dtype=torch.float64))
  torch.manual_seed(1)
  R = torch.tensor([1.0, 2.0 ** 0.5, 0.5 ** 0.5], dtype=torch.float64)
  w = torch.randn(4, 5, 3, 3, dtype=torch.float64)
  w = w / w.norm(dim=-1, keepdim=True) * 0.95 * torch.rand(4, 5, 3, 1, dtype=torch.float64)
  w[0, 0] = 0.0                                          # the origin
  feats = hyla((w * R[:, None]).flatten(-2))             # (4, 5, 9) Cartesian state
  assert feats.shape == (4, 5, 24) and feats.dtype == torch.float64
  P = torch.log((1 - w.square().sum(-1, keepdim=True))
                / (w.unsqueeze(-2) - hyla.omegas).square().sum(-1))   # (4, 5, 3, 8)
  ref = torch.exp(P) * torch.cos(hyla.lambdas * P + hyla.biases) / math.sqrt(24)
  assert torch.allclose(feats, ref.flatten(-2), atol=1e-12)
  assert torch.allclose(feats[0, 0], hyla.biases.cos().flatten() / math.sqrt(24))


def test_hyla_features_estimate_the_isometry_invariant_kernel():
  """E[<phi(x), phi(y)>] = k_lambda(d_H(x, y)) (paper Thm 4.1); on H^3 the
  spherical function is elementary, k_lambda(d) = sin(lambda d) / (2 lambda sinh d)."""
  D, lam = 400_000, 0.8
  hyla = _hyla([3], [-1.0], D, seed=3)
  hyla.lambdas.fill_(lam)                                # one eigenvalue, no lambda mixture
  pts = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.3, -0.2, 0.4],
                      [-0.6, 0.1, 0.2]], dtype=torch.float64)
  feats = hyla(pts[None])[0]                             # (4, D)
  gram = feats @ feats.T
  sq = pts.square().sum(-1)
  dist = torch.acosh(1 + 2 * (pts[:, None] - pts[None]).square().sum(-1)
                     / ((1 - sq[:, None]) * (1 - sq[None]))).clamp_min(1e-12)
  kernel = torch.sin(lam * dist) / (2 * lam * torch.sinh(dist))
  assert torch.allclose(gram.diagonal(), torch.full((4,), 0.5, dtype=torch.float64), atol=0.02)
  assert torch.allclose(gram, kernel, atol=0.02)


def test_hyla_is_not_a_model_option():
  """`HyLaFeatures` stays a class, but neither model config can select it."""
  import omegaconf
  from hydra.errors import ConfigCompositionException
  with pytest.raises(ConfigCompositionException, match='model.hyla_dim'):
    _compose(['model.hyla_dim=32'])
  for name in ('tiny', 'small'):
    keys = omegaconf.OmegaConf.load(REPO_ROOT / 'configs' / 'model' / f'{name}-hyperbolic-dit.yaml')
    assert not any(key.startswith('hyla') for key in keys)


def test_hyla_radius_cap_is_a_finite_pin_radius():
  """With `radius_cap` the map is evaluated at min(s, cap): a boundary-pinned state
  (s ~ 35, `_clean_state`), whose features are all ~0 without the cap, gets the
  features of the same direction at s = cap; states inside the cap are untouched."""
  dims, curvs = [3, 3], [-1.0, -0.5]
  torch.manual_seed(2)
  u = torch.randn(2, 3, 2, 3, dtype=torch.float64)
  u = u / u.norm(dim=-1, keepdim=True)
  R = torch.tensor([1.0, 2.0 ** 0.5], dtype=torch.float64)

  def state(s):
    return (R[:, None] * math.tanh(s / 2) * u).flatten(-2)

  capped, plain = _hyla(dims, curvs, 16, radius_cap=5.0), _hyla(dims, curvs, 16)
  assert torch.allclose(capped(state(35.0)), capped(state(5.0)))
  assert torch.allclose(capped(state(5.0)), plain(state(5.0)))
  assert torch.allclose(capped(state(1.0)), plain(state(1.0)))
  assert plain(state(35.0)).abs().max() < 1e-10
  assert capped(state(35.0)).norm(dim=-1).min() > 1e-2


def test_hyla_unnormalized_features_match_the_reference_code():
  """`normalize=False` is the reference implementation's feature,
  cos(lambda P + b) * PoissonKernel^((d-1)/2) with no 1/sqrt(D) factor."""
  hyla, plain = _hyla([3], [-1.0], 8, normalize=False), _hyla([3], [-1.0], 8)
  z = torch.tensor([[[0.3, -0.2, 0.4]]], dtype=torch.float64)
  assert torch.allclose(hyla(z), plain(z) * math.sqrt(8))


def test_hyla_matches_the_reference_implementation():
  """github.com/ydtydr/HyLa hype/models.py::HyLa.forward, verbatim:
       PsK = (1 - |x|^2) / |x - b|^2;  angles = Lambdas / 2 * log(PsK)
       eigs = cos(angles + bias) * sqrt(PsK) ** (dim - 1)
  on the unit ball (K = -1). Our map is the same function with lambda applied in
  full (the paper's eq. 1), so the reference's `Lambdas` are 2 x our `lambdas`,
  and `normalize=False` drops our 1 / sqrt(D)."""
  hyla = _hyla([3], [-1.0], 16, scale=0.3, seed=5, normalize=False)
  torch.manual_seed(3)
  x = torch.randn(7, 3, dtype=torch.float64)
  x = x / x.norm(dim=-1, keepdim=True) * torch.rand(7, 1, dtype=torch.float64) * 0.9
  b, Lambdas, bias = hyla.omegas[0], 2 * hyla.lambdas[0], hyla.biases[0]
  X = x.view(x.size(0), 1, x.size(-1))
  PsK = (1 - torch.norm(X, 2, dim=-1) ** 2) / (torch.norm(X - b, 2, dim=-1) ** 2)
  angles = Lambdas / 2.0 * torch.log(PsK)
  eigs = torch.cos(angles + bias) * torch.sqrt(PsK) ** (3 - 1)
  assert torch.allclose(hyla(x[None])[0], eigs, atol=1e-12)


def test_hyla_radius_cap_stops_the_gradient_at_the_pin():
  """Past the cap the map is evaluated at a frozen pin: the features of a pinned
  state do not backpropagate into its direction (the word embedding), while a
  state inside the cap keeps its full gradient."""
  hyla = _hyla([3, 3], [-1.0, -0.5], 16, radius_cap=5.0)
  torch.manual_seed(4)
  u = torch.randn(1, 2, 2, 3, dtype=torch.float64)
  u = u / u.norm(dim=-1, keepdim=True)
  R = torch.tensor([1.0, 2.0 ** 0.5], dtype=torch.float64)
  s = torch.tensor([35.0, 1.0], dtype=torch.float64)        # position 0 pinned, position 1 inside
  z = (R[:, None] * torch.tanh(s / 2)[None, :, None, None] * u).flatten(-2).requires_grad_(True)
  hyla(z).square().sum().backward()
  assert z.grad[0, 0].abs().max() == 0 and z.grad[0, 1].abs().max() > 0


# ---------------------------------------------------------------------------
# sphere: hyperbolic-harmonic spherical-harmonic features (random_features.hyperbolic_spherical_feat)
# ---------------------------------------------------------------------------

def _sphere(dims, curvs, degree, radial_scales=(1.0, 4.0, 16.0), mc_samples=None, radial_type='exp_decay'):
  from random_features.hyperbolic_spherical_feat import SphericalFeature
  return _BoundFeatureMap(
    SphericalFeature(len(dims), degree, list(radial_scales), radial_type=radial_type, mc_samples=mc_samples),
    dims, curvs)


def _unit(*shape, seed=0):
  torch.manual_seed(seed)
  u = torch.randn(*shape, 3, dtype=torch.float64)
  return u / u.norm(dim=-1, keepdim=True)


def test_harmonic_radial_functions_solve_the_h3_laplace_equation():
  """f_l(s) = r^l 2F1(l, -1/2; l+3/2; r^2) / 2F1(l, -1/2; l+3/2; 1), r = tanh(s/2),
  is the radial part of a harmonic function f_l(s) Y_lm(u) on H^3:
  f'' + 2 coth(s) f' - l(l+1) f / sinh^2(s) = 0 (checked on the table by central
  differences), regular at the origin (f_l(0) = [l = 0]), monotone, and 1 at the
  boundary. For l = 1 the series sums in closed form, which also checks the
  tabulation: with a = atanh(r) = s/2,
  2F1(1, -1/2; 5/2; r^2) = -3 [(r a - 1)/8 - a/(4r) + (a/r - 1)/(8 r^2)]."""
  sphere = _sphere([3], [-1.0], 8)
  h = sphere.TABLE_STEP
  f = sphere.table[:-1]                                      # (N, 9) at s = 0, h, 2h, ...
  s = torch.arange(f.shape[0], dtype=torch.float64) * h
  ell = torch.arange(9, dtype=torch.float64)
  d1 = (f[2:] - f[:-2]) / (2 * h)
  d2 = (f[2:] - 2 * f[1:-1] + f[:-2]) / h ** 2
  sm = s[1:-1, None]
  residual = d2 + 2 / torch.tanh(sm) * d1 - ell * (ell + 1) / torch.sinh(sm) ** 2 * f[1:-1]
  inside = (sm[:, 0] > 0.25) & (sm[:, 0] < 12)
  assert residual[inside].abs().max() < 1e-4
  assert torch.allclose(f[0], torch.tensor([1.0] + [0.0] * 8, dtype=torch.float64))
  assert torch.allclose(f[-1], torch.ones(9, dtype=torch.float64), atol=1e-12)
  assert (f[1:] >= f[:-1] - 1e-12).all()
  for s_ in (0.5, 1.0, 2.0, 4.0, 8.0):
    r, a = math.tanh(s_ / 2), s_ / 2
    closed = (4 / 3) * r * -3 * ((r * a - 1) / 8 - a / (4 * r) + (a / r - 1) / (8 * r * r))
    assert abs(float(sphere.radial(torch.tensor(s_, dtype=torch.float64))[1]) - closed) < 1e-5
  # The design note's table (2 decimals).
  doc = {(1, 0.5): 0.32, (1, 1): 0.59, (1, 2): 0.89, (1, 3): 0.98, (2, 0.5): 0.09, (2, 1): 0.32,
         (2, 2): 0.76, (2, 3): 0.94, (4, 1): 0.09, (4, 2): 0.52, (4, 3): 0.85, (4, 4): 0.97,
         (8, 2): 0.22, (8, 3): 0.67, (8, 4): 0.91}
  for (l, s_), v in doc.items():
    assert abs(float(sphere.radial(torch.tensor(s_, dtype=torch.float64))[l]) - v) < 0.006, (l, s_)


def test_spherical_harmonics_are_orthonormal():
  """The real Y_lm, l <= 6, are orthonormal under the uniform probability measure
  on S^2 (so each has unit RMS): Gauss-Legendre in cos(theta) x the trapezoid
  rule in phi, exact at this degree. Y_00 = 1 and the l = 1 block is
  sqrt(3) (y, z, x) (m = -1, 0, 1): the state's own coordinates."""
  import numpy as np
  sphere = _sphere([3], [-1.0], 6)
  zs, ws = np.polynomial.legendre.leggauss(8)
  z = torch.tensor(zs, dtype=torch.float64)[:, None].expand(8, 16)
  phi = (2 * math.pi * torch.arange(16, dtype=torch.float64) / 16)[None].expand(8, 16)
  u = torch.stack([(1 - z ** 2).sqrt() * phi.cos(), (1 - z ** 2).sqrt() * phi.sin(), z], -1)
  Y = sphere.spherical_harmonics(u)                                 # (8, 16, 49)
  w = (torch.tensor(ws, dtype=torch.float64) / 2)[:, None].expand(8, 16) / 16
  gram = torch.einsum('ij,ijk,ijl->kl', w, Y, Y)
  assert torch.allclose(gram, torch.eye(49, dtype=torch.float64), atol=1e-10)
  assert torch.allclose(Y[..., 0], torch.ones(8, 16, dtype=torch.float64))
  assert torch.allclose(Y[..., 1:4], math.sqrt(3) * u[..., [1, 2, 0]])


def test_harmonic_features_are_bounded_and_finite_up_to_the_pin():
  """Over the whole radial range the readback can produce, s in [0, 34.66] (the
  boundary pin of `_clean_state` included), every feature is finite,
  |f_l Y_lm| <= sqrt(2l + 1), the radial channels lie in [0, 1], and the gradient
  wrt the state is finite: at the origin, and at the pin, where the direction
  still carries a bounded gradient into the word embedding."""
  dims, curvs = [3, 3], [-1.0, -0.5]
  R = torch.tensor([1.0, 2.0 ** 0.5], dtype=torch.float64)
  sphere = _sphere(dims, curvs, 6)
  n = 64
  s = torch.cat([torch.zeros(1), torch.linspace(1e-3, 34.66, n - 2), torch.tensor([34.66])]).double()
  u = _unit(n, 2, seed=5)
  z = (R[:, None] * torch.tanh(s / 2)[:, None, None] * u).flatten(-2)[None].requires_grad_(True)
  feats = sphere(z)
  assert feats.shape == (1, n, 2 * (49 + 3)) and feats.dtype == torch.float64
  assert torch.isfinite(feats).all()
  per = feats.unflatten(-1, (2, 52))
  assert (per[..., :49].abs() <= (2 * sphere.ell_of_feature + 1).double().sqrt() + 1e-9).all()
  assert ((per[..., 49:] >= 0) & (per[..., 49:] <= 1)).all()     # tanh(34.66 / 1) == 1.0 in float64
  feats.square().sum().backward()
  assert torch.isfinite(z.grad).all()
  pin = (R[:, None] * u[0]).flatten()[None, None].requires_grad_(True)   # |z| = R, as `_clean_state` emits it
  fp = sphere(pin)
  assert torch.isfinite(fp).all()
  assert torch.allclose(fp.unflatten(-1, (2, 52))[0, 0, :, :49], sphere.spherical_harmonics(u[0]), atol=1e-9)
  fp.square().sum().backward()
  assert torch.isfinite(pin.grad).all() and 0 < pin.grad.abs().max() < 1e4


def test_harmonic_features_converge_to_the_spherical_harmonics():
  """s -> inf: f_l -> 1 and h_lm -> Y_lm(u), the Poisson integral's boundary
  values (within 1e-6 by s = 16); at the origin only the constant l = 0 survives;
  in between the l = 1 block is f_1(s) times the coordinates' direction."""
  sphere = _sphere([3], [-1.0], 4, radial_scales=[2.0])
  u = _unit(5, 1, seed=6)                                     # (5, 1, 3)
  Y = sphere.spherical_harmonics(u[:, 0])                     # (5, 25)
  for s_ in (16.0, 25.0, 34.66):
    feats = sphere((math.tanh(s_ / 2) * u).flatten(-2)[None])[0]        # (5, 26)
    assert torch.allclose(feats[:, :25], Y, atol=1e-6)
    assert torch.allclose(feats[:, 25], torch.full((5,), math.tanh(s_ / 2), dtype=torch.float64), atol=1e-6)
  origin = sphere(torch.zeros(1, 5, 3, dtype=torch.float64))[0]
  assert torch.allclose(origin, torch.tensor([1.0] + [0.0] * 25, dtype=torch.float64).expand(5, -1))
  feats = sphere((math.tanh(0.5) * u).flatten(-2)[None])[0]
  f1 = float(sphere.radial(torch.tensor(1.0, dtype=torch.float64))[1])
  assert 0.5 < f1 < 0.7 and torch.allclose(feats[:, 1:4], f1 * Y[:, 1:4])


@needs_gpu
def test_sphere_backbone_end_to_end(monkeypatch):
  model, _ = _build_model(monkeypatch, ['model.random_feature_type=sphere', 'model.sphere_degree=4', 'model.sphere_radial_scales=[1,4,16]'])
  sphere = _BoundFeatureMap(model.backbone.random_feature, *model.backbone.prod_factors)
  assert sphere.degree == 4
  assert sphere.num_features == 4 * (25 + 3) == model.backbone.in_proj.in_features
  assert not any(k.startswith('backbone.random_feature.') for k in model.state_dict())   # nothing random
  B, L = 2, 8
  x = _tokens(model, B, L)
  loss, _ = model.nll(x, None, None)
  assert loss.shape == (B, L) and torch.isfinite(loss).all()
  loss.sum().backward()
  for p in (model.backbone.sphere_embed.weight, model.backbone.in_proj.weight):
    assert p.grad is not None and torch.isfinite(p.grad).all()
  # The sudoku prompt's boundary pin (`_clean_state`) is seen through its
  # direction: the features there are Y_lm of the word's direction; the
  # origin (the sampler's start) is seen too.
  pin = model._clean_state(x)
  u = torch.nn.functional.normalize(model.word_embedding[x].double().unflatten(-1, (4, 3)), dim=-1)
  assert torch.allclose(sphere(pin).unflatten(-1, (4, 28))[..., :25], sphere.spherical_harmonics(u), atol=1e-9)
  sigma = model._sigma_from_alphat(_alpha_at_heat_time(model, B, 0.3))
  for state in (pin, torch.zeros(B, L, 12, dtype=torch.float64, device=model.device)):
    assert torch.isfinite(model.forward(xt=state, sigma=sigma)).all()


def test_feature_factory_builds_the_sphere_map_and_rejects_the_rest():
  from random_features.feature_factory import RandomFeatureFactory, RandomFeatureType
  from random_features.hyperbolic_spherical_feat import SphericalFeature
  sphere = RandomFeatureFactory.create_random_features(
    RandomFeatureType.SPHERE, degree=2, radial_scales=[1.0], num_factors=3, harmonic_mc_samples=4)
  assert isinstance(sphere, SphericalFeature)
  assert (sphere.degree, sphere.mc_samples, sphere.radial_type) == (2, 4, 'exp_decay')
  assert sphere.num_features == 3 * (9 + 1)
  with pytest.raises(ValueError):
    RandomFeatureFactory.create_random_features('hyla', degree=2, radial_scales=[1.0])


# sphere_radial_type: the radial profile of the harmonic features

def test_harmonic_rejects_unsupported_radial_types():
  for radial_type in ('bogus', 'constant'):               # constant: see SphericalFeature.radial
    with pytest.raises(ValueError):
      _sphere([3], [-1.0], 2, radial_type=radial_type)


def test_harmonic_state_gradient_is_bounded_at_the_origin():
  """f_l(s) ~ s^l cancels the 1 / s of du/dz, so the gradient w.r.t. the state stays bounded
  as s -> 0 and finite exactly at the origin: the property a flat f_l = 1 loses."""
  sphere = _sphere([3], [-1.0], 4)
  u = _unit(1, 1, seed=13)
  cotangent = torch.randn(1, 1, sphere.num_features, dtype=torch.float64)
  for s in (1e-1, 1e-3, 1e-6, 0.0):
    z = (math.tanh(s / 2) * u).clone().requires_grad_(True)
    (cotangent * sphere(z)).sum().backward()
    assert z.grad.norm() < 20.0, s


# From `HarmonicFeatures` at commit 7830625, the code behind the recorded experiment numbers, for the state below.
SPHERE_GOLDEN_STATE = [[[0.3, -0.2, 0.5, -0.4, 0.7, 0.1], [0.0, 0.0, 0.0, 0.8485, 0.0, 1.1313]]]
SPHERE_GOLDEN_FEATURES = [
  1, -0.42457729118358295, 1.0614432279589574, 0.63686593677537429,
  -0.32844615759867513, -0.54741026266445869, 0.58468780563700451, 0.82111539399668798,
  0.13685256566611462, -0.15183230691487243, -0.48510218923185316, -0.44486806661293238,
  0.11481510945078345, 0.66730209991939859, 0.20212591217993878, -0.029706320918127228,
  0.89339333376361973, 0.34488146399061514, 1, 1.063631028833639,
  0.15194728983337702, -0.60778915933350808, -0.78060637638617147, 0.19515159409654287,
  -0.50701871422990619, -0.11151519662659591, -0.46000018608470805, -0.0083524721201448503,
  -0.16367435828141991, -0.39465762213599376, -0.1456479241883919, 0.22551864122056786,
  -0.096450961130122428, 0.62524219870799236, 0.86384400699819985, 0.31588953585894847,
  1, 0, 0, 0,
  0, 0, -0, 0,
  0, 0, 0, -0,
  -0, -0, 0, 0,
  0, 0, 1, 0,
  1.3856259105064692, 1.0392500530935553, 0, 0,
  1.0285456505000961, 1.8590471982398249, 0.6971632403900343, 0,
  0, 0, 0.21158598497487319, 2.1386184447295236,
  1.4756006871007035, 0.45182190580912662, 0.99999999864707911, 0.98985197869765396,
]
SPHERE_GOLDEN_GRAD = [
  -1.9520729503980525, -9.3195914053073192, -4.3567245861129429, -2.1913201927807959,
  -0.63221322588785878, -2.081550662102543, 0, 0,
  0, 40.679510650778987, 5.8265345360743552, 55.295514761994184,
]


def test_sphere_features_match_the_recorded_model():
  """(H^3)^2 at K = -1, -0.5, degree 3: features and gradient at an interior state, at one
  factor's origin and at a factor near the boundary stay what the recorded runs computed."""
  sphere = _sphere([3, 3], [-1.0, -0.5], 3, radial_scales=(1.0, 4.0))
  z = torch.tensor(SPHERE_GOLDEN_STATE, dtype=torch.float64, requires_grad=True)
  out = sphere(z)
  expected = torch.tensor(SPHERE_GOLDEN_FEATURES, dtype=torch.float64)
  assert torch.allclose(out.detach().flatten(), expected, rtol=1e-9, atol=1e-10)
  (torch.linspace(-1.0, 1.0, 36, dtype=torch.float64) * out).sum().backward()
  expected = torch.tensor(SPHERE_GOLDEN_GRAD, dtype=torch.float64)
  assert torch.allclose(z.grad.flatten(), expected, rtol=1e-9, atol=1e-10)


def test_sphere_radial_type_reaches_the_backbone(monkeypatch):
  model, _ = _build_model(monkeypatch, ['model.random_feature_type=sphere', 'model.sphere_radial_type=exp_decay'])
  assert model.backbone.random_feature.radial_type == 'exp_decay'


def test_random_features_are_off_by_default_with_the_best_sphere_settings(monkeypatch):
  """`random_feature_type` alone is the switch; the `sphere_*` defaults are the best settings
  measured (experiments/claude_test_hbfm_rlf_*/RESULTS.md, stages 4-7)."""
  import omegaconf
  from models.hyperbolic_dit import HyperbolicDiT
  for name in ('tiny', 'small'):
    config = omegaconf.OmegaConf.load(REPO_ROOT / 'configs' / 'model' / f'{name}-hyperbolic-dit.yaml')
    assert config.random_feature_type is None
    assert (config.sphere_degree, list(config.sphere_radial_scales), config.sphere_radial_type,
            config.sphere_mc_samples) == (4, [1, 2, 4, 8, 16, 32], 'exp_decay', None)
  off, _ = _build_model(monkeypatch)
  assert off.backbone.random_feature is None and off.backbone.in_proj.in_features == 12
  on, _ = _build_model(monkeypatch, ['model.random_feature_type=sphere'])
  assert on.backbone.random_feature.degree == 4 and on.backbone.in_proj.in_features == 4 * (25 + 6)
  # A config written before the key existed keeps the coordinates.
  older = _compose()
  with omegaconf.open_dict(older.model):
    del older.model['random_feature_type']
  assert HyperbolicDiT(older, vocab_size=12).random_feature is None


# sphere_mc_samples: Monte-Carlo sphere features, w ~ P(z, .) (unit importance weights)

def test_harmonic_mc_off_is_the_closed_form():
  """mc_samples=None is the closed form f_l(s) Y_lm(u) next to the radial
  channels, bit for bit, and draws nothing from the global RNG."""
  from numeric.geo_bridge import GeoUtils
  dims, curvs = [3, 3], [-1.0, -0.5]
  R = torch.tensor([1.0, 2.0 ** 0.5], dtype=torch.float64)
  sphere = _sphere(dims, curvs, 4)
  u = _unit(3, 5, 2, seed=11)
  s = 3 * torch.rand(3, 5, dtype=torch.float64)
  z = (R[:, None] * torch.tanh(s / 2)[..., None, None] * u).flatten(-2)   # (3, 5, 6)
  rng = torch.get_rng_state()
  out = sphere(z)
  assert torch.equal(torch.get_rng_state(), rng)
  rhos, us = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(z, dims, curvs)
  kappas = rhos.new_tensor([1.0 / GeoUtils._curvature_scale(k) for k in curvs])
  us, ss = us.unflatten(-1, (2, 3)), rhos * kappas
  ref = torch.cat([sphere.spherical_harmonics(us) * sphere.radial(ss)[..., sphere.ell_of_feature],
                   torch.tanh(ss.unsqueeze(-1) / sphere.radial_scales)], dim=-1).flatten(-2)
  assert torch.equal(out, ref)


def test_harmonic_mc_is_unbiased_with_the_closed_form_variance():
  """w ~ P(z, .) exactly, so the mean of Y_lm(w) over the M draws is unbiased for
  the closed form f_l(s) Y_lm(u) (within 5 standard errors per feature), and by
  the addition theorem sum_m Var = (2l + 1)(1 - f_l^2) / M (within 15 %): the
  draws are independent across samples and positions. The two factors sit at
  different radii, so each is sampled at its own s."""
  dims, curvs = [3, 3], [-1.0, -0.5]
  R = torch.tensor([1.0, 2.0 ** 0.5], dtype=torch.float64)
  M, reps, deg = 16, 4000, 4
  n = (deg + 1) ** 2
  mc, closed = _sphere(dims, curvs, deg, mc_samples=M), _sphere(dims, curvs, deg)
  radii = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
  sr = torch.tensor([radii, radii[::-1]], dtype=torch.float64).T        # (6, 2): factor 1 reversed
  u = _unit(len(radii), 1, 2, seed=12).expand(-1, reps, -1, -1)     # one direction pair per position
  s = sr[:, None].expand(-1, reps, -1)                                 # (6, reps, 2)
  z = (R[:, None] * torch.tanh(s / 2)[..., None] * u).flatten(-2).reshape(1, -1, 6)
  torch.manual_seed(0)
  est = mc(z).unflatten(-1, (2, n + 3))[0, ..., :n].unflatten(0, (len(radii), reps))   # (6, reps, 2, 25)
  ref = closed(z).unflatten(-1, (2, n + 3))[0, ..., :n].unflatten(0, (len(radii), reps))[:, 0]
  ell = mc.ell_of_feature.double()
  f = mc.radial(sr)[..., mc.ell_of_feature]                            # (6, 2, 25)
  sum_var = (2 * ell + 1) * (1 - f ** 2)          # sum_m Var[Y_lm(w)] / M, per degree
  bound = 5 * (est.var(1) / reps).sqrt() + 1e-12                       # each feature's own SE
  assert ((est.mean(1) - ref).abs() <= bound).all()
  var = est.var(1) * M                                                 # (6, 2, 25)
  for i in range(len(radii)):
    for j in range(2):
      if float(sr[i, j]) not in (0.5, 1.0, 2.0):
        continue
      for l in range(1, deg + 1):
        sel = mc.ell_of_feature == l
        emp, want = float(var[i, j, sel].sum()), float(sum_var[i, j, sel][0])
        assert abs(emp / want - 1) < 0.15, (float(sr[i, j]), l, emp, want)


def test_harmonic_mc_is_bounded_and_exact_at_the_pin(monkeypatch):
  """Unit weights make the estimate a convex combination of Y_lm values: over the
  readback range s in [0, 34.66] it is finite, |h_lm| <= sqrt(2l + 1), the l = 0
  feature is 1 and the radial channels are the closed form's; at the boundary pin
  the kernel is a point mass and the estimate is Y_lm(u). A first coordinate
  rounded to c0 = -1 at s = 34.66 makes the boost scale the rest by ~e^s; the
  forward's renormalisation restores |w| = 1."""
  from numeric.geo_bridge import HyperbolicHeatKernel as HK
  dims, curvs = [3, 3], [-1.0, -0.5]
  R = torch.tensor([1.0, 2.0 ** 0.5], dtype=torch.float64)
  mc, closed = _sphere(dims, curvs, 6, mc_samples=16), _sphere(dims, curvs, 6)
  bound = (2 * mc.ell_of_feature + 1).double().sqrt() + 1e-9
  n = 64
  s = torch.cat([torch.zeros(1), torch.linspace(1e-3, 34.66, n - 2), torch.tensor([34.66])]).double()
  u = _unit(n, 2, seed=5)
  z = (R[:, None] * torch.tanh(s / 2)[:, None, None] * u).flatten(-2)[None]
  torch.manual_seed(0)
  per = mc(z).unflatten(-1, (2, 52))
  assert torch.isfinite(per).all()
  assert (per[..., :49].abs() <= bound).all()
  assert (per[..., 0] - 1).abs().max() <= 1e-12
  assert torch.equal(per[..., 49:], closed(z).unflatten(-1, (2, 52))[..., 49:])
  pin = (R[:, None] * u[0]).flatten()[None, None]       # |z| = R, as `_clean_state` emits it
  fp = mc(pin).unflatten(-1, (2, 52))[0, 0, :, :49]
  assert torch.allclose(fp, mc.spherical_harmonics(u[0]), rtol=0, atol=1e-12)

  def crafted(shape, d, dtype, device):                 # c0 = -1.0 exactly, rest ~1e-8
    u0 = torch.zeros(*shape, d, dtype=dtype, device=device)
    u0[..., 0], u0[..., 1] = -1.0, 1e-8
    return u0
  monkeypatch.setattr(HK, '_free_direction', staticmethod(crafted))
  raw = HK._reflect_to_target(HK._angular_boost(torch.full((4,), 34.66, dtype=torch.float64), 3),
                              u[0, :1].expand(4, 3))
  assert raw.norm(dim=-1).min() > 1e6                   # the hazard
  w = raw / raw.norm(dim=-1, keepdim=True)
  assert ((w.norm(dim=-1) - 1).abs() <= 1e-12).all()
  fp = mc(pin).unflatten(-1, (2, 52))[..., :49]         # the forward under the crafted draws
  assert torch.isfinite(fp).all() and (fp.abs() <= bound).all()


def test_harmonic_mc_gradient_is_the_closed_form_gradient():
  """The draws carry no graph: the gradient wrt the state is the closed form's,
  bit for bit, from the origin to the boundary pin, where it stays bounded."""
  dims, curvs = [3, 3], [-1.0, -0.5]
  R = torch.tensor([1.0, 2.0 ** 0.5], dtype=torch.float64)
  mc, closed = _sphere(dims, curvs, 6, mc_samples=16), _sphere(dims, curvs, 6)
  n = 64
  s = torch.cat([torch.zeros(1), torch.linspace(1e-3, 34.66, n - 2), torch.tensor([34.66])]).double()
  u = _unit(n + 1, 2, seed=7)
  z = torch.cat([(R[:, None] * torch.tanh(s / 2)[:, None, None] * u[:n]).flatten(-2),
                 (R[:, None] * u[n]).flatten()[None]])[None]           # grid (origin first) + the pin
  a = torch.randn(1, n + 1, 2 * 52, dtype=torch.float64)               # a fixed cotangent
  grads = []
  for m in (mc, closed):
    zz = z.clone().requires_grad_(True)
    (a * m(zz)).sum().backward()
    grads.append(zz.grad)
  assert torch.equal(grads[0], grads[1])
  assert torch.isfinite(grads[0]).all()
  assert 0 < grads[0][0, n].abs().max() < 1e4


def test_harmonic_mc_draws_are_fresh_seeded_and_shape_determined():
  """Fresh draws every forward from the global RNG, the same in train and eval;
  the number drawn depends on the shape only, B * L * m * M * 3 normals (none
  shared across the batch or the factors); no checkpoint keys."""
  dims, curvs = [3, 3], [-1.0, -0.5]
  R = torch.tensor([1.0, 2.0 ** 0.5], dtype=torch.float64)
  mc = _sphere(dims, curvs, 4, mc_samples=8)
  z = (R[:, None] * math.tanh(0.5) * _unit(2, 3, 2, seed=8)).flatten(-2)    # s = 1, B = 2
  z2 = (R[:, None] * math.tanh(1.5) * _unit(2, 3, 2, seed=9)).flatten(-2)   # s = 3, same shape
  assert not torch.equal(mc(z), mc(z))
  same = mc(z[:1].expand(2, -1, -1))
  assert not torch.equal(same[0], same[1])                                  # fresh per batch row
  torch.manual_seed(5)
  mc(z)
  st = torch.get_rng_state()
  torch.manual_seed(5)
  torch.randn(2, 3, 2, 8, 3, dtype=torch.float64)                           # (B, L, m, M, 3)
  assert torch.equal(torch.get_rng_state(), st)
  torch.manual_seed(3)
  a = mc(z)
  torch.manual_seed(3)
  assert torch.equal(mc(z), a)
  mc.eval()
  torch.manual_seed(3)
  assert torch.equal(mc(z), a)
  mc.train()
  torch.manual_seed(4)
  mc(z)
  st = torch.get_rng_state()
  torch.manual_seed(4)
  mc(z2)
  assert torch.equal(torch.get_rng_state(), st)
  assert len(mc.state_dict()) == 0


@needs_gpu
def test_sphere_mc_backbone_end_to_end(monkeypatch):
  base = ['model.random_feature_type=sphere', 'model.sphere_degree=4', 'model.sphere_radial_scales=[1,4,16]']
  model, _ = _build_model(monkeypatch, base + ['model.sphere_mc_samples=8'])
  sphere = _BoundFeatureMap(model.backbone.random_feature, *model.backbone.prod_factors)
  assert sphere.mc_samples == 8 and model.backbone.in_proj.in_features == 4 * (25 + 3)
  B, L = 2, 8
  x = _tokens(model, B, L)
  loss, _ = model.nll(x, None, None)
  assert loss.shape == (B, L) and torch.isfinite(loss).all()
  loss.sum().backward()
  for p in (model.backbone.sphere_embed.weight, model.backbone.in_proj.weight):
    assert p.grad is not None and torch.isfinite(p.grad).all()
  # Checkpoints are interchangeable with the closed-form map's.
  closed, _ = _build_model(monkeypatch, base)
  assert set(model.state_dict()) == set(closed.state_dict())
  model.load_state_dict(closed.state_dict(), strict=True)
  closed.load_state_dict(model.state_dict(), strict=True)
  pin = model._clean_state(x)
  u = torch.nn.functional.normalize(model.word_embedding[x].double().unflatten(-1, (4, 3)), dim=-1)
  assert torch.allclose(sphere(pin).unflatten(-1, (4, 28))[..., :25], sphere.spherical_harmonics(u), atol=1e-9)
  sigma = model._sigma_from_alphat(_alpha_at_heat_time(model, B, 0.3))
  for state in (pin, torch.zeros(B, L, 12, dtype=torch.float64, device=model.device)):
    assert torch.isfinite(model.forward(xt=state, sigma=sigma)).all()


# sphere_radial_type=phase: the boundary-phase eigenfunctions r_{2+i/c,l} / C (the radial design note)

def _phase_closed_form(lam, l, s):
  """r_{mu,l}(s) / C(mu, l), mu = 2 + i lam, in closed form for l <= 1 (the Rodrigues
  step of the design note, k = mu - 1): l = 0 is sinh(k s) / sinh(s), l = 1 is
  (k cosh(ks) sinh(s) - sinh(ks) cosh(s)) / ((k - 1) sinh^2(s)) (lam > 0)."""
  import numpy as np
  k = 1 + 1j * lam
  if l == 0:
    return np.sinh(k * s) / np.sinh(s)
  return (k * np.cosh(k * s) * np.sinh(s) - np.sinh(k * s) * np.cosh(s)) / ((k - 1) * np.sinh(s) ** 2)


def test_phase_radial_matches_the_closed_forms():
  """The ODE table (`phase_radial`) is the regular solution of the H^3 radial
  eigen-equation, boundary-normalised: it equals the closed forms for l <= 1, is the
  harmonic f_l at mu = 2 (lambda = 0) for every l, tends to e^{i lambda s} (modulus 1)
  at the boundary for every l, and stays bounded in between (the note's sup: sqrt(1 +
  lambda^2) at l = 0, 1.08 at l >= 1)."""
  import numpy as np
  from random_features.hyperbolic_spherical_feat import SphericalFeature
  sphere = _sphere([3], [-1.0], 4)
  s = np.arange(0.0, 34.66, 1.0 / 64)
  for l in range(5):
    f = SphericalFeature.phase_radial(2 + 0j, l, s)
    table = sphere.radial(torch.tensor(s, dtype=torch.float64))[:, l].numpy()
    assert np.abs(f - table).max() < 1e-5, l      # the f_l table is itself interpolated
    assert np.abs(f.imag).max() < 1e-12
  pts = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
  for lam in (1.0, 0.25, 1 / 32):
    for l in (0, 1):
      got = SphericalFeature.phase_radial(2 + 1j * lam, l, np.concatenate([[0.0], pts]))[1:]
      assert np.abs(got - _phase_closed_form(lam, l, pts)).max() < 1e-7, (lam, l)
    for l in range(5):
      r = SphericalFeature.phase_radial(2 + 1j * lam, l, s)
      assert np.isfinite(r).all()
      assert np.abs(r).max() <= (math.sqrt(1 + lam ** 2) if l == 0 else 1.08) + 1e-9, (lam, l)
      far = SphericalFeature.phase_radial(2 + 1j * lam, l, np.array([0.0, 16.0, 34.66]))[1:]
      assert np.abs(far - np.exp(1j * lam * np.array([16.0, 34.66]))).max() < 1e-6, (lam, l)


def test_phase_features_layout_bounds_and_pin():
  """radial_type='phase': per factor the (L+1)^2 harmonics f_l Y_lm (as exp_decay's),
  then Re and Im of Y_lm(u) r_{2+i/c,l}(s) / C per scale c; finite with a finite
  gradient from the origin to the pin, where they are Y_lm(u) cos / sin(s / c); at the
  origin only l = 0 survives, r_{mu,0}(0) / C = mu - 1 = 1 + i / c."""
  import numpy as np
  from random_features.hyperbolic_spherical_feat import SphericalFeature
  dims, curvs, scales = [3, 3], [-1.0, -0.5], [1.0, 4.0]
  R = torch.tensor([1.0, 2.0 ** 0.5], dtype=torch.float64)
  phase = _sphere(dims, curvs, 4, scales, radial_type='phase')
  exp_decay = _sphere(dims, curvs, 4, scales)
  assert phase.num_features == 2 * 25 * (1 + 2 * 2) and len(phase.state_dict()) == 0
  n = 32
  s = torch.cat([torch.zeros(1), torch.linspace(1e-3, 34.66, n - 2), torch.tensor([34.66])]).double()
  u = _unit(n, 2, seed=5)
  z = (R[:, None] * torch.tanh(s / 2)[:, None, None] * u).flatten(-2)[None].requires_grad_(True)
  feats = phase(z)
  assert feats.shape == (1, n, 250) and feats.dtype == torch.float64 and torch.isfinite(feats).all()
  per = feats.unflatten(-1, (2, 125))[0]                                  # (n, factor, 125)
  assert torch.allclose(per[..., :25], exp_decay(z.detach()).unflatten(-1, (2, 27))[0, ..., :25])
  Y = phase.spherical_harmonics(u)                                        # (n, 2, 25)
  ch = per[..., 25:].unflatten(-1, (2, 25, 2))                            # (n, factor, c, lm, re/im)
  ell = phase.ell_of_feature
  from numeric.geo_bridge import GeoUtils
  # The radius the map reads back from the state (near the pin it differs from s by
  # float64 rounding of tanh(s / 2), which the c = 1 phase resolves).
  s_read = (GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(z.detach(), dims, curvs)[0] / R)[0]   # (n, factor)
  for j, c in enumerate(scales):
    for i in (5, 17, n - 1):
      for f in range(2):
        ref = np.stack([SphericalFeature.phase_radial(2 + 1j / c, l, np.array([0.0, float(s_read[i, f])]))[1]
                        for l in range(5)])[ell.numpy()]
        ref = torch.tensor(np.stack([ref.real, ref.imag], -1))            # (25, 2)
        assert torch.allclose(ch[i, f, j], Y[i, f, :, None] * ref, atol=1e-5), (c, i, f)
  assert torch.allclose(ch[0, :, :, ell == 0, 0], torch.ones(2, 2, 1, dtype=torch.float64))
  assert torch.allclose(ch[0, :, :, ell == 0, 1], torch.tensor([1.0, 0.25], dtype=torch.float64)[None, :, None])
  assert ch[0, :, :, ell > 0].abs().max() < 1e-9
  feats.square().sum().backward()
  assert torch.isfinite(z.grad).all()
  pin = (R[:, None] * u[0]).flatten()[None, None]                         # |z| = R, the `_clean_state` pin
  fp = phase(pin).unflatten(-1, (2, 125))[0, 0, :, 25:].unflatten(-1, (2, 25, 2))
  s_pin = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(pin, dims, curvs)[0][0, 0] / R
  assert (s_pin > 34).all()                                               # the readback's largest radius
  for j, c in enumerate(scales):
    rot = torch.stack([(s_pin / c).cos(), (s_pin / c).sin()], -1)          # (factor, 2)
    assert torch.allclose(fp[:, j], Y[0, :, :, None] * rot[:, None], atol=1e-4), c


@needs_gpu
def test_sphere_phase_backbone_end_to_end(monkeypatch):
  base = ['model.random_feature_type=sphere', 'model.sphere_degree=2', 'model.sphere_radial_scales=[1,4]']
  model, _ = _build_model(monkeypatch, base + ['model.sphere_radial_type=phase'])
  sphere = model.backbone.random_feature
  assert sphere.radial_type == 'phase' and model.backbone.in_proj.in_features == 4 * 9 * (1 + 2 * 2)
  assert not any(k.startswith('backbone.random_feature.') for k in model.state_dict())
  B, L = 2, 8
  x = _tokens(model, B, L)
  loss, _ = model.nll(x, None, None)
  assert loss.shape == (B, L) and torch.isfinite(loss).all()
  loss.sum().backward()
  for p in (model.backbone.sphere_embed.weight, model.backbone.in_proj.weight):
    assert p.grad is not None and torch.isfinite(p.grad).all()
  sigma = model._sigma_from_alphat(_alpha_at_heat_time(model, B, 0.3))
  for state in (model._clean_state(x), torch.zeros(B, L, 12, dtype=torch.float64, device=model.device)):
    assert torch.isfinite(model.forward(xt=state, sigma=sigma)).all()


# ---------------------------------------------------------------------------
# FeatureBase: the feature maps read the state in any `Coordinate`
# ---------------------------------------------------------------------------

FACTOR_DIMS, FACTOR_CURVS = [3, 3], [-1.0, -0.5]
FACTOR_RADII = torch.tensor([1.0, 2.0 ** 0.5], dtype=torch.float64)


def _feature_maps():
  return _hyla(FACTOR_DIMS, FACTOR_CURVS, 16), _sphere(FACTOR_DIMS, FACTOR_CURVS, 3)


def _polar_state():
  """Intrinsic radii (2, 4, 2) and concatenated unit directions (2, 4, 6)."""
  torch.manual_seed(0)
  rhos = 6 * torch.rand(2, 4, 2, dtype=torch.float64)
  us = torch.nn.functional.normalize(
    torch.randn(2, 4, 2, 3, dtype=torch.float64), dim=-1).flatten(-2)
  return rhos, us


def _concat_blocks(radii, us):
  """The per-factor vectors radius_i * u_i, concatenated."""
  return (radii.unsqueeze(-1) * us.unflatten(-1, (2, 3))).flatten(-2)


@pytest.mark.parametrize('map_index, num_features', [(0, 16), (1, 2 * (16 + 3))], ids=['hyla', 'sphere'])
def test_feature_maps_agree_across_input_coordinates(map_index, num_features):
  """The same state gives the same features in every coordinate system, read with
  each factor's own dimension and curvature."""
  feature_map = _feature_maps()[map_index]
  rhos, us = _polar_state()
  rs = FACTOR_RADII * torch.tanh(rhos / (2 * FACTOR_RADII))
  ref = feature_map((rhos, us), Coordinate.HYPERBOLIC_POLAR)
  assert feature_map.num_features == num_features
  assert ref.shape == (2, 4, num_features) and ref.dtype == torch.float64
  for state, coord in [(_concat_blocks(rs, us), Coordinate.CARTESIAN),
                       (_concat_blocks(rhos, us), Coordinate.HYPERBOLIC_CARTESIAN),
                       ((rs, us), Coordinate.POINCARE_DISK_POLAR)]:
    assert torch.allclose(feature_map(state, coord), ref, atol=1e-10), coord
  with pytest.raises(ValueError):
    feature_map(_concat_blocks(rs, us), 'not_a_coordinate')


@pytest.mark.parametrize('map_index', [0, 1], ids=['hyla', 'sphere'])
def test_feature_maps_take_the_curvature_per_call(map_index):
  """Curvature enters only through s = rho sqrt(|K|): one map, K = -4 at half the radii
  equals K = -1, and differs from K = -1 at the same radii."""
  feature_map = _feature_maps()[map_index].feature_map
  rhos, us = _polar_state()
  at_k1 = feature_map((rhos, us), Coordinate.HYPERBOLIC_POLAR, FACTOR_DIMS, [-1.0, -1.0])
  at_k4 = feature_map((rhos / 2, us), Coordinate.HYPERBOLIC_POLAR, FACTOR_DIMS, [-4.0, -4.0])
  assert torch.allclose(at_k1, at_k4, atol=1e-10)
  assert not torch.allclose(
    at_k1, feature_map((rhos, us), Coordinate.HYPERBOLIC_POLAR, FACTOR_DIMS, [-4.0, -4.0]), atol=1e-3)


def test_feature_maps_reject_factors_they_cannot_read():
  from random_features.hyla import HyLaFeatures
  from random_features.hyperbolic_spherical_feat import SphericalFeature
  z = torch.zeros(1, 1, 12, dtype=torch.float64)
  cart = Coordinate.CARTESIAN
  with pytest.raises(ValueError):                         # 30 is not a multiple of 4 factors
    HyLaFeatures(30, 4, 1.0, 0)(z, cart, [3] * 4, [-1.0] * 4)
  with pytest.raises(ValueError):                         # no shared factor dimension
    HyLaFeatures(32, 2, 1.0, 0)(z, cart, [4, 8], [-1.0, -1.0])
  with pytest.raises(ValueError):                         # (H^4)^3: no S^2 harmonics
    SphericalFeature(3, 2, [1.0])(z, cart, [4] * 3, [-1.0] * 3)
  with pytest.raises(ValueError):                         # built for 3 factors, so num_features would lie
    SphericalFeature(3, 2, [1.0])(z, cart, [3] * 4, [-1.0] * 4)
  with pytest.raises(ValueError):                         # built for 4 factors
    HyLaFeatures(24, 4, 1.0, 0)(z, cart, [4] * 3, [-1.0] * 3)
  hyla = HyLaFeatures(24, 4, 1.0, 0)
  hyla(z, cart, [3] * 4, [-1.0] * 4)
  with pytest.raises(ValueError):                         # the first forward fixed the layout
    hyla(torch.zeros(1, 1, 8, dtype=torch.float64), cart, [2] * 4, [-1.0] * 4)


def test_hyla_checkpoint_restores_the_draw_before_the_first_forward():
  """A map that has not drawn yet takes the checkpoint's draw, not one from its own seed,
  and moves it to the state's device (Lightning hands it a CPU checkpoint)."""
  from random_features.hyla import HyLaFeatures
  saved = _hyla(FACTOR_DIMS, FACTOR_CURVS, 16, seed=1)
  restored = HyLaFeatures(16, len(FACTOR_DIMS), 1.0, seed=2)
  assert len(restored.state_dict()) == 0
  restored.load_state_dict(saved.state_dict(), strict=True)
  rhos, us = _polar_state()
  z = _concat_blocks(FACTOR_RADII * torch.tanh(rhos / (2 * FACTOR_RADII)), us)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  out = restored(z.to(device), Coordinate.CARTESIAN, FACTOR_DIMS, FACTOR_CURVS)
  assert torch.allclose(out.cpu(), saved(z), atol=1e-12)     # up to CPU / GPU rounding
  fresh = _hyla(FACTOR_DIMS, FACTOR_CURVS, 16, seed=2)
  assert not torch.allclose(out.cpu(), fresh(z))


@pytest.mark.parametrize('map_index', [0, 1], ids=['hyla', 'sphere'])
def test_feature_maps_read_float32_states_in_float64(map_index):
  feature_map = _feature_maps()[map_index]
  rhos, us = _polar_state()
  rs = FACTOR_RADII * torch.tanh(rhos / (2 * FACTOR_RADII))
  for state, coord in [(_concat_blocks(rs, us).float(), Coordinate.CARTESIAN),
                       (_concat_blocks(rhos, us).float(), Coordinate.HYPERBOLIC_CARTESIAN),
                       ((rs.float(), us.float()), Coordinate.POINCARE_DISK_POLAR),
                       ((rhos.float(), us.float()), Coordinate.HYPERBOLIC_POLAR)]:
    assert feature_map(state, coord).dtype == torch.float64, coord


@pytest.mark.parametrize('map_index', [0, 1], ids=['hyla', 'sphere'])
def test_poincare_polar_state_on_the_boundary_is_finite(map_index):
  """r = R exactly is capped like the Cartesian readback, instead of giving rho = inf."""
  feature_map = _feature_maps()[map_index]
  _, us = _polar_state()
  rs = FACTOR_RADII.expand(2, 4, 2)
  assert feature_map((rs, us), Coordinate.POINCARE_DISK_POLAR).isfinite().all()


def test_hyla_first_forward_under_inference_mode_keeps_usable_buffers():
  """The draw outlives the call that made it, so sampling first must not break a later gradient."""
  from random_features.hyla import HyLaFeatures
  dims, curvs = [3, 3], [-1.0, -0.5]
  z = torch.full((1, 1, 6), 0.1, dtype=torch.float64)
  hyla = HyLaFeatures(16, 2, 1.0, 0)
  with torch.inference_mode():
    hyla(z, Coordinate.CARTESIAN, dims, curvs)
  assert not any(getattr(hyla, name).is_inference() for name in hyla.FEATURE_BUFFERS)
  state = z.clone().requires_grad_(True)
  hyla(state, Coordinate.CARTESIAN, dims, curvs).square().sum().backward()
  assert state.grad.isfinite().all()


def test_building_a_feature_map_leaves_the_global_rng_alone():
  """HyLa draws from its own generator, so the trunk's init stream is unchanged."""
  torch.manual_seed(7)
  expected = torch.rand(3)
  torch.manual_seed(7)
  _feature_maps()
  assert torch.equal(torch.rand(3), expected)
