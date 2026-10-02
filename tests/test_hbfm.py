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
  assert torch.allclose(log_p, logits.double().log_softmax(-1))


def test_unknown_forward_type_raises_at_readout(monkeypatch):
  model, _ = _build_model(monkeypatch, ['algo.forward_type=bogus'])
  with pytest.raises(ValueError):
    model._process_model_output(torch.zeros(2, 8, model.vocab_size), None, None, None)


# ---------------------------------------------------------------------------
# End to end on the real model
# ---------------------------------------------------------------------------

@needs_gpu
def test_nll_end_to_end_is_finite_and_differentiable(monkeypatch):
  model, _ = _build_model(monkeypatch)
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
  model, _ = _build_model(monkeypatch)
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
  ['model.hyla_dim=30'],      # not a multiple of the 4 factors
])
def test_validate_configuration_rejects(monkeypatch, overrides):
  with pytest.raises(ValueError):
    _build_model(monkeypatch, overrides)


# ---------------------------------------------------------------------------
# HyLa: random Laplacian features of the state (models.hyperbolic_dit)
# ---------------------------------------------------------------------------

def _hyla(dims, curvs, num_features, scale=1.0, seed=0, radius_cap=None):
  from models.hyperbolic_dit import HyLaFeatures
  return HyLaFeatures(dims, curvs, num_features, scale, seed, radius_cap)


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


@needs_gpu
def test_hyla_backbone_end_to_end(monkeypatch):
  model, _ = _build_model(monkeypatch, ['model.hyla_dim=32', 'model.hyla_scale=0.5'])
  hyla = model.backbone.hyla
  assert hyla is not None and hyla.omegas.shape == (4, 8, 3)
  assert model.backbone.in_proj.in_features == 32
  keys = set(model.state_dict())                          # checkpointed draws
  assert {'backbone.hyla.omegas', 'backbone.hyla.lambdas', 'backbone.hyla.biases'} <= keys
  assert 'backbone.hyla.kappas' not in keys               # derived from the config
  B, L = 2, 8
  x = _tokens(model, B, L)
  # The feature map is differentiable in the state, so the embedding keeps
  # its gradient path through the bridge once the zero-init readout moves.
  xt, _ = model.q_xt(x, _alpha_at_heat_time(model, B, 0.3), use_pure_noise=False)
  xt = xt.detach().requires_grad_(True)
  hyla(xt).square().sum().backward()
  assert torch.isfinite(xt.grad).all() and xt.grad.abs().sum() > 0
  loss, t = model.nll(x, None, None)
  assert loss.shape == (B, L) and torch.isfinite(loss).all()
  loss.sum().backward()
  for p in (model.backbone.sphere_embed.weight, model.backbone.in_proj.weight):
    assert p.grad is not None and torch.isfinite(p.grad).all()
  # The sampler starts at the origin, whose features are cos(b) / sqrt(D).
  origin = torch.zeros(B, L, 12, dtype=torch.float64, device=model.device)
  assert torch.allclose(hyla(origin)[0, 0], hyla.biases.cos().flatten() / math.sqrt(32))
  sigma = model._sigma_from_alphat(_alpha_at_heat_time(model, B, 0.3))
  assert torch.isfinite(model.forward(xt=origin, sigma=sigma)).all()


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


@needs_gpu
def test_hyla_concat_state_backbone(monkeypatch):
  model, _ = _build_model(monkeypatch, [
    'model.hyla_dim=32', 'model.hyla_concat_state=true', 'model.hyla_radius_cap=5.0'])
  assert model.backbone.in_proj.in_features == 12 + 32
  assert model.backbone.hyla.radius_cap == 5.0
  loss, _ = model.nll(_tokens(model, 2, 8), None, None)
  assert torch.isfinite(loss).all()
