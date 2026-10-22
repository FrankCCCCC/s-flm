"""Adaptive per-position curvature of HBFM (`algo.curvature_bound` > 1).

Independent checks of experiments/hbfm_rlf_ada_curv/math.md:

  * the closed-form H^3 radial sampler the per-position clocks use is the
    heat-kernel marginal, with and without curvature, and is differentiable in
    the heat time (section 6);
  * `curvature_scale` is the base curvature at init and stays inside
    [1 / bound, bound] (section 4), and the trunk is still trained by the loss
    after the no-grad curvature pass;
  * `q_xt` draws position i at the base bridge's heat time lambda_i * t and
    returns the weight lambda_i / q(t) (sections 1-2);
  * the MEAN of the weighted loss does not depend on the curvature while its
    VARIANCE does (sections 3-4), and the adaptive path at lambda = 1 is the
    fixed-curvature model;
  * the gradient the curvature receives is the variance gradient: the pathwise
    estimate through the hook equals the score-form E[loss^2 (1 - r t)]
    (section 6);
  * the sampler runs position i on the clock lambda_i * t: the SDE reproduces
    the exact bridge at heat time lambda_i * t and, with the Bayes oracle,
    still decodes the prior at every position (section 7).

The loss tests use the real model at init with the horosphere readout: its
zero-init residual makes the posterior the context-free Bayes posterior, for
which the invariance is exact. They need a GPU (the DiT's flash_attn kernels).
"""
import math

import numpy as np
import pytest
import torch
from scipy import stats

import samplers
import trainer_base
from numeric.geo_bridge import Coordinate, GeoUtils, HyperbolicHeatKernel
from test_hbfm import DEVICE, _alpha_at_heat_time, _build_model, _tokens, needs_gpu
from test_hbfm_sampler import _OracleModel, _unit

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# The closed-form H^3 radial sampler
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('t', [0.05, 1.0, 8.0])
def test_exact_d3_radial_sampler_is_the_heat_kernel_marginal(t):
  torch.manual_seed(1)
  ts = torch.full((1,), t, dtype=torch.float64)
  rho = HyperbolicHeatKernel._sample_radial_exact_d3(ts, 20000)[0]
  cdf = lambda r: HyperbolicHeatKernel._radial_cdf_exact_d3(  # noqa: E731
    torch.as_tensor(r), ts).numpy()
  assert stats.kstest(rho.numpy(), cdf).pvalue > 1e-3


def test_exact_d3_radial_sampler_matches_the_grid_sampler_under_curvature():
  torch.manual_seed(2)
  ts = torch.full((1,), 0.7, dtype=torch.float64)
  exact = HyperbolicHeatKernel.sample_radial(ts, 3, 20000, -0.5, exact_d3=True)[0]
  grid = HyperbolicHeatKernel.sample_radial(ts, 3, 20000, -0.5)[0]
  assert stats.ks_2samp(exact.numpy(), grid.numpy()).pvalue > 1e-3


def test_exact_d3_radial_sampler_is_differentiable_in_the_heat_time():
  # E[rho^2] = E||sqrt(t) g + t e_1||^2 = 3 t + t^2.
  torch.manual_seed(3)
  ts = torch.full((1,), 0.7, dtype=torch.float64, requires_grad=True)
  rho = HyperbolicHeatKernel._sample_radial_exact_d3(ts, 400000)
  (grad,) = torch.autograd.grad(rho.square().mean(), ts)
  assert abs(grad.item() - (3 + 2 * 0.7)) < 0.05


# ---------------------------------------------------------------------------
# The curvature channel and the per-position clock
# ---------------------------------------------------------------------------

ADA = ['algo.curvature_bound=2.0']


def _fix_curvature(monkeypatch, model, curvature):
  """Make `curvature_scale` return the row pattern `curvature` ([L] or scalar)."""
  monkeypatch.setattr(
    model, 'curvature_scale',
    lambda xt: torch.as_tensor(curvature, dtype=torch.float64, device=xt.device)
    .expand(xt.shape[0], xt.shape[1]))


@needs_gpu
def test_curvature_scale_is_the_base_curvature_at_init_and_bounded(monkeypatch):
  model, _ = _build_model(monkeypatch, ADA)
  x = _tokens(model, B=3)
  origin = torch.zeros(3, 8, model.backbone.embed_dim, dtype=torch.float64, device=DEVICE)
  curvature = model.curvature_scale(origin)
  assert curvature.shape == (3, 8)
  assert torch.equal(curvature, torch.ones_like(curvature))   # zero-init channel
  # Any raw output stays inside [1 / bound, bound].
  torch.nn.init.normal_(model.backbone.curvature_layer.linear.weight, std=100.0)
  curvature = model.curvature_scale(model._clean_state(x))
  assert (curvature >= 0.5).all() and (curvature <= 2.0).all()
  assert curvature.min() < 0.6 and curvature.max() > 1.9


@needs_gpu
def test_curvature_is_read_without_dropout(monkeypatch):
  """Training reads the clock the sampler reads: in train mode the curvature's trunk pass runs
  without dropout, so it is deterministic and equal to the eval-mode read."""
  model, cfg = _build_model(monkeypatch, ADA)
  assert cfg.model.dropout > 0
  torch.nn.init.normal_(model.backbone.curvature_layer.linear.weight, std=10.0)
  z = model._clean_state(_tokens(model, B=3))
  model.train()
  first, second = model.curvature_scale(z), model.curvature_scale(z)
  assert model.backbone.training                                  # the mode is restored
  model.eval()
  assert torch.equal(first, second) and torch.equal(first, model.curvature_scale(z))


def test_q_xt_draws_every_position_at_its_own_heat_time(monkeypatch):
  model, _ = _build_model(monkeypatch, ADA)
  torch.manual_seed(4)
  B, t0 = 3000, 0.4
  x = _tokens(model, B=B)
  pattern = torch.tensor([0.5] * 4 + [2.0] * 4, dtype=torch.float64, device=DEVICE)
  alpha = _alpha_at_heat_time(model, B, t0)
  xt, weight = model.q_xt(x, alpha, False, curvature=pattern.expand(B, 8))
  _, base_weight = model.q_xt(x, alpha, False)
  assert xt.shape == (B, 8, model.backbone.embed_dim) and xt.dtype == torch.float64
  assert torch.allclose(weight, base_weight * pattern)        # lambda_i / q(t)
  polar = lambda z: GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(  # noqa: E731
    z.detach(), model.prod_factor_dim, model.prod_factor_gaussian_curvature)
  rho, theta = polar(xt)
  for positions, scale in [(slice(0, 4), 0.5), (slice(4, 8), 2.0)]:
    # The fixed-curvature bridge at heat time lambda * t0.
    ref, _ = model.q_xt(x, _alpha_at_heat_time(model, B, scale * t0), False)
    rho_ref, theta_ref = polar(ref)
    e = GeoUtils._polar_direction(
      model.word_embedding.detach().double()[x][..., :3])     # factor 0's target
    cos = (theta[..., :3] * e).sum(-1)
    cos_ref = (theta_ref[..., :3] * e).sum(-1)
    a = rho[:, positions, 0].flatten().cpu().numpy()
    b = rho_ref[:, positions, 0].flatten().cpu().numpy()
    assert stats.ks_2samp(a, b).pvalue > 1e-3, (scale, a.mean(), b.mean())
    assert stats.ks_2samp(cos[:, positions].flatten().cpu().numpy(),
                          cos_ref[:, positions].flatten().cpu().numpy()).pvalue > 1e-3


@pytest.mark.parametrize('bound', [0.5, 0.0])
def test_validate_configuration_rejects_a_bound_below_one(monkeypatch, bound):
  with pytest.raises(ValueError):
    _build_model(monkeypatch, [f'algo.curvature_bound={bound}'])


# ---------------------------------------------------------------------------
# The loss: mean invariant, variance not
# ---------------------------------------------------------------------------

LOSS = ['algo.forward_type=horosphere', 'algo.readout_precision=float64']


def _sequence_losses(model, x, repeats):
  """Per-sequence token-mean loss, `repeats` batches of fresh (t, noise)."""
  with torch.no_grad():
    return torch.cat([
      model.nll(x, None, trainer_base.TrainingContext())[0].mean(-1)
      for _ in range(repeats)])


@needs_gpu
def test_loss_mean_is_invariant_to_the_curvature_and_its_variance_is_not(monkeypatch):
  model, _ = _build_model(monkeypatch, LOSS + ['algo.curvature_bound=4.0'])
  base, _ = _build_model(monkeypatch, LOSS)                   # fixed curvature
  torch.manual_seed(5)
  x = _tokens(model, B=4096)
  ref = _sequence_losses(base, x, repeats=8)
  stat = {}
  for name, pattern in [('1', 1.0), ('1/2', 0.5), ('2', 2.0), ('4', 4.0),
                        ('mixed', [0.25, 0.5, 1.0, 2.0, 4.0, 0.5, 2.0, 1.0])]:
    _fix_curvature(monkeypatch, model, pattern)
    losses = _sequence_losses(model, x, repeats=8)
    stat[name] = (losses.mean().item(), losses.var().item())
    # Mean: equal to the fixed-curvature model's within Monte-Carlo error.
    z = (losses.mean() - ref.mean()) / (
      losses.var() / len(losses) + ref.var() / len(ref)).sqrt()
    assert abs(z.item()) < 4.0, (name, stat[name], ref.mean().item(), z.item())
  # Variance: lambda = 1 through the adaptive path is the fixed-curvature model
  # ...
  assert abs(stat['1'][1] / ref.var().item() - 1) < 0.15, (stat, ref.var().item())
  # ... and it grows with the curvature (this proposal sits on the sharp side
  # of the optimum, where the second moment is linear in lambda).
  assert stat['1/2'][1] < 0.75 * stat['1'][1] < 0.75 ** 2 * stat['2'][1] \
    < 0.75 ** 3 * stat['4'][1], stat


@needs_gpu
def test_curvature_gradient_is_the_variance_gradient(monkeypatch):
  """Global lambda under the exp proposal: the pathwise gradient the hook
  delivers, d E[loss_b^2] / d log lambda, equals E[loss_b^2 (1 - r t)]."""
  model, cfg = _build_model(monkeypatch, LOSS + ['algo.curvature_bound=4.0'])
  torch.manual_seed(6)
  B, rate = 4096, cfg.algo.time_exp_rate
  x = _tokens(model, B=B)
  log_scale = torch.zeros((), dtype=torch.float64, device=DEVICE, requires_grad=True)
  monkeypatch.setattr(model, 'curvature_scale',
                      lambda xt: log_scale.exp().expand(xt.shape[0], xt.shape[1]))
  pathwise, score = [], []
  hook_scale = 1 / cfg.model.hidden_size ** 0.5               # the hook's own factor
  for _ in range(16):
    loss, t = model.nll(x, None, trainer_base.TrainingContext())
    seq_loss = loss.mean(-1)
    (grad,) = torch.autograd.grad(seq_loss.mean(), log_scale)
    heat = -torch.log(1 - model.noise(t)[1].double()) / rate
    pathwise.append(grad.item() / hook_scale)
    score.append((seq_loss.detach().square() * (1 - rate * heat)).mean().item())
  pathwise, score = np.array(pathwise), np.array(score)
  err = math.hypot(pathwise.std(ddof=1), score.std(ddof=1)) / math.sqrt(len(score))
  assert abs(pathwise.mean() - score.mean()) < 4 * err, (pathwise.mean(), score.mean(), err)
  assert pathwise.mean() > 10 * err                           # a non-trivial gradient


@needs_gpu
def test_nll_trains_the_curvature_channel(monkeypatch):
  model, _ = _build_model(monkeypatch, ADA + ['algo.forward_type=horosphere'])
  torch.manual_seed(7)
  x = _tokens(model, B=4)
  valid = torch.ones_like(x)
  valid[:, :3] = 0                                            # sudoku-style prompt
  # A non-zero readout, so that the loss reaches the trunk.
  torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.02)
  with torch.autocast('cuda', dtype=torch.bfloat16):          # as in training
    loss, _ = model.nll(x, None, trainer_base.TrainingContext(), valid_tokens=valid)
  assert loss.shape == x.shape and torch.isfinite(loss).all()
  (loss * valid).sum().backward()
  grad = model.backbone.curvature_layer.linear.weight.grad
  assert torch.isfinite(grad).all() and grad.abs().max() > 0  # the curvature readout
  # The trunk is trained by the loss forward that follows the no-grad
  # curvature pass (the autocast weight cache is cleared in between).
  trunk = model.backbone.blocks[0].adaLN_modulation.bias.grad
  assert torch.isfinite(trunk).all() and trunk.abs().max() > 0
  with torch.no_grad():                                       # validation path: no hook
    loss, _ = model.nll(x, None, trainer_base.TrainingContext(), valid_tokens=valid)
  assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# The sampler: position i on the clock lambda_i * t
# ---------------------------------------------------------------------------

def test_sde_with_per_position_clocks_reproduces_the_exact_bridge():
  torch.manual_seed(8)
  n, dims, curvs, t_end, n_steps = 4096, [3], [-0.5], 1.0, 1500
  scales = [0.5, 2.0]
  endpoint = _unit(3)
  z = torch.zeros(n, 2, 3, dtype=torch.float64)
  dt = (t_end / n_steps) * torch.tensor(scales, dtype=torch.float64).view(1, 2, 1)
  for _ in range(n_steps):
    z = samplers.HBFMSampler.euler_maruyama_step(
      z, endpoint.expand(n, 2, -1), None, dt, dims, curvs)
  rho, theta = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(z, dims, curvs)
  for i, scale in enumerate(scales):
    rho_ex, theta_ex = HyperbolicHeatKernel.poincare_bridge_prod(
      torch.full((n,), scale * t_end, dtype=torch.float64),
      torch.zeros(n, 1, dtype=torch.long), endpoint[None],
      output_coord=Coordinate.HYPERBOLIC_POLAR,
      prod_factor_dim=dims, prod_factor_gaussian_curvature=curvs)
    assert stats.ks_2samp(rho[:, i, 0].numpy(), rho_ex[:, 0, 0].numpy()).pvalue > 1e-3
    assert stats.ks_2samp((theta[:, i] @ endpoint).numpy(),
                          (theta_ex[:, 0] @ endpoint).numpy()).pvalue > 1e-3


def test_oracle_bridge_decodes_the_prior_under_per_position_curvature():
  torch.manual_seed(9)
  V, dims, curvs = 6, [3, 3], [-1.0, -2.0]
  prior = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.07, 0.03], dtype=torch.float64)
  model = _OracleModel(V, dims, curvs, prior.log(), steps=4800, length=2)
  model.curvature_bound = 2.0
  model.curvature_scale = lambda xt: torch.tensor(
    [0.5, 2.0], dtype=torch.float64).expand(xt.shape[0], 2)
  sampler = samplers.HBFMSampler(noise_removal='greedy', velocity='exact', use_float64=True,
                                 temperature=1.0, p_nucleus=1.0, top_k=-1,
                                 top_k_velocity=-1, t_max=24.0)
  n = 3000
  tokens, _ = samplers.run_sampler(sampler, model, n, num_steps=4800)
  assert tokens.shape == (n, 2)
  for i in range(2):   # the slow and the fast position both end on the prior
    counts = torch.bincount(tokens[:, i], minlength=V).numpy()
    assert stats.chisquare(counts, prior.numpy() * n).pvalue > 1e-3, (i, counts)


@needs_gpu
def test_real_model_sampler_steps_on_the_predicted_curvature(monkeypatch):
  model, _ = _build_model(monkeypatch, ADA + ['sampler.steps=5', 'sampler.t_max=1.0'])
  B, P = 3, 2
  prefix = torch.randint(0, model.vocab_size, (B, P), device=DEVICE)
  lengths = torch.full((B,), P, device=DEVICE)
  tokens, meta = model.generate_samples(B, num_steps=5, prefix_tokens=prefix,
                                        prefix_lengths=lengths)
  assert tokens.shape == (B, model.num_tokens) and torch.equal(tokens[:, :P], prefix)
  assert meta['nfe'] == 5
  # A position whose curvature scale is 0 does not move; the others do.
  _fix_curvature(monkeypatch, model, [1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 1.0])
  with torch.no_grad():
    state = model.sampler.init_state(model, B, num_steps=5, prefix_tokens=prefix,
                                     prefix_lengths=lengths)
    assert state.curvature.shape == (B, model.num_tokens)
    state = model.sampler.step(model, state)
  moved = state.xt.abs().sum(-1) > 0
  assert not moved[:, 2].any() and moved[:, 3:].all()
