"""Correctness tests for the EFLM SDE sampler (EFLMSampler._sde_update).

Ground truth: the 1-D linear bridge x_b = (1-b)*x1 + b*eps with
x1 in {-1, +1} (uniform) and eps ~ N(0, 1), indexed by the noise fraction
b (b=1 pure noise, b=0 data; derivation.md sec. 11). Everything is
closed-form:

  p_b(x)     = 0.5 N(1-b, b^2) + 0.5 N(-(1-b), b^2)
  E[x1 | x]  = tanh((1-b) x / b^2)
  vel        = E[x1|x] - x           (the raw model direction p@E - x)
  E[X^2]     = (1-b)^2 + b^2
  E[|X|]     = same as E|Z|, Z ~ N(1-b, b^2)  (mixture symmetry)

_sde_update takes the sampler-native quantities -- un-normalized vel,
normalized dt = d/b, norm_factor = b, timestep = 1-b -- de-normalizes them
internally, and must implement the sec.-8 Euler--Maruyama step exactly.
The SDE must reproduce the bridge marginals for any eta; eta=0 must reduce
to the ODE step X + dt * vel.
"""
import types

import pytest
import torch

from conftest import REPO_ROOT  # noqa: F401  (sys.path setup)
import noise_schedules
import samplers


def make_sampler(**overrides):
  kwargs = dict(
    noise_removal='ancestral', velocity='exact', use_float64=True,
    lerp_float64=True, eps=1e-6, temperature=1.0, p_nucleus=1.0,
    top_k=-1, top_k_velocity=-1, invert_time_convention=True,
    prior_cov=1.0)
  kwargs.update(overrides)
  return samplers.EFLMSampler(**kwargs)


def exact_vel(x, b):
  # vel = x_hat - x with the exact posterior mean x_hat = E[x1 | x_b = x].
  return torch.tanh((1.0 - b) * x / b ** 2) - x


def exact_moments(b):
  # E[X^2] and E[|X|] of p_b; |X| ~ |N(1-b, b^2)| by mixture symmetry.
  m, s = 1.0 - b, b
  ex2 = m ** 2 + s ** 2
  z = torch.tensor(m / s)
  phi = 0.5 * (1.0 + torch.erf(z / 2 ** 0.5))
  eabs = (s * (2.0 / torch.pi) ** 0.5 * torch.exp(-z ** 2 / 2)
          + m * (2.0 * phi - 1.0))
  return ex2, float(eabs)


def test_gscheduler():
  t = torch.tensor(0.25)
  gsched = noise_schedules.get_gscheduler(noise_schedules.GT_Method.LINEAR)
  g_prime, g = gsched(t)
  assert g == pytest.approx(0.75)
  assert g_prime == pytest.approx(-1.0)
  with pytest.raises(NotImplementedError):
    noise_schedules.get_gscheduler(noise_schedules.GT_Method.LOG).g_t(t)
  with pytest.raises(ValueError):
    noise_schedules.get_gscheduler('bogus')


def test_eflm_timestep_and_norm_factor():
  alpha = torch.tensor(0.7)
  # invert: alpha is the noise fraction b -> timestep 1-b, norm b.
  assert samplers.eflm_timestep(alpha, True, 1e-6) == pytest.approx(0.3)
  assert samplers.eflm_norm_factor(alpha, True, 1e-6) == pytest.approx(0.7)
  # MDLM: alpha is the signal fraction -> timestep alpha, norm 1-alpha.
  assert samplers.eflm_timestep(alpha, False, 1e-6) == pytest.approx(0.7)
  assert samplers.eflm_norm_factor(alpha, False, 1e-6) == pytest.approx(0.3)


def test_sde_update_eta_zero_is_euler_ode():
  torch.manual_seed(0)
  sampler = make_sampler()
  x = torch.randn(64, 8, 4)
  v = torch.randn_like(x)  # un-normalized vel
  b, dt = torch.tensor(0.7), torch.tensor(0.01)  # dt normalized: d / b
  out = sampler._sde_update(timestep=1.0 - b, x=x, velocity=v,
                            dt=dt, norm_factor=b, eta=0.0)
  assert torch.allclose(out, x + dt * v)


def test_sde_update_matches_flow_time_form():
  """derivation.md sec. 8: after de-normalizing by norm_factor = 1-t, the
  step must equal the flow-time Euler--Maruyama form evaluated directly."""
  sampler = make_sampler()
  torch.manual_seed(1)
  x = torch.randn(256, 4, dtype=torch.float64)
  v_fm = torch.randn_like(x)  # marginal FM velocity v(X, t)
  t, dtau, eta = 0.4, 0.01, 1.3
  b = torch.tensor(1.0 - t, dtype=torch.float64)

  # sec. 8 reference: X + [(1 + coef*t) v - coef*X] dt + sqrt(eta) g sqrt(dt) xi
  g = 1.0 - t
  coef = eta * g ** 2 / (2 * (1 - t))
  torch.manual_seed(2)
  ref = (x + ((1 + coef * t) * v_fm - coef * x) * dtau
         + eta ** 0.5 * g * dtau ** 0.5 * torch.randn_like(x))

  torch.manual_seed(2)
  out = sampler._sde_update(
    timestep=torch.tensor(t, dtype=torch.float64), x=x,
    velocity=b * v_fm,  # un-normalized vel = (1-t) * v_FM
    dt=torch.tensor(dtau, dtype=torch.float64) / b,  # normalized dt
    norm_factor=b, eta=eta)
  assert torch.allclose(out, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('eta', [0.0, 1.0, 4.0])
def test_sde_preserves_bridge_marginals(eta):
  """Integrate the exact-model direction with _sde_update; the ensemble
  must match the closed-form bridge moments at intermediate and final b."""
  torch.manual_seed(0)
  sampler = make_sampler()
  n_particles, n_steps, b_end = 200_000, 1_000, 0.01
  bs = torch.linspace(1.0 - 1e-4, b_end, n_steps + 1)
  x = torch.randn(n_particles)  # p_{b=1} = N(0, 1)
  for k in range(n_steps):
    b, d = bs[k], bs[k] - bs[k + 1]
    x = sampler._sde_update(timestep=1.0 - b, x=x,
                            velocity=exact_vel(x, b),
                            dt=d / b, norm_factor=b, eta=eta)
    # The midpoint check is load-bearing: near b_end the exact velocity is
    # strongly contracting and injected noise ~ b^2 vanishes, so Wiener- or
    # eta-scaling bugs are only visible at intermediate b.
    if k + 1 in (n_steps // 2, n_steps):
      ex2, eabs = exact_moments(float(bs[k + 1]))
      assert abs(float(x.mean())) < 0.02
      assert abs(float((x ** 2).mean()) - ex2) < 0.02
      assert abs(float(x.abs().mean()) - eabs) < 0.02


class BayesEFLM:
  """Closed-form Bayes-optimal EFLM 'model': 2-token vocab, 1-D embeddings
  e_0 = -1, e_1 = +1. forward() returns the exact bridge posterior
  p(x1 = e_k | x_b), so p @ E is the exact E[x1 | x_b].

  noise(t) = 1 - t works for BOTH conventions: invert (t_schedule eps->1,
  alpha = noise fraction, decreasing) and MDLM (t_schedule 1->eps, alpha =
  signal fraction, increasing). The fake _sigma_from_alphat is the identity
  so forward() can recover the noise fraction from sigma per convention."""

  def __init__(self, num_tokens, num_steps, eta, invert=True):
    self.device = torch.device('cpu')
    self.num_tokens = num_tokens
    self.self_conditioning = False
    self.invert = invert
    self.backbone = types.SimpleNamespace(embed_dim=1)
    self.config = types.SimpleNamespace(
      sampler=types.SimpleNamespace(steps=num_steps, eta=eta))
    self.E = torch.tensor([[-1.0], [1.0]])

  def noise(self, t):
    return None, 1.0 - t

  def _sigma_from_alphat(self, alpha_t):
    return alpha_t

  def _sc_embed_table(self):
    return self.E

  def forward(self, xt, sigma, context):
    alpha = sigma.reshape(())  # identity _sigma_from_alphat
    b = alpha if self.invert else 1.0 - alpha  # noise fraction
    var = (b ** 2).clamp(min=1e-12)
    d2 = ((xt.unsqueeze(-2) - (1.0 - b) * self.E) ** 2).sum(-1)  # [B, L, 2]
    return (-d2 / (2.0 * var)).log_softmax(-1)


@pytest.mark.parametrize('invert', [True, False])
@pytest.mark.parametrize('eta', [0.0, 1.0])
def test_full_sampler_loop(eta, invert):
  torch.manual_seed(0)
  n_samples, n_tokens, n_steps = 256, 32, 200
  model = BayesEFLM(n_tokens, n_steps, eta, invert=invert)
  sampler = make_sampler(invert_time_convention=invert)
  state = sampler.init_state(model, n_samples)
  assert state.eta == eta  # picked up from config.sampler.eta

  penult_xt = None
  while not state.done:
    if state.step_idx == n_steps - 1:
      penult_xt = state.xt.clone()
    state = sampler.step(model, state)

  # Continuous marginals just before decode match the bridge closed form.
  alpha_pen = float(model.noise(state.t_schedule[n_steps - 1])[1])
  b_penult = alpha_pen if invert else 1.0 - alpha_pen
  ex2, eabs = exact_moments(b_penult)
  # 0.04 ~= 3.6 standard errors of the 256*32-token ensemble mean.
  assert abs(float(penult_xt.mean())) < 0.04
  assert abs(float((penult_xt ** 2).mean()) - ex2) < 0.05
  assert abs(float(penult_xt.abs().mean()) - eabs) < 0.05

  # Decoded tokens: ids in {0, 1}, balanced across the ensemble.
  tokens = state.xt
  assert tokens.shape == (n_samples, n_tokens)
  assert tokens.dtype in (torch.int64, torch.int32)
  assert abs(float(tokens.float().mean()) - 0.5) < 0.02
