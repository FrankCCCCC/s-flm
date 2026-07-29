"""Autonomous (log-time) flow + VDM SNR-weighted CE.

Autonomous clock (slides/jul09_2026): tau = -log((T-t)/T) maps the finite
bridge horizon onto [0, inf), so the drift becomes v(X) = y - X. Under
`noise=autonomous` the noise fraction is b_t = 1 - alpha_t = exp(-tau) with
tau = tau_max * (1 - t), which makes (a) tau uniform in t and (b) the EFLM
Euler step (b_t - b_s)/b_t constant along the schedule -- the discrete
signature of a time-invariant velocity field.

`algo.snr_weight` is the VDM (Kingma et al., 2021) Eq. 16 weight -SNR'(t)/2
for that interpolant, checked here against autograd through SNR(t).
"""
import math

import pytest
import torch

from algo import snr_weight
from noise_schedules import Autonomous, LogLinear
from samplers import sfm_step_size

EPS = 1e-3
TAU_MAXS = [1.0, 3.0, 6.9]

# float64 throughout: alpha_t = 1 - (1-eps) exp(-tau) cancels against 1, which
# costs ~1e-4 relative precision on the noise fraction in float32 (harmless for
# training -- alpha only feeds the -log(alpha) time conditioning -- but it
# swamps the tolerances these algebraic identities are checked at).
def grid(*args, **kwargs):
  return torch.linspace(*args, **kwargs, dtype=torch.float64)


def noise_fraction(sched, t):
  return 1 - sched.alpha_t(t)


@pytest.mark.parametrize('tau_max', TAU_MAXS)
def test_endpoints_match_log_linear_floor(tau_max):
  sched = Autonomous(eps=EPS, tau_max=tau_max)
  # t = 1: pure noise, alpha floored at eps so -log(alpha) stays finite.
  assert sched.alpha_t(
    torch.tensor(1.0, dtype=torch.float64)).item() == pytest.approx(EPS)
  # t = 0: the flow is truncated at tau_max, never reaching the target.
  assert sched.alpha_t(
    torch.tensor(0.0, dtype=torch.float64)).item() == pytest.approx(
      1 - (1 - EPS) * math.exp(-tau_max))


@pytest.mark.parametrize('tau_max', TAU_MAXS)
def test_tau_is_uniform_in_t(tau_max):
  """tau = -log(b_t) is affine in t -> a uniform t grid is uniform in tau."""
  sched = Autonomous(eps=EPS, tau_max=tau_max)
  t = grid(0.0, 1.0, 51)
  tau = -torch.log(noise_fraction(sched, t))
  d_tau = (tau[1:] - tau[:-1]).abs()  # tau runs backwards: t=0 is clean
  assert torch.allclose(d_tau, d_tau.mean().expand_as(d_tau), atol=1e-6)
  assert d_tau.mean().item() == pytest.approx(tau_max / 50, rel=1e-5)


@pytest.mark.parametrize('tau_max', TAU_MAXS)
def test_euler_step_size_is_constant(tau_max):
  """The autonomy signature: every sampler step advances the same d_tau,
  so the EFLM/SFM step size (b_t - b_s)/b_t is the same at every step."""
  sched = Autonomous(eps=EPS, tau_max=tau_max)
  t = grid(1.0, 0.0, 33)  # invert_time_convention=false schedule
  steps = torch.stack([
    sfm_step_size(sched.alpha_t(t[i]), sched.alpha_t(t[i + 1]),
                  invert_time_convention=False, eps=1e-6)
    for i in range(len(t) - 1)])
  assert torch.allclose(steps, steps.mean().expand_as(steps), atol=1e-6)
  assert steps.mean().item() == pytest.approx(
    1 - math.exp(-tau_max / 32), rel=1e-5)
  # log-linear, in contrast, has a step size that blows up toward the target.
  ll = LogLinear(eps=EPS)
  ll_steps = torch.stack([
    sfm_step_size(ll.alpha_t(t[i]), ll.alpha_t(t[i + 1]),
                  invert_time_convention=False, eps=1e-6)
    for i in range(len(t) - 1)])
  assert ll_steps[-1] > 10 * ll_steps[0]


@pytest.mark.parametrize('tau_max', TAU_MAXS)
def test_alpha_prime_matches_autograd(tau_max):
  sched = Autonomous(eps=EPS, tau_max=tau_max)
  t = grid(0.0, 1.0, 11).requires_grad_(True)
  grad, = torch.autograd.grad(sched.alpha_t(t).sum(), t)
  assert torch.allclose(grad, sched.alpha_prime_t(t.detach()), atol=1e-6)


@pytest.mark.parametrize('sched', [Autonomous(eps=EPS, tau_max=3.0),
                                   LogLinear(eps=EPS)])
@pytest.mark.parametrize('invert', [False, True])
def test_snr_weight_matches_minus_half_dsnr_dt(sched, invert):
  """|SNR'(t)|/2 with SNR(t) = (1-b_t)^2 / b_t^2, b_t the noise fraction.

  VDM's -SNR'(t)/2 assumes t = 0 is clean (invert_time_convention=false, what
  the EFLM scripts run); under the flipped convention time runs the other way
  and the (always positive) bound weight is the magnitude.
  """
  t = grid(0.05, 0.95, 19).requires_grad_(True)
  alpha = sched.alpha_t(t)
  b_t = alpha if invert else 1 - alpha
  snr = ((1 - b_t) / b_t) ** 2
  dsnr, = torch.autograd.grad(snr.sum(), t)

  w = snr_weight(sched.alpha_t(t.detach()).unsqueeze(-1),
                 sched.alpha_prime_t(t.detach()).unsqueeze(-1),
                 invert_time_convention=invert, eps=1e-6).squeeze(-1)
  assert torch.allclose(w, dsnr.abs() / 2, rtol=1e-6)
  assert (w > 0).all()
  if not invert:
    assert (dsnr < 0).all()  # SNR decays as t -> 1, so -SNR'/2 > 0
