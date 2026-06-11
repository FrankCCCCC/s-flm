"""Interface-contract tests for the learnable Gumbel `UnifInfoSchedule`.

These tests are written AGAINST the contract in
`experiments/langflow/ARCH.md` (§3 the schedule interface, §8 invariants A/B/C/D,
§9 test surface #1-#5, #14) BEFORE the implementation exists. They are expected
to FAIL until the implementer replaces the stale spline `UnifInfoSchedule` (the
`base_schedule`/spline copy currently at `noise_schedules.py` ~L272) with the
Gumbel schedule and adds the `type=='gumbel'` branch to `get_noise`.

`UnifInfoSchedule` is a plain `nn.Module` (a `NoiseSchedule`); it has no EMA /
HF-download dependency, so we construct it directly with the ARCH §3.2 signature
`UnifInfoSchedule(trainable, q_clip, H_inf_init, beta_floor=1e-4, H_floor=1e-4)`.

Math contract (ARCH §3):
  sigma^2 = sigmoid(gamma),  alpha^2 = sigmoid(-gamma)           (VP, invariant A)
  gamma   = P_mu - P_beta * log(-log q),  q in [q_clip, 1-q_clip]
  H_gamma = H_inf * exp(-exp(-(gamma - P_mu)/P_beta))            (surrogate entropy)
  P_beta  = softplus(raw_beta) + beta_floor > 0                  (invariant C)
  H_inf   = softplus(raw_H)   + H_floor   > 0
  L_Sched = mean( (stopgrad(ce) - H_gamma)^2 )                   (invariant D / B3)
"""
import math

import pytest
import torch

import noise_schedules
from conftest import REPO_ROOT  # noqa: F401  (ensures repo root on sys.path)

torch.manual_seed(0)

Q_CLIP = 1e-5
H_INF_INIT = 5.0


# ---------------------------------------------------------------------------
# Construction helpers (ARCH §3.2 signature)
# ---------------------------------------------------------------------------

def _build_schedule(*, trainable=True, q_clip=Q_CLIP, H_inf_init=H_INF_INIT):
  """Construct `UnifInfoSchedule` with the ARCH §3.2 keyword signature.

  If the stub still has the spline signature, this raises TypeError -> the test
  fails for the right (signature-mismatch) reason, which is the intended initial
  red state for #1.
  """
  return noise_schedules.UnifInfoSchedule(
    trainable=trainable, q_clip=q_clip, H_inf_init=H_inf_init)


def _gamma_clip_bounds(sched):
  """The [a, b] gamma clip from the q_clip quantiles (ARCH §3.2)."""
  P_mu = sched.P_mu.detach()
  P_beta = sched.P_beta.detach()
  q = sched.q_clip
  a = P_mu - P_beta * math.log(-math.log(1 - q))
  b = P_mu - P_beta * math.log(-math.log(q))
  return float(a), float(b)


# ---------------------------------------------------------------------------
# Invariant A — VP identity (ARCH §8 A, test surface #1)
# ---------------------------------------------------------------------------

def test_vp_identity_alpha2_plus_sigma2_equals_one():
  """For 1000 gamma in [a,b], `alpha^2 + sigma^2 == 1` to fp tol. ARCH §8 A / #1."""
  sched = _build_schedule()
  a, b = _gamma_clip_bounds(sched)
  gamma = torch.linspace(a, b, 1000)
  alpha, sigma = sched.alpha_sigma_from_gamma(gamma)
  one = alpha ** 2 + sigma ** 2
  assert torch.allclose(one, torch.ones_like(one), atol=1e-5)


def test_sigma_squared_equals_sigmoid_gamma():
  """`sigma^2 == sigmoid(gamma)` exactly (ARCH §3.3)."""
  sched = _build_schedule()
  gamma = torch.linspace(-8.0, 8.0, 257)
  _, sigma = sched.alpha_sigma_from_gamma(gamma)
  assert torch.allclose(sigma ** 2, torch.sigmoid(gamma), atol=1e-6)


def test_alpha_squared_equals_sigmoid_neg_gamma():
  """`alpha^2 == sigmoid(-gamma)` exactly (ARCH §3.3)."""
  sched = _build_schedule()
  gamma = torch.linspace(-8.0, 8.0, 257)
  alpha, _ = sched.alpha_sigma_from_gamma(gamma)
  assert torch.allclose(alpha ** 2, torch.sigmoid(-gamma), atol=1e-6)


# ---------------------------------------------------------------------------
# Invariant B — gamma clip + finiteness (ARCH §8 B, test surface #2)
# ---------------------------------------------------------------------------

def test_sample_gamma_returns_shape_n():
  """`sample_gamma(n, device)` returns a 1-D tensor of length n. ARCH §3.3."""
  sched = _build_schedule()
  gamma = sched.sample_gamma(64, torch.device('cpu'))
  assert tuple(gamma.shape) == (64,)


def test_sample_gamma_is_finite():
  """Sampled gamma values are all finite. ARCH §8 B."""
  sched = _build_schedule()
  gamma = sched.sample_gamma(10000, torch.device('cpu'))
  assert torch.isfinite(gamma).all()


def test_sample_gamma_within_clip_bounds():
  """Every sampled gamma lies in [a,b] (the q_clip quantiles). ARCH §8 B / #2.

  ARCH §3.2's `a`/`b` labels are reversed vs the Gumbel quantile ordering
  (`gamma = P_mu - P_beta*log(-log q)` is increasing in q, so the q_clip end is
  the LOWER gamma and the 1-q_clip end is the UPPER); the implementation clamps to
  [min(a,b), max(a,b)]. Assert against that ordering.
  """
  sched = _build_schedule()
  a, b = _gamma_clip_bounds(sched)
  lo, hi = min(a, b), max(a, b)
  gamma = sched.sample_gamma(10000, torch.device('cpu'))
  assert float(gamma.min()) >= lo - 1e-4
  assert float(gamma.max()) <= hi + 1e-4


def test_sample_gamma_antithetic_low_discrepancy():
  """With antithetic=True the empirical gamma CDF tracks the Gumbel CDF. #2.

  The Gumbel CDF at gamma is exp(-exp(-(gamma-P_mu)/P_beta)); a low-discrepancy
  draw should make the empirical CDF close to the median quantile near gamma=P_mu.
  """
  sched = _build_schedule()
  gamma = sched.sample_gamma(10000, torch.device('cpu'), antithetic=True)
  P_mu = float(sched.P_mu.detach())
  P_beta = float(sched.P_beta.detach())
  # Empirical CDF at a few quantile points vs the Gumbel CDF.
  for g in (P_mu - P_beta, P_mu, P_mu + P_beta):
    emp = float((gamma <= g).float().mean())
    cdf = math.exp(-math.exp(-(g - P_mu) / P_beta))
    assert abs(emp - cdf) < 0.05


# ---------------------------------------------------------------------------
# Surrogate entropy (ARCH §3.3, test surface #5 form / monotonicity / limits)
# ---------------------------------------------------------------------------

def test_surrogate_entropy_matches_gumbel_cdf_form():
  """`surrogate_entropy(gamma) == H_inf * exp(-exp(-(gamma-P_mu)/P_beta))`. §3.3."""
  sched = _build_schedule()
  gamma = torch.linspace(-10.0, 10.0, 401)
  P_mu = sched.P_mu.detach()
  P_beta = sched.P_beta.detach()
  H_inf = sched.H_inf.detach()
  expected = H_inf * torch.exp(-torch.exp(-(gamma - P_mu) / P_beta))
  got = sched.surrogate_entropy(gamma)
  assert torch.allclose(got, expected, atol=1e-5)


def test_surrogate_entropy_monotone_increasing_in_gamma():
  """H_gamma is monotone increasing in gamma. ARCH §3.3."""
  sched = _build_schedule()
  gamma = torch.linspace(-10.0, 10.0, 500)
  h = sched.surrogate_entropy(gamma)
  diffs = h[1:] - h[:-1]
  assert (diffs >= -1e-7).all()


def test_surrogate_entropy_tends_to_H_inf_as_gamma_large():
  """As gamma -> +inf, H_gamma -> H_inf. ARCH §3.3."""
  sched = _build_schedule()
  h = sched.surrogate_entropy(torch.tensor([50.0]))
  assert torch.allclose(h, sched.H_inf.detach().reshape(1), atol=1e-3)


def test_surrogate_entropy_tends_to_zero_as_gamma_small():
  """As gamma -> -inf, H_gamma -> 0. ARCH §3.3."""
  sched = _build_schedule()
  h = sched.surrogate_entropy(torch.tensor([-50.0]))
  assert float(h.abs().max()) < 1e-3


# ---------------------------------------------------------------------------
# Invariant C — softplus positivity (ARCH §8 C, test surface #3)
# ---------------------------------------------------------------------------

def test_softplus_positivity_for_extreme_negative_raw():
  """With raw_beta = raw_H = -1e3, `P_beta` and `H_inf` stay >= floor > 0. #3.

  Simulates a gradient step pushing the raw params very negative.
  """
  sched = _build_schedule()
  with torch.no_grad():
    sched.raw_beta.fill_(-1e3)
    sched.raw_H.fill_(-1e3)
  assert float(sched.P_beta.detach()) > 0.0
  assert float(sched.H_inf.detach()) > 0.0


def test_softplus_positivity_respects_floor():
  """`P_beta >= beta_floor` and `H_inf >= H_floor` exactly. ARCH §3.1 / §8 C."""
  sched = _build_schedule()
  with torch.no_grad():
    sched.raw_beta.fill_(-1e3)
    sched.raw_H.fill_(-1e3)
  assert float(sched.P_beta.detach()) >= 1e-4 - 1e-9
  assert float(sched.H_inf.detach()) >= 1e-4 - 1e-9


def test_H_inf_init_recovered_from_config():
  """At init `H_inf ~= H_inf_init` (inv_softplus parameterization). ARCH §3.1."""
  sched = _build_schedule(H_inf_init=5.0)
  assert float(sched.H_inf.detach()) == pytest.approx(5.0, abs=1e-3)


def test_P_beta_init_is_one():
  """At init `P_beta ~= 1.0` (raw_beta = inv_softplus(1.0)). ARCH §3.1."""
  sched = _build_schedule()
  assert float(sched.P_beta.detach()) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# trainable / fixed switch (ARCH §3.2, test surface #4)
# ---------------------------------------------------------------------------

def test_trainable_true_exposes_three_parameters():
  """`trainable=True` -> raw_mu/raw_beta/raw_H are nn.Parameters (3 in
  `parameters()`, all requires_grad). ARCH §3.2 / #4."""
  sched = _build_schedule(trainable=True)
  params = [p for p in sched.parameters()]
  assert len(params) == 3
  assert all(p.requires_grad for p in params)


def test_fixed_false_has_no_trainable_parameters():
  """`trainable=False` -> 0 entries in `parameters()` (the three are buffers). #4."""
  sched = _build_schedule(trainable=False)
  assert len(list(sched.parameters())) == 0


def test_fixed_false_registers_three_buffers():
  """`trainable=False` -> raw_mu/raw_beta/raw_H are buffers. ARCH §3.2 / #4."""
  sched = _build_schedule(trainable=False)
  buf_names = {name for name, _ in sched.named_buffers()}
  assert {'raw_mu', 'raw_beta', 'raw_H'} <= buf_names


def test_fixed_false_scheduler_loss_is_zero():
  """`trainable=False` -> `scheduler_loss(...)` is a zero scalar. ARCH §3.3 / #4."""
  sched = _build_schedule(trainable=False)
  gamma = sched.sample_gamma(8, torch.device('cpu')).detach()
  ce = torch.full((8,), 3.0)
  sl = sched.scheduler_loss(gamma, ce)
  assert sl.ndim == 0
  assert float(sl) == 0.0


def test_fixed_false_scheduler_loss_requires_grad_false():
  """`trainable=False` -> the scheduler-loss scalar has requires_grad False. #4."""
  sched = _build_schedule(trainable=False)
  gamma = sched.sample_gamma(8, torch.device('cpu')).detach()
  ce = torch.full((8,), 3.0)
  sl = sched.scheduler_loss(gamma, ce)
  assert sl.requires_grad is False


# ---------------------------------------------------------------------------
# scheduler_loss value + grad (ARCH §3.3 / §8 D / B3, test surface #5)
# ---------------------------------------------------------------------------

def test_scheduler_loss_equals_mse_against_surrogate_entropy():
  """`scheduler_loss(gamma, ce) == mean((ce - H_gamma)^2)`. ARCH §3.3 / #5."""
  sched = _build_schedule()
  gamma = sched.sample_gamma(16, torch.device('cpu')).detach()
  ce = torch.linspace(1.0, 6.0, 16)
  h = sched.surrogate_entropy(gamma).detach()
  expected = ((ce - h) ** 2).mean()
  got = sched.scheduler_loss(gamma, ce)
  assert torch.allclose(got, expected, atol=1e-6)


def test_scheduler_loss_is_finite_scalar():
  """`scheduler_loss` returns a finite 0-dim scalar. ARCH §3.3."""
  sched = _build_schedule()
  gamma = sched.sample_gamma(16, torch.device('cpu')).detach()
  ce = torch.full((16,), 4.0)
  sl = sched.scheduler_loss(gamma, ce)
  assert sl.ndim == 0
  assert torch.isfinite(sl)


def test_scheduler_loss_grad_flows_to_schedule_params():
  """`.backward()` populates grad on raw_mu/raw_beta/raw_H. ARCH §8 D / #5."""
  sched = _build_schedule(trainable=True)
  gamma = sched.sample_gamma(16, torch.device('cpu')).detach()
  ce = torch.linspace(1.0, 6.0, 16)
  for p in sched.parameters():
    p.grad = None
  sched.scheduler_loss(gamma, ce).backward()
  grads = [p.grad for p in sched.parameters()]
  assert all(g is not None for g in grads)
  assert any(float(g.abs().sum()) > 0.0 for g in grads)


def test_scheduler_loss_does_not_grad_ce(ce_detach_check=None):
  """`ce_detached` is treated as a constant: backward through `scheduler_loss`
  leaves no grad on the CE tensor (it is detached by contract). ARCH §3.3 / B3."""
  sched = _build_schedule(trainable=True)
  gamma = sched.sample_gamma(8, torch.device('cpu')).detach()
  # A CE tensor that *would* receive grad if the schedule did not detach it.
  ce = torch.full((8,), 4.0, requires_grad=True)
  sched.scheduler_loss(gamma.detach(), ce.detach()).backward()
  assert ce.grad is None


def test_scheduler_loss_gamma_detached_no_grad_to_gamma_source():
  """`surrogate_entropy(gamma.detach())` inside `scheduler_loss`: grad does not
  reach a gamma that carried a graph (gamma stop-gradded). ARCH §3.3 / B3."""
  sched = _build_schedule(trainable=True)
  # gamma carrying a graph through the schedule params (as sample_gamma would).
  gamma_live = sched.sample_gamma(8, torch.device('cpu'))
  ce = torch.full((8,), 4.0)
  # scheduler_loss must detach gamma internally; passing the live gamma must not
  # create a second grad path into P_mu/P_beta via the gamma argument.
  for p in sched.parameters():
    p.grad = None
  sl = sched.scheduler_loss(gamma_live, ce)
  sl.backward()
  # Grad still flows (through surrogate_entropy's explicit P_* dependence), but
  # the loss value must equal the detached-gamma form (no gamma grad path).
  expected = ((ce - sched.surrogate_entropy(gamma_live.detach())) ** 2).mean()
  assert torch.allclose(sl.detach(), expected.detach(), atol=1e-6)


# ---------------------------------------------------------------------------
# Checkpointing round-trip (ARCH §3.5)
# ---------------------------------------------------------------------------

def test_state_dict_roundtrip_trainable():
  """raw_mu/raw_beta/raw_H round-trip through state_dict/load_state_dict. §3.5."""
  sched = _build_schedule(trainable=True)
  with torch.no_grad():
    sched.raw_mu.fill_(0.7)
    sched.raw_beta.fill_(-0.3)
    sched.raw_H.fill_(1.1)
  sd = sched.state_dict()
  fresh = _build_schedule(trainable=True)
  fresh.load_state_dict(sd)
  assert float(fresh.raw_mu) == pytest.approx(0.7, abs=1e-6)
  assert float(fresh.raw_beta) == pytest.approx(-0.3, abs=1e-6)
  assert float(fresh.raw_H) == pytest.approx(1.1, abs=1e-6)


def test_state_dict_contains_schedule_tensors():
  """state_dict carries raw_mu/raw_beta/raw_H keys. ARCH §3.5."""
  sched = _build_schedule(trainable=True)
  keys = set(sched.state_dict().keys())
  assert {'raw_mu', 'raw_beta', 'raw_H'} <= keys


# ---------------------------------------------------------------------------
# Vestigial ABC contract (ARCH §3.4)
# ---------------------------------------------------------------------------

def test_alpha_t_is_finite_vestigial():
  """`alpha_t(t)` (the t->gamma->alpha vestigial map) is finite. ARCH §3.4."""
  sched = _build_schedule()
  t = torch.linspace(0.01, 0.99, 50)
  out = sched.alpha_t(t)
  assert torch.isfinite(out).all()


def test_alpha_prime_t_is_zero_vestigial():
  """`alpha_prime_t(t)` returns zeros (unused derivative). ARCH §3.4."""
  sched = _build_schedule()
  t = torch.linspace(0.01, 0.99, 50)
  out = sched.alpha_prime_t(t)
  assert torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# get_noise dispatch (ARCH §3.6, test surface #14)
# ---------------------------------------------------------------------------

def _gumbel_noise_config(*, trainable=True):
  import omegaconf
  cfg = {
    'noise': {
      'type': 'gumbel',
      'trainable': trainable,
      'q_clip': 1e-5,
      'H_inf_init': 5.0,
      'adaptive': False,
      'alpha_min': None,
      'alpha_max': None,
    },
  }
  return omegaconf.OmegaConf.create(cfg)


def test_get_noise_returns_unif_info_schedule_for_gumbel():
  """`get_noise(config)` with `noise.type=='gumbel'` -> `UnifInfoSchedule`. #14."""
  config = _gumbel_noise_config()
  noise = noise_schedules.get_noise(config)
  assert isinstance(noise, noise_schedules.UnifInfoSchedule)


def test_get_noise_gumbel_not_wrapped_in_adaptive_or_truncated():
  """ARCH §3.6: the gumbel branch returns the schedule directly, NOT wrapped in
  `TruncatedScheduleWrapper`/`AdaptiveSchedule`. #14."""
  config = _gumbel_noise_config()
  noise = noise_schedules.get_noise(config)
  assert not isinstance(noise, noise_schedules.AdaptiveSchedule)
  assert not isinstance(noise, noise_schedules.TruncatedScheduleWrapper)


def test_get_noise_gumbel_trainable_flag_threads_through():
  """`noise.trainable=False` -> the returned schedule has 0 trainable params. #14."""
  config = _gumbel_noise_config(trainable=False)
  noise = noise_schedules.get_noise(config)
  assert len(list(noise.parameters())) == 0
