"""Contract tests for the learned noise schedule (MuLAN-style).

  models/var_noise.py         GammaNet -> (tau, dtau/dt)
  noise_schedules.py          VariationalAdaptiveSchedule

Math contract:
  tau(c, 0) = 0, tau(c, 1) = 1, dtau/dt >= 0                (monotone warp)
  alpha_t(c, t)  = alpha_base(tau(c, t))
  alpha'_t(c, t) = alpha'_base(tau) * dtau/dt
  w = (-alpha') / (alpha(0) - alpha(1)),  E_{t~U[0,1]}[w] = 1
  At init (zero head weight + identity bias) tau == t, so alpha == alpha_base
  and w == 1: the schedule starts exactly at the un-warped EFLM baseline.
"""
import pytest
import torch

import noise_schedules
from conftest import REPO_ROOT  # noqa: F401  (ensures repo root on sys.path)
from models.var_noise import GammaNet

torch.manual_seed(0)

# The DiT trunk goes through flash-attn's triton rotary kernel, which needs
# CUDA tensors; run the whole suite on the GPU when there is one.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

B, L, D = 3, 7, 16
EMBED, LAYERS = 32, 1

CASES = [(basis, scope, model_type)
         for basis in ('polynomial', 'rbf')
         for scope in ('positional', 'global')
         for model_type in ('mlp', 'dit')]
IDS = ['-'.join(c) for c in CASES]


def _net(basis='polynomial', scope='positional', model_type='mlp', degree=5,
         in_dim=D):
  return GammaNet(in_dim=in_dim, embed_dim=EMBED, num_layer=LAYERS,
                  model_type=model_type, scope=scope, basis=basis,
                  degree=degree).to(DEVICE).eval()


def _schedule(**kw):
  return noise_schedules.VariationalAdaptiveSchedule(
    base_schedule=noise_schedules.LogLinear(1e-3),
    gamma_net=_net(**kw)).to(DEVICE).eval()


def _ctx():
  return torch.randn(B, L, D, device=DEVICE)


def _expected_shape(scope):
  return (B,) if scope == 'global' else (B, L)


# ---------------------------------------------------------------------------
# GammaNet: shapes, endpoints, monotonicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('basis,scope,model_type', CASES, ids=IDS)
def test_shapes(basis, scope, model_type):
  net = _net(basis=basis, scope=scope, model_type=model_type)
  tau, dtau = net(torch.rand(B, device=DEVICE), _ctx())
  assert tau.shape == _expected_shape(scope)
  assert dtau.shape == _expected_shape(scope)


@pytest.mark.parametrize('basis,scope,model_type', CASES, ids=IDS)
def test_endpoints(basis, scope, model_type):
  net = _net(basis=basis, scope=scope, model_type=model_type)
  # Randomize the head so the warp is a non-trivial function of c.
  torch.nn.init.normal_(net.head.weight, std=0.5)
  c = _ctx()
  tau0, _ = net(torch.zeros(B, device=DEVICE), c)
  tau1, _ = net(torch.ones(B, device=DEVICE), c)
  assert torch.allclose(tau0, torch.zeros_like(tau0), atol=1e-5)
  assert torch.allclose(tau1, torch.ones_like(tau1), atol=1e-5)


@pytest.mark.parametrize('basis,scope,model_type', CASES, ids=IDS)
def test_monotone(basis, scope, model_type):
  net = _net(basis=basis, scope=scope, model_type=model_type)
  torch.nn.init.normal_(net.head.weight, std=0.5)
  c = _ctx()
  grid = torch.linspace(0.0, 1.0, 21, device=DEVICE)
  taus, dtaus = zip(*[net(g.expand(B), c) for g in grid])
  taus = torch.stack(taus)          # [21, ...]
  assert (taus[1:] - taus[:-1] >= -1e-6).all(), 'tau must increase in t'
  assert (torch.stack(dtaus) >= -1e-6).all(), 'dtau/dt must be >= 0'


def test_identity_init_polynomial():
  """Zero head weight + the identity bias must give tau(t) == t exactly."""
  net = _net(basis='polynomial')
  t = torch.rand(B, device=DEVICE)
  tau, dtau = net(t, _ctx())
  assert torch.allclose(tau, t[:, None].expand_as(tau), atol=1e-5)
  assert torch.allclose(dtau, torch.ones_like(dtau), atol=1e-5)


def test_identity_init_rbf():
  """The rbf basis only approximates the identity: a comb of Gaussians on
  [0, 1] sags near the endpoints (no bases outside the interval), so tau'
  ripples by ~20%. tau itself stays close to t, which is what matters."""
  net = _net(basis='rbf')
  t = torch.rand(B, device=DEVICE)
  tau, dtau = net(t, _ctx())
  assert torch.allclose(tau, t[:, None].expand_as(tau), atol=5e-2)
  assert (dtau > 0.5).all() and (dtau < 1.5).all()


def test_rejects_even_degree():
  with pytest.raises(ValueError):
    _net(basis='polynomial', degree=4)


@pytest.mark.parametrize('degree', [7, 9])
def test_higher_polynomial_degree(degree):
  """degree 7/9 (p cubic/quartic): identity init exact, endpoints exact,
  monotone, and the analytic dtau/dt matches autograd."""
  net = _net(basis='polynomial', degree=degree)
  t = torch.rand(B, device=DEVICE)
  tau, dtau = net(t, _ctx())
  assert torch.allclose(tau, t[:, None].expand_as(tau), atol=1e-5)
  assert torch.allclose(dtau, torch.ones_like(dtau), atol=1e-5)

  torch.nn.init.normal_(net.head.weight, std=0.5)
  c = _ctx()
  tau0, _ = net(torch.zeros(B, device=DEVICE), c)
  tau1, _ = net(torch.ones(B, device=DEVICE), c)
  assert torch.allclose(tau0, torch.zeros_like(tau0), atol=1e-5)
  assert torch.allclose(tau1, torch.ones_like(tau1), atol=1e-5)
  grid = torch.linspace(0.0, 1.0, 41, device=DEVICE)
  taus = torch.stack([net(g.expand(B), c)[0] for g in grid])
  assert (taus[1:] - taus[:-1] >= -1e-6).all()
  tg = torch.full((B,), 0.61, device=DEVICE, requires_grad=True)
  tau_g, dtau_g = net(tg, c)
  grad = torch.autograd.grad(tau_g.sum(), tg)[0]
  assert torch.allclose(grad, dtau_g.sum(-1), rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# PromptContext (inference-available schedule context, MuLAN D.1)
# ---------------------------------------------------------------------------

def _prompt_ctx(z_dim=8, vocab=12, width=16, max_len=L):
  from models.var_noise import PromptContext
  return PromptContext(vocab, z_dim, width, max_len).to(DEVICE)


def test_prompt_context_shapes():
  pc = _prompt_ctx()
  x0 = torch.randint(0, 12, (B, L), device=DEVICE)
  z = pc(x0)
  assert z.shape == (B, 8)
  assert pc.expand(z, L).shape == (B, L, 8)


def test_prompt_context_reads_only_the_prompt():
  """z must be a function of the PROMPT (valid_tokens == 0) alone: it is the
  inference-available context, so the generated (solution) tokens must not
  leak in -- and changing the prompt must change it."""
  pc = _prompt_ctx()
  x0 = torch.randint(0, 12, (B, L), device=DEVICE)
  valid = torch.ones(B, L, device=DEVICE)
  valid[:, : L // 2] = 0
  x0_gen = x0.clone()
  x0_gen[:, L // 2:] = (x0_gen[:, L // 2:] + 1) % 12   # change solution only
  assert torch.allclose(pc(x0, valid), pc(x0_gen, valid), atol=1e-6)
  x0_pr = x0.clone()
  x0_pr[:, : L // 2] = (x0_pr[:, : L // 2] + 1) % 12   # change prompt only
  assert not torch.allclose(pc(x0, valid), pc(x0_pr, valid), atol=1e-4)


def test_prompt_context_matches_sampler_construction():
  """Training pools the prompt slice of the full x0; the sampler encodes the
  prefix tokens alone.  Both must give the same z."""
  pc = _prompt_ctx()
  P = L // 2
  x0 = torch.randint(0, 12, (B, L), device=DEVICE)
  valid = torch.ones(B, L, device=DEVICE)
  valid[:, :P] = 0
  z_train = pc(x0, valid)
  prefix = x0[:, :P]
  z_sample = pc(prefix, torch.zeros_like(prefix))
  assert torch.allclose(z_train, z_sample, atol=1e-6)


def test_prompt_context_position_sensitivity():
  """Clue LOCATIONS matter (sudoku): permuting the prompt tokens must change
  z even though the token multiset is unchanged."""
  pc = _prompt_ctx()
  x0 = torch.randint(0, 12, (B, L), device=DEVICE)
  valid = torch.ones(B, L, device=DEVICE)
  valid[:, : L // 2] = 0
  x0_perm = x0.clone()
  x0_perm[:, : L // 2] = x0_perm[:, : L // 2].flip(dims=[1])
  assert not torch.allclose(pc(x0, valid), pc(x0_perm, valid), atol=1e-4)


def test_prompt_context_feeds_gamma_net():
  """Different prompts -> different per-position schedules."""
  pc = _prompt_ctx()
  net = _net(in_dim=8)  # GammaNet input = the z_dim-wide context
  torch.nn.init.normal_(net.head.weight, std=0.5)
  t = torch.full((B,), 0.5, device=DEVICE)
  x1 = torch.randint(0, 12, (B, L), device=DEVICE)
  x2 = torch.randint(0, 12, (B, L), device=DEVICE)
  tau1, _ = net(t, pc.expand(pc(x1), L))
  tau2, _ = net(t, pc.expand(pc(x2), L))
  assert tau1.shape == (B, L)
  assert not torch.allclose(tau1, tau2, atol=1e-4)


# ---------------------------------------------------------------------------
# Warm-up gate + train-mode determinism
# ---------------------------------------------------------------------------

def test_gate_zero_reproduces_base_schedule():
  """gate 0 must give exactly the base schedule (up to the u-relabelling)
  regardless of how far the warp has trained -- the warm-up anchor."""
  sched = _schedule()
  torch.nn.init.normal_(sched.gamma_net.head.weight, std=1.0)
  t = torch.rand(B, device=DEVICE)
  c = _ctx()
  a0 = sched.alpha_t(t, c, gate=0.0)
  base = sched.base_schedule.alpha_t(t)[:, None].expand_as(a0)
  assert torch.allclose(a0, base, atol=1e-5)
  w = sched.loss_weight(sched(t, c, gate=0.0)[0])
  assert torch.allclose(w, torch.ones_like(w), atol=1e-5)


def test_train_mode_forwards_are_deterministic():
  """The loss weight divides two GammaNet evaluations; train-mode dropout
  would give them different coefficients (external-review finding).  The net
  must be deterministic in train mode."""
  net = _net(model_type='dit')
  torch.nn.init.normal_(net.head.weight, std=0.5)
  net.train()
  t = torch.rand(B, device=DEVICE)
  c = _ctx()
  tau1, dtau1 = net(t, c)
  tau2, dtau2 = net(t, c)
  assert torch.equal(tau1, tau2) and torch.equal(dtau1, dtau2)


@pytest.mark.parametrize('basis', ['polynomial', 'rbf'])
def test_endpoints_survive_collapsed_coefficients(basis):
  """When the basis coefficients collapse toward zero the ridge on f' must
  keep the warp well-defined with tau(1) = 1 exact (no 0/0, no shrunken
  endpoint from a clamped normalizer)."""
  net = _net(basis=basis)
  net.head.weight.data.zero_()
  # polynomial: psi ~ 0 -> f_poly ~ 0; rbf: softplus(-30) ~ 0 weights.
  net.head.bias.data.fill_(1e-8 if basis == 'polynomial' else -30.0)
  c = _ctx()
  tau1, _ = net(torch.ones(B, device=DEVICE), c)
  tau0, _ = net(torch.zeros(B, device=DEVICE), c)
  assert torch.allclose(tau1, torch.ones_like(tau1), atol=1e-5)
  assert torch.allclose(tau0, torch.zeros_like(tau0), atol=1e-5)
  t = torch.rand(B, device=DEVICE)
  tau, dtau = net(t, c)
  assert torch.isfinite(tau).all() and torch.isfinite(dtau).all()
  # ridge-dominated warp is the identity
  assert torch.allclose(tau, t[:, None].expand_as(tau), atol=1e-4)


def test_derivative_matches_autograd():
  net = _net()
  torch.nn.init.normal_(net.head.weight, std=0.5)
  c = _ctx()
  t = torch.full((B,), 0.37, device=DEVICE, requires_grad=True)
  tau, dtau = net(t, c)
  grad = torch.autograd.grad(tau.sum(), t)[0]        # d(sum_l tau)/dt per b
  assert torch.allclose(grad, dtau.sum(-1), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# VariationalAdaptiveSchedule
# ---------------------------------------------------------------------------

def test_falls_back_to_base_without_context():
  sched = _schedule()
  t = torch.rand(B, device=DEVICE)
  base = sched.base_schedule
  assert torch.allclose(sched.alpha_t(t), base.alpha_t(t))
  assert torch.allclose(sched.alpha_prime_t(t), base.alpha_prime_t(t))
  dalpha, alpha = sched(t)
  assert torch.allclose(alpha, base.alpha_t(t))
  assert torch.allclose(dalpha, base.alpha_prime_t(t))


def test_init_reproduces_base_schedule():
  """At init the learned schedule IS the base schedule, weight w == 1."""
  sched = _schedule()
  t = torch.rand(B, device=DEVICE)
  c = _ctx()
  dalpha, alpha = sched(t, c)
  base_alpha = sched.base_schedule.alpha_t(t)[:, None].expand_as(alpha)
  assert torch.allclose(alpha, base_alpha, atol=1e-5)
  w = sched.loss_weight(dalpha)
  assert torch.allclose(w, torch.ones_like(w), atol=1e-5)


@pytest.mark.parametrize('scope', ['positional', 'global'])
def test_alpha_decreases_in_t(scope):
  sched = _schedule(scope=scope)
  torch.nn.init.normal_(sched.gamma_net.head.weight, std=0.5)
  c = _ctx()
  grid = torch.linspace(0.0, 1.0, 21, device=DEVICE)
  alphas = torch.stack([sched.alpha_t(g.expand(B), c) for g in grid])
  assert (alphas[1:] - alphas[:-1] <= 1e-6).all()
  assert torch.allclose(alphas[0], torch.ones_like(alphas[0]), atol=1e-3)


@pytest.mark.parametrize('scope', ['positional', 'global'])
def test_loss_weight_averages_to_one(scope):
  """E_{t~U[0,1]}[w] == 1 for ANY warp -- this is what makes the MuLAN
  objective invariant for a global schedule (no degenerate optimum)."""
  sched = _schedule(scope=scope)
  torch.nn.init.normal_(sched.gamma_net.head.weight, std=0.5)
  c = _ctx()
  # Midpoint rule over a fine grid.
  grid = (torch.arange(400, device=DEVICE) + 0.5) / 400
  w = torch.stack([sched.loss_weight(sched(g.expand(B), c)[0])
                   for g in grid])
  assert torch.allclose(w.mean(0), torch.ones_like(w[0]), atol=1e-2)


def test_loss_weight_exact_on_truncated_interval():
  """Training samples t ~ U[t_min, 1].  The u-parametrized warp pins its
  endpoints on that interval, so E_t[w | c] == 1 exactly there for an
  arbitrary warp with the constant span -- no per-context correction
  needed."""
  t_min = 0.05  # exaggerated sampling_eps so any interval bias is visible
  sched = noise_schedules.VariationalAdaptiveSchedule(
    base_schedule=noise_schedules.LogLinear(1e-3),
    gamma_net=_net(), t_min=t_min).to(DEVICE).eval()
  torch.nn.init.normal_(sched.gamma_net.head.weight, std=1.0)
  c = _ctx()
  grid = t_min + (1 - t_min) * (torch.arange(800, device=DEVICE) + 0.5) / 800
  w = torch.stack([
    sched.loss_weight(sched(g.expand(B), c)[0]) for g in grid])
  assert torch.allclose(w.mean(0), torch.ones_like(w[0]), atol=5e-3)


def test_learned_schedule_is_context_dependent():
  """Different contexts must give different schedules once the head is not
  zero -- otherwise there is no decoding order to learn."""
  sched = _schedule()
  torch.nn.init.normal_(sched.gamma_net.head.weight, std=0.5)
  t = torch.full((B,), 0.5, device=DEVICE)
  a1 = sched.alpha_t(t, _ctx())
  a2 = sched.alpha_t(t, _ctx())
  assert not torch.allclose(a1, a2, atol=1e-4)
  assert a1.std(dim=-1).min() > 0, 'alpha must vary across positions'


def test_params_are_registered():
  """The gamma net must be a submodule so TrainerBase._get_parameters (and
  therefore the optimizer + EMA) picks it up via self.noise.parameters()."""
  sched = _schedule()
  names = {n for n, _ in sched.named_parameters()}
  assert any(n.startswith('gamma_net.') for n in names)
  assert len(list(sched.parameters())) > 0


# ---------------------------------------------------------------------------
# EFLM.dlm_state: the context pass must not poison the autocast weight cache
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='autocast weight cache is CUDA-only')
def test_context_pass_keeps_backbone_in_the_graph():
  """`EFLM.dlm_state` runs the backbone under no_grad INSIDE the training
  step's autocast region. autocast caches the casted copy of every weight for
  the lifetime of the outermost region; if that cache is filled by a no_grad
  pass, the loss forward reuses those detached copies and the whole backbone
  drops out of the graph -- DDP then fails with "parameters that were not used
  in producing the loss". Guarded by cache_enabled=False in dlm_state.
  """
  import omegaconf

  import algo
  import models.sphere_dit
  import trainer_base

  d, length, vocab = 16, 6, 12
  cfg = omegaconf.OmegaConf.create({
    'model': {'name': 'tiny', 'type': 'sphere-dit', 'hidden_size': d,
              'cond_dim': 32, 'length': length, 'n_blocks': 1, 'n_heads': 4,
              'dropout': 0.0, 'init': 'unit_var', 'init_std': None,
              'learn_temperature_scaling': False, 'eps': 1e-6,
              'pretrained_ckpt_path': None},
    'algo': {'name': 'eflm', 'adaLN': True, 'time_conditioning': True},
  })
  backbone = models.sphere_dit.SphereDiT(cfg, vocab_size=vocab).to(DEVICE)

  class _Stub:  # binds the real EFLM methods without Diffusion.__init__
    def __init__(self):
      self.config = cfg
      self.backbone = backbone
      self.time_conditioning = True

    def __getattr__(self, name):
      for cls in (algo.EFLM, trainer_base.Diffusion, trainer_base.TrainerBase):
        attr = cls.__dict__.get(name)
        if attr is not None and callable(attr):
          return attr.__get__(self, type(self))
      raise AttributeError(name)

  stub = _Stub()
  stub.variational_noise = False  # read by EFLM._process_sigma
  xt = torch.randn(2, length, d, device=DEVICE)
  sigma = torch.full((2, 1), 0.5, device=DEVICE)

  with torch.autocast('cuda', dtype=torch.bfloat16):
    c = stub.dlm_state(xt, sigma)          # no-grad context pass
    out = stub.forward(xt=xt, sigma=sigma)  # loss pass
  assert c.shape == (2, length, d)
  out.float().sum().backward()

  # sphere_embed is excluded: this test feeds xt directly, so the embedding
  # table is genuinely unused here.  Everything else lives inside the
  # backbone's bf16 autocast region -- exactly what the cache would detach.
  no_grad = [n for n, p in backbone.named_parameters()
             if p.requires_grad and p.grad is None
             and not n.startswith('sphere_embed')]
  assert not no_grad, f'backbone params cut out of the graph: {no_grad}'
