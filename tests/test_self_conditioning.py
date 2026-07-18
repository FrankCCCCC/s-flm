"""Self-conditioning for the flow LMs (SFM / EFLM / HFLM).

Mirrors the LangFlow reference (`algo.LangFlow._self_cond_pass`,
`SphereDiT.W_in/W_sc`; see tests/test_langflow.py): the shared
`algo.SelfConditioning` mixin runs a no-grad first pass with z_sc=0
(with prob p_self_cond at train time, always at eval) and feeds the
expected model-input embedding back via `context.z_sc`.
"""
import pytest
import torch

import algo
import samplers
import trainer_base
from conftest import REPO_ROOT  # noqa: F401  (ensures repo root on sys.path)

torch.manual_seed(0)

ALGOS = ['sfm', 'eflm', 'hflm']

# models/dit.py hard-imports flash_attn, whose rotary kernel only takes GPU
# tensors -- any DiT-block forward needs CUDA (same constraint as the
# pre-existing SphereDiT forward tests in this env).
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
needs_gpu = pytest.mark.skipif(
  not torch.cuda.is_available(),
  reason='DiT forward requires flash-attn rotary (GPU tensors)')


# ---------------------------------------------------------------------------
# Config + stub helpers
# ---------------------------------------------------------------------------

def _make_config(algo_name, *, d=16, length=6, vocab_size=12, n_blocks=1,
                 n_heads=4, self_conditioning=True, p_self_cond=0.25):
  import omegaconf
  is_hflm = algo_name == 'hflm'
  cfg = {
    'model': {
      'name': 'tiny',
      'type': 'hyperbolic-dit' if is_hflm else 'sphere-dit',
      'hidden_size': d,
      'cond_dim': 32,
      'length': length,
      'n_blocks': n_blocks,
      'n_heads': n_heads,
      'dropout': 0.0,
      'init': 'hyperbolic' if is_hflm else 'unit_var',
      'learn_temperature_scaling': False,
      'eps': 1e-6,
      'pretrained_ckpt_path': None,
    },
    'algo': {
      'name': algo_name,
      'parameterization': 'mean',
      'time_conditioning': True,
      'loss_type': 'ce',
      'T': 0,
      'adaLN': True,
      'slerp_precision': 'float64',
      'eps': 1e-6,
      'invert_time_convention': False,
      'renormalize_weights': False,
      'self_conditioning': self_conditioning,
      'p_self_cond': p_self_cond,
      'prior_cov': 0.25,
      'rho_max': 12,
      'gaussian_curvature': -1.0,
    },
  }
  return omegaconf.OmegaConf.create(cfg)


def _make_backbone(config, vocab_size):
  if config.model.type == 'hyperbolic-dit':
    import models.hyperbolic_dit
    backbone = models.hyperbolic_dit.HyperbolicDiT(
      config, vocab_size=vocab_size)
  else:
    import models.sphere_dit
    backbone = models.sphere_dit.SphereDiT(config, vocab_size=vocab_size)
  backbone.eval()  # deterministic forward (no dropout) for allclose checks
  return backbone.to(DEVICE)


_ALGO_CLS = {'sfm': algo.SFM, 'eflm': algo.EFLM, 'hflm': algo.HFLM}


class _Stub:
  """Binds the real algo-class methods to a light object carrying just the
  attributes the self-cond path reads, without `Diffusion.__init__` (same
  pattern as tests/test_langflow.py `_LangFlowStub`)."""

  def __init__(self, algo_name, config, backbone):
    self.config = config
    self.backbone = backbone
    self.device = DEVICE
    self.T = 0
    self.time_conditioning = True
    a = config.algo
    self.eps = a.eps
    self.renormalize_weights = a.renormalize_weights
    self.invert_time_convention = a.invert_time_convention
    self.self_conditioning = a.self_conditioning
    self.p_self_cond = a.p_self_cond
    if algo_name == 'hflm':
      self.prior_cov = a.prior_cov
      self.rho_max = a.rho_max
      self.gaussian_curvature = a.gaussian_curvature
    self._algo_cls = _ALGO_CLS[algo_name]

  def __getattr__(self, name):
    for cls in (self.__dict__['_algo_cls'], algo.SelfConditioning,
                trainer_base.Diffusion, trainer_base.TrainerBase):
      attr = cls.__dict__.get(name)
      if attr is not None and callable(attr):
        return attr.__get__(self, type(self))
    raise AttributeError(name)


def _build(algo_name, **cfg_kwargs):
  vocab_size = cfg_kwargs.pop('vocab_size', 12)
  config = _make_config(algo_name, vocab_size=vocab_size, **cfg_kwargs)
  backbone = _make_backbone(config, vocab_size)
  return _Stub(algo_name, config, backbone), config, backbone


def _make_inputs(stub, B=2):
  L = stub.config.model.length
  x0 = torch.randint(0, stub.backbone.vocab_size, (B, L), device=DEVICE)
  alpha_t = torch.full((B, 1), 0.7, device=DEVICE)
  xt = stub.q_xt(x0, alpha_t, use_pure_noise=False)
  sigma = stub._sigma_from_alphat(alpha_t)
  return xt, sigma


# ===========================================================================
# Backbone: zero-init W_in/W_sc  (HyperbolicDiT gains the SphereDiT block)
# ===========================================================================

@pytest.mark.parametrize('algo_name', ALGOS)
def test_backbone_self_cond_params_exist_iff_enabled(algo_name):
  _, config, backbone = _build(algo_name, self_conditioning=True)
  assert backbone.self_conditioning
  assert torch.count_nonzero(backbone.W_in.weight) == 0
  assert torch.count_nonzero(backbone.W_sc.weight) == 0
  _, _, backbone_off = _build(algo_name, self_conditioning=False)
  assert not hasattr(backbone_off, 'W_in')
  assert not hasattr(backbone_off, 'W_sc')


@needs_gpu
@pytest.mark.parametrize('algo_name', ALGOS)
def test_zero_init_self_cond_forward_equals_baseline(algo_name):
  """At step 0 (zero W_in/W_sc), a forward with any z_sc equals both the
  z_sc=None forward and the plain no-context forward."""
  stub, _, backbone = _build(algo_name, self_conditioning=True)
  xt, sigma = _make_inputs(stub)
  sigma_1d = stub._process_sigma(sigma)
  with torch.no_grad():
    base = backbone(None, xt, sigma_1d, None)
    ctx_none = trainer_base.TrainingContext()
    out_none = backbone(None, xt, sigma_1d, ctx_none)
    ctx_rand = trainer_base.TrainingContext(z_sc=torch.randn_like(xt))
    out_rand = backbone(None, xt, sigma_1d, ctx_rand)
  assert torch.allclose(base, out_none, atol=1e-5)
  assert torch.allclose(base, out_rand, atol=1e-5)


# ===========================================================================
# _sc_embed_table: the model-input space of xt, per geometry
# ===========================================================================

def test_sc_embed_table_sfm_is_unit_norm():
  stub, _, backbone = _build('sfm')
  table = stub._sc_embed_table()
  assert table.shape == backbone.sphere_embed.weight.shape
  assert torch.allclose(table.norm(p=2, dim=-1),
                        torch.ones(backbone.vocab_size, device=DEVICE),
                        atol=1e-5)


def test_sc_embed_table_eflm_is_raw_weight():
  stub, _, backbone = _build('eflm')
  assert stub._sc_embed_table() is backbone.sphere_embed.weight


def test_sc_embed_table_hflm_is_clamped_poincare_map():
  stub, _, backbone = _build('hflm')
  table = stub._sc_embed_table()
  w = backbone.sphere_embed.weight
  # K=-1 => R=1: z = tanh(rho_eff / 2) * u with rho_eff = 12*tanh(rho/12).
  rho_eff = 12 * torch.tanh(w.norm(p=2, dim=-1) / 12)
  expected = (torch.tanh(rho_eff / 2).unsqueeze(-1)
              * w / w.norm(p=2, dim=-1, keepdim=True))
  assert torch.allclose(table, expected, atol=1e-5)
  assert (table.norm(p=2, dim=-1) < 1).all()


# ===========================================================================
# _self_cond_pass: gating + carry value (paper Alg. 1 lines 7-13)
# ===========================================================================

@needs_gpu
@pytest.mark.parametrize('algo_name', ALGOS)
def test_self_cond_pass_sets_soft_carry(algo_name):
  stub, _, _ = _build(algo_name, p_self_cond=1.0)
  xt, sigma = _make_inputs(stub)
  ctx = trainer_base.TrainingContext()
  stub._self_cond_pass(xt, sigma, ctx, train_mode=True)
  assert ctx.z_sc is not None
  assert ctx.z_sc.shape == xt.shape
  assert not ctx.z_sc.requires_grad
  # Carry == exp(first-pass log-probs) @ table, first pass run with z_sc=0.
  with torch.no_grad():
    log_xhat = stub.forward(x0=None, xt=xt, sigma=sigma,
                            context=trainer_base.TrainingContext())
    expected = log_xhat.exp() @ stub._sc_embed_table()
  assert torch.allclose(ctx.z_sc, expected, atol=1e-5)


@needs_gpu
@pytest.mark.parametrize('algo_name', ALGOS)
def test_self_cond_pass_gating(algo_name):
  stub, _, _ = _build(algo_name, p_self_cond=0.0)
  xt, sigma = _make_inputs(stub)
  ctx = trainer_base.TrainingContext(z_sc=torch.randn_like(xt))
  stub._self_cond_pass(xt, sigma, ctx, train_mode=True)
  assert ctx.z_sc is None  # train + p=0 -> reset to the no-self-cond pass
  stub._self_cond_pass(xt, sigma, ctx, train_mode=False)
  assert ctx.z_sc is not None  # eval -> always on

  stub_off, _, _ = _build(algo_name, self_conditioning=False)
  ctx = trainer_base.TrainingContext()
  stub_off._self_cond_pass(xt, sigma, ctx, train_mode=True)
  assert ctx.z_sc is None


# ===========================================================================
# Sampler carry helper
# ===========================================================================

@pytest.mark.parametrize('algo_name', ALGOS)
def test_flow_self_cond_carry_matches_training(algo_name):
  stub, _, _ = _build(algo_name)
  B, L, V = 2, stub.config.model.length, stub.backbone.vocab_size
  log_p = torch.randn(B, L, V, device=DEVICE).log_softmax(-1)
  carry = samplers.flow_self_cond_carry(stub, log_p, torch.float32)
  expected = log_p.exp() @ stub._sc_embed_table()
  assert torch.allclose(carry, expected, atol=1e-5)
  assert carry.dtype == torch.float32


def test_flow_self_cond_carry_none_when_off():
  stub, _, _ = _build('sfm', self_conditioning=False)
  log_p = torch.randn(2, 6, 12, device=DEVICE).log_softmax(-1)
  assert samplers.flow_self_cond_carry(stub, log_p, torch.float32) is None
  assert samplers.flow_self_cond_carry(object(), log_p, torch.float32) is None
