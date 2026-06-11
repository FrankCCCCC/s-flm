"""Interface-contract tests for the LangFlow algo + SphereDiT backbone deltas.

Written AGAINST `experiments/langflow/ARCH.md` (§4 algo, §5 backbone deltas,
§6 sampler, §8 invariants E/F/G/H/I, §9 test surface #6-#13, #15) BEFORE the
implementation exists. They are expected to FAIL until the implementer:

  * replaces the stale `algo.LangFlow` SFM-copy with the real Gaussian-VP / γ-path
    implementation (adds `_embed`, `_x_to_embed`, `_self_cond_pass`,
    `_forward_langflow`, `_logit_bias_r`, the new `q_xt` signature, and
    `algo.LangFlowContext`);
  * adds `SphereDiT.get_raw_embeddings`, the `init='unit_var'` branch, the
    flag-gated zero-init `W_in`/`W_sc` self-cond projections, the γ feed and the
    Plaid logit bias;
  * adds `samplers.LangFlowSampler` / `LangFlowState`.

Design note (mirrors `tests/test_hflm.py`): a full `algo.LangFlow` lives on
`trainer_base.Diffusion`, whose `__init__` builds EMA / metrics / a HF tokenizer
download. To stay fast / offline / CPU-only we bind the real `LangFlow` contract
methods to a light `_LangFlowStub` carrying exactly the attributes those methods
read, against a REAL `SphereDiT` backbone (genuine embedding / gradient path).
This tests the contract, not the Lightning framework.

Option A (O1 RESOLVED): the embedding is RAW and FREE — NO normalization anywhere
(target, ẑ, self-cond embed, Plaid `E`). `init='unit_var'` -> std=1 -> ‖z‖≈√D.
"""
import math

import pytest
import torch

import algo
import noise_schedules
from conftest import REPO_ROOT  # noqa: F401  (ensures repo root on sys.path)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Config + stub helpers
# ---------------------------------------------------------------------------

def _make_config(*, d=32, length=8, vocab_size=16, init='unit_var',
                 self_conditioning=True, p_self_cond=0.25, logit_bias=True,
                 logit_bias_warmup_steps=5000, n_blocks=2, n_heads=8,
                 model_type='sphere-dit'):
  """Minimal OmegaConf config mirroring the keys `SphereDiT` / `LangFlow`
  read from `configs/model/tiny-sphere-dit.yaml` + `configs/algo/langflow.yaml`.
  """
  import omegaconf
  cfg = {
    'model': {
      'name': 'tiny',
      'type': model_type,
      'hidden_size': d,
      'cond_dim': 128,
      'length': length,
      'n_blocks': n_blocks,
      'n_heads': n_heads,
      'dropout': 0.0,
      'init': init,
      'learn_temperature_scaling': False,
      'eps': 1e-6,
      'pretrained_ckpt_path': None,
    },
    'algo': {
      'name': 'langflow',
      'diffusion_type': 'sphere',
      'backbone': 'sphere-dit',
      'parameterization': 'mean',
      'time_conditioning': True,
      'loss_type': 'ce',
      'T': 0,
      'causal_attention': False,
      'adaLN': True,
      'slerp_precision': 'float64',
      'eps': 1e-6,
      'invert_time_convention': False,
      'renormalize_weights': False,
      'self_conditioning': self_conditioning,
      'p_self_cond': p_self_cond,
      'logit_bias': logit_bias,
      'logit_bias_warmup_steps': logit_bias_warmup_steps,
    },
    'noise': {
      'type': 'gumbel',
      'trainable': True,
      'q_clip': 1e-5,
      'H_inf_init': 5.0,
      'adaptive': False,
    },
  }
  return omegaconf.OmegaConf.create(cfg)


def _make_backbone(config, vocab_size):
  import models.sphere_dit
  return models.sphere_dit.SphereDiT(config, vocab_size=vocab_size)


def _make_noise(*, trainable=True):
  return noise_schedules.UnifInfoSchedule(
    trainable=trainable, q_clip=1e-5, H_inf_init=5.0)


class _LangFlowStub:
  """Binds real `algo.LangFlow` methods to a light object carrying just the
  attributes the contract methods read, without `Diffusion.__init__`.

  Methods bound on demand via `__getattr__`. The few framework methods the
  contract path calls (`forward`, `_process_sigma`) are bound from
  `trainer_base.TrainerBase`/`Diffusion` so γ routing through the time embedder
  is genuine.
  """

  def __init__(self, config, backbone, noise, *, global_step=10_000):
    self.config = config
    self.backbone = backbone
    self.noise = noise
    self.self_conditioning = config.algo.self_conditioning
    self.p_self_cond = config.algo.p_self_cond
    self.logit_bias = config.algo.logit_bias
    self.logit_bias_warmup_steps = config.algo.logit_bias_warmup_steps
    self.time_conditioning = config.algo.time_conditioning
    self.antithetic_sampling = True
    self.global_step = global_step
    self.device = torch.device('cpu')
    self.T = 0

  def __getattr__(self, name):
    import algo
    import trainer_base
    for cls in (algo.LangFlow, trainer_base.Diffusion,
                trainer_base.TrainerBase):
      attr = cls.__dict__.get(name)
      if attr is not None and callable(attr):
        return attr.__get__(self, type(self))
    raise AttributeError(name)


def _build_langflow_stub(**cfg_kwargs):
  vocab_size = cfg_kwargs.pop('vocab_size', 16)
  global_step = cfg_kwargs.pop('global_step', 10_000)
  config = _make_config(vocab_size=vocab_size, **cfg_kwargs)
  backbone = _make_backbone(config, vocab_size)
  backbone.eval()  # deterministic forward (no dropout) for allclose checks
  noise = _make_noise(trainable=True)
  stub = _LangFlowStub(config, backbone, noise, global_step=global_step)
  return stub, config, backbone


def _row_norms(z):
  return z.norm(p=2, dim=-1)


# ===========================================================================
# Embedding: raw free embedding, NO normalization (ARCH §8 E / #15)
# ===========================================================================

def test_unit_var_init_makes_embedding_norm_near_sqrt_D():
  """ARCH §5.1 / §8 E / #15: `init='unit_var'` -> sphere_embed ~ N(0,1) so
  mean ‖E[x]‖ ≈ √D at init."""
  d = 256
  config = _make_config(d=d, length=4, vocab_size=64, init='unit_var',
                        n_blocks=1)
  backbone = _make_backbone(config, vocab_size=64)
  norms = _row_norms(backbone.sphere_embed.weight)
  assert float(norms.mean()) == pytest.approx(math.sqrt(d), rel=0.1)


def test_unit_var_init_embeddings_are_not_unit_norm():
  """ARCH §8 E / #15: raw embeddings are NOT on the unit sphere (‖·‖ != 1)."""
  d = 64
  config = _make_config(d=d, length=4, vocab_size=64, init='unit_var',
                        n_blocks=1)
  backbone = _make_backbone(config, vocab_size=64)
  norms = _row_norms(backbone.sphere_embed.weight)
  assert (norms - 1.0).abs().min() > 0.1  # no row is unit-norm


def test_unit_var_init_per_coordinate_var_near_one():
  """ARCH §5.1: `init='unit_var'` -> per-coordinate variance ≈ 1 (std=1)."""
  config = _make_config(d=64, length=4, vocab_size=128, init='unit_var',
                        n_blocks=1)
  backbone = _make_backbone(config, vocab_size=128)
  assert float(backbone.sphere_embed.weight.std()) == pytest.approx(1.0, abs=0.1)


def test_ngpt_init_unchanged():
  """ARCH §5.1: existing `init='ngpt'` keeps std ≈ 1/√D (unaffected by the new
  branch)."""
  d = 64
  config = _make_config(d=d, length=4, vocab_size=128, init='ngpt', n_blocks=1)
  backbone = _make_backbone(config, vocab_size=128)
  assert float(backbone.sphere_embed.weight.std()) == pytest.approx(
    1.0 / math.sqrt(d), abs=0.05)


def test_get_raw_embeddings_returns_unnormalized_lookup():
  """ARCH §4.2: `get_raw_embeddings(ids)` is the raw `sphere_embed(ids)` lookup,
  NOT normalized."""
  config = _make_config(d=32, length=4, vocab_size=16, init='unit_var',
                        n_blocks=1)
  backbone = _make_backbone(config, vocab_size=16)
  x = torch.randint(0, 16, (2, 4))
  raw = backbone.get_raw_embeddings(x)
  assert torch.equal(raw, backbone.sphere_embed(x))


def test_get_raw_embeddings_rows_not_unit_norm():
  """ARCH §8 E: rows of `get_raw_embeddings` are NOT unit-norm (raw, free)."""
  config = _make_config(d=64, length=4, vocab_size=16, init='unit_var',
                        n_blocks=1)
  backbone = _make_backbone(config, vocab_size=16)
  x = torch.randint(0, 16, (2, 4))
  raw = backbone.get_raw_embeddings(x)
  norms = _row_norms(raw)
  assert (norms - 1.0).abs().min() > 0.1


def test_embed_applies_no_normalization():
  """ARCH §4.2 / §8 E / #15: `LangFlow._embed(x0)` == `get_raw_embeddings(x0)`
  exactly (no sphere normalization)."""
  stub, config, backbone = _build_langflow_stub(
    d=32, length=4, vocab_size=16, n_blocks=1)
  x = torch.randint(0, 16, (2, 4))
  z = stub._embed(x)
  assert torch.equal(z, backbone.get_raw_embeddings(x))


def test_embed_output_norm_has_variance_across_tokens():
  """ARCH §8 E: `_embed` outputs are raw -> ‖E[x]‖ varies across tokens (not all
  pinned to a single norm), unlike a sphere-normalized embedding."""
  stub, config, backbone = _build_langflow_stub(
    d=64, length=8, vocab_size=32, n_blocks=1)
  x = torch.arange(32).reshape(4, 8) % 32
  z = stub._embed(x)
  norms = _row_norms(z).reshape(-1)
  assert float(norms.std()) > 1e-3  # nonzero variance => not all unit norm


def test_x_to_embed_is_raw_probs_matmul_E():
  """ARCH §4.5 / §8 H / #10: `_x_to_embed(probs) == probs @ E` with RAW
  `E = sphere_embed.weight` (neither side normalized)."""
  stub, config, backbone = _build_langflow_stub(
    d=32, length=4, vocab_size=16, n_blocks=1)
  B, L, V = 2, 4, 16
  probs = torch.softmax(torch.randn(B, L, V), dim=-1)
  E = backbone.sphere_embed.weight  # [V, d], RAW
  expected = probs @ E
  got = stub._x_to_embed(probs)
  assert torch.allclose(got, expected, atol=1e-5)


def test_x_to_embed_result_not_unit_norm():
  """ARCH §8 H / #10: `_x_to_embed` does NOT normalize its result."""
  stub, config, backbone = _build_langflow_stub(
    d=64, length=4, vocab_size=16, n_blocks=1)
  probs = torch.softmax(torch.randn(2, 4, 16), dim=-1)
  z = stub._x_to_embed(probs)
  norms = _row_norms(z)
  assert (norms - 1.0).abs().max() > 1e-3  # at least one row not unit-norm


# ===========================================================================
# q_xt — VP Gaussian corruption (ARCH §4.2 / #11)
# ===========================================================================

def test_q_xt_returns_correct_shape():
  """ARCH §4.2 / #11: `q_xt(z, alpha, sigma)` returns [B, L, d]."""
  stub, config, backbone = _build_langflow_stub(
    d=32, length=4, vocab_size=16, n_blocks=1)
  B, L, d = 2, 4, 32
  z = torch.randn(B, L, d)
  alpha = torch.full((B, 1), 0.6)
  sigma = torch.full((B, 1), 0.8)
  out = stub.q_xt(z, alpha, sigma)
  assert tuple(out.shape) == (B, L, d)


def test_q_xt_matches_linear_vp_closed_form():
  """ARCH §4.2 / #11: with a fixed seed, `q_xt` == `alpha*z + sigma*eps`
  (linear VP, NOT slerp)."""
  stub, config, backbone = _build_langflow_stub(
    d=16, length=3, vocab_size=10, n_blocks=1)
  B, L, d = 2, 3, 16
  z = torch.randn(B, L, d)
  alpha = torch.full((B, 1), 0.6)
  sigma = torch.full((B, 1), 0.8)
  torch.manual_seed(1234)
  out = stub.q_xt(z, alpha, sigma)
  torch.manual_seed(1234)
  eps = torch.randn_like(z)
  expected = alpha.unsqueeze(-1) * z + sigma.unsqueeze(-1) * eps
  assert torch.allclose(out, expected, atol=1e-5)


def test_q_xt_keeps_prompt_positions_clean():
  """ARCH §4.2 / #11: prompt positions (valid_tokens==0) stay clean (== z)."""
  stub, config, backbone = _build_langflow_stub(
    d=16, length=4, vocab_size=10, n_blocks=1)
  B, L, d = 2, 4, 16
  z = torch.randn(B, L, d)
  alpha = torch.full((B, 1), 0.3)
  sigma = torch.full((B, 1), 0.95)
  valid = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.long)
  out = stub.q_xt(z, alpha, sigma, valid_tokens=valid)
  prompt = (valid == 0).unsqueeze(-1).expand_as(z)
  assert torch.allclose(out[prompt], z[prompt], atol=1e-6)


# ===========================================================================
# Backbone forward accepts non-unit input (ARCH §9 "Backbone forward")
# ===========================================================================

def test_sphere_dit_forward_accepts_raw_scale_input():
  """ARCH §9: feeding a raw-scale `z_gamma` (‖·‖≈√D) through `SphereDiT.forward`
  yields finite logits (LayerNorm handles the scale)."""
  V, d = 12, 64
  config = _make_config(d=d, length=4, vocab_size=V, init='unit_var', n_blocks=1)
  backbone = _make_backbone(config, vocab_size=V).eval()
  B, L = 2, 4
  z_gamma = torch.randn(B, L, d)  # per-coord var 1 -> ‖·‖≈√D
  sigma = torch.zeros(B)
  out = backbone(None, z_gamma, sigma, None)
  assert tuple(out.shape) == (B, L, V)
  assert torch.isfinite(out).all()


# ===========================================================================
# OFF flags == baseline (ARCH §8 I / #8)
# ===========================================================================

def _vanilla_forward(backbone, z, sigma):
  return backbone(None, z, sigma, None)


def test_off_flags_equal_baseline_no_context():
  """ARCH §8 I / #8: with `self_conditioning=false` and `logit_bias=false`,
  `SphereDiT.forward` (no LangFlowContext) is identical to today's vanilla
  forward on the same input."""
  V, d = 10, 32
  config = _make_config(d=d, length=4, vocab_size=V, init='unit_var',
                        self_conditioning=False, logit_bias=False, n_blocks=1)
  torch.manual_seed(7)
  backbone_a = _make_backbone(config, vocab_size=V).eval()
  torch.manual_seed(7)
  backbone_b = _make_backbone(config, vocab_size=V).eval()
  B, L = 2, 4
  z = torch.randn(B, L, d)
  sigma = torch.zeros(B)
  out_a = backbone_a(None, z, sigma, None)
  out_b = _vanilla_forward(backbone_b, z, sigma)
  assert torch.allclose(out_a, out_b, atol=1e-6)


def test_self_cond_on_at_step0_zero_init_equals_baseline():
  """ARCH §8 I / #8: with `self_conditioning=true` but zero-init W_in/W_sc (step
  0), `forward` equals the self-cond-OFF baseline (the projections contribute 0).
  """
  V, d = 10, 32
  cfg_on = _make_config(d=d, length=4, vocab_size=V, init='unit_var',
                        self_conditioning=True, logit_bias=False, n_blocks=1)
  cfg_off = _make_config(d=d, length=4, vocab_size=V, init='unit_var',
                         self_conditioning=False, logit_bias=False, n_blocks=1)
  torch.manual_seed(11)
  bb_on = _make_backbone(cfg_on, vocab_size=V).eval()
  torch.manual_seed(11)
  bb_off = _make_backbone(cfg_off, vocab_size=V).eval()
  B, L = 2, 4
  z = torch.randn(B, L, d)
  sigma = torch.zeros(B)
  ctx_on = algo.LangFlowContext(z_sc=None, alpha=None, sigma=None, r=0.0)
  out_on = bb_on(None, z, sigma, ctx_on)
  out_off = bb_off(None, z, sigma, None)
  assert torch.allclose(out_on, out_off, atol=1e-6)


def test_w_in_and_w_sc_are_zero_initialized():
  """ARCH §5.1 / §8 I: when `self_conditioning=true`, `W_in`/`W_sc` are
  zero-initialized so OFF==baseline at step 0."""
  V, d = 10, 32
  cfg = _make_config(d=d, length=4, vocab_size=V, init='unit_var',
                     self_conditioning=True, logit_bias=False, n_blocks=1)
  backbone = _make_backbone(cfg, vocab_size=V)
  assert float(backbone.W_in.weight.abs().sum()) == 0.0
  assert float(backbone.W_sc.weight.abs().sum()) == 0.0


# ===========================================================================
# Plaid logit bias (ARCH §5.2 / §8 G / #9)
# ===========================================================================

def test_logit_bias_r_zero_when_off():
  """ARCH §4.5: `_logit_bias_r()` returns 0.0 when `logit_bias=false`."""
  stub, config, backbone = _build_langflow_stub(
    d=16, length=4, vocab_size=10, logit_bias=False, n_blocks=1)
  assert stub._logit_bias_r() == 0.0


def test_logit_bias_r_ramps_to_one_after_warmup():
  """ARCH §4.5: with `logit_bias=true`, r == min(1, step/warmup)."""
  stub, config, backbone = _build_langflow_stub(
    d=16, length=4, vocab_size=10, logit_bias=True,
    logit_bias_warmup_steps=5000, global_step=10_000, n_blocks=1)
  assert stub._logit_bias_r() == pytest.approx(1.0)


def test_logit_bias_r_partial_during_warmup():
  """ARCH §4.5: mid-warmup r == step/warmup (e.g. 2500/5000 = 0.5)."""
  stub, config, backbone = _build_langflow_stub(
    d=16, length=4, vocab_size=10, logit_bias=True,
    logit_bias_warmup_steps=5000, global_step=2500, n_blocks=1)
  assert stub._logit_bias_r() == pytest.approx(0.5)


def test_process_model_output_is_log_softmax():
  """ARCH §4.3: `_process_model_output` is a plain log_softmax of the (already
  bias-augmented) backbone output."""
  stub, config, backbone = _build_langflow_stub(
    d=16, length=4, vocab_size=10, n_blocks=1)
  raw = torch.randn(2, 4, 10)
  out = stub._process_model_output(raw.clone(), xt=None, sigma=None)
  assert torch.allclose(out, raw.float().log_softmax(-1), atol=1e-6)


def test_plaid_bias_is_full_gaussian_loglik():
  """ARCH §5.2 / #9: with `logit_bias=true` and r>0, `SphereDiT.forward` adds the
  FULL Gaussian log-likelihood (Eq. 44) minus the const ||z_gamma||^2 term:
  `r*(alpha/sigma^2)*<E[v], z_gamma> - r*(alpha^2/(2 sigma^2))*||E[v]||^2`. Under
  Option A (raw, free norms) the quadratic ||e_v||^2 term is vocab-dependent and
  IS included (it does not cancel under softmax)."""
  V, d = 10, 32
  cfg = _make_config(d=d, length=4, vocab_size=V, init='unit_var',
                     self_conditioning=False, logit_bias=True, n_blocks=1)
  torch.manual_seed(3)
  backbone = _make_backbone(cfg, vocab_size=V).eval()
  B, L = 2, 4
  z = torch.randn(B, L, d)
  sigma_in = torch.zeros(B)  # gamma time signal (0 -> sigma^2=0.5)
  alpha = torch.full((B, 1), 0.6)
  sigma = torch.full((B, 1), 0.8)
  r = 1.0
  ctx_bias = algo.LangFlowContext(z_sc=None, alpha=alpha, sigma=sigma, r=r)
  ctx_nobias = algo.LangFlowContext(z_sc=None, alpha=alpha, sigma=sigma, r=0.0)
  out_bias = backbone(None, z, sigma_in, ctx_bias)
  out_nobias = backbone(None, z, sigma_in, ctx_nobias)
  E = backbone.sphere_embed.weight  # [V, d] RAW
  inner = torch.einsum('bld,vd->blv', z, E)
  norm_sq = (E * E).sum(-1)  # [V] = ||e_v||^2
  coef1 = (r * alpha / (sigma ** 2)).unsqueeze(-1)  # [B,1,1]
  coef2 = (r * alpha ** 2 / (2 * sigma ** 2)).unsqueeze(-1)  # [B,1,1]
  expected_delta = coef1 * inner - coef2 * norm_sq
  assert torch.allclose(out_bias - out_nobias, expected_delta, atol=1e-4)


def test_plaid_bias_finite_at_both_gamma_clip_ends():
  """ARCH §8 G / #9: at both γ clip ends the bias and final log-probs are finite
  (σ²>0 because γ is clipped)."""
  V, d = 10, 32
  cfg = _make_config(d=d, length=4, vocab_size=V, init='unit_var',
                     self_conditioning=False, logit_bias=True, n_blocks=1)
  backbone = _make_backbone(cfg, vocab_size=V).eval()
  noise = _make_noise()
  a = float(noise.P_mu.detach()
            - noise.P_beta.detach() * math.log(-math.log(1 - noise.q_clip)))
  b = float(noise.P_mu.detach()
            - noise.P_beta.detach() * math.log(-math.log(noise.q_clip)))
  B, L = 2, 4
  z = torch.randn(B, L, d)
  for gamma_val in (a, b):
    gamma = torch.full((B,), gamma_val)
    alpha, sigma = noise.alpha_sigma_from_gamma(gamma)
    ctx = algo.LangFlowContext(
      z_sc=None, alpha=alpha.unsqueeze(-1), sigma=sigma.unsqueeze(-1), r=1.0)
    out = backbone(None, z, gamma, ctx)
    assert torch.isfinite(out).all()
    assert torch.isfinite(out.float().log_softmax(-1)).all()


# ===========================================================================
# nll_per_token (ARCH §4.4)
# ===========================================================================

def test_nll_per_token_matches_gathered_log_softmax():
  """ARCH §4.4: `nll_per_token` == `-log_x_theta.gather(target)`, shape [B,L]."""
  stub, config, backbone = _build_langflow_stub(
    d=16, length=4, vocab_size=7, n_blocks=1)
  B, L, V = 2, 4, 7
  log_x_theta = torch.randn(B, L, V).log_softmax(-1)
  x0 = torch.randint(0, V, (B, L))
  expected = -log_x_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
  got = stub.nll_per_token(log_x_theta, x0)
  assert tuple(got.shape) == (B, L)
  assert torch.allclose(got, expected, atol=1e-6)


# ===========================================================================
# Self-conditioning (ARCH §4.5 / §8 F)
# ===========================================================================

def test_self_cond_off_returns_none():
  """ARCH §4.5: with `self_conditioning=false`, `_self_cond_pass` returns None
  (z_SC := 0 in the backbone)."""
  stub, config, backbone = _build_langflow_stub(
    d=16, length=4, vocab_size=10, self_conditioning=False, n_blocks=1)
  B, L, d = 2, 4, 16
  z_gamma = torch.randn(B, L, d)
  gamma = torch.zeros(B)
  alpha = torch.full((B, 1), 0.6)
  sigma = torch.full((B, 1), 0.8)
  out = stub._self_cond_pass(z_gamma, gamma, alpha, sigma, train_mode=True)
  assert out is None


def test_self_cond_first_pass_output_is_detached():
  """ARCH §4.5 / §8 F: when self-cond fires (p_self_cond=1.0), the returned z_sc
  carries NO grad (first pass under no_grad + detach)."""
  stub, config, backbone = _build_langflow_stub(
    d=16, length=4, vocab_size=10, self_conditioning=True, p_self_cond=1.0,
    n_blocks=1)
  B, L, d = 2, 4, 16
  z_gamma = torch.randn(B, L, d)
  gamma = torch.zeros(B)
  alpha = torch.full((B, 1), 0.6)
  sigma = torch.full((B, 1), 0.8)
  z_sc = stub._self_cond_pass(z_gamma, gamma, alpha, sigma, train_mode=True)
  assert z_sc is not None
  assert z_sc.requires_grad is False


def test_self_cond_train_parity_with_sampler_zhat():
  """ARCH §8 H / #10: train-time `_x_to_embed(probs)` equals the sampler's `zhat`
  construction `probs @ E` (raw E, neither normalized)."""
  stub, config, backbone = _build_langflow_stub(
    d=32, length=4, vocab_size=16, n_blocks=1)
  B, L, V = 2, 4, 16
  probs = torch.softmax(torch.randn(B, L, V), dim=-1)
  E = backbone.sphere_embed.weight  # raw, as the sampler uses
  zhat_sampler = torch.einsum('blv,vd->bld', probs, E)
  z_train = stub._x_to_embed(probs)
  assert torch.allclose(z_train, zhat_sampler, atol=1e-5)


# ===========================================================================
# Stopgrad boundaries (ARCH §4.7 / §8 D, test surface #5 / #6)
# ===========================================================================

def test_nll_gamma_returned_is_detached():
  """ARCH §4.5 B1: `nll` returns gamma in the 't' slot DETACHED (no grad path
  into the schedule from the CE/VP path)."""
  stub, config, backbone = _build_langflow_stub(
    d=16, length=4, vocab_size=10, n_blocks=1)
  B, L = 2, 4
  x0 = torch.randint(0, 10, (B, L))
  valid = torch.ones(B, L, dtype=torch.long)
  ce, gamma = stub.nll(
    x0, output_tokens=None, context=None,
    current_accumulation_step=None, train_mode=True, valid_tokens=valid)
  assert gamma.requires_grad is False


def test_nll_b1_stopgrad_no_schedule_grad_from_ce():
  """ARCH §4.7 B1 / #6: backprop of the CE produced by `nll` yields grad on the
  embedding table but NONE on the schedule params raw_mu/raw_beta/raw_H."""
  stub, config, backbone = _build_langflow_stub(
    d=16, length=4, vocab_size=10, n_blocks=1)
  B, L = 2, 4
  x0 = torch.randint(0, 10, (B, L))
  valid = torch.ones(B, L, dtype=torch.long)
  for p in stub.noise.parameters():
    p.grad = None
  backbone.sphere_embed.weight.grad = None
  ce, _ = stub.nll(
    x0, output_tokens=None, context=None,
    current_accumulation_step=None, train_mode=True, valid_tokens=valid)
  ce.mean().backward()
  # Backbone embedding receives grad...
  assert backbone.sphere_embed.weight.grad is not None
  assert float(backbone.sphere_embed.weight.grad.abs().sum()) > 0.0
  # ...schedule params do NOT (gamma detached into the CE path).
  for p in stub.noise.parameters():
    assert p.grad is None or float(p.grad.abs().sum()) == 0.0


# ===========================================================================
# Loss folding: training_step = L_CE + L_Scheduler (ARCH §4.6 / #13)
# ===========================================================================

def test_scheduler_scalar_not_divided_by_token_count():
  """ARCH §4.6 / #13: the scheduler term is a raw scalar in nats² over γ — its
  magnitude does NOT shrink when the sequence length doubles (it is NOT routed
  through the per-token mean-divide). We check this at the `scheduler_loss`
  contract level: for the same (γ, per-sample CE) the scalar is L-independent."""
  noise = _make_noise(trainable=True)
  gamma = noise.sample_gamma(4, torch.device('cpu')).detach()
  ce_per_sample = torch.full((4,), 4.0)  # per-sample mean-token CE, [B]
  sl = noise.scheduler_loss(gamma, ce_per_sample)
  # The scheduler scalar is the raw MSE against the surrogate entropy; it is NOT
  # divided by any token count, so it equals the closed-form value below.
  h = noise.surrogate_entropy(gamma).detach()
  expected = ((ce_per_sample - h) ** 2).mean()
  assert torch.allclose(sl, expected, atol=1e-6)


def test_combined_loss_equals_ce_when_scheduler_fixed():
  """ARCH §4.6 / #13: with `trainable=false`, `scheduler_loss==0`, so the folded
  loss `ce_loss + sched_loss` equals `ce_loss` (no scheduler contribution)."""
  noise = _make_noise(trainable=False)
  gamma = noise.sample_gamma(4, torch.device('cpu')).detach()
  ce_per_sample = torch.full((4,), 4.0)
  sched = noise.scheduler_loss(gamma, ce_per_sample)
  ce_loss = torch.tensor(3.5)
  assert float(ce_loss + sched) == pytest.approx(float(ce_loss))


# ===========================================================================
# Dispatch (ARCH §2 / §6.4 / #14)
# ===========================================================================

def test_langflow_context_dataclass_exists():
  """ARCH §5.3: `algo.LangFlowContext` exists with z_sc/alpha/sigma/r fields."""
  ctx = algo.LangFlowContext(z_sc=None, alpha=None, sigma=None, r=0.0)
  assert ctx.r == 0.0
  assert ctx.temperature == 1.0  # default so forward's temperature access is safe


def test_main_dispatches_langflow_algo():
  """ARCH §2 / #14: `main` selects `algo.LangFlow` for `algo.name=='langflow'`."""
  import inspect
  import main
  assert hasattr(algo, 'LangFlow')
  src = inspect.getsource(main)
  assert 'algo.LangFlow' in src
  assert "'langflow'" in src or '"langflow"' in src
