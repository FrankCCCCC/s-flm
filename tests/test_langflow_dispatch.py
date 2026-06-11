"""Dispatch + sampler contract tests for LangFlow (ARCH §6 sampler, §6.4
get_sampler dispatch, §2 main dispatch, §8 invariant H, test surface #12 / #14).

These pin the wiring/sampler the implementer must add:
  * `samplers.LangFlowSampler` / `LangFlowState` are importable.
  * `LangFlowSampler.__init__` takes `temperature` / `p_nucleus` / `top_k`.
  * `get_sampler` resolves `predictor=='langflow'` -> `LangFlowSampler`.
  * `init_state` builds a raw-D-space prior `z_0 ~ N(0, sigma_0^2 I)` and zeroed
    self-cond carry; prefix positions are clamped to clean raw embeds.
  * `step` performs the Algorithm-2 Euler-on-γ update; the final step argmaxes.
  * the sampler's `zhat` carry == raw `probs @ E` (train/sample parity, inv H).

`samplers` imports cleanly today (the SFM/HFLM samplers exist); only the
LangFlow-specific names are missing, so the failures are clean
AttributeError / assertion failures, not import-time breakage.
"""
import inspect

import pytest
import torch

import samplers
import noise_schedules
from conftest import REPO_ROOT  # noqa: F401

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Minimal config + model stub (mirrors tests/test_hflm_dispatch.py)
# ---------------------------------------------------------------------------

def _sampler_config(*, predictor='langflow', steps=4):
  import omegaconf
  cfg = {
    'sampler': {
      'predictor': predictor,
      'steps': steps,
      'noise_removal': 'greedy',
      'use_float64': False,
      'p_nucleus': 1.0,
      'top_k': -1,
      'temperature': 1.0,
      'num_sample_batches': 2,
      'num_sample_log': 2,
    },
    'algo': {
      'name': 'langflow',
      'self_conditioning': True,
      'logit_bias': True,
    },
    'noise': {
      'type': 'gumbel',
      'trainable': True,
      'q_clip': 1e-5,
      'H_inf_init': 5.0,
    },
  }
  return omegaconf.OmegaConf.create(cfg)


class _StubBackbone(torch.nn.Module):
  """Backbone exposing what the LangFlow sampler reads: embed_dim, sphere_embed."""

  def __init__(self, d, vocab_size):
    super().__init__()
    self.embed_dim = d
    self.vocab_size = vocab_size
    self.sphere_embed = torch.nn.Embedding(vocab_size, d)
    torch.nn.init.normal_(self.sphere_embed.weight, std=1.0)

  def get_raw_embeddings(self, token_ids):
    return self.sphere_embed(token_ids)


class _StubModel:
  """A light model carrying the attributes LangFlowSampler reads. `noise` is a
  REAL gumbel `UnifInfoSchedule` so the gamma schedule / VP map are genuine."""

  def __init__(self, d=16, vocab_size=8, length=4, steps=4):
    self.num_tokens = length
    self.device = torch.device('cpu')
    self.backbone = _StubBackbone(d, vocab_size)
    self.vocab_size = vocab_size
    self.self_conditioning = True
    self.config = _sampler_config(steps=steps)
    self.noise = noise_schedules.UnifInfoSchedule(
      trainable=True, q_clip=1e-5, H_inf_init=5.0)

  def forward(self, *, x0=None, xt=None, sigma=None, context=None):
    B, L = xt.shape[0], xt.shape[1]
    # Deterministic non-uniform log-probs so argmax is token 0.
    logits = torch.zeros(B, L, self.vocab_size, dtype=torch.float32)
    logits[..., 0] = 5.0
    return logits.log_softmax(-1)


def _build_langflow_sampler():
  return samplers.LangFlowSampler(
    temperature=1.0, p_nucleus=1.0, top_k=-1)


# ---------------------------------------------------------------------------
# Importability + signature (ARCH §6 / §6.4)
# ---------------------------------------------------------------------------

def test_langflow_sampler_importable():
  """ARCH §6: `samplers.LangFlowSampler` exists."""
  assert hasattr(samplers, 'LangFlowSampler')


def test_langflow_state_importable():
  """ARCH §6.1: `samplers.LangFlowState` exists."""
  assert hasattr(samplers, 'LangFlowState')


def test_langflow_sampler_init_signature():
  """ARCH §6.4: `LangFlowSampler.__init__` takes temperature/p_nucleus/top_k
  (no velocity/slerp/invert_time_convention — γ-path Euler needs none)."""
  sig = inspect.signature(samplers.LangFlowSampler.__init__)
  assert 'temperature' in sig.parameters
  assert 'p_nucleus' in sig.parameters
  assert 'top_k' in sig.parameters
  assert 'velocity' not in sig.parameters


# ---------------------------------------------------------------------------
# init_state: raw-D prior z_0 ~ N(0, sigma_0^2 I), zeroed self-cond carry (ARCH §6.2)
# ---------------------------------------------------------------------------

def test_init_state_prior_shape():
  """ARCH §6.2: `init_state` -> `xt` of shape [N, L, d]."""
  sampler = _build_langflow_sampler()
  model = _StubModel(d=16, vocab_size=8, length=4, steps=4)
  state = sampler.init_state(model, 3, num_steps=4)
  assert tuple(state.xt.shape) == (3, model.num_tokens, model.backbone.embed_dim)


def test_init_state_prior_is_raw_gaussian_not_unit_sphere():
  """ARCH §6.2 / §8 E: the prior `z_0 ~ N(0, sigma_0^2 I)` is a raw diagonal
  Gaussian (NOT projected to the unit sphere); rows are not unit-norm."""
  sampler = _build_langflow_sampler()
  model = _StubModel(d=64, vocab_size=8, length=4, steps=4)
  state = sampler.init_state(model, 8, num_steps=4)
  norms = state.xt.norm(p=2, dim=-1)
  assert (norms - 1.0).abs().min() > 0.1


def test_init_state_prior_std_matches_sigma0():
  """ARCH §6.2: `z_0` per-coord std ≈ sigma_0 (the noisiest end of the γ clip)."""
  sampler = _build_langflow_sampler()
  model = _StubModel(d=64, vocab_size=8, length=8, steps=16)
  state = sampler.init_state(model, 64, num_steps=16)
  sigma0 = float(state.sigmas[0])
  assert float(state.xt.std()) == pytest.approx(sigma0, rel=0.2)


def test_init_state_self_cond_carry_starts_zero():
  """ARCH §6.2: the self-cond carry `z_sc` is zeros at k=0."""
  sampler = _build_langflow_sampler()
  model = _StubModel(d=16, vocab_size=8, length=4, steps=4)
  state = sampler.init_state(model, 3, num_steps=4)
  assert torch.count_nonzero(state.z_sc) == 0


def test_init_state_gamma_schedule_length():
  """ARCH §6.2: the precomputed γ schedule has N+1 entries."""
  sampler = _build_langflow_sampler()
  model = _StubModel(d=16, vocab_size=8, length=4, steps=4)
  state = sampler.init_state(model, 3, num_steps=4)
  assert tuple(state.gammas.shape) == (5,)


def test_init_state_clamps_prefix_to_clean_raw_embeds():
  """Prefix conditioning (sudoku): `init_state` clamps prefix positions of the
  prior to the CLEAN raw embeddings (matches q_xt's valid_tokens=0 = clean)."""
  sampler = _build_langflow_sampler()
  model = _StubModel(d=16, vocab_size=8, length=4, steps=4)
  prefix = torch.tensor([[3, 5], [1, 2], [7, 4]], dtype=torch.long)
  lengths = torch.full((3,), 2, dtype=torch.long)
  state = sampler.init_state(model, 3, num_steps=4,
                             prefix_tokens=prefix, prefix_lengths=lengths)
  clean = model.backbone.get_raw_embeddings(prefix).to(state.xt.dtype)
  assert torch.allclose(state.xt[:, :2], clean)
  # Suffix positions remain the Gaussian prior (not clamped).
  assert not torch.allclose(state.xt[:, 2:], clean)


def test_step_keeps_prefix_clamped_and_decodes_prefix_tokens():
  """Each Euler step re-clamps the prefix to clean embeds; the final argmax
  decode returns the TRUE prefix tokens at prefix positions (stub argmax
  is token 0 everywhere else)."""
  sampler = _build_langflow_sampler()
  model = _StubModel(d=16, vocab_size=8, length=4, steps=4)
  prefix = torch.tensor([[3, 5], [1, 2], [7, 4]], dtype=torch.long)
  lengths = torch.full((3,), 2, dtype=torch.long)
  state = sampler.init_state(model, 3, num_steps=4,
                             prefix_tokens=prefix, prefix_lengths=lengths)
  clean = model.backbone.get_raw_embeddings(prefix).to(state.xt.dtype)
  while not state.done:
    state = sampler.step(model, state)
    if not state.done:  # mid-integration: prefix stays clamped to clean embeds
      assert torch.allclose(state.xt[:, :2], clean)
  assert state.xt.shape == (3, 4)
  assert torch.equal(state.xt[:, :2], prefix)   # decoded prefix == true tokens
  assert (state.xt[:, 2:] == 0).all()           # stub argmax is token 0


# ---------------------------------------------------------------------------
# step: Euler-on-γ update closed form (ARCH §6.3 / test surface #12)
# ---------------------------------------------------------------------------

def test_step_euler_update_matches_closed_form():
  """ARCH §6.3 / #12: one non-last `step` matches the Algorithm-2 update
  `z_{k+1} = s_next * (z_k/sigma_k + (a_next/s_next - alpha_k/sigma_k) zhat)`
  for the deterministic `zhat = probs @ E`."""
  sampler = _build_langflow_sampler()
  model = _StubModel(d=8, vocab_size=4, length=3, steps=4)
  N = 2
  state = sampler.init_state(model, N, num_steps=4)
  k = state.step_idx
  alpha_k, sigma_k = state.alphas[k], state.sigmas[k]
  a_next, s_next = state.alphas[k + 1], state.sigmas[k + 1]
  z_k = state.xt.clone()
  # Reproduce the model's deterministic log_p -> zhat.
  B, L = z_k.shape[0], z_k.shape[1]
  log_p = model.forward(xt=z_k)
  E = model.backbone.sphere_embed.weight
  zhat = torch.einsum('blv,vd->bld', log_p.exp(), E)
  expected = s_next * (z_k / sigma_k
                       + (a_next / s_next - alpha_k / sigma_k) * zhat)

  state = sampler.step(model, state)
  assert torch.allclose(state.xt, expected, atol=1e-4)


def test_step_zhat_carry_matches_raw_probs_matmul_E():
  """ARCH §8 H / #10: the sampler's `zhat` self-cond carry == raw `probs @ E`
  (matches `LangFlow._x_to_embed`), neither side normalized."""
  sampler = _build_langflow_sampler()
  model = _StubModel(d=16, vocab_size=6, length=4, steps=4)
  N = 2
  state = sampler.init_state(model, N, num_steps=4)
  log_p = model.forward(xt=state.xt)
  E = model.backbone.sphere_embed.weight
  expected_zhat = torch.einsum('blv,vd->bld', log_p.exp(), E)
  state = sampler.step(model, state)  # non-last -> sets z_sc carry
  assert torch.allclose(state.z_sc, expected_zhat, atol=1e-4)


def test_step_advances_index_and_counts_nfe():
  """ARCH §6.3: a non-last `step` advances `step_idx` and increments `nfe`."""
  sampler = _build_langflow_sampler()
  model = _StubModel(d=16, vocab_size=6, length=4, steps=4)
  state = sampler.init_state(model, 2, num_steps=4)
  state = sampler.step(model, state)
  assert state.step_idx == 1
  assert state.nfe == 1


def test_last_step_decodes_argmax_tokens():
  """ARCH §6.3 / #12: Algorithm 2 does N Euler updates + 1 final decode at the
  cleanest point (z_N, gamma_N), so `num_steps=N` ⇒ N+1 model evals (nfe==N+1)
  and the final `step` returns int tokens [B, L] via argmax with `done=True`.
  Fixed logits put all mass on token 0."""
  num_steps = 2
  sampler = _build_langflow_sampler()
  model = _StubModel(d=16, vocab_size=8, length=4, steps=num_steps)
  N = 3
  state = sampler.init_state(model, N, num_steps=num_steps)
  for _ in range(num_steps):
    state = sampler.step(model, state)  # N Euler updates (non-last)
    assert not state.done
  state = sampler.step(model, state)    # final decode at (z_N, gamma_N)
  assert state.done
  assert state.nfe == num_steps + 1     # N Euler updates + 1 final decode
  assert state.step_idx == num_steps    # decode at the last schedule entry
  assert tuple(state.xt.shape) == (N, model.num_tokens)
  assert state.xt.dtype in (torch.int64, torch.long)
  assert torch.equal(state.xt, torch.zeros_like(state.xt))


def test_full_trajectory_stays_finite():
  """ARCH §6: the whole Euler trajectory stays finite; the final (N+1'th) step
  decodes ids at (z_N, gamma_N)."""
  num_steps = 4
  sampler = _build_langflow_sampler()
  model = _StubModel(d=16, vocab_size=8, length=4, steps=num_steps)
  N = 3
  state = sampler.init_state(model, N, num_steps=num_steps)
  for step_idx in range(num_steps + 1):  # N Euler updates + 1 final decode
    state = sampler.step(model, state)
    if step_idx < num_steps:
      assert torch.isfinite(state.xt).all()
    else:
      assert state.done
      assert tuple(state.xt.shape) == (N, model.num_tokens)


# ---------------------------------------------------------------------------
# get_sampler dispatch (ARCH §6.4 / #14)
# ---------------------------------------------------------------------------

def test_get_sampler_resolves_langflow():
  """ARCH §6.4 / #14: `get_sampler` returns a `LangFlowSampler` for
  `predictor=='langflow'`."""
  config = _sampler_config(predictor='langflow')
  sampler = samplers.get_sampler(config)
  assert isinstance(sampler, samplers.LangFlowSampler)


def test_get_sampler_langflow_sets_temperature():
  """ARCH §6.4: dispatch threads `sampler.temperature` into the sampler."""
  config = _sampler_config(predictor='langflow')
  sampler = samplers.get_sampler(config)
  assert sampler.temperature == pytest.approx(1.0)
