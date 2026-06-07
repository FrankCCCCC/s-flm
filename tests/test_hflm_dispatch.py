"""Dispatch / wiring contract tests for HFLM (ARCH §9, test contracts
#8, #10-#15).

These pin the wiring the implementer must add:
  * `trainer_base` maps `model.type=='hyperbolic-dit'` -> `HyperbolicDiT`.
  * `main` maps `algo.name=='hflm'` -> `algo.HFLM`.
  * `samplers.get_sampler` resolves `predictor=='hflm'` -> `HFLMSampler` with
    `prior_cov` / `rho_max` set.
  * `samplers.HFLMSampler` is importable, `init_state` yields a Poincaré prior
    (‖xt‖<1), `step` keeps ‖xt‖<1 and moves xt, last `step` decodes argmax.

`import algo` / `import main` currently fail with the WIP SyntaxError in
`algo.HFLM._hyeprbolic_geodesic`; those import-time failures are the accepted
initial red state and resolve once the implementer finishes.
"""
import inspect

import pytest
import torch

import samplers
from geo_bridge import GeoUtils
from conftest import REPO_ROOT  # noqa: F401

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Minimal config + model stub
# ---------------------------------------------------------------------------

def _sampler_config(*, predictor='hflm', prior_cov=0.25, rho_max=12.0,
                    invert_time_convention=False, steps=4):
  import omegaconf
  cfg = {
    'sampler': {
      'predictor': predictor,
      'steps': steps,
      'noise_removal': 'greedy',
      'use_float64': True,
      'velocity': 'exact',
      'p_nucleus': 1.0,
      'top_k': -1,
      'top_k_velocity': 1,
      'temperature': 1.0,
      'num_sample_batches': 2,
      'num_sample_log': 2,
    },
    'algo': {
      'name': 'hflm',
      'slerp_precision': 'float64',
      'eps': 1e-6,
      'invert_time_convention': invert_time_convention,
      'prior_cov': prior_cov,
      'rho_max': rho_max,
    },
  }
  return omegaconf.OmegaConf.create(cfg)


class _StubBackbone(torch.nn.Module):
  """Backbone exposing only what the sampler reads: embed_dim, sphere_embed,
  get_hyperbolic_polar_embeddings, and a fixed-logits forward."""

  def __init__(self, d, vocab_size):
    super().__init__()
    self.embed_dim = d
    self.vocab_size = vocab_size
    self.sphere_embed = torch.nn.Embedding(vocab_size, d)
    torch.nn.init.normal_(self.sphere_embed.weight, std=0.3)

  def get_hyperbolic_polar_embeddings(self, token_ids):
    emb = self.sphere_embed(token_ids)
    rhos = emb.norm(p=2, dim=-1, keepdim=True)
    thetas = emb / rhos.clamp_min(torch.finfo(emb.dtype).tiny)
    return rhos, thetas


class _StubModel:
  """A light model carrying the attributes HFLMSampler.{init_state,step} read."""

  def __init__(self, d=16, vocab_size=8, length=4, steps=4):
    self.num_tokens = length
    self.device = torch.device('cpu')
    self.backbone = _StubBackbone(d, vocab_size)
    self.vocab_size = vocab_size
    self.config = _sampler_config(steps=steps)

  def noise(self, t):
    # Return (dalpha_t, alpha_t); matches the real LogLinear schedule where
    # alpha = 1 - t, so for invert_time_convention=false the geodesic step
    # size dt lands in (0, 1) at every step.
    t = torch.as_tensor(t)
    alpha_t = (1.0 - t).clamp(1e-4, 1.0)
    return -torch.ones_like(alpha_t), alpha_t

  def _sigma_from_alphat(self, alpha_t):
    return -torch.log(torch.as_tensor(alpha_t).clamp_min(1e-6))

  def forward(self, *, xt=None, sigma=None, context=None):
    B, L = xt.shape[0], xt.shape[1]
    # Deterministic non-uniform logits so argmax / velocity are well-defined.
    logits = torch.zeros(B, L, self.vocab_size, dtype=torch.float64)
    logits[..., 0] = 1.0
    return logits.log_softmax(-1)


def _build_hflm_sampler():
  return samplers.HFLMSampler(
    noise_removal='greedy', velocity='exact', use_float64=True,
    slerp_float64=True, eps=1e-6, temperature=1.0, p_nucleus=1.0,
    top_k=-1, top_k_velocity=1, invert_time_convention=False,
    prior_cov=0.25, rho_max=12.0)


# ---------------------------------------------------------------------------
# Sampler importability + signature (test contract #8)
# ---------------------------------------------------------------------------

def test_hflm_sampler_importable():
  """ARCH §3.3: `samplers.HFLMSampler` exists."""
  assert hasattr(samplers, 'HFLMSampler')


def test_hflm_sampler_init_signature_has_prior_cov_and_rho_max():
  """ARCH §3.3: `HFLMSampler.__init__` takes `prior_cov` and `rho_max`
  (the two hyperbolic-specific fields beyond the sphere sampler).
  """
  sig = inspect.signature(samplers.HFLMSampler.__init__)
  assert 'prior_cov' in sig.parameters
  assert 'rho_max' in sig.parameters


# ---------------------------------------------------------------------------
# Sampler init_state -> Poincaré prior (test contract #10)
# ---------------------------------------------------------------------------

def test_hflm_sampler_init_state_poincare_prior():
  """Test contract #10: `init_state` -> `xt` with ‖xt‖<1 and shape `[N,L,d]`."""
  sampler = _build_hflm_sampler()
  model = _StubModel(d=16, vocab_size=8, length=4, steps=4)
  N = 3
  state = sampler.init_state(model, N, num_steps=4)
  assert tuple(state.xt.shape) == (N, model.num_tokens, model.backbone.embed_dim)
  assert float(state.xt.norm(p=2, dim=-1).max()) < 1.0


# ---------------------------------------------------------------------------
# Sampler step (non-last) keeps ‖xt‖<1 and moves xt (test contract #11)
# ---------------------------------------------------------------------------

def test_hflm_sampler_step_keeps_in_ball_and_moves():
  """Test contract #11 (ARCH §7): every non-last `step` keeps ‖xt‖<1, stays
  finite, preserves shape and moves xt; run the FULL num_steps trajectory."""
  num_steps = 4
  sampler = _build_hflm_sampler()
  model = _StubModel(d=16, vocab_size=8, length=4, steps=num_steps)
  N = 3
  state = sampler.init_state(model, N, num_steps=num_steps)
  cont_shape = tuple(state.xt.shape)
  for step_idx in range(num_steps):
    before = state.xt.clone()
    state = sampler.step(model, state)
    if step_idx < num_steps - 1:
      # Non-last steps stay in the Poincaré ball, finite, same shape, moving.
      assert tuple(state.xt.shape) == cont_shape
      assert torch.isfinite(state.xt).all()
      assert float(state.xt.norm(p=2, dim=-1).max()) < 1.0
      assert not torch.allclose(state.xt, before)
    else:
      # Last step decodes to int tokens [N, L].
      assert state.done
      assert tuple(state.xt.shape) == (N, model.num_tokens)


# ---------------------------------------------------------------------------
# Sampler step (last) -> int tokens, done=True (test contract #12)
# ---------------------------------------------------------------------------

def test_hflm_sampler_last_step_decodes_tokens():
  """Test contract #12: the last `step` returns int tokens `[N,L]` (argmax
  posterior) and sets `state.done=True`.
  """
  sampler = _build_hflm_sampler()
  model = _StubModel(d=16, vocab_size=8, length=4, steps=2)
  N = 3
  state = sampler.init_state(model, N, num_steps=2)
  state = sampler.step(model, state)   # step 0 (non-last)
  state = sampler.step(model, state)   # step 1 (last)
  assert state.done
  assert tuple(state.xt.shape) == (N, model.num_tokens)
  assert state.xt.dtype in (torch.int64, torch.long)
  # Fixed logits put all mass on token 0 -> argmax decode is token 0.
  assert torch.equal(state.xt, torch.zeros_like(state.xt))


# ---------------------------------------------------------------------------
# get_sampler resolves 'hflm' (test contract #15)
# ---------------------------------------------------------------------------

def test_get_sampler_resolves_hflm():
  """Test contract #15: `samplers.get_sampler` returns an `HFLMSampler` for
  `predictor='hflm'` with `prior_cov` / `rho_max` set from config.algo.
  """
  config = _sampler_config(predictor='hflm', prior_cov=0.25, rho_max=12.0)
  sampler = samplers.get_sampler(config)
  assert isinstance(sampler, samplers.HFLMSampler)
  assert sampler.prior_cov == pytest.approx(0.25)
  assert sampler.rho_max == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# trainer_base maps model.type='hyperbolic-dit' -> HyperbolicDiT (contract #13)
# ---------------------------------------------------------------------------

def test_trainer_base_dispatches_hyperbolic_dit_branch():
  """Test contract #13: `trainer_base` constructs `HyperbolicDiT` for
  `model.type='hyperbolic-dit'`.

  Constructing the full Lightning module pulls in EMA / metrics (HF download),
  so we assert the dispatch BRANCH is reachable: the backbone class exists and
  is referenced by name in `trainer_base`'s source. (After implementation the
  branch in `trainer_base.py:69-85` constructs it.)
  """
  import trainer_base
  import models
  assert hasattr(models, 'hyperbolic_dit')
  assert hasattr(models.hyperbolic_dit, 'HyperbolicDiT')
  src = inspect.getsource(trainer_base)
  assert "hyperbolic-dit" in src
  assert "HyperbolicDiT" in src


# ---------------------------------------------------------------------------
# main maps algo.name='hflm' -> algo.HFLM (test contract #14)
# ---------------------------------------------------------------------------

def test_main_dispatches_hflm_algo():
  """Test contract #14: `main` selects `algo.HFLM` for `algo.name='hflm'`.

  `main.main` is hydra-decorated; we assert the dispatch branch is wired in
  source and that `algo.HFLM` is reachable. (Importing `main`/`algo` currently
  fails on the WIP SyntaxError -- the expected initial red state.)
  """
  import algo
  import main
  assert hasattr(algo, 'HFLM')
  src = inspect.getsource(main)
  assert "algo.HFLM" in src
  assert "'hflm'" in src or '"hflm"' in src
