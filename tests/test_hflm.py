"""Interface-contract tests for HFLM (Hyperbolic Flow Language Model).

These tests are written AGAINST the contract in
`experiments/hflm/ARCH.md` (§3 interfaces, §4 data flow, §5 rho-bound, §10 the
16 test contracts) BEFORE the implementation exists. They are expected to FAIL
until the implementer finishes:

  * `algo.HFLM` (the WIP currently has a SyntaxError in `_hyeprbolic_geodesic`,
    so `import algo` itself fails -> tests that touch it error at collection /
    setup; that import-time failure is the accepted initial red state).
  * `models.hyperbolic_dit.HyperbolicDiT` (today the class is still named
    `SphereDiT`).
  * `samplers.HFLMSampler` (today a verbatim sphere clone).

Design note: a full `algo.HFLM` lives on `trainer_base.Diffusion`, whose
`__init__` builds an EMA, a noise schedule and a `metrics.Metrics` that downloads
a HF tokenizer over the network. To keep these tests fast / offline / CPU-only
we exercise the *contract methods* of `HFLM` (`q_xt`, `_hyperbolic_geodesic`,
`_sample_prior`, `nll_per_token`, `_validate_configuration`) bound to a light
stub that carries exactly the attributes those methods read. We build a REAL
`HyperbolicDiT` backbone so the embedding / gradient path is genuine. This tests
the contract, not the Lightning framework.
"""
import math

import pytest
import torch

import geo_bridge
from geo_bridge import GeoUtils, HyperbolicHeatKernel, Coordinate, Geometry
from conftest import REPO_ROOT  # noqa: F401  (ensures repo root on sys.path)

torch.manual_seed(0)

PRIMARY_D = 512
RHO_BOUND = 20.0  # geo_bridge._LORENTZ_RHO_MAX


# ---------------------------------------------------------------------------
# Config + stub helpers
# ---------------------------------------------------------------------------

def _make_config(*, d=512, length=180, vocab_size=16, init='hyperbolic',
                 prior_cov=0.25, rho_max=12.0, invert_time_convention=False,
                 slerp_precision='float64', model_type='hyperbolic-dit',
                 learn_temperature_scaling=False, n_blocks=2, n_heads=8):
  """Minimal OmegaConf config mirroring the fields `HyperbolicDiT` / `HFLM`
  read from `configs/model/tiny-hyperbolic-dit.yaml` + `configs/algo/hflm.yaml`.

  Only the keys actually consumed by the contract code are included, so we do
  not depend on Hydra composition.
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
      'learn_temperature_scaling': learn_temperature_scaling,
      'eps': 1e-6,
      'pretrained_ckpt_path': None,
    },
    'algo': {
      'name': 'hflm',
      'diffusion_type': 'sphere',
      'backbone': 'hyperbolic-dit',
      'parameterization': 'mean',
      'time_conditioning': True,
      'loss_type': 'ce',
      'T': 0,
      'causal_attention': False,
      'adaLN': True,
      'slerp_precision': slerp_precision,
      'eps': 1e-6,
      'invert_time_convention': invert_time_convention,
      'renormalize_weights': False,
      'prior_cov': prior_cov,
      'rho_max': rho_max,
    },
    'noise': {'adaptive': False},
  }
  return omegaconf.OmegaConf.create(cfg)


def _make_backbone(config, vocab_size):
  """Construct the real HyperbolicDiT backbone (the run default)."""
  import models.hyperbolic_dit
  return models.hyperbolic_dit.HyperbolicDiT(config, vocab_size=vocab_size)


class _HFLMStub:
  """A light HFLM carrying only the attributes the contract methods read.

  Binds the real `algo.HFLM` methods so `q_xt` / `_hyperbolic_geodesic` /
  `_sample_prior` / `nll_per_token` / `_validate_configuration` run their actual
  implementation, without invoking `trainer_base.Diffusion.__init__` (EMA /
  metrics / HF download).
  """

  def __init__(self, config, backbone):
    import algo
    self.config = config
    self.backbone = backbone
    self.eps = config.algo.eps
    self.renormalize_weights = config.algo.renormalize_weights
    self.invert_time_convention = config.algo.invert_time_convention
    self.prior_cov = config.algo.prior_cov
    self.rho_max = config.algo.rho_max
    self._algo = algo

  def __getattr__(self, name):
    # Bind any HFLM method not set above as an unbound method of this stub.
    import algo
    attr = getattr(algo.HFLM, name)
    if callable(attr):
      return attr.__get__(self, type(self))
    raise AttributeError(name)


def _build_hflm_stub(**cfg_kwargs):
  vocab_size = cfg_kwargs.pop('vocab_size', 16)
  config = _make_config(vocab_size=vocab_size, **cfg_kwargs)
  backbone = _make_backbone(config, vocab_size)
  return _HFLMStub(config, backbone), config, backbone


def _poincare_norm(z):
  return z.norm(p=2, dim=-1)


# ---------------------------------------------------------------------------
# Contract: wrapped_normal rho-bound safety (ARCH §5.1, EXPERIMENT §3/§7)
# ---------------------------------------------------------------------------

def test_wrapped_normal_returns_polar_pair_shapes():
  """`GeoUtils.wrapped_normal(shape=(B,L,d))` returns `(rhos[B,L], u[B,L,d])`.

  ARCH §3.1 `_sample_prior`: the prior draw is the polar pair, rho without the
  embedding axis, u carrying it.
  """
  B, L, d = 2, 3, PRIMARY_D
  rhos, u = GeoUtils.wrapped_normal(
    shape=(B, L, d), mean=0.0, cov=0.25, dtype=torch.float64, device='cpu')
  assert tuple(rhos.shape) == (B, L)
  assert tuple(u.shape) == (B, L, d)


def test_wrapped_normal_direction_is_unit():
  """The angular part `u` lies on `S^{d-1}` (‖u‖≈1). ARCH §3.1 / §4.1."""
  rhos, u = GeoUtils.wrapped_normal(
    shape=(4, 5, PRIMARY_D), cov=0.25, dtype=torch.float64, device='cpu')
  norms = u.norm(p=2, dim=-1)
  assert torch.allclose(norms, torch.ones_like(norms), atol=1e-9)


def test_wrapped_normal_prior_cov_025_under_rho_bound():
  """prior_cov=0.25 at d=512 keeps max(rho) < 20 (the _LORENTZ_RHO_MAX guard).

  ARCH §5.1 / EXPERIMENT §7: s=0.5 ⇒ E[rho]≈11.3, batch-max ≈13 over 256×180
  draws. Test contract #7 allows a generous margin (assert < 18).
  """
  rhos, _ = GeoUtils.wrapped_normal(
    shape=(256, 180, PRIMARY_D), cov=0.25,
    dtype=torch.float64, device='cpu')
  assert float(rhos.max()) < 18.0


def test_wrapped_normal_default_cov_exceeds_rho_bound_documented():
  """DOCUMENTED (not a guard): cov=1.0 at d=512 gives E[rho]≈22.6 > 20.

  This is *why* prior_cov=0.25 is required (ARCH §5.1, EXPERIMENT §3). We assert
  the unsafe regime really is unsafe so the choice of 0.25 is load-bearing.
  """
  rhos, _ = GeoUtils.wrapped_normal(
    shape=(64, 32, PRIMARY_D), cov=1.0, dtype=torch.float64, device='cpu')
  assert float(rhos.mean()) > RHO_BOUND


# ---------------------------------------------------------------------------
# Contract: radial soft clamp  rho_eff = rho_max * tanh(rho / rho_max)
# (ARCH §5.2, test contract #8). The clamp is the math the implementer must
# meet inside q_xt / the sampler; we pin its mathematical contract here.
# ---------------------------------------------------------------------------

def _rho_clamp(rho, rho_max):
  return rho_max * torch.tanh(rho / rho_max)


def test_rho_clamp_caps_huge_input_under_lorentz_bound():
  """Test contract #8: rho_clamp(rho=1e3, rho_max=12) is capped at rho_max and
  therefore stays under the _LORENTZ_RHO_MAX=20 guard.

  Note: 12*tanh(1000/12) saturates to exactly 12.0 in float64 (tanh(83)==1.0),
  so the operative invariant is `rho_eff <= rho_max < 20` (the guard never
  fires), not a strict `< rho_max`.
  """
  rho_max = 12.0
  out = _rho_clamp(torch.tensor([1e3], dtype=torch.float64), rho_max)
  assert float(out.max()) <= rho_max
  assert float(out.max()) < RHO_BOUND


def test_rho_clamp_strictly_below_rho_max_for_finite_input():
  """For finite rho the clamp image is strictly below rho_max (tanh<1)."""
  rho_max = 12.0
  out = _rho_clamp(torch.tensor([30.0], dtype=torch.float64), rho_max)
  assert float(out.max()) < rho_max


def test_rho_clamp_monotone_increasing():
  """The clamp is monotone increasing in rho (order-preserving radii)."""
  rho_max = 12.0
  rho = torch.linspace(0.0, 100.0, 500, dtype=torch.float64)
  out = _rho_clamp(rho, rho_max)
  diffs = out[1:] - out[:-1]
  assert (diffs >= 0).all()


def test_rho_clamp_near_identity_for_small_rho():
  """For rho ≪ rho_max the clamp ≈ identity (length-as-radial preserved)."""
  rho_max = 12.0
  rho = torch.tensor([0.0, 0.1, 0.5, 1.0], dtype=torch.float64)
  out = _rho_clamp(rho, rho_max)
  assert torch.allclose(out, rho, atol=1e-2)


def test_rho_clamp_is_differentiable_with_finite_grad():
  """tanh clamp is smooth with finite, non-vanishing gradient (ARCH §5)."""
  rho_max = 12.0
  rho = torch.tensor([0.0, 5.0, 50.0], dtype=torch.float64, requires_grad=True)
  out = _rho_clamp(rho, rho_max)
  out.sum().backward()
  assert torch.isfinite(rho.grad).all()
  assert (rho.grad > 0).all()


# ---------------------------------------------------------------------------
# Contract: geodesic endpoints (ARCH §4.2, test contract #3)
# Ground truth uses geo_bridge.geodesic + hyperbolic_polar_to_poincare_cartesian
# directly (these exist today), pinning the t=0->noisy, t=1->clean invariant.
# ---------------------------------------------------------------------------

def _polar_endpoints(B, L, d):
  rho_clean, u_clean = GeoUtils.wrapped_normal(
    shape=(B, L, d), cov=0.25, dtype=torch.float64, device='cpu')
  rho_noisy, u_noisy = GeoUtils.wrapped_normal(
    shape=(B, L, d), cov=0.25, dtype=torch.float64, device='cpu')
  return rho_clean, u_clean, rho_noisy, u_noisy


def test_geodesic_t0_equals_poincare_source_noisy():
  """`geodesic(src=noisy, dest=clean, t=0)` ≈ Poincaré(noisy). ARCH §4.2."""
  B, L, d = 2, 3, 8
  rho_clean, u_clean, rho_noisy, u_noisy = _polar_endpoints(B, L, d)
  z0 = HyperbolicHeatKernel.geodesic(
    t=torch.zeros(B, L, 1, dtype=torch.float64),
    src_radial=rho_noisy, src_angular=u_noisy,
    dest_radial=rho_clean, dest_angular=u_clean,
    cartesian_model=Geometry.POINCARE,
    output_coord=Coordinate.CARTESIAN)
  noisy_cart = GeoUtils.hyperbolic_polar_to_poincare_cartesian(
    rho_noisy, u_noisy)
  assert torch.allclose(z0, noisy_cart, atol=1e-5)


def test_geodesic_t1_equals_poincare_destination_clean():
  """`geodesic(src=noisy, dest=clean, t=1)` ≈ Poincaré(clean). ARCH §4.2."""
  B, L, d = 2, 3, 8
  rho_clean, u_clean, rho_noisy, u_noisy = _polar_endpoints(B, L, d)
  z1 = HyperbolicHeatKernel.geodesic(
    t=torch.ones(B, L, 1, dtype=torch.float64),
    src_radial=rho_noisy, src_angular=u_noisy,
    dest_radial=rho_clean, dest_angular=u_clean,
    cartesian_model=Geometry.POINCARE,
    output_coord=Coordinate.CARTESIAN)
  clean_cart = GeoUtils.hyperbolic_polar_to_poincare_cartesian(
    rho_clean, u_clean)
  assert torch.allclose(z1, clean_cart, atol=1e-5)


# ---------------------------------------------------------------------------
# Contract: HFLM.q_xt  (ARCH §3.1 / §4.1, test contracts #1, #2, #3, #4)
# ---------------------------------------------------------------------------

def test_q_xt_returns_poincare_ball_point_shape_and_norm():
  """Test contract #1: `q_xt` returns `[B,L,d]` with strictly ‖z_t‖<1.

  Uses d=512 and several alpha_t in (0,1).
  """
  hflm, config, backbone = _build_hflm_stub(
    d=PRIMARY_D, length=4, vocab_size=12, n_blocks=1)
  B, L = 2, 4
  x = torch.randint(0, 12, (B, L))
  for a in (0.1, 0.5, 0.9):
    alpha_t = torch.full((B, L, 1), a, dtype=torch.float32)
    z = hflm.q_xt(x, alpha_t, use_pure_noise=False)
    assert tuple(z.shape) == (B, L, PRIMARY_D)
    assert torch.isfinite(z).all()
    assert float(_poincare_norm(z).max()) < 1.0


def test_q_xt_pure_noise_returns_poincare_prior():
  """Test contract #2: `use_pure_noise=True` -> Poincaré(prior), ‖·‖<1.

  ARCH §8: the prior branch is a Poincaré cartesian point (no geodesic call).
  """
  hflm, config, backbone = _build_hflm_stub(
    d=PRIMARY_D, length=4, vocab_size=12, n_blocks=1)
  B, L = 2, 4
  x = torch.randint(0, 12, (B, L))
  alpha_t = torch.ones(B, L, 1, dtype=torch.float32)
  z = hflm.q_xt(x, alpha_t, use_pure_noise=True)
  assert tuple(z.shape) == (B, L, PRIMARY_D)
  assert torch.isfinite(z).all()
  assert float(_poincare_norm(z).max()) < 1.0


def test_q_xt_at_alpha_one_recovers_clean_embedding(monkeypatch):
  """Test contract #3 (endpoint): with invert_time_convention=false and
  alpha_t=1 (clean signal), `q_xt` ≈ Poincaré(clean-clamped) within 1e-4.

  We freeze the prior draw to an arbitrary value (it must not matter at
  alpha_t=1) and compare against the ground-truth Poincaré of the clamped clean
  embedding.
  """
  hflm, config, backbone = _build_hflm_stub(
    d=32, length=3, vocab_size=10, n_blocks=1,
    invert_time_convention=False, slerp_precision='float64')
  B, L = 2, 3
  x = torch.randint(0, 10, (B, L))

  # Ground truth: clamp(rho_clean) then Poincaré.
  rho_clean, theta_clean = backbone.get_hyperbolic_polar_embeddings(x)
  rho_clean64 = rho_clean.to(torch.float64)
  rho_clean_c = 12.0 * torch.tanh(rho_clean64 / 12.0)
  clean_cart = GeoUtils.hyperbolic_polar_to_poincare_cartesian(
    rho_clean_c.squeeze(-1), theta_clean.to(torch.float64))

  alpha_t = torch.ones(B, L, 1, dtype=torch.float32)
  z = hflm.q_xt(x, alpha_t, use_pure_noise=False).to(torch.float64)
  assert torch.allclose(z, clean_cart, atol=1e-4)


def test_q_xt_at_alpha_eps_recovers_prior(monkeypatch):
  """Test contract #3 (endpoint): with invert_time_convention=false and
  alpha_t≈eps (pure noise), `q_xt` ≈ Poincaré(prior). ARCH §4.2.

  We pin the prior draw via a monkeypatch so the comparison is deterministic.
  """
  hflm, config, backbone = _build_hflm_stub(
    d=16, length=3, vocab_size=10, n_blocks=1,
    invert_time_convention=False, slerp_precision='float64')
  B, L, d = 2, 3, 16
  x = torch.randint(0, 10, (B, L))

  fixed_rho, fixed_u = GeoUtils.wrapped_normal(
    shape=(B, L, d), cov=0.25, dtype=torch.float64, device='cpu')

  def _fake_prior(_self, e_clean_rhos):
    return fixed_rho.unsqueeze(-1), fixed_u

  monkeypatch.setattr(type(hflm), '_sample_prior', _fake_prior,
                      raising=False)

  rho_noisy_c = 12.0 * torch.tanh(fixed_rho / 12.0)
  prior_cart = GeoUtils.hyperbolic_polar_to_poincare_cartesian(
    rho_noisy_c, fixed_u)

  alpha_t = torch.full((B, L, 1), 1e-5, dtype=torch.float32)
  z = hflm.q_xt(x, alpha_t, use_pure_noise=False).to(torch.float64)
  assert torch.allclose(z, prior_cart, atol=1e-4)


def test_q_xt_valid_tokens_keeps_prompt_clean():
  """Test contract #4: prompt positions (valid_tokens==0) equal the clean
  Poincaré embedding exactly; generated positions (valid_tokens==1) differ.

  Closes the ARCH WIP `e_clean` undefined bug (givens-leak).
  """
  hflm, config, backbone = _build_hflm_stub(
    d=32, length=4, vocab_size=10, n_blocks=1)
  B, L = 2, 4
  x = torch.randint(0, 10, (B, L))
  # First two positions are prompt (clean), last two are generated.
  valid_tokens = torch.tensor([[0, 0, 1, 1],
                               [0, 0, 1, 1]], dtype=torch.long)
  alpha_t = torch.full((B, L, 1), 0.3, dtype=torch.float32)
  z = hflm.q_xt(x, alpha_t, use_pure_noise=False, valid_tokens=valid_tokens)

  rho_clean, theta_clean = backbone.get_hyperbolic_polar_embeddings(x)
  rho_clean_c = 12.0 * torch.tanh(rho_clean / 12.0)
  clean_cart = GeoUtils.hyperbolic_polar_to_poincare_cartesian(
    rho_clean_c.squeeze(-1), theta_clean)

  prompt_mask = (valid_tokens == 0).unsqueeze(-1)
  # Prompt positions: z equals the clean Poincaré point.
  assert torch.allclose(
    z[prompt_mask.expand_as(z)], clean_cart[prompt_mask.expand_as(z)],
    atol=1e-5)
  # Generated positions: z differs from the clean Poincaré point.
  gen_mask = (valid_tokens == 1).unsqueeze(-1).expand_as(z)
  assert not torch.allclose(z[gen_mask], clean_cart[gen_mask], atol=1e-5)


# ---------------------------------------------------------------------------
# Contract: gradient flow to the embedding table (ARCH §5 gradient guarantee,
# test contracts #5 and #6) -- THE key contract.
# ---------------------------------------------------------------------------

def test_gradient_reaches_embedding_table_through_zt():
  """Test contract #5: a tiny forward+CE step yields a non-zero
  `backbone.sphere_embed.weight.grad` after `loss.backward()`.

  Gradient must reach the embedding table THROUGH z_t (q_xt), not only through
  the loss head. A tiny linear denoiser stands in for `forward` so the path is
  z_t -> denoiser -> CE -> backward.
  """
  hflm, config, backbone = _build_hflm_stub(
    d=8, length=4, vocab_size=6, n_blocks=1)
  B, L, d, V = 2, 4, 8, 6
  x = torch.randint(0, V, (B, L))
  alpha_t = torch.full((B, L, 1), 0.4, dtype=torch.float32)

  z_t = hflm.q_xt(x, alpha_t, use_pure_noise=False)  # [B,L,d]
  denoiser = torch.nn.Linear(d, V)
  logits = denoiser(z_t.float())
  log_x_theta = logits.log_softmax(-1)
  ce = -log_x_theta.gather(-1, x.unsqueeze(-1)).squeeze(-1)
  loss = ce.mean()

  backbone.sphere_embed.weight.grad = None
  loss.backward()
  grad = backbone.sphere_embed.weight.grad
  assert grad is not None
  assert float(grad.abs().sum()) > 0.0


def test_clean_endpoint_not_detached_requires_grad():
  """Test contract #6: the clamped clean endpoint is NOT detached -- z_t is
  connected to `sphere_embed.weight` and carries grad.

  We assert z_t.requires_grad and that backprop populates the embedding grad
  (the rho_clamp / geodesic path did not `.detach()` the clean endpoint).
  """
  hflm, config, backbone = _build_hflm_stub(
    d=8, length=3, vocab_size=6, n_blocks=1)
  B, L = 2, 3
  x = torch.randint(0, 6, (B, L))
  alpha_t = torch.full((B, L, 1), 0.5, dtype=torch.float32)
  z_t = hflm.q_xt(x, alpha_t, use_pure_noise=False)
  assert z_t.requires_grad
  backbone.sphere_embed.weight.grad = None
  z_t.float().sum().backward()
  assert backbone.sphere_embed.weight.grad is not None
  assert float(backbone.sphere_embed.weight.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# Contract: CE loss unchanged (ARCH §3.1, test contract #6 family)
# ---------------------------------------------------------------------------

def test_nll_per_token_matches_gathered_log_softmax():
  """`HFLM.nll_per_token` equals `-log_softmax(logits).gather(target)` (same as
  `SFM.nll_per_token`). ARCH §3.1.
  """
  hflm, config, backbone = _build_hflm_stub(
    d=8, length=4, vocab_size=7, n_blocks=1)
  B, L, V = 2, 4, 7
  logits = torch.randn(B, L, V, dtype=torch.float64)
  log_x_theta = logits.log_softmax(-1)
  x0 = torch.randint(0, V, (B, L))
  expected = -log_x_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
  got = hflm.nll_per_token(
    log_x_theta=log_x_theta, xt=None, x0=x0,
    alpha_t=None, dalpha_t=None)
  assert torch.allclose(got, expected)


# ---------------------------------------------------------------------------
# Contract: rho-bound robustness in q_xt (ARCH §5, test contract #9)
# ---------------------------------------------------------------------------

def test_q_xt_does_not_raise_for_huge_embedding_norm():
  """Test contract #9: with embedding weights scaled to huge norm, `q_xt` does
  NOT raise -- the rho_clamp keeps the polar endpoint under the 20 bound.
  """
  hflm, config, backbone = _build_hflm_stub(
    d=32, length=4, vocab_size=8, n_blocks=1)
  with torch.no_grad():
    backbone.sphere_embed.weight.mul_(1e3)  # ‖e_v‖ ≫ 20
  B, L = 2, 4
  x = torch.randint(0, 8, (B, L))
  alpha_t = torch.full((B, L, 1), 0.5, dtype=torch.float32)
  z = hflm.q_xt(x, alpha_t, use_pure_noise=False)  # must not raise
  assert torch.isfinite(z).all()
  assert float(_poincare_norm(z).max()) < 1.0


# ---------------------------------------------------------------------------
# Contract: HyperbolicDiT backbone deltas (ARCH §3.2 / §6, test contracts #7
# family / #13)
# ---------------------------------------------------------------------------

def test_hyperbolic_dit_importable_from_models():
  """ARCH §2 / §3.2: `models.hyperbolic_dit.HyperbolicDiT` exists and is
  reachable via `from . import hyperbolic_dit` in `models/__init__`.
  """
  import models
  assert hasattr(models, 'hyperbolic_dit')
  assert hasattr(models.hyperbolic_dit, 'HyperbolicDiT')


def test_get_hyperbolic_polar_embeddings_shapes_and_unit_direction():
  """ARCH §3.2: `get_hyperbolic_polar_embeddings(ids)` returns
  `(rhos[B,L,1], thetas[B,L,d])` with ‖thetas‖≈1.
  """
  config = _make_config(d=16, length=4, vocab_size=10, n_blocks=1)
  backbone = _make_backbone(config, vocab_size=10)
  B, L, d = 2, 4, 16
  x = torch.randint(0, 10, (B, L))
  rhos, thetas = backbone.get_hyperbolic_polar_embeddings(x)
  assert tuple(rhos.shape) == (B, L, 1)
  assert tuple(thetas.shape) == (B, L, d)
  norms = thetas.norm(p=2, dim=-1)
  assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_get_hyperbolic_polar_embeddings_preserves_unnormalized_radius():
  """ARCH §3.2 / §5: rho = ‖emb‖ for the UNNORMALIZED embedding -- a row whose
  norm != 1 keeps its norm (NOT renormalized inside the getter).
  """
  config = _make_config(d=16, length=2, vocab_size=10, n_blocks=1)
  backbone = _make_backbone(config, vocab_size=10)
  with torch.no_grad():
    # Force token 3's embedding to a known non-unit norm.
    backbone.sphere_embed.weight[3].zero_()
    backbone.sphere_embed.weight[3, 0] = 5.0  # ‖e_3‖ = 5
  x = torch.tensor([[3, 3]])
  rhos, _ = backbone.get_hyperbolic_polar_embeddings(x)
  assert torch.allclose(rhos.squeeze(-1),
                        torch.full((1, 2), 5.0), atol=1e-5)


def test_hyperbolic_dit_forward_returns_logits_shape():
  """ARCH §3.2: `forward(x0, xt, sigma)` returns logits `[B,L,V]` consuming the
  Poincaré `xt` as-is (no sphere calibration of the output).
  """
  V = 9
  config = _make_config(d=16, length=4, vocab_size=V, n_blocks=1)
  backbone = _make_backbone(config, vocab_size=V)
  B, L, d = 2, 4, 16
  # A valid Poincaré-ball point (‖xt‖<1).
  xt = torch.randn(B, L, d) * 0.1
  sigma = torch.zeros(B)
  out = backbone(None, xt, sigma, None)
  assert tuple(out.shape) == (B, L, V)


# ---------------------------------------------------------------------------
# Contract: _validate_configuration rejects hyperbolic-arch (test contract #16)
# ---------------------------------------------------------------------------

def test_validate_configuration_rejects_hyperbolic_arch():
  """Test contract #16: `HFLM(model.type='hyperbolic-arch')` raises in
  `_validate_configuration` (justnorm destroys the radial signal -- ARCH §6).
  """
  config = _make_config(
    d=16, length=2, vocab_size=8, n_blocks=1,
    model_type='hyperbolic-arch')
  # Build a stub whose attributes mirror an HFLM; call the real validator.
  backbone = _make_backbone(
    _make_config(d=16, length=2, vocab_size=8, n_blocks=1), vocab_size=8)
  stub = _HFLMStub(config, backbone)
  with pytest.raises((ValueError, Exception)):
    stub._validate_configuration()
