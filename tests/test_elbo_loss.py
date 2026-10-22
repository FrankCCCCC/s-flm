"""Tests of the ELBO losses of EFLM and HBFM (`algo.loss_type=elbo`).

The NELBO of a diffusion model is tight at the Bayes posterior: with
p_theta(v | x_t) = q(v | x_t) the learned reverse transitions are the true
ones and the bound telescopes to the data entropy H(q_data)
(slides/oct07_2026 "Bayes Optimal"; experiments/elfm_hbfm_ada_curv_elbo_sudoku/
math.md). Both losses are checked against that value on data whose Bayes
posterior is available in closed form, through the real `nll` path (one t per
sequence, the importance weight, the reconstruction and prior endpoints):

  * EFLM: a unigram over fixed embeddings; the posterior is the softmax of
    the Gaussian log-likelihoods plus the log prior, fed in by a stubbed
    `forward`.
  * HBFM: uniform words through the horosphere readout with a zero residual,
    the context-free Bayes posterior; the adaptive curvature must leave the
    mean unchanged.

A mis-specified posterior must give a larger value (the quantity is a bound),
the CE path must be untouched, and on a GPU the real backbones must be
trainable through both losses under bf16 autocast.
"""
import math

import pytest
import torch

from conftest import REPO_ROOT  # noqa: F401  (repo root on sys.path)
import trainer_base
from test_hbfm import DEVICE, _NoMetrics, _build_model, _tokens, needs_gpu
from test_hbfm_ada_curv import _fix_curvature

torch.manual_seed(0)

EFLM_OVERRIDES = [
  'algo=eflm', 'model=tiny-sphere-dit', 'sampler=eflm', 'data=sudoku',
  'noise=autonomous', 'noise.tau_max=2.0', 'algo.invert_time_convention=false',
  'algo.rho_min=3.0', 'algo.rho_max=3.0', 'algo.loss_type=elbo',
  'model.hidden_size=64', 'model.n_heads=4', 'model.cond_dim=32',
  'model.n_blocks=2', 'model.length=8', 'model.dropout=0.0',
  'loader.global_batch_size=4', 'loader.batch_size=4',
  'trainer.devices=1', 'trainer.num_nodes=1',
]
HBFM_ELBO = ['algo.forward_type=horosphere', 'algo.readout_precision=float64',
             'algo.loss_type=elbo']


def _build_eflm(monkeypatch, overrides=()):
  import algo
  import dataloader
  import hydra
  import main  # noqa: F401  registers the hydra resolvers
  monkeypatch.setattr(trainer_base.metrics, 'Metrics', _NoMetrics)
  with hydra.initialize(config_path='../configs', version_base=None):
    cfg = hydra.compose(config_name='config',
                        overrides=EFLM_OVERRIDES + list(overrides))
  torch.manual_seed(0)
  model = algo.EFLM(cfg, tokenizer=dataloader.get_tokenizer(cfg))
  return model.to(DEVICE), cfg


def _sequence_losses(model, x, repeats):
  """Per-sequence token-mean NELBO, `repeats` draws of (t, noise) per sequence."""
  with torch.no_grad():
    return torch.cat([
      model.nll(x, None, trainer_base.TrainingContext())[0].mean(-1)
      for _ in range(repeats)])


def _summary(losses):
  return losses.mean().item(), losses.std().item() / math.sqrt(len(losses))


# ---------------------------------------------------------------------------
# EFLM: Gaussian interpolant, closed-form Bayes posterior
# ---------------------------------------------------------------------------

def _bayes_eflm_forward(model, log_prior):
  """q(v | x_t) of the unigram: x_t = (1 - b) e_v + b eps, b the noise fraction
  the time conditioning encodes (sigma = -log alpha_t, b = 1 - alpha_t)."""
  table = model._sc_embed_table().detach()                      # [V, d]

  def forward(*, x0=None, xt, sigma, context=None):
    b = 1.0 - torch.exp(-sigma)                                 # [B, 1]
    mean = (1.0 - b)[..., None, None] * table                   # [B, 1, V, d]
    sq = (xt[:, :, None, :] - mean).square().sum(-1)            # [B, L, V]
    return (log_prior - sq / (2.0 * b.square()[..., None])).log_softmax(-1)
  return forward


def test_eflm_nelbo_is_tight_at_the_bayes_posterior(monkeypatch):
  model, cfg = _build_eflm(monkeypatch)
  torch.manual_seed(1)
  log_prior = torch.randn(model.vocab_size, device=DEVICE).log_softmax(-1)
  entropy = -(log_prior.exp() * log_prior).sum().item()
  B, L = 2048, cfg.model.length
  x = torch.multinomial(log_prior.exp(), B * L, replacement=True).view(B, L)

  monkeypatch.setattr(model, 'forward', _bayes_eflm_forward(model, log_prior))
  mean, se = _summary(_sequence_losses(model, x, repeats=8))
  # Tight: the only slack is KL(q(x_1) || N(0, I)) at the noise floor, < 1e-4.
  assert abs(mean - entropy) < 4 * se + 0.01 * entropy, (mean, entropy, se)

  # A tempered posterior is not the Bayes one: the bound loosens.
  bayes = _bayes_eflm_forward(model, log_prior)
  monkeypatch.setattr(model, 'forward',
                      lambda **kw: (0.5 * bayes(**kw)).log_softmax(-1))
  worse, se_w = _summary(_sequence_losses(model, x, repeats=8))
  assert worse - entropy > 4 * math.hypot(se, se_w) + 0.05 * entropy, (worse, entropy)


def test_eflm_ce_loss_type_is_the_plain_ce(monkeypatch):
  model, cfg = _build_eflm(monkeypatch, ['algo.loss_type=ce'])
  V, B, L = model.vocab_size, 3, cfg.model.length
  log_p = torch.randn(B, L, V, device=DEVICE).log_softmax(-1)
  x = torch.randint(0, V, (B, L), device=DEVICE)
  alpha = torch.rand(B, 1, device=DEVICE)
  loss = model.nll_per_token(log_p, None, x, alpha, -torch.ones_like(alpha))
  assert torch.allclose(loss, -log_p.gather(-1, x[..., None]).squeeze(-1))


@pytest.mark.parametrize('build', [_build_eflm, _build_model])
def test_unknown_loss_type_is_rejected(monkeypatch, build):
  with pytest.raises(ValueError, match='loss_type'):
    build(monkeypatch, ['algo.loss_type=bogus'])


# ---------------------------------------------------------------------------
# HBFM: horosphere readout with a zero residual = the Bayes posterior
# ---------------------------------------------------------------------------

def _residual(monkeypatch, model, value):
  """Replace the trunk by a constant residual `value` ([V] or 0)."""
  V = model.vocab_size

  def forward(x0, xt, sigma, context=None):
    return torch.zeros(*xt.shape[:2], V, device=xt.device) + value
  monkeypatch.setattr(model.backbone, 'forward', forward)


def test_hbfm_nelbo_is_tight_at_the_bayes_posterior(monkeypatch):
  model, _ = _build_model(monkeypatch, HBFM_ELBO)
  _residual(monkeypatch, model, 0.0)
  torch.manual_seed(2)
  x = _tokens(model, B=2048)                                   # uniform words
  entropy = math.log(model.vocab_size)
  mean, se = _summary(_sequence_losses(model, x, repeats=8))
  # The slack is the proposal's floor at the origin (heat times below 1e-3
  # are never drawn): at most 2 (d-1)^2 kappa^2 x 1e-3 per factor, ~0.3 %.
  assert abs(mean - entropy) < 4 * se + 0.01 * entropy, (mean, entropy, se)

  # A biased posterior (a fixed non-zero residual) loosens the bound.
  _residual(monkeypatch, model, torch.randn(model.vocab_size, device=DEVICE))
  worse, se_w = _summary(_sequence_losses(model, x, repeats=8))
  assert worse - entropy > 4 * math.hypot(se, se_w) + 0.05 * entropy, (worse, entropy)


def test_hbfm_nelbo_mean_is_invariant_to_the_adaptive_curvature(monkeypatch):
  model, _ = _build_model(monkeypatch, HBFM_ELBO + ['algo.curvature_bound=4.0'])
  _residual(monkeypatch, model, 0.0)
  torch.manual_seed(3)
  x = _tokens(model, B=2048)
  entropy = math.log(model.vocab_size)
  _fix_curvature(monkeypatch, model, 1.0)
  ref, se_ref = _summary(_sequence_losses(model, x, repeats=8))
  _fix_curvature(monkeypatch, model, [0.25, 0.5, 1.0, 2.0, 4.0, 0.5, 2.0, 1.0])
  mixed, se_mixed = _summary(_sequence_losses(model, x, repeats=8))
  # Position i integrates the base-curvature NELBO on its own clock
  # lambda_i t; the weight lambda_i / q(t) and the horizon term keep the mean.
  # The floor of the proposal scales with lambda_i, hence the 1.5 % allowance.
  assert abs(mixed - ref) < 4 * math.hypot(se_ref, se_mixed) + 0.015 * entropy, (mixed, ref)
  assert abs(mixed - entropy) < 4 * se_mixed + 0.015 * entropy, (mixed, entropy)


def test_hbfm_ce_loss_type_is_the_weighted_ce(monkeypatch):
  model, _ = _build_model(monkeypatch, ['algo.forward_type=horosphere'])
  V, B, L = model.vocab_size, 3, 8
  log_p = torch.randn(B, L, V, device=DEVICE).log_softmax(-1)
  x = _tokens(model, B, L)
  weight = torch.rand(B, 1, dtype=torch.float64, device=DEVICE)
  loss = model.nll_per_token(log_p, None, x, None, None, weight)
  assert torch.allclose(loss, weight * -log_p.gather(-1, x[..., None]).squeeze(-1))


def _explicit_path_kl(logits, targets, rhos, thetas, phis, dims, curvatures):
  """The docstring's form of Loss.bridge_loss_elbo_refactor: per-word unit vectors w_v,
  (d_m - 1)^2 kappa_m^2 / 2 ||sum_v mu_v w_v||^2 per factor (materializes (B, L, V, d_m))."""
  from numeric.geo_bridge import GeoUtils
  V = logits.shape[-1]
  mu = logits.double().softmax(-1) - torch.nn.functional.one_hot(targets, V).double()
  loss, offset = 0.0, 0
  for i, (d, K) in enumerate(zip(dims, curvatures)):
    theta = thetas[..., offset:offset + d]
    phi = torch.nn.functional.normalize(phis[:, offset:offset + d], dim=-1)
    offset += d
    kappa = 1.0 / GeoUtils._curvature_scale(K)
    u = (kappa * rhos[..., i])[..., None]
    inner = theta @ phi.T
    sin2, cos2 = (1 - inner) / 2, (1 + inner) / 2
    radial = cos2 * torch.exp(-u) - sin2 * torch.exp(u)
    denom = sin2 * torch.exp(u) + cos2 * torch.exp(-u)
    perp = phi[None, None] - inner[..., None] * theta[..., None, :]
    w = (radial[..., None] * theta[..., None, :] + perp) / denom[..., None]
    err = (mu[..., None] * w).sum(-2)
    loss = loss + (d - 1) ** 2 * kappa ** 2 * err.square().sum(-1) / 2
  return loss


@pytest.mark.parametrize('grad', [False, True])
def test_hbfm_path_kl_matches_the_explicit_per_word_form(grad):
  """The memory-light form (no (B, L, V, d) tensors, factors recomputed in backward) is the
  same function, value and gradient."""
  from numeric.loss import Loss
  torch.manual_seed(5)
  B, L, V, dims, curvatures = 2, 5, 12, [3, 3], [-0.5, -1.0]
  logits = torch.randn(B, L, V, device=DEVICE, requires_grad=grad)
  targets = torch.randint(0, V, (B, L), device=DEVICE)
  rhos = torch.rand(B, L, len(dims), device=DEVICE, dtype=torch.float64) * 3
  thetas = torch.cat([torch.nn.functional.normalize(torch.randn(B, L, d, device=DEVICE, dtype=torch.float64), dim=-1)
                      for d in dims], -1)
  phis = torch.randn(V, sum(dims), device=DEVICE, dtype=torch.float64)
  with torch.set_grad_enabled(grad):
    got = Loss.bridge_loss_elbo_refactor(logits, targets, rhos, thetas, phis, dims, curvatures)
    want = _explicit_path_kl(logits, targets, rhos, thetas, phis, dims, curvatures)
  assert got.shape == (B, L) and torch.allclose(got, want, rtol=1e-10, atol=1e-12), (got - want).abs().max()
  if grad:
    g_got, = torch.autograd.grad(got.sum(), logits, retain_graph=True)
    g_want, = torch.autograd.grad(want.sum(), logits)
    assert torch.allclose(g_got, g_want, rtol=1e-6, atol=1e-9), (g_got - g_want).abs().max()


# ---------------------------------------------------------------------------
# The real backbones train through both losses
# ---------------------------------------------------------------------------

@needs_gpu
def test_elbo_losses_train_the_real_backbones(monkeypatch):
  torch.manual_seed(4)
  hbfm, _ = _build_model(monkeypatch, HBFM_ELBO + ['algo.curvature_bound=4.0'])
  eflm, _ = _build_eflm(monkeypatch)
  # A non-zero readout, so that the loss reaches the trunk (zero-init otherwise).
  for model in (hbfm, eflm):
    torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.02)
  for model, extra in [(hbfm, hbfm.backbone.curvature_layer.linear.weight),
                       (eflm, None)]:
    x = _tokens(model, B=4)
    valid = torch.ones_like(x)
    valid[:, :3] = 0                                            # sudoku-style prompt
    with torch.autocast('cuda', dtype=torch.bfloat16):          # as in training
      loss, _ = model.nll(x, None, trainer_base.TrainingContext(), valid_tokens=valid)
    assert loss.shape == x.shape and torch.isfinite(loss).all()
    (loss * valid).sum().backward()
    # adaLN-Zero: the blocks' branches are gated by the zero-initialized
    # modulation, so at init only the modulation itself receives a gradient
    # from the trunk (as in test_hbfm_ada_curv.test_nll_trains_the_curvature_channel).
    for p in [model.backbone.sphere_embed.weight,
              model.backbone.blocks[0].adaLN_modulation.bias, extra]:
      if p is not None:
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().max() > 0
