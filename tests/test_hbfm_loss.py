"""HBFM loss / heat-time / weight contract (ARCH Section 3 / Section 9).

Covers:
- `nll_per_token` == plain SFM CE, optionally scaled by the constant proposal
  weight `interval = t_max - t_min` when `weighted_ce` is on;
- `_sample_heat_t(n)` returns `t in [t_min, t_max]` and a constant `interval`;
- `sigma = t / t_max` is what the backbone receives;
- the embedding table is never renormalized (free norm; `renormalize_weights`
  is False and `q_xt` does not call `backbone.renormalize_weights`);
- the loss-through-the-head backward reaches the embedding once the zero-gated
  adaLN-Zero head is perturbed (ARCH Section 9, option (b)).

Written against `experiments/hbfm/ARCH.md` before implementation; EXPECTED to
FAIL until `_sample_heat_t`, the `interval`-based `nll_per_token`, and the
`sigma = t/t_max` plumbing exist.  CPU-only, tiny dims.
"""
import math

import pytest
import torch

from conftest import HBFM_T_MIN, HBFM_T_MAX


# --------------------------------------------------------------------------
# _sample_heat_t: uniform heat-time and constant proposal weight.
# --------------------------------------------------------------------------
def test_sample_heat_t_returns_n_times(hbfm_d8):
  ts, _interval = hbfm_d8._sample_heat_t(5)
  assert tuple(ts.shape) == (5,)


def test_sample_heat_t_within_bounds(hbfm_d8):
  ts, _interval = hbfm_d8._sample_heat_t(64)
  assert ts.min().item() >= hbfm_d8.t_min
  assert ts.max().item() <= hbfm_d8.t_max


def test_sample_heat_t_interval_is_t_max_minus_t_min(hbfm_d8):
  _ts, interval = hbfm_d8._sample_heat_t(3)
  assert math.isclose(float(interval), hbfm_d8.t_max - hbfm_d8.t_min,
                      rel_tol=1e-9)


def test_sample_heat_t_is_float32(hbfm_d8):
  ts, _interval = hbfm_d8._sample_heat_t(3)
  assert ts.dtype == torch.float32


# --------------------------------------------------------------------------
# nll_per_token: plain CE (weighted_ce=False) vs CE * interval (True).
# --------------------------------------------------------------------------
def _ref_ce(log_x_theta, x0):
  return -log_x_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1)


def test_nll_per_token_equals_plain_ce_when_unweighted(hbfm_d8, x0):
  torch.manual_seed(0)
  log_x_theta = torch.randn(x0.shape[0], x0.shape[1], hbfm_d8.vocab_size)
  log_x_theta = log_x_theta.log_softmax(-1)
  interval = hbfm_d8.t_max - hbfm_d8.t_min
  assert hbfm_d8.weighted_ce is False
  out = hbfm_d8.nll_per_token(log_x_theta, xt=None, x0=x0, interval=interval)
  assert torch.allclose(out, _ref_ce(log_x_theta, x0))


def test_nll_per_token_scales_by_interval_when_weighted(hbfm_d8, x0):
  torch.manual_seed(0)
  log_x_theta = torch.randn(x0.shape[0], x0.shape[1], hbfm_d8.vocab_size)
  log_x_theta = log_x_theta.log_softmax(-1)
  interval = hbfm_d8.t_max - hbfm_d8.t_min
  hbfm_d8.weighted_ce = True
  out = hbfm_d8.nll_per_token(log_x_theta, xt=None, x0=x0, interval=interval)
  assert torch.allclose(out, _ref_ce(log_x_theta, x0) * interval)


def test_nll_per_token_shape_is_b_l(hbfm_d8, x0):
  torch.manual_seed(0)
  log_x_theta = torch.randn(x0.shape[0], x0.shape[1], hbfm_d8.vocab_size)
  log_x_theta = log_x_theta.log_softmax(-1)
  interval = hbfm_d8.t_max - hbfm_d8.t_min
  out = hbfm_d8.nll_per_token(log_x_theta, xt=None, x0=x0, interval=interval)
  assert tuple(out.shape) == (x0.shape[0], x0.shape[1])


# --------------------------------------------------------------------------
# sigma = t / t_max is what the backbone forward receives in nll().
# --------------------------------------------------------------------------
def test_nll_feeds_sigma_equal_t_over_t_max(hbfm_d8, x0, valid_tokens, monkeypatch):
  captured = {}
  real_forward = hbfm_d8.forward

  def spy_forward(*, x0=None, xt=None, sigma=None, context=None):
    captured['sigma'] = sigma.detach().clone()
    return real_forward(x0=x0, xt=xt, sigma=sigma, context=context)

  monkeypatch.setattr(hbfm_d8, 'forward', spy_forward)
  torch.manual_seed(0)
  hbfm_d8.nll(x0, output_tokens=None, context=None,
              train_mode=True, valid_tokens=valid_tokens)
  sigma = captured['sigma']
  # sigma is t/t_max in [t_min/t_max, 1]; never the alpha-schedule mapping.
  assert sigma.max().item() <= 1.0 + 1e-6
  assert sigma.min().item() >= hbfm_d8.t_min / hbfm_d8.t_max - 1e-6


# --------------------------------------------------------------------------
# Embedding is not renormalized: flag is False and q_xt never calls the
# backbone's renormalize_weights.  (We avoid Lightning's optimizer_step, which
# needs a Trainer; the testable contract is "no HBFM-side renormalization".)
# --------------------------------------------------------------------------
def test_renormalize_weights_flag_is_false(hbfm_d8):
  assert hbfm_d8.renormalize_weights is False


def test_q_xt_does_not_renormalize_embedding(hbfm_d8, x0, monkeypatch):
  calls = {'n': 0}
  orig = hbfm_d8.backbone.renormalize_weights

  def spy():
    calls['n'] += 1
    return orig()

  monkeypatch.setattr(hbfm_d8.backbone, 'renormalize_weights', spy)
  before = hbfm_d8.backbone.sphere_embed.weight.detach().clone()
  t = (torch.rand(x0.shape[0]) * (hbfm_d8.t_max - hbfm_d8.t_min)
       + hbfm_d8.t_min).float()
  hbfm_d8.q_xt(x0, t, use_pure_noise=False)
  after = hbfm_d8.backbone.sphere_embed.weight.detach()
  assert calls['n'] == 0
  assert torch.equal(before, after)


def test_embedding_rows_not_unit_normalized_at_init(hbfm_d8):
  """A free-norm table is not on the unit sphere (norms differ from 1)."""
  norms = hbfm_d8.backbone.sphere_embed.weight.norm(dim=-1)
  assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


# --------------------------------------------------------------------------
# Loss-through-the-head backward reaches the embedding once the zero-gated
# head is perturbed (ARCH Section 9, option (b)).  Do NOT assert this on a
# fresh model's first backward (adaLN-Zero gates => zero emb grad).
# --------------------------------------------------------------------------
def test_loss_backward_gives_nonzero_emb_grad_after_head_perturbed(hbfm_d8, x0):
  # Perturb the zero-init readout AND the block adaLN gates so the loss
  # actually depends on xt (=q_xt output), hence on the embedding.
  with torch.no_grad():
    hbfm_d8.backbone.output_layer.linear.weight.normal_(0.0, 0.02)
    for block in hbfm_d8.backbone.blocks:
      block.adaLN_modulation.weight.normal_(0.0, 0.02)
      block.adaLN_modulation.bias.normal_(0.0, 0.02)
  hbfm_d8.backbone.sphere_embed.weight.grad = None

  t = (torch.rand(x0.shape[0]) * (hbfm_d8.t_max - hbfm_d8.t_min)
       + hbfm_d8.t_min).float()
  z = hbfm_d8.q_xt(x0, t, use_pure_noise=False)
  sigma = (t / hbfm_d8.t_max).unsqueeze(-1)
  log_x_theta = hbfm_d8.forward(x0=x0, xt=z, sigma=sigma)
  loss = -log_x_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1).mean()
  loss.backward()
  grad = hbfm_d8.backbone.sphere_embed.weight.grad
  assert grad is not None and grad.norm().item() > 0.0
