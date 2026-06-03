"""`q_xt` is a single differentiable bridge call (ARCH Section 3 / Section 9).

These tests pin the headline HBFM contract: `q_xt` runs the hyperbolic bridge
directly on the raw, *un-renormalized* `backbone.sphere_embed.weight` so the
result is an in-ball Poincare point (`||z|| < 1`) carrying a live gradient back
to that weight.  The bridge normalizes the target DIRECTION internally
(`x/||x||` for general d; `atan2(e1,e0)` for d=2) and draws the radius `rho`
independently of the embedding, so `q_xt(use_pure_noise=False)` depends ONLY on
the embedding *direction*: it is INVARIANT to the embedding norm (global and
per-row) and SENSITIVE to its direction.  Written against the interface in
`experiments/hbfm/ARCH.md` (Sections 8/10).

Everything runs CPU-only on tiny dims (d in {2, 8}, V=12, B=2, L=4).
"""
import pytest
import torch

import geo_bridge
from conftest import HBFM_T_MIN, HBFM_T_MAX, SMALL_D, BINARY_D  # noqa: F401


def _heat_t(hbfm, n):
  """Draw a heat-time vector in [t_min, t_max] without relying on the (yet
  unimplemented) `_sample_heat_t` so the q_xt tests fail on the q_xt mechanism
  itself (in-ball / dispatch / grad), not on a missing helper.

  Falls back to the conftest bounds when `t_min` / `t_max` are not yet set on
  the algo object, so these tests reach the actual `q_xt` call rather than
  dying earlier on a missing attribute (the attribute contract is asserted
  separately in test_hbfm_loss.py)."""
  torch.manual_seed(0)
  t_min = getattr(hbfm, 't_min', HBFM_T_MIN)
  t_max = getattr(hbfm, 't_max', HBFM_T_MAX)
  return (torch.rand(n) * (t_max - t_min) + t_min).float()


# --------------------------------------------------------------------------
# q_xt invariants: shape / dtype / in-ball / differentiable
# --------------------------------------------------------------------------
def test_q_xt_returns_blD_shape_general_d(hbfm_d8, x0):
  t = _heat_t(hbfm_d8, x0.shape[0])
  z = hbfm_d8.q_xt(x0, t, use_pure_noise=False)
  assert tuple(z.shape) == (x0.shape[0], x0.shape[1], SMALL_D)


def test_q_xt_returns_blD_shape_binary_d(hbfm_d2, x0):
  t = _heat_t(hbfm_d2, x0.shape[0])
  z = hbfm_d2.q_xt(x0, t, use_pure_noise=False)
  assert tuple(z.shape) == (x0.shape[0], x0.shape[1], BINARY_D)


def test_q_xt_returns_float32_general_d(hbfm_d8, x0):
  t = _heat_t(hbfm_d8, x0.shape[0])
  z = hbfm_d8.q_xt(x0, t, use_pure_noise=False)
  assert z.dtype == torch.float32


def test_q_xt_in_ball_general_d(hbfm_d8, x0):
  """`||z|| < 1` elementwise on the general-d Poincare path."""
  t = _heat_t(hbfm_d8, x0.shape[0])
  z = hbfm_d8.q_xt(x0, t, use_pure_noise=False)
  assert z.norm(dim=-1).max().item() < 1.0


def test_q_xt_in_ball_binary_d(hbfm_d2, x0):
  """`||z|| < 1` elementwise on the d=2 closed-form path."""
  t = _heat_t(hbfm_d2, x0.shape[0])
  z = hbfm_d2.q_xt(x0, t, use_pure_noise=False)
  assert z.norm(dim=-1).max().item() < 1.0


def test_q_xt_requires_grad_general_d(hbfm_d8, x0):
  t = _heat_t(hbfm_d8, x0.shape[0])
  z = hbfm_d8.q_xt(x0, t, use_pure_noise=False)
  assert z.requires_grad is True


def test_q_xt_requires_grad_binary_d(hbfm_d2, x0):
  t = _heat_t(hbfm_d2, x0.shape[0])
  z = hbfm_d2.q_xt(x0, t, use_pure_noise=False)
  assert z.requires_grad is True


def test_q_xt_is_finite_general_d(hbfm_d8, x0):
  t = _heat_t(hbfm_d8, x0.shape[0])
  z = hbfm_d8.q_xt(x0, t, use_pure_noise=False)
  assert torch.isfinite(z).all().item()


def test_q_xt_valid_tokens_all_ones_is_noop(hbfm_d8, x0, valid_tokens):
  """All-ones `valid_tokens` (the Sudoku case) leaves the bridge in-ball."""
  t = _heat_t(hbfm_d8, x0.shape[0])
  z = hbfm_d8.q_xt(x0, t, use_pure_noise=False, valid_tokens=valid_tokens)
  assert z.norm(dim=-1).max().item() < 1.0


# --------------------------------------------------------------------------
# Gradient reaches the embedding through q_xt (headline requirement).
# Backprop a scalar of z directly: z depends on emb via the bridge direction,
# independent of the (zero-gated, adaLN-Zero) head, so a fresh model already
# yields a nonzero emb grad here (ARCH Section 9, option (a)).
# --------------------------------------------------------------------------
def test_q_xt_backward_gives_nonzero_emb_grad_general_d(hbfm_d8, x0):
  hbfm_d8.backbone.sphere_embed.weight.grad = None
  t = _heat_t(hbfm_d8, x0.shape[0])
  z = hbfm_d8.q_xt(x0, t, use_pure_noise=False)
  z.sum().backward()
  grad = hbfm_d8.backbone.sphere_embed.weight.grad
  assert grad is not None and grad.norm().item() > 0.0


def test_q_xt_backward_gives_nonzero_emb_grad_binary_d(hbfm_d2, x0):
  hbfm_d2.backbone.sphere_embed.weight.grad = None
  t = _heat_t(hbfm_d2, x0.shape[0])
  z = hbfm_d2.q_xt(x0, t, use_pure_noise=False)
  z.sum().backward()
  grad = hbfm_d2.backbone.sphere_embed.weight.grad
  assert grad is not None and grad.norm().item() > 0.0


# --------------------------------------------------------------------------
# Embedding-norm invariance / direction-sensitivity (ARCH Section 8/10, the
# "raw `sphere_embed.weight` not `get_sphere_embeddings`" contract).
#
# The differentiable bridge normalizes the target DIRECTION (`x/||x||` for
# general d; `atan2(e1,e0)` for d=2) and draws the radius `rho` independently of
# the embedding, so `q_xt(use_pure_noise=False)` depends ONLY on the embedding
# *direction* and is INVARIANT to its norm -- both a global rescale and a
# per-row rescale leave the output unchanged. It IS sensitive to the embedding
# *direction*, which is how we know it genuinely indexes the raw weight (a
# pre-normalized table would still respond to direction, but the bridge reads
# `backbone.sphere_embed.weight` directly, not `get_sphere_embeddings`).
#
# RNG must be pinned immediately before each `q_xt` call so the bridge's
# stochastic radius/direction draws coincide and `allclose` is meaningful.
# --------------------------------------------------------------------------
def _clone_hbfm(hbfm):
  """A weight-identical copy so we can perturb the embedding of the clone and
  compare against the untouched original (same RNG, same draws)."""
  import copy
  clone = copy.deepcopy(hbfm)
  clone.load_state_dict(hbfm.state_dict())
  return clone


def test_q_xt_invariant_to_global_embedding_norm_general_d(hbfm_d8, x0):
  """Multiplying the whole `sphere_embed.weight` by a constant leaves the d=8
  bridge output unchanged: only the direction (normalized inside the bridge)
  feeds `q_xt`."""
  t = _heat_t(hbfm_d8, x0.shape[0])
  clone = _clone_hbfm(hbfm_d8)
  torch.manual_seed(1)
  z_base = hbfm_d8.q_xt(x0, t, use_pure_noise=False)
  with torch.no_grad():
    clone.backbone.sphere_embed.weight.mul_(3.0)
  torch.manual_seed(1)
  z_scaled = clone.q_xt(x0, t, use_pure_noise=False)
  assert torch.allclose(z_base, z_scaled, atol=1e-5)


def test_q_xt_invariant_to_global_embedding_norm_binary_d(hbfm_d2, x0):
  """Same global-norm invariance on the d=2 closed-form (`atan2`) path."""
  t = _heat_t(hbfm_d2, x0.shape[0])
  clone = _clone_hbfm(hbfm_d2)
  torch.manual_seed(1)
  z_base = hbfm_d2.q_xt(x0, t, use_pure_noise=False)
  with torch.no_grad():
    clone.backbone.sphere_embed.weight.mul_(3.0)
  torch.manual_seed(1)
  z_scaled = clone.q_xt(x0, t, use_pure_noise=False)
  assert torch.allclose(z_base, z_scaled, atol=1e-5)


def test_q_xt_invariant_to_single_row_embedding_norm_general_d(hbfm_d8, x0):
  """Rescaling a single embedding row's norm (a token present in `x0`) does not
  change the d=8 bridge output."""
  token = int(x0[0, 0].item())
  t = _heat_t(hbfm_d8, x0.shape[0])
  clone = _clone_hbfm(hbfm_d8)
  torch.manual_seed(1)
  z_base = hbfm_d8.q_xt(x0, t, use_pure_noise=False)
  with torch.no_grad():
    clone.backbone.sphere_embed.weight[token].mul_(5.0)
  torch.manual_seed(1)
  z_scaled = clone.q_xt(x0, t, use_pure_noise=False)
  assert torch.allclose(z_base, z_scaled, atol=1e-5)


def test_q_xt_invariant_to_single_row_embedding_norm_binary_d(hbfm_d2, x0):
  """Per-row norm invariance on the d=2 path."""
  token = int(x0[0, 0].item())
  t = _heat_t(hbfm_d2, x0.shape[0])
  clone = _clone_hbfm(hbfm_d2)
  torch.manual_seed(1)
  z_base = hbfm_d2.q_xt(x0, t, use_pure_noise=False)
  with torch.no_grad():
    clone.backbone.sphere_embed.weight[token].mul_(5.0)
  torch.manual_seed(1)
  z_scaled = clone.q_xt(x0, t, use_pure_noise=False)
  assert torch.allclose(z_base, z_scaled, atol=1e-5)


def test_q_xt_sensitive_to_embedding_direction_general_d(hbfm_d8, x0):
  """Perturbing the DIRECTION of an embedding row for a token present in `x0`
  changes the d=8 bridge output: `q_xt` genuinely indexes the embedding."""
  token = int(x0[0, 0].item())
  t = _heat_t(hbfm_d8, x0.shape[0])
  clone = _clone_hbfm(hbfm_d8)
  torch.manual_seed(1)
  z_base = hbfm_d8.q_xt(x0, t, use_pure_noise=False)
  with torch.no_grad():
    torch.manual_seed(99)
    clone.backbone.sphere_embed.weight[token].add_(
      torch.randn(SMALL_D) * 0.5)
  torch.manual_seed(1)
  z_perturbed = clone.q_xt(x0, t, use_pure_noise=False)
  assert not torch.allclose(z_base, z_perturbed, atol=1e-4)


def test_q_xt_sensitive_to_embedding_direction_binary_d(hbfm_d2, x0):
  """Rotating a d=2 embedding row's angle for a token in `x0` changes the
  closed-form (`atan2`) bridge output."""
  token = int(x0[0, 0].item())
  t = _heat_t(hbfm_d2, x0.shape[0])
  clone = _clone_hbfm(hbfm_d2)
  torch.manual_seed(1)
  z_base = hbfm_d2.q_xt(x0, t, use_pure_noise=False)
  with torch.no_grad():
    e = clone.backbone.sphere_embed.weight[token].clone()
    ca, sa = torch.cos(torch.tensor(1.0)), torch.sin(torch.tensor(1.0))
    clone.backbone.sphere_embed.weight[token] = torch.stack(
      [ca * e[0] - sa * e[1], sa * e[0] + ca * e[1]])
  torch.manual_seed(1)
  z_perturbed = clone.q_xt(x0, t, use_pure_noise=False)
  assert not torch.allclose(z_base, z_perturbed, atol=1e-4)


def test_q_xt_reads_raw_sphere_embed_weight_not_normalized_accessor(
    hbfm_d8, x0):
  """`q_xt` consumes `backbone.sphere_embed.weight` directly; it must NOT route
  through `get_sphere_embeddings` (which would unit-normalize and discard the
  free-norm signal)."""
  import types
  calls = {'n': 0}
  orig = hbfm_d8.backbone.get_sphere_embeddings

  def spy(self, *a, **k):
    calls['n'] += 1
    return orig(*a, **k)

  hbfm_d8.backbone.get_sphere_embeddings = types.MethodType(
    spy, hbfm_d8.backbone)
  t = _heat_t(hbfm_d8, x0.shape[0])
  torch.manual_seed(1)
  hbfm_d8.q_xt(x0, t, use_pure_noise=False)
  assert calls['n'] == 0


# --------------------------------------------------------------------------
# Grad route is q_xt only: detaching z before the head leaves emb grad-free
# (no tied readout leak). torch-2.7-safe: use loss.backward() + check .grad,
# never autograd.grad on a detached tensor.
# --------------------------------------------------------------------------
def test_emb_grad_is_none_when_qxt_output_detached(hbfm_d8, x0):
  hbfm_d8.backbone.sphere_embed.weight.grad = None
  t = _heat_t(hbfm_d8, x0.shape[0])
  z = hbfm_d8.q_xt(x0, t, use_pure_noise=False).detach()
  t_max = getattr(hbfm_d8, 't_max', HBFM_T_MAX)
  sigma = (t / t_max).unsqueeze(-1)
  log_x_theta = hbfm_d8.forward(x0=x0, xt=z, sigma=sigma)
  loss = -log_x_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1).mean()
  loss.backward()
  grad = hbfm_d8.backbone.sphere_embed.weight.grad
  assert grad is None or grad.norm().item() == 0.0


# --------------------------------------------------------------------------
# d=2 vs general-d dispatch: the two paths call the correct bridge primitive.
# --------------------------------------------------------------------------
def test_general_d_dispatches_to_poincare_bridge(hbfm_d8, x0, monkeypatch):
  calls = {'general': 0, 'binary': 0}
  orig_general = geo_bridge.HyperbolicHeatKernel.poincare_bridge
  orig_binary = geo_bridge.BinaryHyperbolicHeatKernel.binary_poincare_bridge

  def spy_general(*a, **k):
    calls['general'] += 1
    return orig_general(*a, **k)

  def spy_binary(*a, **k):
    calls['binary'] += 1
    return orig_binary(*a, **k)

  monkeypatch.setattr(geo_bridge.HyperbolicHeatKernel, 'poincare_bridge',
                      staticmethod(spy_general))
  monkeypatch.setattr(geo_bridge.BinaryHyperbolicHeatKernel,
                      'binary_poincare_bridge', staticmethod(spy_binary))

  t = _heat_t(hbfm_d8, x0.shape[0])
  hbfm_d8.q_xt(x0, t, use_pure_noise=False)
  assert calls['general'] >= 1 and calls['binary'] == 0


def test_binary_d_dispatches_to_binary_poincare_bridge(hbfm_d2, x0, monkeypatch):
  calls = {'general': 0, 'binary': 0}
  orig_general = geo_bridge.HyperbolicHeatKernel.poincare_bridge
  orig_binary = geo_bridge.BinaryHyperbolicHeatKernel.binary_poincare_bridge

  def spy_general(*a, **k):
    calls['general'] += 1
    return orig_general(*a, **k)

  def spy_binary(*a, **k):
    calls['binary'] += 1
    return orig_binary(*a, **k)

  monkeypatch.setattr(geo_bridge.HyperbolicHeatKernel, 'poincare_bridge',
                      staticmethod(spy_general))
  monkeypatch.setattr(geo_bridge.BinaryHyperbolicHeatKernel,
                      'binary_poincare_bridge', staticmethod(spy_binary))

  t = _heat_t(hbfm_d2, x0.shape[0])
  hbfm_d2.q_xt(x0, t, use_pure_noise=False)
  assert calls['binary'] >= 1 and calls['general'] == 0


# --------------------------------------------------------------------------
# pure-noise branch: free kernel at t_max, target-independent, in-ball.
# --------------------------------------------------------------------------
def test_q_xt_pure_noise_in_ball_general_d(hbfm_d8, x0):
  t_max = getattr(hbfm_d8, 't_max', HBFM_T_MAX)
  t = torch.full((x0.shape[0],), t_max, dtype=torch.float32)
  z = hbfm_d8.q_xt(x0, t, use_pure_noise=True)
  assert z.norm(dim=-1).max().item() < 1.0


def test_q_xt_pure_noise_is_target_independent_general_d(hbfm_d8):
  """The free kernel ignores the target tokens, so two different token batches
  drawn under the same seed give the same pure-noise sample."""
  B, L = 2, hbfm_d8.num_tokens
  t_max = getattr(hbfm_d8, 't_max', HBFM_T_MAX)
  t = torch.full((B,), t_max, dtype=torch.float32)
  x_a = torch.zeros(B, L, dtype=torch.long)
  x_b = torch.full((B, L), 5, dtype=torch.long)
  torch.manual_seed(7)
  z_a = hbfm_d8.q_xt(x_a, t, use_pure_noise=True)
  torch.manual_seed(7)
  z_b = hbfm_d8.q_xt(x_b, t, use_pure_noise=True)
  assert torch.allclose(z_a, z_b)
