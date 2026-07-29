"""SphereDiT.rescale_radius: map word-embedding norms into [rho_min, rho_max].

Semantics under test (used by algo.EFLM via get_rescaled_embeddings /
_sc_embed_table):
  both None          -> identity
  only rho_min       -> norm + rho_min (floor by shift)
  rho_min == rho_max -> every norm pinned to that value (normalized embeddings)
  rho_min < rho_max  -> rho_min + range * tanh(norm / range), range = max - min
Directions are always preserved.
"""
import pytest
import torch

from models.sphere_dit import SphereDiT


def rescale(rho_min, rho_max, x):
  return SphereDiT.rescale_radius(x, rho_min=rho_min, rho_max=rho_max)


def cos_to_input(x, y):
  return torch.nn.functional.cosine_similarity(x, y, dim=-1)


@pytest.fixture
def x():
  torch.manual_seed(0)
  # [B, L, d] with norms spanning ~1e-2 .. ~1e2
  return torch.randn(4, 7, 16) * torch.logspace(-2, 2, 7).view(1, 7, 1)


def test_identity_when_off(x):
  assert torch.equal(rescale(None, None, x), x)


def test_fixed_norm_pins_all_radii(x):
  out = rescale(4.0, 4.0, x)
  assert torch.allclose(out.norm(dim=-1), torch.full(x.shape[:-1], 4.0),
                        atol=1e-5)
  assert torch.allclose(cos_to_input(x, out), torch.ones(x.shape[:-1]),
                        atol=1e-5)


def test_fixed_norm_rho_max_only_equivalent(x):
  # rho_min=None with rho_max=R degenerates to the tanh clamp with floor 0.
  out = rescale(None, 6.0, x)
  n = out.norm(dim=-1)
  # tanh saturates to 1.0 in float32, so large norms land exactly on rho_max
  assert (n > 0).all() and (n <= 6.0 + 1e-5).all()
  big = rescale(None, 6.0, 1e6 * torch.ones(1, 1, 16))
  assert torch.allclose(big.norm(dim=-1), torch.tensor([[6.0]]), atol=1e-4)


def test_soft_clamp_stays_in_range_and_monotone(x):
  lo, hi = 2.0, 6.0
  out = rescale(lo, hi, x)
  n_in = x.norm(dim=-1).flatten()
  n_out = out.norm(dim=-1).flatten()
  assert (n_out > lo).all() and (n_out <= hi + 1e-5).all()
  order = n_in.argsort()
  assert (n_out[order].diff() >= -1e-6).all()
  assert torch.allclose(cos_to_input(x, out), torch.ones(x.shape[:-1]),
                        atol=1e-5)


def test_soft_clamp_small_radius_limit():
  # tanh(r/range) ~ r/range for r << range: rho_eff ~ rho_min + r.
  lo, hi = 2.0, 6.0
  v = torch.zeros(1, 1, 16); v[..., 0] = 1e-3
  out = rescale(lo, hi, v)
  assert torch.allclose(out.norm(dim=-1), torch.tensor([[lo + 1e-3]]),
                        atol=1e-6)


def test_floor_only_shifts_norms(x):
  out = rescale(1.5, None, x)
  assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1) + 1.5, atol=1e-4)


def test_invalid_range_raises(x):
  with pytest.raises(ValueError):
    rescale(6.0, 2.0, x)


def test_2d_embedding_table():
  torch.manual_seed(0)
  table = torch.randn(12, 512)  # [V, d], sudoku-sized
  out = rescale(3.0, 3.0, table)
  assert out.shape == table.shape
  assert torch.allclose(out.norm(dim=-1), torch.full((12,), 3.0), atol=1e-5)
