"""Tests for the geometry-specific truncation bounds alpha_star_*.

The sphere bound must reproduce Table 4 of the hyperspherical-flows
paper (papers/Language Modeling with Hyperspherical Flows.pdf); the
Euclidean / hyperbolic analogs are checked for the script defaults
and for the qualitative behavior the derivations predict.
"""
import math

import pytest
import torch

from noise_schedules import (
  alpha_star_sphere, alpha_star_euclidean, alpha_star_hyperbolic,
  LogLinear, TruncatedScheduleWrapper)


# ── sphere: reproduce paper Table 4 ─────────────────────────────

@pytest.mark.parametrize('vocab,dim,delta,expected', [
  (12, 256, 0.1, 0.132),
  (12, 512, 0.1, 0.093),    # Sudoku value used by sfm_truncated.sh
  (12, 512, 0.01, 0.111),
  (12, 768, 0.1, 0.076),
  (50000, 768, 0.1, 0.121),  # TinyStories/OWT value (sfm scripts)
  (50000, 768, 0.01, 0.131),
  (100000, 1024, 0.1, 0.108),
])
def test_sphere_reproduces_paper_table4(vocab, dim, delta, expected):
  assert alpha_star_sphere(vocab, dim, delta) == pytest.approx(
    expected, abs=5e-4)


# ── script defaults (lock the values baked into scripts/train) ──

def test_script_default_values():
  # EFLM: ngpt init (||e||~=1), N(0, I) prior.
  assert alpha_star_euclidean(12) == pytest.approx(0.767, abs=1e-3)
  assert alpha_star_euclidean(50257) == pytest.approx(0.840, abs=1e-3)
  # HFLM Sudoku: d=512, init=hyperbolic (std 0.3).
  assert alpha_star_hyperbolic(12, 512) == pytest.approx(
    0.624, abs=1e-3)
  # HFLM TinyStories: d=768, init=ngpt (std 1/sqrt(d)).
  assert alpha_star_hyperbolic(
    50257, 768, embed_std=1 / math.sqrt(768)) == pytest.approx(
      0.979, abs=1e-3)
  # HFLM OWT: d=768, init=hyperbolic (std 0.3).
  assert alpha_star_hyperbolic(50257, 768) == pytest.approx(
    0.608, abs=1e-3)


# ── qualitative behavior of the analogs ─────────────────────────

def test_all_bounds_in_unit_interval():
  for v, d in [(12, 512), (50257, 768), (100, 64)]:
    assert 0 < alpha_star_sphere(v, d) < 1
    assert 0 < alpha_star_euclidean(v) < 1
    assert 0 < alpha_star_hyperbolic(v, d) < 1


def test_euclidean_scale_matched_noise_approaches_sphere():
  # With noise scale-matched to the embeddings (||noise|| ~= ||e||,
  # i.e. per-coord std 1/sqrt(d)), the Euclidean bound lands within
  # a small constant of the sphere bound: a/(1+a) vs (2/pi)asin(a).
  for v, d in [(12, 512), (50000, 768)]:
    eu = alpha_star_euclidean(v, noise_std=1 / math.sqrt(d))
    sp = alpha_star_sphere(v, d)
    assert sp < eu < 2 * sp


def test_euclidean_monotonic_in_scale_ratio():
  # More noise (or smaller embeddings) pushes the Voronoi-collapse
  # point toward the clean end.
  assert (alpha_star_euclidean(12, noise_std=2.0)
          > alpha_star_euclidean(12, noise_std=1.0))
  assert (alpha_star_euclidean(12, embed_norm=2.0)
          < alpha_star_euclidean(12, embed_norm=1.0))


def test_hyperbolic_monotonic_in_clean_radius():
  # Larger clean-embedding radius -> target separates from the
  # origin-hugging noise sooner -> smaller truncation bound.
  small = alpha_star_hyperbolic(12, 512, embed_std=0.1)
  large = alpha_star_hyperbolic(12, 512, embed_std=0.5)
  assert large < small


def test_hyperbolic_far_from_sphere_bound():
  # The documented failure: applying the sphere bound (0.093) to
  # HFLM collapses it. The hyperbolic bound must be much larger.
  assert alpha_star_hyperbolic(12, 512) > 5 * alpha_star_sphere(
    12, 512)


# ── integration: bounds plug into TruncatedScheduleWrapper ──────

def test_bounds_drive_truncated_schedule():
  alpha_max = alpha_star_hyperbolic(12, 512)
  sched = TruncatedScheduleWrapper(
    LogLinear(eps=1e-3), alpha_min=0.0, alpha_max=alpha_max,
    eps=1e-3)
  t = torch.linspace(0, 1, 101)
  alpha = sched.alpha_t(t)
  assert alpha.max().item() == pytest.approx(alpha_max, abs=1e-6)
  assert alpha.min().item() >= 0
