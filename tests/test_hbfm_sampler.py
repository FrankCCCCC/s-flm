"""HBFMSampler (samplers.py): Euler-Maruyama sampler of the learned bridge SDE on
a product of Poincaré balls.

Two independent statistical checks that need no trained model:
  * SDE vs the exact bridge: integrating the SDE toward ONE fixed endpoint from
    the origin must reproduce the marginal law that
    HyperbolicHeatKernel.poincare_bridge_prod samples in closed form -- per
    factor, both the radius and the angular concentration <e, theta> -- also
    at curvature K != -1;
  * Bayes oracle: with the exact Poisson-kernel posterior under a fixed prior,
    the decoded endpoint of the learned bridge is distributed as the prior and
    the posterior is resolved (near one-hot) at the horizon.
Plus the sampler's contract on the real HyperbolicBoundaryFM (GPU): prefix
tokens pinned, integer output, NFE == steps, all velocity modes run.
"""
import math

import numpy as np
import pytest
import torch
from scipy import stats

import numeric.hyper_model as hyper_model
from numeric.horosphere import HorosphereGeometry
import samplers
from numeric.geo_bridge import Coordinate, GeoUtils, HyperbolicHeatKernel
from conftest import REPO_ROOT  # noqa: F401

torch.manual_seed(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
needs_gpu = pytest.mark.skipif(
  DEVICE != 'cuda', reason='HyperbolicDiT runs flash_attn kernels (CUDA only)')


def _unit(*shape):
  u = torch.randn(*shape, dtype=torch.float64)
  return u / u.norm(dim=-1, keepdim=True)


def _sde_to_endpoint(endpoint, dims, curvs, t_end, n_steps, n_traj):
  """Integrate the SDE toward one boundary point from the origin; returns z [n_traj, E]."""
  z = torch.zeros(n_traj, sum(dims), dtype=torch.float64)
  e = endpoint.expand(n_traj, -1)
  dt = torch.tensor(t_end / n_steps, dtype=torch.float64)
  for _ in range(n_steps):
    z = samplers.HBFMSampler.euler_maruyama_step(z, e, None, dt, dims, curvs)
  return z


@pytest.mark.parametrize('dims,curvs,t_end', [
  ([3], [-1.0], 1.0),
  ([3, 3], [-1.0, -4.0], 0.5),
  ([4], [-0.25], 2.0),
])
def test_sde_reproduces_the_exact_bridge_marginals(dims, curvs, t_end):
  torch.manual_seed(1)
  n = 4096
  E = sum(dims)
  endpoint = torch.cat([_unit(d) for d in dims])          # one word's boundary point
  z = _sde_to_endpoint(endpoint, dims, curvs, t_end, n_steps=1500, n_traj=n)
  rho_sde, u_sde = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(z, dims, curvs)
  # Exact bridge samples toward the same endpoint at the same heat time.
  ts = torch.full((n,), t_end, dtype=torch.float64)
  rho_ex, u_ex = HyperbolicHeatKernel.poincare_bridge_prod(
    ts, torch.zeros(n, 1, dtype=torch.long), endpoint[None],
    output_coord=Coordinate.HYPERBOLIC_POLAR,
    prod_factor_dim=dims, prod_factor_gaussian_curvature=curvs)
  rho_ex, u_ex = rho_ex[:, 0], u_ex[:, 0]
  off = 0
  for m, d in enumerate(dims):
    e_m = endpoint[off:off + d]
    cos_sde = (u_sde[:, off:off + d] @ e_m).numpy()
    cos_ex = (u_ex[:, off:off + d] @ e_m).numpy()
    off += d
    a, b = rho_sde[:, m].numpy(), rho_ex[:, m].numpy()
    # Two-sample KS on the radius and on the angular concentration.
    p_rho = stats.ks_2samp(a, b).pvalue
    p_cos = stats.ks_2samp(cos_sde, cos_ex).pvalue
    assert p_rho > 1e-3, (m, p_rho, np.quantile(a, [.1, .5, .9]), np.quantile(b, [.1, .5, .9]))
    assert p_cos > 1e-3, (m, p_cos, np.quantile(cos_sde, [.1, .5, .9]), np.quantile(cos_ex, [.1, .5, .9]))


class _OracleModel:
  """The Bayes-optimal HBFM for a vocabulary of boundary points under a fixed
  prior: forward returns log prior + horosphere log-densities (zero residual)."""

  class _Readout(hyper_model.HyperbolicModelBase):
    def __init__(self, emb, dims, curvs):
      super().__init__()
      self._emb = emb
      self.prod_factor_dim, self.prod_factor_gaussian_curvature = dims, curvs

    @property
    def word_embedding(self):
      return self._emb

    def model_forward(self, z, t):
      raise NotImplementedError

  def __init__(self, V, dims, curvs, log_prior, steps, length=1):
    torch.manual_seed(2)
    self.prod_factor_dim, self.prod_factor_gaussian_curvature = dims, curvs
    self.num_tokens = length
    self.device = torch.device('cpu')
    self.word_embedding = torch.randn(V, sum(dims), dtype=torch.float64)
    self.backbone = type('B', (), {'embed_dim': sum(dims)})()
    self.config = type('C', (), {'sampler': type('S', (), {'steps': steps})(),
                                 'algo': type('A', (), {
                                   'time_exp_rate': 1.0, 'time_conversion_mode': 'exp',
                                   'time_range_upper_bound': 1.0})()})()
    self.invert_time_convention = False
    self.log_prior = log_prior
    self._readout = self._Readout(self.word_embedding, dims, curvs)

  def _sigma_from_alphat(self, alpha_t):
    return -torch.log(alpha_t)

  def _clean_state(self, x):
    rhos = torch.full((*x.shape, len(self.prod_factor_dim)), 350.0, dtype=torch.float64)
    return GeoUtils.hyperbolic_polar_to_poincare_cartesian_prod(
      rhos, self.word_embedding[x], self.prod_factor_dim, self.prod_factor_gaussian_curvature)

  def forward(self, *, xt, sigma, context):
    rhos, thetas = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(
      xt, self.prod_factor_dim, self.prod_factor_gaussian_curvature)
    horo = HorosphereGeometry.compute_horosphere(
      theta=thetas, radius=rhos, word_embedding=self._readout.word_embedding,
      prod_factor_dim=self.prod_factor_dim,
      prod_factor_gaussian_curvature=self.prod_factor_gaussian_curvature)
    self.last_log_p = (horo + self.log_prior).log_softmax(-1)
    return self.last_log_p


def test_oracle_bridge_decodes_the_prior():
  torch.manual_seed(3)
  V, dims, curvs = 6, [3, 3], [-1.0, -2.0]
  prior = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.07, 0.03], dtype=torch.float64)
  model = _OracleModel(V, dims, curvs, prior.log(), steps=600)
  sampler = samplers.HBFMSampler(noise_removal='greedy', velocity='exact', use_float64=True,
                                 temperature=1.0, p_nucleus=1.0, top_k=-1,
                                 top_k_velocity=-1, t_max=12.0)
  n = 3000
  tokens, meta = samplers.run_sampler(sampler, model, n, num_steps=600)
  assert tokens.shape == (n, 1) and tokens.dtype == torch.long and meta['nfe'] == 600
  # Resolved at the horizon: the posterior the decode used is near one-hot.
  p_max = model.last_log_p.exp().max(-1).values
  assert (p_max > 0.99).float().mean() > 0.95
  # The endpoint law is the prior (chi-square on the decoded counts).
  counts = torch.bincount(tokens[:, 0], minlength=V).numpy()
  p = stats.chisquare(counts, prior.numpy() * n).pvalue
  assert p > 1e-3, (counts, prior * n, p)


def test_oracle_sample_velocity_also_decodes_the_prior():
  torch.manual_seed(4)
  V, dims, curvs = 5, [3], [-1.0]
  prior = torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05], dtype=torch.float64)
  model = _OracleModel(V, dims, curvs, prior.log(), steps=600)
  sampler = samplers.HBFMSampler(noise_removal='greedy', velocity='sample', use_float64=True,
                                 temperature=1.0, p_nucleus=1.0, top_k=-1,
                                 top_k_velocity=-1, t_max=20.0)
  n = 3000
  tokens, _ = samplers.run_sampler(sampler, model, n, num_steps=600)
  counts = torch.bincount(tokens[:, 0], minlength=V).numpy()
  assert stats.chisquare(counts, prior.numpy() * n).pvalue > 1e-3, counts


# ---------------------------------------------------------------------------
# Contract on the real model
# ---------------------------------------------------------------------------

BASE_OVERRIDES = [
  'algo=hbfm', 'model=tiny-hyperbolic-dit', 'sampler=hbfm', 'data=sudoku',
  'noise=log-linear', 'model.embed_dim=12', 'model.hidden_size=64', 'model.n_heads=4',
  'model.cond_dim=32', 'model.n_blocks=2', 'model.length=8', 'model.dropout=0.0',
  'algo.prod_factor_dim=3', 'algo.prod_factor_gaussian_curvature=-1.0',
  'algo.time_exp_rate=1.0',   # heat times in [1e-3, 6.9]: informative states, non-zero CE
  'loader.global_batch_size=4', 'loader.batch_size=4', 'trainer.devices=1',
  'trainer.num_nodes=1', 'sampler.steps=5', 'sampler.t_max=1.0',
]


class _NoMetrics:
  def __init__(self, **kwargs):
    pass

  def to(self, *args, **kwargs):
    return self


def _build_model(monkeypatch, overrides=()):
  import hydra
  import algo
  import dataloader
  import main  # noqa: F401
  import trainer_base
  monkeypatch.setattr(trainer_base.metrics, 'Metrics', _NoMetrics)
  with hydra.initialize(config_path='../configs', version_base=None):
    cfg = hydra.compose(config_name='config', overrides=BASE_OVERRIDES + list(overrides))
  torch.manual_seed(0)
  return algo.HyperbolicBoundaryFM(cfg, tokenizer=dataloader.get_tokenizer(cfg)).to(DEVICE)


@needs_gpu
@pytest.mark.parametrize('overrides', [
  [],
  ['sampler.velocity=sample'],
  ['sampler.top_k_velocity=2'],
  ['sampler.noise_removal=ancestral', 'sampler.temperature=0.5'],
  ['algo.time_conversion_mode=unif'],   # sampler.t_max=1.0 == time_range_upper_bound
  ['algo.time_conversion_mode=trunc_exp'],
])
def test_real_model_sampler_contract(monkeypatch, overrides):
  model = _build_model(monkeypatch, overrides)
  assert isinstance(model.sampler, samplers.HBFMSampler)
  B, P = 3, 2
  prefix = torch.randint(0, model.vocab_size, (B, P), device=model.device)
  lengths = torch.full((B,), P, device=model.device)
  tokens, meta = model.generate_samples(B, num_steps=5, prefix_tokens=prefix, prefix_lengths=lengths)
  assert tokens.shape == (B, model.num_tokens) and tokens.dtype == torch.long
  assert torch.equal(tokens[:, :P], prefix)
  assert (tokens >= 0).all() and (tokens < model.vocab_size).all()
  assert meta['nfe'] == 5
