"""SimpFLM: E-FLM with the word-embedding matrix replaced by the diagonal R*I_V.

Every "embedding" is a one-hot simplex vertex, so the Gaussian Euclidean flow
runs in the one-hot (logit) space R^V. The substitution has to be exact in two
places, and both are checked here against E-FLM's OWN code path evaluated with
the materialized diagonal (`SimpFLM._sc_embed_table`, feasible at the small
vocabularies used here, 10 GB at V = 50257):

  * the interpolant endpoint  `_clean_embeddings(x)` = rows of E   (algo)
  * the sampler drift         `R * p - x` = `p @ E - x`            (samplers)

The decode point carries over for a third reason: for ORTHOGONAL vertices the
impostor score <x_t, e_v> = R b_t z_v is LITERALLY Gaussian, so
`alpha_star_euclidean(V, embed_norm=R)`'s union bound applies verbatim rather
than through E-FLM's sub-Gaussian estimate of random directions in d dimensions.
Checked by Monte Carlo on the impostor event the bound actually controls — the
target coordinate's own noise is neglected in that derivation (for E-FLM too),
so the end-to-end argmax hit rate sits a few points under 1 - delta.
"""
import math
import types

import omegaconf
import pytest
import torch
import torch.nn.functional as F

from conftest import REPO_ROOT  # noqa: F401  (sys.path setup)
import algo
import samplers
from noise_schedules import alpha_star_euclidean, get_noise

V = 16
L = 5
B = 4


# ── stubs ───────────────────────────────────────────────────────

def make_algo(vocab_size=V, rho=1.0, invert=False):
  """SimpFLM carrying only the attributes the flow-side methods read.

  Bypasses the Lightning constructor (which would build a 170M-param DiT); the
  methods under test (`_clean_embeddings`, the inherited `q_xt` / `_lerp` /
  `nll_per_token`, `vertex_radius`, `_sc_embed_table`) touch nothing else.
  """
  m = object.__new__(algo.SimpFLM)
  m.vocab_size = vocab_size
  m.rho_min = rho
  m.rho_max = rho
  m.invert_time_convention = invert
  m.snr_weighted_ce = False
  m.eps = 1e-6
  m.config = omegaconf.OmegaConf.create(
    {'algo': {'slerp_precision': 'float32'}, 'noise': {'adaptive': False},
     'sampler': {'predictor': 'simpflm'}})
  m._device = torch.device('cpu')  # LightningModule.device is read-only
  return m


def make_sampler(cls=samplers.SimpFLMSampler, **overrides):
  kwargs = dict(
    noise_removal='greedy', velocity='exact', use_float64=False,
    lerp_float64=False, eps=1e-6, temperature=1.0, p_nucleus=1.0,
    top_k=-1, top_k_velocity=-1, invert_time_convention=False,
    prior_cov=1.0)
  kwargs.update(overrides)
  return cls(**kwargs)


class StubModel:
  """Minimal duck-type of the trainer for the sampler loop.

  `forward` returns log-probs peaked on a fixed target sequence, so a converged
  flow must decode exactly that sequence.
  """

  def __init__(self, target, rho=1.0, vocab_size=V, steps=8):
    self.target = target                 # [B, L] int
    self.vocab_size = vocab_size
    self.num_tokens = target.shape[1]
    self.device = torch.device('cpu')
    self._algo = make_algo(vocab_size=vocab_size, rho=rho)
    self.noise = get_noise(omegaconf.OmegaConf.create({
      'noise': {'type': 'autonomous', 'eps': 1e-3, 'tau_max': 1.834,
                'alpha_min': None, 'alpha_max': None, 'adaptive': False}}))
    self.config = omegaconf.OmegaConf.create(
      {'sampler': {'steps': steps, 'eta': 0.0, 'gt_method': 'linear'}})

  # the two hooks the sampler reaches into the algo for
  vertex_radius = property(lambda self: self._algo.vertex_radius)

  def _clean_embeddings(self, tokens):
    return self._algo._clean_embeddings(tokens)

  def _sc_embed_table(self):
    return self._algo._sc_embed_table()

  def _sigma_from_alphat(self, alpha_t):
    return -torch.log(alpha_t)

  def forward(self, xt, sigma, context=None):
    del xt, sigma, context
    return F.one_hot(self.target, self.vocab_size).float().mul(
      8.0).log_softmax(-1)


# ── the diagonal itself ─────────────────────────────────────────

@pytest.mark.parametrize('rho', [0.5, 1.0, 2.5])
def test_embed_table_is_the_scaled_diagonal(rho):
  m = make_algo(rho=rho)
  E = m._sc_embed_table()
  assert E.shape == (V, V)
  assert torch.allclose(E, rho * torch.eye(V))
  assert m.vertex_radius == pytest.approx(rho)
  # every row is a one-hot vertex of norm R, and they are mutually orthogonal
  assert torch.allclose(E.norm(dim=-1), torch.full((V,), rho))
  assert torch.allclose(E @ E.T, rho ** 2 * torch.eye(V), atol=1e-6)


def test_radius_one_is_the_plain_one_hot():
  m = make_algo(rho=1.0)
  x0 = torch.randint(0, V, (B, L))
  assert torch.equal(m._clean_embeddings(x0), F.one_hot(x0, V).float())


@pytest.mark.parametrize('rho', [0.5, 1.0, 2.5])
def test_clean_embeddings_are_rows_of_the_diagonal(rho):
  """The algo-side half of the substitution, against the materialized E."""
  m = make_algo(rho=rho)
  x0 = torch.randint(0, V, (B, L))
  assert torch.allclose(m._clean_embeddings(x0),
                        F.embedding(x0, m._sc_embed_table()))


# ── the inherited E-FLM interpolant ─────────────────────────────

@pytest.mark.parametrize('invert', [False, True])
def test_q_xt_is_the_euclidean_lerp(invert):
  """x_t = (1 - b) R 1_{x0} + b z, b read off alpha_t in either convention."""
  torch.manual_seed(0)
  m = make_algo(rho=1.7, invert=invert)
  x0 = torch.randint(0, V, (256, L))
  alpha_t = torch.rand(256, 1)
  b = (alpha_t if invert else 1 - alpha_t).reshape(-1, 1, 1)

  xt = m.q_xt(x0, alpha_t, use_pure_noise=False)
  assert xt.shape == (256, L, V)
  z = (xt - (1 - b) * m._clean_embeddings(x0)) / b  # residual is b * z
  assert z.mean().abs() < 0.02
  assert z.std() == pytest.approx(1.0, abs=0.02)


def test_q_xt_pure_noise_forgets_x0():
  torch.manual_seed(0)
  m = make_algo()
  x0 = torch.randint(0, V, (512, L))
  xt = m.q_xt(x0, torch.zeros(512, 1), use_pure_noise=True)
  assert (xt.argmax(-1) == x0).float().mean() == pytest.approx(
    1.0 / V, abs=0.02)


def test_q_xt_keeps_prompt_positions_clean():
  torch.manual_seed(0)
  m = make_algo(rho=3.0)
  x0 = torch.randint(0, V, (B, L))
  valid = torch.ones(B, L, dtype=torch.bool)
  valid[:, :2] = False  # prompt
  xt = m.q_xt(x0, torch.full((B, 1), 0.5), use_pure_noise=False,
              valid_tokens=valid)
  clean = m._clean_embeddings(x0)
  assert torch.equal(xt[:, :2], clean[:, :2])
  assert not torch.equal(xt[:, 2:], clean[:, 2:])


def test_loss_is_plain_cross_entropy():
  m = make_algo()
  x0 = torch.randint(0, V, (B, L))
  log_p = torch.randn(B, L, V).log_softmax(-1)
  loss = m.nll_per_token(log_p, xt=None, x0=x0, alpha_t=None, dalpha_t=None)
  assert torch.allclose(loss, F.cross_entropy(
    log_p.reshape(-1, V), x0.reshape(-1), reduction='none').reshape(B, L))


def test_validate_rejects_the_unsupported_eflm_knobs():
  m = make_algo()
  m.self_conditioning = False
  m.renormalize_weights = False
  m.config = omegaconf.OmegaConf.create(
    {'noise': {'adaptive': False}, 'model': {'type': 'sphere-dit'},
     'sampler': {'predictor': 'simpflm'}})
  with pytest.raises(ValueError, match='flm-dit'):
    m._validate_configuration()
  m.config.model.type = 'flm-dit'
  m._validate_configuration()  # now clean
  m.self_conditioning = True
  with pytest.raises(ValueError, match='self_conditioning'):
    m._validate_configuration()
  m.self_conditioning = False
  m.renormalize_weights = True
  with pytest.raises(ValueError, match='renormalize'):
    m._validate_configuration()
  # RHO is the radius of the simplex sphere: a RANGE has nothing to clamp.
  m.renormalize_weights = False
  for lo, hi in [(0.5, 2.0), (None, 1.0), (1.0, None), (None, None)]:
    m.rho_min, m.rho_max = lo, hi
    with pytest.raises(ValueError, match='rho_min == algo.rho_max'):
      m._validate_configuration()
  # R = 0 would put every vertex on the origin: no signal about x0.
  m.rho_min = m.rho_max = 0.0
  with pytest.raises(ValueError, match='strictly positive'):
    m._validate_configuration()
  # and the diagonal only makes sense with the sampler that knows about it
  m.rho_min = m.rho_max = 1.0
  m.config.sampler.predictor = 'eflm'
  with pytest.raises(ValueError, match='sampler=simpflm'):
    m._validate_configuration()


# ── the sampler-side substitution ───────────────────────────────

@pytest.mark.parametrize('top_k_velocity', [-1, 1, 3])
@pytest.mark.parametrize('rho', [1.0, 4.0])
def test_velocity_matches_eflm_evaluated_on_the_diagonal(top_k_velocity, rho):
  """SimpFLMSampler's `R * p - x` IS EFLMSampler's `p @ E - x` at E = R I_V."""
  torch.manual_seed(0)
  model = StubModel(torch.zeros(B, L, dtype=torch.long), rho=rho)
  state = types.SimpleNamespace(xt=torch.randn(B, L, V), start_idx=0)
  log_p = torch.randn(B, L, V).log_softmax(-1)

  x_s, v_s = make_sampler(top_k_velocity=top_k_velocity)._velocity(
    model, state, log_p)
  x_e, v_e = make_sampler(cls=samplers.EFLMSampler,
                          top_k_velocity=top_k_velocity)._velocity(
    model, state, log_p)  # goes through _get_embed_table -> the real diagonal

  assert torch.allclose(x_s, x_e)
  assert torch.allclose(v_s, v_e, atol=1e-5)


def test_sampled_velocity_targets_a_single_vertex():
  """velocity='sample' aims at one vertex, drawn from the top-k support."""
  torch.manual_seed(0)
  log_p = torch.randn(B, L, V).log_softmax(-1)
  p = make_sampler(velocity='sample', top_k_velocity=2)._target_probs(log_p)
  assert torch.equal(p.sum(-1), torch.ones(B, L))
  assert (p.max(-1).values == 1.0).all()
  top2 = log_p.topk(2, dim=-1).indices
  assert (p.argmax(-1).unsqueeze(-1) == top2).any(-1).all()


def test_unknown_velocity_mode_raises():
  with pytest.raises(ValueError, match='velocity'):
    make_sampler(velocity='bogus')._target_probs(
      torch.randn(1, 1, V).log_softmax(-1))


# ── sampler loop ────────────────────────────────────────────────

def test_sampler_integrates_in_vocab_space_and_decodes_tokens():
  torch.manual_seed(0)
  target = torch.randint(0, V, (B, L))
  model = StubModel(target, steps=8)
  sampler = make_sampler(top_k_velocity=1)

  state = sampler.init_state(model, B, num_steps=8, eps=1e-5)
  assert state.xt.shape == (B, L, V)  # the flow lives in R^V, not R^d

  # the flow itself does the work: after the 7 integration steps (the 8th
  # decodes) the continuous state's projection onto the TARGET vertex has grown
  # from ~0 to R(1 - b*). The truncation leaves b* = exp(-tau_max) = 0.16 of
  # noise on purpose -- that residual is what the greedy last step removes, so
  # the pre-decode argmax is only *mostly* right, not exact.
  for _ in range(7):
    state = sampler.step(model, state)
  assert state.xt.shape == (B, L, V)
  contraction = (1 - (1 - math.exp(-1.834 / 8))) ** 7   # ~0.20
  tgt_coord = state.xt.gather(-1, target.unsqueeze(-1)).squeeze(-1)
  assert tgt_coord.mean().item() == pytest.approx(1 - contraction, abs=0.15)
  assert (state.xt.argmax(-1) == target).float().mean().item() > 0.8

  xt, meta = samplers.run_sampler(sampler, model, B, num_steps=8, eps=1e-5)
  assert xt.shape == (B, L) and xt.dtype == torch.int64
  assert meta['nfe'] == 8
  assert torch.equal(xt, target)


def test_sampling_never_materializes_the_diagonal():
  """The reason SimpFLMSampler exists: R * I_V is 10 GB at V = 50257.

  The whole sampling loop must run without ever calling `_sc_embed_table`.
  The contrast case shows the assertion has teeth -- E-FLM's inherited
  `_velocity`, which SimpFLMSampler overrides, does reach for it.
  """
  torch.manual_seed(0)
  target = torch.randint(0, V, (B, L))

  def boom():
    raise AssertionError('the V x V diagonal was materialized')

  model = StubModel(target, steps=6)
  model._sc_embed_table = boom
  xt, _ = samplers.run_sampler(make_sampler(top_k_velocity=1), model, B,
                               num_steps=6, eps=1e-5)
  assert torch.equal(xt, target)

  state = types.SimpleNamespace(xt=torch.randn(B, L, V), start_idx=0)
  with pytest.raises(AssertionError, match='materialized'):
    make_sampler(cls=samplers.EFLMSampler, top_k_velocity=1)._velocity(
      model, state, torch.randn(B, L, V).log_softmax(-1))


def test_sampler_respects_a_prefix():
  torch.manual_seed(0)
  target = torch.randint(0, V, (B, L))
  model = StubModel(target, steps=4)
  prefix = torch.randint(0, V, (B, 2))
  lengths = torch.full((B,), 2, dtype=torch.long)

  xt, _ = samplers.run_sampler(
    make_sampler(top_k_velocity=1), model, B, num_steps=4, eps=1e-5,
    prefix_tokens=prefix, prefix_lengths=lengths)
  assert torch.equal(xt[:, :2], prefix)


# ── decode point / schedule composition ─────────────────────────

N_MC = 4096
VOCAB_MC = 2000
DELTA = 0.1


def impostor_failure_rate(m, alpha, n=N_MC):
  """P(max_{v != x0} <x_t, e_v> > the target's mean score) at signal level alpha.

  Exactly the event `alpha_star_euclidean` bounds: <x_t, e_v> = R x_t[v] is
  R (1 - alpha) z_v for the impostors and has mean alpha R^2 at v = x0.
  """
  x0 = torch.randint(0, m.vocab_size, (n, 1))
  xt = m.q_xt(x0, torch.full((n, 1), alpha), use_pure_noise=False)
  impostor = xt.scatter(-1, x0.unsqueeze(-1), float('-inf')).max(-1).values
  beaten = m.vertex_radius * impostor > alpha * m.vertex_radius ** 2
  return beaten.float().mean().item(), (xt.argmax(-1) == x0).float().mean()


@pytest.mark.parametrize('rho', [0.5, 1.0, 4.0])
def test_decode_point_holds_the_impostor_union_bound(rho):
  """At alpha*(R) the impostors clear the target's mean w.p. >= 1 - delta."""
  torch.manual_seed(0)
  alpha = alpha_star_euclidean(VOCAB_MC, delta=DELTA, embed_norm=rho)
  m = make_algo(vocab_size=VOCAB_MC, rho=rho)

  fail, _ = impostor_failure_rate(m, alpha)
  assert 0.0 < fail <= DELTA          # holds, and is not vacuous
  fail_lo, _ = impostor_failure_rate(m, alpha * 0.5)
  assert fail_lo > DELTA              # halfway to the noise end it is gone


def test_decode_point_is_scale_invariant():
  """Only noise/R enters the bound, so alpha*(R) buys the same decode quality
  at every R — the property that makes RHO a pure reparameterization of the
  truncation."""
  hits = []
  for rho in (0.5, 1.0, 4.0):
    torch.manual_seed(0)
    m = make_algo(vocab_size=VOCAB_MC, rho=rho)
    alpha = alpha_star_euclidean(VOCAB_MC, delta=DELTA, embed_norm=rho)
    hits.append(impostor_failure_rate(m, alpha)[1])
  assert torch.allclose(torch.stack(hits), hits[0])
  # the neglected target-coordinate noise costs a few points off 1 - delta.
  assert 0.8 < hits[0] < 1 - DELTA


def test_autonomous_horizon_is_the_decode_point():
  """tau*(R) = -log(1 - alpha*(R)) = log(1 + C/R); R=1 is the script default."""
  tau = alpha_star_euclidean(50257, embed_norm=1.0, auto_clock=True)
  assert tau == pytest.approx(1.834, abs=1e-3)
  assert tau == pytest.approx(
    -math.log(1 - alpha_star_euclidean(50257, embed_norm=1.0)), abs=1e-6)


def test_trunc_auto_ada_compose_into_one_schedule():
  """SimpFLM reads b_t off `self.noise`, so the three wrappers stack."""
  import noise_schedules
  cfg = omegaconf.OmegaConf.create({
    'noise': {'type': 'autonomous', 'eps': 1e-3, 'tau_max': 1.834,
              'alpha_min': None, 'alpha_max': 0.84, 'adaptive': True,
              'adaptive_buffer_size': 16, 'adaptive_refit_every': 2,
              'adaptive_n_grid': 32, 'adaptive_n_knots': 5,
              'adaptive_spline_degree': 3, 'adaptive_ridge_alpha': 1e-3,
              'adaptive_uniform_mix': 1e-3, 'adaptive_warmup_steps': 0,
              'adaptive_ema': 0.0, 'adaptive_plot_profile': False,
              'adaptive_plot_dir': 'x', 'adaptive_log_importance': False},
    'loader': {'global_batch_size': 16},
    'trainer': {'max_steps': 10}})
  noise = get_noise(cfg)
  assert isinstance(noise, noise_schedules.AdaptiveSchedule)
  base = noise.base_schedule
  assert isinstance(base, noise_schedules.TruncatedScheduleWrapper)
  assert isinstance(base.base_schedule, noise_schedules.Autonomous)
  # truncated at the decode point: the signal level never exceeds alpha_max.
  assert noise.alpha_t(torch.linspace(0, 1, 32)).max().item() <= 0.84 + 1e-6


# ── wiring ──────────────────────────────────────────────────────

def test_get_sampler_resolves_simpflm():
  cfg = omegaconf.OmegaConf.create({
    'sampler': {'predictor': 'simpflm', 'steps': 8, 'noise_removal': 'greedy',
                'velocity': 'exact', 'use_float64': False, 'p_nucleus': 1.0,
                'top_k': -1, 'top_k_velocity': 1, 'temperature': 1.0},
    'algo': {'name': 'simpflm', 'slerp_precision': 'float32', 'eps': 1e-6,
             'invert_time_convention': False, 'prior_cov': 1.0}})
  s = samplers.get_sampler(cfg)
  assert isinstance(s, samplers.SimpFLMSampler)
  assert s.top_k_velocity == 1


def test_eflm_keeps_its_own_embedding_table():
  """SimpFLM must ADD a diagonal table, not take E-FLM's away.

  Both classes have to override `SelfConditioning._sc_embed_table` (which only
  raises), and with distinct implementations -- E-FLM's sampler reaches through
  `_get_embed_table` -> `model._sc_embed_table()` on every step, so an E-FLM
  that inherited the stub would crash at sampling time and nothing else in the
  suite would notice.
  """
  base = algo.SelfConditioning._sc_embed_table
  assert algo.EFLM._sc_embed_table is not base
  assert algo.SimpFLM._sc_embed_table is not base
  assert algo.SimpFLM._sc_embed_table is not algo.EFLM._sc_embed_table
  assert algo.SimpFLM._clean_embeddings is not algo.EFLM._clean_embeddings


def test_main_dispatches_simpflm():
  import inspect
  import main
  src = inspect.getsource(main.main)
  assert "config.algo.name == 'simpflm'" in src
  assert 'algo.SimpFLM' in src
