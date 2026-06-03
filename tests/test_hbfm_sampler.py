"""HBFMSampler init/step on-manifold contract (ARCH Section 6 / Section 9).

`get_sampler(config)` must return an `HBFMSampler` for
`config.sampler.predictor == 'hbfm'`.  `init_state` seeds an in-ball point from
the free kernel with a decreasing heat-time grid of length `steps + 1`; a
non-final `step` keeps the state in the ball / finite and increments `nfe`; the
terminal step decodes to int tokens `(B, L)`.

Per ARCH Section 9 we assert only the on-manifold / finite / nfe / schedule
invariants -- NOT the exact trajectory.  Written against the interface before
implementation; EXPECTED to FAIL until `samplers.HBFMSampler` and the
`get_sampler` `predictor == 'hbfm'` branch exist.  CPU-only, tiny dims.
"""
import pytest
import torch

from conftest import (_compose_config, _make_hbfm, SMALL_D, BINARY_D,
                      HBFM_T_MIN, HBFM_T_MAX)

STEPS = 4  # tiny sampling budget


@pytest.fixture
def config_sampler_d8():
  return _compose_config(SMALL_D, sampler_predictor='hbfm')


@pytest.fixture
def config_sampler_d2():
  return _compose_config(BINARY_D, sampler_predictor='hbfm')


@pytest.fixture
def hbfm_sampler_model_d8(config_sampler_d8, tokenizer):
  return _make_hbfm(config_sampler_d8, tokenizer)


@pytest.fixture
def hbfm_sampler_model_d2(config_sampler_d2, tokenizer):
  return _make_hbfm(config_sampler_d2, tokenizer)


# --------------------------------------------------------------------------
# get_sampler dispatch.
# --------------------------------------------------------------------------
def test_get_sampler_returns_hbfm_sampler(config_sampler_d8):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d8)
  assert isinstance(sampler, samplers.HBFMSampler)


# --------------------------------------------------------------------------
# init_state: in-ball seed + decreasing schedule of length steps+1.
# --------------------------------------------------------------------------
def test_init_state_xt_is_in_ball_general_d(hbfm_sampler_model_d8, config_sampler_d8):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d8)
  state = sampler.init_state(hbfm_sampler_model_d8, num_samples=2,
                             num_steps=STEPS)
  assert state.xt.norm(dim=-1).max().item() < 1.0


def test_init_state_xt_shape_general_d(hbfm_sampler_model_d8, config_sampler_d8):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d8)
  state = sampler.init_state(hbfm_sampler_model_d8, num_samples=2,
                             num_steps=STEPS)
  L = hbfm_sampler_model_d8.num_tokens
  assert tuple(state.xt.shape) == (2, L, SMALL_D)


def test_init_state_xt_is_in_ball_binary_d(hbfm_sampler_model_d2, config_sampler_d2):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d2)
  state = sampler.init_state(hbfm_sampler_model_d2, num_samples=2,
                             num_steps=STEPS)
  assert state.xt.norm(dim=-1).max().item() < 1.0


def test_init_state_schedule_length_is_steps_plus_one(hbfm_sampler_model_d8,
                                                      config_sampler_d8):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d8)
  state = sampler.init_state(hbfm_sampler_model_d8, num_samples=2,
                             num_steps=STEPS)
  assert state.t_schedule.shape[0] == STEPS + 1


def test_init_state_schedule_decreasing_from_t_max(hbfm_sampler_model_d8,
                                                   config_sampler_d8):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d8)
  state = sampler.init_state(hbfm_sampler_model_d8, num_samples=2,
                             num_steps=STEPS)
  diffs = state.t_schedule[1:] - state.t_schedule[:-1]
  assert (diffs <= 0).all().item()


def test_init_state_schedule_starts_at_t_max(hbfm_sampler_model_d8,
                                             config_sampler_d8):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d8)
  state = sampler.init_state(hbfm_sampler_model_d8, num_samples=2,
                             num_steps=STEPS)
  assert pytest.approx(float(state.t_schedule[0]), abs=1e-5) == hbfm_sampler_model_d8.t_max


def test_init_state_nfe_starts_at_zero(hbfm_sampler_model_d8, config_sampler_d8):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d8)
  state = sampler.init_state(hbfm_sampler_model_d8, num_samples=2,
                             num_steps=STEPS)
  assert state.nfe == 0


# --------------------------------------------------------------------------
# step: a non-final step stays in the ball / finite and increments nfe.
# --------------------------------------------------------------------------
def test_non_final_step_keeps_xt_in_ball(hbfm_sampler_model_d8, config_sampler_d8):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d8)
  state = sampler.init_state(hbfm_sampler_model_d8, num_samples=2,
                             num_steps=STEPS)
  state = sampler.step(hbfm_sampler_model_d8, state)
  assert state.xt.norm(dim=-1).max().item() < 1.0


def test_non_final_step_keeps_xt_in_ball_binary_d(hbfm_sampler_model_d2,
                                                  config_sampler_d2):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d2)
  state = sampler.init_state(hbfm_sampler_model_d2, num_samples=2,
                             num_steps=STEPS)
  state = sampler.step(hbfm_sampler_model_d2, state)
  assert state.xt.norm(dim=-1).max().item() < 1.0
  assert torch.isfinite(state.xt).all().item()


def test_non_final_step_keeps_xt_finite(hbfm_sampler_model_d8, config_sampler_d8):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d8)
  state = sampler.init_state(hbfm_sampler_model_d8, num_samples=2,
                             num_steps=STEPS)
  state = sampler.step(hbfm_sampler_model_d8, state)
  assert torch.isfinite(state.xt).all().item()


def test_non_final_step_increments_nfe(hbfm_sampler_model_d8, config_sampler_d8):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d8)
  state = sampler.init_state(hbfm_sampler_model_d8, num_samples=2,
                             num_steps=STEPS)
  state = sampler.step(hbfm_sampler_model_d8, state)
  assert state.nfe == 1


def test_non_final_step_advances_step_idx(hbfm_sampler_model_d8, config_sampler_d8):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d8)
  state = sampler.init_state(hbfm_sampler_model_d8, num_samples=2,
                             num_steps=STEPS)
  state = sampler.step(hbfm_sampler_model_d8, state)
  assert state.step_idx == 1


# --------------------------------------------------------------------------
# terminal step decodes to int tokens (B, L) and marks done.
# --------------------------------------------------------------------------
def _run_to_last(sampler, model):
  state = sampler.init_state(model, num_samples=2, num_steps=STEPS)
  for _ in range(STEPS):
    state = sampler.step(model, state)
  return state


def test_terminal_step_decodes_to_int_tokens(hbfm_sampler_model_d8,
                                             config_sampler_d8):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d8)
  state = _run_to_last(sampler, hbfm_sampler_model_d8)
  L = hbfm_sampler_model_d8.num_tokens
  assert tuple(state.xt.shape) == (2, L)
  assert not state.xt.is_floating_point()


def test_terminal_step_marks_done(hbfm_sampler_model_d8, config_sampler_d8):
  import samplers
  sampler = samplers.get_sampler(config_sampler_d8)
  state = _run_to_last(sampler, hbfm_sampler_model_d8)
  assert state.done is True
