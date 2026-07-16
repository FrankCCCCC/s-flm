"""Regression tests for the AdaptiveSchedule refit trigger.

The first refit used to require `end >= buffer_size` (true only on the
exact wrap-around step) to coincide with `step % refit_every == 0`.
With buffer_size a multiple of the per-step batch (every production
config: buffer = fill_steps * global_batch), the wrap step is
`fill_steps - 1 (mod fill_steps)` while refit steps are multiples of
refit_every — disjoint whenever fill_steps divides refit_every, so the
schedule never activated. The fix tracks the total samples seen.
"""
import pytest
import torch

from noise_schedules import AdaptiveSchedule, LogLinear


def _run(buffer_size, refit_every, warmup, batch, steps, seed=0,
         log_importance=False, loss_fn=None):
  torch.manual_seed(seed)
  sched = AdaptiveSchedule(
    LogLinear(eps=1e-3), buffer_size=buffer_size,
    refit_every=refit_every, n_grid=100, n_knots=10,
    spline_degree=3, ridge_alpha=1e-3, uniform_mix=1e-3,
    max_steps=steps, warmup_steps=warmup, ema=0.9,
    log_importance=log_importance)
  for step in range(steps):
    t = torch.rand(batch)
    if loss_fn is None:
      loss = 2 * t + 0.1 * torch.rand(batch)  # increasing loss profile
    else:
      loss = loss_fn(t)
    sched.record_time_loss_pair(t, loss, step)
  return sched


@pytest.mark.parametrize('buffer,refit,warmup,batch,steps', [
  (50 * 256, 50, 0, 256, 101),      # sudoku *_truncated_adaptive.sh
  (50 * 512, 50, 1000, 512, 1101),  # owt sfm.sh (+ default warmup)
  (256 * 500 // 10, 500, 1000, 256, 1600),  # log-linear-adaptive.yaml
])
def test_refit_fires_with_production_knobs(
    buffer, refit, warmup, batch, steps):
  sched = _run(buffer, refit, warmup, batch, steps)
  assert bool(sched.has_schedule)
  assert int(sched.refit_count) >= 1


def test_first_refit_at_first_refit_step_after_fill():
  # Buffer fills during step 49; first multiple of refit_every with a
  # full buffer is step 50 -> exactly one refit by step 51.
  sched = _run(50 * 256, 50, 0, 256, 51)
  assert int(sched.refit_count) == 1


def test_no_refit_before_buffer_filled():
  sched = _run(50 * 256, 50, 0, 256, 49)
  assert int(sched.refit_count) == 0
  assert not bool(sched.has_schedule)


def test_warmup_delays_recording():
  sched = _run(50 * 256, 50, 1000, 256, 500)
  assert int(sched.n_seen) == 0
  assert int(sched.refit_count) == 0


def test_state_dict_schema_unchanged():
  # n_seen is non-persistent so checkpoints written before/after the
  # refit-trigger fix stay interchangeable under strict loading, and
  # a resumed run with a schedule keeps refitting via has_schedule.
  sched = _run(50 * 64, 50, 0, 64, 101)
  assert 'n_seen' not in sched.state_dict()
  fresh = AdaptiveSchedule(
    LogLinear(eps=1e-3), buffer_size=50 * 64, refit_every=50,
    n_grid=100, n_knots=10, spline_degree=3, ridge_alpha=1e-3,
    uniform_mix=1e-3, max_steps=101, warmup_steps=0, ema=0.9)
  fresh.load_state_dict(sched.state_dict())
  assert bool(fresh.has_schedule)
  t = torch.rand(64)
  fresh.record_time_loss_pair(t, 2 * t, step=150)  # 150 % 50 == 0
  assert int(fresh.refit_count) == int(sched.refit_count) + 1


def test_log_importance_spreads_over_exponential_ramp():
  # An exponentially decaying loss profile (HFLM's ramp geometry) has
  # |dL/dt| concentrated near t=1 but near-constant |d log L/dt|.
  # Linear importance therefore warps the schedule hard; log importance
  # should stay close to the base schedule (near-uniform importance).
  ramp = lambda t: torch.exp(6 * (t - 1))  # e^-6 .. 1
  lin = _run(50 * 64, 50, 0, 64, 101, loss_fn=ramp)
  log = _run(50 * 64, 50, 0, 64, 101, log_importance=True, loss_fn=ramp)
  assert bool(lin.has_schedule) and bool(log.has_schedule)
  t = torch.linspace(0, 1, 201)
  base = lin.base_schedule.alpha_t(t)
  dev_lin = (lin.alpha_t(t) - base).abs().mean()
  dev_log = (log.alpha_t(t) - base).abs().mean()
  assert dev_log < dev_lin


def test_adapted_schedule_stays_within_base_range():
  # alpha_t after refits must remain inside the base schedule's range
  # (the remap only reallocates time, it does not extend the range) —
  # this is what keeps truncation (alpha_max) intact under adaptation.
  sched = _run(50 * 64, 50, 0, 64, 101)
  assert bool(sched.has_schedule)
  t = torch.linspace(0, 1, 201)
  alpha = sched.alpha_t(t)
  base = sched.base_schedule.alpha_t(t)
  assert alpha.max() <= base.max() + 1e-6
  assert alpha.min() >= base.min() - 1e-6
