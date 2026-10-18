# AdaptiveSchedule refit bugs (pre-existing, not fixed)

Two bugs in `noise_schedules.AdaptiveSchedule` (the `*-adaptive` noise configs). Both predate this
experiment: the faulty line is in the repo's initial commit `18d586f`. They are documented here and
deliberately **not fixed**, because fixing them changes the method behind every earlier adaptive result.

| | Bug 1: remap applied to the base schedule | Bug 2: one refit per micro-batch |
|---|---|---|
| Where | `noise_schedules.py:321-324` (`_refit`) | `trainer_base.py:471-472` + `noise_schedules.py:282` |
| Effect | Schedule settles ~40% of the way from the base schedule to the loss-equalizing one (`ema=0.9`); with `ema=0` it oscillates and drifts | With A gradient-accumulation micro-batches, each refit step refits A times. A ≤ 4: same as A = 1. A ≥ 16: the schedule collapses |
| Affects | every `*-adaptive` noise (log-linear, autonomous, cosine-squared), every algo that uses it | runs with A > 1 (TinyStories); sudoku runs use A = 1 |

**For this experiment:** both TinyStories adaptive arms (`autonomous-adaptive`, `log-linear-adaptive+trunc`)
must run 4 GPUs × 32 per GPU (A = 4, as in the earlier adaptive sweeps); see `setup.md`. Sudoku runs
1 GPU × 256 (A = 1). Every adaptive arm runs the same code, so results stay comparable with the earlier
adaptive runs, but "adaptive" means the half-adapted schedule described below.

---

## Bug 1: the inverse CDF is applied to the wrong schedule

### Code

`record_time_loss_pair` stores the raw uniform `t` of each sample together with its loss. That loss was
measured at `alpha = S_k(t)`, where `S_k` is the **current** (already adapted) schedule
(`EFLM.nll`: `dalpha_t, alpha_t = self.noise(t)`). `_refit` then builds the loss CDF over that `t` and maps
its inverse through the **base** schedule:

```python
# noise_schedules.py:321-324
t_remapped = np.clip(
  PchipInterpolator(cdf, self._grid)(self._grid), 0.0, 1.0)
t_torch = torch.as_tensor(t_remapped, dtype=torch.float32)
av = self.base_schedule.alpha_t(t_torch).numpy()      # <- base schedule, not S_k
```

### Why it is wrong

Write the current schedule as `S_k = B ∘ w_k`, with `B` the base schedule and `w_k` a monotone time warp
(`w_0 = id`). Let `F_B` be the normalized loss CDF measured in base time, so `w* = F_B⁻¹` is the
loss-equalizing warp. The class docstring asks for this warp: "concentrate sampling where |dL/dt| is largest".

- The buffer holds `L(B(w_k(t)))`, so the CDF `_refit` measures is `F_k = F_B ∘ w_k`.
- The code sets `w_{k+1} = F_k⁻¹ = w_k⁻¹ ∘ w*`.
  - Refit 1 (`w_0 = id`) gives `w*`, which is ideal.
  - Refit 2 gives `w*⁻¹ ∘ w* = id`, i.e. back to the base schedule.
  - Without smoothing (`ema=0`) the schedule therefore alternates between `w*` and `id`.
  - With EMA smoothing it settles near the fixed point `w ∘ w = w*`. That is a functional square root of the ideal warp: half-adapted.
- The consistent update composes with the current schedule: `S_{k+1} = S_k ∘ F_k⁻¹ = B ∘ w_k ∘ w_k⁻¹ ∘ w* = B ∘ w*`.
  It reaches the ideal at refit 1 and stays there.

(The EMA averages alpha values rather than warps, so "square root" is the qualitative picture; the
simulation below measures the actual behaviour.)

### Measured (repo's own class, CPU, no model)

Setup: the real `AdaptiveSchedule` with the scripts' refit settings (`refit_every=50`, buffer `50 × batch`,
`n_grid=1000`, `n_knots=20`, `ridge_alpha=1e-3`, `uniform_mix=1e-3`). The base schedule is sudoku
`log-linear-adaptive+trunc` at R=8 (alpha in [1e-3, 0.291]), batch 256. The synthetic per-sample loss
`CE(alpha) = 2.3·sigmoid((0.12 − alpha)/0.015) + 0.01` (×(1 + 0.1·noise)) changes only inside the band
alpha ∈ [0.09, 0.15].

Metric: the **band share**, i.e. the fraction of uniform training times `t` whose alpha lands in the band where the
loss actually changes. The base schedule has 0.207; the loss-equalizing schedule has about 0.75.

| band share after refit k | 1 | 2 | 3 | 4 | 5 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|---|---|---|
| repo as-is, `ema=0.9` (all scripts) | 0.750 | **0.328** | 0.391 | 0.418 | 0.431 | 0.435 | 0.430 | 0.426 | **0.427** |
| repo as-is, `ema=0` | 0.750 | 0.199 | 0.775 | 0.250 | 0.665 | 0.019 | 0.093 | 0.057 | **0.004** |
| fixed (remap the current schedule), `ema=0.9` | 0.750 | 0.750 | 0.753 | 0.749 | 0.750 | 0.753 | 0.757 | 0.753 | **0.754** |

- As-is with `ema=0.9`, refit 1 is ideal (0.750). Refit 2 falls most of the way back (0.328), and the schedule then settles at
  0.427, i.e. (0.427 − 0.207)/(0.750 − 0.207) ≈ **40% of the way** to the loss-equalizing schedule.
- With `ema=0` it alternates (0.75 / 0.20 / 0.78 / 0.25 / …) and then drifts to a schedule that puts
  almost no samples in the band (0.004 at refit 60).
- Fixed, it reaches the loss-equalizing schedule at refit 1 and stays there.
- Three independent re-derivations during the code review, with different synthetic losses and the autonomous
  (τ = 0.344) base, found the same picture: a stable state roughly halfway between base and ideal, and oscillation without the EMA.

## Bug 2: refits fire once per micro-batch under gradient accumulation

`training_step` runs once per micro-batch, and `_loss` calls
`self.noise.record_time_loss_pair(t_all, loss_all, self.global_step)` (`trainer_base.py:471-472`).
Lightning's `global_step` counts optimizer steps, so it is the same for all A micro-batches of an
accumulation window. The refit condition `step % self.refit_every == 0` (`noise_schedules.py:282`) is
therefore true A times in a row, so every refit step runs `_refit` A times on almost the same buffer. Each extra refit:

- moves the EMA again, so the effective EMA per refit step is `0.9^A`;
- increments `refit_count`;
- is one more iteration of Bug 1's faulty map. Later micro-batches of the same step are noised under the
  schedule that was just changed.

Measured with the same simulation at the TinyStories layout (batch 512, buffer 25 600), A micro-batches per
step, repo as-is with `ema=0.9`:

| A (accumulation) | band share, last 6 refits | max change per refit (last 10) |
|---|---|---|
| 1 | 0.428 0.428 0.428 0.428 0.429 0.429 | 0.001 |
| 4 | 0.429 0.431 0.428 0.428 0.430 0.428 | 0.003 |
| 16 | 0.021 0.007 0.022 0.007 0.024 0.010 | 0.017 |
| 64 | 0.004 0.004 0.001 0.006 0.004 0.005 | 0.006 |

- A = 4 (the earlier TinyStories layout, 4 GPUs × 32) behaves like A = 1.
- A = 16 and A = 64 never settle on a useful schedule: the band share falls far below even the base
  schedule's 0.207, so training puts almost no samples where the loss changes.
- The TinyStories script defaults (`DEVICES=1`, `PER_GPU_BS=8`, i.e. A = 64) are in this broken regime.
- A review simulation with a different synthetic loss found A = 32/64 alternating between two schedules every refit. Either way, A ≥ 16 is unusable.

Sudoku scripts train on 1 GPU with batch 256 = global batch, so A = 1 and Bug 2 does not occur.

## Impact on existing results

- Every earlier `*-adaptive` result was produced with Bug 1, e.g.:
  - sudoku `log-linear-adaptive` (53.9% at R=5);
  - sudoku `autonomous-adaptive` (50–52%);
  - TinyStories `log-linear-adaptive+trunc` (GenPPL 10.36 / 10.99 at R=1, A = 4).
- Their "adaptive" schedule is the half-adapted one above, not the loss-equalizing schedule the class docstring describes. It still beat the non-adaptive baselines in several settings; whether a fully adapted schedule does better is untested.
- Within this experiment every adaptive arm uses the same code, so comparisons between cells are consistent. Comparisons between adaptive and non-adaptive noise measure the schedule as implemented.

## Proposed fixes (not applied)

Bug 1, one line in `_refit` (`noise_schedules.py:324`): apply the remap to the current schedule, i.e. the one
the recorded `t` were measured under. Endpoints are unchanged because the current schedule maps [0, 1] onto the same alpha range:

```python
av = self.alpha_t(t_torch).numpy()   # was: self.base_schedule.alpha_t(t_torch).numpy()
```

Bug 2: refit at most once per optimizer step, e.g. a guard in `record_time_loss_pair`:

```python
if buffer_full and step % self.refit_every == 0 and step != self._last_refit_step:
  self._last_refit_step = step     # plain attribute, initialised to -1 in __init__
  self._refit()
```

Either fix changes the behaviour of every adaptive schedule. Apply them as a separate, explicitly
ablated change, and rerun the adaptive baselines, rather than inside this experiment.

<details>
<summary>Reproduction script (CPU, ~3 min)</summary>

```python
"""Reproduce the AdaptiveSchedule refit bugs with the repo's own class (CPU, no model)."""
import sys
sys.path.insert(0, '.')      # run from the repo root
import numpy as np, torch
from noise_schedules import AdaptiveSchedule, LogLinear, TruncatedScheduleWrapper

EPS, AMAX, LO, HI = 1e-3, 0.2911, 0.09, 0.15
GRID = torch.linspace(0, 1, 1000)

def ce(alpha, g):
  return (2.3 * torch.sigmoid((0.12 - alpha) / 0.015) + 0.01) * (1 + 0.1 * torch.randn(alpha.shape, generator=g))

class Fixed(AdaptiveSchedule):
  """Bug-1 fix: apply the inverse CDF to the CURRENT schedule (the one the t were recorded under)."""
  def _refit(self):
    base, outer = self.base_schedule, self
    class Current(torch.nn.Module):
      def alpha_t(_, t): return outer._eval_spline(outer._alpha_spline, base.alpha_t, t)
    self.base_schedule = Current()
    try: super()._refit()
    finally: self.base_schedule = base

def make(cls, gbs, ema):
  base = TruncatedScheduleWrapper(LogLinear(EPS), EPS, AMAX, EPS)
  return cls(base, buffer_size=50 * gbs, refit_every=50, n_grid=1000, n_knots=20, spline_degree=3,
             ridge_alpha=1e-3, uniform_mix=1e-3, max_steps=10**6, warmup_steps=0, ema=ema)

def share(sched):
  a = sched.alpha_t(GRID)
  return ((a >= LO) & (a <= HI)).float().mean().item()

def run(cls=AdaptiveSchedule, gbs=256, ema=0.9, accum=1, refits=60, seed=0):
  g = torch.Generator().manual_seed(seed)
  s = make(cls, gbs, ema)
  hist = [share(s)]
  for step in range(1, 50 * refits + 1):
    u = (torch.rand(gbs, generator=g) / gbs + torch.arange(gbs) / gbs) % 1       # antithetic, as _sample_t
    t_all = (1 - 1e-3) * u + 1e-3
    for chunk in t_all.chunk(accum):          # one record (and possible refit) per micro-batch, as in training
      _, alpha = s(chunk)
      s.record_time_loss_pair(chunk, ce(alpha, g), step)
    if step % 50 == 0:
      hist.append(share(s))
  return np.array(hist)

if __name__ == '__main__':
  show = [1, 2, 3, 4, 5, 10, 20, 40, 60]
  for name, kw in (('as-is ema=0.9', {}), ('as-is ema=0', {'ema': 0.0}), ('fixed ema=0.9', {'cls': Fixed})):
    h = run(**kw); print(name, [round(h[k], 3) for k in show])
  for A in (1, 4, 16, 64):
    h = run(gbs=512, accum=A, refits=40); print(f'A={A}', np.round(h[-6:], 3), np.abs(np.diff(h[-11:])).max())
```

</details>
