"""Noise schedule definitions"""

import abc
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


class NoiseSchedule(torch.nn.Module, abc.ABC):
  def forward(self, t):
    return self.alpha_prime_t(t), self.alpha_t(t)

  @abc.abstractmethod
  def alpha_t(self, t):
    pass

  @abc.abstractmethod
  def alpha_prime_t(self, t):
    pass

  def record_time_loss_pair(self, t, loss, step):
    """For adaptive noise scheduling.

    Args:
      t: (B,) timesteps.
      loss: (B,) per-sample losses.
      step: global training step index (optimizer steps).
    """
    pass


class CosineSquared(NoiseSchedule):
  """alpha_t = (1 - cos^2(pi/2 * (1-t)))"""
  def __init__(self, eps):
    super().__init__()
    self.eps = eps
    self.half_pi = torch.pi / 2

  def alpha_t(self, t):
    angle = self.half_pi * (1 - t)
    base_alpha = torch.sin(angle) ** 2
    return self.eps + (1 - self.eps) * base_alpha

  def alpha_prime_t(self, t):
    angle = self.half_pi * (1 - t)
    return (-(1 - self.eps) * 2 * torch.sin(angle)
            * torch.cos(angle) * self.half_pi)


class LogLinear(NoiseSchedule):
  """alpha_t = 1 - t"""
  def __init__(self, eps):
    super().__init__()
    self.eps = eps

  def alpha_t(self, t):
    return self.eps + (1 - self.eps) * (1 - t)

  def alpha_prime_t(self, t):
    return -(1 - self.eps) * torch.ones_like(t)


def alpha_star_sphere(vocab_size, dim, delta=0.1):
  """Truncation bound for S-FLM (Eq. 17, hyperspherical-flows paper).

  Tractable model: embeddings i.i.d. uniform on S^{d-1}, z_a =
  SLERP(noise, e_k, a). Smallest signal level a at which e_k is the
  nearest neighbor of z_a w.p. >= 1-delta (union bound over the
  |V|-1 impostors; impostor similarity is sub-Gaussian with
  parameter 1/sqrt(d), target similarity ~ sin(pi*a/2)).
  Use as noise.alpha_max (MDLM convention, invert_time_convention
  =false: alpha_t is the signal level).
  """
  t = math.sqrt(2 * math.log(2 * (vocab_size - 1) / delta) / dim)
  return (2 / math.pi) * math.asin(t)


def alpha_star_euclidean(vocab_size, delta=0.1, noise_std=1.0,
                         embed_norm=1.0):
  """Truncation bound for EFLM (flat-space analog of Eq. 17).

  Tractable model: embeddings i.i.d. uniform on the sphere of
  radius embed_norm (ngpt init: ||e_v|| ~= 1), noise ~ N(0,
  noise_std^2 I_d), z_a = a*e_k + (1-a)*z_0. Nearest neighbor
  (= max inner product, all ||e_v|| equal) analysis:
    target   <z_a, e_k> ~= a*r^2
    impostor max_v <z_a, e_v> <= ||z_a|| * r * t,
             t = sqrt(2*log(2(|V|-1)/delta)/d),
  with ||z_a||^2 ~= a^2 r^2 + (1-a)^2 s^2 d. Solving gives
    a/(1-a) >= (s/r) * sqrt(2*log(2(|V|-1)/delta)) =: z
  (dimension-free: the sqrt(d) of the noise norm cancels the
  1/sqrt(d) impostor concentration). Much larger than the sphere
  bound because the N(0, I) noise norm (sqrt(d)) dwarfs the
  unit-norm embeddings.
  """
  z = (noise_std / embed_norm) * math.sqrt(
    2 * math.log(2 * (vocab_size - 1) / delta))
  return z / (1 + z)


def alpha_star_hyperbolic(vocab_size, dim, delta=0.1, prior_cov=0.25,
                          embed_std=0.3, rho_max=12.0):
  """Truncation bound for HFLM (hyperbolic analog of Eq. 17, K=-1).

  Tractable model: clean embeddings at radius rho_1 = clamp(
  embed_std*sqrt(d)) with i.i.d. uniform directions, origin
  wrapped-normal noise at radius rho_0 = clamp(sqrt(prior_cov*d));
  clamp(r) = rho_max*tanh(r/rho_max) mirrors HFLM._rho_clamp. z_a
  sits at arc-length fraction a along the geodesic noise -> e_k of
  length D ~= rho_0 + rho_1 - log(2) (high-d directions are nearly
  orthogonal, so geodesics pass within O(1) of the origin and
  distances are tree-like: d(x,y) ~= r_x + r_y - log 2). e_k
  becomes the nearest neighbor once z_a crosses onto the outward
  leg toward e_k:
    (1-a)*D <= (a*D - rho_0) + rho_1 - log2 - t
  (impostor angles fluctuate by cos(theta) ~ t, entering only
  additively at scale t = sqrt(2*log(2(|V|-1)/delta)/d)), giving
    a >= (rho_0 + t/2) / (rho_0 + rho_1 - log 2).
  The sphere bound (0.093 at d=512) collapses HFLM (see
  experiments/hflm/RESULTS.md); this bound is much larger because
  the noise radius far exceeds the clean-embedding radius.
  """
  def clamp(r):
    return rho_max * math.tanh(r / rho_max)
  rho_noise = clamp(math.sqrt(prior_cov * dim))
  rho_clean = clamp(embed_std * math.sqrt(dim))
  t = math.sqrt(2 * math.log(2 * (vocab_size - 1) / delta) / dim)
  return (rho_noise + t / 2) / (rho_noise + rho_clean - math.log(2))


class TruncatedScheduleWrapper(NoiseSchedule):
  """Rescale a base schedule to be in [alpha_min, alpha_max]."""
  def __init__(self, base_schedule, alpha_min, alpha_max, eps):
    super().__init__()
    if not 0 <= alpha_min < alpha_max <= 1:
      raise ValueError(
        f'Expected 0 <= alpha_min < alpha_max <= 1, got '
        f'alpha_min={alpha_min}, alpha_max={alpha_max}')
    self.base_schedule = base_schedule
    self.eps = eps
    base_max = base_schedule.alpha_t(torch.tensor(0.0)).item()
    base_min = base_schedule.alpha_t(torch.tensor(1.0)).item()
    self.scale = (alpha_max - alpha_min) / (base_max - base_min)
    self.offset = alpha_min - self.scale * base_min

  def alpha_t(self, t):
    return (self.scale * self.base_schedule.alpha_t(t)
            + self.offset).clamp(min=self.eps)

  def alpha_prime_t(self, t):
    return self.scale * self.base_schedule.alpha_prime_t(t)


class AdaptiveSchedule(NoiseSchedule):
  """
  Collects (t, loss) pairs during training, periodically fits
  a spline to the loss profile, and remaps time to concentrate
  sampling where |dL/dt| is largest.
  """

  def __init__(self, base_schedule, buffer_size,
               refit_every, n_grid, n_knots, spline_degree,
               ridge_alpha, uniform_mix, max_steps, warmup_steps,
               ema, plot_profile=False,
               plot_dir='adaptive_noise_plots',
               log_importance=False):
    super().__init__()
    self.base_schedule = base_schedule
    self.buffer_size = buffer_size
    self.refit_every = refit_every
    self.n_knots = n_knots
    self.spline_degree = spline_degree
    self.ridge_alpha = ridge_alpha
    self.uniform_mix = uniform_mix
    self.warmup_steps = warmup_steps
    self.ema = ema
    self.log_importance = log_importance
    self.plot_profile = plot_profile
    self.plot_dir = plot_dir
    self._step_fmt = f'0{len(str(max_steps))}d'

    # Use buffers to be saved automatically in checkpoints
    self.register_buffer('t_buf', 
      torch.zeros(buffer_size, dtype=torch.float64))
    self.register_buffer('loss_buf', 
      torch.zeros(buffer_size, dtype=torch.float64))
    self.register_buffer('buf_pos', 
      torch.tensor(0, dtype=torch.long))
    self.register_buffer('alpha_vals', 
      torch.zeros(n_grid, dtype=torch.float64))
    self.register_buffer('has_schedule', torch.tensor(False))
    self.register_buffer('ema_alpha_vals',
      torch.zeros(n_grid, dtype=torch.float64))
    self.register_buffer('refit_count',
      torch.tensor(0, dtype=torch.long))
    # persistent=False: keeps old/new checkpoints interchangeable
    # (strict load). On resume n_seen restarts at 0 — harmless: runs
    # with a schedule keep refitting via has_schedule; runs without
    # one simply refill the buffer first.
    self.register_buffer('n_seen',
      torch.tensor(0, dtype=torch.long), persistent=False)

    self._grid = np.linspace(0, 1, n_grid)
    self._alpha_spline = None
    self._dalpha_spline = None

  def record_time_loss_pair(self, t, loss, step):
    if step < self.warmup_steps:
      return
    n = len(t)
    pos = self.buf_pos.item()
    end = pos + n
    t_val = t.detach().to(self.t_buf.dtype)
    l_val = loss.detach().to(self.loss_buf.dtype)
    if end <= self.buffer_size:
      self.t_buf[pos:end] = t_val
      self.loss_buf[pos:end] = l_val
    else:
      # Wrap around: fill end of buffer, spill remainder to start
      first = self.buffer_size - pos
      self.t_buf[pos:] = t_val[:first]
      self.loss_buf[pos:] = l_val[:first]
      self.t_buf[:n - first] = t_val[first:]
      self.loss_buf[:n - first] = l_val[first:]
    self.buf_pos.fill_(end % self.buffer_size)
    self.n_seen += n
    # Refit once the buffer has been filled at least once. n_seen (not
    # `end >= buffer_size`, which only holds on the exact wrap-around
    # step) so the first refit fires: with buffer_size a multiple of
    # the batch size, the wrap step is `filled-1 (mod fill_steps)` and
    # never coincides with `step % refit_every == 0`.
    buffer_full = (self.n_seen.item() >= self.buffer_size
                   or self.has_schedule.item())
    if (buffer_full and step % self.refit_every == 0):
      self._refit()
      if self.plot_profile:
        self._plot_profile(step)

  def _refit(self):
    # 1. Fit spline to loss profile
    t_np = self.t_buf.cpu().numpy()
    loss_np = self.loss_buf.cpu().numpy()
    model = make_pipeline(
      SplineTransformer(n_knots=self.n_knots,
                        degree=self.spline_degree,
                        extrapolation='continue'),
      Ridge(alpha=self.ridge_alpha))
    model.fit(t_np.reshape(-1, 1), loss_np)

    # 2. Smoothed loss on grid -> gradient -> CDF
    loss_smooth = model.predict(self._grid.reshape(-1, 1))
    if self.log_importance:
      # Slope of log-loss: an exponentially decaying profile (HFLM's
      # ramp-shaped geometry) has near-constant |d log L/dt|, so the
      # remap spreads samples across the whole ramp instead of piling
      # onto the linear-scale band edge.
      loss_smooth = np.log(np.maximum(loss_smooth, 1e-12))
    dloss_dt = np.gradient(loss_smooth, self._grid)
    # Loss should be always increasing with more noise.
    #  If it is decreasing, it is an artifact -> remove
    importance = np.maximum(dloss_dt, 0)
    # Uniform smoothing. Default: 1e-3. Ensures the CDF is
    #  always strictly increasing, which is needed to invert.
    #  Using uniform_mix = 1 ignores the adaptive schedule.
    importance = (1 - self.uniform_mix) * importance + self.uniform_mix
    cdf = np.cumsum(importance)
    cdf = cdf / cdf[-1]

    # 3. Inverse CDF to get t -> alpha map with high density
    #  on regions where the loss has high derivative. Clip: queries
    #  below cdf[0] extrapolate to t < 0, which would push alpha
    #  above the base range (past alpha_max under truncation).
    t_remapped = np.clip(
      PchipInterpolator(cdf, self._grid)(self._grid), 0.0, 1.0)
    t_torch = torch.as_tensor(t_remapped, dtype=torch.float32)
    av = self.base_schedule.alpha_t(t_torch).numpy()

    # 4. EMA smoothing with bias correction (like Adam)
    av_torch = torch.from_numpy(av).to(self.ema_alpha_vals.device)
    self.refit_count += 1
    if self.ema > 0:
      self.ema_alpha_vals.mul_(self.ema).add_(
        av_torch, alpha=1 - self.ema)
      # Bias correction: divide by (1 - beta^t)
      correction = 1 - self.ema ** self.refit_count.item()
      av_corrected = (self.ema_alpha_vals / correction).cpu().numpy()
    else:
      av_corrected = av

    # 5. Store schedule as spline + save values for checkpointing
    self._alpha_spline = PchipInterpolator(self._grid, av_corrected)
    self._dalpha_spline = self._alpha_spline.derivative()
    self.alpha_vals.copy_(torch.from_numpy(av_corrected))
    self.has_schedule.fill_(True)

  def load_state_dict(self, sd, strict=True, *args, **kwargs):
    super().load_state_dict(sd, strict=False, *args, **kwargs)
    # Reconstruct splines from the loaded alpha_vals buffer
    if self.has_schedule.item():
      av = self.alpha_vals.cpu().numpy()
      self._alpha_spline = PchipInterpolator(self._grid, av)
      self._dalpha_spline = self._alpha_spline.derivative()

  def _plot_profile(self, step):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(self.plot_dir, exist_ok=True)
    t_grid = self._grid
    t_torch = torch.as_tensor(t_grid, dtype=torch.float32)
    alpha_base = self.base_schedule.alpha_t(t_torch).numpy()
    alpha_adapt = self._alpha_spline(t_grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(t_grid, alpha_base, 'k--', label='Base')
    ax1.plot(t_grid, alpha_adapt, 'r-', label='Adapted')
    ax1.set(xlabel='t', ylabel='alpha_t', title=f'Step {step}')
    ax1.legend()

    ax2.scatter(self.t_buf.cpu().numpy()[::4],
                self.loss_buf.cpu().numpy()[::4],
                alpha=0.05, s=1, c='gray')
    ax2.set(xlabel='t', ylabel='Loss', title='Buffer')

    fig.tight_layout()
    fig.savefig(os.path.join(
      self.plot_dir, f'step_{step:{self._step_fmt}}.png'), dpi=100)
    plt.close(fig)

  def _eval_spline(self, spline, base_fn, t):
    if spline is None:
      return base_fn(t)
    vals = spline(t.detach().cpu().numpy())
    return torch.as_tensor(vals, dtype=t.dtype, device=t.device)

  def _maybe_rebuild_spline(self):
    # After loading from a checkpoint, Lightning copies the registered
    # buffers (alpha_vals/has_schedule) via the top-level module's
    # load_state_dict, which does NOT dispatch to this child's
    # load_state_dict override -- so the splines are never rebuilt and
    # alpha_t() silently falls back to the base schedule. Rebuild lazily
    # from the loaded buffer on first use. No-op during training (the
    # spline is already set by _refit once has_schedule is True).
    if self._alpha_spline is None and bool(self.has_schedule.item()):
      av = self.alpha_vals.cpu().numpy()
      self._alpha_spline = PchipInterpolator(self._grid, av)
      self._dalpha_spline = self._alpha_spline.derivative()

  def alpha_t(self, t):
    self._maybe_rebuild_spline()
    return self._eval_spline(
      self._alpha_spline, self.base_schedule.alpha_t, t)

  def alpha_prime_t(self, t):
    self._maybe_rebuild_spline()
    return self._eval_spline(
      self._dalpha_spline, self.base_schedule.alpha_prime_t, t)

def _inv_softplus(y):
  return math.log(math.expm1(y))


class UnifInfoSchedule(NoiseSchedule):
  """Learnable Gumbel schedule over gamma = logNSR (LangFlow).

  Variance-preserving: sigma^2 = sigmoid(gamma), alpha^2 = sigmoid(-gamma).
  gamma ~ Gumbel via inverse-CDF: gamma = P_mu - P_beta * log(-log q).
  Surrogate entropy H_gamma = H_inf * exp(-exp(-(gamma - P_mu)/P_beta)).
  """

  def __init__(self, trainable: bool, q_clip: float, H_inf_init: float,
               beta_floor: float = 1e-4, H_floor: float = 1e-4):
    super().__init__()
    self.trainable = trainable
    self.q_clip = q_clip
    self.beta_floor = beta_floor
    self.H_floor = H_floor

    raw_mu = torch.tensor(0.0)
    raw_beta = torch.tensor(_inv_softplus(1.0))
    raw_H = torch.tensor(_inv_softplus(H_inf_init - H_floor))
    if trainable:
      self.raw_mu = torch.nn.Parameter(raw_mu)
      self.raw_beta = torch.nn.Parameter(raw_beta)
      self.raw_H = torch.nn.Parameter(raw_H)
    else:
      self.register_buffer('raw_mu', raw_mu)
      self.register_buffer('raw_beta', raw_beta)
      self.register_buffer('raw_H', raw_H)

  @property
  def P_mu(self):
    return self.raw_mu

  @property
  def P_beta(self):
    return F.softplus(self.raw_beta) + self.beta_floor

  @property
  def H_inf(self):
    return F.softplus(self.raw_H) + self.H_floor

  def _gamma_clip_bounds(self):
    lo = self.P_mu - self.P_beta * math.log(-math.log(self.q_clip))
    hi = self.P_mu - self.P_beta * math.log(-math.log(1 - self.q_clip))
    return lo, hi

  def sample_gamma(self, n, device, *, antithetic=True):
    eps = torch.rand(n, device=device)
    if antithetic:
      offset = torch.arange(n, device=device) / n
      eps = (eps / n + offset) % 1
    q = eps.clamp(self.q_clip, 1 - self.q_clip)
    gamma = self.P_mu - self.P_beta * torch.log(-torch.log(q))
    lo, hi = self._gamma_clip_bounds()
    return gamma.clamp(lo, hi)

  def alpha_sigma_from_gamma(self, gamma):
    sigma = torch.sqrt(torch.sigmoid(gamma))
    alpha = torch.sqrt(torch.sigmoid(-gamma))
    return alpha, sigma

  def surrogate_entropy(self, gamma):
    return self.H_inf * torch.exp(
      -torch.exp(-(gamma - self.P_mu) / self.P_beta))

  def scheduler_loss(self, gamma, ce_detached):
    if not self.trainable:
      return torch.zeros((), device=gamma.device, dtype=self.P_mu.dtype)
    H_gamma = self.surrogate_entropy(gamma.detach())
    return ((ce_detached - H_gamma) ** 2).mean()

  def alpha_t(self, t):
    gamma = self.P_mu - self.P_beta * torch.log(
      -torch.log(t.clamp(self.q_clip, 1 - self.q_clip)))
    alpha, _ = self.alpha_sigma_from_gamma(gamma)
    return alpha

  def alpha_prime_t(self, t):
    return torch.zeros_like(t)


def get_noise(config):
  noise_config = config.noise
  if noise_config.type == 'gumbel':
    noise = UnifInfoSchedule(
      trainable=noise_config.trainable,
      q_clip=noise_config.q_clip,
      H_inf_init=noise_config.H_inf_init)
    return noise
  if noise_config.type == 'log-linear':
    noise = LogLinear(noise_config.eps)
  elif noise_config.type == 'cosine-squared':
    noise = CosineSquared(noise_config.eps)
  else:
    raise ValueError(f'Unknown noise type: {noise_config.type}')

  if noise_config.alpha_min is not None or noise_config.alpha_max is not None:
    alpha_min = noise_config.alpha_min
    alpha_max = noise_config.alpha_max
    if alpha_min is None:
      alpha_min = noise.alpha_t(torch.tensor(1.0)).item()
    if alpha_max is None:
      alpha_max = noise.alpha_t(torch.tensor(0.0)).item()
    noise = TruncatedScheduleWrapper(noise, alpha_min,
                                     alpha_max, noise_config.eps)

  if noise_config.adaptive:
    gbs = config.loader.global_batch_size
    buf = noise_config.adaptive_buffer_size
    assert buf % gbs == 0, (
      f'adaptive_buffer_size ({buf}) must be a multiple of '
      f'global_batch_size ({gbs})')
    noise = AdaptiveSchedule(
      noise,
      buffer_size=buf,
      refit_every=noise_config.adaptive_refit_every,
      n_grid=noise_config.adaptive_n_grid,
      n_knots=noise_config.adaptive_n_knots,
      spline_degree=noise_config.adaptive_spline_degree,
      ridge_alpha=noise_config.adaptive_ridge_alpha,
      uniform_mix=noise_config.adaptive_uniform_mix,
      max_steps=config.trainer.max_steps,
      warmup_steps=noise_config.adaptive_warmup_steps,
      ema=noise_config.adaptive_ema,
      plot_profile=noise_config.adaptive_plot_profile,
      plot_dir=noise_config.adaptive_plot_dir,
      log_importance=noise_config.get('adaptive_log_importance', False))

  return noise
