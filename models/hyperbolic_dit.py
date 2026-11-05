import math

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from numeric.geo_bridge import GeoUtils, HyperbolicHeatKernel
from .dit import (
  DDiTBlock,
  DDiTFinalLayer,
  TimestepEmbedder,
  Rotary,
)


class HyLaFeatures(nn.Module):
  """Random Laplacian features (HyLa, Yu & De Sa, ICLR 2023) of the HBFM state.

  Maps the Cartesian bridge state ([B, L, sum(d_i)], one Poincaré-ball point
  per product factor) once into `num_features` Euclidean features whose inner
  products estimate an isometry-invariant kernel, so the Euclidean DiT after
  it sees the geometry through the kernel instead of the raw ball
  coordinates. Per factor, `num_features / num_factors` boundary points
  omega ~ Unif(S^{d-1}), eigenvalue parameters lambda ~ N(0, scale^2) and
  phases b ~ Unif[0, 2 pi) are drawn once from `seed` (buffers, so a
  checkpoint restores them and every DDP rank holds the same draw). The
  feature of the factor's dimensionless polar state (s = rho / R, u) is

      HyLa(z) = exp((d - 1) / 2 <omega, z>_H) cos(lambda <omega, z>_H + b)

  divided by sqrt(num_features), with `<omega, z>_H = -log(cosh s - sinh s
  <u, omega>)` the signed hyperbolic distance from the origin to the
  horocycle through z at omega: the paper's log((1 - |w|^2) / |w - omega|^2)
  for the unit-ball image w = z / R, i.e. minus the Busemann term of
  `numeric.horosphere`. Each feature is an eigenfunction of its factor's
  Laplace-Beltrami operator (eigenvalue -(lambda^2 + (d-1)^2/4) |K|), and the
  paper's kernel identity holds per factor, so the concatenation's inner
  product estimates the mean of the factors' isometry-invariant kernels.
  Curvature enters only through s, as in the horosphere readout.

  `normalize=False` drops the 1 / sqrt(num_features) factor, as the reference
  implementation (github.com/ydtydr/HyLa, hype/models.py) does; that code also
  halves the drawn lambda in the angle, so its `lambda_scale` is 2x `scale`.

  `radius_cap` evaluates the map at min(s, radius_cap): with a finite number
  of features the eigenfunctions are all ~0 past s ~ 6 (their e^{(d-1)P/2}
  envelope is a Poisson kernel, sparse near the boundary), so a state pinned
  at the boundary (`HyperbolicBoundaryFM._clean_state`, s ~ 35) or a bridge
  state far out at sampling time would be invisible to the trunk; the cap is
  a finite pin radius, at which the direction still identifies the word.
  """

  def __init__(self, prod_factor_dim, prod_factor_gaussian_curvature,
               num_features, scale, seed, radius_cap=None, normalize=True):
    super().__init__()
    if len(set(prod_factor_dim)) != 1:
      raise ValueError(
        f'HyLa needs one shared factor dimension; got {prod_factor_dim}.')
    num_factors, factor_dim = len(prod_factor_dim), prod_factor_dim[0]
    if num_features % num_factors != 0:
      raise ValueError(f'model.hyla_dim={num_features} must be a multiple of '
                       f'the number of product factors ({num_factors}).')
    self.prod_factor_dim = prod_factor_dim
    self.prod_factor_gaussian_curvature = prod_factor_gaussian_curvature
    self.num_features = num_features
    self.radius_cap = radius_cap
    self.norm = 1.0 / math.sqrt(num_features) if normalize else 1.0
    per_factor = num_features // num_factors
    gen = torch.Generator().manual_seed(seed)
    omegas = torch.randn(num_factors, per_factor, factor_dim,
                         generator=gen, dtype=torch.float64)
    self.register_buffer(
      'omegas', omegas / omegas.norm(dim=-1, keepdim=True))
    self.register_buffer('lambdas', scale * torch.randn(
      num_factors, per_factor, generator=gen, dtype=torch.float64))
    self.register_buffer('biases', 2 * math.pi * torch.rand(
      num_factors, per_factor, generator=gen, dtype=torch.float64))
    self.register_buffer('kappas', torch.tensor(
      [1.0 / GeoUtils._curvature_scale(k)
       for k in prod_factor_gaussian_curvature], dtype=torch.float64),
      persistent=False)                                   # derived from the config

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    """[B, L, sum(d_i)] Cartesian state -> [B, L, num_features], float64."""
    rhos, us = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(
      z.to(torch.float64), self.prod_factor_dim,
      self.prod_factor_gaussian_curvature)
    us = us.unflatten(-1, self.omegas.shape[:1] + self.omegas.shape[-1:])
    ss = (rhos * self.kappas).unsqueeze(-1)                     # (B, L, m, 1)
    if self.radius_cap is not None:
      # A state past the cap (a prompt pinned at the boundary) is evaluated at
      # a frozen pin: no gradient into its direction, i.e. into the word's
      # embedding. The envelope's sensitivity to the direction grows like
      # e^{2P} sinh(s), and letting it reach the embedding from every prompt
      # position drove the embedding gradient to ~1e3 and collapsed training.
      pinned = ss > self.radius_cap
      ss = ss.clamp_max(self.radius_cap)
      us = torch.where(pinned, us.detach(), us)
    dot = torch.einsum('blmd,mkd->blmk', us, self.omegas)       # (B, L, m, k)
    # <omega, z>_H = -log(cosh s - sinh s <u, omega>), e^{s} pulled out of the
    # log so it never overflows; <u, omega> = 0 at the origin gives 0.
    horo = -(ss + ((1 - dot) / 2 + (1 + dot) / 2 * (-2 * ss).exp())
             .clamp_min(torch.finfo(torch.float64).tiny).log())
    feats = ((self.omegas.shape[-1] - 1) / 2 * horo).exp() * torch.cos(
      self.lambdas * horo + self.biases) * self.norm
    return feats.flatten(-2)


class HarmonicFeatures(nn.Module):
  """Hyperbolic-harmonic spherical-harmonic features of the HBFM state (the
  `sphere` input map, `model.sphere_degree`).

  Per H^3 factor, the dimensionless polar state (s = rho / R, u in S^2) is
  mapped to the harmonic extensions of the real spherical harmonics,

      h_lm(z) = f_l(s) Y_lm(u),   l = 0..L, m = -l..l,   (L + 1)^2 features,

      f_l(s) = r^l 2F1(l, -1/2; l + 3/2; r^2) / 2F1(l, -1/2; l + 3/2; 1),
      r = tanh(s / 2),

  f_l being the radial solution of the H^3 Laplace equation regular at the
  origin, normalized to f_l -> 1 at the boundary. h_lm is the Poisson integral
  of Y_lm over the boundary, so it is harmonic (Delta h_lm = 0), bounded
  (|h_lm| <= max |Y_lm| = sqrt(2l + 1)) and tends to Y_lm(u) as s -> inf: a
  state pinned at the boundary (`HyperbolicBoundaryFM._clean_state`) is seen
  through its direction, where HyLa's features vanish. f_l rises from 0
  (l >= 1) at the origin to 1 past s ~ ln(2l), so the basis is a low-pass
  filter on the direction whose cutoff degree grows with the radius; the
  plain coordinates z = R tanh(s/2) u are the L = 1 case. Y_lm are the real
  harmonics, orthonormal under the uniform probability measure on S^2
  (Y_00 = 1, Y_1m = sqrt(3) u_m). f_l depends on s only: it is tabulated at
  construction (scipy's 2F1 on a grid of step `TABLE_STEP` up to `TABLE_MAX`,
  past which f_l = 1 to double precision) and linearly interpolated.

  f_0 = 1 and every f_l saturates at a small radius, so the trunk, which has
  no time conditioning, is also given the radial channels tanh(s / c) for c in
  `radial_scales`: num_features = num_factors * ((L + 1)^2 + len(radial_scales)).
  `mc_samples` = M (null = off) replaces the harmonics' value by their Monte-Carlo
  Poisson integral over M fresh boundary angles w ~ P(z, .) = (cosh s - sinh s <u, w>)^{-2},
  the Poisson kernel being its own importance proposal (every weight is 1): the mean of
  Y_lm(w) is unbiased for f_l(s) Y_lm(u), bounded by sqrt(2l + 1), and collapses onto
  Y_lm(u) at the boundary, where the kernel does; the gradient is the closed form's and
  the radial channels stay exact. The module adds no checkpoint keys.
  """
  TABLE_STEP = 1.0 / 256
  TABLE_MAX = 36.0        # > 2 atanh(1 - 8 eps) = 34.66, the readback's largest s

  def __init__(self, prod_factor_dim, prod_factor_gaussian_curvature, degree,
               radial_scales, mc_samples=None):
    super().__init__()
    if any(d != 3 for d in prod_factor_dim):
      raise ValueError(
        f'the sphere map needs H^3 factors (harmonics on S^2); got {prod_factor_dim}.')
    degree = int(degree)
    if degree < 0 or not radial_scales:
      raise ValueError('model.sphere_degree must be >= 0 and '
                       'model.sphere_radial_scales non-empty.')
    if mc_samples is not None and int(mc_samples) < 1:
      raise ValueError(f'model.sphere_mc_samples={mc_samples} must be null or an int >= 1.')
    self.prod_factor_dim = prod_factor_dim
    self.prod_factor_gaussian_curvature = prod_factor_gaussian_curvature
    self.degree = degree
    self.mc_samples = None if mc_samples is None else int(mc_samples)
    self.num_features = len(prod_factor_dim) * (
      (degree + 1) ** 2 + len(radial_scales))
    import numpy as np
    from scipy.special import hyp2f1
    s = np.arange(0.0, self.TABLE_MAX + self.TABLE_STEP, self.TABLE_STEP)
    r = np.tanh(s / 2)
    table = np.stack([
      r ** l * hyp2f1(l, -0.5, l + 1.5, r * r)
      / (math.gamma(l + 1.5) / (math.gamma(1.5) * math.gamma(l + 2)))   # Gauss: 2F1(..; 1)
      for l in range(degree + 1)], axis=-1)
    table = np.concatenate([table, table[-1:]], axis=0)   # one spare row for the interpolation
    self.register_buffer('table', torch.tensor(table, dtype=torch.float64),
                         persistent=False)
    self.register_buffer('ell_of_feature', torch.tensor(
      [l for l in range(degree + 1) for _ in range(2 * l + 1)]), persistent=False)
    self.register_buffer('radial_scales', torch.tensor(
      [float(c) for c in radial_scales], dtype=torch.float64), persistent=False)
    self.register_buffer('kappas', torch.tensor(
      [1.0 / GeoUtils._curvature_scale(k)
       for k in prod_factor_gaussian_curvature], dtype=torch.float64),
      persistent=False)                                   # derived from the config

  def radial(self, s: torch.Tensor) -> torch.Tensor:
    """f_l(s), l = 0..L, for dimensionless radii s (...) -> (..., L + 1)."""
    x = (s / self.TABLE_STEP).clamp(0.0, self.table.shape[0] - 2)
    i = x.floor().long()
    w = (x - i).unsqueeze(-1)
    return self.table[i] * (1 - w) + self.table[i + 1] * w

  def spherical_harmonics(self, u: torch.Tensor) -> torch.Tensor:
    """Real Y_lm(u) of unit vectors u (..., 3) -> (..., (L + 1)^2), l = 0..L,
    m = -l..l, orthonormal under the uniform probability measure on S^2:
    Y_lm = N_lm dP_l^m(z) Re (x + iy)^m, Y_l,-m = N_lm dP_l^m(z) Im (x + iy)^m
    (m > 0) and Y_l0 = N_l0 P_l(z), with dP_l^m = d^m P_l / dz^m (the
    associated Legendre function without its (1 - z^2)^{m/2}, which (x + iy)^m
    carries) by the standard recursions, no Condon-Shortley phase."""
    x, y, z = u.unbind(-1)
    cols = {}
    re, im = torch.ones_like(x), torch.zeros_like(x)     # (x + iy)^m
    for m in range(self.degree + 1):
      if m > 0:
        re, im = x * re - y * im, x * im + y * re
      p_prev, p = None, torch.full_like(x, float(math.prod(range(1, 2 * m, 2))))  # dP_m^m = (2m - 1)!!
      for l in range(m, self.degree + 1):
        if l == m + 1:
          p_prev, p = p, (2 * m + 1) * z * p
        elif l > m + 1:
          p_prev, p = p, ((2 * l - 1) * z * p - (l + m - 1) * p_prev) / (l - m)
        norm = math.sqrt((2 * l + 1) * math.factorial(l - m) / math.factorial(l + m))
        if m == 0:
          cols[(l, 0)] = norm * p
        else:
          cols[(l, m)] = math.sqrt(2) * norm * p * re
          cols[(l, -m)] = math.sqrt(2) * norm * p * im
    return torch.stack([cols[(l, m)] for l in range(self.degree + 1)
                        for m in range(-l, l + 1)], dim=-1)

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    """[B, L, sum(d_i)] Cartesian state -> [B, L, num_features], float64:
    per factor, h_lm(z) for l = 0..L, m = -l..l, then tanh(s / c) per scale."""
    rhos, us = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(
      z.to(torch.float64), self.prod_factor_dim,
      self.prod_factor_gaussian_curvature)
    us = us.unflatten(-1, (len(self.prod_factor_dim), 3))     # (B, L, m, 3)
    ss = rhos * self.kappas                                     # (B, L, m)
    harmonics = (self.spherical_harmonics(us)
                 * self.radial(ss)[..., self.ell_of_feature])   # (B, L, m, (L+1)^2)
    if self.mc_samples is not None:
      # Monte Carlo over the boundary angle, w ~ P(z, .) exactly (unit importance
      # weights): no weight to spike near the boundary. The draws carry no graph;
      # the gradient is the closed form's (straight-through).
      w = HyperbolicHeatKernel._reflect_to_target(
        HyperbolicHeatKernel._angular_boost(
          ss.detach().unsqueeze(-1).expand(*ss.shape, self.mc_samples), 3),
        us.detach().unsqueeze(-2))                                 # (B, L, m, M, 3)
      w = w / w.norm(dim=-1, keepdim=True)   # the boost's c0 == -1 rounding at large s
      mc = self.spherical_harmonics(w).mean(-2)                    # (B, L, m, (L+1)^2)
      harmonics = harmonics + (mc - harmonics).detach()
    channels = torch.tanh(ss.unsqueeze(-1) / self.radial_scales)   # (B, L, m, |c|)
    return torch.cat([harmonics, channels], dim=-1).flatten(-2)


class HyperbolicDiT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if isinstance(config, dict):
      config = omegaconf.OmegaConf.create(config)
    self.config = config
    self.vocab_size = vocab_size
    self.adaLN = config.algo.adaLN
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    # Manifold (embedding) dimension. Defaults to the hidden width, in which
    # case the state is consumed as-is; a narrower embed_dim (an HBFM product
    # manifold smaller than the DiT) is lifted into the residual stream by
    # `in_proj`.
    embed_dim = config.model.get('embed_dim', None) or dim
    self.embed_dim = embed_dim
    self.init_mode = config.model.init
    self.eps = config.model.eps
    self.init_std = config.model.get('init_std', None)

    # Embedding param name kept as `sphere_embed` for checkpoint / sampler
    # compatibility (see ARCH §6); the class is renamed, the param is not.
    self.sphere_embed = nn.Embedding(vocab_size, embed_dim)
    if self.init_mode == 'random':
      nn.init.normal_(self.sphere_embed.weight, std=0.02)
    elif self.init_mode == 'ngpt':
      nn.init.normal_(self.sphere_embed.weight, std=1.0 / math.sqrt(embed_dim))
    elif self.init_mode == 'hyperbolic':
      # std=0.3 -> ‖e_v‖≈0.3·√d≈6.8 at d=512: under rho_max=12 with headroom,
      # same order as E[rho_prior]≈11.3 so clean/noisy radii match at t≈0.
      nn.init.normal_(self.sphere_embed.weight, std=0.3)
    elif self.init_mode == 'unit_var':
      nn.init.normal_(self.sphere_embed.weight, std=1.0)
    elif self.init_mode == 'custom':
      if self.init_std is None:
        raise ValueError(f"init_std is required if init_mode = custom")
      nn.init.normal_(self.sphere_embed.weight, std=float(self.init_std))
    elif self.init_mode == 'pretrained':
      nn.init.zeros_(self.sphere_embed.weight)
    else:
      raise ValueError(self.init_mode)

    # Input map of the state: random Laplacian features (`HyLaFeatures`,
    # `hyla_dim`) or the hyperbolic-harmonic basis (`HarmonicFeatures`,
    # `sphere_degree`); the trunk then consumes the features instead of the
    # Poincaré coordinates (or, with `hyla_concat_state`, next to them); null
    # keeps the coordinates.
    hyla_dim = config.model.get('hyla_dim', None)
    sphere_degree = config.model.get('sphere_degree', None)
    if hyla_dim is not None and sphere_degree is not None:
      raise ValueError('model.hyla_dim and model.sphere_degree are alternative '
                       'input maps; set at most one.')
    sphere_mc_samples = config.model.get('sphere_mc_samples', None)
    if sphere_mc_samples is not None and sphere_degree is None:
      raise ValueError('model.sphere_mc_samples needs the sphere map (model.sphere_degree).')
    self.hyla = None
    self.sphere = None
    self.hyla_concat_state = bool(config.model.get('hyla_concat_state', False))
    factors = GeoUtils.resolve_prod_factors(
      config.algo.prod_factor_dim, config.algo.prod_factor_gaussian_curvature,
      embed_dim)
    if hyla_dim is not None:
      self.hyla = HyLaFeatures(
        *factors, num_features=hyla_dim, scale=config.model.hyla_scale,
        seed=config.seed, radius_cap=config.model.get('hyla_radius_cap', None),
        normalize=bool(config.model.get('hyla_normalize', True)))
      in_dim = hyla_dim + (embed_dim if self.hyla_concat_state else 0)
    elif sphere_degree is not None:
      self.sphere = HarmonicFeatures(
        *factors, degree=sphere_degree,
        radial_scales=list(config.model.get('sphere_radial_scales', [1, 2, 4, 8, 16, 32])),
        mc_samples=sphere_mc_samples)
      in_dim = self.sphere.num_features
    else:
      in_dim = embed_dim
    self.in_proj = nn.Linear(in_dim, dim) if in_dim != dim else None

    self.self_conditioning = getattr(
      config.algo, 'self_conditioning', False)
    if self.self_conditioning:
      assert self.in_proj is None, (
        'self-conditioning assumes the embedding lives in the hidden space')
      self.W_in = nn.Linear(dim, dim, bias=False)
      self.W_sc = nn.Linear(dim, dim, bias=False)
      self.W_in.weight.data.zero_()
      self.W_sc.weight.data.zero_()

    if self.adaLN:
      self.sigma_map = TimestepEmbedder(cond_dim)

    self.rotary_emb = Rotary(dim // config.model.n_heads)

    self.blocks = nn.ModuleList([
      DDiTBlock(
        dim=dim,
        n_heads=config.model.n_heads,
        cond_dim=cond_dim,
        adaLN=self.adaLN,
        dropout=config.model.dropout)
      for _ in range(config.model.n_blocks)
    ])
    self.out_temperature_scaling = config.model.learn_temperature_scaling
    if self.out_temperature_scaling:
      out_channels = vocab_size + 1
    else:
      out_channels = vocab_size
    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=out_channels,
      cond_dim=cond_dim,
      adaLN=self.adaLN)

    # Required by TrainerBase.ctx_cached_len property.
    self.ctx_cached_len = 0

    if config.model.pretrained_ckpt_path is not None:
      self.load_pretrained_from(config.model.pretrained_ckpt_path)

  def load_pretrained_from(self, ckpt_path: str) -> None:
    """Adapt an AR / DUO Lightning checkpoint into this backbone on the fly.

    The source state_dict lives under keys like `backbone.<param>`; we strip
    that prefix so it maps directly onto SphereDiT, and rename the AR / DUO
    embedding parameter to the sphere-embedding name:

        backbone.vocab_embed.embedding  ->  sphere_embed.weight

    DDiTBlock / DDiTBlockCausal / DDiTFinalLayer share parameter names and
    shapes, so block and output-layer weights transfer cleanly. Target-only
    params (e.g. sigma_map and adaLN_modulation when the source is AR) keep
    their fresh-init values. Source-only params (DUO teachers, etc.) and
    shape-mismatched entries are dropped.

    When the source had no adaLN (AR: `algo.adaLN=False`), the target's
    fresh-init block adaLN (zero weight, zero bias) makes gate_msa=gate_mlp=0,
    which silences the attention/MLP outputs at step 0. We patch the bias so
    the modulation starts as an identity: shift=0, scale=0, gate=1 per block
    — matching the non-adaLN forward `x_skip + attn_out(norm1(x))`.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    src_sd = (ckpt['state_dict']
              if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt)

    # Pull source metadata to decide on adaLN identity-init.
    src_hp = ckpt.get('hyper_parameters', {}) if isinstance(ckpt, dict) else {}
    src_cfg = src_hp.get('config', {}) or {}
    src_algo = src_cfg.get('algo', {}) or {}
    src_algo_name = src_algo.get('name', '<unknown>')
    src_had_adaLN = bool(src_algo.get('adaLN', True))

    own_sd = self.state_dict()
    loaded, sliced, skipped = 0, [], 0
    for k, v in src_sd.items():
      if k.startswith('teacher') or not k.startswith('backbone.'):
        continue
      k = k[len('backbone.'):]
      if k == 'vocab_embed.embedding':
        k = 'sphere_embed.weight'
      if k not in own_sd:
        continue
      tgt = own_sd[k]
      if tgt.shape == v.shape:
        own_sd[k] = v
        loaded += 1
      elif (tgt.ndim == v.ndim and v.ndim >= 1
            and tgt.shape[1:] == v.shape[1:]):
        # Dim-0 mismatch only (e.g., AR trains with an extra mask_index
        # row appended; SFM has no mask). Copy the overlapping prefix;
        # the remainder keeps its fresh init.
        n = min(tgt.shape[0], v.shape[0])
        new_tgt = tgt.clone()
        new_tgt[:n] = v[:n]
        own_sd[k] = new_tgt
        sliced.append((k, tuple(v.shape), tuple(tgt.shape)))
        loaded += 1
      else:
        skipped += 1

    # If the source has no adaLN (AR), patch block adaLN biases so the
    # modulation is an identity at step 0: shift=0, scale=0, gate=1. The
    # DDiTFinalLayer's 2-way chunk (shift, scale) is already an identity
    # from fresh init — no patch needed there.
    adaLN_identity_patched = 0
    if self.adaLN and not src_had_adaLN:
      dim = self.config.model.hidden_size
      for i, block in enumerate(self.blocks):
        if not block.adaLN:
          continue
        k = f'blocks.{i}.adaLN_modulation.bias'
        if k not in own_sd:
          continue
        b = own_sd[k].clone()
        b.zero_()
        b[2 * dim: 3 * dim] = 1.0
        b[5 * dim: 6 * dim] = 1.0
        own_sd[k] = b
        adaLN_identity_patched += 1

    self.load_state_dict(own_sd, strict=True)
    fresh = len(own_sd) - loaded
    print(
      f'[HyperbolicDiT.load_pretrained_from] {ckpt_path}\n'
      f'  source   : algo.name={src_algo_name!r}  adaLN={src_had_adaLN}\n'
      f'  loaded   : {loaded}/{len(own_sd)}\n'
      f'  fresh    : {fresh}\n'
      f'  skipped  : {skipped} (not in target or shape mismatch)')
    for k, s_shape, t_shape in sliced:
      print(f'  sliced   : {k}  src{s_shape} -> tgt{t_shape}')
    if adaLN_identity_patched:
      print(f'  adaLN    : {adaLN_identity_patched} blocks patched to '
            f'identity (shift=0, scale=0, gate=1)')

  def init_sphere_embed_from_pretrained(self, pretrained_weight):
    if pretrained_weight.shape != (self.vocab_size, self.embed_dim):
      raise ValueError(
        f'Expected pretrained_weight shape ({self.vocab_size}, '
        f'{self.embed_dim}), got {tuple(pretrained_weight.shape)}')
    with torch.no_grad():
      self.sphere_embed.weight.copy_(pretrained_weight.float())

  def get_hyperbolic_polar_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
    emb = self.sphere_embed(token_ids)  # [B, L, d]
    rhos = emb.norm(p=2, dim=-1, keepdim=True)
    thetas = utils.sphere_normalize(emb)
    return rhos, thetas

  def reset_kv_cache(self):
    self.ctx_cached_len = 0

  def forward(self, x0, xt: torch.Tensor, sigma: torch.Tensor,
              context=None) -> torch.Tensor:
    del x0

    # [B, L, embed_dim], a Poincaré-ball point consumed as-is (or through its
    # HyLa / harmonic features); the HBFM bridge state arrives in float64, the
    # trunk runs at the parameters' precision.
    if self.hyla is not None:
      x = self.hyla(xt)
      if self.hyla_concat_state:
        x = torch.cat([xt.to(x.dtype), x], dim=-1)
    elif self.sphere is not None:
      x = self.sphere(xt)
    else:
      x = xt
    x = x.to(self.sphere_embed.weight.dtype)
    if self.in_proj is not None:
      x = self.in_proj(x)
    lf = context if hasattr(context, 'z_sc') else None

    if self.self_conditioning and lf is not None:
      z_sc = lf.z_sc if lf.z_sc is not None else torch.zeros_like(x)
      x = x + self.W_in(x) + self.W_sc(z_sc)

    if self.adaLN:
      t_cond = F.silu(self.sigma_map(sigma))
    else:
      t_cond = None

    rotary_cos_sin = self.rotary_emb(x)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for block in self.blocks:
        x = block(x, rotary_cos_sin, c=t_cond)
      x = self.output_layer(x, c=t_cond)
      if self.out_temperature_scaling:
        pre_temperature = x[:, :, [-1]]
        x = x[:, :, :-1]
        t = 1 + self.eps + torch.tanh(pre_temperature) * (1 - self.eps) 
        x = x / t
    return x
