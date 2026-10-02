import math

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from numeric.geo_bridge import GeoUtils
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

  `radius_cap` evaluates the map at min(s, radius_cap): with a finite number
  of features the eigenfunctions are all ~0 past s ~ 6 (their e^{(d-1)P/2}
  envelope is a Poisson kernel, sparse near the boundary), so a state pinned
  at the boundary (`HyperbolicBoundaryFM._clean_state`, s ~ 35) or a bridge
  state far out at sampling time would be invisible to the trunk; the cap is
  a finite pin radius, at which the direction still identifies the word.
  """

  def __init__(self, prod_factor_dim, prod_factor_gaussian_curvature,
               num_features, scale, seed, radius_cap=None):
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
      ss = ss.clamp_max(self.radius_cap)
    dot = torch.einsum('blmd,mkd->blmk', us, self.omegas)       # (B, L, m, k)
    # <omega, z>_H = -log(cosh s - sinh s <u, omega>), e^{s} pulled out of the
    # log so it never overflows; <u, omega> = 0 at the origin gives 0.
    horo = -(ss + ((1 - dot) / 2 + (1 + dot) / 2 * (-2 * ss).exp())
             .clamp_min(torch.finfo(torch.float64).tiny).log())
    feats = ((self.omegas.shape[-1] - 1) / 2 * horo).exp() * torch.cos(
      self.lambdas * horo + self.biases) / math.sqrt(self.num_features)
    return feats.flatten(-2)


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

    # Random Laplacian features of the state (`HyLaFeatures`): the trunk then
    # consumes `hyla_dim` Euclidean features instead of the Poincaré
    # coordinates (or, with `hyla_concat_state`, next to them); null keeps the
    # coordinates.
    hyla_dim = config.model.get('hyla_dim', None)
    self.hyla = None
    self.hyla_concat_state = bool(config.model.get('hyla_concat_state', False))
    if hyla_dim is not None:
      self.hyla = HyLaFeatures(
        *GeoUtils.resolve_prod_factors(
          config.algo.prod_factor_dim,
          config.algo.prod_factor_gaussian_curvature, embed_dim),
        num_features=hyla_dim, scale=config.model.hyla_scale,
        seed=config.seed, radius_cap=config.model.get('hyla_radius_cap', None))
    in_dim = embed_dim if self.hyla is None else (
      hyla_dim + (embed_dim if self.hyla_concat_state else 0))
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
    # HyLa features); the HBFM bridge state arrives in float64, the trunk runs
    # at the parameters' precision.
    if self.hyla is None:
      x = xt
    else:
      x = self.hyla(xt)
      if self.hyla_concat_state:
        x = torch.cat([xt.to(x.dtype), x], dim=-1)
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
