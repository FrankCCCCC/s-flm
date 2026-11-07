import math
from typing import List, Tuple, Union

import torch

from numeric.geo_bridge import GeoUtils
from .feature_base import FeatureBase


class HyLaFeatures(FeatureBase):
  """Random Laplacian features (HyLa, Yu & De Sa, ICLR 2023) of the HBFM state.

  Maps the Cartesian bridge state ([B, L, sum(d_i)], one Poincaré-ball point
  per product factor) once into `num_features` Euclidean features whose inner
  products estimate an isometry-invariant kernel, so the Euclidean DiT after
  it sees the geometry through the kernel instead of the raw ball
  coordinates. Per factor, `num_features / num_factors` boundary points
  omega ~ Unif(S^{d-1}), eigenvalue parameters lambda ~ N(0, scale^2) and
  phases b ~ Unif[0, 2 pi) are drawn once from `seed` at the first forward,
  whose factors set their shapes (buffers, so a checkpoint restores them and
  every DDP rank holds the same draw). The feature of the factor's
  dimensionless polar state (s = rho / R, u) is

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

  FEATURE_BUFFERS = ('omegas', 'lambdas', 'biases')

  def __init__(
    self,
    num_features: int,
    num_factors: int,
    scale: float,
    seed: int,
    radius_cap: float = None,
    normalize: bool = True
  ):
    super().__init__()
    self._num_features = num_features
    self.num_factors = num_factors
    self.prod_factor_dim = None
    self.scale = scale
    self.seed = seed
    self.radius_cap = radius_cap
    self.norm = 1.0 / math.sqrt(num_features) if normalize else 1.0
    for name in self.FEATURE_BUFFERS:
      self.register_buffer(name, None)

  @property
  def num_features(self) -> int:
    """The number of features produced by this module, over all product factors."""
    return self._num_features

  def _draw_features(self, prod_factor_dim):
    """Draw omega, lambda and b from `seed`, for the factors of the first forward."""
    if len(set(prod_factor_dim)) != 1:
      raise ValueError(
        f'HyLa needs one shared factor dimension; got {prod_factor_dim}.')
    num_factors, factor_dim = len(prod_factor_dim), prod_factor_dim[0]
    if self._num_features % num_factors != 0:
      raise ValueError(f'num_features={self._num_features} must be a multiple of '
                       f'the number of product factors ({num_factors}).')
    per_factor = self.num_features // num_factors
    gen = torch.Generator().manual_seed(self.seed)
    omegas = torch.randn(num_factors, per_factor, factor_dim,
                         generator=gen, dtype=torch.float64)
    self.omegas = omegas / omegas.norm(dim=-1, keepdim=True)
    self.lambdas = self.scale * torch.randn(
      num_factors, per_factor, generator=gen, dtype=torch.float64)
    self.biases = 2 * math.pi * torch.rand(
      num_factors, per_factor, generator=gen, dtype=torch.float64)

  def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
    # A checkpoint can arrive before the first forward: its draw sets the shapes.
    for name in self.FEATURE_BUFFERS:
      if prefix + name in state_dict:
        setattr(self, name, torch.empty_like(state_dict[prefix + name]))
    super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

  def prod_factor_dim_validator(
    self,
    prod_factor_dim: List[int]
  ):
    """Check that the product factor match the first forward's.

    Args:
      prod_factor_dim (`List[int]`): the dimension of each factor; there must be
        `num_factors` of them. The first forward draws the features for this
        layout, and a later layout must match it.
    """
    if self.prod_factor_dim is None:
      self.prod_factor_dim = prod_factor_dim
    else:
      for (dim, self_dim) in zip(prod_factor_dim, self.prod_factor_dim):
        if dim != self_dim:
          raise ValueError(f'this map first ran with factors {self.prod_factor_dim}; got {prod_factor_dim}.')
    return True


  def forward(
    self,
    z: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    input_coord: str,
    prod_factor_dim: List[int],
    prod_factor_gaussian_curvature: List[float],
  ) -> torch.Tensor:
    """Map a state to its HyLa features, concatenated over the product factors.

    Args:
      z (`torch.Tensor` or `Tuple[torch.Tensor, torch.Tensor]`): the state in the
        coordinates `input_coord`, see `FeatureBase.forward`; the Cartesian bridge
        state has shape `(B, L, sum(d_i))`.
      input_coord (`str`): the `Coordinate` tag of `z`.
      prod_factor_dim (`List[int]`): the dimension of each factor; there must be
        `num_factors` of them. The first forward draws the features for this layout, and a
        later layout must match it.
      prod_factor_gaussian_curvature (`List[float]`): the curvature of each factor,
        which sets the dimensionless radius s = rho / R of this call.

    Returns:
      `torch.Tensor` of shape `(B, L, num_features)`, float64: per factor, the
      features `exp((d - 1) / 2 <omega, z>_H) cos(lambda <omega, z>_H + b)` of the
      class docstring, evaluated at `min(s, radius_cap)` if `radius_cap` is set.
    """
    if len(prod_factor_dim) != self.num_factors:
      raise ValueError(f'this map was built for {self.num_factors} factors; got {prod_factor_dim}.')
    self.prod_factor_dim_validator(prod_factor_dim)
    rhos, us = super().forward(z, input_coord, prod_factor_dim, prod_factor_gaussian_curvature)
    with torch.inference_mode(False):          # the buffers outlive this call, which may be under inference
      if self.omegas is None:
        self._draw_features(prod_factor_dim)
      if self.omegas.device != rhos.device:    # a fresh draw, or a CPU checkpoint's
        self.to(rhos.device)
    us = us.unflatten(-1, (len(prod_factor_dim), prod_factor_dim[0]))
    kappas = rhos.new_tensor(
      [1.0 / GeoUtils._curvature_scale(k) for k in prod_factor_gaussian_curvature])
    ss = (rhos * kappas).unsqueeze(-1)                          # (B, L, m, 1)
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

