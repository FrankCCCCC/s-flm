"""Closed-form binary (`d == 2`) hyperbolic heat-kernel sampler and geodesic.

This module is the self-contained `H^2` slice of the heat-kernel machinery:

- [`BinaryHyperbolicHeatKernel`]: the closed-form `d == 2` free heat
  kernel, its target-conditioned bridge, and the constant-speed geodesic.
- [`GeoUtils`]: pure coordinate converters between Poincare-disk and
  Lorentz-Cartesian representations plus the numeric boundary guards; no
  random state.

Numerical guards:
- `_LORENTZ_RHO_MAX = 20`: any Lorentz-Cartesian output beyond this raises
  `ValueError`. Polar outputs are unrestricted.
- `_SPHERE_T_MAX = 0.5`: carried over from the extraction and unused here.

See `unigram/hyper_dm.md` for the underlying math.
"""

import math
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


@dataclass
class Geometry:
    """String tags identifying which manifold representation a tensor lives in."""

    POINCARE: str = "poincare"
    LORENTZ: str = "lorentz"

@dataclass
class Coordinate:
    """String tags selecting polar vs Cartesian output from kernel/bridge methods."""

    HYPERBOLIC_POLAR: str = "polar"
    CARTESIAN: str = "cartesian"

# ---------------------------------------------------------------------------
# Numerical-stability boundary constants
# ---------------------------------------------------------------------------
# Lorentz-Cartesian output becomes unreliable in float64 once cosh(rho) is large
# enough that the relative cancellation noise `cosh^2 * eps` exceeds the on-
# manifold tolerance. At `rho = 20`, `cosh(rho) ~ 2.4e8`, and the absolute defect
# `<z, z>_L + 1` floats at ~5e-8 — still well within an O(1e-6) acceptance — so 20
# is the conservative cutoff. Above this the polar form remains exact; only the
# Cartesian Lorentz conversion is refused.
_LORENTZ_RHO_MAX: float = 20.0

# The spherical Gruet ansatz `cos(phi) = v^2 + (1 - v^2) cos(s)` with the
# hyperbolic Poisson rate (d-1)^2 t / 8 is approximately correct only in the
# small-t regime; for `t > _SPHERE_T_MAX` the radial moments drift visibly from
# the analytic heat-kernel marginal (e.g. at d=2, t=10 the empirical
# `E[cos phi] ~ 0.34` versus the analytic `exp(-d t) ~ 2e-9`). The sampler refuses
# inputs that would exceed this regime.
_SPHERE_T_MAX: float = 0.5

class GeoUtils:
    # ---------------------------------------------------------------------------
    # Boundary-check helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _check_lorentz_rho_bound(
        rhos: torch.Tensor,
        d: int,
        ts: Optional[torch.Tensor] = None,
        gaussian_curvature: float = -1.0,
    ) -> None:
        """Refuse Lorentz-Cartesian conversion when float64 precision is insufficient.

        The on-manifold defect floats at order `cosh(rho / R) * eps` on the
        curvature-`K` hyperboloid of radius `R = 1/sqrt(|K|)`, so for
        `rho / R > _LORENTZ_RHO_MAX = 20` the Cartesian image cannot be trusted to
        better than ~1e-6 even at float64. Polar output is unaffected; callers who
        need large rho should keep ``output_coord=HYPERBOLIC_POLAR``.
        """
        if rhos.numel() == 0:
            return
        R = GeoUtils._curvature_scale(gaussian_curvature)
        rho_max = float(rhos.max().item())
        if rho_max > _LORENTZ_RHO_MAX * R:
            t_info = ""
            if ts is not None and ts.numel() > 0:
                t_info = f"max(ts)={float(ts.max().item()):.3g}, "
            raise ValueError(
                f"Lorentz-Cartesian output requires rho <= {_LORENTZ_RHO_MAX} * R "
                f"(R={R:.3g}) for float64 on-manifold precision; got d={d}, {t_info}"
                f"max(rho)={rho_max:.3f}. "
                f"Use output_coord=HYPERBOLIC_POLAR for these parameters."
            )


    @staticmethod
    def _uniform_sphere(
        shape, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Sample `B` points uniformly on `S^{d-1}` via normalized Gaussian draws."""
        if shape[-1] == 1:
            raise ValueError("uniform sphere on S^0 (d=1) is not supported")
        g = torch.randn(shape, dtype=dtype, device=device)
        return g / g.norm(p=2, dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)

    @staticmethod
    def wrapped_normal(
        shape: Union[torch.Size, Tuple[int, ...]],
        mean: Union[float, torch.Tensor] = 0.0,
        cov: Union[float, torch.Tensor] = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Sample `(rho, u)` from the wrapped normal on `H^d` centred at the origin.

        Draws a tangent-space Gaussian `v ~ N(mean, Sigma)` in `T_o H^d ≅ R^d` and
        reads off the polar coordinates of its exponential map at the origin. Since
        `exp_o(v) = (cosh ||v||, sinh ||v|| * v / ||v||)`, the radial coordinate is
        `rho = ||v||` (the geodesic distance to the origin) and the angular coordinate
        is the unit direction `u = v / ||v||` on `S^{d-1}` — the polar form shared by
        the Poincare and Lorentz models. With the default `mean = 0` this is the
        canonical origin-centred wrapped normal of Nagano et al. (2019); a non-zero
        `mean` shifts the tangent Gaussian within `T_o`. Convert to Cartesian with
        [`hyperbolic_polar_to_poincare_cartesian`] / [`hyperbolic_polar_to_lorentz_cartesian`].

        Args:
            shape (`torch.Size` or tuple of `int`):
                Output sample shape `(..., embedding_size)`; the last axis
                `d = embedding_size` is the tangent / embedding dimension.
            mean (`float` or `torch.Tensor` of shape `()`, `(d,)`, or `(..., d)`, *optional*, defaults to `0`):
                Tangent-space Gaussian mean in `T_o H^d`, broadcast over the batch axes.
            cov (`float` or `torch.Tensor` of shape `()`, `(d,)`, `(d, d)`, or `(*batch, d, d)`, *optional*, defaults to `1`):
                Tangent-space Gaussian **covariance** `Sigma`. A scalar gives the
                isotropic `Sigma = cov * I`; a `(d,)` vector gives the diagonal
                `Sigma = diag(cov)` (entries are variances, not standard deviations); a
                matrix is the full covariance, factored by Cholesky so
                `Cov(v) = L L^T = cov` exactly. Scalar/diagonal variances must be
                non-negative and a matrix must be symmetric positive-definite, else a
                `ValueError` is raised. A batched matrix's leading dims must be
                **broadcastable to `shape[:-1]`** by standard right-aligned rules, so a
                per-row covariance for `shape = (B, S, d)` is passed as `(B, 1, d, d)`,
                not `(B, d, d)`.
            dtype (`torch.dtype`, *optional*):
                Output dtype; defaults to a tensor `cov`/`mean`'s dtype if given, else
                the torch default (so a `float64` `cov` is not silently down-cast).
            device (`torch.device`, *optional*): Output device.

        Returns:
            tuple `(rhos, u)`:
                rhos (`torch.Tensor` of shape `(...,)`):
                    Radial coordinate `rho = ||v|| >= 0`, the geodesic distance to the
                    origin. Unrestricted (like [`sample_radial`]): a large `mean`/`cov`
                    can push `rho` past `_LORENTZ_RHO_MAX`, which the Lorentz-Cartesian
                    converter then refuses.
                u (`torch.Tensor` of shape `(..., embedding_size)`):
                    Unit direction `u = v / ||v||` on `S^{d-1}` (the canonical `e_0`
                    where `v == 0`, e.g. `cov = 0`).
        """

        # Wrapped normal on H^d with base point at the origin (Nagano et al. 2019):
        # sample a tangent-space Gaussian v ~ N(mean, Sigma) and exp-map it at the
        # origin. Since exp_o(v) = (cosh||v||, sinh||v|| * v/||v||), the polar
        # coordinates are simply rho = ||v|| and direction u = v/||v||.
        if dtype is None:
            for _param in (cov, mean):
                if torch.is_tensor(_param):
                    dtype = _param.dtype
                    break
        eps = torch.randn(tuple(shape), dtype=dtype, device=device)

        # Colour the standard-normal draw by the covariance Sigma: scalar / (d,) is a
        # (diagonal) variance, so v = eps * sqrt(Sigma); a (d, d) (optionally batched)
        # matrix is the full Sigma, factored as Sigma = L L^T via Cholesky so that
        # v = L eps has Cov(v) = Sigma exactly.
        cov_t = torch.as_tensor(cov, dtype=eps.dtype, device=eps.device)
        if cov_t.ndim < 2:
            if (cov_t < 0).any():
                raise ValueError(
                    "wrapped_normal: scalar/diagonal `cov` is a variance and must be "
                    f"non-negative; got min(cov)={float(cov_t.min()):.3g}."
                )
            v = eps * cov_t.sqrt()
        else:
            try:
                L = torch.linalg.cholesky(cov_t)
            except RuntimeError as e:
                raise ValueError(
                    "wrapped_normal: matrix `cov` must be symmetric positive-definite "
                    f"(Cholesky failed for shape {tuple(cov_t.shape)}): {e}"
                ) from e
            # Sigma broadcasts against the sample batch by standard right-aligned
            # matmul rules: cov_t's leading dims must be broadcastable to shape[:-1].
            v = (L @ eps.unsqueeze(-1)).squeeze(-1)

        # Shift by the tangent-space mean (a vector in T_o H^d), broadcasting over
        # the leading batch axes.
        mean_t = torch.as_tensor(mean, dtype=eps.dtype, device=eps.device)
        v = v + mean_t

        tiny = torch.finfo(v.dtype).tiny
        rhos = v.norm(p=2, dim=-1)
        u = v / rhos.unsqueeze(-1).clamp_min(tiny)
        # Where v == 0 (e.g. cov = 0) the direction is undefined; emit the canonical
        # e_0 so the ||u|| == 1 return contract holds for every input.
        degenerate = rhos <= tiny
        if degenerate.any():
            e0 = torch.zeros(v.shape[-1], dtype=v.dtype, device=v.device)
            e0[0] = 1.0
            u = torch.where(degenerate.unsqueeze(-1), e0, u)
        return rhos, u

    @staticmethod
    def _check_sphere_t_bound(ts: torch.Tensor, d: int) -> None:
        """Refuse spherical-heat-kernel inputs outside the small-t regime where the
        Gruet ansatz `cos(phi) = v^2 + (1 - v^2) cos(s)` is empirically reliable.
        """
        if ts.numel() == 0:
            return
        t_max = float(ts.max().item())
        if t_max > _SPHERE_T_MAX:
            raise ValueError(
                f"FreeSphericalHeatKernel is only validated for t <= {_SPHERE_T_MAX}; "
                f"got d={d}, max(ts)={t_max:.3g}. Larger t requires a different sampler."
            )

    # ---------------------------------------------------------------------------
    # Geodesic helper
    # ---------------------------------------------------------------------------

    @staticmethod
    def _geodesic_kernel(
        x: torch.Tensor,
        y: torch.Tensor,
        t: Union[float, torch.FloatTensor],
        gaussian_curvature: float,
    ) -> torch.Tensor:
        """Constant-speed geodesic from x to y at fraction t on the curvature-`K` surface.

        The sign of `gaussian_curvature` selects the geometry and its magnitude the
        model radius `R = 1/sqrt(|K|)`:

        K < 0: Lorentz lerp on the radius-`R` hyperboloid.  d_H = R * arccosh(-<x, y>_L / R^2).
        K > 0: SLERP on the radius-`R` sphere.              d_S = R * arccos(<x, y> / R^2).

        The interpolation angle `theta = d / R` is computed from the differential form
        `cosh(theta) - 1 = <x - y, x - y>_L / (2 R^2)` (hyperbolic) and
        `1 - cos(theta) = ||x - y||^2 / (2 R^2)`        (spherical), which avoids the
        catastrophic cancellation that would otherwise plague the inner product at
        high dimensions.
        """
        R = GeoUtils._curvature_scale(gaussian_curvature)
        R_sq = R * R
        diff = x - y
        if gaussian_curvature < 0:
            diff_inner = -diff[..., 0] * diff[..., 0] + (diff[..., 1:] * diff[..., 1:]).sum(-1)
            cosh_d_minus_one = (diff_inner / (2.0 * R_sq)).clamp_min(0.0)
            d = torch.acosh(1.0 + cosh_d_minus_one)
            f = torch.sinh
        else:
            diff_sq = (diff * diff).sum(-1)
            one_minus_cos = (diff_sq / (2.0 * R_sq)).clamp(0.0, 2.0)
            d = torch.acos((1.0 - one_minus_cos).clamp(-1.0, 1.0))
            f = torch.sin

        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        else:
            t = t.to(dtype=x.dtype, device=x.device)

        euclid = (1.0 - t) * x + t * y
        tiny = torch.finfo(x.dtype).tiny
        # Broadcast the scalar geodesic quantities against the ambient last axis, so a
        # per-sample column `t` of shape `(..., 1)` works as well as a python scalar.
        d_col = d.unsqueeze(-1)
        fd = f(d_col).clamp_min(tiny)
        coef_x = f((1.0 - t) * d_col) / fd
        coef_y = f(t * d_col) / fd
        gamma = coef_x * x + coef_y * y
        return torch.where(d_col < 1e-6, euclid, gamma)

    @staticmethod
    def _curvature_scale(
        gaussian_curvature: Union[float, List[float], torch.Tensor]
    ) -> Union[float, List[float], torch.Tensor]:
        """Validate `K != 0` and return the model radius `R = 1/sqrt(|K|)`.

        The curvature-`K` model surfaces are the unit ones scaled by `R`: the
        hyperboloid `<z,z>_L = -R^2 = 1/K` (`K < 0`, with its Poincare ball) or
        the sphere of radius `R` (`K > 0`), with intrinsic polar `(rho, u)`
        unchanged (`rho` = geodesic distance to the origin). Dividing Cartesian
        coordinates by `R` is the isometry onto the unit surface and maps the
        intrinsic radial `rho -> rho / R`.
        """
        if isinstance(gaussian_curvature, (float, int)):
            if gaussian_curvature == 0.0:
                raise ValueError(
                    "gaussian_curvature must be nonzero (K < 0 hyperbolic, K > 0 spherical)"
                )
            return 1.0 / math.sqrt(abs(gaussian_curvature))
        elif isinstance(gaussian_curvature, (list, tuple)):
            curv = [1.0 / math.sqrt(abs(gc)) for gc in gaussian_curvature]
            return curv
        elif torch.is_tensor(gaussian_curvature):
            return 1.0 / torch.sqrt(gaussian_curvature.abs())
        else:
            raise TypeError(f"Arguement gaussian_curvature only supports float, List[float], or torch.Tensor, {type(gaussian_curvature)} is not supported.")

    # ---------------------------------------------------------------------------
    # Random-distribution generators
    # ---------------------------------------------------------------------------

    @staticmethod
    def sample_chi(ns: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Sample `chi(n)` via the Gamma identity `chi^2(n) ~ Gamma(n/2, scale=2)`.

        Direct Gamma sampling avoids materializing `sum(ns)` standard normals,
        which is critical when `ns` carries large counts (e.g. via Poisson rates
        at large `t`).

        Args:
            ns (`torch.Tensor` of shape `(batch_size,)`):
                Integer degrees of freedom.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float64`):
                Floating-point precision of the draw.

        Returns:
            `torch.Tensor` of shape `(batch_size,)`: chi samples.
        """
        concentration = ns.to(dtype) / 2
        rate = torch.tensor(0.5, device=ns.device, dtype=dtype)
        chi2 = torch.distributions.Gamma(concentration, rate).sample()
        return chi2.sqrt()

    @staticmethod
    def sample_chi_old(ns: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Legacy `chi(n)` sampler via summed squared normals.

        Retained for reference and parity checks against [`sample_chi`]; not used
        on the hot path because it allocates `sum(ns)` standard normals.

        Args:
            ns (`torch.Tensor`):
                Integer degrees of freedom, any shape.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float64`):
                Floating-point precision of the draw.

        Returns:
            `torch.Tensor` of the same shape as `ns`: chi samples.
        """
        nshape = ns.shape
        ns = ns.reshape(-1)
        M = ns.sum().item()
        x = torch.randn(M, device=ns.device, dtype=dtype).square()
        chi2 = torch.segment_reduce(x,'sum',lengths=ns)
        return chi2.sqrt().reshape(nshape)

    # ---------------------------------------------------------------------------
    # Coordinate converters
    # ---------------------------------------------------------------------------

    @staticmethod
    def _binary_polar_direction(
        thetas: torch.Tensor
    ) -> torch.Tensor:
        """Resolve a scalar angle (d=2) into a unit direction vector.

        Args:
            thetas (`torch.FloatTensor` of shape `(batch_size,)`): angles.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, 2)`: unit vectors
            `(cos theta, sin theta)`, one per row (`||.|| == 1`).
        """
        return torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=-1)

    @staticmethod
    def _polar_direction(
        thetas: torch.Tensor
    ) -> torch.Tensor:
        """Normalize a `d`-dimensional direction vector onto `S^{d-1}`.

        For `H^d` (`d >= 3`) the angular coordinate is a unit vector `u in S^{d-1}`,
        not a scalar angle. Upstream samplers may leave a residual norm error, so
        this re-projects to the sphere (the `d == 2` scalar-angle case is handled by
        [`_binary_polar_direction`]).

        Args:
            thetas (`torch.FloatTensor` of shape `(..., d)`): direction vectors.

        Returns:
            `torch.FloatTensor` of shape `(..., d)`: unit vectors (`||.|| == 1`).
        """
        tiny = torch.finfo(thetas.dtype).tiny
        return thetas / thetas.norm(dim=-1, keepdim=True).clamp_min(tiny)

    @staticmethod
    def binary_hyperbolic_polar_to_poincare_cartesian(
        rhos: torch.Tensor,
        thetas: torch.Tensor,
        gaussian_curvature: float=-1.0,
    ) -> torch.Tensor:
        """Compute z = R * tanh(rho / 2R) * direction inside the open disk of radius `R = 1/sqrt(|K|)`.

        `tanh` saturates to 1.0 in float64 for its argument `>= ~36`. The scale is
        clamped below 1 by one ulp so the strict invariant `||z|| < R` holds for
        arbitrarily large `rho`.

        Args:
            rhos (`torch.FloatTensor` of shape `(batch_size,)`): hyperbolic radial.
            thetas (`torch.FloatTensor` of shape `(batch_size,)`): angles.
            gaussian_curvature (`float`): curvature `K < 0`; `-1.0` = unit disk model.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, 2)`: Poincare-disk Cartesian
            coordinates `(x, y)` with `||z|| < R` (`< 1` at `K = -1`).
        """
        R = GeoUtils._curvature_scale(gaussian_curvature)
        direction = GeoUtils._binary_polar_direction(thetas=thetas)
        scale = torch.tanh(rhos / (2.0 * R))
        one_minus_eps = 1.0 - torch.finfo(scale.dtype).eps
        scale = scale.clamp(max=one_minus_eps) * R
        return scale.unsqueeze(-1) * direction

    @staticmethod
    def binary_hyperbolic_polar_to_lorentz_cartesian(
        rhos: torch.FloatTensor,
        thetas: torch.FloatTensor,
        gaussian_curvature: float=-1.0,
    ) -> torch.FloatTensor:
        """Convert `(rho, theta)` on `H^2` to Lorentz-Cartesian coordinates.

        The `d == 2` lift consuming a scalar angle per sample, on the curvature-`K`
        hyperboloid of radius `R = 1/sqrt(|K|)`:
        `z = R * (cosh(rho/R), sinh(rho/R) cos theta, sinh(rho/R) sin theta)`.

        Args:
            rhos (`torch.FloatTensor` of shape `(batch_size,)`): hyperbolic radial.
            thetas (`torch.FloatTensor` of shape `(batch_size,)`): azimuthal angle.
            gaussian_curvature (`float`): curvature `K < 0`; `-1.0` = unit hyperboloid.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, 3)`: Lorentz-Cartesian
            coordinates satisfying `<z,z>_L = 1/K` (`= -1` at `K = -1`).
        """
        R = GeoUtils._curvature_scale(gaussian_curvature)
        unit_rhos = rhos / R
        sinh_r = torch.sinh(unit_rhos)
        return torch.stack(
            [torch.cosh(unit_rhos), sinh_r * thetas.cos(), sinh_r * thetas.sin()],
            dim=-1,
        ) * R

    @staticmethod
    def hyperbolic_polar_to_poincare_cartesian(
        rhos: torch.Tensor,
        thetas: torch.Tensor,
        gaussian_curvature: float=-1.0,
    ) -> torch.Tensor:
        """Compute `z = R * tanh(rho / 2R) * u` inside the open ball of radius `R = 1/sqrt(|K|)`.

        `tanh` saturates to 1.0 in float64 for its argument `>= ~36`. The scale is
        clamped below 1 by one ulp so the strict invariant `||z|| < R` holds for
        arbitrarily large `rho`.

        Args:
            rhos (`torch.FloatTensor` of shape `(...,)`): hyperbolic radial coordinate.
            thetas (`torch.FloatTensor` of shape `(..., d)`): unit direction on `S^{d-1}`.
            gaussian_curvature (`float`): curvature `K < 0`; `-1.0` = unit ball model.

        Returns:
            `torch.FloatTensor` of shape `(..., d)`: Poincare-ball Cartesian
            coordinates with `||z|| < R`.
        """
        R = GeoUtils._curvature_scale(gaussian_curvature)
        direction = GeoUtils._polar_direction(thetas=thetas)
        scale = torch.tanh(rhos / (2.0 * R))
        one_minus_eps = 1.0 - torch.finfo(scale.dtype).eps
        scale = scale.clamp(max=one_minus_eps) * R
        return scale.unsqueeze(-1) * direction

    @staticmethod
    def hyperbolic_polar_to_lorentz_cartesian(
        rhos: torch.FloatTensor,
        thetas: torch.FloatTensor,
        gaussian_curvature: float=-1.0,
    ) -> torch.FloatTensor:
        """Convert `(rho, u)` on `H^d` to Lorentz-Cartesian coordinates.

        Lifts to `z = R * (cosh(rho/R), sinh(rho/R) * u)` on the curvature-`K`
        hyperboloid of radius `R = 1/sqrt(|K|)`. The spatial block is rescaled so its
        norm-squared equals the algebraic target `cosh^2 - 1` exactly, so the
        Minkowski invariant cancels at the float precision of `cosh^2` rather than
        accumulating `sinh * unit-norm` error.

        Args:
            rhos (`torch.FloatTensor` of shape `(...,)`):
                Hyperbolic radial coordinate.
            thetas (`torch.FloatTensor` of shape `(..., d)`):
                Unit direction on `S^{d-1}`.
            gaussian_curvature (`float`): curvature `K < 0`; `-1.0` = unit hyperboloid.

        Returns:
            `torch.FloatTensor` of shape `(..., d + 1)`: Lorentz-Cartesian
            coordinates satisfying `<z,z>_L = 1/K` (`= -1` at `K = -1`).
        """
        R = GeoUtils._curvature_scale(gaussian_curvature)
        direction = GeoUtils._polar_direction(thetas=thetas)
        cosh = torch.cosh(rhos / R)
        target_sq = (cosh * cosh - 1.0).clamp_min(0.0)
        current_sq = (direction * direction).sum(-1, keepdim=True).clamp_min(
            torch.finfo(direction.dtype).tiny
        )
        scale = (target_sq.unsqueeze(-1) / current_sq).sqrt()
        spatial = direction * scale
        return torch.cat([cosh.unsqueeze(-1), spatial], dim=-1) * R

    @staticmethod
    def binary_lorentz_cartesian_to_hyperbolic_polar(
        z: torch.Tensor,
        gaussian_curvature: float=-1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert Lorentz-Cartesian to polar `(rho, thetas)`.

        `rho = R * arccosh(z[0] / R)` recovers the radial geodesic distance to the
        origin on the curvature-`K` hyperboloid of radius `R = 1/sqrt(|K|)`; the
        angular part is `theta = atan2(z[2], z[1])` over the two spatial components in
        `d == 2`.

        Args:
            z (`torch.Tensor` of shape `(..., 3)`):
                Ambient Lorentz-Cartesian coordinates on the curvature-`K`
                hyperboloid with `z[0] >= R`.
            gaussian_curvature (`float`): curvature `K < 0`; `-1.0` = unit hyperboloid.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`:
                - `rhos` of shape `(...)`, `>= 0`.
                - `thetas` of shape `(...)`, scalar angle in `(-pi, pi]` for `d == 2`.
        """
        R = GeoUtils._curvature_scale(gaussian_curvature)
        d = z.shape[-1]
        if d != 3:
            raise ValueError(f"Cartesian dimension should be 3, not {d}.")
        rhos = R * torch.acosh((z[..., 0] / R).clamp_min(1.0))
        theta = torch.atan2(z[..., 2], z[..., 1])
        return rhos, theta

    @staticmethod
    def lorentz_cartesian_to_hyperbolic_polar(
        z: torch.Tensor,
        gaussian_curvature: float=-1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert Lorentz-Cartesian on `H^d` to polar `(rho, u)`.

        `rho = R * arccosh(z[0] / R)` recovers the radial geodesic distance to the
        origin on the curvature-`K` hyperboloid of radius `R = 1/sqrt(|K|)`; the
        direction is the normalized spatial block `u = z[1:] / ||z[1:]||` on
        `S^{d-1}`. Inverse of [`hyperbolic_polar_to_lorentz_cartesian`].

        Args:
            z (`torch.Tensor` of shape `(..., d + 1)`):
                Ambient Lorentz-Cartesian coordinates on the curvature-`K`
                hyperboloid with `z[0] >= R`.
            gaussian_curvature (`float`): curvature `K < 0`; `-1.0` = unit hyperboloid.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`:
                - `rhos` of shape `(...)`, `>= 0`.
                - `u` of shape `(..., d)`, unit direction on `S^{d-1}`.
        """
        R = GeoUtils._curvature_scale(gaussian_curvature)
        tiny = torch.finfo(z.dtype).tiny
        rhos = R * torch.acosh((z[..., 0] / R).clamp_min(1.0))
        spatial = z[..., 1:]
        u = spatial / spatial.norm(dim=-1, keepdim=True).clamp_min(tiny)
        return rhos, u

    @staticmethod
    def poincare_cartesian_to_lorentz_cartesian(
        z: torch.Tensor,
        gaussian_curvature: float=-1.0,
    ) -> torch.Tensor:
        """Convert Poincare-disk Cartesian to Lorentz-Cartesian via stereographic lift.

        At curvature `K` the map takes the ball of radius `R = 1/sqrt(|K|)` into the
        upper hyperboloid of radius `R` in `R^{1, d}`: the unit-model lift
        `w -> ((1 + ||w||^2) / (1 - ||w||^2), 2 w / (1 - ||w||^2))` applied to
        `w = z / R`, scaled back by `R`.

        Args:
            z (`torch.Tensor` of shape `(..., d)`):
                Poincare-disk Cartesian coordinates satisfying `||z|| < R`.
            gaussian_curvature (`float`): curvature `K < 0`; `-1.0` = unit models.

        Returns:
            `torch.Tensor` of shape `(..., d + 1)`: ambient Lorentz-Cartesian coordinates
            satisfying `-z[0]^2 + sum(z[1:]^2) = 1/K` (`= -1` at `K = -1`).
        """
        R = GeoUtils._curvature_scale(gaussian_curvature)
        norm_sq = (z * z).sum(-1) / (R * R)
        denom = (1.0 - norm_sq).clamp_min(torch.finfo(z.dtype).tiny)
        t = R * (1.0 + norm_sq) / denom
        spatial = 2.0 * z / denom.unsqueeze(-1)
        return torch.cat([t.unsqueeze(-1), spatial], dim=-1)

    @staticmethod
    def lorentz_cartesian_to_poincare_cartesian(
        z: torch.Tensor,
        gaussian_curvature: float=-1.0,
    ) -> torch.Tensor:
        """Convert Lorentz-Cartesian to Poincare-disk-Cartesian via stereographic projection.

        The map is `z -> R * z[..., 1:] / (R + z[..., 0])`, the inverse of
        [`poincare_cartesian_to_lorentz_cartesian`], taking the upper hyperboloid of
        radius `R = 1/sqrt(|K|)` in `R^{1, d}` into the open disk of radius `R`.

        Args:
            z (`torch.Tensor` of shape `(..., d + 1)`):
                Lorentz-Cartesian coordinates on the curvature-`K` hyperboloid with
                `z[0] >= R`.
            gaussian_curvature (`float`): curvature `K < 0`; `-1.0` = unit models.

        Returns:
            `torch.Tensor` of shape `(..., d)`: Poincare-disk Cartesian coordinates
            satisfying `||z|| < R`.
        """
        R = GeoUtils._curvature_scale(gaussian_curvature)
        spatial = z[..., 1:]
        t = z[..., 0]
        denom = (R + t).unsqueeze(-1).clamp_min(torch.finfo(z.dtype).tiny)
        return R * spatial / denom

class BinaryHyperbolicHeatKernel(GeoUtils):
    """Closed-form free hyperbolic heat kernel and bridge on `H^2` (d=2).

    Implements Gruet's series representation specialized to the disk: a Poisson
    count `n ~ Poisson(t / 8)`, a chi draw `s = sqrt(t) * chi(2n + 3)`, and a
    uniform mixing variable `v` jointly realize
    `rho = arccosh(v^2 + (1 - v^2) cosh(s))` distributed as the radial marginal
    of `H^2` Brownian motion at time `t`. The free heat kernel from the origin is
    rotationally symmetric, so its azimuthal angle is uniform on `[-pi, pi)`. The
    target-conditioned bridge methods reshape that uniform angle through the Poisson
    kernel `(cosh rho - sinh rho cos theta)^{-1}` before rotating it to the target.

    All methods are `@staticmethod` and run under `torch.no_grad()`; they accept
    a `ts` batch of heat times of shape `(batch_size,)` and an optional
    `output_coord` in `{Coordinate.HYPERBOLIC_POLAR, Coordinate.CARTESIAN}` selecting the
    return geometry.
    """

    @staticmethod
    def binary_free_hyperbolic_heat_kernel(
        ts: torch.FloatTensor,
    ):
        r"""Sample (rho, theta) from the free hyperbolic heat kernel on H^2.

        `rho` is the radial coordinate (the geodesic distance to the origin);
        `theta` is the angular coordinate, shared by the Poincare and Lorentz models.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`): heat times, `> 0`.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)`, each `torch.FloatTensor` of
            shape `(batch_size,)`; `rhos >= 0` and `thetas` uniform on `[-pi, pi)`.
        """
        ns = torch.poisson(ts/8).to(torch.int64)
        ss = ts.sqrt() * BinaryHyperbolicHeatKernel.sample_chi(2*ns+3, ts.dtype)
        vs = torch.rand_like(ts)
        rhos = torch.acosh(vs.square() + (1-vs.square())*torch.cosh(ss))
        us = torch.rand_like(ts)
        # theta ~ Uniform[-pi, pi): the free heat kernel from the origin is
        # rotationally symmetric, so the angle is uniform. (The target-conditioned
        # Poisson-kernel angle is applied later in the bridge methods, not here.)
        # `torch.pi` is a float constant, not a callable.
        thetas = (us - 0.5) * (2 * torch.pi)
        return rhos, thetas

    @staticmethod
    def binary_free_poincare_heat_kernel(
        ts: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        r"""Sample (rho, theta) from the free hyperbolic heat kernel on H^2.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`): heat times.
            output_coord (`str`, *optional*): `"polar"` (default) or `"cartesian"`.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)`, each `torch.FloatTensor` of
            shape `(batch_size,)`. CARTESIAN: `torch.FloatTensor` of shape
            `(batch_size, 2)`, the Poincare-disk point `z` with `||z|| < 1`.
        """
        rhos, thetas = BinaryHyperbolicHeatKernel.binary_free_hyperbolic_heat_kernel(ts=ts)
        if output_coord == Coordinate.CARTESIAN:
            return GeoUtils.binary_hyperbolic_polar_to_poincare_cartesian(rhos, thetas)
        return rhos, thetas

    @staticmethod
    def binary_free_lorentz_heat_kernel(
        ts: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        """Sample from the free hyperbolic heat kernel on `H^2` in Lorentz form.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            output_coord (`str`, *optional*, defaults to `Coordinate.CARTESIAN`):
                Either `Coordinate.HYPERBOLIC_POLAR` (returns `(rho, theta)`) or
                `Coordinate.CARTESIAN` (returns Lorentz-Cartesian coords).

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` each of shape `(batch_size,)`.
            CARTESIAN: `torch.FloatTensor` of shape `(batch_size, 3)`. Raises
            `ValueError` if `max(rho) > _LORENTZ_RHO_MAX`.
        """
        rhos, thetas = BinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel(
            ts=ts, output_coord=Coordinate.HYPERBOLIC_POLAR
        )
        if output_coord == Coordinate.HYPERBOLIC_POLAR:
            return rhos, thetas
        GeoUtils._check_lorentz_rho_bound(rhos, d=2, ts=ts)
        return GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(rhos, thetas)

    @staticmethod
    def binary_poincare_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        """Sample the `H^2` bridge endpoint conditioned on a target embedding.

        The free kernel's uniform angle is reshaped by the Poisson-kernel
        inverse-CDF (concentration `exp(-rho)`, giving angular density
        `(cosh rho - sinh rho cos theta)^{-1}`) and then rotated by
        `atan2(e[1], e[0])` so the sample concentrates near the target direction
        `e = word_embedding[targets]`.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            targets (`torch.LongTensor` of shape `(batch_size,)`):
                Vocabulary indices into `word_embedding`.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, 2)`):
                Word-embedding table. Only the angular part is used.
            output_coord (`str`, *optional*, defaults to `Coordinate.HYPERBOLIC_POLAR`):
                `Coordinate.HYPERBOLIC_POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` each of shape `(batch_size,)`.
                `thetas` is unwrapped (the target rotation may push it outside
                `(-pi, pi]`); downstream consumers use it only via `cos`/`sin`.
            CARTESIAN: Poincare-disk coordinates of shape `(batch_size, 2)`.
        """
        rhos, thetas = BinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel(
            ts=ts, output_coord=Coordinate.HYPERBOLIC_POLAR
        )
        # reshape the uniform free angle into the Poisson-kernel angle (concentration
        # exp(-rho), centered at 0): density (cosh rho - sinh rho cos theta)^{-1}.
        thetas = 2 * torch.atan((-rhos).exp() * torch.tan(thetas * 0.5))
        e = word_embedding[targets].to(ts.dtype)
        target_angle = torch.atan2(e[..., 1], e[..., 0])
        thetas = thetas + target_angle
        if output_coord == Coordinate.CARTESIAN:
            return GeoUtils.binary_hyperbolic_polar_to_poincare_cartesian(rhos, thetas)
        return rhos, thetas

    @staticmethod
    def binary_lorentz_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        """Lorentz-form `H^2` bridge endpoint conditioned on a target embedding.

        Lorentz analogue of [`binary_poincare_bridge`].

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            targets (`torch.LongTensor` of shape `(batch_size,)`):
                Vocabulary indices into `word_embedding`.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, 2)`):
                Word-embedding table.
            output_coord (`str`, *optional*, defaults to `Coordinate.CARTESIAN`):
                `Coordinate.HYPERBOLIC_POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` each of shape `(batch_size,)`.
            CARTESIAN: Lorentz-Cartesian coords of shape `(batch_size, 3)`. Raises
            `ValueError` if `max(rho) > _LORENTZ_RHO_MAX`.
        """
        rhos, thetas = BinaryHyperbolicHeatKernel.binary_poincare_bridge(
            ts=ts,
            targets=targets,
            word_embedding=word_embedding,
            output_coord=Coordinate.HYPERBOLIC_POLAR,
        )
        if output_coord == Coordinate.HYPERBOLIC_POLAR:
            return rhos, thetas
        GeoUtils._check_lorentz_rho_bound(rhos, d=2, ts=ts)
        return GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(rhos, thetas)

    @staticmethod
    def geodesic(
        t,
        gaussian_curvature: float = -1.0,
        src_cartesian: Optional[torch.FloatTensor] = None,
        dest_cartesian: Optional[torch.FloatTensor] = None,
        cartesian_model: Optional[str] = None,
        src_radial: Optional[torch.FloatTensor] = None,
        src_angular: Optional[torch.FloatTensor] = None,
        dest_radial: Optional[torch.FloatTensor] = None,
        dest_angular: Optional[torch.FloatTensor] = None,
        output_coord: Optional[str] = None,
    ):
        """Constant-speed hyperbolic geodesic on `H^2` from source to destination at fraction `t`.

        Each endpoint is accepted either as a Cartesian tensor
        (`src_cartesian` / `dest_cartesian`, interpreted per `cartesian_model` as
        Poincare-disk or Lorentz) or as a polar pair (`src_radial`, `src_angular`
        / `dest_radial`, `dest_angular`); exactly one form per endpoint must be
        provided. The intrinsic distance uses the differential form
        `cosh d - 1 = <x - y, x - y>_L / 2` to avoid cancellation at large `d`.

        Args:
            t (`float`, or `torch.Tensor` of shape `()` or `(batch_size, 1)`):
                Fraction along the geodesic (`0` -> source, `1` -> destination).
                A per-sample column `(batch_size, 1)` broadcasts against the
                `(batch_size, 3)` ambient points; a bare `(batch_size,)` vector
                does not and is unsupported.
            src_cartesian (`torch.FloatTensor`, *optional*):
                Cartesian source, interpreted per `cartesian_model`: shape
                `(batch_size, 3)` Lorentz when `cartesian_model == Geometry.LORENTZ`,
                or `(batch_size, 2)` Poincare-disk when `== Geometry.POINCARE`.
            dest_cartesian (`torch.FloatTensor`, *optional*):
                Cartesian destination; same shape/interpretation as `src_cartesian`.
            cartesian_model (`str`, *optional*):
                `Geometry.POINCARE` or `Geometry.LORENTZ`; the local chart of the
                Cartesian coordinates. Required whenever a cartesian endpoint is
                given or cartesian output is requested. Governs both endpoints.
            src_radial (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar radial coordinate of the source.
            src_angular (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar angle of the source.
            dest_radial (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar radial coordinate of the destination.
            dest_angular (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar angle of the destination.
            output_coord (`str`, *optional*):
                `Coordinate.CARTESIAN` or `Coordinate.HYPERBOLIC_POLAR`. Defaults to
                `Coordinate.CARTESIAN` when a cartesian source is given, else
                `Coordinate.HYPERBOLIC_POLAR`.

        Returns:
            CARTESIAN: chart-aware Cartesian output (requires `cartesian_model`) -
                `torch.FloatTensor` of shape `(batch_size, 3)` Lorentz-Cartesian when
                `cartesian_model == Geometry.LORENTZ`, or `(batch_size, 2)`
                Poincare-disk when `cartesian_model == Geometry.POINCARE`.
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` each `torch.FloatTensor` of shape
                `(batch_size,)`.

        Raises:
            ValueError: if neither or both forms of an endpoint are provided, if a
                cartesian endpoint or cartesian output lacks a valid `cartesian_model`,
                or if a polar input has `rho > _LORENTZ_RHO_MAX`.
        """
        if (src_cartesian is not None and (src_radial is not None or src_angular is not None)) or (
            src_cartesian is None and (src_radial is None or src_angular is None)
        ):
            raise ValueError(
                "Only accept one source, either src or (src_radial, src_angular)"
            )
        if (dest_cartesian is not None and (dest_radial is not None or dest_angular is not None)) or (
            dest_cartesian is None and (dest_radial is None or dest_angular is None)
        ):
            raise ValueError(
                "Only accept one destination, either dest or (dest_radial, dest_angular)"
            )

        if output_coord is None:
            output_coord = Coordinate.CARTESIAN if src_cartesian is not None else Coordinate.HYPERBOLIC_POLAR

        if gaussian_curvature >= 0.0:
            raise ValueError(
                f"hyperbolic geodesic requires gaussian_curvature < 0; got {gaussian_curvature}"
            )
        if src_cartesian is not None:
            if cartesian_model == Geometry.POINCARE:
                x_amb = GeoUtils.poincare_cartesian_to_lorentz_cartesian(
                    z=src_cartesian, gaussian_curvature=gaussian_curvature)
            elif cartesian_model == Geometry.LORENTZ:
                x_amb = src_cartesian
            else:
                raise ValueError(f"cartesian_model should be ({Geometry.POINCARE}, {Geometry.LORENTZ}), not {cartesian_model}.")
        else:
            GeoUtils._check_lorentz_rho_bound(
                src_radial, d=2, gaussian_curvature=gaussian_curvature)
            x_amb = GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(
                rhos=src_radial, thetas=src_angular, gaussian_curvature=gaussian_curvature)
        if dest_cartesian is not None:
            if cartesian_model == Geometry.POINCARE:
                y_amb = GeoUtils.poincare_cartesian_to_lorentz_cartesian(
                    z=dest_cartesian, gaussian_curvature=gaussian_curvature)
            elif cartesian_model == Geometry.LORENTZ:
                y_amb = dest_cartesian
            else:
                raise ValueError(f"cartesian_model should be ({Geometry.POINCARE}, {Geometry.LORENTZ}), not {cartesian_model}.")
        else:
            GeoUtils._check_lorentz_rho_bound(
                dest_radial, d=2, gaussian_curvature=gaussian_curvature)
            y_amb = GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(
                rhos=dest_radial, thetas=dest_angular, gaussian_curvature=gaussian_curvature)

        interpolate = BinaryHyperbolicHeatKernel._geodesic_kernel(
            x_amb, y_amb, t, gaussian_curvature=gaussian_curvature)

        if output_coord == Coordinate.CARTESIAN:
            if cartesian_model == Geometry.LORENTZ:
                return interpolate
            elif cartesian_model == Geometry.POINCARE:
                return GeoUtils.lorentz_cartesian_to_poincare_cartesian(
                    z=interpolate, gaussian_curvature=gaussian_curvature)
            else:
                raise ValueError(f"cartesian_model, {cartesian_model}, is not supported, only support ({Geometry.LORENTZ}, {Geometry.POINCARE}).")
        return GeoUtils.binary_lorentz_cartesian_to_hyperbolic_polar(
            interpolate, gaussian_curvature=gaussian_curvature)

class HyperbolicHeatKernel(GeoUtils):
    """Closed-form free hyperbolic heat kernel and bridge on `H^d` (`d >= 2`).

    The radial marginal is sampled from the *correct* `H^d` heat-kernel law
    `pi(rho) ∝ sinh^{d-1}(rho) p_H(rho; t)` (generator `½ Δ`), grounded in the
    Grigor'yan–Noguchi Millson recurrence — NOT the buggy `(d-1)^2 t/8` / `2n+d+1`
    extrapolation; see `unigram/notes/hyperbolic_heat_kernel_dd_derivation.md`. The
    **free** angle is uniform on `S^{d-1}` (rotational symmetry of `H^d` from the
    origin). The target-conditioned **bridge** draws the Poisson-kernel direction
    `(cosh rho - sinh rho <x,u>)^{-(d-1)}` (a Lorentz boost of a uniform direction,
    centred at `e_1`) and Householder-reflects `e_1 -> x` to the normalized target.

    Shapes use a sequence layout with `d = embedding_size`: `rhos` is
    `(batch_size, seq_len)`; the direction `u` / Poincare-ball point are
    `(batch_size, seq_len, d)`; the Lorentz point is `(batch_size, seq_len, d + 1)`.
    `ts` of shape `(batch_size,)` broadcasts across `seq_len`. All methods are
    `@staticmethod` under `torch.no_grad()`.
    """

    _RADIAL_NGRID: int = 2000
    _RADIAL_NU: int = 1024
    _RADIAL_BCHUNK: int = 16

    @staticmethod
    def _euclid_mean(d: int) -> float:
        """`E[chi_d] = sqrt(2) Gamma((d+1)/2)/Gamma(d/2)`: the short-time radial scale."""
        # lgamma form: the direct Gamma ratio overflows float range for d >= 343
        # (Gamma(256.5) is +inf), so evaluate the ratio in log-space.
        return math.sqrt(2.0) * math.exp(
            math.lgamma((d + 1) / 2.0) - math.lgamma(d / 2.0))

    @staticmethod
    def _ddx(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Central finite-difference `df/dx` along the last axis (per row)."""
        df = torch.empty_like(f)
        df[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (x[..., 2:] - x[..., :-2])
        df[..., 0] = (f[..., 1] - f[..., 0]) / (x[..., 1] - x[..., 0])
        df[..., -1] = (f[..., -1] - f[..., -2]) / (x[..., -1] - x[..., -2])
        return df

    @staticmethod
    def _mckean_base(rho: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """McKean `p_2(rho; t)` base for the even-`d` recurrence, up to a `rho`-independent
        constant. `rho`: `(B, ngrid)` (increasing per row), `ts`: `(B, 1)`. Returns `(B, ngrid)`.

        `p_2 ∝ ∫_0^inf s e^{-s^2/2t} (2/sinh s) du`, `s = arccosh(cosh rho + u^2)`; the
        `u^2 = cosh s - cosh rho` substitution removes the `1/sqrt(cosh s - cosh rho)`
        endpoint singularity. `u_max` is set per row so `e^{-s^2/2t}` is negligible beyond it.
        """
        tiny = torch.finfo(rho.dtype).tiny
        nu = HyperbolicHeatKernel._RADIAL_NU
        # Per-rho upper limit: the integrand mass sits at s in [rho, rho + ~8 sqrt(t)]
        # (e^{-s^2/2t} decreasing, lower limit s = rho). Nodes are placed uniformly in
        # s -- not in u = sqrt(cosh s - cosh rho), whose exponential stretching packs
        # the whole e^{-s^2/2t} mass into the first few nodes of a linear u-grid once
        # 8 sqrt(t) >> 1 (measured: ~10% radial-mean bias at d=2, t=5 against the
        # exact Gruet sampler). The trapezoid runs on the induced nonuniform u grid,
        # which still removes the 1/sqrt(cosh s - cosh rho) endpoint singularity.
        s_span = 8.0 * ts.sqrt() + 1.0                              # (B, 1)
        unit = torch.linspace(0.0, 1.0, nu, dtype=rho.dtype, device=rho.device)
        s = rho.unsqueeze(-1) + unit.view(1, 1, nu) * s_span.unsqueeze(-1)  # (B, ngrid, nu)
        uu = (torch.cosh(s) - torch.cosh(rho).unsqueeze(-1)).clamp_min(0.0).sqrt()
        integ = s * torch.exp(-s * s / (2.0 * ts.unsqueeze(-1))) * 2.0 / torch.sinh(s).clamp_min(tiny)
        return torch.trapezoid(integ, uu, dim=-1)                  # (B, ngrid)

    @staticmethod
    def sample_radial(
        ts: torch.FloatTensor,
        d: int,
        seq_len: int,
        gaussian_curvature: float=-1.0,
        exact_d3: bool=False,
    ) -> torch.FloatTensor:
        """Sample `seq_len` radial coordinates per heat time from the `H^d` heat-kernel
        marginal `pi(rho) ∝ sinh^{d-1}(rho) p_H(rho; t)` (generator `½ Δ`).

        Correct for all `d >= 2`: odd `d` applies the Millson curvature operator
        `O[f] = -f'/sinh rho`, `(d-1)/2` times, to the Gaussian core; even `d` seeds
        from the McKean `p_2` base and applies it `(d-2)/2` times. Inverse-CDF on an
        `x = rho/sqrt(t)` adaptive grid resolves every `t`. (The buggy reference
        `(d-1)^2 t/8` / `2n+d+1` Gruet extrapolation is NOT used.)

        Curvature enters by a single scaling. `H^d_K` is the unit `H^d` with every
        length multiplied by `R = 1/sqrt(|K|)`, so `Delta_K = R^-2 Delta_1` and Brownian
        motion on `H^d_K` for time `t` is, read in the unit model, Brownian motion for
        time `t / R^2`. The radial law is therefore `rho = R * rho_1(t / R^2)`: sample
        the unit-model marginal at the rescaled time and stretch the result by `R`. At
        `K = -1` both factors are exactly `1.0` and the computation is unchanged.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`): heat times `> 0`.
            d (`int`): hyperbolic dimension `>= 2`.
            seq_len (`int`): samples per heat time.
            exact_d3 (`bool`, *optional*, defaults to `False`): for `d == 3` only,
                invert the closed-form CDF instead of the quadrature grid. Exact
                and free of the `_radial_t_max` ceiling.
            gaussian_curvature (`float`, *optional*, defaults to `-1.0`):
                Gaussian (sectional) curvature `K < 0`, setting the model radius
                `R = 1/sqrt(|K|)`. `rho` is intrinsic (a geodesic distance), so it
                carries the units of `R`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, seq_len)`: radial samples `>= 0`.

        Note:
            The marginal `sinh^{d-1}(rho) p_H(rho; t)` is formed in linear (not log)
            space, so it overflows float64 once `rho ~ 709/(d-1)` — i.e. at very large
            `t`, since `E[rho] ~ (d-1) t / 2`. This covers the diffusion regime
            (`t = O(1)`) comfortably; the cartesian paths refuse `rho > 20 R` regardless,
            so only very-large-`t` POLAR output is affected (would need a log-space
            marginal). The overflow threshold is on the unit-model radial `rho / R` at
            the unit-model time `t / R^2`, which is what the quadrature sees. See
            `unigram/notes/hyperbolic_heat_kernel_dd_derivation.md`.
        """
        assert gaussian_curvature < 0.0, f"Hyperbolic gaussian_curvature should be negative"
        if d < 2:
            raise ValueError(f"HyperbolicHeatKernel requires d >= 2; got d={d}")
        B = ts.shape[0]
        if B == 0:
            return ts.new_empty(0, seq_len)
        R = GeoUtils._curvature_scale(gaussian_curvature)
        if exact_d3:
            if d == 3:
                return R * HyperbolicHeatKernel._sample_radial_exact_d3(ts / (R * R), seq_len)
        rho, cdf = HyperbolicHeatKernel.radial_cdf(ts / (R * R), d)
        u = torch.rand(B, seq_len, dtype=ts.dtype, device=ts.device)
        return R * HyperbolicHeatKernel._radial_inverse_cdf(rho, cdf, u)

    @staticmethod
    def _radial_cdf_exact_d3(rhos: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        r"""Closed-form CDF of the unit-`H^3` radial marginal.

        `pi(rho) ∝ sinh^2(rho) p_3(rho;t) = rho sinh(rho) e^{-rho^2/2t}` on
        `rho > 0`. Splitting `sinh` turns it into a difference of the `N(+-t, t)`
        densities, `pi(rho) = rho [phi_+(rho) - phi_-(rho)] / t` (the normaliser
        is exactly `E[N(t,t)] = t`), and `int_0^rho s phi(s) ds` is elementary:

            F(rho) = Phi(a) - Phi(-b) + (varphi(b) - varphi(a)) / sqrt(t),
            a = (rho - t)/sqrt(t),   b = (rho + t)/sqrt(t).

        `Phi(a) - Phi(-b)`, not the algebraically equal `Phi(b) - Phi(-a)`:
        `-b < a` always, so the subtracted term is the small one and the
        difference never cancels.

        Args:
            rhos (`torch.Tensor`): radial coordinates `>= 0`.
            ts (`torch.Tensor`): heat times `> 0`, broadcastable against `rhos`.

        Returns:
            `torch.Tensor`: `F(rho) in [0, 1]`, same broadcast shape.
        """
        st = ts.sqrt()
        a = (rhos - ts) / st
        b = (rhos + ts) / st
        norm = 1.0 / math.sqrt(2.0 * math.pi)
        pdf_b = norm * torch.exp(-0.5 * b * b)
        pdf_a = norm * torch.exp(-0.5 * a * a)
        return (
            torch.special.ndtr(a) - torch.special.ndtr(-b) + (pdf_b - pdf_a) / st
        ).clamp(0.0, 1.0)

    _EXACT_D3_BISECT: int = 48

    @staticmethod
    def _sample_radial_exact_d3(ts: torch.FloatTensor, seq_len: int) -> torch.FloatTensor:
        """Exact `H^3` radial sampler: inverse of `_radial_cdf_exact_d3` by bisection.

        No quadrature and no grid, so unlike `radial_cdf` it has no
        `_radial_t_max` ceiling -- the marginal is never formed in linear space.
        The bracket `[0, t + 14 sqrt(t)]` holds both regimes (`rho ~ sqrt(t)
        chi_3` as `t -> 0`, `rho ~ N(t, t)` as `t -> inf`), and 48 halvings take
        it to ~1e-14 of it, below float64's reach on `rho`.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`): heat times `> 0`.
            seq_len (`int`): samples per heat time.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, seq_len)`: radial samples `>= 0`.
        """
        tcol = ts.unsqueeze(-1)
        u = torch.rand(ts.shape[0], seq_len, dtype=ts.dtype, device=ts.device)
        lo = torch.zeros_like(u)
        hi = (tcol + 14.0 * tcol.sqrt()).expand_as(u).contiguous()
        for _ in range(HyperbolicHeatKernel._EXACT_D3_BISECT):
            mid = 0.5 * (lo + hi)
            below = HyperbolicHeatKernel._radial_cdf_exact_d3(mid, tcol) < u
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        return 0.5 * (lo + hi)

    @staticmethod
    def radial_cdf(ts: torch.FloatTensor, d: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The radial marginal's CDF on its per-`t` grid: `(rho, cdf)`, both `(B, ngrid)`,
        `rho` increasing per row and `cdf` running from 0 to 1. This is the expensive
        half of `sample_radial` (for even `d` the McKean base is a `(B, ngrid, nu)`
        quadrature); `_radial_inverse_cdf` is the cheap half.
        """
        B = ts.shape[0]
        tiny = torch.finfo(ts.dtype).tiny
        ng = HyperbolicHeatKernel._RADIAL_NGRID
        st = ts.sqrt().unsqueeze(-1)                               # (B,1)
        x_max = HyperbolicHeatKernel._euclid_mean(d) + 16.0 + 0.5 * (d - 1) * st  # (B,1)
        # grid in x = rho/sqrt(t); the 1e-3 lower clip is harmless because the
        # sinh^{d-1}(rho) volume factor kills the marginal at the origin for d >= 2.
        unit = torch.linspace(1e-3, 1.0, ng, dtype=ts.dtype, device=ts.device)
        xg = unit.unsqueeze(0) * x_max                            # (B, ng)
        rho = st * xg                                             # (B, ng)
        sinh_rho = torch.sinh(rho)
        if d % 2 == 1:
            f = torch.exp(-xg * xg / 2.0)
            k = (d - 1) // 2
        else:
            bc = HyperbolicHeatKernel._RADIAL_BCHUNK
            f = torch.cat(
                [HyperbolicHeatKernel._mckean_base(rho[i:i + bc], (st[i:i + bc]) ** 2)
                 for i in range(0, B, bc)],
                dim=0,
            )
            k = (d - 2) // 2
        for _ in range(k):
            f = -HyperbolicHeatKernel._ddx(f, rho) / sinh_rho.clamp_min(tiny)
        m = (sinh_rho ** (d - 1) * f).clamp_min(0.0)              # (B, ng)
        cdf = torch.cumulative_trapezoid(m, rho, dim=-1)          # (B, ng-1)
        cdf = torch.cat([torch.zeros(B, 1, dtype=ts.dtype, device=ts.device), cdf], dim=-1)
        return rho, cdf / cdf[:, -1:].clamp_min(tiny)

    @staticmethod
    def _radial_inverse_cdf(rho: torch.Tensor, cdf: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Linear-interpolated inverse of `radial_cdf`'s `(rho, cdf)` at `u` of shape
        `(B, k)`; returns `(B, k)`."""
        tiny = torch.finfo(rho.dtype).tiny
        ng = rho.shape[-1]
        idx = torch.searchsorted(cdf, u).clamp(1, ng - 1)         # (B, k)
        c_lo = torch.gather(cdf, 1, idx - 1); c_hi = torch.gather(cdf, 1, idx)
        r_lo = torch.gather(rho, 1, idx - 1); r_hi = torch.gather(rho, 1, idx)
        w = ((u - c_lo) / (c_hi - c_lo).clamp_min(tiny)).clamp(0.0, 1.0)
        return r_lo * (1.0 - w) + r_hi * w                        # (B, k)

    @staticmethod
    def _free_direction(shape, d: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Uniform direction on `S^{d-1}` (normalized Gaussian), shape `(*shape, d)`."""
        g = torch.randn(*shape, d, dtype=dtype, device=device)
        return g / g.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)

    @staticmethod
    def _angular_boost(
        rhos: torch.Tensor, d: int, gaussian_curvature: float = -1.0
    ) -> torch.Tensor:
        """Poisson-kernel direction centred at `e_1` via the Lorentz boost of rapidity
        `rho / R`, `R = 1/sqrt(|K|)`. `rhos`: `(...)`. Returns `(..., d)` unit vectors
        with angular density `(cosh(rho/R) - sinh(rho/R) <e_1, u>)^{-(d-1)}`
        (derivation note §4.1). The boost rapidity is the DIMENSIONLESS radial
        coordinate, so curvature enters only through `rho / R`; at `K = -1` (`R = 1`)
        this is the identity and the expression is unchanged.
        """
        tiny = torch.finfo(rhos.dtype).tiny
        unit_rhos = rhos / GeoUtils._curvature_scale(gaussian_curvature)
        u0 = HyperbolicHeatKernel._free_direction(rhos.shape, d, rhos.dtype, rhos.device)
        c0 = u0[..., 0]
        b = torch.exp(-2.0 * unit_rhos)
        T = ((1.0 + c0) + b * (1.0 - c0)).clamp_min(tiny)
        u_first = (1.0 - 2.0 * b * (1.0 - c0) / T).clamp(-1.0, 1.0)
        rest = (2.0 * torch.exp(-unit_rhos) / T).unsqueeze(-1) * u0[..., 1:]
        return torch.cat([u_first.unsqueeze(-1), rest], dim=-1)

    @staticmethod
    def _reflect_to_target(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Householder reflection mapping `e_1 -> x` applied to `u`. `x` unit, `(..., d)`."""
        d = u.shape[-1]
        e1 = torch.zeros(d, dtype=u.dtype, device=u.device)
        e1[0] = 1.0
        v = e1 - x
        v_norm_sq = (v * v).sum(-1, keepdim=True)
        dot = (u * v).sum(-1, keepdim=True)
        reflected = u - 2.0 * dot / v_norm_sq.clamp_min(1e-300) * v
        return torch.where(v_norm_sq > 1e-24, reflected, u)

    @staticmethod
    def free_hyperbolic_heat_kernel(
        ts: torch.FloatTensor,
        seq_len: int,
        embedding_size: int,
        gaussian_curvature: float = -1.0,
    ):
        r"""Sample `(rhos, u)` from the free hyperbolic heat kernel on `H^d`.

        `rho` is the radial coordinate (geodesic distance to the origin); `u` is the
        angular coordinate — a **unit direction on `S^{d-1}`**, uniform by rotational
        symmetry — shared by the Poincare and Lorentz models.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`): heat times, `> 0`.
            seq_len (`int`): tokens per sequence (draws per heat time).
            embedding_size (`int`): hyperbolic dimension `d >= 2`.
            gaussian_curvature (`float`, *optional*, defaults to `-1.0`):
                Gaussian (sectional) curvature `K < 0`. `rho` is intrinsic (a geodesic
                distance) and so carries the units of the model radius
                `R = 1/sqrt(|K|)`; the angular coordinate is dimensionless.
                `-1.0` = unit models.

        Returns:
            tuple `(rhos, u)`: `rhos` of shape `(batch_size, seq_len)` (`>= 0`), `u` of
            shape `(batch_size, seq_len, embedding_size)` (`||u|| == 1`, uniform on `S^{d-1}`).
        """
        assert gaussian_curvature < 0.0, f"Hyperbolic gaussian_curvature should be negative"
        d = embedding_size
        # Only the radial law feels the curvature: the free kernel from the origin is
        # rotationally symmetric on every `H^d_K`, so `u` stays uniform on `S^{d-1}`.
        rhos = HyperbolicHeatKernel.sample_radial(ts, d, seq_len, gaussian_curvature)
        u = HyperbolicHeatKernel._free_direction(rhos.shape, d, ts.dtype, ts.device)
        return rhos, u

    @staticmethod
    def free_poincare_heat_kernel(
        ts: torch.FloatTensor,
        seq_len: int,
        embedding_size: int,
        output_coord: Optional[str] = None,
        gaussian_curvature: float = -1.0,
    ):
        r"""Sample from the free `H^d` heat kernel in Poincare-ball form.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`): heat times.
            seq_len (`int`): tokens per sequence.
            embedding_size (`int`): hyperbolic dimension `d >= 2`.
            output_coord (`str`, *optional*): `Coordinate.HYPERBOLIC_POLAR` (default)
                or `Coordinate.CARTESIAN`.
            gaussian_curvature (`float`, *optional*, defaults to `-1.0`):
                Gaussian (sectional) curvature `K < 0`. `rho` carries the units of the
                model radius `R = 1/sqrt(|K|)` and the Cartesian output lives on the
                curvature-`K` model surface. `-1.0` = unit models.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, u)`, shapes `(batch_size, seq_len)` and
            `(batch_size, seq_len, embedding_size)`. CARTESIAN: Poincare-ball point of
            shape `(batch_size, seq_len, embedding_size)` with `||z|| < R`
            (`< 1` at `K = -1`).
        """
        assert gaussian_curvature < 0.0, f"Hyperbolic gaussian_curvature should be negative"
        rhos, u = HyperbolicHeatKernel.free_hyperbolic_heat_kernel(
            ts, seq_len, embedding_size, gaussian_curvature=gaussian_curvature
        )
        if output_coord == Coordinate.CARTESIAN:
            return GeoUtils.hyperbolic_polar_to_poincare_cartesian(
                rhos, u, gaussian_curvature=gaussian_curvature
            )
        return rhos, u

    @staticmethod
    def free_lorentz_heat_kernel(
        ts: torch.FloatTensor,
        seq_len: int,
        embedding_size: int,
        output_coord: Optional[str] = None,
        gaussian_curvature: float = -1.0,
    ):
        """Sample from the free `H^d` heat kernel in Lorentz-Cartesian form.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            seq_len (`int`): tokens per sequence.
            embedding_size (`int`): hyperbolic dimension `d >= 2`.
            output_coord (`str`, *optional*, defaults to `Coordinate.CARTESIAN`):
                `Coordinate.HYPERBOLIC_POLAR` (returns `(rhos, u)`) or
                `Coordinate.CARTESIAN` (returns Lorentz-Cartesian coords).
            gaussian_curvature (`float`, *optional*, defaults to `-1.0`):
                Gaussian (sectional) curvature `K < 0`. `rho` carries the units of the
                model radius `R = 1/sqrt(|K|)` and the Cartesian output lives on the
                curvature-`K` model surface. `-1.0` = unit models.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, u)`, shapes `(batch_size, seq_len)` and
            `(batch_size, seq_len, embedding_size)`. CARTESIAN: `torch.FloatTensor` of
            shape `(batch_size, seq_len, embedding_size + 1)` satisfying
            `<z,z>_L = 1/K`. Raises `ValueError` if `max(rho) > _LORENTZ_RHO_MAX * R`.
        """
        assert gaussian_curvature < 0.0, f"Hyperbolic gaussian_curvature should be negative"
        rhos, u = HyperbolicHeatKernel.free_poincare_heat_kernel(
            ts, seq_len, embedding_size, output_coord=Coordinate.HYPERBOLIC_POLAR,
            gaussian_curvature=gaussian_curvature,
        )
        if output_coord == Coordinate.HYPERBOLIC_POLAR:
            return rhos, u
        GeoUtils._check_lorentz_rho_bound(
            rhos, d=embedding_size, ts=ts, gaussian_curvature=gaussian_curvature
        )
        return GeoUtils.hyperbolic_polar_to_lorentz_cartesian(
            rhos, u, gaussian_curvature=gaussian_curvature
        )

    @staticmethod
    def poincare_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        output_coord: Optional[str] = None,
        gaussian_curvature: float = -1.0,
    ):
        """Sample the `H^d` bridge endpoint conditioned on a target embedding.

        Draws the radial coordinate from the heat-kernel marginal and the direction
        from the Poisson kernel `(cosh rho - sinh rho <x,u>)^{-(d-1)}` (a Lorentz boost
        centred at `e_1`), then Householder-reflects `e_1` to the normalized target
        direction `x = word_embedding[targets]`, so the sample concentrates around `x`.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            targets (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Vocabulary indices into `word_embedding`.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, embedding_size)`):
                Word-embedding table; `d = embedding_size` sets the hyperbolic dimension.
            output_coord (`str`, *optional*, defaults to `Coordinate.HYPERBOLIC_POLAR`):
                `Coordinate.HYPERBOLIC_POLAR` or `Coordinate.CARTESIAN`.
            gaussian_curvature (`float`, *optional*, defaults to `-1.0`):
                Gaussian (sectional) curvature `K < 0`. `rho` carries the units of the
                model radius `R = 1/sqrt(|K|)` and the Cartesian output lives on the
                curvature-`K` model surface. `-1.0` = unit models.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, u)`, shapes `(batch_size, seq_len)` and
                `(batch_size, seq_len, embedding_size)` (`u` a unit direction on
                `S^{d-1}` concentrated toward the target).
            CARTESIAN: Poincare-ball coordinates of shape
                `(batch_size, seq_len, embedding_size)`.
        """
        assert gaussian_curvature < 0.0, f"Hyperbolic gaussian_curvature should be negative"
        d = word_embedding.shape[-1]
        seq_len = targets.shape[-1]
        rhos = HyperbolicHeatKernel.sample_radial(ts, d, seq_len, gaussian_curvature)
        u = HyperbolicHeatKernel._angular_boost(rhos, d, gaussian_curvature)
        x = word_embedding[targets].to(ts.dtype)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(ts.dtype).tiny)
        u = HyperbolicHeatKernel._reflect_to_target(u, x)
        if output_coord == Coordinate.CARTESIAN:
            return GeoUtils.hyperbolic_polar_to_poincare_cartesian(
                rhos, u, gaussian_curvature=gaussian_curvature
            )
        return rhos, u

    @staticmethod
    def lorentz_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        output_coord: Optional[str] = None,
        gaussian_curvature: float = -1.0,
    ):
        """Lorentz-form `H^d` bridge endpoint conditioned on a target embedding.

        Lorentz analogue of [`poincare_bridge`].

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            targets (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Vocabulary indices into `word_embedding`.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, embedding_size)`):
                Word-embedding table; `d = embedding_size`.
            output_coord (`str`, *optional*, defaults to `Coordinate.CARTESIAN`):
                `Coordinate.HYPERBOLIC_POLAR` or `Coordinate.CARTESIAN`.
            gaussian_curvature (`float`, *optional*, defaults to `-1.0`):
                Gaussian (sectional) curvature `K < 0`. `rho` carries the units of the
                model radius `R = 1/sqrt(|K|)` and the Cartesian output lives on the
                curvature-`K` model surface. `-1.0` = unit models.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, u)`, shapes `(batch_size, seq_len)` and
                `(batch_size, seq_len, embedding_size)`.
            CARTESIAN: Lorentz-Cartesian coords of shape
                `(batch_size, seq_len, embedding_size + 1)` satisfying `<z,z>_L = 1/K`.
                Raises `ValueError` if `max(rho) > _LORENTZ_RHO_MAX * R`.
        """
        assert gaussian_curvature < 0.0, f"Hyperbolic gaussian_curvature should be negative"
        rhos, u = HyperbolicHeatKernel.poincare_bridge(
            ts, targets, word_embedding, output_coord=Coordinate.HYPERBOLIC_POLAR, gaussian_curvature=gaussian_curvature,
        )
        if output_coord == Coordinate.HYPERBOLIC_POLAR:
            return rhos, u
        GeoUtils._check_lorentz_rho_bound(
            rhos, d=word_embedding.shape[-1], ts=ts, gaussian_curvature=gaussian_curvature
        )
        return GeoUtils.hyperbolic_polar_to_lorentz_cartesian(
            rhos, u, gaussian_curvature=gaussian_curvature
        )

    @staticmethod
    def poincare_bridge_prod(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        output_coord: Optional[str] = None,
        prod_factor_dim: Optional[List[int]] = None,
        prod_factor_gaussian_curvature: Optional[List[float]] = None,
    ):
        """Sample the product-manifold bridge endpoint conditioned on a target embedding.

        The product `H^{d_1}_{K_1} x ... x H^{d_m}_{K_m}` carries the direct-sum
        Laplacian, so its Brownian motion is one independent motion per factor and the
        bridge factorizes exactly: each factor runs [`poincare_bridge`] on its own slice
        of the embedding — radial from that factor's curvature-`K_i` heat-kernel
        marginal, direction from its Poisson kernel, Householder-reflected onto the
        normalized target block `word_embedding[targets][..., slice_i]`. The factor
        dimensions partition `embedding_size`, so the concatenated Cartesian output has
        the same shape as [`poincare_bridge`]'s.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            targets (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Vocabulary indices into `word_embedding`.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, embedding_size)`):
                Word-embedding table; its last axis is split into the factor blocks.
            output_coord (`str`, *optional*, defaults to `Coordinate.HYPERBOLIC_POLAR`):
                `Coordinate.HYPERBOLIC_POLAR` or `Coordinate.CARTESIAN`.
            prod_factor_dim (`List[int]`, *optional*):
                Dimension `d_i >= 2` of each factor; must sum to `embedding_size`.
            prod_factor_gaussian_curvature (`List[float]`, *optional*):
                Curvature `K_i < 0` of each factor; same length as `prod_factor_dim`.
                Both lists default together to the single factor `[embedding_size]` at
                `[-1.0]`, which reproduces [`poincare_bridge`]; passing only one raises.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, u)` of shapes
                `(batch_size, seq_len, num_factors)` — one radial per factor, each the
                geodesic distance to that factor's origin — and
                `(batch_size, seq_len, embedding_size)`, the per-factor unit directions
                concatenated in factor order.
            CARTESIAN: the per-factor Poincare-ball points concatenated in factor order,
                shape `(batch_size, seq_len, embedding_size)`; block `i` satisfies
                `||z_i|| < R_i = 1/sqrt(|K_i|)`.
        """
        d = word_embedding.shape[-1]
        if prod_factor_dim is None and prod_factor_gaussian_curvature is None:
            prod_factor_dim = [d]
            prod_factor_gaussian_curvature = [-1.0]
        elif isinstance(prod_factor_dim, list) and isinstance(prod_factor_gaussian_curvature, list):
            assert len(prod_factor_dim) == len(prod_factor_gaussian_curvature)
        else:
            raise TypeError(
                "prod_factor_dim and prod_factor_gaussian_curvature must both be lists "
                f"or both be None; got {type(prod_factor_dim)} and "
                f"{type(prod_factor_gaussian_curvature)}."
            )
        assert sum(prod_factor_dim) == d, (
            f"prod_factor_dim {prod_factor_dim} should sum to the embedding size {d}"
        )
        for factor_dim, factor_curv in zip(prod_factor_dim, prod_factor_gaussian_curvature):
            assert factor_curv < 0.0, f"Hyperbolic curvature should be negative"
            assert factor_dim >= 2, f"Each product factor needs dim >= 2, not {factor_dim}"

        seq_len = targets.shape[-1]
        tiny = torch.finfo(ts.dtype).tiny
        x = word_embedding[targets].to(ts.dtype)
        rhos, us, offset = [], [], 0
        for factor_dim, factor_curv in zip(prod_factor_dim, prod_factor_gaussian_curvature):
            x_i = x[..., offset:offset + factor_dim]
            offset += factor_dim
            x_i = x_i / x_i.norm(dim=-1, keepdim=True).clamp_min(tiny)
            rho_i = HyperbolicHeatKernel.sample_radial(ts, factor_dim, seq_len, factor_curv)
            u_i = HyperbolicHeatKernel._angular_boost(rho_i, factor_dim, factor_curv)
            rhos.append(rho_i)
            us.append(HyperbolicHeatKernel._reflect_to_target(u_i, x_i))

        if output_coord == Coordinate.CARTESIAN:
            return torch.cat(
                [
                    GeoUtils.hyperbolic_polar_to_poincare_cartesian(
                        rho_i, u_i, gaussian_curvature=factor_curv
                    )
                    for rho_i, u_i, factor_curv in zip(rhos, us, prod_factor_gaussian_curvature)
                ],
                dim=-1,
            )
        return torch.stack(rhos, dim=-1), torch.cat(us, dim=-1)

    @staticmethod
    def geodesic(
        t,
        gaussian_curvature: float = -1.0,
        src_cartesian: Optional[torch.FloatTensor] = None,
        dest_cartesian: Optional[torch.FloatTensor] = None,
        cartesian_model: Optional[str] = None,
        src_radial: Optional[torch.FloatTensor] = None,
        src_angular: Optional[torch.FloatTensor] = None,
        dest_radial: Optional[torch.FloatTensor] = None,
        dest_angular: Optional[torch.FloatTensor] = None,
        output_coord: Optional[str] = None,
    ):
        """Constant-speed hyperbolic geodesic on `H^d` from source to destination at fraction `t`.

        Each endpoint is accepted either as a Cartesian tensor
        (`src_cartesian` / `dest_cartesian`, interpreted per `cartesian_model` as
        Poincare-disk or Lorentz) or as a polar pair (`src_radial`, `src_angular`
        / `dest_radial`, `dest_angular`); exactly one form per endpoint must be
        provided. The intrinsic distance uses the differential form
        `cosh d - 1 = <x - y, x - y>_L / 2` to avoid cancellation at large `d`.

        Args:
            t (`float`, or `torch.Tensor` of shape `()` or `(batch_size, seq_len, 1)`):
                Fraction along the geodesic (`0` -> source, `1` -> destination).
                A per-sample column `(batch_size, seq_len, 1)` broadcasts against the
                `(batch_size, seq_len, embedding_size + 1)` ambient points;
                a bare `(batch_size, seq_len)` vector does not and is unsupported.
            gaussian_curvature (`float`, defaults to `-1.0`):
                Gaussian (sectional) curvature `K < 0` of the hyperbolic space.
                Cartesian endpoints/outputs live on the curvature-`K` model surfaces
                (hyperboloid / Poincare ball of radius `R = 1/sqrt(|K|)`); polar
                coordinates are intrinsic (`rho` = geodesic distance) and
                chart-free. Conversions and the interpolation kernel are natively
                curvature-aware. `-1.0` = unit models.
            src_cartesian (`torch.FloatTensor`, *optional*):
                Cartesian source, interpreted per `cartesian_model`: shape
                `(batch_size, seq_len, embedding_size + 1)` Lorentz when 
                `cartesian_model == Geometry.LORENTZ`,
                or `(batch_size, seq_len, embedding_size)` Poincare-disk 
                when `== Geometry.POINCARE`.
            dest_cartesian (`torch.FloatTensor`, *optional*):
                Cartesian destination; same shape/interpretation as `src_cartesian`.
            cartesian_model (`str`, *optional*):
                `Geometry.POINCARE` or `Geometry.LORENTZ`; the local chart of the
                Cartesian coordinates. Required whenever a cartesian endpoint is
                given or cartesian output is requested. Governs both endpoints.
            src_radial (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Polar radial coordinate of the source.
            src_angular (`torch.FloatTensor` of shape `(batch_size, seq_len, embedding_size)`, *optional*):
                Polar direction (unit vector on `S^{d-1}`) of the source.
            dest_radial (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Polar radial coordinate of the destination.
            dest_angular (`torch.FloatTensor` of shape `(batch_size, seq_len, embedding_size)`, *optional*):
                Polar direction (unit vector on `S^{d-1}`) of the destination.
            output_coord (`str`, *optional*):
                `Coordinate.CARTESIAN` or `Coordinate.HYPERBOLIC_POLAR`. Defaults to
                `Coordinate.CARTESIAN` when a cartesian source is given, else
                `Coordinate.HYPERBOLIC_POLAR`.

        Returns:
            CARTESIAN: chart-aware Cartesian output (requires `cartesian_model`) -
                `torch.FloatTensor` of shape `(batch_size, seq_len, embedding_size + 1)` 
                Lorentz-Cartesian when `cartesian_model == Geometry.LORENTZ`, or 
                `(batch_size, seq_len, embedding_size)` Poincare-disk when 
                `cartesian_model == Geometry.POINCARE`.
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` with `torch.FloatTensor` of shape
                `(batch_size, seq_len)` and `(batch_size, seq_len, embedding_size)`

        Raises:
            ValueError: if neither or both forms of an endpoint are provided, if a
                cartesian endpoint or cartesian output lacks a valid `cartesian_model`,
                or if a polar input has `rho > _LORENTZ_RHO_MAX`.
        """
        if (src_cartesian is not None and (src_radial is not None or src_angular is not None)) or (
            src_cartesian is None and (src_radial is None or src_angular is None)
        ):
            raise ValueError(
                "Only accept one source, either src_cartesian or (src_radial, src_angular)"
            )
        if (dest_cartesian is not None and (dest_radial is not None or dest_angular is not None)) or (
            dest_cartesian is None and (dest_radial is None or dest_angular is None)
        ):
            raise ValueError(
                "Only accept one destination, either dest_cartesian or (dest_radial, dest_angular)"
            )

        if output_coord is None:
            output_coord = Coordinate.CARTESIAN if src_cartesian is not None else Coordinate.HYPERBOLIC_POLAR

        if gaussian_curvature >= 0.0:
            raise ValueError(
                f"hyperbolic geodesic requires gaussian_curvature < 0; got {gaussian_curvature}"
            )
        if src_cartesian is not None:
            if cartesian_model == Geometry.POINCARE:
                x_amb = GeoUtils.poincare_cartesian_to_lorentz_cartesian(
                    z=src_cartesian, gaussian_curvature=gaussian_curvature)
            elif cartesian_model == Geometry.LORENTZ:
                x_amb = src_cartesian
            else:
                raise ValueError(f"cartesian_model should be ({Geometry.POINCARE}, {Geometry.LORENTZ}), not {cartesian_model}.")
        else:
            GeoUtils._check_lorentz_rho_bound(
                src_radial, d=src_angular.shape[-1],
                gaussian_curvature=gaussian_curvature)
            x_amb = GeoUtils.hyperbolic_polar_to_lorentz_cartesian(
                rhos=src_radial, thetas=src_angular, gaussian_curvature=gaussian_curvature)
        if dest_cartesian is not None:
            if cartesian_model == Geometry.POINCARE:
                y_amb = GeoUtils.poincare_cartesian_to_lorentz_cartesian(
                    z=dest_cartesian, gaussian_curvature=gaussian_curvature)
            elif cartesian_model == Geometry.LORENTZ:
                y_amb = dest_cartesian
            else:
                raise ValueError(f"cartesian_model should be ({Geometry.POINCARE}, {Geometry.LORENTZ}), not {cartesian_model}.")
        else:
            GeoUtils._check_lorentz_rho_bound(
                dest_radial, d=dest_angular.shape[-1],
                gaussian_curvature=gaussian_curvature)
            y_amb = GeoUtils.hyperbolic_polar_to_lorentz_cartesian(
                rhos=dest_radial, thetas=dest_angular, gaussian_curvature=gaussian_curvature)

        interpolate = GeoUtils._geodesic_kernel(
            x_amb, y_amb, t, gaussian_curvature=gaussian_curvature)

        if output_coord == Coordinate.CARTESIAN:
            if cartesian_model == Geometry.LORENTZ:
                return interpolate
            elif cartesian_model == Geometry.POINCARE:
                return GeoUtils.lorentz_cartesian_to_poincare_cartesian(
                    z=interpolate, gaussian_curvature=gaussian_curvature)
            else:
                raise ValueError(
                    f"cartesian_model, {cartesian_model}, is not supported, only "
                    f"support ({Geometry.LORENTZ}, {Geometry.POINCARE})."
                )
        return GeoUtils.lorentz_cartesian_to_hyperbolic_polar(
            interpolate, gaussian_curvature=gaussian_curvature)