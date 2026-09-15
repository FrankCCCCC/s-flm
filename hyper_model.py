import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from geo_bridge import GeoUtils

class HyperbolicModelBase(ABC):
    """Shared readout for models whose state lives on a product of Poincare balls.

    The manifold is `H^{d_1}_{K_1} x ... x H^{d_m}_{K_m}` and the state is the
    `Coordinate.HYPERBOLIC_POLAR` output of
    `HyperbolicHeatKernel.poincare_bridge_prod`: one radial coordinate per factor
    (`radius`, last axis `m`) plus the per-factor unit boundary directions
    concatenated (`theta`, last axis `sum(d_i) == embedding_size`). Both lists
    `None` means the single factor `[embedding_size]` at `[-1.0]`, i.e. plain
    `H^d` -- the same default `poincare_bridge_prod` uses.

    Subclasses supply the trunk (`model_forward`), the boundary table
    (`word_embedding`), and three attributes this class reads: `lm_head`
    (the trunk-features -> vocabulary readout), `output_radial_dim` (how many
    trailing trunk channels are the predicted radius rather than features), and
    the factor spec `prod_factor_dim` / `prod_factor_gaussian_curvature` the
    horosphere readout defaults to. `word_embedding` is required, not optional:
    the horosphere readout has no table of its own to fall back on.

    Two readouts, differing only in who adds the geometry:
      naive       -- returns `lm_head`'s logits untouched. Per Invariant 1 those
                     are a RESIDUAL, so the consumer must add
                     `horosphere_geometry` itself (this is what `loss.py` does).
      horosphere  -- adds `horosphere_geometry` here, so the returned logits are
                     already the full log-posterior and must NOT be corrected a
                     second time.
    """

    def __init__(
        self,
    ):
        super().__init__()

    @property
    @abstractmethod
    def word_embedding(self) -> torch.Tensor:
        pass

    @staticmethod
    def prod_factors(
        prod_factor_dim: Optional[Union[int, List[int]]],
        prod_factor_gaussian_curvature: Optional[Union[float, List[float]]],
        embedding_size: int,
    ):
        """
        Resolve and validate the product-factor split of a boundary of dimension
        `embedding_size`.

        Returns:
            `tuple[List[int], List[float]]`: the per-factor dimensions `d_i >= 2`
                (summing to `embedding_size`) and curvatures `K_i < 0`.
        """
        dims = prod_factor_dim
        curvatures = prod_factor_gaussian_curvature
        if dims is None and curvatures is None:
            return [embedding_size], [-1.0]
        if not (isinstance(dims, list) and isinstance(curvatures, list)):
            raise TypeError(
                "prod_factor_dim and prod_factor_gaussian_curvature must both be "
                f"lists or both be None; got {type(dims)} and {type(curvatures)}."
            )
        if len(dims) != len(curvatures):
            raise ValueError(
                f"prod_factor_dim {dims} and prod_factor_gaussian_curvature "
                f"{curvatures} must have the same length."
            )
        if sum(dims) != embedding_size:
            raise ValueError(
                f"prod_factor_dim {dims} should sum to the embedding size {embedding_size}."
            )
        for factor_dim, factor_curvature in zip(dims, curvatures):
            if factor_dim < 2:
                raise ValueError(f"Each product factor needs dim >= 2, not {factor_dim}.")
            if factor_curvature >= 0.0:
                raise ValueError(f"Hyperbolic curvature should be negative, not {factor_curvature}.")
        return dims, curvatures

    def forward_combined(
        self,
        z: torch.Tensor,
        theta: torch.Tensor,
        radius: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        forward_type: str = "naive",
        return_radial: bool=False,
        prod_factor_dim: Optional[Union[int, List[int]]] = None,
        prod_factor_gaussian_curvature: Optional[Union[float, List[float]]] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Predict vocabulary logits from a time-conditioned state.

        Args:
            z (`torch.Tensor` of shape `(batch_size, max_seq_len, input_theta_dim)`):
                Input state in Cartesian coordinate
            theta (`torch.Tensor` of shape `(batch_size, max_seq_len, input_theta_dim)`):
                Input state in polar coordinate, angles
            radius (`torch.Tensor` of shape `(batch_size, max_seq_len, input_radius_dim)`):
                Input state in polar coordinate, radius, consider product manifold
            t (`torch.Tensor` of shape `(batch_size,)` or `(batch_size, 1)`):
                Per-example time values, optional
            forward_type (`str`, *optional*, defaults to `"naive"`):
                `"naive"` for the residual logits, `"horosphere"` for the
                geometry-corrected ones.
            return_radial (`bool`, *optional*, defaults to `False`):
                Also return the trunk's radial prediction.
            prod_factor_dim (`Union[int, List[int]]`, *optional*):
                Product-factor split, `"horosphere"` only. Both `None` falls
                back to the model's own factors.
            prod_factor_gaussian_curvature (`Union[float, List[float]]`, *optional*):
                Curvature `K_i < 0` of each factor.

        Returns:
            `torch.Tensor` of shape `(batch_size, max_seq_len, vocab_size)`:
                Vocabulary logits.
            `torch.Tensor` of shape `(batch_size, max_seq_len, output_radial_dim)`:
                if return_radial, the predicted radius
        """
        if forward_type == "naive":
            return self.forward_naive(
                z=z,
                theta=theta,
                radius=radius,
                prod_factor_dim=prod_factor_dim,
                prod_factor_gaussian_curvature=prod_factor_gaussian_curvature,
                t=t,
                return_radial=return_radial,
            )
        elif forward_type == "horosphere":
            return self.forward_horosphere(
                z=z,
                theta=theta,
                radius=radius,
                prod_factor_dim=prod_factor_dim,
                prod_factor_gaussian_curvature=prod_factor_gaussian_curvature,
                t=t,
                return_radial=return_radial,
            )
        else:
            raise ValueError(f"forward_type, {forward_type}, is not supported.")

    @abstractmethod
    def model_forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
    ):
        pass

    def forward_naive(
        self,
        z: torch.Tensor,
        theta: torch.Tensor,
        radius: torch.Tensor,
        prod_factor_dim: Optional[Union[int, List[int]]] = None,
        prod_factor_gaussian_curvature: Optional[Union[float, List[float]]] = None,
        t: Optional[torch.Tensor] = None,
        return_radial: bool=False
    ) -> Tuple[torch.Tensor]:
        """
        Predict vocabulary logits from a time-conditioned state.

        The state comes in ONE of the two coordinate systems: either Cartesian
        `z`, or polar `(theta, radius)` -- concatenated on the last axis, so the
        trunk sees `input_theta_dim + input_radius_dim` channels.

        Args:
            z (`torch.Tensor` of shape `(batch_size, max_seq_len, input_theta_dim)`):
                Input state in Cartesian coordinate
            theta (`torch.Tensor` of shape `(batch_size, max_seq_len, input_theta_dim)`):
                Input state in polar coordinate, angles
            radius (`torch.Tensor` of shape `(batch_size, max_seq_len, input_radius_dim)`):
                Input state in polar coordinate, radius, consider product manifold
            t (`torch.Tensor` of shape `(batch_size,)` or `(batch_size, 1)`):
                Per-example time values, optional
            return_radial (`bool`, *optional*, defaults to `False`):
                Also return the trunk's radial prediction.

        Returns:
            `torch.Tensor` of shape `(batch_size, max_seq_len, vocab_size)`:
                Vocabulary logits.
            `torch.Tensor` of shape `(batch_size, max_seq_len, output_radial_dim)`:
                if return_radial, the predicted radius
        """
        if z is not None and (theta is not None or radius is not None):
            raise ValueError(
                "Pass the state either as Cartesian z or as polar (theta, radius), not both."
            )
        if z is None and (theta is None or radius is None):
            raise ValueError(
                "The polar state needs BOTH theta and radius; got "
                f"theta={type(theta)}, radius={type(radius)}."
            )

        input = None
        if z is not None:
            input = z
        else:
            # The TRUNK is fed the dimensionless radius u = kappa*rho, so the
            # predictor is scale-equivariant (the Bayes posterior depends on
            # (theta, u), not on rho). horosphere_geometry below still receives
            # the intrinsic radius and applies kappa itself -- rescaling here
            # only, so the curvature is never applied twice.
            _, curvatures = HyperbolicModelBase.prod_factors(
                prod_factor_dim=prod_factor_dim,
                prod_factor_gaussian_curvature=prod_factor_gaussian_curvature,
                embedding_size=theta.shape[-1],
            )
            kappas = radius.new_tensor(
                [1.0 / GeoUtils._curvature_scale(k) for k in curvatures]
            )
            input = torch.cat([theta, radius * kappas], dim=-1)

        output = self.model_forward(z=input, t=t)
        # The trunk emits the boundary features first and the radial channels
        # last; splitting by a positive index (rather than -output_radial_dim)
        # keeps output_radial_dim == 0 -- a model that predicts no radius -- from
        # slicing the features away entirely.
        split = output.shape[-1] - self.output_radial_dim
        if return_radial:
            return self.lm_head(output[..., :split]), output[..., split:]
        return self.lm_head(output[..., :split])

    @staticmethod
    def radius_conversion(
        radius: torch.Tensor,
        prod_factor_dim: Union[int, List[int]],
        prod_factor_gaussian_curvature: Union[float, List[float]],
    ):
        """
        Convert to radius on Poincare disk, consider product manifold

        Factor `i` has model radius `R_i = 1/sqrt(|K_i|)` and its intrinsic
        (geodesic) radial coordinate maps to the ball radius
        `R_i tanh(rho_i / 2 R_i)` -- the radial part of
        `GeoUtils.hyperbolic_polar_to_poincare_cartesian`, so the result is
        bounded by that factor's own ball radius. Scalar arguments describe the
        single-factor case.

        Args:
            radius (`torch.Tensor` of shape `(..., num_factors)`):
                Intrinsic radial coordinate of each product factor.
            prod_factor_dim (`int` or `List[int]`):
                Dimension of each factor. Only its length is used here; the
                conversion depends on the curvature alone.
            prod_factor_gaussian_curvature (`float` or `List[float]`):
                Curvature `K_i < 0` of each factor.

        Returns:
            `torch.Tensor` of shape `(..., num_factors)`:
                Poincare-ball radius of each factor.
        """
        if isinstance(prod_factor_dim, list) != isinstance(prod_factor_gaussian_curvature, list):
            raise TypeError(
                "prod_factor_dim and prod_factor_gaussian_curvature must both be "
                f"lists or both be scalars; got {type(prod_factor_dim)} and "
                f"{type(prod_factor_gaussian_curvature)}."
            )
        if not isinstance(prod_factor_dim, list):
            prod_factor_dim = [prod_factor_dim]
            prod_factor_gaussian_curvature = [prod_factor_gaussian_curvature]
        assert len(prod_factor_dim) == len(prod_factor_gaussian_curvature), (
            f"prod_factor_dim {prod_factor_dim} and prod_factor_gaussian_curvature "
            f"{prod_factor_gaussian_curvature} must have the same length."
        )
        assert radius.shape[-1] == len(prod_factor_dim), (
            f"radius must carry one radial coordinate per product factor "
            f"({len(prod_factor_dim)}); got {radius.shape[-1]}."
        )
        model_radius = radius.new_tensor(
            [GeoUtils._curvature_scale(k) for k in prod_factor_gaussian_curvature]
        )
        return model_radius * torch.tanh(radius / (2.0 * model_radius))

    def horosphere_geometry(
        self,
        theta: torch.Tensor,
        radius: torch.Tensor,
        prod_factor_dim: Optional[Union[int, List[int]]] = None,
        prod_factor_gaussian_curvature: Optional[Union[float, List[float]]] = None,
    ):
        """
        Horocycle distance (Poisson kernel) for each word embedding.

        `horosphere_dists[..., v] = sum_i -(d_i - 1) B^i_v(z_i)` -- `B^i_v` the
        Busemann function of word `v`'s boundary point in factor `i`. It is the
        log density of the bridge direction at word `v`, up to a `v`-independent
        constant, so `softmax(horosphere_dists + log p)` is exactly the Bayes
        posterior `q(y | z_t)`; the factors are independent Brownian motions, so
        their Busemann terms simply add. This is the product-manifold form of
        `HyperBridge.horosphere_geometry`, and every consumer of the naive logits
        must treat them as a RESIDUAL on top of this term.

        The half-angle form is the load-bearing one (Invariant 5):
        `cosh s - sinh s <u, phi_v> = e^{+s} sin^2(a_v/2) + e^{-s} cos^2(a_v/2)`,
        with `s = rho_i / R_i` the DIMENSIONLESS radial -- curvature enters only
        here, exactly as in `HyperbolicHeatKernel._angular_boost`. Pulling
        `e^{+s}` out of the log makes it overflow-free, the squared-difference
        forms are cancellation-free, and `clamp_min` keeps the log finite once
        both terms underflow (`s > ~372`, where `sin_half_sq` is exactly 0 for
        the target word). Computed in float64 per Invariant 3.

        Args:
            theta (`torch.Tensor` of shape `(batch_size, max_seq_len, input_theta_dim)`):
                Per-factor unit boundary directions, concatenated.
            radius (`torch.Tensor` of shape `(batch_size, max_seq_len, input_radius_dim)`):
                Intrinsic radial coordinate of each product factor.
            prod_factor_dim (`Union[int, List[int]]`, *optional*):
                Dimension of each factor; they must all be EQUAL here. Both
                lists `None` means the single factor `[embedding_size]` at
                `[-1.0]`.
            prod_factor_gaussian_curvature (`Union[float, List[float]]`, *optional*):
                Curvature `K_i < 0` of each factor, free to differ per factor.

        Returns:
            `torch.Tensor` of shape `(batch_size, max_seq_len, vocab_size)`:
                Horosphere (Busemann) log-densities, float64.
        """
        embedding = self.word_embedding
        if embedding is None:
            raise ValueError(
                f"{type(self).__name__}.word_embedding is None; the horosphere "
                "readout needs the per-word boundary directions. Every model on "
                "this path owns a table -- MLPLMRefactor's lm_head.weight, "
                "OptimalModelRefactor's frozen uniform_sphere_points buffer."
            )
        embedding = embedding.to(torch.float64)
        dims, curvatures = HyperbolicModelBase.prod_factors(
            prod_factor_dim=prod_factor_dim,
            prod_factor_gaussian_curvature=prod_factor_gaussian_curvature,
            embedding_size=embedding.shape[-1],
        )
        if theta.shape[-1] != embedding.shape[-1]:
            raise ValueError(
                f"theta must carry one direction per factor, {embedding.shape[-1]} "
                f"channels in total; got {theta.shape[-1]}."
            )
        if radius.shape[-1] != len(dims):
            raise ValueError(
                f"radius must carry one radial coordinate per product factor "
                f"({len(dims)}); got {radius.shape[-1]}."
            )
        if len(set(dims)) != 1:
            raise ValueError(
                f"horosphere_geometry needs one shared factor dimension; got {dims}."
            )
        theta = theta.to(torch.float64)
        radius = radius.to(torch.float64)
        tiny = torch.finfo(torch.float64).tiny

        # No python loop over factors: one shared factor dimension splits the
        # concatenated boundary axis by a reshape, and the whole formula is
        # elementwise in the resulting factor axis, which the final sum
        # contracts away. Curvature stays per-factor, as the `kappas` vector.
        factor_dim, num_factors = dims[0], len(dims)
        # us: (..., 1, m, d) against phis: (V, m, d) -> (..., V, m, d)
        us = theta.unflatten(-1, (num_factors, factor_dim)).unsqueeze(-3)
        phis = embedding.unflatten(-1, (num_factors, factor_dim))
        phis = phis / phis.norm(dim=-1, p=2, keepdim=True).clamp_min(tiny)
        #   sin^2(a_v/2) = (1 - <u, phi_v>) / 2 = ||u - phi_v||^2 / 4
        #   cos^2(a_v/2) = (1 + <u, phi_v>) / 2 = ||u + phi_v||^2 / 4
        sin_half_sq = (us - phis).square().sum(-1) / 4
        cos_half_sq = (us + phis).square().sum(-1) / 4
        kappas = radius.new_tensor(
            [1.0 / GeoUtils._curvature_scale(k) for k in curvatures]
        )
        ss = (radius * kappas).unsqueeze(-2)
        return -(factor_dim - 1) * (
            ss + (
                sin_half_sq + cos_half_sq * (-2.0 * ss).exp()
            ).clamp_min(tiny).log()
        ).sum(-1)

    def forward_horosphere(
        self,
        z: torch.Tensor,
        theta: torch.Tensor,
        radius: torch.Tensor,
        prod_factor_dim: Optional[Union[int, List[int]]] = None,
        prod_factor_gaussian_curvature: Optional[Union[float, List[float]]] = None,
        t: Optional[torch.Tensor] = None,
        return_radial: bool=False
    ) -> Tuple[torch.Tensor]:
        """
        Predict vocabulary logits from a time-conditioned state.

        The returned logits ALREADY carry the horosphere geometry, i.e. they are
        the log-posterior up to a word-independent constant: `softmax` over them
        is the model's posterior over words. Unlike `forward_naive`'s residual
        logits they must not be corrected a second time (Invariant 1).

        Args:
            z (`torch.Tensor` of shape `(batch_size, max_seq_len, input_theta_dim)`):
                Input state in Cartesian coordinate
            theta (`torch.Tensor` of shape `(batch_size, max_seq_len, input_theta_dim)`):
                Input state in polar coordinate, angles
            radius (`torch.Tensor` of shape `(batch_size, max_seq_len, input_radius_dim)`):
                Input state in polar coordinate, radius on Poincare disk model, consider product manifold
            prod_factor_dim (`Union[int, List[int]]`, *optional*):
                Dimension of each factor. Both `None` falls back to the model's
                own `prod_factor_dim` / `prod_factor_gaussian_curvature`.
            prod_factor_gaussian_curvature (`Union[float, List[float]]`, *optional*):
                Curvature `K_i < 0` of each factor.
            t (`torch.Tensor` of shape `(batch_size,)` or `(batch_size, 1)`):
                Per-example time values, optional
            return_radial (`bool`, *optional*, defaults to `False`):
                Also return the trunk's radial prediction.

        Returns:
            `torch.Tensor` of shape `(batch_size, max_seq_len, vocab_size)`:
                Vocabulary logits, float64.
            `torch.Tensor` of shape `(batch_size, max_seq_len, output_radial_dim)`:
                if return_radial, the predicted radius
        """
        if theta is None or radius is None:
            raise ValueError(
                "The horosphere readout needs the polar state (theta, radius) to "
                "evaluate the Busemann terms."
            )
        pred_radius = None
        if return_radial:
            pred_logit, pred_radius = self.forward_naive(
                z=z, theta=theta, radius=radius, t=t, return_radial=True,
                prod_factor_dim=prod_factor_dim,
                prod_factor_gaussian_curvature=prod_factor_gaussian_curvature,
            )
        else:
            pred_logit = self.forward_naive(
                z=z, theta=theta, radius=radius, t=t, return_radial=False,
                prod_factor_dim=prod_factor_dim,
                prod_factor_gaussian_curvature=prod_factor_gaussian_curvature,
            )
        horo_dist = self.horosphere_geometry(
            theta=theta,
            radius=radius,
            prod_factor_dim=prod_factor_dim,
            prod_factor_gaussian_curvature=prod_factor_gaussian_curvature,
        )
        pred_logit = pred_logit.to(horo_dist.dtype) + horo_dist

        if return_radial:
            return pred_logit, pred_radius
        return pred_logit