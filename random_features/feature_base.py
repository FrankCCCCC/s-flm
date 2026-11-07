from typing import List, Tuple, Union

import torch
import torch.nn as nn

from numeric.geo_bridge import GeoUtils, Coordinate

class FeatureBase(nn.Module):
    @property
    def num_features(self) -> int:
        """The number of features produced by this module."""
        raise NotImplementedError

    def forward(
        self,
        z: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        input_coord: str,
        prod_factor_dim: List[int],
        prod_factor_gaussian_curvature: List[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a state on the product manifold to float64 hyperbolic polar coordinates.

        With `m` product factors of dimensions `d_i` and `D = sum(d_i)`, `z` is, by `input_coord`:

            - `Coordinate.CARTESIAN`: `(..., D)`, the Poincare-ball points `z_i`, concatenated.
            - `Coordinate.HYPERBOLIC_CARTESIAN`: `(..., D)`, the vectors `rho_i * u_i`, concatenated.
            - `Coordinate.POINCARE_DISK_POLAR`: `(rs, us)`, the Poincare-ball radii `(..., m)` and
              the unit directions `(..., D)`.
            - `Coordinate.HYPERBOLIC_POLAR`: `(rhos, us)`, the intrinsic radii `(..., m)` and the
              unit directions `(..., D)`.

        Args:
            z (`torch.Tensor` or `Tuple[torch.Tensor, torch.Tensor]`): the state, see above.
            input_coord (`str`): the `Coordinate` tag of `z`.
            prod_factor_dim (`List[int]`): the dimension `d_i` of each factor.
            prod_factor_gaussian_curvature (`List[float]`): the curvature `K_i` of each factor.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`:
                - `rhos` of shape `(..., m)`, one intrinsic radius per factor.
                - `us` of shape `(..., D)`, the per-factor unit directions concatenated.
        """
        if input_coord == Coordinate.HYPERBOLIC_POLAR:
            rhos, thetas = z
            return rhos.to(torch.float64), thetas.to(torch.float64)
        elif input_coord == Coordinate.POINCARE_DISK_POLAR:
            rs, thetas = z
            return GeoUtils.poincare_polar_to_hyperbolic_polar_prod(
                rs=rs.to(torch.float64), thetas=thetas.to(torch.float64),
                prod_factor_dim=prod_factor_dim,
                prod_factor_gaussian_curvature=prod_factor_gaussian_curvature
            )
        elif input_coord == Coordinate.HYPERBOLIC_CARTESIAN:
            return GeoUtils.hyperbolic_cartesian_to_hyperbolic_polar_prod(
                z=z.to(torch.float64), prod_factor_dim=prod_factor_dim
            )
        elif input_coord == Coordinate.CARTESIAN:
            return GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(
                z=z.to(torch.float64), prod_factor_dim=prod_factor_dim,
                prod_factor_gaussian_curvature=prod_factor_gaussian_curvature
            )
        else:
            raise ValueError(f"Invalid coordinate type: {input_coord}")
