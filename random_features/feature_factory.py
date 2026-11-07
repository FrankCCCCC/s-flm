from dataclasses import dataclass
from typing import List

from .feature_base import FeatureBase

@dataclass
class RandomFeatureType:
    """Random feature type for the DiT."""
    SPHERE: str = 'sphere'

class RandomFeatureFactory:
    """Factory for the random-feature input maps of the DiT."""
    @staticmethod
    def create_random_features(
        feature_type: str,
        degree: int = None,
        radial_scales: List[float] = None,
        num_factors: int = None,
        harmonic_mc_samples: int = None,
        harmonic_radial_type: str = 'exp_decay',
    ) -> FeatureBase:
        if feature_type == RandomFeatureType.SPHERE:
            from .hyperbolic_spherical_feat import SphericalFeature
            return SphericalFeature(
                degree=degree, radial_scales=radial_scales, num_factors=num_factors,
                mc_samples=harmonic_mc_samples, radial_type=harmonic_radial_type)
        else:
            raise ValueError(f'Unknown feature type {feature_type}.')