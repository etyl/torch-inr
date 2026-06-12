"""torch-inr: Implicit Neural Representation components for PyTorch."""

from .layers import FinerLayer, SineLayer
from .coords import get_coords, get_input_shape, get_target
from .encoders import (
    FourierEncoding,
    FourierGaussEncoding,
    HashGridEncoding,
    NoiseEncoding,
    PositionalEncoder,
)
from .samplers import GDSampler, LMCSampler, Sampler, UniformSampler

__version__ = "0.1.0"

__all__ = [
    "FinerLayer",
    "SineLayer",
    "PositionalEncoder",
    "FourierEncoding",
    "FourierGaussEncoding",
    "HashGridEncoding",
    "NoiseEncoding",
    "Sampler",
    "UniformSampler",
    "LMCSampler",
    "GDSampler",
    "get_coords",
    "get_input_shape",
    "get_target",
    "__version__",
]
