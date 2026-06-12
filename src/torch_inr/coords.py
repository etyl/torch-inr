from collections.abc import Sequence
from math import prod

import torch
from torch import Tensor


def get_coords(
    shape: Sequence[int], predict_dims: Sequence[int] | None = None
) -> Tensor:
    """Build a cartesian grid of coordinates normalized to [-1, 1].

    Dimensions listed in ``predict_dims`` are excluded from the grid (they are
    predicted by the network instead of being part of its input).
    """
    if predict_dims is None:
        predict_dims = []
    input_shape = [shape[k] for k in range(len(shape)) if k not in predict_dims]
    coords = torch.cartesian_prod(
        *[2 * torch.arange(s).to(torch.float32) / s - 1 for s in input_shape]
    )
    if coords.dim() == 1:
        coords = coords.unsqueeze(1)
    return coords


def get_input_shape(shape: Sequence[int], predict_dims: Sequence[int]) -> list[int]:
    return [shape[k] for k in range(len(shape)) if k not in predict_dims]


def get_output_dim(shape: Sequence[int], predict_dims: Sequence[int]) -> int:
    return prod([shape[k] for k in predict_dims]) if len(predict_dims) > 0 else 1


def get_target(X: Tensor, predict_dims: Sequence[int]) -> Tensor:
    """Reorder and flatten ``X`` so it matches the coordinate order of ``get_coords``."""
    shape = X.shape
    input_dims = [k for k in range(len(shape)) if k not in predict_dims]
    X_target = X.permute(*(input_dims + list(predict_dims)))
    output_dim = get_output_dim(shape, predict_dims)
    X_target = X_target.reshape(*([-1] + [output_dim])).to(X)
    return X_target
