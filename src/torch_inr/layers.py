import math

import torch
from torch import Tensor, nn
from torch.nn import init


def _init_weights(
    linear: nn.Linear, omega: float = 1.0, is_first: bool = False
) -> None:
    if hasattr(linear, "weight"):
        features_in = linear.weight.size(-1)
        if is_first:
            bound = 1 / features_in
        else:
            bound = math.sqrt(1 / features_in) / omega
        init.uniform_(linear.weight, -bound, bound)


def _init_bias(
    linear: nn.Linear, fbs: float | None = None, is_first: bool = True
) -> None:
    if (
        is_first
        and fbs is not None
        and hasattr(linear, "bias")
        and linear.bias is not None
    ):
        init.uniform_(linear.bias, -fbs, fbs)


class SineLayer(nn.Module):
    """SIREN layer: ``sin(omega * (Wx + b))`` with frequency-aware init."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega: float = 1.0,
    ):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        _init_weights(self.linear, omega, is_first=is_first)

    def forward(self, input: Tensor) -> Tensor:
        return torch.sin(self.omega * self.linear(input))


def _finer_activation(x: Tensor, omega: float) -> Tensor:
    """FINER activation ``sin(omega * (|x| + 1) * x)``.

    The magnitude term is detached so gradients only flow through the sine argument.
    """
    with torch.no_grad():
        alpha = torch.abs(x) + 1
    return torch.sin(omega * alpha * x)


class FinerLayer(nn.Module):
    """FINER layer (variable-periodic sine) with first-layer bias init in [-fbs, fbs]."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega: float = 1.0,
        fbs: float = 10.0,
    ):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        _init_weights(self.linear, omega, is_first)
        _init_bias(self.linear, fbs, is_first)

    def forward(self, x: Tensor) -> Tensor:
        wx_b = self.linear(x)
        return _finer_activation(wx_b, self.omega)
