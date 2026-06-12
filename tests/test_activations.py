import math

import torch

from torch_inr import FinerLayer, SineLayer
from torch_inr.layers import _finer_activation


def test_sine_layer_forward_shape():
    layer = SineLayer(2, 16, is_first=True, omega=30.0)
    out = layer(torch.rand(8, 2))
    assert out.shape == (8, 16)
    assert out.abs().max() <= 1.0


def test_sine_layer_init_bounds():
    in_features, omega = 64, 30.0
    first = SineLayer(in_features, 16, is_first=True, omega=omega)
    assert first.linear.weight.abs().max() <= 1 / in_features

    hidden = SineLayer(in_features, 16, is_first=False, omega=omega)
    assert hidden.linear.weight.abs().max() <= math.sqrt(1 / in_features) / omega


def test_finer_layer_forward_shape():
    layer = FinerLayer(2, 16, is_first=True)
    out = layer(torch.rand(8, 2))
    assert out.shape == (8, 16)
    assert out.abs().max() <= 1.0


def test_finer_first_layer_bias_init():
    fbs = 5.0
    layer = FinerLayer(2, 256, is_first=True, fbs=fbs)
    bias = layer.linear.bias
    assert bias.abs().max() <= fbs
    # with fbs >> 1 some biases should land outside the default linear init range
    assert bias.abs().max() > 1.0


def test_finer_activation_matches_formula():
    x = torch.linspace(-2, 2, 11)
    omega = 3.0
    expected = torch.sin(omega * (x.abs() + 1) * x)
    assert torch.allclose(_finer_activation(x, omega), expected)


def test_finer_activation_gradient_flows():
    x = torch.randn(5, requires_grad=True)
    _finer_activation(x, 1.0).sum().backward()
    assert x.grad is not None
