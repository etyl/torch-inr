"""End-to-end convergence tests: each layer, encoder and sampler must fit a
simple smooth 2D signal to low reconstruction error."""

import math

import pytest
import torch
from torch import nn

from torch_inr import (
    FinerLayer,
    FourierEncoding,
    FourierGaussEncoding,
    GDSampler,
    LMCSampler,
    NoiseEncoding,
    SineLayer,
    UniformSampler,
    get_coords,
)

GRID = 16


def make_target(n: int = GRID, freq: float = 1.0) -> torch.Tensor:
    """Smooth low-frequency 2D signal in [-1, 1]."""
    coords = get_coords((n, n))
    vals = torch.sin(freq * math.pi * coords[:, 0]) * torch.cos(freq * math.pi * coords[:, 1])
    return vals.reshape(n, n)


def fit(model: nn.Module, sampler, n_steps: int = 200, lr: float = 1e-3) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_steps):
        coords = sampler.sample()
        loss = sampler.compute_loss(model(coords))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def full_grid_mse(model: nn.Module, X: torch.Tensor) -> float:
    eval_sampler = GDSampler(X)
    with torch.no_grad():
        pred = model(eval_sampler.sample())
    return eval_sampler.compute_loss(pred).item()


def make_siren(omega: float = 30.0) -> nn.Module:
    return nn.Sequential(
        SineLayer(2, 32, is_first=True, omega=omega),
        SineLayer(32, 32, omega=omega),
        nn.Linear(32, 1),
    )


def make_finer(omega: float = 30.0) -> nn.Module:
    return nn.Sequential(
        FinerLayer(2, 32, is_first=True, omega=omega),
        FinerLayer(32, 32, omega=omega),
        nn.Linear(32, 1),
    )


def make_relu_mlp(encoder, hidden: int = 64) -> nn.Module:
    return nn.Sequential(
        encoder,
        nn.Linear(encoder.output_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


# The target has unit-ish variance, so an untrained model sits around MSE ~0.5;
# 1e-3 means the INR genuinely fit the signal.
CONVERGED_MSE = 1e-3


# ---------------------------------------------------------------------------
# Layers: SIREN and FINER networks fit the signal with full-grid training.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_model", [make_siren, make_finer], ids=["siren", "finer"])
def test_layer_network_converges(make_model):
    torch.manual_seed(0)
    X = make_target()
    model = make_model()
    fit(model, GDSampler(X))
    assert full_grid_mse(model, X) < CONVERGED_MSE


# ---------------------------------------------------------------------------
# Encoders: encoder + ReLU MLP fits the signal.
# ---------------------------------------------------------------------------

def test_fourier_encoding_converges():
    torch.manual_seed(0)
    # FourierEncoding only contains period-1 features (sin/cos of 2*pi*x), so
    # the target must be expressible at that frequency.
    X = make_target(freq=2.0)
    model = make_relu_mlp(FourierEncoding(input_dim=2))
    fit(model, GDSampler(X), lr=5e-3)
    assert full_grid_mse(model, X) < CONVERGED_MSE


def test_fourier_gauss_encoding_converges():
    torch.manual_seed(0)
    X = make_target()
    model = make_relu_mlp(FourierGaussEncoding(input_dim=2, num_frequencies=64, sigma=3.0))
    fit(model, GDSampler(X), lr=5e-3)
    assert full_grid_mse(model, X) < CONVERGED_MSE


def test_noise_encoding_converges(tmp_path):
    torch.manual_seed(0)
    X = make_target()
    pretrain_sampler = UniformSampler(X, batch_size=64)
    encoder = NoiseEncoding(
        input_dim=2,
        output_dim=32,
        n_layers=2,
        sampler=pretrain_sampler,
        cache_dir=str(tmp_path),
        pretrain_kwargs={"n_steps": 50, "n_decoders": 2},
    )
    model = make_relu_mlp(encoder)
    fit(model, GDSampler(X), lr=5e-3)
    # encoder is frozen, only the head trains, so allow a looser threshold
    assert full_grid_mse(model, X) < 1e-2


def test_hashgrid_encoding_converges():
    pytest.importorskip("tinycudann")
    from torch_inr import HashGridEncoding

    torch.manual_seed(0)
    X = make_target().cuda()
    model = make_relu_mlp(HashGridEncoding(input_dim=2)).cuda()
    fit(model, GDSampler(X), lr=5e-3)
    assert full_grid_mse(model, X) < CONVERGED_MSE


# ---------------------------------------------------------------------------
# Samplers: each sampling strategy trains a SIREN to convergence.
# ---------------------------------------------------------------------------

def test_uniform_sampler_converges():
    torch.manual_seed(0)
    X = make_target()
    model = make_siren()
    fit(model, UniformSampler(X, batch_size=128), n_steps=1000)
    assert full_grid_mse(model, X) < CONVERGED_MSE


# LMCSampler trains on continuous coordinates but floors them to grid cells
# for the target lookup, so on a 16x16 grid the reconstruction carries an
# irreducible half-cell discretization bias of ~0.02 MSE for this signal.
LMC_CONVERGED_MSE = 5e-2


def test_lmc_sampler_converges():
    torch.manual_seed(0)
    X = make_target()
    model = make_siren()
    fit(model, LMCSampler(X, batch_size=128), n_steps=1000)
    assert full_grid_mse(model, X) < LMC_CONVERGED_MSE


def test_lmc_soft_mining_converges():
    torch.manual_seed(0)
    X = make_target()
    model = make_siren()
    fit(model, LMCSampler(X, batch_size=128, soft_mining=True, alpha=1.0), n_steps=1000)
    assert full_grid_mse(model, X) < LMC_CONVERGED_MSE


def test_gd_sampler_loss_decreases_monotonically_overall():
    torch.manual_seed(0)
    X = make_target()
    model = make_siren()
    sampler = GDSampler(X)
    initial = full_grid_mse(model, X)
    fit(model, sampler, n_steps=200)
    final = full_grid_mse(model, X)
    assert final < initial / 100


# ---------------------------------------------------------------------------
# Multi-channel target via predict_dims (e.g. RGB image).
# ---------------------------------------------------------------------------

def test_multichannel_convergence_predict_dims():
    torch.manual_seed(0)
    n, channels = GRID, 3
    coords = get_coords((n, n))
    X = torch.stack(
        [torch.sin(math.pi * (c + 1) * coords[:, 0] / 2) * torch.cos(math.pi * coords[:, 1]) for c in range(channels)],
        dim=-1,
    ).reshape(n, n, channels)

    model = nn.Sequential(
        SineLayer(2, 32, is_first=True, omega=30.0),
        SineLayer(32, 32, omega=30.0),
        nn.Linear(32, channels),
    )
    sampler = GDSampler(X, predict_dims=(2,))
    fit(model, sampler)

    with torch.no_grad():
        mse = sampler.compute_loss(model(sampler.sample())).item()
    assert mse < CONVERGED_MSE
