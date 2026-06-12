import torch
from torch import nn

from torch_inr import GDSampler, LMCSampler, UniformSampler, get_coords


def test_uniform_sampler():
    X = torch.rand(16, 16)
    sampler = UniformSampler(X, batch_size=32)
    coords = sampler.sample()
    assert coords.shape == (32, 2)
    assert coords.min() >= -1 and coords.max() <= 1

    loss = sampler.compute_loss(torch.zeros(32, 1))
    assert loss.dim() == 0 and loss >= 0


def test_uniform_sampler_predict_dims():
    X = torch.rand(16, 16, 3)
    sampler = UniformSampler(X, batch_size=32, predict_dims=(2,))
    coords = sampler.sample()
    assert coords.shape == (32, 2)
    loss = sampler.compute_loss(torch.zeros(32, 3))
    assert loss.dim() == 0


def test_lmc_sampler_updates_coords():
    X = torch.rand(16, 16)
    sampler = LMCSampler(X, batch_size=32)
    model = nn.Linear(2, 1)

    coords0 = sampler.sample()
    assert coords0.shape == (32, 2)
    loss = sampler.compute_loss(model(coords0))
    assert loss.dim() == 0
    loss.backward()

    coords1 = sampler.sample()
    assert not torch.allclose(coords0.detach(), coords1.detach())
    assert coords1.min() >= -1 and coords1.max() <= 1
    sampler.compute_loss(model(coords1)).backward()


def test_gd_sampler_covers_grid():
    X = torch.rand(8, 8)
    sampler = GDSampler(X)
    coords = sampler.sample()
    assert coords.shape == (64, 2)
    assert torch.equal(coords, get_coords(X.shape))

    loss = sampler.compute_loss(torch.zeros(64, 1))
    assert torch.isclose(loss, (X ** 2).mean())
