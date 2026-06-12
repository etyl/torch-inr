import pytest
import torch

from torch_inr import (
    FourierEncoding,
    FourierGaussEncoding,
    NoiseEncoding,
)


def test_fourier_encoding():
    enc = FourierEncoding(input_dim=3)
    x = torch.rand(8, 3) * 2 - 1
    out = enc(x)
    assert out.shape == (8, enc.output_dim)
    assert enc.output_dim == 6
    assert out.abs().max() <= 1.0


def test_fourier_gauss_encoding():
    enc = FourierGaussEncoding(input_dim=2, num_frequencies=32, sigma=10.0)
    x = torch.rand(8, 2) * 2 - 1
    out = enc(x)
    assert out.shape == (8, enc.output_dim)
    assert enc.output_dim == 2 * 32
    assert out.abs().max() <= 1.0


def test_fourier_gauss_projection_is_buffer():
    enc = FourierGaussEncoding(input_dim=2, num_frequencies=4, sigma=1.0)
    assert "B" in dict(enc.named_buffers())
    assert "B" in enc.state_dict()


def test_hashgrid_requires_tinycudann():
    pytest.importorskip("tinycudann")
    from torch_inr import HashGridEncoding

    enc = HashGridEncoding(input_dim=2)
    assert enc.output_dim == 32


def _make_sampler():
    from torch_inr import UniformSampler

    return UniformSampler(torch.rand(8, 8), batch_size=32)


def test_noise_encoding_requires_sampler_or_weights():
    with pytest.raises(ValueError):
        NoiseEncoding(input_dim=2, output_dim=16, n_layers=2)


def test_noise_encoding_forward(tmp_path):
    enc = NoiseEncoding(
        input_dim=2,
        output_dim=16,
        n_layers=2,
        sampler=_make_sampler(),
        cache_dir=str(tmp_path),
        pretrain_kwargs={"n_steps": 5, "n_decoders": 2},
    )
    x = torch.rand(8, 2) * 2 - 1
    out = enc(x)
    assert out.shape == (8, 16)
    assert not out.requires_grad


def test_noise_encoding_pretrain_caches_weights(tmp_path):
    kwargs = dict(
        input_dim=2,
        output_dim=8,
        n_layers=2,
        cache_dir=str(tmp_path),
        pretrain_kwargs={"n_steps": 5, "n_decoders": 2},
    )
    enc = NoiseEncoding(sampler=_make_sampler(), **kwargs)
    cache_files = list(tmp_path.glob("noise_encoding_*.pth"))
    assert len(cache_files) == 1

    # Same hyperparameters: loads the cached weights instead of re-pretraining.
    enc2 = NoiseEncoding(sampler=_make_sampler(), **kwargs)
    x = torch.rand(4, 2) * 2 - 1
    assert torch.allclose(enc(x), enc2(x))

    # Loading from an explicit path gives the same weights too.
    enc3 = NoiseEncoding(
        input_dim=2, output_dim=8, n_layers=2, weights_path=str(cache_files[0])
    )
    assert torch.allclose(enc(x), enc3(x))
