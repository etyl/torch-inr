from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import prod
import os
from typing import Optional

import torch
from torch import Tensor, nn
from tqdm import tqdm

from .layers import FinerLayer
from .samplers import UniformSampler, Sampler


class PositionalEncoder(nn.Module, ABC):
    """Maps input coordinates to a higher-dimensional embedding."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass


class FourierEncoding(PositionalEncoder):
    """Concatenated ``sin`` and ``cos`` of ``2 * pi * x``."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    @property
    def output_dim(self) -> int:
        return 2 * self.input_dim

    def forward(self, x: Tensor) -> Tensor:
        x_proj = 2.0 * torch.pi * x
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierGaussEncoding(PositionalEncoder):
    """Random Fourier features with a Gaussian projection matrix (Tancik et al., 2020)."""

    def __init__(self, input_dim: int, num_frequencies: int, sigma: float):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        B = sigma * torch.randn(num_frequencies, input_dim)
        self.register_buffer("B", B.T)

    @property
    def output_dim(self) -> int:
        return 2 * self.num_frequencies

    def forward(self, x: Tensor) -> Tensor:
        x_proj = (2.0 * torch.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class HashGridEncoding(PositionalEncoder):
    """Multi-resolution hash grid encoding backed by tinycudann (requires CUDA)."""

    def __init__(
        self,
        input_dim: int,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 21,
        base_resolution: int = 32,
        per_level_scale: float = 2.0,
    ):
        super().__init__()
        try:
            import tinycudann as tcnn
        except ImportError as e:
            raise ImportError(
                "HashGridEncoding requires tinycudann. "
                "Install it with `pip install torch-inr[hashgrid]` or "
                "`pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`."
            ) from e
        self.input_dim = input_dim
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.encoder = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
                "fixed_point_pos": False,
            },
        )

    @property
    def output_dim(self) -> int:
        return self.n_levels * self.n_features_per_level

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class NoiseEncoding(PositionalEncoder):
    """Learnable encoding built from stacked FINER layers.

    Pretraining is mandatory: pass either a ``sampler`` (pretrains eagerly on
    construction, matching the sampler's data shape, ``predict_dims`` and
    device) or a ``weights_path`` to load saved weights. Pretrained weights
    are cached in ``cache_dir`` under a key derived from the hyperparameters
    (layer type, width, depth, input dim); constructing again with the same
    hyperparameters loads the cached weights instead of re-pretraining.
    The encoding is frozen at inference.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 3,
        sampler: Optional[Sampler] = None,
        backbone = None,
        weights_path: Optional[str] = None,
        cache_dir: str = "weights",
        pretrain_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self._output_dim = output_dim

        if backbone is not None:
            self.backbone = backbone
        else:
            layers = [FinerLayer(input_dim, output_dim, is_first=True)]
            for _ in range(n_layers - 1):
                layers.append(FinerLayer(output_dim, output_dim))
            self.backbone = nn.Sequential(*layers)

        if weights_path is not None:
            self._load(weights_path)
        elif sampler is not None:
            cache_path = os.path.join(cache_dir, f"noise_encoding_{self._cache_key()}.pth")
            if os.path.exists(cache_path):
                self._load(cache_path)
                self.backbone.to(sampler.X.device)
            else:
                self._pretrain(
                    sampler.X.shape,
                    batch_size=getattr(sampler, "batch_size", 1024),
                    predict_dims=getattr(sampler, "predict_dims", ()),
                    device=sampler.X.device,
                    save_path=cache_path,
                    **(pretrain_kwargs or {}),
                )
        else:
            raise ValueError("Either `sampler` or `weights_path` must be provided for NoiseEncoding.")

    def _cache_key(self) -> str:
        """Cache key from the backbone hyperparameters: layer type, width, depth, input dim."""
        layers = list(self.backbone.children()) or [self.backbone]
        layer_type = type(layers[0]).__name__
        return f"{layer_type}_in{self.input_dim}_w{self._output_dim}_d{len(layers)}"

    def _load(self, path: str) -> None:
        self.backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.backbone.eval()

    def _pretrain(
        self,
        data_shape: Sequence[int],
        batch_size: int,
        n_steps: int = 1000,
        n_decoders: int = 10,
        lr: float = 1e-3,
        predict_dims: Sequence[int] = (),
        device: Optional[torch.device] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Fit the encoding on random targets through ``n_decoders`` linear heads."""
        predict_dims = list(predict_dims)
        if device is not None:
            self.backbone.to(device)
        device = next(self.backbone.parameters()).device

        out_shape = [data_shape[k] for k in predict_dims] or [1]
        targets = 2 * torch.rand((n_decoders, *data_shape), device=device) - 1
        samplers = [
            UniformSampler(t, batch_size=max(batch_size // n_decoders, 1), predict_dims=predict_dims)
            for t in targets
        ]
        decoders = [nn.Linear(self._output_dim, prod(out_shape)).to(device) for _ in range(n_decoders)]
        params = list(self.backbone.parameters()) + [p for d in decoders for p in d.parameters()]
        optimizer = torch.optim.Adam(params, lr=lr)
        for _ in tqdm(range(n_steps), desc="Pretraining NoiseEncoding", unit="step"):
            loss = torch.zeros((), device=device)
            for sampler, decoder in zip(samplers, decoders):
                coords = sampler.sample()
                X_proj = self.backbone(coords)
                pred = decoder(X_proj).reshape(coords.shape[0], *out_shape)
                loss = loss + sampler.compute_loss(pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if save_path is not None:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.backbone.state_dict(), save_path)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.backbone(x)
