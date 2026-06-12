# torch-inr

Implicit Neural Representation (INR) components for PyTorch.

- **Encoders** (`torch_inr.encoders`) — positional encodings mapping coordinates to embeddings:
  `FourierEncoding`, `FourierGaussEncoding` (random Fourier features), `HashGridEncoding`
  (tinycudann multi-resolution hash grid), `NoiseEncoding` (learnable FINER-based encoding).
- **Activations** (`torch_inr.activations`) — sine-based INR layers with frequency-aware
  initialization: `SineLayer` (SIREN), `FinerLayer` (FINER).
- **Samplers** (`torch_inr.samplers`) — coordinate/target sampling strategies for training:
  `UniformSampler`, `LMCSampler` (Langevin Monte Carlo), `GDSampler` (full grid).

Each family has a string registry (`ENCODERS`, `ACTIVATIONS`, `SAMPLERS`) for
config-driven construction.

## Installation

```bash
pip install -e .
# with the tinycudann hash grid encoder (requires CUDA):
pip install -e .[hashgrid]
# development (tests):
pip install -e .[dev]
```

Requires Python >= 3.10 and PyTorch.

## Quickstart

Fit a small SIREN-style network to a 2D signal:

```python
import torch
from torch import nn
from torch_inr import FourierGaussEncoding, SineLayer, UniformSampler

X = torch.rand(64, 64)  # signal to represent

encoder = FourierGaussEncoding(input_dim=2, num_frequencies=32, sigma=10.0)
model = nn.Sequential(
    encoder,
    SineLayer(encoder.output_dim, 64, is_first=True, omega=30.0),
    SineLayer(64, 64, omega=30.0),
    nn.Linear(64, 1),
)

sampler = UniformSampler(X, batch_size=1024)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for step in range(500):
    coords = sampler.sample()          # (batch, 2) in [-1, 1]
    loss = sampler.compute_loss(model(coords))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Coordinates are always normalized to `[-1, 1]`. Use `torch_inr.get_coords(X.shape)`
to build the full coordinate grid for reconstruction:

```python
from torch_inr import get_coords

with torch.no_grad():
    X_rec = model(get_coords(X.shape)).view(X.shape)
```

For tensors where some dimensions should be predicted as output channels rather
than used as input coordinates (e.g. RGB images of shape `(H, W, 3)`), pass
`predict_dims`:

```python
sampler = UniformSampler(X_rgb, batch_size=1024, predict_dims=(2,))  # model outputs 3 channels
```

## Testing

```bash
pip install -e .[dev]
pytest
```
