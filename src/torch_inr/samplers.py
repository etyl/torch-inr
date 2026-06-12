from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import prod, sqrt

import torch
from torch import Tensor

from .coords import get_coords, get_input_shape, get_target


class Sampler(ABC):
    """
    Common interface:
    ``sample()`` returns a coordinate batch in range [-1, 1],
    ``compute_loss(pred)`` returns the training loss for the last batch.
    """

    @abstractmethod
    def sample(self) -> Tensor:
        pass

    @abstractmethod
    def compute_loss(self, X_pred: Tensor) -> Tensor:
        pass


class UniformSampler(Sampler):
    """Uniform random sampling of grid coordinates."""

    def __init__(self, X: Tensor, batch_size: int, predict_dims: Sequence[int] = ()):
        self.X = X
        self.shape = X.shape
        self.predict_dims = list(predict_dims)
        self.input_shape = get_input_shape(self.shape, self.predict_dims)
        self.batch_size = batch_size

        self.device = X.device
        self.X_target = get_target(X, self.predict_dims)

        self._input_shape_tensor = torch.tensor(self.input_shape, device=self.device, dtype=torch.float32).unsqueeze(0)
        self._multipliers = torch.tensor(
            [prod(self.input_shape[i + 1:]) for i in range(len(self.input_shape))],
            device=self.device,
            dtype=torch.long,
        )

    @torch.no_grad()
    def sample(self) -> Tensor:
        self.idx = torch.empty((self.batch_size, len(self.input_shape)), dtype=torch.long, device=self.device)
        for i, dim in enumerate(self.input_shape):
            self.idx[:, i] = torch.randint(0, dim, (self.batch_size,), device=self.device)

        self.coords = self.idx / self._input_shape_tensor
        self.coords = self.coords * 2 - 1
        return self.coords.to(self.X.dtype)

    def compute_loss(self, X_pred: Tensor) -> Tensor:
        flat_idx = (self.idx * self._multipliers).sum(dim=1)
        target = self.X_target[flat_idx]

        dist = (X_pred - target) ** 2
        return dist.mean()


class LMCSampler(Sampler):
    """Langevin Monte Carlo sampler: coordinates drift toward high-error regions
    via ``a * grad(log q) + sqrt(b) * noise``, wrapped back into [-1, 1]."""

    def __init__(
        self,
        X: Tensor,
        batch_size: int,
        a: float = 0.01,
        b: float = 0.01,
        alpha: float = 1.0,
        soft_mining: bool = False,
    ):
        self.X = X
        self.shape = X.shape
        self.batch_size = batch_size
        self.device = X.device

        self.a = a
        self.b = b
        self.alpha = alpha
        self.soft_mining = soft_mining

        self.grad_q = None
        self.coords = torch.rand((self.batch_size, len(self.shape)), device=self.device) * 2 - 1

    def sample(self) -> Tensor:
        if self.grad_q is None:
            self.coords = self.coords.detach().requires_grad_(True)
            return self.coords

        with torch.no_grad():
            noise = torch.randn_like(self.coords) * sqrt(self.b)
            self.coords = self.coords + self.a * self.grad_q + noise

            # take modulo to stay in the valid range
            for i in range(len(self.shape)):
                self.coords[:, i] = (self.coords[:, i] + 1) % 2 - 1

        self.coords = self.coords.detach().requires_grad_(True)
        return self.coords

    def compute_loss(self, X_pred: Tensor) -> Tensor:
        coords_int = ((self.coords + 1) * torch.tensor(self.shape, device=X_pred.device) / 2).floor().long()
        coords_int = coords_int.clamp_min(0)
        for i in range(len(self.shape)):
            coords_int[:, i] = coords_int[:, i].clamp_max(self.shape[i] - 1)
        target = self.X[tuple(coords_int.t())]
        target = target.unsqueeze(-1)
        diff = X_pred.float() - target.float()

        log_q = torch.log(diff.abs().clamp_min(1e-4))
        self.grad_q = torch.autograd.grad(log_q.sum(), self.coords, retain_graph=True)[0]

        dist = diff ** 2

        if self.soft_mining:
            with torch.no_grad():
                q = diff.abs() + 1e-7
                q_alpha = torch.pow(q, self.alpha).clamp_min(1e-6)
            dist = dist / q_alpha

        return dist.mean()


class GDSampler(Sampler):
    """Deterministic full-grid sampler: every coordinate, every iteration."""

    def __init__(self, X: Tensor, predict_dims: Sequence[int] = ()):
        self.X = X
        self.shape = X.shape
        self.predict_dims = list(predict_dims)

        self.device = X.device
        self.X_target = get_target(X, self.predict_dims).float()
        self.coords = get_coords(X.shape, self.predict_dims).to(X)

    @torch.no_grad()
    def sample(self) -> Tensor:
        return self.coords

    def compute_loss(self, X_pred: Tensor) -> Tensor:
        dist = (X_pred - self.X_target) ** 2
        return dist.mean()
