"""Low-memory fixed coordinate sketches for per-step directional logging."""

from __future__ import annotations

import math
import random

import torch
from torch import Tensor


class GradientSketcher:
    """Sample fixed coordinates without materializing a dense projection matrix.

    The sketch is intended for inexpensive per-step diagnostics. Confirmatory
    direction measurements should use full gradients on a fixed probe batch.
    """

    def __init__(self, dimension: int, sketch_size: int, seed: int, device: torch.device) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if sketch_size < 1:
            raise ValueError("sketch_size must be positive")
        self.dimension = int(dimension)
        self.sketch_size = min(int(sketch_size), self.dimension)
        indices = random.Random(seed).sample(range(self.dimension), self.sketch_size)
        self.indices = torch.tensor(indices, dtype=torch.long, device=device)
        self.scale = math.sqrt(self.dimension / self.sketch_size)

    def __call__(self, flattened_gradient: Tensor) -> Tensor:
        if flattened_gradient.numel() != self.dimension:
            raise ValueError("gradient dimension changed during the run")
        return flattened_gradient[self.indices] * self.scale

    def state_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "sketch_size": self.sketch_size,
            "indices": self.indices.detach().cpu(),
            "scale": self.scale,
        }
