"""Small models for diagnostic-scale experiments."""

from __future__ import annotations

import torch.nn as nn


class SmallCNN(nn.Module):
    """Small CNN for 28x28 grayscale inputs (FashionMNIST/MNIST)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MLP(nn.Module):
    """Small MLP for vector inputs."""

    def __init__(self, in_dim: int = 784, hidden: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN(num_classes=num_classes)
    if name == "mlp":
        return MLP(num_classes=num_classes)
    raise ValueError(f"unknown model: {name}")
