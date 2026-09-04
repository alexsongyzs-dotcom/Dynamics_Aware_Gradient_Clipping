"""Diagnostic and confirmatory vision models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tv_models


class SmallCNN(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class MLP(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], hidden: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        in_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class CifarVisionTransformer(nn.Module):
    """Small ViT with an explicit, inspectable training stack."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        input_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        patches_per_side = image_size // patch_size
        patch_count = patches_per_side**2
        self.patch_embed = nn.Conv2d(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, patch_count + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([class_token, tokens], dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded[:, 0]))


def _cifar_resnet(name: str, input_channels: int, num_classes: int) -> nn.Module:
    constructors = {
        "resnet18": tv_models.resnet18,
        "resnet34": tv_models.resnet34,
        "resnet50": tv_models.resnet50,
    }
    model = constructors[name](weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(
    name: str,
    num_classes: int = 10,
    input_channels: int = 3,
    image_size: int = 32,
    **kwargs,
) -> nn.Module:
    """Build a model without downloading pretrained weights."""

    name = name.lower()
    if name == "small_cnn":
        return SmallCNN(input_channels=input_channels, num_classes=num_classes)
    if name == "mlp":
        return MLP(
            input_shape=(input_channels, image_size, image_size),
            hidden=int(kwargs.get("hidden", 256)),
            num_classes=num_classes,
        )
    if name in {"resnet18", "resnet34", "resnet50"}:
        return _cifar_resnet(name, input_channels, num_classes)
    if name in {"vit_tiny", "vit_small"}:
        defaults = (
            {"embed_dim": 192, "depth": 6, "heads": 3}
            if name == "vit_tiny"
            else {"embed_dim": 384, "depth": 8, "heads": 6}
        )
        defaults.update(kwargs)
        return CifarVisionTransformer(
            image_size=image_size,
            input_channels=input_channels,
            num_classes=num_classes,
            **defaults,
        )
    raise ValueError(f"unknown model: {name}")
