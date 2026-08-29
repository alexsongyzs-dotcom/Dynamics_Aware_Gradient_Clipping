"""Dataset loaders for quick experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_loaders(
    dataset: str = "fashion_mnist",
    data_dir: str = "data",
    batch_size: int = 256,
    num_workers: int = 2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader)."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    if dataset == "fashion_mnist":
        tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.286,), (0.353,))])
        train_set = datasets.FashionMNIST(root, train=True, transform=tr, download=download)
        test_set = datasets.FashionMNIST(root, train=False, transform=tr, download=download)
    elif dataset == "mnist":
        tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(root, train=True, transform=tr, download=download)
        test_set = datasets.MNIST(root, train=False, transform=tr, download=download)
    elif dataset == "cifar10":
        tr = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        te = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
        )
        train_set = datasets.CIFAR10(root, train=True, transform=tr, download=download)
        test_set = datasets.CIFAR10(root, train=False, transform=te, download=download)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
