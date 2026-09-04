"""Datasets and replayable sampling for paired causal branches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterator, Sized

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import datasets, transforms


class StatefulRandomSampler(Sampler[int]):
    """Random sampler whose permutation, cursor, and RNG state are checkpointed."""

    def __init__(self, data_source: Sized, seed: int) -> None:
        self.data_source = data_source
        self.generator = torch.Generator().manual_seed(int(seed))
        self.epoch = 0
        self.cursor = 0
        self.permutation = torch.randperm(len(data_source), generator=self.generator)

    def __iter__(self) -> Iterator[int]:
        if self.cursor >= len(self.permutation):
            self.epoch += 1
            self.cursor = 0
            self.permutation = torch.randperm(len(self.data_source), generator=self.generator)
        while self.cursor < len(self.permutation):
            index = int(self.permutation[self.cursor])
            self.cursor += 1
            yield index

    def __len__(self) -> int:
        return len(self.data_source) - self.cursor

    def state_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "permutation": self.permutation.clone(),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        permutation = state["permutation"]
        generator_state = state["generator_state"]
        if not isinstance(permutation, torch.Tensor) or not isinstance(generator_state, torch.Tensor):
            raise TypeError("invalid sampler state")
        if len(permutation) != len(self.data_source):
            raise ValueError("sampler state belongs to a different dataset")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self.permutation = permutation.clone()
        self.generator.set_state(generator_state)


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    probe_loader: DataLoader
    train_sampler: StatefulRandomSampler
    num_classes: int
    input_channels: int


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _datasets(name: str, root: Path, download: bool):
    name = name.lower()
    if name == "fashion_mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.286,), (0.353,))]
        )
        return (
            datasets.FashionMNIST(root, train=True, transform=transform, download=download),
            datasets.FashionMNIST(root, train=False, transform=transform, download=download),
            10,
            1,
        )
    if name == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return (
            datasets.MNIST(root, train=True, transform=transform, download=download),
            datasets.MNIST(root, train=False, transform=transform, download=download),
            10,
            1,
        )

    statistics = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), datasets.CIFAR10, 10),
        "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), datasets.CIFAR100, 100),
    }
    if name not in statistics:
        raise ValueError(f"unknown dataset: {name}")
    mean, standard_deviation, dataset_class, classes = statistics[name]
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )
    evaluation_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, standard_deviation)]
    )
    train_set = dataset_class(root, train=True, transform=train_transform, download=download)
    test_set = dataset_class(root, train=False, transform=evaluation_transform, download=download)
    probe_set = dataset_class(root, train=True, transform=evaluation_transform, download=download)
    return train_set, test_set, classes, 3, probe_set


def build_data_bundle(
    dataset: str = "cifar10",
    data_dir: str = "data",
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 0,
    probe_size: int = 1024,
    download: bool = False,
    pin_memory: bool = True,
) -> DataBundle:
    """Create training, evaluation, and fixed-probe loaders.

    Confirmatory paired branches should use ``num_workers=0`` so restoring the
    process RNG also restores stochastic data augmentation exactly.
    """

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    built = _datasets(dataset, root, download)
    if len(built) == 4:
        train_set, test_set, num_classes, input_channels = built
        probe_set = test_set
    else:
        train_set, test_set, num_classes, input_channels, probe_set = built

    sampler = StatefulRandomSampler(train_set, seed)
    loader_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_seed,
        generator=loader_generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_seed,
        generator=torch.Generator().manual_seed(seed + 2),
    )
    probe_generator = torch.Generator().manual_seed(seed + 3)
    count = min(int(probe_size), len(probe_set))
    indices = torch.randperm(len(probe_set), generator=probe_generator)[:count].tolist()
    probe_loader = DataLoader(
        Subset(probe_set, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return DataBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        probe_loader=probe_loader,
        train_sampler=sampler,
        num_classes=int(num_classes),
        input_channels=int(input_channels),
    )


def build_loaders(
    dataset: str = "cifar10",
    data_dir: str = "data",
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Backward-compatible two-loader interface."""

    bundle = build_data_bundle(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
    )
    return bundle.train_loader, bundle.test_loader
