"""Download official datasets once, verify their torchvision checksums, record sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchvision import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["fashion_mnist", "cifar10", "cifar100"])
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    constructors = dict(fashion_mnist=datasets.FashionMNIST, cifar10=datasets.CIFAR10,
                        cifar100=datasets.CIFAR100, mnist=datasets.MNIST)
    # Official FashionMNIST repository; torchvision still validates each file
    # against the original published MD5 rather than trusting a download alone.
    datasets.FashionMNIST.mirrors = [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
        *datasets.FashionMNIST.mirrors,
    ]
    manifest = {}
    for name in args.datasets:
        constructor = constructors[name]
        train = constructor(args.data_dir, train=True, download=True)
        test = constructor(args.data_dir, train=False, download=True)
        manifest[name] = dict(training_examples=len(train), test_examples=len(test),
                              classes=len(train.classes), checksum_validation="torchvision dataset integrity checks")
        print(name, manifest[name], flush=True)
    (args.data_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
