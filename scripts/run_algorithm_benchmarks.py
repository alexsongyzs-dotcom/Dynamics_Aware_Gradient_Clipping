"""Run reproducible A3 ablations and A4 multi-seed diagnostic benchmarks.

Examples:
    python scripts/run_algorithm_benchmarks.py --suite a3
    python scripts/run_algorithm_benchmarks.py --suite a4
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.train import train_run
METRICS = ("test_acc", "mean_loss", "f_clip", "i_clip", "n_switch", "mean_c1", "mean_c2")


def base_config(seed: int, epochs: int, train_size: int, test_size: int) -> dict:
    return {
        "dataset": "fashion_mnist",
        "model": "small_cnn",
        "data_dir": str(ROOT / "data"),
        "epochs": epochs,
        "batch_size": 256,
        "lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "optimizer": "sgd",
        "seed": seed,
        "device": "cuda",
        "num_workers": 0,
        "train_size": train_size,
        "test_size": test_size,
        "projection_dim": 8,
    }


def a3_conditions() -> list[tuple[str, dict]]:
    return [
        ("full_dagc", {"name": "dagc", "gamma": 0.05, "beta": 0.9, "relax": 0.3, "osc_weight": 3.0, "init_c": 1.0}),
        ("no_exposure_feedback", {"name": "dagc", "gamma": 0.05, "beta": 0.9, "relax": 0.0, "osc_weight": 3.0, "init_c": 1.0}),
        ("no_alignment_feedback", {"name": "dagc", "gamma": 0.05, "beta": 0.9, "relax": 0.3, "osc_weight": 0.0, "init_c": 1.0}),
        ("fixed_threshold", {"name": "dagc", "gamma": 0.0, "beta": 0.9, "relax": 0.3, "osc_weight": 3.0, "init_c": 1.0}),
        ("fixed_global", {"name": "fixed", "threshold": 1.0}),
    ]


def a4_conditions() -> list[tuple[str, str, str, dict, float]]:
    policies = [
        ("none", {"name": "none"}),
        ("fixed", {"name": "fixed", "threshold": 1.0}),
        ("agc", {"name": "agc", "clip_value": 0.01, "eps": 1e-3}),
        ("dagc", {"name": "dagc", "gamma": 0.05, "beta": 0.9, "relax": 0.3, "osc_weight": 3.0, "init_c": 1.0}),
    ]
    rows = []
    for dataset, model, optimizer, (policy, clipping) in itertools.product(
        ("mnist", "fashion_mnist"), ("mlp", "small_cnn"), ("sgd", "adamw"), policies
    ):
        lr = 0.001 if optimizer == "adamw" else 0.05
        rows.append((dataset, model, optimizer, clipping | {"label": policy}, lr))
    return rows


def write_summary(rows: list[dict], out_dir: Path) -> None:
    groups: dict[tuple, list[dict]] = {}
    keys = ("suite", "condition", "dataset", "model", "optimizer", "policy")
    for row in rows:
        groups.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    summary = []
    for group_key, values in groups.items():
        item = dict(zip(keys, group_key))
        item["n"] = len(values)
        for metric in METRICS:
            xs = np.array([float(v[metric]) for v in values])
            item[f"{metric}_mean"] = float(xs.mean())
            item[f"{metric}_ci95"] = float(1.96 * xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0
        summary.append(item)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("a3", "a4"), required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    out_dir = ROOT / "results" / ("a3_ablation_small" if args.suite == "a3" else "a4_multiseed_small")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_runs.csv"
    rows: list[dict] = []
    if raw_path.exists():
        with raw_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Keep the most recent record when a prior smoke run used the same key.
        unique: dict[tuple, dict] = {}
        for row in rows:
            key = (row["condition"], row["dataset"], row["model"], row["optimizer"], row["policy"], row["seed"])
            unique[key] = row
        rows = list(unique.values())

    if args.suite == "a3":
        runs = [(label, "fashion_mnist", "small_cnn", "sgd", clip, 0.05, seed) for label, clip in a3_conditions() for seed in args.seeds]
    else:
        runs = [
            (clip["label"], dataset, model, optimizer, {k: v for k, v in clip.items() if k != "label"}, lr, seed)
            for dataset, model, optimizer, clip, lr in a4_conditions() for seed in args.seeds
        ]

    completed = {
        (row["condition"], row["dataset"], row["model"], row["optimizer"], row["policy"], int(row["seed"]))
        for row in rows
    }
    print(f"{args.suite}: {len(runs)} runs ({len(completed)} already complete)")
    for index, (label, dataset, model, optimizer, clipping, lr, seed) in enumerate(runs, start=1):
        cfg = base_config(seed, args.epochs, args.train_size, args.test_size)
        cfg.update({"dataset": dataset, "model": model, "optimizer": optimizer, "lr": lr, "clipping": clipping})
        key = (label if args.suite == "a3" else "", dataset, model, optimizer, clipping["name"], seed)
        if key in completed:
            continue
        result = train_run(cfg)
        row = {"suite": args.suite, "condition": label if args.suite == "a3" else "", "dataset": dataset, "model": model, "optimizer": optimizer, "policy": clipping["name"], "seed": seed}
        row.update({metric: result[metric] for metric in METRICS})
        rows.append(row)
        with raw_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[{index}/{len(runs)}] {dataset}/{model}/{optimizer}/{label}/seed{seed}: {result['test_acc']:.4f}", flush=True)

    write_summary(rows, out_dir)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
