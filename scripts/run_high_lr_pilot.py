"""Map a high-learning-rate stability region for DAGC.

This is a diagnostic pilot, not the final tuned benchmark. It uses a fixed
global-clipping threshold of 1.0 and the default DAGC controller to identify
learning rates worth the later calibration/evaluation split.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import train_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.40, 0.80])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    out_dir = ROOT / "results" / "a1_high_lr_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_runs.csv"
    rows: list[dict] = []
    if raw_path.exists():
        with raw_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    completed = {(float(r["lr"]), r["policy"], int(r["seed"])) for r in rows}

    policies = {
        "none": {"name": "none"},
        "fixed": {"name": "fixed", "threshold": 1.0},
        "dagc": {"name": "dagc", "gamma": 0.05, "beta": 0.9, "relax": 0.3, "osc_weight": 3.0, "init_c": 1.0},
    }
    total = len(args.lrs) * len(policies) * len(args.seeds)
    for lr in args.lrs:
        for policy, clipping in policies.items():
            for seed in args.seeds:
                if (lr, policy, seed) in completed:
                    continue
                cfg = {
                    "dataset": "fashion_mnist", "model": "small_cnn", "data_dir": str(ROOT / "data"),
                    "epochs": args.epochs, "batch_size": 256, "lr": lr, "momentum": 0.9,
                    "weight_decay": 1e-4, "optimizer": "sgd", "seed": seed, "device": "cuda",
                    "num_workers": 0, "train_size": args.train_size, "test_size": args.test_size,
                    "projection_dim": 8, "clipping": clipping,
                }
                try:
                    result = train_run(cfg)
                    row = {"lr": lr, "policy": policy, "seed": seed, "status": "complete", "test_acc": result["test_acc"], "mean_loss": result["mean_loss"], "f_clip": result["f_clip"], "i_clip": result["i_clip"], "n_switch": result["n_switch"]}
                except Exception as exc:
                    row = {"lr": lr, "policy": policy, "seed": seed, "status": "failed", "error": type(exc).__name__}
                rows.append(row)
                fields = sorted({key for item in rows for key in item})
                with raw_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"[{len(rows)}/{total}] lr={lr:g} {policy} seed={seed}: {row['status']} {row.get('test_acc', '')}", flush=True)


if __name__ == "__main__":
    main()
