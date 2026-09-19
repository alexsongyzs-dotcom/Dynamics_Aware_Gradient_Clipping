"""Held-out evaluation of the frozen DAGC-V2 configuration at lr=0.8."""

from __future__ import annotations

import csv
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import train_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.8)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(300, 308)))
    args = parser.parse_args()
    out_dir = ROOT / "results" / f"dagc_v2_evaluation_lr{args.lr:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "raw_runs.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8"))) if path.exists() else []
    done = {(r["policy"], int(r["seed"])) for r in rows}
    policies = {
        "none": {"name": "none"},
        "fixed_tuned": {"name": "fixed", "threshold": 0.25},
        "agc_default": {"name": "agc", "clip_value": 0.01, "eps": 1e-3},
        "dagc_v1": {"name": "dagc", "init_c": 0.5, "gamma": 0.05, "beta": 0.9, "relax": 0.3, "osc_weight": 3.0},
        "dagc_v2": {"name": "dagc", "init_c": 0.5, "gamma": 0.05, "beta": 0.9, "relax": 0.3, "osc_weight": 3.0, "exposure_target": 0.1},
    }
    for policy, clipping in policies.items():
        for seed in args.seeds:
            if (policy, seed) in done:
                continue
            cfg = {
                "dataset": "fashion_mnist", "model": "small_cnn", "data_dir": str(ROOT / "data"),
                "epochs": 15, "batch_size": 256, "lr": args.lr, "momentum": 0.9,
                "weight_decay": 1e-4, "optimizer": "sgd", "seed": seed, "device": "cuda",
                "num_workers": 0, "train_size": 5000, "test_size": 2000, "projection_dim": 8,
                "divergence_loss": 100.0, "divergence_jump": 50.0, "clipping": clipping,
            }
            result = train_run(cfg)
            rows.append({
                "policy": policy, "seed": seed, "test_acc": result["test_acc"],
                "diverged": int(result["diverged"]), "mean_loss": result["mean_loss"],
                "f_clip": result["f_clip"], "i_clip": result["i_clip"],
            })
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            print(f"{policy} seed={seed}: {result['test_acc']:.4f}, diverged={int(result['diverged'])}", flush=True)


if __name__ == "__main__":
    main()
