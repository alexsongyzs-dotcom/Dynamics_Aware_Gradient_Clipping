"""Development-only search for DAGC-V2 exposure-target control.

Uses seeds 200--202, which are disjoint from every V1 calibration and
evaluation seed. The selected V2 configuration must still be evaluated later
on fresh held-out seeds.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import train_run


def main() -> None:
    out_dir = ROOT / "results" / "dagc_v2_development"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "raw_runs.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8"))) if path.exists() else []
    done = {(r["label"], int(r["seed"])) for r in rows}
    policies = {
        "fixed@0.25": {"name": "fixed", "threshold": 0.25},
        "agc@default": {"name": "agc", "clip_value": 0.01, "eps": 1e-3},
        "dagc_v1": {"name": "dagc", "init_c": 0.5, "gamma": 0.05, "beta": 0.9, "relax": 0.3, "osc_weight": 3.0},
    }
    for target in (0.10, 0.20, 0.35, 0.50, 0.70):
        for gamma in (0.05, 0.10):
            policies[f"dagc_v2@target={target:g},gamma={gamma:g}"] = {
                "name": "dagc", "init_c": 0.5, "gamma": gamma, "beta": 0.9,
                "relax": 0.3, "osc_weight": 3.0, "exposure_target": target,
            }
    seeds = (200, 201, 202)
    for label, clipping in policies.items():
        for seed in seeds:
            if (label, seed) in done:
                continue
            cfg = {
                "dataset": "fashion_mnist", "model": "small_cnn", "data_dir": str(ROOT / "data"),
                "epochs": 15, "batch_size": 256, "lr": 0.8, "momentum": 0.9,
                "weight_decay": 1e-4, "optimizer": "sgd", "seed": seed, "device": "cuda",
                "num_workers": 0, "train_size": 5000, "test_size": 2000,
                "projection_dim": 8, "divergence_loss": 100.0, "divergence_jump": 50.0,
                "clipping": clipping,
            }
            result = train_run(cfg)
            rows.append({
                "label": label, "seed": seed, "test_acc": result["test_acc"],
                "diverged": int(result["diverged"]), "mean_loss": result["mean_loss"],
                "f_clip": result["f_clip"], "i_clip": result["i_clip"],
            })
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            print(f"{label} seed={seed}: {result['test_acc']:.4f}, diverged={int(result['diverged'])}", flush=True)


if __name__ == "__main__":
    main()
