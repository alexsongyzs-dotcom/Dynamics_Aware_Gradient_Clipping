"""Run a hyperparameter sweep from a Hydra sweep config.

Usage:
    python scripts/run_sweep.py --config configs/sweeps/phase_diagram.yaml
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    grid = cfg["sweep"]["grid"]
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"Sweep {cfg['sweep']['name']}: {len(combos)} configurations x {len(cfg['sweep']['seeds'])} seeds")

    for values in combos:
        overrides = [f"{k}={v}" for k, v in zip(keys, values)]
        for seed in cfg["sweep"]["seeds"]:
            cmd = [sys.executable, "-m", "src.train", f"seed={seed}"] + overrides
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
