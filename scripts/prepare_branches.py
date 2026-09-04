"""Create paired causal-branch configs from a completed reference step log."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.branching import prepare_causal_branches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-steps", required=True, type=Path)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "configs/experiments/p2_causal_reference.yaml",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/branches")
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=50)
    args = parser.parse_args()
    manifest = prepare_causal_branches(
        reference_steps=args.reference_steps,
        base_config=args.base_config,
        output_dir=args.output,
        shuffle_seed=args.shuffle_seed,
        block_size=args.block_size,
    )
    print(f"branch manifest: {manifest}")
    print("configuration generation only; no branch was trained")


if __name__ == "__main__":
    main()
