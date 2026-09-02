"""Prepare a small measurement smoke-test matrix; execution is opt-in."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.configuration import deep_merge, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "configs/experiments/p0_measurement.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/plans/quick")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    base = load_config(args.base)
    policies = {
        "none": {"name": "none", "placement": "pre_moment"},
        "fixed": {"name": "fixed", "threshold": 1.0, "placement": "pre_moment"},
        "autoclip": {
            "name": "autoclip",
            "percentile": 10.0,
            "window": 200,
            "warmup": 20,
            "placement": "pre_moment",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    for name, clipping in policies.items():
        config = deep_merge(
            base,
            {
                "run_id": f"p0-smoke-{name}",
                "output_dir": str(args.output / "runs"),
                "clipping": clipping,
            },
        )
        path = args.output / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        commands.append([sys.executable, "-m", "src.train", "--config", str(path), "--verbose"])

    print(f"prepared {len(commands)} measurement configs in {args.output}")
    if not args.execute:
        print("planning only; no model code was run")
        return
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
