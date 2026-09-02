"""Materialize a sweep plan and optionally execute it on the target machine.

Default behavior is planning only. Training starts only with ``--execute``.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.configuration import load_config, set_dotted


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def materialize_sweep(sweep_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    payload = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    sweep = payload["sweep"]
    base_path = Path(sweep["base_config"])
    if not base_path.is_absolute():
        base_path = sweep_path.parent / base_path
    grid = dict(sweep["grid"])
    keys = list(grid)
    combinations = itertools.product(*(grid[key] for key in keys))
    seeds = [int(seed) for seed in sweep["seeds"]]
    output_root = Path(sweep.get("output_root", f"results/sweeps/{sweep['name']}"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    planned: list[dict[str, Any]] = []
    for values in combinations:
        labels = [f"{key.replace('.', '-')}-{_slug(value)}" for key, value in zip(keys, values)]
        for seed in seeds:
            run_id = "__".join([sweep["name"], *labels, f"seed-{seed}"])
            config = load_config(base_path)
            for key, value in zip(keys, values):
                set_dotted(config, key, value)
            config["seed"] = seed
            config["run_id"] = run_id
            config["output_dir"] = str(output_root / "runs")
            config_path = config_dir / f"{run_id}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            planned.append(
                {
                    "run_id": run_id,
                    "config": str(config_path),
                    "command": [
                        sys.executable,
                        "-m",
                        "src.train",
                        "--config",
                        str(config_path),
                        "--verbose",
                    ],
                }
            )
    plan_path = output_root / "sweep_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "sweep": sweep["name"],
                "source": str(sweep_path.resolve()),
                "base_config": str(base_path.resolve()),
                "run_count": len(planned),
                "runs": planned,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return plan_path, planned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the materialized configurations; omitted means plan only",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    plan_path, planned = materialize_sweep(args.config.resolve())
    selected = planned if args.max_runs is None else planned[: args.max_runs]
    print(f"materialized {len(planned)} runs: {plan_path}")
    if not args.execute:
        print("planning only; pass --execute on the Linux training host")
        return
    for run in selected:
        print(" ".join(run["command"]), flush=True)
        subprocess.run(run["command"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
