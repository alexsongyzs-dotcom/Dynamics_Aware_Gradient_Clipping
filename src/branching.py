"""Prepare immutable intervention sequences and paired-branch manifests.

This module performs no training. It turns a completed reference run into the
frozen-gain, timing-shuffle, random-gate, and post-update target sequences used
by the causal tests in the research plan.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.clipping import (
    make_block_shuffled_gains,
    make_random_gate_gains,
    make_time_shuffled_gains,
)
from src.configuration import deep_merge, load_config


def read_step_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"row {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"reference step log is empty: {source}")
    return rows


def _numeric(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for index, row in enumerate(rows):
        value = row.get(field)
        if value is None:
            raise KeyError(f"missing {field!r} at reference row {index}")
        values.append(float(value))
    return values


def sequence_digest(values: list[float]) -> str:
    canonical = json.dumps(values, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def prepare_causal_branches(
    *,
    reference_steps: str | Path,
    base_config: str | Path,
    output_dir: str | Path,
    shuffle_seed: int,
    block_size: int = 50,
) -> Path:
    """Create causal sequences, complete branch configs, and a manifest."""

    reference_path = Path(reference_steps).expanduser().resolve()
    base_path = Path(base_config).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    rows = read_step_rows(reference_path)
    gains = _numeric(rows, "coefficient")
    target_norms = _numeric(rows, "applied_update_norm")
    sequences = {
        "reference_gain": gains,
        "time_shuffled_gain": make_time_shuffled_gains(gains, shuffle_seed),
        "block_shuffled_gain": make_block_shuffled_gains(gains, shuffle_seed, block_size),
        "random_gate_gain": make_random_gate_gains(gains, shuffle_seed),
        "reference_applied_update_norm": target_norms,
    }
    sequence_path = destination / "intervention_sequences.json"
    sequence_path.write_text(
        json.dumps(sequences, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    base = load_config(base_path)
    branch_specs = {
        "reference_gain_replay_pre_moment": {
            "name": "frozen_gain",
            "placement": "pre_moment",
            "field": "reference_gain",
        },
        "time_shuffled_pre_moment": {
            "name": "frozen_gain",
            "placement": "pre_moment",
            "field": "time_shuffled_gain",
        },
        "block_shuffled_pre_moment": {
            "name": "frozen_gain",
            "placement": "pre_moment",
            "field": "block_shuffled_gain",
        },
        "random_gate_pre_moment": {
            "name": "frozen_gain",
            "placement": "pre_moment",
            "field": "random_gate_gain",
        },
        "reference_norm_post_update": {
            "name": "frozen_target_norm",
            "placement": "post_update",
            "field": "reference_applied_update_norm",
        },
    }

    config_paths: dict[str, str] = {}
    for branch_name, clipping in branch_specs.items():
        clipping = {
            **clipping,
            "sequence_path": str(sequence_path),
            "repeat_last": False,
        }
        config = deep_merge(
            base,
            {
                "run_id": branch_name,
                "clipping": clipping,
                "causal_branch": {
                    "reference_steps": str(reference_path),
                    "shuffle_seed": int(shuffle_seed),
                    "block_size": int(block_size),
                },
            },
        )
        config_path = destination / f"{branch_name}.yaml"
        _write_yaml(config_path, config)
        config_paths[branch_name] = str(config_path)

    manifest = {
        "schema_version": 1,
        "reference_steps": str(reference_path),
        "base_config": str(base_path),
        "number_of_steps": len(rows),
        "shuffle_seed": int(shuffle_seed),
        "block_size": int(block_size),
        "sequence_file": str(sequence_path),
        "sequence_digests": {
            name: sequence_digest(values) for name, values in sequences.items()
        },
        "branch_configs": config_paths,
        "invariants": {
            "time_shuffle_preserves_gain_multiset": True,
            "block_shuffle_preserves_gain_multiset": True,
            "random_gate_preserves_active_gain_multiset": True,
            "post_update_replays_reference_applied_update_norm": True,
        },
    }
    manifest_path = destination / "branch_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path
