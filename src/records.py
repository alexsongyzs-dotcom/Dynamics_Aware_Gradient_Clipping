"""Tidy, append-only run records with explicit schema and configuration hash."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import copy
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 2


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def config_hash(config: dict) -> str:
    scientific_config = copy.deepcopy(config)
    checkpointing = scientific_config.get("checkpointing")
    if isinstance(checkpointing, dict):
        checkpointing.pop("resume_from", None)
    canonical = json.dumps(
        _json_value(scientific_config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class RunRecorder:
    """Append per-step rows and write a separate immutable run manifest."""

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str,
        config: dict,
        append_existing: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.config = config
        self.config_hash = config_hash(config)
        self.steps_path = self.output_dir / f"{self.run_id}.steps.jsonl"
        self.summary_path = self.output_dir / f"{self.run_id}.summary.json"
        self.manifest_path = self.output_dir / f"{self.run_id}.manifest.json"
        existing = [
            path
            for path in (self.steps_path, self.summary_path, self.manifest_path)
            if path.exists()
        ]
        if existing and not append_existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(f"run_id already has artifacts: {names}")
        if append_existing:
            if not self.manifest_path.exists():
                raise FileNotFoundError("resume requested but the run manifest is absent")
            prior = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if prior.get("config_hash") != self.config_hash:
                raise ValueError("resume configuration does not match the existing run manifest")
            return
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "config": _json_value(config),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def append_step(self, row: dict[str, Any]) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            **row,
        }
        with self.steps_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_value(payload), ensure_ascii=False, sort_keys=True) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            **summary,
        }
        self.summary_path.write_text(
            json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
