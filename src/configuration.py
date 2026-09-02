"""Plain-YAML configuration loading with explicit inheritance and overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML config, resolving an optional ``base_config`` recursively."""

    source = Path(path).expanduser().resolve()
    if source in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, source))
        raise ValueError(f"cyclic base_config chain: {chain}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"configuration root must be a mapping: {source}")
    base_ref = payload.pop("base_config", None)
    if base_ref is None:
        return payload
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = source.parent / base_path
    base = load_config(base_path, (*_stack, source))
    return deep_merge(base, payload)


def set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    """Set ``optimizer.lr``-style paths in a nested configuration."""

    parts = key.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid dotted key: {key!r}")
    cursor = config
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(f"cannot set {key!r}: {part!r} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse one ``key=value`` CLI override using YAML scalar semantics."""

    if "=" not in text:
        raise ValueError(f"override must be key=value: {text!r}")
    key, raw_value = text.split("=", 1)
    return key, yaml.safe_load(raw_value)
