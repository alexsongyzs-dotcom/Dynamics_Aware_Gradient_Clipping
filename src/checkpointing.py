"""Complete experiment checkpoints for deterministic causal branches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ResumePosition:
    epoch: int
    global_step: int


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict,
    scheduler: object | None = None,
    policy: object | None = None,
    sampler: object | None = None,
    episode_tracker: object | None = None,
    history_tracker: object | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable checkpoint containing every branch-relevant state."""

    payload: dict[str, Any] = {
        "schema_version": 2,
        "position": {"epoch": int(epoch), "global_step": int(global_step)},
        "config": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(),
        "metadata": metadata or {},
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if policy is not None:
        payload["policy"] = policy.state_dict()
    if sampler is not None:
        payload["sampler"] = sampler.state_dict()
    if episode_tracker is not None:
        payload["episode_tracker"] = episode_tracker.state_dict()
    if history_tracker is not None:
        payload["history_tracker"] = history_tracker.state_dict()
    return payload


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically save a checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None = None,
    policy: object | None = None,
    sampler: object | None = None,
    episode_tracker: object | None = None,
    history_tracker: object | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> tuple[ResumePosition, dict[str, Any]]:
    """Restore all available state and return the resume position and payload."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if int(payload.get("schema_version", 0)) != 2:
        raise ValueError("unsupported checkpoint schema")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if policy is not None and "policy" in payload:
        policy.load_state_dict(payload["policy"])
    if sampler is not None and "sampler" in payload:
        sampler.load_state_dict(payload["sampler"])
    if episode_tracker is not None and "episode_tracker" in payload:
        episode_tracker.load_state_dict(payload["episode_tracker"])
    if history_tracker is not None and "history_tracker" in payload:
        history_tracker.load_state_dict(payload["history_tracker"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    position = ResumePosition(**payload["position"])
    return position, payload
