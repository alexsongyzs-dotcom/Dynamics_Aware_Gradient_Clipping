"""Streaming diagnostics for clipping-event and optimizer-state studies.

The module deliberately separates observations from control. Quantities in
this file may be logged by any clipping policy without feeding them back into
the optimizer. This prevents a diagnostic from silently becoming part of an
intervention.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Iterable

import torch
from torch import Tensor


EPS = 1e-12


def stability_pressure(learning_rate: float, top_eigenvalue: float) -> float:
    """Return ``eta * lambda_max``.

    The classical boundary at two is meaningful only for deterministic
    gradient descent on a local quadratic model. Callers must label this as a
    diagnostic proxy for momentum, adaptive optimizers, or stochastic batches.
    """

    return float(learning_rate * top_eigenvalue)


def clipping_coordinate(grad_norm: Tensor | float, threshold: float) -> Tensor:
    """Return ``q_t = ||g_t|| / c_t`` without changing the input device."""

    value = grad_norm if isinstance(grad_norm, Tensor) else torch.tensor(float(grad_norm))
    return value / (float(threshold) + EPS)


def safe_cosine(x: Tensor, y: Tensor) -> float:
    """Cosine similarity with a finite zero-vector convention."""

    x_flat = x.detach().reshape(-1)
    y_flat = y.detach().reshape(-1)
    denominator = torch.linalg.vector_norm(x_flat) * torch.linalg.vector_norm(y_flat)
    if float(denominator) <= EPS:
        return 0.0
    return float(torch.dot(x_flat, y_flat) / denominator)


def gradient_alignment(g_t: Tensor, g_prev: Tensor) -> float:
    """Return the one-step gradient-direction cosine similarity."""

    return safe_cosine(g_t, g_prev)


def curvature_proxy(g_t: Tensor, g_prev: Tensor, theta_t: Tensor, theta_prev: Tensor) -> float:
    """Finite-difference local-curvature proxy, not a Hessian eigenvalue."""

    numerator = torch.linalg.vector_norm(g_t.detach().reshape(-1) - g_prev.detach().reshape(-1))
    denominator = torch.linalg.vector_norm(
        theta_t.detach().reshape(-1) - theta_prev.detach().reshape(-1)
    )
    return float(numerator / (denominator + EPS))


def exposure_ema(event: float, previous: float, beta: float) -> float:
    """Exponentially smoothed exposure indicator."""

    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    return float(beta * previous + (1.0 - beta) * event)


class EpisodeTransition(str, Enum):
    """State transition emitted by :class:`ClippingEpisodeTracker`."""

    NONE = "none"
    ENTER = "enter"
    EXIT = "exit"


@dataclass(frozen=True)
class HysteresisConfig:
    """Thresholds for robust clipping-episode detection."""

    on_margin: float = 0.05
    off_margin: float = 0.05
    min_dwell_steps: int = 3

    def __post_init__(self) -> None:
        if self.on_margin < 0.0 or self.off_margin < 0.0:
            raise ValueError("hysteresis margins must be non-negative")
        if self.min_dwell_steps < 1:
            raise ValueError("min_dwell_steps must be positive")


@dataclass(frozen=True)
class EpisodeObservation:
    """One step of a hysteretic clipping-state sequence."""

    step: int
    q: float
    clipped_state: bool
    transition: EpisodeTransition
    episode_id: int
    dwell_steps: int
    time_since_transition: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["transition"] = self.transition.value
        return result


class ClippingEpisodeTracker:
    """Convert a noisy clipping coordinate into persistent episodes."""

    def __init__(self, config: HysteresisConfig | None = None) -> None:
        self.config = config or HysteresisConfig()
        self.clipped_state = False
        self.episode_id = 0
        self.dwell_steps = 0
        self.last_transition_step = -1
        self.switch_count = 0

    def update(self, q: float, step: int) -> EpisodeObservation:
        if step < 0:
            raise ValueError("step must be non-negative")
        transition = EpisodeTransition.NONE

        if not self.clipped_state and q > 1.0 + self.config.on_margin:
            self.clipped_state = True
            self.episode_id += 1
            self.dwell_steps = 1
            self.last_transition_step = step
            self.switch_count += 1
            transition = EpisodeTransition.ENTER
        elif (
            self.clipped_state
            and self.dwell_steps >= self.config.min_dwell_steps
            and q < 1.0 - self.config.off_margin
        ):
            self.clipped_state = False
            self.dwell_steps = 0
            self.last_transition_step = step
            self.switch_count += 1
            transition = EpisodeTransition.EXIT
        elif self.clipped_state:
            self.dwell_steps += 1

        time_since = step - self.last_transition_step if self.last_transition_step >= 0 else step + 1
        return EpisodeObservation(
            step=step,
            q=float(q),
            clipped_state=self.clipped_state,
            transition=transition,
            episode_id=self.episode_id,
            dwell_steps=self.dwell_steps,
            time_since_transition=time_since,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "clipped_state": self.clipped_state,
            "episode_id": self.episode_id,
            "dwell_steps": self.dwell_steps,
            "last_transition_step": self.last_transition_step,
            "switch_count": self.switch_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        config = state.get("config")
        if isinstance(config, dict):
            self.config = HysteresisConfig(**config)
        self.clipped_state = bool(state["clipped_state"])
        self.episode_id = int(state["episode_id"])
        self.dwell_steps = int(state["dwell_steps"])
        self.last_transition_step = int(state["last_transition_step"])
        self.switch_count = int(state["switch_count"])


@dataclass(frozen=True)
class HistoryFeatures:
    """Causal rolling features available at the current training step."""

    exposure_ema: float
    duty_cycle: float
    switch_rate: float
    burst_age: int
    time_since_transition: int
    mean_intensity: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class DynamicsHistory:
    """Maintain strictly backward-looking clipping-history features."""

    def __init__(self, window: int = 100, beta: float = 0.95) -> None:
        if window < 2:
            raise ValueError("window must be at least two")
        self.window = int(window)
        self.beta = float(beta)
        self._states: deque[float] = deque(maxlen=self.window)
        self._transitions: deque[float] = deque(maxlen=self.window)
        self._intensities: deque[float] = deque(maxlen=self.window)
        self._ema = 0.0

    def update(self, observation: EpisodeObservation, intensity: float) -> HistoryFeatures:
        state = float(observation.clipped_state)
        switched = float(observation.transition is not EpisodeTransition.NONE)
        self._ema = exposure_ema(state, self._ema, self.beta)
        self._states.append(state)
        self._transitions.append(switched)
        self._intensities.append(max(0.0, float(intensity)))
        return HistoryFeatures(
            exposure_ema=self._ema,
            duty_cycle=sum(self._states) / len(self._states),
            switch_rate=sum(self._transitions) / len(self._transitions),
            burst_age=observation.dwell_steps if observation.clipped_state else 0,
            time_since_transition=observation.time_since_transition,
            mean_intensity=sum(self._intensities) / len(self._intensities),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "window": self.window,
            "beta": self.beta,
            "states": list(self._states),
            "transitions": list(self._transitions),
            "intensities": list(self._intensities),
            "ema": self._ema,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.window = int(state["window"])
        self.beta = float(state["beta"])
        self._states = deque((float(x) for x in state["states"]), maxlen=self.window)
        self._transitions = deque((float(x) for x in state["transitions"]), maxlen=self.window)
        self._intensities = deque((float(x) for x in state["intensities"]), maxlen=self.window)
        self._ema = float(state["ema"])


def _parameter_state_pairs(
    parameters: Iterable[Tensor], optimizer: torch.optim.Optimizer
) -> Iterable[tuple[Tensor, dict]]:
    for parameter in parameters:
        if parameter.grad is not None:
            yield parameter, optimizer.state.get(parameter, {})


def adam_second_moment_mismatch(
    parameters: Iterable[Tensor], optimizer: torch.optim.Optimizer
) -> float:
    """RMS(raw gradient) divided by RMS of Adam's debiased second moment."""

    selected = {id(parameter) for parameter in parameters}
    grad_sq_sum = 0.0
    moment_sum = 0.0
    count = 0
    found = False
    for group in optimizer.param_groups:
        beta2 = float(group.get("betas", (0.0, 0.0))[1])
        for parameter in group["params"]:
            if id(parameter) not in selected or parameter.grad is None:
                continue
            state = optimizer.state.get(parameter, {})
            second = state.get("exp_avg_sq")
            if second is None:
                continue
            step_value = state.get("step", 0)
            step = int(step_value.item()) if isinstance(step_value, Tensor) else int(step_value)
            correction = 1.0 - beta2**step if step > 0 else 1.0
            found = True
            grad = parameter.grad.detach()
            grad_sq_sum += float(torch.sum(grad * grad))
            moment_sum += float(torch.sum(second.detach() / max(correction, EPS)))
            count += grad.numel()
    if not found or count == 0:
        return math.nan
    return math.sqrt((grad_sq_sum / count + EPS) / (moment_sum / count + EPS))


def momentum_buffer_alignment(
    parameters: Iterable[Tensor], optimizer: torch.optim.Optimizer
) -> float:
    """Cosine similarity between current gradients and SGD momentum buffers."""

    numerator = 0.0
    grad_sq = 0.0
    buffer_sq = 0.0
    found = False
    for parameter, state in _parameter_state_pairs(parameters, optimizer):
        buffer = state.get("momentum_buffer")
        if buffer is None:
            continue
        found = True
        grad = parameter.grad.detach()
        buf = buffer.detach()
        numerator += float(torch.sum(grad * buf))
        grad_sq += float(torch.sum(grad * grad))
        buffer_sq += float(torch.sum(buf * buf))
    if not found:
        return math.nan
    denominator = math.sqrt(grad_sq * buffer_sq)
    return 0.0 if denominator <= EPS else numerator / denominator


def optimizer_state_metrics(
    parameters: Iterable[Tensor], optimizer: torch.optim.Optimizer
) -> dict[str, float]:
    """Return optimizer-aware diagnostics without modifying optimizer state."""

    params = list(parameters)
    update_rms = [
        float(state["update_rms"])
        for state in optimizer.state.values()
        if "update_rms" in state
    ]
    update_scales = [
        float(state["update_clip_scale"])
        for state in optimizer.state.values()
        if "update_clip_scale" in state
    ]
    return {
        "adam_second_moment_mismatch": adam_second_moment_mismatch(params, optimizer),
        "momentum_buffer_alignment": momentum_buffer_alignment(params, optimizer),
        "stable_adamw_mean_update_rms": (
            sum(update_rms) / len(update_rms) if update_rms else math.nan
        ),
        "stable_adamw_max_clip_scale": max(update_scales) if update_scales else math.nan,
    }
