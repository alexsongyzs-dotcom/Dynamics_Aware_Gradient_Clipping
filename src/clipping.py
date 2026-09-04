"""Clipping policies and causal intervention primitives.

The code distinguishes three concepts that must not be conflated:

1. a policy *decides* a scalar gain or target update norm;
2. a placement determines whether that decision is applied before optimizer
   state updates or after the optimizer proposes a parameter update;
3. diagnostics observe the resulting event sequence.

This separation supports frozen-gain replay, time/block shuffles, random-gate
controls, and pre-state versus post-update interventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
import csv
import json
import math
from pathlib import Path
import random
from typing import Iterable, Sequence

import torch
from torch import Tensor


EPS = 1e-12


class ClippingPlacement(str, Enum):
    """Where a scalar intervention enters the optimization pipeline."""

    PRE_MOMENT = "pre_moment"
    POST_UPDATE = "post_update"


@dataclass(frozen=True)
class PolicyContext:
    """Information available when a policy chooses a gain."""

    step: int
    grad_norm: float
    loss: float = math.nan
    alignment: float = math.nan
    moment_mismatch: float = math.nan
    update_norm: float = math.nan


@dataclass(frozen=True)
class ClipDecision:
    """A policy decision, independent of where it will be applied."""

    coefficient: float = 1.0
    threshold: float = math.inf
    target_update_norm: float | None = None
    active: bool = False
    source: str = "none"
    controller_state: str = "normal"
    hazard: float = math.nan

    def __post_init__(self) -> None:
        if not 0.0 <= self.coefficient <= 1.0:
            raise ValueError("coefficient must be in [0, 1]")
        if self.target_update_norm is not None and self.target_update_norm < 0.0:
            raise ValueError("target_update_norm must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def clipping_coefficient(grad_norm: Tensor | float, threshold: float) -> Tensor:
    """Return ``min(1, threshold / ||g||)``."""

    value = grad_norm if isinstance(grad_norm, Tensor) else torch.tensor(float(grad_norm))
    return torch.clamp(float(threshold) / (value + EPS), min=0.0, max=1.0)


def clipping_indicator(grad_norm: Tensor | float, threshold: float) -> Tensor:
    """Return ``1{||g|| > threshold}``."""

    value = grad_norm if isinstance(grad_norm, Tensor) else torch.tensor(float(grad_norm))
    return (value > float(threshold)).float()


def clipping_intensity(grad_norm: Tensor | float, threshold: float) -> Tensor:
    """Return ``(1 - threshold / ||g||)_+``."""

    value = grad_norm if isinstance(grad_norm, Tensor) else torch.tensor(float(grad_norm))
    return torch.clamp(1.0 - float(threshold) / (value + EPS), min=0.0, max=1.0)


def global_norm_clip(gradient: Tensor, threshold: float) -> Tensor:
    """Return a clipped copy of a single tensor."""

    norm = torch.linalg.vector_norm(gradient)
    coefficient = clipping_coefficient(norm, threshold).to(gradient)
    return gradient * coefficient


def trainable_parameters(module: torch.nn.Module) -> list[Tensor]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def flatten_gradients(parameters: Iterable[Tensor]) -> Tensor:
    """Flatten current gradients without mutating them."""

    parts = [parameter.grad.detach().reshape(-1) for parameter in parameters if parameter.grad is not None]
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def gradient_norm(parameters: Iterable[Tensor]) -> float:
    """Global L2 norm of current gradients."""

    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = torch.sum(parameter.grad.detach() ** 2)
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(torch.sqrt(squared))


def scale_gradients_(parameters: Iterable[Tensor], coefficient: float) -> None:
    """Apply an in-place scalar gain before the optimizer updates its state."""

    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("coefficient must be in [0, 1]")
    if coefficient == 1.0:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(coefficient)


def apply_agc_(parameters: Iterable[Tensor], clip_value: float = 0.01, eps: float = 1e-3) -> float:
    """Apply unit/tensor-wise Adaptive Gradient Clipping in place.

    Returns the smallest coefficient applied to any tensor for logging. Bias
    and one-dimensional tensors use their complete tensor norms.
    """

    minimum = 1.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter_norm = torch.linalg.vector_norm(parameter.detach()).clamp_min(eps)
        gradient_value = parameter.grad.detach()
        grad_value_norm = torch.linalg.vector_norm(gradient_value)
        maximum_norm = parameter_norm * clip_value
        coefficient = float(torch.clamp(maximum_norm / (grad_value_norm + EPS), max=1.0))
        if coefficient < 1.0:
            parameter.grad.mul_(coefficient)
        minimum = min(minimum, coefficient)
    return minimum


def snapshot_parameters(parameters: Iterable[Tensor]) -> list[Tensor]:
    """Clone parameter values before ``optimizer.step``."""

    return [parameter.detach().clone() for parameter in parameters]


@dataclass(frozen=True)
class UpdateApplication:
    raw_update_norm: float
    applied_update_norm: float
    coefficient: float


def apply_post_update_decision_(
    parameters: Sequence[Tensor],
    before: Sequence[Tensor],
    decision: ClipDecision,
    allow_enlarge: bool = False,
) -> UpdateApplication:
    """Scale an optimizer-proposed parameter delta while preserving its state.

    The optimizer is stepped before this function is called, so momentum or
    Adam moments retain the raw-gradient information.  Only the parameter
    delta applied to the model is rescaled.
    """

    if len(parameters) != len(before):
        raise ValueError("parameter snapshot does not match parameter list")

    raw_sq = 0.0
    for parameter, old in zip(parameters, before):
        delta = parameter.detach() - old
        raw_sq += float(torch.sum(delta * delta))
    raw_norm = math.sqrt(raw_sq)

    if decision.target_update_norm is not None:
        coefficient = decision.target_update_norm / (raw_norm + EPS)
        if not allow_enlarge:
            coefficient = min(1.0, coefficient)
    else:
        coefficient = decision.coefficient
    coefficient = max(0.0, float(coefficient))

    with torch.no_grad():
        for parameter, old in zip(parameters, before):
            parameter.copy_(old + coefficient * (parameter - old))
    return UpdateApplication(
        raw_update_norm=raw_norm,
        applied_update_norm=raw_norm * coefficient,
        coefficient=coefficient,
    )


def parameter_update_norm(parameters: Sequence[Tensor], before: Sequence[Tensor]) -> float:
    """Norm of the actual parameter change since a snapshot."""

    if len(parameters) != len(before):
        raise ValueError("parameter snapshot does not match parameter list")
    squared = 0.0
    for parameter, old in zip(parameters, before):
        delta = parameter.detach() - old
        squared += float(torch.sum(delta * delta))
    return math.sqrt(squared)


class GainPolicy(ABC):
    """Stateful scalar-gain policy interface."""

    name = "abstract"

    @abstractmethod
    def decide(self, context: PolicyContext) -> ClipDecision:
        raise NotImplementedError

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state:
            raise ValueError(f"{self.name} is stateless but received state")


class NoClippingPolicy(GainPolicy):
    name = "none"

    def decide(self, context: PolicyContext) -> ClipDecision:
        return ClipDecision(source=self.name)


class FixedThresholdPolicy(GainPolicy):
    name = "fixed"

    def __init__(self, threshold: float) -> None:
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)

    def decide(self, context: PolicyContext) -> ClipDecision:
        coefficient = min(1.0, self.threshold / (context.grad_norm + EPS))
        return ClipDecision(
            coefficient=coefficient,
            threshold=self.threshold,
            active=coefficient < 1.0,
            source=self.name,
        )


class AutoClipPolicy(GainPolicy):
    """Historical-gradient-norm percentile baseline."""

    name = "autoclip"

    def __init__(self, percentile: float = 10.0, window: int = 1000, warmup: int = 20) -> None:
        if not 0.0 < percentile <= 100.0:
            raise ValueError("percentile must be in (0, 100]")
        self.percentile = float(percentile)
        self.window = int(window)
        self.warmup = int(warmup)
        self.history: deque[float] = deque(maxlen=self.window)

    def decide(self, context: PolicyContext) -> ClipDecision:
        self.history.append(float(context.grad_norm))
        if len(self.history) < self.warmup or self.percentile == 100.0:
            return ClipDecision(source=self.name)
        values = torch.tensor(list(self.history), dtype=torch.float64)
        threshold = float(torch.quantile(values, self.percentile / 100.0))
        coefficient = min(1.0, threshold / (context.grad_norm + EPS))
        return ClipDecision(
            coefficient=coefficient,
            threshold=threshold,
            active=coefficient < 1.0,
            source=self.name,
        )

    def state_dict(self) -> dict[str, object]:
        return {"history": list(self.history)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.history = deque((float(x) for x in state["history"]), maxlen=self.window)


class AdaGCPolicy(GainPolicy):
    """AdaGC per-tensor EMA baseline with global-clipping warm-up.

    The EMA is updated from clipped tensor norms, matching the paper's stated
    recurrence. A single minimum coefficient is returned for coarse logging;
    the actual intervention remains tensor-wise.
    """

    name = "adagc"

    def __init__(
        self,
        relative_threshold: float = 1.04,
        beta: float = 0.99,
        warmup_steps: int = 100,
        warmup_global_threshold: float = 1.0,
    ) -> None:
        if relative_threshold <= 0.0 or warmup_global_threshold <= 0.0:
            raise ValueError("AdaGC thresholds must be positive")
        if not 0.0 <= beta < 1.0 or warmup_steps < 0:
            raise ValueError("invalid AdaGC beta or warmup_steps")
        self.relative_threshold = float(relative_threshold)
        self.beta = float(beta)
        self.warmup_steps = int(warmup_steps)
        self.warmup_global_threshold = float(warmup_global_threshold)
        self.step = 0
        self.ema_norms: list[float] = []

    def decide(self, context: PolicyContext) -> ClipDecision:
        return ClipDecision(source=self.name)

    def apply_(self, parameters: Sequence[Tensor], global_grad_norm: float) -> float:
        with_grad = [parameter for parameter in parameters if parameter.grad is not None]
        if self.ema_norms and len(self.ema_norms) != len(with_grad):
            raise ValueError("AdaGC parameter set changed during training")

        minimum = 1.0
        if self.step < self.warmup_steps:
            coefficient = min(1.0, self.warmup_global_threshold / (global_grad_norm + EPS))
            scale_gradients_(with_grad, coefficient)
            clipped_norms = [float(torch.linalg.vector_norm(parameter.grad.detach())) for parameter in with_grad]
            if not self.ema_norms:
                self.ema_norms = clipped_norms
            else:
                self.ema_norms = [
                    min(previous, current)
                    for previous, current in zip(self.ema_norms, clipped_norms)
                ]
            minimum = coefficient
        else:
            if not self.ema_norms:
                self.ema_norms = [
                    float(torch.linalg.vector_norm(parameter.grad.detach()))
                    for parameter in with_grad
                ]
            updated: list[float] = []
            for parameter, previous in zip(with_grad, self.ema_norms):
                raw_norm = float(torch.linalg.vector_norm(parameter.grad.detach()))
                coefficient = min(
                    1.0,
                    self.relative_threshold * previous / (raw_norm + EPS),
                )
                parameter.grad.mul_(coefficient)
                clipped_norm = raw_norm * coefficient
                updated.append(self.beta * previous + (1.0 - self.beta) * clipped_norm)
                minimum = min(minimum, coefficient)
            self.ema_norms = updated
        self.step += 1
        return minimum

    def state_dict(self) -> dict[str, object]:
        return {"step": self.step, "ema_norms": list(self.ema_norms)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.step = int(state["step"])
        self.ema_norms = [float(value) for value in state["ema_norms"]]


class FrozenGainPolicy(GainPolicy):
    """Exogenous gain sequence for replay and timing controls."""

    name = "frozen_gain"

    def __init__(self, gains: Sequence[float], repeat_last: bool = False) -> None:
        if not gains:
            raise ValueError("frozen gain sequence must be non-empty")
        if any(not 0.0 <= float(value) <= 1.0 for value in gains):
            raise ValueError("all frozen gains must be in [0, 1]")
        self.gains = [float(value) for value in gains]
        self.repeat_last = bool(repeat_last)

    def decide(self, context: PolicyContext) -> ClipDecision:
        if context.step >= len(self.gains):
            if not self.repeat_last:
                raise IndexError("frozen gain sequence exhausted")
            coefficient = self.gains[-1]
        else:
            coefficient = self.gains[context.step]
        threshold = coefficient * context.grad_norm if coefficient < 1.0 else math.inf
        return ClipDecision(
            coefficient=coefficient,
            threshold=threshold,
            active=coefficient < 1.0,
            source=self.name,
        )


class FrozenTargetNormPolicy(GainPolicy):
    """Post-update branch policy that replays target applied-update norms."""

    name = "frozen_target_norm"

    def __init__(self, target_norms: Sequence[float], repeat_last: bool = False) -> None:
        if not target_norms or any(float(value) < 0.0 for value in target_norms):
            raise ValueError("target norm sequence must be non-empty and non-negative")
        self.target_norms = [float(value) for value in target_norms]
        self.repeat_last = bool(repeat_last)

    def decide(self, context: PolicyContext) -> ClipDecision:
        if context.step >= len(self.target_norms):
            if not self.repeat_last:
                raise IndexError("target norm sequence exhausted")
            target = self.target_norms[-1]
        else:
            target = self.target_norms[context.step]
        return ClipDecision(
            target_update_norm=target,
            active=True,
            source=self.name,
        )


class EventTimedClippingPolicy(GainPolicy):
    """Contingent DAGC-v2 controller with a single aggressiveness parameter.

    This policy is intentionally marked experimental. It should only be used
    after event-history predictors and causal interventions pass the gates in
    ``docs/paper_idea_review_and_revised_plan_cn.md``.
    """

    name = "event_timed"

    def __init__(
        self,
        aggressiveness: float = 0.5,
        window: int = 200,
        warmup: int = 50,
        min_dwell_steps: int = 5,
    ) -> None:
        if not 0.0 <= aggressiveness <= 1.0:
            raise ValueError("aggressiveness must be in [0, 1]")
        self.aggressiveness = float(aggressiveness)
        self.window = int(window)
        self.warmup = int(warmup)
        self.min_dwell_steps = int(min_dwell_steps)
        self.norm_history: deque[float] = deque(maxlen=self.window)
        self.state = "normal"
        self.state_age = 0

    def _robust_shock(self, grad_norm: float) -> float:
        values = torch.tensor(list(self.norm_history), dtype=torch.float64)
        logs = torch.log(values.clamp_min(EPS))
        center = torch.median(logs)
        mad = torch.median(torch.abs(logs - center)).clamp_min(1e-3)
        return float((math.log(max(grad_norm, EPS)) - center) / (1.4826 * mad))

    def decide(self, context: PolicyContext) -> ClipDecision:
        self.norm_history.append(float(context.grad_norm))
        if len(self.norm_history) < self.warmup:
            return ClipDecision(source=self.name, controller_state=self.state)

        shock = max(0.0, self._robust_shock(context.grad_norm))
        reversal = 0.0 if math.isnan(context.alignment) else max(0.0, -context.alignment)
        mismatch = (
            0.0
            if math.isnan(context.moment_mismatch)
            else max(0.0, math.log(max(context.moment_mismatch, EPS)))
        )
        hazard = max(shock / 3.0, reversal, mismatch / 2.0)
        on_threshold = 1.0 - 0.5 * self.aggressiveness
        off_threshold = 0.5 * on_threshold

        if self.state == "normal" and hazard > on_threshold:
            self.state = "protect"
            self.state_age = 0
        elif (
            self.state == "protect"
            and self.state_age >= self.min_dwell_steps
            and hazard < off_threshold
        ):
            self.state = "normal"
            self.state_age = 0
        else:
            self.state_age += 1

        quantile = 0.99 if self.state == "normal" else 0.90 - 0.50 * self.aggressiveness
        values = torch.tensor(list(self.norm_history), dtype=torch.float64)
        threshold = float(torch.quantile(values, quantile))
        coefficient = min(1.0, threshold / (context.grad_norm + EPS))
        return ClipDecision(
            coefficient=coefficient,
            threshold=threshold,
            active=coefficient < 1.0,
            source=self.name,
            controller_state=self.state,
            hazard=hazard,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "norm_history": list(self.norm_history),
            "state": self.state,
            "state_age": self.state_age,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.norm_history = deque(
            (float(x) for x in state["norm_history"]), maxlen=self.window
        )
        self.state = str(state["state"])
        self.state_age = int(state["state_age"])


def make_time_shuffled_gains(gains: Sequence[float], seed: int) -> list[float]:
    """Randomly permute a gain multiset while preserving every value."""

    result = [float(value) for value in gains]
    random.Random(seed).shuffle(result)
    return result


def make_block_shuffled_gains(gains: Sequence[float], seed: int, block_size: int) -> list[float]:
    """Shuffle contiguous gain blocks while preserving within-block order."""

    if block_size < 1:
        raise ValueError("block_size must be positive")
    blocks = [list(map(float, gains[start : start + block_size])) for start in range(0, len(gains), block_size)]
    random.Random(seed).shuffle(blocks)
    return [value for block in blocks for value in block]


def make_random_gate_gains(gains: Sequence[float], seed: int) -> list[float]:
    """Match active-step count and active gains but randomize their locations."""

    active = [float(value) for value in gains if float(value) < 1.0]
    result = [1.0] * len(gains)
    rng = random.Random(seed)
    positions = rng.sample(range(len(gains)), k=len(active))
    rng.shuffle(active)
    for position, value in zip(positions, active):
        result[position] = value
    return result


def load_numeric_sequence(path: str | Path, field: str) -> list[float]:
    """Load a numeric sequence from JSON, JSONL, or CSV."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        values = payload[field] if isinstance(payload, dict) else payload
        return [float(value) for value in values]
    if suffix == ".jsonl":
        values = []
        with source.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                values.append(float(row[field]))
        return values
    if suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as stream:
            return [float(row[field]) for row in csv.DictReader(stream)]
    raise ValueError(f"unsupported sequence file: {source}")


def build_policy(config: dict) -> GainPolicy:
    """Construct a scalar policy from a plain configuration mapping."""

    name = str(config.get("name", "none")).lower()
    if name == "none":
        return NoClippingPolicy()
    if name == "fixed":
        return FixedThresholdPolicy(float(config["threshold"]))
    if name == "autoclip":
        return AutoClipPolicy(
            percentile=float(config.get("percentile", 10.0)),
            window=int(config.get("window", 1000)),
            warmup=int(config.get("warmup", 20)),
        )
    if name == "adagc":
        return AdaGCPolicy(
            relative_threshold=float(config.get("relative_threshold", 1.04)),
            beta=float(config.get("beta", 0.99)),
            warmup_steps=int(config.get("warmup_steps", 100)),
            warmup_global_threshold=float(config.get("warmup_global_threshold", 1.0)),
        )
    if name in {"frozen_gain", "time_shuffled", "block_shuffled", "random_gate"}:
        gains = load_numeric_sequence(config["sequence_path"], config.get("field", "coefficient"))
        seed = int(config.get("shuffle_seed", 0))
        if name == "time_shuffled":
            gains = make_time_shuffled_gains(gains, seed)
        elif name == "block_shuffled":
            gains = make_block_shuffled_gains(gains, seed, int(config.get("block_size", 10)))
        elif name == "random_gate":
            gains = make_random_gate_gains(gains, seed)
        return FrozenGainPolicy(gains, repeat_last=bool(config.get("repeat_last", False)))
    if name == "frozen_target_norm":
        targets = load_numeric_sequence(config["sequence_path"], config.get("field", "applied_update_norm"))
        return FrozenTargetNormPolicy(targets, repeat_last=bool(config.get("repeat_last", False)))
    if name == "event_timed":
        return EventTimedClippingPolicy(
            aggressiveness=float(config.get("aggressiveness", 0.5)),
            window=int(config.get("window", 200)),
            warmup=int(config.get("warmup", 50)),
            min_dwell_steps=int(config.get("min_dwell_steps", 5)),
        )
    if name == "agc":
        return NoClippingPolicy()  # structured AGC is applied explicitly by the runner
    raise ValueError(f"unknown clipping policy: {name}")
