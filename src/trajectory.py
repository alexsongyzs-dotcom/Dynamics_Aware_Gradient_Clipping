"""Functional trajectory and final-solution comparisons for paired runs."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


EPS = 1e-12


def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _flatten_parameters(value: Tensor | nn.Module | Iterable[Tensor]) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().reshape(-1)
    parameters = value.parameters() if isinstance(value, nn.Module) else value
    parts = [parameter.detach().reshape(-1) for parameter in parameters]
    return torch.cat(parts) if parts else torch.zeros(1)


def parameter_distance(
    theta1: Tensor | nn.Module | Iterable[Tensor],
    theta2: Tensor | nn.Module | Iterable[Tensor],
) -> float:
    """Normalized parameter distance, retained only as a diagnostic."""

    first = _flatten_parameters(theta1)
    second = _flatten_parameters(theta2).to(first)
    if first.shape != second.shape:
        raise ValueError("parameter vectors have different shapes")
    return float(torch.linalg.vector_norm(first - second) / (1.0 + torch.linalg.vector_norm(first)))


@torch.no_grad()
def function_distance(
    model1: nn.Module,
    model2: nn.Module,
    probe_loader,
    metric: str = "probability_l2",
) -> float:
    """Average functional distance on a fixed, deterministic probe set."""

    metric = metric.lower()
    training1, training2 = model1.training, model2.training
    model1.eval()
    model2.eval()
    device1, device2 = _device(model1), _device(model2)
    total = 0.0
    count = 0
    for batch in probe_loader:
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits1 = model1(inputs.to(device1))
        logits2 = model2(inputs.to(device2)).to(logits1)
        if metric in {"l2", "logit_l2"}:
            values = torch.linalg.vector_norm(logits1 - logits2, dim=1)
        elif metric == "probability_l2":
            values = torch.linalg.vector_norm(
                torch.softmax(logits1, dim=1) - torch.softmax(logits2, dim=1), dim=1
            )
        elif metric == "symmetric_kl":
            logp = torch.log_softmax(logits1, dim=1)
            logq = torch.log_softmax(logits2, dim=1)
            p, q = logp.exp(), logq.exp()
            values = 0.5 * (
                torch.sum(p * (logp - logq), dim=1) + torch.sum(q * (logq - logp), dim=1)
            )
        elif metric == "disagreement":
            values = (logits1.argmax(dim=1) != logits2.argmax(dim=1)).float()
        else:
            raise ValueError(f"unknown function-distance metric: {metric}")
        total += float(torch.sum(values))
        count += int(values.numel())
    model1.train(training1)
    model2.train(training2)
    return total / max(count, 1)


@torch.no_grad()
def _mean_cross_entropy(model: nn.Module, data_loader) -> float:
    device = _device(model)
    training = model.training
    model.eval()
    total_loss = 0.0
    total_count = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        total_loss += float(F.cross_entropy(model(inputs), targets, reduction="sum"))
        total_count += int(targets.numel())
    model.train(training)
    return total_loss / max(total_count, 1)


@torch.no_grad()
def linear_mode_connectivity_barrier(
    model1: nn.Module,
    model2: nn.Module,
    data_loader,
    n_steps: int = 11,
) -> float:
    """Maximum interpolation loss above the worse endpoint loss."""

    if n_steps < 2:
        raise ValueError("n_steps must be at least two")
    first_state = model1.state_dict()
    second_state = model2.state_dict()
    if first_state.keys() != second_state.keys():
        raise ValueError("models do not have matching state dictionaries")
    interpolated = copy.deepcopy(model1)
    losses: list[float] = []
    for weight in torch.linspace(0.0, 1.0, n_steps):
        state: dict[str, Tensor] = {}
        scalar = float(weight)
        for name in first_state:
            first = first_state[name]
            second = second_state[name].to(first)
            state[name] = (1.0 - scalar) * first + scalar * second if first.is_floating_point() else first
        interpolated.load_state_dict(state)
        losses.append(_mean_cross_entropy(interpolated, data_loader))
    return max(0.0, max(losses) - max(losses[0], losses[-1]))


def _resolve_layers(model: nn.Module, layer_ids: Sequence[str | nn.Module]) -> list[nn.Module]:
    named = dict(model.named_modules())
    layers = []
    for layer in layer_ids:
        if isinstance(layer, str):
            if layer not in named:
                raise KeyError(f"unknown layer: {layer}")
            layers.append(named[layer])
        else:
            layers.append(layer)
    return layers


def _collect_features(model: nn.Module, data_loader, layers: Sequence[nn.Module]) -> list[Tensor]:
    collected: list[list[Tensor]] = [[] for _ in layers]
    handles = []
    for index, layer in enumerate(layers):
        def hook(_module, _inputs, output, slot=index):
            value = output[0] if isinstance(output, (tuple, list)) else output
            collected[slot].append(value.detach().flatten(start_dim=1).cpu())

        handles.append(layer.register_forward_hook(hook))
    training = model.training
    model.eval()
    device = _device(model)
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            model(inputs.to(device))
    model.train(training)
    for handle in handles:
        handle.remove()
    return [torch.cat(parts, dim=0).float() for parts in collected]


def linear_cka(first: Tensor, second: Tensor) -> float:
    """Linear centered-kernel alignment between two feature matrices."""

    if first.shape[0] != second.shape[0]:
        raise ValueError("feature matrices must have the same number of rows")
    first = first - first.mean(dim=0, keepdim=True)
    second = second - second.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(first.T @ second) ** 2
    first_norm = torch.linalg.matrix_norm(first.T @ first)
    second_norm = torch.linalg.matrix_norm(second.T @ second)
    return float(cross / (first_norm * second_norm + EPS))


def cka_similarity(
    model1: nn.Module,
    model2: nn.Module,
    data_loader,
    layer_ids: Sequence[str],
) -> float:
    """Mean linear CKA over matching named layers."""

    if not layer_ids:
        raise ValueError("at least one layer is required")
    features1 = _collect_features(model1, data_loader, _resolve_layers(model1, layer_ids))
    features2 = _collect_features(model2, data_loader, _resolve_layers(model2, layer_ids))
    return sum(linear_cka(first, second) for first, second in zip(features1, features2)) / len(layer_ids)


@torch.no_grad()
def expected_calibration_error(model: nn.Module, data_loader, bins: int = 15) -> float:
    """Top-label expected calibration error."""

    device = _device(model)
    training = model.training
    model.eval()
    confidences, correctness = [], []
    for inputs, targets in data_loader:
        probabilities = torch.softmax(model(inputs.to(device)), dim=1)
        confidence, prediction = probabilities.max(dim=1)
        confidences.append(confidence.cpu())
        correctness.append((prediction.cpu() == targets).float())
    model.train(training)
    confidence = torch.cat(confidences)
    correct = torch.cat(correctness)
    edges = torch.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if torch.any(mask):
            error += float(mask.float().mean() * torch.abs(correct[mask].mean() - confidence[mask].mean()))
    return error
