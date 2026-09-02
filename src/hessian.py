"""Checkpoint-only curvature diagnostics based on Hessian-vector products."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor


EPS = 1e-12


def _flatten(tensors: Sequence[Tensor]) -> Tensor:
    if not tensors:
        return torch.zeros(1)
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _normalize(vectors: Sequence[Tensor]) -> list[Tensor]:
    norm = torch.linalg.vector_norm(_flatten(vectors))
    if float(norm) <= EPS:
        raise ValueError("cannot normalize a zero vector")
    return [vector / norm for vector in vectors]


def hessian_vector_product(loss: Tensor, parameters: Sequence[Tensor], vectors: Sequence[Tensor]) -> list[Tensor]:
    """Compute ``H v`` using double backpropagation."""

    if len(parameters) != len(vectors):
        raise ValueError("parameters and vectors must have equal length")
    gradients = torch.autograd.grad(
        loss,
        parameters,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    safe_gradients = [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    ]
    inner_product = sum(
        torch.sum(gradient * vector)
        for gradient, vector in zip(safe_gradients, vectors)
    )
    products = torch.autograd.grad(
        inner_product,
        parameters,
        retain_graph=False,
        allow_unused=True,
    )
    return [
        torch.zeros_like(parameter) if product is None else product
        for parameter, product in zip(parameters, products)
    ]


def rayleigh_quotient(
    loss_fn: Callable[[object], Tensor],
    parameters: Sequence[Tensor],
    data_batch: object,
    vectors: Sequence[Tensor],
) -> float:
    """Evaluate ``v^T H v / v^T v`` on a fixed probe batch."""

    loss = loss_fn(data_batch)
    products = hessian_vector_product(loss, parameters, vectors)
    flat_v = _flatten(vectors)
    return float(torch.dot(flat_v, _flatten(products)) / (torch.dot(flat_v, flat_v) + EPS))


def top_eigenpair(
    loss_fn: Callable[[object], Tensor],
    parameters: Sequence[Tensor],
    data_batch: object,
    power_iters: int = 20,
    tolerance: float = 1e-6,
    generator: torch.Generator | None = None,
) -> tuple[float, list[Tensor]]:
    """Estimate the largest-magnitude Hessian eigenpair by power iteration."""

    if not parameters:
        raise ValueError("at least one parameter is required")
    vectors = [
        torch.randn(parameter.shape, device=parameter.device, dtype=parameter.dtype, generator=generator)
        for parameter in parameters
    ]
    eigenvalue = 0.0
    for _ in range(int(power_iters)):
        vectors = _normalize(vectors)
        loss = loss_fn(data_batch)
        products = hessian_vector_product(loss, parameters, vectors)
        flat_v = _flatten(vectors)
        flat_hv = _flatten(products)
        updated = float(torch.dot(flat_v, flat_hv) / (torch.dot(flat_v, flat_v) + EPS))
        vectors = products
        if abs(updated - eigenvalue) <= tolerance * (1.0 + abs(eigenvalue)):
            eigenvalue = updated
            break
        eigenvalue = updated
    return eigenvalue, _normalize(vectors)


def top_eigenvalue(
    loss_fn: Callable[[object], Tensor],
    parameters: Sequence[Tensor],
    data_batch: object,
    power_iters: int = 20,
    tol: float = 1e-6,
) -> float:
    """Backward-compatible wrapper returning only the eigenvalue."""

    value, _ = top_eigenpair(
        loss_fn,
        parameters,
        data_batch,
        power_iters=power_iters,
        tolerance=tol,
    )
    return value


def adam_diagonal_preconditioner(
    parameters: Sequence[Tensor], optimizer: torch.optim.Optimizer, eps: float = 1e-8
) -> list[Tensor] | None:
    """Return ``(v_t + eps)^(-1/4)`` for preconditioned-curvature probes.

    The square root on both sides of ``P^(1/2) H P^(1/2)`` yields the fourth
    root of Adam's second moment. ``None`` means buffers are not initialized.
    """

    result: list[Tensor] = []
    for parameter in parameters:
        second = optimizer.state.get(parameter, {}).get("exp_avg_sq")
        if second is None:
            return None
        result.append(torch.pow(second.detach() + eps, -0.25))
    return result


def preconditioned_rayleigh_quotient(
    loss_fn: Callable[[object], Tensor],
    parameters: Sequence[Tensor],
    data_batch: object,
    direction: Sequence[Tensor],
    sqrt_preconditioner: Sequence[Tensor],
) -> float:
    """Rayleigh quotient of ``P^(1/2) H P^(1/2)`` in one direction."""

    if not (len(parameters) == len(direction) == len(sqrt_preconditioner)):
        raise ValueError("preconditioned direction shapes do not match parameters")
    scaled = [vector * scale for vector, scale in zip(direction, sqrt_preconditioner)]
    loss = loss_fn(data_batch)
    products = hessian_vector_product(loss, parameters, scaled)
    transformed = [product * scale for product, scale in zip(products, sqrt_preconditioner)]
    flat_direction = _flatten(direction)
    return float(
        torch.dot(flat_direction, _flatten(transformed))
        / (torch.dot(flat_direction, flat_direction) + EPS)
    )
