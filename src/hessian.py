"""Hessian spectral measurements.

Expensive curvature information is measured only at selected checkpoints using
Hessian-vector products and power iteration (no full eigendecomposition during
ordinary training).
"""

from __future__ import annotations

import torch
from torch import Tensor


def hvp(loss: Tensor, params: list[Tensor], v: list[Tensor]) -> list[Tensor]:
    """Hessian-vector product H v via double backprop.

    Args:
        loss: scalar loss (already backward-computed with create_graph).
        params: parameters with respect to which the Hessian is taken.
        v: list of vectors, same shapes as params.

    Returns:
        List Hv, same shapes as params.
    """
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    flat_grads = [g.reshape(-1) for g in grads]
    flat_v = [vv.reshape(-1) for vv in v]
    Hv = torch.autograd.grad(flat_grads, params, grad_outputs=flat_v, retain_graph=True)
    return Hv


def _flat(vs: list[Tensor]) -> Tensor:
    return torch.cat([v.reshape(-1) for v in vs])


def top_eigenvalue(
    loss_fn,
    params: list[Tensor],
    data_batch,
    power_iters: int = 20,
    tol: float = 1e-6,
) -> float:
    """Largest eigenvalue of the Hessian via power iteration.

    Args:
        loss_fn: callable producing a scalar loss from data_batch (must set
                 requires_grad on params).
        params: parameters of the model.
        data_batch: batch consumed by loss_fn.
        power_iters: number of power iterations.
        tol: relative convergence tolerance.

    Returns:
        Estimated top eigenvalue (float).
    """
    v = [torch.randn_like(p) for p in params]
    lam = 0.0
    for _ in range(power_iters):
        vn = _flat(v).norm()
        if vn < 1e-12:
            break
        v = [vv / vn for vv in v]
        loss = loss_fn(data_batch)
        Hv = hvp(loss, params, v)
        num = _flat(Hv).dot(_flat(v))
        den = _flat(v).dot(_flat(v)) + 1e-12
        lam_new = float(num / den)
        if abs(lam_new - lam) < tol * (1.0 + abs(lam)):
            lam = lam_new
            break
        lam = lam_new
    return lam


def top_eigenpair(
    loss_fn,
    params: list[Tensor],
    data_batch,
    power_iters: int = 20,
) -> tuple[float, list[Tensor]]:
    """Largest eigenvalue and eigenvector of the Hessian."""
    v = [torch.randn_like(p) for p in params]
    lam = 0.0
    for _ in range(power_iters):
        vn = _flat(v).norm()
        v = [vv / vn for vv in v]
        loss = loss_fn(data_batch)
        Hv = hvp(loss, params, v)
        num = _flat(Hv).dot(_flat(v))
        lam = float(num / (_flat(v).dot(_flat(v)) + 1e-12))
    return lam, v
