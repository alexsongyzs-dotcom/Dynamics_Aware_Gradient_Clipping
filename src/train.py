"""Training entry point with dynamical diagnostics.

Runs a single training run with configurable model, dataset, optimizer,
clipping policy, and dynamical logging. Supports Hessian measurements at
selected checkpoints.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.clipping import DynamicsAwareClipping, clipping_coefficient, clipping_indicator, clipping_intensity
from src.data import build_loaders
from src.dynamics import gradient_alignment
from src.hessian import top_eigenvalue
from src.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_grads(model: nn.Module) -> Tensor:
    parts = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def _apply_coefficient(model: nn.Module, coeff: float) -> None:
    if coeff >= 1.0:
        return
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(coeff)


def train_run(cfg: dict, verbose: bool = False) -> dict:
    """Train one configuration; return diagnostics and metrics.

    cfg keys: model, dataset, data_dir, epochs, batch_size, lr, momentum,
    weight_decay, clipping (dict), seed, device, log_every, hessian_epochs,
    projection_dim (random projections of gradients).
    verbose: print per-epoch progress for live monitoring.
    """
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg["seed"])
    num_classes = 10 if cfg["dataset"] != "cifar10" else 10

    model = build_model(cfg["model"], num_classes=num_classes).to(device)
    train_loader, test_loader = build_loaders(
        cfg["dataset"],
        cfg.get("data_dir", "data"),
        cfg.get("batch_size", 256),
        num_workers=cfg.get("num_workers", 2),
        download=True,
    )

    if cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0)
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=cfg.get("momentum", 0.0),
            weight_decay=cfg.get("weight_decay", 0.0),
        )

    loss_fn = nn.CrossEntropyLoss()
    clip_cfg = cfg["clipping"]
    clip_name = clip_cfg.get("name", "none")

    dagc: DynamicsAwareClipping | None = None
    fixed_threshold: float | None = clip_cfg.get("threshold") if clip_name == "fixed" else None
    if clip_name == "dagc":
        dagc = DynamicsAwareClipping(
            gamma=clip_cfg.get("gamma", 0.05),
            beta=clip_cfg.get("beta", 0.9),
            relax=clip_cfg.get("relax", 0.3),
            init_c=clip_cfg.get("init_c", 1.0),
        )
        fixed_threshold = None

    # diagnostics
    log: dict[str, list] = {
        "loss": [], "grad_norm": [], "coeff": [], "update_norm": [],
        "alignment": [], "exposure": [], "signed": [], "threshold": [],
    }
    proj_dim = cfg.get("projection_dim", 4)
    # random projection basis for gradient directions (fixed across run)
    model_d = sum(p.numel() for p in model.parameters())
    proj = torch.randn(proj_dim, model_d, device=device) / (model_d ** 0.5)
    grad_proj: list[np.ndarray] = []
    switch_count = 0
    s_prev: float | None = None

    def hessian_loss_fn(batch) -> Tensor:
        xb, yb = batch
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        return loss_fn(out, yb)

    g_prev: Tensor | None = None
    steps = 0
    hessian_epochs = set(cfg.get("hessian_epochs", []))
    hessian_evals: list[tuple[int, float]] = []

    model.train()
    for epoch in range(cfg["epochs"]):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()

            gcat = _flatten_grads(model)
            gn = float(torch.norm(gcat))
            a = 1.0 if g_prev is None else gradient_alignment(gcat, g_prev)
            g_prev = gcat

            # clipping policy
            if clip_name == "none":
                coeff = 1.0
                c_used = float("inf")
            elif clip_name == "fixed":
                coeff = float(clipping_coefficient(torch.tensor(gn), fixed_threshold))
                c_used = fixed_threshold
            else:  # dagc
                assert dagc is not None
                dagc.update(gn, a)
                c_used = dagc.c
                coeff = dagc.clip_coefficient(gn)

            _apply_coefficient(model, coeff)
            optimizer.step()

            e = float(clipping_indicator(torch.tensor(gn), c_used if c_used != float("inf") else float("inf") + 1))
            upd_norm = float(torch.norm(torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]))) if coeff < 1 else gn * coeff
            s_t = gn / (c_used + 1e-12) - 1.0
            if s_prev is not None and s_t * s_prev < 0:
                switch_count += 1
            s_prev = s_t

            log["loss"].append(float(loss.item()))
            log["grad_norm"].append(gn)
            log["coeff"].append(coeff)
            log["update_norm"].append(upd_norm)
            log["alignment"].append(a)
            log["exposure"].append(dagc.E if dagc is not None else e)
            log["signed"].append(s_t)
            log["threshold"].append(c_used if np.isfinite(c_used) else np.nan)

            gp = (gcat.reshape(1, -1) @ proj.t()).reshape(-1).cpu().numpy()
            grad_proj.append(gp)

            steps += 1

        # Hessian top eigenvalue at checkpoint epochs
        if epoch in hessian_epochs:
            batch = next(iter(train_loader))
            lam = top_eigenvalue(hessian_loss_fn, [p for p in model.parameters() if p.requires_grad], batch, power_iters=12)
            hessian_evals.append((epoch, lam))

        if verbose:
            mean_loss = float(np.mean(log["loss"][-len(train_loader):]))
            print(f"  epoch {epoch + 1}/{cfg['epochs']}  loss {mean_loss:.4f}  "
                  f"steps {steps}", flush=True)

    # evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    test_acc = correct / max(total, 1)

    # aggregate diagnostics
    gp_arr = np.array(grad_proj)
    from src.oscillation import dominant_frequency, one_step_alignment, two_step_alignment

    c1 = one_step_alignment(gp_arr)
    c2 = two_step_alignment(gp_arr)

    summary = {
        "policy": clip_name,
        "test_acc": test_acc,
        "final_loss": log["loss"][-1],
        "mean_loss": float(np.mean(log["loss"][-200:])),
        "f_clip": float(np.mean([1.0 if np.isfinite(t) and gn > t else 0.0 for gn, t in zip(log["grad_norm"], log["threshold"])])) if clip_name != "none" else 0.0,
        "i_clip": float(np.mean(np.clip(1.0 - np.array(log["threshold"]) / (np.array(log["grad_norm"]) + 1e-12), 0, None))) if clip_name != "none" else 0.0,
        "n_switch": switch_count,
        "mean_c1": float(np.mean(c1)),
        "mean_c2": float(np.mean(c2)),
        "loss_dom_freq": dominant_frequency(log["loss"]),
        "gn_dom_freq": dominant_frequency(log["grad_norm"]),
        "hessian_evals": hessian_evals,
        "max_grad_norm": float(np.max(log["grad_norm"])),
        "median_grad_norm": float(np.median(log["grad_norm"])),
        "log": log,
    }
    return summary
