"""Training entry point with dynamical diagnostics.

Runs a single training run with configurable model, dataset, optimizer,
clipping policy, and lightweight dynamical logging.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.clipping import (
    AdaGC,
    DynamicsAwareClipping,
    ZClip,
    adaptive_gradient_clip_,
    clipping_coefficient,
    finite_mask,
    nanmean_or_nan,
)
from src.data import build_loaders
from src.dynamics import gradient_alignment
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
    weight_decay, clipping (dict), seed, device, log_every, projection_dim
    (random projections of gradients).
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
        train_size=cfg.get("train_size"),
        test_size=cfg.get("test_size"),
        subset_seed=cfg["seed"],
        loader_seed=cfg["seed"],
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
    zclip: ZClip | None = None
    adagc: AdaGC | None = None
    fixed_threshold: float | None = clip_cfg.get("threshold") if clip_name == "fixed" else None
    if clip_name == "dagc":
        dagc = DynamicsAwareClipping(
            gamma=clip_cfg.get("gamma", 0.05),
            beta=clip_cfg.get("beta", 0.9),
            relax=clip_cfg.get("relax", 0.3),
            osc_weight=clip_cfg.get("osc_weight", 3.0),
            c_min_scale=clip_cfg.get("c_min_scale", 0.1),
            c_max_scale=clip_cfg.get("c_max_scale", 10.0),
            init_c=clip_cfg.get("init_c", 1.0),
            exposure_target=clip_cfg.get("exposure_target"),
        )
        fixed_threshold = None
    elif clip_name == "zclip":
        zclip = ZClip(
            alpha=clip_cfg.get("alpha", 0.97),
            z_thres=clip_cfg.get("z_thres", 2.5),
            warmup=clip_cfg.get("warmup", 25),
        )
    elif clip_name == "adagc":
        adagc = AdaGC(
            lambda_rel=clip_cfg.get("lambda_rel", 0.5),
            beta=clip_cfg.get("beta_adagc", 0.999),
            t_start=clip_cfg.get("t_start", 100),
            lambda_abs=clip_cfg.get("lambda_abs", 1.0),
        )

    # diagnostics
    log: dict[str, list] = {
        "loss": [], "grad_norm": [], "coeff": [], "update_norm": [],
        "alignment": [], "exposure": [], "intensity": [], "signed": [], "threshold": [],
    }
    proj_dim = cfg.get("projection_dim", 4)
    # random projection basis for gradient directions (fixed across run)
    model_d = sum(p.numel() for p in model.parameters())
    proj = torch.randn(proj_dim, model_d, device=device) / (model_d ** 0.5)
    grad_proj: list[np.ndarray] = []
    switch_count = 0
    s_prev: float | None = None

    g_prev: Tensor | None = None
    steps = 0
    norm_history = []
    rate_exposure = 0.0
    rate_threshold = clip_cfg.get("init_c", 0.5)
    # Optional diagnostic hook: called once per step with the model, returns a
    # flat dict of extra scalars to record in the run log.
    hook: Callable | None = cfg.get("hook")
    diverged = False
    divergence_step: int | None = None
    divergence_reason: str | None = None

    model.train()
    max_steps = cfg.get("max_steps")
    for epoch in range(cfg["epochs"]):
        for x, y in train_loader:
            for change in cfg.get("lr_changes", []):
                if steps == change["step"]:
                    for group in optimizer.param_groups:
                        group["lr"] = change["lr"]
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
                target = float("inf")
            elif clip_name == "fixed":
                coeff = float(clipping_coefficient(torch.tensor(gn), fixed_threshold))
                target = fixed_threshold
                _apply_coefficient(model, coeff)
            elif clip_name in ("autoclip", "rate_tracking", "replay"):
                if clip_name == "autoclip":
                    norm_history.append(gn)
                    target = float(np.quantile(norm_history, clip_cfg.get("quantile", 0.1)))
                elif clip_name == "replay":
                    schedule = clip_cfg["schedule"]
                    target = float(schedule[min(steps, len(schedule) - 1)])
                else:
                    target = rate_threshold
                    event = float(gn > target)
                    rate_exposure = 0.9 * rate_exposure + 0.1 * event
                    rate_threshold *= np.exp(0.05 * np.tanh(0.3 * (rate_exposure - 0.1)))
                coeff = min(1.0, target / (gn + 1e-12))
                _apply_coefficient(model, coeff)
            elif clip_name == "agc":
                adaptive_gradient_clip_(
                    model.parameters(),
                    clip_value=clip_cfg.get("clip_value", 0.01),
                    eps=clip_cfg.get("eps", 1e-3),
                )
                clipped_norm = float(torch.norm(_flatten_grads(model)))
                coeff = clipped_norm / (gn + 1e-12)
                target = coeff * gn
            elif clip_name == "dagc":
                assert dagc is not None
                target = dagc.c
                coeff = dagc.clip_coefficient(gn)
                _apply_coefficient(model, coeff)
            elif clip_name == "zclip":
                assert zclip is not None
                coeff = zclip.clip_coefficient(gn) if np.isfinite(gn) else 1.0
                target = zclip.target if zclip.target is not None else float("inf")
                if coeff != 1.0 and np.isfinite(coeff):
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(coeff)
            elif clip_name == "adagc":
                assert adagc is not None
                adagc.step(model.parameters(), steps + 1)
                clipped_norm = float(torch.norm(_flatten_grads(model)))
                coeff = clipped_norm / (gn + 1e-12)
                target = coeff * gn
            else:
                raise ValueError(f"unknown clipping policy: {clip_name}")

            before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
            optimizer.step()
            update_sq = sum(float((p.detach() - old).square().sum()) for p, old in zip((p for p in model.parameters() if p.requires_grad), before))
            upd_norm = update_sq ** 0.5

            # DAGC applies c_t to the current update, then computes c_{t+1}.
            if dagc is not None:
                dagc.update(gn, a)
            if zclip is not None:
                zclip.update(gn)

            # Exposure and intensity are derived uniformly from the effective
            # target norm after clipping.  This is the only definition that
            # works across all policies: per-unit methods (AGC, AdaGC) have no
            # single scalar threshold, and ZClip may scale a step UP, which
            # must not be counted as clipping.
            e = float(np.isfinite(gn) and gn > target + 1e-12)
            intensity = float(max(0.0, 1.0 - target / (gn + 1e-12))) if np.isfinite(gn) else 0.0
            s_t = (gn / target - 1.0) if np.isfinite(target) and target > 0 else float("nan")
            if s_prev is not None and np.isfinite(s_t) and np.isfinite(s_prev) and s_t * s_prev < 0:
                switch_count += 1
            s_prev = s_t

            loss_value = float(loss.item())
            log["loss"].append(loss_value)
            log["grad_norm"].append(gn)
            log["coeff"].append(coeff)
            log["update_norm"].append(upd_norm)
            log["alignment"].append(a)
            log["exposure"].append(e)
            log["intensity"].append(intensity)
            log["signed"].append(s_t)
            log["threshold"].append(target if np.isfinite(target) else np.nan)

            if hook is not None:
                for key, value in hook(model).items():
                    log.setdefault(key, []).append(value)

            # Divergence: a non-finite loss/gradient, the loss leaving any
            # plausible range for the cross-entropy on these tasks, or a
            # single-step loss jump too large to be ordinary progress.  These
            # are lower bounds on failure; a run that merely degrades without
            # tripping one is not counted.
            prev_loss = log["loss"][-2] if len(log["loss"]) > 1 else None
            if not np.isfinite(loss_value) or not np.isfinite(gn):
                diverged = True
                divergence_step = steps + 1
                if divergence_reason is None:
                    divergence_reason = "nonfinite"
            elif loss_value > cfg.get("divergence_loss", 100.0):
                diverged = True
                divergence_step = steps + 1
                if divergence_reason is None:
                    divergence_reason = "loss_explosion"
            elif (
                prev_loss is not None
                and np.isfinite(prev_loss)
                and loss_value > max(cfg.get("divergence_jump", 50.0), 20.0 * prev_loss + 1.0)
            ):
                diverged = True
                divergence_step = steps + 1
                if divergence_reason is None:
                    divergence_reason = "loss_spike"

            gp = (gcat.reshape(1, -1) @ proj.t()).reshape(-1).cpu().numpy()
            grad_proj.append(gp)

            steps += 1

            if diverged:
                # Stop the run at the divergence point.  Continuing would only
                # waste compute on a model that has already blown up, and the
                # step at which it happened is the measurement we want.
                break

            if max_steps is not None and steps >= max_steps:
                break

        if diverged or (max_steps is not None and steps >= max_steps):
            break

        if verbose:
            mean_loss = nanmean_or_nan(log["loss"][-len(train_loader):])
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

    loss_arr = np.asarray(log["loss"], dtype=float)
    gn_arr = np.asarray(log["grad_norm"], dtype=float)
    loss_finite = loss_arr[finite_mask(loss_arr)]
    gn_finite = gn_arr[finite_mask(gn_arr)]
    total_steps = cfg["epochs"] * len(train_loader)

    summary = {
        "policy": clip_name,
        "test_acc": float(test_acc),
        "final_loss": log["loss"][-1],
        # A diverged run writes NaN into the log; averaging without filtering
        # would turn every downstream summary into NaN.
        "mean_loss": nanmean_or_nan(loss_finite[-200:]),
        "f_clip": nanmean_or_nan(log["exposure"]),
        "i_clip": nanmean_or_nan(log["intensity"]),
        "n_switch": switch_count,
        "mean_c1": nanmean_or_nan(c1),
        "mean_c2": nanmean_or_nan(c2),
        "loss_dom_freq": dominant_frequency(loss_finite),
        "gn_dom_freq": dominant_frequency(gn_finite),
        "max_grad_norm": float(gn_finite.max()) if gn_finite.size else float("nan"),
        "median_grad_norm": float(np.median(gn_finite)) if gn_finite.size else float("nan"),
        "diverged": bool(diverged),
        "divergence_step": divergence_step,
        "divergence_reason": divergence_reason,
        "steps_completed": steps,
        "steps_planned": total_steps,
        "log": log,
    }
    return summary
