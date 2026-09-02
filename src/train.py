"""Training engine for measurement, replay, and causal branch experiments.

Nothing in this module executes on import. Linux runs should invoke
``python -m src.train --config <yaml>`` after the environment and data are
prepared.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from src.checkpointing import build_checkpoint, load_checkpoint, save_checkpoint
from src.clipping import (
    AdaGCPolicy,
    ClipDecision,
    ClippingPlacement,
    PolicyContext,
    apply_agc_,
    apply_post_update_decision_,
    build_policy,
    flatten_gradients,
    gradient_norm,
    parameter_update_norm,
    scale_gradients_,
    snapshot_parameters,
    trainable_parameters,
)
from src.configuration import load_config, parse_override, set_dotted
from src.data import build_data_bundle
from src.dynamics import (
    ClippingEpisodeTracker,
    DynamicsHistory,
    HysteresisConfig,
    gradient_alignment,
    optimizer_state_metrics,
)
from src.hessian import top_eigenvalue
from src.models import build_model
from src.optimizers import StableAdamW
from src.projection import GradientSketcher
from src.records import RunRecorder, config_hash


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def _mapping(value: Any, default_name: str) -> dict[str, Any]:
    if value is None:
        return {"name": default_name}
    if isinstance(value, str):
        return {"name": value}
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"expected a string or mapping, got {type(value).__name__}")


def _section(config: dict, singular: str, plural: str, default_name: str) -> dict[str, Any]:
    return _mapping(config.get(singular, config.get(plural)), default_name)


def _device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _optimizer(parameters, config: dict) -> torch.optim.Optimizer:
    name = str(config.get("name", "sgd")).lower()
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(config.get("lr", 0.1)),
            momentum=float(config.get("momentum", 0.0)),
            weight_decay=float(config.get("weight_decay", 0.0)),
            nesterov=bool(config.get("nesterov", False)),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(config.get("lr", 1e-3)),
            betas=tuple(float(value) for value in config.get("betas", (0.9, 0.999))),
            eps=float(config.get("eps", 1e-8)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    if name == "stable_adamw":
        return StableAdamW(
            parameters,
            lr=float(config.get("lr", 1e-3)),
            betas=tuple(float(value) for value in config.get("betas", (0.9, 0.999))),
            eps=float(config.get("eps", 1e-8)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    raise ValueError(f"unknown optimizer: {name}")


def _scheduler(
    optimizer: torch.optim.Optimizer, config: dict, epochs: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = str(config.get("schedule", "none")).lower()
    if name in {"none", "constant"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(value) for value in config.get("milestones", (60, 80))],
            gamma=float(config.get("gamma", 0.1)),
        )
    raise ValueError(f"unknown learning-rate schedule: {name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader, loss_fn: nn.Module, device: torch.device) -> dict[str, float]:
    training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        total_loss += float(loss_fn(logits, targets)) * targets.numel()
        correct += int((logits.argmax(dim=1) == targets).sum())
        count += int(targets.numel())
    model.train(training)
    return {"loss": total_loss / max(count, 1), "accuracy": correct / max(count, 1)}


def train_run(config: dict, verbose: bool = False) -> dict[str, Any]:
    """Execute one declared run and return a compact summary.

    This function is intentionally explicit about intervention placement and
    checkpoint state. It is not called during repository construction.
    """

    cfg = copy.deepcopy(config)
    seed = int(cfg.get("seed", 0))
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    device = _device(str(cfg.get("device", "cuda")))

    dataset_cfg = _section(cfg, "dataset", "datasets", "cifar10")
    model_cfg = _section(cfg, "model", "models", "resnet18")
    optimizer_cfg = _section(cfg, "optimizer", "optimizers", "sgd")
    clipping_cfg = _mapping(cfg.get("clipping"), "none")
    logging_cfg = dict(cfg.get("logging", {}))
    checkpoint_cfg = dict(cfg.get("checkpointing", {}))

    epochs = int(cfg.get("epochs", 1))
    bundle = build_data_bundle(
        dataset=str(dataset_cfg.get("name", "cifar10")),
        data_dir=str(dataset_cfg.get("data_dir", cfg.get("data_dir", "data"))),
        batch_size=int(cfg.get("batch_size", 128)),
        num_workers=int(cfg.get("num_workers", 0)),
        seed=seed,
        probe_size=int(logging_cfg.get("probe_size", 1024)),
        download=bool(dataset_cfg.get("download", False)),
        pin_memory=bool(cfg.get("pin_memory", True)),
    )
    image_size = 28 if dataset_cfg.get("name") in {"mnist", "fashion_mnist"} else 32
    model_kwargs = {key: value for key, value in model_cfg.items() if key not in {"name", "num_classes", "pretrained"}}
    model = build_model(
        str(model_cfg.get("name", "resnet18")),
        num_classes=bundle.num_classes,
        input_channels=bundle.input_channels,
        image_size=image_size,
        **model_kwargs,
    ).to(device)
    parameters = trainable_parameters(model)
    optimizer = _optimizer(parameters, optimizer_cfg)
    scheduler = _scheduler(optimizer, optimizer_cfg, epochs)
    loss_fn = nn.CrossEntropyLoss()

    policy = build_policy(clipping_cfg)
    placement = ClippingPlacement(clipping_cfg.get("placement", "pre_moment"))
    if clipping_cfg.get("name") in {"agc", "adagc"} and placement is not ClippingPlacement.PRE_MOMENT:
        raise ValueError("AGC and AdaGC are defined only for pre-moment placement")

    episode_cfg = dict(logging_cfg.get("episode", {}))
    episode_tracker = ClippingEpisodeTracker(
        HysteresisConfig(
            on_margin=float(episode_cfg.get("on_margin", 0.05)),
            off_margin=float(episode_cfg.get("off_margin", 0.05)),
            min_dwell_steps=int(episode_cfg.get("min_dwell_steps", 3)),
        )
    )
    history_tracker = DynamicsHistory(
        window=int(logging_cfg.get("history_window", 100)),
        beta=float(logging_cfg.get("exposure_beta", 0.95)),
    )

    total_dimension = sum(parameter.numel() for parameter in parameters)
    sketcher = GradientSketcher(
        total_dimension,
        int(logging_cfg.get("gradient_sketch_size", 64)),
        seed + 101,
        device,
    )
    previous_gradient_sketch: torch.Tensor | None = None
    previous_probe_gradient_sketch: torch.Tensor | None = None
    fixed_probe_batch = next(iter(bundle.probe_loader))

    run_id = str(cfg.get("run_id", f"run-{config_hash(cfg)}-seed{seed}"))
    start_epoch = 0
    global_step = 0
    resume_path = checkpoint_cfg.get("resume_from")
    recorder = None
    if cfg.get("output_dir"):
        recorder = RunRecorder(
            cfg["output_dir"],
            run_id,
            cfg,
            append_existing=bool(resume_path),
        )
    if resume_path:
        position, payload = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            policy=policy,
            sampler=bundle.train_sampler,
            episode_tracker=episode_tracker,
            history_tracker=history_tracker,
            map_location=device,
        )
        start_epoch = position.epoch
        global_step = position.global_step
        previous = payload.get("metadata", {}).get("previous_gradient_sketch")
        if isinstance(previous, torch.Tensor):
            previous_gradient_sketch = previous.to(device)
        previous_probe = payload.get("metadata", {}).get("previous_probe_gradient_sketch")
        if isinstance(previous_probe, torch.Tensor):
            previous_probe_gradient_sketch = previous_probe.to(device)

    rows: list[dict[str, Any]] = []
    hessian_records: list[dict[str, float | int | str]] = []
    hessian_epochs = {int(value) for value in logging_cfg.get("hessian_checkpoints", [])}
    checkpoint_steps = {int(value) for value in checkpoint_cfg.get("steps", [])}
    checkpoint_every_epochs = int(checkpoint_cfg.get("every_epochs", 0))
    checkpoint_dir = Path(checkpoint_cfg.get("directory", "results/checkpoints"))
    failed = False
    failure_reason = ""

    model.train()
    for epoch in range(start_epoch, epochs):
        for inputs, targets in bundle.train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            if not torch.isfinite(loss):
                failed = True
                failure_reason = "non_finite_loss"
                break
            loss.backward()

            raw_gradient = flatten_gradients(parameters)
            raw_grad_norm = float(torch.linalg.vector_norm(raw_gradient))
            gradient_sketch = sketcher(raw_gradient)
            alignment = (
                math.nan
                if previous_gradient_sketch is None
                else gradient_alignment(gradient_sketch, previous_gradient_sketch)
            )
            previous_gradient_sketch = gradient_sketch.detach().clone()
            state_metrics = optimizer_state_metrics(parameters, optimizer)
            mismatch = state_metrics["adam_second_moment_mismatch"]
            context = PolicyContext(
                step=global_step,
                grad_norm=raw_grad_norm,
                loss=float(loss.detach()),
                alignment=alignment,
                moment_mismatch=mismatch,
            )
            decision = policy.decide(context)

            before = snapshot_parameters(parameters)
            if clipping_cfg.get("name") == "agc":
                coefficient = apply_agc_(
                    parameters,
                    clip_value=float(clipping_cfg.get("clip_value", 0.01)),
                    eps=float(clipping_cfg.get("eps", 1e-3)),
                )
                decision = ClipDecision(
                    coefficient=coefficient,
                    threshold=coefficient * raw_grad_norm if coefficient < 1.0 else math.inf,
                    active=coefficient < 1.0,
                    source="agc",
                )
            elif clipping_cfg.get("name") == "adagc":
                if not isinstance(policy, AdaGCPolicy):
                    raise TypeError("AdaGC configuration did not build an AdaGC policy")
                coefficient = policy.apply_(parameters, raw_grad_norm)
                decision = ClipDecision(
                    coefficient=coefficient,
                    threshold=coefficient * raw_grad_norm if coefficient < 1.0 else math.inf,
                    active=coefficient < 1.0,
                    source="adagc",
                )
            elif placement is ClippingPlacement.PRE_MOMENT:
                scale_gradients_(parameters, decision.coefficient)

            applied_grad_norm = gradient_norm(parameters)
            optimizer.step()
            proposed_update_norm = parameter_update_norm(parameters, before)
            if placement is ClippingPlacement.POST_UPDATE:
                update_application = apply_post_update_decision_(
                    parameters,
                    before,
                    decision,
                    allow_enlarge=bool(clipping_cfg.get("allow_enlarge", False)),
                )
                applied_update_norm = update_application.applied_update_norm
                applied_coefficient = update_application.coefficient
            else:
                applied_update_norm = proposed_update_norm
                applied_coefficient = decision.coefficient
            applied_active = applied_coefficient < 1.0 - 1e-12

            threshold = decision.threshold
            if math.isfinite(threshold) and threshold > 0.0:
                q = raw_grad_norm / threshold
            elif decision.active and applied_coefficient > 0.0:
                q = 1.0 / applied_coefficient
            else:
                q = 0.0
            episode = episode_tracker.update(q, global_step)
            history = history_tracker.update(episode, 1.0 - min(1.0, applied_coefficient))
            learning_rate = float(optimizer.param_groups[0]["lr"])
            row: dict[str, Any] = {
                **decision.to_dict(),
                "epoch": epoch,
                "step": global_step,
                "loss": float(loss.detach()),
                "learning_rate": learning_rate,
                "raw_grad_norm": raw_grad_norm,
                "applied_grad_norm": applied_grad_norm,
                "proposed_update_norm": proposed_update_norm,
                "applied_update_norm": applied_update_norm,
                "coefficient": applied_coefficient,
                "active": applied_active,
                "threshold": threshold,
                "clipping_coordinate": q,
                "placement": placement.value,
                "gradient_alignment": alignment,
                **state_metrics,
                **episode.to_dict(),
                **history.to_dict(),
            }
            if bool(logging_cfg.get("save_gradient_sketch", False)):
                row["gradient_sketch"] = gradient_sketch.detach().cpu()
            rows.append(row)
            if recorder is not None:
                recorder.append_step(row)

            global_step += 1
            if global_step in checkpoint_steps:
                payload = build_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    policy=policy,
                    sampler=bundle.train_sampler,
                    episode_tracker=episode_tracker,
                    history_tracker=history_tracker,
                    epoch=epoch,
                    global_step=global_step,
                    config=cfg,
                    metadata={
                        "previous_gradient_sketch": previous_gradient_sketch.detach().cpu(),
                        "previous_probe_gradient_sketch": (
                            None
                            if previous_probe_gradient_sketch is None
                            else previous_probe_gradient_sketch.detach().cpu()
                        ),
                    },
                )
                save_checkpoint(checkpoint_dir / f"{run_id}.step-{global_step}.pt", payload)

        if failed:
            break

        if epoch in hessian_epochs:
            training = model.training
            model.eval()

            def probe_loss(batch) -> torch.Tensor:
                probe_inputs, probe_targets = batch
                return loss_fn(model(probe_inputs.to(device)), probe_targets.to(device))

            probe_value = probe_loss(fixed_probe_batch)
            probe_gradients = torch.autograd.grad(
                probe_value,
                parameters,
                retain_graph=False,
                allow_unused=True,
            )
            probe_flat = torch.cat(
                [
                    (torch.zeros_like(parameter) if gradient is None else gradient)
                    .detach()
                    .reshape(-1)
                    for parameter, gradient in zip(parameters, probe_gradients)
                ]
            )
            probe_gradient_sketch = sketcher(probe_flat)
            probe_alignment = (
                math.nan
                if previous_probe_gradient_sketch is None
                else gradient_alignment(probe_gradient_sketch, previous_probe_gradient_sketch)
            )
            previous_probe_gradient_sketch = probe_gradient_sketch.detach().clone()
            eigenvalue = top_eigenvalue(
                probe_loss,
                parameters,
                fixed_probe_batch,
                power_iters=int(logging_cfg.get("hessian_power_iterations", 20)),
            )
            hessian_records.append(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "top_eigenvalue": eigenvalue,
                    "probe_loss": float(probe_value.detach()),
                    "probe_gradient_norm": float(torch.linalg.vector_norm(probe_flat)),
                    "probe_gradient_alignment": probe_alignment,
                    "scope": "fixed_probe_batch",
                }
            )
            model.train(training)

        if scheduler is not None:
            scheduler.step()

        if checkpoint_every_epochs > 0 and (epoch + 1) % checkpoint_every_epochs == 0:
            payload = build_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                policy=policy,
                sampler=bundle.train_sampler,
                episode_tracker=episode_tracker,
                history_tracker=history_tracker,
                epoch=epoch + 1,
                global_step=global_step,
                config=cfg,
                metadata={
                    "previous_gradient_sketch": previous_gradient_sketch.detach().cpu(),
                    "previous_probe_gradient_sketch": (
                        None
                        if previous_probe_gradient_sketch is None
                        else previous_probe_gradient_sketch.detach().cpu()
                    ),
                },
            )
            save_checkpoint(checkpoint_dir / f"{run_id}.epoch-{epoch + 1}.pt", payload)

        if verbose and rows:
            epoch_losses = [row["loss"] for row in rows if row["epoch"] == epoch]
            print(
                f"epoch {epoch + 1}/{epochs} loss={np.mean(epoch_losses):.5f} step={global_step}",
                flush=True,
            )

    evaluation = evaluate(model, bundle.test_loader, loss_fn, device) if not failed else {
        "loss": math.nan,
        "accuracy": math.nan,
    }
    active = [float(row["active"]) for row in rows]
    summary: dict[str, Any] = {
        "run_id": run_id,
        "config_hash": config_hash(cfg),
        "seed": seed,
        "dataset": str(dataset_cfg.get("name", "cifar10")),
        "model": str(model_cfg.get("name", "resnet18")),
        "optimizer": str(optimizer_cfg.get("name", "sgd")),
        "learning_rate": float(optimizer_cfg.get("lr", 0.1)),
        "clipping_policy": str(clipping_cfg.get("name", "none")),
        "clipping_threshold": clipping_cfg.get("threshold"),
        "placement": placement.value,
        "failed": failed,
        "failure_reason": failure_reason,
        "completed_steps": global_step,
        "test_loss": evaluation["loss"],
        "test_accuracy": evaluation["accuracy"],
        "final_train_loss": rows[-1]["loss"] if rows else math.nan,
        "clipping_frequency": float(np.mean(active)) if active else 0.0,
        "mean_clipping_intensity": (
            float(np.mean([1.0 - float(row["coefficient"]) for row in rows])) if rows else 0.0
        ),
        "episode_transitions": episode_tracker.switch_count,
        "episode_count": episode_tracker.episode_id,
        "hessian_records": hessian_records,
    }
    if bool(cfg.get("return_step_rows", False)):
        summary["rows"] = rows
    if recorder is not None:
        recorder.write_summary(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one declared clipping experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a nested value, for example optimizer.lr=0.2",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for override in args.set:
        key, value = parse_override(override)
        set_dotted(config, key, value)
    train_run(config, verbose=args.verbose)


if __name__ == "__main__":
    main()
