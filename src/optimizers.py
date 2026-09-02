"""Reference optimizer baselines used by the clipping-placement study."""

from __future__ import annotations

import torch


class StableAdamW(torch.optim.Optimizer):
    """AdamW with Adafactor-style per-tensor update clipping.

    The tensor learning rate is divided by ``max(1, RMS(g^2 / v_hat))``.
    This deliberately simple implementation favors inspectability over fused
    performance and is intended as the StableAdamW baseline.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0 or eps < 0.0 or weight_decay < 0.0:
            raise ValueError("lr, eps, and weight_decay must be non-negative")
        if not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("betas must be in [0, 1)")
        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("StableAdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                first = state["exp_avg"]
                second = state["exp_avg_sq"]
                first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

                bias1 = 1.0 - beta1**step
                bias2 = 1.0 - beta2**step
                second_hat = second / bias2
                rms = torch.sqrt(torch.mean(gradient.square() / (second_hat + group["eps"])))
                scale = max(1.0, float(rms))
                tensor_lr = group["lr"] / scale
                if group["weight_decay"]:
                    parameter.mul_(1.0 - tensor_lr * group["weight_decay"])
                denominator = second_hat.sqrt().add_(group["eps"])
                parameter.addcdiv_(first / bias1, denominator, value=-tensor_lr)
                state["update_rms"] = float(rms)
                state["update_clip_scale"] = scale
        return loss
