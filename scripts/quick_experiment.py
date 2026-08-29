"""Quick small-scale experiment: no clipping vs fixed vs DAGC.

Runs a short probe to estimate the gradient-norm scale, then compares three
clipping policies on a small CNN + FashionMNIST, prints a summary table and
saves diagnostic plots to results/figures/.

Usage:
    python scripts/quick_experiment.py [--epochs 10] [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path as P

sys.path.insert(0, str(P(__file__).resolve().parents[1]))

from src.train import train_run

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"


def base_cfg(seed: int, epochs: int) -> dict:
    return {
        "model": "small_cnn",
        "dataset": "fashion_mnist",
        "data_dir": "data",
        "epochs": epochs,
        "batch_size": 256,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "optimizer": "sgd",
        "seed": seed,
        "device": "cuda",
        "projection_dim": 4,
        "hessian_epochs": [0, max(1, epochs // 3), max(2, 2 * epochs // 3), epochs - 1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"== quick experiment: {args.epochs} epochs, seed {args.seed} ==")

    # ---- probe: no clipping, estimate gradient scale ----
    print("[probe] no clipping ...")
    probe_cfg = base_cfg(args.seed, min(3, args.epochs))
    probe_cfg["clipping"] = {"name": "none"}
    probe = train_run(probe_cfg)
    med_gn = probe["median_grad_norm"]
    print(f"  probe median grad norm = {med_gn:.4f}")

    c_fixed = float(med_gn)  # transition regime: clipping ~50% of steps
    print(f"  fixed threshold set to median = {c_fixed:.4f}")

    # ---- runs ----
    policies = [
        ("none", {"name": "none"}),
        ("fixed", {"name": "fixed", "threshold": c_fixed}),
        ("dagc", {"name": "dagc", "gamma": 0.05, "beta": 0.9, "init_c": c_fixed}),
    ]

    results = {}
    for name, clip in policies:
        print(f"[run] policy = {name} ...")
        cfg = base_cfg(args.seed, args.epochs)
        cfg["clipping"] = clip
        results[name] = train_run(cfg)

    # ---- summary table ----
    print()
    print(f"{'policy':<8} {'test_acc':>9} {'mean_loss':>9} {'F_clip':>7} {'I_clip':>7} "
          f"{'N_switch':>9} {'meanC1':>8} {'meanC2':>8} {'loss_freq':>9}")
    for name in ["none", "fixed", "dagc"]:
        r = results[name]
        print(f"{name:<8} {r['test_acc']*100:>8.2f}% {r['mean_loss']:>9.4f} "
              f"{r['f_clip']:>7.3f} {r['i_clip']:>7.3f} {r['n_switch']:>9d} "
              f"{r['mean_c1']:>8.3f} {r['mean_c2']:>8.3f} {r['loss_dom_freq']:>9.4f}")

    for name in ["none", "fixed", "dagc"]:
        r = results[name]
        he = r.get("hessian_evals", [])
        if he:
            s = ", ".join(f"e{e}:{lam:.1f}" for e, lam in he)
            print(f"  [{name}] hessian top eigenvalues: {s}")

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"none": "#888888", "fixed": "#1f77b4", "dagc": "#d62728"}
    for name in ["none", "fixed", "dagc"]:
        r = results[name]
        lg = r["log"]
        t = np.arange(len(lg["loss"]))
        axes[0, 0].plot(t, lg["loss"], label=name, color=colors[name], lw=0.8)
        axes[0, 1].plot(t, lg["grad_norm"], label=name, color=colors[name], lw=0.6)
        axes[1, 0].plot(t, lg["alignment"], label=name, color=colors[name], lw=0.6)
        axes[1, 1].plot(t, lg["threshold"], label=name, color=colors[name], lw=1.0)
    axes[0, 0].set_title("train loss"); axes[0, 0].set_yscale("log"); axes[0, 0].legend()
    axes[0, 1].set_title("gradient norm"); axes[0, 1].set_yscale("log"); axes[0, 1].legend()
    axes[1, 0].set_title("gradient alignment C1"); axes[1, 0].set_ylim(-1.05, 1.05); axes[1, 0].legend()
    axes[1, 1].set_title("clipping threshold c_t"); axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("step")
    fig.tight_layout()
    png = OUT / f"quick_experiment_ep{args.epochs}_seed{args.seed}.png"
    fig.savefig(png, dpi=130)
    print(f"\nplot saved: {png}")


if __name__ == "__main__":
    main()