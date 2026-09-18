"""Create compact figures from A3/A4 benchmark summaries."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def a3_plot(rows: list[dict]) -> None:
    labels = [r["condition"].replace("_", "\n") for r in rows]
    means = [100 * float(r["test_acc_mean"]) for r in rows]
    cis = [100 * float(r["test_acc_ci95"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, means, yerr=cis, capsize=4, color=["#c23b22"] + ["#5c7ea4"] * (len(rows) - 1))
    ax.bar_label(bars, labels=[f"{x:.2f}" for x in means], padding=3, fontsize=9)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("A3: DAGC component ablation (5 seeds)")
    ax.set_ylim(min(means) - 3, max(means) + 3)
    fig.tight_layout()
    out = ROOT / "results" / "a3_ablation_small" / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "a3_ablation_accuracy.png", dpi=180)
    plt.close(fig)


def a4_plot(rows: list[dict]) -> None:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["model"], row["optimizer"]), []).append(row)
    policies = ["none", "fixed", "agc", "dagc"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    for ax, ((dataset, model, optimizer), values) in zip(axes.flat, groups.items()):
        ordered = {r["policy"]: r for r in values}
        means = [100 * float(ordered[p]["test_acc_mean"]) for p in policies]
        cis = [100 * float(ordered[p]["test_acc_ci95"]) for p in policies]
        bars = ax.bar(policies, means, yerr=cis, capsize=3, color=["#777777", "#4c78a8", "#72b7b2", "#c23b22"])
        ax.bar_label(bars, labels=[f"{x:.1f}" for x in means], padding=2, fontsize=8)
        ax.set_title(f"{dataset} | {model} | {optimizer}")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_ylim(min(means) - 4, max(means) + 3)
    fig.suptitle("A4: multi-seed diagnostic benchmark (5 seeds per policy)", y=1.01)
    fig.tight_layout()
    out = ROOT / "results" / "a4_multiseed_small" / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "a4_accuracy_by_setting.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    a3_plot(read_csv(ROOT / "results" / "a3_ablation_small" / "summary.csv"))
    a4_plot(read_csv(ROOT / "results" / "a4_multiseed_small" / "summary.csv"))


if __name__ == "__main__":
    main()
