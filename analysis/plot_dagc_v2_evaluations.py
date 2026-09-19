"""Create paper-ready summaries for the held-out DAGC-V2 evaluations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "dagc_v2"

CONDITIONS = [
    ("FashionMNIST + CNN\nSGD, lr=0.4", ROOT / "results" / "dagc_v2_evaluation_lr0.4" / "raw_runs.csv"),
    ("FashionMNIST + CNN\nSGD, lr=0.8", ROOT / "results" / "dagc_v2_evaluation" / "raw_runs.csv"),
    ("FashionMNIST + MLP\nSGD, lr=0.8", ROOT / "results" / "mlp_evaluation_lr0.8" / "raw_runs.csv"),
]
ORDER = ["fixed_tuned", "agc_default", "dagc_v1", "dagc_v2"]
LABELS = {"fixed_tuned": "Fixed", "agc_default": "AGC", "dagc_v1": "DAGC-V1", "dagc_v2": "DAGC-V2"}
COLORS = {"fixed_tuned": "#8c8c8c", "agc_default": "#4c78a8", "dagc_v1": "#e45756", "dagc_v2": "#54a24b"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for condition, path in CONDITIONS:
        df = pd.read_csv(path)
        for policy in ORDER:
            values = df.loc[df.policy == policy, "test_acc"].to_numpy() * 100
            records.append({
                "condition": condition.replace("\n", " "), "policy": policy, "method": LABELS[policy],
                "mean_accuracy": values.mean(), "ci95": 1.96 * values.std(ddof=1) / np.sqrt(len(values)),
                "n": len(values), "divergence_rate": df.loc[df.policy == policy, "diverged"].mean(),
            })
    summary = pd.DataFrame(records)
    summary.to_csv(OUT / "heldout_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.65), sharey=True)
    for ax, (condition, _) in zip(axes, CONDITIONS):
        s = summary[summary.condition == condition.replace("\n", " ")].set_index("policy").loc[ORDER]
        x = np.arange(len(ORDER))
        ax.bar(x, s.mean_accuracy, yerr=s.ci95, capsize=3,
               color=[COLORS[p] for p in ORDER], edgecolor="white", linewidth=0.8)
        for idx, value in enumerate(s.mean_accuracy):
            ax.text(idx, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(condition, fontsize=10)
        ax.set_xticks(x, [LABELS[p] for p in ORDER], rotation=24, ha="right")
        ax.set_ylim(70, 91)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Test accuracy (%)")
    fig.text(0.5, 0.01, "Error bars: 95% CI across 8 held-out paired seeds", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "heldout_accuracy.pdf", bbox_inches="tight")
    fig.savefig(OUT / "heldout_accuracy.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
