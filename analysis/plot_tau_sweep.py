"""Confirmation plot: realized exposure does not track the reference level.

Loads the V2 development sweep over the reference exposure level ``tau`` and
draws two panels.

Left: accuracy against ``tau``.  If the controller worked by driving realized
exposure towards ``tau``, accuracy would depend on ``tau``.  It does not: the
three V2 points lie inside each other's intervals.  The V1 value is drawn at
``tau = 1``, which is exactly the V1 rule, to show the size of the effect that
the one-sign change produces by comparison.

Right: realized exposure against ``tau``.  The measured exposure is 1.000 for
every V2 run, so the curve is flat at the ceiling while the reference level
rises.  Plotting the reference level alongside makes the failure of the tracking
reading visible without needing any curve fitting.

Outputs: results/figures/dagc_v2/tau_sweep_accuracy.{pdf,png}
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "results" / "dagc_v2_development" / "raw_runs.csv"
OUT = ROOT / "results" / "figures" / "dagc_v2"

COLOR_V2 = "#54a24b"
COLOR_V1 = "#e45756"
COLOR_REF = "#4c78a8"
COLOR_BAND = "#54a24b"


def ci95(values: np.ndarray) -> float:
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def parse(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a numeric tau column; V1 is the tau = 1 special case."""
    def tau_of(label: str) -> float:
        if label == "dagc_v1":
            return 1.0
        match = re.search(r"target=([0-9.]+)", label)
        if match is None:
            return float("nan")
        return float(match.group(1))

    df = df.copy()
    df["tau"] = df["label"].map(tau_of)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = parse(pd.read_csv(DEV))
    accuracy = df["test_acc"] * 100

    sweep = (
        df[df.label.str.startswith("dagc_v2")]
        .groupby("tau")
        .agg(acc=("test_acc", lambda s: s.mean() * 100),
             acc_ci=("test_acc", lambda s: ci95(s * 100)),
             f_clip=("f_clip", "mean"),
             i_clip=("i_clip", "mean"),
             n=("test_acc", "size"))
        .reset_index()
        .sort_values("tau")
    )
    v1 = df[df.label == "dagc_v1"]
    v1_acc = float(v1["test_acc"].mean() * 100)
    v1_ci = ci95(v1["test_acc"] * 100)
    v1_fclip = float(v1["f_clip"].mean())

    spread = float(sweep["acc"].max() - sweep["acc"].min())
    band_lo, band_hi = float(sweep["acc"].min()), float(sweep["acc"].max())

    fig, (ax_acc, ax_exp) = plt.subplots(1, 2, figsize=(9.6, 3.7))

    # ---- left: accuracy vs tau -------------------------------------------------
    ax_acc.axhspan(band_lo, band_hi, color=COLOR_BAND, alpha=0.13, zorder=0,
                   label=f"V2 spread ({spread:.2f} pts)")
    # individual seeds, jittered by tau so no points are hidden
    rng = np.random.default_rng(0)
    seeds = df[df.label.str.startswith("dagc_v2")]
    ax_acc.scatter(seeds["tau"] + rng.uniform(-0.012, 0.012, len(seeds)),
                   seeds["test_acc"] * 100, s=14, color=COLOR_V2, alpha=0.45,
                   edgecolor="none", zorder=2, label="V2, individual seeds")
    ax_acc.errorbar(sweep["tau"], sweep["acc"], yerr=sweep["acc_ci"], marker="o",
                    markersize=6, capsize=3, linewidth=1.6, color=COLOR_V2, zorder=3,
                    label="V2, mean $\\pm$95% CI")
    ax_acc.errorbar([1.0], [v1_acc], yerr=[v1_ci], marker="s", markersize=6, capsize=3,
                    linewidth=0, elinewidth=1.4, color=COLOR_V1, zorder=3,
                    label="V1 ($\\tau=1$, one-sided)")
    ax_acc.annotate("one-sign change\n$\\rightarrow$ %+.1f pts" % (sweep["acc"].mean() - v1_acc),
                    xy=(0.955, 82.9), xytext=(0.86, 82.2), fontsize=8.5,
                    color=COLOR_V1, ha="center",
                    arrowprops=dict(arrowstyle="->", color=COLOR_V1, lw=1.1))
    ax_acc.annotate(f"{sweep['tau'].max() / sweep['tau'].min():.0f}$\\times$ change in $\\tau$ $\\rightarrow$ {spread:.2f} pts",
                    xy=(0.30, 86.0), xytext=(0.30, 83.7), fontsize=8.5,
                    color="#2f6b2a", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#2f6b2a", lw=1.1))
    ax_acc.set_xlabel("Reference exposure level $\\tau$")
    ax_acc.set_ylabel("Test accuracy (%)")
    ax_acc.set_title("Accuracy is insensitive to $\\tau$", fontsize=10, pad=12)
    ax_acc.set_xlim(0.02, 1.12)
    ax_acc.set_ylim(78.5, 89.2)
    ax_acc.grid(alpha=0.25)
    ax_acc.legend(fontsize=7.6, loc="lower left", framealpha=0.92)

    # ---- right: realized vs reference exposure ---------------------------------
    ax_exp.axhline(1.0, color=COLOR_V2, linewidth=1.8, zorder=3,
                   label="realized exposure (V2)")
    ax_exp.scatter(sweep["tau"], sweep["f_clip"], s=26, color=COLOR_V2, zorder=4,
                   edgecolor="white", linewidth=0.7)
    ts = np.linspace(0.05, 0.4, 100)
    ax_exp.plot(ts, ts, linestyle="--", linewidth=1.5, color=COLOR_REF, zorder=2,
                label="reference level $\\tau$")
    ax_exp.scatter([1.0], [v1_fclip], s=26, marker="s", color=COLOR_V1, zorder=4,
                   edgecolor="white", linewidth=0.7)
    ax_exp.annotate("V1", xy=(1.0, v1_fclip), xytext=(0.90, v1_fclip - 0.085),
                    fontsize=8.5, color=COLOR_V1, ha="center")
    ax_exp.annotate("realized exposure pinned at the ceiling:\nthe target is never approached",
                    xy=(0.30, 1.0), xytext=(0.36, 0.56), fontsize=8.5, color="#2f6b2a",
                    ha="left", arrowprops=dict(arrowstyle="->", color="#2f6b2a", lw=1.1))
    ax_exp.set_xlabel("Reference exposure level $\\tau$")
    ax_exp.set_ylabel("Realized clipping exposure $f_\\mathrm{clip}$")
    ax_exp.set_title("Realized exposure never reaches $\\tau$", fontsize=10, pad=12)
    ax_exp.set_xlim(0.02, 1.12)
    ax_exp.set_ylim(0.50, 1.09)
    ax_exp.grid(alpha=0.25)
    ax_exp.legend(fontsize=7.6, loc="center left", framealpha=0.92)

    fig.text(0.5, -0.015,
             "FashionMNIST, small CNN, SGD at lr=0.8, development seeds 200-202. "
             "V2 points pool $\\gamma\\in\\{0.05,0.1\\}$ ($n=6$); V1 and the baselines use $n=3$.",
             ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "tau_sweep_accuracy.pdf", bbox_inches="tight")
    fig.savefig(OUT / "tau_sweep_accuracy.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    sweep.to_csv(OUT / "tau_sweep_summary.csv", index=False)
    print(sweep.to_string(index=False))
    print(f"\nV1 (tau=1): acc={v1_acc:.2f} +/- {v1_ci:.2f}  f_clip={v1_fclip:.3f}")
    print(f"V2 spread across tau: {spread:.2f} pts")
    print(f"saved: {OUT / 'tau_sweep_accuracy.png'}")


if __name__ == "__main__":
    main()
