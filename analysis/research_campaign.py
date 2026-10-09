"""Run-level gate reports for the original timing/placement research program.

Scientific gates are conservative and may stop a campaign with a negative or
inconclusive result. No absent result is interpreted as a pass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from analysis.event_detection import event_sensitivity_grid, surrogate_episode_counts
from analysis.predictive_increment import add_forward_event_target, grouped_predictive_increment
from analysis.statistics import benjamini_hochberg, bootstrap_interval, paired_difference
from src.dynamics import HysteresisConfig
from src.records import _json_value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def summaries(configs: list[dict]) -> pd.DataFrame:
    rows = []
    for config in configs:
        path = Path(config["output_dir"]) / f"{config['run_id']}.summary.json"
        value = json.loads(path.read_text())
        value.update(family=config["campaign"]["family"], stage=config["campaign"]["stage"],
                     label=config["campaign"]["label"], config_path=config["campaign"]["config_path"])
        rows.append(value)
    return pd.DataFrame(rows)


def screen_report(configs: list[dict], settings: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    summary = summaries(configs)
    summary.to_csv(output / "run_summary.csv", index=False)
    frames, sensitivities, surrogates = [], [], []
    for config in configs:
        path = Path(config["output_dir"]) / f"{config['run_id']}.steps.jsonl"
        frame = pd.read_json(path, lines=True)
        if frame.empty:
            continue
        frame = frame.sort_values("step").reset_index(drop=True)
        frame["family"] = config["campaign"]["family"]
        frame["seed"] = config["seed"]
        frame["config_group"] = f"{config['campaign']['family']}/{config['optimizer']['lr']}/{config['campaign']['label']}"
        frame["adverse_event"] = frame.get("adverse_event", pd.Series(False, index=frame.index)).fillna(False).astype(int)
        frame["progress"] = frame["epoch"] / config["epochs"]
        # Slopes and labels are constructed per run before any thinning.
        frame["loss_slope"] = frame["loss"].diff().ewm(alpha=0.05, adjust=False).mean()
        frame = add_forward_event_target(frame, event_column="adverse_event", horizon=settings["horizon"])
        usable = frame.dropna(subset=["future_event", "raw_grad_norm", "applied_update_norm"])
        usable = usable.loc[usable["step"] % settings["row_stride"] == 0]
        frames.append(usable)
        q_frame = frame.dropna(subset=["clipping_coordinate"])
        if config["clipping"]["name"] == "fixed" and len(q_frame):
            grid = event_sensitivity_grid(q_frame, settings["on_margins"], settings["off_margins"], settings["dwell_steps"])
            grid["run_id"] = config["run_id"]
            grid["family"] = config["campaign"]["family"]
            sensitivities.append(grid)
            null = surrogate_episode_counts(q_frame["clipping_coordinate"], HysteresisConfig(), range(20), block_size=20)
            null["run_id"] = config["run_id"]
            surrogates.append(null)
    if not frames:
        report = {"gate": "G1", "passed": False, "reason": "no_usable_trajectories"}
        write_json(output / "gate_g1.json", report)
        return report
    data = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    base = ["progress", "learning_rate", "loss", "loss_slope", "raw_grad_norm", "applied_update_norm",
            "gradient_alignment", "adam_second_moment_mismatch", "momentum_buffer_alignment"]
    history = ["exposure_ema", "duty_cycle", "switch_rate", "burst_age", "time_since_transition", "mean_intensity"]
    for name in base + history:
        if name not in data:
            data[name] = np.nan
    data[["run_id", "family", "seed", "config_group", "future_event", *base, *history]].to_csv(
        output / "predictor_development_data.csv.gz", index=False, compression="gzip")
    reports = {}
    run_increments = []
    # Both initialization groups and whole configurations are held out. The
    # second check prevents nearby seeds of one setting from being sufficient.
    for grouping in ("seed", "config_group", "family"):
        try:
            folds, predictions = grouped_predictive_increment(data, base_features=base,
                history_features=history, group_column=grouping, splits=settings["grouped_folds"])
        except ValueError as exc:
            reports[grouping] = {"valid": False, "reason": str(exc)}
            continue
        folds.to_csv(output / f"predictive_folds_{grouping}.csv", index=False)
        predictions.to_csv(output / f"predictive_predictions_{grouping}.csv.gz", index=False, compression="gzip")
        local = []
        if len(predictions):
            usable = data.dropna(subset=["future_event", grouping]).reset_index(drop=True)
            wide = predictions.pivot(index="row_index", columns="model", values="probability")
            wide = wide.join(usable[["run_id", "family", "seed", "future_event"]])
            for run_id, group in wide.groupby("run_id"):
                if group["future_event"].nunique() < 2 or group[["base", "base_plus_history"]].isna().any().any():
                    continue
                target = group["future_event"].astype(int)
                item = dict(grouping=grouping, run_id=run_id, family=group["family"].iloc[0],
                            seed=int(group["seed"].iloc[0]), n_steps=len(group),
                            positive_steps=int(target.sum()),
                            auprc_increment=average_precision_score(target, group["base_plus_history"])
                                - average_precision_score(target, group["base"]),
                            brier_improvement=brier_score_loss(target, group["base"])
                                - brier_score_loss(target, group["base_plus_history"]))
                local.append(item)
                run_increments.append(item)
        if not local:
            reports[grouping] = {"valid": False, "reason": "no_runs_with_both_target_classes"}
            continue
        increments = pd.DataFrame(local)
        interval = bootstrap_interval(increments["auprc_increment"], samples=settings["bootstrap_samples"])
        family_means = increments.groupby("family")["auprc_increment"].mean().to_dict()
        reports[grouping] = dict(valid=True, n_informative_runs=len(increments),
                                mean=interval.estimate, lower=interval.lower, upper=interval.upper,
                                family_means=family_means,
                                passes_increment=len(increments) >= settings["min_informative_runs"]
                                    and interval.estimate >= settings["minimum_auprc_increment"]
                                    and interval.lower > 0 and len(family_means) >= 3
                                    and all(value > 0 for value in family_means.values()))
    pd.DataFrame(run_increments).to_csv(output / "predictive_run_increments.csv", index=False)
    robust_fraction = 0.0
    if sensitivities:
        sensitivity = pd.concat(sensitivities, ignore_index=True)
        sensitivity.to_csv(output / "event_sensitivity.csv", index=False)
        robust = []
        for _, group in sensitivity.groupby("run_id"):
            counts = group["episode_count"]
            if counts.median() >= 2:
                robust.append((counts.max() - counts.min()) / counts.median() <= settings["maximum_relative_episode_range"])
        robust_fraction = float(np.mean(robust)) if robust else 0.0
        if surrogates:
            pd.concat(surrogates, ignore_index=True).to_csv(output / "event_surrogates.csv", index=False)
    passed = all(reports.get(name, {}).get("passes_increment", False) for name in ("seed", "config_group", "family"))
    passed = passed and robust_fraction >= settings["minimum_robust_run_fraction"]
    report = dict(gate="G1", passed=bool(passed), prediction=reports,
                  robust_event_run_fraction=robust_fraction,
                  reason="criteria_met" if passed else "predictive_or_event_robustness_gate_not_met",
                  scope="exploratory screening; bootstrap units are runs, not steps",
                  selected_cells=select_cells(summary))
    write_json(output / "gate_g1.json", report)
    plot_screen(summary, output)
    return report


def select_cells(summary: pd.DataFrame) -> list[dict]:
    """Select one diagnostic cell per family using validation data only.

    Prefer an unstable unmodified setting with a trainable fixed policy. Within
    that setting prefer nondegenerate exposure, then validation performance.
    Selection happens before independent causal seeds are run.
    """
    selected = []
    for family, table in summary.groupby("family"):
        candidates = []
        for lr, cell in table.groupby("learning_rate"):
            no_clip = cell[cell["clipping_policy"] == "none"]
            fail_rate = float(no_clip["failed"].mean()) if len(no_clip) else 0.0
            for label, fixed in cell[cell["clipping_policy"] == "fixed"].groupby("label"):
                if fixed["failed"].any() or fixed["evaluation_accuracy"].isna().any():
                    continue
                exposure = float(fixed["clipping_frequency"].mean())
                nondegenerate = 0.05 < exposure < 0.95
                score = (fail_rate, int(nondegenerate), float(fixed["evaluation_accuracy"].mean()))
                candidates.append((score, fixed.iloc[0]))
        if candidates:
            _, row = max(candidates, key=lambda value: value[0])
            selected.append(dict(family=family, source_config=row["config_path"],
                                 learning_rate=float(row["learning_rate"]), threshold=float(row["clipping_threshold"])))
    return selected


def plot_screen(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    families = list(summary["family"].unique())
    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4), squeeze=False)
    for axis, family in zip(axes[0], families):
        table = summary[(summary["family"] == family) & (summary["clipping_policy"] == "fixed")]
        grid = table.pivot_table(index="clipping_threshold", columns="learning_rate", values="evaluation_accuracy", aggfunc="mean")
        if grid.empty:
            continue
        im = axis.imshow(grid.to_numpy() * 100, origin="lower", aspect="auto", vmin=0, vmax=100, cmap="viridis")
        axis.set_xticks(range(len(grid.columns)), [f"{x:g}" for x in grid.columns])
        axis.set_yticks(range(len(grid.index)), [f"{x:g}" for x in grid.index])
        axis.set(title=family, xlabel="Learning rate", ylabel="Clipping threshold")
        fig.colorbar(im, ax=axis, label="Validation accuracy (%)")
    fig.tight_layout()
    fig.savefig(output / "screening_accuracy.png", dpi=180)
    fig.savefig(output / "screening_accuracy.pdf")
    plt.close(fig)


def causal_report(records: list[dict], settings: dict, output: Path) -> dict:
    """Pair only independent seeds; adjust the declared causal test family."""
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "causal_run_endpoints.csv", index=False)
    if frame.empty:
        report = dict(gate="G2", passed=False, reason="no_complete_pairs")
        write_json(output / "gate_g2.json", report)
        return report
    effects = []
    for family, table in frame.groupby("family"):
        reference = table[table["intervention"] == "reference_gain_replay_pre_moment"].set_index("seed")
        for label, intervention in table.groupby("intervention"):
            if label == "reference_gain_replay_pre_moment":
                continue
            intervention = intervention.set_index("seed")
            for outcome in ("failed", "accuracy_with_failures", "short_horizon_loss"):
                pairs = reference[[outcome]].join(intervention[[outcome]], lsuffix="_ref", rsuffix="_alt").dropna()
                effect = paired_difference(pairs[f"{outcome}_alt"].to_numpy(), pairs[f"{outcome}_ref"].to_numpy())
                effects.append(dict(family=family, intervention=label, outcome=outcome, **effect))
    effects_frame = pd.DataFrame(effects)
    if not len(effects_frame):
        report = dict(gate="G2", passed=False, reason="no_complete_pairs")
    else:
        effects_frame["adjusted_pvalue"] = benjamini_hochberg(effects_frame["permutation_pvalue"])
        margins = dict(failed=settings["minimum_failure_effect"], accuracy_with_failures=settings["minimum_accuracy_effect"],
                       short_horizon_loss=settings["minimum_loss_effect"])
        effects_frame["meaningful"] = [
            row.n_pairs >= 10 and row.adjusted_pvalue < settings["alpha"] and
            (row.ci_lower > margins[row.outcome] or row.ci_upper < -margins[row.outcome])
            for row in effects_frame.itertuples()
        ]
        effects_frame.to_csv(output / "causal_paired_effects.csv", index=False)
        supported = effects_frame[effects_frame["meaningful"]]
        # Require the same outcome/intervention to replicate in two families.
        replicated = supported.groupby(["intervention", "outcome"]).filter(lambda group: len(group) >= 2 and
            (np.sign(group["mean_difference"]) == np.sign(group["mean_difference"].iloc[0])).all())
        report = dict(gate="G2", passed=bool(len(replicated)), supported_effects=supported.to_dict("records"),
                      replicated_effects=replicated.to_dict("records"),
                      reason="criteria_met" if len(replicated) else "no_replicated_functional_or_stability_effect")
    write_json(output / "gate_g2.json", report)
    return report
