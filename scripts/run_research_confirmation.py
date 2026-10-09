"""Contingent predictor freeze, fair calibration, confirmation and scale phases."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import yaml

from analysis.predictive_increment import grouped_predictive_increment
from analysis.research_campaign import summaries, write_json
from analysis.statistics import paired_difference, benjamini_hochberg


def freeze_predictor(campaign) -> dict | None:
    output = campaign.output / "p3_predictor"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "predictor.json"
    if path.exists():
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    data = pd.read_csv(campaign.output / "p1/analysis/predictor_development_data.csv.gz")
    base = ["progress", "learning_rate", "loss", "loss_slope", "raw_grad_norm", "applied_update_norm",
            "gradient_alignment", "adam_second_moment_mismatch", "momentum_buffer_alignment"]
    candidates = ["exposure_ema", "duty_cycle", "switch_rate", "burst_age", "time_since_transition", "mean_intensity"]
    screening = []
    for feature in candidates:
        folds, _ = grouped_predictive_increment(data, base_features=base, history_features=[feature],
                                                group_column="family", splits=3)
        wide = folds.pivot(index="fold", columns="model", values="auprc")
        increments = wide["base_plus_history"] - wide["base"]
        screening.append(dict(feature=feature, mean_increment=float(increments.mean()),
                              positive_families=int((increments > 0).sum()), valid_families=int(increments.notna().sum())))
    table = pd.DataFrame(screening).sort_values("mean_increment", ascending=False)
    table.to_csv(output / "single_signal_validation.csv", index=False)
    supported = table[(table["mean_increment"] > 0) & (table["positive_families"] == 3) & (table["valid_families"] == 3)]
    selected = supported["feature"].head(2).tolist()
    if not selected:
        write_json(output / "freeze_status.json", dict(passed=False, reason="no_individual_history_signal_transfers_across_three_families"))
        return None
    data = data.dropna(subset=["future_event"])
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    values = imputer.fit_transform(data[selected])
    normalized = scaler.fit_transform(values)
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(normalized, data["future_event"].astype(int))
    probability = model.predict_proba(normalized)[:, 1]
    predictor = dict(features=selected, medians=imputer.statistics_.tolist(), means=scaler.mean_.tolist(),
                     scales=scaler.scale_.tolist(), coefficients=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
                     reference_probability=float(np.quantile(probability, 0.9)),
                     training_source="P1 screening only; no calibration/confirmation outcomes used",
                     feature_timing="history through t-1 predicts events in t..t+H-1",
                     horizon=campaign.protocol["prediction"]["horizon"])
    write_json(path, predictor)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def policy_candidates(campaign, family: dict, threshold: float, lr: float, predictor: dict) -> dict[str, list[dict]]:
    calibration = campaign.protocol["calibration"]
    candidates = {
        "none": [dict(clipping=dict(name="none"), lr=lr, optimizer=family["optimizer"])],
        "fixed": [dict(clipping=dict(name="fixed", threshold=float(value)), lr=lr, optimizer=family["optimizer"])
                  for value in (threshold / 3, threshold, threshold * 3)],
        "autoclip": [dict(clipping=dict(name="autoclip", percentile=value), lr=lr, optimizer=family["optimizer"])
                     for value in calibration["autoclip"]],
        "agc": [dict(clipping=dict(name="agc", clip_value=value), lr=lr, optimizer=family["optimizer"])
                for value in calibration["agc"]],
        "adagc": [dict(clipping=dict(name="adagc", relative_threshold=value), lr=lr, optimizer=family["optimizer"])
                  for value in calibration["adagc"]],
        "event_timed": [dict(clipping=dict(name="predictive_event", predictor_path=predictor["path"],
                                             predictor_sha256=predictor["sha256"], aggressiveness=value),
                              lr=lr, optimizer=family["optimizer"]) for value in calibration["event_timed"]],
    }
    if family["optimizer"] == "adamw":
        candidates["stable_adamw"] = [dict(clipping=dict(name="none"), lr=lr * value, optimizer="stable_adamw")
                                        for value in calibration["stable_adamw_lr_multipliers"]]
    return candidates


def calibrate(campaign, cells: list[dict], predictor: dict, stage: str = "p3_calibration") -> list[dict]:
    configs = []
    specifications = []
    seeds = campaign.protocol["calibration"]["seeds"] if stage == "p3_calibration" else campaign.protocol["scale"]["calibration_seeds"]
    for cell in cells:
        source = yaml.safe_load(Path(cell["source_config"]).read_text())
        family = dict(name=cell["family"], dataset=source["dataset"]["name"], model=source["model"]["name"], optimizer=source["optimizer"]["name"])
        candidates = policy_candidates(campaign, family, cell["threshold"], cell["learning_rate"], predictor)
        for method, settings in candidates.items():
            for index, setting in enumerate(settings):
                label = f"{method}-candidate-{index}"
                specifications.append(dict(family=family, method=method, label=label, setting=setting))
                for seed in seeds:
                    run_family = {**family, "optimizer": setting["optimizer"]}
                    cfg = campaign.configuration(run_family, stage, label, seed, campaign.protocol["calibration"]["epochs"],
                                                 setting["lr"], setting["clipping"])
                    configs.append(campaign.materialize(cfg))
    campaign.run_jobs(configs, stage)
    table = summaries(configs)
    table["accuracy_with_failures"] = table["evaluation_accuracy"].fillna(0)
    table.to_csv(campaign.output / stage / "calibration_runs.csv", index=False)
    selected = []
    for family in sorted({item["family"]["name"] for item in specifications}):
        for method in sorted({item["method"] for item in specifications if item["family"]["name"] == family}):
            candidates = [item for item in specifications if item["family"]["name"] == family and item["method"] == method]
            scored = []
            for candidate in candidates:
                rows = table[(table["family"] == family) & (table["label"] == candidate["label"])]
                if len(rows) != len(seeds):
                    raise ValueError("incomplete calibration cell")
                score = (float(rows["failed"].mean()), -float(rows["accuracy_with_failures"].mean()), candidate["label"])
                scored.append((score, candidate))
            score, candidate = min(scored, key=lambda value: value[0])
            selected.append({**candidate, "calibration_failure_rate": score[0], "calibration_accuracy": -score[1]})
    write_json(campaign.output / stage / "frozen_selections.json", selected)
    return selected


def confirm(campaign, selected: list[dict], stage: str, epochs: int, seeds: list[int]) -> tuple[list[dict], pd.DataFrame]:
    configs = []
    for candidate in selected:
        setting = candidate["setting"]
        family = {**candidate["family"], "optimizer": setting["optimizer"]}
        for seed in seeds:
            cfg = campaign.configuration(family, stage, candidate["method"], seed, epochs,
                                         setting["lr"], setting["clipping"], evaluate_test=True)
            configs.append(campaign.materialize(cfg))
    campaign.run_jobs(configs, stage)
    table = summaries(configs)
    table["accuracy_with_failures"] = table["evaluation_accuracy"].fillna(0)
    table["loss_with_failures"] = table["evaluation_loss"].fillna(campaign.protocol["training"]["failure_loss"])
    table.to_csv(campaign.output / stage / "confirmation_runs.csv", index=False)
    return configs, table


def confirmation_gate(campaign, table: pd.DataFrame, selected: list[dict], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    effects = []
    for family, group in table.groupby("family"):
        event = group[group["label"] == "event_timed"].set_index("seed")
        for label, baseline in group.groupby("label"):
            if label == "event_timed":
                continue
            baseline = baseline.set_index("seed")
            for outcome in ("accuracy_with_failures", "loss_with_failures", "failed", "adverse_event_count"):
                paired = event[[outcome]].join(baseline[[outcome]], lsuffix="_event", rsuffix="_baseline").dropna()
                effect = paired_difference(paired[f"{outcome}_event"].to_numpy(dtype=float),
                                           paired[f"{outcome}_baseline"].to_numpy(dtype=float))
                effects.append(dict(family=family, baseline=label, outcome=outcome, **effect))
    frame = pd.DataFrame(effects)
    frame["adjusted_pvalue"] = benjamini_hochberg(frame["permutation_pvalue"])
    frame.to_csv(output / "paired_effects.csv", index=False)
    settings = campaign.protocol["confirmation"]
    family_reports = []
    for family, group in frame.groupby("family"):
        # The strong adaptive comparator is chosen on independent calibration
        # outcomes, never by selecting a convenient confirmation result.
        calibrated = [item for item in selected if item["family"]["name"] == family and item["method"] not in {"none", "fixed", "event_timed"}]
        strongest = min(calibrated, key=lambda item: (item["calibration_failure_rate"], -item["calibration_accuracy"]))["method"]
        comparisons = group[group["baseline"].isin({"fixed", strongest})]
        accuracy = comparisons[comparisons["outcome"] == "accuracy_with_failures"]
        loss = comparisons[comparisons["outcome"] == "loss_with_failures"]
        stability = comparisons[comparisons["outcome"].isin({"failed", "adverse_event_count"})]
        noninferior = bool((accuracy["ci_lower"] > -settings["accuracy_noninferiority_margin"]).all() and
                          (loss["ci_upper"] < settings["loss_noninferiority_margin"]).all())
        benefits = []
        for baseline, rows in stability.groupby("baseline"):
            good = any(row.adjusted_pvalue < 0.05 and row.ci_upper < 0 and
                       row.mean_difference <= -(settings["minimum_failure_improvement"] if row.outcome == "failed"
                                                else settings["minimum_adverse_event_improvement"])
                       for row in rows.itertuples())
            benefits.append(good)
        family_reports.append(dict(family=family, strongest_calibrated_baseline=strongest,
                                   accuracy_loss_noninferior=noninferior,
                                   stability_benefit=all(benefits) and bool(benefits),
                                   passed=noninferior and all(benefits) and bool(benefits)))
    passed = sum(item["passed"] for item in family_reports) >= 2
    report = dict(gate="G3", passed=passed, families=family_reports, frozen_selections=selected,
                  reason="criteria_met" if passed else "stability_and_noninferiority_gate_not_met")
    write_json(output / "gate_g3.json", report)
    means = table.groupby(["family", "label"]).agg(accuracy=("accuracy_with_failures", "mean"),
              accuracy_sd=("accuracy_with_failures", "std"), failures=("failed", "mean"), seeds=("seed", "size"))
    means.to_csv(output / "accuracy_stability.csv")
    return report


def run_confirmation(campaign, cells: list[dict]) -> dict:
    campaign.event("predictor_freeze_started", stage="p3")
    predictor = freeze_predictor(campaign)
    if predictor is None:
        return dict(gate="G3", passed=False, reason="no_validated_one_or_two_signal_predictor")
    selections = calibrate(campaign, cells, predictor)
    _, table = confirm(campaign, selections, "p3_confirmation", campaign.protocol["confirmation"]["epochs"],
                       campaign.protocol["confirmation"]["seeds"])
    return confirmation_gate(campaign, table, selections, campaign.output / "p3_confirmation/analysis")


def run_scale(campaign, g3: dict) -> dict:
    passed_families = {row["family"] for row in g3["families"] if row["passed"]}
    source = next(item for item in g3["frozen_selections"] if item["family"]["name"] in passed_families
                  and item["method"] == "fixed")
    scale = campaign.protocol["scale"]
    family = {**source["family"], "name": "cifar100_resnet34", "dataset": scale["dataset"], "model": scale["model"]}
    setting = source["setting"]
    # Scale is a transfer test: the controller/risk model stays frozen. Baseline
    # hyperparameters get the same three-candidate calibration budget again.
    source_config = campaign.configuration(family, "p4_calibration", "source", scale["calibration_seeds"][0],
                                           campaign.protocol["calibration"]["epochs"], setting["lr"], setting["clipping"])
    source_config = campaign.materialize(source_config)
    cell = dict(family=family["name"], source_config=source_config["campaign"]["config_path"],
                threshold=setting["clipping"]["threshold"], learning_rate=setting["lr"])
    predictor_path = campaign.output / "p3_predictor/predictor.json"
    predictor = dict(path=str(predictor_path), sha256=hashlib.sha256(predictor_path.read_bytes()).hexdigest())
    selections = calibrate(campaign, [cell], predictor, stage="p4_calibration")
    # Preserve the source event aggressiveness: calibration runs for its three
    # candidates are reported but are not used to retune the transferred policy.
    event_source = next(item for item in g3["frozen_selections"] if item["family"]["name"] == source["family"]["name"]
                        and item["method"] == "event_timed")
    for item in selections:
        if item["method"] == "event_timed":
            item["setting"]["clipping"] = copy.deepcopy(event_source["setting"]["clipping"])
    write_json(campaign.output / "p4_calibration/frozen_transfer_selections.json", selections)
    _, table = confirm(campaign, selections, "p4_confirmation", scale["epochs"], scale["seeds"])
    results = confirmation_gate(campaign, table, selections, campaign.output / "p4_confirmation/analysis")
    results["gate"] = "P4_transfer"
    results["passed"] = all(row["passed"] for row in results["families"])
    results["training_completed"] = True
    write_json(campaign.output / "p4_confirmation/analysis/transfer_report.json", results)
    return results
