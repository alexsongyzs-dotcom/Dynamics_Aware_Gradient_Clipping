"""Grouped, out-of-run evaluation of event-history predictive value."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def add_forward_event_target(
    frame: pd.DataFrame,
    *,
    event_column: str,
    horizon: int,
    run_column: str = "run_id",
    output_column: str = "future_event",
) -> pd.DataFrame:
    """Label whether an event occurs in the next ``1..horizon`` steps."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    result = frame.copy()
    result[output_column] = np.nan
    for _, indices in result.groupby(run_column, sort=False).groups.items():
        positions = np.asarray(list(indices))
        events = result.loc[positions, event_column].to_numpy(dtype=int)
        target = np.zeros(len(events), dtype=int)
        for offset in range(1, horizon + 1):
            if offset < len(events):
                target[:-offset] |= events[offset:] > 0
        valid_stop = max(0, len(events) - horizon)
        if valid_stop:
            result.loc[positions[:valid_stop], output_column] = target[:valid_stop]
    return result


def _model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    if len(np.unique(target)) < 2:
        return {"auprc": float("nan"), "auroc": float("nan"), "brier": float("nan"), "log_loss": float("nan")}
    return {
        "auprc": float(average_precision_score(target, probability)),
        "auroc": float(roc_auc_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
    }


def grouped_predictive_increment(
    frame: pd.DataFrame,
    *,
    base_features: Sequence[str],
    history_features: Sequence[str],
    target_column: str = "future_event",
    group_column: str = "run_id",
    splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare Base and Base+History models with whole groups held out."""

    needed = set(base_features) | set(history_features) | {target_column, group_column}
    missing = needed - set(frame.columns)
    if missing:
        raise KeyError(f"missing predictive-analysis columns: {sorted(missing)}")
    data = frame.dropna(subset=[target_column, group_column]).reset_index(drop=True)
    groups = data[group_column].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("grouped validation requires at least two independent groups")
    splitter = GroupKFold(n_splits=min(int(splits), len(unique_groups)))
    target = data[target_column].to_numpy(dtype=int)
    feature_sets = {
        "base": list(base_features),
        "base_plus_history": list(base_features) + list(history_features),
    }
    fold_rows = []
    prediction_rows = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(data, target, groups)):
        for model_name, features in feature_sets.items():
            model = _model()
            model.fit(data.loc[train_indices, features], target[train_indices])
            probability = model.predict_proba(data.loc[test_indices, features])[:, 1]
            metrics = _metrics(target[test_indices], probability)
            fold_rows.append({"fold": fold, "model": model_name, **metrics})
            prediction_rows.extend(
                {
                    "row_index": int(index),
                    "fold": fold,
                    "model": model_name,
                    "target": int(target[index]),
                    "probability": float(value),
                    "group": groups[index],
                }
                for index, value in zip(test_indices, probability)
            )
    folds = pd.DataFrame(fold_rows)
    base = folds[folds["model"] == "base"].set_index("fold")
    augmented = folds[folds["model"] == "base_plus_history"].set_index("fold")
    for metric in ("auprc", "auroc"):
        folds.attrs[f"increment_{metric}"] = float((augmented[metric] - base[metric]).mean())
    for metric in ("brier", "log_loss"):
        folds.attrs[f"improvement_{metric}"] = float((base[metric] - augmented[metric]).mean())
    return folds, pd.DataFrame(prediction_rows)
