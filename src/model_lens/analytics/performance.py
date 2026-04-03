from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score

from model_lens.domain.performance_metrics import default_performance_metric_names, normalize_performance_metric_names


def compute_bin_edges(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    return np.histogram_bin_edges(values[~np.isnan(values)], bins=n_bins)


def compute_classification_metrics(df: pd.DataFrame, prediction_col: str, label_col: str) -> dict[str, float]:
    pred = pd.to_numeric(df[prediction_col], errors="coerce").to_numpy()
    truth = pd.to_numeric(df[label_col], errors="coerce").to_numpy()
    mask = ~(np.isnan(pred) | np.isnan(truth))
    pred = pred[mask]
    truth = truth[mask]
    if len(pred) == 0:
        return {}
    pred_binary = (pred >= 0.5).astype(int)
    truth_binary = truth.astype(int)
    return {
        "f1": round(float(f1_score(truth_binary, pred_binary, zero_division=0)), 4),
        "precision": round(float(precision_score(truth_binary, pred_binary, zero_division=0)), 4),
        "recall": round(float(recall_score(truth_binary, pred_binary, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(truth_binary, pred_binary)), 4),
    }


def compute_regression_metrics(df: pd.DataFrame, prediction_col: str, label_col: str) -> dict[str, float]:
    pred = pd.to_numeric(df[prediction_col], errors="coerce").to_numpy()
    truth = pd.to_numeric(df[label_col], errors="coerce").to_numpy()
    mask = ~(np.isnan(pred) | np.isnan(truth))
    pred = pred[mask]
    truth = truth[mask]
    if len(pred) == 0:
        return {}
    rmse = float(np.sqrt(mean_squared_error(truth, pred)))
    mae = float(mean_absolute_error(truth, pred))
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
    }


def _numeric_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float, copy=False)
    return pd.to_numeric(series, errors="coerce").to_numpy()


def rank_degradation_contributors(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
    prediction_col: str,
    label_col: str,
    metric_names: tuple[str, ...] | list[str] | None = None,
    n_bins: int = 10,
    problem_type: str = "classification",
) -> pd.DataFrame:
    rows: list[dict] = []
    normalized_problem_type = (problem_type or "classification").strip().lower()
    regression_mode = normalized_problem_type == "regression"
    selected_metric_names = normalize_performance_metric_names(
        normalized_problem_type,
        metric_names or default_performance_metric_names(normalized_problem_type),
    )
    for feature in feature_columns:
        if feature not in baseline_df.columns or feature not in current_df.columns:
            continue
        base_values = _numeric_array(baseline_df[feature])
        cur_values = _numeric_array(current_df[feature])
        clean = base_values[~np.isnan(base_values)]
        if len(clean) < n_bins:
            continue
        edges = compute_bin_edges(base_values, n_bins=n_bins)
        base_bins = np.digitize(base_values, edges[1:-1], right=False)
        cur_bins = np.digitize(cur_values, edges[1:-1], right=False)
        total_current = max(len(current_df), 1)
        for index in range(len(edges) - 1):
            base_slice = baseline_df.iloc[np.where(base_bins == index)[0]]
            cur_slice = current_df.iloc[np.where(cur_bins == index)[0]]
            if base_slice.empty or cur_slice.empty:
                continue
            if regression_mode:
                base_metrics = compute_regression_metrics(base_slice, prediction_col, label_col)
                cur_metrics = compute_regression_metrics(cur_slice, prediction_col, label_col)
            else:
                base_metrics = compute_classification_metrics(base_slice, prediction_col, label_col)
                cur_metrics = compute_classification_metrics(cur_slice, prediction_col, label_col)
            if not base_metrics or not cur_metrics:
                continue
            volume_pct = round(float(len(cur_slice) / total_current * 100), 2)
            for metric_name in selected_metric_names:
                if metric_name not in base_metrics or metric_name not in cur_metrics:
                    continue
                baseline_metric = base_metrics[metric_name]
                current_metric = cur_metrics[metric_name]
                delta = (
                    baseline_metric - current_metric
                    if regression_mode
                    else current_metric - baseline_metric
                )
                rows.append({
                    "feature_name": feature,
                    "bin_label": f"[{edges[index]:.4g}, {edges[index + 1]:.4g})",
                    "baseline_metric": baseline_metric,
                    "current_metric": current_metric,
                    "delta": round(delta, 4),
                    "volume_pct": volume_pct,
                    "contribution": round(delta * volume_pct / 100, 4),
                    "metric_name": metric_name,
                })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["metric_name", "contribution"])
