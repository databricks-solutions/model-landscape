from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from model_landscape.domain.performance_metrics import default_performance_metric_names, normalize_performance_metric_names
from model_landscape.services.class_filters import normalized_binary_series, resolved_prediction_binary_series


def _clean_binning_values(values: np.ndarray, *, clip_percentile: float | None = None) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    clean = clean[~np.isnan(clean)]
    if clean.size == 0:
        return clean
    if clip_percentile is not None and 0.0 < clip_percentile < 50.0:
        lower = float(np.nanpercentile(clean, clip_percentile))
        upper = float(np.nanpercentile(clean, 100.0 - clip_percentile))
        clean = np.clip(clean, lower, upper)
    return clean


def _pad_constant_edges(value: float) -> np.ndarray:
    padding = max(abs(float(value)) * 0.01, 0.5)
    return np.array([float(value) - padding, float(value) + padding], dtype=float)


def compute_bin_edges(
    values: np.ndarray,
    n_bins: int = 10,
    *,
    mode: str = "quantile",
    clip_percentile: float | None = None,
) -> np.ndarray:
    clean = _clean_binning_values(values, clip_percentile=clip_percentile)
    if clean.size == 0:
        return np.array([], dtype=float)
    normalized_mode = str(mode or "quantile").strip().lower()
    if normalized_mode == "fixed_width":
        min_value = float(np.nanmin(clean))
        max_value = float(np.nanmax(clean))
        if min_value == max_value:
            return _pad_constant_edges(min_value)
        return np.linspace(min_value, max_value, num=n_bins + 1, dtype=float)
    if clean.size == 1:
        return _pad_constant_edges(float(clean[0]))
    quantiles = np.linspace(0.0, 1.0, num=n_bins + 1, dtype=float)
    edges = np.unique(np.quantile(clean, quantiles))
    if edges.size < 2:
        return _pad_constant_edges(float(clean[0]))
    return edges.astype(float, copy=False)


def _resolved_binary_arrays(
    df: pd.DataFrame,
    *,
    prediction_col: str,
    label_col: str,
    prediction_score_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pred = resolved_prediction_binary_series(
        df,
        prediction_col=prediction_col,
        prediction_score_col=prediction_score_col,
    )
    truth = normalized_binary_series(df[label_col])
    mask = pred.notna() & truth.notna()
    if not mask.any():
        return np.array([], dtype=int), np.array([], dtype=int)
    return (
        pred.loc[mask].astype(int).to_numpy(),
        truth.loc[mask].astype(int).to_numpy(),
    )


def compute_classification_metrics(
    df: pd.DataFrame,
    prediction_col: str,
    label_col: str,
    prediction_score_col: str | None = None,
) -> dict[str, float | None]:
    pred_binary, truth_binary = _resolved_binary_arrays(
        df,
        prediction_col=prediction_col,
        label_col=label_col,
        prediction_score_col=prediction_score_col,
    )
    if len(pred_binary) == 0:
        return {}
    tp = int(((pred_binary == 1) & (truth_binary == 1)).sum())
    fp = int(((pred_binary == 1) & (truth_binary == 0)).sum())
    fn = int(((pred_binary == 0) & (truth_binary == 1)).sum())
    tn = int(((pred_binary == 0) & (truth_binary == 0)).sum())
    predicted_positive_count = tp + fp
    precision = (tp / predicted_positive_count) if predicted_positive_count > 0 else None
    recall = (tp / (tp + fn)) if predicted_positive_count > 0 and (tp + fn) > 0 else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    accuracy = float(accuracy_score(truth_binary, pred_binary)) if len(pred_binary) > 0 else None
    return {
        "f1": round(float(f1), 4) if f1 is not None else None,
        "precision": round(float(precision), 4) if precision is not None else None,
        "recall": round(float(recall), 4) if recall is not None else None,
        "accuracy": round(float(accuracy), 4) if accuracy is not None else None,
    }


def compute_daily_classification_metrics(
    df: pd.DataFrame,
    prediction_col: str,
    label_col: str,
    prediction_score_col: str | None = None,
) -> dict[str, float | int | None]:
    pred_binary, truth_binary = _resolved_binary_arrays(
        df,
        prediction_col=prediction_col,
        label_col=label_col,
        prediction_score_col=prediction_score_col,
    )
    if len(pred_binary) == 0:
        return {
            "actual_positive_count": 0,
            "actual_negative_count": 0,
            "predicted_positive_count": 0,
            "predicted_negative_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "accuracy": None,
        }
    tp = int(((pred_binary == 1) & (truth_binary == 1)).sum())
    fp = int(((pred_binary == 1) & (truth_binary == 0)).sum())
    fn = int(((pred_binary == 0) & (truth_binary == 1)).sum())
    tn = int(((pred_binary == 0) & (truth_binary == 0)).sum())
    predicted_positive_count = tp + fp
    precision = (tp / predicted_positive_count) if predicted_positive_count > 0 else None
    recall = (tp / (tp + fn)) if predicted_positive_count > 0 and (tp + fn) > 0 else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    accuracy = (tp + tn) / len(pred_binary) if len(pred_binary) > 0 else None
    return {
        "actual_positive_count": int((truth_binary == 1).sum()),
        "actual_negative_count": int((truth_binary == 0).sum()),
        "predicted_positive_count": int((pred_binary == 1).sum()),
        "predicted_negative_count": int((pred_binary == 0).sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(float(precision), 4) if precision is not None else None,
        "recall": round(float(recall), 4) if recall is not None else None,
        "f1": round(float(f1), 4) if f1 is not None else None,
        "accuracy": round(float(accuracy), 4) if accuracy is not None else None,
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
    prediction_score_col: str | None = None,
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
                base_metrics = compute_classification_metrics(
                    base_slice,
                    prediction_col,
                    label_col,
                    prediction_score_col=prediction_score_col,
                )
                cur_metrics = compute_classification_metrics(
                    cur_slice,
                    prediction_col,
                    label_col,
                    prediction_score_col=prediction_score_col,
                )
            if not base_metrics or not cur_metrics:
                continue
            volume_pct = round(float(len(cur_slice) / total_current * 100), 2)
            for metric_name in selected_metric_names:
                if metric_name not in base_metrics or metric_name not in cur_metrics:
                    continue
                baseline_metric = base_metrics[metric_name]
                current_metric = cur_metrics[metric_name]
                if baseline_metric is None or current_metric is None:
                    continue
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
