from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from model_lens.analytics.drift import (
    compute_categorical_distribution_metrics,
    compute_feature_drift,
    compute_js,
    compute_kl,
    compute_psi,
)
from model_lens.analytics.performance import (
    compute_bin_edges,
    compute_daily_classification_metrics,
    compute_classification_metrics,
    compute_regression_metrics,
    rank_degradation_contributors,
)
from model_lens.domain.models import BaselinePolicy, MonitorConfig, RefreshResult
from model_lens.domain.performance_metrics import default_performance_metric_names
from model_lens.services.class_filters import normalized_binary_series, resolved_prediction_binary_series, supports_binary_class_filters
from model_lens.services.incidents import build_incident_history, build_incidents


def _safe_float(value: object) -> float | None:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return None
    return float(converted)


WindowKey = tuple[str, str, str, str]


def _normalize_frame(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    ordered = df.copy()
    ordered[timestamp_col] = pd.to_datetime(ordered[timestamp_col], errors="coerce", utc=True).dt.tz_localize(None)
    ordered = ordered[ordered[timestamp_col].notna()].sort_values(timestamp_col).reset_index(drop=True)
    return ordered


def _slice_window(
    ordered: pd.DataFrame,
    *,
    timestamp_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    normalized = ordered[timestamp_col].dt.normalize()
    return ordered[(normalized >= start) & (normalized <= end)].copy()


def prepare_window_frame(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    return _normalize_frame(df, timestamp_col)


def slice_window_pair(
    ordered: pd.DataFrame,
    *,
    timestamp_col: str,
    metadata: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_start = pd.Timestamp(metadata["baseline_start"]).normalize()
    baseline_end = pd.Timestamp(metadata["baseline_end"]).normalize()
    current_start = pd.Timestamp(metadata["window_start"]).normalize()
    current_end = pd.Timestamp(metadata["window_end"]).normalize()
    baseline_df = _slice_window(ordered, timestamp_col=timestamp_col, start=baseline_start, end=baseline_end)
    current_df = _slice_window(ordered, timestamp_col=timestamp_col, start=current_start, end=current_end)
    return baseline_df, current_df


def _window_key(metadata: dict[str, str]) -> WindowKey:
    return (
        str(metadata["baseline_start"]),
        str(metadata["baseline_end"]),
        str(metadata["window_start"]),
        str(metadata["window_end"]),
    )


def _window_metadata(
    *,
    model_key: str,
    baseline_kind: str,
    baseline_start: pd.Timestamp,
    baseline_end: pd.Timestamp,
    current_start: pd.Timestamp,
    current_end: pd.Timestamp,
) -> dict[str, str]:
    baseline_start_text = baseline_start.date().isoformat()
    baseline_end_text = baseline_end.date().isoformat()
    current_start_text = current_start.date().isoformat()
    current_end_text = current_end.date().isoformat()
    return {
        "window_id": "|".join(
            (
                model_key,
                baseline_kind,
                baseline_start_text,
                baseline_end_text,
                current_start_text,
                current_end_text,
            )
        ),
        "baseline_kind": baseline_kind,
        "baseline_start": baseline_start_text,
        "baseline_end": baseline_end_text,
        "window_start": current_start_text,
        "window_end": current_end_text,
        "window_grain": "daily",
    }


def _rolling_metadata(
    *,
    model_key: str,
    earliest_date: pd.Timestamp,
    latest_date: pd.Timestamp,
    policy: BaselinePolicy,
) -> list[dict[str, str]]:
    n_days = max(policy.n_days, 1)
    earliest_current_end = earliest_date + timedelta(days=(2 * n_days) - 1)
    if earliest_current_end > latest_date:
        return []
    recent_floor = latest_date - timedelta(days=max(policy.max_comparison_days - 1, 0))
    start_current_end = max(earliest_current_end, recent_floor)
    windows: list[dict[str, str]] = []
    current_end = start_current_end
    while current_end <= latest_date:
        current_start = current_end - timedelta(days=n_days - 1)
        baseline_end = current_start - timedelta(days=1)
        baseline_start = baseline_end - timedelta(days=n_days - 1)
        windows.append(
            _window_metadata(
                model_key=model_key,
                baseline_kind=policy.kind,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                current_start=current_start,
                current_end=current_end,
            )
        )
        current_end += timedelta(days=1)
    return windows


def _fixed_metadata(
    *,
    model_key: str,
    latest_date: pd.Timestamp,
    policy: BaselinePolicy,
) -> list[dict[str, str]]:
    baseline_start = pd.Timestamp(policy.baseline_start).normalize()
    baseline_end = pd.Timestamp(policy.baseline_end).normalize()
    earliest_current_end = baseline_end + timedelta(days=policy.n_days)
    if earliest_current_end > latest_date:
        return []
    recent_floor = latest_date - timedelta(days=max(policy.max_comparison_days - 1, 0))
    start_current_end = max(earliest_current_end, recent_floor)
    windows: list[dict[str, str]] = []
    current_end = start_current_end
    while current_end <= latest_date:
        current_start = current_end - timedelta(days=policy.n_days - 1)
        if current_start <= baseline_end:
            current_end += timedelta(days=1)
            continue
        windows.append(
            _window_metadata(
                model_key=model_key,
                baseline_kind=policy.kind,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                current_start=current_start,
                current_end=current_end,
            )
        )
        current_end += timedelta(days=1)
    return windows


def generate_window_pairs(
    df: pd.DataFrame,
    timestamp_col: str,
    baseline: BaselinePolicy | int,
    model_key: str = "",
    existing_window_keys: set[WindowKey] | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]]:
    ordered = _normalize_frame(df, timestamp_col)
    if ordered.empty:
        return []
    policy = baseline if isinstance(baseline, BaselinePolicy) else BaselinePolicy(kind="rolling", n_days=int(baseline))
    earliest_date = ordered[timestamp_col].min().normalize()
    latest_date = ordered[timestamp_col].max().normalize()
    metadata_list = (
        _fixed_metadata(model_key=model_key, latest_date=latest_date, policy=policy)
        if policy.kind == "fixed"
        else _rolling_metadata(model_key=model_key, earliest_date=earliest_date, latest_date=latest_date, policy=policy)
    )
    existing = existing_window_keys or set()
    pairs: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]] = []
    for metadata in metadata_list:
        if _window_key(metadata) in existing:
            continue
        baseline_start = pd.Timestamp(metadata["baseline_start"]).normalize()
        baseline_end = pd.Timestamp(metadata["baseline_end"]).normalize()
        current_start = pd.Timestamp(metadata["window_start"]).normalize()
        current_end = pd.Timestamp(metadata["window_end"]).normalize()
        baseline_df = _slice_window(ordered, timestamp_col=timestamp_col, start=baseline_start, end=baseline_end)
        current_df = _slice_window(ordered, timestamp_col=timestamp_col, start=current_start, end=current_end)
        if baseline_df.empty or current_df.empty:
            continue
        pairs.append((baseline_df, current_df, metadata))
    return pairs


def generate_window_metadata(
    *,
    min_date: str | None,
    max_date: str | None,
    baseline: BaselinePolicy | int,
    model_key: str = "",
    existing_window_keys: set[WindowKey] | None = None,
) -> list[dict[str, str]]:
    if not min_date or not max_date:
        return []
    earliest_date = pd.Timestamp(min_date).normalize()
    latest_date = pd.Timestamp(max_date).normalize()
    if pd.isna(earliest_date) or pd.isna(latest_date) or earliest_date > latest_date:
        return []
    policy = baseline if isinstance(baseline, BaselinePolicy) else BaselinePolicy(kind="rolling", n_days=int(baseline))
    metadata_list = (
        _fixed_metadata(model_key=model_key, latest_date=latest_date, policy=policy)
        if policy.kind == "fixed"
        else _rolling_metadata(model_key=model_key, earliest_date=earliest_date, latest_date=latest_date, policy=policy)
    )
    existing = existing_window_keys or set()
    return [metadata for metadata in metadata_list if _window_key(metadata) not in existing]


def split_baseline_current(
    df: pd.DataFrame,
    timestamp_col: str,
    baseline: BaselinePolicy | int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = _normalize_frame(df, timestamp_col)
    if ordered.empty:
        return ordered, ordered
    pairs = generate_window_pairs(ordered, timestamp_col, baseline)
    if not pairs:
        return ordered.iloc[0:0].copy(), ordered.iloc[0:0].copy()
    latest_baseline, latest_current, _ = pairs[-1]
    return latest_baseline, latest_current


def _build_drift_rows(
    *,
    config: MonitorConfig,
    drift_metrics: pd.DataFrame,
    metadata: dict[str, str],
    computed_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in drift_metrics.iterrows():
        for metric_name in ("psi", "kl_divergence", "js_divergence"):
            rows.append({
                "model_key": config.model_key,
                "feature_name": row["feature_name"],
                "metric_name": metric_name,
                "metric_value": float(row[metric_name]),
                "window_id": metadata["window_id"],
                "window_start": metadata["window_start"],
                "window_end": metadata["window_end"],
                "baseline_start": metadata["baseline_start"],
                "baseline_end": metadata["baseline_end"],
                "ref_mean": float(row["ref_mean"]),
                "cur_mean": float(row["cur_mean"]),
                "ref_std": float(row["ref_std"]),
                "cur_std": float(row["cur_std"]),
                "ref_null_pct": float(row["ref_null_pct"]),
                "cur_null_pct": float(row["cur_null_pct"]),
                "ref_count": int(row["ref_count"]),
                "cur_count": int(row["cur_count"]),
                "computed_at": computed_at,
            })
    return rows


def _build_performance_rows(
    *,
    config: MonitorConfig,
    performance_frame: pd.DataFrame,
    metadata: dict[str, str],
    computed_at: str,
) -> list[dict[str, Any]]:
    return [{
        "model_key": config.model_key,
        "window_id": metadata["window_id"],
        "feature_name": row["feature_name"],
        "bin_label": row["bin_label"],
        "baseline_metric": float(row["baseline_metric"]),
        "current_metric": float(row["current_metric"]),
        "delta": float(row["delta"]),
        "volume_pct": float(row["volume_pct"]),
        "contribution": float(row["contribution"]),
        "metric_name": str(row.get("metric_name") or config.default_performance_metric or default_performance_metric_names(config.problem_type)[0]),
        "window_start": metadata["window_start"],
        "window_end": metadata["window_end"],
        "computed_at": computed_at,
    } for _, row in performance_frame.iterrows()]


def _build_quality_rows(
    *,
    config: MonitorConfig,
    inference_df: pd.DataFrame,
    computed_at: str,
) -> list[dict[str, Any]]:
    return [{
        "model_key": config.model_key,
        "total_rows": int(len(inference_df)),
        "min_date": str(pd.to_datetime(inference_df[config.contract.timestamp_col]).min().date()),
        "max_date": str(pd.to_datetime(inference_df[config.contract.timestamp_col]).max().date()),
        "prediction_mean": _safe_float(pd.to_numeric(inference_df[config.contract.prediction_col], errors="coerce").mean()),
        "prediction_std": _safe_float(pd.to_numeric(inference_df[config.contract.prediction_col], errors="coerce").std()),
        "daily_volume": json.dumps({
            str(index): int(value)
            for index, value in inference_df.groupby(
                pd.to_datetime(inference_df[config.contract.timestamp_col]).dt.date
            ).size().items()
        }),
        "null_rates": json.dumps({
            feature: round(float(inference_df[feature].isna().mean() * 100), 2)
            for feature in config.contract.feature_columns
            if feature in inference_df.columns
        }),
        "computed_at": computed_at,
    }]


def build_quality_rows_from_profile(
    *,
    config: MonitorConfig,
    profile: dict[str, Any],
    computed_at: str,
) -> list[dict[str, Any]]:
    total_rows = int(profile.get("total_rows", 0) or 0)
    if total_rows <= 0:
        return []
    return [{
        "model_key": config.model_key,
        "total_rows": total_rows,
        "min_date": str(profile.get("min_date") or ""),
        "max_date": str(profile.get("max_date") or ""),
        "prediction_mean": _safe_float(profile.get("prediction_mean")),
        "prediction_std": _safe_float(profile.get("prediction_std")),
        "daily_volume": json.dumps(profile.get("daily_volume") or {}),
        "null_rates": json.dumps(profile.get("null_rates") or {}),
        "computed_at": computed_at,
    }]


def _build_quality_history_row(
    *,
    config: MonitorConfig,
    current_df: pd.DataFrame,
    metadata: dict[str, str],
    computed_at: str,
) -> dict[str, Any]:
    return {
        "model_key": config.model_key,
        "window_id": metadata["window_id"],
        "window_start": metadata["window_start"],
        "window_end": metadata["window_end"],
        "baseline_start": metadata["baseline_start"],
        "baseline_end": metadata["baseline_end"],
        "row_count": int(len(current_df)),
        "prediction_mean": _safe_float(pd.to_numeric(current_df[config.contract.prediction_col], errors="coerce").mean()),
        "prediction_std": _safe_float(pd.to_numeric(current_df[config.contract.prediction_col], errors="coerce").std()),
        "null_rates": json.dumps({
            feature: round(float(current_df[feature].isna().mean() * 100), 2)
            for feature in config.contract.feature_columns
            if feature in current_df.columns
        }),
        "computed_at": computed_at,
    }


def _profile_date_series(frame: pd.DataFrame, timestamp_col: str) -> pd.Series:
    return pd.to_datetime(frame[timestamp_col], errors="coerce").dt.date.astype(str)


def _daily_profile_groups(frame: pd.DataFrame, timestamp_col: str) -> list[tuple[str, pd.DataFrame]]:
    if frame.empty or timestamp_col not in frame.columns:
        return []
    working = frame.copy()
    working["_model_lens_profile_date"] = _profile_date_series(working, timestamp_col)
    working = working[working["_model_lens_profile_date"].notna()]
    if working.empty:
        return []
    return [
        (str(profile_date), group.drop(columns=["_model_lens_profile_date"]).reset_index(drop=True))
        for profile_date, group in working.groupby("_model_lens_profile_date", sort=True)
    ]


def _serialize_numeric_distribution(values: pd.Series, n_bins: int = 20, sample_limit: int = 1000) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return json.dumps({})
    histogram_values = numeric.to_numpy(dtype=float, copy=False)
    if len(histogram_values) == 1:
        value = float(histogram_values[0])
        return json.dumps({"edges": [value, value], "counts": [1], "sample_values": [round(value, 6)]})
    edges = compute_bin_edges(histogram_values, n_bins=min(n_bins, max(2, len(histogram_values))))
    counts, _ = np.histogram(histogram_values, bins=edges)
    sample_values = [round(float(value), 6) for value in histogram_values[:sample_limit].tolist()]
    return json.dumps({
        "edges": [round(float(edge), 6) for edge in edges.tolist()],
        "counts": [int(count) for count in counts.tolist()],
        "sample_values": sample_values,
    })


def _serialize_categorical_distribution(values: pd.Series) -> str:
    categorical = values.astype("string").fillna("__NULL__")
    if categorical.empty:
        return json.dumps({})
    counts = categorical.value_counts(dropna=False).sort_index()
    return json.dumps({str(key): int(value) for key, value in counts.items()})


def _rows_for_date_range(rows: list[dict[str, Any]], start_date: str, end_date: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if start_date <= str(row.get("profile_date") or "") <= end_date
    ]


def _weighted_null_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    total_rows = sum(int(row.get("row_count", 0) or 0) for row in rows)
    if total_rows <= 0:
        return {}
    weighted_sums: dict[str, float] = {}
    for row in rows:
        row_count = int(row.get("row_count", 0) or 0)
        try:
            null_rates = json.loads(str(row.get("null_rates") or "{}"))
        except json.JSONDecodeError:
            null_rates = {}
        if not isinstance(null_rates, dict):
            continue
        for feature, value in null_rates.items():
            try:
                weighted_sums[str(feature)] = weighted_sums.get(str(feature), 0.0) + (float(value) * row_count)
            except (TypeError, ValueError):
                continue
    return {
        feature: round(weighted_sum / total_rows, 2)
        for feature, weighted_sum in weighted_sums.items()
    }


def _combine_mean_std(
    rows: list[dict[str, Any]],
    *,
    count_key: str,
    mean_key: str,
    std_key: str,
    sample_std: bool,
    population_output: bool,
) -> tuple[float | None, float | None, int]:
    total_count = 0
    weighted_mean_sum = 0.0
    for row in rows:
        count = int(row.get(count_key, 0) or 0)
        mean = _safe_float(row.get(mean_key))
        if count <= 0 or mean is None:
            continue
        total_count += count
        weighted_mean_sum += mean * count
    if total_count <= 0:
        return None, None, 0
    combined_mean = weighted_mean_sum / total_count
    variance_numerator = 0.0
    for row in rows:
        count = int(row.get(count_key, 0) or 0)
        mean = _safe_float(row.get(mean_key))
        std = _safe_float(row.get(std_key))
        if count <= 0 or mean is None:
            continue
        if std is not None and count > 1:
            if sample_std:
                variance_numerator += (count - 1) * (std ** 2)
            else:
                variance_numerator += count * (std ** 2)
        variance_numerator += count * ((mean - combined_mean) ** 2)
    if population_output:
        combined_std = float(np.sqrt(variance_numerator / total_count)) if total_count > 0 else None
    else:
        combined_std = float(np.sqrt(variance_numerator / (total_count - 1))) if total_count > 1 else None
    return combined_mean, combined_std, total_count


def _downsample_numeric_values(values: list[float], limit: int = 5000) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)
    if len(values) <= limit:
        return np.asarray(values, dtype=float)
    indices = np.linspace(0, len(values) - 1, num=limit, dtype=int)
    return np.asarray([values[index] for index in indices], dtype=float)


def _aggregate_numeric_profile_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = sum(int(row.get("row_count", 0) or 0) for row in rows)
    non_null_count = sum(int(row.get("non_null_count", 0) or 0) for row in rows)
    mean, std, count = _combine_mean_std(
        rows,
        count_key="non_null_count",
        mean_key="mean",
        std_key="std",
        sample_std=True,
        population_output=True,
    )
    sample_values: list[float] = []
    for row in rows:
        try:
            payload = json.loads(str(row.get("distribution_json") or "{}"))
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            continue
        raw_values = payload.get("sample_values") or []
        if isinstance(raw_values, list):
            for value in raw_values:
                numeric = _safe_float(value)
                if numeric is not None:
                    sample_values.append(numeric)
    return {
        "row_count": total_rows,
        "non_null_count": non_null_count,
        "null_pct": round(float((total_rows - non_null_count) / total_rows * 100), 2) if total_rows else 0.0,
        "mean": mean,
        "std": std,
        "sample_values": _downsample_numeric_values(sample_values),
        "count": count,
    }


def _aggregate_categorical_counts(rows: list[dict[str, Any]]) -> tuple[dict[str, float], int, int]:
    counts: dict[str, float] = {}
    total_rows = 0
    non_null_count = 0
    for row in rows:
        total_rows += int(row.get("row_count", 0) or 0)
        non_null_count += int(row.get("non_null_count", 0) or 0)
        try:
            payload = json.loads(str(row.get("distribution_json") or "{}"))
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            try:
                counts[str(key)] = counts.get(str(key), 0.0) + float(value)
            except (TypeError, ValueError):
                continue
    return counts, total_rows, non_null_count


def _build_window_rows(config: MonitorConfig, metadata_list: list[dict[str, str]], computed_at: str) -> list[dict[str, Any]]:
    return [
        {
            "window_id": metadata["window_id"],
            "model_key": config.model_key,
            "window_grain": metadata["window_grain"],
            "window_start": metadata["window_start"],
            "window_end": metadata["window_end"],
            "baseline_start": metadata["baseline_start"],
            "baseline_end": metadata["baseline_end"],
            "baseline_kind": metadata["baseline_kind"],
            "created_at": computed_at,
        }
        for metadata in metadata_list
    ]


def _derive_quality_history_rows(
    *,
    config: MonitorConfig,
    daily_quality_profile_rows: list[dict[str, Any]],
    metadata_list: list[dict[str, str]],
    computed_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata in metadata_list:
        current_rows = _rows_for_date_range(
            daily_quality_profile_rows,
            metadata["window_start"],
            metadata["window_end"],
        )
        if not current_rows:
            continue
        prediction_mean, prediction_std, _ = _combine_mean_std(
            current_rows,
            count_key="row_count",
            mean_key="prediction_mean",
            std_key="prediction_std",
            sample_std=True,
            population_output=False,
        )
        rows.append({
            "model_key": config.model_key,
            "window_id": metadata["window_id"],
            "window_start": metadata["window_start"],
            "window_end": metadata["window_end"],
            "baseline_start": metadata["baseline_start"],
            "baseline_end": metadata["baseline_end"],
            "row_count": sum(int(row.get("row_count", 0) or 0) for row in current_rows),
            "prediction_mean": prediction_mean,
            "prediction_std": prediction_std,
            "null_rates": json.dumps(_weighted_null_rates(current_rows)),
            "computed_at": computed_at,
        })
    return rows


def _derive_drift_rows(
    *,
    config: MonitorConfig,
    daily_feature_profile_rows: list[dict[str, Any]],
    metadata_list: list[dict[str, str]],
    computed_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profiles_by_feature: dict[str, list[dict[str, Any]]] = {}
    for row in daily_feature_profile_rows:
        profiles_by_feature.setdefault(str(row.get("feature_name") or ""), []).append(row)
    categorical_set = set(config.contract.categorical_columns)
    for metadata in metadata_list:
        for feature in config.contract.feature_columns:
            feature_rows = profiles_by_feature.get(feature, [])
            if not feature_rows:
                continue
            baseline_rows = _rows_for_date_range(feature_rows, metadata["baseline_start"], metadata["baseline_end"])
            current_rows = _rows_for_date_range(feature_rows, metadata["window_start"], metadata["window_end"])
            if not baseline_rows or not current_rows:
                continue
            if feature in categorical_set:
                ref_counts, ref_total, ref_non_null = _aggregate_categorical_counts(baseline_rows)
                cur_counts, cur_total, cur_non_null = _aggregate_categorical_counts(current_rows)
                if not ref_counts or not cur_counts:
                    continue
                psi, kl_divergence, js_divergence = compute_categorical_distribution_metrics(ref_counts, cur_counts)
                rows.extend([
                    {
                        "model_key": config.model_key,
                        "feature_name": feature,
                        "metric_name": metric_name,
                        "metric_value": float(metric_value),
                        "window_id": metadata["window_id"],
                        "window_start": metadata["window_start"],
                        "window_end": metadata["window_end"],
                        "baseline_start": metadata["baseline_start"],
                        "baseline_end": metadata["baseline_end"],
                        "ref_mean": float("nan"),
                        "cur_mean": float("nan"),
                        "ref_std": float("nan"),
                        "cur_std": float("nan"),
                        "ref_null_pct": round(float((ref_total - ref_non_null) / ref_total * 100), 2) if ref_total else 0.0,
                        "cur_null_pct": round(float((cur_total - cur_non_null) / cur_total * 100), 2) if cur_total else 0.0,
                        "ref_count": ref_non_null,
                        "cur_count": cur_non_null,
                        "computed_at": computed_at,
                    }
                    for metric_name, metric_value in (
                        ("psi", psi),
                        ("kl_divergence", kl_divergence),
                        ("js_divergence", js_divergence),
                    )
                ])
                continue
            baseline_stats = _aggregate_numeric_profile_rows(baseline_rows)
            current_stats = _aggregate_numeric_profile_rows(current_rows)
            ref_samples = baseline_stats["sample_values"]
            cur_samples = current_stats["sample_values"]
            if len(ref_samples) < 2 or len(cur_samples) < 2:
                continue
            rows.extend([
                {
                    "model_key": config.model_key,
                    "feature_name": feature,
                    "metric_name": metric_name,
                    "metric_value": round(float(metric_value), 6),
                    "window_id": metadata["window_id"],
                    "window_start": metadata["window_start"],
                    "window_end": metadata["window_end"],
                    "baseline_start": metadata["baseline_start"],
                    "baseline_end": metadata["baseline_end"],
                    "ref_mean": float(baseline_stats["mean"]) if baseline_stats["mean"] is not None else float("nan"),
                    "cur_mean": float(current_stats["mean"]) if current_stats["mean"] is not None else float("nan"),
                    "ref_std": float(baseline_stats["std"]) if baseline_stats["std"] is not None else float("nan"),
                    "cur_std": float(current_stats["std"]) if current_stats["std"] is not None else float("nan"),
                    "ref_null_pct": baseline_stats["null_pct"],
                    "cur_null_pct": current_stats["null_pct"],
                    "ref_count": int(baseline_stats["non_null_count"]),
                    "cur_count": int(current_stats["non_null_count"]),
                    "computed_at": computed_at,
                }
                for metric_name, metric_value in (
                    ("psi", compute_psi(ref_samples, cur_samples)),
                    ("kl_divergence", compute_kl(ref_samples, cur_samples)),
                    ("js_divergence", compute_js(ref_samples, cur_samples)),
                )
            ])
    return rows


def _aggregate_performance_metric_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], tuple[float, int]]:
    totals: dict[tuple[str, str, str], tuple[float, int]] = {}
    for row in rows:
        key = (
            str(row.get("feature_name") or ""),
            str(row.get("bin_label") or ""),
            str(row.get("metric_name") or ""),
        )
        metric_value = _safe_float(row.get("metric_value"))
        row_count = int(row.get("row_count", 0) or 0)
        if metric_value is None or row_count <= 0:
            continue
        weighted_sum, total_rows = totals.get(key, (0.0, 0))
        totals[key] = (weighted_sum + (metric_value * row_count), total_rows + row_count)
    return totals


def _class_mask(
    day_frame: pd.DataFrame,
    class_basis: str,
    class_value: str,
    prediction_col: str,
    label_col: str,
    prediction_score_col: str | None = None,
) -> pd.Series:
    if class_basis == "predicted":
        predictions = resolved_prediction_binary_series(
            day_frame,
            prediction_col=prediction_col,
            prediction_score_col=prediction_score_col,
        )
        if class_value == "positive":
            return predictions == 1
        return predictions == 0
    labels = normalized_binary_series(day_frame[label_col])
    if class_value == "positive":
        return labels == 1
    return labels == 0


def build_performance_bin_specs(
    *,
    config: MonitorConfig,
    inference_df: pd.DataFrame,
    existing_specs: dict[str, tuple[float, ...]] | None = None,
    n_bins: int = 10,
) -> dict[str, tuple[float, ...]]:
    specs = {
        str(feature_name): tuple(float(value) for value in edges)
        for feature_name, edges in (existing_specs or {}).items()
        if feature_name and len(edges) >= 2
    }
    for feature in config.contract.feature_columns:
        if feature not in inference_df.columns or feature in specs:
            continue
        numeric = pd.to_numeric(inference_df[feature], errors="coerce").dropna()
        if len(numeric) < max(2, n_bins):
            continue
        edges = compute_bin_edges(numeric.to_numpy(dtype=float, copy=False), n_bins=n_bins)
        if len(edges) < 2:
            continue
        specs[feature] = tuple(float(value) for value in edges.tolist())
    return specs


def _derive_performance_rows(
    *,
    config: MonitorConfig,
    daily_quality_profile_rows: list[dict[str, Any]],
    daily_performance_profile_rows: list[dict[str, Any]],
    metadata_list: list[dict[str, str]],
    computed_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    regression_mode = (config.problem_type or "classification").strip().lower() == "regression"
    for metadata in metadata_list:
        baseline_daily_rows = _rows_for_date_range(
            daily_performance_profile_rows,
            metadata["baseline_start"],
            metadata["baseline_end"],
        )
        current_daily_rows = _rows_for_date_range(
            daily_performance_profile_rows,
            metadata["window_start"],
            metadata["window_end"],
        )
        if not baseline_daily_rows or not current_daily_rows:
            continue
        baseline_totals = _aggregate_performance_metric_rows(baseline_daily_rows)
        current_totals = _aggregate_performance_metric_rows(current_daily_rows)
        current_quality_rows = _rows_for_date_range(
            daily_quality_profile_rows,
            metadata["window_start"],
            metadata["window_end"],
        )
        total_current_rows = sum(int(row.get("row_count", 0) or 0) for row in current_quality_rows)
        if total_current_rows <= 0:
            total_current_rows = sum(total for _, total in current_totals.values())
        if total_current_rows <= 0:
            continue
        for key in sorted(set(baseline_totals) & set(current_totals)):
            baseline_weighted_sum, baseline_row_count = baseline_totals[key]
            current_weighted_sum, current_row_count = current_totals[key]
            if baseline_row_count <= 0 or current_row_count <= 0:
                continue
            baseline_metric = baseline_weighted_sum / baseline_row_count
            current_metric = current_weighted_sum / current_row_count
            delta = (
                baseline_metric - current_metric
                if regression_mode
                else current_metric - baseline_metric
            )
            volume_pct = round(float(current_row_count / total_current_rows * 100), 2)
            rows.append({
                "model_key": config.model_key,
                "window_id": metadata["window_id"],
                "feature_name": key[0],
                "bin_label": key[1],
                "baseline_metric": round(float(baseline_metric), 4),
                "current_metric": round(float(current_metric), 4),
                "delta": round(float(delta), 4),
                "volume_pct": volume_pct,
                "contribution": round(float(delta * volume_pct / 100), 4),
                "metric_name": key[2],
                "window_start": metadata["window_start"],
                "window_end": metadata["window_end"],
                "computed_at": computed_at,
            })
    return rows


def build_daily_quality_profile_rows(
    *,
    config: MonitorConfig,
    inference_df: pd.DataFrame,
    computed_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile_date, day_frame in _daily_profile_groups(inference_df, config.contract.timestamp_col):
        rows.append({
            "model_key": config.model_key,
            "profile_date": profile_date,
            "row_count": int(len(day_frame)),
            "prediction_mean": _safe_float(pd.to_numeric(day_frame[config.contract.prediction_col], errors="coerce").mean()),
            "prediction_std": _safe_float(pd.to_numeric(day_frame[config.contract.prediction_col], errors="coerce").std()),
            "null_rates": json.dumps({
                feature: round(float(day_frame[feature].isna().mean() * 100), 2)
                for feature in config.contract.feature_columns
                if feature in day_frame.columns
            }),
            "label_row_count": (
                int(day_frame[config.contract.label_col].notna().sum())
                if config.contract.label_col and config.contract.label_col in day_frame.columns
                else 0
            ),
            "computed_at": computed_at,
        })
    return rows


def build_daily_class_quality_profile_rows(
    *,
    config: MonitorConfig,
    inference_df: pd.DataFrame,
    computed_at: str,
) -> list[dict[str, Any]]:
    if not supports_binary_class_filters(config) or not config.contract.label_col:
        return []
    rows: list[dict[str, Any]] = []
    label_col = config.contract.label_col
    prediction_col = config.contract.prediction_col
    for profile_date, day_frame in _daily_profile_groups(inference_df, config.contract.timestamp_col):
        for class_basis in ("actual", "predicted"):
            for class_value in ("positive", "negative"):
                mask = _class_mask(
                    day_frame,
                    class_basis,
                    class_value,
                    prediction_col,
                    label_col,
                    config.contract.prediction_score_col,
                )
                filtered = day_frame[mask.fillna(False)]
                if filtered.empty:
                    continue
                rows.append({
                    "model_key": config.model_key,
                    "profile_date": profile_date,
                    "class_basis": class_basis,
                    "class_value": class_value,
                    "row_count": int(len(filtered)),
                    "prediction_mean": _safe_float(pd.to_numeric(filtered[prediction_col], errors="coerce").mean()),
                    "prediction_std": _safe_float(pd.to_numeric(filtered[prediction_col], errors="coerce").std()),
                    "null_rates": json.dumps({
                        feature: round(float(filtered[feature].isna().mean() * 100), 2)
                        for feature in config.contract.feature_columns
                        if feature in filtered.columns
                    }),
                    "label_row_count": int(filtered[label_col].notna().sum()),
                    "computed_at": computed_at,
                })
    return rows


def build_daily_feature_profile_rows(
    *,
    config: MonitorConfig,
    inference_df: pd.DataFrame,
    computed_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    categorical_set = set(config.contract.categorical_columns)
    for profile_date, day_frame in _daily_profile_groups(inference_df, config.contract.timestamp_col):
        for feature in config.contract.feature_columns:
            if feature not in day_frame.columns:
                continue
            series = day_frame[feature]
            row_count = int(len(series))
            feature_kind = "categorical" if feature in categorical_set else "numeric"
            if feature_kind == "categorical":
                rows.append({
                    "model_key": config.model_key,
                    "profile_date": profile_date,
                    "feature_name": feature,
                    "feature_kind": feature_kind,
                    "row_count": row_count,
                    "non_null_count": int(series.notna().sum()),
                    "null_pct": round(float(series.isna().mean() * 100), 2) if row_count else 0.0,
                    "mean": None,
                    "std": None,
                    "min_value": None,
                    "max_value": None,
                    "distribution_json": _serialize_categorical_distribution(series),
                    "computed_at": computed_at,
                })
                continue
            numeric = pd.to_numeric(series, errors="coerce")
            rows.append({
                "model_key": config.model_key,
                "profile_date": profile_date,
                "feature_name": feature,
                "feature_kind": feature_kind,
                "row_count": row_count,
                "non_null_count": int(numeric.notna().sum()),
                "null_pct": round(float(numeric.isna().mean() * 100), 2) if row_count else 0.0,
                "mean": _safe_float(numeric.mean()),
                "std": _safe_float(numeric.std()),
                "min_value": _safe_float(numeric.min()),
                "max_value": _safe_float(numeric.max()),
                "distribution_json": _serialize_numeric_distribution(numeric),
                "computed_at": computed_at,
            })
    return rows


def build_daily_class_feature_profile_rows(
    *,
    config: MonitorConfig,
    inference_df: pd.DataFrame,
    computed_at: str,
) -> list[dict[str, Any]]:
    if not supports_binary_class_filters(config) or not config.contract.label_col:
        return []
    rows: list[dict[str, Any]] = []
    categorical_set = set(config.contract.categorical_columns)
    label_col = config.contract.label_col
    prediction_col = config.contract.prediction_col
    for profile_date, day_frame in _daily_profile_groups(inference_df, config.contract.timestamp_col):
        for class_basis in ("actual", "predicted"):
            for class_value in ("positive", "negative"):
                mask = _class_mask(
                    day_frame,
                    class_basis,
                    class_value,
                    prediction_col,
                    label_col,
                    config.contract.prediction_score_col,
                )
                filtered = day_frame[mask.fillna(False)]
                if filtered.empty:
                    continue
                for feature in config.contract.feature_columns:
                    if feature not in filtered.columns:
                        continue
                    series = filtered[feature]
                    row_count = int(len(series))
                    feature_kind = "categorical" if feature in categorical_set else "numeric"
                    if feature_kind == "categorical":
                        rows.append({
                            "model_key": config.model_key,
                            "profile_date": profile_date,
                            "class_basis": class_basis,
                            "class_value": class_value,
                            "feature_name": feature,
                            "feature_kind": feature_kind,
                            "row_count": row_count,
                            "non_null_count": int(series.notna().sum()),
                            "null_pct": round(float(series.isna().mean() * 100), 2) if row_count else 0.0,
                            "mean": None,
                            "std": None,
                            "min_value": None,
                            "max_value": None,
                            "distribution_json": _serialize_categorical_distribution(series),
                            "computed_at": computed_at,
                        })
                        continue
                    numeric = pd.to_numeric(series, errors="coerce")
                    rows.append({
                        "model_key": config.model_key,
                        "profile_date": profile_date,
                        "class_basis": class_basis,
                        "class_value": class_value,
                        "feature_name": feature,
                        "feature_kind": feature_kind,
                        "row_count": row_count,
                        "non_null_count": int(numeric.notna().sum()),
                        "null_pct": round(float(numeric.isna().mean() * 100), 2) if row_count else 0.0,
                        "mean": _safe_float(numeric.mean()),
                        "std": _safe_float(numeric.std()),
                        "min_value": _safe_float(numeric.min()),
                        "max_value": _safe_float(numeric.max()),
                        "distribution_json": _serialize_numeric_distribution(numeric),
                        "computed_at": computed_at,
                    })
    return rows


def build_daily_label_metric_rows(
    *,
    config: MonitorConfig,
    inference_df: pd.DataFrame,
    computed_at: str,
) -> list[dict[str, Any]]:
    if not supports_binary_class_filters(config) or not config.contract.label_col:
        return []
    rows: list[dict[str, Any]] = []
    for profile_date, day_frame in _daily_profile_groups(inference_df, config.contract.timestamp_col):
        metrics = compute_daily_classification_metrics(
            day_frame,
            config.contract.prediction_col,
            config.contract.label_col,
            prediction_score_col=config.contract.prediction_score_col,
        )
        rows.append({
            "model_key": config.model_key,
            "profile_date": profile_date,
            "actual_positive_count": int(metrics["actual_positive_count"] or 0),
            "actual_negative_count": int(metrics["actual_negative_count"] or 0),
            "predicted_positive_count": int(metrics["predicted_positive_count"] or 0),
            "predicted_negative_count": int(metrics["predicted_negative_count"] or 0),
            "tp": int(metrics["tp"] or 0),
            "fp": int(metrics["fp"] or 0),
            "fn": int(metrics["fn"] or 0),
            "tn": int(metrics["tn"] or 0),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "computed_at": computed_at,
        })
    return rows


def build_daily_performance_profile_rows(
    *,
    config: MonitorConfig,
    inference_df: pd.DataFrame,
    computed_at: str,
    bin_specs: dict[str, tuple[float, ...]] | None = None,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    if not config.contract.label_col:
        return []
    rows: list[dict[str, Any]] = []
    regression_mode = (config.problem_type or "classification").strip().lower() == "regression"
    selected_metric_names = tuple(config.performance_metric_names or default_performance_metric_names(config.problem_type))
    feature_edges: dict[str, np.ndarray] = {}
    if bin_specs is None:
        resolved_bin_specs = build_performance_bin_specs(
            config=config,
            inference_df=inference_df,
            existing_specs=None,
            n_bins=n_bins,
        )
    else:
        resolved_bin_specs = {
            str(feature_name): tuple(float(value) for value in edges)
            for feature_name, edges in bin_specs.items()
            if feature_name and len(edges) >= 2
        }
    for feature, edges in resolved_bin_specs.items():
        if feature not in inference_df.columns:
            continue
        feature_edges[feature] = np.asarray(edges, dtype=float)
    for profile_date, day_frame in _daily_profile_groups(inference_df, config.contract.timestamp_col):
        total_rows = max(len(day_frame), 1)
        for feature, edges in feature_edges.items():
            if feature not in day_frame.columns:
                continue
            numeric = pd.to_numeric(day_frame[feature], errors="coerce")
            bins = np.digitize(numeric.to_numpy(dtype=float, copy=False), edges[1:-1], right=False)
            for index in range(len(edges) - 1):
                day_slice = day_frame.iloc[np.where(bins == index)[0]]
                if day_slice.empty:
                    continue
                metrics = (
                    compute_regression_metrics(day_slice, config.contract.prediction_col, config.contract.label_col)
                    if regression_mode
                    else compute_classification_metrics(
                        day_slice,
                        config.contract.prediction_col,
                        config.contract.label_col,
                        prediction_score_col=config.contract.prediction_score_col,
                    )
                )
                if not metrics:
                    continue
                for metric_name in selected_metric_names:
                    metric_value = metrics.get(metric_name)
                    if metric_value is None:
                        continue
                    rows.append({
                        "model_key": config.model_key,
                        "profile_date": profile_date,
                        "feature_name": feature,
                        "bin_label": f"[{edges[index]:.4g}, {edges[index + 1]:.4g})",
                        "metric_name": metric_name,
                        "metric_value": float(metric_value),
                        "row_count": int(len(day_slice)),
                        "volume_pct": round(float(len(day_slice) / total_rows * 100), 2),
                        "computed_at": computed_at,
                    })
    return rows


def refresh_monitor_backfill(
    config: MonitorConfig,
    inference_df: pd.DataFrame,
    *,
    existing_window_keys: set[WindowKey] | None = None,
    prior_open_incidents: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    include_drift_quality: bool = True,
    include_performance: bool = True,
) -> RefreshResult:
    pairs = generate_window_pairs(
        inference_df,
        config.contract.timestamp_col,
        config.baseline,
        model_key=config.model_key,
        existing_window_keys=existing_window_keys,
    )
    if not pairs:
        return RefreshResult(
            drift_rows=[],
            quality_rows=[],
            performance_rows=[],
            incident_rows=[],
            incident_history_rows=[],
            quality_history_rows=[],
            window_rows=[],
        )

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    all_drift_rows: list[dict[str, Any]] = []
    all_incident_history_rows: list[dict[str, Any]] = []
    all_quality_history_rows: list[dict[str, Any]] = []
    all_performance_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    latest_window_end = ""
    latest_window_drift_rows: list[dict[str, Any]] = []

    for baseline_df, current_df, metadata in pairs:
        if include_drift_quality:
            all_window_rows.append({
                "window_id": metadata["window_id"],
                "model_key": config.model_key,
                "window_grain": metadata["window_grain"],
                "window_start": metadata["window_start"],
                "window_end": metadata["window_end"],
                "baseline_start": metadata["baseline_start"],
                "baseline_end": metadata["baseline_end"],
                "baseline_kind": metadata["baseline_kind"],
                "created_at": now,
            })
            all_quality_history_rows.append(
                _build_quality_history_row(
                    config=config,
                    current_df=current_df,
                    metadata=metadata,
                    computed_at=now,
                )
            )
            drift_metrics = compute_feature_drift(
                baseline_df,
                current_df,
                list(config.contract.feature_columns),
                list(config.contract.categorical_columns),
            )
            window_drift_rows = _build_drift_rows(
                config=config,
                drift_metrics=drift_metrics,
                metadata=metadata,
                computed_at=now,
            )
            all_drift_rows.extend(window_drift_rows)
            if metadata["window_end"] >= latest_window_end:
                latest_window_end = metadata["window_end"]
                latest_window_drift_rows = window_drift_rows

        if include_performance and config.contract.label_col and config.contract.label_col in current_df.columns:
            performance_frame = rank_degradation_contributors(
                baseline_df=baseline_df,
                current_df=current_df,
                feature_columns=list(config.contract.feature_columns),
                prediction_col=config.contract.prediction_col,
                label_col=config.contract.label_col,
                prediction_score_col=config.contract.prediction_score_col,
                metric_names=config.performance_metric_names,
                problem_type=config.problem_type,
            )
            all_performance_rows.extend(
                _build_performance_rows(
                    config=config,
                    performance_frame=performance_frame,
                    metadata=metadata,
                    computed_at=now,
                )
            )

    if include_drift_quality and not all_drift_rows:
        return RefreshResult(
            drift_rows=[],
            quality_rows=_build_quality_rows(config=config, inference_df=inference_df, computed_at=now) if include_drift_quality else [],
            performance_rows=[],
            incident_rows=[],
            incident_history_rows=[],
            quality_history_rows=all_quality_history_rows if include_drift_quality else [],
            window_rows=all_window_rows if include_drift_quality else [],
        )

    if include_drift_quality:
        all_incident_history_rows = build_incident_history(
            all_drift_rows,
            all_window_rows,
            prior_open_incidents=prior_open_incidents,
            thresholds=config.threshold_overrides,
        )

    return RefreshResult(
        drift_rows=all_drift_rows,
        quality_rows=_build_quality_rows(config=config, inference_df=inference_df, computed_at=now) if include_drift_quality else [],
        performance_rows=all_performance_rows,
        incident_rows=build_incidents(latest_window_drift_rows, thresholds=config.threshold_overrides) if include_drift_quality else [],
        incident_history_rows=all_incident_history_rows,
        quality_history_rows=all_quality_history_rows,
        window_rows=all_window_rows,
    )


def refresh_monitor(config: MonitorConfig, inference_df: pd.DataFrame) -> RefreshResult:
    result = refresh_monitor_backfill(config, inference_df)
    if not result.drift_rows:
        return result
    latest_window_end = max(str(row["window_end"]) for row in result.drift_rows)
    latest_drift_rows = [row for row in result.drift_rows if str(row["window_end"]) == latest_window_end]
    latest_performance_rows = [row for row in result.performance_rows if str(row["window_end"]) == latest_window_end]
    return RefreshResult(
        drift_rows=latest_drift_rows,
        quality_rows=result.quality_rows,
        performance_rows=latest_performance_rows,
        incident_rows=build_incidents(latest_drift_rows, thresholds=config.threshold_overrides),
        incident_history_rows=[
            row for row in result.incident_history_rows if str(row["window_end"]) == latest_window_end
        ],
        quality_history_rows=[
            row for row in result.quality_history_rows if str(row["window_end"]) == latest_window_end
        ],
        window_rows=[row for row in result.window_rows if str(row["window_end"]) == latest_window_end],
    )


def derive_refresh_result_from_daily_profiles(
    *,
    config: MonitorConfig,
    metadata_list: list[dict[str, str]],
    daily_quality_profile_rows: list[dict[str, Any]],
    daily_feature_profile_rows: list[dict[str, Any]],
    daily_performance_profile_rows: list[dict[str, Any]],
    daily_class_quality_profile_rows: list[dict[str, Any]] | None = None,
    daily_class_feature_profile_rows: list[dict[str, Any]] | None = None,
    daily_label_metric_rows: list[dict[str, Any]] | None = None,
    computed_at: str,
    prior_open_incidents: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    include_drift_quality: bool = True,
    include_performance: bool = True,
) -> RefreshResult:
    window_rows = _build_window_rows(config, metadata_list, computed_at) if include_drift_quality else []
    quality_history_rows = (
        _derive_quality_history_rows(
            config=config,
            daily_quality_profile_rows=daily_quality_profile_rows,
            metadata_list=metadata_list,
            computed_at=computed_at,
        )
        if include_drift_quality
        else []
    )
    drift_rows = (
        _derive_drift_rows(
            config=config,
            daily_feature_profile_rows=daily_feature_profile_rows,
            metadata_list=metadata_list,
            computed_at=computed_at,
        )
        if include_drift_quality
        else []
    )
    performance_rows = (
        _derive_performance_rows(
            config=config,
            daily_quality_profile_rows=daily_quality_profile_rows,
            daily_performance_profile_rows=daily_performance_profile_rows,
            metadata_list=metadata_list,
            computed_at=computed_at,
        )
        if include_performance and config.contract.label_col
        else []
    )
    latest_window_end = max((metadata["window_end"] for metadata in metadata_list), default="")
    latest_drift_rows = [row for row in drift_rows if str(row["window_end"]) == latest_window_end]
    incident_rows = build_incidents(latest_drift_rows, thresholds=config.threshold_overrides) if include_drift_quality else []
    incident_history_rows = (
        build_incident_history(
            drift_rows,
            window_rows,
            prior_open_incidents=prior_open_incidents,
            thresholds=config.threshold_overrides,
        )
        if include_drift_quality
        else []
    )
    return RefreshResult(
        drift_rows=drift_rows,
        quality_rows=[],
        performance_rows=performance_rows,
        incident_rows=incident_rows,
        incident_history_rows=incident_history_rows,
        quality_history_rows=quality_history_rows,
        window_rows=window_rows,
        daily_class_quality_profile_rows=list(daily_class_quality_profile_rows or []),
        daily_class_feature_profile_rows=list(daily_class_feature_profile_rows or []),
        daily_label_metric_rows=list(daily_label_metric_rows or []),
    )


def refresh_monitor_window_frames(
    config: MonitorConfig,
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    metadata: dict[str, str],
    *,
    include_drift_quality: bool = True,
    include_performance: bool = True,
) -> RefreshResult:
    if baseline_df.empty or current_df.empty:
        return RefreshResult(
            drift_rows=[],
            quality_rows=[],
            performance_rows=[],
            incident_rows=[],
            incident_history_rows=[],
            quality_history_rows=[],
            window_rows=[],
        )

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    drift_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    quality_history_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    if include_drift_quality:
        window_rows.append({
            "window_id": metadata["window_id"],
            "model_key": config.model_key,
            "window_grain": metadata["window_grain"],
            "window_start": metadata["window_start"],
            "window_end": metadata["window_end"],
            "baseline_start": metadata["baseline_start"],
            "baseline_end": metadata["baseline_end"],
            "baseline_kind": metadata["baseline_kind"],
            "created_at": now,
        })
        quality_history_rows.append(
            _build_quality_history_row(
                config=config,
                current_df=current_df,
                metadata=metadata,
                computed_at=now,
            )
        )
        drift_metrics = compute_feature_drift(
            baseline_df,
            current_df,
            list(config.contract.feature_columns),
            list(config.contract.categorical_columns),
        )
        drift_rows = _build_drift_rows(
            config=config,
            drift_metrics=drift_metrics,
            metadata=metadata,
            computed_at=now,
        )

    if include_performance and config.contract.label_col and config.contract.label_col in current_df.columns:
        performance_frame = rank_degradation_contributors(
            baseline_df=baseline_df,
            current_df=current_df,
            feature_columns=list(config.contract.feature_columns),
            prediction_col=config.contract.prediction_col,
            label_col=config.contract.label_col,
            prediction_score_col=config.contract.prediction_score_col,
            metric_names=config.performance_metric_names,
            problem_type=config.problem_type,
        )
        performance_rows = _build_performance_rows(
            config=config,
            performance_frame=performance_frame,
            metadata=metadata,
            computed_at=now,
        )

    return RefreshResult(
        drift_rows=drift_rows,
        quality_rows=[],
        performance_rows=performance_rows,
        incident_rows=build_incidents(drift_rows, thresholds=config.threshold_overrides) if drift_rows else [],
        incident_history_rows=[],
        quality_history_rows=quality_history_rows,
        window_rows=window_rows,
    )
