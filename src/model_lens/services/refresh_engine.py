from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from model_lens.analytics.drift import compute_feature_drift
from model_lens.analytics.performance import rank_degradation_contributors
from model_lens.domain.models import BaselinePolicy, MonitorConfig, RefreshResult
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
        "feature_name": row["feature_name"],
        "bin_label": row["bin_label"],
        "baseline_metric": float(row["baseline_metric"]),
        "current_metric": float(row["current_metric"]),
        "delta": float(row["delta"]),
        "volume_pct": float(row["volume_pct"]),
        "contribution": float(row["contribution"]),
        "metric_name": "f1",
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


def refresh_monitor_backfill(
    config: MonitorConfig,
    inference_df: pd.DataFrame,
    *,
    existing_window_keys: set[WindowKey] | None = None,
    prior_open_incidents: dict[tuple[str, str, str], dict[str, Any]] | None = None,
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

        if config.contract.label_col and config.contract.label_col in current_df.columns:
            performance_frame = rank_degradation_contributors(
                baseline_df=baseline_df,
                current_df=current_df,
                feature_columns=list(config.contract.feature_columns),
                prediction_col=config.contract.prediction_col,
                label_col=config.contract.label_col,
            )
            all_performance_rows.extend(
                _build_performance_rows(
                    config=config,
                    performance_frame=performance_frame,
                    metadata=metadata,
                    computed_at=now,
                )
            )

    if not all_drift_rows:
        return RefreshResult(
            drift_rows=[],
            quality_rows=[],
            performance_rows=[],
            incident_rows=[],
            incident_history_rows=[],
            quality_history_rows=all_quality_history_rows,
            window_rows=all_window_rows,
        )

    all_incident_history_rows = build_incident_history(
        all_drift_rows,
        all_window_rows,
        prior_open_incidents=prior_open_incidents,
    )

    return RefreshResult(
        drift_rows=all_drift_rows,
        quality_rows=_build_quality_rows(config=config, inference_df=inference_df, computed_at=now),
        performance_rows=all_performance_rows,
        incident_rows=build_incidents(latest_window_drift_rows),
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
        incident_rows=build_incidents(latest_drift_rows),
        incident_history_rows=[
            row for row in result.incident_history_rows if str(row["window_end"]) == latest_window_end
        ],
        quality_history_rows=[
            row for row in result.quality_history_rows if str(row["window_end"]) == latest_window_end
        ],
        window_rows=[row for row in result.window_rows if str(row["window_end"]) == latest_window_end],
    )
