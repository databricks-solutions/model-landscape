from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from model_lens.analytics.drift import compute_feature_drift
from model_lens.analytics.performance import rank_degradation_contributors
from model_lens.domain.models import MonitorConfig, RefreshResult
from model_lens.services.incidents import build_incidents


def _safe_float(value: object) -> float | None:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return None
    return float(converted)


def split_baseline_current(df: pd.DataFrame, timestamp_col: str, n_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.copy()
    ordered[timestamp_col] = pd.to_datetime(ordered[timestamp_col])
    ordered = ordered.sort_values(timestamp_col)
    if ordered.empty:
        return ordered, ordered
    baseline_start = ordered[timestamp_col].min().normalize()
    baseline_end = baseline_start + timedelta(days=max(n_days - 1, 0))
    baseline = ordered[ordered[timestamp_col].dt.normalize() <= baseline_end]
    current = ordered[ordered[timestamp_col].dt.normalize() > baseline_end]
    return baseline, current


def refresh_monitor(config: MonitorConfig, inference_df: pd.DataFrame) -> RefreshResult:
    baseline_df, current_df = split_baseline_current(
        inference_df,
        config.contract.timestamp_col,
        config.baseline.n_days,
    )
    if baseline_df.empty or current_df.empty:
        return RefreshResult(drift_rows=[], quality_rows=[], performance_rows=[], incident_rows=[])

    drift_metrics = compute_feature_drift(
        baseline_df,
        current_df,
        list(config.contract.feature_columns),
    )

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    baseline_start = str(pd.to_datetime(baseline_df[config.contract.timestamp_col]).min().date())
    baseline_end = str(pd.to_datetime(baseline_df[config.contract.timestamp_col]).max().date())
    current_end = str(pd.to_datetime(current_df[config.contract.timestamp_col]).max().date())

    drift_rows: list[dict] = []
    for _, row in drift_metrics.iterrows():
        for metric_name in ("psi", "kl_divergence", "js_divergence"):
            drift_rows.append({
                "model_key": config.model_key,
                "feature_name": row["feature_name"],
                "metric_name": metric_name,
                "metric_value": float(row[metric_name]),
                "window_start": baseline_end,
                "window_end": current_end,
                "baseline_start": baseline_start,
                "baseline_end": baseline_end,
                "ref_mean": float(row["ref_mean"]),
                "cur_mean": float(row["cur_mean"]),
                "ref_std": float(row["ref_std"]),
                "cur_std": float(row["cur_std"]),
                "ref_null_pct": float(row["ref_null_pct"]),
                "cur_null_pct": float(row["cur_null_pct"]),
                "ref_count": int(row["ref_count"]),
                "cur_count": int(row["cur_count"]),
                "computed_at": now,
            })

    quality_rows = [{
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
        "computed_at": now,
    }]

    performance_rows: list[dict] = []
    if config.contract.label_col and config.contract.label_col in inference_df.columns:
        performance_frame = rank_degradation_contributors(
            baseline_df=baseline_df,
            current_df=current_df,
            feature_columns=list(config.contract.feature_columns),
            prediction_col=config.contract.prediction_col,
            label_col=config.contract.label_col,
        )
        performance_rows = [{
            "model_key": config.model_key,
            "feature_name": row["feature_name"],
            "bin_label": row["bin_label"],
            "baseline_metric": float(row["baseline_metric"]),
            "current_metric": float(row["current_metric"]),
            "delta": float(row["delta"]),
            "volume_pct": float(row["volume_pct"]),
            "contribution": float(row["contribution"]),
            "metric_name": "f1",
            "window_start": baseline_end,
            "window_end": current_end,
            "computed_at": now,
        } for _, row in performance_frame.iterrows()]

    incident_rows = build_incidents(drift_rows)
    return RefreshResult(
        drift_rows=drift_rows,
        quality_rows=quality_rows,
        performance_rows=performance_rows,
        incident_rows=incident_rows,
    )
