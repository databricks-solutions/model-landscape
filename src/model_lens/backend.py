from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from datetime import timezone
from threading import Lock
from time import monotonic

import numpy as np
import pandas as pd

from model_lens.analytics.performance import compute_classification_metrics, compute_regression_metrics
from model_lens.config import settings
from model_lens.domain.models import MonitorConfig, MonitorDiscoveryResult, MonitorRuntimeState
from model_lens.services.class_filters import normalize_class_filter, supports_binary_class_filters
from model_lens.services.control_plane import ControlPlaneRepository, build_repository
from model_lens.services.monitor_discovery import MonitorDiscoveryService
from model_lens.services.onboarding import baseline_label
from model_lens.services.refresh_diagnostics import build_refresh_diagnostics
from model_lens.services.refresh_engine import (
    build_daily_class_feature_profile_rows,
    derive_refresh_result_from_daily_profiles,
    split_baseline_current,
)
from model_lens.services.refresh_jobs import resolve_shared_workflow_schedule_status
from model_lens.services.thresholds import get_thresholds, merged_thresholds


logger = logging.getLogger(__name__)

_MAX_DASHBOARD_WINDOW_HISTORY = 400
_MAX_DASHBOARD_PERFORMANCE_WINDOWS = 180
_MAX_DASHBOARD_DAILY_PROFILE_DAYS = 400
_EXACT_SOURCE_DAILY_METRIC_CACHE_TTL_SECONDS = 30.0
_EXACT_SOURCE_DAILY_METRIC_CACHE: dict[tuple[str, str, str, str], tuple[float, pd.DataFrame]] = {}
_EXACT_SOURCE_DAILY_METRIC_CACHE_LOCK = Lock()


def _safe_json_dict(value: object) -> dict:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def clear_exact_source_daily_metric_cache() -> None:
    with _EXACT_SOURCE_DAILY_METRIC_CACHE_LOCK:
        _EXACT_SOURCE_DAILY_METRIC_CACHE.clear()


def _safe_json_list(value: object) -> list[float]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        numeric: list[float] = []
        for item in value:
            series = pd.to_numeric(pd.Series([item]), errors="coerce").dropna()
            if not series.empty:
                numeric.append(float(series.iloc[0]))
        return numeric
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    numeric: list[float] = []
    for item in parsed:
        series = pd.to_numeric(pd.Series([item]), errors="coerce").dropna()
        if not series.empty:
            numeric.append(float(series.iloc[0]))
    return numeric


def _approximate_histogram_values(
    edges: list[float],
    counts: list[float],
    *,
    max_points: int = 256,
) -> list[float]:
    if len(edges) < 2 or len(counts) != len(edges) - 1:
        return []
    positive_bins = [
        (index, max(float(count), 0.0))
        for index, count in enumerate(counts)
        if float(count) > 0
    ]
    if not positive_bins:
        return []
    total = sum(count for _, count in positive_bins)
    if total <= 0:
        return []
    values: list[float] = []
    allocated = 0
    for index, count in positive_bins:
        left = float(edges[index])
        right = float(edges[index + 1])
        midpoint = (left + right) / 2.0
        share = max(1, int(round((count / total) * max_points)))
        remaining = max_points - allocated
        repeats = min(share, max(remaining, 0))
        if repeats <= 0:
            continue
        values.extend([midpoint] * repeats)
        allocated += repeats
        if allocated >= max_points:
            break
    return values


def _safe_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return 0.0
    return float(numeric)


def _safe_optional_float(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _safe_int(value: object) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return 0
    return int(numeric)


def _non_empty_text_bounds(values: list[object] | tuple[object, ...]) -> tuple[str | None, str | None]:
    normalized = sorted(str(value).strip() for value in values if str(value).strip())
    if not normalized:
        return None, None
    return normalized[0], normalized[-1]


def _combine_weighted_mean_std(parts: list[tuple[int, float | None, float | None]]) -> tuple[float | None, float | None]:
    valid_parts = [(count, mean, std) for count, mean, std in parts if count > 0 and mean is not None]
    if not valid_parts:
        return None, None
    total_count = sum(count for count, _, _ in valid_parts)
    if total_count <= 0:
        return None, None
    combined_mean = sum(count * float(mean) for count, mean, _ in valid_parts) / total_count
    if total_count <= 1:
        return combined_mean, None
    total_ss = 0.0
    for count, mean, std in valid_parts:
        local_ss = 0.0
        if std is not None and count > 1:
            local_ss = (count - 1) * (float(std) ** 2)
        total_ss += local_ss + (count * ((float(mean) - combined_mean) ** 2))
    combined_std = (total_ss / (total_count - 1)) ** 0.5 if total_ss > 0 else 0.0
    return combined_mean, combined_std


def _safe_series_min(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.min())


def _apply_feature_outlier_filter(
    frame: pd.DataFrame,
    feature: str,
    *,
    mode: str,
    value: float | None,
) -> pd.DataFrame:
    if frame.empty or feature not in frame.columns:
        return pd.DataFrame(columns=frame.columns)
    numeric = pd.to_numeric(frame[feature], errors="coerce")
    valid_mask = numeric.notna()
    if not valid_mask.any():
        return frame.iloc[0:0].copy()
    clean = numeric.loc[valid_mask].to_numpy(dtype=float, copy=False)
    normalized_mode = str(mode or "off").strip().lower()
    if normalized_mode == "percentile_clip" and value is not None and value > 0:
        lower = float(np.nanpercentile(clean, value))
        upper = float(np.nanpercentile(clean, 100.0 - value))
        valid_mask &= numeric.between(lower, upper, inclusive="both")
    elif normalized_mode == "iqr_fence" and value is not None and value > 0:
        q1 = float(np.nanpercentile(clean, 25.0))
        q3 = float(np.nanpercentile(clean, 75.0))
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            lower = q1 - (float(value) * iqr)
            upper = q3 + (float(value) * iqr)
            valid_mask &= numeric.between(lower, upper, inclusive="both")
    return frame.loc[valid_mask].copy()


def _resolve_dynamic_bin_edges(
    baseline_values: np.ndarray,
    current_values: np.ndarray,
    *,
    binning_mode: str,
    n_bins: int,
    custom_edges: list[float] | None,
) -> np.ndarray:
    normalized_mode = str(binning_mode or "auto").strip().lower()
    if normalized_mode == "custom" and custom_edges and len(custom_edges) >= 2:
        return np.asarray(custom_edges, dtype=float)
    combined = np.concatenate([values for values in (baseline_values, current_values) if values.size > 0])
    if combined.size == 0:
        return np.asarray([], dtype=float)
    if normalized_mode == "fixed":
        return np.histogram_bin_edges(combined, bins=max(int(n_bins), 2))
    return np.histogram_bin_edges(combined, bins="auto")


def _aggregated_classification_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame is None or frame.empty:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "accuracy": None,
        }
    tp = int(pd.to_numeric(frame.get("tp", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    fp = int(pd.to_numeric(frame.get("fp", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    fn = int(pd.to_numeric(frame.get("fn", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    tn = int(pd.to_numeric(frame.get("tn", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    accuracy_denominator = tp + fp + fn + tn
    accuracy = ((tp + tn) / accuracy_denominator) if accuracy_denominator > 0 else None
    return {
        "precision": round(float(precision), 4) if precision is not None else None,
        "recall": round(float(recall), 4) if recall is not None else None,
        "f1": round(float(f1), 4) if f1 is not None else None,
        "accuracy": round(float(accuracy), 4) if accuracy is not None else None,
    }


def _weighted_average(values: pd.Series, weights: pd.Series) -> float | None:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    valid = numeric_values.notna() & numeric_weights.notna() & (numeric_weights > 0)
    if not valid.any():
        return None
    weighted_sum = float((numeric_values[valid] * numeric_weights[valid]).sum())
    total_weight = float(numeric_weights[valid].sum())
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _null_rate_dict(value: object) -> dict[str, float]:
    parsed = _safe_json_dict(value)
    rates: dict[str, float] = {}
    for key, raw_value in parsed.items():
        numeric = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            continue
        rates[str(key)] = float(numeric)
    return rates


def _quality_history_from_daily_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["profile_date_ts"] = pd.to_datetime(working["profile_date"], errors="coerce")
    working["period"] = working["profile_date_ts"].dt.date.astype(str)
    working["window_start"] = working["period"]
    working["window_end"] = working["period"]
    working["baseline_start"] = ""
    working["baseline_end"] = ""
    working["window_id"] = working["period"].apply(lambda value: f"daily_profile|{value}")
    working["row_count"] = pd.to_numeric(working["row_count"], errors="coerce").fillna(0).astype(int)
    working["prediction_mean"] = pd.to_numeric(working["prediction_mean"], errors="coerce")
    working["prediction_std"] = pd.to_numeric(working["prediction_std"], errors="coerce")
    working["null_rates_dict"] = working["null_rates"].apply(_null_rate_dict)
    working["max_null_rate"] = working["null_rates_dict"].apply(
        lambda values: max(values.values()) if values else 0.0
    )
    working["computed_at_ts"] = pd.to_datetime(working.get("computed_at"), errors="coerce")
    working = working.sort_values(["period", "computed_at_ts", "profile_date_ts"]).drop_duplicates("period", keep="last")
    return working.sort_values("profile_date_ts").reset_index(drop=True)


def _quality_summary_from_daily_profiles(model_key: str, rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    frame["profile_date_ts"] = pd.to_datetime(frame["profile_date"], errors="coerce")
    frame = frame[frame["profile_date_ts"].notna()].copy()
    if frame.empty:
        return {}
    frame["row_count"] = pd.to_numeric(frame["row_count"], errors="coerce").fillna(0).astype(int)
    total_rows = int(frame["row_count"].sum())
    if total_rows <= 0:
        return {}
    frame["prediction_mean"] = pd.to_numeric(frame["prediction_mean"], errors="coerce")
    frame["prediction_std"] = pd.to_numeric(frame["prediction_std"], errors="coerce")
    prediction_mean, prediction_std = _combine_weighted_mean_std(
        [
            (
                int(row["row_count"]),
                _safe_optional_float(row["prediction_mean"]),
                _safe_optional_float(row["prediction_std"]),
            )
            for _, row in frame.iterrows()
        ]
    )
    daily_volume = {
        str(row["profile_date_ts"].date()): int(row["row_count"])
        for _, row in frame.sort_values("profile_date_ts").iterrows()
    }
    null_totals: dict[str, float] = {}
    for _, row in frame.iterrows():
        row_count = int(row["row_count"])
        if row_count <= 0:
            continue
        for feature_name, null_pct in _null_rate_dict(row.get("null_rates")).items():
            null_totals[feature_name] = null_totals.get(feature_name, 0.0) + (float(null_pct) * row_count)
    null_rates = {
        feature_name: round(weighted_total / total_rows, 2)
        for feature_name, weighted_total in sorted(null_totals.items())
    }
    return {
        "model_key": model_key,
        "total_rows": total_rows,
        "min_date": str(frame["profile_date_ts"].min().date()),
        "max_date": str(frame["profile_date_ts"].max().date()),
        "prediction_mean": float(prediction_mean) if prediction_mean is not None else None,
        "prediction_std": float(prediction_std) if prediction_std is not None else None,
        "daily_volume": daily_volume,
        "null_rates": null_rates,
        "computed_at": "",
    }


def _coerce_timestamp(value: object) -> pd.Timestamp | None:
    if value in (None, "", pd.NaT):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _empty_frame_with_reason(reason: str) -> pd.DataFrame:
    frame = pd.DataFrame()
    frame.attrs["_empty_reason"] = str(reason or "").strip().lower()
    return frame


def _freshness_status(config: MonitorConfig, runtime_state: MonitorRuntimeState | None) -> str:
    if runtime_state is None or runtime_state.bootstrap_status != "completed":
        return "pending_bootstrap"
    if runtime_state.last_run_status == "failed":
        return "failed"
    if not config.schedule_enabled:
        return "manual"
    now = pd.Timestamp.now(tz="UTC")
    due_points = [
        _coerce_timestamp(runtime_state.next_drift_due_at),
        _coerce_timestamp(runtime_state.next_performance_due_at) if config.has_labels else None,
    ]
    active_due_points = [value for value in due_points if value is not None]
    if active_due_points and min(active_due_points) <= now:
        return "stale"
    return "fresh"


def _period_label(series: pd.Series, granularity: str) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    labels = timestamps.dt.date.astype("object")
    if granularity == "monthly":
        labels = timestamps.dt.to_period("M").dt.to_timestamp().dt.date.astype("object")
    elif granularity == "weekly":
        labels = timestamps.dt.to_period("W").dt.end_time.dt.date.astype("object")
    labels = labels.where(timestamps.notna(), None)
    return labels.map(lambda value: str(value) if value is not None and not pd.isna(value) else None)


def _sql_placeholders(count: int) -> str:
    return ", ".join(["%s"] * max(count, 1))


def _drift_results_from_frame(frame: pd.DataFrame, granularity: str = "daily") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["window_end_ts"] = pd.to_datetime(working["window_end"], errors="coerce")
    working["computed_at_ts"] = pd.to_datetime(working["computed_at"], errors="coerce")
    working["period"] = _period_label(working["window_end"], granularity)
    working = working[working["period"].notna()].copy()
    metrics = (
        working.pivot_table(
            index=["feature_name", "period"],
            columns="metric_name",
            values="metric_value",
            aggfunc="max",
        )
        .reset_index()
    )
    window_level = working.drop_duplicates(
        subset=[
            "feature_name",
            "period",
            "baseline_start",
            "baseline_end",
            "window_start",
            "window_end",
        ]
    )
    latest_static = (
        window_level.sort_values(["feature_name", "period", "window_end_ts", "computed_at_ts"])
        .drop_duplicates(subset=["feature_name", "period"], keep="last")
        [
            [
                "feature_name",
                "period",
                "window_start",
                "window_end",
                "baseline_start",
                "baseline_end",
                "ref_mean",
                "cur_mean",
                "ref_std",
                "cur_std",
                "ref_null_pct",
                "cur_null_pct",
                "computed_at",
            ]
        ]
    )
    aggregated_counts = (
        window_level.groupby(["feature_name", "period"], as_index=False)
        .agg(
            ref_count=("ref_count", "sum"),
            cur_count=("cur_count", "sum"),
        )
    )
    result = latest_static.merge(aggregated_counts, on=["feature_name", "period"], how="left").merge(
        metrics,
        on=["feature_name", "period"],
        how="left",
    ).rename(columns={"feature_name": "feature"})
    for metric in ("psi", "js_divergence", "kl_divergence"):
        if metric not in result.columns:
            result[metric] = 0.0
    return result.sort_values(["period", "feature"]).reset_index(drop=True)


@dataclass
class DashboardBackend:
    repository: ControlPlaneRepository

    @property
    def _warehouse(self):
        return self.repository._warehouse

    def _published_generation_id(self, model_id: str) -> str | None:
        getter = getattr(self.repository, "get_latest_published_generation_id", None)
        if callable(getter):
            return getter(model_id)
        return None

    def _resolved_dashboard_source_bounds(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        fallback_dates: pd.Series | list[object] | tuple[object, ...] | None = None,
    ) -> tuple[str | None, str | None]:
        resolved_start = str(start_date or "").strip() or None
        resolved_end = str(end_date or "").strip() or None
        if resolved_start and resolved_end:
            return resolved_start, resolved_end
        if fallback_dates is None:
            return None, None
        fallback_series = pd.to_datetime(pd.Series(list(fallback_dates)), errors="coerce").dropna()
        if fallback_series.empty:
            return None, None
        if not resolved_start:
            resolved_start = str(pd.Timestamp(fallback_series.min()).date())
        if not resolved_end:
            resolved_end = str(pd.Timestamp(fallback_series.max()).date())
        if resolved_start and resolved_end:
            return resolved_start, resolved_end
        return None, None

    def _recent_unfiltered_quality_bounds(self, model_id: str) -> tuple[str | None, str | None]:
        loader = getattr(self.repository, "get_daily_quality_profile_rows", None)
        rows: list[dict[str, object]] = []
        if callable(loader):
            try:
                rows = loader(model_id)
            except Exception:
                logger.exception("Failed to load recent daily quality profiles for source-fallback bounds", extra={"model_key": model_id})
                rows = []
        if rows:
            frame = pd.DataFrame(rows)
            profile_dates = pd.to_datetime(frame.get("profile_date"), errors="coerce").dropna()
            if not profile_dates.empty:
                return (
                    str(pd.Timestamp(profile_dates.min()).date()),
                    str(pd.Timestamp(profile_dates.max()).date()),
                )
        history = self.get_quality_history(model_id)
        if history.empty:
            return None, None
        history_dates = pd.to_datetime(history.get("period"), errors="coerce").dropna()
        if history_dates.empty:
            history_dates = pd.to_datetime(history.get("window_end"), errors="coerce").dropna()
        if history_dates.empty:
            return None, None
        return (
            str(pd.Timestamp(history_dates.min()).date()),
            str(pd.Timestamp(history_dates.max()).date()),
        )

    def list_models(self) -> list[dict]:
        configs = self.repository.list_monitor_configs(status="active")
        if not configs:
            return []
        summary = self.repository.get_monitor_summary()
        runtime_states = (
            self.repository.list_monitor_runtime_states([config.model_key for config in configs])
            if hasattr(self.repository, "list_monitor_runtime_states")
            else {}
        )
        summary_map = {
            str(row["model_key"]): row
            for _, row in summary.iterrows()
        }
        models: list[dict] = []
        for config in configs:
            row = summary_map.get(config.model_key, {})
            runtime_state = runtime_states.get(config.model_key)
            versions = [config.model_version_value] if config.model_version_value else []
            description_parts = [config.source_table]
            if config.model_id_value:
                description_parts.append(f"model_id={config.model_id_value}")
            if config.model_version_value:
                description_parts.append(f"version={config.model_version_value}")
            models.append(
                {
                    "id": config.model_key,
                    "name": config.display_name,
                    "description": " | ".join(description_parts),
                    "versions": versions,
                    "feature_count": len(config.contract.feature_columns),
                    "slice_columns": list(config.contract.slice_columns),
                    "has_labels": bool(config.contract.label_col),
                    "baseline_days": config.baseline.n_days,
                    "baseline_kind": config.baseline.kind,
                    "baseline_label": baseline_label(config.baseline),
                    "max_psi": _safe_float(row.get("max_psi", 0)),
                    "total_rows": _safe_int(row.get("total_rows", 0)),
                    "open_incident_count": _safe_int(row.get("open_incident_count", 0)),
                    "freshness_status": _freshness_status(config, runtime_state),
                    "last_run_status": (runtime_state.last_run_status if runtime_state else None) or "",
                }
            )
        return models

    def get_model_map(self) -> dict[str, dict]:
        return {model["id"]: model for model in self.list_models()}

    def list_reference_models(self, status: str | None = "active") -> list[dict[str, str]]:
        configs = self.repository.list_monitor_configs(status=status)
        return [
            {
                "id": config.model_key,
                "name": config.display_name,
                "status": config.status,
            }
            for config in configs
        ]

    def get_monitor_config(self, model_id: str, status: str | list[str] | tuple[str, ...] | None = "active") -> MonitorConfig | None:
        for config in self.repository.list_monitor_configs(status=status):
            if config.model_key == model_id:
                return config
        return None

    def discover_monitor(
        self,
        *,
        source_table: str,
        labels_table: str | None = None,
        mlflow_experiment_name: str | None = None,
        mlflow_registered_model_name: str | None = None,
        baseline_days: int = 7,
    ) -> MonitorDiscoveryResult:
        service = MonitorDiscoveryService(self.repository)
        return service.discover(
            source_table=source_table,
            labels_table=labels_table,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_registered_model_name=mlflow_registered_model_name,
            baseline_days=baseline_days,
        )

    def _get_window_metadata(
        self,
        model_id: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, str]]:
        comparison_windows = getattr(self.repository.table_names, "comparison_windows", "")
        if not comparison_windows:
            return []
        filters = ["model_key = %s"]
        params: list[object] = [model_id]
        generation_id = self._published_generation_id(model_id)
        if not generation_id:
            return []
        filters.append("source_run_id = %s")
        params.append(generation_id)
        if start_date:
            filters.append("window_end >= CAST(%s AS DATE)")
            params.append(start_date)
        if end_date:
            filters.append("window_end <= CAST(%s AS DATE)")
            params.append(end_date)
        frame = self._warehouse.query_params(
            f"""
            WITH filtered_windows AS (
                SELECT window_id, model_key, window_grain, window_start, window_end, baseline_start, baseline_end, baseline_kind
                FROM {comparison_windows}
                WHERE {' AND '.join(filters)}
            ),
            recent_window_ends AS (
                SELECT window_end
                FROM filtered_windows
                GROUP BY window_end
                ORDER BY window_end DESC
                LIMIT {_MAX_DASHBOARD_WINDOW_HISTORY}
            )
            SELECT window_id, model_key, window_grain, window_start, window_end, baseline_start, baseline_end, baseline_kind
            FROM filtered_windows
            WHERE window_end IN (SELECT window_end FROM recent_window_ends)
            ORDER BY window_end, window_start
            """,
            tuple(params),
        )
        if frame.empty:
            return []
        return [
            {
                "window_id": str(row.get("window_id") or ""),
                "model_key": str(row.get("model_key") or model_id),
                "window_grain": str(row.get("window_grain") or "day"),
                "window_start": str(row.get("window_start") or ""),
                "window_end": str(row.get("window_end") or ""),
                "baseline_start": str(row.get("baseline_start") or ""),
                "baseline_end": str(row.get("baseline_end") or ""),
                "baseline_kind": str(row.get("baseline_kind") or "rolling"),
            }
            for _, row in frame.iterrows()
        ]

    def get_drift_results(
        self,
        model_id: str,
        granularity: str = "daily",
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        class_basis: str | None = None,
        class_value: str | None = None,
    ) -> pd.DataFrame:
        normalized_class_basis, normalized_class_value, class_filter_active = normalize_class_filter(class_basis, class_value)
        config = self.get_monitor_config(model_id)
        if class_filter_active:
            if not supports_binary_class_filters(config):
                return _empty_frame_with_reason("unsupported_class_filter")
            metadata_list = self._get_window_metadata(model_id, start_date=start_date, end_date=end_date)
            if not metadata_list:
                return _empty_frame_with_reason("missing_class_facts")
            load_start, load_end = _non_empty_text_bounds(
                [
                    value
                    for metadata in metadata_list
                    for value in (
                        metadata.get("baseline_start"),
                        metadata.get("baseline_end"),
                        metadata.get("window_start"),
                        metadata.get("window_end"),
                    )
                ]
            )
            if not load_start or not load_end:
                return _empty_frame_with_reason("missing_class_facts")
            class_feature_rows = (
                self.repository.get_daily_class_feature_profile_rows(
                    model_id,
                    start_date=load_start,
                    end_date=load_end,
                    class_basis=normalized_class_basis,
                    class_value=normalized_class_value,
                )
                if hasattr(self.repository, "get_daily_class_feature_profile_rows")
                else []
            )
            if not class_feature_rows:
                source_class_rows, source_reason = self._source_daily_class_feature_rows_fallback(
                    model_id,
                    start_date=load_start,
                    end_date=load_end,
                    class_basis=normalized_class_basis,
                    class_value=normalized_class_value,
                )
                if source_class_rows:
                    class_feature_rows = source_class_rows
                else:
                    if source_reason == "no_filtered_rows":
                        return _empty_frame_with_reason("no_filtered_rows")
                    source_probe_rows, probe_reason = self._source_daily_quality_rows_fallback(
                        model_id,
                        start_date=load_start,
                        end_date=load_end,
                        class_basis=normalized_class_basis,
                        class_value=normalized_class_value,
                    )
                    if probe_reason == "filtered_source_bounds_unavailable":
                        return _empty_frame_with_reason(probe_reason)
                    if probe_reason is None and source_probe_rows == []:
                        return _empty_frame_with_reason("no_filtered_rows")
                    return _empty_frame_with_reason("missing_class_facts")
            derived = derive_refresh_result_from_daily_profiles(
                config=config,
                metadata_list=metadata_list,
                daily_quality_profile_rows=[],
                daily_feature_profile_rows=class_feature_rows,
                daily_performance_profile_rows=[],
                computed_at=pd.Timestamp.now(tz=timezone.utc).isoformat(),
                include_drift_quality=True,
                include_performance=False,
            )
            return _drift_results_from_frame(pd.DataFrame(derived.drift_rows), granularity=granularity)
        filters = ["model_key = %s"]
        params: list[object] = [model_id]
        generation_id = self._published_generation_id(model_id)
        if not generation_id:
            return pd.DataFrame()
        filters.append("source_run_id = %s")
        params.append(generation_id)
        if start_date:
            filters.append("window_end >= CAST(%s AS DATE)")
            params.append(start_date)
        if end_date:
            filters.append("window_end <= CAST(%s AS DATE)")
            params.append(end_date)
        frame = self._warehouse.query_params(
            f"""
            WITH filtered_metrics AS (
                SELECT
                    feature_name,
                    metric_name,
                    metric_value,
                    window_start,
                    window_end,
                    baseline_start,
                    baseline_end,
                    ref_mean,
                    cur_mean,
                    ref_std,
                    cur_std,
                    ref_null_pct,
                    cur_null_pct,
                    ref_count,
                    cur_count,
                    computed_at
                FROM {self.repository.table_names.drift_metrics}
                WHERE {' AND '.join(filters)}
            ),
            recent_windows AS (
                SELECT window_end
                FROM filtered_metrics
                GROUP BY window_end
                ORDER BY window_end DESC
                LIMIT {_MAX_DASHBOARD_WINDOW_HISTORY}
            )
            SELECT
                feature_name,
                metric_name,
                metric_value,
                window_start,
                window_end,
                baseline_start,
                baseline_end,
                ref_mean,
                cur_mean,
                ref_std,
                cur_std,
                ref_null_pct,
                cur_null_pct,
                ref_count,
                cur_count,
                computed_at
            FROM filtered_metrics
            WHERE window_end IN (SELECT window_end FROM recent_windows)
            ORDER BY window_end, feature_name, metric_name
            """,
            tuple(params),
        )
        return _drift_results_from_frame(frame, granularity=granularity)

    def get_latest_drift(self, model_id: str, metric: str = "psi", top_n: int = 10) -> pd.DataFrame:
        drift = self.get_drift_results(model_id)
        if drift.empty or metric not in drift.columns:
            return pd.DataFrame()
        latest_period = drift["period"].max()
        return drift[drift["period"] == latest_period].nlargest(top_n, metric).reset_index(drop=True)

    def _daily_quality_rows(
        self,
        model_id: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        class_basis: str | None = None,
        class_value: str | None = None,
    ) -> tuple[list[dict[str, object]], str]:
        normalized_class_basis, normalized_class_value, class_filter_active = normalize_class_filter(class_basis, class_value)
        if class_filter_active:
            persisted_rows = (
                self.repository.get_daily_class_quality_profile_rows(
                    model_id,
                    start_date=start_date,
                    end_date=end_date,
                    class_basis=normalized_class_basis,
                    class_value=normalized_class_value,
                )
                if hasattr(self.repository, "get_daily_class_quality_profile_rows")
                else []
            )
            if persisted_rows:
                return persisted_rows, ""
            source_rows, empty_reason = self._source_daily_quality_rows_fallback(
                model_id,
                start_date=start_date,
                end_date=end_date,
                class_basis=normalized_class_basis,
                class_value=normalized_class_value,
            )
            if source_rows is not None:
                if source_rows:
                    return source_rows, ""
                return [], empty_reason or "no_filtered_rows"
            return [], empty_reason or "missing_class_facts"
        rows = (
            self.repository.get_daily_quality_profile_rows(
                model_id,
                start_date=start_date,
                end_date=end_date,
            )
            if hasattr(self.repository, "get_daily_quality_profile_rows")
            else []
        )
        return rows, ""

    def _source_daily_quality_rows_fallback(
        self,
        model_id: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        class_basis: str | None = None,
        class_value: str | None = None,
    ) -> tuple[list[dict[str, object]] | None, str | None]:
        loader = getattr(self.repository, "get_source_daily_quality_profile_rows", None)
        if loader is None:
            return None, "missing_class_facts"
        config = self.get_monitor_config(model_id, status=None)
        if not supports_binary_class_filters(config):
            return None, "unsupported_class_filter"
        resolved_start, resolved_end = self._resolved_dashboard_source_bounds(
            start_date=start_date,
            end_date=end_date,
            fallback_dates=None if (start_date or end_date) else list(self._recent_unfiltered_quality_bounds(model_id)),
        )
        if not resolved_start or not resolved_end:
            return None, "filtered_source_bounds_unavailable"
        try:
            rows = loader(
                config,
                start_date=resolved_start,
                end_date=resolved_end,
                class_basis=class_basis,
                class_value=class_value,
            )
            return rows, None
        except Exception:
            logger.exception(
                "Failed to derive filtered daily quality profiles directly from source rows",
                extra={"model_key": model_id},
            )
            return None, "missing_class_facts"

    def get_quality_stats(
        self,
        model_id: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        class_basis: str | None = None,
        class_value: str | None = None,
    ) -> dict:
        _, _, class_filter_active = normalize_class_filter(class_basis, class_value)
        if start_date or end_date or class_filter_active:
            rows, empty_reason = self._daily_quality_rows(
                model_id,
                start_date=start_date,
                end_date=end_date,
                class_basis=class_basis,
                class_value=class_value,
            )
            if rows:
                return _quality_summary_from_daily_profiles(model_id, rows)
            if class_filter_active:
                return {"_empty_reason": empty_reason or "no_filtered_rows"}
            return {}
        generation_id = self._published_generation_id(model_id)
        filters = ["model_key = %s"]
        params: list[object] = [model_id]
        if not generation_id:
            return {}
        filters.append("source_run_id = %s")
        params.append(generation_id)
        frame = self._warehouse.query_params(
            f"""
            SELECT *
            FROM {self.repository.table_names.quality_metrics}
            WHERE {' AND '.join(filters)}
            ORDER BY computed_at DESC
            LIMIT 1
            """,
            tuple(params),
        )
        if frame.empty:
            return {}
        row = frame.iloc[0]
        return {
            "total_rows": _safe_int(row.get("total_rows")),
            "min_date": str(row.get("min_date") or ""),
            "max_date": str(row.get("max_date") or ""),
            "prediction_mean": _safe_optional_float(row.get("prediction_mean")),
            "prediction_std": _safe_optional_float(row.get("prediction_std")),
            "daily_volume": _safe_json_dict(row.get("daily_volume")),
            "null_rates": _null_rate_dict(row.get("null_rates")),
            "computed_at": str(row.get("computed_at") or ""),
        }

    def get_quality_history(
        self,
        model_id: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        class_basis: str | None = None,
        class_value: str | None = None,
    ) -> pd.DataFrame:
        _, _, class_filter_active = normalize_class_filter(class_basis, class_value)
        if start_date or end_date or class_filter_active:
            return _quality_history_from_daily_profiles(
                pd.DataFrame(
                    self._daily_quality_rows(
                        model_id,
                        start_date=start_date,
                        end_date=end_date,
                        class_basis=class_basis,
                        class_value=class_value,
                    )[0]
                )
            )
        generation_id = self._published_generation_id(model_id)
        filters = ["model_key = %s"]
        params: list[object] = [model_id]
        if not generation_id:
            return pd.DataFrame()
        filters.append("source_run_id = %s")
        params.append(generation_id)
        frame = self._warehouse.query_params(
            f"""
            WITH recent_history AS (
                SELECT *
                FROM {self.repository.table_names.quality_history}
                WHERE {' AND '.join(filters)}
                ORDER BY window_end DESC
                LIMIT {_MAX_DASHBOARD_WINDOW_HISTORY}
            )
            SELECT *
            FROM recent_history
            ORDER BY window_end
            """,
            tuple(params),
        )
        if frame.empty:
            daily_quality_profiles = getattr(self.repository.table_names, "daily_quality_profiles", "")
            if not daily_quality_profiles:
                return pd.DataFrame()
            daily_filters = ["model_key = %s"]
            daily_params: list[object] = [model_id]
            daily_filters.append("source_run_id = %s")
            daily_params.append(generation_id)
            daily_frame = self._warehouse.query_params(
                f"""
                WITH recent_profiles AS (
                    SELECT *
                    FROM {daily_quality_profiles}
                    WHERE {' AND '.join(daily_filters)}
                    ORDER BY profile_date DESC
                    LIMIT {_MAX_DASHBOARD_DAILY_PROFILE_DAYS}
                )
                SELECT *
                FROM recent_profiles
                ORDER BY profile_date
                """,
                tuple(daily_params),
            )
            return _quality_history_from_daily_profiles(daily_frame)
        working = frame.copy()
        working["window_end_ts"] = pd.to_datetime(working["window_end"], errors="coerce")
        working["computed_at_ts"] = pd.to_datetime(working["computed_at"], errors="coerce")
        working["period"] = working["window_end_ts"].dt.date.astype(str)
        working["row_count"] = pd.to_numeric(working["row_count"], errors="coerce").fillna(0).astype(int)
        working["prediction_mean"] = pd.to_numeric(working["prediction_mean"], errors="coerce")
        working["prediction_std"] = pd.to_numeric(working["prediction_std"], errors="coerce")
        working["null_rates_dict"] = working["null_rates"].apply(_null_rate_dict)
        working["max_null_rate"] = working["null_rates_dict"].apply(
            lambda values: max(values.values()) if values else 0.0
        )
        working = working.sort_values(["period", "computed_at_ts", "window_end_ts"]).drop_duplicates("period", keep="last")
        return working.sort_values("window_end_ts").reset_index(drop=True)

    def get_null_rate_history(
        self,
        model_id: str,
        top_n: int = 12,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        class_basis: str | None = None,
        class_value: str | None = None,
    ) -> pd.DataFrame:
        history = self.get_quality_history(
            model_id,
            start_date=start_date,
            end_date=end_date,
            class_basis=class_basis,
            class_value=class_value,
        )
        if history.empty:
            return pd.DataFrame()
        exploded_rows: list[dict] = []
        for _, row in history.iterrows():
            rates = row.get("null_rates_dict") or {}
            for feature, null_rate in rates.items():
                exploded_rows.append({
                    "period": row["period"],
                    "feature": feature,
                    "null_rate": float(null_rate),
                    "window_end": row.get("window_end"),
                })
        if not exploded_rows:
            return pd.DataFrame()
        frame = pd.DataFrame(exploded_rows)
        top_features = (
            frame.groupby("feature", as_index=False)["null_rate"]
            .max()
            .sort_values("null_rate", ascending=False)
            .head(top_n)["feature"]
            .tolist()
        )
        return frame[frame["feature"].isin(top_features)].sort_values(["period", "feature"]).reset_index(drop=True)

    def _source_daily_class_feature_rows_fallback(
        self,
        model_id: str,
        *,
        start_date: str,
        end_date: str,
        class_basis: str,
        class_value: str,
    ) -> tuple[list[dict[str, object]], str | None]:
        config = self.get_monitor_config(model_id, status=None)
        if not supports_binary_class_filters(config):
            return [], "unsupported_class_filter"
        if not self._supports_safe_bounded_monitor_frame_load():
            return [], "filtered_source_bounds_unavailable"
        frame = self._load_monitor_frame_bounded(
            config,
            start_date=start_date,
            end_date=end_date,
            feature_columns=tuple(config.contract.feature_columns),
        )
        if frame.empty:
            return [], "no_filtered_rows"
        rows = build_daily_class_feature_profile_rows(
            config=config,
            inference_df=frame,
            computed_at=pd.Timestamp.now(tz=timezone.utc).isoformat(),
        )
        filtered_rows = [
            row
            for row in rows
            if str(row.get("class_basis") or "").strip().lower() == class_basis
            and str(row.get("class_value") or "").strip().lower() == class_value
        ]
        if not filtered_rows:
            return [], "no_filtered_rows"
        return filtered_rows, None

    def _latest_quality_map(self, model_ids: list[str]) -> dict[str, dict[str, object]]:
        if not model_ids:
            return {}
        placeholders = _sql_placeholders(len(model_ids))
        refresh_runs_table = getattr(self.repository.table_names, "refresh_runs", "")
        if not refresh_runs_table:
            frame = self._warehouse.query_params(
                f"""
                SELECT model_key, total_rows, min_date, max_date, prediction_mean, prediction_std, daily_volume, null_rates, computed_at
                FROM (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (PARTITION BY model_key ORDER BY computed_at DESC) AS row_num
                    FROM {self.repository.table_names.quality_metrics}
                    WHERE model_key IN ({placeholders})
                ) latest_quality
                WHERE row_num = 1
                """,
                tuple(model_ids),
            )
        else:
            frame = self._warehouse.query_params(
                f"""
                WITH latest_published AS (
                    SELECT model_key, generation_id
                    FROM (
                        SELECT
                            model_key,
                            generation_id,
                            ROW_NUMBER() OVER (
                                PARTITION BY model_key
                                ORDER BY published_at DESC, completed_at DESC, started_at DESC
                            ) AS row_num
                        FROM {refresh_runs_table}
                        WHERE generation_id IS NOT NULL
                          AND generation_id <> ''
                          AND published_at IS NOT NULL
                          AND model_key IN ({placeholders})
                    ) ranked_generations
                    WHERE row_num = 1
                )
                SELECT model_key, total_rows, min_date, max_date, prediction_mean, prediction_std, daily_volume, null_rates, computed_at
                FROM (
                    SELECT
                        quality.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY quality.model_key
                            ORDER BY quality.computed_at DESC
                        ) AS row_num
                    FROM {self.repository.table_names.quality_metrics} quality
                    LEFT JOIN latest_published published
                        ON quality.model_key = published.model_key
                    WHERE quality.model_key IN ({placeholders})
                      AND quality.source_run_id = published.generation_id
                ) latest_quality
                WHERE row_num = 1
                """,
                tuple(model_ids) + tuple(model_ids),
            )
        if frame.empty:
            return {}
        quality_map: dict[str, dict[str, object]] = {}
        for _, row in frame.iterrows():
            model_key = str(row.get("model_key") or "").strip()
            if not model_key:
                continue
            null_rates = _null_rate_dict(row.get("null_rates"))
            quality_map[model_key] = {
                "total_rows": _safe_int(row.get("total_rows")),
                "min_date": str(row.get("min_date") or ""),
                "max_date": str(row.get("max_date") or ""),
                "prediction_mean": _safe_float(row.get("prediction_mean")),
                "prediction_std": _safe_float(row.get("prediction_std")),
                "daily_volume": _safe_json_dict(row.get("daily_volume")),
                "null_rates": null_rates,
                "max_null_rate": max(null_rates.values()) if null_rates else 0.0,
                "computed_at": str(row.get("computed_at") or ""),
            }
        return quality_map

    def _historical_drift_summary_map(
        self,
        model_ids: list[str],
        metric: str,
        threshold_map: dict[str, dict[str, dict[str, float]]] | None = None,
    ) -> dict[str, dict[str, object]]:
        if not model_ids:
            return {}
        placeholders = _sql_placeholders(len(model_ids))
        refresh_runs_table = getattr(self.repository.table_names, "refresh_runs", "")
        if not refresh_runs_table:
            frame = self._warehouse.query_params(
                f"""
                WITH feature_metric_history AS (
                    SELECT
                        model_key,
                        feature_name,
                        metric_name,
                        MAX(metric_value) AS metric_value
                    FROM {self.repository.table_names.drift_metrics}
                    WHERE model_key IN ({placeholders})
                    GROUP BY model_key, feature_name, metric_name
                )
                SELECT model_key, feature_name, metric_name, metric_value
                FROM feature_metric_history
                ORDER BY model_key, feature_name, metric_name
                """,
                tuple(model_ids),
            )
        else:
            frame = self._warehouse.query_params(
                f"""
                WITH latest_published AS (
                    SELECT model_key, generation_id
                    FROM (
                        SELECT
                            model_key,
                            generation_id,
                            ROW_NUMBER() OVER (
                                PARTITION BY model_key
                                ORDER BY published_at DESC, completed_at DESC, started_at DESC
                            ) AS row_num
                        FROM {refresh_runs_table}
                        WHERE generation_id IS NOT NULL
                          AND generation_id <> ''
                          AND published_at IS NOT NULL
                          AND model_key IN ({placeholders})
                    ) ranked_generations
                    WHERE row_num = 1
                ),
                feature_metric_history AS (
                    SELECT
                        drift.model_key,
                        drift.feature_name,
                        drift.metric_name,
                        MAX(drift.metric_value) AS metric_value
                    FROM {self.repository.table_names.drift_metrics} drift
                    LEFT JOIN latest_published published
                        ON drift.model_key = published.model_key
                    WHERE drift.model_key IN ({placeholders})
                      AND drift.source_run_id = published.generation_id
                    GROUP BY drift.model_key, drift.feature_name, drift.metric_name
                )
                SELECT model_key, feature_name, metric_name, metric_value
                FROM feature_metric_history
                ORDER BY model_key, feature_name, metric_name
                """,
                tuple(model_ids) + tuple(model_ids),
            )
        if frame.empty:
            return {}
        working = frame.copy()
        pivoted = (
            working.pivot_table(
                index=["model_key", "feature_name"],
                columns="metric_name",
                values="metric_value",
                aggfunc="max",
            )
            .reset_index()
        )
        drift_map: dict[str, dict[str, object]] = {}
        for model_key, group in pivoted.groupby("model_key", sort=False):
            warning_threshold, critical_threshold = get_thresholds(
                metric,
                (threshold_map or {}).get(str(model_key)),
            )
            metric_series = (
                pd.to_numeric(group[metric], errors="coerce").fillna(0.0)
                if metric in group.columns
                else pd.Series(dtype=float)
            )
            js_series = (
                pd.to_numeric(group["js_divergence"], errors="coerce").fillna(0.0)
                if "js_divergence" in group.columns
                else pd.Series(dtype=float)
            )
            top_drifter = "N/A"
            if not metric_series.empty:
                top_index = metric_series.idxmax()
                top_drifter = str(group.loc[top_index, "feature_name"])
            drift_map[str(model_key)] = {
                "max_metric": _safe_float(metric_series.max()) if not metric_series.empty else 0.0,
                "avg_metric": _safe_float(metric_series.mean()) if not metric_series.empty else 0.0,
                "avg_js": _safe_float(js_series.mean()) if not js_series.empty else 0.0,
                "drifting_features": int((metric_series >= warning_threshold).sum()) if not metric_series.empty else 0,
                "total_features": int(len(group.index)),
                "top_drifter": top_drifter,
                "threshold_warning": float(warning_threshold),
                "threshold_critical": float(critical_threshold),
            }
        return drift_map

    def get_overview_rows(self, metric: str = "psi") -> list[dict]:
        models = self.list_models()
        model_ids = [str(model["id"]) for model in models if str(model.get("id") or "").strip()]
        quality_map = self._latest_quality_map(model_ids)
        active_configs = {
            config.model_key: config
            for config in self.repository.list_monitor_configs(status="active")
        }
        threshold_map = {
            model_id: merged_thresholds(getattr(active_configs.get(model_id), "threshold_overrides", None))
            for model_id in model_ids
        }
        drift_map = self._historical_drift_summary_map(model_ids, metric, threshold_map=threshold_map)
        rows: list[dict] = []
        for model in models:
            drift = drift_map.get(model["id"], {})
            quality = quality_map.get(model["id"], {})
            computing = not bool(drift)
            rows.append(
                {
                    "model_id": model["id"],
                    "model_name": model["name"],
                    "description": model["description"],
                    "versions": model["versions"],
                    "max_metric": _safe_float(drift.get("max_metric")),
                    "max_psi": _safe_float(drift.get("max_metric")),
                    "avg_metric": _safe_float(drift.get("avg_metric")),
                    "avg_psi": _safe_float(drift.get("avg_metric")),
                    "avg_js": _safe_float(drift.get("avg_js")),
                    "drifting_features": int(drift.get("drifting_features") or 0),
                    "total_features": int(model.get("feature_count") or 0),
                    "top_drifter": str(drift.get("top_drifter") or ("Computing/Pending" if computing else "N/A")),
                    "max_null_rate": _safe_float(quality.get("max_null_rate")),
                    "has_labels": model["has_labels"],
                    "computing": computing,
                    "freshness_status": model["freshness_status"],
                    "last_run_status": model["last_run_status"],
                    "metric": metric,
                    "threshold_warning": float(drift.get("threshold_warning") or get_thresholds(metric, threshold_map.get(model["id"]))[0]),
                    "threshold_critical": float(drift.get("threshold_critical") or get_thresholds(metric, threshold_map.get(model["id"]))[1]),
                    "thresholds": threshold_map.get(model["id"], merged_thresholds()),
                }
            )
        return rows

    def get_feature_options(self, model_id: str) -> list[str]:
        config = self.get_monitor_config(model_id)
        if not config:
            return []
        return list(config.contract.feature_columns)

    def get_dimension_options(self, model_id: str) -> list[str]:
        config = self.get_monitor_config(model_id)
        if not config:
            return []
        return list(config.contract.slice_columns)

    def _load_baseline_current(
        self,
        model_id: str,
        *,
        feature_columns: tuple[str, ...] | None = None,
    ) -> tuple[MonitorConfig | None, pd.DataFrame, pd.DataFrame]:
        config = self.get_monitor_config(model_id)
        if not config:
            return None, pd.DataFrame(), pd.DataFrame()
        drift = self.get_drift_results(model_id)
        if drift.empty or "window_end" not in drift.columns or "baseline_start" not in drift.columns:
            return config, pd.DataFrame(), pd.DataFrame()
        latest = drift.sort_values("window_end").iloc[-1]
        start_date = str(latest.get("baseline_start") or "")
        end_date = str(latest.get("window_end") or "")
        frame = self._load_monitor_frame_bounded(
            config,
            start_date=start_date or None,
            end_date=end_date or None,
            feature_columns=feature_columns or config.contract.feature_columns,
        )
        if frame.empty or config.contract.timestamp_col not in frame.columns:
            return config, pd.DataFrame(), pd.DataFrame()
        baseline, current = split_baseline_current(frame, config.contract.timestamp_col, config.baseline)
        return config, baseline, current

    def _load_current_window_frame(
        self,
        model_id: str,
        *,
        feature_columns: tuple[str, ...] | None = None,
    ) -> tuple[MonitorConfig | None, pd.DataFrame]:
        config = self.get_monitor_config(model_id)
        if not config:
            return None, pd.DataFrame()
        bounds = self._latest_window_bounds(model_id)
        if not bounds:
            fallback_config, _, current = self._load_baseline_current(model_id, feature_columns=feature_columns)
            return fallback_config, current
        start_date = bounds["window_start"] or None
        end_date = bounds["window_end"] or None
        frame = self._load_monitor_frame_bounded(
            config,
            start_date=start_date,
            end_date=end_date,
            feature_columns=feature_columns or config.contract.feature_columns,
        )
        return config, frame

    def _load_monitor_frame_bounded(
        self,
        config: MonitorConfig,
        *,
        start_date: str | None,
        end_date: str | None,
        feature_columns: tuple[str, ...] | None,
    ) -> pd.DataFrame:
        loader = getattr(self.repository, "load_monitor_frame", None)
        if loader is None:
            return pd.DataFrame()
        signature = inspect.signature(loader)
        parameter_names = set(signature.parameters)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        def _supported(name: str) -> bool:
            return accepts_var_kwargs or name in parameter_names

        # Feature Deep Dive raw fallbacks are only safe when the repository can
        # enforce an absolute cap on returned rows for the selected window.
        if not (_supported("start_date") and _supported("end_date") and _supported("max_total_rows")):
            return pd.DataFrame()

        kwargs = {
            "start_date": start_date,
            "end_date": end_date,
            "feature_columns": feature_columns,
        }
        if _supported("sample_rows_per_day"):
            kwargs["sample_rows_per_day"] = settings.feature_detail_sample_rows_per_day
        if _supported("max_total_rows"):
            kwargs["max_total_rows"] = settings.feature_detail_max_rows
        accepted_kwargs = {key: value for key, value in kwargs.items() if _supported(key)}
        if not accepted_kwargs:
            return pd.DataFrame()
        return loader(config, **accepted_kwargs)

    def _supports_safe_bounded_monitor_frame_load(self) -> bool:
        loader = getattr(self.repository, "load_monitor_frame", None)
        if loader is None:
            return False
        signature = inspect.signature(loader)
        parameter_names = set(signature.parameters)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        def _supported(name: str) -> bool:
            return accepts_var_kwargs or name in parameter_names

        return _supported("start_date") and _supported("end_date") and _supported("max_total_rows")

    def _latest_window_bounds(self, model_id: str) -> dict[str, str] | None:
        comparison_windows = getattr(self.repository.table_names, "comparison_windows", "")
        if not comparison_windows:
            return None
        filters = ["model_key = %s"]
        params: list[object] = [model_id]
        generation_id = self._published_generation_id(model_id)
        if not generation_id:
            return None
        filters.append("source_run_id = %s")
        params.append(generation_id)
        frame = self._warehouse.query_params(
            f"""
            SELECT baseline_start, baseline_end, window_start, window_end
            FROM {comparison_windows}
            WHERE {' AND '.join(filters)}
            ORDER BY window_end DESC, created_at DESC
            LIMIT 1
            """,
            tuple(params),
        )
        if frame.empty:
            return None
        row = frame.iloc[0]
        return {
            "baseline_start": str(row.get("baseline_start") or ""),
            "baseline_end": str(row.get("baseline_end") or ""),
            "window_start": str(row.get("window_start") or ""),
            "window_end": str(row.get("window_end") or ""),
        }

    def _feature_samples_from_daily_profiles(self, model_id: str, feature: str) -> tuple[pd.Series, pd.Series, bool]:
        daily_feature_profiles = getattr(self.repository.table_names, "daily_feature_profiles", "")
        if not daily_feature_profiles:
            return pd.Series(dtype=float), pd.Series(dtype=float), False
        bounds = self._latest_window_bounds(model_id)
        if not bounds:
            return pd.Series(dtype=float), pd.Series(dtype=float), False
        min_profile_date, max_profile_date = _non_empty_text_bounds(
            (
                bounds["baseline_start"],
                bounds["baseline_end"],
                bounds["window_start"],
                bounds["window_end"],
            )
        )
        if not min_profile_date or not max_profile_date:
            return pd.Series(dtype=float), pd.Series(dtype=float), False
        filters = [
            "model_key = %s",
            "feature_name = %s",
            "profile_date BETWEEN CAST(%s AS DATE) AND CAST(%s AS DATE)",
        ]
        params: list[object] = [model_id, feature, min_profile_date, max_profile_date]
        generation_id = self._published_generation_id(model_id)
        if not generation_id:
            return pd.Series(dtype=float), pd.Series(dtype=float), False
        filters.append("source_run_id = %s")
        params.append(generation_id)
        frame = self._warehouse.query_params(
            f"""
            SELECT profile_date, distribution_json
            FROM {daily_feature_profiles}
            WHERE {' AND '.join(filters)}
            ORDER BY profile_date
            """,
            tuple(params),
        )
        if frame.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float), False

        baseline_values: list[float] = []
        current_values: list[float] = []
        used_histogram_approximation = False
        baseline_start = bounds["baseline_start"]
        baseline_end = bounds["baseline_end"]
        window_start = bounds["window_start"]
        window_end = bounds["window_end"]
        for _, row in frame.iterrows():
            profile_date = str(row.get("profile_date") or "")
            payload = _safe_json_dict(row.get("distribution_json"))
            sample_values = _safe_json_list(payload.get("sample_values"))
            if not sample_values:
                sample_values = _approximate_histogram_values(
                    _safe_json_list(payload.get("edges")),
                    _safe_json_list(payload.get("counts")),
                )
                used_histogram_approximation = used_histogram_approximation or bool(sample_values)
            if not sample_values:
                continue
            if baseline_start <= profile_date <= baseline_end:
                baseline_values.extend(sample_values)
            elif window_start <= profile_date <= window_end:
                current_values.extend(sample_values)
        return pd.Series(baseline_values, dtype=float), pd.Series(current_values, dtype=float), used_histogram_approximation

    def get_feature_distribution(self, model_id: str, feature: str) -> tuple[pd.Series, pd.Series]:
        details = self.get_feature_distribution_details(model_id, feature)
        return details["baseline"], details["current"]

    def get_feature_distribution_details(
        self,
        model_id: str,
        feature: str,
        *,
        require_exact_samples: bool = False,
    ) -> dict[str, object]:
        baseline_samples, current_samples, used_histogram_approximation = self._feature_samples_from_daily_profiles(model_id, feature)
        bounds = self._latest_window_bounds(model_id)
        window_label = "Latest comparison window unavailable."
        config = self.get_monitor_config(model_id)
        raw_fallback_supported = self._supports_safe_bounded_monitor_frame_load()
        if bounds:
            window_label = (
                f"Baseline: {bounds['baseline_start'] or '—'} to {bounds['baseline_end'] or '—'} | "
                f"Current: {bounds['window_start'] or '—'} to {bounds['window_end'] or '—'}"
            )
        if not baseline_samples.empty and not current_samples.empty and not (require_exact_samples and used_histogram_approximation):
            return {
                "baseline": baseline_samples,
                "current": current_samples,
                "distribution_source": "persisted_histogram" if used_histogram_approximation else "persisted_samples",
                "approximate": bool(used_histogram_approximation),
                "window_label": window_label,
            }
        baseline, current = pd.DataFrame(), pd.DataFrame()
        if config:
            config, baseline, current = self._load_baseline_current(model_id, feature_columns=(feature,))
        if not config or feature not in baseline.columns or feature not in current.columns:
            return {
                "baseline": pd.Series(dtype=float),
                "current": pd.Series(dtype=float),
                "distribution_source": (
                    "unavailable_requested_raw"
                    if require_exact_samples
                    else ("unavailable_unsafe_bounded_read" if not raw_fallback_supported else "unavailable")
                ),
                "approximate": False,
                "window_label": window_label,
            }
        return {
            "baseline": pd.to_numeric(baseline[feature], errors="coerce").dropna(),
            "current": pd.to_numeric(current[feature], errors="coerce").dropna(),
            "distribution_source": "bounded_window_read",
            "approximate": False,
            "window_label": window_label,
        }

    def get_exact_performance_breakdown(
        self,
        model_id: str,
        *,
        metric_name: str = "f1",
        binning_mode: str = "auto",
        n_bins: int = 40,
        custom_edges: list[float] | None = None,
        outlier_mode: str = "off",
        outlier_value: float | None = None,
    ) -> dict[str, object]:
        config = self.get_monitor_config(model_id, status=None)
        if not config or not config.contract.label_col:
            return {
                "rows": pd.DataFrame(),
                "contributors": pd.DataFrame(),
                "message": "This monitor does not have labels configured for performance binning.",
            }
        bounds = self._latest_window_bounds(model_id)
        if not bounds:
            return {
                "rows": pd.DataFrame(),
                "contributors": pd.DataFrame(),
                "message": "No comparison window is available yet.",
            }
        feature_columns = tuple(config.contract.feature_columns or ())
        if not feature_columns:
            return {
                "rows": pd.DataFrame(),
                "contributors": pd.DataFrame(),
                "message": "No tracked feature columns are configured for this monitor.",
            }
        frame = self._load_monitor_frame_bounded(
            config,
            start_date=bounds.get("baseline_start") or None,
            end_date=bounds.get("window_end") or None,
            feature_columns=feature_columns,
        )
        if frame.empty or config.contract.timestamp_col not in frame.columns:
            return {
                "rows": pd.DataFrame(),
                "contributors": pd.DataFrame(),
                "message": "Exact bounded performance rows are unavailable for this monitor.",
            }
        working = frame.copy()
        ts_col = config.contract.timestamp_col
        working["_model_lens_ts"] = pd.to_datetime(working[ts_col], errors="coerce")
        working = working[working["_model_lens_ts"].notna()].copy()
        if working.empty:
            return {
                "rows": pd.DataFrame(),
                "contributors": pd.DataFrame(),
                "message": "No timestamped rows are available in the latest comparison window.",
            }
        baseline_start = pd.to_datetime(bounds.get("baseline_start") or "", errors="coerce")
        baseline_end = pd.to_datetime(bounds.get("baseline_end") or "", errors="coerce")
        window_start = pd.to_datetime(bounds.get("window_start") or "", errors="coerce")
        window_end = pd.to_datetime(bounds.get("window_end") or "", errors="coerce")
        baseline = working[
            working["_model_lens_ts"].between(baseline_start, baseline_end, inclusive="both")
        ].copy()
        current = working[
            working["_model_lens_ts"].between(window_start, window_end, inclusive="both")
        ].copy()
        regression_mode = (config.problem_type or "classification").strip().lower() == "regression"
        rows: list[dict[str, object]] = []
        for feature in feature_columns:
            if feature not in baseline.columns or feature not in current.columns:
                continue
            feature_baseline = _apply_feature_outlier_filter(
                baseline,
                feature,
                mode=outlier_mode,
                value=outlier_value,
            )
            feature_current = _apply_feature_outlier_filter(
                current,
                feature,
                mode=outlier_mode,
                value=outlier_value,
            )
            baseline_values = pd.to_numeric(feature_baseline.get(feature), errors="coerce").dropna().to_numpy(dtype=float, copy=False)
            current_values = pd.to_numeric(feature_current.get(feature), errors="coerce").dropna().to_numpy(dtype=float, copy=False)
            edges = _resolve_dynamic_bin_edges(
                baseline_values,
                current_values,
                binning_mode=binning_mode,
                n_bins=n_bins,
                custom_edges=custom_edges,
            )
            if len(edges) < 2:
                continue
            baseline_numeric = pd.to_numeric(feature_baseline[feature], errors="coerce")
            current_numeric = pd.to_numeric(feature_current[feature], errors="coerce")
            baseline_valid = baseline_numeric.notna()
            current_valid = current_numeric.notna()
            if not baseline_valid.any() or not current_valid.any():
                continue
            baseline_slice_frame = feature_baseline.loc[baseline_valid].copy()
            current_slice_frame = feature_current.loc[current_valid].copy()
            baseline_bins = np.digitize(
                baseline_numeric.loc[baseline_valid].to_numpy(dtype=float, copy=False),
                edges[1:-1],
                right=False,
            )
            current_bins = np.digitize(
                current_numeric.loc[current_valid].to_numpy(dtype=float, copy=False),
                edges[1:-1],
                right=False,
            )
            total_current = max(len(current_slice_frame), 1)
            for index in range(len(edges) - 1):
                base_slice = baseline_slice_frame.iloc[np.where(baseline_bins == index)[0]]
                cur_slice = current_slice_frame.iloc[np.where(current_bins == index)[0]]
                if base_slice.empty or cur_slice.empty:
                    continue
                if regression_mode:
                    base_metrics = compute_regression_metrics(base_slice, config.contract.prediction_col, config.contract.label_col)
                    cur_metrics = compute_regression_metrics(cur_slice, config.contract.prediction_col, config.contract.label_col)
                else:
                    base_metrics = compute_classification_metrics(
                        base_slice,
                        config.contract.prediction_col,
                        config.contract.label_col,
                        prediction_score_col=config.contract.prediction_score_col,
                    )
                    cur_metrics = compute_classification_metrics(
                        cur_slice,
                        config.contract.prediction_col,
                        config.contract.label_col,
                        prediction_score_col=config.contract.prediction_score_col,
                    )
                baseline_metric = base_metrics.get(metric_name) if base_metrics else None
                current_metric = cur_metrics.get(metric_name) if cur_metrics else None
                if baseline_metric is None or current_metric is None:
                    continue
                delta = (
                    baseline_metric - current_metric
                    if regression_mode
                    else current_metric - baseline_metric
                )
                current_share = float(len(cur_slice) / total_current * 100.0)
                rows.append(
                    {
                        "feature": feature,
                        "bin_label": f"[{edges[index]:.4g}, {edges[index + 1]:.4g})",
                        "baseline_metric": float(baseline_metric),
                        "current_metric": float(current_metric),
                        "delta": round(float(delta), 4),
                        "current_volume_pct": round(current_share, 2),
                        "degradation_contribution": round(float(delta) * current_share / 100.0, 4),
                        "baseline_count": int(len(base_slice)),
                        "current_count": int(len(cur_slice)),
                    }
                )
        result = pd.DataFrame(rows)
        if not result.empty:
            result["_bin_sort"] = result["bin_label"].str.extract(r"\[([^,]+),", expand=False).apply(_safe_float)
            result = result.sort_values(["feature", "_bin_sort", "bin_label"]).drop(columns=["_bin_sort"]).reset_index(drop=True)
        contributors = (
            result.groupby("feature", as_index=False)["degradation_contribution"]
            .sum()
            .rename(columns={"degradation_contribution": "weighted_delta"})
            .sort_values("weighted_delta")
            .reset_index(drop=True)
            if not result.empty
            else pd.DataFrame(columns=["feature", "weighted_delta"])
        )
        return {
            "rows": result,
            "contributors": contributors,
            "message": (
                ""
                if not result.empty
                else "No valid bin-level performance slices are available with the current controls."
            ),
        }

    def get_dimension_breakdown(self, model_id: str, feature: str, dimension: str) -> pd.DataFrame:
        config, current = self._load_current_window_frame(model_id, feature_columns=(feature, dimension))
        if not config or feature not in current.columns or dimension not in current.columns:
            return pd.DataFrame()
        working = current[[dimension, feature]].copy()
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
        if working.empty:
            return pd.DataFrame()
        dimension_values = working[dimension]
        top_dimension_values = dimension_values.value_counts(dropna=False).head(20).index.tolist()
        if not top_dimension_values:
            return pd.DataFrame()
        mask = pd.Series(False, index=working.index)
        for value in top_dimension_values:
            if pd.isna(value):
                mask = mask | dimension_values.isna()
            else:
                mask = mask | (dimension_values == value)
        filtered = working[mask].copy()
        filtered["_dimension_value"] = filtered[dimension].apply(
            lambda value: "(missing)" if pd.isna(value) or not str(value).strip() else str(value)
        )
        breakdown = (
            filtered.groupby("_dimension_value", dropna=False)
            .agg(
                feature_average=(feature, "mean"),
                feature_p25=(feature, lambda values: float(values.quantile(0.25)) if values.notna().any() else float("nan")),
                feature_p50=(feature, lambda values: float(values.quantile(0.50)) if values.notna().any() else float("nan")),
                feature_p75=(feature, lambda values: float(values.quantile(0.75)) if values.notna().any() else float("nan")),
                row_count=(feature, "size"),
            )
            .reset_index()
            .rename(columns={"_dimension_value": "dimension_value"})
            .sort_values("row_count", ascending=False)
            .head(20)
        )
        return breakdown

    def get_prediction_distribution(self, model_id: str) -> pd.Series:
        config, current = self._load_current_window_frame(model_id, feature_columns=None)
        if not config or config.contract.prediction_col not in current.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(current[config.contract.prediction_col], errors="coerce").dropna()

    def get_daily_label_metrics(
        self,
        model_id: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if not hasattr(self.repository, "get_daily_label_metric_rows"):
            return pd.DataFrame()
        frame = pd.DataFrame(
            self.repository.get_daily_label_metric_rows(
                model_id,
                start_date=start_date,
                end_date=end_date,
            )
        )
        if frame.empty:
            return frame
        frame["profile_date_ts"] = pd.to_datetime(frame["profile_date"], errors="coerce")
        for column in (
            "actual_positive_count",
            "actual_negative_count",
            "predicted_positive_count",
            "predicted_negative_count",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "f1",
            "accuracy",
        ):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.sort_values("profile_date_ts").reset_index(drop=True)

    def get_daily_performance_profiles(
        self,
        model_id: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if not hasattr(self.repository, "get_daily_performance_profile_rows"):
            return pd.DataFrame()
        frame = pd.DataFrame(
            self.repository.get_daily_performance_profile_rows(
                model_id,
                start_date=start_date,
                end_date=end_date,
            )
        )
        if frame.empty:
            return frame
        frame["profile_date_ts"] = pd.to_datetime(frame.get("profile_date"), errors="coerce")
        if "metric_name" in frame.columns:
            frame["metric_name"] = frame["metric_name"].astype(str).str.strip().str.lower()
        for column in ("metric_value", "row_count", "volume_pct"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        sort_columns = [column for column in ("profile_date_ts", "feature_name", "bin_label", "metric_name") if column in frame.columns]
        return frame.sort_values(sort_columns).reset_index(drop=True) if sort_columns else frame.reset_index(drop=True)

    def _daily_performance_timeline_fallback(self, model_id: str, metric_name: str) -> list[dict[str, float | None]]:
        profiles = self.get_daily_performance_profiles(model_id)
        if profiles.empty or "metric_name" not in profiles.columns or "profile_date_ts" not in profiles.columns:
            return []
        metric_key = str(metric_name or "").strip().lower()
        dated_profiles = profiles[profiles["profile_date_ts"].notna()].copy()
        if dated_profiles.empty:
            return []
        selected = dated_profiles[dated_profiles["metric_name"] == metric_key].copy()
        timeline: list[dict[str, float | None]] = []
        for profile_date in sorted(dated_profiles["profile_date_ts"].dropna().unique()):
            group = selected[selected["profile_date_ts"] == profile_date]
            value = (
                _weighted_average(
                    group.get("metric_value", pd.Series(dtype=float)),
                    group.get("row_count", pd.Series(dtype=float)),
                )
                if not group.empty
                else None
            )
            timeline.append(
                {
                    "period": str(pd.Timestamp(profile_date).date()),
                    metric_key: round(float(value), 4) if value is not None else None,
                }
            )
        return timeline

    def _window_performance_timeline(self, dated: pd.DataFrame, metric_name: str) -> list[dict[str, float | None]]:
        if dated.empty:
            return []
        timeline: list[dict[str, float | None]] = []
        for window_end, group in dated.groupby("window_end", sort=True):
            value = _weighted_average(
                group.get("current_metric", pd.Series(dtype=float)),
                group.get("current_volume_pct", pd.Series(dtype=float)),
            )
            timeline.append(
                {
                    "period": str(window_end.date()),
                    metric_name: round(float(value), 4) if value is not None else None,
                }
            )
        return timeline

    def _source_daily_label_metrics_fallback(
        self,
        model_id: str,
        *,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        loader = getattr(self.repository, "get_source_daily_label_metric_rows", None)
        if loader is None:
            return pd.DataFrame()
        config = self.get_monitor_config(model_id, status=None)
        if not supports_binary_class_filters(config):
            return pd.DataFrame()
        generation_id = self._published_generation_id(model_id) or ""
        cache_key = (model_id, start_date, end_date, generation_id)
        now = monotonic()
        with _EXACT_SOURCE_DAILY_METRIC_CACHE_LOCK:
            cached = _EXACT_SOURCE_DAILY_METRIC_CACHE.get(cache_key)
            if cached and cached[0] > now:
                return cached[1].copy()
        try:
            frame = pd.DataFrame(
                loader(
                    config,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        except Exception:
            logger.exception("Failed to derive daily label metrics directly from the source rows", extra={"model_key": model_id})
            return pd.DataFrame()
        with _EXACT_SOURCE_DAILY_METRIC_CACHE_LOCK:
            expired_keys = [key for key, (expires_at, _) in _EXACT_SOURCE_DAILY_METRIC_CACHE.items() if expires_at <= now]
            for key in expired_keys:
                _EXACT_SOURCE_DAILY_METRIC_CACHE.pop(key, None)
        if frame.empty:
            with _EXACT_SOURCE_DAILY_METRIC_CACHE_LOCK:
                _EXACT_SOURCE_DAILY_METRIC_CACHE[cache_key] = (
                    now + _EXACT_SOURCE_DAILY_METRIC_CACHE_TTL_SECONDS,
                    pd.DataFrame(),
                )
            return frame
        frame["profile_date_ts"] = pd.to_datetime(frame["profile_date"], errors="coerce")
        for column in (
            "actual_positive_count",
            "actual_negative_count",
            "predicted_positive_count",
            "predicted_negative_count",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "f1",
            "accuracy",
        ):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        normalized = frame.sort_values("profile_date_ts").reset_index(drop=True)
        with _EXACT_SOURCE_DAILY_METRIC_CACHE_LOCK:
            _EXACT_SOURCE_DAILY_METRIC_CACHE[cache_key] = (
                now + _EXACT_SOURCE_DAILY_METRIC_CACHE_TTL_SECONDS,
                normalized.copy(),
            )
        return normalized

    def _resolved_daily_label_metrics(
        self,
        model_id: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        exact_source_start_date: str | None = None,
        exact_source_end_date: str | None = None,
    ) -> pd.DataFrame:
        if exact_source_start_date and exact_source_end_date:
            exact = self._source_daily_label_metrics_fallback(
                model_id,
                start_date=exact_source_start_date,
                end_date=exact_source_end_date,
            )
            if not exact.empty:
                return exact
        return self.get_daily_label_metrics(
            model_id,
            start_date=start_date,
            end_date=end_date,
        )

    def _latest_window_metric_fallback_from_daily_profiles(
        self,
        model_id: str,
        *,
        window_start: str | None,
        window_end: str | None,
    ) -> dict[str, float | None]:
        profiles = self.get_daily_performance_profiles(
            model_id,
            start_date=window_start,
            end_date=window_end,
        )
        if profiles.empty or "metric_name" not in profiles.columns:
            return {}
        metrics: dict[str, float | None] = {}
        for metric_name in ("precision", "recall", "f1", "accuracy"):
            selected = profiles[profiles["metric_name"] == metric_name]
            if selected.empty:
                metrics[metric_name] = None
                continue
            value = _weighted_average(
                selected.get("metric_value", pd.Series(dtype=float)),
                selected.get("row_count", pd.Series(dtype=float)),
            )
            metrics[metric_name] = round(float(value), 4) if value is not None else None
        return metrics

    def _latest_window_metric_fallback_from_window_rows(self, model_id: str) -> dict[str, float | None]:
        metrics: dict[str, float | None] = {}
        for metric_name in ("precision", "recall", "f1", "accuracy"):
            rows = self.get_performance_rows(model_id, metric_name=metric_name)
            if rows.empty:
                metrics[metric_name] = None
                continue
            rows = rows.copy()
            rows["window_end"] = pd.to_datetime(rows["window_end"], errors="coerce")
            dated = rows[rows["window_end"].notna()].copy()
            if dated.empty:
                metrics[metric_name] = None
                continue
            latest = dated[dated["window_end"] == dated["window_end"].max()]
            value = _weighted_average(
                latest.get("current_metric", pd.Series(dtype=float)),
                latest.get("current_volume_pct", pd.Series(dtype=float)),
            )
            metrics[metric_name] = round(float(value), 4) if value is not None else None
        return metrics

    def get_latest_window_metrics(self, model_id: str) -> dict[str, object]:
        config = self.get_monitor_config(model_id, status=None)
        if not supports_binary_class_filters(config):
            return {
                "supported": False,
                "metrics": {},
                "message": "Latest-window performance snapshot is available only for binary classification monitors with labels.",
            }
        bounds = self._latest_window_bounds(model_id)
        if not bounds or not bounds.get("window_start") or not bounds.get("window_end"):
            return {
                "supported": True,
                "metrics": {},
                "message": "No comparison window is available yet for a latest-window performance snapshot.",
                "window_start": (bounds or {}).get("window_start") or "",
                "window_end": (bounds or {}).get("window_end") or "",
            }
        frame = self._resolved_daily_label_metrics(
            model_id,
            start_date=bounds.get("window_start") or None,
            end_date=bounds.get("window_end") or None,
            exact_source_start_date=bounds.get("window_start") or None,
            exact_source_end_date=bounds.get("window_end") or None,
        )
        if frame.empty:
            window_metrics = self._latest_window_metric_fallback_from_window_rows(model_id)
            if any(value is not None for value in window_metrics.values()):
                return {
                    "supported": True,
                    "metrics": window_metrics,
                    "message": "Showing recent performance trends.",
                    "window_start": bounds.get("window_start") or "",
                    "window_end": bounds.get("window_end") or "",
                }
            return {
                "supported": True,
                "metrics": {},
                "message": "Latest-window performance metrics are unavailable until labeled daily facts or comparison-window performance rows are populated.",
                "window_start": bounds.get("window_start") or "",
                "window_end": bounds.get("window_end") or "",
            }
        return {
            "supported": True,
            "metrics": _aggregated_classification_metrics(frame),
            "window_start": bounds.get("window_start") or "",
            "window_end": bounds.get("window_end") or "",
        }

    def get_performance_rows(self, model_id: str, metric_name: str = "f1") -> pd.DataFrame:
        filters = ["model_key = %s", "metric_name = %s"]
        params: list[object] = [model_id, metric_name]
        generation_id = self._published_generation_id(model_id)
        if not generation_id:
            return pd.DataFrame()
        filters.append("source_run_id = %s")
        params.append(generation_id)
        frame = self._warehouse.query_params(
            f"""
            WITH filtered_metrics AS (
                SELECT *
                FROM {self.repository.table_names.performance_metrics}
                WHERE {' AND '.join(filters)}
            ),
            recent_windows AS (
                SELECT window_end
                FROM filtered_metrics
                GROUP BY window_end
                ORDER BY window_end DESC
                LIMIT {_MAX_DASHBOARD_PERFORMANCE_WINDOWS}
            )
            SELECT *
            FROM filtered_metrics
            WHERE window_end IN (SELECT window_end FROM recent_windows)
            ORDER BY window_end, feature_name, bin_label
            """,
            tuple(params),
        )
        if frame.empty:
            return pd.DataFrame()
        renamed = frame.rename(
            columns={
                "feature_name": "feature",
                "volume_pct": "current_volume_pct",
                "contribution": "degradation_contribution",
            }
        )
        for column in ("baseline_metric", "current_metric", "delta", "current_volume_pct", "degradation_contribution"):
            if column in renamed.columns:
                renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
        return renamed

    def get_performance_summary(self, model_id: str, metric_name: str = "f1") -> dict:
        frame = self.get_performance_rows(model_id, metric_name=metric_name)
        if frame.empty:
            return {
                "timeline": [],
                "contributors": pd.DataFrame(),
                "latest_bins": pd.DataFrame(),
                "all_bins": pd.DataFrame(),
                "has_significant_degradation": False,
                "timeline_unavailable_reason": "",
            }
        config = self.get_monitor_config(model_id, status=None)
        classification_metric_names = {"precision", "recall", "f1", "accuracy"}
        uses_daily_classification_timeline = bool(
            supports_binary_class_filters(config)
            and str(metric_name or "").strip().lower() in classification_metric_names
        )
        frame = frame.copy()
        frame["window_end"] = pd.to_datetime(frame["window_end"], errors="coerce")
        dated = frame[frame["window_end"].notna()].copy()
        label_metrics = pd.DataFrame()
        if uses_daily_classification_timeline and not dated.empty:
            start_date = str(dated["window_end"].min().date())
            end_date = str(dated["window_end"].max().date())
            exact_source_start_date, exact_source_end_date = self._resolved_dashboard_source_bounds(
                start_date=start_date,
                end_date=end_date,
            )
            label_metrics = self._resolved_daily_label_metrics(
                model_id,
                start_date=start_date,
                end_date=end_date,
                exact_source_start_date=exact_source_start_date,
                exact_source_end_date=exact_source_end_date,
            )
        elif uses_daily_classification_timeline:
            label_metrics = self._resolved_daily_label_metrics(model_id)
        timeline_unavailable_reason = ""
        if uses_daily_classification_timeline and not label_metrics.empty and metric_name in label_metrics.columns:
            timeline = [
                {
                    "period": str(row["profile_date_ts"].date()),
                    metric_name: _safe_optional_float(row.get(metric_name)),
                }
                for _, row in label_metrics.iterrows()
                if pd.notna(row.get("profile_date_ts"))
            ]
        elif uses_daily_classification_timeline:
            timeline = self._window_performance_timeline(dated, metric_name)
            if not timeline:
                timeline_unavailable_reason = (
                    f"{metric_name.upper()} over time is unavailable until daily labeled facts or comparison-window performance rows are populated for this monitor."
                )
        else:
            timeline = self._window_performance_timeline(dated, metric_name)
        latest_bins = dated.copy() if dated.empty else dated[dated["window_end"] == dated["window_end"].max()].copy()
        contributors = (
            latest_bins.groupby("feature", as_index=False)
            .agg(weighted_delta=("degradation_contribution", "sum"))
            .sort_values("weighted_delta")
            if not latest_bins.empty
            else pd.DataFrame(columns=["feature", "weighted_delta"])
        )
        delta_source = latest_bins if not latest_bins.empty else frame
        has_significant_degradation = bool(
            "delta" in delta_source.columns and (pd.to_numeric(delta_source["delta"], errors="coerce").fillna(0.0) < -0.005).any()
        )
        return {
            "timeline": timeline,
            "contributors": contributors,
            "latest_bins": latest_bins,
            "all_bins": frame,
            "has_significant_degradation": has_significant_degradation,
            "worst_weighted_delta": _safe_series_min(
                contributors["weighted_delta"] if "weighted_delta" in contributors.columns else pd.Series(dtype=float)
            ),
            "timeline_unavailable_reason": timeline_unavailable_reason,
        }

    def get_reference_data(self, model_id: str) -> dict:
        config = self.get_monitor_config(model_id, status=None)
        summary = self.repository.get_monitor_summary()
        summary_row = summary[summary["model_key"] == model_id]
        runtime_state = self.repository.get_monitor_runtime_state(model_id) if hasattr(self.repository, "get_monitor_runtime_state") else None
        recent_runs = (
            self.repository.get_recent_refresh_runs(model_id, limit=12)
            if hasattr(self.repository, "get_recent_refresh_runs")
            else []
        )
        try:
            shared_schedule = resolve_shared_workflow_schedule_status()
            shared_schedule_payload = {
                "configured": shared_schedule.configured,
                "resolved": shared_schedule.resolved,
                "job_id": shared_schedule.job_id,
                "job_name": shared_schedule.job_name,
                "scheduler_mode": shared_schedule.scheduler_mode,
                "current_expression": shared_schedule.current_expression,
                "current_interval_hours": shared_schedule.current_interval_hours,
                "current_label": shared_schedule.current_label,
                "timezone_id": shared_schedule.timezone_id,
                "paused": shared_schedule.paused,
                "editable": shared_schedule.editable,
                "supported": shared_schedule.supported,
                "checked_at": shared_schedule.checked_at,
                "management_available": shared_schedule.management_available,
                "blocking_issues": list(shared_schedule.blocking_issues),
                "warnings": list(shared_schedule.warnings),
            }
        except Exception as error:
            shared_schedule_payload = {
                "configured": False,
                "resolved": False,
                "job_id": None,
                "job_name": "",
                "scheduler_mode": "missing",
                "current_expression": "",
                "current_interval_hours": None,
                "current_label": "Unavailable",
                "timezone_id": "UTC",
                "paused": False,
                "editable": False,
                "supported": False,
                "checked_at": "",
                "management_available": None,
                "blocking_issues": [],
                "warnings": [str(error)],
            }
        return {
            "config": config,
            "status": config.status if config else "",
            "summary": summary_row.iloc[0].to_dict() if not summary_row.empty else {},
            "runtime_state": runtime_state.__dict__ if runtime_state else {},
            "recent_runs": recent_runs,
            "refresh_diagnostics": build_refresh_diagnostics(recent_runs),
            "recent_incident_history": (
                self.repository.get_recent_incident_history(model_id, limit=8)
                if hasattr(self.repository, "get_recent_incident_history")
                else []
            ),
            "settings": {
                "app_title": settings.app_title,
                "control_plane_catalog": self.repository.table_names.catalog,
                "control_plane_schema": self.repository.table_names.schema,
                "sql_warehouse_id": settings.sql_warehouse_id,
                "refresh_job_id": settings.refresh_job_id,
                "refresh_job_name": settings.refresh_job_name,
                "bootstrap_refresh_job_id": settings.bootstrap_refresh_job_id,
                "bootstrap_refresh_job_name": settings.bootstrap_refresh_job_name,
                "use_lakebase_read_model": settings.use_lakebase_read_model,
                "lakebase_database_name": settings.lakebase_database_name,
                "genie_space_id": settings.genie_space_id,
                "shared_schedule": shared_schedule_payload,
            },
        }

    def get_incidents_data(self, *, limit_history: int = 50) -> dict:
        configs = self.repository.list_monitor_configs(status=None)
        model_metadata = {
            config.model_key: {
                "display_name": config.display_name,
                "status": getattr(config, "status", "active"),
            }
            for config in configs
        }
        models = [
            {
                "id": config.model_key,
                "name": config.display_name,
                "status": getattr(config, "status", "active"),
            }
            for config in configs
        ]
        open_incidents = self.repository.get_open_incidents().copy()
        if not open_incidents.empty:
            open_incidents["status"] = "open"
            open_incidents["display_name"] = open_incidents["model_key"].map(
                lambda value: model_metadata.get(str(value), {}).get("display_name", str(value))
            )
            open_incidents["monitor_status"] = open_incidents["model_key"].map(
                lambda value: model_metadata.get(str(value), {}).get("status", "")
            )
        history_rows = (
            self.repository.get_recent_incident_history_all(limit=limit_history)
            if hasattr(self.repository, "get_recent_incident_history_all")
            else []
        )
        incident_history = pd.DataFrame(history_rows)
        if not incident_history.empty:
            incident_history["display_name"] = incident_history["model_key"].map(
                lambda value: model_metadata.get(str(value), {}).get("display_name", str(value))
            )
            incident_history["monitor_status"] = incident_history["model_key"].map(
                lambda value: model_metadata.get(str(value), {}).get("status", "")
            )
        return {
            "models": models,
            "open_incidents": open_incidents,
            "history": incident_history,
        }


def build_dashboard_backend(
    *,
    catalog: str | None = None,
    schema: str | None = None,
    lakebase_instance_name: str | None = None,
    lakebase_database_name: str | None = None,
    lakebase_schema: str | None = None,
) -> DashboardBackend:
    repository = build_repository(
        catalog=catalog,
        schema=schema,
        lakebase_instance_name=lakebase_instance_name,
        lakebase_database_name=lakebase_database_name,
        lakebase_schema=lakebase_schema,
    )
    return DashboardBackend(repository=repository)
