from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from model_lens.config import settings
from model_lens.domain.models import MonitorConfig, MonitorDiscoveryResult, MonitorRuntimeState
from model_lens.services.control_plane import ControlPlaneRepository, build_repository
from model_lens.services.monitor_discovery import MonitorDiscoveryService
from model_lens.services.onboarding import baseline_label
from model_lens.services.refresh_engine import split_baseline_current


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


def _safe_json_list(value: object) -> list[float]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
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


def _safe_int(value: object) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return 0
    return int(numeric)


def _safe_series_min(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.min())


def _null_rate_dict(value: object) -> dict[str, float]:
    return {key: float(parsed) for key, parsed in _safe_json_dict(value).items()}


def _inclusive_end_bound(value: str) -> pd.Timestamp:
    bound = pd.Timestamp(value)
    if pd.isna(bound):
        return bound
    if bound == bound.normalize():
        return bound + pd.Timedelta(days=1)
    return bound


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
    working["prediction_mean"] = pd.to_numeric(working["prediction_mean"], errors="coerce").fillna(0.0)
    working["prediction_std"] = pd.to_numeric(working["prediction_std"], errors="coerce").fillna(0.0)
    working["null_rates_dict"] = working["null_rates"].apply(_null_rate_dict)
    working["max_null_rate"] = working["null_rates_dict"].apply(
        lambda values: max(values.values()) if values else 0.0
    )
    return working.sort_values("profile_date_ts").reset_index(drop=True)


def _coerce_timestamp(value: object) -> pd.Timestamp | None:
    if value in (None, "", pd.NaT):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


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
    if granularity == "monthly":
        return timestamps.dt.to_period("M").dt.to_timestamp().dt.date.astype(str)
    if granularity == "weekly":
        return timestamps.dt.to_period("W").dt.end_time.dt.date.astype(str)
    return timestamps.dt.date.astype(str)


def _sql_placeholders(count: int) -> str:
    return ", ".join(["%s"] * max(count, 1))


@dataclass
class DashboardBackend:
    repository: ControlPlaneRepository

    @property
    def _warehouse(self):
        return self.repository._warehouse

    def list_models(self) -> list[dict]:
        configs = self.repository.list_monitor_configs(status="active")
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

    def get_drift_results(self, model_id: str, granularity: str = "daily") -> pd.DataFrame:
        frame = self._warehouse.query_params(
            f"""
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
            WHERE model_key = %s
            ORDER BY window_end, feature_name, metric_name
            """,
            (model_id,),
        )
        if frame.empty:
            return pd.DataFrame()

        working = frame.copy()
        working["window_end_ts"] = pd.to_datetime(working["window_end"], errors="coerce")
        working["computed_at_ts"] = pd.to_datetime(working["computed_at"], errors="coerce")
        working["period"] = _period_label(working["window_end"], granularity)
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
        static = latest_static.merge(aggregated_counts, on=["feature_name", "period"], how="left")
        result = static.merge(metrics, on=["feature_name", "period"], how="left").rename(columns={"feature_name": "feature"})
        for metric in ("psi", "js_divergence", "kl_divergence"):
            if metric not in result.columns:
                result[metric] = 0.0
        return result.sort_values(["period", "feature"]).reset_index(drop=True)

    def get_latest_drift(self, model_id: str, metric: str = "psi", top_n: int = 10) -> pd.DataFrame:
        drift = self.get_drift_results(model_id)
        if drift.empty or metric not in drift.columns:
            return pd.DataFrame()
        latest_period = drift["period"].max()
        return drift[drift["period"] == latest_period].nlargest(top_n, metric).reset_index(drop=True)

    def get_quality_stats(self, model_id: str) -> dict:
        frame = self._warehouse.query_params(
            f"""
            SELECT *
            FROM {self.repository.table_names.quality_metrics}
            WHERE model_key = %s
            ORDER BY computed_at DESC
            LIMIT 1
            """,
            (model_id,),
        )
        if frame.empty:
            return {}
        row = frame.iloc[0]
        return {
            "total_rows": _safe_int(row.get("total_rows")),
            "min_date": str(row.get("min_date") or ""),
            "max_date": str(row.get("max_date") or ""),
            "prediction_mean": _safe_float(row.get("prediction_mean")),
            "prediction_std": _safe_float(row.get("prediction_std")),
            "daily_volume": _safe_json_dict(row.get("daily_volume")),
            "null_rates": _null_rate_dict(row.get("null_rates")),
            "computed_at": str(row.get("computed_at") or ""),
        }

    def get_quality_history(self, model_id: str) -> pd.DataFrame:
        frame = self._warehouse.query_params(
            f"""
            SELECT *
            FROM {self.repository.table_names.quality_history}
            WHERE model_key = %s
            ORDER BY window_end
            """,
            (model_id,),
        )
        if frame.empty:
            daily_quality_profiles = getattr(self.repository.table_names, "daily_quality_profiles", "")
            if not daily_quality_profiles:
                return pd.DataFrame()
            daily_frame = self._warehouse.query_params(
                f"""
                SELECT *
                FROM {daily_quality_profiles}
                WHERE model_key = %s
                ORDER BY profile_date
                """,
                (model_id,),
            )
            return _quality_history_from_daily_profiles(daily_frame)
        working = frame.copy()
        working["window_end_ts"] = pd.to_datetime(working["window_end"], errors="coerce")
        working["period"] = working["window_end_ts"].dt.date.astype(str)
        working["row_count"] = pd.to_numeric(working["row_count"], errors="coerce").fillna(0).astype(int)
        working["prediction_mean"] = pd.to_numeric(working["prediction_mean"], errors="coerce").fillna(0.0)
        working["prediction_std"] = pd.to_numeric(working["prediction_std"], errors="coerce").fillna(0.0)
        working["null_rates_dict"] = working["null_rates"].apply(_null_rate_dict)
        working["max_null_rate"] = working["null_rates_dict"].apply(
            lambda values: max(values.values()) if values else 0.0
        )
        return working.sort_values("window_end_ts").reset_index(drop=True)

    def get_null_rate_history(self, model_id: str, top_n: int = 5) -> pd.DataFrame:
        history = self.get_quality_history(model_id)
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

    def _latest_quality_map(self, model_ids: list[str]) -> dict[str, dict[str, object]]:
        if not model_ids:
            return {}
        placeholders = _sql_placeholders(len(model_ids))
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

    def _latest_drift_snapshot_map(self, model_ids: list[str], metric: str) -> dict[str, dict[str, object]]:
        if not model_ids:
            return {}
        placeholders = _sql_placeholders(len(model_ids))
        frame = self._warehouse.query_params(
            f"""
            WITH ranked_drift AS (
                SELECT
                    model_key,
                    feature_name,
                    metric_name,
                    metric_value,
                    window_end,
                    computed_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY model_key, feature_name, metric_name
                        ORDER BY window_end DESC, computed_at DESC
                    ) AS row_num
                FROM {self.repository.table_names.drift_metrics}
                WHERE model_key IN ({placeholders})
            )
            SELECT model_key, feature_name, metric_name, metric_value, window_end, computed_at
            FROM ranked_drift
            WHERE row_num = 1
            ORDER BY model_key, feature_name, metric_name
            """,
            tuple(model_ids),
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
                "drifting_features": int((metric_series > 0.1).sum()) if not metric_series.empty else 0,
                "total_features": int(len(group.index)),
                "top_drifter": top_drifter,
            }
        return drift_map

    def get_overview_rows(self, metric: str = "psi") -> list[dict]:
        models = self.list_models()
        model_ids = [str(model["id"]) for model in models if str(model.get("id") or "").strip()]
        quality_map = self._latest_quality_map(model_ids)
        drift_map = self._latest_drift_snapshot_map(model_ids, metric)
        rows: list[dict] = []
        for model in models:
            drift = drift_map.get(model["id"], {})
            quality = quality_map.get(model["id"], {})
            rows.append(
                {
                    "model_id": model["id"],
                    "model_name": model["name"],
                    "description": model["description"],
                    "versions": model["versions"],
                    "max_psi": _safe_float(drift.get("max_metric")),
                    "avg_psi": _safe_float(drift.get("avg_metric")),
                    "avg_js": _safe_float(drift.get("avg_js")),
                    "drifting_features": int(drift.get("drifting_features") or 0),
                    "total_features": int(drift.get("total_features") or 0),
                    "top_drifter": str(drift.get("top_drifter") or "N/A"),
                    "max_null_rate": _safe_float(quality.get("max_null_rate")),
                    "has_labels": model["has_labels"],
                    "computing": not bool(drift),
                    "freshness_status": model["freshness_status"],
                    "last_run_status": model["last_run_status"],
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
        try:
            frame = self.repository.load_monitor_frame(
                config,
                start_date=start_date or None,
                end_date=end_date or None,
                feature_columns=feature_columns or config.contract.feature_columns,
                sample_rows_per_day=settings.feature_detail_sample_rows_per_day,
                max_total_rows=settings.feature_detail_max_rows,
            )
        except TypeError:
            try:
                frame = self.repository.load_monitor_frame(
                    config,
                    start_date=start_date or None,
                    end_date=end_date or None,
                    feature_columns=feature_columns or config.contract.feature_columns,
                    max_total_rows=settings.feature_detail_max_rows,
                )
            except TypeError:
                frame = self.repository.load_monitor_frame(config)
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
        try:
            frame = self.repository.load_monitor_frame(
                config,
                start_date=start_date,
                end_date=end_date,
                feature_columns=feature_columns or config.contract.feature_columns,
                sample_rows_per_day=settings.feature_detail_sample_rows_per_day,
                max_total_rows=settings.feature_detail_max_rows,
            )
        except TypeError:
            try:
                frame = self.repository.load_monitor_frame(
                    config,
                    start_date=start_date,
                    end_date=end_date,
                    feature_columns=feature_columns or config.contract.feature_columns,
                    max_total_rows=settings.feature_detail_max_rows,
                )
            except TypeError:
                frame = self.repository.load_monitor_frame(config)
                if config.contract.timestamp_col in frame.columns:
                    timestamps = pd.to_datetime(frame[config.contract.timestamp_col], errors="coerce")
                    if start_date:
                        frame = frame.loc[timestamps >= pd.Timestamp(start_date)]
                    if end_date:
                        frame = frame.loc[timestamps < _inclusive_end_bound(end_date)]
        return config, frame

    def _latest_window_bounds(self, model_id: str) -> dict[str, str] | None:
        comparison_windows = getattr(self.repository.table_names, "comparison_windows", "")
        if not comparison_windows:
            return None
        frame = self._warehouse.query_params(
            f"""
            SELECT baseline_start, baseline_end, window_start, window_end
            FROM {comparison_windows}
            WHERE model_key = %s
            ORDER BY window_end DESC, created_at DESC
            LIMIT 1
            """,
            (model_id,),
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

    def _feature_samples_from_daily_profiles(self, model_id: str, feature: str) -> tuple[pd.Series, pd.Series]:
        daily_feature_profiles = getattr(self.repository.table_names, "daily_feature_profiles", "")
        if not daily_feature_profiles:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        bounds = self._latest_window_bounds(model_id)
        if not bounds:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        min_profile_date = min(
            value
            for value in (
                bounds["baseline_start"],
                bounds["baseline_end"],
                bounds["window_start"],
                bounds["window_end"],
            )
            if value
        )
        max_profile_date = max(
            value
            for value in (
                bounds["baseline_start"],
                bounds["baseline_end"],
                bounds["window_start"],
                bounds["window_end"],
            )
            if value
        )
        frame = self._warehouse.query_params(
            f"""
            SELECT profile_date, distribution_json
            FROM {daily_feature_profiles}
            WHERE model_key = %s
              AND feature_name = %s
              AND profile_date BETWEEN CAST(%s AS DATE) AND CAST(%s AS DATE)
            ORDER BY profile_date
            """,
            (model_id, feature, min_profile_date, max_profile_date),
        )
        if frame.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        baseline_values: list[float] = []
        current_values: list[float] = []
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
            if not sample_values:
                continue
            if baseline_start <= profile_date <= baseline_end:
                baseline_values.extend(sample_values)
            elif window_start <= profile_date <= window_end:
                current_values.extend(sample_values)
        return pd.Series(baseline_values, dtype=float), pd.Series(current_values, dtype=float)

    def get_feature_distribution(self, model_id: str, feature: str) -> tuple[pd.Series, pd.Series]:
        baseline_samples, current_samples = self._feature_samples_from_daily_profiles(model_id, feature)
        if not baseline_samples.empty and not current_samples.empty:
            return baseline_samples, current_samples
        config, baseline, current = self._load_baseline_current(model_id, feature_columns=(feature,))
        if not config or feature not in baseline.columns or feature not in current.columns:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        return (
            pd.to_numeric(baseline[feature], errors="coerce").dropna(),
            pd.to_numeric(current[feature], errors="coerce").dropna(),
        )

    def get_dimension_breakdown(self, model_id: str, feature: str, dimension: str) -> pd.DataFrame:
        config, current = self._load_current_window_frame(model_id, feature_columns=(feature, dimension))
        if not config or feature not in current.columns or dimension not in current.columns:
            return pd.DataFrame()
        working = current[[dimension, feature]].copy()
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
        breakdown = (
            working.groupby(dimension, dropna=False)
            .agg(
                feature_mean=(feature, "mean"),
                feature_std=(feature, "std"),
                null_pct=(feature, lambda values: float(values.isna().mean() * 100)),
                row_count=(feature, "size"),
            )
            .reset_index()
            .rename(columns={dimension: "dimension_value"})
            .sort_values("row_count", ascending=False)
            .head(20)
        )
        breakdown["feature_std"] = breakdown["feature_std"].fillna(0.0)
        return breakdown

    def get_prediction_distribution(self, model_id: str) -> pd.Series:
        config, current = self._load_current_window_frame(model_id, feature_columns=None)
        if not config or config.contract.prediction_col not in current.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(current[config.contract.prediction_col], errors="coerce").dropna()

    def get_performance_rows(self, model_id: str, metric_name: str = "f1") -> pd.DataFrame:
        frame = self._warehouse.query_params(
            f"""
            SELECT *
            FROM {self.repository.table_names.performance_metrics}
            WHERE model_key = %s AND metric_name = %s
            ORDER BY window_end, feature_name, bin_label
            """,
            (model_id, metric_name),
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
            }
        frame = frame.copy()
        frame["window_end"] = pd.to_datetime(frame["window_end"], errors="coerce")
        dated = frame[frame["window_end"].notna()].copy()
        timeline = (
            [
                {
                    "period": str(window_end.date()),
                    metric_name: float(
                        (group["current_metric"] * group["current_volume_pct"]).sum()
                        / max(group["current_volume_pct"].sum(), 1)
                    ),
                }
                for window_end, group in dated.groupby("window_end", sort=True)
            ]
            if not dated.empty
            else []
        )
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
        }

    def get_reference_data(self, model_id: str) -> dict:
        config = self.get_monitor_config(model_id, status=None)
        summary = self.repository.get_monitor_summary()
        summary_row = summary[summary["model_key"] == model_id]
        runtime_state = self.repository.get_monitor_runtime_state(model_id) if hasattr(self.repository, "get_monitor_runtime_state") else None
        return {
            "config": config,
            "status": config.status if config else "",
            "summary": summary_row.iloc[0].to_dict() if not summary_row.empty else {},
            "runtime_state": runtime_state.__dict__ if runtime_state else {},
            "recent_runs": (
                self.repository.get_recent_refresh_runs(model_id, limit=8)
                if hasattr(self.repository, "get_recent_refresh_runs")
                else []
            ),
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
                "use_lakebase_read_model": settings.use_lakebase_read_model,
                "lakebase_database_name": settings.lakebase_database_name,
                "genie_space_id": settings.genie_space_id,
            },
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
