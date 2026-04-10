from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta, timezone
from typing import Literal

import pandas as pd

from model_lens.config import settings
from model_lens.domain.models import MonitorConfig, MonitorRuntimeState, RefreshResult
from model_lens.services.control_plane import ControlPlaneRepository
from model_lens.services.refresh_engine import (
    build_daily_class_feature_profile_rows,
    build_daily_class_quality_profile_rows,
    build_daily_label_metric_rows,
    build_daily_feature_profile_rows,
    build_daily_performance_profile_rows,
    build_daily_quality_profile_rows,
    build_performance_bin_specs,
    build_quality_rows_from_profile,
    derive_refresh_result_from_daily_profiles,
    generate_window_metadata,
)
from model_lens.services.spark_refresh import SparkDailyProfiles


RefreshScope = Literal["scheduler", "bootstrap", "drift_quality", "performance_repair"]


@dataclass(frozen=True)
class RefreshCounts:
    models: int
    drift_rows: int
    quality_rows: int
    performance_rows: int
    incident_rows: int


@dataclass(frozen=True)
class MonitorRefreshResult:
    model_key: str
    scope: Literal["bootstrap", "drift_quality", "performance_repair"]
    status: Literal["completed", "skipped", "failed"]
    counts: RefreshCounts
    error: str | None = None


@dataclass(frozen=True)
class RefreshBatchResult:
    requested_scope: RefreshScope
    requested_mode: str
    results: tuple[MonitorRefreshResult, ...]

    @property
    def models(self) -> int:
        return sum(result.counts.models for result in self.results)

    @property
    def drift_rows(self) -> int:
        return sum(result.counts.drift_rows for result in self.results)

    @property
    def quality_rows(self) -> int:
        return sum(result.counts.quality_rows for result in self.results)

    @property
    def performance_rows(self) -> int:
        return sum(result.counts.performance_rows for result in self.results)

    @property
    def incident_rows(self) -> int:
        return sum(result.counts.incident_rows for result in self.results)


@dataclass(frozen=True)
class RefreshTarget:
    config: MonitorConfig
    scope: Literal["bootstrap", "drift_quality", "performance_repair"]
    scheduled_at: pd.Timestamp


@dataclass(frozen=True)
class TargetSelectionResult:
    targets: tuple[RefreshTarget, ...]
    debug_lines: tuple[str, ...] = ()


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=timezone.utc)


def _coerce_timestamp(value: object) -> pd.Timestamp | None:
    if value in (None, "", pd.NaT):
        return None
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)


def _frame_date_range(frame: pd.DataFrame, timestamp_col: str) -> tuple[str | None, str | None]:
    if frame.empty or timestamp_col not in frame.columns:
        return None, None
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce").dropna()
    if timestamps.empty:
        return None, None
    return str(timestamps.min().date()), str(timestamps.max().date())


def _drift_cadence_delta(preset: str) -> timedelta | None:
    normalized = (preset or "6h").strip().lower()
    if normalized == "hourly":
        return timedelta(hours=1)
    if normalized == "6h":
        return timedelta(hours=6)
    if normalized == "daily":
        return timedelta(days=1)
    return None


def _performance_cadence_delta(preset: str) -> timedelta | None:
    normalized = (preset or "disabled").strip().lower()
    if normalized == "6h_3d_repair":
        return timedelta(hours=6)
    if normalized in {"daily_7d_repair", "daily_14d_repair"}:
        return timedelta(days=1)
    return None


def _performance_repair_days(preset: str) -> int | None:
    normalized = (preset or "disabled").strip().lower()
    if normalized == "6h_3d_repair":
        return 3
    if normalized == "daily_7d_repair":
        return 7
    if normalized == "daily_14d_repair":
        return 14
    return None


def _default_runtime_state(config: MonitorConfig) -> MonitorRuntimeState:
    now = _utc_now().isoformat()
    return MonitorRuntimeState(
        model_key=config.model_key,
        bootstrap_status="pending",
        next_drift_due_at=now if config.schedule_enabled else None,
        next_performance_due_at=(
            now
            if config.schedule_enabled and config.has_labels and config.performance_cadence_preset != "disabled"
            else None
        ),
        last_run_started_at=None,
        last_run_completed_at=None,
        backoff_until=None,
        consecutive_failures=0,
    )


def _inferred_runtime_state(
    repository: ControlPlaneRepository,
    config: MonitorConfig,
    *,
    now: pd.Timestamp | None = None,
) -> MonitorRuntimeState:
    try:
        existing_window_keys = repository.get_existing_window_keys(config.model_key)
    except Exception:
        existing_window_keys = set()
    state = _default_runtime_state(config)
    if not existing_window_keys:
        return state
    due_now = now or _utc_now()
    return MonitorRuntimeState(
        model_key=config.model_key,
        bootstrap_status="completed",
        next_drift_due_at=(
            due_now.isoformat()
            if config.schedule_enabled and config.drift_cadence_preset != "manual"
            else None
        ),
        next_performance_due_at=(
            due_now.isoformat()
            if config.schedule_enabled and config.has_labels and config.performance_cadence_preset not in {"disabled", "manual"}
            else None
        ),
    )


def _next_due_iso(completed_at: pd.Timestamp, delta: timedelta | None, enabled: bool = True) -> str | None:
    if not enabled or delta is None:
        return None
    return (completed_at + delta).isoformat()


def _bootstrap_range(config: MonitorConfig, latest_date: pd.Timestamp) -> tuple[str | None, str | None]:
    latest = latest_date.normalize()
    if config.baseline.kind == "fixed" and config.baseline.baseline_start:
        return config.baseline.baseline_start, latest.date().isoformat()
    lookback_days = config.baseline.max_comparison_days + (2 * config.baseline.n_days) - 2
    start = latest - timedelta(days=max(lookback_days, 0))
    return start.date().isoformat(), latest.date().isoformat()


def _drift_range(config: MonitorConfig, state: MonitorRuntimeState, latest_date: pd.Timestamp) -> tuple[str | None, str | None]:
    latest = latest_date.normalize()
    if config.baseline.kind == "fixed" and config.baseline.baseline_start:
        return config.baseline.baseline_start, latest.date().isoformat()
    last_refresh = _coerce_timestamp(state.last_drift_refresh_at)
    if last_refresh is None:
        return _bootstrap_range(config, latest)
    start = last_refresh.normalize() - timedelta(days=max((2 * config.baseline.n_days) - 2, 0))
    return start.date().isoformat(), latest.date().isoformat()


def _performance_range(config: MonitorConfig, latest_date: pd.Timestamp) -> tuple[str | None, str | None]:
    latest = latest_date.normalize()
    repair_days = _performance_repair_days(config.performance_cadence_preset)
    if repair_days is None:
        return None, None
    if config.baseline.kind == "fixed" and config.baseline.baseline_start:
        return config.baseline.baseline_start, latest.date().isoformat()
    start = latest - timedelta(days=max(repair_days + (2 * config.baseline.n_days) - 2, 0))
    return start.date().isoformat(), latest.date().isoformat()


def _failed_recently(state: MonitorRuntimeState, now: pd.Timestamp) -> bool:
    if state is None:
        return False
    backoff_until = _coerce_timestamp(getattr(state, "backoff_until", None))
    if backoff_until is None:
        return False
    return backoff_until > now


def _is_running(repository: ControlPlaneRepository, model_key: str, scope: str) -> bool:
    return _get_latest_refresh_run(repository, model_key, scope, statuses=("running",)) is not None


def _safe_state_value(state: MonitorRuntimeState | object | None, field: str, default: object = "") -> object:
    if state is None:
        return default
    return getattr(state, field, default)


def _normalized_bootstrap_status(state: MonitorRuntimeState | object | None) -> str:
    value = str(_safe_state_value(state, "bootstrap_status", "pending") or "").strip().lower()
    return value or "pending"


def _positive_selection_limit(configured_limit: int, *, stage: str, debug_lines: list[str]) -> int:
    if configured_limit >= 1:
        return configured_limit
    debug_lines.append(
        f"stage={stage} outcome=warning reason=non_positive_limit_clamped "
        f"configured_limit={configured_limit} effective_limit=1"
    )
    return 1


def _emit_selection_debug_lines(debug_lines: tuple[str, ...]) -> None:
    for line in debug_lines:
        print(f"refresh-control-plane selection: {line}")


def _is_due(next_due_at: str | None, now: pd.Timestamp) -> bool:
    if not next_due_at:
        return False
    due_at = _coerce_timestamp(next_due_at)
    if due_at is None:
        return False
    return due_at <= now


def _select_targets(
    repository: ControlPlaneRepository,
    configs: list[MonitorConfig],
    *,
    model_key: str,
    scope: RefreshScope,
) -> TargetSelectionResult:
    now = _utc_now()
    runtime_states = _list_runtime_states(repository, configs)
    config_map = {config.model_key: config for config in configs}
    table_names = getattr(repository, "table_names", None)
    namespace = getattr(table_names, "namespace", "")
    warehouse_id = getattr(getattr(repository, "_warehouse", None), "_warehouse_id", "")
    debug_lines = [
        f"requested_scope={scope} requested_model_key={model_key or ''} active_configs={len(configs)} "
        f"namespace={namespace} warehouse_id={warehouse_id}",
    ]

    if model_key:
        config = config_map.get(model_key)
        if not config:
            debug_lines.append(f"requested_model_key={model_key} outcome=missing_active_config")
            return TargetSelectionResult(targets=(), debug_lines=tuple(debug_lines))
        forced_scope = "bootstrap" if scope == "scheduler" else scope
        if forced_scope == "scheduler":
            forced_scope = "bootstrap"
        debug_lines.append(
            f"stage={forced_scope} model_key={model_key} outcome=selected reason=explicit_target"
        )
        return TargetSelectionResult(
            targets=(RefreshTarget(config=config, scope=forced_scope, scheduled_at=now),),
            debug_lines=tuple(debug_lines),
        )

    def runtime_state_for(config: MonitorConfig) -> MonitorRuntimeState:
        return runtime_states.get(config.model_key) or _get_runtime_state(repository, config, now=now)

    targets: list[RefreshTarget] = []
    selected_model_keys: set[str] = set()

    if scope in {"scheduler", "bootstrap"}:
        pending_bootstraps = []
        configured_bootstrap_limit = settings.max_bootstraps_per_run
        bootstrap_limit = _positive_selection_limit(
            configured_bootstrap_limit,
            stage="bootstrap",
            debug_lines=debug_lines,
        )
        for config in configs:
            state = runtime_state_for(config)
            bootstrap_status = _normalized_bootstrap_status(state)
            if bootstrap_status == "completed":
                debug_lines.append(
                    f"stage=bootstrap model_key={config.model_key} outcome=skipped "
                    f"reason=bootstrap_already_completed bootstrap_status={bootstrap_status}"
                )
                continue
            if _is_running(repository, config.model_key, "bootstrap"):
                debug_lines.append(
                    f"stage=bootstrap model_key={config.model_key} outcome=skipped "
                    f"reason=bootstrap_run_already_running"
                )
                continue
            if _failed_recently(state, now):
                backoff_until = str(_safe_state_value(state, "backoff_until", "") or "")
                debug_lines.append(
                    f"stage=bootstrap model_key={config.model_key} outcome=skipped "
                    f"reason=recent_failure_backoff backoff_until={backoff_until}"
                )
                continue
            pending_bootstraps.append(RefreshTarget(config=config, scope="bootstrap", scheduled_at=now))
        selected_bootstraps = pending_bootstraps[:bootstrap_limit]
        targets.extend(selected_bootstraps)
        selected_model_keys.update(target.config.model_key for target in selected_bootstraps)
        for target in selected_bootstraps:
            debug_lines.append(
                f"stage=bootstrap model_key={target.config.model_key} outcome=selected "
                f"reason=pending_bootstrap"
            )
        for target in pending_bootstraps[bootstrap_limit:]:
            debug_lines.append(
                f"stage=bootstrap model_key={target.config.model_key} outcome=skipped "
                f"reason=max_bootstraps_per_run_reached configured_limit={configured_bootstrap_limit} "
                f"effective_limit={bootstrap_limit}"
            )
        debug_lines.append(
            f"stage=bootstrap candidates={len(pending_bootstraps)} selected={len(selected_bootstraps)} "
            f"configured_limit={configured_bootstrap_limit} effective_limit={bootstrap_limit}"
        )
        if scope == "bootstrap":
            return TargetSelectionResult(targets=tuple(targets), debug_lines=tuple(debug_lines))

    if scope in {"scheduler", "drift_quality"}:
        due_drift: list[RefreshTarget] = []
        configured_drift_limit = settings.max_drift_monitors_per_run
        drift_limit = _positive_selection_limit(
            configured_drift_limit,
            stage="drift_quality",
            debug_lines=debug_lines,
        )
        for config in configs:
            if scope == "scheduler" and config.model_key in selected_model_keys:
                continue
            state = runtime_state_for(config)
            if _normalized_bootstrap_status(state) != "completed":
                continue
            if not config.schedule_enabled or config.drift_cadence_preset == "manual":
                continue
            if not _is_due(state.next_drift_due_at, now):
                continue
            if _is_running(repository, config.model_key, "drift_quality"):
                continue
            if _failed_recently(state, now):
                continue
            due_drift.append(RefreshTarget(config=config, scope="drift_quality", scheduled_at=now))
        if scope == "drift_quality":
            return TargetSelectionResult(targets=tuple(due_drift[:drift_limit]), debug_lines=tuple(debug_lines))
        selected_drift = due_drift[:drift_limit]
        targets.extend(selected_drift)
        selected_model_keys.update(target.config.model_key for target in selected_drift)

    if scope in {"scheduler", "performance_repair"}:
        due_performance: list[RefreshTarget] = []
        configured_performance_limit = settings.max_performance_monitors_per_run
        performance_limit = _positive_selection_limit(
            configured_performance_limit,
            stage="performance_repair",
            debug_lines=debug_lines,
        )
        for config in configs:
            if scope == "scheduler" and config.model_key in selected_model_keys:
                continue
            state = runtime_state_for(config)
            if _normalized_bootstrap_status(state) != "completed":
                continue
            if not config.schedule_enabled or not config.has_labels:
                continue
            if config.performance_cadence_preset in {"disabled", "manual"}:
                continue
            if not _is_due(state.next_performance_due_at, now):
                continue
            if _is_running(repository, config.model_key, "performance_repair"):
                continue
            if _failed_recently(state, now):
                continue
            due_performance.append(RefreshTarget(config=config, scope="performance_repair", scheduled_at=now))
        if scope == "performance_repair":
            return TargetSelectionResult(targets=tuple(due_performance[:performance_limit]), debug_lines=tuple(debug_lines))
        targets.extend(due_performance[:performance_limit])

    return TargetSelectionResult(targets=tuple(targets), debug_lines=tuple(debug_lines))


def _list_runtime_states(repository: ControlPlaneRepository, configs: list[MonitorConfig]) -> dict[str, MonitorRuntimeState]:
    if hasattr(repository, "list_monitor_runtime_states"):
        return repository.list_monitor_runtime_states([config.model_key for config in configs])
    return {}


def _get_runtime_state(
    repository: ControlPlaneRepository,
    config: MonitorConfig,
    *,
    now: pd.Timestamp | None = None,
) -> MonitorRuntimeState:
    if hasattr(repository, "get_monitor_runtime_state"):
        state = repository.get_monitor_runtime_state(config.model_key)
        if state is not None:
            return state
    return _inferred_runtime_state(repository, config, now=now)


def _upsert_runtime_state(repository: ControlPlaneRepository, state: MonitorRuntimeState) -> None:
    if hasattr(repository, "upsert_monitor_runtime_state"):
        repository.upsert_monitor_runtime_state(state)


def _get_source_date_range(repository: ControlPlaneRepository, config: MonitorConfig) -> tuple[str | None, str | None]:
    if hasattr(repository, "get_source_date_range"):
        return repository.get_source_date_range(config)
    frame = repository.load_monitor_frame(config)
    return _frame_date_range(frame, config.contract.timestamp_col)


def _get_source_profile(
    repository: ControlPlaneRepository,
    config: MonitorConfig,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, object]:
    if hasattr(repository, "get_source_profile"):
        return repository.get_source_profile(config, start_date=start_date, end_date=end_date)
    try:
        frame = repository.load_monitor_frame(config, start_date=start_date, end_date=end_date)
    except TypeError:
        frame = repository.load_monitor_frame(config)
    min_date, max_date = _frame_date_range(frame, config.contract.timestamp_col)
    return {
        "total_rows": int(len(frame)),
        "min_date": min_date,
        "max_date": max_date,
        "prediction_mean": pd.to_numeric(frame.get(config.contract.prediction_col, pd.Series(dtype=float)), errors="coerce").mean(),
        "prediction_std": pd.to_numeric(frame.get(config.contract.prediction_col, pd.Series(dtype=float)), errors="coerce").std(),
        "daily_volume": {
            str(index): int(value)
            for index, value in frame.groupby(pd.to_datetime(frame[config.contract.timestamp_col], errors="coerce").dt.date).size().items()
        } if not frame.empty and config.contract.timestamp_col in frame.columns else {},
        "null_rates": {
            feature: round(float(frame[feature].isna().mean() * 100), 2)
            for feature in config.contract.feature_columns
            if feature in frame.columns
        },
        "label_row_count": (
            int(pd.to_numeric(frame.get(config.contract.label_col, pd.Series(dtype=float)), errors="coerce").notna().sum())
            if config.contract.label_col
            else 0
        ),
    }


def _get_label_watermark(
    repository: ControlPlaneRepository,
    config: MonitorConfig,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str | None:
    if hasattr(repository, "get_label_watermark"):
        try:
            return repository.get_label_watermark(config, start_date=start_date, end_date=end_date)
        except TypeError:
            return repository.get_label_watermark(config)
    return None


def _has_daily_label_metric_rows(
    repository: ControlPlaneRepository,
    model_key: str,
) -> bool:
    if hasattr(repository, "has_daily_label_metric_rows"):
        try:
            return bool(repository.has_daily_label_metric_rows(model_key))
        except TypeError:
            return bool(repository.has_daily_label_metric_rows(model_key=model_key))
        except Exception:
            # Preserve the existing skip behavior when the existence check is unavailable.
            return True
    return True


def _merge_daily_rows(
    persisted_rows: list[dict[str, object]],
    current_rows: list[dict[str, object]],
    *,
    key_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    merged: dict[tuple[object, ...], dict[str, object]] = {}
    for row in persisted_rows:
        key = tuple(row.get(field) for field in key_fields)
        merged[key] = dict(row)
    for row in current_rows:
        key = tuple(row.get(field) for field in key_fields)
        merged[key] = dict(row)
    return list(merged.values())


def _load_persisted_daily_rows(
    repository: ControlPlaneRepository,
    config: MonitorConfig,
    *,
    start_date: str,
    end_date: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    quality_rows = (
        repository.get_daily_quality_profile_rows(config.model_key, start_date=start_date, end_date=end_date)
        if hasattr(repository, "get_daily_quality_profile_rows")
        else []
    )
    feature_rows = (
        repository.get_daily_feature_profile_rows(config.model_key, start_date=start_date, end_date=end_date)
        if hasattr(repository, "get_daily_feature_profile_rows")
        else []
    )
    performance_rows = (
        repository.get_daily_performance_profile_rows(config.model_key, start_date=start_date, end_date=end_date)
        if hasattr(repository, "get_daily_performance_profile_rows")
        else []
    )
    return quality_rows, feature_rows, performance_rows


def _get_latest_refresh_run(
    repository: ControlPlaneRepository,
    model_key: str,
    scope: str,
    *,
    statuses: tuple[str, ...] | None = None,
) -> dict | None:
    if hasattr(repository, "get_latest_refresh_run"):
        return repository.get_latest_refresh_run(model_key, scope, statuses=statuses)
    return None


def _start_refresh_run(repository: ControlPlaneRepository, **kwargs) -> str:
    try:
        return repository.start_refresh_run(**kwargs)
    except TypeError:
        fallback = {
            "model_key": kwargs["model_key"],
            "requested_mode": kwargs["requested_mode"],
            "run_kind": kwargs["run_kind"],
            "data_min_date": kwargs.get("data_min_date"),
            "data_max_date": kwargs.get("data_max_date"),
        }
        return repository.start_refresh_run(**fallback)


def _update_refresh_run_metadata(repository: ControlPlaneRepository, run_id: str, **kwargs) -> None:
    if hasattr(repository, "update_refresh_run_metadata"):
        repository.update_refresh_run_metadata(run_id, **kwargs)


def _get_stale_running_refresh_runs(repository: ControlPlaneRepository, started_before: str) -> list[dict]:
    if hasattr(repository, "get_stale_running_refresh_runs"):
        return repository.get_stale_running_refresh_runs(started_before)
    return []


def _schedule_state_after_success(
    config: MonitorConfig,
    state: MonitorRuntimeState,
    *,
    scope: Literal["bootstrap", "drift_quality", "performance_repair"],
    completed_at: pd.Timestamp,
    label_watermark: str | None = None,
) -> MonitorRuntimeState:
    drift_next_due = state.next_drift_due_at
    performance_next_due = state.next_performance_due_at
    last_drift_refresh_at = state.last_drift_refresh_at
    last_performance_refresh_at = state.last_performance_refresh_at

    if scope in {"bootstrap", "drift_quality"}:
        last_drift_refresh_at = completed_at.isoformat()
        drift_next_due = _next_due_iso(
            completed_at,
            _drift_cadence_delta(config.drift_cadence_preset),
            enabled=config.schedule_enabled,
        )
    if scope in {"bootstrap", "performance_repair"}:
        last_performance_refresh_at = completed_at.isoformat()
        performance_next_due = _next_due_iso(
            completed_at,
            _performance_cadence_delta(config.performance_cadence_preset),
            enabled=config.schedule_enabled and config.has_labels,
        )

    return MonitorRuntimeState(
        model_key=config.model_key,
        bootstrap_status="completed",
        last_drift_refresh_at=last_drift_refresh_at,
        last_performance_refresh_at=last_performance_refresh_at,
        next_drift_due_at=drift_next_due,
        next_performance_due_at=performance_next_due,
        last_label_watermark=label_watermark if label_watermark is not None else state.last_label_watermark,
        last_run_status="completed",
        last_run_error=None,
        last_run_started_at=state.last_run_started_at,
        last_run_completed_at=completed_at.isoformat(),
        backoff_until=None,
        consecutive_failures=0,
    )


def _schedule_state_after_skip(
    config: MonitorConfig,
    state: MonitorRuntimeState,
    *,
    scope: Literal["bootstrap", "drift_quality", "performance_repair"],
    completed_at: pd.Timestamp,
    message: str,
) -> MonitorRuntimeState:
    next_drift_due_at = state.next_drift_due_at
    next_performance_due_at = state.next_performance_due_at
    bootstrap_status = state.bootstrap_status
    if scope == "bootstrap":
        bootstrap_status = "pending"
        next_drift_due_at = _next_due_iso(completed_at, timedelta(hours=1), enabled=True)
        if config.has_labels and config.performance_cadence_preset != "disabled":
            next_performance_due_at = _next_due_iso(completed_at, timedelta(hours=1), enabled=True)
    elif scope == "drift_quality":
        next_drift_due_at = _next_due_iso(
            completed_at,
            _drift_cadence_delta(config.drift_cadence_preset),
            enabled=config.schedule_enabled,
        )
    elif scope == "performance_repair":
        next_performance_due_at = _next_due_iso(
            completed_at,
            _performance_cadence_delta(config.performance_cadence_preset),
            enabled=config.schedule_enabled and config.has_labels,
        )
    return MonitorRuntimeState(
        model_key=config.model_key,
        bootstrap_status=bootstrap_status,
        last_drift_refresh_at=state.last_drift_refresh_at,
        last_performance_refresh_at=state.last_performance_refresh_at,
        next_drift_due_at=next_drift_due_at,
        next_performance_due_at=next_performance_due_at,
        last_label_watermark=state.last_label_watermark,
        last_run_status="skipped",
        last_run_error=message,
        last_run_started_at=state.last_run_started_at,
        last_run_completed_at=completed_at.isoformat(),
        backoff_until=None,
        consecutive_failures=0,
    )


def _schedule_state_running(
    state: MonitorRuntimeState,
    *,
    started_at: pd.Timestamp,
) -> MonitorRuntimeState:
    return MonitorRuntimeState(
        model_key=state.model_key,
        bootstrap_status=state.bootstrap_status,
        last_drift_refresh_at=state.last_drift_refresh_at,
        last_performance_refresh_at=state.last_performance_refresh_at,
        next_drift_due_at=state.next_drift_due_at,
        next_performance_due_at=state.next_performance_due_at,
        last_label_watermark=state.last_label_watermark,
        last_run_status="running",
        last_run_error=None,
        last_run_started_at=started_at.isoformat(),
        last_run_completed_at=getattr(state, "last_run_completed_at", None),
        backoff_until=None,
        consecutive_failures=getattr(state, "consecutive_failures", 0),
    )


def _schedule_state_after_failure(
    state: MonitorRuntimeState,
    *,
    completed_at: pd.Timestamp,
    error_message: str,
) -> MonitorRuntimeState:
    return MonitorRuntimeState(
        model_key=state.model_key,
        bootstrap_status=state.bootstrap_status,
        last_drift_refresh_at=state.last_drift_refresh_at,
        last_performance_refresh_at=state.last_performance_refresh_at,
        next_drift_due_at=state.next_drift_due_at,
        next_performance_due_at=state.next_performance_due_at,
        last_label_watermark=state.last_label_watermark,
        last_run_status="failed",
        last_run_error=error_message,
        last_run_started_at=state.last_run_started_at,
        last_run_completed_at=completed_at.isoformat(),
        backoff_until=(completed_at + timedelta(minutes=settings.refresh_failure_backoff_minutes)).isoformat(),
        consecutive_failures=max(state.consecutive_failures, 0) + 1,
    )


def _reconcile_stale_running_runs(repository: ControlPlaneRepository, now: pd.Timestamp) -> None:
    stale_before = (now - timedelta(minutes=settings.refresh_stale_run_minutes)).isoformat()
    stale_runs = _get_stale_running_refresh_runs(repository, stale_before)
    if not stale_runs:
        return
    for row in stale_runs:
        run_id = str(row.get("run_id") or "").strip()
        model_key = str(row.get("model_key") or "").strip()
        if not run_id or not model_key:
            continue
        message = "Marked failed after exceeding stale-run timeout."
        repository.complete_refresh_run(run_id, status="failed", error_message=message)
        state = repository.get_monitor_runtime_state(model_key) if hasattr(repository, "get_monitor_runtime_state") else None
        if state is None:
            continue
        _upsert_runtime_state(
            repository,
            _schedule_state_after_failure(
                state,
                completed_at=now,
                error_message=message,
            ),
        )


def _unexpected_failure_result(target: RefreshTarget, error: Exception) -> MonitorRefreshResult:
    return MonitorRefreshResult(
        model_key=target.config.model_key,
        scope=target.scope,
        status="failed",
        counts=RefreshCounts(models=0, drift_rows=0, quality_rows=0, performance_rows=0, incident_rows=0),
        error=str(error),
    )


def _load_window_frame(
    repository: ControlPlaneRepository,
    config: MonitorConfig,
    *,
    range_start: str,
    range_end: str,
) -> pd.DataFrame:
    try:
        return repository.load_monitor_frame(
            config,
            start_date=range_start,
            end_date=range_end,
            feature_columns=config.contract.feature_columns,
            sample_rows_per_day=settings.refresh_sample_rows_per_day,
            max_total_rows=settings.refresh_max_rows_per_window,
        )
    except TypeError:
        try:
            return repository.load_monitor_frame(
                config,
                start_date=range_start,
                end_date=range_end,
                feature_columns=config.contract.feature_columns,
                max_total_rows=settings.refresh_max_rows_per_window,
            )
        except TypeError:
            try:
                return repository.load_monitor_frame(
                    config,
                    start_date=range_start,
                    end_date=range_end,
                    feature_columns=config.contract.feature_columns,
                )
            except TypeError:
                return repository.load_monitor_frame(config)


def _build_range_daily_profiles(
    repository: ControlPlaneRepository,
    config: MonitorConfig,
    *,
    range_start: str,
    range_end: str,
    computed_at: str,
    include_drift_quality: bool,
    include_performance: bool,
    existing_bin_specs: dict[str, tuple[float, ...]] | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, tuple[float, ...]],
]:
    if hasattr(repository, "build_daily_profiles"):
        spark_profiles = repository.build_daily_profiles(
            config,
            start_date=range_start,
            end_date=range_end,
            computed_at=computed_at,
            include_drift_quality=include_drift_quality,
            include_performance=include_performance,
            existing_bin_specs=existing_bin_specs,
        )
        if isinstance(spark_profiles, SparkDailyProfiles):
            return (
                list(spark_profiles.daily_quality_profile_rows),
                list(spark_profiles.daily_class_quality_profile_rows),
                list(spark_profiles.daily_feature_profile_rows),
                list(spark_profiles.daily_class_feature_profile_rows),
                list(spark_profiles.daily_performance_profile_rows),
                list(spark_profiles.daily_label_metric_rows),
                dict(spark_profiles.performance_bin_specs),
            )
    range_frame = _load_window_frame(
        repository,
        config,
        range_start=range_start,
        range_end=range_end,
    )
    daily_quality_profile_rows = (
        build_daily_quality_profile_rows(config=config, inference_df=range_frame, computed_at=computed_at)
        if include_drift_quality
        else []
    )
    daily_class_quality_profile_rows = (
        build_daily_class_quality_profile_rows(config=config, inference_df=range_frame, computed_at=computed_at)
        if include_drift_quality
        else []
    )
    daily_feature_profile_rows = (
        build_daily_feature_profile_rows(config=config, inference_df=range_frame, computed_at=computed_at)
        if include_drift_quality
        else []
    )
    daily_class_feature_profile_rows = (
        build_daily_class_feature_profile_rows(config=config, inference_df=range_frame, computed_at=computed_at)
        if include_drift_quality
        else []
    )
    performance_bin_specs = (
        build_performance_bin_specs(
            config=config,
            inference_df=range_frame,
            existing_specs=existing_bin_specs,
        )
        if include_performance
        else {}
    )
    daily_performance_profile_rows = (
        build_daily_performance_profile_rows(
            config=config,
            inference_df=range_frame,
            computed_at=computed_at,
            bin_specs=performance_bin_specs,
        )
        if include_performance
        else []
    )
    daily_label_metric_rows = (
        build_daily_label_metric_rows(config=config, inference_df=range_frame, computed_at=computed_at)
        if include_performance
        else []
    )
    return (
        daily_quality_profile_rows,
        daily_class_quality_profile_rows,
        daily_feature_profile_rows,
        daily_class_feature_profile_rows,
        daily_performance_profile_rows,
        daily_label_metric_rows,
        performance_bin_specs,
    )


def _execute_target(
    repository: ControlPlaneRepository,
    target: RefreshTarget,
    *,
    requested_mode: str,
) -> MonitorRefreshResult:
    config = target.config
    state = _get_runtime_state(repository, config)
    scheduled_at_text = target.scheduled_at.isoformat()
    started_at = _utc_now()
    run_kind = "backfill" if target.scope == "bootstrap" else "incremental"
    run_id: str | None = _start_refresh_run(
        repository,
        model_key=config.model_key,
        requested_mode=requested_mode,
        run_kind=run_kind,
        scope=target.scope,
        scheduled_at=scheduled_at_text,
    )
    running_state = _schedule_state_running(state, started_at=started_at)
    _upsert_runtime_state(repository, running_state)

    try:
        data_min_date, data_max_date = _get_source_date_range(repository, config)
        if not data_max_date:
            _update_refresh_run_metadata(
                repository,
                run_id,
                data_min_date=data_min_date,
                data_max_date=data_max_date,
            )
            repository.complete_refresh_run(
                run_id,
                status="skipped",
                error_message="No source rows available for this monitor.",
            )
            _upsert_runtime_state(
                repository,
                _schedule_state_after_skip(
                    config,
                    running_state,
                    scope=target.scope,
                    completed_at=_utc_now(),
                    message="No source rows available for this monitor.",
                )
            )
            return MonitorRefreshResult(
                model_key=config.model_key,
                scope=target.scope,
                status="skipped",
                counts=RefreshCounts(models=0, drift_rows=0, quality_rows=0, performance_rows=0, incident_rows=0),
            )

        latest_date = pd.Timestamp(data_max_date)
        if target.scope == "bootstrap":
            range_start, range_end = _bootstrap_range(config, latest_date)
        elif target.scope == "drift_quality":
            range_start, range_end = _drift_range(config, state, latest_date)
        else:
            range_start, range_end = _performance_range(config, latest_date)

        profile = _get_source_profile(repository, config, start_date=range_start, end_date=range_end)
        bounded_min_date = str(profile.get("min_date") or "") or None
        bounded_max_date = str(profile.get("max_date") or "") or None
        _update_refresh_run_metadata(
            repository,
            run_id,
            data_min_date=data_min_date,
            data_max_date=data_max_date,
            range_start=range_start,
            range_end=range_end,
            rows_scanned=int(profile.get("total_rows", 0) or 0),
            label_rows_scanned=int(profile.get("label_row_count", 0) or 0),
        )

        if target.scope == "performance_repair":
            latest_watermark = _get_label_watermark(
                repository,
                config,
                start_date=range_start,
                end_date=range_end,
            )
            has_daily_label_rows = _has_daily_label_metric_rows(repository, config.model_key)
            if (
                latest_watermark
                and latest_watermark == state.last_label_watermark
                and state.last_performance_refresh_at
                and has_daily_label_rows
            ):
                repository.complete_refresh_run(
                    run_id,
                    status="skipped",
                    error_message="Label watermark has not advanced since the previous performance refresh.",
                )
                _upsert_runtime_state(
                    repository,
                    _schedule_state_after_skip(
                        config,
                        running_state,
                        scope=target.scope,
                        completed_at=_utc_now(),
                        message="Label watermark has not advanced since the previous performance refresh.",
                    )
                )
                return MonitorRefreshResult(
                    model_key=config.model_key,
                    scope=target.scope,
                    status="skipped",
                    counts=RefreshCounts(models=0, drift_rows=0, quality_rows=0, performance_rows=0, incident_rows=0),
                )

        if not bounded_max_date or int(profile.get("total_rows", 0) or 0) <= 0:
            repository.complete_refresh_run(run_id, status="skipped", error_message="No rows found inside the refresh range.")
            _upsert_runtime_state(
                repository,
                _schedule_state_after_skip(
                    config,
                    running_state,
                    scope=target.scope,
                    completed_at=_utc_now(),
                    message="No rows found inside the refresh range.",
                )
            )
            return MonitorRefreshResult(
                model_key=config.model_key,
                scope=target.scope,
                status="skipped",
                counts=RefreshCounts(models=0, drift_rows=0, quality_rows=0, performance_rows=0, incident_rows=0),
            )

        existing_window_keys = (
            repository.get_existing_window_keys(config.model_key)
            if target.scope == "drift_quality"
            else None
        )
        metadata_list = generate_window_metadata(
            min_date=bounded_min_date,
            max_date=bounded_max_date,
            baseline=config.baseline,
            model_key=config.model_key,
            existing_window_keys=existing_window_keys if target.scope == "drift_quality" else None,
        )
        if target.scope == "performance_repair":
            repair_days = _performance_repair_days(config.performance_cadence_preset)
            if repair_days is not None:
                cutoff_date = (latest_date.normalize() - timedelta(days=max(repair_days - 1, 0))).date().isoformat()
                metadata_list = [
                    metadata for metadata in metadata_list
                    if str(metadata["window_end"]) >= cutoff_date
                ]
        if not metadata_list:
            repository.complete_refresh_run(
                run_id,
                status="skipped",
                error_message="No new comparison windows are due for this monitor.",
            )
            _upsert_runtime_state(
                repository,
                _schedule_state_after_skip(
                    config,
                    running_state,
                    scope=target.scope,
                    completed_at=_utc_now(),
                    message="No new comparison windows are due for this monitor.",
                )
            )
            return MonitorRefreshResult(
                model_key=config.model_key,
                scope=target.scope,
                status="skipped",
                counts=RefreshCounts(models=0, drift_rows=0, quality_rows=0, performance_rows=0, incident_rows=0),
            )

        include_drift_quality = target.scope != "performance_repair"
        include_performance = target.scope != "drift_quality"
        computed_at_text = _utc_now().isoformat(timespec="seconds")
        quality_rows = (
            build_quality_rows_from_profile(config=config, profile=profile, computed_at=computed_at_text)
            if include_drift_quality
            else []
        )
        existing_bin_specs = (
            repository.get_performance_bin_specs(config.model_key)
            if (include_drift_quality or include_performance)
            and hasattr(repository, "get_performance_bin_specs")
            and target.scope != "bootstrap"
            else {}
        )
        (
            daily_quality_profile_rows,
            daily_class_quality_profile_rows,
            daily_feature_profile_rows,
            daily_class_feature_profile_rows,
            daily_performance_profile_rows,
            daily_label_metric_rows,
            performance_bin_specs,
        ) = _build_range_daily_profiles(
            repository,
            config,
            range_start=range_start or bounded_min_date or data_min_date or bounded_max_date,
            range_end=range_end or bounded_max_date,
            computed_at=computed_at_text,
            include_drift_quality=include_drift_quality,
            include_performance=include_performance,
            existing_bin_specs=existing_bin_specs,
        )
        derivation_start = min(
            [str(metadata["baseline_start"]) for metadata in metadata_list],
            default=range_start or bounded_min_date or data_min_date or bounded_max_date or "",
        )
        derivation_end = max(
            [str(metadata["window_end"]) for metadata in metadata_list],
            default=range_end or bounded_max_date or data_max_date or derivation_start,
        )
        if hasattr(repository, "derive_refresh_result_from_daily_profile_rows"):
            derived_result = repository.derive_refresh_result_from_daily_profile_rows(
                config=config,
                metadata_list=metadata_list,
                current_daily_quality_profile_rows=daily_quality_profile_rows,
                current_daily_class_quality_profile_rows=daily_class_quality_profile_rows,
                current_daily_feature_profile_rows=daily_feature_profile_rows,
                current_daily_class_feature_profile_rows=daily_class_feature_profile_rows,
                current_daily_performance_profile_rows=daily_performance_profile_rows,
                current_daily_label_metric_rows=daily_label_metric_rows,
                derivation_start=derivation_start,
                derivation_end=derivation_end,
                computed_at=computed_at_text,
                prior_open_incidents=repository.get_current_incident_state(config.model_key),
                include_drift_quality=include_drift_quality,
                include_performance=include_performance,
            )
        else:
            merged_daily_quality_rows = list(daily_quality_profile_rows)
            merged_daily_feature_rows = list(daily_feature_profile_rows)
            merged_daily_performance_rows = list(daily_performance_profile_rows)
            if target.scope != "bootstrap" and derivation_start and derivation_end:
                persisted_quality_rows, persisted_feature_rows, persisted_performance_rows = _load_persisted_daily_rows(
                    repository,
                    config,
                    start_date=derivation_start,
                    end_date=derivation_end,
                )
                merged_daily_quality_rows = _merge_daily_rows(
                    persisted_quality_rows,
                    daily_quality_profile_rows,
                    key_fields=("profile_date",),
                )
                merged_daily_feature_rows = _merge_daily_rows(
                    persisted_feature_rows,
                    daily_feature_profile_rows,
                    key_fields=("profile_date", "feature_name"),
                )
                merged_daily_performance_rows = _merge_daily_rows(
                    persisted_performance_rows,
                    daily_performance_profile_rows,
                    key_fields=("profile_date", "feature_name", "bin_label", "metric_name"),
                )
            derived_result = derive_refresh_result_from_daily_profiles(
                config=config,
                metadata_list=metadata_list,
                daily_quality_profile_rows=merged_daily_quality_rows,
                daily_feature_profile_rows=merged_daily_feature_rows,
                daily_performance_profile_rows=merged_daily_performance_rows,
                daily_class_quality_profile_rows=daily_class_quality_profile_rows,
                daily_class_feature_profile_rows=daily_class_feature_profile_rows,
                daily_label_metric_rows=daily_label_metric_rows,
                computed_at=computed_at_text,
                prior_open_incidents=repository.get_current_incident_state(config.model_key),
                include_drift_quality=include_drift_quality,
                include_performance=include_performance,
            )

        if not any((derived_result.drift_rows, quality_rows, derived_result.performance_rows, derived_result.incident_rows, derived_result.window_rows)):
            repository.complete_refresh_run(run_id, status="skipped", error_message="No comparable windows available.")
            _upsert_runtime_state(
                repository,
                _schedule_state_after_skip(
                    config,
                    running_state,
                    scope=target.scope,
                    completed_at=_utc_now(),
                    message="No comparable windows available.",
                )
            )
            return MonitorRefreshResult(
                model_key=config.model_key,
                scope=target.scope,
                status="skipped",
                counts=RefreshCounts(models=0, drift_rows=0, quality_rows=0, performance_rows=0, incident_rows=0),
            )

        result = RefreshResult(
            drift_rows=derived_result.drift_rows,
            quality_rows=quality_rows,
            performance_rows=derived_result.performance_rows,
            incident_rows=derived_result.incident_rows,
            incident_history_rows=derived_result.incident_history_rows,
            quality_history_rows=derived_result.quality_history_rows,
            window_rows=derived_result.window_rows,
            daily_quality_profile_rows=daily_quality_profile_rows,
            daily_feature_profile_rows=daily_feature_profile_rows,
            daily_performance_profile_rows=daily_performance_profile_rows,
            performance_bin_specs=performance_bin_specs,
        )

        if target.scope == "bootstrap":
            generation_id = run_id
            repository.replace_all_refresh_results(config.model_key, result, source_run_id=generation_id)
        else:
            generation_getter = getattr(repository, "get_latest_published_generation_id", None)
            generation_id = generation_getter(config.model_key) if callable(generation_getter) else None
            generation_id = generation_id or run_id
            repository.append_refresh_result(config.model_key, result, source_run_id=generation_id)

        repository.complete_refresh_run(
            run_id,
            status="completed",
            window_count=len(result.window_rows),
            drift_row_count=len(result.drift_rows),
            quality_row_count=len(result.quality_rows),
            performance_row_count=len(result.performance_rows),
            incident_row_count=len(result.incident_rows),
            generation_id=generation_id,
            publish=True,
        )
        completed_at = _utc_now()
        _upsert_runtime_state(
            repository,
            _schedule_state_after_success(
                config,
                running_state,
                scope=target.scope,
                completed_at=completed_at,
                label_watermark=(
                    _get_label_watermark(
                        repository,
                        config,
                        start_date=range_start,
                        end_date=range_end,
                    )
                    if include_performance and config.has_labels
                    else None
                ),
            )
        )
        return MonitorRefreshResult(
            model_key=config.model_key,
            scope=target.scope,
            status="completed",
            counts=RefreshCounts(
                models=1,
                drift_rows=len(result.drift_rows),
                quality_rows=len(result.quality_rows),
                performance_rows=len(result.performance_rows),
                incident_rows=len(result.incident_rows),
            ),
        )
    except Exception as error:
        completed_at = _utc_now()
        error_text = str(error)
        try:
            if run_id:
                repository.complete_refresh_run(run_id, status="failed", error_message=error_text)
            failure_state = _schedule_state_after_failure(
                _schedule_state_running(state, started_at=started_at),
                completed_at=completed_at,
                error_message=error_text,
            )
            _upsert_runtime_state(repository, failure_state)
        except Exception:
            raise
        return MonitorRefreshResult(
            model_key=config.model_key,
            scope=target.scope,
            status="failed",
            counts=RefreshCounts(models=0, drift_rows=0, quality_rows=0, performance_rows=0, incident_rows=0),
            error=error_text,
        )


def run_refresh_cycle(
    repository: ControlPlaneRepository,
    model_key: str = "",
    mode: str = "auto",
    scope: RefreshScope = "scheduler",
) -> RefreshBatchResult:
    _reconcile_stale_running_runs(repository, _utc_now())
    configs = repository.list_monitor_configs(status="active")
    selection = _select_targets(repository, configs, model_key=model_key, scope=scope)
    targets = list(selection.targets)
    if scope == "bootstrap" or model_key or not targets:
        _emit_selection_debug_lines(selection.debug_lines)

    max_workers = max(1, settings.max_parallel_refresh_workers)
    recommended_worker_cap = getattr(repository, "recommended_max_parallel_refresh_workers", None)
    if callable(recommended_worker_cap):
        try:
            max_workers = min(max_workers, max(1, int(recommended_worker_cap())))
        except Exception:
            max_workers = 1
    results: list[MonitorRefreshResult] = []
    if len(targets) <= 1 or max_workers <= 1 or not hasattr(repository, "fork_for_worker"):
        for target in targets:
            try:
                results.append(_execute_target(repository, target, requested_mode=mode))
            except Exception as error:
                results.append(_unexpected_failure_result(target, error))
        batch = RefreshBatchResult(requested_scope=scope, requested_mode=mode, results=tuple(results))
        sync_read_model = getattr(repository, "_sync_read_model", None)
        if callable(sync_read_model) and batch.models > 0:
            sync_read_model()
        return batch

    worker_count = min(max_workers, len(targets))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="model-lens-refresh") as executor:
        future_map = {
            executor.submit(_execute_target, repository.fork_for_worker(), target, requested_mode=mode): target
            for target in targets
        }
        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as error:
                results.append(_unexpected_failure_result(future_map[future], error))
    sync_read_model = getattr(repository, "_sync_read_model", None)
    batch = RefreshBatchResult(requested_scope=scope, requested_mode=mode, results=tuple(results))
    if callable(sync_read_model) and batch.models > 0:
        sync_read_model()
    return batch
