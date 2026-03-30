from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from model_lens.services.control_plane import ControlPlaneRepository
from model_lens.services.refresh_engine import generate_window_pairs, refresh_monitor_backfill


@dataclass(frozen=True)
class RefreshCounts:
    models: int
    drift_rows: int
    quality_rows: int
    performance_rows: int
    incident_rows: int


def _frame_date_range(frame: pd.DataFrame, timestamp_col: str) -> tuple[str | None, str | None]:
    if frame.empty or timestamp_col not in frame.columns:
        return None, None
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce").dropna()
    if timestamps.empty:
        return None, None
    return str(timestamps.min().date()), str(timestamps.max().date())


def _window_keys(pairs: list[tuple[object, object, dict[str, str]]]) -> set[tuple[str, str, str, str]]:
    return {
        (
            metadata["baseline_start"],
            metadata["baseline_end"],
            metadata["window_start"],
            metadata["window_end"],
        )
        for _, _, metadata in pairs
    }


def run_refresh_cycle(
    repository: ControlPlaneRepository,
    model_key: str = "",
    mode: str = "auto",
) -> RefreshCounts:
    configs = repository.list_monitor_configs(status="active")
    if model_key:
        configs = [config for config in configs if config.model_key == model_key]

    model_count = 0
    drift_count = 0
    quality_count = 0
    performance_count = 0
    incident_count = 0

    for config in configs:
        frame = repository.load_monitor_frame(config)
        actual_mode = mode
        data_min_date, data_max_date = _frame_date_range(frame, config.contract.timestamp_col)
        run_id: str | None = None
        try:
            if frame.empty:
                run_id = repository.start_refresh_run(
                    model_key=config.model_key,
                    requested_mode=mode,
                    run_kind="skipped",
                    data_min_date=data_min_date,
                    data_max_date=data_max_date,
                )
                repository.complete_refresh_run(run_id, status="skipped")
                continue

            all_pairs = generate_window_pairs(frame, config.contract.timestamp_col, config.baseline, model_key=config.model_key)
            if not all_pairs:
                run_id = repository.start_refresh_run(
                    model_key=config.model_key,
                    requested_mode=mode,
                    run_kind="skipped",
                    data_min_date=data_min_date,
                    data_max_date=data_max_date,
                )
                repository.complete_refresh_run(
                    run_id,
                    status="skipped",
                    error_message="No comparable windows available for the configured baseline policy.",
                )
                continue

            existing_keys = repository.get_existing_window_keys(config.model_key)
            if mode == "auto":
                if not existing_keys:
                    actual_mode = "backfill"
                elif existing_keys.issubset(_window_keys(all_pairs)):
                    actual_mode = "incremental"
                else:
                    actual_mode = "backfill"

            run_id = repository.start_refresh_run(
                model_key=config.model_key,
                requested_mode=mode,
                run_kind=actual_mode,
                data_min_date=data_min_date,
                data_max_date=data_max_date,
            )
            prior_open_incidents = repository.get_current_incident_state(config.model_key) if actual_mode == "incremental" else {}
            if actual_mode == "incremental":
                result = refresh_monitor_backfill(
                    config,
                    frame,
                    existing_window_keys=existing_keys,
                    prior_open_incidents=prior_open_incidents,
                )
            else:
                result = refresh_monitor_backfill(config, frame, prior_open_incidents=prior_open_incidents)
            if not any((result.drift_rows, result.quality_rows, result.performance_rows, result.incident_rows)):
                repository.complete_refresh_run(run_id, status="skipped")
                continue
            if actual_mode == "incremental":
                repository.append_refresh_result(config.model_key, result, source_run_id=run_id)
            else:
                repository.replace_all_refresh_results(config.model_key, result, source_run_id=run_id)
            repository.complete_refresh_run(
                run_id,
                status="completed",
                window_count=len(result.window_rows),
                drift_row_count=len(result.drift_rows),
                quality_row_count=len(result.quality_rows),
                performance_row_count=len(result.performance_rows),
                incident_row_count=len(result.incident_rows),
            )
            model_count += 1
            drift_count += len(result.drift_rows)
            quality_count += len(result.quality_rows)
            performance_count += len(result.performance_rows)
            incident_count += len(result.incident_rows)
        except Exception as error:
            if run_id:
                repository.complete_refresh_run(run_id, status="failed", error_message=str(error))
            raise

    return RefreshCounts(
        models=model_count,
        drift_rows=drift_count,
        quality_rows=quality_count,
        performance_rows=performance_count,
        incident_rows=incident_count,
    )
