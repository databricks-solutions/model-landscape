from __future__ import annotations

from dataclasses import dataclass

from ml_drift_monitor_next.services.control_plane import ControlPlaneRepository
from ml_drift_monitor_next.services.refresh_engine import refresh_monitor


@dataclass(frozen=True)
class RefreshCounts:
    models: int
    drift_rows: int
    quality_rows: int
    performance_rows: int
    incident_rows: int


def run_refresh_cycle(repository: ControlPlaneRepository, model_key: str = "") -> RefreshCounts:
    repository.ensure_control_plane()
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
        if frame.empty:
            continue
        result = refresh_monitor(config, frame)
        if not any((result.drift_rows, result.quality_rows, result.performance_rows, result.incident_rows)):
            continue
        repository.replace_refresh_result(config.model_key, result)
        model_count += 1
        drift_count += len(result.drift_rows)
        quality_count += len(result.quality_rows)
        performance_count += len(result.performance_rows)
        incident_count += len(result.incident_rows)

    return RefreshCounts(
        models=model_count,
        drift_rows=drift_count,
        quality_rows=quality_count,
        performance_rows=performance_count,
        incident_rows=incident_count,
    )
