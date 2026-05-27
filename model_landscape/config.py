from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _optional_bool(name: str) -> bool | None:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not a valid integer; using default %s", name, raw, default)
        return default
    if min_value is not None and value < min_value:
        logger.warning("%s=%r is below minimum %s; using default %s", name, raw, min_value, default)
        return default
    return value


@dataclass(frozen=True)
class Settings:
    app_title: str = os.getenv("APP_TITLE", "Model Landscape")
    control_plane_catalog: str = os.getenv("CONTROL_PLANE_CATALOG", "model_observability")
    control_plane_schema: str = os.getenv("CONTROL_PLANE_SCHEMA", "control_plane")
    sql_warehouse_id: str = os.getenv("SQL_WAREHOUSE_ID", "")
    refresh_job_id: str = os.getenv("REFRESH_JOB_ID", "")
    refresh_job_name: str = os.getenv("REFRESH_JOB_NAME", "model-landscape-refresh")
    bootstrap_refresh_job_id: str = os.getenv("BOOTSTRAP_REFRESH_JOB_ID", "")
    bootstrap_refresh_job_name: str = os.getenv("BOOTSTRAP_REFRESH_JOB_NAME", "")
    genie_space_id: str = os.getenv("GENIE_SPACE_ID", "")
    lakebase_instance_name: str = os.getenv("LAKEBASE_INSTANCE_NAME", "")
    lakebase_database_name: str = os.getenv("LAKEBASE_DATABASE_NAME", os.getenv("PGDATABASE", ""))
    lakebase_host: str = os.getenv("LAKEBASE_HOST", os.getenv("PGHOST", ""))
    lakebase_port: int = _env_int(
        "LAKEBASE_PORT", _env_int("PGPORT", 5432, min_value=1), min_value=1
    )
    lakebase_pguser: str = os.getenv("LAKEBASE_PGUSER", os.getenv("PGUSER", ""))
    lakebase_password: str = os.getenv("LAKEBASE_PASSWORD", os.getenv("PGPASSWORD", ""))
    lakebase_sslmode: str = os.getenv("LAKEBASE_SSLMODE", os.getenv("PGSSLMODE", "require"))
    lakebase_schema: str = os.getenv("LAKEBASE_SCHEMA", "model_landscape_ui")
    refresh_failure_backoff_minutes: int = _env_int(
        "REFRESH_FAILURE_BACKOFF_MINUTES", 30, min_value=0
    )
    refresh_stale_run_minutes: int = _env_int("REFRESH_STALE_RUN_MINUTES", 75, min_value=1)
    max_bootstraps_per_run: int = _env_int("MAX_BOOTSTRAPS_PER_RUN", 5, min_value=1)
    max_drift_monitors_per_run: int = _env_int("MAX_DRIFT_MONITORS_PER_RUN", 20, min_value=1)
    max_performance_monitors_per_run: int = _env_int(
        "MAX_PERFORMANCE_MONITORS_PER_RUN", 20, min_value=1
    )
    max_parallel_refresh_workers: int = _env_int("MAX_PARALLEL_REFRESH_WORKERS", 2, min_value=1)
    refresh_sample_rows_per_day: int = _env_int("REFRESH_SAMPLE_ROWS_PER_DAY", 50000, min_value=1)
    refresh_max_rows_per_window: int = _env_int("REFRESH_MAX_ROWS_PER_WINDOW", 250000, min_value=1)
    feature_detail_sample_rows_per_day: int = _env_int(
        "FEATURE_DETAIL_SAMPLE_ROWS_PER_DAY", 50000, min_value=1
    )
    feature_detail_max_rows: int = _env_int("FEATURE_DETAIL_MAX_ROWS", 200000, min_value=1)
    use_lakebase_read_model: bool = (
        _optional_bool("USE_LAKEBASE_READ_MODEL")
        if _optional_bool("USE_LAKEBASE_READ_MODEL") is not None
        else bool(
            os.getenv("LAKEBASE_INSTANCE_NAME", "").strip()
            or os.getenv("LAKEBASE_DATABASE_NAME", "").strip()
            or os.getenv("LAKEBASE_HOST", "").strip()
            or os.getenv("PGDATABASE", "").strip()
            or os.getenv("PGHOST", "").strip()
        )
    )
    app_port: int = _env_int("DATABRICKS_APP_PORT", 8080, min_value=1)


settings = Settings()
