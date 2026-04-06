from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


REFRESH_JOB_PYPI_DEPENDENCIES: tuple[str, ...] = (
    "dash>=2.18,<3.0",
    "dash-bootstrap-components>=1.6,<2.0",
    "databricks-sdk>=0.81,<1.0",
    "databricks-sql-connector>=3.0,<4.0",
    "mlflow-skinny>=2.20,<3.0",
    "numpy>=1.26,<3.0",
    "pandas>=2.2,<3.0",
    "plotly>=5.24,<6.0",
    "psycopg[binary]>=3.2,<4.0",
    "scikit-learn>=1.5,<2.0",
)

MANUAL_SOURCE_ITEMS: tuple[str, ...] = (
    "requirements.txt",
    "pyproject.toml",
    "src",
)


def _yaml_string(value: str) -> str:
    return json.dumps(value)


@dataclass(frozen=True)
class ManualAppSettings:
    app_name: str
    sql_warehouse_id: str
    control_plane_catalog: str
    control_plane_schema: str
    refresh_job_id: str = ""
    refresh_job_name: str = ""
    lakebase_instance_name: str = ""
    lakebase_database_name: str = ""
    lakebase_schema: str = "model_lens_ui"
    genie_space_id: str = ""

    def resolved_refresh_job_name(self) -> str:
        if self.refresh_job_id.strip():
            return self.refresh_job_name.strip()
        if self.refresh_job_name.strip():
            return self.refresh_job_name.strip()
        return f"{self.app_name}-refresh"


@dataclass(frozen=True)
class ManualRefreshJobSettings:
    app_name: str
    wheel_workspace_path: str
    sql_warehouse_id: str
    control_plane_catalog: str
    control_plane_schema: str
    spark_version: str = "15.4.x-scala2.12"
    node_type_id: str = ""
    num_workers: int = 4
    timeout_seconds: int = 14400
    lakebase_instance_name: str = ""
    lakebase_database_name: str = ""
    lakebase_pguser: str = ""
    lakebase_schema: str = "model_lens_ui"
    use_lakebase_read_model: bool = False


def build_manual_app_yaml(settings: ManualAppSettings) -> str:
    refresh_job_name = settings.resolved_refresh_job_name()
    lines = [
        "command:",
        "  - python",
        "  - -m",
        "  - model_lens.app",
        "",
        "env:",
        "  - name: APP_TITLE",
        f"    value: {_yaml_string('Model Lens')}",
        "  - name: PYTHONPATH",
        f"    value: {_yaml_string('src')}",
        "  - name: CONTROL_PLANE_CATALOG",
        f"    value: {_yaml_string(settings.control_plane_catalog)}",
        "  - name: CONTROL_PLANE_SCHEMA",
        f"    value: {_yaml_string(settings.control_plane_schema)}",
        "  - name: SQL_WAREHOUSE_ID",
        f"    value: {_yaml_string(settings.sql_warehouse_id)}",
        "  - name: LAKEBASE_INSTANCE_NAME",
        f"    value: {_yaml_string(settings.lakebase_instance_name)}",
        "  - name: LAKEBASE_DATABASE_NAME",
        f"    value: {_yaml_string(settings.lakebase_database_name)}",
        "  - name: LAKEBASE_SCHEMA",
        f"    value: {_yaml_string(settings.lakebase_schema)}",
        "  - name: REFRESH_JOB_ID",
        f"    value: {_yaml_string(settings.refresh_job_id.strip())}",
        "  - name: REFRESH_JOB_NAME",
        f"    value: {_yaml_string(refresh_job_name)}",
        "  - name: GENIE_SPACE_ID",
        f"    value: {_yaml_string(settings.genie_space_id)}",
    ]
    return "\n".join(lines) + "\n"


def build_manual_refresh_job_payload(settings: ManualRefreshJobSettings) -> dict[str, object]:
    named_parameters: dict[str, str] = {
        "warehouse-id": settings.sql_warehouse_id,
        "catalog": settings.control_plane_catalog,
        "schema": settings.control_plane_schema,
        "scope": "scheduler",
    }
    if settings.use_lakebase_read_model:
        named_parameters.update(
            {
                "use-lakebase-read-model": "true",
                "lakebase-instance-name": settings.lakebase_instance_name,
                "lakebase-database-name": settings.lakebase_database_name,
                "lakebase-pguser": settings.lakebase_pguser,
                "lakebase-schema": settings.lakebase_schema,
            }
        )

    return {
        "name": f"{settings.app_name}-refresh",
        "max_concurrent_runs": 1,
        "queue": {"enabled": True},
        "job_clusters": [
            {
                "job_cluster_key": "refresh_compute",
                "new_cluster": {
                    "spark_version": settings.spark_version,
                    "node_type_id": settings.node_type_id,
                    "num_workers": settings.num_workers,
                },
            }
        ],
        "tasks": [
            {
                "task_key": "refresh_control_plane",
                "python_wheel_task": {
                    "package_name": "model_lens",
                    "entry_point": "model-lens-refresh",
                    "named_parameters": named_parameters,
                },
                "job_cluster_key": "refresh_compute",
                "libraries": [
                    {"whl": settings.wheel_workspace_path},
                    *[
                        {"pypi": {"package": dependency}}
                        for dependency in REFRESH_JOB_PYPI_DEPENDENCIES
                    ],
                ],
                "max_retries": 2,
                "min_retry_interval_millis": 60000,
                "timeout_seconds": settings.timeout_seconds,
            }
        ],
        "schedule": {
            "quartz_cron_expression": "0 0 * * * ?",
            "timezone_id": "UTC",
            "pause_status": "UNPAUSED",
        },
    }


def find_built_wheel(dist_dir: Path) -> Path | None:
    wheels = sorted(dist_dir.glob("model_lens-*.whl"))
    return wheels[-1] if wheels else None
