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
    "model_landscape",
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
    bootstrap_refresh_job_id: str = ""
    bootstrap_refresh_job_name: str = ""
    lakebase_instance_name: str = ""
    lakebase_database_name: str = ""
    lakebase_schema: str = "model_landscape_ui"
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
    data_security_mode: str = "USER_ISOLATION"
    node_type_id: str = ""
    num_workers: int = 4
    timeout_seconds: int = 14400
    lakebase_instance_name: str = ""
    lakebase_database_name: str = ""
    lakebase_host: str = ""
    lakebase_port: int = 5432
    lakebase_pguser: str = ""
    lakebase_sslmode: str = "require"
    lakebase_schema: str = "model_landscape_ui"
    use_lakebase_read_model: bool = False


def build_manual_app_yaml(settings: ManualAppSettings) -> str:
    refresh_job_name = settings.resolved_refresh_job_name()
    lines = [
        "command:",
        "  - python",
        "  - -m",
        "  - model_landscape.app",
        "",
        "env:",
        "  - name: APP_TITLE",
        f"    value: {_yaml_string('Model Landscape')}",
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
        "  - name: BOOTSTRAP_REFRESH_JOB_ID",
        f"    value: {_yaml_string(settings.bootstrap_refresh_job_id.strip())}",
        "  - name: BOOTSTRAP_REFRESH_JOB_NAME",
        f"    value: {_yaml_string(settings.bootstrap_refresh_job_name.strip())}",
        "  - name: GENIE_SPACE_ID",
        f"    value: {_yaml_string(settings.genie_space_id)}",
    ]
    return "\n".join(lines) + "\n"


def build_manual_refresh_job_payload(settings: ManualRefreshJobSettings) -> dict[str, object]:
    job_parameters = [
        {"name": "warehouse_id", "default": settings.sql_warehouse_id},
        {"name": "control_plane_catalog", "default": settings.control_plane_catalog},
        {"name": "control_plane_schema", "default": settings.control_plane_schema},
        {"name": "scope", "default": "scheduler"},
        {"name": "model_key", "default": ""},
        {
            "name": "use_lakebase_read_model",
            "default": "true" if settings.use_lakebase_read_model else "false",
        },
        {"name": "lakebase_instance_name", "default": settings.lakebase_instance_name},
        {"name": "lakebase_database_name", "default": settings.lakebase_database_name},
        {"name": "lakebase_host", "default": settings.lakebase_host},
        {"name": "lakebase_port", "default": str(settings.lakebase_port)},
        {"name": "lakebase_pguser", "default": settings.lakebase_pguser},
        {"name": "lakebase_sslmode", "default": settings.lakebase_sslmode},
        {"name": "lakebase_schema", "default": settings.lakebase_schema},
    ]

    return {
        "name": f"{settings.app_name}-refresh",
        "max_concurrent_runs": 1,
        "queue": {"enabled": True},
        "job_clusters": [
            {
                "job_cluster_key": "refresh_compute",
                "new_cluster": {
                    "spark_version": settings.spark_version,
                    "data_security_mode": settings.data_security_mode,
                    "node_type_id": settings.node_type_id,
                    "num_workers": settings.num_workers,
                },
            }
        ],
        "parameters": job_parameters,
        "tasks": [
            {
                "task_key": "refresh_control_plane",
                "python_wheel_task": {
                    "package_name": "model_landscape",
                    "entry_point": "model-landscape-refresh",
                    "named_parameters": {
                        "warehouse-id": "{{job.parameters.warehouse_id}}",
                        "catalog": "{{job.parameters.control_plane_catalog}}",
                        "schema": "{{job.parameters.control_plane_schema}}",
                        "scope": "{{job.parameters.scope}}",
                        "model-key": "{{job.parameters.model_key}}",
                        "use-lakebase-read-model": "{{job.parameters.use_lakebase_read_model}}",
                        "lakebase-instance-name": "{{job.parameters.lakebase_instance_name}}",
                        "lakebase-database-name": "{{job.parameters.lakebase_database_name}}",
                        "lakebase-host": "{{job.parameters.lakebase_host}}",
                        "lakebase-port": "{{job.parameters.lakebase_port}}",
                        "lakebase-pguser": "{{job.parameters.lakebase_pguser}}",
                        "lakebase-sslmode": "{{job.parameters.lakebase_sslmode}}",
                        "lakebase-schema": "{{job.parameters.lakebase_schema}}",
                    },
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
    wheels = sorted(dist_dir.glob("model_landscape-*.whl"))
    return wheels[-1] if wheels else None
