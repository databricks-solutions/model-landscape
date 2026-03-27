from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_title: str = os.getenv("APP_TITLE", "Model Lens")
    control_plane_catalog: str = os.getenv("CONTROL_PLANE_CATALOG", "model_observability")
    control_plane_schema: str = os.getenv("CONTROL_PLANE_SCHEMA", "control_plane")
    sql_warehouse_id: str = os.getenv("SQL_WAREHOUSE_ID", "")
    refresh_job_id: str = os.getenv("REFRESH_JOB_ID", "")
    genie_space_id: str = os.getenv("GENIE_SPACE_ID", "")
    use_lakebase_read_model: bool = os.getenv("USE_LAKEBASE_READ_MODEL", "").strip().lower() in {"1", "true", "yes", "on"}
    lakebase_instance_name: str = os.getenv("LAKEBASE_INSTANCE_NAME", "")
    lakebase_database_name: str = os.getenv("LAKEBASE_DATABASE_NAME", os.getenv("PGDATABASE", ""))
    lakebase_host: str = os.getenv("LAKEBASE_HOST", os.getenv("PGHOST", ""))
    lakebase_port: int = int(os.getenv("LAKEBASE_PORT", os.getenv("PGPORT", "5432")))
    lakebase_pguser: str = os.getenv("LAKEBASE_PGUSER", os.getenv("PGUSER", ""))
    lakebase_password: str = os.getenv("LAKEBASE_PASSWORD", os.getenv("PGPASSWORD", ""))
    lakebase_sslmode: str = os.getenv("LAKEBASE_SSLMODE", os.getenv("PGSSLMODE", "require"))
    lakebase_schema: str = os.getenv("LAKEBASE_SCHEMA", "model_lens_ui")
    app_port: int = int(os.getenv("DATABRICKS_APP_PORT", "8080"))


settings = Settings()
