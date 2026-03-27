from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_title: str = os.getenv("APP_TITLE", "ML Drift Monitor Next")
    control_plane_catalog: str = os.getenv("CONTROL_PLANE_CATALOG", "model_observability")
    control_plane_schema: str = os.getenv("CONTROL_PLANE_SCHEMA", "control_plane")
    sql_warehouse_id: str = os.getenv("SQL_WAREHOUSE_ID", "")
    refresh_job_id: str = os.getenv("REFRESH_JOB_ID", "")
    genie_space_id: str = os.getenv("GENIE_SPACE_ID", "")
    app_port: int = int(os.getenv("DATABRICKS_APP_PORT", "8080"))


settings = Settings()

