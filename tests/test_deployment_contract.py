from __future__ import annotations

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_jobs_bundle_uses_spark_cluster_libraries() -> None:
    text = (REPO_ROOT / "resources" / "jobs.yml").read_text()

    assert "job_clusters:" in text
    assert "new_cluster:" in text
    assert "libraries:" in text
    assert "../dist/*.whl" in text
    assert "${var.lakebase_instance_name}" in text
    assert "${var.lakebase_database_name}" in text
    assert "${var.lakebase_pguser}" in text
    assert "mlflow-skinny>=2.20,<3.0" in text


def test_app_resource_uses_only_supported_sql_warehouse_binding() -> None:
    text = (REPO_ROOT / "resources" / "app.yml").read_text()

    assert "database:" not in text
    assert "\n          job:" not in text
    assert "sql_warehouse:" in text


def test_app_yaml_does_not_require_unsupported_job_binding() -> None:
    text = (REPO_ROOT / "app.yaml").read_text()

    assert "valueFrom: refresh_job" not in text
    assert 'value: ""' in text
    assert '  - name: CONTROL_PLANE_CATALOG\n    value: ""' in text
    assert '  - name: CONTROL_PLANE_SCHEMA\n    value: ""' in text


def test_bundle_requires_explicit_control_plane_namespace_vars() -> None:
    text = (REPO_ROOT / "databricks.yml").read_text()

    assert "control_plane_catalog:" in text
    assert "control_plane_schema:" in text
    assert "description: Existing control-plane catalog for this workspace" in text
    assert "description: Existing control-plane schema for this workspace" in text
    assert "control_plane_catalog:\n    default:" not in text
    assert "control_plane_schema:\n    default:" not in text


def test_wrapper_scripts_avoid_serverless_fragile_path_patterns() -> None:
    for relative_path in (
        "notebooks/model_landscape_refresh.py",
        "notebooks/model_landscape_setup.py",
        "model_landscape/workflows/refresh_job.py",
        "model_landscape/workflows/setup_control_plane.py",
    ):
        text = (REPO_ROOT / relative_path).read_text()
        assert "Path(__file__).resolve()" not in text
        assert "raise SystemExit" not in text


def test_bundle_declares_spark_refresh_job_variables() -> None:
    text = (REPO_ROOT / "databricks.yml").read_text()

    assert "refresh_spark_version:" in text
    assert "refresh_data_security_mode:" in text
    assert "refresh_node_type_id:" in text
    assert "refresh_num_workers:" in text
    assert "refresh_timeout_seconds:" in text
    assert "warehouse_only" in text
    assert "default: true" in text


def test_project_dev_dependencies_include_local_spark_support() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    dev_dependencies = payload["project"]["optional-dependencies"]["dev"]

    assert "pyspark>=3.5,<4.0" in dev_dependencies
