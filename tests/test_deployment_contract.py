from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_jobs_bundle_uses_serverless_environment_dependencies() -> None:
    text = (REPO_ROOT / "resources" / "jobs.yml").read_text()

    assert "libraries:" not in text
    assert "../dist/*.whl" in text
    assert "${var.lakebase_instance_name}" not in text
    assert "${var.lakebase_database_name}" not in text
    assert "${var.lakebase_pguser}" not in text


def test_app_resource_uses_only_supported_sql_warehouse_binding() -> None:
    text = (REPO_ROOT / "resources" / "app.yml").read_text()

    assert "database:" not in text
    assert "\n          job:" not in text
    assert "sql_warehouse:" in text


def test_app_yaml_does_not_require_unsupported_job_binding() -> None:
    text = (REPO_ROOT / "app.yaml").read_text()

    assert "valueFrom: refresh_job" not in text
    assert 'value: ""' in text


def test_wrapper_scripts_avoid_serverless_fragile_path_patterns() -> None:
    for relative_path in (
        "scripts/model_lens_refresh.py",
        "scripts/model_lens_setup.py",
        "src/model_lens/workflows/refresh_job.py",
        "src/model_lens/workflows/setup_control_plane.py",
    ):
        text = (REPO_ROOT / relative_path).read_text()
        assert "Path(__file__).resolve()" not in text
        assert "raise SystemExit" not in text


def test_lakebase_parameters_are_added_only_in_lakebase_targets() -> None:
    text = (REPO_ROOT / "databricks.yml").read_text()

    assert "default: true" in text
    assert "warehouse_only" in text
    assert text.count("--use-lakebase-read-model") == 2
