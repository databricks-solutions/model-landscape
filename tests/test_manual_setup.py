from __future__ import annotations

import subprocess
from pathlib import Path

from model_lens.manual_setup import (
    ManualAppSettings,
    ManualRefreshJobSettings,
    build_manual_app_yaml,
    build_manual_refresh_job_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_manual_app_yaml_uses_literal_warehouse_id() -> None:
    text = build_manual_app_yaml(
        ManualAppSettings(
            app_name="ml-drift-monitor",
            sql_warehouse_id="wh-123",
            control_plane_catalog="gc_prod_mlproduct",
            control_plane_schema="mlp_rsch",
        )
    )

    assert 'value: "wh-123"' in text
    assert "valueFrom: sql_warehouse" not in text
    assert 'value: "ml-drift-monitor-refresh"' in text


def test_build_manual_refresh_job_payload_uses_workspace_wheel_path() -> None:
    payload = build_manual_refresh_job_payload(
        ManualRefreshJobSettings(
            app_name="ml-drift-monitor",
            wheel_workspace_path="/Workspace/Users/test/model-lens-manual/dist/model_lens-0.1.0-py3-none-any.whl",
            sql_warehouse_id="wh-123",
            control_plane_catalog="gc_prod_mlproduct",
            control_plane_schema="mlp_rsch",
            use_lakebase_read_model=True,
            lakebase_instance_name="lakebase-instance",
            lakebase_database_name="lakebase-db",
            lakebase_pguser="lakebase-user",
        )
    )

    assert payload["name"] == "ml-drift-monitor-refresh"
    task = payload["tasks"][0]
    assert task["python_wheel_task"]["package_name"] == "model_lens"
    assert task["python_wheel_task"]["entry_point"] == "model-lens-refresh"
    assert task["python_wheel_task"]["named_parameters"]["scope"] == "scheduler"
    assert payload["schedule"]["quartz_cron_expression"] == "0 0 * * * ?"
    assert payload["schedule"]["pause_status"] == "UNPAUSED"
    assert payload["environments"][0]["spec"]["dependencies"][0].endswith(".whl")
    assert task["python_wheel_task"]["named_parameters"]["use-lakebase-read-model"] == "true"


def test_prepare_existing_app_source_script_writes_manual_app_yaml(tmp_path: Path) -> None:
    output_dir = tmp_path / "manual-source"
    result = subprocess.run(
        [
            "python3",
            str(REPO_ROOT / "scripts" / "prepare_existing_app_source.py"),
            "--app-name",
            "ml-drift-monitor",
            "--sql-warehouse-id",
            "wh-123",
            "--control-plane-catalog",
            "gc_prod_mlproduct",
            "--control-plane-schema",
            "mlp_rsch",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    text = (output_dir / "app.yaml").read_text()
    assert 'value: "wh-123"' in text
    assert "valueFrom: sql_warehouse" not in text
    assert (output_dir / "src" / "model_lens" / "app.py").exists()
    assert (output_dir / "requirements.txt").exists()
    assert "Prepared existing-app source tree" in result.stdout
