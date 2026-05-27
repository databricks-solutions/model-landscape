from __future__ import annotations

import subprocess
from pathlib import Path

from model_landscape.manual_setup import (
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
            bootstrap_refresh_job_name="ml-drift-monitor-bootstrap-refresh",
        )
    )

    assert 'value: "wh-123"' in text
    assert "valueFrom: sql_warehouse" not in text
    assert 'value: "ml-drift-monitor-refresh"' in text
    assert 'value: "ml-drift-monitor-bootstrap-refresh"' in text


def test_build_manual_refresh_job_payload_uses_workspace_wheel_path() -> None:
    payload = build_manual_refresh_job_payload(
        ManualRefreshJobSettings(
            app_name="ml-drift-monitor",
            wheel_workspace_path="/Workspace/Users/test/model-landscape-manual/dist/model_landscape-0.1.0-py3-none-any.whl",
            sql_warehouse_id="wh-123",
            control_plane_catalog="gc_prod_mlproduct",
            control_plane_schema="mlp_rsch",
            node_type_id="m5d.large",
            use_lakebase_read_model=True,
            lakebase_instance_name="lakebase-instance",
            lakebase_database_name="lakebase-db",
            lakebase_pguser="lakebase-user",
        )
    )

    assert payload["name"] == "ml-drift-monitor-refresh"
    task = payload["tasks"][0]
    assert task["python_wheel_task"]["package_name"] == "model_landscape"
    assert task["python_wheel_task"]["entry_point"] == "model-landscape-refresh"
    assert payload["parameters"][3] == {"name": "scope", "default": "scheduler"}
    assert task["python_wheel_task"]["named_parameters"]["scope"] == "{{job.parameters.scope}}"
    assert task["python_wheel_task"]["named_parameters"]["model-key"] == "{{job.parameters.model_key}}"
    assert payload["schedule"]["quartz_cron_expression"] == "0 0 * * * ?"
    assert payload["schedule"]["pause_status"] == "UNPAUSED"
    assert payload["job_clusters"][0]["new_cluster"]["node_type_id"] == "m5d.large"
    assert payload["job_clusters"][0]["new_cluster"]["data_security_mode"] == "USER_ISOLATION"
    assert payload["job_clusters"][0]["new_cluster"]["num_workers"] == 4
    assert task["job_cluster_key"] == "refresh_compute"
    assert task["libraries"][0]["whl"].endswith(".whl")
    assert task["timeout_seconds"] == 14400
    assert any(
        library.get("pypi", {}).get("package") == "mlflow-skinny>=2.20,<3.0"
        for library in task["libraries"][1:]
    )
    assert task["python_wheel_task"]["named_parameters"]["use-lakebase-read-model"] == "{{job.parameters.use_lakebase_read_model}}"


def test_prepare_existing_app_source_script_writes_manual_app_yaml(tmp_path: Path) -> None:
    output_dir = tmp_path / "manual-source"
    result = subprocess.run(
        [
            "python3",
            str(REPO_ROOT / "notebooks" / "prepare_existing_app_source.py"),
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
    assert (output_dir / "model_landscape" / "app.py").exists()
    assert (output_dir / "requirements.txt").exists()
    assert "Prepared existing-app source tree" in result.stdout
