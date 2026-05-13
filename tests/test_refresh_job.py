from __future__ import annotations

import sys
from types import SimpleNamespace

from model_landscape.workflows import refresh_job


def test_parse_args_accepts_explicit_lakebase_bool_value(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model-landscape-refresh",
            "--catalog",
            "model_observability",
            "--schema",
            "control_plane",
            "--warehouse-id",
            "wh-123",
            "--use-lakebase-read-model",
            "false",
        ],
    )

    args = refresh_job.parse_args()

    assert args.use_lakebase_read_model == "false"


def test_parse_args_accepts_raw_job_parameter_underscore_names(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model-landscape-refresh",
            "--warehouse_id=wh-123",
            "--control_plane_catalog=hive_metastore",
            "--control_plane_schema=model_landscape_control_plane",
            "--model_key=fraud_model_demo",
            "--use_lakebase_read_model=false",
            "--lakebase_database_name=",
        ],
    )

    args = refresh_job.parse_args()

    assert args.warehouse_id == "wh-123"
    assert args.catalog == "hive_metastore"
    assert args.schema == "model_landscape_control_plane"
    assert args.model_key == "fraud_model_demo"
    assert args.use_lakebase_read_model == "false"
    assert args.lakebase_database_name == ""


def test_refresh_job_defaults_single_model_scheduler_run_to_bootstrap(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        refresh_job,
        "parse_args",
        lambda: SimpleNamespace(
            catalog="model_observability",
            schema="control_plane",
            warehouse_id="wh-123",
            model_key="fraud_model_demo",
            use_lakebase_read_model="false",
            lakebase_instance_name="",
            lakebase_database_name="",
            lakebase_host="",
            lakebase_port=5432,
            lakebase_pguser="",
            lakebase_password="",
            lakebase_sslmode="require",
            lakebase_schema="",
            scope="scheduler",
            mode="auto",
        ),
    )
    monkeypatch.setattr(refresh_job, "build_refresh_repository", lambda **kwargs: object())

    def _fake_run_refresh_cycle(repository, *, model_key, mode, scope):
        captured["model_key"] = model_key
        captured["mode"] = mode
        captured["scope"] = scope
        return SimpleNamespace(models=1, drift_rows=2, quality_rows=3, performance_rows=4, incident_rows=5)

    monkeypatch.setattr(refresh_job, "run_refresh_cycle", _fake_run_refresh_cycle)

    assert refresh_job.main() == 0
    assert captured == {
        "model_key": "fraud_model_demo",
        "mode": "auto",
        "scope": "bootstrap",
    }
