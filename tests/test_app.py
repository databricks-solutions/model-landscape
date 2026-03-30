from __future__ import annotations

from collections.abc import Iterator

from dash.development.base_component import Component

from model_lens import app as app_module
from model_lens.app import (
    _control_plane_ready,
    _feature_candidates,
    _non_numeric_features,
    _selected_model_from_search,
    _workspace_lakebase_instances,
    create_app,
)


def _walk(component: Component) -> Iterator[Component]:
    yield component
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            if isinstance(child, Component):
                yield from _walk(child)
    elif isinstance(children, Component):
        yield from _walk(children)


def test_app_layout_exposes_setup_scan_onboarding_and_results() -> None:
    app = create_app()
    shell_ids = {component.id for component in _walk(app.layout) if getattr(component, "id", None)}
    page_ids = {component.id for component in _walk(app.validation_layout) if getattr(component, "id", None)}
    ids = shell_ids | page_ids
    assert {
        "url",
        "page-content",
        "global-model-select",
        "session-config-store",
        "reload-token",
    }.issubset(shell_ids)
    assert {
        "setup-control-plane-btn",
        "control-plane-catalog-input",
        "control-plane-schema-input",
        "lakebase-instance-input",
        "lakebase-database-input",
        "lakebase-schema-input",
        "onboarding-current-step",
        "control-plane-ready-store",
        "wizard-back-btn",
        "wizard-next-btn",
        "wizard-step-guidance",
        "onboarding-review-summary",
        "refresh-all-btn",
        "refresh-selected-btn",
        "source-table-input",
        "labels-table-input",
        "mlflow-experiment-input",
        "mlflow-registered-model-input",
        "scan-source-btn",
        "save-monitor-btn",
        "model-id-value-input",
        "model-version-value-input",
        "labels-order-col-input",
        "monitor-summary",
        "incident-summary",
    }.issubset(ids)


def test_schema_helpers_flag_non_numeric_selected_features() -> None:
    scan_data = {
        "columns": ["event_ts", "model_id", "prediction", "amount", "country"],
        "schema": [
            {"col_name": "event_ts", "data_type": "timestamp"},
            {"col_name": "model_id", "data_type": "string"},
            {"col_name": "prediction", "data_type": "double"},
            {"col_name": "amount", "data_type": "double"},
            {"col_name": "country", "data_type": "string"},
        ],
    }

    assert _feature_candidates(scan_data, ["event_ts", "model_id", "prediction"]) == ["amount", "country"]
    assert _non_numeric_features(["amount", "country"], scan_data) == ["country"]


def test_workspace_lakebase_probe_is_skipped_outside_databricks_app(monkeypatch) -> None:
    monkeypatch.delenv("DATABRICKS_APP_PORT", raising=False)
    app_module._workspace_lakebase_instances.cache_clear()

    assert _workspace_lakebase_instances() == ()


def test_selected_model_from_search_parses_query_string() -> None:
    assert _selected_model_from_search("?model=fraud_model_demo") == "fraud_model_demo"
    assert _selected_model_from_search("?model=fraud_model_demo&foo=bar") == "fraud_model_demo"
    assert _selected_model_from_search("?foo=bar") is None
    assert _selected_model_from_search("") is None


def test_control_plane_ready_requires_successful_setup_state() -> None:
    assert (
        _control_plane_ready(
            {},
            control_plane_catalog="model_observability",
            control_plane_schema="control_plane",
            lakebase_instance_name="",
            lakebase_database_name="",
            lakebase_schema="model_lens_ui",
        )
        is False
    )
    assert (
        _control_plane_ready(
            {
                "control_plane_catalog": "model_observability",
                "control_plane_schema": "control_plane",
                "lakebase_instance_name": "",
                "lakebase_database_name": "",
                "lakebase_schema": "model_lens_ui",
            },
            control_plane_catalog="model_observability",
            control_plane_schema="control_plane",
            lakebase_instance_name="",
            lakebase_database_name="",
            lakebase_schema="model_lens_ui",
        )
        is True
    )
