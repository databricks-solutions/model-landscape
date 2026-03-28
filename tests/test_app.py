from __future__ import annotations

from collections.abc import Iterator

from dash.development.base_component import Component

from model_lens import app as app_module
from model_lens.app import _feature_candidates, _non_numeric_features, _workspace_lakebase_instances, create_app


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
    ids = {component.id for component in _walk(app.layout) if getattr(component, "id", None)}
    assert {
        "setup-control-plane-btn",
        "refresh-all-btn",
        "refresh-selected-btn",
        "source-table-input",
        "scan-source-btn",
        "save-monitor-btn",
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
