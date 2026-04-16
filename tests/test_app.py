from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pandas as pd
from dash import dcc, no_update
from dash.development.base_component import Component

from model_lens import app as app_module
from model_lens import callbacks as callbacks_module
from model_lens.app import (
    _control_plane_ready,
    _feature_candidates,
    _non_numeric_features,
    _selected_model_from_search,
    _workspace_lakebase_instances,
    create_app,
)
from model_lens.callbacks import _format_runtime_setting_value, _ready_for_session, _setup_retry_message
from model_lens.pages import data_quality, drift_analysis, feature_deep_dive, incidents, overview, performance, reference
from model_lens.ui import charts


RENDER_WIZARD_CALLBACK = (
    "..wizard-steps-indicator.children...wizard-step-guidance.children...wizard-step-workspace.style"
    "...wizard-step-source.style...wizard-step-contract.style...wizard-step-review.style"
    "...wizard-back-btn.style...wizard-next-btn.style...wizard-next-btn.disabled...wizard-next-btn.children"
    "...save-monitor-btn.disabled...onboarding-review-summary.children.."
)
RENDER_DRIFT_CALLBACK = (
    "..drift-heatmap-container.children...drift-categorical-note.children...drift-timeline-container.children"
    "...drift-top-drifters-container.children.."
)
RENDER_PERFORMANCE_CALLBACK = (
    "..perf-labels-alert.children...perf-kpi-cards.children...perf-timeline-container.children"
    "...perf-contributors-container.children...perf-feature-select.options...perf-feature-select.value"
    "...perf-date-range-note.children.."
)
RENDER_QUALITY_CALLBACK = (
    "..quality-kpi-cards.children...quality-volume-container.children...quality-null-rates-container.children"
    "...quality-prediction-container.children.."
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


def _find_callback_by_input(app, input_id: str):
    for meta in app.callback_map.values():
        if any(item["id"] == input_id for item in meta.get("inputs", [])):
            callback = meta["callback"]
            return getattr(callback, "__wrapped__", callback)
    raise AssertionError(f"callback with input {input_id!r} not found")


def _find_callback_by_output(app, output_id: str):
    for key, meta in app.callback_map.items():
        if key.startswith(f"{output_id}.") or f"{output_id}." in key:
            callback = meta["callback"]
            return getattr(callback, "__wrapped__", callback)
    raise AssertionError(f"callback with output {output_id!r} not found")


def _find_callback_meta_by_output(app, output_id: str):
    for key, meta in app.callback_map.items():
        if key.startswith(f"{output_id}.") or f"{output_id}." in key:
            return meta
    raise AssertionError(f"callback metadata with output {output_id!r} not found")


def _find_callback_by_input_and_output(app, input_id: str, output_id: str):
    for key, meta in app.callback_map.items():
        if output_id not in key:
            continue
        if any(item["id"] == input_id for item in meta.get("inputs", [])):
            callback = meta["callback"]
            return getattr(callback, "__wrapped__", callback)
    raise AssertionError(f"callback with input {input_id!r} and output {output_id!r} not found")


def test_app_layout_exposes_slimmed_onboarding_flow() -> None:
    app = create_app()
    shell_components = [component for component in _walk(app.layout) if getattr(component, "id", None)]
    page_components = [component for component in _walk(app.validation_layout) if getattr(component, "id", None)]
    shell_ids = {component.id for component in shell_components}
    page_ids = {component.id for component in page_components}
    ids = shell_ids | page_ids
    assert {
        "url",
        "page-content",
        "global-model-select",
        "sidebar-primary-nav",
        "sidebar-onboarding-link",
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
        "workspace-readiness-store",
        "workspace-readiness-status",
        "validate-workspace-wiring-btn",
        "wizard-back-btn",
        "wizard-next-btn",
        "wizard-step-guidance",
        "onboarding-review-summary",
        "source-table-input",
        "labels-table-input",
        "mlflow-experiment-input",
        "mlflow-registered-model-input",
        "scan-source-btn",
        "save-monitor-btn",
        "review-drift-cadence-select",
        "review-performance-cadence-select",
        "review-performance-metrics-dropdown",
        "review-default-performance-metric-select",
        "review-schedule-enabled-toggle",
        "baseline-kind-input",
        "baseline-fixed-range-input",
        "create-catalog-toggle",
        "model-id-value-input",
        "model-version-value-input",
        "labels-order-col-input",
        "deepdive-context-container",
        "reference-page-status",
        "incidents-page-body",
        "incidents-monitor-filter",
        "incidents-severity-filter",
        "incidents-status-filter",
        "incidents-metric-filter",
    }.issubset(ids)

    components_by_id = {component.id: component for component in page_components}
    assert getattr(components_by_id["onboarding-current-step"], "storage_type", None) in (None, "memory")
    assert getattr(components_by_id["control-plane-ready-store"], "storage_type", None) in (None, "memory")
    assert getattr(components_by_id["wizard-step-workspace"], "style", {}) == {}
    assert getattr(components_by_id["wizard-step-source"], "style", {}) == {"display": "none"}
    assert getattr(components_by_id["wizard-step-contract"], "style", {}) == {"display": "none"}
    assert getattr(components_by_id["wizard-step-review"], "style", {}) == {"display": "none"}
    layout_text = str(app.validation_layout)
    assert "Permission checklist" in layout_text
    assert "CAN_USE on the SQL warehouse" in layout_text
    assert "CAN MANAGE on the refresh workflow" in layout_text
    assert "Validate Workspace Wiring" in layout_text
    assert "Save Monitor And Trigger Refresh" in layout_text


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


def test_monitor_contract_ready_accepts_shared_labels_join_without_entity_id_column() -> None:
    scan_data = {
        "columns": ["event_ts", "model_id", "prediction", "gc_transaction", "amount"],
    }

    assert callbacks_module._monitor_contract_ready(
        scan_data=scan_data,
        display_name="Fraud Model Demo",
        model_key="fraud_model_demo",
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        model_id_value=None,
        model_version_col=None,
        model_version_value=None,
        entity_id_col=None,
        source_label_col=None,
        external_label_col="label",
        labels_table="main.demo.labels",
        labels_join_col="gc_transaction",
        feature_columns=["amount"],
        baseline_kind="rolling",
        baseline_days=7,
        baseline_start=None,
        baseline_end=None,
    ) is True


def test_save_monitor_allows_table_scoped_monitor_without_model_id_column(monkeypatch) -> None:
    saved = {"validated": None, "upserted": None, "pending": None}

    class _Repository:
        def validate_monitor_source(self, config):
            saved["validated"] = config

        def upsert_monitor_config(self, config):
            saved["upserted"] = config

        def mark_monitor_bootstrap_pending(self, config):
            saved["pending"] = config

    class _FakeBackend:
        def __init__(self):
            self.repository = _Repository()

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    monkeypatch.setattr(
        callbacks_module,
        "trigger_refresh_job",
        lambda **kwargs: (_ for _ in ()).throw(Exception("run-now denied")),
    )
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "save-monitor-btn", "action-status")

    scan_data = {
        "table_name": "main.demo.inference",
        "columns": ["event_ts", "prediction", "label", "amount", "velocity_7d", "segment"],
        "schema": [
            {"col_name": "event_ts", "data_type": "timestamp"},
            {"col_name": "prediction", "data_type": "double"},
            {"col_name": "label", "data_type": "int"},
            {"col_name": "amount", "data_type": "double"},
            {"col_name": "velocity_7d", "data_type": "double"},
            {"col_name": "segment", "data_type": "string"},
        ],
        "discovery": {},
    }

    result = fn(
        1,
        scan_data,
        "Fraud Model Demo",
        "fraud_model_demo",
        "event_ts",
        None,
        "prediction",
        "",
        "",
        None,
        None,
        None,
        "label",
        "",
        "",
        "",
        "",
        ["amount", "velocity_7d", "segment"],
        ["segment"],
        ["segment"],
        "classification",
        "rolling",
        7,
        None,
        None,
        "6h",
        "daily_7d_repair",
        ["enabled"],
        ["f1", "precision", "recall"],
        "f1",
        "model_observability",
        "control_plane",
        "",
        "",
        "",
        {
            "control_plane_catalog": "model_observability",
            "control_plane_schema": "control_plane",
        },
        {
            "overall_mode": "scheduler_only",
            "blocking_issues": [],
            "warnings": [],
        },
    )

    assert saved["validated"] is not None
    assert saved["upserted"] is not None
    assert saved["pending"] is not None
    assert saved["upserted"].contract.model_id_col is None
    assert saved["upserted"].model_id_value is None
    assert saved["upserted"].performance_metric_names == ("f1", "precision", "recall")
    assert saved["upserted"].default_performance_metric == "f1"
    assert "Initial refresh is pending on the shared refresh job" in str(result[0])
    assert "The shared workflow can still pick it up on its next hourly run" in str(result[0])
    assert result[1]


def test_save_monitor_rejects_duplicate_model_key(monkeypatch) -> None:
    existing = SimpleNamespace(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        status="inactive",
    )

    class _Repository:
        def list_monitor_configs(self, status=None):
            assert status is None
            return [existing]

        def validate_monitor_source(self, config):
            raise AssertionError("duplicate-key save should not validate source")

        def upsert_monitor_config(self, config):
            raise AssertionError("duplicate-key save should not upsert")

        def mark_monitor_bootstrap_pending(self, config):
            raise AssertionError("duplicate-key save should not queue bootstrap")

    class _FakeBackend:
        def __init__(self):
            self.repository = _Repository()

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "save-monitor-btn", "action-status")

    scan_data = {
        "table_name": "main.demo.inference",
        "columns": ["event_ts", "prediction", "label", "amount"],
        "schema": [
            {"col_name": "event_ts", "data_type": "timestamp"},
            {"col_name": "prediction", "data_type": "double"},
            {"col_name": "label", "data_type": "int"},
            {"col_name": "amount", "data_type": "double"},
        ],
        "discovery": {},
    }

    result = fn(
        1,
        scan_data,
        "Fraud Model Demo",
        "fraud_model_demo",
        "event_ts",
        None,
        "prediction",
        "",
        "",
        None,
        None,
        None,
        "label",
        "",
        "",
        "",
        "",
        ["amount"],
        [],
        [],
        "classification",
        "rolling",
        7,
        None,
        None,
        "6h",
        "daily_7d_repair",
        ["enabled"],
        ["f1", "precision", "recall"],
        "f1",
        "model_observability",
        "control_plane",
        "",
        "",
        "",
        {
            "control_plane_catalog": "model_observability",
            "control_plane_schema": "control_plane",
        },
        {
            "overall_mode": "scheduler_only",
            "blocking_issues": [],
            "warnings": [],
        },
    )

    assert "A monitor with this model key already exists" in str(result[0])
    assert "Fraud Model Demo (fraud_model_demo, Archived)" in str(result[0])
    assert result[1] is no_update


def test_save_monitor_blocks_when_workspace_wiring_is_not_ready(monkeypatch) -> None:
    saved = {"validated": None}

    class _Repository:
        def validate_monitor_source(self, config):
            saved["validated"] = config

        def upsert_monitor_config(self, config):
            raise AssertionError("should not save when workspace readiness is blocked")

        def mark_monitor_bootstrap_pending(self, config):
            raise AssertionError("should not mark bootstrap pending when workspace readiness is blocked")

    class _FakeBackend:
        def __init__(self):
            self.repository = _Repository()

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "save-monitor-btn", "action-status")

    scan_data = {
        "table_name": "main.demo.inference",
        "columns": ["event_ts", "prediction", "label", "amount"],
        "schema": [
            {"col_name": "event_ts", "data_type": "timestamp"},
            {"col_name": "prediction", "data_type": "double"},
            {"col_name": "label", "data_type": "int"},
            {"col_name": "amount", "data_type": "double"},
        ],
        "discovery": {},
    }

    result = fn(
        1,
        scan_data,
        "Fraud Model Demo",
        "fraud_model_demo",
        "event_ts",
        None,
        "prediction",
        "",
        "",
        None,
        None,
        None,
        "label",
        "",
        "",
        "",
        "",
        ["amount"],
        [],
        [],
        "classification",
        "rolling",
        7,
        None,
        None,
        "6h",
        "daily_7d_repair",
        ["enabled"],
        ["f1", "precision", "recall"],
        "f1",
        "model_observability",
        "control_plane",
        "",
        "",
        "",
        {
            "control_plane_catalog": "model_observability",
            "control_plane_schema": "control_plane",
        },
        {
            "overall_mode": "not_ready",
            "blocking_issues": ["Shared refresh workflow is missing."],
            "warnings": [],
        },
    )

    assert saved["validated"] is None
    assert "Validate Workspace Wiring successfully before saving a monitor." in str(result[0])
    assert "Shared refresh workflow is missing." in str(result[0])


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
    assert (
        _control_plane_ready(
            {
                "control_plane_catalog": "model_observability",
                "control_plane_schema": "control_plane",
                "lakebase_instance_name": "",
                "lakebase_database_name": "",
                "lakebase_schema": "",
            },
            control_plane_catalog="model_observability",
            control_plane_schema="control_plane",
            lakebase_instance_name=None,
            lakebase_database_name=None,
            lakebase_schema="model_lens_ui",
        )
        is True
    )
    assert (
        _control_plane_ready(
            {
                "control_plane_catalog": "model_observability",
                "control_plane_schema": "control_plane",
            },
            control_plane_catalog="different_catalog",
            control_plane_schema="different_schema",
            lakebase_instance_name=None,
            lakebase_database_name=None,
            lakebase_schema=None,
        )
        is True
    )


def test_ready_for_session_requires_matching_setup_namespace() -> None:
    ready_state = {
        "control_plane_catalog": "model_observability",
        "control_plane_schema": "control_plane",
    }

    assert _ready_for_session(
        ready_state,
        {
            "control_plane_catalog": "model_observability",
            "control_plane_schema": "control_plane",
        },
    )
    assert not _ready_for_session(
        ready_state,
        {
            "control_plane_catalog": "different_catalog",
            "control_plane_schema": "control_plane",
        },
    )


def test_reference_runtime_settings_show_explicit_placeholders() -> None:
    assert _format_runtime_setting_value("use_lakebase_read_model", False) == "false"
    assert _format_runtime_setting_value("lakebase_database_name", "") == "(not configured)"
    assert _format_runtime_setting_value("genie_space_id", None) == "(not configured)"


def test_reference_page_copy_mentions_selected_model_scope() -> None:
    layout = reference.layout()

    assert "selected model" in str(layout.children[1].children).lower()


def test_setup_retry_message_tells_user_to_click_setup_again() -> None:
    message = _setup_retry_message("warehouse permission denied")

    assert "click Setup Control Plane again to retry" in message
    assert "warehouse permission denied" in message


def test_render_onboarding_wizard_callback_executes_for_step_two() -> None:
    app = create_app()
    callback = app.callback_map[RENDER_WIZARD_CALLBACK]["callback"]
    fn = getattr(callback, "__wrapped__", callback)

    args = [None] * 38
    args[0] = 2
    args[1] = {"control_plane_catalog": "model_observability", "control_plane_schema": "control_plane"}
    args[2] = {"overall_mode": "scheduler_only", "blocking_issues": [], "warnings": []}
    args[3] = "model_observability"
    args[4] = "control_plane"
    args[15] = "entity_id"
    args[16] = "label"
    args[24] = "rolling"
    args[25] = 7
    args[28] = "6h"
    args[29] = "disabled"
    args[30] = ["enabled"]
    args[31] = ["f1", "precision", "recall"]
    args[32] = "f1"

    result = fn(*args)

    assert len(result) == 12
    assert result[3] == {}
    assert result[2] == {"display": "none"}


def test_render_onboarding_wizard_blocks_when_workspace_wiring_is_not_ready() -> None:
    app = create_app()
    callback = app.callback_map[RENDER_WIZARD_CALLBACK]["callback"]
    fn = getattr(callback, "__wrapped__", callback)

    args = [None] * 38
    args[0] = 1
    args[1] = {"control_plane_catalog": "model_observability", "control_plane_schema": "control_plane"}
    args[2] = {"overall_mode": "not_ready", "blocking_issues": ["Shared refresh workflow is missing."], "warnings": []}
    args[3] = "model_observability"
    args[4] = "control_plane"
    args[24] = "rolling"
    args[25] = 7
    args[28] = "6h"
    args[29] = "disabled"
    args[30] = ["enabled"]
    args[31] = ["f1", "precision", "recall"]
    args[32] = "f1"

    result = fn(*args)

    assert result[8] is True
    assert "Shared refresh workflow is missing." in str(result[1])


def test_render_onboarding_wizard_surfaces_error_alert_instead_of_raising(monkeypatch) -> None:
    app = create_app()
    callback = app.callback_map[RENDER_WIZARD_CALLBACK]["callback"]
    fn = getattr(callback, "__wrapped__", callback)

    monkeypatch.setattr(
        callbacks_module,
        "_monitor_contract_ready",
        lambda **_: (_ for _ in ()).throw(RuntimeError("wizard exploded")),
    )

    args = [None] * 38
    args[0] = 3
    args[1] = {"control_plane_catalog": "model_observability", "control_plane_schema": "control_plane"}
    args[2] = {"overall_mode": "fully_ready", "blocking_issues": [], "warnings": []}
    args[3] = "model_observability"
    args[4] = "control_plane"

    result = fn(*args)

    assert len(result) == 12
    assert "Could not load onboarding wizard" in str(result[1])
    assert result[8] is True
    assert result[10] is True


def test_validate_workspace_wiring_callback_renders_readiness_card(monkeypatch) -> None:
    monkeypatch.setattr(
        callbacks_module,
        "_workspace_readiness_for_session",
        lambda ready_state, session: {
            "overall_mode": "scheduler_only",
            "control_plane_ready": True,
            "warehouse_ready": True,
            "refresh_workflow_resolved": True,
            "refresh_workflow_configured_via": "id",
            "refresh_workflow_configured_value": "123",
            "refresh_workflow_name": "model-lens-refresh",
            "refresh_workflow_id": 123,
            "scheduler_path_available": True,
            "scheduler_mode": "schedule",
            "run_now_available": None,
            "lakebase_ready": True,
            "blocking_issues": [],
            "warnings": [
                "Could not read refresh workflow permissions to verify immediate Run now access. Direct trigger may still work.",
            ],
        },
    )
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "validate-workspace-wiring-btn", "workspace-readiness-status")

    result = fn(
        1,
        "model_observability",
        "control_plane",
        "",
        "",
        "",
        {"control_plane_catalog": "model_observability", "control_plane_schema": "control_plane"},
    )

    assert "Workspace Readiness" in str(result[0])
    assert "Verification unavailable" in str(result[0])
    assert "Direct trigger may still work" in str(result[0])
    assert result[1]["overall_mode"] == "scheduler_only"


def test_validate_workspace_wiring_callback_distinguishes_unknown_bootstrap_acl(monkeypatch) -> None:
    monkeypatch.setattr(
        callbacks_module,
        "_workspace_readiness_for_session",
        lambda ready_state, session: {
            "overall_mode": "scheduler_only",
            "control_plane_ready": True,
            "warehouse_ready": True,
            "refresh_workflow_resolved": True,
            "refresh_workflow_configured_via": "id",
            "refresh_workflow_configured_value": "123",
            "refresh_workflow_name": "model-lens-refresh",
            "refresh_workflow_id": 123,
            "scheduler_path_available": True,
            "scheduler_mode": "schedule",
            "run_now_available": True,
            "bootstrap_workflow_mode": "separate",
            "bootstrap_workflow_resolved": True,
            "bootstrap_workflow_name": "model-lens-bootstrap",
            "bootstrap_workflow_id": 456,
            "bootstrap_run_now_available": None,
            "lakebase_ready": True,
            "blocking_issues": [],
            "warnings": [
                "Could not read refresh workflow permissions to verify immediate Run now access. Direct trigger may still work.",
            ],
        },
    )
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "validate-workspace-wiring-btn", "workspace-readiness-status")

    result = fn(
        1,
        "model_observability",
        "control_plane",
        "",
        "",
        "",
        {"control_plane_catalog": "model_observability", "control_plane_schema": "control_plane"},
    )

    assert "Bootstrap Trigger" in str(result[0])
    assert "Verification unavailable" in str(result[0])
    assert "Grant CAN_MANAGE_RUN on job 456" not in str(result[0])


def test_refresh_job_unavailable_message_includes_explicit_grant_for_configured_job_id(monkeypatch) -> None:
    monkeypatch.setattr(
        callbacks_module,
        "settings",
        SimpleNamespace(refresh_job_id="321"),
    )

    message = callbacks_module._refresh_job_unavailable_message("fraud_model_demo", RuntimeError("permission denied"))

    assert "Grant the app service principal CAN_MANAGE_RUN on job 321." in message


def test_render_overview_shows_empty_state_when_no_monitors_exist(monkeypatch) -> None:
    class _FakeBackend:
        def get_overview_rows(self, metric="psi"):
            assert metric == "psi"
            return []

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "overview-page-body")

    result = fn("/", None, {})

    assert "No monitors onboarded yet. Go to Onboarding to add your first model." in str(result)


def test_render_overview_surfaces_backend_errors_instead_of_raising(monkeypatch) -> None:
    class _FailingBackend:
        def get_overview_rows(self, metric="psi"):
            assert metric == "psi"
            raise RuntimeError("warehouse timeout")

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FailingBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "overview-page-body")

    result = fn("/", None, {})

    assert "Could not load overview: warehouse timeout" in str(result)
    assert "Overview is unavailable right now." in str(result)


def test_render_overview_surfaces_computing_pending_bucket(monkeypatch) -> None:
    class _FakeBackend:
        def get_overview_rows(self, metric="psi"):
            assert metric == "psi"
            return [
                {
                    "model_id": "fraud_model_demo",
                    "model_name": "Fraud Model Demo",
                    "description": "main.demo.fraud",
                    "versions": [],
                    "max_psi": 0.12,
                    "avg_psi": 0.09,
                    "avg_js": 0.03,
                    "drifting_features": 1,
                    "total_features": 2,
                    "top_drifter": "amount",
                    "max_null_rate": 0.5,
                    "has_labels": True,
                    "computing": False,
                    "freshness_status": "fresh",
                    "last_run_status": "completed",
                },
                {
                    "model_id": "spoof_model_demo",
                    "model_name": "Spoof Model Demo",
                    "description": "main.demo.spoof",
                    "versions": [],
                    "max_psi": 0.0,
                    "avg_psi": 0.0,
                    "avg_js": 0.0,
                    "drifting_features": 0,
                    "total_features": 3,
                    "top_drifter": "Computing/Pending",
                    "max_null_rate": 0.0,
                    "has_labels": False,
                    "computing": True,
                    "freshness_status": "pending_bootstrap",
                    "last_run_status": "",
                },
            ]

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "overview-page-body")

    result = fn("/", None, {})

    rendered = str(result)
    assert "Computing/Pending" in rendered
    assert "No drift history yet" in rendered


def test_populate_model_selector_returns_empty_when_no_monitors_exist(monkeypatch) -> None:
    class _FakeBackend:
        def list_models(self):
            return []

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "global-model-select")

    options, value = fn("/", "", None, {}, None)

    assert options == []
    assert value is None


def test_populate_model_selector_labels_include_model_key(monkeypatch) -> None:
    class _FakeBackend:
        def list_models(self):
            return [
                {"id": "fraud_model_demo", "name": "Fraud Model Demo"},
                {"id": "spoof_detection_ios_v1", "name": "Spoof Detection iOS V1"},
            ]

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "global-model-select")

    options, value = fn("/", "", None, {}, None)

    assert options[0]["label"] == "Fraud Model Demo (fraud_model_demo)"
    assert options[1]["label"] == "Spoof Detection iOS V1 (spoof_detection_ios_v1)"
    assert value == "fraud_model_demo"


def test_performance_metric_selector_uses_monitor_configured_metrics(monkeypatch) -> None:
    class _FakeBackend:
        def get_monitor_config(self, model_id):
            assert model_id == "fraud_model_demo"
            return SimpleNamespace(
                problem_type="classification",
                performance_metric_names=("f1", "precision", "recall"),
                default_performance_metric="precision",
            )

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "perf-metric-select")

    options, value = fn("fraud_model_demo", {}, None)

    assert [option["value"] for option in options] == ["f1", "precision", "recall"]
    assert value == "precision"


def test_populate_monitor_form_suggests_unique_model_key_when_default_exists(monkeypatch) -> None:
    class _FakeBackend:
        def list_reference_models(self, status=None):
            assert status is None
            return [{"id": "fraud_model_demo", "name": "Fraud Model Demo", "status": "active"}]

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "display-name-input")

    scan_data = {
        "table_name": "main.demo.fraud_model_demo",
        "columns": ["event_ts", "prediction", "amount"],
        "schema": [
            {"col_name": "event_ts", "data_type": "timestamp"},
            {"col_name": "prediction", "data_type": "double"},
            {"col_name": "amount", "data_type": "double"},
        ],
        "discovery": {},
    }

    result = fn(scan_data, {})

    assert result[0] == "Fraud Model Demo"
    assert result[1] == "fraud_model_demo_2"


def test_render_drift_callback_reports_when_only_one_window_exists(monkeypatch) -> None:
    class _FakeBackend:
        def get_monitor_config(self, model_id):
            return SimpleNamespace(
                contract=SimpleNamespace(categorical_columns=("segment",), label_col="label"),
                problem_type="classification",
            )

        def get_drift_results(self, model_id, granularity="daily", **kwargs):
            return pd.DataFrame(
                [
                    {
                        "feature": "amount",
                        "period": "2026-01-21",
                        "psi": 0.08,
                        "js_divergence": 0.03,
                        "kl_divergence": 0.02,
                    }
                ]
            )

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    callback = app.callback_map[RENDER_DRIFT_CALLBACK]["callback"]
    fn = getattr(callback, "__wrapped__", callback)

    result = fn("/drift", "fraud_model_demo", 0, {}, 0, "psi", "weekly", 10, None, None, "all", "all", False)

    assert "Only one weekly comparison window is available" in str(result[1])
    assert "Categorical features are stored" in str(result[1])


def test_render_drift_callback_respects_top_n_selection(monkeypatch) -> None:
    class _FakeBackend:
        def get_monitor_config(self, model_id):
            return SimpleNamespace(
                contract=SimpleNamespace(categorical_columns=(), label_col="label"),
                problem_type="classification",
            )

        def get_drift_results(self, model_id, granularity="daily", **kwargs):
            assert granularity == "daily"
            return pd.DataFrame(
                [
                    {"feature": "amount", "period": "2026-01-20", "psi": 10.97, "js_divergence": 0.61, "kl_divergence": 0.52},
                    {"feature": "velocity_7d", "period": "2026-01-20", "psi": 4.20, "js_divergence": 0.34, "kl_divergence": 0.28},
                    {"feature": "device_score", "period": "2026-01-20", "psi": 3.10, "js_divergence": 0.26, "kl_divergence": 0.22},
                    {"feature": "ip_risk", "period": "2026-01-20", "psi": 2.60, "js_divergence": 0.22, "kl_divergence": 0.19},
                    {"feature": "txn_count", "period": "2026-01-20", "psi": 1.70, "js_divergence": 0.15, "kl_divergence": 0.13},
                    {"feature": "geo_score", "period": "2026-01-20", "psi": 1.30, "js_divergence": 0.12, "kl_divergence": 0.1},
                    {"feature": "amount", "period": "2026-01-21", "psi": 0.02, "js_divergence": 0.01, "kl_divergence": 0.01},
                    {"feature": "velocity_7d", "period": "2026-01-21", "psi": 0.01, "js_divergence": 0.01, "kl_divergence": 0.01},
                    {"feature": "device_score", "period": "2026-01-21", "psi": 0.01, "js_divergence": 0.01, "kl_divergence": 0.01},
                    {"feature": "ip_risk", "period": "2026-01-21", "psi": 0.01, "js_divergence": 0.01, "kl_divergence": 0.01},
                    {"feature": "txn_count", "period": "2026-01-21", "psi": 0.0, "js_divergence": 0.0, "kl_divergence": 0.0},
                    {"feature": "geo_score", "period": "2026-01-21", "psi": 0.0, "js_divergence": 0.0, "kl_divergence": 0.0},
                ]
            )

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    callback = app.callback_map[RENDER_DRIFT_CALLBACK]["callback"]
    fn = getattr(callback, "__wrapped__", callback)

    result_top_3 = fn("/drift", "fraud_model_demo", 0, {}, 0, "psi", "daily", 3, None, None, "all", "all", False)
    result_top_5 = fn("/drift", "fraud_model_demo", 0, {}, 0, "psi", "daily", 5, None, None, "all", "all", False)
    result_top_6 = fn("/drift", "fraud_model_demo", 0, {}, 1, "psi", "daily", 6, None, None, "all", "all", False)
    result_thresholds = fn("/drift", "fraud_model_demo", 0, {}, 2, "psi", "daily", 5, None, None, "all", "all", True)

    heatmap_3 = result_top_3[0].children.children.figure
    top_3_figure = result_top_3[3].children.children.figure
    heatmap_5 = result_top_5[0].children.children.figure
    top_5_figure = result_top_5[3].children.children.figure
    top_6_figure = result_top_6[3].children.children.figure
    threshold_heatmap = result_thresholds[0].children.children.figure
    threshold_bar = result_thresholds[3].children.children.figure

    assert heatmap_3.layout.title.text == "Daily Feature Drift Heatmap"
    assert len(heatmap_3.data[0].y) == 3
    assert top_3_figure.layout.title.text == "Top 3 Drifting Features (Historical Max)"
    assert len(top_3_figure.data[0].y) == 3
    assert "highest historical PSI" in str(result_top_5[1])
    assert "95th percentile" in str(result_top_5[1])
    assert heatmap_5.layout.title.text == "Daily Feature Drift Heatmap"
    assert len(heatmap_5.data[0].y) == 5
    assert top_5_figure.layout.title.text == "Top 5 Drifting Features (Historical Max)"
    assert len(top_5_figure.data[0].y) == 5
    assert "device_score" in top_5_figure.data[0].y
    assert top_6_figure.layout.title.text == "Top 6 Drifting Features (Historical Max)"
    assert len(top_6_figure.data[0].y) == 6
    assert heatmap_5.data[0].colorscale != threshold_heatmap.data[0].colorscale
    assert top_5_figure.data[0].marker.color != threshold_bar.data[0].marker.color


def test_describe_drift_heatmap_scale_caps_large_outliers() -> None:
    drift = pd.DataFrame(
        [
            {"feature": f"feature_{index}", "period": period, "psi": value}
            for period, values in {
                "2026-01-20": [0.011, 0.013, 0.016, 0.018, 0.022],
                "2026-01-21": [0.012, 0.014, 0.017, 0.019, 5.25],
            }.items()
            for index, value in enumerate(values, start=1)
        ]
    )

    scale = charts.describe_drift_heatmap_scale(drift, metric="psi", show_thresholds=False)

    assert scale["clip_cap"] is not None
    assert float(scale["zmax"]) < 5.25
    assert int(scale["clip_count"]) == 1
    assert "95th percentile" in str(scale["clip_note"])


def test_describe_drift_heatmap_scale_falls_back_for_small_samples_and_respects_threshold_floor() -> None:
    drift = pd.DataFrame(
        [
            {"feature": "amount", "period": "2026-01-20", "psi": 0.01},
            {"feature": "velocity_7d", "period": "2026-01-20", "psi": 0.02},
            {"feature": "amount", "period": "2026-01-21", "psi": 0.03},
            {"feature": "velocity_7d", "period": "2026-01-21", "psi": 0.5},
        ]
    )

    neutral_scale = charts.describe_drift_heatmap_scale(drift, metric="psi", show_thresholds=False)
    threshold_scale = charts.describe_drift_heatmap_scale(drift, metric="psi", show_thresholds=True)

    assert float(neutral_scale["zmax"]) == 0.5
    assert neutral_scale["clip_note"] == ""
    assert float(threshold_scale["zmax"]) >= 0.2


def test_render_performance_callback_surfaces_zero_delta_state(monkeypatch) -> None:
    latest_bins = pd.DataFrame(
        [
            {
                "feature": "amount",
                "bin_label": "[0, 100)",
                "baseline_metric": 0.84,
                "current_metric": 0.84,
                "delta": 0.0,
                "current_volume_pct": 55.0,
                "degradation_contribution": 0.0,
                "window_start": "2026-01-14",
                "window_end": "2026-01-21",
            }
        ]
    )

    class _FakeBackend:
        def get_monitor_config(self, model_id):
            return SimpleNamespace(
                contract=SimpleNamespace(label_col="label"),
                problem_type="classification",
                performance_metric_names=("f1", "precision", "recall"),
            )

        def get_performance_summary(self, model_id, metric_name="f1"):
            value_map = {"f1": 0.84, "precision": 0.9, "recall": 0.78}
            return {
                "timeline": [{"period": "2026-01-21", metric_name: value_map.get(metric_name)}],
                "contributors": pd.DataFrame([{"feature": "amount", "weighted_delta": 0.0}]),
                "latest_bins": latest_bins,
                "all_bins": latest_bins,
                "has_significant_degradation": False,
                "worst_weighted_delta": 0.0,
                "timeline_unavailable_reason": "",
            }

        def get_drift_results(self, model_id, granularity="daily"):
            assert granularity == "daily"
            return pd.DataFrame(
                [
                    {"feature": "amount", "period": "2026-01-21", "psi": 0.12},
                ]
            )

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    callback = app.callback_map[RENDER_PERFORMANCE_CALLBACK]["callback"]
    fn = getattr(callback, "__wrapped__", callback)

    result = fn("/performance", "fraud_model_demo", "f1", "psi", 0, {}, None)

    assert "Feature impact metric: F1 Score" in str(result[0])
    assert "no significant degradation" in str(result[0]).lower()
    assert "Performance Metrics Over Time" in str(result[2])
    assert "PSI Over Time (Top Drifting Features)" in str(result[3])
    assert "Latest Bin Metrics" in str(result[3])
    assert "Only one comparison window is available" in str(result[6])


def test_render_performance_callback_handles_partial_window_note_and_missing_metric_column(monkeypatch) -> None:
    latest_bins = pd.DataFrame(
        [
            {
                "feature": "amount",
                "bin_label": "[0, 100)",
                "baseline_metric": 0.84,
                "current_metric": 0.84,
                "delta": 0.0,
                "current_volume_pct": 55.0,
                "degradation_contribution": 0.0,
                "window_end": "2026-01-21",
            }
        ]
    )

    class _FakeBackend:
        def get_monitor_config(self, model_id):
            return SimpleNamespace(
                contract=SimpleNamespace(label_col="label"),
                problem_type="classification",
                performance_metric_names=("f1", "precision", "recall"),
            )

        def get_performance_summary(self, model_id, metric_name="precision"):
            timeline_map = {
                "f1": [{"period": "2026-01-21", "f1": 0.84}],
                "precision": [{"period": "2026-01-21", "precision": None}],
                "recall": [{"period": "2026-01-21", "recall": 0.73}],
            }
            reason_map = {
                "precision": "PRECISION over time is unavailable until daily labeled facts are populated for this monitor.",
            }
            return {
                "timeline": timeline_map.get(metric_name, []),
                "contributors": pd.DataFrame([{"feature": "amount", "weighted_delta": 0.0}]),
                "latest_bins": latest_bins,
                "all_bins": latest_bins,
                "has_significant_degradation": False,
                "worst_weighted_delta": 0.0,
                "timeline_unavailable_reason": reason_map.get(metric_name, ""),
            }

        def get_drift_results(self, model_id, granularity="daily"):
            assert granularity == "daily"
            return pd.DataFrame(
                [
                    {"feature": "amount", "period": "2026-01-21", "psi": 0.12},
                ]
            )

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    callback = app.callback_map[RENDER_PERFORMANCE_CALLBACK]["callback"]
    fn = getattr(callback, "__wrapped__", callback)

    result = fn("/performance", "fraud_model_demo", "precision", "psi", 0, {}, None)

    assert "unavailable until daily labeled facts are populated" in str(result[0]).lower()
    assert "Latest comparison window end: 2026-01-21" in str(result[6])
    assert "Performance Metrics Over Time" in str(result[2])
    assert "Chart gaps mean the metric was undefined on those days, not zero." in str(result[6])


def test_render_performance_callback_uses_selected_drift_metric(monkeypatch) -> None:
    latest_bins = pd.DataFrame(
        [
            {
                "feature": "amount",
                "bin_label": "[0, 100)",
                "baseline_metric": 0.84,
                "current_metric": 0.82,
                "delta": -0.02,
                "current_volume_pct": 55.0,
                "degradation_contribution": -0.01,
                "window_start": "2026-01-14",
                "window_end": "2026-01-21",
            }
        ]
    )

    class _FakeBackend:
        def get_monitor_config(self, model_id):
            return SimpleNamespace(
                contract=SimpleNamespace(label_col="label"),
                problem_type="classification",
                performance_metric_names=("f1", "precision", "recall"),
            )

        def get_performance_summary(self, model_id, metric_name="f1"):
            value_map = {"f1": 0.84, "precision": 0.9, "recall": 0.78}
            return {
                "timeline": [{"period": "2026-01-21", metric_name: value_map.get(metric_name)}],
                "contributors": pd.DataFrame([{"feature": "amount", "weighted_delta": -0.01}]),
                "latest_bins": latest_bins,
                "all_bins": latest_bins,
                "has_significant_degradation": True,
                "worst_weighted_delta": -0.01,
                "timeline_unavailable_reason": "",
            }

        def get_drift_results(self, model_id, granularity="daily"):
            assert granularity == "daily"
            return pd.DataFrame(
                [
                    {"feature": "amount", "period": "2026-01-20", "psi": 0.12, "js_divergence": 0.03, "kl_divergence": 0.02},
                    {"feature": "amount", "period": "2026-01-21", "psi": 0.18, "js_divergence": 0.05, "kl_divergence": 0.04},
                ]
            )

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    callback = app.callback_map[RENDER_PERFORMANCE_CALLBACK]["callback"]
    fn = getattr(callback, "__wrapped__", callback)

    result = fn("/performance", "fraud_model_demo", "f1", "js_divergence", 0, {}, None)

    assert "Jensen-Shannon Divergence Over Time (Top Drifting Features)" in str(result[3])
    assert "Compare the Jensen-Shannon Divergence trend below" in str(result[3])


def test_render_quality_callback_surfaces_history_and_latest_snapshot(monkeypatch) -> None:
    class _FakeBackend:
        def get_monitor_config(self, model_id):
            return SimpleNamespace(contract=SimpleNamespace(label_col="label"), problem_type="classification")

        def get_quality_stats(self, model_id, **kwargs):
            return {
                "total_rows": 840,
                "min_date": "2026-01-01",
                "max_date": "2026-01-21",
                "prediction_mean": 0.44,
                "prediction_std": 0.13,
                "daily_volume": {"2026-01-21": 40},
                "null_rates": {"amount": 0.0, "velocity_7d": 1.2},
            }

        def get_quality_history(self, model_id, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "period": "2026-01-20",
                        "row_count": 110,
                        "prediction_mean": 0.42,
                        "prediction_std": 0.12,
                    },
                    {
                        "period": "2026-01-21",
                        "row_count": 120,
                        "prediction_mean": 0.44,
                        "prediction_std": 0.13,
                    },
                ]
            )

        def get_null_rate_history(self, model_id, **kwargs):
            return pd.DataFrame(
                [
                    {"period": "2026-01-20", "feature": "velocity_7d", "null_rate": 0.8},
                    {"period": "2026-01-21", "feature": "velocity_7d", "null_rate": 1.2},
                ]
            )

        def get_latest_window_metrics(self, model_id):
            return {
                "supported": True,
                "metrics": {
                    "precision": 0.75,
                    "recall": 0.6,
                    "f1": 0.6667,
                    "accuracy": 0.7,
                },
                "window_start": "2026-01-14",
                "window_end": "2026-01-21",
            }

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    callback = app.callback_map[RENDER_QUALITY_CALLBACK]["callback"]
    fn = getattr(callback, "__wrapped__", callback)

    result = fn("/quality", "fraud_model_demo", 0, {}, 0, None, None, "all", "all", False)
    result_with_guides = fn("/quality", "fraud_model_demo", 0, {}, 1, None, None, "all", "all", True)

    assert "Monitoring Rows" in str(result[0])
    assert "Rows Per Comparison Window" in str(result[1])
    assert "Daily Monitoring Rows shows daily row volume in persisted monitoring history" in str(result[1])
    assert "Null Rate Trends" in str(result[2])
    null_rate_figure = result[2].children[0].children.children.figure
    null_rate_guided = result_with_guides[2].children[0].children.children.figure
    snapshot_figure = result[3].children[1].children.children.figure
    assert snapshot_figure.layout.title.text == "Latest Window Performance Snapshot"
    assert len(null_rate_figure.layout.shapes or ()) == 0
    assert len(null_rate_guided.layout.shapes or ()) == 1


def test_performance_timeline_uses_distinct_metric_colors() -> None:
    fig = charts.build_performance_timeline(
        [
            {"period": "2026-01-20", "precision": 0.91, "recall": 0.72, "f1": 0.8},
            {"period": "2026-01-21", "precision": 0.88, "recall": 0.69, "f1": 0.77},
        ],
        metric_name="f1",
        metric_names=["precision", "recall", "f1"],
    )

    colors_by_name = {trace.name: trace.line.color for trace in fig.data}
    assert colors_by_name["Precision"] == charts.PERFORMANCE_METRIC_COLORS["precision"]
    assert colors_by_name["Recall"] == charts.PERFORMANCE_METRIC_COLORS["recall"]
    assert colors_by_name["Precision"] != colors_by_name["Recall"]


def test_render_quality_callback_uses_na_for_missing_prediction_mean_and_shows_std(monkeypatch) -> None:
    class _FakeBackend:
        def get_monitor_config(self, model_id):
            return SimpleNamespace(contract=SimpleNamespace(label_col="label"), problem_type="classification")

        def get_quality_stats(self, model_id, **kwargs):
            return {
                "total_rows": 840,
                "min_date": "2026-01-01",
                "max_date": "2026-01-21",
                "prediction_mean": None,
                "prediction_std": 0.13,
                "daily_volume": {"2026-01-21": 40},
                "null_rates": {"amount": 0.0, "velocity_7d": 1.2},
            }

        def get_quality_history(self, model_id, **kwargs):
            return pd.DataFrame([{"period": "2026-01-21", "row_count": 120, "prediction_mean": None, "prediction_std": 0.13}])

        def get_null_rate_history(self, model_id, **kwargs):
            return pd.DataFrame()

        def get_latest_window_metrics(self, model_id):
            return {"supported": False, "metrics": {}, "message": "No labeled snapshot is available for this monitor."}

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    callback = app.callback_map[RENDER_QUALITY_CALLBACK]["callback"]
    fn = getattr(callback, "__wrapped__", callback)

    result = fn("/quality", "fraud_model_demo", 0, {}, 0, None, None, "all", "all", False)

    rendered = str(result[0])
    assert "Prediction Average" in rendered
    assert "N/A" in rendered
    assert "Prediction Std" in rendered
    assert "No labeled snapshot is available for this monitor." in str(result[3])


def test_render_quality_callback_surfaces_backend_errors_instead_of_raising(monkeypatch) -> None:
    class _FailingBackend:
        def get_monitor_config(self, model_id):
            raise RuntimeError("sql endpoint unavailable")

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FailingBackend())
    app = create_app()
    callback = app.callback_map[RENDER_QUALITY_CALLBACK]["callback"]
    fn = getattr(callback, "__wrapped__", callback)

    result = fn("/quality", "fraud_model_demo", 0, {}, 0, None, None, "all", "all", False)

    assert "Could not load data quality: sql endpoint unavailable" in str(result[0])
    assert "Data quality is unavailable right now." in str(result[0])


def test_drift_and_quality_callbacks_use_apply_buttons_for_expensive_queries() -> None:
    app = create_app()

    drift_meta = app.callback_map[RENDER_DRIFT_CALLBACK]
    drift_input_ids = [item["id"] for item in drift_meta.get("inputs", [])]
    drift_state_ids = [item["id"] for item in drift_meta.get("state", [])]
    assert "drift-apply-filters-btn" in drift_input_ids
    assert "drift-metric-select" in drift_state_ids
    assert "drift-granularity-select" in drift_state_ids
    assert "drift-top-n" in drift_state_ids
    assert "drift-date-range" in drift_state_ids
    assert "drift-class-basis-select" in drift_state_ids
    assert "drift-class-value-select" in drift_state_ids
    assert "drift-threshold-toggle" in drift_state_ids

    quality_meta = app.callback_map[RENDER_QUALITY_CALLBACK]
    quality_input_ids = [item["id"] for item in quality_meta.get("inputs", [])]
    quality_state_ids = [item["id"] for item in quality_meta.get("state", [])]
    assert "quality-apply-filters-btn" in quality_input_ids
    assert "quality-date-range" in quality_state_ids
    assert "quality-class-basis-select" in quality_state_ids
    assert "quality-class-value-select" in quality_state_ids
    assert "quality-threshold-toggle" in quality_state_ids


def test_scan_source_table_failure_clears_prior_scan_data(monkeypatch) -> None:
    class _FakeBackend:
        def discover_monitor(self, **kwargs):
            raise RuntimeError("warehouse offline")

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "scan-source-btn", "scan-data")

    result = fn(1, "main.demo.inference", "", "", "", {})

    assert result[0] is None
    assert "Scan failed: warehouse offline" in str(result[1])


def test_populate_feature_deep_dive_prefers_most_drifted_feature(monkeypatch) -> None:
    class _FakeBackend:
        def get_feature_options(self, model_id):
            return ["amount", "device_score", "velocity_7d"]

        def get_dimension_options(self, model_id):
            return ["region"]

        def get_drift_results(self, model_id, granularity="daily"):
            return pd.DataFrame(
                [
                    {"feature": "amount", "period": "2026-01-21", "psi": 0.4},
                    {"feature": "device_score", "period": "2026-01-21", "psi": 2.1},
                    {"feature": "velocity_7d", "period": "2026-01-21", "psi": 0.9},
                ]
            )

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "deepdive-feature-select")

    options, value, dimension_options, selected_dimension = fn("/features", "fraud_model_demo", 0, {}, None, "")

    assert [option["value"] for option in options] == ["amount", "device_score", "velocity_7d"]
    assert value == "device_score"
    assert [option["value"] for option in dimension_options] == ["", "region"]
    assert selected_dimension == ""


def test_render_feature_deep_dive_reports_distribution_context(monkeypatch) -> None:
    class _FakeBackend:
        def get_feature_distribution_details(self, model_id, feature, require_exact_samples=False):
            return {
                "baseline": pd.Series([1.0, 2.0], dtype=float),
                "current": pd.Series([3.0, 4.0], dtype=float),
                "distribution_source": "persisted_histogram",
                "approximate": True,
                "window_label": "Baseline: 2026-01-01 to 2026-01-07 | Current: 2026-01-08 to 2026-01-14",
            }

        def get_dimension_breakdown(self, model_id, feature, dimension):
            return pd.DataFrame(
                [
                    {
                        "dimension_value": "(missing)",
                        "feature_average": 1.5,
                        "feature_p25": 1.2,
                        "feature_p50": 1.4,
                        "feature_p75": 1.7,
                    }
                ]
            )

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "deepdive-feature-select", "deepdive-distribution-container")

    distribution, dimension, context = fn("/features", "fraud_model_demo", "amount", "region", 0, {}, 0, "auto", 40, "", "off", 1.0)

    assert "Distribution: amount" in str(distribution)
    assert "amount by region" in str(dimension).lower()
    assert "approximate histogram reconstruction" in str(context)
    assert "Baseline: 2026-01-01 to 2026-01-07" in str(context)
    assert "Outlier Mode: Off" in str(context)


def test_render_feature_deep_dive_iqr_mode_requests_exact_samples(monkeypatch) -> None:
    class _FakeBackend:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def get_feature_distribution_details(self, model_id, feature, require_exact_samples=False):
            self.calls.append(bool(require_exact_samples))
            return {
                "baseline": pd.Series([1.0, 2.0, 50.0], dtype=float),
                "current": pd.Series([3.0, 4.0, 60.0], dtype=float),
                "distribution_source": "bounded_window_read",
                "approximate": False,
                "window_label": "Baseline: 2026-01-01 to 2026-01-07 | Current: 2026-01-08 to 2026-01-14",
            }

        def get_dimension_breakdown(self, model_id, feature, dimension):
            return pd.DataFrame()

    backend = _FakeBackend()
    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: backend)
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "deepdive-feature-select", "deepdive-distribution-container")

    distribution, _, context = fn("/features", "fraud_model_demo", "amount", "", 0, {}, 1, "fixed", 20, "", "iqr_fence", 1.5)

    assert backend.calls == [True]
    assert "Distribution: amount" in str(distribution)
    assert "Outlier Mode: IQR Fence (K=1.50)" in str(context)


def test_render_feature_deep_dive_handles_backend_errors(monkeypatch) -> None:
    class _FakeBackend:
        def get_feature_distribution_details(self, model_id, feature, require_exact_samples=False):
            raise RuntimeError("feature read failed")

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "deepdive-feature-select", "deepdive-distribution-container")

    distribution, dimension, context = fn("/features", "fraud_model_demo", "amount", "", 0, {}, 0, "auto", 40, "", "off", 1.0)

    assert "Could not load feature detail: feature read failed" in str(distribution)
    assert "Feature detail is unavailable right now." in str(context)


def test_render_feature_deep_dive_reports_unsafe_raw_fallback(monkeypatch) -> None:
    class _FakeBackend:
        def get_feature_distribution_details(self, model_id, feature, require_exact_samples=False):
            return {
                "baseline": pd.Series(dtype=float),
                "current": pd.Series(dtype=float),
                "distribution_source": "unavailable_unsafe_bounded_read",
                "approximate": False,
                "window_label": "Baseline: 2026-01-01 to 2026-01-07 | Current: 2026-01-08 to 2026-01-14",
            }

        def get_dimension_breakdown(self, model_id, feature, dimension):
            return pd.DataFrame()

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "deepdive-feature-select", "deepdive-distribution-container")

    distribution, _, context = fn("/features", "fraud_model_demo", "amount", "", 0, {}, 0, "auto", 40, "", "off", 1.0)

    assert "cannot enforce a hard cap on raw source-window reads" in str(distribution)
    assert "cannot enforce a hard cap on raw source-window reads" in str(context)


def test_render_feature_deep_dive_uses_prevent_initial_call() -> None:
    app = create_app()

    for meta in app._callback_list:
        if "deepdive-distribution-container.children" not in str(meta.get("output")):
            continue
        assert meta.get("prevent_initial_call") is True
        return

    raise AssertionError("feature deep dive render callback not found")


def test_render_feature_deep_dive_uses_apply_button_for_distribution_controls() -> None:
    app = create_app()

    for meta in app._callback_list:
        output = str(meta.get("output"))
        if "deepdive-distribution-container.children" not in output:
            continue
        input_ids = {item["id"] for item in meta.get("inputs", [])}
        state_ids = {item["id"] for item in meta.get("state", [])}
        assert "deepdive-apply-controls-btn" in input_ids
        assert "deepdive-binning-mode-select" not in input_ids
        assert "deepdive-bin-count-input" not in input_ids
        assert "deepdive-custom-edges-input" not in input_ids
        assert "deepdive-outlier-mode-select" not in input_ids
        assert "deepdive-outlier-value-input" not in input_ids
        assert "deepdive-binning-mode-select" in state_ids
        assert "deepdive-bin-count-input" in state_ids
        assert "deepdive-custom-edges-input" in state_ids
        assert "deepdive-outlier-mode-select" in state_ids
        assert "deepdive-outlier-value-input" in state_ids
        return

    raise AssertionError("feature deep dive render callback not found")


def test_analysis_pages_include_loading_wrappers() -> None:
    pages = [
        overview.layout(),
        drift_analysis.layout(),
        performance.layout(),
        data_quality.layout(),
        feature_deep_dive.layout(),
        reference.layout(),
        incidents.layout(),
    ]

    assert all(any(isinstance(component, dcc.Loading) for component in _walk(page)) for page in pages)


def test_labels_discovery_surfaces_zero_match_warning() -> None:
    component = callbacks_module._render_labels_discovery(
        labels_table="main.demo.labels",
        label_schema=pd.DataFrame([{"col_name": "unique_hash", "data_type": "string"}]),
        label_preview=pd.DataFrame([{"unique_hash": "hash-1", "label": "1"}]),
        label_validation={
            "inference_rows": 10,
            "matched_rows": 0,
            "unmatched_rows": 10,
            "duplicate_join_keys": 0,
            "match_rate_pct": 0.0,
            "distinct_label_values": ("0", "1"),
            "binary_compatible": True,
        },
        join_col="unique_hash",
        label_col="label",
        order_col="label_timestamp",
    )

    assert "No rows matched between inference and labels tables on this join column" in str(component)


def test_render_reference_callback_shows_archive_and_delete_actions(monkeypatch) -> None:
    config = SimpleNamespace(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.demo.inference",
        contract=SimpleNamespace(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            model_version_col=None,
            label_col="label",
            entity_id_col="entity_id",
            feature_columns=("amount",),
            categorical_columns=("segment",),
            slice_columns=("segment",),
        ),
        model_id_value="fraud_model_v1",
        model_version_value=None,
        labels_table=None,
        labels_join_col=None,
        labels_order_col=None,
        mlflow=SimpleNamespace(
            experiment_name=None,
            experiment_id=None,
            run_id=None,
            registered_model_name=None,
            model_version=None,
        ),
        baseline=SimpleNamespace(kind="rolling", n_days=7, baseline_start=None, baseline_end=None),
        problem_type="classification",
        drift_cadence_preset="6h",
        performance_cadence_preset="daily_7d_repair",
        schedule_enabled=True,
        status="active",
    )

    class _FakeBackend:
        def get_reference_data(self, model_id):
            assert model_id == "fraud_model_demo"
            return {
                "config": config,
                "summary": {},
                "runtime_state": {},
                "recent_runs": [],
                "recent_incident_history": [],
                "settings": {"refresh_job_id": "", "refresh_job_name": "model-lens-refresh"},
            }

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "reference-page-body")

    result = fn("/reference", "fraud_model_demo", None, 0, {})

    assert "Contract" in str(result)
    assert "Settings" in str(result)
    assert "Admin" in str(result)
    assert "Source: main.demo.inference" in str(result)
    assert "Model ID Value: fraud_model_v1" in str(result)
    assert "Archive Monitor" in str(result)
    assert "Restore Monitor" not in str(result)
    assert "Delete Monitor And History" in str(result)
    assert "Monitor Lifecycle" in str(result)


def test_render_reference_callback_shows_restore_for_archived_monitor(monkeypatch) -> None:
    config = SimpleNamespace(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.demo.inference",
        contract=SimpleNamespace(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            model_version_col=None,
            label_col="label",
            entity_id_col="entity_id",
            feature_columns=("amount",),
            categorical_columns=("segment",),
            slice_columns=("segment",),
        ),
        model_id_value="fraud_model_v1",
        model_version_value=None,
        labels_table=None,
        labels_join_col=None,
        labels_order_col=None,
        mlflow=SimpleNamespace(
            experiment_name=None,
            experiment_id=None,
            run_id=None,
            registered_model_name=None,
            model_version=None,
        ),
        baseline=SimpleNamespace(kind="rolling", n_days=7, baseline_start=None, baseline_end=None),
        problem_type="classification",
        drift_cadence_preset="6h",
        performance_cadence_preset="daily_7d_repair",
        schedule_enabled=False,
        status="inactive",
    )

    class _FakeBackend:
        def get_reference_data(self, model_id):
            assert model_id == "fraud_model_demo"
            return {
                "config": config,
                "summary": {},
                "runtime_state": {},
                "recent_runs": [],
                "recent_incident_history": [],
                "settings": {"refresh_job_id": "", "refresh_job_name": "model-lens-refresh"},
            }

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "reference-page-body")

    result = fn("/reference", "fraud_model_demo", None, 0, {})

    assert "Restore Monitor" in str(result)
    assert "Archive Monitor" not in str(result)
    assert "Delete Monitor And History" in str(result)


def test_render_reference_callback_shows_refresh_diagnostics(monkeypatch) -> None:
    config = SimpleNamespace(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.demo.inference",
        contract=SimpleNamespace(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            model_version_col=None,
            label_col="label",
            entity_id_col="entity_id",
            feature_columns=("amount",),
            categorical_columns=("segment",),
            slice_columns=("segment",),
        ),
        model_id_value="fraud_model_v1",
        model_version_value=None,
        labels_table=None,
        labels_join_col=None,
        labels_order_col=None,
        mlflow=SimpleNamespace(
            experiment_name=None,
            experiment_id=None,
            run_id=None,
            registered_model_name=None,
            model_version=None,
        ),
        baseline=SimpleNamespace(kind="rolling", n_days=7, baseline_start=None, baseline_end=None),
        problem_type="classification",
        drift_cadence_preset="6h",
        performance_cadence_preset="daily_7d_repair",
        schedule_enabled=True,
        status="active",
    )

    class _FakeBackend:
        def get_reference_data(self, model_id):
            assert model_id == "fraud_model_demo"
            return {
                "config": config,
                "summary": {},
                "runtime_state": {},
                "recent_runs": [],
                "refresh_diagnostics": {
                    "state": "ready",
                    "summary": {
                        "recent_run_count": 4,
                        "successful_run_count": 4,
                        "success_rate_pct": 100.0,
                        "median_duration_ms": 240000,
                        "dominant_bottleneck": "Daily Profiles Bound",
                        "trend": "Stable",
                        "recommendations": [
                            "Daily profile generation dominates. Consider more Spark workers for this workload."
                        ],
                    },
                    "recent_runs": [
                        {
                            "started_at": "2026-01-21T10:00:00",
                            "scope": "drift_quality",
                            "status": "completed",
                            "total_duration_ms": 240000,
                            "dominant_stage": "Daily Profiles",
                            "recommendation": "Daily profile generation dominates. Consider more Spark workers for this workload.",
                        }
                    ],
                },
                "recent_incident_history": [],
                "settings": {"refresh_job_id": "", "refresh_job_name": "model-lens-refresh"},
            }

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "reference-page-body")

    result = fn("/reference", "fraud_model_demo", None, 0, {})

    assert "Refresh Diagnostics" in str(result)
    assert "Thresholds" in str(result)
    assert "Shared Workflow Schedule" in str(result)
    assert "Compute Guidance" in str(result)
    assert "Recent Median Duration" in str(result)
    assert "Daily Profiles Bound" in str(result)
    assert "Recent Diagnosed Runs" in str(result)


def test_render_incidents_page_shows_open_and_recent_rows(monkeypatch) -> None:
    class _FakeBackend:
        def get_incidents_data(self, limit_history=100):
            assert limit_history == 100
            return {
                "models": [
                    {"id": "fraud_model_demo", "name": "Fraud Model Demo", "status": "active"},
                    {"id": "payments_model_demo", "name": "Payments Demo", "status": "inactive"},
                ],
                "open_incidents": pd.DataFrame(
                    [
                        {
                            "model_key": "fraud_model_demo",
                            "display_name": "Fraud Model Demo",
                            "feature_name": "amount",
                            "metric_name": "psi",
                            "severity": "critical",
                            "metric_value": 0.22,
                            "window_end": "2026-01-21",
                            "observed_at": "2026-01-21T10:00:00",
                            "status": "open",
                        }
                    ]
                ),
                "history": pd.DataFrame(
                    [
                        {
                            "model_key": "fraud_model_demo",
                            "display_name": "Fraud Model Demo",
                            "event_type": "opened",
                            "feature_name": "amount",
                            "metric_name": "psi",
                            "severity": "critical",
                            "status": "open",
                            "metric_value": 0.22,
                            "window_end": "2026-01-21",
                            "observed_at": "2026-01-21T10:00:00",
                        }
                    ]
                ),
            }

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "incidents-page-body")

    result = fn("/incidents", None, "all", "all", None, 0, {})

    assert "Open Incidents" in str(result)
    assert "Recent Incident History" in str(result)
    assert "Fraud Model Demo" in str(result)
    assert "Critical" in str(result)


def test_populate_incident_filters_uses_incident_data(monkeypatch) -> None:
    class _FakeBackend:
        def get_incidents_data(self, limit_history=100):
            return {
                "models": [{"id": "fraud_model_demo", "name": "Fraud Model Demo", "status": "active"}],
                "open_incidents": pd.DataFrame([{"metric_name": "psi"}]),
                "history": pd.DataFrame([{"metric_name": "js_divergence"}]),
            }

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "incidents-monitor-filter")

    model_options, model_value, metric_options, metric_value = fn("/incidents", 0, {}, None, None)

    assert model_value is None
    assert metric_value is None
    assert model_options == [{"label": "Fraud Model Demo (Active)", "value": "fraud_model_demo"}]
    assert metric_options == [
        {"label": "js_divergence", "value": "js_divergence"},
        {"label": "psi", "value": "psi"},
    ]


def test_render_reference_callback_shows_recent_incident_history(monkeypatch) -> None:
    config = SimpleNamespace(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.demo.inference",
        contract=SimpleNamespace(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            model_version_col=None,
            label_col="label",
            entity_id_col="entity_id",
            feature_columns=("amount",),
            categorical_columns=("segment",),
            slice_columns=("segment",),
        ),
        model_id_value="fraud_model_v1",
        model_version_value=None,
        labels_table=None,
        labels_join_col=None,
        labels_order_col=None,
        mlflow=SimpleNamespace(
            experiment_name=None,
            experiment_id=None,
            run_id=None,
            registered_model_name=None,
            model_version=None,
        ),
        baseline=SimpleNamespace(kind="rolling", n_days=7, baseline_start=None, baseline_end=None),
        problem_type="classification",
        drift_cadence_preset="6h",
        performance_cadence_preset="daily_7d_repair",
        schedule_enabled=True,
        status="active",
    )

    class _FakeBackend:
        def get_reference_data(self, model_id):
            assert model_id == "fraud_model_demo"
            return {
                "config": config,
                "summary": {},
                "runtime_state": {},
                "recent_runs": [],
                "recent_incident_history": [
                    {
                        "event_type": "opened",
                        "feature_name": "amount",
                        "metric_name": "psi",
                        "severity": "warning",
                        "status": "open",
                        "metric_value": 0.12,
                        "window_end": "2026-01-21",
                        "observed_at": "2026-01-21T10:00:00",
                    }
                ],
                "settings": {"refresh_job_id": "", "refresh_job_name": "model-lens-refresh"},
            }

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "reference-page-body")

    result = fn("/reference", "fraud_model_demo", None, 0, {})

    assert "Recent Incident History" in str(result)
    assert "opened" in str(result)
    assert "amount" in str(result)


def test_reference_bootstrap_retry_callback_triggers_shared_job(monkeypatch) -> None:
    config = SimpleNamespace(model_key="fraud_model_demo", display_name="Fraud Model Demo")

    class _FakeBackend:
        def get_monitor_config(self, model_id):
            assert model_id == "fraud_model_demo"
            return config

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    monkeypatch.setattr(
        callbacks_module,
        "trigger_refresh_job",
        lambda **kwargs: SimpleNamespace(job_id=123, run_id=456),
    )
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "reference-run-bootstrap-btn", "reference-page-status")

    result = fn(
        1,
        "fraud_model_demo",
        None,
        {
            "control_plane_catalog": "model_observability",
            "control_plane_schema": "control_plane",
            "lakebase_instance_name": "",
            "lakebase_database_name": "",
            "lakebase_schema": "",
        },
    )

    assert "Triggered the initial refresh for Fraud Model Demo" in str(result[0])
    assert result[1]


def test_reference_model_selector_prefers_sidebar_selection_over_stale_page_value(monkeypatch) -> None:
    class _FakeBackend:
        def list_reference_models(self, status="active"):
            assert status == "active"
            return [
                {"id": "fraud_model_demo", "name": "Fraud Model Demo", "status": "active"},
                {"id": "spoof_model_demo", "name": "Spoof Detection Ios V1", "status": "active"},
            ]

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_output(app, "reference-monitor-select")

    options, value = fn("/reference", "active", "spoof_model_demo", 0, {}, "fraud_model_demo")

    assert len(options) == 2
    assert options[0]["label"] == "Fraud Model Demo (fraud_model_demo, Active)"
    assert options[1]["label"] == "Spoof Detection Ios V1 (spoof_model_demo, Active)"
    assert value == "spoof_model_demo"


def test_clear_reference_status_on_reference_navigation_and_selection_change() -> None:
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "reference-monitor-status-filter", "reference-page-status")

    result = fn("/reference", "fraud_model_demo", "fraud_model_demo", "active")

    assert getattr(result, "children", None) is None


def test_archive_reference_monitor_callback_archives_selected_monitor(monkeypatch) -> None:
    repository = SimpleNamespace(archived=[], deleted=[])

    def archive_monitor(model_id):
        repository.archived.append(model_id)

    repository.archive_monitor = archive_monitor
    repository.delete_monitor = lambda model_id: repository.deleted.append(model_id)

    config = SimpleNamespace(display_name="Fraud Model Demo")

    class _FakeBackend:
        def __init__(self):
            self.repository = repository

        def get_monitor_config(self, model_id):
            assert model_id == "fraud_model_demo"
            return config

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "reference-archive-confirm-btn", "reference-page-status")

    result = fn(1, "fraud_model_demo", None, {})

    assert repository.archived == ["fraud_model_demo"]
    assert "Archived Fraud Model Demo" in str(result[0])
    assert result[1]


def test_restore_reference_monitor_callback_ignores_non_click_invocations(monkeypatch) -> None:
    repository = SimpleNamespace(restored=[])

    def restore_monitor(model_id):
        repository.restored.append(model_id)

    repository.restore_monitor = restore_monitor

    class _FakeBackend:
        def __init__(self):
            self.repository = repository

        def get_monitor_config(self, model_id):
            assert model_id == "fraud_model_demo"
            return SimpleNamespace(display_name="Fraud Model Demo", model_key="fraud_model_demo")

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    monkeypatch.setattr(callbacks_module, "ctx", SimpleNamespace(triggered_id="reference-restore-monitor-btn"))
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "reference-restore-monitor-btn", "reference-page-status")

    result = fn(None, "fraud_model_demo", None, {})

    assert repository.restored == []
    assert result == (no_update, no_update)


def test_restore_reference_monitor_callback_restores_selected_monitor(monkeypatch) -> None:
    repository = SimpleNamespace(restored=[])

    def restore_monitor(model_id):
        repository.restored.append(model_id)

    repository.restore_monitor = restore_monitor

    class _FakeBackend:
        def __init__(self):
            self.repository = repository

        def get_monitor_config(self, model_id):
            assert model_id == "fraud_model_demo"
            return SimpleNamespace(display_name="Fraud Model Demo", model_key="fraud_model_demo")

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    monkeypatch.setattr(callbacks_module, "ctx", SimpleNamespace(triggered_id="reference-restore-monitor-btn"))
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "reference-restore-monitor-btn", "reference-page-status")

    result = fn(1, "fraud_model_demo", None, {})

    assert repository.restored == ["fraud_model_demo"]
    assert "Restored Fraud Model Demo" in str(result[0])
    assert result[1]


def test_save_reference_schedule_persists_threshold_overrides(monkeypatch) -> None:
    saved = {}

    config = callbacks_module.MonitorConfig(
        model_key="fraud_model_demo",
        display_name="Fraud Model Demo",
        source_table="main.demo.inference",
        contract=callbacks_module.build_inference_contract(
            columns=["event_ts", "prediction", "label", "amount"],
            timestamp_col="event_ts",
            model_id_col=None,
            prediction_col="prediction",
            label_col="label",
            feature_columns=["amount"],
        ),
        baseline=callbacks_module.build_default_baseline(n_days=7),
        problem_type="classification",
        performance_metric_names=("f1",),
        default_performance_metric="f1",
    )

    class _FakeBackend:
        def __init__(self):
            self.repository = SimpleNamespace(
                upsert_monitor_config=lambda updated: saved.setdefault("config", updated),
                get_monitor_runtime_state=lambda _: None,
            )

        def get_monitor_config(self, model_id):
            assert model_id == "fraud_model_demo"
            return config

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "reference-save-schedule-btn", "reference-page-status")

    result = fn(
        1,
        "fraud_model_demo",
        None,
        "6h",
        "daily_7d_repair",
        ["enabled"],
        ["f1"],
        "f1",
        "quantile",
        None,
        0.15,
        0.35,
        0.08,
        0.2,
        0.12,
        0.4,
        2.0,
        8.0,
        {},
    )

    assert saved["config"].threshold_overrides == {
        "psi": {"warning": 0.15, "critical": 0.35},
        "js_divergence": {"warning": 0.08, "critical": 0.2},
        "kl_divergence": {"warning": 0.12, "critical": 0.4},
        "null_rate": {"warning": 2.0, "critical": 8.0},
    }
    assert "Updated monitor settings for Fraud Model Demo" in str(result[0])


def test_save_reference_shared_schedule_callback_updates_shared_job(monkeypatch) -> None:
    monkeypatch.setattr(
        callbacks_module,
        "update_shared_workflow_schedule",
        lambda interval_hours: SimpleNamespace(current_label="Every 12 Hours", job_id=321),
    )
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "reference-save-shared-schedule-btn", "reference-page-status")

    result = fn(1, 12)

    assert "Updated the shared refresh workflow to every 12 hours for job 321." in str(result[0])
    assert result[1]


def test_delete_reference_monitor_callback_deletes_selected_monitor(monkeypatch) -> None:
    repository = SimpleNamespace(archived=[], deleted=[])

    def delete_monitor(model_id):
        repository.deleted.append(model_id)

    repository.archive_monitor = lambda model_id: repository.archived.append(model_id)
    repository.delete_monitor = delete_monitor

    config = SimpleNamespace(display_name="Fraud Model Demo", model_key="fraud_model_demo")

    class _FakeBackend:
        def __init__(self):
            self.repository = repository

        def get_monitor_config(self, model_id):
            assert model_id == "fraud_model_demo"
            return config

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "reference-delete-confirm-btn", "reference-page-status")

    result = fn(1, "fraud_model_demo", None, "fraud_model_demo", {})

    assert repository.deleted == ["fraud_model_demo"]
    assert "Deleted Fraud Model Demo (fraud_model_demo)" in str(result[0])
    assert result[1]


def test_delete_reference_monitor_callback_blocks_when_model_key_is_ambiguous(monkeypatch) -> None:
    repository = SimpleNamespace(deleted=[])

    def delete_monitor(model_id):
        repository.deleted.append(model_id)

    repository.delete_monitor = delete_monitor
    repository.list_monitor_configs = lambda status=None: [
        SimpleNamespace(model_key="fraud_model_demo", display_name="Fraud Model Demo A", status="active"),
        SimpleNamespace(model_key="fraud_model_demo", display_name="Fraud Model Demo B", status="inactive"),
    ]

    class _FakeBackend:
        def __init__(self):
            self.repository = repository

        def get_monitor_config(self, model_id):
            return SimpleNamespace(display_name="Fraud Model Demo", model_key="fraud_model_demo")

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "reference-delete-confirm-btn", "reference-page-status")

    result = fn(1, "fraud_model_demo", None, "fraud_model_demo", {})

    assert repository.deleted == []
    assert "Multiple monitors currently share model key fraud_model_demo" in str(result[0])
    assert result[1] is no_update


def test_reference_delete_modal_toggle_clears_confirmation_input() -> None:
    app = create_app()
    fn = _find_callback_by_output(app, "reference-delete-modal")

    result = fn(1, None, None, False)

    assert result == (True, "")


def test_reference_delete_confirmation_enables_delete_only_on_exact_match() -> None:
    app = create_app()
    fn = _find_callback_by_output(app, "reference-delete-confirm-btn")

    disabled, valid, invalid, status = fn("fraud_model_demo", "fraud_model_demo")

    assert disabled is False
    assert valid is True
    assert invalid is False
    assert "Delete is enabled" in str(status)

    disabled, valid, invalid, status = fn("fraud_model_demo ", "fraud_model_demo")

    assert disabled is False
    assert valid is True
    assert invalid is False

    disabled, valid, invalid, status = fn("Fraud_Model_Demo", "fraud_model_demo")

    assert disabled is True
    assert valid is False
    assert invalid is True
    assert "does not match" in str(status)


def test_sidebar_status_wraps_long_monitor_description(monkeypatch) -> None:
    description = "cjc_aws_workspace_catalog.model_lens_demo.inference_logs | model_id=fraud_model_v1"

    class _FakeBackend:
        def get_model_map(self):
            return {
                "fraud_model_v1": {
                    "description": description,
                    "open_incident_count": 0,
                    "feature_count": 12,
                    "baseline_label": "Rolling 7 days",
                    "total_rows": 840,
                }
            }

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    fn = _find_callback_by_input_and_output(app, "global-model-select", "sidebar-status")

    status, _, _ = fn("fraud_model_v1", None, {})

    description_component = status.children[0]
    assert "model-lens-sidebar-description" in description_component.className
    assert description_component.title == description
