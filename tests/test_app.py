from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pandas as pd
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
from model_lens.pages import reference


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
        if key.startswith(f"{output_id}.") or f"...{output_id}." in key:
            callback = meta["callback"]
            return getattr(callback, "__wrapped__", callback)
    raise AssertionError(f"callback with output {output_id!r} not found")


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
        "reference-page-status",
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
    assert "CAN MANAGE RUN on the refresh workflow" in layout_text
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
                "Could not confirm immediate Run now permission; scheduler-only mode assumed.",
                "Grant the app service principal CAN_MANAGE_RUN on job 123.",
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
    assert "scheduler-only mode" in str(result[0]).lower()
    assert "Grant CAN_MANAGE_RUN on job 123" in str(result[0])
    assert result[1]["overall_mode"] == "scheduler_only"


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

    assert "No monitors onboarded yet." in str(result)


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


def test_render_drift_callback_reports_when_only_one_window_exists(monkeypatch) -> None:
    class _FakeBackend:
        def get_monitor_config(self, model_id):
            return SimpleNamespace(contract=SimpleNamespace(categorical_columns=("segment",)))

        def get_drift_results(self, model_id, granularity="daily"):
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

    result = fn("/drift", "fraud_model_demo", "psi", "weekly", 10, 0, {})

    assert "Only one weekly comparison window is available" in str(result[1])
    assert "Categorical features are stored" in str(result[1])


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
            return SimpleNamespace(contract=SimpleNamespace(label_col="label"))

        def get_performance_summary(self, model_id, metric_name="f1"):
            return {
                "timeline": [{"period": "2026-01-21", "f1": 0.84}],
                "contributors": pd.DataFrame([{"feature": "amount", "weighted_delta": 0.0}]),
                "latest_bins": latest_bins,
                "all_bins": latest_bins,
                "has_significant_degradation": False,
                "worst_weighted_delta": 0.0,
            }

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    callback = app.callback_map[RENDER_PERFORMANCE_CALLBACK]["callback"]
    fn = getattr(callback, "__wrapped__", callback)

    result = fn("/performance", "fraud_model_demo", "f1", 0, {}, None)

    assert "no significant degradation" in str(result[0]).lower()
    assert "Latest Bin Metrics" in str(result[3])
    assert "Only one comparison window is available" in str(result[6])


def test_render_quality_callback_surfaces_history_and_latest_snapshot(monkeypatch) -> None:
    class _FakeBackend:
        def get_quality_stats(self, model_id):
            return {
                "total_rows": 840,
                "min_date": "2026-01-01",
                "max_date": "2026-01-21",
                "prediction_mean": 0.44,
                "prediction_std": 0.13,
                "daily_volume": {"2026-01-21": 40},
                "null_rates": {"amount": 0.0, "velocity_7d": 1.2},
            }

        def get_quality_history(self, model_id):
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

        def get_null_rate_history(self, model_id):
            return pd.DataFrame(
                [
                    {"period": "2026-01-20", "feature": "velocity_7d", "null_rate": 0.8},
                    {"period": "2026-01-21", "feature": "velocity_7d", "null_rate": 1.2},
                ]
            )

        def get_prediction_distribution(self, model_id):
            return pd.Series([0.2, 0.4, 0.8], dtype=float)

    monkeypatch.setattr(callbacks_module, "_make_backend", lambda session_data: _FakeBackend())
    app = create_app()
    callback = app.callback_map[RENDER_QUALITY_CALLBACK]["callback"]
    fn = getattr(callback, "__wrapped__", callback)

    result = fn("/quality", "fraud_model_demo", 0, {})

    assert "Rows Per Comparison Window" in str(result[1])
    assert "Null Rate Trends" in str(result[2])
    assert "Prediction Mean Over Time" in str(result[3])


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

    assert "Archive Monitor" in str(result)
    assert "Delete Monitor And History" in str(result)
    assert "Monitor Lifecycle" in str(result)
    assert "Run Initial Refresh Now" in str(result)
    assert "REFRESH_JOB_ID is preferred" in str(result)


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
    assert "Deleted Fraud Model Demo" in str(result[0])
    assert result[1]
