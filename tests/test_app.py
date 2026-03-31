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
        "baseline-kind-input",
        "baseline-fixed-range-input",
        "create-catalog-toggle",
        "model-id-value-input",
        "model-version-value-input",
        "labels-order-col-input",
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

    result = fn(
        2,
        {"control_plane_catalog": "model_observability", "control_plane_schema": "control_plane"},
        "model_observability",
        "control_plane",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "entity_id",
        "label",
        None,
        None,
        None,
        None,
        "rolling",
        7,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    assert len(result) == 12
    assert result[3] == {}
    assert result[2] == {"display": "none"}


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
