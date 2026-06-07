from __future__ import annotations

from model_landscape.services.mlflow_artifacts import (
    PANEL_TYPES,
    discover_lens_artifact_refs,
    parse_lens_drift,
    parse_lens_panel,
    parse_lens_summary,
    summarize_lens_run,
)


def test_discovers_lens_artifact_refs_from_tags() -> None:
    refs = discover_lens_artifact_refs(
        {
            "lens.version": "0.2.0",
            "lens.has_summary": "true",
            "lens.has_drift": "true",
            "lens.panel.feature_importance": "true",
            "lens.panel.roc_curve": "false",
        }
    )

    assert [ref.kind for ref in refs] == ["summary", "drift", "panel"]
    assert refs[0].path == "lens/summary.json"
    assert refs[1].path == "lens/drift.json"
    assert refs[2].path == "lens/panels/feature_importance.json"


def test_summarizes_lens_run_artifacts() -> None:
    summary = summarize_lens_run(
        "run-123",
        {
            "lens.version": "0.2.0",
            "lens.has_summary": "true",
            "lens.panel.confusion_matrix": "true",
        },
    )

    assert summary.run_id == "run-123"
    assert summary.lens_version == "0.2.0"
    assert summary.has_summary is True
    assert summary.has_drift is False
    assert summary.panels == ("confusion_matrix",)
    assert summary.artifact_types == ("summary", "confusion_matrix")


def test_parses_summary_payload_and_preserves_extra_fields() -> None:
    parsed = parse_lens_summary(
        {
            "lens_version": "0.2.0",
            "schema_version": "1",
            "task": "classification",
            "primary_metric": "auc",
            "score": "0.91",
            "dataset": "credit_v3",
            "notes": "baseline",
            "custom": {"fold": 1},
        }
    )

    assert parsed.lens_version == "0.2.0"
    assert parsed.score == 0.91
    assert parsed.extra == {"custom": {"fold": 1}}


def test_parses_panel_payload_and_accepts_unknown_panel_types() -> None:
    parsed = parse_lens_panel(
        {
            "lens_version": "0.2.0",
            "schema_version": "1",
            "type": "custom_panel",
            "data": [{"x": 1}],
            "top_n": 10,
        }
    )

    assert parsed.panel_type == "custom_panel"
    assert parsed.data == [{"x": 1}]
    assert parsed.metadata == {"top_n": 10}


def test_parses_drift_payload() -> None:
    parsed = parse_lens_drift(
        {
            "lens_version": "0.2.0",
            "schema_version": "2",
            "reference_run_id": "ref-1",
            "features": [{"feature": "amount", "psi": 0.12}],
            "prediction_shift": {"psi": 0.02},
            "custom": "kept",
        }
    )

    assert parsed.schema_version == "2"
    assert parsed.reference_run_id == "ref-1"
    assert parsed.features == ({"feature": "amount", "psi": 0.12},)
    assert parsed.prediction_shift == {"psi": 0.02}
    assert parsed.extra == {"custom": "kept"}


def test_app_contract_tracks_sdk_panel_registry() -> None:
    from mlflow_lens.panels import PANEL_TYPES as SDK_PANEL_TYPES

    assert set(PANEL_TYPES) == set(SDK_PANEL_TYPES)
