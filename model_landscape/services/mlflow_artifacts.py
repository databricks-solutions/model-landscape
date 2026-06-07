from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from model_landscape.domain.models import MLflowLensArtifacts


LENS_VERSION_TAG = "lens.version"
LENS_HAS_SUMMARY_TAG = "lens.has_summary"
LENS_HAS_DRIFT_TAG = "lens.has_drift"
LENS_PANEL_TAG_PREFIX = "lens.panel."

LENS_SUMMARY_ARTIFACT_PATH = "lens/summary.json"
LENS_DRIFT_ARTIFACT_PATH = "lens/drift.json"
LENS_DRIFT_SNAPSHOT_ARTIFACT_PATH = "lens/drift_snapshot.json"
LENS_PANELS_ARTIFACT_DIR = "lens/panels"

PANEL_TYPES = (
    "confusion_matrix",
    "roc_curve",
    "feature_importance",
    "learning_curve",
    "classification_report",
    "precision_recall_curve",
    "class_prediction_error",
    "discrimination_threshold",
    "prediction_error",
    "residuals",
    "alpha_selection",
    "validation_curve",
    "cv_scores",
)


@dataclass(frozen=True)
class LensArtifactRef:
    kind: str
    path: str
    panel_type: str | None = None
    tag_name: str | None = None


@dataclass(frozen=True)
class LensSummaryArtifact:
    schema_version: str
    lens_version: str | None = None
    task: str | None = None
    primary_metric: str | None = None
    score: float | None = None
    dataset: str | None = None
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LensPanelArtifact:
    schema_version: str
    panel_type: str
    lens_version: str | None = None
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LensDriftArtifact:
    schema_version: str
    lens_version: str | None = None
    reference_run_id: str | None = None
    features: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    prediction_shift: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _normalize(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truthy_tag(value: object) -> bool:
    return _normalize(value).lower() in {"1", "true", "yes", "y"}


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def panel_artifact_path(panel_type: str) -> str:
    return f"{LENS_PANELS_ARTIFACT_DIR}/{panel_type}.json"


def discover_lens_artifact_refs(tags: Mapping[str, Any]) -> tuple[LensArtifactRef, ...]:
    refs: list[LensArtifactRef] = []
    if _truthy_tag(tags.get(LENS_HAS_SUMMARY_TAG)):
        refs.append(LensArtifactRef(kind="summary", path=LENS_SUMMARY_ARTIFACT_PATH, tag_name=LENS_HAS_SUMMARY_TAG))
    if _truthy_tag(tags.get(LENS_HAS_DRIFT_TAG)):
        refs.append(LensArtifactRef(kind="drift", path=LENS_DRIFT_ARTIFACT_PATH, tag_name=LENS_HAS_DRIFT_TAG))

    panel_tags = sorted(
        (str(key), value)
        for key, value in tags.items()
        if str(key).startswith(LENS_PANEL_TAG_PREFIX)
    )
    for tag_name, tag_value in panel_tags:
        if not _truthy_tag(tag_value):
            continue
        panel_type = tag_name[len(LENS_PANEL_TAG_PREFIX):].strip()
        if panel_type:
            refs.append(
                LensArtifactRef(
                    kind="panel",
                    path=panel_artifact_path(panel_type),
                    panel_type=panel_type,
                    tag_name=tag_name,
                )
            )
    return tuple(refs)


def summarize_lens_run(run_id: str | None, tags: Mapping[str, Any]) -> MLflowLensArtifacts:
    refs = discover_lens_artifact_refs(tags)
    panels = tuple(ref.panel_type for ref in refs if ref.kind == "panel" and ref.panel_type)
    return MLflowLensArtifacts(
        run_id=_normalize(run_id) or None,
        lens_version=_normalize(tags.get(LENS_VERSION_TAG)) or None,
        has_summary=any(ref.kind == "summary" for ref in refs),
        has_drift=any(ref.kind == "drift" for ref in refs),
        panels=panels,
    )


def parse_lens_summary(payload: Mapping[str, Any]) -> LensSummaryArtifact:
    known = {"lens_version", "schema_version", "task", "primary_metric", "score", "dataset", "notes"}
    return LensSummaryArtifact(
        schema_version=_normalize(payload.get("schema_version")) or "unknown",
        lens_version=_normalize(payload.get("lens_version")) or None,
        task=_normalize(payload.get("task")) or None,
        primary_metric=_normalize(payload.get("primary_metric")) or None,
        score=_float_or_none(payload.get("score")),
        dataset=_normalize(payload.get("dataset")) or None,
        notes=_normalize(payload.get("notes")) or None,
        extra={str(key): value for key, value in payload.items() if key not in known},
    )


def parse_lens_panel(payload: Mapping[str, Any]) -> LensPanelArtifact:
    known = {"lens_version", "schema_version", "type", "data"}
    return LensPanelArtifact(
        schema_version=_normalize(payload.get("schema_version")) or "unknown",
        lens_version=_normalize(payload.get("lens_version")) or None,
        panel_type=_normalize(payload.get("type")) or "unknown",
        data=payload.get("data"),
        metadata={str(key): value for key, value in payload.items() if key not in known},
    )


def parse_lens_drift(payload: Mapping[str, Any]) -> LensDriftArtifact:
    known = {"lens_version", "schema_version", "reference_run_id", "features", "prediction_shift"}
    raw_features = payload.get("features")
    features = tuple(dict(item) for item in raw_features if isinstance(item, Mapping)) if isinstance(raw_features, list) else ()
    raw_prediction_shift = payload.get("prediction_shift")
    prediction_shift = dict(raw_prediction_shift) if isinstance(raw_prediction_shift, Mapping) else None
    return LensDriftArtifact(
        schema_version=_normalize(payload.get("schema_version")) or "unknown",
        lens_version=_normalize(payload.get("lens_version")) or None,
        reference_run_id=_normalize(payload.get("reference_run_id")) or None,
        features=features,
        prediction_shift=prediction_shift,
        extra={str(key): value for key, value in payload.items() if key not in known},
    )
