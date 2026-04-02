from __future__ import annotations

from collections.abc import Iterable

from model_lens.domain.models import InferenceContract
from model_lens.services.inference_contracts import build_inference_contract


def build_contract(
    *,
    columns: Iterable[str],
    timestamp_col: str,
    model_id_col: str | None = None,
    prediction_col: str,
    feature_columns: Iterable[str],
    categorical_columns: Iterable[str] = (),
    slice_columns: Iterable[str] = (),
    model_version_col: str | None = None,
    prediction_score_col: str | None = None,
    label_col: str | None = None,
    entity_id_col: str | None = None,
) -> InferenceContract:
    return build_inference_contract(
        columns=columns,
        timestamp_col=timestamp_col,
        model_id_col=model_id_col,
        prediction_col=prediction_col,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        slice_columns=slice_columns,
        model_version_col=model_version_col,
        prediction_score_col=prediction_score_col,
        label_col=label_col,
        entity_id_col=entity_id_col,
    )
