from __future__ import annotations

from collections.abc import Iterable

from model_lens.domain.models import InferenceContract


def build_contract(
    *,
    columns: Iterable[str],
    timestamp_col: str,
    model_id_col: str,
    prediction_col: str,
    feature_columns: Iterable[str],
    categorical_columns: Iterable[str] = (),
    slice_columns: Iterable[str] = (),
    model_version_col: str | None = None,
    prediction_score_col: str | None = None,
    label_col: str | None = None,
    entity_id_col: str | None = None,
) -> InferenceContract:
    known_columns = set(columns)
    required = {
        "timestamp_col": timestamp_col,
        "model_id_col": model_id_col,
        "prediction_col": prediction_col,
    }
    for name, value in required.items():
        if value not in known_columns:
            raise ValueError(f"{name}={value!r} is not present in the mapped table")

    features = tuple(dict.fromkeys(feature_columns))
    categorical = tuple(dict.fromkeys(categorical_columns))
    slices = tuple(dict.fromkeys(slice_columns))

    reserved = {
        timestamp_col,
        model_id_col,
        prediction_col,
        model_version_col,
        prediction_score_col,
        label_col,
        entity_id_col,
    }
    overlap = reserved.intersection(features)
    if overlap:
        raise ValueError(f"Feature columns cannot include reserved fields: {sorted(overlap)}")

    unknown_features = [column for column in features if column not in known_columns]
    if unknown_features:
        raise ValueError(f"Unknown feature columns: {unknown_features}")

    unknown_slices = [column for column in slices if column not in known_columns]
    if unknown_slices:
        raise ValueError(f"Unknown slice columns: {unknown_slices}")

    bad_categorical = [column for column in categorical if column not in features]
    if bad_categorical:
        raise ValueError("Categorical columns must be a subset of feature columns")

    return InferenceContract(
        timestamp_col=timestamp_col,
        model_id_col=model_id_col,
        prediction_col=prediction_col,
        model_version_col=model_version_col,
        prediction_score_col=prediction_score_col,
        label_col=label_col,
        entity_id_col=entity_id_col,
        feature_columns=features,
        slice_columns=slices,
        categorical_columns=categorical,
    )
