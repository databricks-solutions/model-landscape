from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from model_lens.domain.models import MLflowDiscovery, MonitorConfig, MonitorDiscoveryResult
from model_lens.services.contracts import build_contract
from model_lens.services.control_plane import ControlPlaneRepository
from model_lens.services.mlflow_discovery import MLflowDiscoveryService
from model_lens.services.onboarding import build_default_baseline


NUMERIC_TYPE_TOKENS = ("tinyint", "smallint", "int", "bigint", "float", "double", "decimal", "numeric", "real")
STRING_TYPE_TOKENS = ("string", "varchar", "char", "text")
TIMESTAMP_TYPE_TOKENS = ("timestamp", "date")
RESERVED_FEATURE_TOKENS = ("id", "timestamp", "ts", "date", "time", "label", "target", "prediction", "score")


def _normalize(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _option_texts(values: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = _normalize(value)
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _schema_types(schema: pd.DataFrame) -> dict[str, str]:
    if schema.empty or "col_name" not in schema.columns:
        return {}
    return {
        str(row["col_name"]): str(row.get("data_type") or "")
        for _, row in schema.iterrows()
    }


def _is_type(data_type: str, tokens: tuple[str, ...]) -> bool:
    lowered = _normalize(data_type).lower()
    return any(token in lowered for token in tokens)


def _matches_pattern(column_name: str, patterns: tuple[str, ...]) -> int:
    lower = column_name.lower()
    for index, pattern in enumerate(patterns):
        if lower == pattern:
            return index
    for index, pattern in enumerate(patterns):
        if lower.startswith(pattern):
            return len(patterns) + index
    for index, pattern in enumerate(patterns):
        if pattern in lower:
            return (len(patterns) * 2) + index
    return 10_000


def _rank_columns(columns: list[str], patterns: tuple[str, ...], data_types: dict[str, str], preferred_types: tuple[str, ...] = ()) -> list[str]:
    ranked = sorted(
        columns,
        key=lambda column: (
            _matches_pattern(column, patterns),
            0 if preferred_types and _is_type(data_types.get(column, ""), preferred_types) else 1,
            column.lower(),
        ),
    )
    return [column for column in ranked if _matches_pattern(column, patterns) < 10_000]


def _table_label(table_name: str) -> tuple[str, str]:
    leaf = table_name.split(".")[-1] if table_name else "monitor"
    display_name = leaf.replace("_", " ").title()
    model_key = re.sub(r"[^a-zA-Z0-9_]", "_", leaf).lower()
    return display_name, model_key


def _preview_unique_count(preview: pd.DataFrame, column_name: str) -> int:
    if preview.empty or column_name not in preview.columns:
        return 0
    values = preview[column_name].dropna().astype(str).str.strip()
    values = values[values != ""]
    return int(values.nunique())


def _fallback_column(columns: list[str], data_types: dict[str, str], preferred_types: tuple[str, ...] = ()) -> str:
    if not columns:
        return ""
    if preferred_types:
        for column in columns:
            if _is_type(data_types.get(column, ""), preferred_types):
                return column
    return columns[0]


class MonitorDiscoveryService:
    def __init__(
        self,
        repository: ControlPlaneRepository,
        *,
        mlflow: MLflowDiscoveryService | None = None,
    ) -> None:
        self._repository = repository
        self._mlflow = mlflow or MLflowDiscoveryService()

    def discover(
        self,
        *,
        source_table: str,
        labels_table: str | None = None,
        mlflow_experiment_name: str | None = None,
        mlflow_registered_model_name: str | None = None,
        baseline_days: int = 7,
    ) -> MonitorDiscoveryResult:
        labels_table = _normalize(labels_table) or None
        columns, preview, schema = self._repository.scan_source_table(source_table.strip())
        schema_types = _schema_types(schema)
        warnings: list[str] = []
        requires_review = False

        mlflow = self._mlflow.discover(
            experiment_name_or_id=mlflow_experiment_name,
            registered_model_name=mlflow_registered_model_name,
        )
        warnings.extend(mlflow.warnings)

        timestamp_candidates = _rank_columns(
            columns,
            ("event_ts", "timestamp", "datetime", "date", "time", "_ts"),
            schema_types,
            preferred_types=TIMESTAMP_TYPE_TOKENS,
        )
        model_id_candidates = [
            column
            for column in _rank_columns(columns, ("model_id", "model", "model_name"), schema_types)
            if "version" not in column.lower()
        ]
        prediction_candidates = _rank_columns(
            columns,
            ("prediction", "score", "probability", "prob", "predicted"),
            schema_types,
            preferred_types=NUMERIC_TYPE_TOKENS,
        )
        version_candidates = _rank_columns(columns, ("model_version", "version"), schema_types)
        entity_id_candidates = _rank_columns(columns, ("entity_id", "request_id", "user_id", "account_id", "id"), schema_types)
        label_candidates = _rank_columns(columns, ("label", "target", "actual", "ground_truth"), schema_types)

        timestamp_col = timestamp_candidates[0] if timestamp_candidates else _fallback_column(columns, schema_types, TIMESTAMP_TYPE_TOKENS)
        model_id_col = model_id_candidates[0] if model_id_candidates else _fallback_column(columns, schema_types, STRING_TYPE_TOKENS)
        prediction_col = prediction_candidates[0] if prediction_candidates else _fallback_column(columns, schema_types, NUMERIC_TYPE_TOKENS)
        model_version_col = version_candidates[0] if version_candidates else None
        entity_id_col = entity_id_candidates[0] if entity_id_candidates else None
        source_label_col = label_candidates[0] if label_candidates else None

        required_candidates = {
            "timestamp": timestamp_candidates,
            "model ID": model_id_candidates,
            "prediction": prediction_candidates,
        }
        for label, candidates in required_candidates.items():
            if not candidates:
                warnings.append(f"Could not confidently detect a {label} column from the source schema.")
                requires_review = True
            elif len(candidates) > 1:
                warnings.append(f"Multiple plausible {label} columns were found; review the inferred mapping.")
                requires_review = True

        reserved = {
            timestamp_col,
            model_id_col,
            prediction_col,
            model_version_col,
            entity_id_col,
            source_label_col,
        }
        numeric_features = [
            column
            for column in columns
            if column not in reserved
            and _is_type(schema_types.get(column, ""), NUMERIC_TYPE_TOKENS)
            and not any(token == column.lower() or token in column.lower() for token in RESERVED_FEATURE_TOKENS)
        ]
        low_cardinality_slices = [
            column
            for column in columns
            if column not in reserved
            and _is_type(schema_types.get(column, ""), STRING_TYPE_TOKENS)
            and _preview_unique_count(preview, column) <= 12
            and not column.lower().endswith("_id")
        ]

        mlflow_features = [column for column in mlflow.feature_columns if column in numeric_features]
        selected_features = mlflow_features or numeric_features
        if mlflow.feature_columns and not mlflow_features:
            warnings.append("MLflow signature features did not overlap the numeric source columns; using table heuristics.")
            requires_review = True
        if not selected_features:
            fallback_features = [column for column in columns if column not in reserved]
            selected_features = fallback_features
            warnings.append("No numeric feature columns were detected; review the inferred feature set before activation.")
            requires_review = True

        labels_join_col = None
        labels_order_col = None
        label_col = source_label_col
        label_columns: tuple[str, ...] = ()
        if labels_table:
            label_columns, _, label_schema = self._repository.scan_source_table(labels_table, preview_rows=3)
            label_types = _schema_types(label_schema)
            label_columns = tuple(label_columns)
            label_candidates_external = _rank_columns(
                list(label_columns),
                ("label", "target", "actual", "ground_truth"),
                label_types,
            )
            join_candidates = []
            if entity_id_col and entity_id_col in label_columns:
                join_candidates.append(entity_id_col)
            join_candidates.extend(
                column
                for column in _rank_columns(
                    list(label_columns),
                    ("entity_id", "request_id", "user_id", "account_id", "id"),
                    label_types,
                )
                if column not in join_candidates
            )
            order_candidates = _rank_columns(
                list(label_columns),
                ("label_timestamp", "updated_at", "event_ts", "created_at", "timestamp"),
                label_types,
                preferred_types=TIMESTAMP_TYPE_TOKENS,
            )

            label_col = label_candidates_external[0] if label_candidates_external else source_label_col
            labels_join_col = join_candidates[0] if join_candidates else entity_id_col
            labels_order_col = order_candidates[0] if order_candidates else None

            if not label_col or not labels_join_col:
                warnings.append("External labels table needs a join column and label column; review advanced mappings.")
                requires_review = True

        model_id_value, model_scope_requires_review = self._infer_model_scope(
            source_table=source_table,
            model_id_col=model_id_col,
            mlflow=mlflow,
            warnings=warnings,
        )
        requires_review = requires_review or model_scope_requires_review
        if model_id_col and not model_id_value:
            sampled_model_ids = self._repository.sample_distinct_values(source_table, model_id_col, limit=5)
            if len(sampled_model_ids) > 1:
                warnings.append("Source table contains multiple model IDs; review the monitored model scope.")
                requires_review = True

        model_version_value = self._infer_model_version_scope(
            source_table=source_table,
            model_version_col=model_version_col,
            mlflow=mlflow,
        )

        if model_version_col and not model_version_value:
            sampled_versions = self._repository.sample_distinct_values(source_table, model_version_col, limit=5)
            if len(sampled_versions) > 1:
                warnings.append("Source table contains multiple model versions; review the monitored version scope.")
                requires_review = True

        problem_type = mlflow.problem_type or self._infer_problem_type(
            prediction_col=prediction_col,
            label_col=label_col,
            preview=preview,
        )
        display_name, model_key = _table_label(source_table)
        if model_id_value:
            display_name = model_id_value.replace("_", " ").title()
            model_key = re.sub(r"[^a-zA-Z0-9_]", "_", model_id_value).lower()

        contract = build_contract(
            columns=columns,
            timestamp_col=timestamp_col,
            model_id_col=model_id_col,
            prediction_col=prediction_col,
            feature_columns=selected_features,
            slice_columns=low_cardinality_slices,
            model_version_col=model_version_col,
            label_col=label_col,
            entity_id_col=entity_id_col,
        )
        config = MonitorConfig(
            model_key=model_key,
            display_name=display_name,
            source_table=source_table.strip(),
            contract=contract,
            baseline=build_default_baseline(int(baseline_days or 7)),
            problem_type=problem_type,
            model_id_value=model_id_value,
            model_version_value=model_version_value,
            labels_table=labels_table,
            labels_join_col=labels_join_col,
            labels_order_col=labels_order_col,
            mlflow=mlflow.lineage,
            created_by="app",
        )
        if not timestamp_candidates:
            warnings.append(f"Fell back to {timestamp_col!r} as the timestamp column.")
            requires_review = True
        if not model_id_candidates:
            warnings.append(f"Fell back to {model_id_col!r} as the model ID column.")
            requires_review = True
        if not prediction_candidates:
            warnings.append(f"Fell back to {prediction_col!r} as the prediction column.")
            requires_review = True
        confidence = "high"
        if requires_review or warnings:
            confidence = "medium"
        if not timestamp_col or not model_id_col or not prediction_col:
            confidence = "low"
            requires_review = True
        return MonitorDiscoveryResult(
            config=config,
            columns=tuple(columns),
            schema_rows=tuple(schema.fillna("").astype(str).to_dict("records")),
            preview_rows=tuple(preview.fillna("").astype(str).to_dict("records")),
            label_columns=label_columns,
            confidence=confidence,
            requires_review=requires_review,
            warnings=tuple(dict.fromkeys(_option_texts(warnings))),
        )

    def _infer_model_scope(
        self,
        *,
        source_table: str,
        model_id_col: str,
        mlflow: MLflowDiscovery,
        warnings: list[str],
    ) -> tuple[str | None, bool]:
        if not model_id_col:
            return None, False
        sampled = self._repository.sample_distinct_values(source_table, model_id_col, limit=5)
        if len(sampled) == 1:
            return sampled[0], False
        if len(sampled) <= 1:
            return None, False
        candidates = {
            _normalize(mlflow.lineage.registered_model_name).lower(),
            _normalize(mlflow.lineage.experiment_name).split("/")[-1].lower(),
        }
        normalized = {value.lower(): value for value in sampled}
        for candidate in candidates:
            if candidate and candidate in normalized:
                return normalized[candidate], False
        warnings.append("Could not infer a single monitored model_id value from source data and MLflow metadata.")
        return None, True

    def _infer_model_version_scope(
        self,
        *,
        source_table: str,
        model_version_col: str | None,
        mlflow: MLflowDiscovery,
    ) -> str | None:
        if not model_version_col:
            return None
        sampled = self._repository.sample_distinct_values(source_table, model_version_col, limit=10)
        if len(sampled) == 1:
            return sampled[0]
        version_hint = _normalize(mlflow.lineage.model_version)
        if version_hint and version_hint in sampled:
            return version_hint
        return None

    def _infer_problem_type(
        self,
        *,
        prediction_col: str,
        label_col: str | None,
        preview: pd.DataFrame,
    ) -> str:
        if preview.empty or prediction_col not in preview.columns:
            return "classification"
        prediction_values = pd.to_numeric(preview[prediction_col], errors="coerce").dropna()
        if prediction_col.lower().endswith(("_score", "_probability", "_proba")):
            return "classification"
        if not prediction_values.empty and prediction_values.between(0, 1).all():
            return "classification"
        if label_col and label_col in preview.columns:
            unique_labels = preview[label_col].dropna().astype(str).str.strip()
            if unique_labels.nunique() <= 10:
                return "classification"
        return "regression"
