from __future__ import annotations

import json
import logging
from typing import Any

from model_landscape.domain.models import MLflowDiscovery, MLflowLineage

logger = logging.getLogger(__name__)


def _normalize(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _field_names_from_signature(inputs: Any) -> tuple[str, ...]:
    if inputs is None:
        return ()
    items = getattr(inputs, "inputs", inputs)
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except json.JSONDecodeError:
            return ()
    if not isinstance(items, (list, tuple)):
        return ()
    names: list[str] = []
    for item in items:
        name = getattr(item, "name", None)
        if name is None and isinstance(item, dict):
            name = item.get("name")
        if name is None and hasattr(item, "to_dict"):
            try:
                name = item.to_dict().get("name")
            except Exception:
                name = None
        text = _normalize(name)
        if text and text not in names:
            names.append(text)
    return tuple(names)


class MLflowDiscoveryService:
    def __init__(self) -> None:
        self._initialized = False
        self._available = False
        self._init_error = ""
        self._mlflow = None
        self._client = None

    @property
    def available(self) -> bool:
        self._ensure_client()
        return self._available

    def discover(
        self,
        *,
        experiment_name_or_id: str | None = None,
        registered_model_name: str | None = None,
    ) -> MLflowDiscovery:
        experiment_name_or_id = _normalize(experiment_name_or_id)
        registered_model_name = _normalize(registered_model_name)
        if not experiment_name_or_id and not registered_model_name:
            return MLflowDiscovery()

        self._ensure_client()
        if not self._available:
            warning = f"MLflow is unavailable: {self._init_error or 'client initialization failed'}"
            return MLflowDiscovery(warnings=(warning,))

        registered_model = (
            self._discover_registered_model(registered_model_name)
            if registered_model_name
            else MLflowDiscovery()
        )
        experiment = (
            self._discover_experiment(experiment_name_or_id)
            if experiment_name_or_id
            else MLflowDiscovery()
        )
        return self._merge_discoveries(registered_model, experiment)

    def _ensure_client(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        try:
            import mlflow

            mlflow.set_registry_uri("databricks-uc")
            self._mlflow = mlflow
            self._client = mlflow.MlflowClient()
            self._available = True
        except Exception as error:
            self._available = False
            self._init_error = str(error)
            logger.info("MLflow discovery unavailable: %s", error)

    def _merge_discoveries(
        self, primary: MLflowDiscovery, secondary: MLflowDiscovery
    ) -> MLflowDiscovery:
        warnings = tuple(dict.fromkeys(primary.warnings + secondary.warnings))
        lineage = MLflowLineage(
            experiment_name=primary.lineage.experiment_name or secondary.lineage.experiment_name,
            experiment_id=primary.lineage.experiment_id or secondary.lineage.experiment_id,
            run_id=primary.lineage.run_id or secondary.lineage.run_id,
            registered_model_name=primary.lineage.registered_model_name
            or secondary.lineage.registered_model_name,
            model_version=primary.lineage.model_version or secondary.lineage.model_version,
        )
        features = primary.feature_columns or secondary.feature_columns
        problem_type = primary.problem_type or secondary.problem_type
        if (
            primary.lineage.registered_model_name
            and secondary.lineage.registered_model_name
            and primary.lineage.registered_model_name != secondary.lineage.registered_model_name
        ):
            warnings += (
                "Registered model and experiment metadata point at different model names.",
            )
        return MLflowDiscovery(
            lineage=lineage,
            feature_columns=features,
            problem_type=problem_type,
            warnings=warnings,
        )

    def _discover_experiment(self, experiment_name_or_id: str) -> MLflowDiscovery:
        experiment = None
        warnings: list[str] = []
        try:
            experiment = self._client.get_experiment_by_name(experiment_name_or_id)
        except Exception:
            experiment = None
        if experiment is None:
            try:
                experiment = self._client.get_experiment(experiment_name_or_id)
            except Exception:
                experiment = None
        if experiment is None:
            return MLflowDiscovery(
                warnings=(f"MLflow experiment {experiment_name_or_id!r} was not found.",)
            )

        try:
            runs = list(
                self._client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=20,
                )
            )
        except Exception as error:
            return MLflowDiscovery(
                lineage=MLflowLineage(
                    experiment_name=_normalize(getattr(experiment, "name", None))
                    or experiment_name_or_id,
                    experiment_id=_normalize(getattr(experiment, "experiment_id", None)),
                ),
                warnings=(f"MLflow experiment search failed: {error}",),
            )

        chosen_run = None
        feature_columns: tuple[str, ...] = ()
        for run in runs:
            run_id = _normalize(getattr(getattr(run, "info", None), "run_id", None))
            feature_columns = self._signature_feature_columns_for_run(run_id)
            chosen_run = run
            if feature_columns:
                break

        if chosen_run is None:
            warnings.append("MLflow experiment has no runs.")
            return MLflowDiscovery(
                lineage=MLflowLineage(
                    experiment_name=_normalize(getattr(experiment, "name", None))
                    or experiment_name_or_id,
                    experiment_id=_normalize(getattr(experiment, "experiment_id", None)),
                ),
                warnings=tuple(warnings),
            )

        run_data = getattr(chosen_run, "data", None)
        tags = getattr(run_data, "tags", {}) or {}
        params = getattr(run_data, "params", {}) or {}
        if not feature_columns:
            warnings.append(
                "MLflow run did not expose a model signature; using table heuristics for features."
            )
        return MLflowDiscovery(
            lineage=MLflowLineage(
                experiment_name=_normalize(getattr(experiment, "name", None))
                or experiment_name_or_id,
                experiment_id=_normalize(getattr(experiment, "experiment_id", None)),
                run_id=_normalize(getattr(getattr(chosen_run, "info", None), "run_id", None))
                or None,
                registered_model_name=_normalize(tags.get("mlflow.registeredModelName")) or None,
                model_version=(
                    _normalize(tags.get("model_version"))
                    or _normalize(params.get("model_version"))
                    or _normalize(tags.get("mlflow.runName"))
                    or None
                ),
            ),
            feature_columns=feature_columns,
            problem_type=self._infer_problem_type(chosen_run),
            warnings=tuple(warnings),
        )

    def _discover_registered_model(self, registered_model_name: str) -> MLflowDiscovery:
        warnings: list[str] = []
        try:
            versions = list(self._client.search_model_versions(f"name='{registered_model_name}'"))
        except Exception as error:
            return MLflowDiscovery(warnings=(f"MLflow registered model lookup failed: {error}",))
        if not versions:
            return MLflowDiscovery(
                warnings=(f"Registered model {registered_model_name!r} was not found.",)
            )

        versions.sort(key=lambda version: self._version_sort_key(getattr(version, "version", None)))
        latest = versions[-1]
        run_id = _normalize(getattr(latest, "run_id", None))
        model_version = _normalize(getattr(latest, "version", None))
        feature_columns = self._signature_feature_columns_for_uri(
            f"models:/{registered_model_name}/{model_version}"
        )
        if not feature_columns and run_id:
            feature_columns = self._signature_feature_columns_for_run(run_id)
        if not feature_columns:
            warnings.append(
                "Registered model did not expose a readable signature; using table heuristics for features."
            )

        run = None
        if run_id:
            try:
                run = self._client.get_run(run_id)
            except Exception:
                run = None

        experiment_name = None
        experiment_id = _normalize(getattr(getattr(run, "info", None), "experiment_id", None))
        if experiment_id:
            try:
                experiment = self._client.get_experiment(experiment_id)
                experiment_name = _normalize(getattr(experiment, "name", None)) or None
            except Exception:
                experiment_name = None

        return MLflowDiscovery(
            lineage=MLflowLineage(
                experiment_name=experiment_name,
                experiment_id=experiment_id or None,
                run_id=run_id or None,
                registered_model_name=registered_model_name,
                model_version=model_version or None,
            ),
            feature_columns=feature_columns,
            problem_type=self._infer_problem_type(run),
            warnings=tuple(warnings),
        )

    def _signature_feature_columns_for_run(self, run_id: str) -> tuple[str, ...]:
        if not run_id:
            return ()
        for model_uri in (f"runs:/{run_id}/model", f"runs:/{run_id}/artifact_path/model"):
            feature_columns = self._signature_feature_columns_for_uri(model_uri)
            if feature_columns:
                return feature_columns
        return ()

    def _signature_feature_columns_for_uri(self, model_uri: str) -> tuple[str, ...]:
        try:
            model_info = self._mlflow.models.get_model_info(model_uri)
        except Exception:
            return ()
        signature = getattr(model_info, "signature", None)
        if signature is None:
            return ()
        return _field_names_from_signature(getattr(signature, "inputs", None))

    def _infer_problem_type(self, run: Any) -> str | None:
        if run is None:
            return None
        run_data = getattr(run, "data", None)
        if run_data is None:
            return None
        metric_names = {
            str(name).lower() for name in (getattr(run_data, "metrics", {}) or {}).keys()
        }
        param_values = {
            str(value).lower()
            for value in (getattr(run_data, "params", {}) or {}).values()
            if value is not None
        }
        tag_values = {
            str(value).lower()
            for value in (getattr(run_data, "tags", {}) or {}).values()
            if value is not None
        }
        classifier_tokens = {"classification", "classifier", "binary", "multiclass"}
        regression_tokens = {"regression", "regressor"}
        if param_values.intersection(classifier_tokens) or tag_values.intersection(
            classifier_tokens
        ):
            return "classification"
        if param_values.intersection(regression_tokens) or tag_values.intersection(
            regression_tokens
        ):
            return "regression"
        if metric_names.intersection({"auc", "roc_auc", "f1", "precision", "recall", "log_loss"}):
            return "classification"
        if metric_names.intersection({"rmse", "mae", "mse", "r2", "mape"}):
            return "regression"
        return None

    def _version_sort_key(self, value: Any) -> tuple[int, str]:
        text = _normalize(value)
        if text.isdigit():
            return int(text), text
        return -1, text
