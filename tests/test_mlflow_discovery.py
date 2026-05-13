from __future__ import annotations

import types

from model_landscape.services.mlflow_discovery import MLflowDiscoveryService


class _FakeField:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSignature:
    def __init__(self, names: list[str]) -> None:
        self.inputs = [_FakeField(name) for name in names]


class _FakeModelInfo:
    def __init__(self, names: list[str]) -> None:
        self.signature = _FakeSignature(names)


class _FakeRunInfo:
    def __init__(self, run_id: str, experiment_id: str = "exp-1") -> None:
        self.run_id = run_id
        self.experiment_id = experiment_id


class _FakeRunData:
    def __init__(self, *, metrics: dict[str, float], params: dict[str, str] | None = None, tags: dict[str, str] | None = None) -> None:
        self.metrics = metrics
        self.params = params or {}
        self.tags = tags or {}


class _FakeRun:
    def __init__(self, run_id: str, *, metrics: dict[str, float], params: dict[str, str] | None = None, tags: dict[str, str] | None = None) -> None:
        self.info = _FakeRunInfo(run_id)
        self.data = _FakeRunData(metrics=metrics, params=params, tags=tags)


class _FakeExperiment:
    def __init__(self, experiment_id: str, name: str) -> None:
        self.experiment_id = experiment_id
        self.name = name


class _FakeVersion:
    def __init__(self, version: str, run_id: str) -> None:
        self.version = version
        self.run_id = run_id


class _FakeClient:
    def __init__(self) -> None:
        self._run = _FakeRun(
            "run-123",
            metrics={"auc": 0.93, "f1": 0.84},
            params={"problem_type": "classification"},
            tags={"mlflow.registeredModelName": "catalog.schema.fraud_model"},
        )

    def get_experiment_by_name(self, name: str):
        if name == "/Users/example/fraud":
            return _FakeExperiment("exp-1", name)
        return None

    def get_experiment(self, experiment_id: str):
        return _FakeExperiment(experiment_id, f"/Users/example/{experiment_id}")

    def search_runs(self, experiment_ids, order_by, max_results):
        del experiment_ids, order_by, max_results
        return [self._run]

    def search_model_versions(self, query: str):
        del query
        return [_FakeVersion("5", "run-123")]

    def get_run(self, run_id: str):
        assert run_id == "run-123"
        return self._run


def test_mlflow_discovery_reads_registered_model_signature(monkeypatch) -> None:
    fake_mlflow = types.SimpleNamespace(
        MlflowClient=_FakeClient,
        set_registry_uri=lambda uri: uri,
        models=types.SimpleNamespace(get_model_info=lambda uri: _FakeModelInfo(["amount", "velocity_7d"])),
    )
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    service = MLflowDiscoveryService()
    discovery = service.discover(registered_model_name="catalog.schema.fraud_model")

    assert discovery.lineage.registered_model_name == "catalog.schema.fraud_model"
    assert discovery.lineage.model_version == "5"
    assert discovery.lineage.run_id == "run-123"
    assert discovery.feature_columns == ("amount", "velocity_7d")
    assert discovery.problem_type == "classification"


def test_mlflow_discovery_reads_experiment_metadata(monkeypatch) -> None:
    fake_mlflow = types.SimpleNamespace(
        MlflowClient=_FakeClient,
        set_registry_uri=lambda uri: uri,
        models=types.SimpleNamespace(get_model_info=lambda uri: _FakeModelInfo(["amount", "velocity_7d"])),
    )
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    service = MLflowDiscoveryService()
    discovery = service.discover(experiment_name_or_id="/Users/example/fraud")

    assert discovery.lineage.experiment_name == "/Users/example/fraud"
    assert discovery.lineage.experiment_id == "exp-1"
    assert discovery.lineage.run_id == "run-123"
    assert discovery.lineage.registered_model_name == "catalog.schema.fraud_model"
    assert discovery.feature_columns == ("amount", "velocity_7d")
