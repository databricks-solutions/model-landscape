from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableNames:
    catalog: str
    schema: str

    @property
    def namespace(self) -> str:
        return f"{self.catalog}.{self.schema}"

    @property
    def monitor_configs(self) -> str:
        return f"{self.namespace}.monitor_configs"

    @property
    def drift_metrics(self) -> str:
        return f"{self.namespace}.drift_metrics"

    @property
    def quality_metrics(self) -> str:
        return f"{self.namespace}.quality_metrics"

    @property
    def quality_history(self) -> str:
        return f"{self.namespace}.quality_history"

    @property
    def performance_metrics(self) -> str:
        return f"{self.namespace}.performance_metrics"

    @property
    def incidents(self) -> str:
        return f"{self.namespace}.incidents"

    @property
    def incident_history(self) -> str:
        return f"{self.namespace}.incident_history"

    @property
    def refresh_runs(self) -> str:
        return f"{self.namespace}.refresh_runs"

    @property
    def comparison_windows(self) -> str:
        return f"{self.namespace}.comparison_windows"
