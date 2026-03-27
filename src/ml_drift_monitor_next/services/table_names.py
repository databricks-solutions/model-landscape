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
    def performance_metrics(self) -> str:
        return f"{self.namespace}.performance_metrics"

    @property
    def incidents(self) -> str:
        return f"{self.namespace}.incidents"

