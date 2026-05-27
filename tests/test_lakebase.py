from __future__ import annotations

import pandas as pd

from model_landscape.domain.models import InferenceContract, MonitorConfig
from model_landscape.services.lakebase import LakebaseReadModel


class FakeLakebaseConnection:
    def __init__(self) -> None:
        self.configured = True
        self.executed: list[tuple[str, tuple]] = []
        self.executed_many: list[tuple[str, list[tuple]]] = []

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.executed.append((sql, params))

    def execute_many(self, sql: str, rows: list[tuple]) -> None:
        self.executed_many.append((sql, rows))

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        del sql, params
        return pd.DataFrame()


def test_replace_dashboard_projection_uses_upserts_before_pruning() -> None:
    connection = FakeLakebaseConnection()
    read_model = LakebaseReadModel(connection=connection, schema="model_landscape_ui")
    config = MonitorConfig(
        model_key="payments_risk_v1",
        display_name="Payments Risk",
        source_table="catalog.schema.inference_logs",
        contract=InferenceContract(
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            feature_columns=("amount",),
        ),
    )

    read_model.replace_dashboard_projection(
        configs=[config],
        summary=pd.DataFrame(
            [
                {
                    "model_key": "payments_risk_v1",
                    "display_name": "Payments Risk",
                    "max_psi": 0.42,
                    "feature_count": 1,
                    "latest_window_end": "2026-01-21",
                    "total_rows": 500,
                    "latest_data_date": "2026-01-21",
                    "last_refresh_at": "2026-01-21T12:00:00+00:00",
                    "open_incident_count": 1,
                }
            ]
        ),
        incidents=pd.DataFrame(),
    )

    inventory_insert = connection.executed_many[0][0]
    summary_insert = connection.executed_many[1][0]
    assert "ON CONFLICT (model_key) DO UPDATE" in inventory_insert
    assert "ON CONFLICT (model_key) DO UPDATE" in summary_insert
    assert all(
        "DELETE FROM" not in sql or "WHERE model_key NOT IN" in sql or "open_incidents" in sql
        for sql, _ in connection.executed[3:]
    )
