from ml_drift_monitor_next.services.incidents import build_incidents


def test_build_incidents_deduplicates_by_business_key() -> None:
    incidents = build_incidents([
        {
            "model_key": "m1",
            "feature_name": "amount",
            "metric_name": "psi",
            "metric_value": 0.12,
            "window_end": "2026-03-27",
        },
        {
            "model_key": "m1",
            "feature_name": "amount",
            "metric_name": "psi",
            "metric_value": 0.31,
            "window_end": "2026-03-27",
        },
    ])
    assert len(incidents) == 1
    assert incidents[0]["severity"] == "critical"
    assert incidents[0]["metric_value"] == 0.31

