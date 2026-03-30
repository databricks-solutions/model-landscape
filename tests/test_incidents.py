from model_lens.services.incidents import build_incident_history, build_incidents


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


def test_build_incident_history_tracks_lifecycle_across_windows() -> None:
    history = build_incident_history(
        [
            {
                "model_key": "m1",
                "feature_name": "amount",
                "metric_name": "psi",
                "metric_value": 0.12,
                "window_id": "w1",
                "window_start": "2026-03-01",
                "window_end": "2026-03-07",
                "baseline_start": "2026-02-23",
                "baseline_end": "2026-02-29",
                "computed_at": "2026-03-07T00:00:00+00:00",
            },
            {
                "model_key": "m1",
                "feature_name": "amount",
                "metric_name": "psi",
                "metric_value": 0.31,
                "window_id": "w2",
                "window_start": "2026-03-02",
                "window_end": "2026-03-08",
                "baseline_start": "2026-02-24",
                "baseline_end": "2026-03-01",
                "computed_at": "2026-03-08T00:00:00+00:00",
            },
        ],
        [
            {
                "window_id": "w1",
                "window_start": "2026-03-01",
                "window_end": "2026-03-07",
                "baseline_start": "2026-02-23",
                "baseline_end": "2026-02-29",
                "created_at": "2026-03-07T00:00:00+00:00",
            },
            {
                "window_id": "w2",
                "window_start": "2026-03-02",
                "window_end": "2026-03-08",
                "baseline_start": "2026-02-24",
                "baseline_end": "2026-03-01",
                "created_at": "2026-03-08T00:00:00+00:00",
            },
            {
                "window_id": "w3",
                "window_start": "2026-03-03",
                "window_end": "2026-03-09",
                "baseline_start": "2026-02-25",
                "baseline_end": "2026-03-02",
                "created_at": "2026-03-09T00:00:00+00:00",
            },
        ],
    )

    assert [row["event_type"] for row in history] == ["opened", "escalated", "recovered"]
    assert [row["status"] for row in history] == ["open", "open", "closed"]
    assert history[-1]["metric_value"] == 0.0


def test_build_incident_history_uses_prior_open_state_for_incremental_windows() -> None:
    history = build_incident_history(
        [],
        [
            {
                "window_id": "w4",
                "window_start": "2026-03-04",
                "window_end": "2026-03-10",
                "baseline_start": "2026-02-26",
                "baseline_end": "2026-03-03",
                "created_at": "2026-03-10T00:00:00+00:00",
            }
        ],
        prior_open_incidents={
            ("m1", "amount", "psi"): {
                "model_key": "m1",
                "feature_name": "amount",
                "metric_name": "psi",
                "severity": "warning",
                "metric_value": 0.12,
                "status": "open",
            }
        },
    )

    assert len(history) == 1
    assert history[0]["event_type"] == "recovered"
    assert history[0]["status"] == "closed"
