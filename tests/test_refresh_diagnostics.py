from __future__ import annotations

from model_landscape.services.refresh_diagnostics import build_refresh_diagnostics


def test_build_refresh_diagnostics_reports_no_runs_cleanly() -> None:
    diagnostics = build_refresh_diagnostics([])

    assert diagnostics["state"] == "no_runs"
    assert diagnostics["summary"]["dominant_bottleneck"] == "No Runs Yet"
    assert diagnostics["summary"]["compute_footprint"] == "No compute footprint data yet"
    assert diagnostics["summary"]["recommendations"] == ["Run the first refresh to start collecting diagnostics."]


def test_build_refresh_diagnostics_classifies_source_scan_bound_runs() -> None:
    runs = [
        {
            "status": "completed",
            "scope": "drift_quality",
            "started_at": "2026-01-21T10:00:00",
            "source_metadata_ms": 220_000,
            "daily_profiles_ms": 90_000,
            "derivation_ms": 80_000,
            "persistence_ms": 50_000,
            "total_duration_ms": 440_000,
            "rows_scanned": 2_000_000,
        },
        {
            "status": "completed",
            "scope": "drift_quality",
            "started_at": "2026-01-20T10:00:00",
            "source_metadata_ms": 240_000,
            "daily_profiles_ms": 80_000,
            "derivation_ms": 70_000,
            "persistence_ms": 50_000,
            "total_duration_ms": 440_000,
            "rows_scanned": 2_500_000,
        },
        {
            "status": "completed",
            "scope": "drift_quality",
            "started_at": "2026-01-19T10:00:00",
            "source_metadata_ms": 210_000,
            "daily_profiles_ms": 95_000,
            "derivation_ms": 85_000,
            "persistence_ms": 50_000,
            "total_duration_ms": 440_000,
            "rows_scanned": 1_800_000,
        },
    ]

    diagnostics = build_refresh_diagnostics(runs)

    assert diagnostics["state"] == "ready"
    assert diagnostics["summary"]["dominant_bottleneck"] == "Source Scan Bound"
    assert diagnostics["summary"]["compute_footprint"] == "Low"
    assert "Source scans dominate" in diagnostics["summary"]["recommendations"][0]
    assert diagnostics["recent_runs"][0]["bottleneck_category"] == "source_scan_bound"


def test_build_refresh_diagnostics_detects_degrading_trend() -> None:
    runs = [
        {
            "status": "completed",
            "scope": "drift_quality",
            "started_at": f"2026-01-{day:02d}T10:00:00",
            "source_metadata_ms": duration // 4,
            "daily_profiles_ms": duration // 4,
            "derivation_ms": duration // 4,
            "persistence_ms": duration // 4,
            "total_duration_ms": duration,
        }
        for day, duration in (
            (21, 240_000),
            (20, 250_000),
            (19, 260_000),
            (18, 270_000),
            (17, 280_000),
            (16, 180_000),
            (15, 190_000),
            (14, 200_000),
        )
    ]

    diagnostics = build_refresh_diagnostics(runs)

    assert diagnostics["summary"]["trend"] == "Degrading"


def test_build_refresh_diagnostics_flags_elevated_compute_footprint() -> None:
    runs = [
        {
            "status": "completed",
            "scope": "drift_quality",
            "started_at": "2026-01-21T10:00:00",
            "source_metadata_ms": 200_000,
            "daily_profiles_ms": 200_000,
            "derivation_ms": 200_000,
            "persistence_ms": 100_000,
            "total_duration_ms": 700_000,
            "rows_scanned": 6_500_000,
        },
        {
            "status": "completed",
            "scope": "drift_quality",
            "started_at": "2026-01-20T10:00:00",
            "source_metadata_ms": 180_000,
            "daily_profiles_ms": 180_000,
            "derivation_ms": 180_000,
            "persistence_ms": 80_000,
            "total_duration_ms": 620_000,
            "rows_scanned": 5_100_000,
        },
        {
            "status": "completed",
            "scope": "drift_quality",
            "started_at": "2026-01-19T10:00:00",
            "source_metadata_ms": 170_000,
            "daily_profiles_ms": 170_000,
            "derivation_ms": 170_000,
            "persistence_ms": 80_000,
            "total_duration_ms": 590_000,
            "rows_scanned": 4_900_000,
        },
    ]

    diagnostics = build_refresh_diagnostics(runs)

    assert diagnostics["summary"]["compute_footprint"] == "Elevated"


def test_build_refresh_diagnostics_flags_failure_heavy_history() -> None:
    runs = [
        {"status": "failed", "scope": "bootstrap", "started_at": "2026-01-21T10:00:00", "total_duration_ms": 0},
        {"status": "failed", "scope": "bootstrap", "started_at": "2026-01-20T10:00:00", "total_duration_ms": 0},
        {
            "status": "completed",
            "scope": "drift_quality",
            "started_at": "2026-01-19T10:00:00",
            "source_metadata_ms": 40_000,
            "daily_profiles_ms": 120_000,
            "derivation_ms": 40_000,
            "persistence_ms": 20_000,
            "total_duration_ms": 220_000,
        },
    ]

    diagnostics = build_refresh_diagnostics(runs)

    assert diagnostics["state"] == "insufficient_data"
    assert diagnostics["summary"]["success_rate_pct"] == 33.3
    assert diagnostics["summary"]["recommendations"][0].startswith("Recent failures limit timing guidance")
