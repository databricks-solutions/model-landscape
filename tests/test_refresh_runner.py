"""Unit tests for refresh_runner orchestration logic.

Tests the target selection, cadence scheduling, retry handling, and
result aggregation — the orchestration heart of the refresh system.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from model_landscape.domain.models import (
    BaselinePolicy,
    InferenceContract,
    MonitorConfig,
    MonitorRuntimeState,
)
from model_landscape.services.refresh_runner import (
    MonitorRefreshResult,
    RefreshBatchResult,
    RefreshCounts,
    RefreshTarget,
    _bootstrap_range,
    _coerce_timestamp,
    _drift_cadence_delta,
    _failed_recently,
    _is_due,
    _performance_cadence_delta,
    _performance_repair_days,
    _schedule_state_after_failure,
    _schedule_state_after_success,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    model_key: str = "test_model",
    drift_cadence: str = "daily",
    performance_cadence: str = "disabled",
    baseline_n_days: int = 7,
) -> MonitorConfig:
    return MonitorConfig(
        model_key=model_key,
        display_name=model_key,
        source_table="catalog.schema.inference",
        contract=InferenceContract(
            timestamp_col="event_ts",
            prediction_col="prediction",
        ),
        baseline=BaselinePolicy(kind="rolling", n_days=baseline_n_days),
        drift_cadence_preset=drift_cadence,
        performance_cadence_preset=performance_cadence,
    )


def _make_state(
    model_key: str = "test_model",
    bootstrap_status: str = "completed",
    last_drift_refresh_at: str | None = None,
    next_drift_due_at: str | None = None,
    consecutive_failures: int = 0,
    backoff_until: str | None = None,
) -> MonitorRuntimeState:
    return MonitorRuntimeState(
        model_key=model_key,
        bootstrap_status=bootstrap_status,
        last_drift_refresh_at=last_drift_refresh_at,
        next_drift_due_at=next_drift_due_at,
        consecutive_failures=consecutive_failures,
        backoff_until=backoff_until,
    )


# ---------------------------------------------------------------------------
# Cadence delta tests
# ---------------------------------------------------------------------------

class TestCadenceDeltas:
    def test_drift_hourly(self) -> None:
        assert _drift_cadence_delta("hourly") == timedelta(hours=1)

    def test_drift_6h(self) -> None:
        assert _drift_cadence_delta("6h") == timedelta(hours=6)

    def test_drift_daily(self) -> None:
        assert _drift_cadence_delta("daily") == timedelta(days=1)

    def test_drift_manual_returns_none(self) -> None:
        assert _drift_cadence_delta("manual") is None

    def test_performance_disabled_returns_none(self) -> None:
        assert _performance_cadence_delta("disabled") is None

    def test_performance_6h_3d(self) -> None:
        assert _performance_cadence_delta("6h_3d_repair") == timedelta(hours=6)

    def test_performance_daily_7d(self) -> None:
        assert _performance_cadence_delta("daily_7d_repair") == timedelta(days=1)

    def test_performance_daily_14d(self) -> None:
        assert _performance_cadence_delta("daily_14d_repair") == timedelta(days=1)

    def test_performance_repair_days(self) -> None:
        assert _performance_repair_days("6h_3d_repair") == 3
        assert _performance_repair_days("daily_7d_repair") == 7
        assert _performance_repair_days("daily_14d_repair") == 14
        assert _performance_repair_days("disabled") is None
        assert _performance_repair_days("manual") is None


# ---------------------------------------------------------------------------
# Timestamp coercion
# ---------------------------------------------------------------------------

class TestCoerceTimestamp:
    def test_iso_string(self) -> None:
        result = _coerce_timestamp("2026-01-15T10:00:00+00:00")
        assert isinstance(result, pd.Timestamp)

    def test_none_returns_none(self) -> None:
        assert _coerce_timestamp(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _coerce_timestamp("") is None

    def test_pandas_timestamp(self) -> None:
        ts = pd.Timestamp("2026-01-15", tz="UTC")
        assert _coerce_timestamp(ts) == ts


# ---------------------------------------------------------------------------
# Is-due logic
# ---------------------------------------------------------------------------

class TestIsDue:
    def test_none_is_not_due(self) -> None:
        """No next_due_at means the monitor hasn't been scheduled yet."""
        now = pd.Timestamp("2026-01-15T12:00:00", tz="UTC")
        assert _is_due(None, now) is False

    def test_past_due(self) -> None:
        now = pd.Timestamp("2026-01-15T12:00:00", tz="UTC")
        assert _is_due("2026-01-15T11:00:00+00:00", now) is True

    def test_not_yet_due(self) -> None:
        now = pd.Timestamp("2026-01-15T12:00:00", tz="UTC")
        assert _is_due("2026-01-15T13:00:00+00:00", now) is False


# ---------------------------------------------------------------------------
# Failed-recently / backoff
# ---------------------------------------------------------------------------

class TestFailedRecently:
    def test_no_failures(self) -> None:
        state = _make_state(consecutive_failures=0)
        now = pd.Timestamp("2026-01-15T12:00:00", tz="UTC")
        assert _failed_recently(state, now) is False

    def test_backoff_active(self) -> None:
        state = _make_state(
            consecutive_failures=2,
            backoff_until="2026-01-15T13:00:00+00:00",
        )
        now = pd.Timestamp("2026-01-15T12:00:00", tz="UTC")
        assert _failed_recently(state, now) is True

    def test_backoff_expired(self) -> None:
        state = _make_state(
            consecutive_failures=2,
            backoff_until="2026-01-15T11:00:00+00:00",
        )
        now = pd.Timestamp("2026-01-15T12:00:00", tz="UTC")
        assert _failed_recently(state, now) is False


# ---------------------------------------------------------------------------
# Bootstrap range
# ---------------------------------------------------------------------------

class TestBootstrapRange:
    def test_rolling_baseline(self) -> None:
        config = _make_config(baseline_n_days=7)
        latest = pd.Timestamp("2026-01-20")
        start, end = _bootstrap_range(config, latest)
        assert start is not None
        assert end is not None
        # End should be close to latest
        assert end == "2026-01-20"


# ---------------------------------------------------------------------------
# Schedule state transitions
# ---------------------------------------------------------------------------

class TestScheduleStateAfterSuccess:
    def test_drift_success_sets_next_due(self) -> None:
        config = _make_config(drift_cadence="daily")
        state = _make_state()
        completed = pd.Timestamp("2026-01-15T12:00:00", tz="UTC")
        new_state = _schedule_state_after_success(
            config, state, scope="drift_quality",
            completed_at=completed, label_watermark=None,
        )
        assert new_state.last_drift_refresh_at is not None
        assert new_state.next_drift_due_at is not None
        assert new_state.consecutive_failures == 0
        assert new_state.last_run_status == "completed"

    def test_bootstrap_success_marks_completed(self) -> None:
        config = _make_config()
        state = _make_state(bootstrap_status="pending")
        completed = pd.Timestamp("2026-01-15T12:00:00", tz="UTC")
        new_state = _schedule_state_after_success(
            config, state, scope="bootstrap",
            completed_at=completed, label_watermark=None,
        )
        assert new_state.bootstrap_status == "completed"


class TestScheduleStateAfterFailure:
    def test_failure_increments_counter(self) -> None:
        state = _make_state(consecutive_failures=1)
        completed = pd.Timestamp("2026-01-15T12:00:00", tz="UTC")
        new_state = _schedule_state_after_failure(
            state, completed_at=completed, error_message="test error",
        )
        assert new_state.consecutive_failures == 2
        assert new_state.last_run_status == "failed"
        assert new_state.last_run_error == "test error"
        assert new_state.backoff_until is not None


# ---------------------------------------------------------------------------
# Batch result aggregation
# ---------------------------------------------------------------------------

class TestRefreshBatchResult:
    def test_empty_batch(self) -> None:
        batch = RefreshBatchResult(requested_scope="scheduler", requested_mode="auto", results=())
        assert batch.models == 0
        assert batch.drift_rows == 0
        assert batch.performance_rows == 0
        assert batch.incident_rows == 0

    def test_aggregation(self) -> None:
        r1 = MonitorRefreshResult(
            model_key="a", scope="drift_quality", status="completed",
            counts=RefreshCounts(models=1, drift_rows=10, quality_rows=5, performance_rows=3, incident_rows=2),
        )
        r2 = MonitorRefreshResult(
            model_key="b", scope="drift_quality", status="completed",
            counts=RefreshCounts(models=1, drift_rows=20, quality_rows=8, performance_rows=0, incident_rows=1),
        )
        batch = RefreshBatchResult(requested_scope="scheduler", requested_mode="auto", results=(r1, r2))
        assert batch.models == 2
        assert batch.drift_rows == 30
        assert batch.quality_rows == 13
        assert batch.performance_rows == 3
        assert batch.incident_rows == 3

    def test_failed_result_still_aggregates(self) -> None:
        r = MonitorRefreshResult(
            model_key="x", scope="bootstrap", status="failed",
            counts=RefreshCounts(models=1, drift_rows=0, quality_rows=0, performance_rows=0, incident_rows=0),
            error="boom",
        )
        batch = RefreshBatchResult(requested_scope="bootstrap", requested_mode="auto", results=(r,))
        assert batch.models == 1
        assert batch.drift_rows == 0
