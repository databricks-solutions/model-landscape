from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any


_STAGES: tuple[tuple[str, str], ...] = (
    ("source_metadata_ms", "Source Metadata"),
    ("daily_profiles_ms", "Daily Profiles"),
    ("derivation_ms", "Derivation"),
    ("persistence_ms", "Persistence"),
)

_BOTTLENECK_LABELS = {
    "source_scan_bound": "Source Scan Bound",
    "daily_profiles_bound": "Daily Profiles Bound",
    "derivation_bound": "Derivation Bound",
    "persistence_bound": "Persistence Bound",
    "mixed": "Mixed",
    "insufficient_data": "Insufficient Data",
    "no_runs": "No Runs Yet",
}

_TREND_LABELS = {
    "improving": "Improving",
    "stable": "Stable",
    "degrading": "Degrading",
    "not_enough_history": "Not Enough History",
}


def _safe_int(value: object) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_str(value: object) -> str:
    return str(value or "").strip()


def _duration_ms(run: dict[str, Any]) -> int:
    total = _safe_int(run.get("total_duration_ms"))
    if total > 0:
        return total
    return sum(_safe_int(run.get(column)) for column, _ in _STAGES)


def _stage_breakdown(run: dict[str, Any], *, total_duration_ms: int) -> dict[str, dict[str, float | int | str]]:
    if total_duration_ms <= 0:
        return {
            stage_key: {"label": stage_label, "ms": _safe_int(run.get(stage_key)), "share_pct": 0.0}
            for stage_key, stage_label in _STAGES
        }
    breakdown: dict[str, dict[str, float | int | str]] = {}
    for stage_key, stage_label in _STAGES:
        stage_ms = _safe_int(run.get(stage_key))
        breakdown[stage_key] = {
            "label": stage_label,
            "ms": stage_ms,
            "share_pct": round((stage_ms / total_duration_ms) * 100.0, 1),
        }
    return breakdown


def _classify_bottleneck(status: str, stage_breakdown: dict[str, dict[str, float | int | str]], *, total_duration_ms: int) -> tuple[str, str]:
    if status != "completed" or total_duration_ms <= 0:
        return "insufficient_data", "Insufficient Data"
    dominant_stage_key = max(stage_breakdown, key=lambda key: float(stage_breakdown[key]["share_pct"]))
    dominant_share_pct = float(stage_breakdown[dominant_stage_key]["share_pct"])
    dominant_stage_label = str(stage_breakdown[dominant_stage_key]["label"])
    if dominant_share_pct < 45.0:
        return "mixed", dominant_stage_label
    category_map = {
        "source_metadata_ms": "source_scan_bound",
        "daily_profiles_ms": "daily_profiles_bound",
        "derivation_ms": "derivation_bound",
        "persistence_ms": "persistence_bound",
    }
    return category_map[dominant_stage_key], dominant_stage_label


def _recommendations_for_category(category: str) -> list[str]:
    recommendation_map = {
        "source_scan_bound": [
            "Source scans dominate. Check timestamp partitioning and monitor filters.",
            "For very large bootstraps, narrow the initial source range where possible.",
        ],
        "daily_profiles_bound": [
            "Daily profile generation dominates. Consider more Spark workers for this workload.",
            "For very wide monitors, reduced distribution mode is the next cost lever.",
        ],
        "derivation_bound": [
            "Window derivation dominates. Check affected-window churn and repair horizon size.",
            "For heavy bootstrap/backfill workloads, consider the optional separate bootstrap lane.",
        ],
        "persistence_bound": [
            "Persistence dominates. Check write amplification and the size of the repair/backfill span.",
            "If long repair windows are common, reduce unnecessary rewrites before increasing compute.",
        ],
        "mixed": [
            "No single stage dominates recent successful runs. Review raw stage timings before changing compute.",
        ],
        "insufficient_data": [
            "Need at least 3 successful timed runs for stable diagnostics.",
        ],
        "no_runs": [
            "Run the first refresh to start collecting diagnostics.",
        ],
    }
    return recommendation_map.get(category, recommendation_map["insufficient_data"])


def _trend_label(successful_timed_runs: list[dict[str, Any]]) -> str:
    if len(successful_timed_runs) < 6:
        return "not_enough_history"
    recent = successful_timed_runs[:5]
    previous = successful_timed_runs[5:10]
    if len(previous) < 3:
        return "not_enough_history"
    recent_median = median(int(run["total_duration_ms"]) for run in recent)
    previous_median = median(int(run["total_duration_ms"]) for run in previous)
    if previous_median <= 0:
        return "not_enough_history"
    if recent_median <= previous_median * 0.85:
        return "improving"
    if recent_median >= previous_median * 1.15:
        return "degrading"
    return "stable"


def build_refresh_diagnostics(runs: list[dict[str, Any]] | None) -> dict[str, Any]:
    recent_runs = list(runs or [])
    if not recent_runs:
        return {
            "state": "no_runs",
            "summary": {
                "recent_run_count": 0,
                "successful_run_count": 0,
                "success_rate_pct": 0.0,
                "median_duration_ms": 0,
                "dominant_bottleneck": _BOTTLENECK_LABELS["no_runs"],
                "trend": _TREND_LABELS["not_enough_history"],
                "recommendations": _recommendations_for_category("no_runs"),
            },
            "recent_runs": [],
        }

    diagnosed_runs: list[dict[str, Any]] = []
    for run in recent_runs:
        status = _safe_str(run.get("status")).lower()
        total_duration_ms = _duration_ms(run)
        stage_breakdown = _stage_breakdown(run, total_duration_ms=total_duration_ms)
        category, dominant_stage_label = _classify_bottleneck(status, stage_breakdown, total_duration_ms=total_duration_ms)
        recommendation = _recommendations_for_category(category)[0]
        diagnosed_runs.append(
            {
                "run_id": _safe_str(run.get("run_id")),
                "scope": _safe_str(run.get("scope")),
                "status": _safe_str(run.get("status")),
                "started_at": run.get("started_at"),
                "completed_at": run.get("completed_at"),
                "total_duration_ms": total_duration_ms,
                "dominant_stage": dominant_stage_label,
                "bottleneck_category": category,
                "bottleneck_label": _BOTTLENECK_LABELS[category],
                "recommendation": recommendation,
                "source_metadata_pct": float(stage_breakdown["source_metadata_ms"]["share_pct"]),
                "daily_profiles_pct": float(stage_breakdown["daily_profiles_ms"]["share_pct"]),
                "derivation_pct": float(stage_breakdown["derivation_ms"]["share_pct"]),
                "persistence_pct": float(stage_breakdown["persistence_ms"]["share_pct"]),
            }
        )

    successful_timed_runs = [
        run
        for run in diagnosed_runs
        if run["status"].lower() == "completed" and int(run["total_duration_ms"]) > 0
    ]
    success_rate_pct = round(
        (
            sum(1 for run in diagnosed_runs if run["status"].lower() == "completed")
            / max(len(diagnosed_runs), 1)
        )
        * 100.0,
        1,
    )
    if len(successful_timed_runs) >= 3:
        category_counter = Counter(run["bottleneck_category"] for run in successful_timed_runs)
        dominant_category = category_counter.most_common(1)[0][0]
        state = "ready"
    else:
        dominant_category = "insufficient_data"
        state = "insufficient_data"
    if not successful_timed_runs and diagnosed_runs:
        state = "failure_heavy"
    recommendations = list(_recommendations_for_category(dominant_category))
    if success_rate_pct < 60.0 and diagnosed_runs:
        recommendations.insert(0, "Recent failures limit timing guidance. Review recent error messages first.")
    summary = {
        "recent_run_count": len(diagnosed_runs),
        "successful_run_count": len([run for run in diagnosed_runs if run["status"].lower() == "completed"]),
        "success_rate_pct": success_rate_pct,
        "median_duration_ms": int(median(run["total_duration_ms"] for run in successful_timed_runs)) if successful_timed_runs else 0,
        "dominant_bottleneck": _BOTTLENECK_LABELS[dominant_category],
        "dominant_bottleneck_category": dominant_category,
        "trend": _TREND_LABELS[_trend_label(successful_timed_runs)],
        "recommendations": recommendations,
    }
    return {
        "state": state,
        "summary": summary,
        "recent_runs": diagnosed_runs,
    }
