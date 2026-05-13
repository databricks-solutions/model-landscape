from __future__ import annotations

from model_landscape.domain.models import BaselinePolicy


def build_default_baseline(n_days: int = 7, max_comparison_days: int = 90) -> BaselinePolicy:
    if n_days < 1:
        raise ValueError("n_days must be positive")
    return BaselinePolicy(
        kind="rolling",
        n_days=n_days,
        max_comparison_days=max_comparison_days,
    )


def build_fixed_baseline(
    baseline_start: str,
    baseline_end: str,
    *,
    max_comparison_days: int = 90,
) -> BaselinePolicy:
    return BaselinePolicy(
        kind="fixed",
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        max_comparison_days=max_comparison_days,
    )


def baseline_label(policy: BaselinePolicy) -> str:
    return policy.label
