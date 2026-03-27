from __future__ import annotations

import argparse

from model_lens.services.control_plane import build_repository
from model_lens.services.refresh_engine import refresh_monitor, split_baseline_current
from model_lens.services.refresh_runner import run_refresh_cycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh control-plane metrics")
    parser.add_argument("--catalog", required=False)
    parser.add_argument("--schema", required=False)
    parser.add_argument("--warehouse-id", required=False)
    parser.add_argument("--model-key", required=False, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = build_repository(
        warehouse_id=args.warehouse_id or "",
        catalog=args.catalog,
        schema=args.schema,
    )
    counts = run_refresh_cycle(repository, model_key=args.model_key)
    print(
        "refresh-control-plane complete: "
        f"models={counts.models} drift_rows={counts.drift_rows} "
        f"quality_rows={counts.quality_rows} performance_rows={counts.performance_rows} "
        f"incident_rows={counts.incident_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
