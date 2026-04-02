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
    parser.add_argument("--use-lakebase-read-model", action="store_true")
    parser.add_argument("--lakebase-instance-name", required=False, default="")
    parser.add_argument("--lakebase-database-name", required=False, default="")
    parser.add_argument("--lakebase-host", required=False, default="")
    parser.add_argument("--lakebase-port", required=False, type=int, default=5432)
    parser.add_argument("--lakebase-pguser", required=False, default="")
    parser.add_argument("--lakebase-password", required=False, default="")
    parser.add_argument("--lakebase-sslmode", required=False, default="require")
    parser.add_argument("--lakebase-schema", required=False, default="")
    parser.add_argument("--scope", choices=["scheduler", "bootstrap", "drift_quality", "performance_repair"], default="scheduler")
    parser.add_argument("--mode", choices=["auto", "backfill", "incremental"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = build_repository(
        warehouse_id=args.warehouse_id or "",
        catalog=args.catalog,
        schema=args.schema,
        use_lakebase_read_model=args.use_lakebase_read_model,
        lakebase_instance_name=args.lakebase_instance_name or None,
        lakebase_database_name=args.lakebase_database_name or None,
        lakebase_host=args.lakebase_host or None,
        lakebase_port=args.lakebase_port,
        lakebase_pguser=args.lakebase_pguser or None,
        lakebase_password=args.lakebase_password or None,
        lakebase_sslmode=args.lakebase_sslmode or None,
        lakebase_schema=args.lakebase_schema or None,
    )
    counts = run_refresh_cycle(repository, model_key=args.model_key, mode=args.mode, scope=args.scope)
    print(
        "refresh-control-plane complete: "
        f"scope={args.scope} mode={args.mode} models={counts.models} drift_rows={counts.drift_rows} "
        f"quality_rows={counts.quality_rows} performance_rows={counts.performance_rows} "
        f"incident_rows={counts.incident_rows}"
    )
    return 0


if __name__ == "__main__":
    main()
