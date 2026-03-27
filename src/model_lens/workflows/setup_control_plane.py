from __future__ import annotations

import argparse

from model_lens.services.control_plane import build_repository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create control-plane catalog, schema, and tables")
    parser.add_argument("--catalog", required=False)
    parser.add_argument("--schema", required=False)
    parser.add_argument("--warehouse-id", required=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = build_repository(
        warehouse_id=args.warehouse_id or "",
        catalog=args.catalog,
        schema=args.schema,
    )
    repository.ensure_control_plane()
    print(
        "setup-control-plane complete: "
        f"{repository.table_names.catalog}.{repository.table_names.schema}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
