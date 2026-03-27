from __future__ import annotations

import json
import re
from typing import Any


_IDENTIFIER_RE = re.compile(r"\A[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*\Z")


def validate_identifier(name: str) -> str:
    if not name or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def quote_column(name: str) -> str:
    validate_identifier(name)
    return f"`{name}`"


def array_literal(values: list[str] | tuple[str, ...]) -> str:
    if not values:
        return "CAST(ARRAY() AS ARRAY<STRING>)"
    cleaned = [validate_identifier(value) for value in values]
    return "ARRAY(" + ", ".join(f"'{value}'" for value in cleaned) + ")"


def parse_string_array(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return tuple(str(item) for item in parsed)
        except json.JSONDecodeError:
            pass
        stripped = stripped.strip("[]{}")
        if not stripped:
            return ()
        return tuple(part.strip().strip("'\"") for part in stripped.split(",") if part.strip())
    return ()


def catalog_schema(table_name: str) -> str:
    parts = validate_identifier(table_name).split(".")
    if len(parts) < 3:
        raise ValueError("Expected fully qualified table name catalog.schema.table")
    return f"{parts[0]}.{parts[1]}"

