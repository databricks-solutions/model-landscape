from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

_IDENTIFIER_PART_RE = re.compile(r"\A[a-zA-Z0-9_]+\Z")
_COLUMN_IDENTIFIER_RE = re.compile(r"\A[a-zA-Z0-9_][a-zA-Z0-9_-]*\Z")


def validate_identifier(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    if "." in text:
        parts = text.split(".")
        if not all(_IDENTIFIER_PART_RE.match(part) for part in parts):
            raise ValueError(f"Invalid SQL identifier: {name!r}")
        return text
    if not _COLUMN_IDENTIFIER_RE.match(text):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return text


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
        return tuple(str(item) for item in value)
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
    if isinstance(value, Mapping):
        return ()
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return ()
