from __future__ import annotations

import numpy as np

from model_landscape.services.sql_utils import array_literal, parse_string_array, quote_column, validate_identifier


def test_parse_string_array_handles_numpy_arrays() -> None:
    value = np.array(["amount", "velocity_7d", "device_score"])

    assert parse_string_array(value) == ("amount", "velocity_7d", "device_score")


def test_parse_string_array_ignores_mappings() -> None:
    value = {"amount": 1, "velocity_7d": 2}

    assert parse_string_array(value) == ()


def test_validate_identifier_allows_hyphenated_column_names() -> None:
    assert validate_identifier("us-central1") == "us-central1"
    assert quote_column("us-central1") == "`us-central1`"
    assert array_literal(["us-central1", "velocity_7d"]) == "ARRAY('us-central1', 'velocity_7d')"


def test_validate_identifier_still_rejects_hyphens_in_qualified_names() -> None:
    try:
        validate_identifier("catalog.schema.us-central1")
    except ValueError as error:
        assert "Invalid SQL identifier" in str(error)
    else:
        raise AssertionError("expected invalid qualified identifier")
