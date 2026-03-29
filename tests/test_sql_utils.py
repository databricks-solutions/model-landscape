from __future__ import annotations

import numpy as np

from model_lens.services.sql_utils import parse_string_array


def test_parse_string_array_handles_numpy_arrays() -> None:
    value = np.array(["amount", "velocity_7d", "device_score"])

    assert parse_string_array(value) == ("amount", "velocity_7d", "device_score")


def test_parse_string_array_ignores_mappings() -> None:
    value = {"amount": 1, "velocity_7d": 2}

    assert parse_string_array(value) == ()
