"""Tests for utility helper functions."""

from __future__ import annotations

from typing import Any

import pytest

from custom_components.assist_canonicalizer.utils import is_valid_range_value, parse_float


@pytest.mark.parametrize(
    ("val_str", "expected"),
    [
        ("27.5", 27.5),
        ("27,5", 27.5),
        ("0", 0.0),
        ("-5", -5.0),
        ("-5°", -5.0),
        (" -3.5 % ", -3.5),
        ("27,5%", 27.5),
        (" +3.14 ", 3.14),
        ("27.5°", 27.5),
        ("27.5 °", 27.5),
        ("50%", 50.0),
        ("50 %", 50.0),
        ("  50 %  ", 50.0),
        ("abc", None),
        ("°", None),
        ("%", None),
        ("", None),
        ("   ", None),
        (None, None),
        (123, None),
    ],
)
def test_parse_float(val_str: Any, expected: float | None) -> None:
    """Verify that parse_float accurately parses various string representations of floats."""
    assert parse_float(val_str) == expected


def test_is_valid_range_value() -> None:
    """Verify range step grid alignment calculations under various conditions."""
    # No step: any value within boundary is valid
    assert is_valid_range_value(5.0, 0.0, None, None) is True

    # Numeric step matching
    assert is_valid_range_value(10.0, 10.0, 2.0, None) is True
    assert is_valid_range_value(8.0, 10.0, 2.0, None) is True
    assert is_valid_range_value(9.0, 10.0, 2.0, None) is False

    # Epsilon / float precision tolerance matching
    assert is_valid_range_value(10.0000000001, 10.0, 2.0, None) is True

    # Descending ranges / starting points
    assert is_valid_range_value(4.0, 10.0, 2.0, None) is True
    assert is_valid_range_value(5.0, 10.0, 2.0, None) is False

    # Fraction type matching: 'halves' (allows integers and .5 values)
    assert is_valid_range_value(2.5, 1.0, 0.5, "halves") is True
    assert is_valid_range_value(2.5, 1.0, 1.0, "halves") is True
    assert is_valid_range_value(2.25, 1.0, 1.0, "halves") is False
    assert is_valid_range_value(3.0, 1.0, 1.0, "halves") is True

    # Fraction type matching: 'tenths'
    assert is_valid_range_value(2.1, 1.0, 1.0, "tenths") is True
    assert is_valid_range_value(2.15, 1.0, 1.0, "tenths") is False

    # Zero step safety check
    assert is_valid_range_value(5.0, 0.0, 0.0, None) is False
