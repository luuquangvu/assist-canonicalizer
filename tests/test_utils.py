"""Tests for utility helper functions."""

from __future__ import annotations

from typing import Any

import pytest

from custom_components.assist_canonicalizer.utils import (
    is_valid_range_value,
    parse_float,
    strip_hotword_prefix,
)


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


def test_strip_hotword_prefix() -> None:
    """Test stripping matched hotword prefixes and leading punctuation from queries."""
    # 1. Single word hotword with comma
    assert (
        strip_hotword_prefix("Jarvis, what is the weather tomorrow?", "Jarvis")
        == "what is the weather tomorrow?"
    )

    # 2. Multi-word hotword with colon
    assert strip_hotword_prefix("Hey Jarvis: tell me a joke!", "Hey Jarvis") == "tell me a joke!"

    # 3. Multi-word with internal and trailing punctuation
    assert strip_hotword_prefix("Hey, Jarvis, how are you?", "Hey Jarvis") == "how are you?"

    # 4. Single word hotword with dash
    assert strip_hotword_prefix("Jarvis - what is 2+2?", "Jarvis") == "what is 2+2?"

    # 5. Fuzzy matched typo in query
    assert strip_hotword_prefix("Jarviss turn off the lights", "Jarvis") == "turn off the lights"

    # 6. Multi-word with diacritics / Vietnamese
    assert (
        strip_hotword_prefix("Trợ lý ơi, bật đèn phòng khách giùm tôi", "Tro ly oi")
        == "bật đèn phòng khách giùm tôi"
    )

    # 7. Hotword only - must retain original query text to avoid empty prompt to LLM
    assert strip_hotword_prefix("Jarvis", "Jarvis") == "Jarvis"
    assert strip_hotword_prefix("Jarvis???", "Jarvis") == "Jarvis???"
    assert strip_hotword_prefix("Hey Jarvis", "Hey Jarvis") == "Hey Jarvis"
    assert strip_hotword_prefix("  Hey Jarvis  ", "Hey Jarvis") == "Hey Jarvis"

    # 8. Empty input edge cases
    assert strip_hotword_prefix("", "Jarvis") == ""
    assert strip_hotword_prefix("Hello world", "") == "Hello world"
    assert strip_hotword_prefix("   ", "Jarvis") == ""

    # 9. Numeric hotwords with decimal, time colon, unit suffix, and expressions
    assert (
        strip_hotword_prefix("Jarvis 2.0, what is the weather?", "Jarvis 2.0")
        == "what is the weather?"
    )
    assert strip_hotword_prefix("2+2 equals four", "2+2") == "equals four"
    assert strip_hotword_prefix("Jarvis 12:30 set an alarm", "Jarvis 12:30") == "set an alarm"
    assert (
        strip_hotword_prefix("Jarvis 50% increase the brightness", "Jarvis 50%")
        == "increase the brightness"
    )

    # 10. Unicode minus sign and compatibility-width characters
    assert strip_hotword_prefix("Jarvis \u22125 degrees", "Jarvis -5") == "degrees"
    assert (
        strip_hotword_prefix("Jarvis \u22125, what is the temperature?", "Jarvis \u22125")
        == "what is the temperature?"
    )
    assert (
        strip_hotword_prefix("\uff2a\uff41\uff52\uff56\uff49\uff53 turn off the lights", "Jarvis")
        == "turn off the lights"
    )
    assert (
        strip_hotword_prefix(
            "\uff2a\uff41\uff52\uff56\uff49\uff53\u3000\u2212\uff15 degrees",
            "Jarvis -5",
        )
        == "degrees"
    )
    assert (
        strip_hotword_prefix(
            "\uff28\uff45\uff59\u3000\uff2a\uff41\uff52\uff56\uff49\uff53: tell me a joke!",
            "Hey Jarvis",
        )
        == "tell me a joke!"
    )

    # 11. Full-width punctuation stripping
    assert (
        strip_hotword_prefix("Hey Jarvis\uff1a tell me a joke!", "Hey Jarvis") == "tell me a joke!"
    )
    assert (
        strip_hotword_prefix("Hey Jarvis\uff1f what is the time?", "Hey Jarvis")
        == "what is the time?"
    )
    assert strip_hotword_prefix("Hey Jarvis\uff0c turn on lights", "Hey Jarvis") == "turn on lights"
    assert strip_hotword_prefix("Hey Jarvis\uff01 turn on lights", "Hey Jarvis") == "turn on lights"
    assert strip_hotword_prefix("Hey Jarvis\u3002 turn on lights", "Hey Jarvis") == "turn on lights"
    assert strip_hotword_prefix("Jarvis\uff1f\uff1f\uff1f", "Jarvis") == "Jarvis\uff1f\uff1f\uff1f"

    # 12. Unmatched prefixes - must return original query unmodified
    assert strip_hotword_prefix("Hello world", "Jarvis") == "Hello world"
    assert (
        strip_hotword_prefix("Turn on the living room lights", "Jarvis")
        == "Turn on the living room lights"
    )
    assert (
        strip_hotword_prefix("Japanese restaurants nearby", "Jarvis")
        == "Japanese restaurants nearby"
    )
    assert strip_hotword_prefix("Hey", "Hey Jarvis") == "Hey"
