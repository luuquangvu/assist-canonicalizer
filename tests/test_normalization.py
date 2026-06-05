"""Tests for language-agnostic normalization."""

import pytest

from custom_components.assist_canonicalizer.normalization import (
    char_ngrams,
    normalize_text,
    tokenize_text,
)


def test_normalize_text_applies_nfkc_casefold_punctuation_and_spaces() -> None:
    """Normalize text without language-specific rewriting."""
    assert normalize_text("  Bật, ĐÈN!!  Phòng   Khách  ") == "bật đèn phòng khách"


def test_tokenize_text_returns_normalized_tokens() -> None:
    """Tokenize normalized text on whitespace."""
    assert tokenize_text("Turn   ON kitchen-light") == ("turn", "on", "kitchen", "light")


def test_char_ngrams_compacts_normalized_text() -> None:
    """Build character n-grams from compact normalized text."""
    assert char_ngrams("abc def", size=3) == frozenset({"abc", "bcd", "cde", "def"})


def test_tokenize_text_empty() -> None:
    """Tokenize empty or whitespace text."""
    assert tokenize_text("   ") == ()


def test_char_ngrams_validation_and_empty() -> None:
    """Test n-gram boundaries and empty outputs."""
    with pytest.raises(ValueError, match="N-gram size must be positive"):
        char_ngrams("abc", size=0)
    assert char_ngrams("   ") == frozenset()
