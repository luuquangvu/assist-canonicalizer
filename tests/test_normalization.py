"""Tests for language-agnostic normalization."""

import pytest

from custom_components.assist_canonicalizer.normalization import (
    char_ngrams,
    char_ngrams_normalized,
    normalize_text,
    normalize_text_no_diacritics,
    tokenize_normalized,
    tokenize_text,
)


def test_normalize_text_no_diacritics() -> None:
    """Test diacritics removal across different languages."""
    # Vietnamese
    assert normalize_text_no_diacritics("Bật ĐÈN phòng khách", "vi") == "bat den phong khach"
    # German transliteration
    assert normalize_text_no_diacritics("Küche", "de") == "kueche"
    # French
    assert normalize_text_no_diacritics("château", "fr") == "chateau"
    assert normalize_text_no_diacritics("lumière", "fr") == "lumiere"
    # English
    assert normalize_text_no_diacritics("living room light", "en") == "living room light"
    # Empty
    assert normalize_text_no_diacritics("") == ""

    # Assertions for all mappings in GENERIC_LATIN_REPLACEMENTS
    assert normalize_text_no_diacritics("đ") == "d"
    assert normalize_text_no_diacritics("ß") == "ss"
    assert normalize_text_no_diacritics("æ") == "ae"
    assert normalize_text_no_diacritics("œ") == "oe"
    assert normalize_text_no_diacritics("ø") == "o"
    assert normalize_text_no_diacritics("ł") == "l"
    assert normalize_text_no_diacritics("ı") == "i"  # noqa: RUF001
    assert normalize_text_no_diacritics("ð") == "d"
    assert normalize_text_no_diacritics("þ") == "th"

    # Assertions for all mappings in LANGUAGE_SPECIFIC_OVERRIDES (e.g. German "de")
    assert normalize_text_no_diacritics("ä", "de") == "ae"
    assert normalize_text_no_diacritics("ö", "de") == "oe"
    assert normalize_text_no_diacritics("ü", "de") == "ue"

    # Assertions that overrides are language-specific and fall back to standard diacritic stripping
    assert normalize_text_no_diacritics("ä", "en") == "a"
    assert normalize_text_no_diacritics("ö", "en") == "o"
    assert normalize_text_no_diacritics("ü", "en") == "u"


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


def test_tokenize_normalized() -> None:
    """Test tokenize_normalized helper."""
    assert tokenize_normalized("bật đèn") == ("bật", "đèn")
    assert tokenize_normalized("") == ()


def test_char_ngrams_edge_cases() -> None:
    """Test edge cases for char_ngrams and char_ngrams_normalized."""
    # short text
    assert char_ngrams("ab", size=3) == frozenset({"ab"})

    # char_ngrams_normalized positive size validation
    with pytest.raises(ValueError, match="N-gram size must be positive"):
        char_ngrams_normalized("abc", size=0)

    # char_ngrams_normalized empty text
    assert char_ngrams_normalized("") == frozenset()

    # char_ngrams_normalized short text
    assert char_ngrams_normalized("ab", size=3) == frozenset({"ab"})

    # char_ngrams_normalized normal text
    assert char_ngrams_normalized("abcdef", size=3) == frozenset({"abc", "bcd", "cde", "def"})
