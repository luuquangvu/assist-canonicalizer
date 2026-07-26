"""Tests for language-agnostic normalization."""

import pytest

from custom_components.assist_canonicalizer.normalization import (
    char_ngrams_normalized,
    clear_normalization_caches,
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


@pytest.mark.parametrize(
    "text",
    [
        "বাতি চালু করুন",
        "बत्ती जलाओ",
        "ഓണാക്കൂ",
        "ਚਲਾਦੋ",
        "లైట్ ఆన్ చెయ్యి",
        "เปิดไฟห้องครัว",
        "اَلْعَرَبِيَّةُ",
        "מִטְבָּח",
    ],
)
def test_normalize_text_preserves_script_forming_marks(text: str) -> None:
    """Keep Unicode marks that are part of letters across installed scripts."""
    assert normalize_text(text) == text


def test_normalize_text_handles_join_controls_and_orphan_visual_marks() -> None:
    """Remove format controls without splitting words or retaining emoji marks."""
    assert normalize_text("می\u200cخواهم") == "میخواهم"  # noqa: RUF001
    assert normalize_text("❤️ light") == "light"
    assert normalize_text("\u0301\u0300 light") == "light"
    assert normalize_text("बत्ती।जलाओ!") == "बत्ती जलाओ"


def test_no_diacritic_normalization_only_folds_latin_marks() -> None:
    """Do not turn non-Latin words into a different lexical representation."""
    assert normalize_text_no_diacritics("Cafe\u0301") == "cafe"
    assert normalize_text_no_diacritics("बत्ती जलाओ", "hi") == "बत्ती जलाओ"
    assert normalize_text_no_diacritics("اَلْعَرَبِيَّةُ", "ar") == "اَلْعَرَبِيَّةُ"


def test_tokenize_text_returns_normalized_tokens() -> None:
    """Tokenize normalized text on whitespace."""
    assert tokenize_text("Turn   ON kitchen-light") == ("turn", "on", "kitchen", "light")


def test_char_ngrams_compacts_normalized_text() -> None:
    """Build character n-grams from compact normalized text."""
    assert char_ngrams_normalized(normalize_text("abc def"), size=3) == frozenset(
        {"abc", "bcd", "cde", "def"}
    )


def test_tokenize_text_empty() -> None:
    """Tokenize empty or whitespace text."""
    assert tokenize_text("   ") == ()


def test_tokenize_normalized() -> None:
    """Test tokenize_normalized helper."""
    assert tokenize_normalized("bật đèn") == ("bật", "đèn")
    assert tokenize_normalized("") == ()


def test_char_ngrams_edge_cases() -> None:
    """Test edge cases for char_ngrams_normalized."""
    # char_ngrams_normalized positive size validation
    with pytest.raises(ValueError, match="N-gram size must be positive"):
        char_ngrams_normalized("abc", size=0)

    # char_ngrams_normalized empty text
    assert char_ngrams_normalized("") == frozenset()

    # char_ngrams_normalized short text
    assert char_ngrams_normalized("ab", size=3) == frozenset({"ab"})

    # char_ngrams_normalized normal text
    assert char_ngrams_normalized("abcdef", size=3) == frozenset({"abc", "bcd", "cde", "def"})


def test_normalize_text_preserves_target_punctuation() -> None:
    """Verify that only context-legit punctuation is preserved during normalization."""
    # Preserved contexts (floats, degrees, percentages, timers)
    assert (
        normalize_text("set living room temperature to 27.5")
        == "set living room temperature to 27.5"
    )
    assert normalize_text("set temperature to 20,5") == "set temperature to 20,5"
    assert normalize_text("ac temperature 27°") == "ac temperature 27°"
    assert normalize_text("ac temperature 27 °") == "ac temperature 27 °"
    assert normalize_text("set brightness to 50%") == "set brightness to 50%"
    assert normalize_text("set brightness to 50 %") == "set brightness to 50 %"
    assert normalize_text("set timer for 12:30") == "set timer for 12:30"
    assert normalize_text("set timer for 10-15 minutes") == "set timer for 10-15 minutes"
    assert normalize_text("set temperature range 20°-25°") == "set temperature range 20°-25°"
    assert normalize_text("humidity between 50%-60%") == "humidity between 50%-60%"
    assert normalize_text("set temperature to -5") == "set temperature to -5"
    assert normalize_text("-27.5 degrees") == "-27.5 degrees"
    assert normalize_text("-.5 degrees") == "-.5 degrees"
    assert normalize_text("-,5 degrees") == "-,5 degrees"
    assert normalize_text("\N{MINUS SIGN}5 degrees") == "-5 degrees"
    assert normalize_text("\N{MINUS SIGN}.5 degrees") == "-.5 degrees"
    assert normalize_text("temperature (\N{MINUS SIGN}.5)") == "temperature -.5"
    assert normalize_text("minus - 5 spaced") == "minus 5 spaced"

    # Stripped contexts (non-digits or incorrect boundary)
    assert normalize_text("temperature is.") == "temperature is"
    assert normalize_text(".5 seconds") == ".5 seconds"
    assert normalize_text(",5 seconds") == ",5 seconds"
    assert normalize_text("temperature is 27. degrees") == "temperature is 27 degrees"
    assert normalize_text("degrees sign ° alone") == "degrees sign alone"
    assert normalize_text("brightness % generic") == "brightness generic"
    assert normalize_text("timer: generic") == "timer generic"
    assert normalize_text("kitchen-light") == "kitchen light"


def test_clear_normalization_caches() -> None:
    """Verify that clear_normalization_caches clears LRU cache entries."""
    _ = tokenize_normalized("test cache entry")
    cache_info_before = tokenize_normalized.cache_info()
    assert cache_info_before.currsize > 0

    clear_normalization_caches()
    cache_info_after = tokenize_normalized.cache_info()
    assert cache_info_after.currsize == 0
