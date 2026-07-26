"""Language-agnostic text normalization helpers."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from .const import (
    GENERIC_LATIN_REPLACEMENTS,
    LANGUAGE_SPECIFIC_OVERRIDES,
    PRESERVED_UNIT_SUFFIXES,
)

_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_HAS_DIGIT_RE = re.compile(r"\d")
_NUMERIC_UNICODE_MINUS_RE = re.compile(
    r"(?<!\w)[\u2010\u2011\u2012\u2013\u2212\ufe63\uff0d](?=(?:\d|[.,]\d))"
)
_WORD_JOIN_CONTROLS = frozenset(("\u200c", "\u200d"))

# Frozenset of source characters in GENERIC_LATIN_REPLACEMENTS for fast
# membership testing via frozenset.isdisjoint (C-level, single-pass).
_GENERIC_LATIN_CHARS: frozenset[str] = frozenset(GENERIC_LATIN_REPLACEMENTS)


def _is_variation_selector(char: str) -> bool:
    """Return whether a mark only selects a visual glyph presentation."""
    codepoint = ord(char)
    return 0xFE00 <= codepoint <= 0xFE0F or 0xE0100 <= codepoint <= 0xE01EF


def _mark_attachment_flags(text: str) -> list[bool]:
    """Precompute whether each mark position is attached to a word-char base."""
    flags = [False] * len(text)
    last_is_word_base = False
    for i, char in enumerate(text):
        if char in _WORD_JOIN_CONTROLS:
            continue
        if unicodedata.category(char).startswith("M"):
            flags[i] = last_is_word_base
            continue
        last_is_word_base = char == "_" or char.isalnum()
    return flags


def _strip_non_word_characters(text: str) -> str:
    """Preserve attached script marks while replacing punctuation with spaces."""
    if text.isascii():
        # ASCII text contains no combining marks, join controls, or variation
        # selectors, so every non-word character simply becomes a space.
        return _NON_WORD_RE.sub(" ", text)
    flags = _mark_attachment_flags(text)

    def _replace(match: re.Match[str]) -> str:
        char = match.group(0)
        if char in _WORD_JOIN_CONTROLS:
            return ""
        if (
            not unicodedata.category(char).startswith("M")
            or _is_variation_selector(char)
            or char == "\u20e3"
        ):
            return " "
        return char if flags[match.start()] else ""

    return _NON_WORD_RE.sub(_replace, text)


def _strip_latin_diacritics(text: str) -> str:
    """Strip marks attached to Latin letters without rewriting other scripts."""
    decomposed = unicodedata.normalize("NFD", text)
    result: list[str] = []
    previous_base_is_latin = False
    removed_mark = False
    for char in decomposed:
        if unicodedata.category(char).startswith("M"):
            if previous_base_is_latin:
                removed_mark = True
                continue
            result.append(char)
            continue
        result.append(char)
        previous_base_is_latin = unicodedata.category(char).startswith("L") and unicodedata.name(
            char, ""
        ).startswith("LATIN")
    return unicodedata.normalize("NFC", "".join(result)) if removed_mark else text


def _make_placeholder(name: str) -> str:
    """Generate a highly unique placeholder to temporarily represent preserved punctuation.

    The prefix '__ASSIST_CANONICALIZER_' and suffix '__' ensure that these internal
    placeholders do not collide with any valid user input sentences.
    """
    return f"__ASSIST_CANONICALIZER_{name}__"


# Regex patterns to find and temporarily replace punctuation in their proper contexts:
# - Decimal/comma floats: 27.5, 20,5
# - Temperature degrees: 27° or 27 °
# - Percentage: 50% or 50 %
# - Timer colon/hyphen: 12:30, 10-15
class _PunctuationPlaceholderManager:
    """Manages context-aware punctuation placeholders during normalization.

    Translates special characters (e.g. decimal points, suffixes, colons, hyphens) to
    unique placeholders before punctuation stripping, and restores them afterwards.

    This ensures that:
    1. Decimals (both dots '.' and commas ',') are preserved in numbers so they can be
       parsed by `parse_float` during range-slot resolution.
    2. Suffixes (defined in `PRESERVED_UNIT_SUFFIXES` in `const.py`, currently '°' and '%')
       remain attached to numbers during normalization to avoid being stripped as normal
       punctuation, maintaining alignment during lexical matching.
    3. They are subsequently stripped inside `parse_float` when converting tokens to actual
       floats for range check validation.
    """

    def __init__(self) -> None:
        """Initialize the pre-compiled regex patterns for punctuation placeholders."""
        float_dot_placeholder = _make_placeholder("F_DOT")
        float_comma_placeholder = _make_placeholder("F_COMMA")
        numeric_minus_placeholder = _make_placeholder("N_MINUS")
        self.placeholders: list[tuple[re.Pattern[str], str, str, str]] = [
            (
                re.compile(r"(?<!\w)\.(?=\d)|(?<=\d)\.(?=\d)"),
                float_dot_placeholder,
                float_dot_placeholder,
                ".",
            ),
            (
                re.compile(r"(?<!\w),(?=\d)|(?<=\d),(?=\d)"),
                float_comma_placeholder,
                float_comma_placeholder,
                ",",
            ),
            (
                re.compile(r"(?<=\d):(?=\d)"),
                _make_placeholder("T_COLON"),
                _make_placeholder("T_COLON"),
                ":",
            ),
            (
                re.compile(r"(?<=\d)-(?=\d)"),
                _make_placeholder("T_HYPHEN"),
                _make_placeholder("T_HYPHEN"),
                "-",
            ),
            (
                re.compile(
                    rf"(?<!\w)-(?=(?:\d|{re.escape(float_dot_placeholder)}\d|"
                    rf"{re.escape(float_comma_placeholder)}\d))"
                ),
                numeric_minus_placeholder,
                numeric_minus_placeholder,
                "-",
            ),
        ]
        for _char, _name in PRESERVED_UNIT_SUFFIXES.items():
            _pl = _make_placeholder(_name)
            self.placeholders.append(
                (re.compile(r"(?<=\d)(\s*)" + re.escape(_char)), r"\1" + _pl, _pl, _char)
            )

    def apply(self, text: str) -> str:
        """Apply temporary placeholders to the text before stripping punctuation."""
        for pattern, replacement, _, _ in self.placeholders:
            text = pattern.sub(replacement, text)
        return text

    def restore(self, text: str) -> str:
        """Restore original punctuation characters from their temporary placeholders."""
        for _, _, placeholder, restore_val in self.placeholders:
            text = text.replace(placeholder, restore_val)
        return text


_PLACEHOLDER_MANAGER = _PunctuationPlaceholderManager()


@lru_cache(maxsize=65536)
def normalize_text(text: str) -> str:
    """Normalize text without applying language-specific rules.

    We preserve numeric punctuation (decimals, degrees, percentages, signs) using
    placeholders before stripping normal punctuation. This ensures numeric values
    in canonicalized templates correctly match numeric values in input queries.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    if _HAS_DIGIT_RE.search(normalized) is not None:
        normalized = _NUMERIC_UNICODE_MINUS_RE.sub("-", normalized)
        with_placeholders = _PLACEHOLDER_MANAGER.apply(normalized)
        without_punctuation = _strip_non_word_characters(with_placeholders)
        collapsed = _WHITESPACE_RE.sub(" ", without_punctuation).strip()
        return _PLACEHOLDER_MANAGER.restore(collapsed)

    without_punctuation = _strip_non_word_characters(normalized)
    return _WHITESPACE_RE.sub(" ", without_punctuation).strip()


def normalize_text_no_diacritics(text: str, language: str | None = None) -> str:
    """Normalize text and strip diacritics/accents for multi-language compatibility.

    The normalization pipeline executes in the following strict order:
    1. Base Unicode NFKC normalization, lowercasing (casefold), punctuation
       removal, and whitespace collapse (via `normalize_text`).
    2. Language-specific overrides (from `LANGUAGE_SPECIFIC_OVERRIDES` in const.py).
    3. Generic Latin replacements (from `GENERIC_LATIN_REPLACEMENTS` in const.py).
    4. Accent/diacritic stripping from Latin letters only. Marks that form
       letters in other scripts are retained to avoid changing word identity.
    """
    return normalize_text_no_diacritics_from_normalized(normalize_text(text), language)


@lru_cache(maxsize=65536)
def normalize_text_no_diacritics_from_normalized(
    normalized: str, language: str | None = None
) -> str:
    """Strip diacritics/accents from already-normalized text."""
    if not normalized:
        return ""

    if normalized.isascii():
        return normalized

    if language:
        lang_code = language.split("-")[0].lower()
        if lang_code in LANGUAGE_SPECIFIC_OVERRIDES:
            for source, target in LANGUAGE_SPECIFIC_OVERRIDES[lang_code].items():
                normalized = normalized.replace(source, target)

    if not _GENERIC_LATIN_CHARS.isdisjoint(normalized):
        for source, target in GENERIC_LATIN_REPLACEMENTS.items():
            normalized = normalized.replace(source, target)

    return _strip_latin_diacritics(normalized)


def tokenize_text(text: str) -> tuple[str, ...]:
    """Return whitespace tokens from normalized text."""
    normalized = normalize_text(text)
    return tuple(normalized.split()) if normalized else ()


@lru_cache(maxsize=65536)
def tokenize_normalized(text: str) -> tuple[str, ...]:
    """Return whitespace tokens from already-normalized text."""
    return tuple(text.split()) if text else ()


@lru_cache(maxsize=65536)
def char_ngrams_normalized(text: str, size: int = 3) -> frozenset[str]:
    """Return character n-grams from already-normalized text."""
    if size < 1:
        raise ValueError("N-gram size must be positive")
    if compact := text.replace(" ", ""):
        return (
            frozenset({compact})
            if len(compact) <= size
            else frozenset(
                compact[index : index + size] for index in range(len(compact) - size + 1)
            )
        )
    return frozenset()


@lru_cache(maxsize=2048)
def literal_token_variants(literal_text: str) -> tuple[frozenset[str], ...]:
    """Return normalized literal token variants for intent action scoring."""
    variants = []
    for variant in literal_text.split("|"):
        if not variant.strip():
            continue
        if literal_tokens := frozenset(normalize_text(variant).split()):
            variants.append(literal_tokens)
    return tuple(variants)


@lru_cache(maxsize=2048)
def literal_tokens_list(literal_text: str) -> tuple[str, ...]:
    """Return all normalized tokens inside a literal text template."""
    return tuple(normalize_text(literal_text).split())


def clear_normalization_caches() -> None:
    """Clear all global LRU caches in normalization module."""
    normalize_text.cache_clear()
    normalize_text_no_diacritics_from_normalized.cache_clear()
    tokenize_normalized.cache_clear()
    char_ngrams_normalized.cache_clear()
    literal_token_variants.cache_clear()
    literal_tokens_list.cache_clear()
