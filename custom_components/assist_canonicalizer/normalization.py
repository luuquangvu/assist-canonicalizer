"""Language-agnostic text normalization helpers."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from .const import GENERIC_LATIN_REPLACEMENTS, LANGUAGE_SPECIFIC_OVERRIDES

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

# Precomputed BMP translation table that deletes all Unicode combining marks.
# Built once at import — covers all diacritics used by DE, EN, FR, NL, VI and
# avoids per-character Python overhead in normalize_text_no_diacritics.
_COMBINING_TABLE: dict[int, None] = {
    cp: None for cp in range(0x10000) if unicodedata.combining(chr(cp))
}

# Frozenset of source characters in GENERIC_LATIN_REPLACEMENTS for fast
# membership testing via frozenset.isdisjoint (C-level, single-pass).
_GENERIC_LATIN_CHARS: frozenset[str] = frozenset(GENERIC_LATIN_REPLACEMENTS)


def normalize_text(text: str) -> str:
    """Normalize text without applying language-specific rules."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    without_punctuation = _PUNCTUATION_RE.sub(" ", normalized)
    return _WHITESPACE_RE.sub(" ", without_punctuation).strip()


def normalize_text_no_diacritics(text: str, language: str | None = None) -> str:
    """Normalize text and strip diacritics/accents for multi-language compatibility.

    The normalization pipeline executes in the following strict order:
    1. Base Unicode NFKC normalization, lowercasing (casefold), punctuation
       removal, and whitespace collapse (via `normalize_text`).
    2. Language-specific overrides (from `LANGUAGE_SPECIFIC_OVERRIDES` in const.py).
    3. Generic Latin replacements (from `GENERIC_LATIN_REPLACEMENTS` in const.py).
    4. Accent/diacritic stripping (via Unicode NFD decomposition followed by
       removing combining diacritics/accents).
    """
    normalized = normalize_text(text)
    if not normalized:
        return ""

    if language:
        lang_code = language.split("-")[0].lower()
        if lang_code in LANGUAGE_SPECIFIC_OVERRIDES:
            for source, target in LANGUAGE_SPECIFIC_OVERRIDES[lang_code].items():
                normalized = normalized.replace(source, target)

    if not _GENERIC_LATIN_CHARS.isdisjoint(normalized):
        for source, target in GENERIC_LATIN_REPLACEMENTS.items():
            normalized = normalized.replace(source, target)

    nfd_form = unicodedata.normalize("NFD", normalized)
    return nfd_form.translate(_COMBINING_TABLE)


def tokenize_text(text: str) -> tuple[str, ...]:
    """Return whitespace tokens from normalized text."""
    normalized = normalize_text(text)
    if not normalized:
        return ()
    return tuple(normalized.split())


def tokenize_normalized(text: str) -> tuple[str, ...]:
    """Return whitespace tokens from already-normalized text."""
    if not text:
        return ()
    return tuple(text.split())


def char_ngrams_normalized(text: str, size: int = 3) -> frozenset[str]:
    """Return character n-grams from already-normalized text."""
    if size < 1:
        raise ValueError("N-gram size must be positive")
    compact = text.replace(" ", "")
    if not compact:
        return frozenset()
    if len(compact) <= size:
        return frozenset({compact})
    return frozenset(compact[index : index + size] for index in range(len(compact) - size + 1))


@lru_cache(maxsize=8192)
def literal_token_variants(literal_text: str) -> tuple[frozenset[str], ...]:
    """Return normalized literal token variants for intent action scoring."""
    variants = []
    for variant in literal_text.split("|"):
        if not variant.strip():
            continue
        literal_tokens = frozenset(normalize_text(variant).split())
        if literal_tokens:
            variants.append(literal_tokens)
    return tuple(variants)
