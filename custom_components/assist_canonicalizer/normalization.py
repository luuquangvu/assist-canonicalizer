"""Language-agnostic text normalization helpers."""

from __future__ import annotations

import re
import unicodedata

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text without applying language-specific rules."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    without_punctuation = _PUNCTUATION_RE.sub(" ", normalized)
    return _WHITESPACE_RE.sub(" ", without_punctuation).strip()


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


def char_ngrams(text: str, size: int = 3) -> frozenset[str]:
    """Return character n-grams for normalized text."""
    if size < 1:
        raise ValueError("N-gram size must be positive")
    compact = normalize_text(text).replace(" ", "")
    if not compact:
        return frozenset()
    if len(compact) <= size:
        return frozenset({compact})
    return frozenset(compact[index : index + size] for index in range(len(compact) - size + 1))


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
