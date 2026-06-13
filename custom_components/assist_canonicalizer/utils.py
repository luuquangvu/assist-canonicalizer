"""Shared utility helpers for Assist Canonicalizer."""

from __future__ import annotations

from functools import lru_cache
from time import monotonic
from typing import Any

from .builtin_intents import language_variant_for
from .const import CONF_MIN_CONFIDENCE, CONF_MIN_MARGIN, DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_MARGIN


def elapsed_ms(started_at: float) -> float:
    """Return elapsed milliseconds from a monotonic timestamp."""
    return round((monotonic() - started_at) * 1000, 3)


def resolve_entry_thresholds(entry: Any) -> tuple[float, float]:
    """Return (min_confidence, min_margin) from entry options with data fallback."""
    options = (getattr(entry, "options", {}) or {}) if entry is not None else {}
    data = (getattr(entry, "data", {}) or {}) if entry is not None else {}
    min_confidence = options.get(
        CONF_MIN_CONFIDENCE,
        data.get(CONF_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE),
    )
    min_margin = options.get(
        CONF_MIN_MARGIN,
        data.get(CONF_MIN_MARGIN, DEFAULT_MIN_MARGIN),
    )
    return min_confidence, min_margin


@lru_cache(maxsize=128)
def normalize_language(language: str) -> str:
    """Return the Home Assistant language variant as a canonical cache key."""
    requested = str(language).strip()
    if not requested:
        raise ValueError("Language must not be empty")
    language_variant = language_variant_for(requested)
    if language_variant is not None:
        return language_variant
    return requested.replace("_", "-").lower()
