"""Shared utility helpers for Assist Canonicalizer."""

from __future__ import annotations

import contextlib
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
    if not isinstance(language, str):
        raise ValueError("Language must be a non-empty string")
    requested = language.strip()
    if not requested:
        raise ValueError("Language must not be empty")
    language_variant = language_variant_for(requested)
    if language_variant is not None:
        return language_variant
    return requested.replace("_", "-").lower()


@lru_cache(maxsize=128)
def wildcard_slot_names(language: str | None = None) -> frozenset[str]:
    r"""Return all known wildcard slot names for the given language from home_assistant_intents.

    Wildcard slots (``"wildcard": true`` in HassIL lists) capture free-form
    user text, they have no predefined values.  Candidate templates expand
    these slots using the literal list name (e.g. ``"shopping_list_item"``),
    and the placeholder must be rehydrated from the original query before
    delegation.

    The result is computed once and cached.
    """
    names: set[str] = set()
    try:
        import home_assistant_intents as intents_module
    except ImportError:
        return frozenset()
    with contextlib.suppress(Exception):
        languages = [language] if language else intents_module.get_languages()
        for lang in languages:
            try:
                data = intents_module.get_intents(lang)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            lists = data.get("lists", {})
            if not isinstance(lists, dict):
                continue
            for name, config in lists.items():
                if isinstance(config, dict) and config.get("wildcard"):
                    names.add(name)
    return frozenset(names)


@lru_cache(maxsize=128)
def wildcard_slot_names_sorted(language: str | None = None) -> tuple[str, ...]:
    """Return all known wildcard slot names sorted by length descending (cached)."""
    names = wildcard_slot_names(language)
    return tuple(sorted(names, key=len, reverse=True))
