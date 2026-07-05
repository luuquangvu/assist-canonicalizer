"""Shared utility helpers for Assist Canonicalizer."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Mapping
from functools import lru_cache
from math import isclose
from string import whitespace
from time import monotonic
from typing import Any

from .builtin_intents import language_variant_for
from .const import (
    CONF_MIN_CONFIDENCE,
    CONF_MIN_MARGIN,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    PRESERVED_UNIT_SUFFIXES,
)
from .normalization import normalize_text

_RANGE_EPSILON = 1e-9  # Tolerance for float comparison when validating range values.
_HALVES_FRACTIONS = (0.0, 0.5)  # Permitted fractional offsets when set to "halves".
_TENTHS_FRACTIONS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)  # Permitted fractional offsets when set to "tenths".

_STRIPPED_CHARS = "".join(PRESERVED_UNIT_SUFFIXES) + whitespace
_HAS_DIGIT_RE = re.compile(r"\d")

_CUSTOM_WILDCARD_SLOTS: dict[str, set[str]] = {}
NormalizedIntentContext = Mapping[str, frozenset[str]]


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


def intent_context_from_area_name(area_name: str | None) -> dict[str, dict[str, str]] | None:
    """Return HassIL-style intent context for a request area name."""
    if not isinstance(area_name, str) or not area_name.strip():
        return None
    return {"area": {"value": area_name, "text": area_name}}


def normalized_slot_value_tokens(value: Any) -> frozenset[str]:
    """Return normalized tokens for a scalar slot or context value."""
    if value is None:
        return frozenset()
    if isinstance(value, bool):
        return frozenset({str(value).casefold()})
    normalized = normalize_text(str(value))
    return frozenset(normalized.split()) if normalized else frozenset()


def normalize_intent_context(
    intent_context: Mapping[str, Any] | None,
) -> NormalizedIntentContext:
    """Return normalized HassIL intent context tokens keyed by slot name."""
    if not intent_context:
        return {}
    normalized: dict[str, frozenset[str]] = {}
    for key, raw_value in intent_context.items():
        if not isinstance(key, str) or not key:
            continue
        values: list[Any] = []
        if isinstance(raw_value, Mapping):
            values.extend(raw_value.get(field) for field in ("value", "text"))
        else:
            values.append(raw_value)
        if tokens := frozenset(
            token for value in values for token in normalized_slot_value_tokens(value)
        ):
            normalized[key] = tokens
    return normalized


def register_custom_wildcards_from_sources(
    language: str | None, intent_sources: Mapping[str, Mapping[str, Any]]
) -> None:
    """Extract wildcard slot names from intent sources and register them."""
    if not intent_sources:
        return
    wildcards: set[str] = set()
    for source_config in intent_sources.values():
        lists = source_config.get("lists", {})
        if isinstance(lists, Mapping):
            for name, list_config in lists.items():
                if isinstance(list_config, Mapping) and list_config.get("wildcard"):
                    wildcards.add(name)
    if wildcards:
        try:
            norm_lang = normalize_language(language) if language else ""
        except ValueError:
            norm_lang = ""
        if norm_lang not in _CUSTOM_WILDCARD_SLOTS:
            _CUSTOM_WILDCARD_SLOTS[norm_lang] = set()
        _CUSTOM_WILDCARD_SLOTS[norm_lang].update(wildcards)
        _wildcard_slot_names.cache_clear()
        wildcard_slot_names_sorted.cache_clear()


@lru_cache(maxsize=128)
def _wildcard_slot_names(language: str | None = None) -> frozenset[str]:
    """Return all known wildcard slot names for the given language from home_assistant_intents.

    Wildcard slots (``"wildcard": true`` in HassIL lists) capture free-form
    user text, they have no predefined values.  Candidate templates expand
    these slots using the literal list name (e.g. ``"shopping_list_item"``),
    and the placeholder must be rehydrated from the original query before
    delegation.

    The result is computed once and cached.
    """
    names: set[str] = set()
    if language:
        try:
            norm_lang = normalize_language(language)
        except ValueError:
            norm_lang = ""
        if norm_lang in _CUSTOM_WILDCARD_SLOTS:
            names.update(_CUSTOM_WILDCARD_SLOTS[norm_lang])
    else:
        for custom_set in _CUSTOM_WILDCARD_SLOTS.values():
            names.update(custom_set)

    with contextlib.suppress(Exception):
        import home_assistant_intents

        languages = [language] if language else home_assistant_intents.get_languages()
        for lang in languages:
            try:
                data = home_assistant_intents.get_intents(lang)
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


class _WildcardSlotNamesWrapper:
    """Wrapper for wildcard_slot_names to support dynamic cache clearing."""

    def __init__(self, func: Any) -> None:
        """Initialize the wrapper with the cached function."""
        self._func = func

    def __call__(self, language: str | None = None) -> frozenset[str]:
        """Return all known wildcard slot names for the given language."""
        return self._func(language)

    def cache_clear(self, language: str | None = None) -> None:
        """Clear custom wildcards and delegation caches.

        Passing a language removes only that language's custom wildcard
        registrations while still clearing the shared LRU caches.
        """
        if language is None:
            _CUSTOM_WILDCARD_SLOTS.clear()
        else:
            try:
                norm_lang = normalize_language(language)
            except ValueError:
                norm_lang = ""
            _CUSTOM_WILDCARD_SLOTS.pop(norm_lang, None)
        self._func.cache_clear()
        wildcard_slot_names_sorted.cache_clear()


wildcard_slot_names = _WildcardSlotNamesWrapper(_wildcard_slot_names)


@lru_cache(maxsize=128)
def wildcard_slot_names_sorted(language: str | None = None) -> tuple[str, ...]:
    """Return all known wildcard slot names sorted by length descending (cached)."""
    names = wildcard_slot_names(language)
    return tuple(sorted(names, key=len, reverse=True))


@lru_cache(maxsize=1024)
def parse_float(val_str: str) -> float | None:
    """Parse a string to float with a fast character check to avoid exceptions.

    This function aligns with the placeholder-based normalization in `normalization.py`.
    It strips trailing unit suffixes (e.g. '°' or '%') using `_STRIPPED_CHARS` (which
    inherits from `PRESERVED_UNIT_SUFFIXES`) and handles both comma and dot decimal
    separators by normalizing them to standard Python floats (replacing ',' with '.').
    """
    if not isinstance(val_str, str):
        return None
    if _HAS_DIGIT_RE.search(val_str) is None:
        return None
    val_str = val_str.strip().rstrip(_STRIPPED_CHARS)
    if not val_str:
        return None
    with contextlib.suppress(ValueError):
        return float(val_str.replace(",", "."))


def is_valid_range_value(
    val: float,
    from_val: float,
    step: float | None,
    fraction_type: str | None,
) -> bool:
    """Return True if a float value satisfies range step and fraction constraints.

    A value is valid if it lies on a grid defined by:
        val = from_val + (k * step) + fraction_offset
    where k is an integer, and fraction_offset is one of the allowed offsets
    based on the fraction_type (e.g. halves or tenths).

    Note: When `step` is `None` (omitted in intent configuration), it defaults to a step
    size of 1.0. This enforces an integer grid relative to `from_val` (e.g. accepting 7
    but rejecting 7.3), strictly aligning with HassIL range list specification matching logic.

    Tolerance check uses absolute tolerance (`_RANGE_EPSILON` = 1e-9) to determine
    closeness to the grid.
    """
    if step is not None and isclose(step, 0.0):
        return False
    s = step if step is not None else 1.0
    if fraction_type == "halves":
        fractions = _HALVES_FRACTIONS
    elif fraction_type == "tenths":
        fractions = _TENTHS_FRACTIONS
    else:
        fractions = (0.0,)

    # Check if the remaining difference after subtracting each fraction offset
    # is a multiple of the step size (i.e. diff / step is close to an integer).
    for f in fractions:
        diff = val - f - from_val
        if isclose(diff / s, round(diff / s), abs_tol=_RANGE_EPSILON):
            return True
    return False


def to_output_value(val_float: float) -> int | float:
    """Convert float value to int if it represents an integer, else return float."""
    return int(val_float) if val_float.is_integer() else val_float
