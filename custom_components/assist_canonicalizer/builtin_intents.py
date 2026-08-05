"""Load Home Assistant built-in and custom sentence configs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import orjson
import yaml
from hassil.util import merge_dict
from homeassistant.util import language as language_module

type IntentSource = Mapping[str, object]


class TextReader(Protocol):
    """Readable text stream accepted by the intents JSON loader."""

    def read(self) -> str | bytes:
        """Read the remaining text."""
        ...


def load_language_intent_sources(
    language: str,
    *,
    config_path: Callable[..., str] | None = None,
) -> dict[str, IntentSource]:
    """Load built-in and custom sentence sources for a language."""
    language_variant = language_variant_for(language)
    if language_variant is None:
        return {}

    built_in = _load_built_in_intents(language_variant)
    custom = _load_custom_sentences(language_variant, config_path)

    sources: dict[str, IntentSource] = {}
    if built_in and custom:
        effective = deepcopy(dict(built_in))
        merge_dict(effective, custom)
        sources["built_in"] = _project_intent_source(effective, built_in)
        sources["custom_sentence"] = _project_intent_source(effective, custom)
    else:
        if built_in:
            sources["built_in"] = built_in
        if custom:
            sources["custom_sentence"] = custom

    return sources


@lru_cache(maxsize=128)
def language_variant_for(language: str) -> str | None:
    """Return the Home Assistant intents language variant for a language.

    The result depends only on the installed intents package, which cannot
    change without a restart, so caching is safe.
    """
    if not language.strip():
        return None

    try:
        import home_assistant_intents
    except ImportError:
        return language

    get_languages = home_assistant_intents.get_languages
    supported_languages = sorted(get_languages())
    matches = language_module.matches(language, supported_languages)
    return matches[0] if matches else None


@lru_cache(maxsize=128)
def _load_built_in_intents(language_variant: str) -> dict[str, object]:
    """Load built-in Home Assistant intents for a language variant."""
    try:
        import home_assistant_intents
    except ImportError:
        return {}

    get_intents = home_assistant_intents.get_intents
    try:
        intents = get_intents(language_variant, json_load=_json_load)
    except TypeError:
        intents = get_intents(language_variant)
    return dict(intents) if isinstance(intents, Mapping) else {}


def clear_builtin_intents_caches() -> None:
    """Clear cached built-in intent sources."""
    _load_built_in_intents.cache_clear()
    language_variant_for.cache_clear()


def _load_custom_sentences(
    language_variant: str,
    config_path: Callable[..., str] | None,
) -> dict[str, object]:
    """Load custom sentence YAML files for a language variant."""
    if config_path is None:
        return {}

    custom_dir = Path(config_path("custom_sentences", language_variant))
    if not custom_dir.is_dir():
        return {}

    merged: dict[str, object] = {}
    for sentence_file in sorted(custom_dir.rglob("*.yaml")):
        with sentence_file.open(encoding="utf-8") as file_handle:
            loaded = yaml.safe_load(file_handle)
        if isinstance(loaded, Mapping):
            merge_dict(merged, loaded)
    return merged


def _json_load(file_handle: TextReader) -> dict[str, object]:
    """Load a JSON object from a file handle."""
    loaded = orjson.loads(file_handle.read())
    if not isinstance(loaded, dict):
        return {}
    return {key: value for key, value in loaded.items() if isinstance(key, str)}


def _project_intent_source(
    effective: Mapping[str, object],
    source: Mapping[str, object],
) -> dict[str, object]:
    """Return effective grammar context with only one source's intent data."""
    projected = deepcopy(dict(effective))
    effective_intents = effective.get("intents", {})
    source_intents = source.get("intents", {})
    if not isinstance(effective_intents, Mapping) or not isinstance(source_intents, Mapping):
        projected["intents"] = deepcopy(source_intents)
        return projected

    projected_intents: dict[str, object] = {}
    for intent_name, source_intent in source_intents.items():
        if not isinstance(intent_name, str):
            continue
        effective_intent = effective_intents.get(intent_name, source_intent)
        if not isinstance(source_intent, Mapping) or not isinstance(effective_intent, Mapping):
            projected_intents[intent_name] = deepcopy(source_intent)
            continue
        projected_intent = deepcopy(dict(effective_intent))
        if "data" in source_intent:
            projected_intent["data"] = deepcopy(source_intent.get("data"))
        else:
            projected_intent.pop("data", None)
        projected_intents[intent_name] = projected_intent
    projected["intents"] = projected_intents
    return projected
