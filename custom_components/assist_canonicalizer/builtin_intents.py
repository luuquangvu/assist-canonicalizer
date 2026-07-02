"""Load Home Assistant built-in and custom sentence configs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import orjson


def load_language_intent_sources(
    language: str,
    *,
    config_path: Callable[..., str] | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Load built-in and custom sentence sources for a language."""
    language_variant = language_variant_for(language)
    if language_variant is None:
        return {}

    sources: dict[str, Mapping[str, Any]] = {}
    if built_in := _load_built_in_intents(language_variant):
        sources["built_in"] = built_in

    if custom := _load_custom_sentences(language_variant, config_path):
        sources["custom_sentence"] = custom

    return sources


def language_variant_for(language: str) -> str | None:
    """Return the Home Assistant intents language variant for a language."""
    if not language.strip():
        return None
    try:
        import home_assistant_intents as intents_module
        from homeassistant.util import language as language_module
    except ImportError:
        return language

    get_languages = intents_module.get_languages
    matches = language_module.matches(language, set(get_languages()))
    return matches[0] if matches else None


def _load_built_in_intents(language_variant: str) -> Mapping[str, Any]:
    """Load built-in Home Assistant intents for a language variant."""
    try:
        import home_assistant_intents as intents_module
    except ImportError:
        return {}

    get_intents = intents_module.get_intents
    try:
        intents = get_intents(language_variant, json_load=_json_load)
    except TypeError:
        intents = get_intents(language_variant)
    return intents if isinstance(intents, Mapping) else {}


def _load_custom_sentences(
    language_variant: str,
    config_path: Callable[..., str] | None,
) -> Mapping[str, Any]:
    """Load custom sentence YAML files for a language variant."""
    if config_path is None:
        return {}

    try:
        import yaml as yaml_module
    except ImportError:
        return {}

    custom_dir = Path(config_path("custom_sentences", language_variant))
    if not custom_dir.is_dir():
        return {}

    merged: dict[str, Any] = {}
    for sentence_file in sorted(custom_dir.rglob("*.yaml")):
        with sentence_file.open(encoding="utf-8") as file_handle:
            loaded = yaml_module.safe_load(file_handle)
        if isinstance(loaded, Mapping):
            _merge_dict(merged, loaded)
    return merged


def _json_load(file_handle: Any) -> dict[str, Any]:
    """Load a JSON object from a file handle."""
    loaded = orjson.loads(file_handle.read())
    return loaded if isinstance(loaded, dict) else {}


def _merge_dict(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Recursively merge a mapping into a target dictionary."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, Mapping):
            _merge_dict(target[key], value)
        else:
            target[key] = value
