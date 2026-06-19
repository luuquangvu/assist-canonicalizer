"""Home Assistant service handlers for Assist Canonicalizer."""

from __future__ import annotations

import time
from typing import Any

import voluptuous as vol
from homeassistant.core import ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv

from .const import (
    ATTR_ACCEPTED,
    ATTR_CANDIDATE_COUNT,
    ATTR_INTENT_NAME,
    ATTR_LANGUAGE,
    ATTR_NORMALIZED_TEXT,
    ATTR_SELECTED_CANDIDATE,
    ATTR_SOURCE,
    ATTR_TEXT,
    ATTR_TOP_CANDIDATES,
    DATA_RUNTIME,
    DEFAULT_MAX_DYNAMIC_CANDIDATES,
    DEFAULT_MAX_DYNAMIC_SLOT_VALUES,
    DOMAIN,
    SERVICE_CLEAR_INDEX,
    SERVICE_DIAGNOSTICS,
    SERVICE_DUMP_CANDIDATES,
    SERVICE_REBUILD_INDEX,
    SERVICE_TEST_MATCH,
)
from .grammar_loader import rehydrate_wildcard_text
from .indexer import CanonicalIndex
from .normalization import normalize_text
from .ranking import RankedCandidate, accepted_candidate
from .runtime import CanonicalizerRuntime
from .utils import elapsed_ms, normalize_language, resolve_entry_thresholds

ATTR_REBUILD = "rebuild"


def validate_supported_language(value: Any) -> str:
    """Validate that the language is supported by Home Assistant.

    Retrieves the list of supported languages dynamically from home_assistant_intents.
    """
    lang = cv.string(value)
    if not lang.strip():
        raise vol.Invalid("Language cannot be empty")
    try:
        import home_assistant_intents as intents_module
        from homeassistant.util import language as language_module
    except ImportError:
        return lang

    get_languages = intents_module.get_languages
    matches = language_module.matches(lang, set(get_languages()))
    if not matches:
        raise vol.Invalid(f"Language '{lang}' is not supported by Home Assistant")
    return matches[0]


TEST_MATCH_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_TEXT): cv.string,
        vol.Optional(ATTR_LANGUAGE): validate_supported_language,
    }
)

REBUILD_INDEX_SCHEMA = vol.Schema({vol.Optional(ATTR_LANGUAGE): validate_supported_language})
CLEAR_INDEX_SCHEMA = vol.Schema({vol.Optional(ATTR_LANGUAGE): validate_supported_language})
DIAGNOSTICS_SCHEMA = vol.Schema({})
DUMP_CANDIDATES_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_LANGUAGE): validate_supported_language,
        vol.Optional(ATTR_REBUILD, default=False): cv.boolean,
    }
)


def async_setup_services(hass: Any) -> None:
    """Register Assist Canonicalizer services."""

    async def handle_test_match(call: ServiceCall) -> dict[str, Any]:
        """Handle test_match service calls."""
        return await _handle_test_match(hass, call)

    async def handle_rebuild_index(call: ServiceCall) -> dict[str, Any]:
        """Handle rebuild_index service calls."""
        return await _handle_rebuild_index(hass, call)

    async def handle_dump_candidates(call: ServiceCall) -> dict[str, Any]:
        """Handle dump_candidates service calls."""
        return await _handle_dump_candidates(hass, call)

    async def handle_clear_index(call: ServiceCall) -> dict[str, Any]:
        """Handle clear_index service calls."""
        return await _handle_clear_index(hass, call)

    hass.services.async_register(
        DOMAIN,
        SERVICE_TEST_MATCH,
        handle_test_match,
        schema=TEST_MATCH_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_REBUILD_INDEX,
        handle_rebuild_index,
        schema=REBUILD_INDEX_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_CLEAR_INDEX,
        handle_clear_index,
        schema=CLEAR_INDEX_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_DIAGNOSTICS,
        lambda call: _handle_diagnostics(hass, call),
        schema=DIAGNOSTICS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_DUMP_CANDIDATES,
        handle_dump_candidates,
        schema=DUMP_CANDIDATES_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )


def async_unload_services(hass: Any) -> None:
    """Remove Assist Canonicalizer services."""
    hass.services.async_remove(DOMAIN, SERVICE_TEST_MATCH)
    hass.services.async_remove(DOMAIN, SERVICE_REBUILD_INDEX)
    hass.services.async_remove(DOMAIN, SERVICE_CLEAR_INDEX)
    hass.services.async_remove(DOMAIN, SERVICE_DIAGNOSTICS)
    hass.services.async_remove(DOMAIN, SERVICE_DUMP_CANDIDATES)


async def _handle_test_match(hass: Any, call: ServiceCall) -> dict[str, Any]:
    """Return ranked candidates for a text input with lexical scoring and custom thresholds."""
    runtime = _runtime_from_hass(hass)
    language = _service_language(hass, call)
    text = call.data[ATTR_TEXT]
    index = await _index_for_language(hass, runtime, language, rebuild_if_missing=True)
    if index is None:
        raise HomeAssistantError("Assist Canonicalizer index could not be built")
    ranked = await hass.async_add_executor_job(
        runtime.rank_with_dynamic_candidates,
        language,
        index,
        text,
    )
    domain_data = hass.data.get(DOMAIN, {})
    entry = None
    for entry_data in domain_data.values():
        if entry_data.get(DATA_RUNTIME) is runtime:
            entry = entry_data.get("entry")
            break

    min_confidence, min_margin = (
        resolve_entry_thresholds(entry) if entry else resolve_entry_thresholds(None)
    )

    selected = accepted_candidate(
        ranked,
        min_confidence=min_confidence,
        min_margin=min_margin,
    )
    return {
        ATTR_LANGUAGE: language,
        ATTR_NORMALIZED_TEXT: normalize_text(text),
        ATTR_CANDIDATE_COUNT: index.candidate_count,
        "index_cached": True,
        "dynamic_candidate_count": runtime.diagnostics.dynamic_candidate_count,
        ATTR_ACCEPTED: selected is not None,
        ATTR_SELECTED_CANDIDATE: (
            _ranked_candidate_response(selected, query=text) if selected else None
        ),
        ATTR_TOP_CANDIDATES: [_ranked_candidate_response(item, query=text) for item in ranked],
    }


async def _handle_rebuild_index(hass: Any, call: ServiceCall) -> dict[str, Any]:
    """Rebuild one language index from automatic candidate sources."""
    runtime = _runtime_from_hass(hass)
    language = _service_language(hass, call)
    started_at = time.monotonic()
    index = await _rebuild_index(hass, runtime, language)
    return {
        ATTR_LANGUAGE: language,
        ATTR_CANDIDATE_COUNT: index.candidate_count,
        "rebuild_latency_ms": elapsed_ms(started_at),
        "index_cached": True,
    }


async def _handle_clear_index(hass: Any, call: ServiceCall) -> dict[str, Any]:
    """Clear one language index or all indexes."""
    runtime = _runtime_from_hass(hass)
    language = call.data.get(ATTR_LANGUAGE)
    await runtime.async_clear_index(
        hass,
        normalize_language(language) if isinstance(language, str) else None,
    )
    return {ATTR_CANDIDATE_COUNT: runtime.total_candidate_count()}


def _handle_diagnostics(hass: Any, call: ServiceCall) -> dict[str, Any]:
    """Return runtime diagnostics."""
    runtime = _runtime_from_hass(hass)
    diagnostics = runtime.diagnostics.as_dict()
    diagnostics.update(
        {
            "cached_languages": sorted(runtime.indexes),
            "cached_candidate_counts": {
                language: index.candidate_count
                for language, index in sorted(runtime.indexes.items())
            },
            "pending_rebuild_languages": sorted(runtime.rebuild_tasks),
            "registry_slot_counts": _registry_slot_counts(runtime),
            "dynamic_candidate_generation": {
                "enabled": True,
                "max_slot_values_per_slot": DEFAULT_MAX_DYNAMIC_SLOT_VALUES,
                "max_candidates_per_query": DEFAULT_MAX_DYNAMIC_CANDIDATES,
            },
            "subscribed_intent_source_counts": runtime.subscribed_source_counts(),
        }
    )
    return diagnostics


async def _handle_dump_candidates(hass: Any, call: ServiceCall) -> dict[str, Any]:
    """Return candidate pool details for a language."""
    runtime = _runtime_from_hass(hass)
    language = _service_language(hass, call)
    should_rebuild = bool(call.data.get(ATTR_REBUILD, False))
    started_at = time.monotonic()
    index = await _index_for_language(
        hass,
        runtime,
        language,
        rebuild_if_missing=should_rebuild,
    )
    if index is None:
        intent_source_counts = await hass.async_add_executor_job(runtime.source_counts, language)
        return {
            ATTR_LANGUAGE: language,
            ATTR_CANDIDATE_COUNT: 0,
            "index_cached": False,
            "rebuild_required": True,
            "intent_source_counts": intent_source_counts,
            "candidate_source_counts": {},
            "intent_counts": {},
            "registry_slot_counts": _registry_slot_counts(runtime),
            "sample_candidates": [],
        }

    source_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    for candidate in index.candidates:
        source_counts[candidate.source.value] = source_counts.get(candidate.source.value, 0) + 1
        intent_counts[candidate.intent_name] = intent_counts.get(candidate.intent_name, 0) + 1
    intent_source_counts = await hass.async_add_executor_job(runtime.source_counts, language)
    return {
        ATTR_LANGUAGE: language,
        ATTR_CANDIDATE_COUNT: index.candidate_count,
        "index_cached": True,
        "rebuild_latency_ms": elapsed_ms(started_at) if should_rebuild else None,
        "intent_source_counts": intent_source_counts,
        "candidate_source_counts": source_counts,
        "intent_counts": dict(sorted(intent_counts.items())),
        "registry_slot_counts": _registry_slot_counts(runtime),
        "sample_candidates": [
            _ranked_candidate_candidate_response(candidate) for candidate in index.candidates[:50]
        ],
    }


async def _index_for_language(
    hass: Any,
    runtime: CanonicalizerRuntime,
    language: str,
    *,
    rebuild_if_missing: bool,
) -> CanonicalIndex | None:
    """Return a cached index, optionally rebuilding it in the executor."""
    index = runtime.get_index(language)
    if index is not None or not rebuild_if_missing:
        return index
    return await _rebuild_index(hass, runtime, language)


async def _rebuild_index(
    hass: Any,
    runtime: CanonicalizerRuntime,
    language: str,
) -> CanonicalIndex:
    """Rebuild an index once outside the Home Assistant event loop."""
    return await runtime.async_rebuild_index(hass, language)


def _runtime_from_hass(hass: Any) -> CanonicalizerRuntime:
    """Return the active runtime object."""
    domain_data = hass.data.get(DOMAIN, {})
    for entry_data in domain_data.values():
        runtime = entry_data.get(DATA_RUNTIME)
        if isinstance(runtime, CanonicalizerRuntime):
            return runtime
    raise HomeAssistantError("Assist Canonicalizer is not loaded")


def _service_language(hass: Any, call: ServiceCall) -> str:
    """Return the service language, falling back to Home Assistant config."""
    language = call.data.get(ATTR_LANGUAGE) or hass.config.language
    return normalize_language(str(language))


def _registry_slot_counts(runtime: CanonicalizerRuntime) -> dict[str, int]:
    """Return registry slot value counts for diagnostics."""
    return {
        slot_name: len(values) for slot_name, values in sorted(runtime.registry_slot_values.items())
    }


def _ranked_candidate_candidate_response(candidate: Any) -> dict[str, Any]:
    """Return serializable candidate metadata without scores."""
    return {
        ATTR_TEXT: candidate.text,
        ATTR_INTENT_NAME: candidate.intent_name,
        ATTR_SOURCE: candidate.source.value,
        ATTR_NORMALIZED_TEXT: candidate.normalized_text,
    }


def _ranked_candidate_response(ranked: RankedCandidate, query: str | None = None) -> dict[str, Any]:
    """Return a serializable ranked candidate response."""
    candidate = ranked.candidate
    text = candidate.text
    normalized_text = candidate.normalized_text
    if query is not None:
        text = rehydrate_wildcard_text(text, query, candidate.language)
        normalized_text = normalize_text(text)
    return {
        ATTR_TEXT: text,
        ATTR_INTENT_NAME: candidate.intent_name,
        ATTR_SOURCE: candidate.source.value,
        ATTR_NORMALIZED_TEXT: normalized_text,
        "score": ranked.scores.final_score,
        "scores": {
            "rapidfuzz": ranked.scores.rapidfuzz_score,
            "char_ngram": ranked.scores.char_ngram_score,
            "bm25": ranked.scores.bm25_score,
            "intent": ranked.scores.intent_score,
            "penalty": ranked.scores.penalty,
            "final": ranked.scores.final_score,
        },
    }
