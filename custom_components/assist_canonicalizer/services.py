"""Home Assistant service handlers for Assist Canonicalizer."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable, Coroutine
from functools import partial
from typing import Any

import voluptuous as vol
from homeassistant.components.conversation.agent_manager import async_get_agent
from homeassistant.components.conversation.const import HOME_ASSISTANT_AGENT
from homeassistant.core import ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv

from .builtin_intents import language_variant_for
from .candidate import Candidate
from .const import (
    ATTR_ACCEPTED,
    ATTR_AGENT_ID,
    ATTR_CANDIDATE_COUNT,
    ATTR_INTENT_NAME,
    ATTR_LANGUAGE,
    ATTR_NORMALIZED_TEXT,
    ATTR_SELECTED_CANDIDATE,
    ATTR_SOURCE,
    ATTR_TEXT,
    ATTR_TOP_CANDIDATES,
    CONF_FALLBACK_AGENT_ID,
    DATA_RUNTIME,
    DEFAULT_MAX_DYNAMIC_CANDIDATES,
    DEFAULT_MAX_DYNAMIC_SLOT_VALUES,
    DEFAULT_MAX_REGISTRY_VALUES_NOMINATED,
    DEFAULT_MAX_REGISTRY_VALUES_SCORED_PER_QUERY,
    DOMAIN,
    SERVICE_CLEAR_INDEX,
    SERVICE_DIAGNOSTICS,
    SERVICE_DUMP_CANDIDATES,
    SERVICE_REBUILD_INDEX,
    SERVICE_SET_FALLBACK_AGENT,
    SERVICE_TEST_MATCH,
)
from .indexer import CanonicalIndex
from .normalization import normalize_text
from .ranking import RankedCandidate, evaluate_confidence_gates
from .rehydration import get_wildcard_rehydration
from .runtime import CanonicalizerRuntime
from .utils import elapsed_ms, normalize_language, resolve_entry_thresholds

_LOGGER = logging.getLogger(__name__)

ATTR_REBUILD = "rebuild"
_CANDIDATE_SAMPLE_LIMIT = 50


def validate_supported_language(value: Any) -> str:
    """Validate that the language is supported by Home Assistant.

    Resolve the language through the shared Home Assistant language-pack matcher.
    """
    lang = cv.string(value)
    if not lang.strip():
        raise vol.Invalid("Language cannot be empty")

    language_variant = language_variant_for(lang)
    if language_variant is not None:
        return language_variant
    raise vol.Invalid(f"Language '{lang}' is not supported by Home Assistant")


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
SET_FALLBACK_AGENT_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_AGENT_ID): vol.All(
            cv.string,
            str.strip,
            vol.Length(min=1),
        )
    }
)


def async_setup_services(hass: Any) -> None:
    """Register Assist Canonicalizer services."""
    hass.services.async_register(
        DOMAIN,
        SERVICE_SET_FALLBACK_AGENT,
        partial(_handle_set_fallback_agent, hass),
        schema=SET_FALLBACK_AGENT_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_TEST_MATCH,
        partial(_handle_test_match, hass),
        schema=TEST_MATCH_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_REBUILD_INDEX,
        partial(_handle_rebuild_index, hass),
        schema=REBUILD_INDEX_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_CLEAR_INDEX,
        partial(_handle_clear_index, hass),
        schema=CLEAR_INDEX_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_DIAGNOSTICS,
        partial(_handle_diagnostics, hass),
        schema=DIAGNOSTICS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_DUMP_CANDIDATES,
        partial(_handle_dump_candidates, hass),
        schema=DUMP_CANDIDATES_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )


def async_unload_services(hass: Any) -> None:
    """Remove Assist Canonicalizer services."""
    hass.services.async_remove(DOMAIN, SERVICE_SET_FALLBACK_AGENT)
    hass.services.async_remove(DOMAIN, SERVICE_TEST_MATCH)
    hass.services.async_remove(DOMAIN, SERVICE_REBUILD_INDEX)
    hass.services.async_remove(DOMAIN, SERVICE_CLEAR_INDEX)
    hass.services.async_remove(DOMAIN, SERVICE_DIAGNOSTICS)
    hass.services.async_remove(DOMAIN, SERVICE_DUMP_CANDIDATES)


def _wrap_service_errors(
    action_name: str,
) -> Callable[[Callable[..., Coroutine[Any, Any, Any]]], Callable[..., Coroutine[Any, Any, Any]]]:
    """Wrap service exceptions into a sanitized user-facing HomeAssistantError."""

    def decorator(
        func: Callable[..., Coroutine[Any, Any, Any]],
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        """Decorate the service handler."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrap execution and catch specific error types."""
            try:
                return await func(*args, **kwargs)
            except HomeAssistantError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as err:
                _LOGGER.error(
                    "%s failed: %s",
                    action_name,
                    err,
                    exc_info=err,
                )
                raise HomeAssistantError(f"{action_name} failed; see logs for details") from err

        return wrapper

    return decorator


@_wrap_service_errors("Fallback agent update")
async def _handle_set_fallback_agent(hass: Any, call: ServiceCall) -> dict[str, Any]:
    """Persist a new fallback conversation agent without reloading the entry."""
    _runtime, entry = _runtime_entry_from_hass(hass)
    if entry is None:
        raise HomeAssistantError("Assist Canonicalizer config entry is not available")

    agent_id = call.data[ATTR_AGENT_ID]
    if agent_id == entry.entry_id:
        raise HomeAssistantError("Assist Canonicalizer cannot use itself as the fallback agent")

    try:
        agent = async_get_agent(hass, agent_id)
    except (KeyError, ValueError):
        agent = None
    if agent is None:
        raise HomeAssistantError(f"Conversation agent '{agent_id}' is not available")
    if _agent_belongs_to_entry(agent, entry.entry_id):
        raise HomeAssistantError("Assist Canonicalizer cannot use itself as the fallback agent")

    previous_agent_id = _configured_fallback_agent_id(entry)
    options = dict(getattr(entry, "options", {}) or {})
    options[CONF_FALLBACK_AGENT_ID] = agent_id
    config_entry_changed = hass.config_entries.async_update_entry(entry, options=options)
    return {
        CONF_FALLBACK_AGENT_ID: agent_id,
        "previous_fallback_agent_id": previous_agent_id,
        "changed": config_entry_changed,
    }


@_wrap_service_errors("Matching test")
async def _handle_test_match(hass: Any, call: ServiceCall) -> dict[str, Any]:
    """Return ranked candidates for a text input with lexical scoring and custom thresholds."""
    runtime, entry = _runtime_entry_from_hass(hass)
    language = _service_language(hass, call)
    text = call.data[ATTR_TEXT]
    index = await _index_for_language(hass, runtime, language, rebuild_if_missing=True)
    if index is None:
        raise HomeAssistantError("Assist Canonicalizer index could not be built")
    min_confidence, min_margin = resolve_entry_thresholds(entry)

    ranked = await hass.async_add_executor_job(
        partial(
            runtime.rank_with_dynamic_candidates,
            min_confidence=min_confidence,
            min_margin=min_margin,
        ),
        language,
        index,
        text,
    )

    decision = evaluate_confidence_gates(
        ranked,
        min_confidence=min_confidence,
        min_margin=min_margin,
        query=text,
        language=language,
    )
    selected = decision.accepted_candidate

    return {
        ATTR_LANGUAGE: language,
        ATTR_NORMALIZED_TEXT: normalize_text(text),
        ATTR_CANDIDATE_COUNT: index.candidate_count,
        "dynamic_candidate_count": runtime.diagnostics.dynamic_candidate_count,
        "evaluation": {
            "scope": "lexical",
            "candidate_metadata_authoritative": False,
            "live_recognition": "not_run",
            "production_decision_path": "/api/conversation/process",
        },
        "confidence_gate": decision.as_dict(),
        ATTR_ACCEPTED: selected is not None,
        ATTR_SELECTED_CANDIDATE: (
            _ranked_candidate_response(selected, query=text) if selected else None
        ),
        ATTR_TOP_CANDIDATES: [_ranked_candidate_response(item, query=text) for item in ranked],
    }


@_wrap_service_errors("Index rebuild")
async def _handle_rebuild_index(hass: Any, call: ServiceCall) -> dict[str, Any]:
    """Rebuild one language index from automatic candidate sources."""
    runtime = _runtime_from_hass(hass)
    language = _service_language(hass, call)
    started_at = time.monotonic()
    index = await _rebuild_index(hass, runtime, language)
    if index is None:
        raise HomeAssistantError("Index rebuild failed or was cancelled")
    return {
        ATTR_LANGUAGE: language,
        ATTR_CANDIDATE_COUNT: index.candidate_count,
        "rebuild_latency_ms": elapsed_ms(started_at),
    }


@_wrap_service_errors("Clear index")
async def _handle_clear_index(hass: Any, call: ServiceCall) -> dict[str, Any]:
    """Clear one language index or all indexes."""
    runtime = _runtime_from_hass(hass)
    requested_language = call.data.get(ATTR_LANGUAGE)
    language = (
        normalize_language(requested_language) if isinstance(requested_language, str) else None
    )
    clear_result = await runtime.async_clear_index(hass, language)
    return {
        ATTR_LANGUAGE: language,
        "scope": "all" if language is None else "language",
        "cleared_cached_languages": list(clear_result.cleared_cached_languages),
        "cleared_candidate_count": clear_result.cleared_candidate_count,
        "remaining_candidate_count": clear_result.remaining_candidate_count,
        "remaining_cached_languages": list(clear_result.remaining_cached_languages),
    }


@_wrap_service_errors("Diagnostics")
async def _handle_diagnostics(hass: Any, call: ServiceCall) -> dict[str, Any]:
    """Return runtime diagnostics."""
    runtime = _runtime_from_hass(hass)
    diagnostics = runtime.diagnostics.as_dict()
    diagnostics.pop(ATTR_CANDIDATE_COUNT, None)
    diagnostics.pop("index_version", None)
    cached_indexes = {
        language: {
            ATTR_CANDIDATE_COUNT: index.candidate_count,
            "version": index.version,
        }
        for language, index in sorted(runtime.indexes.items())
    }
    diagnostics.update(
        {
            "total_cached_candidate_count": runtime.total_candidate_count(),
            "cached_indexes": cached_indexes,
            "pending_rebuild_languages": sorted(runtime.rebuild_tasks),
            "registry_slot_counts": _registry_slot_counts(runtime),
            "dynamic_candidate_generation": {
                "enabled": True,
                "max_slot_values_per_slot": DEFAULT_MAX_DYNAMIC_SLOT_VALUES,
                "max_candidates_per_query": DEFAULT_MAX_DYNAMIC_CANDIDATES,
                "max_registry_values_nominated_per_slot": (DEFAULT_MAX_REGISTRY_VALUES_NOMINATED),
                "max_registry_values_scored_per_query": (
                    DEFAULT_MAX_REGISTRY_VALUES_SCORED_PER_QUERY
                ),
            },
            "subscribed_intent_source_counts": runtime.subscribed_source_counts(),
        }
    )
    return diagnostics


@_wrap_service_errors("Dump candidates")
async def _handle_dump_candidates(hass: Any, call: ServiceCall) -> dict[str, Any]:
    """Return candidate pool details for a language."""
    runtime = _runtime_from_hass(hass)
    language = _service_language(hass, call)
    should_rebuild = bool(call.data.get(ATTR_REBUILD, False))
    rebuild_latency_ms: float | None = None
    if should_rebuild:
        started_at = time.monotonic()
        index = await _rebuild_index(hass, runtime, language)
        if index is None:
            raise HomeAssistantError("Index rebuild failed or was cancelled")
        rebuild_latency_ms = elapsed_ms(started_at)
        index_status = "rebuilt"
    else:
        index = runtime.get_index(language)
        index_status = "cached" if index is not None else "missing"

    intent_source_counts = await hass.async_add_executor_job(runtime.source_counts, language)
    if index is None:
        return {
            ATTR_LANGUAGE: language,
            ATTR_CANDIDATE_COUNT: 0,
            "index_status": index_status,
            "rebuild_latency_ms": rebuild_latency_ms,
            "intent_source_counts": intent_source_counts,
            "candidate_source_counts": {},
            "intent_counts": {},
            "registry_slot_counts": _registry_slot_counts(runtime),
            "candidate_sample": {
                "truncated": False,
                "candidates": [],
            },
        }

    source_counts, intent_counts = await hass.async_add_executor_job(
        _count_candidate_sources_and_intents, index
    )
    sample_candidates = [
        _ranked_candidate_candidate_response(candidate)
        for candidate in index.candidates[:_CANDIDATE_SAMPLE_LIMIT]
    ]
    return {
        ATTR_LANGUAGE: language,
        ATTR_CANDIDATE_COUNT: index.candidate_count,
        "index_status": index_status,
        "rebuild_latency_ms": rebuild_latency_ms,
        "intent_source_counts": intent_source_counts,
        "candidate_source_counts": source_counts,
        "intent_counts": dict(sorted(intent_counts.items())),
        "registry_slot_counts": _registry_slot_counts(runtime),
        "candidate_sample": {
            "truncated": index.candidate_count > len(sample_candidates),
            "candidates": sample_candidates,
        },
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
) -> CanonicalIndex | None:
    """Rebuild an index once outside the Home Assistant event loop."""
    return await runtime.async_rebuild_index(hass, language, log_error=False, raise_on_error=True)


def _count_candidate_sources_and_intents(
    index: CanonicalIndex,
) -> tuple[dict[str, int], dict[str, int]]:
    """Count candidates by source and intent outside the event loop."""
    source_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    for candidate in index.candidates:
        source_counts[candidate.source.value] = source_counts.get(candidate.source.value, 0) + 1
        intent_counts[candidate.intent_name] = intent_counts.get(candidate.intent_name, 0) + 1
    return source_counts, intent_counts


def _runtime_from_hass(hass: Any) -> CanonicalizerRuntime:
    """Return the active runtime object."""
    runtime, _entry = _runtime_entry_from_hass(hass)
    return runtime


def _runtime_entry_from_hass(hass: Any) -> tuple[CanonicalizerRuntime, Any]:
    """Return the active runtime object and its config entry."""
    domain_data = hass.data.get(DOMAIN, {})
    for entry_data in domain_data.values():
        runtime = entry_data.get(DATA_RUNTIME)
        if isinstance(runtime, CanonicalizerRuntime):
            return runtime, entry_data.get("entry")
    raise HomeAssistantError("Assist Canonicalizer is not loaded")


def _agent_belongs_to_entry(agent: Any, entry_id: str) -> bool:
    """Return whether a conversation agent belongs to the canonicalizer entry."""
    registry_entry = getattr(agent, "registry_entry", None)
    return getattr(agent, "unique_id", None) == f"{entry_id}-conversation" or (
        registry_entry is not None and getattr(registry_entry, "config_entry_id", None) == entry_id
    )


def _configured_fallback_agent_id(entry: Any) -> str:
    """Return the effective fallback agent configured before a service update."""
    options = getattr(entry, "options", {}) or {}
    data = getattr(entry, "data", {}) or {}
    configured = options.get(CONF_FALLBACK_AGENT_ID) or data.get(CONF_FALLBACK_AGENT_ID)
    if not isinstance(configured, str) or configured == entry.entry_id:
        return HOME_ASSISTANT_AGENT
    return configured


def _service_language(hass: Any, call: ServiceCall) -> str:
    """Return the service language, falling back to Home Assistant config."""
    language = call.data.get(ATTR_LANGUAGE) or hass.config.language
    return normalize_language(str(language))


def _registry_slot_counts(runtime: CanonicalizerRuntime) -> dict[str, int]:
    """Return registry slot value counts for diagnostics."""
    return {
        slot_name: len(values) for slot_name, values in sorted(runtime.registry_slot_values.items())
    }


def _ranked_candidate_candidate_response(candidate: Candidate) -> dict[str, Any]:
    """Return serializable candidate metadata without scores."""
    return {
        ATTR_TEXT: candidate.text,
        ATTR_INTENT_NAME: candidate.intent_name,
        ATTR_SOURCE: candidate.source.value,
        ATTR_NORMALIZED_TEXT: candidate.normalized_text,
        "slots": candidate.parsed_slots,
        "wildcard_slots": sorted(
            {wildcard_name for _index, wildcard_name in candidate.wildcard_infos}
        ),
        "sentence_template": candidate.metadata.get("sentence_template"),
    }


def _ranked_candidate_response(ranked: RankedCandidate, query: str | None = None) -> dict[str, Any]:
    """Return a serializable ranked candidate response."""
    candidate = ranked.candidate
    text = candidate.text
    normalized_text = candidate.normalized_text
    replacements: dict[str, str] = {}
    if query is not None:
        text, replacements = get_wildcard_rehydration(candidate, query)
        normalized_text = normalize_text(text)
    return {
        ATTR_TEXT: text,
        ATTR_INTENT_NAME: candidate.intent_name,
        ATTR_SOURCE: candidate.source.value,
        ATTR_NORMALIZED_TEXT: normalized_text,
        "slots": candidate.parsed_slots,
        "wildcard_replacements": replacements,
        "scores": {
            "rapidfuzz": ranked.scores.rapidfuzz_score,
            "char_ngram": ranked.scores.char_ngram_score,
            "bm25": ranked.scores.bm25_score,
            "intent": ranked.scores.intent_score,
            "penalty": ranked.scores.penalty,
            "final": ranked.scores.final_score,
        },
    }
