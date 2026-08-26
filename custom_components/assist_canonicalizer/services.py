"""Home Assistant service handlers for Assist Canonicalizer."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable, Coroutine, Iterable, Mapping, Sequence
from functools import partial
from typing import TypedDict

import voluptuous as vol
from homeassistant.components.conversation.agent_manager import async_get_agent
from homeassistant.components.conversation.const import HOME_ASSISTANT_AGENT
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.util.json import JsonObjectType, JsonValueType

from .builtin_intents import language_variant_for
from .candidate import Candidate
from .const import (
    DATA_RUNTIME,
    DEFAULT_MAX_DYNAMIC_CANDIDATES,
    DEFAULT_MAX_DYNAMIC_SLOT_VALUES,
    DEFAULT_MAX_REGISTRY_VALUES_NOMINATED,
    DEFAULT_MAX_REGISTRY_VALUES_SCORED_PER_QUERY,
    DOMAIN,
    AttributeName,
    ConfigKey,
    ServiceName,
)
from .indexer import CanonicalIndex
from .normalization import normalize_text
from .ranking import ConfidenceGatePayload, RankedCandidate, evaluate_confidence_gates
from .rehydration import get_wildcard_rehydration
from .runtime import CanonicalizerRuntime
from .utils import elapsed_ms, normalize_language, resolve_entry_thresholds

_LOGGER = logging.getLogger(__name__)

_CANDIDATE_SAMPLE_LIMIT = 50


class ScorePayload(TypedDict):
    """Serialized lexical scores for one ranked candidate."""

    rapidfuzz: float
    char_ngram: float
    bm25: float
    intent: float
    penalty: float
    final: float


class RankedCandidatePayload(TypedDict):
    """Serialized ranked candidate returned by the test-match service."""

    text: str
    intent_name: str
    source: str
    normalized_text: str
    slots: JsonObjectType
    wildcard_replacements: dict[str, str]
    scores: ScorePayload


class EvaluationPayload(TypedDict):
    """Scope metadata for a lexical-only service evaluation."""

    scope: str
    candidate_metadata_authoritative: bool
    live_recognition: str
    production_decision_path: str


class TestMatchPayload(TypedDict):
    """Response from the lexical test-match service."""

    language: str
    normalized_text: str
    candidate_count: int
    dynamic_candidate_count: int
    evaluation: EvaluationPayload
    confidence_gate: ConfidenceGatePayload
    accepted: bool
    selected_candidate: RankedCandidatePayload | None
    top_candidates: list[RankedCandidatePayload]


class RebuildPayload(TypedDict):
    """Response from the index-rebuild service."""

    language: str
    candidate_count: int
    rebuild_latency_ms: float


class ClearIndexPayload(TypedDict):
    """Response from the index-clear service."""

    language: str | None
    scope: str
    cleared_cached_languages: list[str]
    cleared_candidate_count: int
    remaining_candidate_count: int
    remaining_cached_languages: list[str]


class CandidateMetadataPayload(TypedDict):
    """Serialized candidate metadata without scores."""

    text: str
    intent_name: str
    source: str
    normalized_text: str
    slots: JsonObjectType
    wildcard_slots: list[str]
    sentence_template: str | None


class CandidateSamplePayload(TypedDict):
    """Bounded candidate sample in a dump response."""

    truncated: bool
    candidates: list[CandidateMetadataPayload]


class DumpCandidatesPayload(TypedDict):
    """Response from the candidate-dump service."""

    language: str
    candidate_count: int
    index_status: str
    rebuild_latency_ms: float | None
    intent_source_counts: dict[str, int]
    candidate_source_counts: dict[str, int]
    intent_counts: dict[str, int]
    registry_slot_counts: dict[str, int]
    candidate_sample: CandidateSamplePayload


def _json_int_mapping(values: Mapping[str, int]) -> JsonObjectType:
    """Return integer metrics in a JSON-object-compatible mapping."""
    result: JsonObjectType = dict(values.items())
    return result


def _json_object_list(values: Sequence[JsonObjectType]) -> list[JsonValueType]:
    """Return JSON objects in a recursively typed JSON list."""
    return [*values]


def _json_string_list(values: Iterable[str]) -> list[JsonValueType]:
    """Return strings in a recursively typed JSON list."""
    result: list[JsonValueType] = []
    result.extend(values)
    return result


def validate_supported_language(value: object) -> str:
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
        vol.Required(AttributeName.TEXT.value): cv.string,
        vol.Optional(AttributeName.LANGUAGE.value): validate_supported_language,
    }
)

REBUILD_INDEX_SCHEMA = vol.Schema(
    {vol.Optional(AttributeName.LANGUAGE.value): validate_supported_language}
)
CLEAR_INDEX_SCHEMA = vol.Schema(
    {vol.Optional(AttributeName.LANGUAGE.value): validate_supported_language}
)
DIAGNOSTICS_SCHEMA = vol.Schema({})
DUMP_CANDIDATES_SCHEMA = vol.Schema(
    {
        vol.Optional(AttributeName.LANGUAGE.value): validate_supported_language,
        vol.Optional(AttributeName.REBUILD.value, default=False): cv.boolean,
    }
)
SET_FALLBACK_AGENT_SCHEMA = vol.Schema(
    {
        vol.Required(AttributeName.AGENT_ID.value): vol.All(
            cv.string,
            str.strip,
            vol.Length(min=1),
        )
    }
)


def _json_value(value: object) -> tuple[bool, JsonValueType]:
    """Convert an arbitrary value into a supported JsonValueType."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True, value
    if isinstance(value, Mapping):
        return True, _json_object_dict(value)
    if isinstance(value, (list, tuple)):
        items: list[JsonValueType] = []
        for item in value:
            valid, converted_item = _json_value(item)
            if valid:
                items.append(converted_item)
        return True, items
    return False, None


def _json_object_dict[K](values: Mapping[K, object]) -> JsonObjectType:
    """Return a dictionary formatted for Home Assistant service responses."""
    result: JsonObjectType = {}
    for key, value in values.items():
        if not isinstance(key, str):
            continue
        valid, converted = _json_value(value)
        if valid:
            result[key] = converted
    return result


def _registered_service_handler[R: Mapping[str, object]](
    hass: HomeAssistant,
    handler: Callable[[HomeAssistant, ServiceCall], Coroutine[object, object, R]],
) -> Callable[[ServiceCall], Coroutine[object, object, JsonObjectType]]:
    """Adapt a precisely typed response to Home Assistant's service boundary."""

    async def registered(call: ServiceCall) -> JsonObjectType:
        """Execute the service handler and return service response payload."""
        response = await handler(hass, call)
        return _json_object_dict(response)

    return registered


def async_setup_services(hass: HomeAssistant) -> None:
    """Register Assist Canonicalizer services."""
    hass.services.async_register(
        DOMAIN,
        ServiceName.SET_FALLBACK_AGENT,
        _registered_service_handler(hass, _handle_set_fallback_agent),
        schema=SET_FALLBACK_AGENT_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        ServiceName.TEST_MATCH,
        _registered_service_handler(hass, _handle_test_match),
        schema=TEST_MATCH_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        ServiceName.REBUILD_INDEX,
        _registered_service_handler(hass, _handle_rebuild_index),
        schema=REBUILD_INDEX_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        ServiceName.CLEAR_INDEX,
        _registered_service_handler(hass, _handle_clear_index),
        schema=CLEAR_INDEX_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        ServiceName.DIAGNOSTICS,
        _registered_service_handler(hass, _handle_diagnostics),
        schema=DIAGNOSTICS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        ServiceName.DUMP_CANDIDATES,
        _registered_service_handler(hass, _handle_dump_candidates),
        schema=DUMP_CANDIDATES_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )


def async_unload_services(hass: HomeAssistant) -> None:
    """Remove Assist Canonicalizer services."""
    hass.services.async_remove(DOMAIN, ServiceName.SET_FALLBACK_AGENT)
    hass.services.async_remove(DOMAIN, ServiceName.TEST_MATCH)
    hass.services.async_remove(DOMAIN, ServiceName.REBUILD_INDEX)
    hass.services.async_remove(DOMAIN, ServiceName.CLEAR_INDEX)
    hass.services.async_remove(DOMAIN, ServiceName.DIAGNOSTICS)
    hass.services.async_remove(DOMAIN, ServiceName.DUMP_CANDIDATES)


def _wrap_service_errors[R](
    action_name: str,
) -> Callable[
    [Callable[[HomeAssistant, ServiceCall], Coroutine[object, object, R]]],
    Callable[[HomeAssistant, ServiceCall], Coroutine[object, object, R]],
]:
    """Wrap service exceptions into a sanitized user-facing HomeAssistantError."""

    def decorator(
        func: Callable[[HomeAssistant, ServiceCall], Coroutine[object, object, R]],
    ) -> Callable[[HomeAssistant, ServiceCall], Coroutine[object, object, R]]:
        """Decorate the service handler."""

        @functools.wraps(func)
        async def wrapper(hass: HomeAssistant, call: ServiceCall) -> R:
            """Wrap execution and catch specific error types."""
            try:
                return await func(hass, call)
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
async def _handle_set_fallback_agent(hass: HomeAssistant, call: ServiceCall) -> JsonObjectType:
    """Persist a new fallback conversation agent without reloading the entry."""
    _runtime, entry = _runtime_entry_from_hass(hass)
    if entry is None:
        raise HomeAssistantError("Assist Canonicalizer config entry is not available")

    agent_id = call.data[AttributeName.AGENT_ID]
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
    options[ConfigKey.FALLBACK_AGENT_ID] = agent_id
    config_entry_changed = hass.config_entries.async_update_entry(entry, options=options)
    return {
        ConfigKey.FALLBACK_AGENT_ID: agent_id,
        "previous_fallback_agent_id": previous_agent_id,
        "changed": config_entry_changed,
    }


@_wrap_service_errors("Matching test")
async def _handle_test_match(hass: HomeAssistant, call: ServiceCall) -> TestMatchPayload:
    """Return ranked candidates for a text input with lexical scoring and custom thresholds."""
    runtime, entry = _runtime_entry_from_hass(hass)
    language = _service_language(hass, call)
    text = call.data[AttributeName.TEXT]
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

    return TestMatchPayload(
        language=language,
        normalized_text=normalize_text(text),
        candidate_count=index.candidate_count,
        dynamic_candidate_count=runtime.diagnostics.dynamic_candidate_count,
        evaluation={
            "scope": "lexical",
            "candidate_metadata_authoritative": False,
            "live_recognition": "not_run",
            "production_decision_path": "/api/conversation/process",
        },
        confidence_gate=decision.as_dict(),
        accepted=selected is not None,
        selected_candidate=(_ranked_candidate_response(selected, query=text) if selected else None),
        top_candidates=[_ranked_candidate_response(item, query=text) for item in ranked],
    )


@_wrap_service_errors("Index rebuild")
async def _handle_rebuild_index(hass: HomeAssistant, call: ServiceCall) -> RebuildPayload:
    """Rebuild one language index from automatic candidate sources."""
    runtime = _runtime_from_hass(hass)
    language = _service_language(hass, call)
    started_at = time.monotonic()
    index = await _rebuild_index(hass, runtime, language)
    if index is None:
        raise HomeAssistantError("Index rebuild failed or was cancelled")
    return RebuildPayload(
        language=language,
        candidate_count=index.candidate_count,
        rebuild_latency_ms=elapsed_ms(started_at),
    )


@_wrap_service_errors("Clear index")
async def _handle_clear_index(hass: HomeAssistant, call: ServiceCall) -> ClearIndexPayload:
    """Clear one language index or all indexes."""
    runtime = _runtime_from_hass(hass)
    requested_language = call.data.get(AttributeName.LANGUAGE)
    language = (
        normalize_language(requested_language) if isinstance(requested_language, str) else None
    )
    clear_result = await runtime.async_clear_index(hass, language)
    return ClearIndexPayload(
        language=language,
        scope="all" if language is None else "language",
        cleared_cached_languages=list(clear_result.cleared_cached_languages),
        cleared_candidate_count=clear_result.cleared_candidate_count,
        remaining_candidate_count=clear_result.remaining_candidate_count,
        remaining_cached_languages=list(clear_result.remaining_cached_languages),
    )


@_wrap_service_errors("Diagnostics")
async def _handle_diagnostics(hass: HomeAssistant, call: ServiceCall) -> JsonObjectType:
    """Return runtime diagnostics."""
    runtime = _runtime_from_hass(hass)
    diagnostics = runtime.diagnostics.as_dict()
    diagnostics.pop(AttributeName.CANDIDATE_COUNT, None)
    diagnostics.pop("index_version", None)
    cached_indexes: JsonObjectType = {}
    for language, index in sorted(runtime.indexes.items()):
        index_summary: JsonObjectType = {
            AttributeName.CANDIDATE_COUNT: index.candidate_count,
            "version": index.version,
        }
        cached_indexes[language] = index_summary
    dynamic_generation: JsonObjectType = {
        "enabled": True,
        "max_slot_values_per_slot": DEFAULT_MAX_DYNAMIC_SLOT_VALUES,
        "max_candidates_per_query": DEFAULT_MAX_DYNAMIC_CANDIDATES,
        "max_registry_values_nominated_per_slot": DEFAULT_MAX_REGISTRY_VALUES_NOMINATED,
        "max_registry_values_scored_per_query": DEFAULT_MAX_REGISTRY_VALUES_SCORED_PER_QUERY,
    }
    pending_languages: list[JsonValueType] = [*sorted(runtime.rebuild_tasks)]
    diagnostics.update(
        {
            "total_cached_candidate_count": runtime.total_candidate_count(),
            "cached_indexes": cached_indexes,
            "pending_rebuild_languages": pending_languages,
            "registry_slot_counts": _json_int_mapping(_registry_slot_counts(runtime)),
            "dynamic_candidate_generation": dynamic_generation,
            "subscribed_intent_source_counts": _json_int_mapping(
                runtime.subscribed_source_counts()
            ),
        }
    )
    return diagnostics


@_wrap_service_errors("Dump candidates")
async def _handle_dump_candidates(hass: HomeAssistant, call: ServiceCall) -> DumpCandidatesPayload:
    """Return candidate pool details for a language."""
    runtime = _runtime_from_hass(hass)
    language = _service_language(hass, call)
    should_rebuild = bool(call.data.get(AttributeName.REBUILD, False))
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
        empty_sample = CandidateSamplePayload(
            truncated=False,
            candidates=[],
        )
        return DumpCandidatesPayload(
            language=language,
            candidate_count=0,
            index_status=index_status,
            rebuild_latency_ms=rebuild_latency_ms,
            intent_source_counts=intent_source_counts,
            candidate_source_counts={},
            intent_counts={},
            registry_slot_counts=_registry_slot_counts(runtime),
            candidate_sample=empty_sample,
        )

    source_counts, intent_counts = await hass.async_add_executor_job(
        _count_candidate_sources_and_intents, index
    )
    sample_candidates = [
        _ranked_candidate_candidate_response(candidate)
        for candidate in index.candidates[:_CANDIDATE_SAMPLE_LIMIT]
    ]
    candidate_sample = CandidateSamplePayload(
        truncated=index.candidate_count > len(sample_candidates),
        candidates=sample_candidates,
    )
    return DumpCandidatesPayload(
        language=language,
        candidate_count=index.candidate_count,
        index_status=index_status,
        rebuild_latency_ms=rebuild_latency_ms,
        intent_source_counts=intent_source_counts,
        candidate_source_counts=source_counts,
        intent_counts=dict(sorted(intent_counts.items())),
        registry_slot_counts=_registry_slot_counts(runtime),
        candidate_sample=candidate_sample,
    )


async def _index_for_language(
    hass: HomeAssistant,
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
    hass: HomeAssistant,
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


def _runtime_from_hass(hass: HomeAssistant) -> CanonicalizerRuntime:
    """Return the active runtime object."""
    runtime, _entry = _runtime_entry_from_hass(hass)
    return runtime


def _runtime_entry_from_hass(
    hass: HomeAssistant,
) -> tuple[CanonicalizerRuntime, ConfigEntry | None]:
    """Return the active runtime object and its config entry."""
    domain_data = hass.data.get(DOMAIN, {})
    for entry_data in domain_data.values():
        runtime = entry_data.get(DATA_RUNTIME)
        if isinstance(runtime, CanonicalizerRuntime):
            entry = entry_data.get("entry")
            return runtime, entry if entry is not None else None
    raise HomeAssistantError("Assist Canonicalizer is not loaded")


def _agent_belongs_to_entry(agent: object, entry_id: str) -> bool:
    """Return whether a conversation agent belongs to the canonicalizer entry."""
    registry_entry = getattr(agent, "registry_entry", None)
    return getattr(agent, "unique_id", None) == f"{entry_id}-conversation" or (
        registry_entry is not None and getattr(registry_entry, "config_entry_id", None) == entry_id
    )


def _configured_fallback_agent_id(entry: ConfigEntry) -> str:
    """Return the effective fallback agent configured before a service update."""
    options = getattr(entry, "options", {}) or {}
    data = getattr(entry, "data", {}) or {}
    configured = options.get(ConfigKey.FALLBACK_AGENT_ID) or data.get(ConfigKey.FALLBACK_AGENT_ID)
    if not isinstance(configured, str) or configured == entry.entry_id:
        return HOME_ASSISTANT_AGENT
    return configured


def _service_language(hass: HomeAssistant, call: ServiceCall) -> str:
    """Return the service language, falling back to Home Assistant config."""
    language = call.data.get(AttributeName.LANGUAGE) or hass.config.language
    return normalize_language(str(language))


def _registry_slot_counts(runtime: CanonicalizerRuntime) -> dict[str, int]:
    """Return registry slot value counts for diagnostics."""
    return {
        slot_name: len(values) for slot_name, values in sorted(runtime.registry_slot_values.items())
    }


def _ranked_candidate_candidate_response(candidate: Candidate) -> CandidateMetadataPayload:
    """Return serializable candidate metadata without scores."""
    wildcard_slots = sorted({wildcard_name for _index, wildcard_name in candidate.wildcard_infos})
    return CandidateMetadataPayload(
        text=candidate.text,
        intent_name=candidate.intent_name,
        source=candidate.source.value,
        normalized_text=candidate.normalized_text,
        slots=candidate.parsed_slots,
        wildcard_slots=wildcard_slots,
        sentence_template=candidate.metadata.get("sentence_template"),
    )


def _ranked_candidate_response(
    ranked: RankedCandidate, query: str | None = None
) -> RankedCandidatePayload:
    """Return a serializable ranked candidate response."""
    candidate = ranked.candidate
    text = candidate.text
    normalized_text = candidate.normalized_text
    replacements: dict[str, str] = {}
    if query is not None:
        text, replacements = get_wildcard_rehydration(candidate, query)
        normalized_text = normalize_text(text)
    scores: ScorePayload = {
        "rapidfuzz": ranked.scores.rapidfuzz_score,
        "char_ngram": ranked.scores.char_ngram_score,
        "bm25": ranked.scores.bm25_score,
        "intent": ranked.scores.intent_score,
        "penalty": ranked.scores.penalty,
        "final": ranked.scores.final_score,
    }
    return RankedCandidatePayload(
        text=text,
        intent_name=candidate.intent_name,
        source=candidate.source.value,
        normalized_text=normalized_text,
        slots=candidate.parsed_slots,
        wildcard_replacements=replacements,
        scores=scores,
    )
