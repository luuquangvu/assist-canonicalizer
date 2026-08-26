"""Runtime state for Assist Canonicalizer."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence, Set
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from uuid import uuid4

import orjson
from homeassistant.core import HomeAssistant
from homeassistant.helpers import storage
from homeassistant.util.json import JsonObjectType, JsonValueType

from .bm25 import clear_bm25_caches
from .builtin_intents import (
    IntentSource,
    clear_builtin_intents_caches,
    load_language_intent_sources,
)
from .candidate import Candidate, CandidateSource
from .const import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DOMAIN,
    FallbackReason,
)
from .diagnostics import CanonicalizerDiagnostics
from .grammar_loader import (
    DynamicRegistryIntent,
    RegistryRetrievalStats,
    RegistrySlotIndex,
    build_candidates_from_intent_sources,
    build_query_registry_candidates,
    build_registry_slot_index,
    clear_grammar_loader_caches,
    compile_dynamic_registry_intents,
)
from .indexer import CanonicalIndex, build_index
from .normalization import char_ngrams_normalized, clear_normalization_caches, normalize_text
from .ranking import (
    CharNGramIndex,
    ConfidenceGateDecision,
    RankedCandidate,
    _limit_ranked_candidates,
    apply_intent_disambiguation,
    clear_ranking_caches,
    evaluate_confidence_gates,
    rank_candidates,
)
from .rehydration import clear_rehydration_caches
from .utils import (
    clear_utils_caches,
    elapsed_ms,
    normalize_language,
    register_custom_wildcards_from_sources,
)

_LOGGER = logging.getLogger(__name__)


_STORE_HAS_SERIALIZE_IN_EVENT_LOOP = False
with contextlib.suppress(Exception):
    sig = inspect.signature(storage.Store.__init__)
    _STORE_HAS_SERIALIZE_IN_EVENT_LOOP = "serialize_in_event_loop" in sig.parameters
_INDEX_STORE_VERSION = 1
_INDEX_BUILD_VERSION = 6
_INDEX_STORE_PREFIX = f"{DOMAIN}.index_"
_INDEX_MANIFEST_KEY = f"{DOMAIN}.index_manifest"
_INDEX_MANIFEST_VERSION = 1
_MAX_REBUILD_ATTEMPTS = 5

IndexGeneration = tuple[int, int]


def _new_set_event() -> asyncio.Event:
    """Return an asyncio event initialized in the set state."""
    event = asyncio.Event()
    event.set()
    return event


@dataclass(frozen=True, slots=True)
class IndexBuildSnapshot:
    """Immutable inputs and fingerprint for one candidate index build."""

    language: str
    intent_sources: dict[str, IntentSource]
    registry_slot_values: dict[str, tuple[str, ...]]
    dynamic_registry_intents: tuple[DynamicRegistryIntent, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class IndexClearResult:
    """Immutable cache state captured by one atomic index clear."""

    cleared_cached_languages: tuple[str, ...]
    cleared_candidate_count: int
    remaining_candidate_count: int
    remaining_cached_languages: tuple[str, ...]


def _cleared_diagnostic_traces(
    diagnostics: CanonicalizerDiagnostics,
    *,
    clear_request_trace: bool,
    clear_recognition_trace: bool,
) -> CanonicalizerDiagnostics:
    """Return diagnostics with the requested transient trace fields cleared."""
    if clear_request_trace:
        return replace(
            diagnostics,
            last_request_id=None,
            selected_delegated_text_hash=None,
            selected_candidate_source=None,
            confidence_gate=None,
            execution_result=None,
            recognition_kind=None,
            recognition_intent=None,
            recognition_unmatched_count=0,
            recognition_latency_ms=None,
            preflight_attempt_count=0,
            metadata_diverged=False,
            metadata_intent_matches_observed=None,
            metadata_slots_match_observed=None,
            metadata_divergence_reason=None,
            recovery_used=False,
            registry_postings_consulted=0,
            registry_values_nominated=0,
            registry_values_scored=0,
            fuzzy_dynamic_candidates=0,
            registry_retrieval_latency_ms=None,
            selected_from_fuzzy_registry=False,
        )
    if clear_recognition_trace:
        return replace(
            diagnostics,
            recognition_kind=None,
            recognition_intent=None,
            recognition_unmatched_count=0,
            recognition_latency_ms=None,
            metadata_diverged=False,
            metadata_intent_matches_observed=None,
            metadata_slots_match_observed=None,
            metadata_divergence_reason=None,
        )
    return diagnostics


@dataclass(slots=True)
class CanonicalizerRuntime:
    """Mutable runtime state shared by the integration entry and agent.

    Lifecycle invariant: shutdown closes work admission before it cancels tasks.
    Methods that start or publish work must reject a closed runtime, and methods
    that span an await must recheck the relevant generation before publishing.
    """

    indexes: dict[str, CanonicalIndex] = field(default_factory=dict)
    diagnostics: CanonicalizerDiagnostics = field(default_factory=CanonicalizerDiagnostics)
    # Serializes the diagnostics read-modify-write across executor threads and
    # the event loop; without it concurrent requests lose/cross-mix updates.
    _diagnostics_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Serializes source-backed cache entries with their generation stamps.
    # Query ranking runs in executor threads while the event loop invalidates
    # these caches, so both halves of each logical entry must change atomically.
    _source_cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    intent_sources: dict[str, IntentSource] = field(default_factory=dict)
    language_intent_sources: dict[str, dict[str, IntentSource]] = field(default_factory=dict)
    config_path: Callable[..., str] | None = None
    registry_slot_values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    registry_slot_index: RegistrySlotIndex = field(default_factory=lambda: RegistrySlotIndex({}))
    registry_slot_indexes: dict[str, RegistrySlotIndex] = field(default_factory=dict)
    dynamic_registry_intents: dict[str, tuple[DynamicRegistryIntent, ...]] = field(
        default_factory=dict
    )
    # Generation stamps guard per-language caches against executor threads
    # publishing results computed from inputs that the event loop replaced
    # mid-build. A missing stamp means the entry was injected directly (tests,
    # tooling) and is trusted as current. _source_cache_lock prevents runtime
    # invalidation from exposing that same state transiently.
    _registry_slot_index_generations: dict[str, int] = field(default_factory=dict, repr=False)
    _language_source_generations: dict[str, int] = field(default_factory=dict, repr=False)
    _dynamic_intent_generations: dict[str, int] = field(default_factory=dict, repr=False)
    registry_generation: int = 0
    cleanup_callbacks: list[Callable[[], None]] = field(default_factory=list)
    rebuild_tasks: dict[str, tuple[IndexGeneration, asyncio.Task[CanonicalIndex | None]]] = field(
        default_factory=dict
    )
    warmup_tasks: set[asyncio.Task[object]] = field(default_factory=set, repr=False)
    _logged_rebuilds: dict[tuple[str, IndexGeneration], int] = field(
        default_factory=dict,
        repr=False,
    )
    index_generation: int = 0
    _language_index_generations: dict[str, int] = field(default_factory=dict, repr=False)
    source_generation: int = 0
    # Bumped only when intent sources themselves change (clear_sources=True),
    # unlike source_generation which also advances on registry-value changes.
    # Per-language intent-source caches stamp against this counter so registry
    # events do not spuriously invalidate them.
    intent_source_generation: int = 0
    rebuild_timer_cancel: Callable[[], None] | None = None
    _storage_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _active_index_loads: int = field(default=0, init=False, repr=False)
    _index_loads_drained: asyncio.Event = field(
        default_factory=_new_set_event,
        init=False,
        repr=False,
    )
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def closed(self) -> bool:
        """Return whether this runtime is shutting down or fully closed."""
        return self._closed

    def get_index(self, language: str) -> CanonicalIndex | None:
        """Return the cached index for a language."""
        return None if self._closed else self.indexes.get(normalize_language(language))

    def set_index(self, index: CanonicalIndex) -> None:
        """Store a language-specific index."""
        if self._closed:
            return
        self.indexes[normalize_language(index.language)] = index
        self.update_diagnostics(
            candidate_count=index.candidate_count,
            index_version=index.version,
        )

    def _active_rebuild_task(
        self,
        language: str,
        generation: IndexGeneration,
    ) -> asyncio.Task[CanonicalIndex | None] | None:
        """Return a reusable rebuild task and discard obsolete task state."""
        task_state = self.rebuild_tasks.get(language)
        if task_state is None:
            return None
        task_generation, task = task_state
        if task_generation != generation or task.done():
            self.rebuild_tasks.pop(language, None)
            return None
        return task

    def _start_rebuild_task(
        self,
        hass: HomeAssistant,
        language: str,
        generation: IndexGeneration,
    ) -> asyncio.Task[CanonicalIndex | None] | None:
        """Create and register a rebuild task unless shutdown has started."""
        if self._closed:
            return None
        self._logged_rebuilds.pop((language, generation), None)
        task = hass.async_create_task(_run_rebuild(self, hass, language, generation))
        self.rebuild_tasks[language] = (generation, task)
        return task

    def _log_rebuild_failure_once(
        self,
        language: str,
        generation: IndexGeneration,
        error: Exception,
        *,
        log_error: bool,
    ) -> None:
        """Log each rebuild generation once at its strongest requested severity."""
        key = (language, generation)
        logged_generations = tuple(
            logged_generation
            for logged_language, logged_generation in self._logged_rebuilds
            if logged_language == language
        )
        if any(logged_generation > generation for logged_generation in logged_generations):
            return
        for logged_generation in logged_generations:
            if logged_generation < generation:
                self._logged_rebuilds.pop((language, logged_generation), None)
        level = logging.ERROR if log_error else logging.INFO
        if self._logged_rebuilds.get(key, 0) >= level:
            return
        self._logged_rebuilds[key] = level
        _LOGGER.log(
            level,
            "Unexpected error rebuilding index for language %s",
            language,
            exc_info=error,
        )

    def _discard_finished_rebuild_task(
        self,
        language: str,
        task: asyncio.Task[CanonicalIndex | None],
    ) -> None:
        """Unregister only completed tasks so cancelled awaiters do not orphan work."""
        task_state = self.rebuild_tasks.get(language)
        if task_state and task_state[1] is task and task.done():
            self.rebuild_tasks.pop(language, None)

    async def async_rebuild_index(
        self,
        hass: HomeAssistant,
        language: str,
        *,
        log_error: bool = True,
        raise_on_error: bool = False,
    ) -> CanonicalIndex | None:
        """Rebuild one language index once while concurrent callers await it."""
        if self._closed:
            return None
        language = normalize_language(language)
        generation = self._index_generation_for(language)
        task = self._active_rebuild_task(language, generation)
        if task is None and (task := self._start_rebuild_task(hass, language, generation)) is None:
            return None

        try:
            return await task
        except asyncio.CancelledError:
            raise
        except Exception as err:
            self._log_rebuild_failure_once(
                language,
                generation,
                err,
                log_error=log_error,
            )
            if raise_on_error:
                raise
            return None
        finally:
            self._discard_finished_rebuild_task(language, task)

    async def async_load_index_from_store(
        self, hass: HomeAssistant, language: str
    ) -> CanonicalIndex | None:
        """Load an index only when its persisted source fingerprint is current."""
        if not self._start_index_load():
            return None
        try:
            return await self._async_load_index_from_store(hass, language)
        finally:
            self._finish_index_load()

    async def _async_load_index_from_store(
        self,
        hass: HomeAssistant,
        language: str,
    ) -> CanonicalIndex | None:
        """Load one persisted index while the public operation tracks its lifetime."""
        language = normalize_language(language)
        generation = self._index_generation_for(language)
        source_generation = self.source_generation
        build_inputs = self._capture_build_inputs()
        snapshot = await _async_add_executor_job_drained(
            hass,
            _create_build_snapshot_and_register_wildcards,
            language,
            *build_inputs,
        )
        if _index_load_invalidated(self, language, generation, source_generation):
            return None

        async with self._storage_lock:
            if _index_load_invalidated(self, language, generation, source_generation):
                return None
            candidates = await _async_load_persisted_candidates(
                self,
                hass,
                language,
                snapshot.fingerprint,
            )
            if candidates is None:
                return None

        index = await _async_add_executor_job_drained(hass, build_index, language, candidates)
        if _index_load_invalidated(self, language, generation, source_generation):
            return None
        with self._source_cache_lock:
            if _index_load_invalidated(self, language, generation, source_generation):
                return None
            self.language_intent_sources[language] = snapshot.intent_sources
            self._language_source_generations[language] = self.intent_source_generation
            self.dynamic_registry_intents[language] = snapshot.dynamic_registry_intents
            self._dynamic_intent_generations[language] = self.intent_source_generation
        self.set_index(index)
        return index

    async def async_save_index_to_store(
        self,
        hass: HomeAssistant,
        index: CanonicalIndex,
        fingerprint: str,
        *,
        expected_index_generation: IndexGeneration | None = None,
        expected_source_generation: int | None = None,
    ) -> bool:
        """Save an index with the source fingerprint that produced it."""
        if self._closed:
            return False
        language = normalize_language(index.language)
        candidates_data = await _async_add_executor_job_drained(
            hass,
            _serialize_candidates,
            index.candidates,
        )
        serialized_candidates: list[JsonValueType] = [*candidates_data]
        data: JsonObjectType = {
            "build_version": _INDEX_BUILD_VERSION,
            "language": language,
            "fingerprint": fingerprint,
            "candidate_count": len(candidates_data),
            "candidates": serialized_candidates,
        }

        async with self._storage_lock:
            if not self._storage_generation_matches(
                language,
                expected_index_generation,
                expected_source_generation,
            ):
                return False
            manifest = await self._async_load_store_manifest(hass)
            if manifest is None:
                cache_epoch = uuid4().hex
                persisted_languages: set[str] = set()
            else:
                cache_epoch, persisted_languages = manifest
            if not self._storage_generation_matches(
                language,
                expected_index_generation,
                expected_source_generation,
            ):
                return False

            data["cache_epoch"] = cache_epoch
            await _async_await_drained(_index_store(hass, language).async_save(data))
            if not self._storage_generation_matches(
                language,
                expected_index_generation,
                expected_source_generation,
            ):
                return False
            persisted_languages.add(language)
            await self._async_save_store_manifest(hass, cache_epoch, persisted_languages)
        return True

    async def async_clear_index(
        self,
        hass: HomeAssistant,
        language: str | None = None,
    ) -> IndexClearResult:
        """Clear indexes and return the cache state captured under the storage lock."""
        if self._closed:
            return IndexClearResult(
                cleared_cached_languages=(),
                cleared_candidate_count=0,
                remaining_candidate_count=self.total_candidate_count(),
                remaining_cached_languages=tuple(sorted(self.indexes)),
            )
        normalized_language = normalize_language(language) if language is not None else None
        async with self._storage_lock:
            cached_candidate_counts = {
                cached_language: index.candidate_count
                for cached_language, index in self.indexes.items()
            }
            cleared_cached_languages = (
                tuple(sorted(cached_candidate_counts))
                if normalized_language is None
                else (
                    (normalized_language,) if normalized_language in cached_candidate_counts else ()
                )
            )
            cleared_candidate_count = sum(
                cached_candidate_counts[cached_language]
                for cached_language in cleared_cached_languages
            )
            self.clear_index(normalized_language)
            result = IndexClearResult(
                cleared_cached_languages=cleared_cached_languages,
                cleared_candidate_count=cleared_candidate_count,
                remaining_candidate_count=self.total_candidate_count(),
                remaining_cached_languages=tuple(sorted(self.indexes)),
            )
            manifest = await self._async_load_store_manifest(hass)
            if normalized_language is not None:
                await _async_await_drained(_index_store(hass, normalized_language).async_remove())
                if manifest is not None:
                    cache_epoch, persisted_languages = manifest
                    persisted_languages.discard(normalized_language)
                    await self._async_save_store_manifest(
                        hass,
                        cache_epoch,
                        persisted_languages,
                    )
                return result

            persisted_languages = manifest[1] if manifest is not None else set()
            await self._async_save_store_manifest(hass, uuid4().hex, set())
            for persisted_language in persisted_languages:
                await _async_await_drained(_index_store(hass, persisted_language).async_remove())
            return result

    def rank_with_dynamic_candidates(
        self,
        language: str,
        index: CanonicalIndex,
        query: str,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        *,
        slot_preferences: set[tuple[str, str]] | None = None,
        intent_context: Mapping[str, object] | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_margin: float = DEFAULT_MIN_MARGIN,
    ) -> tuple[RankedCandidate, ...]:
        """Rank cached index candidates plus query-scoped registry expansions.

        ``intent_context`` is forwarded as the HassIL-style mapping consumed by
        indexed and dynamic ranking paths.
        """
        if self._closed:
            return ()
        language = normalize_language(language)
        ranked = index.rank(
            query,
            max_candidates=max_candidates,
            slot_preferences=slot_preferences,
            intent_context=intent_context,
            min_confidence=min_confidence,
        )
        self.update_diagnostics(dynamic_candidate_count=0)
        if _is_perfect_rank_result(ranked):
            return ranked
        intent_sources = self._intent_sources_for_query(language)

        registry_slot_values, registry_slot_index = self._registry_slot_snapshot_for_language(
            language
        )
        wildcard_literal_rescue, include_literal_only_templates = _literal_rescue_flags(
            ranked,
            query,
            language,
            min_confidence,
            min_margin,
        )
        dynamic_candidates = _build_and_filter_dynamic_candidates(
            self,
            language,
            intent_sources,
            registry_slot_values,
            registry_slot_index,
            query,
            include_literal_only_templates=include_literal_only_templates,
            wildcard_literal_rescue=wildcard_literal_rescue,
        )
        if not dynamic_candidates:
            return ranked
        self.update_diagnostics(dynamic_candidate_count=len(dynamic_candidates))
        dynamic_ranked = _rank_dynamic_candidates(
            query,
            dynamic_candidates,
            index,
            language,
            max_candidates,
            slot_preferences,
            intent_context,
            min_confidence,
        )
        return _merge_ranked_candidates(ranked, dynamic_ranked, max_candidates)

    def rank_and_evaluate(
        self,
        language: str,
        index: CanonicalIndex,
        query: str,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        *,
        slot_preferences: set[tuple[str, str]] | None = None,
        intent_context: Mapping[str, object] | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_margin: float = DEFAULT_MIN_MARGIN,
    ) -> tuple[tuple[RankedCandidate, ...], ConfidenceGateDecision]:
        """Rank candidates and evaluate confidence gates in one call.

        Bundling both steps lets the conversation entity run the full
        decision path inside a single executor job instead of evaluating
        pairwise gate checks on the event loop.
        """
        ranked = self.rank_with_dynamic_candidates(
            language,
            index,
            query,
            max_candidates,
            slot_preferences=slot_preferences,
            intent_context=intent_context,
            min_confidence=min_confidence,
            min_margin=min_margin,
        )
        decision = evaluate_confidence_gates(
            ranked,
            min_confidence=min_confidence,
            min_margin=min_margin,
            query=query,
            language=normalize_language(language),
        )
        return ranked, decision

    def clear_index(self, language: str | None = None) -> None:
        """Clear one language index or all indexes and invalidate active rebuilds."""
        if language is not None:
            language = normalize_language(language)
        if language is None:
            self.index_generation += 1
            self._language_index_generations.clear()
            self.indexes.clear()
            self.rebuild_tasks.clear()
        else:
            self._language_index_generations[language] = (
                self._language_index_generations.get(language, 0) + 1
            )
            self.rebuild_tasks.pop(language, None)
            self.indexes.pop(language, None)
        self.update_diagnostics(candidate_count=self.total_candidate_count())

    def total_candidate_count(self) -> int:
        """Return the total number of cached candidates."""
        return sum(index.candidate_count for index in self.indexes.values())

    def configure_config_path(self, config_path: Callable[..., str]) -> None:
        """Configure Home Assistant config path access for custom sentences."""
        with self._source_cache_lock:
            if self._closed:
                return
            self.config_path = config_path
            self._invalidate_source_dependent_indexes(clear_sources=True)

    def update_registry_slot_values(self, slot_values: Mapping[str, tuple[str, ...]]) -> bool:
        """Update cached registry metadata used for candidate expansion.

        Returns whether the cached values changed so callers can skip
        scheduling rebuilds after no-op registry events.
        """
        updated_values = {key: tuple(values) for key, values in slot_values.items()}
        with self._source_cache_lock:
            if self._closed or updated_values == self.registry_slot_values:
                return False
        updated_index = build_registry_slot_index(updated_values)
        registry_record_count = updated_index.record_count
        with self._source_cache_lock:
            if self._closed or updated_values == self.registry_slot_values:
                return False
            self.registry_generation += 1
            self.registry_slot_values = updated_values
            self.registry_slot_index = updated_index
            self.registry_slot_indexes.clear()
            self._registry_slot_index_generations.clear()
            self._invalidate_source_dependent_indexes(clear_sources=False)
            source_generation = self.source_generation
        self.update_diagnostics(
            registry_record_count=registry_record_count,
            registry_generation=source_generation,
            registry_fingerprint=hashlib.sha256(
                orjson.dumps(_canonical_fingerprint_value(updated_values))
            ).hexdigest(),
        )
        return True

    def update_intent_sources(self, intents_update: Mapping[object, IntentSource]) -> bool:
        """Merge changed Home Assistant conversation intent sources.

        Returns whether the merged sources changed so callers can skip
        scheduling rebuilds after no-op intent updates.
        """
        updated_sources = {
            _source_key(source): deepcopy(dict(source_config))
            for source, source_config in intents_update.items()
        }
        with self._source_cache_lock:
            if self._closed:
                return False
            merged_sources = dict(self.intent_sources) | updated_sources
            if merged_sources == self.intent_sources:
                return False
            self.intent_sources = merged_sources
            self._invalidate_source_dependent_indexes(clear_sources=True)
        return True

    def subscribed_source_counts(self) -> dict[str, int]:
        """Return subscribed intent counts by source without loading language files."""
        counts: dict[str, int] = {}
        for source_key, source_config in self.intent_sources.items():
            intents = source_config.get("intents", {})
            counts[source_key] = len(intents) if isinstance(intents, Mapping) else 0
        return counts

    def source_counts(self, language: str) -> dict[str, int]:
        """Return intent counts by source for diagnostics."""
        language = normalize_language(language)
        counts: dict[str, int] = {}
        for source_key, source_config in self._all_intent_sources(language).items():
            intents = source_config.get("intents", {})
            counts[source_key] = len(intents) if isinstance(intents, Mapping) else 0
        return counts

    def _intent_sources_for_query(self, language: str) -> dict[str, IntentSource]:
        """Return cached intent sources for query-time candidate expansion."""
        language = normalize_language(language)
        with self._source_cache_lock:
            if self._closed:
                return {}
            cached = self.language_intent_sources.get(language)
            if cached is not None and self._generation_stamp_current(
                self._language_source_generations, language, self.intent_source_generation
            ):
                return cached
        return self._all_intent_sources(language)

    def _dynamic_registry_intents_for_query(
        self, language: str
    ) -> tuple[DynamicRegistryIntent, ...]:
        """Return compiled query-independent registry templates for a language."""
        language = normalize_language(language)
        with self._source_cache_lock:
            if self._closed:
                return ()
            cached = self.dynamic_registry_intents.get(language)
            if cached is not None and self._generation_stamp_current(
                self._dynamic_intent_generations, language, self.intent_source_generation
            ):
                return cached
            generation = self.intent_source_generation
        compiled = compile_dynamic_registry_intents(
            self._intent_sources_for_query(language),
            language,
            include_literal_only_templates=True,
            include_area_only_templates=False,
        )
        with self._source_cache_lock:
            if not self._closed and self.intent_source_generation == generation:
                self.dynamic_registry_intents[language] = compiled
                self._dynamic_intent_generations[language] = generation
        return compiled

    @staticmethod
    def _generation_stamp_current(
        stamps: Mapping[str, int],
        language: str,
        current_generation: int,
    ) -> bool:
        """Return whether a cached per-language entry belongs to current inputs.

        Entries without a stamp were injected directly and are trusted.
        """
        stamp = stamps.get(language)
        return stamp is None or stamp == current_generation

    def _registry_slot_index_for_language(self, language: str) -> RegistrySlotIndex:
        """Return registry slot records normalized for the query language."""
        return self._registry_slot_snapshot_for_language(language)[1]

    def _registry_slot_snapshot_for_language(
        self,
        language: str,
    ) -> tuple[dict[str, tuple[str, ...]], RegistrySlotIndex]:
        """Return one registry values snapshot with its matching language index."""
        language = normalize_language(language)
        with self._source_cache_lock:
            generation = self.registry_generation
            # Values are stored as tuples by update_registry_slot_values; a
            # plain dict copy keeps snapshot isolation from shutdown's clear()
            # without re-wrapping every value per request.
            registry_slot_values = dict(self.registry_slot_values)
            cached = self.registry_slot_indexes.get(language)
            if cached is not None and self._generation_stamp_current(
                self._registry_slot_index_generations, language, generation
            ):
                return registry_slot_values, cached
        built = build_registry_slot_index(registry_slot_values, language)
        with self._source_cache_lock:
            if not self._closed and self.registry_generation == generation:
                self.registry_slot_indexes[language] = built
                self._registry_slot_index_generations[language] = generation
        return registry_slot_values, built

    def _all_intent_sources(self, language: str) -> dict[str, IntentSource]:
        """Return built-in, custom, and subscribed intent sources."""
        language = normalize_language(language)
        with self._source_cache_lock:
            if self._closed:
                return {}
            generation = self.intent_source_generation
            config_path = self.config_path
            intent_sources = dict(self.intent_sources)
        sources = load_language_intent_sources(language, config_path=config_path)
        sources.update(intent_sources)
        with self._source_cache_lock:
            if not self._closed and self.intent_source_generation == generation:
                self.language_intent_sources[language] = sources
                self._language_source_generations[language] = generation
        return sources

    def _capture_build_inputs(
        self,
    ) -> tuple[
        Callable[..., str] | None,
        dict[str, Mapping[str, object]],
        dict[str, tuple[str, ...]],
    ]:
        """Copy mutable runtime inputs before executor work begins."""
        with self._source_cache_lock:
            return (
                self.config_path,
                deepcopy(self.intent_sources),
                {key: tuple(values) for key, values in self.registry_slot_values.items()},
            )

    def _invalidate_source_dependent_indexes(self, *, clear_sources: bool) -> None:
        """Invalidate source caches while the caller holds _source_cache_lock."""
        self.source_generation += 1
        self.indexes.clear()
        if clear_sources:
            self.intent_source_generation += 1
            self.language_intent_sources.clear()
            self._language_source_generations.clear()
            self.dynamic_registry_intents.clear()
            self._dynamic_intent_generations.clear()
        self.update_diagnostics(candidate_count=0)

    def _storage_generation_matches(
        self,
        language: str,
        expected_index_generation: IndexGeneration | None,
        expected_source_generation: int | None,
    ) -> bool:
        """Return whether a pending store write still belongs to current inputs."""
        return (
            not self._closed
            and (
                expected_index_generation is None
                or self._index_generation_for(language) == expected_index_generation
            )
            and (
                expected_source_generation is None
                or self.source_generation == expected_source_generation
            )
        )

    def _index_generation_for(self, language: str) -> IndexGeneration:
        """Return the global and language-scoped invalidation generation."""
        language = normalize_language(language)
        return (
            self.index_generation,
            self._language_index_generations.get(language, 0),
        )

    def _start_index_load(self) -> bool:
        """Register an index load unless runtime shutdown has started."""
        if self._closed:
            return False
        if self._active_index_loads == 0:
            self._index_loads_drained.clear()
        self._active_index_loads += 1
        return True

    def _finish_index_load(self) -> None:
        """Release one index load and signal when every load has drained."""
        self._active_index_loads -= 1
        if self._active_index_loads == 0:
            self._index_loads_drained.set()

    async def _async_load_store_manifest(self, hass: HomeAssistant) -> tuple[str, set[str]] | None:
        """Load the cache epoch and known persisted language keys."""
        try:
            data = await _async_await_drained(_manifest_store(hass).async_load())
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        cache_epoch = data.get("cache_epoch")
        languages = data.get("languages")
        if not isinstance(cache_epoch, str) or not cache_epoch:
            return None
        if not isinstance(languages, list) or not all(
            isinstance(language, str) for language in languages
        ):
            return None
        return cache_epoch, {
            normalize_language(language) for language in languages if isinstance(language, str)
        }

    async def _async_save_store_manifest(
        self,
        hass: HomeAssistant,
        cache_epoch: str,
        languages: set[str],
    ) -> None:
        """Persist the cache epoch and known language store keys."""
        serialized_languages: list[JsonValueType] = [*sorted(languages)]
        manifest_data: JsonObjectType = {
            "cache_epoch": cache_epoch,
            "languages": serialized_languages,
        }
        await _async_await_drained(_manifest_store(hass).async_save(manifest_data))

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Remember a cleanup callback for unload."""
        if self._closed:
            callback()
            return
        self.cleanup_callbacks.append(callback)

    def track_warmup_task(self, task: object) -> None:
        """Track a warmup task so shutdown can cancel and drain it."""
        if not isinstance(task, asyncio.Task):
            return
        if self._closed:
            task.cancel()
            return
        self.warmup_tasks.add(task)
        task.add_done_callback(lambda done_task: self.warmup_tasks.discard(done_task))

    async def async_shutdown(self) -> None:
        """Cancel runtime-owned work, wait for storage, and clear caches."""
        tasks = self._begin_shutdown()
        current_task = _current_task_or_none()
        if await_tasks := tuple(task for task in tasks if task is not current_task):
            await asyncio.gather(*await_tasks, return_exceptions=True)
        await self._index_loads_drained.wait()
        async with self._storage_lock:
            pass
        self._clear_runtime_state_and_caches()

    def cleanup(self) -> None:
        """Run registered cleanup callbacks and purge global module-level caches.

        Caches across normalization, BM25 indices, rehydration matching, utility wildcards,
        and grammar loaders are cleared here during unload or reload. This ensures that
        stale sentence configurations, custom wildcard registries, and parsed templates
        do not leak across lifecycle re-initializations (e.g., when the user reloads the
        integration or updates configuration options flow).

        Ordering: timers and callbacks are torn down first so that no in-flight work
        can observe partially cleared cache state.
        """
        self._begin_shutdown()
        self._clear_runtime_state_and_caches()

    def _begin_shutdown(self) -> tuple[asyncio.Task[object], ...]:
        """Start shutdown synchronously and return runtime-owned tasks to drain."""
        if not self._closed:
            self._closed = True
            self.index_generation += 1
            self.source_generation += 1
        if self.rebuild_timer_cancel is not None:
            self.rebuild_timer_cancel()
            self.rebuild_timer_cancel = None
        callbacks = list(self.cleanup_callbacks)
        self.cleanup_callbacks.clear()
        for callback in callbacks:
            callback()
        tasks: set[asyncio.Task[object]] = set(self.warmup_tasks)
        tasks.update(task for _generation, task in self.rebuild_tasks.values())
        self.warmup_tasks.clear()
        self.rebuild_tasks.clear()
        self._logged_rebuilds.clear()
        current_task = _current_task_or_none()
        for task in tasks:
            if task is not current_task and not task.done():
                task.cancel()
        return tuple(tasks)

    def _clear_runtime_state_and_caches(self) -> None:
        """Clear runtime-owned collections and shared module-level caches."""
        with self._source_cache_lock:
            self.indexes.clear()
            self.intent_sources.clear()
            self.language_intent_sources.clear()
            self._language_source_generations.clear()
            self.registry_slot_values.clear()
            self.registry_slot_index = RegistrySlotIndex({})
            self.registry_slot_indexes.clear()
            self._registry_slot_index_generations.clear()
            self.dynamic_registry_intents.clear()
            self._dynamic_intent_generations.clear()
            self._language_index_generations.clear()
            self.rebuild_tasks.clear()
            self.warmup_tasks.clear()
        self.update_diagnostics(candidate_count=0, dynamic_candidate_count=0)
        clear_normalization_caches()
        clear_bm25_caches()
        clear_rehydration_caches()
        clear_utils_caches()
        clear_grammar_loader_caches()
        clear_ranking_caches()
        clear_builtin_intents_caches()

    def update_diagnostics(
        self,
        *,
        candidate_count: int | None = None,
        index_version: int | None = None,
        last_query_latency_ms: float | None = None,
        last_fallback_reason: FallbackReason | str | None = None,
        last_error: str | None = None,
        dynamic_candidate_count: int | None = None,
        last_request_id: str | None = None,
        selected_delegated_text_hash: str | None = None,
        selected_candidate_source: str | None = None,
        confidence_gate: Mapping[str, JsonValueType] | None = None,
        execution_result: str | None = None,
        recognition_kind: str | None = None,
        recognition_intent: str | None = None,
        recognition_unmatched_count: int | None = None,
        recognition_latency_ms: float | None = None,
        preflight_attempt_count: int | None = None,
        metadata_diverged: bool | None = None,
        metadata_intent_matches_observed: bool | None = None,
        metadata_slots_match_observed: bool | None = None,
        metadata_divergence_reason: str | None = None,
        recovery_used: bool | None = None,
        registry_record_count: int | None = None,
        registry_generation: int | None = None,
        registry_fingerprint: str | None = None,
        registry_postings_consulted: int | None = None,
        registry_values_nominated: int | None = None,
        registry_values_scored: int | None = None,
        fuzzy_dynamic_candidates: int | None = None,
        registry_retrieval_latency_ms: float | None = None,
        selected_from_fuzzy_registry: bool | None = None,
        clear_last_fallback_reason: bool = False,
        clear_last_error: bool = False,
        clear_request_trace: bool = False,
        clear_recognition_trace: bool = False,
    ) -> None:
        """Update the diagnostics snapshot."""
        with self._diagnostics_lock:
            diagnostics = _cleared_diagnostic_traces(
                self.diagnostics,
                clear_request_trace=clear_request_trace,
                clear_recognition_trace=clear_recognition_trace,
            )
            updates: dict[str, object] = {
                key: value
                for key, value in {
                    "candidate_count": candidate_count,
                    "index_version": index_version,
                    "last_query_latency_ms": last_query_latency_ms,
                    "dynamic_candidate_count": dynamic_candidate_count,
                    "last_request_id": last_request_id,
                    "selected_delegated_text_hash": selected_delegated_text_hash,
                    "selected_candidate_source": selected_candidate_source,
                    "confidence_gate": dict(confidence_gate)
                    if confidence_gate is not None
                    else None,
                    "execution_result": execution_result,
                    "recognition_kind": recognition_kind,
                    "recognition_intent": recognition_intent,
                    "recognition_unmatched_count": recognition_unmatched_count,
                    "recognition_latency_ms": recognition_latency_ms,
                    "preflight_attempt_count": preflight_attempt_count,
                    "metadata_diverged": metadata_diverged,
                    "metadata_intent_matches_observed": metadata_intent_matches_observed,
                    "metadata_slots_match_observed": metadata_slots_match_observed,
                    "metadata_divergence_reason": metadata_divergence_reason,
                    "recovery_used": recovery_used,
                    "registry_record_count": registry_record_count,
                    "registry_generation": registry_generation,
                    "registry_fingerprint": registry_fingerprint,
                    "registry_postings_consulted": registry_postings_consulted,
                    "registry_values_nominated": registry_values_nominated,
                    "registry_values_scored": registry_values_scored,
                    "fuzzy_dynamic_candidates": fuzzy_dynamic_candidates,
                    "registry_retrieval_latency_ms": registry_retrieval_latency_ms,
                    "selected_from_fuzzy_registry": selected_from_fuzzy_registry,
                }.items()
                if value is not None
            }
            updates["last_fallback_reason"] = _updated_optional_text(
                current=diagnostics.last_fallback_reason,
                value=last_fallback_reason,
                clear=clear_last_fallback_reason,
            )
            updates["last_error"] = _updated_optional_text(
                current=diagnostics.last_error,
                value=last_error,
                clear=clear_last_error,
            )
            self.diagnostics = replace(diagnostics, **updates)


def _publish_rebuilt_index(
    runtime: CanonicalizerRuntime,
    language: str,
    generation: IndexGeneration,
    source_generation: int,
    snapshot: IndexBuildSnapshot,
    index: CanonicalIndex,
) -> bool | None:
    """Publish a stable rebuild, returning None when it must stop or False to retry."""
    with runtime._source_cache_lock:
        if runtime.closed or runtime._index_generation_for(language) != generation:
            return None
        if runtime.source_generation != source_generation:
            return False
        runtime.language_intent_sources[language] = snapshot.intent_sources
        runtime._language_source_generations[language] = runtime.intent_source_generation
        runtime.dynamic_registry_intents[language] = snapshot.dynamic_registry_intents
        runtime._dynamic_intent_generations[language] = runtime.intent_source_generation
    runtime.set_index(index)
    return True


async def _run_rebuild_attempt(
    runtime: CanonicalizerRuntime,
    hass: HomeAssistant,
    language: str,
    generation: IndexGeneration,
) -> tuple[CanonicalIndex | None, bool]:
    """Run one rebuild attempt.

    Returns:
        A pair containing a published index when successful and whether the
        caller should retry after source instability.
    """
    if runtime.closed:
        return None, False
    source_generation = runtime.source_generation
    build_inputs = runtime._capture_build_inputs()
    snapshot = await _async_add_executor_job_drained(
        hass,
        _create_build_snapshot_and_register_wildcards,
        language,
        *build_inputs,
    )
    if runtime.closed:
        return None, False
    index = await _async_add_executor_job_drained(
        hass,
        _build_index_from_snapshot,
        snapshot,
    )
    if runtime.closed or runtime._index_generation_for(language) != generation:
        return None, False
    if runtime.source_generation != source_generation:
        return None, True

    saved = await runtime.async_save_index_to_store(
        hass,
        index,
        snapshot.fingerprint,
        expected_index_generation=generation,
        expected_source_generation=source_generation,
    )
    if runtime.closed or runtime._index_generation_for(language) != generation:
        return None, False
    if runtime.source_generation != source_generation or not saved:
        return None, True

    published = _publish_rebuilt_index(
        runtime,
        language,
        generation,
        source_generation,
        snapshot,
        index,
    )
    return (index, False) if published else (None, published is False)


async def _run_rebuild(
    runtime: CanonicalizerRuntime,
    hass: HomeAssistant,
    language: str,
    generation: IndexGeneration,
) -> CanonicalIndex | None:
    """Retry index construction until one stable source generation is published."""
    try:
        for _ in range(_MAX_REBUILD_ATTEMPTS):
            index, retry = await _run_rebuild_attempt(runtime, hass, language, generation)
            if index is not None or not retry:
                return index
    except asyncio.CancelledError:
        raise
    except Exception as err:
        runtime.update_diagnostics(last_error=str(err))
        raise
    return None


def _create_build_snapshot_and_register_wildcards(
    language: str,
    config_path: Callable[..., str] | None,
    subscribed_sources: dict[str, Mapping[str, object]],
    registry_slot_values: dict[str, tuple[str, ...]],
) -> IndexBuildSnapshot:
    """Load sources, register in-memory wildcards, and fingerprint build inputs."""
    sources = load_language_intent_sources(language, config_path=config_path)
    sources.update(subscribed_sources)
    register_custom_wildcards_from_sources(language, sources)
    dynamic_registry_intents = compile_dynamic_registry_intents(
        sources,
        language,
        include_literal_only_templates=True,
        include_area_only_templates=False,
    )
    fingerprint_payload = {
        "build_version": _INDEX_BUILD_VERSION,
        "language": language,
        "intent_sources": sources,
        "registry_slot_values": registry_slot_values,
    }
    fingerprint = hashlib.sha256(
        orjson.dumps(_canonical_fingerprint_value(fingerprint_payload))
    ).hexdigest()
    return IndexBuildSnapshot(
        language=language,
        intent_sources=sources,
        registry_slot_values=registry_slot_values,
        dynamic_registry_intents=dynamic_registry_intents,
        fingerprint=fingerprint,
    )


def _build_index_from_snapshot(snapshot: IndexBuildSnapshot) -> CanonicalIndex:
    """Build an index from the inputs covered by a source fingerprint."""
    candidates = build_candidates_from_intent_sources(
        snapshot.language,
        snapshot.intent_sources,
        snapshot.registry_slot_values,
    )
    return build_index(snapshot.language, candidates)


def _index_load_invalidated(
    runtime: CanonicalizerRuntime,
    language: str,
    generation: IndexGeneration,
    source_generation: int,
) -> bool:
    """Return whether shutdown or newer inputs invalidated a pending index load."""
    return (
        runtime._closed
        or runtime._index_generation_for(language) != generation
        or runtime.source_generation != source_generation
    )


async def _async_discard_persisted_index(
    runtime: CanonicalizerRuntime,
    hass: HomeAssistant,
    store: storage.Store[JsonObjectType],
    language: str,
    cache_epoch: str,
    persisted_languages: set[str],
) -> None:
    """Remove one stale persisted index and update the manifest.

    Must be called while holding the runtime storage lock.
    """
    await _async_await_drained(store.async_remove())
    persisted_languages.discard(language)
    await runtime._async_save_store_manifest(
        hass,
        cache_epoch,
        persisted_languages,
    )


async def _async_load_persisted_candidates(
    runtime: CanonicalizerRuntime,
    hass: HomeAssistant,
    language: str,
    fingerprint: str,
) -> list[Candidate] | None:
    """Load and validate persisted candidates for one language, or None.

    Must be called while holding the runtime storage lock so manifest reads,
    store reads, and stale-store cleanup stay atomic against concurrent
    storage writers. Invalid or mismatched stores are discarded.
    """
    manifest = await runtime._async_load_store_manifest(hass)
    if manifest is None:
        return None
    cache_epoch, persisted_languages = manifest
    if language not in persisted_languages:
        return None
    store = _index_store(hass, language)
    try:
        data = await _async_await_drained(store.async_load())
    except Exception:
        return None
    if data is None:
        return None

    if not _valid_store_metadata(
        data,
        language=language,
        fingerprint=fingerprint,
        cache_epoch=cache_epoch,
    ):
        await _async_discard_persisted_index(
            runtime,
            hass,
            store,
            language,
            cache_epoch,
            persisted_languages,
        )
        return None
    candidates = await _async_add_executor_job_drained(
        hass,
        _deserialize_candidates,
        data,
    )
    if candidates is None or len(candidates) != data["candidate_count"]:
        await _async_discard_persisted_index(
            runtime,
            hass,
            store,
            language,
            cache_epoch,
            persisted_languages,
        )
        return None
    return candidates


def _literal_rescue_flags(
    ranked: tuple[RankedCandidate, ...],
    query: str,
    language: str,
    min_confidence: float,
    min_margin: float,
) -> tuple[bool, bool]:
    """Return wildcard-rescue and literal-only-template inclusion flags."""
    static_decision = evaluate_confidence_gates(
        ranked,
        min_confidence=min_confidence,
        min_margin=min_margin,
        query=query,
        language=language,
    )
    static_accepted = static_decision.accepted_candidate
    numeric_literal_rescue = _query_needs_literal_only_dynamic(query)
    wildcard_literal_rescue = static_accepted is None and not numeric_literal_rescue
    include_literal_only_templates = numeric_literal_rescue or wildcard_literal_rescue
    return wildcard_literal_rescue, include_literal_only_templates


def _build_and_filter_dynamic_candidates(
    runtime: CanonicalizerRuntime,
    language: str,
    intent_sources: dict[str, IntentSource],
    registry_slot_values: dict[str, tuple[str, ...]],
    registry_slot_index: RegistrySlotIndex,
    query: str,
    *,
    include_literal_only_templates: bool,
    wildcard_literal_rescue: bool,
) -> tuple[Candidate, ...]:
    """Build query-scoped registry candidates and publish retrieval diagnostics.

    During a wildcard literal rescue, only candidates that carry a wildcard or
    query-derived slots are kept.
    """
    retrieval_stats = RegistryRetrievalStats(record_count=registry_slot_index.record_count)
    retrieval_started_at = time.monotonic()
    dynamic_candidates = build_query_registry_candidates(
        language,
        intent_sources,
        registry_slot_values,
        query,
        registry_slot_index=registry_slot_index,
        compiled_intents=runtime._dynamic_registry_intents_for_query(language),
        include_literal_only_templates=include_literal_only_templates,
        include_area_only_templates=False,
        literal_only_wildcards_only=wildcard_literal_rescue,
        retrieval_stats=retrieval_stats,
    )
    runtime.update_diagnostics(
        registry_record_count=retrieval_stats.record_count,
        registry_postings_consulted=retrieval_stats.postings_consulted,
        registry_values_nominated=retrieval_stats.values_nominated,
        registry_values_scored=retrieval_stats.values_scored,
        fuzzy_dynamic_candidates=retrieval_stats.fuzzy_dynamic_candidates,
        registry_retrieval_latency_ms=elapsed_ms(retrieval_started_at),
    )
    if wildcard_literal_rescue:
        dynamic_candidates = tuple(
            candidate
            for candidate in dynamic_candidates
            if candidate.has_wildcard or candidate.metadata.get("query_slots")
        )
    return dynamic_candidates


def _rank_dynamic_candidates(
    query: str,
    dynamic_candidates: tuple[Candidate, ...],
    index: CanonicalIndex,
    language: str,
    max_candidates: int,
    slot_preferences: set[tuple[str, str]] | None,
    intent_context: Mapping[str, object] | None,
    min_confidence: float,
) -> tuple[RankedCandidate, ...]:
    """Rank dynamic candidates via an exact-match shortcut or a char n-gram index."""
    exact_normalized_lookup = _dynamic_exact_normalized_lookup(query, dynamic_candidates)
    if exact_normalized_lookup is not None:
        matches = next(iter(exact_normalized_lookup.values()))
        if len(matches) > 1:
            exact_normalized_lookup = None
    dynamic_char_index = (
        None
        if exact_normalized_lookup is not None
        else CharNGramIndex.from_grams(
            tuple(
                char_ngrams_normalized(candidate.normalized_text)
                for candidate in dynamic_candidates
            )
        )
    )
    return rank_candidates(
        query,
        dynamic_candidates,
        max_candidates=max_candidates,
        reference_bm25_index=index.bm25_index,
        candidate_char_index=dynamic_char_index,
        exact_normalized_lookup=exact_normalized_lookup,
        positional_literal_tokens=index.positional_literal_tokens,
        language=language,
        reference_slot_token_index=index.slot_token_index,
        slot_preferences=slot_preferences,
        intent_context=intent_context,
        min_confidence=min_confidence,
    )


def _current_task_or_none() -> asyncio.Task[object] | None:
    """Return the current asyncio task when called inside a running loop."""
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


async def _async_add_executor_job_drained[R](
    hass: HomeAssistant, func: Callable[..., R], *args: object
) -> R:
    """Run teardown-critical executor work before honoring cancellation.

    Use only for runtime-owned work that must finish before shutdown releases a
    storage or lifecycle barrier. Do not use it for request-scoped work that is
    safe to abandon, because cancellation waits for the executor job to finish.
    """
    return await _async_await_drained(hass.async_add_executor_job(func, *args))


async def _async_await_drained[T](awaitable: Awaitable[T]) -> T:
    """Await work to completion before propagating cancellation to the caller.

    Shielding means cancellation cannot stop executor-backed storage or index
    work. During shutdown, keep awaiting that work so its caller cannot release
    a lifecycle barrier while the underlying operation can still mutate state.
    The awaitable must therefore be safe to complete after cancellation.

    Catching one cancellation permits the next await to block normally. The
    loop retries only when another cancellation is delivered, and must retain
    that behavior so shutdown still drains the underlying work before raising.
    """
    job = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(job)
    except asyncio.CancelledError:
        while not job.done():
            try:
                await asyncio.shield(job)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not job.cancelled():
            with contextlib.suppress(Exception):
                job.result()
        raise


def _canonical_fingerprint_value(value: object) -> JsonValueType:
    """Return a deterministic, order-preserving representation for hashing."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return _canonical_fingerprint_value(value.value)
    if isinstance(value, Mapping):
        canonical_items = [
            (
                _canonical_fingerprint_value(key),
                _canonical_fingerprint_value(item),
            )
            for key, item in value.items()
        ]
        canonical_items.sort(key=lambda pair: orjson.dumps(pair[0]))
        return {"mapping": [[key, item] for key, item in canonical_items]}
    if isinstance(value, Sequence):
        return {"sequence": [_canonical_fingerprint_value(item) for item in value]}
    if isinstance(value, Set):
        canonical_items = [_canonical_fingerprint_value(item) for item in value]
        canonical_items.sort(key=orjson.dumps)
        return {"set": canonical_items}
    return {
        "object_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "representation": repr(value),
    }


def _index_store(hass: HomeAssistant, language: str) -> storage.Store[JsonObjectType]:
    """Return the versioned Home Assistant Store for one language index."""
    if _STORE_HAS_SERIALIZE_IN_EVENT_LOOP:
        return storage.Store(
            hass,
            _INDEX_STORE_VERSION,
            f"{_INDEX_STORE_PREFIX}{language}",
            serialize_in_event_loop=False,
        )
    return storage.Store(hass, _INDEX_STORE_VERSION, f"{_INDEX_STORE_PREFIX}{language}")


def _manifest_store(hass: HomeAssistant) -> storage.Store[JsonObjectType]:
    """Return the Store tracking the current cache epoch and language keys."""
    return storage.Store(hass, _INDEX_MANIFEST_VERSION, _INDEX_MANIFEST_KEY)


def _valid_store_metadata(
    data: object,
    *,
    language: str,
    fingerprint: str,
    cache_epoch: str,
) -> bool:
    """Return whether persisted metadata matches the current build inputs."""
    if not isinstance(data, dict):
        return False
    candidates = data.get("candidates")
    candidate_count = data.get("candidate_count")
    return (
        data.get("build_version") == _INDEX_BUILD_VERSION
        and data.get("language") == language
        and data.get("fingerprint") == fingerprint
        and data.get("cache_epoch") == cache_epoch
        and type(candidate_count) is int
        and candidate_count >= 0
        and isinstance(candidates, list)
        and candidate_count == len(candidates)
    )


def _serialize_candidates(candidates: Sequence[Candidate]) -> list[JsonObjectType]:
    """Serialize all index candidates in one executor job to spare the event loop."""
    return [_serialize_candidate(candidate) for candidate in candidates]


def _serialize_candidate(candidate: Candidate) -> JsonObjectType:
    """Return the persisted representation of one candidate.

    ``wildcard_infos`` is persisted so store-loaded candidates keep the exact
    template-derived wildcard positions instead of re-deriving them through the
    lossy affix heuristic, which can flag literal tokens such as "playlist"
    for a ``{list}`` wildcard.
    """
    return {
        "text": candidate.text,
        "intent_name": candidate.intent_name,
        "source": candidate.source.value,
        "language": candidate.language,
        "metadata": dict(candidate.metadata),
        "slot_values": list(candidate.slot_values),
        "normalized_text": candidate.normalized_text,
        "wildcard_infos": [list(info) for info in candidate.wildcard_infos],
    }


def _deserialized_wildcard_infos(value: object) -> tuple[tuple[int, str], ...] | None:
    """Validate and convert persisted wildcard infos, or None when invalid."""
    if not isinstance(value, list | tuple):
        return None
    infos: list[tuple[int, str]] = []
    for entry in value:
        if (
            not isinstance(entry, list | tuple)
            or len(entry) != 2
            or not isinstance(entry[0], int)
            or isinstance(entry[0], bool)
            or not isinstance(entry[1], str)
        ):
            return None
        infos.append((entry[0], entry[1]))
    return tuple(infos)


def _deserialize_candidates(data: Mapping[str, object]) -> list[Candidate] | None:
    """Deserialize all candidates, rejecting the whole cache on any invalid record."""
    serialized_candidates = data.get("candidates")
    if not isinstance(serialized_candidates, list):
        return None
    candidates: list[Candidate] = []
    for candidate_data in serialized_candidates:
        if not isinstance(candidate_data, dict):
            return None
        text = candidate_data.get("text")
        intent_name = candidate_data.get("intent_name")
        source = candidate_data.get("source")
        language = candidate_data.get("language")
        metadata = candidate_data.get("metadata")
        slot_values = candidate_data.get("slot_values", ())
        normalized_text = candidate_data.get("normalized_text")
        wildcard_infos = _deserialized_wildcard_infos(candidate_data.get("wildcard_infos"))
        if (
            not isinstance(text, str)
            or not isinstance(intent_name, str)
            or not isinstance(source, str)
            or (language is not None and not isinstance(language, str))
            or not isinstance(metadata, Mapping)
            or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
            )
            or not isinstance(slot_values, list | tuple)
            or not all(isinstance(value, str) for value in slot_values)
            or not isinstance(normalized_text, str)
            or not normalized_text
            or wildcard_infos is None
        ):
            return None
        try:
            typed_metadata = {
                key: value
                for key, value in metadata.items()
                if isinstance(key, str) and isinstance(value, str)
            }
            typed_slot_values = tuple(value for value in slot_values if isinstance(value, str))
            candidate = Candidate(
                text=text,
                intent_name=intent_name,
                source=CandidateSource(source),
                language=language,
                metadata=typed_metadata,
                slot_values=typed_slot_values,
                normalized_text=normalized_text,
            )
        except (TypeError, ValueError):
            return None
        object.__setattr__(candidate, "_wildcard_infos", wildcard_infos)
        candidates.append(candidate)
    return candidates


def _merge_ranked_candidates(
    primary: tuple[RankedCandidate, ...],
    dynamic: tuple[RankedCandidate, ...],
    max_candidates: int,
) -> tuple[RankedCandidate, ...]:
    """Merge ranked candidates while keeping the strongest score per text and intent."""
    selected: dict[tuple[str, str], RankedCandidate] = {}
    for ranked_candidate in (*primary, *dynamic):
        key = (
            ranked_candidate.candidate.normalized_text,
            ranked_candidate.candidate.intent_name,
        )
        existing = selected.get(key)
        if existing is None or _ranked_candidate_sort_key(
            ranked_candidate
        ) > _ranked_candidate_sort_key(existing):
            selected[key] = ranked_candidate
    ranked = sorted(
        selected.values(),
        key=_ranked_candidate_sort_key,
        reverse=True,
    )
    limited = list(_limit_ranked_candidates(ranked, max_candidates))
    # Within-pass tie handling ran before the merge; re-resolve an exact
    # cross-pass final-score tie between opposing intents the same way.
    apply_intent_disambiguation(limited)
    return tuple(limited)


def _is_perfect_rank_result(ranked: tuple[RankedCandidate, ...]) -> bool:
    """Return whether ranking found only exact lexical matches."""
    if not ranked:
        return False
    for item in ranked:
        scores = item.scores
        if (
            scores.rapidfuzz_score != 1.0
            or scores.char_ngram_score != 1.0
            or scores.bm25_score != 1.0
            or scores.intent_score != 1.0
            or scores.final_score != 1.0
            or scores.penalty != 0.0
        ):
            return False
    return True


def _dynamic_exact_normalized_lookup(
    query: str,
    candidates: tuple[Candidate, ...],
) -> dict[str, list[Candidate]] | None:
    """Return an exact normalized lookup for dynamic candidates matching a query."""
    normalized_query = normalize_text(query)
    exact_matches = [
        candidate for candidate in candidates if candidate.normalized_text == normalized_query
    ]
    return {normalized_query: exact_matches} if exact_matches else None


def _query_needs_literal_only_dynamic(query: str) -> bool:
    """Return whether static caps are likely to miss exact base-list expansions."""
    return any(any(char.isdigit() for char in token) for token in normalize_text(query).split())


def _ranked_candidate_sort_key(ranked_candidate: RankedCandidate) -> tuple[float, int]:
    """Return a deterministic ranking key for merged candidates."""
    return (
        ranked_candidate.scores.final_score,
        -ranked_candidate.candidate.source_priority,
    )


def _updated_optional_text(current: str | None, value: str | None, *, clear: bool) -> str | None:
    """Return an updated optional diagnostics string.

    When ``clear`` is True the field is unconditionally replaced by ``value``
    (which may be ``None`` to reset it).  When ``clear`` is False the field is
    only updated if an explicit ``value`` is provided; passing ``value=None``
    with ``clear=False`` preserves the current value.
    """
    if clear:
        return value
    return current if value is None else value


def _source_key(source: object) -> str:
    """Return a stable string key for a Home Assistant intent source."""
    name = getattr(source, "name", None)
    return name.lower() if isinstance(name, str) else str(source).lower()
