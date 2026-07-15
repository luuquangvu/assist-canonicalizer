"""Runtime state for Assist Canonicalizer."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

import orjson
from homeassistant.helpers import storage

from .bm25 import clear_bm25_caches
from .builtin_intents import load_language_intent_sources
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
    RankedCandidate,
    accepted_candidate,
    clear_ranking_caches,
    rank_candidates,
)
from .rehydration import clear_rehydration_caches
from .utils import (
    clear_utils_caches,
    normalize_language,
    register_custom_wildcards_from_sources,
    wildcard_slot_names,
    wildcard_slot_names_sorted,
)

_LOGGER = logging.getLogger(__name__)


_STORE_HAS_SERIALIZE_IN_EVENT_LOOP = False
with contextlib.suppress(Exception):
    sig = inspect.signature(storage.Store.__init__)
    _STORE_HAS_SERIALIZE_IN_EVENT_LOOP = "serialize_in_event_loop" in sig.parameters
_INDEX_STORE_VERSION = 1
_INDEX_BUILD_VERSION = 4
_INDEX_STORE_PREFIX = f"{DOMAIN}.index_"
_INDEX_MANIFEST_KEY = f"{DOMAIN}.index_manifest"
_INDEX_MANIFEST_VERSION = 1
_MAX_REBUILD_ATTEMPTS = 5


def _new_set_event() -> asyncio.Event:
    """Return an asyncio event initialized in the set state."""
    event = asyncio.Event()
    event.set()
    return event


@dataclass(frozen=True, slots=True)
class IndexBuildSnapshot:
    """Immutable inputs and fingerprint for one candidate index build."""

    language: str
    intent_sources: dict[str, Mapping[str, Any]]
    registry_slot_values: dict[str, tuple[str, ...]]
    dynamic_registry_intents: tuple[DynamicRegistryIntent, ...]
    fingerprint: str


@dataclass(slots=True)
class CanonicalizerRuntime:
    """Mutable runtime state shared by the integration entry and agent.

    Lifecycle invariant: shutdown closes work admission before it cancels tasks.
    Methods that start or publish work must reject a closed runtime, and methods
    that span an await must recheck the relevant generation before publishing.
    """

    indexes: dict[str, CanonicalIndex] = field(default_factory=dict)
    diagnostics: CanonicalizerDiagnostics = field(default_factory=CanonicalizerDiagnostics)
    intent_sources: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    language_intent_sources: dict[str, dict[str, Mapping[str, Any]]] = field(default_factory=dict)
    config_path: Callable[..., str] | None = None
    registry_slot_values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    registry_slot_index: RegistrySlotIndex = field(default_factory=lambda: RegistrySlotIndex({}))
    registry_slot_indexes: dict[str, RegistrySlotIndex] = field(default_factory=dict)
    dynamic_registry_intents: dict[str, tuple[DynamicRegistryIntent, ...]] = field(
        default_factory=dict
    )
    cleanup_callbacks: list[Callable[[], None]] = field(default_factory=list)
    rebuild_tasks: dict[str, tuple[int, asyncio.Task[CanonicalIndex | None]]] = field(
        default_factory=dict
    )
    warmup_tasks: set[asyncio.Task[Any]] = field(default_factory=set, repr=False)
    _logged_rebuilds: set[tuple[str, int]] = field(default_factory=set, repr=False)
    index_generation: int = 0
    source_generation: int = 0
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

    async def async_rebuild_index(
        self,
        hass: Any,
        language: str,
        *,
        log_error: bool = True,
        raise_on_error: bool = False,
    ) -> CanonicalIndex | None:
        """Rebuild one language index once while concurrent callers await it."""
        if self._closed:
            return None
        language = normalize_language(language)
        generation = self.index_generation
        task_state = self.rebuild_tasks.get(language)
        if task_state is not None:
            generation, task = task_state
            if generation != self.index_generation or task.done():
                self.rebuild_tasks.pop(language, None)
                task = None
        else:
            task = None

        if task is None:
            if self._closed:
                return None
            self._logged_rebuilds.discard((language, generation))
            task = hass.async_create_task(
                _run_rebuild(
                    self,
                    hass,
                    language,
                    generation,
                )
            )
            self.rebuild_tasks[language] = (generation, task)

        try:
            return await task
        except asyncio.CancelledError:
            raise
        except Exception as err:
            key = (language, generation)
            if key not in self._logged_rebuilds:
                self._logged_rebuilds.add(key)
                log_level = logging.ERROR if log_error else logging.INFO
                _LOGGER.log(
                    log_level,
                    "Unexpected error rebuilding index for language %s",
                    language,
                    exc_info=err,
                )
            if raise_on_error:
                raise
            return None
        finally:
            task_state = self.rebuild_tasks.get(language)
            if task_state and task_state[1] is task:
                self.rebuild_tasks.pop(language, None)

    async def async_load_index_from_store(self, hass: Any, language: str) -> CanonicalIndex | None:
        """Load an index only when its persisted source fingerprint is current."""
        if not self._start_index_load():
            return None
        try:
            return await self._async_load_index_from_store(hass, language)
        finally:
            self._finish_index_load()

    async def _async_load_index_from_store(
        self,
        hass: Any,
        language: str,
    ) -> CanonicalIndex | None:
        """Load one persisted index while the public operation tracks its lifetime."""
        language = normalize_language(language)
        generation = self.index_generation
        source_generation = self.source_generation
        build_inputs = self._capture_build_inputs()
        snapshot = await _async_add_executor_job_drained(
            hass,
            _create_build_snapshot_and_register_wildcards,
            language,
            *build_inputs,
        )
        if (
            self._closed
            or self.index_generation != generation
            or self.source_generation != source_generation
        ):
            return None

        async with self._storage_lock:
            if (
                self._closed
                or self.index_generation != generation
                or self.source_generation != source_generation
            ):
                return None
            manifest = await self._async_load_store_manifest(hass)
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

            if not _valid_store_metadata(
                data,
                language=language,
                fingerprint=snapshot.fingerprint,
                cache_epoch=cache_epoch,
            ):
                await _async_await_drained(store.async_remove())
                persisted_languages.discard(language)
                await self._async_save_store_manifest(
                    hass,
                    cache_epoch,
                    persisted_languages,
                )
                return None
            candidates = _deserialize_candidates(data)
            if candidates is None or len(candidates) != data["candidate_count"]:
                await _async_await_drained(store.async_remove())
                persisted_languages.discard(language)
                await self._async_save_store_manifest(
                    hass,
                    cache_epoch,
                    persisted_languages,
                )
                return None

        index = await _async_add_executor_job_drained(hass, build_index, language, candidates)
        if (
            self._closed
            or self.index_generation != generation
            or self.source_generation != source_generation
        ):
            return None
        self.language_intent_sources[language] = snapshot.intent_sources
        self.dynamic_registry_intents[language] = snapshot.dynamic_registry_intents
        self.set_index(index)
        return index

    async def async_save_index_to_store(
        self,
        hass: Any,
        index: CanonicalIndex,
        fingerprint: str,
        *,
        expected_index_generation: int | None = None,
        expected_source_generation: int | None = None,
    ) -> bool:
        """Save an index with the source fingerprint that produced it."""
        if self._closed:
            return False
        language = normalize_language(index.language)
        candidates_data = [_serialize_candidate(candidate) for candidate in index.candidates]
        data = {
            "build_version": _INDEX_BUILD_VERSION,
            "language": language,
            "fingerprint": fingerprint,
            "candidate_count": len(candidates_data),
            "candidates": candidates_data,
        }

        async with self._storage_lock:
            if not self._storage_generation_matches(
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
                expected_index_generation,
                expected_source_generation,
            ):
                return False

            data["cache_epoch"] = cache_epoch
            await _async_await_drained(_index_store(hass, language).async_save(data))
            if not self._storage_generation_matches(
                expected_index_generation,
                expected_source_generation,
            ):
                return False
            persisted_languages.add(language)
            await self._async_save_store_manifest(hass, cache_epoch, persisted_languages)
        return True

    async def async_clear_index(self, hass: Any, language: str | None = None) -> None:
        """Clear memory and persisted indexes for one language or every language."""
        if self._closed:
            return
        normalized_language = normalize_language(language) if language is not None else None
        async with self._storage_lock:
            self.clear_index(normalized_language)
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
                return

            persisted_languages = manifest[1] if manifest is not None else set()
            await self._async_save_store_manifest(hass, uuid4().hex, set())
            for persisted_language in persisted_languages:
                await _async_await_drained(_index_store(hass, persisted_language).async_remove())

    def rank_with_dynamic_candidates(
        self,
        language: str,
        index: CanonicalIndex,
        query: str,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        *,
        slot_preferences: set[tuple[str, str]] | None = None,
        intent_context: Mapping[str, Any] | None = None,
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
        static_accepted = accepted_candidate(
            ranked,
            min_confidence=min_confidence,
            min_margin=min_margin,
        )
        numeric_literal_rescue = _query_needs_literal_only_dynamic(query)
        wildcard_literal_rescue = static_accepted is None and not numeric_literal_rescue
        include_literal_only_templates = numeric_literal_rescue or wildcard_literal_rescue
        dynamic_candidates = build_query_registry_candidates(
            language,
            intent_sources,
            registry_slot_values,
            query,
            registry_slot_index=registry_slot_index,
            compiled_intents=self._dynamic_registry_intents_for_query(language),
            include_literal_only_templates=include_literal_only_templates,
            include_area_only_templates=False,
        )
        if wildcard_literal_rescue:
            dynamic_candidates = tuple(
                candidate
                for candidate in dynamic_candidates
                if candidate.has_wildcard or candidate.metadata.get("query_slots")
            )
        if not dynamic_candidates:
            return ranked
        self.update_diagnostics(dynamic_candidate_count=len(dynamic_candidates))
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
        dynamic_ranked = rank_candidates(
            query,
            dynamic_candidates,
            max_candidates=max_candidates,
            reference_bm25_index=index.bm25_index,
            candidate_char_index=dynamic_char_index,
            exact_normalized_lookup=exact_normalized_lookup,
            language=language,
            slot_preferences=slot_preferences,
            intent_context=intent_context,
            min_confidence=min_confidence,
        )
        if _is_perfect_rank_result(dynamic_ranked):
            return dynamic_ranked
        if (
            static_accepted is not None
            and dynamic_ranked
            and ranked
            and dynamic_ranked[0].scores.final_score - ranked[0].scores.final_score < min_margin
        ):
            return ranked
        return _merge_ranked_candidates(ranked, dynamic_ranked, max_candidates)

    def clear_index(self, language: str | None = None) -> None:
        """Clear one language index or all indexes and invalidate active rebuilds."""
        self.index_generation += 1
        if language is not None:
            language = normalize_language(language)
        if language is None:
            self.indexes.clear()
            self.rebuild_tasks.clear()
        else:
            self.rebuild_tasks.pop(language, None)
            self.indexes.pop(language, None)
        self.update_diagnostics(candidate_count=self.total_candidate_count())

    def total_candidate_count(self) -> int:
        """Return the total number of cached candidates."""
        return sum(index.candidate_count for index in self.indexes.values())

    def configure_config_path(self, config_path: Callable[..., str]) -> None:
        """Configure Home Assistant config path access for custom sentences."""
        if self._closed:
            return
        self.config_path = config_path
        self._invalidate_source_dependent_indexes(clear_sources=True)

    def update_registry_slot_values(self, slot_values: Mapping[str, tuple[str, ...]]) -> None:
        """Update cached registry metadata used for candidate expansion."""
        if self._closed:
            return
        updated_values = {key: tuple(values) for key, values in slot_values.items()}
        if updated_values == self.registry_slot_values:
            return
        self.registry_slot_values = updated_values
        self.registry_slot_index = build_registry_slot_index(updated_values)
        self.registry_slot_indexes.clear()
        self._invalidate_source_dependent_indexes(clear_sources=False)

    def update_intent_sources(self, intents_update: Mapping[Any, Mapping[str, Any]]) -> None:
        """Update cached Home Assistant conversation intent sources."""
        if self._closed:
            return
        updated_sources = {
            _source_key(source): source_config for source, source_config in intents_update.items()
        }
        self.intent_sources = updated_sources
        self._invalidate_source_dependent_indexes(clear_sources=True)

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

    def _intent_sources_for_query(self, language: str) -> dict[str, Mapping[str, Any]]:
        """Return cached intent sources for query-time candidate expansion."""
        if self._closed:
            return {}
        language = normalize_language(language)
        cached = self.language_intent_sources.get(language)
        return cached if cached is not None else self._all_intent_sources(language)

    def _dynamic_registry_intents_for_query(
        self, language: str
    ) -> tuple[DynamicRegistryIntent, ...]:
        """Return compiled query-independent registry templates for a language."""
        if self._closed:
            return ()
        language = normalize_language(language)
        cached = self.dynamic_registry_intents.get(language)
        if cached is not None:
            return cached
        compiled = compile_dynamic_registry_intents(
            self._intent_sources_for_query(language),
            language,
            include_literal_only_templates=True,
            include_area_only_templates=False,
        )
        self.dynamic_registry_intents[language] = compiled
        return compiled

    def _registry_slot_index_for_language(self, language: str) -> RegistrySlotIndex:
        """Return registry slot records normalized for the query language."""
        return self._registry_slot_snapshot_for_language(language)[1]

    def _registry_slot_snapshot_for_language(
        self,
        language: str,
    ) -> tuple[dict[str, tuple[str, ...]], RegistrySlotIndex]:
        """Return one registry values snapshot with its matching language index."""
        language = normalize_language(language)
        registry_slot_values = {
            key: tuple(values) for key, values in self.registry_slot_values.items()
        }
        cached = self.registry_slot_indexes.get(language)
        if cached is not None and registry_slot_values == self.registry_slot_values:
            return registry_slot_values, cached
        built = build_registry_slot_index(registry_slot_values, language)
        if registry_slot_values == self.registry_slot_values:
            self.registry_slot_indexes[language] = built
        return registry_slot_values, built

    def _all_intent_sources(self, language: str) -> dict[str, Mapping[str, Any]]:
        """Return built-in, custom, and subscribed intent sources."""
        if self._closed:
            return {}
        language = normalize_language(language)
        sources = load_language_intent_sources(language, config_path=self.config_path)
        sources.update(self.intent_sources)
        self.language_intent_sources[language] = sources
        return sources

    def _capture_build_inputs(
        self,
    ) -> tuple[
        Callable[..., str] | None,
        dict[str, Mapping[str, Any]],
        dict[str, tuple[str, ...]],
    ]:
        """Copy mutable runtime inputs before executor work begins."""
        return (
            self.config_path,
            dict(self.intent_sources),
            {key: tuple(values) for key, values in self.registry_slot_values.items()},
        )

    def _invalidate_source_dependent_indexes(self, *, clear_sources: bool) -> None:
        """Invalidate indexes whenever candidate-producing inputs change."""
        self.source_generation += 1
        self.indexes.clear()
        if clear_sources:
            self.language_intent_sources.clear()
            self.dynamic_registry_intents.clear()
        self.update_diagnostics(candidate_count=0)

    def _storage_generation_matches(
        self,
        expected_index_generation: int | None,
        expected_source_generation: int | None,
    ) -> bool:
        """Return whether a pending store write still belongs to current inputs."""
        return (
            not self._closed
            and (
                expected_index_generation is None
                or self.index_generation == expected_index_generation
            )
            and (
                expected_source_generation is None
                or self.source_generation == expected_source_generation
            )
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

    async def _async_load_store_manifest(self, hass: Any) -> tuple[str, set[str]] | None:
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
        return cache_epoch, {normalize_language(language) for language in languages}

    async def _async_save_store_manifest(
        self,
        hass: Any,
        cache_epoch: str,
        languages: set[str],
    ) -> None:
        """Persist the cache epoch and known language store keys."""
        await _async_await_drained(
            _manifest_store(hass).async_save(
                {
                    "cache_epoch": cache_epoch,
                    "languages": sorted(languages),
                }
            )
        )

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Remember a cleanup callback for unload."""
        if self._closed:
            callback()
            return
        self.cleanup_callbacks.append(callback)

    def track_warmup_task(self, task: Any) -> None:
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

    def _begin_shutdown(self) -> tuple[asyncio.Task[Any], ...]:
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
        tasks: set[asyncio.Task[Any]] = set(self.warmup_tasks)
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
        self.indexes.clear()
        self.intent_sources.clear()
        self.language_intent_sources.clear()
        self.registry_slot_values.clear()
        self.registry_slot_index = RegistrySlotIndex({})
        self.registry_slot_indexes.clear()
        self.dynamic_registry_intents.clear()
        self.rebuild_tasks.clear()
        self.warmup_tasks.clear()
        self.update_diagnostics(candidate_count=0, dynamic_candidate_count=0)
        clear_normalization_caches()
        clear_bm25_caches()
        clear_rehydration_caches()
        clear_utils_caches()
        clear_grammar_loader_caches()
        clear_ranking_caches()

    def update_diagnostics(
        self,
        *,
        candidate_count: int | None = None,
        index_version: int | None = None,
        last_query_latency_ms: float | None = None,
        last_fallback_reason: FallbackReason | str | None = None,
        last_error: str | None = None,
        dynamic_candidate_count: int | None = None,
        clear_last_fallback_reason: bool = False,
        clear_last_error: bool = False,
    ) -> None:
        """Update the diagnostics snapshot."""
        self.diagnostics = CanonicalizerDiagnostics(
            candidate_count=(
                self.diagnostics.candidate_count if candidate_count is None else candidate_count
            ),
            index_version=self.diagnostics.index_version
            if index_version is None
            else index_version,
            last_query_latency_ms=(
                self.diagnostics.last_query_latency_ms
                if last_query_latency_ms is None
                else last_query_latency_ms
            ),
            last_fallback_reason=_updated_optional_text(
                current=self.diagnostics.last_fallback_reason,
                value=last_fallback_reason,
                clear=clear_last_fallback_reason,
            ),
            last_error=_updated_optional_text(
                current=self.diagnostics.last_error,
                value=last_error,
                clear=clear_last_error,
            ),
            dynamic_candidate_count=(
                self.diagnostics.dynamic_candidate_count
                if dynamic_candidate_count is None
                else dynamic_candidate_count
            ),
        )


async def _run_rebuild(
    runtime: CanonicalizerRuntime,
    hass: Any,
    language: str,
    generation: int,
) -> CanonicalIndex | None:
    """Build from a stable source generation and persist the matching fingerprint."""
    try:
        for _ in range(_MAX_REBUILD_ATTEMPTS):
            if runtime.closed:
                return None
            source_generation = runtime.source_generation
            build_inputs = runtime._capture_build_inputs()
            snapshot = await _async_add_executor_job_drained(
                hass,
                _create_build_snapshot_and_register_wildcards,
                language,
                *build_inputs,
            )
            if runtime.closed:
                return None
            index = await _async_add_executor_job_drained(
                hass,
                _build_index_from_snapshot,
                snapshot,
            )
            if runtime.closed or runtime.index_generation != generation:
                return None
            if runtime.source_generation != source_generation:
                continue

            saved = await runtime.async_save_index_to_store(
                hass,
                index,
                snapshot.fingerprint,
                expected_index_generation=generation,
                expected_source_generation=source_generation,
            )
            if runtime.closed or runtime.index_generation != generation:
                return None
            if runtime.source_generation != source_generation or not saved:
                continue

            if runtime.closed:
                return None
            runtime.language_intent_sources[language] = snapshot.intent_sources
            runtime.dynamic_registry_intents[language] = snapshot.dynamic_registry_intents
            runtime.set_index(index)
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
    subscribed_sources: dict[str, Mapping[str, Any]],
    registry_slot_values: dict[str, tuple[str, ...]],
) -> IndexBuildSnapshot:
    """Load candidate sources, register custom wildcards, and fingerprint build inputs.

    Side Effects:
    - Registers custom wildcard slots for the target language to global state.
    - Pre-warms the wildcard slot name caches (``wildcard_slot_names`` and
      ``wildcard_slot_names_sorted``) so that subsequent event-loop calls to
      ``rehydration`` and ``grammar_loader`` always hit a warm cache and never
      trigger the blocking ``open()`` call inside ``home_assistant_intents.get_intents``.
    """
    sources = load_language_intent_sources(language, config_path=config_path)
    sources.update(subscribed_sources)
    register_custom_wildcards_from_sources(language, sources)
    wildcard_slot_names(language)
    wildcard_slot_names_sorted(language)
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


def _current_task_or_none() -> asyncio.Task[Any] | None:
    """Return the current asyncio task when called inside a running loop."""
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


async def _async_add_executor_job_drained(hass: Any, func: Callable[..., Any], *args: Any) -> Any:
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


def _canonical_fingerprint_value(value: Any) -> Any:
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


def _index_store(hass: Any, language: str) -> Any:
    """Return the versioned Home Assistant Store for one language index."""
    kwargs: dict[str, Any] = {}
    if _STORE_HAS_SERIALIZE_IN_EVENT_LOOP:
        kwargs["serialize_in_event_loop"] = False
    return storage.Store(
        hass,
        _INDEX_STORE_VERSION,
        f"{_INDEX_STORE_PREFIX}{language}",
        **kwargs,
    )


def _manifest_store(hass: Any) -> Any:
    """Return the Store tracking the current cache epoch and language keys."""
    return storage.Store(hass, _INDEX_MANIFEST_VERSION, _INDEX_MANIFEST_KEY)


def _valid_store_metadata(
    data: Any,
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


def _serialize_candidate(candidate: Candidate) -> dict[str, Any]:
    """Return the persisted representation of one candidate."""
    return {
        "text": candidate.text,
        "intent_name": candidate.intent_name,
        "source": str(candidate.source),
        "language": candidate.language,
        "metadata": dict(candidate.metadata),
        "slot_values": list(candidate.slot_values),
        "normalized_text": candidate.normalized_text,
    }


def _deserialize_candidates(data: dict[str, Any]) -> list[Candidate] | None:
    """Deserialize all candidates, rejecting the whole cache on any invalid record."""
    candidates: list[Candidate] = []
    for candidate_data in data["candidates"]:
        if not isinstance(candidate_data, dict):
            return None
        text = candidate_data.get("text")
        intent_name = candidate_data.get("intent_name")
        source = candidate_data.get("source")
        language = candidate_data.get("language")
        metadata = candidate_data.get("metadata")
        slot_values = candidate_data.get("slot_values", ())
        normalized_text = candidate_data.get("normalized_text")
        if (
            not isinstance(text, str)
            or not isinstance(intent_name, str)
            or not isinstance(source, str)
            or (language is not None and not isinstance(language, str))
            or not isinstance(metadata, Mapping)
            or not isinstance(slot_values, list | tuple)
            or not all(isinstance(value, str) for value in slot_values)
            or not isinstance(normalized_text, str)
            or not normalized_text
        ):
            return None
        try:
            candidates.append(
                Candidate(
                    text=text,
                    intent_name=intent_name,
                    source=CandidateSource(source),
                    language=language,
                    metadata=metadata,
                    slot_values=tuple(slot_values),
                    normalized_text=normalized_text,
                )
            )
        except (TypeError, ValueError):
            return None
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
    return tuple(ranked[:max_candidates])


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


def _source_key(source: Any) -> str:
    """Return a stable string key for a Home Assistant intent source."""
    name = getattr(source, "name", None)
    return name.lower() if isinstance(name, str) else str(source).lower()
