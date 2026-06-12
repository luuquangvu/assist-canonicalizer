"""Runtime state for Assist Canonicalizer."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, cast
from uuid import uuid4

import orjson

from .builtin_intents import language_variant_for, load_language_intent_sources
from .candidate import Candidate, CandidateSource
from .const import DEFAULT_MAX_CANDIDATES, DOMAIN, FallbackReason
from .diagnostics import CanonicalizerDiagnostics
from .grammar_loader import (
    DynamicRegistryIntent,
    RegistrySlotIndex,
    build_candidates_from_intent_sources,
    build_query_registry_candidates,
    build_registry_slot_index,
    compile_dynamic_registry_intents,
)
from .indexer import CanonicalIndex, build_index
from .normalization import char_ngrams_normalized
from .ranking import CharNGramIndex, RankedCandidate, rank_candidates

storage: Any = cast(Any, None)

try:
    from homeassistant.helpers import storage as _storage

    storage = _storage
except (ImportError, RuntimeError):
    storage = cast(Any, None)
    _HAS_STORAGE = False


_INDEX_STORE_VERSION = 2
_INDEX_BUILD_VERSION = 1
_INDEX_STORE_PREFIX = f"{DOMAIN}.index_"
_INDEX_MANIFEST_KEY = f"{DOMAIN}.index_manifest"
_INDEX_MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class IndexBuildSnapshot:
    """Immutable inputs and fingerprint for one candidate index build."""

    language: str
    intent_sources: dict[str, Mapping[str, Any]]
    registry_slot_values: dict[str, tuple[str, ...]]
    fingerprint: str


@dataclass(slots=True)
class CanonicalizerRuntime:
    """Mutable runtime state shared by the integration entry and agent."""

    indexes: dict[str, CanonicalIndex] = field(default_factory=dict)
    diagnostics: CanonicalizerDiagnostics = field(default_factory=CanonicalizerDiagnostics)
    intent_sources: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    language_intent_sources: dict[str, dict[str, Mapping[str, Any]]] = field(default_factory=dict)
    config_path: Callable[..., str] | None = None
    registry_slot_values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    registry_slot_index: RegistrySlotIndex = field(default_factory=dict)
    dynamic_registry_intents: dict[str, tuple[DynamicRegistryIntent, ...]] = field(
        default_factory=dict
    )
    cleanup_callbacks: list[Callable[[], None]] = field(default_factory=list)
    rebuild_tasks: dict[str, tuple[int, asyncio.Task[CanonicalIndex]]] = field(default_factory=dict)
    index_generation: int = 0
    source_generation: int = 0
    rebuild_timer_cancel: Callable[[], None] | None = None
    _storage_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def get_index(self, language: str) -> CanonicalIndex | None:
        """Return the cached index for a language."""
        return self.indexes.get(normalize_language(language))

    def set_index(self, index: CanonicalIndex) -> None:
        """Store a language-specific index."""
        self.indexes[normalize_language(index.language)] = index
        self.update_diagnostics(
            candidate_count=index.candidate_count,
            index_version=index.version,
        )

    def rebuild_index(self, language: str) -> CanonicalIndex:
        """Rebuild and store one language index from automatic candidate sources."""
        language = normalize_language(language)
        snapshot = self._create_build_snapshot(language)
        index = _build_index_from_snapshot(snapshot)
        self.language_intent_sources[language] = snapshot.intent_sources
        self.set_index(index)
        return index

    async def async_rebuild_index(self, hass: Any, language: str) -> CanonicalIndex:
        """Rebuild one language index once while concurrent callers await it."""
        language = normalize_language(language)
        task_state = self.rebuild_tasks.get(language)
        if task_state is not None:
            generation, task = task_state
            if generation == self.index_generation and not task.done():
                return await task
            self.rebuild_tasks.pop(language, None)

        generation = self.index_generation

        async def run_rebuild() -> CanonicalIndex:
            """Build from a stable source generation and persist the matching fingerprint."""
            while True:
                source_generation = self.source_generation
                build_inputs = self._capture_build_inputs()
                snapshot = await hass.async_add_executor_job(
                    _create_build_snapshot,
                    language,
                    *build_inputs,
                )
                index = await hass.async_add_executor_job(_build_index_from_snapshot, snapshot)
                if self.index_generation != generation:
                    return index
                if self.source_generation != source_generation:
                    continue

                self.language_intent_sources[language] = snapshot.intent_sources
                self.set_index(index)
                saved = await self.async_save_index_to_store(
                    hass,
                    index,
                    snapshot.fingerprint,
                    expected_index_generation=generation,
                    expected_source_generation=source_generation,
                )
                if self.index_generation != generation:
                    return index
                if self.source_generation != source_generation or not saved:
                    continue
                return index

        task = hass.async_create_task(run_rebuild())
        self.rebuild_tasks[language] = (generation, task)
        try:
            return await task
        finally:
            if self.rebuild_tasks.get(language) == (generation, task):
                self.rebuild_tasks.pop(language, None)

    async def async_load_index_from_store(self, hass: Any, language: str) -> CanonicalIndex | None:
        """Load an index only when its persisted source fingerprint is current."""
        language = normalize_language(language)
        generation = self.index_generation
        source_generation = self.source_generation
        build_inputs = self._capture_build_inputs()
        snapshot = await hass.async_add_executor_job(
            _create_build_snapshot,
            language,
            *build_inputs,
        )
        if self.index_generation != generation or self.source_generation != source_generation:
            return None

        async with self._storage_lock:
            manifest = await self._async_load_store_manifest(hass)
            if manifest is None:
                return None
            cache_epoch, persisted_languages = manifest
            if language not in persisted_languages:
                return None
            store = _index_store(hass, language)
            try:
                data = await store.async_load()
            except Exception:
                return None

            if not _valid_store_metadata(
                data,
                language=language,
                fingerprint=snapshot.fingerprint,
                cache_epoch=cache_epoch,
            ):
                await store.async_remove()
                persisted_languages.discard(language)
                await self._async_save_store_manifest(
                    hass,
                    cache_epoch,
                    persisted_languages,
                )
                return None

        candidates = _deserialize_candidates(data)
        if candidates is None:
            await self._async_remove_persisted_language(hass, language)
            return None
        index = await hass.async_add_executor_job(build_index, language, candidates)
        if index.candidate_count != data["candidate_count"]:
            await self._async_remove_persisted_language(hass, language)
            return None
        if self.index_generation != generation or self.source_generation != source_generation:
            return None
        self.language_intent_sources[language] = snapshot.intent_sources
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
            await _index_store(hass, language).async_save(data)
            persisted_languages.add(language)
            await self._async_save_store_manifest(hass, cache_epoch, persisted_languages)
        return True

    async def async_clear_index(self, hass: Any, language: str | None = None) -> None:
        """Clear memory and persisted indexes for one language or every language."""
        normalized_language = normalize_language(language) if language is not None else None
        async with self._storage_lock:
            self.clear_index(normalized_language)
            manifest = await self._async_load_store_manifest(hass)
            if normalized_language is not None:
                await _index_store(hass, normalized_language).async_remove()
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
                await _index_store(hass, persisted_language).async_remove()

    def _build_index(self, language: str) -> CanonicalIndex:
        """Build one language index without mutating runtime cache."""
        return _build_index_from_snapshot(self._create_build_snapshot(language))

    def rank_with_dynamic_candidates(
        self,
        language: str,
        index: CanonicalIndex,
        query: str,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> tuple[RankedCandidate, ...]:
        """Rank cached index candidates plus query-scoped registry expansions."""
        language = normalize_language(language)
        ranked = index.rank(query, max_candidates=max_candidates)
        self.update_diagnostics(dynamic_candidate_count=0)
        dynamic_candidates = build_query_registry_candidates(
            language,
            self._intent_sources_for_query(language),
            self.registry_slot_values,
            query,
            registry_slot_index=self.registry_slot_index,
            compiled_intents=self._dynamic_registry_intents_for_query(language),
        )
        if not dynamic_candidates:
            return ranked
        self.update_diagnostics(dynamic_candidate_count=len(dynamic_candidates))
        dynamic_ranked = rank_candidates(
            query,
            dynamic_candidates,
            max_candidates=max_candidates,
            candidate_char_index=CharNGramIndex.from_grams(
                tuple(
                    char_ngrams_normalized(candidate.normalized_text)
                    for candidate in dynamic_candidates
                )
            ),
        )
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
        self.config_path = config_path
        self._invalidate_source_dependent_indexes(clear_sources=True)

    def update_registry_slot_values(self, slot_values: Mapping[str, tuple[str, ...]]) -> None:
        """Update cached registry metadata used for candidate expansion."""
        updated_values = {key: tuple(values) for key, values in slot_values.items()}
        if updated_values == self.registry_slot_values:
            return
        self.registry_slot_values = updated_values
        self.registry_slot_index = build_registry_slot_index(self.registry_slot_values)
        self._invalidate_source_dependent_indexes(clear_sources=False)

    def update_intent_sources(self, intents_update: Mapping[Any, Mapping[str, Any]]) -> None:
        """Update cached Home Assistant conversation intent sources."""
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
        language = normalize_language(language)
        cached = self.language_intent_sources.get(language)
        if cached is not None:
            return cached
        return self._all_intent_sources(language)

    def _dynamic_registry_intents_for_query(
        self, language: str
    ) -> tuple[DynamicRegistryIntent, ...]:
        """Return compiled query-independent registry templates for a language."""
        language = normalize_language(language)
        cached = self.dynamic_registry_intents.get(language)
        if cached is not None:
            return cached
        compiled = compile_dynamic_registry_intents(self._intent_sources_for_query(language))
        self.dynamic_registry_intents[language] = compiled
        return compiled

    def _all_intent_sources(self, language: str) -> dict[str, Mapping[str, Any]]:
        """Return built-in, custom, and subscribed intent sources."""
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

    def _create_build_snapshot(self, language: str) -> IndexBuildSnapshot:
        """Load and fingerprint the current inputs for one language."""
        return _create_build_snapshot(
            normalize_language(language),
            *self._capture_build_inputs(),
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
            expected_index_generation is None or self.index_generation == expected_index_generation
        ) and (
            expected_source_generation is None
            or self.source_generation == expected_source_generation
        )

    async def _async_load_store_manifest(self, hass: Any) -> tuple[str, set[str]] | None:
        """Load the cache epoch and known persisted language keys."""
        try:
            data = await _manifest_store(hass).async_load()
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
        await _manifest_store(hass).async_save(
            {
                "cache_epoch": cache_epoch,
                "languages": sorted(languages),
            }
        )

    async def _async_remove_persisted_language(self, hass: Any, language: str) -> None:
        """Remove one invalid language store and its manifest entry."""
        async with self._storage_lock:
            await _index_store(hass, language).async_remove()
            manifest = await self._async_load_store_manifest(hass)
            if manifest is None:
                return
            cache_epoch, persisted_languages = manifest
            persisted_languages.discard(language)
            await self._async_save_store_manifest(hass, cache_epoch, persisted_languages)

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Remember a cleanup callback for unload."""
        self.cleanup_callbacks.append(callback)

    def cleanup(self) -> None:
        """Run registered cleanup callbacks."""
        if self.rebuild_timer_cancel is not None:
            self.rebuild_timer_cancel()
            self.rebuild_timer_cancel = None
        callbacks = list(self.cleanup_callbacks)
        self.cleanup_callbacks.clear()
        for callback in callbacks:
            callback()

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


def _create_build_snapshot(
    language: str,
    config_path: Callable[..., str] | None,
    subscribed_sources: dict[str, Mapping[str, Any]],
    registry_slot_values: dict[str, tuple[str, ...]],
) -> IndexBuildSnapshot:
    """Load candidate sources and fingerprint the exact build inputs."""
    sources = load_language_intent_sources(language, config_path=config_path)
    sources.update(subscribed_sources)
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


def _canonical_fingerprint_value(value: Any) -> Any:
    """Return a deterministic, order-preserving representation for hashing."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return _canonical_fingerprint_value(value.value)
    if isinstance(value, Mapping):
        return {
            "mapping": [
                [
                    _canonical_fingerprint_value(key),
                    _canonical_fingerprint_value(item),
                ]
                for key, item in value.items()
            ]
        }
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
    return storage.Store(
        hass,
        _INDEX_STORE_VERSION,
        f"{_INDEX_STORE_PREFIX}{language}",
        serialize_in_event_loop=False,
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
        normalized_text = candidate_data.get("normalized_text")
        if (
            not isinstance(text, str)
            or not isinstance(intent_name, str)
            or not isinstance(source, str)
            or (language is not None and not isinstance(language, str))
            or not isinstance(metadata, Mapping)
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
    """Merge ranked candidates while keeping the strongest score per text."""
    selected: dict[str, RankedCandidate] = {}
    for ranked_candidate in (*primary, *dynamic):
        key = ranked_candidate.candidate.normalized_text
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


def _ranked_candidate_sort_key(ranked_candidate: RankedCandidate) -> tuple[float, int]:
    """Return a deterministic ranking key for merged candidates."""
    return (
        ranked_candidate.scores.final_score,
        -ranked_candidate.candidate.source_priority,
    )


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


def _updated_optional_text(current: str | None, value: str | None, *, clear: bool) -> str | None:
    """Return an updated optional diagnostics string."""
    if clear:
        return value
    if value is None:
        return current
    return value


def _source_key(source: Any) -> str:
    """Return a stable string key for a Home Assistant intent source."""
    name = getattr(source, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(source).lower()
