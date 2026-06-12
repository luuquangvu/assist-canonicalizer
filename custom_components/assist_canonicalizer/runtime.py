"""Runtime state for Assist Canonicalizer."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from .builtin_intents import load_language_intent_sources
from .candidate import Candidate, CandidateSource
from .const import DEFAULT_MAX_CANDIDATES, FallbackReason
from .diagnostics import CanonicalizerDiagnostics
from .grammar_loader import (
    build_candidates_from_intent_sources,
    build_query_registry_candidates,
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


@dataclass(slots=True)
class CanonicalizerRuntime:
    """Mutable runtime state shared by the integration entry and agent."""

    indexes: dict[str, CanonicalIndex] = field(default_factory=dict)
    diagnostics: CanonicalizerDiagnostics = field(default_factory=CanonicalizerDiagnostics)
    intent_sources: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    language_intent_sources: dict[str, dict[str, Mapping[str, Any]]] = field(default_factory=dict)
    config_path: Callable[..., str] | None = None
    registry_slot_values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    cleanup_callbacks: list[Callable[[], None]] = field(default_factory=list)
    rebuild_tasks: dict[str, tuple[int, asyncio.Task[CanonicalIndex]]] = field(default_factory=dict)
    index_generation: int = 0
    rebuild_timer_cancel: Callable[[], None] | None = None

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
        index = self._build_index(language)
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
            """Run the blocking index build in Home Assistant's executor."""
            index = await hass.async_add_executor_job(self._build_index, language)
            if self.index_generation == generation:
                self.set_index(index)
                await self.async_save_index_to_store(hass, index)
            return index

        task = hass.async_create_task(run_rebuild())
        self.rebuild_tasks[language] = (generation, task)
        try:
            return await task
        finally:
            if self.rebuild_tasks.get(language) == (generation, task):
                self.rebuild_tasks.pop(language, None)

    async def async_load_index_from_store(self, hass: Any, language: str) -> CanonicalIndex | None:
        """Load candidate list from store and rebuild index."""
        language = normalize_language(language)
        store = storage.Store(hass, 1, f"assist_canonicalizer.index_{language}")
        try:
            data = await store.async_load()
        except Exception:
            return None

        if not data or not isinstance(data, dict):
            return None

        candidates_data = data.get("candidates")
        if not isinstance(candidates_data, list):
            return None

        candidates = []
        for c in candidates_data:
            try:
                candidates.append(
                    Candidate(
                        text=c["text"],
                        intent_name=c["intent_name"],
                        source=CandidateSource(c["source"]),
                        language=c.get("language"),
                        metadata=c.get("metadata", {}),
                        normalized_text=c.get("normalized_text", ""),
                    )
                )
            except Exception:
                continue

        index = await hass.async_add_executor_job(build_index, language, candidates)
        self.set_index(index)
        return index

    async def async_save_index_to_store(self, hass: Any, index: CanonicalIndex) -> None:
        """Save index candidate list to store."""
        store = storage.Store(hass, 1, f"assist_canonicalizer.index_{index.language}")
        data = {
            "candidates": [
                {
                    "text": c.text,
                    "intent_name": c.intent_name,
                    "source": str(c.source),
                    "language": c.language,
                    "metadata": dict(c.metadata),
                    "normalized_text": c.normalized_text,
                }
                for c in index.candidates
            ]
        }
        await store.async_save(data)

    def _build_index(self, language: str) -> CanonicalIndex:
        """Build one language index without mutating runtime cache."""
        language = normalize_language(language)
        candidates = build_candidates_from_intent_sources(
            language,
            self._all_intent_sources(language),
            self.registry_slot_values,
        )
        return build_index(language, candidates)

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
        self.language_intent_sources.clear()

    def update_registry_slot_values(self, slot_values: Mapping[str, tuple[str, ...]]) -> None:
        """Update cached registry metadata used for candidate expansion."""
        self.registry_slot_values = dict(slot_values)

    def update_intent_sources(self, intents_update: Mapping[Any, Mapping[str, Any]]) -> None:
        """Update cached Home Assistant conversation intent sources."""
        self.language_intent_sources.clear()
        for source, source_config in intents_update.items():
            self.intent_sources[_source_key(source)] = source_config

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

    def _all_intent_sources(self, language: str) -> dict[str, Mapping[str, Any]]:
        """Return built-in, custom, and subscribed intent sources."""
        language = normalize_language(language)
        sources = load_language_intent_sources(language, config_path=self.config_path)
        sources.update(self.intent_sources)
        self.language_intent_sources[language] = sources
        return sources

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


def normalize_language(language: str) -> str:
    """Return a canonical language cache key."""
    normalized = str(language).strip().lower()
    if not normalized:
        raise ValueError("Language must not be empty")
    return normalized


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
