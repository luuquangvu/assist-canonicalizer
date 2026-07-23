"""Tests for runtime rebuild coordination."""

from __future__ import annotations

import asyncio
import inspect
import sys
import threading
from collections.abc import Mapping
from enum import Enum
from types import ModuleType
from typing import Any, ClassVar
from unittest.mock import patch

import homeassistant.helpers.event
import homeassistant.helpers.storage
import pytest

import custom_components.assist_canonicalizer as integration
from custom_components.assist_canonicalizer import (
    _subscribe_registry_updates,
    grammar_loader,
)
from custom_components.assist_canonicalizer import (
    runtime as runtime_module,
)
from custom_components.assist_canonicalizer.candidate import Candidate, CandidateSource
from custom_components.assist_canonicalizer.const import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_CANDIDATES_PER_TEMPLATE,
    DEFAULT_MAX_REGISTRY_VALUES_SCORED_PER_QUERY,
)
from custom_components.assist_canonicalizer.grammar_loader import (
    build_candidates_from_intent_sources,
    build_registry_slot_index,
)
from custom_components.assist_canonicalizer.indexer import CanonicalIndex, build_index
from custom_components.assist_canonicalizer.ranking import RankedCandidate, ScoreBreakdown
from custom_components.assist_canonicalizer.runtime import (
    _INDEX_BUILD_VERSION,
    CanonicalizerRuntime,
    _build_index_from_snapshot,
    _canonical_fingerprint_value,
    _create_build_snapshot_and_register_wildcards,
    _deserialize_candidates,
    _is_perfect_rank_result,
    _merge_ranked_candidates,
    _updated_optional_text,
    _valid_store_metadata,
)
from custom_components.assist_canonicalizer.utils import normalize_language


class FakeHass:
    """Minimal Home Assistant executor facade for runtime tests."""

    def __init__(self) -> None:
        """Initialize executor coordination events."""
        self.job_started = asyncio.Event()
        self.release_job = asyncio.Event()

    def async_create_task(self, coro: Any) -> asyncio.Task[Any]:
        """Create an asyncio task like Home Assistant."""
        return asyncio.create_task(coro)

    async def async_add_executor_job(self, func: Any, *args: Any) -> Any:
        """Delay executor work so concurrent rebuild callers overlap."""
        self.job_started.set()
        await self.release_job.wait()
        return func(*args)


class _SnapshotBuildCounter:
    """Count snapshot index builds and return a small index."""

    def __init__(self) -> None:
        """Initialize the call counter."""
        self.calls = 0

    def __call__(self, snapshot: Any) -> CanonicalIndex:
        """Count rebuild calls and return a small index."""
        self.calls += 1
        return build_index(
            snapshot.language,
            [
                Candidate(
                    text="turn on light",
                    intent_name="HassTurnOn",
                    language=snapshot.language,
                )
            ],
        )


async def _run_coalesced_rebuild_scenario(
    runtime: CanonicalizerRuntime,
) -> tuple[CanonicalIndex | None, CanonicalIndex | None]:
    """Start overlapping rebuild requests for the same language."""
    hass = FakeHass()
    first_task = asyncio.create_task(runtime.async_rebuild_index(hass, "en"))
    await hass.job_started.wait()
    second_task = asyncio.create_task(runtime.async_rebuild_index(hass, "en-US"))
    await asyncio.sleep(0)
    hass.release_job.set()
    return await asyncio.gather(first_task, second_task)


def _fail_dynamic_registry_build(*args: Any, **kwargs: Any) -> tuple[Candidate, ...]:
    """Fail if exact static matches still trigger dynamic generation."""
    raise AssertionError("dynamic candidates should not be built for static exact matches")


class _DynamicBuildCounter:
    """Count dynamic candidate generation calls."""

    def __init__(self, original_build: Any) -> None:
        """Initialize with the original build callable."""
        self.original_build = original_build
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Candidate, ...]:
        """Count dynamic candidate generation calls."""
        self.calls += 1
        return self.original_build(*args, **kwargs)


class _DynamicBuildRecorder:
    """Record dynamic candidate generation inputs."""

    def __init__(self) -> None:
        """Initialize the recorded input store."""
        self.registry_slot_values: Mapping[str, tuple[str, ...]] | None = None
        self.registry_slot_index_name_values: tuple[str, ...] = ()

    def __call__(
        self,
        _language: str,
        _intent_sources: Mapping[str, Mapping[str, Any]],
        registry_slot_values: Mapping[str, tuple[str, ...]],
        _query: str,
        **kwargs: Any,
    ) -> tuple[Candidate, ...]:
        """Record registry inputs passed to query-scoped dynamic generation."""
        self.registry_slot_values = registry_slot_values
        registry_slot_index = kwargs["registry_slot_index"]
        self.registry_slot_index_name_values = tuple(
            record.text for record in registry_slot_index["name"]
        )
        return ()


class _NormalizationTracker:
    """Record registry values normalized while building the snapshot."""

    def __init__(self, original_normalize_text: Any) -> None:
        """Initialize with the original normalization function."""
        self.original_normalize_text = original_normalize_text
        self.normalized_values: list[str] = []

    def __call__(self, text: str) -> str:
        """Record registry values normalized while building the snapshot."""
        self.normalized_values.append(text)
        return self.original_normalize_text(text)


class _SourceChangingBuild:
    """Change registry inputs during the first index build."""

    def __init__(self, runtime: CanonicalizerRuntime) -> None:
        """Initialize with the runtime under test."""
        self.runtime = runtime
        self.calls = 0

    def __call__(self, snapshot: Any) -> CanonicalIndex:
        """Change registry inputs during the first index build."""
        self.calls += 1
        name = snapshot.registry_slot_values["name"][0]
        if self.calls == 1:
            self.runtime.update_registry_slot_values({"name": ("new lamp",)})
        return build_index(
            snapshot.language,
            [Candidate(text=f"turn on {name}", intent_name="HassTurnOn")],
        )


class _AlwaysSourceChangingBuild:
    """Change registry inputs during every index build."""

    def __init__(self, runtime: CanonicalizerRuntime) -> None:
        """Initialize with the runtime under test."""
        self.runtime = runtime
        self.calls = 0

    def __call__(self, snapshot: Any) -> CanonicalIndex:
        """Change registry inputs during every index build."""
        self.calls += 1
        self.runtime.update_registry_slot_values({"name": (f"lamp {self.calls}",)})
        return build_index(
            snapshot.language,
            [Candidate(text=f"turn on lamp {self.calls}", intent_name="HassTurnOn")],
        )


class _IndexClearingBuild:
    """Clear the language index during the first index build."""

    def __init__(self, runtime: CanonicalizerRuntime) -> None:
        """Initialize with the runtime under test."""
        self.runtime = runtime
        self.calls = 0

    def __call__(self, snapshot: Any) -> CanonicalIndex:
        """Clear the index generation and return a small index."""
        self.calls += 1
        if self.calls == 1:
            self.runtime.clear_index(snapshot.language)
        return build_index(
            snapshot.language,
            [
                Candidate(
                    text=f"sample {self.calls}",
                    intent_name="Sample",
                    language=snapshot.language,
                )
            ],
        )


class _DebouncedFakeTimer:
    """Fake timer callback handle."""

    def __init__(self, scheduled_callbacks: list[Any], callback: Any) -> None:
        """Initialize fake timer."""
        self.scheduled_callbacks = scheduled_callbacks
        self.callback = callback

    def __call__(self) -> None:
        """Fire timer."""
        self.scheduled_callbacks.remove(self)
        self.callback(None)


class _TimerCancellation:
    """Callable cancellation handle for a fake timer."""

    def __init__(self, scheduled_callbacks: list[Any], timer: _DebouncedFakeTimer) -> None:
        """Initialize with the timer to remove."""
        self.scheduled_callbacks = scheduled_callbacks
        self.timer = timer

    def __call__(self) -> None:
        """Cancel the scheduled timer."""
        self.scheduled_callbacks.remove(self.timer)


class _AsyncCallLaterRecorder:
    """Record delayed callback scheduling."""

    def __init__(self, scheduled_callbacks: list[Any]) -> None:
        """Initialize with shared scheduled callback storage."""
        self.scheduled_callbacks = scheduled_callbacks

    def __call__(self, hass: Any, delay: float, action: Any) -> Any:
        """Mock scheduling timer."""
        timer = _DebouncedFakeTimer(self.scheduled_callbacks, action)
        self.scheduled_callbacks.append(timer)
        return _TimerCancellation(self.scheduled_callbacks, timer)


async def test_async_rebuild_index_coalesces_concurrent_language_jobs(monkeypatch: Any) -> None:
    """Coalesce equivalent language variants into one rebuild job."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()

    runtime = CanonicalizerRuntime()
    build_counter = _SnapshotBuildCounter()

    monkeypatch.setattr(runtime_module, "_build_index_from_snapshot", build_counter)
    first, second = await _run_coalesced_rebuild_scenario(runtime)

    assert first is second
    assert build_counter.calls == 1
    assert runtime.get_index("en") is first
    assert runtime.rebuild_tasks == {}


async def test_async_rebuild_index_replaces_stale_task_with_current_generation(
    monkeypatch: Any,
) -> None:
    """Use the current generation when replacing a stale rebuild task."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()
    runtime = CanonicalizerRuntime()
    build_counter = _SnapshotBuildCounter()
    hass = HashableFakeHass(async_create_task=lambda coro: asyncio.create_task(coro))

    monkeypatch.setattr(runtime_module, "_build_index_from_snapshot", build_counter)
    stale_generation = runtime._index_generation_for("en")
    runtime.clear_index("en")

    async def completed_rebuild() -> CanonicalIndex | None:
        """Return a completed task carrying the stale generation."""
        return None

    stale_task = asyncio.create_task(completed_rebuild())
    await stale_task
    runtime.rebuild_tasks["en"] = (stale_generation, stale_task)

    index = await runtime.async_rebuild_index(hass, "en")

    assert index is not None
    assert build_counter.calls == 1
    assert runtime.get_index("en") is index
    assert runtime.rebuild_tasks == {}


async def test_rank_with_dynamic_candidates_includes_tail_registry_alias() -> None:
    """Rank a query-time registry alias even when build-time expansion capped it."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "built_in": {
            "intents": {
                "HassTurnOff": {
                    "data": [
                        {
                            "sentences": ["tắt {name}"],
                            "requires_context": {"domain": "light"},
                        }
                    ]
                }
            }
        }
    }
    names = (
        *(f"Đèn giả {index}" for index in range(DEFAULT_MAX_CANDIDATES_PER_TEMPLATE + 10)),
        "Đèn tròn phòng khách",
    )
    registry_slots = {"name": names, "name:light": names}
    indexed_candidates = build_candidates_from_intent_sources(
        "vi",
        intent_sources,
        registry_slots,
    )
    assert all(
        candidate.normalized_text != "tắt đèn tròn phòng khách" for candidate in indexed_candidates
    )

    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values(registry_slots)
    runtime.language_intent_sources["vi"] = intent_sources
    ranked = runtime.rank_with_dynamic_candidates(
        "vi",
        build_index("vi", indexed_candidates),
        "tắt đèn tròn phòng khách",
    )

    assert ranked[0].candidate.normalized_text == "tắt đèn tròn phòng khách"
    assert ranked[0].scores.final_score == 1.0


def test_fuzzy_tail_registry_retrieval_is_bounded_independently_of_registry_size() -> None:
    """Recover a typo after the static cap while fully scoring a fixed-size pool."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "built_in": {
            "intents": {
                "HassTurnOn": {
                    "data": [
                        {
                            "sentences": ["turn on {name}"],
                            "requires_context": {"domain": "light"},
                        },
                        {
                            "sentences": ["turn on lights in {area}"],
                            "requires_context": {"domain": "light"},
                        },
                    ]
                }
            }
        }
    }
    names = (*(f"Synthetic fixture {index}" for index in range(2_000)), "Garden beacon")
    areas = (*(f"Synthetic area {index}" for index in range(2_000)), "Atrium")
    registry_slots = {"name": names, "name:light": names, "area": areas}
    static_candidates = build_candidates_from_intent_sources(
        "en",
        intent_sources,
        registry_slots,
    )
    assert all(candidate.text != "turn on Garden beacon" for candidate in static_candidates)

    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values(registry_slots)
    runtime.language_intent_sources["en"] = intent_sources
    ranked = runtime.rank_with_dynamic_candidates(
        "en",
        build_index("en", static_candidates),
        "turn on Garden becon",
    )

    assert ranked[0].candidate.text == "turn on Garden beacon"
    assert ranked[0].candidate.metadata["registry_retrieval"] == "fuzzy"
    atrium_ranked = runtime.rank_with_dynamic_candidates(
        "en",
        build_index("en", static_candidates),
        "turn on lights in atrum",
    )
    assert atrium_ranked[0].candidate.text == "turn on lights in Atrium"
    assert atrium_ranked[0].candidate.metadata["registry_retrieval"] == "fuzzy"
    assert runtime.diagnostics.registry_record_count == len(names) + len(areas)
    assert (
        runtime.diagnostics.registry_values_scored <= DEFAULT_MAX_REGISTRY_VALUES_SCORED_PER_QUERY
    )
    assert runtime.diagnostics.registry_values_nominated <= (
        DEFAULT_MAX_REGISTRY_VALUES_SCORED_PER_QUERY
    )
    assert runtime.diagnostics.fuzzy_dynamic_candidates >= 1
    assert runtime.diagnostics.registry_fingerprint


def test_rank_with_dynamic_candidates_uses_language_specific_registry_index() -> None:
    """Match registry values using language-specific transliteration rules."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "built_in": {
            "intents": {
                "HassTurnOn": {
                    "data": [
                        {
                            "sentences": ["schalte {name} ein"],
                        }
                    ]
                }
            }
        }
    }
    registry_slots = {"name": ("Küche Licht",)}
    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values(registry_slots)
    runtime.language_intent_sources["de"] = intent_sources
    index = build_index("de", [Candidate(text="unrelated", intent_name="HassNevermind")])

    ranked = runtime.rank_with_dynamic_candidates("de", index, "schalte kueche licht ein")

    assert ranked[0].candidate.text == "schalte Küche Licht ein"
    assert runtime.diagnostics.dynamic_candidate_count == 1
    assert runtime.registry_slot_index["name"][0].normalized_no_diacritics == "kuche licht"
    assert runtime.registry_slot_indexes["de"]["name"][0].normalized_no_diacritics == "kueche licht"


def test_rank_with_dynamic_candidates_includes_capped_domain_area_alias() -> None:
    """Rank a domain-area command even when static area expansion capped it."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "built_in": {
            "intents": {
                "HassTurnOn": {
                    "data": [
                        {
                            "sentences": ["turn on fan {area}"],
                            "slots": {"domain": "fan", "name": "all"},
                        }
                    ]
                }
            }
        }
    }
    areas = (
        *(f"area {index}" for index in range(DEFAULT_MAX_CANDIDATES_PER_TEMPLATE + 10)),
        "living room",
    )
    registry_slots = {"area": areas, "area_name": areas}
    indexed_candidates = build_candidates_from_intent_sources(
        "en",
        intent_sources,
        registry_slots,
    )
    assert all(
        candidate.normalized_text != "turn on fan living room" for candidate in indexed_candidates
    )

    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values(registry_slots)
    runtime.language_intent_sources["en"] = intent_sources
    ranked = runtime.rank_with_dynamic_candidates(
        "en",
        build_index("en", indexed_candidates),
        "turn on fan living room",
    )

    assert ranked[0].candidate.normalized_text == "turn on fan living room"
    assert ranked[0].scores.final_score == 1.0


def test_rank_with_dynamic_candidates_keeps_generic_area_only_templates_disabled() -> None:
    """Do not rescue broad generic area-only templates at query time."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "built_in": {
            "intents": {
                "HassTurnOn": {
                    "data": [{"sentences": ["turn on {area}"]}],
                }
            }
        }
    }
    areas = (
        *(f"area {index}" for index in range(DEFAULT_MAX_CANDIDATES_PER_TEMPLATE + 10)),
        "living room",
    )
    registry_slots = {"area": areas, "area_name": areas}
    indexed_candidates = build_candidates_from_intent_sources(
        "en",
        intent_sources,
        registry_slots,
    )
    assert all(
        candidate.normalized_text != "turn on living room" for candidate in indexed_candidates
    )

    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values(registry_slots)
    runtime.language_intent_sources["en"] = intent_sources
    ranked = runtime.rank_with_dynamic_candidates(
        "en",
        build_index("en", indexed_candidates),
        "turn on living room",
    )

    assert all(candidate.candidate.normalized_text != "turn on living room" for candidate in ranked)
    assert runtime.diagnostics.dynamic_candidate_count == 0


def test_rank_with_dynamic_candidates_rescues_literal_only_templates() -> None:
    """Use query-time base-list expansion when static caps omit exact variants."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "builtin": {
            "lists": {"timer_seconds": {"range": {"from": 1, "to": 2}}},
            "intents": {
                "HassStartTimer": {
                    "data": [{"sentences": ["timer for {timer_seconds:seconds}( |-)second[s]"]}]
                }
            },
        }
    }
    runtime = CanonicalizerRuntime()
    runtime.language_intent_sources["en"] = intent_sources

    ranked = runtime.rank_with_dynamic_candidates(
        "en",
        build_index("en", []),
        "timer for 1 second",
    )

    assert len(ranked) == 1
    assert ranked[0].candidate.text == "timer for 1 second"
    assert ranked[0].candidate.intent_name == "HassStartTimer"
    assert ranked[0].scores.final_score == 1.0
    assert runtime.diagnostics.dynamic_candidate_count >= 1


def test_rank_with_dynamic_candidates_rescues_non_numeric_wildcard_templates() -> None:
    """Use query-time fair expansion for free-text wildcard templates."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "builtin": {
            "lists": {"search_query": {"wildcard": True}},
            "intents": {
                "HassMediaSearchAndPlay": {"data": [{"sentences": ["(start|play) {search_query}"]}]}
            },
        }
    }
    runtime = CanonicalizerRuntime()
    runtime.language_intent_sources["en"] = intent_sources

    ranked = runtime.rank_with_dynamic_candidates(
        "en",
        build_index("en", []),
        "play Jazz",
    )

    assert ranked[0].candidate.intent_name == "HassMediaSearchAndPlay"
    assert ranked[0].candidate.text == "play search_query"
    assert (
        grammar_loader.rehydrate_wildcard_text(
            ranked[0].candidate.text,
            "play Jazz",
            "en",
        )
        == "play Jazz"
    )
    assert runtime.diagnostics.dynamic_candidate_count >= 1


async def test_rank_with_dynamic_candidates_does_not_starve_later_intents() -> None:
    """Rank exact dynamic candidates from later intents before earlier fuzzy candidates."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "built_in": {
            "intents": {
                "HassTurnOff": {
                    "data": [
                        {
                            "sentences": [
                                "stop {name}",
                                "disable {name}",
                                "turn {name} off",
                                "power down {name}",
                                "shut down {name}",
                                "switch {name} off",
                                "deactivate {name}",
                                "halt {name}",
                                "kill {name}",
                                "cut {name}",
                                "end {name}",
                            ],
                            "requires_context": {"domain": "light"},
                        }
                    ]
                },
                "HassTurnOn": {
                    "data": [
                        {
                            "sentences": ["turn {name} on"],
                            "requires_context": {"domain": "light"},
                        }
                    ]
                },
            }
        }
    }
    names = ("tail light", *(f"tail light alias {index}" for index in range(30)))
    registry_slots = {"name": names, "name:light": names}

    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values(registry_slots)
    runtime.language_intent_sources["en"] = intent_sources
    ranked = runtime.rank_with_dynamic_candidates(
        "en",
        build_index("en", [Candidate(text="unrelated", intent_name="HassNevermind")]),
        "turn tail light on",
    )

    assert ranked[0].candidate.intent_name == "HassTurnOn"
    assert ranked[0].candidate.normalized_text == "turn tail light on"
    assert ranked[0].scores.final_score == 1.0


def test_rank_with_dynamic_candidates_skips_dynamic_for_static_exact(
    monkeypatch: Any,
) -> None:
    """Avoid query-scoped dynamic work when static ranking already matched exactly."""
    runtime = CanonicalizerRuntime()
    index = build_index(
        "en",
        [
            Candidate(
                text="turn on kitchen light",
                intent_name="HassTurnOn",
                language="en",
            )
        ],
    )

    monkeypatch.setattr(
        runtime_module,
        "build_query_registry_candidates",
        _fail_dynamic_registry_build,
    )

    ranked = runtime.rank_with_dynamic_candidates("en", index, "turn on kitchen light")

    assert len(ranked) == 1
    assert ranked[0].candidate.normalized_text == "turn on kitchen light"
    assert ranked[0].scores.final_score == 1.0
    assert runtime.diagnostics.dynamic_candidate_count == 0


def test_rank_with_dynamic_candidates_keeps_dynamic_for_non_exact_static(
    monkeypatch: Any,
) -> None:
    """Still build exact dynamic candidates when static ranking is only fuzzy."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "built_in": {
            "intents": {
                "HassTurnOn": {
                    "data": [
                        {
                            "sentences": ["turn on {name}"],
                            "requires_context": {"domain": "light"},
                        }
                    ]
                }
            }
        }
    }
    registry_slots = {"name": ("kitchen light",), "name:light": ("kitchen light",)}
    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values(registry_slots)
    runtime.language_intent_sources["en"] = intent_sources
    index = build_index(
        "en",
        [
            Candidate(
                text="turn on kitchen lamp",
                intent_name="HassTurnOn",
                language="en",
            )
        ],
    )
    original_build = runtime_module.build_query_registry_candidates
    build_counter = _DynamicBuildCounter(original_build)

    monkeypatch.setattr(
        runtime_module,
        "build_query_registry_candidates",
        build_counter,
    )
    monkeypatch.setattr(
        runtime_module.CharNGramIndex,
        "from_grams",
        lambda *args, **kwargs: pytest.fail("dynamic exact matches should skip char index"),
    )

    ranked = runtime.rank_with_dynamic_candidates("en", index, "turn on kitchen light")

    assert build_counter.calls == 1
    assert ranked[0].candidate.normalized_text == "turn on kitchen light"
    assert ranked[0].scores.final_score == 1.0


def test_rank_with_dynamic_candidates_preserves_accepted_static_for_fuzzy_dynamic() -> None:
    """Avoid letting non-exact query-scoped candidates displace an accepted static match."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "built_in": {
            "lists": {"color": {"values": ["white"]}},
            "intents": {
                "HassLightSet": {
                    "data": [{"sentences": ["turn {name} {color}"]}],
                }
            },
        }
    }
    registry_slots = {"name": ("living room light",)}
    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values(registry_slots)
    runtime.language_intent_sources["en"] = intent_sources
    index = build_index(
        "en",
        [
            Candidate(
                text="turn living room light on",
                intent_name="HassTurnOn",
                language="en",
            )
        ],
    )

    ranked = runtime.rank_with_dynamic_candidates("en", index, "turn living room light")

    assert ranked[0].candidate.intent_name == "HassTurnOn"
    assert ranked[0].candidate.text == "turn living room light on"


def test_rank_with_dynamic_candidates_uses_single_registry_slot_snapshot(
    monkeypatch: Any,
) -> None:
    """Pass matching registry values and index to query-scoped dynamic generation."""
    runtime = CanonicalizerRuntime()
    old_values = {"name": ("old lamp",)}
    new_values = {"name": ("new lamp",)}
    runtime.update_registry_slot_values(old_values)
    runtime.language_intent_sources["en"] = {
        "built_in": {
            "intents": {
                "HassTurnOn": {
                    "data": [{"sentences": ["turn on {name}"]}],
                }
            }
        }
    }
    index = build_index("en", [Candidate(text="unrelated", intent_name="HassNevermind")])
    build_recorder = _DynamicBuildRecorder()

    def mutate_before_legacy_index_lookup(_runtime: CanonicalizerRuntime, _language: str) -> Any:
        runtime.update_registry_slot_values(new_values)
        return build_registry_slot_index(new_values, "en")

    monkeypatch.setattr(
        CanonicalizerRuntime,
        "_registry_slot_index_for_language",
        mutate_before_legacy_index_lookup,
    )
    monkeypatch.setattr(
        runtime_module,
        "build_query_registry_candidates",
        build_recorder,
    )

    runtime.rank_with_dynamic_candidates("en", index, "turn on old lamp")

    assert build_recorder.registry_slot_values == old_values
    assert build_recorder.registry_slot_index_name_values == ("old lamp",)


def test_runtime_precomputes_and_shares_registry_slot_records(monkeypatch: Any) -> None:
    """Normalize identical registry slot tuples once when refreshing the snapshot."""
    normalization_tracker = _NormalizationTracker(grammar_loader.normalize_text)

    monkeypatch.setattr(grammar_loader, "normalize_text", normalization_tracker)
    values_1 = ("Kitchen Light", "Desk Lamp")
    values_2 = ("Kitchen Light", "Desk Lamp")
    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values({"name": values_1, "entity": values_2})

    assert normalization_tracker.normalized_values == list(values_1)
    assert runtime.registry_slot_index["name"] is runtime.registry_slot_index["entity"]
    assert runtime.registry_slot_index["name"][0].normalized_text == "kitchen light"
    assert runtime.registry_slot_index["name"][0].tokens == ("kitchen", "light")


def test_runtime_registry_snapshot_update_replaces_dynamic_aliases() -> None:
    """Use the latest precomputed registry snapshot for dynamic candidates."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "built_in": {
            "intents": {
                "HassTurnOn": {
                    "data": [
                        {
                            "sentences": ["turn on {name}"],
                            "requires_context": {"domain": "light"},
                        }
                    ]
                }
            }
        }
    }
    runtime = CanonicalizerRuntime()
    runtime.language_intent_sources["en"] = intent_sources
    index = build_index("en", [Candidate(text="unrelated", intent_name="HassNevermind")])

    _test_runtime_registry_snapshot_update_replaces_dynamic_aliases(
        runtime, "old lamp", index, "turn on old lamp"
    )
    _test_runtime_registry_snapshot_update_replaces_dynamic_aliases(
        runtime, "new lamp", index, "turn on new lamp"
    )
    assert all(record.text != "old lamp" for record in runtime.registry_slot_index["name"])


def _test_runtime_registry_snapshot_update_replaces_dynamic_aliases(
    runtime: CanonicalizerRuntime,
    new_value: str,
    index: CanonicalIndex,
    query: str,
) -> None:
    """Helper for test_runtime_registry_snapshot_update_replaces_dynamic_aliases."""
    runtime.update_registry_slot_values({"name": (new_value,), "name:light": (new_value,)})
    ranked = runtime.rank_with_dynamic_candidates("en", index, query)
    assert ranked[0].candidate.normalized_text == query


def test_rank_with_dynamic_candidates_preserves_decimal_range_selection() -> None:
    """Do not let normalized decimal tokens demote the matching range candidate."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "builtin": {
            "lists": {
                "temperature": {
                    "range": {
                        "type": "temperature",
                        "from": 0,
                        "to": 100,
                        "fractions": "halves",
                    }
                }
            },
            "intents": {
                "HassClimateSetTemperature": {
                    "data": [{"sentences": ["set [the] {name} temperature to {temperature}"]}]
                }
            },
        }
    }
    static_index = build_index(
        "en",
        [
            Candidate(
                text="what's Large Bedroom temperature",
                intent_name="HassClimateGetTemperature",
                language="en",
                metadata={"literal_text": "what's temperature"},
            )
        ],
    )
    runtime = CanonicalizerRuntime()
    runtime.language_intent_sources["en"] = intent_sources
    runtime.update_registry_slot_values(
        {
            "name": ("Large Bedroom AC",),
            "name:climate": ("Large Bedroom AC",),
            "area": ("Large Bedroom",),
        }
    )

    ranked = runtime.rank_with_dynamic_candidates(
        "en",
        static_index,
        "set large bedroom temperature to 27.5",
    )

    assert ranked[0].candidate.intent_name == "HassClimateSetTemperature"
    assert ranked[0].candidate.text == "set Large Bedroom AC temperature to 27.5"


def test_rank_with_dynamic_candidates_preserves_comma_decimal_range_selection() -> None:
    """Do not let normalized comma-decimal tokens demote the matching range candidate."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "builtin": {
            "lists": {
                "temperature": {
                    "range": {
                        "type": "temperature",
                        "from": 0,
                        "to": 100,
                        "fractions": "halves",
                    }
                }
            },
            "intents": {
                "HassClimateSetTemperature": {
                    "data": [{"sentences": ["set [the] {name} temperature to {temperature}"]}]
                }
            },
        }
    }
    static_index = build_index(
        "en",
        [
            Candidate(
                text="what's Large Bedroom temperature",
                intent_name="HassClimateGetTemperature",
                language="en",
                metadata={"literal_text": "what's temperature"},
            )
        ],
    )
    runtime = CanonicalizerRuntime()
    runtime.language_intent_sources["en"] = intent_sources
    runtime.update_registry_slot_values(
        {
            "name": ("Large Bedroom AC",),
            "name:climate": ("Large Bedroom AC",),
            "area": ("Large Bedroom",),
        }
    )

    ranked = runtime.rank_with_dynamic_candidates(
        "en",
        static_index,
        "set large bedroom temperature to 27,5",
    )

    assert ranked[0].candidate.intent_name == "HassClimateSetTemperature"
    assert ranked[0].candidate.text == "set Large Bedroom AC temperature to 27,5"


def test_runtime_intent_update_invalidates_compiled_dynamic_templates() -> None:
    """Discard compiled templates when subscribed intent sources change."""
    runtime = CanonicalizerRuntime()
    runtime.update_intent_sources({"config": {"intents": {"OldIntent": {}}}})
    runtime.dynamic_registry_intents["en"] = ()

    runtime.update_intent_sources({"config": {"intents": {"NewIntent": {}}}})

    assert runtime.dynamic_registry_intents == {}


def test_runtime_intent_updates_merge_partial_sources_and_isolate_snapshots() -> None:
    """Retain unchanged sources across deltas and copy mutable callback payloads."""
    runtime = CanonicalizerRuntime()
    config_source: dict[str, Any] = {
        "intents": {"ConfigIntent": {"data": [{"sentences": ["config"]}]}}
    }
    trigger_source: dict[str, Any] = {
        "intents": {"TriggerIntent": {"data": [{"sentences": ["trigger"]}]}}
    }

    runtime.update_intent_sources(
        {
            "config": config_source,
            "trigger": trigger_source,
        }
    )
    config_source["intents"]["ConfigIntent"]["data"][0]["sentences"][0] = "mutated"
    runtime.update_intent_sources(
        {"trigger": {"intents": {"UpdatedTrigger": {"data": [{"sentences": ["updated trigger"]}]}}}}
    )

    assert set(runtime.intent_sources) == {"config", "trigger"}
    assert runtime.intent_sources["config"]["intents"]["ConfigIntent"]["data"][0]["sentences"] == [
        "config"
    ]
    assert "UpdatedTrigger" in runtime.intent_sources["trigger"]["intents"]

    runtime.update_intent_sources({"trigger": {}})

    assert "config" in runtime.intent_sources
    assert runtime.intent_sources["trigger"] == {}


def test_runtime_normalizes_language_cache_keys() -> None:
    """Store and retrieve indexes with canonical language keys."""
    runtime = CanonicalizerRuntime()
    runtime.set_index(build_index("Vi", [Candidate(text="bật đèn", intent_name="HassTurnOn")]))

    assert runtime.get_index("vi") is not None
    assert runtime.get_index("VI") is runtime.get_index("vi")
    assert runtime.get_index("vi-VN") is runtime.get_index("vi")
    assert sorted(runtime.indexes) == ["vi"]


@pytest.mark.current_intents
def test_normalize_language_preserves_supported_regional_variants() -> None:
    """Keep regional variants that have distinct Home Assistant language packs."""
    assert normalize_language("vi-VN") == "vi"
    assert normalize_language("en_US") == "en"
    assert normalize_language("pt-br") == "pt-BR"
    assert normalize_language("de-ch") == "de-CH"
    assert normalize_language("zh-tw") == "zh-TW"


def test_normalize_language_rejects_empty_values() -> None:
    """Reject empty language keys before they reach caches."""
    with pytest.raises(ValueError, match="Language must not be empty"):
        normalize_language(" ")


class MockStore:
    """Mock Home Assistant Store helper for runtime tests."""

    stored_data: ClassVar[dict[str, dict[str, Any]]] = {}
    removed_keys: ClassVar[list[str]] = []

    def __init__(self, hass: Any, version: int, key: str, **kwargs: Any) -> None:
        """Initialize mock store with key and version."""
        self.hass = hass
        self.version = version
        self.key = key
        self.kwargs = kwargs

    async def async_load(self) -> dict[str, Any] | None:
        """Simulate loading data."""
        return MockStore.stored_data.get(self.key)

    async def async_save(self, data: dict[str, Any]) -> None:
        """Simulate saving data."""
        MockStore.stored_data[self.key] = data

    async def async_remove(self) -> None:
        """Simulate removing persisted data."""
        MockStore.stored_data.pop(self.key, None)
        MockStore.removed_keys.append(self.key)

    @classmethod
    def reset(cls) -> None:
        """Reset all mock stores between tests."""
        cls.stored_data = {}
        cls.removed_keys = []


class HashableFakeHass:
    """Hashable fake Home Assistant object for testing."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize dictionary with attributes."""
        self.executor_jobs: list[tuple[Any, tuple[Any, ...]]] = []
        self.__dict__.update(kwargs)

    def __hash__(self) -> int:
        """Return hash value based on object identity."""
        return id(self)

    def __eq__(self, other: Any) -> bool:
        """Check equality by object identity."""
        return self is other

    def add_job(self, target: Any, *args: Any) -> Any:
        """Mock add_job by calling target or creating a task."""
        if inspect.iscoroutinefunction(target):
            return asyncio.create_task(target(*args))
        return target(*args)

    async def async_add_executor_job(self, target: Any, *args: Any) -> Any:
        """Mock executor submission by recording and running the job."""
        self.executor_jobs.append((target, args))
        return target(*args)


async def test_persistent_store_save_and_load(monkeypatch: Any) -> None:
    """Verify that indices can be saved and loaded from the persistent store."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)

    runtime = CanonicalizerRuntime()
    hass = HashableFakeHass()

    candidates = [
        Candidate(
            text="bật đèn",
            intent_name="HassTurnOn",
            source=CandidateSource.BUILT_IN,
            language="vi",
            metadata={"sentence_template": "bật {name}"},
            slot_values=("đèn",),
        )
    ]
    index = build_index("vi", candidates)

    MockStore.reset()
    try:
        snapshot = _create_build_snapshot_and_register_wildcards(
            "vi", *runtime._capture_build_inputs()
        )
        await runtime.async_save_index_to_store(hass, index, snapshot.fingerprint)

        stored_index = MockStore.stored_data["assist_canonicalizer.index_vi"]
        assert stored_index["fingerprint"] == snapshot.fingerprint
        assert stored_index["candidate_count"] == 1
        stored_candidates = stored_index["candidates"]
        assert len(stored_candidates) == 1
        assert stored_candidates[0]["text"] == "bật đèn"
        assert stored_candidates[0]["intent_name"] == "HassTurnOn"
        assert stored_candidates[0]["source"] == "built_in"
        assert stored_candidates[0]["metadata"]["sentence_template"] == "bật {name}"
        assert stored_candidates[0]["slot_values"] == ["đèn"]

        clean_runtime = CanonicalizerRuntime()
        loaded_index = await clean_runtime.async_load_index_from_store(hass, "vi")
        assert loaded_index is not None
        assert loaded_index.language == "vi"
        assert loaded_index.candidate_count == 1
        assert len(hass.executor_jobs) == 2
        executor_target, executor_args = hass.executor_jobs[1]
        assert executor_target is build_index
        assert executor_args[0] == "vi"
        assert len(executor_args[1]) == 1
        loaded_cand = loaded_index.candidates[0]
        assert loaded_cand.text == "bật đèn"
        assert loaded_cand.intent_name == "HassTurnOn"
        assert loaded_cand.source == CandidateSource.BUILT_IN
        assert loaded_cand.metadata["sentence_template"] == "bật {name}"
        assert loaded_cand.slot_values == ("đèn",)
        assert clean_runtime.get_index("vi") is loaded_index
        assert clean_runtime.dynamic_registry_intents["vi"] == snapshot.dynamic_registry_intents
    finally:
        MockStore.reset()


async def test_persistent_store_rejects_old_build_version(monkeypatch: Any) -> None:
    """Reject a persisted index produced by an older candidate build version."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()
    hass = HashableFakeHass()
    runtime = CanonicalizerRuntime()
    snapshot = _create_build_snapshot_and_register_wildcards("en", *runtime._capture_build_inputs())
    index = build_index(
        "en",
        [Candidate(text="turn on light", intent_name="HassTurnOn", language="en")],
    )
    await runtime.async_save_index_to_store(hass, index, snapshot.fingerprint)
    MockStore.stored_data["assist_canonicalizer.index_en"]["build_version"] = (
        _INDEX_BUILD_VERSION - 1
    )

    loaded = await CanonicalizerRuntime().async_load_index_from_store(hass, "en")

    assert loaded is None
    assert "assist_canonicalizer.index_en" not in MockStore.stored_data
    assert MockStore.stored_data["assist_canonicalizer.index_manifest"]["languages"] == []


async def test_persistent_store_rejects_stale_fingerprint(monkeypatch: Any) -> None:
    """Reject and remove a persisted index after registry inputs change."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()
    hass = HashableFakeHass()
    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values({"name": ("old lamp",)})
    snapshot = _create_build_snapshot_and_register_wildcards("en", *runtime._capture_build_inputs())
    index = build_index("en", [Candidate(text="turn on old lamp", intent_name="HassTurnOn")])
    await runtime.async_save_index_to_store(hass, index, snapshot.fingerprint)

    clean_runtime = CanonicalizerRuntime()
    clean_runtime.update_registry_slot_values({"name": ("new lamp",)})
    loaded = await clean_runtime.async_load_index_from_store(hass, "en")

    assert loaded is None
    assert "assist_canonicalizer.index_en" not in MockStore.stored_data
    assert MockStore.stored_data["assist_canonicalizer.index_manifest"]["languages"] == []


async def test_persistent_store_rejects_malformed_candidate(monkeypatch: Any) -> None:
    """Reject the entire cache instead of silently loading a partial index."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()
    hass = HashableFakeHass()
    runtime = CanonicalizerRuntime()
    snapshot = _create_build_snapshot_and_register_wildcards("en", *runtime._capture_build_inputs())
    index = build_index("en", [Candidate(text="turn on light", intent_name="HassTurnOn")])
    await runtime.async_save_index_to_store(hass, index, snapshot.fingerprint)
    MockStore.stored_data["assist_canonicalizer.index_en"]["candidates"][0].pop("normalized_text")

    loaded = await CanonicalizerRuntime().async_load_index_from_store(hass, "en")

    assert loaded is None
    assert "assist_canonicalizer.index_en" not in MockStore.stored_data


async def test_async_clear_index_removes_specific_and_all_stores(monkeypatch: Any) -> None:
    """Remove persisted data for one language and rotate the epoch for clear-all."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()
    hass = HashableFakeHass()
    runtime = CanonicalizerRuntime()
    for language in ("en", "vi"):
        snapshot = _create_build_snapshot_and_register_wildcards(
            language, *runtime._capture_build_inputs()
        )
        index = build_index(
            language,
            [Candidate(text=f"sample {language}", intent_name="Sample", language=language)],
        )
        runtime.set_index(index)
        await runtime.async_save_index_to_store(hass, index, snapshot.fingerprint)

    await runtime.async_clear_index(hass, "en-US")
    assert "en" not in runtime.indexes
    assert "vi" in runtime.indexes
    assert "assist_canonicalizer.index_en" not in MockStore.stored_data
    manifest = MockStore.stored_data["assist_canonicalizer.index_manifest"]
    old_epoch = manifest["cache_epoch"]
    assert manifest["languages"] == ["vi"]

    await runtime.async_clear_index(hass)
    assert runtime.indexes == {}
    assert "assist_canonicalizer.index_vi" not in MockStore.stored_data
    manifest = MockStore.stored_data["assist_canonicalizer.index_manifest"]
    assert manifest["languages"] == []
    assert manifest["cache_epoch"] != old_epoch


async def test_async_rebuild_retries_after_source_generation_changes(monkeypatch: Any) -> None:
    """Do not publish or persist a build whose source snapshot became stale."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()
    hass = HashableFakeHass(async_create_task=lambda coro: asyncio.create_task(coro))
    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values({"name": ("old lamp",)})
    build_counter = _SourceChangingBuild(runtime)

    monkeypatch.setattr(runtime_module, "_build_index_from_snapshot", build_counter)

    index = await runtime.async_rebuild_index(hass, "en")

    assert build_counter.calls == 2
    assert index is not None
    assert index.candidates[0].text == "turn on new lamp"
    assert runtime.get_index("en") is index
    stored = MockStore.stored_data["assist_canonicalizer.index_en"]
    assert stored["candidates"][0]["text"] == "turn on new lamp"


async def test_async_rebuild_stops_after_repeated_source_generation_changes(
    monkeypatch: Any,
) -> None:
    """Bound rebuild retries when source inputs never stabilize."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()
    hass = HashableFakeHass(async_create_task=lambda coro: asyncio.create_task(coro))
    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values({"name": ("old lamp",)})
    build_counter = _AlwaysSourceChangingBuild(runtime)

    monkeypatch.setattr(runtime_module, "_build_index_from_snapshot", build_counter)

    index = await runtime.async_rebuild_index(hass, "en")

    assert build_counter.calls == runtime_module._MAX_REBUILD_ATTEMPTS
    assert index is None
    assert runtime.get_index("en") is None
    assert "assist_canonicalizer.index_en" not in MockStore.stored_data


async def test_async_rebuild_does_not_publish_after_index_clear(monkeypatch: Any) -> None:
    """Do not repopulate memory or storage after clear_index invalidates a rebuild."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()
    hass = HashableFakeHass(async_create_task=lambda coro: asyncio.create_task(coro))
    runtime = CanonicalizerRuntime()
    build_counter = _IndexClearingBuild(runtime)

    monkeypatch.setattr(runtime_module, "_build_index_from_snapshot", build_counter)

    index = await runtime.async_rebuild_index(hass, "en")

    assert build_counter.calls == 1
    assert index is None
    assert runtime.get_index("en") is None
    assert "assist_canonicalizer.index_en" not in MockStore.stored_data
    assert runtime.rebuild_tasks == {}


async def test_language_clear_does_not_invalidate_other_language_rebuild(
    monkeypatch: Any,
) -> None:
    """Publish a VI rebuild that overlaps an unrelated EN clear."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()
    runtime = CanonicalizerRuntime()
    hass = FakeHass()
    build_counter = _SnapshotBuildCounter()
    monkeypatch.setattr(runtime_module, "_build_index_from_snapshot", build_counter)

    rebuild_task = asyncio.create_task(runtime.async_rebuild_index(hass, "vi"))
    await hass.job_started.wait()
    runtime.clear_index("en")
    hass.release_job.set()

    index = await rebuild_task

    assert index is not None
    assert index.language == "vi"
    assert runtime.get_index("vi") is index
    assert build_counter.calls == 1


async def test_async_shutdown_prevents_blocked_rebuild_publication(monkeypatch: Any) -> None:
    """Do not publish or persist a rebuild that finishes after runtime shutdown starts."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()
    runtime = CanonicalizerRuntime()
    build_started = threading.Event()
    build_finished = threading.Event()
    release_build = threading.Event()
    save_called = False
    set_index_called = False

    def blocking_build(snapshot: Any) -> CanonicalIndex:
        """Block inside executor-backed index construction."""
        build_started.set()
        assert release_build.wait(timeout=5)
        build_finished.set()
        return build_index(
            snapshot.language,
            [Candidate(text="turn on light", intent_name="HassTurnOn", language=snapshot.language)],
        )

    class ThreadedBuildHass(HashableFakeHass):
        """Fake hass that runs the build step in a real executor thread."""

        def async_create_task(self, coro: Any) -> asyncio.Task[Any]:
            """Create an asyncio task."""
            return asyncio.create_task(coro)

        async def async_add_executor_job(self, target: Any, *args: Any) -> Any:
            """Run the blocked build in an executor thread."""
            if target is blocking_build:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, target, *args)
            return target(*args)

    original_save = CanonicalizerRuntime.async_save_index_to_store

    async def record_save(
        self: CanonicalizerRuntime,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Record unexpected persistence attempts."""
        nonlocal save_called
        if self is runtime:
            save_called = True
        return await original_save(self, *args, **kwargs)

    original_set_index = CanonicalizerRuntime.set_index

    def record_set_index(self: CanonicalizerRuntime, index: CanonicalIndex) -> None:
        """Record unexpected in-memory publication attempts."""
        nonlocal set_index_called
        if self is runtime:
            set_index_called = True
            return
        original_set_index(self, index)

    monkeypatch.setattr(runtime_module, "_build_index_from_snapshot", blocking_build)
    monkeypatch.setattr(CanonicalizerRuntime, "async_save_index_to_store", record_save)
    monkeypatch.setattr(CanonicalizerRuntime, "set_index", record_set_index)
    hass = ThreadedBuildHass()

    rebuild_task = asyncio.create_task(runtime.async_rebuild_index(hass, "en"))
    assert await asyncio.to_thread(build_started.wait, 5)
    shutdown_task = asyncio.create_task(runtime.async_shutdown())
    await asyncio.sleep(0)
    release_build.set()

    try:
        rebuild_result = await rebuild_task
    except asyncio.CancelledError:
        rebuild_result = None
    await shutdown_task

    assert rebuild_result is None
    assert build_finished.is_set()
    assert not save_called
    assert not set_index_called
    assert runtime.get_index("en") is None
    assert "assist_canonicalizer.index_en" not in MockStore.stored_data
    assert runtime.rebuild_tasks == {}


async def test_async_shutdown_waits_for_active_store_operation(monkeypatch: Any) -> None:
    """Use the storage lock as a barrier for saves that already started."""
    save_started = asyncio.Event()
    release_save = asyncio.Event()

    class BlockingSaveStore(MockStore):
        """Mock store that blocks index saves until released."""

        async def async_save(self, data: dict[str, Any]) -> None:
            """Block the language index save to test the shutdown barrier."""
            if self.key == "assist_canonicalizer.index_en":
                save_started.set()
                await release_save.wait()
            await super().async_save(data)

    monkeypatch.setattr(homeassistant.helpers.storage, "Store", BlockingSaveStore)
    BlockingSaveStore.reset()
    runtime = CanonicalizerRuntime()
    hass = HashableFakeHass()
    index = build_index("en", [Candidate(text="turn on light", intent_name="HassTurnOn")])

    save_task = asyncio.create_task(runtime.async_save_index_to_store(hass, index, "fingerprint"))
    await save_started.wait()
    shutdown_task = asyncio.create_task(runtime.async_shutdown())
    await asyncio.sleep(0)

    assert not shutdown_task.done()

    release_save.set()
    assert await save_task is False
    await shutdown_task
    assert "assist_canonicalizer.index_manifest" not in BlockingSaveStore.stored_data


async def test_async_shutdown_drains_cancelled_rebuild_store_writer(monkeypatch: Any) -> None:
    """Wait for executor-backed storage even when shutdown cancels its rebuild."""
    save_started = threading.Event()
    release_save = threading.Event()
    save_finished = threading.Event()

    class ExecutorBackedSaveStore(MockStore):
        """Mock a Home Assistant store whose file write runs in an executor."""

        async def async_save(self, data: dict[str, Any]) -> None:
            """Run the language-store write in a thread that cancellation cannot stop."""
            if self.key != "assist_canonicalizer.index_en":
                await super().async_save(data)
                return

            def blocking_write() -> None:
                """Block the executor writer until the test releases it."""
                save_started.set()
                assert release_save.wait(timeout=30)
                MockStore.stored_data[self.key] = data
                save_finished.set()

            await asyncio.get_running_loop().run_in_executor(None, blocking_write)

    monkeypatch.setattr(homeassistant.helpers.storage, "Store", ExecutorBackedSaveStore)
    MockStore.reset()
    runtime = CanonicalizerRuntime()
    hass = HashableFakeHass(async_create_task=lambda coro: asyncio.create_task(coro))

    rebuild_task = asyncio.create_task(runtime.async_rebuild_index(hass, "en"))
    assert await asyncio.to_thread(save_started.wait, 30)
    shutdown_task = asyncio.create_task(runtime.async_shutdown())
    await asyncio.sleep(0.05)

    assert not shutdown_task.done()
    assert not save_finished.is_set()

    release_save.set()
    await asyncio.gather(rebuild_task, return_exceptions=True)
    await shutdown_task

    assert save_finished.is_set()
    assert "assist_canonicalizer.index_en" in MockStore.stored_data
    assert "assist_canonicalizer.index_manifest" not in MockStore.stored_data


async def test_async_shutdown_drains_direct_store_load_build(monkeypatch: Any) -> None:
    """Wait for a direct persisted-index build before completing shutdown."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()
    hass = HashableFakeHass()
    source_runtime = CanonicalizerRuntime()
    snapshot = _create_build_snapshot_and_register_wildcards(
        "en",
        *source_runtime._capture_build_inputs(),
    )
    stored_index = build_index(
        "en",
        [Candidate(text="turn on light", intent_name="HassTurnOn", language="en")],
    )
    assert await source_runtime.async_save_index_to_store(
        hass,
        stored_index,
        snapshot.fingerprint,
    )

    build_started = threading.Event()
    release_build = threading.Event()
    build_finished = threading.Event()

    def blocking_build(language: str, candidates: Any) -> CanonicalIndex:
        """Block reconstruction of the loaded index in a real executor thread."""
        build_started.set()
        assert release_build.wait(timeout=5)
        rebuilt = build_index(language, candidates)
        build_finished.set()
        return rebuilt

    class ThreadedLoadHass(HashableFakeHass):
        """Fake hass that runs persisted-index reconstruction in an executor."""

        async def async_add_executor_job(self, target: Any, *args: Any) -> Any:
            """Run only the patched index builder in a real executor thread."""
            if target is blocking_build:
                return await asyncio.get_running_loop().run_in_executor(None, target, *args)
            return target(*args)

    monkeypatch.setattr(runtime_module, "build_index", blocking_build)
    runtime = CanonicalizerRuntime()
    load_task = asyncio.create_task(runtime.async_load_index_from_store(ThreadedLoadHass(), "en"))
    assert await asyncio.to_thread(build_started.wait, 5)
    shutdown_task = asyncio.create_task(runtime.async_shutdown())
    await asyncio.sleep(0.05)

    assert not shutdown_task.done()
    assert not build_finished.is_set()

    release_build.set()
    assert await load_task is None
    await shutdown_task

    assert build_finished.is_set()
    assert runtime.get_index("en") is None


async def test_async_shutdown_cancels_and_drains_rebuild_tasks() -> None:
    """Cancel and await all tracked per-language rebuild tasks."""
    runtime = CanonicalizerRuntime()

    async def pending_rebuild() -> CanonicalIndex | None:
        """Never finish unless canceled by shutdown."""
        await asyncio.Event().wait()
        return None

    task_en = asyncio.create_task(pending_rebuild())
    task_vi = asyncio.create_task(pending_rebuild())
    runtime.rebuild_tasks["en"] = (runtime._index_generation_for("en"), task_en)
    runtime.rebuild_tasks["vi"] = (runtime._index_generation_for("vi"), task_vi)

    await runtime.async_shutdown()

    assert task_en.cancelled()
    assert task_vi.cancelled()
    assert runtime.rebuild_tasks == {}
    assert runtime.warmup_tasks == set()


async def test_async_shutdown_cancels_and_drains_tracked_warmup_task() -> None:
    """Cancel and await a warmup registered through the public tracking API."""
    runtime = CanonicalizerRuntime()
    warmup_started = asyncio.Event()

    async def pending_warmup() -> None:
        """Remain active until runtime shutdown cancels the warmup."""
        warmup_started.set()
        await asyncio.Event().wait()

    warmup_task = asyncio.create_task(pending_warmup())
    runtime.track_warmup_task(warmup_task)
    await warmup_started.wait()

    assert warmup_task in runtime.warmup_tasks

    await runtime.async_shutdown()

    assert warmup_task.cancelled()
    assert runtime.warmup_tasks == set()


async def test_async_await_drained_retries_repeated_cancellation() -> None:
    """Drain work after every cancellation delivered while it remains active."""
    work_started = asyncio.Event()
    release_work = asyncio.Event()
    work_finished = asyncio.Event()

    async def pending_work() -> None:
        """Block until the test permits the drained operation to complete."""
        work_started.set()
        await release_work.wait()
        work_finished.set()

    drain_task = asyncio.create_task(runtime_module._async_await_drained(pending_work()))
    await work_started.wait()

    drain_task.cancel()
    await asyncio.sleep(0)
    drain_task.cancel()
    await asyncio.sleep(0)

    assert not drain_task.done()

    release_work.set()
    with pytest.raises(asyncio.CancelledError):
        await drain_task

    assert work_finished.is_set()


async def test_async_rebuild_index_returns_none_after_shutdown() -> None:
    """Do not create rebuild work after the runtime is closed."""
    runtime = CanonicalizerRuntime()
    await runtime.async_shutdown()
    hass = HashableFakeHass(async_create_task=lambda coro: pytest.fail("no task expected"))

    result = await runtime.async_rebuild_index(hass, "en")

    assert result is None
    assert runtime.rebuild_tasks == {}


async def test_debounced_rebuild_coalesces_events(monkeypatch: Any) -> None:
    """Verify that multiple registry events are debounced and only rebuild once."""
    scheduled_callbacks = []

    monkeypatch.setattr(
        homeassistant.helpers.event,
        "async_call_later",
        _AsyncCallLaterRecorder(scheduled_callbacks),
    )

    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = build_index(
        "en", [Candidate(text="turn on light", intent_name="HassTurnOn")]
    )

    listeners = []

    class FakeBus:
        """Fake bus mock."""

        def async_listen(self, event_type: str, callback: Any) -> Any:
            """Register listener."""
            listeners.append(callback)
            return lambda: None

    async def async_add_executor_job_mock(func: Any, *args: Any) -> Any:
        """Mock async_add_executor_job by executing the function synchronously."""
        return func(*args)

    hass = HashableFakeHass(
        bus=FakeBus(),
        async_create_task=lambda coro: asyncio.create_task(coro),
        async_add_executor_job=async_add_executor_job_mock,
    )

    exposed_entities_module = ModuleType("homeassistant.components.homeassistant.exposed_entities")
    attr_name = "async_listen_entity_updates"
    setattr(
        exposed_entities_module,
        attr_name,
        lambda h, engine, cb: cb() or listeners.append(cb) or (lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "homeassistant.components.homeassistant.exposed_entities",
        exposed_entities_module,
    )
    monkeypatch.setattr(integration, "exposed_entities", exposed_entities_module)

    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.registry.async_registry_slot_values",
        lambda h: {"name": ("light",)},
    )

    rebuild_calls = 0
    rebuild_called = asyncio.Event()

    async def async_rebuild_index_mock(
        _runtime: CanonicalizerRuntime,
        _hass: Any,
        _language: str,
    ) -> None:
        """Record a rebuild without starting unrelated index construction."""
        nonlocal rebuild_calls
        rebuild_calls += 1
        rebuild_called.set()

    monkeypatch.setattr(
        CanonicalizerRuntime,
        "async_rebuild_index",
        async_rebuild_index_mock,
    )

    _subscribe_registry_updates(hass, runtime)

    # Trigger entity registry update
    listeners[0]({})
    assert len(scheduled_callbacks) == 1

    # Trigger another entity registry update, should debounce
    listeners[0]({})
    assert len(scheduled_callbacks) == 1

    # Fire debounced function
    scheduled_callbacks[0]()

    await asyncio.wait_for(rebuild_called.wait(), timeout=1)

    assert rebuild_calls == 1


def test_canonical_fingerprint_value_sorting() -> None:
    """Verify that _canonical_fingerprint_value is order-insensitive for mappings."""
    dict_a = {"a": 1, "b": 2}
    dict_b = {"b": 2, "a": 1}

    fp_a = _canonical_fingerprint_value(dict_a)
    fp_b = _canonical_fingerprint_value(dict_b)

    assert fp_a == fp_b

    # Test nested mappings
    nested_a = {"x": {"a": 1, "b": 2}, "y": 3}
    nested_b = {"y": 3, "x": {"b": 2, "a": 1}}

    assert _canonical_fingerprint_value(nested_a) == _canonical_fingerprint_value(nested_b)


async def test_rank_with_dynamic_candidates_filters_exact_matches() -> None:
    """Verify that only candidates with score >= 1.0 are returned if an exact match exists."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "built_in": {
            "intents": {
                "HassTurnOn": {
                    "data": [
                        {
                            "sentences": ["bật {name}"],
                            "requires_context": {"domain": "light"},
                        }
                    ]
                }
            }
        }
    }
    registry_slots = {
        "name": ("đèn phòng khách", "đèn bếp"),
        "name:light": ("đèn phòng khách", "đèn bếp"),
    }

    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values(registry_slots)
    runtime.language_intent_sources["vi"] = intent_sources

    indexed_candidates = [
        Candidate(
            text="bật đèn bếp",
            intent_name="HassTurnOn",
            source=CandidateSource.BUILT_IN,
            language="vi",
            metadata={"literal_text": "bật"},
        )
    ]
    index = build_index("vi", indexed_candidates)

    ranked = runtime.rank_with_dynamic_candidates("vi", index, "bật đèn bếp")

    assert len(ranked) == 1
    assert ranked[0].candidate.normalized_text == "bật đèn bếp"
    assert ranked[0].scores.final_score == 1.0


async def test_rank_with_dynamic_candidates_preserves_multiple_exact_matches() -> None:
    """Verify that multiple exact matches (with different intents) are all preserved."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "built_in": {
            "intents": {
                "HassShoppingListAddItem": {
                    "data": [
                        {
                            "sentences": ["đặt {name} vào danh sách mua sắm"],
                        }
                    ]
                },
                "HassListAddItem": {
                    "data": [
                        {
                            "sentences": ["đặt {name} vào danh sách mua sắm"],
                        }
                    ]
                },
            }
        }
    }
    registry_slots = {"name": ("sữa",), "name:shopping_list": ("sữa",)}

    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values(registry_slots)
    runtime.language_intent_sources["vi"] = intent_sources

    indexed_candidates = [
        Candidate(
            text="đặt sữa vào danh sách mua sắm",
            intent_name="HassShoppingListAddItem",
            source=CandidateSource.BUILT_IN,
            language="vi",
            metadata={"literal_text": "đặt|vào|danh|sách|mua|sắm"},
        ),
        Candidate(
            text="đặt sữa vào danh sách mua sắm",
            intent_name="HassListAddItem",
            source=CandidateSource.BUILT_IN,
            language="vi",
            metadata={"literal_text": "đặt|vào|danh|sách|mua|sắm"},
        ),
    ]
    index = build_index("vi", indexed_candidates)

    ranked = runtime.rank_with_dynamic_candidates("vi", index, "đặt sữa vào danh sách mua sắm")

    assert len(ranked) == 2
    expected_intents = {"HassShoppingListAddItem", "HassListAddItem"}
    assert {r.candidate.intent_name for r in ranked} == expected_intents
    assert all(r.scores.final_score == 1.0 for r in ranked)


async def test_rank_dynamic_ambiguous_exact_slot_preferences() -> None:
    """Verify that slot preferences resolve ties among multiple exact dynamic matches."""
    intent_sources: dict[str, Mapping[str, Any]] = {
        "built_in": {
            "intents": {
                "IntentA": {
                    "data": [
                        {
                            "sentences": ["add {slot_a} to list"],
                        }
                    ]
                },
                "IntentB": {
                    "data": [
                        {
                            "sentences": ["add {slot_b} to list"],
                        }
                    ]
                },
            }
        }
    }
    registry_slots = {"slot_a": ("milk",), "slot_b": ("milk",)}

    runtime = CanonicalizerRuntime()
    # Mock known wildcard slot names so they are recognized as wildcards
    with patch(
        "custom_components.assist_canonicalizer.candidate.wildcard_slot_names_sorted",
        return_value=("slot_a", "slot_b"),
    ):
        runtime.update_registry_slot_values(registry_slots)
        runtime.language_intent_sources["en"] = intent_sources

        indexed_candidates = [
            Candidate(
                text="add slot_a to list",
                intent_name="IntentA",
                source=CandidateSource.BUILT_IN,
                language="en",
                metadata={"literal_text": "add|to|list"},
            ),
            Candidate(
                text="add slot_b to list",
                intent_name="IntentB",
                source=CandidateSource.BUILT_IN,
                language="en",
                metadata={"literal_text": "add|to|list"},
            ),
        ]
        index = build_index("en", indexed_candidates)

        # Call without preferences first. Default order from the index list (IntentA first)
        ranked_no_prefs = runtime.rank_with_dynamic_candidates("en", index, "add milk to list")
        assert len(ranked_no_prefs) == 2
        assert ranked_no_prefs[0].candidate.intent_name == "IntentA"

        # Call with preference for IntentB's slot
        ranked_with_prefs = runtime.rank_with_dynamic_candidates(
            "en",
            index,
            "add milk to list",
            slot_preferences={("slot_b", "milk")},
        )
        assert len(ranked_with_prefs) == 2
        assert ranked_with_prefs[0].candidate.intent_name == "IntentB"


def test_rebuild_index_synchronous() -> None:
    """Verify that index building from snapshot constructs correct candidates."""
    runtime = CanonicalizerRuntime()
    intent_sources = {
        "built_in": {
            "intents": {
                "HassTurnOn": {
                    "data": [
                        {
                            "sentences": ["bật {name}"],
                        }
                    ]
                }
            }
        }
    }
    runtime.update_intent_sources(intent_sources)
    runtime.update_registry_slot_values({"name": ("đèn",)})

    with patch(
        "custom_components.assist_canonicalizer.runtime.load_language_intent_sources",
        return_value={},
    ):
        snapshot = _create_build_snapshot_and_register_wildcards(
            "vi", *runtime._capture_build_inputs()
        )

        index = _build_index_from_snapshot(snapshot)

    assert snapshot.dynamic_registry_intents
    assert index.language == "vi"
    assert index.candidate_count > 0
    assert index.candidates[0].text == "bật đèn"


async def test_async_rebuild_index_real_flow(monkeypatch: Any) -> None:
    """Verify async_rebuild_index runs the real build snapshot and build index functions."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()
    runtime = CanonicalizerRuntime()
    intent_sources = {
        "built_in": {
            "intents": {
                "HassTurnOn": {
                    "data": [
                        {
                            "sentences": ["bật {name}"],
                        }
                    ]
                }
            }
        }
    }
    runtime.update_intent_sources(intent_sources)
    runtime.update_registry_slot_values({"name": ("đèn",)})

    class DummyHass:
        """Dummy Home Assistant instance for async executor testing."""

        async def async_add_executor_job(self, func: Any, *args: Any) -> Any:
            """Execute a function synchronously."""
            return func(*args)

        def async_create_task(self, coro: Any) -> Any:
            """Create and run an asyncio task."""
            return asyncio.create_task(coro)

    hass = DummyHass()
    with patch(
        "custom_components.assist_canonicalizer.runtime.load_language_intent_sources",
        return_value={},
    ):
        index = await runtime.async_rebuild_index(hass, "vi")

    assert index is not None
    assert index.language == "vi"
    assert index.candidate_count > 0
    assert index.candidates[0].text == "bật đèn"
    assert runtime.get_index("vi") is index


class DummyEnum(Enum):
    """Dummy enum for fingerprint testing."""

    VAL1 = "val1"


def test_canonical_fingerprint_value_edge_cases() -> None:
    """Test Enum, Set, and unknown type fingerprinting in _canonical_fingerprint_value."""
    # Enum
    assert _canonical_fingerprint_value(DummyEnum.VAL1) == "val1"
    # Set
    s = {"b", "a"}
    fp_set = _canonical_fingerprint_value(s)
    assert isinstance(fp_set, dict)
    assert fp_set["set"] == ["a", "b"]
    # Unknown type
    obj = object()
    fp_obj = _canonical_fingerprint_value(obj)
    assert isinstance(fp_obj, dict)
    assert "object_type" in fp_obj
    assert fp_obj["representation"] == repr(obj)


def test_updated_optional_text_clear_with_none() -> None:
    """Test updated_optional_text behavior when clear is True and value is None."""
    assert _updated_optional_text("old", None, clear=True) is None
    assert _updated_optional_text("old", "new", clear=True) == "new"
    assert _updated_optional_text("old", None, clear=False) == "old"
    assert _updated_optional_text("old", "new", clear=False) == "new"


async def test_async_clear_index_specific_language() -> None:
    """Test clear_index for a specific language."""
    runtime = CanonicalizerRuntime()
    # Add dummy index
    runtime.indexes["en"] = build_index("en", [])
    runtime.indexes["vi"] = build_index("vi", [])

    class DummyStore:
        """Dummy store for testing."""

        async def async_remove(self) -> None:
            """Remove store."""

        async def async_load(self) -> Any:
            """Load store."""
            return {"cache_epoch": "epoch1", "languages": ["en", "vi"]}

        async def async_save(self, data: Any) -> None:
            """Save store."""

    class DummyHass:
        """Dummy Home Assistant for testing."""

        def __init__(self) -> None:
            """Initialize dummy hass."""
            self.data = {runtime_module.DOMAIN: DummyStore()}

    hass = DummyHass()

    store_patch = patch(
        "custom_components.assist_canonicalizer.runtime._index_store",
        return_value=DummyStore(),
    )
    manifest_patch = patch(
        "custom_components.assist_canonicalizer.runtime._manifest_store",
        return_value=DummyStore(),
    )
    with store_patch, manifest_patch:
        await runtime.async_clear_index(hass, "en")
    assert "en" not in runtime.indexes
    assert "vi" in runtime.indexes


def test_is_perfect_rank_result_false() -> None:
    """Test _is_perfect_rank_result with imperfect scores."""
    cand = Candidate(text="test", intent_name="test")
    rc = RankedCandidate(
        candidate=cand,
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=1.0,
            bm25_score=1.0,
            intent_score=1.0,
            final_score=0.9,
        ),
    )
    assert not _is_perfect_rank_result((rc,))


def test_valid_store_metadata_rejections() -> None:
    """Test metadata validations in _valid_store_metadata."""
    # Non-dict
    assert not _valid_store_metadata(None, language="en", fingerprint="fp", cache_epoch="ep")
    # Bad counts or list
    data = {
        "build_version": _INDEX_BUILD_VERSION,
        "language": "en",
        "fingerprint": "fp",
        "cache_epoch": "ep",
        "candidate_count": 5,
        "candidates": [],
    }
    assert not _valid_store_metadata(data, language="en", fingerprint="fp", cache_epoch="ep")


def test_deserialize_candidates_invalid() -> None:
    """Test deserialize_candidates handles invalid format or values."""
    # Candidate entry is not dict
    assert _deserialize_candidates({"candidates": [None]}) is None
    # Candidate entry missing keys or wrong types
    assert _deserialize_candidates({"candidates": [{"text": 123}]}) is None
    # Bad candidate source
    assert (
        _deserialize_candidates(
            {
                "candidates": [
                    {
                        "text": "test",
                        "intent_name": "test",
                        "source": "invalid_source",
                        "metadata": {},
                        "slot_values": [],
                        "normalized_text": "test",
                    }
                ]
            }
        )
        is None
    )


def test_merge_ranked_candidates_sorting() -> None:
    """Test merge_ranked_candidates preference sorting based on source priority."""
    c1 = Candidate(text="test", intent_name="intent1", source=CandidateSource.BUILT_IN)
    c2 = Candidate(text="test", intent_name="intent1", source=CandidateSource.CUSTOM_SENTENCE)
    rc1 = RankedCandidate(c1, ScoreBreakdown(1.0, 1.0, 1.0, 1.0, 0.8))
    rc2 = RankedCandidate(c2, ScoreBreakdown(1.0, 1.0, 1.0, 1.0, 0.8))

    merged = _merge_ranked_candidates((rc1,), (rc2,), max_candidates=2)
    # Custom sentence has higher priority (source_priority is lower, i.e., 0 vs 1)
    assert merged[0].candidate.source == CandidateSource.CUSTOM_SENTENCE


def test_merge_ranked_candidates_retains_meaningful_confidence_competitor() -> None:
    """Do not let one dynamic intent crowd every alternative out of the result cap."""
    primary_candidate = Candidate(
        text="open the bedroom window",
        intent_name="HassTurnOn",
        metadata={"slots": '{"name":"bedroom window"}'},
    )
    primary = (
        RankedCandidate(
            primary_candidate,
            ScoreBreakdown(0.49, 0.49, 0.49, 0.49, 0.49),
        ),
    )
    dynamic = tuple(
        RankedCandidate(
            Candidate(
                text=f"is bedroom window opening variant {index}",
                intent_name="HassGetState",
                metadata={
                    "slots": '{"name":"bedroom window","state":"opening"}',
                },
            ),
            ScoreBreakdown(
                0.80 - index / 1000,
                0.80 - index / 1000,
                0.80 - index / 1000,
                0.80 - index / 1000,
                0.80 - index / 1000,
            ),
        )
        for index in range(DEFAULT_MAX_CANDIDATES + 5)
    )

    merged = _merge_ranked_candidates(
        primary,
        dynamic,
        max_candidates=DEFAULT_MAX_CANDIDATES,
    )

    assert len(merged) == DEFAULT_MAX_CANDIDATES
    assert merged[-1].candidate is primary_candidate


def test_subscribed_source_counts_invalid() -> None:
    """Test subscribed_source_counts when intents is not a mapping."""
    runtime = CanonicalizerRuntime()
    runtime.intent_sources = {"test_src": {"intents": None}}
    assert runtime.subscribed_source_counts() == {"test_src": 0}


def test_registry_slot_index_caching_and_cleanup() -> None:
    """Test registry slot snapshot caching and cleanup logic."""
    runtime = CanonicalizerRuntime()
    runtime.registry_slot_values = {"name": ("light",)}

    # Call internal helper
    res1 = runtime._registry_slot_index_for_language("en")
    assert res1 is not None

    # Cleanup call
    assert runtime.rebuild_timer_cancel is None
    called = False

    def mock_cancel() -> None:
        nonlocal called
        called = True

    runtime.rebuild_timer_cancel = mock_cancel

    runtime.cleanup()
    assert called
    assert runtime.rebuild_timer_cancel is None
