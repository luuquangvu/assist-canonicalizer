"""Tests for runtime rebuild coordination."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any, ClassVar

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
from custom_components.assist_canonicalizer.const import DEFAULT_MAX_CANDIDATES_PER_TEMPLATE
from custom_components.assist_canonicalizer.grammar_loader import (
    build_candidates_from_intent_sources,
)
from custom_components.assist_canonicalizer.indexer import CanonicalIndex, build_index
from custom_components.assist_canonicalizer.runtime import (
    CanonicalizerRuntime,
    _canonical_fingerprint_value,
    normalize_language,
)


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


async def test_async_rebuild_index_coalesces_concurrent_language_jobs(monkeypatch: Any) -> None:
    """Coalesce equivalent language variants into one rebuild job."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()

    runtime = CanonicalizerRuntime()
    calls = 0

    def fake_build_index(snapshot: Any) -> CanonicalIndex:
        """Count rebuild calls and return a small index."""
        nonlocal calls
        calls += 1
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

    async def scenario() -> tuple[CanonicalIndex, CanonicalIndex]:
        """Start overlapping rebuild requests for the same language."""
        hass = FakeHass()
        first_task = asyncio.create_task(runtime.async_rebuild_index(hass, "en"))
        await hass.job_started.wait()
        second_task = asyncio.create_task(runtime.async_rebuild_index(hass, "en-US"))
        await asyncio.sleep(0)
        hass.release_job.set()
        return await asyncio.gather(first_task, second_task)

    monkeypatch.setattr(runtime_module, "_build_index_from_snapshot", fake_build_index)
    first, second = await scenario()

    assert first is second
    assert calls == 1
    assert runtime.get_index("en") is first
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


def test_runtime_precomputes_and_shares_registry_slot_records(monkeypatch: Any) -> None:
    """Normalize identical registry slot tuples once when refreshing the snapshot."""
    normalized_values: list[str] = []
    original_normalize_text = grammar_loader.normalize_text

    def track_normalization(text: str) -> str:
        """Record registry values normalized while building the snapshot."""
        normalized_values.append(text)
        return original_normalize_text(text)

    monkeypatch.setattr(grammar_loader, "normalize_text", track_normalization)
    values_1 = ("Kitchen Light", "Desk Lamp")
    values_2 = ("Kitchen Light", "Desk Lamp")
    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values({"name": values_1, "entity": values_2})

    assert normalized_values == list(values_1)
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

    runtime.update_registry_slot_values({"name": ("old lamp",), "name:light": ("old lamp",)})
    old_ranked = runtime.rank_with_dynamic_candidates("en", index, "turn on old lamp")
    assert old_ranked[0].candidate.normalized_text == "turn on old lamp"

    runtime.update_registry_slot_values({"name": ("new lamp",), "name:light": ("new lamp",)})
    new_ranked = runtime.rank_with_dynamic_candidates("en", index, "turn on new lamp")

    assert new_ranked[0].candidate.normalized_text == "turn on new lamp"
    assert all(record.text != "old lamp" for record in runtime.registry_slot_index["name"])


def test_runtime_intent_update_invalidates_compiled_dynamic_templates() -> None:
    """Discard compiled templates when subscribed intent sources change."""
    runtime = CanonicalizerRuntime()
    runtime.dynamic_registry_intents["en"] = ()

    runtime.update_intent_sources({})

    assert runtime.dynamic_registry_intents == {}


def test_runtime_normalizes_language_cache_keys() -> None:
    """Store and retrieve indexes with canonical language keys."""
    runtime = CanonicalizerRuntime()
    runtime.set_index(build_index("Vi", [Candidate(text="bật đèn", intent_name="HassTurnOn")]))

    assert runtime.get_index("vi") is not None
    assert runtime.get_index("VI") is runtime.get_index("vi")
    assert runtime.get_index("vi-VN") is runtime.get_index("vi")
    assert sorted(runtime.indexes) == ["vi"]


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
        if asyncio.iscoroutinefunction(target):
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
        )
    ]
    index = build_index("vi", candidates)

    MockStore.reset()
    try:
        snapshot = runtime._create_build_snapshot("vi")
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
        assert clean_runtime.get_index("vi") is loaded_index
    finally:
        MockStore.reset()


async def test_persistent_store_rejects_stale_fingerprint(monkeypatch: Any) -> None:
    """Reject and remove a persisted index after registry inputs change."""
    monkeypatch.setattr(homeassistant.helpers.storage, "Store", MockStore)
    MockStore.reset()
    hass = HashableFakeHass()
    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values({"name": ("old lamp",)})
    snapshot = runtime._create_build_snapshot("en")
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
    snapshot = runtime._create_build_snapshot("en")
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
        snapshot = runtime._create_build_snapshot(language)
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
    build_calls = 0

    def build_and_change_source(snapshot: Any) -> CanonicalIndex:
        """Change registry inputs during the first index build."""
        nonlocal build_calls
        build_calls += 1
        name = snapshot.registry_slot_values["name"][0]
        if build_calls == 1:
            runtime.update_registry_slot_values({"name": ("new lamp",)})
        return build_index(
            snapshot.language,
            [Candidate(text=f"turn on {name}", intent_name="HassTurnOn")],
        )

    monkeypatch.setattr(runtime_module, "_build_index_from_snapshot", build_and_change_source)

    index = await runtime.async_rebuild_index(hass, "en")

    assert build_calls == 2
    assert index.candidates[0].text == "turn on new lamp"
    assert runtime.get_index("en") is index
    stored = MockStore.stored_data["assist_canonicalizer.index_en"]
    assert stored["candidates"][0]["text"] == "turn on new lamp"


async def test_debounced_rebuild_coalesces_events(monkeypatch: Any) -> None:
    """Verify that multiple registry events are debounced and only rebuild once."""
    scheduled_callbacks = []

    class FakeTimer:
        """Fake timer mock."""

        def __init__(self, callback: Any) -> None:
            """Initialize fake timer."""
            self.callback = callback

        def __call__(self) -> None:
            """Fire timer."""
            scheduled_callbacks.remove(self)
            self.callback(None)

    def mock_async_call_later(hass: Any, delay: float, action: Any) -> Any:
        """Mock scheduling timer."""
        timer = FakeTimer(action)
        scheduled_callbacks.append(timer)
        return lambda: scheduled_callbacks.remove(timer)

    monkeypatch.setattr(homeassistant.helpers.event, "async_call_later", mock_async_call_later)

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

    hass = HashableFakeHass(
        bus=FakeBus(),
        async_create_task=lambda coro: asyncio.create_task(coro),
        async_add_executor_job=lambda func, *args: func(*args),
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
    original_rebuild = CanonicalizerRuntime.async_rebuild_index

    async def mock_rebuild(self: Any, h: Any, language: str) -> Any:
        """Count rebuild calls."""
        nonlocal rebuild_calls
        rebuild_calls += 1
        return await original_rebuild(self, h, language)

    monkeypatch.setattr(CanonicalizerRuntime, "async_rebuild_index", mock_rebuild)

    _subscribe_registry_updates(hass, runtime)

    # Trigger entity registry update
    listeners[0]({})
    assert len(scheduled_callbacks) == 1

    # Trigger another entity registry update, should debounce
    listeners[0]({})
    assert len(scheduled_callbacks) == 1

    # Fire debounced function
    scheduled_callbacks[0]()

    # Allow task to progress
    await asyncio.sleep(0.01)

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
