"""Tests for Assist Canonicalizer services."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry as HassMockConfigEntry
from voluptuous import Invalid

from custom_components.assist_canonicalizer.candidate import Candidate
from custom_components.assist_canonicalizer.const import (
    ATTR_AGENT_ID,
    CONF_FALLBACK_AGENT_ID,
    CONF_MIN_CONFIDENCE,
    CONF_MIN_MARGIN,
    DATA_RUNTIME,
    DOMAIN,
    SERVICE_CLEAR_INDEX,
    SERVICE_DIAGNOSTICS,
    SERVICE_DUMP_CANDIDATES,
    SERVICE_REBUILD_INDEX,
    SERVICE_SET_FALLBACK_AGENT,
    SERVICE_TEST_MATCH,
)
from custom_components.assist_canonicalizer.indexer import build_index
from custom_components.assist_canonicalizer.ranking import RankedCandidate, ScoreBreakdown
from custom_components.assist_canonicalizer.runtime import CanonicalizerRuntime, IndexClearResult
from custom_components.assist_canonicalizer.services import (
    CLEAR_INDEX_SCHEMA,
    DIAGNOSTICS_SCHEMA,
    DUMP_CANDIDATES_SCHEMA,
    REBUILD_INDEX_SCHEMA,
    SET_FALLBACK_AGENT_SCHEMA,
    TEST_MATCH_SCHEMA,
    _handle_clear_index,
    _handle_diagnostics,
    _handle_dump_candidates,
    _handle_rebuild_index,
    _handle_set_fallback_agent,
    _handle_test_match,
    _runtime_from_hass,
    _service_language,
    async_setup_services,
    async_unload_services,
    validate_supported_language,
)
from custom_components.assist_canonicalizer.utils import wildcard_slot_names


def _as_hass(value: object) -> HomeAssistant:
    """Type a deliberately minimal service test double as Home Assistant."""
    return cast(HomeAssistant, value)


_EXPECTED_SERVICE_SCHEMAS = {
    SERVICE_SET_FALLBACK_AGENT: SET_FALLBACK_AGENT_SCHEMA,
    SERVICE_TEST_MATCH: TEST_MATCH_SCHEMA,
    SERVICE_REBUILD_INDEX: REBUILD_INDEX_SCHEMA,
    SERVICE_CLEAR_INDEX: CLEAR_INDEX_SCHEMA,
    SERVICE_DIAGNOSTICS: DIAGNOSTICS_SCHEMA,
    SERVICE_DUMP_CANDIDATES: DUMP_CANDIDATES_SCHEMA,
}
_EXPECTED_SERVICE_RESPONSE_SUPPORT = {
    SERVICE_SET_FALLBACK_AGENT: SupportsResponse.OPTIONAL,
    SERVICE_TEST_MATCH: SupportsResponse.ONLY,
    SERVICE_REBUILD_INDEX: SupportsResponse.ONLY,
    SERVICE_CLEAR_INDEX: SupportsResponse.ONLY,
    SERVICE_DIAGNOSTICS: SupportsResponse.ONLY,
    SERVICE_DUMP_CANDIDATES: SupportsResponse.ONLY,
}


class MockConfig:
    """Mock Home Assistant config."""

    def __init__(self) -> None:
        """Initialize language."""
        self.language = "vi"


class MockHass:
    """Mock Home Assistant instance for service tests."""

    def __init__(self, runtime: CanonicalizerRuntime, entry: Any = None) -> None:
        """Initialize data registry."""
        self.data = {
            DOMAIN: {
                "mock_entry_id": {
                    DATA_RUNTIME: runtime,
                    "entry": entry,
                }
            }
        }
        self.config = MockConfig()
        self.config_entries = MagicMock()

    async def async_add_executor_job(self, func: Any, *args: Any) -> Any:
        """Execute the job immediately."""
        return func(*args)


class MockServiceCall:
    """Mock service call data payload."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize call data dictionary."""
        self.data = data


class MockConfigEntry:
    """Mock config entry with dynamic options and data dictionary."""

    def __init__(self, options: dict[str, Any], data: dict[str, Any]) -> None:
        """Initialize options and data properties."""
        self.entry_id = "mock_entry_id"
        self.options = options
        self.data = data


class _ServiceRegistrationRecorder:
    """Record service callbacks registered by async_setup_services."""

    def __init__(self) -> None:
        """Initialize service callback storage."""
        self.registered_services: dict[str, Any] = {}
        self.registration_kwargs: dict[str, dict[str, Any]] = {}

    def __call__(
        self,
        domain: str,
        service: str,
        callback: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Mock service registration callback."""
        assert domain == DOMAIN, f"Expected domain {DOMAIN}, got {domain}"
        assert kwargs.get("schema") is _EXPECTED_SERVICE_SCHEMAS[service]
        assert kwargs.get("supports_response") is _EXPECTED_SERVICE_RESPONSE_SUPPORT[service]
        self.registered_services[service] = callback
        self.registration_kwargs[service] = dict(kwargs)


def test_service_language_normalizes_cache_keys() -> None:
    """Verify service language normalization prevents duplicate cache keys."""
    runtime = CanonicalizerRuntime()
    hass = MockHass(runtime)
    call = MockServiceCall({"text": "bật đèn", "language": "Vi"})

    assert _service_language(_as_hass(hass), cast(ServiceCall, call)) == "vi"


@pytest.mark.asyncio
async def test_set_fallback_agent_service_reports_config_entry_change(
    hass: HomeAssistant,
) -> None:
    """Report the real Home Assistant config-entry update result."""
    runtime = CanonicalizerRuntime()
    entry = HassMockConfigEntry(
        domain=DOMAIN,
        options={CONF_MIN_CONFIDENCE: 0.7},
        data={CONF_FALLBACK_AGENT_ID: "old_agent"},
    )
    entry.add_to_hass(hass)
    hass.data[DOMAIN] = {
        entry.entry_id: {
            DATA_RUNTIME: runtime,
            "entry": entry,
        }
    }
    call = MockServiceCall({ATTR_AGENT_ID: "new_agent"})

    with patch(
        "custom_components.assist_canonicalizer.services.async_get_agent",
        return_value=MagicMock(unique_id="new-agent"),
    ):
        result = await _handle_set_fallback_agent(_as_hass(hass), cast(ServiceCall, call))
        unchanged_result = await _handle_set_fallback_agent(_as_hass(hass), cast(ServiceCall, call))

    assert result == {
        CONF_FALLBACK_AGENT_ID: "new_agent",
        "previous_fallback_agent_id": "old_agent",
        "changed": True,
    }
    assert dict(entry.options) == {
        CONF_MIN_CONFIDENCE: 0.7,
        CONF_FALLBACK_AGENT_ID: "new_agent",
    }
    assert unchanged_result == {
        CONF_FALLBACK_AGENT_ID: "new_agent",
        "previous_fallback_agent_id": "new_agent",
        "changed": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_id", "agent", "error"),
    [
        ("mock_entry_id", MagicMock(), "cannot use itself"),
        ("missing_agent", None, "is not available"),
        (
            "conversation.assist_canonicalizer",
            MagicMock(unique_id="mock_entry_id-conversation"),
            "cannot use itself",
        ),
    ],
)
async def test_set_fallback_agent_service_rejects_invalid_targets(
    agent_id: str,
    agent: Any,
    error: str,
) -> None:
    """Reject missing agents and both canonicalizer agent identifiers."""
    runtime = CanonicalizerRuntime()
    entry = MockConfigEntry(options={}, data={})
    hass = MockHass(runtime, entry)
    hass.config_entries = MagicMock()

    with (
        patch(
            "custom_components.assist_canonicalizer.services.async_get_agent",
            return_value=agent,
        ),
        pytest.raises(HomeAssistantError, match=error),
    ):
        await _handle_set_fallback_agent(
            _as_hass(hass),
            cast(ServiceCall, MockServiceCall({ATTR_AGENT_ID: agent_id})),
        )

    hass.config_entries.async_update_entry.assert_not_called()


def test_set_fallback_agent_schema_trims_and_rejects_empty_ids() -> None:
    """Normalize hand-authored action data before resolving an agent."""
    assert SET_FALLBACK_AGENT_SCHEMA({ATTR_AGENT_ID: "  agent-id  "}) == {ATTR_AGENT_ID: "agent-id"}
    with pytest.raises(Invalid):
        SET_FALLBACK_AGENT_SCHEMA({ATTR_AGENT_ID: "  "})


@pytest.mark.asyncio
async def test_handle_test_match_index_none() -> None:
    """Verify test_match raises HomeAssistantError when index cannot be built."""
    runtime = CanonicalizerRuntime()
    hass = MockHass(runtime)
    call = MockServiceCall({"text": "bật đèn", "language": "vi"})

    with (
        patch(
            "custom_components.assist_canonicalizer.services._index_for_language",
            AsyncMock(return_value=None),
        ),
        pytest.raises(HomeAssistantError, match="Assist Canonicalizer index could not be built"),
    ):
        await _handle_test_match(_as_hass(hass), cast(ServiceCall, call))


@pytest.mark.asyncio
async def test_handle_test_match_entry_none_and_data_fallback() -> None:
    """Verify test_match handles entry being None, or options falling back to data configuration."""
    runtime = CanonicalizerRuntime()
    candidates = [Candidate(text="tắt đèn bếp", intent_name="HassTurnOff", language="vi")]
    runtime.set_index(build_index("vi", candidates))
    call = MockServiceCall({"text": "tắt đèn bếp", "language": "vi"})

    # Case 1: Entry is None in hass.data
    hass_none = MockHass(runtime, entry=None)
    hass_none.data[DOMAIN]["mock_entry_id"]["entry"] = None
    result = await _handle_test_match(_as_hass(hass_none), cast(ServiceCall, call))
    assert result["accepted"] is True
    assert result["evaluation"] == {
        "scope": "lexical",
        "candidate_metadata_authoritative": False,
        "live_recognition": "not_run",
        "production_decision_path": "/api/conversation/process",
    }
    assert result["confidence_gate"]["accepted"] is True
    assert result["confidence_gate"]["margin_policy"] == "no_competitor"
    selected_candidate = result["selected_candidate"]
    assert selected_candidate is not None
    assert "score" not in selected_candidate
    assert selected_candidate["scores"]["final"] == 1.0

    # Case 2: entry is not None, options is empty, fallback to data
    entry_data = MockConfigEntry(
        options={}, data={CONF_MIN_CONFIDENCE: 0.10, CONF_MIN_MARGIN: 0.01}
    )
    hass_data = MockHass(runtime, entry=entry_data)
    result_data = await _handle_test_match(_as_hass(hass_data), cast(ServiceCall, call))
    assert result_data["accepted"] is True


@pytest.mark.asyncio
async def test_handle_test_match_rehydrates_from_candidate_metadata_with_cold_cache() -> None:
    """Rehydrate service results without loading intents on the event loop."""
    runtime = CanonicalizerRuntime()
    candidate = Candidate(
        text="add shopping_list_item to shopping list",
        intent_name="HassShoppingListAddItem",
        language="en",
        metadata={
            "sentence_template": "add {shopping_list_item} to shopping list",
            "wildcard_slots": "shopping_list_item",
        },
    )
    runtime.set_index(build_index("en", [candidate]))
    ranked = RankedCandidate(
        candidate=candidate,
        scores=ScoreBreakdown(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    hass = MockHass(runtime)
    call = MockServiceCall({"text": "add milk to shopping list", "language": "en"})

    wildcard_slot_names.cache_clear("en")
    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(ranked,),
        ),
        patch("home_assistant_intents.get_intents") as get_intents,
    ):
        result = await _handle_test_match(_as_hass(hass), cast(ServiceCall, call))

    selected_candidate = result["selected_candidate"]
    assert selected_candidate is not None
    assert selected_candidate["text"] == "add milk to shopping list"
    assert selected_candidate["wildcard_replacements"] == {"shopping_list_item": "milk"}
    assert result["top_candidates"][0]["text"] == "add milk to shopping list"
    get_intents.assert_not_called()
    wildcard_slot_names.cache_clear("en")


@pytest.mark.asyncio
async def test_rebuild_index_service() -> None:
    """Test rebuild_index service handler."""
    runtime = CanonicalizerRuntime()
    hass = MockHass(runtime)
    call = MockServiceCall({"language": "vi"})

    with patch.object(
        CanonicalizerRuntime, "async_rebuild_index", AsyncMock(return_value=build_index("vi", []))
    ):
        result = await _handle_rebuild_index(_as_hass(hass), cast(ServiceCall, call))
        assert result["language"] == "vi"
        assert result["candidate_count"] == 0
        assert result["rebuild_latency_ms"] >= 0
        assert "index_cached" not in result


@pytest.mark.asyncio
async def test_clear_index_service() -> None:
    """Test clear_index service handler."""
    runtime = CanonicalizerRuntime()
    runtime.set_index(
        build_index("en", [Candidate(text="turn on", intent_name="HassTurnOn", language="en")])
    )
    runtime.set_index(
        build_index(
            "vi",
            [
                Candidate(text="bật đèn", intent_name="HassTurnOn", language="vi"),
                Candidate(text="tắt đèn", intent_name="HassTurnOff", language="vi"),
            ],
        )
    )
    hass = MockHass(runtime)

    def clear_index(_hass: Any, language: str | None = None) -> IndexClearResult:
        """Return the cache state captured by the mocked atomic clear."""
        cached_candidate_counts = {
            cached_language: index.candidate_count
            for cached_language, index in runtime.indexes.items()
        }
        cleared_cached_languages = (
            tuple(sorted(cached_candidate_counts))
            if language is None
            else ((language,) if language in cached_candidate_counts else ())
        )
        cleared_candidate_count = sum(
            cached_candidate_counts[cached_language] for cached_language in cleared_cached_languages
        )
        runtime.clear_index(language)
        return IndexClearResult(
            cleared_cached_languages=cleared_cached_languages,
            cleared_candidate_count=cleared_candidate_count,
            remaining_candidate_count=runtime.total_candidate_count(),
            remaining_cached_languages=tuple(sorted(runtime.indexes)),
        )

    with patch.object(
        CanonicalizerRuntime,
        "async_clear_index",
        AsyncMock(side_effect=clear_index),
    ) as mock_clear:
        call = MockServiceCall({"language": "en-US"})
        result = await _handle_clear_index(_as_hass(hass), cast(ServiceCall, call))
        assert result == {
            "language": "en",
            "scope": "language",
            "cleared_cached_languages": ["en"],
            "cleared_candidate_count": 1,
            "remaining_candidate_count": 2,
            "remaining_cached_languages": ["vi"],
        }
        assert "en" not in runtime.indexes
        mock_clear.assert_awaited_once_with(hass, "en")

        runtime.set_index(
            build_index(
                "en",
                [Candidate(text="turn off", intent_name="HassTurnOff", language="en")],
            )
        )
        call_all = MockServiceCall({})
        result_all = await _handle_clear_index(_as_hass(hass), cast(ServiceCall, call_all))
        assert result_all == {
            "language": None,
            "scope": "all",
            "cleared_cached_languages": ["en", "vi"],
            "cleared_candidate_count": 3,
            "remaining_candidate_count": 0,
            "remaining_cached_languages": [],
        }
        mock_clear.assert_awaited_with(hass, None)


@pytest.mark.asyncio
async def test_clear_index_service_uses_atomic_runtime_result() -> None:
    """Build the response from the clear snapshot instead of mutable runtime state."""
    runtime = CanonicalizerRuntime()
    runtime.set_index(
        build_index("en", [Candidate(text="turn on", intent_name="HassTurnOn", language="en")])
    )
    hass = MockHass(runtime)
    clear_result = IndexClearResult(
        cleared_cached_languages=("en",),
        cleared_candidate_count=1,
        remaining_candidate_count=0,
        remaining_cached_languages=(),
    )

    with patch.object(
        CanonicalizerRuntime,
        "async_clear_index",
        AsyncMock(return_value=clear_result),
    ):
        result = await _handle_clear_index(
            _as_hass(hass),
            cast(ServiceCall, MockServiceCall({"language": "en"})),
        )

    assert result["cleared_cached_languages"] == ["en"]
    assert result["cleared_candidate_count"] == 1
    assert result["remaining_candidate_count"] == 0
    assert result["remaining_cached_languages"] == []


@pytest.mark.asyncio
async def test_diagnostics_service() -> None:
    """Test diagnostics service handler."""
    runtime = CanonicalizerRuntime()
    runtime.set_index(
        build_index("en", [Candidate(text="turn on", intent_name="HassTurnOn", language="en")])
    )
    runtime.set_index(
        build_index(
            "vi",
            [
                Candidate(text="bật đèn", intent_name="HassTurnOn", language="vi"),
                Candidate(text="tắt đèn", intent_name="HassTurnOff", language="vi"),
            ],
        )
    )
    hass = MockHass(runtime)
    call = MockServiceCall({})

    result = await _handle_diagnostics(_as_hass(hass), cast(ServiceCall, call))
    assert result["total_cached_candidate_count"] == 3
    assert result["cached_indexes"] == {
        "en": {"candidate_count": 1, "version": 1},
        "vi": {"candidate_count": 2, "version": 1},
    }
    assert "candidate_count" not in result
    assert "index_version" not in result
    assert "cached_languages" not in result
    assert "cached_candidate_counts" not in result
    assert "dynamic_candidate_generation" in result
    assert "registry_retrieval" in result
    registry_retrieval = cast(dict[str, Any], result["registry_retrieval"])
    assert registry_retrieval["values_scored"] == 0


@pytest.mark.asyncio
async def test_dump_candidates_service() -> None:
    """Test dump_candidates service handler for cached and uncached index cases."""
    runtime = CanonicalizerRuntime()
    hass = MockHass(runtime)

    # 1. Uncached index, no rebuild
    call_no_rebuild = MockServiceCall({"language": "vi", "rebuild": False})
    result = await _handle_dump_candidates(_as_hass(hass), cast(ServiceCall, call_no_rebuild))
    result_dict = cast(dict[str, Any], result)
    intent_source_counts = result_dict.pop("intent_source_counts")
    assert intent_source_counts.get("built_in", 0) > 0
    assert result_dict == {
        "language": "vi",
        "candidate_count": 0,
        "index_status": "missing",
        "rebuild_latency_ms": None,
        "candidate_source_counts": {},
        "intent_counts": {},
        "registry_slot_counts": {},
        "candidate_sample": {"truncated": False, "candidates": []},
    }

    # 2. Rebuild is forced even when a stale index is already cached.
    runtime.set_index(build_index("vi", [Candidate(text="cũ", intent_name="Old", language="vi")]))
    call_rebuild = MockServiceCall({"language": "vi", "rebuild": True})
    rebuilt_candidate = Candidate(
        text="bật shopping_list_item",
        intent_name="On",
        language="vi",
        metadata={
            "sentence_template": "bật {shopping_list_item}",
            "wildcard_slots": "shopping_list_item",
            "slots": '{"domain":"light"}',
        },
    )
    with patch.object(
        CanonicalizerRuntime,
        "async_rebuild_index",
        AsyncMock(return_value=build_index("vi", [rebuilt_candidate])),
    ) as mock_rebuild:
        result = await _handle_dump_candidates(_as_hass(hass), cast(ServiceCall, call_rebuild))
        mock_rebuild.assert_awaited_once()
        assert result["index_status"] == "rebuilt"
        rebuild_latency_ms = result["rebuild_latency_ms"]
        assert rebuild_latency_ms is not None
        assert rebuild_latency_ms >= 0
        assert result["candidate_count"] == 1
        assert result["candidate_sample"] == {
            "truncated": False,
            "candidates": [
                {
                    "text": "bật shopping_list_item",
                    "intent_name": "On",
                    "source": "generated_sample",
                    "normalized_text": "bật shopping_list_item",
                    "slots": {"domain": "light"},
                    "wildcard_slots": ["shopping_list_item"],
                    "sentence_template": "bật {shopping_list_item}",
                }
            ],
        }


@pytest.mark.asyncio
async def test_dump_candidates_reports_sample_truncation() -> None:
    """Report when the bounded candidate sample omits indexed candidates."""
    runtime = CanonicalizerRuntime()
    runtime.set_index(
        build_index(
            "en",
            [
                Candidate(text=f"sample {index}", intent_name="Sample", language="en")
                for index in range(51)
            ],
        )
    )
    hass = MockHass(runtime)

    result = await _handle_dump_candidates(
        _as_hass(hass),
        cast(ServiceCall, MockServiceCall({"language": "en"})),
    )

    assert result["index_status"] == "cached"
    assert result["candidate_count"] == 51
    assert result["candidate_sample"]["truncated"] is True
    assert len(result["candidate_sample"]["candidates"]) == 50


def test_runtime_from_hass_error() -> None:
    """Test _runtime_from_hass raising error when component is not loaded."""
    hass = MagicMock()
    hass.data = {}
    with pytest.raises(HomeAssistantError, match="Assist Canonicalizer is not loaded"):
        _runtime_from_hass(hass)


def test_async_setup_and_unload_services() -> None:
    """Test service registration and removal."""
    hass = MagicMock()

    async_setup_services(hass)
    assert hass.services.async_register.call_count == 6

    async_unload_services(hass)
    assert hass.services.async_remove.call_count == 6


def test_runtime_from_hass_skips_invalid_runtime() -> None:
    """Verify _runtime_from_hass skips non-runtime instances."""
    runtime = CanonicalizerRuntime()
    hass = MagicMock()
    hass.data = {
        DOMAIN: {
            "invalid_entry": {DATA_RUNTIME: "not_a_runtime_object"},
            "valid_entry": {DATA_RUNTIME: runtime},
        }
    }
    assert _runtime_from_hass(hass) is runtime


@pytest.mark.asyncio
async def test_async_services_dispatch() -> None:
    """Test that registered service callbacks dispatch successfully to handlers."""
    hass = MagicMock()
    recorder = _ServiceRegistrationRecorder()
    hass.services.async_register = recorder

    # Mock corresponding handlers called inside callbacks
    with (
        patch(
            "custom_components.assist_canonicalizer.services._handle_set_fallback_agent",
            AsyncMock(return_value={"status": "updated"}),
        ) as mock_set_fallback,
        patch(
            "custom_components.assist_canonicalizer.services._handle_test_match",
            AsyncMock(return_value={"status": "tested"}),
        ) as mock_test,
        patch(
            "custom_components.assist_canonicalizer.services._handle_rebuild_index",
            AsyncMock(return_value={"status": "rebuilt"}),
        ) as mock_rebuild,
        patch(
            "custom_components.assist_canonicalizer.services._handle_dump_candidates",
            AsyncMock(return_value={"status": "dumped"}),
        ) as mock_dump,
        patch(
            "custom_components.assist_canonicalizer.services._handle_clear_index",
            AsyncMock(return_value={"status": "cleared"}),
        ) as mock_clear,
        patch(
            "custom_components.assist_canonicalizer.services._handle_diagnostics",
            AsyncMock(return_value={"status": "diagnosed"}),
        ) as mock_diagnostics,
    ):
        async_setup_services(hass)
        assert len(recorder.registered_services) == 6

        call = MockServiceCall({})

        res_set_fallback = await recorder.registered_services[SERVICE_SET_FALLBACK_AGENT](call)
        assert res_set_fallback == {"status": "updated"}
        mock_set_fallback.assert_called_once_with(hass, call)

        # Test handle_test_match
        res_test = await recorder.registered_services[SERVICE_TEST_MATCH](call)
        assert res_test == {"status": "tested"}
        mock_test.assert_called_once_with(hass, call)

        # Test handle_rebuild_index
        res_rebuild = await recorder.registered_services[SERVICE_REBUILD_INDEX](call)
        assert res_rebuild == {"status": "rebuilt"}
        mock_rebuild.assert_called_once_with(hass, call)

        res_clear = await recorder.registered_services[SERVICE_CLEAR_INDEX](call)
        assert res_clear == {"status": "cleared"}
        mock_clear.assert_called_once_with(hass, call)

        # Test handle_dump_candidates
        res_dump = await recorder.registered_services[SERVICE_DUMP_CANDIDATES](call)
        assert res_dump == {"status": "dumped"}
        mock_dump.assert_called_once_with(hass, call)

        # Test handle_diagnostics
        res_diagnostics = await recorder.registered_services[SERVICE_DIAGNOSTICS](call)
        assert res_diagnostics == {"status": "diagnosed"}
        mock_diagnostics.assert_called_once_with(hass, call)


def test_validate_supported_language() -> None:
    """Test that dynamic language validator correctly rejects or accepts inputs."""
    assert validate_supported_language("vi") == "vi"
    assert validate_supported_language("en") == "en"
    assert validate_supported_language("VI") == "vi"

    with pytest.raises(Invalid, match="is not supported by Home Assistant"):
        validate_supported_language("invalid_lang_code")

    with pytest.raises(Invalid, match="Language cannot be empty"):
        validate_supported_language("")


@pytest.mark.asyncio
async def test_services_exception_wrapping() -> None:
    """Verify service handlers wrap general exceptions in HomeAssistantError."""
    runtime = CanonicalizerRuntime()
    hass = MockHass(runtime)
    call = MockServiceCall({"language": "vi", "text": "test"})

    # 1. _handle_test_match raising error during rebuild or matching
    with (
        patch(
            "custom_components.assist_canonicalizer.services._index_for_language",
            AsyncMock(side_effect=ValueError("Test parsing error")),
        ),
        pytest.raises(HomeAssistantError, match="Matching test failed; see logs for details"),
    ):
        await _handle_test_match(_as_hass(hass), cast(ServiceCall, call))

    # 2. _handle_set_fallback_agent raising error while persisting the option
    entry = MockConfigEntry(options={}, data={})
    hass.data[DOMAIN]["mock_entry_id"]["entry"] = entry
    hass.config_entries = MagicMock()
    hass.config_entries.async_update_entry.side_effect = RuntimeError("Test update error")
    with (
        patch(
            "custom_components.assist_canonicalizer.services.async_get_agent",
            return_value=MagicMock(unique_id="other-agent"),
        ),
        pytest.raises(
            HomeAssistantError,
            match="Fallback agent update failed; see logs for details",
        ),
    ):
        await _handle_set_fallback_agent(
            _as_hass(hass),
            cast(ServiceCall, MockServiceCall({ATTR_AGENT_ID: "other_agent"})),
        )

    # 3. _handle_rebuild_index raising error
    with (
        patch(
            "custom_components.assist_canonicalizer.services._rebuild_index",
            AsyncMock(side_effect=KeyError("Test key error")),
        ),
        pytest.raises(HomeAssistantError, match="Index rebuild failed; see logs for details"),
    ):
        await _handle_rebuild_index(_as_hass(hass), cast(ServiceCall, call))

    # 4. _handle_clear_index raising error
    with (
        patch.object(
            CanonicalizerRuntime,
            "async_clear_index",
            AsyncMock(side_effect=ValueError("Test clear error")),
        ),
        pytest.raises(HomeAssistantError, match="Clear index failed; see logs for details"),
    ):
        await _handle_clear_index(_as_hass(hass), cast(ServiceCall, call))

    # 5. _handle_dump_candidates raising error
    with (
        patch.object(
            CanonicalizerRuntime,
            "get_index",
            side_effect=TypeError("Test type error"),
        ),
        pytest.raises(HomeAssistantError, match="Dump candidates failed; see logs for details"),
    ):
        await _handle_dump_candidates(_as_hass(hass), cast(ServiceCall, call))

    # 6. Verify RuntimeError is also caught and wrapped (it is a subclass of Exception)
    with (
        patch(
            "custom_components.assist_canonicalizer.services._index_for_language",
            AsyncMock(side_effect=RuntimeError("Unexpected system failure")),
        ),
        pytest.raises(HomeAssistantError, match="Matching test failed; see logs for details"),
    ):
        await _handle_test_match(_as_hass(hass), cast(ServiceCall, call))
