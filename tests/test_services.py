"""Tests for Assist Canonicalizer services."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import ServiceCall
from homeassistant.exceptions import HomeAssistantError
from voluptuous import Invalid

from custom_components.assist_canonicalizer.candidate import Candidate
from custom_components.assist_canonicalizer.const import (
    CONF_MIN_CONFIDENCE,
    CONF_MIN_MARGIN,
    DATA_RUNTIME,
    DOMAIN,
    SERVICE_DUMP_CANDIDATES,
    SERVICE_REBUILD_INDEX,
    SERVICE_TEST_MATCH,
)
from custom_components.assist_canonicalizer.indexer import build_index
from custom_components.assist_canonicalizer.runtime import CanonicalizerRuntime
from custom_components.assist_canonicalizer.services import (
    _handle_clear_index,
    _handle_diagnostics,
    _handle_dump_candidates,
    _handle_rebuild_index,
    _handle_test_match,
    _runtime_from_hass,
    _service_language,
    async_setup_services,
    async_unload_services,
    validate_supported_language,
)


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
        self.options = options
        self.data = data


def test_service_language_normalizes_cache_keys() -> None:
    """Verify service language normalization prevents duplicate cache keys."""
    runtime = CanonicalizerRuntime()
    hass = MockHass(runtime)
    call = MockServiceCall({"text": "bật đèn", "language": "Vi"})

    assert _service_language(hass, cast(ServiceCall, call)) == "vi"


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
        await _handle_test_match(hass, cast(ServiceCall, call))


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
    result = await _handle_test_match(hass_none, cast(ServiceCall, call))
    assert result["accepted"] is True

    # Case 2: entry is not None, options is empty, fallback to data
    entry_data = MockConfigEntry(
        options={}, data={CONF_MIN_CONFIDENCE: 0.10, CONF_MIN_MARGIN: 0.01}
    )
    hass_data = MockHass(runtime, entry=entry_data)
    result_data = await _handle_test_match(hass_data, cast(ServiceCall, call))
    assert result_data["accepted"] is True


@pytest.mark.asyncio
async def test_rebuild_index_service() -> None:
    """Test rebuild_index service handler."""
    runtime = CanonicalizerRuntime()
    hass = MockHass(runtime)
    call = MockServiceCall({"language": "vi"})

    with patch.object(
        CanonicalizerRuntime, "async_rebuild_index", AsyncMock(return_value=build_index("vi", []))
    ):
        result = await _handle_rebuild_index(hass, cast(ServiceCall, call))
        assert result["language"] == "vi"
        assert result["candidate_count"] == 0


def test_clear_index_service() -> None:
    """Test clear_index service handler."""
    runtime = CanonicalizerRuntime()
    runtime.set_index(build_index("en", []))
    hass = MockHass(runtime)

    # 1. Clear specific language
    call = MockServiceCall({"language": "en"})
    result = _handle_clear_index(hass, cast(ServiceCall, call))
    assert result["candidate_count"] == 0
    assert "en" not in runtime.indexes

    # 2. Clear all
    runtime.set_index(build_index("en", []))
    call_all = MockServiceCall({})
    result_all = _handle_clear_index(hass, cast(ServiceCall, call_all))
    assert result_all["candidate_count"] == 0


def test_diagnostics_service() -> None:
    """Test diagnostics service handler."""
    runtime = CanonicalizerRuntime()
    hass = MockHass(runtime)
    call = MockServiceCall({})

    result = _handle_diagnostics(hass, cast(ServiceCall, call))
    assert "cached_languages" in result
    assert "dynamic_candidate_generation" in result


@pytest.mark.asyncio
async def test_dump_candidates_service() -> None:
    """Test dump_candidates service handler for cached and uncached index cases."""
    runtime = CanonicalizerRuntime()
    hass = MockHass(runtime)

    # 1. Uncached index, no rebuild
    call_no_rebuild = MockServiceCall({"language": "vi", "rebuild": False})
    result = await _handle_dump_candidates(hass, cast(ServiceCall, call_no_rebuild))
    assert result["index_cached"] is False
    assert result["candidate_count"] == 0

    # 2. Uncached index with rebuild
    call_rebuild = MockServiceCall({"language": "vi", "rebuild": True})
    with patch.object(
        CanonicalizerRuntime,
        "async_rebuild_index",
        AsyncMock(return_value=build_index("vi", [Candidate(text="bật", intent_name="On")])),
    ):
        result = await _handle_dump_candidates(hass, cast(ServiceCall, call_rebuild))
        assert result["index_cached"] is True
        assert result["candidate_count"] == 1
        assert len(result["sample_candidates"]) == 1
        assert result["sample_candidates"][0]["text"] == "bật"


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
    assert hass.services.async_register.call_count == 5

    async_unload_services(hass)
    assert hass.services.async_remove.call_count == 5


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
    registered_services = {}

    def mock_register(domain: str, service: str, callback: Any, *args: Any, **kwargs: Any) -> None:
        """Mock service registration callback."""
        registered_services[service] = callback

    hass.services.async_register = mock_register

    async_setup_services(hass)
    assert len(registered_services) == 5

    # Mock corresponding handlers called inside callbacks
    with (
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
    ):
        call = MockServiceCall({})

        # Test handle_test_match
        res_test = await registered_services[SERVICE_TEST_MATCH](call)
        assert res_test == {"status": "tested"}
        mock_test.assert_called_once_with(hass, call)

        # Test handle_rebuild_index
        res_rebuild = await registered_services[SERVICE_REBUILD_INDEX](call)
        assert res_rebuild == {"status": "rebuilt"}
        mock_rebuild.assert_called_once_with(hass, call)

        # Test handle_dump_candidates
        res_dump = await registered_services[SERVICE_DUMP_CANDIDATES](call)
        assert res_dump == {"status": "dumped"}
        mock_dump.assert_called_once_with(hass, call)


def test_validate_supported_language() -> None:
    """Test that dynamic language validator correctly rejects or accepts inputs."""
    assert validate_supported_language("vi") == "vi"
    assert validate_supported_language("en") == "en"
    assert validate_supported_language("VI") == "vi"

    with pytest.raises(Invalid, match="is not supported by Home Assistant"):
        validate_supported_language("invalid_lang_code")

    with pytest.raises(Invalid, match="Language cannot be empty"):
        validate_supported_language("")
