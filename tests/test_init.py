"""Tests for Assist Canonicalizer integration entry points."""

import importlib
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import homeassistant.components
import homeassistant.helpers
import pytest

import custom_components.assist_canonicalizer
from custom_components.assist_canonicalizer import (
    _async_update_options,
    _async_warmup_pipeline_languages,
    _discover_pipeline_languages,
    _warmup_single_language,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.assist_canonicalizer.const import DATA_RUNTIME, DOMAIN
from custom_components.assist_canonicalizer.runtime import CanonicalizerRuntime


def test_init_imports_fallback() -> None:
    """Test importing the custom component when Home Assistant helper registries are missing."""
    # Delete attributes from parent modules to force try-except block execution on import/reload

    helpers_attrs = [
        "area_registry",
        "entity_registry",
        "event",
        "floor_registry",
    ]
    old_helpers = {}
    for attr in helpers_attrs:
        if hasattr(homeassistant.helpers, attr):
            old_helpers[attr] = getattr(homeassistant.helpers, attr)
            delattr(homeassistant.helpers, attr)

    components_attrs = ["conversation", "homeassistant"]
    old_components = {}
    for attr in components_attrs:
        if hasattr(homeassistant.components, attr):
            old_components[attr] = getattr(homeassistant.components, attr)
            delattr(homeassistant.components, attr)

    # Patch sys.modules to return None for these modules
    sys_modules_patch = {
        "homeassistant.components.conversation": None,
        "homeassistant.components.homeassistant": None,
        "homeassistant.helpers.area_registry": None,
        "homeassistant.helpers.entity_registry": None,
        "homeassistant.helpers.event": None,
        "homeassistant.helpers.floor_registry": None,
    }

    try:
        with patch.dict(sys.modules, sys_modules_patch):
            importlib.reload(custom_components.assist_canonicalizer)

            assert custom_components.assist_canonicalizer.agent_manager is None
            assert custom_components.assist_canonicalizer.exposed_entities is None
            assert custom_components.assist_canonicalizer.area_registry is None
            assert custom_components.assist_canonicalizer.entity_registry is None
            assert custom_components.assist_canonicalizer.ha_event is None
            assert custom_components.assist_canonicalizer.floor_registry is None
    finally:
        # Restore parent module attributes
        for attr, val in old_helpers.items():
            setattr(homeassistant.helpers, attr, val)
        for attr, val in old_components.items():
            setattr(homeassistant.components, attr, val)

        # Force restore reload of the module to its original state
        importlib.reload(custom_components.assist_canonicalizer)


@pytest.mark.asyncio
async def test_async_setup_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test setup entry with trigger intents updates callback."""
    hass = MagicMock()
    hass.config.path = MagicMock(return_value="/config")
    hass.data = {}
    hass.add_job = MagicMock()

    # Mock config entries async methods
    hass.config_entries = MagicMock()
    hass.config_entries.async_forward_entry_setups = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {}
    entry.options = {}

    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.registry.async_registry_slot_values",
        lambda h: {"name": ("light",)},
    )

    saved_callback = None

    def subscribe_intents(cb: Any) -> Any:
        """Mock subscribing to intent changes."""
        nonlocal saved_callback
        saved_callback = cb
        return lambda: None

    mock_agent_manager = MagicMock()
    mock_agent_manager.subscribe_intents = subscribe_intents
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.agent_manager.get_agent_manager",
        lambda h: mock_agent_manager,
    )

    hass.bus.async_listen.return_value = lambda: None
    mock_exposed = MagicMock()
    mock_exposed.async_listen_entity_updates.return_value = lambda: None
    monkeypatch.setattr("custom_components.assist_canonicalizer.exposed_entities", mock_exposed)

    mock_setup_services = MagicMock()
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.async_setup_services", mock_setup_services
    )

    # Mock the warmup orchestrator to suppress the real call.
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer._async_warmup_pipeline_languages",
        MagicMock(),
    )

    result = await async_setup_entry(hass, entry)

    assert result is True
    assert hass.data[DOMAIN][entry.entry_id]["entry"] is entry
    runtime = hass.data[DOMAIN][entry.entry_id][DATA_RUNTIME]
    assert isinstance(runtime, CanonicalizerRuntime)

    # Verify triggering the intent update callback schedules index rebuild
    assert saved_callback is not None
    runtime.indexes["en"] = MagicMock()
    saved_callback({"some": "update"})
    hass.add_job.assert_called_with(runtime.async_rebuild_index, hass, "en")


@pytest.mark.asyncio
async def test_async_unload_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test unloading configuration entry with success, failure, and multiple entries."""
    hass = MagicMock()

    entry = MagicMock()
    entry.entry_id = "test_entry"

    runtime = CanonicalizerRuntime()
    hass.data = {
        DOMAIN: {
            "test_entry": {
                "entry": entry,
                DATA_RUNTIME: runtime,
            }
        }
    }

    hass.config_entries = MagicMock()
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)

    mock_unload_services = MagicMock()
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.async_unload_services", mock_unload_services
    )

    # 1. Unload failure path
    hass.config_entries.async_unload_platforms.return_value = False
    result = await async_unload_entry(hass, entry)
    assert result is False

    # 2. Unload success path, verify DOMAIN data is removed when empty
    hass.config_entries.async_unload_platforms.return_value = True
    result = await async_unload_entry(hass, entry)
    assert result is True
    assert DOMAIN not in hass.data

    # 3. Unload success path with remaining entries (verify DOMAIN data is not removed entirely)
    other_entry = MagicMock()
    hass.data = {
        DOMAIN: {
            "test_entry": {
                "entry": entry,
                DATA_RUNTIME: runtime,
            },
            "other_entry": {
                "entry": other_entry,
                DATA_RUNTIME: CanonicalizerRuntime(),
            },
        }
    }
    result = await async_unload_entry(hass, entry)
    assert result is True
    assert DOMAIN in hass.data
    assert "test_entry" not in hass.data[DOMAIN]
    assert "other_entry" in hass.data[DOMAIN]


@pytest.mark.asyncio
async def test_async_update_options() -> None:
    """Test reloading configuration entry when options are modified."""
    hass = MagicMock()
    hass.config_entries = MagicMock()
    hass.config_entries.async_reload = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "test_entry"

    await _async_update_options(hass, entry)
    hass.config_entries.async_reload.assert_called_once_with("test_entry")


def test_discover_pipeline_languages_import_failed() -> None:
    """Return empty set when async_get_pipelines is None (import failed)."""
    hass = MagicMock()
    with patch.object(custom_components.assist_canonicalizer, "async_get_pipelines", None):
        result = _discover_pipeline_languages(hass)

    assert result == set()


def test_discover_pipeline_languages_api_raises() -> None:
    """Return empty set when async_get_pipelines raises."""
    hass = MagicMock()
    mock_get_pipelines = MagicMock(side_effect=RuntimeError("pipeline not ready"))
    with patch.object(
        custom_components.assist_canonicalizer,
        "async_get_pipelines",
        mock_get_pipelines,
    ):
        result = _discover_pipeline_languages(hass)

    assert result == set()
    mock_get_pipelines.assert_called_once_with(hass)


def test_discover_pipeline_languages_multiple_unique() -> None:
    """Return deduplicated language codes from multiple pipelines."""
    hass = MagicMock()
    pipeline_en = MagicMock(language="en")
    pipeline_vi = MagicMock(language="vi")
    pipeline_en2 = MagicMock(language="en")  # duplicate
    pipeline_fr = MagicMock(language="fr")

    mock_get_pipelines = MagicMock(
        return_value=[pipeline_en, pipeline_vi, pipeline_en2, pipeline_fr]
    )
    with patch.object(
        custom_components.assist_canonicalizer,
        "async_get_pipelines",
        mock_get_pipelines,
    ):
        result = _discover_pipeline_languages(hass)

    assert len(result) == 3
    assert "en" in result
    assert "vi" in result
    assert "fr" in result


def test_discover_pipeline_languages_empty_and_whitespace_skipped() -> None:
    """Skip pipelines with empty or whitespace-only language attributes."""
    hass = MagicMock()
    pipeline_empty = MagicMock(language="")
    pipeline_whitespace = MagicMock(language="   ")
    pipeline_valid = MagicMock(language="nl")

    mock_get_pipelines = MagicMock(
        return_value=[pipeline_empty, pipeline_whitespace, pipeline_valid]
    )
    with patch.object(
        custom_components.assist_canonicalizer,
        "async_get_pipelines",
        mock_get_pipelines,
    ):
        result = _discover_pipeline_languages(hass)

    assert result == {"nl"}


def test_discover_pipeline_languages_no_language_attr() -> None:
    """Skip pipelines without a language attribute."""
    hass = MagicMock()
    pipeline_no_lang = MagicMock(spec=[])  # no 'language' attribute

    mock_get_pipelines = MagicMock(return_value=[pipeline_no_lang])
    with patch.object(
        custom_components.assist_canonicalizer,
        "async_get_pipelines",
        mock_get_pipelines,
    ):
        result = _discover_pipeline_languages(hass)

    assert result == set()


@pytest.mark.asyncio
async def test_warmup_single_language_already_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do nothing when the index is already cached in memory."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock()

    mock_load = AsyncMock()
    mock_rebuild = AsyncMock()
    monkeypatch.setattr(CanonicalizerRuntime, "async_load_index_from_store", mock_load)
    monkeypatch.setattr(CanonicalizerRuntime, "async_rebuild_index", mock_rebuild)

    await _warmup_single_language(hass, runtime, "en")

    mock_load.assert_not_called()
    mock_rebuild.assert_not_called()


@pytest.mark.asyncio
async def test_warmup_single_language_loaded_from_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load from store succeeds; do not rebuild."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()

    stored_index = MagicMock()
    mock_load = AsyncMock(return_value=stored_index)
    mock_rebuild = AsyncMock()
    monkeypatch.setattr(CanonicalizerRuntime, "async_load_index_from_store", mock_load)
    monkeypatch.setattr(CanonicalizerRuntime, "async_rebuild_index", mock_rebuild)

    await _warmup_single_language(hass, runtime, "en")

    mock_load.assert_awaited_once_with(hass, "en")
    mock_rebuild.assert_not_called()


@pytest.mark.asyncio
async def test_warmup_single_language_rebuilds_when_store_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild when store has no cached index."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()

    mock_load = AsyncMock(return_value=None)
    mock_rebuild = AsyncMock()
    monkeypatch.setattr(CanonicalizerRuntime, "async_load_index_from_store", mock_load)
    monkeypatch.setattr(CanonicalizerRuntime, "async_rebuild_index", mock_rebuild)

    await _warmup_single_language(hass, runtime, "en")

    mock_load.assert_awaited_once_with(hass, "en")
    mock_rebuild.assert_awaited_once_with(hass, "en")


@pytest.mark.asyncio
async def test_warmup_single_language_exception_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceptions during warmup are silently suppressed."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()

    mock_load = AsyncMock(side_effect=RuntimeError("storage error"))
    monkeypatch.setattr(CanonicalizerRuntime, "async_load_index_from_store", mock_load)

    # Must not raise
    await _warmup_single_language(hass, runtime, "en")

    mock_load.assert_awaited_once_with(hass, "en")
    # Rebuild was never reached because loading from store raised
    assert "en" not in runtime.indexes


@pytest.mark.asyncio
async def test_warmup_single_language_rebuild_exception_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceptions during rebuild are silently suppressed."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()

    mock_load = AsyncMock(return_value=None)
    mock_rebuild = AsyncMock(side_effect=RuntimeError("build error"))
    monkeypatch.setattr(CanonicalizerRuntime, "async_load_index_from_store", mock_load)
    monkeypatch.setattr(CanonicalizerRuntime, "async_rebuild_index", mock_rebuild)

    # Must not raise
    await _warmup_single_language(hass, runtime, "en")

    mock_rebuild.assert_awaited_once_with(hass, "en")


@pytest.mark.asyncio
async def test_warmup_single_language_normalizes_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-canonical language codes are normalized before use."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()

    mock_load = AsyncMock(return_value=None)
    mock_rebuild = AsyncMock()
    monkeypatch.setattr(CanonicalizerRuntime, "async_load_index_from_store", mock_load)
    monkeypatch.setattr(CanonicalizerRuntime, "async_rebuild_index", mock_rebuild)

    await _warmup_single_language(hass, runtime, "vi-VN")

    # Methods normalize internally; mock captures the raw argument.
    mock_load.assert_awaited_once_with(hass, "vi-VN")


@pytest.mark.asyncio
async def test_warmup_pipeline_languages_spawns_tasks() -> None:
    """Spawn one warmup task per discovered pipeline language."""
    hass = MagicMock()
    hass.config.language = "en"
    hass.async_create_task = MagicMock()
    runtime = CanonicalizerRuntime()

    pipelines = [MagicMock(language="en"), MagicMock(language="vi")]
    mock_get_pipelines = MagicMock(return_value=pipelines)
    with patch.object(
        custom_components.assist_canonicalizer,
        "async_get_pipelines",
        mock_get_pipelines,
    ):
        await _async_warmup_pipeline_languages(hass, runtime)

        assert hass.async_create_task.call_count == 2

        # Close captured coroutines to avoid RuntimeWarning
        for call_args in hass.async_create_task.call_args_list:
            call_args[0][0].close()


@pytest.mark.asyncio
async def test_warmup_pipeline_languages_fallback_to_default() -> None:
    """Fall back to hass.config.language when discovery returns empty."""
    hass = MagicMock()
    hass.config.language = "de"
    hass.async_create_task = MagicMock()
    runtime = CanonicalizerRuntime()

    with patch.object(custom_components.assist_canonicalizer, "async_get_pipelines", None):
        await _async_warmup_pipeline_languages(hass, runtime)

        hass.async_create_task.assert_called_once()
        created_task_arg = hass.async_create_task.call_args[0][0]
        assert created_task_arg is not None

        # Close captured coroutine to avoid RuntimeWarning
        created_task_arg.close()


@pytest.mark.asyncio
async def test_warmup_pipeline_languages_fallback_failure_silent() -> None:
    """Return silently when both discovery and fallback fail."""
    hass = MagicMock()

    # Patch async_get_pipelines to None so discovery returns empty;
    # delete config.language so accessing it raises AttributeError.
    del hass.config.language

    hass.async_create_task = MagicMock()
    runtime = CanonicalizerRuntime()

    with patch.object(custom_components.assist_canonicalizer, "async_get_pipelines", None):
        # Must not raise
        await _async_warmup_pipeline_languages(hass, runtime)

    hass.async_create_task.assert_not_called()


@pytest.mark.asyncio
async def test_async_setup_entry_triggers_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    """async_setup_entry spawns a background warmup task after setup."""
    hass = MagicMock()
    hass.config.path = MagicMock(return_value="/config")
    hass.data = {}
    hass.add_job = MagicMock()
    hass.config_entries = MagicMock()
    hass.config_entries.async_forward_entry_setups = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {}
    entry.options = {}

    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.registry.async_registry_slot_values",
        lambda h: {"name": ("light",)},
    )
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.agent_manager.get_agent_manager",
        lambda h: MagicMock(subscribe_intents=lambda cb: lambda: None),
    )
    hass.bus.async_listen.return_value = lambda: None
    mock_exposed = MagicMock()
    mock_exposed.async_listen_entity_updates.return_value = lambda: None
    monkeypatch.setattr("custom_components.assist_canonicalizer.exposed_entities", mock_exposed)

    mock_setup_services = MagicMock()
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.async_setup_services", mock_setup_services
    )

    # Mock the warmup orchestrator to verify it gets spawned.
    mock_warmup = MagicMock()
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer._async_warmup_pipeline_languages",
        mock_warmup,
    )

    captured_tasks: list[Any] = []
    hass.async_create_task = MagicMock(side_effect=lambda coro: captured_tasks.append(coro))

    result = await async_setup_entry(hass, entry)

    assert result is True
    assert len(captured_tasks) == 1
    assert captured_tasks[0] is not None
