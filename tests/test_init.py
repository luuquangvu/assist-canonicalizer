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
