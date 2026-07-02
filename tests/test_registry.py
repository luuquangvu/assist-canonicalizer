"""Tests for registry-derived candidate slot values."""

import importlib
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

import custom_components.assist_canonicalizer.registry as registry
from custom_components.assist_canonicalizer.registry import (
    _area_names,
    _entity_names,
    _entry_names_and_aliases,
    _exposed_entity_names_by_domain,
    _floor_names,
    build_registry_slot_values,
    merge_slot_values,
)


def test_build_registry_slot_values_maps_entities_areas_and_floors() -> None:
    """Map HA metadata names to common Hassil slot names."""
    slot_values = build_registry_slot_values(
        entity_names=["Kitchen Light", "kitchen light", "Desk Lamp"],
        entity_names_by_domain={"light": ["Kitchen Light"], "media_player": ["Speaker"]},
        area_names=["Kitchen", "Bếp"],
        floor_names=["Ground Floor", "Tầng trệt"],
    )
    assert slot_values["name"] == ("Kitchen Light", "Desk Lamp")
    assert slot_values["entity"] == ("Kitchen Light", "Desk Lamp")
    assert slot_values["entity_name"] == ("Kitchen Light", "Desk Lamp")
    assert slot_values["area"] == ("Kitchen", "Bếp")
    assert slot_values["area_name"] == ("Kitchen", "Bếp")
    assert slot_values["floor"] == ("Ground Floor", "Tầng trệt")
    assert slot_values["floor_name"] == ("Ground Floor", "Tầng trệt")
    assert slot_values["name:light"] == ("Kitchen Light",)
    assert slot_values["entity:media_player"] == ("Speaker",)


def test_merge_slot_values_preserves_explicit_intent_values() -> None:
    """Keep explicit list values ahead of registry fallback values."""
    merged = merge_slot_values(
        {"name": ("explicit light",)},
        {"name": ("registry light",), "area": ("Kitchen",)},
    )
    assert merged == {"name": ("explicit light",), "area": ("Kitchen",)}


def test_entry_names_and_aliases_excludes_registry_ids() -> None:
    """Do not expose machine ids as spoken area or floor names."""
    names = tuple(
        _entry_names_and_aliases(
            [
                SimpleNamespace(
                    name="Kitchen Room",
                    id="kitchen",
                    floor_id="first_floor",
                    aliases=["Bếp"],
                )
            ]
        )
    )

    assert names == ("Kitchen Room", "Bếp")


def test_entity_names_extraction() -> None:
    """Test extracting spoke entity names and aliases."""
    hass = MagicMock()
    er = MagicMock()

    er.async_get_entity_aliases = MagicMock(return_value=[" Alias1 ", "Alias2"])
    entry = SimpleNamespace(name="My Device", original_name="Orig Device")
    state = SimpleNamespace(name="State Device")

    names = list(_entity_names(hass, er, entry, state))
    assert "Alias1" in names
    assert " Alias1 " not in names
    assert "My Device" in names

    er.async_get_entity_aliases = MagicMock(side_effect=AttributeError)
    names = list(_entity_names(hass, er, entry, state))
    assert "My Device" in names

    names = list(_entity_names(hass, er, None, state))
    assert "State Device" in names

    er.async_get_entity_aliases = MagicMock(return_value=[])
    entry_without_names = SimpleNamespace(name=None, original_name=None)
    names = list(_entity_names(hass, er, entry_without_names, state))
    assert names == ["State Device"]

    entry_with_blank_name = SimpleNamespace(name="   ", original_name=" Orig Device ")
    names = list(_entity_names(hass, er, entry_with_blank_name, state))
    assert names == ["Orig Device"]


def test_registry_names_missing_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fallback when registries are not available (no HA environment)."""
    monkeypatch.setattr(registry, "_HAS_HA_REGISTRIES", False)

    hass = MagicMock()
    assert _area_names(hass) == ()
    assert _floor_names(hass) == ()
    assert _exposed_entity_names_by_domain(hass) == {}


def test_registry_import_fallback() -> None:
    """Test registry module import fallback when HA registries are missing."""
    # Patch HA imports to None to trigger ImportError/RuntimeError
    patches = {
        "homeassistant.components.homeassistant.exposed_entities": None,
        "homeassistant.helpers.area_registry": None,
        "homeassistant.helpers.entity_registry": None,
        "homeassistant.helpers.floor_registry": None,
    }
    with patch.dict(sys.modules, patches):
        importlib.reload(registry)
        assert registry.async_should_expose is None
        assert registry.ar is None
        assert registry.er is None
        assert registry.fr is None
        assert registry._HAS_HA_REGISTRIES is False

    # Restore module state
    importlib.reload(registry)


def test_build_registry_slot_values_non_string_domains() -> None:
    """Test build_registry_slot_values ignores non-string or empty domain keys."""
    slot_values = build_registry_slot_values(
        entity_names_by_domain=cast(Any, {None: ["name"], "": ["name"], 42: ["name"]})
    )
    assert not any(key.endswith(("None", "42")) for key in slot_values)


def test_exposed_entity_names_by_domain_normal() -> None:
    """Test _exposed_entity_names_by_domain with mock HA registries."""
    hass = MagicMock()

    # Mock states
    state_1 = SimpleNamespace(entity_id="light.kitchen", name="Kitchen Light")
    state_2 = SimpleNamespace(entity_id="switch.bedroom", name="Bedroom Switch")
    # Invalid entity_ids to cover branch checks
    state_invalid_1 = SimpleNamespace(entity_id=123)
    state_invalid_2 = SimpleNamespace(entity_id="invalid_no_dot")

    hass.states.async_all = MagicMock(
        return_value=[state_1, state_2, state_invalid_1, state_invalid_2]
    )

    # Mock entity registry
    mock_entity_registry = MagicMock()
    entry_1 = SimpleNamespace(name="Kitchen Light Entry", original_name="Kitchen Light Orig")
    mock_entity_registry.async_get = MagicMock(
        side_effect=lambda eid: entry_1 if eid == "light.kitchen" else None
    )

    # Mock area registry
    mock_area_registry = MagicMock()
    area_entry_1 = SimpleNamespace(
        name="Kitchen Area", aliases=["kitchen_alias_1", 123]
    )  # 123 is non-string alias
    area_entry_2 = SimpleNamespace(name=None, aliases=[])
    mock_area_registry.async_list_areas = MagicMock(return_value=[area_entry_1, area_entry_2])

    # Mock floor registry
    mock_floor_registry = MagicMock()
    floor_entry_1 = SimpleNamespace(name="Ground Floor", aliases=["floor_alias_1"])
    mock_floor_registry.async_list_floors = MagicMock(return_value=[floor_entry_1])

    # Patch registry functions/attributes
    with (
        patch.object(registry, "_HAS_HA_REGISTRIES", True),
        patch.object(registry.er, "async_get", return_value=mock_entity_registry),
        patch.object(
            registry.er,
            "async_get_entity_aliases",
            side_effect=lambda hass, entry, allow_empty: (
                ["alias_kitchen"] if entry == entry_1 else []
            ),
            create=True,
        ),
        patch.object(registry.ar, "async_get", return_value=mock_area_registry),
        patch.object(registry.fr, "async_get", return_value=mock_floor_registry),
        patch.object(registry, "async_should_expose", return_value=True),
    ):
        _test_exposed_entity_names_by_domain_normal(hass)
    # Test AttributeError/RuntimeError handling in registries
    with (
        patch.object(registry, "_HAS_HA_REGISTRIES", True),
        patch.object(registry.er, "async_get", side_effect=RuntimeError),
        patch.object(registry.ar, "async_get", side_effect=AttributeError),
        patch.object(registry.fr, "async_get", side_effect=RuntimeError),
    ):
        assert _exposed_entity_names_by_domain(hass) == {}
        assert _area_names(hass) == ()
        assert _floor_names(hass) == ()

    # Test async_should_expose raises KeyError/RuntimeError
    with (
        patch.object(registry, "_HAS_HA_REGISTRIES", True),
        patch.object(registry.er, "async_get", return_value=mock_entity_registry),
        patch.object(registry, "async_should_expose", side_effect=KeyError),
    ):
        assert _exposed_entity_names_by_domain(hass) == {}


def _test_exposed_entity_names_by_domain_normal(hass: Any) -> None:
    """Test _exposed_entity_names_by_domain with normal mock registries."""
    # Test exposed_entity_names
    exposed = _exposed_entity_names_by_domain(hass)
    assert "light" in exposed
    assert "switch" in exposed
    assert "alias_kitchen" in exposed["light"]
    assert "Bedroom Switch" in exposed["switch"]

    # Test area_names
    areas = _area_names(hass)
    assert "Kitchen Area" in areas
    assert "kitchen_alias_1" in areas

    # Test floor_names
    floors = _floor_names(hass)
    assert "Ground Floor" in floors
    assert "floor_alias_1" in floors
