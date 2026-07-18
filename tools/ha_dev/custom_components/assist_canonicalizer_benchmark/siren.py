"""Deterministic siren entities for the managed benchmark home."""

from __future__ import annotations

from typing import Any, override

from homeassistant.components.siren import SirenEntity
from homeassistant.components.siren.const import SirenEntityFeature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Add the benchmark sirens."""
    async_add_entities(
        (
            BenchmarkSiren("Indoor Siren", "siren_indoor"),
            BenchmarkSiren("Garden Siren", "siren_garden"),
        )
    )


class BenchmarkSiren(SirenEntity):
    """In-memory siren supporting deterministic on/off operations."""

    _attr_should_poll = False
    _attr_supported_features = SirenEntityFeature.TURN_ON | SirenEntityFeature.TURN_OFF

    def __init__(self, name: str, unique_id: str) -> None:
        """Initialize an inactive siren."""
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._attr_is_on = False

    @override
    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on the siren."""
        self._attr_is_on = True
        self.async_write_ha_state()

    @override
    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off the siren."""
        self._attr_is_on = False
        self.async_write_ha_state()
