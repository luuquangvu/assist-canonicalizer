"""Deterministic lawn-mower entities for the managed benchmark home."""

from __future__ import annotations

from typing import override

from homeassistant.components.lawn_mower import LawnMowerEntity
from homeassistant.components.lawn_mower.const import (
    LawnMowerActivity,
    LawnMowerEntityFeature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Add the benchmark lawn mowers."""
    async_add_entities(
        (
            BenchmarkLawnMower("Front Garden Mower", "mower_front_garden"),
            BenchmarkLawnMower("Back Garden Mower", "mower_back_garden"),
        )
    )


class BenchmarkLawnMower(LawnMowerEntity):
    """In-memory lawn mower with deterministic supported operations."""

    _attr_should_poll = False
    _attr_supported_features = (
        LawnMowerEntityFeature.START_MOWING
        | LawnMowerEntityFeature.PAUSE
        | LawnMowerEntityFeature.DOCK
    )

    def __init__(self, name: str, unique_id: str) -> None:
        """Initialize a docked mower."""
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._attr_activity = LawnMowerActivity.DOCKED

    @override
    async def async_start_mowing(self) -> None:
        """Start mowing."""
        self._attr_activity = LawnMowerActivity.MOWING
        self.async_write_ha_state()

    @override
    async def async_dock(self) -> None:
        """Dock the mower."""
        self._attr_activity = LawnMowerActivity.DOCKED
        self.async_write_ha_state()

    @override
    async def async_pause(self) -> None:
        """Pause mowing."""
        self._attr_activity = LawnMowerActivity.PAUSED
        self.async_write_ha_state()
