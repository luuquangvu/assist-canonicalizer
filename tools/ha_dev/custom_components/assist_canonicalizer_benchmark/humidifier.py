"""Deterministic humidifier entities for the managed benchmark home."""

from __future__ import annotations

from typing import Any, override

from homeassistant.components.humidifier import (
    HumidifierDeviceClass,
    HumidifierEntity,
)
from homeassistant.components.humidifier.const import HumidifierAction
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Add the benchmark humidifiers."""
    async_add_entities(
        (
            BenchmarkHumidifier(
                "Primary Bedroom Humidifier",
                "humidifier_primary_bedroom",
                current_humidity=42,
                target_humidity=48,
                device_class=HumidifierDeviceClass.HUMIDIFIER,
            ),
            BenchmarkHumidifier(
                "Laundry Dehumidifier",
                "dehumidifier_laundry",
                current_humidity=61,
                target_humidity=50,
                device_class=HumidifierDeviceClass.DEHUMIDIFIER,
            ),
        )
    )


class BenchmarkHumidifier(HumidifierEntity):
    """In-memory humidifier whose state is reset with every benchmark home."""

    _attr_should_poll = False

    def __init__(
        self,
        name: str,
        unique_id: str,
        *,
        current_humidity: float,
        target_humidity: float,
        device_class: HumidifierDeviceClass,
    ) -> None:
        """Initialize a deterministic humidifier."""
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._attr_is_on = False
        self._attr_action = HumidifierAction.OFF
        self._attr_current_humidity = current_humidity
        self._attr_target_humidity = target_humidity
        self._attr_device_class = device_class

    @override
    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on the humidifier."""
        self._attr_is_on = True
        self._attr_action = HumidifierAction.HUMIDIFYING
        self.async_write_ha_state()

    @override
    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off the humidifier."""
        self._attr_is_on = False
        self._attr_action = HumidifierAction.OFF
        self.async_write_ha_state()

    @override
    async def async_set_humidity(self, humidity: int) -> None:
        """Set the target humidity."""
        self._attr_target_humidity = humidity
        self.async_write_ha_state()
