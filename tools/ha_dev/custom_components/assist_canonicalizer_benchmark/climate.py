"""Deterministic climate entities for the managed benchmark home."""

from __future__ import annotations

from typing import Any, override

from homeassistant.components.climate import ClimateEntity
from homeassistant.components.climate.const import ClimateEntityFeature, HVACMode
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Add the benchmark climate devices."""
    async_add_entities(
        (
            BenchmarkClimate("Bedroom Thermostat", "climate_1", 21.0),
            BenchmarkClimate("Living Room Thermostat", "climate_2", 22.0),
            BenchmarkClimate("Office Thermostat", "climate_3", 23.0),
        )
    )


class BenchmarkClimate(ClimateEntity):
    """In-memory thermostat accepting the corpus's complete temperature range."""

    _attr_should_poll = False
    _attr_supported_features = (
        ClimateEntityFeature.TARGET_TEMPERATURE
        | ClimateEntityFeature.TURN_ON
        | ClimateEntityFeature.TURN_OFF
    )
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_min_temp = 0.0
    _attr_max_temp = 35.0
    _attr_target_temperature_step = 0.5

    def __init__(self, name: str, unique_id: str, temperature: float) -> None:
        """Initialize one thermostat."""
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._attr_current_temperature = temperature
        self._attr_target_temperature = temperature
        self._attr_hvac_mode = HVACMode.HEAT
        self._attr_hvac_modes = [HVACMode.HEAT, HVACMode.OFF]

    @override
    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set the target temperature."""
        if (temperature := kwargs.get(ATTR_TEMPERATURE)) is not None:
            self._attr_target_temperature = float(temperature)
        self.async_write_ha_state()

    @override
    async def async_turn_on(self) -> None:
        """Turn on heating."""
        self._attr_hvac_mode = HVACMode.HEAT
        self.async_write_ha_state()

    @override
    async def async_turn_off(self) -> None:
        """Turn off heating."""
        self._attr_hvac_mode = HVACMode.OFF
        self.async_write_ha_state()
