"""Deterministic lock entities for the managed benchmark home."""

from __future__ import annotations

from typing import Any, override

from homeassistant.components.lock import LockEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Add the benchmark locks."""
    async_add_entities(
        (
            BenchmarkLock("Front Door Lock", "lock_front_door", locked=True),
            BenchmarkLock("Patio Door Lock", "lock_patio_door", locked=False),
            BenchmarkLock("Garage Entry Lock", "lock_garage_entry", locked=True),
        )
    )


class BenchmarkLock(LockEntity):
    """In-memory lock with immediate deterministic transitions."""

    _attr_should_poll = False

    def __init__(self, name: str, unique_id: str, *, locked: bool) -> None:
        """Initialize a benchmark lock."""
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._attr_is_locked = locked

    @override
    async def async_lock(self, **kwargs: Any) -> None:
        """Lock the entity."""
        self._attr_is_locked = True
        self.async_write_ha_state()

    @override
    async def async_unlock(self, **kwargs: Any) -> None:
        """Unlock the entity."""
        self._attr_is_locked = False
        self.async_write_ha_state()
