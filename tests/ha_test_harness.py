"""Compatibility access to the installed Home Assistant pytest harness.

Home Assistant 2026.11 replaced ``pytest-homeassistant-custom-component`` with
``pytest-homeassistant``. Both harnesses expose the helpers used by this test
suite under the same names, so tests import them from this module instead of
depending on one distribution directly.
"""

import importlib
from collections.abc import Mapping
from datetime import datetime
from types import ModuleType
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

_HARNESS_MODULES = (
    "pytest_homeassistant.common",
    "pytest_homeassistant_custom_component.common",
)


def _load_harness_module() -> ModuleType:
    """Return the installed Home Assistant test harness common module."""
    for module_name in _HARNESS_MODULES:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as err:
            missing_name = err.name or ""
            if module_name != missing_name and not module_name.startswith(f"{missing_name}."):
                raise
    raise ModuleNotFoundError(
        "No Home Assistant pytest harness is installed; install "
        "'pytest-homeassistant' or 'pytest-homeassistant-custom-component'."
    )


HARNESS_MODULE = _load_harness_module()

if TYPE_CHECKING:

    class MockConfigEntry(ConfigEntry):
        """Type-only stand-in for the harness MockConfigEntry helper."""

        def __init__(
            self,
            *,
            created_at: datetime | None = None,
            data: Mapping[str, Any] | None = None,
            domain: str = "test",
            entry_id: str | None = None,
            minor_version: int = 1,
            options: Mapping[str, Any] | None = None,
            title: str = "Mock Title",
            unique_id: str | None = None,
            version: int = 1,
            **kwargs: Any,
        ) -> None:
            """Initialize the type-only configuration entry."""

        def add_to_hass(self, hass: HomeAssistant) -> None:
            """Add the configuration entry to Home Assistant."""

    def async_fire_time_changed(
        hass: HomeAssistant,
        datetime_: datetime | None = None,
        fire_all: bool = False,
    ) -> None:
        """Fire a time changed event in the type-only harness surface."""

    def get_test_config_dir(*add_path: str) -> str:
        """Return a path to the test config dir in the type-only harness surface."""
        return ""

else:
    MockConfigEntry = HARNESS_MODULE.MockConfigEntry
    async_fire_time_changed = HARNESS_MODULE.async_fire_time_changed
    get_test_config_dir = HARNESS_MODULE.get_test_config_dir
