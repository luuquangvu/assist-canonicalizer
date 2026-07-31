"""Fixtures for Home Assistant integration tests."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_test_config_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep Home Assistant helper paths inside the per-test directory."""

    def _get_test_config_dir(*parts: str) -> str:
        return str(tmp_path.joinpath(*parts))

    monkeypatch.setattr(
        "pytest_homeassistant_custom_component.common.get_test_config_dir",
        _get_test_config_dir,
    )


@pytest.fixture
def hass_config_dir(tmp_path: Path) -> str:
    """Provide a per-test Home Assistant config directory."""
    return str(tmp_path)


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable custom integrations for every integration test."""


@pytest.fixture
def mock_storage() -> None:
    """Use Home Assistant's real Store implementation in integration tests."""
    # Intentional no-op override disables the harness storage mock,
    # allowing Store writes to use the isolated tmp_path configuration directory.
