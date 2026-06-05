"""Fixture configurations for Assist Canonicalizer tests."""

import asyncio
import contextlib
import shutil
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    """Override pytest-homeassistant-custom-component autouse fixture.

    Prevents RuntimeError when querying the event loop during sync tests under
    HassEventLoopPolicy.
    """
    with contextlib.suppress(RuntimeError):
        loop = asyncio.get_event_loop()
        loop.set_debug(True)


@pytest.fixture(autouse=True)
def verify_cleanup(
    expected_lingering_tasks: bool,
    expected_lingering_timers: bool,
) -> Generator[None]:
    """Override pytest-homeassistant-custom-component autouse fixture.

    Prevents RuntimeError when querying the event loop during sync tests under
    HassEventLoopPolicy.
    """
    try:
        event_loop = asyncio.get_event_loop()
    except RuntimeError:
        yield
        return

    yield
    with contextlib.suppress(RuntimeError):
        event_loop.run_until_complete(event_loop.shutdown_default_executor())


@pytest.fixture(autouse=True)
def cleanup_magicmock_dir() -> Generator[None]:
    """Remove MagicMock directory created by tests using raw MagicMock for hass."""
    try:
        yield
    finally:
        magicmock_dir = Path("MagicMock")
        if magicmock_dir.is_dir():
            shutil.rmtree(magicmock_dir)
