"""Fixture configurations for Assist Canonicalizer tests."""

import asyncio
import contextlib
import shutil
from collections.abc import Callable, Generator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class MockConversationEntity:
    """Reusable mock conversation entity for fallback-agent isinstance checks."""


class MockConversationAgentManager:
    """Reusable fake conversation manager with agent metadata and instances."""

    def __init__(
        self,
        agent_infos: Sequence[SimpleNamespace],
        agents: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the fake manager."""
        self._agent_infos = list(agent_infos)
        self._agents = dict(agents or {})

    def async_get_agent_info(self) -> list[SimpleNamespace]:
        """Return configured conversation agent metadata."""
        return list(self._agent_infos)

    def async_get_agent(self, agent_id: str) -> Any:
        """Return configured agent instances by id."""
        return self._agents.get(agent_id)


@pytest.fixture
def mock_conversation_entity_type() -> type[MockConversationEntity]:
    """Return the shared mock ConversationEntity type."""
    return MockConversationEntity


@pytest.fixture
def fallback_agent_manager_factory() -> Callable[
    [Sequence[SimpleNamespace], Mapping[str, Any] | None], MockConversationAgentManager
]:
    """Return a factory for fake conversation agent managers."""

    def factory(
        agent_infos: Sequence[SimpleNamespace],
        agents: Mapping[str, Any] | None = None,
    ) -> MockConversationAgentManager:
        return MockConversationAgentManager(agent_infos, agents)

    return factory


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
    if event_loop.is_closed() or event_loop.is_running():
        return
    shutdown_executor = event_loop.shutdown_default_executor()
    try:
        event_loop.run_until_complete(shutdown_executor)
    except RuntimeError:
        shutdown_executor.close()


@pytest.fixture(autouse=True)
def cleanup_magicmock_dir() -> Generator[None]:
    """Remove MagicMock directory created by tests using raw MagicMock for hass."""
    try:
        yield
    finally:
        magicmock_dir = Path("MagicMock")
        if magicmock_dir.is_dir():
            shutil.rmtree(magicmock_dir)
