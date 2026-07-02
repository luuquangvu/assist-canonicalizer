"""Tests for Assist Canonicalizer config flow helpers."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import voluptuous as vol

import custom_components.assist_canonicalizer.config_flow as config_flow
from custom_components.assist_canonicalizer.config_flow import (
    AssistCanonicalizerConfigFlow,
    AssistCanonicalizerOptionsFlow,
    _available_fallback_agents,
    _config_schema,
)


def test_available_fallback_agents_includes_conversation_entities(
    monkeypatch: pytest.MonkeyPatch,
    fallback_agent_manager_factory: Any,
) -> None:
    """Include ConversationEntity agent ids alongside config-entry agents."""
    data_component_key = object()
    manager = fallback_agent_manager_factory(
        [SimpleNamespace(id="entry-agent-id", name="Entry Agent")]
    )

    monkeypatch.setattr(config_flow, "get_agent_manager", lambda hass: manager)
    monkeypatch.setattr(config_flow, "DATA_COMPONENT", data_component_key)
    monkeypatch.setattr(config_flow, "HOME_ASSISTANT_AGENT", "conversation.home_assistant")
    monkeypatch.setattr(config_flow, "_HAS_CONVERSATION_AGENTS", True)

    hass = SimpleNamespace(
        data={
            data_component_key: SimpleNamespace(
                entities=[
                    SimpleNamespace(
                        entity_id="conversation.openai_conversation",
                        name="OpenAI Conversation",
                    ),
                    SimpleNamespace(
                        entity_id="conversation.assist_canonicalizer",
                        name="Assist Canonicalizer",
                    ),
                ],
            ),
        },
    )

    choices = _available_fallback_agents(hass, "conversation.assist_canonicalizer")

    assert choices == {
        "entry-agent-id": "Entry Agent",
        "conversation.openai_conversation": "OpenAI Conversation",
    }


def test_available_fallback_agents_excludes_own_agent_and_entity(
    monkeypatch: pytest.MonkeyPatch,
    fallback_agent_manager_factory: Any,
    mock_conversation_entity_type: type,
) -> None:
    """Exclude the canonicalizer's own entity and agent from fallback choices."""
    data_component_key = object()
    manager = fallback_agent_manager_factory(
        [
            SimpleNamespace(id="entry-agent-id", name="Entry Agent"),
            SimpleNamespace(id="canonicalizer-config-entry-id", name="Assist Canonicalizer Agent"),
        ],
        {"canonicalizer-config-entry-id": mock_conversation_entity_type()},
    )

    monkeypatch.setattr(config_flow, "get_agent_manager", lambda hass: manager)
    monkeypatch.setattr(config_flow, "DATA_COMPONENT", data_component_key)
    monkeypatch.setattr(config_flow, "HOME_ASSISTANT_AGENT", "conversation.home_assistant")
    monkeypatch.setattr(config_flow, "_HAS_CONVERSATION_AGENTS", True)
    monkeypatch.setattr(config_flow, "ConversationEntity", mock_conversation_entity_type)

    class MockRegistryEntry:
        """Mock registry entry containing config_entry_id."""

        def __init__(self, config_entry_id: str) -> None:
            """Initialize entry."""
            self.config_entry_id = config_entry_id

    class MockEntity:
        """Mock entity structure."""

        def __init__(
            self,
            entity_id: str,
            name: str,
            config_entry_id: str | None = None,
            unique_id: str | None = None,
        ) -> None:
            """Initialize entity attributes."""
            self.entity_id = entity_id
            self.name = name
            if config_entry_id:
                self.registry_entry = MockRegistryEntry(config_entry_id)
            if unique_id:
                self.unique_id = unique_id

    hass = SimpleNamespace(
        data={
            data_component_key: SimpleNamespace(
                entities=[
                    MockEntity("conversation.openai_conversation", "OpenAI Conversation"),
                    MockEntity(
                        "conversation.assist_canonicalizer",
                        "Assist Canonicalizer",
                        config_entry_id="canonicalizer-config-entry-id",
                    ),
                    MockEntity(
                        "conversation.assist_canonicalizer_2",
                        "Assist Canonicalizer 2",
                        unique_id="canonicalizer-config-entry-id-conversation",
                    ),
                ],
            ),
        },
    )

    choices = _available_fallback_agents(hass, "canonicalizer-config-entry-id")

    assert choices == {
        "entry-agent-id": "Entry Agent",
        "conversation.openai_conversation": "OpenAI Conversation",
    }


def test_config_schema_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that _config_schema returns correct types and validation fields."""
    monkeypatch.setattr(config_flow, "_HAS_CONVERSATION_AGENTS", True)
    monkeypatch.setattr(config_flow, "HOME_ASSISTANT_AGENT", "conversation.home_assistant")
    monkeypatch.setattr(
        config_flow,
        "_available_fallback_agents",
        lambda hass, exclude_agent_id: {"conversation.home_assistant": "Home Assistant"},
    )

    hass = SimpleNamespace()
    schema = _config_schema(
        hass,
        {
            "min_confidence": 0.5,
            "min_margin": 0.1,
        },
    )

    assert isinstance(schema, vol.Schema)
    res: Any = schema(
        {
            "fallback_agent_id": "conversation.home_assistant",
            "min_confidence": 0.8,
            "min_margin": 0.05,
        }
    )
    assert res["min_confidence"] == 0.8
    assert res["min_margin"] == 0.05


@pytest.mark.asyncio
async def test_config_flow_steps() -> None:
    """Test user step options in config flow."""
    flow = AssistCanonicalizerConfigFlow()
    flow.hass = MagicMock()

    # 1. Existing entries aborts
    with (
        patch.object(flow, "_async_current_entries", return_value=["existing_entry"]),
        patch.object(flow, "async_abort", return_value="abort_result") as mock_abort,
    ):
        res = await flow.async_step_user(None)
        assert res == "abort_result"
        mock_abort.assert_called_once_with(reason="single_instance_allowed")

    # 2. No user input shows form
    with (
        patch.object(flow, "_async_current_entries", return_value=[]),
        patch.object(flow, "async_show_form", return_value="show_form_result"),
    ):
        res = await flow.async_step_user(None)
        assert res == "show_form_result"

    # 3. User input valid creates entry
    with (
        patch.object(flow, "_async_current_entries", return_value=[]),
        patch.object(
            flow, "async_create_entry", return_value="create_entry_result"
        ) as mock_create_entry,
    ):
        valid_input = {}
        res = await flow.async_step_user(valid_input)
        assert res == "create_entry_result"
        mock_create_entry.assert_called_once_with(
            title="Assist Canonicalizer",
            data=valid_input,
        )


@pytest.mark.asyncio
async def test_options_flow_steps() -> None:
    """Test options step flow options."""
    entry = MagicMock()
    entry.data = {}
    entry.options = {}
    entry.entry_id = "test_entry_id"

    flow = AssistCanonicalizerOptionsFlow(entry)
    flow.hass = MagicMock()

    # 1. No input shows form
    with patch.object(flow, "async_show_form", return_value="show_form_result"):
        res = await flow.async_step_init(None)
        assert res == "show_form_result"

    # 2. Valid input creates options entry
    with patch.object(
        flow, "async_create_entry", return_value="create_entry_result"
    ) as mock_create_entry:
        valid_input = {}
        res = await flow.async_step_init(valid_input)
        assert res == "create_entry_result"
        mock_create_entry.assert_called_once_with(
            title="",
            data=valid_input,
        )
