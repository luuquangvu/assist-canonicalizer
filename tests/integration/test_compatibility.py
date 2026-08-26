"""Functional compatibility contract for supported Home Assistant versions."""

import inspect
from datetime import timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant import config_entries
from homeassistant.components import conversation
from homeassistant.components.conversation.const import HOME_ASSISTANT_AGENT
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.core import Context, HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import area_registry, entity_registry, floor_registry
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_fire_time_changed

from custom_components.assist_canonicalizer.const import (
    DATA_RUNTIME,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DOMAIN,
    AttributeName,
    ConfigKey,
    ServiceName,
)
from custom_components.assist_canonicalizer.recognition import (
    RecognitionKind,
    async_observe_delegated_text,
)

pytestmark = pytest.mark.compatibility


@pytest.mark.asyncio
async def test_home_assistant_functional_contract(
    hass: HomeAssistant,
) -> None:
    """Load, index, process, diagnose, and unload through Home Assistant APIs."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, conversation.DOMAIN, {})
    assert await async_setup_component(hass, "assist_pipeline", {})

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Assist Canonicalizer",
        data={},
        options={
            ConfigKey.MIN_CONFIDENCE: 1.0,
            ConfigKey.MIN_MARGIN: 1.0,
        },
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert conversation.async_get_agent_info(hass, entry.entry_id) is not None
    assert hass.services.has_service(DOMAIN, ServiceName.REBUILD_INDEX)
    assert hass.services.has_service(DOMAIN, ServiceName.TEST_MATCH)
    assert hass.services.has_service(DOMAIN, ServiceName.CLEAR_INDEX)
    assert hass.services.has_service(DOMAIN, ServiceName.DUMP_CANDIDATES)
    assert hass.services.has_service(DOMAIN, ServiceName.DIAGNOSTICS)
    assert hass.services.has_service(DOMAIN, ServiceName.SET_FALLBACK_AGENT)

    # 1. Rebuild index service
    rebuild_response = await hass.services.async_call(
        DOMAIN,
        ServiceName.REBUILD_INDEX,
        {AttributeName.LANGUAGE: "en"},
        blocking=True,
        return_response=True,
    )
    assert isinstance(rebuild_response, dict)
    assert rebuild_response[AttributeName.LANGUAGE] == "en"
    candidate_count = rebuild_response[AttributeName.CANDIDATE_COUNT]
    assert isinstance(candidate_count, int)
    assert candidate_count > 0

    # 2. Test match service
    test_match_response = await hass.services.async_call(
        DOMAIN,
        ServiceName.TEST_MATCH,
        {AttributeName.TEXT: "turn on kitchen light", AttributeName.LANGUAGE: "en"},
        blocking=True,
        return_response=True,
    )
    assert isinstance(test_match_response, dict)
    assert test_match_response[AttributeName.LANGUAGE] == "en"
    assert "confidence_gate" in test_match_response
    assert "top_candidates" in test_match_response

    # 3. Dump candidates service
    dump_response = await hass.services.async_call(
        DOMAIN,
        ServiceName.DUMP_CANDIDATES,
        {AttributeName.LANGUAGE: "en"},
        blocking=True,
        return_response=True,
    )
    assert isinstance(dump_response, dict)
    assert dump_response[AttributeName.LANGUAGE] == "en"
    assert "candidate_sample" in dump_response

    # 4. Set fallback agent service
    fallback_response = await hass.services.async_call(
        DOMAIN,
        ServiceName.SET_FALLBACK_AGENT,
        {AttributeName.AGENT_ID: HOME_ASSISTANT_AGENT},
        blocking=True,
        return_response=True,
    )
    assert isinstance(fallback_response, dict)
    assert fallback_response["fallback_agent_id"] == HOME_ASSISTANT_AGENT

    # 5. Process conversation input
    agent = conversation.agent_manager.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    original_process = cast(Any, agent)._async_process_with_runtime
    lifecycle_observations: list[tuple[str, str] | None] = []

    async def observe_chat_lifecycle(user_input: conversation.ConversationInput) -> object:
        """Record the modern HA chat-log context before processing the request."""
        try:
            from homeassistant.components.conversation.chat_log import current_chat_log
        except ImportError:
            lifecycle_observations.append(None)
        else:
            chat_log = current_chat_log.get(None)
            if chat_log is None:
                lifecycle_observations.append(None)
            else:
                user_content = chat_log.content[-1]
                if user_content.role == "user":
                    lifecycle_observations.append((user_content.role, user_content.content))
                else:
                    lifecycle_observations.append(None)
        return await original_process(user_input)

    raw_text = "compatibility smoke phrase with no supported intent"
    with patch.object(
        agent,
        "_async_process_with_runtime",
        observe_chat_lifecycle,
    ):
        result = await conversation.async_converse(
            hass,
            raw_text,
            "compatibility-smoke",
            Context(),
            language="en",
            agent_id=entry.entry_id,
        )
    assert result.response is not None
    if hasattr(conversation.ConversationEntity, "_async_handle_message"):
        assert lifecycle_observations == [("user", raw_text)]

    # 6. Diagnostics service
    diagnostics = await hass.services.async_call(
        DOMAIN,
        ServiceName.DIAGNOSTICS,
        {},
        blocking=True,
        return_response=True,
    )
    assert isinstance(diagnostics, dict)
    assert diagnostics["last_request_id"] == "compatibility-smoke"
    assert isinstance(diagnostics["cached_indexes"], dict)
    assert "en" in diagnostics["cached_indexes"]

    # 7. Clear index service
    clear_response = await hass.services.async_call(
        DOMAIN,
        ServiceName.CLEAR_INDEX,
        {AttributeName.LANGUAGE: "en"},
        blocking=True,
        return_response=True,
    )
    assert isinstance(clear_response, dict)
    assert clear_response[AttributeName.LANGUAGE] == "en"
    assert "cleared_cached_languages" in clear_response

    # 8. Unload entry
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert conversation.async_get_agent_info(hass, entry.entry_id) is None
    assert not hass.services.has_service(DOMAIN, ServiceName.REBUILD_INDEX)
    assert not hass.services.has_service(DOMAIN, ServiceName.TEST_MATCH)
    assert not hass.services.has_service(DOMAIN, ServiceName.CLEAR_INDEX)
    assert not hass.services.has_service(DOMAIN, ServiceName.DUMP_CANDIDATES)
    assert not hass.services.has_service(DOMAIN, ServiceName.DIAGNOSTICS)
    assert not hass.services.has_service(DOMAIN, ServiceName.SET_FALLBACK_AGENT)


@pytest.mark.asyncio
async def test_compatibility_registry_and_observation(
    hass: HomeAssistant,
) -> None:
    """Verify registry slot integration and live default agent observation."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, conversation.DOMAIN, {})

    area_reg = area_registry.async_get(hass)
    entity_reg = entity_registry.async_get(hass)
    floor_reg = floor_registry.async_get(hass)

    area = area_reg.async_get_or_create("compatibility_test_area")
    entity = entity_reg.async_get_or_create(
        "light", "compatibility_test", "light_1", suggested_object_id="test_light"
    )
    entity = entity_reg.async_update_entity(
        entity.entity_id,
        aliases=["Compatibility Lamp Alias"],
        name="Compatibility Test Light",
    )
    hass.states.async_set(
        entity.entity_id,
        "on",
        {"friendly_name": "Compatibility State Light"},
    )
    async_expose_entity(hass, "conversation", entity.entity_id, True)
    floor = floor_reg.async_create("compatibility_test_floor")

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Assist Canonicalizer",
        data={},
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    raw_input_text = "turn on the test light raw input"
    delegated_text = "turn on Compatibility Lamp Alias"
    test_context = Context()

    input_kwargs: dict[str, Any] = {
        "text": raw_input_text,
        "context": test_context,
        "conversation_id": "test-conv-id",
        "device_id": None,
        "language": "en",
    }
    sig_params = inspect.signature(conversation.ConversationInput).parameters
    if "agent_id" in sig_params:
        input_kwargs["agent_id"] = entry.entry_id
    if "satellite_id" in sig_params:
        input_kwargs["satellite_id"] = None

    conv_input = conversation.ConversationInput(**input_kwargs)

    default_agent = conversation.agent_manager.async_get_agent(hass, HOME_ASSISTANT_AGENT)
    assert default_agent is not None

    spy_recognize_intent = AsyncMock(wraps=cast(Any, default_agent).async_recognize_intent)
    with patch.object(default_agent, "async_recognize_intent", spy_recognize_intent):
        observation = await async_observe_delegated_text(hass, conv_input, delegated_text)

    assert spy_recognize_intent.called
    forwarded_input = spy_recognize_intent.call_args[0][0]
    assert forwarded_input.text == delegated_text
    assert forwarded_input.context is test_context
    assert forwarded_input.conversation_id == "test-conv-id"
    assert forwarded_input.language == "en"

    assert observation.kind is RecognitionKind.INTENT
    assert observation.executable
    assert observation.intent_name == "HassTurnOn"
    assert not observation.unmatched_entities

    runtime = hass.data[DOMAIN][entry.entry_id][DATA_RUNTIME]
    assert "Compatibility Test Light" in runtime.registry_slot_values["name"]
    assert "Compatibility Lamp Alias" in runtime.registry_slot_values["name"]
    assert "compatibility_test_area" in runtime.registry_slot_values["area"]
    assert "compatibility_test_floor" in runtime.registry_slot_values["floor"]

    str_entity_event = str(entity_registry.EVENT_ENTITY_REGISTRY_UPDATED)
    str_area_event = str(area_registry.EVENT_AREA_REGISTRY_UPDATED)
    str_floor_event = str(floor_registry.EVENT_FLOOR_REGISTRY_UPDATED)

    events_fired: dict[str, int] = {
        str_entity_event: 0,
        str_area_event: 0,
        str_floor_event: 0,
    }

    @callback
    def _track_event(event: object) -> None:
        """Track count of fired registry update events."""
        event_type = getattr(event, "event_type", None)
        if isinstance(event_type, str) and event_type in events_fired:
            events_fired[event_type] += 1

    for event_name in events_fired:
        hass.bus.async_listen(event_name, _track_event)

    entity_reg.async_update_entity(
        entity.entity_id,
        aliases=["Updated Compatibility Alias"],
        name="Updated Test Light",
    )
    area_reg.async_update(area.id, name="Updated Test Area")
    floor_reg.async_update(floor.floor_id, name="Updated Test Floor")
    await hass.async_block_till_done()

    assert events_fired[str_entity_event] >= 1
    assert events_fired[str_area_event] >= 1
    assert events_fired[str_floor_event] >= 1

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=6))
    await hass.async_block_till_done()
    assert "Updated Test Light" in runtime.registry_slot_values["name"]
    assert "Updated Compatibility Alias" in runtime.registry_slot_values["name"]
    assert "Updated Test Area" in runtime.registry_slot_values["area"]
    assert "Updated Test Floor" in runtime.registry_slot_values["floor"]

    assert await hass.config_entries.async_unload(entry.entry_id)
    assert runtime.closed
    generation_after_unload = runtime.registry_generation
    entity_reg.async_update_entity(entity.entity_id, name="Post Unload Test Light")
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=12))
    await hass.async_block_till_done()
    assert runtime.registry_generation == generation_after_unload
    assert runtime.rebuild_timer_cancel is None


@pytest.mark.asyncio
async def test_config_and_options_flow_framework_contract(hass: HomeAssistant) -> None:
    """Create and configure the integration through Home Assistant's flow managers."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, conversation.DOMAIN, {})

    form = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_USER},
    )
    assert form.get("type") is FlowResultType.FORM
    assert form.get("step_id") == "user"
    assert "errors" in form
    assert form.get("errors") is None

    created = await hass.config_entries.flow.async_configure(
        form["flow_id"],
        {
            "fallback_agent_id": HOME_ASSISTANT_AGENT,
            ConfigKey.MIN_CONFIDENCE: DEFAULT_MIN_CONFIDENCE,
            ConfigKey.MIN_MARGIN: DEFAULT_MIN_MARGIN,
        },
    )
    assert created.get("type") is FlowResultType.CREATE_ENTRY
    entry = created.get("result")
    assert isinstance(entry, config_entries.ConfigEntry)
    assert entry.domain == DOMAIN
    assert entry.data["fallback_agent_id"] == HOME_ASSISTANT_AGENT

    options_form = await hass.config_entries.options.async_init(entry.entry_id)
    assert options_form.get("type") is FlowResultType.FORM
    assert options_form.get("step_id") == "init"

    options_created = await hass.config_entries.options.async_configure(
        options_form["flow_id"],
        {
            "fallback_agent_id": HOME_ASSISTANT_AGENT,
            ConfigKey.MIN_CONFIDENCE: 0.7,
            ConfigKey.MIN_MARGIN: 0.08,
        },
    )
    assert options_created.get("type") is FlowResultType.CREATE_ENTRY
    assert entry.options[ConfigKey.MIN_CONFIDENCE] == 0.7
    assert entry.options[ConfigKey.MIN_MARGIN] == 0.08

    duplicate = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_USER},
    )
    assert duplicate.get("type") is FlowResultType.ABORT
    assert duplicate.get("reason") == "single_instance_allowed"
