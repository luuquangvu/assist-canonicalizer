"""Functional compatibility contract for supported Home Assistant versions."""

import inspect
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.components.conversation.const import HOME_ASSISTANT_AGENT
from homeassistant.core import Context, HomeAssistant, callback
from homeassistant.helpers import area_registry, entity_registry, floor_registry
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.assist_canonicalizer import _refresh_registry_slot_values
from custom_components.assist_canonicalizer.const import (
    ATTR_AGENT_ID,
    ATTR_CANDIDATE_COUNT,
    ATTR_LANGUAGE,
    ATTR_TEXT,
    CONF_MIN_CONFIDENCE,
    CONF_MIN_MARGIN,
    DATA_RUNTIME,
    DOMAIN,
    SERVICE_CLEAR_INDEX,
    SERVICE_DIAGNOSTICS,
    SERVICE_DUMP_CANDIDATES,
    SERVICE_REBUILD_INDEX,
    SERVICE_SET_FALLBACK_AGENT,
    SERVICE_TEST_MATCH,
)
from custom_components.assist_canonicalizer.recognition import async_observe_delegated_text

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
            CONF_MIN_CONFIDENCE: 1.0,
            CONF_MIN_MARGIN: 1.0,
        },
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert conversation.async_get_agent_info(hass, entry.entry_id) is not None
    assert hass.services.has_service(DOMAIN, SERVICE_REBUILD_INDEX)
    assert hass.services.has_service(DOMAIN, SERVICE_TEST_MATCH)
    assert hass.services.has_service(DOMAIN, SERVICE_CLEAR_INDEX)
    assert hass.services.has_service(DOMAIN, SERVICE_DUMP_CANDIDATES)
    assert hass.services.has_service(DOMAIN, SERVICE_DIAGNOSTICS)
    assert hass.services.has_service(DOMAIN, SERVICE_SET_FALLBACK_AGENT)

    # 1. Rebuild index service
    rebuild_response = await hass.services.async_call(
        DOMAIN,
        SERVICE_REBUILD_INDEX,
        {ATTR_LANGUAGE: "en"},
        blocking=True,
        return_response=True,
    )
    assert isinstance(rebuild_response, dict)
    assert rebuild_response[ATTR_LANGUAGE] == "en"
    candidate_count = rebuild_response[ATTR_CANDIDATE_COUNT]
    assert isinstance(candidate_count, int)
    assert candidate_count > 0

    # 2. Test match service
    test_match_response = await hass.services.async_call(
        DOMAIN,
        SERVICE_TEST_MATCH,
        {ATTR_TEXT: "turn on kitchen light", ATTR_LANGUAGE: "en"},
        blocking=True,
        return_response=True,
    )
    assert isinstance(test_match_response, dict)
    assert test_match_response[ATTR_LANGUAGE] == "en"
    assert "confidence_gate" in test_match_response
    assert "top_candidates" in test_match_response

    # 3. Dump candidates service
    dump_response = await hass.services.async_call(
        DOMAIN,
        SERVICE_DUMP_CANDIDATES,
        {ATTR_LANGUAGE: "en"},
        blocking=True,
        return_response=True,
    )
    assert isinstance(dump_response, dict)
    assert dump_response[ATTR_LANGUAGE] == "en"
    assert "candidate_sample" in dump_response

    # 4. Set fallback agent service
    fallback_response = await hass.services.async_call(
        DOMAIN,
        SERVICE_SET_FALLBACK_AGENT,
        {ATTR_AGENT_ID: HOME_ASSISTANT_AGENT},
        blocking=True,
        return_response=True,
    )
    assert isinstance(fallback_response, dict)
    assert fallback_response["fallback_agent_id"] == HOME_ASSISTANT_AGENT

    # 5. Process conversation input
    result = await conversation.async_converse(
        hass,
        "compatibility smoke phrase with no supported intent",
        "compatibility-smoke",
        Context(),
        language="en",
        agent_id=entry.entry_id,
    )
    assert result.response is not None

    # 6. Diagnostics service
    diagnostics = await hass.services.async_call(
        DOMAIN,
        SERVICE_DIAGNOSTICS,
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
        SERVICE_CLEAR_INDEX,
        {ATTR_LANGUAGE: "en"},
        blocking=True,
        return_response=True,
    )
    assert isinstance(clear_response, dict)
    assert clear_response[ATTR_LANGUAGE] == "en"
    assert "cleared_cached_languages" in clear_response

    # 8. Unload entry
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert conversation.async_get_agent_info(hass, entry.entry_id) is None
    assert not hass.services.has_service(DOMAIN, SERVICE_REBUILD_INDEX)
    assert not hass.services.has_service(DOMAIN, SERVICE_TEST_MATCH)
    assert not hass.services.has_service(DOMAIN, SERVICE_CLEAR_INDEX)
    assert not hass.services.has_service(DOMAIN, SERVICE_DUMP_CANDIDATES)
    assert not hass.services.has_service(DOMAIN, SERVICE_DIAGNOSTICS)
    assert not hass.services.has_service(DOMAIN, SERVICE_SET_FALLBACK_AGENT)


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
    delegated_text = "turn on the test light"
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

    assert observation is not None
    assert observation.kind is not None

    runtime = hass.data[DOMAIN][entry.entry_id][DATA_RUNTIME]
    assert runtime.registry_slot_values is not None

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

    entity_reg.async_update_entity(entity.entity_id, name="Updated Test Light")
    area_reg.async_update(area.id, name="Updated Test Area")
    floor_reg.async_update(floor.floor_id, name="Updated Test Floor")
    await hass.async_block_till_done()

    assert events_fired[str_entity_event] >= 1
    assert events_fired[str_area_event] >= 1
    assert events_fired[str_floor_event] >= 1

    assert _refresh_registry_slot_values(hass, runtime) is True

    assert await hass.config_entries.async_unload(entry.entry_id)
