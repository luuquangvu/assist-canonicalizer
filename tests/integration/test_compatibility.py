"""Functional compatibility contract for supported Home Assistant versions."""

import pytest
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.assist_canonicalizer.const import (
    ATTR_CANDIDATE_COUNT,
    ATTR_LANGUAGE,
    CONF_MIN_CONFIDENCE,
    CONF_MIN_MARGIN,
    DOMAIN,
    SERVICE_DIAGNOSTICS,
    SERVICE_REBUILD_INDEX,
)

pytestmark = pytest.mark.compatibility


@pytest.mark.asyncio
async def test_home_assistant_functional_contract(
    hass: HomeAssistant,
) -> None:
    """Load, index, process, diagnose, and unload through Home Assistant APIs."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, conversation.DOMAIN, {})

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
    assert hass.services.has_service(DOMAIN, SERVICE_DIAGNOSTICS)

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

    result = await conversation.async_converse(
        hass,
        "compatibility smoke phrase with no supported intent",
        "compatibility-smoke",
        Context(),
        language="en",
        agent_id=entry.entry_id,
    )
    assert result.response is not None

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

    assert await hass.config_entries.async_unload(entry.entry_id)
    assert conversation.async_get_agent_info(hass, entry.entry_id) is None
    assert not hass.services.has_service(DOMAIN, SERVICE_REBUILD_INDEX)
    assert not hass.services.has_service(DOMAIN, SERVICE_DIAGNOSTICS)
