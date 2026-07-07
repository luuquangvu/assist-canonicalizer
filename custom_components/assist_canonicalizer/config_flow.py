"""Config flow for Assist Canonicalizer."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.components.conversation.agent_manager import get_agent_manager
from homeassistant.components.conversation.const import DATA_COMPONENT, HOME_ASSISTANT_AGENT
from homeassistant.components.conversation.entity import ConversationEntity
from homeassistant.config_entries import ConfigFlowResult
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .const import (
    CONF_FALLBACK_AGENT_ID,
    CONF_MIN_CONFIDENCE,
    CONF_MIN_MARGIN,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DOMAIN,
    NAME,
)


class AssistCanonicalizerConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle the Assist Canonicalizer config flow."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Create a single Assist Canonicalizer config entry."""
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")
        if user_input is None:
            return self.async_show_form(step_id="user", data_schema=_config_schema(self.hass))
        return self.async_create_entry(title=NAME, data=user_input)

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Return the options flow."""
        return AssistCanonicalizerOptionsFlow(config_entry)


class AssistCanonicalizerOptionsFlow(config_entries.OptionsFlow):
    """Handle Assist Canonicalizer options."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Manage Assist Canonicalizer options."""
        current = {**self._config_entry.data, **self._config_entry.options}
        if user_input is None:
            return self.async_show_form(
                step_id="init",
                data_schema=_config_schema(
                    self.hass,
                    current,
                    exclude_agent_id=self._config_entry.entry_id,
                ),
            )
        return self.async_create_entry(title="", data=user_input)


def _config_schema(
    hass: Any,
    defaults: dict[str, Any] | None = None,
    *,
    exclude_agent_id: str | None = None,
) -> vol.Schema:
    """Return config schema for fallback agent and confidence options."""
    defaults = defaults or {}
    fallback_default = str(defaults.get(CONF_FALLBACK_AGENT_ID, ""))
    if not fallback_default and HOME_ASSISTANT_AGENT:
        fallback_default = HOME_ASSISTANT_AGENT
    return vol.Schema(
        {
            vol.Optional(
                CONF_FALLBACK_AGENT_ID,
                default=fallback_default,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        {"value": k, "label": v}
                        for k, v in _fallback_agent_choices(
                            hass, fallback_default, exclude_agent_id
                        ).items()
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_MIN_CONFIDENCE,
                default=defaults.get(CONF_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    mode=NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_MIN_MARGIN,
                default=defaults.get(CONF_MIN_MARGIN, DEFAULT_MIN_MARGIN),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    mode=NumberSelectorMode.BOX,
                )
            ),
        }
    )


def _fallback_agent_choices(
    hass: Any,
    current_agent_id: str,
    exclude_agent_id: str | None,
) -> dict[str, str]:
    """Return fallback agent IDs mapped to display labels for config forms."""
    choices: dict[str, str] = {}
    choices |= _available_fallback_agents(hass, exclude_agent_id)
    if (
        current_agent_id
        and current_agent_id != exclude_agent_id
        and current_agent_id not in choices
    ):
        choices[current_agent_id] = f"Unavailable agent ({current_agent_id})"
    return choices


def _available_fallback_agents(hass: Any, exclude_agent_id: str | None) -> dict[str, str]:
    """Return conversation agent IDs and labels that can be used as fallback targets."""
    agents = {}
    entity_component = hass.data.get(DATA_COMPONENT)
    if entity_component is not None:
        for entity in entity_component.entities:
            agent_id = getattr(entity, "entity_id", None)
            if not isinstance(agent_id, str) or not agent_id:
                continue
            if (
                agent_id == exclude_agent_id
                or getattr(entity, "unique_id", None) == f"{exclude_agent_id}-conversation"
                or (
                    hasattr(entity, "registry_entry")
                    and entity.registry_entry is not None
                    and getattr(entity.registry_entry, "config_entry_id", None) == exclude_agent_id
                )
            ):
                continue
            agent_name = None
            if hasattr(hass, "states") and (state := hass.states.get(agent_id)):
                agent_name = state.name
            if not agent_name:
                agent_name = getattr(entity, "name", None)
            agents[agent_id] = str(agent_name) if agent_name else agent_id

    manager = get_agent_manager(hass)
    for agent_info in manager.async_get_agent_info():
        agent_id = getattr(agent_info, "id", None)
        if not isinstance(agent_id, str) or not agent_id or agent_id == exclude_agent_id:
            continue
        if hasattr(manager, "async_get_agent"):
            agent = manager.async_get_agent(agent_id)
            if agent is not None and isinstance(agent, ConversationEntity):
                continue
        agent_name = getattr(agent_info, "name", None)
        agents[agent_id] = str(agent_name) if agent_name else agent_id
    return dict(sorted(agents.items(), key=lambda item: item[1].casefold()))
