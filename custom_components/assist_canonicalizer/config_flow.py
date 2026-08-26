"""Config flow for Assist Canonicalizer."""

from __future__ import annotations

from collections.abc import Mapping

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.components.conversation.agent_manager import get_agent_manager
from homeassistant.components.conversation.const import DATA_COMPONENT, HOME_ASSISTANT_AGENT
from homeassistant.components.conversation.entity import ConversationEntity
from homeassistant.config_entries import ConfigFlowResult
from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
)

from .const import (
    DEFAULT_ENABLE_HOTWORD,
    DEFAULT_HOTWORD,
    DEFAULT_HOTWORD_MIN_CONFIDENCE,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DOMAIN,
    NAME,
    ConfigKey,
)
from .utils import normalize_hotword_list


class AssistCanonicalizerConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle the Assist Canonicalizer config flow."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, object] | None = None
    ) -> ConfigFlowResult:
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

    async def async_step_init(
        self, user_input: dict[str, object] | None = None
    ) -> ConfigFlowResult:
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
    hass: HomeAssistant,
    defaults: Mapping[str, object] | None = None,
    *,
    exclude_agent_id: str | None = None,
) -> vol.Schema:
    """Return config schema for fallback agent and confidence options."""
    defaults = defaults or {}
    fallback_default = str(defaults.get(ConfigKey.FALLBACK_AGENT_ID, ""))
    if not fallback_default and HOME_ASSISTANT_AGENT:
        fallback_default = HOME_ASSISTANT_AGENT
    raw_hotword = defaults.get(ConfigKey.HOTWORD, DEFAULT_HOTWORD)
    hotword_default = normalize_hotword_list(raw_hotword)
    return vol.Schema(
        {
            vol.Optional(
                ConfigKey.FALLBACK_AGENT_ID.value,
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
                ConfigKey.MIN_CONFIDENCE.value,
                default=defaults.get(ConfigKey.MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    mode=NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                ConfigKey.MIN_MARGIN.value,
                default=defaults.get(ConfigKey.MIN_MARGIN, DEFAULT_MIN_MARGIN),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    mode=NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                ConfigKey.ENABLE_HOTWORD.value,
                default=defaults.get(ConfigKey.ENABLE_HOTWORD, DEFAULT_ENABLE_HOTWORD),
            ): BooleanSelector(),
            vol.Optional(
                ConfigKey.HOTWORD.value,
                default=hotword_default,
            ): TextSelector(TextSelectorConfig(multiple=True)),
            vol.Optional(
                ConfigKey.HOTWORD_MIN_CONFIDENCE.value,
                default=defaults.get(
                    ConfigKey.HOTWORD_MIN_CONFIDENCE,
                    DEFAULT_HOTWORD_MIN_CONFIDENCE,
                ),
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
    hass: HomeAssistant,
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


def _entity_is_excluded(entity: object, agent_id: str, exclude_agent_id: str | None) -> bool:
    """Return whether a conversation entity belongs to the canonicalizer entry."""
    registry_entry = getattr(entity, "registry_entry", None)
    return exclude_agent_id is not None and (
        agent_id == exclude_agent_id
        or getattr(entity, "unique_id", None) == f"{exclude_agent_id}-conversation"
        or (
            registry_entry is not None
            and getattr(registry_entry, "config_entry_id", None) == exclude_agent_id
        )
    )


def _conversation_entity_agents(
    hass: HomeAssistant,
    exclude_agent_id: str | None,
) -> dict[str, str]:
    """Return eligible conversation entity IDs and display labels."""
    agents: dict[str, str] = {}
    hass_data = getattr(hass, "data", {})
    if not isinstance(hass_data, Mapping):
        return agents
    entity_component = hass_data.get(DATA_COMPONENT)
    if entity_component is None:
        return agents
    for entity in entity_component.entities:
        agent_id = getattr(entity, "entity_id", None)
        if (
            not isinstance(agent_id, str)
            or not agent_id
            or _entity_is_excluded(entity, agent_id, exclude_agent_id)
        ):
            continue
        states = getattr(hass, "states", None)
        get_state = getattr(states, "get", None)
        state = get_state(agent_id) if callable(get_state) else None
        agent_name = getattr(state, "name", None) or getattr(entity, "name", None)
        agents[agent_id] = str(agent_name) if agent_name else agent_id
    return agents


def _managed_fallback_agents(hass: HomeAssistant, exclude_agent_id: str | None) -> dict[str, str]:
    """Return eligible non-entity agents registered with the agent manager."""
    agents: dict[str, str] = {}
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
    return agents


def _available_fallback_agents(hass: HomeAssistant, exclude_agent_id: str | None) -> dict[str, str]:
    """Return fallback agent IDs and labels sorted by display name."""
    agents = _conversation_entity_agents(hass, exclude_agent_id)
    agents.update(_managed_fallback_agents(hass, exclude_agent_id))
    return dict(sorted(agents.items(), key=lambda item: item[1].casefold()))
