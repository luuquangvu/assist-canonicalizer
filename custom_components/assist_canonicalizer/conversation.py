"""Conversation platform for Assist Canonicalizer."""

from __future__ import annotations

from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

from homeassistant.components import conversation
from homeassistant.components.conversation.const import HOME_ASSISTANT_AGENT
from homeassistant.components.conversation.models import (
    AbstractConversationAgent,
    ConversationInput,
    ConversationResult,
)
from homeassistant.const import MATCH_ALL
from homeassistant.helpers import intent

if TYPE_CHECKING:
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
else:
    try:
        from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
    except ImportError:
        AddConfigEntryEntitiesCallback = Any

from .const import (
    CONF_FALLBACK_AGENT_ID,
    CONF_MIN_CONFIDENCE,
    CONF_MIN_MARGIN,
    DATA_RUNTIME,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DEFAULT_VALIDATION_CANDIDATES,
    DOMAIN,
    NAME,
    FallbackReason,
)
from .ranking import accepted_candidate
from .runtime import CanonicalizerRuntime, normalize_language


async def async_setup_entry(
    hass: Any,
    config_entry: Any,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Assist Canonicalizer conversation entity."""
    runtime = hass.data[DOMAIN][config_entry.entry_id][DATA_RUNTIME]
    if not isinstance(runtime, CanonicalizerRuntime):
        msg = "Assist Canonicalizer runtime is not loaded"
        raise RuntimeError(msg)
    async_add_entities([AssistCanonicalizerConversationEntity(config_entry, runtime)])


class AssistCanonicalizerConversationEntity(
    conversation.ConversationEntity,
    AbstractConversationAgent,
):
    """Home Assistant conversation entity that canonicalizes before delegating."""

    _attr_has_entity_name = True
    _attr_name = NAME
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: Any, runtime: CanonicalizerRuntime) -> None:
        """Initialize the conversation entity."""
        super().__init__()
        self._entry = entry
        self._runtime = runtime
        self._attr_unique_id = f"{entry.entry_id}-conversation"

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return languages accepted from the active Assist pipeline."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """Register this entity as a Home Assistant conversation agent."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self._entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister this entity from Home Assistant conversation agents."""
        conversation.async_unset_agent(self.hass, self._entry)
        await super().async_will_remove_from_hass()

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a conversation request."""
        started_at = monotonic()
        self._runtime.update_diagnostics(
            clear_last_fallback_reason=True,
            clear_last_error=True,
        )
        try:
            result = await self._async_process_with_runtime(user_input)
        except Exception as err:
            self._runtime.update_diagnostics(
                last_query_latency_ms=self._elapsed_ms(started_at),
                last_fallback_reason=FallbackReason.UNEXPECTED_EXCEPTION,
                last_error=str(err),
            )
            return self._error_result(user_input, str(err))
        self._runtime.update_diagnostics(last_query_latency_ms=self._elapsed_ms(started_at))
        return result

    async def async_prepare(self, language: str | None = None) -> None:
        """Prepare the agent for a language."""
        if language:
            language = normalize_language(language)
            index = self._runtime.get_index(language)
            if index is None:
                index = await self._runtime.async_load_index_from_store(self.hass, language)
                if index is None:
                    await self._runtime.async_rebuild_index(self.hass, language)

    async def async_reload(self, language: str | None = None) -> None:
        """Reload cached indexes for a language."""
        await self._runtime.async_clear_index(
            self.hass,
            normalize_language(language) if language else None,
        )

    async def _async_process_with_runtime(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Rank indexed candidates and delegate to Home Assistant conversation agents."""
        language = normalize_language(user_input.language)
        index = self._runtime.get_index(language)
        if index is None:
            index = await self._runtime.async_load_index_from_store(
                self.hass,
                language,
            )
            if index is None:
                index = await self._runtime.async_rebuild_index(
                    self.hass,
                    language,
                )
        if index is None or index.candidate_count == 0:
            self._runtime.update_diagnostics(last_fallback_reason=FallbackReason.EMPTY_INDEX)
            return await self._delegate_raw_text(user_input)

        try:
            ranked = await self.hass.async_add_executor_job(
                self._runtime.rank_with_dynamic_candidates,
                language,
                index,
                user_input.text,
                DEFAULT_MAX_CANDIDATES,
            )
            options = self._entry.options or {}
            data = self._entry.data or {}
            min_confidence = options.get(
                CONF_MIN_CONFIDENCE,
                data.get(CONF_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE),
            )
            min_margin = options.get(
                CONF_MIN_MARGIN,
                data.get(CONF_MIN_MARGIN, DEFAULT_MIN_MARGIN),
            )
            selected = accepted_candidate(
                ranked,
                min_confidence=min_confidence,
                min_margin=min_margin,
            )
        except Exception as err:
            self._runtime.update_diagnostics(
                last_fallback_reason=FallbackReason.RANKING_FAILED,
                last_error=str(err),
            )
            return await self._delegate_raw_text(user_input)

        if selected is None:
            self._runtime.update_diagnostics(last_fallback_reason=FallbackReason.RANKING_FAILED)
            return await self._delegate_raw_text(user_input)

        validation_result = await self._async_validate_ranked_candidates(ranked, user_input)
        if validation_result is not None:
            return validation_result

        self._runtime.update_diagnostics(last_fallback_reason=FallbackReason.VALIDATION_FAILED)
        return await self._delegate_raw_text(user_input)

    async def _async_validate_ranked_candidates(
        self,
        ranked: tuple[Any, ...],
        user_input: ConversationInput,
    ) -> ConversationResult | None:
        """Validate ranked canonical candidates through the primary Hassil agent."""
        for ranked_candidate in ranked[:DEFAULT_VALIDATION_CANDIDATES]:
            validation_result = await self._delegate_text(
                ranked_candidate.candidate.text,
                user_input,
                primary=True,
            )
            if not self._result_has_error(validation_result):
                return validation_result
        return None

    async def _delegate_raw_text(self, user_input: ConversationInput) -> ConversationResult:
        """Delegate the original user text to the configured fallback path."""
        return await self._delegate_text(user_input.text, user_input, primary=False)

    async def _delegate_text(
        self,
        text: str,
        user_input: ConversationInput,
        *,
        primary: bool,
    ) -> ConversationResult:
        """Delegate text to a Home Assistant conversation agent."""
        agent_id = (
            HOME_ASSISTANT_AGENT if primary else self._fallback_agent_id(HOME_ASSISTANT_AGENT)
        )
        return await conversation.async_converse(
            self.hass,
            text,
            user_input.conversation_id,
            user_input.context,
            language=user_input.language,
            agent_id=agent_id,
            device_id=user_input.device_id,
            satellite_id=getattr(user_input, "satellite_id", None),
            extra_system_prompt=getattr(user_input, "extra_system_prompt", None),
        )

    def _fallback_agent_id(self, default_agent_id: str) -> str:
        """Return a configured fallback agent without allowing self-forwarding."""
        options = getattr(self._entry, "options", {}) or {}
        data = getattr(self._entry, "data", {}) or {}
        configured = options.get(CONF_FALLBACK_AGENT_ID) or data.get(CONF_FALLBACK_AGENT_ID)
        if not isinstance(configured, str) or configured == self._entry.entry_id:
            return default_agent_id
        return configured

    @staticmethod
    def _result_has_error(result: ConversationResult) -> bool:
        """Return whether a conversation result contains an intent error."""
        response = getattr(result, "response", None)
        return getattr(response, "error_code", None) is not None

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        """Return elapsed milliseconds from a monotonic timestamp."""
        return round((monotonic() - started_at) * 1000, 3)

    @staticmethod
    def _error_result(user_input: ConversationInput, message: str) -> ConversationResult:
        """Build a Home Assistant conversation error result."""
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, message)
        return ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
        )
