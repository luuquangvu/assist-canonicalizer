"""Conversation platform for Assist Canonicalizer."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from functools import partial
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

from homeassistant.helpers import area_registry, device_registry, entity_registry

from .const import (
    CONF_FALLBACK_AGENT_ID,
    DATA_RUNTIME,
    DEFAULT_MAX_CANDIDATES,
    DOMAIN,
    NAME,
    FallbackReason,
)
from .grammar_loader import rehydrate_wildcard_text
from .ranking import RankedCandidate, accepted_candidate, confidence_gate_rejection_reason
from .runtime import CanonicalizerRuntime
from .utils import (
    elapsed_ms,
    intent_context_from_area_name,
    normalize_language,
    resolve_entry_thresholds,
)

_LOGGER = logging.getLogger(__name__)


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
        started_at = time.monotonic()
        self._runtime.update_diagnostics(
            clear_last_fallback_reason=True,
            clear_last_error=True,
        )
        try:
            result = await self._async_process_with_runtime(user_input)
        except Exception as err:
            self._runtime.update_diagnostics(
                last_query_latency_ms=elapsed_ms(started_at),
                last_fallback_reason=FallbackReason.UNEXPECTED_EXCEPTION,
                last_error=str(err),
            )
            return self._error_result(user_input, str(err))
        self._runtime.update_diagnostics(last_query_latency_ms=elapsed_ms(started_at))
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

    async def _async_try_assist_pipeline_shortcut(
        self, user_input: ConversationInput
    ) -> ConversationResult | None:
        """Attempt to delegate text directly via the assist pipeline shortcut path if allowed."""
        try:
            from homeassistant.components.assist_pipeline.const import DOMAIN as PIPELINE_DOMAIN

            pipeline_data = self.hass.data.get(PIPELINE_DOMAIN)
            current_pipeline = None
            if (
                pipeline_data
                and user_input.context
                and (pipeline_runs := getattr(pipeline_data, "pipeline_runs", None))
            ):
                for runs_dict in getattr(pipeline_runs, "_pipeline_runs", {}).values():
                    for run in runs_dict.values():
                        if run.context and run.context.id == user_input.context.id:
                            current_pipeline = run.pipeline
                            break
                    if current_pipeline:
                        break

            from homeassistant.components.assist_pipeline.pipeline import async_get_pipeline

            pipeline = current_pipeline or async_get_pipeline(self.hass)
            if pipeline and not getattr(pipeline, "prefer_local_intents", False):
                shortcut_result = await self._delegate_text(
                    user_input.text,
                    user_input,
                    primary=True,
                )
                if not self._result_has_error(shortcut_result):
                    self._runtime.update_diagnostics(clear_last_error=True)
                    return shortcut_result
        except Exception as err:
            _LOGGER.debug(
                "Assist pipeline shortcut path not available: %s",
                err,
            )
        return None

    async def _async_process_with_runtime(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Rank indexed candidates and delegate to Home Assistant conversation agents.

        If the assist pipeline shortcut path is active and successful, we update
        diagnostics to clear the last error and return immediately. Note that early
        returns from this shortcut bypass the canonicalization index matching, meaning
        error diagnostics from failed canonicalizations will not be populated.
        """
        if shortcut_result := await self._async_try_assist_pipeline_shortcut(user_input):
            return shortcut_result

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
        if index is None:
            self._runtime.update_diagnostics(last_fallback_reason=FallbackReason.EMPTY_INDEX)
            return await self._delegate_raw_text(user_input)

        try:
            min_confidence, min_margin = resolve_entry_thresholds(self._entry)
            intent_context = self._intent_context_from_user_input(user_input)
            ranked = await self.hass.async_add_executor_job(
                partial(
                    self._runtime.rank_with_dynamic_candidates,
                    intent_context=intent_context,
                    min_confidence=min_confidence,
                    min_margin=min_margin,
                ),
                language,
                index,
                user_input.text,
                DEFAULT_MAX_CANDIDATES,
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
            self._runtime.update_diagnostics(
                last_fallback_reason=confidence_gate_rejection_reason(
                    ranked,
                    min_confidence=min_confidence,
                    min_margin=min_margin,
                )
            )
            return await self._delegate_raw_text(user_input)

        for ranked_candidate in self._ranked_validation_candidates(
            ranked,
            selected,
            min_confidence,
        ):
            validation_result = await self._async_validate_ranked_candidate(
                ranked_candidate,
                user_input,
            )
            if validation_result is not None:
                self._runtime.update_diagnostics(clear_last_error=True)
                return validation_result

        self._runtime.update_diagnostics(last_fallback_reason=FallbackReason.VALIDATION_FAILED)
        return await self._delegate_raw_text(user_input)

    def _intent_context_from_user_input(
        self, user_input: ConversationInput
    ) -> dict[str, Any] | None:
        """Return HassIL-style intent context from satellite or device area."""
        area = self._area_from_user_input(user_input)
        if area is None:
            return None
        area_name = getattr(area, "name", None)
        return intent_context_from_area_name(area_name)

    def _area_from_user_input(self, user_input: ConversationInput) -> Any | None:
        """Return the request area using Home Assistant registry metadata."""
        try:
            reg_entity = entity_registry.async_get(self.hass)
            reg_device = device_registry.async_get(self.hass)
            reg_area = area_registry.async_get(self.hass)
        except (AttributeError, RuntimeError):
            return None

        area_id: str | None = None
        device_id = user_input.device_id
        satellite_id = getattr(user_input, "satellite_id", None)

        if (
            satellite_id is not None
            and (entity_entry := reg_entity.async_get(satellite_id)) is not None
        ):
            area_id = getattr(entity_entry, "area_id", None)
            if satellite_device_id := getattr(entity_entry, "device_id", None):
                device_id = satellite_device_id

        if (
            area_id is None
            and device_id is not None
            and (device_entry := reg_device.async_get(device_id)) is not None
        ):
            area_id = getattr(device_entry, "area_id", None)

        return None if area_id is None else reg_area.async_get_area(area_id)

    async def _async_validate_ranked_candidate(
        self,
        ranked_candidate: RankedCandidate,
        user_input: ConversationInput,
    ) -> ConversationResult | None:
        """Validate one accepted canonical candidate through the primary HassIL agent.

        Wildcard placeholders (e.g. ``shopping_list_item``) in candidate text
        are rehydrated from the original query before delegation so that the
        downstream agent receives real free-text values.
        """
        candidate = ranked_candidate.candidate
        candidate_text = candidate.text
        rehydrated = rehydrate_wildcard_text(candidate_text, user_input.text, user_input.language)
        if candidate.has_wildcard and rehydrated == candidate_text:
            return None
        try:
            validation_result = await self._delegate_text(
                rehydrated,
                user_input,
                primary=True,
            )
        except Exception as err:
            self._runtime.update_diagnostics(last_error=str(err))
            return None
        return None if self._result_has_error(validation_result) else validation_result

    async def _delegate_raw_text(self, user_input: ConversationInput) -> ConversationResult:
        """Delegate the original user text to the configured fallback path."""
        return await self._delegate_text(user_input.text, user_input, primary=False)

    @staticmethod
    def _ranked_validation_candidates(
        ranked: Sequence[RankedCandidate],
        selected: RankedCandidate,
        min_confidence: float,
    ) -> tuple[RankedCandidate, ...]:
        """Return ranked candidates to validate, preferring the selected candidate."""
        candidates = [selected]
        candidates.extend(
            candidate
            for candidate in ranked
            if candidate is not selected and candidate.scores.final_score >= min_confidence
        )
        return tuple(candidates)

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
    def _error_result(user_input: ConversationInput, message: str) -> ConversationResult:
        """Build a Home Assistant conversation error result."""
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, message)
        return ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
        )
