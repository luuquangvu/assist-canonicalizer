"""Conversation platform for Assist Canonicalizer."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

from homeassistant.components import conversation
from homeassistant.components.conversation.agent_manager import async_get_agent
from homeassistant.components.conversation.const import HOME_ASSISTANT_AGENT
from homeassistant.components.conversation.models import (
    AbstractConversationAgent,
    ConversationInput,
    ConversationResult,
)
from homeassistant.const import MATCH_ALL
from homeassistant.helpers import intent

if TYPE_CHECKING:
    from homeassistant.components.conversation.default_agent import DefaultAgent
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
from .normalization import normalize_text
from .ranking import RankedCandidate, evaluate_confidence_gates
from .rehydration import get_wildcard_rehydration
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


class TaskDeltaWrapper:
    """Task-safe wrapper for chat log delta listener."""

    def __init__(self, original_listener: Callable[[Any, dict], None] | None) -> None:
        """Initialize the wrapper."""
        self.original_listener = original_listener
        self._listener_var: contextvars.ContextVar[Callable[[Any, dict], None] | None] = (
            contextvars.ContextVar("task_listener", default=None)
        )
        self._active_tasks: dict[int, int] = {}

    def __call__(self, log: Any, delta: dict) -> None:
        """Forward callbacks to task-local listener if active, else to original listener."""
        if (listener := self._listener_var.get()) is not None:
            listener(log, delta)
        elif self.original_listener is not None:
            self.original_listener(log, delta)

    def set_listener(self, listener: Callable[[Any, dict], None]) -> contextvars.Token:
        """Register the task-local listener."""
        with contextlib.suppress(RuntimeError):
            if (task := asyncio.current_task()) is not None:
                task_id = id(task)
                self._active_tasks[task_id] = self._active_tasks.get(task_id, 0) + 1
        return self._listener_var.set(listener)

    def reset_listener(self, token: contextvars.Token) -> None:
        """Deregister the task-local listener."""
        with contextlib.suppress(RuntimeError):
            if (task := asyncio.current_task()) is not None:
                task_id = id(task)
                if task_id in self._active_tasks:
                    self._active_tasks[task_id] -= 1
                    if self._active_tasks[task_id] <= 0:
                        del self._active_tasks[task_id]
        self._listener_var.reset(token)

    def has_active_listeners(self) -> bool:
        """Return whether there are active listeners."""
        return len(self._active_tasks) > 0


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
        chat_log = self._get_active_chat_log()
        old_len = len(chat_log.content) if chat_log is not None else None

        try:
            from homeassistant.components.assist_pipeline.const import DOMAIN as PIPELINE_DOMAIN
            from homeassistant.components.assist_pipeline.pipeline import async_get_pipeline

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

            pipeline = current_pipeline or async_get_pipeline(self.hass)
            if not pipeline or getattr(pipeline, "prefer_local_intents", False):
                return None

            shortcut_result = await self._delegate_with_capture(
                user_input.text,
                user_input,
                chat_log,
                old_len,
            )
            if shortcut_result is None:
                return None
        except Exception as err:
            _LOGGER.debug(
                "Assist pipeline shortcut path not available: %s",
                err,
            )
            return None

        self._runtime.update_diagnostics(clear_last_error=True)
        return shortcut_result

    async def _delegate_with_capture(
        self,
        text: str,
        user_input: ConversationInput,
        chat_log: Any,
        old_len: int | None,
        *,
        update_diagnostics_on_error: bool = False,
        return_error_result: bool = False,
    ) -> ConversationResult | None:
        """Delegate text and capture/play back deltas silently, restoring on failure."""
        deltas: list[dict] = []
        try:
            with self._capture_chat_log_deltas(chat_log) as captured:
                result = await self._delegate_text(
                    text,
                    user_input,
                    primary=True,
                )
                deltas = captured
            if self._result_has_error(result):
                self._restore_chat_log_content(chat_log, old_len)
                return result if return_error_result else None
        except Exception as err:
            if update_diagnostics_on_error:
                self._runtime.update_diagnostics(last_error=str(err))
            self._restore_chat_log_content(chat_log, old_len)
            raise

        try:
            self._play_back_deltas(chat_log, deltas)
        except Exception as err:
            _LOGGER.debug("Error during chat log delta playback: %s", err)
        return result

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
            selected, fallback_reason = evaluate_confidence_gates(
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
                last_fallback_reason=fallback_reason or FallbackReason.LOW_CONFIDENCE
            )
            return await self._delegate_raw_text(user_input)

        delegated_text = self._candidate_delegation_text(selected, user_input)
        execution_result = await self._async_execute_ranked_candidate(
            selected,
            user_input,
            delegated_text,
        )
        if execution_result is not None and not self._result_has_error(execution_result):
            self._runtime.update_diagnostics(clear_last_error=True)
            return execution_result

        error_code = self._result_error_code(execution_result)
        recovery = await self._async_ranked_recovery_candidate(
            ranked,
            selected,
            delegated_text,
            error_code,
            min_confidence,
            min_margin,
            user_input,
        )
        if recovery is not None:
            recovery_candidate, recovery_text = recovery
            recovery_result = await self._async_execute_ranked_candidate(
                recovery_candidate,
                user_input,
                recovery_text,
            )
            if recovery_result is not None and not self._result_has_error(recovery_result):
                self._runtime.update_diagnostics(clear_last_error=True)
                return recovery_result

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

    def _candidate_delegation_text(
        self,
        ranked_candidate: RankedCandidate,
        user_input: ConversationInput,
    ) -> str | None:
        """Return the rehydrated canonical text to execute for one candidate."""
        candidate = ranked_candidate.candidate
        rehydrated, _replacements = get_wildcard_rehydration(candidate, user_input.text)
        if candidate.has_wildcard and rehydrated == candidate.text:
            return None
        return rehydrated

    async def _async_execute_ranked_candidate(
        self,
        ranked_candidate: RankedCandidate,
        user_input: ConversationInput,
        delegated_text: str | None = None,
    ) -> ConversationResult | None:
        """Execute one canonical candidate through the primary HassIL agent.

        Wildcard placeholders (e.g. ``shopping_list_item``) in candidate text
        are rehydrated from the original query before delegation so that the
        downstream agent receives real free-text values.
        """
        chat_log = self._get_active_chat_log()
        old_len = len(chat_log.content) if chat_log is not None else None

        if delegated_text is None:
            delegated_text = self._candidate_delegation_text(ranked_candidate, user_input)
        if delegated_text is None:
            return None

        try:
            execution_result = await self._delegate_with_capture(
                delegated_text,
                user_input,
                chat_log,
                old_len,
                update_diagnostics_on_error=True,
                return_error_result=True,
            )
        except Exception:
            return None

        return execution_result

    async def _delegate_raw_text(self, user_input: ConversationInput) -> ConversationResult:
        """Delegate the original user text to the configured fallback path."""
        return await self._delegate_text(user_input.text, user_input, primary=False)

    async def _async_error_allows_candidate_recovery(
        self,
        error_code: str | None,
        delegated_text: str | None,
        user_input: ConversationInput,
    ) -> bool:
        """Return whether a primary error proves that one recovery is safe."""
        if error_code == intent.IntentResponseErrorCode.NO_INTENT_MATCH.value:
            return True
        if (
            error_code != intent.IntentResponseErrorCode.NO_VALID_TARGETS.value
            or delegated_text is None
        ):
            return False
        return await self._async_has_unmatched_entities(delegated_text, user_input)

    async def _async_ranked_recovery_candidate(
        self,
        ranked: Sequence[RankedCandidate],
        selected: RankedCandidate,
        delegated_text: str | None,
        error_code: str | None,
        min_confidence: float,
        min_margin: float,
        user_input: ConversationInput,
    ) -> tuple[RankedCandidate, str] | None:
        """Return at most one fully re-gated, error-compatible recovery candidate."""
        if not await self._async_error_allows_candidate_recovery(
            error_code, delegated_text, user_input
        ):
            return None

        delegated_key = normalize_text(delegated_text) if delegated_text is not None else None
        remaining: list[tuple[RankedCandidate, str]] = []
        for candidate in ranked:
            if candidate is selected:
                continue
            candidate_text = self._candidate_delegation_text(candidate, user_input)
            if candidate_text is None:
                continue
            if delegated_key is not None and normalize_text(candidate_text) == delegated_key:
                continue
            remaining.append((candidate, candidate_text))

        accepted, _reason = evaluate_confidence_gates(
            tuple(candidate for candidate, _text in remaining),
            min_confidence=min_confidence,
            min_margin=min_margin,
        )
        if accepted is None:
            return None
        return next(
            (
                (candidate, candidate_text)
                for candidate, candidate_text in remaining
                if candidate is accepted
            ),
            None,
        )

    async def _async_has_unmatched_entities(
        self,
        delegated_text: str,
        user_input: ConversationInput,
    ) -> bool:
        """Return whether side-effect-free HassIL recognition has unmatched entities."""
        try:
            recognition_input = copy(user_input)
            recognition_input.text = delegated_text
            recognition_input.agent_id = HOME_ASSISTANT_AGENT
            default_agent = cast(
                "DefaultAgent | None",
                async_get_agent(self.hass, HOME_ASSISTANT_AGENT),
            )
            if default_agent is None:
                return False
            recognition_result = await default_agent.async_recognize_intent(recognition_input)
        except Exception as err:
            _LOGGER.debug(
                "Unable to classify no_valid_targets recovery provenance: %s",
                err,
            )
            return False

        return bool(
            recognition_result is not None
            and getattr(recognition_result, "unmatched_entities", None)
        )

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

    @contextlib.contextmanager
    def _capture_chat_log_deltas(self, chat_log: Any) -> Iterator[list[dict]]:
        """Temporarily intercept and capture chat log delta listener callbacks."""
        captured_deltas: list[dict] = []
        if chat_log is None:
            yield captured_deltas
            return

        def task_listener(log: Any, delta: dict) -> None:
            captured_deltas.append(delta)

        current_listener = getattr(chat_log, "delta_listener", None)

        if isinstance(current_listener, TaskDeltaWrapper):
            wrapper = current_listener
            restore_listener = wrapper.original_listener
        else:
            wrapper = TaskDeltaWrapper(current_listener)
            chat_log.delta_listener = wrapper
            restore_listener = current_listener

        token = wrapper.set_listener(task_listener)
        try:
            yield captured_deltas
        finally:
            wrapper.reset_listener(token)
            # Only restore if no other task is actively intercepting and the
            # listener was not reassigned externally while we were running.
            if (
                not wrapper.has_active_listeners()
                and getattr(chat_log, "delta_listener", None) is wrapper
            ):
                chat_log.delta_listener = restore_listener

    def _play_back_deltas(self, chat_log: Any, deltas: list[dict]) -> None:
        """Play back captured deltas to the original listener."""
        if chat_log is not None and deltas:
            listener = getattr(chat_log, "delta_listener", None)
            while isinstance(listener, TaskDeltaWrapper):
                listener = listener.original_listener
            if listener is not None:
                for delta in deltas:
                    listener(chat_log, delta)

    def _get_active_chat_log(self) -> Any:
        """Return the active chat log for the current conversation context."""
        try:
            from homeassistant.components.conversation.chat_log import current_chat_log

            return current_chat_log.get(None)
        except (ImportError, RuntimeError, AttributeError, LookupError) as err:
            _LOGGER.debug("No conversation chat log available: %s", err)
        return None

    @staticmethod
    def _restore_chat_log_content(chat_log: Any, old_len: int | None) -> None:
        """Restore the chat log content to the pre-delegation length."""
        if chat_log is not None and old_len is not None:
            del chat_log.content[old_len:]

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
        return AssistCanonicalizerConversationEntity._result_error_code(result) is not None

    @staticmethod
    def _result_error_code(result: ConversationResult | None) -> str | None:
        """Return a normalized Home Assistant intent error code."""
        if result is None:
            return None
        response = getattr(result, "response", None)
        error_code = getattr(response, "error_code", None)
        if error_code is None:
            return None
        normalized = getattr(error_code, "value", error_code)
        return normalized if isinstance(normalized, str) else str(normalized)

    @staticmethod
    def _error_result(user_input: ConversationInput, message: str) -> ConversationResult:
        """Build a Home Assistant conversation error result."""
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, message)
        return ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
        )
