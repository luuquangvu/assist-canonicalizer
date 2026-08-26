"""Conversation platform for Assist Canonicalizer."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import inspect
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from homeassistant.components import conversation
from homeassistant.components.conversation.const import HOME_ASSISTANT_AGENT
from homeassistant.components.conversation.models import (
    AbstractConversationAgent,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent

if TYPE_CHECKING:
    from homeassistant.components.conversation.chat_log import ChatLog
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from homeassistant.helpers import area_registry, device_registry, entity_registry

from .const import (
    CONF_FALLBACK_AGENT_ID,
    CONVERSATION_INPUT_DEVICE_ID_FIELD,
    CONVERSATION_INPUT_OPTIONAL_FIELDS,
    CONVERSATION_INPUT_SATELLITE_ID_FIELD,
    DATA_RUNTIME,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_PREFLIGHT_ATTEMPTS,
    DOMAIN,
    NAME,
    FallbackReason,
)
from .indexer import CanonicalIndex
from .normalization import normalize_text
from .ranking import (
    ConfidenceGateDecision,
    RankedCandidate,
    evaluate_confidence_gates,
    match_hotword_prefix,
)
from .recognition import (
    RecognitionObservation,
    async_observe_delegated_text,
    metadata_matches_observation,
)
from .rehydration import get_wildcard_rehydration
from .runtime import CanonicalizerRuntime
from .utils import (
    elapsed_ms,
    intent_context_from_area_name,
    normalize_language,
    resolve_entry_hotword_options,
    resolve_entry_thresholds,
)

_LOGGER = logging.getLogger(__name__)

_ASYNC_CONVERSE_PARAMETERS = frozenset(inspect.signature(conversation.async_converse).parameters)
type ChatLogDelta = dict[str, object]


@runtime_checkable
class ChatLogDeltaListener(Protocol):
    """Callback shape used by Home Assistant chat-log delta listeners."""

    def __call__(self, log: object, delta: ChatLogDelta) -> object:
        """Consume one chat-log delta."""
        ...


@dataclass(frozen=True, slots=True)
class _PreflightSelection:
    """One live-valid candidate selected after bounded preflight."""

    ranked_candidate: RankedCandidate
    delegated_text: str
    observation: RecognitionObservation
    decision: ConfidenceGateDecision
    attempt_count: int
    attempted_text_keys: frozenset[str]


@dataclass(frozen=True, slots=True)
class _RankedRequest:
    """Ranked candidates and thresholds for one conversation request."""

    ranked: tuple[RankedCandidate, ...]
    decision: ConfidenceGateDecision
    min_confidence: float
    min_margin: float


@dataclass(frozen=True, slots=True)
class _PreflightAttempt:
    """One distinct delegated text selected for live recognition."""

    ranked_candidate: RankedCandidate
    delegated_text: str
    delegated_key: str
    decision: ConfidenceGateDecision


@runtime_checkable
class _ChatLogWithDeltaListener(Protocol):
    """Protocol for chat log objects exposing a delta listener."""

    delta_listener: ChatLogDeltaListener | TaskDeltaWrapper | None


@dataclass(slots=True)
class _PreflightState:
    """Mutable state for the bounded preflight search."""

    remaining: list[RankedCandidate]
    attempted_texts: set[str]
    intent_divergent_fallback: _PreflightSelection | None
    pending_decision: ConfidenceGateDecision | None


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
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

    def __init__(self, original_listener: ChatLogDeltaListener | None = None) -> None:
        """Initialize the wrapper."""
        self.original_listener = original_listener
        self._listener_var: contextvars.ContextVar[ChatLogDeltaListener | None] = (
            contextvars.ContextVar("task_listener", default=None)
        )
        self._active_tasks: dict[int, int] = {}

    def __call__(self, log: object, delta: ChatLogDelta) -> None:
        """Forward callbacks to task-local listener if active, else to original listener."""
        if (listener := self._listener_var.get()) is not None:
            listener(log, delta)
        elif self.original_listener is not None:
            self.original_listener(log, delta)

    def set_listener(
        self, listener: ChatLogDeltaListener
    ) -> contextvars.Token[ChatLogDeltaListener | None]:
        """Register the task-local listener."""
        with contextlib.suppress(RuntimeError):
            if (task := asyncio.current_task()) is not None:
                task_id = id(task)
                self._active_tasks[task_id] = self._active_tasks.get(task_id, 0) + 1
        return self._listener_var.set(listener)

    def reset_listener(self, token: contextvars.Token[ChatLogDeltaListener | None]) -> None:
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
    """Protect HassIL-supported input, then canonicalize unsupported Assist requests."""

    _attr_has_entity_name = True
    _attr_name = NAME
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: ConfigEntry, runtime: CanonicalizerRuntime) -> None:
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
        """Process a request through the lifecycle provided by the installed HA."""
        if hasattr(conversation.ConversationEntity, "_async_handle_message"):
            return await super().async_process(user_input)
        return await self._async_process_request(user_input)

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Process a request inside the modern Home Assistant chat-log lifecycle."""
        return await self._async_process_request(user_input)

    async def _async_process_request(
        self,
        user_input: ConversationInput,
    ) -> ConversationResult:
        """Process one request after Home Assistant establishes its lifecycle context."""
        started_at = time.monotonic()
        request_id = user_input.conversation_id or getattr(user_input.context, "id", None)
        self._runtime.update_diagnostics(
            clear_last_fallback_reason=True,
            clear_last_error=True,
            clear_request_trace=True,
            last_request_id=request_id,
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

    def _is_hotword_matched(self, user_input: ConversationInput) -> bool:
        """Return whether user input starts with a configured hotword with high confidence."""
        enable_hotword, hotword, min_confidence = resolve_entry_hotword_options(self._entry)
        if not enable_hotword or not hotword or not user_input.text:
            return False
        language = normalize_language(user_input.language) if user_input.language else None
        matched, _score, _matched_hw = match_hotword_prefix(
            user_input.text,
            hotword,
            min_confidence=min_confidence,
            language=language,
        )
        return matched

    async def _async_try_assist_pipeline_shortcut(
        self, user_input: ConversationInput
    ) -> ConversationResult | None:
        """Try HassIL when the active Assist pipeline has not already done so.

        With ``prefer_local_intents`` enabled, Home Assistant tries local intents
        before falling back to this conversation agent, so repeating that work here
        would duplicate the pipeline's HassIL-first routing. When the option is
        disabled, the pipeline delegates directly to this entity and this shortcut
        supplies the equivalent HassIL-first behavior.

        Direct calls without an active pipeline run skip this pipeline-specific
        shortcut. A shortcut error or rejected HassIL result is non-terminal:
        chat-log state is restored, then canonical matching and the configured
        fallback agent remain available.
        """
        chat_log = self._get_active_chat_log()
        old_len = len(chat_log.content) if chat_log is not None else None

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

            if current_pipeline is None or getattr(current_pipeline, "prefer_local_intents", False):
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
        chat_log: ChatLog | None,
        old_len: int | None,
        *,
        update_diagnostics_on_error: bool = False,
        return_error_result: bool = False,
    ) -> ConversationResult | None:
        """Delegate text and capture/play back deltas silently, restoring on failure."""
        deltas: list[ChatLogDelta] = []
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
        """Apply pipeline-aware HassIL protection, canonicalization, and fallback.

        Home Assistant owns the HassIL-first boundary when ``prefer_local_intents`` is
        enabled; otherwise the shortcut supplies it before ranking. Direct entity
        calls have no pipeline-level protection and intentionally skip the shortcut.
        Every unsuccessful local path still delegates the original text to the
        configured fallback agent.
        """
        if self._is_hotword_matched(user_input):
            self._runtime.update_diagnostics(
                last_fallback_reason=FallbackReason.HOTWORD_MATCHED,
                execution_result="hotword_fallback",
            )
            return await self._delegate_raw_text(user_input)

        if shortcut_result := await self._async_try_assist_pipeline_shortcut(user_input):
            return shortcut_result

        language = normalize_language(user_input.language)
        index = await self._async_request_index(language)
        if index is None:
            return await self._delegate_raw_text(user_input)

        ranked_request = await self._async_rank_request(user_input, language, index)
        if ranked_request is None:
            return await self._delegate_raw_text(user_input)

        if ranked_request.decision.accepted_candidate is None:
            self._runtime.update_diagnostics(
                last_fallback_reason=(
                    ranked_request.decision.rejection_reason or FallbackReason.LOW_CONFIDENCE
                ),
                execution_result="raw_fallback",
            )
            return await self._delegate_raw_text(user_input)

        return await self._async_execute_ranked_request(ranked_request, user_input)

    async def _async_request_index(self, language: str) -> CanonicalIndex | None:
        """Return the request index and record acquisition failures."""
        index = self._runtime.get_index(language)
        try:
            index = await self._async_load_or_rebuild_index(language, index)
        except Exception as err:
            self._runtime.update_diagnostics(
                last_fallback_reason=FallbackReason.EMPTY_INDEX,
                last_error=str(err),
            )
            return None
        if index is None:
            self._runtime.update_diagnostics(last_fallback_reason=FallbackReason.EMPTY_INDEX)
        return index

    async def _async_rank_request(
        self,
        user_input: ConversationInput,
        language: str,
        index: CanonicalIndex,
    ) -> _RankedRequest | None:
        """Rank one request and record ranking failures."""
        try:
            min_confidence, min_margin = resolve_entry_thresholds(self._entry)
            ranked, decision = await self._async_rank_user_input(
                user_input,
                language,
                index,
                min_confidence,
                min_margin,
            )
        except Exception as err:
            self._runtime.update_diagnostics(
                last_fallback_reason=FallbackReason.RANKING_FAILED,
                last_error=str(err),
            )
            return None
        return _RankedRequest(ranked, decision, min_confidence, min_margin)

    async def _async_execute_ranked_request(
        self,
        ranked_request: _RankedRequest,
        user_input: ConversationInput,
    ) -> ConversationResult:
        """Execute an accepted request and preserve raw fallback on unexpected errors."""
        try:
            return await self._async_execute_accepted_candidates(
                ranked_request.ranked,
                ranked_request.decision,
                user_input,
                ranked_request.min_confidence,
                ranked_request.min_margin,
            )
        except Exception as err:
            _LOGGER.warning(
                "Unexpected error while executing canonical candidates; "
                "delegating original text to the fallback agent: %s",
                err,
            )
            self._runtime.update_diagnostics(
                last_fallback_reason=FallbackReason.UNEXPECTED_EXCEPTION,
                last_error=str(err),
            )
            return await self._delegate_raw_text(user_input)

    async def _async_load_or_rebuild_index(
        self, language: str, index: CanonicalIndex | None
    ) -> CanonicalIndex | None:
        """Load a persisted language index or rebuild it when none is cached."""
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
        return index

    async def _async_rank_user_input(
        self,
        user_input: ConversationInput,
        language: str,
        index: CanonicalIndex,
        min_confidence: float,
        min_margin: float,
    ) -> tuple[tuple[RankedCandidate, ...], ConfidenceGateDecision]:
        """Rank candidates on the executor and publish the gate decision."""
        intent_context = self._intent_context_from_user_input(user_input)
        ranked, decision = await self.hass.async_add_executor_job(
            partial(
                self._runtime.rank_and_evaluate,
                intent_context=intent_context,
                min_confidence=min_confidence,
                min_margin=min_margin,
            ),
            language,
            index,
            user_input.text,
            DEFAULT_MAX_CANDIDATES,
        )
        self._runtime.update_diagnostics(confidence_gate=decision.as_json_dict())
        return ranked, decision

    async def _async_execute_accepted_candidates(
        self,
        ranked: tuple[RankedCandidate, ...],
        decision: ConfidenceGateDecision,
        user_input: ConversationInput,
        min_confidence: float,
        min_margin: float,
    ) -> ConversationResult:
        """Preflight, execute, and recover gate-accepted candidates."""
        delegation_texts: dict[RankedCandidate, str | None] = {}
        preflight = await self._async_preflight_ranked_candidates(
            ranked,
            user_input,
            min_confidence,
            min_margin,
            initial_decision=decision,
            delegation_texts=delegation_texts,
        )
        if preflight is None:
            self._runtime.update_diagnostics(
                last_fallback_reason=FallbackReason.VALIDATION_FAILED,
                execution_result="raw_fallback",
            )
            return await self._delegate_raw_text(user_input)

        selected = preflight.ranked_candidate
        delegated_text = preflight.delegated_text
        execution_result = await self._async_execute_ranked_candidate(
            selected,
            user_input,
            delegated_text,
        )
        if execution_result is not None and not self._result_has_error(execution_result):
            self._runtime.update_diagnostics(
                clear_last_error=True,
                execution_result="success",
            )
            return execution_result

        error_code = self._result_error_code(execution_result)
        recovery_result = await self._async_attempt_execution_recovery(
            ranked,
            selected,
            delegated_text,
            error_code,
            min_confidence,
            min_margin,
            user_input,
            preflight,
            delegation_texts,
        )
        if recovery_result is not None:
            return recovery_result

        self._runtime.update_diagnostics(
            last_fallback_reason=FallbackReason.VALIDATION_FAILED,
            execution_result=error_code or "candidate_execution_failed",
        )
        return await self._delegate_raw_text(user_input)

    async def _async_attempt_execution_recovery(
        self,
        ranked: tuple[RankedCandidate, ...],
        selected: RankedCandidate,
        delegated_text: str | None,
        error_code: str | None,
        min_confidence: float,
        min_margin: float,
        user_input: ConversationInput,
        preflight: _PreflightSelection,
        delegation_texts: dict[RankedCandidate, str | None],
    ) -> ConversationResult | None:
        """Execute one re-gated recovery candidate, returning only a success result."""
        recovery = await self._async_ranked_recovery_candidate(
            ranked,
            selected,
            delegated_text,
            error_code,
            min_confidence,
            min_margin,
            user_input,
            excluded_delegated_keys=preflight.attempted_text_keys,
            delegation_texts=delegation_texts,
        )
        if recovery is None:
            return None
        recovery_candidate, recovery_text, recovery_observation = recovery
        self._record_recognition_diagnostics(
            recovery_candidate,
            recovery_text,
            recovery_observation,
            preflight.attempt_count + 1,
        )
        self._runtime.update_diagnostics(recovery_used=True)
        recovery_result = await self._async_execute_ranked_candidate(
            recovery_candidate,
            user_input,
            recovery_text,
        )
        if recovery_result is not None and not self._result_has_error(recovery_result):
            self._runtime.update_diagnostics(
                clear_last_error=True,
                execution_result="success_after_execution_recovery",
            )
            return recovery_result
        return None

    def _intent_context_from_user_input(
        self, user_input: ConversationInput
    ) -> dict[str, dict[str, str]] | None:
        """Return HassIL-style intent context from satellite or device area."""
        area = self._area_from_user_input(user_input)
        if area is None:
            return None
        area_name = getattr(area, "name", None)
        return intent_context_from_area_name(area_name)

    def _area_from_user_input(
        self, user_input: ConversationInput
    ) -> area_registry.AreaEntry | None:
        """Return the request area using Home Assistant registry metadata."""
        try:
            reg_entity = entity_registry.async_get(self.hass)
            reg_device = device_registry.async_get(self.hass)
            reg_area = area_registry.async_get(self.hass)
        except (AttributeError, RuntimeError):
            return None

        area_id: str | None = None
        device_id = user_input.device_id
        satellite_id = getattr(user_input, CONVERSATION_INPUT_SATELLITE_ID_FIELD, None)

        if (
            satellite_id is not None
            and (entity_entry := reg_entity.async_get(satellite_id)) is not None
        ):
            area_id = getattr(entity_entry, "area_id", None)
            if satellite_device_id := getattr(
                entity_entry,
                CONVERSATION_INPUT_DEVICE_ID_FIELD,
                None,
            ):
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
        delegation_texts: dict[RankedCandidate, str | None] | None = None,
    ) -> str | None:
        """Return the rehydrated canonical text to execute for one candidate.

        ``delegation_texts`` memoizes results per request so preflight and
        recovery sweeps do not repeat wildcard stem alignment for the same
        ranked candidate.
        """
        if delegation_texts is not None and ranked_candidate in delegation_texts:
            return delegation_texts[ranked_candidate]
        candidate = ranked_candidate.candidate
        rehydrated, _replacements = get_wildcard_rehydration(candidate, user_input.text)
        result = None if candidate.has_wildcard and rehydrated == candidate.text else rehydrated
        if delegation_texts is not None:
            delegation_texts[ranked_candidate] = result
        return result

    def _next_preflight_attempt(
        self,
        state: _PreflightState,
        user_input: ConversationInput,
        min_confidence: float,
        min_margin: float,
        delegation_texts: dict[RankedCandidate, str | None] | None,
    ) -> _PreflightAttempt | None:
        """Return the next distinct gate-accepted text within the attempt budget."""
        while state.remaining and len(state.attempted_texts) < DEFAULT_MAX_PREFLIGHT_ATTEMPTS:
            decision = state.pending_decision
            state.pending_decision = None
            if decision is None:
                decision = self._evaluate_preflight_gate(
                    state.remaining,
                    user_input,
                    min_confidence,
                    min_margin,
                )
            selected = decision.accepted_candidate
            if selected is None:
                return None
            delegated_text = self._candidate_delegation_text(
                selected,
                user_input,
                delegation_texts,
            )
            if delegated_text is None:
                state.remaining = [
                    candidate for candidate in state.remaining if candidate is not selected
                ]
                continue
            delegated_key = normalize_text(delegated_text)
            if not delegated_key or delegated_key in state.attempted_texts:
                state.remaining = self._remaining_distinct_candidates(
                    state.remaining,
                    user_input,
                    delegated_key,
                    delegation_texts,
                )
                continue
            state.attempted_texts.add(delegated_key)
            return _PreflightAttempt(
                selected,
                delegated_text,
                delegated_key,
                decision,
            )
        return None

    async def _async_preflight_ranked_candidates(
        self,
        ranked: Sequence[RankedCandidate],
        user_input: ConversationInput,
        min_confidence: float,
        min_margin: float,
        *,
        initial_decision: ConfidenceGateDecision | None = None,
        delegation_texts: dict[RankedCandidate, str | None] | None = None,
    ) -> _PreflightSelection | None:
        """Return a distinct, fully re-gated, live-valid delegated text.

        An executable parse whose intent disagrees with candidate metadata is
        retained as a safe fallback. Within the bounded attempt budget, prefer
        a lower-ranked surface form of that same intended intent when live
        recognition confirms the metadata. This changes wording, not the
        lexical intent decision, and preserves the original executable result
        when no consistent alternative exists.

        ``initial_decision`` carries the caller's gate evaluation of the full
        ranked sequence so the first iteration does not repeat it.
        """
        state = _PreflightState(list(ranked), set(), None, initial_decision)
        while attempt := self._next_preflight_attempt(
            state,
            user_input,
            min_confidence,
            min_margin,
            delegation_texts,
        ):
            selection, divergent = await self._async_observe_preflight_candidate(
                attempt.ranked_candidate,
                attempt.delegated_text,
                attempt.decision,
                state.attempted_texts,
                user_input,
            )
            if selection is not None:
                return selection
            if divergent is not None and state.intent_divergent_fallback is None:
                state.intent_divergent_fallback = divergent
            state.remaining = self._filter_preflight_remaining(
                state.remaining,
                user_input,
                attempt.delegated_key,
                delegation_texts,
                state.intent_divergent_fallback,
            )

        if state.intent_divergent_fallback is None:
            return None
        return self._finalize_intent_divergent_fallback(
            state.intent_divergent_fallback,
            state.attempted_texts,
        )

    def _evaluate_preflight_gate(
        self,
        remaining: Sequence[RankedCandidate],
        user_input: ConversationInput,
        min_confidence: float,
        min_margin: float,
    ) -> ConfidenceGateDecision:
        """Evaluate confidence gates over remaining candidates and publish the decision."""
        decision = evaluate_confidence_gates(
            remaining,
            min_confidence=min_confidence,
            min_margin=min_margin,
            query=user_input.text,
            language=normalize_language(user_input.language),
        )
        self._runtime.update_diagnostics(confidence_gate=decision.as_json_dict())
        return decision

    async def _async_observe_preflight_candidate(
        self,
        selected: RankedCandidate,
        delegated_text: str,
        decision: ConfidenceGateDecision,
        attempted_texts: set[str],
        user_input: ConversationInput,
    ) -> tuple[_PreflightSelection | None, _PreflightSelection | None]:
        """Observe one delegated text and classify the executable outcome.

        Returns ``(selection, None)`` when live recognition confirms the
        candidate's intended intent, ``(None, selection)`` for an executable
        parse whose intent diverges from candidate metadata, and
        ``(None, None)`` when the text is not executable.
        """
        observation = await async_observe_delegated_text(
            self.hass,
            user_input,
            delegated_text,
        )
        self._record_recognition_diagnostics(
            selected,
            delegated_text,
            observation,
            len(attempted_texts),
        )
        if observation.executable:
            selection = _PreflightSelection(
                ranked_candidate=selected,
                delegated_text=delegated_text,
                observation=observation,
                decision=decision,
                attempt_count=len(attempted_texts),
                attempted_text_keys=frozenset(attempted_texts),
            )
            if observation.intent_name == selected.candidate.intent_name:
                return selection, None
            return None, selection
        return None, None

    def _filter_preflight_remaining(
        self,
        remaining: Sequence[RankedCandidate],
        user_input: ConversationInput,
        delegated_key: str,
        delegation_texts: dict[RankedCandidate, str | None] | None,
        intent_divergent_fallback: _PreflightSelection | None,
    ) -> list[RankedCandidate]:
        """Drop the attempted text and, after a divergence, other-intent candidates."""
        filtered = self._remaining_distinct_candidates(
            remaining,
            user_input,
            delegated_key,
            delegation_texts,
        )
        if intent_divergent_fallback is not None:
            intended_intent = intent_divergent_fallback.ranked_candidate.candidate.intent_name
            filtered = [
                candidate
                for candidate in filtered
                if candidate.candidate.intent_name == intended_intent
            ]
        return filtered

    def _finalize_intent_divergent_fallback(
        self,
        intent_divergent_fallback: _PreflightSelection,
        attempted_texts: set[str],
    ) -> _PreflightSelection:
        """Re-publish diagnostics for the retained intent-divergent fallback selection."""
        self._record_recognition_diagnostics(
            intent_divergent_fallback.ranked_candidate,
            intent_divergent_fallback.delegated_text,
            intent_divergent_fallback.observation,
            len(attempted_texts),
        )
        return _PreflightSelection(
            ranked_candidate=intent_divergent_fallback.ranked_candidate,
            delegated_text=intent_divergent_fallback.delegated_text,
            observation=intent_divergent_fallback.observation,
            decision=intent_divergent_fallback.decision,
            attempt_count=len(attempted_texts),
            attempted_text_keys=frozenset(attempted_texts),
        )

    def _remaining_distinct_candidates(
        self,
        ranked: Sequence[RankedCandidate],
        user_input: ConversationInput,
        attempted_key: str,
        delegation_texts: dict[RankedCandidate, str | None] | None = None,
    ) -> list[RankedCandidate]:
        """Remove every candidate that delegates the attempted normalized text."""
        remaining: list[RankedCandidate] = []
        for candidate in ranked:
            candidate_text = self._candidate_delegation_text(
                candidate, user_input, delegation_texts
            )
            if candidate_text is None or normalize_text(candidate_text) == attempted_key:
                continue
            remaining.append(candidate)
        return remaining

    def _record_recognition_diagnostics(
        self,
        ranked_candidate: RankedCandidate,
        delegated_text: str,
        observation: RecognitionObservation,
        attempt_count: int,
    ) -> None:
        """Publish bounded live-recognition and metadata-divergence evidence."""
        intent_matches, slots_match = metadata_matches_observation(
            ranked_candidate.candidate.intent_name,
            ranked_candidate.candidate.parsed_slots,
            observation,
        )
        metadata_diverged = observation.executable and not (intent_matches and slots_match)
        if not observation.executable or (
            (intent_matches or slots_match) and intent_matches and slots_match
        ):
            divergence_reason = None
        elif not intent_matches and not slots_match:
            divergence_reason = "intent_and_slots"
        elif not intent_matches:
            divergence_reason = "intent"
        else:
            divergence_reason = "slots"
        self._runtime.update_diagnostics(
            clear_recognition_trace=True,
            selected_delegated_text_hash=hashlib.sha256(
                normalize_text(delegated_text).encode()
            ).hexdigest(),
            selected_candidate_source=ranked_candidate.candidate.source.value,
            recognition_kind=observation.kind.value,
            recognition_intent=observation.intent_name,
            recognition_unmatched_count=len(observation.unmatched_entities),
            recognition_latency_ms=observation.latency_ms,
            preflight_attempt_count=attempt_count,
            metadata_diverged=metadata_diverged,
            metadata_intent_matches_observed=(intent_matches if observation.executable else None),
            metadata_slots_match_observed=(slots_match if observation.executable else None),
            metadata_divergence_reason=divergence_reason,
            recovery_used=attempt_count > 1,
            selected_from_fuzzy_registry=(
                ranked_candidate.candidate.metadata.get("registry_retrieval") == "fuzzy"
            ),
        )

    async def _async_execute_ranked_candidate(
        self,
        ranked_candidate: RankedCandidate,
        user_input: ConversationInput,
        delegated_text: str | None = None,
    ) -> ConversationResult | None:
        """Execute one canonical candidate through the primary HassIL agent.

        Wildcard placeholders (e.g. ``shopping_list_item``) in candidate text
        are rehydrated from the original query before delegation so that the
        downstream agent receives real free-text values. Delegate exceptions return
        ``None`` so the caller can preserve the configured fallback-agent contract.
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
        """Send untouched user text to the configured fallback conversation agent.

        This boundary must remain reachable after every unsuccessful local path,
        including handler errors and delegate exceptions. The configured agent may be
        an LLM and must not be bypassed based on assumptions about local side effects.
        """
        return await self._delegate_text(user_input.text, user_input, primary=False)

    async def _async_error_allows_candidate_recovery(
        self,
        error_code: str | None,
        delegated_text: str | None,
        user_input: ConversationInput,
    ) -> bool:
        """Return whether an error permits another local canonical-candidate attempt.

        This gate controls only local recovery. It must never be used to decide whether
        the original text reaches the user-configured fallback conversation agent.

        The re-recognition for NO_VALID_TARGETS is deliberate: execution can
        fail against registry state that changed after preflight, so the
        preflight observation must not substitute for a fresh check.
        """
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
        *,
        excluded_delegated_keys: frozenset[str] = frozenset(),
        delegation_texts: dict[RankedCandidate, str | None] | None = None,
    ) -> tuple[RankedCandidate, str, RecognitionObservation] | None:
        """Return at most one fully re-gated, error-compatible recovery candidate."""
        if not await self._async_error_allows_candidate_recovery(
            error_code, delegated_text, user_input
        ):
            return None

        remaining = self._recovery_candidate_pairs(
            ranked,
            selected,
            delegated_text,
            user_input,
            excluded_delegated_keys,
            delegation_texts,
        )
        selected_pair = self._gated_recovery_pair(
            remaining,
            min_confidence,
            min_margin,
            user_input,
        )
        if selected_pair is None:
            return None
        candidate, candidate_text = selected_pair
        observation = await async_observe_delegated_text(
            self.hass,
            user_input,
            candidate_text,
        )
        if not observation.executable:
            return None
        return candidate, candidate_text, observation

    def _recovery_candidate_pairs(
        self,
        ranked: Sequence[RankedCandidate],
        selected: RankedCandidate,
        delegated_text: str | None,
        user_input: ConversationInput,
        excluded_delegated_keys: frozenset[str],
        delegation_texts: dict[RankedCandidate, str | None] | None,
    ) -> list[tuple[RankedCandidate, str]]:
        """Return delegable candidate-text pairs excluding attempted delegated texts."""
        delegated_key = normalize_text(delegated_text) if delegated_text is not None else None
        remaining: list[tuple[RankedCandidate, str]] = []
        for candidate in ranked:
            if candidate is selected:
                continue
            candidate_text = self._candidate_delegation_text(
                candidate, user_input, delegation_texts
            )
            if candidate_text is None:
                continue
            candidate_key = normalize_text(candidate_text)
            if candidate_key in excluded_delegated_keys or (
                delegated_key is not None and candidate_key == delegated_key
            ):
                continue
            remaining.append((candidate, candidate_text))
        return remaining

    def _gated_recovery_pair(
        self,
        remaining: list[tuple[RankedCandidate, str]],
        min_confidence: float,
        min_margin: float,
        user_input: ConversationInput,
    ) -> tuple[RankedCandidate, str] | None:
        """Re-gate recovery pairs, publish the decision, and return the accepted pair."""
        decision = evaluate_confidence_gates(
            tuple(candidate for candidate, _text in remaining),
            min_confidence=min_confidence,
            min_margin=min_margin,
            query=user_input.text,
            language=normalize_language(user_input.language),
        )
        self._runtime.update_diagnostics(confidence_gate=decision.as_json_dict())
        accepted = decision.accepted_candidate
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
        observation = await async_observe_delegated_text(
            self.hass,
            user_input,
            delegated_text,
        )
        return bool(observation.unmatched_entities)

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
        optional_arguments = {
            argument: getattr(user_input, argument, None)
            for argument in CONVERSATION_INPUT_OPTIONAL_FIELDS
            if argument in _ASYNC_CONVERSE_PARAMETERS
        }
        return await conversation.async_converse(
            self.hass,
            text,
            user_input.conversation_id,
            user_input.context,
            language=user_input.language,
            agent_id=agent_id,
            device_id=user_input.device_id,
            **optional_arguments,
        )

    @contextlib.contextmanager
    def _capture_chat_log_deltas(self, chat_log: object | None) -> Iterator[list[ChatLogDelta]]:
        """Temporarily intercept and capture chat log delta listener callbacks."""
        captured_deltas: list[ChatLogDelta] = []
        if not isinstance(chat_log, _ChatLogWithDeltaListener):
            yield captured_deltas
            return

        def task_listener(log: object, delta: ChatLogDelta) -> None:
            captured_deltas.append(delta)

        current_listener = getattr(chat_log, "delta_listener", None)

        if isinstance(current_listener, TaskDeltaWrapper):
            wrapper = current_listener
            restore_listener = wrapper.original_listener
        else:
            wrapper = TaskDeltaWrapper(
                current_listener if isinstance(current_listener, ChatLogDeltaListener) else None
            )
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

    def _play_back_deltas(self, chat_log: object | None, deltas: list[ChatLogDelta]) -> None:
        """Play back captured deltas to the original listener."""
        if chat_log is not None and deltas:
            listener = getattr(chat_log, "delta_listener", None)
            while isinstance(listener, TaskDeltaWrapper):
                listener = listener.original_listener
            if callable(listener):
                for delta in deltas:
                    listener(chat_log, delta)

    def _get_active_chat_log(self) -> ChatLog | None:
        """Return the active chat log for the current conversation context."""
        try:
            from homeassistant.components.conversation.chat_log import current_chat_log

            return current_chat_log.get(None)
        except (ImportError, RuntimeError, AttributeError, LookupError) as err:
            _LOGGER.debug("No conversation chat log available: %s", err)
        return None

    @staticmethod
    def _restore_chat_log_content(chat_log: ChatLog | None, old_len: int | None) -> None:
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
