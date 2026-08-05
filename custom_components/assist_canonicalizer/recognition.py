"""Side-effect-free observation of Home Assistant Default Agent recognition."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from copy import copy
from dataclasses import dataclass
from enum import StrEnum

from homeassistant.components.conversation.agent_manager import async_get_agent
from homeassistant.components.conversation.const import HOME_ASSISTANT_AGENT
from homeassistant.components.conversation.models import ConversationInput
from homeassistant.core import HomeAssistant
from homeassistant.util.json import JsonObjectType

from .const import (
    CONVERSATION_INPUT_AREA_CONTEXT_FIELD,
    CONVERSATION_INPUT_AREA_CONTEXT_FIELDS,
    CONVERSATION_INPUT_OPTIONAL_CONTEXT_FIELDS,
    CONVERSATION_INPUT_REQUIRED_CONTEXT_FIELDS,
)
from .normalization import normalize_text
from .utils import elapsed_ms

_MAX_OBSERVED_VALUE_LENGTH = 128
type ObservedSlotValue = bool | int | float | str | None


class RecognitionKind(StrEnum):
    """Executable and non-executable live recognition outcomes."""

    INTENT = "intent"
    SENTENCE_TRIGGER = "sentence_trigger"
    NO_MATCH = "no_match"
    UNMATCHED_TARGET = "unmatched_target"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RecognitionObservation:
    """Immutable, bounded facts from one live recognition attempt."""

    kind: RecognitionKind
    intent_name: str | None = None
    slots: tuple[tuple[str, ObservedSlotValue], ...] = ()
    unmatched_entities: tuple[str, ...] = ()
    required_context: tuple[str, ...] = ()
    excluded_context: tuple[str, ...] = ()
    forwarded_context: tuple[str, ...] = ()
    latency_ms: float = 0.0
    error_category: str | None = None

    @property
    def executable(self) -> bool:
        """Return whether the delegated text can safely advance to execution."""
        return self.kind is RecognitionKind.INTENT

    def as_dict(self, *, include_slot_values: bool = True) -> JsonObjectType:
        """Return a deterministic serializable representation."""
        return {
            "kind": self.kind.value,
            "intent_name": self.intent_name,
            "slots": (
                dict(self.slots) if include_slot_values else [name for name, _value in self.slots]
            ),
            "unmatched_entities": list(self.unmatched_entities),
            "required_context": list(self.required_context),
            "excluded_context": list(self.excluded_context),
            "forwarded_context": list(self.forwarded_context),
            "latency_ms": self.latency_ms,
            "error_category": self.error_category,
            "executable": self.executable,
        }


async def async_observe_delegated_text(
    hass: HomeAssistant,
    user_input: ConversationInput,
    delegated_text: str,
) -> RecognitionObservation:
    """Recognize delegated text without invoking a trigger or intent handler."""
    started_at = time.monotonic()
    forwarded_context = _forwarded_context_fields(user_input)
    calls = _recognition_calls_or_error(hass)
    if isinstance(calls, str):
        return _error_observation(started_at, forwarded_context, calls)
    recognize_trigger_call, recognize_intent_call = calls

    try:
        recognition_input = copy(user_input)
        recognition_input.text = delegated_text
        recognition_input.agent_id = HOME_ASSISTANT_AGENT
        trigger_result = await recognize_trigger_call(recognition_input)
        if trigger_result is not None:
            return RecognitionObservation(
                kind=RecognitionKind.SENTENCE_TRIGGER,
                forwarded_context=forwarded_context,
                latency_ms=elapsed_ms(started_at),
            )
        result = await recognize_intent_call(recognition_input)
    except Exception:
        return _error_observation(started_at, forwarded_context, "recognition_failed")

    if result is None:
        return RecognitionObservation(
            kind=RecognitionKind.NO_MATCH,
            forwarded_context=forwarded_context,
            latency_ms=elapsed_ms(started_at),
        )

    return _observation_from_result(result, started_at, forwarded_context)


def _recognition_calls_or_error(
    hass: HomeAssistant,
) -> (
    tuple[
        Callable[[ConversationInput], Awaitable[object]],
        Callable[[ConversationInput], Awaitable[object]],
    ]
    | str
):
    """Return the default agent's recognition callables or an error category."""
    try:
        default_agent = async_get_agent(hass, HOME_ASSISTANT_AGENT)
    except Exception:
        return "agent_lookup_failed"
    if default_agent is None:
        return "agent_unavailable"

    recognize_trigger = getattr(default_agent, "async_recognize_sentence_trigger", None)
    recognize_intent = getattr(default_agent, "async_recognize_intent", None)
    if not callable(recognize_trigger) or not callable(recognize_intent):
        return "recognition_api_unavailable"

    async def call_recognize_trigger(user_input: ConversationInput) -> object:
        """Call the dynamically discovered sentence-trigger recognizer."""
        result = recognize_trigger(user_input)
        if not inspect.isawaitable(result):
            raise TypeError("async_recognize_sentence_trigger returned a non-awaitable result")
        return await result

    async def call_recognize_intent(user_input: ConversationInput) -> object:
        """Call the dynamically discovered intent recognizer."""
        result = recognize_intent(user_input)
        if not inspect.isawaitable(result):
            raise TypeError("async_recognize_intent returned a non-awaitable result")
        return await result

    return call_recognize_trigger, call_recognize_intent


def _observation_from_result(
    result: object,
    started_at: float,
    forwarded_context: tuple[str, ...],
) -> RecognitionObservation:
    """Build a bounded observation from one successful intent recognition result."""
    unmatched = tuple(
        sorted(
            {
                name
                for entity in getattr(result, "unmatched_entities_list", ())
                if isinstance((name := getattr(entity, "name", None)), str) and name
            }
        )
    )
    intent_data = getattr(result, "intent_data", None)
    required_context = _context_constraint_names(getattr(intent_data, "requires_context", None))
    excluded_context = _context_constraint_names(getattr(intent_data, "excludes_context", None))
    intent_result = getattr(result, "intent", None)
    intent_name = getattr(intent_result, "name", None)
    if not isinstance(intent_name, str) or not intent_name:
        return _error_observation(started_at, forwarded_context, "invalid_result")
    slot_items: list[tuple[str, ObservedSlotValue]] = []
    for entity in getattr(result, "entities_list", ()):
        name = getattr(entity, "name", None)
        if isinstance(name, str) and name:
            slot_items.append((name, _bounded_serializable_value(getattr(entity, "value", None))))
    slots = tuple(sorted(slot_items, key=lambda item: item[0]))
    return RecognitionObservation(
        kind=(RecognitionKind.UNMATCHED_TARGET if unmatched else RecognitionKind.INTENT),
        intent_name=intent_name,
        slots=slots,
        unmatched_entities=unmatched,
        required_context=required_context,
        excluded_context=excluded_context,
        forwarded_context=forwarded_context,
        latency_ms=elapsed_ms(started_at),
    )


def metadata_matches_observation(
    intent_name: str,
    slots: Mapping[str, object],
    observation: RecognitionObservation,
) -> tuple[bool, bool]:
    """Compare non-authoritative candidate metadata with a live observation."""
    intent_matches = observation.intent_name == intent_name
    # Observed values were bounded by _bounded_serializable_value before
    # storage; expected values must pass through the same bound or slot
    # values longer than the cap always report a false divergence.
    expected = {
        name: _normalized_observed_value(_bounded_serializable_value(value))
        for name, value in slots.items()
        if isinstance(name, str)
    }
    observed = {name: _normalized_observed_value(value) for name, value in observation.slots}
    return intent_matches, expected == observed


def _forwarded_context_fields(user_input: ConversationInput) -> tuple[str, ...]:
    """Return request-context field names preserved by the copied input."""
    fields: list[str] = list(CONVERSATION_INPUT_REQUIRED_CONTEXT_FIELDS)
    fields.extend(
        field_name
        for field_name in CONVERSATION_INPUT_OPTIONAL_CONTEXT_FIELDS
        if getattr(user_input, field_name, None) is not None
    )
    if any(
        getattr(user_input, field_name, None) is not None
        for field_name in CONVERSATION_INPUT_AREA_CONTEXT_FIELDS
    ):
        fields.append(CONVERSATION_INPUT_AREA_CONTEXT_FIELD)
    return tuple(fields)


def _context_constraint_names(value: object) -> tuple[str, ...]:
    """Return deterministic constraint keys without retaining values."""
    if not isinstance(value, Mapping):
        return ()
    return tuple(sorted(key for key in value if isinstance(key, str) and key))


def _bounded_serializable_value(value: object) -> ObservedSlotValue:
    """Return a bounded JSON-compatible scalar for handler-visible slots."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:_MAX_OBSERVED_VALUE_LENGTH]
    return str(value)[:_MAX_OBSERVED_VALUE_LENGTH]


def _normalized_observed_value(value: object) -> str:
    """Return a stable comparison value for diagnostic metadata divergence."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).casefold()
    return normalize_text(str(value))


def _error_observation(
    started_at: float,
    forwarded_context: tuple[str, ...],
    category: str,
) -> RecognitionObservation:
    """Build a non-executable observation without exception or credential text."""
    return RecognitionObservation(
        kind=RecognitionKind.ERROR,
        forwarded_context=forwarded_context,
        latency_ms=elapsed_ms(started_at),
        error_category=category,
    )
