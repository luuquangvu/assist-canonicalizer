"""Tests for side-effect-free live Default Agent recognition."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components.conversation.const import HOME_ASSISTANT_AGENT
from homeassistant.components.conversation.models import ConversationInput
from homeassistant.core import Context, HomeAssistant

from custom_components.assist_canonicalizer.const import (
    CONVERSATION_INPUT_AGENT_ID_FIELD,
    CONVERSATION_INPUT_AREA_CONTEXT_FIELD,
    CONVERSATION_INPUT_CONTEXT_FIELD,
    CONVERSATION_INPUT_CONVERSATION_ID_FIELD,
    CONVERSATION_INPUT_DEVICE_ID_FIELD,
    CONVERSATION_INPUT_EXTRA_SYSTEM_PROMPT_FIELD,
    CONVERSATION_INPUT_LANGUAGE_FIELD,
    CONVERSATION_INPUT_OPTIONAL_FIELDS,
    CONVERSATION_INPUT_SATELLITE_ID_FIELD,
    CONVERSATION_INPUT_TEXT_FIELD,
)
from custom_components.assist_canonicalizer.recognition import (
    _MAX_OBSERVED_VALUE_LENGTH,
    RecognitionKind,
    RecognitionObservation,
    async_observe_delegated_text,
    metadata_matches_observation,
)


def _conversation_input() -> ConversationInput:
    """Return a request populated with every supported context field."""
    signature = inspect.signature(ConversationInput)
    kwargs: dict[str, Any] = {
        CONVERSATION_INPUT_TEXT_FIELD: "turn teh lamp on",
        CONVERSATION_INPUT_CONTEXT_FIELD: Context(user_id="benchmark-user"),
        CONVERSATION_INPUT_CONVERSATION_ID_FIELD: "recognition-request",
        CONVERSATION_INPUT_DEVICE_ID_FIELD: "device-kitchen",
        CONVERSATION_INPUT_LANGUAGE_FIELD: "en-US",
    }
    optional = {
        CONVERSATION_INPUT_AGENT_ID_FIELD: "conversation.assist_canonicalizer",
        CONVERSATION_INPUT_SATELLITE_ID_FIELD: "assist_satellite.kitchen",
        CONVERSATION_INPUT_EXTRA_SYSTEM_PROMPT_FIELD: "stay local",
    }
    kwargs |= {name: value for name, value in optional.items() if name in signature.parameters}
    return ConversationInput(**kwargs)


@pytest.mark.asyncio
async def test_recognition_forwards_context_and_observes_intent_without_execution() -> None:
    """Copy the request, change only routing text, and return bounded live facts."""
    user_input = _conversation_input()
    recognize_trigger = AsyncMock(return_value=None)
    recognize_intent = AsyncMock(
        return_value=SimpleNamespace(
            intent=SimpleNamespace(name="HassTurnOn"),
            entities_list=(
                SimpleNamespace(name="name", value="Living Room Lamp"),
                SimpleNamespace(name="domain", value="light"),
            ),
            unmatched_entities_list=(),
            intent_data=SimpleNamespace(
                requires_context={"domain": "light"},
                excludes_context={"state": "unavailable"},
            ),
        )
    )
    default_agent = SimpleNamespace(
        async_recognize_sentence_trigger=recognize_trigger,
        async_recognize_intent=recognize_intent,
    )

    with patch(
        "custom_components.assist_canonicalizer.recognition.async_get_agent",
        return_value=default_agent,
    ):
        observation = await async_observe_delegated_text(
            cast(HomeAssistant, object()), user_input, "turn on Living Room Lamp"
        )

    assert observation.kind is RecognitionKind.INTENT
    assert observation.executable
    assert observation.intent_name == "HassTurnOn"
    assert dict(observation.slots) == {"domain": "light", "name": "Living Room Lamp"}
    assert observation.required_context == ("domain",)
    assert observation.excluded_context == ("state",)
    assert set(observation.forwarded_context) >= {
        CONVERSATION_INPUT_CONTEXT_FIELD,
        CONVERSATION_INPUT_CONVERSATION_ID_FIELD,
        CONVERSATION_INPUT_DEVICE_ID_FIELD,
        CONVERSATION_INPUT_LANGUAGE_FIELD,
        CONVERSATION_INPUT_AREA_CONTEXT_FIELD,
    }
    recognize_trigger.assert_awaited_once()
    recognize_intent.assert_awaited_once()
    await_args = recognize_intent.await_args
    assert await_args is not None
    recognition_input = await_args.args[0]
    assert recognition_input is not user_input
    assert recognition_input.text == "turn on Living Room Lamp"
    assert recognition_input.agent_id == HOME_ASSISTANT_AGENT
    assert recognition_input.context is user_input.context
    assert recognition_input.conversation_id == user_input.conversation_id
    assert recognition_input.device_id == user_input.device_id
    assert recognition_input.language == user_input.language
    for optional_field in CONVERSATION_INPUT_OPTIONAL_FIELDS:
        if hasattr(user_input, optional_field):
            assert getattr(recognition_input, optional_field) == getattr(user_input, optional_field)
    assert user_input.text == "turn teh lamp on"


@pytest.mark.asyncio
async def test_recognition_sorts_duplicate_slots_without_comparing_values() -> None:
    """Keep duplicate slot names with heterogeneous values in stable name order."""
    default_agent = SimpleNamespace(
        async_recognize_sentence_trigger=AsyncMock(return_value=None),
        async_recognize_intent=AsyncMock(
            return_value=SimpleNamespace(
                intent=SimpleNamespace(name="HassListAddItem"),
                entities_list=(
                    SimpleNamespace(name="item", value=3),
                    SimpleNamespace(name="domain", value="todo"),
                    SimpleNamespace(name="item", value="milk"),
                ),
                unmatched_entities_list=(),
                intent_data=SimpleNamespace(requires_context=None, excludes_context=None),
            )
        ),
    )

    with patch(
        "custom_components.assist_canonicalizer.recognition.async_get_agent",
        return_value=default_agent,
    ):
        observation = await async_observe_delegated_text(
            cast(HomeAssistant, object()), _conversation_input(), "add milk to my list"
        )

    assert observation.slots == (("domain", "todo"), ("item", 3), ("item", "milk"))


@pytest.mark.asyncio
async def test_sentence_trigger_is_detected_without_invoking_callback() -> None:
    """A rewritten trigger is non-executable and its callback remains untouched."""
    callback = AsyncMock()
    recognize_trigger = AsyncMock(return_value=SimpleNamespace(async_run=callback))
    recognize_intent = AsyncMock()
    default_agent = SimpleNamespace(
        async_recognize_sentence_trigger=recognize_trigger,
        async_recognize_intent=recognize_intent,
    )

    with patch(
        "custom_components.assist_canonicalizer.recognition.async_get_agent",
        return_value=default_agent,
    ):
        observation = await async_observe_delegated_text(
            cast(HomeAssistant, object()), _conversation_input(), "start bedtime"
        )

    assert observation.kind is RecognitionKind.SENTENCE_TRIGGER
    assert not observation.executable
    callback.assert_not_awaited()
    recognize_intent.assert_not_awaited()


@pytest.mark.asyncio
async def test_unmatched_target_is_non_executable_and_deterministic() -> None:
    """Expose unmatched target names in sorted, serializable form."""
    default_agent = SimpleNamespace(
        async_recognize_sentence_trigger=AsyncMock(return_value=None),
        async_recognize_intent=AsyncMock(
            return_value=SimpleNamespace(
                intent=SimpleNamespace(name="HassTurnOn"),
                entities_list=(),
                unmatched_entities_list=(
                    SimpleNamespace(name="zeta"),
                    SimpleNamespace(name="alpha"),
                ),
                intent_data=SimpleNamespace(requires_context=None, excludes_context=None),
            )
        ),
    )
    with patch(
        "custom_components.assist_canonicalizer.recognition.async_get_agent",
        return_value=default_agent,
    ):
        observation = await async_observe_delegated_text(
            cast(HomeAssistant, object()), _conversation_input(), "turn on unknown light"
        )

    assert observation.kind is RecognitionKind.UNMATCHED_TARGET
    assert not observation.executable
    assert observation.unmatched_entities == ("alpha", "zeta")
    assert observation.as_dict()["unmatched_entities"] == ["alpha", "zeta"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "category"),
    [
        (None, "agent_unavailable"),
        (SimpleNamespace(), "recognition_api_unavailable"),
        (
            SimpleNamespace(
                async_recognize_sentence_trigger=AsyncMock(side_effect=RuntimeError("secret")),
                async_recognize_intent=AsyncMock(),
            ),
            "recognition_failed",
        ),
    ],
)
async def test_recognition_api_failures_fail_closed(agent: Any, category: str) -> None:
    """Missing or failing supported APIs become bounded non-executable results."""
    with patch(
        "custom_components.assist_canonicalizer.recognition.async_get_agent",
        return_value=agent,
    ):
        observation = await async_observe_delegated_text(
            cast(HomeAssistant, object()), _conversation_input(), "turn on lamp"
        )

    assert observation.kind is RecognitionKind.ERROR
    assert not observation.executable
    assert observation.error_category == category
    assert "secret" not in str(observation.as_dict())


def test_metadata_divergence_is_diagnostic_only() -> None:
    """Compare generated metadata without changing observation executability."""
    observation = RecognitionObservation(
        kind=RecognitionKind.INTENT,
        intent_name="HassLightSet",
        slots=(("brightness", 50), ("name", "Living Room Lamp")),
    )

    assert metadata_matches_observation(
        "HassLightSet",
        {"brightness": "50", "name": "living room lamp"},
        observation,
    ) == (True, True)
    assert metadata_matches_observation(
        "HassTurnOn",
        {"name": "Kitchen Lamp"},
        observation,
    ) == (False, False)
    assert observation.executable


def test_metadata_matches_observation_bounds_long_expected_values() -> None:
    """Bound expected slot values like observed ones to avoid false divergence."""
    long_value = "x" * (_MAX_OBSERVED_VALUE_LENGTH + 1)
    observation = RecognitionObservation(
        kind=RecognitionKind.INTENT,
        intent_name="HassListAddItem",
        slots=(("item", long_value[:_MAX_OBSERVED_VALUE_LENGTH]),),
    )

    assert metadata_matches_observation(
        "HassListAddItem",
        {"item": long_value},
        observation,
    ) == (True, True)
