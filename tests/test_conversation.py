"""Tests for the Assist Canonicalizer conversation entity platform."""

import asyncio
import contextlib
import inspect
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.conversation.const import HOME_ASSISTANT_AGENT
from homeassistant.components.conversation.models import ConversationInput
from homeassistant.core import Context
from homeassistant.helpers import intent

import custom_components.assist_canonicalizer as assist_canonicalizer
from custom_components.assist_canonicalizer import _discover_pipeline_languages
from custom_components.assist_canonicalizer.candidate import Candidate
from custom_components.assist_canonicalizer.const import (
    CONF_FALLBACK_AGENT_ID,
    CONF_MIN_CONFIDENCE,
    CONF_MIN_MARGIN,
    DATA_RUNTIME,
    DOMAIN,
    FallbackReason,
)
from custom_components.assist_canonicalizer.conversation import (
    AssistCanonicalizerConversationEntity,
    async_setup_entry,
)
from custom_components.assist_canonicalizer.indexer import CanonicalIndex, build_index
from custom_components.assist_canonicalizer.ranking import RankedCandidate, ScoreBreakdown
from custom_components.assist_canonicalizer.recognition import (
    RecognitionKind,
    RecognitionObservation,
)
from custom_components.assist_canonicalizer.runtime import CanonicalizerRuntime
from custom_components.assist_canonicalizer.utils import wildcard_slot_names


class MockConversationInput(ConversationInput):
    """Mock for ConversationInput."""

    def __init__(self, text: str, language: str = "vi", conversation_id: str = "conv-1") -> None:
        """Initialize."""
        sig = inspect.signature(super().__init__)
        kwargs: dict[str, Any] = {
            "text": text,
            "context": Context(),
            "conversation_id": conversation_id,
            "device_id": None,
            "language": language,
        }
        if "agent_id" in sig.parameters:
            kwargs["agent_id"] = "test_agent"
        if "satellite_id" in sig.parameters:
            kwargs["satellite_id"] = None
        super().__init__(**kwargs)
        if not hasattr(self, "agent_id"):
            self.agent_id = "test_agent"
        if not hasattr(self, "satellite_id"):
            self.satellite_id = None


class MockConversationResult:
    """Mock for ConversationResult."""

    def __init__(self, response: Any, conversation_id: str) -> None:
        """Initialize."""
        self.response = response
        self.conversation_id = conversation_id


class MockIntentResponse:
    """Mock for IntentResponse."""

    def __init__(self, language: str) -> None:
        """Initialize."""
        self.language = language
        self.error_code = None

    def async_set_error(self, code: Any, message: str) -> None:
        """Set error."""
        self.error_code = code


async def _executor_job_returning_empty_snapshot_index(target: Any, *args: Any, **kwargs: Any):
    """Mock executor work and return an empty index for snapshot builds."""
    if getattr(target, "__name__", None) == "_build_index_from_snapshot":
        return build_index("vi", [])
    return target(*args, **kwargs)


async def _executor_job_returning_empty_build_index(target: Any, *args: Any, **kwargs: Any):
    """Mock executor work and return an empty index for direct index builds."""
    if getattr(target, "__name__", None) == "build_index":
        return build_index("vi", [])
    return target(*args, **kwargs)


def _active_pipeline_data(context: Context, pipeline: Any) -> SimpleNamespace:
    """Return Home Assistant pipeline-run state matching one request context."""
    run = SimpleNamespace(context=context, pipeline=pipeline)
    return SimpleNamespace(pipeline_runs=SimpleNamespace(_pipeline_runs={"pipeline": {"run": run}}))


def _mock_assist_pipeline_const(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the Assist Pipeline domain without importing optional integration dependencies."""
    package_name = "homeassistant.components.assist_pipeline"
    package_module = ModuleType(package_name)
    const_module = ModuleType(f"{package_name}.const")
    package_attrs = cast(Any, package_module)
    const_attrs = cast(Any, const_module)
    package_attrs.__path__ = []
    package_attrs.const = const_module
    const_attrs.DOMAIN = "assist_pipeline"
    monkeypatch.setitem(sys.modules, package_name, package_module)
    monkeypatch.setitem(sys.modules, f"{package_name}.const", const_module)


@pytest.fixture(autouse=True)
def _default_live_recognition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep existing conversation tests focused while preflight has dedicated coverage."""

    async def observe(_hass: Any, _user_input: Any, _text: str) -> RecognitionObservation:
        """Return a default executable live-recognition observation."""
        return RecognitionObservation(
            kind=RecognitionKind.INTENT,
            intent_name="HassTurnOn",
        )

    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.conversation.async_observe_delegated_text",
        observe,
    )


@pytest.mark.asyncio
async def test_async_setup_entry() -> None:
    """Test setting up the conversation platform entry."""
    hass = MagicMock()
    config_entry = MagicMock()
    config_entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    hass.data = {DOMAIN: {"test_entry": {DATA_RUNTIME: runtime}}}

    async_add_entities = MagicMock()

    await async_setup_entry(hass, config_entry, async_add_entities)
    async_add_entities.assert_called_once()

    hass.data = {DOMAIN: {"test_entry": {DATA_RUNTIME: None}}}
    with pytest.raises(RuntimeError, match="runtime is not loaded"):
        await async_setup_entry(hass, config_entry, async_add_entities)


@pytest.mark.asyncio
async def test_conversation_entity_lifecycle() -> None:
    """Test conversation entity added to and removed from hass."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()

    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock()

    with (
        patch("homeassistant.components.conversation.async_set_agent") as mock_set,
        patch("homeassistant.components.conversation.async_unset_agent") as mock_unset,
    ):
        await entity.async_added_to_hass()
        mock_set.assert_called_once_with(entity.hass, entry, entity)

        await entity.async_will_remove_from_hass()
        mock_unset.assert_called_once_with(entity.hass, entry)


@pytest.mark.asyncio
async def test_conversation_entity_properties_and_reload() -> None:
    """Test simple property getters and index reload callbacks."""
    entry = MagicMock()
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock()

    assert entity.supported_languages == "*"

    mock_index = MagicMock()
    mock_index.language = "en"
    runtime.set_index(mock_index)
    runtime.indexes["en"] = MagicMock()
    assert "en" in runtime.indexes
    with patch.object(
        CanonicalizerRuntime,
        "async_clear_index",
        AsyncMock(side_effect=lambda hass, language=None: runtime.clear_index(language)),
    ) as mock_clear:
        await entity.async_reload("en-US")
        mock_clear.assert_awaited_once_with(entity.hass, "en")
    assert "en" not in runtime.indexes


@pytest.mark.asyncio
async def test_conversation_entity_prepare() -> None:
    """Test async_prepare index load and rebuild flows."""
    entry = MagicMock()
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock()

    runtime.indexes["en"] = MagicMock()
    with patch.object(
        CanonicalizerRuntime, "async_load_index_from_store", AsyncMock()
    ) as mock_load:
        await entity.async_prepare("en")
        mock_load.assert_not_called()

    del runtime.indexes["en"]
    with patch.object(
        CanonicalizerRuntime, "async_load_index_from_store", AsyncMock(return_value=MagicMock())
    ) as mock_load:
        await entity.async_prepare("en")
        mock_load.assert_called_once()

    with (
        patch.object(
            CanonicalizerRuntime, "async_load_index_from_store", AsyncMock(return_value=None)
        ) as mock_load,
        patch.object(
            CanonicalizerRuntime, "async_rebuild_index", AsyncMock(return_value=MagicMock())
        ) as mock_rebuild,
    ):
        await entity.async_prepare("en")
        mock_load.assert_called_once()
        mock_rebuild.assert_called_once()


@pytest.mark.asyncio
async def test_async_process_error_handling() -> None:
    """Test exception during async_process generates error result and updates diagnostics."""
    entry = MagicMock()
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock()

    mock_resp = MockIntentResponse("en")
    user_input = MockConversationInput("tắt đèn", "vi")

    with (
        patch.object(
            entity,
            "_async_process_with_runtime",
            AsyncMock(side_effect=Exception("Database error")),
        ),
        patch("homeassistant.helpers.intent.IntentResponse", return_value=mock_resp),
    ):
        res = await entity.async_process(user_input)
        assert res.response.error_code is not None
        assert runtime.diagnostics.last_fallback_reason == FallbackReason.UNEXPECTED_EXCEPTION
        assert runtime.diagnostics.last_error == "Database error"


@pytest.mark.asyncio
async def test_async_process_with_runtime_flows() -> None:
    """Test _async_process_with_runtime routing, fallbacks, and validations."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)

    hass = MagicMock()
    hass.async_create_task = lambda coro: asyncio.create_task(coro)
    hass.async_add_executor_job = AsyncMock(
        side_effect=_executor_job_returning_empty_snapshot_index
    )
    entity.hass = hass

    user_input = MockConversationInput("tắt đèn bếp", "vi")

    with patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")):
        res = await entity._async_process_with_runtime(user_input)
        assert res == "raw_delegated"
        assert runtime.diagnostics.last_fallback_reason == FallbackReason.NO_CANDIDATE

    runtime.indexes["vi"] = MagicMock(candidate_count=5)
    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            side_effect=Exception("Rank error"),
        ),
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")),
    ):
        res = await entity._async_process_with_runtime(user_input)
        assert res == "raw_delegated"
        assert runtime.diagnostics.last_fallback_reason == FallbackReason.RANKING_FAILED

    with (
        patch.object(CanonicalizerRuntime, "rank_with_dynamic_candidates", return_value=()),
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")),
    ):
        res = await entity._async_process_with_runtime(user_input)
        assert res == "raw_delegated"
        assert runtime.diagnostics.last_fallback_reason == FallbackReason.NO_CANDIDATE

    rc_low_margin_top = RankedCandidate(
        candidate=Candidate(text="tắt đèn bếp", intent_name="HassTurnOff"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.8,
            char_ngram_score=0.8,
            bm25_score=0.8,
            intent_score=1.0,
            final_score=0.8,
        ),
    )
    rc_low_margin_competitor = RankedCandidate(
        candidate=Candidate(text="bật đèn bếp", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.79,
            char_ngram_score=0.79,
            bm25_score=0.79,
            intent_score=1.0,
            final_score=0.79,
        ),
    )
    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(rc_low_margin_top, rc_low_margin_competitor),
        ),
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")),
    ):
        res = await entity._async_process_with_runtime(user_input)
        assert res == "raw_delegated"
        assert runtime.diagnostics.last_fallback_reason == FallbackReason.LOW_MARGIN

    rc = RankedCandidate(
        candidate=Candidate(text="tắt đèn bếp", intent_name="HassTurnOff"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=1.0,
            final_score=0.9,
        ),
    )
    validation_err_res = MagicMock()
    validation_err_res.response.error_code = "error"

    with (
        patch.object(CanonicalizerRuntime, "rank_with_dynamic_candidates", return_value=(rc,)),
        patch.object(
            entity, "_delegate_text", AsyncMock(return_value=validation_err_res)
        ) as mock_del_text,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")),
    ):
        res = await entity._async_process_with_runtime(user_input)
        assert res == "raw_delegated"
        assert runtime.diagnostics.last_fallback_reason == FallbackReason.VALIDATION_FAILED
        mock_del_text.assert_any_call("tắt đèn bếp", user_input, primary=True)

    validation_ok_res = MagicMock()
    validation_ok_res.response.error_code = None

    with (
        patch.object(CanonicalizerRuntime, "rank_with_dynamic_candidates", return_value=(rc,)),
        patch.object(entity, "_delegate_text", AsyncMock(return_value=validation_ok_res)),
    ):
        res = await entity._async_process_with_runtime(user_input)
        assert res == validation_ok_res


@pytest.mark.asyncio
async def test_empty_static_index_still_uses_dynamic_ranking() -> None:
    """Try dynamic ranking even when the cached static index has no candidates."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = build_index("en", [])
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
    entity.hass = hass

    ranked = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(
            rapidfuzz_score=1.0,
            char_ngram_score=1.0,
            bm25_score=1.0,
            intent_score=1.0,
            final_score=1.0,
        ),
    )
    validation_ok_res = MagicMock()
    validation_ok_res.response.error_code = None
    user_input = MockConversationInput("turn on kitchen light", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(ranked,),
        ) as rank,
        patch.object(entity, "_delegate_text", AsyncMock(return_value=validation_ok_res)),
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")) as raw,
    ):
        res = await entity._async_process_with_runtime(user_input)

    assert res == validation_ok_res
    rank.assert_called_once()
    raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_ranking_receives_satellite_area_context() -> None:
    """Pass Home Assistant satellite/device area context into lexical ranking."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=1)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
    entity.hass = hass

    user_input = MockConversationInput("mute", "en")
    user_input.device_id = "device-1"
    user_input.satellite_id = "media_player.satellite"

    entity_registry = MagicMock()
    entity_registry.async_get.return_value = SimpleNamespace(
        area_id="living-room",
        device_id="satellite-device",
    )
    device_registry = MagicMock()
    device_registry.async_get.return_value = SimpleNamespace(area_id="office")
    area_registry = MagicMock()
    area_registry.async_get_area.return_value = SimpleNamespace(name="Living Room")

    ranked = RankedCandidate(
        candidate=Candidate(text="mute", intent_name="HassMediaPlayerMute"),
        scores=ScoreBreakdown(
            rapidfuzz_score=1.0,
            char_ngram_score=1.0,
            bm25_score=1.0,
            intent_score=1.0,
            final_score=1.0,
        ),
    )
    validation_ok_res = MagicMock()
    validation_ok_res.response.error_code = None

    with (
        patch(
            "custom_components.assist_canonicalizer.conversation.entity_registry.async_get",
            return_value=entity_registry,
        ),
        patch(
            "custom_components.assist_canonicalizer.conversation.device_registry.async_get",
            return_value=device_registry,
        ),
        patch(
            "custom_components.assist_canonicalizer.conversation.area_registry.async_get",
            return_value=area_registry,
        ),
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(ranked,),
        ) as rank,
        patch.object(entity, "_delegate_text", AsyncMock(return_value=validation_ok_res)),
    ):
        res = await entity._async_process_with_runtime(user_input)

    assert res is validation_ok_res
    area_registry.async_get_area.assert_called_once_with("living-room")
    device_registry.async_get.assert_not_called()
    assert rank.call_args.kwargs["intent_context"] == {
        "area": {"value": "Living Room", "text": "Living Room"}
    }


@pytest.mark.asyncio
async def test_exact_collision_delegates_text_to_hassil_with_original_context() -> None:
    """Delegate exact same-text collisions to HassIL without executing candidate metadata."""
    first = Candidate(
        text="all fan on",
        intent_name="HassGetState",
        language="en",
        metadata={"slots": '{"domain":"fan","state":"on"}'},
    )
    second = Candidate(
        text="all fan on",
        intent_name="HassTurnOn",
        language="en",
        metadata={"slots": '{"domain":"fan"}'},
    )

    for ordered_candidates in ((first, second), (second, first)):
        entry = MagicMock()
        entry.options = {"min_confidence": 0.60, "min_margin": 0.99}
        entry.entry_id = "test_entry"
        runtime = CanonicalizerRuntime()
        runtime.indexes["en"] = CanonicalIndex(language="en", candidates=ordered_candidates)
        entity = AssistCanonicalizerConversationEntity(entry, runtime)

        hass = MagicMock()
        hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
        entity.hass = hass
        context = Context()
        user_input = MockConversationInput("all fan on", "en", conversation_id="conv-exact")
        user_input.context = context
        user_input.device_id = "device-1"
        user_input.satellite_id = "assist_satellite.kitchen"
        user_input.extra_system_prompt = "stay local"
        hassil_result = MagicMock(name="hassil_result")
        hassil_result.response.error_code = None

        with (
            patch.object(
                entity,
                "_async_try_assist_pipeline_shortcut",
                AsyncMock(return_value=None),
            ),
            patch(
                "custom_components.assist_canonicalizer.conversation.conversation.async_converse",
                AsyncMock(return_value=hassil_result),
            ) as converse,
        ):
            result = await entity._async_process_with_runtime(user_input)

        assert result is hassil_result
        converse.assert_awaited_once_with(
            hass,
            "all fan on",
            "conv-exact",
            context,
            language="en",
            agent_id=HOME_ASSISTANT_AGENT,
            device_id="device-1",
            satellite_id="assist_satellite.kitchen",
            extra_system_prompt="stay local",
        )


@pytest.mark.asyncio
async def test_exact_collision_hassil_rejection_falls_back_to_raw_text() -> None:
    """Continue through validation failure and raw fallback when HassIL rejects exact text."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.99}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = CanonicalIndex(
        language="en",
        candidates=(
            Candidate(text="all fans off", intent_name="HassGetState", language="en"),
            Candidate(text="all fans off", intent_name="HassTurnOff", language="en"),
        ),
    )
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
    entity.hass = hass
    user_input = MockConversationInput("all fans off", "en")
    validation_error = MagicMock()
    validation_error.response.error_code = "no_intent_match"

    with (
        patch.object(
            entity,
            "_async_try_assist_pipeline_shortcut",
            AsyncMock(return_value=None),
        ),
        patch.object(
            entity,
            "_delegate_text",
            AsyncMock(side_effect=[validation_error, "raw_result"]),
        ) as delegate,
    ):
        result = await entity._async_process_with_runtime(user_input)

    assert result == "raw_result"
    assert delegate.await_args_list[0].args == ("all fans off", user_input)
    assert delegate.await_args_list[0].kwargs == {"primary": True}
    assert delegate.await_args_list[1].args == ("all fans off", user_input)
    assert delegate.await_args_list[1].kwargs == {"primary": False}
    assert runtime.diagnostics.last_fallback_reason == FallbackReason.VALIDATION_FAILED


def test_area_from_user_input_falls_back_to_original_device_area() -> None:
    """Use the request device area when a satellite entity has no area or device."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock()

    user_input = MockConversationInput("mute", "en")
    user_input.device_id = "device-1"
    user_input.satellite_id = "media_player.satellite"

    entity_registry = MagicMock()
    entity_registry.async_get.return_value = SimpleNamespace(area_id=None, device_id=None)
    device_registry = MagicMock()
    device_registry.async_get.return_value = SimpleNamespace(area_id="office")
    area = SimpleNamespace(name="Office")
    area_registry = MagicMock()
    area_registry.async_get_area.return_value = area

    with (
        patch(
            "custom_components.assist_canonicalizer.conversation.entity_registry.async_get",
            return_value=entity_registry,
        ),
        patch(
            "custom_components.assist_canonicalizer.conversation.device_registry.async_get",
            return_value=device_registry,
        ),
        patch(
            "custom_components.assist_canonicalizer.conversation.area_registry.async_get",
            return_value=area_registry,
        ),
    ):
        assert entity._area_from_user_input(user_input) is area

    device_registry.async_get.assert_called_once_with("device-1")
    area_registry.async_get_area.assert_called_once_with("office")


@pytest.mark.asyncio
async def test_validation_only_delegates_accepted_candidate() -> None:
    """Do not validate lower-ranked candidates that failed confidence gates."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=3)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
    entity.hass = hass

    accepted = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=1.0,
            final_score=0.9,
        ),
    )
    rejected = RankedCandidate(
        candidate=Candidate(text="turn off kitchen light", intent_name="HassTurnOff"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.1,
            char_ngram_score=0.1,
            bm25_score=0.1,
            intent_score=0.1,
            final_score=0.1,
        ),
    )
    validation_err_res = MagicMock()
    validation_err_res.response.error_code = intent.IntentResponseErrorCode.NO_INTENT_MATCH
    validation_ok_res = MagicMock()
    validation_ok_res.response.error_code = None
    user_input = MockConversationInput("turn kitchen light", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(accepted, rejected),
        ),
        patch.object(
            entity,
            "_delegate_text",
            AsyncMock(side_effect=[validation_err_res, validation_ok_res]),
        ) as mock_del_text,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")),
    ):
        res = await entity._async_process_with_runtime(user_input)

    assert res == "raw_delegated"
    mock_del_text.assert_awaited_once_with("turn on kitchen light", user_input, primary=True)
    assert runtime.diagnostics.last_fallback_reason == FallbackReason.VALIDATION_FAILED


@pytest.mark.asyncio
async def test_validation_tries_remaining_confident_ranked_candidates() -> None:
    """Re-gate one remaining candidate after HassIL reports no intent match."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=2)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
    entity.hass = hass

    accepted = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=1.0,
            final_score=0.9,
        ),
    )
    next_ranked = RankedCandidate(
        candidate=Candidate(text="turn on kitchen lamp", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.8,
            char_ngram_score=0.8,
            bm25_score=0.8,
            intent_score=1.0,
            final_score=0.8,
        ),
    )
    validation_err_res = MagicMock()
    validation_err_res.response.error_code = intent.IntentResponseErrorCode.NO_INTENT_MATCH
    validation_ok_res = MagicMock()
    validation_ok_res.response.error_code = None
    user_input = MockConversationInput("turn kitchen light", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(accepted, next_ranked),
        ),
        patch.object(
            entity,
            "_delegate_text",
            AsyncMock(side_effect=[validation_err_res, validation_ok_res]),
        ) as mock_del_text,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")) as raw,
    ):
        res = await entity._async_process_with_runtime(user_input)

    assert res == validation_ok_res
    assert mock_del_text.await_args_list[0].args == ("turn on kitchen light", user_input)
    assert mock_del_text.await_args_list[0].kwargs == {"primary": True}
    assert mock_del_text.await_args_list[1].args == ("turn on kitchen lamp", user_input)
    assert mock_del_text.await_args_list[1].kwargs == {"primary": True}
    raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_intent_match_reranks_cross_intent_candidate() -> None:
    """Recover a lower-ranked intent after the selected text has no HassIL match."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=2)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
    entity.hass = hass

    turn_on = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(0.9, 0.9, 0.9, 1.0, 0.9),
    )
    turn_off = RankedCandidate(
        candidate=Candidate(text="turn off kitchen light", intent_name="HassTurnOff"),
        scores=ScoreBreakdown(0.8, 0.8, 0.8, 1.0, 0.8),
    )
    no_match = MagicMock()
    no_match.response.error_code = intent.IntentResponseErrorCode.NO_INTENT_MATCH
    success = MagicMock()
    success.response.error_code = None
    user_input = MockConversationInput("turn kitchen light", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(turn_on, turn_off),
        ),
        patch.object(
            entity,
            "_delegate_text",
            AsyncMock(side_effect=[no_match, success]),
        ) as delegate,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw")) as raw,
    ):
        result = await entity._async_process_with_runtime(user_input)

    assert result == success
    assert [call.args[0] for call in delegate.await_args_list] == [
        "turn on kitchen light",
        "turn off kitchen light",
    ]
    assert all(call.kwargs == {"primary": True} for call in delegate.await_args_list)
    raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_supported_target_replaces_unsupported_higher_rank() -> None:
    """Execute the query-supported target without trying an unsupported leader."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=3)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
    entity.hass = hass

    selected = RankedCandidate(
        candidate=Candidate(
            text="set office light to 20",
            intent_name="HassLightSet",
            metadata={"slots": '{"name":"office light","brightness":20}'},
        ),
        scores=ScoreBreakdown(0.9, 0.9, 0.9, 1.0, 0.9),
    )
    correct_target = RankedCandidate(
        candidate=Candidate(
            text="set kitchen light to 50",
            intent_name="HassLightSet",
            metadata={"slots": '{"name":"kitchen light","brightness":50}'},
        ),
        scores=ScoreBreakdown(0.85, 0.85, 0.85, 1.0, 0.85),
    )
    compatible_target = RankedCandidate(
        candidate=Candidate(
            text="set living room light to 50",
            intent_name="HassLightSet",
            metadata={"slots": '{"name":"living room light","brightness":50}'},
        ),
        scores=ScoreBreakdown(0.8, 0.8, 0.8, 1.0, 0.8),
    )
    success = MagicMock()
    success.response.error_code = None
    user_input = MockConversationInput("set kitchen light to 50", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(selected, correct_target, compatible_target),
        ),
        patch(
            "custom_components.assist_canonicalizer.conversation.async_observe_delegated_text",
            AsyncMock(
                return_value=RecognitionObservation(
                    kind=RecognitionKind.INTENT,
                    intent_name="HassLightSet",
                )
            ),
        ),
        patch.object(
            entity,
            "_delegate_text",
            AsyncMock(return_value=success),
        ) as delegate,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw")) as raw,
    ):
        result = await entity._async_process_with_runtime(user_input)

    assert result == success
    assert [call.args[0] for call in delegate.await_args_list] == [
        "set kitchen light to 50",
    ]
    raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_valid_targets_recovers_after_unmatched_entity_recognition() -> None:
    """Retry once when re-recognition confirms pre-handler unmatched entities."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=2)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
    entity.hass = hass
    selected = RankedCandidate(
        candidate=Candidate(
            text="set kitchen light to 50",
            intent_name="HassLightSet",
            metadata={"slots": '{"name":"kitchen light","brightness":50}'},
        ),
        scores=ScoreBreakdown(0.9, 0.9, 0.9, 1.0, 0.9),
    )
    alternative = RankedCandidate(
        candidate=Candidate(
            text="turn on living room light",
            intent_name="HassTurnOn",
            metadata={"slots": '{"name":"living room light"}'},
        ),
        scores=ScoreBreakdown(0.8, 0.8, 0.8, 1.0, 0.8),
    )
    no_targets = MagicMock()
    no_targets.response.error_code = intent.IntentResponseErrorCode.NO_VALID_TARGETS
    success = MagicMock()
    success.response.error_code = None
    user_input = MockConversationInput("turn on living room light", "en")
    observations = iter(
        (
            RecognitionObservation(
                kind=RecognitionKind.INTENT,
                intent_name="HassLightSet",
            ),
            RecognitionObservation(
                kind=RecognitionKind.UNMATCHED_TARGET,
                intent_name="HassLightSet",
                unmatched_entities=("name",),
            ),
            RecognitionObservation(
                kind=RecognitionKind.INTENT,
                intent_name="HassTurnOn",
            ),
        )
    )

    async def observe(*_args: Any) -> RecognitionObservation:
        """Return the next recognition observation in the recovery sequence."""
        return next(observations)

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(selected, alternative),
        ),
        patch.object(
            entity,
            "_delegate_text",
            AsyncMock(side_effect=[no_targets, success]),
        ) as delegate,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw")) as raw,
        patch(
            "custom_components.assist_canonicalizer.conversation.async_observe_delegated_text",
            side_effect=observe,
        ) as observe_live,
    ):
        result = await entity._async_process_with_runtime(user_input)

    assert result == success
    assert [call.args[0] for call in delegate.await_args_list] == [
        "set kitchen light to 50",
        "turn on living room light",
    ]
    raw.assert_not_awaited()
    assert observe_live.await_count == 3
    assert user_input.text == "turn on living room light"


@pytest.mark.asyncio
async def test_no_valid_targets_falls_back_after_handler_target_failure() -> None:
    """Do not retry when re-recognition shows HassIL reached target handling."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=2)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
    entity.hass = hass
    selected = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(0.9, 0.9, 0.9, 1.0, 0.9),
    )
    alternative = RankedCandidate(
        candidate=Candidate(text="turn on living room light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(0.8, 0.8, 0.8, 1.0, 0.8),
    )
    no_targets = MagicMock()
    no_targets.response.error_code = intent.IntentResponseErrorCode.NO_VALID_TARGETS
    user_input = MockConversationInput("turn on the light", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(selected, alternative),
        ),
        patch.object(entity, "_delegate_text", AsyncMock(return_value=no_targets)) as delegate,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw")) as raw,
        patch(
            "custom_components.assist_canonicalizer.conversation.async_observe_delegated_text",
            AsyncMock(
                side_effect=(
                    RecognitionObservation(
                        kind=RecognitionKind.INTENT,
                        intent_name="HassTurnOn",
                    ),
                    RecognitionObservation(
                        kind=RecognitionKind.INTENT,
                        intent_name="HassTurnOn",
                    ),
                )
            ),
        ) as observe_live,
    ):
        result = await entity._async_process_with_runtime(user_input)

    assert result == "raw"
    delegate.assert_awaited_once_with("turn on kitchen light", user_input, primary=True)
    assert observe_live.await_count == 2
    raw.assert_awaited_once_with(user_input)


@pytest.mark.asyncio
async def test_unmatched_entity_provenance_check_fails_closed(
    conversation_entity: AssistCanonicalizerConversationEntity,
) -> None:
    """Treat unavailable or failed side-effect-free recognition as indeterminate."""
    user_input = MockConversationInput("turn on the light", "en")

    for observation in (
        RecognitionObservation(kind=RecognitionKind.ERROR, error_category="agent_unavailable"),
        RecognitionObservation(kind=RecognitionKind.ERROR, error_category="recognition_failed"),
        RecognitionObservation(kind=RecognitionKind.NO_MATCH),
    ):
        with patch(
            "custom_components.assist_canonicalizer.conversation.async_observe_delegated_text",
            AsyncMock(return_value=observation),
        ):
            assert not await conversation_entity._async_has_unmatched_entities(
                "turn on kitchen light", user_input
            )


@pytest.mark.parametrize(
    "error_code",
    [
        intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
        intent.IntentResponseErrorCode.UNKNOWN,
        "future_error_code",
    ],
)
@pytest.mark.asyncio
async def test_nonretryable_hassil_errors_do_not_execute_another_candidate(
    error_code: object,
) -> None:
    """Do not retry after errors that may follow handler side effects."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=2)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
    entity.hass = hass
    selected = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(0.9, 0.9, 0.9, 1.0, 0.9),
    )
    remaining = RankedCandidate(
        candidate=Candidate(text="turn on kitchen lamp", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(0.8, 0.8, 0.8, 1.0, 0.8),
    )
    failed = MagicMock()
    failed.response.error_code = error_code
    user_input = MockConversationInput("turn kitchen light", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(selected, remaining),
        ),
        patch.object(entity, "_delegate_text", AsyncMock(return_value=failed)) as delegate,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw")) as raw,
    ):
        result = await entity._async_process_with_runtime(user_input)

    assert result == "raw"
    delegate.assert_awaited_once_with("turn on kitchen light", user_input, primary=True)
    raw.assert_awaited_once_with(user_input)


@pytest.mark.asyncio
async def test_candidate_recovery_is_capped_at_one_additional_hassil_call() -> None:
    """Fall back after two failed primary executions without trying a third candidate."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=3)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
    entity.hass = hass
    ranked = tuple(
        RankedCandidate(
            candidate=Candidate(text=text, intent_name="HassTurnOn"),
            scores=ScoreBreakdown(score, score, score, 1.0, score),
        )
        for text, score in (
            ("turn on kitchen light", 0.9),
            ("turn on kitchen lamp", 0.8),
            ("switch on kitchen light", 0.7),
        )
    )
    no_match = MagicMock()
    no_match.response.error_code = intent.IntentResponseErrorCode.NO_INTENT_MATCH
    user_input = MockConversationInput("turn kitchen light", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=ranked,
        ),
        patch.object(
            entity,
            "_delegate_text",
            AsyncMock(side_effect=[no_match, no_match]),
        ) as delegate,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw")) as raw,
    ):
        result = await entity._async_process_with_runtime(user_input)

    assert result == "raw"
    assert [call.args[0] for call in delegate.await_args_list] == [
        "turn on kitchen light",
        "turn on kitchen lamp",
    ]
    raw.assert_awaited_once_with(user_input)


@pytest.mark.asyncio
async def test_recovery_handler_error_routes_to_fallback_agent() -> None:
    """Route the original query to the fallback agent after recovery fails."""
    entry = MagicMock(entry_id="test_entry")
    entry.options = {
        "min_confidence": 0.60,
        "min_margin": 0.05,
        CONF_FALLBACK_AGENT_ID: "llm_agent",
    }
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=2)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock(
        async_add_executor_job=AsyncMock(side_effect=lambda target, *args: target(*args))
    )
    ranked = (
        RankedCandidate(
            Candidate(text="turn on kitchen light", intent_name="HassTurnOn"),
            ScoreBreakdown(0.9, 0.9, 0.9, 1.0, 0.9),
        ),
        RankedCandidate(
            Candidate(text="turn on kitchen lamp", intent_name="HassTurnOn"),
            ScoreBreakdown(0.8, 0.8, 0.8, 1.0, 0.8),
        ),
    )
    no_match = MagicMock()
    no_match.response.error_code = intent.IntentResponseErrorCode.NO_INTENT_MATCH
    failed = MagicMock()
    failed.response.error_code = intent.IntentResponseErrorCode.FAILED_TO_HANDLE
    fallback_result = MagicMock()
    fallback_result.response.error_code = None
    user_input = MockConversationInput("turn kitchen light", "en")

    with (
        patch.object(CanonicalizerRuntime, "rank_with_dynamic_candidates", return_value=ranked),
        patch(
            "homeassistant.components.conversation.async_converse",
            AsyncMock(side_effect=[no_match, failed, fallback_result]),
        ) as converse,
    ):
        result = await entity._async_process_with_runtime(user_input)

    assert result is fallback_result
    assert [call.args[1] for call in converse.await_args_list] == [
        "turn on kitchen light",
        "turn on kitchen lamp",
        user_input.text,
    ]
    assert [call.kwargs["agent_id"] for call in converse.await_args_list] == [
        HOME_ASSISTANT_AGENT,
        HOME_ASSISTANT_AGENT,
        "llm_agent",
    ]


@pytest.mark.asyncio
async def test_candidate_recovery_skips_normalized_duplicate_text(
    conversation_entity: AssistCanonicalizerConversationEntity,
) -> None:
    """Do not send the same delegated command twice during recovery."""
    selected = RankedCandidate(
        candidate=Candidate(text="Turn on kitchen light!", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(0.9, 0.9, 0.9, 1.0, 0.9),
    )
    duplicate = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(0.85, 0.85, 0.85, 1.0, 0.85),
    )
    distinct = RankedCandidate(
        candidate=Candidate(text="switch on kitchen light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(0.8, 0.8, 0.8, 1.0, 0.8),
    )
    user_input = MockConversationInput("turn kitchen light", "en")

    recovery = await conversation_entity._async_ranked_recovery_candidate(
        (selected, duplicate, distinct),
        selected,
        "Turn on kitchen light!",
        intent.IntentResponseErrorCode.NO_INTENT_MATCH.value,
        0.60,
        0.05,
        user_input,
    )

    assert recovery is not None
    assert recovery[0] is distinct
    assert recovery[1] == "switch on kitchen light"


@pytest.mark.asyncio
async def test_live_preflight_skips_duplicate_text_and_allows_metadata_divergence() -> None:
    """Execute a lower distinct text when live recognition rejects the lexical winner."""
    entry = MagicMock(entry_id="test_entry")
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=3)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock(
        async_add_executor_job=AsyncMock(side_effect=lambda target, *args: target(*args))
    )
    top = RankedCandidate(
        Candidate(text="turn on ghost lamp", intent_name="IntentGeneratedTop"),
        ScoreBreakdown(0.90, 0.90, 0.90, 0.90, 0.90),
    )
    duplicate = RankedCandidate(
        Candidate(text="Turn on ghost lamp!", intent_name="IntentDuplicateMetadata"),
        ScoreBreakdown(0.85, 0.85, 0.85, 0.85, 0.85),
    )
    lower = RankedCandidate(
        Candidate(
            text="turn on living room lamp",
            intent_name="IntentGeneratedLower",
            metadata={"slots": '{"name":"generated name"}'},
        ),
        ScoreBreakdown(0.80, 0.80, 0.80, 0.80, 0.80),
    )
    observations = (
        RecognitionObservation(kind=RecognitionKind.NO_MATCH),
        RecognitionObservation(
            kind=RecognitionKind.INTENT,
            intent_name="HassTurnOn",
            slots=(("name", "Living Room Lamp"),),
        ),
    )
    success = MagicMock()
    success.response.error_code = None
    user_input = MockConversationInput("turn teh lamp on", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(top, duplicate, lower),
        ),
        patch(
            "custom_components.assist_canonicalizer.conversation.async_observe_delegated_text",
            AsyncMock(side_effect=observations),
        ) as observe,
        patch.object(entity, "_delegate_text", AsyncMock(return_value=success)) as delegate,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw")) as raw,
    ):
        result = await entity._async_process_with_runtime(user_input)

    assert result is success
    assert [call.args[2] for call in observe.await_args_list] == [
        "turn on ghost lamp",
        "turn on living room lamp",
    ]
    delegate.assert_awaited_once_with("turn on living room lamp", user_input, primary=True)
    raw.assert_not_awaited()
    assert runtime.diagnostics.preflight_attempt_count == 2
    assert runtime.diagnostics.metadata_diverged
    assert runtime.diagnostics.metadata_divergence_reason == "intent_and_slots"
    assert runtime.diagnostics.recognition_intent == "HassTurnOn"


@pytest.mark.asyncio
async def test_live_preflight_prefers_same_intent_surface_with_aligned_recognition() -> None:
    """Prefer an alternate wording when it preserves and validates the ranked intent."""
    entry = MagicMock(entry_id="test_entry")
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=3)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock(
        async_add_executor_job=AsyncMock(side_effect=lambda target, *args: target(*args))
    )
    ambiguous_surface = RankedCandidate(
        Candidate(text="bedroom light on", intent_name="HassTurnOn"),
        ScoreBreakdown(0.90, 0.90, 0.90, 0.90, 0.90),
    )
    unrelated_intent = RankedCandidate(
        Candidate(text="turn off bedroom light", intent_name="HassTurnOff"),
        ScoreBreakdown(0.84, 0.84, 0.84, 0.84, 0.84),
    )
    explicit_surface = RankedCandidate(
        Candidate(text="turn on bedroom light", intent_name="HassTurnOn"),
        ScoreBreakdown(0.82, 0.82, 0.82, 0.82, 0.82),
    )
    observations = (
        RecognitionObservation(
            kind=RecognitionKind.INTENT,
            intent_name="HassLightSet",
        ),
        RecognitionObservation(
            kind=RecognitionKind.INTENT,
            intent_name="HassTurnOn",
        ),
    )
    success = MagicMock()
    success.response.error_code = None
    user_input = MockConversationInput("turn bedrom light", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(ambiguous_surface, unrelated_intent, explicit_surface),
        ),
        patch(
            "custom_components.assist_canonicalizer.conversation.async_observe_delegated_text",
            AsyncMock(side_effect=observations),
        ) as observe,
        patch.object(entity, "_delegate_text", AsyncMock(return_value=success)) as delegate,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw")) as raw,
    ):
        result = await entity._async_process_with_runtime(user_input)

    assert result is success
    assert [call.args[2] for call in observe.await_args_list] == [
        "bedroom light on",
        "turn on bedroom light",
    ]
    delegate.assert_awaited_once_with("turn on bedroom light", user_input, primary=True)
    raw.assert_not_awaited()
    assert runtime.diagnostics.preflight_attempt_count == 2
    assert not runtime.diagnostics.metadata_diverged
    assert runtime.diagnostics.recognition_intent == "HassTurnOn"


@pytest.mark.asyncio
async def test_live_preflight_stops_after_three_distinct_texts() -> None:
    """Fail closed after the configured three-text recognition budget."""
    entry = MagicMock(entry_id="test_entry")
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=4)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock(
        async_add_executor_job=AsyncMock(side_effect=lambda target, *args: target(*args))
    )
    ranked = tuple(
        RankedCandidate(
            Candidate(text=f"command {index}", intent_name="IntentSame"),
            ScoreBreakdown(score, score, score, score, score),
        )
        for index, score in enumerate((0.90, 0.85, 0.80, 0.75), start=1)
    )
    user_input = MockConversationInput("raw original", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=ranked,
        ),
        patch(
            "custom_components.assist_canonicalizer.conversation.async_observe_delegated_text",
            AsyncMock(return_value=RecognitionObservation(kind=RecognitionKind.NO_MATCH)),
        ) as observe,
        patch.object(entity, "_delegate_text", AsyncMock()) as delegate,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw")) as raw,
    ):
        result = await entity._async_process_with_runtime(user_input)

    assert result == "raw"
    assert observe.await_count == 3
    delegate.assert_not_awaited()
    raw.assert_awaited_once_with(user_input)
    assert runtime.diagnostics.preflight_attempt_count == 3


@pytest.mark.asyncio
async def test_live_preflight_sentence_trigger_uses_raw_fallback() -> None:
    """Never execute a fuzzy rewrite that newly matches a sentence trigger."""
    entry = MagicMock(entry_id="test_entry")
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=1)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock(
        async_add_executor_job=AsyncMock(side_effect=lambda target, *args: target(*args))
    )
    ranked = (
        RankedCandidate(
            Candidate(text="start bedtime", intent_name="GeneratedIntent"),
            ScoreBreakdown(0.90, 0.90, 0.90, 0.90, 0.90),
        ),
    )
    user_input = MockConversationInput("stert bedtime", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=ranked,
        ),
        patch(
            "custom_components.assist_canonicalizer.conversation.async_observe_delegated_text",
            AsyncMock(return_value=RecognitionObservation(kind=RecognitionKind.SENTENCE_TRIGGER)),
        ),
        patch.object(entity, "_delegate_text", AsyncMock()) as delegate,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw")) as raw,
    ):
        result = await entity._async_process_with_runtime(user_input)

    assert result == "raw"
    delegate.assert_not_awaited()
    raw.assert_awaited_once_with(user_input)
    assert runtime.diagnostics.recognition_kind == "sentence_trigger"


@pytest.mark.asyncio
async def test_candidate_recovery_reapplies_margin_gate(
    conversation_entity: AssistCanonicalizerConversationEntity,
) -> None:
    """Reject recovery when the remaining top intents are still ambiguous."""
    selected = RankedCandidate(
        candidate=Candidate(text="first command", intent_name="IntentA"),
        scores=ScoreBreakdown(0.9, 0.9, 0.9, 1.0, 0.9),
    )
    remaining = RankedCandidate(
        candidate=Candidate(text="second command", intent_name="IntentB"),
        scores=ScoreBreakdown(0.8, 0.8, 0.8, 1.0, 0.8),
    )
    competitor = RankedCandidate(
        candidate=Candidate(text="third command", intent_name="IntentC"),
        scores=ScoreBreakdown(0.78, 0.78, 0.78, 1.0, 0.78),
    )

    recovery = await conversation_entity._async_ranked_recovery_candidate(
        (selected, remaining, competitor),
        selected,
        "first command",
        intent.IntentResponseErrorCode.NO_INTENT_MATCH.value,
        0.60,
        0.05,
        MockConversationInput("ambiguous command", "en"),
    )

    assert recovery is None


@pytest.mark.asyncio
async def test_validation_delegate_exception_does_not_retry_candidates() -> None:
    """Do not execute another candidate after an indeterminate delegate exception."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=2)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
    entity.hass = hass

    accepted = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=1.0,
            final_score=0.9,
        ),
    )
    next_ranked = RankedCandidate(
        candidate=Candidate(text="turn on kitchen lamp", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.8,
            char_ngram_score=0.8,
            bm25_score=0.8,
            intent_score=1.0,
            final_score=0.8,
        ),
    )
    user_input = MockConversationInput("turn kitchen light", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(accepted, next_ranked),
        ),
        patch.object(
            entity,
            "_delegate_text",
            AsyncMock(side_effect=RuntimeError("primary failed")),
        ) as mock_del_text,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")) as raw,
    ):
        res = await entity._async_process_with_runtime(user_input)

    assert res == "raw_delegated"
    assert mock_del_text.await_args_list[0].args == ("turn on kitchen light", user_input)
    assert mock_del_text.await_args_list[0].kwargs == {"primary": True}
    assert len(mock_del_text.await_args_list) == 1
    raw.assert_awaited_once_with(user_input)
    assert runtime.diagnostics.last_error == "primary failed"


@pytest.mark.asyncio
async def test_validation_delegate_exception_fallback_keeps_error() -> None:
    """Keep a candidate execution exception when falling back without retrying."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock(candidate_count=2)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))
    entity.hass = hass

    accepted = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=1.0,
            final_score=0.9,
        ),
    )
    next_ranked = RankedCandidate(
        candidate=Candidate(text="turn on kitchen lamp", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.8,
            char_ngram_score=0.8,
            bm25_score=0.8,
            intent_score=1.0,
            final_score=0.8,
        ),
    )
    user_input = MockConversationInput("turn kitchen light", "en")

    with (
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=(accepted, next_ranked),
        ),
        patch.object(
            entity,
            "_delegate_text",
            AsyncMock(side_effect=RuntimeError("primary failed")),
        ) as mock_del_text,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")) as raw,
    ):
        res = await entity._async_process_with_runtime(user_input)

    assert res == "raw_delegated"
    assert mock_del_text.await_args_list[0].args == ("turn on kitchen light", user_input)
    assert mock_del_text.await_args_list[0].kwargs == {"primary": True}
    assert len(mock_del_text.await_args_list) == 1
    raw.assert_awaited_once_with(user_input)
    assert runtime.diagnostics.last_fallback_reason == FallbackReason.VALIDATION_FAILED
    assert runtime.diagnostics.last_error == "primary failed"


@pytest.mark.asyncio
async def test_conversation_resolves_updated_options_without_reload() -> None:
    """Resolve every current option from the live config entry for each request."""
    entry = MagicMock()
    entry.entry_id = "this_agent"
    entry.data = {}
    entry.options = {
        CONF_FALLBACK_AGENT_ID: "first_agent",
        CONF_MIN_CONFIDENCE: 0.60,
        CONF_MIN_MARGIN: 0.05,
    }
    entity = AssistCanonicalizerConversationEntity(entry, CanonicalizerRuntime())
    user_input = MockConversationInput("turn on the light", "en")
    decision = MagicMock()

    with patch.object(
        entity,
        "_async_rank_user_input",
        AsyncMock(return_value=((), decision)),
    ) as rank:
        first_request = await entity._async_rank_request(user_input, "en", MagicMock())
        assert entity._fallback_agent_id("default") == "first_agent"

        entry.options = {
            CONF_FALLBACK_AGENT_ID: "second_agent",
            CONF_MIN_CONFIDENCE: 0.80,
            CONF_MIN_MARGIN: 0.10,
        }
        second_request = await entity._async_rank_request(user_input, "en", MagicMock())

    assert first_request is not None
    assert (first_request.min_confidence, first_request.min_margin) == (0.60, 0.05)
    assert second_request is not None
    assert (second_request.min_confidence, second_request.min_margin) == (0.80, 0.10)
    assert entity._fallback_agent_id("default") == "second_agent"
    assert [(call.args[-2], call.args[-1]) for call in rank.await_args_list] == [
        (0.60, 0.05),
        (0.80, 0.10),
    ]


@pytest.mark.asyncio
async def test_conversation_delegate_and_fallback_agent_logic() -> None:
    """Test delegation and fallback agent ID selection branch options."""
    entry = MagicMock()
    entry.entry_id = "this_agent"
    entry.options = cast(dict[str, Any], {CONF_MIN_CONFIDENCE: 0.60, CONF_MIN_MARGIN: 0.05})
    entry.data = {}
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()

    hass.async_add_executor_job = AsyncMock(side_effect=_executor_job_returning_empty_build_index)
    entity.hass = hass

    user_input = MockConversationInput("tắt đèn bếp", "vi")

    # 1. Test fallback_agent_id with options
    entry.options[CONF_FALLBACK_AGENT_ID] = "options_agent"
    assert entity._fallback_agent_id("default") == "options_agent"

    # 2. Test fallback_agent_id with options matching entry_id (self-forwarding prevention)
    entry.options[CONF_FALLBACK_AGENT_ID] = "this_agent"
    assert entity._fallback_agent_id("default") == "default"

    # 3. Test fallback_agent_id with invalid type
    entry.options[CONF_FALLBACK_AGENT_ID] = 123
    assert entity._fallback_agent_id("default") == "default"

    # 4. Test fallback_agent_id fallback to data
    entry.options = {}
    entry.data[CONF_FALLBACK_AGENT_ID] = "data_agent"
    assert entity._fallback_agent_id("default") == "data_agent"

    # 5. Test actual delegate_raw_text calls conversation.async_converse
    entry.options = {CONF_FALLBACK_AGENT_ID: "options_agent"}
    mock_result = MagicMock()
    with patch(
        "homeassistant.components.conversation.async_converse", AsyncMock(return_value=mock_result)
    ) as mock_converse:
        res = await entity._delegate_raw_text(user_input)
        assert res is mock_result
        mock_converse.assert_called_once_with(
            entity.hass,
            "tắt đèn bếp",
            user_input.conversation_id,
            user_input.context,
            language=user_input.language,
            agent_id="options_agent",
            device_id=user_input.device_id,
            satellite_id=None,
            extra_system_prompt=None,
        )

    # 6. Test async_process success path executes and updates diagnostics
    runtime.indexes["vi"] = MagicMock(candidate_count=5)
    rc = RankedCandidate(
        candidate=Candidate(text="tắt đèn bếp", intent_name="HassTurnOff"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=1.0,
            final_score=0.9,
        ),
    )
    validation_ok_res = MagicMock()
    validation_ok_res.response.error_code = None

    with (
        patch.object(CanonicalizerRuntime, "rank_with_dynamic_candidates", return_value=(rc,)),
        patch.object(entity, "_delegate_text", AsyncMock(return_value=validation_ok_res)),
    ):
        res = await entity.async_process(user_input)
        assert res is validation_ok_res
        assert runtime.diagnostics.last_fallback_reason is None

    # 7. Test async_prepare empty/None language branch
    await entity.async_prepare(None)


@pytest.mark.asyncio
async def test_async_process_prefer_local_intents_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test prefer_local_intents shortcut behavior when it is False."""
    _mock_assist_pipeline_const(monkeypatch)

    class DummyPipeline:
        """Dummy pipeline class for testing."""

        prefer_local_intents = False

    entry = MagicMock(entry_id="test-entry")
    entry.options = {}
    entry.data = {}
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock()
    user_input = MockConversationInput("tắt đèn bếp", "vi")
    entity.hass.data = {
        "assist_pipeline": _active_pipeline_data(user_input.context, DummyPipeline())
    }
    validation_ok_res = MagicMock()
    validation_ok_res.response.error_code = None

    with (
        patch.object(
            entity, "_delegate_text", AsyncMock(return_value=validation_ok_res)
        ) as mock_delegate,
        patch.object(CanonicalizerRuntime, "rank_with_dynamic_candidates") as mock_rank,
    ):
        res = await entity.async_process(user_input)
        assert res is validation_ok_res
        mock_delegate.assert_called_once_with(
            "tắt đèn bếp",
            user_input,
            primary=True,
        )
        mock_rank.assert_not_called()


@pytest.mark.asyncio
async def test_async_process_prefer_local_intents_true_uses_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep HassIL first when Home Assistant filters its local-intent pre-pass."""
    _mock_assist_pipeline_const(monkeypatch)

    class DummyPipeline:
        """Dummy pipeline class for testing."""

        prefer_local_intents = True

    entry = MagicMock(entry_id="test-entry")
    entry.options = {}
    entry.data = {}
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock()

    user_input = MockConversationInput("tắt đèn bếp", "vi")
    entity.hass.data = {
        "assist_pipeline": _active_pipeline_data(user_input.context, DummyPipeline())
    }
    validation_ok_res = MagicMock()
    validation_ok_res.response.error_code = None

    with (
        patch.object(
            entity, "_delegate_text", AsyncMock(return_value=validation_ok_res)
        ) as mock_delegate,
        patch.object(CanonicalizerRuntime, "rank_with_dynamic_candidates") as mock_rank,
    ):
        res = await entity.async_process(user_input)
        assert res is validation_ok_res
        mock_delegate.assert_called_once_with(
            "tắt đèn bếp",
            user_input,
            primary=True,
        )
        mock_rank.assert_not_called()


@pytest.mark.parametrize(
    "simulate_exception",
    [False, True],
)
@pytest.mark.asyncio
async def test_async_process_shortcut_restores_chat_log(
    monkeypatch: pytest.MonkeyPatch,
    simulate_exception: bool,
) -> None:
    """Test that the chat log is restored if the shortcut path fails or raises an exception."""
    _mock_assist_pipeline_const(monkeypatch)

    class DummyPipeline:
        """Dummy pipeline class for testing."""

        prefer_local_intents = False

    entry = MagicMock(entry_id="test-entry")
    entry.options = {}
    entry.data = {}
    runtime = CanonicalizerRuntime()
    mock_index = MagicMock()
    mock_index.language = "vi"
    runtime.set_index(mock_index)
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock()

    mock_user_message = MagicMock()
    mock_user_message.role = "user"
    mock_user_message.content = "tắt đèn bếp"

    mock_chat_log = MagicMock()
    mock_chat_log.content = [mock_user_message]

    entity.hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))

    user_input = MockConversationInput("tắt đèn bếp", "vi", conversation_id="conv-1")
    entity.hass.data = {
        "assist_pipeline": _active_pipeline_data(user_input.context, DummyPipeline())
    }

    # Shortcut result has an error
    shortcut_res = MagicMock()
    shortcut_res.response.error_code = "intent-failed"

    # Fallback validation succeeds
    validation_ok_res = MagicMock()
    validation_ok_res.response.error_code = None

    with (
        patch.object(
            entity,
            "_get_active_chat_log",
            return_value=mock_chat_log,
        ),
        patch.object(
            entity,
            "_delegate_text",
            AsyncMock(),
        ) as mock_delegate,
        patch.object(
            CanonicalizerRuntime,
            "rank_with_dynamic_candidates",
            return_value=[
                RankedCandidate(
                    candidate=Candidate(
                        text="tắt đèn bếp",
                        intent_name="HassTurnOff",
                        slot_values=(),
                        language="vi",
                    ),
                    scores=ScoreBreakdown(1.0, 1.0, 1.0, 1.0, 1.0),
                )
            ],
        ),
    ):
        call_count = 0

        def side_effect(*args, **kwargs):
            """Side effect helper to simulate shortcut failure followed by success."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                mock_chat_log.content.append(MagicMock(role="assistant", content="error response"))
                if simulate_exception:
                    raise RuntimeError("Shortcut failed")
                return shortcut_res
            return validation_ok_res

        mock_delegate.side_effect = side_effect

        res = await entity.async_process(user_input)
        assert res is validation_ok_res

        # Verify that mock_chat_log.content was restored to the original snapshot
        assert mock_chat_log.content == [mock_user_message]


@pytest.mark.parametrize(
    "simulate_exception",
    [False, True],
)
@pytest.mark.asyncio
async def test_async_execute_ranked_candidate_restores_chat_log(
    simulate_exception: bool,
) -> None:
    """Restore the chat log when candidate execution fails or raises an exception."""
    entry = MagicMock()
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock()

    # Create mock chat log
    mock_user_message = MagicMock()
    mock_user_message.role = "user"
    mock_user_message.content = "original query"

    mock_chat_log = MagicMock()
    mock_chat_log.content = [mock_user_message]

    user_input = MockConversationInput("original query", "vi", conversation_id="conv-1")

    ranked_candidate = RankedCandidate(
        candidate=Candidate(
            text="tắt đèn bếp",
            intent_name="HassTurnOff",
            slot_values=(),
            language="vi",
        ),
        scores=ScoreBreakdown(1.0, 1.0, 1.0, 1.0, 1.0),
    )

    # Validation result has an error
    validation_res = MagicMock()
    validation_res.response.error_code = "intent-failed"

    with (
        patch.object(
            entity,
            "_get_active_chat_log",
            return_value=mock_chat_log,
        ),
        patch.object(
            entity,
            "_delegate_text",
            AsyncMock(),
        ) as mock_delegate,
    ):

        def side_effect(*args, **kwargs):
            """Side effect helper to simulate validation outcome."""
            # During _delegate_text execution, a user message and
            # assistant message get appended to the chat log
            mock_chat_log.content.extend(
                [
                    MagicMock(role="user", content="tắt đèn bếp"),
                    MagicMock(role="assistant", content="error response"),
                ]
            )
            if simulate_exception:
                raise RuntimeError("Validation exception")
            return validation_res

        mock_delegate.side_effect = side_effect

        res = await entity._async_execute_ranked_candidate(ranked_candidate, user_input)
        if simulate_exception:
            assert res is None
        else:
            assert res is validation_res

        # Verify that mock_chat_log.content was restored (keeping only the original user message)
        assert mock_chat_log.content == [mock_user_message]


@pytest.mark.asyncio
async def test_async_execute_rehydrates_from_candidate_metadata_with_cold_cache() -> None:
    """Rehydrate execution text without loading intents on the event loop."""
    entry = MagicMock()
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock()
    user_input = MockConversationInput("add milk to shopping list", "en")
    candidate = Candidate(
        text="add shopping_list_item to shopping list",
        intent_name="HassShoppingListAddItem",
        language="en",
        metadata={
            "sentence_template": "add {shopping_list_item} to shopping list",
            "wildcard_slots": "shopping_list_item",
        },
    )
    ranked = RankedCandidate(
        candidate=candidate,
        scores=ScoreBreakdown(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    validation_result = MagicMock()
    validation_result.response.error_code = None

    wildcard_slot_names.cache_clear("en")
    try:
        with (
            patch.object(
                entity,
                "_delegate_text",
                AsyncMock(return_value=validation_result),
            ) as delegate,
            patch("home_assistant_intents.get_intents") as get_intents,
        ):
            result = await entity._async_execute_ranked_candidate(ranked, user_input)

        assert result is validation_result
        delegate.assert_awaited_once_with(
            "add milk to shopping list",
            user_input,
            primary=True,
        )
        get_intents.assert_not_called()
    finally:
        wildcard_slot_names.cache_clear("en")


class DummyChatLog:
    """Dummy chat log class for testing."""

    def __init__(self, delta_listener: Any = None) -> None:
        """Initialize."""
        self.delta_listener = delta_listener


@pytest.fixture
def conversation_entity() -> AssistCanonicalizerConversationEntity:
    """Fixture to provide an AssistCanonicalizerConversationEntity instance."""
    entry = MagicMock()
    runtime = CanonicalizerRuntime()
    return AssistCanonicalizerConversationEntity(entry, runtime)


@pytest.mark.asyncio
async def test_capture_chat_log_deltas_concurrent(
    conversation_entity: AssistCanonicalizerConversationEntity,
) -> None:
    """Test that delta capturing is task-safe under concurrent tasks."""
    chat_log = DummyChatLog()

    task_1_entered = asyncio.Event()
    task_2_entered = asyncio.Event()

    async def task_1() -> list[dict]:
        """First task that captures deltas."""
        with conversation_entity._capture_chat_log_deltas(chat_log) as deltas:
            task_1_entered.set()
            await task_2_entered.wait()
            if chat_log.delta_listener:
                chat_log.delta_listener(chat_log, {"text": "task1"})
            return deltas

    async def task_2() -> list[dict]:
        """Second task that captures deltas."""
        await task_1_entered.wait()
        with conversation_entity._capture_chat_log_deltas(chat_log) as deltas:
            task_2_entered.set()
            if chat_log.delta_listener:
                chat_log.delta_listener(chat_log, {"text": "task2"})
            return deltas

    res_1, res_2 = await asyncio.gather(task_1(), task_2())

    assert res_1 == [{"text": "task1"}]
    assert res_2 == [{"text": "task2"}]
    assert chat_log.delta_listener is None


@pytest.mark.asyncio
async def test_delta_capture_realtime_silence_and_playback(
    conversation_entity: AssistCanonicalizerConversationEntity,
) -> None:
    """Test that live deltas are captured and not leaked.

    Deltas should not leak to original listener in real-time, then correctly played back.
    """
    orig_called = []

    def orig_listener(log: Any, delta: dict) -> None:
        """Original listener callback."""
        orig_called.append(delta)

    chat_log = DummyChatLog(orig_listener)

    with conversation_entity._capture_chat_log_deltas(chat_log) as deltas:
        # Simulate delta generated during capture
        assert chat_log.delta_listener is not orig_listener
        chat_log.delta_listener(chat_log, {"text": "hello"})

        # Verify that original listener has not received it yet
        assert not orig_called
        assert deltas == [{"text": "hello"}]

    # Play back deltas
    conversation_entity._play_back_deltas(chat_log, deltas)

    # Verify that original listener receives the delta exactly once
    assert orig_called == [{"text": "hello"}]


@pytest.mark.asyncio
async def test_delta_capture_nested_playback_safety(
    conversation_entity: AssistCanonicalizerConversationEntity,
) -> None:
    """Test delta playback on nested/concurrent wrappers.

    Ensures it resolves directly to the original listener.
    """
    orig_called = []

    def orig_listener(log: Any, delta: dict) -> None:
        """Original listener callback."""
        orig_called.append(delta)

    chat_log = DummyChatLog(orig_listener)

    # Task 1 (outer) captures deltas
    with conversation_entity._capture_chat_log_deltas(chat_log) as outer_deltas:
        # Task 2 (inner/nested in same task context) captures deltas
        with conversation_entity._capture_chat_log_deltas(chat_log) as inner_deltas:
            # Trigger delta
            chat_log.delta_listener(chat_log, {"text": "inner"})
            # Verify inner captures it
            assert inner_deltas == [{"text": "inner"}]
            # Verify outer has not captured it directly (since ContextVar is active)
            assert not outer_deltas
            # Verify original listener is silent
            assert not orig_called

        # Now back in outer context, inner context is exited.
        # Play back inner deltas. The delta_listener is still the wrapper.
        conversation_entity._play_back_deltas(chat_log, inner_deltas)

        # Verify that original listener has received the played back delta
        assert orig_called == [{"text": "inner"}]

        # Trigger delta in outer context
        chat_log.delta_listener(chat_log, {"text": "outer"})
        assert outer_deltas == [{"text": "outer"}]

    # Play back outer deltas
    conversation_entity._play_back_deltas(chat_log, outer_deltas)

    # Verify original listener receives the outer delta too, exactly once
    assert orig_called == [{"text": "inner"}, {"text": "outer"}]


@pytest.mark.asyncio
async def test_discover_pipeline_languages_fallback() -> None:
    """Test fallback logic in pipeline language discovery."""
    mock_hass = MagicMock()
    original = assist_canonicalizer.async_get_pipelines
    assist_canonicalizer.async_get_pipelines = assist_canonicalizer._UNINITIALIZED
    try:
        # In the test environment assist_pipeline is not installed, so the import
        # inside _discover_pipeline_languages raises ImportError and the function
        # sets async_get_pipelines = None and returns an empty set.
        langs = _discover_pipeline_languages(mock_hass)
        assert langs == set()
    finally:
        assist_canonicalizer.async_get_pipelines = original


@pytest.mark.asyncio
async def test_capture_chat_log_deltas_exceptional_exit(
    conversation_entity: AssistCanonicalizerConversationEntity,
) -> None:
    """Test capture_chat_log_deltas restores listener when exception occurs."""
    chat_log = DummyChatLog()

    with contextlib.suppress(RuntimeError), conversation_entity._capture_chat_log_deltas(chat_log):
        raise RuntimeError("exceptional exit")
    assert chat_log.delta_listener is None


@pytest.mark.asyncio
async def test_capture_chat_log_deltas_normal_exit_restores_listener(
    conversation_entity: AssistCanonicalizerConversationEntity,
) -> None:
    """Test capture_chat_log_deltas restores the original listener on normal exit."""
    chat_log = DummyChatLog()
    original_listener = object()
    chat_log.delta_listener = original_listener

    with conversation_entity._capture_chat_log_deltas(chat_log):
        # Simulate normal usage within the context
        pass

    # After normal exit, the original listener should be restored
    assert chat_log.delta_listener is original_listener


@pytest.mark.asyncio
async def test_play_back_deltas_without_listener(
    conversation_entity: AssistCanonicalizerConversationEntity,
) -> None:
    """Test play_back_deltas does not raise when original listener is None."""
    chat_log = DummyChatLog(None)
    # This should not raise any exceptions
    conversation_entity._play_back_deltas(chat_log, [{"text": "test"}])


@pytest.mark.asyncio
async def test_async_execute_ranked_candidate_exceptions(
    conversation_entity: AssistCanonicalizerConversationEntity,
) -> None:
    """Test that candidate execution returns None on delegate exceptions."""
    rc = RankedCandidate(
        candidate=Candidate(text="test", intent_name="test"),
        scores=ScoreBreakdown(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    user_input = MockConversationInput("test", "en")

    with patch.object(
        conversation_entity,
        "_delegate_text",
        side_effect=Exception("delegate exception"),
    ):
        res = await conversation_entity._async_execute_ranked_candidate(rc, user_input)
        assert res is None


@pytest.mark.asyncio
async def test_async_process_fallback_missing(
    conversation_entity: AssistCanonicalizerConversationEntity,
) -> None:
    """Test async_process behavior when fallback agent is not found."""
    # Setup runtime with fallback ID that is missing
    conversation_entity._entry.options = {"fallback_agent_id": "missing_agent"}

    async def mock_converse(hass, text, conversation_id, context, language, agent_id, **kwargs):
        """Mock conversation agent fallback target failure."""
        if agent_id == "missing_agent":
            raise ValueError("Agent not found")
        return MagicMock()

    converse_patch = patch(
        "homeassistant.components.conversation.async_converse",
        side_effect=mock_converse,
    )
    with converse_patch:
        user_input = MockConversationInput("hello", "en")
        res = await conversation_entity.async_process(user_input)
        # Should return a default error response when fallback completely fails
        assert res is not None
        assert res.response.error_code == "unknown"


@pytest.mark.asyncio
async def test_execution_phase_exception_delegates_to_fallback_agent() -> None:
    """Unexpected execution errors must reach the fallback agent, not an error result."""
    entry = MagicMock()
    entry.options = {"min_confidence": 0.60, "min_margin": 0.05}
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)

    hass = MagicMock()
    hass.async_create_task = lambda coro: asyncio.create_task(coro)
    hass.async_add_executor_job = AsyncMock(
        side_effect=_executor_job_returning_empty_snapshot_index
    )
    entity.hass = hass
    runtime.indexes["vi"] = MagicMock(candidate_count=5)

    accepted = RankedCandidate(
        candidate=Candidate(text="tắt đèn bếp", intent_name="HassTurnOff"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=1.0,
            final_score=0.9,
        ),
    )
    user_input = MockConversationInput("tắt đèn bếp", "vi")

    with (
        patch.object(
            CanonicalizerRuntime, "rank_with_dynamic_candidates", return_value=(accepted,)
        ),
        patch.object(
            entity,
            "_async_execute_accepted_candidates",
            AsyncMock(side_effect=RuntimeError("registry vanished")),
        ),
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")) as raw,
    ):
        res = await entity._async_process_with_runtime(user_input)

    assert res == "raw_delegated"
    raw.assert_awaited_once_with(user_input)
    assert runtime.diagnostics.last_fallback_reason == FallbackReason.UNEXPECTED_EXCEPTION
    assert runtime.diagnostics.last_error == "registry vanished"


@pytest.mark.asyncio
async def test_store_load_failure_delegates_to_fallback_agent() -> None:
    """Index acquisition errors must forward the raw text, not error out."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock()

    user_input = MockConversationInput("tắt đèn bếp", "vi")

    with (
        patch.object(
            CanonicalizerRuntime,
            "async_load_index_from_store",
            AsyncMock(side_effect=Exception("malformed custom sentences YAML")),
        ),
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")) as raw,
    ):
        res = await entity._async_process_with_runtime(user_input)

    assert res == "raw_delegated"
    assert runtime.diagnostics.last_fallback_reason == FallbackReason.EMPTY_INDEX
    assert "malformed custom sentences YAML" in (runtime.diagnostics.last_error or "")
    raw.assert_awaited_once_with(user_input)
