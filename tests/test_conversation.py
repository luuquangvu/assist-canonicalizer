"""Tests for the Assist Canonicalizer conversation entity platform."""

import asyncio
import inspect
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.conversation.models import ConversationInput
from homeassistant.core import Context

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
from custom_components.assist_canonicalizer.indexer import build_index
from custom_components.assist_canonicalizer.ranking import RankedCandidate, ScoreBreakdown
from custom_components.assist_canonicalizer.runtime import CanonicalizerRuntime


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
        assert runtime.diagnostics.last_fallback_reason == FallbackReason.LOW_CONFIDENCE

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
        assert runtime.diagnostics.last_fallback_reason == FallbackReason.LOW_CONFIDENCE

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
    validation_err_res.response.error_code = "error"
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
    """Try the next confident ranked candidate before falling back to raw text."""
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
    validation_err_res.response.error_code = "error"
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
async def test_validation_delegate_exception_tries_remaining_candidates() -> None:
    """Treat primary-agent validation exceptions as one candidate failing validation."""
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
            AsyncMock(side_effect=[RuntimeError("primary failed"), validation_ok_res]),
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
    assert runtime.diagnostics.last_error is None


@pytest.mark.asyncio
async def test_validation_delegate_exceptions_fallback_keep_last_error() -> None:
    """Keep the last validation exception when every candidate fails validation."""
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
            AsyncMock(
                side_effect=[
                    RuntimeError("first primary failed"),
                    RuntimeError("second primary failed"),
                ]
            ),
        ) as mock_del_text,
        patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")) as raw,
    ):
        res = await entity._async_process_with_runtime(user_input)

    assert res == "raw_delegated"
    assert mock_del_text.await_args_list[0].args == ("turn on kitchen light", user_input)
    assert mock_del_text.await_args_list[0].kwargs == {"primary": True}
    assert mock_del_text.await_args_list[1].args == ("turn on kitchen lamp", user_input)
    assert mock_del_text.await_args_list[1].kwargs == {"primary": True}
    raw.assert_awaited_once_with(user_input)
    assert runtime.diagnostics.last_fallback_reason == FallbackReason.VALIDATION_FAILED
    assert runtime.diagnostics.last_error == "second primary failed"


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

    class DummyPipeline:
        """Dummy pipeline class for testing."""

        prefer_local_intents = False

    mock_pipeline_module = MagicMock()
    mock_pipeline_module.async_get_pipeline = MagicMock(return_value=DummyPipeline())

    monkeypatch.setitem(
        sys.modules,
        "homeassistant.components.assist_pipeline.pipeline",
        mock_pipeline_module,
    )

    entry = MagicMock()
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock()
    entity.hass.data = {"assist_pipeline": MagicMock()}

    user_input = MockConversationInput("tắt đèn bếp", "vi")
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
async def test_async_process_prefer_local_intents_true_no_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test prefer_local_intents behavior when it is True (should not run shortcut)."""

    class DummyPipeline:
        """Dummy pipeline class for testing."""

        prefer_local_intents = True

    mock_pipeline_module = MagicMock()
    mock_pipeline_module.async_get_pipeline = MagicMock(return_value=DummyPipeline())

    monkeypatch.setitem(
        sys.modules,
        "homeassistant.components.assist_pipeline.pipeline",
        mock_pipeline_module,
    )

    entry = MagicMock()
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    entity.hass = MagicMock()
    entity.hass.data = {"assist_pipeline": MagicMock()}
    entity.hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))

    user_input = MockConversationInput("tắt đèn bếp", "vi")
    validation_ok_res = MagicMock()
    validation_ok_res.response.error_code = None

    with (
        patch.object(
            entity, "_delegate_text", AsyncMock(return_value=validation_ok_res)
        ) as mock_delegate,
        patch.object(
            CanonicalizerRuntime, "rank_with_dynamic_candidates", return_value=()
        ) as mock_rank,
        patch.object(
            entity, "_delegate_raw_text", AsyncMock(return_value=validation_ok_res)
        ) as mock_delegate_raw,
        patch.object(
            CanonicalizerRuntime, "async_rebuild_index", AsyncMock(return_value=MagicMock())
        ) as mock_rebuild_index,
    ):
        await entity.async_process(user_input)
        mock_delegate.assert_not_called()
        mock_rebuild_index.assert_awaited_once()
        mock_rank.assert_called_once()
        mock_delegate_raw.assert_awaited_once_with(user_input)


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

    class DummyPipeline:
        """Dummy pipeline class for testing."""

        prefer_local_intents = False

    mock_pipeline_module = MagicMock()
    mock_pipeline_module.async_get_pipeline = MagicMock(return_value=DummyPipeline())

    monkeypatch.setitem(
        sys.modules,
        "homeassistant.components.assist_pipeline.pipeline",
        mock_pipeline_module,
    )

    entry = MagicMock()
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

    entity.hass.data = {
        "assist_pipeline": MagicMock(),
    }
    entity.hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))

    user_input = MockConversationInput("tắt đèn bếp", "vi", conversation_id="conv-1")

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
async def test_async_validate_ranked_candidate_restores_chat_log(
    simulate_exception: bool,
) -> None:
    """Test that the chat log is restored if candidate validation fails or raises an exception."""
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

        res = await entity._async_validate_ranked_candidate(ranked_candidate, user_input)
        assert res is None

        # Verify that mock_chat_log.content was restored (keeping only the original user message)
        assert mock_chat_log.content == [mock_user_message]


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
