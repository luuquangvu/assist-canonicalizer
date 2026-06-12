"""Tests for the Assist Canonicalizer conversation entity platform."""

import asyncio
import inspect
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

    runtime.set_index(MagicMock())
    runtime.indexes["en"] = MagicMock()
    assert "en" in runtime.indexes
    with patch.object(
        CanonicalizerRuntime,
        "async_clear_index",
        AsyncMock(side_effect=lambda hass, language=None: runtime.clear_index(language)),
    ) as mock_clear:
        await entity.async_reload("en")
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

    async def mock_async_add_executor_job(target, *args, **kwargs):
        """Mock executor work and return an empty index for the snapshot build."""
        if getattr(target, "__name__", None) == "_build_index_from_snapshot":
            return build_index("vi", [])
        return target(*args, **kwargs)

    hass = MagicMock()
    hass.async_create_task = lambda coro: asyncio.create_task(coro)
    hass.async_add_executor_job = AsyncMock(side_effect=mock_async_add_executor_job)
    entity.hass = hass

    user_input = MockConversationInput("tắt đèn bếp", "vi")

    with patch.object(entity, "_delegate_raw_text", AsyncMock(return_value="raw_delegated")):
        res = await entity._async_process_with_runtime(user_input)
        assert res == "raw_delegated"
        assert runtime.diagnostics.last_fallback_reason == FallbackReason.EMPTY_INDEX

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
        assert runtime.diagnostics.last_fallback_reason == FallbackReason.RANKING_FAILED

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
async def test_conversation_delegate_and_fallback_agent_logic() -> None:
    """Test delegation and fallback agent ID selection branch options."""
    entry = MagicMock()
    entry.entry_id = "this_agent"
    entry.options = cast(dict[str, Any], {CONF_MIN_CONFIDENCE: 0.60, CONF_MIN_MARGIN: 0.05})
    entry.data = {}
    runtime = CanonicalizerRuntime()
    entity = AssistCanonicalizerConversationEntity(entry, runtime)
    hass = MagicMock()

    async def mock_async_add_executor_job(target, *args, **kwargs):
        """Mock async_add_executor_job; intercept _build_index to return an empty index."""
        if getattr(target, "__name__", None) == "_build_index":
            return build_index("vi", [])
        return target(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=mock_async_add_executor_job)
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
