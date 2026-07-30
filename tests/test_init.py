"""Tests for Assist Canonicalizer integration entry points."""

import asyncio
from functools import partial
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HassJob, HassJobType

import custom_components.assist_canonicalizer
from custom_components.assist_canonicalizer import (
    _async_warmup_pipeline_languages,
    _debounced_registry_rebuild,
    _discover_pipeline_languages,
    _schedule_registry_refresh,
    _subscribe_intent_updates,
    _warmup_single_language,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.assist_canonicalizer.const import DATA_RUNTIME, DOMAIN
from custom_components.assist_canonicalizer.runtime import CanonicalizerRuntime


class _IntentSubscriptionRecorder:
    """Record the callback passed to subscribe_intents."""

    def __init__(self) -> None:
        """Initialize callback storage."""
        self.saved_callback: Any = None

    def subscribe_intents(self, cb: Any) -> Any:
        """Mock subscribing to intent changes."""
        self.saved_callback = cb
        return lambda: None


def test_subscribe_intent_updates_allows_legacy_agent_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip dynamic intent subscriptions on Home Assistant versions without the API."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.agent_manager.get_agent_manager",
        lambda _hass: object(),
    )

    _subscribe_intent_updates(hass, runtime)

    assert runtime.cleanup_callbacks == []


def test_subscribe_intent_updates_rejects_invalid_unsubscribe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail setup when Home Assistant violates its subscription callback contract."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()
    manager = MagicMock()
    manager.subscribe_intents.return_value = None
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.agent_manager.get_agent_manager",
        lambda _hass: manager,
    )

    with pytest.raises(TypeError, match="non-callable unsubscribe callback"):
        _subscribe_intent_updates(hass, runtime)

    assert runtime.cleanup_callbacks == []


def test_registry_debounce_jobs_are_callback_safe() -> None:
    """Keep both event-loop-only registry debounce callbacks off the executor."""
    runtime = CanonicalizerRuntime()
    hass = MagicMock()

    assert HassJob(partial(_debounced_registry_rebuild, hass, runtime)).job_type is (
        HassJobType.Callback
    )
    assert (
        HassJob(partial(_schedule_registry_refresh, hass, runtime, MagicMock())).job_type
        is HassJobType.Callback
    )


@pytest.mark.asyncio
async def test_async_setup_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test setup entry with trigger intents updates callback."""
    hass = MagicMock()
    hass.config.path = MagicMock(return_value="/config")
    hass.data = {}
    hass.add_job = MagicMock()

    # Mock config entries async methods
    hass.config_entries = MagicMock()
    hass.config_entries.async_forward_entry_setups = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {}
    entry.options = {}

    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.registry.async_registry_slot_values",
        lambda h: {"name": ("light",)},
    )

    subscription_recorder = _IntentSubscriptionRecorder()
    mock_agent_manager = MagicMock()
    mock_agent_manager.subscribe_intents = subscription_recorder.subscribe_intents
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.agent_manager.get_agent_manager",
        lambda h: mock_agent_manager,
    )

    hass.bus.async_listen.return_value = lambda: None
    mock_exposed = MagicMock()
    mock_exposed.async_listen_entity_updates.return_value = lambda: None
    monkeypatch.setattr("custom_components.assist_canonicalizer.exposed_entities", mock_exposed)

    mock_setup_services = MagicMock()
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.async_setup_services", mock_setup_services
    )

    # Mock the warmup orchestrator to suppress the real call.
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer._async_warmup_pipeline_languages",
        MagicMock(),
    )

    result = await async_setup_entry(hass, entry)

    assert result is True
    assert hass.data[DOMAIN][entry.entry_id]["entry"] is entry
    runtime = hass.data[DOMAIN][entry.entry_id][DATA_RUNTIME]
    assert isinstance(runtime, CanonicalizerRuntime)

    # Verify triggering the intent update callback schedules index rebuild
    assert subscription_recorder.saved_callback is not None
    runtime.indexes["en"] = MagicMock()
    subscription_recorder.saved_callback({"some": {"intents": {}}})
    hass.add_job.assert_called_with(runtime.async_rebuild_index, hass, "en")


@pytest.mark.asyncio
async def test_async_unload_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test unloading configuration entry with success, failure, and multiple entries."""
    hass = MagicMock()

    entry = MagicMock()
    entry.entry_id = "test_entry"

    runtime = CanonicalizerRuntime()
    hass.data = {
        DOMAIN: {
            "test_entry": {
                "entry": entry,
                DATA_RUNTIME: runtime,
            }
        }
    }

    hass.config_entries = MagicMock()
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)

    mock_unload_services = MagicMock()
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.async_unload_services", mock_unload_services
    )

    # 1. Unload failure path
    hass.config_entries.async_unload_platforms.return_value = False
    result = await async_unload_entry(hass, entry)
    assert result is False

    # 2. Unload success path, verify DOMAIN data is removed when empty
    hass.config_entries.async_unload_platforms.return_value = True
    result = await async_unload_entry(hass, entry)
    assert result is True
    assert DOMAIN not in hass.data

    # 3. Unload success path with remaining entries (verify DOMAIN data is not removed entirely)
    other_entry = MagicMock()
    hass.data = {
        DOMAIN: {
            "test_entry": {
                "entry": entry,
                DATA_RUNTIME: runtime,
            },
            "other_entry": {
                "entry": other_entry,
                DATA_RUNTIME: CanonicalizerRuntime(),
            },
        }
    }
    result = await async_unload_entry(hass, entry)
    assert result is True
    assert DOMAIN in hass.data
    assert "test_entry" not in hass.data[DOMAIN]
    assert "other_entry" in hass.data[DOMAIN]


def test_discover_pipeline_languages_import_failed() -> None:
    """Return empty set when async_get_pipelines is None (import failed)."""
    hass = MagicMock()
    with patch.object(custom_components.assist_canonicalizer, "async_get_pipelines", None):
        result = _discover_pipeline_languages(hass)

    assert result == set()


def test_discover_pipeline_languages_api_raises() -> None:
    """Return empty set when async_get_pipelines raises."""
    hass = MagicMock()
    mock_get_pipelines = MagicMock(side_effect=RuntimeError("pipeline not ready"))
    with patch.object(
        custom_components.assist_canonicalizer,
        "async_get_pipelines",
        mock_get_pipelines,
    ):
        result = _discover_pipeline_languages(hass)

    assert result == set()
    mock_get_pipelines.assert_called_once_with(hass)


def test_discover_pipeline_languages_multiple_unique() -> None:
    """Return deduplicated language codes from multiple pipelines."""
    hass = MagicMock()
    pipeline_en = MagicMock(language="en")
    pipeline_vi = MagicMock(language="vi")
    pipeline_en2 = MagicMock(language="en")  # duplicate
    pipeline_fr = MagicMock(language="fr")

    mock_get_pipelines = MagicMock(
        return_value=[pipeline_en, pipeline_vi, pipeline_en2, pipeline_fr]
    )
    with patch.object(
        custom_components.assist_canonicalizer,
        "async_get_pipelines",
        mock_get_pipelines,
    ):
        result = _discover_pipeline_languages(hass)

    assert len(result) == 3
    assert "en" in result
    assert "vi" in result
    assert "fr" in result


def test_discover_pipeline_languages_empty_and_whitespace_skipped() -> None:
    """Skip pipelines with empty or whitespace-only language attributes."""
    hass = MagicMock()
    pipeline_empty = MagicMock(language="")
    pipeline_whitespace = MagicMock(language="   ")
    pipeline_valid = MagicMock(language="nl")

    mock_get_pipelines = MagicMock(
        return_value=[pipeline_empty, pipeline_whitespace, pipeline_valid]
    )
    with patch.object(
        custom_components.assist_canonicalizer,
        "async_get_pipelines",
        mock_get_pipelines,
    ):
        result = _discover_pipeline_languages(hass)

    assert result == {"nl"}


def test_discover_pipeline_languages_no_language_attr() -> None:
    """Skip pipelines without a language attribute."""
    hass = MagicMock()
    pipeline_no_lang = MagicMock(spec=[])  # no 'language' attribute

    mock_get_pipelines = MagicMock(return_value=[pipeline_no_lang])
    with patch.object(
        custom_components.assist_canonicalizer,
        "async_get_pipelines",
        mock_get_pipelines,
    ):
        result = _discover_pipeline_languages(hass)

    assert result == set()


@pytest.mark.asyncio
async def test_warmup_single_language_already_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do nothing when the index is already cached in memory."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()
    runtime.indexes["en"] = MagicMock()

    mock_load = AsyncMock()
    mock_rebuild = AsyncMock()
    monkeypatch.setattr(CanonicalizerRuntime, "async_load_index_from_store", mock_load)
    monkeypatch.setattr(CanonicalizerRuntime, "async_rebuild_index", mock_rebuild)

    await _warmup_single_language(hass, runtime, "en")

    mock_load.assert_not_called()
    mock_rebuild.assert_not_called()


@pytest.mark.asyncio
async def test_warmup_single_language_loaded_from_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load from store succeeds; do not rebuild."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()

    stored_index = MagicMock()
    mock_load = AsyncMock(return_value=stored_index)
    mock_rebuild = AsyncMock()
    monkeypatch.setattr(CanonicalizerRuntime, "async_load_index_from_store", mock_load)
    monkeypatch.setattr(CanonicalizerRuntime, "async_rebuild_index", mock_rebuild)

    await _warmup_single_language(hass, runtime, "en")

    mock_load.assert_awaited_once_with(hass, "en")
    mock_rebuild.assert_not_called()


@pytest.mark.asyncio
async def test_warmup_single_language_rebuilds_when_store_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild when store has no cached index."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()

    mock_load = AsyncMock(return_value=None)
    mock_rebuild = AsyncMock()
    monkeypatch.setattr(CanonicalizerRuntime, "async_load_index_from_store", mock_load)
    monkeypatch.setattr(CanonicalizerRuntime, "async_rebuild_index", mock_rebuild)

    await _warmup_single_language(hass, runtime, "en")

    mock_load.assert_awaited_once_with(hass, "en")
    mock_rebuild.assert_awaited_once_with(hass, "en", log_error=False)


@pytest.mark.asyncio
async def test_warmup_single_language_exception_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceptions during warmup are silently suppressed."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()

    mock_load = AsyncMock(side_effect=RuntimeError("storage error"))
    monkeypatch.setattr(CanonicalizerRuntime, "async_load_index_from_store", mock_load)

    # Must not raise
    await _warmup_single_language(hass, runtime, "en")

    mock_load.assert_awaited_once_with(hass, "en")
    # Rebuild was never reached because loading from store raised
    assert "en" not in runtime.indexes


@pytest.mark.asyncio
async def test_warmup_single_language_rebuild_exception_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceptions during rebuild are silently suppressed."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()

    mock_load = AsyncMock(return_value=None)
    mock_rebuild = AsyncMock(side_effect=RuntimeError("build error"))
    monkeypatch.setattr(CanonicalizerRuntime, "async_load_index_from_store", mock_load)
    monkeypatch.setattr(CanonicalizerRuntime, "async_rebuild_index", mock_rebuild)

    # Must not raise
    await _warmup_single_language(hass, runtime, "en")

    mock_rebuild.assert_awaited_once_with(hass, "en", log_error=False)


@pytest.mark.asyncio
async def test_warmup_single_language_normalizes_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-canonical language codes are normalized before use."""
    hass = MagicMock()
    runtime = CanonicalizerRuntime()

    mock_load = AsyncMock(return_value=None)
    mock_rebuild = AsyncMock()
    monkeypatch.setattr(CanonicalizerRuntime, "async_load_index_from_store", mock_load)
    monkeypatch.setattr(CanonicalizerRuntime, "async_rebuild_index", mock_rebuild)

    await _warmup_single_language(hass, runtime, "vi-VN")

    # Methods normalize internally; mock captures the raw argument.
    mock_load.assert_awaited_once_with(hass, "vi-VN")


@pytest.mark.asyncio
async def test_warmup_pipeline_languages_awaits_child_warmups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Await one warmup child per discovered pipeline language."""
    hass = MagicMock()
    hass.config.language = "en"
    runtime = CanonicalizerRuntime()
    warmup = AsyncMock()
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer._warmup_single_language",
        warmup,
    )

    pipelines = [MagicMock(language="en"), MagicMock(language="vi")]
    mock_get_pipelines = MagicMock(return_value=pipelines)
    with patch.object(
        custom_components.assist_canonicalizer,
        "async_get_pipelines",
        mock_get_pipelines,
    ):
        await _async_warmup_pipeline_languages(hass, runtime)

    assert warmup.await_count == 2
    assert {call.args[2] for call in warmup.await_args_list} == {"en", "vi"}


@pytest.mark.asyncio
async def test_shutdown_cancels_task_group_warmup_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracked parent warmup cancellation is propagated to TaskGroup children."""
    hass = MagicMock()
    hass.config.language = "en"
    runtime = CanonicalizerRuntime()
    child_started = asyncio.Event()
    child_cancelled = asyncio.Event()

    async def pending_warmup(*_args: Any) -> None:
        """Wait until the parent TaskGroup propagates cancellation."""
        child_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            child_cancelled.set()
            raise

    monkeypatch.setattr(
        "custom_components.assist_canonicalizer._warmup_single_language",
        pending_warmup,
    )
    with patch.object(
        custom_components.assist_canonicalizer,
        "async_get_pipelines",
        MagicMock(return_value=[MagicMock(language="en")]),
    ):
        warmup_task = asyncio.create_task(_async_warmup_pipeline_languages(hass, runtime))
        runtime.track_warmup_task(warmup_task)
        await child_started.wait()
        await runtime.async_shutdown()

    assert warmup_task.cancelled()
    assert child_cancelled.is_set()
    assert runtime.warmup_tasks == set()


@pytest.mark.asyncio
async def test_warmup_pipeline_languages_fallback_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back to hass.config.language when discovery returns empty."""
    hass = MagicMock()
    hass.config.language = "de"
    runtime = CanonicalizerRuntime()
    warmup = AsyncMock()
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer._warmup_single_language",
        warmup,
    )

    with patch.object(custom_components.assist_canonicalizer, "async_get_pipelines", None):
        await _async_warmup_pipeline_languages(hass, runtime)

    warmup.assert_awaited_once_with(hass, runtime, "de")


@pytest.mark.asyncio
async def test_warmup_pipeline_languages_fallback_failure_silent() -> None:
    """Return silently when both discovery and fallback fail."""
    hass = MagicMock()

    # Patch async_get_pipelines to None so discovery returns empty;
    # delete config.language so accessing it raises AttributeError.
    del hass.config.language

    hass.async_create_task = MagicMock()
    runtime = CanonicalizerRuntime()

    with patch.object(custom_components.assist_canonicalizer, "async_get_pipelines", None):
        # Must not raise
        await _async_warmup_pipeline_languages(hass, runtime)

    hass.async_create_task.assert_not_called()


@pytest.mark.asyncio
async def test_async_setup_entry_triggers_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    """async_setup_entry spawns a background warmup task after setup."""
    hass = MagicMock()
    hass.config.path = MagicMock(return_value="/config")
    hass.data = {}
    hass.add_job = MagicMock()
    hass.config_entries = MagicMock()
    hass.config_entries.async_forward_entry_setups = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {}
    entry.options = {}

    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.registry.async_registry_slot_values",
        lambda h: {"name": ("light",)},
    )
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.agent_manager.get_agent_manager",
        lambda h: MagicMock(subscribe_intents=lambda cb: lambda: None),
    )
    hass.bus.async_listen.return_value = lambda: None
    mock_exposed = MagicMock()
    mock_exposed.async_listen_entity_updates.return_value = lambda: None
    monkeypatch.setattr("custom_components.assist_canonicalizer.exposed_entities", mock_exposed)

    mock_setup_services = MagicMock()
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer.async_setup_services", mock_setup_services
    )

    # Mock the warmup orchestrator to verify it gets spawned.
    mock_warmup = MagicMock()
    monkeypatch.setattr(
        "custom_components.assist_canonicalizer._async_warmup_pipeline_languages",
        mock_warmup,
    )

    captured_tasks: list[Any] = []
    hass.async_create_task = MagicMock(side_effect=lambda coro: captured_tasks.append(coro))

    result = await async_setup_entry(hass, entry)

    assert result is True
    assert len(captured_tasks) == 1
    assert captured_tasks[0] is not None
