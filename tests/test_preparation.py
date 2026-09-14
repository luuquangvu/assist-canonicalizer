"""Unit tests for shared language preparation and warmup helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.conversation.const import HOME_ASSISTANT_AGENT
from homeassistant.core import HomeAssistant

from custom_components.assist_canonicalizer.indexer import CanonicalIndex
from custom_components.assist_canonicalizer.preparation import (
    _create_task,
    async_prepare_default_agent,
    async_prepare_language_caches,
    async_prepare_language_ranking,
)
from custom_components.assist_canonicalizer.runtime import (
    CanonicalizerRuntime,
    PreparationOutcome,
)


@pytest.mark.asyncio
async def test_async_prepare_default_agent_success() -> None:
    """Test async_prepare_default_agent succeeds with an awaitable async_prepare."""
    hass = MagicMock(spec=HomeAssistant)
    mock_agent = MagicMock()
    mock_agent.async_prepare = AsyncMock()

    with patch(
        "custom_components.assist_canonicalizer.preparation.async_get_agent",
        return_value=mock_agent,
    ):
        result = await async_prepare_default_agent(hass, "en")
        assert result is True
        mock_agent.async_prepare.assert_awaited_once_with("en")


@pytest.mark.asyncio
async def test_async_prepare_default_agent_sync_callable() -> None:
    """Test async_prepare_default_agent succeeds when async_prepare is a synchronous callable."""
    hass = MagicMock(spec=HomeAssistant)
    mock_agent = MagicMock()
    mock_agent.async_prepare = MagicMock(return_value=None)

    with patch(
        "custom_components.assist_canonicalizer.preparation.async_get_agent",
        return_value=mock_agent,
    ):
        result = await async_prepare_default_agent(hass, "en")
        assert result is True
        mock_agent.async_prepare.assert_called_once_with("en")


@pytest.mark.asyncio
async def test_async_prepare_default_agent_missing_or_no_method() -> None:
    """Test async_prepare_default_agent returns True when agent is None or lacks async_prepare."""
    hass = MagicMock(spec=HomeAssistant)

    with patch(
        "custom_components.assist_canonicalizer.preparation.async_get_agent",
        return_value=None,
    ):
        assert await async_prepare_default_agent(hass, "en") is True

    mock_agent_no_attr = object()
    with patch(
        "custom_components.assist_canonicalizer.preparation.async_get_agent",
        return_value=mock_agent_no_attr,
    ):
        assert await async_prepare_default_agent(hass, "en") is True


@pytest.mark.asyncio
async def test_async_prepare_default_agent_suppresses_exception() -> None:
    """Test async_prepare_default_agent catches and logs exceptions without re-raising."""
    hass = MagicMock(spec=HomeAssistant)
    mock_agent = MagicMock()
    mock_agent.async_prepare = AsyncMock(side_effect=RuntimeError("Agent failure"))

    with patch(
        "custom_components.assist_canonicalizer.preparation.async_get_agent",
        return_value=mock_agent,
    ):
        result = await async_prepare_default_agent(hass, "en")
        assert result is False


@pytest.mark.asyncio
async def test_async_prepare_language_ranking_success() -> None:
    """Test async_prepare_language_ranking delegates to runtime and returns True."""
    hass = MagicMock(spec=HomeAssistant)
    runtime = CanonicalizerRuntime()

    with patch.object(
        CanonicalizerRuntime, "async_prepare_language_ranking", AsyncMock()
    ) as mock_prep:
        result = await async_prepare_language_ranking(hass, runtime, "en")
        assert result is True
        mock_prep.assert_awaited_once_with(hass, "en")


@pytest.mark.asyncio
async def test_async_prepare_language_ranking_suppresses_exception() -> None:
    """Test async_prepare_language_ranking catches exceptions and returns False."""
    hass = MagicMock(spec=HomeAssistant)
    runtime = CanonicalizerRuntime()

    with patch.object(
        CanonicalizerRuntime,
        "async_prepare_language_ranking",
        AsyncMock(side_effect=RuntimeError("Ranking failure")),
    ):
        result = await async_prepare_language_ranking(hass, runtime, "en")
        assert result is False


@pytest.mark.asyncio
async def test_async_prepare_language_caches_marks_prepared_and_skips_subsequent() -> None:
    """Test async_prepare_language_caches prepares caches once and skips subsequent calls."""
    hass = MagicMock(spec=HomeAssistant)
    runtime = CanonicalizerRuntime()
    mock_agent = MagicMock()
    mock_agent.async_prepare = AsyncMock()

    with (
        patch.object(
            CanonicalizerRuntime, "async_prepare_language_ranking", AsyncMock()
        ) as mock_ranking,
        patch(
            "custom_components.assist_canonicalizer.preparation.async_get_agent",
            return_value=mock_agent,
        ) as mock_get_agent,
    ):
        # 1. First run: performs preparation
        await async_prepare_language_caches(hass, runtime, "en")
        assert runtime.is_language_prepared("en") is True
        mock_ranking.assert_awaited_once_with(hass, "en")
        mock_get_agent.assert_called_once_with(hass, HOME_ASSISTANT_AGENT)
        mock_agent.async_prepare.assert_awaited_once_with("en")

        # 2. Second run: already prepared, skips both
        mock_ranking.reset_mock()
        mock_get_agent.reset_mock()
        mock_agent.async_prepare.reset_mock()

        await async_prepare_language_caches(hass, runtime, "en")
        mock_ranking.assert_not_called()
        mock_get_agent.assert_not_called()
        mock_agent.async_prepare.assert_not_called()


@pytest.mark.asyncio
async def test_async_prepare_language_caches_force_reprepares() -> None:
    """Test force=True causes async_prepare_language_caches to re-run preparations."""
    hass = MagicMock(spec=HomeAssistant)
    runtime = CanonicalizerRuntime()
    mock_agent = MagicMock()
    mock_agent.async_prepare = AsyncMock()

    with (
        patch.object(
            CanonicalizerRuntime, "async_prepare_language_ranking", AsyncMock()
        ) as mock_ranking,
        patch(
            "custom_components.assist_canonicalizer.preparation.async_get_agent",
            return_value=mock_agent,
        ),
    ):
        await async_prepare_language_caches(hass, runtime, "en")
        assert runtime.is_language_prepared("en") is True
        assert mock_ranking.await_count == 1

        # Force re-preparation
        await async_prepare_language_caches(hass, runtime, "en", force=True)
        assert mock_ranking.await_count == 2
        assert mock_agent.async_prepare.await_count == 2


@pytest.mark.asyncio
async def test_async_prepare_language_caches_invalidation_triggers_reprepare() -> None:
    """Test that cache invalidation triggers fresh preparation."""
    hass = MagicMock(spec=HomeAssistant)
    runtime = CanonicalizerRuntime()
    mock_agent = MagicMock()
    mock_agent.async_prepare = AsyncMock()

    with (
        patch.object(
            CanonicalizerRuntime, "async_prepare_language_ranking", AsyncMock()
        ) as mock_ranking,
        patch(
            "custom_components.assist_canonicalizer.preparation.async_get_agent",
            return_value=mock_agent,
        ),
    ):
        await async_prepare_language_caches(hass, runtime, "en")
        assert runtime.is_language_prepared("en") is True

        # Invalidate specific language
        runtime.invalidate_language_preparation("en")
        assert runtime.is_language_prepared("en") is False

        await async_prepare_language_caches(hass, runtime, "en")
        assert mock_ranking.await_count == 2

        # Invalidate all via clear_index
        runtime.clear_index("en")
        assert runtime.is_language_prepared("en") is False

        await async_prepare_language_caches(hass, runtime, "en")
        assert mock_ranking.await_count == 3


@pytest.mark.asyncio
async def test_async_prepare_language_caches_deduplicates_in_flight() -> None:
    """Test concurrent preparation calls coalesce onto a single running task."""
    hass = MagicMock(spec=HomeAssistant)
    runtime = CanonicalizerRuntime()
    gate = asyncio.Event()

    async def slow_ranking(_hass: HomeAssistant, _lang: str) -> None:
        """Simulate a long-running ranking preparation."""
        await gate.wait()

    with (
        patch.object(
            CanonicalizerRuntime,
            "async_prepare_language_ranking",
            side_effect=slow_ranking,
        ) as mock_ranking,
        patch(
            "custom_components.assist_canonicalizer.preparation.async_get_agent",
            return_value=None,
        ),
    ):
        # Start call 1 (will pause on gate)
        task1 = asyncio.create_task(async_prepare_language_caches(hass, runtime, "en"))
        await asyncio.sleep(0)  # Yield to start task1

        assert runtime.active_preparation_task("en") is not None

        # Start call 2 while call 1 is still in flight
        task2 = asyncio.create_task(async_prepare_language_caches(hass, runtime, "en"))
        await asyncio.sleep(0)

        # Release gate to allow completion
        gate.set()
        await asyncio.gather(task1, task2)

        # Ranking should have been called exactly once
        mock_ranking.assert_awaited_once_with(hass, "en")
        assert runtime.is_language_prepared("en") is True
        assert runtime.active_preparation_task("en") is None


@pytest.mark.asyncio
async def test_async_prepare_language_caches_creator_cancellation_does_not_cancel_task() -> None:
    """Test that cancelling the creator caller does not cancel the shared task or fail peers."""
    hass = MagicMock(spec=HomeAssistant)
    runtime = CanonicalizerRuntime()
    gate = asyncio.Event()

    async def slow_ranking(_hass: HomeAssistant, _lang: str) -> None:
        """Simulate a long-running ranking preparation."""
        await gate.wait()

    with (
        patch.object(
            CanonicalizerRuntime,
            "async_prepare_language_ranking",
            side_effect=slow_ranking,
        ),
        patch(
            "custom_components.assist_canonicalizer.preparation.async_get_agent",
            return_value=None,
        ),
    ):
        # Caller 1 creates the preparation task and shields it
        caller1 = asyncio.create_task(async_prepare_language_caches(hass, runtime, "en"))
        await asyncio.sleep(0)

        prep_task = runtime.active_preparation_task("en")
        assert prep_task is not None

        # Caller 2 awaits the in-flight task
        caller2 = asyncio.create_task(async_prepare_language_caches(hass, runtime, "en"))
        await asyncio.sleep(0)

        # Cancel caller 1 (the creator)
        caller1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller1

        # Underlying task must NOT be cancelled
        assert not prep_task.cancelled()
        assert not prep_task.done()

        # Let preparation complete
        gate.set()
        await caller2

        # Language must be marked as prepared
        assert runtime.is_language_prepared("en") is True
        assert runtime.active_preparation_task("en") is None


@pytest.mark.asyncio
async def test_async_prepare_language_caches_does_not_cache_on_failure() -> None:
    """Test that failing preparations are not marked as prepared."""
    hass = MagicMock(spec=HomeAssistant)
    runtime = CanonicalizerRuntime()

    with (
        patch.object(
            CanonicalizerRuntime,
            "async_prepare_language_ranking",
            AsyncMock(side_effect=RuntimeError("ranking fail")),
        ),
        patch(
            "custom_components.assist_canonicalizer.preparation.async_get_agent",
            return_value=None,
        ),
    ):
        await async_prepare_language_caches(hass, runtime, "en")
        assert runtime.is_language_prepared("en") is False


@pytest.mark.asyncio
async def test_async_prepare_language_caches_early_return_when_closed() -> None:
    """Test early return when runtime is closed."""
    hass = MagicMock(spec=HomeAssistant)
    runtime = CanonicalizerRuntime()
    runtime._closed = True

    with patch.object(
        CanonicalizerRuntime, "async_prepare_language_ranking", AsyncMock()
    ) as mock_ranking:
        await async_prepare_language_caches(hass, runtime, "en")
        mock_ranking.assert_not_called()


@pytest.mark.asyncio
async def test_create_task_handles_varied_signatures() -> None:
    """Test _create_task handles hass with and without async_create_task name support."""
    # 1. hass without async_create_task attribute
    hass_plain = MagicMock(spec=HomeAssistant)
    del hass_plain.async_create_task

    async def sample_coro() -> int:
        """Sample coroutine for task creation testing."""
        return 42

    task1 = _create_task(hass_plain, sample_coro(), name="plain_task")
    assert isinstance(task1, asyncio.Task)
    assert await task1 == 42

    # 2. hass with async_create_task that rejects name keyword argument
    def mock_create_task_no_name[T](
        target: Coroutine[object, object, T] | asyncio.Task[T],
    ) -> asyncio.Task[T]:
        """Wrap target coroutine into a task."""
        if isinstance(target, asyncio.Task):
            return target
        return asyncio.create_task(target)

    hass_no_name = MagicMock(spec=HomeAssistant)
    hass_no_name.async_create_task = MagicMock(
        side_effect=lambda target, **kwargs: (
            (_ for _ in ()).throw(TypeError("no name"))
            if kwargs.get("name")
            else mock_create_task_no_name(target)
        )
    )

    async def sample_coro2() -> int:
        """Secondary sample coroutine for task creation fallback testing."""
        return 84

    task2 = _create_task(hass_no_name, sample_coro2(), name="task_fallback")
    assert isinstance(task2, asyncio.Task)
    assert await task2 == 84


@pytest.mark.asyncio
async def test_async_prepare_language_caches_restarts_when_generation_stamp_changes() -> None:
    """Test preparation restarts and re-runs when generation stamp changes during in-flight run."""
    hass = MagicMock(spec=HomeAssistant)
    runtime = CanonicalizerRuntime()
    first_run = True

    async def mock_ranking(_hass: HomeAssistant, _lang: str) -> None:
        """Simulate ranking preparation, incrementing generation on first invocation."""
        nonlocal first_run
        if first_run:
            first_run = False
            # Simulate external rebuild or source invalidation changing the stamp
            runtime.source_generation += 1

    with (
        patch.object(
            CanonicalizerRuntime,
            "async_prepare_language_ranking",
            side_effect=mock_ranking,
        ) as ranking_patch,
        patch(
            "custom_components.assist_canonicalizer.preparation.async_get_agent",
            return_value=None,
        ),
    ):
        await async_prepare_language_caches(hass, runtime, "en")
        # Ranking should run twice: once invalidated by stamp change, then restarted
        assert ranking_patch.await_count == 2
        assert runtime.is_language_prepared("en") is True


@pytest.mark.asyncio
async def test_async_prepare_language_caches_restarts_when_task_cancelled_by_rebuild() -> None:
    """Test that a caller restarts when its in-flight task is cancelled by rebuild."""
    hass = MagicMock(spec=HomeAssistant)
    runtime = CanonicalizerRuntime()
    first_run = True

    async def cancelling_ranking(_hass: HomeAssistant, _lang: str) -> None:
        """Simulate an index rebuild cancelling the active preparation task on first run."""
        nonlocal first_run
        if first_run:
            first_run = False
            # Simulate index rebuild starting, which invalidates/cancels the preparation task
            runtime.invalidate_language_preparation("en")
            await asyncio.sleep(0)

    with (
        patch.object(
            CanonicalizerRuntime,
            "async_prepare_language_ranking",
            side_effect=cancelling_ranking,
        ) as ranking_patch,
        patch(
            "custom_components.assist_canonicalizer.preparation.async_get_agent",
            return_value=None,
        ),
    ):
        await async_prepare_language_caches(hass, runtime, "en")
        assert ranking_patch.await_count == 2
        assert runtime.is_language_prepared("en") is True


@pytest.mark.asyncio
async def test_mark_language_prepared_checks_expected_stamp() -> None:
    """Test mark_language_prepared verifies expected_stamp before recording preparation."""
    runtime = CanonicalizerRuntime()
    current_stamp = runtime.preparation_stamp("en")
    stale_stamp = (
        current_stamp[0],
        current_stamp[1],
        current_stamp[2] + 999,
        current_stamp[3],
        current_stamp[4],
    )

    # Mismatched stamp should fail to mark
    assert runtime.mark_language_prepared("en", expected_stamp=stale_stamp) is False
    assert runtime.is_language_prepared("en") is False

    # Matching stamp should succeed
    assert runtime.mark_language_prepared("en", expected_stamp=current_stamp) is True
    assert runtime.is_language_prepared("en") is True


@pytest.mark.asyncio
async def test_rebuild_and_index_clearing_cancels_preparation_tasks() -> None:
    """Test clear_index, set_index, and cancel_preparation_task cancel in-flight tasks."""
    runtime = CanonicalizerRuntime()

    async def dummy_coro() -> PreparationOutcome:
        await asyncio.sleep(100)
        return PreparationOutcome.SUCCESS

    # 1. Test clear_index(language) cancels task
    task1 = asyncio.create_task(dummy_coro())
    runtime.register_preparation_task("en", task1)
    runtime.clear_index("en")
    assert task1.cancelling() > 0
    await asyncio.sleep(0)
    assert task1.cancelled()
    assert runtime.active_preparation_task("en") is None

    # 2. Test set_index cancels task
    task2 = asyncio.create_task(dummy_coro())
    runtime.register_preparation_task("en", task2)
    mock_index = CanonicalIndex(language="en", candidates=())
    runtime.set_index(mock_index)
    assert task2.cancelling() > 0
    await asyncio.sleep(0)
    assert task2.cancelled()
    assert runtime.active_preparation_task("en") is None

    # 3. Test clear_index(None) cancels all tasks
    task_en = asyncio.create_task(dummy_coro())
    task_vi = asyncio.create_task(dummy_coro())
    runtime.register_preparation_task("en", task_en)
    runtime.register_preparation_task("vi", task_vi)
    runtime.clear_index(None)
    assert task_en.cancelling() > 0
    assert task_vi.cancelling() > 0
    await asyncio.sleep(0)
    assert task_en.cancelled()
    assert task_vi.cancelled()
    assert runtime.active_preparation_task("en") is None
    assert runtime.active_preparation_task("vi") is None
