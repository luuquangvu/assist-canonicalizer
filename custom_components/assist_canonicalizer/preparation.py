"""Shared language cache preparation and warmup for Assist Canonicalizer."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Coroutine

from homeassistant.components.conversation.agent_manager import async_get_agent
from homeassistant.components.conversation.const import HOME_ASSISTANT_AGENT
from homeassistant.core import HomeAssistant

from .runtime import CanonicalizerRuntime, PreparationOutcome
from .utils import normalize_language

_LOGGER = logging.getLogger(__name__)

_MAX_STALE_RETRIES = 5


def _is_caller_cancelling() -> bool:
    """Return True if the current calling task has pending cancellations."""
    try:
        current = asyncio.current_task()
    except RuntimeError:
        return False
    return current.cancelling() > 0 if current is not None else False


async def async_prepare_default_agent(
    hass: HomeAssistant,
    language: str,
) -> bool:
    """Precompute Home Assistant default conversation agent intents and slots for a language.

    Returns True if preparation succeeded or was skipped due to agent absence,
    or False if an exception was caught and logged.
    """
    language = normalize_language(language)
    try:
        default_agent = async_get_agent(hass, HOME_ASSISTANT_AGENT)
        if default_agent is not None:
            prepare_fn = getattr(default_agent, "async_prepare", None)
            if callable(prepare_fn):
                res = prepare_fn(language)
                if inspect.isawaitable(res):
                    await res
        return True
    except Exception as err:
        _LOGGER.debug(
            "Failed to prepare default agent for language %s: %s",
            language,
            err,
            exc_info=True,
        )
        return False


async def async_prepare_language_ranking(
    hass: HomeAssistant,
    runtime: CanonicalizerRuntime,
    language: str,
) -> bool:
    """Precompute runtime dynamic ranking templates and slot index in the executor.

    Returns True if preparation succeeded, or False if an exception was caught and logged.
    """
    language = normalize_language(language)
    try:
        await runtime.async_prepare_language_ranking(hass, language)
        return True
    except Exception as err:
        _LOGGER.debug(
            "Failed to prepare ranking for language %s: %s",
            language,
            err,
            exc_info=True,
        )
        return False


def _get_or_create_preparation_task(
    hass: HomeAssistant,
    runtime: CanonicalizerRuntime,
    language: str,
) -> tuple[asyncio.Task[PreparationOutcome], bool]:
    """Return an active preparation task or create and register a new one."""
    task = runtime.active_preparation_task(language)
    if task is not None:
        return task, False
    new_task = _create_task(
        hass,
        _run_language_preparation(hass, runtime, language),
        name=f"assist_canonicalizer_prepare_{language}",
    )
    new_task.add_done_callback(lambda t: runtime.discard_finished_preparation_task(language, t))
    runtime.register_preparation_task(language, new_task)
    return new_task, True


async def async_prepare_language_caches(
    hass: HomeAssistant,
    runtime: CanonicalizerRuntime,
    language: str,
    *,
    force: bool = False,
) -> None:
    """Precompute ranking templates and default conversation agent caches for a language.

    Coalesces concurrent preparation calls for the same language and caches
    successful preparation results to prevent redundant work across entry points.
    Restarts preparation if an in-flight task is cancelled or invalidated by an index
    or source rebuild.

    Raises RuntimeError after _MAX_STALE_RETRIES consecutive stale outcomes.
    """
    if runtime.closed:
        return

    language = normalize_language(language)

    stale_retries = 0
    while not runtime.closed:
        if not force and runtime.is_language_prepared(language):
            return

        task, created = _get_or_create_preparation_task(hass, runtime, language)
        try:
            outcome = await asyncio.shield(task)
        except asyncio.CancelledError:
            if runtime.closed or _is_caller_cancelling():
                raise
            force = False
            continue
        finally:
            if created:
                runtime.discard_finished_preparation_task(language, task)

        if outcome is PreparationOutcome.STALE:
            stale_retries += 1
            if stale_retries >= _MAX_STALE_RETRIES:
                raise RuntimeError(
                    f"Language preparation for {language!r} did not converge"
                    f" after {stale_retries} consecutive stale outcomes"
                )
            force = False
            continue
        return


async def _run_language_preparation(
    hass: HomeAssistant,
    runtime: CanonicalizerRuntime,
    language: str,
) -> PreparationOutcome:
    """Execute ranking preparation and default agent warmup for one language."""
    if runtime.closed:
        return PreparationOutcome.FAILED

    expected_stamp = runtime.preparation_stamp(language)
    ranking_ok = await async_prepare_language_ranking(hass, runtime, language)
    agent_ok = await async_prepare_default_agent(hass, language)

    if not (ranking_ok and agent_ok and not runtime.closed):
        return PreparationOutcome.FAILED

    if runtime.preparation_stamp(language) != expected_stamp:
        return PreparationOutcome.STALE

    marked = runtime.mark_language_prepared(language, expected_stamp=expected_stamp)
    return PreparationOutcome.SUCCESS if marked else PreparationOutcome.STALE


def _create_task[T](
    hass: HomeAssistant, target: Coroutine[object, object, T], name: str | None = None
) -> asyncio.Task[T]:
    """Safely create an asyncio task with fallback for mock HomeAssistant instances."""
    create_task = getattr(hass, "async_create_task", None)
    if callable(create_task):
        try:
            res = create_task(target, name=name) if name else create_task(target)
        except TypeError:
            res = create_task(target)
        if isinstance(res, asyncio.Task):
            return res
    return asyncio.create_task(target, name=name)
