"""Home Assistant entry points for Assist Canonicalizer."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any

from homeassistant.components.conversation import agent_manager
from homeassistant.components.homeassistant import exposed_entities
from homeassistant.const import Platform
from homeassistant.helpers import area_registry, entity_registry, floor_registry
from homeassistant.helpers import event as ha_event

from .const import (
    ASSISTANT_CONVERSATION,
    DATA_RUNTIME,
    DOMAIN,
)
from .registry import async_registry_slot_values
from .runtime import CanonicalizerRuntime
from .services import async_setup_services, async_unload_services
from .utils import normalize_language

_UNINITIALIZED = object()
async_get_pipelines: Any = _UNINITIALIZED

type AssistCanonicalizerConfigEntry = Any
type HomeAssistantInstance = Any

PLATFORMS = [Platform.CONVERSATION]

_LOGGER = logging.getLogger(__name__)


def _runtime_from_entry(
    hass: HomeAssistantInstance, entry: AssistCanonicalizerConfigEntry
) -> CanonicalizerRuntime:
    """Return runtime state for an entry."""
    return hass.data[DOMAIN][entry.entry_id][DATA_RUNTIME]


async def async_setup_entry(
    hass: HomeAssistantInstance, entry: AssistCanonicalizerConfigEntry
) -> bool:
    """Set up Assist Canonicalizer from a config entry."""
    runtime = CanonicalizerRuntime()
    runtime.configure_config_path(hass.config.path)

    runtime.add_cleanup_callback(
        agent_manager.get_agent_manager(hass).subscribe_intents(
            partial(_handle_intent_updates, hass, runtime)
        )
    )
    _refresh_registry_slot_values(hass, runtime)
    _subscribe_registry_updates(hass, runtime)
    domain_data = hass.data.setdefault(DOMAIN, {})
    domain_data[entry.entry_id] = {
        "entry": entry,
        DATA_RUNTIME: runtime,
    }
    entry.async_on_unload(entry.add_update_listener(_async_update_options))
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    async_setup_services(hass)

    runtime.track_warmup_task(
        hass.async_create_task(_async_warmup_pipeline_languages(hass, runtime))
    )

    return True


async def async_unload_entry(
    hass: HomeAssistantInstance, entry: AssistCanonicalizerConfigEntry
) -> bool:
    """Unload Assist Canonicalizer from a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if not unload_ok:
        return False
    await _runtime_from_entry(hass, entry).async_shutdown()
    async_unload_services(hass)
    domain_data: dict[str, Any] = hass.data.get(DOMAIN, {})
    domain_data.pop(entry.entry_id, None)
    if not domain_data:
        hass.data.pop(DOMAIN, None)
    return True


async def _async_update_options(
    hass: HomeAssistantInstance, entry: AssistCanonicalizerConfigEntry
) -> None:
    """Reload the config entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)


def _refresh_registry_slot_values(
    hass: HomeAssistantInstance, runtime: CanonicalizerRuntime
) -> None:
    """Refresh registry-derived slot values used for template expansion."""
    runtime.update_registry_slot_values(async_registry_slot_values(hass))


def _subscribe_registry_updates(hass: HomeAssistantInstance, runtime: CanonicalizerRuntime) -> None:
    """Subscribe to Home Assistant registry metadata changes."""
    debounced_rebuild = partial(_debounced_registry_rebuild, hass, runtime)
    refresh_from_event = partial(
        _schedule_registry_refresh,
        hass,
        runtime,
        debounced_rebuild,
    )

    runtime.add_cleanup_callback(
        hass.bus.async_listen(entity_registry.EVENT_ENTITY_REGISTRY_UPDATED, refresh_from_event)
    )
    runtime.add_cleanup_callback(
        hass.bus.async_listen(area_registry.EVENT_AREA_REGISTRY_UPDATED, refresh_from_event)
    )
    runtime.add_cleanup_callback(
        hass.bus.async_listen(floor_registry.EVENT_FLOOR_REGISTRY_UPDATED, refresh_from_event)
    )
    runtime.add_cleanup_callback(
        exposed_entities.async_listen_entity_updates(
            hass, ASSISTANT_CONVERSATION, refresh_from_event
        )
    )


def _handle_intent_updates(
    hass: HomeAssistantInstance,
    runtime: CanonicalizerRuntime,
    intents_update: Any,
) -> None:
    """Update intent sources and trigger background rebuilds for active languages."""
    if runtime.closed:
        return
    active_languages = list(runtime.indexes)
    runtime.update_intent_sources(intents_update)
    for language in active_languages:
        hass.add_job(runtime.async_rebuild_index, hass, language)


def _debounced_registry_rebuild(
    hass: HomeAssistantInstance,
    runtime: CanonicalizerRuntime,
    now: Any = None,
) -> None:
    """Refresh registry slot values and rebuild active indexes."""
    if runtime.closed:
        return
    runtime.rebuild_timer_cancel = None
    active_languages = list(runtime.indexes)
    _refresh_registry_slot_values(hass, runtime)
    for language in active_languages:
        hass.add_job(runtime.async_rebuild_index, hass, language)


def _schedule_registry_refresh(
    hass: HomeAssistantInstance,
    runtime: CanonicalizerRuntime,
    debounced_rebuild: Any,
    event: Any = None,
) -> None:
    """Schedule a debounced refresh after a registry update."""
    if runtime.closed:
        return
    if runtime.rebuild_timer_cancel is not None:
        runtime.rebuild_timer_cancel()
        runtime.rebuild_timer_cancel = None
    runtime.rebuild_timer_cancel = ha_event.async_call_later(
        hass,
        5,
        debounced_rebuild,
    )


def _discover_pipeline_languages(hass: HomeAssistantInstance) -> set[str]:
    """Return unique language codes from all configured Assist pipelines."""
    global async_get_pipelines
    languages: set[str] = set()
    if async_get_pipelines is _UNINITIALIZED:
        try:
            from homeassistant.components.assist_pipeline import (
                async_get_pipelines as _async_get_pipelines,
            )

            async_get_pipelines = _async_get_pipelines
        except (ImportError, RuntimeError) as err:
            _LOGGER.debug("Assist pipeline import failed: %s", err)
            async_get_pipelines = None
    if async_get_pipelines is None:
        return languages

    try:
        pipelines = async_get_pipelines(hass)
    except Exception:
        return languages

    for pipeline in pipelines:
        lang = getattr(pipeline, "language", None)
        if isinstance(lang, str) and lang.strip():
            languages.add(normalize_language(lang))

    return languages


async def _warmup_single_language(
    hass: HomeAssistantInstance,
    runtime: CanonicalizerRuntime,
    language: str,
) -> None:
    """Try to load an index from store, falling back to a rebuild. Never raises."""
    if runtime.closed:
        return
    try:
        index = runtime.get_index(language)
        if index is not None:
            return
        index = await runtime.async_load_index_from_store(hass, language)
        if index is None:
            await runtime.async_rebuild_index(hass, language, log_error=False)
    except Exception as err:
        _LOGGER.debug(
            "Failed to warm up canonicalizer index for language %s: %s",
            language,
            err,
            exc_info=True,
        )


async def _async_warmup_pipeline_languages(
    hass: HomeAssistantInstance,
    runtime: CanonicalizerRuntime,
) -> None:
    """Discover configured pipeline languages and warm every index in the background."""
    if runtime.closed:
        return
    languages = _discover_pipeline_languages(hass)
    if not languages:
        try:
            languages = {normalize_language(str(hass.config.language))}
        except Exception:
            return

    async with asyncio.TaskGroup() as task_group:
        for language in languages:
            if runtime.closed:
                return
            # The setup task tracked by the runtime owns this TaskGroup; the
            # TaskGroup, in turn, cancels and drains its language children.
            task_group.create_task(_warmup_single_language(hass, runtime, language))
