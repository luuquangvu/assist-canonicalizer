"""Provision the deterministic Assist Canonicalizer benchmark home."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import orjson
import voluptuous as vol
from homeassistant.components.conversation.agent_manager import async_get_agent
from homeassistant.components.conversation.const import HOME_ASSISTANT_AGENT
from homeassistant.components.conversation.models import ConversationInput
from homeassistant.components.conversation.trace import (
    async_clear_traces,
    async_get_traces,
)
from homeassistant.components.homeassistant.const import DATA_EXPOSED_ENTITIES
from homeassistant.components.homeassistant.exposed_entities import (
    async_set_assistant_option,
    async_should_expose,
)
from homeassistant.components.intent import async_register_timer_handler
from homeassistant.components.intent.const import TIMER_DATA
from homeassistant.components.media_player.const import DOMAIN as MEDIA_PLAYER_DOMAIN
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED, Platform
from homeassistant.core import Event, HomeAssistant, ServiceCall, SupportsResponse, callback
from homeassistant.helpers import area_registry, entity_registry, floor_registry
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.discovery import async_load_platform
from homeassistant.helpers.typing import ConfigType

DOMAIN = "assist_canonicalizer_benchmark"
SERVICE_REAPPLY = "reapply"
SERVICE_CLEAR_CONVERSATION_TRACES = "clear_conversation_traces"
SERVICE_GET_CONVERSATION_TRACES = "get_conversation_traces"
SERVICE_RECOGNIZE_CANONICAL = "recognize_canonical"
SERVICE_PREPARE_CASE = "prepare_case"
STATUS_ENTITY_ID = "sensor.assist_canonicalizer_benchmark_fixture"
ASSISTANT_CONVERSATION = "conversation"
BENCHMARK_DEVICE_ID = "assist-canonicalizer-benchmark-device"
PROVISION_TIMEOUT_SECONDS = 60.0
POLL_INTERVAL_SECONDS = 0.25
CONFIG_SCHEMA = cv.empty_config_schema(DOMAIN)
RECOGNIZE_CANONICAL_SCHEMA = vol.Schema(
    {
        vol.Required("text"): cv.string,
        vol.Required("language"): cv.string,
        vol.Optional("device_id"): cv.string,
        vol.Optional("satellite_id"): cv.string,
    }
)
PREPARE_CASE_SCHEMA = vol.Schema(
    {
        vol.Required("intent"): cv.string,
        vol.Required("language"): cv.string,
        vol.Optional("slots", default=dict): dict,
    }
)

_TIMER_INTENTS = {
    "HassCancelAllTimers",
    "HassCancelTimer",
    "HassDecreaseTimer",
    "HassIncreaseTimer",
    "HassPauseTimer",
    "HassStartTimer",
    "HassTimerStatus",
    "HassUnpauseTimer",
}
_TIMER_INTENTS_WITH_EXISTING_TIMER = _TIMER_INTENTS - {
    "HassCancelAllTimers",
    "HassStartTimer",
}
_MEDIA_INTENTS_REQUIRING_PLAYBACK = {
    "HassMediaNext",
    "HassMediaPause",
    "HassMediaPrevious",
}
_LIST_INTENTS = {
    "HassListAddItem",
    "HassListCompleteItem",
    "HassListRemoveItem",
}
_SHOPPING_LIST_INTENTS = {
    "HassShoppingListAddItem",
    "HassShoppingListCompleteItem",
    "HassShoppingListLastItems",
}

_LOGGER = logging.getLogger(__name__)
_FIXTURE_PATH = Path(__file__).with_name("fixture.json")
_BENCHMARK_PLATFORMS = (
    Platform.CLIMATE,
    Platform.HUMIDIFIER,
    Platform.LAWN_MOWER,
    Platform.LOCK,
    Platform.MEDIA_PLAYER,
    Platform.SIREN,
    Platform.TODO,
)


@callback
def _handle_timer_event(_event: Any, _timer: Any) -> None:
    """Acknowledge benchmark timer events without external device I/O."""


def _initialize_domain_data(
    hass: HomeAssistant,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Initialize benchmark domain data and timer handling."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    domain_data["lock"] = asyncio.Lock()
    domain_data["manifest"] = manifest
    domain_data["timer_handlers"] = {}
    domain_data["timer_event_handler"] = _handle_timer_event
    _register_timer_device(hass, BENCHMARK_DEVICE_ID)
    return domain_data


def _load_benchmark_platforms(
    hass: HomeAssistant,
    config: ConfigType,
) -> None:
    """Schedule loading of every benchmark entity platform."""
    for platform in _BENCHMARK_PLATFORMS:
        hass.async_create_task(
            async_load_platform(hass, platform, DOMAIN, {}, config),
            f"{DOMAIN}_load_{platform}",
        )


async def _handle_clear_conversation_traces(_call: ServiceCall) -> None:
    """Clear Home Assistant's bounded passive conversation trace buffer."""
    async_clear_traces()


async def _handle_get_conversation_traces(
    _call: ServiceCall,
) -> dict[str, Any]:
    """Return passive traces emitted by the production conversation request."""
    return {"traces": [trace.as_dict() for trace in async_get_traces()]}


async def _recognize_canonical(
    hass: HomeAssistant,
    call: ServiceCall,
) -> dict[str, Any]:
    """Recognize a canonical control without handling it."""
    default_agent = async_get_agent(hass, HOME_ASSISTANT_AGENT)
    recognize = getattr(default_agent, "async_recognize_intent", None)
    if recognize is None:
        raise RuntimeError("Home Assistant Default Agent recognition is unavailable")
    result = await recognize(
        ConversationInput(
            text=call.data["text"],
            context=call.context,
            conversation_id=None,
            device_id=call.data.get("device_id"),
            satellite_id=call.data.get("satellite_id"),
            language=call.data["language"],
            agent_id=HOME_ASSISTANT_AGENT,
        )
    )
    if result is None:
        return {"intent": None, "slots": {}, "unmatched_count": 0}
    return {
        "intent": result.intent.name,
        "slots": {entity.name: entity.value for entity in result.entities_list},
        "unmatched_count": len(result.unmatched_entities),
    }


async def _prepare_media_case(
    hass: HomeAssistant,
    intent_name: str,
) -> None:
    """Start exposed media players when an intent requires playback."""
    if intent_name not in _MEDIA_INTENTS_REQUIRING_PLAYBACK:
        return
    if media_players := [
        entity_id
        for entity_id in hass.states.async_entity_ids(MEDIA_PLAYER_DOMAIN)
        if async_should_expose(hass, ASSISTANT_CONVERSATION, entity_id)
    ]:
        await hass.services.async_call(
            MEDIA_PLAYER_DOMAIN,
            "media_play",
            target={"entity_id": media_players},
            blocking=True,
        )


def _prepare_timer_case(
    hass: HomeAssistant,
    domain_data: Mapping[str, Any],
    intent_name: str,
    language: str,
    slots: dict[str, Any],
) -> None:
    """Reset timers and create an existing timer when required."""
    if intent_name not in _TIMER_INTENTS:
        return
    timer_manager = hass.data[TIMER_DATA]
    for timer_id in tuple(timer_manager.timers):
        timer_manager.cancel_timer(timer_id)
    if intent_name not in _TIMER_INTENTS_WITH_EXISTING_TIMER:
        return
    start_hours = _optional_int_slot(slots, "start_hours")
    start_minutes = _optional_int_slot(slots, "start_minutes")
    start_seconds = _optional_int_slot(slots, "start_seconds")
    if all(value is None for value in (start_hours, start_minutes, start_seconds)):
        start_minutes = 10
    timer_id = timer_manager.start_timer(
        domain_data.get("timer_device_id", BENCHMARK_DEVICE_ID),
        hours=start_hours,
        minutes=start_minutes,
        seconds=start_seconds,
        language=language,
        name=_optional_string_slot(slots, "name"),
    )
    if intent_name == "HassUnpauseTimer":
        timer_manager.pause_timer(timer_id)


def _prepare_list_case(
    domain_data: Mapping[str, Any],
    intent_name: str,
    slots: dict[str, Any],
) -> None:
    """Reset the benchmark to-do list when required."""
    if intent_name not in _LIST_INTENTS:
        return
    todo_entity = domain_data.get("todo_entity")
    reset_items = getattr(todo_entity, "reset_items", None)
    if reset_items is None:
        raise RuntimeError("Benchmark to-do entity is unavailable")
    item = _optional_string_slot(slots, "item") if intent_name != "HassListAddItem" else None
    reset_items(item)


async def _prepare_shopping_list_case(
    hass: HomeAssistant,
    intent_name: str,
    slots: dict[str, Any],
) -> None:
    """Reset the benchmark shopping list when required."""
    if intent_name not in _SHOPPING_LIST_INTENTS:
        return
    if not hass.services.has_service("shopping_list", "complete_all"):
        raise RuntimeError("Shopping list benchmark dependency is unavailable")
    await hass.services.async_call("shopping_list", "complete_all", blocking=True)
    await hass.services.async_call(
        "shopping_list",
        "clear_completed_items",
        blocking=True,
    )
    if intent_name != "HassShoppingListCompleteItem":
        return
    item = _optional_string_slot(slots, "item")
    if item is None:
        raise RuntimeError("Shopping-list completion oracle has no item slot")
    await hass.services.async_call(
        "shopping_list",
        "add_item",
        {"name": item},
        blocking=True,
    )


async def _prepare_case(
    hass: HomeAssistant,
    domain_data: Mapping[str, Any],
    call: ServiceCall,
) -> None:
    """Reset stateful live-intent prerequisites for one request."""
    intent_name = call.data["intent"]
    language = call.data["language"]
    slots = cast(dict[str, Any], call.data["slots"])
    await _prepare_media_case(hass, intent_name)
    _prepare_timer_case(hass, domain_data, intent_name, language, slots)
    _prepare_list_case(domain_data, intent_name, slots)
    await _prepare_shopping_list_case(hass, intent_name, slots)


def _register_benchmark_services(
    hass: HomeAssistant,
    manifest: dict[str, Any],
    domain_data: Mapping[str, Any],
) -> None:
    """Register benchmark control, trace, and preparation services."""

    async def handle_reapply(_call: ServiceCall) -> None:
        """Reapply the benchmark fixture manifest."""
        await _async_provision_safely(hass, manifest)

    async def handle_recognize(call: ServiceCall) -> dict[str, Any]:
        """Recognize canonical text for one service request."""
        return await _recognize_canonical(hass, call)

    async def handle_prepare(call: ServiceCall) -> None:
        """Prepare stateful prerequisites for one benchmark case."""
        await _prepare_case(hass, domain_data, call)

    hass.services.async_register(DOMAIN, SERVICE_REAPPLY, handle_reapply)
    hass.services.async_register(
        DOMAIN,
        SERVICE_CLEAR_CONVERSATION_TRACES,
        _handle_clear_conversation_traces,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_CONVERSATION_TRACES,
        _handle_get_conversation_traces,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RECOGNIZE_CANONICAL,
        handle_recognize,
        schema=RECOGNIZE_CANONICAL_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_PREPARE_CASE,
        handle_prepare,
        schema=PREPARE_CASE_SCHEMA,
    )


def _schedule_fixture_provisioning(
    hass: HomeAssistant,
    manifest: dict[str, Any],
) -> None:
    """Provision now or after Home Assistant has started."""

    @callback
    def start_fixture(_event: Event | None = None) -> None:
        hass.async_create_task(
            _async_provision_safely(hass, manifest),
            f"{DOMAIN}_provision",
        )

    if hass.is_running:
        start_fixture()
    else:
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, start_fixture)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up deterministic floors, areas, entities, aliases, and exposure."""
    manifest = await hass.async_add_executor_job(_load_manifest)
    _validate_manifest(manifest)
    domain_data = _initialize_domain_data(hass, manifest)
    _load_benchmark_platforms(hass, config)
    _register_benchmark_services(hass, manifest, domain_data)

    _schedule_fixture_provisioning(hass, manifest)
    return True


def _load_manifest() -> dict[str, Any]:
    """Load the tracked benchmark fixture manifest."""
    loaded = orjson.loads(_FIXTURE_PATH.read_bytes())
    if not isinstance(loaded, dict):
        raise ValueError("Benchmark fixture root must be an object")
    return loaded


def _validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate fixture counts, keys, and references before changing HA state."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark fixture schema")
    floors = _object_list(manifest, "floors")
    areas = _object_list(manifest, "areas")
    entities = _object_list(manifest, "entities")
    expected_counts = manifest.get("expected_counts")
    if not isinstance(expected_counts, dict):
        raise ValueError("Benchmark fixture expected_counts must be an object")
    actual_counts = {
        "floors": len(floors),
        "areas": len(areas),
        "exposed_entities": len(entities),
    }
    if expected_counts != actual_counts:
        raise ValueError(
            f"Benchmark fixture counts differ: expected={expected_counts} actual={actual_counts}"
        )

    expected_domain_counts = manifest.get("expected_domain_counts")
    if not isinstance(expected_domain_counts, dict) or not all(
        isinstance(domain, str)
        and domain
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        for domain, count in expected_domain_counts.items()
    ):
        raise ValueError("Benchmark fixture expected_domain_counts must map domains to counts")
    actual_domain_counts = dict(
        sorted(Counter(_required_string(entity, "domain") for entity in entities).items())
    )
    if expected_domain_counts != actual_domain_counts:
        raise ValueError(
            "Benchmark fixture domain counts differ: "
            f"expected={expected_domain_counts} actual={actual_domain_counts}"
        )

    floor_keys = _unique_string_values(floors, "key", "floor")
    area_keys = _unique_string_values(areas, "key", "area")
    if unknown_floors := {str(area.get("floor")) for area in areas} - floor_keys:
        raise ValueError(f"Unknown floor references: {sorted(unknown_floors)}")
    if unknown_areas := {str(entity.get("area")) for entity in entities} - area_keys:
        raise ValueError(f"Unknown area references: {sorted(unknown_areas)}")

    identities: set[tuple[str, str, str]] = set()
    for entity in entities:
        identity = (
            _required_string(entity, "domain"),
            _required_string(entity, "platform"),
            _required_string(entity, "unique_id"),
        )
        if identity in identities:
            raise ValueError(f"Duplicate benchmark entity identity: {identity}")
        identities.add(identity)
        _required_string(entity, "name")
        _string_list(entity, "aliases")
        if "vacuum_area_segment" in entity:
            if identity[0] != "vacuum":
                raise ValueError("vacuum_area_segment is only valid for vacuum entities")
            _required_string(entity, "vacuum_area_segment")
        if "timer_anchor" in entity and not isinstance(entity["timer_anchor"], bool):
            raise ValueError("Benchmark fixture timer_anchor must be a boolean")

    timer_anchors = [entity for entity in entities if entity.get("timer_anchor") is True]
    if len(timer_anchors) != 1:
        raise ValueError(
            f"Benchmark fixture must define exactly one timer anchor entity, "
            f"found {len(timer_anchors)}"
        )


def _object_list(mapping: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Return a required list of objects from a manifest mapping."""
    value = mapping.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Benchmark fixture {key} must be a list of objects")
    return value


def _unique_string_values(items: list[dict[str, Any]], key: str, description: str) -> set[str]:
    """Return unique required string values for a manifest object list."""
    values = [_required_string(item, key) for item in items]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate benchmark {description} {key}")
    return set(values)


def _required_string(mapping: dict[str, Any], key: str) -> str:
    """Return one non-empty required manifest string."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Benchmark fixture {key} must be a non-empty string")
    return value


def _string_list(mapping: dict[str, Any], key: str) -> list[str]:
    """Return one manifest list containing only non-empty strings."""
    value = mapping.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"Benchmark fixture {key} must be a list of non-empty strings")
    return value


def _optional_int_slot(slots: dict[str, Any], key: str) -> int | None:
    """Return an optional integer-valued live oracle slot."""
    value = slots.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"Benchmark slot {key} must be numeric")
    return int(float(value))


def _optional_string_slot(slots: dict[str, Any], key: str) -> str | None:
    """Return an optional non-empty string-valued live oracle slot."""
    value = slots.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Benchmark slot {key} must be a non-empty string")
    return value


def _register_timer_device(hass: HomeAssistant, device_id: str) -> None:
    """Register one fixture device as a local intent-timer endpoint."""
    domain_data = hass.data[DOMAIN]
    handlers = cast(dict[str, Any], domain_data["timer_handlers"])
    if device_id in handlers:
        return
    handlers[device_id] = async_register_timer_handler(
        hass,
        device_id,
        domain_data["timer_event_handler"],
    )


async def _async_provision_safely(hass: HomeAssistant, manifest: dict[str, Any]) -> None:
    """Provision the fixture and expose a stable error state on failure."""
    lock: asyncio.Lock = hass.data[DOMAIN]["lock"]
    async with lock:
        _set_status(hass, "provisioning", manifest)
        try:
            summary = await _async_provision(hass, manifest)
        except Exception as err:
            _LOGGER.exception("BENCHMARK_FIXTURE_ERROR: %s", err)
            _set_status(hass, "error", manifest, error=str(err))
            return
        _set_status(hass, "ready", manifest, **summary)
        _LOGGER.info(
            "BENCHMARK_FIXTURE_READY fixture_id=%s fingerprint=%s floors=%d areas=%d "
            "exposed_entities=%d domains=%s",
            manifest["fixture_id"],
            summary["fingerprint"],
            summary["floor_count"],
            summary["area_count"],
            summary["exposed_entity_count"],
            summary["domain_counts"],
        )


async def _async_provision(hass: HomeAssistant, manifest: dict[str, Any]) -> dict[str, Any]:
    """Apply the tracked benchmark model and return its verified summary."""
    entities = _object_list(manifest, "entities")
    resolved_entities = await _async_resolve_entities(hass, entities)
    _apply_timer_device(hass, manifest, resolved_entities)
    floor_ids = _apply_floors(hass, _object_list(manifest, "floors"))
    area_ids = _apply_areas(hass, _object_list(manifest, "areas"), floor_ids)
    _apply_entities(hass, entities, resolved_entities, area_ids)
    _apply_exposure(hass, set(resolved_entities.values()))
    return _verified_summary(hass, manifest, resolved_entities, floor_ids, area_ids)


def _apply_timer_device(
    hass: HomeAssistant, manifest: dict[str, Any], resolved_entities: dict[str, str]
) -> None:
    """Use the designated timer anchor device as the live timer endpoint."""
    timer_anchor_entity = next(
        (entity for entity in manifest.get("entities", []) if entity.get("timer_anchor") is True),
        None,
    )

    if timer_anchor_entity is None:
        raise ValueError("Benchmark fixture manifest is missing a designated timer_anchor entity.")

    identity = _entity_identity(timer_anchor_entity)
    if identity not in resolved_entities:
        raise ValueError(
            f"Expected timer anchor entity '{identity}' was not resolved "
            f"in the active environment. "
            f"Available resolved entities: {sorted(resolved_entities.keys())}"
        )

    satellite_entity_id = resolved_entities[identity]
    entry = entity_registry.async_get(hass).async_get(satellite_entity_id)
    device_id = entry.device_id if entry is not None else None
    if device_id is None:
        device_id = BENCHMARK_DEVICE_ID
    _register_timer_device(hass, device_id)
    hass.data[DOMAIN]["timer_device_id"] = device_id


async def _async_resolve_entities(
    hass: HomeAssistant, entities: list[dict[str, Any]]
) -> dict[str, str]:
    """Wait for every required demo entity and return identity-to-entity mappings."""
    registry = entity_registry.async_get(hass)
    deadline = time.monotonic() + PROVISION_TIMEOUT_SECONDS
    while True:
        resolved: dict[str, str] = {}
        missing: list[str] = []
        for entity in entities:
            domain = _required_string(entity, "domain")
            platform = _required_string(entity, "platform")
            unique_id = _required_string(entity, "unique_id")
            identity = _entity_identity(entity)
            entity_id = registry.async_get_entity_id(domain, platform, unique_id)
            if entity_id is None or hass.states.get(entity_id) is None:
                missing.append(identity)
                continue
            resolved[identity] = entity_id
        if not missing:
            return resolved
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out waiting for benchmark entities: {missing}")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def _apply_floors(hass: HomeAssistant, floors: list[dict[str, Any]]) -> dict[str, str]:
    """Create or normalize the exact fixture floors."""
    registry = floor_registry.async_get(hass)
    floor_ids: dict[str, str] = {}
    for floor in floors:
        name = _required_string(floor, "name")
        aliases = set(_string_list(floor, "aliases"))
        level = floor.get("level")
        if level is not None and not isinstance(level, int):
            raise ValueError(f"Floor level must be an integer or null: {name}")
        entry = registry.async_get_floor_by_name(name)
        if entry is None:
            entry = registry.async_create(name, aliases=aliases, level=level)
        else:
            # The WebSocket API accepts null to clear a floor level, while the
            # registry method's annotation currently omits that supported value.
            entry = registry.async_update(
                entry.floor_id,
                aliases=aliases,
                level=cast(Any, level),
            )
        floor_ids[_required_string(floor, "key")] = entry.floor_id
    actual_names = {entry.name for entry in registry.async_list_floors()}
    expected_names = {_required_string(floor, "name") for floor in floors}
    if actual_names != expected_names:
        raise RuntimeError(
            f"Benchmark floor set differs: expected={sorted(expected_names)} "
            f"actual={sorted(actual_names)}"
        )
    return floor_ids


def _apply_areas(
    hass: HomeAssistant,
    areas: list[dict[str, Any]],
    floor_ids: dict[str, str],
) -> dict[str, str]:
    """Create or normalize the exact fixture areas and floor assignments."""
    registry = area_registry.async_get(hass)
    area_ids: dict[str, str] = {}
    for area in areas:
        name = _required_string(area, "name")
        aliases = set(_string_list(area, "aliases"))
        floor_id = floor_ids[_required_string(area, "floor")]
        entry = registry.async_get_area_by_name(name)
        if entry is None:
            entry = registry.async_create(name, aliases=aliases, floor_id=floor_id)
        else:
            entry = registry.async_update(
                entry.id,
                aliases=aliases,
                floor_id=floor_id,
            )
        area_ids[_required_string(area, "key")] = entry.id
    actual_names = {entry.name for entry in registry.async_list_areas()}
    expected_names = {_required_string(area, "name") for area in areas}
    if actual_names != expected_names:
        raise RuntimeError(
            f"Benchmark area set differs: expected={sorted(expected_names)} "
            f"actual={sorted(actual_names)}"
        )
    return area_ids


def _apply_entities(
    hass: HomeAssistant,
    entities: list[dict[str, Any]],
    resolved_entities: dict[str, str],
    area_ids: dict[str, str],
) -> None:
    """Apply deterministic names, aliases, and area assignments."""
    registry = entity_registry.async_get(hass)
    for entity in entities:
        entity_id = resolved_entities[_entity_identity(entity)]
        registry.async_update_entity(
            entity_id,
            aliases=[entity_registry.COMPUTED_NAME, *_string_list(entity, "aliases")],
            area_id=area_ids[_required_string(entity, "area")],
            name=_required_string(entity, "name"),
        )
        if "vacuum_area_segment" in entity:
            registry.async_update_entity_options(
                entity_id,
                "vacuum",
                _vacuum_options(entity, area_ids),
            )


def _apply_exposure(hass: HomeAssistant, exposed_entity_ids: set[str]) -> None:
    """Expose exactly the fixture entities to Home Assistant conversation."""
    exposed_manager = hass.data[DATA_EXPOSED_ENTITIES]
    exposed_manager.async_set_expose_new_entities(ASSISTANT_CONVERSATION, False)
    for entity_id in hass.states.async_entity_ids():
        async_set_assistant_option(
            hass,
            ASSISTANT_CONVERSATION,
            entity_id,
            "should_expose",
            entity_id in exposed_entity_ids,
        )


def _verify_exposure(
    hass: HomeAssistant,
    expected_exposed: set[str],
) -> None:
    """Verify that exactly the fixture entities are exposed."""
    actual_exposed = {
        entity_id
        for entity_id in hass.states.async_entity_ids()
        if async_should_expose(hass, ASSISTANT_CONVERSATION, entity_id)
    }
    if actual_exposed != expected_exposed:
        raise RuntimeError(
            f"Benchmark exposure differs: expected={sorted(expected_exposed)} "
            f"actual={sorted(actual_exposed)}"
        )


def _verify_entity_registry_entry(
    registry: entity_registry.EntityRegistry,
    entity: dict[str, Any],
    entity_id: str,
    area_ids: Mapping[str, str],
) -> None:
    """Verify one fixture entity's registry state."""
    entry = registry.async_get(entity_id)
    if entry is None:
        raise RuntimeError(f"Benchmark entity disappeared: {entity_id}")
    expected_aliases = [entity_registry.COMPUTED_NAME, *_string_list(entity, "aliases")]
    expected_area_id = area_ids[_required_string(entity, "area")]
    expected_vacuum_options = (
        _vacuum_options(entity, area_ids) if "vacuum_area_segment" in entity else None
    )
    if (
        entry.name != _required_string(entity, "name")
        or entry.aliases != expected_aliases
        or entry.area_id != expected_area_id
        or (
            expected_vacuum_options is not None
            and entry.options.get("vacuum") != expected_vacuum_options
        )
    ):
        raise RuntimeError(f"Benchmark entity registry state differs: {entity_id}")


def _verify_entity_registry(
    hass: HomeAssistant,
    entities: list[dict[str, Any]],
    resolved_entities: Mapping[str, str],
    area_ids: Mapping[str, str],
) -> None:
    """Verify every fixture entity's registry state."""
    registry = entity_registry.async_get(hass)
    for entity in entities:
        entity_id = resolved_entities[_entity_identity(entity)]
        _verify_entity_registry_entry(registry, entity, entity_id, area_ids)


def _verify_floor_registry(
    hass: HomeAssistant,
    floors: list[dict[str, Any]],
    floor_ids: Mapping[str, str],
) -> None:
    """Verify every fixture floor's registry state."""
    registry = floor_registry.async_get(hass)
    for floor in floors:
        entry = registry.async_get_floor(floor_ids[_required_string(floor, "key")])
        if (
            entry is None
            or entry.aliases != set(_string_list(floor, "aliases"))
            or entry.level != floor.get("level")
        ):
            raise RuntimeError(f"Benchmark floor registry state differs: {floor['key']}")


def _verify_area_registry(
    hass: HomeAssistant,
    areas: list[dict[str, Any]],
    area_ids: Mapping[str, str],
    floor_ids: Mapping[str, str],
) -> None:
    """Verify every fixture area's registry state."""
    registry = area_registry.async_get(hass)
    for area in areas:
        entry = registry.async_get_area(area_ids[_required_string(area, "key")])
        if (
            entry is None
            or entry.aliases != set(_string_list(area, "aliases"))
            or entry.floor_id != floor_ids[_required_string(area, "floor")]
        ):
            raise RuntimeError(f"Benchmark area registry state differs: {area['key']}")


def _fixture_summary(
    hass: HomeAssistant,
    manifest: Mapping[str, Any],
    entities: list[dict[str, Any]],
    floor_ids: Mapping[str, str],
    area_ids: Mapping[str, str],
    exposed_entity_ids: set[str],
) -> dict[str, Any]:
    """Build the stable summary for a verified fixture."""
    return {
        "fingerprint": fixture_fingerprint(manifest),
        "floor_count": len(floor_ids),
        "area_count": len(area_ids),
        "exposed_entity_count": len(exposed_entity_ids),
        "runtime_state_count": len(hass.states.async_entity_ids()),
        "domain_counts": dict(sorted(Counter(entity["domain"] for entity in entities).items())),
    }


def _verified_summary(
    hass: HomeAssistant,
    manifest: dict[str, Any],
    resolved_entities: dict[str, str],
    floor_ids: dict[str, str],
    area_ids: dict[str, str],
) -> dict[str, Any]:
    """Verify the applied fixture and return its stable fingerprint summary."""
    entities = _object_list(manifest, "entities")
    expected_exposed = set(resolved_entities.values())
    _verify_exposure(hass, expected_exposed)
    _verify_entity_registry(hass, entities, resolved_entities, area_ids)
    _verify_floor_registry(hass, _object_list(manifest, "floors"), floor_ids)
    _verify_area_registry(
        hass,
        _object_list(manifest, "areas"),
        area_ids,
        floor_ids,
    )
    return _fixture_summary(
        hass,
        manifest,
        entities,
        floor_ids,
        area_ids,
        expected_exposed,
    )


def fixture_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Return the fixture component's canonical user-model fingerprint."""
    keys = (
        "schema_version",
        "fixture_id",
        "expected_counts",
        "expected_domain_counts",
        "floors",
        "areas",
        "entities",
    )
    payload = {key: manifest[key] for key in keys}
    encoded = orjson.dumps(
        payload,
        option=orjson.OPT_SORT_KEYS,
    )
    return hashlib.sha256(encoded).hexdigest()


def _entity_identity(entity: dict[str, Any]) -> str:
    """Return one stable manifest identity for a fixture entity."""
    return ":".join(_required_string(entity, key) for key in ("domain", "platform", "unique_id"))


def _vacuum_options(
    entity: dict[str, Any], area_ids: Mapping[str, str]
) -> dict[str, dict[str, list[str]]]:
    """Return the deterministic all-area segment mapping for one vacuum."""
    segment = _required_string(entity, "vacuum_area_segment")
    return {"area_mapping": {area_id: [segment] for area_id in area_ids.values()}}


def _set_status(
    hass: HomeAssistant,
    state: str,
    manifest: dict[str, Any],
    **attributes: Any,
) -> None:
    """Publish benchmark fixture readiness without adding an exposed entity."""
    hass.states.async_set(
        STATUS_ENTITY_ID,
        state,
        {
            "friendly_name": "Assist Canonicalizer Benchmark Fixture",
            "fixture_id": manifest.get("fixture_id"),
            "schema_version": manifest.get("schema_version"),
            **attributes,
        },
    )
