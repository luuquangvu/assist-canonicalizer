"""Home Assistant registry metadata for candidate generation."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Mapping
from typing import Any, cast

from .const import (
    AREA_SLOT_NAMES,
    ASSISTANT_CONVERSATION,
    ENTITY_SLOT_NAMES,
    FLOOR_SLOT_NAMES,
)

async_should_expose: Any = cast(Any, None)
ar: Any = cast(Any, None)
er: Any = cast(Any, None)
fr: Any = cast(Any, None)

try:
    from homeassistant.components.homeassistant.exposed_entities import (
        async_should_expose as _async_should_expose,
    )
    from homeassistant.helpers import area_registry as _ar
    from homeassistant.helpers import entity_registry as _er
    from homeassistant.helpers import floor_registry as _fr

    async_should_expose = _async_should_expose
    ar = _ar
    er = _er
    fr = _fr
    _HAS_HA_REGISTRIES = True
except (ImportError, RuntimeError):
    async_should_expose = cast(Any, None)
    ar = cast(Any, None)
    er = cast(Any, None)
    fr = cast(Any, None)
    _HAS_HA_REGISTRIES = False


def async_registry_slot_values(hass: Any) -> dict[str, tuple[str, ...]]:
    """Return candidate slot values from exposed HA metadata."""
    entities_by_domain = _exposed_entity_names_by_domain(hass)
    entities = (name for domain_names in entities_by_domain.values() for name in domain_names)
    areas = _area_names(hass)
    floors = _floor_names(hass)
    return build_registry_slot_values(
        entity_names=entities,
        area_names=areas,
        floor_names=floors,
        entity_names_by_domain=entities_by_domain,
    )


def build_registry_slot_values(
    *,
    entity_names: Iterable[str] = (),
    area_names: Iterable[str] = (),
    floor_names: Iterable[str] = (),
    entity_names_by_domain: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Build slot values from supplied registry metadata."""
    slot_values: dict[str, tuple[str, ...]] = {}
    _assign_slot_values(slot_values, ENTITY_SLOT_NAMES, entity_names)
    for domain, names in (entity_names_by_domain or {}).items():
        if isinstance(domain, str) and domain.strip():
            _assign_slot_values(slot_values, _domain_slot_names(ENTITY_SLOT_NAMES, domain), names)
    _assign_slot_values(slot_values, AREA_SLOT_NAMES, area_names)
    _assign_slot_values(slot_values, FLOOR_SLOT_NAMES, floor_names)
    return slot_values


def merge_slot_values(
    primary: Mapping[str, tuple[str, ...]],
    fallback: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Merge slot values while preserving explicit primary values."""
    merged = dict(fallback)
    for slot_name, values in primary.items():
        merged[slot_name] = values
    return merged


def _assign_slot_values(
    slot_values: dict[str, tuple[str, ...]],
    slot_names: Iterable[str],
    values: Iterable[str],
) -> None:
    """Assign cleaned values to each related slot name."""
    cleaned = tuple(_deduplicate_texts(values))
    if not cleaned:
        return
    for slot_name in slot_names:
        slot_values[slot_name] = cleaned


def _domain_slot_names(slot_names: Iterable[str], domain: str) -> tuple[str, ...]:
    """Return domain-scoped slot names."""
    normalized_domain = domain.strip()
    return tuple(f"{slot_name}:{normalized_domain}" for slot_name in slot_names)


def _exposed_entity_names_by_domain(hass: Any) -> dict[str, tuple[str, ...]]:
    """Return names and aliases for entities exposed to Assist grouped by domain."""
    if not _HAS_HA_REGISTRIES:
        return {}

    try:
        entity_registry = er.async_get(hass)
        states = hass.states.async_all()
    except (AttributeError, RuntimeError):
        return {}

    names_by_domain: dict[str, list[str]] = {}
    for state in states:
        entity_id = getattr(state, "entity_id", "")
        if not isinstance(entity_id, str) or "." not in entity_id:
            continue
        try:
            if not async_should_expose(hass, ASSISTANT_CONVERSATION, entity_id):
                continue
        except (KeyError, RuntimeError):
            continue
        domain = entity_id.split(".", 1)[0]
        entry = entity_registry.async_get(entity_id)
        names_by_domain.setdefault(domain, []).extend(_entity_names(hass, er, entry, state))
    return {domain: tuple(_deduplicate_texts(names)) for domain, names in names_by_domain.items()}


def _entity_names(hass: Any, er: Any, entry: Any, state: Any) -> Iterable[str]:
    """Yield spoken entity names from registry aliases or state names."""
    has_yielded = False
    if entry is not None:
        with contextlib.suppress(AttributeError, RuntimeError, ValueError):
            for alias in er.async_get_entity_aliases(hass, entry, allow_empty=False):
                if isinstance(alias, str) and (cleaned_alias := alias.strip()):
                    yield cleaned_alias
                    has_yielded = True
        name = _stripped_or_none(getattr(entry, "name", None)) or _stripped_or_none(
            getattr(entry, "original_name", None)
        )
        if name is not None:
            yield name
            has_yielded = True
    if not has_yielded:
        state_name = getattr(state, "name", None)
        if isinstance(state_name, str):
            yield state_name


def _area_names(hass: Any) -> tuple[str, ...]:
    """Return area names and aliases."""
    if not _HAS_HA_REGISTRIES:
        return ()

    try:
        areas = ar.async_get(hass).async_list_areas()
    except (AttributeError, RuntimeError):
        return ()
    return tuple(_deduplicate_texts(_entry_names_and_aliases(areas)))


def _stripped_or_none(value: Any) -> str | None:
    """Return stripped text when non-empty, otherwise None."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _floor_names(hass: Any) -> tuple[str, ...]:
    """Return floor names and aliases."""
    if not _HAS_HA_REGISTRIES:
        return ()

    try:
        floors = fr.async_get(hass).async_list_floors()
    except (AttributeError, RuntimeError):
        return ()
    return tuple(_deduplicate_texts(_entry_names_and_aliases(floors)))


def _entry_names_and_aliases(entries: Iterable[Any]) -> Iterable[str]:
    """Yield spoken names and aliases from area or floor registry entries."""
    for entry in entries:
        name = getattr(entry, "name", None)
        if isinstance(name, str):
            yield name
        aliases = getattr(entry, "aliases", ())
        if isinstance(aliases, Iterable) and not isinstance(aliases, str):
            for alias in aliases:
                if isinstance(alias, str):
                    yield alias


def _deduplicate_texts(values: Iterable[str]) -> Iterable[str]:
    """Yield non-empty text values once, preserving order."""
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(value.split())
        if not cleaned:
            continue
        normalized = cleaned.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        yield cleaned
