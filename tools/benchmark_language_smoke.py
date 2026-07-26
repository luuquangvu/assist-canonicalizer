"""Generate exact grammar controls for managed-live language smoke requests.

This module samples sentence syntax only. It never recognizes, ranks, or scores a
request; correctness is decided exclusively by the managed Home Assistant process.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any

import home_assistant_intents
from hassil.errors import MissingListError, MissingRuleError
from hassil.intents import Intents, TextSlotList

_sample_sentence = None

with contextlib.suppress(ImportError):
    from hassil.sample import sample_sentence as _sample_sentence

sample_sentence: Any = _sample_sentence
HAS_SAMPLE_SENTENCE = sample_sentence is not None

ACCURACY_GATED_LANGUAGES = frozenset({"de", "en", "fr", "nl", "vi"})
FIXTURE_ENTITY_NAME = "Living Room Lamp"
FIXTURE_AREA_NAME = "Living Room"
FIXTURE_FLOOR_NAME = "Ground Floor"
MAX_SAMPLES_PER_SENTENCE = 128
LANGUAGE_ENTITY_NAMES = {
    "zh-CN": "客厅灯",
    "zh-HK": "客廳燈",
    "zh-TW": "客廳燈",
}
FIXTURE_TARGETS = (
    ("light.living_room_rgbww_lights", "light", FIXTURE_ENTITY_NAME),
    ("switch.ac", "switch", "Coffee Machine"),
)
_PREFERRED_FIXTURE_SLOTS = ("name", "area", "floor")


@dataclass(frozen=True, slots=True)
class LanguageSmokeCommand:
    """One generated exact control for an installed intent language."""

    language: str
    text: str
    target_entity_id: str
    target_domain: str


def build_language_smoke_commands() -> tuple[LanguageSmokeCommand, ...]:
    """Return one deterministic HassTurnOn grammar sample per installed language."""
    commands: list[LanguageSmokeCommand] = []
    for language in home_assistant_intents.get_languages():
        for target_entity_id, target_domain, default_name in FIXTURE_TARGETS:
            entity_name = LANGUAGE_ENTITY_NAMES.get(language, default_name)
            try:
                text = _sample_turn_on_command(language, entity_name, target_domain)
            except ValueError:
                continue
            commands.append(
                LanguageSmokeCommand(
                    language=language,
                    text=text,
                    target_entity_id=target_entity_id,
                    target_domain=target_domain,
                )
            )
            break
        else:
            raise ValueError(
                f"Unable to generate a fixture-bound HassTurnOn command for {language!r}"
            )
    return tuple(commands)


def _intent_data_supports_domain(intent_data: Any, target_domain: str) -> bool:
    """Return whether intent data permits the fixture target domain."""
    fixed_domain = intent_data.slots.get("domain")
    required_valid, required_domains = _normalized_domain_values(
        intent_data.requires_context.get("domain")
    )
    excluded_valid, excluded_domains = _normalized_domain_values(
        intent_data.excludes_context.get("domain")
    )
    if not (required_valid and excluded_valid):
        return False
    if fixed_domain not in (None, target_domain):
        return False
    if required_domains is not None and target_domain not in required_domains:
        return False
    return excluded_domains is None or target_domain not in excluded_domains


def _normalized_domain_values(value: Any) -> tuple[bool, tuple[Any, ...] | None]:
    """Return supported directive values, with None representing no fixed value."""
    if isinstance(value, Mapping):
        value = value.get("value")
    if value is None:
        return True, None
    if isinstance(value, str | bool | int | float):
        return True, (value,)
    if isinstance(value, (list, tuple, set, frozenset)) and all(
        isinstance(item, str | bool | int | float) for item in value
    ):
        return True, tuple(value)
    return False, None


def _fixture_sentence_sample(
    sentence: Any,
    slot_lists: Mapping[str, Any],
    expansion_rules: Mapping[str, Any],
    fixture_values: Mapping[str, str],
    language: str,
) -> tuple[str, frozenset[str], str] | None:
    """Return the preferred fixture slot and sampled sentence."""
    try:
        fixture_slots = frozenset(sentence.list_names(expansion_rules)).intersection(
            _PREFERRED_FIXTURE_SLOTS
        )
        if not fixture_slots:
            return None
        preferred_slot = next(slot for slot in _PREFERRED_FIXTURE_SLOTS if slot in fixture_slots)
        samples = sample_sentence(
            sentence,
            slot_lists,
            expansion_rules,
            language=language,
            skip_optionals=True,
        )
        fixture_samples = [
            (sample_index, sample.strip())
            for sample_index, sample in enumerate(islice(samples, MAX_SAMPLES_PER_SENTENCE))
            if fixture_values[preferred_slot] in sample
        ]
    except (MissingListError, MissingRuleError, ValueError):
        return None
    if not fixture_samples:
        return None
    _, sample = min(
        fixture_samples,
        key=lambda indexed_value: (
            indexed_value[1].count(FIXTURE_AREA_NAME) + indexed_value[1].count(FIXTURE_FLOOR_NAME),
            indexed_value[0],
        ),
    )
    return preferred_slot, fixture_slots, sample


def _turn_on_sample_candidates(
    intents: Intents,
    intent: Any,
    slot_lists: Mapping[str, Any],
    fixture_values: Mapping[str, str],
    target_domain: str,
    language: str,
) -> list[tuple[int, int, int, tuple[int, int], str]]:
    """Return ranked fixture-bound HassTurnOn sentence samples."""
    candidates = []
    for data_index, intent_data in enumerate(intent.data):
        if not _intent_data_supports_domain(intent_data, target_domain):
            continue
        expansion_rules = {
            **intents.expansion_rules,
            **intent_data.expansion_rules,
        }
        data_slot_lists = {**slot_lists, **intent_data.slot_lists}
        for sentence_index, sentence in enumerate(intent_data.sentences):
            sampled = _fixture_sentence_sample(
                sentence,
                data_slot_lists,
                expansion_rules,
                fixture_values,
                language,
            )
            if sampled is None:
                continue
            preferred_slot, fixture_slots, sample = sampled
            candidates.append(
                (
                    _PREFERRED_FIXTURE_SLOTS.index(preferred_slot),
                    sample.count(FIXTURE_AREA_NAME) + sample.count(FIXTURE_FLOOR_NAME),
                    len(fixture_slots),
                    (data_index, sentence_index),
                    sample,
                )
            )
    return candidates


def _sample_turn_on_command(language: str, entity_name: str, target_domain: str) -> str:
    """Sample a fixture-bound command without evaluating it in-process."""
    if sample_sentence is None:
        raise RuntimeError("sample_sentence is not available in the installed version of hassil")
    intent_dict = home_assistant_intents.get_intents(language)
    if intent_dict is None:
        raise ValueError(f"Installed language {language!r} has no intent data")
    intents = Intents.from_dict(intent_dict)
    intent = intents.intents.get("HassTurnOn")
    if intent is None:
        raise ValueError(f"Installed language {language!r} has no HassTurnOn intent")
    fixture_values = {
        "name": entity_name,
        "area": FIXTURE_AREA_NAME,
        "floor": FIXTURE_FLOOR_NAME,
    }
    if candidates := _turn_on_sample_candidates(
        intents,
        intent,
        {
            **intents.slot_lists,
            "name": TextSlotList.from_strings([entity_name]),
            "area": TextSlotList.from_strings([FIXTURE_AREA_NAME]),
            "floor": TextSlotList.from_strings([FIXTURE_FLOOR_NAME]),
        },
        fixture_values,
        target_domain,
        language,
    ):
        return min(candidates)[4]
    raise ValueError(f"Unable to generate a {target_domain} HassTurnOn command for {language!r}")
