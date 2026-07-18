"""Generate exact grammar controls for managed-live language smoke requests.

This module samples sentence syntax only. It never recognizes, ranks, or scores a
request; correctness is decided exclusively by the managed Home Assistant process.
"""

from __future__ import annotations

import contextlib
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


def _sample_turn_on_command(language: str, entity_name: str, target_domain: str) -> str:
    """Sample a fixture-bound command without evaluating it in-process."""
    if sample_sentence is None:
        raise RuntimeError("sample_sentence is not available in the installed version of hassil")
    intent_dict = home_assistant_intents.get_intents(language)
    if intent_dict is None:
        raise ValueError(f"Installed language {language!r} has no intent data")
    intents = Intents.from_dict(intent_dict)
    slot_lists = {
        **intents.slot_lists,
        "name": TextSlotList.from_strings([entity_name]),
        "area": TextSlotList.from_strings([FIXTURE_AREA_NAME]),
        "floor": TextSlotList.from_strings([FIXTURE_FLOOR_NAME]),
    }
    intent = intents.intents.get("HassTurnOn")
    if intent is None:
        raise ValueError(f"Installed language {language!r} has no HassTurnOn intent")

    fixture_values = {
        "name": entity_name,
        "area": FIXTURE_AREA_NAME,
        "floor": FIXTURE_FLOOR_NAME,
    }
    candidates: list[tuple[int, int, int, int, str]] = []
    preferred_slots = ("name", "area", "floor")
    for data_index, intent_data in enumerate(intent.data):
        fixed_domain = intent_data.slots.get("domain")
        required_domain = intent_data.requires_context.get("domain")
        excluded_domains = intent_data.excludes_context.get("domain", ())
        if isinstance(excluded_domains, str):
            excluded_domains = (excluded_domains,)
        if fixed_domain not in (None, target_domain) or target_domain in excluded_domains:
            continue
        if isinstance(required_domain, str) and required_domain != target_domain:
            continue
        if isinstance(required_domain, (list, tuple, set)) and (
            target_domain not in required_domain
        ):
            continue

        expansion_rules = {**intents.expansion_rules, **intent_data.expansion_rules}
        data_slot_lists = {**slot_lists, **intent_data.slot_lists}
        for sentence_index, sentence in enumerate(intent_data.sentences):
            fixture_slots = set(sentence.list_names(expansion_rules)).intersection(preferred_slots)
            if not fixture_slots:
                continue
            preferred_slot = next(slot for slot in preferred_slots if slot in fixture_slots)
            try:
                samples = sample_sentence(
                    sentence,
                    data_slot_lists,
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
                continue
            if fixture_samples:
                _, sample = min(
                    fixture_samples,
                    key=lambda indexed_value: (
                        indexed_value[1].count(FIXTURE_AREA_NAME)
                        + indexed_value[1].count(FIXTURE_FLOOR_NAME),
                        indexed_value[0],
                    ),
                )
                candidates.append(
                    (
                        preferred_slots.index(preferred_slot),
                        sample.count(FIXTURE_AREA_NAME) + sample.count(FIXTURE_FLOOR_NAME),
                        len(fixture_slots),
                        data_index * 1000 + sentence_index,
                        sample,
                    )
                )
    if candidates:
        return min(candidates)[4]
    raise ValueError(f"Unable to generate a {target_domain} HassTurnOn command for {language!r}")
