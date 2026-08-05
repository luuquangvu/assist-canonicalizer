"""Generate exact grammar controls for managed-live language smoke requests.

This module samples sentence syntax only. It never recognizes, ranks, or scores a
request; correctness is decided exclusively by the managed Home Assistant process.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Protocol, runtime_checkable

import home_assistant_intents
from hassil.errors import MissingListError, MissingRuleError
from hassil.expression import Sentence
from hassil.intents import Intent, Intents, SlotList, TextSlotList


class SampleSentence(Protocol):
    """Callable interface shared by supported Hassil sample helpers."""

    def __call__(
        self,
        sentence: Sentence,
        slot_lists: dict[str, SlotList] | None = None,
        expansion_rules: dict[str, Sentence] | None = None,
        language: str | None = None,
        expand_lists: bool = True,
        expand_ranges: bool = True,
        skip_optionals: bool = False,
    ) -> Iterable[str]:
        """Generate sentence samples for one parsed expression."""
        ...


class ListReference(Protocol):
    """List-reference fields consumed by fixture sampling."""

    list_name: str
    slot_name: str


@runtime_checkable
class HasListReferences(Protocol):
    """Expression interface used by supported Hassil releases."""

    def list_references(self, expansion_rules: Mapping[str, Sentence]) -> Iterable[ListReference]:
        """Return list references reachable from this expression."""
        ...


_sample_sentence: SampleSentence | None = None

try:
    from hassil.sample import sample_sentence as imported_sample_sentence
except ImportError:
    pass
else:
    _sample_sentence = imported_sample_sentence

HAS_SAMPLE_SENTENCE = _sample_sentence is not None

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
    expected_slots: tuple[tuple[str, str], ...]


def build_language_smoke_commands() -> tuple[LanguageSmokeCommand, ...]:
    """Return one deterministic HassTurnOn grammar sample per installed language."""
    commands: list[LanguageSmokeCommand] = []
    for language in home_assistant_intents.get_languages():
        for target_entity_id, target_domain, default_name in FIXTURE_TARGETS:
            entity_name = LANGUAGE_ENTITY_NAMES.get(language, default_name)
            try:
                text, expected_slots = _sample_turn_on_command(language, entity_name, target_domain)
            except ValueError:
                continue
            commands.append(
                LanguageSmokeCommand(
                    language=language,
                    text=text,
                    target_entity_id=target_entity_id,
                    target_domain=target_domain,
                    expected_slots=expected_slots,
                )
            )
            break
        else:
            raise ValueError(
                f"Unable to generate a fixture-bound HassTurnOn command for {language!r}"
            )
    return tuple(commands)


def _intent_data_supports_domain(intent_data: object, target_domain: str) -> bool:
    """Return whether intent data permits the fixture target domain."""
    slots = getattr(intent_data, "slots", None)
    requires_context = getattr(intent_data, "requires_context", None)
    excludes_context = getattr(intent_data, "excludes_context", None)
    if (
        not isinstance(slots, Mapping)
        or not isinstance(requires_context, Mapping)
        or not isinstance(excludes_context, Mapping)
    ):
        return False
    fixed_domain = slots.get("domain")
    required_valid, required_domains = _normalized_domain_values(requires_context.get("domain"))
    excluded_valid, excluded_domains = _normalized_domain_values(excludes_context.get("domain"))
    if not (required_valid and excluded_valid):
        return False
    if fixed_domain not in (None, target_domain):
        return False
    if required_domains is not None and target_domain not in required_domains:
        return False
    return excluded_domains is None or target_domain not in excluded_domains


type DomainValue = str | bool | int | float


def _normalized_domain_values(value: object) -> tuple[bool, tuple[DomainValue, ...] | None]:
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
        return True, tuple(item for item in value if isinstance(item, str | bool | int | float))
    return False, None


def _fixture_sentence_sample(
    sentence: object,
    slot_lists: dict[str, SlotList],
    expansion_rules: dict[str, Sentence],
    fixture_values: Mapping[str, str],
    language: str,
) -> tuple[str, frozenset[str], str] | None:
    """Return the preferred fixture slot and sampled sentence."""
    try:
        expression = getattr(sentence, "expression", None)
        if not isinstance(expression, HasListReferences):
            return None
        list_references = tuple(expression.list_references(expansion_rules))
        fixture_references = tuple(
            list_reference
            for list_reference in list_references
            if list_reference.slot_name in _PREFERRED_FIXTURE_SLOTS
        )
        if any(
            list_reference.list_name != list_reference.slot_name
            for list_reference in fixture_references
        ):
            return None
        referenced_fixture_slots = frozenset(
            list_reference.slot_name for list_reference in fixture_references
        )
        if not referenced_fixture_slots:
            return None
        if not isinstance(sentence, Sentence):
            return None
        if _sample_sentence is None:
            return None
        samples = _sample_sentence(
            sentence,
            slot_lists,
            expansion_rules,
            language=language,
            skip_optionals=True,
        )
        fixture_samples = []
        for sample_index, raw_sample in enumerate(islice(samples, MAX_SAMPLES_PER_SENTENCE)):
            sample = raw_sample.strip()
            unmatched_sample = sample
            sampled_fixture_slots: set[str] = set()
            for slot in sorted(
                referenced_fixture_slots,
                key=lambda referenced_slot: len(fixture_values[referenced_slot]),
                reverse=True,
            ):
                fixture_value = fixture_values[slot]
                if fixture_value in unmatched_sample:
                    sampled_fixture_slots.add(slot)
                    unmatched_sample = unmatched_sample.replace(fixture_value, "", 1)
            if sampled_fixture_slots:
                fixture_slots = frozenset(sampled_fixture_slots)
                preferred_slot = next(
                    slot for slot in _PREFERRED_FIXTURE_SLOTS if slot in fixture_slots
                )
                fixture_samples.append((sample_index, sample, preferred_slot, fixture_slots))
    except (MissingListError, MissingRuleError, ValueError):
        return None
    if not fixture_samples:
        return None
    _, sample, preferred_slot, fixture_slots = min(
        fixture_samples,
        key=lambda indexed_value: (
            indexed_value[1].count(FIXTURE_AREA_NAME) + indexed_value[1].count(FIXTURE_FLOOR_NAME),
            indexed_value[0],
        ),
    )
    return preferred_slot, fixture_slots, sample


def _turn_on_sample_candidates(
    intents: Intents,
    intent: Intent,
    slot_lists: dict[str, SlotList],
    fixture_values: Mapping[str, str],
    target_domain: str,
    language: str,
) -> list[tuple[int, int, tuple[int, int], str, tuple[tuple[str, str], ...]]]:
    """Return ranked fixture-bound HassTurnOn sentence samples."""
    candidates = []
    for data_index, intent_data in enumerate(intent.data):
        if not _intent_data_supports_domain(intent_data, target_domain):
            continue
        fixed_fixture_slots = {
            slot: intent_data.slots[slot]
            for slot in _PREFERRED_FIXTURE_SLOTS
            if slot in intent_data.slots
        }
        if any(
            not isinstance(value, str) or value != fixture_values[slot]
            for slot, value in fixed_fixture_slots.items()
        ):
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
            expected_slots = tuple(
                (slot, fixture_values[slot])
                for slot in _PREFERRED_FIXTURE_SLOTS
                if slot in fixture_slots or slot in fixed_fixture_slots
            )
            candidates.append(
                (
                    _PREFERRED_FIXTURE_SLOTS.index(preferred_slot),
                    len(expected_slots),
                    (data_index, sentence_index),
                    sample,
                    expected_slots,
                )
            )
    return candidates


def _sample_turn_on_command(
    language: str, entity_name: str, target_domain: str
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Sample a fixture-bound command without evaluating it in-process."""
    if _sample_sentence is None:
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
        selected = min(candidates)
        return selected[3], selected[4]
    raise ValueError(f"Unable to generate a {target_domain} HassTurnOn command for {language!r}")
