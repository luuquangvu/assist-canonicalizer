"""Candidate utterance models and deduplication."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import orjson

from .const import ENTITY_SLOT_NAME_SET, LOCATION_SLOT_NAME_SET
from .normalization import (
    literal_token_variants,
    normalize_text,
    normalize_text_no_diacritics_from_normalized,
    tokenize_normalized,
)
from .utils import wildcard_slot_names_sorted


class CandidateSource(StrEnum):
    """Known sources for candidate utterances."""

    CUSTOM_SENTENCE = "custom_sentence"
    BUILT_IN = "built_in"
    GENERATED_SAMPLE = "generated_sample"


_SOURCE_PRIORITY = {
    CandidateSource.CUSTOM_SENTENCE: 0,
    CandidateSource.BUILT_IN: 1,
    CandidateSource.GENERATED_SAMPLE: 2,
}


def candidate_source_priority(source: CandidateSource) -> int:
    """Return lower priority values for more trusted candidate sources."""
    try:
        return _SOURCE_PRIORITY[source]
    except KeyError:
        raise ValueError(f"No priority configured for candidate source: {source!r}") from None


@dataclass(frozen=True, slots=True)
class Candidate:
    """Canonical utterance candidate."""

    text: str
    intent_name: str
    source: CandidateSource = CandidateSource.GENERATED_SAMPLE
    language: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    slot_values: tuple[str, ...] = field(default_factory=tuple)
    normalized_text: str = ""
    _normalized_text_no_diacritics: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _normalized_tokens: tuple[str, ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _normalized_tokens_set: frozenset[str] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _slot_tokens_set: frozenset[str] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _literal_variants: tuple[frozenset[str], ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _has_implicit_literal_variant: bool | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _normalized_text_sorted: str | None = field(default=None, init=False, repr=False, compare=False)
    _total_unique_literal_tokens: int | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _wildcard_infos: tuple[tuple[int, str], ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _parsed_slots: dict[str, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _parsed_raw_slots: dict[str, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate and normalize candidate data."""
        if not self.text.strip():
            raise ValueError("Candidate text must not be empty")
        if not self.intent_name.strip():
            raise ValueError("Candidate intent name must not be empty")
        if not self.normalized_text:
            object.__setattr__(self, "normalized_text", normalize_text(self.text))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "slot_values", tuple(self.slot_values))

    def __hash__(self) -> int:
        """Return the hash code for Candidate."""
        return hash(
            (
                self.text,
                self.intent_name,
                self.source,
                self.language,
                frozenset(self.metadata.items()),
                self.slot_values,
            )
        )

    def __eq__(self, other: object) -> bool:
        """Return whether two candidates are equal."""
        if not isinstance(other, Candidate):
            return NotImplemented
        return (
            self.text == other.text
            and self.intent_name == other.intent_name
            and self.source == other.source
            and self.language == other.language
            and self.metadata == other.metadata
            and self.slot_values == other.slot_values
        )

    @property
    def normalized_text_no_diacritics(self) -> str:
        """Return the normalized text without diacritics."""
        val = self._normalized_text_no_diacritics
        if val is None:
            val = normalize_text_no_diacritics_from_normalized(self.normalized_text, self.language)
            object.__setattr__(self, "_normalized_text_no_diacritics", val)
        return val

    @property
    def normalized_tokens(self) -> tuple[str, ...]:
        """Return the sequence of normalized tokens."""
        val = self._normalized_tokens
        if val is None:
            val = tokenize_normalized(self.normalized_text)
            object.__setattr__(self, "_normalized_tokens", val)
        return val

    @property
    def normalized_tokens_set(self) -> frozenset[str]:
        """Return the set of normalized tokens."""
        val = self._normalized_tokens_set
        if val is None:
            val = frozenset(self.normalized_tokens)
            object.__setattr__(self, "_normalized_tokens_set", val)
        return val

    @property
    def slot_tokens_set(self) -> frozenset[str]:
        """Return the set of normalized slot tokens."""
        val = self._slot_tokens_set
        if val is None:
            slot_values = _non_static_slot_values(self.metadata)
            if slot_values is None:
                slot_values = self.slot_values
            if slot_values:
                cand_slot_text = normalize_text(" ".join(slot_values))
                val = frozenset(cand_slot_text.split())
            else:
                val = frozenset()
            object.__setattr__(self, "_slot_tokens_set", val)
        return val

    @property
    def normalized_text_sorted(self) -> str:
        """Return the sorted normalized tokens joined by space."""
        val = self._normalized_text_sorted
        if val is None:
            val = " ".join(sorted(self.normalized_tokens))
            object.__setattr__(self, "_normalized_text_sorted", val)
        return val

    @property
    def literal_variants(self) -> tuple[frozenset[str], ...]:
        """Return the set of literal variants."""
        val = self._literal_variants
        if val is None:
            if variants_text := self.metadata.get("literal_variants"):
                try:
                    loaded = orjson.loads(variants_text)
                    val = _literal_variants_from_loaded(loaded)
                except orjson.JSONDecodeError:
                    val = None
                if val is None:
                    literal_text = self.metadata.get("literal_text")
                    val = literal_token_variants(literal_text) if literal_text else ()
            else:
                literal_text = self.metadata.get("literal_text")
                val = literal_token_variants(literal_text) if literal_text else ()
            object.__setattr__(self, "_literal_variants", val)
        return val

    @property
    def has_implicit_literal_variant(self) -> bool:
        """Return whether the grammar permits this intent without literal evidence."""
        val = self._has_implicit_literal_variant
        if val is None:
            variants = self.literal_variants
            val = not variants or not all(variants)
            object.__setattr__(self, "_has_implicit_literal_variant", val)
        return val

    @property
    def total_unique_literal_tokens(self) -> int:
        """Return the total number of unique literal tokens."""
        val = self._total_unique_literal_tokens
        if val is None:
            variants = self.literal_variants
            val = len({tok for var in variants for tok in var}) if variants else 0
            object.__setattr__(self, "_total_unique_literal_tokens", val)
        return val

    @property
    def source_priority(self) -> int:
        """Return lower priority values for more trusted candidate sources."""
        return candidate_source_priority(self.source)

    @property
    def wildcard_infos(self) -> tuple[tuple[int, str], ...]:
        """Return all (index, wildcard_name) pairs of wildcard tokens."""
        infos = self._wildcard_infos
        if infos is None:
            sentence_template = self.metadata.get("sentence_template")
            wildcard_slots_meta = self.metadata.get("wildcard_slots")
            if sentence_template is not None:
                wildcards = (
                    tuple(slot for slot in wildcard_slots_meta.split(",") if slot)
                    if wildcard_slots_meta
                    else ()
                )
            else:
                wildcards = wildcard_slot_names_sorted(self.language)
            infos_list: list[tuple[int, str]] = []
            if wildcards:
                text = self.text
                if any(wc in text for wc in wildcards):
                    for idx, token in enumerate(self.normalized_tokens):
                        for wc in wildcards:
                            if token == wc or (
                                wc in token
                                and ("_" in wc or token.startswith(wc) or token.endswith(wc))
                            ):
                                infos_list.append((idx, wc))
                                break
            infos = tuple(infos_list)
            object.__setattr__(self, "_wildcard_infos", infos)
        return infos

    @property
    def has_wildcard(self) -> bool:
        """Return whether the candidate text contains any wildcard placeholders."""
        return bool(self.wildcard_infos)

    @property
    def parsed_slots(self) -> dict[str, Any]:
        """Return decoded slot metadata, cached after the first access."""
        val = self._parsed_slots
        if val is None:
            val = candidate_slot_map(self)
            object.__setattr__(self, "_parsed_slots", val)
        return val

    @property
    def parsed_raw_slots(self) -> dict[str, Any]:
        """Return decoded raw-slot metadata, cached after the first access."""
        val = self._parsed_raw_slots
        if val is None:
            val = candidate_raw_slot_map(self)
            object.__setattr__(self, "_parsed_raw_slots", val)
        return val


def _non_static_slot_values(metadata: Mapping[str, str]) -> tuple[str, ...] | None:
    """Return non-static slot values from serialized candidate metadata."""
    slots_text = metadata.get("slots")
    if not isinstance(slots_text, str) or not slots_text:
        return None
    try:
        slots = orjson.loads(slots_text)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(slots, dict):
        return None
    static_slots_text = metadata.get("static_slots", "")
    static_slots = (
        {slot for slot in static_slots_text.split(",") if slot}
        if isinstance(static_slots_text, str)
        else set()
    )
    values: list[str] = []
    for name, value in slots.items():
        if not isinstance(name, str) or name in static_slots:
            continue
        if isinstance(value, str) and value:
            values.append(value)
    return tuple(values)


def _literal_variants_from_loaded(loaded: object) -> tuple[frozenset[str], ...] | None:
    """Return cached literal variants only when JSON shape is valid."""
    if not isinstance(loaded, list):
        return None
    variants: list[frozenset[str]] = []
    for variant in loaded:
        if not isinstance(variant, list):
            return None
        tokens: list[str] = []
        for token in variant:
            if not isinstance(token, str):
                return None
            tokens.append(token)
        variants.append(frozenset(tokens))
    return tuple(variants)


def _decode_candidate_metadata_dict(candidate: Candidate, key: str) -> dict[str, Any]:
    """Helper to read, validate, JSON decode, and check dict for candidate metadata key."""
    val = candidate.metadata.get(key)
    if not isinstance(val, str) or not val:
        return {}
    try:
        decoded = orjson.loads(val)
    except orjson.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def candidate_slot_map(candidate: Candidate) -> dict[str, Any]:
    """Return decoded candidate slot metadata."""
    return _decode_candidate_metadata_dict(candidate, "slots")


def candidate_raw_slot_map(candidate: Candidate) -> dict[str, Any]:
    """Return decoded raw candidate slot metadata."""
    return _decode_candidate_metadata_dict(candidate, "slots_raw")


def slot_alias_values_by_key(
    slots: Mapping[str, Any],
    slot_mappings: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, tuple[Any, ...]]:
    """Return slot values indexed by direct and aliased slot names."""
    aliases: dict[str, list[Any]] = {}
    slot_mappings = slot_mappings or {}
    for slot_name, slot_value in slots.items():
        candidate_keys = [slot_name]
        if ":" in slot_name:
            shared_slot_name = slot_name.split(":", 1)[0]
            candidate_keys.append(shared_slot_name)
            candidate_keys.extend(slot_mappings.get(shared_slot_name, ()))
        candidate_keys.extend(slot_mappings.get(slot_name, ()))

        seen_keys: set[str] = set()
        for candidate_key in candidate_keys:
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)
            aliases.setdefault(candidate_key, []).append(slot_value)
    return {slot_name: tuple(values) for slot_name, values in aliases.items()}


def candidate_dedupe_preference_key(candidate: Candidate) -> tuple[int, int, int, int]:
    """Return a source-first HassIL-aligned candidate dedupe key."""
    slots = candidate_slot_map(candidate)
    has_generic_entity = bool(slots.keys() & ENTITY_SLOT_NAME_SET)
    has_location = bool(slots.keys() & LOCATION_SLOT_NAME_SET)
    has_domain = "domain" in slots
    return (
        -candidate.source_priority,
        int(has_location and has_domain),
        int(has_generic_entity),
        int(not has_domain),
    )


def deduplicate_candidates(candidates: list[Candidate]) -> tuple[Candidate, ...]:
    """Deduplicate by (normalized text, intent name) preserving best source priority.

    Candidates with identical text but different intents are kept, the downstream
    HassIL validation loop resolves the intent based on real home context
    (requires_context / excludes_context).
    """
    selected: dict[tuple[str, str], tuple[Candidate, tuple[int, int, int, int]]] = {}
    for candidate in candidates:
        key = (candidate.normalized_text, candidate.intent_name)
        pref_key = candidate_dedupe_preference_key(candidate)
        existing_info = selected.get(key)
        if existing_info is None or pref_key > existing_info[1]:
            selected[key] = (candidate, pref_key)

    return tuple(
        sorted(
            (item[0] for item in selected.values()),
            key=lambda item: (item.source_priority, item.normalized_text),
        )
    )
