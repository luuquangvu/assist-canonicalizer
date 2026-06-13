"""Candidate loading from Home Assistant conversation intent sources."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import orjson

from .candidate import Candidate, CandidateSource
from .const import (
    DEFAULT_MAX_CANDIDATES_PER_INTENT,
    DEFAULT_MAX_CANDIDATES_PER_TEMPLATE,
    DEFAULT_MAX_DYNAMIC_CANDIDATES,
    DEFAULT_MAX_DYNAMIC_SLOT_VALUES,
    DEFAULT_MAX_TOTAL_CANDIDATES_PER_LANGUAGE,
)
from .normalization import normalize_text, normalize_text_no_diacritics
from .ranking import _literal_token_variants
from .registry import AREA_SLOT_NAMES, ENTITY_SLOT_NAMES, merge_slot_values

_TEMPLATE_MARKERS = frozenset("{}[]<>|()")
_SLOT_PATTERN = re.compile(r"{([^{}]+)}")
_RULE_PATTERN = re.compile(r"<([^<>]+)>")
_ENTITY_SLOT_NAMES = frozenset(ENTITY_SLOT_NAMES)
_AREA_SLOT_NAMES = frozenset(AREA_SLOT_NAMES)


@dataclass(frozen=True, slots=True)
class RegistrySlotValue:
    """Precomputed lexical data for one registry slot value."""

    text: str
    normalized_text: str
    tokens: tuple[str, ...]
    position: int
    normalized_no_diacritics: str
    tokens_no_diacritics: tuple[str, ...]


class RegistrySlotIndex(dict[str, tuple[RegistrySlotValue, ...]]):
    """Precomputed registry values index with an inverted index helper."""

    def __init__(self, data: dict[str, tuple[RegistrySlotValue, ...]]):
        """Initialize and precompute inverted indexes for all slots."""
        super().__init__(data)
        self.inverted: dict[str, dict[str, list[RegistrySlotValue]]] = {}
        for key, records in self.items():
            slot_inv = self.inverted.setdefault(key, {})
            for record in records:
                for token in record.tokens:
                    slot_inv.setdefault(token, []).append(record)
                for token in record.tokens_no_diacritics:
                    if token not in record.tokens:
                        slot_inv.setdefault(token, []).append(record)
        self._inverted_cache: dict[int, dict[str, list[RegistrySlotValue]]] = {}

    def get_inverted_for_records(
        self, records: tuple[RegistrySlotValue, ...]
    ) -> dict[str, list[RegistrySlotValue]]:
        """Get or build the inverted index for the given records tuple."""
        record_id = id(records)
        lookup = self._inverted_cache.get(record_id)
        if lookup is not None:
            return lookup

        lookup = {}
        for record in records:
            for token in record.tokens:
                lookup.setdefault(token, []).append(record)
            for token in record.tokens_no_diacritics:
                if token not in record.tokens:
                    lookup.setdefault(token, []).append(record)
        self._inverted_cache[record_id] = lookup
        return lookup


@dataclass(frozen=True, slots=True)
class DynamicRegistryTemplate:
    """Query-independent data needed to expand one registry template."""

    sentence: str
    sentence_slots: frozenset[str]
    entity_slots: tuple[str, ...]
    domains: tuple[str, ...]
    expansion_rules: Mapping[str, str]
    base_data_slot_values: Mapping[str, tuple[str, ...]]
    required_slots: frozenset[str]
    metadata: Mapping[str, str]
    literal_token_variants: tuple[frozenset[str], ...]
    literal_token_variants_no_diac: tuple[frozenset[str], ...]


@dataclass(frozen=True, slots=True)
class DynamicRegistryIntent:
    """Compiled registry templates for one intent."""

    source: CandidateSource
    intent_name: str
    templates: tuple[DynamicRegistryTemplate, ...]


def build_candidates_from_intent_sources(
    language: str,
    intent_sources: Mapping[str, Mapping[str, Any]],
    registry_slot_values: Mapping[str, tuple[str, ...]] | None = None,
    *,
    max_candidates: int | None = DEFAULT_MAX_TOTAL_CANDIDATES_PER_LANGUAGE,
) -> tuple[Candidate, ...]:
    """Build a bounded candidate set from conversation intent source configs."""
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    candidates: list[Candidate] = []
    for source_key, source_config in intent_sources.items():
        candidate_source = _candidate_source_from_key(source_key)
        intents = source_config.get("intents", {})
        if not isinstance(intents, Mapping):
            continue
        for intent_name, intent_config in intents.items():
            if not isinstance(intent_name, str) or not isinstance(intent_config, Mapping):
                continue
            remaining = (
                DEFAULT_MAX_CANDIDATES_PER_INTENT
                if max_candidates is None
                else max_candidates - len(candidates)
            )
            if remaining <= 0:
                return tuple(candidates)
            intent_cap = min(DEFAULT_MAX_CANDIDATES_PER_INTENT, remaining)
            candidates.extend(
                _candidates_from_intent_config(
                    language,
                    source_key,
                    source_config,
                    candidate_source,
                    intent_name,
                    intent_config,
                    registry_slot_values or {},
                    max_candidates=intent_cap,
                )
            )
            if max_candidates is not None and len(candidates) >= max_candidates:
                return tuple(candidates[:max_candidates])
    return tuple(candidates)


def build_registry_slot_index(
    registry_slot_values: Mapping[str, tuple[str, ...]],
    language: str | None = None,
) -> RegistrySlotIndex:
    """Precompute normalized registry values while sharing identical slot tuples."""
    shared: dict[tuple[str, ...], tuple[RegistrySlotValue, ...]] = {}
    data: dict[str, tuple[RegistrySlotValue, ...]] = {}
    for slot_name, values in registry_slot_values.items():
        records = shared.get(values)
        if records is None:
            records = tuple(
                RegistrySlotValue(
                    text=value,
                    normalized_text=normalized,
                    tokens=tokens,
                    position=position,
                    normalized_no_diacritics=normalized_no_diac,
                    tokens_no_diacritics=tuple(dict.fromkeys(normalized_no_diac.split())),
                )
                for position, value in enumerate(values)
                if (normalized := normalize_text(value))
                and (tokens := tuple(dict.fromkeys(normalized.split())))
                and (normalized_no_diac := _cached_normalize_no_diac(value, language)) is not None
            )
            shared[values] = records
        data[slot_name] = records
    return RegistrySlotIndex(data)


def compile_dynamic_registry_intents(
    intent_sources: Mapping[str, Mapping[str, Any]],
    language: str | None = None,
) -> tuple[DynamicRegistryIntent, ...]:
    """Compile query-independent dynamic registry template data."""
    compiled: list[DynamicRegistryIntent] = []
    for source_key, source_config in intent_sources.items():
        source = _candidate_source_from_key(source_key)
        intents = source_config.get("intents", {})
        if not isinstance(intents, Mapping):
            continue
        for intent_name, intent_config in intents.items():
            if not isinstance(intent_name, str) or not isinstance(intent_config, Mapping):
                continue
            templates: list[DynamicRegistryTemplate] = []
            data_items = intent_config.get("data", [])
            if not isinstance(data_items, list):
                continue
            for data_item in data_items:
                if not isinstance(data_item, Mapping):
                    continue
                sentences = data_item.get("sentences", [])
                if not isinstance(sentences, list):
                    continue
                expansion_rules = _expansion_rules(source_config, intent_config, data_item)
                base_data_slot_values = _slot_values(source_config, intent_config, data_item)
                domains = _context_domains(data_item)
                for sentence in sentences:
                    if not isinstance(sentence, str):
                        continue
                    sentence_slots = _template_slot_names(sentence, expansion_rules)
                    entity_slots = tuple(sorted(sentence_slots & _ENTITY_SLOT_NAMES))
                    if not entity_slots:
                        continue
                    variants = _template_literal_token_variants(sentence, expansion_rules)
                    no_diac_variants = tuple(
                        frozenset(_cached_normalize_no_diac(token, language) for token in tokens)
                        for tokens in variants
                    )
                    templates.append(
                        DynamicRegistryTemplate(
                            sentence=sentence,
                            sentence_slots=sentence_slots,
                            entity_slots=entity_slots,
                            domains=domains,
                            expansion_rules=expansion_rules,
                            base_data_slot_values=base_data_slot_values,
                            required_slots=frozenset(_required_slots(sentence, expansion_rules)),
                            metadata=_candidate_metadata(source_key, sentence, expansion_rules),
                            literal_token_variants=variants,
                            literal_token_variants_no_diac=no_diac_variants,
                        )
                    )
            if templates:
                compiled.append(
                    DynamicRegistryIntent(
                        source=source,
                        intent_name=intent_name,
                        templates=tuple(templates),
                    )
                )
    return tuple(compiled)


def build_query_registry_candidates(
    language: str,
    intent_sources: Mapping[str, Mapping[str, Any]],
    registry_slot_values: Mapping[str, tuple[str, ...]],
    query: str,
    *,
    max_candidates: int = DEFAULT_MAX_DYNAMIC_CANDIDATES,
    registry_slot_index: RegistrySlotIndex | None = None,
    compiled_intents: Sequence[DynamicRegistryIntent] | None = None,
) -> tuple[Candidate, ...]:
    """Build query-scoped registry candidates without expanding every entity."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    query_normalized = normalize_text(query)
    if not query_normalized or not registry_slot_values:
        return ()
    query_tokens = frozenset(query_normalized.split())
    if registry_slot_index is None:
        registry_slot_index = build_registry_slot_index(registry_slot_values, language)
    if compiled_intents is None:
        compiled_intents = compile_dynamic_registry_intents(intent_sources, language)

    query_no_diac = normalize_text_no_diacritics(query, language)
    query_tokens_no_diac = frozenset(query_no_diac.split())

    candidates: list[Candidate] = []
    relevant_cache: dict[int, tuple[str, ...]] = {}
    scoped_cache: dict[tuple[str, tuple[str, ...]], tuple[RegistrySlotValue, ...]] = {}
    for compiled_intent in compiled_intents:
        candidates.extend(
            _query_candidates_from_compiled_intent(
                language,
                compiled_intent,
                registry_slot_values,
                registry_slot_index,
                query_normalized,
                query_tokens,
                relevant_cache,
                scoped_cache,
                max_candidates=DEFAULT_MAX_CANDIDATES_PER_INTENT,
                query_no_diac=query_no_diac,
                query_tokens_no_diac=query_tokens_no_diac,
            )
        )
    if len(candidates) <= max_candidates:
        return tuple(candidates)
    candidates.sort(
        key=lambda candidate: _query_candidate_relevance_key(
            candidate, query_normalized, query_tokens
        ),
        reverse=True,
    )
    return tuple(candidates[:max_candidates])


def _query_candidate_relevance_key(
    candidate: Candidate,
    query_normalized: str,
    query_tokens: frozenset[str],
) -> tuple[int, int, float, int]:
    """Return a stable relevance key for capping query-scoped candidates."""
    candidate_normalized = candidate.normalized_text
    if not candidate_normalized:
        return (0, 0, 0.0, 0)
    candidate_tokens = candidate.normalized_tokens_set
    union_size = len(query_tokens | candidate_tokens)
    overlap_score = (len(query_tokens & candidate_tokens) / union_size) if union_size else 0.0
    return (
        int(candidate_normalized == query_normalized),
        int(query_normalized in candidate_normalized or candidate_normalized in query_normalized),
        overlap_score,
        -abs(len(candidate_normalized) - len(query_normalized)),
    )


def is_fixed_sentence(sentence: str) -> bool:
    """Return whether a sentence has no Hassil template syntax."""
    return bool(sentence.strip()) and not any(marker in sentence for marker in _TEMPLATE_MARKERS)


def expand_sentence_template(
    sentence: str,
    slot_values: Mapping[str, tuple[str, ...]],
    expansion_rules: Mapping[str, str],
    *,
    max_expansions: int = DEFAULT_MAX_CANDIDATES_PER_TEMPLATE,
) -> tuple[str, ...]:
    """Expand a bounded subset of Hassil sentence template syntax."""
    if max_expansions < 1:
        raise ValueError("max_expansions must be positive")
    top_node = _parse_hassil(sentence)
    expansions = top_node.expand(slot_values, expansion_rules, frozenset(), max_expansions)
    non_empty = [_clean_expanded_text(v) for v in expansions if v.strip()]
    deduplicated = tuple(_deduplicate_texts(non_empty, max_expansions))
    return deduplicated[:max_expansions]


def _build_candidate(
    expanded_sentence: str,
    intent_name: str,
    source: CandidateSource,
    language: str,
    base_metadata: Mapping[str, str],
    presorted_values: dict[str, list[str]],
) -> Candidate:
    """Construct a Candidate with extracted slot metadata."""
    slots = _extract_slots_from_expanded_text(expanded_sentence, presorted_values)
    metadata = dict(base_metadata)
    if slots:
        metadata["slots"] = orjson.dumps(slots).decode("utf-8")
    return Candidate(
        text=expanded_sentence,
        intent_name=intent_name,
        source=source,
        language=language,
        metadata=metadata,
    )


def _candidates_from_intent_config(
    language: str,
    source_key: str,
    source_config: Mapping[str, Any],
    source: CandidateSource,
    intent_name: str,
    intent_config: Mapping[str, Any],
    registry_slot_values: Mapping[str, tuple[str, ...]],
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES_PER_INTENT,
) -> tuple[Candidate, ...]:
    """Extract fixed and bounded template candidates from one intent config.

    Data items whose templates use ``{name}`` slots are processed first so
    that generic entity coverage is not starved by area-only or
    expansion-rule-only templates that consume the per-intent cap.
    """
    if max_candidates < 1:
        return ()
    candidates: list[Candidate] = []
    data_items = intent_config.get("data", [])
    if not isinstance(data_items, list):
        return ()
    _name_data_items: list[Mapping[str, Any]] = []
    _other_data_items: list[Mapping[str, Any]] = []
    for di in data_items:
        if not isinstance(di, Mapping):
            continue
        sentences = di.get("sentences", [])
        if not isinstance(sentences, list):
            continue
        has_name = any(isinstance(s, str) and "{name}" in s for s in sentences)
        if has_name:
            _name_data_items.append(di)
        else:
            _other_data_items.append(di)
    ordered_data_items = _name_data_items + _other_data_items
    for data_item in ordered_data_items:
        if not isinstance(data_item, Mapping):
            continue
        sentences = data_item.get("sentences", [])
        if not isinstance(sentences, list):
            continue
        expansion_rules = _expansion_rules(source_config, intent_config, data_item)
        base_data_slot_values = _slot_values(source_config, intent_config, data_item)
        for sentence in sentences:
            if not isinstance(sentence, str):
                continue
            sentence_slots = _template_slot_names(sentence, expansion_rules)
            slot_values = merge_slot_values(
                base_data_slot_values,
                _registry_slot_values_for_template(
                    data_item,
                    sentence_slots=sentence_slots,
                    registry_slot_values=registry_slot_values,
                ),
            )
            required = _required_slots(sentence, expansion_rules)
            if any(not slot_values.get(slot) for slot in required):
                continue
            base_metadata = _candidate_metadata(source_key, sentence, expansion_rules)
            presorted_values = _presort_slot_values(slot_values)
            for expanded_sentence in _candidate_texts(sentence, slot_values, expansion_rules):
                candidates.append(
                    _build_candidate(
                        expanded_sentence,
                        intent_name,
                        source,
                        language,
                        base_metadata,
                        presorted_values,
                    )
                )
                if len(candidates) >= max_candidates:
                    return tuple(candidates)
    return tuple(candidates)


def _query_candidates_from_compiled_intent(
    language: str,
    compiled_intent: DynamicRegistryIntent,
    registry_slot_values: Mapping[str, tuple[str, ...]],
    registry_slot_index: RegistrySlotIndex,
    query_normalized: str,
    query_tokens: frozenset[str],
    relevant_cache: dict[int, tuple[str, ...]],
    scoped_cache: dict[tuple[str, tuple[str, ...]], tuple[RegistrySlotValue, ...]],
    *,
    max_candidates: int,
    query_no_diac: str | None = None,
    query_tokens_no_diac: frozenset[str] | None = None,
) -> tuple[Candidate, ...]:
    """Expand one compiled intent using query-relevant registry values."""
    candidates: list[Candidate] = []
    for template in compiled_intent.templates:
        if not _literal_token_variants_match_query(
            template.literal_token_variants,
            template.literal_token_variants_no_diac,
            query_tokens,
            query_tokens_no_diac=query_tokens_no_diac,
        ):
            continue
        dynamic_registry_slots = _compiled_query_registry_slot_values(
            template,
            registry_slot_values,
            registry_slot_index,
            query_normalized,
            query_tokens,
            relevant_cache,
            scoped_cache,
            query_no_diac=query_no_diac,
            query_tokens_no_diac=query_tokens_no_diac,
        )
        if not dynamic_registry_slots:
            continue
        slot_values = merge_slot_values(
            template.base_data_slot_values,
            dynamic_registry_slots,
        )
        if any(not slot_values.get(slot) for slot in template.required_slots):
            continue
        presorted_values = _presort_slot_values(slot_values)
        for expanded_sentence in _candidate_texts(
            template.sentence,
            slot_values,
            template.expansion_rules,
        ):
            candidates.append(
                _build_candidate(
                    expanded_sentence,
                    compiled_intent.intent_name,
                    compiled_intent.source,
                    language,
                    template.metadata,
                    presorted_values,
                )
            )
            if len(candidates) >= max_candidates:
                return tuple(candidates)
    return tuple(candidates)


def _compiled_query_registry_slot_values(
    template: DynamicRegistryTemplate,
    registry_slot_values: Mapping[str, tuple[str, ...]],
    registry_slot_index: RegistrySlotIndex,
    query_normalized: str,
    query_tokens: frozenset[str],
    relevant_cache: dict[int, tuple[str, ...]],
    scoped_cache: dict[tuple[str, tuple[str, ...]], tuple[RegistrySlotValue, ...]],
    *,
    query_no_diac: str | None = None,
    query_tokens_no_diac: frozenset[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return narrowed registry slots for a compiled template."""
    constrained = dict(registry_slot_values)
    if template.sentence_slots & _AREA_SLOT_NAMES:
        for slot_name in AREA_SLOT_NAMES:
            constrained.pop(slot_name, None)

    if template.domains:
        for slot_name in _ENTITY_SLOT_NAMES:
            constrained.pop(slot_name, None)
            records = _scoped_registry_slot_records(
                slot_name,
                template.domains,
                registry_slot_index,
                scoped_cache,
            )
            if records:
                constrained[slot_name] = tuple(record.text for record in records)

    matched_entity_slot = False
    for slot_name in template.entity_slots:
        records = _scoped_registry_slot_records(
            slot_name,
            template.domains,
            registry_slot_index,
            scoped_cache,
        )
        if not records:
            constrained.pop(slot_name, None)
            continue
        cache_key = id(records)
        relevant = relevant_cache.get(cache_key)
        if relevant is None:
            relevant = _query_relevant_precomputed_slot_values(
                records,
                query_normalized,
                query_tokens,
                registry_slot_index=registry_slot_index,
                query_no_diac=query_no_diac,
                query_tokens_no_diac=query_tokens_no_diac,
            )
            relevant_cache[cache_key] = relevant
        if relevant:
            constrained[slot_name] = relevant
            matched_entity_slot = True
        else:
            constrained.pop(slot_name, None)
    return constrained if matched_entity_slot else {}


def _scoped_registry_slot_records(
    slot_name: str,
    domains: tuple[str, ...],
    registry_slot_index: Mapping[str, tuple[RegistrySlotValue, ...]],
    scoped_cache: dict[tuple[str, tuple[str, ...]], tuple[RegistrySlotValue, ...]],
) -> tuple[RegistrySlotValue, ...]:
    """Return precomputed registry records for a generic or domain-scoped slot."""
    if not domains:
        return registry_slot_index.get(slot_name, ())
    cache_key = (slot_name, domains)
    cached = scoped_cache.get(cache_key)
    if cached is not None:
        return cached
    if len(domains) == 1:
        records = registry_slot_index.get(f"{slot_name}:{domains[0]}", ())[
            :DEFAULT_MAX_CANDIDATES_PER_INTENT
        ]
        scoped_cache[cache_key] = records
        return records

    selected: list[RegistrySlotValue] = []
    seen: set[str] = set()
    for domain in domains:
        for record in registry_slot_index.get(f"{slot_name}:{domain}", ()):
            if record.text in seen:
                continue
            seen.add(record.text)
            selected.append(record)
            if len(selected) >= DEFAULT_MAX_CANDIDATES_PER_INTENT:
                records = tuple(selected)
                scoped_cache[cache_key] = records
                return records
    records = tuple(selected)
    scoped_cache[cache_key] = records
    return records


def _query_relevant_precomputed_slot_values(
    values: tuple[RegistrySlotValue, ...],
    query_normalized: str,
    query_tokens: frozenset[str],
    *,
    limit: int = DEFAULT_MAX_DYNAMIC_SLOT_VALUES,
    registry_slot_index: RegistrySlotIndex | None = None,
    query_no_diac: str | None = None,
    query_tokens_no_diac: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """Return relevant values using precomputed normalization and tokens."""
    if registry_slot_index is None:
        candidate_records = set(values)
    else:
        lookup = registry_slot_index.get_inverted_for_records(values)
        candidate_records = set()
        for token in query_tokens:
            if token in lookup:
                candidate_records.update(lookup[token])
        if query_tokens_no_diac:
            for token in query_tokens_no_diac:
                if token in lookup:
                    candidate_records.update(lookup[token])

    scored: list[tuple[tuple[bool, bool, bool, int, int], int, str]] = []
    for value in candidate_records:
        match_key = _slot_value_query_match_key_from_tokens(
            value.normalized_text,
            value.tokens,
            query_normalized,
            query_tokens,
            value_no_diac=value.normalized_no_diacritics,
            value_tokens_no_diac=value.tokens_no_diacritics,
            query_no_diac=query_no_diac,
            query_tokens_no_diac=query_tokens_no_diac,
        )
        if match_key is not None:
            scored.append((match_key, value.position, value.text))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return tuple(_deduplicate_texts((value for _, _, value in scored), limit))


def _candidate_metadata(
    source_key: str,
    sentence: str,
    expansion_rules: Mapping[str, str],
) -> dict[str, str]:
    """Return metadata for ranking candidates by localized template literals."""
    metadata = {"intent_source": source_key, "sentence_template": sentence}
    literal_text = _template_literal_text(sentence, expansion_rules)
    if literal_text:
        metadata["literal_text"] = literal_text
    return metadata


def _presort_slot_values(
    slot_values: Mapping[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    """Return slot values pre-sorted longest-first for substring extraction."""
    return {
        slot_name: sorted(values, key=len, reverse=True)
        for slot_name, values in slot_values.items()
    }


def _extract_slots_from_expanded_text(
    text: str,
    slot_values: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Extract slot values present in the expanded text.

    Values must be pre-sorted longest-first (see :func:`_presort_slot_values`).
    """
    slots = {}
    for slot_name, values in slot_values.items():
        for val in values:
            if val in text:
                slots[slot_name] = val
                break
    return slots


def _candidate_texts(
    sentence: str,
    slot_values: Mapping[str, tuple[str, ...]],
    expansion_rules: Mapping[str, str],
) -> tuple[str, ...]:
    """Return candidate texts for a sentence template."""
    if is_fixed_sentence(sentence):
        return (_clean_expanded_text(sentence),)
    return expand_sentence_template(
        sentence,
        slot_values,
        expansion_rules,
        max_expansions=DEFAULT_MAX_CANDIDATES_PER_TEMPLATE,
    )


def _registry_slot_values_for_template(
    data_item: Mapping[str, Any],
    *,
    sentence_slots: frozenset[str],
    registry_slot_values: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Return registry slots constrained by Hassil context and template shape."""
    constrained = dict(registry_slot_values)
    if sentence_slots & _ENTITY_SLOT_NAMES and sentence_slots & _AREA_SLOT_NAMES:
        for slot_name in AREA_SLOT_NAMES:
            constrained.pop(slot_name, None)

    domains = _context_domains(data_item)
    if not domains:
        return constrained
    for slot_name in _ENTITY_SLOT_NAMES:
        constrained.pop(slot_name, None)
        scoped_values = _scoped_slot_values(slot_name, domains, registry_slot_values)
        if scoped_values:
            constrained[slot_name] = scoped_values
    return constrained


def _slot_value_query_match_key_from_tokens(
    value_normalized: str,
    value_tokens: tuple[str, ...],
    query_normalized: str,
    query_tokens: frozenset[str],
    *,
    value_no_diac: str | None = None,
    value_tokens_no_diac: tuple[str, ...] | None = None,
    query_no_diac: str | None = None,
    query_tokens_no_diac: frozenset[str] | None = None,
) -> tuple[bool, bool, bool, int, int] | None:
    """Return registry relevance using precomputed normalized value tokens."""
    if not value_tokens:
        return None
    token_count = len(value_tokens)

    # 1. Exact matches
    exact = value_normalized == query_normalized
    if not exact and value_no_diac and query_no_diac:
        exact = value_no_diac == query_no_diac
    if exact:
        return (True, True, True, token_count, -token_count)

    # 2. Substring matches
    substring = value_normalized in query_normalized
    if not substring and value_no_diac and query_no_diac:
        substring = value_no_diac in query_no_diac
    if substring:
        return (False, True, True, token_count, -token_count)

    # 3. Token overlaps
    matched = len(query_tokens.intersection(value_tokens))
    if value_tokens_no_diac and query_tokens_no_diac:
        matched_no_diac = len(query_tokens_no_diac.intersection(value_tokens_no_diac))
        matched = max(matched, matched_no_diac)

    if matched < min(2, token_count):
        return None
    return (
        False,
        False,
        matched == token_count,
        matched,
        -token_count,
    )


def _literal_token_variants_match_query(
    literal_variants: tuple[frozenset[str], ...],
    literal_variants_no_diac: tuple[frozenset[str], ...],
    query_tokens: frozenset[str],
    *,
    query_tokens_no_diac: frozenset[str] | None = None,
) -> bool:
    """Return whether precomputed template literal variants match a query."""
    if not literal_variants:
        return True
    for literal_tokens in literal_variants:
        matched = len(literal_tokens & query_tokens)
        if matched == len(literal_tokens):
            return True
        if matched > 0 and matched / len(literal_tokens) >= 0.5:
            return True
    if query_tokens_no_diac:
        for literal_tokens_no_diac in literal_variants_no_diac:
            matched_no_diac = len(literal_tokens_no_diac & query_tokens_no_diac)
            if matched_no_diac == len(literal_tokens_no_diac):
                return True
            if matched_no_diac > 0 and matched_no_diac / len(literal_tokens_no_diac) >= 0.5:
                return True
    return False


def _template_literal_text(
    text: str,
    expansion_rules: Mapping[str, str],
) -> str:
    """Return localized literal words from a Hassil sentence template."""
    return _cached_template_literal_text(text, _expansion_rules_cache_key(expansion_rules))


def _template_literal_token_variants(
    text: str,
    expansion_rules: Mapping[str, str],
) -> tuple[frozenset[str], ...]:
    """Return normalized literal token variants for query matching."""
    return _cached_template_literal_token_variants(
        text, _expansion_rules_cache_key(expansion_rules)
    )


def _expansion_rules_cache_key(
    expansion_rules: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    """Return a stable cache key for expansion rules."""
    return tuple(sorted(expansion_rules.items()))


@lru_cache(maxsize=8192)
def _cached_normalize_no_diac(text: str, language: str | None = None) -> str:
    """Cache diacritics removal for efficient lookups."""
    return normalize_text_no_diacritics(text, language)


@lru_cache(maxsize=8192)
def _cached_template_literal_text(
    text: str,
    expansion_rules_key: tuple[tuple[str, str], ...],
) -> str:
    """Return cached localized literal words from a Hassil sentence template."""
    expansion_rules = dict(expansion_rules_key)
    top_node = _parse_hassil(text)
    variants = top_node.literal_variants(
        expansion_rules, frozenset(), DEFAULT_MAX_CANDIDATES_PER_TEMPLATE
    )
    non_empty = [_clean_expanded_text(v) for v in variants if v.strip()]
    deduplicated = tuple(dict.fromkeys(non_empty))
    return "|".join(deduplicated)


@lru_cache(maxsize=8192)
def _cached_template_literal_token_variants(
    text: str,
    expansion_rules_key: tuple[tuple[str, str], ...],
) -> tuple[frozenset[str], ...]:
    """Return cached normalized literal token variants for query matching."""
    literal_text = _cached_template_literal_text(text, expansion_rules_key)
    if not literal_text:
        return ()

    return _literal_token_variants(literal_text)


class _HassilNode:
    """Base class for Hassil template AST nodes."""

    def required_slots(self, rules: Mapping[str, str], seen: frozenset[str]) -> set[str]:
        """Return slot names required by this node."""
        raise NotImplementedError()

    def literal_variants(
        self,
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Return unique literal word variants for this node."""
        raise NotImplementedError()

    def expand(
        self,
        slot_values: Mapping[str, tuple[str, ...]],
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        raise NotImplementedError()


class _HassilTextNode(_HassilNode):
    """AST node representing literal text."""

    def __init__(self, text: str):
        """Initialize TextNode."""
        self.text = text

    def required_slots(self, rules: Mapping[str, str], seen: frozenset[str]) -> set[str]:
        """Return slot names required by this node."""
        return set()

    def literal_variants(
        self,
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Return unique literal word variants for this node."""
        return (self.text,)

    def expand(
        self,
        slot_values: Mapping[str, tuple[str, ...]],
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        return (self.text,)


class _HassilSlotNode(_HassilNode):
    """AST node representing a template slot."""

    def __init__(self, name: str):
        """Initialize SlotNode."""
        self.name = name

    def required_slots(self, rules: Mapping[str, str], seen: frozenset[str]) -> set[str]:
        """Return slot names required by this node."""
        return {self.name.split(":")[0]}

    def literal_variants(
        self,
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Return unique literal word variants for this node."""
        return ("",)

    def expand(
        self,
        slot_values: Mapping[str, tuple[str, ...]],
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        base_name = self.name.split(":")[0]
        values = slot_values.get(self.name) or slot_values.get(base_name)
        if not values:
            return ()
        return tuple(values[:limit])


class _HassilRuleNode(_HassilNode):
    """AST node representing an expansion rule reference."""

    def __init__(self, name: str):
        """Initialize RuleNode."""
        self.name = name

    def required_slots(self, rules: Mapping[str, str], seen: frozenset[str]) -> set[str]:
        """Return slot names required by this node."""
        if self.name in seen:
            return set()
        rule_text = rules.get(self.name)
        if rule_text is None:
            return set()
        node = _parse_hassil(rule_text)
        return node.required_slots(rules, seen | {self.name})

    def literal_variants(
        self,
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Return unique literal word variants for this node."""
        if self.name in seen:
            return ("",)
        rule_text = rules.get(self.name)
        if rule_text is None:
            return ("",)
        node = _parse_hassil(rule_text)
        return node.literal_variants(rules, seen | {self.name}, limit)

    def expand(
        self,
        slot_values: Mapping[str, tuple[str, ...]],
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        if self.name in seen:
            return ()
        rule_text = rules.get(self.name)
        if rule_text is None:
            return ()
        node = _parse_hassil(rule_text)
        return node.expand(slot_values, rules, seen | {self.name}, limit)


class _HassilOptionalNode(_HassilNode):
    """AST node representing an optional group."""

    def __init__(self, child: _HassilNode):
        """Initialize OptionalNode."""
        self.child = child

    def required_slots(self, rules: Mapping[str, str], seen: frozenset[str]) -> set[str]:
        """Return slot names required by this node."""
        return set()

    def literal_variants(
        self,
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Return unique literal word variants for this node."""
        child_variants = self.child.literal_variants(rules, seen, limit)
        unique_variants: dict[str, None] = {"": None}
        for v in child_variants:
            if v not in unique_variants:
                unique_variants[v] = None
                if len(unique_variants) >= limit:
                    break
        return tuple(unique_variants.keys())

    def expand(
        self,
        slot_values: Mapping[str, tuple[str, ...]],
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        child_expansions = self.child.expand(slot_values, rules, seen, limit)
        unique_expansions: dict[str, None] = {"": None}
        for val in child_expansions:
            if val not in unique_expansions:
                unique_expansions[val] = None
                if len(unique_expansions) >= limit:
                    break
        return tuple(unique_expansions.keys())


class _HassilAlternativeNode(_HassilNode):
    """AST node representing an alternative choice group."""

    def __init__(self, branches: list[_HassilNode]):
        """Initialize AlternativeNode."""
        self.branches = branches

    def required_slots(self, rules: Mapping[str, str], seen: frozenset[str]) -> set[str]:
        """Return slot names required by this node."""
        if not self.branches:
            return set()
        req_sets = [b.required_slots(rules, seen) for b in self.branches]
        return set.intersection(*req_sets)

    def literal_variants(
        self,
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Return unique literal word variants for this node."""
        unique_variants: dict[str, None] = {}
        for branch in self.branches:
            remaining = limit - len(unique_variants)
            if remaining < 1:
                break
            for var in branch.literal_variants(rules, seen, remaining):
                if var not in unique_variants:
                    unique_variants[var] = None
                    if len(unique_variants) >= limit:
                        break
        return tuple(unique_variants.keys())

    def expand(
        self,
        slot_values: Mapping[str, tuple[str, ...]],
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        unique_expansions: dict[str, None] = {}
        for branch in self.branches:
            remaining = limit - len(unique_expansions)
            if remaining < 1:
                break
            for val in branch.expand(slot_values, rules, seen, remaining):
                if val not in unique_expansions:
                    unique_expansions[val] = None
                    if len(unique_expansions) >= limit:
                        break
        return tuple(unique_expansions.keys())


class _HassilSequenceNode(_HassilNode):
    """AST node representing a sequence of template nodes."""

    def __init__(self, children: list[_HassilNode]):
        """Initialize SequenceNode."""
        self.children = children

    def required_slots(self, rules: Mapping[str, str], seen: frozenset[str]) -> set[str]:
        """Return slot names required by this node."""
        req = set()
        for child in self.children:
            req.update(child.required_slots(rules, seen))
        return req

    def _accumulate(
        self,
        child_fragments: list[tuple[str, ...]],
        limit: int,
    ) -> tuple[str, ...]:
        """Cartesian-product accumulation of child fragments with deduplication."""
        if not child_fragments:
            return ("",)
        current = ("",)
        for fragments in child_fragments:
            next_results: dict[str, None] = {}
            for prefix in current:
                for fragment in fragments:
                    combined = f"{prefix}{fragment}"
                    if combined not in next_results:
                        next_results[combined] = None
                        if len(next_results) >= limit:
                            break
                if len(next_results) >= limit:
                    break
            current = tuple(next_results.keys())
            if not current:
                break
        return current

    def literal_variants(
        self,
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Return unique literal word variants for this node."""
        if not self.children:
            return ("",)
        child_fragments = [child.literal_variants(rules, seen, limit) for child in self.children]
        return self._accumulate(child_fragments, limit)

    def expand(
        self,
        slot_values: Mapping[str, tuple[str, ...]],
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        if not self.children:
            return ("",)
        child_fragments = [child.expand(slot_values, rules, seen, limit) for child in self.children]
        return self._accumulate(child_fragments, limit)


_PARSED_TEMPLATE_CACHE: dict[str, _HassilNode] = {}


def _parse_hassil(text: str) -> _HassilNode:
    """Parse Hassil template string into an AST node, caching the results."""
    cached = _PARSED_TEMPLATE_CACHE.get(text)
    if cached is not None:
        return cached

    def parse_expr(i: int) -> tuple[_HassilNode, int]:
        """Parse a Hassil expression from a given index."""
        current_branch: list[_HassilNode] = []
        branches: list[_HassilNode] = []

        while i < len(text):
            char = text[i]
            if char == "[":
                child, i = parse_expr(i + 1)
                current_branch.append(_HassilOptionalNode(child))
            elif char == "]":
                i += 1
                break
            elif char == "(":
                child, i = parse_expr(i + 1)
                current_branch.append(child)
            elif char == ")":
                i += 1
                break
            elif char == "|":
                branches.append(
                    _HassilSequenceNode(list(current_branch))
                    if len(current_branch) != 1
                    else current_branch[0]
                )
                current_branch = []
                i += 1
            elif char == "{":
                end = text.find("}", i)
                if end == -1:
                    current_branch.append(_HassilTextNode(text[i]))
                    i += 1
                else:
                    slot_name = text[i + 1 : end].strip()
                    current_branch.append(_HassilSlotNode(slot_name))
                    i = end + 1
            elif char == "<":
                end = text.find(">", i)
                if end == -1:
                    current_branch.append(_HassilTextNode(text[i]))
                    i += 1
                else:
                    rule_name = text[i + 1 : end].strip()
                    current_branch.append(_HassilRuleNode(rule_name))
                    i = end + 1
            else:
                current_branch.append(_HassilTextNode(char))
                i += 1

        branches.append(
            _HassilSequenceNode(list(current_branch))
            if len(current_branch) != 1
            else current_branch[0]
        )

        if len(branches) == 1:
            return branches[0], i
        return _HassilAlternativeNode(branches), i

    node, _ = parse_expr(0)
    _PARSED_TEMPLATE_CACHE[text] = node
    return node


def _required_slots(
    text: str,
    expansion_rules: Mapping[str, str],
    seen_rules: frozenset[str] = frozenset(),
) -> set[str]:
    """Return all slot names that are required in the template."""
    top_node = _parse_hassil(text)
    return top_node.required_slots(expansion_rules, seen_rules)


def _template_slot_names(
    text: str,
    expansion_rules: Mapping[str, str],
    seen_rules: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Return slot names referenced by a sentence template and its rules."""
    slots = {match.group(1).split(":")[0].strip() for match in _SLOT_PATTERN.finditer(text)}
    for rule_match in _RULE_PATTERN.finditer(text):
        rule_name = rule_match.group(1).strip()
        if rule_name in seen_rules:
            continue
        rule_text = expansion_rules.get(rule_name)
        if rule_text is not None:
            slots.update(
                _template_slot_names(
                    rule_text,
                    expansion_rules,
                    seen_rules | frozenset({rule_name}),
                )
            )
    return frozenset(slots)


def _context_domains(data_item: Mapping[str, Any]) -> tuple[str, ...]:
    """Return required entity domains for one Hassil data item."""
    domains: list[str] = []
    for key in ("requires_context", "slots"):
        context = data_item.get(key, {})
        if not isinstance(context, Mapping):
            continue
        domains.extend(_domain_values(context.get("domain")))
    return tuple(_deduplicate_texts(domains, len(domains) or 1))


def _domain_values(value: Any) -> Iterable[str]:
    """Yield domain names from Hassil context values."""
    if isinstance(value, str) and value.strip():
        yield value.strip()
        return
    if isinstance(value, Iterable) and not isinstance(value, str):
        for item in value:
            if isinstance(item, str) and item.strip():
                yield item.strip()


def _scoped_slot_values(
    slot_name: str,
    domains: tuple[str, ...],
    registry_slot_values: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Return domain-scoped registry values for one entity slot."""
    values = (
        value
        for domain in domains
        for value in registry_slot_values.get(f"{slot_name}:{domain}", ())
    )
    return tuple(_deduplicate_texts(values, DEFAULT_MAX_CANDIDATES_PER_INTENT))


def _slot_values(
    source_config: Mapping[str, Any],
    intent_config: Mapping[str, Any],
    data_item: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Return available slot input values for template expansion."""
    values: dict[str, tuple[str, ...]] = {}
    for config in (source_config, intent_config, data_item):
        lists = config.get("lists", {})
        if isinstance(lists, Mapping):
            for list_name, list_config in lists.items():
                if isinstance(list_name, str):
                    extracted = tuple(_values_from_list_config(list_config, list_name))
                    if extracted:
                        values[list_name] = extracted
    slots = data_item.get("slots", {})
    if isinstance(slots, Mapping):
        for slot_name, slot_value in slots.items():
            if isinstance(slot_name, str) and slot_name not in values:
                extracted = tuple(_string_values(slot_value))
                if extracted:
                    values[slot_name] = extracted
    return values


def _expansion_rules(
    source_config: Mapping[str, Any],
    intent_config: Mapping[str, Any],
    data_item: Mapping[str, Any],
) -> dict[str, str]:
    """Return expansion rule templates by name."""
    rules: dict[str, str] = {}
    for config in (source_config, intent_config, data_item):
        expansion_rules = config.get("expansion_rules", {})
        if not isinstance(expansion_rules, Mapping):
            continue
        for rule_name, rule_value in expansion_rules.items():
            if isinstance(rule_name, str) and isinstance(rule_value, str):
                rules[rule_name] = rule_value
    return rules


def _values_from_list_config(list_config: Any, list_name: str | None = None) -> Iterable[str]:
    """Yield spoken values from a Hassil list config."""
    if isinstance(list_config, Mapping):
        if "values" in list_config:
            values = list_config.get("values", [])
            if isinstance(values, list):
                for value in values:
                    yield from _value_item_inputs(value)
            return

        if "range" in list_config:
            range_data = list_config["range"]
            try:
                from_val = range_data.get("from", 0)
                to_val = range_data.get("to", 100)
                yield str(from_val)
                if from_val != to_val:
                    yield str(to_val)
            except (ValueError, TypeError):
                yield from ("1", "100")
            return

        if list_config.get("wildcard"):
            yield list_name if list_name else "wildcard"
            return

    if isinstance(list_config, list):
        for value in list_config:
            yield from _value_item_inputs(value)


def _value_item_inputs(value: Any) -> Iterable[str]:
    """Yield spoken inputs from one list value item."""
    if isinstance(value, str):
        yield value
        return
    if not isinstance(value, Mapping):
        return
    input_value = value.get("in")
    yield from _string_values(input_value)


def _string_values(value: Any) -> Iterable[str]:
    """Yield non-empty string values from strings or lists."""
    if isinstance(value, str) and value.strip():
        yield value
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                yield item


def _deduplicate_texts(texts: Iterable[str], limit: int) -> Iterable[str]:
    """Yield unique expanded texts up to the provided limit."""
    seen: set[str] = set()
    for text in texts:
        cleaned = _clean_expanded_text(text)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        yield cleaned
        if len(seen) >= limit:
            return


def _clean_expanded_text(text: str) -> str:
    """Normalize whitespace introduced by template expansion."""
    return " ".join(text.split())


def _candidate_source_from_key(source_key: str) -> CandidateSource:
    """Map a Home Assistant intent source key to a candidate source."""
    if source_key.lower() in {"config", "trigger", "custom_sentence"}:
        return CandidateSource.CUSTOM_SENTENCE
    return CandidateSource.BUILT_IN
