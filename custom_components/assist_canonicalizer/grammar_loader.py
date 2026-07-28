"""Candidate loading from Home Assistant conversation intent sources."""

from __future__ import annotations

import contextlib
import logging
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field, replace
from functools import lru_cache
from heapq import nlargest
from itertools import permutations, product
from threading import Lock
from typing import Any, NamedTuple

import orjson

from .candidate import (
    Candidate,
    CandidateSource,
    candidate_dedupe_preference_key,
    candidate_source_priority,
)
from .const import (
    DEFAULT_MAX_CANDIDATES_PER_INTENT,
    DEFAULT_MAX_CANDIDATES_PER_TEMPLATE,
    DEFAULT_MAX_DYNAMIC_CANDIDATES,
    DEFAULT_MAX_DYNAMIC_SLOT_VALUES,
    DEFAULT_MAX_REGISTRY_VALUES_NOMINATED,
    DEFAULT_MAX_REGISTRY_VALUES_SCORED_PER_QUERY,
    DEFAULT_MAX_TOTAL_CANDIDATES_PER_LANGUAGE,
    ENTITY_SLOT_NAME_SET,
    LOCATION_SLOT_NAME_SET,
    LOCATION_SLOT_NAMES,
    REGISTRY_FUZZY_TOKEN_MIN_LENGTH,
    SLOT_FUZZY_TOKEN_MIN_LENGTH,
)
from .normalization import normalize_text, normalize_text_no_diacritics
from .registry import merge_slot_values
from .rehydration import rehydrate_wildcard_slots as rehydrate_wildcard_slots
from .rehydration import rehydrate_wildcard_text as rehydrate_wildcard_text
from .utils import (
    is_valid_range_value,
    parse_float,
    register_custom_wildcards_from_sources,
    to_output_value,
)

_LOGGER = logging.getLogger(__name__)

_TEMPLATE_MARKERS = frozenset("{}[]<>|();")
_CLEAN_ANCHOR_STRIP_CHARS = "[](){}<>|:,.?!;\"'"
_SLOT_PATTERN = re.compile(r"{([^{}]+)}")
_RULE_PATTERN = re.compile(r"<([^<>]+)>")
_COMPACT_SLOT_TOKEN_MIN_LENGTH = 4
_COMPACT_SCRIPT_QUERY_SPAN_MIN_LENGTH = 2
_COMPACT_SCRIPT_QUERY_SPAN_MAX_LENGTH = 16
_MAX_COMPOUND_QUERY_TOKENS = 4
_SLOT_ANCHOR_PATTERN_LIMIT = 64
_SLOT_QUERY_WINDOW_LIMIT = 8
_SLOT_ANCHOR_MARKER = "assist_canonicalizer_slot_marker"
_OTHER_SLOT_ANCHOR_MARKER = "assist_canonicalizer_other_slot_marker"
_COMPACT_SCRIPT_NAME_PREFIXES = (
    "BOPOMOFO",
    "CJK",
    "HANGUL",
    "HIRAGANA",
    "IDEOGRAPHIC",
    "KATAKANA",
    "KHMER",
    "LAO",
    "MYANMAR",
    "THAI",
)
_TEMPLATE_RELEVANCE_EXPANSION_LIMIT = 200
_TEXT_RUN_STOP_CHARS = frozenset("[]()|{<;")
_CONTEXT_ANCHOR_WINDOW = 3
_MAX_TEMPLATE_NESTING_DEPTH = 25
_EXPANSION_MARKER_CODEPOINTS = range(0xE000, 0xF900)


@lru_cache(maxsize=256)
def _warn_unknown_expansion_rule(name: str) -> None:
    """Log one warning per unknown or self-recursive expansion rule name."""
    _LOGGER.warning(
        "Expansion rule <%s> is undefined or self-recursive; expanding it as empty text",
        name,
    )


@dataclass(frozen=True, slots=True)
class RegistrySlotValue:
    """Precomputed lexical data for one registry slot value."""

    text: str
    normalized_text: str
    tokens: tuple[str, ...]
    position: int
    normalized_no_diacritics: str
    tokens_no_diacritics: tuple[str, ...]


@dataclass(slots=True)
class RegistryRetrievalStats:
    """Bounded counters from one query-time registry retrieval pass."""

    record_count: int = 0
    postings_consulted: int = 0
    values_nominated: int = 0
    values_scored: int = 0
    fuzzy_dynamic_candidates: int = 0
    anchored_dynamic_candidates: int = 0
    fuzzy_values: set[str] = field(default_factory=set, repr=False)


@dataclass(frozen=True, slots=True)
class TemplateSlotReference:
    """Slot reference parsed from HassIL template syntax."""

    list_name: str
    output_name: str
    raw_name: str


@dataclass(frozen=True, slots=True)
class SlotAnchorPattern:
    """Normalized literal boundaries surrounding one unresolved registry slot."""

    prefix: str
    suffix: str
    prefix_no_diacritics: str
    suffix_no_diacritics: str


@dataclass(frozen=True, slots=True)
class SlotQueryWindow:
    """One query span isolated by a compiled slot-anchor pattern."""

    normalized_text: str
    normalized_no_diacritics: str | None
    anchor_length: int


type RegistryRelevanceCache = dict[
    tuple[int, str, str | None, bool],
    tuple[str, ...],
]


@dataclass(frozen=True, slots=True)
class _SlotBinding:
    """One exact slot value selected while expanding a template."""

    list_name: str
    output_name: str
    raw_value: str


@dataclass(frozen=True, slots=True)
class _ExpansionTagContext:
    """Collision-free marker and decoders for one template expansion."""

    marker: str
    binding_pattern: re.Pattern[str]
    wildcard_pattern: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class _CandidateExpansion:
    """Rendered candidate text and the slot bindings that produced it."""

    text: str
    slot_bindings: tuple[_SlotBinding, ...] = ()
    wildcard_infos: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _DataItemCandidateState:
    """Shared state for candidates generated from one intent data item."""

    language: str
    source_key: str
    source: CandidateSource
    intent_name: str
    expansion_rules: Mapping[str, str]
    base_slot_values: Mapping[str, tuple[str, ...]]
    output_value_maps: Mapping[str, Mapping[str, Any]]
    wildcard_slots: frozenset[str]
    context_slots: frozenset[str]
    static_slots: dict[str, str]


@dataclass(frozen=True, slots=True)
class _PreparedCandidateSentence:
    """Resolved expansion inputs and metadata for one sentence template."""

    sentence: str
    slot_references: tuple[TemplateSlotReference, ...]
    slot_values: Mapping[str, tuple[str, ...]]
    wildcard_slots: frozenset[str]
    metadata: Mapping[str, str]
    literal_variants: tuple[frozenset[str], ...]


@dataclass(frozen=True, slots=True)
class _TemplateCompilationState:
    """Per-data-item state shared while compiling registry sentence templates."""

    source_key: str
    expansion_rules: Mapping[str, str]
    base_data_slot_values: Mapping[str, tuple[str, ...]]
    slot_output_value_maps: Mapping[str, Mapping[str, Any]]
    wildcard_slots: frozenset[str]
    static_slots: Mapping[str, str]
    context_slots: frozenset[str]
    domains: tuple[str, ...]
    language: str | None
    range_slots: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    range_step_multipliers: Mapping[str, RangeStepMultiplier] = field(default_factory=dict)


class RegistrySlotIndex(dict[str, tuple[RegistrySlotValue, ...]]):
    """Precomputed registry values index with an inverted index helper."""

    def __init__(self, data: dict[str, tuple[RegistrySlotValue, ...]]):
        """Initialize and create the inverted cache store."""
        super().__init__(data)
        self._index_record_ids = {id(records) for records in data.values()}
        self._scoped_records_cache: dict[
            tuple[str, tuple[str, ...]], tuple[RegistrySlotValue, ...]
        ] = {}
        self._scoped_records_lock = Lock()
        self._inverted_cache: dict[
            int,
            tuple[tuple[RegistrySlotValue, ...], dict[str, list[RegistrySlotValue]]],
        ] = {}
        self._deletion_cache: dict[
            int,
            tuple[tuple[RegistrySlotValue, ...], dict[str, tuple[str, ...]]],
        ] = {}
        self.record_count = sum(
            len(records) for records in {id(records): records for records in data.values()}.values()
        )
        for records in {id(records): records for records in data.values()}.values():
            lookup = self.get_inverted_for_records(records)
            self._tokens_by_deletion_for_records(records, lookup)

    def get_scoped_records(
        self,
        slot_name: str,
        domains: tuple[str, ...],
    ) -> tuple[RegistrySlotValue, ...]:
        """Return a reusable complete record tuple for one domain combination."""
        if not domains:
            return self.get(slot_name, ())
        if len(domains) == 1:
            return self.get(f"{slot_name}:{domains[0]}", ())

        cache_key = (slot_name, domains)
        cached = self._scoped_records_cache.get(cache_key)
        if cached is not None:
            return cached
        with self._scoped_records_lock:
            return self._get_or_create_scoped_records_locked(cache_key, slot_name, domains)

    def _get_or_create_scoped_records_locked(
        self,
        cache_key: tuple[str, tuple[str, ...]],
        slot_name: str,
        domains: tuple[str, ...],
    ) -> tuple[RegistrySlotValue, ...]:
        """Return cached merged records while the scoped-record lock is held."""
        cached = self._scoped_records_cache.get(cache_key)
        if cached is not None:
            return cached
        selected: list[RegistrySlotValue] = []
        seen: set[str] = set()
        for domain in domains:
            for record in self.get(f"{slot_name}:{domain}", ()):
                if record.text in seen:
                    continue
                seen.add(record.text)
                selected.append(record)
        records = tuple(selected)
        self._scoped_records_cache[cache_key] = records
        self._index_record_ids.add(id(records))
        return records

    def get_inverted_for_records(
        self, records: tuple[RegistrySlotValue, ...]
    ) -> dict[str, list[RegistrySlotValue]]:
        """Get or build the inverted index for the given records tuple."""
        record_id = id(records)
        cached = self._inverted_cache.get(record_id)
        if cached is not None:
            cached_records, lookup = cached
            if cached_records is records:
                return lookup

        lookup = {}
        for record in records:
            seen_tokens: set[str] = set()
            for token in record.tokens:
                seen_tokens.add(token)
                lookup.setdefault(token, []).append(record)
            for token in record.tokens_no_diacritics:
                if token not in seen_tokens:
                    seen_tokens.add(token)
                    lookup.setdefault(token, []).append(record)
            for value in (record.normalized_text, record.normalized_no_diacritics):
                compact = _compact_slot_text(value)
                if compact and compact not in seen_tokens:
                    seen_tokens.add(compact)
                    lookup.setdefault(compact, []).append(record)
        if record_id in self._index_record_ids:
            self._inverted_cache[record_id] = (records, lookup)
        return lookup

    def get_fuzzy_for_records(
        self,
        records: tuple[RegistrySlotValue, ...],
        query_tokens: Iterable[str],
        stats: RegistryRetrievalStats | None = None,
    ) -> set[RegistrySlotValue]:
        """Return records reached through bounded one-edit token lookup."""
        lookup = self.get_inverted_for_records(records)
        fuzzy_query_tokens = tuple(
            token
            for token in query_tokens
            if token not in lookup and len(token) >= SLOT_FUZZY_TOKEN_MIN_LENGTH
        )
        if not fuzzy_query_tokens:
            return set()

        tokens_by_deletion = self._tokens_by_deletion_for_records(records, lookup)
        matched_records: set[RegistrySlotValue] = set()
        for query_token in fuzzy_query_tokens:
            possible_tokens = _fuzzy_registry_value_tokens(
                query_token,
                lookup,
                tokens_by_deletion,
                stats,
            )
            for value_token in possible_tokens:
                if _is_bounded_registry_token_match(value_token, query_token):
                    matched_records.update(lookup[value_token])
        if stats is not None:
            stats.fuzzy_values.update(record.normalized_text for record in matched_records)
        return matched_records

    def _tokens_by_deletion_for_records(
        self,
        records: tuple[RegistrySlotValue, ...],
        lookup: Mapping[str, Sequence[RegistrySlotValue]],
    ) -> dict[str, tuple[str, ...]]:
        """Return the cached deletion-neighborhood tokens for selected records."""
        record_id = id(records)
        cached = self._deletion_cache.get(record_id)
        if cached is not None and cached[0] is records:
            return cached[1]
        deletions: dict[str, list[str]] = {}
        for value_token in lookup:
            if len(value_token) < SLOT_FUZZY_TOKEN_MIN_LENGTH:
                continue
            for deletion in _single_character_deletions(value_token):
                deletions.setdefault(deletion, []).append(value_token)
        tokens_by_deletion = {deletion: tuple(tokens) for deletion, tokens in deletions.items()}
        if record_id in self._index_record_ids:
            self._deletion_cache[record_id] = (records, tokens_by_deletion)
        return tokens_by_deletion


def _fuzzy_registry_value_tokens(
    query_token: str,
    lookup: Mapping[str, Sequence[RegistrySlotValue]],
    tokens_by_deletion: Mapping[str, Sequence[str]],
    stats: RegistryRetrievalStats | None,
) -> set[str]:
    """Return value tokens reached through one-edit deletion neighborhoods."""
    if stats is not None:
        stats.postings_consulted += 1
    possible_tokens = set(tokens_by_deletion.get(query_token, ()))
    for deletion in _single_character_deletions(query_token):
        if stats is not None:
            stats.postings_consulted += 2
        if deletion in lookup:
            possible_tokens.add(deletion)
        possible_tokens.update(tokens_by_deletion.get(deletion, ()))
    return possible_tokens


@dataclass(frozen=True, slots=True)
class RangeStepMultiplier:
    """Step, multiplier, fraction type, and original boundaries for a range slot."""

    step: float | None
    multiplier: float | None
    fraction_type: str | None
    original_from: float | None


@dataclass(frozen=True, slots=True)
class DynamicRegistryTemplate:
    """Query-independent data needed to expand one registry template."""

    sentence: str
    slot_references: tuple[TemplateSlotReference, ...]
    sentence_slots: frozenset[str]
    wildcard_slots: frozenset[str]
    slot_output_values: Mapping[str, Mapping[str, Any]]
    entity_slots: tuple[str, ...]
    query_slots: tuple[str, ...]
    domains: tuple[str, ...]
    expansion_rules: Mapping[str, str]
    base_data_slot_values: Mapping[str, tuple[str, ...]]
    static_slots: Mapping[str, str]
    required_slots: frozenset[str]
    metadata: Mapping[str, str]
    literal_token_variants: tuple[frozenset[str], ...]
    literal_token_variants_no_diac: tuple[frozenset[str], ...]
    range_slots: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    range_step_multipliers: Mapping[str, RangeStepMultiplier] = field(default_factory=dict)
    range_slot_anchors: Mapping[str, list[tuple[str | None, str | None]]] = field(
        default_factory=dict
    )
    slot_anchor_patterns: Mapping[str, tuple[SlotAnchorPattern, ...]] = field(default_factory=dict)


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
    register_custom_wildcards_from_sources(language, intent_sources)

    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    candidates: list[Candidate] = []
    ordered_sources = sorted(
        intent_sources.items(),
        # Tie-break same-priority sources by key: dict insertion order varies
        # with subscription arrival across restarts, and when per-intent caps
        # bind, the surviving candidate set must not depend on it (the store
        # fingerprint sorts mappings, so it cannot detect the difference).
        key=lambda item: (
            candidate_source_priority(_candidate_source_from_key(item[0])),
            item[0],
        ),
    )
    for source_key, source_config in ordered_sources:
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


def _slot_anchor_expansion_values(
    state: _TemplateCompilationState,
    slot_references: tuple[TemplateSlotReference, ...],
    target_slot: str,
) -> dict[str, tuple[str, ...]]:
    """Return bounded template values with one target marker and no unknown spans."""
    values = dict(state.base_data_slot_values)
    for reference in slot_references:
        if reference.list_name == target_slot:
            marker_values = (_SLOT_ANCHOR_MARKER,)
            values[reference.raw_name] = marker_values
            values[reference.list_name] = marker_values
            continue
        if reference.raw_name in values or reference.list_name in values:
            continue
        static_value = state.static_slots.get(reference.output_name)
        if static_value is None:
            static_value = state.static_slots.get(reference.list_name)
        marker_values = (
            (str(static_value),) if static_value is not None else (_OTHER_SLOT_ANCHOR_MARKER,)
        )
        values[reference.raw_name] = marker_values
        values.setdefault(reference.list_name, marker_values)
    return values


def _split_slot_anchor_pattern(
    expanded_text: str,
    language: str | None,
) -> SlotAnchorPattern | None:
    """Return literal boundaries when one expansion contains exactly one target marker."""
    normalized = normalize_text(expanded_text)
    if normalized.count(_SLOT_ANCHOR_MARKER) != 1 or _OTHER_SLOT_ANCHOR_MARKER in normalized:
        return None
    prefix, _marker, suffix = normalized.partition(_SLOT_ANCHOR_MARKER)

    normalized_no_diacritics = _cached_normalize_no_diac(expanded_text, language)
    if normalized_no_diacritics.count(_SLOT_ANCHOR_MARKER) != 1:
        return None
    prefix_no_diacritics, _marker, suffix_no_diacritics = normalized_no_diacritics.partition(
        _SLOT_ANCHOR_MARKER
    )
    return SlotAnchorPattern(
        prefix=prefix,
        suffix=suffix,
        prefix_no_diacritics=prefix_no_diacritics,
        suffix_no_diacritics=suffix_no_diacritics,
    )


def _compile_slot_anchor_patterns(
    state: _TemplateCompilationState,
    sentence: str,
    slot_references: tuple[TemplateSlotReference, ...],
    query_slots: tuple[str, ...],
) -> Mapping[str, tuple[SlotAnchorPattern, ...]]:
    """Compile reliable literal boundaries for a single unresolved registry slot.

    Templates with another unresolved span are deliberately excluded: without a
    literal separator, assigning either span would recreate the ambiguity this path
    is intended to avoid. Spaces present in the HassIL template remain part of the
    boundary, so compact matching never silently removes a required word boundary.
    """
    if len(query_slots) != 1:
        return {}
    target_slot = query_slots[0]
    values = _slot_anchor_expansion_values(state, slot_references, target_slot)
    expansions = _parse_hassil(sentence).expand(
        values,
        state.expansion_rules,
        frozenset(),
        _SLOT_ANCHOR_PATTERN_LIMIT,
        fair=True,
    )
    patterns: dict[SlotAnchorPattern, None] = {}
    for expanded_text in expansions:
        pattern = _split_slot_anchor_pattern(expanded_text, state.language)
        if pattern is None:
            continue
        patterns[pattern] = None
        if len(patterns) >= _SLOT_ANCHOR_PATTERN_LIMIT:
            break
    return {target_slot: tuple(patterns)} if patterns else {}


def _template_sentence_excluded(
    state: _TemplateCompilationState,
    sentence: str,
    sentence_slots: frozenset[str],
    entity_slots: tuple[str, ...],
    query_slots: tuple[str, ...],
    variants: tuple[frozenset[str], ...],
    *,
    include_literal_only_templates: bool,
    include_area_only_templates: bool,
) -> bool:
    """Return whether a sentence template should be skipped during compilation."""
    if (
        not include_area_only_templates
        and query_slots
        and not entity_slots
        and not _is_domain_scoped_location_template(sentence_slots, state.domains)
    ):
        return True
    if not include_literal_only_templates and not query_slots:
        return True
    return not query_slots and (not variants or is_fixed_sentence(sentence))


def _template_base_metadata(
    state: _TemplateCompilationState,
    sentence: str,
    literal_text: str,
    variants: tuple[frozenset[str], ...],
    query_slots: tuple[str, ...],
    sentence_wildcards: frozenset[str],
) -> dict[str, str]:
    """Build base candidate metadata for one compiled sentence template."""
    base_metadata = _candidate_metadata(
        state.source_key,
        sentence,
        state.expansion_rules,
        literal_text=literal_text,
        literal_variants=variants,
    )
    if state.static_slots:
        base_metadata["static_slots"] = ",".join(sorted(state.static_slots.keys()))
    if state.context_slots:
        base_metadata["context_slots"] = ",".join(sorted(state.context_slots))
    if query_slots:
        base_metadata["query_slots"] = ",".join(query_slots)
    if sentence_wildcards:
        base_metadata["wildcard_slots"] = ",".join(sorted(sentence_wildcards))
    return base_metadata


class _TemplateSlotStructure(NamedTuple):
    """Resolved and unresolved slot names for one compiled sentence."""

    references: tuple[TemplateSlotReference, ...]
    sentence_slots: frozenset[str]
    entity_slots: tuple[str, ...]
    query_slots: tuple[str, ...]


def _template_slot_structure(
    state: _TemplateCompilationState,
    sentence: str,
) -> _TemplateSlotStructure:
    """Classify the slot references in one sentence template."""
    references = _template_slot_references(sentence, state.expansion_rules)
    sentence_slots = frozenset(reference.list_name for reference in references)
    resolved_slots = set(state.base_data_slot_values) | set(state.static_slots)
    unresolved_slots = sentence_slots - resolved_slots
    return _TemplateSlotStructure(
        references=references,
        sentence_slots=sentence_slots,
        entity_slots=tuple(sorted(unresolved_slots & ENTITY_SLOT_NAME_SET)),
        query_slots=tuple(
            sorted(unresolved_slots & (ENTITY_SLOT_NAME_SET | LOCATION_SLOT_NAME_SET))
        ),
    )


def _template_range_metadata(
    state: _TemplateCompilationState,
    sentence: str,
    sentence_slots: frozenset[str],
) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, RangeStepMultiplier],
    dict[str, list[tuple[str | None, str | None]]],
]:
    """Return range data restricted to slots referenced by one sentence."""
    range_slots = {
        slot_name: state.range_slots[slot_name]
        for slot_name in sentence_slots
        if slot_name in state.range_slots
    }
    return (
        range_slots,
        {
            slot_name: state.range_step_multipliers[slot_name]
            for slot_name in sentence_slots
            if slot_name in state.range_step_multipliers
        },
        {slot_name: _find_slot_context_anchors(sentence, slot_name) for slot_name in range_slots},
    )


def _compile_template_from_sentence(
    state: _TemplateCompilationState,
    sentence: str,
    *,
    include_literal_only_templates: bool,
    include_area_only_templates: bool,
) -> DynamicRegistryTemplate | None:
    """Compile a single sentence into a DynamicRegistryTemplate, or None to skip."""
    slot_structure = _template_slot_structure(state, sentence)
    literal_text, variants = _template_literals(sentence, state.expansion_rules)
    if _template_sentence_excluded(
        state,
        sentence,
        slot_structure.sentence_slots,
        slot_structure.entity_slots,
        slot_structure.query_slots,
        variants,
        include_literal_only_templates=include_literal_only_templates,
        include_area_only_templates=include_area_only_templates,
    ):
        return None
    no_diac_variants = tuple(
        frozenset(_cached_normalize_no_diac(token, state.language) for token in tokens)
        for tokens in variants
    )
    sentence_wildcards = slot_structure.sentence_slots & state.wildcard_slots
    base_metadata = _template_base_metadata(
        state,
        sentence,
        literal_text,
        variants,
        slot_structure.query_slots,
        sentence_wildcards,
    )
    range_slots, range_step_multipliers, range_slot_anchors = _template_range_metadata(
        state,
        sentence,
        slot_structure.sentence_slots,
    )
    return DynamicRegistryTemplate(
        sentence=sentence,
        slot_references=slot_structure.references,
        sentence_slots=slot_structure.sentence_slots,
        wildcard_slots=sentence_wildcards,
        slot_output_values={
            slot_name: state.slot_output_value_maps[slot_name]
            for slot_name in slot_structure.sentence_slots
            if slot_name in state.slot_output_value_maps
        },
        entity_slots=slot_structure.entity_slots,
        query_slots=slot_structure.query_slots,
        domains=state.domains,
        expansion_rules=state.expansion_rules,
        base_data_slot_values=state.base_data_slot_values,
        static_slots=state.static_slots,
        required_slots=_required_slots(sentence, state.expansion_rules),
        metadata=base_metadata,
        literal_token_variants=variants,
        literal_token_variants_no_diac=no_diac_variants,
        range_slots=range_slots,
        range_step_multipliers=range_step_multipliers,
        range_slot_anchors=range_slot_anchors,
        slot_anchor_patterns=_compile_slot_anchor_patterns(
            state,
            sentence,
            slot_structure.references,
            slot_structure.query_slots,
        ),
    )


def _compile_templates_from_data_item(
    source_key: str,
    source_config: Mapping[str, Any],
    intent_config: Mapping[str, Any],
    data_item: Mapping[str, Any],
    language: str | None,
    *,
    include_literal_only_templates: bool,
    include_area_only_templates: bool,
) -> list[DynamicRegistryTemplate]:
    """Compile all valid sentence templates from a single data item."""
    sentences = data_item.get("sentences", [])
    if not isinstance(sentences, list):
        return []
    expansion_rules = _expansion_rules(source_config, intent_config, data_item)
    base_data_slot_values = _slot_values(source_config, intent_config, data_item)
    range_slots, range_step_multipliers = _compile_range_data(
        source_config, intent_config, data_item
    )
    output_value_maps = _slot_output_value_maps(source_config, intent_config, data_item)
    wildcard_slots = _wildcard_slots(source_config, intent_config, data_item)
    static_slots = _static_slot_values(data_item)
    context_slots = _context_slot_names(data_item)
    domains = _context_domains(data_item)
    state = _TemplateCompilationState(
        source_key=source_key,
        expansion_rules=expansion_rules,
        base_data_slot_values=base_data_slot_values,
        slot_output_value_maps=output_value_maps,
        wildcard_slots=wildcard_slots,
        static_slots=static_slots,
        context_slots=context_slots,
        domains=domains,
        language=language,
        range_slots=range_slots,
        range_step_multipliers=range_step_multipliers,
    )
    templates = []
    for sentence in sentences:
        if not isinstance(sentence, str):
            continue
        template = _compile_template_from_sentence(
            state,
            sentence,
            include_literal_only_templates=include_literal_only_templates,
            include_area_only_templates=include_area_only_templates,
        )
        if template is not None:
            templates.append(template)
    return templates


def compile_dynamic_registry_intents(
    intent_sources: Mapping[str, Mapping[str, Any]],
    language: str | None = None,
    *,
    include_literal_only_templates: bool = True,
    include_area_only_templates: bool = True,
) -> tuple[DynamicRegistryIntent, ...]:
    """Compile query-independent dynamic registry template data."""
    register_custom_wildcards_from_sources(language, intent_sources)

    compiled: list[DynamicRegistryIntent] = []
    for source_key, source_config in intent_sources.items():
        source = _candidate_source_from_key(source_key)
        intents = source_config.get("intents", {})
        if not isinstance(intents, Mapping):
            continue
        for intent_name, intent_config in intents.items():
            if not isinstance(intent_name, str) or not isinstance(intent_config, Mapping):
                continue
            data_items = intent_config.get("data", [])
            if not isinstance(data_items, list):
                continue
            templates: list[DynamicRegistryTemplate] = []
            for data_item in data_items:
                if not isinstance(data_item, Mapping):
                    continue
                templates.extend(
                    _compile_templates_from_data_item(
                        source_key,
                        source_config,
                        intent_config,
                        data_item,
                        language,
                        include_literal_only_templates=include_literal_only_templates,
                        include_area_only_templates=include_area_only_templates,
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
    include_literal_only_templates: bool = True,
    include_area_only_templates: bool = True,
    literal_only_wildcards_only: bool = False,
    retrieval_stats: RegistryRetrievalStats | None = None,
) -> tuple[Candidate, ...]:
    """Build query-scoped registry candidates without expanding every entity."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    query_normalized = normalize_text(query)
    if not query_normalized or (not registry_slot_values and not include_literal_only_templates):
        return ()
    query_tokens = frozenset(query_normalized.split())
    if registry_slot_index is None:
        registry_slot_index = build_registry_slot_index(registry_slot_values, language)
    if retrieval_stats is not None:
        retrieval_stats.record_count = registry_slot_index.record_count
    if compiled_intents is None:
        compiled_intents = compile_dynamic_registry_intents(
            intent_sources,
            language,
            include_literal_only_templates=include_literal_only_templates,
            include_area_only_templates=include_area_only_templates,
        )

    query_no_diac = normalize_text_no_diacritics(query, language)
    query_tokens_no_diac = frozenset(query_no_diac.split())
    query_numeric_tokens = _query_numeric_tokens(query_normalized)
    selected = _collect_query_registry_candidates(
        language,
        compiled_intents,
        registry_slot_values,
        registry_slot_index,
        query_normalized,
        query_tokens,
        max_candidates=max_candidates,
        query_no_diac=query_no_diac,
        query_tokens_no_diac=query_tokens_no_diac,
        include_literal_only_templates=include_literal_only_templates,
        include_area_only_templates=include_area_only_templates,
        literal_only_wildcards_only=literal_only_wildcards_only,
        query_numeric_tokens=query_numeric_tokens,
        retrieval_stats=retrieval_stats,
    )
    if retrieval_stats is None:
        return selected
    return _tag_fuzzy_retrieval_candidates(selected, retrieval_stats)


def _query_numeric_tokens(query_normalized: str) -> list[tuple[str, float, int]]:
    """Precompute query numeric tokens once per query search run."""
    query_numeric_tokens: list[tuple[str, float, int]] = []
    for idx, token in enumerate(query_normalized.split()):
        val = parse_float(token)
        if val is not None:
            query_numeric_tokens.append((token, val, idx))
    return query_numeric_tokens


def _collect_query_registry_candidates(
    language: str,
    compiled_intents: Sequence[DynamicRegistryIntent],
    registry_slot_values: Mapping[str, tuple[str, ...]],
    registry_slot_index: RegistrySlotIndex,
    query_normalized: str,
    query_tokens: frozenset[str],
    *,
    max_candidates: int,
    query_no_diac: str,
    query_tokens_no_diac: frozenset[str],
    include_literal_only_templates: bool,
    include_area_only_templates: bool,
    literal_only_wildcards_only: bool,
    query_numeric_tokens: Sequence[tuple[str, float, int]],
    retrieval_stats: RegistryRetrievalStats | None,
) -> tuple[Candidate, ...]:
    """Collect and rank query-scoped candidates across all compiled intents."""
    candidates: list[Candidate] = []
    relevant_cache: RegistryRelevanceCache = {}
    scoped_cache: dict[tuple[str, tuple[str, ...]], tuple[RegistrySlotValue, ...]] = {}
    for compiled_intent in compiled_intents:
        if intent_candidates := _query_candidates_from_compiled_intent(
            language,
            compiled_intent,
            registry_slot_values,
            registry_slot_index,
            query_normalized,
            query_tokens,
            relevant_cache,
            scoped_cache,
            max_candidates=max_candidates,
            query_no_diac=query_no_diac,
            query_tokens_no_diac=query_tokens_no_diac,
            include_literal_only_templates=include_literal_only_templates,
            include_area_only_templates=include_area_only_templates,
            literal_only_wildcards_only=literal_only_wildcards_only,
            query_numeric_tokens=query_numeric_tokens,
            retrieval_stats=retrieval_stats,
        ):
            candidates = _merge_top_query_candidates(
                candidates,
                intent_candidates,
                max_candidates,
                query_normalized,
                query_tokens,
            )
    return tuple(
        _top_query_candidates(
            candidates,
            max_candidates,
            query_normalized,
            query_tokens,
        )
    )


def _merge_top_query_candidates(
    candidates: list[Candidate],
    additions: Sequence[Candidate],
    max_candidates: int,
    query_normalized: str,
    query_tokens: frozenset[str],
) -> list[Candidate]:
    """Append candidates and retain only the strongest bounded set."""
    candidates.extend(additions)
    if len(candidates) <= max_candidates:
        return candidates
    return _top_query_candidates(
        candidates,
        max_candidates,
        query_normalized,
        query_tokens,
    )


def _tag_fuzzy_retrieval_candidates(
    selected: tuple[Candidate, ...],
    retrieval_stats: RegistryRetrievalStats,
) -> tuple[Candidate, ...]:
    """Record retrieval stats and tag candidates built from fuzzy registry values."""
    retrieval_stats.anchored_dynamic_candidates = sum(
        candidate.metadata.get("registry_retrieval") == "anchored" for candidate in selected
    )
    if not retrieval_stats.fuzzy_values:
        return selected
    tagged = tuple(
        replace(
            candidate,
            metadata={**candidate.metadata, "registry_retrieval": "fuzzy"},
        )
        if _candidate_uses_fuzzy_registry_value(candidate, retrieval_stats.fuzzy_values)
        else candidate
        for candidate in selected
    )
    retrieval_stats.fuzzy_dynamic_candidates = sum(
        candidate.metadata.get("registry_retrieval") == "fuzzy" for candidate in tagged
    )
    return tagged


def _candidate_uses_fuzzy_registry_value(
    candidate: Candidate,
    normalized_fuzzy_values: set[str],
) -> bool:
    """Return whether candidate slot output contains a fuzzy-nominated value."""
    return any(
        normalize_text(str(value)) in normalized_fuzzy_values
        for value in candidate.parsed_slots.values()
        if value is not None
    )


def _query_candidate_relevance_key(
    candidate: Candidate,
    query_normalized: str,
    query_tokens: frozenset[str],
) -> tuple[int, int, float, int]:
    """Return a stable relevance key for capping query-scoped candidates."""
    return _text_relevance_key(candidate.normalized_text, query_normalized, query_tokens)


def _deduplicate_query_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Deduplicate query candidates by normalized text and intent."""
    selected: dict[tuple[str, str], Candidate] = {}
    order: list[tuple[str, str]] = []
    for candidate in candidates:
        key = (candidate.normalized_text, candidate.intent_name)
        existing = selected.get(key)
        if existing is None:
            order.append(key)
            selected[key] = candidate
        elif candidate_dedupe_preference_key(candidate) > candidate_dedupe_preference_key(existing):
            selected[key] = candidate
    return [selected[key] for key in order]


def _top_query_candidates(
    candidates: Iterable[Candidate],
    limit: int,
    query_normalized: str,
    query_tokens: frozenset[str],
) -> list[Candidate]:
    """Return the strongest query-scoped candidates up to limit."""
    deduplicated = _deduplicate_query_candidates(candidates)
    deduplicated.sort(
        key=lambda candidate: _query_candidate_relevance_key(
            candidate, query_normalized, query_tokens
        ),
        reverse=True,
    )
    return deduplicated[:limit]


@lru_cache(maxsize=8192)
def _text_relevance_key(
    candidate_normalized: str,
    query_normalized: str,
    query_tokens: frozenset[str],
) -> tuple[int, int, float, int]:
    """Return a stable relevance key for normalized candidate text.

    Cached because per-intent trimming re-sorts overlapping candidate lists
    several times per request, re-keying the same (candidate, query) pairs.
    """
    if not candidate_normalized:
        return (0, 0, 0.0, 0)
    candidate_tokens = frozenset(candidate_normalized.split())
    union_size = len(query_tokens | candidate_tokens)
    overlap_score = (len(query_tokens & candidate_tokens) / union_size) if union_size else 0.0
    return (
        int(candidate_normalized == query_normalized),
        int(query_normalized in candidate_normalized or candidate_normalized in query_normalized),
        overlap_score,
        -abs(len(candidate_normalized) - len(query_normalized)),
    )


def is_fixed_sentence(sentence: str) -> bool:
    """Return whether a sentence has no HassIL template syntax."""
    return bool(sentence.strip()) and all(marker not in sentence for marker in _TEMPLATE_MARKERS)


def expand_sentence_template(
    sentence: str,
    slot_values: Mapping[str, tuple[str, ...]],
    expansion_rules: Mapping[str, str],
    *,
    max_expansions: int = DEFAULT_MAX_CANDIDATES_PER_TEMPLATE,
    fair: bool = False,
    expansion_filter: Callable[[str], bool] | None = None,
) -> tuple[str, ...]:
    """Expand a bounded subset of HassIL sentence template syntax."""
    if max_expansions < 1:
        raise ValueError("max_expansions must be positive")
    top_node = _parse_hassil(sentence)
    expansions = top_node.expand(
        slot_values,
        expansion_rules,
        frozenset(),
        max_expansions,
        fair=fair,
        expansion_filter=expansion_filter,
    )
    return tuple(_deduplicate_texts(expansions, max_expansions))


def _clean_wildcard_tokens(
    expanded_sentence: str,
    tag_pattern: re.Pattern[str],
) -> tuple[str, list[tuple[int, str]]]:
    """Strip owned wildcard tags and return their normalized token positions."""
    clean_parts: list[str] = []
    wildcard_infos: list[tuple[int, str]] = []
    cursor = 0
    for match in tag_pattern.finditer(expanded_sentence):
        clean_parts.append(expanded_sentence[cursor : match.start()])
        wildcard_name = match.group(1)
        clean_parts.append(wildcard_name)
        if normalized_prefix := normalize_text("".join(clean_parts)):
            wildcard_infos.append((len(normalized_prefix.split()) - 1, wildcard_name))
        cursor = match.end()
    clean_parts.append(expanded_sentence[cursor:])
    return _clean_expanded_text("".join(clean_parts)), wildcard_infos


def _build_candidate(
    expansion: _CandidateExpansion,
    intent_name: str,
    source: CandidateSource,
    language: str,
    base_metadata: Mapping[str, str],
    slot_output_values: Mapping[str, Mapping[str, Any]] | None = None,
    static_slots_dict: dict[str, str] | None = None,
    literal_variants: tuple[frozenset[str], ...] | None = None,
    normalized_expanded_sentence: str | None = None,
) -> Candidate | None:
    """Construct a Candidate with exact slot metadata from template expansion."""
    slot_mappings = _slots_from_bindings(expansion.slot_bindings, slot_output_values)
    if slot_mappings is None:
        return None
    slots, raw_slots = slot_mappings

    if static_slots_dict:
        slots = {**static_slots_dict, **slots}
        raw_slots = {**static_slots_dict, **raw_slots}
    metadata = dict(base_metadata)
    if slots:
        metadata["slots"] = orjson.dumps(slots).decode("utf-8")
        metadata["slots_raw"] = orjson.dumps(raw_slots).decode("utf-8")
    candidate = Candidate(
        text=expansion.text,
        intent_name=intent_name,
        source=source,
        language=language,
        metadata=metadata,
        slot_values=tuple(str(value) for value in slots.values() if value is not None),
        normalized_text=(
            normalized_expanded_sentence if normalized_expanded_sentence is not None else ""
        ),
    )
    if not candidate.normalized_text:
        # A candidate whose text normalizes to empty (e.g. pure punctuation)
        # can never match a query, and the store deserializer rejects the
        # whole persisted index when it encounters one, which silently
        # forces a full rebuild on every restart.
        return None
    if literal_variants is not None:
        object.__setattr__(candidate, "_literal_variants", literal_variants)
    object.__setattr__(candidate, "_wildcard_infos", expansion.wildcard_infos)
    return candidate


def _expanded_text_length_key(text: str) -> tuple[int, int]:
    """Return the candidate length sort key without allocating token lists."""
    return ((text.count(" ") + 1) if text else 0, len(text))


def _data_item_candidate_state(
    language: str,
    source_key: str,
    source_config: Mapping[str, Any],
    source: CandidateSource,
    intent_name: str,
    intent_config: Mapping[str, Any],
    data_item: Mapping[str, Any],
) -> _DataItemCandidateState:
    """Build data-item state shared by all of its sentence templates."""
    return _DataItemCandidateState(
        language=language,
        source_key=source_key,
        source=source,
        intent_name=intent_name,
        expansion_rules=_expansion_rules(source_config, intent_config, data_item),
        base_slot_values=_slot_values(source_config, intent_config, data_item),
        output_value_maps=_slot_output_value_maps(source_config, intent_config, data_item),
        wildcard_slots=_wildcard_slots(source_config, intent_config, data_item),
        context_slots=_context_slot_names(data_item),
        static_slots=_static_slot_values(data_item),
    )


def _resolve_candidate_sentence_slots(
    sentence: str,
    data_item: Mapping[str, Any],
    state: _DataItemCandidateState,
    registry_slot_values: Mapping[str, tuple[str, ...]],
) -> (
    tuple[
        tuple[TemplateSlotReference, ...],
        dict[str, tuple[str, ...]],
        frozenset[str],
    ]
    | None
):
    """Resolve slot references and values, or return None for an incomplete sentence."""
    slot_references = _template_slot_references(sentence, state.expansion_rules)
    sentence_slots = frozenset(ref.list_name for ref in slot_references)
    slot_values = merge_slot_values(
        state.base_slot_values,
        _registry_slot_values_for_template(
            data_item,
            sentence_slots=sentence_slots,
            base_data_slot_values=state.base_slot_values,
            registry_slot_values=registry_slot_values,
        ),
    )
    if any(not slot_values.get(slot) for slot in _required_slots(sentence, state.expansion_rules)):
        return None
    return slot_references, slot_values, sentence_slots & state.wildcard_slots


def _candidate_sentence_metadata(
    sentence: str,
    wildcard_slots: frozenset[str],
    state: _DataItemCandidateState,
) -> tuple[dict[str, str], tuple[frozenset[str], ...]]:
    """Build candidate metadata shared by every expansion of one sentence."""
    literal_text, variants = _template_literals(sentence, state.expansion_rules)
    metadata = _candidate_metadata(
        state.source_key,
        sentence,
        state.expansion_rules,
        literal_text=literal_text,
        literal_variants=variants,
    )
    if state.static_slots:
        metadata["static_slots"] = ",".join(sorted(state.static_slots))
    if state.context_slots:
        metadata["context_slots"] = ",".join(sorted(state.context_slots))
    if wildcard_slots:
        metadata["wildcard_slots"] = ",".join(sorted(wildcard_slots))
    return metadata, variants


def _prepare_candidate_sentence(
    sentence: str,
    data_item: Mapping[str, Any],
    state: _DataItemCandidateState,
    registry_slot_values: Mapping[str, tuple[str, ...]],
) -> _PreparedCandidateSentence | None:
    """Prepare one sentence for bounded candidate expansion."""
    resolved_slots = _resolve_candidate_sentence_slots(
        sentence,
        data_item,
        state,
        registry_slot_values,
    )
    if resolved_slots is None:
        return None
    slot_references, slot_values, wildcard_slots = resolved_slots
    metadata, variants = _candidate_sentence_metadata(sentence, wildcard_slots, state)
    return _PreparedCandidateSentence(
        sentence=sentence,
        slot_references=slot_references,
        slot_values=slot_values,
        wildcard_slots=wildcard_slots,
        metadata=metadata,
        literal_variants=variants,
    )


def _prepared_sentence_candidates(
    prepared: _PreparedCandidateSentence,
    state: _DataItemCandidateState,
) -> Iterable[Candidate]:
    """Yield candidates for one fully prepared sentence template."""
    expansions = _candidate_expansions(
        prepared.sentence,
        prepared.slot_values,
        state.expansion_rules,
        prepared.slot_references,
        prepared.wildcard_slots,
        slot_output_values=state.output_value_maps,
    )
    for expansion in sorted(
        expansions,
        key=lambda item: _expanded_text_length_key(item.text),
    ):
        candidate = _build_candidate(
            expansion,
            state.intent_name,
            state.source,
            state.language,
            prepared.metadata,
            state.output_value_maps,
            state.static_slots,
            literal_variants=prepared.literal_variants,
        )
        if candidate is not None:
            yield candidate


def _data_item_candidates_generator(
    language: str,
    source_key: str,
    source_config: Mapping[str, Any],
    source: CandidateSource,
    intent_name: str,
    intent_config: Mapping[str, Any],
    data_item: Mapping[str, Any],
    registry_slot_values: Mapping[str, tuple[str, ...]],
) -> Iterable[Candidate]:
    """Generate candidates for a single data item, sorting each sentence's candidates by length."""
    sentences = data_item.get("sentences", [])
    if not isinstance(sentences, list):
        return
    state = _data_item_candidate_state(
        language,
        source_key,
        source_config,
        source,
        intent_name,
        intent_config,
        data_item,
    )

    for sentence in sentences:
        if not isinstance(sentence, str):
            continue
        prepared = _prepare_candidate_sentence(
            sentence,
            data_item,
            state,
            registry_slot_values,
        )
        if prepared is None:
            continue
        yield from _prepared_sentence_candidates(prepared, state)


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
        has_name = any(isinstance(s, str) and _uses_entity_slot_alias(s) for s in sentences)
        if has_name:
            _name_data_items.append(di)
        else:
            _other_data_items.append(di)
    ordered_data_items = _name_data_items + _other_data_items

    generators = [
        _data_item_candidates_generator(
            language,
            source_key,
            source_config,
            source,
            intent_name,
            intent_config,
            data_item,
            registry_slot_values,
        )
        for data_item in ordered_data_items
    ]

    candidates: list[Candidate] = []
    iterators = [iter(g) for g in generators]
    while iterators:
        next_iterators = []
        for it in iterators:
            with contextlib.suppress(StopIteration):
                candidates.append(next(it))
                if len(candidates) >= max_candidates:
                    return tuple(candidates)
                next_iterators.append(it)
        iterators = next_iterators

    return tuple(candidates)


def _uses_entity_slot_alias(sentence: str) -> bool:
    """Return whether a sentence references an entity slot alias."""
    return any(
        f"<{slot_name}" in sentence or f"{{{slot_name}" in sentence
        for slot_name in ENTITY_SLOT_NAME_SET
    )


def _is_area_only_excluded(
    template: DynamicRegistryTemplate,
    *,
    include_area_only_templates: bool,
) -> bool:
    """Return True if the template should be skipped as an unrescued area-only template."""
    if include_area_only_templates or not template.query_slots or template.entity_slots:
        return False
    return not _is_domain_scoped_location_template(template.sentence_slots, template.domains)


def _resolve_range_grid_params(
    template: DynamicRegistryTemplate,
    slot_name: str,
    from_val: float,
) -> tuple[float | None, str | None, float]:
    """Resolve step, fraction_type, and grid_start for a range slot."""
    range_info = template.range_step_multipliers.get(slot_name)
    step = range_info.step if range_info else None
    fraction_type = range_info.fraction_type if range_info else None
    original_from = range_info.original_from if range_info else None
    grid_start = original_from if original_from is not None else from_val
    return step, fraction_type, grid_start


def _find_slot_context_anchors(
    sentence: str,
    slot_name: str,
) -> list[tuple[str | None, str | None]]:
    """Return list of (prev_anchor, next_anchor) pairs for a slot in the template.

    Anchors are simple literal word tokens adjacent to the range slot in the
    original template, skipping over optional or alternative tokens. They are used
    to map numbers in ambiguous multi-number queries to the correct range slot.
    """
    t_raw_tokens = sentence.split()
    t_clean_tokens = [t.strip(_CLEAN_ANCHOR_STRIP_CHARS) for t in t_raw_tokens]

    def is_literal(w: str) -> bool:
        return all(marker not in w for marker in _TEMPLATE_MARKERS)

    pat1 = f"{{{slot_name}}}"
    pat2 = f"{{{slot_name}:"
    pat3 = f"<{slot_name}>"
    pat4 = f"<{slot_name}:"
    pat5 = f":{slot_name}}}"
    pat6 = f":{slot_name}>"
    pat7 = f":{slot_name}:"

    anchors = []
    for i, raw_t in enumerate(t_raw_tokens):
        if (
            pat1 in raw_t
            or pat2 in raw_t
            or pat3 in raw_t
            or pat4 in raw_t
            or pat5 in raw_t
            or pat6 in raw_t
            or pat7 in raw_t
        ):
            prev_anchor = None
            idx_left = i - 1
            while idx_left >= 0:
                if is_literal(t_raw_tokens[idx_left]):
                    candidate_anchor = normalize_text(t_clean_tokens[idx_left])
                    if len(candidate_anchor.split()) == 1:
                        prev_anchor = candidate_anchor
                        break
                idx_left -= 1

            next_anchor = None
            idx_right = i + 1
            while idx_right < len(t_raw_tokens):
                if is_literal(t_raw_tokens[idx_right]):
                    candidate_anchor = normalize_text(t_clean_tokens[idx_right])
                    if len(candidate_anchor.split()) == 1:
                        next_anchor = candidate_anchor
                        break
                idx_right += 1

            anchors.append((prev_anchor, next_anchor))
    return anchors


def _matches_context(
    q_tokens: list[str],
    j: int,
    anchors: list[tuple[str | None, str | None]],
) -> bool:
    """Return True if the query numeric token at index j matches any anchor context.

    If the slot has literal neighbor words in the template, we require that the numeric
    token in the query is adjacent (or close to, in case of intermediate optional tokens)
    to the corresponding literal neighbor.
    """
    if not anchors:
        return True
    if all(p is None and n is None for p, n in anchors):
        return True
    for p, n in anchors:
        if p is not None and j > 0:
            left_window = q_tokens[max(0, j - _CONTEXT_ANCHOR_WINDOW) : j]
            if p in left_window:
                return True
        if n is not None and j < len(q_tokens) - 1:
            right_window = q_tokens[j + 1 : j + 1 + _CONTEXT_ANCHOR_WINDOW]
            if n in right_window:
                return True
    return False


def _matched_range_query_tokens(
    template: DynamicRegistryTemplate,
    slot_name: str,
    boundaries: tuple[float, float],
    query_tokens: list[str],
    query_numeric_tokens: Sequence[tuple[str, float, int]],
) -> tuple[str, ...]:
    """Return query tokens valid for one compiled range slot."""
    from_value, to_value = boundaries
    step, fraction_type, grid_start = _resolve_range_grid_params(
        template,
        slot_name,
        from_value,
    )
    anchors = template.range_slot_anchors.get(slot_name, [])
    if len(query_numeric_tokens) > 1 and (
        not anchors
        or all(previous is None and following is None for previous, following in anchors)
    ):
        return ()
    return tuple(
        token
        for token, value, index in query_numeric_tokens
        if from_value <= value <= to_value
        and is_valid_range_value(value, grid_start, step, fraction_type)
        and (len(query_numeric_tokens) <= 1 or _matches_context(query_tokens, index, anchors))
    )


def _resolve_range_slot_values(
    template: DynamicRegistryTemplate,
    query_normalized: str,
    query_numeric_tokens: Sequence[tuple[str, float, int]] | None = None,
) -> Mapping[str, tuple[str, ...]]:
    """Resolve and return matched numeric values for range slots.

    Args:
        template: Compiled template containing range metadata.
        query_normalized: Normalized user query.
        query_numeric_tokens: Optional pre-parsed ``(text, value, index)`` tuples.

    Returns:
        Base slot values extended with unambiguous, range-valid query numbers.
    """
    if not template.range_slots:
        return template.base_data_slot_values

    base_data_slot_values = dict(template.base_data_slot_values)
    q_tokens = query_normalized.split()
    if query_numeric_tokens is None:
        query_numeric_tokens = _query_numeric_tokens(query_normalized)

    for slot_name, boundaries in template.range_slots.items():
        if matched_numbers := _matched_range_query_tokens(
            template,
            slot_name,
            boundaries,
            q_tokens,
            query_numeric_tokens,
        ):
            existing = base_data_slot_values.get(slot_name, ())
            base_data_slot_values[slot_name] = tuple(dict.fromkeys((*existing, *matched_numbers)))

    return base_data_slot_values


def _resolve_template_slot_values(
    template: DynamicRegistryTemplate,
    registry_slot_values: Mapping[str, tuple[str, ...]],
    registry_slot_index: RegistrySlotIndex,
    query_normalized: str,
    query_tokens: frozenset[str],
    relevant_cache: RegistryRelevanceCache,
    scoped_cache: dict[tuple[str, tuple[str, ...]], tuple[RegistrySlotValue, ...]],
    *,
    query_no_diac: str | None = None,
    query_tokens_no_diac: frozenset[str] | None = None,
    query_numeric_tokens: Sequence[tuple[str, float, int]] | None = None,
    retrieval_stats: RegistryRetrievalStats | None = None,
    anchored_slot_windows: Mapping[str, tuple[SlotQueryWindow, ...]] | None = None,
    anchored_query_slots: set[str] | None = None,
) -> Mapping[str, tuple[str, ...]] | None:
    """Return merged slot values for a template, or None if required slots are missing."""
    if template.query_slots:
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
            retrieval_stats=retrieval_stats,
            anchored_slot_windows=anchored_slot_windows,
            anchored_query_slots=anchored_query_slots,
        )
        if not dynamic_registry_slots:
            return None
    else:
        dynamic_registry_slots = {}

    base_data_slot_values = _resolve_range_slot_values(
        template,
        query_normalized,
        query_numeric_tokens=query_numeric_tokens,
    )

    slot_values = merge_slot_values(base_data_slot_values, dynamic_registry_slots)
    if any(not slot_values.get(slot) for slot in template.required_slots):
        return None
    return slot_values


def _expand_template_candidates(
    template: DynamicRegistryTemplate,
    intent_name: str,
    source: CandidateSource,
    language: str,
    slot_values: Mapping[str, tuple[str, ...]],
    query_normalized: str,
    query_tokens: frozenset[str],
    query_no_diac: str | None,
    *,
    limit: int,
) -> tuple[Candidate, ...]:
    """Expand a single template into query-relevant candidates up to limit."""
    if limit <= 0:
        return ()
    candidates: list[Candidate] = []
    seen: set[tuple[str, str]] = set()

    for preferred_values in _exact_slot_preferred_value_maps(
        template,
        slot_values,
        query_normalized,
        query_tokens,
        query_no_diac,
        language,
    ):
        before_count = len(candidates)
        _append_query_template_candidates(
            candidates,
            seen,
            template,
            intent_name,
            source,
            language,
            preferred_values,
            query_normalized,
            query_tokens,
            query_no_diac,
            limit=limit,
        )
        if len(candidates) >= limit:
            return tuple(candidates)
        if _has_exact_query_candidate(
            candidates[before_count:],
            query_normalized,
            query_no_diac,
            language,
        ):
            return tuple(candidates)

    _append_query_template_candidates(
        candidates,
        seen,
        template,
        intent_name,
        source,
        language,
        slot_values,
        query_normalized,
        query_tokens,
        query_no_diac,
        limit=limit,
    )
    return tuple(candidates)


def _process_range_slot_output(
    template: DynamicRegistryTemplate,
    slot_name: str,
    from_val: float,
    to_val: float,
    values: tuple[str, ...],
    slot_map: dict[str, Any],
) -> None:
    """Process output mappings for a single range slot."""
    for val_str in values:
        if val_str not in slot_map:
            val_float = parse_float(val_str)
            if val_float is not None and from_val <= val_float <= to_val:
                step, fraction_type, grid_start = _resolve_range_grid_params(
                    template, slot_name, from_val
                )
                if is_valid_range_value(val_float, grid_start, step, fraction_type):
                    range_info = template.range_step_multipliers.get(slot_name)
                    multiplier = range_info.multiplier if range_info else None
                    mult_val = val_float
                    if multiplier is not None:
                        mult_val = val_float * multiplier
                    slot_map[val_str] = to_output_value(mult_val)


def _build_slot_output_values(
    template: DynamicRegistryTemplate,
    slot_values: Mapping[str, tuple[str, ...]],
) -> Mapping[str, Mapping[str, Any]]:
    """Build slot output value mapping by processing range slots and base configurations.

    This scales the matched numeric tokens by the slot multiplier to build the final
    output slot dictionary (e.g. converting "volume down by 20" to a slot value of -20).
    """
    slot_output_values = dict(template.slot_output_values)
    if not template.range_slots:
        return slot_output_values

    for slot_name, (from_val, to_val) in template.range_slots.items():
        if values := slot_values.get(slot_name):
            slot_map = dict(slot_output_values.get(slot_name, {}))
            _process_range_slot_output(
                template,
                slot_name,
                from_val,
                to_val,
                values,
                slot_map,
            )
            slot_output_values[slot_name] = slot_map
    return slot_output_values


def _append_query_template_candidates(
    candidates: list[Candidate],
    seen: set[tuple[str, str]],
    template: DynamicRegistryTemplate,
    intent_name: str,
    source: CandidateSource,
    language: str,
    slot_values: Mapping[str, tuple[str, ...]],
    query_normalized: str,
    query_tokens: frozenset[str],
    query_no_diac: str | None,
    *,
    limit: int,
) -> None:
    """Append query-expanded template candidates until the template limit is reached."""
    slot_output_values = _build_slot_output_values(template, slot_values)
    static_slots_dict = dict(template.static_slots)

    base_metadata = dict(template.metadata)

    for expansion, expanded_normalized in _query_candidate_expansions(
        template.sentence,
        slot_values,
        template.expansion_rules,
        template.slot_references,
        template.wildcard_slots,
        query_normalized,
        query_tokens,
        slot_output_values=slot_output_values,
    ):
        key = (expanded_normalized, intent_name)
        if key in seen:
            continue
        candidate = _build_candidate(
            expansion,
            intent_name,
            source,
            language,
            base_metadata,
            slot_output_values,
            static_slots_dict,
            literal_variants=template.literal_token_variants,
            normalized_expanded_sentence=expanded_normalized,
        )
        if candidate is None:
            continue
        seen.add(key)
        candidates.append(candidate)
        if len(candidates) >= limit:
            return


def _has_exact_query_candidate(
    candidates: Sequence[Candidate],
    query_normalized: str,
    query_no_diac: str | None,
    language: str,
) -> bool:
    """Return whether candidates contain an exact normalized query match."""
    for candidate in candidates:
        if candidate.normalized_text == query_normalized:
            return True
        if query_no_diac and _cached_normalize_no_diac(candidate.text, language) == query_no_diac:
            return True
    return False


def _base_exact_slot_value_map(
    template: DynamicRegistryTemplate,
    slot_values: Mapping[str, tuple[str, ...]],
    exact_slots: Mapping[str, tuple[str, ...]],
    optional_location_slots: Set[str],
) -> dict[str, tuple[str, ...]]:
    """Return the primary slot map with all exact query values prioritized."""
    values = dict(slot_values)
    values |= exact_slots
    if exact_slots.keys() & set(template.entity_slots):
        for location_slot in optional_location_slots - exact_slots.keys():
            values.pop(location_slot, None)
    return values


def _entity_exact_slot_value_maps(
    template: DynamicRegistryTemplate,
    slot_values: Mapping[str, tuple[str, ...]],
    exact_slots: Mapping[str, tuple[str, ...]],
    optional_location_slots: Set[str],
) -> Iterable[dict[str, tuple[str, ...]]]:
    """Yield maps prioritizing each exact entity independently."""
    required_location_slots = template.required_slots & LOCATION_SLOT_NAME_SET
    for slot_name in template.entity_slots:
        exact_values = exact_slots.get(slot_name)
        if not exact_values:
            continue
        values = dict(slot_values)
        values[slot_name] = exact_values
        for location_slot in optional_location_slots:
            values.pop(location_slot, None)
        for location_slot in required_location_slots:
            if exact_location_values := exact_slots.get(location_slot):
                values[location_slot] = exact_location_values
        yield values


def _domain_location_exact_slot_value_maps(
    template: DynamicRegistryTemplate,
    slot_values: Mapping[str, tuple[str, ...]],
    exact_slots: Mapping[str, tuple[str, ...]],
) -> Iterable[dict[str, tuple[str, ...]]]:
    """Yield domain-scoped maps prioritizing exact location slots."""
    if not template.domains:
        return
    for slot_name in template.query_slots:
        if slot_name not in LOCATION_SLOT_NAME_SET:
            continue
        if exact_values := exact_slots.get(slot_name):
            values = dict(slot_values)
            values[slot_name] = exact_values
            yield values


def _exact_slot_preferred_value_maps(
    template: DynamicRegistryTemplate,
    slot_values: Mapping[str, tuple[str, ...]],
    query_normalized: str,
    query_tokens: frozenset[str],
    query_no_diac: str | None,
    language: str,
) -> tuple[Mapping[str, tuple[str, ...]], ...]:
    """Return slot maps that give exact query slot values first expansion priority."""
    query_tokens_no_diac = frozenset(query_no_diac.split()) if query_no_diac else None
    if not _literal_token_variants_fully_match_query(
        template.literal_token_variants,
        template.literal_token_variants_no_diac,
        template.metadata.get("literal_text"),
        query_tokens,
        query_normalized,
        query_tokens_no_diac=query_tokens_no_diac,
        query_no_diac=query_no_diac,
        language=language,
    ):
        return ()

    exact_slots = _exact_query_slot_values(
        slot_values,
        template.sentence_slots,
        query_normalized,
        query_no_diac,
        language,
    )
    if not exact_slots:
        return ()

    optional_location_slots = set(LOCATION_SLOT_NAMES) - template.required_slots
    preferred = [
        _base_exact_slot_value_map(
            template,
            slot_values,
            exact_slots,
            optional_location_slots,
        ),
        *_entity_exact_slot_value_maps(
            template,
            slot_values,
            exact_slots,
            optional_location_slots,
        ),
        *_domain_location_exact_slot_value_maps(
            template,
            slot_values,
            exact_slots,
        ),
    ]
    return tuple(_deduplicate_slot_value_maps(preferred))


def _literal_token_variants_fully_match_query(
    literal_variants: tuple[frozenset[str], ...],
    literal_variants_no_diac: tuple[frozenset[str], ...],
    literal_text: str | None,
    query_tokens: frozenset[str],
    query_normalized: str,
    *,
    query_tokens_no_diac: frozenset[str] | None = None,
    query_no_diac: str | None = None,
    language: str | None = None,
) -> bool:
    """Return whether at least one full literal/action variant appears in the query."""
    if not literal_variants:
        return True
    for literal_tokens in literal_variants:
        if not literal_tokens or literal_tokens.issubset(query_tokens):
            return True
    if query_tokens_no_diac:
        for literal_tokens_no_diac in literal_variants_no_diac:
            if not literal_tokens_no_diac or literal_tokens_no_diac.issubset(query_tokens_no_diac):
                return True
    return bool(
        literal_text
        and _compact_script_phrase_fallback_enabled(query_normalized)
        and _literal_text_phrase_matches_query(
            literal_text,
            query_normalized,
            query_no_diac,
            language,
        )
    )


def _literal_text_phrase_matches_query(
    literal_text: str,
    query_normalized: str,
    query_no_diac: str | None,
    language: str | None,
) -> bool:
    """Return whether any literal text variant occurs as a normalized query phrase."""
    for literal_normalized, literal_no_diac in _normalized_literal_phrase_variants(
        literal_text,
        language,
    ):
        if _normalized_phrase_occurs_in_query(literal_normalized, query_normalized):
            return True
        if query_no_diac and _normalized_phrase_occurs_in_query(literal_no_diac, query_no_diac):
            return True
    return False


@lru_cache(maxsize=4096)
def _normalized_literal_phrase_variants(
    literal_text: str,
    language: str | None,
) -> tuple[tuple[str, str], ...]:
    """Return cached normalized literal phrase variants."""
    variants: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for literal_variant in literal_text.split("|"):
        if not literal_variant.strip():
            continue
        literal_normalized = normalize_text(literal_variant)
        if not literal_normalized:
            continue
        normalized = (
            literal_normalized,
            _cached_normalize_no_diac(literal_variant, language),
        )
        if normalized in seen:
            continue
        seen.add(normalized)
        variants.append(normalized)
    return tuple(variants)


def _exact_query_slot_values(
    slot_values: Mapping[str, tuple[str, ...]],
    slot_names: Iterable[str],
    query_normalized: str,
    query_no_diac: str | None,
    language: str,
) -> dict[str, tuple[str, ...]]:
    """Return slot values whose normalized text occurs exactly as a query phrase."""
    exact_slots: dict[str, tuple[str, ...]] = {}
    for slot_name in slot_names:
        values = slot_values.get(slot_name)
        if not values:
            continue
        if exact_values := tuple(
            value
            for value in values
            if _slot_value_occurs_in_query(value, query_normalized, query_no_diac, language)
        ):
            exact_slots[slot_name] = exact_values
    return exact_slots


def _slot_value_occurs_in_query(
    value: str,
    query_normalized: str,
    query_no_diac: str | None,
    language: str,
) -> bool:
    """Return whether a registry value appears as a normalized phrase in the query."""
    value_normalized = normalize_text(value)
    if _normalized_phrase_occurs_in_query(value_normalized, query_normalized):
        return True
    return bool(
        query_no_diac
        and _normalized_phrase_occurs_in_query(
            _cached_normalize_no_diac(value, language),
            query_no_diac,
        )
    )


def _normalized_phrase_occurs_in_query(phrase: str, query_normalized: str) -> bool:
    """Return whether a normalized phrase occurs on query token boundaries.

    Compact-script phrases (CJK, Thai, ...) are additionally matched against
    the space-stripped query: speech-to-text output for those languages may
    include segmentation spaces that the grammar literals do not carry.
    """
    if not phrase:
        return False
    if f" {phrase} " in f" {query_normalized} ":
        return True
    return bool(
        " " not in phrase
        and _uses_compact_non_latin_script(phrase)
        and phrase in query_normalized.replace(" ", "")
    )


def _extract_slot_query_window(
    query_normalized: str,
    prefix: str,
    suffix: str,
) -> str | None:
    """Extract the entire normalized span between exact template boundaries."""
    if prefix and not query_normalized.startswith(prefix):
        return None
    if suffix and not query_normalized.endswith(suffix):
        return None
    start = len(prefix)
    end = len(query_normalized) - len(suffix) if suffix else len(query_normalized)
    if start > end:
        return None
    window = query_normalized[start:end].strip()
    return window or None


def _template_anchored_slot_query_windows(
    template: DynamicRegistryTemplate,
    query_normalized: str,
    query_no_diac: str | None,
) -> Mapping[str, tuple[SlotQueryWindow, ...]]:
    """Return strongest bounded whole-slot windows supported by template literals."""
    selected: dict[str, tuple[SlotQueryWindow, ...]] = {}
    for slot_name, patterns in template.slot_anchor_patterns.items():
        windows: dict[tuple[str, str | None], SlotQueryWindow] = {}
        for pattern in patterns:
            normalized_window = _extract_slot_query_window(
                query_normalized,
                pattern.prefix,
                pattern.suffix,
            )
            no_diac_window = (
                _extract_slot_query_window(
                    query_no_diac,
                    pattern.prefix_no_diacritics,
                    pattern.suffix_no_diacritics,
                )
                if query_no_diac
                else None
            )
            if normalized_window is None and no_diac_window is None:
                continue
            primary_window = normalized_window or no_diac_window
            if primary_window is None:
                continue
            window = SlotQueryWindow(
                normalized_text=primary_window,
                normalized_no_diacritics=no_diac_window,
                anchor_length=len(pattern.prefix.strip()) + len(pattern.suffix.strip()),
            )
            key = (window.normalized_text, window.normalized_no_diacritics)
            existing = windows.get(key)
            if existing is None or window.anchor_length > existing.anchor_length:
                windows[key] = window
        if windows:
            selected[slot_name] = tuple(
                sorted(
                    windows.values(),
                    key=lambda window: (window.anchor_length, -len(window.normalized_text)),
                    reverse=True,
                )[:_SLOT_QUERY_WINDOW_LIMIT]
            )
    return selected


def _deduplicate_slot_value_maps(
    maps: Iterable[dict[str, tuple[str, ...]]],
) -> Iterable[Mapping[str, tuple[str, ...]]]:
    """Yield unique slot-value maps while preserving priority order."""
    seen: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()
    for values in maps:
        key = tuple(sorted(values.items()))
        if key in seen:
            continue
        seen.add(key)
        yield values


def _compiled_template_enabled_for_query(
    template: DynamicRegistryTemplate,
    *,
    include_literal_only_templates: bool,
    include_area_only_templates: bool,
    literal_only_wildcards_only: bool,
) -> bool:
    """Return whether query-time configuration permits a compiled template."""
    if not include_literal_only_templates and not template.query_slots:
        return False
    if literal_only_wildcards_only and not template.query_slots and not template.wildcard_slots:
        return False
    return not _is_area_only_excluded(
        template,
        include_area_only_templates=include_area_only_templates,
    )


def _query_candidates_from_compiled_intent(
    language: str,
    compiled_intent: DynamicRegistryIntent,
    registry_slot_values: Mapping[str, tuple[str, ...]],
    registry_slot_index: RegistrySlotIndex,
    query_normalized: str,
    query_tokens: frozenset[str],
    relevant_cache: RegistryRelevanceCache,
    scoped_cache: dict[tuple[str, tuple[str, ...]], tuple[RegistrySlotValue, ...]],
    *,
    max_candidates: int,
    query_no_diac: str | None = None,
    query_tokens_no_diac: frozenset[str] | None = None,
    include_literal_only_templates: bool = True,
    include_area_only_templates: bool = True,
    literal_only_wildcards_only: bool = False,
    query_numeric_tokens: Sequence[tuple[str, float, int]] | None = None,
    retrieval_stats: RegistryRetrievalStats | None = None,
) -> tuple[Candidate, ...]:
    """Expand one compiled intent using query-relevant registry values."""
    candidates: list[Candidate] = []
    for template in compiled_intent.templates:
        if not _compiled_template_enabled_for_query(
            template,
            include_literal_only_templates=include_literal_only_templates,
            include_area_only_templates=include_area_only_templates,
            literal_only_wildcards_only=literal_only_wildcards_only,
        ):
            continue
        anchored_slot_windows = _template_anchored_slot_query_windows(
            template,
            query_normalized,
            query_no_diac,
        )
        if not _template_matches_query_literals(
            language,
            template,
            query_normalized,
            query_tokens,
            query_no_diac,
            query_tokens_no_diac,
            anchored_slot_windows,
            include_area_only_templates=include_area_only_templates,
        ):
            continue
        if template_candidates := _query_candidates_from_template(
            language,
            template,
            compiled_intent,
            registry_slot_values,
            registry_slot_index,
            query_normalized,
            query_tokens,
            relevant_cache,
            scoped_cache,
            max_candidates=max_candidates,
            query_no_diac=query_no_diac,
            query_tokens_no_diac=query_tokens_no_diac,
            query_numeric_tokens=query_numeric_tokens,
            retrieval_stats=retrieval_stats,
            anchored_slot_windows=anchored_slot_windows,
        ):
            candidates = _merge_top_query_candidates(
                candidates,
                template_candidates,
                max_candidates,
                query_normalized,
                query_tokens,
            )
    return tuple(
        _top_query_candidates(
            candidates,
            max_candidates,
            query_normalized,
            query_tokens,
        )
    )


def _template_matches_query_literals(
    language: str,
    template: DynamicRegistryTemplate,
    query_normalized: str,
    query_tokens: frozenset[str],
    query_no_diac: str | None,
    query_tokens_no_diac: frozenset[str] | None,
    anchored_slot_windows: Mapping[str, tuple[SlotQueryWindow, ...]],
    *,
    include_area_only_templates: bool,
) -> bool:
    """Return whether a template's literals justify expanding it for this query."""
    domain_area_rescue = (
        not include_area_only_templates
        and template.query_slots
        and not template.entity_slots
        and _is_domain_scoped_location_template(template.sentence_slots, template.domains)
    )
    if (
        domain_area_rescue
        and not anchored_slot_windows
        and not _has_exact_literal_variant(
            template.literal_token_variants,
            query_tokens,
            template.literal_token_variants_no_diac,
            query_tokens_no_diac,
        )
    ):
        return False
    return bool(anchored_slot_windows) or _literal_token_variants_match_query(
        template.literal_token_variants,
        template.literal_token_variants_no_diac,
        query_tokens,
        literal_text=template.metadata.get("literal_text"),
        query_normalized=query_normalized,
        query_tokens_no_diac=query_tokens_no_diac,
        query_no_diac=query_no_diac,
        language=language,
    )


def _query_candidates_from_template(
    language: str,
    template: DynamicRegistryTemplate,
    compiled_intent: DynamicRegistryIntent,
    registry_slot_values: Mapping[str, tuple[str, ...]],
    registry_slot_index: RegistrySlotIndex,
    query_normalized: str,
    query_tokens: frozenset[str],
    relevant_cache: RegistryRelevanceCache,
    scoped_cache: dict[tuple[str, tuple[str, ...]], tuple[RegistrySlotValue, ...]],
    *,
    max_candidates: int,
    query_no_diac: str | None,
    query_tokens_no_diac: frozenset[str] | None,
    query_numeric_tokens: Sequence[tuple[str, float, int]] | None,
    retrieval_stats: RegistryRetrievalStats | None,
    anchored_slot_windows: Mapping[str, tuple[SlotQueryWindow, ...]],
) -> tuple[Candidate, ...]:
    """Resolve one template's slot values and expand its query candidates."""
    anchored_query_slots: set[str] = set()
    slot_values = _resolve_template_slot_values(
        template,
        registry_slot_values,
        registry_slot_index,
        query_normalized,
        query_tokens,
        relevant_cache,
        scoped_cache,
        query_no_diac=query_no_diac,
        query_tokens_no_diac=query_tokens_no_diac,
        query_numeric_tokens=query_numeric_tokens,
        retrieval_stats=retrieval_stats,
        anchored_slot_windows=anchored_slot_windows,
        anchored_query_slots=anchored_query_slots,
    )
    if slot_values is None:
        return ()
    template_limit = min(DEFAULT_MAX_CANDIDATES_PER_TEMPLATE, max_candidates)
    template_candidates = _expand_template_candidates(
        template,
        compiled_intent.intent_name,
        compiled_intent.source,
        language,
        slot_values,
        query_normalized,
        query_tokens,
        query_no_diac,
        limit=template_limit,
    )
    if anchored_query_slots:
        template_candidates = tuple(
            replace(
                candidate,
                metadata={**candidate.metadata, "registry_retrieval": "anchored"},
            )
            for candidate in template_candidates
        )
    return template_candidates


def _has_exact_literal_variant(
    literal_variants: tuple[frozenset[str], ...],
    query_tokens: frozenset[str],
    literal_variants_no_diac: tuple[frozenset[str], ...],
    query_tokens_no_diac: frozenset[str] | None,
) -> bool:
    """Return whether one non-empty action-literal variant is fully anchored."""
    if any(variant and variant.issubset(query_tokens) for variant in literal_variants):
        return True
    return bool(
        query_tokens_no_diac
        and any(
            variant and variant.issubset(query_tokens_no_diac)
            for variant in literal_variants_no_diac
        )
    )


def _compiled_query_registry_slot_values(
    template: DynamicRegistryTemplate,
    registry_slot_values: Mapping[str, tuple[str, ...]],
    registry_slot_index: RegistrySlotIndex,
    query_normalized: str,
    query_tokens: frozenset[str],
    relevant_cache: RegistryRelevanceCache,
    scoped_cache: dict[tuple[str, tuple[str, ...]], tuple[RegistrySlotValue, ...]],
    *,
    query_no_diac: str | None = None,
    query_tokens_no_diac: frozenset[str] | None = None,
    retrieval_stats: RegistryRetrievalStats | None = None,
    anchored_slot_windows: Mapping[str, tuple[SlotQueryWindow, ...]] | None = None,
    anchored_query_slots: set[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return narrowed registry slots for a compiled template."""
    constrained = _constrained_registry_slot_values(
        template,
        registry_slot_values,
        registry_slot_index,
        scoped_cache,
    )
    matched_query_slot = False
    for slot_name in template.query_slots:
        slot_domains = template.domains if slot_name in ENTITY_SLOT_NAME_SET else ()
        records = _scoped_registry_slot_records(
            slot_name,
            slot_domains,
            registry_slot_index,
            scoped_cache,
        )
        if not records:
            constrained.pop(slot_name, None)
            continue
        if relevant := _relevant_query_slot_values(
            template,
            slot_name,
            records,
            registry_slot_index,
            query_normalized,
            query_tokens,
            relevant_cache,
            query_no_diac=query_no_diac,
            query_tokens_no_diac=query_tokens_no_diac,
            retrieval_stats=retrieval_stats,
            anchored_slot_windows=anchored_slot_windows,
            anchored_query_slots=anchored_query_slots,
        ):
            constrained[slot_name] = relevant
            matched_query_slot = True
        else:
            constrained.pop(slot_name, None)
    return constrained if matched_query_slot else {}


def _constrained_registry_slot_values(
    template: DynamicRegistryTemplate,
    registry_slot_values: Mapping[str, tuple[str, ...]],
    registry_slot_index: RegistrySlotIndex,
    scoped_cache: dict[tuple[str, tuple[str, ...]], tuple[RegistrySlotValue, ...]],
) -> dict[str, tuple[str, ...]]:
    """Build the initial slot-value constraints before query-slot narrowing."""
    registry_slots = template.sentence_slots - set(template.base_data_slot_values)
    constrained = _registry_slot_values_for_slots(
        registry_slot_values,
        sorted(registry_slots),
        domains=template.domains,
    )
    if template.entity_slots:
        for slot_name in LOCATION_SLOT_NAMES:
            if slot_name not in template.sentence_slots:
                constrained.pop(slot_name, None)

    if template.domains:
        for slot_name in template.entity_slots:
            constrained.pop(slot_name, None)
            if records := _scoped_registry_slot_records(
                slot_name,
                template.domains,
                registry_slot_index,
                scoped_cache,
            ):
                constrained[slot_name] = tuple(record.text for record in records)
    return constrained


def _relevant_query_slot_values(
    template: DynamicRegistryTemplate,
    slot_name: str,
    records: tuple[RegistrySlotValue, ...],
    registry_slot_index: RegistrySlotIndex,
    query_normalized: str,
    query_tokens: frozenset[str],
    relevant_cache: RegistryRelevanceCache,
    *,
    query_no_diac: str | None,
    query_tokens_no_diac: frozenset[str] | None,
    retrieval_stats: RegistryRetrievalStats | None,
    anchored_slot_windows: Mapping[str, tuple[SlotQueryWindow, ...]] | None,
    anchored_query_slots: set[str] | None,
) -> tuple[str, ...]:
    """Narrow one query slot via anchored windows, then whole-query fallback."""
    relevant: tuple[str, ...] = ()
    slot_windows = anchored_slot_windows.get(slot_name, ()) if anchored_slot_windows else ()
    for window in slot_windows:
        window_tokens = frozenset(window.normalized_text.split())
        window_tokens_no_diac = (
            frozenset(window.normalized_no_diacritics.split())
            if window.normalized_no_diacritics
            else None
        )
        relevant = _cached_query_relevant_slot_values(
            records,
            window.normalized_text,
            window_tokens,
            relevant_cache,
            registry_slot_index,
            query_no_diac=window.normalized_no_diacritics,
            query_tokens_no_diac=window_tokens_no_diac,
            retrieval_stats=retrieval_stats,
            require_whole_query=True,
        )
        if relevant:
            if anchored_query_slots is not None:
                anchored_query_slots.add(slot_name)
            break

    allow_unanchored_fallback = not template.slot_anchor_patterns or " " in query_normalized
    if not relevant and allow_unanchored_fallback:
        relevant = _cached_query_relevant_slot_values(
            records,
            query_normalized,
            query_tokens,
            relevant_cache,
            registry_slot_index,
            query_no_diac=query_no_diac,
            query_tokens_no_diac=query_tokens_no_diac,
            retrieval_stats=retrieval_stats,
            require_whole_query=False,
        )
    return relevant


def _cached_query_relevant_slot_values(
    records: tuple[RegistrySlotValue, ...],
    query_normalized: str,
    query_tokens: frozenset[str],
    relevant_cache: RegistryRelevanceCache,
    registry_slot_index: RegistrySlotIndex,
    *,
    query_no_diac: str | None,
    query_tokens_no_diac: frozenset[str] | None,
    retrieval_stats: RegistryRetrievalStats | None,
    require_whole_query: bool,
) -> tuple[str, ...]:
    """Return cached registry relevance for one normalized query span."""
    cache_key = (id(records), query_normalized, query_no_diac, require_whole_query)
    relevant = relevant_cache.get(cache_key)
    if relevant is not None:
        return relevant
    relevant = _query_relevant_precomputed_slot_values(
        records,
        query_normalized,
        query_tokens,
        registry_slot_index=registry_slot_index,
        query_no_diac=query_no_diac,
        query_tokens_no_diac=query_tokens_no_diac,
        retrieval_stats=retrieval_stats,
        require_whole_query=require_whole_query,
    )
    relevant_cache[cache_key] = relevant
    return relevant


def _scoped_registry_slot_records(
    slot_name: str,
    domains: tuple[str, ...],
    registry_slot_index: RegistrySlotIndex,
    scoped_cache: dict[tuple[str, tuple[str, ...]], tuple[RegistrySlotValue, ...]],
) -> tuple[RegistrySlotValue, ...]:
    """Return precomputed registry records for a generic or domain-scoped slot."""
    if not domains:
        return registry_slot_index.get(slot_name, ())
    cache_key = (slot_name, domains)
    cached = scoped_cache.get(cache_key)
    if cached is not None:
        return cached
    # Retrieval remains bounded by the sparse nomination and scoring budgets;
    # truncating here would make valid tail records permanently unreachable.
    records = registry_slot_index.get_scoped_records(slot_name, domains)
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
    retrieval_stats: RegistryRetrievalStats | None = None,
    require_whole_query: bool = False,
) -> tuple[str, ...]:
    """Return relevant values using precomputed normalization and tokens."""
    if registry_slot_index is None:
        candidate_records = set(values)
        has_specific_fully_anchored_record = False
    else:
        candidate_records, has_specific_fully_anchored_record = _gather_candidate_registry_records(
            values,
            registry_slot_index,
            query_normalized,
            query_tokens,
            query_no_diac,
            query_tokens_no_diac,
            retrieval_stats,
            require_whole_query=require_whole_query,
        )
    nominated_records = _nominated_registry_records(
        candidate_records,
        query_normalized,
        query_tokens,
        query_no_diac,
        query_tokens_no_diac,
        retrieval_stats,
    )
    return _score_nominated_registry_records(
        nominated_records,
        query_normalized,
        query_tokens,
        query_no_diac,
        query_tokens_no_diac,
        has_specific_fully_anchored_record=has_specific_fully_anchored_record,
        require_whole_query=require_whole_query,
        limit=limit,
    )


def _has_fully_anchored_registry_record(
    candidate_records: set[RegistrySlotValue],
    query_normalized: str,
    query_tokens: frozenset[str],
    query_no_diac: str | None,
    query_tokens_no_diac: frozenset[str] | None,
    *,
    require_whole_query: bool,
) -> bool:
    """Return whether one candidate record is specific and fully anchored."""
    if require_whole_query:
        return any(
            _slot_value_query_match_key_from_tokens(
                record.normalized_text,
                record.tokens,
                query_normalized,
                query_tokens,
                value_no_diac=record.normalized_no_diacritics,
                value_tokens_no_diac=record.tokens_no_diacritics,
                query_no_diac=query_no_diac,
                query_tokens_no_diac=query_tokens_no_diac,
                allow_fuzzy_tokens=False,
                require_whole_query=True,
            )
            is not None
            for record in candidate_records
        )
    return any(
        len(record.tokens) >= 2
        and (
            _normalized_phrase_occurs_in_query(record.normalized_text, query_normalized)
            or set(record.tokens).issubset(query_tokens)
            or (
                query_no_diac
                and (
                    _normalized_phrase_occurs_in_query(
                        record.normalized_no_diacritics,
                        query_no_diac,
                    )
                    or set(record.tokens_no_diacritics).issubset(query_tokens_no_diac or ())
                )
            )
        )
        for record in candidate_records
    )


def _gather_candidate_registry_records(
    values: tuple[RegistrySlotValue, ...],
    registry_slot_index: RegistrySlotIndex,
    query_normalized: str,
    query_tokens: frozenset[str],
    query_no_diac: str | None,
    query_tokens_no_diac: frozenset[str] | None,
    retrieval_stats: RegistryRetrievalStats | None,
    *,
    require_whole_query: bool,
) -> tuple[set[RegistrySlotValue], bool]:
    """Collect inverted-index candidate records, with fuzzy fallback when unanchored."""
    lookup = registry_slot_index.get_inverted_for_records(values)
    candidate_records: set[RegistrySlotValue] = set()
    lookup_tokens = set(query_tokens)
    lookup_tokens.update(_compound_query_tokens(query_normalized))
    if query_no_diac:
        lookup_tokens.update(_compound_query_tokens(query_no_diac))
    if query_tokens_no_diac:
        lookup_tokens.update(query_tokens_no_diac)
    for token in lookup_tokens:
        if retrieval_stats is not None:
            retrieval_stats.postings_consulted += 1
        if token in lookup:
            candidate_records.update(lookup[token])
    has_specific_fully_anchored_record = _has_fully_anchored_registry_record(
        candidate_records,
        query_normalized,
        query_tokens,
        query_no_diac,
        query_tokens_no_diac,
        require_whole_query=require_whole_query,
    )
    if not has_specific_fully_anchored_record:
        candidate_records.update(
            registry_slot_index.get_fuzzy_for_records(
                values,
                lookup_tokens,
                retrieval_stats,
            )
        )
    return candidate_records, has_specific_fully_anchored_record


def _nominated_registry_records(
    candidate_records: set[RegistrySlotValue],
    query_normalized: str,
    query_tokens: frozenset[str],
    query_no_diac: str | None,
    query_tokens_no_diac: frozenset[str] | None,
    retrieval_stats: RegistryRetrievalStats | None,
) -> list[RegistrySlotValue]:
    """Nominate the best candidate records within the per-query scoring budget."""
    score_limit = DEFAULT_MAX_REGISTRY_VALUES_NOMINATED
    if retrieval_stats is not None:
        score_limit = min(
            score_limit,
            max(
                0,
                DEFAULT_MAX_REGISTRY_VALUES_SCORED_PER_QUERY - retrieval_stats.values_scored,
            ),
        )
    if score_limit <= 0:
        return []
    nominated_records = nlargest(
        score_limit,
        candidate_records,
        key=lambda record: _registry_nomination_key(
            record,
            query_normalized,
            query_tokens,
            query_no_diac,
            query_tokens_no_diac,
        ),
    )
    if retrieval_stats is not None:
        retrieval_stats.values_nominated += len(nominated_records)
        retrieval_stats.values_scored += len(nominated_records)
    return nominated_records


def _score_nominated_registry_records(
    nominated_records: list[RegistrySlotValue],
    query_normalized: str,
    query_tokens: frozenset[str],
    query_no_diac: str | None,
    query_tokens_no_diac: frozenset[str] | None,
    *,
    has_specific_fully_anchored_record: bool,
    require_whole_query: bool,
    limit: int,
) -> tuple[str, ...]:
    """Score nominated records and return deduplicated texts in relevance order."""
    scored: list[tuple[tuple[bool, bool, bool, int, int], int, str]] = []
    query_match_tokens = query_tokens | _compound_query_tokens(query_normalized)
    query_no_diac_match_tokens = (
        query_tokens_no_diac | _compound_query_tokens(query_no_diac)
        if query_tokens_no_diac and query_no_diac
        else query_tokens_no_diac
    )
    for value in nominated_records:
        match_key = _slot_value_query_match_key_from_tokens(
            value.normalized_text,
            value.tokens,
            query_normalized,
            query_match_tokens,
            value_no_diac=value.normalized_no_diacritics,
            value_tokens_no_diac=value.tokens_no_diacritics,
            query_no_diac=query_no_diac,
            query_tokens_no_diac=query_no_diac_match_tokens,
            allow_fuzzy_tokens=not has_specific_fully_anchored_record,
            require_whole_query=require_whole_query,
        )
        if match_key is not None:
            scored.append((match_key, value.position, value.text))
    scored.sort(key=lambda item: (item[0], -item[1], item[2]), reverse=True)
    return tuple(_deduplicate_texts((value for _, _, value in scored), limit))


def _registry_nomination_key(
    record: RegistrySlotValue,
    query_normalized: str,
    query_tokens: frozenset[str],
    query_no_diac: str | None,
    query_tokens_no_diac: frozenset[str] | None,
) -> tuple[bool, bool, int, int, str]:
    """Return a cheap exact-first key before bounded full registry scoring."""
    phrase_match = _normalized_phrase_occurs_in_query(
        record.normalized_text,
        query_normalized,
    ) or bool(
        query_no_diac
        and _normalized_phrase_occurs_in_query(
            record.normalized_no_diacritics,
            query_no_diac,
        )
    )
    exact_tokens = len(set(record.tokens).intersection(query_tokens))
    if query_tokens_no_diac:
        exact_tokens = max(
            exact_tokens,
            len(set(record.tokens_no_diacritics).intersection(query_tokens_no_diac)),
        )
    return (
        record.normalized_text == query_normalized,
        phrase_match,
        exact_tokens,
        -record.position,
        # Deterministic tiebreak: candidate records come from sets, so without
        # this the nomination cut depends on hash-randomized iteration order.
        record.normalized_text,
    )


def _compact_slot_text(text: str | None) -> str:
    """Return a compact lookup key for multi-token or compound slot matching."""
    if not text:
        return ""
    compact = text.replace(" ", "")
    return "" if len(compact) < _COMPACT_SLOT_TOKEN_MIN_LENGTH else compact


def _single_character_deletions(token: str) -> frozenset[str]:
    """Return unique strings formed by deleting one character from *token*."""
    return frozenset(token[:index] + token[index + 1 :] for index in range(len(token)))


def _is_bounded_registry_token_match(value_token: str, query_token: str) -> bool:
    """Return whether registry and query tokens differ by one bounded edit."""
    value_len = len(value_token)
    query_len = len(query_token)
    if (
        not value_token
        or not query_token
        or value_token == query_token
        or value_token[0] != query_token[0]
        or min(value_len, query_len) < SLOT_FUZZY_TOKEN_MIN_LENGTH
        or abs(value_len - query_len) > 1
    ):
        return False
    if value_len != query_len:
        longer = value_token if value_len > query_len else query_token
        shorter = query_token if value_len > query_len else value_token
        return shorter in _single_character_deletions(longer)

    differences = [
        index
        for index, (value_char, query_char) in enumerate(zip(value_token, query_token, strict=True))
        if value_char != query_char
    ]
    if len(differences) == 1:
        return True
    return (
        len(differences) == 2
        and differences[1] == differences[0] + 1
        and value_token[differences[0]] == query_token[differences[1]]
        and value_token[differences[1]] == query_token[differences[0]]
    )


def _bounded_registry_token_match_count(
    value_tokens: tuple[str, ...],
    query_tokens: frozenset[str],
) -> int:
    """Return maximum one-to-one exact or bounded fuzzy token matches."""
    token_matches = tuple(
        frozenset(
            query_token
            for query_token in query_tokens
            if value_token == query_token
            or _is_bounded_registry_token_match(value_token, query_token)
        )
        for value_token in value_tokens
    )
    query_token_owner: dict[str, int] = {}

    def assign(value_index: int, visited: set[str]) -> bool:
        for query_token in token_matches[value_index]:
            if query_token in visited:
                continue
            visited.add(query_token)
            previous_owner = query_token_owner.get(query_token)
            if previous_owner is None or assign(previous_owner, visited):
                query_token_owner[query_token] = value_index
                return True
        return False

    return sum(
        assign(value_index, set())
        for value_index in sorted(
            range(len(token_matches)), key=lambda index: len(token_matches[index])
        )
        if token_matches[value_index]
    )


@lru_cache(maxsize=2048)
def _compound_query_tokens(query_normalized: str) -> frozenset[str]:
    """Return contiguous compact query token spans for compound slot lookup."""
    tokens = query_normalized.split()
    compounds: set[str] = set()
    for start in range(len(tokens)):
        max_end = min(len(tokens), start + _MAX_COMPOUND_QUERY_TOKENS)
        for end in range(start + 2, max_end + 1):
            if compact := _compact_slot_text("".join(tokens[start:end])):
                compounds.add(compact)
    if len(tokens) == 1 and _uses_compact_non_latin_script(query_normalized):
        compounds.update(_compact_script_query_spans(query_normalized))
    return frozenset(compounds)


def _compact_script_query_spans(query_normalized: str) -> Iterable[str]:
    """Yield bounded character spans for languages that commonly omit spaces."""
    max_length = min(len(query_normalized), _COMPACT_SCRIPT_QUERY_SPAN_MAX_LENGTH)
    for span_length in range(_COMPACT_SCRIPT_QUERY_SPAN_MIN_LENGTH, max_length + 1):
        for start in range(len(query_normalized) - span_length + 1):
            yield query_normalized[start : start + span_length]


@lru_cache(maxsize=2048)
def _compact_script_phrase_fallback_enabled(query_normalized: str) -> bool:
    """Return whether literal phrase fallback can find compact-script text.

    Segmentation spaces from speech-to-text do not disable the fallback:
    ``_uses_compact_non_latin_script`` already ignores whitespace and ASCII.
    """
    return bool(query_normalized and _uses_compact_non_latin_script(query_normalized))


@lru_cache(maxsize=65536)
def _uses_compact_non_latin_script(text: str) -> bool:
    """Return whether text contains a non-Latin script that often omits spaces."""
    if not text or text.isascii():
        return False
    found_compact_character = False
    for char in text:
        if char.isspace() or char.isascii():
            continue
        if unicodedata.category(char)[0] not in {"L", "M", "N"}:
            continue
        name = unicodedata.name(char, "")
        if not name.startswith(_COMPACT_SCRIPT_NAME_PREFIXES):
            return False
        found_compact_character = True
    return found_compact_character


def _candidate_metadata(
    source_key: str,
    sentence: str,
    expansion_rules: Mapping[str, str],
    *,
    literal_text: str | None = None,
    literal_variants: tuple[frozenset[str], ...] | None = None,
) -> dict[str, str]:
    """Return metadata for ranking candidates by localized template literals."""
    metadata = {"intent_source": source_key, "sentence_template": sentence}
    if literal_text is None or literal_variants is None:
        computed_text, computed_variants = _template_literals(sentence, expansion_rules)
        literal_text = computed_text if literal_text is None else literal_text
        literal_variants = computed_variants if literal_variants is None else literal_variants
    if literal_text:
        metadata["literal_text"] = literal_text
    if literal_variants:
        metadata["literal_variants"] = orjson.dumps([sorted(v) for v in literal_variants]).decode(
            "utf-8"
        )
    return metadata


def _slot_binding_output_value(
    binding: _SlotBinding,
    slot_output_values: Mapping[str, Mapping[str, Any]] | None,
) -> Any:
    """Return the resolved output value for one slot binding."""
    if slot_output_values is None:
        return binding.raw_value
    return slot_output_values.get(binding.list_name, {}).get(
        binding.raw_value,
        binding.raw_value,
    )


def _slots_from_bindings(
    slot_bindings: Sequence[_SlotBinding],
    slot_output_values: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return slot mappings, or reject bindings that conflict by output name."""
    slots: dict[str, Any] = {}
    raw_slots: dict[str, Any] = {}
    for binding in slot_bindings:
        output_value = _slot_binding_output_value(binding, slot_output_values)
        if binding.output_name in slots:
            if (
                slots[binding.output_name] != output_value
                or raw_slots[binding.output_name] != binding.raw_value
            ):
                return None
            continue
        slots[binding.output_name] = output_value
        raw_slots[binding.output_name] = binding.raw_value
    return slots, raw_slots


def _expansion_tag_context(
    sentence: str,
    slot_values: Mapping[str, tuple[str, ...]],
    expansion_rules: Mapping[str, str],
) -> _ExpansionTagContext:
    """Return compact tag decoders with a marker absent from rendered inputs."""
    used_chars = set(sentence)
    for rule in expansion_rules.values():
        used_chars.update(rule)
    for slot_name, values in slot_values.items():
        used_chars.update(slot_name)
        for value in values:
            used_chars.update(value)
    for codepoint in _EXPANSION_MARKER_CODEPOINTS:
        marker = chr(codepoint)
        if marker not in used_chars:
            return _cached_expansion_tag_context(marker)
    raise ValueError("No collision-free expansion marker is available")


@lru_cache(maxsize=128)
def _cached_expansion_tag_context(marker: str) -> _ExpansionTagContext:
    """Return cached binding and wildcard decoders for one marker."""
    escaped_marker = re.escape(marker)
    return _ExpansionTagContext(
        marker=marker,
        binding_pattern=re.compile(rf"{escaped_marker}b:(\d+):(\d+){escaped_marker}"),
        wildcard_pattern=re.compile(
            rf"{escaped_marker}w:(.+?){escaped_marker}",
            re.DOTALL,
        ),
    )


def _tagged_slot_values(
    slot_values: Mapping[str, tuple[str, ...]],
    slot_references: Sequence[TemplateSlotReference],
    wildcard_slots: frozenset[str],
    tag_context: _ExpansionTagContext,
) -> tuple[dict[str, tuple[str, ...]], dict[str, _SlotBinding]]:
    """Tag each slot choice so rendered text retains its exact binding."""
    tagged_values = dict(slot_values)
    bindings_by_tag: dict[str, _SlotBinding] = {}
    tagged_reference_names: set[str] = set()
    for reference_index, reference in enumerate(slot_references):
        if reference.raw_name in tagged_reference_names:
            continue
        tagged_reference_names.add(reference.raw_name)
        values = slot_values.get(reference.list_name)
        if not values:
            continue
        is_wildcard = reference.list_name in wildcard_slots
        tagged_reference_values: list[str] = []
        for value_index, raw_value in enumerate(dict.fromkeys(values)):
            binding_tag = (
                f"{tag_context.marker}b:{reference_index}:{value_index}{tag_context.marker}"
            )
            binding_value = reference.list_name if is_wildcard else raw_value
            bindings_by_tag[binding_tag] = _SlotBinding(
                list_name=reference.list_name,
                output_name=reference.output_name,
                raw_value=binding_value,
            )
            rendered_value = (
                f"{tag_context.marker}w:{reference.list_name}{tag_context.marker}"
                if is_wildcard
                else raw_value
            )
            tagged_reference_values.append(f"{binding_tag}{rendered_value}")
        tagged_values[reference.raw_name] = tuple(tagged_reference_values)
    return tagged_values, bindings_by_tag


def _decode_candidate_expansion(
    tagged_text: str,
    bindings_by_tag: Mapping[str, _SlotBinding],
    tag_context: _ExpansionTagContext,
) -> _CandidateExpansion:
    """Strip this expansion's collision-free tags and return its bindings."""
    selected_by_tag: dict[str, tuple[int, int, _SlotBinding]] = {}

    def _strip_known_tag(match: re.Match[str]) -> str:
        tag = match.group(0)
        binding = bindings_by_tag.get(tag)
        if binding is None:
            raise RuntimeError(f"Missing binding for injected slot tag: {tag!r}")
        selected_by_tag.setdefault(
            tag,
            (int(match.group(1)), int(match.group(2)), binding),
        )
        return ""

    text_without_bindings = tag_context.binding_pattern.sub(_strip_known_tag, tagged_text)
    selected_bindings = sorted(
        selected_by_tag.values(),
        key=lambda selected: (selected[0], selected[1]),
    )
    clean_text, wildcard_infos = _clean_wildcard_tokens(
        text_without_bindings,
        tag_context.wildcard_pattern,
    )
    return _CandidateExpansion(
        text=clean_text,
        slot_bindings=tuple(selected[2] for selected in selected_bindings),
        wildcard_infos=tuple(wildcard_infos),
    )


def _binding_tags_are_consistent(
    tagged_text: str,
    bindings_by_tag: Mapping[str, _SlotBinding],
    tag_context: _ExpansionTagContext,
    slot_output_values: Mapping[str, Mapping[str, Any]] | None,
    repeated_output_names: frozenset[str],
) -> bool:
    """Return whether repeated output slots select the same values."""
    output_values_by_output: dict[str, Any] = {}
    raw_values_by_output: dict[str, str] = {}
    for match in tag_context.binding_pattern.finditer(tagged_text):
        tag = match.group(0)
        binding = bindings_by_tag.get(tag)
        if binding is None:
            raise RuntimeError(f"Missing binding for injected slot tag: {tag!r}")
        if binding.output_name not in repeated_output_names:
            continue
        output_value = _slot_binding_output_value(binding, slot_output_values)
        if binding.output_name in raw_values_by_output:
            if (
                output_values_by_output[binding.output_name] != output_value
                or raw_values_by_output[binding.output_name] != binding.raw_value
            ):
                return False
            continue
        output_values_by_output[binding.output_name] = output_value
        raw_values_by_output[binding.output_name] = binding.raw_value
    return True


def _binding_consistency_filter(
    bindings_by_tag: Mapping[str, _SlotBinding],
    tag_context: _ExpansionTagContext,
    slot_output_values: Mapping[str, Mapping[str, Any]] | None,
    repeated_output_names: frozenset[str],
) -> Callable[[str], bool] | None:
    """Return a tagged-expansion filter for repeated output slots."""
    if not repeated_output_names:
        return None

    def binding_tags_are_consistent(tagged_text: str) -> bool:
        return _binding_tags_are_consistent(
            tagged_text,
            bindings_by_tag,
            tag_context,
            slot_output_values,
            repeated_output_names,
        )

    return binding_tags_are_consistent


def _expand_candidate_template(
    sentence: str,
    slot_values: Mapping[str, tuple[str, ...]],
    expansion_rules: Mapping[str, str],
    slot_references: Sequence[TemplateSlotReference],
    wildcard_slots: frozenset[str],
    *,
    max_expansions: int,
    fair: bool = False,
    slot_output_values: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[_CandidateExpansion, ...]:
    """Expand a template while preserving exact slot bindings."""
    if is_fixed_sentence(sentence):
        return (_CandidateExpansion(_clean_expanded_text(sentence)),)
    if not slot_references:
        return tuple(
            _CandidateExpansion(text)
            for text in expand_sentence_template(
                sentence,
                slot_values,
                expansion_rules,
                max_expansions=max_expansions,
                fair=fair,
            )
        )
    tag_context = _expansion_tag_context(sentence, slot_values, expansion_rules)
    tagged_values, bindings_by_tag = _tagged_slot_values(
        slot_values,
        slot_references,
        wildcard_slots,
        tag_context,
    )
    repeated_output_names = _template_repeated_output_names(sentence, expansion_rules)
    tagged_expansions = expand_sentence_template(
        sentence,
        tagged_values,
        expansion_rules,
        max_expansions=max_expansions,
        fair=fair,
        expansion_filter=_binding_consistency_filter(
            bindings_by_tag,
            tag_context,
            slot_output_values,
            repeated_output_names,
        ),
    )
    unique: dict[
        tuple[str, tuple[_SlotBinding, ...], tuple[tuple[int, str], ...]],
        _CandidateExpansion,
    ] = {}
    for tagged_text in tagged_expansions:
        expansion = _decode_candidate_expansion(tagged_text, bindings_by_tag, tag_context)
        unique.setdefault(
            (expansion.text, expansion.slot_bindings, expansion.wildcard_infos),
            expansion,
        )
    return tuple(unique.values())


def _candidate_expansions(
    sentence: str,
    slot_values: Mapping[str, tuple[str, ...]],
    expansion_rules: Mapping[str, str],
    slot_references: Sequence[TemplateSlotReference],
    wildcard_slots: frozenset[str],
    *,
    slot_output_values: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[_CandidateExpansion, ...]:
    """Return bounded candidate expansions for a sentence template.

    Expansion is deliberately not ``fair``: interleaving alternation branches
    spreads the per-template budget across verb forms at the cost of entity
    coverage per form, which measurably hurts matching; missing surface forms
    are recovered by the query-scoped dynamic registry path instead.
    """
    return _expand_candidate_template(
        sentence,
        slot_values,
        expansion_rules,
        slot_references,
        wildcard_slots,
        max_expansions=DEFAULT_MAX_CANDIDATES_PER_TEMPLATE,
        slot_output_values=slot_output_values,
    )


def _query_candidate_expansions(
    sentence: str,
    slot_values: Mapping[str, tuple[str, ...]],
    expansion_rules: Mapping[str, str],
    slot_references: Sequence[TemplateSlotReference],
    wildcard_slots: frozenset[str],
    query_normalized: str,
    query_tokens: frozenset[str],
    *,
    slot_output_values: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[tuple[_CandidateExpansion, str], ...]:
    """Return query-relevant candidate expansions for a dynamic template."""
    expanded = _expand_candidate_template(
        sentence,
        slot_values,
        expansion_rules,
        slot_references,
        wildcard_slots,
        max_expansions=max(
            DEFAULT_MAX_CANDIDATES_PER_TEMPLATE,
            _TEMPLATE_RELEVANCE_EXPANSION_LIMIT,
        ),
        fair=True,
        slot_output_values=slot_output_values,
    )
    normalized_expanded = tuple(
        (expansion, normalize_text(expansion.text)) for expansion in expanded
    )
    return tuple(
        sorted(
            normalized_expanded,
            key=lambda item: _text_relevance_key(item[1], query_normalized, query_tokens),
            reverse=True,
        )[:DEFAULT_MAX_CANDIDATES_PER_TEMPLATE]
    )


def _registry_slot_values_for_slots(
    registry_slot_values: Mapping[str, tuple[str, ...]],
    slot_names: Iterable[str],
    *,
    domains: tuple[str, ...] = (),
) -> dict[str, tuple[str, ...]]:
    """Return registry values keyed only by the requested template slot names."""
    requested = tuple(dict.fromkeys(slot_names))
    selected = {}
    for slot_name in requested:
        if values := registry_slot_values.get(slot_name):
            selected[slot_name] = values
    for slot_name in requested:
        if slot_name not in ENTITY_SLOT_NAME_SET:
            continue
        if scoped_values := _scoped_slot_values(slot_name, domains, registry_slot_values):
            selected[slot_name] = scoped_values
    return selected


def _registry_slot_values_for_template(
    data_item: Mapping[str, Any],
    *,
    sentence_slots: frozenset[str],
    base_data_slot_values: Mapping[str, tuple[str, ...]],
    registry_slot_values: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Return registry slots constrained by HassIL context and template shape."""
    registry_slots = sentence_slots - set(base_data_slot_values)
    domains = _context_domains(data_item)
    constrained = _registry_slot_values_for_slots(
        registry_slot_values,
        sorted(registry_slots),
        domains=domains,
    )
    registry_entity_slots = registry_slots & ENTITY_SLOT_NAME_SET
    if registry_entity_slots and sentence_slots & LOCATION_SLOT_NAME_SET:
        for slot_name in LOCATION_SLOT_NAMES:
            constrained.pop(slot_name, None)

    if not domains:
        return constrained
    for slot_name in registry_entity_slots:
        constrained.pop(slot_name, None)
        if scoped_values := _scoped_slot_values(slot_name, domains, registry_slot_values):
            constrained[slot_name] = scoped_values
    return constrained


def _is_domain_scoped_location_template(
    sentence_slots: frozenset[str],
    domains: tuple[str, ...],
) -> bool:
    """Return whether a location-only template is constrained to one or more domains."""
    return bool(domains and sentence_slots & LOCATION_SLOT_NAME_SET)


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
    allow_fuzzy_tokens: bool = True,
    require_whole_query: bool = False,
) -> tuple[bool, bool, bool, int, int] | None:
    """Return registry relevance using precomputed normalized value tokens."""
    if not value_tokens:
        return None
    token_count = len(value_tokens)

    exact = value_normalized == query_normalized
    if not exact and value_no_diac and query_no_diac:
        exact = value_no_diac == query_no_diac
    if exact:
        return (True, True, True, token_count, -token_count)

    if require_whole_query:
        return _whole_query_match_key(
            value_normalized,
            query_normalized,
            value_no_diac,
            query_no_diac,
            token_count,
            allow_fuzzy_tokens=allow_fuzzy_tokens,
        )

    if _phrase_or_compact_value_match(
        value_normalized,
        query_normalized,
        query_tokens,
        value_no_diac,
        query_no_diac,
        query_tokens_no_diac,
    ):
        return (False, True, True, token_count, -token_count)

    return _token_overlap_match_key(
        value_tokens,
        query_tokens,
        value_tokens_no_diac,
        query_tokens_no_diac,
        allow_fuzzy_tokens=allow_fuzzy_tokens,
    )


def _whole_query_match_key(
    value_normalized: str,
    query_normalized: str,
    value_no_diac: str | None,
    query_no_diac: str | None,
    token_count: int,
    *,
    allow_fuzzy_tokens: bool,
) -> tuple[bool, bool, bool, int, int] | None:
    """Match a value against the whole query using compact exact and fuzzy forms."""
    value_compact = value_normalized.replace(" ", "")
    query_compact = query_normalized.replace(" ", "")
    compact_exact = bool(value_compact and value_compact == query_compact)
    if not compact_exact and value_no_diac and query_no_diac:
        compact_exact = value_no_diac.replace(" ", "") == query_no_diac.replace(" ", "")
    if compact_exact:
        return (False, True, True, token_count, -token_count)

    compact_fuzzy = (
        allow_fuzzy_tokens
        and min(len(value_compact), len(query_compact)) >= REGISTRY_FUZZY_TOKEN_MIN_LENGTH
        and _is_bounded_registry_token_match(
            value_compact,
            query_compact,
        )
    )
    if not compact_fuzzy and allow_fuzzy_tokens and value_no_diac and query_no_diac:
        value_no_diac_compact = value_no_diac.replace(" ", "")
        query_no_diac_compact = query_no_diac.replace(" ", "")
        compact_fuzzy = min(
            len(value_no_diac_compact), len(query_no_diac_compact)
        ) >= REGISTRY_FUZZY_TOKEN_MIN_LENGTH and _is_bounded_registry_token_match(
            value_no_diac_compact,
            query_no_diac_compact,
        )
    if compact_fuzzy:
        return (False, False, True, token_count, -token_count)
    return None


def _phrase_or_compact_value_match(
    value_normalized: str,
    query_normalized: str,
    query_tokens: frozenset[str],
    value_no_diac: str | None,
    query_no_diac: str | None,
    query_tokens_no_diac: frozenset[str] | None,
) -> bool:
    """Match token-boundary phrases, then compound/spacing-insensitive phrases."""
    phrase_match = _normalized_phrase_occurs_in_query(value_normalized, query_normalized)
    if not phrase_match and value_no_diac and query_no_diac:
        phrase_match = _normalized_phrase_occurs_in_query(value_no_diac, query_no_diac)
    if phrase_match:
        return True

    value_compact = _compact_slot_text(value_normalized)
    compact_match = bool(value_compact and value_compact in query_tokens)
    if (
        not compact_match
        and value_no_diac
        and query_tokens_no_diac
        and (value_no_diac_compact := _compact_slot_text(value_no_diac))
    ):
        compact_match = value_no_diac_compact in query_tokens_no_diac
    return compact_match


def _token_overlap_match_key(
    value_tokens: tuple[str, ...],
    query_tokens: frozenset[str],
    value_tokens_no_diac: tuple[str, ...] | None,
    query_tokens_no_diac: frozenset[str] | None,
    *,
    allow_fuzzy_tokens: bool,
) -> tuple[bool, bool, bool, int, int] | None:
    """Score exact and bounded-fuzzy token overlaps between value and query."""
    token_count = len(value_tokens)
    exact_matched = len(query_tokens.intersection(value_tokens))
    matched = exact_matched
    if allow_fuzzy_tokens:
        matched = max(matched, _bounded_registry_token_match_count(value_tokens, query_tokens))
    if value_tokens_no_diac and query_tokens_no_diac:
        matched_no_diac = len(query_tokens_no_diac.intersection(value_tokens_no_diac))
        exact_matched = max(exact_matched, matched_no_diac)
        if allow_fuzzy_tokens:
            matched_no_diac = max(
                matched_no_diac,
                _bounded_registry_token_match_count(value_tokens_no_diac, query_tokens_no_diac),
            )
        matched = max(matched, matched_no_diac)

    if (
        token_count == 1
        and exact_matched == 0
        and len(value_tokens[0]) < REGISTRY_FUZZY_TOKEN_MIN_LENGTH
    ):
        return None

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
    literal_text: str | None = None,
    query_normalized: str = "",
    query_tokens_no_diac: frozenset[str] | None = None,
    query_no_diac: str | None = None,
    language: str | None = None,
) -> bool:
    """Return whether precomputed template literal variants match a query."""
    if not literal_variants:
        return True
    for literal_tokens in literal_variants:
        if not literal_tokens:
            return True
        if literal_tokens.isdisjoint(query_tokens):
            continue
        if literal_tokens.issubset(query_tokens):
            return True
        matched = len(literal_tokens & query_tokens)
        if matched > 0 and matched / len(literal_tokens) >= 0.5:
            return True
    if query_tokens_no_diac:
        for literal_tokens_no_diac in literal_variants_no_diac:
            if literal_tokens_no_diac.isdisjoint(query_tokens_no_diac):
                continue
            if literal_tokens_no_diac.issubset(query_tokens_no_diac):
                return True
            matched_no_diac = len(literal_tokens_no_diac & query_tokens_no_diac)
            if matched_no_diac > 0 and matched_no_diac / len(literal_tokens_no_diac) >= 0.5:
                return True
    return bool(
        literal_text
        and query_normalized
        and _compact_script_phrase_fallback_enabled(query_normalized)
        and _literal_text_phrase_matches_query(
            literal_text,
            query_normalized,
            query_no_diac,
            language,
        )
    )


def _template_literal_text(
    text: str,
    expansion_rules: Mapping[str, str],
) -> str:
    """Return localized literal words from a HassIL sentence template."""
    literal_text, _ = _template_literals(text, expansion_rules)
    return literal_text


def _template_literal_token_variants(
    text: str,
    expansion_rules: Mapping[str, str],
) -> tuple[frozenset[str], ...]:
    """Return normalized literal token variants for query matching."""
    _, token_variants = _template_literals(text, expansion_rules)
    return token_variants


def _template_literals(
    text: str,
    expansion_rules: Mapping[str, str],
) -> tuple[str, tuple[frozenset[str], ...]]:
    """Return literal text and normalized token variants for query matching."""
    return _cached_template_literals(text, _expansion_rules_cache_key(expansion_rules))


def _expansion_rules_cache_key(
    expansion_rules: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    """Return a stable cache key for expansion rules."""
    return tuple(sorted(expansion_rules.items()))


@lru_cache(maxsize=65536)
def _cached_normalize_no_diac(text: str, language: str | None = None) -> str:
    """Cache diacritics removal for efficient lookups."""
    return normalize_text_no_diacritics(text, language)


@lru_cache(maxsize=2048)
def _cached_template_literals(
    text: str,
    expansion_rules_key: tuple[tuple[str, str], ...],
) -> tuple[str, tuple[frozenset[str], ...]]:
    """Return cached literal text and token variants from one template traversal."""
    if all(marker not in text for marker in _TEMPLATE_MARKERS):
        literal_text = _clean_expanded_text(text)
        if not literal_text:
            return "", (frozenset(),)
        if tokens := frozenset(normalize_text(literal_text).split()):
            return literal_text, (tokens,)
        return literal_text, ()

    expansion_rules = dict(expansion_rules_key)
    top_node = _parse_hassil(text)
    variants = top_node.literal_variants(
        expansion_rules,
        frozenset(),
        max(
            DEFAULT_MAX_CANDIDATES_PER_TEMPLATE,
            _TEMPLATE_RELEVANCE_EXPANSION_LIMIT,
        ),
    )
    literal_texts: dict[str, None] = {}
    token_variants: dict[frozenset[str], None] = {}
    for variant in variants:
        literal_text = _clean_expanded_text(variant)
        if not literal_text:
            token_variants[frozenset()] = None
            continue
        literal_texts[literal_text] = None
        if tokens := frozenset(normalize_text(literal_text).split()):
            token_variants[tokens] = None
    return "|".join(literal_texts), tuple(token_variants)


def _unique_capped_with_empty(items: Iterable[str], limit: int) -> tuple[str, ...]:
    """Return unique items from items, prepended with an empty string, capped at limit."""
    unique: dict[str, None] = {"": None}
    for item in items:
        if item not in unique:
            unique[item] = None
            if len(unique) >= limit:
                break
    return tuple(unique.keys())


def _interleave_unique_capped(groups: Sequence[Sequence[str]], limit: int) -> tuple[str, ...]:
    """Return unique items from each group without starving later groups."""
    if limit < 1:
        return ()
    max_len = max((len(group) for group in groups), default=0)
    unique: dict[str, None] = {}
    for offset in range(max_len):
        for group in groups:
            if offset >= len(group):
                continue
            item = group[offset]
            if item in unique:
                continue
            unique[item] = None
            if len(unique) >= limit:
                return tuple(unique.keys())
    return tuple(unique.keys())


class _HassilNode:
    """Base class for HassIL template AST nodes."""

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
        *,
        fair: bool = False,
        expansion_filter: Callable[[str], bool] | None = None,
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
        *,
        fair: bool = False,
        expansion_filter: Callable[[str], bool] | None = None,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        return (self.text,)


class _HassilSlotNode(_HassilNode):
    """AST node representing a template slot."""

    def __init__(self, name: str):
        """Initialize SlotNode.

        Component whitespace is stripped so ``{name : area}`` resolves the
        same slot keys as ``{name:area}``, matching ``_slot_reference``.
        """
        self.name = ":".join(part.strip() for part in name.split(":"))

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
        *,
        fair: bool = False,
        expansion_filter: Callable[[str], bool] | None = None,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        base_name = self.name.split(":")[0]
        values = slot_values.get(self.name) or slot_values.get(base_name)
        return tuple(values[:limit]) if values else ()


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
        *,
        fair: bool = False,
        expansion_filter: Callable[[str], bool] | None = None,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        if self.name in seen:
            return ("",)
        rule_text = rules.get(self.name)
        if rule_text is None:
            # Mirror literal_variants: an unknown rule contributes empty text
            # instead of silently zeroing out the whole sentence product.
            _warn_unknown_expansion_rule(self.name)
            return ("",)
        node = _parse_hassil(rule_text)
        return node.expand(
            slot_values,
            rules,
            seen | {self.name},
            limit,
            fair=fair,
            expansion_filter=expansion_filter,
        )


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
        return _unique_capped_with_empty(self.child.literal_variants(rules, seen, limit), limit)

    def expand(
        self,
        slot_values: Mapping[str, tuple[str, ...]],
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
        *,
        fair: bool = False,
        expansion_filter: Callable[[str], bool] | None = None,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        return _unique_capped_with_empty(
            self.child.expand(
                slot_values,
                rules,
                seen,
                limit,
                fair=fair,
                expansion_filter=expansion_filter,
            ),
            limit,
        )


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
        branch_variants = [branch.literal_variants(rules, seen, limit) for branch in self.branches]
        return _interleave_unique_capped(branch_variants, limit)

    def expand(
        self,
        slot_values: Mapping[str, tuple[str, ...]],
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
        *,
        fair: bool = False,
        expansion_filter: Callable[[str], bool] | None = None,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        if fair:
            branch_expansions = [
                branch.expand(
                    slot_values,
                    rules,
                    seen,
                    limit,
                    fair=True,
                    expansion_filter=expansion_filter,
                )
                for branch in self.branches
            ]
            return _interleave_unique_capped(branch_expansions, limit)
        unique_expansions: dict[str, None] = {}
        for branch in self.branches:
            remaining = limit - len(unique_expansions)
            if remaining < 1:
                break
            for val in branch.expand(
                slot_values,
                rules,
                seen,
                remaining,
                fair=False,
                expansion_filter=expansion_filter,
            ):
                if val not in unique_expansions:
                    unique_expansions[val] = None
                    if len(unique_expansions) >= limit:
                        break
        return tuple(unique_expansions.keys())


class _HassilPermutationNode(_HassilNode):
    """AST node representing a HassIL permutation group."""

    def __init__(self, branches: list[_HassilNode]):
        """Initialize PermutationNode."""
        self.branches = branches

    def required_slots(self, rules: Mapping[str, str], seen: frozenset[str]) -> set[str]:
        """Return slot names required by this node."""
        required = set()
        for branch in self.branches:
            required.update(branch.required_slots(rules, seen))
        return required

    def literal_variants(
        self,
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        """Return unique literal word variants for this node."""
        return self._permuted_texts(
            [branch.literal_variants(rules, seen, limit) for branch in self.branches],
            limit,
        )

    def expand(
        self,
        slot_values: Mapping[str, tuple[str, ...]],
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
        *,
        fair: bool = False,
        expansion_filter: Callable[[str], bool] | None = None,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        return self._permuted_texts(
            [
                branch.expand(
                    slot_values,
                    rules,
                    seen,
                    limit,
                    fair=fair,
                    expansion_filter=expansion_filter,
                )
                for branch in self.branches
            ],
            limit,
            expansion_filter=expansion_filter,
        )

    @staticmethod
    def _permuted_texts(
        branch_fragments: list[tuple[str, ...]],
        limit: int,
        *,
        expansion_filter: Callable[[str], bool] | None = None,
    ) -> tuple[str, ...]:
        """Return capped text variants for every branch order."""
        if not branch_fragments:
            return ("",)
        unique: dict[str, None] = {}
        for ordered_fragments in permutations(branch_fragments):
            for fragments in product(*ordered_fragments):
                text = " ".join(fragment.strip() for fragment in fragments if fragment.strip())
                if expansion_filter is not None and not expansion_filter(text):
                    continue
                if text not in unique:
                    unique[text] = None
                    if len(unique) >= limit:
                        return tuple(unique.keys())
        return tuple(unique.keys())


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
        *,
        fair: bool,
        expansion_filter: Callable[[str], bool] | None = None,
    ) -> tuple[str, ...]:
        """Cartesian-product accumulation of child fragments with deduplication."""
        if not child_fragments:
            return ("",)
        current = ("",)
        for fragments in child_fragments:
            next_results: dict[str, None] = {}
            outer_values, inner_values = (fragments, current) if fair else (current, fragments)
            for outer in outer_values:
                for inner in inner_values:
                    prefix, fragment = (inner, outer) if fair else (outer, inner)
                    combined = f"{prefix}{fragment}"
                    if expansion_filter is not None and not expansion_filter(combined):
                        continue
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
        return self._accumulate(child_fragments, limit, fair=True)

    def expand(
        self,
        slot_values: Mapping[str, tuple[str, ...]],
        rules: Mapping[str, str],
        seen: frozenset[str],
        limit: int,
        *,
        fair: bool = False,
        expansion_filter: Callable[[str], bool] | None = None,
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        if not self.children:
            return ("",)
        child_fragments = [
            child.expand(
                slot_values,
                rules,
                seen,
                limit,
                fair=fair,
                expansion_filter=expansion_filter,
            )
            for child in self.children
        ]
        return self._accumulate(
            child_fragments,
            limit,
            fair=fair,
            expansion_filter=expansion_filter,
        )


def _merge_slot_output_occurrences(
    nodes: Iterable[_HassilNode],
    rules: Mapping[str, str],
    seen_rules: frozenset[str],
    *,
    alternatives: bool,
) -> dict[str, int]:
    """Merge maximum output-slot occurrences across child nodes."""
    merged: dict[str, int] = {}
    for node in nodes:
        for output_name, count in _slot_output_occurrences(node, rules, seen_rules).items():
            if alternatives:
                merged[output_name] = max(merged.get(output_name, 0), count)
            else:
                merged[output_name] = min(2, merged.get(output_name, 0) + count)
    return merged


def _slot_output_occurrences(
    node: _HassilNode,
    rules: Mapping[str, str],
    seen_rules: frozenset[str],
) -> dict[str, int]:
    """Return maximum output-slot occurrences on any path through one node."""
    if isinstance(node, _HassilSlotNode):
        output_name = _slot_reference(node.name).output_name
        return {output_name: 1} if output_name else {}
    if isinstance(node, _HassilRuleNode):
        if node.name in seen_rules or (rule_text := rules.get(node.name)) is None:
            return {}
        return _slot_output_occurrences(
            _parse_hassil(rule_text),
            rules,
            seen_rules | {node.name},
        )
    if isinstance(node, _HassilOptionalNode):
        return _slot_output_occurrences(node.child, rules, seen_rules)
    if isinstance(node, _HassilAlternativeNode):
        return _merge_slot_output_occurrences(
            node.branches,
            rules,
            seen_rules,
            alternatives=True,
        )
    if isinstance(node, _HassilPermutationNode):
        return _merge_slot_output_occurrences(
            node.branches,
            rules,
            seen_rules,
            alternatives=False,
        )
    if isinstance(node, _HassilSequenceNode):
        return _merge_slot_output_occurrences(
            node.children,
            rules,
            seen_rules,
            alternatives=False,
        )
    return {}


def _make_branch_node(branch: list[_HassilNode]) -> _HassilNode:
    """Wrap a branch list into a single AST node."""
    return _HassilSequenceNode(list(branch)) if len(branch) != 1 else branch[0]


def _parse_delimited_node(
    text: str,
    i: int,
    close_char: str,
    node_factory: Callable[[str], _HassilNode],
) -> tuple[_HassilNode, int]:
    """Parse a {slot} or <rule> delimited token starting at position i."""
    end = text.find(close_char, i)
    if end == -1:
        return _HassilTextNode(text[i]), i + 1
    return node_factory(text[i + 1 : end].strip()), end + 1


def _parse_hassil_group(
    text: str,
    index: int,
    opener: str,
    depth: int,
) -> tuple[_HassilNode, int]:
    """Parse a grouped expression and wrap optional groups."""
    close_char = "]" if opener == "[" else ")"
    child, next_index = _parse_hassil_expr(
        text,
        index + 1,
        close_char,
        depth=depth + 1,
    )
    return (_HassilOptionalNode(child) if opener == "[" else child), next_index


def _start_hassil_branch(
    separator: str,
    current_branch: list[_HassilNode],
    branches: list[_HassilNode],
    active_separator: str | None,
) -> tuple[list[_HassilNode], str | None]:
    """Finish a branch or preserve a conflicting separator as literal text."""
    if active_separator is not None and active_separator != separator:
        current_branch.append(_HassilTextNode(separator))
        return current_branch, active_separator
    branches.append(_make_branch_node(current_branch))
    return [], separator


@lru_cache(maxsize=8192)
def _parse_hassil(text: str) -> _HassilNode:
    """Parse HassIL template string into an AST node, caching the results.

    Note on caching semantics:
    - Bounded to 8192 distinct templates to prevent unbounded memory growth.
    - Evicted nodes are garbage-collected only if not referenced by compiled intents.
    - Object identity is not used for AST comparisons, so eviction does not affect correctness.
    """
    node, _ = _parse_hassil_expr(text, 0, depth=0)
    return node


def _parse_hassil_expr(
    text: str,
    i: int,
    close_char: str | None = None,
    *,
    depth: int,
) -> tuple[_HassilNode, int]:
    """Parse a HassIL expression from a given index.

    Args:
        text: Complete HassIL template text.
        i: Index where parsing starts.
        close_char: Delimiter that ends this recursive expression.
        depth: Current grouping depth.

    Returns:
        Parsed node and the next unread text index.

    Raises:
        ValueError: If nested groups exceed the configured safety limit.
    """
    if depth >= _MAX_TEMPLATE_NESTING_DEPTH:
        raise ValueError(f"Max template nesting depth ({_MAX_TEMPLATE_NESTING_DEPTH}) exceeded.")
    current_branch: list[_HassilNode] = []
    branches: list[_HassilNode] = []
    branch_separator: str | None = None

    while i < len(text):
        char = text[i]
        match char:
            case "[" | "(":
                node, i = _parse_hassil_group(text, i, char, depth)
                current_branch.append(node)
            case "]" | ")":
                if char == close_char:
                    i += 1
                    break
                current_branch.append(_HassilTextNode(char))
                i += 1
            case "|" | ";":
                current_branch, branch_separator = _start_hassil_branch(
                    char,
                    current_branch,
                    branches,
                    branch_separator,
                )
                i += 1
            case "{":
                node, i = _parse_delimited_node(text, i, "}", _HassilSlotNode)
                current_branch.append(node)
            case "<":
                node, i = _parse_delimited_node(text, i, ">", _HassilRuleNode)
                current_branch.append(node)
            case _:
                start = i
                i += 1
                while i < len(text) and text[i] not in _TEXT_RUN_STOP_CHARS:
                    i += 1
                current_branch.append(_HassilTextNode(text[start:i]))

    branches.append(_make_branch_node(current_branch))
    if len(branches) == 1:
        return branches[0], i
    if branch_separator == ";":
        return _HassilPermutationNode(branches), i
    return _HassilAlternativeNode(branches), i


def _required_slots(
    text: str,
    expansion_rules: Mapping[str, str],
    seen_rules: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Return all slot names that are required in the template."""
    return _cached_required_slots(
        text,
        _expansion_rules_cache_key(expansion_rules),
        seen_rules,
    )


def _template_repeated_output_names(
    text: str,
    expansion_rules: Mapping[str, str],
) -> frozenset[str]:
    """Return cached output names that can occur repeatedly in one expansion."""
    return _cached_repeated_output_names(
        text,
        _expansion_rules_cache_key(expansion_rules),
    )


@lru_cache(maxsize=4096)
def _cached_repeated_output_names(
    text: str,
    expansion_rules_key: tuple[tuple[str, str], ...],
) -> frozenset[str]:
    """Return output names whose maximum path occurrence exceeds one."""
    occurrences = _slot_output_occurrences(
        _parse_hassil(text),
        dict(expansion_rules_key),
        frozenset(),
    )
    return frozenset(name for name, count in occurrences.items() if count > 1)


@lru_cache(maxsize=4096)
def _cached_required_slots(
    text: str,
    expansion_rules_key: tuple[tuple[str, str], ...],
    seen_rules: frozenset[str],
) -> frozenset[str]:
    """Return cached required slot names."""
    top_node = _parse_hassil(text)
    return frozenset(top_node.required_slots(dict(expansion_rules_key), seen_rules))


def _template_slot_names(
    text: str,
    expansion_rules: Mapping[str, str],
    seen_rules: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Return slot names referenced by a sentence template and its rules."""
    return frozenset(
        reference.list_name
        for reference in _template_slot_references(text, expansion_rules, seen_rules)
    )


def _template_slot_references(
    text: str,
    expansion_rules: Mapping[str, str],
    seen_rules: frozenset[str] = frozenset(),
) -> tuple[TemplateSlotReference, ...]:
    """Return slot references from a sentence template and its rules."""
    return _cached_template_slot_references(
        text,
        _expansion_rules_cache_key(expansion_rules),
        seen_rules,
    )


@lru_cache(maxsize=4096)
def _cached_template_slot_references(
    text: str,
    expansion_rules_key: tuple[tuple[str, str], ...],
    seen_rules: frozenset[str],
) -> tuple[TemplateSlotReference, ...]:
    """Return cached slot references from a sentence template and its rules."""
    references: list[TemplateSlotReference] = []
    seen_references: set[tuple[str, str]] = set()

    def add_reference(reference: TemplateSlotReference) -> None:
        if not reference.list_name:
            return
        key = (reference.list_name, reference.output_name)
        if key in seen_references:
            return
        seen_references.add(key)
        references.append(reference)

    for match in _SLOT_PATTERN.finditer(text):
        add_reference(_slot_reference(match.group(1)))
    expansion_rules = dict(expansion_rules_key)
    for rule_match in _RULE_PATTERN.finditer(text):
        rule_name = rule_match.group(1).strip()
        if rule_name in seen_rules:
            continue
        rule_text = expansion_rules.get(rule_name)
        if rule_text is not None:
            for reference in _cached_template_slot_references(
                rule_text,
                expansion_rules_key,
                seen_rules | frozenset({rule_name}),
            ):
                add_reference(reference)
    return tuple(references)


def _slot_reference(raw: str) -> TemplateSlotReference:
    """Parse a HassIL slot reference into expansion list and output slot names.

    ``raw_name`` strips whitespace around the colon so ``{name : area}``
    stores tagged values under the same key that ``_HassilSlotNode``
    (which normalizes component whitespace) later resolves.
    """
    raw_name = ":".join(part.strip() for part in raw.strip().split(":"))
    list_name, separator, output_name = raw_name.partition(":")
    list_name = list_name.strip()
    output_name = output_name.strip() if separator else list_name
    return TemplateSlotReference(
        list_name=list_name,
        output_name=output_name or list_name,
        raw_name=raw_name,
    )


def _context_domains(data_item: Mapping[str, Any]) -> tuple[str, ...]:
    """Return required entity domains for one HassIL data item."""
    domains: list[str] = []
    for key in ("requires_context", "slots"):
        context = data_item.get(key, {})
        if not isinstance(context, Mapping):
            continue
        domains.extend(_domain_values(context.get("domain")))
    return tuple(_deduplicate_texts(domains, len(domains) or 1))


def _context_slot_names(data_item: Mapping[str, Any]) -> frozenset[str]:
    """Return context keys that HassIL injects as slots."""
    requires_context = data_item.get("requires_context", {})
    if not isinstance(requires_context, Mapping):
        return frozenset()
    return frozenset(
        slot_name
        for slot_name, slot_config in requires_context.items()
        if isinstance(slot_name, str)
        and isinstance(slot_config, Mapping)
        and slot_config.get("slot") is True
    )


def _domain_values(value: Any) -> Iterable[str]:
    """Yield domain names from HassIL context values."""
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


def _effective_list_configs(
    source_config: Mapping[str, Any],
    intent_config: Mapping[str, Any],
    data_item: Mapping[str, Any],
) -> dict[str, Any]:
    """Return HassIL-effective list configs by list name.

    HassIL combines root/external lists with data-item lists by dictionary
    override, so a later same-named list replaces the whole earlier list.
    """
    list_configs: dict[str, Any] = {}
    for config in (source_config, intent_config, data_item):
        lists = config.get("lists", {})
        if not isinstance(lists, Mapping):
            continue
        for list_name, list_config in lists.items():
            if isinstance(list_name, str):
                list_configs[list_name] = list_config
    return list_configs


def _wildcard_slots(
    source_config: Mapping[str, Any],
    intent_config: Mapping[str, Any],
    data_item: Mapping[str, Any],
) -> frozenset[str]:
    """Return list names configured as wildcards in the effective grammar."""
    return frozenset(
        list_name
        for list_name, list_config in _effective_list_configs(
            source_config,
            intent_config,
            data_item,
        ).items()
        if isinstance(list_config, Mapping) and list_config.get("wildcard")
    )


def _range_boundaries(
    range_data: Mapping[str, Any],
) -> tuple[float, float, tuple[float, float]] | None:
    """Return original and ordered range boundaries when both are valid."""
    if "from" not in range_data or "to" not in range_data:
        return None
    try:
        from_value = float(range_data["from"])
        to_value = float(range_data["to"])
    except (ValueError, TypeError):
        return None
    lower, upper = sorted((from_value, to_value))
    return from_value, to_value, (lower, upper)


def _optional_range_number(
    range_data: Mapping[str, Any],
    key: str,
    *,
    positive: bool = False,
) -> float | None:
    """Return an optional numeric range setting when it is valid."""
    if key not in range_data:
        return None
    try:
        value = float(range_data[key])
    except (ValueError, TypeError):
        return None
    return value if not positive or value > 0 else None


def _range_fraction_type(range_data: Mapping[str, Any], list_name: str) -> str | None:
    """Return a supported fraction type and log invalid configured strings."""
    fraction_type = range_data.get("fractions")
    if not isinstance(fraction_type, str):
        return None
    if fraction_type in ("halves", "tenths"):
        return fraction_type
    _LOGGER.debug(
        "Invalid fraction type '%s' in range config for list '%s', ignoring.",
        fraction_type,
        list_name,
    )
    return None


def _compile_range_data(
    source_config: Mapping[str, Any],
    intent_config: Mapping[str, Any],
    data_item: Mapping[str, Any],
) -> tuple[dict[str, tuple[float, float]], dict[str, RangeStepMultiplier]]:
    """Compile range boundaries and step multipliers in a single pass."""
    ranges: dict[str, tuple[float, float]] = {}
    step_mult: dict[str, RangeStepMultiplier] = {}
    for list_name, list_config in _effective_list_configs(
        source_config,
        intent_config,
        data_item,
    ).items():
        if not isinstance(list_config, Mapping):
            continue

        range_data = list_config.get("range")
        if not isinstance(range_data, Mapping):
            continue
        boundaries = _range_boundaries(range_data)
        if boundaries is None:
            continue
        from_val, _to_val, (low, high) = boundaries
        ranges[list_name] = (low, high)
        step_mult[list_name] = RangeStepMultiplier(
            step=_optional_range_number(range_data, "step", positive=True),
            multiplier=_optional_range_number(range_data, "multiplier"),
            fraction_type=_range_fraction_type(range_data, list_name),
            original_from=from_val,
        )

    return ranges, step_mult


def _slot_values(
    source_config: Mapping[str, Any],
    intent_config: Mapping[str, Any],
    data_item: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Return available slot input values for template expansion."""
    values: dict[str, tuple[str, ...]] = {}
    for list_name, list_config in _effective_list_configs(
        source_config,
        intent_config,
        data_item,
    ).items():
        if extracted := tuple(_values_from_list_config(list_config, list_name)):
            values[list_name] = extracted
    slots = data_item.get("slots", {})
    if isinstance(slots, Mapping):
        for slot_name, slot_value in slots.items():
            if (
                isinstance(slot_name, str)
                and slot_name not in values
                and (extracted := tuple(_string_values(slot_value)))
            ):
                values[slot_name] = extracted
    return values


def _slot_output_value_maps(
    source_config: Mapping[str, Any],
    intent_config: Mapping[str, Any],
    data_item: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return spoken-input to output-value maps for text slot lists.

    When a list appears in multiple config layers, later layers replace the
    whole list, matching HassIL's effective slot-list precedence.
    """
    maps: dict[str, dict[str, Any]] = {}
    for list_name, list_config in _effective_list_configs(
        source_config,
        intent_config,
        data_item,
    ).items():
        if output_map := dict(_value_outputs_from_list_config(list_config, list_name)):
            maps[list_name] = output_map
    return maps


def _static_slot_values(data_item: Mapping[str, Any]) -> dict[str, str]:
    """Return static slots declared on a HassIL data item.

    HassIL permits numeric and boolean static slots; they are kept as their
    string form so slot metadata and ranking signals see them.
    """
    static_slots: dict[str, str] = {}
    slots = data_item.get("slots", {})
    if not isinstance(slots, Mapping):
        return static_slots
    for slot_name, slot_value in slots.items():
        if not isinstance(slot_name, str):
            continue
        if isinstance(slot_value, str):
            static_slots[slot_name] = slot_value
        elif isinstance(slot_value, bool | int | float):
            static_slots[slot_name] = str(slot_value)
    return static_slots


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


def _list_range_endpoints(range_data: Any) -> tuple[Any, Any]:
    """Return configured range endpoints or safe fallback values."""
    try:
        return range_data.get("from", 0), range_data.get("to", 100)
    except (AttributeError, ValueError, TypeError):
        return 0, 100


def _range_endpoint_output(value: Any, multiplier: float | None) -> Any:
    """Return an endpoint output scaled consistently with interior range values."""
    if multiplier is None:
        return value
    parsed = parse_float(str(value))
    return value if parsed is None else to_output_value(parsed * multiplier)


def _values_from_list_config(list_config: Any, list_name: str | None = None) -> Iterable[str]:
    """Yield spoken values from a HassIL list config."""
    if isinstance(list_config, Mapping):
        if "values" in list_config:
            values = list_config.get("values", [])
            if isinstance(values, list):
                for value in values:
                    for input_value, _ in _value_item_input_outputs(value):
                        yield input_value
            return

        if "range" in list_config:
            from_val, to_val = _list_range_endpoints(list_config["range"])
            yield str(from_val)
            if from_val != to_val:
                yield str(to_val)
            return

        if list_config.get("wildcard"):
            yield list_name or "wildcard"
            return

    if isinstance(list_config, list):
        for value in list_config:
            for input_value, _ in _value_item_input_outputs(value):
                yield input_value


def _value_outputs_from_list_config(
    list_config: Any,
    list_name: str | None = None,
) -> Iterable[tuple[str, Any]]:
    """Yield spoken input to output value pairs from a HassIL list config."""
    if isinstance(list_config, Mapping):
        if "values" in list_config:
            values = list_config.get("values", [])
            if isinstance(values, list):
                for value in values:
                    yield from _value_item_input_outputs(value)
            return

        if "range" in list_config:
            range_data = list_config["range"]
            from_val, to_val = _list_range_endpoints(range_data)
            multiplier = (
                _optional_range_number(range_data, "multiplier")
                if isinstance(range_data, Mapping)
                else None
            )
            for value in (from_val, to_val):
                text_value = str(value)
                yield text_value, _range_endpoint_output(value, multiplier)
            return

        if list_config.get("wildcard"):
            wildcard = list_name or "wildcard"
            yield wildcard, wildcard
            return

    if isinstance(list_config, list):
        for value in list_config:
            yield from _value_item_input_outputs(value)


def _value_item_input_outputs(value: Any) -> Iterable[tuple[str, Any]]:
    """Yield expanded spoken inputs paired with one output value."""
    if isinstance(value, str):
        for input_value in _expand_list_input_value(value):
            yield input_value, input_value
        return
    if not isinstance(value, Mapping):
        return
    output_value = value.get("out")
    for raw_input in _string_values(value.get("in")):
        expanded_inputs = tuple(_expand_list_input_value(raw_input))
        for input_value in expanded_inputs:
            yield input_value, output_value if "out" in value else input_value


def _expand_list_input_value(value: str) -> Iterable[str]:
    """Yield spoken variants from one text-list input value."""
    if not value.strip():
        return
    if is_fixed_sentence(value):
        yield _clean_expanded_text(value)
        return
    yield from expand_sentence_template(
        value,
        {},
        {},
        max_expansions=DEFAULT_MAX_CANDIDATES_PER_TEMPLATE,
    )


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


def clear_grammar_loader_caches() -> None:
    """Clear all global LRU caches in grammar loader module."""
    _warn_unknown_expansion_rule.cache_clear()
    _normalized_literal_phrase_variants.cache_clear()
    _text_relevance_key.cache_clear()
    _compound_query_tokens.cache_clear()
    _compact_script_phrase_fallback_enabled.cache_clear()
    _uses_compact_non_latin_script.cache_clear()
    _cached_normalize_no_diac.cache_clear()
    _cached_expansion_tag_context.cache_clear()
    _cached_template_literals.cache_clear()
    _cached_repeated_output_names.cache_clear()
    _cached_required_slots.cache_clear()
    _cached_template_slot_references.cache_clear()
    _parse_hassil.cache_clear()
