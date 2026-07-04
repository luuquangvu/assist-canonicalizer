"""Candidate loading from Home Assistant conversation intent sources."""

from __future__ import annotations

import contextlib
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product
from typing import Any

import orjson

from .candidate import Candidate, CandidateSource, candidate_dedupe_preference_key
from .const import (
    DEFAULT_MAX_CANDIDATES_PER_INTENT,
    DEFAULT_MAX_CANDIDATES_PER_TEMPLATE,
    DEFAULT_MAX_DYNAMIC_CANDIDATES,
    DEFAULT_MAX_DYNAMIC_SLOT_VALUES,
    DEFAULT_MAX_TOTAL_CANDIDATES_PER_LANGUAGE,
    ENTITY_SLOT_NAME_SET,
    LOCATION_SLOT_NAME_SET,
    LOCATION_SLOT_NAMES,
)
from .normalization import normalize_text, normalize_text_no_diacritics
from .registry import merge_slot_values
from .rehydration import rehydrate_wildcard_slots as rehydrate_wildcard_slots
from .rehydration import rehydrate_wildcard_text as rehydrate_wildcard_text
from .utils import register_custom_wildcards_from_sources, wildcard_slot_names

_TEMPLATE_MARKERS = frozenset("{}[]<>|()")
_SLOT_PATTERN = re.compile(r"{([^{}]+)}")
_RULE_PATTERN = re.compile(r"<([^<>]+)>")
_COMPACT_SLOT_TOKEN_MIN_LENGTH = 4
_COMPACT_SCRIPT_QUERY_SPAN_MIN_LENGTH = 2
_COMPACT_SCRIPT_QUERY_SPAN_MAX_LENGTH = 16
_MAX_COMPOUND_QUERY_TOKENS = 4
_TEMPLATE_RELEVANCE_EXPANSION_LIMIT = 200
_TEXT_RUN_STOP_CHARS = frozenset("[]()|{<;")


@dataclass(frozen=True, slots=True)
class RegistrySlotValue:
    """Precomputed lexical data for one registry slot value."""

    text: str
    normalized_text: str
    tokens: tuple[str, ...]
    position: int
    normalized_no_diacritics: str
    tokens_no_diacritics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TemplateSlotReference:
    """Slot reference parsed from HassIL template syntax."""

    list_name: str
    output_name: str


@dataclass(frozen=True, slots=True)
class _TemplateCompilationState:
    """Per-data-item state shared while compiling registry sentence templates."""

    source_key: str
    expansion_rules: Mapping[str, str]
    base_data_slot_values: Mapping[str, tuple[str, ...]]
    slot_output_value_maps: Mapping[str, Mapping[str, Any]]
    static_slots: Mapping[str, str]
    context_slots: frozenset[str]
    domains: tuple[str, ...]
    language: str | None


class RegistrySlotIndex(dict[str, tuple[RegistrySlotValue, ...]]):
    """Precomputed registry values index with an inverted index helper."""

    def __init__(self, data: dict[str, tuple[RegistrySlotValue, ...]]):
        """Initialize and create the inverted cache store."""
        super().__init__(data)
        self._index_record_ids = frozenset(id(records) for records in data.values())
        self._inverted_cache: dict[
            int,
            tuple[tuple[RegistrySlotValue, ...], dict[str, list[RegistrySlotValue]]],
        ] = {}

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


@dataclass(frozen=True, slots=True)
class DynamicRegistryTemplate:
    """Query-independent data needed to expand one registry template."""

    sentence: str
    slot_references: tuple[TemplateSlotReference, ...]
    sentence_slots: frozenset[str]
    slot_output_names: Mapping[str, tuple[str, ...]]
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


def _compile_template_from_sentence(
    state: _TemplateCompilationState,
    sentence: str,
    *,
    include_literal_only_templates: bool,
    include_area_only_templates: bool,
) -> DynamicRegistryTemplate | None:
    """Compile a single sentence into a DynamicRegistryTemplate, or None to skip."""
    slot_references = _template_slot_references(sentence, state.expansion_rules)
    sentence_slots = frozenset(ref.list_name for ref in slot_references)
    slot_output_names = _slot_output_names(slot_references)
    literal_text, variants = _template_literals(sentence, state.expansion_rules)
    resolved_slots = set(state.base_data_slot_values) | set(state.static_slots)
    unresolved_slots = sentence_slots - resolved_slots
    entity_slots = tuple(sorted(unresolved_slots & ENTITY_SLOT_NAME_SET))
    query_slots = tuple(sorted(unresolved_slots & (ENTITY_SLOT_NAME_SET | LOCATION_SLOT_NAME_SET)))
    if (
        not include_area_only_templates
        and query_slots
        and not entity_slots
        and not _is_domain_scoped_location_template(sentence_slots, state.domains)
    ):
        return None
    if not include_literal_only_templates and not query_slots:
        return None
    if not query_slots and (not variants or is_fixed_sentence(sentence)):
        return None
    no_diac_variants = tuple(
        frozenset(_cached_normalize_no_diac(token, state.language) for token in tokens)
        for tokens in variants
    )
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
    wildcards = wildcard_slot_names(state.language)
    if sentence_wildcards := sentence_slots & wildcards:
        base_metadata["wildcard_slots"] = ",".join(sorted(sentence_wildcards))
    return DynamicRegistryTemplate(
        sentence=sentence,
        slot_references=slot_references,
        sentence_slots=sentence_slots,
        slot_output_names=slot_output_names,
        slot_output_values={
            slot_name: state.slot_output_value_maps[slot_name]
            for slot_name in sentence_slots
            if slot_name in state.slot_output_value_maps
        },
        entity_slots=entity_slots,
        query_slots=query_slots,
        domains=state.domains,
        expansion_rules=state.expansion_rules,
        base_data_slot_values=state.base_data_slot_values,
        static_slots=state.static_slots,
        required_slots=frozenset(_required_slots(sentence, state.expansion_rules)),
        metadata=base_metadata,
        literal_token_variants=variants,
        literal_token_variants_no_diac=no_diac_variants,
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
    output_value_maps = _slot_output_value_maps(source_config, intent_config, data_item)
    static_slots = _static_slot_values(data_item)
    context_slots = _context_slot_names(data_item)
    domains = _context_domains(data_item)
    state = _TemplateCompilationState(
        source_key=source_key,
        expansion_rules=expansion_rules,
        base_data_slot_values=base_data_slot_values,
        slot_output_value_maps=output_value_maps,
        static_slots=static_slots,
        context_slots=context_slots,
        domains=domains,
        language=language,
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
    if compiled_intents is None:
        compiled_intents = compile_dynamic_registry_intents(
            intent_sources,
            language,
            include_literal_only_templates=include_literal_only_templates,
            include_area_only_templates=include_area_only_templates,
        )

    query_no_diac = normalize_text_no_diacritics(query, language)
    query_tokens_no_diac = frozenset(query_no_diac.split())

    candidates: list[Candidate] = []
    relevant_cache: dict[int, tuple[str, ...]] = {}
    scoped_cache: dict[tuple[str, tuple[str, ...]], tuple[RegistrySlotValue, ...]] = {}
    for compiled_intent in compiled_intents:
        intent_candidates = _query_candidates_from_compiled_intent(
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
        )
        if not intent_candidates:
            continue
        candidates.extend(intent_candidates)
        if len(candidates) > max_candidates:
            candidates = _top_query_candidates(
                candidates,
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


def _text_relevance_key(
    candidate_normalized: str,
    query_normalized: str,
    query_tokens: frozenset[str],
) -> tuple[int, int, float, int]:
    """Return a stable relevance key for normalized candidate text."""
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
    """Return whether a sentence has no Hassil template syntax."""
    return bool(sentence.strip()) and all(marker not in sentence for marker in _TEMPLATE_MARKERS)


def expand_sentence_template(
    sentence: str,
    slot_values: Mapping[str, tuple[str, ...]],
    expansion_rules: Mapping[str, str],
    *,
    max_expansions: int = DEFAULT_MAX_CANDIDATES_PER_TEMPLATE,
    fair: bool = False,
) -> tuple[str, ...]:
    """Expand a bounded subset of Hassil sentence template syntax."""
    if max_expansions < 1:
        raise ValueError("max_expansions must be positive")
    top_node = _parse_hassil(sentence)
    expansions = top_node.expand(
        slot_values,
        expansion_rules,
        frozenset(),
        max_expansions,
        fair=fair,
    )
    return tuple(_deduplicate_texts(expansions, max_expansions))


def _build_candidate(
    expanded_sentence: str,
    intent_name: str,
    source: CandidateSource,
    language: str,
    base_metadata: Mapping[str, str],
    presorted_values: dict[str, list[str]],
    slot_output_names: Mapping[str, tuple[str, ...]] | None = None,
    slot_output_values: Mapping[str, Mapping[str, Any]] | None = None,
    static_slots_dict: dict[str, str] | None = None,
    literal_variants: tuple[frozenset[str], ...] | None = None,
) -> Candidate:
    """Construct a Candidate with extracted slot metadata."""
    slots = _extract_slots_from_expanded_text(
        expanded_sentence,
        presorted_values,
        slot_output_names,
        slot_output_values,
    )
    if static_slots_dict:
        slots = {**static_slots_dict, **slots}
    metadata = dict(base_metadata)
    if slots:
        metadata["slots"] = orjson.dumps(slots).decode("utf-8")
    candidate = Candidate(
        text=expanded_sentence,
        intent_name=intent_name,
        source=source,
        language=language,
        metadata=metadata,
        slot_values=tuple(str(value) for value in slots.values() if value is not None),
    )
    if literal_variants is not None:
        object.__setattr__(candidate, "_literal_variants", literal_variants)
    return candidate


def _expanded_text_length_key(text: str) -> tuple[int, int]:
    """Return the candidate length sort key without allocating token lists."""
    return ((text.count(" ") + 1) if text else 0, len(text))


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
    expansion_rules = _expansion_rules(source_config, intent_config, data_item)
    base_data_slot_values = _slot_values(source_config, intent_config, data_item)
    output_value_maps = _slot_output_value_maps(source_config, intent_config, data_item)
    context_slots = _context_slot_names(data_item)

    for sentence in sentences:
        if not isinstance(sentence, str):
            continue
        slot_references = _template_slot_references(sentence, expansion_rules)
        sentence_slots = frozenset(ref.list_name for ref in slot_references)
        slot_output_names = _slot_output_names(slot_references)
        slot_values = merge_slot_values(
            base_data_slot_values,
            _registry_slot_values_for_template(
                data_item,
                sentence_slots=sentence_slots,
                base_data_slot_values=base_data_slot_values,
                registry_slot_values=registry_slot_values,
            ),
        )
        required = _required_slots(sentence, expansion_rules)
        if any(not slot_values.get(slot) for slot in required):
            continue

        static_slots_dict = _static_slot_values(data_item)

        literal_text, variants = _template_literals(sentence, expansion_rules)
        base_metadata = _candidate_metadata(
            source_key,
            sentence,
            expansion_rules,
            literal_text=literal_text,
            literal_variants=variants,
        )
        if static_slots_dict:
            base_metadata = dict(base_metadata)
            base_metadata["static_slots"] = ",".join(sorted(static_slots_dict.keys()))
        if context_slots:
            if not isinstance(base_metadata, dict):
                base_metadata = dict(base_metadata)
            base_metadata["context_slots"] = ",".join(sorted(context_slots))

        wildcards = wildcard_slot_names(language)
        if sentence_wildcards := sentence_slots & wildcards:
            if not isinstance(base_metadata, dict):
                base_metadata = dict(base_metadata)
            base_metadata["wildcard_slots"] = ",".join(sorted(sentence_wildcards))

        presorted_values = _presort_slot_values(
            _referenced_slot_values(slot_values, slot_references)
        )
        expanded_sentences = _candidate_texts(sentence, slot_values, expansion_rules)
        # Sort each sentence's candidates by length (word count, then character length)
        sorted_sentences = sorted(expanded_sentences, key=_expanded_text_length_key)
        for expanded_sentence in sorted_sentences:
            yield _build_candidate(
                expanded_sentence,
                intent_name,
                source,
                language,
                base_metadata,
                presorted_values,
                slot_output_names,
                output_value_maps,
                static_slots_dict,
                literal_variants=variants,
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


def _resolve_template_slot_values(
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
        )
        if not dynamic_registry_slots:
            return None
    else:
        dynamic_registry_slots = {}
    slot_values = merge_slot_values(template.base_data_slot_values, dynamic_registry_slots)
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
    domain_area_exact_rescue: bool,
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
            domain_area_exact_rescue=domain_area_exact_rescue,
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
        domain_area_exact_rescue=domain_area_exact_rescue,
        limit=limit,
    )
    return tuple(candidates)


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
    domain_area_exact_rescue: bool,
    limit: int,
) -> None:
    """Append query-expanded template candidates until the template limit is reached."""
    presorted_values = _presort_slot_values(
        _referenced_slot_values(slot_values, template.slot_references)
    )
    static_slots_dict = dict(template.static_slots)
    for expanded_sentence in _query_candidate_texts(
        template.sentence,
        slot_values,
        template.expansion_rules,
        query_normalized,
        query_tokens,
    ):
        if domain_area_exact_rescue and not _is_exact_query_rescue_candidate(
            expanded_sentence,
            query_normalized,
            query_no_diac,
            language,
        ):
            continue
        key = (normalize_text(expanded_sentence), intent_name)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            _build_candidate(
                expanded_sentence,
                intent_name,
                source,
                language,
                template.metadata,
                presorted_values,
                template.slot_output_names,
                template.slot_output_values,
                static_slots_dict,
                literal_variants=template.literal_token_variants,
            )
        )
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
    values = dict(slot_values)
    for slot_name, exact_values in exact_slots.items():
        values[slot_name] = exact_values
    if exact_slots.keys() & set(template.entity_slots):
        for location_slot in optional_location_slots:
            if location_slot not in exact_slots:
                values.pop(location_slot, None)
    preferred: list[dict[str, tuple[str, ...]]] = [values]
    for slot_name in template.entity_slots:
        if exact_values := exact_slots.get(slot_name):
            values = dict(slot_values)
            values[slot_name] = exact_values
            for location_slot in optional_location_slots:
                values.pop(location_slot, None)
            for location_slot in template.required_slots & LOCATION_SLOT_NAME_SET:
                if exact_location_values := exact_slots.get(location_slot):
                    values[location_slot] = exact_location_values
            preferred.append(values)

    if template.domains:
        for slot_name in template.query_slots:
            if slot_name not in LOCATION_SLOT_NAME_SET:
                continue
            if exact_values := exact_slots.get(slot_name):
                values = dict(slot_values)
                values[slot_name] = exact_values
                preferred.append(values)

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
            if _slot_value_occurs_in_query(
                value,
                query_normalized,
                query_no_diac,
                language,
            )
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
    """Return whether a normalized phrase occurs on query token boundaries."""
    if not phrase:
        return False
    if f" {phrase} " in f" {query_normalized} ":
        return True
    return bool(
        " " not in phrase
        and " " not in query_normalized
        and _uses_compact_non_latin_script(phrase)
        and phrase in query_normalized
    )


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
    include_literal_only_templates: bool = True,
    include_area_only_templates: bool = True,
) -> tuple[Candidate, ...]:
    """Expand one compiled intent using query-relevant registry values."""
    candidates: list[Candidate] = []
    for template in compiled_intent.templates:
        if not include_literal_only_templates and not template.query_slots:
            continue
        if _is_area_only_excluded(
            template, include_area_only_templates=include_area_only_templates
        ):
            continue
        domain_area_exact_rescue = (
            not include_area_only_templates
            and template.query_slots
            and not template.entity_slots
            and _is_domain_scoped_location_template(template.sentence_slots, template.domains)
        )
        if not _literal_token_variants_match_query(
            template.literal_token_variants,
            template.literal_token_variants_no_diac,
            query_tokens,
            literal_text=template.metadata.get("literal_text"),
            query_normalized=query_normalized,
            query_tokens_no_diac=query_tokens_no_diac,
            query_no_diac=query_no_diac,
            language=language,
        ):
            continue
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
        )
        if slot_values is None:
            continue
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
            domain_area_exact_rescue=bool(domain_area_exact_rescue),
            limit=template_limit,
        )
        candidates.extend(template_candidates)
        if len(candidates) > max_candidates:
            candidates = _top_query_candidates(
                candidates,
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


def _is_exact_query_rescue_candidate(
    expanded_sentence: str,
    query_normalized: str,
    query_no_diac: str | None,
    language: str,
) -> bool:
    """Return whether a domain-area dynamic rescue exactly matches the query."""
    candidate_normalized = normalize_text(expanded_sentence)
    if candidate_normalized == query_normalized:
        return True
    return bool(
        query_no_diac and _cached_normalize_no_diac(expanded_sentence, language) == query_no_diac
    )


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
            matched_query_slot = True
        else:
            constrained.pop(slot_name, None)
    return constrained if matched_query_slot else {}


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
    records = tuple(selected[:DEFAULT_MAX_CANDIDATES_PER_INTENT])
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
        lookup_tokens = set(query_tokens)
        lookup_tokens.update(_compound_query_tokens(query_normalized))
        if query_no_diac:
            lookup_tokens.update(_compound_query_tokens(query_no_diac))
        if query_tokens_no_diac:
            lookup_tokens.update(query_tokens_no_diac)
        for token in lookup_tokens:
            if token in lookup:
                candidate_records.update(lookup[token])

    scored: list[tuple[tuple[bool, bool, bool, int, int], int, str]] = []
    query_compact = _compact_slot_text(query_normalized)
    query_no_diac_compact = _compact_slot_text(query_no_diac) if query_no_diac else None
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
            query_compact=query_compact,
            query_no_diac_compact=query_no_diac_compact,
        )
        if match_key is not None:
            scored.append((match_key, value.position, value.text))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return tuple(_deduplicate_texts((value for _, _, value in scored), limit))


def _compact_slot_text(text: str | None) -> str:
    """Return a compact lookup key for multi-token or compound slot matching."""
    if not text:
        return ""
    compact = text.replace(" ", "")
    return "" if len(compact) < _COMPACT_SLOT_TOKEN_MIN_LENGTH else compact


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
    """Return whether literal phrase fallback can find compact-script text."""
    return bool(
        query_normalized
        and " " not in query_normalized
        and _uses_compact_non_latin_script(query_normalized)
    )


@lru_cache(maxsize=65536)
def _uses_compact_non_latin_script(text: str) -> bool:
    """Return whether text contains a non-Latin script that often omits spaces."""
    if not text or text.isascii():
        return False
    for char in text:
        if char.isspace() or char.isascii():
            continue
        name = unicodedata.name(char, "")
        if name and not name.startswith("LATIN"):
            return True
    return False


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


def _presort_slot_values(
    slot_values: Mapping[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    """Return slot values pre-sorted longest-first for substring extraction."""
    shared: dict[tuple[str, ...], list[str]] = {}
    presorted: dict[str, list[str]] = {}
    for slot_name, values in slot_values.items():
        sorted_values = shared.get(values)
        if sorted_values is None:
            sorted_values = sorted(values, key=len, reverse=True)
            shared[values] = sorted_values
        presorted[slot_name] = sorted_values
    return presorted


def _extract_slots_from_expanded_text(
    text: str,
    slot_values: Mapping[str, Sequence[str]],
    slot_output_names: Mapping[str, tuple[str, ...]] | None = None,
    slot_output_values: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract slot values present in the expanded text.

    Values must be pre-sorted longest-first (see :func:`_presort_slot_values`).
    """
    slots = {}
    previous_values: Sequence[str] | None = None
    previous_match: str | None = None
    for slot_name, values in slot_values.items():
        if values is previous_values:
            matched = previous_match
        else:
            matched = next((val for val in values if val in text), None)
            previous_values = values
            previous_match = matched
        if matched is not None:
            output_value = (
                slot_output_values.get(slot_name, {}).get(matched, matched)
                if slot_output_values is not None
                else matched
            )
            output_names = (
                slot_output_names.get(slot_name, (slot_name,))
                if slot_output_names is not None
                else (slot_name,)
            )
            for output_name in output_names:
                slots.setdefault(output_name, output_value)
    return slots


def _referenced_slot_values(
    slot_values: Mapping[str, tuple[str, ...]],
    slot_references: Iterable[TemplateSlotReference],
) -> dict[str, tuple[str, ...]]:
    """Return only slot values referenced by the current template."""
    selected = {}
    for reference in slot_references:
        if reference.list_name in selected:
            continue
        if values := slot_values.get(reference.list_name):
            selected[reference.list_name] = values
    return selected


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


def _query_candidate_texts(
    sentence: str,
    slot_values: Mapping[str, tuple[str, ...]],
    expansion_rules: Mapping[str, str],
    query_normalized: str,
    query_tokens: frozenset[str],
) -> tuple[str, ...]:
    """Return query-relevant candidate texts for a dynamic template."""
    if is_fixed_sentence(sentence):
        return (_clean_expanded_text(sentence),)
    expanded = expand_sentence_template(
        sentence,
        slot_values,
        expansion_rules,
        max_expansions=max(
            DEFAULT_MAX_CANDIDATES_PER_TEMPLATE,
            _TEMPLATE_RELEVANCE_EXPANSION_LIMIT,
        ),
        fair=True,
    )
    return tuple(
        sorted(
            expanded,
            key=lambda text: _text_relevance_key(
                normalize_text(text), query_normalized, query_tokens
            ),
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
    """Return registry slots constrained by Hassil context and template shape."""
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
    query_compact: str | None = None,
    query_no_diac_compact: str | None = None,
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

    # 3. Compound/spacing-insensitive substring matches
    value_compact = _compact_slot_text(value_normalized)
    compact_substring = bool(value_compact and query_compact and value_compact in query_compact)
    if (
        not compact_substring
        and value_no_diac
        and query_no_diac_compact
        and (value_no_diac_compact := _compact_slot_text(value_no_diac))
    ):
        compact_substring = value_no_diac_compact in query_no_diac_compact
    if compact_substring:
        return (False, True, True, token_count, -token_count)

    # 4. Token overlaps
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
    """Return localized literal words from a Hassil sentence template."""
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
        *,
        fair: bool = False,
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
        *,
        fair: bool = False,
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
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        if self.name in seen:
            return ()
        rule_text = rules.get(self.name)
        if rule_text is None:
            return ()
        node = _parse_hassil(rule_text)
        return node.expand(slot_values, rules, seen | {self.name}, limit, fair=fair)


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
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        return _unique_capped_with_empty(
            self.child.expand(slot_values, rules, seen, limit, fair=fair),
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
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        if fair:
            branch_expansions = [
                branch.expand(slot_values, rules, seen, limit, fair=True)
                for branch in self.branches
            ]
            return _interleave_unique_capped(branch_expansions, limit)
        unique_expansions: dict[str, None] = {}
        for branch in self.branches:
            remaining = limit - len(unique_expansions)
            if remaining < 1:
                break
            for val in branch.expand(slot_values, rules, seen, remaining, fair=False):
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
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        return self._permuted_texts(
            [branch.expand(slot_values, rules, seen, limit, fair=fair) for branch in self.branches],
            limit,
        )

    @staticmethod
    def _permuted_texts(branch_fragments: list[tuple[str, ...]], limit: int) -> tuple[str, ...]:
        """Return capped text variants for every branch order."""
        if not branch_fragments:
            return ("",)
        unique: dict[str, None] = {}
        for ordered_fragments in permutations(branch_fragments):
            for fragments in product(*ordered_fragments):
                text = " ".join(fragment.strip() for fragment in fragments if fragment.strip())
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
    ) -> tuple[str, ...]:
        """Expand node into all spoken text variants using slots and rules."""
        if not self.children:
            return ("",)
        child_fragments = [
            child.expand(slot_values, rules, seen, limit, fair=fair) for child in self.children
        ]
        return self._accumulate(child_fragments, limit, fair=fair)


_PARSED_TEMPLATE_CACHE: dict[str, _HassilNode] = {}


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


def _parse_hassil(text: str) -> _HassilNode:
    """Parse Hassil template string into an AST node, caching the results."""
    cached = _PARSED_TEMPLATE_CACHE.get(text)
    if cached is not None:
        return cached

    node, _ = _parse_hassil_expr(text, 0)
    _PARSED_TEMPLATE_CACHE[text] = node
    return node


def _parse_hassil_expr(text: str, i: int, close_char: str | None = None) -> tuple[_HassilNode, int]:
    """Parse a Hassil expression from a given index."""
    current_branch: list[_HassilNode] = []
    branches: list[_HassilNode] = []
    branch_separator: str | None = None

    while i < len(text):
        char = text[i]
        match char:
            case "[":
                child, i = _parse_hassil_expr(text, i + 1, "]")
                current_branch.append(_HassilOptionalNode(child))
            case "]" | ")":
                if char == close_char:
                    i += 1
                    break
                current_branch.append(_HassilTextNode(char))
                i += 1
            case "(":
                child, i = _parse_hassil_expr(text, i + 1, ")")
                current_branch.append(child)
            case "|":
                if branch_separator == ";":
                    current_branch.append(_HassilTextNode(char))
                    i += 1
                    continue
                branch_separator = "|"
                branches.append(_make_branch_node(current_branch))
                current_branch = []
                i += 1
            case ";":
                if branch_separator == "|":
                    current_branch.append(_HassilTextNode(char))
                    i += 1
                    continue
                branch_separator = ";"
                branches.append(_make_branch_node(current_branch))
                current_branch = []
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
    for rule_match in _RULE_PATTERN.finditer(text):
        rule_name = rule_match.group(1).strip()
        if rule_name in seen_rules:
            continue
        rule_text = expansion_rules.get(rule_name)
        if rule_text is not None:
            for reference in _template_slot_references(
                rule_text,
                expansion_rules,
                seen_rules | frozenset({rule_name}),
            ):
                add_reference(reference)
    return tuple(references)


def _slot_reference(raw: str) -> TemplateSlotReference:
    """Parse a HassIL slot reference into expansion list and output slot names."""
    list_name, separator, output_name = raw.partition(":")
    list_name = list_name.strip()
    output_name = output_name.strip() if separator else list_name
    return TemplateSlotReference(list_name=list_name, output_name=output_name or list_name)


def _slot_output_names(
    slot_references: Iterable[TemplateSlotReference],
) -> dict[str, tuple[str, ...]]:
    """Return output slot names keyed by expansion list name."""
    output_names: dict[str, list[str]] = {}
    for reference in slot_references:
        names = output_names.setdefault(reference.list_name, [])
        if reference.output_name not in names:
            names.append(reference.output_name)
    return {list_name: tuple(names) for list_name, names in output_names.items()}


def _context_domains(data_item: Mapping[str, Any]) -> tuple[str, ...]:
    """Return required entity domains for one Hassil data item."""
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
    """Return static string slots declared on a Hassil data item."""
    static_slots: dict[str, str] = {}
    slots = data_item.get("slots", {})
    if not isinstance(slots, Mapping):
        return static_slots
    for slot_name, slot_value in slots.items():
        if isinstance(slot_name, str) and isinstance(slot_value, str):
            static_slots[slot_name] = slot_value
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


def _values_from_list_config(list_config: Any, list_name: str | None = None) -> Iterable[str]:
    """Yield spoken values from a Hassil list config."""
    if isinstance(list_config, Mapping):
        if "values" in list_config:
            values = list_config.get("values", [])
            if isinstance(values, list):
                for value in values:
                    for input_value, _ in _value_item_input_outputs(value):
                        yield input_value
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
    """Yield spoken input to output value pairs from a Hassil list config."""
    if isinstance(list_config, Mapping):
        if "values" in list_config:
            values = list_config.get("values", [])
            if isinstance(values, list):
                for value in values:
                    yield from _value_item_input_outputs(value)
            return

        if "range" in list_config:
            range_data = list_config["range"]
            try:
                from_val = range_data.get("from", 0)
                to_val = range_data.get("to", 100)
            except (AttributeError, ValueError, TypeError):
                from_val = "1"
                to_val = "100"
            for value in (from_val, to_val):
                text_value = str(value)
                yield text_value, value
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
