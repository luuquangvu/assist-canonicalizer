"""Quality checks for real-world evaluation datasets."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import hassil
import hassil.errors
import orjson
import pytest

from custom_components.assist_canonicalizer import ranking
from custom_components.assist_canonicalizer.candidate import (
    Candidate,
    CandidateSource,
    slot_alias_values_by_key,
)
from custom_components.assist_canonicalizer.const import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    FallbackReason,
)
from custom_components.assist_canonicalizer.grammar_loader import (
    DynamicRegistryIntent,
    RegistrySlotIndex,
    build_candidates_from_intent_sources,
    build_query_registry_candidates,
    build_registry_slot_index,
    compile_dynamic_registry_intents,
)
from custom_components.assist_canonicalizer.ranking import (
    RankedCandidate,
    ScoreBreakdown,
)
from tools import benchmark

benchmark._bootstrap_project_imports()

DATASET_DIR = Path("tests/real_world")
_LANGUAGES: tuple[str, ...] | None = None
_UNCAPPED_STATIC_CANDIDATE_LIMIT = 1_000_000


def _discover_languages() -> tuple[str, ...]:
    """Discover dataset languages by scanning ``tests/real_world/*.json``."""
    global _LANGUAGES
    if _LANGUAGES is not None:
        return _LANGUAGES
    repo_root = str(Path(__file__).resolve().parent.parent)
    safe_dir = benchmark.sanitize_path(repo_root, str(DATASET_DIR))
    languages: list[str] = []
    safe_dir_real = os.path.realpath(safe_dir)
    for filename in sorted(os.listdir(safe_dir)):
        if filename.endswith(".json"):
            file_path = os.path.join(safe_dir, filename)
            real_path = os.path.realpath(file_path)
            if not os.path.isfile(real_path):
                continue
            if not real_path.startswith(safe_dir_real + os.sep):
                continue
            languages.append(filename[:-5])
    _LANGUAGES = tuple(languages)
    return _LANGUAGES


HARD_CATEGORIES = frozenset(
    {
        "complex_distortion",
        "extra_words",
        "missing_words",
        "semantic_challenge",
        "spelling_mistake",
        "synonym_paraphrase",
    }
)
HASSIL_ALIGN_CATEGORIES = frozenset(
    {
        "exact_match",
        "intent_coverage",
        "supported_filler",
    }
)
WORD_ADDITION_CATEGORIES = frozenset({"extra_words", "supported_filler"})
REQUIRED_CATEGORY_MINIMUMS = {
    "complex_distortion": 10,
    "exact_match": 10,
    "extra_words": 10,
    "intent_coverage": 15,
    "missing_words": 10,
    "semantic_challenge": 10,
    "spelling_mistake": 10,
    "supported_filler": 10,
    "synonym_paraphrase": 10,
}
KNOWN_CATEGORIES = frozenset(REQUIRED_CATEGORY_MINIMUMS)


def _candidate_slot_keys(candidate: Candidate) -> frozenset[str]:
    """Return slot keys represented by one generated candidate."""
    slot_keys: set[str] = set()
    slots_text = candidate.metadata.get("slots")
    if slots_text:
        try:
            slots = orjson.loads(slots_text)
        except orjson.JSONDecodeError:
            slots = {}
        if isinstance(slots, dict):
            slot_keys.update(key for key in slots if isinstance(key, str))
    if wildcard_slots := candidate.metadata.get("wildcard_slots"):
        slot_keys.update(slot.strip() for slot in wildcard_slots.split(",") if slot.strip())
    return frozenset(slot_keys)


def _candidate_slot_keys_cover(
    candidate_slot_keys: frozenset[str],
    expected_slot_keys: frozenset[str],
    language: str | None = None,
) -> bool:
    """Return whether candidate slot keys satisfy expected slot-key semantics."""
    mapping = _slot_key_mapping(language)
    aliases = set(slot_alias_values_by_key({key: key for key in candidate_slot_keys}, mapping))

    for expected_slot_key in expected_slot_keys:
        if expected_slot_key in aliases:
            continue
        if (
            expected_slot_key == "name"
            and "domain" in aliases
            and any(slot_name in aliases for slot_name in benchmark.LOCATION_SLOT_NAMES)
        ):
            continue
        if ":" in expected_slot_key and expected_slot_key.split(":", 1)[0] in aliases:
            continue
        return False
    return True


def _slot_key_mapping(language: str | None) -> dict[str, frozenset[str]]:
    """Return slot-key aliases using the benchmark slot-matching rules."""
    mapping = benchmark._SLOT_MAPPINGS_BY_LANG.get(language) if language else None
    slot_mapping = {} if mapping is None else dict(mapping)
    for key, aliases in benchmark._GLOBAL_SLOT_ALIASES.items():
        slot_mapping[key] = slot_mapping.get(key, frozenset()) | aliases
    return slot_mapping


@dataclass(frozen=True, slots=True)
class DatasetContext:
    """Reusable HassIL context for one real-world dataset."""

    language: str
    cases: tuple[dict[str, Any], ...]
    sources: dict[str, Mapping[str, Any]]
    slots: dict[str, tuple[str, ...]]
    registry_slot_index: RegistrySlotIndex
    dynamic_registry_intents: tuple[DynamicRegistryIntent, ...]
    static_non_wildcard_pairs: frozenset[tuple[str, str]]
    static_wildcard_pairs: frozenset[tuple[str, str]]
    static_wildcard_texts_by_intent: dict[str, tuple[str, ...]]
    static_candidate_slot_keys_by_intent: dict[str, tuple[frozenset[str], ...]]
    static_normalized_texts: frozenset[str]
    hassil_results_by_query: dict[tuple[str, tuple[tuple[str, Any], ...]], tuple[Any, ...]]

    @property
    def static_candidate_pairs(self) -> frozenset[tuple[str, str]]:
        """Return union of wildcard and non-wildcard static candidate pairs."""
        return self.static_non_wildcard_pairs | self.static_wildcard_pairs

    intents: hassil.intents.Intents
    slot_lists: dict[str, hassil.intents.SlotList]


@pytest.fixture(scope="module", params=_discover_languages())
def dataset_context(request: pytest.FixtureRequest) -> DatasetContext:
    """Return validated dataset context for one language."""
    language = str(request.param)
    path = DATASET_DIR / f"{language}.json"
    data = orjson.loads(path.read_text(encoding="utf-8"))
    raw_cases = data["test_cases"]
    cases = tuple(benchmark._validate_test_cases(raw_cases, language, str(path)))
    slots = benchmark._dataset_registry_slots(data, language)
    sources = benchmark.load_language_intent_sources(language)
    registry_slot_index = build_registry_slot_index(slots, language)
    dynamic_registry_intents = compile_dynamic_registry_intents(
        sources,
        language,
        include_literal_only_templates=False,
        include_area_only_templates=False,
    )
    with patch(
        "custom_components.assist_canonicalizer.grammar_loader.DEFAULT_MAX_CANDIDATES_PER_INTENT",
        _UNCAPPED_STATIC_CANDIDATE_LIMIT,
    ):
        candidates = build_candidates_from_intent_sources(
            language,
            sources,
            slots,
            max_candidates=None,
        )
    merged_intents: dict[str, Any] = {}
    for source in sources.values():
        hassil.merge_dict(merged_intents, source)
    static_wildcard_pairs = frozenset(
        (candidate.intent_name, candidate.text)
        for candidate in candidates
        if candidate.has_wildcard
    )
    static_wildcard_texts_by_intent: dict[str, list[str]] = {}
    for intent_name, text in static_wildcard_pairs:
        static_wildcard_texts_by_intent.setdefault(intent_name, []).append(text)
    static_candidate_slot_keys_by_intent: dict[str, list[frozenset[str]]] = {}
    for candidate in candidates:
        static_candidate_slot_keys_by_intent.setdefault(candidate.intent_name, []).append(
            _candidate_slot_keys(candidate)
        )
    return DatasetContext(
        language=language,
        cases=cases,
        sources=sources,
        slots=slots,
        registry_slot_index=registry_slot_index,
        dynamic_registry_intents=dynamic_registry_intents,
        static_non_wildcard_pairs=frozenset(
            (candidate.intent_name, candidate.text)
            for candidate in candidates
            if not candidate.has_wildcard
        ),
        static_wildcard_pairs=static_wildcard_pairs,
        static_wildcard_texts_by_intent={
            intent_name: tuple(texts)
            for intent_name, texts in static_wildcard_texts_by_intent.items()
        },
        static_candidate_slot_keys_by_intent={
            intent_name: tuple(slot_keys)
            for intent_name, slot_keys in static_candidate_slot_keys_by_intent.items()
        },
        static_normalized_texts=frozenset(candidate.normalized_text for candidate in candidates),
        hassil_results_by_query={},
        intents=hassil.intents.Intents.from_dict(merged_intents),
        slot_lists=benchmark.make_hassil_slot_lists(slots),
    )


@pytest.mark.current_intents
def test_real_world_categories_are_balanced(dataset_context: DatasetContext) -> None:
    """Assert every language has enough coverage in each behavior category."""
    counts: dict[str, int] = {}
    for case in dataset_context.cases:
        category = case["category"]
        counts[category] = counts.get(category, 0) + 1
    missing = {
        category: minimum
        for category, minimum in REQUIRED_CATEGORY_MINIMUMS.items()
        if counts.get(category, 0) < minimum
    }
    unexpected = sorted(set(counts) - KNOWN_CATEGORIES)
    assert not unexpected, f"{dataset_context.language}: unknown categories: {unexpected}"
    assert not missing, f"{dataset_context.language}: category counts too low: {missing}"


def test_real_world_category_contracts_cover_every_category() -> None:
    """Require every dataset category to declare one HassIL support contract."""
    assert HARD_CATEGORIES.isdisjoint(HASSIL_ALIGN_CATEGORIES)
    assert HARD_CATEGORIES | HASSIL_ALIGN_CATEGORIES == KNOWN_CATEGORIES


def test_real_world_datasets_retain_one_expected_fallback_ambiguity() -> None:
    """Keep exactly one deliberate action-omission case as an ordinary fallback."""
    fallback_cases: list[tuple[str, str]] = []
    for language in _discover_languages():
        path = DATASET_DIR / f"{language}.json"
        data = orjson.loads(path.read_bytes())
        cases = benchmark._validate_test_cases(data["test_cases"], language, str(path))
        fallback_cases.extend(
            (language, case["query"]) for case in cases if case["expected_fallback"]
        )

    assert fallback_cases == [("de", "schalte wohnzimmerlicht")]


@pytest.mark.current_intents
def test_real_world_expected_intents_are_hassil_candidates(
    dataset_context: DatasetContext,
) -> None:
    """Assert every expected intent/slot shape is represented in generated candidates.

    Expected canonical strings are benchmark targets, not necessarily literal
    HassIL sentence outputs. Home Assistant intent datasets can change sentence
    variants while preserving the same intent/slot behavior, so this test checks
    candidate slot-key coverage and leaves exact outcome quality to the benchmark.
    """
    missing: list[dict[str, Any]] = []
    for case in dataset_context.cases:
        expected_intent = case["expected_intent"]
        expected_slot_keys = frozenset(case.get("expected_slots", {})) - frozenset(
            case.get("context", {})
        )
        candidate_slot_keys = dataset_context.static_candidate_slot_keys_by_intent.get(
            expected_intent, ()
        )
        if expected_slot_keys:
            has_matching_shape = any(
                _candidate_slot_keys_cover(
                    slot_keys,
                    expected_slot_keys,
                    dataset_context.language,
                )
                for slot_keys in candidate_slot_keys
            )
        else:
            has_matching_shape = bool(candidate_slot_keys)
        if not has_matching_shape:
            missing.append(
                {
                    "query": case["query"],
                    "expected_intent": expected_intent,
                    "expected_slots": sorted(expected_slot_keys),
                }
            )
    assert not missing, (
        f"{dataset_context.language}: expected intent/slot candidates missing: {missing}"
    )


@pytest.mark.current_intents
def test_real_world_exact_match_category_is_literal(
    dataset_context: DatasetContext,
) -> None:
    """Assert exact_match cases are literal canonical commands."""
    failures = []
    for case in dataset_context.cases:
        if case["category"] != "exact_match":
            continue
        if case["query"] != case["expected_canonical"]:
            failures.append((case["query"], case["expected_canonical"]))
    assert not failures, f"{dataset_context.language}: exact_match is not literal: {failures}"


@pytest.mark.current_intents
def test_real_world_hard_categories_are_not_direct_hassil_matches(
    dataset_context: DatasetContext,
) -> None:
    """Assert hard categories are not already handled directly by HassIL."""
    failures = []
    for case in dataset_context.cases:
        if case["category"] not in HARD_CATEGORIES:
            continue
        exact_static = (
            benchmark.normalize_text(case["query"]) in dataset_context.static_normalized_texts
        )
        direct_hassil_match = bool(
            _hassil_results(dataset_context, case["query"], case.get("context"))
        )
        if exact_static or direct_hassil_match:
            failures.append(
                {
                    "query": case["query"],
                    "category": case["category"],
                    "exact_static": exact_static,
                    "direct_hassil_match": direct_hassil_match,
                }
            )
    if failures:
        pytest.fail(
            f"{dataset_context.language}: hard category too easy "
            f"(HassIL now supports directly): {failures}. "
            "Consider updating the dataset to use truly unsupported "
            "queries for hard categories."
        )


@pytest.mark.current_intents
def test_real_world_registry_has_domain_scoped_slots(
    dataset_context: DatasetContext,
) -> None:
    """Assert every dataset-tested intent produces at least one candidate.

    When an intent with ``requires_context: {domain: X}`` lacks domain-scoped
    registry keys (e.g. ``name:vacuum``), the grammar loader silently produces
    0 candidates, a behaviour drift between the benchmark dataset and a real
    Home Assistant deployment.

    Rather than reverse-engineering which domain-scoped keys are needed, this
    test directly checks the output: every intent referenced by the dataset
    must appear in the generated candidate pairs.
    """
    tested_intents: set[str] = {case["expected_intent"] for case in dataset_context.cases}
    intent_candidates: dict[str, list[str]] = {}
    for intent_name, text in dataset_context.static_candidate_pairs:
        intent_candidates.setdefault(intent_name, []).append(text)

    zero_candidate: list[str] = sorted(
        intent for intent in tested_intents if not intent_candidates.get(intent)
    )

    assert not zero_candidate, (
        f"{dataset_context.language}: these dataset-tested intents produce 0 candidates "
        f"(registry may be missing domain-scoped keys): {zero_candidate}"
    )


@pytest.mark.current_intents
def test_real_world_word_addition_categories_align_with_hassil_support(
    dataset_context: DatasetContext,
) -> None:
    """Separate HassIL-supported filler from genuinely unsupported extra words."""
    failures: list[dict[str, Any]] = []
    for case in dataset_context.cases:
        category = case["category"]
        if category not in WORD_ADDITION_CATEGORIES:
            continue
        query = case["query"]
        canonical = case["expected_canonical"]
        has_added_text = benchmark.normalize_text(query) != benchmark.normalize_text(canonical)
        hassil_supports = _recognizes_expected_hassil_outcome(dataset_context, case)
        expected_hassil_support = category == "supported_filler"
        if not has_added_text or hassil_supports != expected_hassil_support:
            failures.append(
                {
                    "query": query,
                    "category": category,
                    "has_added_text": has_added_text,
                    "hassil_supports": hassil_supports,
                    "expected_hassil_support": expected_hassil_support,
                }
            )
    assert not failures, f"{dataset_context.language}: word-addition category mismatch: {failures}"


@pytest.mark.current_intents
def test_real_world_hassil_align_categories_match_production_result(
    dataset_context: DatasetContext,
) -> None:
    """Require aligned categories to match Home Assistant's selected HassIL result."""
    failures = [
        {
            "query": case["query"],
            "category": case["category"],
            "expected_intent": case["expected_intent"],
            "expected_slots": case.get("expected_slots", {}),
        }
        for case in dataset_context.cases
        if case["category"] in HASSIL_ALIGN_CATEGORIES
        and not _recognizes_expected_hassil_outcome(dataset_context, case)
    ]
    assert not failures, f"{dataset_context.language}: HassIL-aligned category mismatch: {failures}"


@pytest.mark.current_intents
def test_real_world_expected_intents_align_with_hassil(
    dataset_context: DatasetContext,
) -> None:
    """Assert expected_intent is consistent with HassIL for align categories.

    For exact_match and intent_coverage, if HassIL picks a different
    intent *and* the canonicalizer can generate a candidate for that
    intent with the same canonical text, the test label is out of sync
    and must be corrected.

    Cases where the canonicalizer does not generate the HassIL-chosen
    intent (e.g.  HassListAddItem vs HassShoppingListAddItem with
    different slot names) are not errors, they reflect genuine
    differences in the intent matching space.
    """
    dynamic_cache: dict[str, set[tuple[str, str]]] = {}
    failures: list[dict[str, str]] = []
    for case in dataset_context.cases:
        if case["category"] not in HASSIL_ALIGN_CATEGORIES:
            continue
        results = _hassil_results(dataset_context, case["query"], case.get("context"))
        if not results:
            continue
        top_result = results[0]
        if benchmark._intents_match(top_result.intent.name, case["expected_intent"]):
            continue
        canonical = case["expected_canonical"]
        cache_key = f"{top_result.intent.name}::{canonical}"
        if cache_key not in dynamic_cache:
            candidates = build_query_registry_candidates(
                dataset_context.language,
                dataset_context.sources,
                dataset_context.slots,
                canonical,
                registry_slot_index=dataset_context.registry_slot_index,
                compiled_intents=dataset_context.dynamic_registry_intents,
                include_literal_only_templates=False,
                include_area_only_templates=False,
            )
            dynamic_cache[cache_key] = {(c.intent_name, c.text) for c in candidates}
        if (top_result.intent.name, canonical) in dynamic_cache[cache_key] or (
            top_result.intent.name,
            canonical,
        ) in dataset_context.static_candidate_pairs:
            failures.append(
                {
                    "query": case["query"],
                    "category": case["category"],
                    "expected_intent": case["expected_intent"],
                    "hassil_intent": top_result.intent.name,
                }
            )
    assert not failures, f"{dataset_context.language}: expected_intent mismatch HassIL: {failures}"


def _slot_value_matches(expected: Any, hassil_value: Any, query: str) -> bool:
    """Check *expected* matches a HassIL entity value.

    Accepts string equality, numeric type coercion, and substring
    containment (HassIL decomposes compound names like
    ``đèn phòng khách`` into ``name=đèn`` + ``area=phòng khách``,
    while the canonicalizer keeps the full compound as ``name``).

    When the HassIL value is a computed domain-level sentinel not
    appearing literally in *query* (e.g. ``all``, ``tất cả``,
    ``alle``), the slot is accepted, the canonicalizer extracts
    the literal text while HassIL resolves to a domain operation.
    """
    if isinstance(expected, str) and isinstance(hassil_value, str):
        if hassil_value not in query:
            return True
        return expected == hassil_value or hassil_value in expected
    try:
        return float(expected) == float(hassil_value)
    except (ValueError, TypeError):
        return False


@pytest.mark.current_intents
def test_real_world_expected_slots_align_with_hassil(
    dataset_context: DatasetContext,
) -> None:
    """Assert expected_slots match HassIL entities for align categories.

    For exact_match and intent_coverage, when HassIL recognizes the
    expected intent, every key in *expected_slots* that also appears
    in HassIL entities is compared, a mismatch means the dataset
    label is wrong and must be corrected to match HassIL ground truth.

    HassIL entity values not present literally in the query text are
    treated as computed domain-level sentinels and skipped (e.g.
    ``name=all`` for ``"tắt quạt"``), since the canonicalizer extracts
    literal slot text while HassIL resolves domain operations.

    Keys present only in one system are skipped, naming conventions
    differ by design (e.g. ``shopping_list_item`` vs ``item``,
    ``timer_seconds`` vs ``minutes``).
    """
    failures: list[dict[str, str]] = []
    for case in dataset_context.cases:
        if case["category"] not in HASSIL_ALIGN_CATEGORIES:
            continue
        expected_slots = case.get("expected_slots", {})
        if not expected_slots:
            continue
        parses = [
            {name: ent.value for name, ent in result.entities.items()}
            for result in _hassil_results(dataset_context, case["query"], case.get("context"))
            if benchmark._intents_match(result.intent.name, case["expected_intent"])
        ]
        if not parses:
            continue
        query = case["query"]
        any_aligns = False
        all_mismatches: list[str] = []
        for entities in parses:
            mismatched = []
            for key, expected_value in expected_slots.items():
                hassil_value = entities.get(key)
                if hassil_value is None:
                    continue
                if not _slot_value_matches(expected_value, hassil_value, query):
                    mismatched.append(
                        f"{key}: expected={expected_value!r}, hassil={hassil_value!r}"
                    )
            if not mismatched:
                any_aligns = True
                break
            all_mismatches.append(", ".join(mismatched))
        if not any_aligns:
            failures.append(
                {
                    "query": case["query"],
                    "category": case["category"],
                    "expected_intent": case["expected_intent"],
                    "mismatches": " | ".join(all_mismatches),
                }
            )
    assert not failures, (
        f"{dataset_context.language}: expected_slots mismatch HassIL: {failures}. "
        "If this is due to upstream grammar updates, please run: "
        "uv run tools/benchmark.py --regenerate-expectations"
    )


def _hassil_cache_key(
    query: str,
    intent_context: Mapping[str, Any] | None = None,
) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """Return a stable HassIL result cache key for query plus context."""
    context_items = (
        tuple(sorted(intent_context.items(), key=lambda item: item[0])) if intent_context else ()
    )
    return query, context_items


def _recognizes_expected_hassil_outcome(
    context: DatasetContext,
    case: Mapping[str, Any],
) -> bool:
    """Return whether HassIL recognizes a case like its clean canonical command."""
    intent_context = case.get("context")
    expected_intent = case["expected_intent"]
    canonical_results = _hassil_results(
        context,
        case["expected_canonical"],
        intent_context,
    )
    expected_slots = [
        {name: entity.value for name, entity in result.entities.items()}
        for result in canonical_results
        if benchmark._intents_match(result.intent.name, expected_intent)
    ] or [case.get("expected_slots", {})]

    result = benchmark.run_hassil_recognize_best(
        case["query"],
        context.intents,
        context.slot_lists,
        intent_context,
        context.language,
    )
    return bool(
        result is not None
        and benchmark._intents_match(result.intent.name, expected_intent)
        and benchmark._slots_match_any(
            {name: entity.value for name, entity in result.entities.items()},
            expected_slots,
            language=context.language,
        )
    )


def _hassil_results(
    context: DatasetContext,
    query: str,
    intent_context: Mapping[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Return cached HassIL recognition results for a dataset query."""
    cache_key = _hassil_cache_key(query, intent_context)
    cached = context.hassil_results_by_query.get(cache_key)
    if cached is not None:
        return cached
    results = tuple(
        benchmark.run_hassil_recognize_all(
            query,
            context.intents,
            context.slot_lists,
            intent_context,
        )
    )
    context.hassil_results_by_query[cache_key] = results
    return results


def test_candidate_slot_keys_include_non_string_slot_values() -> None:
    """Slot-key coverage should not depend on serialized slot value types."""
    candidate = Candidate(
        text="set brightness",
        intent_name="HassLightSet",
        metadata={"slots": '{"brightness":100,"supported":true}'},
    )

    assert _candidate_slot_keys(candidate) == frozenset({"brightness", "supported"})


def test_candidate_slot_keys_cover_domain_scoped_expectations_with_base_slot() -> None:
    """Scoped expected keys can be represented by the generated base slot key."""
    assert _candidate_slot_keys_cover(
        frozenset({"name", "todo_list_item"}),
        frozenset({"name", "name:todo", "todo_list_item"}),
    )


def test_candidate_slot_keys_cover_list_item_aliases() -> None:
    """Dataset slot-shape checks should use benchmark list-item aliases."""
    assert _candidate_slot_keys_cover(
        frozenset({"item"}),
        frozenset({"shopping_list_item", "todo_list_item"}),
    )


def test_benchmark_rank_stage_query_sampling_is_even_and_unique() -> None:
    """Rank-stage profiling should sample stable, evenly spaced unique queries."""
    queries = ("q0", "q1", "q2", "q3", "q4")

    assert benchmark._sample_rank_stage_queries(queries, 3) == ("q0", "q2", "q4")
    assert benchmark._sample_rank_stage_queries(("same", "same", "other"), 3) == (
        "same",
        "other",
    )
    assert benchmark._sample_rank_stage_queries(queries, 0) == ()


def test_benchmark_normalized_bm25_scores_from_raw() -> None:
    """Rank-stage BM25 normalization should match rank_candidates score scaling."""
    assert benchmark._normalized_bm25_scores_from_raw is ranking._normalized_bm25_scores_from_raw
    assert benchmark._normalized_bm25_scores_from_raw((0.0, 2.0, 4.0), 3) == (
        0.0,
        0.5,
        1.0,
    )
    assert benchmark._normalized_bm25_scores_from_raw((), 3) == (0.0, 0.0, 0.0)
    assert benchmark._normalized_bm25_scores_from_raw((1.0,), 0) == ()


def test_benchmark_runtime_observation_uses_production_rank_outputs() -> None:
    """Runtime profiling should classify observable production ranking outcomes."""
    candidate = Candidate(text="turn on light", intent_name="HassTurnOn")
    ranked = (
        RankedCandidate(
            candidate=candidate,
            scores=ScoreBreakdown(
                rapidfuzz_score=1.0,
                char_ngram_score=1.0,
                bm25_score=1.0,
                intent_score=1.0,
                final_score=1.0,
            ),
        ),
    )
    case = benchmark.RuntimeQueryCase(query="turn on light", category="exact_match")

    static_observation = benchmark._runtime_query_observation(case, ranked, 0)
    dynamic_observation = benchmark._runtime_query_observation(case, ranked, 2)

    assert "static_perfect_short_circuit" in static_observation["tags"]
    assert "dynamic_attempted" not in static_observation["tags"]
    assert "dynamic_perfect" in dynamic_observation["tags"]
    assert dynamic_observation["dynamic_candidate_count"] == 2


def test_benchmark_runtime_coverage_counts_each_query_once() -> None:
    """Runtime coverage counters should be based on one measured pass."""
    counts = benchmark._new_runtime_coverage_counts()

    benchmark._record_runtime_coverage(
        counts,
        ("category:exact_match", "dynamic_attempted", "dynamic_candidates", "accepted"),
    )

    assert counts["total_queries"] == 1
    assert counts["dynamic_attempted"] == 1
    assert counts["dynamic_candidates"] == 1
    assert counts["accepted"] == 1
    assert "category:exact_match" not in counts


def test_benchmark_runtime_slow_query_payload_orders_by_mean() -> None:
    """Slow-query payload should identify the largest mean latency first."""
    values = {
        ("en", "exact_match", "fast"): [0.001, 0.002],
        ("en", "extra_words", "slow"): [0.010, 0.020],
    }
    meta = {
        ("en", "extra_words", "slow"): {
            "tags": ("dynamic_attempted",),
            "dynamic_candidate_count": 3,
            "accepted": True,
            "top_score": 0.9,
        }
    }

    payload = benchmark._runtime_slow_query_payload(values, meta, limit=1)

    assert payload[0]["query"] == "slow"
    assert payload[0]["dynamic_candidate_count"] == 3


def test_benchmark_accuracy_report_exposes_schema_metadata(tmp_path: Path) -> None:
    """Accuracy reports should advertise machine-readable schema changes."""
    report = {
        "report_schema": "assist_canonicalizer_accuracy",
        "report_schema_version": benchmark.ACCURACY_REPORT_SCHEMA_VERSION,
        "languages": {},
        "overall": {"summary": {}},
    }
    json_path = tmp_path / "report.json"

    benchmark._write_json_report(str(json_path), report)
    payload = orjson.loads(json_path.read_bytes())
    markdown = benchmark._markdown_report(report)
    text = benchmark._text_report(report)

    assert payload["report_schema"] == "assist_canonicalizer_accuracy"
    assert payload["report_schema_version"] == benchmark.ACCURACY_REPORT_SCHEMA_VERSION
    assert f"**Report schema:** v{benchmark.ACCURACY_REPORT_SCHEMA_VERSION}" in markdown
    assert f"Report Schema: v{benchmark.ACCURACY_REPORT_SCHEMA_VERSION}" in text


def test_benchmark_ablation_payload_keeps_only_meaningful_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ablations should exclude ineligible cases and serialize lossless counters."""
    ablations = benchmark._new_ablation_results()
    selected = RankedCandidate(
        candidate=Candidate(
            text="turn on kitchen light",
            intent_name="HassTurnOn",
            language="en",
        ),
        scores=ScoreBreakdown(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    base_case = {
        "query": "turn on kitchen light",
        "expected_canonical": "turn on kitchen light",
        "expected_intent": "HassTurnOn",
        "category": "test",
    }

    benchmark._record_ablations(ablations, (selected,), True, base_case, [{}], language="en")
    benchmark._record_ablations(
        ablations,
        (selected,),
        False,
        {**base_case, "expected_fallback": True},
        [{}],
        language="en",
    )
    benchmark._record_ablations(
        ablations,
        (selected,),
        True,
        {**base_case, "drift": True},
        [{}],
        language="en",
    )
    benchmark._record_ablations(ablations, (), False, base_case, [{}], language="en")

    payload = benchmark._ablation_payload(ablations)
    assert payload["cohort"] == {
        "dataset_cases": 4,
        "evaluated": 3,
        "production_fallbacks": 2,
        "excluded": {
            "drift": 1,
        },
    }
    component_counts = payload["categories"]["test"]["components"]["rapidfuzz"]
    assert component_counts == {
        "canonical_correct": 1,
        "intent_correct": 1,
        "slots_correct": 1,
        "intent_slots_correct": 1,
    }
    assert "fallback" not in component_counts
    assert "average_latency_ms" not in component_counts

    benchmark._print_ablation_table("Production-Flow Component Top-1: TEST", ablations)
    console_output = capsys.readouterr().out
    text_output = "\n".join(
        benchmark._text_ablation_table(
            "Production-Flow Component Top-1: TEST",
            payload,
        )
    )

    assert "Fallback" not in console_output
    assert "Fallback" not in text_output
    assert "Cohort: 3/4 evaluated" in console_output
    assert "production_fallbacks=2" in console_output
    assert "Cohort: 3/4 evaluated" in text_output
    assert "production_fallbacks=2" in text_output
    assert "Exact Canonical" in console_output
    assert "Exact Canonical" in text_output
    assert benchmark.ABLATION_METRIC_NOTE not in console_output
    assert benchmark.ABLATION_METRIC_NOTE not in text_output
    assert "Intent/Slot" in console_output
    assert "Intent/Slot" in text_output

    console_lines = console_output.splitlines()
    table_start = next(index for index, line in enumerate(console_lines) if line.startswith("-"))
    assert len({len(line) for line in console_lines[table_start:]}) == 1

    benchmark._print_ablation_table(
        "Production-Flow Component Top-1: ALL LANGUAGES",
        ablations,
        show_metric_note=True,
    )
    aggregate_console = capsys.readouterr().out
    aggregate_text = "\n".join(
        benchmark._text_ablation_table(
            "Production-Flow Component Top-1: ALL LANGUAGES",
            payload,
            show_metric_note=True,
        )
    )
    markdown_lines = benchmark._markdown_ablation_lines(payload)
    markdown_table = [line for line in markdown_lines if line.startswith("|")]
    assert aggregate_console.count(benchmark.ABLATION_METRIC_NOTE) == 1
    assert aggregate_text.count(benchmark.ABLATION_METRIC_NOTE) == 1
    assert markdown_lines.count(benchmark.ABLATION_METRIC_NOTE) == 1
    assert len({len(line) for line in markdown_table}) == 1
    assert markdown_table[0].startswith("| Component")
    assert markdown_table[-1].startswith("| `final`")


def test_benchmark_final_ablation_preserves_production_rank_order() -> None:
    """Final component selection must be the candidate production ranked first."""
    scores = ScoreBreakdown(0.8, 0.8, 0.8, 0.8, 0.8)
    production_top = RankedCandidate(
        candidate=Candidate(
            text="device turn off",
            intent_name="HassTurnOff",
            source=CandidateSource.GENERATED_SAMPLE,
        ),
        scores=scores,
    )
    more_trusted_tie = RankedCandidate(
        candidate=Candidate(
            text="turn off device",
            intent_name="HassTurnOff",
            source=CandidateSource.BUILT_IN,
        ),
        scores=scores,
    )

    selected = benchmark._select_ablation_candidate(
        (production_top, more_trusted_tie),
        "final",
    )

    assert selected is production_top


def test_benchmark_ablation_candidate_with_none_scores() -> None:
    """Ablation selection must filter out None component scores and handle all-None fallback."""
    # Candidate 1: Rapidfuzz score is None, but high final score
    c1 = RankedCandidate(
        candidate=Candidate(
            text="c1",
            intent_name="HassTurnOff",
            source=CandidateSource.GENERATED_SAMPLE,
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=cast(Any, None),
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=0.9,
            final_score=0.9,
        ),
    )
    # Candidate 2: Rapidfuzz score is 0.5, lower final score
    c2 = RankedCandidate(
        candidate=Candidate(
            text="c2",
            intent_name="HassTurnOff",
            source=CandidateSource.GENERATED_SAMPLE,
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.5,
            char_ngram_score=0.8,
            bm25_score=0.8,
            intent_score=0.8,
            final_score=0.8,
        ),
    )

    # When component is rapidfuzz, c1 has None, c2 has 0.5.
    # It should filter out c1 and return c2.
    selected = benchmark._select_ablation_candidate((c1, c2), "rapidfuzz")
    assert selected is c2

    # When all candidates have None component scores, it should fallback to ranked[0].
    # Let's say rapidfuzz score for both c1 and c3 is None.
    c3 = RankedCandidate(
        candidate=Candidate(
            text="c3",
            intent_name="HassTurnOff",
            source=CandidateSource.GENERATED_SAMPLE,
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=cast(Any, None),
            char_ngram_score=0.7,
            bm25_score=0.7,
            intent_score=0.7,
            final_score=0.7,
        ),
    )
    selected_all_none = benchmark._select_ablation_candidate((c1, c3), "rapidfuzz")
    assert selected_all_none is c1


def test_benchmark_case_accounting_preserves_production_selection() -> None:
    """Expected answers must assess, not replace, the production-selected candidate."""
    stats = benchmark.CategoryStats()
    production_selected = RankedCandidate(
        candidate=Candidate(text="device turn off", intent_name="HassTurnOff", language="en"),
        scores=ScoreBreakdown(0.8, 0.8, 0.8, 0.8, 0.8),
    )

    is_ok, reason, _slots, selected = benchmark._record_case_result(
        stats,
        production_selected,
        "turn off device",
        "HassTurnOff",
        [{}],
    )

    assert not is_ok
    assert reason == "canonical"
    assert selected is production_selected
    assert stats.correct == 0
    assert stats.intent_correct == 1


def test_benchmark_hassil_missing_list_retry_fails_on_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated MissingListError for the same list should fail instead of looping."""
    calls = 0

    def fake_recognize_all(*_args: Any, **_kwargs: Any) -> list[Any]:
        nonlocal calls
        calls += 1
        raise hassil.errors.MissingListError("Missing list {missing_list}")

    monkeypatch.setattr(benchmark.hassil, "recognize_all", fake_recognize_all)

    with pytest.raises(hassil.errors.MissingListError):
        benchmark.run_hassil_recognize_all("hello", cast(hassil.intents.Intents, object()), {})

    assert calls == 2


def test_benchmark_validate_test_cases_accepts_context() -> None:
    """Dataset validation should preserve optional HassIL intent context."""
    cases = [
        {
            "query": "mute",
            "expected_intent": "HassMediaPlayerMute",
            "expected_canonical": "mute",
            "expected_slots": {},
            "category": "intent_coverage",
            "context": {"area": "kitchen", "floor": 1, "enabled": True},
        }
    ]

    validated = benchmark._validate_test_cases(cases, "en", "test.json")

    assert validated[0]["context"] == {"area": "kitchen", "floor": 1, "enabled": True}


def test_benchmark_validate_test_cases_accepts_expected_fallback() -> None:
    """Dataset validation should preserve an explicit fallback expectation."""
    cases = [
        {
            "query": "turn kitchen light",
            "expected_intent": "HassTurnOn",
            "expected_canonical": "turn on kitchen light",
            "category": "missing_words",
            "expected_fallback": True,
        }
    ]

    validated = benchmark._validate_test_cases(cases, "en", "test.json")

    assert validated[0]["expected_fallback"] is True


def test_benchmark_validate_test_cases_rejects_invalid_expected_fallback() -> None:
    """Fallback expectations must be explicit booleans."""
    cases = [
        {
            "query": "turn kitchen light",
            "expected_intent": "HassTurnOn",
            "expected_canonical": "turn on kitchen light",
            "category": "missing_words",
            "expected_fallback": "yes",
        }
    ]

    with pytest.raises(ValueError, match="expected_fallback must be a boolean"):
        benchmark._validate_test_cases(cases, "en", "test.json")


def test_benchmark_validate_test_cases_rejects_invalid_context() -> None:
    """Dataset context must be a flat object with scalar values."""
    cases = [
        {
            "query": "mute",
            "expected_intent": "HassMediaPlayerMute",
            "expected_canonical": "mute",
            "expected_slots": {},
            "category": "intent_coverage",
            "context": {"area": ["kitchen"]},
        }
    ]

    with pytest.raises(ValueError, match="context entry 'area'"):
        benchmark._validate_test_cases(cases, "en", "test.json")


def test_benchmark_validate_test_cases_rejects_duplicate_commands() -> None:
    """Dataset validation must expose duplicate commands instead of hiding them."""
    base_case = {
        "query": "turn on the light",
        "expected_intent": "HassTurnOn",
        "expected_canonical": "turn on light",
        "expected_slots": {"name": "light"},
        "category": "supported_filler",
    }
    cases = [base_case, {**base_case, "query": "  TURN on the light  "}]

    with pytest.raises(ValueError, match=r"case #2 duplicates query from case #1"):
        benchmark._validate_test_cases(cases, "en", "test.json")


def test_benchmark_run_hassil_recognize_all_passes_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark HassIL recognition should forward per-case context."""
    captured: list[Mapping[str, Any] | None] = []

    def fake_recognize_all(*_args: Any, **kwargs: Any) -> list[Any]:
        captured.append(kwargs.get("intent_context"))
        return []

    monkeypatch.setattr(benchmark.hassil, "recognize_all", fake_recognize_all)

    benchmark.run_hassil_recognize_all(
        "mute",
        cast(hassil.intents.Intents, object()),
        {},
        {"area": "kitchen"},
    )

    assert captured == [{"area": "kitchen"}]


def test_benchmark_run_hassil_recognize_best_matches_home_assistant_preferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-result recognition should use Home Assistant's strict-match preferences."""
    captured: dict[str, Any] = {}
    expected = object()

    def fake_recognize_best(*_args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(benchmark.hassil, "recognize_best", fake_recognize_best)

    result = benchmark.run_hassil_recognize_best(
        "turn on light",
        cast(hassil.intents.Intents, object()),
        {},
        {"area": "kitchen"},
        "en",
    )

    assert result is expected
    assert captured["best_metadata_key"] == "hass_custom_sentence"
    assert captured["best_slot_name"] == "name"
    assert captured["intent_context"] == {"area": "kitchen"}
    assert captured["language"] == "en"


def test_benchmark_lexical_mode_uses_production_hassil_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production evaluation must not rank a command already accepted by HassIL."""
    hassil_result = SimpleNamespace(
        intent=SimpleNamespace(name="HassTurnOn"),
        entities={"name": SimpleNamespace(value="light")},
    )
    monkeypatch.setattr(
        benchmark,
        "run_hassil_recognize_best",
        lambda *_args, **_kwargs: hassil_result,
    )

    class FailingRuntime:
        """Fail if the production shortcut incorrectly reaches ranking."""

        def rank_with_dynamic_candidates(self, *_args: Any, **_kwargs: Any) -> None:
            """Raise because shortcut cases must never enter candidate ranking."""
            raise AssertionError("ranking must be bypassed")

    ranked = benchmark._evaluate_mode_candidates(
        "lexical",
        "please turn on the light",
        "en",
        FailingRuntime(),
        object(),
        object(),
        {},
        None,
        "turn on the light",
        "HassTurnOn",
        [{"name": "light"}],
    )

    assert ranked[0].candidate.intent_name == "HassTurnOn"
    assert ranked[0].candidate.metadata["evaluation_path"] == "hassil_shortcut"


def test_benchmark_lexical_mode_ranks_after_production_hassil_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production evaluation must rank only when raw HassIL recognition fails."""
    monkeypatch.setattr(
        benchmark,
        "run_hassil_recognize_best",
        lambda *_args, **_kwargs: None,
    )
    expected_ranked = cast(tuple[RankedCandidate, ...], (object(),))
    runtime = SimpleNamespace(
        rank_with_dynamic_candidates=lambda *_args, **_kwargs: expected_ranked
    )

    ranked = benchmark._evaluate_mode_candidates(
        "lexical",
        "unsupported filler turn on the light",
        "en",
        runtime,
        object(),
        object(),
        {},
        None,
        "turn on the light",
        "HassTurnOn",
        [{"name": "light"}],
    )

    assert ranked is expected_ranked


def test_benchmark_expected_fallback_uses_ordinary_fallback_accounting() -> None:
    """Count deliberate fallback in the common denominator and fallback bucket."""
    safe_stats = benchmark.CategoryStats()

    is_ok, reason, _slots, selected = benchmark._record_case_result(
        safe_stats,
        None,
        "turn on kitchen light",
        "HassTurnOn",
        [{}],
        expected_fallback=True,
    )

    assert is_ok
    assert reason == "expected_fallback"
    assert selected is None
    assert safe_stats.total == 1
    assert safe_stats.fallback == 1
    assert safe_stats.intent_slots_correct == 0
    assert safe_stats.mismatch == 0

    unsafe_stats = benchmark.CategoryStats()
    unsafe_candidate = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(1.0, 1.0, 1.0, 1.0, 1.0),
    )

    is_ok, reason, _slots, selected = benchmark._record_case_result(
        unsafe_stats,
        unsafe_candidate,
        "turn on kitchen light",
        "HassTurnOn",
        [{}],
        expected_fallback=True,
    )

    assert not is_ok
    assert reason == "unsafe_selection"
    assert selected is unsafe_candidate
    assert unsafe_stats.total == 1
    assert unsafe_stats.mismatch == 1


def test_benchmark_case_row_uses_equivalent_list_intents() -> None:
    """Report list-intent aliases with the same equivalence used by accounting."""
    selected = RankedCandidate(
        candidate=Candidate(
            text="add milk to the list",
            intent_name="HassListAddItem",
            language="en",
        ),
        scores=ScoreBreakdown(1.0, 1.0, 1.0, 1.0, 1.0),
    )

    row = benchmark._case_row(
        "en",
        "lexical",
        {
            "query": "add milk to the list",
            "expected_canonical": "add milk to the list",
            "expected_intent": "HassShoppingListAddItem",
            "category": "intent_coverage",
        },
        selected,
        "",
        {"item": "milk"},
        0.0,
        {},
        [{"item": "milk"}],
    )

    assert row["intent_ok"] is True
    assert row["intent_slots_ok"] is True
    assert row["outcome_ok"] is True


def test_benchmark_thresholds_cover_language_accuracy_fallback_and_mismatch() -> None:
    """Apply every retained quality dimension to a labeled threshold scope."""
    stats = benchmark.CategoryStats(
        total=10,
        intent_slots_correct=7,
        fallback=1,
    )

    failures = benchmark._threshold_failures(
        stats,
        min_intent_slot_accuracy=80.0,
        max_fallback_rate=9.0,
        max_mismatch_rate=15.0,
        scope="NL",
    )

    assert failures == [
        "NL: intent/slot accuracy 70.0% is below 80.0%",
        "NL: fallback rate 10.0% is above 9.0%",
        "NL: mismatch rate 20.0% is above 15.0%",
    ]


def test_benchmark_profile_regressions_are_labeled_by_target() -> None:
    """Collect regression messages across independently profiled targets."""
    assert benchmark._profile_regressions(
        {
            "rank": {"regressions": ["REGRESSION [aggregate.p95]"]},
            "runtime": {"regressions": ["REGRESSION [scenario_stats.rejected.p99]"]},
        }
    ) == [
        "rank: REGRESSION [aggregate.p95]",
        "runtime: REGRESSION [scenario_stats.rejected.p99]",
    ]


def test_dataset_hassil_cache_includes_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dataset HassIL cache should not reuse one query across different contexts."""
    calls: list[dict[str, Any]] = []

    def fake_run_hassil(
        query: str,
        _intents: hassil.intents.Intents,
        _slot_lists: dict[str, hassil.intents.SlotList],
        intent_context: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        assert query == "mute"
        context = dict(intent_context or {})
        calls.append(context)
        return [context]

    monkeypatch.setattr(benchmark, "run_hassil_recognize_all", fake_run_hassil)
    context = cast(
        DatasetContext,
        SimpleNamespace(
            hassil_results_by_query={},
            intents=cast(hassil.intents.Intents, object()),
            slot_lists={},
        ),
    )

    kitchen = _hassil_results(context, "mute", {"area": "kitchen"})
    office = _hassil_results(context, "mute", {"area": "office"})
    kitchen_again = _hassil_results(context, "mute", {"area": "kitchen"})

    assert kitchen == ({"area": "kitchen"},)
    assert office == ({"area": "office"},)
    assert kitchen_again == kitchen
    assert calls == [{"area": "kitchen"}, {"area": "office"}]


def test_slots_match_uses_alias_when_primary_slot_is_sentinel() -> None:
    """Allow namespaced slot aliases to satisfy generic expected entity slots."""
    assert benchmark._slots_match(
        {
            "domain": "fan",
            "name": "all",
            "name:fan": "bathroom fan",
            "area": "bathroom",
        },
        {"name": "bathroom fan"},
    )


def test_slots_match_accepts_compound_name_location_decomposition() -> None:
    """Allow HassIL-style name+location decomposition for compound entity labels."""
    assert benchmark._slots_match(
        {"name": "đèn", "area": "hành lang"},
        {"name": "đèn hành lang"},
    )
    assert not benchmark._slots_match(
        {"name": "đèn", "area": "phòng khách"},
        {"name": "đèn hành lang"},
    )
    assert not benchmark._slots_match(
        {"name": "all", "area": "hallway"},
        {"name": "hallway light"},
    )


def test_select_accepted_with_gate_diagnostics() -> None:
    """Verify that _select_accepted_with_gate exposes structured reasons."""
    # 1. Empty ranked list
    res, diag = benchmark._select_accepted_with_gate(())
    assert res is None
    assert diag["reason"] == FallbackReason.NO_CANDIDATE.value

    # 2. Accepted candidate (score above confidence threshold)
    cand_1 = Candidate(text="turn on light", intent_name="HassTurnOn")
    accepted_score = DEFAULT_MIN_CONFIDENCE + 0.2
    rc_1 = RankedCandidate(
        candidate=cand_1,
        scores=ScoreBreakdown(
            rapidfuzz_score=accepted_score,
            char_ngram_score=accepted_score,
            bm25_score=accepted_score,
            intent_score=1.0,
            final_score=accepted_score,
        ),
    )
    res, diag = benchmark._select_accepted_with_gate((rc_1,))
    assert res is rc_1
    assert diag["reason"] == "accepted"

    # 3. Low confidence rejection (score below confidence threshold)
    low_conf_score = DEFAULT_MIN_CONFIDENCE - 0.05
    rc_low = RankedCandidate(
        candidate=cand_1,
        scores=ScoreBreakdown(
            rapidfuzz_score=low_conf_score,
            char_ngram_score=low_conf_score,
            bm25_score=low_conf_score,
            intent_score=0.1,
            final_score=low_conf_score,
        ),
    )
    res, diag = benchmark._select_accepted_with_gate((rc_low,))
    assert res is None
    assert diag["reason"] == FallbackReason.LOW_CONFIDENCE.value

    # 4. Low margin rejection (competing score within min margin of top score)
    cand_2 = Candidate(text="turn off light", intent_name="HassTurnOff")
    winner_score = DEFAULT_MIN_CONFIDENCE + 0.1
    competitor_score = winner_score - (DEFAULT_MIN_MARGIN / 2)

    rc_winner = RankedCandidate(
        candidate=cand_1,
        scores=ScoreBreakdown(
            rapidfuzz_score=winner_score,
            char_ngram_score=winner_score,
            bm25_score=winner_score,
            intent_score=1.0,
            final_score=winner_score,
        ),
    )
    rc_competitor = RankedCandidate(
        candidate=cand_2,
        scores=ScoreBreakdown(
            rapidfuzz_score=competitor_score,
            char_ngram_score=competitor_score,
            bm25_score=competitor_score,
            intent_score=1.0,
            final_score=competitor_score,
        ),
    )
    res, diag = benchmark._select_accepted_with_gate((rc_winner, rc_competitor))
    assert res is None
    assert diag["reason"] == FallbackReason.LOW_MARGIN.value
