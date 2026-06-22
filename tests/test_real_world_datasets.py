"""Quality checks for real-world evaluation datasets."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import hassil
import hassil.errors
import orjson
import pytest

from custom_components.assist_canonicalizer import ranking
from custom_components.assist_canonicalizer.candidate import Candidate, slot_alias_values_by_key
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
    }
)
REQUIRED_CATEGORY_MINIMUMS = {
    "complex_distortion": 5,
    "exact_match": 10,
    "extra_words": 5,
    "intent_coverage": 15,
    "missing_words": 5,
    "semantic_challenge": 5,
    "spelling_mistake": 5,
    "synonym_paraphrase": 5,
}


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
    hassil_results_by_query: dict[str, tuple[Any, ...]]

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
    assert not missing, f"{dataset_context.language}: category counts too low: {missing}"


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
        expected_slot_keys = frozenset(case.get("expected_slots", {}))
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
        hassil_ok = _recognizes_expected(dataset_context, case)
        if exact_static or hassil_ok:
            failures.append(
                {
                    "query": case["query"],
                    "category": case["category"],
                    "exact_static": exact_static,
                    "hassil_ok": hassil_ok,
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
def test_real_world_extra_words_are_not_exact_candidates(
    dataset_context: DatasetContext,
) -> None:
    """Assert extra_words cases contain text beyond a static canonical command."""
    failures = []
    for case in dataset_context.cases:
        if case["category"] != "extra_words":
            continue
        if benchmark.normalize_text(case["query"]) in dataset_context.static_normalized_texts:
            failures.append(case["query"])
    assert not failures, f"{dataset_context.language}: extra_words exact candidates: {failures}"


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
        results = _hassil_results(dataset_context, case["query"])
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
            for result in _hassil_results(dataset_context, case["query"])
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


def _recognizes_expected(context: DatasetContext, case: Mapping[str, Any]) -> bool:
    """Return whether HassIL directly recognizes a case as the expected result."""
    results = _hassil_results(context, case["query"])
    expected_slots = case.get("expected_slots", {})
    return any(
        benchmark._intents_match(result.intent.name, case["expected_intent"])
        and benchmark._slots_match(
            {name: entity.value for name, entity in result.entities.items()},
            expected_slots,
            language=context.language,
        )
        for result in results
    )


def _hassil_results(context: DatasetContext, query: str) -> tuple[Any, ...]:
    """Return cached HassIL recognition results for a dataset query."""
    cached = context.hassil_results_by_query.get(query)
    if cached is not None:
        return cached
    results = tuple(benchmark.run_hassil_recognize_all(query, context.intents, context.slot_lists))
    context.hassil_results_by_query[query] = results
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
    assert diag["reason"] == FallbackReason.LOW_CONFIDENCE.value

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
