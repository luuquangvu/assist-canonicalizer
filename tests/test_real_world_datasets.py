"""Quality checks for real-world evaluation datasets."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hassil
import orjson
import pytest

from custom_components.assist_canonicalizer.grammar_loader import (
    build_candidates_from_intent_sources,
    build_query_registry_candidates,
)
from tools import evaluate_metrics

evaluate_metrics._bootstrap_project_imports()

DATASET_DIR = Path("tests/real_world")
_LANGUAGES: tuple[str, ...] | None = None


def _discover_languages() -> tuple[str, ...]:
    """Discover dataset languages by scanning ``tests/real_world/*.json``."""
    global _LANGUAGES
    if _LANGUAGES is not None:
        return _LANGUAGES
    repo_root = str(Path(__file__).resolve().parent.parent)
    safe_dir = evaluate_metrics._sanitize_path(repo_root, str(DATASET_DIR))
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


@dataclass(frozen=True, slots=True)
class DatasetContext:
    """Reusable HassIL context for one real-world dataset."""

    language: str
    cases: tuple[dict[str, Any], ...]
    sources: dict[str, Mapping[str, Any]]
    slots: dict[str, tuple[str, ...]]
    static_candidate_pairs: frozenset[tuple[str, str]]
    static_normalized_texts: frozenset[str]
    intents: hassil.intents.Intents
    slot_lists: dict[str, hassil.intents.SlotList]


@pytest.fixture(scope="module", params=_discover_languages())
def dataset_context(request: pytest.FixtureRequest) -> DatasetContext:
    """Return validated dataset context for one language."""
    language = str(request.param)
    path = DATASET_DIR / f"{language}.json"
    data = orjson.loads(path.read_text(encoding="utf-8"))
    raw_cases = data["test_cases"]
    cases = tuple(evaluate_metrics._validate_test_cases(raw_cases, language, str(path)))
    slots = evaluate_metrics._dataset_registry_slots(data, language)
    sources = evaluate_metrics.load_language_intent_sources(language)
    candidates = build_candidates_from_intent_sources(language, sources, slots)
    merged_intents: dict[str, Any] = {}
    for source in sources.values():
        hassil.merge_dict(merged_intents, source)
    return DatasetContext(
        language=language,
        cases=cases,
        sources=sources,
        slots=slots,
        static_candidate_pairs=frozenset(
            (candidate.intent_name, candidate.text) for candidate in candidates
        ),
        static_normalized_texts=frozenset(candidate.normalized_text for candidate in candidates),
        intents=hassil.intents.Intents.from_dict(merged_intents),
        slot_lists=evaluate_metrics.make_hassil_slot_lists(slots),
    )


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


def test_real_world_expected_canonicals_are_hassil_candidates(
    dataset_context: DatasetContext,
) -> None:
    """Assert expected canonical commands come from generated HassIL candidates."""
    dynamic_cache: dict[str, frozenset[tuple[str, str]]] = {}
    missing = []
    for case in dataset_context.cases:
        if not _has_expected_candidate(dataset_context, case, dynamic_cache):
            missing.append((case["query"], case["expected_intent"], case["expected_canonical"]))
    assert not missing, f"{dataset_context.language}: expected canonical missing: {missing}"


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


def test_real_world_hard_categories_are_not_direct_hassil_matches(
    dataset_context: DatasetContext,
) -> None:
    """Assert hard categories are not already handled directly by HassIL."""
    failures = []
    for case in dataset_context.cases:
        if case["category"] not in HARD_CATEGORIES:
            continue
        exact_static = (
            evaluate_metrics.normalize_text(case["query"])
            in dataset_context.static_normalized_texts
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
    assert not failures, f"{dataset_context.language}: hard category too easy: {failures}"


def test_real_world_extra_words_are_not_exact_candidates(
    dataset_context: DatasetContext,
) -> None:
    """Assert extra_words cases contain text beyond a static canonical command."""
    failures = []
    for case in dataset_context.cases:
        if case["category"] != "extra_words":
            continue
        if (
            evaluate_metrics.normalize_text(case["query"])
            in dataset_context.static_normalized_texts
        ):
            failures.append(case["query"])
    assert not failures, f"{dataset_context.language}: extra_words exact candidates: {failures}"


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
    different slot names) are not errors — they reflect genuine
    differences in the intent matching space.
    """
    dynamic_cache: dict[str, set[tuple[str, str]]] = {}
    failures: list[dict[str, str]] = []
    for case in dataset_context.cases:
        if case["category"] not in HASSIL_ALIGN_CATEGORIES:
            continue
        results = evaluate_metrics.run_hassil_recognize_all(
            case["query"], dataset_context.intents, dataset_context.slot_lists
        )
        if not results:
            continue
        top_result = results[0]
        if top_result.intent.name == case["expected_intent"]:
            continue
        canonical = case["expected_canonical"]
        cache_key = f"{top_result.intent.name}::{canonical}"
        if cache_key not in dynamic_cache:
            candidates = build_query_registry_candidates(
                dataset_context.language,
                dataset_context.sources,
                dataset_context.slots,
                canonical,
            )
            dynamic_cache[cache_key] = {(c.intent_name, c.text) for c in candidates}
        if (top_result.intent.name, canonical) in dynamic_cache[cache_key]:
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
    ``alle``), the slot is accepted — the canonicalizer extracts
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


def test_real_world_expected_slots_align_with_hassil(
    dataset_context: DatasetContext,
) -> None:
    """Assert expected_slots match HassIL entities for align categories.

    For exact_match and intent_coverage, when HassIL recognizes the
    expected intent, every key in *expected_slots* that also appears
    in HassIL entities is compared — a mismatch means the dataset
    label is wrong and must be corrected to match HassIL ground truth.

    HassIL entity values not present literally in the query text are
    treated as computed domain-level sentinels and skipped (e.g.
    ``name=all`` for ``"tắt quạt"``), since the canonicalizer extracts
    literal slot text while HassIL resolves domain operations.

    Keys present only in one system are skipped — naming conventions
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
        results = evaluate_metrics.run_hassil_recognize_all(
            case["query"], dataset_context.intents, dataset_context.slot_lists
        )
        hassil_entities_by_intent: dict[str, dict[str, Any]] = {}
        for r in results:
            hassil_entities_by_intent.setdefault(r.intent.name, {}).update(
                {name: ent.value for name, ent in r.entities.items()}
            )
        hassil_entities = hassil_entities_by_intent.get(case["expected_intent"])
        if hassil_entities is None:
            continue
        query = case["query"]
        mismatched: list[str] = []
        for key, expected_value in expected_slots.items():
            hassil_value = hassil_entities.get(key)
            if hassil_value is None:
                continue
            if not _slot_value_matches(expected_value, hassil_value, query):
                mismatched.append(f"{key}: expected={expected_value!r}, hassil={hassil_value!r}")
        if mismatched:
            failures.append(
                {
                    "query": case["query"],
                    "category": case["category"],
                    "expected_intent": case["expected_intent"],
                    "mismatches": ", ".join(mismatched),
                }
            )
    assert not failures, f"{dataset_context.language}: expected_slots mismatch HassIL: {failures}"


def _has_expected_candidate(
    context: DatasetContext,
    case: Mapping[str, Any],
    dynamic_cache: dict[str, frozenset[tuple[str, str]]],
) -> bool:
    """Return whether a case expected command exists in static or dynamic candidates."""
    pair = (case["expected_intent"], case["expected_canonical"])
    if pair in context.static_candidate_pairs:
        return True
    canonical = case["expected_canonical"]
    if canonical not in dynamic_cache:
        dynamic_cache[canonical] = frozenset(
            (candidate.intent_name, candidate.text)
            for candidate in build_query_registry_candidates(
                context.language,
                context.sources,
                context.slots,
                canonical,
            )
        )
    return pair in dynamic_cache[canonical]


def _recognizes_expected(context: DatasetContext, case: Mapping[str, Any]) -> bool:
    """Return whether HassIL directly recognizes a case as the expected result."""
    results = evaluate_metrics.run_hassil_recognize_all(
        case["query"], context.intents, context.slot_lists
    )
    expected_slots = case.get("expected_slots", {})
    return any(
        result.intent.name == case["expected_intent"]
        and evaluate_metrics._slots_match(
            {name: entity.value for name, entity in result.entities.items()},
            expected_slots,
        )
        for result in results
    )
