"""Quality checks for real-world evaluation datasets."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hassil
import hassil.errors
import orjson
import pytest

from custom_components.assist_canonicalizer.grammar_loader import (
    build_candidates_from_intent_sources,
    build_query_registry_candidates,
)
from tools import evaluate_metrics

evaluate_metrics._bootstrap_project_imports()

DATASET_DIR = Path("tests/real_world")
LANGUAGES = ("de", "en", "fr", "nl", "vi")
HARD_CATEGORIES = frozenset(
    {
        "complex_distortion",
        "missing_words",
        "semantic_challenge",
        "spelling_mistake",
        "synonym_paraphrase",
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


@pytest.fixture(scope="module", params=LANGUAGES)
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
        slot_lists=_slot_lists(slots),
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


def _slot_lists(slots: Mapping[str, tuple[str, ...]]) -> dict[str, hassil.intents.SlotList]:
    """Return HassIL slot lists from registry slot fixtures."""
    lists: dict[str, hassil.intents.SlotList] = {}
    for slot_name, values in slots.items():
        lists[slot_name] = hassil.TextSlotList(
            name=slot_name,
            values=[
                hassil.TextSlotValue(
                    text_in=hassil.parse_sentence(value).expression,
                    value_out=value,
                )
                for value in values
            ],
        )
    return lists


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
    slot_lists = dict(context.slot_lists)
    while True:
        try:
            results = list(
                hassil.recognize_all(
                    case["query"],
                    context.intents,
                    slot_lists=slot_lists,
                )
            )
            break
        except hassil.errors.MissingListError as err:
            match = re.search(r"\{([^}]+)\}", str(err))
            if match is None:
                raise
            list_name = match.group(1)
            slot_lists[list_name] = hassil.TextSlotList(list_name, [])
    expected_slots = case.get("expected_slots", {})
    return any(
        result.intent.name == case["expected_intent"]
        and evaluate_metrics._slots_match(
            {name: entity.value for name, entity in result.entities.items()},
            expected_slots,
        )
        for result in results
    )
