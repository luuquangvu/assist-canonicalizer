"""CLI tool to evaluate Assist Canonicalizer matching metrics on real-world test cases.

Loads candidates from actual HA built-in intents and evaluates lexical matching.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hassil
import hassil.errors
import orjson

if TYPE_CHECKING:
    from custom_components.assist_canonicalizer.ranking import RankedCandidate

_BOOTSTRAPPED = False


DEFAULT_MIN_CONFIDENCE: float = 0.0
DEFAULT_MIN_MARGIN: float = 0.0
CanonicalizerRuntime: Any = None
build_candidates_from_intent_sources: Any = None
build_index: Any = None
load_language_intent_sources: Any = None
normalize_text: Any = None
_RankedCandidate: Any = None
_ScoreBreakdown: Any = None
Candidate: Any = None
FallbackReason: Any = None


def _bootstrap_project_imports() -> None:
    """Import project modules after adding the repository root to sys.path."""
    global _BOOTSTRAPPED
    global DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_MARGIN
    global CanonicalizerRuntime
    global build_candidates_from_intent_sources, build_index, load_language_intent_sources
    global normalize_text, _RankedCandidate, _ScoreBreakdown, Candidate, FallbackReason

    if _BOOTSTRAPPED:
        return
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)

    from custom_components.assist_canonicalizer import const as const_module
    from custom_components.assist_canonicalizer.builtin_intents import (
        load_language_intent_sources as imported_load_language_intent_sources,
    )
    from custom_components.assist_canonicalizer.candidate import (
        Candidate as ImportedCandidate,
    )
    from custom_components.assist_canonicalizer.grammar_loader import (
        build_candidates_from_intent_sources as imported_build_candidates_from_intent_sources,
    )
    from custom_components.assist_canonicalizer.indexer import build_index as imported_build_index
    from custom_components.assist_canonicalizer.normalization import (
        normalize_text as imported_normalize_text,
    )
    from custom_components.assist_canonicalizer.ranking import (
        RankedCandidate as ImportedRankedCandidate,
    )
    from custom_components.assist_canonicalizer.ranking import (
        ScoreBreakdown as ImportedScoreBreakdown,
    )
    from custom_components.assist_canonicalizer.runtime import (
        CanonicalizerRuntime as ImportedCanonicalizerRuntime,
    )

    DEFAULT_MIN_CONFIDENCE = const_module.DEFAULT_MIN_CONFIDENCE
    DEFAULT_MIN_MARGIN = const_module.DEFAULT_MIN_MARGIN
    CanonicalizerRuntime = ImportedCanonicalizerRuntime
    build_candidates_from_intent_sources = imported_build_candidates_from_intent_sources
    build_index = imported_build_index
    load_language_intent_sources = imported_load_language_intent_sources
    normalize_text = imported_normalize_text
    _RankedCandidate = ImportedRankedCandidate
    _ScoreBreakdown = ImportedScoreBreakdown
    Candidate = ImportedCandidate
    FallbackReason = const_module.FallbackReason
    _BOOTSTRAPPED = True


REGISTRY_SLOTS: dict[str, dict[str, tuple[str, ...]]] = {
    "en": {
        "name": (
            "Living Room Light",
            "Living Room Fan",
            "Bedroom Light",
            "Kitchen Light",
        ),
    },
    "vi": {
        "name": (
            "đèn phòng khách",
            "quạt phòng khách",
            "đèn phòng ngủ",
        ),
    },
    "de": {
        "name": (
            "Wohnzimmerlampe",
            "Badezimmerlüfter",
            "Schlafzimmerlampe",
        ),
    },
    "fr": {
        "name": (
            "lumière du salon",
            "ventilateur du salon",
            "lumière de la chambre",
        ),
    },
    "nl": {
        "name": (
            "woonkamerlamp",
            "keukenlamp",
            "badkamerventilator",
            "stofzuiger",
        ),
    },
}


def _slots_from_candidate(selected: RankedCandidate | None) -> dict[str, Any]:
    """Return slot values from a ranked candidate."""
    if selected is None:
        return {}
    slots_text = selected.candidate.metadata.get("slots")
    if not slots_text:
        return {}
    try:
        decoded = orjson.loads(slots_text)
    except orjson.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _slots_match(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Check if actual candidate slots contain all expected slot values."""
    if not expected:
        return True
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value is None:
            return False
        if isinstance(expected_value, str) and isinstance(actual_value, str):
            if normalize_text(expected_value) != normalize_text(actual_value):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _score_value(selected: RankedCandidate | None, field: str) -> float | None:
    """Extract one score field from a ranked candidate."""
    if selected is None:
        return None
    return getattr(selected.scores, field, None)


def _metric_str(count: int, total: int) -> str:
    """Return a formatted metric cell with count and percentage."""
    if total == 0:
        return f"0/{0} (0.0%)"
    pct = (count / total) * 100
    return f"{count}/{total} ({pct:.1f}%)"


def _short_names(names: list[str]) -> str:
    """Return a shortened comma-separated intent name list."""
    short = ", ".join(list(names)[:5])
    if len(names) > 5:
        short += f" ... ({len(names)} total)"
    return short


@dataclass
class CategoryStats:
    """Aggregated metrics for one language, mode, and category."""

    total: int = 0
    correct: int = 0
    intent_correct: int = 0
    slots_correct: int = 0
    intent_slots_correct: int = 0
    fallback: int = 0
    latency_ms_total: float = 0.0

    @property
    def average_latency_ms(self) -> float:
        """Return average per-case latency in milliseconds."""
        return self.latency_ms_total / self.total if self.total else 0.0

    @property
    def canonical_accuracy(self) -> float:
        """Return percentage of exact canonical matches."""
        return (self.correct / self.total * 100) if self.total else 0.0

    @property
    def intent_accuracy(self) -> float:
        """Return percentage of correct intent matches."""
        return (self.intent_correct / self.total * 100) if self.total else 0.0

    @property
    def slots_accuracy(self) -> float:
        """Return percentage of correct slot matches."""
        return (self.slots_correct / self.total * 100) if self.total else 0.0

    @property
    def intent_slot_accuracy(self) -> float:
        """Return percentage of correct intent+slot matches."""
        return (self.intent_slots_correct / self.total * 100) if self.total else 0.0

    @property
    def mismatch(self) -> int:
        """Return count of cases that are not fallback but still wrong."""
        return self.total - self.intent_slots_correct - self.fallback

    @property
    def mismatch_rate(self) -> float:
        """Return percentage of mismatch cases."""
        return (self.mismatch / self.total * 100) if self.total else 0.0

    @property
    def fallback_rate(self) -> float:
        """Return percentage of fallback cases."""
        return (self.fallback / self.total * 100) if self.total else 0.0

    def merge(self, other: CategoryStats) -> None:
        """Merge another stats container into this one."""
        self.total += other.total
        self.correct += other.correct
        self.intent_correct += other.intent_correct
        self.slots_correct += other.slots_correct
        self.intent_slots_correct += other.intent_slots_correct
        self.fallback += other.fallback
        self.latency_ms_total += other.latency_ms_total

    def as_dict(self) -> dict[str, Any]:
        """Return serializable stats values."""
        return {
            "total": self.total,
            "canonical_accuracy": self.canonical_accuracy,
            "intent_accuracy": self.intent_accuracy,
            "slots_accuracy": self.slots_accuracy,
            "intent_slot_accuracy": self.intent_slot_accuracy,
            "mismatch_rate": self.mismatch_rate,
            "fallback_rate": self.fallback_rate,
            "average_latency_ms": self.average_latency_ms,
        }


ABLATION_COMPONENTS = (
    "rapidfuzz",
    "char_ngram",
    "bm25",
    "intent_action",
    "final",
)


def _stringify_keys(obj: object) -> object:
    """Recursively convert dictionary keys to strings."""
    if isinstance(obj, dict):
        return {str(key): _stringify_keys(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_stringify_keys(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_stringify_keys(item) for item in obj)
    return obj


def _new_results() -> dict[str, dict[str, CategoryStats]]:
    """Return an empty metrics container for all evaluation modes."""
    return {"hassil": {}, "lexical": {}}


def _new_ablation_results() -> dict[str, dict[str, CategoryStats]]:
    """Return an empty metrics container for ablation components."""
    return {component: {} for component in ABLATION_COMPONENTS}


def _stats_for(
    results: dict[str, dict[str, CategoryStats]],
    mode_name: str,
    category: str,
) -> CategoryStats:
    """Return the mutable stats object for one mode/category pair."""
    mode_stats = results[mode_name]
    if category not in mode_stats:
        mode_stats[category] = CategoryStats()
    return mode_stats[category]


def _merge_results(
    target: dict[str, dict[str, CategoryStats]],
    source: dict[str, dict[str, CategoryStats]],
) -> None:
    """Merge per-language metrics into the global metrics container."""
    for mode_name, categories in source.items():
        for category, stats in categories.items():
            _stats_for(target, mode_name, category).merge(stats)


def _intent_names_from_sources(sources: Mapping[str, Mapping[str, Any]]) -> set[str]:
    """Return intent names present in loaded Home Assistant intent sources."""
    intent_names: set[str] = set()
    for source_config in sources.values():
        intents = source_config.get("intents", {})
        if isinstance(intents, Mapping):
            intent_names.update(name for name in intents if isinstance(name, str))
    return intent_names


def _dataset_registry_slots(data: Mapping[str, Any], lang: str) -> dict[str, tuple[str, ...]]:
    """Return registry slots supplied by a dataset, falling back to built-in fixtures."""
    raw_slots = data.get("registry_slots")
    if raw_slots is None:
        return {key: tuple(values) for key, values in REGISTRY_SLOTS.get(lang, {}).items()}
    if not isinstance(raw_slots, Mapping):
        raise ValueError("registry_slots must be an object when present")
    slots: dict[str, tuple[str, ...]] = {}
    for slot_name, values in raw_slots.items():
        if not isinstance(slot_name, str) or not isinstance(values, list):
            raise ValueError("registry_slots must map string names to string lists")
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError("registry_slots values must be non-empty strings")
        slots[slot_name] = tuple(values)
    return slots


def _validate_test_cases(test_cases: list[Any], lang: str, path: str) -> list[dict[str, Any]]:
    """Validate and return real-world test cases from one dataset."""
    validated: list[dict[str, Any]] = []
    required = ("query", "expected_intent", "expected_canonical", "category")
    seen: set[tuple[str, str, str]] = set()
    for index, case in enumerate(test_cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"{path}: test case #{index} must be an object")
        missing = [
            key for key in required if not isinstance(case.get(key), str) or not case[key].strip()
        ]
        if missing:
            raise ValueError(f"{path}: test case #{index} missing fields: {missing}")
        case_lang = case.get("language", lang)
        if case_lang != lang:
            raise ValueError(
                f"{path}: test case #{index} language '{case_lang}' does not match dataset"
            )
        query = case["query"]
        expected_intent = case["expected_intent"]
        expected_canonical = case["expected_canonical"]
        expected_slots = case.get("expected_slots", {})
        if not isinstance(expected_slots, dict):
            raise ValueError(f"{path}: test case #{index} expected_slots must be an object")
        for key, value in expected_slots.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(
                    f"{path}: test case #{index} expected_slots entry {key!r} must be string→string"
                )
            if not value.strip():
                raise ValueError(
                    f"{path}: test case #{index} expected_slots entry {key!r} value is empty"
                )
        dedup_key = (query, expected_intent, expected_canonical)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        validated.append(
            {
                "query": query,
                "expected_intent": expected_intent,
                "expected_canonical": expected_canonical,
                "expected_slots": expected_slots,
                "category": case["category"],
            }
        )
    return validated


def _component_score(item: RankedCandidate, component: str) -> float | None:
    """Return a score for one ablation component."""
    if component == "rapidfuzz":
        return item.scores.rapidfuzz_score
    if component == "char_ngram":
        return item.scores.char_ngram_score
    if component == "bm25":
        return item.scores.bm25_score
    if component == "intent_action":
        return item.scores.intent_score
    if component == "final":
        return item.scores.final_score
    raise ValueError(f"Unknown ablation component: {component}")


def _select_accepted_with_gate(
    ranked: tuple[RankedCandidate, ...],
) -> tuple[RankedCandidate | None, dict[str, Any]]:
    """Return the accepted candidate plus explicit acceptance gate diagnostics."""
    if not ranked:
        return None, {
            "accepted": False,
            "reason": "empty_ranking",
            "top_score": None,
            "competing_score": None,
            "margin": None,
        }
    top_candidate = ranked[0]
    top_score = top_candidate.scores.final_score
    competing_candidate = next(
        (
            item
            for item in ranked[1:]
            if item.candidate.intent_name != top_candidate.candidate.intent_name
        ),
        None,
    )
    competing_score = (
        competing_candidate.scores.final_score if competing_candidate is not None else None
    )
    margin = top_score - competing_score if competing_score is not None else None
    if top_score < DEFAULT_MIN_CONFIDENCE:
        return None, {
            "accepted": False,
            "reason": FallbackReason.LOW_CONFIDENCE.value,
            "top_score": top_score,
            "competing_score": competing_score,
            "margin": margin,
        }
    if (
        competing_candidate is not None
        and margin is not None
        and margin < DEFAULT_MIN_MARGIN
        and not (
            _is_exact_lexical_match(top_candidate)
            and not _is_exact_lexical_match(competing_candidate)
        )
    ):
        return None, {
            "accepted": False,
            "reason": FallbackReason.LOW_MARGIN.value,
            "top_score": top_score,
            "competing_score": competing_score,
            "margin": margin,
        }
    return top_candidate, {
        "accepted": True,
        "reason": "accepted",
        "top_score": top_score,
        "competing_score": competing_score,
        "margin": margin,
    }


def _is_exact_lexical_match(ranked_candidate: RankedCandidate) -> bool:
    """Return whether a ranked candidate exactly matches query text lexically."""
    scores = ranked_candidate.scores
    return scores.rapidfuzz_score == 1.0 and scores.char_ngram_score == 1.0


def _select_ablation_candidate(
    ranked: tuple[RankedCandidate, ...],
    component: str,
) -> RankedCandidate | None:
    """Return the top candidate when ranking only by one score component."""
    scored = [
        (score, item) for item in ranked if (score := _component_score(item, component)) is not None
    ]
    if not scored:
        return None
    return max(
        scored,
        key=lambda pair: (
            pair[0],
            pair[1].scores.final_score,
            -pair[1].candidate.source_priority,
        ),
    )[1]


def _record_case_result(
    stats: CategoryStats,
    selected: RankedCandidate | None,
    expected_canonical: str,
    expected_intent: str,
    expected_slots: Mapping[str, Any],
    latency_ms: float,
) -> tuple[bool, str, dict[str, Any]]:
    """Record one evaluated case and return whether it matched completely."""
    stats.total += 1
    stats.latency_ms_total += latency_ms
    actual_slots = _slots_from_candidate(selected)
    if selected is None:
        stats.fallback += 1
        return False, "fallback", actual_slots
    is_canonical_ok = selected.candidate.text == expected_canonical
    is_intent_ok = selected.candidate.intent_name == expected_intent
    is_slots_ok = _slots_match(actual_slots, expected_slots)
    if is_canonical_ok:
        stats.correct += 1
    if is_intent_ok:
        stats.intent_correct += 1
    if is_slots_ok:
        stats.slots_correct += 1
    if is_intent_ok and is_slots_ok:
        stats.intent_slots_correct += 1
    reasons = []
    if not is_canonical_ok:
        reasons.append("canonical")
    if not is_intent_ok:
        reasons.append("intent")
    if not is_slots_ok:
        reasons.append("slots")
    return not reasons, "+".join(reasons), actual_slots


def _case_row(
    lang: str,
    mode_name: str,
    case: Mapping[str, Any],
    selected: RankedCandidate | None,
    reason: str,
    actual_slots: Mapping[str, Any],
    latency_ms: float,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one JSON row for a normal evaluation mode."""
    expected_slots = case.get("expected_slots", {})
    actual_text = selected.candidate.text if selected is not None else None
    actual_intent = selected.candidate.intent_name if selected is not None else None
    canonical_ok = actual_text == case["expected_canonical"]
    intent_ok = actual_intent == case["expected_intent"]
    slots_ok = _slots_match(actual_slots, expected_slots)
    return {
        "language": lang,
        "mode": mode_name,
        "category": case["category"],
        "query": case["query"],
        "expected_canonical": case["expected_canonical"],
        "actual_canonical": actual_text,
        "expected_intent": case["expected_intent"],
        "actual_intent": actual_intent,
        "expected_slots": expected_slots,
        "actual_slots": dict(actual_slots),
        "canonical_ok": canonical_ok,
        "intent_ok": intent_ok,
        "slots_ok": slots_ok,
        "intent_slots_ok": intent_ok and slots_ok,
        "fallback": selected is None,
        "reason": reason,
        "final_score": _score_value(selected, "final_score"),
        "top_score": gate.get("top_score"),
        "competing_score": gate.get("competing_score"),
        "margin": gate.get("margin"),
        "acceptance_reason": gate.get("reason"),
        "latency_ms": latency_ms,
    }


def _failure_detail(
    mode_name: str,
    case: Mapping[str, Any],
    selected: RankedCandidate | None,
    reason: str,
    actual_slots: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a compact failure record for manual algorithm tuning."""
    if selected is None:
        return {
            "mode": mode_name,
            "category": case["category"],
            "query": case["query"],
            "reason": reason,
            "expected": case["expected_canonical"],
            "actual": None,
            "expected_intent": case["expected_intent"],
            "actual_intent": None,
            "expected_slots": case.get("expected_slots", {}),
            "actual_slots": actual_slots,
            "final_score": None,
            "acceptance_reason": gate.get("reason"),
            "top_score": gate.get("top_score"),
            "competing_score": gate.get("competing_score"),
            "margin": gate.get("margin"),
        }
    return {
        "mode": mode_name,
        "category": case["category"],
        "query": case["query"],
        "reason": reason,
        "expected": case["expected_canonical"],
        "actual": selected.candidate.text,
        "expected_intent": case["expected_intent"],
        "actual_intent": selected.candidate.intent_name,
        "expected_slots": case.get("expected_slots", {}),
        "actual_slots": actual_slots,
        "final_score": selected.scores.final_score,
    }


def _coverage_payload(
    test_cases: list[dict[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    candidate_intents: set[str],
    candidate_count: int,
    build_latency_ms: float,
) -> dict[str, Any]:
    """Return language-level dataset and candidate coverage details."""
    source_intents = _intent_names_from_sources(sources)
    dataset_intents = {case["expected_intent"] for case in test_cases}
    return {
        "builtin_intents": sorted(source_intents),
        "candidate_intents": sorted(candidate_intents),
        "dataset_intents": sorted(dataset_intents),
        "candidate_count": candidate_count,
        "build_latency_ms": build_latency_ms,
        "missing_candidate_intents": sorted(source_intents - candidate_intents),
        "untested_candidate_intents": sorted(candidate_intents - dataset_intents),
        "dataset_intents_without_candidates": sorted(dataset_intents - candidate_intents),
    }


def _print_coverage(
    lang: str, test_cases: list[dict[str, Any]], coverage: Mapping[str, Any]
) -> None:
    """Print language-level dataset and candidate coverage details."""
    missing_candidate_intents = coverage["missing_candidate_intents"]
    untested_candidate_intents = coverage["untested_candidate_intents"]
    missing_dataset_candidates = coverage["dataset_intents_without_candidates"]
    print(f"\nLanguage: {lang.upper()} ({len(test_cases)} cases)")
    print(
        "Coverage: "
        f"builtin_intents={len(coverage['builtin_intents'])} | "
        f"candidate_intents={len(coverage['candidate_intents'])} | "
        f"dataset_intents={len(coverage['dataset_intents'])} | "
        f"candidates={coverage['candidate_count']} | "
        f"build_latency={coverage['build_latency_ms']:.1f}ms"
    )
    if missing_candidate_intents:
        print(
            "Missing candidate intents: "
            f"{len(missing_candidate_intents)} ({_short_names(missing_candidate_intents)})"
        )
    if untested_candidate_intents:
        print(
            "Untested candidate intents: "
            f"{len(untested_candidate_intents)} ({_short_names(untested_candidate_intents)})"
        )
    if missing_dataset_candidates:
        print(
            "Dataset intents without candidates: "
            f"{len(missing_dataset_candidates)} ({_short_names(missing_dataset_candidates)})"
        )


def _aggregate_mode_stats(
    results: dict[str, dict[str, CategoryStats]], mode_name: str
) -> CategoryStats:
    """Return aggregate stats across categories for one mode/component."""
    total = CategoryStats()
    for stats in results[mode_name].values():
        total.merge(stats)
    return total


def _summary_payload(results: dict[str, dict[str, CategoryStats]]) -> dict[str, Any]:
    """Return JSON summary metrics by mode and category."""
    payload: dict[str, Any] = {}
    for mode_name, categories in results.items():
        overall = _aggregate_mode_stats(results, mode_name)
        if overall.total == 0:
            continue
        payload[mode_name] = {
            "categories": {
                category: stats.as_dict() for category, stats in sorted(categories.items())
            },
            "overall": overall.as_dict(),
        }
    return payload


def _print_summary_table(title: str, results: dict[str, dict[str, CategoryStats]]) -> None:
    """Print aggregate metrics for one language or the global run."""
    print(f"\n{title}")
    print("-" * 166)
    headers = (
        f"{'Category':<24} | {'Total':<5} | "
        f"{'Hass Acc':<17} | {'Lex Acc':<17} | "
        f"{'Hass Mis':<17} | {'Lex Mis':<17} | "
        f"{'Hass Fall':<17} | {'Lex Fall':<17} | "
        f"{'Lex ms':<8}"
    )
    print(headers)
    print("-" * 166)
    categories = sorted(results["lexical"].keys())
    for category in categories:
        hass_stats = results["hassil"].get(category, CategoryStats())
        lex_stats = results["lexical"].get(category, CategoryStats())
        print(
            f"{category:<24} | {lex_stats.total:<5} | "
            f"{_metric_str(hass_stats.intent_slots_correct, hass_stats.total):<17} | "
            f"{_metric_str(lex_stats.intent_slots_correct, lex_stats.total):<17} | "
            f"{_metric_str(hass_stats.mismatch, hass_stats.total):<17} | "
            f"{_metric_str(lex_stats.mismatch, lex_stats.total):<17} | "
            f"{_metric_str(hass_stats.fallback, hass_stats.total):<17} | "
            f"{_metric_str(lex_stats.fallback, lex_stats.total):<17} | "
            f"{lex_stats.average_latency_ms:<8.1f}"
        )
    hass_total = _aggregate_mode_stats(results, "hassil")
    lex_total = _aggregate_mode_stats(results, "lexical")
    print("-" * 166)
    print(
        f"{'Overall':<24} | {lex_total.total:<5} | "
        f"{_metric_str(hass_total.intent_slots_correct, hass_total.total):<17} | "
        f"{_metric_str(lex_total.intent_slots_correct, lex_total.total):<17} | "
        f"{_metric_str(hass_total.mismatch, hass_total.total):<17} | "
        f"{_metric_str(lex_total.mismatch, lex_total.total):<17} | "
        f"{_metric_str(hass_total.fallback, hass_total.total):<17} | "
        f"{_metric_str(lex_total.fallback, lex_total.total):<17} | "
        f"{lex_total.average_latency_ms:<8.1f}"
    )
    print("-" * 166)


def _print_ablation_table(title: str, ablations: dict[str, dict[str, CategoryStats]]) -> None:
    """Print top-1 ablation metrics by scoring component."""
    print(f"\n{title}")
    print("-" * 96)
    print(
        f"{'Component':<16} | {'Total':<5} | {'Canonical':<16} | "
        f"{'Intent/Slot':<19} | {'Fallback':<14}"
    )
    print("-" * 96)
    for component in ABLATION_COMPONENTS:
        stats = _aggregate_mode_stats(ablations, component)
        if stats.total == 0:
            continue
        print(
            f"{component:<16} | {stats.total:<5} | "
            f"{_metric_str(stats.correct, stats.total):<16} | "
            f"{_metric_str(stats.intent_slots_correct, stats.total):<19} | "
            f"{_metric_str(stats.fallback, stats.total):<14}"
        )
    print("-" * 96)


def _print_failure_details(failures: list[dict[str, Any]], failure_limit: int) -> None:
    """Print bounded per-case failures for manual inspection."""
    if failure_limit < 1 or not failures:
        return
    print(f"\nFailure details (first {min(len(failures), failure_limit)} of {len(failures)}):")
    for item in failures[:failure_limit]:
        final_score = item["final_score"]
        final_score_str = "none" if final_score is None else f"{final_score:.3f}"
        print(
            f"- [{item['mode']}][{item['category']}] {item['query']!r} "
            f"reason={item['reason']} final={final_score_str}"
        )
        print(
            f"  expected={item['expected']!r} ({item['expected_intent']}, "
            f"slots={item['expected_slots']})"
        )
        print(
            f"  actual={item['actual']!r} ({item['actual_intent']}, slots={item['actual_slots']})"
        )


def _record_ablations(
    ablations: dict[str, dict[str, CategoryStats]],
    ranked: tuple[RankedCandidate, ...],
    case: Mapping[str, Any],
) -> None:
    """Record component-only top-1 metrics for one ranked candidate set."""
    for component in ABLATION_COMPONENTS:
        selected = _select_ablation_candidate(ranked, component)
        _record_case_result(
            _stats_for(ablations, component, case["category"]),
            selected,
            case["expected_canonical"],
            case["expected_intent"],
            case.get("expected_slots", {}),
            0.0,
        )


def _markdown_metric(stats: Mapping[str, Any]) -> str:
    """Return a compact Markdown metric string."""
    return (
        f"{stats['intent_slot_accuracy']:.1f}% intent/slot, "
        f"{stats['canonical_accuracy']:.1f}% canonical, "
        f"{stats['mismatch_rate']:.1f}% mismatch, "
        f"{stats['fallback_rate']:.1f}% fallback"
    )


def _markdown_report(report: Mapping[str, Any]) -> str:
    """Return a human-readable Markdown report."""
    lines = [
        "# Assist Canonicalizer Evaluation",
        "",
    ]
    overall_summary = report["overall"]["summary"]
    lines.extend(["## Overall", ""])
    for mode_name, payload in overall_summary.items():
        total = payload["overall"]["total"]
        if total:
            lines.append(f"- `{mode_name}`: {_markdown_metric(payload['overall'])}")
    for lang, payload in sorted(report["languages"].items()):
        coverage = payload["coverage"]
        lines.extend(
            [
                "",
                f"## {lang.upper()}",
                "",
                f"- Builtin intents: {len(coverage['builtin_intents'])}",
                f"- Candidate intents: {len(coverage['candidate_intents'])}",
                f"- Dataset intents: {len(coverage['dataset_intents'])}",
                f"- Candidates: {coverage['candidate_count']} "
                f"(build latency: {coverage['build_latency_ms']:.1f}ms)",
                f"- Missing candidate intents: {len(coverage['missing_candidate_intents'])}",
                f"- Untested candidate intents: {len(coverage['untested_candidate_intents'])}",
                "",
                "| Mode | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for mode_name, summary in payload["summary"].items():
            stats = summary["overall"]
            if not stats["total"]:
                continue
            lines.append(
                f"| `{mode_name}` | {stats['total']} | "
                f"{stats['intent_slot_accuracy']:.1f}% | "
                f"{stats['canonical_accuracy']:.1f}% | "
                f"{stats['mismatch_rate']:.1f}% | "
                f"{stats['fallback_rate']:.1f}% | "
                f"{stats['average_latency_ms']:.1f} |"
            )
    threshold_failures = report["overall"].get("threshold_failures", [])
    if threshold_failures:
        lines.extend(["", "## Threshold Failures", ""])
        lines.extend(f"- {failure}" for failure in threshold_failures)
    return "\n".join(lines) + "\n"


def _write_json_report(path: str, report: Mapping[str, Any]) -> None:
    """Write the evaluation report as JSON using an atomic replacement."""
    safe_report = _stringify_keys(report)
    output_path = Path(path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(
        orjson.dumps(safe_report, option=orjson.OPT_INDENT_2).decode("utf-8") + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(output_path)


def _write_markdown_report(path: str, report: Mapping[str, Any]) -> None:
    """Write the evaluation report as Markdown using an atomic replacement."""
    output_path = Path(path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(_markdown_report(report), encoding="utf-8")
    tmp_path.replace(output_path)


def _threshold_failures(
    stats: CategoryStats,
    min_intent_slot_accuracy: float | None,
    max_fallback_rate: float | None,
) -> list[str]:
    """Return threshold failure messages for the selected aggregate stats."""
    failures = []
    if (
        min_intent_slot_accuracy is not None
        and stats.intent_slot_accuracy < min_intent_slot_accuracy
    ):
        failures.append(
            f"intent/slot accuracy {stats.intent_slot_accuracy:.1f}% "
            f"is below {min_intent_slot_accuracy:.1f}%"
        )
    if max_fallback_rate is not None and stats.fallback_rate > max_fallback_rate:
        failures.append(
            f"fallback rate {stats.fallback_rate:.1f}% is above {max_fallback_rate:.1f}%"
        )
    return failures


async def run_evaluation(
    datasets: dict[str, str],
    failure_limit: int,
    output_json: str | None,
    output_md: str | None,
    min_intent_slot_accuracy: float | None,
    max_fallback_rate: float | None,
) -> bool:
    """Run evaluation on the datasets and print the summary report."""
    _bootstrap_project_imports()
    if not datasets:
        print("Error: No datasets found in tests/real_world/")
        return False
    if failure_limit < 0:
        print("Error: --failure-limit must be zero or positive")
        return False

    print(
        "======================================================================================================================"
    )
    print("ASSIST CANONICALIZER PERFORMANCE EVALUATION REPORT")
    print(
        "======================================================================================================================"
    )
    print("Dataset Directory: tests/real_world/")
    print(f"Total Languages: {len(datasets)}")
    print(f"Failure Detail Limit: {failure_limit}")
    if output_json:
        print(f"JSON Output: {output_json}")
    if output_md:
        print(f"Markdown Output: {output_md}")
    print(
        "======================================================================================================================"
    )

    overall_success = True
    global_results = _new_results()
    global_ablations = _new_ablation_results()
    all_case_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "languages": {},
    }

    for lang, path in sorted(datasets.items()):
        with open(path, encoding="utf-8") as f:
            data = orjson.loads(f.read())
        if not isinstance(data, dict):
            print(f"Error: Dataset root must be an object: {path}")
            return False
        raw_cases = data.get("test_cases", [])
        if not isinstance(raw_cases, list):
            print(f"Error: test_cases must be a list: {path}")
            return False
        try:
            test_cases = _validate_test_cases(raw_cases, lang, path)
            slots = _dataset_registry_slots(data, lang)
        except ValueError as err:
            print(f"Error: {err}")
            return False
        if not test_cases:
            continue

        sources = load_language_intent_sources(lang)
        start_build = time.perf_counter()
        lang_candidates = build_candidates_from_intent_sources(lang, sources, slots)
        build_latency_ms = (time.perf_counter() - start_build) * 1000
        index = build_index(lang, lang_candidates)
        candidate_intents = {candidate.intent_name for candidate in lang_candidates}
        coverage = _coverage_payload(
            test_cases,
            sources,
            candidate_intents,
            len(lang_candidates),
            build_latency_ms,
        )
        _print_coverage(lang, test_cases, coverage)

        merged_intents = {}
        for s in sources.values():
            hassil.merge_dict(merged_intents, s)
        hassil_intents = hassil.intents.Intents.from_dict(merged_intents)
        hassil_slot_lists: dict[str, hassil.intents.SlotList] = {}
        for slot_name, slot_values in slots.items():
            text_values = []
            for val in slot_values:
                text_values.append(
                    hassil.TextSlotValue(
                        text_in=hassil.parse_sentence(val).expression,
                        value_out=val,
                    )
                )
            hassil_slot_lists[slot_name] = hassil.TextSlotList(name=slot_name, values=text_values)

        results = _new_results()
        ablations = _new_ablation_results()
        failures: list[dict[str, Any]] = []
        language_rows: list[dict[str, Any]] = []

        for mode_name in ("hassil", "lexical"):
            runtime = CanonicalizerRuntime()
            runtime.set_index(index)
            runtime.update_registry_slot_values(slots)

            for case in test_cases:
                category = case["category"]
                query = case["query"]
                expected_canonical = case["expected_canonical"]
                expected_intent = case["expected_intent"]
                expected_slots = case.get("expected_slots", {})
                stats = _stats_for(results, mode_name, category)

                start_time = time.perf_counter()
                if mode_name == "hassil":
                    res_list = []
                    while True:
                        try:
                            res_list = list(
                                hassil.recognize_all(
                                    query,
                                    hassil_intents,
                                    slot_lists=hassil_slot_lists,
                                )
                            )
                            break
                        except hassil.errors.MissingListError as err:
                            match = re.search(r"\{([^}]+)\}", str(err))
                            if match:
                                list_name = match.group(1)
                                hassil_slot_lists[list_name] = hassil.TextSlotList(list_name, [])
                            else:
                                raise
                    res = None
                    for r in res_list:
                        actual_slots = {name: entity.value for name, entity in r.entities.items()}
                        if r.intent.name == expected_intent and _slots_match(
                            actual_slots, expected_slots
                        ):
                            res = r
                            break
                    if res is None and res_list:
                        res = res_list[0]
                    if res is not None:
                        actual_slots = {name: entity.value for name, entity in res.entities.items()}
                        is_intent_ok = res.intent.name == expected_intent
                        is_slots_ok = _slots_match(actual_slots, expected_slots)
                        canonical_text = (
                            expected_canonical
                            if (is_intent_ok and is_slots_ok)
                            else f"mismatch: {res.intent.name}"
                        )
                        candidate = Candidate(
                            text=canonical_text,
                            intent_name=res.intent.name,
                            metadata={"slots": orjson.dumps(actual_slots).decode("utf-8")},
                        )
                        scores = _ScoreBreakdown(
                            rapidfuzz_score=1.0,
                            char_ngram_score=1.0,
                            bm25_score=1.0,
                            intent_score=1.0,
                            final_score=1.0,
                        )
                        ranked = (_RankedCandidate(candidate=candidate, scores=scores),)
                    else:
                        ranked = ()
                else:
                    ranked = runtime.rank_with_dynamic_candidates(lang, index, query)

                selected, gate = _select_accepted_with_gate(ranked)
                latency_ms = (time.perf_counter() - start_time) * 1000
                is_ok, reason, actual_slots = _record_case_result(
                    stats,
                    selected,
                    expected_canonical,
                    expected_intent,
                    expected_slots,
                    latency_ms,
                )
                row = _case_row(
                    lang,
                    mode_name,
                    case,
                    selected,
                    reason,
                    actual_slots,
                    latency_ms,
                    gate,
                )
                language_rows.append(row)
                all_case_rows.append(row)
                if mode_name == "lexical":
                    _record_ablations(ablations, ranked, case)
                if not is_ok:
                    failures.append(
                        _failure_detail(
                            mode_name,
                            case,
                            selected,
                            reason,
                            actual_slots,
                            gate,
                        )
                    )

        _print_summary_table(f"Summary: {lang.upper()}", results)
        _print_ablation_table(f"Ablation Top-1: {lang.upper()}", ablations)
        _print_failure_details(failures, failure_limit)
        _merge_results(global_results, results)
        _merge_results(global_ablations, ablations)
        report["languages"][lang] = {
            "coverage": coverage,
            "summary": _summary_payload(results),
            "ablations": _summary_payload(ablations),
            "failures": failures,
            "cases": language_rows,
        }

    _print_summary_table("Summary: ALL LANGUAGES", global_results)
    _print_ablation_table("Ablation Top-1: ALL LANGUAGES", global_ablations)
    lex_stats = _aggregate_mode_stats(global_results, "lexical")
    threshold_failures = _threshold_failures(
        lex_stats,
        min_intent_slot_accuracy,
        max_fallback_rate,
    )
    if threshold_failures:
        overall_success = False
        print("\nThreshold failures:")
        for failure in threshold_failures:
            print(f"- {failure}")

    report["overall"] = {
        "summary": _summary_payload(global_results),
        "ablations": _summary_payload(global_ablations),
        "threshold_failures": threshold_failures,
    }
    if output_json:
        _write_json_report(output_json, report)
    if output_md:
        _write_markdown_report(output_md, report)
    print("\nEvaluation Complete.")
    return overall_success


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Assist Canonicalizer matching metrics")
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="tests/real_world",
        help="Directory containing the JSON dataset files",
    )
    parser.add_argument(
        "--failure-limit",
        type=int,
        default=0,
        help="Maximum number of detailed failures to print per language",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional JSON report output path",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional Markdown report output path",
    )
    parser.add_argument(
        "--min-intent-slot-accuracy",
        type=float,
        default=None,
        help="Fail when overall lexical intent/slot accuracy is below this percentage",
    )
    parser.add_argument(
        "--max-fallback-rate",
        type=float,
        default=None,
        help="Fail when overall lexical fallback rate is above this percentage",
    )
    args = parser.parse_args()

    datasets = {}
    if os.path.exists(args.datasets_dir):
        for filename in os.listdir(args.datasets_dir):
            if filename.endswith(".json"):
                lang = filename[:-5]
                datasets[lang] = os.path.join(args.datasets_dir, filename)

    success = asyncio.run(
        run_evaluation(
            datasets=datasets,
            failure_limit=args.failure_limit,
            output_json=args.output_json,
            output_md=args.output_md,
            min_intent_slot_accuracy=args.min_intent_slot_accuracy,
            max_fallback_rate=args.max_fallback_rate,
        )
    )
    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
