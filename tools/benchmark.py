"""Unified benchmark tool for Assist Canonicalizer.

Supports accuracy matching metrics evaluation (--mode accuracy) and algorithmic
performance profiling (--mode performance) across throughput, latency, memory,
and scoring components.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import io
import json
import math
import os
import pstats
import re
import statistics
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from string import ascii_letters, ascii_lowercase, digits
from typing import TYPE_CHECKING, Any

import hassil
import hassil.errors
import orjson

if TYPE_CHECKING:
    from custom_components.assist_canonicalizer.ranking import RankedCandidate

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_PATH_ALLOWED_CHARS = ascii_letters + digits + "/._-"
_TARGET_ALLOWED_CHARS = ascii_lowercase + "_"
_MISSING_LIST_RE = re.compile(r"\{([^}]+)\}")

_BOOTSTRAPPED = False

# Global bindings for custom component modules loaded via bootstrap
DEFAULT_MIN_CONFIDENCE: float = 0.0
CanonicalizerRuntime: Any = None
build_candidates_from_intent_sources: Any = None
build_index: Any = None
load_language_intent_sources: Any = None
normalize_text: Any = None
normalize_text_no_diacritics: Any = None
char_ngrams_normalized: Any = None
_RankedCandidate: Any = None
_ScoreBreakdown: Any = None
_accepted_candidate: Any = None
Candidate: Any = None
FallbackReason: Any = None
BM25Index: Any = None
CharNGramIndex: Any = None
rapidfuzz_similarity_normalized: Any = None
lexical_score: Any = None


def _bootstrap_project_imports() -> None:
    """Import custom component modules after verifying sys.path."""
    global _BOOTSTRAPPED
    global DEFAULT_MIN_CONFIDENCE
    global CanonicalizerRuntime
    global _accepted_candidate
    global build_candidates_from_intent_sources, build_index, load_language_intent_sources
    global normalize_text, normalize_text_no_diacritics, char_ngrams_normalized
    global _RankedCandidate, _ScoreBreakdown, Candidate, FallbackReason
    global BM25Index, CharNGramIndex, rapidfuzz_similarity_normalized, lexical_score

    if _BOOTSTRAPPED:
        return

    repo_root = Path(__file__).resolve().parent.parent
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)

    from custom_components.assist_canonicalizer import const as const_module
    from custom_components.assist_canonicalizer.bm25 import BM25Index as ImportedBM25Index
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
        char_ngrams_normalized as imported_char_ngrams,
    )
    from custom_components.assist_canonicalizer.normalization import (
        normalize_text as imported_normalize_text,
    )
    from custom_components.assist_canonicalizer.normalization import (
        normalize_text_no_diacritics as imported_normalize_no_diac,
    )
    from custom_components.assist_canonicalizer.ranking import (
        CharNGramIndex as ImportedCharNGramIndex,
    )
    from custom_components.assist_canonicalizer.ranking import (
        RankedCandidate as ImportedRankedCandidate,
    )
    from custom_components.assist_canonicalizer.ranking import (
        ScoreBreakdown as ImportedScoreBreakdown,
    )
    from custom_components.assist_canonicalizer.ranking import (
        accepted_candidate as imported_accepted_candidate,
    )
    from custom_components.assist_canonicalizer.ranking import (
        lexical_score as imported_lexical_score,
    )
    from custom_components.assist_canonicalizer.ranking import (
        rapidfuzz_similarity_normalized as imported_rf_sim,
    )
    from custom_components.assist_canonicalizer.runtime import (
        CanonicalizerRuntime as ImportedCanonicalizerRuntime,
    )

    DEFAULT_MIN_CONFIDENCE = const_module.DEFAULT_MIN_CONFIDENCE
    CanonicalizerRuntime = ImportedCanonicalizerRuntime
    build_candidates_from_intent_sources = imported_build_candidates_from_intent_sources
    build_index = imported_build_index
    load_language_intent_sources = imported_load_language_intent_sources
    normalize_text = imported_normalize_text
    normalize_text_no_diacritics = imported_normalize_no_diac
    char_ngrams_normalized = imported_char_ngrams
    _RankedCandidate = ImportedRankedCandidate
    _ScoreBreakdown = ImportedScoreBreakdown
    _accepted_candidate = imported_accepted_candidate
    Candidate = ImportedCandidate
    FallbackReason = const_module.FallbackReason
    BM25Index = ImportedBM25Index
    CharNGramIndex = ImportedCharNGramIndex
    rapidfuzz_similarity_normalized = imported_rf_sim
    lexical_score = imported_lexical_score

    _BOOTSTRAPPED = True


# ---------------------------------------------------------------------------
# Shared Core Helpers
# ---------------------------------------------------------------------------

REGISTRY_SLOTS: dict[str, dict[str, tuple[str, ...]]] = {
    "en": {
        "name": (
            "Living Room Light",
            "Living Room Fan",
            "Bedroom Light",
            "Kitchen Light",
        ),
        "name:vacuum": ("vacuum",),
        "name:todo": ("shopping list",),
    },
    "vi": {
        "name": (
            "đèn phòng khách",
            "quạt phòng khách",
            "đèn phòng ngủ",
        ),
        "name:vacuum": ("máy hút bụi", "robot hút bụi"),
        "name:todo": ("danh sách việc cần làm",),
    },
    "de": {
        "name": (
            "Wohnzimmerlampe",
            "Badezimmerlüfter",
            "Schlafzimmerlampe",
        ),
        "name:vacuum": ("staubsauger",),
        "name:todo": ("einkaufsliste",),
    },
    "fr": {
        "name": (
            "lumière du salon",
            "ventilateur du salon",
            "lumière de la chambre",
        ),
        "name:vacuum": ("aspirateur",),
        "name:todo": ("liste de courses",),
    },
    "nl": {
        "name": (
            "woonkamerlamp",
            "keukenlamp",
            "badkamerventilator",
            "stofzuiger",
        ),
        "name:vacuum": ("stofzuiger",),
        "name:todo": ("boodschappenlijst", "todolijst"),
    },
}


def sanitize_chars(value: str, allowed: str) -> str:
    """Validate and sanitize a string using allowed characters to break taint."""
    if not isinstance(value, str):
        raise ValueError("Expected a string.")
    safe_chars: list[str] = []
    for char in value:
        idx = allowed.find(char)
        if idx == -1:
            raise ValueError(f"character {char!r} is not allowed.")
        safe_chars.append(allowed[idx])
    return "".join(safe_chars)


def sanitize_path(root_path: str, user_path: str) -> str:
    """Validate, sanitize and contain a user-supplied file path."""
    try:
        clean = sanitize_chars(user_path, _PATH_ALLOWED_CHARS)
    except ValueError as err:
        raise ValueError(f"Invalid path {user_path!r}; {err}") from err

    root = os.path.realpath(root_path)
    fullpath = os.path.realpath(os.path.normpath(os.path.join(root, clean)))

    if fullpath != root and not fullpath.startswith(root + os.sep):
        raise ValueError(f"Resolved path {fullpath!r} escapes allowed root {root!r}.")
    return fullpath


def sanitize_path_required(root: str, label: str, path: str) -> str:
    """Sanitize *path* under *root* or print error to stderr and exit(1)."""
    try:
        return sanitize_path(root, path)
    except ValueError as err:
        print(f"Error: {label} must be inside {root}: {path} — {err}", file=sys.stderr)
        sys.exit(1)


def discover_datasets(datasets_dir: str, languages: list[str] | None = None) -> dict[str, str]:
    """Discover real-world JSON datasets ordered by language code."""
    results: dict[str, str] = {}
    safe_dir_real = os.path.realpath(datasets_dir)
    if not os.path.isdir(safe_dir_real):
        return results
    for entry in sorted(os.listdir(safe_dir_real)):
        if not entry.endswith(".json"):
            continue
        full_path = os.path.join(safe_dir_real, entry)
        real_path = os.path.realpath(full_path)
        if not os.path.isfile(real_path):
            continue
        if not real_path.startswith(safe_dir_real + os.sep):
            continue
        lang_code = entry[:-5]
        if languages is not None and lang_code not in languages:
            continue
        results[lang_code] = real_path
    return results


def align_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    *,
    alignments: str = "<",
    padding: int = 1,
    sep: str = " | ",
) -> tuple[str, str, list[str]]:
    """Return dynamically-aligned table lines: ``(header, separator, data_lines)``."""
    if not headers:
        return "", "", []
    ncols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    widths = [w + padding for w in widths]

    if len(alignments) == 1:
        aligns = list(alignments * ncols)
    else:
        aligns = list(alignments)
        if len(aligns) < ncols:
            # Pad with last alignment character to match column count
            aligns.extend([aligns[-1]] * (ncols - len(aligns)))
        aligns = aligns[:ncols]

    hdr_parts = [f"{h:{a}{w}}" for h, a, w in zip(headers, aligns, widths, strict=True)]
    header_line = sep.join(hdr_parts)
    sep_line = sep.join("-" * w for w in widths)

    data_lines: list[str] = []
    for row in rows:
        parts = [f"{c:{a}{w}}" for c, a, w in zip(row, aligns, widths, strict=True)]
        data_lines.append(sep.join(parts))
    return header_line, sep_line, data_lines


def atomic_write(path: str, content: str) -> None:
    """Atomically write *content* to *path* via a temporary file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(output_path)


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


def make_hassil_slot_lists(
    slots: Mapping[str, tuple[str, ...]],
) -> dict[str, hassil.intents.SlotList]:
    """Build HassIL slot lists from registry slot fixtures."""
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


def run_hassil_recognize_all(
    query: str,
    intents: hassil.intents.Intents,
    slot_lists: dict[str, hassil.intents.SlotList],
) -> list[Any]:
    """Run HassIL recognize_all with lazy slot-list injection on MissingListError."""
    working_lists = dict(slot_lists)
    while True:
        try:
            return list(hassil.recognize_all(query, intents, slot_lists=working_lists))
        except hassil.errors.MissingListError as err:
            match = _MISSING_LIST_RE.search(str(err))
            if match is None:
                raise
            list_name = match.group(1)
            working_lists[list_name] = hassil.TextSlotList(list_name, [])


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


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two slot values with numeric type coercion."""
    if isinstance(a, str) and isinstance(b, str):
        return normalize_text(a) == normalize_text(b)
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        try:
            return float(a) == float(b)
        except (ValueError, TypeError):
            return False
    return a == b


def _slots_match(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Check if actual candidate slots contain all expected slot values."""
    if not expected:
        return True
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value is None:
            return False
        if not _values_equal(actual_value, expected_value):
            return False
    return True


# ---------------------------------------------------------------------------
# Mode: Accuracy metrics evaluation
# ---------------------------------------------------------------------------


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
        """Return serializable stats values including raw counts for text rendering."""
        return {
            "total": self.total,
            "correct": self.correct,
            "intent_correct": self.intent_correct,
            "slots_correct": self.slots_correct,
            "intent_slots_correct": self.intent_slots_correct,
            "fallback": self.fallback,
            "mismatch": self.mismatch,
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


def _deduce_rejection_reason(
    ranked: tuple[RankedCandidate, ...],
) -> str:
    """Return the rejection reason for a top candidate rejected by gates."""
    if not ranked:
        return FallbackReason.EMPTY_INDEX.value
    if ranked[0].scores.final_score < DEFAULT_MIN_CONFIDENCE:
        return FallbackReason.LOW_CONFIDENCE.value
    return FallbackReason.LOW_MARGIN.value


def _select_accepted_with_gate(
    ranked: tuple[RankedCandidate, ...],
) -> tuple[RankedCandidate | None, dict[str, Any]]:
    """Return the accepted candidate with acceptance gate diagnostics."""
    result = _accepted_candidate(ranked)
    reason = "accepted" if result is not None else _deduce_rejection_reason(ranked)

    top_score = ranked[0].scores.final_score if ranked else None
    competing_candidate = (
        next(
            (
                item
                for item in ranked[1:]
                if item.candidate.intent_name != ranked[0].candidate.intent_name
            ),
            None,
        )
        if ranked
        else None
    )
    competing_score = (
        competing_candidate.scores.final_score if competing_candidate is not None else None
    )
    margin = (
        top_score - competing_score
        if (top_score is not None and competing_score is not None)
        else None
    )
    return result, {
        "accepted": result is not None,
        "reason": reason,
        "top_score": top_score,
        "competing_score": competing_score,
        "margin": margin,
    }


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
    latency_ms: float | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Record one evaluated case and return whether it matched completely."""
    stats.total += 1
    if latency_ms is not None:
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
    if mode_name == "hassil":
        final_score = None
        top_score = None
        competing_score = None
        margin = None
        acceptance_reason = None
    else:
        final_score = _score_value(selected, "final_score") if selected else None
        top_score = gate.get("top_score")
        competing_score = gate.get("competing_score")
        margin = gate.get("margin")
        acceptance_reason = gate.get("reason")
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
        "final_score": final_score,
        "top_score": top_score,
        "competing_score": competing_score,
        "margin": margin,
        "acceptance_reason": acceptance_reason,
        "latency_ms": latency_ms,
    }


def _score_value(selected: RankedCandidate | None, field: str) -> float | None:
    """Extract one score field from a ranked candidate."""
    if selected is None:
        return None
    return getattr(selected.scores, field, None)


def _metric_str(count: int, total: int) -> str:
    """Return a formatted metric cell with count and percentage."""
    if total == 0:
        return "0/0 (0.0%)"
    pct = (count / total) * 100
    return f"{count}/{total} ({pct:.1f}%)"


def _short_names(names: list[str]) -> str:
    """Return a shortened comma-separated intent name list."""
    short = ", ".join(list(names)[:5])
    if len(names) > 5:
        short += f" ... ({len(names)} total)"
    return short


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
            latency_ms=None,
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
    """Return a human-readable Markdown report with dynamically aligned columns."""
    _headers = ("Mode", "Total", "Intent/Slot", "Canonical", "Mismatch", "Fallback", "Avg ms")
    _col_aligns = ("<", ">", ">", ">", ">", ">", ">")
    _col_widths: list[int] = [len(h) for h in _headers]

    class _Row:
        """Helper row representation for markdown report generation."""

        __slots__ = (
            "avg_ms_s",
            "backticked_mode",
            "canonical_s",
            "fallback_s",
            "intent_slot_s",
            "lang",
            "mismatch_s",
            "total_s",
        )

        def __init__(self, lang: str, mode_name: str, stats: Mapping[str, Any]) -> None:
            """Initialize the markdown helper row with stats."""
            self.lang = lang
            self.backticked_mode = f"`{mode_name}`"
            self.total_s = str(int(stats.get("total", 0)))
            self.intent_slot_s = f"{stats.get('intent_slot_accuracy', 0):.1f}%"
            self.canonical_s = f"{stats.get('canonical_accuracy', 0):.1f}%"
            self.mismatch_s = f"{stats.get('mismatch_rate', 0):.1f}%"
            self.fallback_s = f"{stats.get('fallback_rate', 0):.1f}%"
            self.avg_ms_s = f"{stats.get('average_latency_ms', 0):.1f}"

    _all_rows: list[_Row] = []
    for lang, payload in sorted(report.get("languages", {}).items()):
        for mode_name, summary in payload.get("summary", {}).items():
            stats = summary.get("overall", {})
            if not stats.get("total"):
                continue
            row = _Row(lang, mode_name, stats)
            _all_rows.append(row)
            cols = (
                row.backticked_mode,
                row.total_s,
                row.intent_slot_s,
                row.canonical_s,
                row.mismatch_s,
                row.fallback_s,
                row.avg_ms_s,
            )
            for i, s in enumerate(cols):
                _col_widths[i] = max(_col_widths[i], len(s))

    _col_widths = [max(w, 3) for w in _col_widths]

    def _md_sep_line() -> str:
        parts: list[str] = []
        for i, w in enumerate(_col_widths):
            dashes = w - 1
            if _col_aligns[i] == ">":
                parts.append(" " + "-" * dashes + ": ")
            else:
                parts.append(" :" + "-" * dashes + " ")
        return "|" + "|".join(parts) + "|"

    def _md_header_line() -> str:
        parts: list[str] = []
        for i, h in enumerate(_headers):
            w = _col_widths[i]
            parts.append(f" {h:<{w}} ")
        return "|" + "|".join(parts) + "|"

    def _md_data_row(row: _Row) -> str:
        cols = (
            f" {row.backticked_mode:<{_col_widths[0]}} ",
            f" {row.total_s:>{_col_widths[1]}} ",
            f" {row.intent_slot_s:>{_col_widths[2]}} ",
            f" {row.canonical_s:>{_col_widths[3]}} ",
            f" {row.mismatch_s:>{_col_widths[4]}} ",
            f" {row.fallback_s:>{_col_widths[5]}} ",
            f" {row.avg_ms_s:>{_col_widths[6]}} ",
        )
        return "|" + "|".join(cols) + "|"

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

    last_lang: str | None = None
    for row in _all_rows:
        if row.lang != last_lang:
            last_lang = row.lang
            payload = report["languages"][row.lang]
            coverage = payload["coverage"]
            lines.extend(
                [
                    "",
                    f"## {row.lang.upper()}",
                    "",
                    f"- Builtin intents: {len(coverage['builtin_intents'])}",
                    f"- Candidate intents: {len(coverage['candidate_intents'])}",
                    f"- Dataset intents: {len(coverage['dataset_intents'])}",
                    f"- Candidates: {coverage['candidate_count']} "
                    f"(build latency: {coverage['build_latency_ms']:.1f}ms)",
                    f"- Missing candidate intents: {len(coverage['missing_candidate_intents'])}",
                    f"- Untested candidate intents: {len(coverage['untested_candidate_intents'])}",
                    "",
                    _md_header_line(),
                    _md_sep_line(),
                ]
            )
        lines.append(_md_data_row(row))

    threshold_failures = report["overall"].get("threshold_failures", [])
    if threshold_failures:
        lines.extend(["", "## Threshold Failures", ""])
        lines.extend(f"- {failure}" for failure in threshold_failures)
    return "\n".join(lines) + "\n"


def _text_report(report: Mapping[str, Any]) -> str:
    """Return a plain-text report matching the full console output."""
    lines: list[str] = []
    lines.append("=" * 120)
    lines.append("ASSIST CANONICALIZER PERFORMANCE EVALUATION REPORT")
    lines.append("=" * 120)
    lines.append(f"Dataset Directory: {report.get('datasets_dir', 'tests/real_world')}/")
    lines.append(f"Total Languages: {report.get('total_languages', 0)}")
    lines.append(f"Failure Detail Limit: {report.get('failure_limit', 0)}")
    lines.append("=" * 120)

    for lang, payload in sorted(report.get("languages", {}).items()):
        coverage = payload.get("coverage", {})
        lines.append(f"\nLanguage: {lang.upper()} ({coverage.get('case_count', 0)} cases)")
        lines.append(
            f"Coverage: "
            f"builtin_intents={len(coverage.get('builtin_intents', []))} | "
            f"candidate_intents={len(coverage.get('candidate_intents', []))} | "
            f"dataset_intents={len(coverage.get('dataset_intents', []))} | "
            f"candidates={coverage.get('candidate_count', 0)} | "
            f"build_latency={coverage.get('build_latency_ms', 0):.1f}ms"
        )
        _miss = coverage.get("missing_candidate_intents", [])
        if _miss:
            lines.append(f"Missing candidate intents: {len(_miss)} ({_short_names(list(_miss))})")
        _untested = coverage.get("untested_candidate_intents", [])
        if _untested:
            lines.append(
                f"Untested candidate intents: {len(_untested)} ({_short_names(list(_untested))})"
            )
        _missing_ds = coverage.get("dataset_intents_without_candidates", [])
        if _missing_ds:
            lines.append(
                f"Dataset intents without candidates: "
                f"{len(_missing_ds)} ({_short_names(list(_missing_ds))})"
            )
        lines.append(f"\nProcessing {lang.upper()} ...")

        _summary_lines = _text_summary_table(f"Summary: {lang.upper()}", payload.get("summary", {}))
        lines.extend(_summary_lines)

        if payload.get("ablations"):
            _lbl_lines = _text_ablation_table(
                f"Ablation Top-1: {lang.upper()}", payload.get("ablations", {})
            )
            lines.extend(_lbl_lines)

        _failure_limit = report.get("failure_limit", 0)
        _failures = payload.get("failures", [])
        if _failure_limit > 0 and _failures:
            _cnt = min(len(_failures), _failure_limit)
            lines.append(f"\nFailure details (first {_cnt} of {len(_failures)}):")
            for item in _failures[:_failure_limit]:
                _fs = item.get("final_score")
                _fs_str = "none" if _fs is None else f"{_fs:.3f}"
                lines.append(
                    f"- [{item.get('mode', '')}][{item.get('category', '')}] "
                    f"{item.get('query', '')!r} "
                    f"reason={item.get('reason', '')} final={_fs_str}"
                )
                lines.append(
                    f"  expected={item.get('expected', '')!r} "
                    f"({item.get('expected_intent', '')}, slots={item.get('expected_slots', {})})"
                )
                lines.append(
                    f"  actual={item.get('actual', '')!r} "
                    f"({item.get('actual_intent', '')}, slots={item.get('actual_slots', {})})"
                )

    overall = report.get("overall", {})
    _global_summary = _text_summary_table("Summary: ALL LANGUAGES", overall.get("summary", {}))
    lines.extend(_global_summary)
    _global_ablation = _text_ablation_table(
        "Ablation Top-1: ALL LANGUAGES", overall.get("ablations", {})
    )
    lines.extend(_global_ablation)

    threshold_failures = overall.get("threshold_failures", [])
    if threshold_failures:
        lines.append("\nThreshold failures:")
        for failure in threshold_failures:
            lines.append(f"- {failure}")

    lines.append("\nEvaluation Complete.")
    return "\n".join(lines) + "\n"


def _text_summary_table(title: str, payload: Mapping[str, Any]) -> list[str]:
    """Return text lines for a summary table with dynamically aligned columns."""
    lines: list[str] = [f"\n{title}"]
    _headers = (
        "Category",
        "Total",
        "Hass Acc",
        "Lex Acc",
        "Hass Mis",
        "Lex Mis",
        "Hass Fall",
        "Lex Fall",
        "Lex ms",
    )
    hassil_data = payload.get("hassil", {})
    lex_data = payload.get("lexical", {})
    cat_keys = hassil_data.get("categories", {}).keys() | lex_data.get("categories", {}).keys()

    cat_rows: list[tuple[str, ...]] = []
    for cat in sorted(cat_keys):
        hass_cat = hassil_data.get("categories", {}).get(cat, {})
        lex_cat = lex_data.get("categories", {}).get(cat, {})
        lex_total = lex_cat.get("total", 0)
        cat_rows.append(
            (
                cat,
                str(lex_total),
                _metric_str(hass_cat.get("intent_slots_correct", 0), hass_cat.get("total", 0)),
                _metric_str(lex_cat.get("intent_slots_correct", 0), lex_cat.get("total", 0)),
                _metric_str(hass_cat.get("mismatch", 0), hass_cat.get("total", 0)),
                _metric_str(lex_cat.get("mismatch", 0), lex_cat.get("total", 0)),
                _metric_str(hass_cat.get("fallback", 0), hass_cat.get("total", 0)),
                _metric_str(lex_cat.get("fallback", 0), lex_cat.get("total", 0)),
                f"{lex_cat.get('average_latency_ms', 0):.1f}",
            )
        )

    hass_overall = hassil_data.get("overall", {})
    lex_overall = lex_data.get("overall", {})
    overall_row: tuple[str, ...] = (
        "Overall",
        str(lex_overall.get("total", 0)),
        _metric_str(hass_overall.get("intent_slots_correct", 0), hass_overall.get("total", 0)),
        _metric_str(lex_overall.get("intent_slots_correct", 0), lex_overall.get("total", 0)),
        _metric_str(hass_overall.get("mismatch", 0), hass_overall.get("total", 0)),
        _metric_str(lex_overall.get("mismatch", 0), lex_overall.get("total", 0)),
        _metric_str(hass_overall.get("fallback", 0), hass_overall.get("total", 0)),
        _metric_str(lex_overall.get("fallback", 0), lex_overall.get("total", 0)),
        f"{lex_overall.get('average_latency_ms', 0):.1f}",
    )
    cat_rows.append(overall_row)
    hdr, sep, data = align_table(_headers, cat_rows, alignments="<")
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)
    lines.extend(data[:-1])
    lines.append(sep)
    lines.append(data[-1])
    lines.append(sep)
    return lines


def _text_ablation_table(title: str, payload: Mapping[str, Any]) -> list[str]:
    """Return text lines for an ablation table with dynamically aligned columns."""
    lines: list[str] = [f"\n{title}"]
    _headers = ("Component", "Total", "Canonical", "Intent/Slot", "Fallback")
    ab_rows: list[tuple[str, ...]] = []
    for comp in ABLATION_COMPONENTS:
        comp_data = payload.get(comp, {})
        overall = comp_data.get("overall", {})
        total = overall.get("total", 0)
        if total == 0:
            continue
        ab_rows.append(
            (
                comp,
                str(total),
                _metric_str(overall.get("correct", 0), overall.get("total", 0)),
                _metric_str(overall.get("intent_slots_correct", 0), overall.get("total", 0)),
                _metric_str(overall.get("fallback", 0), overall.get("total", 0)),
            )
        )
    hdr, sep, data = align_table(_headers, ab_rows, alignments="<")
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)
    lines.extend(data)
    lines.append(sep)
    return lines


def _write_json_report(path: str, report: Mapping[str, Any]) -> None:
    """Write the evaluation report as JSON using an atomic replacement."""
    atomic_write(
        path,
        orjson.dumps(_stringify_keys(report), option=orjson.OPT_INDENT_2).decode("utf-8") + "\n",
    )


def _write_markdown_report(path: str, report: Mapping[str, Any]) -> None:
    """Write the evaluation report as Markdown using an atomic replacement."""
    atomic_write(path, _markdown_report(report))


def _write_text_report(path: str, report: Mapping[str, Any]) -> None:
    """Write the evaluation report as plain text using an atomic replacement."""
    atomic_write(path, _text_report(report))


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
    datasets_dir: str = "tests/real_world",
    skip_hassil: bool = False,
    skip_ablations: bool = False,
    output_txt: str | None = None,
) -> bool:
    """Run evaluation on the datasets and print the summary report."""
    _bootstrap_project_imports()
    if not datasets:
        print(f"Error: No datasets found in {datasets_dir}/")
        return False
    if failure_limit < 0:
        print("Error: --failure-limit must be zero or positive")
        return False

    print("=" * 120)
    print("ASSIST CANONICALIZER PERFORMANCE EVALUATION REPORT")
    print("=" * 120)
    print(f"Dataset Directory: {datasets_dir}/")
    print(f"Total Languages: {len(datasets)}")
    print(f"Failure Detail Limit: {failure_limit}")
    if output_json:
        try:
            rel_json = Path(output_json).relative_to(_REPO_ROOT)
        except ValueError:
            rel_json = Path(output_json)
        print(f"JSON Output: {rel_json}")
    if output_md:
        try:
            rel_md = Path(output_md).relative_to(_REPO_ROOT)
        except ValueError:
            rel_md = Path(output_md)
        print(f"Markdown Output: {rel_md}")
    print("=" * 120)

    overall_success = True
    global_results = _new_results()
    global_ablations = _new_ablation_results()
    all_case_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "languages": {},
        "datasets_dir": datasets_dir,
        "total_languages": len(datasets),
        "failure_limit": failure_limit,
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
        coverage: dict[str, Any] = _coverage_payload(
            test_cases,
            sources,
            candidate_intents,
            len(lang_candidates),
            build_latency_ms,
        )
        coverage["case_count"] = len(test_cases)
        _print_coverage(lang, test_cases, coverage)

        merged_intents = {}
        for s in sources.values():
            hassil.merge_dict(merged_intents, s)
        hassil_intents = hassil.intents.Intents.from_dict(merged_intents)
        hassil_slot_lists = make_hassil_slot_lists(slots)

        print(f"Processing {lang.upper()} ({len(test_cases)} cases) ...", flush=True)

        results = _new_results()
        ablations = _new_ablation_results()
        failures: list[dict[str, Any]] = []
        language_rows: list[dict[str, Any]] = []

        runtime = CanonicalizerRuntime()
        runtime.set_index(index)
        runtime.update_registry_slot_values(slots)

        modes: list[str] = []
        if not skip_hassil:
            modes.append("hassil")
        modes.append("lexical")

        for mode_name in modes:
            for case in test_cases:
                category = case["category"]
                query = case["query"]
                expected_canonical = case["expected_canonical"]
                expected_intent = case["expected_intent"]
                expected_slots = case.get("expected_slots", {})
                stats = _stats_for(results, mode_name, category)

                start_time = time.perf_counter()
                if mode_name == "hassil":
                    res_list = run_hassil_recognize_all(query, hassil_intents, hassil_slot_lists)
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
                if mode_name == "lexical" and not skip_ablations:
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
        if not skip_ablations:
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
    if not skip_ablations:
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
    if output_txt:
        _write_text_report(output_txt, report)
    print("\nEvaluation Complete.")
    return overall_success


# ---------------------------------------------------------------------------
# Mode: Performance Profiling
# ---------------------------------------------------------------------------

DEFAULT_ITERATIONS = 10
DEFAULT_WARMUP = 3
DEFAULT_GRANULARITY = "medium"
DEFAULT_MAX_REGRESSION_PCT = 10.0
BENCHMARK_DIR = "scratch/profile"
BASELINE_DIR = "scratch/profile/baseline"

MODE_ACCURACY = "accuracy"
MODE_PERFORMANCE = "performance"
BENCHMARK_MODES = (MODE_ACCURACY, MODE_PERFORMANCE)

PROFILING_TARGETS = ("evaluate", "build_index", "rank", "components", "all")
GRANULARITY_LEVELS = ("coarse", "medium", "fine")

SCORING_COMPONENT_NAMES = (
    "normalize_text",
    "normalize_text_no_diacritics",
    "char_ngrams_normalized",
    "bm25_score",
    "char_ngram_score",
    "rapidfuzz_similarity",
    "exact_intent_score",
    "positional_intent_score",
    "lexical_score",
    "intent_disambiguation",
)


@dataclass
class StatsResult:
    """Aggregated statistics for a metric across iterations."""

    mean: float
    median: float
    p50: float
    p95: float
    p99: float
    stddev: float
    min_val: float
    max_val: float
    cov: float
    raw_values: list[float] = field(default_factory=list, repr=False)


class StatsEngine:
    """Statistical aggregation for multi-iteration profiling data."""

    @staticmethod
    def compute(values: list[float]) -> StatsResult:
        """Compute aggregate statistics for a list of measurement values."""
        if not values:
            return StatsResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean = statistics.mean(sorted_vals)
        median = statistics.median(sorted_vals)
        stddev = statistics.stdev(sorted_vals) if n >= 2 else 0.0
        cov = (stddev / mean * 100.0) if mean > 0 else 0.0

        def _percentile(p: float) -> float:
            k = (n - 1) * p / 100.0
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_vals[int(k)]
            lower = sorted_vals[f]
            upper = sorted_vals[c] if c < n else sorted_vals[-1]
            return lower + (upper - lower) * (k - f)

        return StatsResult(
            mean=mean,
            median=median,
            p50=_percentile(50),
            p95=_percentile(95),
            p99=_percentile(99),
            stddev=stddev,
            min_val=sorted_vals[0],
            max_val=sorted_vals[-1],
            cov=cov,
            raw_values=list(values),
        )

    @staticmethod
    def as_dict(result: StatsResult) -> dict[str, Any]:
        """Serialize a StatsResult to a JSON-compatible dictionary."""
        return {
            "mean": round(result.mean, 6),
            "median": round(result.median, 6),
            "p50": round(result.p50, 6),
            "p95": round(result.p95, 6),
            "p99": round(result.p99, 6),
            "stddev": round(result.stddev, 6),
            "min": round(result.min_val, 6),
            "max": round(result.max_val, 6),
            "cov_pct": round(result.cov, 1),
        }


class _PhaseContext:
    """Context manager returned by PhaseTimer.phase."""

    def __init__(self, timer: PhaseTimer, name: str) -> None:
        """Initialize the phase timing context manager."""
        self._timer = timer
        self._name = name

    def __enter__(self) -> _PhaseContext:
        """Enter the phase context and start the timer."""
        self._timer.start(self._name)
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the phase context and stop the timer."""
        self._timer.stop()


class PhaseTimer:
    """Hierarchical phase timing with memory delta tracking."""

    def __init__(self, resource_monitor: ResourceMonitor | None = None) -> None:
        """Initialize PhaseTimer with an optional ResourceMonitor."""
        self.phases: dict[str, list[float]] = {}
        self.memory_deltas: dict[str, list[float]] = {}
        self.current_phase: str | None = None
        self._stack: list[tuple[str, float, float]] = []
        self._monitor = resource_monitor

    def _current_rss(self) -> float:
        """Get the current process RSS memory utilization in MB."""
        if self._monitor is not None:
            return self._monitor.current_rss_mb
        try:
            with open("/proc/self/stat", "rb") as f:
                parts = f.read().decode().split()
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(parts[23]) * page_size / (1024 * 1024)
        except Exception:
            return 0.0

    def start(self, name: str) -> None:
        """Start measuring a phase by name."""
        rss = self._current_rss()
        self.current_phase = name
        self._stack.append((name, time.perf_counter(), rss))

    def stop(self) -> None:
        """Stop the current active phase timing and record statistics."""
        if not self._stack:
            return
        name, start_time, start_rss = self._stack.pop()
        elapsed = time.perf_counter() - start_time
        rss_delta = max(0.0, self._current_rss() - start_rss)
        self.phases.setdefault(name, []).append(elapsed)
        self.memory_deltas.setdefault(name, []).append(rss_delta)
        self.current_phase = self._stack[-1][0] if self._stack else None

    def phase(self, name: str) -> _PhaseContext:
        """Return a context manager to easily measure a phase."""
        return _PhaseContext(self, name)

    def record(self, name: str, elapsed: float, rss_delta: float = 0.0) -> None:
        """Directly record performance numbers for a given phase name."""
        self.phases.setdefault(name, []).append(elapsed)
        self.memory_deltas.setdefault(name, []).append(rss_delta)

    def stats(self) -> dict[str, dict[str, StatsResult]]:
        """Compute statistical data for all recorded phases."""
        result: dict[str, dict[str, StatsResult]] = {}
        for name in self.phases:
            result[name] = {
                "elapsed": StatsEngine.compute(self.phases[name]),
                "memory_delta_mb": StatsEngine.compute(self.memory_deltas.get(name, [0.0])),
            }
        return result


class ResourceMonitor(threading.Thread):
    """Monitors CPU, memory (RSS, VmSize, VmPeak), and GC of the current process."""

    def __init__(self, interval: float = 0.02) -> None:
        """Initialize the background resource monitoring thread."""
        super().__init__(daemon=True)
        self.interval: float = interval
        self.stop_event: threading.Event = threading.Event()
        self._lock: threading.Lock = threading.Lock()
        self.cpu_samples: list[tuple[float, float]] = []
        self.rss_samples: list[float] = []
        self.vm_size_samples: list[float] = []
        self.vm_peak_samples: list[float] = []
        self.current_rss_mb: float = 0.0
        self.current_vm_size_mb: float = 0.0
        self.current_vm_peak_mb: float = 0.0
        self.gc_snapshots: list[dict[str, Any]] = []

        try:
            self.clk_tck: float = float(os.sysconf("SC_CLK_TCK"))
        except Exception:
            self.clk_tck = 100.0

        try:
            self.page_size: int = os.sysconf("SC_PAGE_SIZE")
        except Exception:
            self.page_size = 4096

    def run(self) -> None:
        """Periodically sample resource usage statistics from procfs."""
        while not self.stop_event.is_set():
            try:
                t = time.perf_counter()
                with open("/proc/self/stat", "rb") as f:
                    stat_line = f.read().decode("utf-8")
                parts = stat_line.split()
                utime = int(parts[13])
                stime = int(parts[14])
                rss_pages = int(parts[23])
                rss_mb = (rss_pages * self.page_size) / (1024 * 1024)

                vm_size = 0
                vm_peak = 0
                try:
                    with open("/proc/self/status") as sf:
                        for line in sf:
                            if line.startswith("VmSize:"):
                                vm_size = int(line.split()[1])
                            elif line.startswith("VmPeak:"):
                                vm_peak = int(line.split()[1])
                except OSError:
                    pass

                with self._lock:
                    self.cpu_samples.append((t, float(utime + stime)))
                    self.rss_samples.append(rss_mb)
                    self.vm_size_samples.append(vm_size / 1024.0)
                    self.vm_peak_samples.append(vm_peak / 1024.0)
                    self.current_rss_mb = rss_mb
                    self.current_vm_size_mb = vm_size / 1024.0
                    self.current_vm_peak_mb = vm_peak / 1024.0
            except Exception:
                pass
            time.sleep(self.interval)

    def stop_monitor(self) -> None:
        """Signal the monitoring thread to stop execution."""
        self.stop_event.set()

    def snapshot_gc(self) -> None:
        """Take a snapshot of current garbage collection stats."""
        try:
            gc_stats = gc.get_stats()
            self.gc_snapshots.append(
                {
                    "timestamp": time.perf_counter(),
                    "generations": [
                        {
                            "collections": gen.get("collections", 0),
                            "collected": gen.get("collected", 0),
                            "uncollectable": gen.get("uncollectable", 0),
                        }
                        for gen in gc_stats
                    ],
                    "gc_enabled": gc.isenabled(),
                }
            )
        except Exception:
            pass

    def get_cpu_metrics(self) -> dict[str, float]:
        """Compute average and peak CPU utilization percentages."""
        with self._lock:
            if len(self.cpu_samples) < 2:
                return {"avg_pct": 0.0, "peak_pct": 0.0}
            percentages: list[float] = []
            for i in range(1, len(self.cpu_samples)):
                t1, ticks1 = self.cpu_samples[i - 1]
                t2, ticks2 = self.cpu_samples[i]
                dt = t2 - t1
                dticks = ticks2 - ticks1
                if dt > 0:
                    pct = (dticks / self.clk_tck) / dt * 100.0
                    percentages.append(pct)
            return {
                "avg_pct": sum(percentages) / len(percentages) if percentages else 0.0,
                "peak_pct": max(percentages) if percentages else 0.0,
            }

    def get_memory_metrics(self) -> dict[str, float]:
        """Compute average and peak memory (RSS and Vm) metrics in MB."""
        with self._lock:
            return {
                "rss_avg_mb": sum(self.rss_samples) / len(self.rss_samples)
                if self.rss_samples
                else 0.0,
                "rss_peak_mb": max(self.rss_samples) if self.rss_samples else 0.0,
                "vm_size_avg_mb": sum(self.vm_size_samples) / len(self.vm_size_samples)
                if self.vm_size_samples
                else 0.0,
                "vm_size_peak_mb": max(self.vm_size_samples) if self.vm_size_samples else 0.0,
                "vm_peak_mb": max(self.vm_peak_samples) if self.vm_peak_samples else 0.0,
            }


class BaselineManager:
    """Load, save, and compare performance baselines for regression detection."""

    def __init__(self, repo_root: str) -> None:
        """Initialize BaselineManager with the repository root path."""
        self._baseline_dir: Path = Path(repo_root) / BASELINE_DIR

    def load(self, target: str, *, warn_on_missing: bool = False) -> dict[str, Any] | None:
        """Load baseline data for a profiling target."""
        path = self._baseline_dir / f"{target}_baseline.json"
        if not path.is_file():
            if warn_on_missing:
                print(f"Warning: baseline file not found: {path}", file=sys.stderr)
            return None
        try:
            data_str = path.read_text(encoding="utf-8")
            data = json.loads(data_str)
            if not isinstance(data, dict):
                raise ValueError("Content is not a JSON object")
            return data
        except Exception as err:
            if warn_on_missing:
                print(f"Warning: cannot load baseline file: {path} — {err}", file=sys.stderr)
            return None

    def save(self, target: str, data: dict[str, Any]) -> None:
        """Save current profiling data as the baseline for a target."""
        self._baseline_dir.mkdir(parents=True, exist_ok=True)
        path = self._baseline_dir / f"{target}_baseline.json"
        atomic_write(str(path), json.dumps(data, indent=2) + "\n")
        print(f"Baseline saved to {path}")

    def compare(
        self,
        target: str,
        current: dict[str, Any],
        max_regression_pct: float = DEFAULT_MAX_REGRESSION_PCT,
        *,
        warn_on_missing: bool = False,
    ) -> list[str]:
        """Compare current results against baseline data to detect regressions."""
        baseline = self.load(target, warn_on_missing=warn_on_missing)
        if baseline is None:
            return []

        regressions: list[str] = []

        def _check(key_path: str, cur: float, base: float, label: str) -> None:
            if base == 0:
                return
            pct_change = (cur - base) / base * 100.0
            if pct_change > max_regression_pct:
                regressions.append(
                    f"REGRESSION [{label}]: {pct_change:+.1f}% "
                    f"(baseline={base:.4f}, current={cur:.4f})"
                )

        def _compare_recursive(cur_val: Any, base_val: Any, path: str) -> None:
            if isinstance(cur_val, dict) and isinstance(base_val, dict):
                for k in cur_val:
                    if k in base_val:
                        _compare_recursive(cur_val[k], base_val[k], f"{path}.{k}" if path else k)
            elif isinstance(cur_val, (int, float)) and isinstance(base_val, (int, float)):
                leaf = path.split(".")[-1] if path else ""
                if leaf in ("mean", "median", "p95", "p99", "max", "min") or "." not in path:
                    _check(path, float(cur_val), float(base_val), path)

        _compare_recursive(current, baseline, "")
        return regressions


class ReportGenerator:
    """Generate multi-format profiling reports (terminal, JSON, Markdown, text)."""

    @staticmethod
    def terminal(report: dict[str, Any]) -> None:
        """Print the profiling report to the terminal."""
        print("\n" + "=" * 90)
        print("ALGORITHMIC PERFORMANCE PROFILING REPORT")
        print("=" * 90)
        print(f"Target:          {report.get('target', 'unknown')}")
        print(f"Iterations:      {report.get('iterations', 0)}")
        print(f"Warmup:          {report.get('warmup', 0)}")
        print(f"Granularity:     {report.get('granularity', 'coarse')}")
        langs = report.get("languages", [])
        print(f"Languages:       {', '.join(langs) if langs else 'all'}")
        print("-" * 90)

        agg = report.get("aggregate", {})
        if agg:
            _print_stat_block("Aggregate Performance", agg)

        res = report.get("resource", {})
        if res:
            _print_resource_block("Resource Utilization", res)

        phases = report.get("phases", {})
        if phases:
            _print_phase_table("Phase Timing Breakdown", phases)

        components = report.get("components", {})
        if components:
            _print_phase_table("Scoring Component Micro-Profile", components)

        per_lang = report.get("per_language", {})
        for lang_key in sorted(per_lang):
            lang_data = per_lang[lang_key]
            print(f"\n{'─' * 90}")
            print(f"Language: {lang_key.upper()}")
            if lang_data.get("aggregate"):
                _print_stat_block("  Aggregate", lang_data["aggregate"])
            if lang_data.get("phases"):
                _print_phase_table("  Phase Timing", lang_data["phases"])

        regressions = report.get("regressions", [])
        if regressions:
            print(f"\n{'!' * 90}")
            print("REGRESSION DETECTIONS:")
            for r in regressions:
                print(f"  {r}")
            print(f"{'!' * 90}")

        stability = report.get("stability", {})
        if stability:
            print(f"\nStability: {stability}")

        print("\n" + "=" * 90)
        print("PROFILE_OK")
        print("=" * 90)

    @staticmethod
    def json_report(report: dict[str, Any], path: str) -> None:
        """Save the profiling report to a JSON file."""
        out = Path(path)

        def _serialize(obj: object) -> object:
            if isinstance(obj, StatsResult):
                return StatsEngine.as_dict(obj)
            if isinstance(obj, dict):
                return {str(k): _serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_serialize(v) for v in obj]
            return obj

        serializable = _serialize(report)
        atomic_write(str(out), json.dumps(serializable, indent=2, default=str) + "\n")
        print(f"JSON report saved to {out}")

    @staticmethod
    def markdown_report(report: dict[str, Any], path: str) -> None:
        """Save the profiling report to a Markdown file."""
        lines: list[str] = [
            "# Assist Canonicalizer — Algorithmic Performance Profile",
            "",
            f"**Target:** `{report.get('target', 'unknown')}`  ",
            f"**Iterations:** {report.get('iterations', 0)} | "
            f"**Warmup:** {report.get('warmup', 0)} | "
            f"**Granularity:** {report.get('granularity', 'coarse')}",
            "",
        ]

        agg = report.get("aggregate", {})
        if agg:
            lines.extend(_md_stat_table("## Aggregate Performance", agg))

        phases = report.get("phases", {})
        if phases:
            lines.append("## Phase Timing")
            lines.append("")
            ph_headers = (
                "Phase",
                "Mean (ms)",
                "Median (ms)",
                "p95 (ms)",
                "p99 (ms)",
                "StdDev (ms)",
                "Memory Δ (MB)",
            )
            ph_rows: list[tuple[str, ...]] = []
            for name, phase_data in phases.items():
                e = phase_data.get("elapsed", {})
                m = phase_data.get("memory_delta_mb", {})
                ph_rows.append(
                    (
                        name,
                        f"{e.get('mean', 0) * 1000:.2f}",
                        f"{e.get('median', 0) * 1000:.2f}",
                        f"{e.get('p95', 0) * 1000:.2f}",
                        f"{e.get('p99', 0) * 1000:.2f}",
                        f"{e.get('stddev', 0) * 1000:.2f}",
                        f"{m.get('mean', 0):.2f}",
                    )
                )
            lines.extend(_md_aligned_table(ph_headers, "<>", ph_rows))
            lines.append("")

        components = report.get("components", {})
        if components:
            lines.append("## Scoring Component Micro-Profile")
            lines.append("")
            cp_headers = ("Component", "Mean (μs)", "Median (μs)", "p95 (μs)", "p99 (μs)", "CoV%")
            cp_rows: list[tuple[str, ...]] = []
            for name, comp_data in components.items():
                e = comp_data.get("elapsed", {})
                cp_rows.append(
                    (
                        name,
                        f"{e.get('mean', 0) * 1_000_000:.1f}",
                        f"{e.get('median', 0) * 1_000_000:.1f}",
                        f"{e.get('p95', 0) * 1_000_000:.1f}",
                        f"{e.get('p99', 0) * 1_000_000:.1f}",
                        f"{e.get('cov_pct', 0):.1f}",
                    )
                )
            lines.extend(_md_aligned_table(cp_headers, "<>", cp_rows))
            lines.append("")

        regressions = report.get("regressions", [])
        if regressions:
            lines.append("## Regression Detections")
            lines.append("")
            for r in regressions:
                lines.append(f"- {r}")
            lines.append("")

        while lines and lines[-1] == "":
            lines.pop()
        atomic_write(path, "\n".join(lines) + "\n")
        print(f"Markdown report saved to {Path(path)}")

    @staticmethod
    def text_report(report: dict[str, Any], path: str) -> None:
        """Save the profiling report to a text file."""
        lines: list[str] = [
            "ALGORITHMIC PERFORMANCE PROFILING REPORT",
            "=" * 80,
            f"Target: {report.get('target', 'unknown')}",
            f"Iterations: {report.get('iterations', 0)}",
            f"Warmup: {report.get('warmup', 0)}",
            f"Granularity: {report.get('granularity', 'coarse')}",
            "-" * 80,
        ]

        _headers_agg = (
            "Metric",
            "Mean",
            "Median",
            "p95",
            "p99",
            "StdDev",
            "Min",
            "Max",
            "CoV%",
        )
        _headers_ph = (
            "Phase",
            "Mean(ms)",
            "Median(ms)",
            "p95(ms)",
            "p99(ms)",
            "Std(ms)",
            "MemΔ(MB)",
        )

        agg = report.get("aggregate", {})
        if agg:
            _rows_agg: list[tuple[str, ...]] = []
            for name, s in agg.items():
                if isinstance(s, dict):
                    _rows_agg.append(
                        (
                            name,
                            f"{s.get('mean', 0):.4f}",
                            f"{s.get('median', 0):.4f}",
                            f"{s.get('p95', 0):.4f}",
                            f"{s.get('p99', 0):.4f}",
                            f"{s.get('stddev', 0):.4f}",
                            f"{s.get('min', 0):.4f}",
                            f"{s.get('max', 0):.4f}",
                            f"{s.get('cov_pct', 0):.1f}%",
                        )
                    )
            if _rows_agg:
                hdr, sep, data = align_table(_headers_agg, _rows_agg, alignments="<>")
                lines.append("\nAggregate Performance:")
                lines.append(hdr)
                lines.append(sep)
                lines.extend(data)

        phases = report.get("phases", {})
        if phases:
            _rows_ph: list[tuple[str, ...]] = []
            for name, phase_data in phases.items():
                e = phase_data.get("elapsed", {})
                m = phase_data.get("memory_delta_mb", {})
                _rows_ph.append(
                    (
                        name,
                        f"{e.get('mean', 0) * 1000:.3f}",
                        f"{e.get('median', 0) * 1000:.3f}",
                        f"{e.get('p95', 0) * 1000:.3f}",
                        f"{e.get('p99', 0) * 1000:.3f}",
                        f"{e.get('stddev', 0) * 1000:.3f}",
                        f"{m.get('mean', 0):.3f}",
                    )
                )
            if _rows_ph:
                hdr, sep, data = align_table(_headers_ph, _rows_ph, alignments="<>")
                lines.extend(["", "Phase Timing Breakdown:", hdr, sep, *data])

        components = report.get("components", {})
        if components:
            _headers_cp = ("Component", "Mean(μs)", "Median(μs)", "p95(μs)", "p99(μs)", "CoV%")
            _rows_cp: list[tuple[str, ...]] = []
            for name, comp_data in components.items():
                e = comp_data.get("elapsed", {})
                _rows_cp.append(
                    (
                        name,
                        f"{e.get('mean', 0) * 1_000_000:.1f}",
                        f"{e.get('median', 0) * 1_000_000:.1f}",
                        f"{e.get('p95', 0) * 1_000_000:.1f}",
                        f"{e.get('p99', 0) * 1_000_000:.1f}",
                        f"{e.get('cov_pct', 0):.1f}",
                    )
                )
            if _rows_cp:
                hdr, sep, data = align_table(_headers_cp, _rows_cp, alignments="<>")
                lines.extend(["", "Scoring Component Micro-Profile:", hdr, sep, *data])

        per_lang = report.get("per_language", {})
        for lang_key in sorted(per_lang):
            lang_data = per_lang[lang_key]
            lines.extend(["", "─" * 80, f"Language: {lang_key.upper()}"])
            lang_agg = lang_data.get("aggregate", {})
            if lang_agg:
                _la_rows: list[tuple[str, ...]] = []
                for name, s in lang_agg.items():
                    if isinstance(s, dict):
                        _la_rows.append(
                            (
                                name,
                                f"{s.get('mean', 0):.4f}",
                                f"{s.get('median', 0):.4f}",
                                f"{s.get('p95', 0):.4f}",
                                f"{s.get('p99', 0):.4f}",
                                f"{s.get('stddev', 0):.4f}",
                                f"{s.get('min', 0):.4f}",
                                f"{s.get('max', 0):.4f}",
                                f"{s.get('cov_pct', 0):.1f}%",
                            )
                        )
                if _la_rows:
                    hdr, sep, data = align_table(_headers_agg, _la_rows, alignments="<>")
                    lines.extend(["  Aggregate:", "  " + hdr, "  " + sep])
                    lines.extend("  " + d for d in data)

            lang_phases = lang_data.get("phases", {})
            if lang_phases:
                _lp_rows: list[tuple[str, ...]] = []
                for name, phase_data in lang_phases.items():
                    e = phase_data.get("elapsed", {})
                    m = phase_data.get("memory_delta_mb", {})
                    _lp_rows.append(
                        (
                            name,
                            f"{e.get('mean', 0) * 1000:.3f}",
                            f"{e.get('median', 0) * 1000:.3f}",
                            f"{e.get('p95', 0) * 1000:.3f}",
                            f"{e.get('p99', 0) * 1000:.3f}",
                            f"{e.get('stddev', 0) * 1000:.3f}",
                            f"{m.get('mean', 0):.3f}",
                        )
                    )
                if _lp_rows:
                    hdr, sep, data = align_table(_headers_ph, _lp_rows, alignments="<>")
                    lines.extend(["  Phase Timing:", "  " + hdr, "  " + sep])
                    lines.extend("  " + d for d in data)

        regressions = report.get("regressions", [])
        if regressions:
            lines.append("\nREGRESSION DETECTIONS:")
            for r in regressions:
                lines.append(f"  {r}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("PROFILE_OK")
        lines.append("=" * 80)

        while lines and lines[-1] == "":
            lines.pop()
        atomic_write(path, "\n".join(lines) + "\n")
        print(f"Text report saved to {Path(path)}")


def _print_stat_block(title: str, stats: dict[str, Any]) -> None:
    """Print standard stats block to the terminal in tabular format."""
    print(f"\n{title}:")
    _headers = ("Metric", "Mean", "Median", "p95", "p99", "StdDev", "Min", "Max", "CoV%")
    _rows: list[tuple[str, ...]] = []
    for name, s in stats.items():
        if isinstance(s, dict):
            _rows.append(
                (
                    name,
                    f"{s.get('mean', 0):.4f}",
                    f"{s.get('median', 0):.4f}",
                    f"{s.get('p95', 0):.4f}",
                    f"{s.get('p99', 0):.4f}",
                    f"{s.get('stddev', 0):.4f}",
                    f"{s.get('min', 0):.4f}",
                    f"{s.get('max', 0):.4f}",
                    f"{s.get('cov_pct', 0):.1f}%",
                )
            )
    hdr, sep, data = align_table(_headers, _rows, alignments="<>")
    print(hdr)
    print(sep)
    for line in data:
        print(line)


def _print_resource_block(title: str, stats: dict[str, float]) -> None:
    """Print resource utilization statistics (CPU/memory) to the terminal."""
    print(f"\n{title}:")
    for key, val in stats.items():
        print(f"  {key}: {val:.2f}")


def _print_phase_table(title: str, phases: dict[str, Any]) -> None:
    """Print the phase execution table to the terminal."""
    print(f"\n{title}:")
    _headers = ("Phase", "Mean(ms)", "Median(ms)", "p95(ms)", "p99(ms)", "Std(ms)", "MemΔ(MB)")
    _rows: list[tuple[str, ...]] = []
    for name, phase_data in phases.items():
        e = phase_data.get("elapsed", {})
        m = phase_data.get("memory_delta_mb", {})
        _rows.append(
            (
                name,
                f"{e.get('mean', 0) * 1000:.3f}",
                f"{e.get('median', 0) * 1000:.3f}",
                f"{e.get('p95', 0) * 1000:.3f}",
                f"{e.get('p99', 0) * 1000:.3f}",
                f"{e.get('stddev', 0) * 1000:.3f}",
                f"{m.get('mean', 0):.3f}",
            )
        )
    hdr, sep, data = align_table(_headers, _rows, alignments="<>")
    print(hdr)
    print(sep)
    for line in data:
        print(line)


def _md_aligned_table(
    headers: tuple[str, ...],
    alignments: str,
    rows: list[tuple[str, ...]],
) -> list[str]:
    """Generate a markdown formatted table with dynamically aligned columns."""
    if not headers:
        return []
    ncols = len(headers)
    if len(alignments) == 1:
        aligns = list(alignments * ncols)
    else:
        aligns = list(alignments)
        if len(aligns) < ncols:
            aligns.extend([aligns[-1]] * (ncols - len(aligns)))
        aligns = aligns[:ncols]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    widths = [max(w, 3) for w in widths]

    lines: list[str] = []
    hdr_parts = [f" {h:{a}{w}} " for h, a, w in zip(headers, aligns, widths, strict=True)]
    lines.append("|" + "|".join(hdr_parts) + "|")

    sep_parts: list[str] = []
    for i, w in enumerate(widths):
        dashes = w - 1
        if aligns[i] == ">":
            sep_parts.append(" " + "-" * dashes + ": ")
        else:
            sep_parts.append(" :" + "-" * dashes + " ")
    lines.append("|" + "|".join(sep_parts) + "|")

    for row in rows:
        parts = [f" {c:{a}{w}} " for c, a, w in zip(row, aligns, widths, strict=True)]
        lines.append("|" + "|".join(parts) + "|")

    return lines


def _md_stat_table(title: str, stats: dict[str, Any]) -> list[str]:
    """Generate a markdown formatted table containing stats results."""
    headers = ("Metric", "Mean", "Median", "p95", "p99", "StdDev", "Min", "Max", "CoV%")
    rows: list[tuple[str, ...]] = []
    for name, s in stats.items():
        if isinstance(s, dict):
            rows.append(
                (
                    name,
                    f"{s.get('mean', 0):.4f}",
                    f"{s.get('median', 0):.4f}",
                    f"{s.get('p95', 0):.4f}",
                    f"{s.get('p99', 0):.4f}",
                    f"{s.get('stddev', 0):.4f}",
                    f"{s.get('min', 0):.4f}",
                    f"{s.get('max', 0):.4f}",
                    f"{s.get('cov_pct', 0):.1f}",
                )
            )
    lines: list[str] = [title, ""]
    if rows:
        lines.extend(_md_aligned_table(headers, "<>", rows))
        lines.append("")
    return lines


def _profile_evaluate(
    datasets: dict[str, str],
    iterations: int,
    warmup: int,
    granularity: str,
    timer: PhaseTimer,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    """Profile the full evaluate accuracy pipeline."""
    output_dir = Path(_REPO_ROOT) / BENCHMARK_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = str(output_dir / "performance_benchmark_report.json")
    md_path = str(output_dir / "performance_benchmark_report.md")

    total_runs = warmup + iterations
    aggregate_elapsed: list[float] = []

    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup
        label = f"evaluate_run_{run_idx}"

        if is_warmup:
            print(f"Warmup run {run_idx + 1}/{warmup} ...", flush=True)
        else:
            print(f"Profiling run {run_idx - warmup + 1}/{iterations} ...", flush=True)

        if granularity == "coarse" or is_warmup:
            timer.start(label)
            gc.collect()
            monitor.snapshot_gc()

            asyncio.run(
                run_evaluation(
                    datasets=dict(datasets),
                    failure_limit=0,
                    output_json=json_path if not is_warmup else None,
                    output_md=md_path if not is_warmup else None,
                    min_intent_slot_accuracy=None,
                    max_fallback_rate=None,
                )
            )
            timer.stop()
        elif granularity == "medium":
            timer.start(label)
            gc.collect()
            monitor.snapshot_gc()
            with timer.phase("evaluate_total"):
                asyncio.run(
                    run_evaluation(
                        datasets=dict(datasets),
                        failure_limit=0,
                        output_json=json_path if not is_warmup else None,
                        output_md=md_path if not is_warmup else None,
                        min_intent_slot_accuracy=None,
                        max_fallback_rate=None,
                    )
                )
            timer.stop()
        else:  # fine
            timer.start(label)
            gc.collect()
            monitor.snapshot_gc()
            with timer.phase("evaluate_total"):
                asyncio.run(
                    run_evaluation(
                        datasets=dict(datasets),
                        failure_limit=0,
                        output_json=json_path if not is_warmup else None,
                        output_md=md_path if not is_warmup else None,
                        min_intent_slot_accuracy=None,
                        max_fallback_rate=None,
                    )
                )
            timer.stop()

        if not is_warmup and label in timer.phases:
            aggregate_elapsed.append(timer.phases[label][-1])

    result: dict[str, Any] = {}
    if aggregate_elapsed:
        result["aggregate"] = {
            "evaluate_wall_time_sec": StatsEngine.as_dict(StatsEngine.compute(aggregate_elapsed))
        }
    return result


def _profile_build_index(
    datasets: dict[str, str],
    iterations: int,
    warmup: int,
    granularity: str,
    timer: PhaseTimer,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    """Profile only index building."""
    _bootstrap_project_imports()
    total_runs = warmup + iterations
    per_language: dict[str, dict[str, Any]] = {}
    all_elapsed: list[float] = []

    for lang, path in sorted(datasets.items()):
        try:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        slots = _dataset_registry_slots(data, lang)
        lang_elapsed: list[float] = []

        for run_idx in range(total_runs):
            is_warmup = run_idx < warmup
            label = f"build_index_{lang}"

            sources = load_language_intent_sources(lang)
            gc.collect()
            monitor.snapshot_gc()

            timer.start(label)
            with timer.phase(f"build_candidates_{lang}"):
                candidates = build_candidates_from_intent_sources(lang, sources, slots)
            with timer.phase(f"build_index_{lang}"):
                build_index(lang, candidates)
            timer.stop()

            if not is_warmup and label in timer.phases:
                lang_elapsed.append(timer.phases[label][-1])
                all_elapsed.append(timer.phases[label][-1])

        if lang_elapsed:
            per_language[lang] = {
                "aggregate": {
                    "build_index_wall_time_sec": StatsEngine.as_dict(
                        StatsEngine.compute(lang_elapsed)
                    )
                }
            }

    result: dict[str, Any] = {"per_language": per_language}
    if all_elapsed:
        result["aggregate"] = {
            "build_index_wall_time_sec_total": StatsEngine.as_dict(StatsEngine.compute(all_elapsed))
        }
    return result


def _profile_rank(
    datasets: dict[str, str],
    iterations: int,
    warmup: int,
    granularity: str,
    timer: PhaseTimer,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    """Profile candidate ranking hot path."""
    _bootstrap_project_imports()
    total_runs = warmup + iterations
    per_language: dict[str, dict[str, Any]] = {}
    all_elapsed: list[float] = []

    for lang, path in sorted(datasets.items()):
        try:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        slots = _dataset_registry_slots(data, lang)
        sources = load_language_intent_sources(lang)
        candidates = build_candidates_from_intent_sources(lang, sources, slots)
        index = build_index(lang, candidates)
        raw_cases = data.get("test_cases", [])
        if not isinstance(raw_cases, list):
            continue

        queries = [
            case["query"]
            for case in raw_cases
            if isinstance(case, dict) and isinstance(case.get("query"), str)
        ]
        if not queries:
            continue

        lang_elapsed: list[float] = []
        per_query_times: list[list[float]] = [[] for _ in queries]

        for run_idx in range(total_runs):
            is_warmup = run_idx < warmup
            label = f"rank_{lang}"

            gc.collect()
            monitor.snapshot_gc()

            timer.start(label)
            if granularity != "fine":
                for qi, query in enumerate(queries):
                    q_start = time.perf_counter()
                    _ = index.rank(query)
                    q_elapsed = time.perf_counter() - q_start
                    if not is_warmup:
                        per_query_times[qi].append(q_elapsed)
            else:
                for qi, query in enumerate(queries):
                    q_start = time.perf_counter()
                    from contextlib import suppress

                    with timer.phase("rank_candidates_inner"), suppress(Exception):
                        _ = index.rank(query)
                    q_elapsed = time.perf_counter() - q_start
                    if not is_warmup:
                        per_query_times[qi].append(q_elapsed)
            timer.stop()

            if not is_warmup and label in timer.phases:
                lang_elapsed.append(timer.phases[label][-1])
                all_elapsed.append(timer.phases[label][-1])

        avg_query_times: list[float] = []
        for q_times in per_query_times:
            if q_times:
                avg_query_times.append(statistics.mean(q_times))

        if lang_elapsed:
            rank_agg: dict[str, Any] = {
                "rank_total_wall_time_sec": StatsEngine.as_dict(StatsEngine.compute(lang_elapsed))
            }
            if avg_query_times:
                rank_agg["rank_per_query_wall_time_sec"] = StatsEngine.as_dict(
                    StatsEngine.compute(avg_query_times)
                )
                rank_agg["query_count"] = len(queries)
            per_language[lang] = {"aggregate": rank_agg}

    result: dict[str, Any] = {"per_language": per_language}
    if all_elapsed:
        result["aggregate"] = {
            "rank_wall_time_sec_total": StatsEngine.as_dict(StatsEngine.compute(all_elapsed))
        }
    return result


def _profile_components(
    datasets: dict[str, str],
    iterations: int,
    warmup: int,
    timer: PhaseTimer,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    """Micro-benchmark isolated scoring components."""
    _bootstrap_project_imports()
    from custom_components.assist_canonicalizer.ranking import (
        _build_positional_lookup,
        _exact_intent_score,
        _positional_intent_score_from_lookup,
    )

    all_queries: list[tuple[str, str, str, str | None]] = []
    all_per_lang: dict[str, list[tuple[str, str, str, str | None]]] = {}

    for lang, path in sorted(datasets.items()):
        try:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        slots = _dataset_registry_slots(data, lang)
        sources = load_language_intent_sources(lang)
        candidates = build_candidates_from_intent_sources(lang, sources, slots)
        build_index(lang, candidates)
        raw_cases = data.get("test_cases", [])
        if not isinstance(raw_cases, list):
            continue

        lang_entries: list[tuple[str, str, str, str | None]] = []
        for case in raw_cases:
            if not isinstance(case, dict):
                continue
            query = case.get("query")
            if not isinstance(query, str):
                continue
            norm = normalize_text(query)
            lang_entries.append((query, norm, lang, None))

        for candidate in candidates[:50]:
            raw_text = candidate.text
            norm_text = candidate.normalized_text
            lit_text = candidate.metadata.get("literal_text") if candidate.metadata else None
            lang_entries.append((raw_text, norm_text, lang, lit_text))

        all_queries.extend(lang_entries)
        all_per_lang[lang] = lang_entries

    total_runs = warmup + iterations
    component_results: dict[str, dict[str, list[float]]] = {
        comp: {"elapsed": []} for comp in SCORING_COMPONENT_NAMES
    }

    all_norm_texts: list[str] = []
    all_candidates: list[Any] = []
    for lang, path in sorted(datasets.items()):
        try:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        slots = _dataset_registry_slots(data, lang)
        sources = load_language_intent_sources(lang)
        candidates = build_candidates_from_intent_sources(lang, sources, slots)
        for c in candidates:
            all_norm_texts.append(c.normalized_text)
            all_candidates.append(c)

    if not all_queries:
        return {}

    bm25_idx = BM25Index.from_normalized_texts(all_norm_texts) if all_norm_texts else None
    char_grams = [char_ngrams_normalized(t) for t in all_norm_texts]
    char_idx = CharNGramIndex.from_grams(char_grams) if char_grams else None

    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup
        for query_raw, query_norm, lang, lit_text in all_queries:
            if is_warmup:
                _ = normalize_text(query_raw)
                _ = normalize_text_no_diacritics(query_raw, lang)
                _ = char_ngrams_normalized(query_norm)
                if bm25_idx is not None:
                    _ = bm25_idx.score(query_norm)
                q_grams = char_ngrams_normalized(query_norm)
                if char_idx is not None:
                    _ = char_idx.score(q_grams)
                if all_norm_texts:
                    _ = rapidfuzz_similarity_normalized(query_norm, all_norm_texts[0])
                q_tokens = frozenset(query_norm.split())
                if lit_text:
                    _ = _exact_intent_score(lit_text, q_tokens)
                _ = lexical_score(0.5, 0.5, 0.5, 0.5)
                continue

            t0 = time.perf_counter()
            _ = normalize_text(query_raw)
            component_results["normalize_text"]["elapsed"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            _ = normalize_text_no_diacritics(query_raw, lang)
            component_results["normalize_text_no_diacritics"]["elapsed"].append(
                time.perf_counter() - t0
            )

            t0 = time.perf_counter()
            _ = char_ngrams_normalized(query_norm)
            component_results["char_ngrams_normalized"]["elapsed"].append(time.perf_counter() - t0)

            if bm25_idx is not None:
                t0 = time.perf_counter()
                _ = bm25_idx.score(query_norm)
                component_results["bm25_score"]["elapsed"].append(time.perf_counter() - t0)

            q_grams = char_ngrams_normalized(query_norm)
            if char_idx is not None:
                t0 = time.perf_counter()
                _ = char_idx.score(q_grams)
                component_results["char_ngram_score"]["elapsed"].append(time.perf_counter() - t0)

            if all_norm_texts:
                t0 = time.perf_counter()
                _ = rapidfuzz_similarity_normalized(query_norm, all_norm_texts[0])
                component_results["rapidfuzz_similarity"]["elapsed"].append(
                    time.perf_counter() - t0
                )

            q_tokens = frozenset(query_norm.split())
            if lit_text:
                t0 = time.perf_counter()
                _ = _exact_intent_score(lit_text, q_tokens)
                component_results["exact_intent_score"]["elapsed"].append(time.perf_counter() - t0)

            if lit_text:
                from custom_components.assist_canonicalizer.ranking import literal_token_variants

                variants = literal_token_variants(lit_text)
                all_lit_tokens = frozenset().union(*variants) if variants else frozenset()
                pos_lookup = _build_positional_lookup(all_lit_tokens, q_tokens)

                t0 = time.perf_counter()
                _ = _positional_intent_score_from_lookup(lit_text, q_tokens, pos_lookup, None)
                component_results["positional_intent_score"]["elapsed"].append(
                    time.perf_counter() - t0
                )

            t0 = time.perf_counter()
            _ = lexical_score(0.5, 0.5, 0.5, 0.5)
            component_results["lexical_score"]["elapsed"].append(time.perf_counter() - t0)

            if all_candidates and len(all_candidates) >= 2:
                fake_scores_a = _ScoreBreakdown(0.8, 0.8, 0.8, 0.9, 0.85)
                fake_scores_b = _ScoreBreakdown(0.8, 0.8, 0.8, 0.95, 0.84)
                fake_ranked = [
                    _RankedCandidate(candidate=all_candidates[0], scores=fake_scores_a),
                    _RankedCandidate(candidate=all_candidates[1], scores=fake_scores_b),
                ]
                from custom_components.assist_canonicalizer.ranking import (
                    _apply_intent_disambiguation,
                )

                t0 = time.perf_counter()
                _apply_intent_disambiguation(fake_ranked)
                component_results["intent_disambiguation"]["elapsed"].append(
                    time.perf_counter() - t0
                )

    result: dict[str, Any] = {}
    for comp_name in SCORING_COMPONENT_NAMES:
        vals = component_results[comp_name]["elapsed"]
        if vals:
            result[comp_name] = {"elapsed": StatsEngine.as_dict(StatsEngine.compute(vals))}
    return result


def _build_report(
    target: str,
    iterations: int,
    warmup: int,
    granularity: str,
    languages: list[str] | None,
    timer: PhaseTimer,
    monitor: ResourceMonitor,
    target_result: dict[str, Any],
    baseline: BaselineManager,
    max_regression_pct: float,
    *,
    warn_on_missing: bool = False,
) -> dict[str, Any]:
    """Construct a unified report dict by combining timers, monitor data and baseline comparison."""
    report: dict[str, Any] = {
        "target": target,
        "iterations": iterations,
        "warmup": warmup,
        "granularity": granularity,
        "languages": languages or [],
    }

    raw_phase_stats = timer.stats()
    phase_stats: dict[str, dict[str, dict[str, Any]]] = {}
    for phase_name, metrics in raw_phase_stats.items():
        phase_stats[phase_name] = {
            metric_name: StatsEngine.as_dict(sr) if isinstance(sr, StatsResult) else sr
            for metric_name, sr in metrics.items()
        }
    if phase_stats:
        report["phases"] = phase_stats

    agg = target_result.get("aggregate", {})
    if agg:
        report["aggregate"] = agg

    comps = target_result.get("components", {})
    if comps:
        report["components"] = comps

    per_lang = target_result.get("per_language", {})
    if per_lang:
        report["per_language"] = per_lang

    cpu_metrics = monitor.get_cpu_metrics()
    mem_metrics = monitor.get_memory_metrics()
    report["resource"] = {**cpu_metrics, **mem_metrics}

    regressions = baseline.compare(
        target, report, max_regression_pct, warn_on_missing=warn_on_missing
    )
    report["regressions"] = regressions

    cov_values: list[float] = []
    for _phase_name, phase_data in phase_stats.items():
        e = phase_data.get("elapsed")
        if isinstance(e, dict) and e.get("cov_pct", 0) > 0:
            cov_values.append(float(e["cov_pct"]))
    if cov_values:
        avg_cov = statistics.mean(cov_values)
        if avg_cov < 5.0:
            report["stability"] = f"High stability — average CoV {avg_cov:.1f}% across phases"
        elif avg_cov < 15.0:
            report["stability"] = f"Moderate stability — average CoV {avg_cov:.1f}% across phases"
        else:
            report["stability"] = (
                f"Low stability — average CoV {avg_cov:.1f}% across phases. "
                "Consider increasing iterations or closing background processes."
            )
    return report


def _run_profiling(
    target: str,
    datasets: dict[str, str],
    iterations: int,
    warmup: int,
    granularity: str,
    baseline: BaselineManager,
    max_regression_pct: float,
    output_json: str | None,
    output_md: str | None,
    output_txt: str | None,
    save_baseline: bool,
    languages: list[str] | None,
    *,
    warn_on_missing: bool = False,
) -> dict[str, Any]:
    """Execute the profiling lifecycle for a target over multiple iterations."""
    print(f"\n{'=' * 90}")
    print(f"Profiling target: {target}")
    print(f"{'=' * 90}")
    print(f"Iterations: {iterations}  |  Warmup: {warmup}  |  Granularity: {granularity}")
    print(f"Languages: {', '.join(sorted(datasets.keys()))}")
    print(f"{'=' * 90}")

    monitor = ResourceMonitor(interval=0.02)
    timer = PhaseTimer(monitor)

    monitor.start()
    monitor.snapshot_gc()
    gc.collect()
    gc.disable()

    try:
        if target == "evaluate":
            target_result = _profile_evaluate(
                datasets, iterations, warmup, granularity, timer, monitor
            )
        elif target == "build_index":
            target_result = _profile_build_index(
                datasets, iterations, warmup, granularity, timer, monitor
            )
        elif target == "rank":
            target_result = _profile_rank(datasets, iterations, warmup, granularity, timer, monitor)
        elif target == "components":
            target_result = _profile_components(datasets, iterations, warmup, timer, monitor)
            target_result = {"components": target_result}
        else:
            print(f"Unknown target: {target}", file=sys.stderr)
            return {}
    finally:
        gc.enable()
        monitor.snapshot_gc()
        monitor.stop_monitor()
        monitor.join(timeout=5.0)

    report = _build_report(
        target,
        iterations,
        warmup,
        granularity,
        languages,
        timer,
        monitor,
        target_result,
        baseline,
        max_regression_pct,
        warn_on_missing=warn_on_missing,
    )

    ReportGenerator.terminal(report)
    if output_json:
        ReportGenerator.json_report(report, output_json)
    if output_md:
        ReportGenerator.markdown_report(report, output_md)
    if output_txt:
        ReportGenerator.text_report(report, output_txt)

    if save_baseline:
        baseline.save(target, report)
    return report


def _write_profile_all_markdown(all_reports: dict[str, Any], path: str) -> None:
    """Generate and write a consolidated Markdown report for all profiling targets."""
    lines: list[str] = [
        "# Assist Canonicalizer — Consolidated Performance Profile (All Targets)",
        "",
        ("This report aggregates performance statistics across all measured profiling targets."),
        "",
    ]

    for target, report in all_reports.items():
        lines.append(f"## Target: `{target}`")
        lines.append("")

        # Aggregate Performance
        agg = report.get("aggregate", {})
        if agg:
            lines.extend(_md_stat_table(f"### Aggregate Performance ({target})", agg))

        # Resource Utilization
        res = report.get("resource", {})
        if res:
            lines.append(f"### Resource Utilization ({target})")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| :--- | :--- |")
            for k, v in sorted(res.items()):
                lines.append(f"| {k} | {v:.2f} |")
            lines.append("")

        # Phase Timing
        phases = report.get("phases", {})
        if phases:
            lines.append(f"### Phase Timing ({target})")
            lines.append("")
            ph_headers = (
                "Phase",
                "Mean (ms)",
                "Median (ms)",
                "p95 (ms)",
                "p99 (ms)",
                "StdDev (ms)",
                "Memory Δ (MB)",
            )
            ph_rows: list[tuple[str, ...]] = []
            for name, phase_data in phases.items():
                e = phase_data.get("elapsed", {})
                m = phase_data.get("memory_delta_mb", {})
                ph_rows.append(
                    (
                        name,
                        f"{e.get('mean', 0) * 1000:.2f}",
                        f"{e.get('median', 0) * 1000:.2f}",
                        f"{e.get('p95', 0) * 1000:.2f}",
                        f"{e.get('p99', 0) * 1000:.2f}",
                        f"{e.get('stddev', 0) * 1000:.2f}",
                        f"{m.get('mean', 0):.2f}",
                    )
                )
            lines.extend(_md_aligned_table(ph_headers, "<>", ph_rows))
            lines.append("")

        # Micro-profile components (if present)
        components = report.get("components", {})
        if components:
            lines.append(f"### Scoring Component Micro-Profile ({target})")
            lines.append("")
            cp_headers = ("Component", "Mean (μs)", "Median (μs)", "p95 (μs)", "p99 (μs)", "CoV%")
            cp_rows: list[tuple[str, ...]] = []
            for name, comp_data in components.items():
                e = comp_data.get("elapsed", {})
                cp_rows.append(
                    (
                        name,
                        f"{e.get('mean', 0) * 1_000_000:.1f}",
                        f"{e.get('median', 0) * 1_000_000:.1f}",
                        f"{e.get('p95', 0) * 1_000_000:.1f}",
                        f"{e.get('p99', 0) * 1_000_000:.1f}",
                        f"{e.get('cov_pct', 0):.1f}",
                    )
                )
            lines.extend(_md_aligned_table(cp_headers, "<>", cp_rows))
            lines.append("")

        # Regressions
        regressions = report.get("regressions", [])
        if regressions:
            lines.append(f"### Regression Detections ({target})")
            lines.append("")
            for r in regressions:
                lines.append(f"- {r}")
            lines.append("")

        lines.append("---")
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    atomic_write(path, "\n".join(lines) + "\n")


def _write_profile_all_text(all_reports: dict[str, Any], path: str) -> None:
    """Generate and write a consolidated plain text report for all profiling targets."""
    lines: list[str] = [
        "ALGORITHMIC PERFORMANCE PROFILING REPORT (ALL TARGETS)",
        "=" * 90,
        "",
    ]
    for target, report in all_reports.items():
        lines.append(f"Target: {target}")
        lines.append("-" * 90)

        agg = report.get("aggregate", {})
        if agg:
            lines.append("Aggregate Performance:")
            _headers = ("Metric", "Mean", "Median", "p95", "p99", "StdDev", "Min", "Max", "CoV%")
            _rows: list[tuple[str, ...]] = []
            for name, s in agg.items():
                if isinstance(s, dict):
                    _rows.append(
                        (
                            name,
                            f"{s.get('mean', 0):.4f}",
                            f"{s.get('median', 0):.4f}",
                            f"{s.get('p95', 0):.4f}",
                            f"{s.get('p99', 0):.4f}",
                            f"{s.get('stddev', 0):.4f}",
                            f"{s.get('min', 0):.4f}",
                            f"{s.get('max', 0):.4f}",
                            f"{s.get('cov_pct', 0):.1f}%",
                        )
                    )
            hdr, sep, data = align_table(_headers, _rows, alignments="<>")
            lines.append(hdr)
            lines.append(sep)
            lines.extend(data)
            lines.append("")

        res = report.get("resource", {})
        if res:
            lines.append("Resource Utilization:")
            for k, v in sorted(res.items()):
                lines.append(f"  {k}: {v:.2f}")
            lines.append("")

        phases = report.get("phases", {})
        if phases:
            lines.append("Phase Timing:")
            _headers_ph = (
                "Phase",
                "Mean(ms)",
                "Median(ms)",
                "p95(ms)",
                "p99(ms)",
                "Std(ms)",
                "MemΔ(MB)",
            )
            _rows_ph: list[tuple[str, ...]] = []
            for name, phase_data in phases.items():
                e = phase_data.get("elapsed", {})
                m = phase_data.get("memory_delta_mb", {})
                _rows_ph.append(
                    (
                        name,
                        f"{e.get('mean', 0) * 1000:.3f}",
                        f"{e.get('median', 0) * 1000:.3f}",
                        f"{e.get('p95', 0) * 1000:.3f}",
                        f"{e.get('p99', 0) * 1000:.3f}",
                        f"{e.get('stddev', 0) * 1000:.3f}",
                        f"{m.get('mean', 0):.3f}",
                    )
                )
            hdr, sep, data = align_table(_headers_ph, _rows_ph, alignments="<>")
            lines.append(hdr)
            lines.append(sep)
            lines.extend(data)
            lines.append("")

        components = report.get("components", {})
        if components:
            lines.append("Scoring Component Micro-Profile:")
            _headers_cp = ("Component", "Mean(μs)", "Median(μs)", "p95(μs)", "p99(μs)", "CoV%")
            _rows_cp: list[tuple[str, ...]] = []
            for name, comp_data in components.items():
                e = comp_data.get("elapsed", {})
                _rows_cp.append(
                    (
                        name,
                        f"{e.get('mean', 0) * 1_000_000:.1f}",
                        f"{e.get('median', 0) * 1_000_000:.1f}",
                        f"{e.get('p95', 0) * 1_000_000:.1f}",
                        f"{e.get('p99', 0) * 1_000_000:.1f}",
                        f"{e.get('cov_pct', 0):.1f}",
                    )
                )
            hdr, sep, data = align_table(_headers_cp, _rows_cp, alignments="<>")
            lines.append(hdr)
            lines.append(sep)
            lines.extend(data)
            lines.append("")

        regressions = report.get("regressions", [])
        if regressions:
            lines.append("Regression Detections:")
            for r in regressions:
                lines.append(f"  {r}")
            lines.append("")

        lines.append("=" * 90)
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    atomic_write(path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI Main Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for unified accuracy evaluation and profiling benchmark."""
    parser = argparse.ArgumentParser(
        description="Unified Accuracy and Performance Benchmark for Assist Canonicalizer"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=BENCHMARK_MODES,
        default=MODE_ACCURACY,
        help=f"Benchmark execution mode (default: {MODE_ACCURACY})",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help=(
            "Profiling target (evaluate, build_index, rank, components, all; "
            "default is evaluate for accuracy mode and rank for performance mode)"
        ),
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="tests/real_world",
        help="Directory containing dataset JSON files (default: tests/real_world)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Optional comma-separated or space-separated language codes to filter",
    )
    parser.add_argument(
        "--failure-limit",
        type=int,
        default=0,
        help="Maximum detailed failures to print per language in accuracy mode",
    )
    parser.add_argument(
        "--skip-hassil",
        action="store_true",
        default=False,
        help="Skip HassIL baseline evaluation (lexical mode only)",
    )
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        default=False,
        help="Skip component-only ablation analysis",
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
        "--output-txt",
        type=str,
        default=None,
        help="Optional plain text report output path",
    )

    # Accuracy mode thresholds
    parser.add_argument(
        "--min-intent-slot-accuracy",
        type=float,
        default=None,
        help="Fail when lexical intent/slot accuracy falls below this percentage",
    )
    parser.add_argument(
        "--max-fallback-rate",
        type=float,
        default=None,
        help="Fail when lexical fallback rate exceeds this percentage",
    )

    # Performance mode stats
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of iterations in performance mode (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Number of warmup runs in performance mode (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=GRANULARITY_LEVELS,
        default=DEFAULT_GRANULARITY,
        help=f"Granularity of profile timers (default: {DEFAULT_GRANULARITY})",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline profiling JSON file or directory for regression comparisons",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        default=False,
        help="Save profiling result as new historical baseline",
    )
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=DEFAULT_MAX_REGRESSION_PCT,
        help=(
            "Maximum regression percentage allowed before flagging "
            f"(default: {DEFAULT_MAX_REGRESSION_PCT})"
        ),
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        default=False,
        help="Trigger standard cProfile dump during rankings profiling",
    )

    args = parser.parse_args()
    _bootstrap_project_imports()

    # Validate mode
    if args.mode not in BENCHMARK_MODES:
        print(
            f"Error: Invalid mode {args.mode!r}. Must be one of {BENCHMARK_MODES}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse and validate target
    target = args.target
    if target is None:
        target = "evaluate" if args.mode == MODE_ACCURACY else "rank"

    # Validate target explicitly against the allow list
    if target not in PROFILING_TARGETS:
        print(f"Error: Target {target!r} is not a valid benchmark target.", file=sys.stderr)
        print(f"Allowed targets: {', '.join(sorted(PROFILING_TARGETS))}", file=sys.stderr)
        sys.exit(1)

    if args.mode == MODE_ACCURACY and target != "evaluate":
        print(
            f"Error: Target {target!r} is not supported for accuracy mode. "
            "Only 'evaluate' is supported.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        target = sanitize_chars(target, _TARGET_ALLOWED_CHARS)
    except ValueError as err:
        print(f"Error: Target contains invalid characters: {err}", file=sys.stderr)
        sys.exit(1)

    safe_target = target

    # Validate numeric/threshold inputs
    if args.failure_limit < 0:
        print("Error: --failure-limit must be zero or positive", file=sys.stderr)
        sys.exit(1)
    if args.min_intent_slot_accuracy is not None and not (
        0.0 <= args.min_intent_slot_accuracy <= 100.0
    ):
        print(
            "Error: --min-intent-slot-accuracy must be between 0.0 and 100.0",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.max_fallback_rate is not None and not (0.0 <= args.max_fallback_rate <= 100.0):
        print(
            "Error: --max-fallback-rate must be between 0.0 and 100.0",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.iterations < 1:
        print("Error: --iterations must be positive", file=sys.stderr)
        sys.exit(1)
    if args.warmup < 0:
        print("Error: --warmup must be non-negative", file=sys.stderr)
        sys.exit(1)
    if args.max_regression_pct < 0.0:
        print("Error: --max-regression-pct must be non-negative", file=sys.stderr)
        sys.exit(1)
    if args.granularity not in GRANULARITY_LEVELS:
        print(f"Error: Invalid granularity {args.granularity!r}", file=sys.stderr)
        sys.exit(1)

    # Sanitize datasets directory
    safe_datasets_dir = sanitize_path_required(_REPO_ROOT, "datasets_dir", args.datasets_dir)

    # Sanitize outputs
    safe_output_json = (
        sanitize_path_required(_REPO_ROOT, "output_json", args.output_json)
        if args.output_json
        else None
    )
    safe_output_md = (
        sanitize_path_required(_REPO_ROOT, "output_md", args.output_md) if args.output_md else None
    )
    safe_output_txt = (
        sanitize_path_required(_REPO_ROOT, "output_txt", args.output_txt)
        if args.output_txt
        else None
    )

    # Handle language filters (support both comma-separated and space-separated list configurations)
    language_filter: list[str] | None = None
    if args.languages:
        # Split by comma first, then handle spaces
        langs_split = []
        for term in args.languages.split(","):
            langs_split.extend(term.split())
        language_filter = [lang.strip() for lang in langs_split if lang.strip()]
        for lang in language_filter:
            if not lang.isalnum() and not all(c in "_-" for c in lang):
                print(f"Error: Invalid language code {lang!r}", file=sys.stderr)
                sys.exit(1)

    # Discover datasets
    datasets = discover_datasets(safe_datasets_dir, language_filter)
    if not datasets:
        print(f"Error: No datasets found in {safe_datasets_dir}", file=sys.stderr)
        sys.exit(1)

    if args.mode == MODE_ACCURACY:
        success = asyncio.run(
            run_evaluation(
                datasets=datasets,
                failure_limit=args.failure_limit,
                output_json=safe_output_json,
                output_md=safe_output_md,
                output_txt=safe_output_txt,
                min_intent_slot_accuracy=args.min_intent_slot_accuracy,
                max_fallback_rate=args.max_fallback_rate,
                datasets_dir=str(Path(safe_datasets_dir).relative_to(_REPO_ROOT)),
                skip_hassil=args.skip_hassil,
                skip_ablations=args.skip_ablations,
            )
        )
        sys.exit(0 if success else 1)

    else:
        # Performance Mode
        if args.iterations < 1:
            print("Error: --iterations must be positive", file=sys.stderr)
            sys.exit(1)
        if args.warmup < 0:
            print("Error: --warmup must be non-negative", file=sys.stderr)
            sys.exit(1)

        baseline_mgr = BaselineManager(_REPO_ROOT)
        _explicit_baseline = bool(args.baseline)
        if args.baseline:
            safe_baseline = sanitize_path_required(_REPO_ROOT, "baseline", args.baseline)
            baseline_path = Path(safe_baseline)
            baseline_mgr._baseline_dir = (
                baseline_path.parent if baseline_path.suffix == ".json" else baseline_path
            )

        print("PROFILE_START", flush=True)
        try:
            if safe_target == "all":
                all_reports = {}
                for tgt in ("build_index", "rank", "components", "evaluate"):
                    out_json = (
                        os.path.join(_REPO_ROOT, BENCHMARK_DIR, f"profile_{tgt}.json")
                        if safe_output_json is None
                        else safe_output_json
                    )
                    out_md = (
                        os.path.join(_REPO_ROOT, BENCHMARK_DIR, f"profile_{tgt}.md")
                        if safe_output_md is None
                        else safe_output_md
                    )
                    out_txt = (
                        os.path.join(_REPO_ROOT, BENCHMARK_DIR, f"profile_{tgt}.txt")
                        if safe_output_txt is None
                        else safe_output_txt
                    )

                    report = _run_profiling(
                        tgt,
                        datasets,
                        args.iterations,
                        args.warmup,
                        args.granularity,
                        baseline_mgr,
                        args.max_regression_pct,
                        out_json,
                        out_md,
                        out_txt,
                        args.save_baseline,
                        language_filter,
                        warn_on_missing=_explicit_baseline,
                    )
                    all_reports[tgt] = report

                output_dir = Path(_REPO_ROOT) / BENCHMARK_DIR
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save JSON
                agg_path = (
                    Path(safe_output_json)
                    if safe_output_json is not None
                    else output_dir / "profile_all.json"
                )
                agg_json = {"target": "all", "targets": all_reports}
                atomic_write(str(agg_path), json.dumps(agg_json, indent=2, default=str) + "\n")
                print(f"\nAggregate all-targets JSON report saved to {agg_path}")

                # Save MD
                agg_md_path = (
                    Path(safe_output_md)
                    if safe_output_md is not None
                    else output_dir / "profile_all.md"
                )
                _write_profile_all_markdown(all_reports, str(agg_md_path))
                print(f"Aggregate all-targets Markdown report saved to {agg_md_path}")

                # Save TXT
                agg_txt_path = (
                    Path(safe_output_txt)
                    if safe_output_txt is not None
                    else output_dir / "profile_all.txt"
                )
                _write_profile_all_text(all_reports, str(agg_txt_path))
                print(f"Aggregate all-targets Text report saved to {agg_txt_path}")
            else:
                _run_profiling(
                    safe_target,
                    datasets,
                    args.iterations,
                    args.warmup,
                    args.granularity,
                    baseline_mgr,
                    args.max_regression_pct,
                    safe_output_json,
                    safe_output_md,
                    safe_output_txt,
                    args.save_baseline,
                    language_filter,
                    warn_on_missing=_explicit_baseline,
                )

            # Optional cProfile dump during rankings profiling
            if args.cprofile:
                print("\nRunning cProfile snapshot ...", flush=True)
                import cProfile as cProf

                pr = cProf.Profile()
                pr.enable()

                _bootstrap_project_imports()
                for lang, path in sorted(datasets.items()):
                    with open(path, "rb") as f:
                        data = orjson.loads(f.read())
                    slots = _dataset_registry_slots(data, lang)
                    sources = load_language_intent_sources(lang)
                    candidates = build_candidates_from_intent_sources(lang, sources, slots)
                    index = build_index(lang, candidates)
                    raw_cases = data.get("test_cases", [])
                    if isinstance(raw_cases, list):
                        for case in raw_cases:
                            if isinstance(case, dict) and isinstance(case.get("query"), str):
                                index.rank(case["query"])

                pr.disable()
                profile_dir = Path(_REPO_ROOT) / BENCHMARK_DIR
                profile_dir.mkdir(parents=True, exist_ok=True)
                prof_dump = profile_dir / "profile_metrics.prof"
                pr.dump_stats(str(prof_dump))
                print(f"cProfile dump saved to {prof_dump}")

                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
                ps.print_stats(50)
                print("\nTOP 50 FUNCTIONS BY CUMULATIVE TIME:")
                print("-" * 80)
                print(s.getvalue())

        except KeyboardInterrupt:
            print("\nPROFILE_INTERRUPTED", flush=True)
            sys.exit(1)
        except Exception as exc:
            print(f"\nPROFILE_FAILED: {exc}", flush=True)
            raise

        print("PROFILE_OK", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
