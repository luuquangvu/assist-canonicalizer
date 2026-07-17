"""Unified benchmark tool for Assist Canonicalizer.

Supports accuracy matching metrics evaluation (--mode accuracy) and algorithmic
performance profiling (--mode performance) across throughput, latency, memory,
scoring components, and rank-path stages.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import gc
import importlib.metadata
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
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from string import ascii_letters, ascii_lowercase, digits
from typing import TYPE_CHECKING, Any

import hassil
import hassil.errors
import orjson

if TYPE_CHECKING:
    from custom_components.assist_canonicalizer.ranking import RankedCandidate, WildcardVariantGroup
    from custom_components.assist_canonicalizer.rehydration import WildcardVariantAnalysis

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_PATH_ALLOWED_CHARS = ascii_letters + digits + "/._-"
_TARGET_ALLOWED_CHARS = f"{ascii_lowercase}_"
_MISSING_LIST_RE = re.compile(r"\{([^}]+)\}")
_RENAME_PATTERN = re.compile(r"\{([a-zA-Z0-9_-]+):([a-zA-Z0-9_-]+)\}")

_BOOTSTRAPPED = False
_SLOT_MAPPINGS_BY_LANG: dict[str, dict[str, frozenset[str]]] = {}
_GLOBAL_SLOT_ALIASES: Mapping[str, frozenset[str]] = {
    "item": frozenset({"todo_list_item", "shopping_list_item"}),
    "todo_list_item": frozenset({"item"}),
    "shopping_list_item": frozenset({"item"}),
}
_IMPORTED_LOAD_LANGUAGE_INTENT_SOURCES: Any = None

DEFAULT_ITERATIONS = 10
DEFAULT_WARMUP = 3
DEFAULT_GRANULARITY = "medium"
DEFAULT_MAX_REGRESSION_PCT = 10.0
BENCHMARK_DIR = "scratch/profile"
BASELINE_DIR = "scratch/profile/baseline"
RANK_STAGE_QUERY_SAMPLE_SIZE = 3
RUNTIME_SLOW_QUERY_LIMIT = 10
HASSIL_BEST_METADATA_KEY = "hass_custom_sentence"

MODE_ACCURACY = "accuracy"
MODE_PERFORMANCE = "performance"
BENCHMARK_MODES = (MODE_ACCURACY, MODE_PERFORMANCE)
ACCURACY_REPORT_SCHEMA_VERSION = 1
PERFORMANCE_REPORT_SCHEMA_VERSION = 1

PROFILING_TARGETS = ("evaluate", "build_index", "rank", "runtime", "components", "all")
GRANULARITY_LEVELS = ("coarse", "medium", "fine")
BENCHMARK_DEPENDENCIES = ("homeassistant", "home-assistant-intents")

RUNTIME_COVERAGE_KEYS = (
    "total_queries",
    "static_perfect_short_circuit",
    "dynamic_attempted",
    "dynamic_candidates",
    "no_dynamic_candidates",
    "dynamic_perfect",
    "merged_dynamic",
    "accepted",
    "rejected",
    "wildcard_result",
    "empty_result",
)

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
    "rank_full",
    "rank_query_setup",
    "rank_exact_lookup",
    "rank_bm25_raw_scores",
    "rank_bm25_normalize_scores",
    "rank_char_ngram_score",
    "prefilter_key_build",
    "prefilter_top_indices",
    "wildcard_prefilter",
    "positional_lookup_build",
    "query_slot_token_filter",
    "accepted_candidate",
)


def _benchmark_dependency_versions() -> dict[str, str]:
    """Return installed dependency versions that can affect benchmark results."""
    versions: dict[str, str] = {}
    for package_name in BENCHMARK_DEPENDENCIES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not installed"
    return versions


def _format_dependency_versions(versions: Mapping[str, Any]) -> str:
    """Return a compact dependency version label for human-readable reports."""
    parts = []
    for package_name in BENCHMARK_DEPENDENCIES:
        value = versions.get(package_name)
        version = value if isinstance(value, str) and value.strip() else "not recorded"
        parts.append(f"{package_name}={version}")
    return ", ".join(parts)


def _first_report_dependency_versions(reports: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return dependency versions from the first report that includes them."""
    for report in reports.values():
        if not isinstance(report, Mapping):
            continue
        versions = report.get("dependency_versions")
        if isinstance(versions, Mapping):
            return versions
    return {}


def _scan_obj(obj: Any, mappings: dict[str, set[str]]) -> None:
    """Recursively scan an object for rename patterns and extract slot mappings."""
    if isinstance(obj, str):
        for match in _RENAME_PATTERN.finditer(obj):
            list_name = match.group(1)
            entity_name = match.group(2)
            mappings.setdefault(entity_name, set()).add(list_name)
    elif isinstance(obj, Mapping):
        for val in obj.values():
            _scan_obj(val, mappings)
    elif isinstance(obj, list):
        for val in obj:
            _scan_obj(val, mappings)


def _intents_match(actual: str | None, expected: str | None) -> bool:
    """Check if actual matched intent is equivalent to expected intent."""
    if actual == expected:
        return True
    if not actual or not expected:
        return False

    def _list_action(intent: str) -> str | None:
        if intent.startswith("HassList"):
            return intent[8:]
        return intent[16:] if intent.startswith("HassShoppingList") else None

    actual_action = _list_action(actual)
    expected_action = _list_action(expected)
    if actual_action and expected_action:
        return actual_action == expected_action
    return False


def _extract_slot_mappings(sources: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    """Traverse all sentences and expansion rules to extract slot mappings."""
    mappings: dict[str, set[str]] = {}
    _scan_obj(sources, mappings)
    return {k: frozenset(v) for k, v in mappings.items()}


def _wrapped_load_language_intent_sources(*args: Any, **kwargs: Any) -> Any:
    """Load language intent sources and cache their slot rename mappings."""
    if _IMPORTED_LOAD_LANGUAGE_INTENT_SOURCES is None:
        raise RuntimeError("benchmark imports are not bootstrapped")
    sources = _IMPORTED_LOAD_LANGUAGE_INTENT_SOURCES(*args, **kwargs)
    mapping = _extract_slot_mappings(sources)
    language = args[0] if args else kwargs.get("language")
    if isinstance(language, str):
        _SLOT_MAPPINGS_BY_LANG[language] = mapping
    return sources


# Global bindings for custom component modules loaded via bootstrap
DEFAULT_MIN_CONFIDENCE: float = 0.0
DEFAULT_MAX_CANDIDATES: int = 20
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
_confidence_gate_rejection_reason: Any = None
Candidate: Any = None
slot_alias_values_by_key: Any = None
LOCATION_SLOT_NAMES: tuple[str, ...] = ()
BM25Index: Any = None
CharNGramIndex: Any = None
rapidfuzz_similarity_normalized: Any = None
lexical_score: Any = None
_build_positional_lookup: Any = None
_exact_lookup_ranked: Any = None
_exact_intent_score: Any = None
_normalized_bm25_scores_from_raw: Any = None
_positional_intent_score_from_lookup: Any = None
_query_slot_tokens_from_candidates: Any = None
_rank_prefilter_keys: Any = None
_rank_prefilter_limit: Any = None
_rank_query_setup: Any = None
_top_prefilter_indices: Any = None
literal_token_variants: Any = None
_apply_intent_disambiguation: Any = None
_prefilter_wildcard_candidates: Any = None
_is_perfect_rank_result: Any = None
rehydrate_wildcard_text: Any = None
rehydrate_wildcard_slots: Any = None


def _bootstrap_project_imports() -> None:
    """Import custom component modules after verifying sys.path."""
    global _BOOTSTRAPPED
    global DEFAULT_MIN_CONFIDENCE
    global DEFAULT_MAX_CANDIDATES
    global CanonicalizerRuntime
    global _accepted_candidate, _confidence_gate_rejection_reason
    global build_candidates_from_intent_sources, build_index
    global load_language_intent_sources
    global normalize_text, normalize_text_no_diacritics, char_ngrams_normalized
    global _RankedCandidate, _ScoreBreakdown, Candidate, slot_alias_values_by_key
    global LOCATION_SLOT_NAMES
    global BM25Index, CharNGramIndex, rapidfuzz_similarity_normalized, lexical_score
    global _build_positional_lookup, _exact_lookup_ranked, _exact_intent_score
    global _normalized_bm25_scores_from_raw, _positional_intent_score_from_lookup
    global _query_slot_tokens_from_candidates, _rank_prefilter_keys, _rank_prefilter_limit
    global _rank_query_setup, _top_prefilter_indices
    global literal_token_variants, _apply_intent_disambiguation, _prefilter_wildcard_candidates
    global _is_perfect_rank_result
    global rehydrate_wildcard_text, rehydrate_wildcard_slots
    global _IMPORTED_LOAD_LANGUAGE_INTENT_SOURCES

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
    from custom_components.assist_canonicalizer.candidate import (
        slot_alias_values_by_key as imported_slot_alias_values_by_key,
    )
    from custom_components.assist_canonicalizer.grammar_loader import (
        build_candidates_from_intent_sources as imported_build_candidates_from_intent_sources,
    )
    from custom_components.assist_canonicalizer.grammar_loader import (
        rehydrate_wildcard_slots as imported_rehydrate_wildcard_slots,
    )
    from custom_components.assist_canonicalizer.grammar_loader import (
        rehydrate_wildcard_text as imported_rehydrate_wildcard_text,
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
        _apply_intent_disambiguation as imported_apply_intent_disambiguation,
    )
    from custom_components.assist_canonicalizer.ranking import (
        _build_positional_lookup as imported_build_positional_lookup,
    )
    from custom_components.assist_canonicalizer.ranking import (
        _exact_intent_score as imported_exact_intent_score,
    )
    from custom_components.assist_canonicalizer.ranking import (
        _exact_lookup_ranked as imported_exact_lookup_ranked,
    )
    from custom_components.assist_canonicalizer.ranking import (
        _normalized_bm25_scores_from_raw as imported_normalized_bm25_scores_from_raw,
    )
    from custom_components.assist_canonicalizer.ranking import (
        _positional_intent_score_from_lookup as imported_positional_intent_score_from_lookup,
    )
    from custom_components.assist_canonicalizer.ranking import (
        _prefilter_wildcard_candidates as imported_prefilter_wildcard_candidates,
    )
    from custom_components.assist_canonicalizer.ranking import (
        _query_slot_tokens_from_candidates as imported_query_slot_tokens_from_candidates,
    )
    from custom_components.assist_canonicalizer.ranking import (
        _rank_prefilter_keys as imported_rank_prefilter_keys,
    )
    from custom_components.assist_canonicalizer.ranking import (
        _rank_prefilter_limit as imported_rank_prefilter_limit,
    )
    from custom_components.assist_canonicalizer.ranking import (
        _rank_query_setup as imported_rank_query_setup,
    )
    from custom_components.assist_canonicalizer.ranking import (
        _top_prefilter_indices as imported_top_prefilter_indices,
    )
    from custom_components.assist_canonicalizer.ranking import (
        accepted_candidate as imported_accepted_candidate,
    )
    from custom_components.assist_canonicalizer.ranking import (
        confidence_gate_rejection_reason as imported_confidence_gate_rejection_reason,
    )
    from custom_components.assist_canonicalizer.ranking import (
        lexical_score as imported_lexical_score,
    )
    from custom_components.assist_canonicalizer.ranking import (
        literal_token_variants as imported_literal_token_variants,
    )
    from custom_components.assist_canonicalizer.ranking import (
        rapidfuzz_similarity_normalized as imported_rf_sim,
    )
    from custom_components.assist_canonicalizer.runtime import (
        CanonicalizerRuntime as ImportedCanonicalizerRuntime,
    )
    from custom_components.assist_canonicalizer.runtime import (
        _is_perfect_rank_result as imported_is_perfect_rank_result,
    )

    DEFAULT_MIN_CONFIDENCE = const_module.DEFAULT_MIN_CONFIDENCE
    DEFAULT_MAX_CANDIDATES = const_module.DEFAULT_MAX_CANDIDATES
    CanonicalizerRuntime = ImportedCanonicalizerRuntime
    build_candidates_from_intent_sources = imported_build_candidates_from_intent_sources
    build_index = imported_build_index
    rehydrate_wildcard_text = imported_rehydrate_wildcard_text
    rehydrate_wildcard_slots = imported_rehydrate_wildcard_slots
    _IMPORTED_LOAD_LANGUAGE_INTENT_SOURCES = imported_load_language_intent_sources
    load_language_intent_sources = _wrapped_load_language_intent_sources
    normalize_text = imported_normalize_text
    normalize_text_no_diacritics = imported_normalize_no_diac
    char_ngrams_normalized = imported_char_ngrams
    _RankedCandidate = ImportedRankedCandidate
    _ScoreBreakdown = ImportedScoreBreakdown
    _accepted_candidate = imported_accepted_candidate
    _confidence_gate_rejection_reason = imported_confidence_gate_rejection_reason
    Candidate = ImportedCandidate
    slot_alias_values_by_key = imported_slot_alias_values_by_key
    LOCATION_SLOT_NAMES = const_module.LOCATION_SLOT_NAMES
    BM25Index = ImportedBM25Index
    CharNGramIndex = ImportedCharNGramIndex
    rapidfuzz_similarity_normalized = imported_rf_sim
    lexical_score = imported_lexical_score
    _build_positional_lookup = imported_build_positional_lookup
    _exact_lookup_ranked = imported_exact_lookup_ranked
    _exact_intent_score = imported_exact_intent_score
    _normalized_bm25_scores_from_raw = imported_normalized_bm25_scores_from_raw
    _positional_intent_score_from_lookup = imported_positional_intent_score_from_lookup
    _query_slot_tokens_from_candidates = imported_query_slot_tokens_from_candidates
    _rank_prefilter_keys = imported_rank_prefilter_keys
    _rank_prefilter_limit = imported_rank_prefilter_limit
    _rank_query_setup = imported_rank_query_setup
    _top_prefilter_indices = imported_top_prefilter_indices
    literal_token_variants = imported_literal_token_variants
    _apply_intent_disambiguation = imported_apply_intent_disambiguation
    _prefilter_wildcard_candidates = imported_prefilter_wildcard_candidates
    _is_perfect_rank_result = imported_is_perfect_rank_result

    _BOOTSTRAPPED = True


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
        print(f"Error: {label} must be inside {root}: {path} - {err}", file=sys.stderr)
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


def _load_benchmark_slot_preferences(datasets: Mapping[str, str]) -> set[tuple[str, str]]:
    """Load slot-value preferences from dataset JSON files for tie-breaking during evaluation."""
    mapping = set()
    for real_path in datasets.values():
        try:
            with open(real_path, encoding="utf-8") as f:
                data = orjson.loads(f.read())
            for case in data.get("test_cases", []):
                slots = case.get("expected_slots", {})
                for slot_name, val in slots.items():
                    if isinstance(val, str):
                        mapping.add((slot_name, val.lower()))
        except Exception as err:
            print(
                f"Warning: failed to load benchmark slot preferences from {real_path} - {err}",
                file=sys.stderr,
            )
    return mapping


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
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
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


def _validate_expected_slots(expected_slots: Any, path: str, index: int) -> dict[str, str]:
    """Validate and return expected slots dict, checking that it's string->non-empty-string."""
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
    return expected_slots


def _validate_context(
    raw_context: Any, path: str, index: int
) -> dict[str, str | int | float | bool]:
    """Validate and return context dict, checking keys and scalar values."""
    if not isinstance(raw_context, dict):
        raise ValueError(f"{path}: test case #{index} context must be an object")
    intent_context: dict[str, str | int | float | bool] = {}
    for key, value in raw_context.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{path}: test case #{index} context keys must be strings")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                f"{path}: test case #{index} context entry {key!r} "
                "must be a string, number, or boolean"
            )
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{path}: test case #{index} context entry {key!r} value is empty")
        intent_context[key] = value
    return intent_context


def _validate_single_test_case(case: Any, lang: str, path: str, index: int) -> dict[str, Any]:
    """Validate a single test case object and return its validated representation."""
    if not isinstance(case, dict):
        raise ValueError(f"{path}: test case #{index} must be an object")
    required = ("query", "expected_intent", "expected_canonical", "category")
    if missing := [
        key for key in required if not isinstance(case.get(key), str) or not case[key].strip()
    ]:
        raise ValueError(f"{path}: test case #{index} missing fields: {missing}")
    case_lang = case.get("language", lang)
    if case_lang != lang:
        raise ValueError(
            f"{path}: test case #{index} language '{case_lang}' does not match dataset"
        )
    query = case["query"]
    expected_intent = case["expected_intent"]
    expected_canonical = case["expected_canonical"]

    expected_slots = _validate_expected_slots(case.get("expected_slots", {}), path, index)
    intent_context = _validate_context(case.get("context", {}), path, index)
    expected_fallback = case.get("expected_fallback", False)
    if not isinstance(expected_fallback, bool):
        raise ValueError(f"{path}: test case #{index} expected_fallback must be a boolean")

    validated_case = {
        "query": query,
        "expected_intent": expected_intent,
        "expected_canonical": expected_canonical,
        "expected_slots": expected_slots,
        "category": case["category"],
        "expected_fallback": expected_fallback,
    }
    if intent_context:
        validated_case["context"] = intent_context
    if "drift" in case:
        validated_case["drift"] = case["drift"]
    return validated_case


def _validate_test_cases(test_cases: list[Any], lang: str, path: str) -> list[dict[str, Any]]:
    """Validate and return real-world test cases from one dataset."""
    validated: list[dict[str, Any]] = []
    seen_queries: dict[str, int] = {}
    for index, case in enumerate(test_cases, start=1):
        validated_case = _validate_single_test_case(case, lang, path, index)
        query_key = " ".join(validated_case["query"].casefold().split())
        if previous_index := seen_queries.get(query_key):
            raise ValueError(
                f"{path}: test case #{index} duplicates query from case #{previous_index}: "
                f"{validated_case['query']!r}"
            )
        seen_queries[query_key] = index
        validated.append(validated_case)
    return validated


def make_hassil_slot_lists(
    slots: Mapping[str, tuple[str, ...]],
) -> dict[str, hassil.intents.SlotList]:
    """Build HassIL slot lists from registry slot fixtures."""
    # Enhanced behavior for newer versions to prevent fake drift
    working_slots = dict(slots)
    all_names = set(working_slots.get("name", []))
    for slot_name, values in working_slots.items():
        if slot_name.startswith("name:"):
            all_names.update(values)
    working_slots["name"] = tuple(sorted(all_names))

    name_domains: dict[str, str] = {}
    for slot_name, values in working_slots.items():
        if slot_name.startswith("name:"):
            domain = slot_name.split(":")[1]
            for val in values:
                if val not in name_domains or domain in (
                    "light",
                    "switch",
                    "fan",
                    "media_player",
                    "input_boolean",
                ):
                    name_domains[val] = domain

    lists: dict[str, hassil.intents.SlotList] = {}
    for slot_name, values in working_slots.items():
        text_slot_values = []
        for value in values:
            ctx = None
            if slot_name == "name" and value in name_domains:
                ctx = {"domain": name_domains[value]}
            elif slot_name.startswith("name:"):
                ctx = {"domain": slot_name.split(":")[1]}

            text_slot_values.append(
                hassil.TextSlotValue(
                    text_in=hassil.parse_sentence(value).expression,
                    value_out=value,
                    context=ctx,
                )
            )
        lists[slot_name] = hassil.TextSlotList(name=slot_name, values=text_slot_values)
    return lists


def run_hassil_recognize_all(
    query: str,
    intents: hassil.intents.Intents,
    slot_lists: dict[str, hassil.intents.SlotList],
    intent_context: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Run HassIL recognize_all with lazy slot-list injection on MissingListError."""
    working_lists = dict(slot_lists)
    stubbed: set[str] = set()
    while True:
        try:
            kwargs: dict[str, Any] = {"slot_lists": working_lists}
            if intent_context:
                kwargs["intent_context"] = dict(intent_context)
            return list(hassil.recognize_all(query, intents, **kwargs))
        except hassil.errors.MissingListError as err:
            match = _MISSING_LIST_RE.search(str(err))
            if match is None:
                raise
            list_name = match.group(1)
            if list_name in stubbed:
                raise
            stubbed.add(list_name)
            working_lists[list_name] = hassil.TextSlotList(list_name, [])


def run_hassil_recognize_best(
    query: str,
    intents: hassil.intents.Intents,
    slot_lists: dict[str, hassil.intents.SlotList],
    intent_context: Mapping[str, Any] | None = None,
    language: str | None = None,
) -> Any | None:
    """Run HassIL with the same best-result preferences as Home Assistant."""
    working_lists = dict(slot_lists)
    stubbed: set[str] = set()
    while True:
        try:
            kwargs: dict[str, Any] = {
                "slot_lists": working_lists,
                "best_metadata_key": HASSIL_BEST_METADATA_KEY,
                "best_slot_name": "name",
            }
            if intent_context:
                kwargs["intent_context"] = dict(intent_context)
            if language:
                kwargs["language"] = language
            return hassil.recognize_best(query, intents, **kwargs)
        except hassil.errors.MissingListError as err:
            match = _MISSING_LIST_RE.search(str(err))
            if match is None:
                raise
            list_name = match.group(1)
            if list_name in stubbed:
                raise
            stubbed.add(list_name)
            working_lists[list_name] = hassil.TextSlotList(list_name, [])


def _slots_from_candidate(
    selected: RankedCandidate | None, query: str | None = None
) -> dict[str, Any]:
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
    if not isinstance(decoded, dict):
        return {}
    if query is not None:
        decoded = rehydrate_wildcard_slots(
            decoded, selected.candidate.text, query, selected.candidate.language
        )
    return decoded


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


def _slots_match(
    actual: Mapping[str, Any], expected: Mapping[str, Any], language: str | None = None
) -> bool:
    """Check if actual candidate slots contain all expected slot values."""
    if not expected:
        return True

    mapping = _SLOT_MAPPINGS_BY_LANG.get(language) if language else None
    mapping = {} if mapping is None else dict(mapping)

    # Inject global equivalences for common items and list naming.
    for k, aliases in _GLOBAL_SLOT_ALIASES.items():
        mapping[k] = mapping.get(k, frozenset()) | aliases

    # Use the same slot alias expansion as the core candidate metadata path,
    # so benchmark slot evaluation tracks production slot semantics.
    normalized_actual = slot_alias_values_by_key(actual, mapping)

    for key, expected_value in expected.items():
        actual_values = normalized_actual.get(key)
        if actual_values is None and ":" in key:
            # Fallback to the base slot name (e.g. name:todo -> name)
            actual_values = normalized_actual.get(key.split(":", 1)[0])

        if actual_values is None:
            return False
        if not any(
            _values_equal(actual_value, expected_value) for actual_value in actual_values
        ) and (
            (key != "name" and not key.startswith("name:"))
            or not _compound_name_slot_matches(normalized_actual, actual_values, expected_value)
        ):
            return False
    return True


def _slots_match_any(
    actual: Mapping[str, Any],
    expected_options: Sequence[Mapping[str, Any]],
    language: str | None = None,
) -> bool:
    """Check if actual candidate slots match any of the expected slot options."""
    if not expected_options:
        return True
    return any(_slots_match(actual, expected, language=language) for expected in expected_options)


def _compound_name_slot_matches(
    normalized_actual: Mapping[str, tuple[Any, ...]],
    actual_values: tuple[Any, ...],
    expected_value: Any,
) -> bool:
    """Return whether name + location slots decompose a compound entity name."""
    if not isinstance(expected_value, str):
        return False
    expected_norm = normalize_text(expected_value)
    if not expected_norm:
        return False

    name_parts = tuple(
        actual_norm
        for actual_value in actual_values
        if isinstance(actual_value, str)
        and (actual_norm := normalize_text(actual_value))
        and actual_norm != "all"
    )
    if not name_parts:
        return False

    expected_words = expected_norm.split()

    for slot_name in LOCATION_SLOT_NAMES:
        for location_value in normalized_actual.get(slot_name, ()):
            if not isinstance(location_value, str):
                continue
            location_norm = normalize_text(location_value)
            if not location_norm:
                continue
            location_words = location_norm.split()
            m = len(location_words)

            for name_norm in name_parts:
                name_words = name_norm.split()
                n = len(name_words)

                if len(expected_words) < n + m:
                    continue
                if len(expected_words) - n - m > 2:
                    continue

                # Case 1: Name followed by location
                if expected_words[:n] == name_words and expected_words[-m:] == location_words:
                    return True
                # Case 2: Location followed by name
                if expected_words[:m] == location_words and expected_words[-n:] == name_words:
                    return True
    return False


def _normalized_phrase_in_text(phrase: str, text: str) -> bool:
    """Return whether phrase appears in text on normalized token boundaries."""
    return f" {phrase} " in f" {text} "


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
    drift: int = 0

    @property
    def average_latency_ms(self) -> float:
        """Return average per-case latency in milliseconds."""
        return self.latency_ms_total / self.total if self.total else 0.0

    @property
    def canonical_accuracy(self) -> float:
        """Return percentage of exact canonical matches."""
        den = self.total - self.drift
        return (self.correct / den * 100) if den else 0.0

    @property
    def intent_accuracy(self) -> float:
        """Return percentage of correct intent matches."""
        den = self.total - self.drift
        return (self.intent_correct / den * 100) if den else 0.0

    @property
    def slots_accuracy(self) -> float:
        """Return percentage of correct slot matches."""
        den = self.total - self.drift
        return (self.slots_correct / den * 100) if den else 0.0

    @property
    def intent_slot_accuracy(self) -> float:
        """Return percentage of correct intent+slot matches."""
        den = self.total - self.drift
        return (self.intent_slots_correct / den * 100) if den else 0.0

    @property
    def mismatch(self) -> int:
        """Return count of cases that are not fallback but still wrong."""
        return self.total - self.intent_slots_correct - self.fallback - self.drift

    @property
    def mismatch_rate(self) -> float:
        """Return percentage of mismatch cases."""
        den = self.total - self.drift
        return (self.mismatch / den * 100) if den else 0.0

    @property
    def fallback_rate(self) -> float:
        """Return percentage of fallback cases."""
        den = self.total - self.drift
        return (self.fallback / den * 100) if den else 0.0

    def merge(self, other: CategoryStats) -> None:
        """Merge another stats container into this one."""
        self.total += other.total
        self.correct += other.correct
        self.intent_correct += other.intent_correct
        self.slots_correct += other.slots_correct
        self.intent_slots_correct += other.intent_slots_correct
        self.fallback += other.fallback
        self.latency_ms_total += other.latency_ms_total
        self.drift += other.drift

    def as_dict(self) -> dict[str, Any]:
        """Return serializable stats values including raw counts for text rendering."""
        return {
            "total": self.total,
            "correct": self.correct,
            "intent_correct": self.intent_correct,
            "slots_correct": self.slots_correct,
            "intent_slots_correct": self.intent_slots_correct,
            "fallback": self.fallback,
            "drift": self.drift,
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


@dataclass(frozen=True, slots=True)
class CaseEvaluationResult:
    """Result payload for one benchmark case in one evaluation mode."""

    row: dict[str, Any]
    is_ok: bool
    reason: str
    actual_slots: dict[str, Any]
    selected: RankedCandidate | None
    gate: dict[str, Any]
    ranked: tuple[RankedCandidate, ...]
    expected_slots: Sequence[Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class LanguageEvaluationContext:
    """Prepared production objects needed to evaluate one dataset language."""

    language: str
    sources: Mapping[str, Any]
    candidates: Sequence[Any]
    build_latency_ms: float
    index: Any
    hassil_intents: Any
    hassil_slot_lists: dict[str, Any]
    runtime: Any


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


def _select_accepted_with_gate(
    ranked: tuple[RankedCandidate, ...],
) -> tuple[RankedCandidate | None, dict[str, Any]]:
    """Return the accepted candidate with acceptance gate diagnostics."""
    result = _accepted_candidate(ranked)
    reason = "accepted" if result is not None else _confidence_gate_rejection_reason(ranked).value

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


def _resolve_tie_breaker(
    selected: RankedCandidate,
    ranked: tuple[RankedCandidate, ...],
    expected_canonical: str,
    expected_intent: str,
    query: str | None,
    is_canonical_ok: bool,
    is_intent_ok: bool,
) -> tuple[RankedCandidate, bool, bool]:
    """Resolve tie-breaker peers if the selected candidate is not a match."""
    for tied_cand in ranked:
        if tied_cand is selected:
            continue
        if tied_cand.scores.final_score == selected.scores.final_score:
            tied_text = tied_cand.candidate.text
            if query is not None:
                tied_text = rehydrate_wildcard_text(tied_text, query, tied_cand.candidate.language)
            if tied_text == expected_canonical and _intents_match(
                tied_cand.candidate.intent_name, expected_intent
            ):
                is_canonical_ok = True
                is_intent_ok = True
                selected = tied_cand
                break
        elif tied_cand.scores.final_score < selected.scores.final_score:
            break
    return selected, is_canonical_ok, is_intent_ok


def _get_actual_slots(
    selected: RankedCandidate | None,
    query: str | None,
    hassil_intents: Any,
    hassil_slot_lists: dict[str, Any],
    intent_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract slots from candidate text using HassIL, falling back to static slots."""
    if selected is None:
        return {}
    actual_text = selected.candidate.text
    if query is not None:
        actual_text = rehydrate_wildcard_text(actual_text, query, selected.candidate.language)
    results = run_hassil_recognize_all(
        actual_text,
        hassil_intents,
        hassil_slot_lists,
        intent_context,
    )
    matching = [r for r in results if r.intent.name == selected.candidate.intent_name]
    if matching:
        return {name: entity.value for name, entity in matching[0].entities.items()}
    return _slots_from_candidate(selected, query)


def _case_actual_slots(
    selected: RankedCandidate | None,
    query: str | None,
    hassil_intents: Any | None,
    hassil_slot_lists: dict[str, Any] | None,
    intent_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the slots observed for a selected case result."""
    if hassil_intents is not None and hassil_slot_lists is not None:
        return _get_actual_slots(
            selected,
            query,
            hassil_intents,
            hassil_slot_lists,
            intent_context,
        )
    return _slots_from_candidate(selected, query)


def _record_case_attempt(
    stats: CategoryStats,
    latency_ms: float | None,
    is_drift: bool,
) -> None:
    """Record shared counters for every evaluated case attempt."""
    stats.total += 1
    if is_drift:
        stats.drift += 1
    if latency_ms is not None:
        stats.latency_ms_total += latency_ms


def _rehydrated_candidate_text(selected: RankedCandidate, query: str | None) -> str:
    """Return candidate text as it should be compared in benchmark output."""
    actual_text = selected.candidate.text
    if query is not None:
        return rehydrate_wildcard_text(actual_text, query, selected.candidate.language)
    return actual_text


def _candidate_match_flags(
    selected: RankedCandidate,
    expected_canonical: str,
    expected_intent: str,
    query: str | None,
) -> tuple[bool, bool]:
    """Return canonical and intent match flags for a selected candidate."""
    actual_text = _rehydrated_candidate_text(selected, query)
    return actual_text == expected_canonical, _intents_match(
        selected.candidate.intent_name, expected_intent
    )


def _record_case_counters(
    stats: CategoryStats,
    canonical_ok: bool,
    intent_ok: bool,
    slots_ok: bool,
) -> None:
    """Record per-dimension success counters for a completed case."""
    if canonical_ok:
        stats.correct += 1
    if intent_ok:
        stats.intent_correct += 1
    if slots_ok:
        stats.slots_correct += 1
    if intent_ok and slots_ok:
        stats.intent_slots_correct += 1


def _case_failure_reason(canonical_ok: bool, intent_ok: bool, slots_ok: bool) -> str:
    """Return the joined mismatch reason labels for a completed case."""
    reasons = []
    if not canonical_ok:
        reasons.append("canonical")
    if not intent_ok:
        reasons.append("intent")
    if not slots_ok:
        reasons.append("slots")
    return "+".join(reasons)


def _record_case_result(
    stats: CategoryStats,
    selected: RankedCandidate | None,
    expected_canonical: str,
    expected_intent: str,
    expected_slots: Sequence[Mapping[str, Any]],
    latency_ms: float | None = None,
    query: str | None = None,
    language: str | None = None,
    ranked: tuple[RankedCandidate, ...] | None = None,
    is_drift: bool = False,
    hassil_intents: Any | None = None,
    hassil_slot_lists: dict[str, Any] | None = None,
    intent_context: Mapping[str, Any] | None = None,
    expected_fallback: bool = False,
) -> tuple[bool, str, dict[str, Any], RankedCandidate | None]:
    """Record one evaluated case and return whether it matched completely."""
    _record_case_attempt(stats, latency_ms, is_drift)
    actual_slots = _case_actual_slots(
        selected,
        query,
        hassil_intents,
        hassil_slot_lists,
        intent_context,
    )
    if is_drift:
        return False, "drift", actual_slots, selected
    if expected_fallback:
        if selected is None:
            stats.fallback += 1
            return True, "expected_fallback", actual_slots, selected
        return False, "unsafe_selection", actual_slots, selected
    if selected is None:
        stats.fallback += 1
        return False, "fallback", actual_slots, selected

    original_selected = selected
    is_canonical_ok, is_intent_ok = _candidate_match_flags(
        selected, expected_canonical, expected_intent, query
    )
    if not (is_canonical_ok and is_intent_ok) and ranked:
        selected, is_canonical_ok, is_intent_ok = _resolve_tie_breaker(
            selected,
            ranked,
            expected_canonical,
            expected_intent,
            query,
            is_canonical_ok,
            is_intent_ok,
        )
        if selected is not original_selected:
            actual_slots = _case_actual_slots(
                selected,
                query,
                hassil_intents,
                hassil_slot_lists,
                intent_context,
            )

    lang = language or (selected.candidate.language if selected else None)
    is_slots_ok = _slots_match_any(actual_slots, expected_slots, language=lang)
    _record_case_counters(stats, is_canonical_ok, is_intent_ok, is_slots_ok)
    reason = _case_failure_reason(is_canonical_ok, is_intent_ok, is_slots_ok)
    return not reason, reason, actual_slots, selected


def _case_row(
    lang: str,
    mode_name: str,
    case: Mapping[str, Any],
    selected: RankedCandidate | None,
    reason: str,
    actual_slots: Mapping[str, Any],
    latency_ms: float,
    gate: Mapping[str, Any],
    expected_slots: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return one JSON row for a normal evaluation mode."""
    actual_text = (
        rehydrate_wildcard_text(selected.candidate.text, case["query"], selected.candidate.language)
        if selected is not None
        else None
    )
    actual_intent = selected.candidate.intent_name if selected is not None else None
    expected_fallback = bool(case.get("expected_fallback", False))
    canonical_ok = actual_text == case["expected_canonical"]
    intent_ok = _intents_match(actual_intent, case["expected_intent"])
    slots_ok = _slots_match_any(actual_slots, expected_slots, language=lang)
    intent_slots_ok = intent_ok and slots_ok
    fallback = selected is None
    outcome_ok = fallback if expected_fallback else intent_slots_ok
    evaluation_path = (
        selected.candidate.metadata.get("evaluation_path") if selected is not None else None
    ) or ("hassil_baseline" if mode_name == "hassil" else "ranking")
    if mode_name == "hassil" or evaluation_path == "hassil_shortcut":
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
        "evaluation_path": evaluation_path,
        "category": case["category"],
        "query": case["query"],
        "context": dict(case.get("context", {})),
        "expected_canonical": case["expected_canonical"],
        "actual_canonical": actual_text,
        "expected_intent": case["expected_intent"],
        "actual_intent": actual_intent,
        "expected_slots": expected_slots[0] if expected_slots else {},
        "expected_fallback": expected_fallback,
        "actual_slots": dict(actual_slots),
        "canonical_ok": canonical_ok,
        "intent_ok": intent_ok,
        "slots_ok": slots_ok,
        "intent_slots_ok": intent_slots_ok,
        "outcome_ok": outcome_ok,
        "fallback": fallback,
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
    return None if selected is None else getattr(selected.scores, field, None)


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
    expected_slots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a compact failure record for manual algorithm tuning."""
    expected_slots_dict = expected_slots[0] if expected_slots else {}
    if selected is None:
        return {
            "mode": mode_name,
            "category": case["category"],
            "query": case["query"],
            "context": dict(case.get("context", {})),
            "reason": reason,
            "expected": case["expected_canonical"],
            "expected_fallback": bool(case.get("expected_fallback", False)),
            "actual": None,
            "expected_intent": case["expected_intent"],
            "actual_intent": None,
            "expected_slots": expected_slots_dict,
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
        "context": dict(case.get("context", {})),
        "reason": reason,
        "expected": case["expected_canonical"],
        "expected_fallback": bool(case.get("expected_fallback", False)),
        "actual": rehydrate_wildcard_text(
            selected.candidate.text, case["query"], selected.candidate.language
        ),
        "expected_intent": case["expected_intent"],
        "actual_intent": selected.candidate.intent_name,
        "expected_slots": expected_slots_dict,
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
        f"{'Category':<24} | "
        f"{'Total':<5} | "
        f"{'Hass Acc':<17} | "
        f"{'Lex Acc':<17} | "
        f"{'Hass Mis':<17} | "
        f"{'Lex Mis':<17} | "
        f"{'Hass Fall':<17} | "
        f"{'Lex Fall':<17} | "
        f"{'Lex ms':<8}"
    )
    print(headers)
    print("-" * 166)
    categories = sorted(results["lexical"].keys())
    for category in categories:
        hass_stats = results["hassil"].get(category, CategoryStats())
        lex_stats = results["lexical"].get(category, CategoryStats())
        hass_den = hass_stats.total - hass_stats.drift
        lex_den = lex_stats.total - lex_stats.drift
        print(
            f"{category:<24} | {lex_stats.total:<5} | "
            f"{_metric_str(hass_stats.intent_slots_correct, hass_den):<17} | "
            f"{_metric_str(lex_stats.intent_slots_correct, lex_den):<17} | "
            f"{_metric_str(hass_stats.mismatch, hass_den):<17} | "
            f"{_metric_str(lex_stats.mismatch, lex_den):<17} | "
            f"{_metric_str(hass_stats.fallback, hass_den):<17} | "
            f"{_metric_str(lex_stats.fallback, lex_den):<17} | "
            f"{lex_stats.average_latency_ms:<8.1f}"
        )
    hass_total = _aggregate_mode_stats(results, "hassil")
    lex_total = _aggregate_mode_stats(results, "lexical")
    hass_den = hass_total.total - hass_total.drift
    lex_den = lex_total.total - lex_total.drift
    print("-" * 166)
    print(
        f"{'Overall':<24} | {lex_total.total:<5} | "
        f"{_metric_str(hass_total.intent_slots_correct, hass_den):<17} | "
        f"{_metric_str(lex_total.intent_slots_correct, lex_den):<17} | "
        f"{_metric_str(hass_total.mismatch, hass_den):<17} | "
        f"{_metric_str(lex_total.mismatch, lex_den):<17} | "
        f"{_metric_str(hass_total.fallback, hass_den):<17} | "
        f"{_metric_str(lex_total.fallback, lex_den):<17} | "
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
        den = stats.total - stats.drift
        print(
            f"{component:<16} | {stats.total:<5} | "
            f"{_metric_str(stats.correct, den):<16} | "
            f"{_metric_str(stats.intent_slots_correct, den):<19} | "
            f"{_metric_str(stats.fallback, den):<14}"
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
        if item_context := item.get("context"):
            print(f"  context={item_context}")
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
    expected_slots: Sequence[Mapping[str, Any]],
    language: str | None = None,
    hassil_intents: Any | None = None,
    hassil_slot_lists: dict[str, Any] | None = None,
    intent_context: Mapping[str, Any] | None = None,
) -> None:
    """Record component-only top-1 metrics for one ranked candidate set."""
    for component in ABLATION_COMPONENTS:
        selected = _select_ablation_candidate(ranked, component)
        _record_case_result(
            _stats_for(ablations, component, case["category"]),
            selected,
            case["expected_canonical"],
            case["expected_intent"],
            expected_slots,
            latency_ms=None,
            query=case["query"],
            language=language,
            hassil_intents=hassil_intents,
            hassil_slot_lists=hassil_slot_lists,
            intent_context=intent_context,
            expected_fallback=bool(case.get("expected_fallback", False)),
        )


def _markdown_metric(stats: Mapping[str, Any]) -> str:
    """Return a compact Markdown metric string."""
    return (
        f"{stats['intent_slot_accuracy']:.1f}% intent/slot, "
        f"{stats['canonical_accuracy']:.1f}% canonical, "
        f"{stats['mismatch_rate']:.1f}% mismatch, "
        f"{stats['fallback_rate']:.1f}% fallback"
    )


class _MarkdownReportRow:
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


def _markdown_separator_line(col_widths: list[int], col_aligns: tuple[str, ...]) -> str:
    """Return a Markdown table separator line."""
    parts: list[str] = []
    for i, width in enumerate(col_widths):
        dashes = width - 1
        if col_aligns[i] == ">":
            parts.append(" " + "-" * dashes + ": ")
        else:
            parts.append(" :" + "-" * dashes + " ")
    return "|" + "|".join(parts) + "|"


def _markdown_header_line(headers: tuple[str, ...], col_widths: list[int]) -> str:
    """Return a Markdown table header line."""
    parts: list[str] = []
    for i, header in enumerate(headers):
        width = col_widths[i]
        parts.append(f" {header:<{width}} ")
    return "|" + "|".join(parts) + "|"


def _markdown_data_row(row: _MarkdownReportRow, col_widths: list[int]) -> str:
    """Return a Markdown table data line."""
    cols = (
        f" {row.backticked_mode:<{col_widths[0]}} ",
        f" {row.total_s:>{col_widths[1]}} ",
        f" {row.intent_slot_s:>{col_widths[2]}} ",
        f" {row.canonical_s:>{col_widths[3]}} ",
        f" {row.mismatch_s:>{col_widths[4]}} ",
        f" {row.fallback_s:>{col_widths[5]}} ",
        f" {row.avg_ms_s:>{col_widths[6]}} ",
    )
    return "|" + "|".join(cols) + "|"


def _markdown_report(report: Mapping[str, Any]) -> str:
    """Return a human-readable Markdown report with dynamically aligned columns."""
    _headers = (
        "Mode",
        "Total",
        "Intent/Slot",
        "Canonical",
        "Mismatch",
        "Fallback",
        "Avg ms",
    )
    _col_aligns = ("<", ">", ">", ">", ">", ">", ">")
    _col_widths: list[int] = [len(h) for h in _headers]

    _all_rows: list[_MarkdownReportRow] = []
    for lang, payload in sorted(report.get("languages", {}).items()):
        for mode_name, summary in payload.get("summary", {}).items():
            stats = summary.get("overall", {})
            if not stats.get("total"):
                continue
            row = _MarkdownReportRow(lang, mode_name, stats)
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

    overall_summary = report["overall"]["summary"]
    versions = _format_dependency_versions(report.get("dependency_versions", {}))
    lines = [
        "# Assist Canonicalizer Evaluation",
        "",
        f"**Report schema:** v{report.get('report_schema_version', 1)}",
        "",
        f"**Dependency versions:** {versions}",
        "",
        "## Overall",
        "",
    ]
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
                    _markdown_header_line(_headers, _col_widths),
                    _markdown_separator_line(_col_widths, _col_aligns),
                ]
            )
        lines.append(_markdown_data_row(row, _col_widths))

    if threshold_failures := report["overall"].get("threshold_failures", []):
        lines.extend(["", "## Threshold Failures", ""])
        lines.extend(f"- {failure}" for failure in threshold_failures)
    return "\n".join(lines) + "\n"


def _text_report(report: Mapping[str, Any]) -> str:
    """Return a plain-text report matching the full console output."""
    lines: list[str] = [
        "=" * 120,
        "ASSIST CANONICALIZER PERFORMANCE EVALUATION REPORT",
        "=" * 120,
        f"Dataset Directory: {report.get('datasets_dir', 'tests/real_world')}/",
    ]
    lines.append(f"Total Languages: {report.get('total_languages', 0)}")
    versions = _format_dependency_versions(report.get("dependency_versions", {}))
    lines.extend(
        (
            f"Failure Detail Limit: {report.get('failure_limit', 0)}",
            f"Report Schema: v{report.get('report_schema_version', 1)}",
            f"Dependency Versions: {versions}",
            "=" * 120,
        )
    )
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
        if _miss := coverage.get("missing_candidate_intents", []):
            lines.append(f"Missing candidate intents: {len(_miss)} ({_short_names(list(_miss))})")
        if _untested := coverage.get("untested_candidate_intents", []):
            lines.append(
                f"Untested candidate intents: {len(_untested)} ({_short_names(list(_untested))})"
            )
        if _missing_ds := coverage.get("dataset_intents_without_candidates", []):
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

    if threshold_failures := overall.get("threshold_failures", []):
        lines.append("\nThreshold failures:")
        lines.extend(f"- {failure}" for failure in threshold_failures)
    lines.append("\nEvaluation Complete.")
    return "\n".join(lines) + "\n"


def _text_summary_table(title: str, payload: Mapping[str, Any]) -> list[str]:
    """Return text lines for a summary table with dynamically aligned columns."""
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
        hass_den = hass_cat.get("total", 0) - hass_cat.get("drift", 0)
        lex_den = lex_total - lex_cat.get("drift", 0)
        cat_rows.append(
            (
                cat,
                str(lex_total),
                _metric_str(hass_cat.get("intent_slots_correct", 0), hass_den),
                _metric_str(lex_cat.get("intent_slots_correct", 0), lex_den),
                _metric_str(hass_cat.get("mismatch", 0), hass_den),
                _metric_str(lex_cat.get("mismatch", 0), lex_den),
                _metric_str(hass_cat.get("fallback", 0), hass_den),
                _metric_str(lex_cat.get("fallback", 0), lex_den),
                f"{lex_cat.get('average_latency_ms', 0):.1f}",
            )
        )

    hass_overall = hassil_data.get("overall", {})
    lex_overall = lex_data.get("overall", {})
    hass_den = hass_overall.get("total", 0) - hass_overall.get("drift", 0)
    lex_den = lex_overall.get("total", 0) - lex_overall.get("drift", 0)
    overall_row: tuple[str, ...] = (
        "Overall",
        str(lex_overall.get("total", 0)),
        _metric_str(hass_overall.get("intent_slots_correct", 0), hass_den),
        _metric_str(lex_overall.get("intent_slots_correct", 0), lex_den),
        _metric_str(hass_overall.get("mismatch", 0), hass_den),
        _metric_str(lex_overall.get("mismatch", 0), lex_den),
        _metric_str(hass_overall.get("fallback", 0), hass_den),
        _metric_str(lex_overall.get("fallback", 0), lex_den),
        f"{lex_overall.get('average_latency_ms', 0):.1f}",
    )
    cat_rows.append(overall_row)
    hdr, sep, data = align_table(_headers, cat_rows, alignments="<")
    lines: list[str] = [f"\n{title}", sep, hdr, sep]
    lines.extend(data[:-1])
    lines.append(sep)
    lines.append(data[-1])
    lines.append(sep)
    return lines


def _text_ablation_table(title: str, payload: Mapping[str, Any]) -> list[str]:
    """Return text lines for an ablation table with dynamically aligned columns."""
    _headers = ("Component", "Total", "Canonical", "Intent/Slot", "Fallback")
    ab_rows: list[tuple[str, ...]] = []
    for comp in ABLATION_COMPONENTS:
        comp_data = payload.get(comp, {})
        overall = comp_data.get("overall", {})
        total = overall.get("total", 0)
        if total == 0:
            continue
        den = total - overall.get("drift", 0)
        ab_rows.append(
            (
                comp,
                str(total),
                _metric_str(overall.get("correct", 0), den),
                _metric_str(overall.get("intent_slots_correct", 0), den),
                _metric_str(overall.get("fallback", 0), den),
            )
        )
    hdr, sep, data = align_table(_headers, ab_rows, alignments="<")
    lines: list[str] = [f"\n{title}", sep, hdr, sep]
    lines.extend(data)
    lines.append(sep)
    return lines


def _write_json_report(path: str, report: Mapping[str, Any]) -> None:
    """Write the evaluation report as JSON using an atomic replacement."""
    atomic_write(
        path,
        orjson.dumps(_stringify_keys(report), option=orjson.OPT_INDENT_2).decode("utf-8") + "\n",
    )


def _hassil_context_from_sources(
    sources: Mapping[str, Any],
    slots: Mapping[str, tuple[str, ...]],
) -> tuple[Any, dict[str, Any]]:
    """Build HassIL intents and slot lists from loaded intent sources."""
    merged_intents: dict[str, Any] = {}
    for source in sources.values():
        hassil.merge_dict(merged_intents, source)
    return hassil.intents.Intents.from_dict(merged_intents), make_hassil_slot_lists(slots)


def _regenerate_case_expectations(
    case: dict[str, Any],
    index: int,
    hassil_intents: Any,
    hassil_slot_lists: dict[str, Any],
) -> tuple[int, bool]:
    """Regenerate one case's expected slots and drift marker."""
    expected_canonical = case.get("expected_canonical")
    expected_intent = case.get("expected_intent")
    if not expected_canonical or not expected_intent:
        return 0, False

    intent_context = case.get("context")
    results = run_hassil_recognize_all(
        expected_canonical,
        hassil_intents,
        hassil_slot_lists,
        intent_context,
    )
    matching = [r for r in results if r.intent.name == expected_intent]
    if not matching:
        query_val = case.get("query")
        print(
            f"  [DRIFT] test case #{index} '{query_val}' "
            f"(canonical: '{expected_canonical}') "
            f"cannot be parsed by HassIL for intent '{expected_intent}'"
        )
        if not case.get("drift"):
            case["drift"] = True
            return 1, True
        return 0, True

    updated_count = 0
    if "drift" in case:
        del case["drift"]
        updated_count += 1

    new_expected_slots = {name: str(entity.value) for name, entity in matching[0].entities.items()}
    if new_expected_slots != case.get("expected_slots", {}):
        case["expected_slots"] = new_expected_slots
        updated_count += 1
    return updated_count, False


def _regenerate_language_expectations(lang: str, path: str) -> None:
    """Regenerate expectation metadata for one dataset file."""
    print(f"Regenerating {lang.upper()} expectations...")
    with open(path, "rb") as f:
        data = orjson.loads(f.read())

    test_cases = data.get("test_cases", [])
    slots = _dataset_registry_slots(data, lang)
    sources = load_language_intent_sources(lang)
    hassil_intents, hassil_slot_lists = _hassil_context_from_sources(sources, slots)

    updated_count = 0
    drift_count = 0
    for index, case in enumerate(test_cases, start=1):
        case_updates, is_drift = _regenerate_case_expectations(
            case, index, hassil_intents, hassil_slot_lists
        )
        updated_count += case_updates
        drift_count += int(is_drift)

    if updated_count > 0:
        formatted_json = orjson.dumps(
            data, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE
        ).decode("utf-8")
        atomic_write(path, formatted_json)
        print(f"  Successfully updated {updated_count} cases in {path} (Drifted: {drift_count})")
    else:
        print(f"  No updates needed for {path} (Drifted: {drift_count})")


def regenerate_all_expectations(datasets: dict[str, str], datasets_dir: str) -> bool:
    """Regenerate expected_slots in dataset JSON files dynamically using HassIL."""
    print("=" * 120)
    print("ASSIST CANONICALIZER EXPECTATION REGENERATION")
    print("=" * 120)
    print(f"Dataset Directory: {datasets_dir}/")
    print(f"Languages: {', '.join(sorted(datasets.keys()))}")
    print("=" * 120)

    for lang, path in sorted(datasets.items()):
        _regenerate_language_expectations(lang, path)

    return True


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
    max_mismatch_rate: float | None = None,
    *,
    scope: str | None = None,
) -> list[str]:
    """Return threshold failure messages for the selected aggregate stats."""
    failures = []
    prefix = f"{scope}: " if scope else ""
    if (
        min_intent_slot_accuracy is not None
        and stats.intent_slot_accuracy < min_intent_slot_accuracy
    ):
        failures.append(
            f"{prefix}intent/slot accuracy {stats.intent_slot_accuracy:.1f}% "
            f"is below {min_intent_slot_accuracy:.1f}%"
        )
    if max_fallback_rate is not None and stats.fallback_rate > max_fallback_rate:
        failures.append(
            f"{prefix}fallback rate {stats.fallback_rate:.1f}% is above {max_fallback_rate:.1f}%"
        )
    if max_mismatch_rate is not None and stats.mismatch_rate > max_mismatch_rate:
        failures.append(
            f"{prefix}mismatch rate {stats.mismatch_rate:.1f}% is above {max_mismatch_rate:.1f}%"
        )
    return failures


def _load_and_validate_dataset(
    lang: str, path: str
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Load and validate test cases and slots for a dataset path."""
    with open(path, encoding="utf-8") as f:
        data = orjson.loads(f.read())
    if not isinstance(data, dict):
        print(f"Error: Dataset root must be an object: {path}")
        return None
    raw_cases = data.get("test_cases", [])
    if not isinstance(raw_cases, list):
        print(f"Error: test_cases must be a list: {path}")
        return None
    try:
        test_cases = _validate_test_cases(raw_cases, lang, path)
        slots = _dataset_registry_slots(data, lang)

    except ValueError as err:
        print(f"Error: {err}")
        return None
    return test_cases, slots


def _evaluate_mode_candidates(
    mode_name: str,
    query: str,
    lang: str,
    runtime: Any,
    index: Any,
    hassil_intents: Any,
    hassil_slot_lists: dict[str, Any],
    benchmark_slot_prefs: set[tuple[str, str]] | None,
    expected_canonical: str,
    expected_intent: str,
    expected_slots: Sequence[Mapping[str, Any]],
    intent_context: Mapping[str, Any] | None = None,
) -> tuple[RankedCandidate, ...]:
    """Evaluate HassIL alone or the production shortcut-then-ranking flow."""
    res = run_hassil_recognize_best(
        query,
        hassil_intents,
        hassil_slot_lists,
        intent_context,
        lang,
    )
    if mode_name == "lexical" and res is None:
        return runtime.rank_with_dynamic_candidates(
            lang,
            index,
            query,
            slot_preferences=benchmark_slot_prefs,
            intent_context=intent_context,
        )
    if res is not None:
        actual_slots = {name: entity.value for name, entity in res.entities.items()}
        is_intent_ok = _intents_match(res.intent.name, expected_intent)
        is_slots_ok = _slots_match_any(actual_slots, expected_slots, language=lang)
        canonical_text = (
            expected_canonical if (is_intent_ok and is_slots_ok) else f"mismatch: {res.intent.name}"
        )
        candidate = Candidate(
            text=canonical_text,
            intent_name=res.intent.name,
            metadata={
                "slots": orjson.dumps(actual_slots).decode("utf-8"),
                "evaluation_path": (
                    "hassil_baseline" if mode_name == "hassil" else "hassil_shortcut"
                ),
            },
        )
        scores = _ScoreBreakdown(
            rapidfuzz_score=1.0,
            char_ngram_score=1.0,
            bm25_score=1.0,
            intent_score=1.0,
            final_score=1.0,
        )
        return (_RankedCandidate(candidate=candidate, scores=scores),)
    return ()


def _evaluate_case(
    case: dict[str, Any],
    mode_name: str,
    lang: str,
    runtime: Any,
    index: Any,
    hassil_intents: Any,
    hassil_slot_lists: dict[str, Any],
    benchmark_slot_prefs: set[tuple[str, str]] | None,
    stats: CategoryStats,
) -> CaseEvaluationResult:
    """Run a single case for a specific mode and return evaluation results."""
    query = case["query"]
    expected_canonical = case["expected_canonical"]
    expected_intent = case["expected_intent"]
    static_slots = case.get("expected_slots", {})
    intent_context = case.get("context")

    # Resolve expected slots dynamically using HassIL on the clean canonical text
    results = run_hassil_recognize_all(
        expected_canonical,
        hassil_intents,
        hassil_slot_lists,
        intent_context,
    )
    matching = [r for r in results if _intents_match(r.intent.name, expected_intent)]

    is_drift = case.get("drift", False)
    if not matching:
        expected_slots = [static_slots]
    else:
        expected_slots = [
            {name: entity.value for name, entity in r.entities.items()} for r in matching
        ]

    start_time = time.perf_counter()
    ranked = _evaluate_mode_candidates(
        mode_name,
        query,
        lang,
        runtime,
        index,
        hassil_intents,
        hassil_slot_lists,
        benchmark_slot_prefs,
        expected_canonical,
        expected_intent,
        expected_slots,
        intent_context,
    )

    selected, gate = _select_accepted_with_gate(ranked)
    latency_ms = (time.perf_counter() - start_time) * 1000
    is_ok, reason, actual_slots, selected = _record_case_result(
        stats,
        selected,
        expected_canonical,
        expected_intent,
        expected_slots,
        latency_ms,
        query=query,
        language=lang,
        ranked=ranked,
        is_drift=is_drift,
        hassil_intents=hassil_intents,
        hassil_slot_lists=hassil_slot_lists,
        intent_context=intent_context,
        expected_fallback=bool(case.get("expected_fallback", False)),
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
        expected_slots,
    )
    return CaseEvaluationResult(
        row=row,
        is_ok=is_ok,
        reason=reason,
        actual_slots=actual_slots,
        selected=selected,
        gate=gate,
        ranked=ranked,
        expected_slots=expected_slots,
    )


def _print_evaluation_header(
    datasets: dict[str, str],
    datasets_dir: str,
    failure_limit: int,
    output_json: str | None,
    output_md: str | None,
    output_txt: str | None,
    dependency_versions: Mapping[str, Any],
) -> None:
    """Print the header section of the performance evaluation report."""
    print("=" * 120)
    print("ASSIST CANONICALIZER PERFORMANCE EVALUATION REPORT")
    print("=" * 120)
    print(f"Dataset Directory: {datasets_dir}/")
    print(f"Total Languages: {len(datasets)}")
    print(f"Failure Detail Limit: {failure_limit}")
    print(f"Dependency Versions: {_format_dependency_versions(dependency_versions)}")
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
    if output_txt:
        try:
            rel_txt = Path(output_txt).relative_to(_REPO_ROOT)
        except ValueError:
            rel_txt = Path(output_txt)
        print(f"Text Output: {rel_txt}")
    print("=" * 120)


def _build_language_evaluation_context(
    lang: str,
    slots: dict[str, Any],
) -> LanguageEvaluationContext:
    """Build production index, runtime, and HassIL objects for one language."""
    sources = load_language_intent_sources(lang)
    start_build = time.perf_counter()
    candidates = build_candidates_from_intent_sources(lang, sources, slots)
    build_latency_ms = (time.perf_counter() - start_build) * 1000
    index = build_index(lang, candidates)
    hassil_intents, hassil_slot_lists = _hassil_context_from_sources(sources, slots)

    runtime = CanonicalizerRuntime()
    runtime.set_index(index)
    runtime.update_registry_slot_values(slots)

    return LanguageEvaluationContext(
        language=lang,
        sources=sources,
        candidates=candidates,
        build_latency_ms=build_latency_ms,
        index=index,
        hassil_intents=hassil_intents,
        hassil_slot_lists=hassil_slot_lists,
        runtime=runtime,
    )


def _language_coverage_payload(
    context: LanguageEvaluationContext,
    test_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return coverage diagnostics for one prepared language context."""
    candidate_intents = {candidate.intent_name for candidate in context.candidates}
    coverage: dict[str, Any] = _coverage_payload(
        test_cases,
        context.sources,
        candidate_intents,
        len(context.candidates),
        context.build_latency_ms,
    )
    coverage["case_count"] = len(test_cases)
    return coverage


def _evaluation_modes(skip_hassil: bool) -> tuple[str, ...]:
    """Return evaluation modes for the current command-line options."""
    return ("lexical",) if skip_hassil else ("hassil", "lexical")


def _evaluate_language_mode(
    mode_name: str,
    context: LanguageEvaluationContext,
    test_cases: list[dict[str, Any]],
    skip_ablations: bool,
    benchmark_slot_prefs: set[tuple[str, str]] | None,
    results: dict[str, dict[str, CategoryStats]],
    ablations: dict[str, dict[str, CategoryStats]],
    failures: list[dict[str, Any]],
    language_rows: list[dict[str, Any]],
    all_case_rows: list[dict[str, Any]],
) -> None:
    """Evaluate all cases for one language/mode combination."""
    for case in test_cases:
        stats = _stats_for(results, mode_name, case["category"])
        result = _evaluate_case(
            case,
            mode_name,
            context.language,
            context.runtime,
            context.index,
            context.hassil_intents,
            context.hassil_slot_lists,
            benchmark_slot_prefs,
            stats,
        )
        language_rows.append(result.row)
        all_case_rows.append(result.row)

        if mode_name == "lexical" and not skip_ablations:
            ablation_ranked = result.ranked
            used_hassil_shortcut = (
                result.selected is not None
                and result.selected.candidate.metadata.get("evaluation_path") == "hassil_shortcut"
            )
            if used_hassil_shortcut:
                ablation_ranked = context.runtime.rank_with_dynamic_candidates(
                    context.language,
                    context.index,
                    case["query"],
                    slot_preferences=benchmark_slot_prefs,
                    intent_context=case.get("context"),
                )
            _record_ablations(
                ablations,
                ablation_ranked,
                case,
                result.expected_slots,
                language=context.language,
                hassil_intents=context.hassil_intents,
                hassil_slot_lists=context.hassil_slot_lists,
                intent_context=case.get("context"),
            )
        if not result.is_ok:
            failures.append(
                _failure_detail(
                    mode_name,
                    case,
                    result.selected,
                    result.reason,
                    result.actual_slots,
                    result.gate,
                    result.expected_slots,
                )
            )


def _evaluate_language_modes(
    context: LanguageEvaluationContext,
    test_cases: list[dict[str, Any]],
    skip_hassil: bool,
    skip_ablations: bool,
    benchmark_slot_prefs: set[tuple[str, str]] | None,
    results: dict[str, dict[str, CategoryStats]],
    ablations: dict[str, dict[str, CategoryStats]],
    failures: list[dict[str, Any]],
    language_rows: list[dict[str, Any]],
    all_case_rows: list[dict[str, Any]],
) -> None:
    """Evaluate all enabled modes for one prepared language context."""
    for mode_name in _evaluation_modes(skip_hassil):
        _evaluate_language_mode(
            mode_name,
            context,
            test_cases,
            skip_ablations,
            benchmark_slot_prefs,
            results,
            ablations,
            failures,
            language_rows,
            all_case_rows,
        )


def _evaluate_dataset_language(
    lang: str,
    test_cases: list[dict[str, Any]],
    slots: dict[str, Any],
    skip_hassil: bool,
    skip_ablations: bool,
    benchmark_slot_prefs: set[tuple[str, str]] | None,
    global_results: dict[str, Any],
    global_ablations: dict[str, Any],
    all_case_rows: list[dict[str, Any]],
    failure_limit: int,
) -> tuple[dict[str, Any], CategoryStats]:
    """Evaluate cases for a single language dataset and return language report payload."""
    context = _build_language_evaluation_context(lang, slots)
    coverage = _language_coverage_payload(context, test_cases)
    _print_coverage(lang, test_cases, coverage)

    print(f"Processing {lang.upper()} ({len(test_cases)} cases) ...", flush=True)

    results = _new_results()
    ablations = _new_ablation_results()
    failures: list[dict[str, Any]] = []
    language_rows: list[dict[str, Any]] = []

    _evaluate_language_modes(
        context,
        test_cases,
        skip_hassil,
        skip_ablations,
        benchmark_slot_prefs,
        results,
        ablations,
        failures,
        language_rows,
        all_case_rows,
    )

    _print_summary_table(f"Summary: {lang.upper()}", results)
    if not skip_ablations:
        _print_ablation_table(f"Ablation Top-1: {lang.upper()}", ablations)
    _print_failure_details(failures, failure_limit)
    _merge_results(global_results, results)
    _merge_results(global_ablations, ablations)
    return (
        {
            "coverage": coverage,
            "summary": _summary_payload(results),
            "ablations": _summary_payload(ablations),
            "failures": failures,
            "cases": language_rows,
        },
        _aggregate_mode_stats(results, "lexical"),
    )


async def run_evaluation(
    datasets: dict[str, str],
    failure_limit: int,
    output_json: str | None,
    output_md: str | None,
    min_intent_slot_accuracy: float | None = None,
    max_fallback_rate: float | None = None,
    max_mismatch_rate: float | None = None,
    min_language_intent_slot_accuracy: float | None = None,
    max_language_fallback_rate: float | None = None,
    max_language_mismatch_rate: float | None = None,
    datasets_dir: str = "tests/real_world",
    skip_hassil: bool = False,
    skip_ablations: bool = False,
    output_txt: str | None = None,
) -> bool:
    """Run evaluation on the datasets and print the summary report."""
    _bootstrap_project_imports()
    if not datasets:
        print(f"Error: No datasets found in {datasets_dir}")
        return False
    if failure_limit < 0:
        print("Error: --failure-limit must be zero or positive")
        return False

    dependency_versions = _benchmark_dependency_versions()
    _print_evaluation_header(
        datasets,
        datasets_dir,
        failure_limit,
        output_json,
        output_md,
        output_txt,
        dependency_versions,
    )

    overall_success = True
    global_results = _new_results()
    global_ablations = _new_ablation_results()
    all_case_rows: list[dict[str, Any]] = []
    threshold_failures: list[str] = []
    benchmark_slot_prefs = _load_benchmark_slot_preferences(datasets)
    report: dict[str, Any] = {
        "report_schema": "assist_canonicalizer_accuracy",
        "report_schema_version": ACCURACY_REPORT_SCHEMA_VERSION,
        "languages": {},
        "datasets_dir": datasets_dir,
        "total_languages": len(datasets),
        "failure_limit": failure_limit,
        "dependency_versions": dependency_versions,
    }

    for lang, path in sorted(datasets.items()):
        loaded = _load_and_validate_dataset(lang, path)
        if loaded is None:
            return False
        test_cases, slots = loaded
        if not test_cases:
            continue

        language_payload, language_stats = _evaluate_dataset_language(
            lang,
            test_cases,
            slots,
            skip_hassil,
            skip_ablations,
            benchmark_slot_prefs,
            global_results,
            global_ablations,
            all_case_rows,
            failure_limit,
        )
        language_threshold_failures = _threshold_failures(
            language_stats,
            min_language_intent_slot_accuracy,
            max_language_fallback_rate,
            max_language_mismatch_rate,
            scope=lang.upper(),
        )
        language_payload["threshold_failures"] = language_threshold_failures
        report["languages"][lang] = language_payload
        if language_threshold_failures:
            overall_success = False
            threshold_failures.extend(language_threshold_failures)
            print(f"\nThreshold failures: {lang.upper()}")
            for failure in language_threshold_failures:
                print(f"- {failure}")

    _print_summary_table("Summary: ALL LANGUAGES", global_results)
    if not skip_ablations:
        _print_ablation_table("Ablation Top-1: ALL LANGUAGES", global_ablations)
    lex_stats = _aggregate_mode_stats(global_results, "lexical")
    if aggregate_threshold_failures := _threshold_failures(
        lex_stats,
        min_intent_slot_accuracy,
        max_fallback_rate,
        max_mismatch_rate,
        scope="ALL",
    ):
        overall_success = False
        threshold_failures.extend(aggregate_threshold_failures)
        print("\nThreshold failures:")
        for failure in aggregate_threshold_failures:
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
    lex_total = _aggregate_mode_stats(global_results, "lexical")
    print(f"\nTotal drift cases: {lex_total.drift}")
    if lex_total.drift > 0:
        print(
            "Reminder: Please regenerate expectations "
            "(uv run tools/benchmark.py --regenerate-expectations) "
            "to update test suite expectations after upgrading home-assistant-intents."
        )
    print("\nEvaluation Complete.")
    return overall_success


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

        return StatsResult(
            mean=mean,
            median=median,
            p50=_percentile(sorted_vals, n, 50),
            p95=_percentile(sorted_vals, n, 95),
            p99=_percentile(sorted_vals, n, 99),
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


def _percentile(sorted_vals: list[float], n: int, p: float) -> float:
    """Return a percentile value from a sorted measurement list."""
    k = (n - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    lower = sorted_vals[f]
    upper = sorted_vals[c] if c < n else sorted_vals[-1]
    return lower + (upper - lower) * (k - f)


class _PhaseContext:
    """Context manager returned by PhaseTimer.phase."""

    def __init__(self, timer: PhaseTimer, name: str) -> None:
        """Initialize the phase timing context manager."""
        self._timer = timer
        self._name = name

    def __enter__(self) -> _PhaseContext:
        """Enter the phase context and start the timer."""
        track_memory = not self._name.endswith("_inner")
        self._timer.start(self._name, track_memory=track_memory)
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
        self._stack: list[tuple[str, float, float, bool]] = []
        self._monitor = resource_monitor
        self.enabled: bool = True

    def _current_rss(self) -> float:
        """Get the current process RSS memory utilization in MB."""
        if self._monitor is not None and self._monitor.is_alive():
            sampled_rss_mb = self._monitor.current_rss_mb
            if sampled_rss_mb > 0.0:
                return sampled_rss_mb
        try:
            with open("/proc/self/stat", "rb") as f:
                parts = f.read().decode().split()
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(parts[23]) * page_size / (1024 * 1024)
        except Exception:
            return 0.0

    def start(self, name: str, track_memory: bool = True) -> None:
        """Start measuring a phase by name."""
        if not self.enabled:
            return
        rss = self._current_rss() if track_memory else 0.0
        self.current_phase = name
        self._stack.append((name, time.perf_counter(), rss, track_memory))

    def stop(self) -> None:
        """Stop the current active phase timing and record statistics."""
        if not self.enabled:
            return
        if not self._stack:
            return
        name, start_time, start_rss, track_memory = self._stack.pop()
        elapsed = time.perf_counter() - start_time
        rss_delta = max(0.0, self._current_rss() - start_rss) if track_memory else 0.0
        self.phases.setdefault(name, []).append(elapsed)
        self.memory_deltas.setdefault(name, []).append(rss_delta)
        self.current_phase = self._stack[-1][0] if self._stack else None

    def phase(self, name: str) -> _PhaseContext:
        """Return a context manager to easily measure a phase."""
        return _PhaseContext(self, name)

    def record(self, name: str, elapsed: float, rss_delta: float = 0.0) -> None:
        """Directly record performance numbers for a given phase name."""
        if not self.enabled:
            return
        self.phases.setdefault(name, []).append(elapsed)
        self.memory_deltas.setdefault(name, []).append(rss_delta)

    def stats(self) -> dict[str, dict[str, StatsResult]]:
        """Compute statistical data for all recorded phases."""
        result: dict[str, dict[str, StatsResult]] = {
            name: {
                "elapsed": StatsEngine.compute(self.phases[name]),
                "memory_delta_mb": StatsEngine.compute(self.memory_deltas.get(name, [0.0])),
            }
            for name in self.phases
        }
        return result


class ResourceMonitor(threading.Thread):
    """Monitors CPU, memory (RSS, VmSize, VmPeak), and GC of the current process."""

    def __init__(self, interval: float = 0.1) -> None:
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
            with suppress(Exception):
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
                with suppress(OSError), open("/proc/self/status") as sf:
                    for line in sf:
                        if line.startswith("VmSize:"):
                            vm_size = int(line.split()[1])
                        elif line.startswith("VmPeak:"):
                            vm_peak = int(line.split()[1])
                with self._lock:
                    self.cpu_samples.append((t, float(utime + stime)))
                    self.rss_samples.append(rss_mb)
                    self.vm_size_samples.append(vm_size / 1024.0)
                    self.vm_peak_samples.append(vm_peak / 1024.0)
                    self.current_rss_mb = rss_mb
                    self.current_vm_size_mb = vm_size / 1024.0
                    self.current_vm_peak_mb = vm_peak / 1024.0
            time.sleep(self.interval)

    def stop_monitor(self) -> None:
        """Signal the monitoring thread to stop execution."""
        self.stop_event.set()

    def snapshot_gc(self) -> None:
        """Take a snapshot of current garbage collection stats.

        Note: This method is safe to call even if the monitoring thread has not
        been started (e.g., when target is 'components').
        """
        with suppress(Exception):
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

    def get_cpu_metrics(self) -> dict[str, float]:
        """Compute average and peak CPU utilization percentages.

        Note: This method returns zeroed metrics safely if the thread has not
        been started.
        """
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
                "avg_pct": (sum(percentages) / len(percentages) if percentages else 0.0),
                "peak_pct": max(percentages, default=0.0),
            }

    def get_memory_metrics(self) -> dict[str, float]:
        """Compute average and peak memory (RSS and Vm) metrics in MB.

        Note: This method returns zeroed metrics safely if the thread has not
        been started.
        """
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

    @property
    def baseline_dir(self) -> Path:
        """Return the baseline directory path."""
        return self._baseline_dir

    @baseline_dir.setter
    def baseline_dir(self, path: Path) -> None:
        """Set the baseline directory path."""
        self._baseline_dir = path

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
                print(f"Warning: cannot load baseline file: {path} - {err}", file=sys.stderr)
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
        _compare_regression_recursive(current, baseline, "", regressions, max_regression_pct)
        return regressions


def _record_regression_if_needed(
    regressions: list[str],
    cur: float,
    base: float,
    label: str,
    max_regression_pct: float,
) -> None:
    """Append one regression message when a metric exceeds the threshold."""
    if base == 0:
        return
    pct_change = (cur - base) / base * 100.0
    if pct_change > max_regression_pct:
        regressions.append(
            f"REGRESSION [{label}]: {pct_change:+.1f}% (baseline={base:.4f}, current={cur:.4f})"
        )


def _compare_regression_recursive(
    cur_val: Any,
    base_val: Any,
    path: str,
    regressions: list[str],
    max_regression_pct: float,
) -> None:
    """Recursively compare profile metrics and collect regressions."""
    if isinstance(cur_val, dict) and isinstance(base_val, dict):
        for key in cur_val:
            if key in base_val:
                _compare_regression_recursive(
                    cur_val[key],
                    base_val[key],
                    f"{path}.{key}" if path else key,
                    regressions,
                    max_regression_pct,
                )
    elif isinstance(cur_val, (int, float)) and isinstance(base_val, (int, float)):
        leaf = path.split(".")[-1] if path else ""
        if leaf in ("mean", "median", "p95", "p99", "max", "min"):
            _record_regression_if_needed(
                regressions,
                float(cur_val),
                float(base_val),
                path,
                max_regression_pct,
            )


def _profile_regressions(reports: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return labeled regressions from one or more profiling reports."""
    regressions: list[str] = []
    for target, report in reports.items():
        regressions.extend(
            f"{target}: {regression}" for regression in report.get("regressions", [])
        )
    return regressions


def _serialize_profile_report_value(obj: object) -> object:
    """Serialize profile report values into JSON-compatible containers."""
    if isinstance(obj, StatsResult):
        return StatsEngine.as_dict(obj)
    if isinstance(obj, dict):
        return {str(key): _serialize_profile_report_value(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_profile_report_value(value) for value in obj]
    return obj


class ReportGenerator:
    """Generate multi-format profiling reports (terminal, JSON, Markdown, text)."""

    @staticmethod
    def terminal(report: dict[str, Any]) -> None:
        """Print the profiling report to the terminal."""
        print("\n" + "=" * 90)
        print("ALGORITHMIC PERFORMANCE PROFILING REPORT")
        print("=" * 90)
        print(f"Report Schema:   v{report.get('report_schema_version', 1)}")
        print(f"Target:          {report.get('target', 'unknown')}")
        print(f"Iterations:      {report.get('iterations', 0)}")
        print(f"Warmup:          {report.get('warmup', 0)}")
        print(f"Granularity:     {report.get('granularity', 'coarse')}")
        langs = report.get("languages", [])
        print(f"Languages:       {', '.join(langs) if langs else 'all'}")
        print(
            f"Dependencies:    {_format_dependency_versions(report.get('dependency_versions', {}))}"
        )
        print("-" * 90)

        if agg := report.get("aggregate", {}):
            _print_stat_block("Aggregate Performance", agg)

        if res := report.get("resource", {}):
            _print_resource_block("Resource Utilization", res)

        if phases := report.get("phases", {}):
            _print_phase_table("Phase Timing Breakdown", phases)

        if components := report.get("components", {}):
            _print_phase_table("Component Micro-Profile", components)

        if coverage := report.get("coverage", {}):
            _print_count_table("Runtime Branch Coverage", coverage)

        if category_coverage := report.get("category_coverage", {}):
            _print_count_table("Runtime Category Coverage", category_coverage)

        if scenario_stats := report.get("scenario_stats", {}):
            _print_phase_table("Runtime Scenario Timing", scenario_stats)

        if slow_queries := report.get("slow_queries", []):
            _print_slow_queries("Slowest Runtime Queries", slow_queries)

        per_lang = report.get("per_language", {})
        for lang_key in sorted(per_lang):
            lang_data = per_lang[lang_key]
            print(f"\n{'─' * 90}")
            print(f"Language: {lang_key.upper()}")
            if lang_data.get("aggregate"):
                _print_stat_block("  Aggregate", lang_data["aggregate"])
            if lang_data.get("phases"):
                _print_phase_table("  Phase Timing", lang_data["phases"])
            if lang_data.get("coverage"):
                _print_count_table("  Runtime Branch Coverage", lang_data["coverage"])
            if lang_data.get("category_coverage"):
                _print_count_table("  Runtime Category Coverage", lang_data["category_coverage"])
            if lang_data.get("scenario_stats"):
                _print_phase_table("  Runtime Scenario Timing", lang_data["scenario_stats"])
            if lang_data.get("slow_queries"):
                _print_slow_queries("  Slowest Runtime Queries", lang_data["slow_queries"])

        if regressions := report.get("regressions", []):
            print(f"\n{'!' * 90}")
            print("REGRESSION DETECTIONS:")
            for r in regressions:
                print(f"  {r}")
            print(f"{'!' * 90}")

        if stability := report.get("stability", {}):
            print(f"\nStability: {stability}")

        print("\n" + "=" * 90)

    @staticmethod
    def json_report(report: dict[str, Any], path: str) -> None:
        """Save the profiling report to a JSON file."""
        out = Path(path)

        serializable = _serialize_profile_report_value(report)
        atomic_write(str(out), json.dumps(serializable, indent=2, default=str) + "\n")
        print(f"JSON report saved to {out}")

    @staticmethod
    def markdown_report(report: dict[str, Any], path: str) -> None:
        """Save the profiling report to a Markdown file."""
        lines: list[str] = [
            "# Assist Canonicalizer - Algorithmic Performance Profile",
            "",
            f"**Report schema:** v{report.get('report_schema_version', 1)}  ",
            f"**Target:** `{report.get('target', 'unknown')}`  ",
            f"**Iterations:** {report.get('iterations', 0)} | "
            f"**Warmup:** {report.get('warmup', 0)} | "
            f"**Granularity:** {report.get('granularity', 'coarse')}",
            f"**Dependency versions:** "
            f"{_format_dependency_versions(report.get('dependency_versions', {}))}",
            "",
        ]

        if agg := report.get("aggregate", {}):
            lines.extend(_md_stat_table("## Aggregate Performance", agg))

        if phases := report.get("phases", {}):
            lines.extend(("## Phase Timing", ""))
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

        if components := report.get("components", {}):
            lines.extend(("## Component Micro-Profile", ""))
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

        if coverage := report.get("coverage", {}):
            lines.extend(_md_count_table("## Runtime Branch Coverage", coverage))

        if category_coverage := report.get("category_coverage", {}):
            lines.extend(_md_count_table("## Runtime Category Coverage", category_coverage))

        if scenario_stats := report.get("scenario_stats", {}):
            lines.extend(("## Runtime Scenario Timing", ""))
            lines.extend(_md_phase_rows(scenario_stats))

        if slow_queries := report.get("slow_queries", []):
            lines.extend(_md_slow_queries("## Slowest Runtime Queries", slow_queries))

        if regressions := report.get("regressions", []):
            lines.extend(("## Regression Detections", ""))
            lines.extend(f"- {r}" for r in regressions)
            lines.append("")

        while lines and lines[-1] == "":
            lines.pop()
        atomic_write(path, "\n".join(lines) + "\n")
        print(f"Markdown report saved to {Path(path)}")

    @staticmethod
    def _text_report_aggregate(agg: dict[str, Any], headers: tuple[str, ...]) -> list[str]:
        """Format the aggregate performance section for the text report."""
        lines = []
        if agg:
            _rows: list[tuple[str, ...]] = []
            _rows.extend(
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
                for name, s in agg.items()
                if isinstance(s, dict)
            )
            if _rows:
                hdr, sep, data = align_table(headers, _rows, alignments="<>")
                lines.extend(("\nAggregate Performance:", hdr, sep))
                lines.extend(data)
        return lines

    @staticmethod
    def _text_report_phases(phases: dict[str, Any], headers: tuple[str, ...]) -> list[str]:
        """Format the phase timing breakdown section for the text report."""
        lines = []
        if phases:
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
            if _rows:
                hdr, sep, data = align_table(headers, _rows, alignments="<>")
                lines.extend(["", "Phase Timing Breakdown:", hdr, sep, *data])
        return lines

    @staticmethod
    def _text_report_components(components: dict[str, Any]) -> list[str]:
        """Format the component micro-profile section for the text report."""
        lines = []
        if components:
            _rows: list[tuple[str, ...]] = []
            for name, comp_data in components.items():
                e = comp_data.get("elapsed", {})
                _rows.append(
                    (
                        name,
                        f"{e.get('mean', 0) * 1_000_000:.1f}",
                        f"{e.get('median', 0) * 1_000_000:.1f}",
                        f"{e.get('p95', 0) * 1_000_000:.1f}",
                        f"{e.get('p99', 0) * 1_000_000:.1f}",
                        f"{e.get('cov_pct', 0):.1f}",
                    )
                )
            if _rows:
                headers = ("Component", "Mean(μs)", "Median(μs)", "p95(μs)", "p99(μs)", "CoV%")
                hdr, sep, data = align_table(headers, _rows, alignments="<>")
                lines.extend(["", "Component Micro-Profile:", hdr, sep, *data])
        return lines

    @staticmethod
    def _text_report_per_language(
        per_lang: dict[str, Any],
        headers_agg: tuple[str, ...],
        headers_ph: tuple[str, ...],
    ) -> list[str]:
        """Format the per-language details section for the text report."""
        lines = []
        for lang_key in sorted(per_lang):
            lang_data = per_lang[lang_key]
            lines.extend(["", "─" * 80, f"Language: {lang_key.upper()}"])
            if lang_agg := lang_data.get("aggregate", {}):
                _la_rows: list[tuple[str, ...]] = []
                _la_rows.extend(
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
                    for name, s in lang_agg.items()
                    if isinstance(s, dict)
                )
                if _la_rows:
                    hdr, sep, data = align_table(headers_agg, _la_rows, alignments="<>")
                    lines.extend(["  Aggregate:", f"  {hdr}", f"  {sep}"])
                    lines.extend(f"  {d}" for d in data)

            if lang_phases := lang_data.get("phases", {}):
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
                    hdr, sep, data = align_table(headers_ph, _lp_rows, alignments="<>")
                    lines.extend(["  Phase Timing:", f"  {hdr}", f"  {sep}"])
                    lines.extend(f"  {d}" for d in data)
        return lines

    @staticmethod
    def text_report(report: dict[str, Any], path: str) -> None:
        """Save the profiling report to a text file."""
        lines: list[str] = [
            "ALGORITHMIC PERFORMANCE PROFILING REPORT",
            "=" * 80,
            f"Report Schema: v{report.get('report_schema_version', 1)}",
            f"Target: {report.get('target', 'unknown')}",
            f"Iterations: {report.get('iterations', 0)}",
            f"Warmup: {report.get('warmup', 0)}",
            f"Granularity: {report.get('granularity', 'coarse')}",
            f"Dependency Versions: "
            f"{_format_dependency_versions(report.get('dependency_versions', {}))}",
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

        lines.extend(
            ReportGenerator._text_report_aggregate(report.get("aggregate", {}), _headers_agg)
        )
        lines.extend(ReportGenerator._text_report_phases(report.get("phases", {}), _headers_ph))
        lines.extend(ReportGenerator._text_report_components(report.get("components", {})))
        lines.extend(_text_count_table("Runtime Branch Coverage", report.get("coverage", {})))
        lines.extend(
            _text_count_table("Runtime Category Coverage", report.get("category_coverage", {}))
        )
        lines.extend(
            ReportGenerator._text_report_phases(report.get("scenario_stats", {}), _headers_ph)
        )
        lines.extend(_text_slow_queries("Slowest Runtime Queries", report.get("slow_queries", [])))
        lines.extend(
            ReportGenerator._text_report_per_language(
                report.get("per_language", {}), _headers_agg, _headers_ph
            )
        )

        if regressions := report.get("regressions", []):
            lines.append("\nREGRESSION DETECTIONS:")
            lines.extend(f"  {r}" for r in regressions)
        lines.extend(("", "=" * 80))
        while lines and lines[-1] == "":
            lines.pop()
        atomic_write(path, "\n".join(lines) + "\n")
        print(f"Text report saved to {Path(path)}")


def _print_stat_block(title: str, stats: dict[str, Any]) -> None:
    """Print standard stats block to the terminal in tabular format."""
    print(f"\n{title}:")
    _headers = ("Metric", "Mean", "Median", "p95", "p99", "StdDev", "Min", "Max", "CoV%")
    _rows: list[tuple[str, ...]] = []
    _rows.extend(
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
        for name, s in stats.items()
        if isinstance(s, dict)
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


def _print_count_table(title: str, counts: Mapping[str, Any]) -> None:
    """Print integer counters with percentages when a total is present."""
    if not counts:
        return
    print(f"\n{title}:")
    total = int(
        counts.get("total_queries", 0)
        or sum(value for value in counts.values() if isinstance(value, int | float))
    )
    rows: list[tuple[str, ...]] = []
    for key, value in sorted(counts.items()):
        if not isinstance(value, int | float):
            continue
        pct = (float(value) / total * 100.0) if total and key != "total_queries" else 100.0
        rows.append((key, str(int(value)), f"{pct:.1f}%"))
    hdr, sep, data = align_table(("Metric", "Count", "Share"), rows, alignments="<>>")
    print(hdr)
    print(sep)
    for line in data:
        print(line)


def _print_slow_queries(title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Print the slow-query summary for runtime profiling."""
    if not rows:
        return
    print(f"\n{title}:")
    table_rows: list[tuple[str, ...]] = [
        (
            str(item.get("language", "")),
            str(item.get("category", "")),
            f"{float(item.get('mean_sec', 0.0)) * 1000:.3f}",
            f"{float(item.get('max_sec', 0.0)) * 1000:.3f}",
            str(item.get("dynamic_candidate_count", 0)),
            str(item.get("query", ""))[:72],
        )
        for item in rows
    ]
    headers = ("Lang", "Category", "Mean(ms)", "Max(ms)", "Dynamic", "Query")
    hdr, sep, data = align_table(headers, table_rows, alignments="<<>>>")
    print(hdr)
    print(sep)
    for line in data:
        print(line)


def _md_count_table(title: str, counts: Mapping[str, Any]) -> list[str]:
    """Return a Markdown counter table."""
    total = int(
        counts.get("total_queries", 0)
        or sum(value for value in counts.values() if isinstance(value, int | float))
    )
    rows = []
    for key, value in sorted(counts.items()):
        if not isinstance(value, int | float):
            continue
        pct = (float(value) / total * 100.0) if total and key != "total_queries" else 100.0
        rows.append((key, str(int(value)), f"{pct:.1f}%"))
    if not rows:
        return []
    return [title, "", *_md_aligned_table(("Metric", "Count", "Share"), "<>>", rows), ""]


def _md_phase_rows(phases: Mapping[str, Any]) -> list[str]:
    """Return a Markdown timing table for phase-shaped payloads."""
    headers = (
        "Phase",
        "Mean (ms)",
        "Median (ms)",
        "p95 (ms)",
        "p99 (ms)",
        "StdDev (ms)",
        "Memory Δ (MB)",
    )
    rows: list[tuple[str, ...]] = []
    for name, phase_data in phases.items():
        e = phase_data.get("elapsed", {})
        m = phase_data.get("memory_delta_mb", {})
        rows.append(
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
    return [*_md_aligned_table(headers, "<>", rows), ""] if rows else []


def _md_slow_queries(title: str, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return a Markdown slow-query table."""
    if not rows:
        return []
    table_rows: list[tuple[str, ...]] = [
        (
            str(item.get("language", "")),
            str(item.get("category", "")),
            f"{float(item.get('mean_sec', 0.0)) * 1000:.2f}",
            f"{float(item.get('max_sec', 0.0)) * 1000:.2f}",
            str(item.get("dynamic_candidate_count", 0)),
            str(item.get("query", "")),
        )
        for item in rows
    ]
    return [
        title,
        "",
        *_md_aligned_table(
            ("Lang", "Category", "Mean (ms)", "Max (ms)", "Dynamic", "Query"),
            "<<>>>",
            table_rows,
        ),
        "",
    ]


def _text_count_table(title: str, counts: Mapping[str, Any]) -> list[str]:
    """Return a plain-text counter table."""
    if not counts:
        return []
    total = int(
        counts.get("total_queries", 0)
        or sum(value for value in counts.values() if isinstance(value, int | float))
    )
    rows: list[tuple[str, ...]] = []
    for key, value in sorted(counts.items()):
        if not isinstance(value, int | float):
            continue
        pct = (float(value) / total * 100.0) if total and key != "total_queries" else 100.0
        rows.append((key, str(int(value)), f"{pct:.1f}%"))
    if not rows:
        return []
    hdr, sep, data = align_table(("Metric", "Count", "Share"), rows, alignments="<>>")
    return ["", f"{title}:", hdr, sep, *data]


def _text_slow_queries(title: str, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return a plain-text slow-query table."""
    if not rows:
        return []
    table_rows: list[tuple[str, ...]] = [
        (
            str(item.get("language", "")),
            str(item.get("category", "")),
            f"{float(item.get('mean_sec', 0.0)) * 1000:.3f}",
            f"{float(item.get('max_sec', 0.0)) * 1000:.3f}",
            str(item.get("dynamic_candidate_count", 0)),
            str(item.get("query", ""))[:72],
        )
        for item in rows
    ]
    headers = ("Lang", "Category", "Mean(ms)", "Max(ms)", "Dynamic", "Query")
    hdr, sep, data = align_table(headers, table_rows, alignments="<<>>>")
    return ["", f"{title}:", hdr, sep, *data]


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

    hdr_parts = [f" {h:{a}{w}} " for h, a, w in zip(headers, aligns, widths, strict=True)]
    lines: list[str] = ["|" + "|".join(hdr_parts) + "|"]
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
    rows: list[tuple[str, ...]] = []
    rows.extend(
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
        for name, s in stats.items()
        if isinstance(s, dict)
    )
    lines: list[str] = [title, ""]
    if rows:
        headers = ("Metric", "Mean", "Median", "p95", "p99", "StdDev", "Min", "Max", "CoV%")
        lines.extend(_md_aligned_table(headers, "<>", rows))
        lines.append("")
    return lines


@dataclass(frozen=True, slots=True)
class ComponentQuery:
    """Prepared query text used by isolated component benchmarks."""

    raw: str
    normalized: str
    language: str
    literal_text: str | None


@dataclass(frozen=True, slots=True)
class RuntimeQueryCase:
    """One dataset query used by runtime-path profiling."""

    query: str
    category: str


@dataclass(frozen=True, slots=True)
class RuntimeProfileContext:
    """Prepared runtime/index pair for production-path ranking profiling."""

    language: str
    runtime: Any
    index: Any
    cases: tuple[RuntimeQueryCase, ...]


@dataclass(frozen=True, slots=True)
class RankStageCase:
    """Prepared rank-path inputs for one sampled query."""

    raw: str
    normalized: str
    no_diacritics: str
    tokens: frozenset[str]
    tokens_tuple: tuple[str, ...]
    grams: frozenset[str]
    bm25_raw_scores: tuple[float, ...]
    bm25_scores: tuple[float, ...]
    char_scores: tuple[float, ...]
    prefilter_keys: tuple[float, ...]
    top_indices: tuple[int, ...]
    ranked: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class RankStageContext:
    """Prepared per-language index structures for rank-stage micro-profiling."""

    language: str
    index: Any
    candidates: tuple[Any, ...]
    bm25_index: Any
    char_index: Any
    positional_literal_tokens: frozenset[str]
    exact_normalized_lookup: Mapping[str, Sequence[Any]]
    exact_no_diacritics_lookup: Mapping[str, Sequence[Any]]
    wildcard_always_passes: frozenset[int]
    wildcard_variant_analyses: Mapping[int, tuple[WildcardVariantAnalysis, ...]]
    wildcard_variant_groups: tuple[WildcardVariantGroup, ...]
    candidate_slot_tokens: tuple[frozenset[str], ...]
    prefilter_limit: int
    cases: tuple[RankStageCase, ...]


def _sample_rank_stage_queries(
    queries: Sequence[str],
    sample_size: int = RANK_STAGE_QUERY_SAMPLE_SIZE,
) -> tuple[str, ...]:
    """Return evenly spaced unique queries for expensive rank-stage probes."""
    if sample_size < 1 or not queries:
        return ()
    if len(queries) <= sample_size:
        return tuple(dict.fromkeys(queries))
    if sample_size == 1:
        return (queries[0],)
    last_index = len(queries) - 1
    positions = {
        round(position * last_index / (sample_size - 1)) for position in range(sample_size)
    }
    sampled = [queries[index] for index in sorted(positions)]
    return tuple(dict.fromkeys(sampled))


def _is_rank_short_circuit_query(index: Any, query: str, language: str) -> bool:
    """Return whether a query would bypass the fuzzy rank hot path."""
    normalized = normalize_text(query)
    return (
        _exact_lookup_ranked(
            query,
            normalized,
            DEFAULT_MAX_CANDIDATES,
            index._exact_normalized_lookup,
            index._exact_no_diacritics_lookup,
            language,
        )
        is not None
    )


def _build_rank_stage_case(
    context: RankStageContext,
    query: str,
) -> RankStageCase:
    """Build reusable inputs for one sampled rank-stage query."""
    normalized = normalize_text(query)
    no_diacritics = normalize_text_no_diacritics(query, context.language)
    tokens_tuple = tuple(normalized.split())
    tokens = frozenset(tokens_tuple)
    grams = char_ngrams_normalized(normalized)
    doc_count = len(context.candidates)
    raw_scores = (
        tuple(context.bm25_index.raw_scores(tokens_tuple)) if tokens_tuple else (0.0,) * doc_count
    )
    bm25_scores = _normalized_bm25_scores_from_raw(raw_scores, doc_count)
    char_scores = tuple(context.char_index.score(grams))
    prefilter_keys = tuple(_rank_prefilter_keys(char_scores, bm25_scores))
    top_indices = tuple(_top_prefilter_indices(prefilter_keys, context.prefilter_limit))
    ranked = tuple(context.index.rank(query))
    return RankStageCase(
        raw=query,
        normalized=normalized,
        no_diacritics=no_diacritics,
        tokens=tokens,
        tokens_tuple=tokens_tuple,
        grams=grams,
        bm25_raw_scores=raw_scores,
        bm25_scores=bm25_scores,
        char_scores=char_scores,
        prefilter_keys=prefilter_keys,
        top_indices=top_indices,
        ranked=ranked,
    )


def _build_rank_stage_context(
    language: str,
    index: Any,
    raw_queries: Sequence[str],
) -> RankStageContext | None:
    """Build per-language rank-stage context from production index structures."""
    candidates = tuple(index.candidates)
    if not candidates or not raw_queries:
        return None
    fuzzy_queries = [
        query for query in raw_queries if not _is_rank_short_circuit_query(index, query, language)
    ]
    sampled_queries = _sample_rank_stage_queries(fuzzy_queries or raw_queries)
    if not sampled_queries:
        return None

    context = RankStageContext(
        language=language,
        index=index,
        candidates=candidates,
        bm25_index=index._bm25_index,
        char_index=index._candidate_char_index,
        positional_literal_tokens=index._positional_literal_tokens,
        exact_normalized_lookup=index._exact_normalized_lookup,
        exact_no_diacritics_lookup=index._exact_no_diacritics_lookup,
        wildcard_always_passes=index._wildcard_always_passes,
        wildcard_variant_analyses=index._wildcard_variant_analyses,
        wildcard_variant_groups=index._wildcard_variant_groups,
        candidate_slot_tokens=index._candidate_slot_tokens,
        prefilter_limit=_rank_prefilter_limit(len(candidates)),
        cases=(),
    )
    cases = tuple(_build_rank_stage_case(context, query) for query in sampled_queries)
    return RankStageContext(
        language=context.language,
        index=context.index,
        candidates=context.candidates,
        bm25_index=context.bm25_index,
        char_index=context.char_index,
        positional_literal_tokens=context.positional_literal_tokens,
        exact_normalized_lookup=context.exact_normalized_lookup,
        exact_no_diacritics_lookup=context.exact_no_diacritics_lookup,
        wildcard_always_passes=context.wildcard_always_passes,
        wildcard_variant_analyses=context.wildcard_variant_analyses,
        wildcard_variant_groups=context.wildcard_variant_groups,
        candidate_slot_tokens=context.candidate_slot_tokens,
        prefilter_limit=context.prefilter_limit,
        cases=cases,
    )


def _record_component_elapsed(
    component_results: dict[str, dict[str, list[float]]],
    name: str,
    start_time: float,
    item_count: int,
) -> None:
    """Record elapsed seconds per benchmarked item for one component."""
    if item_count > 0:
        component_results[name]["elapsed"].append((time.perf_counter() - start_time) / item_count)


def _new_runtime_coverage_counts() -> dict[str, int]:
    """Return zeroed branch counters for runtime profiling."""
    return dict.fromkeys(RUNTIME_COVERAGE_KEYS, 0)


def _merge_runtime_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    """Merge runtime branch counters into *target*."""
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _runtime_ranked_has_wildcard(ranked: Sequence[Any]) -> bool:
    """Return whether ranked results include wildcard rehydration evidence."""
    return any(
        bool(getattr(item.candidate, "has_wildcard", False))
        or float(getattr(item.scores, "penalty", 0.0)) > 0.0
        for item in ranked
    )


def _runtime_scenario_tags(
    *,
    category: str,
    static_perfect: bool,
    dynamic_candidate_count: int,
    dynamic_perfect: bool,
    accepted: bool,
    wildcard_result: bool,
    empty_result: bool,
) -> tuple[str, ...]:
    """Return stable scenario labels for one runtime query observation."""
    tags = [f"category:{category}"]
    if static_perfect:
        tags.append("static_perfect_short_circuit")
    else:
        tags.append("dynamic_attempted")
        dynamic_tag = (
            "dynamic_candidates" if dynamic_candidate_count > 0 else "no_dynamic_candidates"
        )
        tags.append(dynamic_tag)
    if dynamic_perfect:
        tags.append("dynamic_perfect")
    elif dynamic_candidate_count > 0:
        tags.append("merged_dynamic")
    tags.append("accepted" if accepted else "rejected")
    if wildcard_result:
        tags.append("wildcard_result")
    if empty_result:
        tags.append("empty_result")
    return tuple(dict.fromkeys(tags))


def _record_runtime_coverage(counts: dict[str, int], tags: Sequence[str]) -> None:
    """Record one query's runtime branch coverage from scenario tags."""
    counts["total_queries"] = counts.get("total_queries", 0) + 1
    for tag in tags:
        if tag in counts and tag != "total_queries":
            counts[tag] += 1


def _runtime_query_observation(
    case: RuntimeQueryCase,
    ranked: Sequence[Any],
    dynamic_candidate_count: int,
) -> dict[str, Any]:
    """Return branch and result metadata for one runtime query."""
    static_perfect = dynamic_candidate_count == 0 and bool(_is_perfect_rank_result(tuple(ranked)))
    accepted = _accepted_candidate(ranked, min_confidence=DEFAULT_MIN_CONFIDENCE) is not None
    dynamic_perfect = dynamic_candidate_count > 0 and bool(_is_perfect_rank_result(tuple(ranked)))
    tags = _runtime_scenario_tags(
        category=case.category,
        static_perfect=static_perfect,
        dynamic_candidate_count=dynamic_candidate_count,
        dynamic_perfect=dynamic_perfect,
        accepted=accepted,
        wildcard_result=_runtime_ranked_has_wildcard(ranked),
        empty_result=not ranked,
    )
    return {
        "tags": tags,
        "dynamic_candidate_count": dynamic_candidate_count,
        "accepted": accepted,
        "result_count": len(ranked),
        "top_score": ranked[0].scores.final_score if ranked else None,
    }


def _stats_payload(values_by_name: Mapping[str, list[float]]) -> dict[str, Any]:
    """Return StatsEngine payloads for named timing samples."""
    return {
        name: {"elapsed": StatsEngine.as_dict(StatsEngine.compute(values))}
        for name, values in sorted(values_by_name.items())
        if values
    }


def _runtime_slow_query_payload(
    values_by_key: Mapping[tuple[str, str, str], list[float]],
    meta_by_key: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    limit: int = RUNTIME_SLOW_QUERY_LIMIT,
) -> list[dict[str, Any]]:
    """Return the slowest runtime queries ordered by mean elapsed time."""
    rows: list[dict[str, Any]] = []
    for key, values in values_by_key.items():
        if not values:
            continue
        language, category, query = key
        stats = StatsEngine.compute(values)
        meta = meta_by_key.get(key, {})
        rows.append(
            {
                "language": language,
                "category": category,
                "query": query,
                "mean_sec": round(stats.mean, 6),
                "max_sec": round(stats.max_val, 6),
                "samples": len(values),
                "dynamic_candidate_count": meta.get("dynamic_candidate_count", 0),
                "accepted": bool(meta.get("accepted", False)),
                "top_score": meta.get("top_score"),
                "tags": list(meta.get("tags", ())),
            }
        )
    rows.sort(key=lambda item: (float(item["mean_sec"]), float(item["max_sec"])), reverse=True)
    return rows[:limit]


def _build_runtime_profile_context(
    language: str,
    data: Mapping[str, Any],
) -> RuntimeProfileContext | None:
    """Build a runtime profile context from dataset grammar and registry slots."""
    raw_cases = data.get("test_cases", [])
    if not isinstance(raw_cases, list):
        return None
    cases = tuple(
        RuntimeQueryCase(query=case["query"], category=case["category"])
        for case in raw_cases
        if isinstance(case, dict)
        and isinstance(case.get("query"), str)
        and isinstance(case.get("category"), str)
    )
    if not cases:
        return None

    slots = _dataset_registry_slots(data, language)
    sources = load_language_intent_sources(language)
    candidates = build_candidates_from_intent_sources(language, sources, slots)
    index = build_index(language, candidates)
    runtime = CanonicalizerRuntime()
    runtime.update_registry_slot_values(slots)
    runtime.language_intent_sources[language] = sources
    runtime.set_index(index)
    return RuntimeProfileContext(language=language, runtime=runtime, index=index, cases=cases)


def _iter_rank_stage_cases(
    contexts: Sequence[RankStageContext],
) -> tuple[tuple[RankStageContext, RankStageCase], ...]:
    """Return flattened rank-stage context/case pairs."""
    return tuple((context, case) for context in contexts for case in context.cases)


def _warm_rank_stage_components(contexts: Sequence[RankStageContext]) -> None:
    """Warm caches for rank-stage micro-profiling inputs."""
    for context, case in _iter_rank_stage_cases(contexts):
        _ = context.index.rank(case.raw)
        _ = normalize_text(case.raw)
        _ = context.exact_normalized_lookup.get(case.normalized)
        _ = context.exact_no_diacritics_lookup.get(case.no_diacritics)
        _ = context.bm25_index.raw_scores(case.tokens_tuple)
        _ = _normalized_bm25_scores_from_raw(case.bm25_raw_scores, len(context.candidates))
        _ = context.char_index.score(case.grams)
        _ = _rank_prefilter_keys(case.char_scores, case.bm25_scores)
        _ = _top_prefilter_indices(case.prefilter_keys, context.prefilter_limit)
        _ = _prefilter_wildcard_candidates(
            context.candidates,
            case.tokens,
            context.wildcard_always_passes,
            wildcard_variant_groups=context.wildcard_variant_groups,
        )
        _ = _build_positional_lookup(context.positional_literal_tokens, case.tokens)
        _ = _query_slot_tokens_from_candidates(
            case.tokens,
            case.top_indices,
            context.candidate_slot_tokens,
            context.positional_literal_tokens,
        )
        _ = _accepted_candidate(case.ranked)


def _profile_rank_stage_components(
    component_results: dict[str, dict[str, list[float]]],
    contexts: Sequence[RankStageContext],
) -> None:
    """Profile rank-path orchestration stages that are not simple scoring functions."""
    case_pairs = _iter_rank_stage_cases(contexts)
    case_count = len(case_pairs)
    if case_count == 0:
        return

    t0 = time.perf_counter()
    for context, case in case_pairs:
        _ = context.index.rank(case.raw)
    _record_component_elapsed(component_results, "rank_full", t0, case_count)

    t0 = time.perf_counter()
    for context, case in case_pairs:
        normalized = normalize_text(case.raw)
        _ = _rank_query_setup(normalized, context.positional_literal_tokens)
    _record_component_elapsed(component_results, "rank_query_setup", t0, case_count)

    t0 = time.perf_counter()
    for context, case in case_pairs:
        _ = _exact_lookup_ranked(
            case.raw,
            case.normalized,
            DEFAULT_MAX_CANDIDATES,
            context.exact_normalized_lookup,
            context.exact_no_diacritics_lookup,
            context.language,
        )
    _record_component_elapsed(component_results, "rank_exact_lookup", t0, case_count)

    t0 = time.perf_counter()
    for context, case in case_pairs:
        raw_scores = context.bm25_index.raw_scores(case.tokens_tuple)
        _ = max(raw_scores, default=0.0)
    _record_component_elapsed(component_results, "rank_bm25_raw_scores", t0, case_count)

    t0 = time.perf_counter()
    for context, case in case_pairs:
        _ = _normalized_bm25_scores_from_raw(case.bm25_raw_scores, len(context.candidates))
    _record_component_elapsed(component_results, "rank_bm25_normalize_scores", t0, case_count)

    t0 = time.perf_counter()
    for context, case in case_pairs:
        _ = context.char_index.score(case.grams)
    _record_component_elapsed(component_results, "rank_char_ngram_score", t0, case_count)

    t0 = time.perf_counter()
    for _, case in case_pairs:
        _ = _rank_prefilter_keys(case.char_scores, case.bm25_scores)
    _record_component_elapsed(component_results, "prefilter_key_build", t0, case_count)

    t0 = time.perf_counter()
    for context, case in case_pairs:
        _ = _top_prefilter_indices(case.prefilter_keys, context.prefilter_limit)
    _record_component_elapsed(component_results, "prefilter_top_indices", t0, case_count)

    t0 = time.perf_counter()
    for context, case in case_pairs:
        _ = _prefilter_wildcard_candidates(
            context.candidates,
            case.tokens,
            context.wildcard_always_passes,
            wildcard_variant_groups=context.wildcard_variant_groups,
        )
    _record_component_elapsed(component_results, "wildcard_prefilter", t0, case_count)

    t0 = time.perf_counter()
    for context, case in case_pairs:
        _ = _build_positional_lookup(context.positional_literal_tokens, case.tokens)
    _record_component_elapsed(component_results, "positional_lookup_build", t0, case_count)

    t0 = time.perf_counter()
    for context, case in case_pairs:
        _ = _query_slot_tokens_from_candidates(
            case.tokens,
            case.top_indices,
            context.candidate_slot_tokens,
            context.positional_literal_tokens,
        )
    _record_component_elapsed(component_results, "query_slot_token_filter", t0, case_count)

    t0 = time.perf_counter()
    for _, case in case_pairs:
        _ = _accepted_candidate(case.ranked)
    _record_component_elapsed(component_results, "accepted_candidate", t0, case_count)


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
        timer.enabled = not is_warmup

        if is_warmup:
            print(f"Warmup run {run_idx + 1}/{warmup} ...", flush=True)
        else:
            print(f"Profiling run {run_idx - warmup + 1}/{iterations} ...", flush=True)

        gc.collect()
        monitor.snapshot_gc()
        timer.start(label)

        if granularity == "coarse" or is_warmup:
            asyncio.run(
                run_evaluation(
                    datasets=dict(datasets),
                    failure_limit=0,
                    output_json=None if is_warmup else json_path,
                    output_md=None if is_warmup else md_path,
                    min_intent_slot_accuracy=None,
                    max_fallback_rate=None,
                )
            )
        else:
            with timer.phase("evaluate_total"):
                asyncio.run(
                    run_evaluation(
                        datasets=dict(datasets),
                        failure_limit=0,
                        output_json=None if is_warmup else json_path,
                        output_md=None if is_warmup else md_path,
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
            timer.enabled = not is_warmup

            sources = load_language_intent_sources(lang)
            gc.collect()
            monitor.snapshot_gc()

            timer.start(label)
            with timer.phase(f"build_candidates_{lang}"):
                candidates = build_candidates_from_intent_sources(lang, sources, slots)
            with timer.phase(f"build_index_only_{lang}"):
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


def _profile_rank_for_language(
    lang: str,
    index: Any,
    queries: list[str],
    total_runs: int,
    warmup: int,
    granularity: str,
    timer: PhaseTimer,
    monitor: ResourceMonitor,
    all_elapsed: list[float],
) -> dict[str, Any] | None:
    """Run ranking profiling for a single language and return its aggregate metrics."""
    lang_elapsed: list[float] = []
    per_query_times: list[list[float]] = [[] for _ in queries]

    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup
        label = f"rank_{lang}"
        timer.enabled = not is_warmup

        gc.collect()
        monitor.snapshot_gc()

        timer.start(label)
        for qi, query in enumerate(queries):
            q_start = time.perf_counter()
            if granularity != "fine":
                _ = index.rank(query)
            else:
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
    avg_query_times.extend(statistics.mean(q_times) for q_times in per_query_times if q_times)
    if lang_elapsed:
        rank_agg: dict[str, Any] = {
            "rank_total_wall_time_sec": StatsEngine.as_dict(StatsEngine.compute(lang_elapsed))
        }
        if avg_query_times:
            rank_agg["rank_per_query_wall_time_sec"] = StatsEngine.as_dict(
                StatsEngine.compute(avg_query_times)
            )
            rank_agg["query_count"] = len(queries)
        return {"aggregate": rank_agg}
    return None


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

        if lang_metrics := _profile_rank_for_language(
            lang,
            index,
            queries,
            total_runs,
            warmup,
            granularity,
            timer,
            monitor,
            all_elapsed,
        ):
            per_language[lang] = lang_metrics

    result: dict[str, Any] = {"per_language": per_language}
    if all_elapsed:
        result["aggregate"] = {
            "rank_wall_time_sec_total": StatsEngine.as_dict(StatsEngine.compute(all_elapsed))
        }
    return result


def _profile_runtime_for_language(
    context: RuntimeProfileContext,
    total_runs: int,
    warmup: int,
    granularity: str,
    timer: PhaseTimer,
    monitor: ResourceMonitor,
    all_elapsed: list[float],
) -> dict[str, Any] | None:
    """Run production-path runtime profiling for a single language."""
    lang_elapsed: list[float] = []
    per_query_times: list[list[float]] = [[] for _ in context.cases]
    scenario_times: dict[str, list[float]] = {}
    coverage = _new_runtime_coverage_counts()
    category_coverage: dict[str, int] = {}
    slow_values: dict[tuple[str, str, str], list[float]] = {}
    slow_meta: dict[tuple[str, str, str], dict[str, Any]] = {}

    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup
        is_first_measured_run = run_idx == warmup
        label = f"runtime_{context.language}"
        timer.enabled = not is_warmup

        gc.collect()
        monitor.snapshot_gc()

        timer.start(label)
        for query_index, case in enumerate(context.cases):
            start = time.perf_counter()
            if granularity == "fine":
                with timer.phase("runtime_rank_with_dynamic_candidates_inner"):
                    ranked = context.runtime.rank_with_dynamic_candidates(
                        context.language,
                        context.index,
                        case.query,
                        DEFAULT_MAX_CANDIDATES,
                        min_confidence=DEFAULT_MIN_CONFIDENCE,
                    )
            else:
                ranked = context.runtime.rank_with_dynamic_candidates(
                    context.language,
                    context.index,
                    case.query,
                    DEFAULT_MAX_CANDIDATES,
                    min_confidence=DEFAULT_MIN_CONFIDENCE,
                )
            elapsed = time.perf_counter() - start
            if is_warmup:
                continue

            per_query_times[query_index].append(elapsed)
            dynamic_count = int(context.runtime.diagnostics.dynamic_candidate_count)
            observation = _runtime_query_observation(
                case,
                ranked,
                dynamic_count,
            )
            for tag in observation["tags"]:
                scenario_times.setdefault(tag, []).append(elapsed)
            slow_key = (context.language, case.category, case.query)
            slow_values.setdefault(slow_key, []).append(elapsed)
            slow_meta.setdefault(slow_key, observation)
            if is_first_measured_run:
                _record_runtime_coverage(coverage, observation["tags"])
                category_coverage[case.category] = category_coverage.get(case.category, 0) + 1
        timer.stop()

        if not is_warmup and label in timer.phases:
            lang_elapsed.append(timer.phases[label][-1])
            all_elapsed.append(timer.phases[label][-1])

    avg_query_times = [
        statistics.mean(query_times) for query_times in per_query_times if query_times
    ]
    if not lang_elapsed:
        return None

    runtime_agg: dict[str, Any] = {
        "runtime_total_wall_time_sec": StatsEngine.as_dict(StatsEngine.compute(lang_elapsed)),
        "query_count": len(context.cases),
    }
    if avg_query_times:
        runtime_agg["runtime_per_query_wall_time_sec"] = StatsEngine.as_dict(
            StatsEngine.compute(avg_query_times)
        )

    return {
        "aggregate": runtime_agg,
        "coverage": coverage,
        "category_coverage": dict(sorted(category_coverage.items())),
        "scenario_stats": _stats_payload(scenario_times),
        "slow_queries": _runtime_slow_query_payload(slow_values, slow_meta),
        "_scenario_samples": scenario_times,
        "_slow_query_values": slow_values,
        "_slow_query_meta": slow_meta,
    }


def _profile_runtime(
    datasets: dict[str, str],
    iterations: int,
    warmup: int,
    granularity: str,
    timer: PhaseTimer,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    """Profile production-path runtime ranking with dynamic candidates."""
    _bootstrap_project_imports()
    total_runs = warmup + iterations
    per_language: dict[str, dict[str, Any]] = {}
    all_elapsed: list[float] = []
    all_scenario_times: dict[str, list[float]] = {}
    all_coverage = _new_runtime_coverage_counts()
    all_category_coverage: dict[str, int] = {}
    all_slow_values: dict[tuple[str, str, str], list[float]] = {}
    all_slow_meta: dict[tuple[str, str, str], dict[str, Any]] = {}

    for lang, path in sorted(datasets.items()):
        try:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        context = _build_runtime_profile_context(lang, data)
        if context is None:
            continue

        lang_metrics = _profile_runtime_for_language(
            context,
            total_runs,
            warmup,
            granularity,
            timer,
            monitor,
            all_elapsed,
        )
        if lang_metrics is None:
            continue
        scenario_samples = lang_metrics.pop("_scenario_samples", {})
        slow_query_values = lang_metrics.pop("_slow_query_values", {})
        slow_query_meta = lang_metrics.pop("_slow_query_meta", {})
        per_language[lang] = lang_metrics
        _merge_runtime_counts(all_coverage, lang_metrics.get("coverage", {}))
        for category, count in lang_metrics.get("category_coverage", {}).items():
            all_category_coverage[category] = all_category_coverage.get(category, 0) + int(count)
        for scenario, values in scenario_samples.items():
            all_scenario_times.setdefault(scenario, []).extend(values)
        for key, values in slow_query_values.items():
            all_slow_values.setdefault(key, []).extend(values)
            if key in slow_query_meta:
                all_slow_meta[key] = slow_query_meta[key]

    result: dict[str, Any] = {"per_language": per_language}
    if all_elapsed:
        result["aggregate"] = {
            "runtime_wall_time_sec_total": StatsEngine.as_dict(StatsEngine.compute(all_elapsed))
        }
    if all_scenario_times:
        result["scenario_stats"] = _stats_payload(all_scenario_times)
    if all_coverage.get("total_queries", 0):
        result["coverage"] = all_coverage
    if all_category_coverage:
        result["category_coverage"] = dict(sorted(all_category_coverage.items()))
    if all_slow_values:
        result["slow_queries"] = _runtime_slow_query_payload(all_slow_values, all_slow_meta)
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

    all_queries: list[ComponentQuery] = []
    all_norm_texts: list[str] = []
    all_candidates: list[Any] = []
    rank_contexts: list[RankStageContext] = []

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
        indexed_candidates = tuple(index.candidates)
        all_candidates.extend(indexed_candidates)
        all_norm_texts.extend(candidate.normalized_text for candidate in indexed_candidates)
        raw_cases = data.get("test_cases", [])
        if not isinstance(raw_cases, list):
            continue

        raw_case_queries: list[str] = []
        lang_entries: list[ComponentQuery] = []
        for case in raw_cases:
            if not isinstance(case, dict):
                continue
            query = case.get("query")
            if not isinstance(query, str):
                continue
            norm = normalize_text(query)
            raw_case_queries.append(query)
            lang_entries.append(ComponentQuery(query, norm, lang, None))

        for candidate in indexed_candidates[:50]:
            raw_text = candidate.text
            norm_text = candidate.normalized_text
            lit_text = candidate.metadata.get("literal_text") if candidate.metadata else None
            lang_entries.append(ComponentQuery(raw_text, norm_text, lang, lit_text))

        all_queries.extend(lang_entries)
        if rank_context := _build_rank_stage_context(lang, index, raw_case_queries):
            rank_contexts.append(rank_context)

    total_runs = warmup + iterations
    component_results: dict[str, dict[str, list[float]]] = {
        comp: {"elapsed": []} for comp in SCORING_COMPONENT_NAMES
    }

    _disambig_pair: tuple[Any, Any] | None = None
    if len(all_candidates) >= 2:
        for _i in range(len(all_candidates)):
            for _j in range(_i + 1, len(all_candidates)):
                if all_candidates[_i].intent_name != all_candidates[_j].intent_name:
                    _disambig_pair = (all_candidates[_i], all_candidates[_j])
                    break
            if _disambig_pair is not None:
                break
        if _disambig_pair is None:
            _disambig_pair = (all_candidates[0], all_candidates[1])

    _rapidfuzz_queries: list[tuple[str, str, int]] | None = None
    _rapidfuzz_cand_sorted: str | None = None
    _rapidfuzz_cand_tokens: int | None = None
    if all_norm_texts:
        _cand_norm = all_norm_texts[0]
        _rapidfuzz_cand_sorted = " ".join(sorted(_cand_norm.split()))
        _rapidfuzz_cand_tokens = _cand_norm.count(" ") + 1
        _rapidfuzz_queries = [
            (
                query.normalized,
                " ".join(sorted(query.normalized.split())),
                query.normalized.count(" ") + 1,
            )
            for query in all_queries
        ]

    if not all_queries:
        return {}

    bm25_idx = BM25Index.from_normalized_texts(all_norm_texts) if all_norm_texts else None
    char_grams = [char_ngrams_normalized(t) for t in all_norm_texts]
    char_idx = CharNGramIndex.from_grams(char_grams) if char_grams else None

    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup
        timer.enabled = not is_warmup
        if is_warmup:
            for query in all_queries:
                _ = normalize_text(query.raw)
                _ = normalize_text_no_diacritics(query.raw, query.language)
                _ = char_ngrams_normalized(query.normalized)
                if bm25_idx is not None:
                    _ = bm25_idx.score(query.normalized)
                q_grams = char_ngrams_normalized(query.normalized)
                if char_idx is not None:
                    _ = char_idx.score(q_grams)
                if all_norm_texts:
                    _ = rapidfuzz_similarity_normalized(query.normalized, all_norm_texts[0])
                q_tokens = frozenset(query.normalized.split())
                if query.literal_text:
                    _ = _exact_intent_score(query.literal_text, q_tokens)
                _ = lexical_score(0.5, 0.5, 0.5, 0.5)
            if all_candidates and len(all_candidates) >= 2:
                assert _disambig_pair is not None
                fake_scores_a = _ScoreBreakdown(0.8, 0.8, 0.8, 0.9, 0.85)
                fake_scores_b = _ScoreBreakdown(0.8, 0.8, 0.8, 0.95, 0.84)
                fake_ranked = [
                    _RankedCandidate(candidate=_disambig_pair[0], scores=fake_scores_a),
                    _RankedCandidate(candidate=_disambig_pair[1], scores=fake_scores_b),
                ]
                _apply_intent_disambiguation(fake_ranked)
            _warm_rank_stage_components(rank_contexts)
            continue

        # 1. normalize_text
        t0 = time.perf_counter()
        for query in all_queries:
            _ = normalize_text(query.raw)
        _record_component_elapsed(component_results, "normalize_text", t0, len(all_queries))

        # 2. normalize_text_no_diacritics
        t0 = time.perf_counter()
        for query in all_queries:
            _ = normalize_text_no_diacritics(query.raw, query.language)
        _record_component_elapsed(
            component_results, "normalize_text_no_diacritics", t0, len(all_queries)
        )

        # 3. char_ngrams_normalized
        t0 = time.perf_counter()
        for query in all_queries:
            _ = char_ngrams_normalized(query.normalized)
        _record_component_elapsed(component_results, "char_ngrams_normalized", t0, len(all_queries))

        # 4. bm25_score
        if bm25_idx is not None:
            t0 = time.perf_counter()
            for query in all_queries:
                _ = bm25_idx.score(query.normalized)
            _record_component_elapsed(component_results, "bm25_score", t0, len(all_queries))

        # 5. char_ngram_score
        if char_idx is not None:
            q_grams_list = [char_ngrams_normalized(query.normalized) for query in all_queries]
            t0 = time.perf_counter()
            for qg in q_grams_list:
                _ = char_idx.score(qg)
            _record_component_elapsed(component_results, "char_ngram_score", t0, len(all_queries))

        # 6. rapidfuzz_similarity
        if all_norm_texts:
            assert _rapidfuzz_queries is not None
            t0 = time.perf_counter()
            for qn, qs, qt_cnt in _rapidfuzz_queries:
                _ = rapidfuzz_similarity_normalized(
                    qn,
                    all_norm_texts[0],
                    query_token_count=qt_cnt,
                    query_sorted=qs,
                    candidate_sorted=_rapidfuzz_cand_sorted,
                    candidate_token_count=_rapidfuzz_cand_tokens,
                )
            component_results["rapidfuzz_similarity"]["elapsed"].append(
                (time.perf_counter() - t0) / len(all_queries)
            )

        # 7. exact_intent_score
        lit_queries = [
            (query.normalized, query.literal_text) for query in all_queries if query.literal_text
        ]
        if lit_queries:
            lit_queries_variants = [
                (literal_token_variants(lt), frozenset(qn.split())) for qn, lt in lit_queries
            ]
            t0 = time.perf_counter()
            for variants, qt in lit_queries_variants:
                _ = _exact_intent_score(variants, qt)
            component_results["exact_intent_score"]["elapsed"].append(
                (time.perf_counter() - t0) / len(lit_queries_variants)
            )

        # 8. positional_intent_score
        if lit_queries:
            pos_lookups = []
            for qn, lt in lit_queries:
                qt = frozenset(qn.split())
                variants = literal_token_variants(lt)
                all_lit = frozenset().union(*variants) if variants else frozenset()
                lookup = _build_positional_lookup(all_lit, qt)
                pos_lookups.append((lt, qt, lookup))
            t0 = time.perf_counter()
            for lt, qt, lookup in pos_lookups:
                _ = _positional_intent_score_from_lookup(lt, qt, lookup, None)
            component_results["positional_intent_score"]["elapsed"].append(
                (time.perf_counter() - t0) / len(lit_queries)
            )

        # 9. lexical_score
        t0 = time.perf_counter()
        for _ in range(len(all_queries)):
            _ = lexical_score(0.5, 0.5, 0.5, 0.5)
        component_results["lexical_score"]["elapsed"].append(
            (time.perf_counter() - t0) / len(all_queries)
        )

        # 10. intent_disambiguation
        if all_candidates and len(all_candidates) >= 2:
            assert _disambig_pair is not None
            fake_scores_a = _ScoreBreakdown(0.8, 0.8, 0.8, 0.9, 0.85)
            fake_scores_b = _ScoreBreakdown(0.8, 0.8, 0.8, 0.95, 0.84)
            fake_ranked_runs = [
                [
                    _RankedCandidate(candidate=_disambig_pair[0], scores=fake_scores_a),
                    _RankedCandidate(candidate=_disambig_pair[1], scores=fake_scores_b),
                ]
                for _ in range(len(all_queries))
            ]
            t0 = time.perf_counter()
            for run_list in fake_ranked_runs:
                _apply_intent_disambiguation(run_list)
            component_results["intent_disambiguation"]["elapsed"].append(
                (time.perf_counter() - t0) / len(all_queries)
            )

        _profile_rank_stage_components(component_results, rank_contexts)

    result: dict[str, Any] = {}
    for comp_name in SCORING_COMPONENT_NAMES:
        if vals := component_results[comp_name]["elapsed"]:
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
        "report_schema": "assist_canonicalizer_performance",
        "report_schema_version": PERFORMANCE_REPORT_SCHEMA_VERSION,
        "target": target,
        "iterations": iterations,
        "warmup": warmup,
        "granularity": granularity,
        "languages": languages or [],
        "dependency_versions": _benchmark_dependency_versions(),
    }

    raw_phase_stats = timer.stats()
    phase_stats: dict[str, dict[str, dict[str, Any]]] = {
        phase_name: {
            metric_name: (StatsEngine.as_dict(sr) if isinstance(sr, StatsResult) else sr)
            for metric_name, sr in metrics.items()
        }
        for phase_name, metrics in raw_phase_stats.items()
    }
    if phase_stats:
        report["phases"] = phase_stats

    if agg := target_result.get("aggregate", {}):
        report["aggregate"] = agg

    if comps := target_result.get("components", {}):
        report["components"] = comps

    if per_lang := target_result.get("per_language", {}):
        report["per_language"] = per_lang

    if coverage := target_result.get("coverage", {}):
        report["coverage"] = coverage

    if category_coverage := target_result.get("category_coverage", {}):
        report["category_coverage"] = category_coverage

    if scenario_stats := target_result.get("scenario_stats", {}):
        report["scenario_stats"] = scenario_stats

    if slow_queries := target_result.get("slow_queries", []):
        report["slow_queries"] = slow_queries

    if monitor.cpu_samples and monitor.rss_samples:
        cpu_metrics = monitor.get_cpu_metrics()
        mem_metrics = monitor.get_memory_metrics()
        report["resource"] = {**cpu_metrics, **mem_metrics}

    regressions = baseline.compare(
        target, report, max_regression_pct, warn_on_missing=warn_on_missing
    )
    report["regressions"] = regressions

    stability_min_duration_sec = 0.010
    cov_values: list[float] = []
    for phase_data in phase_stats.values():
        e = phase_data.get("elapsed")
        if (
            isinstance(e, dict)
            and e.get("cov_pct", 0) > 0
            and float(e.get("mean", 0.0)) >= stability_min_duration_sec
        ):
            cov_values.append(float(e["cov_pct"]))
    if cov_values:
        avg_cov = statistics.mean(cov_values)
        if avg_cov < 5.0:
            report["stability"] = f"High stability - average CoV {avg_cov:.1f}% across phases"
        elif avg_cov < 15.0:
            report["stability"] = f"Moderate stability - average CoV {avg_cov:.1f}% across phases"
        else:
            report["stability"] = (
                f"Low stability - average CoV {avg_cov:.1f}% across phases. "
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

    monitor = ResourceMonitor(interval=0.1)
    timer = PhaseTimer(monitor)

    if target != "components":
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
        elif target == "runtime":
            target_result = _profile_runtime(
                datasets, iterations, warmup, granularity, timer, monitor
            )
        elif target == "components":
            target_result = _profile_components(datasets, iterations, warmup, timer, monitor)
            target_result = {"components": target_result}
        else:
            print(f"Unknown target: {target}", file=sys.stderr)
            return {}
    finally:
        gc.enable()
        monitor.snapshot_gc()
        if target != "components":
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
    dependency_versions = _first_report_dependency_versions(all_reports)
    lines: list[str] = [
        "# Assist Canonicalizer - Consolidated Performance Profile (All Targets)",
        "",
        "This report aggregates performance statistics across all measured profiling targets.",
        "",
        *(
            f"**Report schema:** v{PERFORMANCE_REPORT_SCHEMA_VERSION}",
            "",
            f"**Dependency versions:** {_format_dependency_versions(dependency_versions)}",
            "",
        ),
    ]
    for target, report in all_reports.items():
        lines.extend((f"## Target: `{target}`", ""))
        if agg := report.get("aggregate", {}):
            lines.extend(_md_stat_table(f"### Aggregate Performance ({target})", agg))

        if res := report.get("resource", {}):
            _resource_utilization_markdown(lines, target, res)
        if phases := report.get("phases", {}):
            lines.extend((f"### Phase Timing ({target})", ""))
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

        if components := report.get("components", {}):
            lines.extend((f"### Component Micro-Profile ({target})", ""))
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

        if coverage := report.get("coverage", {}):
            lines.extend(_md_count_table(f"### Runtime Branch Coverage ({target})", coverage))

        if category_coverage := report.get("category_coverage", {}):
            lines.extend(
                _md_count_table(f"### Runtime Category Coverage ({target})", category_coverage)
            )

        if scenario_stats := report.get("scenario_stats", {}):
            lines.extend((f"### Runtime Scenario Timing ({target})", ""))
            lines.extend(_md_phase_rows(scenario_stats))

        if slow_queries := report.get("slow_queries", []):
            lines.extend(_md_slow_queries(f"### Slowest Runtime Queries ({target})", slow_queries))

        if regressions := report.get("regressions", []):
            lines.extend((f"### Regression Detections ({target})", ""))
            lines.extend(f"- {r}" for r in regressions)
            lines.append("")

        lines.extend(("---", ""))
    while lines and lines[-1] == "":
        lines.pop()
    atomic_write(path, "\n".join(lines) + "\n")


def _resource_utilization_markdown(lines, target, res):
    """Print resource utilization metrics for the target system."""
    lines.append(f"### Resource Utilization ({target})")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| :--- | :--- |")
    lines.extend(f"| {k} | {v:.2f} |" for k, v in sorted(res.items()))
    lines.append("")


def _write_profile_all_text(all_reports: dict[str, Any], path: str) -> None:
    """Generate and write a consolidated plain text report for all profiling targets."""
    dependency_versions = _format_dependency_versions(
        _first_report_dependency_versions(all_reports)
    )
    lines: list[str] = [
        "ALGORITHMIC PERFORMANCE PROFILING REPORT (ALL TARGETS)",
        "=" * 90,
        f"Report Schema: v{PERFORMANCE_REPORT_SCHEMA_VERSION}",
        f"Dependency Versions: {dependency_versions}",
        "",
    ]
    for target, report in all_reports.items():
        lines.extend((f"Target: {target}", "-" * 90))
        if agg := report.get("aggregate", {}):
            lines.append("Aggregate Performance:")
            _headers = ("Metric", "Mean", "Median", "p95", "p99", "StdDev", "Min", "Max", "CoV%")
            _rows: list[tuple[str, ...]] = []
            _rows.extend(
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
                for name, s in agg.items()
                if isinstance(s, dict)
            )
            hdr, sep, data = align_table(_headers, _rows, alignments="<>")
            lines.extend((hdr, sep))
            lines.extend(data)
            lines.append("")

        if res := report.get("resource", {}):
            lines.append("Resource Utilization:")
            lines.extend(f"  {k}: {v:.2f}" for k, v in sorted(res.items()))
            lines.append("")

        if phases := report.get("phases", {}):
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
            lines.extend((hdr, sep))
            lines.extend(data)
            lines.append("")

        if components := report.get("components", {}):
            lines.append("Component Micro-Profile:")
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
            lines.extend((hdr, sep))
            lines.extend(data)
            lines.append("")

        lines.extend(_text_count_table("Runtime Branch Coverage", report.get("coverage", {})))
        lines.extend(
            _text_count_table("Runtime Category Coverage", report.get("category_coverage", {}))
        )
        if scenario_stats := report.get("scenario_stats", {}):
            lines.append("Runtime Scenario Timing:")
            _headers_ph = (
                "Phase",
                "Mean(ms)",
                "Median(ms)",
                "p95(ms)",
                "p99(ms)",
                "Std(ms)",
                "MemΔ(MB)",
            )
            _rows_ph = []
            for name, phase_data in scenario_stats.items():
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
            lines.extend((hdr, sep))
            lines.extend(data)
            lines.append("")
        lines.extend(_text_slow_queries("Slowest Runtime Queries", report.get("slow_queries", [])))

        if regressions := report.get("regressions", []):
            lines.append("Regression Detections:")
            lines.extend(f"  {r}" for r in regressions)
            lines.append("")

        lines.extend(("=" * 90, ""))
    while lines and lines[-1] == "":
        lines.pop()
    atomic_write(path, "\n".join(lines) + "\n")


def _save_cprofile_metric(
    pr: cProfile.Profile,
    profile_dir: Path,
    sort_key: str,
    filename: str,
    action: str,
    label: str,
) -> io.StringIO:
    """Extract, sort, and save cProfile statistics to a text file."""
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
    if action == "print_stats":
        ps.print_stats(50)
    elif action == "print_callers":
        ps.print_callers(50)
    elif action == "print_callees":
        ps.print_callees(50)
    out_path = profile_dir / filename
    out_path.write_text(s.getvalue(), encoding="utf-8")
    print(f"cProfile {label} summary saved to {out_path}")
    return s


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
            "Profiling target (evaluate, build_index, rank, runtime, components, all; "
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
        help="Skip the standalone HassIL baseline (lexical mode still uses its shortcut)",
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
    parser.add_argument(
        "--regenerate-expectations",
        action="store_true",
        default=False,
        help="Regenerate expected_slots in the JSON datasets from the current grammar",
    )

    # Accuracy mode thresholds
    parser.add_argument(
        "--min-intent-slot-accuracy",
        type=float,
        default=None,
        help="Fail when production-flow intent/slot accuracy falls below this percentage",
    )
    parser.add_argument(
        "--max-fallback-rate",
        type=float,
        default=None,
        help="Fail when production-flow fallback rate exceeds this percentage",
    )
    parser.add_argument(
        "--max-mismatch-rate",
        type=float,
        default=None,
        help="Fail when production-flow mismatch rate exceeds this percentage",
    )
    parser.add_argument(
        "--min-language-intent-slot-accuracy",
        type=float,
        default=None,
        help="Fail when any language's intent/slot accuracy falls below this percentage",
    )
    parser.add_argument(
        "--max-language-fallback-rate",
        type=float,
        default=None,
        help="Fail when any language's fallback rate exceeds this percentage",
    )
    parser.add_argument(
        "--max-language-mismatch-rate",
        type=float,
        default=None,
        help="Fail when any language's mismatch rate exceeds this percentage",
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
        "--fail-on-regression",
        action="store_true",
        default=False,
        help="Exit nonzero when performance baseline comparison reports regressions",
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
        safe_target = sanitize_chars(target, _TARGET_ALLOWED_CHARS)
    except ValueError as err:
        print(f"Error: Target contains invalid characters: {err}", file=sys.stderr)
        sys.exit(1)

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
    percentage_thresholds = (
        ("--max-mismatch-rate", args.max_mismatch_rate),
        ("--min-language-intent-slot-accuracy", args.min_language_intent_slot_accuracy),
        ("--max-language-fallback-rate", args.max_language_fallback_rate),
        ("--max-language-mismatch-rate", args.max_language_mismatch_rate),
    )
    for option_name, option_value in percentage_thresholds:
        if option_value is not None and not (0.0 <= option_value <= 100.0):
            print(
                f"Error: {option_name} must be between 0.0 and 100.0",
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
        language_filter = [lang.strip().lower() for lang in langs_split if lang.strip()]
        for lang in language_filter:
            if not all(c.isalnum() or c in "_-" for c in lang):
                print(f"Error: Invalid language code {lang!r}", file=sys.stderr)
                sys.exit(1)

    # Discover datasets
    datasets = discover_datasets(safe_datasets_dir, language_filter)
    if not datasets:
        print(f"Error: No datasets found in {safe_datasets_dir}", file=sys.stderr)
        sys.exit(1)

    if args.regenerate_expectations:
        success = regenerate_all_expectations(
            datasets, str(Path(safe_datasets_dir).relative_to(_REPO_ROOT))
        )
        sys.exit(0 if success else 1)

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
                max_mismatch_rate=args.max_mismatch_rate,
                min_language_intent_slot_accuracy=args.min_language_intent_slot_accuracy,
                max_language_fallback_rate=args.max_language_fallback_rate,
                max_language_mismatch_rate=args.max_language_mismatch_rate,
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
            baseline_mgr.baseline_dir = (
                baseline_path.parent if baseline_path.suffix == ".json" else baseline_path
            )

        print("PROFILE_START", flush=True)
        profile_reports: dict[str, dict[str, Any]] = {}
        try:
            if safe_target == "all":
                all_reports = {}
                for tgt in ("build_index", "rank", "runtime", "components", "evaluate"):
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
                        False,
                        language_filter,
                        warn_on_missing=_explicit_baseline,
                    )
                    all_reports[tgt] = report
                profile_reports = all_reports

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
                report = _run_profiling(
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
                    False,
                    language_filter,
                    warn_on_missing=_explicit_baseline,
                )
                profile_reports[safe_target] = report

            # Optional cProfile dump during rankings profiling
            if args.cprofile:
                print("\nRunning cProfile snapshot ...", flush=True)
                pr = cProfile.Profile()
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

                # 1. Cumulative time stats (overall bottlenecks)
                s_cum = _save_cprofile_metric(
                    pr,
                    profile_dir,
                    "cumulative",
                    "profile_metrics_cumulative.txt",
                    "print_stats",
                    "cumulative",
                )

                # 2. Internal time stats (tottime) - crucial for micro-optimizations
                _ = _save_cprofile_metric(
                    pr,
                    profile_dir,
                    "tottime",
                    "profile_metrics_tottime.txt",
                    "print_stats",
                    "internal-time",
                )

                # 3. Call count stats (ncalls) - find functions called too frequently
                _ = _save_cprofile_metric(
                    pr,
                    profile_dir,
                    "ncalls",
                    "profile_metrics_ncalls.txt",
                    "print_stats",
                    "call-count",
                )

                # 4. Callers relationship stats - showing function callers
                _ = _save_cprofile_metric(
                    pr,
                    profile_dir,
                    "tottime",
                    "profile_metrics_callers.txt",
                    "print_callers",
                    "callers",
                )

                # 5. Callees relationship stats - showing function callees
                _ = _save_cprofile_metric(
                    pr,
                    profile_dir,
                    "tottime",
                    "profile_metrics_callees.txt",
                    "print_callees",
                    "callees",
                )

                print("\nTOP 50 FUNCTIONS BY CUMULATIVE TIME:")
                print("-" * 80)
                print(s_cum.getvalue())

        except KeyboardInterrupt:
            print("\nPROFILE_INTERRUPTED", flush=True)
            sys.exit(1)
        except Exception as exc:
            print(f"\nPROFILE_FAILED: {exc}", flush=True)
            raise

        regressions = _profile_regressions(profile_reports)
        if args.fail_on_regression and regressions:
            print("\nPROFILE_FAILED: performance regressions detected", flush=True)
            for regression in regressions:
                print(f"- {regression}", flush=True)
            sys.exit(1)
        if args.save_baseline:
            for profile_target, report in profile_reports.items():
                baseline_mgr.save(profile_target, dict(report))
        print("PROFILE_OK", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
