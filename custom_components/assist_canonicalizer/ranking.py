"""Lexical candidate ranking."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
from heapq import nsmallest
from math import isclose
from typing import Any

from rapidfuzz import fuzz

from .bm25 import BM25Index
from .candidate import Candidate, candidate_raw_slot_map
from .const import (
    BM25_WEIGHT,
    CHAR_NGRAM_WEIGHT,
    CONTEXT_SLOT_MATCH_BOOST,
    CONTEXT_SLOT_MISMATCH_PENALTY,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES,
    ENTITY_ONLY_UNCOVERED_QUERY_PENALTY,
    ENTITY_SLOT_NAME_SET,
    HIGH_CONFIDENCE_RELAXED_MIN_MARGIN,
    HIGH_CONFIDENCE_RELAXED_MIN_SCORE,
    INTENT_ACTION_WEIGHT,
    LOCATION_SLOT_NAME_SET,
    NON_ENTITY_PENALTY_BLEND,
    NUMERIC_SLOT_MISMATCH_PENALTY,
    NUMERIC_SLOT_WITHOUT_QUERY_PENALTY,
    POSITIONAL_FUZZY_TOKEN_MAX_LENGTH_RATIO,
    POSITIONAL_SIMILARITY_BASE_THRESHOLD,
    POSITIONAL_SIMILARITY_MEDIUM_THRESHOLD,
    POSITIONAL_SIMILARITY_PARTIAL_CREDIT,
    POSITIONAL_SIMILARITY_SHORT_3_THRESHOLD,
    POSITIONAL_SIMILARITY_VERY_SHORT_THRESHOLD,
    RAPIDFUZZ_WEIGHT,
    SAFE_EMPTY_SLOT_RELAXED_MIN_MARGIN,
    SAFE_EMPTY_SLOT_RELAXED_MIN_SCORE,
    SAFE_INTENT_EVIDENCE_MAX_SCORE,
    SAFE_INTENT_EVIDENCE_MIN_ADVANTAGE,
    SAFE_INTENT_EVIDENCE_MIN_SCORE,
    SLOT_FUZZY_TOKEN_MIN_LENGTH,
    SLOT_TOKEN_MATCH_THRESHOLD,
    STATIC_ENTITY_UNCOVERED_QUERY_PENALTY,
    STATIC_SLOT_QUERY_CONFLICT_PENALTY,
    TIEBREAKER_INTENT_MARGIN,
    UNANCHORED_ENTITY_SLOT_PENALTY,
    WILDCARD_KNOWN_SLOT_TOKEN_PENALTY,
    WILDCARD_LENGTH_PENALTY_FACTOR,
    ZERO_INTENT_EVIDENCE_MAX_MARGIN,
    ZERO_INTENT_EVIDENCE_MAX_SCORE,
    FallbackReason,
)
from .normalization import (
    char_ngrams_normalized,
    literal_token_variants,
    normalize_text,
    normalize_text_no_diacritics,
)
from .rehydration import get_wildcard_rehydration, wildcard_variants_analysis
from .utils import (
    NormalizedIntentContext,
    normalize_intent_context,
    normalized_slot_value_tokens,
    parse_float,
)


@lru_cache(maxsize=4096)
def _raw_cached_fuzz_ratio(s1: str, s2: str) -> float:
    """Return RapidFuzz's ratio with LRU caching."""
    return fuzz.ratio(s1, s2)


def _cached_fuzz_ratio(s1: str, s2: str) -> float:
    """Return RapidFuzz's ratio with LRU caching.

    The argument order is normalized to maximize cache hit rate.
    """
    return _raw_cached_fuzz_ratio(s1, s2) if s1 <= s2 else _raw_cached_fuzz_ratio(s2, s1)


_LiteralVariantAnalysis = list[tuple[int, int, list[frozenset[str]]]]
_KNOWN_OPPOSING_INTENT_TIE_PAIRS: tuple[tuple[str, str], ...] = (
    ("HassTurnOn", "HassTurnOff"),
    ("HassMediaUnpause", "HassMediaPause"),
    ("HassMediaPlayerUnmute", "HassMediaPlayerMute"),
    ("HassUnpauseTimer", "HassPauseTimer"),
    ("HassIncreaseTimer", "HassDecreaseTimer"),
    ("HassMediaNext", "HassMediaPrevious"),
)
_KNOWN_OPPOSING_INTENT_TIE_PREFERENCES: Mapping[frozenset[str], Mapping[str, int]] = {
    frozenset((preferred_intent, other_intent)): {
        preferred_intent: 1,
        other_intent: 0,
    }
    for preferred_intent, other_intent in _KNOWN_OPPOSING_INTENT_TIE_PAIRS
}
_STRUCTURAL_TIE_ABS_TOLERANCE = 1e-12
_NUMERIC_MATCH_ABS_TOLERANCE = 1e-5


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Normalized score components for a candidate."""

    rapidfuzz_score: float
    char_ngram_score: float
    bm25_score: float
    intent_score: float
    final_score: float
    penalty: float = 0.0


_PERFECT_SCORE = ScoreBreakdown(
    rapidfuzz_score=1.0,
    char_ngram_score=1.0,
    bm25_score=1.0,
    intent_score=1.0,
    final_score=1.0,
)


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """Candidate paired with ranking scores."""

    candidate: Candidate
    scores: ScoreBreakdown


@dataclass(frozen=True, slots=True)
class _RankedItem:
    """Intermediate ranking record before public score objects are built."""

    final_score: float
    candidate: Candidate
    rapidfuzz_score: float
    char_ngram_score: float
    bm25_score: float
    intent_score: float
    index: int
    penalty: float
    slot_specificity: int


@dataclass(frozen=True, slots=True)
class CharNGramIndex:
    """Exact character n-gram scorer backed by posting lists."""

    gram_counts: tuple[int, ...]
    postings: dict[str, tuple[int, ...]]

    @classmethod
    def from_grams(cls, candidate_grams: Sequence[frozenset[str]]) -> CharNGramIndex:
        """Build a character n-gram posting index from candidate gram sets."""
        postings = defaultdict(list)
        for index, grams in enumerate(candidate_grams):
            for gram in grams:
                postings[gram].append(index)
        return cls(
            gram_counts=tuple(len(grams) for grams in candidate_grams),
            postings={gram: tuple(indexes) for gram, indexes in postings.items()},
        )

    def score(self, query_grams: frozenset[str]) -> list[float]:
        """Return exact normalized Jaccard scores for all indexed candidates."""
        _nothing: tuple[int, ...] = ()
        if not query_grams:
            return [0.0] * len(self.gram_counts)
        intersections = [0] * len(self.gram_counts)
        _postings = self.postings
        _postings_get = _postings.get
        for gram in query_grams:
            for index in _postings_get(gram, _nothing):
                intersections[index] += 1
        query_count = len(query_grams)
        _gram_counts = self.gram_counts
        scores = [0.0] * len(self.gram_counts)
        for index, intersection_size in enumerate(intersections):
            if intersection_size == 0:
                continue
            union_size = query_count + _gram_counts[index] - intersection_size
            scores[index] = intersection_size / union_size if union_size else 0.0
        return scores


def rapidfuzz_similarity_normalized(
    query: str,
    candidate: str,
    *,
    query_token_count: int | None = None,
    query_sorted: str | None = None,
    candidate_sorted: str | None = None,
    candidate_token_count: int | None = None,
) -> float:
    """Return a RapidFuzz score that penalizes unmatched extra tokens."""
    if not query or not candidate:
        return 0.0
    wratio: float = fuzz.WRatio(query, candidate)
    if query_sorted is not None and candidate_sorted is not None:
        token_sort: float = _cached_fuzz_ratio(query_sorted, candidate_sorted)
    else:
        token_sort: float = fuzz.token_sort_ratio(query, candidate)
    token_set: float = fuzz.token_set_ratio(query, candidate)

    query_count = query_token_count if query_token_count is not None else query.count(" ") + 1
    candidate_count = (
        candidate_token_count if candidate_token_count is not None else candidate.count(" ") + 1
    )
    len_ratio = (
        min(query_count, candidate_count) / max(query_count, candidate_count)
        if query_count and candidate_count
        else 0.0
    )
    token_set *= len_ratio

    return (wratio + token_sort + token_set) / 300.0


def token_count_ratio(query: str, candidate: str, *, query_token_count: int | None = None) -> float:
    """Return a length ratio that penalizes unmatched extra tokens."""
    if not query or not candidate:
        return 0.0
    query_count = query_token_count if query_token_count is not None else query.count(" ") + 1
    candidate_count = candidate.count(" ") + 1
    return min(query_count, candidate_count) / max(query_count, candidate_count)


def lexical_score(
    rapidfuzz_score: float,
    char_ngram_score: float,
    bm25_score: float,
    intent_score: float = 1.0,
) -> float:
    """Combine normalized lexical and built-in intent action score components."""
    return (
        RAPIDFUZZ_WEIGHT * rapidfuzz_score
        + CHAR_NGRAM_WEIGHT * char_ngram_score
        + BM25_WEIGHT * bm25_score
        + INTENT_ACTION_WEIGHT * intent_score
    )


def _positional_similarity(a: str, b: str) -> float:
    """Character-level positional similarity, cheap edit-distance proxy."""
    if a == b:
        return 1.0
    len_a = len(a)
    len_b = len(b)
    max_len = max(len_a, len_b)
    if max_len == 0:
        return 0.0
    matches = sum(x == y for x, y in zip(a, b, strict=False))
    return matches / max_len


def _query_token_coverage(
    query_tokens: frozenset[str],
    candidate_tokens: frozenset[str],
) -> float:
    """Return the fraction of query tokens covered by candidate tokens.

    The squaring amplifies the penalty for unmatched query tokens so
    that entity-only candidates do not outrank action-bearing
    candidates when the query contains (possibly misspelled) action
    words.
    """
    if not query_tokens:
        return 1.0
    matched = len(query_tokens & candidate_tokens)
    coverage = matched / len(query_tokens)
    return coverage * coverage


def _exact_intent_score(
    literal_text_or_variants: str | tuple[frozenset[str], ...],
    query_tokens: frozenset[str],
) -> float:
    """Return how well query tokens cover localized template literal words."""
    if isinstance(literal_text_or_variants, str):
        variants = literal_token_variants(literal_text_or_variants)
    else:
        variants = literal_text_or_variants
    if not variants:
        return 1.0
    max_score = 0.0
    for literal_tokens in variants:
        if literal_tokens.issubset(query_tokens):
            return 1.0
        if literal_tokens.isdisjoint(query_tokens):
            score = 0.0
        else:
            matched = len(literal_tokens & query_tokens)
            score = matched / len(literal_tokens)
        if score > max_score:
            max_score = score
    return max_score


def _per_pair_positional_threshold(a: str, b: str) -> float:
    """Return the positional similarity threshold for a pair of tokens.

    Shorter tokens are more susceptible to false-positive matches
    (e.g. ``tắt`` vs ``tát`` = 2/3 = 0.667) than long tokens where
    positional overlap is a reliable signal.  The threshold is driven
    by the shorter token in the pair.
    """
    min_len = min(len(a), len(b))
    if min_len <= 2:
        return POSITIONAL_SIMILARITY_VERY_SHORT_THRESHOLD
    if min_len <= 3:
        return POSITIONAL_SIMILARITY_SHORT_3_THRESHOLD
    if min_len <= 5:
        return POSITIONAL_SIMILARITY_MEDIUM_THRESHOLD
    return POSITIONAL_SIMILARITY_BASE_THRESHOLD


def _build_positional_lookup(
    literal_tokens_set: frozenset[str],
    query_tokens: frozenset[str],
) -> dict[str, frozenset[str]]:
    """Precompute which query tokens each literal token positionally matches.

    Uses first-character bucketing to prune the O(|literal|x|query|) inner
    loop, positional similarity requires at least the first character to
    match, so tokens that differ at position 0 can never reach any
    non-trivial positional similarity threshold.
    """
    first_char_index: dict[str, list[str]] = {}
    for qtok in query_tokens:
        if qtok:
            first_char_index.setdefault(qtok[0], []).append(qtok)

    lookup: dict[str, frozenset[str]] = {}
    for literal_token in literal_tokens_set:
        if literal_token in query_tokens:
            continue
        candidate_q_tokens = first_char_index.get(literal_token[0], []) if literal_token else []
        if not candidate_q_tokens:
            continue
        matched: list[str] = []
        for qtok in candidate_q_tokens:
            sim = _positional_similarity(literal_token, qtok)
            if sim >= _per_pair_positional_threshold(
                literal_token,
                qtok,
            ) or _is_fuzzy_literal_token_match(literal_token, qtok):
                matched.append(qtok)
                if sim >= 0.99:
                    break
        if matched:
            lookup[literal_token] = frozenset(matched)
    return lookup


def _is_fuzzy_literal_token_match(literal_token: str, query_token: str) -> bool:
    """Return whether two same-bucket literal tokens differ by one insertion."""
    literal_len = len(literal_token)
    query_len = len(query_token)
    if abs(literal_len - query_len) != 1:
        return False
    min_len = min(literal_len, query_len)
    if min_len < 2:
        return False
    length_ratio = max(literal_len, query_len) / min_len
    if length_ratio > POSITIONAL_FUZZY_TOKEN_MAX_LENGTH_RATIO:
        return False
    if literal_len < query_len:
        shorter, longer = literal_token, query_token
    else:
        shorter, longer = query_token, literal_token

    skipped = False
    short_idx = 0
    for long_char in longer:
        if short_idx < min_len and shorter[short_idx] == long_char:
            short_idx += 1
            continue
        if skipped:
            return False
        skipped = True
    return short_idx == min_len


def _is_fuzzy_slot_token_match(slot_token: str, query_token: str) -> bool:
    """Return whether a query token is a bounded one-edit slot-token typo."""
    slot_len = len(slot_token)
    query_len = len(query_token)
    if (
        not slot_token
        or not query_token
        or slot_token == query_token
        or slot_token[0] != query_token[0]
    ):
        return False
    if min(slot_len, query_len) < SLOT_FUZZY_TOKEN_MIN_LENGTH:
        return False
    if abs(slot_len - query_len) > 1:
        return False
    if slot_len == query_len:
        return (
            sum(
                slot_char != query_char
                for slot_char, query_char in zip(slot_token, query_token, strict=True)
            )
            == 1
        )
    return _is_fuzzy_literal_token_match(slot_token, query_token)


def _precompute_literal_analysis(
    literal_text_or_variants: str | tuple[frozenset[str], ...],
    query_tokens: frozenset[str],
    positional_lookup: dict[str, frozenset[str]],
) -> _LiteralVariantAnalysis:
    """Precompute positional match data for *literal_text* against *query_tokens*.

    Returns a list with one entry per variant of *literal_text*::

        (total_token_count, exact_match_count, positional_hit_frozensets)

    *exact_match_count* counts how many literal tokens appear verbatim in
    *query_tokens*.  *positional_hit_frozensets* lists the query-token
    frozensets that positionally match each remaining literal token (only
    entries where ``positional_lookup`` has a hit are stored).

    This data is computed once per unique ``literal_text`` per
    ``rank_candidates`` invocation; the per-candidate scoring step then only
    needs to check whether each frozenset is a subset of the candidate entity,
    an O(variants * positional_hits) operation instead of the previous
    O(variants * all_tokens) loop with repeated dict lookups.
    """
    if isinstance(literal_text_or_variants, str):
        variants = literal_token_variants(literal_text_or_variants)
    else:
        variants = literal_text_or_variants
    result: _LiteralVariantAnalysis = []
    for literal_tokens in variants:
        exact_count = 0
        positional_hits: list[frozenset[str]] = []
        for token in literal_tokens:
            if token in query_tokens:
                exact_count += 1
            else:
                matching = positional_lookup.get(token)
                if matching is not None:
                    positional_hits.append(matching)
        result.append((len(literal_tokens), exact_count, positional_hits))
    return result


def _positional_intent_score_from_lookup(
    literal_text: str,
    query_tokens: frozenset[str],
    positional_lookup: dict[str, frozenset[str]],
    candidate_entity: frozenset[str] | None = None,
) -> float:
    """Entity-aware positional intent score using a precomputed lookup table.

    ``positional_lookup`` maps each literal token → query tokens that
    positionally match it.  When ``candidate_entity`` is provided those
    tokens (entity slots belonging to the candidate) are excluded from
    the search space, preventing e.g. ``tắt`` (turn off) from
    accidentally matching ``tắm`` (bath) in a ``bật quạt phòng tắm`` query.
    """
    variants = literal_token_variants(literal_text)
    if not variants:
        return 1.0
    if candidate_entity is not None and query_tokens.issubset(candidate_entity):
        return _exact_intent_score(variants, query_tokens)
    best_score = 0.0
    for literal_tokens in variants:
        matched_weight = 0.0
        for token in literal_tokens:
            if token in query_tokens:
                matched_weight += 1.0
                continue
            matching = positional_lookup.get(token)
            if matching is not None and (
                candidate_entity is None or not matching.issubset(candidate_entity)
            ):
                matched_weight += POSITIONAL_SIMILARITY_PARTIAL_CREDIT
        score = matched_weight / len(literal_tokens)
        if score > best_score:
            best_score = score
    return best_score


def _non_entity_coverage(
    query_tokens: frozenset[str],
    positional_literal_tokens: frozenset[str],
    candidate_tokens: frozenset[str],
    *,
    non_entity: frozenset[str] | None = None,
) -> float:
    """Return how well a candidate covers query tokens that are not part of any template literal.

    Tokens in the query that do not appear in any template's static literal list
    (``positional_literal_tokens`` represents template-level static literal tokens, excluding
    dynamic slot values like entities or areas) are typically a mixture of entity/area slot values
    and politeness words, filler words, or action synonyms. A candidate whose tokens cover
    more of these should be preferred.
    """
    if non_entity is None:
        non_entity = query_tokens - positional_literal_tokens
    if not non_entity:
        return 1.0
    matched = len(non_entity & candidate_tokens)
    coverage = matched / len(non_entity)
    return coverage * coverage


def _rehydrated_bm25_score(
    cand_tokens: tuple[str, ...],
    query_tokens: tuple[str, ...],
    index: BM25Index,
    max_raw_score: float,
) -> float:
    """Compute the normalized BM25 score of a rehydrated candidate scaled by max_raw_score."""
    raw_score = index.raw_score_tokens(cand_tokens, query_tokens)
    if max_raw_score <= 0.0:
        return 1.0 if raw_score > 0.0 else 0.0
    return min(1.0, raw_score / max_raw_score)


def _wildcard_variants_match(
    variants_with_len: tuple[tuple[frozenset[str], int, int], ...],
    query_tokens: frozenset[str],
) -> bool:
    """Return True if any wildcard variant has sufficient token overlap with the query."""
    for variant, var_len, req in variants_with_len:
        if var_len == 0:
            return True
        if len(variant & query_tokens) >= req:
            return True
    return False


def _check_precomputed_wildcard(
    i: int,
    query_tokens: frozenset[str],
    wildcard_always_passes: frozenset[int],
    wildcard_variants_with_len: dict[int, tuple[tuple[frozenset[str], int, int], ...]] | None,
    wildcard_literal_tokens_by_index: dict[int, frozenset[str]] | None,
    wildcard_min_required_by_index: dict[int, int] | None,
) -> bool:
    """Return True if precomputed wildcard candidate i passes the token filter."""
    if i in wildcard_always_passes:
        return True
    variants_with_len = wildcard_variants_with_len.get(i) if wildcard_variants_with_len else None
    if not variants_with_len:
        return False
    if wildcard_literal_tokens_by_index is not None and wildcard_min_required_by_index is not None:
        literal_tokens = wildcard_literal_tokens_by_index.get(i)
        min_required = wildcard_min_required_by_index.get(i)
        if (
            literal_tokens is not None
            and min_required is not None
            and min_required > 0
            and len(literal_tokens & query_tokens) < min_required
        ):
            return False
    return _wildcard_variants_match(variants_with_len, query_tokens)


def _check_onthefly_wildcard(cand: Candidate, query_tokens: frozenset[str]) -> bool:
    """Return True if an on-the-fly analyzed wildcard candidate passes the token filter."""
    var_with_len, all_literal = wildcard_variants_analysis(cand)
    always_passes = not cand.literal_variants or any(length == 0 for _, length, _ in var_with_len)
    if always_passes:
        return True
    if all_literal.isdisjoint(query_tokens):
        return False
    return _wildcard_variants_match(var_with_len, query_tokens)


def _prefilter_wildcard_candidates(
    candidates: Sequence[Candidate],
    query_tokens: frozenset[str],
    wildcard_always_passes: frozenset[int] | None,
    wildcard_variants_with_len: dict[int, tuple[tuple[frozenset[str], int, int], ...]] | None,
    wildcard_token_to_indices: dict[str, tuple[int, ...]] | None,
    wildcard_literal_tokens_by_index: dict[int, frozenset[str]] | None = None,
    wildcard_min_required_by_index: dict[int, int] | None = None,
) -> set[int]:
    """Prefilter wildcard candidates using precomputed structures or on-the-fly coverage."""
    if (
        wildcard_always_passes is not None
        and wildcard_variants_with_len is not None
        and wildcard_token_to_indices is not None
        and wildcard_literal_tokens_by_index is not None
        and wildcard_min_required_by_index is not None
    ):
        candidates_to_check: set[int] = set(wildcard_always_passes)
        for token in query_tokens:
            if indices := wildcard_token_to_indices.get(token):
                candidates_to_check.update(indices)
        return {
            i
            for i in candidates_to_check
            if _check_precomputed_wildcard(
                i,
                query_tokens,
                wildcard_always_passes,
                wildcard_variants_with_len,
                wildcard_literal_tokens_by_index,
                wildcard_min_required_by_index,
            )
        }
    return {
        i
        for i, cand in enumerate(candidates)
        if cand.has_wildcard and _check_onthefly_wildcard(cand, query_tokens)
    }


def _exact_lookup_ranked(
    query: str,
    query_normalized: str,
    max_candidates: int,
    exact_normalized_lookup: dict[str, list[Candidate]] | None,
    exact_no_diacritics_lookup: dict[str, list[Candidate]] | None,
    language: str | None,
) -> tuple[RankedCandidate, ...] | None:
    """Return exact-match ranked candidates or None when fuzzy ranking is required."""
    if exact_normalized_lookup is not None and (
        exact_matches := exact_normalized_lookup.get(query_normalized)
    ):
        return tuple(
            RankedCandidate(candidate=c, scores=_PERFECT_SCORE)
            for c in exact_matches[:max_candidates]
        )
    if exact_no_diacritics_lookup is not None:
        query_no_diac = normalize_text_no_diacritics(query, language)
        if no_diac_matches := exact_no_diacritics_lookup.get(query_no_diac):
            unique_intents = {c.intent_name for c in no_diac_matches}
            if len(unique_intents) == 1:
                return tuple(
                    RankedCandidate(candidate=c, scores=_PERFECT_SCORE)
                    for c in no_diac_matches[:max_candidates]
                )
    return None


def _rank_query_setup(
    query_normalized: str,
    positional_literal_tokens: frozenset[str],
) -> tuple[frozenset[str], tuple[str, ...], int, str, frozenset[str] | None]:
    """Return normalized query token structures shared by ranking and profiling."""
    query_tokens_tuple = tuple(query_normalized.split())
    query_tokens = frozenset(query_tokens_tuple)
    query_token_count = len(query_tokens_tuple)
    query_sorted = " ".join(sorted(query_tokens_tuple))
    non_entity_tokens: frozenset[str] | None = None
    if positional_literal_tokens:
        non_entity_scratch = query_tokens - positional_literal_tokens
        non_entity_tokens = non_entity_scratch or None
    return query_tokens, query_tokens_tuple, query_token_count, query_sorted, non_entity_tokens


def _normalized_bm25_scores_from_raw(
    raw_scores: Sequence[float],
    doc_count: int,
) -> tuple[float, ...]:
    """Return normalized BM25 scores from raw document scores."""
    if doc_count < 1:
        return ()
    max_raw_score = max(raw_scores, default=0.0)
    if max_raw_score <= 0.0:
        return (0.0,) * doc_count
    inv_max = 1.0 / max_raw_score
    return tuple(score * inv_max for score in raw_scores)


def _rank_prefilter_keys(
    char_scores: Sequence[float],
    bm25_scores: Sequence[float],
) -> list[float]:
    """Return heap keys used to select rank prefilter candidates."""
    return [
        -(CHAR_NGRAM_WEIGHT * char_score + BM25_WEIGHT * bm25_score)
        for char_score, bm25_score in zip(char_scores, bm25_scores, strict=True)
    ]


def _rank_prefilter_limit(
    candidate_count: int,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    rapidfuzz_prefilter_candidates: int = DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES,
) -> int:
    """Return the number of candidates considered by expensive final ranking."""
    disambiguation_limit = max(2, max_candidates)
    return min(candidate_count, max(rapidfuzz_prefilter_candidates, disambiguation_limit))


def _top_prefilter_indices(
    prefilter_keys: Sequence[float],
    prefilter_limit: int,
) -> list[int]:
    """Return candidate indices selected by the prefilter heap."""
    return nsmallest(prefilter_limit, range(len(prefilter_keys)), key=prefilter_keys.__getitem__)


def _query_slot_tokens_from_index(
    query_tokens: frozenset[str],
    top_indices: Sequence[int],
    slot_token_to_indices: dict[str, tuple[int, ...]],
) -> frozenset[str]:
    """Return indexed slot tokens referenced exactly or fuzzily by the query."""
    top_index_set = set(top_indices)
    active_slot_tokens: list[str] = []
    matched_tokens: set[str] = set()
    for token, indexes in slot_token_to_indices.items():
        if all(idx not in top_index_set for idx in indexes):
            continue
        active_slot_tokens.append(token)
        if token in query_tokens:
            matched_tokens.add(token)

    for query_token in query_tokens:
        if query_token in matched_tokens:
            continue
        matched_tokens.update(
            slot_token
            for slot_token in active_slot_tokens
            if slot_token not in matched_tokens
            and _is_fuzzy_slot_token_match(slot_token, query_token)
        )
    return frozenset(matched_tokens)


# ---------------------------------------------------------------------------
# Slot/context conflict detection helpers
# ---------------------------------------------------------------------------
# Slot/context scoring guard matrix:
# - Explicit query slot tokens must be covered by candidate slot tokens, static
#   text, or rehydrated wildcard text; otherwise the local slot penalty applies.
# - Non-static numeric/entity slots need query anchors, while broad static
#   domain/name candidates are penalized when query slot words point elsewhere.
# - HassIL intent context is applied after slot penalties: matching explicit
#   context boosts, conflicting explicit context penalizes, and metadata-only
#   ``context_slots`` boosts only slots that HassIL says are context-supplied.
# - Wildcard candidates are additionally penalized when a free-text wildcard
#   absorbs known slot words that should have remained concrete.
# Fuzzy checks in this section run after candidate prefiltering and operate on
# short query-token/slot-token sets; helpers keep cheap exact/static guards first
# so RapidFuzz work is skipped for unrelated or already-covered candidates.


def _static_slot_names(candidate: Candidate) -> frozenset[str]:
    """Return slot names declared static in candidate metadata."""
    static_slots_text = candidate.metadata.get("static_slots", "")
    if not isinstance(static_slots_text, str) or not static_slots_text:
        return frozenset()
    return frozenset(slot for slot in static_slots_text.split(",") if slot)


def _context_slot_names(candidate: Candidate) -> frozenset[str]:
    """Return slot names supplied by HassIL intent context."""
    context_slots_text = candidate.metadata.get("context_slots", "")
    if not isinstance(context_slots_text, str) or not context_slots_text:
        return frozenset()
    return frozenset(slot for slot in context_slots_text.split(",") if slot)


def _context_key_aliases(slot_name: str) -> frozenset[str]:
    """Return context keys equivalent to a candidate slot name."""
    if slot_name in LOCATION_SLOT_NAME_SET:
        return LOCATION_SLOT_NAME_SET
    return frozenset({slot_name})


def _context_slot_adjustment(
    slots: Mapping[str, Any],
    context_slot_names: frozenset[str],
    normalized_context: NormalizedIntentContext,
) -> float:
    """Return a language-neutral score adjustment from intent context.

    Positive values reward explicit or context-injected slot agreement.
    Negative values penalize explicit candidate slots that conflict with
    context. Missing explicit slots are only rewarded when HassIL marks
    the slot as context-supplied.
    """
    if not normalized_context:
        return 0.0

    score = 0.0
    for context_name, context_tokens in normalized_context.items():
        if candidate_values := [
            slot_value
            for slot_name, slot_value in slots.items()
            if context_name in _context_key_aliases(slot_name)
        ]:
            if any(
                normalized_slot_value_tokens(slot_value) == context_tokens
                for slot_value in candidate_values
            ):
                score += 1.0
            else:
                score -= 1.0
            continue
        if context_name in context_slot_names:
            score += 1.0
    return score


def _has_numeric_query_token(query_tokens: frozenset[str]) -> bool:
    """Return whether the query contains any numeric token."""
    return any(any(char.isdigit() for char in token) for token in query_tokens)


def _is_numeric_slot_value(value: Any) -> bool:
    """Return whether a slot value is numeric-like."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    try:
        float(stripped)
    except ValueError:
        return False
    return True


def _has_unanchored_numeric_slot(
    slots: Mapping[str, Any],
    static_slots: frozenset[str],
    *,
    query_has_number: bool,
) -> bool:
    """Return whether a non-static numeric slot lacks a numeric query anchor."""
    if query_has_number:
        return False
    return any(
        slot_name not in static_slots and _is_numeric_slot_value(slot_value)
        for slot_name, slot_value in slots.items()
    )


def _query_numeric_values(query_tokens: frozenset[str]) -> set[float]:
    """Extract all numeric values from query tokens."""
    vals = set()
    for token in query_tokens:
        val = parse_float(token)
        if val is not None:
            vals.add(val)
    return vals


def _has_numeric_slot_mismatch(
    candidate: Candidate,
    slots_dict: Mapping[str, Any],
    query_numbers: set[float],
    static_slots: frozenset[str],
) -> bool:
    """Return whether any dynamic numeric slot value does not match any query number.

    To verify if slot numbers match the query numbers:
    1. We extract dynamic (non-static) slots.
    2. We prefer looking up the original spoken text from `slots_raw` (which maps to
       the unmultiplied raw spoken token) so that multipliers (e.g. converting
       "volume down by 20" to a slot value of -20) do not cause a false-positive mismatch.
    3. If `slots_raw` is not populated, we fall back to checking both the output value
       and its negation (e.g., matching `-20` against `20`) to handle multipliers.
    """
    if not query_numbers:
        return False

    raw_slots_dict = candidate_raw_slot_map(candidate)
    dynamic_numeric_values = []
    for name, value in slots_dict.items():
        if name in static_slots:
            continue
        # Get raw value if available, else fall back to multiplied output value
        raw_val = raw_slots_dict.get(name, value)
        val_float = None
        if isinstance(raw_val, str):
            val_float = parse_float(raw_val)
        elif isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
            val_float = float(raw_val)

        if val_float is not None:
            dynamic_numeric_values.append(val_float)

    if not dynamic_numeric_values:
        return False

    return any(
        not any(
            isclose(q_num, val, abs_tol=_NUMERIC_MATCH_ABS_TOLERANCE)
            or isclose(q_num, -val, abs_tol=_NUMERIC_MATCH_ABS_TOLERANCE)
            for q_num in query_numbers
        )
        for val in dynamic_numeric_values
    )


def _has_unanchored_entity_slot(
    slots: Mapping[str, Any],
    static_slots: frozenset[str],
    query_tokens_tuple: tuple[str, ...],
) -> bool:
    """Return whether a non-static entity slot lacks lexical query evidence.

    Static domain/name slots are broad HassIL expansions and are handled by
    the static-slot conflict checks below.
    """
    query_tokens = frozenset(query_tokens_tuple)
    for slot_name, slot_value in slots.items():
        if slot_name in static_slots or slot_name not in ENTITY_SLOT_NAME_SET:
            continue
        slot_tokens = normalized_slot_value_tokens(slot_value)
        if not slot_tokens:
            continue
        if any(
            slot_token in query_tokens
            or any(
                _cached_fuzz_ratio(slot_token, query_token) >= SLOT_TOKEN_MATCH_THRESHOLD
                for query_token in query_tokens_tuple
            )
            for slot_token in slot_tokens
        ):
            continue
        return True
    return False


def _has_entity_only_uncovered_query_tokens(
    query_tokens: frozenset[str],
    candidate_tokens: frozenset[str],
    candidate: Candidate,
    slots: Mapping[str, Any],
    static_slots: frozenset[str],
) -> bool:
    """Return whether an entity-only candidate leaves query words unexplained."""
    if candidate.literal_variants or not query_tokens:
        return False
    if all(slot_name in static_slots for slot_name in slots):
        return False
    if not (candidate.slot_tokens_set & query_tokens):
        return False
    return bool(query_tokens - candidate_tokens)


def _has_static_entity_uncovered_query_tokens(
    query_tokens_tuple: tuple[str, ...],
    candidate_tokens: frozenset[str],
    slots: Mapping[str, Any],
    static_slots: frozenset[str],
) -> bool:
    """Return whether a broad static entity candidate leaves query tokens unexplained.

    This fires only when static entity slots are the sole entity evidence;
    candidates with an explicit entity slot are checked by the unanchored
    entity-slot guard.
    """
    if not query_tokens_tuple:
        return False
    has_static_entity = any(
        slot_name in static_slots and slot_name in ENTITY_SLOT_NAME_SET for slot_name in slots
    )
    if not has_static_entity:
        return False
    has_explicit_entity = any(
        slot_name not in static_slots and slot_name in ENTITY_SLOT_NAME_SET for slot_name in slots
    )
    if has_explicit_entity:
        return False
    for query_token in query_tokens_tuple:
        if query_token in candidate_tokens:
            continue
        if all(
            _cached_fuzz_ratio(query_token, candidate_token) < SLOT_TOKEN_MATCH_THRESHOLD
            for candidate_token in candidate_tokens
        ):
            return True
    return False


def _has_static_slot_query_conflict(
    query_slot_tokens: frozenset[str],
    candidate_tokens: frozenset[str],
    slots: Mapping[str, Any],
    static_slots: frozenset[str],
) -> bool:
    """Return whether a broad static slot candidate misses explicit query slots."""
    if not query_slot_tokens:
        return False
    if all(slot_name not in static_slots for slot_name in slots):
        return False
    for query_token in query_slot_tokens:
        if query_token in candidate_tokens:
            continue
        if all(
            _cached_fuzz_ratio(query_token, candidate_token) < SLOT_TOKEN_MATCH_THRESHOLD
            for candidate_token in candidate_tokens
        ):
            return True
    return False


def _has_wildcard_known_slot_token_absorption(
    query_slot_tokens: frozenset[str],
    wildcard_tokens: frozenset[str],
    concrete_slot_tokens: frozenset[str],
) -> bool:
    """Return whether a wildcard consumed query terms that look like known slots."""
    return bool(query_slot_tokens and wildcard_tokens and not concrete_slot_tokens) and bool(
        query_slot_tokens & wildcard_tokens
    )


# ---------------------------------------------------------------------------
# Wildcard rehydration helpers
# ---------------------------------------------------------------------------


def _rehydrate_and_rescore_wildcard(
    candidate: Candidate,
    query: str,
    query_tokens_tuple: tuple[str, ...],
    query_grams: frozenset[str],
    bm25_ref: BM25Index | None,
    max_raw_score: float,
    original_char_score: float,
    original_bm25_score: float,
) -> tuple[str | None, dict[str, str] | None, float, float]:
    """Rehydrate wildcard candidate and recompute its lexical scores."""
    rehydrated, replacements = get_wildcard_rehydration(candidate, query, query_tokens_tuple)
    if not replacements:
        return None, None, original_char_score, original_bm25_score

    rehydrated_norm = normalize_text(rehydrated)

    # Recompute Char-Ngram score
    rehydrated_grams = char_ngrams_normalized(rehydrated_norm)
    if rehydrated_grams and query_grams:
        intersection = len(rehydrated_grams & query_grams)
        union = len(rehydrated_grams | query_grams)
        char_score = intersection / union if union else 0.0
    else:
        char_score = 0.0

    # Recompute BM25 score
    bm25_score = original_bm25_score
    if bm25_ref is not None:
        bm25_score = _rehydrated_bm25_score(
            tuple(rehydrated_norm.split()),
            query_tokens_tuple,
            bm25_ref,
            max_raw_score,
        )

    return rehydrated_norm, replacements, char_score, bm25_score


@dataclass(frozen=True, slots=True)
class _ScoringContext:
    """Shared scoring context and caches for candidate ranking."""

    query: str
    query_normalized: str
    query_tokens: frozenset[str]
    query_tokens_tuple: tuple[str, ...]
    query_token_count: int
    query_sorted: str
    query_grams: frozenset[str]
    query_slot_tokens: frozenset[str]
    query_has_number: bool
    query_numbers: set[float]
    bm25_ref: BM25Index | None
    max_raw_score: float
    positional_lookup: dict[str, frozenset[str]]
    positional_literal_tokens: frozenset[str] | None
    non_entity_tokens: frozenset[str] | None
    candidate_slot_tokens: tuple[frozenset[str], ...] | None
    slot_tokens_by_index: dict[int, frozenset[str]]
    min_confidence: float
    normalized_context: NormalizedIntentContext
    wildcard_passed_set: frozenset[int] | set[int]
    rehydrated_cache: dict[int, tuple[str, dict[str, str]]]
    intent_score_cache: dict[tuple[frozenset[str], ...], float]
    literal_analysis_cache: dict[tuple[str, tuple[frozenset[str], ...]], _LiteralVariantAnalysis]


def _best_positional_score(
    analysis: _LiteralVariantAnalysis,
    candidate_tokens: frozenset[str],
) -> float:
    """Calculate the best score from positional hits analysis."""
    best = 0.0
    for total_len, exact_count, positional_hits in analysis:
        matched = float(exact_count)
        for hits in positional_hits:
            if not hits.issubset(candidate_tokens):
                matched += POSITIONAL_SIMILARITY_PARTIAL_CREDIT
        score = matched / total_len if total_len else 0.0
        if score > best:
            best = score
    return best


def _get_wildcard_slot_tokens(
    idx: int,
    candidate: Candidate,
    context: _ScoringContext,
    rehydrated: tuple[str, dict[str, str]] | None,
) -> tuple[frozenset[str], frozenset[str], bool]:
    """Retrieve candidate slot tokens, wildcard tokens, and check leading placeholder wildcard."""
    cand_slot_tokens = _candidate_slot_tokens_at(
        idx,
        context.candidate_slot_tokens,
        context.slot_tokens_by_index,
    )
    wildcard_infos = candidate.wildcard_infos
    wildcard_tokens = frozenset()
    leading_placeholder_only_wildcard = False
    if cand_slot_tokens and rehydrated is not None and wildcard_infos:
        _, replacements = rehydrated
        placeholder_tokens = frozenset(
            tok for _, name in wildcard_infos for tok in normalize_text(name).split()
        )
        cand_slot_tokens = cand_slot_tokens - placeholder_tokens
        first_wc_idx = min(wc_idx for wc_idx, _ in wildcard_infos)
        leading_placeholder_only_wildcard = first_wc_idx == 0 and not cand_slot_tokens
        if first_wc_idx > 0 or cand_slot_tokens:
            wildcard_tokens = frozenset(
                token
                for wildcard_value in replacements.values()
                for token in normalize_text(wildcard_value).split()
            )
    return cand_slot_tokens, wildcard_tokens, leading_placeholder_only_wildcard


def _check_and_calculate_conflict_penalty(
    cand_slot_tokens: frozenset[str],
    wildcard_tokens: frozenset[str],
    candidate: Candidate,
    context: _ScoringContext,
) -> float:
    """Calculate penalty when there is a conflict between query and candidate slot tokens."""
    if not cand_slot_tokens or not context.query_slot_tokens:
        return 1.0

    allowed_cand_tokens = cand_slot_tokens | wildcard_tokens | candidate.normalized_tokens_set
    has_conflict = False
    for q_tok in context.query_slot_tokens:
        is_matched = q_tok in allowed_cand_tokens or any(
            _cached_fuzz_ratio(q_tok, c_tok) >= SLOT_TOKEN_MATCH_THRESHOLD
            for c_tok in allowed_cand_tokens
        )
        if not is_matched:
            has_conflict = True
            break

    if not has_conflict:
        return 1.0

    cand_matched = sum(
        c_tok in context.query_tokens_tuple
        or any(
            _cached_fuzz_ratio(c_tok, q_tok) >= SLOT_TOKEN_MATCH_THRESHOLD
            for q_tok in context.query_tokens_tuple
        )
        for c_tok in cand_slot_tokens
    )
    cand_coverage = cand_matched / len(cand_slot_tokens) if cand_slot_tokens else 1.0

    query_matched = sum(
        q_tok in allowed_cand_tokens
        or any(
            _cached_fuzz_ratio(q_tok, c_tok) >= SLOT_TOKEN_MATCH_THRESHOLD
            for c_tok in allowed_cand_tokens
        )
        for q_tok in context.query_slot_tokens
    )
    query_coverage = (
        query_matched / len(context.query_slot_tokens) if context.query_slot_tokens else 1.0
    )

    return cand_coverage * (0.8 + 0.2 * query_coverage)


def _check_leading_placeholder_conflict(
    wildcard_tokens: frozenset[str],
    candidate: Candidate,
    context: _ScoringContext,
) -> bool:
    """Check if there is a conflict for a leading placeholder only wildcard."""
    allowed_cand_tokens = wildcard_tokens | candidate.normalized_tokens_set
    return any(
        q_tok not in allowed_cand_tokens
        and all(
            _cached_fuzz_ratio(q_tok, c_tok) < SLOT_TOKEN_MATCH_THRESHOLD
            for c_tok in allowed_cand_tokens
        )
        for q_tok in context.query_slot_tokens
    )


def _calculate_slot_penalty(
    idx: int,
    candidate: Candidate,
    context: _ScoringContext,
    rehydrated: tuple[str, dict[str, str]] | None,
) -> tuple[float, frozenset[str], frozenset[str]]:
    """Calculate slot penalty, wildcard tokens, and candidate slot tokens."""
    if not candidate.has_wildcard and not candidate.slot_tokens_set:
        return 1.0, frozenset(), frozenset()

    cand_slot_tokens, wildcard_tokens, leading_placeholder_only_wildcard = (
        _get_wildcard_slot_tokens(idx, candidate, context, rehydrated)
    )

    slot_penalty = 1.0
    if cand_slot_tokens and context.query_slot_tokens:
        slot_penalty = _check_and_calculate_conflict_penalty(
            cand_slot_tokens,
            wildcard_tokens,
            candidate,
            context,
        )
    elif (
        leading_placeholder_only_wildcard
        and context.query_slot_tokens
        and _check_leading_placeholder_conflict(wildcard_tokens, candidate, context)
    ):
        slot_penalty = 0.0

    return slot_penalty, wildcard_tokens, cand_slot_tokens


def _apply_candidate_slot_penalties(
    candidate: Candidate,
    candidate_tokens: frozenset[str],
    context: _ScoringContext,
    score: float,
) -> float:
    """Apply numeric, entity, and static slot mismatch penalties to the score."""
    slots = candidate.parsed_slots
    if not slots:
        return score

    static_slots = _static_slot_names(candidate)
    if _has_unanchored_numeric_slot(
        slots,
        static_slots,
        query_has_number=context.query_has_number,
    ):
        score *= NUMERIC_SLOT_WITHOUT_QUERY_PENALTY
    if _has_numeric_slot_mismatch(candidate, slots, context.query_numbers, static_slots):
        score *= NUMERIC_SLOT_MISMATCH_PENALTY
    if _has_unanchored_entity_slot(slots, static_slots, context.query_tokens_tuple):
        score *= UNANCHORED_ENTITY_SLOT_PENALTY
    if _has_entity_only_uncovered_query_tokens(
        context.query_tokens,
        candidate_tokens,
        candidate,
        slots,
        static_slots,
    ):
        score *= ENTITY_ONLY_UNCOVERED_QUERY_PENALTY
    if _has_static_entity_uncovered_query_tokens(
        context.query_tokens_tuple,
        candidate_tokens,
        slots,
        static_slots,
    ):
        score *= STATIC_ENTITY_UNCOVERED_QUERY_PENALTY
    if _has_static_slot_query_conflict(
        context.query_slot_tokens,
        candidate_tokens,
        slots,
        static_slots,
    ):
        score *= STATIC_SLOT_QUERY_CONFLICT_PENALTY

    return score


def _get_candidate_text_and_variants(
    candidate: Candidate,
    rehydrated: tuple[str, dict[str, str]] | None,
) -> tuple[str, str, int, frozenset[str], tuple[frozenset[str], ...], int]:
    """Get candidate text features and literal variants, handling rehydration if active."""
    if rehydrated is not None:
        cand_text, replacements = rehydrated
        cand_tokens_list = cand_text.split()
        cand_sorted = " ".join(sorted(cand_tokens_list))
        cand_token_count = len(cand_tokens_list)
        candidate_tokens = frozenset(cand_tokens_list)

        norm_replacements = {wc: normalize_text(val).split() for wc, val in replacements.items()}
        rehydrated_variants = []
        for variant in candidate.literal_variants:
            if variant.isdisjoint(norm_replacements):
                rehydrated_variants.append(variant)
                continue
            new_variant = set()
            for token in variant:
                if token in norm_replacements:
                    new_variant.update(norm_replacements[token])
                else:
                    new_variant.add(token)
            rehydrated_variants.append(frozenset(new_variant))
        literal_variants = tuple(rehydrated_variants)
        total_unique_literal_tokens = (
            len({tok for var in literal_variants for tok in var}) if literal_variants else 0
        )
    else:
        cand_text = candidate.normalized_text
        cand_sorted = candidate.normalized_text_sorted
        cand_token_count = len(candidate.normalized_tokens)
        candidate_tokens = candidate.normalized_tokens_set
        literal_variants = candidate.literal_variants
        total_unique_literal_tokens = candidate.total_unique_literal_tokens

    return (
        cand_text,
        cand_sorted,
        cand_token_count,
        candidate_tokens,
        literal_variants,
        total_unique_literal_tokens,
    )


def _calculate_intent_score(
    candidate: Candidate,
    candidate_tokens: frozenset[str],
    literal_variants: tuple[frozenset[str], ...],
    total_unique_literal_tokens: int,
    context: _ScoringContext,
) -> float:
    """Calculate the intent score based on literal variants coverage and positional hits."""
    literal_text = candidate.metadata.get("literal_text")
    coverage = _query_token_coverage(context.query_tokens, candidate_tokens)
    intent_score = coverage
    if not literal_text:
        return intent_score

    exact = context.intent_score_cache.get(literal_variants)
    if exact is None:
        exact = _exact_intent_score(literal_variants, context.query_tokens)
        context.intent_score_cache[literal_variants] = exact
    if exact >= 1.0:
        matched_non_empty = any(
            var and var.issubset(context.query_tokens) for var in literal_variants
        )
        if not matched_non_empty and not context.query_tokens.issubset(candidate_tokens):
            exact = 0.0
    if exact >= 1.0:
        if total_unique_literal_tokens >= 2:
            matched_q = len(context.query_tokens & candidate_tokens)
            intent_score = matched_q / len(context.query_tokens) if context.query_tokens else 1.0
        elif candidate.slot_tokens_set:
            intent_score = exact
    elif context.query_tokens.issubset(candidate_tokens) or not context.positional_lookup:
        intent_score = exact
    else:
        analysis_key = (literal_text, literal_variants)
        analysis = context.literal_analysis_cache.get(analysis_key)
        if analysis is None:
            analysis = _precompute_literal_analysis(
                literal_variants, context.query_tokens, context.positional_lookup
            )
            context.literal_analysis_cache[analysis_key] = analysis
        intent_score = _best_positional_score(analysis, candidate_tokens)
    if context.positional_literal_tokens:
        penalty = _non_entity_coverage(
            context.query_tokens,
            context.positional_literal_tokens,
            candidate_tokens,
            non_entity=context.non_entity_tokens,
        )
        intent_score *= 1.0 - NON_ENTITY_PENALTY_BLEND + NON_ENTITY_PENALTY_BLEND * penalty
    return intent_score


def _score_single_candidate(
    idx: int,
    candidate: Candidate,
    bm25_score: float,
    char_score: float,
    context: _ScoringContext,
) -> _RankedItem | None:
    """Score a single candidate and return a _RankedItem or None if filtered out."""
    rehydrated = None
    if candidate.has_wildcard:
        if idx not in context.wildcard_passed_set:
            return None
        rehydrated_norm, replacements, char_score, bm25_score = _rehydrate_and_rescore_wildcard(
            candidate,
            context.query,
            context.query_tokens_tuple,
            context.query_grams,
            context.bm25_ref,
            context.max_raw_score,
            char_score,
            bm25_score,
        )
        if not replacements or rehydrated_norm is None:
            return None
        context.rehydrated_cache[idx] = (rehydrated_norm, replacements)
        rehydrated = (rehydrated_norm, replacements)

    (
        cand_text,
        cand_sorted,
        cand_token_count,
        candidate_tokens,
        literal_variants,
        total_unique_literal_tokens,
    ) = _get_candidate_text_and_variants(candidate, rehydrated)

    rapidfuzz_score = rapidfuzz_similarity_normalized(
        context.query_normalized,
        cand_text,
        query_token_count=context.query_token_count,
        query_sorted=context.query_sorted,
        candidate_sorted=cand_sorted,
        candidate_token_count=cand_token_count,
    )

    intent_score = _calculate_intent_score(
        candidate,
        candidate_tokens,
        literal_variants,
        total_unique_literal_tokens,
        context,
    )

    combined = lexical_score(rapidfuzz_score, char_score, bm25_score, intent_score)

    # Slot matching penalty
    slot_penalty, wildcard_tokens, cand_slot_tokens = _calculate_slot_penalty(
        idx,
        candidate,
        context,
        rehydrated,
    )

    # Apply penalty only if less than 100% of slot tokens match
    if slot_penalty < 1.0:
        base_multiplier = min(1.0, max(0.1, context.min_confidence - 0.05))
        combined *= base_multiplier + (1.0 - base_multiplier) * slot_penalty
    if _has_wildcard_known_slot_token_absorption(
        context.query_slot_tokens,
        wildcard_tokens,
        cand_slot_tokens,
    ):
        combined *= WILDCARD_KNOWN_SLOT_TOKEN_PENALTY

    # Apply numeric, entity, and static slot mismatch penalties
    combined = _apply_candidate_slot_penalties(
        candidate,
        candidate_tokens,
        context,
        combined,
    )

    slots = candidate.parsed_slots
    slot_specificity = len(slots)
    context_adjustment = _context_slot_adjustment(
        slots,
        _context_slot_names(candidate),
        context.normalized_context,
    )
    if context_adjustment > 0.0:
        combined = min(1.0, combined + (CONTEXT_SLOT_MATCH_BOOST * context_adjustment))
    elif context_adjustment < 0.0:
        combined *= CONTEXT_SLOT_MISMATCH_PENALTY ** abs(context_adjustment)

    penalty_val = 0.0
    if rehydrated is not None:
        _, replacements = rehydrated
        wc_len = sum(len(val.split()) for val in replacements.values())
        penalty_val = WILDCARD_LENGTH_PENALTY_FACTOR * wc_len
        combined -= penalty_val
        combined = max(combined, 0.0)

    return _RankedItem(
        final_score=combined,
        candidate=candidate,
        rapidfuzz_score=rapidfuzz_score,
        char_ngram_score=char_score,
        bm25_score=bm25_score,
        intent_score=intent_score,
        index=idx,
        penalty=penalty_val,
        slot_specificity=slot_specificity,
    )


def rank_candidates(
    query: str,
    candidates: Sequence[Candidate],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    *,
    bm25_index: BM25Index | None = None,
    reference_bm25_index: BM25Index | None = None,
    candidate_char_index: CharNGramIndex | None = None,
    positional_literal_tokens: frozenset[str] | None = None,
    rapidfuzz_prefilter_candidates: int = DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES,
    exact_normalized_lookup: dict[str, list[Candidate]] | None = None,
    exact_no_diacritics_lookup: dict[str, list[Candidate]] | None = None,
    language: str | None = None,
    wildcard_always_passes: frozenset[int] | None = None,
    wildcard_variants_with_len: (
        dict[int, tuple[tuple[frozenset[str], int, int], ...]] | None
    ) = None,
    wildcard_token_to_indices: dict[str, tuple[int, ...]] | None = None,
    wildcard_literal_tokens_by_index: dict[int, frozenset[str]] | None = None,
    wildcard_min_required_by_index: dict[int, int] | None = None,
    candidate_slot_tokens: tuple[frozenset[str], ...] | None = None,
    slot_token_to_indices: dict[str, tuple[int, ...]] | None = None,
    slot_preferences: set[tuple[str, str]] | None = None,
    intent_context: Mapping[str, Any] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[RankedCandidate, ...]:
    """Rank candidates for a query using lexical scoring.

    Candidate text containing wildcard placeholders is rehydrated from
    *query* before scoring so that real free-text values contribute to
    the semantic match instead of placeholder tokens.
    ``intent_context`` accepts HassIL-style context mappings and is normalized
    by ``utils.normalize_intent_context``.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    disambiguation_limit = max(2, max_candidates)
    if rapidfuzz_prefilter_candidates < max_candidates:
        raise ValueError("rapidfuzz_prefilter_candidates must be at least max_candidates")
    if not candidates:
        return ()
    if candidate_char_index is not None and len(candidate_char_index.gram_counts) != len(
        candidates
    ):
        raise ValueError("candidate_char_index length must match candidates")
    if candidate_slot_tokens is not None and len(candidate_slot_tokens) != len(candidates):
        raise ValueError("candidate_slot_tokens length must match candidates")

    _rehydrated_cache: dict[int, tuple[str, dict[str, str]]] = {}

    query_normalized = normalize_text(query)
    exact_ranked = _exact_lookup_ranked(
        query,
        query_normalized,
        max_candidates,
        exact_normalized_lookup,
        exact_no_diacritics_lookup,
        language,
    )
    if exact_ranked is not None:
        return exact_ranked

    if positional_literal_tokens is None:
        all_tokens: set[str] = set()
        for candidate in candidates:
            literal_text = candidate.metadata.get("literal_text")
            if literal_text:
                for variant in literal_token_variants(literal_text):
                    all_tokens.update(variant)
        positional_literal_tokens = frozenset(all_tokens)
    (
        query_tokens,
        query_tokens_tuple,
        query_token_count,
        query_sorted,
        non_entity_tokens,
    ) = _rank_query_setup(query_normalized, positional_literal_tokens)
    normalized_context = normalize_intent_context(intent_context)
    query_has_number = _has_numeric_query_token(query_tokens)
    query_numbers = _query_numeric_values(query_tokens)
    intent_score_cache: dict[tuple[frozenset[str], ...], float] = {}

    max_raw_score = 0.0
    if reference_bm25_index is not None:
        bm25_scores = reference_bm25_index.score_custom_documents_tokens(
            query_tokens_tuple, candidates
        )
        max_idx = max(range(len(bm25_scores)), key=bm25_scores.__getitem__, default=0)
        if bm25_scores[max_idx] > 0.0:
            max_cand = candidates[max_idx]
            max_raw_score = reference_bm25_index.raw_score_tokens(
                max_cand.normalized_tokens, query_tokens_tuple
            )
    else:
        if bm25_index is None:
            # Fallback: rebuild the BM25 index on the fly. This occurs during dynamic candidate
            # matching passes where a pre-warmed reference BM25 index is absent due to safety
            # capping of combinatorial expansions. While this introduces a performance trade-off,
            # it is necessary to ensure correct lexical scoring of all candidate templates.
            bm25_index = BM25Index.from_normalized_texts(
                tuple(candidate.normalized_text for candidate in candidates)
            )
        doc_count = len(candidates)
        if not query_tokens_tuple or not doc_count:
            bm25_scores = (0.0,) * doc_count
        else:
            raw_scores = bm25_index.raw_scores(query_tokens_tuple)
            max_raw_score = max(raw_scores, default=0.0)
            bm25_scores = _normalized_bm25_scores_from_raw(raw_scores, doc_count)

    _bm25_ref = reference_bm25_index or bm25_index
    if candidate_char_index is None:
        candidate_char_index = CharNGramIndex.from_grams(
            tuple(char_ngrams_normalized(candidate.normalized_text) for candidate in candidates)
        )
    query_grams = char_ngrams_normalized(query_normalized)
    char_scores = candidate_char_index.score(query_grams)

    prefilter_keys = _rank_prefilter_keys(char_scores, bm25_scores)
    prefilter_limit = _rank_prefilter_limit(
        len(candidates), max_candidates, rapidfuzz_prefilter_candidates
    )
    top_indices = _top_prefilter_indices(prefilter_keys, prefilter_limit)

    wildcard_passed_set = _prefilter_wildcard_candidates(
        candidates,
        query_tokens,
        wildcard_always_passes,
        wildcard_variants_with_len,
        wildcard_token_to_indices,
        wildcard_literal_tokens_by_index,
        wildcard_min_required_by_index,
    )
    if wildcard_passed_set:
        top_set = set(top_indices)
        for wi in wildcard_passed_set:
            if wi not in top_set:
                top_indices.append(wi)

    positional_lookup = (
        _build_positional_lookup(positional_literal_tokens, query_tokens)
        if positional_literal_tokens
        else {}
    )

    if candidate_slot_tokens is None:
        slot_tokens_by_index = {idx: candidates[idx].slot_tokens_set for idx in top_indices}
        active_slot_tokens = frozenset(
            token for tokens in slot_tokens_by_index.values() for token in tokens
        )
        query_slot_tokens = query_tokens & active_slot_tokens
    elif slot_token_to_indices is not None:
        slot_tokens_by_index = {}
        query_slot_tokens = _query_slot_tokens_from_index(
            query_tokens,
            top_indices,
            slot_token_to_indices,
        )
    else:
        slot_tokens_by_index = {}
        active_slot_tokens = frozenset(
            token for idx in top_indices for token in candidate_slot_tokens[idx]
        )
        query_slot_tokens = query_tokens & active_slot_tokens

    literal_analysis_cache: dict[
        tuple[str, tuple[frozenset[str], ...]], _LiteralVariantAnalysis
    ] = {}
    ranked_tuples: list[_RankedItem] = []
    context = _ScoringContext(
        query=query,
        query_normalized=query_normalized,
        query_tokens=query_tokens,
        query_tokens_tuple=query_tokens_tuple,
        query_token_count=query_token_count,
        query_sorted=query_sorted,
        query_grams=query_grams,
        query_slot_tokens=query_slot_tokens,
        query_has_number=query_has_number,
        query_numbers=query_numbers,
        bm25_ref=_bm25_ref,
        max_raw_score=max_raw_score,
        positional_lookup=positional_lookup,
        positional_literal_tokens=positional_literal_tokens,
        non_entity_tokens=non_entity_tokens,
        candidate_slot_tokens=candidate_slot_tokens,
        slot_tokens_by_index=slot_tokens_by_index,
        min_confidence=min_confidence,
        normalized_context=normalized_context,
        wildcard_passed_set=wildcard_passed_set,
        rehydrated_cache=_rehydrated_cache,
        intent_score_cache=intent_score_cache,
        literal_analysis_cache=literal_analysis_cache,
    )
    for idx in top_indices:
        candidate = candidates[idx]
        bm25_score = bm25_scores[idx]
        char_score = char_scores[idx]
        item = _score_single_candidate(
            idx,
            candidate,
            bm25_score,
            char_score,
            context,
        )
        if item is not None:
            ranked_tuples.append(item)

    intent_tie_preferences = _intent_tie_preferences_by_index(
        ranked_tuples,
        slot_preferences=slot_preferences,
        rehydrated_cache=_rehydrated_cache,
    )
    ranked_tuples.sort(
        key=partial(
            _ranked_tuple_sort_key,
            slot_preferences=slot_preferences,
            rehydrated_cache=_rehydrated_cache,
            intent_tie_preferences=intent_tie_preferences,
        ),
        reverse=True,
    )
    ranked = [
        RankedCandidate(
            candidate=item.candidate,
            scores=ScoreBreakdown(
                rapidfuzz_score=item.rapidfuzz_score,
                char_ngram_score=item.char_ngram_score,
                bm25_score=item.bm25_score,
                intent_score=item.intent_score,
                final_score=item.final_score,
                penalty=item.penalty,
            ),
        )
        for item in ranked_tuples[:disambiguation_limit]
    ]
    _apply_intent_disambiguation(ranked)
    return tuple(ranked[:max_candidates])


# ---------------------------------------------------------------------------
# Ranked-item sort helpers
# ---------------------------------------------------------------------------


def _candidate_slot_tokens_at(
    index: int,
    candidate_slot_tokens: tuple[frozenset[str], ...] | None,
    slot_tokens_by_index: dict[int, frozenset[str]],
) -> frozenset[str]:
    """Return precomputed slot tokens for a candidate index."""
    if candidate_slot_tokens is not None:
        return candidate_slot_tokens[index]
    return slot_tokens_by_index.get(index, frozenset())


def _ranked_tuple_slot_preference(
    item: _RankedItem,
    *,
    slot_preferences: set[tuple[str, str]] | None,
    rehydrated_cache: dict[int, tuple[str, dict[str, str]]],
) -> float:
    """Return the slot-preference boost for an intermediate score tuple."""
    idx = item.index
    if not slot_preferences or idx not in rehydrated_cache:
        return 0.0
    _, replacements = rehydrated_cache[idx]
    return next(
        (
            1.0
            for slot_name, val in replacements.items()
            if (slot_name, val.lower()) in slot_preferences
        ),
        0.0,
    )


def _intent_tie_preferences_by_index(
    ranked_tuples: Sequence[_RankedItem],
    *,
    slot_preferences: set[tuple[str, str]] | None,
    rehydrated_cache: dict[int, tuple[str, dict[str, str]]],
) -> dict[int, tuple[int, float, float]]:
    """Return intent preferences for known opposing exact structural tie groups.

    Near-ties keep their numeric score ordering; this preference only resolves
    candidates tied within float noise on final score, slot preference, and
    slot specificity. Values map candidate index to ``(preference,
    normalized_final_score, normalized_slot_preference)`` so the final sort key
    treats epsilon-different structural ties as equal.
    """
    tie_groups: list[tuple[float, float, int, list[_RankedItem]]] = []
    for item in ranked_tuples:
        slot_preference = _ranked_tuple_slot_preference(
            item,
            slot_preferences=slot_preferences,
            rehydrated_cache=rehydrated_cache,
        )
        for group_final, group_slot_preference, group_specificity, group in tie_groups:
            if (
                item.slot_specificity == group_specificity
                and isclose(
                    item.final_score,
                    group_final,
                    rel_tol=0.0,
                    abs_tol=_STRUCTURAL_TIE_ABS_TOLERANCE,
                )
                and isclose(
                    slot_preference,
                    group_slot_preference,
                    rel_tol=0.0,
                    abs_tol=_STRUCTURAL_TIE_ABS_TOLERANCE,
                )
            ):
                group.append(item)
                break
        else:
            tie_groups.append((item.final_score, slot_preference, item.slot_specificity, [item]))

    preferences_by_index: dict[int, tuple[int, float, float]] = {}
    for group_final, group_slot_preference, _, group in tie_groups:
        if len(group) < 2:
            continue
        tied_intents = frozenset(item.candidate.intent_name for item in group)
        group_preferences = {
            item.index: _intent_tie_preference(item.candidate, tied_intents) for item in group
        }
        if not any(group_preferences.values()):
            continue
        for item in group:
            preferences_by_index[item.index] = (
                group_preferences[item.index],
                group_final,
                group_slot_preference,
            )
    return preferences_by_index


def _ranked_tuple_sort_key(
    item: _RankedItem,
    *,
    slot_preferences: set[tuple[str, str]] | None,
    rehydrated_cache: dict[int, tuple[str, dict[str, str]]],
    intent_tie_preferences: Mapping[int, tuple[int, float, float]],
) -> tuple[float, float, int, int, int]:
    """Return the stable ranking sort key for an intermediate score tuple."""
    idx = item.index
    slot_specificity = item.slot_specificity
    tb = _ranked_tuple_slot_preference(
        item,
        slot_preferences=slot_preferences,
        rehydrated_cache=rehydrated_cache,
    )
    final_score = item.final_score
    intent_tie_preference = 0
    if tie_adjustment := intent_tie_preferences.get(idx):
        intent_tie_preference, final_score, tb = tie_adjustment

    return (final_score, tb, slot_specificity, intent_tie_preference, -idx)


def _intent_tie_preference(candidate: Candidate, tied_intents: frozenset[str]) -> int:
    """Return deterministic preference only for known opposing intent ties."""
    return next(
        (
            preferences.get(candidate.intent_name, 0)
            for intent_names, preferences in _KNOWN_OPPOSING_INTENT_TIE_PREFERENCES.items()
            if tied_intents == intent_names
        ),
        0,
    )


def _apply_intent_disambiguation(
    ranked: list[RankedCandidate],
    tiebreaker_margin: float = TIEBREAKER_INTENT_MARGIN,
) -> None:
    """Re-rank when a different-intent candidate is nearly tied with the top.

    When the top two candidates belong to different intents and their score
    gap is less than ``tiebreaker_margin``, the second candidate is promoted
    to position 0 if it has a higher intent_score.
    """
    if len(ranked) < 2:
        return
    top = ranked[0]
    competitor = ranked[1]
    if competitor.candidate.intent_name == top.candidate.intent_name:
        return
    margin = top.scores.final_score - competitor.scores.final_score
    if margin > tiebreaker_margin:
        return
    if competitor.scores.intent_score > top.scores.intent_score:
        ranked[0], ranked[1] = competitor, top


def _passes_high_confidence_relaxed_margin(
    top_candidate: RankedCandidate,
    margin: float,
    min_margin: float,
) -> bool:
    """Return whether a high-confidence top candidate clears the relaxed margin gate."""
    if min_margin > DEFAULT_MIN_MARGIN:
        return False
    return (
        top_candidate.scores.final_score >= HIGH_CONFIDENCE_RELAXED_MIN_SCORE
        and margin >= HIGH_CONFIDENCE_RELAXED_MIN_MARGIN
    )


def _has_weak_zero_intent_evidence(
    top_candidate: RankedCandidate,
    margin: float,
) -> bool:
    """Return whether a fuzzy top match has no action evidence and a close rival."""
    return (
        top_candidate.scores.intent_score <= 0.0
        and top_candidate.scores.final_score < ZERO_INTENT_EVIDENCE_MAX_SCORE
        and margin <= ZERO_INTENT_EVIDENCE_MAX_MARGIN
    )


def _uses_broad_static_entity(candidate: Candidate) -> bool:
    """Return whether a candidate uses static broad-domain entity expansion."""
    static_slots_text = candidate.metadata.get("static_slots", "")
    static_slots = (
        {slot for slot in static_slots_text.split(",") if slot}
        if isinstance(static_slots_text, str)
        else set()
    )
    return {"domain", "name"} <= static_slots


def _has_safe_relaxed_intent_evidence(
    top_candidate: RankedCandidate,
    competing_candidate: RankedCandidate,
    margin: float,
    min_margin: float,
) -> bool:
    """Return whether a close fuzzy winner has enough low-risk evidence."""
    if min_margin > DEFAULT_MIN_MARGIN:
        return False
    top_slots = top_candidate.candidate.parsed_slots
    competing_slots = competing_candidate.candidate.parsed_slots
    if (
        not top_slots
        and not competing_slots
        and top_candidate.scores.final_score >= SAFE_EMPTY_SLOT_RELAXED_MIN_SCORE
        and margin >= SAFE_EMPTY_SLOT_RELAXED_MIN_MARGIN
    ):
        return True
    intent_advantage = top_candidate.scores.intent_score - competing_candidate.scores.intent_score
    return (
        intent_advantage >= SAFE_INTENT_EVIDENCE_MIN_ADVANTAGE
        and top_candidate.scores.final_score >= SAFE_INTENT_EVIDENCE_MIN_SCORE
        and top_candidate.scores.final_score < SAFE_INTENT_EVIDENCE_MAX_SCORE
        and top_candidate.candidate.metadata.get("query_slots") != "name"
        and not _uses_broad_static_entity(top_candidate.candidate)
    )


def _is_turn_on_off_same_slot_tie(
    top_candidate: RankedCandidate,
    competing_candidate: RankedCandidate,
) -> bool:
    """Return whether an exact TurnOn/TurnOff tie has identical extracted slots."""
    if top_candidate.candidate.intent_name != "HassTurnOn":
        return False
    if competing_candidate.candidate.intent_name != "HassTurnOff":
        return False
    if not isclose(
        top_candidate.scores.final_score,
        competing_candidate.scores.final_score,
        rel_tol=0.0,
        abs_tol=_STRUCTURAL_TIE_ABS_TOLERANCE,
    ):
        return False
    return top_candidate.candidate.parsed_slots == competing_candidate.candidate.parsed_slots


def _is_same_text_same_slot_competitor(
    candidate: RankedCandidate,
    other: RankedCandidate,
) -> bool:
    """Return whether a same-text competitor preserves the same slot payload."""
    return (
        other.candidate.normalized_text == candidate.candidate.normalized_text
        and other.candidate.parsed_slots == candidate.candidate.parsed_slots
    )


def evaluate_confidence_gates(
    ranked: Sequence[RankedCandidate],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> tuple[RankedCandidate | None, FallbackReason | None]:
    """Evaluate confidence gates and return the accepted candidate.

    Returns the accepted candidate and the fallback reason if rejected.
    """
    if not ranked:
        return None, FallbackReason.NO_CANDIDATE
    top_candidate = ranked[0]
    if top_candidate.scores.final_score < min_confidence:
        return None, FallbackReason.LOW_CONFIDENCE
    competing_candidate = next(
        (
            item
            for item in ranked[1:]
            if item.candidate.intent_name != top_candidate.candidate.intent_name
            and not _is_same_text_same_slot_competitor(top_candidate, item)
        ),
        None,
    )
    if competing_candidate is None:
        return top_candidate, None
    if _is_exact_lexical_match(top_candidate):
        return top_candidate, None
    if _is_turn_on_off_same_slot_tie(top_candidate, competing_candidate):
        return top_candidate, None
    margin = top_candidate.scores.final_score - competing_candidate.scores.final_score
    if _has_weak_zero_intent_evidence(top_candidate, margin):
        return None, FallbackReason.LOW_CONFIDENCE
    if _has_safe_relaxed_intent_evidence(
        top_candidate,
        competing_candidate,
        margin,
        min_margin,
    ):
        return top_candidate, None
    if _passes_high_confidence_relaxed_margin(top_candidate, margin, min_margin):
        return top_candidate, None
    if margin < min_margin:
        return None, FallbackReason.LOW_MARGIN
    return top_candidate, None


def accepted_candidate(
    ranked: Sequence[RankedCandidate],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> RankedCandidate | None:
    """Return the accepted top candidate or None when confidence gates reject it."""
    candidate, _ = evaluate_confidence_gates(ranked, min_confidence, min_margin)
    return candidate


def confidence_gate_rejection_reason(
    ranked: Sequence[RankedCandidate],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> FallbackReason:
    """Return the fallback reason for candidates rejected by confidence gates."""
    _, reason = evaluate_confidence_gates(ranked, min_confidence, min_margin)
    return reason or FallbackReason.LOW_CONFIDENCE


def _is_exact_lexical_match(ranked_candidate: RankedCandidate) -> bool:
    """Return whether a ranked candidate exactly matches query text lexically."""
    scores = ranked_candidate.scores
    return scores.rapidfuzz_score == 1.0 and scores.char_ngram_score == 1.0


def clear_ranking_caches() -> None:
    """Clear all global LRU caches in ranking module."""
    _raw_cached_fuzz_ratio.cache_clear()
