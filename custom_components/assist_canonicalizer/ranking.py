"""Lexical candidate ranking."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from heapq import nsmallest

from rapidfuzz import fuzz

from .bm25 import BM25Index
from .candidate import Candidate
from .const import (
    BM25_WEIGHT,
    CHAR_NGRAM_WEIGHT,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES,
    INTENT_ACTION_WEIGHT,
    NON_ENTITY_PENALTY_BLEND,
    POSITIONAL_SIMILARITY_BASE_THRESHOLD,
    POSITIONAL_SIMILARITY_MEDIUM_THRESHOLD,
    POSITIONAL_SIMILARITY_PARTIAL_CREDIT,
    POSITIONAL_SIMILARITY_SHORT_3_THRESHOLD,
    POSITIONAL_SIMILARITY_VERY_SHORT_THRESHOLD,
    RAPIDFUZZ_WEIGHT,
    SLOT_TOKEN_MATCH_THRESHOLD,
    TIEBREAKER_INTENT_MARGIN,
    WILDCARD_LENGTH_PENALTY_FACTOR,
    FallbackReason,
)
from .normalization import (
    char_ngrams_normalized,
    literal_token_variants,
    normalize_text,
    normalize_text_no_diacritics,
)
from .rehydration import get_wildcard_rehydration, wildcard_variants_analysis

_LiteralVariantAnalysis = list[tuple[int, int, list[frozenset[str]]]]


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
        token_sort: float = fuzz.ratio(query_sorted, candidate_sorted)
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
            if sim >= _per_pair_positional_threshold(literal_token, qtok):
                matched.append(qtok)
                if sim >= 0.99:
                    break
        if matched:
            lookup[literal_token] = frozenset(matched)
    return lookup


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
    """Return how well a candidate covers query tokens that no entity contains.

    Tokens in the query that do not appear in any candidate's entity slots
    (``positional_literal_tokens`` represents all known entity/literal tokens)
    are typically politeness words, filler words, or action synonyms.  A
    candidate whose tokens cover more of these should be preferred.
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
    """Return query tokens that appear in selected candidates' indexed slot tokens."""
    top_index_set = set(top_indices)
    return frozenset(
        token
        for token in query_tokens
        if any(idx in top_index_set for idx in slot_token_to_indices.get(token, ()))
    )


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
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[RankedCandidate, ...]:
    """Rank candidates for a query using lexical scoring.

    Candidate text containing wildcard placeholders is rehydrated from
    *query* before scoring so that real free-text values contribute to
    the semantic match instead of placeholder tokens.
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
            query_tokens, top_indices, slot_token_to_indices
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
    ranked_tuples: list[tuple[float, Candidate, float, float, float, float, int, float]] = []
    for idx in top_indices:
        candidate = candidates[idx]
        bm25_score = bm25_scores[idx]
        char_score = char_scores[idx]
        if candidate.has_wildcard:
            if idx not in wildcard_passed_set:
                continue
            rehydrated_norm, replacements, char_score, bm25_score = _rehydrate_and_rescore_wildcard(
                candidate,
                query,
                query_tokens_tuple,
                query_grams,
                _bm25_ref,
                max_raw_score,
                char_score,
                bm25_score,
            )
            if not replacements or rehydrated_norm is None:
                continue
            _rehydrated_cache[idx] = (rehydrated_norm, replacements)
        if idx in _rehydrated_cache:
            cand_text, replacements = _rehydrated_cache[idx]
            cand_tokens_list = cand_text.split()
            cand_sorted = " ".join(sorted(cand_tokens_list))
            cand_token_count = len(cand_tokens_list)
            candidate_tokens = frozenset(cand_tokens_list)

            norm_replacements = {
                wc: normalize_text(val).split() for wc, val in replacements.items()
            }
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

        rapidfuzz_score = rapidfuzz_similarity_normalized(
            query_normalized,
            cand_text,
            query_token_count=query_token_count,
            query_sorted=query_sorted,
            candidate_sorted=cand_sorted,
            candidate_token_count=cand_token_count,
        )
        literal_text = candidate.metadata.get("literal_text")
        coverage = _query_token_coverage(query_tokens, candidate_tokens)
        intent_score = coverage
        if literal_text:
            exact = intent_score_cache.get(literal_variants)
            if exact is None:
                exact = _exact_intent_score(literal_variants, query_tokens)
                intent_score_cache[literal_variants] = exact
            if exact >= 1.0:
                matched_non_empty = any(
                    var and var.issubset(query_tokens) for var in literal_variants
                )
                if not matched_non_empty and not query_tokens.issubset(candidate_tokens):
                    exact = 0.0
            if exact >= 1.0:
                if total_unique_literal_tokens >= 2:
                    matched_q = len(query_tokens & candidate_tokens)
                    intent_score = matched_q / len(query_tokens) if query_tokens else 1.0
                elif candidate.slot_tokens_set:
                    intent_score = exact
            elif query_tokens.issubset(candidate_tokens) or not positional_lookup:
                intent_score = exact
            else:
                analysis_key = (literal_text, literal_variants)
                analysis = literal_analysis_cache.get(analysis_key)
                if analysis is None:
                    analysis = _precompute_literal_analysis(
                        literal_variants, query_tokens, positional_lookup
                    )
                    literal_analysis_cache[analysis_key] = analysis
                best = 0.0
                for total_len, exact_count, positional_hits in analysis:
                    matched = float(exact_count)
                    for hits in positional_hits:
                        if not hits.issubset(candidate_tokens):
                            matched += POSITIONAL_SIMILARITY_PARTIAL_CREDIT
                    score = matched / total_len if total_len else 0.0
                    if score > best:
                        best = score
                    intent_score = best
            if positional_literal_tokens:
                penalty = _non_entity_coverage(
                    query_tokens,
                    positional_literal_tokens,
                    candidate_tokens,
                    non_entity=non_entity_tokens,
                )
                intent_score *= 1.0 - NON_ENTITY_PENALTY_BLEND + NON_ENTITY_PENALTY_BLEND * penalty
        combined = lexical_score(rapidfuzz_score, char_score, bm25_score, intent_score)

        # Slot matching penalty
        slot_penalty = 1.0
        cand_slot_tokens = _candidate_slot_tokens_at(
            idx,
            candidate_slot_tokens,
            slot_tokens_by_index,
        )
        wildcard_info = candidate.wildcard_info
        wildcard_tokens = frozenset()
        leading_placeholder_only_wildcard = False
        if cand_slot_tokens and idx in _rehydrated_cache and wildcard_info is not None:
            _, replacements = _rehydrated_cache[idx]
            wildcard_index, wildcard_name = wildcard_info
            placeholder_tokens = frozenset(normalize_text(wildcard_name).split())
            cand_slot_tokens = cand_slot_tokens - placeholder_tokens
            leading_placeholder_only_wildcard = wildcard_index == 0 and not cand_slot_tokens
            if wildcard_index > 0 or cand_slot_tokens:
                wildcard_tokens = frozenset(
                    token
                    for wildcard_value in replacements.values()
                    for token in normalize_text(wildcard_value).split()
                )
        if cand_slot_tokens and query_slot_tokens:
            has_conflict = False
            allowed_cand_tokens = (
                cand_slot_tokens | wildcard_tokens | candidate.normalized_tokens_set
            )
            for q_tok in query_slot_tokens:
                is_matched = q_tok in allowed_cand_tokens or any(
                    fuzz.ratio(q_tok, c_tok) >= SLOT_TOKEN_MATCH_THRESHOLD
                    for c_tok in allowed_cand_tokens
                )
                if not is_matched:
                    has_conflict = True
                    break

            if has_conflict:
                cand_matched = 0
                for c_tok in cand_slot_tokens:
                    if c_tok in query_tokens_tuple or any(
                        fuzz.ratio(c_tok, q_tok) >= SLOT_TOKEN_MATCH_THRESHOLD
                        for q_tok in query_tokens_tuple
                    ):
                        cand_matched += 1
                cand_coverage = cand_matched / len(cand_slot_tokens)

                query_matched = 0
                for q_tok in query_slot_tokens:
                    if q_tok in allowed_cand_tokens or any(
                        fuzz.ratio(q_tok, c_tok) >= SLOT_TOKEN_MATCH_THRESHOLD
                        for c_tok in allowed_cand_tokens
                    ):
                        query_matched += 1
                query_coverage = query_matched / len(query_slot_tokens)

                slot_penalty = cand_coverage * (0.8 + 0.2 * query_coverage)
        elif leading_placeholder_only_wildcard and query_slot_tokens:
            allowed_cand_tokens = wildcard_tokens | candidate.normalized_tokens_set
            has_conflict = any(
                q_tok not in allowed_cand_tokens
                and all(
                    fuzz.ratio(q_tok, c_tok) < SLOT_TOKEN_MATCH_THRESHOLD
                    for c_tok in allowed_cand_tokens
                )
                for q_tok in query_slot_tokens
            )
            if has_conflict:
                slot_penalty = 0.0

        # Apply penalty only if less than 100% of slot tokens match
        if slot_penalty < 1.0:
            base_multiplier = min(1.0, max(0.1, min_confidence - 0.05))
            combined *= base_multiplier + (1.0 - base_multiplier) * slot_penalty

        penalty_val = 0.0
        if idx in _rehydrated_cache:
            _, replacements = _rehydrated_cache[idx]
            wc_len = sum(len(val.split()) for val in replacements.values())
            penalty_val = WILDCARD_LENGTH_PENALTY_FACTOR * wc_len
            combined -= penalty_val
            combined = max(combined, 0.0)
        ranked_tuples.append(
            (
                combined,
                candidate,
                rapidfuzz_score,
                char_score,
                bm25_score,
                intent_score,
                idx,
                penalty_val,
            )
        )

    ranked_tuples.sort(
        key=partial(
            _ranked_tuple_sort_key,
            slot_preferences=slot_preferences,
            rehydrated_cache=_rehydrated_cache,
        ),
        reverse=True,
    )
    ranked = [
        RankedCandidate(
            candidate=item[1],
            scores=ScoreBreakdown(
                rapidfuzz_score=item[2],
                char_ngram_score=item[3],
                bm25_score=item[4],
                intent_score=item[5],
                final_score=item[0],
                penalty=item[7],
            ),
        )
        for item in ranked_tuples[:disambiguation_limit]
    ]
    _apply_intent_disambiguation(ranked)
    return tuple(ranked[:max_candidates])


def _candidate_slot_tokens_at(
    index: int,
    candidate_slot_tokens: tuple[frozenset[str], ...] | None,
    slot_tokens_by_index: dict[int, frozenset[str]],
) -> frozenset[str]:
    """Return precomputed slot tokens for a candidate index."""
    if candidate_slot_tokens is not None:
        return candidate_slot_tokens[index]
    return slot_tokens_by_index.get(index, frozenset())


def _ranked_tuple_sort_key(
    item: tuple[float, Candidate, float, float, float, float, int, float],
    *,
    slot_preferences: set[tuple[str, str]] | None,
    rehydrated_cache: dict[int, tuple[str, dict[str, str]]],
) -> tuple[float, float, int]:
    """Return the stable ranking sort key for an intermediate score tuple."""
    idx = item[6]
    tb = 0.0
    if slot_preferences and idx in rehydrated_cache:
        _, replacements = rehydrated_cache[idx]
        for slot_name, val in replacements.items():
            if (slot_name, val.lower()) in slot_preferences:
                tb = 1.0
                break

    return (item[0], tb, -idx)


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


def accepted_candidate(
    ranked: Sequence[RankedCandidate],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> RankedCandidate | None:
    """Return the accepted top candidate or None when confidence gates reject it."""
    if not ranked:
        return None
    top_candidate = ranked[0]
    if top_candidate.scores.final_score < min_confidence:
        return None
    competing_candidate = next(
        (
            item
            for item in ranked[1:]
            if item.candidate.intent_name != top_candidate.candidate.intent_name
            and item.candidate.normalized_text != top_candidate.candidate.normalized_text
        ),
        None,
    )
    if competing_candidate is None:
        return top_candidate
    if _is_exact_lexical_match(top_candidate):
        return top_candidate
    margin = top_candidate.scores.final_score - competing_candidate.scores.final_score
    return None if margin < min_margin else top_candidate


def confidence_gate_rejection_reason(
    ranked: Sequence[RankedCandidate],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> FallbackReason:
    """Return the fallback reason for candidates rejected by confidence gates."""
    if not ranked or ranked[0].scores.final_score < min_confidence:
        return FallbackReason.LOW_CONFIDENCE
    top_candidate = ranked[0]
    competing_candidate = next(
        (
            item
            for item in ranked[1:]
            if item.candidate.intent_name != top_candidate.candidate.intent_name
            and item.candidate.normalized_text != top_candidate.candidate.normalized_text
        ),
        None,
    )
    if competing_candidate is None or _is_exact_lexical_match(top_candidate):
        return FallbackReason.LOW_CONFIDENCE
    margin = top_candidate.scores.final_score - competing_candidate.scores.final_score
    return FallbackReason.LOW_MARGIN if margin < min_margin else FallbackReason.LOW_CONFIDENCE


def _is_exact_lexical_match(ranked_candidate: RankedCandidate) -> bool:
    """Return whether a ranked candidate exactly matches query text lexically."""
    scores = ranked_candidate.scores
    return scores.rapidfuzz_score == 1.0 and scores.char_ngram_score == 1.0
