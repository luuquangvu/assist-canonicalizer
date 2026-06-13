"""Lexical candidate ranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
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
    TIEBREAKER_INTENT_MARGIN,
)
from .normalization import (
    char_ngrams_normalized,
    normalize_text,
    normalize_text_no_diacritics,
)


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Normalized score components for a candidate."""

    rapidfuzz_score: float
    char_ngram_score: float
    bm25_score: float
    intent_score: float
    final_score: float


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
        postings: dict[str, list[int]] = {}
        for index, grams in enumerate(candidate_grams):
            for gram in grams:
                postings.setdefault(gram, []).append(index)
        return cls(
            gram_counts=tuple(len(grams) for grams in candidate_grams),
            postings={gram: tuple(indexes) for gram, indexes in postings.items()},
        )

    def score(self, query_grams: frozenset[str]) -> tuple[float, ...]:
        """Return exact normalized Jaccard scores for all indexed candidates."""
        if not query_grams:
            return tuple(0.0 for _ in self.gram_counts)
        intersections = [0] * len(self.gram_counts)
        for gram in query_grams:
            for index in self.postings.get(gram, ()):
                intersections[index] += 1
        query_count = len(query_grams)
        scores = [0.0] * len(self.gram_counts)
        for index, intersection_size in enumerate(intersections):
            if intersection_size == 0:
                continue
            union_size = query_count + self.gram_counts[index] - intersection_size
            scores[index] = intersection_size / union_size if union_size else 0.0
        return tuple(scores)


def rapidfuzz_similarity_normalized(query: str, candidate: str) -> float:
    """Return a RapidFuzz score that penalizes unmatched extra tokens."""
    if not query or not candidate:
        return 0.0
    wratio = float(fuzz.WRatio(query, candidate))
    token_sort = float(fuzz.token_sort_ratio(query, candidate))
    token_set = float(fuzz.token_set_ratio(query, candidate))
    token_set *= token_count_ratio(query, candidate)
    return (wratio + token_sort + token_set) / 300.0


def token_count_ratio(query: str, candidate: str) -> float:
    """Return a length ratio that penalizes unmatched extra tokens."""
    query_count = len(query.split())
    candidate_count = len(candidate.split())
    if query_count == 0 or candidate_count == 0:
        return 0.0
    return min(query_count, candidate_count) / max(query_count, candidate_count)


def char_ngram_similarity_from_grams(
    query_grams: frozenset[str],
    candidate_grams: frozenset[str],
) -> float:
    """Return character n-gram Jaccard similarity from precomputed grams."""
    if not query_grams or not candidate_grams:
        return 0.0
    intersection_size = len(query_grams & candidate_grams)
    union_size = len(query_grams) + len(candidate_grams) - intersection_size
    return intersection_size / union_size if union_size else 0.0


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
    """Character-level positional similarity — cheap edit-distance proxy."""
    if a == b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    matches = 0
    for c1, c2 in zip(a, b, strict=False):
        if c1 == c2:
            matches += 1
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
    literal_text: str,
    query_tokens: frozenset[str],
) -> float:
    """Return how well query tokens cover localized template literal words."""
    variants = _literal_token_variants(literal_text)
    if not variants:
        return 1.0
    max_score = 0.0
    for literal_tokens in variants:
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
    loop — positional similarity requires at least the first character to
    match, so tokens that differ at position 0 can never reach any
    non-trivial positional similarity threshold.
    """
    first_char_index: dict[str, set[str]] = {}
    for qtok in query_tokens:
        if qtok:
            first_char_index.setdefault(qtok[0], set()).add(qtok)

    lookup: dict[str, frozenset[str]] = {}
    for literal_token in literal_tokens_set:
        if literal_token in query_tokens:
            continue
        candidate_q_tokens = (
            first_char_index.get(literal_token[0], set()) if literal_token else set()
        )
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
    variants = _literal_token_variants(literal_text)
    if not variants:
        return 1.0
    if candidate_entity is not None and not query_tokens - candidate_entity:
        return _exact_intent_score(literal_text, query_tokens)
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
) -> float:
    """Return how well a candidate covers query tokens that no entity contains.

    Tokens in the query that do not appear in any candidate's entity slots
    (``positional_literal_tokens`` represents all known entity/literal tokens)
    are typically politeness words, filler words, or action synonyms.  A
    candidate whose tokens cover more of these should be preferred.
    """
    non_entity = query_tokens - positional_literal_tokens
    if not non_entity:
        return 1.0
    matched = sum(1 for token in non_entity if token in candidate_tokens)
    coverage = matched / len(non_entity)

    return coverage * coverage


@lru_cache(maxsize=8192)
def _literal_token_variants(literal_text: str) -> tuple[frozenset[str], ...]:
    """Return normalized literal token variants for intent action scoring."""
    variants = []
    for variant in literal_text.split("|"):
        if not variant.strip():
            continue
        literal_tokens = frozenset(normalize_text(variant).split())
        if literal_tokens:
            variants.append(literal_tokens)
    return tuple(variants)


def rank_candidates(
    query: str,
    candidates: Sequence[Candidate],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    *,
    bm25_index: BM25Index | None = None,
    reference_bm25_index: BM25Index | None = None,
    candidate_char_grams: Sequence[frozenset[str]] | None = None,
    candidate_char_index: CharNGramIndex | None = None,
    positional_literal_tokens: frozenset[str] | None = None,
    rapidfuzz_prefilter_candidates: int = DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES,
    exact_normalized_lookup: dict[str, list[Candidate]] | None = None,
    exact_no_diacritics_lookup: dict[str, list[Candidate]] | None = None,
    language: str | None = None,
) -> tuple[RankedCandidate, ...]:
    """Rank candidates for a query using lexical scoring."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if rapidfuzz_prefilter_candidates < max_candidates:
        raise ValueError("rapidfuzz_prefilter_candidates must be at least max_candidates")
    if not candidates:
        return ()
    if candidate_char_grams is not None and len(candidate_char_grams) != len(candidates):
        raise ValueError("candidate_char_grams length must match candidates")
    if candidate_char_index is not None and len(candidate_char_index.gram_counts) != len(
        candidates
    ):
        raise ValueError("candidate_char_index length must match candidates")

    query_normalized = normalize_text(query)

    # 1. Exact normalized match check
    if exact_normalized_lookup is not None:
        exact_matches = exact_normalized_lookup.get(query_normalized)
        if exact_matches:
            return tuple(
                RankedCandidate(
                    candidate=c,
                    scores=ScoreBreakdown(
                        rapidfuzz_score=1.0,
                        char_ngram_score=1.0,
                        bm25_score=1.0,
                        intent_score=1.0,
                        final_score=1.0,
                    ),
                )
                for c in exact_matches[:max_candidates]
            )

    # 2. Exact no-diacritics match check
    if exact_no_diacritics_lookup is not None:
        query_no_diac = normalize_text_no_diacritics(query, language)
        no_diac_matches = exact_no_diacritics_lookup.get(query_no_diac)
        if no_diac_matches:
            unique_intents = {c.intent_name for c in no_diac_matches}
            if len(unique_intents) == 1:
                return tuple(
                    RankedCandidate(
                        candidate=c,
                        scores=ScoreBreakdown(
                            rapidfuzz_score=1.0,
                            char_ngram_score=1.0,
                            bm25_score=1.0,
                            intent_score=1.0,
                            final_score=1.0,
                        ),
                    )
                    for c in no_diac_matches[:max_candidates]
                )
    query_grams = char_ngrams_normalized(query_normalized)
    query_tokens = frozenset(query_normalized.split())
    intent_score_cache: dict[str, float] = {}
    if positional_literal_tokens is None:
        all_tokens: set[str] = set()
        for candidate in candidates:
            literal_text = candidate.metadata.get("literal_text")
            if literal_text:
                for variant in _literal_token_variants(literal_text):
                    all_tokens.update(variant)
        positional_literal_tokens = frozenset(all_tokens)

    if reference_bm25_index is not None:
        bm25_scores = reference_bm25_index.score_custom_documents(
            query_normalized, tuple(candidate.normalized_text for candidate in candidates)
        )
    elif bm25_index is None:
        bm25_index = BM25Index.from_normalized_texts(
            tuple(candidate.normalized_text for candidate in candidates)
        )
        bm25_scores = bm25_index.score(query_normalized)
    else:
        bm25_scores = bm25_index.score(query_normalized)
    if candidate_char_index is not None:
        char_scores = candidate_char_index.score(query_grams)
    else:
        char_scores = tuple(
            char_ngram_similarity_from_grams(
                query_grams,
                char_ngrams_normalized(candidate.normalized_text)
                if candidate_char_grams is None
                else candidate_char_grams[index],
            )
            for index, candidate in enumerate(candidates)
        )

    prefilter_keys = [
        -(CHAR_NGRAM_WEIGHT * cs + BM25_WEIGHT * bs)
        for cs, bs in zip(char_scores, bm25_scores, strict=True)
    ]
    prefilter_limit = min(len(candidates), rapidfuzz_prefilter_candidates)
    top_indices = nsmallest(
        prefilter_limit, range(len(prefilter_keys)), key=lambda i: prefilter_keys[i]
    )
    prefiltered_literal_tokens = set()
    for idx in top_indices:
        literal_text = candidates[idx].metadata.get("literal_text")
        if literal_text:
            for variant in _literal_token_variants(literal_text):
                prefiltered_literal_tokens.update(variant)
    positional_lookup = (
        _build_positional_lookup(frozenset(prefiltered_literal_tokens), query_tokens)
        if prefiltered_literal_tokens
        else {}
    )
    ranked: list[RankedCandidate] = []
    for idx in top_indices:
        candidate = candidates[idx]
        bm25_score = bm25_scores[idx]
        char_score = char_scores[idx]
        rapidfuzz_score = rapidfuzz_similarity_normalized(
            query_normalized, candidate.normalized_text
        )
        literal_text = candidate.metadata.get("literal_text")
        candidate_tokens = candidate.normalized_tokens_set
        coverage = _query_token_coverage(query_tokens, candidate_tokens)
        intent_score = coverage
        if literal_text:
            exact = intent_score_cache.get(literal_text)
            if exact is None:
                exact = _exact_intent_score(literal_text, query_tokens)
                intent_score_cache[literal_text] = exact
            if exact >= 1.0:
                variants = _literal_token_variants(literal_text)
                total_unique = len({tok for var in variants for tok in var})
                if total_unique >= 2:
                    matched_q = sum(1 for t in query_tokens if t in candidate_tokens)
                    unsq = matched_q / len(query_tokens) if query_tokens else 1.0
                    intent_score = max(coverage, unsq)
            else:
                intent_score = _positional_intent_score_from_lookup(
                    literal_text, query_tokens, positional_lookup, candidate_tokens
                )
            if positional_literal_tokens:
                penalty = _non_entity_coverage(
                    query_tokens,
                    positional_literal_tokens,
                    candidate_tokens,
                )
                intent_score *= 1.0 - NON_ENTITY_PENALTY_BLEND + NON_ENTITY_PENALTY_BLEND * penalty
        combined = lexical_score(rapidfuzz_score, char_score, bm25_score, intent_score)
        scores = ScoreBreakdown(
            rapidfuzz_score=rapidfuzz_score,
            char_ngram_score=char_score,
            bm25_score=bm25_score,
            intent_score=intent_score,
            final_score=combined,
        )
        ranked.append(RankedCandidate(candidate=candidate, scores=scores))
    ranked.sort(key=lambda item: item.scores.final_score, reverse=True)
    _apply_intent_disambiguation(ranked)
    return tuple(ranked[:max_candidates])


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
    if margin < min_margin:
        return None
    return top_candidate


def _is_exact_lexical_match(ranked_candidate: RankedCandidate) -> bool:
    """Return whether a ranked candidate exactly matches query text lexically."""
    scores = ranked_candidate.scores
    return scores.rapidfuzz_score == 1.0 and scores.char_ngram_score == 1.0
