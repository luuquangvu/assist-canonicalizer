"""Lexical candidate ranking."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence, Set
from dataclasses import dataclass, field
from functools import lru_cache, partial
from heapq import nlargest, nsmallest
from math import isclose
from typing import NamedTuple, TypedDict, overload

from homeassistant.util.json import JsonObjectType
from rapidfuzz import fuzz

from .bm25 import BM25Index
from .candidate import Candidate
from .const import (
    BM25_WEIGHT,
    CHAR_NGRAM_WEIGHT,
    CONTEXT_SLOT_MATCH_BOOST,
    CONTEXT_SLOT_MISMATCH_PENALTY,
    DEFAULT_HOTWORD_MIN_CONFIDENCE,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES,
    DEFAULT_SLOT_PREFILTER_RESCUE_CANDIDATES,
    DEFAULT_WILDCARD_PREFILTER_RESCUE_CANDIDATES,
    ENTITY_ONLY_UNCOVERED_QUERY_PENALTY,
    ENTITY_SLOT_NAME_SET,
    HIGH_CONFIDENCE_RELAXED_MIN_MARGIN,
    HIGH_CONFIDENCE_RELAXED_MIN_SCORE,
    INTENT_ACTION_WEIGHT,
    LOCATION_SLOT_NAME_SET,
    NON_ENTITY_PENALTY_BLEND,
    NUMERIC_LITERAL_WITHOUT_QUERY_PENALTY,
    NUMERIC_SLOT_MISMATCH_PENALTY,
    NUMERIC_SLOT_WITHOUT_QUERY_PENALTY,
    POSITIONAL_FUZZY_TOKEN_MAX_LENGTH_RATIO,
    POSITIONAL_SIMILARITY_BASE_THRESHOLD,
    POSITIONAL_SIMILARITY_MEDIUM_THRESHOLD,
    POSITIONAL_SIMILARITY_PARTIAL_CREDIT,
    POSITIONAL_SIMILARITY_SHORT_3_THRESHOLD,
    POSITIONAL_SIMILARITY_VERY_SHORT_THRESHOLD,
    QUERY_NUMBER_WITHOUT_CANDIDATE_PENALTY,
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
    UNANCHORED_ENTITY_SLOT_PENALTY,
    UNANCHORED_SEMANTIC_SLOT_PENALTY,
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
    normalize_text_no_diacritics_from_normalized,
    tokenize_normalized,
)
from .rehydration import (
    WildcardVariantAnalysis,
    get_wildcard_rehydration,
    wildcard_variants_analysis,
)
from .utils import (
    NormalizedIntentContext,
    normalize_hotword_list,
    normalize_intent_context,
    normalized_slot_value_tokens,
    parse_float,
)


class _LiteralVariantAnalysis(NamedTuple):
    """Precomputed positional evidence for one literal-token variant."""

    total_token_count: int
    exact_match_tokens: frozenset[str]
    positional_hits: tuple[frozenset[str], ...]
    positional_query_tokens: frozenset[str]
    positional_match_count: int
    requires_unique_alignment: bool


class WildcardVariantGroup(NamedTuple):
    """Candidates sharing the same wildcard literal-token coverage variants."""

    variants: tuple[WildcardVariantAnalysis, ...]
    literal_tokens: frozenset[str]
    min_required_match_count: int
    candidate_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SlotTokenIndex:
    """Candidate postings for exact and one-edit registry slot tokens.

    The deletion-neighborhood map keeps query-time lookup proportional to the
    query token lengths instead of scanning every entity/location token in a
    large language index. Candidate membership is validated by the bounded
    slot matcher before it can affect ranking.
    """

    candidate_count: int
    candidate_indices_by_token: dict[str, tuple[int, ...]]
    tokens_by_deletion: dict[str, tuple[str, ...]]
    query_match_cache: dict[tuple[frozenset[str], frozenset[str] | None], frozenset[str]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def from_candidate_tokens(
        cls,
        candidate_slot_tokens: Sequence[frozenset[str]],
    ) -> SlotTokenIndex:
        """Build reusable slot-token and deletion-neighborhood postings."""
        candidate_indices_by_token: defaultdict[str, list[int]] = defaultdict(list)
        for index, slot_tokens in enumerate(candidate_slot_tokens):
            for token in slot_tokens:
                candidate_indices_by_token[token].append(index)

        tokens_by_deletion: defaultdict[str, list[str]] = defaultdict(list)
        for token in candidate_indices_by_token:
            for deletion in _single_character_deletions(token):
                tokens_by_deletion[deletion].append(token)

        return cls(
            candidate_count=len(candidate_slot_tokens),
            candidate_indices_by_token={
                token: tuple(indices) for token, indices in candidate_indices_by_token.items()
            },
            tokens_by_deletion={
                deletion: tuple(tokens) for deletion, tokens in tokens_by_deletion.items()
            },
        )


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
_EQUIVALENT_INTENT_GROUPS: Sequence[frozenset[str]] = (
    frozenset(("HassListAddItem", "HassShoppingListAddItem")),
    frozenset(("HassListCompleteItem", "HassShoppingListCompleteItem")),
    frozenset(("HassListRemoveItem", "HassShoppingListCompleteItem")),
)
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


class CandidateEvidence(TypedDict):
    """Serializable evidence for one ranked candidate."""

    intent_name: str
    source: str
    text_sha256: str
    score: float


class ConfidenceGatePayload(TypedDict):
    """Serializable confidence-gate evidence with stable field types."""

    configured_min_confidence: float
    configured_base_margin: float
    top_candidate: CandidateEvidence | None
    meaningful_competitor: CandidateEvidence | None
    observed_margin: float | None
    required_margin: float
    margin_policy: str
    relaxation_used: bool
    opposing_action_competition: bool
    action_incompatible_competition: bool
    target_incompatible_competition: bool
    accepted: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class ConfidenceGateDecision:
    """Complete, serializable evidence for one confidence-gate decision."""

    configured_min_confidence: float
    configured_base_margin: float
    top_candidate: RankedCandidate | None
    meaningful_competitor: RankedCandidate | None
    observed_margin: float | None
    required_margin: float
    margin_policy: str
    relaxation_used: bool
    opposing_action_competition: bool
    action_incompatible_competition: bool
    target_incompatible_competition: bool
    accepted_candidate: RankedCandidate | None
    rejection_reason: FallbackReason | None

    @property
    def accepted(self) -> bool:
        """Return whether the decision accepts a candidate."""
        return self.accepted_candidate is not None

    def __iter__(self) -> Iterator[RankedCandidate | FallbackReason | None]:
        """Yield the historical candidate/reason pair during caller migration."""
        yield self.accepted_candidate
        yield self.rejection_reason

    def as_dict(self) -> ConfidenceGatePayload:
        """Return bounded evidence without delegated or registry text."""

        def candidate_evidence(
            ranked_candidate: RankedCandidate | None,
        ) -> CandidateEvidence | None:
            if ranked_candidate is None:
                return None
            candidate = ranked_candidate.candidate
            return {
                "intent_name": candidate.intent_name,
                "source": candidate.source.value,
                "text_sha256": hashlib.sha256(candidate.normalized_text.encode()).hexdigest(),
                "score": ranked_candidate.scores.final_score,
            }

        return {
            "configured_min_confidence": self.configured_min_confidence,
            "configured_base_margin": self.configured_base_margin,
            "top_candidate": candidate_evidence(self.top_candidate),
            "meaningful_competitor": candidate_evidence(self.meaningful_competitor),
            "observed_margin": self.observed_margin,
            "required_margin": self.required_margin,
            "margin_policy": self.margin_policy,
            "relaxation_used": self.relaxation_used,
            "opposing_action_competition": self.opposing_action_competition,
            "action_incompatible_competition": self.action_incompatible_competition,
            "target_incompatible_competition": self.target_incompatible_competition,
            "accepted": self.accepted,
            "reason": self.rejection_reason.value if self.rejection_reason else None,
        }

    def as_json_dict(self) -> JsonObjectType:
        """Return confidence evidence as Home Assistant's recursive JSON type."""
        d = self.as_dict()
        top = d["top_candidate"]
        comp = d["meaningful_competitor"]
        top_dict: JsonObjectType | None = (
            {
                "intent_name": top["intent_name"],
                "source": top["source"],
                "text_sha256": top["text_sha256"],
                "score": top["score"],
            }
            if top is not None
            else None
        )
        comp_dict: JsonObjectType | None = (
            {
                "intent_name": comp["intent_name"],
                "source": comp["source"],
                "text_sha256": comp["text_sha256"],
                "score": comp["score"],
            }
            if comp is not None
            else None
        )
        return {
            "configured_min_confidence": d["configured_min_confidence"],
            "configured_base_margin": d["configured_base_margin"],
            "top_candidate": top_dict,
            "meaningful_competitor": comp_dict,
            "observed_margin": d["observed_margin"],
            "required_margin": d["required_margin"],
            "margin_policy": d["margin_policy"],
            "relaxation_used": d["relaxation_used"],
            "opposing_action_competition": d["opposing_action_competition"],
            "action_incompatible_competition": d["action_incompatible_competition"],
            "target_incompatible_competition": d["target_incompatible_competition"],
            "accepted": d["accepted"],
            "reason": d["reason"],
        }


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
        intersections = self.intersections(query_grams)
        query_count = len(query_grams)
        scores = [0.0] * len(self.gram_counts)
        for index, intersection_size in enumerate(intersections):
            if intersection_size == 0:
                continue
            scores[index] = _char_ngram_score_from_intersection(
                query_count,
                self.gram_counts[index],
                intersection_size,
            )
        return scores

    def intersections(self, query_grams: frozenset[str]) -> list[int]:
        """Return query/candidate n-gram intersection counts for all candidates."""
        _nothing: tuple[int, ...] = ()
        intersections = [0] * len(self.gram_counts)
        if not query_grams:
            return intersections
        _postings = self.postings
        _postings_get = _postings.get
        for gram in query_grams:
            for index in _postings_get(gram, _nothing):
                intersections[index] += 1
        return intersections

    def intersections_with_touched(self, query_grams: frozenset[str]) -> tuple[list[int], set[int]]:
        """Return intersection counts plus the set of candidates with any hit.

        The touched set lets prefilter-key construction stay proportional to
        posting sizes instead of the full corpus, which matters as indexes
        approach the per-language candidate cap.
        """
        _nothing: tuple[int, ...] = ()
        intersections = [0] * len(self.gram_counts)
        touched: set[int] = set()
        if not query_grams:
            return intersections, touched
        _postings = self.postings
        _postings_get = _postings.get
        for gram in query_grams:
            postings = _postings_get(gram, _nothing)
            touched.update(postings)
            for index in postings:
                intersections[index] += 1
        return intersections, touched


@lru_cache(maxsize=32768)
def _raw_cached_fuzz_ratio(s1: str, s2: str) -> float:
    """Return RapidFuzz's ratio with LRU caching.

    The cache is sized to hold one query's full token cross-product so later
    candidates in the same ranking pass reuse earlier pair computations.
    """
    return fuzz.ratio(s1, s2)


def _cached_fuzz_ratio(s1: str, s2: str) -> float:
    """Return RapidFuzz's ratio with LRU caching.

    The argument order is normalized to maximize cache hit rate.
    """
    return _raw_cached_fuzz_ratio(s1, s2) if s1 <= s2 else _raw_cached_fuzz_ratio(s2, s1)


def _slot_tokens_fuzzy_match(token_a: str, token_b: str) -> bool:
    """Return whether two tokens reach the slot fuzzy-match ratio threshold.

    ``fuzz.ratio`` is bounded by ``200 * min_len / (len_a + len_b)``, so pairs
    whose lengths make the threshold unreachable skip the ratio computation
    entirely.
    """
    len_a = len(token_a)
    len_b = len(token_b)
    if 200 * min(len_a, len_b) < SLOT_TOKEN_MATCH_THRESHOLD * (len_a + len_b):
        return False
    return _cached_fuzz_ratio(token_a, token_b) >= SLOT_TOKEN_MATCH_THRESHOLD


def _registry_slot_token_fuzzy_match(
    slot_token: str,
    query_token: str,
    short_fuzzy_query_tokens: frozenset[str],
) -> bool:
    """Fuzzy-match a registry slot value token against a query token.

    Stopword-length pairs (``la``/``loa``) clear the ratio threshold while
    carrying no real evidence, which lets registry aliases from one language
    claim unrelated short words from another. The slot side therefore always
    requires three characters. A short query token keeps its fuzzy anchor
    only when it is listed in ``short_fuzzy_query_tokens`` (it matches no
    corpus literal, so it cannot be a grammar word such as an article and is
    far more likely a truncated target like ``fn`` for ``fan``). Exact
    equality is handled by the callers before this check.
    """
    if len(slot_token) < 3:
        return False
    if len(query_token) < 3 and query_token not in short_fuzzy_query_tokens:
        return False
    return _slot_tokens_fuzzy_match(slot_token, query_token)


@lru_cache(maxsize=256)
def _short_nonliteral_query_tokens(
    query_tokens: frozenset[str],
    positional_literal_tokens: frozenset[str] | None,
) -> frozenset[str]:
    """Return sub-three-character query tokens with no corpus-literal reading.

    Grammar words (articles, prepositions) always appear as template literal
    tokens, so a short query token absent from every literal cannot be
    explained by the grammar and may safely keep bounded fuzzy anchoring
    against registry slot values. When no literal inventory is available the
    exemption stays empty, preserving the strict gate.
    """
    if positional_literal_tokens is None:
        return frozenset()
    return frozenset(
        token for token in query_tokens if len(token) < 3 and token not in positional_literal_tokens
    )


def _char_ngram_score_from_intersection(
    query_count: int,
    candidate_count: int,
    intersection_size: int,
) -> float:
    """Return a Jaccard score from precomputed set cardinalities."""
    union_size = query_count + candidate_count - intersection_size
    return intersection_size / union_size if union_size else 0.0


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


@lru_cache(maxsize=256)
def _build_positional_lookup(
    literal_tokens_set: frozenset[str],
    query_tokens: frozenset[str],
) -> dict[str, frozenset[str]]:
    """Precompute which query tokens each literal token positionally matches.

    Uses first-character bucketing to prune the O(|literal|x|query|) inner
    loop. Beyond pruning, the shared first character is a deliberate accuracy
    guard: verb compounds that differ only in a short prefix while sharing a
    long stem (``aanschakelen``/``uitschakelen``) reach positional thresholds
    despite naming opposite actions, so leading mismatches must never match.

    Cached because the static and dynamic rank passes of one request call
    this with identical arguments; the returned dict is shared and must be
    treated as read-only by callers.
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
            if (
                sim >= _per_pair_positional_threshold(literal_token, qtok)
                or _is_fuzzy_literal_token_match(literal_token, qtok)
            ) and not _has_conflicting_mark_mismatch(literal_token, qtok):
                # No early break: query_tokens iterates a frozenset, so a
                # short-circuit would make the shared cached lookup depend on
                # per-process hash order.
                matched.append(qtok)
        if matched:
            lookup[literal_token] = frozenset(matched)
    return lookup


@lru_cache(maxsize=4096)
def _chars_conflict_in_marks(char_a: str, char_b: str) -> bool:
    """Return whether two marked characters signal deliberately distinct words.

    When both characters carry explicit combining marks over the same base and
    the mark sets share nothing (e.g. Vietnamese ``ó`` vs ``ộ``), the writer
    chose each mark deliberately, so the difference distinguishes words rather
    than indicating a typo. Pairs that share a mark (``á`` vs ``ắ`` both carry
    the acute tone) stay tolerated as plausible input slips, as does a marked
    character against its bare base (``é`` vs ``e``).
    """
    decomposed_a = unicodedata.normalize("NFD", char_a)
    decomposed_b = unicodedata.normalize("NFD", char_b)
    if len(decomposed_a) < 2 or len(decomposed_b) < 2:
        return False
    return decomposed_a[0] == decomposed_b[0] and frozenset(decomposed_a[1:]).isdisjoint(
        decomposed_b[1:]
    )


def _has_conflicting_mark_mismatch(token_a: str, token_b: str) -> bool:
    """Return whether aligned characters differ only by deliberate marks."""
    if token_a.isascii() or token_b.isascii():
        return False
    return any(
        (
            char_a != char_b
            and not char_a.isascii()
            and not char_b.isascii()
            and _chars_conflict_in_marks(char_a, char_b)
        )
        for char_a, char_b in zip(token_a, token_b, strict=False)
    )


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
    """Return whether a query token is a bounded slot-token variation.

    In addition to one-character edits, a canonical slot token accepts a
    four-or-more-character query prefix (a common truncated-name form) and one
    adjacent transposition. The direction of the prefix check is deliberate:
    user input may abbreviate a known registry value, while a short registry
    value must not claim an unrelated longer query token.
    """
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
    if query_len < slot_len and slot_token.startswith(query_token):
        return True
    if abs(slot_len - query_len) > 1:
        return False
    if slot_len == query_len:
        differences = [
            index
            for index, (slot_char, query_char) in enumerate(
                zip(slot_token, query_token, strict=True)
            )
            if slot_char != query_char
        ]
        if len(differences) == 1:
            index = differences[0]
            return not _chars_conflict_in_marks(slot_token[index], query_token[index])
        return (
            len(differences) == 2
            and differences[1] == differences[0] + 1
            and slot_token[differences[0]] == query_token[differences[1]]
            and slot_token[differences[1]] == query_token[differences[0]]
        )
    return _is_fuzzy_literal_token_match(slot_token, query_token)


def _single_character_deletions(token: str) -> frozenset[str]:
    """Return unique strings formed by deleting one character from *token*."""
    return frozenset(token[:index] + token[index + 1 :] for index in range(len(token)))


def _precompute_literal_analysis(
    literal_text_or_variants: str | tuple[frozenset[str], ...],
    query_tokens: frozenset[str],
    positional_lookup: dict[str, frozenset[str]],
) -> list[_LiteralVariantAnalysis]:
    """Precompute positional match data for *literal_text* against *query_tokens*.

    Returns one :class:`_LiteralVariantAnalysis` per variant. ``exact_match_tokens``
    contains literal tokens that appear verbatim in *query_tokens*, while
    ``positional_hits`` contains the query-token sets that positionally match
    each remaining literal token.

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
    result: list[_LiteralVariantAnalysis] = []
    for literal_tokens in variants:
        exact_tokens: set[str] = set()
        positional_hits: list[frozenset[str]] = []
        for token in literal_tokens:
            if token in query_tokens:
                exact_tokens.add(token)
            else:
                matching = positional_lookup.get(token)
                if matching is not None:
                    positional_hits.append(matching)
        seen_query_tokens = set(exact_tokens)
        requires_unique_alignment = False
        for matching in positional_hits:
            if not seen_query_tokens.isdisjoint(matching):
                requires_unique_alignment = True
            seen_query_tokens.update(matching)
        positional_query_tokens = frozenset(
            query_token for matching in positional_hits for query_token in matching
        )
        exact_tokens_frozen = frozenset(exact_tokens)
        positional_match_count = (
            _maximum_unique_positional_matches(positional_hits, exact_tokens_frozen)
            if requires_unique_alignment
            else len(positional_hits)
        )
        result.append(
            _LiteralVariantAnalysis(
                total_token_count=len(literal_tokens),
                exact_match_tokens=exact_tokens_frozen,
                positional_hits=tuple(positional_hits),
                positional_query_tokens=positional_query_tokens,
                positional_match_count=positional_match_count,
                requires_unique_alignment=requires_unique_alignment,
            )
        )
    return result


def _maximum_unique_positional_matches(
    positional_hits: Sequence[frozenset[str]],
    unavailable_query_tokens: frozenset[str],
) -> int:
    """Return the maximum one-to-one alignment count for positional hits.

    A query token is evidence for at most one literal token.  Maximum bipartite
    matching avoids both duplicate credit and greedy-order artifacts when one
    literal has several possible query-token matches.
    """
    available_hits: list[frozenset[str]] = []
    seen_query_tokens: set[str] = set()
    has_overlap = False
    for hits in positional_hits:
        available = (
            hits if hits.isdisjoint(unavailable_query_tokens) else hits - unavailable_query_tokens
        )
        if not available:
            continue
        if not seen_query_tokens.isdisjoint(available):
            has_overlap = True
        seen_query_tokens.update(available)
        available_hits.append(available)

    if not has_overlap:
        return len(available_hits)

    query_token_owner: dict[str, int] = {}

    def assign(literal_index: int, visited: set[str]) -> bool:
        for query_token in available_hits[literal_index]:
            if query_token in visited:
                continue
            visited.add(query_token)
            previous_owner = query_token_owner.get(query_token)
            if previous_owner is None or assign(previous_owner, visited):
                query_token_owner[query_token] = literal_index
                return True
        return False

    matched = 0
    for literal_index in sorted(
        range(len(available_hits)), key=lambda index: len(available_hits[index])
    ):
        if available_hits[literal_index] and assign(literal_index, set()):
            matched += 1
    return matched


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
        exact_tokens: set[str] = set()
        positional_hits: list[frozenset[str]] = []
        for token in literal_tokens:
            if token in query_tokens:
                exact_tokens.add(token)
                continue
            matching = positional_lookup.get(token)
            if matching is not None:
                positional_hits.append(matching)
        unavailable = frozenset(exact_tokens)
        if candidate_entity is not None:
            unavailable |= candidate_entity
        positional_count = _maximum_unique_positional_matches(
            positional_hits,
            unavailable,
        )
        matched_weight = len(exact_tokens) + (
            positional_count * POSITIONAL_SIMILARITY_PARTIAL_CREDIT
        )
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
    variants: tuple[WildcardVariantAnalysis, ...],
    query_tokens: frozenset[str],
) -> bool:
    """Return True if any wildcard variant has sufficient token overlap with the query."""
    for variant in variants:
        literal_token_count = len(variant.literal_tokens)
        if literal_token_count == 0:
            return True
        if variant.required_match_count == literal_token_count:
            if variant.literal_tokens.issubset(query_tokens):
                return True
            continue
        if variant.required_match_count == 1:
            if not variant.literal_tokens.isdisjoint(query_tokens):
                return True
            continue
        matched = 0
        for token in variant.literal_tokens:
            if token in query_tokens:
                matched += 1
                if matched >= variant.required_match_count:
                    return True
    return False


def _check_precomputed_wildcard(
    i: int,
    query_tokens: frozenset[str],
    wildcard_always_passes: frozenset[int],
    wildcard_variant_analyses: dict[int, tuple[WildcardVariantAnalysis, ...]] | None,
    wildcard_literal_tokens_by_index: dict[int, frozenset[str]] | None,
    wildcard_min_required_by_index: dict[int, int] | None,
) -> bool:
    """Return True if precomputed wildcard candidate i passes the token filter."""
    if i in wildcard_always_passes:
        return True
    variants = wildcard_variant_analyses.get(i) if wildcard_variant_analyses else None
    if not variants:
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
    return _wildcard_variants_match(variants, query_tokens)


def _check_onthefly_wildcard(cand: Candidate, query_tokens: frozenset[str]) -> bool:
    """Return True if an on-the-fly analyzed wildcard candidate passes the token filter."""
    variants, all_literal = wildcard_variants_analysis(cand)
    always_passes = not cand.literal_variants or any(
        not variant.literal_tokens for variant in variants
    )
    if always_passes:
        return True
    if all_literal.isdisjoint(query_tokens):
        return False
    return _wildcard_variants_match(variants, query_tokens)


def _prefilter_wildcard_candidates(
    candidates: Sequence[Candidate],
    query_tokens: frozenset[str],
    wildcard_always_passes: frozenset[int] | None,
    wildcard_variant_analyses: dict[int, tuple[WildcardVariantAnalysis, ...]] | None = None,
    wildcard_token_to_indices: dict[str, tuple[int, ...]] | None = None,
    wildcard_literal_tokens_by_index: dict[int, frozenset[str]] | None = None,
    wildcard_min_required_by_index: dict[int, int] | None = None,
    wildcard_variant_groups: tuple[WildcardVariantGroup, ...] | None = None,
) -> set[int]:
    """Prefilter wildcard candidates using precomputed structures or on-the-fly coverage."""
    if wildcard_always_passes is not None and wildcard_variant_groups is not None:
        passed = set(wildcard_always_passes)
        for group in wildcard_variant_groups:
            if group.min_required_match_count > 0 and (
                len(group.literal_tokens & query_tokens) < group.min_required_match_count
            ):
                continue
            if _wildcard_variants_match(group.variants, query_tokens):
                passed.update(group.candidate_indices)
        return passed
    if (
        wildcard_always_passes is not None
        and wildcard_variant_analyses is not None
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
                wildcard_variant_analyses,
                wildcard_literal_tokens_by_index,
                wildcard_min_required_by_index,
            )
        }
    return {
        i
        for i, cand in enumerate(candidates)
        if cand.has_wildcard and _check_onthefly_wildcard(cand, query_tokens)
    }


def _top_additional_wildcard_indices(
    wildcard_indices: set[int],
    query_tokens: frozenset[str],
    wildcard_variant_analyses: dict[int, tuple[WildcardVariantAnalysis, ...]] | None,
    prefilter_keys: Sequence[float],
    limit: int,
) -> list[int]:
    """Return the strongest wildcard rescue candidates within a bounded budget.

    Wildcards need a separate rescue path because their placeholder text can
    score poorly before rehydration. Literal-token coverage provides the
    placeholder-independent retrieval signal; the normal BM25/character score
    breaks ties. Bounding only the additional rescue candidates keeps the
    general lexical prefilter intact while preventing unbounded rehydration.
    """
    if len(wildcard_indices) <= limit:
        return sorted(wildcard_indices)

    variant_relevance: dict[tuple[WildcardVariantAnalysis, ...], tuple[float, int]] = {}

    def relevance(index: int) -> tuple[float, int, float, int]:
        variants = (
            wildcard_variant_analyses.get(index, ())
            if wildcard_variant_analyses is not None
            else ()
        )
        cached_relevance = variant_relevance.get(variants)
        if cached_relevance is None:
            best_coverage = 0.0
            best_matched = 0
            for variant in variants:
                literal_token_count = len(variant.literal_tokens)
                if literal_token_count == 0:
                    continue
                matched = len(variant.literal_tokens & query_tokens)
                coverage = matched / literal_token_count
                if (coverage, matched) > (best_coverage, best_matched):
                    best_coverage = coverage
                    best_matched = matched
            cached_relevance = (best_coverage, best_matched)
            variant_relevance[variants] = cached_relevance
        best_coverage, best_matched = cached_relevance
        return best_coverage, best_matched, -prefilter_keys[index], -index

    return nlargest(limit, wildcard_indices, key=relevance)


def _exact_lookup_ranked(
    query: str,
    query_normalized: str,
    max_candidates: int,
    exact_normalized_lookup: Mapping[str, Sequence[Candidate]] | None,
    exact_no_diacritics_lookup: Mapping[str, Sequence[Candidate]] | None,
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
            unique_intents = {candidate.intent_name for candidate in no_diac_matches}
            same_slots = all(
                candidate.parsed_slots == no_diac_matches[0].parsed_slots
                for candidate in no_diac_matches[1:]
            )
            if len(unique_intents) == 1 and same_slots:
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


def _rank_prefilter_keys_with_sparse_bm25(
    char_scores: Sequence[float],
    raw_bm25_scores: Mapping[int, float],
    bm25_inv_max: float,
) -> list[float]:
    """Return dense-equivalent keys without allocating a dense BM25 score array."""
    keys = [-CHAR_NGRAM_WEIGHT * char_score for char_score in char_scores]
    for index, raw_score in raw_bm25_scores.items():
        normalized_score = raw_score * bm25_inv_max
        keys[index] -= BM25_WEIGHT * normalized_score
    return keys


class _SparsePrefilterKeys(Sequence[float]):
    """Read-only sequence view over sparse prefilter keys.

    Untouched candidates hold the implicit worst key of exactly ``0.0``;
    only candidates with character-gram or BM25 evidence are materialized.
    """

    __slots__ = ("keys", "length")

    def __init__(self, keys: dict[int, float], length: int) -> None:
        """Store the sparse key mapping and the dense logical length."""
        self.keys = keys
        self.length = length

    @overload
    def __getitem__(self, index: int) -> float: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[float]: ...

    def __getitem__(self, index: int | slice) -> float | Sequence[float]:
        """Return the prefilter key for one candidate index or slice."""
        if isinstance(index, slice):
            return [self.keys.get(i, 0.0) for i in range(*index.indices(self.length))]
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        return self.keys.get(index, 0.0)

    def __len__(self) -> int:
        """Return the dense logical length."""
        return self.length


def _sparse_prefilter_keys_from_intersections(
    intersections: Sequence[int],
    touched: set[int],
    query_gram_count: int,
    candidate_gram_counts: Sequence[int],
    raw_bm25_scores: Mapping[int, float],
    bm25_inv_max: float,
) -> dict[int, float]:
    """Build static prefilter keys only for candidates with any evidence.

    Every produced key is strictly negative (weights and scores are
    positive), so absent entries are exactly the dense path's ``0.0`` keys.
    """
    keys: dict[int, float] = {}
    for index in touched:
        intersection_size = intersections[index]
        keys[index] = -CHAR_NGRAM_WEIGHT * (
            intersection_size
            / (query_gram_count + candidate_gram_counts[index] - intersection_size)
        )
    for index, raw_score in raw_bm25_scores.items():
        normalized_bm25 = raw_score * bm25_inv_max
        keys[index] = keys.get(index, 0.0) - BM25_WEIGHT * normalized_bm25
    return keys


def _top_sparse_prefilter_indices(
    sparse_keys: dict[int, float],
    candidate_count: int,
    prefilter_limit: int,
) -> list[int]:
    """Return prefilter-heap indices identical to the dense selection.

    Touched candidates all outrank untouched ones (negative versus ``0.0``
    keys), and the explicit ``(key, index)`` tuple reproduces the dense
    heap's stable ascending-index tie-break. When evidence covers fewer
    candidates than the limit, the dense path would fill remaining slots
    with the lowest untouched indices, replicated here without an O(corpus)
    scan.
    """
    limit = min(prefilter_limit, candidate_count)
    selected = nsmallest(limit, sparse_keys, key=lambda index: (sparse_keys[index], index))
    fill_needed = limit - len(selected)
    if fill_needed > 0:
        index = 0
        while fill_needed > 0:
            if index not in sparse_keys:
                selected.append(index)
                fill_needed -= 1
            index += 1
    return selected


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


@dataclass(frozen=True, slots=True)
class _Bm25Scoring:
    """BM25 values and accessors for one ranking query.

    Static indexes retain only positive raw scores, while dynamic candidates are
    scored against a reference index and require a dense score sequence. The
    accessors keep those representations out of the main ranking flow.
    """

    index: BM25Index
    max_raw_score: float
    dense_scores: Sequence[float] | None = None
    sparse_raw_scores: Mapping[int, float] | None = None
    sparse_inv_max: float = 0.0

    def prefilter_keys(self, char_scores: Sequence[float]) -> list[float]:
        """Return prefilter keys using this query's BM25 representation."""
        if self.sparse_raw_scores is not None:
            return _rank_prefilter_keys_with_sparse_bm25(
                char_scores,
                self.sparse_raw_scores,
                self.sparse_inv_max,
            )
        return _rank_prefilter_keys(char_scores, self.dense_scores or ())

    def score_at(self, index: int) -> float:
        """Return the normalized BM25 score for one candidate."""
        if self.sparse_raw_scores is not None:
            return self.sparse_raw_scores.get(index, 0.0) * self.sparse_inv_max
        return 0.0 if self.dense_scores is None else self.dense_scores[index]


def _prepare_bm25_scoring(
    candidates: Sequence[Candidate],
    query_tokens: tuple[str, ...],
    bm25_index: BM25Index | None,
    reference_bm25_index: BM25Index | None,
) -> _Bm25Scoring:
    """Build query BM25 state for static or reference-index candidate ranking."""
    if reference_bm25_index is not None:
        dense_scores = reference_bm25_index.score_custom_documents_tokens(query_tokens, candidates)
        max_index = max(range(len(dense_scores)), key=dense_scores.__getitem__, default=0)
        max_raw_score = 0.0
        if dense_scores and dense_scores[max_index] > 0.0:
            max_raw_score = reference_bm25_index.raw_score_tokens(
                candidates[max_index].normalized_tokens,
                query_tokens,
            )
            reference_max_raw = max(
                reference_bm25_index.raw_scores_sparse(query_tokens).values(),
                default=0.0,
            )
            if reference_max_raw > max_raw_score > 0.0:
                # Normalize against the stronger reference-corpus match so the
                # best of a weak dynamic set cannot carry an unearned 1.0 into
                # the static/dynamic merge.
                scale = max_raw_score / reference_max_raw
                dense_scores = tuple(score * scale for score in dense_scores)
                max_raw_score = reference_max_raw
        return _Bm25Scoring(
            index=reference_bm25_index,
            max_raw_score=max_raw_score,
            dense_scores=dense_scores,
        )

    index = bm25_index or BM25Index.from_normalized_texts(
        tuple(candidate.normalized_text for candidate in candidates)
    )
    sparse_raw_scores = index.raw_scores_sparse(query_tokens) if query_tokens else {}
    max_raw_score = max(sparse_raw_scores.values(), default=0.0)
    sparse_inv_max = 1.0 / max_raw_score if max_raw_score > 0.0 else 0.0
    return _Bm25Scoring(
        index=index,
        max_raw_score=max_raw_score,
        sparse_raw_scores=sparse_raw_scores,
        sparse_inv_max=sparse_inv_max,
    )


def _query_slot_tokens_from_candidates(
    query_tokens: frozenset[str],
    top_indices: Sequence[int],
    candidate_slot_tokens: tuple[frozenset[str], ...],
    fuzzy_excluded_tokens: frozenset[str] | None = None,
) -> frozenset[str]:
    """Return query slot tokens using only the prefiltered candidate records."""
    active_slot_tokens = {token for index in top_indices for token in candidate_slot_tokens[index]}
    return _query_slot_tokens_from_active(
        query_tokens,
        active_slot_tokens,
        fuzzy_excluded_tokens=fuzzy_excluded_tokens,
    )


def _query_slot_tokens_from_active(
    query_tokens: frozenset[str],
    active_slot_tokens: Sequence[str] | Set[str],
    *,
    fuzzy_excluded_tokens: frozenset[str] | None = None,
) -> frozenset[str]:
    """Return exact and bounded fuzzy matches within active slot tokens."""
    matched_tokens: set[str] = {token for token in active_slot_tokens if token in query_tokens}
    for query_token in query_tokens:
        if query_token in matched_tokens or (
            fuzzy_excluded_tokens is not None and query_token in fuzzy_excluded_tokens
        ):
            continue
        matched_tokens.update(
            slot_token
            for slot_token in active_slot_tokens
            if slot_token not in matched_tokens
            and _is_fuzzy_slot_token_match(slot_token, query_token)
        )
    return frozenset(matched_tokens)


def _slot_index_query_matches(
    query_tokens: frozenset[str],
    slot_token_index: SlotTokenIndex,
    fuzzy_excluded_tokens: frozenset[str] | None = None,
) -> frozenset[str]:
    """Return indexed slot tokens that exactly or fuzzily match the query.

    Exact postings and deletion neighborhoods cover one insertion, deletion,
    substitution, or adjacent transposition. The final bounded matcher rejects
    signature collisions and preserves the slot-token length safeguards.

    Results are memoized on the index instance: the static and dynamic rank
    passes of one request probe the same index with identical query tokens,
    and the deletion-neighborhood expansion is the expensive step.
    """
    cache = slot_token_index.query_match_cache
    cache_key = (query_tokens, fuzzy_excluded_tokens)
    if (cached := cache.get(cache_key)) is not None:
        return cached
    token_postings = slot_token_index.candidate_indices_by_token
    deletion_postings = slot_token_index.tokens_by_deletion
    matched_tokens = {query_token for query_token in query_tokens if query_token in token_postings}

    for query_token in query_tokens:
        if query_token in matched_tokens or (
            fuzzy_excluded_tokens is not None and query_token in fuzzy_excluded_tokens
        ):
            continue
        possible_tokens = set(deletion_postings.get(query_token, ()))
        for deletion in _single_character_deletions(query_token):
            if deletion in token_postings:
                possible_tokens.add(deletion)
            possible_tokens.update(deletion_postings.get(deletion, ()))
        matched_tokens.update(
            slot_token
            for slot_token in possible_tokens
            if _is_fuzzy_slot_token_match(slot_token, query_token)
        )
    result = frozenset(matched_tokens)
    if len(cache) >= 8:
        oldest_key = next(iter(cache), None)
        if oldest_key is not None:
            cache.pop(oldest_key, None)
    cache[cache_key] = result
    return result


def _top_additional_slot_indices(
    query_tokens: frozenset[str],
    candidate_slot_tokens: tuple[frozenset[str], ...],
    slot_token_index: SlotTokenIndex,
    top_indices: Sequence[int],
    prefilter_keys: Sequence[float],
    limit: int = DEFAULT_SLOT_PREFILTER_RESCUE_CANDIDATES,
    fuzzy_excluded_tokens: frozenset[str] | None = None,
) -> list[int]:
    """Return a bounded set of fully query-anchored slot candidates.

    A candidate is eligible only when every one of its non-static slot tokens
    is present in the query within the bounded one-edit matcher. This prevents a
    shared area token from rescuing a conflicting entity while restoring target
    families that the general lexical prefilter omitted.
    """
    if limit <= 0:
        return []
    matched_slot_tokens = _slot_index_query_matches(
        query_tokens,
        slot_token_index,
        fuzzy_excluded_tokens,
    )
    if not matched_slot_tokens:
        return []

    possible_indices = {
        index
        for token in matched_slot_tokens
        for index in slot_token_index.candidate_indices_by_token[token]
    }
    selected_indices = set(top_indices)
    eligible_indices = (
        index
        for index in possible_indices
        if index not in selected_indices
        and candidate_slot_tokens[index]
        and candidate_slot_tokens[index].issubset(matched_slot_tokens)
    )
    return nsmallest(
        limit,
        eligible_indices,
        key=lambda index: (-len(candidate_slot_tokens[index]), prefilter_keys[index], index),
    )


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


@lru_cache(maxsize=8192)
def _slot_names_from_csv(slot_names_text: str) -> frozenset[str]:
    """Parse a comma-separated slot-name metadata value, cached by text.

    Candidates live for the process lifetime while these CSV fields hold a
    small vocabulary, so caching by value removes a per-candidate parse from
    every scoring pass.
    """
    return frozenset(slot for slot in slot_names_text.split(",") if slot)


def _static_slot_names(candidate: Candidate) -> frozenset[str]:
    """Return slot names declared static in candidate metadata."""
    static_slots_text = candidate.metadata.get("static_slots", "")
    if not isinstance(static_slots_text, str) or not static_slots_text:
        return frozenset()
    return _slot_names_from_csv(static_slots_text)


def _context_slot_names(candidate: Candidate) -> frozenset[str]:
    """Return slot names supplied by HassIL intent context."""
    context_slots_text = candidate.metadata.get("context_slots", "")
    if not isinstance(context_slots_text, str) or not context_slots_text:
        return frozenset()
    return _slot_names_from_csv(context_slots_text)


def _wildcard_slot_names(candidate: Candidate) -> frozenset[str]:
    """Return slot names represented by free-text wildcard placeholders."""
    wildcard_slots_text = candidate.metadata.get("wildcard_slots", "")
    if not isinstance(wildcard_slots_text, str) or not wildcard_slots_text:
        return frozenset()
    return _slot_names_from_csv(wildcard_slots_text)


def _context_key_aliases(slot_name: str) -> frozenset[str]:
    """Return context keys equivalent to a candidate slot name."""
    if slot_name in LOCATION_SLOT_NAME_SET:
        return LOCATION_SLOT_NAME_SET
    return frozenset({slot_name})


def _slot_value_has_query_anchor(
    slot_value: object,
    query_tokens: tuple[str, ...],
    short_fuzzy_query_tokens: frozenset[str] = frozenset(),
    anchor_memo: dict[str, bool] | None = None,
) -> bool:
    """Return whether every token in a slot value is present in the query.

    Bounded fuzzy matching preserves explicit-location precedence for normal
    spelling mistakes without treating a shared token such as ``room`` as
    sufficient evidence for a different multi-token location.

    ``anchor_memo`` caches verdicts per ranking pass keyed by the slot value
    text: the query side is fixed for a pass and generated candidates
    massively share slot values.
    """
    memo_key = str(slot_value) if anchor_memo is not None else ""
    if anchor_memo is not None:
        cached = anchor_memo.get(memo_key)
        if cached is not None:
            return cached
    slot_tokens = normalized_slot_value_tokens(slot_value)
    anchored = bool(slot_tokens) and all(
        any(
            slot_token == query_token
            or _is_fuzzy_slot_token_match(slot_token, query_token)
            or _registry_slot_token_fuzzy_match(slot_token, query_token, short_fuzzy_query_tokens)
            for query_token in query_tokens
        )
        for slot_token in slot_tokens
    )
    if anchor_memo is not None:
        anchor_memo[memo_key] = anchored
    return anchored


def _explicit_context_keys_from_query(
    candidates: Sequence[Candidate],
    candidate_indices: Sequence[int],
    normalized_context: NormalizedIntentContext,
    query_tokens: tuple[str, ...],
    short_fuzzy_query_tokens: frozenset[str] = frozenset(),
    anchor_memo: dict[str, bool] | None = None,
) -> frozenset[str]:
    """Return context keys for location values explicitly stated by the user."""
    if not normalized_context:
        return frozenset()
    explicit: set[str] = set()
    for candidate_index in candidate_indices:
        for slot_name, slot_value in candidates[candidate_index].parsed_slots.items():
            if not _slot_value_has_query_anchor(
                slot_value,
                query_tokens,
                short_fuzzy_query_tokens,
                anchor_memo,
            ):
                continue
            if slot_name in ENTITY_SLOT_NAME_SET:
                # An explicit entity is also stronger target evidence than a
                # device/satellite location. The delegated text retains live
                # context, so genuinely generic entities still resolve there.
                explicit.update(
                    context_name
                    for context_name in normalized_context
                    if context_name in LOCATION_SLOT_NAME_SET
                )
            explicit.update(
                context_name
                for context_name in normalized_context
                if context_name in _context_key_aliases(slot_name)
            )
        if len(explicit) == len(normalized_context):
            break
    return frozenset(explicit)


def _context_slot_adjustment(
    slots: Mapping[str, object],
    context_slot_names: frozenset[str],
    normalized_context: NormalizedIntentContext,
    explicit_context_keys: frozenset[str],
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
        if context_name in explicit_context_keys:
            # An explicit user location takes precedence over satellite/device
            # context. Context is only a disambiguator when the query omits it.
            continue
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


def _query_unknown_content_tokens(
    query_tokens: frozenset[str],
    positional_literal_tokens: frozenset[str],
    positional_lookup: Mapping[str, frozenset[str]],
    query_slot_tokens: frozenset[str],
) -> frozenset[str]:
    """Return query tokens with no corpus-literal or slot-value evidence.

    A token that matches no template literal (exactly or positionally) and no
    known slot value usually names a target outside the known vocabulary
    (slang, an unknown device word). Numeric tokens are excluded because
    range slots legitimately accept arbitrary numbers.

    ``query_slot_tokens`` holds the slot-side spelling of fuzzy matches, so
    a query-side typo of a known slot value ("lightt" evidencing "light")
    must be recognized by the reverse fuzzy check; otherwise, a one-edit
    typo in a registry-only target word would count as unknown vocabulary
    and needlessly suppress context disambiguation for the whole query.
    """
    known_tokens = set(query_slot_tokens)
    for matched_query_tokens in positional_lookup.values():
        known_tokens.update(matched_query_tokens)
    return frozenset(
        token
        for token in query_tokens
        if token not in known_tokens
        and token not in positional_literal_tokens
        and not any(char.isdigit() for char in token)
        and not any(
            _is_fuzzy_slot_token_match(slot_token, token) for slot_token in query_slot_tokens
        )
    )


def _has_numeric_query_token(query_tokens: frozenset[str]) -> bool:
    """Return whether the query contains any numeric token."""
    return any(any(char.isdigit() for char in token) for token in query_tokens)


def _is_numeric_slot_value(value: object) -> bool:
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
    slots: Mapping[str, object],
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


def _has_unanchored_numeric_literal(candidate: Candidate, *, query_has_number: bool) -> bool:
    """Return whether generated candidate text adds a number absent from the query."""
    return not query_has_number and any(
        parse_float(token) is not None for token in candidate.normalized_tokens
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
    slots_dict: Mapping[str, object],
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

    raw_slots_dict = candidate.parsed_raw_slots
    dynamic_numeric_values = []
    for name, value in slots_dict.items():
        if name in static_slots:
            continue
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


def _has_unconsumed_query_number(
    candidate_tokens: frozenset[str],
    query_numbers: set[float],
) -> bool:
    """Return whether a numeric query is paired with a candidate containing no number."""
    return bool(query_numbers) and all(parse_float(token) is None for token in candidate_tokens)


def _has_unanchored_semantic_slot(
    candidate: Candidate,
    slots: Mapping[str, object],
    static_slots: frozenset[str],
    query_tokens: tuple[str, ...],
    short_fuzzy_query_tokens: frozenset[str] = frozenset(),
    anchor_memo: dict[str, bool] | None = None,
) -> bool:
    """Return whether a spoken non-target parameter is absent from the query.

    ``slots_raw`` contains the surface value before HassIL output mappings, so
    mapped multilingual synonyms remain anchored by what the user actually
    said. Candidates without raw evidence are left unchanged rather than
    guessing whether an output value was implicit or transformed.
    """
    raw_slots = candidate.parsed_raw_slots
    if not raw_slots:
        return False
    excluded_slots = (
        static_slots
        | _context_slot_names(candidate)
        | _wildcard_slot_names(candidate)
        | ENTITY_SLOT_NAME_SET
        | LOCATION_SLOT_NAME_SET
    )
    for slot_name, raw_value in raw_slots.items():
        if slot_name in excluded_slots or slot_name not in slots:
            continue
        output_value = slots[slot_name]
        if _is_numeric_slot_value(raw_value) or _is_numeric_slot_value(output_value):
            continue
        if not normalized_slot_value_tokens(raw_value):
            continue
        if not _slot_value_has_query_anchor(
            raw_value,
            query_tokens,
            short_fuzzy_query_tokens,
            anchor_memo,
        ):
            return True
    return False


def _has_unanchored_entity_slot(
    slots: Mapping[str, object],
    static_slots: frozenset[str],
    query_tokens_tuple: tuple[str, ...],
    short_fuzzy_query_tokens: frozenset[str] = frozenset(),
    entity_anchor_memo: dict[str, bool] | None = None,
) -> bool:
    """Return whether a non-static entity slot lacks lexical query evidence.

    Static domain/name slots are broad HassIL expansions and are handled by
    the static-slot conflict checks below. ``entity_anchor_memo`` caches
    per-value verdicts for one ranking pass (fixed query side).
    """
    query_tokens = frozenset(query_tokens_tuple)
    for slot_name, slot_value in slots.items():
        if slot_name in static_slots or slot_name not in ENTITY_SLOT_NAME_SET:
            continue
        slot_tokens = normalized_slot_value_tokens(slot_value)
        if not slot_tokens:
            continue
        if entity_anchor_memo is not None:
            memo_key = str(slot_value)
            anchored = entity_anchor_memo.get(memo_key)
            if anchored is None:
                anchored = _entity_slot_tokens_anchored(
                    slot_tokens,
                    query_tokens,
                    query_tokens_tuple,
                    short_fuzzy_query_tokens,
                )
                entity_anchor_memo[memo_key] = anchored
        else:
            anchored = _entity_slot_tokens_anchored(
                slot_tokens,
                query_tokens,
                query_tokens_tuple,
                short_fuzzy_query_tokens,
            )
        if anchored:
            continue
        return True
    return False


def _entity_slot_tokens_anchored(
    slot_tokens: frozenset[str],
    query_tokens: frozenset[str],
    query_tokens_tuple: tuple[str, ...],
    short_fuzzy_query_tokens: frozenset[str],
) -> bool:
    """Return whether any entity slot token has exact or registry-fuzzy query evidence."""
    return any(
        slot_token in query_tokens
        or any(
            _registry_slot_token_fuzzy_match(slot_token, query_token, short_fuzzy_query_tokens)
            for query_token in query_tokens_tuple
        )
        for slot_token in slot_tokens
    )


def _has_entity_only_uncovered_query_tokens(
    query_tokens: frozenset[str],
    candidate_tokens: frozenset[str],
    candidate: Candidate,
    slots: Mapping[str, object],
    static_slots: frozenset[str],
) -> bool:
    """Return whether an action-implicit target candidate leaves query words unexplained.

    A template with an empty literal variant can express its intent using only
    target text, even when other variants add optional determiners. Treating
    those determiners as action evidence lets unrelated query verbs explain an
    implicit state-changing intent.
    """
    if not candidate.has_implicit_literal_variant or not query_tokens:
        return False
    if all(slot_name in static_slots for slot_name in slots):
        return False
    if not (candidate.slot_tokens_set & query_tokens):
        return False
    return bool(query_tokens - candidate_tokens)


def _has_static_entity_uncovered_query_tokens(
    query_tokens_tuple: tuple[str, ...],
    candidate_tokens: frozenset[str],
    slots: Mapping[str, object],
    static_slots: frozenset[str],
    slot_fuzzy_memo: dict[str, tuple[set[str], set[str]]] | None = None,
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
    memo = slot_fuzzy_memo if slot_fuzzy_memo is not None else {}
    for query_token in query_tokens_tuple:
        if query_token in candidate_tokens:
            continue
        if not _token_fuzzy_matches_any(query_token, candidate_tokens, memo):
            return True
    return False


def _has_static_slot_query_conflict(
    query_slot_tokens: frozenset[str],
    candidate_tokens: frozenset[str],
    slots: Mapping[str, object],
    static_slots: frozenset[str],
    slot_fuzzy_memo: dict[str, tuple[set[str], set[str]]] | None = None,
) -> bool:
    """Return whether a broad static slot candidate misses explicit query slots."""
    if not query_slot_tokens:
        return False
    if all(slot_name not in static_slots for slot_name in slots):
        return False
    memo = slot_fuzzy_memo if slot_fuzzy_memo is not None else {}
    for query_token in query_slot_tokens:
        if query_token in candidate_tokens:
            continue
        if not _token_fuzzy_matches_any(query_token, candidate_tokens, memo):
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

    rehydrated_grams = char_ngrams_normalized(rehydrated_norm)
    if rehydrated_grams and query_grams:
        intersection = len(rehydrated_grams & query_grams)
        union = len(rehydrated_grams | query_grams)
        char_score = intersection / union if union else 0.0
    else:
        char_score = 0.0

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
    short_fuzzy_query_tokens: frozenset[str]
    non_entity_tokens: frozenset[str] | None
    candidate_slot_tokens: tuple[frozenset[str], ...] | None
    slot_tokens_by_index: dict[int, frozenset[str]]
    min_confidence: float
    normalized_context: NormalizedIntentContext
    explicit_context_keys: frozenset[str]
    wildcard_passed_set: frozenset[int] | set[int]
    rehydrated_cache: dict[int, tuple[str, dict[str, str]]]
    intent_score_cache: dict[tuple[frozenset[str], ...], float]
    literal_analysis_cache: dict[
        tuple[str, tuple[frozenset[str], ...]], list[_LiteralVariantAnalysis]
    ]
    slot_fuzzy_memo: dict[str, tuple[set[str], set[str]]] = field(default_factory=dict)
    positional_score_cache: dict[tuple[int, frozenset[str]], float] = field(default_factory=dict)
    analysis_union_cache: dict[int, frozenset[str]] = field(default_factory=dict)
    anchor_memo: dict[str, bool] = field(default_factory=dict)
    entity_anchor_memo: dict[str, bool] = field(default_factory=dict)


def _best_positional_score(
    analysis: Sequence[_LiteralVariantAnalysis],
    candidate_tokens: frozenset[str],
) -> float:
    """Calculate the best score from positional hits analysis."""
    best = 0.0
    for variant_analysis in analysis:
        matched = float(len(variant_analysis.exact_match_tokens))
        if variant_analysis.positional_query_tokens.isdisjoint(candidate_tokens):
            matched += (
                variant_analysis.positional_match_count * POSITIONAL_SIMILARITY_PARTIAL_CREDIT
            )
        elif variant_analysis.requires_unique_alignment:
            positional_count = _maximum_unique_positional_matches(
                variant_analysis.positional_hits,
                candidate_tokens | variant_analysis.exact_match_tokens,
            )
            matched += positional_count * POSITIONAL_SIMILARITY_PARTIAL_CREDIT
        else:
            for hits in variant_analysis.positional_hits:
                if not hits.issubset(candidate_tokens):
                    matched += POSITIONAL_SIMILARITY_PARTIAL_CREDIT
        score = (
            matched / variant_analysis.total_token_count
            if variant_analysis.total_token_count
            else 0.0
        )
        if score > best:
            best = score
    return best


def _best_positional_score_memoized(
    analysis: list[_LiteralVariantAnalysis],
    candidate_tokens: frozenset[str],
    context: _ScoringContext,
) -> float:
    """Return the best positional score, memoized per ranking pass.

    The score depends on ``candidate_tokens`` only through its intersection
    with the analysis's positional query tokens (every subset/disjoint check
    inside operates on hit sets drawn from that union), so hundreds of
    candidates sharing the same literal text collapse onto a handful of
    intersection keys. Analyses are identified by ``id`` because they are owned by
    ``literal_analysis_cache`` and live for the whole pass.
    """
    analysis_id = id(analysis)
    union = context.analysis_union_cache.get(analysis_id)
    if union is None:
        union = frozenset(
            query_token
            for variant_analysis in analysis
            for query_token in variant_analysis.positional_query_tokens
        )
        context.analysis_union_cache[analysis_id] = union
    relevant_tokens = candidate_tokens & union
    cache_key = (analysis_id, relevant_tokens)
    score = context.positional_score_cache.get(cache_key)
    if score is None:
        score = _best_positional_score(analysis, relevant_tokens)
        context.positional_score_cache[cache_key] = score
    return score


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


def _token_fuzzy_matches_any(
    token: str,
    others: Set[str],
    memo: dict[str, tuple[set[str], set[str]]],
) -> bool:
    """Return whether token fuzzy-matches any element of others.

    ``_slot_tokens_fuzzy_match`` is symmetric and pure, so verdicts are
    memoized per ranking pass keyed by token: hundreds of candidates repeat
    the same token pairs and the memo turns the common case into a C-level
    ``isdisjoint`` instead of a Python pair loop per candidate.
    """
    entry = memo.get(token)
    if entry is None:
        entry = (set(), set())
        memo[token] = entry
    matched, unmatched = entry
    if not matched.isdisjoint(others):
        return True
    for other in others:
        if other in unmatched or other in matched:
            continue
        if _slot_tokens_fuzzy_match(token, other):
            matched.add(other)
            return True
        unmatched.add(other)
    return False


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
    memo = context.slot_fuzzy_memo
    query_matched = sum(
        q_tok in allowed_cand_tokens or _token_fuzzy_matches_any(q_tok, allowed_cand_tokens, memo)
        for q_tok in context.query_slot_tokens
    )
    if query_matched == len(context.query_slot_tokens):
        return 1.0

    cand_matched = sum(
        c_tok in context.query_tokens or _token_fuzzy_matches_any(c_tok, context.query_tokens, memo)
        for c_tok in cand_slot_tokens
    )
    cand_coverage = cand_matched / len(cand_slot_tokens)
    query_coverage = query_matched / len(context.query_slot_tokens)

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
        and not _token_fuzzy_matches_any(q_tok, allowed_cand_tokens, context.slot_fuzzy_memo)
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
    """Apply query/slot consistency penalties to the score."""
    if _has_unconsumed_query_number(candidate_tokens, context.query_numbers):
        score *= QUERY_NUMBER_WITHOUT_CANDIDATE_PENALTY

    slots = candidate.parsed_slots
    if not slots:
        return score

    static_slots = _static_slot_names(candidate)
    score = _apply_numeric_slot_penalties(
        candidate,
        slots,
        static_slots,
        context,
        score,
    )
    score = _apply_slot_anchor_penalties(
        candidate,
        slots,
        static_slots,
        context,
        score,
    )
    return _apply_slot_coverage_penalties(
        candidate,
        candidate_tokens,
        slots,
        static_slots,
        context,
        score,
    )


def _apply_numeric_slot_penalties(
    candidate: Candidate,
    slots: Mapping[str, object],
    static_slots: frozenset[str],
    context: _ScoringContext,
    score: float,
) -> float:
    """Apply penalties for missing or conflicting numeric query evidence."""
    if _has_unanchored_numeric_slot(
        slots,
        static_slots,
        query_has_number=context.query_has_number,
    ):
        score *= NUMERIC_SLOT_WITHOUT_QUERY_PENALTY
        if _has_unanchored_numeric_literal(
            candidate,
            query_has_number=context.query_has_number,
        ):
            score *= NUMERIC_LITERAL_WITHOUT_QUERY_PENALTY
    if _has_numeric_slot_mismatch(candidate, slots, context.query_numbers, static_slots):
        score *= NUMERIC_SLOT_MISMATCH_PENALTY
    return score


def _apply_slot_anchor_penalties(
    candidate: Candidate,
    slots: Mapping[str, object],
    static_slots: frozenset[str],
    context: _ScoringContext,
    score: float,
) -> float:
    """Apply penalties for entity and semantic slots lacking query anchors."""
    if _has_unanchored_entity_slot(
        slots,
        static_slots,
        context.query_tokens_tuple,
        context.short_fuzzy_query_tokens,
        context.entity_anchor_memo,
    ):
        score *= UNANCHORED_ENTITY_SLOT_PENALTY
    if _has_unanchored_semantic_slot(
        candidate,
        slots,
        static_slots,
        context.query_tokens_tuple,
        context.short_fuzzy_query_tokens,
        context.anchor_memo,
    ):
        score *= UNANCHORED_SEMANTIC_SLOT_PENALTY
    return score


def _apply_slot_coverage_penalties(
    candidate: Candidate,
    candidate_tokens: frozenset[str],
    slots: Mapping[str, object],
    static_slots: frozenset[str],
    context: _ScoringContext,
    score: float,
) -> float:
    """Apply penalties for uncovered query tokens and static-slot conflicts."""
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
        context.slot_fuzzy_memo,
    ):
        score *= STATIC_ENTITY_UNCOVERED_QUERY_PENALTY
    if _has_static_slot_query_conflict(
        context.query_slot_tokens,
        candidate_tokens,
        slots,
        static_slots,
        context.slot_fuzzy_memo,
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


def _cached_exact_intent_score(
    candidate_tokens: frozenset[str],
    literal_variants: tuple[frozenset[str], ...],
    context: _ScoringContext,
) -> float:
    """Return cached exact literal evidence with empty-variant safeguards."""
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
    return exact


def _positional_literal_intent_score(
    literal_text: str,
    literal_variants: tuple[frozenset[str], ...],
    candidate_tokens: frozenset[str],
    context: _ScoringContext,
) -> float:
    """Return memoized positional evidence for a partially matched literal."""
    analysis_key = (literal_text, literal_variants)
    analysis = context.literal_analysis_cache.get(analysis_key)
    if analysis is None:
        analysis = _precompute_literal_analysis(
            literal_variants,
            context.query_tokens,
            context.positional_lookup,
        )
        context.literal_analysis_cache[analysis_key] = analysis
    return _best_positional_score_memoized(analysis, candidate_tokens, context)


def _calculate_intent_score(
    candidate: Candidate,
    candidate_tokens: frozenset[str],
    literal_variants: tuple[frozenset[str], ...],
    total_unique_literal_tokens: int,
    context: _ScoringContext,
) -> float:
    """Calculate literal coverage and positional intent evidence."""
    literal_text = candidate.metadata.get("literal_text")
    intent_score = _query_token_coverage(context.query_tokens, candidate_tokens)
    if not literal_text:
        return intent_score

    exact = _cached_exact_intent_score(
        candidate_tokens,
        literal_variants,
        context,
    )
    if exact >= 1.0:
        if total_unique_literal_tokens >= 2:
            matched_q = len(context.query_tokens & candidate_tokens)
            intent_score = matched_q / len(context.query_tokens) if context.query_tokens else 1.0
        elif candidate.slot_tokens_set:
            intent_score = exact
    elif context.query_tokens.issubset(candidate_tokens) or not context.positional_lookup:
        intent_score = exact
    else:
        intent_score = _positional_literal_intent_score(
            literal_text,
            literal_variants,
            candidate_tokens,
            context,
        )
    if context.positional_literal_tokens:
        penalty = _non_entity_coverage(
            context.query_tokens,
            context.positional_literal_tokens,
            candidate_tokens,
            non_entity=context.non_entity_tokens,
        )
        intent_score *= 1.0 - NON_ENTITY_PENALTY_BLEND + NON_ENTITY_PENALTY_BLEND * penalty
    return intent_score


def _rehydrated_wildcard_state(
    idx: int,
    candidate: Candidate,
    char_score: float,
    bm25_score: float,
    context: _ScoringContext,
) -> tuple[tuple[str, dict[str, str]] | None, float, float] | None:
    """Rehydrate a wildcard candidate and rescore its lexical components.

    Returns ``None`` when the candidate must be dropped (failed prefilter or
    unresolvable rehydration); otherwise ``(rehydrated, char_score,
    bm25_score)`` where ``rehydrated`` is ``None`` for plain candidates.
    """
    if not candidate.has_wildcard:
        return None, char_score, bm25_score
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
    return (rehydrated_norm, replacements), char_score, bm25_score


def _apply_context_adjustment(
    combined: float,
    slot_penalty: float,
    candidate: Candidate,
    context: _ScoringContext,
) -> float:
    """Adjust the combined score by intent-context agreement.

    A positive adjustment scales into the remaining headroom instead of
    clipping at 1.0 so two strong same-area candidates keep their pre-boost
    margin, and is scaled by the slot-match fraction: the slot-conflict blend
    deliberately floors fully conflicting candidates below the acceptance
    gate, and an additive context boost must not lift a wrong-device
    candidate back over it.
    """
    context_adjustment = _context_slot_adjustment(
        candidate.parsed_slots,
        _context_slot_names(candidate),
        context.normalized_context,
        context.explicit_context_keys,
    )
    if context_adjustment > 0.0:
        combined += (
            CONTEXT_SLOT_MATCH_BOOST * context_adjustment * slot_penalty * max(0.0, 1.0 - combined)
        )
    elif context_adjustment < 0.0:
        combined *= CONTEXT_SLOT_MISMATCH_PENALTY ** abs(context_adjustment)
    return combined


def _wildcard_length_penalty(rehydrated: tuple[str, dict[str, str]] | None) -> float:
    """Return the length penalty favoring templates with more literal tokens."""
    if rehydrated is None:
        return 0.0
    _, replacements = rehydrated
    wc_len = sum(len(val.split()) for val in replacements.values())
    return WILDCARD_LENGTH_PENALTY_FACTOR * wc_len


class _CandidateLexicalState(NamedTuple):
    """Lexical evidence computed before slot and context penalties."""

    rehydrated: tuple[str, dict[str, str]] | None
    candidate_tokens: frozenset[str]
    rapidfuzz_score: float
    char_score: float
    bm25_score: float
    intent_score: float


def _candidate_lexical_state(
    idx: int,
    candidate: Candidate,
    bm25_score: float,
    char_score: float,
    context: _ScoringContext,
) -> _CandidateLexicalState | None:
    """Compute lexical evidence or return None when wildcard filtering rejects it."""
    wildcard_state = _rehydrated_wildcard_state(idx, candidate, char_score, bm25_score, context)
    if wildcard_state is None:
        return None
    rehydrated, char_score, bm25_score = wildcard_state

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
    return _CandidateLexicalState(
        rehydrated=rehydrated,
        candidate_tokens=candidate_tokens,
        rapidfuzz_score=rapidfuzz_score,
        char_score=char_score,
        bm25_score=bm25_score,
        intent_score=intent_score,
    )


def _adjust_candidate_score(
    idx: int,
    candidate: Candidate,
    lexical_state: _CandidateLexicalState,
    context: _ScoringContext,
) -> tuple[float, float]:
    """Apply slot, context, and wildcard-length adjustments."""
    combined = lexical_score(
        lexical_state.rapidfuzz_score,
        lexical_state.char_score,
        lexical_state.bm25_score,
        lexical_state.intent_score,
    )
    slot_penalty, wildcard_tokens, cand_slot_tokens = _calculate_slot_penalty(
        idx,
        candidate,
        context,
        lexical_state.rehydrated,
    )
    if slot_penalty < 1.0:
        base_multiplier = min(1.0, max(0.1, context.min_confidence - 0.05))
        combined *= base_multiplier + (1.0 - base_multiplier) * slot_penalty
    if _has_wildcard_known_slot_token_absorption(
        context.query_slot_tokens,
        wildcard_tokens,
        cand_slot_tokens,
    ):
        combined *= WILDCARD_KNOWN_SLOT_TOKEN_PENALTY

    combined = _apply_candidate_slot_penalties(
        candidate,
        lexical_state.candidate_tokens,
        context,
        combined,
    )
    combined = _apply_context_adjustment(combined, slot_penalty, candidate, context)
    penalty = _wildcard_length_penalty(lexical_state.rehydrated)
    if penalty:
        combined = max(combined - penalty, 0.0)
    return combined, penalty


def _score_single_candidate(
    idx: int,
    candidate: Candidate,
    bm25_score: float,
    char_score: float,
    context: _ScoringContext,
) -> _RankedItem | None:
    """Score one candidate or return None when wildcard filtering rejects it."""
    lexical_state = _candidate_lexical_state(
        idx,
        candidate,
        bm25_score,
        char_score,
        context,
    )
    if lexical_state is None:
        return None
    final_score, penalty = _adjust_candidate_score(idx, candidate, lexical_state, context)

    return _RankedItem(
        final_score=final_score,
        candidate=candidate,
        rapidfuzz_score=lexical_state.rapidfuzz_score,
        char_ngram_score=lexical_state.char_score,
        bm25_score=lexical_state.bm25_score,
        intent_score=lexical_state.intent_score,
        index=idx,
        penalty=penalty,
        slot_specificity=len(candidate.parsed_slots),
    )


def _validate_rank_inputs(
    candidates: Sequence[Candidate],
    max_candidates: int,
    rapidfuzz_prefilter_candidates: int,
    bm25_index: BM25Index | None,
    reference_bm25_index: BM25Index | None,
    candidate_char_index: CharNGramIndex | None,
    candidate_slot_tokens: tuple[frozenset[str], ...] | None,
    slot_token_index: SlotTokenIndex | None,
) -> None:
    """Raise ValueError when ranking inputs are inconsistent with candidates."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if rapidfuzz_prefilter_candidates < max_candidates:
        raise ValueError("rapidfuzz_prefilter_candidates must be at least max_candidates")
    if candidate_char_index is not None and len(candidate_char_index.gram_counts) != len(
        candidates
    ):
        raise ValueError("candidate_char_index length must match candidates")
    if candidate_slot_tokens is not None and len(candidate_slot_tokens) != len(candidates):
        raise ValueError("candidate_slot_tokens length must match candidates")
    if slot_token_index is not None and slot_token_index.candidate_count != len(candidates):
        raise ValueError("slot_token_index length must match candidates")
    if (
        reference_bm25_index is None
        and bm25_index is not None
        and bm25_index.size != len(candidates)
    ):
        raise ValueError("bm25_index length must match candidates")


def _collect_positional_literal_tokens(candidates: Sequence[Candidate]) -> frozenset[str]:
    """Derive the literal-token inventory from candidate metadata."""
    all_tokens: set[str] = set()
    for candidate in candidates:
        if literal_text := candidate.metadata.get("literal_text"):
            for variant in literal_token_variants(literal_text):
                all_tokens.update(variant)
    return frozenset(all_tokens)


def _prefilter_top_indices(
    candidates: Sequence[Candidate],
    candidate_char_index: CharNGramIndex,
    query_grams: frozenset[str],
    bm25_scoring: _Bm25Scoring,
    prefilter_limit: int,
) -> tuple[list[int], Sequence[float], Sequence[int] | None, Sequence[float] | None]:
    """Select the lexically strongest candidate indices for full scoring.

    Returns ``(top_indices, prefilter_keys, char_intersections, char_scores)``
    where ``char_intersections`` holds per-candidate gram intersection sizes;
    exactly one of the char representations is populated depending on whether
    the sparse (static-pass) or dense (reference-index) BM25 path is active.
    """
    if bm25_scoring.sparse_raw_scores is not None:
        char_intersections, char_touched = candidate_char_index.intersections_with_touched(
            query_grams
        )
        sparse_keys = _sparse_prefilter_keys_from_intersections(
            char_intersections,
            char_touched,
            len(query_grams),
            candidate_char_index.gram_counts,
            bm25_scoring.sparse_raw_scores,
            bm25_scoring.sparse_inv_max,
        )
        prefilter_keys: Sequence[float] = _SparsePrefilterKeys(sparse_keys, len(candidates))
        top_indices = _top_sparse_prefilter_indices(sparse_keys, len(candidates), prefilter_limit)
        return top_indices, prefilter_keys, char_intersections, None
    char_scores = candidate_char_index.score(query_grams)
    prefilter_keys = bm25_scoring.prefilter_keys(char_scores)
    top_indices = _top_prefilter_indices(prefilter_keys, prefilter_limit)
    return top_indices, prefilter_keys, None, char_scores


def _extend_with_wildcard_rescue(
    top_indices: list[int],
    wildcard_passed_set: frozenset[int] | set[int],
    query_tokens: frozenset[str],
    wildcard_variant_analyses: dict[int, tuple[WildcardVariantAnalysis, ...]] | None,
    prefilter_keys: Sequence[float],
    prefilter_limit: int,
) -> None:
    """Append wildcard candidates that passed prefiltering but missed the cut."""
    if not wildcard_passed_set:
        return
    additional_wildcards = set(wildcard_passed_set) - set(top_indices)
    top_indices.extend(
        _top_additional_wildcard_indices(
            additional_wildcards,
            query_tokens,
            wildcard_variant_analyses,
            prefilter_keys,
            min(prefilter_limit, DEFAULT_WILDCARD_PREFILTER_RESCUE_CANDIDATES),
        )
    )


def _derive_query_slot_tokens(
    candidates: Sequence[Candidate],
    top_indices: Sequence[int],
    query_tokens: frozenset[str],
    candidate_slot_tokens: tuple[frozenset[str], ...] | None,
    reference_slot_token_index: SlotTokenIndex | None,
    positional_literal_tokens: frozenset[str] | None,
) -> tuple[dict[int, frozenset[str]], frozenset[str]]:
    """Return per-candidate slot tokens and the query tokens they evidence.

    Both branches use the same fuzzy matcher so the static and dynamic passes
    agree on which query tokens are slot evidence; a bare exact intersection
    would let a dynamic duplicate of a static candidate escape the
    slot-conflict penalties at merge time.
    """
    if candidate_slot_tokens is None:
        slot_tokens_by_index = {idx: candidates[idx].slot_tokens_set for idx in top_indices}
        active_slot_tokens = frozenset(
            token for tokens in slot_tokens_by_index.values() for token in tokens
        )
        query_slot_tokens = _query_slot_tokens_from_active(
            query_tokens,
            active_slot_tokens,
            fuzzy_excluded_tokens=positional_literal_tokens,
        )
    else:
        slot_tokens_by_index = {}
        query_slot_tokens = _query_slot_tokens_from_candidates(
            query_tokens,
            top_indices,
            candidate_slot_tokens,
            positional_literal_tokens,
        )
    if reference_slot_token_index is not None:
        query_slot_tokens |= _slot_index_query_matches(
            query_tokens,
            reference_slot_token_index,
            positional_literal_tokens,
        )
    return slot_tokens_by_index, query_slot_tokens


def _neutralized_context_keys(
    candidates: Sequence[Candidate],
    top_indices: Sequence[int],
    normalized_context: NormalizedIntentContext,
    explicit_context_keys: frozenset[str],
    query_tokens: frozenset[str],
    positional_literal_tokens: frozenset[str],
    positional_lookup: Mapping[str, frozenset[str]],
    query_slot_tokens: frozenset[str],
) -> frozenset[str]:
    """Neutralize context influence when the query names unknown vocabulary.

    Context disambiguates between resolvable targets; it must never boost a
    candidate over the acceptance margin when the query names vocabulary the
    corpus cannot resolve. Treating every context key as explicitly stated
    makes context neither boost nor penalize.
    """
    if not normalized_context:
        return explicit_context_keys
    unknown_content_tokens = _query_unknown_content_tokens(
        query_tokens,
        positional_literal_tokens,
        positional_lookup,
        query_slot_tokens,
    )
    if unknown_content_tokens and any(
        all(token not in candidates[idx].normalized_tokens_set for idx in top_indices)
        for token in unknown_content_tokens
    ):
        return explicit_context_keys | frozenset(normalized_context)
    return explicit_context_keys


def _score_top_candidates(
    candidates: Sequence[Candidate],
    top_indices: Sequence[int],
    candidate_char_index: CharNGramIndex,
    char_intersections: Sequence[int] | None,
    char_scores: Sequence[float] | None,
    query_grams: frozenset[str],
    bm25_scoring: _Bm25Scoring,
    context: _ScoringContext,
) -> list[_RankedItem]:
    """Score every prefiltered candidate and drop the rejected ones."""
    ranked_tuples: list[_RankedItem] = []
    for idx in top_indices:
        candidate = candidates[idx]
        bm25_score = bm25_scoring.score_at(idx)
        char_score = (
            _char_ngram_score_from_intersection(
                len(query_grams),
                candidate_char_index.gram_counts[idx],
                char_intersections[idx],
            )
            if char_intersections is not None
            else char_scores[idx]
            if char_scores is not None
            else 0.0
        )
        item = _score_single_candidate(
            idx,
            candidate,
            bm25_score,
            char_score,
            context,
        )
        if item is not None:
            ranked_tuples.append(item)
    return ranked_tuples


def _finalize_ranked_candidates(
    ranked_tuples: list[_RankedItem],
    slot_preferences: set[tuple[str, str]] | None,
    rehydrated_cache: dict[int, tuple[str, dict[str, str]]],
    disambiguation_limit: int,
    max_candidates: int,
) -> tuple[RankedCandidate, ...]:
    """Sort scored items, cap the list, and resolve intent ties."""
    intent_tie_preferences = _intent_tie_preferences_by_index(
        ranked_tuples,
        slot_preferences=slot_preferences,
        rehydrated_cache=rehydrated_cache,
    )
    ranked_tuples.sort(
        key=partial(
            _ranked_tuple_sort_key,
            slot_preferences=slot_preferences,
            rehydrated_cache=rehydrated_cache,
            intent_tie_preferences=intent_tie_preferences,
        ),
        reverse=True,
    )
    bounded_ranked_tuples = _limit_ranked_items(
        ranked_tuples,
        disambiguation_limit,
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
        for item in bounded_ranked_tuples
    ]
    apply_intent_disambiguation(ranked)
    return tuple(ranked[:max_candidates])


class _RankQueryState(NamedTuple):
    """Normalized query values reused throughout one ranking pass."""

    normalized: str
    tokens: frozenset[str]
    token_sequence: tuple[str, ...]
    token_count: int
    sorted_tokens: str
    non_entity_tokens: frozenset[str] | None
    grams: frozenset[str]
    positional_lookup: dict[str, frozenset[str]]
    short_fuzzy_tokens: frozenset[str]


class _WildcardPrefilterInputs(NamedTuple):
    """Wildcard indexes and analyses consumed during candidate prefiltering."""

    always_passes: frozenset[int] | None
    variant_analyses: dict[int, tuple[WildcardVariantAnalysis, ...]] | None
    token_to_indices: dict[str, tuple[int, ...]] | None
    literal_tokens_by_index: dict[int, frozenset[str]] | None
    min_required_by_index: dict[int, int] | None
    variant_groups: tuple[WildcardVariantGroup, ...] | None


class _RankPrefilterState(NamedTuple):
    """Candidate-selection state passed into final scoring."""

    bm25_scoring: _Bm25Scoring
    char_index: CharNGramIndex
    top_indices: list[int]
    char_intersections: Sequence[int] | None
    char_scores: Sequence[float] | None
    wildcard_passed: frozenset[int] | set[int]


def _prepare_rank_query_state(
    query_normalized: str,
    positional_literal_tokens: frozenset[str] | None,
) -> _RankQueryState:
    """Build normalized query representations used by ranking stages."""
    literal_tokens = positional_literal_tokens or frozenset()
    (
        query_tokens,
        query_tokens_tuple,
        query_token_count,
        query_sorted,
        non_entity_tokens,
    ) = _rank_query_setup(query_normalized, literal_tokens)
    return _RankQueryState(
        normalized=query_normalized,
        tokens=query_tokens,
        token_sequence=query_tokens_tuple,
        token_count=query_token_count,
        sorted_tokens=query_sorted,
        non_entity_tokens=non_entity_tokens,
        grams=char_ngrams_normalized(query_normalized),
        positional_lookup=_build_positional_lookup(literal_tokens, query_tokens)
        if literal_tokens
        else {},
        short_fuzzy_tokens=_short_nonliteral_query_tokens(
            query_tokens,
            positional_literal_tokens,
        ),
    )


def _prepare_rank_prefilter(
    candidates: Sequence[Candidate],
    query_state: _RankQueryState,
    max_candidates: int,
    rapidfuzz_prefilter_candidates: int,
    bm25_index: BM25Index | None,
    reference_bm25_index: BM25Index | None,
    candidate_char_index: CharNGramIndex | None,
    candidate_slot_tokens: tuple[frozenset[str], ...] | None,
    slot_token_index: SlotTokenIndex | None,
    positional_literal_tokens: frozenset[str] | None,
    wildcard_inputs: _WildcardPrefilterInputs,
) -> _RankPrefilterState:
    """Build lexical indexes and select candidates for full scoring."""
    bm25_scoring = _prepare_bm25_scoring(
        candidates,
        query_state.token_sequence,
        bm25_index,
        reference_bm25_index,
    )
    if candidate_char_index is None:
        candidate_char_index = CharNGramIndex.from_grams(
            tuple(char_ngrams_normalized(candidate.normalized_text) for candidate in candidates)
        )
    prefilter_limit = _rank_prefilter_limit(
        len(candidates),
        max_candidates,
        rapidfuzz_prefilter_candidates,
    )
    top_indices, prefilter_keys, char_intersections, char_scores = _prefilter_top_indices(
        candidates,
        candidate_char_index,
        query_state.grams,
        bm25_scoring,
        prefilter_limit,
    )
    if candidate_slot_tokens is not None and slot_token_index is not None:
        top_indices.extend(
            _top_additional_slot_indices(
                query_state.tokens,
                candidate_slot_tokens,
                slot_token_index,
                top_indices,
                prefilter_keys,
                fuzzy_excluded_tokens=positional_literal_tokens,
            )
        )
    wildcard_passed = _prefilter_wildcard_candidates(
        candidates,
        query_state.tokens,
        wildcard_inputs.always_passes,
        wildcard_inputs.variant_analyses,
        wildcard_inputs.token_to_indices,
        wildcard_inputs.literal_tokens_by_index,
        wildcard_inputs.min_required_by_index,
        wildcard_inputs.variant_groups,
    )
    _extend_with_wildcard_rescue(
        top_indices,
        wildcard_passed,
        query_state.tokens,
        wildcard_inputs.variant_analyses,
        prefilter_keys,
        prefilter_limit,
    )
    return _RankPrefilterState(
        bm25_scoring=bm25_scoring,
        char_index=candidate_char_index,
        top_indices=top_indices,
        char_intersections=char_intersections,
        char_scores=char_scores,
        wildcard_passed=wildcard_passed,
    )


def _build_rank_scoring_context(
    query: str,
    candidates: Sequence[Candidate],
    query_state: _RankQueryState,
    prefilter_state: _RankPrefilterState,
    positional_literal_tokens: frozenset[str] | None,
    candidate_slot_tokens: tuple[frozenset[str], ...] | None,
    reference_slot_token_index: SlotTokenIndex | None,
    intent_context: Mapping[str, object] | None,
    min_confidence: float,
    rehydrated_cache: dict[int, tuple[str, dict[str, str]]],
) -> _ScoringContext:
    """Build candidate-independent evidence and caches for final scoring."""
    normalized_context = normalize_intent_context(intent_context)
    anchor_memo: dict[str, bool] = {}
    explicit_context_keys = _explicit_context_keys_from_query(
        candidates,
        prefilter_state.top_indices,
        normalized_context,
        query_state.token_sequence,
        query_state.short_fuzzy_tokens,
        anchor_memo,
    )
    slot_tokens_by_index, query_slot_tokens = _derive_query_slot_tokens(
        candidates,
        prefilter_state.top_indices,
        query_state.tokens,
        candidate_slot_tokens,
        reference_slot_token_index,
        positional_literal_tokens,
    )
    explicit_context_keys = _neutralized_context_keys(
        candidates,
        prefilter_state.top_indices,
        normalized_context,
        explicit_context_keys,
        query_state.tokens,
        positional_literal_tokens or frozenset(),
        query_state.positional_lookup,
        query_slot_tokens,
    )
    return _ScoringContext(
        query=query,
        query_normalized=query_state.normalized,
        query_tokens=query_state.tokens,
        query_tokens_tuple=query_state.token_sequence,
        query_token_count=query_state.token_count,
        query_sorted=query_state.sorted_tokens,
        query_grams=query_state.grams,
        query_slot_tokens=query_slot_tokens,
        query_has_number=_has_numeric_query_token(query_state.tokens),
        query_numbers=_query_numeric_values(query_state.tokens),
        bm25_ref=prefilter_state.bm25_scoring.index,
        max_raw_score=prefilter_state.bm25_scoring.max_raw_score,
        positional_lookup=query_state.positional_lookup,
        positional_literal_tokens=positional_literal_tokens,
        short_fuzzy_query_tokens=query_state.short_fuzzy_tokens,
        non_entity_tokens=query_state.non_entity_tokens,
        candidate_slot_tokens=candidate_slot_tokens,
        slot_tokens_by_index=slot_tokens_by_index,
        min_confidence=min_confidence,
        normalized_context=normalized_context,
        explicit_context_keys=explicit_context_keys,
        wildcard_passed_set=prefilter_state.wildcard_passed,
        rehydrated_cache=rehydrated_cache,
        intent_score_cache={},
        literal_analysis_cache={},
        anchor_memo=anchor_memo,
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
    wildcard_variant_analyses: dict[int, tuple[WildcardVariantAnalysis, ...]] | None = None,
    wildcard_token_to_indices: dict[str, tuple[int, ...]] | None = None,
    wildcard_literal_tokens_by_index: dict[int, frozenset[str]] | None = None,
    wildcard_min_required_by_index: dict[int, int] | None = None,
    wildcard_variant_groups: tuple[WildcardVariantGroup, ...] | None = None,
    candidate_slot_tokens: tuple[frozenset[str], ...] | None = None,
    slot_token_index: SlotTokenIndex | None = None,
    reference_slot_token_index: SlotTokenIndex | None = None,
    slot_preferences: set[tuple[str, str]] | None = None,
    intent_context: Mapping[str, object] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[RankedCandidate, ...]:
    """Rank candidates for a query using lexical scoring.

    Candidate text containing wildcard placeholders is rehydrated from
    *query* before scoring so that real free-text values contribute to
    the semantic match instead of placeholder tokens.
    A supplied ``bm25_index`` must have been built from ``candidates`` in the
    same order. ``reference_bm25_index`` deliberately uses a separate corpus
    and scores the supplied candidates through its dense path instead.
    ``intent_context`` accepts HassIL-style context mappings and is normalized
    by ``utils.normalize_intent_context``.
    ``reference_slot_token_index`` contributes known target tokens from a
    separate static corpus without assuming index alignment with ``candidates``.
    """
    _validate_rank_inputs(
        candidates,
        max_candidates,
        rapidfuzz_prefilter_candidates,
        bm25_index,
        reference_bm25_index,
        candidate_char_index,
        candidate_slot_tokens,
        slot_token_index,
    )
    if not candidates:
        return ()

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
    if not query_normalized:
        return ()

    if positional_literal_tokens is None:
        positional_literal_tokens = _collect_positional_literal_tokens(candidates)
    query_state = _prepare_rank_query_state(query_normalized, positional_literal_tokens)
    prefilter_state = _prepare_rank_prefilter(
        candidates,
        query_state,
        max_candidates,
        rapidfuzz_prefilter_candidates,
        bm25_index,
        reference_bm25_index,
        candidate_char_index,
        candidate_slot_tokens,
        slot_token_index,
        positional_literal_tokens,
        _WildcardPrefilterInputs(
            always_passes=wildcard_always_passes,
            variant_analyses=wildcard_variant_analyses,
            token_to_indices=wildcard_token_to_indices,
            literal_tokens_by_index=wildcard_literal_tokens_by_index,
            min_required_by_index=wildcard_min_required_by_index,
            variant_groups=wildcard_variant_groups,
        ),
    )
    rehydrated_cache: dict[int, tuple[str, dict[str, str]]] = {}
    context = _build_rank_scoring_context(
        query,
        candidates,
        query_state,
        prefilter_state,
        positional_literal_tokens,
        candidate_slot_tokens,
        reference_slot_token_index,
        intent_context,
        min_confidence,
        rehydrated_cache,
    )
    ranked_tuples = _score_top_candidates(
        candidates,
        prefilter_state.top_indices,
        prefilter_state.char_index,
        prefilter_state.char_intersections,
        prefilter_state.char_scores,
        query_state.grams,
        prefilter_state.bm25_scoring,
        context,
    )
    return _finalize_ranked_candidates(
        ranked_tuples,
        slot_preferences,
        rehydrated_cache,
        max(2, max_candidates),
        max_candidates,
    )


def _candidate_slot_tokens_at(
    index: int,
    candidate_slot_tokens: tuple[frozenset[str], ...] | None,
    slot_tokens_by_index: dict[int, frozenset[str]],
) -> frozenset[str]:
    """Return precomputed slot tokens for a candidate index."""
    if candidate_slot_tokens is not None:
        return candidate_slot_tokens[index]
    return slot_tokens_by_index.get(index, frozenset())


def _payload_tokens_have_query_anchor(
    payload_tokens: frozenset[str],
    query_tokens: tuple[str, ...],
    query_tokens_no_diacritics: frozenset[str],
    language: str | None,
) -> bool:
    """Return whether every distinguishing payload token is query-supported."""
    if not payload_tokens:
        return False
    for payload_token in payload_tokens:
        payload_token_no_diacritics = normalize_text_no_diacritics(payload_token, language)
        if (
            payload_token in query_tokens
            or payload_token_no_diacritics in query_tokens_no_diacritics
            or any(
                _is_fuzzy_slot_token_match(payload_token, query_token)
                or _has_bounded_shared_stem(payload_token, query_token)
                or _slot_tokens_fuzzy_match(payload_token, query_token)
                for query_token in query_tokens
            )
        ):
            continue
        return False
    return True


def _candidate_has_supported_literal_variant(
    candidate: Candidate,
    query_tokens: tuple[str, ...],
    query_tokens_no_diacritics: frozenset[str],
    language: str | None,
) -> bool:
    """Return whether a complete grammar literal variant is query-supported."""
    variants = candidate.literal_variants
    if not variants or not all(variants):
        return True
    return any(
        _payload_tokens_have_query_anchor(
            variant,
            query_tokens,
            query_tokens_no_diacritics,
            language,
        )
        for variant in variants
    )


def _candidate_has_explicit_supported_literal_variant(
    candidate: Candidate,
    query_tokens: tuple[str, ...],
    query_tokens_no_diacritics: frozenset[str],
    language: str | None,
) -> bool:
    """Return whether the query supports one non-empty grammar literal variant."""
    return any(
        variant
        and _payload_tokens_have_query_anchor(
            variant,
            query_tokens,
            query_tokens_no_diacritics,
            language,
        )
        for variant in candidate.literal_variants
    )


def _candidate_covers_query_tokens(
    candidate: Candidate,
    query_tokens: tuple[str, ...],
    language: str | None,
) -> bool:
    """Return whether candidate text explains every query token."""
    candidate_tokens = candidate.normalized_tokens
    candidate_tokens_no_diacritics = frozenset(candidate.normalized_text_no_diacritics.split())
    return _payload_tokens_have_query_anchor(
        frozenset(query_tokens),
        candidate_tokens,
        candidate_tokens_no_diacritics,
        language,
    )


def _wildcard_absorbs_supported_concrete_target(
    candidate: Candidate,
    other: Candidate,
    query_tokens: tuple[str, ...] | None,
    query_tokens_no_diacritics: frozenset[str],
    language: str | None,
) -> bool:
    """Return whether free text consumed a query-supported concrete target."""
    if query_tokens is None or not candidate.has_wildcard or other.has_wildcard:
        return False
    if candidate.concrete_surface_slot_tokens:
        return False
    concrete_target = other.concrete_surface_slot_tokens
    return _payload_tokens_have_query_anchor(
        concrete_target,
        query_tokens,
        query_tokens_no_diacritics,
        language,
    )


def _has_bounded_shared_stem(token: str, query_token: str) -> bool:
    """Return whether two long tokens share a substantial leading stem.

    This is intentionally bounded to long words and requires the common prefix
    to cover at least two thirds of the shorter token. It captures productive
    morphological variants without accepting short same-prefix words.
    """
    shorter_length = min(len(token), len(query_token))
    if shorter_length < 6:
        return False
    common_prefix_length = 0
    for token_char, query_char in zip(token, query_token, strict=False):
        if token_char != query_char:
            break
        common_prefix_length += 1
    return common_prefix_length >= 4 and common_prefix_length * 3 >= shorter_length * 2


def _same_intent_payload_query_support(
    candidate: Candidate,
    other: Candidate,
    query_tokens: tuple[str, ...],
    query_tokens_no_diacritics: frozenset[str],
    language: str | None,
) -> tuple[bool, bool]:
    """Return query support for each candidate's distinguishing slot payload."""
    candidate_payload = candidate.surface_slot_tokens
    other_payload = other.surface_slot_tokens
    shared_payload = candidate_payload & other_payload
    candidate_distinguishing = candidate_payload - other_payload
    other_distinguishing = other_payload - candidate_payload
    return (
        _payload_tokens_have_query_anchor(
            candidate_distinguishing,
            query_tokens,
            query_tokens_no_diacritics,
            language,
        )
        or _single_short_payload_typo_has_context(
            candidate_distinguishing,
            other_distinguishing,
            shared_payload,
            query_tokens,
            query_tokens_no_diacritics,
            language,
        ),
        _payload_tokens_have_query_anchor(
            other_distinguishing,
            query_tokens,
            query_tokens_no_diacritics,
            language,
        )
        or _single_short_payload_typo_has_context(
            other_distinguishing,
            candidate_distinguishing,
            shared_payload,
            query_tokens,
            query_tokens_no_diacritics,
            language,
        ),
    )


def _single_short_payload_typo_has_context(
    payload_tokens: frozenset[str],
    other_payload_tokens: frozenset[str],
    shared_payload_tokens: frozenset[str],
    query_tokens: tuple[str, ...],
    query_tokens_no_diacritics: frozenset[str],
    language: str | None,
) -> bool:
    """Accept one short target typo only inside a strongly anchored target phrase."""
    if (
        len(payload_tokens) != 1
        or not other_payload_tokens
        or len(shared_payload_tokens) < 2
        or not _payload_tokens_have_query_anchor(
            shared_payload_tokens,
            query_tokens,
            query_tokens_no_diacritics,
            language,
        )
    ):
        return False
    payload_token = normalize_text_no_diacritics(next(iter(payload_tokens)), language)
    if len(payload_token) != 3:
        return False
    return any(
        len(query_token) == 3
        and payload_token[0] == query_token[0]
        and sum(
            payload_char != query_char
            for payload_char, query_char in zip(payload_token, query_token, strict=True)
        )
        == 1
        for query_token in query_tokens_no_diacritics
    )


def _surface_payloads_are_nested(candidate: Candidate, other: Candidate) -> bool:
    """Return whether one non-static spoken payload contains the other."""
    candidate_payload = candidate.surface_slot_tokens
    other_payload = other.surface_slot_tokens
    return bool(candidate_payload and other_payload) and (
        candidate_payload.issubset(other_payload) or other_payload.issubset(candidate_payload)
    )


def _candidates_share_executable_target(candidate: Candidate, other: Candidate) -> bool:
    """Return whether candidates describe the same target payload."""
    if candidate.parsed_slots == other.parsed_slots:
        return True
    candidate_payload = candidate.surface_slot_tokens
    other_payload = other.surface_slot_tokens
    return bool(candidate_payload and other_payload) and (
        candidate_payload.issubset(other_payload) or other_payload.issubset(candidate_payload)
    )


def _candidate_has_nonliteral_target_token(
    candidate: Candidate,
    reference: Candidate,
    language: str | None,
) -> bool:
    """Return whether a concrete target contributes evidence beyond grammar words."""
    literal_tokens = frozenset(token for variant in reference.literal_variants for token in variant)
    literal_tokens_no_diacritics = frozenset(
        normalize_text_no_diacritics(token, language) for token in literal_tokens
    )
    return any(
        not any(
            payload_token == literal_token
            or normalize_text_no_diacritics(payload_token, language)
            == normalize_text_no_diacritics(literal_token, language)
            or _is_fuzzy_slot_token_match(payload_token, literal_token)
            or _has_bounded_shared_stem(payload_token, literal_token)
            or _slot_tokens_fuzzy_match(payload_token, literal_token)
            for literal_token in literal_tokens | literal_tokens_no_diacritics
        )
        for payload_token in candidate.surface_slot_tokens
    )


def _intent_group_has_unique_supported_literal(
    ranked: Sequence[RankedCandidate],
    candidate: Candidate,
    other: Candidate,
    query_tokens: tuple[str, ...],
    query_tokens_no_diacritics: frozenset[str],
    language: str | None,
) -> bool:
    """Return whether only the candidate's equivalent group has literal evidence.

    Candidate generation emits multiple word-order and slot-decomposition
    variants for one executable intent/target. Confidence is evaluated on that
    semantic group so an exact grammar action on one variant can disambiguate a
    higher-scoring sibling from an opposing action.
    """

    def _equivalent_group(seed: Candidate) -> tuple[Candidate, ...]:
        return tuple(
            item.candidate
            for item in ranked
            if item.candidate.intent_name == seed.intent_name
            and _candidates_share_executable_target(seed, item.candidate)
        )

    candidate_group = _equivalent_group(candidate)
    other_group = _equivalent_group(other)
    if not any(
        _candidate_has_explicit_supported_literal_variant(
            grouped,
            query_tokens,
            query_tokens_no_diacritics,
            language,
        )
        for grouped in candidate_group
    ):
        return False
    if any(
        _candidate_has_explicit_supported_literal_variant(
            grouped,
            query_tokens,
            query_tokens_no_diacritics,
            language,
        )
        for grouped in other_group
    ):
        return False

    candidate_literal_tokens = frozenset(
        token
        for grouped in candidate_group
        for variant in grouped.literal_variants
        for token in variant
    )
    other_literal_tokens = frozenset(
        token
        for grouped in other_group
        for variant in grouped.literal_variants
        for token in variant
    )
    return any(
        token in query_tokens
        or normalize_text_no_diacritics(token, language) in query_tokens_no_diacritics
        for token in candidate_literal_tokens - other_literal_tokens
    )


def _drop_query_unsupported_same_intent_leaders(
    ranked: Sequence[RankedCandidate],
    query_tokens: tuple[str, ...],
    query_tokens_no_diacritics: frozenset[str],
    language: str | None,
) -> tuple[RankedCandidate, ...] | None:
    """Drop a higher-scoring target when a lower variant has explicit support.

    The lexical scorer can occasionally rank a structurally similar target
    first even though only a lower same-intent target is present in the query.
    Removing the unsupported variants lets the complete confidence policy
    independently evaluate the supported candidate against every other intent.
    """
    if len(ranked) < 2:
        return None
    top_candidate = ranked[0].candidate
    top_payload = top_candidate.surface_slot_tokens
    supported_candidate = next(
        (
            item
            for item in ranked[1:]
            if item.candidate.intent_name == top_candidate.intent_name
            and item.candidate.parsed_slots != top_candidate.parsed_slots
            and item.candidate.surface_slot_tokens
            and (
                top_payload
                or _candidate_has_nonliteral_target_token(
                    item.candidate,
                    top_candidate,
                    language,
                )
            )
            and not _surface_payloads_are_nested(top_candidate, item.candidate)
            and _same_intent_payload_query_support(
                top_candidate,
                item.candidate,
                query_tokens,
                query_tokens_no_diacritics,
                language,
            )
            == (False, True)
        ),
        None,
    )
    if supported_candidate is None:
        return None

    filtered: list[RankedCandidate] = []
    supported_payload = supported_candidate.candidate
    for item in ranked:
        candidate = item.candidate
        if candidate.intent_name != supported_payload.intent_name or item is supported_candidate:
            filtered.append(item)
            continue
        supported, candidate_supported = _same_intent_payload_query_support(
            supported_payload,
            candidate,
            query_tokens,
            query_tokens_no_diacritics,
            language,
        )
        if supported and not candidate_supported:
            continue
        filtered.append(item)
    return tuple(filtered)


def _intent_has_supported_literal_variant(
    ranked: Sequence[RankedCandidate],
    candidate: Candidate,
    query_tokens: tuple[str, ...],
    query_tokens_no_diacritics: frozenset[str],
    language: str | None,
) -> bool:
    """Return whether any generated variant supplies complete intent wording."""
    return any(
        item.candidate.intent_name == candidate.intent_name
        and _candidate_has_supported_literal_variant(
            item.candidate,
            query_tokens,
            query_tokens_no_diacritics,
            language,
        )
        for item in ranked
    )


def _candidates_meaningfully_compete(
    candidate: Candidate,
    other: Candidate,
    *,
    query_tokens: tuple[str, ...] | None = None,
    query_tokens_no_diacritics: frozenset[str] = frozenset(),
    language: str | None = None,
) -> bool:
    """Return whether two candidates differ in query-relevant executable payload."""
    candidate_slots = candidate.parsed_slots
    other_slots = other.parsed_slots
    if candidate.normalized_text == other.normalized_text and candidate_slots == other_slots:
        return False
    if _is_equivalent_intent_competitor(candidate.intent_name, other.intent_name):
        return False
    if candidate.intent_name != other.intent_name:
        return True
    if candidate_slots == other_slots:
        return False

    # HassIL commonly represents one physical target through overlapping slot
    # decompositions. Surface values also avoid false differences caused by
    # language-independent output mappings.
    candidate_payload = candidate.surface_slot_tokens
    other_payload = other.surface_slot_tokens
    if candidate_payload and candidate_payload == other_payload:
        return False
    if _surface_payloads_are_nested(candidate, other):
        return False

    if query_tokens is None:
        return True

    candidate_supported, other_supported = _same_intent_payload_query_support(
        candidate,
        other,
        query_tokens,
        query_tokens_no_diacritics,
        language,
    )
    return not (
        candidate_supported and not other_supported and (candidate_payload or other_payload)
    )


def _limit_ranked_items(
    ranked_items: Sequence[_RankedItem],
    limit: int,
) -> list[_RankedItem]:
    """Cap ranked items while retaining the nearest meaningful competitor."""
    selected = list(ranked_items[:limit])
    if len(selected) < 2:
        return selected
    top_candidate = selected[0].candidate
    if any(
        _candidates_meaningfully_compete(top_candidate, item.candidate) for item in selected[1:]
    ):
        return selected
    competitor = next(
        (
            item
            for item in ranked_items[limit:]
            if _candidates_meaningfully_compete(top_candidate, item.candidate)
        ),
        None,
    )
    if competitor is not None:
        selected[-1] = competitor
    return selected


def _limit_ranked_candidates(
    ranked: Sequence[RankedCandidate],
    limit: int,
) -> tuple[RankedCandidate, ...]:
    """Cap public ranked candidates while retaining a confidence competitor."""
    selected = list(ranked[:limit])
    if len(selected) < 2:
        return tuple(selected)
    top_candidate = selected[0].candidate
    if any(
        _candidates_meaningfully_compete(top_candidate, item.candidate) for item in selected[1:]
    ):
        return tuple(selected)
    competitor = next(
        (
            item
            for item in ranked[limit:]
            if _candidates_meaningfully_compete(top_candidate, item.candidate)
        ),
        None,
    )
    if competitor is not None:
        selected[-1] = competitor
    return tuple(selected)


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
            if (slot_name, val.casefold()) in slot_preferences
        ),
        0.0,
    )


type _StructuralTieGroup = tuple[float, float, int, list[_RankedItem]]


def _matching_structural_tie_group(
    item: _RankedItem,
    slot_preference: float,
    final_bucket: int,
    slot_bucket: int,
    tie_groups: Sequence[_StructuralTieGroup],
    group_indices_by_bucket: Mapping[tuple[int, int, int], Sequence[int]],
) -> int | None:
    """Return the nearby tie group matching an item's numeric evidence."""
    nearby_group_indices: list[int] = []
    for final_offset in (-1, 0, 1):
        for slot_offset in (-1, 0, 1):
            nearby_group_indices.extend(
                group_indices_by_bucket.get(
                    (
                        item.slot_specificity,
                        final_bucket + final_offset,
                        slot_bucket + slot_offset,
                    ),
                    (),
                )
            )
    for group_index in sorted(nearby_group_indices):
        group_final, group_slot_preference, group_specificity, group = tie_groups[group_index]
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
            return group_index
    return None


def _group_structural_ties(
    ranked_tuples: Sequence[_RankedItem],
    slot_preferences: set[tuple[str, str]] | None,
    rehydrated_cache: dict[int, tuple[str, dict[str, str]]],
) -> list[_StructuralTieGroup]:
    """Group candidates whose final ranking evidence is equal within tolerance."""
    tie_groups: list[_StructuralTieGroup] = []
    group_indices_by_bucket: defaultdict[tuple[int, int, int], list[int]] = defaultdict(list)
    tolerance = _STRUCTURAL_TIE_ABS_TOLERANCE
    for item in ranked_tuples:
        slot_preference = _ranked_tuple_slot_preference(
            item,
            slot_preferences=slot_preferences,
            rehydrated_cache=rehydrated_cache,
        )
        final_bucket = round(item.final_score / tolerance)
        slot_bucket = round(slot_preference / tolerance)
        matched_group_index = _matching_structural_tie_group(
            item,
            slot_preference,
            final_bucket,
            slot_bucket,
            tie_groups,
            group_indices_by_bucket,
        )
        if matched_group_index is not None:
            continue
        group_index = len(tie_groups)
        tie_groups.append((item.final_score, slot_preference, item.slot_specificity, [item]))
        group_indices_by_bucket[(item.slot_specificity, final_bucket, slot_bucket)].append(
            group_index
        )
    return tie_groups


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
    preferences_by_index: dict[int, tuple[int, float, float]] = {}
    for group_final, group_slot_preference, _, group in _group_structural_ties(
        ranked_tuples,
        slot_preferences,
        rehydrated_cache,
    ):
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


def apply_intent_disambiguation(
    ranked: list[RankedCandidate],
) -> None:
    """Use intent evidence only to resolve an exact final-score tie.

    The intent component is already included in ``final_score``. Promoting a
    lower final score would count that component twice and can replace the best
    lexical candidate with a weaker action. Same-intent variants are skipped.
    """
    if len(ranked) < 2:
        return
    top = ranked[0]
    competitor_index = next(
        (
            index
            for index, candidate in enumerate(ranked[1:], start=1)
            if candidate.candidate.intent_name != top.candidate.intent_name
        ),
        None,
    )
    if competitor_index is None:
        return
    competitor = ranked[competitor_index]
    if not isclose(
        top.scores.final_score,
        competitor.scores.final_score,
        rel_tol=0.0,
        abs_tol=_STRUCTURAL_TIE_ABS_TOLERANCE,
    ):
        return
    if competitor.scores.intent_score > top.scores.intent_score:
        ranked[0], ranked[competitor_index] = competitor, top


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


def _is_same_text_same_slot_competitor(
    candidate: RankedCandidate,
    other: RankedCandidate,
) -> bool:
    """Return whether a same-text competitor preserves the same slot payload."""
    return (
        other.candidate.normalized_text == candidate.candidate.normalized_text
        and other.candidate.parsed_slots == candidate.candidate.parsed_slots
    )


def _is_equivalent_intent_competitor(intent1: str, intent2: str) -> bool:
    """Return whether two intents are functionally equivalent list intents."""
    return any(intent1 in group and intent2 in group for group in _EQUIVALENT_INTENT_GROUPS)


def _is_known_opposing_action_competition(
    candidate: RankedCandidate,
    other: RankedCandidate,
) -> bool:
    """Return whether two candidates represent a classified opposing action pair."""
    return (
        frozenset((candidate.candidate.intent_name, other.candidate.intent_name))
        in _KNOWN_OPPOSING_INTENT_TIE_PREFERENCES
    )


def _is_action_bearing_intent(intent_name: str) -> bool:
    """Conservatively classify Home Assistant intents that can change state."""
    return intent_name.startswith("Hass") and not (
        "Get" in intent_name
        or intent_name
        in {
            "HassNevermind",
            "HassRespond",
            "HassTimerStatus",
        }
    )


def _confidence_decision(
    *,
    min_confidence: float,
    min_margin: float,
    top_candidate: RankedCandidate | None,
    competing_candidate: RankedCandidate | None,
    observed_margin: float | None,
    required_margin: float,
    margin_policy: str,
    relaxation_used: bool = False,
    opposing_action_competition: bool = False,
    action_incompatible_competition: bool = False,
    target_incompatible_competition: bool = False,
    accepted_candidate: RankedCandidate | None = None,
    rejection_reason: FallbackReason | None = None,
) -> ConfidenceGateDecision:
    """Build one confidence decision with consistent configured evidence."""
    return ConfidenceGateDecision(
        configured_min_confidence=min_confidence,
        configured_base_margin=min_margin,
        top_candidate=top_candidate,
        meaningful_competitor=competing_candidate,
        observed_margin=observed_margin,
        required_margin=required_margin,
        margin_policy=margin_policy,
        relaxation_used=relaxation_used,
        opposing_action_competition=opposing_action_competition,
        action_incompatible_competition=action_incompatible_competition,
        target_incompatible_competition=target_incompatible_competition,
        accepted_candidate=accepted_candidate,
        rejection_reason=rejection_reason,
    )


@dataclass(frozen=True, slots=True)
class _GateEvidence:
    """Precomputed competition evidence shared by every gate policy check."""

    min_confidence: float
    min_margin: float
    top: RankedCandidate
    competitor: RankedCandidate
    margin: float
    query_tokens: tuple[str, ...] | None
    query_tokens_no_diacritics: frozenset[str]
    language: str | None
    opposing_actions: bool
    top_is_action: bool
    competitor_is_action: bool
    unknown_action_competition: bool
    action_incompatible_competition: bool
    target_incompatible_competition: bool
    target_has_query_support: bool


def _build_gate_evidence(
    top_candidate: RankedCandidate,
    competing_candidate: RankedCandidate,
    min_confidence: float,
    min_margin: float,
    query_tokens: tuple[str, ...] | None,
    query_tokens_no_diacritics: frozenset[str],
    language: str | None,
) -> _GateEvidence:
    """Derive the shared competition evidence for one gate evaluation."""
    margin = top_candidate.scores.final_score - competing_candidate.scores.final_score
    opposing_actions = _is_known_opposing_action_competition(top_candidate, competing_candidate)
    target_incompatible_competition = (
        top_candidate.candidate.intent_name == competing_candidate.candidate.intent_name
        and top_candidate.candidate.parsed_slots != competing_candidate.candidate.parsed_slots
    )
    target_has_query_support = True
    if target_incompatible_competition and query_tokens is not None:
        target_has_query_support, _competitor_has_query_support = (
            _same_intent_payload_query_support(
                top_candidate.candidate,
                competing_candidate.candidate,
                query_tokens,
                query_tokens_no_diacritics,
                language,
            )
        )
    top_is_action = _is_action_bearing_intent(top_candidate.candidate.intent_name)
    competitor_is_action = _is_action_bearing_intent(competing_candidate.candidate.intent_name)
    unknown_action_competition = (
        not opposing_actions
        and top_candidate.candidate.intent_name != competing_candidate.candidate.intent_name
        and top_is_action
        and competitor_is_action
    )
    action_incompatible_competition = not opposing_actions and top_is_action != competitor_is_action
    return _GateEvidence(
        min_confidence=min_confidence,
        min_margin=min_margin,
        top=top_candidate,
        competitor=competing_candidate,
        margin=margin,
        query_tokens=query_tokens,
        query_tokens_no_diacritics=query_tokens_no_diacritics,
        language=language,
        opposing_actions=opposing_actions,
        top_is_action=top_is_action,
        competitor_is_action=competitor_is_action,
        unknown_action_competition=unknown_action_competition,
        action_incompatible_competition=action_incompatible_competition,
        target_incompatible_competition=target_incompatible_competition,
        target_has_query_support=target_has_query_support,
    )


def _gate_decision(
    evidence: _GateEvidence,
    *,
    margin_policy: str,
    required_margin: float,
    relaxation_used: bool = False,
    accepted_candidate: RankedCandidate | None = None,
    rejection_reason: FallbackReason | None = None,
) -> ConfidenceGateDecision:
    """Build a decision that carries the evidence's competition flags."""
    return _confidence_decision(
        min_confidence=evidence.min_confidence,
        min_margin=evidence.min_margin,
        top_candidate=evidence.top,
        competing_candidate=evidence.competitor,
        observed_margin=evidence.margin,
        required_margin=required_margin,
        margin_policy=margin_policy,
        relaxation_used=relaxation_used,
        opposing_action_competition=evidence.opposing_actions,
        action_incompatible_competition=evidence.action_incompatible_competition,
        target_incompatible_competition=evidence.target_incompatible_competition,
        accepted_candidate=accepted_candidate,
        rejection_reason=rejection_reason,
    )


def _gate_wildcard_absorption(
    ranked: Sequence[RankedCandidate],
    evidence: _GateEvidence,
) -> ConfidenceGateDecision | None:
    """Reject a top wildcard that absorbed a query-supported concrete target.

    Scans the whole ranked list, not just the first meaningful competitor:
    the concrete-target candidate that the wildcard absorbed may rank below
    another wildcard and would otherwise never be consulted.
    """
    if any(
        _wildcard_absorbs_supported_concrete_target(
            evidence.top.candidate,
            item.candidate,
            evidence.query_tokens,
            evidence.query_tokens_no_diacritics,
            evidence.language,
        )
        for item in ranked[1:]
    ):
        return _gate_decision(
            evidence,
            margin_policy="wildcard_absorbed_concrete_target",
            required_margin=evidence.min_margin,
            rejection_reason=FallbackReason.LOW_CONFIDENCE,
        )
    return None


def _gate_exact_lexical(
    ranked: Sequence[RankedCandidate],
    evidence: _GateEvidence,
) -> ConfidenceGateDecision | None:
    """Accept an exact lexical match without requiring a margin.

    Exact canonical text is delegated back to HassIL with live request
    context; candidate intent/slot metadata is not executed here.
    """
    if _is_exact_lexical_match(evidence.top):
        return _gate_decision(
            evidence,
            margin_policy="exact_lexical",
            required_margin=0.0,
            relaxation_used=True,
            accepted_candidate=evidence.top,
        )
    return None


def _gate_read_only_incomplete_literal(
    ranked: Sequence[RankedCandidate],
    evidence: _GateEvidence,
) -> ConfidenceGateDecision | None:
    """Reject a read-only intent whose literal wording lacks query support."""
    if (
        evidence.query_tokens is not None
        and not evidence.top_is_action
        and not _intent_has_supported_literal_variant(
            ranked,
            evidence.top.candidate,
            evidence.query_tokens,
            evidence.query_tokens_no_diacritics,
            evidence.language,
        )
        and not _candidate_covers_query_tokens(
            evidence.top.candidate,
            evidence.query_tokens,
            evidence.language,
        )
    ):
        return _gate_decision(
            evidence,
            margin_policy="read_only_incomplete_literal",
            required_margin=evidence.min_margin,
            rejection_reason=FallbackReason.LOW_CONFIDENCE,
        )
    return None


def _gate_state_change_incomplete_literal(
    ranked: Sequence[RankedCandidate],
    evidence: _GateEvidence,
) -> ConfidenceGateDecision | None:
    """Reject a state-changing intent competing across action kinds without literal support."""
    if (
        evidence.query_tokens is not None
        and evidence.top_is_action
        and evidence.action_incompatible_competition
        and not _candidate_has_supported_literal_variant(
            evidence.top.candidate,
            evidence.query_tokens,
            evidence.query_tokens_no_diacritics,
            evidence.language,
        )
        and not _candidate_covers_query_tokens(
            evidence.top.candidate,
            evidence.query_tokens,
            evidence.language,
        )
    ):
        return _gate_decision(
            evidence,
            margin_policy="state_change_incomplete_literal",
            required_margin=evidence.min_margin,
            rejection_reason=FallbackReason.LOW_CONFIDENCE,
        )
    return None


def _gate_supported_literal_action_evidence(
    ranked: Sequence[RankedCandidate],
    evidence: _GateEvidence,
) -> ConfidenceGateDecision | None:
    """Accept when only the top action's intent group has literal query support."""
    if (
        evidence.query_tokens is not None
        and evidence.top_is_action
        and evidence.competitor_is_action
        and _intent_group_has_unique_supported_literal(
            ranked,
            evidence.top.candidate,
            evidence.competitor.candidate,
            evidence.query_tokens,
            evidence.query_tokens_no_diacritics,
            evidence.language,
        )
    ):
        return _gate_decision(
            evidence,
            margin_policy="supported_literal_action_evidence",
            required_margin=0.0,
            relaxation_used=True,
            accepted_candidate=evidence.top,
        )
    return None


def _gate_safe_intent_relaxation(
    ranked: Sequence[RankedCandidate],
    evidence: _GateEvidence,
) -> ConfidenceGateDecision | None:
    """Accept compatible targets whose intent evidence safely relaxes the margin."""
    if evidence.target_incompatible_competition or not _has_safe_relaxed_intent_evidence(
        evidence.top,
        evidence.competitor,
        evidence.margin,
        evidence.min_margin,
    ):
        return None
    empty_slot_relaxation = not evidence.top.candidate.parsed_slots and not (
        evidence.competitor.candidate.parsed_slots
    )
    return _gate_decision(
        evidence,
        margin_policy=(
            "safe_empty_slot_relaxation"
            if empty_slot_relaxation
            else "safe_intent_evidence_relaxation"
        ),
        required_margin=(SAFE_EMPTY_SLOT_RELAXED_MIN_MARGIN if empty_slot_relaxation else 0.0),
        relaxation_used=True,
        accepted_candidate=evidence.top,
    )


def _gate_high_confidence_relaxation(
    ranked: Sequence[RankedCandidate],
    evidence: _GateEvidence,
) -> ConfidenceGateDecision | None:
    """Accept a high-confidence compatible top candidate at a relaxed margin."""
    if evidence.target_incompatible_competition or not _passes_high_confidence_relaxed_margin(
        evidence.top,
        evidence.margin,
        evidence.min_margin,
    ):
        return None
    return _gate_decision(
        evidence,
        margin_policy="high_confidence_relaxation",
        required_margin=HIGH_CONFIDENCE_RELAXED_MIN_MARGIN,
        relaxation_used=True,
        accepted_candidate=evidence.top,
    )


def _gate_weak_zero_intent_evidence(
    ranked: Sequence[RankedCandidate],
    evidence: _GateEvidence,
) -> ConfidenceGateDecision | None:
    """Reject a fuzzy top match with no action evidence and a close rival."""
    if _has_weak_zero_intent_evidence(evidence.top, evidence.margin):
        return _gate_decision(
            evidence,
            margin_policy="weak_zero_intent_evidence",
            required_margin=evidence.min_margin,
            rejection_reason=FallbackReason.LOW_CONFIDENCE,
        )
    return None


def _gate_action_incompatible_weaker_evidence(
    ranked: Sequence[RankedCandidate],
    evidence: _GateEvidence,
) -> ConfidenceGateDecision | None:
    """Reject a read-only leader with weaker intent evidence than an action rival."""
    if (
        evidence.action_incompatible_competition
        and not evidence.top_is_action
        and evidence.competitor_is_action
        and evidence.top.scores.intent_score < evidence.competitor.scores.intent_score
    ):
        return _gate_decision(
            evidence,
            margin_policy="action_incompatible_weaker_intent_evidence",
            required_margin=evidence.min_margin,
            rejection_reason=FallbackReason.LOW_CONFIDENCE,
        )
    return None


def _gate_target_without_query_support(
    ranked: Sequence[RankedCandidate],
    evidence: _GateEvidence,
) -> ConfidenceGateDecision | None:
    """Reject a same-intent leader whose target payload lacks query support."""
    if (
        evidence.target_incompatible_competition
        and not evidence.target_has_query_support
        and evidence.top.candidate.surface_slot_tokens
        and evidence.competitor.candidate.surface_slot_tokens
    ):
        return _gate_decision(
            evidence,
            margin_policy="target_without_query_support",
            required_margin=evidence.min_margin,
            rejection_reason=FallbackReason.LOW_CONFIDENCE,
        )
    return None


def _gate_competitive_base_margin(
    ranked: Sequence[RankedCandidate],
    evidence: _GateEvidence,
) -> ConfidenceGateDecision | None:
    """Apply the base margin when the competition is structurally risky."""
    if not (
        evidence.target_incompatible_competition
        or evidence.opposing_actions
        or evidence.unknown_action_competition
        or evidence.action_incompatible_competition
    ):
        return None
    if evidence.target_incompatible_competition:
        margin_policy = "target_base_margin"
    elif evidence.opposing_actions:
        margin_policy = "opposing_action_base_margin"
    elif evidence.action_incompatible_competition:
        margin_policy = "action_incompatible_base_margin"
    else:
        margin_policy = "unknown_action_base_margin"
    accepted = evidence.margin >= evidence.min_margin
    return _gate_decision(
        evidence,
        margin_policy=margin_policy,
        required_margin=evidence.min_margin,
        accepted_candidate=evidence.top if accepted else None,
        rejection_reason=None if accepted else FallbackReason.LOW_MARGIN,
    )


_GATE_POLICY_CHECKS = (
    _gate_wildcard_absorption,
    _gate_exact_lexical,
    _gate_read_only_incomplete_literal,
    _gate_state_change_incomplete_literal,
    _gate_supported_literal_action_evidence,
    _gate_safe_intent_relaxation,
    _gate_high_confidence_relaxation,
    _gate_weak_zero_intent_evidence,
    _gate_action_incompatible_weaker_evidence,
    _gate_target_without_query_support,
    _gate_competitive_base_margin,
)


def _confidence_query_tokens(
    query: str | None,
    language: str | None,
) -> tuple[tuple[str, ...] | None, frozenset[str]]:
    """Return normalized query tokens used by confidence policies."""
    if query is None:
        return None, frozenset()
    return (
        tuple(normalize_text(query).split()),
        frozenset(normalize_text_no_diacritics(query, language).split()),
    )


def _meaningful_confidence_competitor(
    ranked: Sequence[RankedCandidate],
    top_candidate: RankedCandidate,
    query_tokens: tuple[str, ...] | None,
    query_tokens_no_diacritics: frozenset[str],
    language: str | None,
) -> RankedCandidate | None:
    """Return the first candidate that meaningfully competes with the leader."""
    return next(
        (
            item
            for item in ranked[1:]
            if _candidates_meaningfully_compete(
                top_candidate.candidate,
                item.candidate,
                query_tokens=query_tokens,
                query_tokens_no_diacritics=query_tokens_no_diacritics,
                language=language,
            )
        ),
        None,
    )


def _evaluate_gate_policies(
    ranked: Sequence[RankedCandidate],
    evidence: _GateEvidence,
) -> ConfidenceGateDecision:
    """Return the first policy decision or apply the default base margin."""
    for gate_check in _GATE_POLICY_CHECKS:
        decision = gate_check(ranked, evidence)
        if decision is not None:
            return decision
    accepted = evidence.margin >= evidence.min_margin
    return _gate_decision(
        evidence,
        margin_policy="base_margin",
        required_margin=evidence.min_margin,
        accepted_candidate=evidence.top if accepted else None,
        rejection_reason=None if accepted else FallbackReason.LOW_MARGIN,
    )


def evaluate_confidence_gates(
    ranked: Sequence[RankedCandidate],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    *,
    query: str | None = None,
    language: str | None = None,
) -> ConfidenceGateDecision:
    """Evaluate the shared dynamic confidence policy and return all evidence.

    The policy checks in ``_GATE_POLICY_CHECKS`` run in order; the first one
    with an opinion decides. Their order is load-bearing.
    """
    if not ranked:
        return _confidence_decision(
            min_confidence=min_confidence,
            min_margin=min_margin,
            top_candidate=None,
            competing_candidate=None,
            observed_margin=None,
            required_margin=0.0,
            margin_policy="no_candidate",
            rejection_reason=FallbackReason.NO_CANDIDATE,
        )
    query_tokens, query_tokens_no_diacritics = _confidence_query_tokens(query, language)
    if query_tokens is not None and (
        filtered_ranked := _drop_query_unsupported_same_intent_leaders(
            ranked,
            query_tokens,
            query_tokens_no_diacritics,
            language,
        )
    ):
        return evaluate_confidence_gates(
            filtered_ranked,
            min_confidence=min_confidence,
            min_margin=min_margin,
            query=query,
            language=language,
        )
    top_candidate = ranked[0]
    if top_candidate.scores.final_score < min_confidence:
        return _confidence_decision(
            min_confidence=min_confidence,
            min_margin=min_margin,
            top_candidate=top_candidate,
            competing_candidate=None,
            observed_margin=None,
            required_margin=0.0,
            margin_policy="hard_min_confidence",
            rejection_reason=FallbackReason.LOW_CONFIDENCE,
        )
    competing_candidate = _meaningful_confidence_competitor(
        ranked,
        top_candidate,
        query_tokens,
        query_tokens_no_diacritics,
        language,
    )
    if competing_candidate is None:
        return _confidence_decision(
            min_confidence=min_confidence,
            min_margin=min_margin,
            top_candidate=top_candidate,
            competing_candidate=None,
            observed_margin=None,
            required_margin=0.0,
            margin_policy="no_competitor",
            accepted_candidate=top_candidate,
        )
    evidence = _build_gate_evidence(
        top_candidate,
        competing_candidate,
        min_confidence,
        min_margin,
        query_tokens,
        query_tokens_no_diacritics,
        language,
    )
    return _evaluate_gate_policies(ranked, evidence)


def accepted_candidate(
    ranked: Sequence[RankedCandidate],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    *,
    query: str | None = None,
    language: str | None = None,
) -> RankedCandidate | None:
    """Return the accepted top candidate or None when confidence gates reject it."""
    return evaluate_confidence_gates(
        ranked,
        min_confidence,
        min_margin,
        query=query,
        language=language,
    ).accepted_candidate


def confidence_gate_rejection_reason(
    ranked: Sequence[RankedCandidate],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    *,
    query: str | None = None,
    language: str | None = None,
) -> FallbackReason:
    """Return the fallback reason for candidates rejected by confidence gates."""
    decision = evaluate_confidence_gates(
        ranked,
        min_confidence,
        min_margin,
        query=query,
        language=language,
    )
    return decision.rejection_reason or FallbackReason.LOW_CONFIDENCE


def _is_exact_lexical_match(ranked_candidate: RankedCandidate) -> bool:
    """Return whether a ranked candidate exactly matches query text lexically."""
    scores = ranked_candidate.scores
    return scores.rapidfuzz_score == 1.0 and scores.char_ngram_score == 1.0


def _hotword_similarity(text_a: str, text_b: str) -> float:
    """Calculate normalized similarity between two token-normalized strings."""
    return max(
        _raw_cached_fuzz_ratio(text_a, text_b) / 100.0,
        rapidfuzz_similarity_normalized(text_a, text_b),
    )


def match_hotword_prefix(
    query: str,
    hotwords: str | Sequence[str],
    min_confidence: float = DEFAULT_HOTWORD_MIN_CONFIDENCE,
    language: str | None = None,
) -> tuple[bool, float, str | None]:
    """Return whether the query prefix matches any configured hotword with high confidence.

    Splits query and hotword into normalized tokens, slices the query prefix by the
    token length of each target hotword, and computes similarity scores using RapidFuzz
    and token-level lexical comparison. Returns (matched, score, matched_hotword).
    """
    if not query or not hotwords:
        return False, 0.0, None

    candidate_hotwords = normalize_hotword_list(hotwords)
    if not candidate_hotwords:
        return False, 0.0, None

    norm_query = normalize_text(query)
    query_tokens = tokenize_normalized(norm_query)
    if not query_tokens:
        return False, 0.0, None

    norm_query_no_dia = normalize_text_no_diacritics_from_normalized(norm_query, language)
    query_tokens_no_dia = tokenize_normalized(norm_query_no_dia)

    best_score = 0.0
    best_hw: str | None = None

    for hw in candidate_hotwords:
        norm_hw = normalize_text(hw)
        hw_tokens = tokenize_normalized(norm_hw)
        if not hw_tokens:
            continue

        token_len = len(hw_tokens)
        if len(query_tokens) < token_len:
            continue

        prefix_text = " ".join(query_tokens[:token_len])
        score = _hotword_similarity(prefix_text, norm_hw)

        if len(query_tokens_no_dia) >= token_len:
            norm_hw_no_dia = normalize_text_no_diacritics_from_normalized(norm_hw, language)
            prefix_no_dia = " ".join(query_tokens_no_dia[:token_len])
            if prefix_no_dia != prefix_text or norm_hw_no_dia != norm_hw:
                score = max(score, _hotword_similarity(prefix_no_dia, norm_hw_no_dia))

        if score > best_score:
            best_score = score
            best_hw = hw

        if best_score >= 1.0:
            break

    if best_score >= min_confidence and best_hw is not None:
        return True, best_score, best_hw

    return False, best_score, None


def clear_ranking_caches() -> None:
    """Clear all global LRU caches in ranking module."""
    _raw_cached_fuzz_ratio.cache_clear()
    _chars_conflict_in_marks.cache_clear()
    _build_positional_lookup.cache_clear()
    _short_nonliteral_query_tokens.cache_clear()
    _slot_names_from_csv.cache_clear()
