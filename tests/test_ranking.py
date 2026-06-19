"""Tests for lexical ranking and candidate indexing."""

import pytest

from custom_components.assist_canonicalizer import ranking
from custom_components.assist_canonicalizer.candidate import Candidate, CandidateSource
from custom_components.assist_canonicalizer.const import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES,
)
from custom_components.assist_canonicalizer.indexer import build_index
from custom_components.assist_canonicalizer.normalization import normalize_text
from custom_components.assist_canonicalizer.ranking import (
    CharNGramIndex,
    RankedCandidate,
    ScoreBreakdown,
    _query_token_coverage,
    accepted_candidate,
    rank_candidates,
    rapidfuzz_similarity_normalized,
    token_count_ratio,
)


def test_index_deduplicates_by_normalized_text_with_source_priority() -> None:
    """Prefer custom candidates when normalized candidate text duplicates."""
    generated = Candidate(
        text="Turn on kitchen light",
        intent_name="HassTurnOn",
        source=CandidateSource.GENERATED_SAMPLE,
        language="en",
    )
    custom = Candidate(
        text="turn on kitchen light",
        intent_name="HassTurnOn",
        source=CandidateSource.CUSTOM_SENTENCE,
        language="en",
    )
    index = build_index("en", [generated, custom])
    assert index.candidate_count == 1
    assert index.candidates[0].source == CandidateSource.CUSTOM_SENTENCE


def test_rank_candidates_prefers_matching_entity_terms() -> None:
    """Rank the closest lexical candidate first."""
    index = build_index(
        "en",
        [
            Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
            Candidate(text="turn off bedroom light", intent_name="HassTurnOff", language="en"),
        ],
    )
    ranked = index.rank("turn on kitchen lamp")
    assert ranked[0].candidate.intent_name == "HassTurnOn"


def test_accepted_candidate_enforces_margin_for_competing_intents() -> None:
    """Reject ambiguous fuzzy top candidates with an insufficient intent margin."""
    index = build_index(
        "en",
        [
            Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
            Candidate(text="turn kitchen lamp", intent_name="HassLightSet", language="en"),
        ],
    )
    ranked = index.rank("turn kitchen")

    assert (
        accepted_candidate(
            ranked,
            min_confidence=DEFAULT_MIN_CONFIDENCE / 5.0,
            min_margin=DEFAULT_MIN_MARGIN + 0.95,
        )
        is None
    )


def test_accepted_candidate_allows_exact_top_against_fuzzy_competing_intent() -> None:
    """Accept exact top matches when the competing intent is only a fuzzy match."""
    index = build_index(
        "en",
        [
            Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
            Candidate(text="turn on kitchen light 0%", intent_name="HassLightSet", language="en"),
        ],
    )
    ranked = index.rank("turn on kitchen light")

    assert (
        accepted_candidate(
            ranked,
            min_confidence=DEFAULT_MIN_CONFIDENCE / 5.0,
            min_margin=DEFAULT_MIN_MARGIN + 0.95,
        )
        is ranked[0]
    )


def test_accepted_candidate_allows_close_candidates_with_same_intent() -> None:
    """Accept close candidates when they preserve the same Home Assistant intent."""
    index = build_index(
        "en",
        [
            Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
            Candidate(text="turn on kitchen lamp", intent_name="HassTurnOn", language="en"),
        ],
    )
    ranked = index.rank("turn on kitchen")
    assert (
        accepted_candidate(
            ranked,
            min_confidence=DEFAULT_MIN_CONFIDENCE / 5.0,
            min_margin=DEFAULT_MIN_MARGIN + 0.95,
        )
        is not None
    )


def test_accepted_candidate_allows_identical_text_with_different_intent() -> None:
    """Accept candidate when competing intent has identical text."""
    score = DEFAULT_MIN_CONFIDENCE + 0.1
    top = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=score,
            char_ngram_score=score,
            bm25_score=score,
            intent_score=score,
            final_score=score,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOff", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=score,
            char_ngram_score=score,
            bm25_score=score,
            intent_score=score,
            final_score=score,
        ),
    )

    ranked = (top, competitor)
    assert (
        accepted_candidate(
            ranked,
            min_confidence=DEFAULT_MIN_CONFIDENCE,
            min_margin=DEFAULT_MIN_MARGIN,
        )
        is top
    )


def test_index_rank_reuses_prebuilt_lexical_index(monkeypatch) -> None:
    """Reuse prebuilt BM25 data when ranking an existing index."""
    index = build_index(
        "en",
        [
            Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
            Candidate(text="turn off bedroom light", intent_name="HassTurnOff", language="en"),
        ],
    )

    def fail_from_texts(*args, **kwargs):
        """Fail if ranking rebuilds BM25 data."""
        raise AssertionError("BM25 index should be cached by CanonicalIndex")

    monkeypatch.setattr(ranking.BM25Index, "from_texts", fail_from_texts)

    ranked = index.rank("turn on kitchen lamp")

    assert ranked[0].candidate.intent_name == "HassTurnOn"


def test_rank_candidates_prefilters_rapidfuzz_work(monkeypatch) -> None:
    """Limit expensive RapidFuzz scoring to the configured prefilter size."""
    candidates = [
        Candidate(text=f"device {index}", intent_name="HassTurnOn", language="en")
        for index in range(DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES + 50)
    ]
    calls = 0

    def fake_rapidfuzz_similarity(query: str, candidate: str, **kwargs: object) -> float:
        """Count expensive RapidFuzz calls."""
        nonlocal calls
        calls += 1
        return 0.5

    monkeypatch.setattr(
        ranking,
        "rapidfuzz_similarity_normalized",
        fake_rapidfuzz_similarity,
    )

    ranking.rank_candidates("device 1", candidates, max_candidates=5)

    assert calls == DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES


def test_rapidfuzz_similarity_penalizes_extra_area_tokens() -> None:
    """Prefer exact command text over candidates with unrelated extra area tokens."""
    exact = rapidfuzz_similarity_normalized(
        "tắt đèn phòng ngủ to",
        "tắt đèn phòng ngủ to",
    )
    with_extra_area = rapidfuzz_similarity_normalized(
        "tắt đèn phòng ngủ to",
        "kitchen đèn phòng ngủ to tắt",
    )

    assert exact == 1.0
    assert with_extra_area < 0.9


def test_rank_candidates_penalizes_opposite_builtin_action() -> None:
    """Prefer a matching built-in action over an opposite action entity match."""
    index = build_index(
        "vi",
        [
            Candidate(
                text="tắt Quạt thông gió phòng tắm to",
                intent_name="HassTurnOff",
                language="vi",
                metadata={"literal_text": "tắt"},
            ),
            Candidate(
                text="bật Quạt thông gió phòng tắm to",
                intent_name="HassTurnOn",
                language="vi",
                metadata={"literal_text": "bật"},
            ),
        ],
    )

    ranked = index.rank("bật quạt phòng tắm to")

    assert ranked[0].candidate.intent_name == "HassTurnOn"
    assert ranked[0].scores.intent_score == 1.0
    assert ranked[1].scores.intent_score == 0.0


def test_rank_candidates_uses_english_builtin_action_alignment() -> None:
    """Prefer English built-in action words over entity-only overlap."""
    index = build_index(
        "en",
        [
            Candidate(
                text="turn off bathroom fan",
                intent_name="HassTurnOff",
                language="en",
                metadata={"literal_text": "turn off"},
            ),
            Candidate(
                text="turn on bathroom fan",
                intent_name="HassTurnOn",
                language="en",
                metadata={"literal_text": "turn on"},
            ),
        ],
    )

    ranked = index.rank("turn bathroom fan on")

    assert ranked[0].candidate.intent_name == "HassTurnOn"


def test_exact_intent_score_supports_alternatives() -> None:
    """Verify that exact intent action scoring handles multiple options separated by pipe."""
    query_tokens_bat = frozenset(normalize_text("bật quạt").split())
    query_tokens_mo = frozenset(normalize_text("mở quạt").split())
    query_tokens_len = frozenset(normalize_text("bật lên quạt").split())
    query_tokens_tat = frozenset(normalize_text("tắt quạt").split())
    assert ranking._exact_intent_score("bật|mở|bật lên", query_tokens_bat) == 1.0
    assert ranking._exact_intent_score("bật|mở|bật lên", query_tokens_mo) == 1.0
    assert ranking._exact_intent_score("bật|mở|bật lên", query_tokens_len) == 1.0
    assert ranking._exact_intent_score("bật|mở|bật lên", query_tokens_tat) == 0.0


def test_build_index_validation_errors() -> None:
    """Verify build_index raises ValueError on invalid language or max_total_candidates."""
    with pytest.raises(ValueError, match="Language must not be empty"):
        build_index(" ", [])
    with pytest.raises(ValueError, match="max_total_candidates must be positive"):
        build_index("en", [], max_total_candidates=0)


def test_candidate_validation_errors() -> None:
    """Verify Candidate raises ValueError on empty text or empty intent name."""
    with pytest.raises(ValueError, match="Candidate text must not be empty"):
        Candidate(text=" ", intent_name="HassTurnOn")
    with pytest.raises(ValueError, match="Candidate intent name must not be empty"):
        Candidate(text="turn on", intent_name=" ")


def test_query_token_coverage_perfect_overlap() -> None:
    """Return 1.0 when all query tokens appear in candidate tokens."""
    result = _query_token_coverage(
        frozenset({"đèn", "phòng", "khách"}),
        frozenset({"đèn", "phòng", "khách"}),
    )
    assert result == 1.0


def test_query_token_coverage_partial_overlap() -> None:
    """Return fraction < 1.0 when some query tokens are missing."""
    result = _query_token_coverage(
        frozenset({"tát", "đèn", "phòng", "khách", "nhé"}),
        frozenset({"đèn", "phòng", "khách"}),
    )
    assert result == pytest.approx(0.36)


def test_query_token_coverage_empty_query() -> None:
    """Return 1.0 when query tokens are empty."""
    result = _query_token_coverage(
        frozenset(),
        frozenset({"đèn", "phòng", "khách"}),
    )
    assert result == 1.0


def test_query_token_coverage_empty_candidate() -> None:
    """Return 0.0 when candidate has no tokens to cover query."""
    result = _query_token_coverage(
        frozenset({"tắt", "đèn"}),
        frozenset(),
    )
    assert result == 0.0


def test_intent_action_score_fallback_uses_1_0_when_no_literal_text() -> None:
    """Return 1.0 as fallback when candidate has no literal_text metadata."""
    query_tokens = frozenset(normalize_text("tắt đèn phòng khách").split())
    score = ranking._exact_intent_score("", query_tokens)
    assert score == 1.0


def test_rank_candidates_typo_action_demotes_entity_only_candidates() -> None:
    """Prefer correct action candidate over entity-only when query has typo'd action.

    Regression: when query is "tát đèn phòng khách nhé" (typo of "tắt"),
    the correct "tắt đèn phòng khách" (HassTurnOff) should outrank the
    entity-only "đèn phòng khách" (HassTurnOn) which previously got
    a free intent_score=1.0 even though its ``literal_text`` of ``"đèn"``
    covers only one query token.
    """
    index = build_index(
        "vi",
        [
            Candidate(
                text="tắt đèn phòng khách",
                intent_name="HassTurnOff",
                language="vi",
                metadata={"literal_text": "tắt"},
            ),
            Candidate(
                text="đèn phòng khách",
                intent_name="HassTurnOn",
                language="vi",
                metadata={"literal_text": "đèn"},
            ),
            Candidate(
                text="bật đèn phòng khách",
                intent_name="HassTurnOn",
                language="vi",
                metadata={"literal_text": "bật"},
            ),
        ],
    )
    ranked = index.rank("tát đèn phòng khách nhé")
    assert ranked[0].candidate.intent_name == "HassTurnOff"
    assert ranked[0].candidate.text == "tắt đèn phòng khách"


def test_rank_candidates_exact_normalized_short_circuit() -> None:
    """Verify exact normalized matches short-circuit fuzzy ranking and return score 1.0."""
    index = build_index(
        "vi",
        [
            Candidate(text="bật đèn phòng khách", intent_name="HassTurnOn", language="vi"),
            Candidate(text="tắt quạt phòng tắm", intent_name="HassTurnOff", language="vi"),
        ],
    )
    # Match exact
    ranked = index.rank("Bật, đèn phòng khách!")
    assert len(ranked) == 1
    assert ranked[0].candidate.text == "bật đèn phòng khách"
    assert ranked[0].scores.final_score == 1.0


def test_rank_candidates_exact_no_diacritics_short_circuit() -> None:
    """Verify exact no-diacritics matches short-circuit ranking if no collisions exist."""
    index = build_index(
        "vi",
        [
            Candidate(text="bật đèn phòng khách", intent_name="HassTurnOn", language="vi"),
            Candidate(text="tắt quạt phòng tắm", intent_name="HassTurnOff", language="vi"),
        ],
    )
    # No diacritics
    ranked = index.rank("bat den phong khach")
    assert len(ranked) == 1
    assert ranked[0].candidate.text == "bật đèn phòng khách"
    assert ranked[0].scores.final_score == 1.0


def test_rank_candidates_no_diacritics_collision_falls_back_to_fuzzy() -> None:
    """Verify collision under no-diacritics normalization runs full fuzzy ranking."""
    # "bật cửa" (HassTurnOn) vs "bạt cửa" (HassTurnOff)
    # both normalize without diacritics to "bat cua"
    index = build_index(
        "vi",
        [
            Candidate(text="bật cửa", intent_name="HassTurnOn", language="vi"),
            Candidate(text="bạt cửa", intent_name="HassTurnOff", language="vi"),
        ],
    )
    ranked = index.rank("bat cua")
    # Should not early return because of different intents.
    # It will go through BM25 / char ngrams / rapidfuzz, and result in both candidates being ranked
    assert len(ranked) == 2
    assert all(r.scores.final_score < 1.0 for r in ranked)


def test_ranking_helpers_and_validation_errors() -> None:
    """Test ranking utility helpers and guard validations."""
    # test rapidfuzz_similarity_normalized
    assert (
        rapidfuzz_similarity_normalized(normalize_text("bật đèn"), normalize_text("bật đèn")) == 1.0
    )

    # test rapidfuzz_similarity_normalized with empty strings
    assert rapidfuzz_similarity_normalized("", "bật đèn") == 0.0
    assert rapidfuzz_similarity_normalized("bật đèn", "") == 0.0

    # test token_count_ratio with empty strings
    assert token_count_ratio("", "bật") == 0.0
    assert token_count_ratio("bật", "") == 0.0

    # test CharNGramIndex with empty grams
    char_index = CharNGramIndex.from_grams(())
    assert char_index.score(frozenset({"abc"})) == []

    # test CharNGramIndex.score with empty query grams
    char_index_2 = CharNGramIndex.from_grams((frozenset({"abc"}),))
    assert char_index_2.score(frozenset()) == [0.0]

    # test rank_candidates guard validations
    candidates = [Candidate(text="bật đèn", intent_name="HassTurnOn", language="vi")]
    with pytest.raises(ValueError, match="max_candidates must be positive"):
        rank_candidates("bật đèn", candidates, max_candidates=0)

    with pytest.raises(
        ValueError, match="rapidfuzz_prefilter_candidates must be at least max_candidates"
    ):
        rank_candidates("bật đèn", candidates, max_candidates=5, rapidfuzz_prefilter_candidates=2)

    with pytest.raises(ValueError, match="candidate_char_index length must match candidates"):
        rank_candidates("bật đèn", candidates, candidate_char_index=CharNGramIndex.from_grams(()))

    # test rank_candidates empty candidates
    assert rank_candidates("bật đèn", []) == ()


def test_apply_intent_disambiguation_promotes_higher_intent_score_within_margin() -> None:
    """Top-2 have different intents, close final_score, second has higher intent_score."""
    margin = ranking.TIEBREAKER_INTENT_MARGIN

    first = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=0.5,
            final_score=1.0,
        ),
    )
    second = RankedCandidate(
        candidate=Candidate(
            text="turn off bedroom light", intent_name="HassTurnOff", language="en"
        ),
        # final_score is slightly lower than the first but still within the tiebreaker margin
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=0.9,
            final_score=1.0 - margin / 2.0,
        ),
    )

    ranked = [first, second]

    ranking._apply_intent_disambiguation(ranked)

    # The higher intent_score candidate with a different intent should be promoted to rank 0
    assert ranked[0] is second
    assert ranked[1] is first


def test_apply_intent_disambiguation_does_not_reorder_when_gap_exceeds_margin() -> None:
    """Top-2 gap exceeds margin: no reordering even if second has higher intent_score."""
    margin = ranking.TIEBREAKER_INTENT_MARGIN

    first = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=0.5,
            final_score=1.0,
        ),
    )
    second = RankedCandidate(
        candidate=Candidate(
            text="turn off bedroom light", intent_name="HassTurnOff", language="en"
        ),
        # final_score gap exceeds the tiebreaker margin
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=0.9,
            final_score=1.0 - (margin * 2.0),
        ),
    )

    ranked = [first, second]

    ranking._apply_intent_disambiguation(ranked)

    # Because the final_score gap exceeds the margin, the ordering should not change
    assert ranked[0] is first
    assert ranked[1] is second


def test_apply_intent_disambiguation_keeps_order_for_same_intent() -> None:
    """Top-2 share the same intent: order is unchanged even within the margin."""
    margin = ranking.TIEBREAKER_INTENT_MARGIN

    first = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=0.5,
            final_score=1.0,
        ),
    )
    second = RankedCandidate(
        candidate=Candidate(
            text="turn on kitchen lamp", intent_name="HassTurnOn", language="en"
        ),  # same intent as the first candidate
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=0.9,
            final_score=1.0 - margin / 2.0,
        ),
    )

    ranked = [first, second]

    ranking._apply_intent_disambiguation(ranked)

    # Because the intents are identical, the original ordering should be preserved
    assert ranked[0] is first
    assert ranked[1] is second


def test_rank_candidates_rehydrates_wildcard() -> None:
    """Test that rank_candidates correctly rehydrates wildcard placeholder candidates."""
    # We create a candidate with a wildcard slot
    candidate = Candidate(
        text="broadcast message",
        intent_name="HassBroadcast",
        language="en",
        metadata={"literal_text": "broadcast message"},
    )
    assert candidate.has_wildcard

    # We query with a real value
    query = "broadcast dinner is ready"

    # Call rank_candidates. This will trigger wildcard rehydration.
    ranked = rank_candidates(
        query=query,
        candidates=[candidate],
        max_candidates=1,
        language="en",
    )

    # Verify candidate was ranked and has a high score due to rehydration
    assert len(ranked) == 1
    assert ranked[0].candidate == candidate
    assert ranked[0].scores.final_score > 0.0
    assert ranked[0].scores.penalty > 0.0
    assert ranked[0].scores.rapidfuzz_score >= 0.99


def test_rank_candidates_wildcard_bypasses_prefilter() -> None:
    """Test that wildcard candidates bypass the pre-filter and are evaluated."""
    # We create multiple candidates to exceed the pre-filter limit.
    # If we set rapidfuzz_prefilter_candidates=1 and max_candidates=1, then prefilter_limit is 2.
    # We have 3 candidates, so the lowest-scoring one in the pre-filter would be discarded.
    wildcard_cand = Candidate(
        text="broadcast message",
        intent_name="HassBroadcast",
        language="en",
        metadata={"literal_text": "broadcast message"},
    )
    cand_1 = Candidate(
        text="turn on light",
        intent_name="HassTurnOn",
        language="en",
        metadata={"literal_text": "turn on"},
    )
    cand_2 = Candidate(
        text="turn off light",
        intent_name="HassTurnOff",
        language="en",
        metadata={"literal_text": "turn off"},
    )

    # First, verify that if wildcard prefilter structures are empty,
    # wildcard_cand is discarded because it has a lower prefilter score.
    ranked_disabled = rank_candidates(
        query="turn on broadcast dinner is ready",
        candidates=[cand_1, cand_2, wildcard_cand],
        max_candidates=1,
        rapidfuzz_prefilter_candidates=1,
        language="en",
        wildcard_always_passes=frozenset(),
        wildcard_variants_with_len={},
    )
    assert len(ranked_disabled) == 1
    assert ranked_disabled[0].candidate != wildcard_cand

    # Now, test with default parameters (wildcard lookups enabled),
    # verifying that wildcard_cand bypasses the pre-filter and ranks first.
    ranked = rank_candidates(
        query="turn on broadcast dinner is ready",
        candidates=[cand_1, cand_2, wildcard_cand],
        max_candidates=1,
        rapidfuzz_prefilter_candidates=1,
        language="en",
    )
    assert len(ranked) == 1
    assert ranked[0].candidate == wildcard_cand


def test_rank_candidates_applies_slot_preferences_tiebreaker() -> None:
    """Verify that slot_preferences tie-breaks wildcard candidate ranking."""
    cand_shopping = Candidate(
        text="add shopping_list_item",
        intent_name="HassShoppingListAddItem",
        language="en",
        metadata={"literal_text": "add"},
    )
    cand_todo = Candidate(
        text="add todo_list_item",
        intent_name="HassListAddItem",
        language="en",
        metadata={"literal_text": "add"},
    )

    assert cand_shopping.has_wildcard
    assert cand_todo.has_wildcard

    query = "add milk"

    ranked_no_prefs = rank_candidates(
        query=query,
        candidates=[cand_todo, cand_shopping],
        max_candidates=2,
        language="en",
    )
    assert len(ranked_no_prefs) == 2
    assert ranked_no_prefs[0].candidate == cand_todo

    ranked_with_prefs = rank_candidates(
        query=query,
        candidates=[cand_todo, cand_shopping],
        max_candidates=2,
        language="en",
        slot_preferences={("shopping_list_item", "milk")},
    )
    assert len(ranked_with_prefs) == 2
    assert ranked_with_prefs[0].candidate == cand_shopping


def test_wildcard_lookups_coverage() -> None:
    """Exercise all wildcard index precomputation and ranking branches for coverage."""
    cand_always_pass = Candidate(
        text="message",
        intent_name="HassBroadcast",
        language="en",
        metadata={"literal_text": ""},
    )
    cand_var_len_0 = Candidate(
        text="broadcast message",
        intent_name="HassBroadcastVar0",
        language="en",
        metadata={"literal_text": "|broadcast"},
    )
    cand_normal = Candidate(
        text="broadcast message",
        intent_name="HassBroadcastNormal",
        language="en",
        metadata={"literal_text": "broadcast"},
    )

    # 1. Exercise indexer post-init (with wildcard indices)
    index = build_index("en", [cand_always_pass, cand_var_len_0, cand_normal])
    assert index.candidate_count == 3

    # 2. Exercise ranking with precomputed index wildcard structures
    ranked = index.rank("broadcast dinner")
    assert len(ranked) >= 1

    # 3. Exercise fallback ranking when wildcard_indices is None
    ranked_fallback = rank_candidates(
        query="broadcast dinner",
        candidates=[cand_always_pass, cand_var_len_0, cand_normal],
        max_candidates=1,
        rapidfuzz_prefilter_candidates=1,
        language="en",
    )
    assert len(ranked_fallback) == 1


def test_vietnamese_shopping_list_rehydration_selection() -> None:
    """Verify that wildcard length penalty correctly ranks specific template over generic ones."""
    cand_specific = Candidate(
        text="đặt shopping_list_item vào danh sách mua sắm",
        intent_name="HassShoppingListAddItem",
        language="vi",
        metadata={"literal_text": "đặt|vào danh sách mua sắm"},
    )
    cand_generic = Candidate(
        text="đặt shopping_list_item cho",
        intent_name="HassShoppingListAddItem",
        language="vi",
        metadata={"literal_text": "đặt|cho"},
    )

    ranked = rank_candidates(
        query="cho món bánh chuối vào danh sách mua sắm cho anh nhé",
        candidates=[cand_generic, cand_specific],
        max_candidates=2,
        language="vi",
    )

    assert len(ranked) == 2
    # Specific template should be ranked first because cand_generic's wildcard
    # is too long and gets penalized
    assert ranked[0].candidate == cand_specific
