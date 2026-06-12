"""Tests for lexical ranking and candidate indexing."""

import pytest

from custom_components.assist_canonicalizer import ranking
from custom_components.assist_canonicalizer.candidate import Candidate, CandidateSource
from custom_components.assist_canonicalizer.const import DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES
from custom_components.assist_canonicalizer.indexer import build_index
from custom_components.assist_canonicalizer.ranking import (
    _query_token_coverage,
    accepted_candidate,
    intent_action_score,
    rapidfuzz_similarity_normalized,
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

    assert accepted_candidate(ranked, min_confidence=0.1, min_margin=0.99) is None


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

    assert accepted_candidate(ranked, min_confidence=0.1, min_margin=0.99) is ranked[0]


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
    assert accepted_candidate(ranked, min_confidence=0.1, min_margin=0.99) is not None


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

    def fake_rapidfuzz_similarity(query: str, candidate: str) -> float:
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


def test_intent_action_score_supports_alternatives() -> None:
    """Verify that intent_action_score handles multiple options separated by pipe."""
    candidate = Candidate(
        text="bật quạt",
        intent_name="HassTurnOn",
        language="vi",
        metadata={"literal_text": "bật|mở|bật lên"},
    )
    assert ranking.intent_action_score("bật quạt", candidate) == 1.0
    assert ranking.intent_action_score("mở quạt", candidate) == 1.0
    assert ranking.intent_action_score("bật lên quạt", candidate) == 1.0
    assert ranking.intent_action_score("tắt quạt", candidate) == 0.0


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
    candidate = Candidate(
        text="đèn phòng khách",
        intent_name="HassTurnOn",
        language="vi",
    )
    score = intent_action_score("tắt đèn phòng khách", candidate)
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
