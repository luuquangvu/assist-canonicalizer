"""Tests for lexical ranking and candidate indexing."""

from typing import Any, cast

import orjson
import pytest

from custom_components.assist_canonicalizer import ranking
from custom_components.assist_canonicalizer.candidate import (
    Candidate,
    CandidateSource,
    slot_alias_values_by_key,
)
from custom_components.assist_canonicalizer.const import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES,
    HIGH_CONFIDENCE_RELAXED_MIN_MARGIN,
    SAFE_INTENT_EVIDENCE_MAX_SCORE,
    FallbackReason,
)
from custom_components.assist_canonicalizer.indexer import build_index
from custom_components.assist_canonicalizer.normalization import normalize_text
from custom_components.assist_canonicalizer.ranking import (
    CharNGramIndex,
    RankedCandidate,
    ScoreBreakdown,
    _has_static_entity_uncovered_query_tokens,
    _has_static_slot_query_conflict,
    _has_wildcard_known_slot_token_absorption,
    _query_slot_tokens_from_index,
    _query_token_coverage,
    accepted_candidate,
    confidence_gate_rejection_reason,
    rank_candidates,
    rapidfuzz_similarity_normalized,
    token_count_ratio,
)
from custom_components.assist_canonicalizer.rehydration import wildcard_variants_analysis


def _fail_bm25_from_texts(*args: object, **kwargs: object) -> None:
    """Fail if ranking rebuilds BM25 data."""
    raise AssertionError("BM25 index should be cached by CanonicalIndex")


class _RapidFuzzSimilarityCounter:
    """Callable RapidFuzz mock that counts invocations."""

    def __init__(self) -> None:
        """Initialize the call counter."""
        self.calls = 0

    def __call__(self, query: str, candidate: str, **kwargs: object) -> float:
        """Count expensive RapidFuzz calls."""
        self.calls += 1
        return 0.5


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


def test_index_deduplicates_with_hassil_domain_area_preference() -> None:
    """Prefer HassIL-style domain-area candidates for same-source static duplicates."""
    generic = Candidate(
        text="turn on bathroom fan",
        intent_name="HassTurnOn",
        source=CandidateSource.BUILT_IN,
        language="en",
        metadata={
            "slots": orjson.dumps({"name": "bathroom fan", "name:fan": "bathroom fan"}).decode(
                "utf-8"
            )
        },
    )
    domain_area = Candidate(
        text="turn on bathroom fan",
        intent_name="HassTurnOn",
        source=CandidateSource.BUILT_IN,
        language="en",
        metadata={
            "slots": orjson.dumps(
                {
                    "domain": "fan",
                    "name": "all",
                    "name:fan": "bathroom fan",
                    "area": "bathroom",
                }
            ).decode("utf-8"),
            "static_slots": "domain,name",
        },
    )

    index = build_index("en", [generic, domain_area])

    assert index.candidate_count == 1
    slots = orjson.loads(index.candidates[0].metadata["slots"])
    assert slots["domain"] == "fan"
    assert slots["area"] == "bathroom"
    assert slots["name"] == "all"


@pytest.mark.parametrize("location_slot_name", ["area_name", "floor_name"])
def test_index_dedupe_preference_uses_shared_location_slot_names(
    location_slot_name: str,
) -> None:
    """Use shared entity/location slot aliases when selecting static duplicates."""
    generic = Candidate(
        text="turn on upstairs fan",
        intent_name="HassTurnOn",
        source=CandidateSource.BUILT_IN,
        language="en",
        metadata={"slots": orjson.dumps({"entity_name": "upstairs fan"}).decode("utf-8")},
    )
    domain_location = Candidate(
        text="turn on upstairs fan",
        intent_name="HassTurnOn",
        source=CandidateSource.BUILT_IN,
        language="en",
        metadata={
            "slots": orjson.dumps(
                {
                    "domain": "fan",
                    "name": "all",
                    location_slot_name: "upstairs",
                }
            ).decode("utf-8"),
            "static_slots": "domain,name",
        },
    )

    index = build_index("en", [generic, domain_location])

    assert index.candidate_count == 1
    slots = orjson.loads(index.candidates[0].metadata["slots"])
    assert slots["domain"] == "fan"
    assert slots[location_slot_name] == "upstairs"
    assert slots["name"] == "all"


def test_index_keeps_custom_source_before_domain_area_preference() -> None:
    """Prefer custom static duplicates before built-in structural preferences."""
    custom = Candidate(
        text="turn on bathroom fan",
        intent_name="HassTurnOn",
        source=CandidateSource.CUSTOM_SENTENCE,
        language="en",
        metadata={
            "slots": orjson.dumps({"name": "bathroom fan", "name:fan": "bathroom fan"}).decode(
                "utf-8"
            )
        },
    )
    built_in_domain_area = Candidate(
        text="turn on bathroom fan",
        intent_name="HassTurnOn",
        source=CandidateSource.BUILT_IN,
        language="en",
        metadata={
            "slots": orjson.dumps(
                {
                    "domain": "fan",
                    "name": "all",
                    "name:fan": "bathroom fan",
                    "area": "bathroom",
                }
            ).decode("utf-8"),
            "static_slots": "domain,name",
        },
    )

    index = build_index("en", [built_in_domain_area, custom])

    assert index.candidate_count == 1
    assert index.candidates[0].source == CandidateSource.CUSTOM_SENTENCE


def test_slot_alias_values_by_key_includes_direct_namespace_and_mapping_aliases() -> None:
    """Expose core slot aliases for benchmark and production diagnostics."""
    values_by_key = slot_alias_values_by_key(
        {"name:fan": "bathroom fan", "todo_list": "shopping list"},
        {"name": frozenset({"entity", "entity_name"}), "todo_list": frozenset({"name"})},
    )

    assert values_by_key["name"] == ("bathroom fan", "shopping list")
    assert values_by_key["name:fan"] == ("bathroom fan",)
    assert values_by_key["entity"] == ("bathroom fan",)
    assert values_by_key["entity_name"] == ("bathroom fan",)
    assert values_by_key["todo_list"] == ("shopping list",)


@pytest.mark.parametrize("static_slots_meta", [None, ["area"]])
def test_candidate_slot_tokens_ignore_non_string_static_slots_metadata(
    static_slots_meta: object,
) -> None:
    """Malformed static_slots metadata should not block slot-token extraction."""
    candidate = Candidate(
        text="turn on kitchen light",
        intent_name="HassTurnOn",
        metadata=cast(
            Any,
            {
                "slots": orjson.dumps({"area": "kitchen"}).decode("utf-8"),
                "static_slots": static_slots_meta,
            },
        ),
    )

    assert candidate.slot_tokens_set == frozenset({"kitchen"})


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


def test_rank_candidates_prefers_more_specific_slots_on_exact_score_tie() -> None:
    """Prefer richer slot extraction when candidates have identical lexical scores."""
    generic = Candidate(
        text="delete milk from shopping list",
        intent_name="HassShoppingListCompleteItem",
        language="en",
        metadata={"slots": orjson.dumps({"item": "milk"}).decode("utf-8")},
    )
    named_list = Candidate(
        text="delete milk from shopping list",
        intent_name="HassListRemoveItem",
        language="en",
        metadata={"slots": orjson.dumps({"item": "milk", "name": "shopping list"}).decode("utf-8")},
    )

    ranked = rank_candidates(
        "delete milk from shopping list",
        [generic, named_list],
        language="en",
    )

    assert ranked[0].candidate == named_list


@pytest.mark.parametrize(
    ("preferred_intent", "other_intent"),
    ranking._KNOWN_OPPOSING_INTENT_TIE_PAIRS,
)
def test_rank_candidates_prefers_known_opposing_intent_tie(
    preferred_intent: str,
    other_intent: str,
) -> None:
    """Apply HassIL known-pair preferences only for exact structural ties."""
    slots = orjson.dumps({"name": "wohnzimmerlicht"}).decode("utf-8")
    ranked = rank_candidates(
        "schalte wohnzimmerlicht",
        [
            Candidate(
                text="schalte wohnzimmerlicht ab",
                intent_name=other_intent,
                language="de",
                metadata={"slots": slots},
            ),
            Candidate(
                text="schalte wohnzimmerlicht an",
                intent_name=preferred_intent,
                language="de",
                metadata={"slots": slots},
            ),
        ],
        language="de",
    )

    assert ranked[0].candidate.intent_name == preferred_intent
    if preferred_intent == "HassTurnOn":
        assert accepted_candidate(ranked) is ranked[0]


def test_known_opposing_intent_preference_ignores_score_near_ties() -> None:
    """Keep numeric ordering when known opposing intents are not exact ties."""
    preferred_intent, other_intent = ranking._KNOWN_OPPOSING_INTENT_TIE_PAIRS[0]
    preferred = Candidate(text="schalte wohnzimmerlicht an", intent_name=preferred_intent)
    other = Candidate(text="schalte wohnzimmerlicht aus", intent_name=other_intent)

    preferences = ranking._intent_tie_preferences_by_index(
        (
            ranking._RankedItem(0.901, other, 0.901, 0.901, 0.901, 0.901, 0, 0.0, 1),
            ranking._RankedItem(0.900, preferred, 0.900, 0.900, 0.900, 0.900, 1, 0.0, 1),
        ),
        slot_preferences=None,
        rehydrated_cache={},
    )

    assert preferences == {}


def test_known_opposing_intent_preference_handles_float_epsilon_ties() -> None:
    """Treat tiny floating-point drift as an exact structural tie."""
    preferred_intent, other_intent = ranking._KNOWN_OPPOSING_INTENT_TIE_PAIRS[0]
    preferred = Candidate(text="schalte wohnzimmerlicht an", intent_name=preferred_intent)
    other = Candidate(text="schalte wohnzimmerlicht aus", intent_name=other_intent)
    other_item = ranking._RankedItem(
        0.9000000000001,
        other,
        0.9,
        0.9,
        0.9,
        0.9,
        0,
        0.0,
        1,
    )
    preferred_item = ranking._RankedItem(
        0.9,
        preferred,
        0.9,
        0.9,
        0.9,
        0.9,
        1,
        0.0,
        1,
    )

    preferences = ranking._intent_tie_preferences_by_index(
        (other_item, preferred_item),
        slot_preferences=None,
        rehydrated_cache={},
    )

    assert preferences[0][0] == 0
    assert preferences[1][0] == 1
    assert ranking._ranked_tuple_sort_key(
        preferred_item,
        slot_preferences=None,
        rehydrated_cache={},
        intent_tie_preferences=preferences,
    ) > ranking._ranked_tuple_sort_key(
        other_item,
        slot_preferences=None,
        rehydrated_cache={},
        intent_tie_preferences=preferences,
    )


def test_rank_candidates_does_not_apply_turn_on_tiebreak_to_unrelated_intents() -> None:
    """Keep unrelated exact structural ties stable instead of globally preferring TurnOn."""
    unrelated = Candidate(text="licht status", intent_name="HassGetState", language="de")
    turn_on = Candidate(text="licht status", intent_name="HassTurnOn", language="de")

    ranked = rank_candidates(
        "licht status",
        [unrelated, turn_on],
        language="de",
    )

    assert ranked[0].candidate == unrelated


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


def test_accepted_candidate_relaxes_margin_for_high_confidence_top() -> None:
    """Accept a high-confidence top candidate with a small positive margin."""
    top = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.82,
            char_ngram_score=0.82,
            bm25_score=0.82,
            intent_score=0.82,
            final_score=0.82,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="turn off kitchen light", intent_name="HassTurnOff", language="en"
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.81,
            char_ngram_score=0.81,
            bm25_score=0.81,
            intent_score=0.81,
            final_score=0.81,
        ),
    )

    assert accepted_candidate((top, competitor)) is top


def test_accepted_candidate_does_not_relax_margin_below_relaxed_threshold() -> None:
    """Reject high-confidence candidates below the relaxed margin threshold."""
    top = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.82,
            char_ngram_score=0.82,
            bm25_score=0.82,
            intent_score=0.82,
            final_score=0.82,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="turn off kitchen light", intent_name="HassTurnOff", language="en"
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.82,
            char_ngram_score=0.82,
            bm25_score=0.82,
            intent_score=0.82,
            final_score=0.82 - HIGH_CONFIDENCE_RELAXED_MIN_MARGIN + 0.001,
        ),
    )

    assert accepted_candidate((top, competitor)) is None


def test_accepted_candidate_keeps_user_strict_margin() -> None:
    """Do not relax the margin when the user configures a stricter gate."""
    top = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.82,
            char_ngram_score=0.82,
            bm25_score=0.82,
            intent_score=0.82,
            final_score=0.82,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="turn off kitchen light", intent_name="HassTurnOff", language="en"
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.81,
            char_ngram_score=0.81,
            bm25_score=0.81,
            intent_score=0.81,
            final_score=0.81,
        ),
    )

    assert accepted_candidate((top, competitor), min_margin=DEFAULT_MIN_MARGIN + 0.01) is None


def test_accepted_candidate_rejects_low_margin_without_high_confidence() -> None:
    """Keep ordinary fuzzy candidates behind the configured margin gate."""
    top = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.79,
            char_ngram_score=0.79,
            bm25_score=0.79,
            intent_score=0.79,
            final_score=0.79,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="turn off kitchen light", intent_name="HassTurnOff", language="en"
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.78,
            char_ngram_score=0.78,
            bm25_score=0.78,
            intent_score=0.78,
            final_score=0.78,
        ),
    )

    assert accepted_candidate((top, competitor)) is None
    assert confidence_gate_rejection_reason((top, competitor)) == FallbackReason.LOW_MARGIN


def test_accepted_candidate_rejects_weak_zero_intent_evidence() -> None:
    """Reject low-scoring fuzzy winners that only match entity text."""
    top = RankedCandidate(
        candidate=Candidate(text="water heater", intent_name="HassTurnOn", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.65,
            char_ngram_score=0.65,
            bm25_score=0.65,
            intent_score=0.0,
            final_score=0.65,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(text="water heater off", intent_name="HassTurnOff", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.59,
            char_ngram_score=0.59,
            bm25_score=0.59,
            intent_score=0.4,
            final_score=0.59,
        ),
    )

    assert accepted_candidate((top, competitor)) is None
    assert confidence_gate_rejection_reason((top, competitor)) == FallbackReason.LOW_CONFIDENCE


def test_accepted_candidate_keeps_stronger_zero_intent_synonym() -> None:
    """Do not reject stronger zero-intent matches solely because a rival is nearby."""
    top = RankedCandidate(
        candidate=Candidate(text="living room lamp on", intent_name="HassTurnOn", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.74,
            char_ngram_score=0.74,
            bm25_score=0.74,
            intent_score=0.0,
            final_score=0.74,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(text="living room lamp off", intent_name="HassTurnOff", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.69,
            char_ngram_score=0.69,
            bm25_score=0.69,
            intent_score=0.0,
            final_score=0.69,
        ),
    )

    assert accepted_candidate((top, competitor)) is top


def test_accepted_candidate_relaxes_for_safe_intent_evidence() -> None:
    """Accept close fuzzy winners with a clear intent-evidence advantage."""
    top = RankedCandidate(
        candidate=Candidate(
            text="turn on kitchen light",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": orjson.dumps({"area": "kitchen", "domain": "light"}).decode()},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.67,
            char_ngram_score=0.67,
            bm25_score=0.67,
            intent_score=0.70,
            final_score=0.67,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="turn off kitchen light",
            intent_name="HassTurnOff",
            language="en",
            metadata={"slots": orjson.dumps({"area": "kitchen", "domain": "light"}).decode()},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.65,
            char_ngram_score=0.65,
            bm25_score=0.65,
            intent_score=0.58,
            final_score=0.65,
        ),
    )

    assert accepted_candidate((top, competitor)) is top


def test_accepted_candidate_rejects_safe_intent_evidence_above_score_window() -> None:
    """Keep the safe intent-evidence relaxation bounded to medium-confidence matches."""
    top = RankedCandidate(
        candidate=Candidate(
            text="turn on kitchen light",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": orjson.dumps({"area": "kitchen", "domain": "light"}).decode()},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=SAFE_INTENT_EVIDENCE_MAX_SCORE + 0.001,
            char_ngram_score=SAFE_INTENT_EVIDENCE_MAX_SCORE + 0.001,
            bm25_score=SAFE_INTENT_EVIDENCE_MAX_SCORE + 0.001,
            intent_score=0.70,
            final_score=SAFE_INTENT_EVIDENCE_MAX_SCORE + 0.001,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="turn off kitchen light",
            intent_name="HassTurnOff",
            language="en",
            metadata={"slots": orjson.dumps({"area": "kitchen", "domain": "light"}).decode()},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=SAFE_INTENT_EVIDENCE_MAX_SCORE - 0.01,
            char_ngram_score=SAFE_INTENT_EVIDENCE_MAX_SCORE - 0.01,
            bm25_score=SAFE_INTENT_EVIDENCE_MAX_SCORE - 0.01,
            intent_score=0.58,
            final_score=SAFE_INTENT_EVIDENCE_MAX_SCORE - 0.01,
        ),
    )

    assert accepted_candidate((top, competitor)) is None


def test_accepted_candidate_rejects_broad_static_intent_evidence() -> None:
    """Do not relax broad domain/name static candidates on intent evidence alone."""
    top = RankedCandidate(
        candidate=Candidate(
            text="bathroom fan",
            intent_name="HassTurnOn",
            language="en",
            metadata={
                "slots": orjson.dumps(
                    {"area": "bathroom", "domain": "fan", "name": "all"}
                ).decode(),
                "static_slots": "domain,name",
            },
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.67,
            char_ngram_score=0.67,
            bm25_score=0.67,
            intent_score=0.75,
            final_score=0.67,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="turn off bathroom fan",
            intent_name="HassTurnOff",
            language="en",
            metadata={"slots": orjson.dumps({"area": "bathroom", "domain": "fan"}).decode()},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.65,
            char_ngram_score=0.65,
            bm25_score=0.65,
            intent_score=0.50,
            final_score=0.65,
        ),
    )

    assert accepted_candidate((top, competitor)) is None


def test_accepted_candidate_rejects_weak_intent_evidence_score() -> None:
    """Keep low-scoring intent-advantage winners behind the margin gate."""
    top = RankedCandidate(
        candidate=Candidate(
            text="volume up in the kitchen",
            intent_name="HassSetVolumeRelative",
            language="en",
            metadata={"slots": orjson.dumps({"area": "kitchen"}).decode()},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.52,
            char_ngram_score=0.52,
            bm25_score=0.52,
            intent_score=0.64,
            final_score=0.52,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="turn on kitchen light",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": orjson.dumps({"area": "kitchen", "domain": "light"}).decode()},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.51,
            char_ngram_score=0.51,
            bm25_score=0.51,
            intent_score=0.48,
            final_score=0.51,
        ),
    )

    assert accepted_candidate((top, competitor)) is None


def test_accepted_candidate_relaxes_close_empty_slot_intents() -> None:
    """Accept close no-slot informational intents above the relaxed evidence floor."""
    top = RankedCandidate(
        candidate=Candidate(
            text="what is the weather", intent_name="HassGetWeather", language="en"
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.61,
            char_ngram_score=0.61,
            bm25_score=0.61,
            intent_score=0.54,
            final_score=0.61,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="what day is it today", intent_name="HassGetCurrentDate", language="en"
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.575,
            char_ngram_score=0.575,
            bm25_score=0.575,
            intent_score=0.57,
            final_score=0.575,
        ),
    )

    assert accepted_candidate((top, competitor)) is top


def test_accepted_candidate_rejects_turn_tie_with_different_slots() -> None:
    """Do not apply the implicit-on tie rule across different slot targets."""
    top = RankedCandidate(
        candidate=Candidate(
            text="schalte wohnzimmerlicht an",
            intent_name="HassTurnOn",
            language="de",
            metadata={"slots": orjson.dumps({"name": "wohnzimmerlicht"}).decode("utf-8")},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.85,
            char_ngram_score=0.85,
            bm25_score=0.85,
            intent_score=0.85,
            final_score=0.85,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="schalte küchenlicht ab",
            intent_name="HassTurnOff",
            language="de",
            metadata={"slots": orjson.dumps({"name": "küchenlicht"}).decode("utf-8")},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.85,
            char_ngram_score=0.85,
            bm25_score=0.85,
            intent_score=0.85,
            final_score=0.85,
        ),
    )

    assert accepted_candidate((top, competitor)) is None


def test_confidence_gate_rejection_reason_distinguishes_threshold_failures() -> None:
    """Return the specific confidence-gate reason for rejected ranked candidates."""
    low_confidence = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.1,
            char_ngram_score=0.1,
            bm25_score=0.1,
            intent_score=0.1,
            final_score=DEFAULT_MIN_CONFIDENCE - 0.01,
        ),
    )
    top = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.8,
            char_ngram_score=0.8,
            bm25_score=0.8,
            intent_score=1.0,
            final_score=0.8,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="turn off kitchen light", intent_name="HassTurnOff", language="en"
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.79,
            char_ngram_score=0.79,
            bm25_score=0.79,
            intent_score=1.0,
            final_score=0.79,
        ),
    )

    assert confidence_gate_rejection_reason((low_confidence,)) == FallbackReason.LOW_CONFIDENCE
    assert (
        confidence_gate_rejection_reason((top, competitor), min_margin=0.05)
        == FallbackReason.LOW_MARGIN
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


def test_accepted_candidate_rejects_identical_text_with_different_slots() -> None:
    """Treat same-text competitors as ambiguous when their slots differ."""
    score = DEFAULT_MIN_CONFIDENCE + 0.1
    top = RankedCandidate(
        candidate=Candidate(
            text="mở cửa phòng ngủ",
            intent_name="HassTurnOn",
            language="vi",
            metadata={
                "slots": orjson.dumps(
                    {"domain": "cover", "device_class": "door", "area": "phòng ngủ"}
                ).decode("utf-8")
            },
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=score,
            char_ngram_score=score,
            bm25_score=score,
            intent_score=score,
            final_score=score,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="mở cửa phòng ngủ",
            intent_name="HassTurnOff",
            language="vi",
            metadata={
                "slots": orjson.dumps(
                    {"domain": "lock", "name": "all", "area": "phòng ngủ"}
                ).decode("utf-8")
            },
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=score,
            char_ngram_score=score,
            bm25_score=score,
            intent_score=score,
            final_score=score,
        ),
    )

    assert accepted_candidate((top, competitor)) is None


def test_static_entity_uncovered_query_tokens_detects_broad_candidates() -> None:
    """Detect broad static entity candidates that leave specific query terms uncovered."""
    slots = {"domain": "lock", "name": "all", "area": "bedroom"}

    assert _has_static_entity_uncovered_query_tokens(
        ("open", "bedroom", "window"),
        frozenset({"open", "bedroom", "door"}),
        slots,
        frozenset({"domain", "name"}),
    )
    assert not _has_static_entity_uncovered_query_tokens(
        ("open", "bedroom", "window"),
        frozenset({"open", "bedroom", "window"}),
        slots,
        frozenset({"domain", "name"}),
    )
    assert not _has_static_entity_uncovered_query_tokens(
        ("open", "bedroom", "window"),
        frozenset({"open", "bedroom"}),
        {"name": "bedroom window"},
        frozenset(),
    )
    assert not _has_static_entity_uncovered_query_tokens(
        ("open", "bedroom", "window"),
        frozenset({"open", "bedroom", "door"}),
        {"domain": "lock", "name": "all", "entity": "bedroom window"},
        frozenset({"domain", "name"}),
    )


def test_static_slot_query_conflict_detects_missing_active_slot_tokens() -> None:
    """Detect static slot candidates that miss explicit slot-like query tokens."""
    slots = {"area": "bedroom", "domain": "fan"}

    assert _has_static_slot_query_conflict(
        frozenset({"bedroom", "window"}),
        frozenset({"turn", "off", "bedroom", "fan"}),
        slots,
        frozenset({"domain"}),
    )
    assert not _has_static_slot_query_conflict(
        frozenset({"bedroom"}),
        frozenset({"turn", "off", "bedroom", "fan"}),
        slots,
        frozenset({"domain"}),
    )
    assert not _has_static_slot_query_conflict(
        frozenset({"bedroom", "window"}),
        frozenset({"turn", "off", "bedroom", "window"}),
        slots,
        frozenset({"domain"}),
    )
    assert not _has_static_slot_query_conflict(
        frozenset({"bedroom", "window"}),
        frozenset({"turn", "off", "bedroom", "fan"}),
        {"name": "bedroom fan"},
        frozenset(),
    )


def test_query_slot_tokens_from_index_uses_bounded_fuzzy_slot_matches() -> None:
    """Map bounded one-edit query typos to indexed slot tokens."""
    indexed: Any = {
        "badezimmerlüfter": (0,),
        "window": (1,),
        "fan": (2,),
        "door": (3,),
    }

    assert _query_slot_tokens_from_index(
        frozenset({"badzimmerlüfter", "windw", "fon", "doar"}),
        (0, 1, 2, 3),
        indexed,
    ) == frozenset({"badezimmerlüfter", "door", "window"})


def test_query_slot_tokens_from_index_rejects_distant_fuzzy_slot_matches() -> None:
    """Reject slot-token typos outside the bounded one-edit guard."""
    indexed: Any = {"badezimmerlüfter": (0,)}

    assert (
        _query_slot_tokens_from_index(
            frozenset(
                {
                    "badzximmerlüfter",
                    "badezimmerlüfterfoobar",
                    "xadezimmerlüfter",
                }
            ),
            (0,),
            indexed,
        )
        == frozenset()
    )


def test_rank_candidates_penalizes_static_competitor_with_fuzzy_slot_token() -> None:
    """Use fuzzy slot tokens when detecting static-slot query conflicts."""
    correct = Candidate(
        text="schalt badezimmerlüfter aus",
        intent_name="HassTurnOff",
        language="de",
        metadata={
            "literal_text": "schalt aus",
            "slots": orjson.dumps({"name": "badezimmerlüfter"}).decode("utf-8"),
        },
        slot_values=("badezimmerlüfter",),
    )
    static_competitor = Candidate(
        text="mach Fenster auf",
        intent_name="HassTurnOn",
        language="de",
        metadata={
            "literal_text": "mach auf",
            "slots": orjson.dumps({"domain": "cover", "device_class": "window"}).decode("utf-8"),
            "static_slots": "domain,device_class",
        },
        slot_values=("cover", "window"),
    )

    ranked = rank_candidates(
        "mach mal badzimmerlüfter aus bitte",
        [static_competitor, correct],
        max_candidates=2,
    )

    assert ranked[0].candidate == correct
    assert ranked[1].candidate == static_competitor


def test_wildcard_known_slot_token_absorption_detects_slot_like_free_text() -> None:
    """Detect wildcard candidates that absorb known entity/location tokens."""
    assert _has_wildcard_known_slot_token_absorption(
        frozenset({"salon", "lumiere"}),
        frozenset({"salon", "lumiere"}),
        frozenset(),
    )
    assert not _has_wildcard_known_slot_token_absorption(
        frozenset({"salon", "lumiere"}),
        frozenset({"salon", "lumiere"}),
        frozenset({"salon"}),
    )
    assert not _has_wildcard_known_slot_token_absorption(
        frozenset(),
        frozenset({"jazz"}),
        frozenset(),
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

    monkeypatch.setattr(ranking.BM25Index, "from_normalized_texts", _fail_bm25_from_texts)

    ranked = index.rank("turn on kitchen lamp")

    assert ranked[0].candidate.intent_name == "HassTurnOn"


def test_rank_candidates_prefilters_rapidfuzz_work(monkeypatch) -> None:
    """Limit expensive RapidFuzz scoring to the configured prefilter size."""
    candidates = [
        Candidate(text=f"device {index}", intent_name="HassTurnOn", language="en")
        for index in range(DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES + 50)
    ]
    rapidfuzz_counter = _RapidFuzzSimilarityCounter()
    monkeypatch.setattr(
        ranking,
        "rapidfuzz_similarity_normalized",
        rapidfuzz_counter,
    )

    ranking.rank_candidates("device 1", candidates, max_candidates=5)

    assert rapidfuzz_counter.calls == DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES


def test_rank_candidates_validates_candidate_slot_token_length() -> None:
    """Reject precomputed slot-token data that does not match the candidate list."""
    candidates = [Candidate(text="turn on kitchen light", intent_name="HassTurnOn")]

    with pytest.raises(ValueError, match="candidate_slot_tokens length must match candidates"):
        rank_candidates(
            "turn on kitchen light",
            candidates,
            candidate_slot_tokens=(),
        )


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


def test_rank_candidates_scores_single_token_exact_literal_above_coverage() -> None:
    """Keep exact single-token literal matches from falling back to coverage."""
    candidate = Candidate(
        text="turn on lamp",
        intent_name="HassTurnOn",
        language="en",
        metadata={"literal_text": "turn"},
        slot_values=("lamp",),
    )

    ranked = rank_candidates(
        query="turn kitchen lamp",
        candidates=[candidate],
        max_candidates=1,
        language="en",
        positional_literal_tokens=frozenset({"turn", "kitchen"}),
    )

    assert ranked[0].scores.intent_score == 1.0


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


def test_positional_lookup_uses_bounded_fuzzy_literal_matches() -> None:
    """Match high-confidence literal typos without accepting weaker transpositions."""
    lookup = ranking._build_positional_lookup(
        frozenset({"close", "turn", "tắt"}),
        frozenset({"clse", "trun", "tắ"}),
    )

    assert lookup["close"] == frozenset({"clse"})
    assert lookup["tắt"] == frozenset({"tắ"})
    assert "turn" not in lookup


def test_rank_candidates_uses_fuzzy_literal_action_typo() -> None:
    """Boost the intended action when a literal action token has a deletion typo."""
    close_window = Candidate(
        text="close bedroom window",
        intent_name="HassTurnOff",
        language="en",
        metadata={"literal_text": "close"},
    )
    raise_window = Candidate(
        text="raise bedroom window",
        intent_name="HassTurnOn",
        language="en",
        metadata={"literal_text": "raise"},
    )

    ranked = rank_candidates(
        "clse bedroom window",
        [raise_window, close_window],
        max_candidates=2,
        language="en",
    )

    assert ranked[0].candidate is close_window
    assert ranked[0].scores.intent_score > 0.45
    assert accepted_candidate(ranked) is ranked[0]


def test_rank_candidates_penalizes_numeric_slot_without_query_number() -> None:
    """Do not let generated numeric-slot variants win when the query has no number."""
    relative = Candidate(
        text="volume up",
        intent_name="HassSetVolumeRelative",
        language="en",
        metadata={"slots": orjson.dumps({"volume_step": "up"}).decode("utf-8")},
    )
    absolute = Candidate(
        text="volume up",
        intent_name="HassSetVolume",
        language="en",
        metadata={"slots": orjson.dumps({"volume_level": 0}).decode("utf-8")},
    )

    ranked = rank_candidates(
        "volume up",
        [absolute, relative],
        max_candidates=2,
        language="en",
    )

    assert ranked[0].candidate is relative
    assert ranked[1].candidate is absolute


def test_rank_candidates_keeps_numeric_slot_with_query_number() -> None:
    """Allow numeric-slot variants to win when the query contains a number."""
    relative = Candidate(
        text="volume 10",
        intent_name="HassSetVolumeRelative",
        language="en",
        metadata={"slots": orjson.dumps({"volume_step": "up"}).decode("utf-8")},
    )
    absolute = Candidate(
        text="volume 10",
        intent_name="HassSetVolume",
        language="en",
        metadata={"slots": orjson.dumps({"volume_level": 10}).decode("utf-8")},
    )

    ranked = rank_candidates(
        "volume 10",
        [absolute, relative],
        max_candidates=2,
        language="en",
    )

    assert ranked[0].candidate is absolute
    assert ranked[1].candidate is relative


def test_rank_candidates_uses_intent_context_for_location_slots() -> None:
    """Prefer candidates whose location slot agrees with Home Assistant context."""
    living_room = Candidate(
        text="increase volume",
        intent_name="HassSetVolumeRelative",
        language="en",
        metadata={"slots": orjson.dumps({"area": "living room"}).decode("utf-8")},
    )
    office = Candidate(
        text="increase volume",
        intent_name="HassSetVolumeRelative",
        language="en",
        metadata={"slots": orjson.dumps({"area": "office"}).decode("utf-8")},
    )

    ranked = rank_candidates(
        "increase volume",
        [office, living_room],
        max_candidates=2,
        language="en",
        intent_context={"area": {"value": "Living Room", "text": "Living Room"}},
    )

    assert ranked[0].candidate is living_room
    assert ranked[1].candidate is office


def test_rank_candidates_rewards_context_supplied_slots() -> None:
    """Boost candidates that declare a slot supplied by HassIL intent context."""
    context_scoped = Candidate(
        text="mute player",
        intent_name="HassMediaPlayerMute",
        language="en",
        metadata={"context_slots": "area"},
    )
    unscoped = Candidate(
        text="mute player",
        intent_name="HassMediaPlayerMute",
        language="en",
    )

    ranked = rank_candidates(
        "mute",
        [unscoped, context_scoped],
        max_candidates=2,
        language="en",
        intent_context={"area": "living room"},
    )

    assert ranked[0].candidate is context_scoped
    assert ranked[1].candidate is unscoped


def test_rank_candidates_penalizes_unanchored_entity_slot() -> None:
    """Prefer generic static entity slots over named entities absent from the query."""
    generic = Candidate(
        text="turn on light",
        intent_name="HassTurnOn",
        language="en",
        metadata={
            "slots": orjson.dumps({"domain": "light", "name": "all"}).decode("utf-8"),
            "static_slots": "domain,name",
        },
    )
    named = Candidate(
        text="turn on light",
        intent_name="HassTurnOn",
        language="en",
        metadata={"slots": orjson.dumps({"name": "bedroom lamp"}).decode("utf-8")},
    )

    ranked = rank_candidates(
        "turn on light",
        [named, generic],
        max_candidates=2,
        language="en",
    )

    assert ranked[0].candidate is generic
    assert ranked[1].candidate is named


def test_rank_candidates_penalizes_entity_only_uncovered_query_tokens() -> None:
    """Do not let a bare entity candidate explain away extra query action words."""
    entity_only = Candidate(
        text="the bathroom fan",
        intent_name="HassTurnOn",
        language="en",
        metadata={"slots": orjson.dumps({"name": "bathroom fan"}).decode("utf-8")},
    )
    action_candidate = Candidate(
        text="the bathroom fan off",
        intent_name="HassTurnOff",
        language="en",
        metadata={
            "slots": orjson.dumps({"area": "bathroom", "domain": "fan"}).decode("utf-8"),
            "literal_text": "off",
        },
    )

    ranked = rank_candidates(
        "kill the bathroom fan",
        [entity_only, action_candidate],
        max_candidates=2,
        language="en",
    )

    assert ranked[0].candidate is action_candidate
    assert ranked[1].candidate is entity_only


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


@pytest.mark.current_intents
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


@pytest.mark.current_intents
def test_rank_candidates_does_not_slot_penalize_rehydrated_wildcard_placeholder() -> None:
    """Do not treat a rehydrated wildcard placeholder as a missing slot token."""
    candidate = Candidate(
        text="broadcast message",
        intent_name="HassBroadcast",
        language="en",
        metadata={
            "sentence_template": "(broadcast|announce) {message}",
            "literal_text": "broadcast|announce",
            "wildcard_slots": "message",
            "slots": '{"message":"message"}',
        },
        slot_values=("message",),
    )
    second_candidate = Candidate(
        text="status status",
        intent_name="HassStatus",
        language="en",
        metadata={
            "sentence_template": "{status}",
            "slots": '{"status":"status"}',
        },
        slot_values=("status",),
    )

    ranked = rank_candidates(
        query="broadcast dinner is ready status",
        candidates=[candidate, second_candidate],
        max_candidates=2,
        language="en",
    )

    assert len(ranked) == 2
    assert ranked[0].candidate == candidate
    assert ranked[0].scores.final_score > DEFAULT_MIN_CONFIDENCE
    assert accepted_candidate(ranked) is ranked[0]


@pytest.mark.current_intents
def test_rank_candidates_keeps_slot_penalty_for_leading_rehydrated_wildcard() -> None:
    """Keep slot conflict protection for generic leading free-text wildcards."""
    generic_media = Candidate(
        text="search_query starten",
        intent_name="HassMediaSearchAndPlay",
        language="de",
        metadata={
            "sentence_template": "{search_query} <starten_end_of_sentence>",
            "literal_text": "starten",
            "wildcard_slots": "search_query",
            "slots": '{"search_query":"search_query"}',
        },
        slot_values=("search_query",),
    )
    assert generic_media.wildcard_info == (0, "search_query")
    vacuum = Candidate(
        text="staubsauger Reinigung starten",
        intent_name="HassVacuumStart",
        language="de",
        metadata={
            "literal_text": "staubsauger|reinigung|starten",
            "slots": '{"name":"staubsauger"}',
        },
        slot_values=("staubsauger",),
    )

    ranked = rank_candidates(
        query="reinigung mit staubsauger starten",
        candidates=[generic_media, vacuum],
        max_candidates=2,
        language="de",
    )

    assert ranked[0].candidate == vacuum


@pytest.mark.current_intents
def test_rank_candidates_allows_leading_rehydrated_wildcard_with_slot_anchor() -> None:
    """Count rehydrated leading wildcard tokens when another slot anchors the candidate."""
    wildcard_with_anchor = Candidate(
        text="search_query auf fernseher starten",
        intent_name="HassMediaSearchAndPlay",
        language="de",
        metadata={
            "sentence_template": "{search_query} auf {name} <starten_end_of_sentence>",
            "literal_text": "auf|starten",
            "wildcard_slots": "search_query",
            "slots": '{"search_query":"search_query","name":"fernseher"}',
        },
        slot_values=("search_query", "fernseher"),
    )
    assert wildcard_with_anchor.wildcard_info == (0, "search_query")
    music_device = Candidate(
        text="musik auf radio starten",
        intent_name="HassMediaSearchAndPlay",
        language="de",
        metadata={
            "literal_text": "musik|radio|starten",
            "slots": '{"media":"musik","name":"radio"}',
        },
        slot_values=("musik", "radio"),
    )

    ranked = rank_candidates(
        query="musik auf fernseher starten",
        candidates=[wildcard_with_anchor, music_device],
        max_candidates=2,
        language="de",
    )

    assert ranked[0].candidate == wildcard_with_anchor
    assert ranked[0].scores.final_score > 0.95


def test_rank_candidates_clamps_high_confidence_slot_penalty_multiplier() -> None:
    """Do not let min_confidence above 1 turn slot penalties into score boosts."""
    candidate = Candidate(
        text="turn on kitchen lamp",
        intent_name="HassTurnOn",
        language="en",
        slot_values=("kitchen lamp",),
    )

    penalized = rank_candidates(
        query="turn on kitchen light",
        candidates=[candidate],
        language="en",
        min_confidence=1.2,
    )
    unpenalized = rank_candidates(
        query="turn on kitchen light",
        candidates=[candidate],
        candidate_slot_tokens=(frozenset(),),
        language="en",
        min_confidence=1.2,
    )

    assert penalized[0].scores.final_score <= unpenalized[0].scores.final_score


@pytest.mark.current_intents
def test_wildcard_variants_analysis_removes_wildcard_and_deduplicates() -> None:
    """Compute wildcard literal coverage once per equivalent variant/wildcard pair."""
    candidate = Candidate(
        text="message",
        intent_name="HassBroadcast",
        language="en",
        metadata={"literal_text": "message|messages|broadcast message"},
    )

    variants_with_len, all_tokens = wildcard_variants_analysis(candidate)

    assert variants_with_len == (
        (frozenset(), 0, 0),
        (frozenset({"messages"}), 1, 1),
        (frozenset({"broadcast"}), 1, 1),
    )
    assert all_tokens == frozenset({"broadcast", "messages"})


@pytest.mark.current_intents
def test_wildcard_variants_analysis_removes_embedded_structured_wildcard() -> None:
    """Do not keep placeholder-bearing tokens as wildcard literal anchors."""
    candidate = Candidate(
        text="search_querypodcast",
        intent_name="HassMediaSearchAndPlay",
        language="de",
        metadata={"literal_text": "spiel den search_querypodcast|spiel den podcast"},
    )

    variants_with_len, all_tokens = wildcard_variants_analysis(candidate)

    assert variants_with_len == (
        (frozenset({"den", "spiel"}), 2, 2),
        (frozenset({"den", "podcast", "spiel"}), 3, 3),
    )
    assert all_tokens == frozenset({"den", "podcast", "spiel"})


@pytest.mark.current_intents
def test_wildcard_variants_analysis_removes_embedded_single_word_wildcard() -> None:
    """Strip single-word wildcard names embedded in literal tokens."""
    candidate = Candidate(
        text="urgentmessage",
        intent_name="HassBroadcast",
        language="en",
        metadata={"literal_text": "broadcast urgentmessage|broadcast"},
    )

    variants_with_len, all_tokens = wildcard_variants_analysis(candidate)

    assert variants_with_len == ((frozenset({"broadcast"}), 1, 1),)
    assert all_tokens == frozenset({"broadcast"})


@pytest.mark.current_intents
def test_wildcard_variants_analysis_keeps_multilingual_word_containing_wildcard() -> None:
    """Do not strip non-English words that merely contain a wildcard name."""
    candidate = Candidate(
        text="messagerie",
        intent_name="HassBroadcast",
        language="fr",
        metadata={"literal_text": "messagerie|diffuser message"},
    )

    variants_with_len, all_tokens = wildcard_variants_analysis(candidate)

    assert variants_with_len == (
        (frozenset({"messagerie"}), 1, 1),
        (frozenset({"diffuser"}), 1, 1),
    )
    assert all_tokens == frozenset({"diffuser", "messagerie"})


@pytest.mark.current_intents
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
        wildcard_token_to_indices={},
        wildcard_literal_tokens_by_index={},
        wildcard_min_required_by_index={},
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


@pytest.mark.current_intents
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


def test_slot_matching_penalty_robustness_against_wrong_intent() -> None:
    """Verify slot suffix mismatch does not rank correct intent below wrong intent."""
    cand_with_suffix = Candidate(
        text="tắt Quạt phòng khách 1",
        intent_name="HassTurnOff",
        language="vi",
        metadata={"literal_text": "tắt"},
        slot_values=("quạt phòng khách 1",),
    )
    cand_correct_suffix = Candidate(
        text="tắt Quạt phòng khách 2",
        intent_name="HassTurnOff",
        language="vi",
        metadata={"literal_text": "tắt"},
        slot_values=("quạt phòng khách 2",),
    )
    cand_wrong_intent = Candidate(
        text="quạt phòng khách",
        intent_name="HassTurnOn",
        language="vi",
        metadata={"literal_text": ""},
        slot_values=("quạt phòng khách",),
    )

    # 1. Query without digits: "tắt quạt phòng khách"
    # Both candidates with suffixes should rank above the wrong intent,
    # because the wrong intent is missing the critical action verb "tắt".
    ranked = rank_candidates(
        query="tắt quạt phòng khách",
        candidates=[cand_wrong_intent, cand_with_suffix, cand_correct_suffix],
        max_candidates=3,
        language="vi",
    )
    assert len(ranked) == 3
    # The suffix candidates should rank top because digit "1" and "2" are ignored
    assert ranked[0].candidate in (cand_with_suffix, cand_correct_suffix)
    assert ranked[1].candidate in (cand_with_suffix, cand_correct_suffix)

    # 2. Query with digits: "tắt quạt phòng khách 2"
    # Since query has digits, numeric slot tokens are not ignored.
    # cand_correct_suffix should rank first (perfect slot match),
    # and cand_with_suffix should be penalized (mismatching slot token "1" vs "2").
    ranked_with_digits = rank_candidates(
        query="tắt quạt phòng khách 2",
        candidates=[cand_wrong_intent, cand_with_suffix, cand_correct_suffix],
        max_candidates=3,
        language="vi",
    )
    assert len(ranked_with_digits) == 3
    assert ranked_with_digits[0].candidate == cand_correct_suffix
    assert ranked_with_digits[1].candidate == cand_with_suffix


def test_slot_matching_penalty_ignores_static_slots() -> None:
    """Verify that slot matching penalty calculation ignores static slots."""
    metadata = {
        "slots": orjson.dumps({"area": "phòng khách 1", "domain": "fan", "name": "all"}).decode(
            "utf-8"
        ),
        "static_slots": "domain,name",
        "literal_text": "tắt quạt",
    }

    cand_generic = Candidate(
        text="tắt quạt phòng khách 1",
        intent_name="HassTurnOff",
        language="vi",
        metadata=metadata,
        slot_values=("phòng khách 1", "fan", "all"),
    )

    cand_competitor = Candidate(
        text="tắt quạt phòng khách 2",
        intent_name="HassTurnOff",
        language="vi",
        metadata={
            "slots": orjson.dumps({"area": "phòng khách 2"}).decode("utf-8"),
            "literal_text": "tắt quạt",
        },
        slot_values=("phòng khách 2",),
    )

    ranked = rank_candidates(
        query="tắt quạt phòng khách 1",
        candidates=[cand_generic, cand_competitor],
        max_candidates=2,
        language="vi",
    )
    assert len(ranked) == 2
    generic_result = next(r for r in ranked if r.candidate == cand_generic)

    cand_without_static = Candidate(
        text="tắt quạt phòng khách 1",
        intent_name="HassTurnOff",
        language="vi",
        metadata={
            "slots": orjson.dumps({"area": "phòng khách 1", "domain": "fan", "name": "all"}).decode(
                "utf-8"
            ),
            "literal_text": "tắt quạt",
        },
        slot_values=("phòng khách 1", "fan", "all"),
    )
    ranked_without_static = rank_candidates(
        query="tắt quạt phòng khách 1",
        candidates=[cand_without_static, cand_competitor],
        max_candidates=2,
        language="vi",
    )
    without_static_result = next(
        r for r in ranked_without_static if r.candidate == cand_without_static
    )

    assert generic_result.scores.final_score > without_static_result.scores.final_score
