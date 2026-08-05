"""Tests for lexical ranking and candidate indexing."""

import unicodedata
from dataclasses import replace
from typing import Any, cast

import orjson
import pytest

from custom_components.assist_canonicalizer import ranking
from custom_components.assist_canonicalizer.bm25 import BM25Index
from custom_components.assist_canonicalizer.candidate import (
    Candidate,
    CandidateSource,
    candidate_source_priority,
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
    SlotTokenIndex,
    WildcardVariantGroup,
    _best_positional_score,
    _calculate_slot_penalty,
    _check_and_calculate_conflict_penalty,
    _get_wildcard_slot_tokens,
    _has_static_entity_uncovered_query_tokens,
    _has_static_slot_query_conflict,
    _has_wildcard_known_slot_token_absorption,
    _is_numeric_slot_value,
    _per_pair_positional_threshold,
    _positional_similarity,
    _query_slot_tokens_from_candidates,
    _query_token_coverage,
    _rank_prefilter_keys,
    _rank_prefilter_keys_with_sparse_bm25,
    _raw_cached_fuzz_ratio,
    _ScoringContext,
    _sparse_prefilter_keys_from_intersections,
    _SparsePrefilterKeys,
    _top_additional_slot_indices,
    _top_additional_wildcard_indices,
    _top_prefilter_indices,
    _top_sparse_prefilter_indices,
    _wildcard_variants_match,
    accepted_candidate,
    clear_ranking_caches,
    confidence_gate_rejection_reason,
    evaluate_confidence_gates,
    rank_candidates,
    rapidfuzz_similarity_normalized,
    token_count_ratio,
)
from custom_components.assist_canonicalizer.rehydration import (
    WildcardVariantAnalysis,
    _extract_original_span,
    _extract_wc_value,
    _find_rehydration_boundaries,
    _is_wildcard_literal_token,
    _trim_wildcard_overlaps,
    get_wildcard_rehydration,
    rehydrate_wildcard_slots,
    wildcard_variants_analysis,
)
from custom_components.assist_canonicalizer.utils import (
    register_custom_wildcards_from_sources,
    wildcard_slot_names_sorted,
)


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


def test_candidate_source_priority_policy_is_exhaustive() -> None:
    """Require every candidate source to have an intentional trust priority."""
    expected_priorities = {
        CandidateSource.CUSTOM_SENTENCE: 0,
        CandidateSource.BUILT_IN: 1,
        CandidateSource.GENERATED_SAMPLE: 2,
    }

    assert set(expected_priorities) == set(CandidateSource)
    assert {
        source: candidate_source_priority(source) for source in CandidateSource
    } == expected_priorities


def test_candidate_source_priority_rejects_unconfigured_source() -> None:
    """Explain incomplete source-priority policies instead of leaking a KeyError."""
    unsupported_source = cast(CandidateSource, "future_source")

    with pytest.raises(ValueError, match=r"No priority configured.*future_source"):
        candidate_source_priority(unsupported_source)


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


def test_candidate_surface_slot_token_properties_preserve_exclusions() -> None:
    """Cache localized slot tokens while excluding static and wildcard slots."""
    candidate = Candidate(
        text="play local station",
        intent_name="HassMediaSearchAndPlay",
        metadata={
            "slots": '{"domain":"media_player","name":"window","query":"free words"}',
            "slots_raw": '{"name":"fenster","query":"freie wörter"}',
            "static_slots": "domain",
            "wildcard_slots": "query",
        },
    )

    assert candidate.surface_slot_tokens == frozenset({"fenster", "freie", "wörter"})
    assert candidate.concrete_surface_slot_tokens == frozenset({"fenster"})
    assert candidate.surface_slot_tokens is candidate.surface_slot_tokens
    assert candidate.concrete_surface_slot_tokens is candidate.concrete_surface_slot_tokens


def test_concrete_surface_slot_tokens_excludes_inferred_wildcards() -> None:
    """Exclude registered free-text slots when explicit wildcard metadata is absent."""
    language = "candidate-property-wildcard"
    register_custom_wildcards_from_sources(
        language,
        {"custom": {"lists": {"local_free_text": {"wildcard": True}}}},
    )
    candidate = Candidate(
        text="say local_free_text to kitchen speaker",
        intent_name="TestIntent",
        language=language,
        metadata={
            "slots": '{"local_free_text":"message","name":"kitchen speaker"}',
        },
    )

    assert candidate.wildcard_infos == ((1, "local_free_text"),)
    assert candidate.concrete_surface_slot_tokens == frozenset({"kitchen", "speaker"})


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
    """Keep opposing structural ties ordered deterministically but fail closed."""
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
    assert accepted_candidate(ranked) is None


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
        candidate=Candidate(
            text="get kitchen light state", intent_name="HassGetState", language="en"
        ),
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
            text="get kitchen temperature",
            intent_name="HassClimateGetTemperature",
            language="en",
            metadata={"slots": '{"name":"kitchen light"}'},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.81,
            char_ngram_score=0.81,
            bm25_score=0.81,
            intent_score=0.81,
            final_score=0.81,
        ),
    )

    decision = evaluate_confidence_gates((top, competitor))

    assert decision.accepted_candidate is top
    assert decision.margin_policy == "high_confidence_relaxation"


def test_accepted_candidate_does_not_relax_margin_below_relaxed_threshold() -> None:
    """Reject high-confidence candidates below the relaxed margin threshold."""
    top = RankedCandidate(
        candidate=Candidate(
            text="turn on kitchen light",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": '{"name":"kitchen light"}'},
        ),
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
        candidate=Candidate(
            text="water heater",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": '{"name":"water heater"}'},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.65,
            char_ngram_score=0.65,
            bm25_score=0.65,
            intent_score=0.0,
            final_score=0.65,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="water heater off",
            intent_name="HassTurnOff",
            language="en",
            metadata={"slots": '{"name":"water heater"}'},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.59,
            char_ngram_score=0.59,
            bm25_score=0.59,
            intent_score=0.4,
            final_score=0.59,
        ),
    )

    decision = evaluate_confidence_gates((top, competitor))

    assert not decision.accepted
    assert decision.rejection_reason is FallbackReason.LOW_CONFIDENCE
    assert decision.margin_policy == "weak_zero_intent_evidence"


def test_accepted_candidate_requires_margin_between_distinct_same_intent_targets() -> None:
    """Do not treat different target payloads as duplicate intent variants."""
    top = RankedCandidate(
        candidate=Candidate(
            text="turn on kitchen light",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": '{"name":"kitchen light"}'},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.90,
            char_ngram_score=0.90,
            bm25_score=0.90,
            intent_score=0.90,
            final_score=0.90,
        ),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="turn on kitchen fan",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": '{"name":"kitchen fan"}'},
        ),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.89,
            char_ngram_score=0.89,
            bm25_score=0.89,
            intent_score=0.89,
            final_score=0.89,
        ),
    )

    decision = evaluate_confidence_gates((top, competitor))

    assert not decision.accepted
    assert decision.rejection_reason is FallbackReason.LOW_MARGIN
    assert decision.margin_policy == "target_base_margin"
    assert decision.target_incompatible_competition
    assert not decision.relaxation_used


def test_accepted_candidate_ignores_same_intent_variants_with_identical_slots() -> None:
    """Keep wording variants with the same executable payload out of the margin gate."""
    top = RankedCandidate(
        candidate=Candidate(
            text="turn on the kitchen light",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": '{"name":"kitchen light"}'},
        ),
        scores=ScoreBreakdown(0.90, 0.90, 0.90, 0.90, 0.90),
    )
    variant = RankedCandidate(
        candidate=Candidate(
            text="switch on the kitchen light",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": '{"name":"kitchen light"}'},
        ),
        scores=ScoreBreakdown(0.89, 0.89, 0.89, 0.89, 0.89),
    )

    decision = evaluate_confidence_gates((top, variant))

    assert decision.accepted_candidate is top
    assert decision.meaningful_competitor is None
    assert decision.margin_policy == "no_competitor"


def test_accepted_candidate_ignores_overlapping_same_target_decompositions() -> None:
    """Treat overlapping area/domain and name payloads as one physical target."""
    top = RankedCandidate(
        candidate=Candidate(
            text="turn on the living room light",
            intent_name="HassTurnOn",
            language="en",
            metadata={
                "slots": '{"area":"living room","domain":"light"}',
                "static_slots": "domain",
            },
        ),
        scores=ScoreBreakdown(0.90, 0.90, 0.90, 0.90, 0.90),
    )
    variant = RankedCandidate(
        candidate=Candidate(
            text="switch on the living room light",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": '{"name":"living room light"}'},
        ),
        scores=ScoreBreakdown(0.89, 0.89, 0.89, 0.89, 0.89),
    )

    decision = evaluate_confidence_gates((top, variant))

    assert decision.accepted_candidate is top
    assert decision.meaningful_competitor is None
    assert decision.margin_policy == "no_competitor"


def test_accepted_candidate_uses_literal_evidence_from_equivalent_action_variant() -> None:
    """Use an explicit grammar action from a same-target generated variant."""
    top = RankedCandidate(
        candidate=Candidate(
            text="the living room light on",
            intent_name="HassTurnOn",
            language="en",
            metadata={
                "slots": '{"area":"living room","domain":"light"}',
                "static_slots": "domain",
                "literal_text": "light on",
            },
        ),
        scores=ScoreBreakdown(0.67, 0.67, 0.67, 0.67, 0.67),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="the living room light off",
            intent_name="HassTurnOff",
            language="en",
            metadata={
                "slots": '{"area":"living room","domain":"light"}',
                "static_slots": "domain",
                "literal_text": "light off",
            },
        ),
        scores=ScoreBreakdown(0.66, 0.66, 0.66, 0.66, 0.66),
    )
    action_variant = RankedCandidate(
        candidate=Candidate(
            text="illuminate the living room",
            intent_name="HassTurnOn",
            language="en",
            metadata={
                "slots": '{"area":"living room","domain":"light"}',
                "static_slots": "domain",
                "literal_text": "illuminate",
            },
        ),
        scores=ScoreBreakdown(0.65, 0.65, 0.65, 0.80, 0.65),
    )

    decision = evaluate_confidence_gates(
        (top, competitor, action_variant),
        query="illuminate the living room light",
        language="en",
    )

    assert decision.accepted_candidate is top
    assert decision.meaningful_competitor is competitor
    assert decision.margin_policy == "supported_literal_action_evidence"
    assert decision.relaxation_used


def test_accepted_candidate_does_not_relax_shared_target_literal_evidence() -> None:
    """A shared target noun is not enough to choose between opposing actions."""
    top = RankedCandidate(
        candidate=Candidate(
            text="living room light on",
            intent_name="HassTurnOn",
            language="en",
            metadata={
                "slots": '{"name":"living room light"}',
                "literal_text": "light",
            },
        ),
        scores=ScoreBreakdown(0.67, 0.67, 0.67, 0.67, 0.67),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="living room light off",
            intent_name="HassTurnOff",
            language="en",
            metadata={
                "slots": '{"name":"living room light"}',
                "literal_text": "light",
            },
        ),
        scores=ScoreBreakdown(0.66, 0.66, 0.66, 0.66, 0.66),
    )

    decision = evaluate_confidence_gates(
        (top, competitor),
        query="living room light",
        language="en",
    )

    assert not decision.accepted
    assert decision.margin_policy == "opposing_action_base_margin"


def test_accepted_candidate_does_not_relax_fuzzy_unique_action_evidence() -> None:
    """A merely similar unique literal cannot override action ambiguity."""
    top = RankedCandidate(
        candidate=Candidate(
            text="restart the kitchen fan",
            intent_name="HassTurnOn",
            language="en",
            metadata={
                "slots": '{"name":"kitchen fan"}',
                "literal_text": "restart",
            },
        ),
        scores=ScoreBreakdown(0.67, 0.67, 0.67, 0.67, 0.67),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="stop the kitchen fan",
            intent_name="HassTurnOff",
            language="en",
            metadata={
                "slots": '{"name":"kitchen fan"}',
                "literal_text": "stop",
            },
        ),
        scores=ScoreBreakdown(0.66, 0.66, 0.66, 0.66, 0.66),
    )

    decision = evaluate_confidence_gates(
        (top, competitor),
        query="start the kitchen fan",
        language="en",
    )

    assert not decision.accepted
    assert decision.margin_policy == "opposing_action_base_margin"


def test_accepted_candidate_uses_query_supported_same_intent_target() -> None:
    """Do not manufacture ambiguity when only the top target is spoken."""
    top = RankedCandidate(
        candidate=Candidate(
            text="turn off kitchen light",
            intent_name="HassTurnOff",
            language="en",
            metadata={"slots": '{"name":"kitchen light"}'},
        ),
        scores=ScoreBreakdown(0.70, 0.70, 0.70, 0.70, 0.70),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="turn off kitchen fan",
            intent_name="HassTurnOff",
            language="en",
            metadata={"slots": '{"name":"kitchen fan"}'},
        ),
        scores=ScoreBreakdown(0.69, 0.69, 0.69, 0.69, 0.69),
    )

    decision = evaluate_confidence_gates(
        (top, competitor),
        query="turn off the kitchen light",
        language="en",
    )

    assert decision.accepted_candidate is top
    assert decision.meaningful_competitor is None
    assert decision.margin_policy == "no_competitor"


def test_accepted_candidate_uses_diacritic_tolerant_target_evidence() -> None:
    """Use normalized spoken target evidence across a diacritic transcription error."""
    top = RankedCandidate(
        candidate=Candidate(
            text="tắt đèn phòng khách",
            intent_name="HassTurnOff",
            language="vi",
            metadata={"slots": '{"name":"đèn phòng khách"}'},
        ),
        scores=ScoreBreakdown(0.70, 0.70, 0.70, 0.70, 0.70),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="tắt rèm phòng khách",
            intent_name="HassTurnOff",
            language="vi",
            metadata={"slots": '{"name":"rèm phòng khách"}'},
        ),
        scores=ScoreBreakdown(0.69, 0.69, 0.69, 0.69, 0.69),
    )

    decision = evaluate_confidence_gates(
        (top, competitor),
        query="tắt đền phòng khách",
        language="vi",
    )

    assert decision.accepted_candidate is top
    assert decision.meaningful_competitor is None


def test_accepted_candidate_uses_bounded_morphological_target_evidence() -> None:
    """Recognize a long shared target stem without language-specific stemming."""
    top = RankedCandidate(
        candidate=Candidate(
            text="turn off the bedroom ventilator",
            intent_name="HassTurnOff",
            language="en",
            metadata={"slots": '{"name":"bedroom ventilator"}'},
        ),
        scores=ScoreBreakdown(0.75, 0.75, 0.75, 0.75, 0.75),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="turn off the bedroom light",
            intent_name="HassTurnOff",
            language="en",
            metadata={"slots": '{"name":"bedroom light"}'},
        ),
        scores=ScoreBreakdown(0.70, 0.70, 0.70, 0.70, 0.70),
    )

    decision = evaluate_confidence_gates(
        (top, competitor),
        query="stop the bedroom ventilation",
        language="en",
    )

    assert decision.accepted_candidate is top
    assert decision.meaningful_competitor is None


def test_accepted_candidate_uses_contextual_short_target_typo() -> None:
    """Use a short typo only when the rest of the target phrase is anchored."""
    top = RankedCandidate(
        candidate=Candidate(
            text="tắt quạt phòng tắm",
            intent_name="HassTurnOff",
            language="vi",
            metadata={"slots": '{"name":"quạt phòng tắm"}'},
        ),
        scores=ScoreBreakdown(0.70, 0.70, 0.70, 0.70, 0.70),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="tắt quạt phòng làm việc",
            intent_name="HassTurnOff",
            language="vi",
            metadata={"slots": '{"name":"quạt phòng làm việc"}'},
        ),
        scores=ScoreBreakdown(0.69, 0.69, 0.69, 0.69, 0.69),
    )

    decision = evaluate_confidence_gates(
        (top, competitor),
        query="tắt quạt phòg tắh",
        language="vi",
    )

    assert decision.accepted_candidate is top
    assert decision.meaningful_competitor is None


def test_accepted_candidate_rejects_unspoken_same_intent_target_choice() -> None:
    """Never choose between distinct targets when neither one is in the query."""
    top = RankedCandidate(
        candidate=Candidate(
            text="turn on kitchen light",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": '{"name":"kitchen light"}'},
        ),
        scores=ScoreBreakdown(0.90, 0.90, 0.90, 0.90, 0.90),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="turn on kitchen fan",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": '{"name":"kitchen fan"}'},
        ),
        scores=ScoreBreakdown(0.70, 0.70, 0.70, 0.70, 0.70),
    )

    decision = evaluate_confidence_gates(
        (top, competitor),
        query="turn on something in the kitchen",
        language="en",
    )

    assert not decision.accepted
    assert decision.rejection_reason is FallbackReason.LOW_CONFIDENCE
    assert decision.margin_policy == "target_without_query_support"
    assert decision.target_incompatible_competition


def test_accepted_candidate_promotes_query_supported_lower_target() -> None:
    """Drop a higher same-intent payload when only the lower target is spoken."""
    unsupported = RankedCandidate(
        candidate=Candidate(
            text="set office light to 20",
            intent_name="HassLightSet",
            language="en",
            metadata={"slots": '{"name":"office light","brightness":20}'},
        ),
        scores=ScoreBreakdown(0.90, 0.90, 0.90, 0.90, 0.90),
    )
    supported = RankedCandidate(
        candidate=Candidate(
            text="set kitchen light to 50",
            intent_name="HassLightSet",
            language="en",
            metadata={"slots": '{"name":"kitchen light","brightness":50}'},
        ),
        scores=ScoreBreakdown(0.85, 0.85, 0.85, 0.85, 0.85),
    )
    other_intent = RankedCandidate(
        candidate=Candidate(
            text="turn off kitchen light",
            intent_name="HassTurnOff",
            language="en",
            metadata={"slots": '{"name":"kitchen light"}'},
        ),
        scores=ScoreBreakdown(0.70, 0.70, 0.70, 0.70, 0.70),
    )

    decision = evaluate_confidence_gates(
        (unsupported, supported, other_intent),
        query="set kitchen light to 50",
        language="en",
    )

    assert decision.accepted_candidate is supported
    assert decision.top_candidate is supported


def test_accepted_candidate_promotes_nonliteral_target_from_generic_leader() -> None:
    """Use a spoken concrete target when a generic same-intent form ranks first."""
    generic = RankedCandidate(
        candidate=Candidate(
            text="turn off the fans in the room",
            intent_name="HassTurnOff",
            language="en",
            metadata={
                "literal_text": "turn off the fans in the room",
                "slots": '{"domain":"fan"}',
                "static_slots": "domain",
            },
        ),
        scores=ScoreBreakdown(0.75, 0.75, 0.75, 0.75, 0.75),
    )
    supported = RankedCandidate(
        candidate=Candidate(
            text="turn off the bathroom fan",
            intent_name="HassTurnOff",
            language="en",
            metadata={"slots": '{"name":"bathroom fan"}'},
        ),
        scores=ScoreBreakdown(0.70, 0.70, 0.70, 0.70, 0.70),
    )

    decision = evaluate_confidence_gates(
        (generic, supported),
        query="please turn off the bathroom fan",
        language="en",
    )

    assert decision.accepted_candidate is supported
    assert decision.top_candidate is supported


def test_accepted_candidate_does_not_promote_literal_only_target_evidence() -> None:
    """Do not reinterpret an intent grammar word as a concrete target."""
    generic = RankedCandidate(
        candidate=Candidate(
            text="turn on the light",
            intent_name="HassTurnOn",
            language="en",
            metadata={
                "literal_text": "turn on the light",
                "slots": '{"domain":"light"}',
                "static_slots": "domain",
            },
        ),
        scores=ScoreBreakdown(0.90, 0.90, 0.90, 0.90, 0.90),
    )
    concrete = RankedCandidate(
        candidate=Candidate(
            text="turn on light",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": '{"name":"light"}'},
        ),
        scores=ScoreBreakdown(0.70, 0.70, 0.70, 0.70, 0.70),
    )

    decision = evaluate_confidence_gates(
        (generic, concrete),
        query="turn on the light",
        language="en",
    )

    assert decision.accepted_candidate is generic
    assert decision.top_candidate is generic


def test_accepted_candidate_compares_localized_raw_slot_payloads() -> None:
    """Treat mapped and entity forms of one localized target as equivalent."""
    top = RankedCandidate(
        candidate=Candidate(
            text="mở cái cửa sổ phòng ngủ",
            intent_name="HassTurnOn",
            language="vi",
            metadata={
                "slots": '{"name":"cửa sổ phòng ngủ"}',
                "slots_raw": '{"name":"cửa sổ phòng ngủ"}',
            },
        ),
        scores=ScoreBreakdown(0.70, 0.70, 0.70, 0.70, 0.70),
    )
    variant = RankedCandidate(
        candidate=Candidate(
            text="mở cửa sổ phòng ngủ",
            intent_name="HassTurnOn",
            language="vi",
            metadata={
                "slots": '{"domain":"cover","device_class":"window","area":"phòng ngủ"}',
                "slots_raw": ('{"domain":"cover","device_class":"cửa sổ","area":"phòng ngủ"}'),
                "static_slots": "domain",
            },
        ),
        scores=ScoreBreakdown(0.69, 0.69, 0.69, 0.69, 0.69),
    )

    decision = evaluate_confidence_gates(
        (top, variant),
        query="mở hộ cửa sổ phòng ngủ",
        language="vi",
    )

    assert decision.accepted_candidate is top
    assert decision.meaningful_competitor is None
    assert decision.margin_policy == "no_competitor"


def test_accepted_candidate_rejects_wildcard_absorbing_concrete_target() -> None:
    """Do not let exact wildcard rehydration hide a supported concrete target."""
    wildcard = RankedCandidate(
        candidate=Candidate(
            text="play search_query",
            intent_name="HassMediaSearchAndPlay",
            language="en",
            metadata={
                "sentence_template": "play {search_query}",
                "wildcard_slots": "search_query",
                "slots": '{"search_query":"search_query"}',
                "literal_text": "play",
            },
        ),
        scores=ScoreBreakdown(1.0, 1.0, 1.0, 1.0, 0.74),
    )
    concrete = RankedCandidate(
        candidate=Candidate(
            text="start the kitchen vacuum",
            intent_name="HassVacuumStart",
            language="en",
            metadata={"slots": '{"name":"kitchen vacuum"}'},
        ),
        scores=ScoreBreakdown(0.65, 0.65, 0.65, 0.65, 0.62),
    )

    decision = evaluate_confidence_gates(
        (wildcard, concrete),
        query="play the kitchen vacuum",
        language="en",
    )

    assert not decision.accepted
    assert decision.rejection_reason is FallbackReason.LOW_CONFIDENCE
    assert decision.margin_policy == "wildcard_absorbed_concrete_target"


def test_accepted_candidate_rejects_incomplete_read_only_literal() -> None:
    """Require the complete query-form literal for a fuzzy read-only intent."""
    top = RankedCandidate(
        candidate=Candidate(
            text="how warm is it here",
            intent_name="HassClimateGetTemperature",
            language="en",
            metadata={
                "literal_text": "how warm is it|how warm is it here",
                "slots": "{}",
            },
        ),
        scores=ScoreBreakdown(0.80, 0.50, 0.90, 0.75, 0.74),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="how warm is it in the kitchen",
            intent_name="HassClimateGetTemperature",
            language="en",
            metadata={
                "literal_text": "how warm is it in",
                "slots": '{"area":"kitchen"}',
            },
        ),
        scores=ScoreBreakdown(0.65, 0.45, 0.60, 0.65, 0.60),
    )

    decision = evaluate_confidence_gates(
        (top, competitor),
        query="it is too warm here",
        language="en",
    )

    assert not decision.accepted
    assert decision.rejection_reason is FallbackReason.LOW_CONFIDENCE
    assert decision.margin_policy == "read_only_incomplete_literal"


def test_accepted_candidate_allows_omission_only_read_only_query() -> None:
    """Preserve a read-only match when its candidate covers every query token."""
    top = RankedCandidate(
        candidate=Candidate(
            text="what is the weather",
            intent_name="HassGetWeather",
            language="en",
            metadata={"literal_text": "what is the weather", "slots": "{}"},
        ),
        scores=ScoreBreakdown(0.80, 0.70, 0.80, 0.75, 0.74),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="what is the weather in the kitchen",
            intent_name="HassGetWeather",
            language="en",
            metadata={
                "literal_text": "what is the weather in",
                "slots": '{"area":"kitchen"}',
            },
        ),
        scores=ScoreBreakdown(0.65, 0.60, 0.65, 0.65, 0.60),
    )

    decision = evaluate_confidence_gates(
        (top, competitor),
        query="what the weather",
        language="en",
    )

    assert decision.accepted_candidate is top


def test_accepted_candidate_uses_read_only_literal_from_generated_variant() -> None:
    """Preserve intent evidence when a registry expansion has extra target wording."""
    top = RankedCandidate(
        candidate=Candidate(
            text="what is the weather for home weather",
            intent_name="HassGetWeather",
            language="en",
            metadata={
                "literal_text": "what is the weather for",
                "slots": '{"name":"home weather"}',
            },
        ),
        scores=ScoreBreakdown(0.80, 0.70, 0.80, 0.75, 0.74),
    )
    general_variant = RankedCandidate(
        candidate=Candidate(
            text="what is the weather",
            intent_name="HassGetWeather",
            language="en",
            metadata={"literal_text": "what is the weather", "slots": "{}"},
        ),
        scores=ScoreBreakdown(0.65, 0.60, 0.65, 0.65, 0.60),
    )
    other_intent = RankedCandidate(
        candidate=Candidate(
            text="what is the date",
            intent_name="HassGetCurrentDate",
            language="en",
            metadata={"literal_text": "what is the date", "slots": "{}"},
        ),
        scores=ScoreBreakdown(0.55, 0.55, 0.55, 0.55, 0.55),
    )

    decision = evaluate_confidence_gates(
        (top, general_variant, other_intent),
        query="what is the weather today",
        language="en",
    )

    assert decision.accepted_candidate is top
    assert decision.margin_policy == "target_base_margin"


def test_accepted_candidate_rejects_incomplete_state_change_literal() -> None:
    """Do not execute a state change from partial action evidence."""
    top = RankedCandidate(
        candidate=Candidate(
            text="clean the living room",
            intent_name="HassVacuumCleanArea",
            language="en",
            metadata={
                "literal_text": "clean|clean the",
                "slots": '{"area":"living room"}',
            },
        ),
        scores=ScoreBreakdown(0.75, 0.55, 0.90, 0.65, 0.65),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="how dark is the living room",
            intent_name="HassGetState",
            language="en",
            metadata={
                "literal_text": "how dark is",
                "slots": '{"area":"living room"}',
            },
        ),
        scores=ScoreBreakdown(0.65, 0.45, 0.60, 0.50, 0.54),
    )

    decision = evaluate_confidence_gates(
        (top, competitor),
        query="make it dark in the living room",
        language="en",
    )

    assert not decision.accepted
    assert decision.rejection_reason is FallbackReason.LOW_CONFIDENCE
    assert decision.margin_policy == "state_change_incomplete_literal"


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


def test_accepted_candidate_relaxes_action_incompatible_intent_evidence() -> None:
    """Keep calibrated intent evidence and report the action incompatibility."""
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
            text="get kitchen light state",
            intent_name="HassGetState",
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

    decision = evaluate_confidence_gates((top, competitor))

    assert decision.accepted_candidate is top
    assert decision.margin_policy == "safe_intent_evidence_relaxation"
    assert decision.action_incompatible_competition
    assert decision.relaxation_used


def test_accepted_candidate_rejects_action_incompatible_weaker_intent_evidence() -> None:
    """Do not let target similarity override stronger action evidence."""
    top = RankedCandidate(
        candidate=Candidate(
            text="is the bedroom window opening",
            intent_name="HassGetState",
            language="en",
            metadata={"slots": '{"name":"bedroom window","state":"opening"}'},
        ),
        scores=ScoreBreakdown(0.80, 0.80, 0.80, 0.55, 0.75),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="open the bedroom window",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": '{"name":"bedroom window"}'},
        ),
        scores=ScoreBreakdown(0.65, 0.65, 0.65, 0.70, 0.62),
    )

    decision = evaluate_confidence_gates((top, competitor))

    assert not decision.accepted
    assert decision.rejection_reason is FallbackReason.LOW_CONFIDENCE
    assert decision.margin_policy == "action_incompatible_weaker_intent_evidence"
    assert decision.action_incompatible_competition


def test_accepted_candidate_relaxes_safe_non_action_intent_evidence() -> None:
    """Retain the calibrated relaxation when neither close intent changes state."""
    top = RankedCandidate(
        candidate=Candidate(
            text="query kitchen light state",
            intent_name="HassGetState",
            language="en",
            metadata={"slots": orjson.dumps({"area": "kitchen"}).decode()},
        ),
        scores=ScoreBreakdown(0.67, 0.67, 0.67, 0.70, 0.67),
    )
    competitor = RankedCandidate(
        candidate=Candidate(
            text="query kitchen weather",
            intent_name="HassGetWeather",
            language="en",
            metadata={"slots": orjson.dumps({"area": "kitchen"}).decode()},
        ),
        scores=ScoreBreakdown(0.65, 0.65, 0.65, 0.58, 0.65),
    )

    decision = evaluate_confidence_gates((top, competitor))

    assert decision.accepted_candidate is top
    assert decision.margin_policy == "safe_intent_evidence_relaxation"
    assert not decision.action_incompatible_competition


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


def test_exact_action_reports_incompatible_read_only_competitor() -> None:
    """Accept exact text while retaining action-incompatibility diagnostics."""
    top = RankedCandidate(
        Candidate(text="turn on kitchen light", intent_name="HassTurnOn"),
        ScoreBreakdown(1.0, 1.0, 1.0, 0.9, 0.90),
    )
    competitor = RankedCandidate(
        Candidate(text="query kitchen light state", intent_name="HassGetState"),
        ScoreBreakdown(0.99, 0.99, 0.99, 0.9, 0.89),
    )

    decision = evaluate_confidence_gates(
        (top, competitor),
        min_confidence=DEFAULT_MIN_CONFIDENCE / 5.0,
        min_margin=DEFAULT_MIN_MARGIN,
    )

    assert decision.accepted_candidate is top
    assert decision.margin_policy == "exact_lexical"
    assert decision.action_incompatible_competition
    assert decision.relaxation_used


def test_exact_read_only_intent_keeps_safe_lexical_relaxation() -> None:
    """Keep exact-match relaxation when neither competing intent changes state."""
    top = RankedCandidate(
        Candidate(text="query kitchen light state", intent_name="HassGetState"),
        ScoreBreakdown(1.0, 1.0, 1.0, 0.9, 0.90),
    )
    competitor = RankedCandidate(
        Candidate(text="query kitchen weather", intent_name="HassGetWeather"),
        ScoreBreakdown(0.99, 0.99, 0.99, 0.9, 0.89),
    )

    decision = evaluate_confidence_gates((top, competitor), min_margin=0.05)

    assert decision.accepted_candidate is top
    assert decision.margin_policy == "exact_lexical"


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


def test_query_slot_tokens_from_candidates_uses_bounded_fuzzy_slot_matches() -> None:
    """Map bounded edits, transpositions, and prefixes within candidate slots."""
    candidate_slot_tokens = tuple(
        frozenset({token})
        for token in ("badezimmerlüfter", "window", "fan", "door", "phong", "télévision")
    )

    assert _query_slot_tokens_from_candidates(
        frozenset({"badzimmerlüfter", "windw", "fon", "doar", "phogn", "télé"}),
        (0, 1, 2, 3, 4, 5),
        candidate_slot_tokens,
    ) == frozenset({"badezimmerlüfter", "door", "phong", "télévision", "window"})


def test_query_slot_tokens_from_candidates_rejects_distant_fuzzy_slot_matches() -> None:
    """Reject slot-token typos outside the bounded one-edit guard."""
    assert (
        _query_slot_tokens_from_candidates(
            frozenset(
                {
                    "badzximmerlüfter",
                    "badezimmerlüfterfoobar",
                    "xadezimmerlüfter",
                    "bad",
                }
            ),
            (0,),
            (frozenset({"badezimmerlüfter"}),),
        )
        == frozenset()
    )


def test_query_slot_tokens_from_candidates_ignores_unselected_candidates() -> None:
    """Collect active fuzzy slot tokens from prefiltered candidates only."""
    query_tokens = frozenset({"badzimmerlüfter", "windw", "unrelated"})
    candidate_slot_tokens = (
        frozenset({"badezimmerlüfter"}),
        frozenset({"window"}),
        frozenset({"unselected"}),
    )

    assert _query_slot_tokens_from_candidates(
        query_tokens,
        (0, 1),
        candidate_slot_tokens,
    ) == frozenset({"badezimmerlüfter", "window"})


def test_query_slot_tokens_do_not_reuse_literal_as_slot_prefix() -> None:
    """Keep a known action literal from claiming a longer slot-value prefix."""
    assert _query_slot_tokens_from_candidates(
        frozenset({"shut", "window"}),
        (0,),
        (frozenset({"shutter", "window"}),),
        frozenset({"shut"}),
    ) == frozenset({"window"})


def test_slot_token_query_cache_evicts_only_the_oldest_entry() -> None:
    """Keep the bounded FIFO cache warm when a ninth query is inserted."""
    index = SlotTokenIndex.from_candidate_tokens((frozenset({"kitchen"}),))
    query_tokens = [frozenset({f"query{position}"}) for position in range(9)]
    for tokens in query_tokens[:8]:
        ranking._slot_index_query_matches(tokens, index)

    first_key = (query_tokens[0], None)
    retained_keys = {(tokens, None) for tokens in query_tokens[1:8]}
    assert ranking._slot_index_query_matches(query_tokens[0], index) == frozenset()

    ranking._slot_index_query_matches(query_tokens[8], index)

    assert first_key not in index.query_match_cache
    assert retained_keys <= index.query_match_cache.keys()
    assert len(index.query_match_cache) == 8


def test_slot_prefilter_rescue_requires_complete_target_slot_match() -> None:
    """Rescue a typoed target without admitting a shared-area competitor."""
    candidate_slot_tokens = (
        frozenset({"bedroom", "light"}),
        frozenset({"bedroom", "window"}),
        frozenset({"bathroom", "window"}),
    )
    slot_token_index = SlotTokenIndex.from_candidate_tokens(candidate_slot_tokens)

    rescued = _top_additional_slot_indices(
        frozenset({"turn", "off", "bedrom", "windw"}),
        candidate_slot_tokens,
        slot_token_index,
        top_indices=(0,),
        prefilter_keys=(-0.9, -0.1, -0.8),
    )

    assert rescued == [1]


def test_slot_prefilter_rescue_prioritizes_fully_anchored_specific_targets() -> None:
    """Do not let area-only variants exhaust the bounded target-rescue budget."""
    candidate_slot_tokens = (
        frozenset({"bedroom"}),
        frozenset({"bedroom", "window"}),
        frozenset({"bedroom"}),
    )
    slot_token_index = SlotTokenIndex.from_candidate_tokens(candidate_slot_tokens)

    rescued = _top_additional_slot_indices(
        frozenset({"turn", "off", "bedrom", "windw"}),
        candidate_slot_tokens,
        slot_token_index,
        top_indices=(0,),
        prefilter_keys=(-0.9, -0.1, -0.8),
        limit=1,
    )

    assert rescued == [1]


def test_index_rank_rescues_typoed_entity_family_outside_general_prefilter() -> None:
    """Consider a fully anchored entity even when generic retrieval omitted it."""
    distractors = [
        Candidate(
            text=f"turn off the bedroom light option {chr(97 + first)}{chr(97 + second)}",
            intent_name="HassTurnOff",
            language="en",
            metadata={
                "literal_text": "turn off option",
                "slots": orjson.dumps({"name": "bedroom light"}).decode(),
            },
        )
        for first in range(23)
        for second in range(23)
    ]
    bedroom_window = Candidate(
        text="turn off the bedroom window",
        intent_name="HassTurnOff",
        language="en",
        metadata={
            "literal_text": "turn off",
            "slots": orjson.dumps({"name": "bedroom window"}).decode(),
        },
    )
    index = build_index("en", [*distractors, bedroom_window])

    ranked = index.rank("plz turn off the bedrom windw")

    assert ranked[0].candidate == bedroom_window


def test_slot_prefilter_rescue_keeps_expensive_scoring_bounded(monkeypatch) -> None:
    """Cap target rescue work independently of the general prefilter size."""
    candidates = [
        Candidate(
            text=f"turn off bedroom window variant {index}",
            intent_name="HassTurnOff",
            language="en",
            metadata={"slots": orjson.dumps({"name": "bedroom window"}).decode()},
        )
        for index in range(DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES + 50)
    ]
    rapidfuzz_counter = _RapidFuzzSimilarityCounter()
    monkeypatch.setattr(ranking, "rapidfuzz_similarity_normalized", rapidfuzz_counter)

    build_index("en", candidates).rank("turn off bedrom windw", max_candidates=5)

    assert rapidfuzz_counter.calls <= DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES + 32


def test_rank_candidates_uses_truncated_registry_slot_evidence() -> None:
    """Prefer a truncated registry name over unrelated same-intent slots."""
    television = Candidate(
        text="allume la télévision",
        intent_name="HassTurnOn",
        language="fr",
        metadata={
            "literal_text": "allume",
            "slots": orjson.dumps({"name": "télévision"}).decode(),
        },
        slot_values=("télévision",),
    )
    unrelated_light = Candidate(
        text="allume la salon",
        intent_name="HassTurnOn",
        language="fr",
        metadata={
            "literal_text": "allume",
            "slots": orjson.dumps({"domain": "light", "area": "salon"}).decode(),
        },
        slot_values=("light", "salon"),
    )

    ranked = build_index("fr", [unrelated_light, television]).rank("allume la télé")

    assert ranked[0].candidate is television


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


@pytest.mark.parametrize(
    ("char_scores", "raw_bm25_scores"),
    [
        ([0.0, 0.5, 0.0, 0.2, 0.5, 0.0], [0.0, 0.1, 0.0, 0.6, 0.1, 0.0]),
        ([0.0, 0.0, 0.4, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0, 0.0], [0.4, 0.0, 0.2, 0.0]),
    ],
)
def test_sparse_bm25_prefilter_keys_match_dense(
    char_scores: list[float],
    raw_bm25_scores: list[float],
) -> None:
    """Preserve dense keys and selection while storing only touched BM25 scores."""
    max_raw_score = max(raw_bm25_scores, default=0.0)
    inv_max = 1.0 / max_raw_score if max_raw_score > 0.0 else 0.0
    dense_bm25_scores = [score * inv_max for score in raw_bm25_scores]
    sparse_bm25_scores = {
        index: score for index, score in enumerate(raw_bm25_scores) if score > 0.0
    }
    dense_keys = _rank_prefilter_keys(char_scores, dense_bm25_scores)
    hybrid_keys = _rank_prefilter_keys_with_sparse_bm25(
        char_scores,
        sparse_bm25_scores,
        inv_max,
    )

    assert hybrid_keys == dense_keys
    assert _top_prefilter_indices(hybrid_keys, len(hybrid_keys)) == _top_prefilter_indices(
        dense_keys,
        len(dense_keys),
    )


@pytest.mark.parametrize(
    ("candidate_grams", "query_grams", "raw_bm25_scores"),
    [
        (
            (frozenset({"abc", "bcd"}), frozenset({"bcd"}), frozenset()),
            frozenset({"abc", "bcd"}),
            [0.0, 0.1, 0.0],
        ),
        (
            (frozenset({"abc"}), frozenset({"xyz"})),
            frozenset(),
            [0.4, 0.0],
        ),
        ((), frozenset({"abc"}), []),
    ],
)
def test_intersection_prefilter_keys_match_dense_scores(
    candidate_grams: tuple[frozenset[str], ...],
    query_grams: frozenset[str],
    raw_bm25_scores: list[float],
) -> None:
    """Fuse character scoring and prefilter keys without changing their values."""
    char_index = CharNGramIndex.from_grams(candidate_grams)
    dense_char_scores = char_index.score(query_grams)
    max_raw_score = max(raw_bm25_scores, default=0.0)
    inv_max = 1.0 / max_raw_score if max_raw_score > 0.0 else 0.0
    dense_bm25_scores = [score * inv_max for score in raw_bm25_scores]
    sparse_bm25_scores = {
        index: score for index, score in enumerate(raw_bm25_scores) if score > 0.0
    }

    intersections, touched = char_index.intersections_with_touched(query_grams)
    sparse_keys = _sparse_prefilter_keys_from_intersections(
        intersections,
        touched,
        len(query_grams),
        char_index.gram_counts,
        sparse_bm25_scores,
        inv_max,
    )
    fused_keys = _SparsePrefilterKeys(sparse_keys, len(candidate_grams))

    dense_keys = _rank_prefilter_keys(dense_char_scores, dense_bm25_scores)
    assert [fused_keys[index] for index in range(len(fused_keys))] == dense_keys
    assert _top_sparse_prefilter_indices(
        sparse_keys, len(candidate_grams), len(candidate_grams)
    ) == _top_prefilter_indices(dense_keys, len(dense_keys))


def test_sparse_prefilter_keys_obey_dense_sequence_bounds() -> None:
    """Support normal negative indices and terminate sequence iteration."""
    sparse_keys = _SparsePrefilterKeys({0: -0.5, 2: -0.2}, 3)

    assert sparse_keys[-1] == -0.2
    assert sparse_keys[-2] == 0.0
    assert sparse_keys[1:] == [0.0, -0.2]
    assert list(sparse_keys) == [-0.5, 0.0, -0.2]
    with pytest.raises(IndexError):
        _ = sparse_keys[3]
    with pytest.raises(IndexError):
        _ = sparse_keys[-4]


def test_additional_wildcard_prefilter_is_bounded_by_literal_relevance() -> None:
    """Bound wildcard rescues while preferring stronger literal-token evidence."""
    query_tokens = frozenset({"add", "milk", "shopping", "list"})
    variants: dict[int, tuple[WildcardVariantAnalysis, ...]] = {
        0: (
            WildcardVariantAnalysis(
                literal_tokens=frozenset({"add"}),
                required_match_count=1,
            ),
        ),
        1: (
            WildcardVariantAnalysis(
                literal_tokens=frozenset({"add", "shopping", "list"}),
                required_match_count=3,
            ),
        ),
        2: (
            WildcardVariantAnalysis(
                literal_tokens=frozenset({"shopping", "list"}),
                required_match_count=2,
            ),
        ),
        3: (
            WildcardVariantAnalysis(
                literal_tokens=frozenset({"add", "list"}),
                required_match_count=2,
            ),
        ),
    }
    prefilter_keys = [-0.9, -0.2, -0.8, -0.7]

    selected = _top_additional_wildcard_indices(
        set(variants),
        query_tokens,
        variants,
        prefilter_keys,
        limit=2,
    )

    assert selected == [1, 2]


def test_additional_wildcard_prefilter_preserves_all_candidates_within_budget() -> None:
    """Preserve deterministic index order when wildcard rescues fit the budget."""
    assert _top_additional_wildcard_indices(
        {3, 1},
        frozenset({"add"}),
        None,
        [0.0] * 4,
        limit=2,
    ) == [1, 3]


def test_grouped_wildcard_prefilter_evaluates_shared_variants_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expand indexes only after one coverage check for a shared wildcard group."""
    variants = (
        WildcardVariantAnalysis(
            literal_tokens=frozenset({"add", "to", "list"}),
            required_match_count=3,
        ),
    )
    calls = 0
    original = ranking._wildcard_variants_match

    def counted(
        values: tuple[WildcardVariantAnalysis, ...],
        query_tokens: frozenset[str],
    ) -> bool:
        nonlocal calls
        calls += 1
        return original(values, query_tokens)

    monkeypatch.setattr(ranking, "_wildcard_variants_match", counted)

    passed = ranking._prefilter_wildcard_candidates(
        (),
        frozenset({"add", "milk", "to", "list"}),
        frozenset({4}),
        {},
        {},
        {},
        {},
        (
            WildcardVariantGroup(
                variants=variants,
                literal_tokens=frozenset({"add", "to", "list"}),
                min_required_match_count=3,
                candidate_indices=(1, 2, 3),
            ),
        ),
    )

    assert passed == {1, 2, 3, 4}
    assert calls == 1


def test_prebuilt_static_bm25_matches_on_the_fly_sparse_ranking() -> None:
    """Compare prebuilt and on-the-fly sparse-BM25 static ranking."""
    candidates = [
        Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
        Candidate(text="turn off kitchen light", intent_name="HassTurnOff", language="en"),
        Candidate(
            text="set kitchen light to 50 percent", intent_name="HassLightSet", language="en"
        ),
        Candidate(text="turn on bedroom fan", intent_name="HassTurnOn", language="en"),
        Candidate(text="open garage door", intent_name="HassTurnOn", language="en"),
    ]
    query = "turn kitchen light on"
    normalized_texts = tuple(candidate.normalized_text for candidate in candidates)
    bm25_index = BM25Index.from_normalized_texts(normalized_texts)
    char_index = CharNGramIndex.from_grams(
        tuple(ranking.char_ngrams_normalized(text) for text in normalized_texts)
    )

    dense_ranked = rank_candidates(query, candidates, max_candidates=5, language="en")
    hybrid_ranked = rank_candidates(
        query,
        candidates,
        max_candidates=5,
        bm25_index=bm25_index,
        candidate_char_index=char_index,
        language="en",
    )

    assert [item.candidate for item in hybrid_ranked] == [item.candidate for item in dense_ranked]
    assert [item.scores for item in hybrid_ranked] == [item.scores for item in dense_ranked]
    assert accepted_candidate(hybrid_ranked) == accepted_candidate(dense_ranked)


def test_reference_bm25_ranking_uses_dense_normalized_scores() -> None:
    """Use normalized dense reference scores for dynamic candidate ranking."""
    candidates = [
        Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
        Candidate(text="turn off kitchen fan", intent_name="HassTurnOff", language="en"),
    ]
    query = "turn on"
    query_tokens = tuple(normalize_text(query).split())
    reference_bm25_index = BM25Index.from_normalized_texts(
        ("turn on hall light", "turn off garage fan", "set bedroom temperature")
    )
    expected_scores = reference_bm25_index.score_custom_documents_tokens(
        query_tokens,
        candidates,
    )

    scoring = ranking._prepare_bm25_scoring(
        candidates,
        query_tokens,
        None,
        reference_bm25_index,
    )
    ranked = rank_candidates(
        query,
        candidates,
        max_candidates=2,
        reference_bm25_index=reference_bm25_index,
        language="en",
    )

    assert scoring.sparse_raw_scores is None
    assert tuple(scoring.score_at(index) for index in range(len(candidates))) == expected_scores
    assert {item.candidate: item.scores.bm25_score for item in ranked} == dict(
        zip(candidates, expected_scores, strict=True)
    )


def test_reference_slot_index_penalizes_wildcard_absorbing_static_target() -> None:
    """Carry static target knowledge into query-scoped wildcard ranking."""
    wildcard = Candidate(
        text="play search_query",
        intent_name="HassMediaSearchAndPlay",
        language="en",
        metadata={
            "sentence_template": "play {search_query}",
            "wildcard_slots": "search_query",
            "slots": '{"search_query":"search_query"}',
            "literal_text": "play",
        },
    )
    reference_slot_index = SlotTokenIndex.from_candidate_tokens((frozenset({"vacuum"}),))

    without_reference = rank_candidates(
        "play vacuum",
        (wildcard,),
        language="en",
    )
    with_reference = rank_candidates(
        "play vacuum",
        (wildcard,),
        language="en",
        reference_slot_token_index=reference_slot_index,
    )

    assert with_reference[0].scores.final_score < without_reference[0].scores.final_score


def test_rank_candidates_validates_candidate_slot_token_length() -> None:
    """Reject precomputed slot-token data that does not match the candidate list."""
    candidates = [Candidate(text="turn on kitchen light", intent_name="HassTurnOn")]

    with pytest.raises(ValueError, match="candidate_slot_tokens length must match candidates"):
        rank_candidates(
            "turn on kitchen light",
            candidates,
            candidate_slot_tokens=(),
        )


def test_rank_candidates_validates_static_bm25_index_length() -> None:
    """Reject a sparse BM25 index not built from the candidate sequence."""
    candidates = [Candidate(text="turn on kitchen light", intent_name="HassTurnOn")]
    bm25_index = BM25Index.from_normalized_texts(("turn on kitchen light", "turn off fan"))

    with pytest.raises(ValueError, match="bm25_index length must match candidates"):
        rank_candidates("turn on kitchen light", candidates, bm25_index=bm25_index)


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


def test_positional_intent_score_uses_one_to_one_query_token_alignment() -> None:
    """Do not credit one fuzzy query token to multiple template literals."""
    score = ranking._positional_intent_score_from_lookup(
        "dem der",
        frozenset({"den"}),
        {
            "dem": frozenset({"den"}),
            "der": frozenset({"den"}),
        },
    )

    assert score == 0.25


def test_best_positional_score_finds_maximum_unique_alignment() -> None:
    """Use maximum matching when fuzzy literal-token alternatives overlap."""
    analysis = [
        ranking._LiteralVariantAnalysis(
            total_token_count=2,
            exact_match_tokens=frozenset(),
            positional_hits=(frozenset({"a", "b"}), frozenset({"a"})),
            positional_query_tokens=frozenset({"a", "b"}),
            positional_match_count=2,
            requires_unique_alignment=True,
        )
    ]

    assert _best_positional_score(analysis, frozenset()) == 0.5
    assert _best_positional_score(analysis, frozenset({"b"})) == 0.25


def test_best_positional_score_uses_maximum_across_variants() -> None:
    """Select the highest positional score across all literal variants."""
    analysis = [
        ranking._LiteralVariantAnalysis(
            total_token_count=2,
            exact_match_tokens=frozenset(),
            positional_hits=(),
            positional_query_tokens=frozenset(),
            positional_match_count=0,
            requires_unique_alignment=False,
        ),
        ranking._LiteralVariantAnalysis(
            total_token_count=2,
            exact_match_tokens=frozenset(),
            positional_hits=(frozenset({"a", "b"}), frozenset({"a"})),
            positional_query_tokens=frozenset({"a", "b"}),
            positional_match_count=2,
            requires_unique_alignment=True,
        ),
        ranking._LiteralVariantAnalysis(
            total_token_count=2,
            exact_match_tokens=frozenset(),
            positional_hits=(frozenset({"c"}),),
            positional_query_tokens=frozenset({"c"}),
            positional_match_count=1,
            requires_unique_alignment=False,
        ),
    ]

    assert _best_positional_score(analysis, frozenset()) == 0.5


def test_best_positional_score_does_not_reuse_exact_query_tokens() -> None:
    """Reserve exact query-token evidence before fuzzy literal alignment."""
    analysis = [
        ranking._LiteralVariantAnalysis(
            total_token_count=2,
            exact_match_tokens=frozenset({"turn"}),
            positional_hits=(frozenset({"turn"}),),
            positional_query_tokens=frozenset({"turn"}),
            positional_match_count=0,
            requires_unique_alignment=True,
        )
    ]

    assert _best_positional_score(analysis, frozenset()) == 0.5


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


def test_rank_candidates_strongly_penalizes_unspoken_numeric_literal() -> None:
    """Reject generated numeric text when the user did not supply a number."""
    generated_speed = Candidate(
        text="bathroom fan 0",
        intent_name="HassFanSetSpeed",
        language="en",
        metadata={"slots": '{"name":"bathroom fan","percentage":0}'},
    )

    ranked = rank_candidates(
        "bathroom fan off",
        [generated_speed],
        max_candidates=1,
        language="en",
    )

    assert ranked[0].scores.final_score < DEFAULT_MIN_CONFIDENCE


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


def test_rank_candidates_penalizes_candidate_that_ignores_query_number() -> None:
    """Prefer a numeric action over a lexically closer candidate that drops the number."""
    ignores_number = Candidate(
        text="bedroom window open",
        intent_name="HassTurnOn",
        language="en",
        metadata={
            "slots": '{"name":"bedroom window"}',
            "slots_raw": '{"name":"bedroom window"}',
        },
    )
    consumes_number = Candidate(
        text="bedroom window to 0",
        intent_name="HassSetPosition",
        language="en",
        metadata={
            "slots": '{"name":"bedroom window","position":0}',
            "slots_raw": '{"name":"bedroom window","position":"0"}',
        },
    )

    ranked = rank_candidates(
        "bedroom window open 0",
        [ignores_number, consumes_number],
        max_candidates=2,
        language="en",
    )

    assert ranked[0].candidate is consumes_number
    assert ranked[1].candidate is ignores_number


def test_rank_candidates_penalizes_unspoken_semantic_slot() -> None:
    """Do not invent a non-target action parameter absent from the query."""
    turn_on = Candidate(
        text="turn on living room light",
        intent_name="HassTurnOn",
        language="en",
        metadata={
            "slots": '{"name":"living room light"}',
            "slots_raw": '{"name":"living room light"}',
            "literal_text": "turn on",
        },
    )
    set_white = Candidate(
        text="turn living room light white",
        intent_name="HassLightSet",
        language="en",
        metadata={
            "slots": '{"name":"living room light","color":"white"}',
            "slots_raw": '{"name":"living room light","color":"white"}',
            "literal_text": "turn",
        },
    )

    implicit_ranked = rank_candidates(
        "turn living room light",
        [set_white, turn_on],
        max_candidates=2,
        language="en",
    )
    explicit_ranked = rank_candidates(
        "turn living room light white",
        [turn_on, set_white],
        max_candidates=2,
        language="en",
    )

    assert implicit_ranked[0].candidate is turn_on
    assert explicit_ranked[0].candidate is set_white


def test_semantic_slot_anchor_uses_spoken_value_before_output_mapping() -> None:
    """Anchor multilingual mapped parameters by their raw spoken surface value."""
    mapped_color = Candidate(
        text="licht blau",
        intent_name="HassLightSet",
        language="de",
        metadata={
            "slots": '{"name":"licht","color":"blue"}',
            "slots_raw": '{"name":"licht","color":"blau"}',
        },
    )

    assert not ranking._has_unanchored_semantic_slot(
        mapped_color,
        mapped_color.parsed_slots,
        frozenset(),
        ("licht", "blau"),
    )
    assert ranking._has_unanchored_semantic_slot(
        mapped_color,
        mapped_color.parsed_slots,
        frozenset(),
        ("licht",),
    )


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


def test_rank_candidates_explicit_location_overrides_intent_context() -> None:
    """Do not steer an explicit user location toward the satellite area."""
    generic = Candidate(
        text="turn off fans",
        intent_name="HassTurnOff",
        language="en",
        metadata={"slots": '{"domain":"fan"}', "static_slots": "domain"},
    )
    bathroom = Candidate(
        text="turn off bathroom fan",
        intent_name="HassTurnOff",
        language="en",
        metadata={
            "slots": '{"domain":"fan","area":"Bathroom"}',
            "static_slots": "domain",
        },
    )

    ranked = rank_candidates(
        "turn off bath",
        [generic, bathroom],
        max_candidates=2,
        language="en",
        intent_context={"area": "Living Room"},
    )
    without_context = rank_candidates(
        "turn off bath",
        [generic, bathroom],
        max_candidates=2,
        language="en",
    )

    assert [item.candidate for item in ranked] == [item.candidate for item in without_context]
    assert [item.scores.final_score for item in ranked] == pytest.approx(
        [item.scores.final_score for item in without_context]
    )


def test_rank_candidates_unknown_vocabulary_disables_intent_context() -> None:
    """Context must not boost candidates when the query names unknown vocabulary.

    Regression: ``allume la téloche`` (slang for TV, unknown to the corpus and
    registry) was boosted by satellite area context over the acceptance margin
    and executed a generic light candidate instead of falling back.
    """
    generic = Candidate(
        text="turn on the lamp",
        intent_name="HassTurnOn",
        language="en",
        metadata={
            "slots": '{"domain":"light"}',
            "static_slots": "domain",
            "literal_text": "turn on the lamp",
        },
    )
    kitchen = Candidate(
        text="turn on the kitchen",
        intent_name="HassTurnOn",
        language="en",
        metadata={
            "slots": '{"domain":"light","area":"Kitchen"}',
            "static_slots": "domain",
            "literal_text": "turn on the",
        },
    )
    intent_context = {"area": {"value": "Kitchen", "text": "Kitchen"}}

    known_with_context = rank_candidates(
        "turn on the",
        [generic, kitchen],
        max_candidates=2,
        language="en",
        intent_context=intent_context,
    )
    known_without_context = rank_candidates(
        "turn on the",
        [generic, kitchen],
        max_candidates=2,
        language="en",
    )

    assert [item.candidate for item in known_with_context] == [kitchen, generic]
    assert [item.candidate for item in known_without_context] == [generic, kitchen]
    assert known_with_context[0].scores.final_score > known_without_context[1].scores.final_score

    with_context = rank_candidates(
        "turn on the telly",
        [generic, kitchen],
        max_candidates=2,
        language="en",
        intent_context=intent_context,
    )
    without_context = rank_candidates(
        "turn on the telly",
        [generic, kitchen],
        max_candidates=2,
        language="en",
    )

    assert [item.candidate for item in with_context] == [item.candidate for item in without_context]
    assert [item.scores.final_score for item in with_context] == pytest.approx(
        [item.scores.final_score for item in without_context]
    )


def test_rank_candidates_explicit_entity_overrides_intent_context() -> None:
    """Do not boost an area action over an explicitly named entity action."""
    named = Candidate(
        text="start the vacuum",
        intent_name="HassVacuumStart",
        language="en",
        metadata={"slots": '{"name":"Vacuum"}', "literal_text": "start"},
    )
    context_area = Candidate(
        text="clean the area",
        intent_name="HassVacuumCleanArea",
        language="en",
        metadata={"context_slots": "area", "literal_text": "clean"},
    )

    ranked = rank_candidates(
        "start vacuum",
        [context_area, named],
        max_candidates=2,
        language="en",
        intent_context={"area": "Living Room"},
    )
    without_context = rank_candidates(
        "start vacuum",
        [context_area, named],
        max_candidates=2,
        language="en",
    )

    assert [item.candidate for item in ranked] == [item.candidate for item in without_context]
    assert [item.scores.final_score for item in ranked] == pytest.approx(
        [item.scores.final_score for item in without_context]
    )


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


def test_rank_candidates_penalizes_optional_literal_implicit_action() -> None:
    """Optional determiners must not turn an implicit action into verb evidence."""
    implicit_turn_on = Candidate(
        text="the bathroom fan",
        intent_name="HassTurnOn",
        language="en",
        metadata={
            "slots": orjson.dumps({"name": "bathroom fan"}).decode("utf-8"),
            "literal_text": "the|my|our",
            "literal_variants": orjson.dumps([[], ["the"], ["my"], ["our"]]).decode("utf-8"),
        },
    )
    explicit_turn_off = Candidate(
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
        [implicit_turn_on, explicit_turn_off],
        max_candidates=2,
        language="en",
    )

    assert ranked[0].candidate is explicit_turn_off
    assert ranked[1].candidate is implicit_turn_on


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


def test_rank_candidates_partial_diacritic_omission_short_circuits() -> None:
    """Treat missing diacritics as directional omissions even in mixed text."""
    index = build_index(
        "vi",
        [Candidate(text="bật đèn phòng khách", intent_name="HassTurnOn", language="vi")],
    )

    ranked = index.rank("bat đèn phòng khách")

    assert ranked[0].candidate.text == "bật đèn phòng khách"
    assert ranked[0].scores.final_score == 1.0


def test_rank_candidates_no_diacritics_same_intent_slot_collision_is_ambiguous() -> None:
    """Avoid perfect promotion when base-identical candidates carry different slots."""
    index = build_index(
        "vi",
        [
            Candidate(
                text="bật bàn",
                intent_name="HassTurnOn",
                language="vi",
                metadata={"slots": '{"name":"bàn"}'},
            ),
            Candidate(
                text="bật bạn",
                intent_name="HassTurnOn",
                language="vi",
                metadata={"slots": '{"name":"bạn"}'},
            ),
        ],
    )

    ranked = index.rank("bat ban")

    assert len(ranked) == 2
    assert all(candidate.scores.final_score < 1.0 for candidate in ranked)


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
    assert rank_candidates("   !!!   ", candidates) == ()


def test_apply_intent_disambiguation_promotes_higher_intent_score_for_exact_tie() -> None:
    """Use intent evidence only after the combined lexical scores tie."""
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
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=0.9,
            final_score=1.0,
        ),
    )

    ranked = [first, second]

    ranking.apply_intent_disambiguation(ranked)

    # The higher intent_score candidate with a different intent should be promoted to rank 0
    assert ranked[0] is second
    assert ranked[1] is first


def test_apply_intent_disambiguation_does_not_promote_lower_final_score() -> None:
    """Do not double-count intent evidence after final scores are calculated."""
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
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=0.9,
            final_score=0.999,
        ),
    )

    ranked = [first, second]

    ranking.apply_intent_disambiguation(ranked)

    assert ranked[0] is first
    assert ranked[1] is second


def test_apply_intent_disambiguation_keeps_order_for_same_intent() -> None:
    """Top-2 sharing one intent preserve their lexical score order."""
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
            final_score=1.0,
        ),
    )

    ranked = [first, second]

    ranking.apply_intent_disambiguation(ranked)

    # Because the intents are identical, the original ordering should be preserved
    assert ranked[0] is first
    assert ranked[1] is second


def test_apply_intent_disambiguation_skips_same_intent_variants() -> None:
    """Compare the top candidate with the first genuinely competing intent."""
    first = RankedCandidate(
        candidate=Candidate(text="turn kitchen light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(0.9, 0.9, 0.9, 0.5, 1.0),
    )
    same_intent = RankedCandidate(
        candidate=Candidate(text="switch kitchen light", intent_name="HassTurnOn"),
        scores=ScoreBreakdown(0.9, 0.9, 0.9, 0.6, 1.0),
    )
    competitor = RankedCandidate(
        candidate=Candidate(text="turn kitchen light off", intent_name="HassTurnOff"),
        scores=ScoreBreakdown(0.9, 0.9, 0.9, 0.9, 1.0),
    )
    ranked = [first, same_intent, competitor]

    ranking.apply_intent_disambiguation(ranked)

    assert ranked == [competitor, same_intent, first]


@pytest.mark.current_intents
def test_rank_candidates_rehydrates_wildcard() -> None:
    """Test that rank_candidates correctly rehydrates wildcard placeholder candidates."""
    # We create a candidate with a wildcard slot
    candidate = Candidate(
        text="broadcast message",
        intent_name="HassBroadcast",
        language="en",
        metadata={
            "sentence_template": "broadcast {message}",
            "wildcard_slots": "message",
            "literal_text": "broadcast message",
        },
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
    assert generic_media.wildcard_infos == ((0, "search_query"),)
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
    assert wildcard_with_anchor.wildcard_infos == ((0, "search_query"),)
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
        metadata={
            "sentence_template": "{message}",
            "wildcard_slots": "message",
            "literal_text": "message|messages|broadcast message",
        },
    )

    variants, all_tokens = wildcard_variants_analysis(candidate)

    assert variants == (
        WildcardVariantAnalysis(literal_tokens=frozenset(), required_match_count=0),
        WildcardVariantAnalysis(
            literal_tokens=frozenset({"messages"}),
            required_match_count=1,
        ),
        WildcardVariantAnalysis(
            literal_tokens=frozenset({"broadcast"}),
            required_match_count=1,
        ),
    )
    assert all_tokens == frozenset({"broadcast", "messages"})


@pytest.mark.current_intents
def test_wildcard_variants_analysis_removes_embedded_structured_wildcard() -> None:
    """Do not keep placeholder-bearing tokens as wildcard literal anchors."""
    candidate = Candidate(
        text="search_querypodcast",
        intent_name="HassMediaSearchAndPlay",
        language="de",
        metadata={
            "sentence_template": "{search_query}podcast",
            "wildcard_slots": "search_query",
            "literal_text": "spiel den search_querypodcast|spiel den podcast",
        },
    )

    variants, all_tokens = wildcard_variants_analysis(candidate)

    assert variants == (
        WildcardVariantAnalysis(
            literal_tokens=frozenset({"den", "spiel"}),
            required_match_count=2,
        ),
        WildcardVariantAnalysis(
            literal_tokens=frozenset({"den", "podcast", "spiel"}),
            required_match_count=3,
        ),
    )
    assert all_tokens == frozenset({"den", "podcast", "spiel"})


@pytest.mark.current_intents
def test_wildcard_variants_analysis_removes_embedded_single_word_wildcard() -> None:
    """Strip single-word wildcard names embedded in literal tokens."""
    candidate = Candidate(
        text="urgentmessage",
        intent_name="HassBroadcast",
        language="en",
        metadata={
            "sentence_template": "urgent{message}",
            "wildcard_slots": "message",
            "literal_text": "broadcast urgentmessage|broadcast",
        },
    )

    variants, all_tokens = wildcard_variants_analysis(candidate)

    assert variants == (
        WildcardVariantAnalysis(
            literal_tokens=frozenset({"broadcast"}),
            required_match_count=1,
        ),
    )
    assert all_tokens == frozenset({"broadcast"})


@pytest.mark.current_intents
def test_wildcard_variants_analysis_keeps_multilingual_word_containing_wildcard() -> None:
    """Do not strip non-English words that merely contain a wildcard name."""
    candidate = Candidate(
        text="messagerie",
        intent_name="HassBroadcast",
        language="fr",
        metadata={
            "sentence_template": "{message}",
            "wildcard_slots": "message",
            "literal_text": "messagerie|diffuser message",
        },
    )

    variants, all_tokens = wildcard_variants_analysis(candidate)

    assert variants == (
        WildcardVariantAnalysis(
            literal_tokens=frozenset({"messagerie"}),
            required_match_count=1,
        ),
        WildcardVariantAnalysis(
            literal_tokens=frozenset({"diffuser"}),
            required_match_count=1,
        ),
    )
    assert all_tokens == frozenset({"diffuser", "messagerie"})


@pytest.mark.parametrize(
    ("variants", "query_tokens", "expected"),
    [
        (
            (
                WildcardVariantAnalysis(
                    literal_tokens=frozenset(),
                    required_match_count=0,
                ),
            ),
            frozenset(),
            True,
        ),
        (
            (
                WildcardVariantAnalysis(
                    literal_tokens=frozenset({"add", "list"}),
                    required_match_count=2,
                ),
            ),
            frozenset({"add", "list"}),
            True,
        ),
        (
            (
                WildcardVariantAnalysis(
                    literal_tokens=frozenset({"add", "list"}),
                    required_match_count=2,
                ),
            ),
            frozenset({"add"}),
            False,
        ),
        (
            (
                WildcardVariantAnalysis(
                    literal_tokens=frozenset({"play"}),
                    required_match_count=1,
                ),
            ),
            frozenset({"play"}),
            True,
        ),
        (
            (
                WildcardVariantAnalysis(
                    literal_tokens=frozenset({"add", "item", "to", "list"}),
                    required_match_count=3,
                ),
            ),
            frozenset({"add", "to", "list"}),
            True,
        ),
        (
            (
                WildcardVariantAnalysis(
                    literal_tokens=frozenset({"add", "item", "to", "list"}),
                    required_match_count=3,
                ),
            ),
            frozenset({"add", "list"}),
            False,
        ),
    ],
)
def test_wildcard_variants_match_preserves_coverage_semantics(
    variants: tuple[WildcardVariantAnalysis, ...],
    query_tokens: frozenset[str],
    expected: bool,
) -> None:
    """Cover empty, full-subset, single-token, and partial thresholds."""
    assert _wildcard_variants_match(variants, query_tokens) is expected


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
        metadata={
            "sentence_template": "broadcast {message}",
            "wildcard_slots": "message",
            "literal_text": "broadcast message",
        },
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
        wildcard_variant_analyses={},
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
        metadata={
            "sentence_template": "add {shopping_list_item}",
            "wildcard_slots": "shopping_list_item",
            "literal_text": "add",
        },
    )
    cand_todo = Candidate(
        text="add todo_list_item",
        intent_name="HassListAddItem",
        language="en",
        metadata={
            "sentence_template": "add {todo_list_item}",
            "wildcard_slots": "todo_list_item",
            "literal_text": "add",
        },
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
        metadata={
            "sentence_template": "{message}",
            "wildcard_slots": "message",
            "literal_text": "",
        },
    )
    cand_var_len_0 = Candidate(
        text="broadcast message",
        intent_name="HassBroadcastVar0",
        language="en",
        metadata={
            "sentence_template": "broadcast {message}",
            "wildcard_slots": "message",
            "literal_text": "|broadcast",
        },
    )
    cand_normal = Candidate(
        text="broadcast message",
        intent_name="HassBroadcastNormal",
        language="en",
        metadata={
            "sentence_template": "broadcast {message}",
            "wildcard_slots": "message",
            "literal_text": "broadcast",
        },
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


def test_numeric_slot_mismatch_penalty() -> None:
    """Verify that candidates with a mismatched numeric slot value are penalized."""
    cand_mismatch = Candidate(
        text="set living room temperature to 0",
        intent_name="HassClimateSetTemperature",
        language="en",
        metadata={
            "slots": orjson.dumps({"name": "living room", "temperature": 0}).decode("utf-8"),
            "literal_text": "set temperature",
        },
        slot_values=("living room", "0"),
    )

    cand_match = Candidate(
        text="set living room temperature to 27",
        intent_name="HassClimateSetTemperature",
        language="en",
        metadata={
            "slots": orjson.dumps({"name": "living room", "temperature": 27}).decode("utf-8"),
            "literal_text": "set temperature",
        },
        slot_values=("living room", "27"),
    )

    ranked = rank_candidates(
        query="set living room temperature to 27",
        candidates=[cand_mismatch, cand_match],
        max_candidates=2,
        language="en",
    )
    assert len(ranked) == 2
    assert ranked[0].candidate == cand_match

    mismatch_item = next(r for r in ranked if r.candidate == cand_mismatch)
    match_item = next(r for r in ranked if r.candidate == cand_match)
    assert mismatch_item.scores.final_score < match_item.scores.final_score


def test_numeric_slot_mismatch_penalty_with_multiplier() -> None:
    """Verify that range candidates with multipliers are not penalized."""
    cand = Candidate(
        text="volume down by 20",
        intent_name="HassSetVolumeRelative",
        language="en",
        metadata={
            "slots": orjson.dumps({"volume_step": -20}).decode("utf-8"),
        },
        slot_values=("-20",),
    )

    ranked = rank_candidates(
        query="volume down by 20",
        candidates=[cand],
        max_candidates=1,
        language="en",
    )
    assert len(ranked) == 1
    assert ranked[0].scores.final_score > 0.8


def test_numeric_slot_mismatch_penalty_static_number() -> None:
    """Verify that templates with static numbers are not penalized for mismatches."""
    cand = Candidate(
        text="set temperature in area 51 to 20",
        intent_name="HassClimateSetTemperature",
        language="en",
        metadata={
            "slots": orjson.dumps({"temperature": 20}).decode("utf-8"),
        },
        slot_values=("20",),
    )

    ranked = rank_candidates(
        query="set temperature to 20",
        candidates=[cand],
        max_candidates=1,
        language="en",
    )
    assert len(ranked) == 1
    assert ranked[0].scores.final_score > 0.6


def test_numeric_slot_mismatch_static_number_loophole() -> None:
    """Verify mismatch penalty is applied despite candidate text containing static numbers."""
    cand = Candidate(
        text="set temperature in area 51 to 20",
        intent_name="HassClimateSetTemperature",
        language="en",
        metadata={
            "slots": orjson.dumps({"temperature": 20}).decode("utf-8"),
        },
        slot_values=("20",),
    )

    # When query has 51, but candidate sets temperature to 20.
    # The temperature slot (value 20) mismatches 51. So it must be penalized.
    ranked = rank_candidates(
        query="set temperature to 51",
        candidates=[cand],
        max_candidates=1,
        language="en",
    )
    assert len(ranked) == 1
    assert ranked[0].scores.final_score < 0.1


def test_numeric_slot_mismatch_with_slots_raw() -> None:
    """Verify mismatch penalty check uses slots_raw when available."""
    cand = Candidate(
        text="set timer for 5 minutes",
        intent_name="HassSetTimer",
        language="en",
        metadata={
            "slots": orjson.dumps({"duration": 300}).decode("utf-8"),
            "slots_raw": orjson.dumps({"duration": "5"}).decode("utf-8"),
        },
        slot_values=("300",),
    )

    # Query matching raw value (5)
    ranked = rank_candidates(
        query="set timer for 5 minutes",
        candidates=[cand],
        max_candidates=1,
        language="en",
    )
    assert len(ranked) == 1
    assert ranked[0].scores.final_score > 0.8

    # Query with mismatching raw value (10)
    ranked_mismatch = rank_candidates(
        query="set timer for 10 minutes",
        candidates=[cand],
        max_candidates=1,
        language="en",
    )
    assert len(ranked_mismatch) == 1
    assert ranked_mismatch[0].scores.final_score < 0.1


def test_numeric_slot_mismatch_penalty_unsupported_multiplier_without_slots_raw() -> None:
    """Verify Finding 1: Mismatch penalty is strictly applied when slots_raw is missing.

    The multiplier scale factor is a decimal fraction (e.g. 0.25 for 25%).
    """
    # A candidate where brightness is 0.25 (output value) representing 25% (multiplier 0.01).
    # Slots contains the multiplied output value (0.25), but slots_raw is missing.
    cand_without_raw = Candidate(
        text="set brightness to 25",
        intent_name="HassLightSetBrightness",
        language="en",
        metadata={
            "slots": orjson.dumps({"brightness": 0.25}).decode("utf-8"),
        },
        slot_values=("0.25",),
    )

    # When query has 25, it fails to match 0.25 without slots_raw,
    # applying the mismatch penalty.
    ranked = rank_candidates(
        query="set brightness to 25",
        candidates=[cand_without_raw],
        max_candidates=1,
        language="en",
    )
    assert len(ranked) == 1
    assert ranked[0].scores.final_score < 0.1  # Heavily penalized due to mismatch

    # Verify that when slots_raw IS populated with "25" for brightness 0.25, it matches correctly.
    cand_with_raw = Candidate(
        text="set brightness to 25",
        intent_name="HassLightSetBrightness",
        language="en",
        metadata={
            "slots": orjson.dumps({"brightness": 0.25}).decode("utf-8"),
            "slots_raw": orjson.dumps({"brightness": "25"}).decode("utf-8"),
        },
        slot_values=("0.25",),
    )

    ranked_with_raw = rank_candidates(
        query="set brightness to 25",
        candidates=[cand_with_raw],
        max_candidates=1,
        language="en",
    )
    assert len(ranked_with_raw) == 1
    assert ranked_with_raw[0].scores.final_score > 0.8  # Not penalized


def test_wildcard_infos_multiple_wildcards_extraction() -> None:
    """Verify that wildcard_infos extracts all wildcard tokens in a template."""
    # Register two custom wildcards: "wildcard_one" and "wildcard_two"
    register_custom_wildcards_from_sources(
        "en",
        {
            "custom": {
                "lists": {
                    "wildcard_one": {"wildcard": True},
                    "wildcard_two": {"wildcard": True},
                }
            }
        },
    )

    # Verify that both are indeed registered
    sorted_wcs = wildcard_slot_names_sorted("en")
    assert "wildcard_one" in sorted_wcs
    assert "wildcard_two" in sorted_wcs

    # Create a candidate with both wildcards
    cand = Candidate(
        text="wildcard_one wildcard_two",
        intent_name="TestIntent",
        language="en",
    )

    # Verify that wildcard_infos captures both wildcards.
    assert cand.has_wildcard is True
    assert cand.wildcard_infos == ((0, "wildcard_one"), (1, "wildcard_two"))


def test_wildcard_infos_uses_persisted_candidate_metadata() -> None:
    """Restore local wildcard ownership without relying on global registration."""
    candidate = Candidate(
        text="say local_free_text",
        intent_name="TestIntent",
        language="unregistered-language",
        metadata={
            "sentence_template": "say {local_free_text}",
            "wildcard_slots": "local_free_text",
        },
    )

    assert candidate.wildcard_infos == ((1, "local_free_text"),)


def test_multi_wildcard_rehydration() -> None:
    """Verify multiple wildcards are correctly aligned and rehydrated."""
    register_custom_wildcards_from_sources(
        "en",
        {
            "custom": {
                "lists": {
                    "song": {"wildcard": True},
                    "artist": {"wildcard": True},
                }
            }
        },
    )
    cand = Candidate(
        text="play song by artist",
        intent_name="HassPlayMedia",
        language="en",
        metadata={
            "sentence_template": "play {song} by {artist}",
            "wildcard_slots": "song,artist",
            "slots": '{"song":"song","artist":"artist"}',
        },
        slot_values=("song", "artist"),
    )

    # Both wildcards should be detected in wildcard_infos
    assert len(cand.wildcard_infos) == 2
    assert cand.wildcard_infos[0] == (1, "song")
    assert cand.wildcard_infos[1] == (3, "artist")

    rehydrated_text, slots = get_wildcard_rehydration(
        cand,
        query="play yesterday by the beatles",
    )
    assert rehydrated_text == "play yesterday by the beatles"
    assert slots == {"song": "yesterday", "artist": "the beatles"}


def test_adjacent_wildcards_fail_closed_without_a_literal_boundary() -> None:
    """Do not assign overlapping query spans to inseparable free-text slots."""
    register_custom_wildcards_from_sources(
        "en",
        {"custom": {"lists": {"song": {"wildcard": True}, "artist": {"wildcard": True}}}},
    )
    candidate = Candidate(
        text="play song artist",
        intent_name="HassPlayMedia",
        language="en",
        metadata={
            "sentence_template": "play {song} {artist}",
            "wildcard_slots": "song,artist",
        },
        slot_values=("song", "artist"),
    )

    rehydrated_text, slots = get_wildcard_rehydration(
        candidate,
        query="play yesterday the beatles",
    )

    assert rehydrated_text == candidate.text
    assert slots == {}


def test_evaluate_confidence_gates_empty_and_success_states() -> None:
    """Test evaluate_confidence_gates with empty and success candidate states."""
    # Empty sequence should return NO_CANDIDATE fallback reason
    empty_decision = evaluate_confidence_gates(())
    assert empty_decision.accepted_candidate is None
    assert empty_decision.rejection_reason == FallbackReason.NO_CANDIDATE
    assert empty_decision.margin_policy == "no_candidate"
    assert empty_decision.observed_margin is None

    top = RankedCandidate(
        candidate=Candidate(text="turn on kitchen light", intent_name="HassTurnOn", language="en"),
        scores=ScoreBreakdown(
            rapidfuzz_score=0.9,
            char_ngram_score=0.9,
            bm25_score=0.9,
            intent_score=1.0,
            final_score=0.9,
        ),
    )
    # Success branch should return None for the reason
    decision = evaluate_confidence_gates((top,))
    assert decision.accepted_candidate is top
    assert decision.rejection_reason is None
    assert decision.margin_policy == "no_competitor"
    assert decision.required_margin == 0.0
    top_candidate_payload = decision.as_dict()["top_candidate"]
    assert top_candidate_payload is not None
    assert top_candidate_payload["text_sha256"]


@pytest.mark.parametrize(
    ("lexical_score", "expected_policy", "required_margin"),
    [(1.0, "exact_lexical", 0.0), (0.99, "high_confidence_relaxation", 0.005)],
)
def test_opposing_actions_report_exact_or_high_confidence_relaxation(
    lexical_score: float,
    expected_policy: str,
    required_margin: float,
) -> None:
    """Retain calibrated evidence while reporting close on/off competition."""
    top = RankedCandidate(
        Candidate(
            text="turn on kitchen light",
            intent_name="HassTurnOn",
            language="en",
            metadata={"slots": '{"name":"kitchen light"}'},
        ),
        ScoreBreakdown(lexical_score, lexical_score, 1.0, 1.0, 0.90),
    )
    competitor = RankedCandidate(
        Candidate(
            text="turn off kitchen light",
            intent_name="HassTurnOff",
            language="en",
            metadata={"slots": '{"name":"kitchen light"}'},
        ),
        ScoreBreakdown(0.99, 0.99, 0.99, 0.99, 0.89),
    )

    decision = evaluate_confidence_gates((top, competitor), min_margin=DEFAULT_MIN_MARGIN)

    assert decision.accepted_candidate is top
    assert decision.margin_policy == expected_policy
    assert decision.required_margin == required_margin
    assert decision.opposing_action_competition
    assert decision.relaxation_used


def test_base_margin_accepts_equal_boundary_and_rejects_immediately_below() -> None:
    """Use an inclusive required-margin boundary with explicit evidence."""
    top = RankedCandidate(
        Candidate(
            text="query living room state",
            intent_name="HassGetState",
            language="en",
            metadata={"slots": '{"name":"living room"}'},
        ),
        ScoreBreakdown(0.75, 0.75, 0.75, 0.50, 0.75),
    )
    equal = RankedCandidate(
        Candidate(
            text="query living room weather",
            intent_name="HassGetWeather",
            language="en",
            metadata={"slots": '{"name":"living room"}'},
        ),
        ScoreBreakdown(0.70, 0.70, 0.70, 0.50, 0.70),
    )
    below = RankedCandidate(
        equal.candidate,
        ScoreBreakdown(0.700001, 0.700001, 0.700001, 0.50, 0.700001),
    )

    equal_decision = evaluate_confidence_gates((top, equal), min_margin=0.05)
    below_decision = evaluate_confidence_gates((top, below), min_margin=0.05)

    assert equal_decision.accepted
    assert equal_decision.margin_policy == "base_margin"
    assert equal_decision.required_margin == 0.05
    assert not below_decision.accepted
    assert below_decision.rejection_reason is FallbackReason.LOW_MARGIN


def test_clear_ranking_caches() -> None:
    """Test clear_ranking_caches does not raise exceptions."""
    # Warm up cache
    _raw_cached_fuzz_ratio("test", "test")
    # Verify cache is populated
    assert _raw_cached_fuzz_ratio.cache_info().currsize > 0
    # Clear cache
    clear_ranking_caches()
    # Verify cache is empty
    assert _raw_cached_fuzz_ratio.cache_info().currsize == 0


def test_token_count_ratio_empty() -> None:
    """Test token_count_ratio with empty strings."""
    assert token_count_ratio("", "candidate") == 0.0
    assert token_count_ratio("query", "") == 0.0


def test_positional_similarity_empty() -> None:
    """Test _positional_similarity with empty strings."""
    assert _positional_similarity("", "") == 1.0


def test_per_pair_positional_threshold() -> None:
    """Test _per_pair_positional_threshold with various token lengths."""
    # Length <= 2
    assert _per_pair_positional_threshold("a", "bc") > 0.0
    # Length 3
    assert _per_pair_positional_threshold("abc", "def") > 0.0
    # Length <= 5
    assert _per_pair_positional_threshold("abcd", "efgh") > 0.0
    # Length > 5
    assert _per_pair_positional_threshold("abcdefg", "hijklmn") > 0.0


def test_is_numeric_slot_value_edge_cases() -> None:
    """Test _is_numeric_slot_value with edge cases."""
    # Bool (even though bool is a subclass of int, this should not be treated as numeric)
    assert not _is_numeric_slot_value(True)

    # Empty string
    assert not _is_numeric_slot_value("  ")

    # Non-numeric string
    assert not _is_numeric_slot_value("abc")

    # List or other non-scalar types
    assert not _is_numeric_slot_value([])

    # Integers
    assert _is_numeric_slot_value(0)
    assert _is_numeric_slot_value(42)
    assert _is_numeric_slot_value(-7)

    # Floats
    assert _is_numeric_slot_value(3.14)
    assert _is_numeric_slot_value(-0.001)

    # Numeric strings (integers)
    assert _is_numeric_slot_value("42")
    assert _is_numeric_slot_value("-17")

    # Numeric strings (floats)
    assert _is_numeric_slot_value("3.14")
    assert _is_numeric_slot_value("-0.001")


def test_has_static_slot_query_conflict_penalty() -> None:
    """Test _has_static_slot_query_conflict when a query slot token doesn't match."""
    query_slots = frozenset(["light"])
    candidate_tokens = frozenset(["turn", "on", "fan"])
    slots = {"name": "fan"}
    static_slots = frozenset(["name"])

    assert _has_static_slot_query_conflict(query_slots, candidate_tokens, slots, static_slots)


def test_rehydration_edge_cases() -> None:
    """Test rehydration edge cases in rehydration.py."""
    # _find_rehydration_boundaries prefix >= suffix boundary
    # c_prefix aligns to 1, c_suffix aligns to 0
    # query: "a b"
    # cand: "a wc b"
    # boundary check where prefix >= suffix
    assert _find_rehydration_boundaries(("a", "wc", "b"), 1, ("a",)) is None

    # _extract_wc_value fallback
    # When original query has mismatch, fallback to join of tokens
    assert _extract_wc_value("A B C", ("a", "b", "c"), 1, 3) == "B C"

    # _trim_wildcard_overlaps suffix overlap
    assert _trim_wildcard_overlaps("something_else", "something_else_suffix", "_else") == "_else"

    # _extract_original_span mismatch indexes
    # Invalid index returns empty string
    assert _extract_original_span("hello world", -1, 5) == ""

    # _is_wildcard_literal_token underscore check
    # Underscore wildcard name
    assert _is_wildcard_literal_token(
        token="some_wc_name",
        wc_name="some_wc",
        all_variant_tokens=set(),
        variant=frozenset(),
        variants=frozenset(),
    )

    # rehydrate_wildcard_slots no replacements returns unmodified slots
    # Passing slots, candidate with wildcard, and a query that doesn't align
    # should return original slots
    slots = {"some_key": "some_val"}
    assert rehydrate_wildcard_slots(slots, "play {song}", "invalid query", "en") == slots


def test_rehydration_spaced_unit_suffix_keeps_token_alignment() -> None:
    """Keep normalized indices aligned around spaced unit suffixes like "50 %".

    Whole-string normalization preserves a standalone "%"/"°" token after a
    digit, while naive per-token normalization drops it, shifting every later
    token index and silently leaking the placeholder into the output.
    """
    candidate = Candidate(
        text="regle 50 % puis joue media maintenant",
        intent_name="TestIntent",
        language="fr",
        metadata={
            "sentence_template": "regle 50 % puis joue {media} maintenant",
            "wildcard_slots": "media",
        },
    )
    assert candidate.wildcard_infos == ((5, "media"),)

    text, replacements = get_wildcard_rehydration(
        candidate, "regle 50 % puis joue Les Feuilles Mortes maintenant"
    )
    assert replacements == {"media": "Les Feuilles Mortes"}
    assert "media" not in text


def test_extract_original_span_composes_nfd_input() -> None:
    """Slice NFD (decomposed) surface tokens against NFC normalized tokens."""
    nfd_query = unicodedata.normalize("NFD", "allume Café-Lampe maintenant")
    assert _extract_original_span(nfd_query, 1, 2) == unicodedata.normalize("NFD", "Café")


def test_trim_wildcard_overlaps_handles_casefold_expansion() -> None:
    """Trim glued literal prefixes whose casefold changes length ("ß" → "ss")."""
    assert _trim_wildcard_overlaps("Straßenlampe eins", "strassenitem", "item") == "lampe eins"


def test_ranking_internal_helpers() -> None:
    """Directly test internal ranking helpers to ensure their stability."""
    # Non-trivial slot penalty path: conflicting slot tokens.
    # _check_and_calculate_conflict_penalty takes (cand_slot_tokens: frozenset,
    # wildcard_tokens: frozenset, candidate, context). Build a minimal context
    # whose query_slot_tokens conflict with the candidate's slot tokens so that
    # the penalty is non-trivial (i.e. < 1.0).
    conflict_ctx = _ScoringContext(
        query="play jazz songs",
        query_normalized="play jazz songs",
        query_tokens=frozenset(["play", "jazz", "songs"]),
        query_tokens_tuple=("play", "jazz", "songs"),
        query_token_count=3,
        query_sorted="jazz play songs",
        query_grams=frozenset(),
        query_slot_tokens=frozenset(["jazz"]),  # conflicts with candidate's "rock"
        query_has_number=False,
        query_numbers=set(),
        bm25_ref=None,
        max_raw_score=0.0,
        positional_lookup={},
        positional_literal_tokens=None,
        short_fuzzy_query_tokens=frozenset(),
        non_entity_tokens=None,
        candidate_slot_tokens=None,
        slot_tokens_by_index={},
        min_confidence=0.5,
        normalized_context={},
        explicit_context_keys=frozenset(),
        wildcard_passed_set=frozenset(),
        rehydrated_cache={},
        intent_score_cache={},
        literal_analysis_cache={},
    )
    conflict_candidate = Candidate(text="play rock songs", intent_name="HassMediaSearchAndPlay")
    # cand_slot_tokens = {"songs"} IS present in query_tokens_tuple so cand_coverage = 1.0,
    # but query_slot_tokens = {"jazz"} is NOT in allowed_cand_tokens, so has_conflict = True.
    # Result: 1.0 * (0.8 + 0.2 * 0.0) = 0.8  ->  0.0 < 0.8 < 1.0
    conflict_penalty_from_check = _check_and_calculate_conflict_penalty(
        frozenset({"songs"}),  # cand_slot_tokens -- present in query tokens
        frozenset(),  # wildcard_tokens
        conflict_candidate,
        conflict_ctx,
    )

    # The penalty must be non-trivial (conflict reduces it below 1.0)
    assert 0.0 < conflict_penalty_from_check < 1.0

    # The penalty must reduce a combined score
    base_score = 0.8
    combined_score = base_score * conflict_penalty_from_check
    assert combined_score < base_score

    # _best_positional_score
    analysis = [
        ranking._LiteralVariantAnalysis(
            total_token_count=3,
            exact_match_tokens=frozenset({"exact"}),
            positional_hits=(frozenset(["a", "b"]),),
            positional_query_tokens=frozenset({"a", "b"}),
            positional_match_count=1,
            requires_unique_alignment=False,
        )
    ]
    assert _best_positional_score(analysis, frozenset(["a", "b"])) == 1.0 / 3.0
    assert _best_positional_score(analysis, frozenset(["a"])) > 1.0 / 3.0
    assert _best_positional_score([], frozenset()) == 0.0

    # Dummy ScoringContext setup
    ctx = _ScoringContext(
        query="test query",
        query_normalized="test query",
        query_tokens=frozenset(["test", "query"]),
        query_tokens_tuple=("test", "query"),
        query_token_count=2,
        query_sorted="query test",
        query_grams=frozenset(),
        query_slot_tokens=frozenset(["slot1"]),
        query_has_number=False,
        query_numbers=set(),
        bm25_ref=None,
        max_raw_score=0.0,
        positional_lookup={},
        positional_literal_tokens=None,
        short_fuzzy_query_tokens=frozenset(),
        non_entity_tokens=None,
        candidate_slot_tokens=None,
        slot_tokens_by_index={},
        min_confidence=0.5,
        normalized_context={},
        explicit_context_keys=frozenset(),
        wildcard_passed_set=frozenset(),
        rehydrated_cache={},
        intent_score_cache={},
        literal_analysis_cache={},
    )

    candidate_dummy = Candidate(text="dummy", intent_name="dummy")
    assert (
        _check_and_calculate_conflict_penalty(
            frozenset(), frozenset(["slot1"]), candidate_dummy, ctx
        )
        == 1.0
    )

    ctx_empty_query_slots = replace(ctx, query_slot_tokens=frozenset())
    assert (
        _check_and_calculate_conflict_penalty(
            frozenset(["slot1"]), frozenset(), candidate_dummy, ctx_empty_query_slots
        )
        == 1.0
    )

    # Test _calculate_slot_penalty short-circuit
    simple_candidate = Candidate(
        text="turn on lights",
        intent_name="HassTurnOn",
        metadata={},
    )
    penalty, wildcard_toks, cand_toks = _calculate_slot_penalty(
        idx=0,
        candidate=simple_candidate,
        context=ctx,
        rehydrated=None,
    )
    assert penalty == 1.0
    assert wildcard_toks == frozenset()
    assert cand_toks == frozenset()

    # Test _get_wildcard_slot_tokens with rehydrated candidate
    register_custom_wildcards_from_sources(
        "en",
        {"custom": {"lists": {"song": {"wildcard": True}}}},
    )
    wildcard_candidate = Candidate(
        text="play {song}",
        intent_name="HassMediaSearchAndPlay",
        language="en",
        metadata={"sentence_template": "play {song}", "wildcard_slots": "song"},
    )
    rehydrated = ("play thriller", {"song": "thriller"})
    ctx_wildcard = replace(
        ctx,
        candidate_slot_tokens=(frozenset(["song"]),),
        slot_tokens_by_index={0: frozenset(["song"])},
    )

    cand_slot, wildcard_toks, _leading = _get_wildcard_slot_tokens(
        idx=0,
        candidate=wildcard_candidate,
        context=ctx_wildcard,
        rehydrated=rehydrated,
    )
    assert "thriller" in wildcard_toks
    assert "song" not in cand_slot


def test_registry_slot_token_fuzzy_match_short_token_gate() -> None:
    """Block stopword-length bleed while keeping non-literal short-typo anchors.

    ``la``/``loa`` is the FR article vs VI alias cross-language bleed; it must
    stay blocked because ``la`` is a grammar literal. ``fn``/``fan`` is a
    truncated target with no literal reading and must keep its anchor.
    """
    assert not ranking._registry_slot_token_fuzzy_match("loa", "la", frozenset())
    assert not ranking._registry_slot_token_fuzzy_match("fan", "fn", frozenset())
    assert ranking._registry_slot_token_fuzzy_match("fan", "fn", frozenset({"fn"}))
    # The slot side always requires three characters, even for exempt tokens.
    assert not ranking._registry_slot_token_fuzzy_match("la", "lah", frozenset({"la", "lah"}))
    # Three-or-more-character pairs keep plain threshold behavior.
    assert ranking._registry_slot_token_fuzzy_match("het", "hhet", frozenset())


def test_short_nonliteral_query_tokens_excludes_grammar_words() -> None:
    """Only sub-three-character tokens absent from corpus literals are exempt."""
    literals = frozenset({"allume", "la", "lampe", "turn", "off"})
    tokens = frozenset({"la", "fn", "off", "tivi"})
    assert ranking._short_nonliteral_query_tokens(tokens, literals) == frozenset({"fn"})
    assert ranking._short_nonliteral_query_tokens(tokens, None) == frozenset()


def test_prepare_rank_query_state_preserves_missing_literal_inventory() -> None:
    """Keep strict short-token gating when no literal inventory was supplied."""
    missing_literals = ranking._prepare_rank_query_state("fn", None)
    empty_literals = ranking._prepare_rank_query_state("fn", frozenset())

    assert missing_literals.short_fuzzy_tokens == frozenset()
    assert empty_literals.short_fuzzy_tokens == frozenset({"fn"})


def test_query_unknown_content_tokens_accepts_slot_value_typos() -> None:
    """A bounded typo of an evidenced slot value is not unknown vocabulary.

    ``query_slot_tokens`` records the slot-side spelling of fuzzy matches, so
    the guard must apply the same fuzzy matcher in reverse; otherwise a
    one-edit typo in a registry-only target word would suppress context
    disambiguation for the whole query.
    """
    query_tokens = frozenset({"turn", "on", "the", "lightt"})
    literals = frozenset({"turn", "on", "the"})
    unknown = ranking._query_unknown_content_tokens(
        query_tokens,
        literals,
        {},
        frozenset({"light"}),
    )
    assert unknown == frozenset()

    genuinely_unknown = ranking._query_unknown_content_tokens(
        frozenset({"turn", "on", "the", "teloche"}),
        literals,
        {},
        frozenset({"light"}),
    )
    assert genuinely_unknown == frozenset({"teloche"})


def test_has_unanchored_entity_slot_short_typo_exemption() -> None:
    """A truncated entity name anchors only when exempt from the literal gate."""
    slots = {"name": "fan"}
    query = ("turn", "off", "fn")
    assert ranking._has_unanchored_entity_slot(slots, frozenset(), query)
    assert not ranking._has_unanchored_entity_slot(slots, frozenset(), query, frozenset({"fn"}))
    # The FR article never anchors the VI alias regardless of exemptions
    # computed from FR grammar literals (which always contain "la").
    assert ranking._has_unanchored_entity_slot(
        {"name": "loa"}, frozenset(), ("allume", "la", "lampe")
    )


def test_chars_conflict_in_marks_tone_semantics() -> None:
    """Disjoint explicit marks over one base conflict; shared or absent marks do not."""
    assert ranking._chars_conflict_in_marks("ó", "ộ")
    assert not ranking._chars_conflict_in_marks("á", "ắ")
    assert not ranking._chars_conflict_in_marks("é", "e")
    assert not ranking._chars_conflict_in_marks("a", "b")


def test_is_fuzzy_slot_token_match_rejects_conflicting_tone_substitution() -> None:
    """A single substitution between deliberately distinct tones is not a typo."""
    assert not ranking._is_fuzzy_slot_token_match("bóng", "bộng")
    assert not ranking._is_fuzzy_slot_token_match("phóng", "phộng")
    # A marked character against its bare base stays a plausible slip.
    assert ranking._is_fuzzy_slot_token_match("phóng", "phong")


def test_rank_candidates_unknown_filler_pins_context_suppression() -> None:
    """Document the conservative trade-off for out-of-corpus filler tokens.

    An unknown token that appears in no candidate suppresses intent-context
    boosts even when the target itself is resolvable. The delegated text
    retains live context, so area resolution still happens downstream; this
    test pins the ranking-level semantics so any future change is deliberate.
    """
    generic = Candidate(
        text="turn on the light",
        intent_name="HassTurnOn",
        language="en",
        metadata={
            "slots": '{"domain":"light"}',
            "static_slots": "domain",
            "literal_text": "turn on the light",
        },
    )
    kitchen = Candidate(
        text="turn on the kitchen light",
        intent_name="HassTurnOn",
        language="en",
        metadata={
            "slots": '{"domain":"light","area":"Kitchen"}',
            "static_slots": "domain",
            "literal_text": "turn on the light",
        },
    )

    with_context = rank_candidates(
        "euhh turn on the light",
        [generic, kitchen],
        max_candidates=2,
        language="en",
        intent_context={"area": {"value": "Kitchen", "text": "Kitchen"}},
    )
    without_context = rank_candidates(
        "euhh turn on the light",
        [generic, kitchen],
        max_candidates=2,
        language="en",
    )

    assert [item.candidate for item in with_context] == [item.candidate for item in without_context]
    assert [item.scores.final_score for item in with_context] == pytest.approx(
        [item.scores.final_score for item in without_context]
    )


def test_prepare_bm25_scoring_rescales_dynamic_scores_to_reference_max() -> None:
    """Normalize weak dynamic sets by the stronger reference-corpus match."""
    candidates = [
        Candidate(text="kitchen radio", intent_name="HassTurnOn", language="en"),
    ]
    reference_bm25_index = BM25Index.from_normalized_texts(
        ("turn on the kitchen light", "turn off the hall lamp"),
    )
    query_tokens = ("turn", "on", "the", "kitchen", "light")

    scoring = ranking._prepare_bm25_scoring(
        candidates,
        query_tokens,
        None,
        reference_bm25_index,
    )

    reference_max_raw = max(
        reference_bm25_index.raw_scores_sparse(query_tokens).values(),
    )
    dynamic_raw = reference_bm25_index.raw_score_tokens(
        candidates[0].normalized_tokens,
        query_tokens,
    )
    assert scoring.max_raw_score == reference_max_raw
    assert scoring.score_at(0) == pytest.approx(dynamic_raw / reference_max_raw)
    assert scoring.score_at(0) < 1.0


def test_multiwildcard_partial_rehydration_fails_closed() -> None:
    """Never leak a literal placeholder token from a partly aligned template."""
    register_custom_wildcards_from_sources(
        "de",
        {
            "custom": {
                "lists": {
                    "timer_name": {"wildcard": True},
                    "timer_duration": {"wildcard": True},
                }
            }
        },
    )
    candidate = Candidate(
        text="timer timer_name auf timer_duration minuten",
        intent_name="HassTimerSet",
        language="de",
        metadata={
            "sentence_template": "timer {timer_name} auf {timer_duration} minuten",
            "wildcard_slots": "timer_name,timer_duration",
            "slots": '{"timer_name":"timer_name","timer_duration":"timer_duration"}',
        },
        slot_values=("timer_name", "timer_duration"),
    )

    rehydrated_text, replacements = get_wildcard_rehydration(
        candidate,
        "timer Nudeln kochen auf zehn Minuten",
    )

    assert rehydrated_text == candidate.text
    assert replacements == {}
