"""Stability and robustness checks for Assist Canonicalizer core modules.

Verifies correct behavior, error handling, and fallback operations under
atypical input conditions, missing dependencies, and external errors.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

import custom_components.assist_canonicalizer.config_flow as cf
from custom_components.assist_canonicalizer.bm25 import BM25Index
from custom_components.assist_canonicalizer.candidate import (
    Candidate,
    candidate_dedupe_preference_key,
)
from custom_components.assist_canonicalizer.ranking import (
    _non_entity_coverage,
    _positional_intent_score_from_lookup,
    _positional_similarity,
    _prefilter_wildcard_candidates,
    _rehydrate_and_rescore_wildcard,
    _rehydrated_bm25_score,
    rank_candidates,
    token_count_ratio,
)
from custom_components.assist_canonicalizer.rehydration import (
    _align_prefix_boundary,
    _align_suffix_boundary,
    _extract_original_span,
    _replace_wildcard_in_original,
    get_wildcard_rehydration,
    rehydrate_wildcard_slots,
)
from custom_components.assist_canonicalizer.utils import (
    normalize_language,
    register_custom_wildcards_from_sources,
    wildcard_slot_names,
    wildcard_slot_names_sorted,
)


def test_normalize_language_stability() -> None:
    """Verify that normalize_language fails safely on invalid input formats."""
    with pytest.raises(ValueError, match="Language must be a non-empty string"):
        normalize_language(None)

    with pytest.raises(ValueError, match="Language must be a non-empty string"):
        normalize_language(123)

    with pytest.raises(ValueError, match="Language must not be empty"):
        normalize_language("")

    with pytest.raises(ValueError, match="Language must not be empty"):
        normalize_language("   ")


def test_wildcard_slot_names_import_error_stability() -> None:
    """Verify that wildcard_slot_names handles missing home_assistant_intents dependency."""
    with patch.dict(sys.modules, {"home_assistant_intents": None}):
        wildcard_slot_names.cache_clear()
        try:
            assert wildcard_slot_names("en") == frozenset()
        finally:
            wildcard_slot_names.cache_clear()


def test_wildcard_slot_names_cache_clear_can_scope_custom_language() -> None:
    """Scoped cache clearing should not drop other languages' custom wildcards."""
    wildcard_slot_names.cache_clear()
    try:
        register_custom_wildcards_from_sources(
            "xx-one",
            {"custom": {"lists": {"one_custom_wildcard": {"wildcard": True}}}},
        )
        register_custom_wildcards_from_sources(
            "xx-two",
            {"custom": {"lists": {"two_custom_wildcard": {"wildcard": True}}}},
        )

        assert "one_custom_wildcard" in wildcard_slot_names("xx-one")
        assert "two_custom_wildcard" in wildcard_slot_names("xx-two")

        wildcard_slot_names.cache_clear("xx-one")

        assert "one_custom_wildcard" not in wildcard_slot_names("xx-one")
        assert "two_custom_wildcard" in wildcard_slot_names("xx-two")
    finally:
        wildcard_slot_names.cache_clear()


def test_wildcard_slot_names_exceptions_and_types_stability() -> None:
    """Verify that wildcard_slot_names isolates external intents parsing errors."""
    pytest.importorskip("home_assistant_intents")
    import home_assistant_intents as intents_module

    # Exception raised during intents parsing
    wildcard_slot_names.cache_clear()
    with patch.object(intents_module, "get_intents", side_effect=Exception("mock error")):
        try:
            assert wildcard_slot_names("en") == frozenset()
        finally:
            wildcard_slot_names.cache_clear()

    # Returned intents data is not a dict
    wildcard_slot_names.cache_clear()
    with patch.object(intents_module, "get_intents", return_value="not-a-dict"):
        try:
            assert wildcard_slot_names("en") == frozenset()
        finally:
            wildcard_slot_names.cache_clear()

    # Returned lists structure is not a dict
    wildcard_slot_names.cache_clear()
    with patch.object(intents_module, "get_intents", return_value={"lists": []}):
        try:
            assert wildcard_slot_names("en") == frozenset()
        finally:
            wildcard_slot_names.cache_clear()

    # Exception in outer try block
    wildcard_slot_names.cache_clear()
    with patch.object(intents_module, "get_languages", side_effect=ValueError("bad")):
        try:
            assert wildcard_slot_names(None) == frozenset()
        finally:
            wildcard_slot_names.cache_clear()

    # Test sorted wrapper handles outer exceptions
    wildcard_slot_names_sorted.cache_clear()
    with patch.object(intents_module, "get_languages", side_effect=ValueError("bad")):
        try:
            assert wildcard_slot_names_sorted(None) == ()
        finally:
            wildcard_slot_names.cache_clear()
            wildcard_slot_names_sorted.cache_clear()


def test_candidate_literal_variants_recovers_from_corrupt_metadata() -> None:
    """Recover literal variants from literal_text when cached JSON is corrupt."""
    cand = Candidate(
        text="turn on light",
        intent_name="HassTurnOn",
        metadata={
            "literal_text": "turn on|switch on",
            "literal_variants": "{not-json",
        },
    )

    assert cand.literal_variants == (
        frozenset({"turn", "on"}),
        frozenset({"switch", "on"}),
    )


@pytest.mark.parametrize(
    "literal_variants",
    [
        '"turn on"',
        '{"tokens":["turn","on"]}',
        "42",
        '[["turn", 1]]',
        '["turn"]',
    ],
)
def test_candidate_literal_variants_recovers_from_malformed_metadata_shape(
    literal_variants: str,
) -> None:
    """Recover literal variants when cached JSON is valid but has the wrong shape."""
    cand = Candidate(
        text="turn on light",
        intent_name="HassTurnOn",
        metadata={
            "literal_text": "turn on|switch on",
            "literal_variants": literal_variants,
        },
    )

    assert cand.literal_variants == (
        frozenset({"turn", "on"}),
        frozenset({"switch", "on"}),
    )


def test_candidate_slot_tokens_recover_from_corrupt_metadata_slots() -> None:
    """Recover slot tokens from slot_values when serialized slot metadata is corrupt."""
    cand = Candidate(
        text="turn on kitchen light",
        intent_name="HassTurnOn",
        metadata={"slots": "{not-json"},
        slot_values=("kitchen light",),
    )

    assert cand.slot_tokens_set == frozenset({"kitchen", "light"})


def test_candidate_slot_tokens_recover_from_non_object_metadata_slots() -> None:
    """Recover slot tokens from slot_values when serialized slot metadata is not an object."""
    cand = Candidate(
        text="turn on kitchen light",
        intent_name="HassTurnOn",
        metadata={"slots": '["kitchen light"]'},
        slot_values=("kitchen light",),
    )

    assert cand.slot_tokens_set == frozenset({"kitchen", "light"})


def test_candidate_dedupe_preference_recovers_from_non_string_metadata_slots() -> None:
    """Ignore manually constructed non-string slot metadata while deduping."""
    cand = Candidate(
        text="turn on kitchen light",
        intent_name="HassTurnOn",
        metadata=cast(Any, {"slots": {"name": "kitchen light"}}),
    )

    assert candidate_dedupe_preference_key(cand) == (-2, 0, 0, 1)


def test_rehydration_no_wildcard_info_stability() -> None:
    """Verify rehydration returns original text when candidate has no wildcards."""
    cand = Candidate(text="turn on the light", intent_name="HassTurnOn", language="en")
    assert cand.wildcard_info is None
    res_text, res_slots = get_wildcard_rehydration(cand, "turn on the light")
    assert res_text == "turn on the light"
    assert res_slots == {}


def test_rehydration_boundary_and_replace_stability() -> None:
    """Verify alignment boundary edge cases and substring matching robustness."""
    # Pattern not found in original token inside _replace_wildcard_in_original
    res = _replace_wildcard_in_original("turn on light", 2, "lamp", "nonexistent")
    assert res == "turn on light"

    # Token that normalizes to nothing (punctuation)
    res2 = _replace_wildcard_in_original("turn , light", 1, "lamp", "light")
    assert res2 == "turn , lamp"

    # Suffix alignment failure (boundary is -1)
    cand = Candidate(
        text="add shopping_list_item to shopping list", intent_name="dummy", language="en"
    )
    res_text, res_slots = get_wildcard_rehydration(cand, "add milk in the kitchen")
    assert res_text == cand.text
    assert res_slots == {}

    # Prefix alignment boundary >= suffix boundary check
    assert _extract_original_span("test query", 2, 1) == ""


def test_rehydrate_wildcard_slots_stability() -> None:
    """Verify slots rehydration ignores empty slots or texts without placeholders."""
    # Slots dictionary is empty
    assert rehydrate_wildcard_slots({}, "add shopping_list_item", "add milk") == {}

    # No wildcard in candidate text
    slots = {"name": "kitchen"}
    assert rehydrate_wildcard_slots(slots, "turn on light", "turn on light") == slots

    # replacements is empty
    assert rehydrate_wildcard_slots(slots, "add shopping_list_item", "add milk") == slots


def test_align_prefix_boundary_stability() -> None:
    """Verify boundary search logic does not crash on empty prefixes."""
    assert _align_prefix_boundary((), ("a", "b")) == 0


def test_align_suffix_boundary_stability() -> None:
    """Verify boundary search logic does not crash on empty suffixes."""
    assert _align_suffix_boundary((), ("a", "b"), 0) == 2


def test_ranking_stability() -> None:
    """Verify ranking scoring components return stable fallbacks on empty or mismatch inputs."""
    # token_count_ratio: query or candidate empty
    assert token_count_ratio("", "candidate") == 0.0
    assert token_count_ratio("query", "") == 0.0

    # _positional_similarity: identical string comparison matches early
    assert _positional_similarity("", "") == 1.0

    # _positional_intent_score_from_lookup: variants is empty
    assert _positional_intent_score_from_lookup("   ", frozenset(), {}) == 1.0

    # _non_entity_coverage: non_entity is empty
    assert _non_entity_coverage(frozenset({"a"}), frozenset({"a"}), frozenset({"a"})) == 1.0

    # _rehydrated_bm25_score: max_raw_score <= 0.0
    bm_index = BM25Index.from_normalized_texts(("doc1",))
    assert _rehydrated_bm25_score(("doc1",), ("query",), bm_index, 0.0) == 0.0
    assert _rehydrated_bm25_score(("doc1",), ("query",), bm_index, -1.0) == 0.0


def test_prefilter_wildcard_candidates_stability() -> None:
    """Verify prefiltering candidate selector defaults cleanly on empty lookups."""
    # variants_with_len is None when wildcard_always_passes is active
    cands = [Candidate(text="add shopping_list_item", intent_name="dummy", language="en")]
    res = _prefilter_wildcard_candidates(
        candidates=cands,
        query_tokens=frozenset({"add"}),
        wildcard_always_passes=frozenset({0}),
        wildcard_variants_with_len=None,
        wildcard_token_to_indices=None,
    )
    assert res == {0}

    # wildcard_always_passes is None (hits the else block)
    cands_no_wc = [Candidate(text="turn on light", intent_name="dummy", language="en")]
    res_no_wc = _prefilter_wildcard_candidates(
        candidates=cands_no_wc,
        query_tokens=frozenset({"turn"}),
        wildcard_always_passes=None,
        wildcard_variants_with_len=None,
        wildcard_token_to_indices=None,
    )
    assert res_no_wc == set()

    # Candidate with wildcard but disjoint literals
    cand_wc = Candidate(
        text="spiel den search_querypodcast",
        intent_name="dummy",
        language="de",
        metadata={"literal_text": "spiel|den"},
    )
    res_disjoint = _prefilter_wildcard_candidates(
        candidates=[cand_wc],
        query_tokens=frozenset({"abc", "def"}),
        wildcard_always_passes=None,
        wildcard_variants_with_len=None,
        wildcard_token_to_indices=None,
    )
    assert res_disjoint == set()


def test_prefilter_wildcard_candidates_skips_impossible_precomputed_match() -> None:
    """Skip wildcard variant scans when query overlap cannot meet the required hits."""
    cands = [Candidate(text="add item", intent_name="dummy", language="en")]
    variants: dict[int, tuple[tuple[frozenset[str], int, int], ...]] = {
        0: (
            (frozenset({"add", "to", "list"}), 3, 3),
            (frozenset({"put", "on", "list"}), 3, 3),
        )
    }

    skipped = _prefilter_wildcard_candidates(
        candidates=cands,
        query_tokens=frozenset({"add"}),
        wildcard_always_passes=frozenset(),
        wildcard_variants_with_len=variants,
        wildcard_token_to_indices={"add": (0,)},
        wildcard_literal_tokens_by_index={0: frozenset({"add", "to", "list", "put", "on"})},
        wildcard_min_required_by_index={0: 3},
    )
    assert skipped == set()

    matched = _prefilter_wildcard_candidates(
        candidates=cands,
        query_tokens=frozenset({"add", "to", "list"}),
        wildcard_always_passes=frozenset(),
        wildcard_variants_with_len=variants,
        wildcard_token_to_indices={"add": (0,), "to": (0,), "list": (0,)},
        wildcard_literal_tokens_by_index={0: frozenset({"add", "to", "list", "put", "on"})},
        wildcard_min_required_by_index={0: 3},
    )
    assert matched == {0}


def test_prefilter_wildcard_candidates_fallback_when_reverse_index_missing() -> None:
    """Verify fallback to on-the-fly wildcard scanning when reverse index is missing."""
    with patch(
        "custom_components.assist_canonicalizer.candidate.wildcard_slot_names_sorted",
        return_value=("shopping_list_item",),
    ):
        cands = [Candidate(text="add shopping_list_item", intent_name="dummy", language="en")]
        # If wildcard_always_passes is not None but wildcard_token_to_indices is missing (None),
        # we should fall back to on-the-fly scanning.
        res = _prefilter_wildcard_candidates(
            candidates=cands,
            query_tokens=frozenset({"add"}),
            wildcard_always_passes=frozenset(),
            wildcard_variants_with_len=None,
            wildcard_token_to_indices=None,
        )
        assert res == {0}


def test_prefilter_wildcard_candidates_fallback_when_precompute_bundle_incomplete() -> None:
    """Use on-the-fly wildcard filtering when any precomputed structure is missing."""
    with patch(
        "custom_components.assist_canonicalizer.candidate.wildcard_slot_names_sorted",
        return_value=("shopping_list_item",),
    ):
        cands = [
            Candidate(
                text="add shopping_list_item",
                intent_name="dummy",
                language="en",
                metadata={"literal_text": "add"},
            )
        ]
        res = _prefilter_wildcard_candidates(
            candidates=cands,
            query_tokens=frozenset({"add"}),
            wildcard_always_passes=frozenset(),
            wildcard_variants_with_len={0: ((frozenset({"other"}), 1, 1),)},
            wildcard_token_to_indices={"add": (0,)},
            wildcard_literal_tokens_by_index=None,
            wildcard_min_required_by_index={0: 1},
        )

    assert res == {0}


def test_rehydrate_and_rescore_wildcard_no_replacements_stability() -> None:
    """Verify candidate rescoring returns original scores if query cannot align."""
    cand = Candidate(text="add shopping_list_item to list", intent_name="dummy", language="en")
    res_text, res_repl, c_score, b_score = _rehydrate_and_rescore_wildcard(
        candidate=cand,
        query="add",
        query_tokens_tuple=("add",),
        query_grams=frozenset(),
        bm25_ref=None,
        max_raw_score=1.0,
        original_char_score=0.5,
        original_bm25_score=0.5,
    )
    assert res_text is None
    assert res_repl is None
    assert c_score == 0.5
    assert b_score == 0.5


def test_rank_candidates_argument_bounds_stability() -> None:
    """Verify rank_candidates enforces positive bounds for search limit sizes."""
    with pytest.raises(ValueError, match="max_candidates must be positive"):
        rank_candidates("query", [], max_candidates=0)
    with pytest.raises(
        ValueError, match="rapidfuzz_prefilter_candidates must be at least max_candidates"
    ):
        rank_candidates("query", [], max_candidates=5, rapidfuzz_prefilter_candidates=3)


def test_bm25_stability() -> None:
    """Verify BM25 index handles empty sets and invalid parameter ranges gracefully."""
    # Construct an empty BM25 index to test average_length == 0
    empty_index = BM25Index.from_normalized_texts(())
    assert empty_index.raw_scores(("query",)) == []
    assert empty_index.raw_score_tokens(("doc",), ("query",)) == 0.0

    # Test score_custom_documents with empty queries/docs
    assert empty_index.score_custom_documents("query", ()) == ()
    assert empty_index.score_custom_documents("", ("doc",)) == (0.0,)

    # Test invalid parameter checks
    full_index = BM25Index.from_normalized_texts(("doc1",))
    with pytest.raises(ValueError, match="k1 must be positive"):
        full_index.score_custom_documents("query", ("doc1",), k1=-1.0)
    with pytest.raises(ValueError, match="b must be between 0 and 1"):
        full_index.score_custom_documents("query", ("doc1",), b=1.5)


def test_config_flow_available_fallback_agents_filtering_stability(
    fallback_agent_manager_factory: Any,
    mock_conversation_entity_type: type,
) -> None:
    """Verify fallback agent choices filter unavailable, entity, or excluded agents."""

    class MockEntity:
        """Mock Entity representation for test purposes."""

        def __init__(
            self,
            entity_id: str | None,
            name: str | None = None,
            unique_id: str | None = None,
            registry_entry: Any = None,
        ) -> None:
            """Initialize mock entity attributes."""
            self.entity_id = entity_id
            self.name = name
            self.unique_id = unique_id
            self.registry_entry = registry_entry

    class MockState:
        """Mock State representation for test purposes."""

        def __init__(self, name: str) -> None:
            """Initialize mock state."""
            self.name = name

    class MockStates:
        """Mock States collection for test purposes."""

        def get(self, entity_id: str) -> MockState | None:
            """Get state representation by entity_id."""
            if entity_id == "conversation.state_name":
                return MockState("State Name Agent")
            return None

    hass = MagicMock()
    hass.states = MockStates()

    entity_component = MagicMock()
    entity_component.entities = [
        MockEntity(None),
        MockEntity(""),
        MockEntity("conversation.excluded", unique_id="conversation.excluded"),
        MockEntity("conversation.state_name", name="Ignored Name"),
        MockEntity(
            "conversation.registered",
            name="Registered Agent",
            registry_entry=MagicMock(config_entry_id="conversation.excluded"),
        ),
    ]
    hass.data = {cf.DATA_COMPONENT: entity_component}

    manager = fallback_agent_manager_factory(
        [
            SimpleNamespace(id=None, name="No ID"),
            SimpleNamespace(id="conversation.excluded", name="Excluded"),
            SimpleNamespace(id="conversation.entry_agent", name="Entry Agent"),
            SimpleNamespace(id="conversation.entity_agent", name="Entity Agent"),
        ],
        {"conversation.entity_agent": mock_conversation_entity_type()},
    )

    with (
        patch(
            "custom_components.assist_canonicalizer.config_flow.get_agent_manager",
            return_value=manager,
        ),
        patch(
            "custom_components.assist_canonicalizer.config_flow.ConversationEntity",
            mock_conversation_entity_type,
        ),
    ):
        choices = cf._available_fallback_agents(hass, "conversation.excluded")

        assert "conversation.state_name" in choices
        assert choices["conversation.state_name"] == "State Name Agent"
        assert "conversation.entry_agent" in choices
        assert choices["conversation.entry_agent"] == "Entry Agent"
        assert "conversation.registered" not in choices
        assert "conversation.entity_agent" not in choices
