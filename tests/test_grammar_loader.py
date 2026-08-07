"""Tests for automatic conversation intent candidate loading."""

import re
from collections.abc import Iterator, Mapping
from typing import Any
from unittest.mock import patch

import hassil
import hassil.errors
import orjson
import pytest

from custom_components.assist_canonicalizer import grammar_loader as gl
from custom_components.assist_canonicalizer.builtin_intents import load_language_intent_sources
from custom_components.assist_canonicalizer.candidate import CandidateSource
from custom_components.assist_canonicalizer.grammar_loader import (
    build_candidates_from_intent_sources,
    build_query_registry_candidates,
    build_registry_slot_index,
    compile_dynamic_registry_intents,
    expand_sentence_template,
    is_fixed_sentence,
)
from custom_components.assist_canonicalizer.normalization import normalize_text
from custom_components.assist_canonicalizer.rehydration import get_wildcard_rehydration
from custom_components.assist_canonicalizer.utils import (
    register_custom_wildcards_from_sources,
    wildcard_slot_names,
)

_HASSIL_MISSING_LIST_RE = re.compile(r"\{([^}]+)\}")


def _make_hassil_slot_lists(slots: Mapping[str, tuple[str, ...]]) -> dict[str, Any]:
    """Build HassIL slot lists from registry slot values."""
    return {
        slot_name: hassil.TextSlotList(
            name=slot_name,
            values=[
                hassil.TextSlotValue(
                    text_in=hassil.parse_sentence(value).expression,
                    value_out=value,
                )
                for value in values
            ],
        )
        for slot_name, values in slots.items()
    }


def _run_hassil_recognize_all(
    query: str,
    intents: hassil.intents.Intents,
    slot_lists: Mapping[str, Any],
) -> list[Any]:
    """Run HassIL recognition while stubbing unrelated missing lists."""
    working_lists = dict(slot_lists)
    stubbed = set()
    while True:
        try:
            return list(hassil.recognize_all(query, intents, slot_lists=working_lists))
        except hassil.errors.MissingListError as err:
            match = _HASSIL_MISSING_LIST_RE.search(str(err))
            if match is None:
                raise
            list_name = match.group(1)
            if list_name in stubbed:
                raise
            stubbed.add(list_name)
            working_lists[list_name] = hassil.TextSlotList(list_name, [])


def test_registry_slot_index_inverted_cache_validates_tuple_identity() -> None:
    """Do not reuse an inverted lookup built for a different records tuple."""
    index = build_registry_slot_index(
        {
            "name": ("kitchen light",),
            "entity": ("bathroom fan",),
        },
        "en",
    )
    name_records = index["name"]
    entity_records = index["entity"]
    stale_lookup = index.get_inverted_for_records(name_records)
    index._inverted_cache[id(entity_records)] = (name_records, stale_lookup)

    lookup = index.get_inverted_for_records(entity_records)

    assert lookup is not stale_lookup
    assert [record.text for record in lookup["bathroom"]] == ["bathroom fan"]
    assert "kitchen" not in lookup


def test_list_range_endpoints_uses_consistent_defaults_for_malformed_data() -> None:
    """Use the normal missing-endpoint defaults when range data is malformed."""
    assert gl._list_range_endpoints({}) == (0, 100)
    assert gl._list_range_endpoints(None) == (0, 100)


def test_registry_slot_index_skips_inverted_cache_for_scoped_records() -> None:
    """Do not retain query-scoped registry record tuples in the index cache."""
    index = build_registry_slot_index(
        {
            "name:light": tuple(
                f"kitchen light {position}"
                for position in range(gl.DEFAULT_MAX_CANDIDATES_PER_INTENT + 1)
            ),
        },
        "en",
    )
    indexed_records = index["name:light"]
    scoped_records = indexed_records[: gl.DEFAULT_MAX_CANDIDATES_PER_INTENT]

    assert scoped_records is not indexed_records

    lookup = index.get_inverted_for_records(scoped_records)

    assert "kitchen" in lookup
    assert id(scoped_records) not in index._inverted_cache


def test_domain_scoped_registry_retrieval_reaches_tail_records() -> None:
    """Search the complete domain index while keeping query-time scoring bounded."""
    values = (
        *(f"Hallway light {position}" for position in range(gl.DEFAULT_MAX_CANDIDATES_PER_INTENT)),
        "Garden beacon",
    )
    index = build_registry_slot_index({"name:light": values}, "en")
    records = gl._scoped_registry_slot_records("name", ("light",), index, {})
    stats = gl.RegistryRetrievalStats()
    query = normalize_text("turn on the Garden becon")

    relevant = gl._query_relevant_precomputed_slot_values(
        records,
        query,
        frozenset(query.split()),
        registry_slot_index=index,
        retrieval_stats=stats,
    )

    assert records is index["name:light"]
    assert relevant == ("Garden beacon",)
    assert stats.values_scored <= gl.DEFAULT_MAX_REGISTRY_VALUES_SCORED_PER_QUERY


def test_multi_domain_registry_scope_reuses_complete_sparse_index() -> None:
    """Cache merged domain records and their postings across query-local scopes."""
    index = build_registry_slot_index(
        {
            "name:light": ("Shared target", "Kitchen light"),
            "name:switch": ("Shared target", "Garden beacon"),
        },
        "en",
    )

    first = gl._scoped_registry_slot_records("name", ("light", "switch"), index, {})
    lookup = index.get_inverted_for_records(first)
    second = gl._scoped_registry_slot_records("name", ("light", "switch"), index, {})

    assert tuple(record.text for record in first) == (
        "Shared target",
        "Kitchen light",
        "Garden beacon",
    )
    assert second is first
    assert lookup["garden"][0].text == "Garden beacon"
    assert id(first) in index._inverted_cache


def test_registry_slot_index_recovers_bounded_multitoken_typos() -> None:
    """Retrieve a registry target when every query target token has one edit."""
    index = build_registry_slot_index(
        {"name": ("Bedroom Window", "Bedroom Light")},
        "en",
    )
    query = normalize_text("turn off the bedrom windw")

    relevant = gl._query_relevant_precomputed_slot_values(
        index["name"],
        query,
        frozenset(query.split()),
        registry_slot_index=index,
    )

    assert relevant == ("Bedroom Window",)


def test_registry_slot_index_matches_spaced_typo_to_compound_alias() -> None:
    """Align a spaced query phrase with a one-edit compound registry alias."""
    index = build_registry_slot_index(
        {"name": ("Badkamerventilator", "Woonkamerventilator")},
        "nl",
    )
    query = normalize_text("badkamer ventillator uitzetten")

    relevant = gl._query_relevant_precomputed_slot_values(
        index["name"],
        query,
        frozenset(query.split()),
        registry_slot_index=index,
    )

    assert relevant == ("Badkamerventilator",)


def test_registry_slot_index_does_not_expand_past_fully_anchored_value() -> None:
    """Keep fuzzy neighbors from disturbing an already complete registry match."""
    index = build_registry_slot_index(
        {"name": ("Light Salon", "Bright Salon")},
        "en",
    )
    query = normalize_text("make light salon brigt")

    relevant = gl._query_relevant_precomputed_slot_values(
        index["name"],
        query,
        frozenset(query.split()),
        registry_slot_index=index,
    )

    assert relevant == ("Light Salon",)


def test_registry_slot_index_does_not_treat_exact_generic_alias_as_specific() -> None:
    """Let a one-token generic alias coexist with a longer fuzzy target."""
    index = build_registry_slot_index(
        {"name": ("Quạt", "Quạt phòng tắm")},
        "vi",
    )
    query = normalize_text("tắt quạt phòg tắh")

    relevant = gl._query_relevant_precomputed_slot_values(
        index["name"],
        query,
        frozenset(query.split()),
        registry_slot_index=index,
    )

    assert relevant == ("Quạt", "Quạt phòng tắm")


def test_registry_slot_index_requires_full_token_boundaries_for_exact_precedence() -> None:
    """Do not let a partial-token substring suppress a complete fuzzy target."""
    index = build_registry_slot_index(
        {"name": ("Room Vent", "Bedroom Vant")},
        "en",
    )
    query = normalize_text("turn off bedroom vent")

    relevant = gl._query_relevant_precomputed_slot_values(
        index["name"],
        query,
        frozenset(query.split()),
        registry_slot_index=index,
    )

    assert relevant == ("Bedroom Vant",)


def test_query_registry_candidates_render_fuzzy_registry_target() -> None:
    """Generate the correct action candidate for a typoed registry entity."""
    slots = {"name": ("Bedroom Window", "Bedroom Light")}
    sources = {
        "builtin": {
            "intents": {
                "HassTurnOff": {"data": [{"sentences": ["turn off {name}"]}]},
            }
        }
    }

    candidates = build_query_registry_candidates(
        "en",
        sources,
        slots,
        "turn off the bedrom windw",
        registry_slot_index=build_registry_slot_index(slots, "en"),
    )

    assert any(candidate.text == "turn off Bedroom Window" for candidate in candidates)


def test_is_fixed_sentence_rejects_hassil_templates() -> None:
    """Reject sentence templates that need slot expansion."""
    assert is_fixed_sentence("turn on kitchen light")
    assert not is_fixed_sentence("turn on {name}")
    assert not is_fixed_sentence("turn [the] light on")


def test_expand_sentence_template_supports_hassil_permutations() -> None:
    """Expand semicolon-separated HassIL permutations without literal semicolons."""
    expanded = expand_sentence_template(
        "(off;{item})",
        {"item": ("groceries",)},
        {},
    )

    assert expanded == ("off groceries", "groceries off")


def test_expand_sentence_template_caps_multi_branch_hassil_permutations() -> None:
    """Cap multi-branch permutations while allowing later branch orders through."""
    expanded = expand_sentence_template(
        "(off;{item};now)",
        {"item": ("groceries", "lights", "fan")},
        {},
        max_expansions=4,
    )

    assert expanded == (
        "off groceries now",
        "off lights now",
        "off fan now",
        "off now groceries",
    )


def test_build_candidates_uses_text_list_output_values() -> None:
    """Use HassIL text-list outputs for slot metadata while expanding inputs."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "lists": {
                    "volume_step": {
                        "values": [
                            {"in": "(up|increase)", "out": "up"},
                            {"in": "(down|decrease)", "out": "down"},
                        ]
                    }
                },
                "intents": {
                    "HassSetVolumeRelative": {"data": [{"sentences": ["volume {volume_step}"]}]}
                },
            }
        },
    )

    slots_by_text = {
        candidate.text: orjson.loads(candidate.metadata["slots"]) for candidate in candidates
    }
    assert slots_by_text["volume up"]["volume_step"] == "up"
    assert slots_by_text["volume increase"]["volume_step"] == "up"
    assert slots_by_text["volume down"]["volume_step"] == "down"
    assert "volume (up|increase)" not in slots_by_text


def test_slot_lists_use_hassil_whole_list_override_precedence() -> None:
    """Layered same-name lists should replace whole lists like HassIL."""
    source_config = {
        "lists": {
            "level": {
                "values": [
                    {"in": "low", "out": 10},
                    {"in": "medium", "out": 50},
                ]
            }
        }
    }
    intent_config = {
        "lists": {
            "level": {
                "values": [
                    {"in": "medium", "out": 55},
                    {"in": "high", "out": 90},
                ]
            }
        }
    }
    data_item = {"lists": {"level": {"values": [{"in": "high", "out": 100}]}}}

    assert gl._slot_values(source_config, intent_config, data_item)["level"] == ("high",)
    assert gl._slot_output_value_maps(source_config, intent_config, data_item)["level"] == {
        "high": 100
    }


def test_build_candidates_records_context_slots() -> None:
    """Expose HassIL context-provided slots as candidate metadata."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassSetVolumeRelative": {
                        "data": [
                            {
                                "sentences": ["volume up"],
                                "requires_context": {"area": {"slot": True}},
                            }
                        ]
                    }
                }
            }
        },
    )

    assert candidates[0].metadata["context_slots"] == "area"


def test_build_candidates_from_intent_sources_uses_fixed_sentences_only() -> None:
    """Build custom candidates from Home Assistant conversation source configs."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "config": {
                "intents": {
                    "CustomIntent": {
                        "data": [
                            {
                                "sentences": [
                                    "activate movie mode",
                                    "turn on {name}",
                                ]
                            }
                        ]
                    }
                }
            }
        },
    )
    assert len(candidates) == 1
    assert candidates[0].text == "activate movie mode"
    assert candidates[0].intent_name == "CustomIntent"
    assert candidates[0].source == CandidateSource.CUSTOM_SENTENCE


def test_expand_sentence_template_uses_lists_optional_groups_and_alternatives() -> None:
    """Expand bounded template syntax into spoken candidate texts."""
    expanded = expand_sentence_template(
        "turn [the] {name} (on|off)",
        {"name": ("kitchen light", "desk lamp")},
        {},
        max_expansions=10,
    )
    assert set(expanded) == {
        "turn kitchen light on",
        "turn desk lamp on",
        "turn kitchen light off",
        "turn desk lamp off",
        "turn the kitchen light on",
        "turn the desk lamp on",
        "turn the kitchen light off",
        "turn the desk lamp off",
    }


def test_expand_sentence_template_fair_cap_preserves_later_alternative_branches() -> None:
    """Fair capped expansion should not starve later action alternatives."""
    many_tails = "|".join(f"tail{i}" for i in range(20))
    expanded = expand_sentence_template(
        "(first|second) {item} [<tail>]",
        {"item": ("value",)},
        {"tail": f"({many_tails})"},
        max_expansions=6,
        fair=True,
    )

    assert "first value" in expanded
    assert "second value" in expanded


def test_template_literal_variants_cap_preserves_later_alternative_branches() -> None:
    """Capped literal variants should keep action words from later branches."""
    many_tails = "|".join(f"tail{i}" for i in range(20))
    variants = gl._template_literal_token_variants(
        "(first|second) {item} [<tail>]",
        {"tail": f"({many_tails})"},
    )

    assert frozenset({"first"}) in variants
    assert frozenset({"second"}) in variants


def test_expand_sentence_template_deduplicates_cleaned_whitespace() -> None:
    """Deduplicate equivalent template expansions after whitespace normalization."""
    expanded = expand_sentence_template(
        "(turn  on|turn on) {name}",
        {"name": ("lamp",)},
        {},
        max_expansions=10,
    )

    assert expanded == ("turn on lamp",)


def test_expand_sentence_template_treats_stray_closers_as_literals() -> None:
    """Preserve unmatched optional and group closers as literal text."""
    assert expand_sentence_template("turn ] light", {}, {}, max_expansions=10) == ("turn ] light",)
    assert expand_sentence_template("turn ) light", {}, {}, max_expansions=10) == ("turn ) light",)
    assert expand_sentence_template("[turn )] light", {}, {}, max_expansions=10) == (
        "light",
        "turn ) light",
    )


def test_build_candidates_from_intent_sources_expands_template_lists() -> None:
    """Build candidates from sentence templates when list inputs are available."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn [the] {name} on"],
                                "lists": {
                                    "name": {
                                        "values": [
                                            {"in": "kitchen light", "out": "light.kitchen"},
                                            {"in": ["desk lamp", "office lamp"]},
                                        ]
                                    }
                                },
                            }
                        ]
                    }
                }
            }
        },
    )
    assert sorted(candidate.text for candidate in candidates) == sorted(
        [
            "turn kitchen light on",
            "turn desk lamp on",
            "turn office lamp on",
            "turn the kitchen light on",
            "turn the desk lamp on",
            "turn the office lamp on",
        ]
    )
    assert {candidate.source for candidate in candidates} == {CandidateSource.BUILT_IN}


def test_build_candidates_from_intent_sources_uses_registry_slot_values() -> None:
    """Expand entity slots from registry metadata fallback values."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn on {name}"],
                            }
                        ]
                    }
                }
            }
        },
        {
            "name": ("kitchen light",),
            "area": ("kitchen",),
        },
    )
    assert [candidate.text for candidate in candidates] == ["turn on kitchen light"]


def test_build_candidates_does_not_cross_product_registry_area_and_entity() -> None:
    """Avoid fake candidates that combine every area with every entity."""
    candidates = build_candidates_from_intent_sources(
        "vi",
        {
            "builtin": {
                "intents": {
                    "HassTurnOff": {
                        "data": [
                            {
                                "sentences": ["{area} {name} tắt", "tắt {area} {name}"],
                            }
                        ]
                    }
                }
            }
        },
        {
            "name": ("Đèn phòng ngủ to",),
            "area": ("Other", "Playroom"),
        },
    )
    assert candidates == ()


def test_build_candidates_keep_required_area_for_explicit_entity_slot() -> None:
    """Keep required location slots when entity text comes from template data."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn on {name} in {area}"],
                                "lists": {"name": {"values": ["light"]}},
                            }
                        ]
                    }
                }
            }
        },
        {
            "name": ("kitchen light",),
            "area": ("kitchen",),
        },
    )

    assert [candidate.text for candidate in candidates] == ["turn on light in kitchen"]
    slots = orjson.loads(candidates[0].metadata["slots"])
    assert slots["name"] == "light"
    assert slots["area"] == "kitchen"


def test_build_candidates_skip_mixed_registry_entity_location_static_path() -> None:
    """Avoid expanding registry entity/location combinations into the static index."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn on {name} in {area}"],
                            }
                        ]
                    }
                }
            }
        },
        {
            "name": ("kitchen light",),
            "area": ("kitchen", "office"),
        },
    )

    assert candidates == ()


def test_build_candidates_prioritizes_entity_slot_alias_templates() -> None:
    """Treat all entity slot aliases as name-priority templates."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {"sentences": ["turn on {area}"]},
                            {"sentences": ["turn on {entity}"]},
                        ]
                    }
                }
            }
        },
        {
            "area": ("kitchen",),
            "entity": ("kitchen light",),
        },
        max_candidates=1,
    )

    assert [candidate.text for candidate in candidates] == ["turn on kitchen light"]


def test_build_candidates_uses_domain_scoped_registry_names() -> None:
    """Use HassIL domain context to choose entity slot values."""
    candidates = build_candidates_from_intent_sources(
        "vi",
        {
            "builtin": {
                "intents": {
                    "HassMediaPause": {
                        "data": [
                            {
                                "sentences": ["dừng {name}"],
                                "requires_context": {"domain": "media_player"},
                            }
                        ]
                    }
                }
            }
        },
        {
            "name": ("Đèn phòng ngủ to", "Loa phòng khách"),
            "name:light": ("Đèn phòng ngủ to",),
            "name:media_player": ("Loa phòng khách",),
        },
    )

    assert [candidate.text for candidate in candidates] == ["dừng Loa phòng khách"]
    slots = orjson.loads(candidates[0].metadata["slots"])
    assert slots == {"name": "Loa phòng khách"}


def test_build_candidates_from_intent_sources_stops_at_total_cap() -> None:
    """Stop expanding later intents once the language candidate cap is reached."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {
                    "FirstIntent": {"data": [{"sentences": ["first {name}"]}]},
                    "SecondIntent": {"data": [{"sentences": ["second {name}"]}]},
                }
            }
        },
        {"name": ("one", "two", "three")},
        max_candidates=2,
    )

    assert [candidate.text for candidate in candidates] == ["first one", "first two"]


def test_build_candidates_reserves_global_cap_for_higher_priority_sources() -> None:
    """Build custom sentences before built-ins can consume the language cap."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "built_in": {"intents": {"BuiltInIntent": {"data": [{"sentences": ["built in"]}]}}},
            "custom_sentence": {
                "intents": {"CustomIntent": {"data": [{"sentences": ["custom phrase"]}]}}
            },
        },
        max_candidates=1,
    )

    assert [(candidate.text, candidate.source) for candidate in candidates] == [
        ("custom phrase", CandidateSource.CUSTOM_SENTENCE)
    ]


def test_build_candidates_uses_multi_domain_context_scoped_registry_names() -> None:
    """Expand entity slots from all domains listed in HassIL context."""
    candidates = build_candidates_from_intent_sources(
        "vi",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["bật {name}"],
                                "requires_context": {"domain": ["fan", "light"]},
                            }
                        ]
                    }
                }
            }
        },
        {
            "name": ("Loa phòng khách", "Quạt thông gió phòng tắm to"),
            "name:fan": ("Quạt thông gió phòng tắm to",),
            "name:light": ("Đèn phòng tắm to",),
            "name:media_player": ("Loa phòng khách",),
        },
    )

    assert sorted(candidate.text for candidate in candidates) == sorted(
        [
            "bật Quạt thông gió phòng tắm to",
            "bật Đèn phòng tắm to",
        ]
    )


def test_build_candidates_stores_localized_template_literals() -> None:
    """Store template literal words for language-agnostic action ranking."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn [the] {name} on"],
                                "lists": {"name": {"values": ["fan"]}},
                            }
                        ]
                    }
                }
            }
        },
    )

    assert candidates[0].metadata["sentence_template"] == "turn [the] {name} on"
    assert candidates[0].metadata["literal_text"] == "turn on|turn the on"


def test_build_candidates_from_intent_sources_prevents_intent_starvation(
    monkeypatch,
) -> None:
    """Ensure that intent limits prevent a single intent from starving others."""
    monkeypatch.setattr(gl, "DEFAULT_MAX_CANDIDATES_PER_INTENT", 1)

    candidates = gl.build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {
                    "FirstIntent": {"data": [{"sentences": ["first {name}"]}]},
                    "SecondIntent": {"data": [{"sentences": ["second {name}"]}]},
                }
            }
        },
        {"name": ("one", "two")},
        max_candidates=10,
    )
    assert [candidate.text for candidate in candidates] == ["first one", "second one"]


def test_expand_sentence_template_splits_pipes_in_expansion_rules() -> None:
    """Verify that rules with pipes and no parentheses are split."""
    candidates = build_candidates_from_intent_sources(
        "vi",
        {
            "builtin": {
                "intents": {
                    "HassTurnOff": {
                        "data": [
                            {
                                "sentences": ["<turn_off> {name}"],
                                "expansion_rules": {"turn_off": "tắt|ngắt|cúp"},
                            }
                        ]
                    }
                }
            }
        },
        {"name": ("đèn",)},
    )
    texts = [candidate.text for candidate in candidates]
    assert "tắt đèn" in texts
    assert "ngắt đèn" in texts
    assert "cúp đèn" in texts

    c_tat = next(c for c in candidates if c.text == "tắt đèn")
    assert "tắt" in c_tat.metadata["literal_text"]
    assert "ngắt" in c_tat.metadata["literal_text"]
    assert "cúp" in c_tat.metadata["literal_text"]


def test_template_literal_text_handles_missing_rules() -> None:
    """Verify that _template_literal_text replaces missing rules with empty string choices."""
    literal = gl._template_literal_text("[hãy] <missing_rule> {name}", {})
    assert "hãy" in literal


def test_template_literal_text_handles_recursive_rules() -> None:
    """Verify that _template_literal_text breaks recursion safely."""
    literal = gl._template_literal_text(
        "<recursive_rule> {name}", {"recursive_rule": "tắt <recursive_rule>"}
    )
    assert "tắt" in literal


def test_build_candidates_from_intent_sources_expands_range_and_wildcard_lists() -> None:
    """Verify that range and wildcard lists expand correctly and optimally."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassSetVolume": {
                        "data": [
                            {
                                "sentences": ["set volume to {volume}"],
                                "lists": {"volume": {"range": {"from": 0, "to": 100}}},
                            }
                        ]
                    },
                    "HassBroadcast": {
                        "data": [
                            {
                                "sentences": ["broadcast {message}"],
                                "lists": {"message": {"wildcard": True}},
                            }
                        ]
                    },
                }
            }
        },
    )
    texts = [c.text for c in candidates]
    # Check range list
    assert "set volume to 0" in texts
    assert "set volume to 100" in texts
    # Check wildcard list - yields list_name ("message")
    assert "broadcast message" in texts


@pytest.mark.parametrize(
    ("sentence", "expected_text", "expected_slots"),
    [
        ("remove {timer_minutes:minutes} from timer", "remove 1 from timer", {"minutes": "1"}),
        ("remove {timer_seconds:seconds} from timer", "remove 2 from timer", {"seconds": "2"}),
        ("fan {fan_speed:percentage}", "fan 42", {"percentage": "42"}),
        ("set volume to {volume:volume_level}", "set volume to 55", {"volume_level": "55"}),
        ("open {cover_classes:device_class}", "open door", {"device_class": "door"}),
    ],
)
def test_build_candidates_uses_slot_output_names_for_renamed_lists(
    sentence: str,
    expected_text: str,
    expected_slots: dict[str, str],
) -> None:
    """Use ``{list_name:entity_name}`` output names in candidate slot metadata."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {
                    "SyntheticIntent": {
                        "data": [
                            {
                                "sentences": [sentence],
                                "lists": {
                                    "timer_minutes": {"values": ["1"]},
                                    "timer_seconds": {"values": ["2"]},
                                    "fan_speed": {"values": ["42"]},
                                    "volume": {"values": ["55"]},
                                    "cover_classes": {"values": ["door"]},
                                    "unrelated_number": {"values": ["1", "2", "42", "55"]},
                                },
                            }
                        ]
                    }
                }
            }
        },
    )

    assert [candidate.text for candidate in candidates] == [expected_text]
    slots = orjson.loads(candidates[0].metadata["slots"])
    assert slots == expected_slots


def test_build_candidates_extracts_only_slots_referenced_by_template_rules() -> None:
    """Do not leak same-valued numeric lists that are unrelated to the template."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {
                    "SyntheticIntent": {
                        "data": [
                            {
                                "sentences": ["remove <duration> from timer"],
                                "expansion_rules": {"duration": "{timer_minutes:minutes} minutes"},
                                "lists": {
                                    "timer_minutes": {"values": ["1"]},
                                    "fan_speed": {"values": ["1"]},
                                    "volume": {"values": ["1"]},
                                },
                            }
                        ]
                    }
                }
            }
        },
    )

    assert [candidate.text for candidate in candidates] == ["remove 1 minutes from timer"]
    slots = orjson.loads(candidates[0].metadata["slots"])
    assert slots == {"minutes": "1"}


def test_build_candidates_preserve_distinct_equal_list_slot_bindings() -> None:
    """Bind each rendered slot occurrence even when two lists have equal values."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "custom_sentence": {
                "intents": {
                    "SyntheticIntent": {
                        "data": [
                            {
                                "sentences": ["{first:left} then {second:right}"],
                                "lists": {
                                    "first": {"values": ["alpha", "beta"]},
                                    "second": {"values": ["alpha", "beta"]},
                                },
                            }
                        ]
                    }
                }
            }
        },
    )

    candidate = next(item for item in candidates if item.text == "alpha then beta")
    assert orjson.loads(candidate.metadata["slots"]) == {
        "left": "alpha",
        "right": "beta",
    }


def test_build_candidates_discard_conflicting_repeated_slot_bindings() -> None:
    """Discard repeated output slots whose rendered values disagree."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "custom_sentence": {
                "intents": {
                    "SyntheticIntent": {
                        "data": [
                            {
                                "sentences": ["{choice} then {choice}"],
                                "lists": {"choice": {"values": ["alpha", "beta"]}},
                            }
                        ]
                    }
                }
            }
        },
    )

    candidates_by_text = {candidate.text: candidate for candidate in candidates}
    assert set(candidates_by_text) == {"alpha then alpha", "beta then beta"}
    for value in ("alpha", "beta"):
        candidate = candidates_by_text[f"{value} then {value}"]
        assert orjson.loads(candidate.metadata["slots"]) == {"choice": value}
        assert orjson.loads(candidate.metadata["slots_raw"]) == {"choice": value}


def test_repeated_slot_conflicts_do_not_consume_template_expansion_cap() -> None:
    """Collect capped valid repeats without counting conflicting combinations."""
    cap = gl.DEFAULT_MAX_CANDIDATES_PER_TEMPLATE
    values = [f"value_{index:03d}" for index in range(cap + 1)]
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "custom_sentence": {
                "intents": {
                    "SyntheticIntent": {
                        "data": [
                            {
                                "sentences": ["{choice} then {choice}"],
                                "lists": {"choice": {"values": values}},
                            }
                        ]
                    }
                }
            }
        },
    )

    candidates_by_text = {candidate.text: candidate for candidate in candidates}
    expected_values = values[:cap]
    assert set(candidates_by_text) == {f"{value} then {value}" for value in expected_values}
    for value in expected_values:
        candidate = candidates_by_text[f"{value} then {value}"]
        assert orjson.loads(candidate.metadata["slots"]) == {"choice": value}


@pytest.mark.parametrize(
    ("sentence", "expansion_rules", "expected"),
    [
        ("say {choice}", {}, frozenset()),
        ("{choice} then {choice}", {}, frozenset({"choice"})),
        ("({first:value}|{second:value})", {}, frozenset()),
        ("({first:value};{second:value})", {}, frozenset({"value"})),
        ("<pick> then <pick>", {"pick": "{choice}"}, frozenset({"choice"})),
        ("[{choice}] {choice}", {}, frozenset({"choice"})),
    ],
)
def test_template_repeated_output_names_follow_expansion_paths(
    sentence: str,
    expansion_rules: dict[str, str],
    expected: frozenset[str],
) -> None:
    """Detect only output slots that can coexist on one expansion path."""
    assert gl._template_repeated_output_names(sentence, expansion_rules) == expected


def test_candidate_expansion_skips_conflict_filter_for_unique_outputs() -> None:
    """Do not scan intermediate expansions when output slots are unique."""
    with patch.object(
        gl,
        "_binding_tags_are_consistent",
        side_effect=AssertionError("unique outputs must bypass the binding filter"),
    ):
        candidates = build_candidates_from_intent_sources(
            "en",
            {
                "custom_sentence": {
                    "intents": {
                        "SyntheticIntent": {
                            "data": [
                                {
                                    "sentences": ["{first:left} then {second:right}"],
                                    "lists": {
                                        "first": {"values": ["alpha"]},
                                        "second": {"values": ["beta"]},
                                    },
                                }
                            ]
                        }
                    }
                }
            },
        )

    assert [candidate.text for candidate in candidates] == ["alpha then beta"]


def test_candidate_expansion_skips_binding_tags_without_slots() -> None:
    """Expand literal template syntax without allocating binding tags."""
    with patch.object(
        gl,
        "_expansion_tag_context",
        side_effect=AssertionError("slot-free templates must bypass binding tags"),
    ):
        candidates = build_candidates_from_intent_sources(
            "en",
            {
                "custom_sentence": {
                    "intents": {"SyntheticIntent": {"data": [{"sentences": ["turn (on|off)"]}]}}
                }
            },
        )

    assert {candidate.text for candidate in candidates} == {"turn on", "turn off"}


def test_build_candidates_do_not_infer_slot_values_from_literal_text() -> None:
    """Record the emitted slot value when another value occurs in template literals."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "custom_sentence": {
                "intents": {
                    "SyntheticIntent": {
                        "data": [
                            {
                                "sentences": ["alpha {choice}"],
                                "lists": {"choice": {"values": ["alpha", "beta"]}},
                            }
                        ]
                    }
                }
            }
        },
    )

    candidate = next(item for item in candidates if item.text == "alpha beta")
    assert orjson.loads(candidate.metadata["slots"]) == {"choice": "beta"}


def test_slot_binding_tags_do_not_let_duplicate_values_consume_expansion_cap() -> None:
    """Deduplicate equal slot inputs before adding internal binding tags."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "custom_sentence": {
                "intents": {
                    "SyntheticIntent": {
                        "data": [
                            {
                                "sentences": ["choose {choice}"],
                                "lists": {
                                    "choice": {
                                        "values": [
                                            *(["alpha"] * gl.DEFAULT_MAX_CANDIDATES_PER_TEMPLATE),
                                            "beta",
                                        ]
                                    }
                                },
                            }
                        ]
                    }
                }
            }
        },
    )

    assert len(candidates) == 2
    assert {candidate.text for candidate in candidates} == {"choose alpha", "choose beta"}


def test_decode_candidate_expansion_strips_owned_tags_in_reference_order() -> None:
    """Decode collision-free tags once and retain reference order."""
    tag_context = gl._expansion_tag_context("alpha then beta __sb_9_9__", {}, {})
    marker = tag_context.marker
    first = gl._SlotBinding("first", "left", "alpha")
    second = gl._SlotBinding("second", "right", "beta")
    first_tag = f"{marker}b:0:0{marker}"
    second_tag = f"{marker}b:1:0{marker}"
    bindings = {
        first_tag: first,
        second_tag: second,
    }

    expansion = gl._decode_candidate_expansion(
        f"{second_tag}beta then {first_tag}alpha and {second_tag}beta __sb_9_9__",
        bindings,
        tag_context,
    )

    assert expansion.text == "beta then alpha and beta __sb_9_9__"
    assert expansion.slot_bindings == (first, second)


def test_expansion_marker_avoids_all_rendered_source_inputs() -> None:
    """Choose a marker absent from sentence, rule, and slot-value text."""
    tag_context = gl._expansion_tag_context(
        "say \ue000 <rule> {choice}",
        {"choice": ("alpha \ue001",)},
        {"rule": "value \ue002"},
    )

    assert tag_context.marker == "\ue003"


def test_expansion_tag_context_reuses_cached_marker_patterns() -> None:
    """Reuse compiled tag patterns when expansions select the same marker."""
    gl._cached_expansion_tag_context.cache_clear()

    first = gl._expansion_tag_context("say alpha", {}, {})
    second = gl._expansion_tag_context("say beta", {}, {})

    assert second is first
    assert gl._cached_expansion_tag_context.cache_info().hits == 1


def test_decode_candidate_expansion_rejects_missing_owned_binding() -> None:
    """Fail explicitly when an injected marker has no binding-map entry."""
    tag_context = gl._expansion_tag_context("say alpha", {}, {})
    unknown_tag = f"{tag_context.marker}b:0:0{tag_context.marker}"

    with pytest.raises(RuntimeError, match="Missing binding for injected slot tag"):
        gl._decode_candidate_expansion(
            f"say {unknown_tag}alpha",
            {},
            tag_context,
        )


def test_build_candidates_preserve_tag_shaped_sentence_literals() -> None:
    """Preserve a literal that exactly matches the former first binding tag."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "custom_sentence": {
                "intents": {
                    "SyntheticIntent": {
                        "data": [
                            {
                                "sentences": ["say __sb_0_0__ {choice}"],
                                "lists": {"choice": {"values": ["alpha"]}},
                            }
                        ]
                    }
                }
            }
        },
    )

    assert [candidate.text for candidate in candidates] == ["say __sb_0_0__ alpha"]
    assert orjson.loads(candidates[0].metadata["slots"]) == {"choice": "alpha"}


def test_build_candidates_preserve_wildcard_tag_shaped_sentence_literals() -> None:
    """Decode only owned wildcard tags and preserve equivalent literal text."""
    sources = {
        "custom_sentence": {
            "intents": {
                "SyntheticIntent": {
                    "data": [
                        {
                            "sentences": ["say __wc_choice__ {choice}"],
                            "lists": {"choice": {"wildcard": True}},
                        }
                    ]
                }
            }
        }
    }
    candidates = build_candidates_from_intent_sources("en", sources)

    assert [candidate.text for candidate in candidates] == ["say __wc_choice__ choice"]
    candidate = candidates[0]
    assert len(candidate.wildcard_infos) == 1
    wildcard_index, wildcard_name = candidate.wildcard_infos[0]
    assert wildcard_name == "choice"
    assert candidate.normalized_tokens[wildcard_index] == "choice"
    assert orjson.loads(candidate.metadata["slots_raw"]) == {"choice": "choice"}

    query_candidates = build_query_registry_candidates(
        "en",
        sources,
        {},
        "say __wc_choice__ choice",
    )
    assert [item.text for item in query_candidates] == ["say __wc_choice__ choice"]
    assert len(query_candidates[0].wildcard_infos) == 1


class TestRehydrateWildcardText:
    """Tests for rehydrate_wildcard_text wildcard placeholder rehydration."""

    @pytest.fixture(autouse=True)
    def registered_wildcards(self) -> Iterator[None]:
        """Register the wildcard corpus used by raw-text compatibility helpers."""
        sources = {
            "built_in": {
                "lists": {
                    "shopping_list_item": {"wildcard": True},
                    "todo_list_item": {"wildcard": True},
                    "timer_name": {"wildcard": True},
                    "message": {"wildcard": True},
                    "search_query": {"wildcard": True},
                }
            }
        }
        wildcard_slot_names.cache_clear()
        for language in ("de", "en", "it", "nl", "vi"):
            register_custom_wildcards_from_sources(language, sources)
        try:
            yield
        finally:
            wildcard_slot_names.cache_clear()

    def test_italian_shopping_list(self) -> None:
        """Rehydrate shopping_list_item from Italian query."""
        result = gl.rehydrate_wildcard_text(
            "aggiungi shopping_list_item alla Lista della Spesa",
            "aggiunge tovaglioli alla lista della spesa",
        )
        assert result == "aggiungi tovaglioli alla Lista della Spesa"

    def test_english_shopping_list(self) -> None:
        """Rehydrate shopping_list_item from English query."""
        result = gl.rehydrate_wildcard_text(
            "add shopping_list_item to shopping list",
            "add milk to shopping list",
        )
        assert result == "add milk to shopping list"

    def test_vietnamese_shopping_list(self) -> None:
        """Rehydrate shopping_list_item from Vietnamese query."""
        result = gl.rehydrate_wildcard_text(
            "đặt shopping_list_item vào danh sách mua sắm",
            "đặt sữa vào danh sách mua sắm",
        )
        assert result == "đặt sữa vào danh sách mua sắm"

    def test_no_wildcard_passthrough(self) -> None:
        """Return candidate unchanged when no wildcard placeholder exists."""
        result = gl.rehydrate_wildcard_text(
            "turn on kitchen light",
            "turn on kitchen light",
        )
        assert result == "turn on kitchen light"

    def test_query_shorter_than_prefix_falls_back(self) -> None:
        """Return candidate unchanged when query is too short for alignment."""
        result = gl.rehydrate_wildcard_text(
            "add shopping_list_item to shopping list",
            "add milk",
        )
        # Query too short to align suffix falls back to original
        assert result == "add shopping_list_item to shopping list"

    def test_multi_word_wildcard_value(self) -> None:
        """Rehydrate a multi-word shopping item."""
        result = gl.rehydrate_wildcard_text(
            "add shopping_list_item to shopping list",
            "add bread and butter to shopping list",
        )
        assert result == "add bread and butter to shopping list"

    def test_no_prefix_wildcard_at_start(self) -> None:
        """Rehydrate when wildcard is the first token (no prefix)."""
        result = gl.rehydrate_wildcard_text(
            "shopping_list_item on my list",
            "milk on my list",
        )
        assert result == "milk on my list"

    @pytest.mark.current_intents
    def test_todo_list_item_rehydration(self) -> None:
        """Rehydrate todo_list_item wildcard."""
        result = gl.rehydrate_wildcard_text(
            "add todo_list_item to my todo list",
            "add buy groceries to my todo list",
        )
        assert result == "add buy groceries to my todo list"

    def test_stt_misspelling_in_query_preserved(self) -> None:
        """Preserve the STT error (aggiunge→aggiungi) from the candidate, not query."""
        result = gl.rehydrate_wildcard_text(
            "aggiungi shopping_list_item alla Lista della Spesa",
            "aggiunge tovaglioli alla lista della spesa",
        )
        # Candidate stem ("aggiungi") is preserved; only the wildcard token
        # is replaced with the query's free-text value ("tovaglioli").
        assert result == "aggiungi tovaglioli alla Lista della Spesa"

    def test_wildcard_slot_names_populated(self) -> None:
        """Verify wildcard_slot_names() returns registered wildcard slots."""
        mock_data = {
            "lists": {
                "shopping_list_item": {"wildcard": True},
                "todo_list_item": {"wildcard": True},
                "timer_name": {"wildcard": True},
                "message": {"wildcard": True},
                "color": {"values": ["red", "blue"]},
            }
        }
        wildcard_slot_names.cache_clear()
        register_custom_wildcards_from_sources("en", {"built_in": mock_data})
        self._test_wildcard_slot_names_populated()

    def _test_wildcard_slot_names_populated(self) -> None:
        """Test that wildcard_slot_names() returns only wildcards, not entities."""
        names = wildcard_slot_names()
        assert "shopping_list_item" in names
        assert "todo_list_item" in names
        assert "timer_name" in names
        assert "message" in names
        # Entity slots that have registry values should NOT be wildcards
        assert "color" not in names

    def test_multiple_wildcards_falls_back_safely(self) -> None:
        """Fall back to original when suffix contains an unresolvable word."""
        # The suffix tokens include "invalid_suffix_word" which won't align against
        # "groceries" safe fallback returns original unchanged.
        result = gl.rehydrate_wildcard_text(
            "add shopping_list_item to my invalid_suffix_word list",
            "add milk to my groceries list",
        )
        # Safe fallback suffix contains invalid_suffix_word that can't align
        assert result == "add shopping_list_item to my invalid_suffix_word list"

    @pytest.mark.current_intents
    def test_original_case_preservation(self) -> None:
        """Verify that capitalization and punctuation in wildcard values are preserved."""
        result = gl.rehydrate_wildcard_text(
            "füge shopping_list_item zur EinkaufsListe hinzu",
            "füge Milch zur einkaufsliste hinzu",
        )
        assert result == "füge Milch zur EinkaufsListe hinzu"

        result = gl.rehydrate_wildcard_text(
            "broadcast message",
            "broadcast dinner's ready",
        )
        assert result == "broadcast dinner's ready"

    @pytest.mark.current_intents
    def test_compound_wildcard_token(self) -> None:
        """Verify that wildcard names inside compound tokens match and rehydrate."""
        result = gl.rehydrate_wildcard_text(
            "spiel den search_querypodcast",
            "spiel den Jazzpodcast",
        )
        assert result == "spiel den Jazzpodcast"

    def test_rehydrate_wildcard_slots(self) -> None:
        """Verify that wildcard placeholders inside slot dictionaries are correctly rehydrated."""
        slots = {"shopping_list_item": "shopping_list_item", "name": "grocery list"}
        result = gl.rehydrate_wildcard_slots(
            slots,
            "add shopping_list_item to shopping list",
            "add milk to shopping list",
            "en",
        )
        assert result == {"shopping_list_item": "milk", "name": "grocery list"}

    def test_rehydrate_wildcard_prefix_boundary_fallback(self) -> None:
        """Verify language-independent boundary when prefix stem alignment fails."""
        # Query "cho bánh chuối vào danh sách mua sắm cho anh nhé"
        # Candidate "đặt shopping_list_item vào danh sách mua sắm"
        # prefix stem "đặt" doesn't match "cho", but we fallback to len(c_prefix).
        result = gl.rehydrate_wildcard_text(
            "đặt shopping_list_item vào danh sách mua sắm",
            "cho bánh chuối vào danh sách mua sắm cho anh nhé",
            "vi",
        )
        assert result == "đặt bánh chuối vào danh sách mua sắm"

        # English synonym test:
        # Query "put milk to shopping list"
        # Candidate "add shopping_list_item to shopping list"
        result = gl.rehydrate_wildcard_text(
            "add shopping_list_item to shopping list",
            "put milk to shopping list",
            "en",
        )
        assert result == "add milk to shopping list"

        # English omitted verb test:
        # Query "milk to shopping list"
        # Candidate "add shopping_list_item to shopping list"
        result = gl.rehydrate_wildcard_text(
            "add shopping_list_item to shopping list",
            "milk to shopping list",
            "en",
        )
        assert result == "add milk to shopping list"

        # Multi-word prefix alignment failure should not perform fallback rehydration
        result = gl.rehydrate_wildcard_text(
            "wie is in de shopping_list_item",
            "welke dag in de week is het",
            "nl",
        )
        assert result == "wie is in de shopping_list_item"

    def test_rehydrate_wildcard_literal_name_conflict(self) -> None:
        """Verify that a literal matching a wildcard name is not treated as a wildcard."""
        intent_sources = {
            "custom": {
                "lists": {"song": {"wildcard": True}},
                "intents": {
                    "HassMediaSearchAndPlay": {"data": [{"sentences": ["play the song {song}"]}]}
                },
            }
        }
        wildcard_slot_names.cache_clear()
        register_custom_wildcards_from_sources("en", intent_sources)

        candidates = list(build_candidates_from_intent_sources("en", intent_sources))
        assert candidates
        candidate = candidates[0]

        rehydrated_text1, replacements1 = get_wildcard_rehydration(candidate, "play the song jazzy")
        assert rehydrated_text1 == "play the song jazzy"
        assert replacements1 == {"song": "jazzy"}

        rehydrated_text2, replacements2 = get_wildcard_rehydration(candidate, "play the tune jazzy")
        assert rehydrated_text2 == "play the song song"
        assert replacements2 == {}


def test_build_candidates_preserves_static_slots_and_does_round_robin() -> None:
    """Verify that static slots are preserved and candidate generation is round-robin."""
    candidates = build_candidates_from_intent_sources(
        "vi",
        {
            "builtin": {
                "intents": {
                    "HassTurnOff": {
                        "data": [
                            {
                                "sentences": ["tắt {name}"],
                                "lists": {
                                    "name": {
                                        "values": [
                                            "đèn phòng khách 1",
                                            "đèn phòng khách 2",
                                            "đèn phòng khách 3",
                                        ]
                                    }
                                },
                            },
                            {
                                "sentences": ["tắt quạt ở {area}"],
                                "slots": {"domain": "fan", "name": "all"},
                            },
                        ]
                    }
                }
            }
        },
        {"area": ("phòng khách",)},
        max_candidates=3,
    )
    texts = [c.text for c in candidates]
    assert "tắt đèn phòng khách 1" in texts
    assert "tắt quạt ở phòng khách" in texts

    fan_candidate = next(c for c in candidates if "tắt quạt" in c.text)
    slots = orjson.loads(fan_candidate.metadata["slots"])
    assert slots == {"area": "phòng khách", "domain": "fan", "name": "all"}
    assert fan_candidate.metadata["static_slots"] == "domain,name"


def test_build_candidates_domain_area_template_ignores_unreferenced_entity_aliases() -> None:
    """Domain-scoped location templates should not leak unrelated entity slots."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn on {area} fan"],
                                "slots": {"domain": "fan", "name": "all"},
                            },
                        ]
                    }
                }
            }
        },
        {
            "area": ("bathroom",),
            "name": ("bathroom fan",),
            "name:fan": ("bathroom fan",),
            "entity": ("bathroom fan",),
            "entity:fan": ("bathroom fan",),
        },
    )

    assert [candidate.text for candidate in candidates] == ["turn on bathroom fan"]
    slots = orjson.loads(candidates[0].metadata["slots"])
    assert slots == {"area": "bathroom", "domain": "fan", "name": "all"}


def test_compile_dynamic_registry_intents_tracks_only_unresolved_query_slots() -> None:
    """Template-provided entity/location slots should not require query registry matches."""
    compiled = compile_dynamic_registry_intents(
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn on {name} in {area} on {floor}"],
                                "lists": {"area": {"values": ["kitchen"]}},
                                "slots": {"domain": "light", "name": "all"},
                            },
                        ]
                    }
                }
            }
        },
        "en",
    )

    template = compiled[0].templates[0]
    assert template.entity_slots == ()
    assert template.query_slots == ("floor",)


def test_query_registry_candidates_allow_static_slot_without_registry_match() -> None:
    """Static template slots can expand even when registry values do not match the query."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn on {name}"],
                                "slots": {"name": "all"},
                            },
                        ]
                    }
                }
            }
        },
        {"name": ("kitchen light",)},
        "turn on all",
    )

    assert [candidate.text for candidate in candidates] == ["turn on all"]


def test_query_registry_candidates_match_compound_slot_spacing() -> None:
    """Match query words to compact registry slot values without language-specific rules."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [{"sentences": ["turn on {name}"]}],
                    }
                }
            }
        },
        {"name": ("kitchenlight",)},
        "turn on kitchen light",
    )

    assert [candidate.text for candidate in candidates] == ["turn on kitchenlight"]


def test_query_registry_candidates_collapse_domain_scoped_entity_slots() -> None:
    """Keep scoped registry values under the template slot name in candidate metadata."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn on {name}"],
                                "requires_context": {"domain": "fan"},
                            },
                        ]
                    }
                }
            }
        },
        {
            "name": ("speaker", "bathroom fan"),
            "name:fan": ("bathroom fan",),
            "name:media_player": ("speaker",),
        },
        "turn on bathroom fan",
    )

    assert [candidate.text for candidate in candidates] == ["turn on bathroom fan"]
    slots = orjson.loads(candidates[0].metadata["slots"])
    assert slots == {"name": "bathroom fan"}


def test_query_registry_candidates_build_mixed_entity_area_dynamically() -> None:
    """Build mixed registry entity/location candidates only for matching queries."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [{"sentences": ["turn on {name} in {area}"]}],
                    }
                }
            }
        },
        {
            "name": ("kitchen light",),
            "area": ("kitchen", "office"),
        },
        "turn on kitchen light in kitchen",
    )

    assert candidates
    assert candidates[0].text == "turn on kitchen light in kitchen"
    slots = orjson.loads(candidates[0].metadata["slots"])
    assert slots["name"] == "kitchen light"
    assert slots["area"] == "kitchen"


def test_query_registry_candidates_do_not_fuzzy_nominate_short_aliases() -> None:
    """Avoid a one-character registry rewrite for four-character aliases."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn on {name}"],
                                "requires_context": {"domain": "light"},
                            }
                        ]
                    }
                }
            }
        },
        {"name": ("Hall",), "name:light": ("Hall",)},
        "turn on hell",
    )

    assert candidates == ()


def test_query_registry_candidates_global_cap_keeps_later_exact_match() -> None:
    """Keep the best dynamic candidate even when earlier templates fill the cap."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurn": {
                        "data": [
                            {"sentences": ["turn {name} off"]},
                            {"sentences": ["turn {name} on"]},
                        ]
                    }
                }
            }
        },
        {"name": ("tail light",)},
        "turn tail light on",
        max_candidates=1,
    )

    assert len(candidates) == 1
    assert candidates[0].normalized_text == "turn tail light on"


def test_query_registry_candidates_prioritize_exact_base_list_values() -> None:
    """Prefer base-list numeric values that occur exactly in the query."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "lists": {
                    "brightness": {"range": {"from": 0, "to": 100}},
                    "color_temperature": {"range": {"from": 1000, "to": 10000}},
                },
                "intents": {
                    "HassLightSet": {
                        "data": [
                            {"sentences": ["{name} {color_temperature:temperature}"]},
                            {"sentences": ["{name} {brightness}"]},
                        ]
                    }
                },
            }
        },
        {"name": ("bedroom light",)},
        "bedroom light 100",
        max_candidates=1,
    )

    assert [candidate.text for candidate in candidates] == ["bedroom light 100"]
    assert orjson.loads(candidates[0].metadata["slots"]) == {
        "name": "bedroom light",
        "brightness": 100,
    }


def _build_range_intent_source(
    list_name: str,
    range_config: dict[str, Any],
    intent_name: str,
    sentences: list[str],
) -> dict[str, Any]:
    """Helper to construct common range-based intent source configurations."""
    return {
        "builtin": {
            "lists": {
                list_name: {"range": range_config},
            },
            "intents": {
                intent_name: {
                    "data": [
                        {"sentences": sentences},
                    ]
                }
            },
        }
    }


def test_query_registry_candidates_match_numeric_range_values() -> None:
    """Test that numbers in range from user query are matched dynamically."""
    candidates = build_query_registry_candidates(
        "en",
        _build_range_intent_source(
            "temperature",
            {"from": 0, "to": 50},
            "HassClimateSetTemperature",
            ["set [the] {name} temperature to {temperature}"],
        ),
        {"name": ("living room",)},
        "set living room temperature to 27",
        max_candidates=1,
    )

    assert [candidate.text for candidate in candidates] == ["set living room temperature to 27"]
    assert orjson.loads(candidates[0].metadata["slots"]) == {
        "name": "living room",
        "temperature": 27,
    }


@pytest.mark.parametrize(("query_value", "slot_value"), [("20.5", 20.5), ("20,5", 20.5)])
def test_query_registry_candidates_preserve_fractional_range_values(
    query_value: str,
    slot_value: float,
) -> None:
    """Preserve decimal range values from the original query text."""
    query = f"set living room temperature to {query_value}"
    candidates = build_query_registry_candidates(
        "en",
        _build_range_intent_source(
            "temperature",
            {
                "type": "temperature",
                "from": 0,
                "to": 100,
                "fractions": "halves",
            },
            "HassClimateSetTemperature",
            ["set [the] {name} temperature to {temperature}"],
        ),
        {"name": ("living room",)},
        query,
        max_candidates=1,
    )

    assert [candidate.text for candidate in candidates] == [
        f"set living room temperature to {query_value}"
    ]
    assert orjson.loads(candidates[0].metadata["slots"]) == {
        "name": "living room",
        "temperature": slot_value,
    }


@pytest.mark.parametrize(
    ("query_value", "canonical_value", "slot_value"),
    [
        ("-.5", "-.5", -0.5),
        ("-,5", "-,5", -0.5),
        ("\N{MINUS SIGN}.5", "-.5", -0.5),
        ("\N{MINUS SIGN}5", "-5", -5),
    ],
)
def test_query_registry_candidates_preserve_signed_range_values(
    query_value: str,
    canonical_value: str,
    slot_value: float,
) -> None:
    """Keep leading-decimal and Unicode-minus values negative through expansion."""
    candidates = build_query_registry_candidates(
        "en",
        _build_range_intent_source(
            "temperature",
            {
                "type": "temperature",
                "from": -10,
                "to": 10,
                "fractions": "halves",
            },
            "HassClimateSetTemperature",
            ["set temperature to {temperature}"],
        ),
        {},
        f"set temperature to {query_value}",
        max_candidates=1,
    )

    assert [candidate.text for candidate in candidates] == [f"set temperature to {canonical_value}"]
    assert orjson.loads(candidates[0].metadata["slots"]) == {
        "temperature": slot_value,
    }


def test_query_registry_candidates_respect_numeric_range_step() -> None:
    """Do not inject query numbers that HassIL range step validation rejects."""
    candidates = build_query_registry_candidates(
        "en",
        _build_range_intent_source(
            "color_temperature",
            {"from": 1000, "to": 10000, "step": 100},
            "HassLightSet",
            ["{name} {color_temperature:temperature}"],
        ),
        {"name": ("bedroom light",)},
        "bedroom light 1050",
        max_candidates=5,
    )

    assert candidates
    assert all(candidate.text != "bedroom light 1050" for candidate in candidates)
    assert all(
        orjson.loads(candidate.metadata["slots"])["temperature"] != 1050 for candidate in candidates
    )


def test_query_registry_candidates_descending_range_step() -> None:
    """Validate that step validation works on descending ranges."""
    intent_sources = _build_range_intent_source(
        "test_range",
        {"from": 10, "to": 3, "step": 2},
        "TestIntent",
        ["set val to {test_range:val}"],
    )
    candidates = build_query_registry_candidates(
        "en",
        intent_sources,
        {},
        "set val to 10",
        max_candidates=1,
    )
    assert len(candidates) == 1
    assert orjson.loads(candidates[0].metadata["slots"]) == {"val": 10}

    candidates_rejected = build_query_registry_candidates(
        "en",
        intent_sources,
        {},
        "set val to 9",
        max_candidates=5,
    )
    assert candidates_rejected
    assert all(cand.text != "set val to 9" for cand in candidates_rejected)
    assert all(orjson.loads(cand.metadata["slots"]).get("val") != 9 for cand in candidates_rejected)


def test_query_registry_candidates_apply_numeric_range_multiplier() -> None:
    """Apply HassIL range multipliers to output slots, not spoken candidate text."""
    candidates = build_query_registry_candidates(
        "en",
        _build_range_intent_source(
            "volume_step_down",
            {"type": "percentage", "from": 0, "to": 100, "multiplier": -1},
            "HassSetVolumeRelative",
            ["volume down by {volume_step_down:volume_step}"],
        ),
        {},
        "volume down by 20",
        max_candidates=1,
    )

    assert [candidate.text for candidate in candidates] == ["volume down by 20"]
    assert orjson.loads(candidates[0].metadata["slots"]) == {"volume_step": -20}


def test_query_registry_candidates_respect_numeric_range_fractions_halves() -> None:
    """Validate that halves fractions are matched and non-halves are rejected."""
    intent_sources = _build_range_intent_source(
        "temperature",
        {
            "type": "temperature",
            "from": 0,
            "to": 100,
            "fractions": "halves",
        },
        "HassClimateSetTemperature",
        ["set temperature to {temperature}"],
    )
    # Matched candidate
    candidates = build_query_registry_candidates(
        "en",
        intent_sources,
        {},
        "set temperature to 27.5",
        max_candidates=1,
    )
    assert [candidate.text for candidate in candidates] == ["set temperature to 27.5"]
    assert orjson.loads(candidates[0].metadata["slots"]) == {"temperature": 27.5}

    # Rejected candidate (not a half)
    candidates_rejected = build_query_registry_candidates(
        "en",
        intent_sources,
        {},
        "set temperature to 27.3",
        max_candidates=5,
    )
    assert candidates_rejected
    assert all(
        orjson.loads(candidate.metadata["slots"]).get("temperature") != 27.3
        for candidate in candidates_rejected
    )
    assert all(candidate.text != "set temperature to 27.3" for candidate in candidates_rejected)
    assert all("27.3" not in candidate.text for candidate in candidates_rejected)


def test_query_registry_candidates_respect_numeric_range_fractions_tenths() -> None:
    """Validate that tenths fractions are matched and non-tenths are rejected."""
    intent_sources = _build_range_intent_source(
        "temperature",
        {
            "type": "temperature",
            "from": 0,
            "to": 100,
            "fractions": "tenths",
        },
        "HassClimateSetTemperature",
        ["set temperature to {temperature}"],
    )
    # Matched candidate
    candidates = build_query_registry_candidates(
        "en",
        intent_sources,
        {},
        "set temperature to 27.3",
        max_candidates=1,
    )
    assert [candidate.text for candidate in candidates] == ["set temperature to 27.3"]
    assert orjson.loads(candidates[0].metadata["slots"]) == {"temperature": 27.3}

    # Rejected candidate (not a tenth, e.g. hundredth)
    candidates_rejected = build_query_registry_candidates(
        "en",
        intent_sources,
        {},
        "set temperature to 27.35",
        max_candidates=5,
    )
    assert candidates_rejected
    assert all(
        orjson.loads(candidate.metadata["slots"]).get("temperature") != 27.35
        for candidate in candidates_rejected
    )
    assert all(candidate.text != "set temperature to 27.35" for candidate in candidates_rejected)
    assert all("27.35" not in candidate.text for candidate in candidates_rejected)


def test_query_registry_candidates_keep_exact_entity_when_partial_slots_hit_cap() -> None:
    """Keep exact text when an earlier entity/location branch fills the dynamic cap."""
    query = "tắt điều hòa phòng ngủ to"
    candidates = build_query_registry_candidates(
        "vi",
        {
            "builtin": {
                "intents": {
                    "HassTurnOff": {
                        "data": [{"sentences": ["tắt [cái] ({area} {name}|{name} [{area}])[đi]"]}]
                    }
                }
            }
        },
        {
            "name": (
                "Điều hòa phòng ngủ to",
                *(f"Điều hòa phòng ngủ {index}" for index in range(1, 25)),
            ),
            "area": (
                "Phòng ngủ to",
                *(f"Phòng ngủ {index}" for index in range(1, 25)),
            ),
        },
        query,
        max_candidates=1,
    )

    assert len(candidates) == 1
    assert candidates[0].normalized_text == query
    assert orjson.loads(candidates[0].metadata["slots"]) == {"name": "Điều hòa phòng ngủ to"}


def test_query_registry_candidates_keep_exact_entity_or_area_domain_match() -> None:
    """Prefer exact entity or exact area/domain candidates over mixed slot noise."""
    query = "turn on living room fan"
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {"sentences": ["turn on ({area} {name}|{name} [in {area}])"]},
                            {
                                "sentences": ["turn on {area} fan"],
                                "slots": {"domain": "fan", "name": "all"},
                            },
                        ]
                    }
                }
            }
        },
        {
            "name": (
                "living room fan",
                *(f"living room device {index}" for index in range(1, 25)),
                "fan",
            ),
            "name:fan": ("living room fan", "fan"),
            "area": (
                "living room",
                *(f"living room {index}" for index in range(1, 25)),
            ),
        },
        query,
        max_candidates=1,
    )

    assert len(candidates) == 1
    assert candidates[0].normalized_text == query
    slots = orjson.loads(candidates[0].metadata["slots"])
    assert slots in [
        {"name": "living room fan"},
        {
            "area": "living room",
            "domain": "fan",
            "name": "all",
        },
    ]


def test_exact_slot_priority_requires_full_literal_match() -> None:
    """Do not promote opposite-action candidates just because an entity is exact."""
    query = "bring up the living room light"
    query_normalized = normalize_text(query)
    query_no_diac = gl.normalize_text_no_diacritics(query, "en")
    sources = {
        "builtin": {
            "intents": {
                "HassTurnOff": {
                    "data": [
                        {
                            "sentences": ["<turn> off ({area} {name}|{name} [in {area}])"],
                            "expansion_rules": {"turn": "turn|bring"},
                        }
                    ]
                }
            }
        }
    }
    registry_slots = {
        "name": ("living room light",),
        "area": ("living room",),
    }
    template = compile_dynamic_registry_intents(sources, "en")[0].templates[0]
    slot_values = gl._resolve_template_slot_values(
        template,
        registry_slots,
        build_registry_slot_index(registry_slots, "en"),
        query_normalized,
        frozenset(query_normalized.split()),
        {},
        {},
        query_no_diac=query_no_diac,
        query_tokens_no_diac=frozenset(query_no_diac.split()),
    )

    assert slot_values is not None
    assert (
        gl._exact_slot_preferred_value_maps(
            template,
            slot_values,
            query_normalized,
            frozenset(query_normalized.split()),
            query_no_diac,
            "en",
        )
        == ()
    )


def test_query_registry_candidates_keep_compact_script_exact_entity() -> None:
    """Keep exact entity candidates for languages that commonly omit spaces."""
    query = "打开一楼主卧室床头阅读灯"
    candidates = build_query_registry_candidates(
        "zh",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [{"sentences": ["打开{name}"]}],
                    }
                }
            }
        },
        {
            "name": (
                "一楼主卧室床头阅读灯",
                "客厅灯",
            )
        },
        query,
        max_candidates=1,
    )

    assert len(candidates) == 1
    assert candidates[0].normalized_text == query
    assert orjson.loads(candidates[0].metadata["slots"]) == {"name": "一楼主卧室床头阅读灯"}


@pytest.mark.parametrize(
    ("sentence", "expected_prefix", "expected_suffix"),
    [
        ("打开{name}", "打开", ""),
        ("включи {name}", "включи ", ""),
        ("{name}をつけて", "", "をつけて"),
    ],
)
def test_compile_slot_anchor_patterns_preserves_template_boundaries(
    sentence: str,
    expected_prefix: str,
    expected_suffix: str,
) -> None:
    """Compile exact literal boundaries without discarding meaningful spaces."""
    compiled = compile_dynamic_registry_intents(
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {"data": [{"sentences": [sentence]}]},
                }
            }
        },
        "ru",
    )

    patterns = compiled[0].templates[0].slot_anchor_patterns["name"]

    assert [(pattern.prefix, pattern.suffix) for pattern in patterns] == [
        (expected_prefix, expected_suffix)
    ]


@pytest.mark.parametrize(
    ("language", "sentence", "query", "entity"),
    [
        ("ru", "включи{name}", "включисвет", "свет"),
        ("ar", "شغل{name}", "شغلالضوء", "الضوء"),
        ("ja", "{name}をつけて", "ライトをつけて", "ライト"),
        ("xx", "打开{name}", "打开café灯", "café灯"),
    ],
)
def test_query_registry_candidates_use_language_agnostic_slot_anchors(
    language: str,
    sentence: str,
    query: str,
    entity: str,
) -> None:
    """Use the grammar's literal boundaries instead of a script allowlist."""
    stats = gl.RegistryRetrievalStats()
    candidates = build_query_registry_candidates(
        language,
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {"data": [{"sentences": [sentence]}]},
                }
            }
        },
        {"name": (entity,)},
        query,
        retrieval_stats=stats,
    )

    assert [candidate.normalized_text for candidate in candidates] == [normalize_text(query)]
    assert orjson.loads(candidates[0].metadata["slots"]) == {"name": entity}
    assert candidates[0].metadata["registry_retrieval"] == "anchored"
    assert stats.anchored_dynamic_candidates == 1


def test_query_registry_candidates_keep_required_anchor_whitespace() -> None:
    """Do not reinterpret an explicitly spaced grammar as a joined command."""
    candidates = build_query_registry_candidates(
        "ru",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {"data": [{"sentences": ["включи {name}"]}]},
                }
            }
        },
        {"name": ("свет",)},
        "включисвет",
    )

    assert candidates == ()


def test_query_registry_candidates_require_whole_anchored_slot_window() -> None:
    """Reject an entity that occupies only a substring of the isolated slot span."""
    candidates = build_query_registry_candidates(
        "zh",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {"data": [{"sentences": ["打开{name}"]}]},
                }
            }
        },
        {"name": ("卧室灯",)},
        "打开卧室灯状态",
    )

    assert candidates == ()


def test_query_registry_candidates_bound_fuzzy_matching_to_anchored_window() -> None:
    """Allow one bounded entity typo after the grammar isolates the complete span."""
    stats = gl.RegistryRetrievalStats()
    candidates = build_query_registry_candidates(
        "ru",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {"data": [{"sentences": ["включи{name}"]}]},
                }
            }
        },
        {"name": ("лампа",)},
        "включиламба",
        retrieval_stats=stats,
    )

    assert [candidate.text for candidate in candidates] == ["включилампа"]
    assert candidates[0].metadata["registry_retrieval"] == "fuzzy"
    assert stats.anchored_dynamic_candidates == 1
    assert stats.fuzzy_dynamic_candidates == 1


def test_anchored_registry_retrieval_reaches_tail_with_bounded_scoring() -> None:
    """Find a tail entity through its whole slot window without scanning the registry."""
    values = (
        *(f"устройство {position}" for position in range(gl.DEFAULT_MAX_CANDIDATES_PER_INTENT)),
        "свет",
    )
    slots = {"name": values}
    sources = {
        "builtin": {
            "intents": {
                "HassTurnOn": {"data": [{"sentences": ["включи{name}"]}]},
            }
        }
    }
    stats = gl.RegistryRetrievalStats()

    candidates = build_query_registry_candidates(
        "ru",
        sources,
        slots,
        "включисвет",
        registry_slot_index=build_registry_slot_index(slots, "ru"),
        compiled_intents=compile_dynamic_registry_intents(sources, "ru"),
        retrieval_stats=stats,
    )

    assert [candidate.text for candidate in candidates] == ["включисвет"]
    assert stats.values_scored <= gl.DEFAULT_MAX_REGISTRY_VALUES_SCORED_PER_QUERY


def test_query_registry_candidates_retain_unanchored_spaced_filler_fallback() -> None:
    """Preserve ordinary token retrieval when conversational filler shifts an anchor."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {"data": [{"sentences": ["turn on {name}"]}]},
                }
            }
        },
        {"name": ("kitchen light",)},
        "please turn on kitchen light",
    )

    assert [candidate.text for candidate in candidates] == ["turn on kitchen light"]
    assert "registry_retrieval" not in candidates[0].metadata


def test_query_registry_candidates_retain_multislot_compact_fallback() -> None:
    """Keep the bounded compact-script fallback when one slot window is ambiguous."""
    candidates = build_query_registry_candidates(
        "zh",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {"data": [{"sentences": ["打开{area}{name}"]}]},
                }
            }
        },
        {"area": ("卧室",), "name": ("床头灯",)},
        "打开卧室床头灯",
    )

    assert [candidate.text for candidate in candidates] == ["打开卧室床头灯"]


@pytest.mark.parametrize(
    "text",
    [
        "打开灯",
        "ライトをつけて",
        "거실등켜줘",
        "เปิดไฟ",
        "ເປີດໄຟ",
        "បើកភ្លើង",
        "မီးဖွင့်",
    ],
)
def test_compact_script_detection_keeps_no_space_scripts(text: str) -> None:
    """Enable bounded substring spans only for scripts that commonly need them."""
    assert gl._uses_compact_non_latin_script(text)


@pytest.mark.parametrize(
    "text",
    [
        "включисвет",
        "άναψεφως",
        "شغلالضوء",
        "הדלקאור",
        "बत्तीजलाओ",
        "միացրուլույսը",
        "ჩართეშუქი",
    ],
)
def test_compact_script_detection_rejects_space_delimited_scripts(text: str) -> None:
    """Do not allow arbitrary substrings in scripts with lexical word boundaries."""
    assert not gl._uses_compact_non_latin_script(text)


def test_compact_phrase_matching_respects_script_word_boundaries() -> None:
    """Keep Han substring support without extending it to Cyrillic words."""
    assert gl._normalized_phrase_occurs_in_query("卧室灯", "打开卧室灯")
    assert not gl._normalized_phrase_occurs_in_query("свет", "включисвет")


def test_registry_slot_values_for_slots_preserves_requested_order() -> None:
    """Deduplicate requested slots without losing deterministic insertion order."""
    selected = gl._registry_slot_values_for_slots(
        {
            "area": ("kitchen",),
            "floor": ("upstairs",),
            "name": ("speaker",),
            "name:fan": ("bathroom fan",),
        },
        ("floor", "name", "area", "name"),
        domains=("fan",),
    )

    assert list(selected) == ["floor", "name", "area"]
    assert selected["name"] == ("bathroom fan",)


def test_query_registry_candidates_allow_slot_only_optional_literal_template() -> None:
    """Match a template when the query uses its slot-only optional-literal form."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [{"sentences": ["{name} [please]"]}],
                    }
                }
            }
        },
        {"name": ("kitchen light",)},
        "kitchen light",
    )

    assert "kitchen light" in [candidate.text for candidate in candidates]


def test_query_registry_candidates_include_area_only_templates() -> None:
    """Build query-scoped candidates for registry templates with no entity slot."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [{"sentences": ["turn on {area} light"]}],
                    }
                }
            }
        },
        {"area": ("kitchen",)},
        "turn on kitchen light",
    )

    assert [candidate.text for candidate in candidates] == ["turn on kitchen light"]


def test_query_registry_candidates_deduplicate_text_intent_pairs() -> None:
    """Return one dynamic candidate for duplicate text within the same intent."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {"sentences": ["turn on {name}", "turn on {entity}"]},
                        ]
                    }
                }
            }
        },
        {
            "name": ("kitchen light",),
            "entity": ("kitchen light",),
        },
        "turn on kitchen light",
    )

    assert [(candidate.intent_name, candidate.text) for candidate in candidates] == [
        ("HassTurnOn", "turn on kitchen light")
    ]


def test_query_registry_candidates_prefer_hassil_domain_area_duplicate() -> None:
    """Prefer the production HassIL domain-area parse for same-source duplicates."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn on {area} fan"],
                                "slots": {"domain": "fan", "name": "all"},
                            },
                            {"sentences": ["turn on {name}"]},
                        ]
                    }
                }
            }
        },
        {
            "name": ("bathroom fan",),
            "name:fan": ("bathroom fan",),
            "area": ("bathroom",),
        },
        "turn on bathroom fan",
        include_literal_only_templates=False,
        include_area_only_templates=False,
    )

    matching = [
        candidate
        for candidate in candidates
        if candidate.intent_name == "HassTurnOn"
        and candidate.normalized_text == "turn on bathroom fan"
    ]
    assert len(matching) == 1

    slots = orjson.loads(matching[0].metadata["slots"])
    assert slots == {"area": "bathroom", "domain": "fan", "name": "all"}


@pytest.mark.current_intents
def test_query_registry_duplicate_preference_matches_hassil_builtin_order() -> None:
    """Keep exact dynamic duplicate selection aligned with HassIL built-in parsing."""
    examples = (
        (
            "en",
            "turn on bathroom fan",
            {
                "name": ("bathroom fan",),
                "name:fan": ("bathroom fan",),
                "area": ("bathroom",),
            },
        ),
        (
            "vi",
            "bật quạt phòng khách",
            {
                "name": ("quạt phòng khách",),
                "name:fan": ("quạt phòng khách",),
                "area": ("phòng khách",),
            },
        ),
    )

    for language, query, slots in examples:
        sources = load_language_intent_sources(language)
        merged_intents: dict[str, Any] = {}
        for source in sources.values():
            hassil.merge_dict(merged_intents, source)
        hassil_results = _run_hassil_recognize_all(
            query,
            hassil.intents.Intents.from_dict(merged_intents),
            _make_hassil_slot_lists(slots),
        )
        assert hassil_results
        hassil_result = hassil_results[0]
        hassil_slots = {name: entity.value for name, entity in hassil_result.entities.items()}

        candidates = build_query_registry_candidates(
            language,
            sources,
            slots,
            query,
            registry_slot_index=build_registry_slot_index(slots, language),
            compiled_intents=compile_dynamic_registry_intents(sources, language),
        )
        matching = [
            candidate
            for candidate in candidates
            if candidate.intent_name == hassil_result.intent.name
            and candidate.normalized_text == normalize_text(query)
        ]
        assert len(matching) == 1
        candidate_slots = orjson.loads(matching[0].metadata["slots"])

        assert {
            slot_name: candidate_slots.get(slot_name) for slot_name in hassil_slots
        } == hassil_slots


def test_query_registry_candidates_keep_custom_source_priority() -> None:
    """Prefer custom dynamic candidates before structural duplicate preferences."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "built_in": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn on {area} fan"],
                                "slots": {"domain": "fan", "name": "all"},
                            },
                        ]
                    }
                }
            },
            "custom_sentence": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {"sentences": ["turn on {name}"]},
                        ]
                    }
                }
            },
        },
        {
            "name": ("bathroom fan",),
            "name:fan": ("bathroom fan",),
            "area": ("bathroom",),
        },
        "turn on bathroom fan",
        include_literal_only_templates=False,
        include_area_only_templates=False,
    )

    matching = [
        candidate
        for candidate in candidates
        if candidate.intent_name == "HassTurnOn"
        and candidate.normalized_text == "turn on bathroom fan"
    ]
    assert len(matching) == 1
    assert matching[0].source == CandidateSource.CUSTOM_SENTENCE


def test_query_registry_candidates_reject_domain_area_action_typos() -> None:
    """Avoid domain-area rescues when the action literal is not fully present."""
    sources = {
        "builtin": {
            "intents": {
                "HassTurnOn": {
                    "data": [
                        {
                            "sentences": ["turn on {area} fan"],
                            "slots": {"domain": "fan", "name": "all"},
                        },
                    ]
                }
            }
        }
    }
    slots = {"area": ("bathroom",)}

    exact = build_query_registry_candidates(
        "en",
        sources,
        slots,
        "turn on bathroom fan",
        include_literal_only_templates=False,
        include_area_only_templates=False,
    )
    fuzzy = build_query_registry_candidates(
        "en",
        sources,
        slots,
        "trun on bathroom fan",
        include_literal_only_templates=False,
        include_area_only_templates=False,
    )

    assert [candidate.text for candidate in exact] == ["turn on bathroom fan"]
    assert fuzzy == ()


def test_compile_dynamic_registry_intents_keeps_domain_area_templates_when_disabled() -> None:
    """Keep domain-scoped area templates while excluding generic area-only templates."""
    compiled = gl.compile_dynamic_registry_intents(
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn on fan {area}"],
                                "slots": {"domain": "fan", "name": "all"},
                            },
                            {"sentences": ["turn on {area}"]},
                        ]
                    }
                }
            }
        },
        "en",
        include_literal_only_templates=False,
        include_area_only_templates=False,
    )

    sentences = [template.sentence for intent in compiled for template in intent.templates]
    assert sentences == ["turn on fan {area}"]


def test_query_registry_candidates_floor_slot() -> None:
    """Build query-scoped candidates for registry templates with a floor slot."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [{"sentences": ["turn on {floor} lights"]}],
                    }
                }
            }
        },
        {"floor": ("upstairs",)},
        "turn on upstairs lights",
    )

    assert [candidate.text for candidate in candidates] == ["turn on upstairs lights"]


def test_query_registry_candidates_include_literal_only_templates_without_registry() -> None:
    """Build exact query candidates for base list templates even without registry slots."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "lists": {"timer_seconds": {"range": {"from": 1, "to": 2}}},
                "intents": {
                    "HassStartTimer": {
                        "data": [{"sentences": ["timer for {timer_seconds:seconds}( |-)second[s]"]}]
                    }
                },
            }
        },
        {},
        "timer for 1 second",
        include_literal_only_templates=True,
        include_area_only_templates=False,
    )

    assert candidates[0].text == "timer for 1 second"
    slots = orjson.loads(candidates[0].metadata["slots"])
    assert slots == {"seconds": 1}


def test_query_registry_candidates_skip_non_wildcard_literal_rescue_work() -> None:
    """Avoid expanding base-list templates during wildcard-only runtime rescue."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "lists": {"color": {"values": ["white"]}},
                "intents": {
                    "HassLightSet": {
                        "data": [{"sentences": ["turn lights {color}"]}],
                    }
                },
            }
        },
        {},
        "turn lights",
        literal_only_wildcards_only=True,
    )

    assert candidates == ()


def test_query_registry_candidates_excludes_floor_only_when_disabled() -> None:
    """Suppress floor-only templates when include_area_only_templates is False."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [{"sentences": ["turn on {floor} lights"]}],
                    }
                }
            }
        },
        {"floor": ("upstairs",)},
        "turn on upstairs lights",
        include_literal_only_templates=False,
        include_area_only_templates=False,
    )

    assert candidates == ()


def test_query_registry_candidates_domain_floor_rescue() -> None:
    """Domain-scoped floor template is rescued as exact match when area-only is disabled."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn on {floor} fan"],
                                "slots": {"domain": "fan", "name": "all"},
                            },
                        ]
                    }
                }
            }
        },
        {"floor": ("upstairs",)},
        "turn on upstairs fan",
        include_literal_only_templates=False,
        include_area_only_templates=False,
    )

    assert [candidate.text for candidate in candidates] == ["turn on upstairs fan"]
    slots = orjson.loads(candidates[0].metadata["slots"])
    assert slots["domain"] == "fan"
    assert slots["floor"] == "upstairs"


def test_compile_dynamic_registry_intents_keeps_domain_floor_templates_when_disabled() -> None:
    """Keep domain-scoped floor templates while excluding generic floor-only templates."""
    compiled = gl.compile_dynamic_registry_intents(
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [
                            {
                                "sentences": ["turn on fan {floor}"],
                                "slots": {"domain": "fan", "name": "all"},
                            },
                            {"sentences": ["turn on {floor}"]},
                        ]
                    }
                }
            }
        },
        "en",
        include_literal_only_templates=False,
        include_area_only_templates=False,
    )

    sentences = [template.sentence for intent in compiled for template in intent.templates]
    assert sentences == ["turn on fan {floor}"]


def test_query_registry_candidates_mixed_entity_floor_prunes_floor_values() -> None:
    """Mixed entity+floor template does not generate O(entities x floors) candidates."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "intents": {
                    "HassTurnOn": {
                        "data": [{"sentences": ["turn on {name} on {floor}"]}],
                    }
                }
            }
        },
        {
            "name": ("kitchen light", "bedroom light"),
            "floor": ("upstairs", "downstairs"),
        },
        "turn on kitchen light",
    )

    # Floor values are pruned from constrained, and because {floor} is a required slot the
    # template produces no candidates at all, so no O(entities x floors) cross-product occurs.
    assert candidates == ()


@pytest.mark.parametrize(
    (
        "list_name",
        "range_dict",
        "intent_name",
        "sentence",
        "query",
        "expected_text",
        "expected_slots",
    ),
    [
        (
            "temperature",
            {"type": "temperature", "from": 0, "to": 100, "fractions": "halves"},
            "HassClimateSetTemperature",
            "set temperature to {temperature}",
            "set temperature to 27.5°",
            "set temperature to 27.5°",
            {"temperature": 27.5},
        ),
        (
            "temperature",
            {"type": "temperature", "from": 0, "to": 100, "fractions": "halves"},
            "HassClimateSetTemperature",
            "set temperature to {temperature}",
            "set temperature to 27.5 °",
            "set temperature to 27.5",
            {"temperature": 27.5},
        ),
        (
            "brightness",
            {"type": "percentage", "from": 0, "to": 100},
            "HassLightSet",
            "set brightness to {brightness}",
            "set brightness to 50%",
            "set brightness to 50%",
            {"brightness": 50},
        ),
        (
            "brightness",
            {"type": "percentage", "from": 0, "to": 100},
            "HassLightSet",
            "set brightness to {brightness}",
            "set brightness to 50 %",
            "set brightness to 50",
            {"brightness": 50},
        ),
    ],
)
def test_query_registry_candidates_with_suffixes(
    list_name: str,
    range_dict: dict[str, Any],
    intent_name: str,
    sentence: str,
    query: str,
    expected_text: str,
    expected_slots: dict[str, Any],
) -> None:
    """Validate queries containing numbers with degree or percentage suffixes."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "lists": {list_name: {"range": range_dict}},
                "intents": {intent_name: {"data": [{"sentences": [sentence]}]}},
            }
        },
        {},
        query,
        max_candidates=1,
    )
    assert [candidate.text for candidate in candidates] == [expected_text]
    assert orjson.loads(candidates[0].metadata["slots"]) == expected_slots


def test_query_registry_candidates_with_overlapping_ranges_anchoring() -> None:
    """Validate overlapping range slots anchoring when multiple qualifying numbers are present."""
    candidates = build_query_registry_candidates(
        "en",
        {
            "builtin": {
                "lists": {
                    "temperature": {"range": {"from": 0, "to": 100}},
                    "area_num": {"range": {"from": 0, "to": 100}},
                },
                "intents": {
                    "HassClimateSetTemperature": {
                        "data": [
                            {"sentences": ["set temperature in area {area_num} to {temperature}"]}
                        ]
                    }
                },
            }
        },
        {},
        "set temperature in area 51 to 20",
        max_candidates=1,
    )
    assert len(candidates) == 1
    slots = orjson.loads(candidates[0].metadata["slots"])
    assert slots.get("temperature") == 20
    assert slots.get("area_num") == 51


@pytest.mark.parametrize(
    "query",
    [
        "set temperature to 22 in area 51",
        "set temperature 22 in area 51",
    ],
)
def test_query_registry_candidates_anchoring_with_optional_tokens(query: str) -> None:
    """Validate range slot resolution works when preceded by optional tokens.

    And query is ambiguous.
    """
    intent_sources = {
        "builtin": {
            "lists": {
                "temperature": {"range": {"from": 0, "to": 100}},
                "area_num": {"range": {"from": 0, "to": 100}},
            },
            "intents": {
                "HassClimateSetTemperature": {
                    "data": [
                        {"sentences": ["set temperature [to] {temperature} in area {area_num}"]}
                    ]
                }
            },
        }
    }

    candidates = build_query_registry_candidates(
        "en",
        intent_sources,
        {},
        query,
        max_candidates=1,
    )
    assert len(candidates) == 1
    slots = orjson.loads(candidates[0].metadata["slots"])
    assert slots.get("temperature") == 22
    assert slots.get("area_num") == 51


def test_query_registry_candidates_anchoring_with_mapped_slots() -> None:
    """Validate range slot anchoring works for slot references.

    With custom list prefixes (e.g. list:slot).
    """
    intent_sources = {
        "builtin": {
            "lists": {
                "timer_minutes": {"range": {"from": 1, "to": 100}},
                "timer_seconds": {"range": {"from": 1, "to": 100}},
            },
            "intents": {
                "HassStartTimer": {
                    "data": [
                        {
                            "sentences": [
                                "timer for {timer_minutes:minutes} minutes "
                                "[and] {timer_seconds:seconds} seconds"
                            ]
                        }
                    ]
                }
            },
        }
    }

    candidates = build_query_registry_candidates(
        "en",
        intent_sources,
        {},
        "timer for 5 minutes and 30 seconds",
        max_candidates=1,
    )
    assert len(candidates) == 1
    slots = orjson.loads(candidates[0].metadata["slots"])
    assert slots.get("minutes") == 5
    assert slots.get("seconds") == 30


def test_expand_sentence_template_nesting_limit() -> None:
    """Verify that templates exceeding the maximum nesting depth raise ValueError."""
    # Nesting depth: 26 levels of brackets '['
    deep_nested = "[" * 26 + "hello" + "]" * 26
    with pytest.raises(ValueError, match=r"Max template nesting depth \(\d+\) exceeded"):
        expand_sentence_template(deep_nested, {}, {})


def test_compact_phrase_matching_tolerates_segmentation_spaces() -> None:
    """Match unsegmented grammar literals against space-segmented STT output."""
    assert gl._normalized_phrase_occurs_in_query("卧室灯", "打开 卧室灯")
    assert gl._normalized_phrase_occurs_in_query("卧室灯", "打开 卧室 灯")
    # Latin scripts keep strict token-boundary semantics.
    assert not gl._normalized_phrase_occurs_in_query("lamp", "turn on the lampshade")


def test_compact_phrase_matching_can_cross_segmentation_boundaries() -> None:
    """Document the accepted precision trade-off of stripping STT spaces.

    STT segmentation for compact scripts is unreliable, so spaces are ignored
    when searching for grammar phrases. The cost is that a phrase may match
    across what was a genuine boundary; this only feeds bounded registry
    nomination, and downstream scoring still gates the candidate.
    """
    assert gl._normalized_phrase_occurs_in_query("京都", "東京 都心")


def test_unknown_expansion_rule_expands_as_empty_text(caplog: pytest.LogCaptureFixture) -> None:
    """A sentence referencing an undefined rule keeps its other segments."""
    gl._warn_unknown_expansion_rule.cache_clear()
    node = gl._parse_hassil("turn on <missing> light")
    with caplog.at_level("WARNING"):
        expanded = node.expand({}, {}, frozenset(), 10)
    assert expanded == ("turn on  light",)
    assert any("missing" in record.message for record in caplog.records)
    # The warning is deduplicated per rule name.
    with caplog.at_level("WARNING"):
        node.expand({}, {}, frozenset(), 10)
    assert sum("missing" in record.message for record in caplog.records) == 1
    gl._warn_unknown_expansion_rule.cache_clear()


def test_self_recursive_expansion_rule_terminates_with_finite_variants() -> None:
    """Recursive rule references terminate as empty text instead of pruning."""
    rules = {"loop": "very <loop>"}
    node = gl._parse_hassil("<loop> bright")
    expanded = node.expand({}, rules, frozenset(), 10)
    assert expanded == ("very  bright",)


def test_range_endpoint_outputs_apply_multiplier() -> None:
    """Scale spoken range endpoints by the multiplier like interior values."""
    intent_sources = _build_range_intent_source(
        "volume_step_down",
        {"type": "percentage", "from": 0, "to": 100, "multiplier": -1},
        "HassSetVolumeRelative",
        ["volume down by {volume_step_down:volume_step}"],
    )
    candidates = build_query_registry_candidates(
        "en",
        intent_sources,
        {},
        "volume down by 100",
        max_candidates=1,
    )

    assert [candidate.text for candidate in candidates] == ["volume down by 100"]
    assert orjson.loads(candidates[0].metadata["slots"]) == {"volume_step": -100}


def test_build_candidates_drop_empty_normalized_text() -> None:
    """Drop expansions whose text normalizes to empty (pure punctuation)."""
    candidates = build_candidates_from_intent_sources(
        "en",
        {
            "builtin": {
                "intents": {"HassTurnOn": {"data": [{"sentences": ["...", "turn on the light"]}]}}
            }
        },
    )

    assert [candidate.text for candidate in candidates] == ["turn on the light"]


def test_is_fixed_sentence_treats_semicolon_as_template() -> None:
    """Bare semicolons are HassIL permutation separators, not literal text."""
    assert not is_fixed_sentence("schalte das licht an;im wohnzimmer")


def test_domain_aware_candidate_pre_filtering() -> None:
    """Filter out data items requiring unavailable domains while keeping domainless ones."""
    intent_sources = {
        "builtin": {
            "intents": {
                "HassTurnOn": {
                    "data": [
                        {
                            "sentences": ["turn on the light"],
                            "slots": {"domain": "light"},
                        },
                        {
                            "sentences": ["turn on the TV"],
                            "slots": {"domain": "media_player"},
                        },
                        {
                            "sentences": ["open the cover"],
                            "inferred_domain": "cover",
                        },
                        {
                            "sentences": ["lock the door"],
                            "name_domains": ["lock"],
                        },
                        {
                            "sentences": ["turn on generic"],
                        },
                    ]
                }
            }
        }
    }

    # Available domains: only 'light' is in registry_slot_values
    registry_slots_light_only = {
        "name:light": ("kitchen light",),
    }

    candidates = build_candidates_from_intent_sources(
        "en",
        intent_sources,
        registry_slots_light_only,
    )

    candidate_texts = [c.text for c in candidates]
    assert "turn on the light" in candidate_texts
    assert "turn on generic" in candidate_texts
    assert "turn on the TV" not in candidate_texts
    assert "open the cover" not in candidate_texts
    assert "lock the door" not in candidate_texts

    # With only unscoped registry mappings (e.g., 'name'), domain filtering is bypassed
    unscoped_registry_slots = {
        "name": ("generic name",),
    }
    candidates_unscoped = build_candidates_from_intent_sources(
        "en",
        intent_sources,
        unscoped_registry_slots,
    )
    all_texts = [c.text for c in candidates_unscoped]
    assert "turn on the light" in all_texts
    assert "turn on generic" in all_texts
    assert "turn on the TV" in all_texts
    assert "open the cover" in all_texts
    assert "lock the door" in all_texts
