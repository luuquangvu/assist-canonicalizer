"""Tests for automatic conversation intent candidate loading."""

from custom_components.assist_canonicalizer import grammar_loader as gl
from custom_components.assist_canonicalizer.candidate import CandidateSource
from custom_components.assist_canonicalizer.grammar_loader import (
    build_candidates_from_intent_sources,
    expand_sentence_template,
    is_fixed_sentence,
)


def test_is_fixed_sentence_rejects_hassil_templates() -> None:
    """Reject sentence templates that need slot expansion."""
    assert is_fixed_sentence("turn on kitchen light")
    assert not is_fixed_sentence("turn on {name}")
    assert not is_fixed_sentence("turn [the] light on")


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
    assert [candidate.text for candidate in candidates] == [
        "turn kitchen light on",
        "turn desk lamp on",
        "turn office lamp on",
        "turn the kitchen light on",
        "turn the desk lamp on",
        "turn the office lamp on",
    ]
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


def test_build_candidates_uses_domain_scoped_registry_names() -> None:
    """Use Hassil domain context to choose entity slot values."""
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


def test_build_candidates_uses_multi_domain_context_scoped_registry_names() -> None:
    """Expand entity slots from all domains listed in Hassil context."""
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

    assert [candidate.text for candidate in candidates] == [
        "bật Quạt thông gió phòng tắm to",
        "bật Đèn phòng tắm to",
    ]


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
