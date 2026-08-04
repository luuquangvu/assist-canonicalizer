"""Tests for the generated all-language compatibility-smoke manifest."""

from __future__ import annotations

import unicodedata
from collections import Counter
from types import SimpleNamespace

import home_assistant_intents
import pytest
from hassil import parse_sentence
from hassil.errors import MissingListError, MissingRuleError
from hassil.intents import TextSlotList

from custom_components.assist_canonicalizer.normalization import normalize_text
from tools.benchmark_language_smoke import (
    ACCURACY_GATED_LANGUAGES,
    HAS_SAMPLE_SENTENCE,
    LANGUAGE_ENTITY_NAMES,
    _fixture_sentence_sample,
    _intent_data_supports_domain,
    build_language_smoke_commands,
)

pytestmark = pytest.mark.skipif(
    not HAS_SAMPLE_SENTENCE,
    reason="hassil.sample.sample_sentence is not available in the installed version of hassil",
)


def test_accuracy_gated_language_manifest_is_explicit() -> None:
    """Keep corpus-backed accuracy claims limited to the reviewed five languages."""
    assert {"de", "en", "fr", "nl", "vi"} == ACCURACY_GATED_LANGUAGES


@pytest.mark.parametrize(
    "error",
    [
        MissingListError("missing list"),
        MissingRuleError("missing rule"),
    ],
)
def test_fixture_sentence_sample_skips_missing_grammar_references(error: Exception) -> None:
    """Skip a sentence whose fixture-slot discovery references missing grammar."""

    class MissingGrammarSentence:
        """Raise the supplied missing-grammar error during list discovery."""

        class Expression:
            """Model the Hassil expression list-reference API."""

            @staticmethod
            def list_references(_expansion_rules: object) -> tuple[object, ...]:
                """Return referenced slot lists for expression."""
                raise error

        expression = Expression()

    assert (
        _fixture_sentence_sample(
            MissingGrammarSentence(),
            {},
            {},
            {"name": "lamp"},
            "en",
        )
        is None
    )


def test_fixture_sentence_sample_rejects_uncontrolled_remapped_target_slot() -> None:
    """Reject a built-in list whose value is emitted as a fixture target slot."""
    assert (
        _fixture_sentence_sample(
            parse_sentence("{default_areas:area} {name}"),
            {
                "default_areas": TextSlotList.from_strings(["Upstairs"]),
                "name": TextSlotList.from_strings(["Living Room Lamp"]),
            },
            {},
            {
                "area": "Living Room",
                "name": "Living Room Lamp",
            },
            "en",
        )
        is None
    )


def test_fixture_sentence_sample_excludes_skipped_optional_slots() -> None:
    """Derive expected slots from the selected sample, not the full expression."""
    assert _fixture_sentence_sample(
        parse_sentence("{name} [in {area}]"),
        {
            "name": TextSlotList.from_strings(["Living Room Lamp"]),
            "area": TextSlotList.from_strings(["Living Room"]),
        },
        {},
        {
            "name": "Living Room Lamp",
            "area": "Living Room",
        },
        "en",
    ) == ("name", frozenset({"name"}), "Living Room Lamp")


@pytest.mark.parametrize(
    ("requires_domain", "excludes_domain", "target_domain", "expected"),
    [
        ({"value": "light", "slot": True}, None, "light", True),
        ({"value": "light", "slot": True}, None, "switch", False),
        (["light", "switch"], None, "switch", True),
        (None, {"value": ["light", "switch"]}, "light", False),
        ({"value": {"unsupported": "light"}}, None, "light", False),
    ],
)
def test_intent_data_supports_normalized_domain_directives(
    requires_domain: object,
    excludes_domain: object,
    target_domain: str,
    expected: bool,
) -> None:
    """Evaluate scalar, list, and mapping-style domain directives safely."""
    intent_data = SimpleNamespace(
        slots={},
        requires_context=({} if requires_domain is None else {"domain": requires_domain}),
        excludes_context=({} if excludes_domain is None else {"domain": excludes_domain}),
    )

    assert _intent_data_supports_domain(intent_data, target_domain) is expected


def test_every_installed_language_has_a_deterministic_fixture_command() -> None:
    """Generate one exact managed-live command for every installed language variant."""
    first = build_language_smoke_commands()
    second = build_language_smoke_commands()

    assert first == second
    assert tuple(command.language for command in first) == tuple(
        home_assistant_intents.get_languages()
    )
    assert all(command.text.strip() for command in first)
    assert all(command.expected_slots for command in first)
    assert all(set(dict(command.expected_slots)) <= {"name", "area", "floor"} for command in first)
    assert len({command.language for command in first}) == len(first)


def test_every_installed_language_preserves_script_forming_marks() -> None:
    """Keep marks in generated commands for every installed language variant."""
    for command in build_language_smoke_commands():
        source = unicodedata.normalize("NFKC", command.text).casefold()
        expected_marks = Counter(
            char for char in source if unicodedata.category(char).startswith("M")
        )
        normalized_marks = Counter(
            char
            for char in normalize_text(command.text)
            if unicodedata.category(char).startswith("M")
        )

        assert normalized_marks == expected_marks, command.language


def test_script_specific_fixture_aliases_are_used() -> None:
    """Use provisioned names where Latin target boundaries are not live-recognizable."""
    commands = {command.language: command.text for command in build_language_smoke_commands()}

    assert all(
        name in commands[language]
        for language, name in LANGUAGE_ENTITY_NAMES.items()
        if language in commands
    )
