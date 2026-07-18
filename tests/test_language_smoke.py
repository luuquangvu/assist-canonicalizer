"""Tests for the generated all-language compatibility-smoke manifest."""

from __future__ import annotations

import unicodedata
from collections import Counter

import home_assistant_intents
import pytest

from custom_components.assist_canonicalizer.normalization import normalize_text
from tools.benchmark_language_smoke import (
    ACCURACY_GATED_LANGUAGES,
    HAS_SAMPLE_SENTENCE,
    LANGUAGE_ENTITY_NAMES,
    build_language_smoke_commands,
)

pytestmark = pytest.mark.skipif(
    not HAS_SAMPLE_SENTENCE,
    reason="hassil.sample.sample_sentence is not available in the installed version of hassil",
)


def test_accuracy_gated_language_manifest_is_explicit() -> None:
    """Keep corpus-backed accuracy claims limited to the reviewed five languages."""
    assert {"de", "en", "fr", "nl", "vi"} == ACCURACY_GATED_LANGUAGES


def test_every_installed_language_has_a_deterministic_fixture_command() -> None:
    """Generate one exact managed-live command for every installed language variant."""
    first = build_language_smoke_commands()
    second = build_language_smoke_commands()

    assert first == second
    assert tuple(command.language for command in first) == tuple(
        home_assistant_intents.get_languages()
    )
    assert all(command.text.strip() for command in first)
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
