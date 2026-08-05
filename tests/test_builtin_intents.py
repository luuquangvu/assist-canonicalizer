"""Tests for language intent source loading."""

import os
import sys
import tempfile
from collections.abc import Iterator
from functools import partial
from typing import Any, cast
from unittest.mock import MagicMock, patch

import orjson
import pytest

from custom_components.assist_canonicalizer.builtin_intents import (
    _json_load,
    _load_built_in_intents,
    _load_custom_sentences,
    clear_builtin_intents_caches,
    language_variant_for,
    load_language_intent_sources,
)
from custom_components.assist_canonicalizer.candidate import CandidateSource
from custom_components.assist_canonicalizer.grammar_loader import (
    build_candidates_from_intent_sources,
)
from custom_components.assist_canonicalizer.runtime import CanonicalizerRuntime


def _as_dict(obj: object) -> dict[str, Any]:
    """Cast a dynamic test object to a dictionary for subscript assertions."""
    return cast(dict[str, Any], obj)


@pytest.fixture(autouse=True)
def _clear_builtin_intents_cache() -> Iterator[None]:
    """Clear cached built-in intent sources before and after each test."""
    clear_builtin_intents_caches()
    try:
        yield
    finally:
        clear_builtin_intents_caches()


def _config_path_in_tmpdir(tmpdir: str, key: str, lang: str) -> str:
    """Return a test config path inside a temporary directory."""
    return os.path.join(tmpdir, key, lang)


def _mock_get_intents_json_load_fallback(lang: str, json_load: Any = None) -> Any:
    """Mock get_intents that rejects the json_load keyword once."""
    if json_load is not None:
        raise TypeError("json_load not supported")
    return {"built_in_key": "built_in_val"}


def test_runtime_merges_language_sources_with_subscribed_sources() -> None:
    """Runtime should keep subscribed sources even without optional built-in packages."""
    runtime = CanonicalizerRuntime()
    runtime.update_intent_sources(
        {
            "config": {
                "intents": {"CustomIntent": {"data": [{"sentences": ["activate movie mode"]}]}}
            }
        }
    )
    sources = runtime._all_intent_sources("zz")
    config_source = _as_dict(_as_dict(sources)["config"])
    assert config_source["intents"]["CustomIntent"]["data"][0]["sentences"] == [
        "activate movie mode"
    ]


def test_load_language_intent_sources_tolerates_missing_optional_packages() -> None:
    """Missing HA intent packages should produce no built-in sources instead of failing."""
    assert isinstance(load_language_intent_sources("zz"), dict)


def test_language_variant_for_invalid() -> None:
    """Test language variant matching with empty language input."""
    assert language_variant_for(" ") is None


def test_language_variant_for_resolves_equal_scores_deterministically() -> None:
    """Use lexical language-pack order when Home Assistant match scores tie."""
    mock_module = MagicMock()
    mock_module.get_languages.side_effect = [
        ("pt-BR", "pt"),
        ("pt", "pt-BR"),
        ("zh-TW", "zh-HK", "zh-CN"),
        ("zh-CN", "zh-HK", "zh-TW"),
    ]

    with patch.dict(sys.modules, {"home_assistant_intents": mock_module}):
        # language_variant_for caches per language code; clear between calls
        # so each assertion exercises a fresh package enumeration order.
        language_variant_for.cache_clear()
        assert language_variant_for("pt-AO") == "pt"
        language_variant_for.cache_clear()
        assert language_variant_for("pt-AO") == "pt"
        language_variant_for.cache_clear()
        assert language_variant_for("zh-SG") == "zh-CN"
        language_variant_for.cache_clear()
        assert language_variant_for("zh-SG") == "zh-CN"
    language_variant_for.cache_clear()


def test_json_load_invalid() -> None:
    """Test json load failure raises orjson.JSONDecodeError."""

    class FakeFile:
        """Fake file object for testing json load."""

        def read(self) -> bytes:
            """Return invalid json content."""
            return b"invalid json content"

    with pytest.raises(orjson.JSONDecodeError):
        _json_load(FakeFile())


def test_load_custom_sentences_empty_config() -> None:
    """Test loading custom sentences when config_path callback is None or path does not exist."""
    assert _load_custom_sentences("en", None) == {}
    assert _load_custom_sentences("en", lambda key, lang: "/non_existent/path/here") == {}


def test_load_custom_sentences_with_yaml() -> None:
    """Test loading and merging custom sentence YAML files from directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_content = """
        intents:
          HassTurnOn:
            data:
              - sentences:
                  - "bật {name}"
        """
        sub_dir = os.path.join(tmpdir, "custom_sentences", "vi")
        os.makedirs(sub_dir, exist_ok=True)
        yaml_file = os.path.join(sub_dir, "test.yaml")
        with open(yaml_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        res = _load_custom_sentences("vi", partial(_config_path_in_tmpdir, tmpdir))
        assert "intents" in res
        res_intents = _as_dict(res["intents"])
        assert res_intents["HassTurnOn"]["data"][0]["sentences"] == ["bật {name}"]


def test_language_variant_for_import_error() -> None:
    """Test language_variant_for fallback when modules fail to import."""
    with patch.dict(sys.modules, {"home_assistant_intents": None}):
        assert language_variant_for("vi") == "vi"


def test_load_built_in_intents_import_error() -> None:
    """Test _load_built_in_intents returns empty dict on ImportError."""
    with patch.dict(sys.modules, {"home_assistant_intents": None}):
        assert _load_built_in_intents("vi") == {}


def test_load_built_in_intents_type_error() -> None:
    """Test _load_built_in_intents fallback when get_intents raises TypeError."""
    mock_module = MagicMock()

    # first call raises TypeError, second call succeeds
    mock_module.get_intents = _mock_get_intents_json_load_fallback

    with patch.dict(sys.modules, {"home_assistant_intents": mock_module}):
        assert _load_built_in_intents("vi") == {"built_in_key": "built_in_val"}


def test_load_custom_sentences_yaml_types_and_recursive_merge() -> None:
    """Test custom sentences yaml loading with lists and recursive dict merging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # File 1: valid dict sentence config
        yaml_content_1 = """
        lists:
          modes:
            values:
              - "day"
        intents:
          HassTurnOn:
            data:
              - sentences:
                  - "bật {name}"
        """
        # File 2: valid dict sentence config to merge recursively
        yaml_content_2 = """
        lists:
          modes:
            values:
              - "night"
        intents:
          HassTurnOn:
            data:
              - sentences:
                  - "mở {name}"
            other_data: 123
        """
        # File 3: YAML containing a list (not a Mapping) to test non-mapping skip
        yaml_content_3 = """
        - not a mapping
        - list item
        """
        sub_dir = os.path.join(tmpdir, "custom_sentences", "vi")
        os.makedirs(sub_dir, exist_ok=True)

        with open(os.path.join(sub_dir, "test1.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_content_1)
        with open(os.path.join(sub_dir, "test2.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_content_2)
        with open(os.path.join(sub_dir, "test3.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_content_3)

        res = _load_custom_sentences("vi", partial(_config_path_in_tmpdir, tmpdir))
        res_dict = _as_dict(res)
        res_intents = _as_dict(res_dict["intents"])
        res_lists = _as_dict(res_dict["lists"])
        assert "intents" in res
        assert [item["sentences"] for item in res_intents["HassTurnOn"]["data"]] == [
            ["bật {name}"],
            ["mở {name}"],
        ]
        assert res_lists["modes"]["values"] == ["day", "night"]
        assert res_intents["HassTurnOn"]["other_data"] == 123


def test_load_language_sources_inherit_effective_grammar_without_mixing_provenance() -> None:
    """Give each source merged grammar context but retain only its own sentence data."""
    built_in = {
        "language": "en",
        "expansion_rules": {"turn": "(turn|switch)"},
        "lists": {"mode": {"values": ["day"]}},
        "intents": {
            "Demo": {
                "data": [{"sentences": ["<turn> {mode}"]}],
            }
        },
    }
    custom = {
        "language": "en",
        "intents": {
            "Demo": {
                "data": [{"sentences": ["custom <turn> {mode}"]}],
            }
        },
    }

    with (
        patch(
            "custom_components.assist_canonicalizer.builtin_intents._load_built_in_intents",
            return_value=built_in,
        ),
        patch(
            "custom_components.assist_canonicalizer.builtin_intents._load_custom_sentences",
            return_value=custom,
        ),
    ):
        sources = load_language_intent_sources("en")

    built_in_src = _as_dict(sources["built_in"])
    custom_src = _as_dict(sources["custom_sentence"])
    assert built_in_src["intents"]["Demo"]["data"] == built_in["intents"]["Demo"]["data"]
    assert custom_src["intents"]["Demo"]["data"] == custom["intents"]["Demo"]["data"]
    candidates = build_candidates_from_intent_sources("en", sources)
    custom_texts = {
        candidate.text for candidate in candidates if candidate.source.value == "custom_sentence"
    }
    assert custom_texts == {"custom turn day", "custom switch day"}


def test_load_language_sources_isolate_missing_data_and_preserve_top_level_context() -> None:
    """Do not borrow intent data while retaining the effective top-level context."""
    built_in = {
        "language": "en",
        "integration_context": {"built_in": True},
        "intents": {
            "BuiltInData": {
                "data": [{"sentences": ["built-in sentence"]}],
            },
            "CustomData": {
                "expansion_rules": {"built_in_rule": "built-in"},
            },
        },
    }
    custom = {
        "integration_context": {"custom": True},
        "custom_top_level_key": {"enabled": True},
        "intents": {
            "BuiltInData": {
                "expansion_rules": {"custom_rule": "custom"},
            },
            "CustomData": {
                "data": [{"sentences": ["custom sentence"]}],
            },
        },
    }

    with (
        patch(
            "custom_components.assist_canonicalizer.builtin_intents._load_built_in_intents",
            return_value=built_in,
        ),
        patch(
            "custom_components.assist_canonicalizer.builtin_intents._load_custom_sentences",
            return_value=custom,
        ),
    ):
        sources = load_language_intent_sources("en")

    expected_context = {"built_in": True, "custom": True}
    for source in sources.values():
        assert source["integration_context"] == expected_context
        assert source["custom_top_level_key"] == {"enabled": True}

    built_in_intents = _as_dict(_as_dict(sources["built_in"])["intents"])
    custom_intents = _as_dict(_as_dict(sources["custom_sentence"])["intents"])
    assert "data" not in built_in_intents["CustomData"]
    assert "data" not in custom_intents["BuiltInData"]

    candidates = build_candidates_from_intent_sources("en", sources)
    assert {(candidate.text, candidate.source) for candidate in candidates} == {
        ("built-in sentence", CandidateSource.BUILT_IN),
        ("custom sentence", CandidateSource.CUSTOM_SENTENCE),
    }
