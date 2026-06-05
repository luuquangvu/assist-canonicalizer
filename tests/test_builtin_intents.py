"""Tests for language intent source loading."""

import os
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import orjson
import pytest

from custom_components.assist_canonicalizer.builtin_intents import (
    _json_load,
    _load_built_in_intents,
    _load_custom_sentences,
    language_variant_for,
    load_language_intent_sources,
)
from custom_components.assist_canonicalizer.runtime import CanonicalizerRuntime


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
    assert sources["config"]["intents"]["CustomIntent"]["data"][0]["sentences"] == [
        "activate movie mode"
    ]


def test_load_language_intent_sources_tolerates_missing_optional_packages() -> None:
    """Missing HA intent packages should produce no built-in sources instead of failing."""
    assert isinstance(load_language_intent_sources("zz"), dict)


def test_language_variant_for_invalid() -> None:
    """Test language variant matching with empty language input."""
    assert language_variant_for(" ") is None


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

        def mock_config_path(key: str, lang: str) -> str:
            return os.path.join(tmpdir, key, lang)

        res = _load_custom_sentences("vi", mock_config_path)
        assert "intents" in res
        assert res["intents"]["HassTurnOn"]["data"][0]["sentences"] == ["bật {name}"]


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
    def mock_get_intents(lang: str, json_load: Any = None) -> Any:
        if json_load is not None:
            raise TypeError("json_load not supported")
        return {"built_in_key": "built_in_val"}

    mock_module.get_intents = mock_get_intents

    with patch(
        "custom_components.assist_canonicalizer.builtin_intents.import_module",
        return_value=mock_module,
    ):
        assert _load_built_in_intents("vi") == {"built_in_key": "built_in_val"}


def test_load_custom_sentences_import_error() -> None:
    """Test _load_custom_sentences returns empty dict on yaml ImportError."""
    with patch.dict(sys.modules, {"yaml": None}):
        assert _load_custom_sentences("vi", lambda key, lang: "/some/path") == {}


def test_load_custom_sentences_yaml_types_and_recursive_merge() -> None:
    """Test custom sentences yaml loading with lists and recursive dict merging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # File 1: valid dict sentence config
        yaml_content_1 = """
        intents:
          HassTurnOn:
            data:
              - sentences:
                  - "bật {name}"
        """
        # File 2: valid dict sentence config to merge recursively
        yaml_content_2 = """
        intents:
          HassTurnOn:
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

        def mock_config_path(key: str, lang: str) -> str:
            return os.path.join(tmpdir, key, lang)

        res = _load_custom_sentences("vi", mock_config_path)
        assert "intents" in res
        assert res["intents"]["HassTurnOn"]["data"][0]["sentences"] == ["bật {name}"]
        assert res["intents"]["HassTurnOn"]["other_data"] == 123
