"""Tests for the managed Home Assistant benchmark home."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import orjson
import pytest

from tools import benchmark
from tools.ha_dev.custom_components.assist_canonicalizer_benchmark import (
    _validate_manifest,
    fixture_fingerprint,
)

FIXTURE_PATH = Path("tools/ha_dev/custom_components/assist_canonicalizer_benchmark/fixture.json")
CONFIG_PATH = Path("tools/ha_dev/configuration.yaml")
EXPECTED_FINGERPRINT = "f63468a726b289243ca9ff6b0e387f8e470bc5c0c5239b051e2846cd93cbf9e8"
EXPECTED_DOMAIN_COUNTS = {
    "alarm_control_panel": 1,
    "binary_sensor": 3,
    "button": 1,
    "climate": 3,
    "cover": 5,
    "fan": 5,
    "humidifier": 2,
    "lawn_mower": 2,
    "light": 6,
    "lock": 3,
    "media_player": 3,
    "sensor": 8,
    "siren": 2,
    "switch": 2,
    "todo": 1,
    "vacuum": 5,
    "valve": 4,
    "water_heater": 2,
    "weather": 2,
}


def _manifest() -> dict[str, Any]:
    """Load the fixture manifest as a JSON object."""
    loaded = orjson.loads(FIXTURE_PATH.read_bytes())
    assert isinstance(loaded, dict)
    return loaded


def test_benchmark_fixture_contract_is_exact() -> None:
    """Keep the predefined home at its reviewed floor, area, and entity sizes."""
    manifest = _manifest()

    _validate_manifest(manifest)

    assert manifest["fixture_id"] == "medium_home_v1"
    assert manifest["expected_counts"] == {
        "floors": 3,
        "areas": 12,
        "exposed_entities": 60,
    }
    assert manifest["expected_domain_counts"] == EXPECTED_DOMAIN_COUNTS
    assert len(manifest["floors"]) == 3
    assert len(manifest["areas"]) == 12
    assert len(manifest["entities"]) == 60
    assert dict(Counter(entity["domain"] for entity in manifest["entities"])) == (
        EXPECTED_DOMAIN_COUNTS
    )


def test_benchmark_fixture_fingerprint_is_reviewed() -> None:
    """Require intentional baseline review for any user-visible model change."""
    assert fixture_fingerprint(_manifest()) == EXPECTED_FINGERPRINT


def test_benchmark_configuration_is_deterministic_and_loopback_only() -> None:
    """Keep the benchmark deterministic and move HTTP ownership out of YAML."""
    configuration = CONFIG_PATH.read_text(encoding="utf-8")

    assert "demo:" in configuration
    assert "assist_canonicalizer_benchmark:" in configuration
    assert not any(line.startswith("http:") for line in configuration.splitlines())
    assert benchmark.BASE_URL == "http://127.0.0.1:8123"
    assert benchmark.WEBSOCKET_URL == "ws://127.0.0.1:8123/api/websocket"


@pytest.mark.current_intents
def test_ha_benchmark_dependencies_are_unpinned_and_match_home_assistant() -> None:
    """Resolve benchmark packages from the active Home Assistant manifests."""
    declared = benchmark._benchmark_group_requirements()
    resolved = benchmark.verify_benchmark_dependencies()

    assert declared
    assert all(not requirement.specifier for requirement in declared)
    assert resolved["homeassistant"]
    assert set(resolved["packages"]) == {
        requirement.name.lower().replace("_", "-") for requirement in declared
    }


def test_real_world_corpus_is_the_full_managed_live_default() -> None:
    """Make all maintained real-world cases the authoritative live benchmark input."""
    suite_id, cases = benchmark.load_cases(benchmark.REAL_WORLD_DATASET_DIR)

    assert suite_id == "managed_live_real_world_v1"
    assert len(cases) >= 599
    assert {case.language for case in cases} == {"de", "en", "fr", "nl", "vi"}
    assert {case.oracle for case in cases} == {"intent_slot"}
    assert {case.satellite_id for case in cases} == {benchmark.CONTEXT_SATELLITE_ID}
    assert {case.category for case in cases} == {
        "complex_distortion",
        "exact_match",
        "extra_words",
        "intent_coverage",
        "missing_words",
        "semantic_challenge",
        "spelling_mistake",
        "supported_filler",
        "synonym_paraphrase",
    }
    assert benchmark._parser().parse_args([]).cases == benchmark.REAL_WORLD_DATASET_DIR


def test_rich_fixture_supports_stateful_live_intent_families() -> None:
    """Keep handler prerequisites in the fingerprinted fixture, not runner simulation."""
    entities = _manifest()["entities"]

    assert any(entity["domain"] == "todo" for entity in entities)
    assert all(
        entity["platform"] == "assist_canonicalizer_benchmark"
        for entity in entities
        if entity["domain"] == "climate"
    )
    assert any(entity.get("vacuum_area_segment") == "living_room" for entity in entities)


def test_live_trace_correlation_uses_the_final_default_agent_attempt() -> None:
    """Use the actual final nested Default Agent call, not a separate recognition run."""
    traces = [
        {
            "events": [
                {
                    "event_type": "async_process",
                    "data": {
                        "conversation_id": "request-1",
                        "agent_id": "conversation.assist_canonicalizer",
                        "text": "trun on lamp",
                    },
                }
            ]
        },
        {
            "events": [
                {
                    "event_type": "async_process",
                    "data": {
                        "conversation_id": "request-1",
                        "agent_id": "conversation.home_assistant",
                        "text": "trun on lamp",
                    },
                }
            ]
        },
        {
            "events": [
                {
                    "event_type": "async_process",
                    "data": {
                        "conversation_id": "request-1",
                        "agent_id": "conversation.home_assistant",
                        "text": "turn on living room light",
                    },
                },
                {
                    "event_type": "tool_call",
                    "data": {
                        "intent_name": "HassTurnOn",
                        "slots": {"area": "living room", "domain": "light"},
                    },
                },
            ]
        },
    ]

    observation = benchmark._conversation_trace_observation(traces, "request-1")

    assert observation.actual_intent == "HassTurnOn"
    assert observation.actual_slots == {"area": "living room", "domain": "light"}
    assert observation.delegated_text == "turn on living room light"
    assert observation.trace_count == 2
    assert len(observation.attempts) == 2


def test_live_slot_oracle_handles_numeric_and_list_equivalence() -> None:
    """Retain reviewed corpus semantics without importing the offline evaluator."""
    assert benchmark._intents_match("HassListAddItem", "HassShoppingListAddItem")
    assert benchmark._slots_match(
        {"shopping_list_item": "Milk", "percentage": 0},
        {"item": "milk", "percentage": "0.0"},
    ) == (True, [])


def test_live_slot_oracle_accepts_selectors_resolving_to_the_same_entities() -> None:
    """Judge selector wording by the entities resolved by the live intent handler."""
    matched = benchmark._semantic_slots_match(
        {"area": "Living Room", "domain": "light"},
        {"name": "Living room light"},
        ["light.living_room_rgbww_lights"],
        ["light.living_room_rgbww_lights"],
        "HassTurnOn",
    )
    wrong_target = benchmark._semantic_slots_match(
        {"area": "Living Room", "domain": "fan"},
        {"name": "Bathroom Fan"},
        ["fan.living_room_fan"],
        ["fan.percentage_limited_fan"],
        "HassTurnOff",
    )

    assert matched == (True, [], "resolved_entities", True)
    assert wrong_target[0] is False
    assert wrong_target[2:] == ("none", False)


def test_live_slot_oracle_keeps_action_parameters_after_target_resolution() -> None:
    """Do not hide a wrong value merely because the same entity was targeted."""
    temperature = benchmark._semantic_slots_match(
        {"area": "Living Room", "temperature": 18},
        {"name": "Living room thermostat", "temperature": "22"},
        ["climate.living_room_thermostat"],
        ["climate.living_room_thermostat"],
        "HassClimateSetTemperature",
    )
    vacuum_area = benchmark._semantic_slots_match(
        {"area": "Kitchen"},
        {"area": "Living Room"},
        ["vacuum.demo_vacuum_0_ground_floor"],
        ["vacuum.demo_vacuum_0_ground_floor"],
        "HassVacuumCleanArea",
    )

    assert temperature[0] is False
    assert temperature[3] is True
    assert vacuum_area[0] is False
    assert vacuum_area[3] is True


def test_live_response_oracle_separates_entities_from_area_targets() -> None:
    """Compare resolved entities without mistaking response metadata for entities."""
    observation = benchmark._response_observation(
        {
            "response": {
                "response_type": "action_done",
                "data": {
                    "success": [
                        {"type": "area", "id": "living_room"},
                        {"type": "entity", "id": "light.living_room_rgbww_lights"},
                    ]
                },
            }
        }
    )

    assert observation["target_ids"] == [
        "light.living_room_rgbww_lights",
        "living_room",
    ]
    assert observation["entity_ids"] == ["light.living_room_rgbww_lights"]


def test_managed_summary_reports_paired_hassil_effectiveness() -> None:
    """Model HassIL-first protection without hiding the direct agent result."""

    def case_result(
        case_id: str,
        *,
        canonicalizer_correct: bool,
        hassil_correct: bool,
        fallback_observed: bool = False,
    ) -> dict[str, Any]:
        return {
            "id": case_id,
            "oracle": "intent_slot",
            "expected_fallback": False,
            "passed": canonicalizer_correct,
            "measured_passes": int(canonicalizer_correct),
            "semantic_passed": canonicalizer_correct,
            "measured_requests": 1,
            "latency_samples_ms": [10.0],
            "hassil_baseline_latency_samples_ms": [4.0],
            "hassil_baseline_passed": hassil_correct,
            "hassil_baseline_last_observation": {
                "execution_success": hassil_correct,
                "fallback_observed": False,
            },
            "last_observation": {
                "execution_success": canonicalizer_correct,
                "canonical_match": canonicalizer_correct,
                "canonical_oracle_intent": "HassTurnOn",
                "canonical_oracle_unmatched_count": 0,
                "corpus_label_intent_matches_oracle": True,
                "corpus_label_slots_match_oracle": True,
                "fallback_observed": fallback_observed,
                "intent_correct": canonicalizer_correct,
                "slots_correct": canonicalizer_correct,
                "slot_match_method": "raw_slots" if canonicalizer_correct else "none",
            },
        }

    summary = benchmark._aggregate(
        [
            case_result(
                "recovered",
                canonicalizer_correct=True,
                hassil_correct=False,
            ),
            case_result(
                "protected-mismatch",
                canonicalizer_correct=False,
                hassil_correct=True,
            ),
            case_result(
                "protected-fallback",
                canonicalizer_correct=False,
                hassil_correct=True,
                fallback_observed=True,
            ),
            case_result(
                "unhandled-mismatch",
                canonicalizer_correct=False,
                hassil_correct=False,
            ),
            case_result(
                "unhandled-fallback",
                canonicalizer_correct=False,
                hassil_correct=False,
                fallback_observed=True,
            ),
            {
                "id": "language-smoke",
                "oracle": "state",
                "passed": True,
                "measured_passes": 1,
                "measured_requests": 1,
                "latency_samples_ms": [5.0],
                "last_observation": {
                    "execution_success": True,
                    "canonical_match": True,
                },
            },
        ]
    )

    assert summary["canonicalizer_accuracy_pct"] == 60.0
    assert summary["canonicalizer_correct_count"] == 3
    assert summary["fallback_count"] == 1
    assert summary["fallback_rate_pct"] == 20.0
    assert summary["mismatch_count"] == 1
    assert summary["mismatch_rate_pct"] == 20.0
    assert summary["direct_canonicalizer_accuracy_pct"] == 20.0
    assert summary["direct_canonicalizer_correct_count"] == 1
    assert summary["direct_canonicalizer_fallback_count"] == 2
    assert summary["direct_canonicalizer_fallback_rate_pct"] == 40.0
    assert summary["direct_canonicalizer_mismatch_count"] == 2
    assert summary["direct_canonicalizer_mismatch_rate_pct"] == 40.0
    assert summary["hassil_baseline_accuracy_pct"] == 40.0
    assert summary["accuracy_uplift_pp"] == 20.0
    assert summary["recovered_case_count"] == 1
    assert summary["shortcut_protected_case_count"] == 2
    assert summary["shortcut_protected_rate_pct"] == 40.0
    assert summary["both_correct_count"] == 0
    assert summary["both_incorrect_count"] == 2
    assert summary["canonical_match_count"] == 1
    assert summary["canonical_match_pct"] == 20.0
    assert (
        summary["canonicalizer_accuracy_pct"]
        + summary["mismatch_rate_pct"]
        + summary["fallback_rate_pct"]
        == 100.0
    )


def test_baseline_context_requires_explicit_home_assistant_upgrade() -> None:
    """Permit runtime drift only for the explicit upgrade-comparison mode."""
    report: dict[str, Any] = {
        "benchmark_mode": "managed_live",
        "suite_id": "suite",
        "case_suite_sha256": "cases",
        "configuration_sha256": "config",
        "settings": {"iterations": 3},
        "environment": {
            "homeassistant_version": "1.0",
            "python_version": "3.14",
            "dependencies": {"hassil": "1.0"},
            "fixture": {
                "fixture_id": "fixture",
                "schema_version": 1,
                "fingerprint": "fingerprint",
                "counts": {"exposed_entities": 60},
                "domain_counts": EXPECTED_DOMAIN_COUNTS,
                "runtime_state_count": 100,
            },
        },
    }
    baseline = deepcopy(report)
    baseline["environment"]["homeassistant_version"] = "0.9"
    baseline["environment"]["fixture"]["runtime_state_count"] = 90

    with pytest.raises(benchmark.BenchmarkError, match="baseline context differs"):
        benchmark._verify_baseline_context(report, baseline, False)

    benchmark._verify_baseline_context(report, baseline, True)


def test_baseline_detects_a_regressed_case_even_when_pass_count_is_unchanged(
    tmp_path: Path,
) -> None:
    """Prevent one improved case from hiding a different functional regression."""
    report: dict[str, Any] = {
        "report_schema_version": benchmark.BENCHMARK_SCHEMA_VERSION,
        "benchmark_mode": "managed_live",
        "suite_id": "suite",
        "case_suite_sha256": "cases",
        "configuration_sha256": "config",
        "settings": {"iterations": 1},
        "environment": {
            "homeassistant_version": "1.0",
            "python_version": "3.14",
            "dependencies": {"hassil": "1.0"},
            "fixture": {
                "fixture_id": "fixture",
                "schema_version": 1,
                "fingerprint": "fingerprint",
                "counts": {"exposed_entities": 60},
                "domain_counts": EXPECTED_DOMAIN_COUNTS,
            },
        },
        "summary": {
            "passed_cases": 1,
            "latency_ms": {"p95": 10.0},
        },
        "cases": [
            {"id": "case-a", "passed": False},
            {"id": "case-b", "passed": True},
        ],
    }
    baseline = deepcopy(report)
    baseline["cases"] = [
        {"id": "case-a", "passed": True},
        {"id": "case-b", "passed": False},
    ]
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_bytes(orjson.dumps(baseline))

    regressions = benchmark._baseline_regressions(
        report,
        baseline_path,
        max_p95_regression_pct=10.0,
        allow_homeassistant_upgrade=False,
    )

    assert any("case-a" in regression for regression in regressions)


def test_get_metric_pct_rejects_bool() -> None:
    """Verify _get_metric_pct rejects boolean values and accepts valid numbers."""
    from tools.update_readme_benchmark import _get_metric_pct

    stats = {
        "valid_int": 90,
        "valid_float": 95.5,
        "bool_true": True,
        "bool_false": False,
        "invalid_str": "90",
    }

    assert _get_metric_pct(stats, "valid_int") == 90.0
    assert _get_metric_pct(stats, "valid_float") == 95.5

    with pytest.raises(TypeError, match="must be numeric"):
        _get_metric_pct(stats, "bool_true")

    with pytest.raises(TypeError, match="must be numeric"):
        _get_metric_pct(stats, "bool_false")

    with pytest.raises(TypeError, match="must be numeric"):
        _get_metric_pct(stats, "invalid_str")


@pytest.mark.parametrize(
    "invalid_count",
    [True, False, 1.5, -1, "-1", "1.5", "abc", None],
)
def test_baseline_regressions_invalid_passed_cases(tmp_path: Path, invalid_count: Any) -> None:
    """Verify _baseline_regressions rejects invalid passed_cases counts."""
    summary: dict[str, Any] = {
        "passed_cases": 1,
        "latency_ms": {"p95": 10.0},
    }
    report: dict[str, Any] = {
        "report_schema_version": benchmark.BENCHMARK_SCHEMA_VERSION,
        "environment": {
            "homeassistant_version": "2026.8.0b4",
            "python_version": "3.14",
            "dependencies": {"hassil": "1.0"},
            "fixture": {
                "fixture_id": "fixture",
                "schema_version": 1,
                "fingerprint": "fingerprint",
                "counts": {"exposed_entities": 60},
                "domain_counts": EXPECTED_DOMAIN_COUNTS,
            },
        },
        "summary": summary,
        "cases": [{"id": "case-a", "passed": True}],
    }

    # Test invalid baseline passed_cases
    baseline_invalid = deepcopy(report)
    baseline_invalid["summary"]["passed_cases"] = invalid_count
    baseline_path = tmp_path / "baseline_invalid.json"
    baseline_path.write_bytes(orjson.dumps(baseline_invalid))

    with pytest.raises(benchmark.BenchmarkError, match="passed-case count"):
        benchmark._baseline_regressions(
            report,
            baseline_path,
            max_p95_regression_pct=10.0,
            allow_homeassistant_upgrade=False,
        )

    # Test invalid current report passed_cases
    report_invalid = deepcopy(report)
    report_invalid["summary"]["passed_cases"] = invalid_count
    valid_baseline_path = tmp_path / "baseline_valid.json"
    valid_baseline_path.write_bytes(orjson.dumps(report))

    with pytest.raises(benchmark.BenchmarkError, match="passed-case count"):
        benchmark._baseline_regressions(
            report_invalid,
            valid_baseline_path,
            max_p95_regression_pct=10.0,
            allow_homeassistant_upgrade=False,
        )
