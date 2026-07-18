"""Security and architecture boundaries for the authoritative live benchmark."""

from __future__ import annotations

import ast
from pathlib import Path

from tools import benchmark


def test_live_benchmark_has_no_production_reimplementation_imports() -> None:
    """Keep production ranking and in-process recognition out of the live runner."""
    module = ast.parse(Path("tools/benchmark.py").read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(module) if isinstance(node, ast.ImportFrom)}
    forbidden_suffixes = {
        "candidate",
        "grammar_loader",
        "ranking",
        "rehydration",
        "runtime",
        "hassil.recognize",
    }

    assert not {
        imported
        for imported in imported_modules
        if any(imported.endswith(suffix) for suffix in forbidden_suffixes)
    }


def test_live_benchmark_cli_exposes_only_managed_execution() -> None:
    """Reject external-instance, credential, and offline execution switches."""
    parser = benchmark._parser()
    help_text = parser.format_help()

    assert not {
        "--attach",
        "--offline",
        "--skip-hassil",
        "--target",
        "--token",
        "--url",
    }.intersection(help_text.split())
    assert benchmark.BENCHMARK_SCHEMA_VERSION >= 2


def test_live_benchmark_redacts_ephemeral_credentials_from_failure_text() -> None:
    """Prevent generated onboarding and bearer secrets from reaching artifacts."""
    secret = "do-not-leak-token"
    sanitized = benchmark._sanitize_text(
        f'{{"password":"{secret}","access_token":"{secret}"}} Authorization: Bearer {secret}'
    )

    assert secret not in sanitized
    assert sanitized.count("<redacted>") == 3


def test_language_smoke_accepts_deterministic_safe_fallback() -> None:
    """Treat fallback as observed compatibility behavior when live execution succeeds."""
    observation = {
        "response_type": "action_done",
        "intent": "HassTurnOn",
        "entity_ids": ["light.living_room_rgbww_lights"],
        "target_state": "on",
        "fallback_reason": "low_confidence",
        "recognition_kind": None,
    }

    assert benchmark._language_smoke_observation_succeeded(
        observation, "light.living_room_rgbww_lights"
    )
