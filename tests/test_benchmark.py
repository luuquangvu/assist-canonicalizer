"""Security and architecture boundaries for the authoritative live benchmark."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

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
    assert benchmark.BENCHMARK_SCHEMA_VERSION >= 4


def test_live_benchmark_rejects_outputs_resolving_to_the_same_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compare output destinations after repository-safe path resolution."""
    parser = benchmark._parser()
    args = parser.parse_args(
        [
            "--output-json",
            "scratch/benchmark/nested/../report",
            "--output-markdown",
            "scratch/benchmark/report",
        ]
    )

    benchmark._validate_cli_arguments(parser, args)
    with pytest.raises(SystemExit) as exc_info:
        benchmark._safe_cli_arguments(parser, args)

    assert exc_info.value.code == 2
    assert "--output-json and --output-markdown must be different paths" in capsys.readouterr().err


def test_safe_cli_arguments_preserves_uncustomized_namespace_values() -> None:
    """Carry future parser arguments forward without another manual field list."""
    parser = benchmark._parser()
    args = parser.parse_args([])
    args.future_option = "preserved"

    safe_args = benchmark._safe_cli_arguments(parser, args)

    assert safe_args.future_option == "preserved"
    assert safe_args.cases == benchmark.REAL_WORLD_DATASET_DIR.resolve()


def test_live_benchmark_list_cases_reports_load_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Convert invalid case-suite input into a normal parser error."""
    invalid_cases = tmp_path / "invalid.json"
    invalid_cases.write_text("{", encoding="utf-8")
    parser = benchmark._parser()
    args = parser.parse_args(["--cases", str(invalid_cases), "--list-cases"])

    with pytest.raises(SystemExit) as exc_info:
        benchmark._list_selected_cases(parser, args)

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "Unable to load benchmark case suite" in error
    assert str(invalid_cases) in error


def test_live_benchmark_global_threshold_rejects_missing_metric() -> None:
    """Do not treat an absent global maximum-rate metric as zero."""
    args = benchmark._parser().parse_args(["--max-fallback-rate", "10"])

    with pytest.raises(
        benchmark.BenchmarkError,
        match="summary metric 'fallback_rate_pct' is missing",
    ):
        benchmark._global_threshold_failures(args, {})

    assert benchmark._global_threshold_failures(args, {"fallback_rate_pct": 5.0}) == []


def test_live_benchmark_global_accuracy_threshold_uses_hassil_first_metric() -> None:
    """Count shortcut-protected cases in the global production accuracy gate."""
    args = benchmark._parser().parse_args(["--min-intent-slot-accuracy", "90"])

    assert (
        benchmark._global_threshold_failures(
            args,
            {
                "canonicalizer_accuracy_pct": 90.0,
                "intent_slot_accuracy_pct": 0.0,
            },
        )
        == []
    )
    assert benchmark._global_threshold_failures(
        args,
        {
            "canonicalizer_accuracy_pct": 89.0,
            "intent_slot_accuracy_pct": 100.0,
        },
    ) == ["production accuracy 89.00% is below 90.00%"]


def test_live_benchmark_language_threshold_rejects_missing_metric() -> None:
    """Do not treat an absent language maximum-rate metric as zero."""
    args = benchmark._parser().parse_args(["--max-language-fallback-rate", "10"])

    with pytest.raises(
        benchmark.BenchmarkError,
        match="summary metric 'fallback_rate_pct' is missing for EN",
    ):
        benchmark._language_threshold_failures(args, {"en": {}})


def test_live_benchmark_language_accuracy_threshold_uses_hassil_first_metric() -> None:
    """Count shortcut-protected cases in each language production accuracy gate."""
    args = benchmark._parser().parse_args(["--min-language-intent-slot-accuracy", "90"])

    assert (
        benchmark._language_threshold_failures(
            args,
            {
                "en": {
                    "canonicalizer_accuracy_pct": 90.0,
                    "intent_slot_accuracy_pct": 0.0,
                }
            },
        )
        == []
    )
    assert benchmark._language_threshold_failures(
        args,
        {
            "en": {
                "canonicalizer_accuracy_pct": 89.0,
                "intent_slot_accuracy_pct": 100.0,
            }
        },
    ) == ["EN: production accuracy 89.00% is below 90.00%"]


def test_live_benchmark_redacts_ephemeral_credentials_from_failure_text() -> None:
    """Prevent generated onboarding and bearer secrets from reaching artifacts."""
    secret = "do-not-leak-token"
    sanitized = benchmark._sanitize_text(
        f'{{"password":"{secret}","access_token":"{secret}"}} Authorization: Bearer {secret}'
    )

    assert secret not in sanitized
    assert sanitized.count("<redacted>") == 3


@pytest.mark.asyncio
async def test_live_benchmark_configures_managed_http_endpoint() -> None:
    """Stage the loopback endpoint when Home Assistant is using its default config."""
    websocket = SimpleNamespace(
        receive_json=AsyncMock(
            side_effect=[
                {"type": "auth_required"},
                {"type": "auth_ok"},
                {
                    "id": 1,
                    "type": "result",
                    "success": True,
                    "result": {
                        "active_config_type": "default",
                        "default": {"server_port": benchmark.PORT},
                    },
                },
                {
                    "id": 2,
                    "type": "result",
                    "success": True,
                    "result": {"restart": True},
                },
            ]
        ),
        send_json=AsyncMock(),
        close=AsyncMock(),
    )
    session = MagicMock(spec=benchmark.aiohttp.ClientSession)
    session.ws_connect = AsyncMock(return_value=websocket)

    restart = await benchmark._configure_managed_http_config(session, "secret-token")

    assert restart is True
    assert [call.args[0] for call in websocket.send_json.await_args_list] == [
        {"type": "auth", "access_token": "secret-token"},
        {"id": 1, "type": "http/config"},
        {
            "id": 2,
            "type": "http/config/configure",
            "config": {
                "server_host": [benchmark.HOST],
                "server_port": benchmark.PORT,
            },
        },
    ]
    websocket.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_benchmark_promotes_managed_pending_http_config() -> None:
    """Confirm the exact managed pending HTTP config before its auto-revert."""
    websocket = SimpleNamespace(
        receive_json=AsyncMock(
            side_effect=[
                {"type": "auth_required"},
                {"type": "auth_ok"},
                {
                    "id": 1,
                    "type": "result",
                    "success": True,
                    "result": {
                        "active_config_type": "pending",
                        "pending": {
                            "server_host": [benchmark.HOST],
                            "server_port": benchmark.PORT,
                        },
                    },
                },
                {"id": 2, "type": "result", "success": True, "result": None},
            ]
        ),
        send_json=AsyncMock(),
        close=AsyncMock(),
    )
    session = MagicMock(spec=benchmark.aiohttp.ClientSession)
    session.ws_connect = AsyncMock(return_value=websocket)

    promoted = await benchmark._confirm_managed_http_config(session, "secret-token")

    assert promoted is True
    assert [call.args[0] for call in websocket.send_json.await_args_list] == [
        {"type": "auth", "access_token": "secret-token"},
        {"id": 1, "type": "http/config"},
        {"id": 2, "type": "http/config/promote"},
    ]
    websocket.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_benchmark_leaves_stable_http_config_unchanged() -> None:
    """Avoid promotion when Home Assistant is already using stable HTTP config."""
    websocket = SimpleNamespace(
        receive_json=AsyncMock(
            side_effect=[
                {"type": "auth_required"},
                {"type": "auth_ok"},
                {
                    "id": 1,
                    "type": "result",
                    "success": True,
                    "result": {
                        "active_config_type": "stable",
                        "stable": {
                            "server_host": [benchmark.HOST],
                            "server_port": benchmark.PORT,
                        },
                        "pending": None,
                    },
                },
            ]
        ),
        send_json=AsyncMock(),
        close=AsyncMock(),
    )
    session = MagicMock(spec=benchmark.aiohttp.ClientSession)
    session.ws_connect = AsyncMock(return_value=websocket)

    promoted = await benchmark._confirm_managed_http_config(session, "secret-token")

    assert promoted is False
    assert [call.args[0] for call in websocket.send_json.await_args_list] == [
        {"type": "auth", "access_token": "secret-token"},
        {"id": 1, "type": "http/config"},
    ]
    websocket.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_benchmark_rejects_unexpected_pending_http_config() -> None:
    """Never promote a pending HTTP config outside the managed endpoint contract."""
    websocket = SimpleNamespace(
        receive_json=AsyncMock(
            side_effect=[
                {"type": "auth_required"},
                {"type": "auth_ok"},
                {
                    "id": 1,
                    "type": "result",
                    "success": True,
                    "result": {
                        "active_config_type": "pending",
                        "pending": {
                            "server_host": ["0.0.0.0"],
                            "server_port": benchmark.PORT,
                        },
                    },
                },
            ]
        ),
        send_json=AsyncMock(),
        close=AsyncMock(),
    )
    session = MagicMock(spec=benchmark.aiohttp.ClientSession)
    session.ws_connect = AsyncMock(return_value=websocket)

    with pytest.raises(
        benchmark.BenchmarkError,
        match="pending HTTP config differs from the managed loopback endpoint",
    ):
        await benchmark._confirm_managed_http_config(session, "secret-token")

    assert {"id": 2, "type": "http/config/promote"} not in [
        call.args[0] for call in websocket.send_json.await_args_list
    ]
    websocket.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_benchmark_policy_failure_omits_process_log_tail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep unrelated Home Assistant warnings out of actionable policy failures."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    managed = object()
    live = SimpleNamespace(suite_failures=[])
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    async def successful_async_step(*_args: object, **_kwargs: object) -> None:
        """Stand in for a successful asynchronous lifecycle operation."""
        return

    async def start_home_assistant(_config_dir: Path) -> object:
        """Return the opaque managed-process test double."""
        return managed

    async def run_live_session(*_args: object, **_kwargs: object) -> SimpleNamespace:
        """Return the completed managed-session test double."""
        return live

    def raise_policy_failure(*_args: object, **_kwargs: object) -> None:
        """Model an actionable report-policy failure."""
        raise benchmark.BenchmarkError("Benchmark threshold failures:\n- expected failure")

    def unexpected_log_tail(_managed: object) -> str:
        """Fail if policy enforcement attempts to read infrastructure logs."""
        pytest.fail("Policy failures must not request the Home Assistant process log")

    monkeypatch.setattr(benchmark, "_benchmark_inputs", lambda _args: object())
    monkeypatch.setattr(benchmark, "_assert_port_available", lambda: None)
    monkeypatch.setattr(benchmark, "_create_config_dir", lambda: config_dir)
    monkeypatch.setattr(benchmark, "_run_config_check", successful_async_step)
    monkeypatch.setattr(benchmark, "_start_home_assistant", start_home_assistant)
    monkeypatch.setattr(benchmark.aiohttp, "ClientSession", lambda **_kwargs: session)
    monkeypatch.setattr(benchmark, "_run_live_benchmark_session", run_live_session)
    monkeypatch.setattr(
        benchmark,
        "_build_benchmark_report",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(benchmark, "_stop_home_assistant", successful_async_step)
    monkeypatch.setattr(benchmark, "_finalize_benchmark_report", raise_policy_failure)
    monkeypatch.setattr(benchmark, "_log_tail", unexpected_log_tail)

    with pytest.raises(
        benchmark.BenchmarkError,
        match="Benchmark threshold failures",
    ) as exc_info:
        await benchmark.run_benchmark(benchmark._parser().parse_args([]))

    assert "Home Assistant process log tail" not in str(exc_info.value)


def test_language_smoke_accepts_deterministic_safe_fallback() -> None:
    """Treat fallback as observed compatibility behavior when live execution succeeds."""
    observation: dict[str, object] = {
        "response_type": "action_done",
        "error_code": None,
        "intent": "HassTurnOn",
        "slots": {"name": "Living Room Lamp"},
        "entity_ids": ["light.living_room_rgbww_lights"],
        "target_state": "on",
        "domain_states": {
            "light.kitchen_main_light": "off",
            "light.living_room_rgbww_lights": "on",
        },
        "fallback_reason": "low_confidence",
        "recognition_kind": None,
    }

    assert benchmark._language_smoke_observation_succeeded(
        observation,
        "light.living_room_rgbww_lights",
        {"name": "Living Room Lamp"},
    )


@pytest.mark.parametrize(
    ("drift", "value"),
    [
        ("entity_ids", ["light.kitchen_main_light", "light.living_room_rgbww_lights"]),
        ("slots", {"area": "Kitchen", "name": "Living Room Lamp"}),
        (
            "domain_states",
            {
                "light.kitchen_main_light": "on",
                "light.living_room_rgbww_lights": "on",
            },
        ),
    ],
)
def test_language_smoke_rejects_target_semantic_drift(drift: str, value: object) -> None:
    """Reject broadened resolution, selector drift, and bystander side effects."""
    observation: dict[str, object] = {
        "response_type": "action_done",
        "error_code": None,
        "intent": "HassTurnOn",
        "slots": {"name": "Living Room Lamp"},
        "entity_ids": ["light.living_room_rgbww_lights"],
        "target_state": "on",
        "domain_states": {
            "light.kitchen_main_light": "off",
            "light.living_room_rgbww_lights": "on",
        },
        drift: value,
    }
    assert not benchmark._language_smoke_observation_succeeded(
        observation,
        "light.living_room_rgbww_lights",
        {"name": "Living Room Lamp"},
    )


@pytest.mark.asyncio
async def test_domain_entity_states_observes_only_the_managed_target_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build a deterministic state map used to detect bystander changes."""
    request_json = AsyncMock(
        return_value=[
            {"entity_id": "switch.coffee_machine", "state": "off"},
            {"entity_id": "light.living_room_rgbww_lights", "state": "on"},
            {"entity_id": "light.kitchen_main_light", "state": "off"},
        ]
    )
    monkeypatch.setattr(benchmark, "_request_json", request_json)
    session = MagicMock()

    assert await benchmark._domain_entity_states(session, "light") == {
        "light.kitchen_main_light": "off",
        "light.living_room_rgbww_lights": "on",
    }
    request_json.assert_awaited_once_with(session, "GET", "/api/states")
