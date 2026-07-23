"""Run the authoritative benchmark in a fresh, managed Home Assistant process."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import math
import os
import re
import secrets
import shutil
import socket
import statistics
import sys
import tempfile
import time
import tomllib
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from string import ascii_letters, digits
from typing import Any, Literal, cast

import aiohttp
import orjson
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    from .benchmark_language_smoke import (
        ACCURACY_GATED_LANGUAGES,
        LanguageSmokeCommand,
        build_language_smoke_commands,
    )
    from .ha_dev.custom_components.assist_canonicalizer_benchmark import (
        fixture_fingerprint,
    )
except ImportError:
    from benchmark_language_smoke import (
        ACCURACY_GATED_LANGUAGES,
        LanguageSmokeCommand,
        build_language_smoke_commands,
    )
    from ha_dev.custom_components.assist_canonicalizer_benchmark import (
        fixture_fingerprint,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
HA_DEV_DIR = REPO_ROOT / "tools" / "ha_dev"
CONFIGURATION_PATH = HA_DEV_DIR / "configuration.yaml"
FIXTURE_PATH = HA_DEV_DIR / "custom_components" / "assist_canonicalizer_benchmark" / "fixture.json"
REAL_WORLD_DATASET_DIR = REPO_ROOT / "tests" / "real_world"
INTEGRATION_PATH = REPO_ROOT / "custom_components" / "assist_canonicalizer"
FIXTURE_COMPONENT_PATH = HA_DEV_DIR / "custom_components" / "assist_canonicalizer_benchmark"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
SCRATCH_DIR = REPO_ROOT / "scratch"
DEFAULT_OUTPUT_JSON = SCRATCH_DIR / "benchmark" / "managed_live_report.json"
DEFAULT_OUTPUT_MARKDOWN = SCRATCH_DIR / "benchmark" / "managed_live_report.md"

BENCHMARK_SCHEMA_VERSION = 2
BENCHMARK_GROUP = "ha-benchmark"
REAL_WORLD_SUITE_ID = "managed_live_real_world_v1"
HOST = "127.0.0.1"
PORT = 8123
BASE_URL = f"http://{HOST}:{PORT}"
FIXTURE_ENTITY_ID = "sensor.assist_canonicalizer_benchmark_fixture"
AGENT_ENTITY_ID = "conversation.assist_canonicalizer"
HOME_ASSISTANT_AGENT_ID = "conversation.home_assistant"
CONTEXT_SATELLITE_ID = "light.living_room_rgbww_lights"
BENCHMARK_DEVICE_ID = "assist-canonicalizer-benchmark-device"
HTTP_TIMEOUT_SECONDS = 30.0
PROCESS_TIMEOUT_SECONDS = 180.0
DEPENDENCIES_WITHOUT_HA_MANIFEST_REQUIREMENTS = frozenset({"colorlog"})
SECRET_PATTERNS = (
    (
        re.compile(r'("(?:access_token|refresh_token|password)"\s*:\s*")[^"]+("?)'),
        r"\1<redacted>\2",
    ),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._~-]+", re.IGNORECASE), r"\1<redacted>"),
)
_NUMERIC_SLOT_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_SLOT_KEY_EQUIVALENTS: Mapping[str, tuple[str, ...]] = {
    "item": ("item", "shopping_list_item", "todo_list_item"),
    "shopping_list_item": ("shopping_list_item", "item"),
    "todo_list_item": ("todo_list_item", "item"),
}
_TARGET_SELECTOR_SLOTS = frozenset(
    {
        "area",
        "device_class",
        "domain",
        "floor",
        "name",
        "preferred_area_id",
        "preferred_floor_id",
    }
)
_AREA_ACTION_INTENTS = frozenset({"HassVacuumCleanArea"})

_PATH_ALLOWED_CHARS = ascii_letters + digits + "/._-"


class BenchmarkError(RuntimeError):
    """Raised when a managed benchmark precondition or operation fails."""


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """One deterministic Home Assistant service call used to prepare a case."""

    domain: str
    service: str
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ExpectedState:
    """Expected state after one production conversation request."""

    entity_id: str
    state: str


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One tracked production-path benchmark request."""

    case_id: str
    language: str
    query: str
    oracle: Literal["outcome", "intent_slot"]
    category: str
    expected_intent: str | None
    expected_canonical: str | None
    expected_slots: Mapping[str, str]
    expected_fallback: bool
    satellite_id: str | None
    expected_response_type: str | None
    expected_target_id: str | None
    expected_state: ExpectedState | None
    setup: ServiceSpec | None


@dataclass(slots=True)
class ManagedProcess:
    """Running Home Assistant process and its temporary log."""

    process: asyncio.subprocess.Process
    log_handle: Any
    log_path: Path


@dataclass(frozen=True, slots=True)
class ConversationTraceObservation:
    """Request-correlated facts emitted by Home Assistant's production trace."""

    actual_intent: str | None
    actual_slots: Mapping[str, Any]
    delegated_text: str
    attempts: tuple[Mapping[str, Any], ...]
    trace_count: int


def _required_string(mapping: Mapping[str, Any], key: str, context: str) -> str:
    """Return a required non-empty string from a JSON object."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{context}.{key} must be a non-empty string")
    return value


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    """Load a UTF-8 JSON file and require an object root."""
    try:
        loaded = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as err:
        raise BenchmarkError(f"Unable to load {description} at {path}: {err}") from err
    if not isinstance(loaded, dict):
        raise BenchmarkError(f"{description} root must be an object: {path}")
    return loaded


def load_cases(path: Path) -> tuple[str, tuple[BenchmarkCase, ...]]:
    """Load a tracked live-oracle file or the rich real-world corpus."""
    if path.is_dir():
        return _load_real_world_cases(path)
    return _load_outcome_cases(path)


def _load_outcome_cases(path: Path) -> tuple[str, tuple[BenchmarkCase, ...]]:
    """Load and validate the explicit production-outcome smoke suite."""
    payload = _load_json_object(path, "benchmark case suite")
    if payload.get("schema_version") != 1:
        raise BenchmarkError("Unsupported benchmark case schema")
    suite_id = _required_string(payload, "suite_id", "suite")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise BenchmarkError("suite.cases must be a non-empty list")

    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        context = f"suite.cases[{index}]"
        if not isinstance(raw_case, dict):
            raise BenchmarkError(f"{context} must be an object")
        raw_case = cast(dict[str, Any], raw_case)
        case_id = _required_string(raw_case, "id", context)
        if case_id in seen_ids:
            raise BenchmarkError(f"Duplicate benchmark case ID: {case_id}")
        seen_ids.add(case_id)
        expected_target_id = raw_case.get("expected_target_id")
        if expected_target_id is not None and (
            not isinstance(expected_target_id, str) or not expected_target_id.strip()
        ):
            raise BenchmarkError(f"{context}.expected_target_id must be a non-empty string")
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                language=_required_string(raw_case, "language", context),
                query=_required_string(raw_case, "query", context),
                oracle="outcome",
                category="managed_live_smoke",
                expected_intent=None,
                expected_canonical=None,
                expected_slots={},
                expected_fallback=False,
                satellite_id=None,
                expected_response_type=_required_string(
                    raw_case, "expected_response_type", context
                ),
                expected_target_id=expected_target_id,
                expected_state=_parse_expected_state(raw_case.get("expected_state"), context),
                setup=_parse_service_spec(raw_case.get("setup"), context),
            )
        )
    return suite_id, tuple(cases)


def _load_real_world_cases(path: Path) -> tuple[str, tuple[BenchmarkCase, ...]]:
    """Load the maintained multilingual corpus as production-traced live cases."""
    dataset_paths = sorted(path.glob("*.json"))
    if not dataset_paths:
        raise BenchmarkError(f"No real-world dataset JSON files found in {path}")
    discovered_languages = {dataset_path.stem for dataset_path in dataset_paths}
    if missing_languages := ACCURACY_GATED_LANGUAGES - discovered_languages:
        raise BenchmarkError(
            "Accuracy-gated datasets are missing: " + ", ".join(sorted(missing_languages))
        )

    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for dataset_path in dataset_paths:
        language = dataset_path.stem
        payload = _load_json_object(dataset_path, "real-world dataset")
        raw_cases = payload.get("test_cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise BenchmarkError(f"{dataset_path}.test_cases must be a non-empty list")
        seen_queries: set[str] = set()
        for index, value in enumerate(raw_cases, start=1):
            context = f"{dataset_path}.test_cases[{index - 1}]"
            if not isinstance(value, dict):
                raise BenchmarkError(f"{context} must be an object")
            raw_case = cast(dict[str, Any], value)
            case_language = raw_case.get("language", language)
            if case_language != language:
                raise BenchmarkError(f"{context}.language must match dataset language {language!r}")
            query = _required_string(raw_case, "query", context)
            query_key = " ".join(query.casefold().split())
            if query_key in seen_queries:
                raise BenchmarkError(f"Duplicate query in {dataset_path}: {query!r}")
            seen_queries.add(query_key)

            expected_slots = _parse_expected_slots(raw_case.get("expected_slots", {}), context)
            expected_fallback = raw_case.get("expected_fallback", False)
            if not isinstance(expected_fallback, bool):
                raise BenchmarkError(f"{context}.expected_fallback must be a boolean")
            satellite_id = _context_satellite_id(raw_case.get("context", {}), context)
            case_id = f"{language}-{index:03d}"
            if case_id in seen_ids:
                raise BenchmarkError(f"Duplicate benchmark case ID: {case_id}")
            seen_ids.add(case_id)
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    language=language,
                    query=query,
                    oracle="intent_slot",
                    category=_required_string(raw_case, "category", context),
                    expected_intent=_required_string(raw_case, "expected_intent", context),
                    expected_canonical=_required_string(raw_case, "expected_canonical", context),
                    expected_slots=expected_slots,
                    expected_fallback=expected_fallback,
                    satellite_id=satellite_id,
                    expected_response_type=None,
                    expected_target_id=None,
                    expected_state=None,
                    setup=None,
                )
            )
    return REAL_WORLD_SUITE_ID, tuple(cases)


def _parse_expected_slots(value: Any, context: str) -> dict[str, str]:
    """Validate the expected subset of slots for one corpus case."""
    if not isinstance(value, dict):
        raise BenchmarkError(f"{context}.expected_slots must be an object")
    expected_slots: dict[str, str] = {}
    for key, slot_value in value.items():
        if not isinstance(key, str) or not key.strip():
            raise BenchmarkError(f"{context}.expected_slots keys must be non-empty strings")
        if not isinstance(slot_value, str) or not slot_value.strip():
            raise BenchmarkError(f"{context}.expected_slots[{key!r}] must be a non-empty string")
        expected_slots[key] = slot_value
    return expected_slots


def _context_satellite_id(value: Any, context: str) -> str | None:
    """Run corpus requests from the fixture's fixed living-room Assist satellite."""
    if not isinstance(value, dict):
        raise BenchmarkError(f"{context}.context must be an object")
    if not value:
        return CONTEXT_SATELLITE_ID
    if set(value) != {"area"} or not isinstance(value.get("area"), str):
        raise BenchmarkError(f"{context}.context is not supported by the managed fixture")
    normalized_area = _normalized_text(value["area"])
    if normalized_area not in {"living room", "wohnzimmer", "salon", "woonkamer"}:
        raise BenchmarkError(
            f"{context}.context area {value['area']!r} is not represented by the fixture"
        )
    return CONTEXT_SATELLITE_ID


def _parse_expected_state(value: Any, context: str) -> ExpectedState | None:
    """Parse an optional expected-state object."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise BenchmarkError(f"{context}.expected_state must be an object")
    return ExpectedState(
        entity_id=_required_string(value, "entity_id", f"{context}.expected_state"),
        state=_required_string(value, "state", f"{context}.expected_state"),
    )


def _parse_service_spec(value: Any, context: str) -> ServiceSpec | None:
    """Parse an optional deterministic setup service call."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise BenchmarkError(f"{context}.setup must be an object")
    data = value.get("data", {})
    if not isinstance(data, dict):
        raise BenchmarkError(f"{context}.setup.data must be an object")
    return ServiceSpec(
        domain=_required_string(value, "domain", f"{context}.setup"),
        service=_required_string(value, "service", f"{context}.setup"),
        data=data,
    )


def _benchmark_group_requirements() -> tuple[Requirement, ...]:
    """Return unpinned dependencies declared by the benchmark dependency group."""
    try:
        payload = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise BenchmarkError(f"Unable to load {PYPROJECT_PATH}: {err}") from err
    groups = payload.get("dependency-groups")
    raw_group = groups.get(BENCHMARK_GROUP) if isinstance(groups, dict) else None
    if not isinstance(raw_group, list) or not raw_group:
        raise BenchmarkError(f"Missing dependency group: {BENCHMARK_GROUP}")

    requirements: list[Requirement] = []
    names: set[str] = set()
    for raw_requirement in raw_group:
        if not isinstance(raw_requirement, str):
            raise BenchmarkError(f"{BENCHMARK_GROUP} dependencies must be strings")
        requirement = Requirement(raw_requirement)
        normalized_name = canonicalize_name(requirement.name)
        if requirement.specifier or requirement.url or requirement.marker or requirement.extras:
            raise BenchmarkError(
                f"{BENCHMARK_GROUP} dependency must not copy a version or source constraint: "
                f"{raw_requirement}"
            )
        if normalized_name in names:
            raise BenchmarkError(f"Duplicate {BENCHMARK_GROUP} dependency: {requirement.name}")
        names.add(normalized_name)
        requirements.append(requirement)
    return tuple(requirements)


def _home_assistant_manifest_requirements() -> dict[str, set[str]]:
    """Collect Home Assistant's own integration requirements by package name."""
    components_spec = importlib.util.find_spec("homeassistant.components")
    locations = components_spec.submodule_search_locations if components_spec else None
    if not locations:
        raise BenchmarkError("Unable to locate installed Home Assistant component manifests")
    component_root = Path(next(iter(locations)))
    requirements: dict[str, set[str]] = {}
    for manifest_path in component_root.glob("*/manifest.json"):
        try:
            manifest = orjson.loads(manifest_path.read_bytes())
        except (OSError, orjson.JSONDecodeError) as err:
            raise BenchmarkError(
                f"Unable to read Home Assistant manifest {manifest_path}: {err}"
            ) from err
        raw_requirements = manifest.get("requirements", [])
        if not isinstance(raw_requirements, list):
            raise BenchmarkError(f"Invalid requirements in Home Assistant manifest {manifest_path}")
        for raw_requirement in raw_requirements:
            if not isinstance(raw_requirement, str):
                continue
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            requirements.setdefault(canonicalize_name(requirement.name), set()).add(raw_requirement)
    return requirements


def verify_benchmark_dependencies() -> dict[str, Any]:
    """Verify runtime packages against the installed Home Assistant manifests."""
    manifest_requirements = _home_assistant_manifest_requirements()
    resolved: dict[str, dict[str, Any]] = {}
    for declared in _benchmark_group_requirements():
        normalized_name = canonicalize_name(declared.name)
        try:
            installed_version = importlib.metadata.version(declared.name)
        except importlib.metadata.PackageNotFoundError as err:
            raise BenchmarkError(
                f"Missing {BENCHMARK_GROUP} dependency {declared.name}; run uv sync --all-groups"
            ) from err

        constraints = sorted(manifest_requirements.get(normalized_name, ()))
        if not constraints and normalized_name not in DEPENDENCIES_WITHOUT_HA_MANIFEST_REQUIREMENTS:
            raise BenchmarkError(
                f"Home Assistant does not declare a requirement for {declared.name}; "
                f"review the {BENCHMARK_GROUP} dependency set"
            )
        for raw_constraint in constraints:
            constraint = Requirement(raw_constraint)
            if installed_version not in constraint.specifier:
                raise BenchmarkError(
                    f"{declared.name} {installed_version} does not match Home Assistant "
                    f"requirement {raw_constraint}; update the lockfile and run "
                    "uv sync --all-groups"
                )
        resolved[normalized_name] = {
            "version": installed_version,
            "home_assistant_requirements": constraints,
        }
    return {
        "homeassistant": importlib.metadata.version("homeassistant"),
        "packages": dict(sorted(resolved.items())),
    }


def _benchmark_environment_summary() -> str:
    """Return a formatted string of key version metadata for terminal logging."""
    py_ver = sys.version.split()[0]
    try:
        ha_ver = importlib.metadata.version("homeassistant")
    except importlib.metadata.PackageNotFoundError:
        ha_ver = "unknown"
    try:
        hassil_ver = importlib.metadata.version("hassil")
    except importlib.metadata.PackageNotFoundError:
        hassil_ver = "unknown"
    try:
        intents_ver = importlib.metadata.version("home-assistant-intents")
    except importlib.metadata.PackageNotFoundError:
        intents_ver = "unknown"

    return f"homeassistant={ha_ver} python={py_ver} hassil={hassil_ver} intents={intents_ver}"


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one tracked benchmark input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> str:
    """Return a stable digest of every regular source file below one directory."""
    digest = hashlib.sha256()
    source_paths = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix not in {".pyc", ".pyo"}
    )
    if not source_paths:
        raise BenchmarkError(f"Benchmark source tree is empty: {path}")
    for source_path in source_paths:
        if source_path.is_symlink():
            raise BenchmarkError(f"Benchmark source tree contains a symlink: {source_path}")
        digest.update(source_path.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_payload_sha256(payload: Any) -> str:
    """Return the digest of one deterministic, non-secret JSON payload."""
    return hashlib.sha256(
        orjson.dumps(
            payload,
            option=orjson.OPT_SORT_KEYS,
        )
    ).hexdigest()


def _case_input_sha256(path: Path) -> str:
    """Hash one case file or every ordered JSON file in a corpus directory."""
    if path.is_file():
        return _file_sha256(path)
    if not path.is_dir():
        raise BenchmarkError(f"Benchmark case input does not exist: {path}")
    digest = hashlib.sha256()
    dataset_paths = sorted(path.glob("*.json"))
    if not dataset_paths:
        raise BenchmarkError(f"No benchmark dataset files found in {path}")
    for dataset_path in dataset_paths:
        digest.update(dataset_path.name.encode())
        digest.update(b"\0")
        digest.update(dataset_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _case_input_files(path: Path) -> list[dict[str, Any]]:
    """Return stable metadata for every tracked case input file."""
    paths = [path] if path.is_file() else sorted(path.glob("*.json"))
    return [
        {
            "path": str(item.relative_to(REPO_ROOT)),
            "sha256": _file_sha256(item),
        }
        for item in paths
    ]


def sanitize_chars(value: str, allowed: str) -> str:
    """Validate and sanitize a string using allowed characters to break taint."""
    if not isinstance(value, str):
        raise ValueError("Expected a string.")
    safe_chars: list[str] = []
    for char in value:
        idx = allowed.find(char)
        if idx == -1:
            raise ValueError(f"character {char!r} is not allowed.")
        safe_chars.append(allowed[idx])
    return "".join(safe_chars)


def _safe_repository_path(raw_path: str | Path, description: str) -> Path:
    """Resolve a command-line path and constrain it to this repository."""
    raw_str = str(raw_path)
    try:
        clean = sanitize_chars(raw_str, _PATH_ALLOWED_CHARS)
    except ValueError as err:
        raise BenchmarkError(f"Invalid path {raw_str!r}; {err}") from err

    path = Path(clean)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise BenchmarkError(f"{description} must remain inside {REPO_ROOT}: {raw_str}")
    return resolved


def _assert_port_available() -> None:
    """Fail before setup unless the fixed loopback endpoint can be owned."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            candidate.bind((HOST, PORT))
        except OSError as err:
            raise BenchmarkError(
                f"Managed benchmark endpoint {HOST}:{PORT} is unavailable"
            ) from err


def _create_config_dir() -> Path:
    """Create one empty Home Assistant configuration linked to tracked inputs."""
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="ha-benchmark-", dir=SCRATCH_DIR))
    try:
        _create_symlinks(run_dir)
    except Exception:
        shutil.rmtree(run_dir)
        raise
    return run_dir


def _create_symlinks(run_dir):
    """Create a temporary Home Assistant configuration with tracked inputs."""
    os.chmod(run_dir, 0o700)
    (run_dir / "configuration.yaml").symlink_to(CONFIGURATION_PATH)
    custom_components = run_dir / "custom_components"
    custom_components.mkdir()
    (custom_components / "assist_canonicalizer").symlink_to(
        INTEGRATION_PATH, target_is_directory=True
    )
    (custom_components / "assist_canonicalizer_benchmark").symlink_to(
        FIXTURE_COMPONENT_PATH,
        target_is_directory=True,
    )


async def _run_config_check(config_dir: Path) -> None:
    """Validate the generated Home Assistant configuration before startup."""
    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "--frozen",
        "--group",
        BENCHMARK_GROUP,
        "hass",
        "--script",
        "check_config",
        "-c",
        str(config_dir),
        "--json",
        cwd=REPO_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=PROCESS_TIMEOUT_SECONDS
        )
    except TimeoutError as err:
        process.kill()
        await process.wait()
        raise BenchmarkError("Home Assistant configuration check timed out") from err
    if process.returncode != 0:
        details = _sanitize_text((stdout + stderr).decode(errors="replace"))[-4000:]
        raise BenchmarkError(f"Home Assistant configuration check failed:\n{details}")


async def _start_home_assistant(config_dir: Path) -> ManagedProcess:
    """Start Home Assistant with dependency installation disabled."""
    log_path = config_dir / "home-assistant-process.log"
    log_handle = log_path.open("wb")
    try:
        process = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "--frozen",
            "--group",
            BENCHMARK_GROUP,
            "hass",
            "-c",
            str(config_dir),
            "--skip-pip",
            "--log-no-color",
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception:
        log_handle.close()
        raise
    return ManagedProcess(process=process, log_handle=log_handle, log_path=log_path)


async def _stop_home_assistant(managed: ManagedProcess | None) -> None:
    """Stop the managed Home Assistant process and close its log."""
    if managed is None:
        return
    if managed.process.returncode is None:
        managed.process.terminate()
        try:
            await asyncio.wait_for(managed.process.wait(), timeout=30.0)
        except TimeoutError:
            managed.process.kill()
            await managed.process.wait()
    managed.log_handle.close()


def _sanitize_text(text: str) -> str:
    """Redact authentication material from diagnostic output."""
    sanitized = text
    for pattern, replacement in SECRET_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def _log_tail(managed: ManagedProcess | None) -> str:
    """Return a bounded sanitized process-log tail for a failed run."""
    if managed is None:
        return ""
    managed.log_handle.flush()
    try:
        text = managed.log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return _sanitize_text(text)[-8000:]


async def _request_json(
    session: aiohttp.ClientSession,
    method: str,
    path: str,
    **kwargs: Any,
) -> Any:
    """Make one Home Assistant HTTP request and require a JSON success response."""
    try:
        async with session.request(method, f"{BASE_URL}{path}", **kwargs) as response:
            text = await response.text()
            if response.status < 200 or response.status >= 300:
                raise BenchmarkError(
                    f"Home Assistant {method} {path} returned {response.status}: "
                    f"{_sanitize_text(text)[:1000]}"
                )
            if not text:
                return None
            try:
                return orjson.loads(text)
            except orjson.JSONDecodeError as err:
                raise BenchmarkError(
                    f"Home Assistant {method} {path} returned invalid JSON"
                ) from err
    except (aiohttp.ClientError, TimeoutError) as err:
        raise BenchmarkError(f"Home Assistant request failed for {method} {path}: {err}") from err


async def _wait_for_http(
    session: aiohttp.ClientSession,
    managed: ManagedProcess,
) -> None:
    """Wait for the onboarding endpoint while watching for early process exit."""
    deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if managed.process.returncode is not None:
            raise BenchmarkError(
                f"Home Assistant exited before HTTP startup with code {managed.process.returncode}"
            )
        with contextlib.suppress(aiohttp.ClientError):
            async with session.get(f"{BASE_URL}/api/onboarding") as response:
                if response.status == 200:
                    return
        await asyncio.sleep(0.25)
    raise BenchmarkError("Timed out waiting for Home Assistant HTTP startup")


async def _onboard(session: aiohttp.ClientSession) -> str:
    """Create one ephemeral owner and return its bearer token."""
    client_id = f"{BASE_URL}/"
    auth_code_payload = await _request_json(
        session,
        "POST",
        "/api/onboarding/users",
        json={
            "name": "Benchmark Owner",
            "username": f"benchmark-{secrets.token_hex(8)}",
            "password": secrets.token_urlsafe(32),
            "client_id": client_id,
            "language": "en",
        },
    )
    if not isinstance(auth_code_payload, dict):
        raise BenchmarkError("Onboarding user response must be an object")
    auth_code = _required_string(auth_code_payload, "auth_code", "onboarding response")
    token_payload = await _request_json(
        session,
        "POST",
        "/auth/token",
        data={
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": client_id,
        },
    )
    if not isinstance(token_payload, dict):
        raise BenchmarkError("Authentication token response must be an object")
    return _required_string(token_payload, "access_token", "token response")


async def _call_service(
    session: aiohttp.ClientSession,
    domain: str,
    service: str,
    data: Mapping[str, Any] | None = None,
    *,
    return_response: bool = False,
) -> Any:
    """Call one Home Assistant service through its production REST API."""
    suffix = "?return_response" if return_response else ""
    return await _request_json(
        session,
        "POST",
        f"/api/services/{domain}/{service}{suffix}",
        json=dict(data or {}),
    )


async def _wait_for_fixture(
    session: aiohttp.ClientSession,
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    """Wait for and verify the exact tracked benchmark fixture contract."""
    expected_fingerprint = fixture_fingerprint(fixture)
    deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS
    last_state: Mapping[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            state = await _request_json(session, "GET", f"/api/states/{FIXTURE_ENTITY_ID}")
        except BenchmarkError:
            await asyncio.sleep(0.25)
            continue
        if isinstance(state, dict):
            last_state = state
            if state.get("state") == "error":
                attributes = state.get("attributes", {})
                raise BenchmarkError(f"Benchmark fixture provisioning failed: {attributes}")
            if state.get("state") == "ready":
                attributes = state.get("attributes")
                if not isinstance(attributes, dict):
                    raise BenchmarkError("Benchmark fixture readiness attributes are missing")
                expected_counts = fixture["expected_counts"]
                checks = {
                    "fixture_id": fixture["fixture_id"],
                    "fingerprint": expected_fingerprint,
                    "floor_count": expected_counts["floors"],
                    "area_count": expected_counts["areas"],
                    "exposed_entity_count": expected_counts["exposed_entities"],
                    "domain_counts": fixture["expected_domain_counts"],
                }
                if differences := {
                    key: {"expected": expected, "actual": attributes.get(key)}
                    for key, expected in checks.items()
                    if attributes.get(key) != expected
                }:
                    raise BenchmarkError(f"Benchmark fixture contract differs: {differences}")
                return attributes
        await asyncio.sleep(0.25)
    raise BenchmarkError(f"Timed out waiting for benchmark fixture; last state={last_state}")


async def _create_config_entry(
    session: aiohttp.ClientSession,
    handler: str,
) -> str:
    """Create one integration through Home Assistant's production config flow."""
    flow = await _request_json(
        session,
        "POST",
        "/api/config/config_entries/flow",
        json={"handler": handler},
    )
    if not isinstance(flow, dict):
        raise BenchmarkError(f"{handler} config-flow initialization must be an object")
    if flow.get("type") == "create_entry":
        completed = flow
    else:
        flow_id = _required_string(flow, "flow_id", "config-flow response")
        completed = await _request_json(
            session,
            "POST",
            f"/api/config/config_entries/flow/{flow_id}",
            json={},
        )
    if not isinstance(completed, dict) or completed.get("type") != "create_entry":
        raise BenchmarkError(f"{handler} config flow did not create an entry: {completed}")
    result = completed.get("result")
    if not isinstance(result, dict):
        raise BenchmarkError("Created config-flow entry is missing its result object")
    return _required_string(result, "entry_id", "config-flow result")


async def _create_integration_entry(session: aiohttp.ClientSession) -> str:
    """Create Assist Canonicalizer through Home Assistant's production config flow."""
    return await _create_config_entry(session, "assist_canonicalizer")


async def _wait_for_agent(session: aiohttp.ClientSession) -> None:
    """Wait until the Assist Canonicalizer conversation entity is available."""
    deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            state = await _request_json(session, "GET", f"/api/states/{AGENT_ENTITY_ID}")
        except BenchmarkError:
            await asyncio.sleep(0.25)
            continue
        if isinstance(state, dict) and state.get("state") != "unavailable":
            return
        await asyncio.sleep(0.25)
    raise BenchmarkError("Timed out waiting for the Assist Canonicalizer conversation entity")


async def _prepare_languages(
    session: aiohttp.ClientSession,
    languages: Sequence[str],
) -> dict[str, Any]:
    """Build each case language index before warmup and measurement."""
    prepared: dict[str, Any] = {}
    for language in sorted(set(languages)):
        response = await _call_service(
            session,
            "assist_canonicalizer",
            "rebuild_index",
            {"language": language},
            return_response=True,
        )
        if not isinstance(response, dict):
            raise BenchmarkError(f"Index rebuild for {language} returned no response")
        prepared[language] = response.get("service_response")
    return prepared


async def _execute_language_smoke(
    session: aiohttp.ClientSession,
    commands: Sequence[LanguageSmokeCommand],
    agent_id: str,
) -> list[dict[str, Any]]:
    """Execute every installed language twice through the production conversation path."""
    results: list[dict[str, Any]] = []
    for command in commands:
        target_entity_id = command.target_entity_id
        case = BenchmarkCase(
            case_id=f"language-smoke-{command.language}",
            language=command.language,
            query=command.text,
            oracle="intent_slot",
            category="compatibility_smoke",
            expected_intent="HassTurnOn",
            expected_canonical=command.text,
            expected_slots={},
            expected_fallback=False,
            satellite_id=CONTEXT_SATELLITE_ID,
            expected_response_type=None,
            expected_target_id=target_entity_id,
            expected_state=None,
            setup=None,
        )
        observations: list[dict[str, Any]] = []
        for attempt in range(2):
            await _call_service(
                session,
                command.target_domain,
                "turn_off",
                {"entity_id": target_entity_id},
            )
            conversation_id = f"language-smoke-{command.language}-{attempt}"
            payload, _latency_ms, trace = await _run_observed_conversation(
                session,
                case,
                agent_id,
                conversation_id,
            )
            diagnostics = await _diagnostics(session)
            if diagnostics.get("last_request_id") != conversation_id:
                raise BenchmarkError(
                    f"Language smoke request correlation failed for {command.language}"
                )
            response = _response_observation(payload)
            state = await _wait_for_expected_state(
                session,
                ExpectedState(entity_id=target_entity_id, state="on"),
            )
            observations.append(
                {
                    "response_type": response["response_type"],
                    "error_code": response["error_code"],
                    "intent": trace.actual_intent,
                    "slots": dict(trace.actual_slots),
                    "entity_ids": response["entity_ids"],
                    "target_state": state.get("state") if isinstance(state, dict) else None,
                    "delegated_text_sha256": hashlib.sha256(
                        trace.delegated_text.encode()
                    ).hexdigest(),
                    "fallback_reason": diagnostics.get("last_fallback_reason"),
                    "recognition_kind": diagnostics.get("recognition_kind"),
                    "selected_delegated_text_hash": diagnostics.get("selected_delegated_text_hash"),
                    "confidence_margin_policy": (
                        diagnostics.get("confidence_gate", {}).get("margin_policy")
                        if isinstance(diagnostics.get("confidence_gate"), dict)
                        else None
                    ),
                }
            )
        if observations[0] != observations[1]:
            raise BenchmarkError(
                f"Language smoke outcome is non-deterministic for {command.language}: "
                f"{observations}"
            )
        observation = observations[0]
        if not _language_smoke_observation_succeeded(observation, target_entity_id):
            raise BenchmarkError(f"Language smoke failed for {command.language}: {observation}")
        results.append(
            {
                "language": command.language,
                "command_sha256": hashlib.sha256(command.text.encode()).hexdigest(),
                "target_domain": command.target_domain,
                "target_entity_id": target_entity_id,
                "outcome": observation,
            }
        )
        await _call_service(
            session,
            command.target_domain,
            "turn_off",
            {"entity_id": target_entity_id},
        )
        await _wait_for_expected_state(
            session,
            ExpectedState(entity_id=target_entity_id, state="off"),
        )
    return results


def _language_smoke_observation_succeeded(
    observation: Mapping[str, Any], target_entity_id: str
) -> bool:
    """Return whether a deterministic smoke observation achieved its live contract."""
    return (
        observation["response_type"] != "error"
        and observation["intent"] == "HassTurnOn"
        and target_entity_id in observation["entity_ids"]
        and observation["target_state"] == "on"
    )


async def _run_conversation(
    session: aiohttp.ClientSession,
    case: BenchmarkCase,
    agent_id: str,
    conversation_id: str,
    *,
    text: str | None = None,
) -> tuple[dict[str, Any], float]:
    """Execute and time one production conversation HTTP request."""
    request_payload = {
        "text": case.query if text is None else text,
        "language": case.language,
        "agent_id": agent_id,
        "conversation_id": conversation_id,
    }
    if case.oracle == "intent_slot":
        request_payload["device_id"] = BENCHMARK_DEVICE_ID
    if case.satellite_id is not None:
        request_payload["satellite_id"] = case.satellite_id
    started = time.perf_counter()
    payload = await _request_json(
        session,
        "POST",
        "/api/conversation/process",
        json=request_payload,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not isinstance(payload, dict):
        raise BenchmarkError(f"Conversation response for {case.case_id} must be an object")
    return payload, elapsed_ms


async def _run_observed_conversation(
    session: aiohttp.ClientSession,
    case: BenchmarkCase,
    agent_id: str,
    conversation_id: str,
    *,
    text: str | None = None,
) -> tuple[dict[str, Any], float, ConversationTraceObservation]:
    """Execute one request and collect its correlated live Default Agent trace."""
    await _clear_conversation_traces(session)
    payload, latency_ms = await _run_conversation(
        session,
        case,
        agent_id,
        conversation_id,
        text=text,
    )
    traces = await _conversation_traces(session)
    trace = _conversation_trace_observation(traces, conversation_id)
    return payload, latency_ms, trace


async def _clear_conversation_traces(session: aiohttp.ClientSession) -> None:
    """Clear the managed process's bounded passive trace buffer before a case."""
    await _call_service(
        session,
        "assist_canonicalizer_benchmark",
        "clear_conversation_traces",
    )


async def _conversation_traces(session: aiohttp.ClientSession) -> list[Mapping[str, Any]]:
    """Read the passive traces emitted by the immediately preceding request."""
    response = await _call_service(
        session,
        "assist_canonicalizer_benchmark",
        "get_conversation_traces",
        return_response=True,
    )
    if not isinstance(response, dict):
        raise BenchmarkError("Conversation trace service returned no response")
    service_response = response.get("service_response")
    traces = service_response.get("traces") if isinstance(service_response, dict) else None
    if not isinstance(traces, list) or not all(isinstance(trace, dict) for trace in traces):
        raise BenchmarkError("Conversation trace service returned an invalid trace list")
    return cast(list[Mapping[str, Any]], traces)


async def _canonical_oracle(
    session: aiohttp.ClientSession, case: BenchmarkCase
) -> Mapping[str, Any] | None:
    """Recognize the reviewed canonical control in the same live HA context."""
    if case.expected_canonical is None:
        return None
    data = {
        "text": case.expected_canonical,
        "language": case.language,
        "device_id": BENCHMARK_DEVICE_ID,
    }
    if case.satellite_id is not None:
        data["satellite_id"] = case.satellite_id
    response = await _call_service(
        session,
        "assist_canonicalizer_benchmark",
        "recognize_canonical",
        data,
        return_response=True,
    )
    if not isinstance(response, dict):
        raise BenchmarkError(f"Canonical oracle returned no response for {case.case_id}")
    service_response = response.get("service_response")
    if not isinstance(service_response, dict):
        raise BenchmarkError(f"Canonical oracle response is invalid for {case.case_id}")
    actual_intent = service_response.get("intent")
    slots = service_response.get("slots")
    unmatched_count = service_response.get("unmatched_count")
    if actual_intent is not None and not isinstance(actual_intent, str):
        raise BenchmarkError(f"Canonical oracle intent is invalid for {case.case_id}")
    if not isinstance(slots, dict) or not isinstance(unmatched_count, int):
        raise BenchmarkError(f"Canonical oracle slots are invalid for {case.case_id}")
    return {
        "intent": actual_intent,
        "slots": slots,
        "unmatched_count": unmatched_count,
    }


async def _prepare_live_case(
    session: aiohttp.ClientSession,
    case: BenchmarkCase,
    canonical_oracle: Mapping[str, Any],
) -> None:
    """Reset stateful handler prerequisites from the live canonical oracle."""
    intent_name = canonical_oracle.get("intent")
    slots = canonical_oracle.get("slots")
    if intent_name is None:
        return
    if not isinstance(intent_name, str) or not isinstance(slots, dict):
        raise BenchmarkError(f"Canonical preparation oracle is invalid for {case.case_id}")
    await _call_service(
        session,
        "assist_canonicalizer_benchmark",
        "prepare_case",
        {
            "intent": intent_name,
            "language": case.language,
            "slots": slots,
        },
    )


async def _canonical_control(
    session: aiohttp.ClientSession,
    case: BenchmarkCase,
    recognition_oracle: Mapping[str, Any],
    conversation_id: str,
) -> Mapping[str, Any]:
    """Execute the reviewed canonical text and return its live semantic oracle."""
    if case.expected_canonical is None:
        raise BenchmarkError(f"Canonical control text is missing for {case.case_id}")
    payload, _, trace = await _run_observed_conversation(
        session,
        case,
        HOME_ASSISTANT_AGENT_ID,
        conversation_id,
        text=case.expected_canonical,
    )
    observation = _response_observation(payload)
    recognized_intent = recognition_oracle.get("intent")
    recognized_slots = recognition_oracle.get("slots")
    if recognized_intent is not None and not isinstance(recognized_intent, str):
        raise BenchmarkError(f"Canonical recognition intent is invalid for {case.case_id}")
    if not isinstance(recognized_slots, dict):
        raise BenchmarkError(f"Canonical recognition slots are invalid for {case.case_id}")
    intent_consistent = _intents_match(trace.actual_intent, recognized_intent)
    slots_consistent = (
        _slots_match(trace.actual_slots, recognized_slots)[0]
        and _slots_match(recognized_slots, trace.actual_slots)[0]
    )
    if not intent_consistent or not slots_consistent:
        raise BenchmarkError(
            f"Canonical control trace differs from recognition for {case.case_id}: "
            f"recognized=({recognized_intent!r}, {recognized_slots!r}), "
            f"executed=({trace.actual_intent!r}, {dict(trace.actual_slots)!r})"
        )
    if observation["response_type"] == "error":
        raise BenchmarkError(
            f"Canonical control execution failed for {case.case_id}: {observation['error_code']!r}"
        )
    return {
        "intent": trace.actual_intent,
        "slots": dict(trace.actual_slots),
        "unmatched_count": recognition_oracle.get("unmatched_count"),
        "recognition_intent": recognized_intent,
        "recognition_slots": dict(recognized_slots),
        "response_type": observation["response_type"],
        "target_ids": observation["target_ids"],
        "entity_ids": observation["entity_ids"],
    }


def _conversation_trace_observation(
    traces: Sequence[Mapping[str, Any]], conversation_id: str
) -> ConversationTraceObservation:
    """Correlate the final Default Agent call and its actual intent tool call."""
    matching: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for trace in traces:
        events = trace.get("events")
        if not isinstance(events, list):
            continue
        async_process_data: Mapping[str, Any] | None = None
        for event in events:
            if not isinstance(event, dict) or event.get("event_type") != "async_process":
                continue
            data = event.get("data")
            if isinstance(data, dict):
                async_process_data = data
                break
        if (
            async_process_data is not None
            and async_process_data.get("conversation_id") == conversation_id
            and async_process_data.get("agent_id") == HOME_ASSISTANT_AGENT_ID
        ):
            matching.append((trace, async_process_data))

    if not matching:
        raise BenchmarkError(
            f"No request-correlated Home Assistant Default Agent trace for {conversation_id}"
        )

    attempts: list[Mapping[str, Any]] = []
    for trace, process_data in matching:
        intent_name: str | None = None
        slots: Mapping[str, Any] = {}
        events = cast(list[Any], trace.get("events", []))
        for event in events:
            if not isinstance(event, dict) or event.get("event_type") != "tool_call":
                continue
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            raw_intent = data.get("intent_name")
            raw_slots = data.get("slots")
            intent_name = raw_intent if isinstance(raw_intent, str) else None
            slots = raw_slots if isinstance(raw_slots, dict) else {}
        delegated_text = process_data.get("text")
        if not isinstance(delegated_text, str):
            raise BenchmarkError(
                f"Default Agent trace for {conversation_id} is missing delegated text"
            )
        attempts.append(
            {
                "delegated_text_sha256": hashlib.sha256(delegated_text.encode()).hexdigest(),
                "intent": intent_name,
                "slots": dict(slots),
            }
        )

    final_attempt = attempts[-1]
    final_text = matching[-1][1].get("text")
    assert isinstance(final_text, str)
    return ConversationTraceObservation(
        actual_intent=cast(str | None, final_attempt["intent"]),
        actual_slots=cast(Mapping[str, Any], final_attempt["slots"]),
        delegated_text=final_text,
        attempts=tuple(attempts),
        trace_count=len(matching),
    )


def _normalized_text(value: str) -> str:
    """Return stable Unicode/case/whitespace normalization for oracle comparison."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalized_slot_value(value: Any) -> tuple[str, str]:
    """Return a typed normalized slot value with numeric coercion."""
    if isinstance(value, bool):
        return "boolean", "true" if value else "false"
    if isinstance(value, int | float | Decimal):
        numeric_text = str(value)
    elif isinstance(value, str):
        numeric_text = value.strip()
    else:
        serialized = orjson.dumps(value, default=str, option=orjson.OPT_SORT_KEYS).decode("utf-8")
        return "json", _normalized_text(serialized)
    if _NUMERIC_SLOT_PATTERN.fullmatch(numeric_text):
        try:
            number = Decimal(numeric_text)
        except InvalidOperation:
            pass
        else:
            return "number", format(number.normalize(), "f")
    return "text", _normalized_text(str(value))


def _intents_match(actual: str | None, expected: str | None) -> bool:
    """Match exact intents plus Home Assistant's generic/list-specific equivalents."""
    if actual == expected:
        return True
    if actual is None or expected is None:
        return False

    def list_action(intent_name: str) -> str | None:
        if intent_name.startswith("HassList"):
            return intent_name[8:]
        return intent_name[16:] if intent_name.startswith("HassShoppingList") else None

    actual_action = list_action(actual)
    expected_action = list_action(expected)
    return actual_action is not None and actual_action == expected_action


def _slots_match(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Compare the expected slot subset against the actual production tool call."""
    failures: list[str] = []
    for expected_key, expected_value in expected.items():
        candidate_keys = _SLOT_KEY_EQUIVALENTS.get(expected_key, (expected_key,))
        actual_values = [actual[key] for key in candidate_keys if key in actual]
        if not actual_values:
            failures.append(f"missing slot {expected_key!r}")
            continue
        expected_normalized = _normalized_slot_value(expected_value)
        if all(
            _normalized_slot_value(actual_value) != expected_normalized
            for actual_value in actual_values
        ):
            failures.append(
                f"slot {expected_key!r} expected {expected_value!r}, got {actual_values!r}"
            )
    return not failures, failures


def _action_slots(slots: Mapping[str, Any], intent_name: str | None) -> dict[str, Any]:
    """Remove target selectors while retaining parameters that change the action."""
    selector_slots = _TARGET_SELECTOR_SLOTS
    if intent_name in _AREA_ACTION_INTENTS:
        selector_slots = selector_slots - {"area"}
    return {key: value for key, value in slots.items() if key not in selector_slots}


def _semantic_slots_match(
    actual_slots: Mapping[str, Any],
    oracle_slots: Mapping[str, Any],
    actual_entity_ids: Sequence[str],
    oracle_entity_ids: Sequence[str],
    intent_name: str | None,
) -> tuple[bool, list[str], str, bool]:
    """Match raw slots or equivalent selectors resolved to the same live entities."""
    raw_correct, raw_failures = _slots_match(actual_slots, oracle_slots)
    if raw_correct:
        return True, [], "raw_slots", False

    actual_targets = frozenset(actual_entity_ids)
    oracle_targets = frozenset(oracle_entity_ids)
    entity_targets_match = bool(actual_targets) and actual_targets == oracle_targets
    if not entity_targets_match:
        return False, raw_failures, "none", False

    actual_action_slots = _action_slots(actual_slots, intent_name)
    oracle_action_slots = _action_slots(oracle_slots, intent_name)
    forward_correct, forward_failures = _slots_match(actual_action_slots, oracle_action_slots)
    reverse_correct, reverse_failures = _slots_match(oracle_action_slots, actual_action_slots)
    if forward_correct and reverse_correct:
        return True, [], "resolved_entities", True
    return (
        False,
        [
            "resolved entities match, but action parameters differ",
            *forward_failures,
            *reverse_failures,
        ],
        "none",
        True,
    )


def _live_oracle_observation(
    case: BenchmarkCase,
    payload: Mapping[str, Any],
    trace: ConversationTraceObservation,
    canonical_oracle: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare one live request with the executed canonical control."""
    observation = _response_observation(payload)
    oracle_intent = canonical_oracle.get("intent")
    oracle_slots = canonical_oracle.get("slots")
    oracle_entity_ids = canonical_oracle.get("entity_ids")
    if oracle_intent is not None and not isinstance(oracle_intent, str):
        raise BenchmarkError(f"Canonical oracle intent is invalid for {case.case_id}")
    if not isinstance(oracle_slots, dict):
        raise BenchmarkError(f"Canonical oracle slots are invalid for {case.case_id}")
    if not isinstance(oracle_entity_ids, list) or not all(
        isinstance(entity_id, str) for entity_id in oracle_entity_ids
    ):
        raise BenchmarkError(f"Canonical oracle entity targets are invalid for {case.case_id}")
    intent_correct = _intents_match(trace.actual_intent, oracle_intent)
    slots_correct, slot_failures, slot_match_method, entity_targets_match = _semantic_slots_match(
        trace.actual_slots,
        oracle_slots,
        cast(list[str], observation["entity_ids"]),
        oracle_entity_ids,
        oracle_intent,
    )
    execution_success = observation["response_type"] != "error"
    observation.update(
        {
            "actual_intent": trace.actual_intent,
            "actual_slots": dict(trace.actual_slots),
            "canonical_oracle_intent": oracle_intent,
            "canonical_oracle_slots": dict(oracle_slots),
            "canonical_oracle_target_ids": list(canonical_oracle.get("target_ids", [])),
            "canonical_oracle_entity_ids": list(oracle_entity_ids),
            "canonical_oracle_unmatched_count": canonical_oracle.get("unmatched_count"),
            "intent_correct": intent_correct,
            "slots_correct": slots_correct,
            "slot_failures": slot_failures,
            "slot_match_method": slot_match_method,
            "entity_targets_match": entity_targets_match,
            "execution_success": execution_success,
            "semantic_correct": intent_correct and slots_correct and execution_success,
            "delegated_text_sha256": hashlib.sha256(trace.delegated_text.encode()).hexdigest(),
            "default_agent_trace_count": trace.trace_count,
            "attempts": [dict(attempt) for attempt in trace.attempts],
        }
    )
    return observation


async def _evaluate_response(
    session: aiohttp.ClientSession,
    case: BenchmarkCase,
    payload: Mapping[str, Any],
    trace: ConversationTraceObservation,
    diagnostics: Mapping[str, Any],
    canonical_oracle: Mapping[str, Any] | None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Evaluate one explicit outcome or production-traced intent/slot oracle."""
    if case.oracle == "intent_slot":
        if canonical_oracle is None:
            raise BenchmarkError(f"Corpus case {case.case_id} has no canonical oracle")
        return _evaluate_intent_slot_response(
            case,
            payload,
            trace,
            diagnostics,
            canonical_oracle,
        )
    return await _evaluate_outcome_response(session, case, payload, trace, diagnostics)


def _response_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract stable response facts shared by both live oracle types."""
    response = payload.get("response")
    if not isinstance(response, dict):
        return {
            "response_type": None,
            "error_code": None,
            "speech": None,
            "target_ids": [],
            "entity_ids": [],
        }
    data = response.get("data")
    speech_payload = response.get("speech")
    plain_speech = speech_payload.get("plain") if isinstance(speech_payload, dict) else None
    success = data.get("success", []) if isinstance(data, dict) else []
    targets = [
        target
        for target in success
        if isinstance(target, dict) and isinstance(target.get("id"), str)
    ]
    return {
        "response_type": response.get("response_type"),
        "error_code": data.get("code") if isinstance(data, dict) else None,
        "speech": plain_speech.get("speech") if isinstance(plain_speech, dict) else None,
        "target_ids": sorted(target["id"] for target in targets),
        "entity_ids": sorted(target["id"] for target in targets if target.get("type") == "entity"),
    }


def _evaluate_intent_slot_response(
    case: BenchmarkCase,
    payload: Mapping[str, Any],
    trace: ConversationTraceObservation,
    diagnostics: Mapping[str, Any],
    canonical_oracle: Mapping[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    """Evaluate corpus labels against the exact live Default Agent tool call."""
    observation = _live_oracle_observation(case, payload, trace, canonical_oracle)
    fallback_reason = diagnostics.get("last_fallback_reason")
    fallback_observed = isinstance(fallback_reason, str) and bool(fallback_reason)
    oracle_intent = observation["canonical_oracle_intent"]
    oracle_slots = cast(Mapping[str, Any], observation["canonical_oracle_slots"])
    intent_correct = bool(observation["intent_correct"])
    slots_correct = bool(observation["slots_correct"])
    slot_failures = cast(list[str], observation["slot_failures"])
    label_intent_matches_oracle = _intents_match(oracle_intent, case.expected_intent)
    label_slots_match_oracle, label_slot_differences = _slots_match(
        oracle_slots, case.expected_slots
    )
    canonical_match = case.expected_canonical is not None and _normalized_text(
        trace.delegated_text
    ) == _normalized_text(case.expected_canonical)
    failures: list[str] = []
    if case.expected_fallback:
        if not fallback_observed:
            failures.append("expected production fallback, but canonical execution was selected")
    else:
        if fallback_observed:
            failures.append(f"unexpected production fallback: {fallback_reason}")
        if not intent_correct:
            failures.append(
                f"intent expected live canonical {oracle_intent!r}, got {trace.actual_intent!r}"
            )
        if not slots_correct:
            failures.extend(slot_failures)

    observation.update(
        {
            "corpus_label_intent_matches_oracle": label_intent_matches_oracle,
            "corpus_label_slots_match_oracle": label_slots_match_oracle,
            "corpus_label_slot_differences": label_slot_differences,
            "canonical_match": canonical_match,
            "delegated_text_sha256": hashlib.sha256(trace.delegated_text.encode()).hexdigest(),
            "expected_canonical_sha256": (
                hashlib.sha256(case.expected_canonical.encode()).hexdigest()
                if case.expected_canonical is not None
                else None
            ),
            "fallback_observed": fallback_observed,
            "fallback_reason": fallback_reason,
        }
    )
    return not failures, failures, observation


async def _evaluate_outcome_response(
    session: aiohttp.ClientSession,
    case: BenchmarkCase,
    payload: Mapping[str, Any],
    trace: ConversationTraceObservation,
    diagnostics: Mapping[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    """Evaluate an explicit response/target/state production oracle."""
    failures: list[str] = []
    observation = _response_observation(payload)
    if observation["response_type"] is None:
        return False, ["response object is missing"], {}
    response_type = observation["response_type"]
    assert case.expected_response_type is not None
    if response_type != case.expected_response_type:
        error_code = observation["error_code"]
        speech = observation["speech"]
        failures.append(
            f"response_type expected {case.expected_response_type!r}, got {response_type!r} "
            f"(code={error_code!r}, speech={speech!r})"
        )
    target_ids = cast(list[str], observation["target_ids"])
    if case.expected_target_id is not None and case.expected_target_id not in target_ids:
        failures.append(f"expected target {case.expected_target_id!r}, got targets {target_ids!r}")

    state_summary: dict[str, Any] = {}
    if case.expected_state is not None:
        state = await _wait_for_expected_state(session, case.expected_state)
        if not isinstance(state, dict):
            failures.append(f"state for {case.expected_state.entity_id!r} is missing")
        else:
            state_summary = {
                "entity_id": state.get("entity_id"),
                "state": state.get("state"),
            }
            if state.get("state") != case.expected_state.state:
                failures.append(
                    f"state expected {case.expected_state.state!r}, got {state.get('state')!r}"
                )
    observation.update(
        {
            "state": state_summary,
            "actual_intent": trace.actual_intent,
            "actual_slots": dict(trace.actual_slots),
            "fallback_reason": diagnostics.get("last_fallback_reason"),
            "execution_success": response_type != "error",
            "default_agent_trace_count": trace.trace_count,
        }
    )
    return not failures, failures, observation


async def _wait_for_expected_state(
    session: aiohttp.ClientSession,
    expected: ExpectedState,
) -> Any:
    """Poll briefly for asynchronous service effects after a response."""
    deadline = time.monotonic() + 5.0
    last_state: Any = None
    while time.monotonic() < deadline:
        last_state = await _request_json(
            session,
            "GET",
            f"/api/states/{expected.entity_id}",
        )
        if isinstance(last_state, dict) and last_state.get("state") == expected.state:
            return last_state
        await asyncio.sleep(0.05)
    return last_state


async def _diagnostics(session: aiohttp.ClientSession) -> Mapping[str, Any]:
    """Read integration diagnostics after one production request."""
    response = await _call_service(
        session,
        "assist_canonicalizer",
        "diagnostics",
        return_response=True,
    )
    if not isinstance(response, dict):
        return {}
    service_response = response.get("service_response")
    return service_response if isinstance(service_response, dict) else {}


async def _execute_suite(
    session: aiohttp.ClientSession,
    cases: Sequence[BenchmarkCase],
    agent_id: str,
    iterations: int,
    warmup: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Execute warmup and measured requests serially with deterministic setup."""
    case_results: list[dict[str, Any]] = []
    suite_failures: list[str] = []
    recognition_oracles: dict[tuple[str, str, str | None], Mapping[str, Any]] = {}
    canonical_controls: dict[tuple[str, str, str | None], Mapping[str, Any]] = {}
    for case in cases:
        if case.expected_canonical is None:
            continue
        oracle_key = (case.language, case.expected_canonical, case.satellite_id)
        if oracle_key not in recognition_oracles:
            loaded_oracle = await _canonical_oracle(session, case)
            if loaded_oracle is None:
                raise BenchmarkError(f"Canonical oracle is missing for {case.case_id}")
            recognition_oracles[oracle_key] = loaded_oracle
        recognition_oracle = recognition_oracles[oracle_key]
        if oracle_key not in canonical_controls:
            await _prepare_live_case(session, case, recognition_oracle)
            canonical_controls[oracle_key] = await _canonical_control(
                session,
                case,
                recognition_oracle,
                f"benchmark-oracle-{case.case_id}",
            )

    for case in cases:
        canonical_oracle: Mapping[str, Any] | None = None
        if case.expected_canonical is not None:
            oracle_key = (case.language, case.expected_canonical, case.satellite_id)
            canonical_oracle = canonical_controls[oracle_key]
        latencies: list[float] = []
        hassil_baseline_latencies: list[float] = []
        measured_passes = 0
        semantic_measured_passes = 0
        hassil_baseline_measured_passes = 0
        failures: list[str] = []
        last_observation: dict[str, Any] = {}
        hassil_baseline_last_observation: dict[str, Any] = {}
        last_diagnostics: Mapping[str, Any] = {}
        total_runs = warmup + iterations
        for run_index in range(total_runs):
            phase = "warmup" if run_index < warmup else "measure"
            phase_index = run_index if phase == "warmup" else run_index - warmup
            baseline_latency_ms: float | None = None
            if case.setup is not None:
                await _call_service(
                    session,
                    case.setup.domain,
                    case.setup.service,
                    case.setup.data,
                )
            if canonical_oracle is not None:
                await _prepare_live_case(session, case, canonical_oracle)
                baseline_conversation_id = f"benchmark-hassil-{case.case_id}-{phase}-{phase_index}"
                (
                    baseline_payload,
                    baseline_latency_ms,
                    baseline_trace,
                ) = await _run_observed_conversation(
                    session,
                    case,
                    HOME_ASSISTANT_AGENT_ID,
                    baseline_conversation_id,
                )
                hassil_baseline_last_observation = _live_oracle_observation(
                    case,
                    baseline_payload,
                    baseline_trace,
                    canonical_oracle,
                )
                await _prepare_live_case(session, case, canonical_oracle)
            conversation_id = f"benchmark-{case.case_id}-{phase}-{phase_index}"
            payload, latency_ms, trace = await _run_observed_conversation(
                session, case, agent_id, conversation_id
            )
            last_diagnostics = await _diagnostics(session)
            if last_diagnostics.get("last_request_id") != conversation_id:
                raise BenchmarkError(
                    f"Production diagnostics request correlation failed for {case.case_id}"
                )
            passed, run_failures, observation = await _evaluate_response(
                session,
                case,
                payload,
                trace,
                last_diagnostics,
                canonical_oracle,
            )
            last_observation = observation
            if phase == "measure":
                latencies.append(latency_ms)
                measured_passes += int(passed)
                semantic_measured_passes += int(bool(observation.get("semantic_correct")))
                if canonical_oracle is not None:
                    if baseline_latency_ms is None:
                        raise BenchmarkError(
                            f"HassIL baseline latency is missing for {case.case_id}"
                        )
                    hassil_baseline_latencies.append(baseline_latency_ms)
                    hassil_baseline_measured_passes += int(
                        bool(hassil_baseline_last_observation.get("semantic_correct"))
                    )
                if not passed:
                    failures.extend(
                        f"{phase}[{phase_index}]: {failure}" for failure in run_failures
                    )

        if case.setup is not None:
            await _call_service(
                session,
                case.setup.domain,
                case.setup.service,
                case.setup.data,
            )
        if failures:
            suite_failures.append(f"{case.case_id}: {'; '.join(failures)}")
        case_results.append(
            {
                "id": case.case_id,
                "language": case.language,
                "query": case.query,
                "oracle": case.oracle,
                "category": case.category,
                "expected_intent": case.expected_intent,
                "expected_slots": dict(case.expected_slots),
                "expected_fallback": case.expected_fallback,
                "passed": not failures and measured_passes == iterations,
                "measured_passes": measured_passes,
                "semantic_measured_passes": semantic_measured_passes,
                "semantic_passed": semantic_measured_passes == iterations,
                "measured_requests": iterations,
                "failures": failures,
                "latency_samples_ms": latencies,
                "latency_ms": _latency_statistics(latencies),
                "last_observation": last_observation,
                "hassil_baseline_measured_passes": hassil_baseline_measured_passes,
                "hassil_baseline_passed": (
                    hassil_baseline_measured_passes == iterations
                    if canonical_oracle is not None
                    else None
                ),
                "hassil_baseline_latency_samples_ms": hassil_baseline_latencies,
                "hassil_baseline_latency_ms": (
                    _latency_statistics(hassil_baseline_latencies)
                    if hassil_baseline_latencies
                    else None
                ),
                "hassil_baseline_last_observation": (
                    hassil_baseline_last_observation if canonical_oracle is not None else None
                ),
                "last_diagnostics": {
                    "last_request_id": last_diagnostics.get("last_request_id"),
                    "last_query_latency_ms": last_diagnostics.get("last_query_latency_ms"),
                    "last_fallback_reason": last_diagnostics.get("last_fallback_reason"),
                    "dynamic_candidate_count": last_diagnostics.get("dynamic_candidate_count"),
                    "selected_delegated_text_hash": last_diagnostics.get(
                        "selected_delegated_text_hash"
                    ),
                    "selected_candidate_source": last_diagnostics.get("selected_candidate_source"),
                    "confidence_gate": last_diagnostics.get("confidence_gate"),
                    "execution_result": last_diagnostics.get("execution_result"),
                    "recognition_kind": last_diagnostics.get("recognition_kind"),
                    "recognition_intent": last_diagnostics.get("recognition_intent"),
                    "recognition_unmatched_count": last_diagnostics.get(
                        "recognition_unmatched_count"
                    ),
                    "recognition_latency_ms": last_diagnostics.get("recognition_latency_ms"),
                    "preflight_attempt_count": last_diagnostics.get("preflight_attempt_count"),
                    "metadata_diverged": last_diagnostics.get("metadata_diverged"),
                    "metadata_divergence_reason": last_diagnostics.get(
                        "metadata_divergence_reason"
                    ),
                    "recovery_used": last_diagnostics.get("recovery_used"),
                    "registry_retrieval": last_diagnostics.get("registry_retrieval"),
                },
            }
        )
    return case_results, suite_failures


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a nearest-rank percentile for a non-empty series."""
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _latency_statistics(values: Sequence[float]) -> dict[str, float]:
    """Return stable latency summary metrics in milliseconds."""
    if not values:
        raise BenchmarkError("Cannot summarize an empty latency sample")
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _aggregate(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate production outcomes, corpus accuracy, and live latency."""
    all_latencies: list[float] = []
    hassil_baseline_latencies: list[float] = []
    passed_requests = 0
    total_requests = 0
    passed_cases = 0
    intent_slot_correct = 0
    fallback_count = 0
    mismatch_count = 0
    execution_success_count = 0
    canonical_match_count = 0
    canonical_oracle_valid_count = 0
    corpus_label_intent_match_count = 0
    corpus_label_slot_match_count = 0
    corpus_case_count = 0
    canonicalizer_correct_count = 0
    hassil_baseline_correct_count = 0
    hassil_baseline_execution_success_count = 0
    recovered_case_count = 0
    regressed_case_count = 0
    both_correct_count = 0
    both_incorrect_count = 0
    resolved_entity_slot_match_count = 0
    for result in case_results:
        samples = result["latency_samples_ms"]
        if not isinstance(samples, list) or not all(
            isinstance(sample, int | float) for sample in samples
        ):
            raise BenchmarkError("Case result latency samples are invalid")
        all_latencies.extend(float(sample) for sample in samples)
        passed_requests += int(result["measured_passes"])
        total_requests += int(result["measured_requests"])
        passed_cases += int(bool(result["passed"]))
        observation = result.get("last_observation")
        if not isinstance(observation, dict):
            raise BenchmarkError("Case result observation is invalid")
        execution_success_count += int(bool(observation.get("execution_success")))
        canonical_match_count += int(bool(observation.get("canonical_match")))
        if result.get("oracle") != "intent_slot":
            continue
        corpus_case_count += 1
        baseline_samples = result.get("hassil_baseline_latency_samples_ms")
        baseline_observation = result.get("hassil_baseline_last_observation")
        if not isinstance(baseline_samples, list) or not all(
            isinstance(sample, int | float) for sample in baseline_samples
        ):
            raise BenchmarkError("HassIL baseline latency samples are invalid")
        if not isinstance(baseline_observation, dict):
            raise BenchmarkError("HassIL baseline observation is invalid")
        hassil_baseline_latencies.extend(float(sample) for sample in baseline_samples)
        fallback_observed = bool(observation.get("fallback_observed"))
        hassil_baseline_fallback = bool(baseline_observation.get("fallback_observed"))
        canonicalizer_correct = bool(result.get("semantic_passed")) and not fallback_observed
        hassil_baseline_correct = (
            bool(result.get("hassil_baseline_passed")) and not hassil_baseline_fallback
        )
        canonicalizer_correct_count += int(canonicalizer_correct)
        hassil_baseline_correct_count += int(hassil_baseline_correct)
        hassil_baseline_execution_success_count += int(
            bool(baseline_observation.get("execution_success"))
        )
        recovered_case_count += int(canonicalizer_correct and not hassil_baseline_correct)
        regressed_case_count += int(hassil_baseline_correct and not canonicalizer_correct)
        both_correct_count += int(canonicalizer_correct and hassil_baseline_correct)
        both_incorrect_count += int(not canonicalizer_correct and not hassil_baseline_correct)
        resolved_entity_slot_match_count += int(
            observation.get("slot_match_method") == "resolved_entities"
        )
        canonical_oracle_valid_count += int(
            observation.get("canonical_oracle_intent") is not None
            and observation.get("canonical_oracle_unmatched_count") == 0
        )
        corpus_label_intent_match_count += int(
            bool(observation.get("corpus_label_intent_matches_oracle"))
        )
        corpus_label_slot_match_count += int(
            bool(observation.get("corpus_label_slots_match_oracle"))
        )
        expected_fallback = bool(result.get("expected_fallback"))
        intent_slots_ok = bool(observation.get("intent_correct")) and bool(
            observation.get("slots_correct")
        )
        intent_slot_correct += int(
            intent_slots_ok and not expected_fallback and not fallback_observed
        )
        fallback_count += int(fallback_observed)
        if expected_fallback:
            mismatch_count += int(not fallback_observed)
        elif not fallback_observed:
            mismatch_count += int(not intent_slots_ok)
    latency_summary = _latency_statistics(all_latencies)
    summary = {
        "case_count": len(case_results),
        "passed_cases": passed_cases,
        "failed_cases": len(case_results) - passed_cases,
        "request_count": total_requests,
        "passed_requests": passed_requests,
        "accuracy_pct": 100.0 * passed_requests / total_requests,
        "latency_ms": latency_summary,
        "request_path_throughput_rps": 1000.0 / latency_summary["mean"],
        "execution_success_count": execution_success_count,
        "execution_success_pct": 100.0 * execution_success_count / len(case_results),
    }
    if corpus_case_count:
        hassil_latency_summary = _latency_statistics(hassil_baseline_latencies)
        canonicalizer_accuracy_pct = 100.0 * canonicalizer_correct_count / corpus_case_count
        hassil_accuracy_pct = 100.0 * hassil_baseline_correct_count / corpus_case_count
        summary |= {
            "corpus_case_count": corpus_case_count,
            "canonicalizer_correct_count": canonicalizer_correct_count,
            "canonicalizer_accuracy_pct": canonicalizer_accuracy_pct,
            "hassil_baseline_correct_count": hassil_baseline_correct_count,
            "hassil_baseline_accuracy_pct": hassil_accuracy_pct,
            "accuracy_uplift_pp": canonicalizer_accuracy_pct - hassil_accuracy_pct,
            "recovered_case_count": recovered_case_count,
            "regressed_case_count": regressed_case_count,
            "net_recovered_case_count": recovered_case_count - regressed_case_count,
            "both_correct_count": both_correct_count,
            "both_incorrect_count": both_incorrect_count,
            "hassil_baseline_execution_success_count": (hassil_baseline_execution_success_count),
            "hassil_baseline_execution_success_pct": 100.0
            * hassil_baseline_execution_success_count
            / corpus_case_count,
            "hassil_baseline_latency_ms": hassil_latency_summary,
            "canonicalizer_mean_latency_overhead_ms": latency_summary["mean"]
            - hassil_latency_summary["mean"],
            "canonicalizer_p95_latency_overhead_ms": latency_summary["p95"]
            - hassil_latency_summary["p95"],
            "resolved_entity_slot_match_count": resolved_entity_slot_match_count,
            "intent_slot_correct": intent_slot_correct,
            "intent_slot_accuracy_pct": 100.0 * intent_slot_correct / corpus_case_count,
            "fallback_count": fallback_count,
            "fallback_rate_pct": 100.0 * fallback_count / corpus_case_count,
            "mismatch_count": mismatch_count,
            "mismatch_rate_pct": 100.0 * mismatch_count / corpus_case_count,
            "canonical_match_count": canonical_match_count,
            "canonical_match_pct": 100.0 * canonical_match_count / corpus_case_count,
            "canonical_oracle_valid_count": canonical_oracle_valid_count,
            "canonical_oracle_valid_pct": 100.0 * canonical_oracle_valid_count / corpus_case_count,
            "corpus_label_intent_match_count": corpus_label_intent_match_count,
            "corpus_label_intent_match_pct": 100.0
            * corpus_label_intent_match_count
            / corpus_case_count,
            "corpus_label_slot_match_count": corpus_label_slot_match_count,
            "corpus_label_slot_match_pct": 100.0
            * corpus_label_slot_match_count
            / corpus_case_count,
        }
    return summary


def _breakdowns(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the same aggregate metrics grouped by language and category."""
    by_language: dict[str, list[Mapping[str, Any]]] = {}
    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for result in case_results:
        language = result.get("language")
        category = result.get("category")
        if not isinstance(language, str) or not isinstance(category, str):
            raise BenchmarkError("Case result language/category metadata is invalid")
        by_language.setdefault(language, []).append(result)
        by_category.setdefault(category, []).append(result)
    return {
        "languages": {key: _aggregate(values) for key, values in sorted(by_language.items())},
        "categories": {key: _aggregate(values) for key, values in sorted(by_category.items())},
    }


def _baseline_regressions(
    report: Mapping[str, Any],
    baseline_path: Path | None,
    max_p95_regression_pct: float,
    allow_homeassistant_upgrade: bool,
) -> list[str]:
    """Compare functional results and p95 latency with an optional baseline."""
    if baseline_path is None:
        return []
    baseline = _load_json_object(baseline_path, "benchmark baseline")
    if baseline.get("report_schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkError("Benchmark baseline schema differs from the current report")
    _verify_baseline_context(report, baseline, allow_homeassistant_upgrade)
    current_summary = report["summary"]
    baseline_summary = baseline.get("summary")
    if not isinstance(baseline_summary, dict):
        raise BenchmarkError("Benchmark baseline summary is missing")
    regressions: list[str] = []
    if int(current_summary["passed_cases"]) < int(baseline_summary.get("passed_cases", 0)):
        regressions.append(
            "passed cases decreased from "
            f"{baseline_summary.get('passed_cases')} to {current_summary['passed_cases']}"
        )
    current_case_passes = _case_pass_map(report, "current report")
    baseline_case_passes = _case_pass_map(baseline, "benchmark baseline")
    if current_case_passes.keys() != baseline_case_passes.keys():
        raise BenchmarkError("Benchmark baseline case identifiers differ")
    if regressed_cases := sorted(
        case_id
        for case_id, baseline_passed in baseline_case_passes.items()
        if baseline_passed and not current_case_passes[case_id]
    ):
        regressions.append(
            f"{len(regressed_cases)} previously passing cases regressed: "
            + ", ".join(regressed_cases)
        )
    baseline_latency = baseline_summary.get("latency_ms")
    if not isinstance(baseline_latency, dict) or not isinstance(
        baseline_latency.get("p95"), int | float
    ):
        raise BenchmarkError("Benchmark baseline p95 latency is missing")
    baseline_p95 = float(baseline_latency["p95"])
    current_p95 = float(current_summary["latency_ms"]["p95"])
    regression_pct = 0.0 if baseline_p95 == 0 else (current_p95 / baseline_p95 - 1.0) * 100.0
    if regression_pct > max_p95_regression_pct:
        regressions.append(
            f"p95 latency regressed {regression_pct:.2f}% "
            f"({baseline_p95:.3f} ms -> {current_p95:.3f} ms)"
        )
    return regressions


def _case_pass_map(report: Mapping[str, Any], description: str) -> dict[str, bool]:
    """Return unique case pass outcomes from a managed-live report."""
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list):
        raise BenchmarkError(f"{description} cases are missing")
    outcomes: dict[str, bool] = {}
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise BenchmarkError(f"{description} contains an invalid case")
        case_id = raw_case.get("id")
        passed = raw_case.get("passed")
        if not isinstance(case_id, str) or not case_id or not isinstance(passed, bool):
            raise BenchmarkError(f"{description} contains invalid case outcomes")
        if case_id in outcomes:
            raise BenchmarkError(f"{description} contains duplicate case {case_id}")
        outcomes[case_id] = passed
    return outcomes


def _verify_baseline_context(
    report: Mapping[str, Any],
    baseline: Mapping[str, Any],
    allow_homeassistant_upgrade: bool,
) -> None:
    """Require equal benchmark inputs, except an explicitly allowed HA upgrade."""
    required_equal = (
        "authoritative",
        "benchmark_mode",
        "execution_tier",
        "suite_id",
        "case_suite_sha256",
        "configuration_sha256",
        "settings",
    )
    differences = {
        key: {"baseline": baseline.get(key), "current": report.get(key)}
        for key in required_equal
        if baseline.get(key) != report.get(key)
    }
    baseline_environment = baseline.get("environment")
    current_environment = report.get("environment")
    if not isinstance(baseline_environment, dict) or not isinstance(current_environment, dict):
        raise BenchmarkError("Benchmark baseline environment is missing")
    baseline_fixture = baseline_environment.get("fixture")
    current_fixture = current_environment.get("fixture")
    if not isinstance(baseline_fixture, dict) or not isinstance(current_fixture, dict):
        raise BenchmarkError("Benchmark baseline fixture metadata is missing")
    fixture_keys = ("fixture_id", "schema_version", "fingerprint", "counts", "domain_counts")
    for key in fixture_keys:
        if baseline_fixture.get(key) != current_fixture.get(key):
            differences[f"environment.fixture.{key}"] = {
                "baseline": baseline_fixture.get(key),
                "current": current_fixture.get(key),
            }
    if not allow_homeassistant_upgrade:
        for key in ("homeassistant_version", "dependencies", "python_version"):
            if baseline_environment.get(key) != current_environment.get(key):
                differences[f"environment.{key}"] = {
                    "baseline": baseline_environment.get(key),
                    "current": current_environment.get(key),
                }
    if baseline_environment.get("installed_intent_languages") != current_environment.get(
        "installed_intent_languages"
    ):
        differences["environment.installed_intent_languages"] = {
            "baseline": baseline_environment.get("installed_intent_languages"),
            "current": current_environment.get("installed_intent_languages"),
        }
    if baseline_environment.get("language_smoke_manifest_sha256") != current_environment.get(
        "language_smoke_manifest_sha256"
    ):
        differences["environment.language_smoke_manifest_sha256"] = {
            "baseline": baseline_environment.get("language_smoke_manifest_sha256"),
            "current": current_environment.get("language_smoke_manifest_sha256"),
        }
    if allow_homeassistant_upgrade and baseline_environment.get(
        "integration_source_sha256"
    ) != current_environment.get("integration_source_sha256"):
        differences["environment.integration_source_sha256"] = {
            "baseline": baseline_environment.get("integration_source_sha256"),
            "current": current_environment.get("integration_source_sha256"),
        }
    if differences:
        raise BenchmarkError(f"Benchmark baseline context differs: {differences}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write the JSON benchmark report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(
        orjson.dumps(
            payload,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE,
        )
    )
    temporary.replace(path)


def align_table(
    headers: tuple[str, ...],
    rows: Sequence[tuple[str, ...]],
    *,
    alignments: str = "<",
    padding: int = 1,
    sep: str = " | ",
) -> tuple[str, str, list[str]]:
    """Return dynamically-aligned table lines: ``(header, separator, data_lines)``."""
    if not headers:
        return "", "", []
    ncols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    widths = [w + padding for w in widths]

    if len(alignments) == 1:
        aligns = list(alignments * ncols)
    else:
        aligns = list(alignments)
        if len(aligns) < ncols:
            # Pad with last alignment character to match column count
            aligns.extend([aligns[-1]] * (ncols - len(aligns)))
        aligns = aligns[:ncols]

    hdr_parts = [f"{h:{a}{w}}" for h, a, w in zip(headers, aligns, widths, strict=True)]
    header_line = sep.join(hdr_parts)
    sep_line = sep.join("-" * w for w in widths)

    data_lines: list[str] = []
    for row in rows:
        parts = [f"{c:{a}{w}}" for c, a, w in zip(row, aligns, widths, strict=True)]
        data_lines.append(sep.join(parts))
    return header_line, sep_line, data_lines


def _md_aligned_table(
    headers: tuple[str, ...],
    alignments: str,
    rows: Sequence[tuple[str, ...]],
) -> list[str]:
    """Generate a markdown formatted table with dynamically aligned columns."""
    if not headers:
        return []
    ncols = len(headers)
    if len(alignments) == 1:
        aligns = list(alignments * ncols)
    else:
        aligns = list(alignments)
        if len(aligns) < ncols:
            aligns.extend([aligns[-1]] * (ncols - len(aligns)))
        aligns = aligns[:ncols]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    widths = [max(w, 3) for w in widths]

    hdr_parts = [f" {h:{a}{w}} " for h, a, w in zip(headers, aligns, widths, strict=True)]
    lines: list[str] = ["|" + "|".join(hdr_parts) + "|"]
    sep_parts: list[str] = []
    for i, w in enumerate(widths):
        dashes = w - 1
        if aligns[i] == ">":
            sep_parts.append(" " + "-" * dashes + ": ")
        elif aligns[i] == "^":
            sep_parts.append(" :" + "-" * (w - 2) + ": ")
        else:
            sep_parts.append(" :" + "-" * dashes + " ")
    lines.append("|" + "|".join(sep_parts) + "|")

    for row in rows:
        parts = [f" {c:{a}{w}} " for c, a, w in zip(row, aligns, widths, strict=True)]
        lines.append("|" + "|".join(parts) + "|")

    return lines


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    """Write a comprehensive human-readable managed-live report."""
    summary = report["summary"]
    fixture = report["environment"]["fixture"]
    language_support = report.get("language_support", {})
    lines = [
        "# Managed-live Assist Canonicalizer benchmark",
        "",
        f"- Home Assistant: `{report['environment']['homeassistant_version']}`",
        f"- Python: `{report['environment'].get('python_version', 'unknown')}`",
        f"- Fixture: `{fixture['fixture_id']}` (`{fixture['fingerprint']}`)",
        f"- Policy-oracle cases passed: {summary['passed_cases']}/{summary['case_count']}",
        f"- Canonicalizer mean latency: {summary['latency_ms']['mean']:.3f} ms",
        f"- Canonicalizer p50 latency: {summary['latency_ms']['median']:.3f} ms",
        f"- Canonicalizer p95 latency: {summary['latency_ms']['p95']:.3f} ms",
        f"- Accuracy-gated languages: {', '.join(language_support.get('accuracy_gated', []))}",
        f"- Managed compatibility-smoke languages: "
        f"{language_support.get('compatibility_smoke_count', 0)}",
    ]
    has_corpus_metrics = "intent_slot_accuracy_pct" in summary
    if has_corpus_metrics:
        lines.extend(
            (
                f"- Canonicalizer semantic accuracy: {summary['canonicalizer_accuracy_pct']:.2f}%",
                f"- Direct HassIL semantic accuracy: "
                f"{summary['hassil_baseline_accuracy_pct']:.2f}%",
                f"- Canonicalizer uplift over HassIL: "
                f"{summary['accuracy_uplift_pp']:+.2f} percentage points",
                f"- Cases recovered/regressed versus HassIL: "
                f"{summary['recovered_case_count']}/{summary['regressed_case_count']}",
                f"- Canonicalizer fallback rate: {summary['fallback_rate_pct']:.2f}%",
                f"- Canonicalizer mismatch rate: {summary['mismatch_rate_pct']:.2f}%",
                f"- Canonicalizer non-error responses: {summary['execution_success_pct']:.2f}%",
                f"- Direct HassIL non-error responses: "
                f"{summary['hassil_baseline_execution_success_pct']:.2f}%",
                f"- Direct HassIL mean latency: "
                f"{summary['hassil_baseline_latency_ms']['mean']:.3f} ms",
                f"- Direct HassIL p50 latency: "
                f"{summary['hassil_baseline_latency_ms']['median']:.3f} ms",
                f"- Direct HassIL p95 latency: "
                f"{summary['hassil_baseline_latency_ms']['p95']:.3f} ms",
                f"- Selector variants accepted by equal resolved entities: "
                f"{summary['resolved_entity_slot_match_count']}",
                f"- Valid live canonical controls: "
                f"{summary['canonical_oracle_valid_count']}/{summary['corpus_case_count']}",
                f"- Curated intent labels agreeing with the live oracle: "
                f"{summary['corpus_label_intent_match_count']}/"
                f"{summary['corpus_case_count']}",
            )
        )

    breakdowns = report.get("breakdowns", {})
    languages = breakdowns.get("languages", {}) if isinstance(breakdowns, dict) else {}
    if isinstance(languages, dict) and languages:
        lines.extend(
            (
                "",
                "## Per-language production results",
                "",
            )
        )
        if has_corpus_metrics:
            lang_headers = (
                "Language",
                "Cases",
                "Canonicalizer",
                "HassIL",
                "Uplift pp",
                "Recovered",
                "Regressed",
                "Fallback",
                "Mismatch",
                "Canon mean",
                "Canon p50",
                "Canon p95",
                "HassIL mean",
                "HassIL p50",
                "HassIL p95",
            )
            lang_aligns = "<>>>>>>>>>>>>>>"
            lang_rows = [
                (
                    f"`{language}`",
                    str(metrics["case_count"]),
                    f"{metrics.get('canonicalizer_accuracy_pct', 0.0):.2f}%",
                    f"{metrics.get('hassil_baseline_accuracy_pct', 0.0):.2f}%",
                    f"{metrics.get('accuracy_uplift_pp', 0.0):+.2f}",
                    str(metrics.get("recovered_case_count", 0)),
                    str(metrics.get("regressed_case_count", 0)),
                    f"{metrics.get('fallback_rate_pct', 0.0):.2f}%",
                    f"{metrics.get('mismatch_rate_pct', 0.0):.2f}%",
                    f"{metrics['latency_ms']['mean']:.3f}",
                    f"{metrics['latency_ms']['median']:.3f}",
                    f"{metrics['latency_ms']['p95']:.3f}",
                    f"{metrics.get('hassil_baseline_latency_ms', {}).get('mean', 0.0):.3f}",
                    f"{metrics.get('hassil_baseline_latency_ms', {}).get('median', 0.0):.3f}",
                    f"{metrics.get('hassil_baseline_latency_ms', {}).get('p95', 0.0):.3f}",
                )
                for language, metrics in languages.items()
            ]
        else:
            lang_headers = (
                "Language",
                "Cases",
                "Passed",
                "Pass rate",
                "Canon mean",
                "Canon p50",
                "Canon p95",
            )
            lang_aligns = "<>>>>>>"
            lang_rows = [
                (
                    f"`{language}`",
                    str(metrics["case_count"]),
                    str(metrics.get("passed_cases", 0)),
                    f"{metrics.get('accuracy_pct', 0.0):.2f}%",
                    f"{metrics['latency_ms']['mean']:.3f}",
                    f"{metrics['latency_ms']['median']:.3f}",
                    f"{metrics['latency_ms']['p95']:.3f}",
                )
                for language, metrics in languages.items()
            ]
        lines.extend(_md_aligned_table(lang_headers, lang_aligns, lang_rows))
    categories = breakdowns.get("categories", {}) if isinstance(breakdowns, dict) else {}
    if isinstance(categories, dict) and categories:
        lines.extend(
            (
                "",
                "## Per-category production results",
                "",
            )
        )
        if has_corpus_metrics:
            cat_headers = (
                "Category",
                "Cases",
                "Canonicalizer",
                "HassIL",
                "Uplift pp",
                "Recovered",
                "Regressed",
                "Fallback",
                "Mismatch",
            )
            cat_aligns = "<>>>>>>>>"
            cat_rows = [
                (
                    f"`{category}`",
                    str(metrics["case_count"]),
                    f"{metrics.get('canonicalizer_accuracy_pct', 0.0):.2f}%",
                    f"{metrics.get('hassil_baseline_accuracy_pct', 0.0):.2f}%",
                    f"{metrics.get('accuracy_uplift_pp', 0.0):+.2f}",
                    str(metrics.get("recovered_case_count", 0)),
                    str(metrics.get("regressed_case_count", 0)),
                    f"{metrics.get('fallback_rate_pct', 0.0):.2f}%",
                    f"{metrics.get('mismatch_rate_pct', 0.0):.2f}%",
                )
                for category, metrics in categories.items()
            ]
        else:
            cat_headers = ("Category", "Cases", "Passed", "Pass rate")
            cat_aligns = "<>>>"
            cat_rows = [
                (
                    f"`{category}`",
                    str(metrics["case_count"]),
                    str(metrics.get("passed_cases", 0)),
                    f"{metrics.get('accuracy_pct', 0.0):.2f}%",
                )
                for category, metrics in categories.items()
            ]
        lines.extend(_md_aligned_table(cat_headers, cat_aligns, cat_rows))
    lines.extend(
        (
            "",
            "## Case results",
            "",
        )
    )
    case_headers = (
        "Case",
        "Language",
        "Category",
        "Policy pass",
        "Canonicalizer correct",
        "HassIL correct",
        "Fallback",
        "Canon mean ms",
        "HassIL mean ms",
    )
    case_rows = []
    for case in report["cases"]:
        observation = case["last_observation"]
        hassil_passed = case.get("hassil_baseline_passed")
        hassil_latency = case.get("hassil_baseline_latency_ms")
        hassil_result = ("yes" if hassil_passed else "no") if hassil_passed is not None else "n/a"
        hassil_mean = f"{hassil_latency['mean']:.3f}" if isinstance(hassil_latency, dict) else "n/a"
        case_rows.append(
            (
                f"`{case['id']}`",
                f"`{case['language']}`",
                f"`{case['category']}`",
                "yes" if case["passed"] else "no",
                "yes" if case.get("semantic_passed") else "no",
                hassil_result,
                "yes" if observation.get("fallback_observed") else "no",
                f"{case['latency_ms']['mean']:.3f}",
                hassil_mean,
            )
        )
    lines.extend(_md_aligned_table(case_headers, "<><^^^^>>", case_rows))
    if failures := report.get("case_failures", []):
        lines.extend(("", "## Failed live oracles", ""))
        lines.extend(f"- {failure}" for failure in failures)
    if regressions := report.get("regressions", []):
        lines.extend(("", "## Regressions", ""))
        lines.extend(f"- {regression}" for regression in regressions)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comma_separated_values(raw_value: str | None) -> tuple[str, ...]:
    """Return normalized unique comma/space separated CLI values."""
    if raw_value is None:
        return ()
    if values := tuple(
        dict.fromkeys(
            value.strip().casefold() for value in re.split(r"[,\s]+", raw_value) if value.strip()
        )
    ):
        return values
    raise BenchmarkError("A case filter was supplied without any values")


def _select_cases(
    cases: Sequence[BenchmarkCase],
    languages: tuple[str, ...],
    categories: tuple[str, ...],
    case_limit: int | None,
) -> tuple[BenchmarkCase, ...]:
    """Apply deterministic development filters without changing tracked inputs."""
    selected = tuple(
        case
        for case in cases
        if (not languages or case.language.casefold() in languages)
        and (not categories or case.category.casefold() in categories)
    )
    if case_limit is not None:
        selected = selected[:case_limit]
    if not selected:
        raise BenchmarkError("Case filters selected zero benchmark cases")
    return selected


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Own the full ephemeral Home Assistant benchmark lifecycle."""
    dependencies = verify_benchmark_dependencies()
    fixture = _load_json_object(FIXTURE_PATH, "benchmark fixture")
    suite_id, all_cases = load_cases(args.cases)
    languages = _comma_separated_values(args.languages)
    categories = _comma_separated_values(args.categories)
    cases = _select_cases(all_cases, languages, categories, args.case_limit)
    language_smoke_commands = build_language_smoke_commands()
    integration_source_sha256 = _tree_sha256(INTEGRATION_PATH)
    language_smoke_manifest_sha256 = _canonical_payload_sha256(
        [
            {
                "language": command.language,
                "text": command.text,
                "target_domain": command.target_domain,
                "target_entity_id": command.target_entity_id,
            }
            for command in language_smoke_commands
        ]
    )
    _assert_port_available()
    config_dir = _create_config_dir()
    managed: ManagedProcess | None = None
    started_at = time.perf_counter()
    try:
        await _run_config_check(config_dir)
        print("HA_CONFIG_OK", flush=True)
        managed = await _start_home_assistant(config_dir)
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await _wait_for_http(session, managed)
            print("HA_HTTP_READY", flush=True)
            token = await _onboard(session)
            session.headers.update({"Authorization": f"Bearer {token}"})
            await _wait_for_fixture(session, fixture)
            await _create_config_entry(session, "shopping_list")
            await _call_service(session, "assist_canonicalizer_benchmark", "reapply")
            fixture_state = await _wait_for_fixture(session, fixture)
            print("BENCHMARK_FIXTURE_VERIFIED", flush=True)

            config = await _request_json(session, "GET", "/api/config")
            if not isinstance(config, dict):
                raise BenchmarkError("Home Assistant config response must be an object")
            if config.get("version") != dependencies["homeassistant"]:
                raise BenchmarkError(
                    "Home Assistant API version differs from the verified Python distribution: "
                    f"{config.get('version')} != {dependencies['homeassistant']}"
                )
            agent_id = await _create_integration_entry(session)
            await _wait_for_agent(session)
            prepared_languages = await _prepare_languages(
                session,
                tuple(
                    {case.language for case in cases}
                    | {command.language for command in language_smoke_commands}
                ),
            )
            case_results, suite_failures = await _execute_suite(
                session,
                cases,
                agent_id,
                args.iterations,
                args.warmup,
            )
            language_smoke = await _execute_language_smoke(
                session,
                language_smoke_commands,
                agent_id,
            )

        summary = _aggregate(case_results)
        case_suite_sha256 = _case_input_sha256(args.cases)
        configuration_sha256 = _file_sha256(CONFIGURATION_PATH)
        context_fingerprint = _canonical_payload_sha256(
            {
                "homeassistant_version": dependencies["homeassistant"],
                "python_version": sys.version.split()[0],
                "dependencies": dependencies["packages"],
                "fixture_fingerprint": fixture_state["fingerprint"],
                "case_suite_sha256": case_suite_sha256,
                "configuration_sha256": configuration_sha256,
                "integration_source_sha256": integration_source_sha256,
                "language_smoke_manifest_sha256": language_smoke_manifest_sha256,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "device_id": BENCHMARK_DEVICE_ID,
                "satellite_id": CONTEXT_SATELLITE_ID,
            }
        )
        report: dict[str, Any] = {
            "report_schema_version": BENCHMARK_SCHEMA_VERSION,
            "authoritative": True,
            "benchmark_mode": "managed_live",
            "execution_tier": "managed_live",
            "suite_id": suite_id,
            "case_suite_sha256": case_suite_sha256,
            "case_input_files": _case_input_files(args.cases),
            "configuration_sha256": configuration_sha256,
            "environment": {
                "homeassistant_version": dependencies["homeassistant"],
                "python_version": sys.version.split()[0],
                "dependencies": dependencies["packages"],
                "installed_intent_languages": [
                    command.language for command in language_smoke_commands
                ],
                "integration_source_sha256": integration_source_sha256,
                "language_smoke_manifest_sha256": language_smoke_manifest_sha256,
                "context_fingerprint": context_fingerprint,
                "fixture": {
                    "fixture_id": fixture["fixture_id"],
                    "schema_version": fixture["schema_version"],
                    "fingerprint": fixture_state["fingerprint"],
                    "counts": fixture["expected_counts"],
                    "domain_counts": fixture["expected_domain_counts"],
                    "runtime_state_count": fixture_state.get("runtime_state_count"),
                },
            },
            "settings": {
                "iterations": args.iterations,
                "warmup": args.warmup,
                "serial_execution": True,
                "endpoint": f"{HOST}:{PORT}",
                "languages": list(languages),
                "categories": list(categories),
                "case_limit": args.case_limit,
                "device_id": BENCHMARK_DEVICE_ID,
                "satellite_id": CONTEXT_SATELLITE_ID,
                "canonical_oracle": "executed_live_default_agent_canonical_control",
                "hassil_baseline": "paired_original_query_to_live_default_agent",
                "observed_result": "production_default_agent_trace_and_resolved_entities",
            },
            "prepared_languages": prepared_languages,
            "language_support": {
                "accuracy_gated": sorted(ACCURACY_GATED_LANGUAGES),
                "compatibility_smoke_count": len(language_smoke),
                "compatibility_smoke": language_smoke,
            },
            "summary": summary,
            "breakdowns": _breakdowns(case_results),
            "cases": case_results,
            "startup_and_run_seconds": time.perf_counter() - started_at,
        }
        regressions = _baseline_regressions(
            report,
            args.baseline,
            args.max_p95_regression_pct,
            args.allow_homeassistant_upgrade,
        )
        report["regressions"] = regressions
        report["case_failures"] = suite_failures
        threshold_failures = []
        if args.min_intent_slot_accuracy is not None:
            actual_accuracy = summary.get("intent_slot_accuracy_pct", 0.0)
            if actual_accuracy < args.min_intent_slot_accuracy:
                threshold_failures.append(
                    f"intent/slot accuracy {actual_accuracy:.2f}% is below "
                    f"{args.min_intent_slot_accuracy:.2f}%"
                )
        if args.max_fallback_rate is not None:
            actual_fallback = summary.get("fallback_rate_pct", 0.0)
            if actual_fallback > args.max_fallback_rate:
                threshold_failures.append(
                    f"fallback rate {actual_fallback:.2f}% is above {args.max_fallback_rate:.2f}%"
                )
        if args.max_mismatch_rate is not None:
            actual_mismatch = summary.get("mismatch_rate_pct", 0.0)
            if actual_mismatch > args.max_mismatch_rate:
                threshold_failures.append(
                    f"mismatch rate {actual_mismatch:.2f}% is above {args.max_mismatch_rate:.2f}%"
                )
        languages_breakdown = report.get("breakdowns", {}).get("languages", {})
        for lang, lang_summary in sorted(languages_breakdown.items()):
            if args.min_language_intent_slot_accuracy is not None:
                actual_accuracy = lang_summary.get("intent_slot_accuracy_pct", 0.0)
                if actual_accuracy < args.min_language_intent_slot_accuracy:
                    threshold_failures.append(
                        f"{lang.upper()}: intent/slot accuracy {actual_accuracy:.2f}% is below "
                        f"{args.min_language_intent_slot_accuracy:.2f}%"
                    )
            if args.max_language_fallback_rate is not None:
                actual_fallback = lang_summary.get("fallback_rate_pct", 0.0)
                if actual_fallback > args.max_language_fallback_rate:
                    threshold_failures.append(
                        f"{lang.upper()}: fallback rate {actual_fallback:.2f}% is above "
                        f"{args.max_language_fallback_rate:.2f}%"
                    )
            if args.max_language_mismatch_rate is not None:
                actual_mismatch = lang_summary.get("mismatch_rate_pct", 0.0)
                if actual_mismatch > args.max_language_mismatch_rate:
                    threshold_failures.append(
                        f"{lang.upper()}: mismatch rate {actual_mismatch:.2f}% is above "
                        f"{args.max_language_mismatch_rate:.2f}%"
                    )
        report["threshold_failures"] = threshold_failures
        _write_json(args.output_json, report)
        _write_markdown(args.output_markdown, report)
        if args.fail_on_case_failure and suite_failures:
            raise BenchmarkError("Functional benchmark failures:\n- " + "\n- ".join(suite_failures))
        if args.fail_on_regression and regressions:
            raise BenchmarkError("Benchmark regressions:\n- " + "\n- ".join(regressions))
        if threshold_failures:
            raise BenchmarkError(
                "Benchmark threshold failures:\n- " + "\n- ".join(threshold_failures)
            )
        return report
    except Exception as err:
        if tail := _log_tail(managed):
            raise BenchmarkError(f"{err}\n\nSanitized Home Assistant log tail:\n{tail}") from err
        raise
    finally:
        await _stop_home_assistant(managed)
        shutil.rmtree(config_dir)


def _parser() -> argparse.ArgumentParser:
    """Build the managed-live benchmark command-line parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark Assist Canonicalizer through a fresh live Home Assistant process"
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=REAL_WORLD_DATASET_DIR,
        help="Tracked corpus directory or explicit live-oracle file (default: tests/real_world)",
    )
    parser.add_argument(
        "--languages",
        default=None,
        help="Optional comma/space-separated language filter",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Optional comma/space-separated category filter",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Optional deterministic prefix limit for investigation only",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Measured serial requests per case (default: 1)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Untimed warmup requests per case (default: 0; indexes are prepared separately)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="JSON report path (default: scratch/benchmark/managed_live_report.json)",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_MARKDOWN,
        help="Markdown report path (default: scratch/benchmark/managed_live_report.md)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional prior managed-live JSON report",
    )
    parser.add_argument(
        "--max-p95-regression-pct",
        type=float,
        default=10.0,
        help="Maximum allowed p95 latency increase (default: 10.0)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit nonzero when the optional baseline exceeds regression limits",
    )
    parser.add_argument(
        "--fail-on-case-failure",
        action="store_true",
        help="Exit nonzero when any case misses its live oracle",
    )
    parser.add_argument(
        "--allow-homeassistant-upgrade",
        action="store_true",
        help=(
            "Compare with a baseline from another Home Assistant/dependency version while "
            "still requiring identical fixture, cases, configuration, and run settings"
        ),
    )
    parser.add_argument(
        "--min-intent-slot-accuracy",
        type=float,
        default=None,
        help="Fail when production-flow intent/slot accuracy falls below this percentage",
    )
    parser.add_argument(
        "--max-fallback-rate",
        type=float,
        default=None,
        help="Fail when production-flow fallback rate exceeds this percentage",
    )
    parser.add_argument(
        "--max-mismatch-rate",
        type=float,
        default=None,
        help="Fail when production-flow mismatch rate exceeds this percentage",
    )
    parser.add_argument(
        "--min-language-intent-slot-accuracy",
        type=float,
        default=None,
        help="Fail when any language's intent/slot accuracy falls below this percentage",
    )
    parser.add_argument(
        "--max-language-fallback-rate",
        type=float,
        default=None,
        help="Fail when any language's fallback rate exceeds this percentage",
    )
    parser.add_argument(
        "--max-language-mismatch-rate",
        type=float,
        default=None,
        help="Fail when any language's mismatch rate exceeds this percentage",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Validate and list cases without starting Home Assistant",
    )
    return parser


def main() -> None:
    """Run the command-line managed-live benchmark."""
    parser = _parser()
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("--case-limit must be positive")
    if args.max_p95_regression_pct < 0:
        parser.error("--max-p95-regression-pct must be non-negative")
    if args.min_intent_slot_accuracy is not None and not (
        0.0 <= args.min_intent_slot_accuracy <= 100.0
    ):
        parser.error("--min-intent-slot-accuracy must be between 0.0 and 100.0")
    if args.max_fallback_rate is not None and not (0.0 <= args.max_fallback_rate <= 100.0):
        parser.error("--max-fallback-rate must be between 0.0 and 100.0")
    if args.max_mismatch_rate is not None and not (0.0 <= args.max_mismatch_rate <= 100.0):
        parser.error("--max-mismatch-rate must be between 0.0 and 100.0")
    if args.min_language_intent_slot_accuracy is not None and not (
        0.0 <= args.min_language_intent_slot_accuracy <= 100.0
    ):
        parser.error("--min-language-intent-slot-accuracy must be between 0.0 and 100.0")
    if args.max_language_fallback_rate is not None and not (
        0.0 <= args.max_language_fallback_rate <= 100.0
    ):
        parser.error("--max-language-fallback-rate must be between 0.0 and 100.0")
    if args.max_language_mismatch_rate is not None and not (
        0.0 <= args.max_language_mismatch_rate <= 100.0
    ):
        parser.error("--max-language-mismatch-rate must be between 0.0 and 100.0")
    if args.output_json == args.output_markdown:
        parser.error("--output-json and --output-markdown must be different paths")

    # Sanitize and resolve path arguments after parsing to break CodeQL taint
    try:
        safe_cases = _safe_repository_path(args.cases, "case suite")
        safe_output_json = _safe_repository_path(args.output_json, "JSON output")
        safe_output_markdown = _safe_repository_path(args.output_markdown, "Markdown output")
        safe_baseline = (
            _safe_repository_path(args.baseline, "baseline") if args.baseline is not None else None
        )
    except BenchmarkError as err:
        parser.error(str(err))

    # Construct safe namespace to pass to down-stream functions
    safe_args = argparse.Namespace(
        cases=safe_cases,
        languages=args.languages,
        categories=args.categories,
        case_limit=args.case_limit,
        iterations=args.iterations,
        warmup=args.warmup,
        output_json=safe_output_json,
        output_markdown=safe_output_markdown,
        baseline=safe_baseline,
        max_p95_regression_pct=args.max_p95_regression_pct,
        allow_homeassistant_upgrade=args.allow_homeassistant_upgrade,
        fail_on_regression=args.fail_on_regression,
        fail_on_case_failure=args.fail_on_case_failure,
        min_intent_slot_accuracy=args.min_intent_slot_accuracy,
        max_fallback_rate=args.max_fallback_rate,
        max_mismatch_rate=args.max_mismatch_rate,
        min_language_intent_slot_accuracy=args.min_language_intent_slot_accuracy,
        max_language_fallback_rate=args.max_language_fallback_rate,
        max_language_mismatch_rate=args.max_language_mismatch_rate,
        list_cases=args.list_cases,
    )

    if safe_args.list_cases:
        suite_id, all_cases = load_cases(safe_args.cases)
        try:
            cases = _select_cases(
                all_cases,
                _comma_separated_values(safe_args.languages),
                _comma_separated_values(safe_args.categories),
                safe_args.case_limit,
            )
        except BenchmarkError as err:
            parser.error(str(err))
        print(f"{suite_id}: {len(cases)} cases")
        for case in cases:
            print(f"{case.case_id}\t{case.language}\t{case.query}")
        return

    env_info = _benchmark_environment_summary()
    print(f"BENCHMARK_START mode=managed_live {env_info}", flush=True)
    try:
        report = asyncio.run(run_benchmark(safe_args))
    except (BenchmarkError, OSError, ValueError) as err:
        print(f"BENCHMARK_FAILED: {err}", file=sys.stderr, flush=True)
        raise SystemExit(1) from err
    summary = report["summary"]
    success_fields = [
        "BENCHMARK_SUCCESS",
        f"cases={summary['passed_cases']}/{summary['case_count']}",
    ]
    if "canonicalizer_accuracy_pct" in summary:
        canonicalizer_lat = summary["latency_ms"]
        hassil_lat = summary["hassil_baseline_latency_ms"]
        success_fields.extend(
            (
                f"canonicalizer_accuracy={summary['canonicalizer_accuracy_pct']:.2f}%",
                f"hassil_accuracy={summary['hassil_baseline_accuracy_pct']:.2f}%",
                f"uplift_pp={summary['accuracy_uplift_pp']:+.2f}",
                f"canonicalizer_fallback={summary['fallback_rate_pct']:.2f}%",
                f"canonicalizer_mismatch={summary['mismatch_rate_pct']:.2f}%",
                f"canonicalizer_mean_ms={canonicalizer_lat['mean']:.3f}",
                f"canonicalizer_p50_ms={canonicalizer_lat['median']:.3f}",
                f"canonicalizer_p95_ms={canonicalizer_lat['p95']:.3f}",
                f"hassil_mean_ms={hassil_lat['mean']:.3f}",
                f"hassil_p50_ms={hassil_lat['median']:.3f}",
                f"hassil_p95_ms={hassil_lat['p95']:.3f}",
            )
        )
    else:
        success_fields.extend(
            (
                f"accuracy={summary['accuracy_pct']:.2f}%",
                f"mean_ms={summary['latency_ms']['mean']:.3f}",
                f"p50_ms={summary['latency_ms']['median']:.3f}",
                f"p95_ms={summary['latency_ms']['p95']:.3f}",
            )
        )
    print(" ".join(success_fields), flush=True)
    print(f"JSON report: {safe_args.output_json}")
    print(f"Markdown report: {safe_args.output_markdown}")


if __name__ == "__main__":
    main()
