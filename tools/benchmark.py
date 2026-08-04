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
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from string import ascii_letters, digits
from typing import Any, BinaryIO, Literal

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

BENCHMARK_SCHEMA_VERSION = 4
BENCHMARK_GROUP = "ha-benchmark"
REAL_WORLD_SUITE_ID = "managed_live_real_world_v1"
HOST = "127.0.0.1"
PORT = 8123
BASE_URL = f"http://{HOST}:{PORT}"
WEBSOCKET_URL = f"ws://{HOST}:{PORT}/api/websocket"
FIXTURE_ENTITY_ID = "sensor.assist_canonicalizer_benchmark_fixture"
AGENT_ENTITY_ID = "conversation.assist_canonicalizer"
HOME_ASSISTANT_AGENT_ID = "conversation.home_assistant"
CONTEXT_SATELLITE_ID = "light.living_room_rgbww_lights"
BENCHMARK_DEVICE_ID = "assist-canonicalizer-benchmark-device"
HTTP_TIMEOUT_SECONDS = 30.0
PROCESS_TIMEOUT_SECONDS = 180.0
RESTART_TIMEOUT_SECONDS = 30.0
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
    data: Mapping[str, object]


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
    log_handle: BinaryIO
    log_path: Path


@dataclass(frozen=True, slots=True)
class ConversationTraceObservation:
    """Request-correlated facts emitted by Home Assistant's production trace."""

    actual_intent: str | None
    actual_slots: Mapping[str, object]
    delegated_text: str
    attempts: tuple[Mapping[str, object], ...]
    trace_count: int


def _required_string(mapping: Mapping[str, object], key: str, context: str) -> str:
    """Return a required non-empty string from a JSON object."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{context}.{key} must be a non-empty string")
    return value


def _json_object(value: object, context: str) -> dict[str, object]:
    """Return a JSON object after validating that every key is a string."""
    if not isinstance(value, dict):
        raise BenchmarkError(f"{context} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise BenchmarkError(f"{context} contains a non-string key")
        result[key] = item
    return result


def _string_list(value: object, context: str) -> list[str]:
    """Return a JSON array after validating that every item is a string."""
    if not isinstance(value, list):
        raise BenchmarkError(f"{context} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise BenchmarkError(f"{context} must contain only strings")
        result.append(item)
    return result


def _load_json_object(path: Path, description: str) -> dict[str, object]:
    """Load a UTF-8 JSON file and require an object root."""
    try:
        loaded = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as err:
        raise BenchmarkError(f"Unable to load {description} at {path}: {err}") from err
    return _json_object(loaded, f"{description} root at {path}")


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
    for index, value in enumerate(raw_cases):
        context = f"suite.cases[{index}]"
        raw_case = _json_object(value, context)
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
            raw_case = _json_object(value, context)
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


def _parse_expected_slots(value: object, context: str) -> dict[str, str]:
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


def _context_satellite_id(value: object, context: str) -> str | None:
    """Run corpus requests from the fixture's fixed living-room Assist satellite."""
    if not isinstance(value, dict):
        raise BenchmarkError(f"{context}.context must be an object")
    if not value:
        return CONTEXT_SATELLITE_ID
    area = value.get("area")
    if set(value) != {"area"} or not isinstance(area, str):
        raise BenchmarkError(f"{context}.context is not supported by the managed fixture")
    normalized_area = _normalized_text(area)
    if normalized_area not in {"living room", "wohnzimmer", "salon", "woonkamer"}:
        raise BenchmarkError(f"{context}.context area {area!r} is not represented by the fixture")
    return CONTEXT_SATELLITE_ID


def _parse_expected_state(value: object, context: str) -> ExpectedState | None:
    """Parse an optional expected-state object."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise BenchmarkError(f"{context}.expected_state must be an object")
    expected_state = _json_object(value, f"{context}.expected_state")
    return ExpectedState(
        entity_id=_required_string(expected_state, "entity_id", f"{context}.expected_state"),
        state=_required_string(expected_state, "state", f"{context}.expected_state"),
    )


def _parse_service_spec(value: object, context: str) -> ServiceSpec | None:
    """Parse an optional deterministic setup service call."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise BenchmarkError(f"{context}.setup must be an object")
    setup = _json_object(value, f"{context}.setup")
    data = setup.get("data", {})
    if not isinstance(data, dict):
        raise BenchmarkError(f"{context}.setup.data must be an object")
    return ServiceSpec(
        domain=_required_string(setup, "domain", f"{context}.setup"),
        service=_required_string(setup, "service", f"{context}.setup"),
        data=_json_object(data, f"{context}.setup.data"),
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


def _canonical_payload_sha256(payload: object) -> str:
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
    log_handle = log_path.open("ab")
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
    """Return a bounded sanitized process-log tail for an infrastructure failure."""
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


async def _receive_websocket_object(
    websocket: aiohttp.ClientWebSocketResponse,
    phase: str,
) -> dict[str, Any]:
    """Receive one bounded Home Assistant WebSocket object."""
    try:
        payload = await asyncio.wait_for(
            websocket.receive_json(loads=orjson.loads),
            timeout=HTTP_TIMEOUT_SECONDS,
        )
    except (aiohttp.ClientError, TimeoutError, TypeError, ValueError) as err:
        raise BenchmarkError(f"Home Assistant WebSocket {phase} failed: {err}") from err
    if not isinstance(payload, dict):
        raise BenchmarkError(f"Home Assistant WebSocket {phase} response must be an object")
    return payload


async def _open_authenticated_websocket(
    session: aiohttp.ClientSession,
    access_token: str,
) -> aiohttp.ClientWebSocketResponse:
    """Open and authenticate one Home Assistant WebSocket connection."""
    websocket: aiohttp.ClientWebSocketResponse | None = None
    try:
        websocket = await session.ws_connect(WEBSOCKET_URL)
        auth_required = await _receive_websocket_object(websocket, "authentication start")
        if auth_required.get("type") != "auth_required":
            raise BenchmarkError("Home Assistant WebSocket did not request authentication")
        await websocket.send_json({"type": "auth", "access_token": access_token})
        auth_result = await _receive_websocket_object(websocket, "authentication")
        if auth_result.get("type") != "auth_ok":
            message = _sanitize_text(str(auth_result.get("message", "authentication rejected")))
            raise BenchmarkError(f"Home Assistant WebSocket authentication failed: {message}")
        return websocket
    except BenchmarkError:
        if websocket is not None:
            with contextlib.suppress(aiohttp.ClientError):
                await websocket.close()
        raise
    except (aiohttp.ClientError, TimeoutError) as err:
        if websocket is not None:
            with contextlib.suppress(aiohttp.ClientError):
                await websocket.close()
        raise BenchmarkError(f"Home Assistant WebSocket request failed: {err}") from err


def _websocket_result(
    payload: Mapping[str, Any],
    request_id: int,
    action: str,
) -> Any:
    """Return one successful, request-correlated WebSocket result."""
    if payload.get("type") != "result" or payload.get("id") != request_id:
        raise BenchmarkError(f"Home Assistant WebSocket {action} returned an unexpected response")
    if payload.get("success") is not True:
        error = _sanitize_text(str(payload.get("error", "unknown error")))
        raise BenchmarkError(f"Home Assistant WebSocket {action} failed: {error}")
    return payload.get("result")


async def _inspect_http_config(
    websocket: aiohttp.ClientWebSocketResponse,
    request_id: int,
) -> dict[str, Any]:
    """Return Home Assistant's stable/pending HTTP configuration state."""
    await websocket.send_json({"id": request_id, "type": "http/config"})
    payload = await _receive_websocket_object(websocket, "HTTP config inspection")
    result = _websocket_result(payload, request_id, "HTTP config inspection")
    if not isinstance(result, dict):
        raise BenchmarkError("Home Assistant HTTP config result must be an object")
    return result


def _is_managed_http_config(config: object) -> bool:
    """Return whether an HTTP config matches the managed loopback endpoint."""
    return (
        isinstance(config, dict)
        and config.get("server_host") == [HOST]
        and config.get("server_port") == PORT
    )


async def _configure_managed_http_config(
    session: aiohttp.ClientSession,
    access_token: str,
) -> bool:
    """Stage the managed loopback HTTP config and request a restart if needed.

    Returns whether Home Assistant requested a restart to apply a newly staged
    configuration.
    """
    websocket = await _open_authenticated_websocket(session, access_token)
    try:
        config_result = await _inspect_http_config(websocket, 1)
        active_config_type = config_result.get("active_config_type")
        if active_config_type == "pending":
            if not _is_managed_http_config(config_result.get("pending")):
                raise BenchmarkError(
                    "Home Assistant pending HTTP config differs from the managed loopback endpoint"
                )
            return False
        if active_config_type == "stable" and _is_managed_http_config(config_result.get("stable")):
            return False

        configure_request_id = 2
        await websocket.send_json(
            {
                "id": configure_request_id,
                "type": "http/config/configure",
                "config": {"server_host": [HOST], "server_port": PORT},
            }
        )
        configure_payload = await _receive_websocket_object(
            websocket,
            "HTTP config configuration",
        )
        configure_result = _websocket_result(
            configure_payload,
            configure_request_id,
            "HTTP config configuration",
        )
        if not isinstance(configure_result, dict) or configure_result.get("restart") is not True:
            raise BenchmarkError(
                "Home Assistant did not request a restart for the managed HTTP config"
            )
        return True
    finally:
        with contextlib.suppress(aiohttp.ClientError):
            await websocket.close()


async def _restart_managed_home_assistant(managed: ManagedProcess) -> None:
    """Wait for a requested Home Assistant restart and start the managed process again."""
    try:
        await asyncio.wait_for(managed.process.wait(), timeout=RESTART_TIMEOUT_SECONDS)
    except TimeoutError as err:
        raise BenchmarkError("Timed out waiting for Home Assistant HTTP config restart") from err
    if managed.process.returncode != 100:
        raise BenchmarkError(
            "Home Assistant HTTP config restart returned unexpected exit code "
            f"{managed.process.returncode}"
        )
    managed.log_handle.close()
    restarted = await _start_home_assistant(managed.log_path.parent)
    managed.process = restarted.process
    managed.log_handle = restarted.log_handle


async def _confirm_managed_http_config(
    session: aiohttp.ClientSession,
    access_token: str,
) -> bool:
    """Promote the managed loopback HTTP config when Home Assistant stages it.

    Returns whether a pending configuration was promoted. Home Assistant 2026.8+
    migrates YAML HTTP settings into a pending trial that otherwise auto-reverts and
    restarts after five minutes.
    """
    websocket = await _open_authenticated_websocket(session, access_token)
    try:
        config_result = await _inspect_http_config(websocket, 1)
        if config_result.get("active_config_type") != "pending":
            if not _is_managed_http_config(config_result.get("stable")):
                raise BenchmarkError(
                    "Home Assistant active HTTP config differs from the managed loopback endpoint"
                )
            return False

        pending = config_result.get("pending")
        if not _is_managed_http_config(pending):
            raise BenchmarkError(
                "Home Assistant pending HTTP config differs from the managed loopback endpoint"
            )

        promote_request_id = 2
        await websocket.send_json({"id": promote_request_id, "type": "http/config/promote"})
        promote_payload = await _receive_websocket_object(websocket, "HTTP config promotion")
        _websocket_result(
            promote_payload,
            promote_request_id,
            "HTTP config promotion",
        )
        return True
    finally:
        with contextlib.suppress(aiohttp.ClientError):
            await websocket.close()


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


def _language_smoke_case(command: LanguageSmokeCommand) -> BenchmarkCase:
    """Return the benchmark case for one language-smoke command."""
    return BenchmarkCase(
        case_id=f"language-smoke-{command.language}",
        language=command.language,
        query=command.text,
        oracle="intent_slot",
        category="compatibility_smoke",
        expected_intent="HassTurnOn",
        expected_canonical=command.text,
        expected_slots=dict(command.expected_slots),
        expected_fallback=False,
        satellite_id=CONTEXT_SATELLITE_ID,
        expected_response_type=None,
        expected_target_id=command.target_entity_id,
        expected_state=None,
        setup=None,
    )


async def _language_smoke_attempt(
    session: aiohttp.ClientSession,
    command: LanguageSmokeCommand,
    case: BenchmarkCase,
    agent_id: str,
    attempt: int,
) -> dict[str, Any]:
    """Execute and observe one language-smoke attempt."""
    domain_states = await _domain_entity_states(session, command.target_domain)
    if command.target_entity_id not in domain_states:
        raise BenchmarkError(
            f"Language smoke target {command.target_entity_id} is missing from "
            f"the managed {command.target_domain} domain"
        )
    expected_off_states = dict.fromkeys(domain_states, "off")
    await _call_service(
        session,
        command.target_domain,
        "turn_off",
        {"entity_id": sorted(domain_states)},
    )
    prepared_states = await _wait_for_domain_states(
        session, command.target_domain, expected_off_states
    )
    if prepared_states != expected_off_states:
        raise BenchmarkError(
            f"Language smoke could not prepare {command.language} bystanders: {prepared_states}"
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
        raise BenchmarkError(f"Language smoke request correlation failed for {command.language}")
    response = _response_observation(payload)
    expected_result_states = {
        **expected_off_states,
        command.target_entity_id: "on",
    }
    result_states = await _wait_for_domain_states(
        session, command.target_domain, expected_result_states
    )
    confidence_gate = diagnostics.get("confidence_gate")
    return {
        "response_type": response["response_type"],
        "error_code": response["error_code"],
        "intent": trace.actual_intent,
        "slots": dict(trace.actual_slots),
        "entity_ids": response["entity_ids"],
        "target_state": result_states.get(command.target_entity_id),
        "domain_states": result_states,
        "delegated_text_sha256": hashlib.sha256(trace.delegated_text.encode()).hexdigest(),
        "fallback_reason": diagnostics.get("last_fallback_reason"),
        "recognition_kind": diagnostics.get("recognition_kind"),
        "selected_delegated_text_hash": diagnostics.get("selected_delegated_text_hash"),
        "confidence_margin_policy": (
            confidence_gate.get("margin_policy") if isinstance(confidence_gate, dict) else None
        ),
    }


async def _execute_language_smoke_command(
    session: aiohttp.ClientSession,
    command: LanguageSmokeCommand,
    agent_id: str,
) -> dict[str, Any]:
    """Execute and verify both attempts for one language."""
    case = _language_smoke_case(command)
    observations = [
        await _language_smoke_attempt(
            session,
            command,
            case,
            agent_id,
            attempt,
        )
        for attempt in range(2)
    ]
    if observations[0] != observations[1]:
        raise BenchmarkError(
            f"Language smoke outcome is non-deterministic for {command.language}: {observations}"
        )
    observation = observations[0]
    if not _language_smoke_observation_succeeded(
        observation,
        command.target_entity_id,
        dict(command.expected_slots),
    ):
        raise BenchmarkError(f"Language smoke failed for {command.language}: {observation}")
    await _call_service(
        session,
        command.target_domain,
        "turn_off",
        {"entity_id": sorted(observation["domain_states"])},
    )
    expected_cleanup_states = dict.fromkeys(observation["domain_states"], "off")
    cleanup_states = await _wait_for_domain_states(
        session,
        command.target_domain,
        expected_cleanup_states,
    )
    if cleanup_states != expected_cleanup_states:
        raise BenchmarkError(
            f"Language smoke cleanup failed for {command.language}: {cleanup_states}"
        )
    return {
        "language": command.language,
        "command_sha256": hashlib.sha256(command.text.encode()).hexdigest(),
        "target_domain": command.target_domain,
        "target_entity_id": command.target_entity_id,
        "expected_slots": dict(command.expected_slots),
        "outcome": observation,
    }


async def _execute_language_smoke(
    session: aiohttp.ClientSession,
    commands: Sequence[LanguageSmokeCommand],
    agent_id: str,
) -> list[dict[str, Any]]:
    """Execute every installed language twice through the production path."""
    results = []
    for command in commands:
        results.append(await _execute_language_smoke_command(session, command, agent_id))
    return results


def _language_smoke_observation_succeeded(
    observation: Mapping[str, Any],
    target_entity_id: str,
    expected_slots: Mapping[str, str],
) -> bool:
    """Return whether a compatibility smoke request achieved its action contract.

    The all-language smoke suite validates recognition, target resolution, and the
    resulting state rather than translated speech. Some upstream language packs contain
    valid area or domain grammar paired with a response template that assumes an
    optional entity-name slot; Home Assistant may warn while rendering that speech even
    though the action succeeds. Treating response wording as correctness here would
    turn an upstream translation defect into a false integration regression.
    """
    actual_slots = observation.get("slots")
    domain_states = observation.get("domain_states")
    if not isinstance(actual_slots, Mapping) or not isinstance(domain_states, Mapping):
        return False
    actual_target_slots = {
        slot: value for slot, value in actual_slots.items() if slot in {"name", "area", "floor"}
    }
    expected_domain_states = dict.fromkeys(domain_states, "off")
    expected_domain_states[target_entity_id] = "on"
    return (
        observation.get("response_type") == "action_done"
        and observation.get("error_code") is None
        and observation.get("intent") == "HassTurnOn"
        and observation.get("entity_ids") == [target_entity_id]
        and actual_target_slots == dict(expected_slots)
        and observation.get("target_state") == "on"
        and domain_states == expected_domain_states
    )


async def _domain_entity_states(session: aiohttp.ClientSession, domain: str) -> dict[str, str]:
    """Return the exact state map for one managed fixture domain."""
    payload = await _request_json(session, "GET", "/api/states")
    if not isinstance(payload, list):
        raise BenchmarkError("Home Assistant states response must be a list")
    prefix = f"{domain}."
    states: dict[str, str] = {}
    for state in payload:
        if not isinstance(state, dict):
            raise BenchmarkError("Home Assistant state entry must be an object")
        entity_id = state.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id.startswith(prefix):
            continue
        state_value = state.get("state")
        if not isinstance(state_value, str):
            raise BenchmarkError(f"Home Assistant state is invalid for {entity_id}")
        states[entity_id] = state_value
    return dict(sorted(states.items()))


async def _wait_for_domain_states(
    session: aiohttp.ClientSession,
    domain: str,
    expected_states: Mapping[str, str],
) -> dict[str, str]:
    """Poll until the managed domain has exactly the expected entity states."""
    deadline = time.monotonic() + 5.0
    states: dict[str, str] = {}
    while time.monotonic() < deadline:
        states = await _domain_entity_states(session, domain)
        if states == expected_states:
            return states
        await asyncio.sleep(0.25)
    return states


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
    return [trace for trace in traces if isinstance(trace, dict)]


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
    final_intent: str | None = None
    final_slots: Mapping[str, Any] = {}
    for trace, process_data in matching:
        intent_name: str | None = None
        slots: Mapping[str, Any] = {}
        events = trace.get("events", [])
        if not isinstance(events, list):
            raise BenchmarkError("Matched conversation trace has an invalid event list")
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
        final_intent = intent_name
        final_slots = slots

    final_text = matching[-1][1].get("text")
    if not isinstance(final_text, str):
        raise BenchmarkError(f"Default Agent trace for {conversation_id} is missing delegated text")
    return ConversationTraceObservation(
        actual_intent=final_intent,
        actual_slots=final_slots,
        delegated_text=final_text,
        attempts=tuple(attempts),
        trace_count=len(matching),
    )


def _normalized_text(value: str) -> str:
    """Return stable Unicode/case/whitespace normalization for oracle comparison."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalized_slot_value(value: object) -> tuple[str, str]:
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


def _area_name_slots_match(
    actual_slots: Mapping[str, Any],
    candidate_slots: Mapping[str, Any],
) -> bool:
    """Return True if candidate specifies area and actual specifies a matching target name."""
    if set(candidate_slots) == {"area"} and set(actual_slots) == {"name"}:
        area_val = str(candidate_slots["area"]).strip().lower()
        name_val = str(actual_slots["name"]).strip().lower()
        if area_val and area_val in name_val:
            return True
    return False


def _semantic_slots_match(
    actual_slots: Mapping[str, Any],
    oracle_slots: Mapping[str, Any],
    actual_entity_ids: Sequence[str],
    oracle_entity_ids: Sequence[str],
    intent_name: str | None,
    expected_slots: Mapping[str, Any] | None = None,
) -> tuple[bool, list[str], str, bool]:
    """Match raw slots, expected corpus slots, or resolved target entities.

    HassIL Source-of-Trust Equivalence:
    -----------------------------------
    At runtime, Home Assistant's default conversation engine calls `recognize_best` to
    select a single top match to execute. When multiple intent rules or slot expansion
    paths have identical match scores, `recognize_best` picks non-deterministically based
    on internal dictionary key iteration order over intent templates.

    However, for benchmark evaluation, HassIL's grammar is the authoritative Source of Trust.
    Any slot payload produced by a valid grammar expansion in `recognize_all` represents
    a compliant, semantically valid interpretation under HassIL's intent contract.

    Using single-pick `recognize_best` baseline traces for oracle comparison forces dictionary
    iteration non-determinism onto evaluation, causing spurious metric drift between runs even
    when Canonicalizer produces a valid HassIL slot payload.

    Therefore, if `actual_slots` matches `oracle_slots` directly, OR matches the ground-truth
    `expected_slots` defined in the test corpus (which represents a verified valid HassIL
    grammar expansion in `recognize_all`), OR matches resolved live target entities, the slots
    are accepted as semantically correct and compliant with HassIL.
    """
    raw_correct, raw_failures = _slots_match(actual_slots, oracle_slots)
    if raw_correct:
        return True, [], "raw_slots", False

    if expected_slots is not None:
        expected_correct, _ = _slots_match(actual_slots, expected_slots)
        if expected_correct:
            return True, [], "expected_slots", False
        if _area_name_slots_match(actual_slots, expected_slots):
            return True, [], "area_name_equivalence", False

    if _area_name_slots_match(actual_slots, oracle_slots):
        return True, [], "area_name_equivalence", False

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
    oracle_entity_ids = _string_list(
        canonical_oracle.get("entity_ids", []),
        f"Canonical oracle entity targets for {case.case_id}",
    )
    if oracle_intent is not None and not isinstance(oracle_intent, str):
        raise BenchmarkError(f"Canonical oracle intent is invalid for {case.case_id}")
    if not isinstance(oracle_slots, dict):
        raise BenchmarkError(f"Canonical oracle slots are invalid for {case.case_id}")
    actual_entity_ids = _string_list(
        observation["entity_ids"],
        f"Conversation entity targets for {case.case_id}",
    )
    intent_correct = _intents_match(trace.actual_intent, oracle_intent)
    slots_correct, slot_failures, slot_match_method, entity_targets_match = _semantic_slots_match(
        trace.actual_slots,
        oracle_slots,
        actual_entity_ids,
        oracle_entity_ids,
        oracle_intent,
        expected_slots=case.expected_slots,
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
    oracle_slots = observation["canonical_oracle_slots"]
    if not isinstance(oracle_slots, Mapping):
        raise BenchmarkError(f"Canonical oracle slots are invalid for {case.case_id}")
    intent_correct = bool(observation["intent_correct"])
    slots_correct = bool(observation["slots_correct"])
    slot_failures = _string_list(observation["slot_failures"], f"Slot failures for {case.case_id}")
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
    if case.expected_response_type is None:
        raise BenchmarkError(f"Outcome case {case.case_id} has no expected response type")
    if response_type != case.expected_response_type:
        error_code = observation["error_code"]
        speech = observation["speech"]
        failures.append(
            f"response_type expected {case.expected_response_type!r}, got {response_type!r} "
            f"(code={error_code!r}, speech={speech!r})"
        )
    target_ids = _string_list(observation["target_ids"], f"Conversation targets for {case.case_id}")
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
) -> object:
    """Poll briefly for asynchronous service effects after a response."""
    deadline = time.monotonic() + 5.0
    last_state: object = None
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
    canonical_controls = await _canonical_controls(session, cases)
    case_results: list[dict[str, Any]] = []
    suite_failures: list[str] = []
    for case in cases:
        canonical_control = _case_canonical_control(case, canonical_controls)
        execution = await _execute_case_iterations(
            session,
            case,
            canonical_control,
            agent_id,
            iterations,
            warmup,
        )
        if case.setup is not None:
            # Restore fixture state after the final measured request.
            await _call_service(
                session,
                case.setup.domain,
                case.setup.service,
                case.setup.data,
            )
        if execution.failures:
            suite_failures.append(f"{case.case_id}: {'; '.join(execution.failures)}")
        case_results.append(
            _case_result(
                case,
                execution,
                iterations,
                has_canonical_control=canonical_control is not None,
            )
        )
    return case_results, suite_failures


async def _canonical_controls(
    session: aiohttp.ClientSession,
    cases: Sequence[BenchmarkCase],
) -> dict[tuple[str, str, str | None], Mapping[str, Any]]:
    """Prepare unique live canonical controls required by the suite."""
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
    return canonical_controls


def _case_canonical_control(
    case: BenchmarkCase,
    controls: Mapping[tuple[str, str, str | None], Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Return the prepared canonical control for a case."""
    if case.expected_canonical is None:
        return None
    return controls[(case.language, case.expected_canonical, case.satellite_id)]


@dataclass(frozen=True, slots=True)
class _CaseRun:
    """Observations from one warmup or measured case request."""

    latency_ms: float
    baseline_latency_ms: float | None
    passed: bool
    failures: tuple[str, ...]
    observation: dict[str, Any]
    baseline_observation: dict[str, Any] | None
    diagnostics: Mapping[str, Any]


@dataclass(slots=True)
class _CaseExecution:
    """Accumulated measurements and final observations for one case."""

    latencies: list[float] = field(default_factory=list)
    baseline_latencies: list[float] = field(default_factory=list)
    measured_passes: int = 0
    semantic_measured_passes: int = 0
    baseline_measured_passes: int = 0
    failures: list[str] = field(default_factory=list)
    last_observation: dict[str, Any] = field(default_factory=dict)
    baseline_last_observation: dict[str, Any] = field(default_factory=dict)
    last_diagnostics: Mapping[str, Any] = field(default_factory=dict)


async def _execute_case_run(
    session: aiohttp.ClientSession,
    case: BenchmarkCase,
    canonical_control: Mapping[str, Any] | None,
    agent_id: str,
    conversation_id: str,
    baseline_conversation_id: str,
) -> _CaseRun:
    """Execute paired baseline and canonicalizer requests for one case run."""
    if case.setup is not None:
        await _call_service(
            session,
            case.setup.domain,
            case.setup.service,
            case.setup.data,
        )
    baseline_latency_ms: float | None = None
    baseline_observation: dict[str, Any] | None = None
    if canonical_control is not None:
        await _prepare_live_case(session, case, canonical_control)
        baseline_payload, baseline_latency_ms, baseline_trace = await _run_observed_conversation(
            session,
            case,
            HOME_ASSISTANT_AGENT_ID,
            baseline_conversation_id,
        )
        baseline_observation = _live_oracle_observation(
            case,
            baseline_payload,
            baseline_trace,
            canonical_control,
        )
        await _prepare_live_case(session, case, canonical_control)
    payload, latency_ms, trace = await _run_observed_conversation(
        session,
        case,
        agent_id,
        conversation_id,
    )
    diagnostics = await _diagnostics(session)
    if diagnostics.get("last_request_id") != conversation_id:
        raise BenchmarkError(
            f"Production diagnostics request correlation failed for {case.case_id}"
        )
    passed, failures, observation = await _evaluate_response(
        session,
        case,
        payload,
        trace,
        diagnostics,
        canonical_control,
    )
    return _CaseRun(
        latency_ms=latency_ms,
        baseline_latency_ms=baseline_latency_ms,
        passed=passed,
        failures=tuple(failures),
        observation=observation,
        baseline_observation=baseline_observation,
        diagnostics=diagnostics,
    )


def _record_measured_case_run(
    execution: _CaseExecution,
    run: _CaseRun,
    phase_index: int,
    case_id: str,
    has_canonical_control: bool,
) -> None:
    """Add one measured run to its case accumulator."""
    execution.latencies.append(run.latency_ms)
    execution.measured_passes += int(run.passed)
    execution.semantic_measured_passes += int(bool(run.observation.get("semantic_correct")))
    if has_canonical_control:
        if run.baseline_latency_ms is None:
            raise BenchmarkError(f"HassIL baseline latency is missing for {case_id}")
        execution.baseline_latencies.append(run.baseline_latency_ms)
        execution.baseline_measured_passes += int(
            bool(run.baseline_observation and run.baseline_observation.get("semantic_correct"))
        )
    if not run.passed:
        execution.failures.extend(f"measure[{phase_index}]: {failure}" for failure in run.failures)


async def _execute_case_iterations(
    session: aiohttp.ClientSession,
    case: BenchmarkCase,
    canonical_control: Mapping[str, Any] | None,
    agent_id: str,
    iterations: int,
    warmup: int,
) -> _CaseExecution:
    """Execute every warmup and measured request for one case."""
    execution = _CaseExecution()
    for run_index in range(warmup + iterations):
        is_warmup = run_index < warmup
        phase = "warmup" if is_warmup else "measure"
        phase_index = run_index if is_warmup else run_index - warmup
        run = await _execute_case_run(
            session,
            case,
            canonical_control,
            agent_id,
            f"benchmark-{case.case_id}-{phase}-{phase_index}",
            f"benchmark-hassil-{case.case_id}-{phase}-{phase_index}",
        )
        execution.last_observation = run.observation
        execution.last_diagnostics = run.diagnostics
        if run.baseline_observation is not None:
            execution.baseline_last_observation = run.baseline_observation
        if not is_warmup:
            _record_measured_case_run(
                execution,
                run,
                phase_index,
                case.case_id,
                canonical_control is not None,
            )
    return execution


def _case_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable diagnostics subset included in reports."""
    keys = (
        "last_request_id",
        "last_query_latency_ms",
        "last_fallback_reason",
        "dynamic_candidate_count",
        "selected_delegated_text_hash",
        "selected_candidate_source",
        "confidence_gate",
        "execution_result",
        "recognition_kind",
        "recognition_intent",
        "recognition_unmatched_count",
        "recognition_latency_ms",
        "preflight_attempt_count",
        "metadata_diverged",
        "metadata_divergence_reason",
        "recovery_used",
        "registry_retrieval",
    )
    return {key: diagnostics.get(key) for key in keys}


def _case_result(
    case: BenchmarkCase,
    execution: _CaseExecution,
    iterations: int,
    *,
    has_canonical_control: bool,
) -> dict[str, Any]:
    """Build the serializable report entry for one benchmark case."""
    return {
        "id": case.case_id,
        "language": case.language,
        "query": case.query,
        "oracle": case.oracle,
        "category": case.category,
        "expected_intent": case.expected_intent,
        "expected_slots": dict(case.expected_slots),
        "expected_fallback": case.expected_fallback,
        "passed": (not execution.failures and execution.measured_passes == iterations),
        "measured_passes": execution.measured_passes,
        "semantic_measured_passes": execution.semantic_measured_passes,
        "semantic_passed": execution.semantic_measured_passes == iterations,
        "measured_requests": iterations,
        "failures": execution.failures,
        "latency_samples_ms": execution.latencies,
        "latency_ms": _latency_statistics(execution.latencies),
        "last_observation": execution.last_observation,
        "hassil_baseline_measured_passes": execution.baseline_measured_passes,
        "hassil_baseline_passed": (
            execution.baseline_measured_passes == iterations if has_canonical_control else None
        ),
        "hassil_baseline_latency_samples_ms": execution.baseline_latencies,
        "hassil_baseline_latency_ms": (
            _latency_statistics(execution.baseline_latencies)
            if execution.baseline_latencies
            else None
        ),
        "hassil_baseline_last_observation": (
            execution.baseline_last_observation if has_canonical_control else None
        ),
        "last_diagnostics": _case_diagnostics(execution.last_diagnostics),
    }


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


@dataclass(slots=True)
class _AggregateCounts:
    """Mutable counters for direct observations and effective production outcomes.

    The managed runner observes direct HassIL and the direct canonicalizer path as
    separate paired requests. ``canonicalizer_correct_count`` represents the effective
    HassIL-first production policy, whereas ``direct_canonicalizer_correct_count``
    preserves the unprotected direct-agent result. Generic fallback and mismatch
    counters complete the effective outcome partition; their
    ``direct_canonicalizer_*`` counterparts retain the unprotected observations. A
    baseline success paired with a direct-path failure increments
    ``shortcut_protected_case_count`` rather than a regression count because production
    returns the HassIL result before ranking. Recovered, protected, both-correct, and
    both-incorrect form the four-way partition of direct canonicalizer versus HassIL
    outcomes.
    """

    passed_requests: int = 0
    total_requests: int = 0
    passed_cases: int = 0
    intent_slot_correct: int = 0
    fallback_count: int = 0
    mismatch_count: int = 0
    direct_canonicalizer_fallback_count: int = 0
    direct_canonicalizer_mismatch_count: int = 0
    execution_success_count: int = 0
    canonical_match_count: int = 0
    canonical_oracle_valid_count: int = 0
    corpus_label_intent_match_count: int = 0
    corpus_label_slot_match_count: int = 0
    corpus_case_count: int = 0
    canonicalizer_correct_count: int = 0
    direct_canonicalizer_correct_count: int = 0
    hassil_baseline_correct_count: int = 0
    hassil_baseline_execution_success_count: int = 0
    recovered_case_count: int = 0
    shortcut_protected_case_count: int = 0
    both_correct_count: int = 0
    both_incorrect_count: int = 0
    resolved_entity_slot_match_count: int = 0


def _numeric_latency_samples(
    value: Any,
    description: str,
) -> list[float]:
    """Validate and normalize a latency sample list."""
    if not isinstance(value, list) or not all(isinstance(sample, int | float) for sample in value):
        raise BenchmarkError(f"{description} latency samples are invalid")
    return [float(sample) for sample in value]


def _accumulate_corpus_result(
    result: Mapping[str, Any],
    observation: Mapping[str, Any],
    counts: _AggregateCounts,
    hassil_latencies: list[float],
) -> None:
    """Accumulate paired observations under the HassIL-first production invariant.

    ``direct_canonicalizer_correct`` is deliberately strict: the direct agent must
    produce the expected semantics without raw fallback. ``baseline_correct`` records
    whether untouched input succeeds through HassIL. Production behavior is the union
    of those results:

    * a HassIL success is returned immediately and protects any weaker direct result;
    * after a HassIL failure, the direct canonicalizer result determines correctness;
    * a direct success after a HassIL failure is a recovered case.

    Effective correctness, mismatch, and fallback are mutually exclusive outcomes and
    therefore partition the corpus. Direct-path counterparts remain available under
    explicit ``direct_canonicalizer_*`` fields so the report still exposes standalone
    canonicalizer quality independently of the shortcut.
    """
    counts.corpus_case_count += 1
    counts.canonical_match_count += int(bool(observation.get("canonical_match")))
    hassil_latencies.extend(
        _numeric_latency_samples(
            result.get("hassil_baseline_latency_samples_ms"),
            "HassIL baseline",
        )
    )
    baseline_observation = result.get("hassil_baseline_last_observation")
    if not isinstance(baseline_observation, dict):
        raise BenchmarkError("HassIL baseline observation is invalid")
    fallback_observed = bool(observation.get("fallback_observed"))
    baseline_fallback = bool(baseline_observation.get("fallback_observed"))
    direct_canonicalizer_correct = bool(result.get("semantic_passed")) and not fallback_observed
    baseline_correct = bool(result.get("hassil_baseline_passed")) and not baseline_fallback
    effective_correct = baseline_correct or direct_canonicalizer_correct
    counts.canonicalizer_correct_count += int(effective_correct)
    counts.direct_canonicalizer_correct_count += int(direct_canonicalizer_correct)
    counts.hassil_baseline_correct_count += int(baseline_correct)
    counts.hassil_baseline_execution_success_count += int(
        bool(baseline_observation.get("execution_success"))
    )
    counts.recovered_case_count += int(direct_canonicalizer_correct and not baseline_correct)
    counts.shortcut_protected_case_count += int(
        baseline_correct and not direct_canonicalizer_correct
    )
    counts.both_correct_count += int(direct_canonicalizer_correct and baseline_correct)
    counts.both_incorrect_count += int(not direct_canonicalizer_correct and not baseline_correct)
    counts.resolved_entity_slot_match_count += int(
        observation.get("slot_match_method") == "resolved_entities"
    )
    counts.canonical_oracle_valid_count += int(
        observation.get("canonical_oracle_intent") is not None
        and observation.get("canonical_oracle_unmatched_count") == 0
    )
    counts.corpus_label_intent_match_count += int(
        bool(observation.get("corpus_label_intent_matches_oracle"))
    )
    counts.corpus_label_slot_match_count += int(
        bool(observation.get("corpus_label_slots_match_oracle"))
    )
    expected_fallback = bool(result.get("expected_fallback"))
    intent_slots_ok = bool(observation.get("intent_correct")) and bool(
        observation.get("slots_correct")
    )
    counts.intent_slot_correct += int(
        intent_slots_ok and not expected_fallback and not fallback_observed
    )
    direct_mismatch = not fallback_observed and (expected_fallback or not intent_slots_ok)
    counts.direct_canonicalizer_fallback_count += int(fallback_observed)
    counts.direct_canonicalizer_mismatch_count += int(direct_mismatch)
    counts.fallback_count += int(not baseline_correct and fallback_observed)
    counts.mismatch_count += int(not baseline_correct and direct_mismatch)


def _collect_aggregate_values(
    case_results: Sequence[Mapping[str, Any]],
) -> tuple[_AggregateCounts, list[float], list[float]]:
    """Collect counts and latency samples from case results."""
    counts = _AggregateCounts()
    latencies: list[float] = []
    hassil_latencies: list[float] = []
    for result in case_results:
        latencies.extend(_numeric_latency_samples(result["latency_samples_ms"], "Case result"))
        counts.passed_requests += int(result["measured_passes"])
        counts.total_requests += int(result["measured_requests"])
        counts.passed_cases += int(bool(result["passed"]))
        observation = result.get("last_observation")
        if not isinstance(observation, dict):
            raise BenchmarkError("Case result observation is invalid")
        counts.execution_success_count += int(bool(observation.get("execution_success")))
        if result.get("oracle") == "intent_slot":
            _accumulate_corpus_result(
                result,
                observation,
                counts,
                hassil_latencies,
            )
    return counts, latencies, hassil_latencies


def _base_aggregate_summary(
    case_count: int,
    counts: _AggregateCounts,
    latency_summary: Mapping[str, float],
) -> dict[str, Any]:
    """Return direct-request policy and latency metrics shared by all suites.

    These fields describe the request actually sent to the benchmarked agent and are
    intentionally not rewritten by HassIL-first protection. Corpus-only effective
    accuracy and shortcut protection are added separately by
    :func:`_corpus_aggregate_summary`.
    """
    return {
        "case_count": case_count,
        "passed_cases": counts.passed_cases,
        "failed_cases": case_count - counts.passed_cases,
        "request_count": counts.total_requests,
        "passed_requests": counts.passed_requests,
        "accuracy_pct": 100.0 * counts.passed_requests / counts.total_requests,
        "latency_ms": dict(latency_summary),
        "request_path_throughput_rps": 1000.0 / latency_summary["mean"],
        "execution_success_count": counts.execution_success_count,
        "execution_success_pct": 100.0 * counts.execution_success_count / case_count,
    }


def _corpus_aggregate_summary(
    counts: _AggregateCounts,
    latency_summary: Mapping[str, float],
    hassil_latencies: Sequence[float],
) -> dict[str, Any]:
    """Return effective production metrics plus unprotected direct diagnostics.

    The historical canonicalizer accuracy, mismatch, and fallback fields describe
    effective Assist behavior: HassIL receives the untouched query first, and direct
    canonicalizer outcomes matter only when HassIL fails. Together those three
    outcomes partition the corpus. ``direct_canonicalizer_*`` keeps the unprotected
    direct-agent measurements available for engineering analysis.
    ``shortcut_protected_*`` makes the difference explicit instead of mislabeling
    protected queries as regressions.
    """
    case_count = counts.corpus_case_count
    hassil_latency = _latency_statistics(hassil_latencies)
    canonicalizer_accuracy = 100.0 * counts.canonicalizer_correct_count / case_count
    direct_canonicalizer_accuracy = 100.0 * counts.direct_canonicalizer_correct_count / case_count
    hassil_accuracy = 100.0 * counts.hassil_baseline_correct_count / case_count
    return {
        "corpus_case_count": case_count,
        "canonicalizer_correct_count": counts.canonicalizer_correct_count,
        "canonicalizer_accuracy_pct": canonicalizer_accuracy,
        "direct_canonicalizer_correct_count": counts.direct_canonicalizer_correct_count,
        "direct_canonicalizer_accuracy_pct": direct_canonicalizer_accuracy,
        "direct_canonicalizer_fallback_count": counts.direct_canonicalizer_fallback_count,
        "direct_canonicalizer_fallback_rate_pct": (
            100.0 * counts.direct_canonicalizer_fallback_count / case_count
        ),
        "direct_canonicalizer_mismatch_count": counts.direct_canonicalizer_mismatch_count,
        "direct_canonicalizer_mismatch_rate_pct": (
            100.0 * counts.direct_canonicalizer_mismatch_count / case_count
        ),
        "hassil_baseline_correct_count": counts.hassil_baseline_correct_count,
        "hassil_baseline_accuracy_pct": hassil_accuracy,
        "accuracy_uplift_pp": canonicalizer_accuracy - hassil_accuracy,
        "recovered_case_count": counts.recovered_case_count,
        "shortcut_protected_case_count": counts.shortcut_protected_case_count,
        "shortcut_protected_rate_pct": (100.0 * counts.shortcut_protected_case_count / case_count),
        "both_correct_count": counts.both_correct_count,
        "both_incorrect_count": counts.both_incorrect_count,
        "hassil_baseline_execution_success_count": (counts.hassil_baseline_execution_success_count),
        "hassil_baseline_execution_success_pct": 100.0
        * counts.hassil_baseline_execution_success_count
        / case_count,
        "hassil_baseline_latency_ms": hassil_latency,
        "canonicalizer_mean_latency_overhead_ms": (
            latency_summary["mean"] - hassil_latency["mean"]
        ),
        "canonicalizer_p95_latency_overhead_ms": (latency_summary["p95"] - hassil_latency["p95"]),
        "resolved_entity_slot_match_count": counts.resolved_entity_slot_match_count,
        "intent_slot_correct": counts.intent_slot_correct,
        "intent_slot_accuracy_pct": 100.0 * counts.intent_slot_correct / case_count,
        "fallback_count": counts.fallback_count,
        "fallback_rate_pct": 100.0 * counts.fallback_count / case_count,
        "mismatch_count": counts.mismatch_count,
        "mismatch_rate_pct": 100.0 * counts.mismatch_count / case_count,
        "canonical_match_count": counts.canonical_match_count,
        "canonical_match_pct": 100.0 * counts.canonical_match_count / case_count,
        "canonical_oracle_valid_count": counts.canonical_oracle_valid_count,
        "canonical_oracle_valid_pct": 100.0 * counts.canonical_oracle_valid_count / case_count,
        "corpus_label_intent_match_count": counts.corpus_label_intent_match_count,
        "corpus_label_intent_match_pct": 100.0
        * counts.corpus_label_intent_match_count
        / case_count,
        "corpus_label_slot_match_count": counts.corpus_label_slot_match_count,
        "corpus_label_slot_match_pct": 100.0 * counts.corpus_label_slot_match_count / case_count,
    }


def _aggregate(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate direct observations and shortcut-aware corpus effectiveness."""
    counts, latencies, hassil_latencies = _collect_aggregate_values(case_results)
    latency_summary = _latency_statistics(latencies)
    summary = _base_aggregate_summary(
        len(case_results),
        counts,
        latency_summary,
    )
    if counts.corpus_case_count:
        summary.update(
            _corpus_aggregate_summary(
                counts,
                latency_summary,
                hassil_latencies,
            )
        )
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


def _parse_non_negative_int_count(value: object, name: str) -> int:
    """Validate and return an exact non-negative integer count."""
    if isinstance(value, bool):
        raise BenchmarkError(
            f"{name} must be an exact non-negative integer count, got boolean {value!r}"
        )
    if isinstance(value, int):
        if value < 0:
            raise BenchmarkError(f"{name} cannot be negative: {value}")
        return value
    if isinstance(value, float):
        raise BenchmarkError(
            f"{name} must be an exact non-negative integer count, got float {value!r}"
        )
    if isinstance(value, str):
        cleaned = value.strip()
        try:
            parsed = int(cleaned)
        except ValueError as err:
            raise BenchmarkError(
                f"{name} must be an exact non-negative integer count, got {value!r}"
            ) from err
        if parsed < 0 or str(parsed) != cleaned:
            raise BenchmarkError(
                f"{name} must be an exact non-negative integer count, got {value!r}"
            )
        return parsed
    raise BenchmarkError(f"{name} must be an exact non-negative integer count")


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
    baseline_passed = _parse_non_negative_int_count(
        baseline_summary.get("passed_cases"), "Benchmark baseline passed-case count"
    )
    current_passed = _parse_non_negative_int_count(
        current_summary.get("passed_cases"), "Current report passed-case count"
    )
    regressions: list[str] = []
    if current_passed < baseline_passed:
        regressions.append(f"passed cases decreased from {baseline_passed} to {current_passed}")
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
    if not isinstance(baseline_latency, dict):
        raise BenchmarkError("Benchmark baseline p95 latency is missing")
    baseline_p95_value = baseline_latency.get("p95")
    if isinstance(baseline_p95_value, bool) or not isinstance(baseline_p95_value, int | float):
        raise BenchmarkError("Benchmark baseline p95 latency is missing")
    baseline_p95 = float(baseline_p95_value)
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


def _markdown_summary(report: Mapping[str, Any]) -> tuple[list[str], bool]:
    """Return report heading and summary lines."""
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
                f"- Assist Canonicalizer semantic accuracy: "
                f"{summary['canonicalizer_accuracy_pct']:.2f}%",
                f"- Direct canonicalizer semantic accuracy: "
                f"{summary['direct_canonicalizer_accuracy_pct']:.2f}%",
                f"- Direct HassIL semantic accuracy: "
                f"{summary['hassil_baseline_accuracy_pct']:.2f}%",
                f"- Assist Canonicalizer uplift over HassIL: "
                f"{summary['accuracy_uplift_pp']:+.2f} percentage points",
                f"- Cases recovered/regressions prevented: "
                f"{summary['recovered_case_count']}/"
                f"{summary['shortcut_protected_case_count']}",
                f"- Canonicalizer fallback rate: {summary['fallback_rate_pct']:.2f}%",
                f"- Canonicalizer mismatch rate: {summary['mismatch_rate_pct']:.2f}%",
                f"- Direct canonicalizer fallback rate: "
                f"{summary['direct_canonicalizer_fallback_rate_pct']:.2f}%",
                f"- Direct canonicalizer mismatch rate: "
                f"{summary['direct_canonicalizer_mismatch_rate_pct']:.2f}%",
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
    return lines, has_corpus_metrics


def _markdown_language_table(
    languages: Mapping[str, Mapping[str, Any]],
    has_corpus_metrics: bool,
) -> list[str]:
    """Return the per-language results table."""
    lines = ["", "## Per-language production results", ""]
    if has_corpus_metrics:
        headers = (
            "Language",
            "Cases",
            "Assist Canonicalizer",
            "Direct canon",
            "HassIL",
            "Uplift pp",
            "Recovered",
            "Regressions prevented",
            "Fallback",
            "Mismatch",
            "Canon mean",
            "Canon p50",
            "Canon p95",
            "HassIL mean",
            "HassIL p50",
            "HassIL p95",
        )
        alignments = "<>>>>>>>>>>>>>>>"
        rows = [
            (
                f"`{language}`",
                str(metrics["case_count"]),
                f"{metrics.get('canonicalizer_accuracy_pct', 0.0):.2f}%",
                f"{metrics.get('direct_canonicalizer_accuracy_pct', 0.0):.2f}%",
                f"{metrics.get('hassil_baseline_accuracy_pct', 0.0):.2f}%",
                f"{metrics.get('accuracy_uplift_pp', 0.0):+.2f}",
                str(metrics.get("recovered_case_count", 0)),
                str(metrics.get("shortcut_protected_case_count", 0)),
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
        headers = (
            "Language",
            "Cases",
            "Passed",
            "Pass rate",
            "Canon mean",
            "Canon p50",
            "Canon p95",
        )
        alignments = "<>>>>>>"
        rows = [
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
    lines.extend(_md_aligned_table(headers, alignments, rows))
    return lines


def _markdown_category_table(
    categories: Mapping[str, Mapping[str, Any]],
    has_corpus_metrics: bool,
) -> list[str]:
    """Return the per-category results table."""
    lines = ["", "## Per-category production results", ""]
    if has_corpus_metrics:
        headers = (
            "Category",
            "Cases",
            "Assist Canonicalizer",
            "Direct canon",
            "HassIL",
            "Uplift pp",
            "Recovered",
            "Regressions prevented",
            "Fallback",
            "Mismatch",
        )
        alignments = "<>>>>>>>>>"
        rows = [
            (
                f"`{category}`",
                str(metrics["case_count"]),
                f"{metrics.get('canonicalizer_accuracy_pct', 0.0):.2f}%",
                f"{metrics.get('direct_canonicalizer_accuracy_pct', 0.0):.2f}%",
                f"{metrics.get('hassil_baseline_accuracy_pct', 0.0):.2f}%",
                f"{metrics.get('accuracy_uplift_pp', 0.0):+.2f}",
                str(metrics.get("recovered_case_count", 0)),
                str(metrics.get("shortcut_protected_case_count", 0)),
                f"{metrics.get('fallback_rate_pct', 0.0):.2f}%",
                f"{metrics.get('mismatch_rate_pct', 0.0):.2f}%",
            )
            for category, metrics in categories.items()
        ]
    else:
        headers = ("Category", "Cases", "Passed", "Pass rate")
        alignments = "<>>>"
        rows = [
            (
                f"`{category}`",
                str(metrics["case_count"]),
                str(metrics.get("passed_cases", 0)),
                f"{metrics.get('accuracy_pct', 0.0):.2f}%",
            )
            for category, metrics in categories.items()
        ]
    lines.extend(_md_aligned_table(headers, alignments, rows))
    return lines


def _markdown_case_table(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return direct, protected, and effective correctness for every case.

    ``Direct correct`` applies the strict direct-agent rule used by aggregation:
    expected semantics without raw fallback. ``Integration correct`` additionally
    accepts a paired HassIL success, and ``Regression prevented`` identifies exactly
    the cases added by that production shortcut.
    """
    headers = (
        "Case",
        "Language",
        "Category",
        "Policy pass",
        "Direct correct",
        "Integration correct",
        "Regression prevented",
        "HassIL correct",
        "Fallback",
        "Canon mean ms",
        "HassIL mean ms",
    )
    rows = []
    for case in cases:
        observation = case["last_observation"]
        hassil_passed = case.get("hassil_baseline_passed")
        direct_correct = bool(case.get("semantic_passed")) and not bool(
            observation.get("fallback_observed")
        )
        effective_correct = bool(hassil_passed) or direct_correct
        shortcut_protected = bool(hassil_passed) and not direct_correct
        hassil_latency = case.get("hassil_baseline_latency_ms")
        hassil_result = ("yes" if hassil_passed else "no") if hassil_passed is not None else "n/a"
        hassil_mean = f"{hassil_latency['mean']:.3f}" if isinstance(hassil_latency, dict) else "n/a"
        rows.append(
            (
                f"`{case['id']}`",
                f"`{case['language']}`",
                f"`{case['category']}`",
                "yes" if case["passed"] else "no",
                "yes" if direct_correct else "no",
                "yes" if effective_correct else "no",
                "yes" if shortcut_protected else "no",
                hassil_result,
                "yes" if observation.get("fallback_observed") else "no",
                f"{case['latency_ms']['mean']:.3f}",
                hassil_mean,
            )
        )
    lines = ["", "## Case results", ""]
    lines.extend(_md_aligned_table(headers, "<><^^^^^^>>", rows))
    return lines


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    """Write a comprehensive human-readable managed-live report."""
    lines, has_corpus_metrics = _markdown_summary(report)
    breakdowns = report.get("breakdowns", {})
    if isinstance(breakdowns, dict):
        languages = breakdowns.get("languages", {})
        if isinstance(languages, dict) and languages:
            lines.extend(_markdown_language_table(languages, has_corpus_metrics))
        categories = breakdowns.get("categories", {})
        if isinstance(categories, dict) and categories:
            lines.extend(_markdown_category_table(categories, has_corpus_metrics))
    cases = report.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, str | bytes):
        raise BenchmarkError("Benchmark report cases are invalid")
    validated_cases: list[Mapping[str, Any]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise BenchmarkError("Benchmark report cases are invalid")
        validated_cases.append(case)
    lines.extend(_markdown_case_table(validated_cases))
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


@dataclass(frozen=True, slots=True)
class _BenchmarkInputs:
    """Static inputs prepared before Home Assistant starts."""

    dependencies: Mapping[str, Any]
    fixture: Mapping[str, Any]
    suite_id: str
    cases: tuple[BenchmarkCase, ...]
    languages: tuple[str, ...]
    categories: tuple[str, ...]
    language_smoke_commands: tuple[LanguageSmokeCommand, ...]
    integration_source_sha256: str
    language_smoke_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _LiveBenchmarkResults:
    """Results collected from the managed Home Assistant session."""

    fixture_state: Mapping[str, Any]
    prepared_languages: dict[str, object]
    case_results: list[dict[str, Any]]
    suite_failures: list[str]
    language_smoke: list[dict[str, object]]


def _language_smoke_manifest(
    commands: Sequence[LanguageSmokeCommand],
) -> list[dict[str, Any]]:
    """Return the stable language-smoke fingerprint payload."""
    return [
        {
            "language": command.language,
            "text": command.text,
            "target_domain": command.target_domain,
            "target_entity_id": command.target_entity_id,
            "expected_slots": dict(command.expected_slots),
        }
        for command in commands
    ]


def _benchmark_inputs(args: argparse.Namespace) -> _BenchmarkInputs:
    """Load and fingerprint all static benchmark inputs."""
    dependencies = verify_benchmark_dependencies()
    fixture = _load_json_object(FIXTURE_PATH, "benchmark fixture")
    suite_id, all_cases = load_cases(args.cases)
    languages = _comma_separated_values(args.languages)
    categories = _comma_separated_values(args.categories)
    cases = _select_cases(all_cases, languages, categories, args.case_limit)
    smoke_commands = build_language_smoke_commands()
    return _BenchmarkInputs(
        dependencies=dependencies,
        fixture=fixture,
        suite_id=suite_id,
        cases=cases,
        languages=languages,
        categories=categories,
        language_smoke_commands=smoke_commands,
        integration_source_sha256=_tree_sha256(INTEGRATION_PATH),
        language_smoke_manifest_sha256=_canonical_payload_sha256(
            _language_smoke_manifest(smoke_commands)
        ),
    )


async def _run_live_benchmark_session(
    session: aiohttp.ClientSession,
    managed: ManagedProcess,
    inputs: _BenchmarkInputs,
    args: argparse.Namespace,
) -> _LiveBenchmarkResults:
    """Prepare Home Assistant and execute all live benchmark requests."""
    await _wait_for_http(session, managed)
    print("HA_HTTP_READY", flush=True)
    token = await _onboard(session)
    session.headers.update({"Authorization": f"Bearer {token}"})
    # The HTTP configure command is accepted only after Home Assistant reaches the
    # running state. Fixture provisioning begins on EVENT_HOMEASSISTANT_STARTED, so
    # readiness is also the benchmark's concrete startup-complete signal.
    await _wait_for_fixture(session, inputs.fixture)
    restart_for_http_config = await _configure_managed_http_config(session, token)
    if restart_for_http_config:
        print("HA_HTTP_CONFIG_RESTARTING", flush=True)
        await _restart_managed_home_assistant(managed)
        await _wait_for_http(session, managed)
        print("HA_HTTP_RESTARTED", flush=True)
        await _wait_for_fixture(session, inputs.fixture)
    promoted_http_config = await _confirm_managed_http_config(session, token)
    print(
        "HA_HTTP_CONFIG_PROMOTED" if promoted_http_config else "HA_HTTP_CONFIG_STABLE",
        flush=True,
    )
    await _wait_for_fixture(session, inputs.fixture)
    await _create_config_entry(session, "shopping_list")
    await _call_service(session, "assist_canonicalizer_benchmark", "reapply")
    fixture_state = await _wait_for_fixture(session, inputs.fixture)
    print("BENCHMARK_FIXTURE_VERIFIED", flush=True)
    config = await _request_json(session, "GET", "/api/config")
    if not isinstance(config, dict):
        raise BenchmarkError("Home Assistant config response must be an object")
    if config.get("version") != inputs.dependencies["homeassistant"]:
        raise BenchmarkError(
            "Home Assistant API version differs from the verified Python distribution: "
            f"{config.get('version')} != {inputs.dependencies['homeassistant']}"
        )
    agent_id = await _create_integration_entry(session)
    await _wait_for_agent(session)
    prepared_languages = await _prepare_languages(
        session,
        tuple(
            {case.language for case in inputs.cases}
            | {command.language for command in inputs.language_smoke_commands}
        ),
    )
    case_results, suite_failures = await _execute_suite(
        session,
        inputs.cases,
        agent_id,
        args.iterations,
        args.warmup,
    )
    language_smoke = await _execute_language_smoke(
        session,
        inputs.language_smoke_commands,
        agent_id,
    )
    return _LiveBenchmarkResults(
        fixture_state=fixture_state,
        prepared_languages=prepared_languages,
        case_results=case_results,
        suite_failures=suite_failures,
        language_smoke=language_smoke,
    )


def _benchmark_context_fingerprint(
    args: argparse.Namespace,
    inputs: _BenchmarkInputs,
    fixture_state: Mapping[str, Any],
    case_suite_sha256: str,
    configuration_sha256: str,
) -> str:
    """Return the fingerprint for all benchmark comparison inputs."""
    return _canonical_payload_sha256(
        {
            "homeassistant_version": inputs.dependencies["homeassistant"],
            "python_version": sys.version.split()[0],
            "dependencies": inputs.dependencies["packages"],
            "fixture_fingerprint": fixture_state["fingerprint"],
            "case_suite_sha256": case_suite_sha256,
            "configuration_sha256": configuration_sha256,
            "integration_source_sha256": inputs.integration_source_sha256,
            "language_smoke_manifest_sha256": (inputs.language_smoke_manifest_sha256),
            "iterations": args.iterations,
            "warmup": args.warmup,
            "device_id": BENCHMARK_DEVICE_ID,
            "satellite_id": CONTEXT_SATELLITE_ID,
        }
    )


def _benchmark_environment(
    inputs: _BenchmarkInputs,
    live: _LiveBenchmarkResults,
    context_fingerprint: str,
) -> dict[str, Any]:
    """Return the report's reproducibility environment."""
    fixture = inputs.fixture
    return {
        "homeassistant_version": inputs.dependencies["homeassistant"],
        "python_version": sys.version.split()[0],
        "dependencies": inputs.dependencies["packages"],
        "installed_intent_languages": [
            command.language for command in inputs.language_smoke_commands
        ],
        "integration_source_sha256": inputs.integration_source_sha256,
        "language_smoke_manifest_sha256": inputs.language_smoke_manifest_sha256,
        "context_fingerprint": context_fingerprint,
        "fixture": {
            "fixture_id": fixture["fixture_id"],
            "schema_version": fixture["schema_version"],
            "fingerprint": live.fixture_state["fingerprint"],
            "counts": fixture["expected_counts"],
            "domain_counts": fixture["expected_domain_counts"],
            "runtime_state_count": live.fixture_state.get("runtime_state_count"),
        },
    }


def _benchmark_settings(
    args: argparse.Namespace,
    inputs: _BenchmarkInputs,
) -> dict[str, Any]:
    """Return execution and interpretation settings recorded in the report.

    The runner executes paired direct requests for deterministic observation. The
    ``effective_result`` marker records that aggregation composes those observations
    using the production HassIL-first invariant rather than treating the direct
    canonicalizer request as the complete user-visible path.
    """
    return {
        "iterations": args.iterations,
        "warmup": args.warmup,
        "serial_execution": True,
        "endpoint": f"{HOST}:{PORT}",
        "languages": list(inputs.languages),
        "categories": list(inputs.categories),
        "case_limit": args.case_limit,
        "device_id": BENCHMARK_DEVICE_ID,
        "satellite_id": CONTEXT_SATELLITE_ID,
        "canonical_oracle": "executed_live_default_agent_canonical_control",
        "hassil_baseline": "paired_original_query_to_live_default_agent",
        "observed_result": "production_default_agent_trace_and_resolved_entities",
        "effective_result": "hassil_first_shortcut_then_direct_canonicalizer",
    }


def _build_benchmark_report(
    args: argparse.Namespace,
    inputs: _BenchmarkInputs,
    live: _LiveBenchmarkResults,
    started_at: float,
) -> dict[str, Any]:
    """Build the managed-live benchmark report."""
    summary = _aggregate(live.case_results)
    case_suite_sha256 = _case_input_sha256(args.cases)
    configuration_sha256 = _file_sha256(CONFIGURATION_PATH)
    context_fingerprint = _benchmark_context_fingerprint(
        args,
        inputs,
        live.fixture_state,
        case_suite_sha256,
        configuration_sha256,
    )
    return {
        "report_schema_version": BENCHMARK_SCHEMA_VERSION,
        "authoritative": True,
        "benchmark_mode": "managed_live",
        "execution_tier": "managed_live",
        "suite_id": inputs.suite_id,
        "case_suite_sha256": case_suite_sha256,
        "case_input_files": _case_input_files(args.cases),
        "configuration_sha256": configuration_sha256,
        "environment": _benchmark_environment(inputs, live, context_fingerprint),
        "settings": _benchmark_settings(args, inputs),
        "prepared_languages": live.prepared_languages,
        "language_support": {
            "accuracy_gated": sorted(ACCURACY_GATED_LANGUAGES),
            "compatibility_smoke_count": len(live.language_smoke),
            "compatibility_smoke": live.language_smoke,
        },
        "summary": summary,
        "breakdowns": _breakdowns(live.case_results),
        "cases": live.case_results,
        "startup_and_run_seconds": time.perf_counter() - started_at,
    }


def _threshold_failure(
    actual: float,
    threshold: float,
    label: str,
    direction: Literal["above", "below"],
    prefix: str = "",
) -> str | None:
    """Return a formatted threshold failure when the limit is violated."""
    violates = actual < threshold if direction == "below" else actual > threshold
    if not violates:
        return None
    return f"{prefix}{label} {actual:.2f}% is {direction} {threshold:.2f}%"


def _required_summary_metric(
    summary: Mapping[str, Any],
    metric: str,
    *,
    scope: str | None = None,
) -> float:
    """Return a required numeric summary metric."""
    if metric not in summary:
        scope_text = f" for {scope}" if scope is not None else ""
        raise BenchmarkError(f"Benchmark summary metric {metric!r} is missing{scope_text}")
    return float(summary[metric])


def _global_threshold_failures(
    args: argparse.Namespace,
    summary: Mapping[str, Any],
) -> list[str]:
    """Return failures for whole-corpus HassIL-first production thresholds.

    All three outcome gates use the same effective production partition. In
    particular, the legacy-named ``--min-intent-slot-accuracy`` option reads
    ``canonicalizer_accuracy_pct`` so a correct HassIL shortcut counts as success,
    just as it does in the user-facing benchmark table. Explicit
    ``direct_canonicalizer_*`` metrics remain diagnostics and are not used by these
    production gates.
    """
    thresholds: tuple[
        tuple[float | None, str, str, Literal["above", "below"]],
        ...,
    ] = (
        (
            args.min_intent_slot_accuracy,
            "canonicalizer_accuracy_pct",
            "production accuracy",
            "below",
        ),
        (args.max_fallback_rate, "fallback_rate_pct", "fallback rate", "above"),
        (args.max_mismatch_rate, "mismatch_rate_pct", "mismatch rate", "above"),
    )
    return [
        failure
        for threshold, metric, label, direction in thresholds
        if threshold is not None
        and (
            failure := _threshold_failure(
                _required_summary_metric(summary, metric),
                threshold,
                label,
                direction,
            )
        )
    ]


def _language_threshold_failures(
    args: argparse.Namespace,
    language_summaries: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Return per-language failures from the HassIL-first outcome partition.

    This deliberately mirrors :func:`_global_threshold_failures`: protected cases
    contribute to effective accuracy and cannot also contribute to mismatch or
    fallback. Direct-agent metrics remain available in each language summary for
    separate diagnosis.
    """
    thresholds: tuple[
        tuple[float | None, str, str, Literal["above", "below"]],
        ...,
    ] = (
        (
            args.min_language_intent_slot_accuracy,
            "canonicalizer_accuracy_pct",
            "production accuracy",
            "below",
        ),
        (
            args.max_language_fallback_rate,
            "fallback_rate_pct",
            "fallback rate",
            "above",
        ),
        (
            args.max_language_mismatch_rate,
            "mismatch_rate_pct",
            "mismatch rate",
            "above",
        ),
    )
    return [
        failure
        for language, summary in sorted(language_summaries.items())
        for threshold, metric, label, direction in thresholds
        if threshold is not None
        and (
            failure := _threshold_failure(
                _required_summary_metric(summary, metric, scope=language.upper()),
                threshold,
                label,
                direction,
                prefix=f"{language.upper()}: ",
            )
        )
    ]


def _benchmark_threshold_failures(
    args: argparse.Namespace,
    report: Mapping[str, Any],
) -> list[str]:
    """Return all configured benchmark threshold failures."""
    summary = report.get("summary")
    breakdowns = report.get("breakdowns")
    if not isinstance(summary, Mapping) or not isinstance(breakdowns, Mapping):
        raise BenchmarkError("Benchmark report summary or breakdowns are invalid")
    raw_language_summaries = breakdowns.get("languages", {})
    if not isinstance(raw_language_summaries, Mapping):
        raise BenchmarkError("Benchmark language summaries are invalid")
    language_summaries: dict[str, Mapping[str, Any]] = {}
    for language, language_summary in raw_language_summaries.items():
        if not isinstance(language, str) or not isinstance(language_summary, Mapping):
            raise BenchmarkError("Benchmark language summaries are invalid")
        language_summaries[language] = language_summary
    return [
        *_global_threshold_failures(args, summary),
        *_language_threshold_failures(args, language_summaries),
    ]


def _finalize_benchmark_report(
    args: argparse.Namespace,
    report: dict[str, Any],
    suite_failures: list[str],
) -> None:
    """Write the report and enforce requested failure policies."""
    regressions = _baseline_regressions(
        report,
        args.baseline,
        args.max_p95_regression_pct,
        args.allow_homeassistant_upgrade,
    )
    threshold_failures = _benchmark_threshold_failures(args, report)
    report["regressions"] = regressions
    report["case_failures"] = suite_failures
    report["threshold_failures"] = threshold_failures
    _write_json(args.output_json, report)
    _write_markdown(args.output_markdown, report)
    if args.fail_on_case_failure and suite_failures:
        raise BenchmarkError("Functional benchmark failures:\n- " + "\n- ".join(suite_failures))
    if args.fail_on_regression and regressions:
        raise BenchmarkError("Benchmark regressions:\n- " + "\n- ".join(regressions))
    if threshold_failures:
        raise BenchmarkError("Benchmark threshold failures:\n- " + "\n- ".join(threshold_failures))


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Own the full ephemeral Home Assistant benchmark lifecycle.

    Home Assistant's sanitized process-log tail is attached only to setup or runtime
    failures where server diagnostics can identify the cause. Report-policy failures
    are enforced outside that exception handler because their generated case,
    regression, and threshold details are already complete and actionable; appending
    unrelated Home Assistant warnings would obscure those details.
    """
    inputs = _benchmark_inputs(args)
    _assert_port_available()
    config_dir = _create_config_dir()
    managed: ManagedProcess | None = None
    started_at = time.perf_counter()
    try:
        try:
            await _run_config_check(config_dir)
            print("HA_CONFIG_OK", flush=True)
            managed = await _start_home_assistant(config_dir)
            timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                live = await _run_live_benchmark_session(
                    session,
                    managed,
                    inputs,
                    args,
                )

            report = _build_benchmark_report(args, inputs, live, started_at)
        except Exception as err:
            if tail := _log_tail(managed):
                raise BenchmarkError(
                    f"{err}\n\nHome Assistant process log tail (credentials redacted):\n{tail}"
                ) from err
            raise
        _finalize_benchmark_report(args, report, live.suite_failures)
        return report
    finally:
        await _stop_home_assistant(managed)
        shutil.rmtree(config_dir)


def _add_benchmark_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Add case-selection and execution arguments."""
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
        "--list-cases",
        action="store_true",
        help="Validate and list cases without starting Home Assistant",
    )


def _add_benchmark_policy_arguments(parser: argparse.ArgumentParser) -> None:
    """Add baseline and failure-policy arguments."""
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


def _add_benchmark_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    """Add whole-corpus and per-language threshold arguments."""
    parser.add_argument(
        "--min-intent-slot-accuracy",
        type=float,
        default=None,
        help="Fail when HassIL-first production accuracy falls below this percentage",
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
        help=(
            "Fail when any language's HassIL-first production accuracy falls below this percentage"
        ),
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


def _parser() -> argparse.ArgumentParser:
    """Build the managed-live benchmark command-line parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark Assist Canonicalizer through a fresh live Home Assistant process"
    )
    _add_benchmark_input_arguments(parser)
    _add_benchmark_policy_arguments(parser)
    _add_benchmark_threshold_arguments(parser)
    return parser


def _validate_cli_percentage(
    parser: argparse.ArgumentParser,
    name: str,
    value: float | None,
) -> None:
    """Validate an optional command-line percentage."""
    if value is not None and not (0.0 <= value <= 100.0):
        parser.error(f"{name} must be between 0.0 and 100.0")


def _validate_cli_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """Validate managed-live command-line arguments."""
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("--case-limit must be positive")
    if args.max_p95_regression_pct < 0:
        parser.error("--max-p95-regression-pct must be non-negative")
    for name, value in (
        ("--min-intent-slot-accuracy", args.min_intent_slot_accuracy),
        ("--max-fallback-rate", args.max_fallback_rate),
        ("--max-mismatch-rate", args.max_mismatch_rate),
        (
            "--min-language-intent-slot-accuracy",
            args.min_language_intent_slot_accuracy,
        ),
        ("--max-language-fallback-rate", args.max_language_fallback_rate),
        ("--max-language-mismatch-rate", args.max_language_mismatch_rate),
    ):
        _validate_cli_percentage(parser, name, value)


def _safe_cli_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Resolve path arguments within the repository and return a safe namespace."""
    try:
        safe_cases = _safe_repository_path(args.cases, "case suite")
        safe_output_json = _safe_repository_path(args.output_json, "JSON output")
        safe_output_markdown = _safe_repository_path(
            args.output_markdown,
            "Markdown output",
        )
        safe_baseline = (
            _safe_repository_path(args.baseline, "baseline") if args.baseline is not None else None
        )
    except BenchmarkError as err:
        parser.error(str(err))
    if safe_output_json == safe_output_markdown:
        parser.error("--output-json and --output-markdown must be different paths")
    safe_values = vars(args).copy()
    safe_values.update(
        cases=safe_cases,
        output_json=safe_output_json,
        output_markdown=safe_output_markdown,
        baseline=safe_baseline,
    )
    return argparse.Namespace(**safe_values)


def _list_selected_cases(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """Validate, select, and print benchmark cases."""
    try:
        suite_id, all_cases = load_cases(args.cases)
        cases = _select_cases(
            all_cases,
            _comma_separated_values(args.languages),
            _comma_separated_values(args.categories),
            args.case_limit,
        )
    except BenchmarkError as err:
        parser.error(str(err))
    print(f"{suite_id}: {len(cases)} cases")
    for case in cases:
        print(f"{case.case_id}\t{case.language}\t{case.query}")


def _benchmark_success_fields(summary: Mapping[str, Any]) -> list[str]:
    """Return the stable successful-run console fields."""
    fields = [
        "BENCHMARK_SUCCESS",
        f"cases={summary['passed_cases']}/{summary['case_count']}",
    ]
    if "canonicalizer_accuracy_pct" not in summary:
        fields.extend(
            (
                f"accuracy={summary['accuracy_pct']:.2f}%",
                f"mean_ms={summary['latency_ms']['mean']:.3f}",
                f"p50_ms={summary['latency_ms']['median']:.3f}",
                f"p95_ms={summary['latency_ms']['p95']:.3f}",
            )
        )
        return fields
    canonicalizer_latency = summary["latency_ms"]
    hassil_latency = summary["hassil_baseline_latency_ms"]
    fields.extend(
        (
            f"canonicalizer_accuracy={summary['canonicalizer_accuracy_pct']:.2f}%",
            f"direct_canonicalizer_accuracy={summary['direct_canonicalizer_accuracy_pct']:.2f}%",
            f"hassil_accuracy={summary['hassil_baseline_accuracy_pct']:.2f}%",
            f"uplift_pp={summary['accuracy_uplift_pp']:+.2f}",
            f"shortcut_protected={summary['shortcut_protected_case_count']}",
            f"canonicalizer_fallback={summary['fallback_rate_pct']:.2f}%",
            f"canonicalizer_mismatch={summary['mismatch_rate_pct']:.2f}%",
            f"direct_canonicalizer_fallback="
            f"{summary['direct_canonicalizer_fallback_rate_pct']:.2f}%",
            f"direct_canonicalizer_mismatch="
            f"{summary['direct_canonicalizer_mismatch_rate_pct']:.2f}%",
            f"canonicalizer_mean_ms={canonicalizer_latency['mean']:.3f}",
            f"canonicalizer_p50_ms={canonicalizer_latency['median']:.3f}",
            f"canonicalizer_p95_ms={canonicalizer_latency['p95']:.3f}",
            f"hassil_mean_ms={hassil_latency['mean']:.3f}",
            f"hassil_p50_ms={hassil_latency['median']:.3f}",
            f"hassil_p95_ms={hassil_latency['p95']:.3f}",
        )
    )
    return fields


def main() -> None:
    """Run the command-line managed-live benchmark."""
    parser = _parser()
    args = parser.parse_args()
    _validate_cli_arguments(parser, args)
    safe_args = _safe_cli_arguments(parser, args)

    if safe_args.list_cases:
        _list_selected_cases(parser, safe_args)
        return

    env_info = _benchmark_environment_summary()
    print(f"BENCHMARK_START mode=managed_live {env_info}", flush=True)
    try:
        report = asyncio.run(run_benchmark(safe_args))
    except (BenchmarkError, OSError, ValueError) as err:
        print(f"BENCHMARK_FAILED: {err}", file=sys.stderr, flush=True)
        raise SystemExit(1) from err
    print(
        " ".join(_benchmark_success_fields(report["summary"])),
        flush=True,
    )
    print(f"JSON report: {safe_args.output_json}")
    print(f"Markdown report: {safe_args.output_markdown}")


if __name__ == "__main__":
    main()
