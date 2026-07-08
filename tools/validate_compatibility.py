"""Multi-version Home Assistant compatibility test suite.

This script manages virtual environments for testing the integration against
multiple Home Assistant core versions.

SECURITY NOTE:
Commands are intentionally hardcoded as explicit list literals in subprocess.run
calls where possible to satisfy static analysis security audits. This prevents
false positives related to command injection.
"""

import argparse
import contextlib
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from collections.abc import Generator
from pathlib import Path
from string import ascii_letters, digits
from typing import Any, TypedDict

import orjson

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)

_VENVS_ROOT = os.path.join(_REPO_ROOT, ".venvs")

_VENV_DEPENDENCY_MARKER = ".assist_canonicalizer_test_dependencies.json"

_INTENTS_PACKAGE = "home-assistant-intents"

_REQUIRED_TEST_DEPS = [
    "hassil",
    _INTENTS_PACKAGE,
    "pytest",
    "pytest-homeassistant-custom-component",
    "rapidfuzz",
]

_COMPATIBILITY_PYTEST_ARGS = ["--no-cov", "-m", "not current_intents"]
_COMPATIBILITY_METADATA_PROBE_TIMEOUT_SECONDS = 60
_VENV_CREATE_TIMEOUT_SECONDS = 120
_INSTALL_TIMEOUT_SECONDS = 300
_CLEANUP_TIMEOUT_SECONDS = 30
_COMPATIBILITY_PYTEST_TIMEOUT_SECONDS = 300

_ALNUM_CHARS = ascii_letters + digits
_SEP_CHAR = "."
_ALLOWED_VERSION_CHARS = f"{_ALNUM_CHARS}."
_ALLOWED_PACKAGE_CHARS = f"{_ALNUM_CHARS}._-"

_VERSION_PATTERN = re.compile(rf"^[{_ALNUM_CHARS}]+(?:\.[{_ALNUM_CHARS}]+)*$")
_PACKAGE_NAME_PATTERN = re.compile(rf"^[{_ALNUM_CHARS}](?:[{_ALNUM_CHARS}._-]*[{_ALNUM_CHARS}])?$")

_MATRIX_FILE = os.path.join(_REPO_ROOT, "tools", "compatibility_matrix.json")

_PYPI_HA_JSON_URL = "https://pypi.org/pypi/homeassistant/json"

_HA_REQUIREMENTS_ALL_GITHUB_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/home-assistant/core/{ha_version}/requirements_all.txt"
)

_HA_REQUIREMENTS_ALL_CDN_URL_TEMPLATE = (
    "https://cdn.jsdelivr.net/gh/home-assistant/core@{ha_version}/requirements_all.txt"
)


class CompatibilityConfig(TypedDict):
    """Validated Home Assistant compatibility test matrix entry."""

    ha_ver: str
    python_ver: str
    pinned_test_dependencies: object


def _expected_required_test_dep_versions(
    pinned_test_dependency_versions: dict[str, str],
) -> dict[str, str]:
    """Return explicitly pinned test dependency versions for a compatibility venv."""
    return dict(pinned_test_dependency_versions)


def _required_test_deps(pinned_test_dependency_versions: dict[str, str]) -> list[str]:
    """Return install specs for required test dependencies."""
    deps: list[str] = []
    seen: set[str] = set()
    for package in _REQUIRED_TEST_DEPS:
        seen.add(package)
        if package in pinned_test_dependency_versions:
            deps.append(f"{package}=={pinned_test_dependency_versions[package]}")
        else:
            deps.append(package)
    deps.extend(
        f"{package}=={version}"
        for package, version in sorted(pinned_test_dependency_versions.items())
        if package not in seen
    )
    return deps


def _dependency_pin_marker_payload(
    pinned_test_dependency_versions: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Return the canonical venv marker payload for resolved test dependency pins."""
    return {
        "pinned_test_dependency_versions": dict(sorted(pinned_test_dependency_versions.items()))
    }


def _venv_dependency_marker_matches(
    venv_path: Path,
    pinned_test_dependency_versions: dict[str, str],
) -> bool:
    """Return whether a compatibility venv was installed with the same resolved pins."""
    marker_path = venv_path / _VENV_DEPENDENCY_MARKER
    try:
        parsed = orjson.loads(marker_path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return False
    return parsed == _dependency_pin_marker_payload(pinned_test_dependency_versions)


def _dependency_marker_requires_reinstall(
    created_venv: bool,
    venv_path: Path,
    pinned_test_dependency_versions: dict[str, str],
) -> bool:
    """Return whether the resolved pin marker requires a full dependency install."""
    if created_venv or _venv_dependency_marker_matches(
        venv_path,
        pinned_test_dependency_versions,
    ):
        return False
    print(
        "STEP_INFO: dependency pin marker changed; reinstalling test dependencies",
        flush=True,
    )
    return True


def _write_venv_dependency_marker(
    venv_path: Path,
    pinned_test_dependency_versions: dict[str, str],
) -> None:
    """Persist the resolved dependency pin marker for future venv reuse checks."""
    marker_path = venv_path / _VENV_DEPENDENCY_MARKER
    marker_path.write_bytes(
        orjson.dumps(
            _dependency_pin_marker_payload(pinned_test_dependency_versions),
            option=orjson.OPT_SORT_KEYS,
        )
    )


def _parse_requirements_dependency_version(
    requirements_text: str,
    package_name: str,
) -> str:
    """Return the exact package version from a requirements file."""
    package = _validate_package_name("package_name", package_name)
    for raw_line in requirements_text.splitlines():
        line = raw_line.strip()
        requirement_name, separator, version_text = line.partition("==")
        if separator != "==":
            continue
        try:
            requirement_package = _validate_package_name(
                "requirements_package_name",
                requirement_name.strip(),
            )
        except ValueError:
            continue
        if requirement_package != package:
            continue
        version_tokens = version_text.split(";", 1)[0].strip().split()
        if not version_tokens:
            raise ValueError(
                f"Invalid {package!r} requirement in Home Assistant requirements_all.txt; "
                "expected a version after '=='."
            )
        version = version_tokens[0]
        return _validate_version_label(f"{package}_version", version)
    raise ValueError(f"Could not find {package!r} in Home Assistant requirements_all.txt")


def _fetch_requirements_all_text(url: str) -> str:
    """Fetch and decode requirements_all.txt from a URL.

    Raises urllib.error.URLError, OSError, or UnicodeDecodeError on failure.
    """
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8")


def _get_home_assistant_intents_version(ha_ver: str) -> str:
    """Fetch the exact home-assistant-intents version required by a Home Assistant tag.

    Tries the jsDelivr CDN mirror first to avoid GitHub raw-content 429 rate
    limits, then falls back to the canonical GitHub URL.
    """
    version = _validate_version_label("ha_ver", ha_ver)
    cdn_url = _HA_REQUIREMENTS_ALL_CDN_URL_TEMPLATE.format(ha_version=version)
    github_url = _HA_REQUIREMENTS_ALL_GITHUB_URL_TEMPLATE.format(ha_version=version)
    last_err: Exception | None = None
    for url in (cdn_url, github_url):
        try:
            requirements_text = _fetch_requirements_all_text(url)
            return _parse_requirements_dependency_version(requirements_text, _INTENTS_PACKAGE)
        except (urllib.error.URLError, OSError, ValueError) as err:
            last_err = err
    raise ValueError(
        f"Failed to fetch Home Assistant requirements_all.txt for {version} "
        f"(tried CDN and GitHub): {last_err}"
    ) from last_err


def _resolve_pinned_test_dependency_versions(
    ha_ver: str,
    pinned_test_dependencies: object,
) -> dict[str, str]:
    """Return matrix pins plus the Home Assistant intents version when unpinned."""
    pinned_versions = _parse_pinned_test_dependency_versions(pinned_test_dependencies)
    if _INTENTS_PACKAGE not in pinned_versions:
        pinned_versions[_INTENTS_PACKAGE] = _get_home_assistant_intents_version(ha_ver)
    return pinned_versions


def _test_dep_packages(pinned_test_dependency_versions: dict[str, str]) -> tuple[str, ...]:
    """Return base and matrix-pinned dependency package names."""
    packages = _REQUIRED_TEST_DEPS
    for package in pinned_test_dependency_versions:
        if package not in packages:
            packages.append(package)
    return tuple(packages)


def _venv_required_test_dep_versions(
    python_bin: Path,
    pinned_test_dependency_versions: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return required test dependency versions installed in a compatibility venv."""
    packages = _test_dep_packages(pinned_test_dependency_versions or {})
    code = (
        "import contextlib, importlib.metadata as md, json\n"
        f"packages = {packages!r}\n"
        "versions = {}\n"
        "for package in packages:\n"
        "    with contextlib.suppress(md.PackageNotFoundError):\n"
        "        if '[' in package and package.endswith(']'):\n"
        "            base_name, extras_str = package[:-1].split('[', 1)\n"
        "            extras = [e.strip() for e in extras_str.split(',')]\n"
        "            base_ver = md.version(base_name)\n"
        "            satisfied = True\n"
        "            reqs = md.requires(base_name) or []\n"
        "            for extra in extras:\n"
        "                for req in reqs:\n"
        "                    if ';' in req:\n"
        "                        dep, marker = req.split(';', 1)\n"
        "                        marker_norm = marker.replace(' ', '').replace('\"', \"'\")\n"
        "                        if f\"extra=='{extra}'\" in marker_norm:\n"
        "                            dep_name = ''\n"
        "                            for c in dep.strip():\n"
        "                                if not (c.isalnum() or c in '.-_'):\n"
        "                                    break\n"
        "                                dep_name += c\n"
        "                            try:\n"
        "                                md.version(dep_name)\n"
        "                            except md.PackageNotFoundError:\n"
        "                                satisfied = False\n"
        "                                break\n"
        "                if not satisfied:\n"
        "                    break\n"
        "            if satisfied:\n"
        "                versions[package] = base_ver\n"
        "        else:\n"
        "            versions[package] = md.version(package)\n"
        "print(json.dumps(versions, sort_keys=True))\n"
    )
    result = subprocess.run(
        [
            "uv",
            "--no-config",
            "run",
            "--no-project",
            "--python",
            str(python_bin),
            "python",
            "-c",
            code,
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=_COMPATIBILITY_METADATA_PROBE_TIMEOUT_SECONDS,
    )
    parsed = orjson.loads(result.stdout)
    if not isinstance(parsed, dict):
        return {}
    return {str(package): str(version) for package, version in parsed.items()}


def _missing_required_test_deps(
    python_bin: Path,
    pinned_test_dependency_versions: dict[str, str],
) -> tuple[str, ...]:
    """Return required test dependency names that are absent from a compatibility venv."""
    try:
        installed = _venv_required_test_dep_versions(python_bin, pinned_test_dependency_versions)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        orjson.JSONDecodeError,
        OSError,
    ):
        return tuple(
            package
            for package in _REQUIRED_TEST_DEPS
            if package not in pinned_test_dependency_versions
        )
    return tuple(
        package
        for package in _REQUIRED_TEST_DEPS
        if package not in pinned_test_dependency_versions and package not in installed
    )


def _stale_pinned_test_deps(
    python_bin: Path,
    pinned_test_dependency_versions: dict[str, str],
) -> tuple[str, ...]:
    """Return pinned test dependency install specs that drifted from the matrix."""
    expected = _expected_required_test_dep_versions(pinned_test_dependency_versions)
    if not expected:
        return ()
    try:
        installed = _venv_required_test_dep_versions(python_bin, pinned_test_dependency_versions)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        orjson.JSONDecodeError,
        OSError,
    ):
        return tuple(f"{package}=={version}" for package, version in expected.items())

    stale = {
        package: (installed.get(package), expected_version)
        for package, expected_version in expected.items()
        if installed.get(package) != expected_version
    }
    if stale:
        details = ", ".join(
            f"{package} {old or 'missing'} -> {new}"
            for package, (old, new) in sorted(stale.items())
        )
        print(f"STEP_INFO: refreshing pinned test dependencies: {details}", flush=True)
    return tuple(f"{package}=={version}" for package, (_old, version) in sorted(stale.items()))


def _load_matrix_data() -> list[dict[str, Any]]:
    """Load compatibility matrix from the repository tools directory."""
    with open(_MATRIX_FILE, encoding="utf-8") as f:
        loaded = orjson.loads(f.read())
    if not isinstance(loaded, list):
        raise ValueError("Compatibility matrix must be a list")
    matrix: list[dict[str, Any]] = []
    for index, entry in enumerate(loaded, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Compatibility matrix entry {index} must be an object")
        matrix.append({str(key): value for key, value in entry.items()})
    return matrix


def _matrix_entry_text(entry: dict[str, Any], key: str) -> str:
    """Return a required text field from a matrix entry."""
    return _require_str_field(key, entry.get(key))


def _test_matrix() -> list[CompatibilityConfig]:
    """Return validated compatibility matrix entries."""
    data = _load_matrix_data()
    entries = []
    for idx, entry in enumerate(data, start=1):
        try:
            ha_ver = _validate_version_label(
                "ha_version",
                _matrix_entry_text(entry, "ha_version"),
            )
            py_ver = _validate_version_label(
                "python_version",
                _matrix_entry_text(entry, "python_version"),
            )
        except ValueError as err:
            raise ValueError(f"Matrix row {idx}: {err}") from err
        entries.append(
            CompatibilityConfig(
                ha_ver=ha_ver,
                python_ver=py_ver,
                pinned_test_dependencies=entry.get("pinned_test_dependencies", []),
            )
        )
    return entries


def _require_str_field(label_name: str, value: object) -> str:
    """Return value typed as str, or raise ValueError if it is not a string."""
    if not isinstance(value, str):
        raise ValueError(f"Invalid {label_name} value {value!r}; expected a string.")
    return value


def _normalize_package_name(package_name: str) -> str:
    """Normalize a package name to a canonical form."""
    return re.sub(r"[._-]+", "-", package_name).lower()


def _validate_version_label(label_name: str, label_value: str) -> str:
    """Validate and sanitize a matrix version label to prevent path injection.

    Uses a strict regex check to enforce structural validity.

    SECURITY NOTE:
    - The regex and `_ALLOWED_VERSION_CHARS` share underlying constant components
      to ensure synchronization while still allowing the regex to enforce strict
      structural validity (e.g., prohibiting consecutive dots).
    - DO NOT simplify the character reconstruction loop (e.g., via comprehension).
      Mapping via integer index to the static `_ALLOWED_VERSION_CHARS` is required
      to completely sever the CodeQL data-flow taint chain.
    - `os.path.basename` is retained to satisfy CodeQL's hardcoded AST sanitizer rules.
    - The loop fails fast on unknown characters, acting as an extra safety net.
    """
    label_value = _require_str_field(label_name, label_value)

    if not label_value:
        raise ValueError(
            f"Invalid {label_name} value {label_value!r}; expected a non-empty version label."
        )

    if not _VERSION_PATTERN.fullmatch(label_value):
        raise ValueError(
            f"Invalid {label_name} value {label_value!r}; must be alphanumeric blocks "
            "separated by a single dot, and cannot contain consecutive, leading, or trailing dots."
        )

    safe_chars: list[str] = []
    for char in label_value:
        idx = _ALLOWED_VERSION_CHARS.find(char)
        if idx == -1:
            raise ValueError(
                f"Invalid {label_name} value {label_value!r}; character {char!r} is not allowed."
            )
        safe_chars.append(_ALLOWED_VERSION_CHARS[idx])

    safe_val = "".join(safe_chars)
    return os.path.basename(safe_val)


def _validate_package_name(label_name: str, package_name: object) -> str:
    """Validate and sanitize a matrix package name to prevent command injection.

    SECURITY NOTE:
    - DO NOT simplify the character reconstruction loop (e.g., via comprehension).
      Mapping via integer index to the static `_ALLOWED_PACKAGE_CHARS` is required
      to completely sever the CodeQL data-flow taint chain.
    - `os.path.basename` is retained to satisfy CodeQL's hardcoded AST sanitizer rules.
    - The loop fails fast on unknown characters, acting as an extra safety net.
    """
    package_name = _require_str_field(label_name, package_name)

    if not _PACKAGE_NAME_PATTERN.fullmatch(package_name):
        raise ValueError(
            f"Invalid {label_name} value {package_name!r}; must be a Python package name."
        )

    safe_chars: list[str] = []
    for char in package_name:
        idx = _ALLOWED_PACKAGE_CHARS.find(char)
        if idx == -1:
            raise ValueError(
                f"Invalid {label_name} value {package_name!r}; character {char!r} is not allowed."
            )
        safe_chars.append(_ALLOWED_PACKAGE_CHARS[idx])

    safe_val = "".join(safe_chars)
    return _normalize_package_name(os.path.basename(safe_val))


def _parse_pinned_test_dependency_versions(
    pinned_test_dependencies: object,
) -> dict[str, str]:
    """Return package version pins from a compatibility matrix dependency list."""
    if pinned_test_dependencies is None:
        return {}
    if not isinstance(pinned_test_dependencies, list):
        raise ValueError("pinned_test_dependencies must be a list")

    versions: dict[str, str] = {}
    for index, dependency in enumerate(pinned_test_dependencies):
        if not isinstance(dependency, dict):
            raise ValueError(f"pinned_test_dependencies[{index}] must be an object")
        package = _validate_package_name(
            f"pinned_test_dependencies[{index}].package",
            dependency.get("package"),
        )
        if package == "homeassistant":
            raise ValueError(
                "pinned_test_dependencies entries must not pin package "
                "'homeassistant'; use ha_version instead."
            )
        version_value = dependency.get("version")
        if not isinstance(version_value, str):
            raise ValueError(
                f"Invalid pinned_test_dependencies[{index}].version value "
                f"{version_value!r}; expected a string."
            )
        version = _validate_version_label(
            f"pinned_test_dependencies[{index}].version",
            version_value,
        )
        if package in versions:
            raise ValueError(f"Duplicate pinned test dependency {package!r}")
        versions[package] = version
    return versions


def _ensure_within_root(root_path: str, candidate_path: str) -> str:
    """Return safe absolute path only if candidate resides within root_path.

    SECURITY: Resolves the root via os.path.realpath (symlink-safe), joins
       with the cleaned path, normalizes via os.path.normpath, and
       verifies containment via startswith.

    Returns the safe absolute path or raises ValueError.
    """
    root = os.path.realpath(root_path)
    fullpath = os.path.realpath(os.path.normpath(os.path.join(root, candidate_path)))

    if fullpath != root and not fullpath.startswith(root + os.sep):
        raise ValueError(f"Resolved path {fullpath!r} escapes allowed root {root!r}.")
    return fullpath


def _format_cmd_str(cmd: Any) -> str:
    """Return a human-readable string for a subprocess command."""
    if isinstance(cmd, (list, tuple)):
        return " ".join(str(arg) for arg in cmd)
    return str(cmd)


def _get_latest_ha_version() -> str:
    """Fetch the latest Home Assistant version from PyPI.

    Returns:
        The latest version string from PyPI.

    Raises:
        ValueError: If fetching or parsing the version fails.
    """
    try:
        with urllib.request.urlopen(_PYPI_HA_JSON_URL, timeout=20) as response:
            data = orjson.loads(response.read())
            version = data["info"]["version"]
    except (urllib.error.URLError, OSError, orjson.JSONDecodeError, KeyError) as err:
        raise ValueError(f"Failed to fetch latest Home Assistant version from PyPI: {err}") from err

    return _validate_version_label("pypi_version", version)


def _get_venv_path(ha_ver: str, py_ver: str) -> str:
    """Construct the virtual environment path for a specific version."""
    ha = _validate_version_label("ha_ver", ha_ver)
    py = _validate_version_label("py_ver", py_ver)

    venv_name = os.path.basename(f"homeassistant_{ha}_python_{py}")

    if os.path.basename(venv_name) != venv_name:
        raise ValueError(f"Invalid venv name: {venv_name}")

    candidate = os.path.join(_VENVS_ROOT, venv_name)
    return _ensure_within_root(_VENVS_ROOT, candidate)


@contextlib.contextmanager
def _overrides_file(ha_ver: str) -> Generator[str]:
    """Write a HA version-pin overrides file and remove it on exit.

    Yields the absolute path to the overrides file.
    """
    overrides_dir = os.path.join(_REPO_ROOT, "scratch")
    os.makedirs(overrides_dir, exist_ok=True)
    overrides_path = os.path.join(overrides_dir, f"overrides_{uuid.uuid4().hex}.txt")
    with open(overrides_path, "w", encoding="utf-8") as f:
        f.write(f"homeassistant == {ha_ver}\n")
    try:
        yield overrides_path
    finally:
        with contextlib.suppress(OSError):
            os.remove(overrides_path)


def _determine_dependency_actions(
    reinstall: bool,
    created_venv: bool,
    python_bin: Path,
    pinned_test_dependency_versions: dict[str, str],
) -> tuple[bool, tuple[str, ...]]:
    """Determine whether dependencies need a full install or just pinned refreshes."""
    needs_install = reinstall or created_venv
    pinned_refresh_deps: tuple[str, ...] = ()
    if not needs_install:
        if missing_deps := _missing_required_test_deps(
            python_bin,
            pinned_test_dependency_versions,
        ):
            details = ", ".join(sorted(missing_deps))
            print(f"STEP_INFO: installing missing test dependencies: {details}", flush=True)
            needs_install = True
        else:
            pinned_refresh_deps = _stale_pinned_test_deps(
                python_bin,
                pinned_test_dependency_versions,
            )
    return needs_install, pinned_refresh_deps


def _ensure_venv(venv_path: Path, py_ver: str) -> bool:
    """Ensure virtual environment exists, creating it if necessary.

    Returns:
        True if a new virtual environment was created, False otherwise.
    """
    python_bin = venv_path / "bin" / "python"
    pytest_bin = venv_path / "bin" / "pytest"
    if venv_path.exists() and python_bin.exists() and pytest_bin.exists():
        return False
    if venv_path.exists():
        print(f"STEP_INFO: Re-creating incomplete virtual environment at {venv_path}", flush=True)
        shutil.rmtree(venv_path, ignore_errors=True)
    print(f"STEP_START: uv venv {venv_path} (Python {py_ver})", flush=True)
    subprocess.run(
        [
            "uv",
            "--no-config",
            "venv",
            "--no-project",
            "--python",
            py_ver,
            venv_path,
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=_VENV_CREATE_TIMEOUT_SECONDS,
    )
    print(f"STEP_OK: uv venv {venv_path} (Python {py_ver})", flush=True)
    return True


def _reset_venv(venv_path: Path, py_ver: str) -> None:
    """Remove and recreate a compatibility virtual environment."""
    if venv_path.exists():
        print(f"STEP_INFO: Resetting virtual environment at {venv_path}", flush=True)
        shutil.rmtree(venv_path, ignore_errors=True)
    _ensure_venv(venv_path, py_ver)


def _install_dependencies(
    venv_path: Path,
    python_bin: Path,
    ha_ver_to_install: str,
    needs_install: bool,
    pinned_refresh_deps: tuple[str, ...],
    pinned_test_dependency_versions: dict[str, str],
    *,
    py_ver: str,
    reset_before_install: bool = False,
) -> None:
    """Install or upgrade required test dependencies in the compatibility venv."""
    if needs_install:
        if reset_before_install:
            _reset_venv(venv_path, py_ver)
        required_test_deps = _required_test_deps(pinned_test_dependency_versions)
        ha_spec = f"homeassistant=={ha_ver_to_install}"
        print(f"STEP_START: uv pip install {ha_spec}", flush=True)
        with _overrides_file(ha_ver_to_install) as overrides_path:
            subprocess.run(
                [
                    "uv",
                    "--no-config",
                    "pip",
                    "install",
                    "--upgrade",
                    "--overrides",
                    overrides_path,
                    "--python",
                    python_bin,
                    ha_spec,
                    *required_test_deps,
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=_REPO_ROOT,
                timeout=_INSTALL_TIMEOUT_SECONDS,
            )
        print(f"STEP_OK: uv pip install {ha_spec}", flush=True)

    elif pinned_refresh_deps:
        refresh_label = " ".join(pinned_refresh_deps)
        print(f"STEP_START: uv pip install {refresh_label}", flush=True)
        with _overrides_file(ha_ver_to_install) as overrides_path:
            subprocess.run(
                [
                    "uv",
                    "--no-config",
                    "pip",
                    "install",
                    "--upgrade",
                    "--overrides",
                    overrides_path,
                    "--python",
                    python_bin,
                    *pinned_refresh_deps,
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=_REPO_ROOT,
                timeout=_INSTALL_TIMEOUT_SECONDS,
            )
        print(f"STEP_OK: uv pip install {refresh_label}", flush=True)

    if needs_install or pinned_refresh_deps:
        print("STEP_START: cleanup __pycache__", flush=True)
        subprocess.run(
            [
                "find",
                ".",
                "-name",
                "__pycache__",
                "-type",
                "d",
                "-exec",
                "rm",
                "-rf",
                "{}",
                "+",
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=_CLEANUP_TIMEOUT_SECONDS,
        )
        print("STEP_OK: cleanup __pycache__", flush=True)
        _write_venv_dependency_marker(venv_path, pinned_test_dependency_versions)


def _get_installed_ha_version(python_bin: Path) -> str:
    """Get the actually installed Home Assistant version inside the venv."""
    actual_ver = "unknown"
    with contextlib.suppress(subprocess.CalledProcessError, subprocess.TimeoutExpired):
        result = subprocess.run(
            ["uv", "--no-config", "pip", "show", "--python", python_bin, "homeassistant"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
            timeout=_COMPATIBILITY_METADATA_PROBE_TIMEOUT_SECONDS,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                actual_ver = line.split(":", 1)[1].strip()
                break
    return actual_ver


def _run_pytest(python_bin: Path, ha_ver_display: str, pytest_args: list[str]) -> None:
    """Run pytest inside the virtual environment."""
    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    print(f"STEP_START: uv run pytest (Home Assistant {ha_ver_display})", flush=True)
    subprocess.run(
        [
            "uv",
            "--no-config",
            "run",
            "--no-project",
            "--python",
            python_bin,
            "pytest",
            *pytest_args,
        ],
        env=env,
        check=True,
        cwd=_REPO_ROOT,
        timeout=_COMPATIBILITY_PYTEST_TIMEOUT_SECONDS,
    )
    print(f"STEP_OK: uv run pytest (Home Assistant {ha_ver_display})", flush=True)


def _prepare_version_and_deps(
    ha_ver: str,
    pinned_test_dependencies: object,
) -> tuple[str, dict[str, str]]:
    """Resolve target HA version and retrieve pinned test dependencies."""
    ha_ver_to_install = ha_ver
    if ha_ver_to_install == "latest":
        ha_ver_to_install = _get_latest_ha_version()
    pinned_test_dependency_versions = _resolve_pinned_test_dependency_versions(
        ha_ver_to_install,
        pinned_test_dependencies,
    )
    return ha_ver_to_install, pinned_test_dependency_versions


def _prepare_venv_and_install(
    venv_path: Path,
    python_bin: Path,
    ha_ver: str,
    ha_ver_to_install: str,
    py_ver: str,
    reinstall: bool,
    pinned_test_dependency_versions: dict[str, str],
) -> bool:
    """Ensure the virtual environment is prepared and dependencies are installed.

    Returns:
        bool: True if setup succeeded, False if python binary is missing.
    """
    created_venv = _ensure_venv(venv_path, py_ver)

    if not python_bin.exists():
        print(f"VALIDATION_ERROR: python not found at {python_bin}", flush=True)
        return False

    installed_ha = _get_installed_ha_version(python_bin)
    marker_requires_reinstall = _dependency_marker_requires_reinstall(
        created_venv,
        venv_path,
        pinned_test_dependency_versions,
    )
    needs_reinstall = (
        reinstall
        or (installed_ha != ha_ver_to_install)
        or ha_ver == "latest"
        or marker_requires_reinstall
    )

    needs_install, pinned_refresh_deps = _determine_dependency_actions(
        needs_reinstall,
        created_venv,
        python_bin,
        pinned_test_dependency_versions,
    )

    _install_dependencies(
        venv_path,
        python_bin,
        ha_ver_to_install,
        needs_install,
        pinned_refresh_deps,
        pinned_test_dependency_versions,
        py_ver=py_ver,
        reset_before_install=needs_reinstall and not created_venv,
    )
    return True


def _verify_and_run_tests(
    python_bin: Path,
    pytest_bin: Path,
    ha_ver_to_install: str,
) -> tuple[bool, str]:
    """Verify virtual environment completeness and run the test suite.

    Returns:
        tuple[bool, str]: (Success status, Installed HA version)
    """
    if not pytest_bin.exists():
        print(f"VALIDATION_ERROR: pytest not found at {pytest_bin}", flush=True)
        return False, ha_ver_to_install

    ha_ver_display = _get_installed_ha_version(python_bin)
    if ha_ver_display != ha_ver_to_install:
        print(
            f"VALIDATION_ERROR: expected Home Assistant {ha_ver_to_install}, "
            f"found {ha_ver_display}",
            flush=True,
        )
        return False, ha_ver_display

    _run_pytest(python_bin, ha_ver_display, _COMPATIBILITY_PYTEST_ARGS)
    return True, ha_ver_display


def _run_tests_for_version(
    ha_ver: str,
    py_ver: str,
    reinstall: bool,
    pinned_test_dependencies: object,
) -> tuple[bool, str]:
    """Run the test suite for a specific Home Assistant version."""
    ha_ver_display = ha_ver

    try:
        ha_ver_to_install, pinned_test_dependency_versions = _prepare_version_and_deps(
            ha_ver,
            pinned_test_dependencies,
        )

        ha_ver_display = ha_ver_to_install
        print(f"TESTING Home Assistant {ha_ver_to_install} (Python {py_ver})", flush=True)

        venv_path = Path(_get_venv_path(ha_ver_to_install, py_ver))
        python_bin = venv_path / "bin" / "python"
        pytest_bin = venv_path / "bin" / "pytest"

        if not _prepare_venv_and_install(
            venv_path=venv_path,
            python_bin=python_bin,
            ha_ver=ha_ver,
            ha_ver_to_install=ha_ver_to_install,
            py_ver=py_ver,
            reinstall=reinstall,
            pinned_test_dependency_versions=pinned_test_dependency_versions,
        ):
            return False, ha_ver_display

        return _verify_and_run_tests(
            python_bin=python_bin,
            pytest_bin=pytest_bin,
            ha_ver_to_install=ha_ver_to_install,
        )

    except ValueError as err:
        print(f"VALIDATION_ERROR: {err}", flush=True)
        return False, ha_ver_display
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        ret_code = getattr(e, "returncode", 1)
        if isinstance(e, subprocess.TimeoutExpired):
            cmd_str = _format_cmd_str(e.cmd)
            print(f"STEP_FAILED: {cmd_str} TIMEOUT={e.timeout}", flush=True)
        elif isinstance(e, subprocess.CalledProcessError):
            cmd_str = _format_cmd_str(e.cmd)
            print(f"STEP_FAILED: {cmd_str} EXIT_CODE={ret_code}", flush=True)
            if e.stdout:
                print("\nSTDOUT:", flush=True)
                print(e.stdout, flush=True)
            if e.stderr:
                print("\nSTDERR:", flush=True)
                print(e.stderr, flush=True)
        else:
            cmd_str = getattr(e, "filename", "Unknown command")
            print(f"VALIDATION_ERROR: '{cmd_str}' not found.", flush=True)
        return False, ha_ver_display


def main() -> None:
    """Main entry point for the multi-version test script."""
    os.environ["NO_COLOR"] = "1"
    results: list[tuple[int, str, str, str, str]] = []

    if os.name != "posix":
        print("VALIDATION_ERROR: Non-POSIX environment detected", flush=True)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Test multiple HA versions.")
    parser.add_argument("--reinstall", action="store_true", help="Force reinstall of dependencies")
    parser.add_argument(
        "--clean", action="store_true", help="Delete all test venvs before starting"
    )
    args = parser.parse_args()

    if not shutil.which("uv"):
        print("VALIDATION_ERROR: 'uv' is not installed.", flush=True)
        sys.exit(1)

    try:
        if args.clean:
            print("Cleaning up all test venvs...", flush=True)
            if os.path.exists(_VENVS_ROOT):
                shutil.rmtree(_VENVS_ROOT)

        for row_index, config in enumerate(_test_matrix(), start=1):
            ha_ver = config["ha_ver"]
            py_ver = config["python_ver"]
            success, ha_version = _run_tests_for_version(
                ha_ver,
                py_ver,
                args.reinstall,
                config["pinned_test_dependencies"],
            )
            results.append(
                (row_index, ha_ver, py_ver, ha_version, "PASSED" if success else "FAILED")
            )
    except (OSError, ValueError) as exc:
        print(f"VALIDATION_ERROR: {exc}", flush=True)
        sys.exit(1)

    print(flush=True)
    all_ok = True
    for row_index, ha_ver, py_ver, ha_version, status in results:
        display_ver = ha_version if ha_version == ha_ver else f"{ha_ver} → {ha_version}"
        print(
            f"Matrix row {row_index}: Home Assistant {display_ver} (Python {py_ver}): {status}",
            flush=True,
        )
        if status != "PASSED":
            all_ok = False

    print(flush=True)
    if all_ok:
        print("VALIDATION_SUCCESS", flush=True)
    else:
        print("VALIDATION_FAILED", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
