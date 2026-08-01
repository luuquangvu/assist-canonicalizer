"""Tests for unified validation tools."""

import importlib.metadata as md
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from tools import validate
from tools.validate import (
    _load_ha_manifest_constraints,
    _load_project_dependency_packages,
    _parse_package_name_from_req,
)


def test_parse_package_name_from_req() -> None:
    """Verify package name extraction and normalization from requirement specifiers."""
    assert _parse_package_name_from_req("HassIL>=3.0.0") == "hassil"
    assert (
        _parse_package_name_from_req("home_assistant_intents==2026.6.1; python_version >= '3.14'")
        == "home-assistant-intents"
    )
    assert _parse_package_name_from_req("pytest-cov[all] >= 4.0") == "pytest-cov"


def test_load_ha_manifest_constraints() -> None:
    """Verify independent parsing of Home Assistant component manifest constraints."""
    constraints = _load_ha_manifest_constraints()
    assert isinstance(constraints, dict)


def test_load_project_dependency_packages_dynamic_groups(tmp_path: Path) -> None:
    """Verify that all declared dependency-groups are loaded dynamically."""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(
        """
[dependency-groups]
dev = ["hassil>=3.0"]
ha-benchmark = ["colorlog"]
custom-group = ["pytest-mock", "home_assistant_intents"]
""",
        encoding="utf-8",
    )

    packages = _load_project_dependency_packages(str(tmp_path))
    assert packages == {"hassil", "colorlog", "pytest-mock", "home-assistant-intents"}


def test_load_project_dependency_packages_pep735_includes(tmp_path: Path) -> None:
    """Verify PEP 735 group include references are resolved recursively."""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(
        """
[dependency-groups]
base = ["hassil"]
dev = [{ include = "base" }, "pytest"]
ci = [{ include-group = "dev" }, "ruff"]
""",
        encoding="utf-8",
    )

    packages = _load_project_dependency_packages(str(tmp_path))
    assert packages == {"hassil", "pytest", "ruff"}


def test_load_project_dependency_packages_cyclic_includes(tmp_path: Path) -> None:
    """Verify cyclic group includes do not cause infinite recursion."""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(
        """
[dependency-groups]
group-a = [{ include = "group-b" }, "pkg-a"]
group-b = [{ include = "group-a" }, "pkg-b"]
""",
        encoding="utf-8",
    )

    packages = _load_project_dependency_packages(str(tmp_path))
    assert packages == {"pkg-a", "pkg-b"}


def test_load_project_dependency_packages_fallback_missing_file(tmp_path: Path) -> None:
    """Verify empty package set when pyproject.toml does not exist."""
    packages = _load_project_dependency_packages(str(tmp_path))
    assert packages == set()


def test_load_project_dependency_packages_fallback_invalid_table(tmp_path: Path) -> None:
    """Verify empty package set when dependency-groups is not a table."""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(
        """
dependency-groups = "invalid"
""",
        encoding="utf-8",
    )

    packages = _load_project_dependency_packages(str(tmp_path))
    assert packages == set()


def test_check_local_ha_harness_alignment_accepts_exact_pair(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Accept a local harness whose exact requirement matches installed HA."""
    versions = {
        "homeassistant": "2026.7.4",
        "pytest-homeassistant-custom-component": "0.13.348",
    }
    monkeypatch.setattr(validate.md, "version", lambda package: versions[package])
    monkeypatch.setattr(
        validate.md,
        "requires",
        lambda _package: ["homeassistant==2026.7.4", "pytest>=8.0"],
    )

    validate._check_local_ha_harness_alignment()

    output = capsys.readouterr().out
    assert "STEP_INFO: local Home Assistant 2026.7.4 matches" in output
    assert "STEP_OK: check local homeassistant/test harness alignment" in output
    assert "HARNESS_MISMATCH" not in output


def test_check_local_ha_harness_alignment_rejects_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject a local harness that targets a different Home Assistant release."""
    versions = {
        "homeassistant": "2026.8.0b0",
        "pytest-homeassistant-custom-component": "0.13.348",
    }
    monkeypatch.setattr(validate.md, "version", lambda package: versions[package])
    monkeypatch.setattr(
        validate.md,
        "requires",
        lambda _package: ["homeassistant==2026.7.4"],
    )

    with pytest.raises(subprocess.CalledProcessError):
        validate._check_local_ha_harness_alignment()

    output = capsys.readouterr().out
    assert "HARNESS_MISMATCH:" in output
    assert "requires Home Assistant 2026.7.4" in output
    assert "local environment has 2026.8.0b0" in output
    assert "STEP_OK: check local homeassistant/test harness alignment" not in output


@pytest.mark.parametrize(
    "requirements",
    [
        None,
        [],
        ["homeassistant>=2026.7.4"],
        ["homeassistant==2026.7.*"],
    ],
)
def test_check_local_ha_harness_alignment_rejects_invalid_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    requirements: list[str] | None,
) -> None:
    """Reject missing or non-exact Home Assistant requirements from the harness."""
    versions = {
        "homeassistant": "2026.7.4",
        "pytest-homeassistant-custom-component": "0.13.348",
    }
    monkeypatch.setattr(validate.md, "version", lambda package: versions[package])
    monkeypatch.setattr(validate.md, "requires", lambda _package: requirements)

    with pytest.raises(subprocess.CalledProcessError):
        validate._check_local_ha_harness_alignment()

    output = capsys.readouterr().out
    assert "HARNESS_MISMATCH:" in output
    assert "STEP_OK: check local homeassistant/test harness alignment" not in output


def test_check_local_ha_harness_alignment_rejects_missing_package(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject a local environment missing either side of the required pair."""

    def missing_version(_package: str) -> str:
        raise md.PackageNotFoundError("homeassistant")

    monkeypatch.setattr(validate.md, "version", missing_version)

    with pytest.raises(subprocess.CalledProcessError):
        validate._check_local_ha_harness_alignment()

    assert "HARNESS_MISMATCH:" in capsys.readouterr().out


def _create_executable(path: Path) -> None:
    """Create a minimal executable at path."""
    path.parent.mkdir(parents=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def test_resolve_global_uv_path_skips_active_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Select global uv even when the active environment appears first on PATH."""
    environment_root = tmp_path / "environment"
    environment_uv = environment_root / "bin" / "uv"
    global_uv = tmp_path / "global" / "bin" / "uv"
    _create_executable(environment_uv)
    _create_executable(global_uv)
    monkeypatch.setattr(validate.sys, "prefix", str(environment_root))
    monkeypatch.setenv("PATH", os.pathsep.join((str(environment_uv.parent), str(global_uv.parent))))

    assert validate.resolve_global_uv_path() == str(global_uv.resolve())
    assert os.environ["PATH"] == str(global_uv.parent.resolve())
    assert shutil.which("uv") == str(global_uv.resolve())


def test_resolve_global_uv_path_keeps_interpreter_bin_outside_virtual_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not mistake a global interpreter's bin directory for an active venv."""
    interpreter_root = tmp_path / "interpreter"
    global_uv = interpreter_root / "bin" / "uv"
    _create_executable(global_uv)
    monkeypatch.setattr(validate.sys, "base_prefix", str(interpreter_root))
    monkeypatch.setattr(validate.sys, "prefix", str(interpreter_root))
    monkeypatch.setenv("PATH", str(global_uv.parent))

    assert validate.resolve_global_uv_path() == str(global_uv.resolve())
    assert os.environ["PATH"] == str(global_uv.parent.resolve())


@pytest.mark.parametrize(
    "unsafe_entry",
    ["relative/bin", "/safe/../bin", "/safe/$USER/bin", "~/bin"],
)
def test_validate_path_entry_rejects_unsafe_paths(unsafe_entry: str) -> None:
    """Reject relative, traversing, expanded, and unsupported PATH expressions."""
    assert validate._validate_path_entry(unsafe_entry) == ""


def test_validate_path_entry_reconstructs_safe_absolute_path() -> None:
    """Reconstruct ordinary absolute PATH entries from the fixed allowlist."""
    assert validate._validate_path_entry("/home/test-user/.local/bin") == (
        "/home/test-user/.local/bin"
    )


def test_pipeline_exits_when_global_uv_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fail validation before any steps when PATH only contains environment uv."""
    environment_root = tmp_path / "environment"
    environment_uv = environment_root / "bin" / "uv"
    _create_executable(environment_uv)
    monkeypatch.setattr(validate.sys, "prefix", str(environment_root))
    monkeypatch.setenv("PATH", str(environment_uv.parent))

    with pytest.raises(SystemExit, match="1"):
        validate._run_pipeline()

    output = capsys.readouterr().out
    assert "VALIDATION_ERROR: 'global uv executable' not found." in output
    assert "VALIDATION_FAILED" in output


def test_resolve_global_uv_path_rejects_symlink_into_active_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a global PATH entry whose uv symlinks back into the active venv."""
    environment_root = tmp_path / "environment"
    environment_uv = environment_root / "bin" / "uv"
    _create_executable(environment_uv)

    spoofed_dir = tmp_path / "spoofed" / "bin"
    spoofed_dir.mkdir(parents=True)
    (spoofed_dir / "uv").symlink_to(environment_uv)

    monkeypatch.setattr(validate.sys, "prefix", str(environment_root))
    monkeypatch.setenv("PATH", str(spoofed_dir))

    with pytest.raises(FileNotFoundError):
        validate.resolve_global_uv_path()
