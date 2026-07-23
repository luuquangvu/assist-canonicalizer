"""Tests for unified validation tools."""

from pathlib import Path

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
