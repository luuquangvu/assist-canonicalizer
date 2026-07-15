"""Tests for the multi-version Home Assistant compatibility runner."""

import contextlib
from pathlib import Path

import pytest

from tools import validate_compatibility

PINNED_INTENTS = {"home-assistant-intents": "2026.6.1"}


def test_stale_pinned_deps_ignore_unpinned_harness_versions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only matrix-pinned dependencies should trigger targeted refreshes."""

    def fake_versions(_python_bin: Path, _pinned_versions: dict[str, str]) -> dict[str, str]:
        return {
            "hassil": "3.3.0",
            "home-assistant-intents": "2024.12.4",
            "pytest": "8.3.4",
            "pytest-homeassistant-custom-component": "0.13.205",
            "rapidfuzz": "3.14.3",
        }

    monkeypatch.setattr(
        validate_compatibility,
        "_venv_required_test_dep_versions",
        fake_versions,
    )

    refresh_specs = validate_compatibility._stale_pinned_test_deps(
        Path("python"),
        PINNED_INTENTS,
    )

    assert refresh_specs == ("home-assistant-intents==2026.6.1",)
    output = capsys.readouterr().out
    assert "home-assistant-intents 2024.12.4 -> 2026.6.1" in output
    assert "pytest-homeassistant-custom-component" not in output
    assert "pytest 8.3.4" not in output


def test_missing_required_deps_request_full_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing harness packages should still make the compatibility venv incomplete."""

    def fake_versions(_python_bin: Path, _pinned_versions: dict[str, str]) -> dict[str, str]:
        return {
            "hassil": "3.3.0",
            "home-assistant-intents": "2026.6.1",
            "pytest": "8.3.4",
            "rapidfuzz": "3.14.3",
        }

    monkeypatch.setattr(
        validate_compatibility,
        "_venv_required_test_dep_versions",
        fake_versions,
    )

    missing_deps = validate_compatibility._missing_required_test_deps(
        Path("python"),
        PINNED_INTENTS,
    )

    assert missing_deps == ("pytest-homeassistant-custom-component",)


def test_missing_pinned_dep_uses_targeted_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing pinned package should not force a full harness reinstall."""

    def fake_versions(_python_bin: Path, _pinned_versions: dict[str, str]) -> dict[str, str]:
        return {
            "hassil": "3.3.0",
            "pytest": "8.3.4",
            "pytest-homeassistant-custom-component": "0.13.205",
            "rapidfuzz": "3.14.3",
        }

    monkeypatch.setattr(
        validate_compatibility,
        "_venv_required_test_dep_versions",
        fake_versions,
    )

    assert (
        validate_compatibility._missing_required_test_deps(
            Path("python"),
            PINNED_INTENTS,
        )
        == ()
    )
    assert validate_compatibility._stale_pinned_test_deps(
        Path("python"),
        PINNED_INTENTS,
    ) == ("home-assistant-intents==2026.6.1",)


def test_required_deps_use_exact_matrix_intents_pin() -> None:
    """Pinned home-assistant-intents should be installed at the exact resolved version."""
    deps = validate_compatibility._required_test_deps({"home-assistant-intents": "2024.12.4"})

    assert "home-assistant-intents==2024.12.4" in deps
    assert "home-assistant-intents" not in deps


def test_venv_dependency_marker_tracks_full_pin_set(tmp_path: Path) -> None:
    """The venv reuse marker should reflect every resolved pinned test dependency."""
    venv_path = tmp_path / "venv"
    venv_path.mkdir()
    pins = {
        "example-dependency": "1.2.3",
        "home-assistant-intents": "2026.6.1",
    }

    assert not validate_compatibility._venv_dependency_marker_matches(venv_path, pins)

    validate_compatibility._write_venv_dependency_marker(venv_path, pins)

    assert validate_compatibility._venv_dependency_marker_matches(
        venv_path,
        {
            "home-assistant-intents": "2026.6.1",
            "example-dependency": "1.2.3",
        },
    )
    assert not validate_compatibility._venv_dependency_marker_matches(
        venv_path,
        {
            "example-dependency": "1.2.4",
            "home-assistant-intents": "2026.6.1",
        },
    )


def test_dependency_marker_mismatch_forces_reinstall(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A stale marker should force a full dependency install before venv reuse."""
    venv_path = tmp_path / "venv"
    venv_path.mkdir()
    validate_compatibility._write_venv_dependency_marker(
        venv_path,
        {"home-assistant-intents": "2026.6.1"},
    )

    assert validate_compatibility._dependency_marker_requires_reinstall(
        False,
        venv_path,
        {"home-assistant-intents": "2026.6.2"},
    )
    assert not validate_compatibility._dependency_marker_requires_reinstall(
        True,
        venv_path,
        {"home-assistant-intents": "2026.6.2"},
    )

    output = capsys.readouterr().out
    assert "dependency pin marker changed" in output


def test_install_dependencies_resets_before_marker_reinstall(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Marker-driven installs should rebuild the venv before installing packages."""
    reset_calls: list[tuple[Path, str]] = []
    run_calls: list[tuple[object, ...]] = []

    def fake_reset_venv(venv_path: Path, py_ver: str) -> None:
        venv_path.mkdir(parents=True, exist_ok=True)
        reset_calls.append((venv_path, py_ver))

    def fake_run(cmd: list[object], **_kwargs: object) -> None:
        run_calls.append(tuple(cmd))

    monkeypatch.setattr(validate_compatibility, "_reset_venv", fake_reset_venv)
    monkeypatch.setattr(
        validate_compatibility,
        "_overrides_file",
        lambda _ha_ver: contextlib.nullcontext("overrides.txt"),
    )
    monkeypatch.setattr(validate_compatibility.subprocess, "run", fake_run)

    venv_path = tmp_path / "venv"
    validate_compatibility._install_dependencies(
        venv_path,
        venv_path / "bin" / "python",
        "2026.6.0",
        True,
        (),
        PINNED_INTENTS,
        py_ver="3.14",
        reset_before_install=True,
    )

    assert reset_calls == [(venv_path, "3.14")]
    assert run_calls[0][:5] == (
        "uv",
        "--no-config",
        "pip",
        "install",
        "--upgrade",
    )


def test_run_tests_resets_before_forced_reinstall(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Forced reinstall paths should rebuild the venv before installing packages."""
    venv_path = tmp_path / "venv"
    bin_path = venv_path / "bin"
    bin_path.mkdir(parents=True)
    (bin_path / "python").write_text("", encoding="utf-8")
    (bin_path / "pytest").write_text("", encoding="utf-8")
    reset_flags: list[bool] = []

    monkeypatch.setattr(validate_compatibility, "_get_venv_path", lambda _ha, _py: str(venv_path))
    monkeypatch.setattr(validate_compatibility, "_ensure_venv", lambda _path, _py: False)
    monkeypatch.setattr(
        validate_compatibility,
        "_resolve_pinned_test_dependency_versions",
        lambda _ha, _deps: PINNED_INTENTS,
    )
    monkeypatch.setattr(validate_compatibility, "_get_installed_ha_version", lambda _py: "2026.6.0")
    monkeypatch.setattr(
        validate_compatibility,
        "_dependency_marker_requires_reinstall",
        lambda _created, _path, _pins: False,
    )
    monkeypatch.setattr(
        validate_compatibility,
        "_determine_dependency_actions",
        lambda reinstall, _created, _python, _pins: (reinstall, ()),
    )

    def fake_install_dependencies(
        _venv_path: Path,
        _python_bin: Path,
        _ha_ver: str,
        _needs_install: bool,
        _pinned_refresh_deps: tuple[str, ...],
        _pinned_test_dependency_versions: dict[str, str],
        *,
        py_ver: str,
        reset_before_install: bool = False,
    ) -> None:
        assert py_ver == "3.14"
        reset_flags.append(reset_before_install)

    monkeypatch.setattr(
        validate_compatibility,
        "_install_dependencies",
        fake_install_dependencies,
    )
    monkeypatch.setattr(validate_compatibility, "_run_pytest", lambda *_args: None)

    success, ha_version = validate_compatibility._run_tests_for_version(
        "2026.6.0",
        "3.14",
        True,
        [],
    )

    assert success
    assert ha_version == "2026.6.0"
    assert reset_flags == [True]


def test_run_tests_reports_latest_lookup_error_as_row_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Latest-version lookup failures should be reported as one failed matrix row."""

    def fail_latest_lookup() -> str:
        raise ValueError("PyPI lookup failed")

    monkeypatch.setattr(
        validate_compatibility,
        "_get_latest_ha_version",
        fail_latest_lookup,
    )

    success, ha_version = validate_compatibility._run_tests_for_version(
        "latest",
        "3.14",
        False,
        [],
    )

    assert not success
    assert ha_version == "latest"
    assert "VALIDATION_ERROR: PyPI lookup failed" in capsys.readouterr().out


def test_compatibility_pytest_args_skip_current_intents_marker() -> None:
    """Compatibility runs should skip current intent corpus tests."""
    assert validate_compatibility._COMPATIBILITY_PYTEST_ARGS == [
        "--no-cov",
        "-m",
        "not current_intents",
    ]


def test_matrix_pinned_dependency_list_supports_future_packages() -> None:
    """Matrix pins should support non-Home Assistant packages without code changes."""
    pinned_versions = validate_compatibility._parse_pinned_test_dependency_versions(
        [
            {
                "package": "example-dependency",
                "version": "1.2.3",
            },
        ]
    )

    assert pinned_versions == {
        "example-dependency": "1.2.3",
    }
    assert "example-dependency==1.2.3" in validate_compatibility._required_test_deps(
        pinned_versions
    )


def test_matrix_specific_dependency_packages_do_not_mutate_required_state() -> None:
    """Keep future package discovery isolated to the matrix row that requested it."""
    required_before = validate_compatibility._REQUIRED_TEST_DEPS

    with_extra = validate_compatibility._test_dep_packages({"example-dependency": "1.2.3"})
    without_extra = validate_compatibility._test_dep_packages({})

    assert with_extra == (*required_before, "example-dependency")
    assert without_extra == required_before
    assert validate_compatibility._REQUIRED_TEST_DEPS is required_before


def test_matrix_pinned_dependency_list_rejects_homeassistant() -> None:
    """Home Assistant must be constrained only by the matrix ha_version field."""
    with pytest.raises(ValueError, match="must not pin package 'homeassistant'"):
        validate_compatibility._parse_pinned_test_dependency_versions(
            [{"package": "homeassistant", "version": "2026.6.0"}]
        )


def test_test_matrix_loads_matrix_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compatibility matrix data should be loaded only when the runtime helper is called."""
    monkeypatch.setattr(
        validate_compatibility,
        "_load_matrix_data",
        lambda: [
            {
                "ha_version": "2026.6.0",
                "python_version": "3.14",
                "pinned_test_dependencies": [],
            }
        ],
    )

    assert validate_compatibility._test_matrix() == [
        {
            "ha_ver": "2026.6.0",
            "python_ver": "3.14",
            "pinned_test_dependencies": [],
        }
    ]


@pytest.mark.parametrize(
    ("field_name", "field_value", "match"),
    [
        ("ha_version", "../2026.6.0", "Invalid ha_version"),
        ("python_version", "3..14", "Invalid python_version"),
    ],
)
def test_test_matrix_validates_version_fields(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    field_value: str,
    match: str,
) -> None:
    """Compatibility matrix version fields should be sanitized before storage."""
    entry: dict[str, object] = {
        "ha_version": "2026.6.0",
        "python_version": "3.14",
        "pinned_test_dependencies": [],
        field_name: field_value,
    }
    monkeypatch.setattr(validate_compatibility, "_load_matrix_data", lambda: [entry])

    with pytest.raises(ValueError, match=match):
        validate_compatibility._test_matrix()


def test_main_preserves_duplicate_ha_python_matrix_rows(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compatibility summary should not collapse rows that share HA and Python versions."""
    matrix: list[validate_compatibility.CompatibilityConfig] = [
        {
            "ha_ver": "2026.6.0",
            "python_ver": "3.14",
            "pinned_test_dependencies": [
                {"package": "home-assistant-intents", "version": "2026.6.1"}
            ],
        },
        {
            "ha_ver": "2026.6.0",
            "python_ver": "3.14",
            "pinned_test_dependencies": [
                {"package": "home-assistant-intents", "version": "2026.6.2"}
            ],
        },
    ]
    calls: list[tuple[str, str, bool, object]] = []

    def fake_run_tests(
        ha_ver: str,
        py_ver: str,
        reinstall: bool,
        pinned_test_dependencies: object,
    ) -> tuple[bool, str]:
        calls.append((ha_ver, py_ver, reinstall, pinned_test_dependencies))
        return True, ha_ver

    monkeypatch.setattr(validate_compatibility, "_test_matrix", lambda: matrix)
    monkeypatch.setattr(validate_compatibility.shutil, "which", lambda _cmd: "/usr/bin/uv")
    monkeypatch.setattr(validate_compatibility.sys, "argv", ["validate_compatibility.py"])
    monkeypatch.setattr(validate_compatibility, "_run_tests_for_version", fake_run_tests)

    validate_compatibility.main()

    output = capsys.readouterr().out
    assert len(calls) == 2
    assert "Matrix row 1: Home Assistant 2026.6.0 (Python 3.14): PASSED" in output
    assert "Matrix row 2: Home Assistant 2026.6.0 (Python 3.14): PASSED" in output


def test_parse_requirements_dependency_version_reads_home_assistant_intents() -> None:
    """Home Assistant requirements should define the exact intents corpus version."""
    assert (
        validate_compatibility._parse_requirements_dependency_version(
            "other-package==1.0.0\nhome-assistant-intents==2026.6.1\n",
            "home-assistant-intents",
        )
        == "2026.6.1"
    )


def test_parse_requirements_dependency_version_normalizes_distribution_names() -> None:
    """Requirement package names should match their canonical distribution form."""
    assert (
        validate_compatibility._parse_requirements_dependency_version(
            "home_assistant_intents==2026.6.1 ; python_version >= '3.14'\n",
            "Home.Assistant_Intents",
        )
        == "2026.6.1"
    )


def test_parse_requirements_dependency_version_rejects_missing_version_token() -> None:
    """Malformed requirement pins should raise ValueError instead of IndexError."""
    with pytest.raises(ValueError, match="expected a version after '=='"):
        validate_compatibility._parse_requirements_dependency_version(
            "home-assistant-intents== ; python_version >= '3.14'\n",
            "home-assistant-intents",
        )


def test_resolve_pinned_test_dependency_versions_uses_ha_requirements_when_unpinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compatibility runner should derive unpinned intents from the tested HA version."""
    monkeypatch.setattr(
        validate_compatibility,
        "_get_home_assistant_intents_version",
        lambda _ha_ver: "2024.12.4",
    )

    assert validate_compatibility._resolve_pinned_test_dependency_versions(
        "2024.12.0",
        [{"package": "example-dependency", "version": "1.2.3"}],
    ) == {
        "example-dependency": "1.2.3",
        "home-assistant-intents": "2024.12.4",
    }


def test_resolve_pinned_test_dependency_versions_prefers_matrix_intents_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matrix pins should override any inferred Home Assistant intents version."""

    def fail_fetch(_ha_ver: str) -> str:
        raise AssertionError("home-assistant-intents should not be fetched when pinned")

    monkeypatch.setattr(
        validate_compatibility,
        "_get_home_assistant_intents_version",
        fail_fetch,
    )

    assert validate_compatibility._resolve_pinned_test_dependency_versions(
        "2024.12.0",
        [{"package": "home-assistant-intents", "version": "2026.6.1"}],
    ) == {"home-assistant-intents": "2026.6.1"}
