"""Tests for the multi-version Home Assistant compatibility runner."""

import subprocess
from pathlib import Path

import pytest

from tools import validate_compatibility

TEST_DEP_VERSIONS = {"home-assistant-intents": "2026.6.1"}
ASSIST_RUNTIME_VERSIONS = {
    "aiodns": "4.0.0",
    "ha-ffmpeg": "3.2.2",
    "mutagen": "1.47.0",
    "pymicro-vad": "1.0.1",
    "pyspeex-noise": "1.0.2",
}


def test_cleanup_compatibility_bytecode_is_scoped_and_missing_safe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Remove target bytecode caches without touching siblings or missing paths."""
    target_dir = tmp_path / "venv"
    target_bytecode = target_dir / "package" / "__pycache__"
    sibling_bytecode = tmp_path / "sibling" / "__pycache__"
    target_bytecode.mkdir(parents=True)
    sibling_bytecode.mkdir(parents=True)
    (target_bytecode / "module.pyc").write_bytes(b"target")
    (sibling_bytecode / "module.pyc").write_bytes(b"sibling")

    validate_compatibility._cleanup_compatibility_bytecode(target_dir)
    successful_output = capsys.readouterr()
    validate_compatibility._cleanup_compatibility_bytecode(tmp_path / "missing")
    missing_output = capsys.readouterr()

    assert not target_bytecode.exists()
    assert sibling_bytecode.is_dir()
    assert "STEP_OK: cleanup __pycache__" in successful_output.out
    assert "STEP_OK: cleanup __pycache__" not in missing_output.out
    assert "target directory does not exist" in missing_output.err


def test_cleanup_compatibility_bytecode_rejects_regular_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject an existing cleanup target that is not a directory."""
    target_file = tmp_path / "venv"
    target_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RuntimeError, match="cleanup target path is not a directory"):
        validate_compatibility._cleanup_compatibility_bytecode(target_file)

    output = capsys.readouterr()
    assert "STEP_OK: cleanup __pycache__" not in output.out
    assert "STEP_FAILED: cleanup __pycache__" in output.err


def test_cleanup_compatibility_bytecode_cleans_siblings_before_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Clean sibling caches before reporting a removal failure."""
    target_dir = tmp_path / "venv"
    blocked_bytecode = target_dir / "blocked" / "__pycache__"
    removable_bytecode = target_dir / "removable" / "__pycache__"
    blocked_bytecode.mkdir(parents=True)
    removable_bytecode.mkdir(parents=True)
    original_rmtree = validate_compatibility.shutil.rmtree

    def remove_bytecode(path: Path) -> None:
        if path == blocked_bytecode:
            raise PermissionError("permission denied")
        original_rmtree(path)

    monkeypatch.setattr(validate_compatibility.shutil, "rmtree", remove_bytecode)

    with pytest.raises(RuntimeError, match="Compatibility bytecode cleanup failed"):
        validate_compatibility._cleanup_compatibility_bytecode(target_dir)

    assert blocked_bytecode.is_dir()
    assert not removable_bytecode.exists()
    output = capsys.readouterr()
    assert "STEP_OK: cleanup __pycache__" not in output.out
    error = output.err
    assert f"unable to remove {blocked_bytecode}" in error
    assert "permission denied" in error
    assert "STEP_FAILED: cleanup __pycache__" in error


def test_cleanup_compatibility_bytecode_timeout_fails_the_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stop a long cleanup at its deadline and report an incomplete step."""
    bytecode_dir = tmp_path / "venv" / "package" / "__pycache__"
    bytecode_dir.mkdir(parents=True)
    final_timestamp = validate_compatibility._CLEANUP_TIMEOUT_SECONDS + 1.0
    timestamps = iter((0.0, final_timestamp))
    monkeypatch.setattr(
        validate_compatibility,
        "monotonic",
        lambda: next(timestamps, final_timestamp),
    )

    with pytest.raises(RuntimeError, match="Compatibility bytecode cleanup failed"):
        validate_compatibility._cleanup_compatibility_bytecode(tmp_path / "venv")

    assert bytecode_dir.is_dir()
    output = capsys.readouterr()
    assert "STEP_OK: cleanup __pycache__" not in output.out
    assert (
        f"timed out after {validate_compatibility._CLEANUP_TIMEOUT_SECONDS} seconds" in output.err
    )
    assert "STEP_FAILED: cleanup __pycache__" in output.err


def test_cleanup_compatibility_bytecode_allows_final_removal_to_cross_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report success when the final cache removal completes after the deadline."""
    removed_bytecode = tmp_path / "venv" / "__pycache__"
    removed_bytecode.mkdir(parents=True)
    final_timestamp = validate_compatibility._CLEANUP_TIMEOUT_SECONDS + 1.0
    current_timestamp = [0.0]
    original_rmtree = validate_compatibility.shutil.rmtree

    def remove_final_bytecode(path: Path) -> None:
        original_rmtree(path)
        current_timestamp[0] = final_timestamp

    monkeypatch.setattr(
        validate_compatibility,
        "monotonic",
        lambda: current_timestamp[0],
    )
    monkeypatch.setattr(validate_compatibility.shutil, "rmtree", remove_final_bytecode)

    validate_compatibility._cleanup_compatibility_bytecode(tmp_path / "venv")

    assert not removed_bytecode.exists()
    output = capsys.readouterr()
    assert "STEP_OK: cleanup __pycache__" in output.out
    assert "timed out after" not in output.err
    assert "STEP_FAILED: cleanup __pycache__" not in output.err


def test_cleanup_compatibility_bytecode_caps_reported_issues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Bound retained cleanup errors while still counting omitted issues."""
    target_dir = tmp_path / "venv"
    issue_count = validate_compatibility._CLEANUP_ISSUE_LIMIT + 2
    for index in range(issue_count):
        (target_dir / f"package_{index}" / "__pycache__").mkdir(parents=True)

    def fail_removal(path: Path) -> None:
        raise PermissionError(f"cannot remove {path}")

    monkeypatch.setattr(validate_compatibility.shutil, "rmtree", fail_removal)

    with pytest.raises(RuntimeError) as raised:
        validate_compatibility._cleanup_compatibility_bytecode(target_dir)

    message = str(raised.value)
    assert message.count("unable to remove") == validate_compatibility._CLEANUP_ISSUE_LIMIT
    assert "2 additional issue(s) omitted" in message


def test_cleanup_compatibility_bytecode_rejects_symlinked_target(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject a symlink root without removing caches from its destination."""
    actual_target = tmp_path / "actual"
    bytecode_dir = actual_target / "package" / "__pycache__"
    bytecode_dir.mkdir(parents=True)
    linked_target = tmp_path / "linked"
    linked_target.symlink_to(actual_target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="refusing to clean symlinked target directory"):
        validate_compatibility._cleanup_compatibility_bytecode(linked_target)

    assert bytecode_dir.is_dir()
    output = capsys.readouterr()
    assert "STEP_OK: cleanup __pycache__" not in output.out
    assert "STEP_FAILED: cleanup __pycache__" in output.err


def test_cleanup_compatibility_bytecode_reports_symlinked_cache(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report a cache symlink without removing its link target."""
    target_dir = tmp_path / "venv"
    cache_target = tmp_path / "external_cache"
    cache_target.mkdir()
    cached_file = cache_target / "module.pyc"
    cached_file.write_bytes(b"external")
    bytecode_link = target_dir / "package" / "__pycache__"
    bytecode_link.parent.mkdir(parents=True)
    bytecode_link.symlink_to(cache_target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="Compatibility bytecode cleanup failed"):
        validate_compatibility._cleanup_compatibility_bytecode(target_dir)

    assert bytecode_link.is_symlink()
    assert cached_file.read_bytes() == b"external"
    output = capsys.readouterr()
    assert f"unable to remove {bytecode_link}" in output.err
    assert "STEP_FAILED: cleanup __pycache__" in output.err


def test_stale_deps_ignore_unpinned_harness_versions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only tracked dependencies should trigger targeted refreshes."""

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

    refresh_specs = validate_compatibility._stale_test_deps(
        Path("python"),
        TEST_DEP_VERSIONS,
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
            **ASSIST_RUNTIME_VERSIONS,
            "hassil": "3.3.0",
            "home-assistant-intents": "2026.6.1",
            "pytest": "8.3.4",
            "pytest-asyncio": "1.3.0",
            "pytest-cov": "7.0.0",
            "pytest-timeout": "2.4.0",
            "pytest-xdist": "3.8.0",
            "rapidfuzz": "3.14.3",
        }

    monkeypatch.setattr(
        validate_compatibility,
        "_venv_required_test_dep_versions",
        fake_versions,
    )

    missing_deps = validate_compatibility._missing_required_test_deps(
        Path("python"),
        TEST_DEP_VERSIONS,
    )

    assert missing_deps == ("pytest-homeassistant-custom-component",)


def test_missing_dep_uses_targeted_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing package should not force a full harness reinstall."""

    def fake_versions(_python_bin: Path, _pinned_versions: dict[str, str]) -> dict[str, str]:
        return {
            **ASSIST_RUNTIME_VERSIONS,
            "hassil": "3.3.0",
            "pytest": "8.3.4",
            "pytest-asyncio": "1.3.0",
            "pytest-cov": "7.0.0",
            "pytest-homeassistant-custom-component": "0.13.205",
            "pytest-timeout": "2.4.0",
            "pytest-xdist": "3.8.0",
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
            TEST_DEP_VERSIONS,
        )
        == ()
    )
    assert validate_compatibility._stale_test_deps(
        Path("python"),
        TEST_DEP_VERSIONS,
    ) == ("home-assistant-intents==2026.6.1",)


def test_required_deps_use_exact_intents_version() -> None:
    """home-assistant-intents should be installed at the exact resolved version."""
    deps = validate_compatibility._required_test_deps({"home-assistant-intents": "2024.12.4"})

    assert "home-assistant-intents==2024.12.4" in deps
    assert "home-assistant-intents" not in deps


def test_required_deps_use_exact_harness_version() -> None:
    """The Home Assistant test harness should use the matrix's exact version."""
    deps = validate_compatibility._required_test_deps(
        {"pytest-homeassistant-custom-component": "0.13.190"}
    )

    assert "pytest-homeassistant-custom-component==0.13.190" in deps
    assert "pytest-homeassistant-custom-component" not in deps


def test_required_deps_pin_assist_runtime_packages() -> None:
    """Assist pipeline runtime packages should use target Home Assistant pins."""
    pins = ASSIST_RUNTIME_VERSIONS

    deps = validate_compatibility._required_test_deps(pins)

    assert all(f"{package}=={version}" in deps for package, version in pins.items())
    assert not set(pins).intersection(deps)


def test_transitive_compatibility_specs_cap_legacy_pycares() -> None:
    """Prevent future pycares majors from breaking historical aiodns releases."""
    assert validate_compatibility._transitive_compatibility_specs({"aiodns": "3.5.0"}) == (
        "pycares<5",
    )
    assert validate_compatibility._transitive_compatibility_specs({"aiodns": "4.0.0b1"}) == ()
    assert validate_compatibility._transitive_compatibility_specs({"aiodns": "4.0.0"}) == ()


def test_refresh_dependencies_preserves_selection_and_legacy_transitive_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply legacy transitive constraints during a targeted dependency refresh."""
    calls: list[tuple[Path, tuple[str, ...], str]] = []

    def record_install(
        python_bin: Path,
        package_args: tuple[str, ...] | list[str],
        step_label: str,
    ) -> None:
        calls.append((python_bin, tuple(package_args), step_label))

    monkeypatch.setattr(validate_compatibility, "_run_uv_pip_install", record_install)
    python_bin = Path("python")
    selected = ("aiodns==3.5.0", "home-assistant-intents==2025.10.1")

    validate_compatibility._refresh_compatibility_dependencies(
        python_bin,
        selected,
        {"aiodns": "3.5.0"},
    )

    assert calls == [
        (
            python_bin,
            (*selected, "pycares<5"),
            "aiodns==3.5.0 home-assistant-intents==2025.10.1",
        )
    ]


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
    assert "dependency marker changed" in output


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
    monkeypatch.setattr(validate_compatibility.subprocess, "run", fake_run)

    venv_path = tmp_path / "venv"
    validate_compatibility._install_dependencies(
        venv_path,
        venv_path / "bin" / "python",
        "2026.6.0",
        True,
        (),
        TEST_DEP_VERSIONS,
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
    assert run_calls[0][5:9] == (
        "--prerelease",
        "allow",
        "--python",
        venv_path / "bin" / "python",
    )


def test_install_dependencies_writes_marker_only_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Leave the dependency marker stale when bytecode cleanup fails."""
    operations: list[str] = []

    monkeypatch.setattr(
        validate_compatibility,
        "_install_compatibility_dependencies",
        lambda *_args: None,
    )

    def fail_cleanup(_venv_path: Path) -> None:
        operations.append("cleanup")
        raise RuntimeError("cleanup failed")

    def record_marker(
        _venv_path: Path,
        _test_dependency_versions: dict[str, str],
    ) -> None:
        operations.append("marker")

    monkeypatch.setattr(validate_compatibility, "_cleanup_compatibility_bytecode", fail_cleanup)
    monkeypatch.setattr(validate_compatibility, "_write_venv_dependency_marker", record_marker)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        validate_compatibility._install_dependencies(
            tmp_path / "venv",
            tmp_path / "venv" / "bin" / "python",
            "2026.6.0",
            True,
            (),
            TEST_DEP_VERSIONS,
            py_ver="3.14",
        )

    assert operations == ["cleanup"]


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
        "_verify_python_version_compatibility",
        lambda _python, _ha: None,
    )
    monkeypatch.setattr(
        validate_compatibility,
        "_resolve_test_dependency_versions",
        lambda _ha, _harness: TEST_DEP_VERSIONS,
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
        _refresh_deps: tuple[str, ...],
        _test_dependency_versions: dict[str, str],
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
    monkeypatch.setattr(validate_compatibility, "_verify_harness_pair", lambda *_args: True)
    monkeypatch.setattr(validate_compatibility, "_run_pytest", lambda *_args: None)

    success, ha_version = validate_compatibility._run_tests_for_version(
        "2026.6.0",
        "0.13.330",
        "3.14",
        True,
    )

    assert success
    assert ha_version == "2026.6.0"
    assert reset_flags == [True]


def test_run_tests_reports_latest_lookup_error_as_row_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Latest-version lookup failures should be reported as one failed matrix row."""

    def fail_latest_lookup() -> validate_compatibility.LatestMatchedPair:
        raise ValueError("PyPI lookup failed")

    monkeypatch.setattr(
        validate_compatibility,
        "_get_latest_matched_pair",
        fail_latest_lookup,
    )

    success, ha_version = validate_compatibility._run_tests_for_version(
        "latest",
        "latest",
        "3.14",
        False,
    )

    assert not success
    assert ha_version == "latest"
    assert "VALIDATION_ERROR: PyPI lookup failed" in capsys.readouterr().out


def test_latest_matched_pair_includes_ha_beta_and_reports_newer_unmatched_ha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The moving gate should follow the newest exact pair and retain the HA edge."""

    def fake_fetch(url: str) -> str:
        if url == validate_compatibility._PYPI_HA_JSON_URL:
            return (
                '{"releases": {'
                '"2026.7.4": [{"filename": "stable.whl"}],'
                '"2026.8.0b0": [{"filename": "matched-beta.whl"}],'
                '"2026.8.0b1": [{"filename": "unmatched-beta.whl"}],'
                '"2026.9.0b0": [],'
                '"invalid-version!": [{"filename": "invalid.whl"}]'
                "}}"
            )
        assert url == validate_compatibility._PYPI_TEST_HARNESS_JSON_URL
        return (
            '{"info": {'
            '"requires_dist": ['
            '"homeassistant==2026.8.0b0",'
            '"pytest>=8.0"'
            "]"
            '}, "releases": {'
            '"0.13.348": [{"filename": "old.whl"}],'
            '"0.13.349": [{"filename": "latest.whl"}]'
            "}}"
        )

    monkeypatch.setattr(validate_compatibility, "_fetch_remote_text", fake_fetch)

    assert validate_compatibility._get_latest_matched_pair() == {
        "ha_ver": "2026.8.0b0",
        "harness_ver": "0.13.349",
        "absolute_latest_ha_ver": "2026.8.0b1",
    }


@pytest.mark.parametrize(
    "requirements",
    [
        [],
        ["homeassistant>=2026.8.0b0"],
        ["homeassistant==2026.8.*"],
        ["homeassistant==2026.8.0b0", "HomeAssistant==2026.8.0b0"],
    ],
)
def test_exact_homeassistant_requirement_rejects_non_exact_metadata(
    requirements: list[str],
) -> None:
    """Harness metadata must expose exactly one non-wildcard Home Assistant pin."""
    with pytest.raises(ValueError, match="Home Assistant"):
        validate_compatibility.exact_homeassistant_requirement(
            requirements,
            "test harness",
        )


def test_prepare_latest_pair_reports_harness_lag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A harness delay should be visible while the matched pair remains testable."""
    monkeypatch.setattr(
        validate_compatibility,
        "_get_latest_matched_pair",
        lambda: validate_compatibility.LatestMatchedPair(
            ha_ver="2026.8.0b0",
            harness_ver="0.13.349",
            absolute_latest_ha_ver="2026.8.0b1",
        ),
    )
    monkeypatch.setattr(
        validate_compatibility,
        "_resolve_test_dependency_versions",
        lambda _ha, _harness: TEST_DEP_VERSIONS,
    )

    assert validate_compatibility._prepare_version_and_deps(
        "latest",
        "latest",
    ) == ("2026.8.0b0", "0.13.349", TEST_DEP_VERSIONS)
    assert "CANARY_LAG: newest Home Assistant 2026.8.0b1" in capsys.readouterr().out


def test_verify_harness_pair_rejects_wrong_target(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The pre-pytest guard should reject a harness pinned to another HA release."""
    monkeypatch.setattr(
        validate_compatibility,
        "_get_installed_harness_pair",
        lambda _python: ("0.13.349", "2026.8.0b0"),
    )

    assert not validate_compatibility._verify_harness_pair(
        Path("python"),
        "2026.8.0b1",
        "0.13.349",
    )
    output = capsys.readouterr().out
    assert "HARNESS_MISMATCH:" in output
    assert "requires Home Assistant 2026.8.0b0" in output


def test_verify_harness_pair_accepts_exact_target(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The pre-pytest guard should accept an exact installed pair."""
    monkeypatch.setattr(
        validate_compatibility,
        "_get_installed_harness_pair",
        lambda _python: ("0.13.349", "2026.8.0b0"),
    )

    assert validate_compatibility._verify_harness_pair(
        Path("python"),
        "2026.8.0b0",
        "0.13.349",
    )
    assert "STEP_OK: harness 0.13.349 matches Home Assistant 2026.8.0b0" in capsys.readouterr().out


def test_verify_harness_pair_accepts_pep440_equivalent_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equivalent normalized versions should not produce harness mismatches."""
    monkeypatch.setattr(
        validate_compatibility,
        "_get_installed_harness_pair",
        lambda _python: ("0.13.349.0", "2026.08.0b0"),
    )

    assert validate_compatibility._verify_harness_pair(
        Path("python"),
        "2026.8.0b0",
        "0.13.349",
    )


def test_get_installed_harness_pair_reports_probe_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose probe output when installed harness metadata cannot be inspected."""
    error = subprocess.CalledProcessError(
        1,
        ["uv", "run"],
        output="probe stdout",
        stderr="probe stderr",
    )

    def fail_probe(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(
        validate_compatibility.subprocess,
        "run",
        fail_probe,
    )

    with pytest.raises(ValueError, match="could not inspect installed test harness") as raised:
        validate_compatibility._get_installed_harness_pair(Path("python"))

    message = str(raised.value)
    assert "probe stdout" in message
    assert "probe stderr" in message


def test_latest_matched_pair_rejects_unpublished_harness_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A harness pin cannot make an unpublished Home Assistant target valid."""

    def fake_fetch(url: str) -> str:
        if url == validate_compatibility._PYPI_HA_JSON_URL:
            return '{"releases": {"2026.8.0b0": [{"filename": "beta.whl"}]}}'
        return (
            '{"info": {"requires_dist": ["homeassistant==2026.8.0b1"]}, '
            '"releases": {"0.13.349": [{"filename": "harness.whl"}]}}'
        )

    monkeypatch.setattr(
        validate_compatibility,
        "_fetch_remote_text",
        fake_fetch,
    )

    with pytest.raises(ValueError, match="targets unpublished Home Assistant"):
        validate_compatibility._get_latest_matched_pair()


def test_compatibility_pytest_args_skip_current_intents_marker() -> None:
    """Compatibility runs should skip current intent corpus tests."""
    assert validate_compatibility._COMPATIBILITY_PYTEST_ARGS == [
        "--no-cov",
        "-m",
        "not current_intents",
    ]


def test_test_matrix_loads_matrix_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compatibility matrix data should be loaded only when the runtime helper is called."""
    monkeypatch.setattr(
        validate_compatibility,
        "_load_matrix_data",
        lambda: [
            {
                "ha_version": "2026.6.0",
                "harness_version": "0.13.330",
                "python_version": "3.14",
            },
            {
                "ha_version": "latest",
                "harness_version": "latest",
                "python_version": "3.14",
            },
        ],
    )
    monkeypatch.setattr(
        validate_compatibility,
        "_minimum_supported_ha_version",
        lambda: "2026.6.0",
    )

    assert validate_compatibility._test_matrix() == [
        {
            "ha_ver": "2026.6.0",
            "harness_ver": "0.13.330",
            "python_ver": "3.14",
        },
        {
            "ha_ver": "latest",
            "harness_ver": "latest",
            "python_ver": "3.14",
        },
    ]


def test_matrix_supported_range_accepts_transition_checkpoints() -> None:
    """Accept ordered API-transition checkpoints and a moving edge."""
    entries: list[validate_compatibility.CompatibilityConfig] = [
        {"ha_ver": "2024.12.0", "harness_ver": "1.0", "python_ver": "3.12"},
        {"ha_ver": "2025.4.4", "harness_ver": "1.1", "python_ver": "3.13"},
        {"ha_ver": "2026.3.4", "harness_ver": "1.2", "python_ver": "3.14"},
        {"ha_ver": "latest", "harness_ver": "latest", "python_ver": "3.14"},
    ]

    validate_compatibility._validate_matrix_supported_range(entries, "2024.12.0")


def test_matrix_supported_range_compares_minimum_semantically() -> None:
    """Accept equivalent minimum versions with different release formatting."""
    entries: list[validate_compatibility.CompatibilityConfig] = [
        {"ha_ver": "2024.5", "harness_ver": "1.0", "python_ver": "3.12"},
        {"ha_ver": "latest", "harness_ver": "latest", "python_ver": "3.14"},
    ]

    validate_compatibility._validate_matrix_supported_range(entries, "2024.5.0")


def test_repository_matrix_tracks_home_assistant_api_transitions() -> None:
    """Keep fixed rows at compatibility-relevant Home Assistant API boundaries."""
    assert [entry["ha_ver"] for entry in validate_compatibility._test_matrix()] == [
        "2024.12.0",  # Declared minimum and legacy conversation input.
        "2025.2.5",  # Extra system prompt before the chat-log lifecycle.
        "2025.4.4",  # ConversationEntity chat-log lifecycle.
        "2025.10.4",  # Satellite-aware ConversationInput.
        "2026.3.4",  # Dynamic intent subscriptions and Python 3.14.
        "2026.4.4",  # Entity-registry alias helper.
        "latest",
    ]


@pytest.mark.parametrize(
    ("entries", "match"),
    [
        (
            [
                {"ha_ver": "2024.12.1", "harness_ver": "1.0", "python_ver": "3.12"},
                {"ha_ver": "latest", "harness_ver": "latest", "python_ver": "3.14"},
            ],
            "first row must match",
        ),
        (
            [
                {"ha_ver": "2024.12.0", "harness_ver": "1.0", "python_ver": "3.12"},
                {"ha_ver": "2024.12.0", "harness_ver": "1.1", "python_ver": "3.12"},
                {"ha_ver": "latest", "harness_ver": "latest", "python_ver": "3.14"},
            ],
            "duplicate fixed",
        ),
        (
            [
                {"ha_ver": "2024.12.0", "harness_ver": "1.0", "python_ver": "3.12"},
            ],
            "must end with",
        ),
        ([], "must not be empty"),
        (
            [
                {"ha_ver": "2024.12.0", "harness_ver": "1.0", "python_ver": "3.12"},
                {"ha_ver": "2026.3.4", "harness_ver": "1.2", "python_ver": "3.14"},
                {"ha_ver": "2025.4.4", "harness_ver": "1.1", "python_ver": "3.13"},
                {"ha_ver": "latest", "harness_ver": "latest", "python_ver": "3.14"},
            ],
            "must be ordered",
        ),
    ],
)
def test_matrix_supported_range_rejects_invalid_structure(
    entries: list[validate_compatibility.CompatibilityConfig],
    match: str,
) -> None:
    """Reject matrix shapes that cannot provide stable transition assurance."""
    with pytest.raises(ValueError, match=match):
        validate_compatibility._validate_matrix_supported_range(entries, "2024.12.0")


@pytest.mark.parametrize(
    ("field_name", "field_value", "match"),
    [
        ("ha_version", "../2026.6.0", "Invalid ha_version"),
        ("harness_version", "0.13..330", "Invalid harness_version"),
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
        "harness_version": "0.13.330",
        "python_version": "3.14",
        field_name: field_value,
    }
    monkeypatch.setattr(validate_compatibility, "_load_matrix_data", lambda: [entry])

    with pytest.raises(ValueError, match=match):
        validate_compatibility._test_matrix()


def test_test_matrix_requires_latest_pair_to_move_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A moving HA row cannot silently use a fixed or independently moving harness."""
    monkeypatch.setattr(
        validate_compatibility,
        "_load_matrix_data",
        lambda: [
            {
                "ha_version": "latest",
                "harness_version": "0.13.349",
                "python_version": "3.14",
            }
        ],
    )

    with pytest.raises(ValueError, match="must both be 'latest' or both be fixed"):
        validate_compatibility._test_matrix()


def test_main_preserves_duplicate_ha_python_matrix_rows(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compatibility summary should not collapse rows that share HA and Python versions."""
    matrix: list[validate_compatibility.CompatibilityConfig] = [
        {
            "ha_ver": "2026.6.0",
            "harness_ver": "0.13.330",
            "python_ver": "3.14",
        },
        {
            "ha_ver": "2026.6.0",
            "harness_ver": "0.13.330",
            "python_ver": "3.14",
        },
    ]
    calls: list[tuple[str, str, str, bool]] = []

    def fake_run_tests(
        ha_ver: str,
        harness_ver: str,
        py_ver: str,
        reinstall: bool,
    ) -> tuple[bool, str]:
        calls.append((ha_ver, harness_ver, py_ver, reinstall))
        return True, ha_ver

    monkeypatch.setattr(validate_compatibility, "_test_matrix", lambda: matrix)
    monkeypatch.setattr(
        validate_compatibility,
        "resolve_global_uv_path",
        lambda: "/usr/bin/uv",
    )
    monkeypatch.setattr(validate_compatibility.sys, "argv", ["validate_compatibility.py"])
    monkeypatch.setattr(validate_compatibility, "_run_tests_for_version", fake_run_tests)

    validate_compatibility.main()

    output = capsys.readouterr().out
    assert len(calls) == 2
    assert (
        "Matrix row 1: Home Assistant 2026.6.0, harness 0.13.330 (Python 3.14): PASSED"
    ) in output
    assert (
        "Matrix row 2: Home Assistant 2026.6.0, harness 0.13.330 (Python 3.14): PASSED"
    ) in output


def test_main_outputs_required_test_dependency_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose the authoritative compatibility dependency metadata for CI."""
    monkeypatch.setattr(
        validate_compatibility.sys,
        "argv",
        [
            "validate_compatibility.py",
            "--required-test-dependency-metadata-json",
        ],
    )

    validate_compatibility.main()

    assert validate_compatibility.orjson.loads(capsys.readouterr().out) == {
        "required_packages": list(validate_compatibility._REQUIRED_TEST_DEPS),
        "homeassistant_constraint_packages": list(validate_compatibility._HA_CONSTRAINED_TEST_DEPS),
        "test_harness_package": validate_compatibility._TEST_HARNESS_PACKAGE,
    }


def test_main_validates_source_matrix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose semantic source-matrix validation as a dedicated CI gate."""
    matrix: list[validate_compatibility.CompatibilityConfig] = [
        {"ha_ver": "2024.12.0", "harness_ver": "0.13.190", "python_ver": "3.12"},
        {"ha_ver": "latest", "harness_ver": "latest", "python_ver": "3.14"},
    ]
    monkeypatch.setattr(validate_compatibility, "_test_matrix", lambda: matrix)
    monkeypatch.setattr(
        validate_compatibility.sys,
        "argv",
        ["validate_compatibility.py", "--validate-matrix"],
    )

    validate_compatibility.main()

    assert capsys.readouterr().out == "STEP_OK: compatibility matrix validated (2 rows)\n"


def test_main_matrix_validation_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Make invalid source matrices fail the dedicated CI gate."""

    def reject_matrix() -> list[validate_compatibility.CompatibilityConfig]:
        raise ValueError("matrix is unordered")

    monkeypatch.setattr(
        validate_compatibility,
        "_test_matrix",
        reject_matrix,
    )
    monkeypatch.setattr(
        validate_compatibility.sys,
        "argv",
        ["validate_compatibility.py", "--validate-matrix"],
    )

    with pytest.raises(SystemExit) as raised:
        validate_compatibility.main()

    assert raised.value.code == 1
    assert capsys.readouterr().out == "VALIDATION_ERROR: matrix is unordered\n"


@pytest.mark.parametrize(
    ("aiodns_version", "expected"),
    [
        ("3.5.0", ["pycares<5"]),
        ("4.0.0b1", []),
        ("4.0.0", []),
        ("4.0.0.post1", []),
    ],
)
def test_main_outputs_transitive_compatibility_specs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    aiodns_version: str,
    expected: list[str],
) -> None:
    """Expose the local runner's transitive compatibility rules to CI."""
    monkeypatch.setattr(
        validate_compatibility.sys,
        "argv",
        [
            "validate_compatibility.py",
            "--transitive-compatibility-specs-json",
            "--aiodns-version",
            aiodns_version,
        ],
    )

    validate_compatibility.main()

    assert validate_compatibility.orjson.loads(capsys.readouterr().out) == expected


def test_verify_pair_requires_uv_before_inspection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject pair verification before invoking a global-uv-backed metadata probe."""
    calls: list[tuple[object, ...]] = []

    def missing_global_uv() -> str:
        raise FileNotFoundError(
            2,
            "global uv executable not found outside the active virtual environment",
            "global uv executable",
        )

    monkeypatch.setattr(
        validate_compatibility,
        "resolve_global_uv_path",
        missing_global_uv,
    )
    monkeypatch.setattr(
        validate_compatibility,
        "_verify_harness_pair",
        lambda *args: calls.append(args) or True,
    )
    monkeypatch.setattr(
        validate_compatibility.sys,
        "argv",
        [
            "validate_compatibility.py",
            "--verify-pair-python",
            "python",
            "--expected-ha",
            "2026.8.0b0",
            "--expected-harness",
            "0.13.349",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        validate_compatibility.main()

    assert raised.value.code == 1
    assert not calls
    assert "VALIDATION_ERROR: 'global uv executable' not found." in capsys.readouterr().out


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


def test_parse_requirements_dependency_version_accepts_pep440_versions() -> None:
    """Allow resolver pins containing PEP 440 epochs, prereleases, and local versions."""
    assert (
        validate_compatibility._parse_requirements_dependency_version(
            "aiodns==1!4.0.0b1+linux.1\n",
            "aiodns",
        )
        == "1!4.0.0b1+linux.1"
    )


def test_parse_requirements_dependency_version_rejects_missing_version_token() -> None:
    """Malformed requirement pins should raise ValueError instead of IndexError."""
    with pytest.raises(ValueError, match="expected a version after '=='"):
        validate_compatibility._parse_requirements_dependency_version(
            "home-assistant-intents== ; python_version >= '3.14'\n",
            "home-assistant-intents",
        )


def test_resolve_test_dependency_versions_uses_ha_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compatibility runner should derive dependencies from the tested HA version."""
    monkeypatch.setattr(
        validate_compatibility,
        "_get_home_assistant_intents_version",
        lambda _ha_ver: "2024.12.4",
    )
    monkeypatch.setattr(
        validate_compatibility,
        "_get_hassil_version",
        lambda _ha_ver: "2.0.5",
    )
    monkeypatch.setattr(
        validate_compatibility,
        "_get_required_package_version",
        lambda _ha_ver, package: ASSIST_RUNTIME_VERSIONS[package],
    )

    assert validate_compatibility._resolve_test_dependency_versions(
        "2024.12.0",
        "0.13.190",
    ) == {
        "home-assistant-intents": "2024.12.4",
        "hassil": "2.0.5",
        "pytest-homeassistant-custom-component": "0.13.190",
        **ASSIST_RUNTIME_VERSIONS,
    }


def test_verify_python_version_compatibility_satisfied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Satisfied requires-python constraints should complete without error."""
    monkeypatch.setattr(
        validate_compatibility,
        "_fetch_remote_text",
        lambda _url: '{"info": {"requires_python": ">=3.14.0"}}',
    )
    monkeypatch.setattr(
        validate_compatibility,
        "_get_python_interpreter_version",
        lambda _bin: "3.14.2",
    )
    validate_compatibility._verify_python_version_compatibility(Path("python"), "2026.3.0")


def test_verify_python_version_compatibility_violated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incompatible Python interpreter versions should raise ValueError."""
    monkeypatch.setattr(
        validate_compatibility,
        "_fetch_remote_text",
        lambda _url: '{"info": {"requires_python": ">=3.14.2"}}',
    )
    monkeypatch.setattr(
        validate_compatibility,
        "_get_python_interpreter_version",
        lambda _bin: "3.14.0",
    )
    with pytest.raises(ValueError, match="does not satisfy"):
        validate_compatibility._verify_python_version_compatibility(Path("python"), "2026.3.0")


def test_verify_python_version_compatibility_raises_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetch or network failures should fail compatibility verification."""

    def fake_fetch(_url: str) -> str:
        raise RuntimeError("DNS resolution disabled in tests")

    monkeypatch.setattr(validate_compatibility, "_fetch_remote_text", fake_fetch)
    with pytest.raises(
        ValueError,
        match="Failed to fetch or verify PyPI requires-python",
    ):
        validate_compatibility._verify_python_version_compatibility(
            Path("python"),
            "2026.3.0",
        )
