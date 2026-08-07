"""Update README benchmark tables from the authoritative managed-live report."""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypeGuard

import orjson

try:
    from .benchmark import BENCHMARK_SCHEMA_VERSION
except ImportError:
    from benchmark import BENCHMARK_SCHEMA_VERSION


def _is_str_mapping(val: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(val, Mapping)


REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
REPORT_JSON_PATH: Final[Path] = REPO_ROOT / "scratch" / "benchmark" / "managed_live_report.json"
README_EN_PATH: Final[Path] = REPO_ROOT / "README.md"
README_VI_PATH: Final[Path] = REPO_ROOT / "README.vi.md"
BENCHMARK_DEPENDENCIES: Final[tuple[str, ...]] = (
    "hassil",
    "home-assistant-intents",
    "homeassistant",
    "python",
)
MANAGED_REPORT_SCHEMA_VERSION: Final[int] = BENCHMARK_SCHEMA_VERSION

OVERALL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(<!-- BENCHMARK_OVERALL_START -->)(.*?)(<!-- BENCHMARK_OVERALL_END -->)", re.DOTALL
)
LANGS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(<!-- BENCHMARK_LANGS_START -->)(.*?)(<!-- BENCHMARK_LANGS_END -->)", re.DOTALL
)


def _format_md_table_row(
    row: tuple[str, ...],
    aligns: list[str],
    widths: list[int],
) -> str:
    """Return one Markdown table row."""
    parts = [f" {c:{a}{w}} " for c, a, w in zip(row, aligns, widths, strict=True)]
    return "|" + "|".join(parts) + "|"


def _render_md_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    *,
    alignments: str = "<",
) -> str:
    """Build a Markdown table with dynamically computed column widths.

    Produces output compatible with Prettier's Markdown table formatting
    (single space between pipe and content on each side).

    Args:
        headers: Column header strings.
        rows: Data rows; each tuple must match *headers* length.
        alignments: Single char for all columns, or one char per column
            (``'<'`` left, ``'>'`` right).

    Returns:
        Markdown table string with properly aligned columns.
    """
    ncols = len(headers)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    if len(alignments) == 1:
        aligns = [alignments] * ncols
    else:
        aligns = list(alignments)
        if len(aligns) < ncols:
            aligns.extend([aligns[-1]] * (ncols - len(aligns)))
        aligns = aligns[:ncols]

    hdr = _format_md_table_row(headers, aligns, widths)

    sep_parts: list[str] = []
    for a, w in zip(aligns, widths, strict=True):
        dashes = w - 1
        if a == ">":
            sep_parts.append(f" {'-' * dashes}: ")
        else:
            sep_parts.append(f" :{'-' * dashes} ")
    sep = "|" + "|".join(sep_parts) + "|"

    data = [_format_md_table_row(row, aligns, widths) for row in rows]
    return "\n".join([hdr, sep, *data])


def _load_report(report_path: Path) -> dict[str, object]:
    """Load and parse the JSON benchmark performance report.

    Args:
        report_path: Path to the JSON benchmark report file.

    Returns:
        The parsed benchmark report dict.

    Raises:
        FileNotFoundError: If the report file does not exist.
        ValueError: If the file is not a valid JSON dictionary.
    """
    if not report_path.is_file():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    data = orjson.loads(report_path.read_bytes())

    if not isinstance(data, dict):
        raise ValueError("Report file must contain a top-level JSON object")
    if data.get("report_schema_version") != MANAGED_REPORT_SCHEMA_VERSION:
        raise ValueError(
            "Report schema is not the current paired managed-live schema: "
            f"{data.get('report_schema_version')!r}"
        )
    if (
        data.get("authoritative") is not True
        or data.get("benchmark_mode") != "managed_live"
        or data.get("execution_tier") != "managed_live"
    ):
        raise ValueError("Report is not an authoritative managed-live report")
    settings = data.get("settings")
    if not isinstance(settings, dict) or settings.get("hassil_baseline") != (
        "paired_original_query_to_live_default_agent"
    ):
        raise ValueError("Report does not contain the paired direct-HassIL baseline")

    return data


def _get_metric_pct(stats: Mapping[str, object], key: str) -> float:
    """Extract a percentage metric value from stats dict."""
    val = stats.get(key)
    if val is None:
        raise KeyError(f"Missing metric '{key}' in stats")
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"Metric '{key}' must be numeric, got {type(val).__name__}: {val!r}")
    return float(val)


def _get_languages(report: Mapping[str, object]) -> list[str]:
    """Derive the list of languages from the report structure dynamically."""
    breakdowns = report.get("breakdowns")
    if not _is_str_mapping(breakdowns):
        raise KeyError("Missing or empty 'breakdowns' section in report")
    languages_section = breakdowns.get("languages")
    if not _is_str_mapping(languages_section) or not languages_section:
        raise KeyError("Missing or empty 'languages' section in report")

    if keys := [k for k in languages_section if isinstance(k, str)]:
        return sorted(
            keys,
            key=lambda lang: "0_en" if lang.lower() == "en" else f"1_{lang.lower()}",
        )
    raise KeyError("No valid string language keys found in 'languages' section")


def _get_dependency_versions(report: Mapping[str, object]) -> dict[str, str]:
    """Return benchmark dependency versions from the report metadata."""
    environment = report.get("environment")
    if not _is_str_mapping(environment):
        raise KeyError("Managed-live report environment is missing")
    raw_dependencies = environment.get("dependencies")
    dependencies = raw_dependencies if _is_str_mapping(raw_dependencies) else {}
    result: dict[str, str] = {}
    for package_name in BENCHMARK_DEPENDENCIES:
        if package_name == "homeassistant":
            value = environment.get("homeassistant_version")
        elif package_name == "python":
            value = environment.get("python_version")
        else:
            package = dependencies.get(package_name)
            value = package.get("version") if _is_str_mapping(package) else None
        result[package_name] = value if isinstance(value, str) and value.strip() else "not recorded"
    return result


def _generate_versions_note(report: Mapping[str, object], is_vi: bool) -> str:
    """Generate a localized dependency-version note for benchmark results."""
    versions = _get_dependency_versions(report)
    ha_version = versions["homeassistant"]
    python_version = versions["python"]
    hassil_version = versions["hassil"]
    intents_version = versions["home-assistant-intents"]
    if is_vi:
        return (
            f"> Phiên bản phụ thuộc benchmark: `Python` {python_version}, "
            f"`homeassistant` {ha_version}, "
            f"`hassil` {hassil_version}, "
            f"`home-assistant-intents` {intents_version}."
        )
    return (
        f"> Benchmark dependency versions: `Python` {python_version}, "
        f"`homeassistant` {ha_version}, "
        f"`hassil` {hassil_version}, "
        f"`home-assistant-intents` {intents_version}."
    )


def _generate_overall_section(report: Mapping[str, object], is_vi: bool) -> str:
    """Generate the shortcut-aware overall benchmark table."""
    summary = report.get("summary")
    if not _is_str_mapping(summary):
        raise KeyError("Managed-live report summary is missing")
    accuracy = _get_metric_pct(summary, "canonicalizer_accuracy_pct")
    hassil_accuracy = _get_metric_pct(summary, "hassil_baseline_accuracy_pct")
    uplift = _get_metric_pct(summary, "accuracy_uplift_pp")
    mismatch = _get_metric_pct(summary, "mismatch_rate_pct")
    fallback = _get_metric_pct(summary, "fallback_rate_pct")
    latency = summary.get("latency_ms")
    if not _is_str_mapping(latency):
        raise KeyError("Managed-live latency summary is missing")
    p50 = _get_metric_pct(latency, "median")
    p95 = _get_metric_pct(latency, "p95")

    data_rows: list[tuple[str, ...]] = [
        (
            "`managed_live`",
            f"**{accuracy:.1f}%**",
            f"{hassil_accuracy:.1f}%",
            f"{uplift:+.1f}",
            str(summary.get("recovered_case_count", 0)),
            str(summary.get("shortcut_protected_case_count", 0)),
            f"{mismatch:.1f}%",
            f"{fallback:.1f}%",
            f"{p50:.1f}",
            f"{p95:.1f}",
        )
    ]

    if is_vi:
        headers = (
            "Chế độ",
            "Assist Canonicalizer",
            "HassIL trực tiếp",
            "Mức tăng (%p)",
            "Khôi phục",
            "Ngăn hồi quy",
            "Nhận diện sai",
            "Dự phòng",
            "P50 ms",
            "P95 ms",
        )
    else:
        headers = (
            "Mode",
            "Assist Canonicalizer",
            "Direct HassIL",
            "Uplift (%p)",
            "Recovered",
            "Regressions prevented",
            "Mismatch",
            "Fallback",
            "P50 ms",
            "P95 ms",
        )

    table = _render_md_table(headers, data_rows, alignments="<>>>>>>>>>")
    versions_note = _generate_versions_note(report, is_vi=is_vi)
    return f"\n\n{versions_note}\n\n{table}\n\n"


def _generate_langs_section(report: Mapping[str, object], is_vi: bool) -> str:
    """Generate shortcut-aware per-language benchmark rows."""
    if is_vi:
        headers = (
            "Ngôn ngữ",
            "Assist Canonicalizer",
            "HassIL trực tiếp",
            "Mức tăng (%p)",
            "Khôi phục",
            "Ngăn hồi quy",
            "Nhận diện sai",
            "Dự phòng",
            "P50 ms",
            "P95 ms",
        )
    else:
        headers = (
            "Language",
            "Assist Canonicalizer",
            "Direct HassIL",
            "Uplift (%p)",
            "Recovered",
            "Regressions prevented",
            "Mismatch",
            "Fallback",
            "P50 ms",
            "P95 ms",
        )

    data_rows: list[tuple[str, ...]] = []

    languages = _get_languages(report)
    for lang in languages:
        breakdowns = report.get("breakdowns")
        if not _is_str_mapping(breakdowns):
            continue
        languages_map = breakdowns.get("languages")
        if not _is_str_mapping(languages_map):
            continue
        lang_data = languages_map.get(lang)
        if not _is_str_mapping(lang_data):
            continue
        latency = lang_data.get("latency_ms")
        if not _is_str_mapping(latency):
            continue
        data_rows.append(
            (
                lang.upper(),
                f"**{_get_metric_pct(lang_data, 'canonicalizer_accuracy_pct'):.1f}%**",
                f"{_get_metric_pct(lang_data, 'hassil_baseline_accuracy_pct'):.1f}%",
                f"{_get_metric_pct(lang_data, 'accuracy_uplift_pp'):+.1f}",
                str(lang_data.get("recovered_case_count", 0)),
                str(lang_data.get("shortcut_protected_case_count", 0)),
                f"{_get_metric_pct(lang_data, 'mismatch_rate_pct'):.1f}%",
                f"{_get_metric_pct(lang_data, 'fallback_rate_pct'):.1f}%",
                f"{_get_metric_pct(latency, 'median'):.1f}",
                f"{_get_metric_pct(latency, 'p95'):.1f}",
            )
        )
    table = _render_md_table(headers, data_rows, alignments="<>>>>>>>>>")
    return "\n\n" + table + "\n\n"


def _update_file(file_path: Path, overall_content: str, langs_content: str) -> None:
    """Update comment blocks in the target README file with new benchmark results.

    Args:
        file_path: Path to the target README file.
        overall_content: The new overall results markdown content.
        langs_content: The new per-language results markdown content.

    Raises:
        FileNotFoundError: If the target file doesn't exist.
        ValueError: If comment markers are missing from the file.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Target file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")

    match_overall = OVERALL_PATTERN.search(content)
    if not match_overall:
        raise ValueError(f"Could not find overall comment markers in {file_path.name}")

    content = content[: match_overall.start(2)] + overall_content + content[match_overall.end(2) :]

    match_langs = LANGS_PATTERN.search(content)
    if not match_langs:
        raise ValueError(f"Could not find language comment markers in {file_path.name}")

    content = content[: match_langs.start(2)] + langs_content + content[match_langs.end(2) :]

    file_path.write_text(content, encoding="utf-8")
    print(f"Successfully updated benchmark results in {file_path.name}")


def main() -> None:
    """Main function to run the benchmark update tool."""
    parser = argparse.ArgumentParser(
        description="Update README benchmark tables from a managed-live JSON report"
    )
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=REPORT_JSON_PATH,
        help="Managed-live JSON report (default: scratch/benchmark/managed_live_report.json)",
    )
    args = parser.parse_args()
    try:
        report = _load_report(args.report)

        _update_readme_benchmark(report, False, README_EN_PATH)
        _update_readme_benchmark(report, True, README_VI_PATH)
        print("Benchmark updates completed successfully.")

    except Exception as err:
        print(f"Error updating README files: {err}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1) from err


def _update_readme_benchmark(
    report: Mapping[str, object],
    is_vi: bool,
    file_path: Path,
) -> None:
    """Update benchmark content for the target README file."""
    overall = _generate_overall_section(report, is_vi=is_vi)
    langs = _generate_langs_section(report, is_vi=is_vi)
    _update_file(file_path, overall_content=overall, langs_content=langs)


if __name__ == "__main__":
    main()
