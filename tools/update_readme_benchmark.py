"""Script to automatically update benchmark performance in README.md and README.vi.md.

Reads metrics from benchmark/performance_benchmark_report.json and replaces
marked overall and per-language tables/texts in the README files.
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path
from typing import Any, Final

import orjson

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
REPORT_JSON_PATH: Final[Path] = REPO_ROOT / "benchmark" / "performance_benchmark_report.json"
README_EN_PATH: Final[Path] = REPO_ROOT / "README.md"
README_VI_PATH: Final[Path] = REPO_ROOT / "README.vi.md"
BENCHMARK_DEPENDENCIES: Final[tuple[str, ...]] = ("homeassistant", "home-assistant-intents")

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

    # Compute maximum column widths from headers and all data cells
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Expand alignment specifier
    if len(alignments) == 1:
        aligns = [alignments] * ncols
    else:
        aligns = list(alignments)
        if len(aligns) < ncols:
            # Pad with last alignment character to match column count
            aligns.extend([aligns[-1]] * (ncols - len(aligns)))
        aligns = aligns[:ncols]

    hdr = _format_md_table_row(headers, aligns, widths)

    # Markdown separator with alignment colons (colon uses 1 char of width)
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


def _load_report(report_path: Path) -> dict[str, Any]:
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

    return data


def _get_metric_pct(stats: dict[str, Any], key: str) -> float:
    """Extract a percentage metric value from stats dict.

    Args:
        stats: Dictionary containing stats values.
        key: The key to look up.

    Returns:
        The float percentage value.

    Raises:
        KeyError: If the key is not present in stats.
        TypeError: If the value for the key is not numeric.
    """
    if key not in stats:
        raise KeyError(f"Missing metric '{key}' in stats; available keys: {list(stats.keys())}")

    val = stats[key]
    if not isinstance(val, (int, float)):
        raise TypeError(f"Metric '{key}' must be numeric, got {type(val).__name__}: {val!r}")

    return float(val)


def _get_languages(report: dict[str, Any]) -> list[str]:
    """Derive the list of languages from the report structure dynamically.

    Returns the languages sorted with EN first, and all other languages
    sorted alphabetically.

    Args:
        report: Parsed JSON report dict.

    Returns:
        Sorted list of language codes.

    Raises:
        KeyError: If the 'languages' section is missing or empty.
    """
    languages_section = report.get("languages")
    if not isinstance(languages_section, dict) or not languages_section:
        raise KeyError("Missing or empty 'languages' section in report")

    if keys := [k for k in languages_section if isinstance(k, str)]:
        return sorted(
            keys,
            key=lambda lang: "0_en" if lang.lower() == "en" else f"1_{lang.lower()}",
        )
    raise KeyError("No valid string language keys found in 'languages' section")


def _get_dependency_versions(report: dict[str, Any]) -> dict[str, str]:
    """Return benchmark dependency versions from the report metadata."""
    raw_versions = report.get("dependency_versions", {})
    versions = raw_versions if isinstance(raw_versions, dict) else {}
    result: dict[str, str] = {}
    for package_name in BENCHMARK_DEPENDENCIES:
        value = versions.get(package_name)
        result[package_name] = value if isinstance(value, str) and value.strip() else "not recorded"
    return result


def _generate_versions_note(report: dict[str, Any], is_vi: bool) -> str:
    """Generate a localized dependency-version note for benchmark results."""
    versions = _get_dependency_versions(report)
    ha_version = versions["homeassistant"]
    intents_version = versions["home-assistant-intents"]
    if is_vi:
        return (
            f"> Phiên bản phụ thuộc benchmark: `homeassistant` {ha_version}, "
            f"`home-assistant-intents` {intents_version}."
        )
    return (
        f"> Benchmark dependency versions: `homeassistant` {ha_version}, "
        f"`home-assistant-intents` {intents_version}."
    )


def _generate_overall_section(report: dict[str, Any], is_vi: bool) -> str:
    """Generate the overall benchmark table and summary paragraph.

    Args:
        report: Parsed JSON report dict.
        is_vi: True if generating in Vietnamese.

    Returns:
        The formatted markdown snippet for the overall results.
    """
    overall_summary = report.get("overall", {}).get("summary", {})
    hassil_stats = overall_summary.get("hassil", {}).get("overall", {})
    lexical_stats = overall_summary.get("lexical", {}).get("overall", {})

    hass_acc = _get_metric_pct(hassil_stats, "intent_slot_accuracy")
    hass_mis = _get_metric_pct(hassil_stats, "mismatch_rate")
    hass_fall = _get_metric_pct(hassil_stats, "fallback_rate")

    lex_acc = _get_metric_pct(lexical_stats, "intent_slot_accuracy")
    lex_mis = _get_metric_pct(lexical_stats, "mismatch_rate")
    lex_fall = _get_metric_pct(lexical_stats, "fallback_rate")

    hass_err = hass_mis + hass_fall
    lex_err = lex_mis + lex_fall

    # Common data rows, cell values identical for both languages
    data_rows: list[tuple[str, ...]] = [
        ("`hassil`", f"{hass_acc:.1f}%", f"{hass_mis:.1f}%", f"{hass_fall:.1f}%"),
        ("`lexical`", f"**{lex_acc:.1f}%**", f"**{lex_mis:.1f}%**", f"**{lex_fall:.1f}%**"),
    ]

    if is_vi:
        headers = ("Chế độ", "Đúng Intent/Slot", "Nhận diện sai (Mismatch)", "Dự phòng (Fallback)")
        summary_sentence = (
            f"> Độ chính xác nhận diện Intent/Slot tăng từ "
            f"**{hass_acc:.1f}% lên {lex_acc:.1f}%**. "
            f"Tổng tỷ lệ lỗi (nhận diện sai + chuyển sang dự phòng) "
            f"giảm mạnh từ **{hass_err:.1f}% xuống còn {lex_err:.1f}%**."
        )
    else:
        headers = ("Mode", "Intent/Slot", "Mismatch", "Fallback")
        summary_sentence = (
            f"> Intent/slot accuracy jumped from "
            f"**{hass_acc:.1f}% to {lex_acc:.1f}%**. "
            f"The combined error rate (mismatch + fallback) "
            f"dropped from **{hass_err:.1f}% to {lex_err:.1f}%**."
        )

    table = _render_md_table(headers, data_rows, alignments="<>")
    versions_note = _generate_versions_note(report, is_vi=is_vi)
    return f"\n\n{versions_note}\n\n{table}\n\n{summary_sentence}\n\n"


def _generate_langs_section(report: dict[str, Any], is_vi: bool) -> str:
    """Generate the per-language breakdown benchmark table.

    Args:
        report: Parsed JSON report dict.
        is_vi: True if generating in Vietnamese.

    Returns:
        The formatted markdown snippet for the per-language breakdown.
    """
    if is_vi:
        headers = (
            "Ngôn ngữ",
            "Chế độ",
            "Đúng Intent/Slot",
            "Nhận diện sai (Mismatch)",
            "Dự phòng (Fallback)",
        )
    else:
        headers = ("Language", "Mode", "Intent/Slot", "Mismatch", "Fallback")

    data_rows: list[tuple[str, ...]] = []

    languages = _get_languages(report)
    for lang in languages:
        lang_data = report.get("languages", {}).get(lang, {})
        lang_summary = lang_data.get("summary", {})

        hassil_stats = lang_summary.get("hassil", {}).get("overall", {})
        lexical_stats = lang_summary.get("lexical", {}).get("overall", {})

        hass_acc = _get_metric_pct(hassil_stats, "intent_slot_accuracy")
        hass_mis = _get_metric_pct(hassil_stats, "mismatch_rate")
        hass_fall = _get_metric_pct(hassil_stats, "fallback_rate")

        lex_acc = _get_metric_pct(lexical_stats, "intent_slot_accuracy")
        lex_mis = _get_metric_pct(lexical_stats, "mismatch_rate")
        lex_fall = _get_metric_pct(lexical_stats, "fallback_rate")

        lang_upper = lang.upper()

        data_rows.extend(
            (
                (
                    lang_upper,
                    "`hassil`",
                    f"{hass_acc:.1f}%",
                    f"{hass_mis:.1f}%",
                    f"{hass_fall:.1f}%",
                ),
                (
                    lang_upper,
                    "`lexical`",
                    f"**{lex_acc:.1f}%**",
                    f"**{lex_mis:.1f}%**",
                    f"**{lex_fall:.1f}%**",
                ),
            )
        )
    table = _render_md_table(headers, data_rows, alignments="<<>")
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
    try:
        report = _load_report(REPORT_JSON_PATH)

        _update_readme_benchmark(report, False, README_EN_PATH)
        _update_readme_benchmark(report, True, README_VI_PATH)
        print("Benchmark updates completed successfully.")

    except Exception as err:
        print(f"Error updating README files: {err}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1) from err


def _update_readme_benchmark(
    report: dict[str, Any],
    is_vi: bool,
    file_path: Path,
) -> None:
    """Update benchmark content for the target README file."""
    overall = _generate_overall_section(report, is_vi=is_vi)
    langs = _generate_langs_section(report, is_vi=is_vi)
    _update_file(file_path, overall_content=overall, langs_content=langs)


if __name__ == "__main__":
    main()
