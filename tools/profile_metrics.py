"""Algorithmic performance profiling for Assist Canonicalizer.

Official performance measurement tool for evaluating algorithmic throughput,
latency, memory usage, and scoring component efficiency across the
canonicalization pipeline. Supports multi-iteration statistical analysis,
per-phase timing, micro-benchmarking of individual scoring components,
baseline regression detection, and multi-format output (terminal, JSON,
Markdown, plain text).

Targets
-------
* ``evaluate``   — Profile the full :func:`~tools.evaluate_metrics.run_evaluation` pipeline.
* ``build_index``— Profile only :func:`.indexer.build_index` and the
  :func:`~.indexer.CanonicalIndex.__post_init__` prebuild phase.
* ``rank``       — Profile :func:`.ranking.rank_candidates` (the core hot path).
* ``components`` — Micro-benchmark individual scoring functions in isolation.
* ``all``        — Run all targets sequentially.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import io
import json
import math
import os
import pstats
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path bootstrap — add repository root so project imports resolve
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Lazy imports — resolved once by _bootstrap_imports()
_BOOTSTRAPPED = False
align_table: Any = None
CanonicalizerRuntime: Any = None
build_candidates_from_intent_sources: Any = None
build_index: Any = None
load_language_intent_sources: Any = None
normalize_text: Any = None
normalize_text_no_diacritics: Any = None
char_ngrams_normalized: Any = None
RankedCandidate: Any = None
ScoreBreakdown: Any = None
Candidate: Any = None
rank_candidates: Any = None
BM25Index: Any = None
CharNGramIndex: Any = None
rapidfuzz_similarity_normalized: Any = None
lexical_score: Any = None
REGISTRY_SLOTS: Any = None
run_evaluation: Any = None
sanitize_path_required: Any = None
atomic_write: Any = None
discover_datasets: Any = None


def _bootstrap_imports() -> None:
    """Import project modules after adding the repository root to sys.path."""
    global _BOOTSTRAPPED
    global align_table
    global CanonicalizerRuntime, build_candidates_from_intent_sources, build_index
    global load_language_intent_sources, normalize_text, normalize_text_no_diacritics
    global char_ngrams_normalized, RankedCandidate, ScoreBreakdown, Candidate
    global rank_candidates, BM25Index, CharNGramIndex
    global rapidfuzz_similarity_normalized, lexical_score, REGISTRY_SLOTS, run_evaluation
    global sanitize_path_required, atomic_write, discover_datasets

    if _BOOTSTRAPPED:
        return

    from custom_components.assist_canonicalizer.bm25 import BM25Index as _BM25Index
    from custom_components.assist_canonicalizer.builtin_intents import (
        load_language_intent_sources as _load_src,
    )
    from custom_components.assist_canonicalizer.candidate import Candidate as _Candidate
    from custom_components.assist_canonicalizer.grammar_loader import (
        build_candidates_from_intent_sources as _build_cands,
    )
    from custom_components.assist_canonicalizer.indexer import (
        build_index as _build_idx,
    )
    from custom_components.assist_canonicalizer.normalization import (
        char_ngrams_normalized as _char_ngrams,
    )
    from custom_components.assist_canonicalizer.normalization import (
        normalize_text as _normalize_text,
    )
    from custom_components.assist_canonicalizer.normalization import (
        normalize_text_no_diacritics as _normalize_no_diac,
    )
    from custom_components.assist_canonicalizer.ranking import (
        CharNGramIndex as _CharNGramIndex,
    )
    from custom_components.assist_canonicalizer.ranking import (
        RankedCandidate as _RankedCandidate,
    )
    from custom_components.assist_canonicalizer.ranking import (
        ScoreBreakdown as _ScoreBreakdown,
    )
    from custom_components.assist_canonicalizer.ranking import (
        lexical_score as _lexical_score,
    )
    from custom_components.assist_canonicalizer.ranking import (
        rank_candidates as _rank_candidates,
    )
    from custom_components.assist_canonicalizer.ranking import (
        rapidfuzz_similarity_normalized as _rf_sim,
    )
    from custom_components.assist_canonicalizer.runtime import (
        CanonicalizerRuntime as _CanonicalizerRuntime,
    )
    from tools.evaluate_metrics import REGISTRY_SLOTS as _REGISTRY_SLOTS
    from tools.evaluate_metrics import align_table as _align_table
    from tools.evaluate_metrics import atomic_write as _aw
    from tools.evaluate_metrics import discover_datasets as _discover
    from tools.evaluate_metrics import run_evaluation as _run_eval
    from tools.evaluate_metrics import sanitize_path_required as _sanitize_req

    BM25Index = _BM25Index
    load_language_intent_sources = _load_src
    Candidate = _Candidate
    build_candidates_from_intent_sources = _build_cands
    build_index = _build_idx
    char_ngrams_normalized = _char_ngrams
    normalize_text = _normalize_text
    normalize_text_no_diacritics = _normalize_no_diac
    CharNGramIndex = _CharNGramIndex
    RankedCandidate = _RankedCandidate
    ScoreBreakdown = _ScoreBreakdown
    lexical_score = _lexical_score
    rank_candidates = _rank_candidates
    rapidfuzz_similarity_normalized = _rf_sim
    CanonicalizerRuntime = _CanonicalizerRuntime
    REGISTRY_SLOTS = _REGISTRY_SLOTS
    atomic_write = _aw
    align_table = _align_table
    discover_datasets = _discover
    run_evaluation = _run_eval
    sanitize_path_required = _sanitize_req

    _BOOTSTRAPPED = True


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Statistical defaults chosen for stable profiling:
# - 10 iterations → reliable percentile (p95/p99) and stddev
# - 3 warmup runs → page cache, __pycache__, and GC patterns stabilize
# - 10 % regression threshold → catches meaningful regressions while
#   tolerating typical run-to-run variance (2-5 % stddev)
DEFAULT_ITERATIONS = 10
DEFAULT_WARMUP = 3
DEFAULT_GRANULARITY = "medium"
DEFAULT_MAX_REGRESSION_PCT = 10.0
BENCHMARK_DIR = "scratch/profile"
BASELINE_DIR = "scratch/profile/baseline"


PROFILING_TARGETS = ("evaluate", "build_index", "rank", "components", "all")
GRANULARITY_LEVELS = ("coarse", "medium", "fine")

SCORING_COMPONENT_NAMES = (
    "normalize_text",
    "normalize_text_no_diacritics",
    "char_ngrams_normalized",
    "bm25_score",
    "char_ngram_score",
    "rapidfuzz_similarity",
    "exact_intent_score",
    "positional_intent_score",
    "lexical_score",
    "intent_disambiguation",
)

# ---------------------------------------------------------------------------
# StatsEngine
# ---------------------------------------------------------------------------


@dataclass
class StatsResult:
    """Aggregated statistics for a metric across iterations."""

    mean: float
    median: float
    p50: float
    p95: float
    p99: float
    stddev: float
    min_val: float
    max_val: float
    cov: float  # coefficient of variation percentage
    raw_values: list[float] = field(default_factory=list, repr=False)


class StatsEngine:
    """Statistical aggregation for multi-iteration profiling data."""

    @staticmethod
    def compute(values: list[float]) -> StatsResult:
        """Compute aggregate statistics for a list of measurement values."""
        if not values:
            return StatsResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean = statistics.mean(sorted_vals)
        median = statistics.median(sorted_vals)
        stddev = statistics.stdev(sorted_vals) if n >= 2 else 0.0
        cov = (stddev / mean * 100.0) if mean > 0 else 0.0

        def _percentile(p: float) -> float:
            """Linear-interpolation percentile function."""
            k = (n - 1) * p / 100.0
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_vals[int(k)]
            lower = sorted_vals[f]
            upper = sorted_vals[c] if c < n else sorted_vals[-1]
            return lower + (upper - lower) * (k - f)

        return StatsResult(
            mean=mean,
            median=median,
            p50=_percentile(50),
            p95=_percentile(95),
            p99=_percentile(99),
            stddev=stddev,
            min_val=sorted_vals[0],
            max_val=sorted_vals[-1],
            cov=cov,
            raw_values=list(values),
        )

    @staticmethod
    def as_dict(result: StatsResult) -> dict[str, Any]:
        """Serialize a :class:`StatsResult` to a JSON-compatible dictionary."""
        return {
            "mean": round(result.mean, 6),
            "median": round(result.median, 6),
            "p50": round(result.p50, 6),
            "p95": round(result.p95, 6),
            "p99": round(result.p99, 6),
            "stddev": round(result.stddev, 6),
            "min": round(result.min_val, 6),
            "max": round(result.max_val, 6),
            "cov_pct": round(result.cov, 1),
        }


# ---------------------------------------------------------------------------
# PhaseTimer
# ---------------------------------------------------------------------------


class _PhaseContext:
    """Context manager returned by :meth:`PhaseTimer.phase`."""

    def __init__(self, timer: PhaseTimer, name: str) -> None:
        """Store the parent timer and phase name for later use."""
        self._timer = timer
        self._name = name

    def __enter__(self) -> _PhaseContext:
        """Start the phase on the parent timer and return self."""
        self._timer.start(self._name)
        return self

    def __exit__(self, *args: object) -> None:
        """Stop the current phase on the parent timer."""
        self._timer.stop()


class PhaseTimer:
    """Hierarchical phase timing with memory delta tracking.

    Supports nested phases via a stack. Each call to :meth:`start` pushes
    a new phase; :meth:`stop` pops it and records elapsed wall-clock time
    and RSS memory delta.
    """

    def __init__(self, resource_monitor: ResourceMonitor | None = None) -> None:
        """Initialize an empty phase timer."""
        self.phases: dict[str, list[float]] = {}
        self.memory_deltas: dict[str, list[float]] = {}
        self.current_phase: str | None = None
        self._stack: list[tuple[str, float, float]] = []
        self._monitor = resource_monitor

    def _current_rss(self) -> float:
        """Return current RSS in MB."""
        if self._monitor is not None:
            return self._monitor.current_rss_mb
        try:
            with open("/proc/self/stat", "rb") as f:
                parts = f.read().decode().split()
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(parts[23]) * page_size / (1024 * 1024)
        except Exception:
            return 0.0

    def start(self, name: str) -> None:
        """Start timing a named phase."""
        rss = self._current_rss()
        self.current_phase = name
        self._stack.append((name, time.perf_counter(), rss))

    def stop(self) -> None:
        """Stop the current phase and record results."""
        if not self._stack:
            return
        name, start_time, start_rss = self._stack.pop()
        elapsed = time.perf_counter() - start_time
        rss_delta = max(0.0, self._current_rss() - start_rss)
        self.phases.setdefault(name, []).append(elapsed)
        self.memory_deltas.setdefault(name, []).append(rss_delta)
        self.current_phase = self._stack[-1][0] if self._stack else None

    def phase(self, name: str) -> _PhaseContext:
        """Return a context manager for the named phase.

        Usage::

            with timer.phase("my_phase"):
                do_work()
        """
        return _PhaseContext(self, name)

    def record(self, name: str, elapsed: float, rss_delta: float = 0.0) -> None:
        """Record a measurement directly without push/pop."""
        self.phases.setdefault(name, []).append(elapsed)
        self.memory_deltas.setdefault(name, []).append(rss_delta)

    def stats(self) -> dict[str, dict[str, StatsResult]]:
        """Return statistical summaries of all recorded phases."""
        result: dict[str, dict[str, StatsResult]] = {}
        for name in self.phases:
            result[name] = {
                "elapsed": StatsEngine.compute(self.phases[name]),
                "memory_delta_mb": StatsEngine.compute(self.memory_deltas.get(name, [0.0])),
            }
        return result


# ---------------------------------------------------------------------------
# ResourceMonitor
# ---------------------------------------------------------------------------


class ResourceMonitor(threading.Thread):
    """Monitors CPU, memory (RSS, VmSize, VmPeak), and GC of the current process.

    Samples ``/proc/self/stat`` and ``/proc/self/status`` at a configurable
    interval.  Runs as a daemon thread so it terminates when the main thread
    exits.
    """

    def __init__(self, interval: float = 0.02) -> None:
        """Initialize the resource monitoring thread.

        Args:
            interval: Sampling interval in seconds.
        """
        super().__init__(daemon=True)
        self.interval: float = interval
        self.stop_event: threading.Event = threading.Event()
        self._lock: threading.Lock = threading.Lock()

        # CPU tracking
        self.cpu_samples: list[tuple[float, float]] = []

        # Memory tracking (sampled)
        self.rss_samples: list[float] = []
        self.vm_size_samples: list[float] = []
        self.vm_peak_samples: list[float] = []

        # Atomic current values
        self.current_rss_mb: float = 0.0
        self.current_vm_size_mb: float = 0.0
        self.current_vm_peak_mb: float = 0.0

        # GC snapshots
        self.gc_snapshots: list[dict[str, Any]] = []

        # clk_tck
        try:
            self.clk_tck: float = float(os.sysconf("SC_CLK_TCK"))
        except Exception:
            self.clk_tck = 100.0

        # page size
        try:
            self.page_size: int = os.sysconf("SC_PAGE_SIZE")
        except Exception:
            self.page_size = 4096

    def run(self) -> None:
        """Sample resource metrics periodically."""
        while not self.stop_event.is_set():
            try:
                t = time.perf_counter()
                with open("/proc/self/stat", "rb") as f:
                    stat_line = f.read().decode("utf-8")
                parts = stat_line.split()
                # field 14 (index 13): utime, field 15 (index 14): stime
                utime = int(parts[13])
                stime = int(parts[14])
                rss_pages = int(parts[23])
                rss_mb = (rss_pages * self.page_size) / (1024 * 1024)

                vm_size = 0
                vm_peak = 0
                try:
                    with open("/proc/self/status") as sf:
                        for line in sf:
                            if line.startswith("VmSize:"):
                                vm_size = int(line.split()[1])
                            elif line.startswith("VmPeak:"):
                                vm_peak = int(line.split()[1])
                except OSError:
                    pass

                with self._lock:
                    self.cpu_samples.append((t, float(utime + stime)))
                    self.rss_samples.append(rss_mb)
                    self.vm_size_samples.append(vm_size / 1024.0)
                    self.vm_peak_samples.append(vm_peak / 1024.0)
                    self.current_rss_mb = rss_mb
                    self.current_vm_size_mb = vm_size / 1024.0
                    self.current_vm_peak_mb = vm_peak / 1024.0
            except Exception:
                pass
            time.sleep(self.interval)

    def stop_monitor(self) -> None:
        """Signal the monitor thread to stop."""
        self.stop_event.set()

    def snapshot_gc(self) -> None:
        """Capture and store current GC statistics."""
        try:
            gc_stats = gc.get_stats()
            self.gc_snapshots.append(
                {
                    "timestamp": time.perf_counter(),
                    "generations": [
                        {
                            "collections": gen.get("collections", 0),
                            "collected": gen.get("collected", 0),
                            "uncollectable": gen.get("uncollectable", 0),
                        }
                        for gen in gc_stats
                    ],
                    "gc_enabled": gc.isenabled(),
                }
            )
        except Exception:
            pass

    def get_cpu_metrics(self) -> dict[str, float]:
        """Compute CPU usage metrics from samples."""
        with self._lock:
            if len(self.cpu_samples) < 2:
                return {"avg_pct": 0.0, "peak_pct": 0.0}
            percentages: list[float] = []
            for i in range(1, len(self.cpu_samples)):
                t1, ticks1 = self.cpu_samples[i - 1]
                t2, ticks2 = self.cpu_samples[i]
                dt = t2 - t1
                dticks = ticks2 - ticks1
                if dt > 0:
                    pct = (dticks / self.clk_tck) / dt * 100.0
                    percentages.append(pct)
            return {
                "avg_pct": sum(percentages) / len(percentages) if percentages else 0.0,
                "peak_pct": max(percentages) if percentages else 0.0,
            }

    def get_memory_metrics(self) -> dict[str, float]:
        """Compute memory metrics from samples."""
        with self._lock:
            return {
                "rss_avg_mb": sum(self.rss_samples) / len(self.rss_samples)
                if self.rss_samples
                else 0.0,
                "rss_peak_mb": max(self.rss_samples) if self.rss_samples else 0.0,
                "vm_size_avg_mb": sum(self.vm_size_samples) / len(self.vm_size_samples)
                if self.vm_size_samples
                else 0.0,
                "vm_size_peak_mb": max(self.vm_size_samples) if self.vm_size_samples else 0.0,
                "vm_peak_mb": max(self.vm_peak_samples) if self.vm_peak_samples else 0.0,
            }


# ---------------------------------------------------------------------------
# BaselineManager
# ---------------------------------------------------------------------------


class BaselineManager:
    """Load, save, and compare performance baselines for regression detection."""

    def __init__(self, repo_root: str) -> None:
        """Initialize with the repository root path."""
        self._baseline_dir: Path = Path(repo_root) / BASELINE_DIR

    def load(self, target: str, *, warn_on_missing: bool = False) -> dict[str, Any] | None:
        """Load a baseline JSON file for *target*.

        Args:
            target: Profiling target name (e.g. ``rank``).
            warn_on_missing: If ``True``, emit a warning to stderr when the
                baseline file is missing, corrupted, or unreadable (intended
                for when ``--baseline`` was explicitly passed by the user).
        """
        path = self._baseline_dir / f"{target}_baseline.json"
        if not path.is_file():
            if warn_on_missing:
                print(f"Warning: baseline file not found: {path}", file=sys.stderr)
            return None
        try:
            data_str = path.read_text(encoding="utf-8")
        except OSError:
            if warn_on_missing:
                print(f"Warning: cannot read baseline file: {path}", file=sys.stderr)
            return None
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            if warn_on_missing:
                print(f"Warning: baseline file is not valid JSON: {path}", file=sys.stderr)
            return None
        if not isinstance(data, dict):
            if warn_on_missing:
                print(
                    f"Warning: baseline file content is not a JSON object: {path}",
                    file=sys.stderr,
                )
            return None
        return data

    def save(self, target: str, data: dict[str, Any]) -> None:
        """Save *data* as the new baseline for *target*."""
        self._baseline_dir.mkdir(parents=True, exist_ok=True)
        path = self._baseline_dir / f"{target}_baseline.json"
        atomic_write(str(path), json.dumps(data, indent=2) + "\n")
        print(f"Baseline saved to {path}")

    def compare(
        self,
        target: str,
        current: dict[str, Any],
        max_regression_pct: float = DEFAULT_MAX_REGRESSION_PCT,
        *,
        warn_on_missing: bool = False,
    ) -> list[str]:
        """Compare current metrics against baseline; return regression messages."""
        baseline = self.load(target, warn_on_missing=warn_on_missing)
        if baseline is None:
            return []

        regressions: list[str] = []

        def _check(key_path: str, cur: float, base: float, label: str) -> None:
            if base == 0:
                return
            pct_change = (cur - base) / base * 100.0
            if pct_change > max_regression_pct:
                regressions.append(
                    f"REGRESSION [{label}]: {pct_change:+.1f}% "
                    f"(baseline={base:.4f}, current={cur:.4f})"
                )

        # Recursively compare per-metric stats
        for key in current:
            if key in baseline:
                cur_val = current[key]
                base_val = baseline[key]
                if isinstance(cur_val, dict) and isinstance(base_val, dict):
                    for metric in ("mean", "median", "p95", "p99", "max", "min"):
                        if metric in cur_val and metric in base_val:
                            _check(
                                f"{key}.{metric}",
                                float(cur_val[metric]),
                                float(base_val[metric]),
                                f"{key}.{metric}",
                            )
                elif isinstance(cur_val, (int, float)) and isinstance(base_val, (int, float)):
                    _check(key, float(cur_val), float(base_val), key)

        return regressions


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Generate multi-format profiling reports (terminal, JSON, Markdown, text)."""

    @staticmethod
    def terminal(report: dict[str, Any]) -> None:
        """Print a formatted report to the terminal."""
        print("\n" + "=" * 90)
        print("ALGORITHMIC PERFORMANCE PROFILING REPORT")
        print("=" * 90)
        print(f"Target:          {report.get('target', 'unknown')}")
        print(f"Iterations:      {report.get('iterations', 0)}")
        print(f"Warmup:          {report.get('warmup', 0)}")
        print(f"Granularity:     {report.get('granularity', 'coarse')}")
        langs = report.get("languages", [])
        print(f"Languages:       {', '.join(langs) if langs else 'all'}")
        print("-" * 90)

        # Aggregate
        agg = report.get("aggregate", {})
        if agg:
            _print_stat_block("Aggregate Performance", agg)

        # Resource
        res = report.get("resource", {})
        if res:
            _print_resource_block("Resource Utilization", res)

        # Phases
        phases = report.get("phases", {})
        if phases:
            _print_phase_table("Phase Timing Breakdown", phases)

        # Components
        components = report.get("components", {})
        if components:
            _print_phase_table("Scoring Component Micro-Profile", components)

        # Per-language
        per_lang = report.get("per_language", {})
        for lang_key in sorted(per_lang):
            lang_data = per_lang[lang_key]
            print(f"\n{'─' * 90}")
            print(f"Language: {lang_key.upper()}")
            if lang_data.get("aggregate"):
                _print_stat_block("  Aggregate", lang_data["aggregate"])
            if lang_data.get("phases"):
                _print_phase_table("  Phase Timing", lang_data["phases"])

        # Regressions
        regressions = report.get("regressions", [])
        if regressions:
            print(f"\n{'!' * 90}")
            print("REGRESSION DETECTIONS:")
            for r in regressions:
                print(f"  {r}")
            print(f"{'!' * 90}")

        stability = report.get("stability", {})
        if stability:
            print(f"\nStability: {stability}")

        print("\n" + "=" * 90)
        print("PROFILE_OK")
        print("=" * 90)

    @staticmethod
    def json_report(report: dict[str, Any], path: str) -> None:
        """Write the profiling report as JSON."""
        out = Path(path)

        def _serialize(obj: object) -> object:
            if isinstance(obj, StatsResult):
                return StatsEngine.as_dict(obj)
            if isinstance(obj, dict):
                return {str(k): _serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_serialize(v) for v in obj]
            return obj

        serializable = _serialize(report)
        atomic_write(str(out), json.dumps(serializable, indent=2, default=str) + "\n")
        print(f"JSON report saved to {out}")

    @staticmethod
    def markdown_report(report: dict[str, Any], path: str) -> None:
        """Write the profiling report as Markdown."""
        lines: list[str] = [
            "# Assist Canonicalizer — Algorithmic Performance Profile",
            "",
            f"**Target:** `{report.get('target', 'unknown')}`  ",
            f"**Iterations:** {report.get('iterations', 0)} | "
            f"**Warmup:** {report.get('warmup', 0)} | "
            f"**Granularity:** {report.get('granularity', 'coarse')}",
            "",
        ]

        agg = report.get("aggregate", {})
        if agg:
            lines.extend(_md_stat_table("## Aggregate Performance", agg))

        phases = report.get("phases", {})
        if phases:
            lines.append("## Phase Timing")
            lines.append("")
            ph_headers = (
                "Phase",
                "Mean (ms)",
                "Median (ms)",
                "p95 (ms)",
                "p99 (ms)",
                "StdDev (ms)",
                "Memory Δ (MB)",
            )
            ph_rows: list[tuple[str, ...]] = []
            for name, phase_data in phases.items():
                e = phase_data.get("elapsed", {})
                m = phase_data.get("memory_delta_mb", {})
                ph_rows.append(
                    (
                        name,
                        f"{e.get('mean', 0) * 1000:.2f}",
                        f"{e.get('median', 0) * 1000:.2f}",
                        f"{e.get('p95', 0) * 1000:.2f}",
                        f"{e.get('p99', 0) * 1000:.2f}",
                        f"{e.get('stddev', 0) * 1000:.2f}",
                        f"{m.get('mean', 0):.2f}",
                    )
                )
            lines.extend(_md_aligned_table(ph_headers, "<>", ph_rows))
            lines.append("")

        components = report.get("components", {})
        if components:
            lines.append("## Scoring Component Micro-Profile")
            lines.append("")
            cp_headers = ("Component", "Mean (μs)", "Median (μs)", "p95 (μs)", "p99 (μs)", "CoV%")
            cp_rows: list[tuple[str, ...]] = []
            for name, comp_data in components.items():
                e = comp_data.get("elapsed", {})
                cp_rows.append(
                    (
                        name,
                        f"{e.get('mean', 0) * 1_000_000:.1f}",
                        f"{e.get('median', 0) * 1_000_000:.1f}",
                        f"{e.get('p95', 0) * 1_000_000:.1f}",
                        f"{e.get('p99', 0) * 1_000_000:.1f}",
                        f"{e.get('cov_pct', 0):.1f}",
                    )
                )
            lines.extend(_md_aligned_table(cp_headers, "<>", cp_rows))
            lines.append("")

        regressions = report.get("regressions", [])
        if regressions:
            lines.append("## ⚠️ Regression Detections")
            lines.append("")
            for r in regressions:
                lines.append(f"- {r}")
            lines.append("")

        while lines and lines[-1] == "":
            lines.pop()
        atomic_write(path, "\n".join(lines) + "\n")
        print(f"Markdown report saved to {Path(path)}")

    @staticmethod
    def text_report(report: dict[str, Any], path: str) -> None:
        """Write the profiling report as plain text with dynamically aligned tables."""
        lines: list[str] = [
            "ALGORITHMIC PERFORMANCE PROFILING REPORT",
            "=" * 80,
            f"Target: {report.get('target', 'unknown')}",
            f"Iterations: {report.get('iterations', 0)}",
            f"Warmup: {report.get('warmup', 0)}",
            f"Granularity: {report.get('granularity', 'coarse')}",
            "-" * 80,
        ]

        _headers_agg = (
            "Metric",
            "Mean",
            "Median",
            "p95",
            "p99",
            "StdDev",
            "Min",
            "Max",
            "CoV%",
        )
        _headers_ph = (
            "Phase",
            "Mean(ms)",
            "Median(ms)",
            "p95(ms)",
            "p99(ms)",
            "Std(ms)",
            "MemΔ(MB)",
        )

        # -- Aggregate -------------------------------------------------------
        agg = report.get("aggregate", {})
        if agg:
            _rows_agg: list[tuple[str, ...]] = []
            for name, s in agg.items():
                if isinstance(s, dict):
                    _rows_agg.append(
                        (
                            name,
                            f"{s.get('mean', 0):.4f}",
                            f"{s.get('median', 0):.4f}",
                            f"{s.get('p95', 0):.4f}",
                            f"{s.get('p99', 0):.4f}",
                            f"{s.get('stddev', 0):.4f}",
                            f"{s.get('min', 0):.4f}",
                            f"{s.get('max', 0):.4f}",
                            f"{s.get('cov_pct', 0):.1f}%",
                        )
                    )
            if _rows_agg:
                hdr, sep, data = align_table(_headers_agg, _rows_agg, alignments="<>")
                lines.append("\nAggregate Performance:")
                lines.append(hdr)
                lines.append(sep)
                lines.extend(data)

        # -- Phases ----------------------------------------------------------
        phases = report.get("phases", {})
        if phases:
            _rows_ph: list[tuple[str, ...]] = []
            for name, phase_data in phases.items():
                e = phase_data.get("elapsed", {})
                m = phase_data.get("memory_delta_mb", {})
                _rows_ph.append(
                    (
                        name,
                        f"{e.get('mean', 0) * 1000:.3f}",
                        f"{e.get('median', 0) * 1000:.3f}",
                        f"{e.get('p95', 0) * 1000:.3f}",
                        f"{e.get('p99', 0) * 1000:.3f}",
                        f"{e.get('stddev', 0) * 1000:.3f}",
                        f"{m.get('mean', 0):.3f}",
                    )
                )
            if _rows_ph:
                hdr, sep, data = align_table(_headers_ph, _rows_ph, alignments="<>")
                lines.extend(["", "Phase Timing Breakdown:", hdr, sep, *data])

        # -- Components ------------------------------------------------------
        components = report.get("components", {})
        if components:
            _headers_cp = ("Component", "Mean(μs)", "Median(μs)", "p95(μs)", "p99(μs)", "CoV%")
            _rows_cp: list[tuple[str, ...]] = []
            for name, comp_data in components.items():
                e = comp_data.get("elapsed", {})
                _rows_cp.append(
                    (
                        name,
                        f"{e.get('mean', 0) * 1_000_000:.1f}",
                        f"{e.get('median', 0) * 1_000_000:.1f}",
                        f"{e.get('p95', 0) * 1_000_000:.1f}",
                        f"{e.get('p99', 0) * 1_000_000:.1f}",
                        f"{e.get('cov_pct', 0):.1f}",
                    )
                )
            if _rows_cp:
                hdr, sep, data = align_table(_headers_cp, _rows_cp, alignments="<>")
                lines.extend(["", "Scoring Component Micro-Profile:", hdr, sep, *data])

        # -- Per-language ----------------------------------------------------
        per_lang = report.get("per_language", {})
        for lang_key in sorted(per_lang):
            lang_data = per_lang[lang_key]
            lines.extend(["", "─" * 80, f"Language: {lang_key.upper()}"])
            lang_agg = lang_data.get("aggregate", {})
            if lang_agg:
                _la_rows: list[tuple[str, ...]] = []
                for name, s in lang_agg.items():
                    if isinstance(s, dict):
                        _la_rows.append(
                            (
                                name,
                                f"{s.get('mean', 0):.4f}",
                                f"{s.get('median', 0):.4f}",
                                f"{s.get('p95', 0):.4f}",
                                f"{s.get('p99', 0):.4f}",
                                f"{s.get('stddev', 0):.4f}",
                                f"{s.get('min', 0):.4f}",
                                f"{s.get('max', 0):.4f}",
                                f"{s.get('cov_pct', 0):.1f}%",
                            )
                        )
                if _la_rows:
                    hdr, sep, data = align_table(_headers_agg, _la_rows, alignments="<>")
                    lines.extend(["  Aggregate:", "  " + hdr, "  " + sep])
                    lines.extend("  " + d for d in data)

            lang_phases = lang_data.get("phases", {})
            if lang_phases:
                _lp_rows: list[tuple[str, ...]] = []
                for name, phase_data in lang_phases.items():
                    e = phase_data.get("elapsed", {})
                    m = phase_data.get("memory_delta_mb", {})
                    _lp_rows.append(
                        (
                            name,
                            f"{e.get('mean', 0) * 1000:.3f}",
                            f"{e.get('median', 0) * 1000:.3f}",
                            f"{e.get('p95', 0) * 1000:.3f}",
                            f"{e.get('p99', 0) * 1000:.3f}",
                            f"{e.get('stddev', 0) * 1000:.3f}",
                            f"{m.get('mean', 0):.3f}",
                        )
                    )
                if _lp_rows:
                    hdr, sep, data = align_table(_headers_ph, _lp_rows, alignments="<>")
                    lines.extend(["  Phase Timing:", "  " + hdr, "  " + sep])
                    lines.extend("  " + d for d in data)

        # -- Regressions -----------------------------------------------------
        regressions = report.get("regressions", [])
        if regressions:
            lines.append("\nREGRESSION DETECTIONS:")
            for r in regressions:
                lines.append(f"  {r}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("PROFILE_OK")
        lines.append("=" * 80)

        while lines and lines[-1] == "":
            lines.pop()
        atomic_write(path, "\n".join(lines) + "\n")
        print(f"Text report saved to {Path(path)}")


# ---------------------------------------------------------------------------
# Dynamic table alignment helper (shared with evaluate_metrics pattern)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Terminal formatting helpers
# ---------------------------------------------------------------------------


def _print_stat_block(title: str, stats: dict[str, Any]) -> None:
    """Print a block of statistical metrics with dynamically aligned columns."""
    print(f"\n{title}:")

    _headers = ("Metric", "Mean", "Median", "p95", "p99", "StdDev", "Min", "Max", "CoV%")
    _rows: list[tuple[str, ...]] = []
    for name, s in stats.items():
        if isinstance(s, dict):
            _rows.append(
                (
                    name,
                    f"{s.get('mean', 0):.4f}",
                    f"{s.get('median', 0):.4f}",
                    f"{s.get('p95', 0):.4f}",
                    f"{s.get('p99', 0):.4f}",
                    f"{s.get('stddev', 0):.4f}",
                    f"{s.get('min', 0):.4f}",
                    f"{s.get('max', 0):.4f}",
                    f"{s.get('cov_pct', 0):.1f}%",
                )
            )

    hdr, sep, data = align_table(_headers, _rows, alignments="<>")
    print(hdr)
    print(sep)
    for line in data:
        print(line)


def _print_resource_block(title: str, stats: dict[str, float]) -> None:
    """Print resource utilization stats to the terminal."""
    print(f"\n{title}:")
    for key, val in stats.items():
        print(f"  {key}: {val:.2f}")


def _print_phase_table(title: str, phases: dict[str, Any]) -> None:
    """Print a phase timing table with dynamically aligned columns."""
    print(f"\n{title}:")

    _headers = ("Phase", "Mean(ms)", "Median(ms)", "p95(ms)", "p99(ms)", "Std(ms)", "MemΔ(MB)")
    _rows: list[tuple[str, ...]] = []
    for name, phase_data in phases.items():
        e = phase_data.get("elapsed", {})
        m = phase_data.get("memory_delta_mb", {})
        _rows.append(
            (
                name,
                f"{e.get('mean', 0) * 1000:.3f}",
                f"{e.get('median', 0) * 1000:.3f}",
                f"{e.get('p95', 0) * 1000:.3f}",
                f"{e.get('p99', 0) * 1000:.3f}",
                f"{e.get('stddev', 0) * 1000:.3f}",
                f"{m.get('mean', 0):.3f}",
            )
        )

    hdr, sep, data = align_table(_headers, _rows, alignments="<>")
    print(hdr)
    print(sep)
    for line in data:
        print(line)


def _md_aligned_table(
    headers: tuple[str, ...],
    alignments: str,
    rows: list[tuple[str, ...]],
) -> list[str]:
    """Return Prettier-compatible aligned Markdown table as a list of lines.

    Column widths are computed dynamically from header and data cell lengths,
    following the :func:`tools.evaluate_metrics._markdown_report` pattern.

    Args:
        headers: Per-column header strings.
        alignments: Alignment spec (same convention as
            :func:`tools.evaluate_metrics.align_table`):
            ``"<"`` — left-align all; ``">"`` — right-align all;
            ``"<>"`` — first column left, remainder right.
        rows: Data rows; each tuple must have the same length as *headers*.
    """
    if not headers:
        return []
    ncols = len(headers)
    # Expand alignments to per-column list (same logic as align_table)
    if len(alignments) == 1:
        aligns = list(alignments * ncols)
    else:
        aligns = list(alignments)
        if len(aligns) < ncols:
            aligns.extend([aligns[-1]] * (ncols - len(aligns)))
        aligns = aligns[:ncols]

    # -- Phase 1: compute max column widths --
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    widths = [max(w, 3) for w in widths]  # guarantee at least 3 dashes for separators

    # -- Phase 2: render --
    lines: list[str] = []

    # Header row
    hdr_parts = [f" {h:{a}{w}} " for h, a, w in zip(headers, aligns, widths, strict=True)]
    lines.append("|" + "|".join(hdr_parts) + "|")

    # Separator row
    sep_parts: list[str] = []
    for i, w in enumerate(widths):
        dashes = w - 1  # colon occupies 1 char
        if aligns[i] == ">":
            sep_parts.append(" " + "-" * dashes + ": ")
        else:
            sep_parts.append(" :" + "-" * dashes + " ")
    lines.append("|" + "|".join(sep_parts) + "|")

    # Data rows
    for row in rows:
        parts = [f" {c:{a}{w}} " for c, a, w in zip(row, aligns, widths, strict=True)]
        lines.append("|" + "|".join(parts) + "|")

    return lines


def _md_stat_table(title: str, stats: dict[str, Any]) -> list[str]:
    """Build a Markdown statistical table section with dynamic column alignment."""
    headers = ("Metric", "Mean", "Median", "p95", "p99", "StdDev", "Min", "Max", "CoV%")
    rows: list[tuple[str, ...]] = []
    for name, s in stats.items():
        if isinstance(s, dict):
            rows.append(
                (
                    name,
                    f"{s.get('mean', 0):.4f}",
                    f"{s.get('median', 0):.4f}",
                    f"{s.get('p95', 0):.4f}",
                    f"{s.get('p99', 0):.4f}",
                    f"{s.get('stddev', 0):.4f}",
                    f"{s.get('min', 0):.4f}",
                    f"{s.get('max', 0):.4f}",
                    f"{s.get('cov_pct', 0):.1f}",
                )
            )
    lines: list[str] = [title, ""]
    if rows:
        lines.extend(_md_aligned_table(headers, "<>", rows))
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Profiling targets
# ---------------------------------------------------------------------------

# -- Import bootstrap helpers for evaluate_metrics compatibility ----------

_EVAL_BOOTSTRAPPED = False
_EVAL_BUILD_CANDIDATES = None
_EVAL_BUILD_INDEX = None
_EVAL_LOAD_INTENT_SOURCES = None
_EVAL_NORMALIZE_TEXT = None
_EVAL_RANKED_CANDIDATE = None
_EVAL_SCORE_BREAKDOWN = None
_EVAL_CANDIDATE = None
_EVAL_FALLBACK_REASON = None
_EVAL_RUNTIME = None
_EVAL_DEFAULT_MIN_CONFIDENCE = 0.5
_EVAL_DEFAULT_MIN_MARGIN = 0.04

# Internal symbols from evaluate_metrics needed for component profiling
_EM_RUN_HASSIL_ALL = None
_EM_SLOTS_FROM = None
_EM_VALUES_EQUAL = None
_EM_SLOTS_MATCH = None
_EM_MAKE_HASSIL_LISTS = None
_EM_SELECT_ACCEPTED = None
_EM_RECORD_CASE = None
_EM_COMPONENT_SCORE = None
_EM_SELECT_ABLATION = None


def _bootstrap_eval_imports() -> None:
    """Import the symbols from ``tools.evaluate_metrics`` needed for the evaluate target."""
    global _EVAL_BOOTSTRAPPED
    global _EVAL_BUILD_CANDIDATES, _EVAL_BUILD_INDEX, _EVAL_LOAD_INTENT_SOURCES
    global _EVAL_NORMALIZE_TEXT, _EVAL_RANKED_CANDIDATE, _EVAL_SCORE_BREAKDOWN
    global _EVAL_CANDIDATE, _EVAL_FALLBACK_REASON, _EVAL_RUNTIME
    global _EVAL_DEFAULT_MIN_CONFIDENCE, _EVAL_DEFAULT_MIN_MARGIN
    global _EM_RUN_HASSIL_ALL, _EM_SLOTS_FROM, _EM_VALUES_EQUAL, _EM_SLOTS_MATCH
    global _EM_MAKE_HASSIL_LISTS, _EM_SELECT_ACCEPTED, _EM_RECORD_CASE
    global _EM_COMPONENT_SCORE, _EM_SELECT_ABLATION

    if _EVAL_BOOTSTRAPPED:
        return

    import tools.evaluate_metrics as _emod

    # Wrapped bootstrap
    _emod._bootstrap_project_imports()

    _EVAL_BUILD_CANDIDATES = _emod.build_candidates_from_intent_sources
    _EVAL_BUILD_INDEX = _emod.build_index
    _EVAL_LOAD_INTENT_SOURCES = _emod.load_language_intent_sources
    _EVAL_NORMALIZE_TEXT = _emod.normalize_text
    _EVAL_RANKED_CANDIDATE = _emod._RankedCandidate
    _EVAL_SCORE_BREAKDOWN = _emod._ScoreBreakdown
    _EVAL_CANDIDATE = _emod.Candidate
    _EVAL_FALLBACK_REASON = _emod.FallbackReason
    _EVAL_RUNTIME = _emod.CanonicalizerRuntime
    _EVAL_DEFAULT_MIN_CONFIDENCE = _emod.DEFAULT_MIN_CONFIDENCE
    _EVAL_DEFAULT_MIN_MARGIN = _emod.DEFAULT_MIN_MARGIN

    _EM_RUN_HASSIL_ALL = _emod.run_hassil_recognize_all
    _EM_SLOTS_FROM = _emod._slots_from_candidate
    _EM_VALUES_EQUAL = _emod._values_equal
    _EM_SLOTS_MATCH = _emod._slots_match
    _EM_MAKE_HASSIL_LISTS = _emod.make_hassil_slot_lists
    _EM_SELECT_ACCEPTED = _emod._select_accepted_with_gate
    _EM_RECORD_CASE = _emod._record_case_result
    _EM_COMPONENT_SCORE = _emod._component_score
    _EM_SELECT_ABLATION = _emod._select_ablation_candidate

    _EVAL_BOOTSTRAPPED = True


# ---------------------------------------------------------------------------
# _profile_evaluate
# ---------------------------------------------------------------------------


def _profile_evaluate(
    datasets: dict[str, str],
    iterations: int,
    warmup: int,
    granularity: str,
    timer: PhaseTimer,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    """Profile the full evaluate_metrics pipeline.

    Returns per-language and aggregate phase timing.
    """
    _bootstrap_eval_imports()

    output_dir = Path(_REPO_ROOT) / BENCHMARK_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = str(output_dir / "performance_benchmark_report.json")
    md_path = str(output_dir / "performance_benchmark_report.md")

    total_runs = warmup + iterations
    aggregate_elapsed: list[float] = []

    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup
        label = f"evaluate_run_{run_idx}"

        if is_warmup:
            print(f"Warmup run {run_idx + 1}/{warmup} ...", flush=True)
        else:
            print(f"Profiling run {run_idx - warmup + 1}/{iterations} ...", flush=True)

        if granularity == "coarse" or is_warmup:
            timer.start(label)
            gc.collect()
            monitor.snapshot_gc()

            asyncio.run(
                run_evaluation(
                    datasets=dict(datasets),
                    failure_limit=0,
                    output_json=json_path if not is_warmup else None,
                    output_md=md_path if not is_warmup else None,
                    min_intent_slot_accuracy=None,
                    max_fallback_rate=None,
                )
            )
            timer.stop()
        elif granularity == "medium":
            # Instrumented evaluate: wrap run_evaluation with phase timing
            timer.start(label)
            gc.collect()
            monitor.snapshot_gc()

            # We call run_evaluation directly but with per-language phases
            # wrapped around the async call
            with timer.phase("evaluate_total"):
                asyncio.run(
                    run_evaluation(
                        datasets=dict(datasets),
                        failure_limit=0,
                        output_json=json_path if not is_warmup else None,
                        output_md=md_path if not is_warmup else None,
                        min_intent_slot_accuracy=None,
                        max_fallback_rate=None,
                    )
                )
            timer.stop()
        else:  # fine
            timer.start(label)
            gc.collect()
            monitor.snapshot_gc()
            with timer.phase("evaluate_total"):
                asyncio.run(
                    run_evaluation(
                        datasets=dict(datasets),
                        failure_limit=0,
                        output_json=json_path if not is_warmup else None,
                        output_md=md_path if not is_warmup else None,
                        min_intent_slot_accuracy=None,
                        max_fallback_rate=None,
                    )
                )
            timer.stop()

        if not is_warmup and label in timer.phases:
            aggregate_elapsed.append(timer.phases[label][-1])

    # Build report section
    result: dict[str, Any] = {}
    if aggregate_elapsed:
        result["aggregate"] = {
            "evaluate_wall_time_sec": StatsEngine.as_dict(StatsEngine.compute(aggregate_elapsed))
        }
    return result


# ---------------------------------------------------------------------------
# _profile_build_index
# ---------------------------------------------------------------------------


def _profile_build_index(
    datasets: dict[str, str],
    iterations: int,
    warmup: int,
    granularity: str,
    timer: PhaseTimer,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    """Profile only the build_index pipeline per language.

    Returns per-language and aggregate build timing.
    """
    _bootstrap_imports()
    total_runs = warmup + iterations

    per_language: dict[str, dict[str, Any]] = {}
    all_elapsed: list[float] = []

    for lang, path in sorted(datasets.items()):
        # Load the dataset to extract registry slots
        try:
            import orjson

            with open(path, "rb") as f:
                data = orjson.loads(f.read())
        except Exception:
            print(f"Warning: could not load dataset {path}, skipping.", file=sys.stderr)
            continue

        if not isinstance(data, dict):
            continue

        # Determine slots
        raw_slots = data.get("registry_slots")
        if raw_slots is not None and isinstance(raw_slots, dict):
            slots: dict[str, tuple[str, ...]] = {
                str(k): tuple(str(v) for v in vals) for k, vals in raw_slots.items()
            }
        else:
            slots = {
                str(key): tuple(values) for key, values in REGISTRY_SLOTS.get(lang, {}).items()
            }

        lang_elapsed: list[float] = []

        for run_idx in range(total_runs):
            is_warmup = run_idx < warmup
            label = f"build_index_{lang}"

            sources = load_language_intent_sources(lang)
            gc.collect()
            monitor.snapshot_gc()

            timer.start(label)
            with timer.phase(f"build_candidates_{lang}"):
                candidates = build_candidates_from_intent_sources(lang, sources, slots)
            with timer.phase(f"build_index_{lang}"):
                build_index(lang, candidates)
            timer.stop()

            if not is_warmup and label in timer.phases:
                lang_elapsed.append(timer.phases[label][-1])
                all_elapsed.append(timer.phases[label][-1])

        if lang_elapsed:
            per_language[lang] = {
                "aggregate": {
                    "build_index_wall_time_sec": StatsEngine.as_dict(
                        StatsEngine.compute(lang_elapsed)
                    )
                }
            }

    result: dict[str, Any] = {"per_language": per_language}
    if all_elapsed:
        result["aggregate"] = {
            "build_index_wall_time_sec_total": StatsEngine.as_dict(StatsEngine.compute(all_elapsed))
        }
    return result


# ---------------------------------------------------------------------------
# _profile_rank
# ---------------------------------------------------------------------------


def _profile_rank(
    datasets: dict[str, str],
    iterations: int,
    warmup: int,
    granularity: str,
    timer: PhaseTimer,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    """Profile the rank_candidates hot path with real queries.

    Builds one index per language and then times ``CanonicalIndex.rank()``
    for every test-case query.
    """
    _bootstrap_imports()
    import orjson

    total_runs = warmup + iterations
    per_language: dict[str, dict[str, Any]] = {}
    all_elapsed: list[float] = []

    for lang, path in sorted(datasets.items()):
        try:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
        except Exception:
            print(f"Warning: could not load dataset {path}", file=sys.stderr)
            continue

        if not isinstance(data, dict):
            continue

        raw_slots = data.get("registry_slots")
        if raw_slots is not None and isinstance(raw_slots, dict):
            slots: dict[str, tuple[str, ...]] = {
                str(k): tuple(str(v) for v in vals) for k, vals in raw_slots.items()
            }
        else:
            slots = {
                str(key): tuple(values) for key, values in REGISTRY_SLOTS.get(lang, {}).items()
            }

        sources = load_language_intent_sources(lang)
        candidates = build_candidates_from_intent_sources(lang, sources, slots)
        index = build_index(lang, candidates)

        raw_cases = data.get("test_cases", [])
        if not isinstance(raw_cases, list):
            continue

        queries: list[str] = [
            case["query"]
            for case in raw_cases
            if isinstance(case, dict) and isinstance(case.get("query"), str)
        ]

        if not queries:
            continue

        lang_elapsed: list[float] = []
        per_query_times: list[list[float]] = [[] for _ in queries]

        for run_idx in range(total_runs):
            is_warmup = run_idx < warmup
            label = f"rank_{lang}"

            gc.collect()
            monitor.snapshot_gc()

            timer.start(label)
            if granularity != "fine":
                for qi, query in enumerate(queries):
                    q_start = time.perf_counter()
                    _ = index.rank(query)
                    q_elapsed = time.perf_counter() - q_start
                    if not is_warmup:
                        per_query_times[qi].append(q_elapsed)
            else:
                # Fine granularity: wrap each rank sub-call
                for qi, query in enumerate(queries):
                    q_start = time.perf_counter()
                    from contextlib import suppress

                    with timer.phase("rank_candidates_inner"), suppress(Exception):
                        _ = index.rank(query)
                    q_elapsed = time.perf_counter() - q_start
                    if not is_warmup:
                        per_query_times[qi].append(q_elapsed)
            timer.stop()

            if not is_warmup and label in timer.phases:
                lang_elapsed.append(timer.phases[label][-1])
                all_elapsed.append(timer.phases[label][-1])

        # Compute per-query stats
        avg_query_times: list[float] = []
        for q_times in per_query_times:
            if q_times:
                avg_query_times.append(statistics.mean(q_times))

        if lang_elapsed:
            rank_agg: dict[str, Any] = {
                "rank_total_wall_time_sec": StatsEngine.as_dict(StatsEngine.compute(lang_elapsed))
            }
            if avg_query_times:
                rank_agg["rank_per_query_wall_time_sec"] = StatsEngine.as_dict(
                    StatsEngine.compute(avg_query_times)
                )
                rank_agg["query_count"] = len(queries)
            per_language[lang] = {"aggregate": rank_agg}

    result: dict[str, Any] = {"per_language": per_language}
    if all_elapsed:
        result["aggregate"] = {
            "rank_wall_time_sec_total": StatsEngine.as_dict(StatsEngine.compute(all_elapsed))
        }
    return result


# ---------------------------------------------------------------------------
# _profile_components
# ---------------------------------------------------------------------------


def _profile_components(
    datasets: dict[str, str],
    iterations: int,
    warmup: int,
    timer: PhaseTimer,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    """Micro-benchmark individual scoring component functions.

    Uses real query/candidate data from all available datasets.  Each
    component is timed in isolation across all queries and candidates.
    """
    _bootstrap_imports()
    import orjson

    from custom_components.assist_canonicalizer.ranking import (
        _build_positional_lookup,
        _exact_intent_score,
        _positional_intent_score_from_lookup,
    )

    # Collect all (query_normalized, candidate_normalized_text, candidate_metadata) tuples
    all_queries: list[tuple[str, str, str, str | None]] = []  # (raw, norm, lang, literal_text)
    all_per_lang: dict[str, list[tuple[str, str, str, str | None]]] = {}

    for lang, path in sorted(datasets.items()):
        try:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        raw_slots = data.get("registry_slots")
        if raw_slots is not None and isinstance(raw_slots, dict):
            slots: dict[str, tuple[str, ...]] = {
                str(k): tuple(str(v) for v in vals) for k, vals in raw_slots.items()
            }
        else:
            slots = {
                str(key): tuple(values) for key, values in REGISTRY_SLOTS.get(lang, {}).items()
            }

        sources = load_language_intent_sources(lang)
        candidates = build_candidates_from_intent_sources(lang, sources, slots)
        build_index(lang, candidates)

        raw_cases = data.get("test_cases", [])
        if not isinstance(raw_cases, list):
            continue

        lang_entries: list[tuple[str, str, str, str | None]] = []
        for case in raw_cases:
            if not isinstance(case, dict):
                continue
            query = case.get("query")
            if not isinstance(query, str):
                continue
            norm = normalize_text(query)
            lang_entries.append((query, norm, lang, None))

        # Also collect candidate data for per-candidate profiling
        for candidate in candidates[:50]:  # limit to top 50 per lang
            raw_text = candidate.text
            norm_text = candidate.normalized_text
            lit_text = candidate.metadata.get("literal_text") if candidate.metadata else None
            lang_entries.append((raw_text, norm_text, lang, lit_text))

        all_queries.extend(lang_entries)
        all_per_lang[lang] = lang_entries

    total_runs = warmup + iterations
    component_results: dict[str, dict[str, list[float]]] = {
        comp: {"elapsed": []} for comp in SCORING_COMPONENT_NAMES
    }

    # Pre-build BM25 index and CharNGramIndex for all candidates across all langs
    all_norm_texts: list[str] = []
    all_candidates: list[Any] = []
    for lang, path in sorted(datasets.items()):
        try:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        raw_slots = data.get("registry_slots")
        if raw_slots is not None and isinstance(raw_slots, dict):
            slots = {str(k): tuple(str(v) for v in vals) for k, vals in raw_slots.items()}
        else:
            slots = {
                str(key): tuple(values) for key, values in REGISTRY_SLOTS.get(lang, {}).items()
            }
        sources = load_language_intent_sources(lang)
        candidates = build_candidates_from_intent_sources(lang, sources, slots)
        for c in candidates:
            all_norm_texts.append(c.normalized_text)
            all_candidates.append(c)

    if not all_queries:
        return {}

    bm25_idx = BM25Index.from_normalized_texts(all_norm_texts) if all_norm_texts else None
    char_grams = [char_ngrams_normalized(t) for t in all_norm_texts]
    char_idx = CharNGramIndex.from_grams(char_grams) if char_grams else None

    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup

        for query_raw, query_norm, lang, lit_text in all_queries:
            if is_warmup:
                # Still execute for JIT warmup
                _ = normalize_text(query_raw)
                _ = normalize_text_no_diacritics(query_raw, lang)
                _ = char_ngrams_normalized(query_norm)
                if bm25_idx is not None:
                    _ = bm25_idx.score(query_norm)
                q_grams = char_ngrams_normalized(query_norm)
                if char_idx is not None:
                    _ = char_idx.score(q_grams)
                # RapidFuzz against first candidate
                if all_norm_texts:
                    _ = rapidfuzz_similarity_normalized(query_norm, all_norm_texts[0])
                q_tokens = frozenset(query_norm.split())
                if lit_text:
                    _ = _exact_intent_score(lit_text, q_tokens)
                _ = lexical_score(0.5, 0.5, 0.5, 0.5)
                continue

            # --- normalize_text ---
            t0 = time.perf_counter()
            _ = normalize_text(query_raw)
            t1 = time.perf_counter()
            component_results["normalize_text"]["elapsed"].append(t1 - t0)

            # --- normalize_text_no_diacritics ---
            t0 = time.perf_counter()
            _ = normalize_text_no_diacritics(query_raw, lang)
            t1 = time.perf_counter()
            component_results["normalize_text_no_diacritics"]["elapsed"].append(t1 - t0)

            # --- char_ngrams_normalized ---
            t0 = time.perf_counter()
            _ = char_ngrams_normalized(query_norm)
            t1 = time.perf_counter()
            component_results["char_ngrams_normalized"]["elapsed"].append(t1 - t0)

            # --- bm25_score ---
            if bm25_idx is not None:
                t0 = time.perf_counter()
                _ = bm25_idx.score(query_norm)
                t1 = time.perf_counter()
                component_results["bm25_score"]["elapsed"].append(t1 - t0)

            # --- char_ngram_score ---
            q_grams = char_ngrams_normalized(query_norm)
            if char_idx is not None:
                t0 = time.perf_counter()
                _ = char_idx.score(q_grams)
                t1 = time.perf_counter()
                component_results["char_ngram_score"]["elapsed"].append(t1 - t0)

            # --- rapidfuzz_similarity ---
            if all_norm_texts:
                t0 = time.perf_counter()
                _ = rapidfuzz_similarity_normalized(query_norm, all_norm_texts[0])
                t1 = time.perf_counter()
                component_results["rapidfuzz_similarity"]["elapsed"].append(t1 - t0)

            # --- exact_intent_score ---
            q_tokens = frozenset(query_norm.split())
            if lit_text:
                t0 = time.perf_counter()
                _ = _exact_intent_score(lit_text, q_tokens)
                t1 = time.perf_counter()
                component_results["exact_intent_score"]["elapsed"].append(t1 - t0)

            # --- positional_intent_score ---
            if lit_text:
                from custom_components.assist_canonicalizer.ranking import (
                    literal_token_variants,
                )

                variants = literal_token_variants(lit_text)
                all_lit_tokens = frozenset().union(*variants) if variants else frozenset()
                pos_lookup = _build_positional_lookup(all_lit_tokens, q_tokens)
                t0 = time.perf_counter()
                _ = _positional_intent_score_from_lookup(lit_text, q_tokens, pos_lookup, None)
                t1 = time.perf_counter()
                component_results["positional_intent_score"]["elapsed"].append(t1 - t0)

            # --- lexical_score ---
            t0 = time.perf_counter()
            _ = lexical_score(0.5, 0.5, 0.5, 0.5)
            t1 = time.perf_counter()
            component_results["lexical_score"]["elapsed"].append(t1 - t0)

            # --- intent_disambiguation (simulated) ---
            # Build two fake ranked candidates and apply disambiguation
            if all_candidates and len(all_candidates) >= 2:
                fake_scores_a = ScoreBreakdown(0.8, 0.8, 0.8, 0.9, 0.85)
                fake_scores_b = ScoreBreakdown(0.8, 0.8, 0.8, 0.95, 0.84)
                fake_ranked = [
                    RankedCandidate(candidate=all_candidates[0], scores=fake_scores_a),
                    RankedCandidate(candidate=all_candidates[1], scores=fake_scores_b),
                ]
                from custom_components.assist_canonicalizer.ranking import (
                    _apply_intent_disambiguation,
                )

                t0 = time.perf_counter()
                _apply_intent_disambiguation(fake_ranked)
                t1 = time.perf_counter()
                component_results["intent_disambiguation"]["elapsed"].append(t1 - t0)

    # Build results
    result: dict[str, Any] = {}
    for comp_name in SCORING_COMPONENT_NAMES:
        vals = component_results[comp_name]["elapsed"]
        if vals:
            result[comp_name] = {"elapsed": StatsEngine.as_dict(StatsEngine.compute(vals))}

    return result


# ---------------------------------------------------------------------------
# ProfilerEngine
# ---------------------------------------------------------------------------


def _build_report(
    target: str,
    iterations: int,
    warmup: int,
    granularity: str,
    languages: list[str] | None,
    timer: PhaseTimer,
    monitor: ResourceMonitor,
    target_result: dict[str, Any],
    baseline: BaselineManager,
    max_regression_pct: float,
    *,
    warn_on_missing: bool = False,
) -> dict[str, Any]:
    """Assemble the full profiling report.

    Args:
        target: The profiling target name.
        iterations: Number of measurement iterations.
        warmup: Number of warmup iterations.
        granularity: Granularity level string.
        languages: Language filter list or None.
        timer: The phase timer with all recorded data.
        monitor: The resource monitor.
        target_result: Per-target result dictionary from the profiler.
        baseline: Baseline manager instance.
        max_regression_pct: Maximum regression threshold percentage.
        warn_on_missing: If ``True``, emit stderr warnings when the baseline
            file is missing, corrupted, or unreadable.

    Returns:
        Complete report dictionary.
    """
    report: dict[str, Any] = {
        "target": target,
        "iterations": iterations,
        "warmup": warmup,
        "granularity": granularity,
        "languages": languages or [],
    }

    # Phase stats — convert StatsResult objects to serializable dicts
    raw_phase_stats = timer.stats()
    phase_stats: dict[str, dict[str, dict[str, Any]]] = {}
    for phase_name, metrics in raw_phase_stats.items():
        phase_stats[phase_name] = {
            metric_name: StatsEngine.as_dict(sr) if isinstance(sr, StatsResult) else sr
            for metric_name, sr in metrics.items()
        }
    if phase_stats:
        report["phases"] = phase_stats

    # Aggregate from target
    agg = target_result.get("aggregate", {})
    if agg:
        report["aggregate"] = agg

    # Components
    comps = target_result.get("components", {})
    if comps:
        report["components"] = comps

    # Per-language
    per_lang = target_result.get("per_language", {})
    if per_lang:
        report["per_language"] = per_lang

    # Resource
    cpu_metrics = monitor.get_cpu_metrics()
    mem_metrics = monitor.get_memory_metrics()
    report["resource"] = {**cpu_metrics, **mem_metrics}

    # Baseline regression
    regressions = baseline.compare(
        target, report, max_regression_pct, warn_on_missing=warn_on_missing
    )
    report["regressions"] = regressions

    # Stability assessment (phase_stats is now a dict of dicts, not StatsResult)
    cov_values: list[float] = []
    for _phase_name, phase_data in phase_stats.items():
        e = phase_data.get("elapsed")
        if isinstance(e, dict) and e.get("cov_pct", 0) > 0:
            cov_values.append(float(e["cov_pct"]))
    if cov_values:
        avg_cov = statistics.mean(cov_values)
        if avg_cov < 5.0:
            report["stability"] = f"High stability — average CoV {avg_cov:.1f}% across phases"
        elif avg_cov < 15.0:
            report["stability"] = f"Moderate stability — average CoV {avg_cov:.1f}% across phases"
        else:
            report["stability"] = (
                f"Low stability — average CoV {avg_cov:.1f}% across phases. "
                f"Consider increasing iterations or closing background processes."
            )

    return report


def _run_profiling(
    target: str,
    datasets: dict[str, str],
    iterations: int,
    warmup: int,
    granularity: str,
    baseline: BaselineManager,
    max_regression_pct: float,
    output_json: str | None,
    output_md: str | None,
    output_txt: str | None,
    save_baseline: bool,
    languages: list[str] | None,
    *,
    warn_on_missing: bool = False,
) -> dict[str, Any]:
    """Coordinate the profiling run for one target.

    Args:
        target: Profile target name.
        datasets: Language → path mapping.
        iterations: Measurement iterations.
        warmup: Warmup iterations.
        granularity: ``coarse``, ``medium``, or ``fine``.
        baseline: Baseline manager.
        max_regression_pct: Regression threshold.
        output_json: Optional JSON output path.
        output_md: Optional Markdown output path.
        output_txt: Optional plain text output path.
        save_baseline: Whether to save results as new baseline.
        languages: Optional language code filter list.
        warn_on_missing: If ``True``, emit stderr warnings when the baseline
            file is missing, corrupted, or unreadable.

    Returns:
        The assembled report dictionary.
    """
    print(f"\n{'=' * 90}")
    print(f"Profiling target: {target}")
    print(f"{'=' * 90}")
    print(f"Iterations: {iterations}  |  Warmup: {warmup}  |  Granularity: {granularity}")
    print(f"Languages: {', '.join(sorted(datasets.keys()))}")
    print(f"{'=' * 90}")

    monitor = ResourceMonitor(interval=0.02)
    timer = PhaseTimer(monitor)

    monitor.start()
    monitor.snapshot_gc()

    gc.collect()
    gc.disable()  # Disable GC during profiling for stable timing

    try:
        if target == "evaluate":
            target_result = _profile_evaluate(
                datasets, iterations, warmup, granularity, timer, monitor
            )
        elif target == "build_index":
            target_result = _profile_build_index(
                datasets, iterations, warmup, granularity, timer, monitor
            )
        elif target == "rank":
            target_result = _profile_rank(datasets, iterations, warmup, granularity, timer, monitor)
        elif target == "components":
            # Components profiling doesn't use granularity (always fine)
            target_result = _profile_components(datasets, iterations, warmup, timer, monitor)
            # Wrap so _build_report can extract via target_result.get("components", {})
            target_result = {"components": target_result}
        else:
            print(f"Unknown target: {target}", file=sys.stderr)
            return {}
    finally:
        gc.enable()
        monitor.snapshot_gc()
        monitor.stop_monitor()
        monitor.join(timeout=5.0)

    report = _build_report(
        target,
        iterations,
        warmup,
        granularity,
        languages,
        timer,
        monitor,
        target_result,
        baseline,
        max_regression_pct,
        warn_on_missing=warn_on_missing,
    )

    # Output
    ReportGenerator.terminal(report)

    if output_json:
        ReportGenerator.json_report(report, output_json)
    if output_md:
        ReportGenerator.markdown_report(report, output_md)
    if output_txt:
        ReportGenerator.text_report(report, output_txt)

    if save_baseline:
        baseline.save(target, report)

    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the algorithmic performance profiler."""
    parser = argparse.ArgumentParser(
        description="Profile Assist Canonicalizer algorithmic performance",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=PROFILING_TARGETS,
        default="rank",
        help="Profiling target (default: rank)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of measurement iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Number of warmup iterations (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=GRANULARITY_LEVELS,
        default=DEFAULT_GRANULARITY,
        help=f"Profiling granularity level (default: {DEFAULT_GRANULARITY})",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="tests/real_world",
        help="Directory containing real-world JSON datasets",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="*",
        default=None,
        help="Space-separated language codes to profile (default: all)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional JSON report output path",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional Markdown report output path",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default=None,
        help="Optional plain text report output path",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline file for regression detection",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        default=False,
        help="Save current results as new baseline",
    )
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=DEFAULT_MAX_REGRESSION_PCT,
        help=(
            f"Maximum regression percentage before flagging (default: {DEFAULT_MAX_REGRESSION_PCT})"
        ),
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        default=False,
        help="Also run cProfile and dump stats to the benchmark directory",
    )

    args = parser.parse_args()
    _bootstrap_imports()

    if args.iterations < 1:
        print("Error: --iterations must be positive", file=sys.stderr)
        sys.exit(1)
    if args.warmup < 0:
        print("Error: --warmup must be non-negative", file=sys.stderr)
        sys.exit(1)

    # Sanitize all user-supplied paths (CodeQL path-injection hardening)
    safe_datasets_dir = sanitize_path_required(_REPO_ROOT, "datasets_dir", args.datasets_dir)
    safe_output_json: str | None = None
    safe_output_md: str | None = None
    safe_output_txt: str | None = None
    if args.output_json is not None:
        safe_output_json = sanitize_path_required(_REPO_ROOT, "output_json", args.output_json)
    if args.output_md is not None:
        safe_output_md = sanitize_path_required(_REPO_ROOT, "output_md", args.output_md)
    if args.output_txt is not None:
        safe_output_txt = sanitize_path_required(_REPO_ROOT, "output_txt", args.output_txt)

    # Discover datasets
    datasets = discover_datasets(safe_datasets_dir, args.languages)
    if not datasets:
        print(f"Error: No datasets found in {safe_datasets_dir}", file=sys.stderr)
        sys.exit(1)

    # Initialize baseline manager
    baseline_mgr = BaselineManager(_REPO_ROOT)

    # Determine whether user explicitly requested baseline regression
    _explicit_baseline: bool = bool(args.baseline)

    # If --baseline specified, load from that path instead
    if args.baseline:
        safe_baseline = sanitize_path_required(_REPO_ROOT, "baseline", args.baseline)
        baseline_mgr = BaselineManager(
            str(Path(safe_baseline).parent.parent) if BASELINE_DIR in safe_baseline else _REPO_ROOT
        )

    print("PROFILE_START", flush=True)

    try:
        if args.target == "all":
            # Run all targets sequentially, aggregate results
            all_reports: dict[str, Any] = {}
            for tgt in ("build_index", "rank", "components", "evaluate"):
                # Set default output paths for each target
                out_json = (
                    os.path.join(_REPO_ROOT, BENCHMARK_DIR, f"profile_{tgt}.json")
                    if safe_output_json is None
                    else safe_output_json
                )
                out_md = (
                    os.path.join(_REPO_ROOT, BENCHMARK_DIR, f"profile_{tgt}.md")
                    if safe_output_md is None
                    else safe_output_md
                )
                out_txt = (
                    os.path.join(_REPO_ROOT, BENCHMARK_DIR, f"profile_{tgt}.txt")
                    if safe_output_txt is None
                    else safe_output_txt
                )

                report = _run_profiling(
                    tgt,
                    datasets,
                    args.iterations,
                    args.warmup,
                    args.granularity,
                    baseline_mgr,
                    args.max_regression_pct,
                    out_json,
                    out_md,
                    out_txt,
                    args.save_baseline,
                    args.languages,
                    warn_on_missing=_explicit_baseline,
                )
                all_reports[tgt] = report

            # Write aggregate all-targets report
            output_dir = Path(_REPO_ROOT) / BENCHMARK_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            agg_path = output_dir / "profile_all.json"
            agg_json = {
                "target": "all",
                "targets": all_reports,
            }
            atomic_write(str(agg_path), json.dumps(agg_json, indent=2, default=str) + "\n")
            print(f"\nAggregate all-targets report saved to {agg_path}")
        else:
            _run_profiling(
                args.target,
                datasets,
                args.iterations,
                args.warmup,
                args.granularity,
                baseline_mgr,
                args.max_regression_pct,
                safe_output_json,
                safe_output_md,
                safe_output_txt,
                args.save_baseline,
                args.languages,
                warn_on_missing=_explicit_baseline,
            )

        # Optional cProfile run
        if args.cprofile:
            print("\nRunning cProfile snapshot ...", flush=True)
            import cProfile as cProf

            pr = cProf.Profile()
            pr.enable()

            # Always profile the rank hot path — the most critical code path
            # that runs on every user utterance.
            _bootstrap_imports()
            for lang, path in sorted(datasets.items()):
                import orjson as _oj

                with open(path, "rb") as f:
                    data = _oj.loads(f.read())
                if not isinstance(data, dict):
                    continue
                raw_slots = data.get("registry_slots")
                if raw_slots is not None and isinstance(raw_slots, dict):
                    slots: dict[str, tuple[str, ...]] = {
                        str(k): tuple(str(v) for v in vals) for k, vals in raw_slots.items()
                    }
                else:
                    slots = {
                        str(key): tuple(values)
                        for key, values in REGISTRY_SLOTS.get(lang, {}).items()
                    }
                sources = load_language_intent_sources(lang)
                candidates = build_candidates_from_intent_sources(lang, sources, slots)
                index = build_index(lang, candidates)
                raw_cases = data.get("test_cases", [])
                if isinstance(raw_cases, list):
                    for case in raw_cases:
                        if isinstance(case, dict) and isinstance(case.get("query"), str):
                            index.rank(case["query"])

            pr.disable()
            profile_dir = Path(_REPO_ROOT) / BENCHMARK_DIR
            profile_dir.mkdir(parents=True, exist_ok=True)
            prof_dump = profile_dir / "profile_metrics.prof"
            pr.dump_stats(str(prof_dump))
            print(f"cProfile dump saved to {prof_dump}")

            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
            ps.print_stats(50)
            print("\nTOP 50 FUNCTIONS BY CUMULATIVE TIME:")
            print("-" * 80)
            print(s.getvalue())

    except KeyboardInterrupt:
        print("\nPROFILE_INTERRUPTED", flush=True)
        sys.exit(1)
    except Exception as exc:
        print(f"\nPROFILE_FAILED: {exc}", flush=True)
        raise

    print("PROFILE_OK", flush=True)


if __name__ == "__main__":
    main()
