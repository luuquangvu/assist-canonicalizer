"""Runtime diagnostics models for Assist Canonicalizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .const import FallbackReason


@dataclass(frozen=True, slots=True)
class CanonicalizerDiagnostics:
    """Serializable runtime diagnostics snapshot."""

    candidate_count: int = 0
    index_version: int | None = None
    last_query_latency_ms: float | None = None
    last_fallback_reason: FallbackReason | str | None = None
    last_error: str | None = None
    dynamic_candidate_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return diagnostics as a dictionary."""
        return {
            "candidate_count": self.candidate_count,
            "index_version": self.index_version,
            "last_query_latency_ms": self.last_query_latency_ms,
            "last_fallback_reason": (
                self.last_fallback_reason.value
                if isinstance(self.last_fallback_reason, FallbackReason)
                else self.last_fallback_reason
            ),
            "last_error": self.last_error,
            "dynamic_candidate_count": self.dynamic_candidate_count,
        }
