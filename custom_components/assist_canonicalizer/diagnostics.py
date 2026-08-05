"""Runtime diagnostics models for Assist Canonicalizer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from homeassistant.util.json import JsonObjectType, JsonValueType

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
    last_request_id: str | None = None
    selected_delegated_text_hash: str | None = None
    selected_candidate_source: str | None = None
    confidence_gate: Mapping[str, JsonValueType] | None = None
    execution_result: str | None = None
    recognition_kind: str | None = None
    recognition_intent: str | None = None
    recognition_unmatched_count: int = 0
    recognition_latency_ms: float | None = None
    preflight_attempt_count: int = 0
    metadata_diverged: bool = False
    metadata_intent_matches_observed: bool | None = None
    metadata_slots_match_observed: bool | None = None
    metadata_divergence_reason: str | None = None
    recovery_used: bool = False
    registry_record_count: int = 0
    registry_generation: int = 0
    registry_fingerprint: str | None = None
    registry_postings_consulted: int = 0
    registry_values_nominated: int = 0
    registry_values_scored: int = 0
    fuzzy_dynamic_candidates: int = 0
    registry_retrieval_latency_ms: float | None = None
    selected_from_fuzzy_registry: bool = False

    def as_dict(self) -> JsonObjectType:
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
            "last_request_id": self.last_request_id,
            "selected_delegated_text_hash": self.selected_delegated_text_hash,
            "selected_candidate_source": self.selected_candidate_source,
            "confidence_gate": dict(self.confidence_gate) if self.confidence_gate else None,
            "execution_result": self.execution_result,
            "recognition_kind": self.recognition_kind,
            "recognition_intent": self.recognition_intent,
            "recognition_unmatched_count": self.recognition_unmatched_count,
            "recognition_latency_ms": self.recognition_latency_ms,
            "preflight_attempt_count": self.preflight_attempt_count,
            "metadata_diverged": self.metadata_diverged,
            "metadata_intent_matches_observed": self.metadata_intent_matches_observed,
            "metadata_slots_match_observed": self.metadata_slots_match_observed,
            "metadata_divergence_reason": self.metadata_divergence_reason,
            "recovery_used": self.recovery_used,
            "registry_retrieval": {
                "record_count": self.registry_record_count,
                "generation": self.registry_generation,
                "fingerprint": self.registry_fingerprint,
                "postings_consulted": self.registry_postings_consulted,
                "values_nominated": self.registry_values_nominated,
                "values_scored": self.registry_values_scored,
                "fuzzy_dynamic_candidates": self.fuzzy_dynamic_candidates,
                "latency_ms": self.registry_retrieval_latency_ms,
                "selected_from_fuzzy_registry": self.selected_from_fuzzy_registry,
            },
        }
