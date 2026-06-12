"""Candidate utterance models and deduplication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from .normalization import normalize_text, normalize_text_no_diacritics


class CandidateSource(StrEnum):
    """Known sources for candidate utterances."""

    CUSTOM_SENTENCE = "custom_sentence"
    BUILT_IN = "built_in"
    GENERATED_SAMPLE = "generated_sample"


_SOURCE_PRIORITY = {
    CandidateSource.CUSTOM_SENTENCE: 0,
    CandidateSource.BUILT_IN: 1,
    CandidateSource.GENERATED_SAMPLE: 2,
}


@dataclass(frozen=True, slots=True)
class Candidate:
    """Canonical utterance candidate."""

    text: str
    intent_name: str
    source: CandidateSource = CandidateSource.GENERATED_SAMPLE
    language: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    normalized_text: str = ""
    normalized_text_no_diacritics: str = ""
    normalized_tokens: tuple[str, ...] = field(default_factory=tuple)
    normalized_tokens_set: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate and normalize candidate data."""
        if not self.text.strip():
            raise ValueError("Candidate text must not be empty")
        if not self.intent_name.strip():
            raise ValueError("Candidate intent name must not be empty")
        if not self.normalized_text:
            object.__setattr__(self, "normalized_text", normalize_text(self.text))
        if not self.normalized_text_no_diacritics:
            object.__setattr__(
                self,
                "normalized_text_no_diacritics",
                normalize_text_no_diacritics(self.text, self.language),
            )
        if not self.normalized_tokens:
            object.__setattr__(self, "normalized_tokens", tuple(self.normalized_text.split()))
        if not self.normalized_tokens_set:
            object.__setattr__(self, "normalized_tokens_set", frozenset(self.normalized_tokens))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def source_priority(self) -> int:
        """Return lower priority values for more trusted candidate sources."""
        return _SOURCE_PRIORITY[self.source]


def deduplicate_candidates(candidates: list[Candidate]) -> tuple[Candidate, ...]:
    """Deduplicate candidates by normalized text while preserving best source priority."""
    selected: dict[str, Candidate] = {}
    for candidate in candidates:
        existing = selected.get(candidate.normalized_text)
        if existing is None or candidate.source_priority < existing.source_priority:
            selected[candidate.normalized_text] = candidate
    return tuple(
        sorted(selected.values(), key=lambda item: (item.source_priority, item.normalized_text))
    )
