"""Candidate utterance models and deduplication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from .normalization import (
    literal_token_variants,
    normalize_text,
    normalize_text_no_diacritics_from_normalized,
    tokenize_normalized,
)
from .utils import wildcard_slot_names_sorted


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
    _normalized_text_no_diacritics: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _normalized_tokens: tuple[str, ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _normalized_tokens_set: frozenset[str] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _literal_variants: tuple[frozenset[str], ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _normalized_text_sorted: str | None = field(default=None, init=False, repr=False, compare=False)
    _total_unique_literal_tokens: int | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _has_wildcard: bool | None = field(default=None, init=False, repr=False, compare=False)
    _wildcard_info: tuple[int, str] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate and normalize candidate data."""
        if not self.text.strip():
            raise ValueError("Candidate text must not be empty")
        if not self.intent_name.strip():
            raise ValueError("Candidate intent name must not be empty")
        if not self.normalized_text:
            object.__setattr__(self, "normalized_text", normalize_text(self.text))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __hash__(self) -> int:
        """Return the hash code for Candidate."""
        return hash(
            (
                self.text,
                self.intent_name,
                self.source,
                self.language,
                frozenset(self.metadata.items()),
            )
        )

    def __eq__(self, other: object) -> bool:
        """Return whether two candidates are equal."""
        if not isinstance(other, Candidate):
            return NotImplemented
        return (
            self.text == other.text
            and self.intent_name == other.intent_name
            and self.source == other.source
            and self.language == other.language
            and self.metadata == other.metadata
        )

    @property
    def normalized_text_no_diacritics(self) -> str:
        """Return the normalized text without diacritics."""
        val = self._normalized_text_no_diacritics
        if val is None:
            val = normalize_text_no_diacritics_from_normalized(self.normalized_text, self.language)
            object.__setattr__(self, "_normalized_text_no_diacritics", val)
        return val

    @property
    def normalized_tokens(self) -> tuple[str, ...]:
        """Return the sequence of normalized tokens."""
        val = self._normalized_tokens
        if val is None:
            val = tokenize_normalized(self.normalized_text)
            object.__setattr__(self, "_normalized_tokens", val)
        return val

    @property
    def normalized_tokens_set(self) -> frozenset[str]:
        """Return the set of normalized tokens."""
        val = self._normalized_tokens_set
        if val is None:
            val = frozenset(self.normalized_tokens)
            object.__setattr__(self, "_normalized_tokens_set", val)
        return val

    @property
    def normalized_text_sorted(self) -> str:
        """Return the sorted normalized tokens joined by space."""
        val = self._normalized_text_sorted
        if val is None:
            val = " ".join(sorted(self.normalized_tokens))
            object.__setattr__(self, "_normalized_text_sorted", val)
        return val

    @property
    def literal_variants(self) -> tuple[frozenset[str], ...]:
        """Return the set of literal variants."""
        val = self._literal_variants
        if val is None:
            literal_text = self.metadata.get("literal_text")
            val = literal_token_variants(literal_text) if literal_text else ()
            object.__setattr__(self, "_literal_variants", val)
        return val

    @property
    def total_unique_literal_tokens(self) -> int:
        """Return the total number of unique literal tokens."""
        val = self._total_unique_literal_tokens
        if val is None:
            variants = self.literal_variants
            val = len({tok for var in variants for tok in var}) if variants else 0
            object.__setattr__(self, "_total_unique_literal_tokens", val)
        return val

    @property
    def source_priority(self) -> int:
        """Return lower priority values for more trusted candidate sources."""
        return _SOURCE_PRIORITY[self.source]

    @property
    def wildcard_info(self) -> tuple[int, str] | None:
        """Return the (index, wildcard_name) of the first wildcard token if any."""
        if self._has_wildcard is None:
            wildcards = wildcard_slot_names_sorted(self.language)
            info = None
            if wildcards:
                text = self.text
                if any(wc in text for wc in wildcards):
                    sentence_template = self.metadata.get("sentence_template")
                    wildcard_slots_meta = self.metadata.get("wildcard_slots")
                    if sentence_template is not None:
                        wildcard_slots = (
                            frozenset(wildcard_slots_meta.split(","))
                            if wildcard_slots_meta
                            else frozenset()
                        )
                        active_wildcards = [wc for wc in wildcards if wc in wildcard_slots]
                    else:
                        active_wildcards = list(wildcards)

                    if active_wildcards:
                        for idx, token in enumerate(self.normalized_tokens):
                            for wc in active_wildcards:
                                if wc in token:
                                    info = (idx, wc)
                                    break
                            if info is not None:
                                break
            object.__setattr__(self, "_wildcard_info", info)
            object.__setattr__(self, "_has_wildcard", info is not None)
        return self._wildcard_info

    @property
    def has_wildcard(self) -> bool:
        """Return whether the candidate text contains any wildcard placeholders."""
        return self.wildcard_info is not None


def deduplicate_candidates(candidates: list[Candidate]) -> tuple[Candidate, ...]:
    """Deduplicate by (normalized text, intent name) preserving best source priority.

    Candidates with identical text but different intents are kept, the downstream
    HassIL validation loop resolves the intent based on real home context
    (requires_context / excludes_context).
    """
    selected: dict[tuple[str, str], Candidate] = {}
    for candidate in candidates:
        key = (candidate.normalized_text, candidate.intent_name)
        existing = selected.get(key)
        if existing is None or candidate.source_priority < existing.source_priority:
            selected[key] = candidate
    return tuple(
        sorted(selected.values(), key=lambda item: (item.source_priority, item.normalized_text))
    )
