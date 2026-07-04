"""Language-specific canonical candidate indexes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .bm25 import BM25Index
from .candidate import Candidate, deduplicate_candidates
from .const import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_TOTAL_CANDIDATES_PER_LANGUAGE,
    DEFAULT_MIN_CONFIDENCE,
)
from .normalization import char_ngrams_normalized, literal_tokens_list
from .ranking import CharNGramIndex, RankedCandidate, rank_candidates
from .rehydration import wildcard_variants_analysis


@dataclass
class _CanonicalIndexData:
    """Derived index structures computed from a candidate list."""

    positional_literal_tokens: frozenset[str]
    exact_normalized_lookup: dict[str, list[Candidate]]
    exact_no_diacritics_lookup: dict[str, list[Candidate]]
    wildcard_always_passes: frozenset[int]
    wildcard_variants_with_len: dict[int, tuple[tuple[frozenset[str], int, int], ...]]
    wildcard_token_to_indices: dict[str, tuple[int, ...]]
    wildcard_literal_tokens_by_index: dict[int, frozenset[str]]
    wildcard_min_required_by_index: dict[int, int]
    candidate_slot_tokens: tuple[frozenset[str], ...]
    slot_token_to_indices: dict[str, tuple[int, ...]]


def _build_index_state(candidates: tuple[Candidate, ...]) -> _CanonicalIndexData:
    """Build all derived index structures from the candidate list."""
    all_tokens: set[str] = set()
    exact_normalized: dict[str, list[Candidate]] = {}
    exact_no_diacritics: dict[str, list[Candidate]] = {}
    wildcard_always_passes: list[int] = []
    wildcard_variants_with_len: dict[int, tuple[tuple[frozenset[str], int, int], ...]] = {}
    wildcard_token_to_indices: defaultdict[str, list[int]] = defaultdict(list)
    wildcard_literal_tokens_by_index: dict[int, frozenset[str]] = {}
    wildcard_min_required_by_index: dict[int, int] = {}
    candidate_slot_tokens: list[frozenset[str]] = []
    slot_token_to_indices: defaultdict[str, list[int]] = defaultdict(list)

    for i, candidate in enumerate(candidates):
        exact_normalized.setdefault(candidate.normalized_text, []).append(candidate)
        no_diac = candidate.normalized_text_no_diacritics
        exact_no_diacritics.setdefault(no_diac, []).append(candidate)
        if literal_text := candidate.metadata.get("literal_text"):
            all_tokens.update(literal_tokens_list(literal_text))
        if candidate.has_wildcard:
            var_with_len, all_tokens_set = wildcard_variants_analysis(candidate)
            wildcard_variants_with_len[i] = var_with_len
            wildcard_literal_tokens_by_index[i] = all_tokens_set
            if var_with_len:
                wildcard_min_required_by_index[i] = min(req for _, _, req in var_with_len)
            always_passes = not candidate.literal_variants or any(
                length == 0 for _, length, _ in var_with_len
            )
            if always_passes:
                wildcard_always_passes.append(i)
            for tok in all_tokens_set:
                wildcard_token_to_indices[tok].append(i)
        slot_tokens = candidate.slot_tokens_set
        candidate_slot_tokens.append(slot_tokens)
        for token in slot_tokens:
            slot_token_to_indices[token].append(i)

    return _CanonicalIndexData(
        positional_literal_tokens=frozenset(all_tokens),
        exact_normalized_lookup=exact_normalized,
        exact_no_diacritics_lookup=exact_no_diacritics,
        wildcard_always_passes=frozenset(wildcard_always_passes),
        wildcard_variants_with_len=wildcard_variants_with_len,
        wildcard_token_to_indices={
            tok: tuple(idxs) for tok, idxs in wildcard_token_to_indices.items()
        },
        wildcard_literal_tokens_by_index=wildcard_literal_tokens_by_index,
        wildcard_min_required_by_index=wildcard_min_required_by_index,
        candidate_slot_tokens=tuple(candidate_slot_tokens),
        slot_token_to_indices={tok: tuple(idxs) for tok, idxs in slot_token_to_indices.items()},
    )


@dataclass(frozen=True, slots=True)
class CanonicalIndex:
    """Immutable candidate index for one language."""

    language: str
    candidates: tuple[Candidate, ...]
    version: int = 1
    _bm25_index: BM25Index = field(init=False, repr=False, compare=False)
    _candidate_char_index: CharNGramIndex = field(init=False, repr=False, compare=False)
    _positional_literal_tokens: frozenset[str] = field(
        init=False, repr=False, compare=False, default_factory=frozenset
    )
    _exact_normalized_lookup: dict[str, list[Candidate]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    _exact_no_diacritics_lookup: dict[str, list[Candidate]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    _wildcard_always_passes: frozenset[int] = field(
        init=False, repr=False, compare=False, default_factory=frozenset
    )
    _wildcard_variants_with_len: dict[int, tuple[tuple[frozenset[str], int, int], ...]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    _wildcard_token_to_indices: dict[str, tuple[int, ...]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    _wildcard_literal_tokens_by_index: dict[int, frozenset[str]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    _wildcard_min_required_by_index: dict[int, int] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    _candidate_slot_tokens: tuple[frozenset[str], ...] = field(
        init=False, repr=False, compare=False, default_factory=tuple
    )
    _slot_token_to_indices: dict[str, tuple[int, ...]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        """Prebuild reusable lexical ranking structures."""
        normalized_texts = tuple(c.normalized_text for c in self.candidates)
        object.__setattr__(self, "_bm25_index", BM25Index.from_normalized_texts(normalized_texts))
        object.__setattr__(
            self,
            "_candidate_char_index",
            CharNGramIndex.from_grams(tuple(char_ngrams_normalized(t) for t in normalized_texts)),
        )
        state = _build_index_state(self.candidates)
        for name, value in vars(state).items():
            object.__setattr__(self, f"_{name}", value)

    @property
    def candidate_count(self) -> int:
        """Return the number of indexed candidates."""
        return len(self.candidates)

    @property
    def bm25_index(self) -> BM25Index:
        """Return the BM25 index."""
        return self._bm25_index

    def rank(
        self,
        query: str,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        *,
        slot_preferences: set[tuple[str, str]] | None = None,
        intent_context: Mapping[str, Any] | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> tuple[RankedCandidate, ...]:
        """Rank indexed candidates for a query.

        ``intent_context`` follows the HassIL-style mapping normalized by
        ``normalize_intent_context`` before ranking.
        """
        return rank_candidates(
            query,
            self.candidates,
            max_candidates=max_candidates,
            bm25_index=self._bm25_index,
            candidate_char_index=self._candidate_char_index,
            positional_literal_tokens=self._positional_literal_tokens,
            exact_normalized_lookup=self._exact_normalized_lookup,
            exact_no_diacritics_lookup=self._exact_no_diacritics_lookup,
            language=self.language,
            wildcard_always_passes=self._wildcard_always_passes,
            wildcard_variants_with_len=self._wildcard_variants_with_len,
            wildcard_token_to_indices=self._wildcard_token_to_indices,
            wildcard_literal_tokens_by_index=self._wildcard_literal_tokens_by_index,
            wildcard_min_required_by_index=self._wildcard_min_required_by_index,
            candidate_slot_tokens=self._candidate_slot_tokens,
            slot_token_to_indices=self._slot_token_to_indices,
            slot_preferences=slot_preferences,
            intent_context=intent_context,
            min_confidence=min_confidence,
        )


def build_index(
    language: str,
    candidates: Sequence[Candidate],
    max_total_candidates: int = DEFAULT_MAX_TOTAL_CANDIDATES_PER_LANGUAGE,
) -> CanonicalIndex:
    """Build a capped, deduplicated canonical index for one language."""
    if not language.strip():
        raise ValueError("Language must not be empty")
    if max_total_candidates < 1:
        raise ValueError("max_total_candidates must be positive")
    matching_language = [
        candidate
        for candidate in candidates
        if candidate.language is None or candidate.language == language
    ]
    deduplicated = deduplicate_candidates(matching_language)
    return CanonicalIndex(language=language, candidates=deduplicated[:max_total_candidates])
