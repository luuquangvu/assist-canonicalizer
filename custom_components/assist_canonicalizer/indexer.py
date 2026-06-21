"""Language-specific canonical candidate indexes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from .bm25 import BM25Index
from .candidate import Candidate, deduplicate_candidates
from .const import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_TOTAL_CANDIDATES_PER_LANGUAGE,
)
from .normalization import char_ngrams_normalized, literal_tokens_list
from .ranking import CharNGramIndex, RankedCandidate, rank_candidates
from .rehydration import wildcard_variants_analysis


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

    def __post_init__(self) -> None:
        """Prebuild reusable lexical ranking structures."""
        normalized_texts = tuple(candidate.normalized_text for candidate in self.candidates)
        object.__setattr__(
            self,
            "_bm25_index",
            BM25Index.from_normalized_texts(normalized_texts),
        )
        object.__setattr__(
            self,
            "_candidate_char_index",
            CharNGramIndex.from_grams(tuple(char_ngrams_normalized(t) for t in normalized_texts)),
        )
        all_tokens: set[str] = set()
        exact_normalized: dict[str, list[Candidate]] = {}
        exact_no_diacritics: dict[str, list[Candidate]] = {}
        wildcard_always_passes = []
        wildcard_variants_with_len = {}
        wildcard_token_to_indices = defaultdict(list)

        for i, candidate in enumerate(self.candidates):
            exact_normalized.setdefault(candidate.normalized_text, []).append(candidate)
            no_diac = candidate.normalized_text_no_diacritics
            exact_no_diacritics.setdefault(no_diac, []).append(candidate)
            literal_text = candidate.metadata.get("literal_text")
            if literal_text:
                all_tokens.update(literal_tokens_list(literal_text))

            if candidate.has_wildcard:
                var_with_len, all_tokens_set = wildcard_variants_analysis(candidate)
                wildcard_variants_with_len[i] = var_with_len

                always_passes = not candidate.literal_variants or any(
                    length == 0 for _, length, _ in var_with_len
                )
                if always_passes:
                    wildcard_always_passes.append(i)

                for tok in all_tokens_set:
                    wildcard_token_to_indices[tok].append(i)

        object.__setattr__(self, "_positional_literal_tokens", frozenset(all_tokens))
        object.__setattr__(self, "_exact_normalized_lookup", exact_normalized)
        object.__setattr__(self, "_exact_no_diacritics_lookup", exact_no_diacritics)
        object.__setattr__(self, "_wildcard_always_passes", frozenset(wildcard_always_passes))
        object.__setattr__(self, "_wildcard_variants_with_len", wildcard_variants_with_len)
        object.__setattr__(
            self,
            "_wildcard_token_to_indices",
            {tok: tuple(indices) for tok, indices in wildcard_token_to_indices.items()},
        )

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
    ) -> tuple[RankedCandidate, ...]:
        """Rank indexed candidates for a query."""
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
            slot_preferences=slot_preferences,
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
