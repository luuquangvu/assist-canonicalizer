"""Language-specific canonical candidate indexes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .bm25 import BM25Index
from .candidate import Candidate, deduplicate_candidates
from .const import DEFAULT_MAX_CANDIDATES, DEFAULT_MAX_TOTAL_CANDIDATES_PER_LANGUAGE
from .normalization import char_ngrams_normalized
from .ranking import CharNGramIndex, RankedCandidate, _literal_token_variants, rank_candidates


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
        for candidate in self.candidates:
            literal_text = candidate.metadata.get("literal_text")
            if literal_text:
                for variant in _literal_token_variants(literal_text):
                    all_tokens.update(variant)
        object.__setattr__(self, "_positional_literal_tokens", frozenset(all_tokens))

    @property
    def candidate_count(self) -> int:
        """Return the number of indexed candidates."""
        return len(self.candidates)

    def rank(
        self, query: str, max_candidates: int = DEFAULT_MAX_CANDIDATES
    ) -> tuple[RankedCandidate, ...]:
        """Rank indexed candidates for a query."""
        return rank_candidates(
            query,
            self.candidates,
            max_candidates=max_candidates,
            bm25_index=self._bm25_index,
            candidate_char_index=self._candidate_char_index,
            positional_literal_tokens=self._positional_literal_tokens,
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
