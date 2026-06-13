"""Language-specific canonical candidate indexes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import orjson

from .bm25 import BM25Index
from .candidate import Candidate, deduplicate_candidates
from .const import DEFAULT_MAX_CANDIDATES, DEFAULT_MAX_TOTAL_CANDIDATES_PER_LANGUAGE
from .normalization import char_ngrams_normalized, normalize_text
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
    _exact_normalized_lookup: dict[str, list[Candidate]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    _exact_no_diacritics_lookup: dict[str, list[Candidate]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        """Prebuild reusable lexical ranking structures."""
        slot_tokens = set()
        for candidate in self.candidates:
            slots_json = candidate.metadata.get("slots")
            if slots_json:
                try:
                    c_slots = orjson.loads(slots_json)
                    for val in c_slots.values():
                        val_norm = normalize_text(str(val))
                        slot_tokens.update(val_norm.split())
                except Exception:
                    pass

        unique_templates = set()
        for candidate in self.candidates:
            temp = candidate.metadata.get("sentence_template")
            if temp:
                unique_templates.add(temp)

        from collections import Counter

        tdf = Counter()
        for temp in unique_templates:
            tokens = set(normalize_text(temp).split())
            tdf.update(tokens)

        total_templates = len(unique_templates)
        ignored_tokens = set()
        if total_templates > 0:
            for token, count in tdf.items():
                ratio = count / total_templates
                if ratio > 0.05 and len(token) <= 3 and token not in slot_tokens:
                    ignored_tokens.add(token)

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
        for candidate in self.candidates:
            exact_normalized.setdefault(candidate.normalized_text, []).append(candidate)
            no_diac = candidate.normalized_text_no_diacritics
            exact_no_diacritics.setdefault(no_diac, []).append(candidate)
            literal_text = candidate.metadata.get("literal_text")
            if literal_text:
                for variant in _literal_token_variants(literal_text):
                    all_tokens.update(variant)
        object.__setattr__(self, "_positional_literal_tokens", frozenset(all_tokens))
        object.__setattr__(self, "_exact_normalized_lookup", exact_normalized)
        object.__setattr__(self, "_exact_no_diacritics_lookup", exact_no_diacritics)

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
            exact_normalized_lookup=self._exact_normalized_lookup,
            exact_no_diacritics_lookup=self._exact_no_diacritics_lookup,
            language=self.language,
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
