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
from .ranking import (
    CharNGramIndex,
    RankedCandidate,
    SlotTokenIndex,
    WildcardVariantGroup,
    rank_candidates,
)
from .rehydration import WildcardVariantAnalysis, wildcard_variants_analysis


@dataclass
class _CanonicalIndexData:
    """Derived index structures computed from a candidate list."""

    positional_literal_tokens: frozenset[str]
    exact_normalized_lookup: dict[str, list[Candidate]]
    exact_no_diacritics_lookup: dict[str, list[Candidate]]
    wildcard_always_passes: frozenset[int]
    wildcard_variant_analyses: dict[int, tuple[WildcardVariantAnalysis, ...]]
    wildcard_variant_groups: tuple[WildcardVariantGroup, ...]
    candidate_slot_tokens: tuple[frozenset[str], ...]
    slot_token_index: SlotTokenIndex


def _build_index_state(candidates: tuple[Candidate, ...]) -> _CanonicalIndexData:
    """Build all derived index structures from the candidate list."""
    all_tokens: set[str] = set()
    exact_normalized: dict[str, list[Candidate]] = {}
    exact_no_diacritics: dict[str, list[Candidate]] = {}
    wildcard_always_passes: list[int] = []
    wildcard_variant_analyses: dict[int, tuple[WildcardVariantAnalysis, ...]] = {}
    wildcard_group_indices: defaultdict[tuple[WildcardVariantAnalysis, ...], list[int]] = (
        defaultdict(list)
    )
    candidate_slot_tokens: list[frozenset[str]] = []

    for i, candidate in enumerate(candidates):
        exact_normalized.setdefault(candidate.normalized_text, []).append(candidate)
        no_diac = candidate.normalized_text_no_diacritics
        exact_no_diacritics.setdefault(no_diac, []).append(candidate)
        if literal_text := candidate.metadata.get("literal_text"):
            all_tokens.update(literal_tokens_list(literal_text))
        if candidate.has_wildcard:
            variants, _ = wildcard_variants_analysis(candidate)
            wildcard_variant_analyses[i] = variants
            always_passes = not candidate.literal_variants or any(
                not variant.literal_tokens for variant in variants
            )
            if always_passes:
                wildcard_always_passes.append(i)
            else:
                wildcard_group_indices[variants].append(i)
        slot_tokens = candidate.slot_tokens_set
        candidate_slot_tokens.append(slot_tokens)

    frozen_candidate_slot_tokens = tuple(candidate_slot_tokens)
    return _CanonicalIndexData(
        positional_literal_tokens=frozenset(all_tokens),
        exact_normalized_lookup=exact_normalized,
        exact_no_diacritics_lookup=exact_no_diacritics,
        wildcard_always_passes=frozenset(wildcard_always_passes),
        wildcard_variant_analyses=wildcard_variant_analyses,
        wildcard_variant_groups=tuple(
            WildcardVariantGroup(
                variants=variants,
                literal_tokens=frozenset(
                    token for variant in variants for token in variant.literal_tokens
                ),
                min_required_match_count=min(variant.required_match_count for variant in variants),
                candidate_indices=tuple(indices),
            )
            for variants, indices in wildcard_group_indices.items()
        ),
        candidate_slot_tokens=frozen_candidate_slot_tokens,
        slot_token_index=SlotTokenIndex.from_candidate_tokens(frozen_candidate_slot_tokens),
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
    _wildcard_variant_analyses: dict[int, tuple[WildcardVariantAnalysis, ...]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    _wildcard_variant_groups: tuple[WildcardVariantGroup, ...] = field(
        init=False, repr=False, compare=False, default_factory=tuple
    )
    _candidate_slot_tokens: tuple[frozenset[str], ...] = field(
        init=False, repr=False, compare=False, default_factory=tuple
    )
    _slot_token_index: SlotTokenIndex = field(init=False, repr=False, compare=False)

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

    @property
    def positional_literal_tokens(self) -> frozenset[str]:
        """Return normalized literal tokens known across the static corpus."""
        return self._positional_literal_tokens

    @property
    def slot_token_index(self) -> SlotTokenIndex:
        """Return the static target-token posting index."""
        return self._slot_token_index

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
            wildcard_variant_analyses=self._wildcard_variant_analyses,
            wildcard_variant_groups=self._wildcard_variant_groups,
            candidate_slot_tokens=self._candidate_slot_tokens,
            slot_token_index=self._slot_token_index,
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
