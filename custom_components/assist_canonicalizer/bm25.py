"""BM25 scoring for canonical utterance candidates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from math import log
from typing import Any

from .normalization import tokenize_normalized, tokenize_text


@dataclass(frozen=True, slots=True)
class BM25Document:
    """Tokenized BM25 document."""

    text: str
    tokens: tuple[str, ...]


@lru_cache(maxsize=8192)
def _analyze_tokens(tokens: tuple[str, ...]) -> tuple[Counter[str], int]:
    """Compute term frequencies and length for a tokenized document."""
    return Counter(tokens), len(tokens)


@lru_cache(maxsize=8192)
def _analyze_document(text: str) -> tuple[tuple[str, ...], Counter[str], int]:
    """Tokenize and compute term frequencies and length for a document text."""
    tokens = tokenize_normalized(text)
    counter, length = _analyze_tokens(tokens)
    return tokens, counter, length


class BM25Index:
    """Small in-memory BM25 index for candidate ranking."""

    def __init__(self, documents: Sequence[BM25Document], k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize a BM25 index from tokenized documents."""
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self._documents = tuple(documents)
        self._k1 = k1
        self._k1_plus_1 = k1 + 1
        self._b = b
        self._one_minus_b = 1.0 - b
        self._term_frequencies = tuple(Counter(document.tokens) for document in self._documents)
        self._document_lengths = tuple(len(document.tokens) for document in self._documents)
        token_count = sum(self._document_lengths)
        self._average_length = token_count / len(self._documents) if self._documents else 0.0
        self._b_over_avg_len = self._b / self._average_length if self._average_length > 0 else 0.0
        self._document_len_factors = tuple(
            self._k1 * (self._one_minus_b + self._b_over_avg_len * length)
            for length in self._document_lengths
        )
        document_frequencies = self._build_document_frequencies()
        self._inverse_document_frequencies = self._build_inverse_document_frequencies(
            document_frequencies
        )
        self._idf_k1_plus_1 = {
            token: idf * self._k1_plus_1
            for token, idf in self._inverse_document_frequencies.items()
        }
        self._postings = self._build_postings()

    @classmethod
    def from_texts(cls, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> BM25Index:
        """Build a BM25 index from raw text documents."""
        return cls(
            tuple(BM25Document(text=text, tokens=tokenize_text(text)) for text in texts), k1=k1, b=b
        )

    @classmethod
    def from_normalized_texts(
        cls, texts: Sequence[str], k1: float = 1.5, b: float = 0.75
    ) -> BM25Index:
        """Build a BM25 index from already-normalized text documents."""
        return cls(
            tuple(BM25Document(text=text, tokens=tokenize_normalized(text)) for text in texts),
            k1=k1,
            b=b,
        )

    @property
    def size(self) -> int:
        """Return the number of indexed documents."""
        return len(self._documents)

    def score(self, query: str) -> tuple[float, ...]:
        """Return normalized BM25 scores for every indexed document."""
        return self.score_tokens(tokenize_text(query))

    def score_tokens(self, query_tokens: tuple[str, ...]) -> tuple[float, ...]:
        """Return normalized BM25 scores for pre-tokenized query tokens."""
        doc_count = len(self._documents)
        if not query_tokens or not doc_count:
            return (0.0,) * doc_count
        raw_scores = self._score_documents(query_tokens)
        max_score = max(raw_scores, default=0.0)
        if max_score <= 0:
            return (0.0,) * doc_count
        inv_max = 1.0 / max_score
        return tuple([score * inv_max for score in raw_scores])

    def _build_document_frequencies(self) -> dict[str, int]:
        """Build document frequency counts for indexed tokens."""
        frequencies: dict[str, int] = {}
        for document in self._documents:
            for token in set(document.tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    def _build_inverse_document_frequencies(
        self, document_frequencies: dict[str, int]
    ) -> dict[str, float]:
        """Build BM25 inverse document frequency values by token."""
        document_count = len(self._documents)
        return {
            token: log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            for token, document_frequency in document_frequencies.items()
        }

    def _build_postings(self) -> dict[str, tuple[tuple[int, float], ...]]:
        """Build posting lists with precomputed per-document score contributions."""
        postings: dict[str, list[tuple[int, float]]] = {}
        for document_index, term_frequencies in enumerate(self._term_frequencies):
            len_factor = self._document_len_factors[document_index]
            idf_k1p1 = self._idf_k1_plus_1
            for token, frequency in term_frequencies.items():
                idf_mult = idf_k1p1.get(token)
                if idf_mult is None:
                    continue
                denominator = frequency + len_factor
                precomputed = (idf_mult * frequency) / denominator
                postings.setdefault(token, []).append((document_index, precomputed))
        return {token: tuple(values) for token, values in postings.items()}

    def _score_documents(self, query_tokens: tuple[str, ...]) -> list[float]:
        """Return unnormalized BM25 scores using precomputed posting contributions."""
        if self._average_length == 0:
            return [0.0] * len(self._documents)
        raw_scores = [0.0] * len(self._documents)
        seen_tokens: set[str] = set()
        for token in query_tokens:
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            postings = self._postings.get(token)
            if postings is None:
                continue
            for document_index, precomputed_score in postings:
                raw_scores[document_index] += precomputed_score
        return raw_scores

    def score_custom_documents(
        self,
        query: str,
        documents: Sequence[str | Any],
        k1: float | None = None,
        b: float | None = None,
    ) -> tuple[float, ...]:
        """Score custom normalized documents using this index for IDF/avg_length.

        Accepts either strings or Candidate-like objects. Caches tokenized tokens,
        Counters, and lengths to avoid repeated work in hot paths.
        """
        return self.score_custom_documents_tokens(tokenize_text(query), documents, k1=k1, b=b)

    def score_custom_documents_tokens(
        self,
        query_tokens: tuple[str, ...],
        documents: Sequence[str | Any],
        k1: float | None = None,
        b: float | None = None,
    ) -> tuple[float, ...]:
        """Score custom documents using pre-tokenized query tokens."""
        if not query_tokens or not documents:
            return tuple(0.0 for _ in documents)

        avg_len = self._average_length
        if avg_len == 0:
            return tuple(0.0 for _ in documents)

        document_count = len(self._documents)
        default_idf = log(1 + (document_count - 0 + 0.5) / (0 + 0.5)) if document_count else 0.0

        use_k1 = k1 if k1 is not None else self._k1
        use_b = b if b is not None else self._b
        if use_k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= use_b <= 1:
            raise ValueError("b must be between 0 and 1")
        use_k1_plus_1 = use_k1 + 1
        one_minus_b = 1.0 - use_b
        b_over_avg = use_b / avg_len

        doc_token_counters: list[Counter[str]] = []
        doc_len_factors: list[float] = []
        temp_postings: dict[str, list[tuple[int, float]]] = {}

        for doc_idx, doc in enumerate(documents):
            text: str
            tokens: tuple[str, ...] | None
            if isinstance(doc, str):
                text = doc
                tokens = None
            else:
                text = doc.normalized_text
                tokens = doc.normalized_tokens

            if tokens is None:
                tokens, counter, length = _analyze_document(text)
            else:
                counter, length = _analyze_tokens(tokens)

            doc_token_counters.append(counter)
            len_factor = use_k1 * (one_minus_b + b_over_avg * length)
            doc_len_factors.append(len_factor)
            for token, frequency in counter.items():
                temp_postings.setdefault(token, []).append(
                    (doc_idx, frequency / (frequency + len_factor))
                )

        raw_scores = [0.0] * len(documents)
        seen_tokens: set[str] = set()
        for token in query_tokens:
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            postings = temp_postings.get(token)
            if postings is None:
                continue
            idf = self._inverse_document_frequencies.get(token)
            if idf is None:
                idf = default_idf
            idf_k1_plus_1 = idf * use_k1_plus_1
            for doc_idx, precomputed in postings:
                raw_scores[doc_idx] += idf_k1_plus_1 * precomputed

        max_score = max(raw_scores, default=0.0)
        if max_score <= 0:
            return tuple(0.0 for _ in raw_scores)
        return tuple(score / max_score for score in raw_scores)
