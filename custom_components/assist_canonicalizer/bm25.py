"""BM25 scoring for canonical utterance candidates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import log

from .normalization import tokenize_normalized, tokenize_text


@dataclass(frozen=True, slots=True)
class BM25Document:
    """Tokenized BM25 document."""

    text: str
    tokens: tuple[str, ...]


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
        query_tokens = tokenize_text(query)
        if not query_tokens or not self._documents:
            return tuple(0.0 for _ in self._documents)
        raw_scores = self._score_documents(query_tokens)
        max_score = max(raw_scores, default=0.0)
        if max_score <= 0:
            return tuple(0.0 for _ in raw_scores)
        return tuple(score / max_score for score in raw_scores)

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

    def _score_documents(self, query_tokens: tuple[str, ...]) -> tuple[float, ...]:
        """Return unnormalized BM25 scores using precomputed posting contributions."""
        if self._average_length == 0:
            return tuple(0.0 for _ in self._documents)
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
        return tuple(raw_scores)

    def score_custom_documents(
        self,
        query: str,
        documents: Sequence[str],
        k1: float | None = None,
        b: float | None = None,
    ) -> tuple[float, ...]:
        """Score custom normalized documents using this index for IDF/avg_length."""
        query_tokens = tokenize_text(query)
        if not query_tokens or not documents:
            return tuple(0.0 for _ in documents)

        tokenized_docs = [tokenize_normalized(doc) for doc in documents]

        avg_len = self._average_length
        if avg_len == 0:
            return tuple(0.0 for _ in documents)

        document_count = len(self._documents)
        default_idf = log(1 + (document_count - 0 + 0.5) / (0 + 0.5)) if document_count else 0.0

        use_k1 = k1 if k1 is not None else self._k1
        use_b = b if b is not None else self._b
        use_k1_plus_1 = use_k1 + 1
        one_minus_b = 1.0 - use_b
        b_over_avg = use_b / avg_len

        doc_token_counters: list[Counter[str]] = []
        doc_len_factors: list[float] = []
        for tokens in tokenized_docs:
            doc_token_counters.append(Counter(tokens))
            doc_len_factors.append(use_k1 * (one_minus_b + b_over_avg * len(tokens)))

        raw_scores = [0.0] * len(documents)
        seen_tokens: set[str] = set()
        for token in query_tokens:
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            idf = self._inverse_document_frequencies.get(token)
            if idf is None:
                if not any(token in counter for counter in doc_token_counters):
                    continue
                idf = default_idf

            idf_k1_plus_1 = idf * use_k1_plus_1
            for doc_idx, counter in enumerate(doc_token_counters):
                frequency = counter.get(token, 0)
                if frequency == 0:
                    continue
                denominator = frequency + doc_len_factors[doc_idx]
                raw_scores[doc_idx] += (idf_k1_plus_1 * frequency) / denominator

        max_score = max(raw_scores, default=0.0)
        if max_score <= 0:
            return tuple(0.0 for _ in raw_scores)
        return tuple(score / max_score for score in raw_scores)
