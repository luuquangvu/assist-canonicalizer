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
        self._term_frequencies = tuple(Counter(document.tokens) for document in self._documents)
        self._document_lengths = tuple(len(document.tokens) for document in self._documents)
        self._document_frequencies = self._build_document_frequencies()
        self._inverse_document_frequencies = self._build_inverse_document_frequencies()
        self._postings = self._build_postings()
        token_count = sum(self._document_lengths)
        self._average_length = token_count / len(self._documents) if self._documents else 0.0

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

    def _build_inverse_document_frequencies(self) -> dict[str, float]:
        """Build BM25 inverse document frequency values by token."""
        document_count = len(self._documents)
        return {
            token: log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            for token, document_frequency in self._document_frequencies.items()
        }

    def _build_postings(self) -> dict[str, tuple[tuple[int, int], ...]]:
        """Build posting lists of document indexes and term frequencies by token."""
        postings: dict[str, list[tuple[int, int]]] = {}
        for document_index, term_frequencies in enumerate(self._term_frequencies):
            for token, frequency in term_frequencies.items():
                postings.setdefault(token, []).append((document_index, frequency))
        return {token: tuple(values) for token, values in postings.items()}

    def _score_documents(self, query_tokens: tuple[str, ...]) -> tuple[float, ...]:
        """Return unnormalized BM25 scores for all matching documents."""
        if self._average_length == 0:
            return tuple(0.0 for _ in self._documents)
        raw_scores = [0.0] * len(self._documents)
        for token in query_tokens:
            inverse_document_frequency = self._inverse_document_frequencies.get(token)
            if inverse_document_frequency is None:
                continue
            for document_index, frequency in self._postings[token]:
                document_length = self._document_lengths[document_index]
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * document_length / self._average_length
                )
                numerator = inverse_document_frequency * (frequency * self._k1_plus_1)
                raw_scores[document_index] += numerator / denominator
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
        doc_lengths = [len(tokens) for tokens in tokenized_docs]

        avg_len = self._average_length
        if avg_len == 0:
            return tuple(0.0 for _ in documents)

        # Default IDF value for unseen tokens
        document_count = len(self._documents)
        default_idf = log(1 + (document_count - 0 + 0.5) / (0 + 0.5)) if document_count else 0.0

        use_k1 = k1 if k1 is not None else self._k1
        use_b = b if b is not None else self._b

        doc_token_counters = [Counter(tokens) for tokens in tokenized_docs]
        use_k1_plus_1 = use_k1 + 1

        raw_scores = [0.0] * len(documents)
        for token in query_tokens:
            idf = self._inverse_document_frequencies.get(token)
            if idf is None:
                if not any(token in counter for counter in doc_token_counters):
                    continue
                idf = default_idf
            for doc_idx, counter in enumerate(doc_token_counters):
                frequency = counter.get(token, 0)
                if frequency == 0:
                    continue
                denominator = frequency + use_k1 * (
                    1 - use_b + use_b * doc_lengths[doc_idx] / avg_len
                )
                numerator = idf * (frequency * use_k1_plus_1)
                raw_scores[doc_idx] += numerator / denominator

        max_score = max(raw_scores, default=0.0)
        if max_score <= 0:
            return tuple(0.0 for _ in raw_scores)
        return tuple(score / max_score for score in raw_scores)
