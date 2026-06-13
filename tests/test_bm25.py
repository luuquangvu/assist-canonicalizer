"""Tests for BM25 search index."""

import pytest

from custom_components.assist_canonicalizer.bm25 import BM25Document, BM25Index


def test_bm25_validation_errors() -> None:
    """Verify BM25Index raises ValueError on invalid parameter boundaries."""
    with pytest.raises(ValueError, match="k1 must be positive"):
        BM25Index([], k1=0)
    with pytest.raises(ValueError, match="k1 must be positive"):
        BM25Index([], k1=-1)
    with pytest.raises(ValueError, match="b must be between 0 and 1"):
        BM25Index([], b=-0.1)
    with pytest.raises(ValueError, match="b must be between 0 and 1"):
        BM25Index([], b=1.1)


def test_bm25_index_size() -> None:
    """Verify size property matches document count."""
    doc1 = BM25Document(text="hello", tokens=("hello",))
    index = BM25Index([doc1])
    assert index.size == 1


def test_bm25_scoring_empty() -> None:
    """Verify scoring with empty queries or empty index documents."""
    index_empty = BM25Index([])
    assert index_empty.score("hello") == ()

    doc1 = BM25Document(text="hello", tokens=("hello",))
    index = BM25Index([doc1])
    assert index.score("  ") == (0.0,)


def test_bm25_scoring_zero_average_length() -> None:
    """Verify scoring handles documents with zero tokens."""
    doc_empty = BM25Document(text="  ", tokens=())
    index = BM25Index([doc_empty])
    assert index.score("hello") == (0.0,)


def test_bm25_from_texts() -> None:
    """Verify building index from raw texts using from_texts classmethod."""
    index = BM25Index.from_texts(["Hello World", "Testing BM25"])
    assert index.size == 2
    assert index.score("World") == (1.0, 0.0)
