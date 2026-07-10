"""Tests for BM25 search index."""

import pytest

from custom_components.assist_canonicalizer.bm25 import (
    BM25Document,
    BM25Index,
    _analyze_document,
    _analyze_tokens,
)


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


def test_bm25_custom_documents_validate_params_before_fast_paths() -> None:
    """Verify custom-document scoring validates parameters before empty-index returns."""
    index = BM25Index([])

    with pytest.raises(ValueError, match="k1 must be positive"):
        index.score_custom_documents_tokens(("hello",), ["hello"], k1=0)
    with pytest.raises(ValueError, match="b must be between 0 and 1"):
        index.score_custom_documents_tokens(("hello",), ["hello"], b=1.1)


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


def test_bm25_score_custom_documents() -> None:
    """Test score_custom_documents with identical documents, unseen tokens, and empty inputs."""
    index = BM25Index.from_texts(["Hello World", "Testing BM25"])

    # 1. Identical documents (normalized) compared against index.score
    res_score = index.score("World")
    res_custom = index.score_custom_documents("World", ["hello world", "testing bm25"])
    assert res_custom == res_score

    # 2. Documents with unseen tokens
    # query has "testing" (in index) and "unseen" (not in index)
    res_unseen = index.score_custom_documents("Testing Unseen", ["testing bm25", "unseen text"])
    # "testing bm25" has "testing", "unseen text" has "unseen" (unseen token gets default idf)
    assert res_unseen[0] > 0.0
    assert res_unseen[1] > 0.0

    # 3. Empty inputs
    assert index.score_custom_documents("", ["hello world"]) == (0.0,)
    assert index.score_custom_documents("World", []) == ()


def test_bm25_score_custom_documents_deduplicates_query_tokens() -> None:
    """Verify query token deduplication prevents double counting in score_custom_documents."""
    index = BM25Index.from_texts(["Hello World", "Testing BM25"])
    # Query has repeated tokens
    res_single = index.score_custom_documents("World", ["hello world", "testing bm25"])
    res_repeated = index.score_custom_documents("World World", ["hello world", "testing bm25"])
    assert res_repeated == res_single


@pytest.mark.parametrize(
    ("documents", "query_tokens"),
    [
        ((), ("hello",)),
        ((BM25Document(text="", tokens=()),), ("hello",)),
        (
            (
                BM25Document(text="alpha beta", tokens=("alpha", "beta")),
                BM25Document(text="beta gamma", tokens=("beta", "gamma")),
                BM25Document(text="alpha alpha", tokens=("alpha", "alpha")),
            ),
            ("alpha", "alpha", "missing"),
        ),
        (
            (
                BM25Document(text="đèn phòng khách", tokens=("đèn", "phòng", "khách")),
                BM25Document(text="den phong khach", tokens=("den", "phong", "khach")),
                BM25Document(text="kitchen-light", tokens=("kitchen", "light")),
            ),
            ("đèn", "khách", "light"),
        ),
    ],
)
def test_bm25_raw_scores_sparse_matches_dense(
    documents: tuple[BM25Document, ...],
    query_tokens: tuple[str, ...],
) -> None:
    """Verify sparse BM25 raw scores are dense-equivalent for touched documents."""
    index = BM25Index(documents)

    dense = index.raw_scores(query_tokens)
    sparse = index.raw_scores_sparse(query_tokens)

    assert {idx for idx, score in enumerate(dense) if score > 0.0} == set(sparse)
    assert tuple(sparse.get(idx, 0.0) for idx in range(len(dense))) == tuple(dense)


def test_bm25_score_custom_documents_with_candidates_and_cache() -> None:
    """Verify score_custom_documents accepts Candidate-like objects and uses the cache."""
    index = BM25Index.from_texts(["Hello World", "Testing BM25"])

    # Define a dummy Candidate-like class
    class DummyCandidate:
        """Dummy Candidate-like class for testing."""

        def __init__(self, normalized_text: str, tokens: tuple[str, ...]) -> None:
            """Initialize the dummy candidate."""
            self.normalized_text = normalized_text
            self.normalized_tokens = tokens

    c1 = DummyCandidate("hello world", ("hello", "world"))
    c2 = DummyCandidate("testing bm25", ("testing", "bm25"))

    _analyze_document.cache_clear()
    _analyze_tokens.cache_clear()

    # Verify we can score it
    res = index.score_custom_documents("World BM25", [c1, c2])
    assert res[0] > 0.0
    assert res[1] > 0.0

    # Since tokens are provided, _analyze_tokens should have been called
    info_tok = _analyze_tokens.cache_info()
    assert info_tok.misses == 2
    assert info_tok.hits == 0

    # Rescoring should hit the cache
    res_cached = index.score_custom_documents("World BM25", [c1, c2])
    assert res_cached == res

    info_tok_after = _analyze_tokens.cache_info()
    assert info_tok_after.hits == 2

    # Verify that string-based scoring uses _analyze_document and hits the cache too
    _analyze_document.cache_clear()
    res_str = index.score_custom_documents("World BM25", ["hello world", "testing bm25"])
    assert res_str == res
    info_doc = _analyze_document.cache_info()
    assert info_doc.misses == 2
    assert info_doc.hits == 0

    res_str_cached = index.score_custom_documents("World BM25", ["hello world", "testing bm25"])
    assert res_str_cached == res_str
    info_doc_after = _analyze_document.cache_info()
    assert info_doc_after.hits == 2


def test_bm25_score_custom_documents_falls_back_to_normalized_text() -> None:
    """Candidate-like objects may expose normalized_text without cached tokens."""
    index = BM25Index.from_texts(["Hello World", "Testing BM25"])

    class TextOnlyCandidate:
        """Dummy Candidate-like object without normalized_tokens."""

        def __init__(self, normalized_text: str) -> None:
            """Initialize the normalized text."""
            self.normalized_text = normalized_text

    assert index.score_custom_documents(
        "World",
        [TextOnlyCandidate("hello world")],
    ) == index.score_custom_documents("World", ["hello world"])
