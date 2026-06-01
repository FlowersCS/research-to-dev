"""Tests for the retriever protocol and data contracts."""

from __future__ import annotations

from research_to_dev.retriever.protocol import (
    Retriever,
    RetrievalError,
    RetrievalResult,
    SearchResult,
)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    """SearchResult dataclass tests."""

    def test_minimal_construction(self) -> None:
        """Only source and title are required."""
        sr = SearchResult(source="arxiv", title="A Paper")
        assert sr.source == "arxiv"
        assert sr.title == "A Paper"
        assert sr.url is None
        assert sr.abstract is None
        assert sr.authors is None
        assert sr.published_date is None
        assert sr.metadata == {}

    def test_full_construction(self) -> None:
        """All fields can be populated."""
        sr = SearchResult(
            source="semantic_scholar",
            title="Deep Learning",
            url="https://semanticscholar.org/paper/abc",
            abstract="A method for...",
            authors=["Alice", "Bob"],
            published_date="2024-01-15",
            metadata={"citations": 42},
        )
        assert sr.source == "semantic_scholar"
        assert sr.title == "Deep Learning"
        assert sr.url == "https://semanticscholar.org/paper/abc"
        assert sr.abstract == "A method for..."
        assert sr.authors == ["Alice", "Bob"]
        assert sr.published_date == "2024-01-15"
        assert sr.metadata == {"citations": 42}

    def test_metadata_defaults_to_empty_dict(self) -> None:
        """metadata field uses default_factory, not shared mutable."""
        a = SearchResult(source="arxiv", title="A")
        b = SearchResult(source="arxiv", title="B")
        a.metadata["key"] = "val"
        assert b.metadata == {}

    def test_source_must_be_literal(self) -> None:
        """The Literal type constrains source at the type-checker level."""
        # Runtime: Python doesn't enforce Literal, but we verify the
        # expected values are accepted.
        for valid in ("tavily", "arxiv", "semantic_scholar"):
            sr = SearchResult(source=valid, title="x")  # type: ignore[arg-type]
            assert sr.source == valid


# ---------------------------------------------------------------------------
# RetrievalError
# ---------------------------------------------------------------------------


class TestRetrievalError:
    """RetrievalError dataclass tests."""

    def test_minimal_construction(self) -> None:
        err = RetrievalError(source="tavily", message="timeout")
        assert err.source == "tavily"
        assert err.message == "timeout"
        assert err.exception_type is None

    def test_with_exception_type(self) -> None:
        err = RetrievalError(
            source="arxiv",
            message="parsing failed",
            exception_type="ParseError",
        )
        assert err.exception_type == "ParseError"


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    """RetrievalResult dataclass tests."""

    def test_empty(self) -> None:
        rr = RetrievalResult(results=[], errors=[])
        assert rr.results == []
        assert rr.errors == []

    def test_partial_failure(self) -> None:
        results = [
            SearchResult(source="arxiv", title="Paper A"),
            SearchResult(source="semantic_scholar", title="Paper B"),
        ]
        errors = [RetrievalError(source="tavily", message="API key missing")]
        rr = RetrievalResult(results=results, errors=errors)
        assert len(rr.results) == 2
        assert len(rr.errors) == 1
        assert rr.errors[0].source == "tavily"


# ---------------------------------------------------------------------------
# Retriever Protocol
# ---------------------------------------------------------------------------


class ValidRetriever:
    """Class structurally conforming to the Retriever Protocol."""

    async def retrieve(self, query: str, max_results: int = 10) -> list[SearchResult]:  # noqa: D102
        return [
            SearchResult(source="arxiv", title=query),
        ]


class MissingMethod:
    """Class missing the retrieve coroutine — does NOT conform."""

    async def search(self, query: str) -> list[SearchResult]:  # noqa: D102
        return []


class TestRetrieverProtocol:
    """Retriever Protocol structural conformance tests."""

    def test_structurally_conforming(self) -> None:
        """A class with the correct method signature is a Retriever."""
        instance = ValidRetriever()
        assert isinstance(instance, Retriever)

    def test_non_conforming(self) -> None:
        """A class without `retrieve` is NOT a Retriever."""
        instance = MissingMethod()
        assert not isinstance(instance, Retriever)
