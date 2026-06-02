"""Unit tests for ExtractionPipeline (REQ-02, REQ-03, REQ-09, REQ-10)."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from research_to_dev.extraction.pipeline import ExtractionPipeline
from research_to_dev.extraction.types import (
    AcademicContent,
    ExtractionError,
    ExtractionResult,
    GeneralContent,
)
from research_to_dev.retriever.protocol import RetrievalResult, SearchResult


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture
def arxiv_result() -> SearchResult:
    return SearchResult(
        source="arxiv",
        title="Quantum Paper",
        url="https://arxiv.org/abs/1234.5678",
        abstract="Quantum computing advances.",
        authors=["Alice"],
        published_date="2025-01-01",
    )


@pytest.fixture
def s2_result() -> SearchResult:
    return SearchResult(
        source="semantic_scholar",
        title="S2 Paper",
        url="https://semanticscholar.org/paper/abc",
        abstract="Interesting findings.",
    )


@pytest.fixture
def tavily_result() -> SearchResult:
    return SearchResult(
        source="tavily",
        title="Web Article",
        url="https://example.com/article",
        abstract="A web article about ML.",
        metadata={
            "raw_content": "<html><body><p>Full article text here.</p></body></html>",
            "score": 0.92,
        },
    )


@pytest.fixture
def mock_scraper() -> AsyncMock:
    scraper = AsyncMock()
    scraper.fetch.return_value = "<html><body>Scraped content.</body></html>"
    return scraper


@pytest.fixture
def mock_compressor() -> AsyncMock:
    compressor = AsyncMock()
    compressor.compress.return_value = "compressed text"
    return compressor


# -------------------------------------------------------------------
# REQ-02, REQ-03: Mixed-source routing
# -------------------------------------------------------------------


class TestSourceRouting:
    """Tests that results are routed to correct extraction path."""

    @pytest.mark.asyncio
    async def test_mixed_sources_produces_typed_output(
        self,
        arxiv_result: SearchResult,
        s2_result: SearchResult,
        tavily_result: SearchResult,
        mock_scraper: AsyncMock,
    ) -> None:
        """REQ-02: RetrievalResult with mixed sources → typed content objects."""
        retrieval = RetrievalResult(
            results=[arxiv_result, s2_result, tavily_result],
            errors=[],
        )

        # Mock trafilatura for the tavily path
        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            return_value="Full article text here.",
        ):
            pipeline = ExtractionPipeline(scraper=mock_scraper)
            result = await pipeline.extract(retrieval)

        assert len(result.contents) == 3
        assert result.errors == []

        # Academic sources
        assert isinstance(result.contents[0], AcademicContent)
        assert result.contents[0].source == "arxiv"
        assert result.contents[0].title == "Quantum Paper"

        assert isinstance(result.contents[1], AcademicContent)
        assert result.contents[1].source == "semantic_scholar"

        # General source
        assert isinstance(result.contents[2], GeneralContent)
        assert result.contents[2].source == "tavily"
        assert result.contents[2].body == "Full article text here."

    @pytest.mark.asyncio
    async def test_academic_source_routes_to_academic_path(
        self, arxiv_result: SearchResult, mock_scraper: AsyncMock
    ) -> None:
        """REQ-03: source=arxiv → AcademicContent, no scraping."""
        retrieval = RetrievalResult(results=[arxiv_result], errors=[])

        pipeline = ExtractionPipeline(scraper=mock_scraper)
        result = await pipeline.extract(retrieval)

        assert len(result.contents) == 1
        assert isinstance(result.contents[0], AcademicContent)
        assert result.contents[0].abstract == "Quantum computing advances."

    @pytest.mark.asyncio
    async def test_tavily_source_routes_to_general_path(
        self,
        tavily_result: SearchResult,
        mock_scraper: AsyncMock,
    ) -> None:
        """REQ-03: source=tavily → GeneralContent."""
        retrieval = RetrievalResult(results=[tavily_result], errors=[])

        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            return_value="Cleaned article.",
        ):
            pipeline = ExtractionPipeline(scraper=mock_scraper)
            result = await pipeline.extract(retrieval)

        assert len(result.contents) == 1
        assert isinstance(result.contents[0], GeneralContent)
        assert result.contents[0].body == "Cleaned article."


# -------------------------------------------------------------------
# REQ-09: Concurrent processing & isolated failure
# -------------------------------------------------------------------


class TestConcurrencyAndFailures:
    """Tests for concurrent processing and partial failure tolerance."""

    @pytest.mark.asyncio
    async def test_isolated_failure_does_not_block_others(
        self,
        arxiv_result: SearchResult,
        s2_result: SearchResult,
        tavily_result: SearchResult,
        mock_scraper: AsyncMock,
    ) -> None:
        """REQ-09: One result's scraper fails — others unaffected."""
        retrieval = RetrievalResult(
            results=[arxiv_result, tavily_result, s2_result],
            errors=[],
        )

        # Make the scraper fail for the tavily result
        mock_scraper.fetch.side_effect = httpx.TimeoutException("timed out")

        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            return_value=None,  # triggers fallback to scraper
        ):
            pipeline = ExtractionPipeline(scraper=mock_scraper)
            result = await pipeline.extract(retrieval)

        # Academic results should be unaffected
        assert len(result.contents) == 3  # All three produce content
        assert isinstance(result.contents[0], AcademicContent)  # arxiv: ok
        assert isinstance(result.contents[2], AcademicContent)  # s2: ok

        # Tavily result: GeneralContent but degraded (body="")
        tavily_content = result.contents[1]
        assert isinstance(tavily_content, GeneralContent)
        assert tavily_content.body == ""

        # Error should be recorded
        assert len(result.errors) == 1
        assert result.errors[0].source == "tavily"
        assert "timed out" in result.errors[0].message

    @pytest.mark.asyncio
    async def test_multiple_failures_all_recorded(
        self,
        tavily_result: SearchResult,
        mock_scraper: AsyncMock,
    ) -> None:
        """Multiple tavily results, all scraper fails → all errors recorded."""
        r1 = SearchResult(
            source="tavily",
            title="Article 1",
            url="https://example.com/1",
            metadata={},
        )
        r2 = SearchResult(
            source="tavily",
            title="Article 2",
            url="https://example.com/2",
            metadata={},
        )
        retrieval = RetrievalResult(results=[r1, r2], errors=[])

        mock_scraper.fetch.side_effect = httpx.TimeoutException("timeout")

        pipeline = ExtractionPipeline(scraper=mock_scraper)
        result = await pipeline.extract(retrieval)

        assert len(result.contents) == 2
        assert len(result.errors) == 2
        assert all(e.exception_type == "TimeoutException" for e in result.errors)

    @pytest.mark.asyncio
    async def test_academic_extraction_never_fails(
        self, arxiv_result: SearchResult, mock_scraper: AsyncMock
    ) -> None:
        """Academic extraction is a pure function — never fails."""
        retrieval = RetrievalResult(results=[arxiv_result], errors=[])

        pipeline = ExtractionPipeline(scraper=mock_scraper)
        result = await pipeline.extract(retrieval)

        assert len(result.contents) == 1
        assert result.errors == []


# -------------------------------------------------------------------
# REQ-10: Empty input
# -------------------------------------------------------------------


class TestEmptyInput:
    """Tests for edge case: empty RetrievalResult."""

    @pytest.mark.asyncio
    async def test_empty_retrieval_returns_empty_result(
        self, mock_scraper: AsyncMock
    ) -> None:
        """REQ-10: Empty results → empty ExtractionResult."""
        retrieval = RetrievalResult(results=[], errors=[])
        pipeline = ExtractionPipeline(scraper=mock_scraper)
        result = await pipeline.extract(retrieval)

        assert result.contents == []
        assert result.errors == []
        assert isinstance(result, ExtractionResult)


# -------------------------------------------------------------------
# REQ-08: Compression threshold
# -------------------------------------------------------------------


class TestCompressionThreshold:
    """Tests for threshold-based compressor routing."""

    @pytest.mark.asyncio
    async def test_below_threshold_compressor_not_called(
        self,
        tavily_result: SearchResult,
        mock_scraper: AsyncMock,
        mock_compressor: AsyncMock,
    ) -> None:
        """Content below threshold → compressor NOT invoked."""
        retrieval = RetrievalResult(results=[tavily_result], errors=[])

        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            return_value="Short text.",  # 11 chars, well below 4000
        ):
            pipeline = ExtractionPipeline(
                scraper=mock_scraper,
                compressor=mock_compressor,
                compress_threshold=4000,
            )
            result = await pipeline.extract(retrieval)

        assert len(result.contents) == 1
        assert result.contents[0].body == "Short text."
        mock_compressor.compress.assert_not_called()

    @pytest.mark.asyncio
    async def test_above_threshold_compressor_called(
        self,
        tavily_result: SearchResult,
        mock_scraper: AsyncMock,
        mock_compressor: AsyncMock,
    ) -> None:
        """Content above threshold → compressor invoked."""
        long_text = "x" * 5000  # 5000 chars > 4000 threshold
        retrieval = RetrievalResult(results=[tavily_result], errors=[])

        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            return_value=long_text,
        ):
            pipeline = ExtractionPipeline(
                scraper=mock_scraper,
                compressor=mock_compressor,
                compress_threshold=4000,
            )
            result = await pipeline.extract(retrieval)

        mock_compressor.compress.assert_called_once_with(long_text)
        assert result.contents[0].body == "compressed text"

    @pytest.mark.asyncio
    async def test_empty_body_skips_compression(
        self,
        tavily_result: SearchResult,
        mock_scraper: AsyncMock,
        mock_compressor: AsyncMock,
    ) -> None:
        """Empty body → compressor NOT called (guard clause)."""
        retrieval = RetrievalResult(results=[tavily_result], errors=[])

        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            return_value=None,  # empty extraction
        ):
            # Make scraper also return unusable content
            mock_scraper.fetch.return_value = ""
            pipeline = ExtractionPipeline(
                scraper=mock_scraper,
                compressor=mock_compressor,
            )
            result = await pipeline.extract(retrieval)

        mock_compressor.compress.assert_not_called()
        assert result.contents[0].body == ""


# -------------------------------------------------------------------
# Integration-style: full pipeline without mocks (unit)
# -------------------------------------------------------------------


class TestExtractionResultShape:
    """Verify ExtractionResult structure from pipeline output."""

    @pytest.mark.asyncio
    async def test_result_has_correct_type(
        self, arxiv_result: SearchResult, mock_scraper: AsyncMock
    ) -> None:
        retrieval = RetrievalResult(results=[arxiv_result], errors=[])
        pipeline = ExtractionPipeline(scraper=mock_scraper)
        result = await pipeline.extract(retrieval)

        assert isinstance(result, ExtractionResult)
        assert isinstance(result.contents, list)
        assert isinstance(result.errors, list)
