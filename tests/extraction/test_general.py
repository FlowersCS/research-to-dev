"""Unit tests for general content extraction (REQ-05, REQ-06, REQ-07)."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from research_to_dev.extraction.general import extract_general_content, _clean_html
from research_to_dev.extraction.types import GeneralContent
from research_to_dev.retriever.protocol import SearchResult


@pytest.fixture
def mock_scraper() -> AsyncMock:
    scraper = AsyncMock()
    scraper.fetch.return_value = "<html><body><p>Scraped content here.</p></body></html>"
    return scraper


@pytest.fixture
def tavily_result_with_raw() -> SearchResult:
    return SearchResult(
        source="tavily",
        title="Deep Learning Guide",
        url="https://example.com/dl",
        abstract="An overview of deep learning.",
        metadata={
            "raw_content": "<html><body><p>Deep learning is transformative.</p></body></html>",
            "score": 0.95,
        },
    )


@pytest.fixture
def tavily_result_no_raw() -> SearchResult:
    return SearchResult(
        source="tavily",
        title="ML Basics",
        url="https://example.com/ml",
        abstract="Machine learning fundamentals.",
        metadata={"score": 0.87},
    )


@pytest.fixture
def tavily_result_no_url() -> SearchResult:
    return SearchResult(
        source="tavily",
        title="No URL Result",
        url=None,
        metadata={},
    )


# -------------------------------------------------------------------
# REQ-05: GeneralContent from raw_content
# -------------------------------------------------------------------


class TestRawContentPath:
    """Tests when metadata[\"raw_content\"] is present."""

    @pytest.mark.asyncio
    async def test_cleans_raw_content_with_trafilatura(
        self,
        tavily_result_with_raw: SearchResult,
        mock_scraper: AsyncMock,
    ) -> None:
        """raw_content present → trafilatura.extract() → cleaned body (REQ-05)."""
        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            return_value="Deep learning is transformative.",
        ):
            content = await extract_general_content(
                tavily_result_with_raw, scraper=mock_scraper
            )

        assert isinstance(content, GeneralContent)
        assert content.source == "tavily"
        assert content.title == "Deep Learning Guide"
        assert content.url == "https://example.com/dl"
        assert content.body == "Deep learning is transformative."
        assert content.metadata["score"] == 0.95
        # Scraper should NOT be called when raw_content is available
        mock_scraper.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_trafilatura_returns_none_triggers_fallback(
        self,
        tavily_result_with_raw: SearchResult,
        mock_scraper: AsyncMock,
    ) -> None:
        """trafilatura returns None → fallback to scraper (REQ-06: empty)."""
        # First call (on raw_content) → None triggers fallback.
        # Second call (on scraped HTML) → returns cleaned text.
        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            side_effect=[None, "Scraped content here."],
        ):
            content = await extract_general_content(
                tavily_result_with_raw, scraper=mock_scraper
            )

        assert content.body == "Scraped content here."
        mock_scraper.fetch.assert_called_once_with("https://example.com/dl")

    @pytest.mark.asyncio
    async def test_trafilatura_returns_empty_string_triggers_fallback(
        self,
        tavily_result_with_raw: SearchResult,
        mock_scraper: AsyncMock,
    ) -> None:
        """trafilatura returns empty/whitespace string → fallback to scraper."""
        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            side_effect=["  ", "Scraped content here."],
        ):
            content = await extract_general_content(
                tavily_result_with_raw, scraper=mock_scraper
            )

        assert content.body == "Scraped content here."
        mock_scraper.fetch.assert_called_once()


# -------------------------------------------------------------------
# REQ-07: Scraping fallback
# -------------------------------------------------------------------


class TestScraperFallback:
    """Tests when raw_content is absent — scraper fallback path."""

    @pytest.mark.asyncio
    async def test_missing_raw_content_falls_back_to_scraper(
        self,
        tavily_result_no_raw: SearchResult,
        mock_scraper: AsyncMock,
    ) -> None:
        """No raw_content → scraper.fetch() → trafilatura → body (REQ-07)."""
        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            return_value="Machine learning is cool.",
        ):
            content = await extract_general_content(
                tavily_result_no_raw, scraper=mock_scraper
            )

        mock_scraper.fetch.assert_called_once_with("https://example.com/ml")
        assert content.body == "Machine learning is cool."

    @pytest.mark.asyncio
    async def test_scraper_timeout_propagates(
        self,
        tavily_result_no_raw: SearchResult,
    ) -> None:
        """Scraper timeout raises to caller."""
        scraper = AsyncMock()
        scraper.fetch.side_effect = httpx.TimeoutException("timed out")

        with pytest.raises(httpx.TimeoutException):
            await extract_general_content(tavily_result_no_raw, scraper=scraper)

    @pytest.mark.asyncio
    async def test_scraper_http_error_propagates(
        self,
        tavily_result_no_raw: SearchResult,
    ) -> None:
        """Scraper HTTP error raises to caller."""
        scraper = AsyncMock()
        scraper.fetch.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=httpx.Request("GET", "https://example.com/ml"),
            response=httpx.Response(404),
        )

        with pytest.raises(httpx.HTTPStatusError):
            await extract_general_content(tavily_result_no_raw, scraper=scraper)


# -------------------------------------------------------------------
# REQ-06: Empty extraction
# -------------------------------------------------------------------


class TestEmptyExtraction:
    """Tests when all paths yield no content."""

    @pytest.mark.asyncio
    async def test_no_url_and_no_raw_content_returns_empty_body(
        self,
        tavily_result_no_url: SearchResult,
        mock_scraper: AsyncMock,
    ) -> None:
        """No URL and no raw_content → body=\"\", scraper not called."""
        content = await extract_general_content(
            tavily_result_no_url, scraper=mock_scraper
        )

        assert content.body == ""
        assert content.title == "No URL Result"
        assert content.source == "tavily"
        mock_scraper.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_both_paths_fail_returns_empty_body(
        self,
        tavily_result_with_raw: SearchResult,
        mock_scraper: AsyncMock,
    ) -> None:
        """trafilatura fails + scraper returns empty → body=\"\"."""
        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            side_effect=[None, ""],
        ):
            content = await extract_general_content(
                tavily_result_with_raw, scraper=mock_scraper
            )

        # First call to trafilatura (on raw_content) returned None
        # → fallback to scraper → second call returned ""
        assert content.body == ""
        mock_scraper.fetch.assert_called_once()


# -------------------------------------------------------------------
# _clean_html helper
# -------------------------------------------------------------------


class TestCleanHtml:
    """Verify the _clean_html internal helper."""

    def test_normal_html(self) -> None:
        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            return_value="  Cleaned text  ",
        ):
            result = _clean_html("<html><body>test</body></html>")
            assert result == "Cleaned text"

    def test_trafilatura_returns_none(self) -> None:
        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            return_value=None,
        ):
            result = _clean_html("<html><body>test</body></html>")
            assert result == ""

    def test_trafilatura_returns_empty(self) -> None:
        with patch(
            "research_to_dev.extraction.general.trafilatura.extract",
            return_value="",
        ):
            result = _clean_html("<html><body>test</body></html>")
            assert result == ""


# -------------------------------------------------------------------
# Metadata preservation
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_preserved_in_output(
    tavily_result_with_raw: SearchResult,
    mock_scraper: AsyncMock,
) -> None:
    """Regardless of path, metadata from SearchResult carries through."""
    with patch(
        "research_to_dev.extraction.general.trafilatura.extract",
        return_value="Some text",
    ):
        content = await extract_general_content(
            tavily_result_with_raw, scraper=mock_scraper
        )

    assert content.metadata["raw_content"] is not None
    assert content.metadata["score"] == 0.95
