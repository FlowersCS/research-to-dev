"""Pytest configuration and shared fixtures for extraction tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from research_to_dev.retriever.protocol import SearchResult


@pytest.fixture
def mock_scraper() -> AsyncMock:
    """AsyncMock that satisfies the Scraper Protocol."""
    scraper = AsyncMock()
    scraper.fetch.return_value = "<html><body>Mocked content.</body></html>"
    return scraper


@pytest.fixture
def mock_compressor() -> AsyncMock:
    """AsyncMock that satisfies the Compressor Protocol."""
    compressor = AsyncMock()
    compressor.compress.return_value = "compressed"
    return compressor


@pytest.fixture
def sample_arxiv_result() -> SearchResult:
    """A typical arXiv SearchResult with all fields populated."""
    return SearchResult(
        source="arxiv",
        title="Sample ArXiv Paper",
        url="https://arxiv.org/abs/2501.00001",
        abstract="Sample abstract for testing.",
        authors=["Author One", "Author Two"],
        published_date="2025-01-15",
        metadata={"pdf_url": "https://arxiv.org/pdf/2501.00001"},
    )


@pytest.fixture
def sample_tavily_result() -> SearchResult:
    """A typical Tavily SearchResult with raw_content in metadata."""
    return SearchResult(
        source="tavily",
        title="Sample Web Article",
        url="https://example.com/article",
        abstract="A sample web article about technology.",
        metadata={
            "raw_content": "<html><body><p>Full article text.</p></body></html>",
            "score": 0.93,
        },
    )


@pytest.fixture
def sample_s2_result() -> SearchResult:
    """A typical Semantic Scholar SearchResult."""
    return SearchResult(
        source="semantic_scholar",
        title="Sample S2 Paper",
        url="https://www.semanticscholar.org/paper/abc123",
        abstract="Semantic Scholar abstract.",
        authors=["Jane Smith"],
        published_date="2024-06-01",
        metadata={"citation_count": 42},
    )
