"""Unit tests for ArxivRetriever using respx HTTP mocking."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from research_to_dev.retriever.arxiv import ArxivRetriever

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def arxiv_xml() -> str:
    return (FIXTURES / "arxiv_response.xml").read_text()


@pytest.fixture
def arxiv_empty_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: all:nonexistent</title>
  <totalResults>0</totalResults>
</feed>"""


@pytest.fixture
def arxiv_minimal_xml() -> str:
    """Entry with only id and title — no authors, summary, published."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/9999.99999v1</id>
    <title>Minimal Paper</title>
  </entry>
</feed>"""


@pytest.mark.asyncio
async def test_happy_path(arxiv_xml: str) -> None:
    """Full Atom feed returns normalized SearchResults."""
    with respx.mock:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=arxiv_xml)
        )
        retriever = ArxivRetriever(min_interval=0)
        results = await retriever.retrieve("deep learning", max_results=3)

    assert len(results) == 3
    assert results[0].source == "arxiv"
    assert results[0].title == "Deep Learning: A Comprehensive Survey"
    assert results[0].url == "http://arxiv.org/abs/2401.00001v1"
    assert "deep learning techniques" in (results[0].abstract or "")
    assert results[0].authors == ["Alice Researcher", "Bob Scientist"]
    assert results[0].published_date == "2024-01-01T00:00:00Z"

    # Entry 2 lacks pdf link — metadata should be empty
    assert results[1].metadata == {}

    # Entry 1 and 3 have pdf links
    assert results[0].metadata.get("pdf_url") == (
        "http://arxiv.org/pdf/2401.00001v1"
    )
    assert results[2].metadata.get("pdf_url") == (
        "http://arxiv.org/pdf/2403.00003v1"
    )


@pytest.mark.asyncio
async def test_empty_feed(arxiv_empty_xml: str) -> None:
    """A feed with zero entries returns an empty list."""
    with respx.mock:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=arxiv_empty_xml)
        )
        retriever = ArxivRetriever(min_interval=0)
        results = await retriever.retrieve("nonexistent")
    assert results == []


@pytest.mark.asyncio
async def test_minimal_entry(arxiv_minimal_xml: str) -> None:
    """Entries with missing optional fields don't crash. Fields become None."""
    with respx.mock:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=arxiv_minimal_xml)
        )
        retriever = ArxivRetriever(min_interval=0)
        results = await retriever.retrieve("minimal")
    assert len(results) == 1
    r = results[0]
    assert r.title == "Minimal Paper"
    assert r.url == "http://arxiv.org/abs/9999.99999v1"
    assert r.abstract is None
    assert r.authors is None
    assert r.published_date is None
    assert r.metadata == {}


@pytest.mark.asyncio
async def test_http_500_returns_empty() -> None:
    """HTTP 5xx returns empty list, never raises."""
    with respx.mock:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(500)
        )
        retriever = ArxivRetriever(min_interval=0)
        results = await retriever.retrieve("query")
    assert results == []


@pytest.mark.asyncio
async def test_timeout_returns_empty() -> None:
    """Connection timeout returns empty list."""
    with respx.mock:
        respx.get("https://export.arxiv.org/api/query").mock(
            side_effect=httpx.TimeoutException("timed out")
        )
        retriever = ArxivRetriever(min_interval=0)
        results = await retriever.retrieve("query")
    assert results == []
