"""Unit tests for SemanticScholarRetriever using respx HTTP mocking."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from research_to_dev.retriever.semantic_scholar import SemanticScholarRetriever

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def s2_data() -> dict:
    return json.loads((FIXTURES / "s2_response.json").read_text())


@pytest.mark.asyncio
async def test_happy_path(s2_data: dict) -> None:
    """Full S2 response returns normalized SearchResults."""
    with respx.mock:
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json=s2_data)
        )
        retriever = SemanticScholarRetriever()
        results = await retriever.retrieve("deep learning", max_results=3)

    assert len(results) == 3
    r = results[0]
    assert r.source == "semantic_scholar"
    assert r.title == "Deep Learning for Image Recognition"
    assert r.url == "https://www.semanticscholar.org/paper/abc123xyz"
    assert r.abstract is not None
    assert r.authors == ["Alice Chen", "Bob Kumar"]
    assert r.published_date == "2024-03-15"
    assert r.metadata.get("citation_count") == 156
    assert r.metadata.get("open_access_pdf") == (
        "https://arxiv.org/pdf/2403.11111.pdf"
    )

    # Paper 2: no open access
    assert results[1].metadata.get("open_access_pdf") is None
    assert results[1].metadata.get("citation_count") == 89

    # Paper 3: minimal — null abstract, year, empty authors
    r3 = results[2]
    assert r3.abstract is None
    assert r3.authors is None
    assert r3.published_date is None
    assert r3.metadata.get("citation_count") == 0


@pytest.mark.asyncio
async def test_rate_limit_429_returns_empty() -> None:
    """HTTP 429 returns empty list, never raises."""
    with respx.mock:
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(429)
        )
        retriever = SemanticScholarRetriever()
        results = await retriever.retrieve("anything")
    assert results == []


@pytest.mark.asyncio
async def test_no_api_key_header_absent() -> None:
    """When no API key is provided, the x-api-key header is absent."""

    async def _check(request: httpx.Request) -> httpx.Response:
        # Verify no x-api-key header
        assert "x-api-key" not in request.headers
        return httpx.Response(200, json={"data": []})

    with respx.mock:
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            side_effect=_check
        )
        retriever = SemanticScholarRetriever(api_key=None)
        await retriever.retrieve("test")


@pytest.mark.asyncio
async def test_with_api_key_header_present() -> None:
    """When an API key IS provided, the x-api-key header is sent."""

    async def _check(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-api-key") == "secret-key"
        return httpx.Response(200, json={"data": []})

    with respx.mock:
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            side_effect=_check
        )
        retriever = SemanticScholarRetriever(api_key="secret-key")
        await retriever.retrieve("test")


@pytest.mark.asyncio
async def test_timeout_returns_empty() -> None:
    """Timeout returns empty list."""
    with respx.mock:
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            side_effect=httpx.TimeoutException("timed out")
        )
        retriever = SemanticScholarRetriever()
        results = await retriever.retrieve("query")
    assert results == []
