"""Unit tests for TavilyRetriever using respx HTTP mocking."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from research_to_dev.retriever.tavily import TavilyRetriever

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def tavily_data() -> dict:
    return json.loads((FIXTURES / "tavily_response.json").read_text())


@pytest.mark.asyncio
async def test_happy_path(tavily_data: dict) -> None:
    """Full Tavily response returns normalized SearchResults."""
    with respx.mock:
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=tavily_data)
        )
        retriever = TavilyRetriever(api_key="test-key")
        results = await retriever.retrieve("deep learning", max_results=3)

    assert len(results) == 3
    r = results[0]
    assert r.source == "tavily"
    assert r.title == "Applications of Deep Learning in Healthcare"
    assert r.url == "https://example.com/deep-learning-healthcare"
    assert r.abstract is not None
    # Tavily doesn't provide academic fields
    assert r.authors is None
    assert r.published_date is None
    # Metadata
    assert r.metadata.get("raw_content") is not None
    assert r.metadata.get("score") == 0.95

    # Second result has no raw_content
    assert results[1].metadata.get("raw_content") is None
    assert results[1].metadata.get("score") == 0.87

    # Third result has raw_content
    assert results[2].metadata.get("raw_content") is not None
    assert results[2].metadata.get("score") == 0.82


@pytest.mark.asyncio
async def test_include_raw_content_in_payload(tavily_data: dict) -> None:
    """REQ-11: The POST payload MUST include include_raw_content: True."""
    with respx.mock:
        route = respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=tavily_data)
        )
        retriever = TavilyRetriever(api_key="test-key")
        await retriever.retrieve("deep learning", max_results=3)

    # Assert the payload sent to the API includes the flag
    assert route.calls.last is not None
    body = json.loads(route.calls.last.request.content)
    assert body.get("include_raw_content") is True, (
        f"Expected include_raw_content=True in payload, got: {body}"
    )


@pytest.mark.asyncio
async def test_raw_content_populated_in_metadata(tavily_data: dict) -> None:
    """REQ-11: metadata[\"raw_content\"] is populated from Tavily response."""
    with respx.mock:
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=tavily_data)
        )
        retriever = TavilyRetriever(api_key="test-key")
        results = await retriever.retrieve("query")

    # First result has raw_content
    raw = results[0].metadata.get("raw_content")
    assert raw is not None
    assert "Deep learning has revolutionized" in raw

    # Second result: no raw_content in fixture
    assert results[1].metadata.get("raw_content") is None


@pytest.mark.asyncio
async def test_missing_api_key_returns_empty() -> None:
    """Empty or missing API key returns empty list."""
    retriever = TavilyRetriever(api_key="")
    results = await retriever.retrieve("anything")
    assert results == []

    # respx never hit — adapter exits early
    with respx.mock:
        respx.post("https://api.tavily.com/search").respond(200)
        retriever2 = TavilyRetriever(api_key="")
        results2 = await retriever2.retrieve("anything")
    assert results2 == []


@pytest.mark.asyncio
async def test_plan_limit_432_returns_empty() -> None:
    """HTTP 432 (plan limit) returns empty list."""
    with respx.mock:
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(432)
        )
        retriever = TavilyRetriever(api_key="test-key")
        results = await retriever.retrieve("anything")
    assert results == []


@pytest.mark.asyncio
async def test_timeout_returns_empty() -> None:
    """Timeout returns empty list."""
    with respx.mock:
        respx.post("https://api.tavily.com/search").mock(
            side_effect=httpx.TimeoutException("timed out")
        )
        retriever = TavilyRetriever(api_key="test-key")
        results = await retriever.retrieve("query")
    assert results == []
