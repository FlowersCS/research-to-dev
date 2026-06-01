"""Live integration tests — validate adapters against real APIs.

All tests here are marked ``@pytest.mark.integration`` and are
NOT run during normal ``pytest`` invocations.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from research_to_dev.retriever.arxiv import ArxivRetriever
from research_to_dev.retriever.semantic_scholar import SemanticScholarRetriever
from research_to_dev.retriever.tavily import TavilyRetriever


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="session")
def _env_loaded() -> None:
    """Load .env once per test session so API keys are available."""
    load_dotenv()


def _skip_if_missing(key: str, service: str) -> str:
    """Return the env value or skip the test with a clear message."""
    value = os.getenv(key, "").strip()
    if not value:
        pytest.skip(f"{service} API key ({key}) not set — skipping live test")
    return value


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_arxiv_live_query() -> None:
    """Query the real arXiv API and verify structural integrity.

    This validates that the Atom XML schema hasn't changed
    and our parser handles real data correctly.
    """
    retriever = ArxivRetriever(min_interval=0)
    results = await retriever.retrieve("machine learning", max_results=3)

    # Should get actual results from arXiv
    assert len(results) >= 1, "Expected at least one result from live arXiv"

    for r in results:
        assert r.source == "arxiv"
        assert r.title, "Every result must have a title"
        assert r.url is not None, "Every result must have a URL"
        # Fields that may or may not be present — no crash is what matters
        assert isinstance(r.authors, list) or r.authors is None
        assert isinstance(r.abstract, str) or r.abstract is None
        assert isinstance(r.published_date, str) or r.published_date is None
        assert isinstance(r.metadata, dict)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_semantic_scholar_live_query(_env_loaded: None) -> None:
    """Query the real Semantic Scholar API and verify structural integrity."""
    s2_key = _skip_if_missing("S2_API_KEY", "Semantic Scholar")

    retriever = SemanticScholarRetriever(api_key=s2_key)
    results = await retriever.retrieve(
        "transformer attention mechanism", max_results=5
    )

    assert len(results) >= 1, (
        f"Expected at least 1 result from live S2, got {len(results)}"
    )

    for r in results:
        assert r.source == "semantic_scholar", (
            f"Expected source='semantic_scholar', got '{r.source}'"
        )
        assert r.title, "Every result must have a non-empty title"
        assert r.url is not None or r.metadata, (
            "Each result must have a URL or metadata"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tavily_live_query(_env_loaded: None) -> None:
    """Query the real Tavily API and verify structural integrity."""
    tavily_key = _skip_if_missing("TAVILY_API_KEY", "Tavily")

    retriever = TavilyRetriever(api_key=tavily_key)
    results = await retriever.retrieve(
        "transformer attention mechanism explained", max_results=5
    )

    assert len(results) >= 1, (
        f"Expected at least 1 result from live Tavily, got {len(results)}"
    )

    for r in results:
        assert r.source == "tavily", (
            f"Expected source='tavily', got '{r.source}'"
        )
        assert r.title, "Every result must have a non-empty title"
        assert r.url is not None, "Each Tavily result must have a URL"
