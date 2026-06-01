"""Unit tests for RetrieverOrchestrator using AsyncMock adapters."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from research_to_dev.retriever.orchestrator import RetrieverOrchestrator
from research_to_dev.retriever.protocol import SearchResult


@pytest.fixture
def sample_results() -> list[SearchResult]:
    return [
        SearchResult(source="arxiv", title="Paper A"),
        SearchResult(source="arxiv", title="Paper B"),
    ]


@pytest.fixture
def s2_results() -> list[SearchResult]:
    return [SearchResult(source="semantic_scholar", title="S2 Paper")]


# -------------------------------------------------------------------
# All succeed
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_succeed(
    sample_results: list[SearchResult],
    s2_results: list[SearchResult],
) -> None:
    """When all adapters return results, they are merged into one list."""
    arxiv_mock = AsyncMock()
    arxiv_mock.retrieve.return_value = sample_results
    s2_mock = AsyncMock()
    s2_mock.retrieve.return_value = s2_results

    orchestrator = RetrieverOrchestrator([arxiv_mock, s2_mock])
    result = await orchestrator.search("test query", max_results=5)

    assert len(result.results) == 3
    assert result.errors == []

    arxiv_mock.retrieve.assert_awaited_once_with("test query", 5)
    s2_mock.retrieve.assert_awaited_once_with("test query", 5)


# -------------------------------------------------------------------
# One fails
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_fails(
    sample_results: list[SearchResult],
) -> None:
    """When one adapter raises, the other results are preserved."""
    good = AsyncMock()
    good.retrieve.return_value = sample_results
    bad = AsyncMock()
    bad.retrieve.side_effect = RuntimeError("connection refused")

    orchestrator = RetrieverOrchestrator([good, bad])
    result = await orchestrator.search("test")

    assert len(result.results) == 2
    assert len(result.errors) == 1
    assert result.errors[0].source == "AsyncMock"
    assert "connection refused" in result.errors[0].message
    assert result.errors[0].exception_type == "RuntimeError"


# -------------------------------------------------------------------
# All fail
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_fail() -> None:
    """When every adapter raises, results are empty, errors has 3 entries."""
    a = AsyncMock()
    a.retrieve.side_effect = ValueError("bad input")
    b = AsyncMock()
    b.retrieve.side_effect = TimeoutError("timeout")
    c = AsyncMock()
    c.retrieve.side_effect = ConnectionError("refused")

    orchestrator = RetrieverOrchestrator([a, b, c])
    result = await orchestrator.search("test")

    assert result.results == []
    assert len(result.errors) == 3
    types = {e.exception_type for e in result.errors}
    assert types == {"ValueError", "TimeoutError", "ConnectionError"}


# -------------------------------------------------------------------
# Verbose logging
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verbose_logging(
    sample_results: list[SearchResult],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When verbose=True, per-adapter timing is logged."""
    caplog.set_level("INFO")

    mock = AsyncMock()
    mock.retrieve.return_value = sample_results

    orchestrator = RetrieverOrchestrator([mock], verbose=True)
    await orchestrator.search("test", max_results=3)

    # Should contain a timing log line
    timing_logs = [
        r.message
        for r in caplog.records
        if "results in" in r.message
    ]
    assert len(timing_logs) == 1
    assert "AsyncMock" in timing_logs[0]
    assert "2 results" in timing_logs[0]
