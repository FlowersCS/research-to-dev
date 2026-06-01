"""Protocol and data contracts for the retriever system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class SearchResult:
    """A single search result from any retriever source.

    Required fields: source and title.
    All other fields may be None if the source doesn't provide them.
    """

    source: Literal["tavily", "arxiv", "semantic_scholar"]
    title: str
    url: str | None = None
    abstract: str | None = None
    authors: list[str] | None = None
    published_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalError:
    """Metadata about a failure from a specific source."""

    source: str
    message: str
    exception_type: str | None = None


@dataclass
class RetrievalResult:
    """Aggregated results from the orchestrator.

    ``results`` contains all successful items across all sources.
    ``errors`` contains one entry per failed source.
    """

    results: list[SearchResult]
    errors: list[RetrievalError]


@runtime_checkable
class Retriever(Protocol):
    """Structural protocol for async retriever adapters.

    Any class with ``async def retrieve(query, max_results)``
    is a valid Retriever — no inheritance required.
    """

    async def retrieve(self, query: str, max_results: int = 10) -> list[SearchResult]: ...  # noqa: D107
