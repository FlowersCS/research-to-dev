"""Data contracts for the content extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AcademicContent:
    """Structured content from academic sources (arXiv, Semantic Scholar).

    ``abstract`` serves as the body text since these sources don't provide
    full article content.
    """

    source: Literal["arxiv", "semantic_scholar"]
    title: str
    url: str | None = None
    abstract: str | None = None
    authors: list[str] | None = None
    published_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneralContent:
    """Cleaned content from general web sources (Tavily).

    ``body`` contains trafilatura-cleaned full text.
    """

    source: Literal["tavily"]
    title: str
    url: str | None = None
    body: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionError:
    """Metadata about a failure processing a single SearchResult.

    Follows the same pattern as ``RetrievalError`` in the retriever layer.
    """

    source: str
    url: str | None = None
    message: str = ""
    exception_type: str | None = None


@dataclass
class ExtractionResult:
    """Aggregation of extraction outputs across all SearchResults.

    ``contents`` holds all successfully extracted content objects.
    ``errors`` holds one entry per failed extraction.
    """

    contents: list[AcademicContent | GeneralContent]
    errors: list[ExtractionError]
