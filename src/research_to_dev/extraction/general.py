"""General content extraction — trafilatura clean + scraper fallback."""

from __future__ import annotations

import logging

import trafilatura

from research_to_dev.extraction.protocols import Scraper
from research_to_dev.extraction.types import GeneralContent
from research_to_dev.retriever.protocol import SearchResult


async def extract_general_content(
    result: SearchResult,
    *,
    scraper: Scraper,
    logger: logging.Logger | None = None,
) -> GeneralContent:
    """Extract cleaned body text from a general web SearchResult.

    Strategy (in order):
    1. Use ``metadata["raw_content"]`` if available (Tavily raw HTML).
    2. If absent, fetch the URL via ``scraper.fetch()``.
    3. Pass HTML through ``trafilatura.extract()`` for boilerplate removal.
    4. If all paths yield no text, ``body`` is set to ``""``.

    Args:
        result: A tavily-sourced SearchResult.
        scraper: Async scraper for fallback HTTP fetching.
        logger: Optional logger for diagnostics.

    Returns:
        GeneralContent with cleaned ``body`` (empty string if extraction fails).

    Raises:
        httpx.TimeoutException: When scraper fetch times out.
        httpx.HTTPStatusError: When scraper fetch gets 4xx/5xx.
    """
    log = logger or logging.getLogger(__name__)
    raw_html: str | None = result.metadata.get("raw_content") if result.metadata else None

    # Path 1: raw_content from Tavily
    if raw_html:
        body = _clean_html(raw_html)
        if body:
            return GeneralContent(
                source="tavily",
                title=result.title,
                url=result.url,
                body=body,
                metadata=result.metadata or {},
            )
        log.warning("trafilatura returned empty for %s — falling back to scrape", result.url)

    # Path 2: scrape + clean
    if result.url:
        scraped_html = await scraper.fetch(result.url)
        body = _clean_html(scraped_html) if scraped_html else ""
        return GeneralContent(
            source="tavily",
            title=result.title,
            url=result.url,
            body=body,
            metadata=result.metadata or {},
        )

    # Path 3: no URL and no raw_content — nothing we can do
    log.warning("No raw_content and no URL for result %r", result.title)
    return GeneralContent(
        source="tavily",
        title=result.title,
        url=result.url,
        body="",
        metadata=result.metadata or {},
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _clean_html(html: str) -> str:
    """Run trafilatura heuristics on raw HTML.

    Returns:
        Cleaned plain text, or ``""`` if trafilatura couldn't extract anything.
    """
    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    return text.strip() if text else ""
