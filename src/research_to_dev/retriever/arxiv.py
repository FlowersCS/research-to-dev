"""ArXiv retriever — queries the public arXiv Atom API."""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET

import httpx

from research_to_dev.retriever.protocol import SearchResult

_ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM_NS = "http://www.w3.org/2005/Atom"


class ArxivRetriever:
    """Async retriever for arXiv via the public Atom XML API.

    Enforces a minimum interval between requests to respect
    arXiv's rate-limit guidelines.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        logger: logging.Logger | None = None,
        timeout: float = 30.0,
        min_interval: float = 3.0,
    ) -> None:
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._logger = logger or logging.getLogger(__name__)
        self._timeout = timeout
        self._min_interval = min_interval
        self._last_request: float = 0.0

    async def retrieve(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Query arXiv and return normalized SearchResults.

        Returns empty list on any error — never raises into caller code.
        """
        await self._rate_limit_guard()
        try:
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": min(max_results, 10),
            }
            response = await self._client.get(_ARXIV_API, params=params)
            response.raise_for_status()
            return self._parse(response.text)
        except Exception as exc:
            self._logger.error("ArXiv retrieval failed for query=%r: %s", query, exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _rate_limit_guard(self) -> None:
        """Sleep if needed to enforce the minimum interval."""
        import time

        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request = time.monotonic()

    @staticmethod
    def _parse(xml_text: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        root = ET.fromstring(xml_text)

        for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
            title = _text(entry, "title") or "Untitled"
            summary = _text(entry, "summary")
            published = _text(entry, "published")
            authors = _authors(entry)
            url = _link_href(entry) or _text(entry, "id")
            pdf_url = _pdf_link(entry)

            results.append(
                SearchResult(
                    source="arxiv",
                    title=title.strip(),
                    url=url,
                    abstract=summary.strip() if summary else None,
                    authors=authors,
                    published_date=published,
                    metadata={"pdf_url": pdf_url} if pdf_url else {},
                )
            )

        return results


# ------------------------------------------------------------------
# XML helpers (module-level for testability)
# ------------------------------------------------------------------


def _text(element: ET.Element, tag: str) -> str | None:
    child = element.find(f"{{{_ATOM_NS}}}{tag}")
    return child.text if child is not None and child.text else None


def _authors(entry: ET.Element) -> list[str] | None:
    names: list[str] = []
    for author in entry.findall(f"{{{_ATOM_NS}}}author"):
        name_el = author.find(f"{{{_ATOM_NS}}}name")
        if name_el is not None and name_el.text:
            names.append(name_el.text.strip())
    return names if names else None


def _link_href(entry: ET.Element) -> str | None:
    for link in entry.findall(f"{{{_ATOM_NS}}}link"):
        rel = link.get("rel", "")
        if rel == "alternate":
            href = link.get("href")
            if href:
                return href
    return None


def _pdf_link(entry: ET.Element) -> str | None:
    for link in entry.findall(f"{{{_ATOM_NS}}}link"):
        if link.get("title") == "pdf":
            href = link.get("href")
            if href:
                return href
    return None
