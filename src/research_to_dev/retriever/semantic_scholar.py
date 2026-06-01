"""Semantic Scholar retriever — queries the S2 relevance search API."""

from __future__ import annotations

import logging

import httpx

from research_to_dev.retriever.protocol import SearchResult

_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = (
    "paperId,title,abstract,year,authors,openAccessPdf,"
    "publicationDate,citationCount"
)


class SemanticScholarRetriever:
    """Async retriever for Semantic Scholar via the relevance search API.

    API key is optional — if omitted, the ``x-api-key`` header is
    not sent, which works for low-volume usage.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
        logger: logging.Logger | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._api_key = api_key
        self._logger = logger or logging.getLogger(__name__)
        self._timeout = timeout

    async def retrieve(
        self, query: str, max_results: int = 10
    ) -> list[SearchResult]:
        """Query Semantic Scholar and return normalized SearchResults.

        Returns empty list on any error — never raises into caller code.
        """
        try:
            params: dict[str, str | int] = {
                "query": query,
                "limit": max_results,
                "fields": _S2_FIELDS,
            }
            headers: dict[str, str] = {}
            if self._api_key:
                headers["x-api-key"] = self._api_key

            response = await self._client.get(
                _S2_API, params=params, headers=headers
            )
            if response.status_code == 429:
                self._logger.warning(
                    "Semantic Scholar rate-limited (429) for query=%r", query
                )
                return []

            response.raise_for_status()
            return self._parse(response.json())
        except Exception as exc:
            self._logger.error(
                "Semantic Scholar retrieval failed for query=%r: %s",
                query,
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(data: dict) -> list[SearchResult]:
        results: list[SearchResult] = []
        for paper in data.get("data", []):
            authors: list[str] | None = None
            raw_authors = paper.get("authors")
            if raw_authors:
                authors = [
                    a["name"] for a in raw_authors if a.get("name")
                ] or None

            # Build URL from paperId
            paper_id = paper.get("paperId")
            url = (
                f"https://www.semanticscholar.org/paper/{paper_id}"
                if paper_id
                else None
            )

            # Metadata: extras that don't fit core fields
            metadata: dict = {}
            if paper.get("citationCount") is not None:
                metadata["citation_count"] = paper["citationCount"]
            oa = paper.get("openAccessPdf")
            if oa and oa.get("url"):
                metadata["open_access_pdf"] = oa["url"]

            results.append(
                SearchResult(
                    source="semantic_scholar",
                    title=paper.get("title") or "Untitled",
                    url=url,
                    abstract=paper.get("abstract"),
                    authors=authors,
                    published_date=paper.get("publicationDate"),
                    metadata=metadata,
                )
            )
        return results
